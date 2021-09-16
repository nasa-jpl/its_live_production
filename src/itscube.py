"""
ITSCube class creates ITS_LIVE datacube based on target projection,
bounding polygon and datetime period provided by the caller.

Authors: Masha Liukis
"""
import copy
from datetime import datetime, timedelta
import gc
import glob
import json
import logging
import os
import psutil
import pyproj
import shutil
import timeit
import zarr

import dask
# from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
import numpy  as np
import pandas as pd
import s3fs
from tqdm import tqdm
import xarray as xr

# Local modules
import itslive_utils
from grid import Bounds, Grid
from itscube_types import Coords, DataVars
import zarr_to_netcdf

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

# Coordinates attributes for the output store
MID_DATE_ATTRS = {
    DataVars.STD_NAME: Coords.STD_NAME[Coords.MID_DATE],
    DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.MID_DATE]
}
X_ATTRS = {
    DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
    DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
}
Y_ATTRS = {
    DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
    DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
}


class ITSCube:
    """
    Class to build ITS_LIVE cube: time series of velocity pairs within a
    polygon of interest for specified time period.
    """
    # Current ITSCube software version
    Version = '1.0'

    # Number of threads for parallel processing
    NUM_THREADS = 4

    # Dask scheduler for parallel processing
    DASK_SCHEDULER = "processes"

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = 'EPSG:4326'

    S3_PREFIX = 's3://'
    HTTP_PREFIX = 'http://'

    # Token within granule's URL that needs to be removed to get file location within S3 bucket:
    # if URL is of the 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0/32628/file.nc' format,
    # S3 bucket location of the file is 's3://its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0/32628/file.nc'
    PATH_URL = ".s3.amazonaws.com"

    # Engine to read xarray data into from NetCDF filecompression
    NC_ENGINE = 'h5netcdf'

    # Date format as it appears in granules filenames:
    # (LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc)
    DATE_FORMAT = "%Y%m%d"

    # Date and time format for acquisition dates of img_info_pair
    DATE_TIME_FORMAT = '%Y%m%dT%H:%M:%S'

    # Granules are written to the file in chunks to avoid out of memory issues.
    # Number of granules to write to the file at a time.
    NUM_GRANULES_TO_WRITE = 1000

    # Grid cell size for the datacube.
    CELL_SIZE = 240.0

    CHIP_SIZE_HEIGHT_NO_VALUE = 65535

    def __init__(self, polygon: tuple, projection: str):
        """
        Initialize object.

        polygon: tuple
            Polygon for the datacube tile.
        projection: str
            Projection in which polygon/datacube is defined.
        """
        self.logger = logging.getLogger("datacube")
        self.logger.info(f"Polygon: {polygon}")
        self.logger.info(f"Projection: {projection}")

        self.projection = projection

        # Set min/max x/y values to filter region by
        self.x = Bounds([each[0] for each in polygon])
        self.y = Bounds([each[1] for each in polygon])

        # Grid for the datacube based on its bounding polygon
        self.grid_x, self.grid_y = Grid.create(self.x, self.y, ITSCube.CELL_SIZE)

        # Ensure lonlat output order
        to_lon_lat_transformer = pyproj.Transformer.from_crs(
            f"EPSG:{projection}",
            ITSCube.LON_LAT_PROJECTION,
            always_xy=True)

        center_x = (self.grid_x.min() + self.grid_x.max())/2
        center_y = (self.grid_y.min() + self.grid_y.max())/2

        # Convert to lon/lat coordinates
        self.center_lon_lat = to_lon_lat_transformer.transform(center_x, center_y)

        # Convert polygon from its target projection to longitude/latitude coordinates
        # which are used by granule search API
        self.polygon_coords = []

        for each in polygon:
            coords = to_lon_lat_transformer.transform(each[0], each[1])
            self.polygon_coords.extend(coords)

        self.logger.info(f"Polygon's longitude/latitude coordinates: {self.polygon_coords}")

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle dates (+ date separation in days as milliseconds),
        # original granules URLs)
        self.ds = []

        self.dates = []
        self.urls = []
        self.num_urls_from_api = None

        # Keep track of skipped granules due to the other than target projection
        self.skipped_proj_granules = {}
        # Keep track of skipped granules due to no data for the polygon of interest
        self.skipped_empty_granules = []
        # Keep track of "double" granules with older processing date which are
        # not included into the cube
        self.skipped_double_granules = []

        # Constructed cube
        self.layers = None

        # Dates when datacube was created or updated
        self.date_created = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.date_updated = None

    def clear_vars(self):
        """
        Clear current set of cube layers.
        """
        self.ds = None
        self.layers = None
        self.dates = []
        self.urls = []

        # Call Python's garbage collector
        gc.collect()

        self.ds = []

    def clear(self):
        """
        Reset all internal data structures.
        """
        self.clear_vars()

        self.num_urls_from_api = None
        self.skipped_proj_granules = {}
        self.skipped_empty_granules = []
        self.skipped_double_granules = []

    def request_granules(self, api_params: dict, num_granules: int):
        """
        Send request to ITS_LIVE API to get a list of granules to satisfy polygon request.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for each in self.polygon_coords])

        self.logger.info(f"ITS_LIVE search API params: {params}")
        start_time = timeit.default_timer()
        found_urls = [each['url'] for each in itslive_utils.get_granule_urls(params)]
        total_num = len(found_urls)
        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Number of found by API granules: {total_num} (took {time_delta} seconds)")

        if len(found_urls) == 0:
            self.logger.info(f"No granules are found for the search API parameters: {params}, " \
                              "skipping datacube generation or update")
            return found_urls

        # Number of granules to examine is specified
        # ATTN: just a way to limit number of granules to be considered for the
        #       datacube generation (testing or debugging only).
        if num_granules:
            found_urls = found_urls[:num_granules]
            self.logger.info(f"Examining only first {len(found_urls)} out of {total_num} found granules")

        urls, self.skipped_double_granules = self.skip_duplicate_granules(found_urls)

        return urls

    def skip_duplicate_granules(self, found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s))
        for the same path row granule.
        """
        self.num_urls_from_api = len(found_urls)

        # Need to remove duplicate granules for the middle date: some granules
        # have newer processing date, keep those.
        keep_urls = {}
        skipped_double_granules = []

        for each_url in tqdm(found_urls, ascii=True, desc='Skipping duplicate granules...'):
            # Extract acquisition and processing dates
            url_acq_1, url_proc_1, path_row_1, url_acq_2, url_proc_2, path_row_2 = \
                ITSCube.get_tokens_from_filename(each_url)

            # Acquisition time and path/row of both images should be identical
            granule_id = '_'.join([
                url_acq_1.strftime(ITSCube.DATE_FORMAT),
                path_row_1,
                url_acq_2.strftime(ITSCube.DATE_FORMAT),
                path_row_2
            ])

            # There is a granule for the mid_date already, check which processing
            # time is newer, keep the one with newer processing date
            if granule_id in keep_urls:
                # Flag if newly found URL should be kept
                keep_found_url = False

                for found_url in keep_urls[granule_id]:
                    # Check already found URLs for processing time
                    _, found_proc_1, _, _, found_proc_2, _ = \
                        ITSCube.get_tokens_from_filename(found_url)

                    # If both granules have identical processing time,
                    # keep them both - granules might be in different projections,
                    # any other than target projection will be handled later
                    if url_proc_1 == found_proc_1 and \
                       url_proc_2 == found_proc_2:
                        keep_urls[granule_id].append(each_url)
                        keep_found_url = True
                        break

                # There are no "identical" granules for "each_url", check if
                # new granule has newer processing dates
                if not keep_found_url:
                    # Check if any of the found URLs have older processing time
                    # than newly found URL
                    remove_urls = []
                    for found_url in keep_urls[granule_id]:
                        # Check already found URL for processing time
                        _, found_proc_1, _, _, found_proc_2, _ = \
                            ITSCube.get_tokens_from_filename(found_url)

                        if url_proc_1 >= found_proc_1 and \
                           url_proc_2 >= found_proc_2:
                            # The granule will need to be replaced with a newer
                            # processed one
                            remove_urls.append(found_url)

                        elif url_proc_1 > found_proc_1:
                            # There are few cases when proc_1 is newer in
                            # each_url and proc_2 is newer in found_url, then
                            # keep the granule with newer proc_1
                            remove_urls.append(found_url)

                    if len(remove_urls):
                        # Some of the URLs need to be removed due to newer
                        # processed granule
                        self.logger.info(f"Skipping {remove_urls} in favor of new {each_url}")
                        skipped_double_granules.extend(remove_urls)

                        # Remove older processed granules
                        keep_urls[granule_id][:] = [each for each in keep_urls[granule_id] if each not in remove_urls]
                        # Add new granule with newer processing date
                        keep_urls[granule_id].append(each_url)

                    else:
                        # New granule has older processing date, don't include
                        self.logger.info(f"Skipping new {each_url} in favor of {keep_urls[granule_id]}")
                        skipped_double_granules.append(each_url)

            else:
                # This is a granule for new ID, append it to URLs to keep
                keep_urls.setdefault(granule_id, []).append(each_url)

        granules = []
        for each in keep_urls.values():
            granules.extend(each)

        self.logger.info(f"Keeping {len(granules)} unique granules")

        return granules, skipped_double_granules

    def exclude_processed_granules(self, found_urls: list, cube_ds: xr.Dataset):
        """
        * Exclude datacube granules, and all skipped granules in existing datacube
        (empty data, wrong projection, duplicate middle date) from found granules.
        * Identify if any of the skipped double mid_date granules from "found_urls"
        are already existing layers in the datacube. Need to mark such layers
        to be deleted from the datacube.
        * Identify if current cube layers and remaining found_urls have duplicate
        mid_date - register these for deletion from the datacube if they appear
        as datacube layers.

        Return:
            granules: list
                List of granules to update datacube with.
            layers_to_delete: list
                List of existing datacube layers to remove.
        """
        self.logger.info("Excluding known to datacube granules...")
        cube_granules = cube_ds[DataVars.URL].values.tolist()
        granules = set(found_urls).difference(cube_granules)
        self.logger.info(f"Removed existing cube granules ({len(cube_granules)}): {len(granules)} granules left ")

        # Remove known empty granules (per cube) from found_urls
        self.skipped_empty_granules = json.loads(cube_ds.attrs[DataVars.SKIP_EMPTY_DATA])
        granules = granules.difference(self.skipped_empty_granules)
        self.logger.info(f"Removed known empty data granules ({len(self.skipped_empty_granules)}): {len(granules)} granules left ")

        # Remove known wrong projection granules (per cube) from found_urls
        self.skipped_proj_granules = json.loads(cube_ds.attrs[DataVars.SKIP_WRONG_PROJECTION])
        known_granules = []
        for each in self.skipped_proj_granules:
            known_granules.extend(self.skipped_proj_granules[each])

        granules = granules.difference(known_granules)
        self.logger.info(f"Removed wrong projection granules ({len(known_granules)}): {len(granules)} granules left ")

        # Identify cube granules that are now skipped due to double middle date
        # in new found_urls granules
        cube_layers_to_delete = list(set(self.skipped_double_granules).intersection(cube_granules))
        self.logger.info(f"After found_urls::skipped_granules: {len(cube_layers_to_delete)} " \
                          f"existing datacube layers to delete due to duplicate mid_date: {cube_layers_to_delete}")

        # Remove known duplicate middle date granules from found_urls:
        # if cube's skipped granules don't appear in found_urls:skipped_granules
        # for whatever reason (different start/end dates are used for cube update)
        # self.skipped_double_granules is populated by self.request_granules()
        # with skipped granules due to double date in "found_urls"
        cube_skipped_double_granules = json.loads(cube_ds.attrs[DataVars.SKIP_DUPLICATE_MID_DATE])
        granules = granules.difference(cube_skipped_double_granules)
        self.logger.info(f"Removed cube's duplicate middle date granules ({len(cube_skipped_double_granules)}): {len(granules)} granules left ")

        # Check if there are any granules between existing cube layers and found_urls
        # that have duplicate middle date
        combined_cube_found_urls = cube_granules + list(granules)
        _, skipped_granules = self.skip_duplicate_granules(combined_cube_found_urls)

        # Check if any of the skipped granules are in the cube
        cube_layers_to_delete.extend(list(set(cube_granules).intersection(skipped_granules)))
        self.logger.info(f"After (cube+found_urls)::skipped_granules: {len(cube_layers_to_delete)} " \
                         f"existing datacube layers to delete due to duplicate middle_date: {cube_layers_to_delete}")

        # Merge two lists of skipped granules (for existing cube, new list
        # of granules from search API, and duplicate granules b/w cube and new granules)
        cube_skipped_double_granules.extend(self.skipped_double_granules)
        cube_skipped_double_granules.extend(skipped_granules)
        self.skipped_double_granules = list(set(cube_skipped_double_granules))

        # Skim down found_urls by newly skipped granules
        granules = list(granules.difference(self.skipped_double_granules))
        self.logger.info(f"Leaving {len(granules)} granules...")

        return granules, cube_layers_to_delete

    @staticmethod
    def get_tokens_from_filename(filename):
        """
        Extract acquisition/processing dates and path/row for two images from the filename.
        """
        # Get acquisition and processing date for both images from url and index_url
        url_tokens = os.path.basename(filename).split('_')
        url_acq_date_1 = datetime.strptime(url_tokens[3], ITSCube.DATE_FORMAT)
        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)
        url_path_row_1 = url_tokens[2]
        url_acq_date_2 = datetime.strptime(url_tokens[11], ITSCube.DATE_FORMAT)
        url_proc_date_2 = datetime.strptime(url_tokens[12], ITSCube.DATE_FORMAT)
        url_path_row_2 = url_tokens[10]

        return url_acq_date_1, url_proc_date_1, url_path_row_1, url_acq_date_2, url_proc_date_2, url_path_row_2

    def add_layer(self, is_empty, layer_projection, mid_date, url, data):
        """
        Examine the layer if it qualifies to be added as a cube layer.
        """

        if data is not None:
            # "Duplicate" granules are handled apriori for newly constructed
            #  cubes (see self.request_granules() method) and for updated cubes
            #  (see self.exclude_processed_granules() method).
            # print(f"Adding {url} for {mid_date}")
            self.dates.append(mid_date)
            self.ds.append(data)
            self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_empty_granules.append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_proj_granules.setdefault(layer_projection, []).append(url)

    @staticmethod
    def init_output_store(output_dir: str):
        """
        Initialize output store for the datacube. It removes existing local
        store if it exists already. This method is useful only if create_* methods
        are called directly by the user - to guarantee that datacube is created
        from the scratch.
        """
        # Remove datacube store if it exists
        if os.path.exists(output_dir):
            logging.info(f"Removing existing {output_dir}")
            shutil.rmtree(output_dir)

        return

    @staticmethod
    def exists(output_dir: str, s3_bucket: str):
        """
        Check if datacube exists. The datacube can be on a local file system or
        in AWS S3 bucket.
        """
        cube_exists = False

        # Check if the datacube is in the S3 bucket
        if len(s3_bucket):
            cube_path = os.path.join(s3_bucket, output_dir)
            s3 = s3fs.S3FileSystem(anon=True)
            cube_glob = s3.glob(cube_path)
            if len(cube_glob):
                cube_exists = True

        else:
            if os.path.exists(output_dir):
                cube_exists = True

        return cube_exists

    @staticmethod
    def init_input_store(input_dir: str, s3_bucket: str):
        """
        Read datacube from provided store. The method detects if S3 bucket
        store or local Zarr archive is provided, and reads xarray.Dataset from
        the Zarr store.
        """
        ds_from_zarr = None
        s3_in = None
        cube_store = None

        if len(s3_bucket) == 0:
            # If reading from the local directory, check if datacube store exists
            if ITSCube.exists(input_dir, s3_bucket):
                logging.info(f"Reading existing {input_dir}")
                # Read dataset in
                ds_from_zarr = xr.open_zarr(input_dir, decode_timedelta=False, consolidated=True)

        elif ITSCube.exists(input_dir, s3_bucket):
            # When datacube is in the AWS S3 bucket, check if it exists.
            cube_path = os.path.join(s3_bucket, input_dir)
            logging.info(f"Reading existing {cube_path}")

            # Open S3FS access to S3 bucket with input datacube
            s3_in = s3fs.S3FileSystem(anon=True)
            cube_store = s3fs.S3Map(root=cube_path, s3=s3_in, check=False)
            ds_from_zarr = xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True)

        if ds_from_zarr is None:
            raise RuntimeError(f"Provided input datacube {input_dir} does not exist (s3={s3_bucket})")

        # Don't use cube_store - keep it in scope only to guarantee valid
        # file-like access.
        return s3_in, cube_store, ds_from_zarr

    def create_or_update(self, api_params: dict, output_dir: str, output_bucket: str, num_granules=None):
        """
        Create new or update existing datacube.
        """
        self.logger.info(f"ITS_LIVE search API parameters: {api_params}")

        if ITSCube.exists(output_dir, output_bucket):
            # Datacube exists, update
            self.update_parallel(api_params, output_dir, output_bucket, num_granules)

        else:
            # Create new datacube
            self.create_parallel(api_params, output_dir, output_bucket, num_granules)

    def update_parallel(self, api_params: dict, output_dir: str, output_bucket: str, num_granules=None):
        """
        Update velocity pair datacube by reading and pre-processing new cube layers in parallel.

        api_params: dict
            Search API required parameters.
        output_dir: str
            Local datacube Zarr store to write updated datacube to.
        output_bucket: str
            AWS S3 bucket if datacube Zarr store resides in the cloud.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        self.logger.info(f"Updating {os.path.join(output_bucket, output_dir)}")

        ITSCube.show_memory_usage('update()')
        s3, cube_store_in, cube_ds = ITSCube.init_input_store(output_dir, output_bucket)

        self.date_updated = self.date_created
        self.date_created = cube_ds.attrs['date_created']

        if s3 is None:
            # If input datacube is on the local filesystem, open S3FS for reading
            # granules from S3 bucket
            s3 = s3fs.S3FileSystem(anon=True)

        self.clear()

        found_urls = self.request_granules(api_params, num_granules)
        if len(found_urls) == 0:
            return found_urls

        # Check if any of the existing cube layers are excluded, mark them to be
        # deleted
        layers_to_delete = []


        # Remove already processed granules
        found_urls, cube_layers_to_delete = self.exclude_processed_granules(found_urls, cube_ds)
        num_cube_layers = len(cube_ds.mid_date.values)

        if len(found_urls) == 0:
            self.logger.info("No granules to update with, exiting.")
            return found_urls

        cube_store_in = None
        cube_ds = None
        gc.collect()

        # If datacube resides in AWS S3 bucket, copy it locally - initial datacube to begin with
        if not os.path.exists(output_dir):
            # Copy datacube locally using AWS CLI to take advantage of parallel copy:
            # have to include "max_concurrent_requests" option for the
            # configuration in ~/.aws/config
            # [default]
            # region = us-west-2
            # output = json
            # s3 =
            #    max_concurrent_requests = 100
            #
            env_copy = os.environ.copy()
            source_url = os.path.join(output_bucket, output_dir)
            if not source_url.startswith(ITSCube.S3_PREFIX):
                source_url = ITSCube.S3_PREFIX + source_url

            command_line = [
                "aws", "s3", "cp", "--recursive",
                source_url,
                output_dir
            ]

            self.logger.info(f"Creating local copy of {source_url}: {output_dir}")
            self.logger.info(' '.join(command_line))

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to copy {source_url} to {output_store}: {command_return.stdout}")

        elif len(output_bucket):
            # datacube exists on local file system even though S3 bucket for the
            # datacube is provided.
            raise RuntimeError(f'Local copy of {output_dir} already exists though {output_bucket} is provided, remove datacube first')

        # Delete identified layers of the cube if any
        is_first_write = False

        if len(cube_layers_to_delete):
            self.logger.info(f"Deleting {len(cube_layers_to_delete)} layers from total {num_cube_layers} layers of {output_dir}")

            if len(cube_layers_to_delete) == num_cube_layers:
                # If all layers need to be deleted, just delete the cube and start from
                # the scratch
                is_first_write = True
                self.logger.info(f"Deleting existing {output_dir}")
                shutil.rmtree(output_dir)

            else:
                # Delete identified layers
                ds_from_zarr = xr.open_zarr(output_dir, decode_timedelta=False, consolidated=True)

                # Identify layer indices that correspond to granule urls
                layers_bool_flag = ds_from_zarr[DataVars.URL].isin(cube_layers_to_delete)

                # Drop the layers
                # layers_mid_dates = ds_from_zarr[DataVars.MID_DATE].values[layers_bool_flag.values]
                dropped_ds = ds_from_zarr.drop_isel(mid_date=layers_bool_flag.values)

                tmp_output_dir = f"{output_dir}.original"
                self.logger.info(f"Moving original {output_dir} to {tmp_output_dir}")
                os.renames(output_dir, tmp_output_dir)

                # Write updated datacube to original store location,
                # but at first re-chunk xr.Dataset to avoid errors
                dropped_ds = dropped_ds.chunk({Coords.MID_DATE: ITSCube.NUM_GRANULES_TO_WRITE})

                self.logger.info(f"Saving updated {output_dir}")
                dropped_ds.to_zarr(output_dir, encoding=zarr_to_netcdf.ENCODING_ZARR, consolidated=True)

                self.logger.info(f"Removing original {tmp_output_dir}")
                shutil.rmtree(tmp_output_dir)

                ds_from_zarr = None
                dropped_ds   = None
                gc.collect()

        start = 0
        num_to_process = len(found_urls)

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls[start:start+num_tasks]]
            self.logger.info(f"Processing {len(tasks)} tasks out of {num_to_process} remaining")

            results = None
            with ProgressBar():  # Does not work with Client() scheduler
                # If to collect performance report (need to define global Client - see above)
                # with performance_report(filename=f"dask-report-{num_granules}.html"):
                #     results = dask.compute(tasks)
                results = dask.compute(
                    tasks,
                    scheduler=ITSCube.DASK_SCHEDULER,
                    num_workers=ITSCube.NUM_THREADS
                )

            del tasks
            gc.collect()

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            del results
            gc.collect()

            wrote_layers = self.combine_layers(output_dir, is_first_write)
            if is_first_write and wrote_layers:
                is_first_write = False

            self.format_stats()

            num_to_process -= num_tasks
            start += num_tasks

        # Remove existing granules with older processing dates if any

        return found_urls

    def create_parallel(self, api_params: dict, output_dir: str, output_bucket: str, num_granules=None):
        """
        Create velocity pair datacube by reading and pre-processing cube layers in parallel.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        self.logger.info(f"Creating {os.path.join(output_bucket, output_dir)}")

        ITSCube.show_memory_usage('create()')
        ITSCube.init_output_store(output_dir)

        self.clear()
        found_urls = self.request_granules(api_params, num_granules)
        if len(found_urls) == 0:
            return found_urls

        # Parallelize layer collection
        s3 = s3fs.S3FileSystem(anon=True)

        # In order to enable Dask profiling, need to create Dask client for
        # processing: using "processes" or "threads" scheduler
        # processes_scheduler = True if ITSCube.DASK_SCHEDULER == 'processes' else False
        # client = Client(processes=processes_scheduler, n_workers=ITSCube.NUM_THREADS)
        # # Use client to collect profile information
        # client.profile(filename=f"dask-profile-{num_granules}-parallel.html")
        is_first_write = True
        start = 0
        num_to_process = len(found_urls)

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls[start:start+num_tasks]]
            self.logger.info(f"Processing {len(tasks)} tasks out of {num_to_process} remaining")

            results = None
            with ProgressBar():  # Does not work with Client() scheduler
                # If to collect performance report (need to define global Client - see above)
                # with performance_report(filename=f"dask-report-{num_granules}.html"):
                #     results = dask.compute(tasks)
                results = dask.compute(
                    tasks,
                    scheduler=ITSCube.DASK_SCHEDULER,
                    num_workers=ITSCube.NUM_THREADS
                )

            del tasks
            gc.collect()

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            del results
            gc.collect()

            wrote_layers = self.combine_layers(output_dir, is_first_write)
            if is_first_write and wrote_layers:
                is_first_write = False

            self.format_stats()

            num_to_process -= num_tasks
            start += num_tasks

        return found_urls

    def create_sequential(self, api_params: dict, output_dir: str, num_granules=None):
        """
        Create velocity pair cube.
        This is non-parallel implementation and used for debugging only.

        api_params: dict
            Search API required parameters.
        output_dir: str
            Zarr store to write datacube to.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        ITSCube.show_memory_usage('create()')
        ITSCube.init_output_store(output_dir)

        self.clear()

        found_urls = self.request_granules(api_params, num_granules)
        if len(found_urls) == 0:
            return found_urls

        # Open S3FS access to public S3 bucket with input granules
        s3 = s3fs.S3FileSystem(anon=True)

        is_first_write = True
        for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):

            s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
            s3_path = s3_path.replace(ITSCube.PATH_URL, '')

            self.logger.info(f"Reading {s3_path}...")
            ITSCube.show_memory_usage(f'before reading {s3_path}')
            # Attempt to fix locked up s3fs==0.5.1 on Linux (AWS Batch processing)
            # s3 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)

            with s3.open(s3_path, mode='rb') as fhandle:
                with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                    self.logger.info(f"Preprocess dataset from {s3_path}...")
                    results = self.preprocess_dataset(ds, each_url)
                    ITSCube.show_memory_usage('after reading {s3_path}')

                    self.logger.info(f"Add layer for {s3_path}...")
                    self.add_layer(*results)

            ITSCube.show_memory_usage(f'after adding layer for {s3_path}')

            # Check if need to write to the file accumulated number of granules
            if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                wrote_layers = self.combine_layers(output_dir, is_first_write)
                if is_first_write and wrote_layers:
                    is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        # Report statistics for skipped granules
        self.format_stats()

        return found_urls

    def create_from_local_no_api(self, output_dir: str, dirpath='data', num_granules=None):
        """
        Create velocity cube by accessing local data stored in "dirpath" directory.
        This is non-parallel implementation and used for debugging only.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        ITSCube.init_output_store(output_dir)

        self.clear()

        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        if len(found_urls) == 0:
            self.logger.info(f"No granules found in {dirpath}, skipping datacube generation")
            return found_urls

        if num_granules is not None:
            found_urls = found_urls[0: num_granules]

        self.num_urls_from_api = len(found_urls)
        found_urls = self.skip_duplicate_granules(found_urls)
        is_first_write = True

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in tqdm(found_urls, ascii=True, desc='Processing local granules'):
            with xr.open_dataset(each_url) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)

                # Check if need to write to the file accumulated number of granules
                if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                    wrote_layers = self.combine_layers(output_dir, is_first_write)
                    if is_first_write and wrote_layers:
                        is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        self.format_stats()

        return found_urls

    def create_from_local_parallel_no_api(self, output_dir: str, dirpath='data', num_granules=None):
        """
        Create velocity cube from local data stored in "dirpath" in parallel.
        This is used for debugging only.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        ITSCube.init_output_store(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        if len(found_urls) == 0:
            self.logger.info(f"No granules found in {dirpath}, skipping datacube generation")
            return found_urls

        if num_granules is not None:
            found_urls = found_urls[0: num_granules]

        found_urls = self.skip_duplicate_granules(found_urls)
        self.num_urls_from_api = len(found_urls)

        num_to_process = len(found_urls)

        is_first_write = True
        start = 0
        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            self.logger.info(f"Number of granules to process: {num_tasks}")

            tasks = [dask.delayed(self.read_dataset)(each_file) for each_file in found_urls[start:start+num_tasks]]
            assert len(tasks) == num_tasks
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler=ITSCube.DASK_SCHEDULER,
                                       num_workers=ITSCube.NUM_THREADS)

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            wrote_layers = self.combine_layers(output_dir, is_first_write)
            if is_first_write and wrote_layers:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()

        return found_urls

    def get_data_var(self, ds: xr.Dataset, var_name: str):
        """
        Return xr.DataArray that corresponds to the data variable if it exists
        in the 'ds' dataset, or empty xr.DataArray if it is not present in the 'ds'.
        Empty xr.DataArray assumes the same dimensions as ds.v data array.
        """

        if var_name in ds:
            return ds[var_name]

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: Can't use None as data to create xr.DataArray - won't be able
        # to set dtype='short' in encoding for writing to the file.
        data_values = np.empty((len(self.grid_y), len(self.grid_x)))
        data_values[:, :] = np.nan

        return xr.DataArray(
            data=data_values,
            coords=[self.grid_y, self.grid_x],
            dims=[Coords.Y, Coords.X]
        )

    @staticmethod
    def get_data_var_attr(
        ds: xr.Dataset,
        ds_url: str,
        var_name: str,
        attr_name: str,
        missing_value: int = None,
        to_date=False):
        """
        Return a list of attributes for the data variable in data set if it exists,
        or missing_value if it is not present.
        If missing_value is set to None, than specified attribute is expected
        to exist for the data variable "var_name" and exception is raised if
        it does not.
        """
        if var_name in ds and attr_name in ds[var_name].attrs:
            value = ds[var_name].attrs[attr_name]
            # print(f"Read value for {var_name}.{attr_name}: {value}")

            # Check if type has "length"
            if hasattr(type(value), '__len__') and len(value) == 1:
                value = value[0]

            if to_date is True:
                try:
                    tokens = value.split('T')
                    if len(tokens) == 3:
                        # Handle malformed datetime in Sentinel 2 granules:
                        # img_pair_info.acquisition_date_img1 = "20190215T205541T00:00:00"
                        value = tokens[0] + 'T' + tokens[1][0:2] + ':' + tokens[1][2:4]+ ':' + tokens[1][4:6]
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                    elif len(value) == 8:
                        # Only date is provided
                        value = datetime.strptime(value[0:8], '%Y%m%d')

                    elif len(value) > 8:
                        # Extract date and time (20200617T00:00:00)
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                except ValueError as exc:
                    raise RuntimeError(f"Error converting {value} to date format '%Y%m%d': {exc} for {var_name}.{attr_name} in {ds_url}")

            # print(f"Return value for {var_name}.{attr_name}: {value}")
            return value

        if missing_value is None:
            # If missing_value is not provided, attribute is expected to exist always
            raise RuntimeError(f"{attr_name} is expected within {var_name} for {ds_url}")

        return missing_value

    def preprocess_dataset(self, ds: xr.Dataset, ds_url: str):
        """
        Pre-process ITS_LIVE dataset in preparation for the cube layer.

        ds: xarray dataset
            Dataset to pre-process.
        ds_url: str
            URL that corresponds to the dataset.

        Returns:
        cube_v:     Filtered data array for the layer.
        mid_date:   Middle date that corresponds to the velicity pair (uses date
                    separation as milliseconds)
        empty:      Flag to indicate if dataset does not contain any data for
                    the cube region.
        projection: Source projection for the dataset.
        url:        Original URL for the granule (have to return for parallel
                    processing: no track of inputs for each task, but have output
                    available for each task).
        """
        # Tried to load the whole dataset into memory to avoid penalty for random read access
        # when accessing S3 bucket (?) - does not make any difference.
        # ds.load()

        # Flag if layer data is empty
        empty = False

        # Layer data
        mask_data = None

        # Layer middle date
        mid_date = None

        # Detect projection
        ds_projection = None
        # if DataVars.UTM_PROJECTION in ds:
        #     ds_projection = ds.UTM_Projection.spatial_epsg
        #
        # elif DataVars.POLAR_STEREOGRAPHIC in ds:
        #     ds_projection = ds.Polar_Stereographic.spatial_epsg

        if DataVars.MAPPING in ds:
            ds_projection = ds.mapping.spatial_epsg

        else:
            # Unknown type of granule is provided
            raise RuntimeError(f"Unsupported projection is detected for {ds_url}. One of [{DataVars.UTM_PROJECTION}, {DataVars.POLAR_STEREOGRAPHIC}, {DataVars.MAPPING}] is supported.")

        # Consider granules with data only within target projection
        if str(int(ds_projection)) == self.projection:
            acq1_datetime = datetime.strptime(ds.img_pair_info.attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1], ITSCube.DATE_TIME_FORMAT)
            mid_date = acq1_datetime + \
                (datetime.strptime(ds.img_pair_info.attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2], ITSCube.DATE_TIME_FORMAT) - acq1_datetime)/2

            # Create unique "token" by using granule's centroid longitude/latitude to
            # increase uniqueness of the mid_date for the layer (xarray: can't drop layers
            # for the cube with mid_date dimension which contains non-unique values).
            # Add the token as microseconds for the middle date: AAOOO
            #
            lat = int(np.abs(ds.img_pair_info.latitude))
            lon = int(np.abs(ds.img_pair_info.longitude))
            mid_date += timedelta(microseconds=int(f'{lat:02d}{lon:03d}'))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.x.min) & (ds.x <= self.x.max)
            mask_lat = (ds.y >= self.y.min) & (ds.y <= self.y.max)
            mask = (mask_lon & mask_lat)
            if mask.values.sum() == 0:
                # One or both masks resulted in no coverage
                mask_data = None
                mid_date = None
                empty = True

            else:
                mask_data = ds.where(mask_lon & mask_lat, drop=True)

                # Another way to filter (have to put min/max values in the order
                # corresponding to the grid)
                # cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

                # If it's a valid velocity layer, add it to the cube.
                if np.any(mask_data.v.notnull()):
                    mask_data.load()

                    # Verify that granule is defined on the same grid cell size as
                    # expected output datacube.
                    cell_x_size = np.abs(mask_data.x.values[0] - mask_data.x.values[1])
                    if cell_x_size != ITSCube.CELL_SIZE:
                        raise RuntimeError(f"Unexpected grid cell size ({cell_x_size}) is detected for {ds_url} vs. expected {ITSCube.CELL_SIZE}")

                else:
                    # Reset cube back to None as it does not contain any valid data
                    mask_data = None
                    mid_date = None
                    empty = True

        # Have to return URL for the dataset, which is provided as an input to the method,
        # to track URL per granule in parallel processing
        return empty, int(ds_projection), mid_date, ds_url, mask_data

    def process_v_attributes(self, var_name: str, mid_date_coord):
        """
        Helper method to clean up attributes for v-related data variables.
        """
        _stable_rmse_vars = [DataVars.VX, DataVars.VY]

        # Dictionary of attributes values for new v*_error data variables:
        # std_name, description
        _attrs = {
            'vx_error': ("x_velocity_error", "error for velocity component in x direction"),
            'vy_error': ("y_velocity_error", "error for velocity component in y direction"),
            'va_error': ("azimuth_velocity_error", "error for velocity in radar azimuth direction"),
            'vr_error': ("range_velocity_error", "error for velocity in radar range direction"),
            'vxp_error': ("projected_x_velocity_error", "error for x-direction velocity determined by projecting radar range measurements onto an a priori flow vector"),
            'vyp_error': ("projected_y_velocity_error", "error for y-direction velocity determined by projecting radar range measurements onto an a priori flow vector"),
            # The following descriptions are the same for all v* data variables
            'error_mask': (None, "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 m/yr identified from an external mask"),
            'error_slow': (None, "RMSE over slowest 25% of retrieved velocities"),
            'error_modeled': (None, "1-sigma error calculated using a modeled error-dt relationship"),
        }

        # Possible attributes for the velocity data variable
        _v_comp_attrs = [
            DataVars.ERROR,
            DataVars.ERROR_MASK,
            DataVars.ERROR_MODELED,
            DataVars.ERROR_SLOW
        ]

        # Names of new data variables - to be included into "encoding" settings
        # for writing to the file store.
        return_vars = []

        # Process attributes
        if DataVars.STABLE_APPLY_DATE in self.layers[var_name].attrs:
            # Remove optical legacy attribute if it propagated to the cube data
            del self.layers[var_name].attrs[DataVars.STABLE_APPLY_DATE]

        # If attribute is propagated as cube's data var attribute, delete it.
        # These attributes were collected based on 'v' data variable
        if DataVars.MAP_SCALE_CORRECTED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.MAP_SCALE_CORRECTED]

        if DataVars.STABLE_SHIFT_APPLIED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT_APPLIED]

        _name_sep = '_'

        for each_prefix in _v_comp_attrs:
            error_name = f'{var_name}{_name_sep}{each_prefix}'
            return_vars.append(error_name)

            # Special care must be taken of v[xy].stable_rmse in
            # optical legacy format vs. v[xy].v[xy]_error in radar format as these
            # are the same
            error_data = None
            if var_name in _stable_rmse_vars:
                error_data = [
                    ITSCube.get_data_var_attr(ds, url, var_name, error_name, DataVars.MISSING_VALUE) if error_name in ds[var_name].attrs else
                    ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_RMSE, DataVars.MISSING_VALUE)
                    for ds, url in zip(self.ds, self.urls)
                ]

                # If attribute is propagated as cube's data var attribute, delete it
                if DataVars.STABLE_RMSE in self.layers[var_name].attrs:
                    del self.layers[var_name].attrs[DataVars.STABLE_RMSE]

            else:
                error_data = [ITSCube.get_data_var_attr(ds, url, var_name, error_name, DataVars.MISSING_VALUE)
                              for ds, url in zip(self.ds, self.urls)]

            error_name_desc = f'{error_name}{_name_sep}{DataVars.ERROR_DESCRIPTION}'
            desc_str = None
            if var_name in self.ds[0] and error_name_desc in self.ds[0][var_name].attrs:
                desc_str = self.ds[0][var_name].attrs[error_name_desc]

            elif each_prefix in _attrs:
                # If generic description is provided
                desc_str = _attrs[each_prefix][1]

            elif error_name in _attrs:
                # If variable specific description is provided
                desc_str = _attrs[error_name][1]

            else:
                raise RuntimeError(f"Unknown description for {error_name} of {var_name}")

            self.layers[error_name] = xr.DataArray(
                data=error_data,
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.M_Y_UNITS,
                    DataVars.STD_NAME: error_name,
                    DataVars.DESCRIPTION_ATTR: desc_str
                }
            )

            # If attribute is propagated as cube's data var attribute, delete it
            if error_name in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[error_name]

            # If attribute description is in the var's attributes, remove it
            if error_name_desc in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[error_name_desc]

        # This attribute appears for all v* data variables of old granule format,
        # capture it only once if it exists
        if DataVars.STABLE_COUNT not in self.layers and \
           var_name in self.ds[0] and \
           DataVars.STABLE_COUNT in self.ds[0][var_name].attrs:
            self.layers[DataVars.STABLE_COUNT] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_COUNT)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.COUNT_UNITS,
                    DataVars.STD_NAME: DataVars.STABLE_COUNT,
                    DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.STABLE_COUNT].format(var_name)
            }
        )
        if DataVars.STABLE_COUNT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_COUNT]

        # This attribute appears for all v* data variables of new granule format,
        # capture it only once if it exists
        # Per Yang: generally yes, though for vxp and vyp it was calculated again
        # but the number should not change quite a bit. so it should be okay to
        # use a single value for all variables
        # (access variable only if it exists in granule)
        if DataVars.STABLE_COUNT_SLOW not in self.layers and \
           var_name in self.ds[0] and \
           DataVars.STABLE_COUNT_SLOW in self.ds[0][var_name].attrs:
            self.layers[DataVars.STABLE_COUNT_SLOW] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_COUNT_SLOW)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.COUNT_UNITS,
                    DataVars.STD_NAME: DataVars.STABLE_COUNT_SLOW,
                    DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.STABLE_COUNT_SLOW]
            }
        )
        if DataVars.STABLE_COUNT_SLOW in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_COUNT_SLOW]

        # This attribute appears for all v* data variables, capture it only once
        # if it exists
        # (access variable only if it exists in granule)
        if DataVars.STABLE_COUNT_MASK not in self.layers and \
           var_name in self.ds[0] and \
           DataVars.STABLE_COUNT_MASK in self.ds[0][var_name].attrs:
            self.layers[DataVars.STABLE_COUNT_MASK] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_COUNT_MASK)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.COUNT_UNITS,
                    DataVars.STD_NAME: DataVars.STABLE_COUNT_MASK,
                    DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.STABLE_COUNT_MASK]
            }
        )
        if DataVars.STABLE_COUNT_MASK in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_COUNT_MASK]

        # This attribute appears for vx and vy data variables, capture it only once.
        # "stable_shift_applied" was incorrectly set in the optical legacy dataset
        # and should be set to "no data" value
        # (access variable only if it exists in granule)
        if DataVars.FLAG_STABLE_SHIFT not in self.layers and \
           var_name in self.ds[0]:
            missing_stable_shift_value = 0.0
            self.layers[DataVars.FLAG_STABLE_SHIFT] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.FLAG_STABLE_SHIFT, missing_stable_shift_value)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.STD_NAME: DataVars.FLAG_STABLE_SHIFT,
                    DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.FLAG_STABLE_SHIFT_DESCRIPTION]
                }
            )

        # Remove DataVars.FLAG_STABLE_SHIFT from velocity variable of the datacube
        # if present
        if DataVars.FLAG_STABLE_SHIFT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.FLAG_STABLE_SHIFT]

        if DataVars.FLAG_STABLE_SHIFT_DESCRIPTION in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.FLAG_STABLE_SHIFT_DESCRIPTION]

        # Create 'stable_shift' specific to the data variable,
        # for example, 'vx_stable_shift' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT])
        self.layers[shift_var_name] = xr.DataArray(
            data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_SHIFT, DataVars.MISSING_VALUE)
                  for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: shift_var_name,
                DataVars.DESCRIPTION_ATTR: f'applied {var_name} shift calibrated using pixels over stable or slow surfaces'
            }
        )
        return_vars.append(shift_var_name)

        if DataVars.STABLE_SHIFT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT]

        # Create 'stable_shift_slow' specific to the data variable,
        # for example, 'vx_stable_shift_slow' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT_SLOW])
        self.layers[shift_var_name] = xr.DataArray(
            data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_SHIFT_SLOW, DataVars.MISSING_VALUE)
                  for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: shift_var_name,
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.STABLE_SHIFT_SLOW].format(var_name)
            }
        )
        return_vars.append(shift_var_name)

        # If attribute is propagated as cube's vx attribute, delete it
        if DataVars.STABLE_SHIFT_SLOW in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT_SLOW]

        # Create 'stable_shift_mask' specific to the data variable,
        # for example, 'vx_stable_shift_mask' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT_MASK])
        self.layers[shift_var_name] = xr.DataArray(
            data=[ITSCube.get_data_var_attr(ds, url, var_name, DataVars.STABLE_SHIFT_MASK, DataVars.MISSING_VALUE)
                  for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: shift_var_name,
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.STABLE_SHIFT_MASK].format(var_name)
            }
        )
        return_vars.append(shift_var_name)

        # If attribute is propagated as cube's vx attribute, delete it
        if DataVars.STABLE_SHIFT_MASK in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT_MASK]

        # Return names of new data variables - to be included into "encoding" settings
        # for writing to the file store.
        return return_vars

    def set_grid_mapping_attr(self, var_name: str, ds_grid_mapping_value: str):
        """
        Check on existence of "grid_mapping" attribute for the variable, set it
        if not present.
        """
        if DataVars.GRID_MAPPING in self.layers[var_name].attrs:
            # Attribute is already set, nothing to do
            return

        self.layers[var_name].attrs[DataVars.GRID_MAPPING] = ds_grid_mapping_value

        # This was for old granule format where some of the data variables were
        # missing the attribute:
        # grid_mapping_values = []
        # for each_ds in self.ds:
        #     if var_name in each_ds and DataVars.GRID_MAPPING in each_ds[var_name].attrs:
        #         grid_mapping_values.append(each_ds[var_name].attrs[DataVars.GRID_MAPPING])
        #
        # # Flag if attribute needs to be set manually
        # set_grid_mapping = False
        # if len(grid_mapping_values) != len(self.ds):
        #     # None or some of the granules provide grid_mapping attribute
        #     # ("var_name" data variable might be present only in Radar format),
        #     # need to set it manually as xr.concat won't preserve the attribute
        #     set_grid_mapping = True
        #
        # unique_values = list(set(grid_mapping_values))
        # if len(unique_values) > 1:
        #     raise RuntimeError(
        #         f"Inconsistent '{var_name}.{DataVars.GRID_MAPPING}' values are "
        #         "detected for current {len(self.ds)} layers: {unique_values}")
        #
        # if len(unique_values) and unique_values[0] != ds_grid_mapping_value:
        #     # Make sure the value is the same as previously detected
        #     raise RuntimeError(
        #         f"Inconsistent '{DataVars.GRID_MAPPING}' value in "
        #         "{var_name}: {self.layers[var_name].attrs[DataVars.GRID_MAPPING]} vs. {ds_grid_mapping_value}")
        #
        # if set_grid_mapping:
        #     self.layers[var_name].attrs[DataVars.GRID_MAPPING] = ds_grid_mapping_value

    @staticmethod
    def show_memory_usage(msg: str=''):
        """
        Display current memory usage.
        """
        _GB = 1024 * 1024 * 1024
        usage = psutil.virtual_memory()

        # Use standard logging to be able to use the method without ITSCube object
        memory_msg = 'Memory '
        if len(msg):
            memory_msg += msg

        logging.info(f"{memory_msg}: total={usage.total/_GB}Gb used={usage.used/_GB}Gb available={usage.available/_GB}Gb")

    def combine_layers(self, output_dir, is_first_write=False):
        """
        Combine selected layers into one xr.Dataset object and write (append) it
        to the Zarr store.
        """
        self.layers = {}
        wrote_layers = False

        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        self.logger.info(f'Combine {len(self.urls)} layers to the {output_dir}...')
        if len(self.ds) == 0:
            self.logger.info('No layers to combine, continue')
            return wrote_layers

        # ITSCube.show_memory_usage('before combining layers')
        wrote_layers = True

        start_time = timeit.default_timer()
        mid_date_coord = pd.Index(self.dates, name=Coords.MID_DATE)

        self.layers = xr.Dataset(
            data_vars = {DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords = {
                Coords.MID_DATE: (
                    Coords.MID_DATE,
                    self.dates,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.MID_DATE],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.MID_DATE]
                    }
                ),
                Coords.X: (
                    Coords.X,
                    self.grid_x,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
                    }
                ),
                Coords.Y: (
                    Coords.Y,
                    self.grid_y,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
                    }
                )
            },
            attrs = {
                'title': 'ITS_LIVE datacube of image_pair velocities',
                'author': 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)',
                'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                'date_created': self.date_created,
                'date_updated': self.date_updated if self.date_updated is not None else self.date_created,
                'datacube_software_version': ITSCube.Version,
                'GDAL_AREA_OR_POINT': 'Area',
                'projection': str(self.projection),
                'longitude': f"{self.center_lon_lat[0]:.2f}",
                'latitude':  f"{self.center_lon_lat[1]:.2f}",
                'skipped_empty_data': json.dumps(self.skipped_empty_granules),
                'skipped_duplicate_middle_date': json.dumps(self.skipped_double_granules),
                'skipped_wrong_projection': json.dumps(self.skipped_proj_granules)
            }
        )

        # Set attributes for 'url' data variable
        self.layers[DataVars.URL].attrs[DataVars.STD_NAME] = DataVars.URL
        self.layers[DataVars.URL].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.URL]

        # Set projection information once for the whole datacube
        if is_first_write:
            proj_data = None
            if DataVars.POLAR_STEREOGRAPHIC in self.ds[0]:
                proj_data = DataVars.POLAR_STEREOGRAPHIC

            elif DataVars.UTM_PROJECTION in self.ds[0]:
                proj_data = DataVars.UTM_PROJECTION

            elif DataVars.MAPPING in self.ds[0]:
                proj_data = DataVars.MAPPING

            # Should never happen - just in case :)
            if proj_data is None:
                raise RuntimeError(f"Missing one of [{DataVars.POLAR_STEREOGRAPHIC}, {DataVars.UTM_PROJECTION}, {DataVars.MAPPING}] in {self.urls[0]}")

            # Can't copy the whole data variable, as it introduces obscure coordinates.
            # Just copy all attributes for the scalar type of the xr.DataArray.
            # Use latest granule format: 'mapping' data variable for projection info.
            self.layers[DataVars.MAPPING] = xr.DataArray(
                data='',
                attrs=self.ds[0][proj_data].attrs,
                coords={},
                dims=[]
            )

            # Set GeoTransform to correspond to the datacube's tile
            x_size = self.grid_x[1] - self.grid_x[0]
            y_size = self.grid_y[1] - self.grid_y[0]

            half_x_cell = x_size/2.0
            half_y_cell = y_size/2.0

            # Format cube's GeoTransform
            new_geo_transform_str = f"{self.grid_x[0] - half_x_cell} {x_size} 0 {self.grid_y[0] - half_y_cell} 0 {y_size}"
            self.layers[DataVars.MAPPING].attrs['GeoTransform'] = new_geo_transform_str

        # ATTN: Assign one data variable at a time to avoid running out of memory.
        #       Delete each variable after it has been processed to free up the
        #       memory.

        # Process 'v' (all formats have v variable - its attributes are inherited,
        # so no need to set them manually)
        v_layers = xr.concat([each_ds.v for each_ds in self.ds], mid_date_coord)

        self.layers[DataVars.V] = v_layers
        self.layers[DataVars.V].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.V]
        new_v_vars = [DataVars.V]

        # Make sure grid_mapping attribute has the same value for all layers
        unique_values = None
        # Remember the value as all 3D data variables need to have this attribute
        # set with the same value
        ds_grid_mapping_value = None
        if self.ds[0].v.attrs[DataVars.GRID_MAPPING] == DataVars.MAPPING:
            # New format granules
            grid_mapping_values = [ds.mapping.attrs[DataVars.GRID_MAPPING_NAME] for ds in self.ds]
            unique_values = list(set(grid_mapping_values))
            if len(unique_values) > 1:
                raise RuntimeError(f"Multiple '{DataVars.MAPPING}' values are detected for current {len(self.ds)} layers: {unique_values}")
            ds_grid_mapping_value = DataVars.MAPPING

        else:
            # Old format granules
            grid_mapping_values = [ds.v.attrs[DataVars.GRID_MAPPING] for ds in self.ds]
            unique_values = list(set(grid_mapping_values))
            if len(unique_values) > 1:
                raise RuntimeError(f"Multiple '{DataVars.GRID_MAPPING}' ('v' attribute) values are detected for current {len(self.ds)} layers: {unique_values}")
            ds_grid_mapping_value = unique_values[0]

        # For old format collect 'v' attributes: these repeat for v* variables, keep only one copy
        # per datacube
        # Create new data var to store map_scale_corrected v's attribute
        if DataVars.MAP_SCALE_CORRECTED in self.ds[0].v.attrs:
            self.layers[DataVars.MAP_SCALE_CORRECTED] = xr.DataArray(
                data = [ITSCube.get_data_var_attr(
                            ds,
                            url,
                            DataVars.V,
                            DataVars.MAP_SCALE_CORRECTED,
                            DataVars.MISSING_BYTE
                        ) for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE]
            )

            # If attribute is propagated as cube's v attribute, delete it
            if DataVars.MAP_SCALE_CORRECTED in self.layers[DataVars.V].attrs:
                del self.layers[DataVars.V].attrs[DataVars.MAP_SCALE_CORRECTED]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [each.drop_vars(DataVars.V) for each in self.ds]
        del v_layers
        gc.collect()

        # Process 'v_error'
        self.layers[DataVars.V_ERROR] = xr.concat(
            [self.get_data_var(ds, DataVars.V_ERROR) for ds in self.ds],
            mid_date_coord
        )
        self.layers[DataVars.V_ERROR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.V_ERROR)

        self.set_grid_mapping_attr(DataVars.V_ERROR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.V_ERROR) if DataVars.V_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vx'
        self.layers[DataVars.VX] = xr.concat([ds.vx for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VX].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VX]
        new_v_vars.append(DataVars.VX)
        new_v_vars.extend(self.process_v_attributes(DataVars.VX, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VX, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VX) for ds in self.ds]
        gc.collect()

        # Process 'vy'
        self.layers[DataVars.VY] = xr.concat([ds.vy for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VY].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VY]
        new_v_vars.append(DataVars.VY)
        new_v_vars.extend(self.process_v_attributes(DataVars.VY, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VY, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VY) for ds in self.ds]
        gc.collect()

        # Process 'va'
        self.layers[DataVars.VA] = xr.concat([self.get_data_var(ds, DataVars.VA) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VA].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VA]
        self.layers[DataVars.VA].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VA]
        self.layers[DataVars.VA].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        self.set_grid_mapping_attr(DataVars.VA, ds_grid_mapping_value)

        new_v_vars.append(DataVars.VA)
        new_v_vars.extend(self.process_v_attributes(DataVars.VA, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VA) if DataVars.VA in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vr'
        self.layers[DataVars.VR] = xr.concat([self.get_data_var(ds, DataVars.VR) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VR]
        self.layers[DataVars.VR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VR]
        self.layers[DataVars.VR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VR)
        new_v_vars.extend(self.process_v_attributes(DataVars.VR, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VR) if DataVars.VR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vxp'
        self.layers[DataVars.VXP] = xr.concat([self.get_data_var(ds, DataVars.VXP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VXP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VXP]
        self.layers[DataVars.VXP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VXP]
        self.layers[DataVars.VXP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VXP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VXP, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VXP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VXP) if DataVars.VXP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vyp'
        self.layers[DataVars.VYP] = xr.concat([self.get_data_var(ds, DataVars.VYP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VYP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VYP]
        self.layers[DataVars.VYP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VYP]
        self.layers[DataVars.VYP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.VYP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VYP, mid_date_coord))

        self.set_grid_mapping_attr(DataVars.VYP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VYP) if DataVars.VYP in ds else ds for ds in self.ds]
        gc.collect()

        # Process chip_size_height: dtype=ushort
        # Optical legacy granules might not have chip_size_height set, use
        # chip_size_width instead
        self.layers[DataVars.CHIP_SIZE_HEIGHT] = xr.concat([
               ds.chip_size_height if
                  np.ma.masked_equal(ds.chip_size_height.values, ITSCube.CHIP_SIZE_HEIGHT_NO_VALUE).count() != 0 else
               ds.chip_size_width for ds in self.ds
            ],
            mid_date_coord)
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.CHIP_SIZE_COORDS] = \
            DataVars.DESCRIPTION[DataVars.CHIP_SIZE_COORDS]
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.DESCRIPTION_ATTR] = \
            DataVars.DESCRIPTION[DataVars.CHIP_SIZE_HEIGHT]

        self.set_grid_mapping_attr(DataVars.CHIP_SIZE_HEIGHT, ds_grid_mapping_value)

        # Report if using chip_size_width in place of chip_size_height
        concat_ind = [ind for ind, ds in enumerate(self.ds) if np.ma.masked_equal(ds.chip_size_height.values, ITSCube.CHIP_SIZE_HEIGHT_NO_VALUE).count() == 0]
        for each in concat_ind:
            self.logger.warning(f'Using chip_size_width in place of chip_size_height for {self.urls[each]}')

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_HEIGHT) for ds in self.ds]
        gc.collect()

        # Process chip_size_width: dtype=ushort
        self.layers[DataVars.CHIP_SIZE_WIDTH] = xr.concat([ds.chip_size_width for ds in self.ds], mid_date_coord)
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.CHIP_SIZE_COORDS] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_COORDS]
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.CHIP_SIZE_WIDTH]

        self.set_grid_mapping_attr(DataVars.CHIP_SIZE_WIDTH, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_WIDTH) for ds in self.ds]
        gc.collect()

        # Process interp_mask: dtype=ubyte
        self.layers[DataVars.INTERP_MASK] = xr.concat([ds.interp_mask for ds in self.ds], mid_date_coord)
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.UNITS] = DataVars.BINARY_UNITS

        self.set_grid_mapping_attr(DataVars.INTERP_MASK, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.INTERP_MASK) for ds in self.ds]
        gc.collect()

        # Process 'vp'
        self.layers[DataVars.VP] = xr.concat([self.get_data_var(ds, DataVars.VP) for ds in self.ds] , mid_date_coord)
        self.layers[DataVars.VP].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VP]
        self.layers[DataVars.VP].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VP]
        self.layers[DataVars.VP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS
        new_v_vars.append(DataVars.VP)

        self.set_grid_mapping_attr(DataVars.VP, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP) if DataVars.VP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vp_error'
        self.layers[DataVars.VP_ERROR] = xr.concat([self.get_data_var(ds, DataVars.VP_ERROR) for ds in self.ds] , mid_date_coord)
        self.layers[DataVars.VP_ERROR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.VP_ERROR]
        self.layers[DataVars.VP_ERROR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.VP_ERROR]
        self.layers[DataVars.VP_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS
        new_v_vars.append(DataVars.VP_ERROR)

        self.set_grid_mapping_attr(DataVars.VP_ERROR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP_ERROR) if DataVars.VP_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        for each in DataVars.ImgPairInfo.ALL:
            # Add new variables that correspond to attributes of 'img_pair_info'
            # (only selected ones)
            self.layers[each] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, each, to_date=DataVars.ImgPairInfo.CONVERT_TO_DATE[each]
                ) for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[each],
                    DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[each]
                }
            )

            if each in DataVars.ImgPairInfo.UNITS:
                # Units attribute exists for the variable
                self.layers[each].attrs[DataVars.UNITS] = DataVars.ImgPairInfo.UNITS[each]

        # Add new variable that corresponds to autoRIFT_software_version
        self.layers[DataVars.AUTORIFT_SOFTWARE_VERSION] = xr.DataArray(
            data=[ds.attrs[DataVars.AUTORIFT_SOFTWARE_VERSION] for ds in self.ds],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: DataVars.AUTORIFT_SOFTWARE_VERSION,
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.AUTORIFT_SOFTWARE_VERSION]
            }
        )
        # Set datacube attribute to capture autoRIFT parameter file
        self.layers.attrs[DataVars.AUTORIFT_PARAMETER_FILE] = self.ds[0].attrs[DataVars.AUTORIFT_PARAMETER_FILE]

        # Make sure all layers have the same parameter file
        all_values = [ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE] for ds in self.ds]
        unique_values = list(set(all_values))
        if len(unique_values) > 1:
            raise RuntimeError(f"Multiple values for '{DataVars.AUTORIFT_PARAMETER_FILE}' are detected for current {len(self.ds)} layers: {unique_values}")


        # Handle acquisition time separately as it has different names in
        # optical and radar formats
        var_name = DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1
        # If not supporting old granule format, remove this backward compatability:
        old_var_name = DataVars.ImgPairInfo.ACQUISITION_IMG1
        self.layers[var_name] = xr.DataArray(
            data=[
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, old_var_name, to_date = True
                ) if old_var_name in ds[DataVars.ImgPairInfo.NAME].attrs else
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, var_name, to_date = True
                ) for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[var_name],
                DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[var_name],
            }
        )

        var_name = DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2
        # If not supporting old granule format, remove this backward compatability:
        old_var_name = DataVars.ImgPairInfo.ACQUISITION_IMG2
        self.layers[var_name] = xr.DataArray(
            data=[
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, old_var_name, to_date = True
                ) if old_var_name in ds[DataVars.ImgPairInfo.NAME].attrs else
                ITSCube.get_data_var_attr(
                    ds, url, DataVars.ImgPairInfo.NAME, var_name, to_date = True
                ) for ds, url in zip(self.ds, self.urls)],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: DataVars.ImgPairInfo.STD_NAME[var_name],
                DataVars.DESCRIPTION_ATTR: DataVars.ImgPairInfo.DESCRIPTION[var_name],
            }
        )

        # Set time standard as datacube attributes
        for var_name in [
            DataVars.ImgPairInfo.TIME_STANDARD_IMG1,
            DataVars.ImgPairInfo.TIME_STANDARD_IMG2
        ]:
            self.layers.attrs[var_name] = self.ds[0].img_pair_info.attrs[var_name]

            # Make sure all layers have the same time standard
            all_values = [ds.img_pair_info.attrs[var_name] for ds in self.ds]
            unique_values = list(set(all_values))
            if len(unique_values) > 1:
                raise RuntimeError(f"Multiple values for '{var_name}' are detected for current {len(self.ds)} layers: {unique_values}")

        # ATTN: Set attributes for the Dataset coordinates as the very last step:
        # when adding data variables that don't have the same attributes for the
        # coordinates, originally set Dataset coordinates will be wiped out
        self.layers[Coords.MID_DATE].attrs = MID_DATE_ATTRS
        self.layers[Coords.X].attrs = X_ATTRS
        self.layers[Coords.Y].attrs = Y_ATTRS

        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Combined {len(self.urls)} layers (took {time_delta} seconds)")
        # ITSCube.show_memory_usage('after combining layers')

        compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)
        compression = {"compressor": compressor}

        start_time = timeit.default_timer()
        # Write to the Zarr store
        if is_first_write:
            # Set missing_value only on first write to the disk store, otherwise
            # will get "ValueError: failed to prevent overwriting existing key missing_value in attrs."
            # "missing_value" attribute is depricated
            # for each in [DataVars.MAP_SCALE_CORRECTED,
            #              DataVars.CHIP_SIZE_HEIGHT,
            #              DataVars.CHIP_SIZE_WIDTH,
            #              DataVars.INTERP_MASK]:
            #     if each in self.layers:
            #         # Since MAP_SCALE_CORRECTED is present only in old granule format
            #         self.layers[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

            # ATTN: Must set '_FillValue' for each data variable that has
            #       its missing_value attribute set
            encoding_settings = {}
            for each in [DataVars.INTERP_MASK,
                         DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH]:
                encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE}

            # Treat it outside of "for" loop
            if DataVars.MAP_SCALE_CORRECTED in self.layers:
                # Since MAP_SCALE_CORRECTED is present only in old granule format
                encoding_settings[DataVars.MAP_SCALE_CORRECTED] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE}
                encoding_settings[DataVars.MAP_SCALE_CORRECTED]['dtype'] = 'byte'

            # Explicitly set dtype to 'byte' for some data variables
            for each in [DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH]:
                encoding_settings[each]['dtype'] = 'ushort'

            # Explicitly set dtype for some variables
            encoding_settings[DataVars.INTERP_MASK]['dtype'] = 'ubyte'
            for each in [
                DataVars.FLAG_STABLE_SHIFT,
                DataVars.STABLE_COUNT_SLOW,
                DataVars.STABLE_COUNT_MASK
                ]:
                encoding_settings.setdefault(each, {})['dtype'] = 'long'

            # Old format granules
            if DataVars.STABLE_COUNT in self.layers:
                encoding_settings.setdefault(DataVars.STABLE_COUNT, {})['dtype'] = 'long'

            for each in new_v_vars:
                encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE}
                encoding_settings[each].update(compression)

                # Set missing_value only on first write to the disk store, otherwise
                # will get "ValueError: failed to prevent overwriting existing key
                # missing_value in attrs."
                if DataVars.MISSING_VALUE_ATTR not in self.layers[each].attrs:
                    self.layers[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_VALUE

            # Explicitly set dtype to 'short' for v* data variables
            for each in [DataVars.V,
                         DataVars.VX,
                         DataVars.VY,
                         DataVars.VA,
                         DataVars.VR,
                         DataVars.VXP,
                         DataVars.VYP,
                         DataVars.VP,
                         DataVars.V_ERROR,
                         DataVars.VP_ERROR]:
                encoding_settings[each]['dtype'] = 'short'

            # Explicitly desable _FillValue for some variables
            for each in [Coords.MID_DATE,
                         DataVars.STABLE_COUNT_SLOW,
                         DataVars.STABLE_COUNT_MASK,
                         DataVars.AUTORIFT_SOFTWARE_VERSION,
                         DataVars.ImgPairInfo.DATE_DT,
                         DataVars.ImgPairInfo.DATE_CENTER,
                         DataVars.ImgPairInfo.SATELLITE_IMG1,
                         DataVars.ImgPairInfo.SATELLITE_IMG2,
                         DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
                         DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
                         DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE,
                         DataVars.ImgPairInfo.MISSION_IMG1,
                         DataVars.ImgPairInfo.MISSION_IMG2,
                         DataVars.ImgPairInfo.SENSOR_IMG1,
                         DataVars.ImgPairInfo.SENSOR_IMG2]:
                encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})
            # If old format granule
            if DataVars.STABLE_COUNT in self.layers:
                encoding_settings.setdefault(DataVars.STABLE_COUNT, {}).update({DataVars.FILL_VALUE_ATTR: None})

            # Set units for all datetime objects
            for each in [DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
                         DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
                         DataVars.ImgPairInfo.DATE_CENTER,
                         Coords.MID_DATE]:
                encoding_settings.setdefault(each, {}).update({DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS})

            self.logger.info(f"Encoding writing to Zarr: {encoding_settings}")
            # self.logger.info(f"Data variables to Zarr:   {json.dumps(list(self.layers.keys()), indent=4)}")

            # This is first write, create Zarr store
            # self.layers.to_zarr(output_dir, encoding=encoding_settings, consolidated=True)
            self.layers.to_zarr(output_dir, encoding=encoding_settings, consolidated=True)

        else:
            # Append layers to existing Zarr store
            # self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE, consolidated=True)
            self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE, consolidated=True)

        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Wrote {len(self.urls)} layers to {output_dir} (took {time_delta} seconds)")

        # Free up memory
        self.clear_vars()

        # No need to sort data by date as we will be appending layers to the datacubes

        # Return a flag if any layers were written to the store
        return wrote_layers

    def format_stats(self):
        """
        Format statistics of the run.
        """
        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_proj_granules.values()])

        self.logger.info(f"Skipped granules due to empty data: {len(self.skipped_empty_granules)} ({100.0 * len(self.skipped_empty_granules)/num_urls}%)")
        self.logger.info(f"Skipped granules due to double mid_date: {len(self.skipped_double_granules)} ({100.0 * len(self.skipped_double_granules)/num_urls}%)")
        self.logger.info(f"Skipped granules due to wrong projection: {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        if len(self.skipped_proj_granules):
            self.logger.info(f"Skipped wrong projections: {sorted(self.skipped_proj_granules.keys())}")

    def read_dataset(self, url: str):
        """
        Read Dataset from the file and pre-process for the cube layer.
        """
        with xr.open_dataset(url) as ds:
            return self.preprocess_dataset(ds, url)

    def read_s3_dataset(self, each_url: str, s3):
        """
        Read Dataset from the S3 bucket and pre-process for the cube layer.
        """
        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

        with s3.open(s3_path, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                return self.preprocess_dataset(ds, each_url)

    @staticmethod
    def plot(cube, variable, boundaries: tuple = None):
        """
        Plot cube's layers data. All layers share the same x/y coordinate labels.
        There is an option to display only a subset of layers by specifying
        start and end index through "boundaries" input parameter.
        """
        if boundaries is not None:
            start, end = boundaries
            cube[variable][start:end].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)

        else:
            cube[variable].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)

    @staticmethod
    def validate_cube_datetime(ds: xr.Dataset, start_date: str, cube_url: str):
        """
        Validate datetime objects of the cube against start_date of the cube.
        This check is introduced to capture corrupted datacubes as early as
        possible in the cube generation.
        """
        # ATTN: This checking assumes that start_date corresponds to the start
        # date of the data used to create the datacube
        start_date = np.datetime64(start_date)
        logging.info(f"Validating datetime objects for {cube_url}")

        values = ds.acquisition_date_img1.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected acquisition_date_img1: {values.min()}")

        values = ds.acquisition_date_img2.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected acquisition_date_img2: {values.min()}")

        values = ds.date_center.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected date_center: {values.min()}")

        values = ds.mid_date.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected mid_date: {values.min()}")


if __name__ == '__main__':
    # Since port forwarding is not working on EC2 to run jupyter lab for now,
    # allow to run test case from itscube.ipynb in standalone mode
    import argparse
    import warnings
    import sys
    import subprocess

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSCube.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-t', '--threads',
        type=int, default=4,
        help='number of Dask workers to use for parallel processing [%(default)d].'
    )
    parser.add_argument(
        '-s', '--scheduler',
        type=str,
        default="processes",
        help="Dask scheduler to use. One of ['threads', 'processes'] (effective only when --parallel option is specified) [%(default)s]."
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Enable parallel processing, default is to process all granules in parallel'
    )
    parser.add_argument(
        '-n', '--numberGranules',
        type=int,
        default=None,
        help="number of ITS_LIVE granules to consider for the cube (due to runtime limitations). "
             " If none is provided, process all found granules."
    )
    parser.add_argument(
        '-l', '--localPath',
        type=str,
        default=None,
        help='Local path that stores ITS_LIVE granules.'
    )
    parser.add_argument(
        '-o', '--outputStore',
        type=str,
        default="cubedata.zarr",
        help="Zarr output directory to write cube data to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--outputBucket',
        type=str,
        default='',
        help="S3 bucket to copy Zarr format of the datacube to [%(default)s]."
    )
    parser.add_argument(
        '-c', '--chunks',
        type=int,
        default=250,
        help="Number of granules to write at a time [%(default)d]."
    )
    parser.add_argument(
        '--targetProjection',
        type=str,
        required=True,
        help="UTM target projection."
    )
    parser.add_argument(
        '--dimSize',
        type=float,
        default=100000,
        help="Cube dimension in meters [%(default)d]."
    )
    parser.add_argument(
        '-g', '--gridCellSize',
        type=int,
        default=240,
        help="Grid cell size of input ITS_LIVE granules [%(default)d]."
    )
    parser.add_argument(
        '--fivePointsPerPolygonSide',
        action='store_true',
        help='Define 5 points per side before re-projecting granule polygon to longitude/latitude coordinates'
    )
    parser.add_argument(
        '--searchAPIStartDate',
        type=str,
        default='1984-01-01',
        help='Start date in YYYY-MM-DD format to pass to search API query to get velocity pair granules'
    )
    parser.add_argument(
        '--searchAPIStopDate',
        type=str,
        default=None,
        help='Stop date in YYYY-MM-DD format to pass to search API query to get velocity pair granules'
    )
    parser.add_argument(
        '--disableCubeValidation',
        action='store_true',
        default=False,
        help='Disable datetime validation for created datacube. This is to identify corrupted Zarr stores at the time of creation.'
    )

    # One of --centroid or --polygon options is allowed for the datacube coordinates
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--centroid',
        type=str,
        action='store',
        help="JSON 2-element list for centroid point (x, y) of the datacube in target EPSG code projection. "
        "Polygon vertices are calculated based on the centroid and cube dimension arguments."
    )
    group.add_argument(
        '--polygon',
        type=str,
        action='store',
        help="JSON list of polygon points ((x1, y1), (x2, y2),... (x1, y1)) to define datacube in target EPSG code projection."
    )

    args = parser.parse_args()
    ITSCube.NUM_THREADS = args.threads
    ITSCube.DASK_SCHEDULER = args.scheduler
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks
    ITSCube.CELL_SIZE = args.gridCellSize

    projection = args.targetProjection

    polygon = None
    if args.centroid:
        # Centroid for the tile is provided in target projection
        c_x, c_y = list(map(float, json.loads(args.centroid)))

        # Offset in meters (1 pixel=240m): 100 km square (with offset=50km)
        # off = 50000
        off = args.dimSize / 2.0
        polygon = (
            (c_x - off, c_y + off),
            (c_x + off, c_y + off),
            (c_x + off, c_y - off),
            (c_x - off, c_y - off),
            (c_x - off, c_y + off))
    else:
        # Polygon for the cube definition is provided
        polygon = json.loads(args.polygon)

    if args.fivePointsPerPolygonSide:
        # Introduce 5 points per each polygon side
        polygon = itslive_utils.add_five_points_to_polygon_side(polygon)

    # Create cube object
    cube = ITSCube(polygon, projection)

    # Record used package versions
    cube.logger.info(f"Command: {sys.argv}")
    cube.logger.info(f"Command args: {args}")
    cube.logger.info(f"{xr.show_versions()}")
    cube.logger.info(f"s3fs: {s3fs.__version__}")

    # Parameters for the search granule API
    end_date = datetime.now().strftime('%Y-%m-%d') if args.searchAPIStopDate is None else args.searchAPIStopDate
    API_params = {
        'start'               : args.searchAPIStartDate,
        'end'                 : end_date,
        'percent_valid_pixels': 1
    }
    cube.logger.info("ITS_LIVE API parameters: %s" %API_params)

    cube.create_or_update(API_params, args.outputStore, args.outputBucket, args.numberGranules)

    # This is for debugging only to be able to run non-parallel processing
    # if not args.parallel:
    #     # Process ITS_LIVE granules sequentially, look at provided number of granules only
    #     cube.logger.info("Processing granules sequentially...")
    #     if args.localPath:
    #         # Granules are downloaded locally
    #         cube.create_from_local_no_api(args.outputStore, args.localPath, args.numberGranules)
    #
    #     else:
    #         cube.create(API_params, args.outputStore, args.numberGranules)
    #
    # else:
    #     # Process ITS_LIVE granules in parallel, look at 100 first granules only
    #     cube.logger.info("Processing granules in parallel...")
    #     if args.localPath:
    #         # Granules are downloaded locally
    #         cube.create_from_local_parallel_no_api(args.outputStore, args.localPath, args.numberGranules)
    #
    #     else:
    #         cube.create_parallel(API_params, args.outputStore, args.numberGranules)

    cube = None
    gc.collect()

    ITSCube.show_memory_usage('at the end of datacube generation')

    if not args.disableCubeValidation and os.path.exists(args.outputStore):
        with xr.open_zarr(args.outputStore, decode_timedelta=False, consolidated=True) as ds:
            ITSCube.validate_cube_datetime(ds, args.searchAPIStartDate, args.outputStore)

        gc.collect()

    if os.path.exists(args.outputStore) and len(args.outputBucket):
        # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
        # resulting in as many error messages as there are files in Zarr store
        # to copy

        # Remove Zarr store in S3 if it exists: updated Zarr, which is stored to the
        # local file system before copying to the S3 bucket, might have different
        # "sub-directory" structure. This will result in original "sub-directories"
        # and "new" ones to co-exist for the same Zarr store. This doubles up
        # the Zarr disk usage in S3 bucket.
        env_copy = os.environ.copy()
        if ITSCube.exists(args.outputStore, args.outputBucket):
            command_line = [
                "aws", "s3", "rm", "--recursive",
                os.path.join(args.outputBucket, args.outputStore)
            ]
            logging.info(' '.join(command_line))

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to remove original {args.outputStore} from {args.outputBucket}: {command_return.stdout}")

        # Enable conversion to NetCDF when the cube is created
        # Convert Zarr to NetCDF and copy to the bucket
        # nc_filename = args.outputStore.replace('.zarr', '.nc')
        # zarr_to_netcdf.main(args.outputStore, nc_filename, ITSCube.NC_ENGINE)
        # ITSCube.show_memory_usage('after Zarr to NetCDF conversion')
        for each_input, each_output, recursive_option in zip(
            # [nc_filename, args.outputStore],
            # [nc_filename, args.outputStore],
            # [None,  "--recursive"]
            [args.outputStore],
            [args.outputStore],
            ["--recursive"]
            ):
            if recursive_option is not None:
                command_line = [
                    "aws", "s3", "cp", recursive_option,
                    each_input,
                    os.path.join(args.outputBucket, os.path.basename(each_output)),
                    "--acl", "bucket-owner-full-control"
                ]

            else:
                command_line = [
                    "aws", "s3", "cp",
                    each_input,
                    os.path.join(args.outputBucket, os.path.basename(each_output)),
                    "--acl", "bucket-owner-full-control"
                ]

            logging.info(' '.join(command_line))

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to copy {each_input} to {args.outputBucket}: {command_return.stdout}")

            if not args.disableCubeValidation:
                s3_datacube = os.path.join(args.outputBucket, os.path.basename(args.outputStore))
                logging.info(f"Opening {s3_datacube} for validation")

                # Validate copied to S3 datacube
                s3_in = s3fs.S3FileSystem(anon=True)
                cube_store = s3fs.S3Map(root=s3_datacube, s3=s3_in, check=False)
                with xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
                    ITSCube.validate_cube_datetime(ds, args.searchAPIStartDate, s3_datacube)

                gc.collect()


        # Remove locally written Zarr store if target location is AWS S3 bucket.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        logging.info(f"Removing local copy of {args.outputStore}")
        shutil.rmtree(args.outputStore)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')
    logging.info(f"Done.")
