"""
ITSCube class creates ITS_LIVE datacube based on target projection,
bounding polygon and datetime period provided by the caller.

Authors: Masha Liukis, Alex Gardner, Mark Fahnestock
"""
import copy
from dateutil.parser import parse
from datetime import datetime, timedelta
import gc
import geopandas as gpd
import glob
import json
import logging
import os
from pathlib import Path
import psutil
import pyproj
import shutil
import time
import timeit
import zarr
import dask
# from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import rioxarray
import s3fs
import subprocess
from tqdm import tqdm
import xarray as xr
from urllib.parse import urlparse

# Local modules
import itslive_utils
from grid import Bounds, Grid
from itscube_types import \
    Coords, \
    DataVars, \
    BinaryFlag, \
    FileExtension, \
    Output, \
    CubeOutput, \
    ShapeFile, \
    to_int_type
import zarr_to_netcdf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    HTTP_PREFIX = 'https://'

    # Token within granule's URL that needs to be removed to get file location within S3 bucket:
    # if URL is of the 'http://its-live-data.s3.amazonaws.com/velocity_image_pair/landsat/v00.0/32628/file.nc' format,
    # S3 bucket location of the file is 's3://its-live-data/velocity_image_pair/landsat/v00.0/32628/file.nc'
    PATH_URL = ".s3.amazonaws.com"
    SHAPE_PATH_URL = '.s3.amazonaws.com'

    # For testing Malaspina cube with latest updates to granules - using file of granules
    # to use instead of queueing searchAPI
    # PATH_URL = '.s3.us-west-2.amazonaws.com'

    # URL path to the target datacube
    URL = ''

    # S3 path to the target datacube
    S3 = ''

    # Local path to the skipped granules info
    SKIPPED_GRANULES_FILE = ''

    # Engine to read xarray data into from NetCDF filecompression
    NC_ENGINE = 'h5netcdf'

    # Date format as it appears in granules filenames:
    # (LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc)
    DATE_FORMAT = "%Y%m%d"

    # Date and time format for acquisition dates of img_info_pair
    DATE_TIME_NO_MICROSECS_FORMAT = '%Y%m%dT%H:%M:%S'
    DATE_TIME_FORMAT = '%Y%m%dT%H:%M:%S.%f'

    # Granules are written to the file in chunks to avoid out of memory issues.
    # Number of granules to write to the file at a time.
    NUM_GRANULES_TO_WRITE = 1000

    # Grid cell size for the datacube.
    CELL_SIZE = 240.0

    CHIP_SIZE_HEIGHT_NO_VALUE = 65535

    # Number of 'aws s3 cp' retries in case of a failure
    NUM_AWS_COPY_RETRIES = 50

    # Number of seconds to sleep between 'aws s3 cp' retries
    AWS_COPY_SLEEP_SECONDS = 60

    # Chunking to apply when writing datacube to the Zarr store
    TIME_CHUNK_VALUE = 20000
    X_Y_CHUNK_VALUE = 10

    # Chunking to apply to 1d data variables when writing datacube to the Zarr store
    TIME_CHUNK_VALUE_1D = 200000

    # ATTN: Character arrays size must be explicitely set before first write:
    # to avoid truncation of the data if first ever written block of data
    # has less than other blocks data in length.

    # Maximum length for the sensor value across all used missions
    MAX_SENSOR_LEN = 2

    # Maximum length of the granule URL
    MAX_GRANULE_URL_LEN = 1024

    # Landsat8 filename prefixes to use when we need to remove duplicate
    # reprocessed granules for Landsat8/9
    # Per Mark comments on Slack:
    # "Should keep both prefixes for L9, but there may not be any ‘LO09’ images -
    # the O means only optical bands were acquired for that frame (no
    # thermal bands), the ‘LC’ means both optical and thermal were acquired.
    # We don’t care about thermal, but we have to deal with the file names USGS
    # uses."
    LANDSAT89_PREFIX = tuple(['LC08', 'LO08', 'LC09', 'LO09'])

    # Token to split image pair filename into two image names
    SPLIT_IMAGES_TOKEN = '_X_'
    IMAGE_TOKEN = '_'

    # An error generated by AWS when PUT request rate exceeds 3500
    AWS_SLOW_DOWN_ERROR = "An error occurred (SlowDown) when calling the PutObject operation"

    # If a list of granules to generate datacube from is provided through input
    # JSON file.
    USE_GRANULES = None

    # Shape file to locate ice masks files that correspond to the composite's EPSG code
    SHAPE_FILE = None

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
        self.polygon = polygon

        # All layers are required to have the same autoRIFT parameter file
        self.autoRIFTParamFile = None

        # Set min/max x/y values to filter region by
        x = Bounds([each[0] for each in polygon])
        y = Bounds([each[1] for each in polygon])

        # Grid for the datacube based on its bounding polygon
        self.grid_x, self.grid_y = Grid.create(x, y, ITSCube.CELL_SIZE)

        self.x_cell = self.grid_x[1] - self.grid_x[0]
        self.y_cell = self.grid_y[1] - self.grid_y[0]

        # Grid cell half size
        self.half_x_cell = self.x_cell/2.0
        self.half_y_cell = self.y_cell/2.0

        abs_x_size = np.abs(self.half_x_cell)
        abs_y_size = np.abs(self.half_y_cell)

        # Define range for x and y based on grid edges
        self.grid_x_min = self.grid_x.min() - abs_x_size
        self.grid_x_max = self.grid_x.max() + abs_x_size

        self.grid_y_min = self.grid_y.min() - abs_y_size
        self.grid_y_max = self.grid_y.max() + abs_y_size

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
            self.polygon_coords.append(list(coords))

        self.logger.info(f"Polygon's longitude/latitude coordinates: {self.polygon_coords}")

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle dates (+ date separation in days as milliseconds),
        # original granules URLs)
        self.ds = []

        self.dates = []
        self.urls = []
        self.num_urls_from_api = None

        # Keep track of skipped granules due to:
        # * no data coverage for the cube
        # * other than target projection
        # * duplicate middle date
        self.skipped_granules = {
            DataVars.SKIP_EMPTY_DATA: [],
            DataVars.SKIP_DUPLICATE: [],
            DataVars.SKIP_PROJECTION: {}
        }
        # # Keep track of skipped granules due to no data for the polygon of interest
        # self.skipped_empty_granules = []
        # # Keep track of "double" granules with older processing date which are
        # # not included into the cube
        # self.skipped_double_granules = []

        # Constructed cube
        self.layers = None

        # Dates when datacube was created or updated
        self.date_created = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        self.date_updated = None

        # Number of layers for cube generation based on the searchAPI query return
        self.max_number_of_layers = 0

        # Find corresponding to EPSG land ice mask file for the cube
        found_row = ITSCube.SHAPE_FILE.loc[ITSCube.SHAPE_FILE[ShapeFile.EPSG] == int(projection)]
        if len(found_row) != 1:
            raise RuntimeError(f'Expected one entry for {projection} in shapefile, got {len(found_row)} rows.')

        # Land ice mask for the cube
        self.land_ice_mask, self.land_ice_mask_url = ITSCube.read_ice_mask(
            found_row, ShapeFile.LANDICE, self.grid_x, self.grid_y
        )

        # Floating ice coverage for the datacube
        self.floating_ice_mask, self.floating_ice_mask_url = ITSCube.read_ice_mask(
            found_row, ShapeFile.FLOATINGICE, self.grid_x, self.grid_y
        )

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
        # Keep track of skipped granules due to:
        # * no data coverage for the cube
        # * other than target projection
        # * duplicate middle date
        self.skipped_granules = {
            DataVars.SKIP_EMPTY_DATA: [],
            DataVars.SKIP_DUPLICATE: [],
            DataVars.SKIP_PROJECTION: {}
        }

    def request_granules(self, api_params: dict, num_granules: int):
        """
        Send request to ITS_LIVE API to get a list of granules to satisfy polygon request.
        Or instead the testing purposes use a list of provided granules through input
        JSON file.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        if ITSCube.USE_GRANULES is not None:
            found_urls = ITSCube.USE_GRANULES

            if num_granules:
                # found_urls = [each['url'] for each in ITSCube.USE_GRANULES][:num_granules]
                found_urls = ITSCube.USE_GRANULES[:num_granules]

                # # Pick S1 or S2 granules to test
                # sentinel_granules = [each for each in ITSCube.USE_GRANULES if os.path.basename(each)[0] == 'S']
                # found_urls.extend(sentinel_granules[:num_granules])

                self.logger.info(f"Examining only first {len(found_urls)} out of {len(ITSCube.USE_GRANULES)} provided granules")

            self.max_number_of_layers = len(found_urls)
            return found_urls

        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for sublist in self.polygon_coords for each in sublist])

        self.logger.info(f"ITS_LIVE search API params: {params}")
        start_time = timeit.default_timer()
        # found_urls = [each['url'] for each in itslive_utils.get_granule_urls_streamed(params, total_retries=10, num_seconds=45)]
        found_urls = [each['url'] for each in itslive_utils.get_granule_urls_compressed(params, total_retries=10, num_seconds=45)]

        # Beware that entries in 'found_urls' list are not always returned in
        # the same order as in previous query. This might result in excluding only
        # some of existing datacube layers from 'found_urls' when trying to determine
        # new granules to consider to add to the cube during the update.
        total_num = len(found_urls)
        time_delta = timeit.default_timer() - start_time
        self.logger.info(f"Number of found by API granules: {total_num} (took {time_delta} seconds)")

        if len(found_urls) == 0:
            self.logger.info(f"No granules are found for the search API parameters: {params}, "
                             "skipping datacube generation or update")
            return found_urls

        self.max_number_of_layers = len(found_urls)

        # Number of granules to examine is specified
        # ATTN: just a way to limit number of granules to be considered for the
        #       datacube generation (testing or debugging only).
        if num_granules:
            found_urls = found_urls[:num_granules]
            self.logger.info(f"Examining only first {len(found_urls)} out of {total_num} found granules")

        # Number of found URL's should report number of granules as returned by
        # searchAPI to provide correct % value for skipped granules if updating the cube
        self.num_urls_from_api = len(found_urls)

        urls, self.skipped_granules[DataVars.SKIP_DUPLICATE] = ITSCube.skip_duplicate_l89_granules(found_urls)

        return urls

    @staticmethod
    def skip_duplicate_l89_granules(found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s))
        for the same path row granule for Landsat8 and Landsat9 data only.

        Examples of the Landsat image pair filename with one of the images from L89 mission group:
        LC08_L1GT_007011_20130819_20200912_02_T2_X_LC08_L1GT_007011_20140806_20200911_02_T2_G0120V02_P044.nc
        LC08_L1TP_013010_20130330_20200913_02_T1_X_LE07_L1TP_012010_20130627_20200907_02_T1_G0120V02_P003.nc
        """
        # Need to remove duplicate granules for the middle date: some granules
        # have newer processing date, keep those.
        keep_urls = {}
        skipped_double_granules = []

        # Unique granules to return
        granules = []

        # Get image pairs with at least one of the Landsat8/9 images
        landsat89_granules = [
            each for each in found_urls
            if os.path.basename(each).split(ITSCube.SPLIT_IMAGES_TOKEN)[0].startswith(ITSCube.LANDSAT89_PREFIX) or
            os.path.basename(each).split(ITSCube.SPLIT_IMAGES_TOKEN)[1].startswith(ITSCube.LANDSAT89_PREFIX)
        ]

        if len(landsat89_granules) == 0:
            # There are no Landsat8 granules, no need to remove duplicates
            return found_urls, skipped_double_granules

        else:
            # Include non-Landsat89 granules into unique granules to return
            # as they don't need to be searched for duplicates
            granules = list(set(found_urls).difference(landsat89_granules))
            logging.info(f'Number of non-Landsat89 granules: {len(granules)}')

        for each_url in tqdm(landsat89_granules, ascii=True, desc=f'Skipping duplicate Landsat89 granules out of {len(landsat89_granules)} granules...'):
            # Extract acquisition and processing dates
            url_proc_1, url_proc_2, granule_id = \
                ITSCube.get_tokens_from_filename(each_url)
            # logging.info(f'ID={granule_id} for granule={each_url}')

            # There is a granule for the mid_date already, check which processing
            # time is newer, keep the one with newer processing date
            if granule_id in keep_urls:
                # Flag if newly found URL should be kept
                keep_found_url = False

                for found_url in keep_urls[granule_id]:
                    # Check already found URLs for processing time
                    found_proc_1, found_proc_2, found_granule_id = \
                        ITSCube.get_tokens_from_filename(found_url)

                    # IDs must match
                    if granule_id != found_granule_id:
                        raise RuntimeError(f'Mismatching IDs for each_url={each_url}: {granule_id} vs. found_url={found_url}: {found_granule_id}')

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
                        found_proc_1, found_proc_2, _ = \
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
                        logging.info(f"Skipping {remove_urls} in favor of new {each_url}")
                        skipped_double_granules.extend(remove_urls)

                        # Remove older processed granules
                        keep_urls[granule_id][:] = [each for each in keep_urls[granule_id] if each not in remove_urls]
                        # Add new granule with newer processing date
                        keep_urls[granule_id].append(each_url)

                    else:
                        # New granule has older processing date, don't include
                        logging.info(f"Skipping new {each_url} in favor of {keep_urls[granule_id]}")
                        skipped_double_granules.append(each_url)

            else:
                # This is a granule for new ID, append it to URLs to keep
                keep_urls.setdefault(granule_id, []).append(each_url)

        for each in keep_urls.values():
            granules.extend(each)

        logging.info(f'Keeping {len(granules)} unique granules, skipping {len(skipped_double_granules)} Landsat89 granules')

        return granules, skipped_double_granules

    def exclude_processed_granules(self, found_urls: list, cube_ds: xr.Dataset, skipped_granules: dict):
        """
        * Exclude datacube granules, and all skipped granules in existing datacube
        (empty data, wrong projection, duplicate middle date) from found granules.
        * Identify if any of the skipped double mid_date granules from "found_urls"
        are already existing layers in the datacube. Need to mark such layers
        to be deleted from the datacube.
        * Identify if current cube layers and remaining found_urls have duplicate
        mid_date - register these for deletion from the datacube if they appear
        as datacube layers.

        as datacube layers.
        Return:
            granules: list
                List of granules to update datacube with.
            layers_to_delete: list
                List of existing datacube layers to remove.
        """
        self.logger.info("Excluding known to datacube granules...")
        self.logger.info(f"Got {len(found_urls)} total granules to consider ({len(set(found_urls))} unique granules)...")
        cube_granules = cube_ds[DataVars.URL].values.tolist()
        self.logger.info(f"Existing datacube granules: {len(cube_granules)} ({len(set(cube_granules))} unique granules)")
        granules = set(found_urls).difference(cube_granules)
        cube_in_found_urls = set(cube_granules).difference(found_urls)
        self.logger.info(f"Cube granules not in found_urls: ({len(cube_in_found_urls)})")
        self.logger.info(f"Removed known cube granules ({len(cube_granules)}): {len(granules)} granules remain")

        # Remove known empty granules (per cube) from found_urls
        self.skipped_granules[DataVars.SKIP_EMPTY_DATA] = skipped_granules[DataVars.SKIP_EMPTY_DATA]
        granules = granules.difference(self.skipped_granules[DataVars.SKIP_EMPTY_DATA])
        self.logger.info(f"Removed known empty data granules ({len(self.skipped_granules[DataVars.SKIP_EMPTY_DATA])}): {len(granules)} granules remain")

        # Remove known wrong projection granules (per projection) from found_urls
        # ATTN: int values get written as strings to json files, so make sure read back in values
        #       for the keys are of int type
        for each_key, each_value in skipped_granules[DataVars.SKIP_PROJECTION].items():
            self.skipped_granules[DataVars.SKIP_PROJECTION][int(each_key)] = each_value

        known_granules = []
        for each in self.skipped_granules[DataVars.SKIP_PROJECTION].values():
            known_granules.extend(each)

        granules = granules.difference(known_granules)
        self.logger.info(f"Removed known wrong projection granules ({len(known_granules)}): {len(granules)} granules remain")

        # Identify if there are any cube granules that now need to be skipped
        # due to double middle date in "new" found_urls granules
        # (self.skipped_granules[DataVars.SKIP_DUPLICATE] is set by self.request_granules())
        cube_layers_to_delete = list(set(self.skipped_granules[DataVars.SKIP_DUPLICATE]).intersection(cube_granules))
        self.logger.info(f"{len(cube_layers_to_delete)} "
                         f"existing datacube layers to delete due to duplicate mid_date: {cube_layers_to_delete}")

        # Remove known duplicate middle date granules from found_urls:
        # if cube's skipped granules don't appear in found_urls.skipped_granules
        # for whatever reason (different start/end dates are used for cube update)
        # self.skipped_granules[DataVars.SKIP_DUPLICATE] is populated by self.request_granules()
        # with skipped granules due to double date in "found_urls"
        cube_skipped_double_granules = skipped_granules[DataVars.SKIP_DUPLICATE]
        granules = granules.difference(cube_skipped_double_granules)
        self.logger.info(f"Removed known cube's duplicate middle date granules ({len(cube_skipped_double_granules)}): {len(granules)} granules remain")

        # Check if there are any granules between existing cube layers and found_urls
        # that have duplicate middle date
        cube_and_found_urls = cube_granules + list(granules)
        _, skipped_landsat_granules = ITSCube.skip_duplicate_l89_granules(cube_and_found_urls)

        # Check if any of the skipped granules are in the cube
        cube_layers_to_delete.extend(list(set(cube_granules).intersection(skipped_landsat_granules)))
        self.logger.info(f"After (cube_granules+found_urls): total of {len(cube_layers_to_delete)} "
                         f"existing datacube layers to delete due to duplicate mid_date: {cube_layers_to_delete}")

        # Merge two lists of skipped granules (for existing cube, new list
        # of granules from search API, and duplicate granules b/w cube and new granules)
        cube_skipped_double_granules.extend(self.skipped_granules[DataVars.SKIP_DUPLICATE])
        cube_skipped_double_granules.extend(skipped_landsat_granules)
        self.skipped_granules[DataVars.SKIP_DUPLICATE] = list(set(cube_skipped_double_granules))

        # Skim down found_urls by newly skipped granules
        granules = list(granules.difference(self.skipped_granules[DataVars.SKIP_DUPLICATE]))
        self.logger.info(f"Leaving {len(granules)} granules...")

        return granules, cube_layers_to_delete

    @staticmethod
    def get_tokens_from_filename(filename):
        """
        Extract processing dates for two images from the filename and construct unique
        identifier for the image pair by removing processing dates, percent valid
        pixels fields and file extension.
        """
        files = os.path.basename(filename).split(ITSCube.SPLIT_IMAGES_TOKEN)

        # Get acquisition, processing date, path_row for both images from url and index_url
        url_tokens = os.path.basename(files[0]).split(ITSCube.IMAGE_TOKEN)

        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)

        # Remove processing date from the first image name: don't replace date
        # token with an empty string as acquisition and processing dates can be
        # the same
        id_tokens = url_tokens[:4]
        id_tokens.extend(url_tokens[5:])

        url_tokens = os.path.basename(files[1]).split(ITSCube.IMAGE_TOKEN)
        url_proc_date_2 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)

        # Remove processing date and _Pxxx.nc from the second image name
        id_tokens.extend(url_tokens[:4])
        id_tokens.extend(url_tokens[5:8])

        id = ITSCube.IMAGE_TOKEN.join(id_tokens)

        return url_proc_date_1, url_proc_date_2, id

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
                self.skipped_granules[DataVars.SKIP_EMPTY_DATA].append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_granules[DataVars.SKIP_PROJECTION].setdefault(layer_projection, []).append(url)

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
            s3 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
            cube_glob = s3.glob(cube_path)
            if len(cube_glob):
                cube_exists = True

        else:
            if os.path.exists(output_dir):
                cube_exists = True

        return cube_exists

    @staticmethod
    def init_input_store(input_dir: str, s3_bucket: str, read_skipped_granules: bool = True):
        """
        Read datacube from provided store. The method detects if S3 bucket
        store or local Zarr archive is provided, and reads xarray.Dataset from
        the Zarr store.
        """
        ds_from_zarr = None
        s3_in = None
        cube_store = None
        skipped_granules = None

        if len(s3_bucket) == 0:
            # If reading from the local directory, check if datacube store exists
            if ITSCube.exists(input_dir, s3_bucket):
                logging.info(f"Reading existing {input_dir}...")

                # Read dataset in
                ds_from_zarr = xr.open_zarr(input_dir, decode_timedelta=False, consolidated=True)

                # Read skipped granules info that corresponds to the cube
                if read_skipped_granules:
                    logging.info(f"Reading existing {ds_from_zarr.attrs[DataVars.SKIPPED_GRANULES]}...")
                    with open(ds_from_zarr.attrs[DataVars.SKIPPED_GRANULES]) as skipped_fh:
                        skipped_granules = json.load(skipped_fh)

        elif ITSCube.exists(input_dir, s3_bucket):
            # When datacube is in the AWS S3 bucket, check if it exists.
            cube_path = os.path.join(s3_bucket, input_dir)
            logging.info(f"Reading existing {cube_path}")

            # Open S3FS access to S3 bucket with input datacube
            s3_in = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
            cube_store = s3fs.S3Map(root=cube_path, s3=s3_in, check=False)
            ds_from_zarr = xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True)

            if read_skipped_granules:
                logging.info(f"Reading existing {ds_from_zarr.attrs[DataVars.SKIPPED_GRANULES]}")
                with s3_in.open(ds_from_zarr.attrs[DataVars.SKIPPED_GRANULES], 'r') as skipped_fh:
                    skipped_granules = json.load(skipped_fh)

        if ds_from_zarr is None:
            raise RuntimeError(f"Provided input datacube {input_dir} does not exist (s3={s3_bucket})")

        # Don't use cube_store - keep it in scope only to guarantee valid
        # file-like access.
        return s3_in, cube_store, ds_from_zarr, skipped_granules

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
        s3, cube_store_in, cube_ds, skipped_granules = ITSCube.init_input_store(output_dir, output_bucket)

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

        # Remove already processed granules
        found_urls, cube_layers_to_delete = self.exclude_processed_granules(found_urls, cube_ds, skipped_granules)
        num_cube_layers = len(cube_ds.mid_date.values)

        if len(found_urls) == 0:
            self.logger.info("No granules to update with, exiting.")
            return found_urls

        # Clean up the open store for the dataset
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
                "awsv2", "s3", "cp", "--recursive",
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
                raise RuntimeError(f"Failed to copy {source_url} to {output_dir}: {command_return.stdout}")

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
                dropped_ds = None
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
                if len(each_ds[0]):
                    # There were exceptions reading the data, log it
                    self.logging.info('--->'.join(each_ds[0]))

                self.add_layer(*each_ds[1:])

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
                if len(each_ds[0]):
                    # There were exceptions reading the data, log it
                    self.logger.info('--->'.join(each_ds[0]))

                self.add_layer(*each_ds[1:])

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
                    ITSCube.show_memory_usage(f'after reading {s3_path}')

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
        found_urls = ITSCube.skip_duplicate_l89_granules(found_urls)
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

        found_urls = ITSCube.skip_duplicate_l89_granules(found_urls)
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

    def get_data_var(self, ds: xr.Dataset, var_name: str, data_dtype: str = 'short', data_fill_value: int = DataVars.MISSING_VALUE):
        """
        Return xr.DataArray that corresponds to the data variable if it exists
        in the 'ds' dataset, or empty xr.DataArray if it is not present in the 'ds'.
        If requested datatype for output data is not of data's original type, convert data
        Empty xr.DataArray assumes the same dimensions as ds.v data array.
        """
        if var_name in ds:
            if data_dtype and ds[var_name].dtype != np.dtype(data_dtype):
                # Return data of requested type with corresponding "missing_value"
                return xr.DataArray(
                    data=to_int_type(ds[var_name].values, data_type=np.dtype(data_dtype), fill_value=data_fill_value),
                    coords=ds[var_name].coords,
                    dims=ds[var_name].dims,
                    attrs=ds[var_name].attrs
                )

            return ds[var_name]

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: Can't use None as data to create xr.DataArray - won't be able
        # to set dtype='short' in encoding for writing to the file.
        return xr.DataArray(
            data=np.full((len(self.grid_y), len(self.grid_x)), data_fill_value, dtype=np.dtype(data_dtype)),
            coords=[self.grid_y, self.grid_x],
            dims=[Coords.Y, Coords.X]
        )

    @staticmethod
    def get_data_var_attr(
        ds: xr.Dataset,
        ds_url: str,
        var_name: str,
        attr_name: str,
        missing_value=None,
        to_date=False,
        data_dtype=np.float32
    ):
        """
        Return attribute for the data variable in data set if it exists,
        or missing_value if it is not present.
        If "missing_value" is set to None, than specified attribute is expected
        to exist for the data variable "var_name" and exception is raised if
        it does not.

        Inputs:
        =======
        ds:            xarray.Dataset the variable belongs to.
        ds_url:        URL of the granule that corresponds to the "ds" dataset (used for
                       error reporting only).
        var_name:      Name of the variable to extract attribute for.
        attr_name:     Name of the attribute to extract value for.
        missing_value: Value to use if attribute is missing for the variable.
                       Default is None, which will result in raising an exception
                       if attribute is missing for the variable.
        to_date:       Flag if attribute value should be converted to datetime object.
                       Default is False.
        data_dtype:    Datatype to use for the attribute value. Default is np.float32.
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
                        value = tokens[0] + 'T' + tokens[1][0:2] + ':' + tokens[1][2:4] + ':' + tokens[1][4:6]
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                    elif len(value) >= 8:
                        value = parse(value)

                except ValueError as exc:
                    raise RuntimeError(f"Error converting {value} to date format '%Y%m%d': {exc} for {var_name}.{attr_name} in {ds_url}")

            else:
                # Convert value to expected datatype
                if data_dtype:
                    value = data_dtype(value)

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
        mid_date:   Middle date that corresponds to the velocity pair (uses date
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
            mid_date = None

            # Sentinel1 granules contain attributes of old naming convention,
            # check for it
            attr_name_1 = DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1
            attr_name_2 = DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2

            acq1_datetime = parse(ds.img_pair_info.attrs[attr_name_1])
            mid_date = acq1_datetime + (parse(ds.img_pair_info.attrs[attr_name_2]) - acq1_datetime)/2

            # Create unique "token" by using granule's centroid longitude/latitude to
            # increase uniqueness of the mid_date for the layer (xarray: can't drop layers
            # for the cube with mid_date dimension which contains non-unique values).
            # Add the token as microseconds for the middle date: AAOOO
            #
            # lat = int(np.abs(ds.img_pair_info.latitude))
            # lon = int(np.abs(ds.img_pair_info.longitude))
            # Lon/lat can be non-unique for some of the granules with the same
            # 'date_center', so use acquisition_date_img1 values instead: YYMMDD
            # Example: for acquisition_date_img1 = "20141121T13:31:15" will use
            # "141121" as microseconds
            mid_date += timedelta(microseconds=int(ds.img_pair_info.attrs[attr_name_1][2:8]))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.grid_x_min) & (ds.x <= self.grid_x_max)
            mask_lat = (ds.y >= self.grid_y_min) & (ds.y <= self.grid_y_max)
            mask = (mask_lon & mask_lat)
            if mask.values.sum() == 0:
                # One or both masks resulted in no coverage
                mask_data = None
                mid_date = None
                empty = True

            else:
                mask_data = ds.where(mask, drop=True)

                # Another way to filter (have to put min/max values in the order
                # corresponding to the grid)
                # cube_v = ds.v.sel(x=slice(self.grid_x.min(), self.grid_x.max()),
                #                   y=slice(self.grid_y.max, self.grid_y.min)).copy()

                # If it's a valid velocity layer, add it to the cube,
                # and skip granules that have only one cell in cube's polygon
                if np.any(mask_data.v.notnull()) and \
                   len(mask_data.x.values) > 1 and len(mask_data.y.values > 1):
                    mask_data = mask_data.load()

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
        Helper method to collect attributes for v-related data variables.

        Inputs:
        =======
        var_name: Name of the variable (vx, vy, va, vr).
        mid_date_coord: Middle date coordinate for collected data.

        Returns:
        =======
        List of new data variables names that were created to correspond to
        the attributes of "var_name" variable. These variables names are used
        to set their encoding parameters when storing datacube to the Zarr store.
        """
        # Dictionary of attributes values for new v*_error data variables:
        # std_name, description
        _attrs = {
            'vx_error': ("x_velocity_error", "error for velocity component in x direction"),
            'vy_error': ("y_velocity_error", "error for velocity component in y direction"),
            'va_error': ("azimuth_velocity_error", "error for velocity in radar azimuth direction"),
            'vr_error': ("range_velocity_error", "error for velocity in radar range direction"),
            # The following descriptions are the same for all v* data variables
            'error_stationary': (None, "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 m/yr identified from an external mask"),
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
        # If attribute is propagated as cube's data var attribute, delete it.
        _name_sep = '_'

        for each_attr in _v_comp_attrs:
            error_name = f'{var_name}{_name_sep}{each_attr}'
            return_vars.append(error_name)

            # Special care must be taken of v[xy].stable_rmse in
            # optical legacy format vs. v[xy].v[xy]_error in radar format as these
            # are the same
            error_data = [ITSCube.get_data_var_attr(ds, url, var_name, each_attr, DataVars.MISSING_VALUE)
                          for ds, url in zip(self.ds, self.urls)]

            error_name_desc = f'{each_attr}{_name_sep}{DataVars.ERROR_DESCRIPTION}'
            desc_str = None
            if var_name in self.ds[0] and error_name_desc in self.ds[0][var_name].attrs:
                desc_str = self.ds[0][var_name].attrs[error_name_desc]

            elif each_attr in _attrs:
                # If generic description is provided
                desc_str = _attrs[each_attr][1]

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
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

            # If attribute description is in the var's attributes, remove it
            if error_name_desc in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[error_name_desc]

        # These attributes appear for all v* data variables of the granule,
        # capture it only once if it exists
        for each_attr, each_attr_units in zip(
            [DataVars.FLAG_STABLE_SHIFT, DataVars.STABLE_COUNT_MASK, DataVars.STABLE_COUNT_SLOW],
            [None, DataVars.COUNT_UNITS, DataVars.COUNT_UNITS]
        ):
            if var_name in self.ds[0] and \
               each_attr not in self.layers and \
               each_attr in self.ds[0][var_name].attrs:
                self.layers[each_attr] = xr.DataArray(
                    data=[ITSCube.get_data_var_attr(ds, url, var_name, each_attr, data_dtype=np.int32)
                          for ds, url in zip(self.ds, self.urls)],
                    coords=[mid_date_coord],
                    dims=[Coords.MID_DATE],
                    attrs={
                        DataVars.STD_NAME: each_attr,
                        DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[each_attr]
                    }
                )

                # Set units if appropriate
                if each_attr_units is not None:
                    self.layers[each_attr].attrs[DataVars.UNITS] = each_attr_units

            # Remove attribute if it made it into datacube as original variable attribute
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

        if DataVars.FLAG_STABLE_SHIFT_DESCRIPTION in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.FLAG_STABLE_SHIFT_DESCRIPTION]

        # Create 'stable_shift' specific to the data variable,
        # for example, 'vx_stable_shift' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT])
        stable_shift_values = np.array(
            [
                ITSCube.get_data_var_attr(
                    ds,
                    url,
                    var_name,
                    DataVars.STABLE_SHIFT,
                    DataVars.MISSING_VALUE
                )
                for ds, url in zip(self.ds, self.urls)
            ]
        )

        # Some of the granules have "stable_shift" attribute set to NaN:
        # set them to zero
        nan_stable_shift_values_mask = np.isnan(stable_shift_values)

        if np.sum(nan_stable_shift_values_mask) > 0:
            self.logger.info(f'Setting {np.sum(nan_stable_shift_values_mask)} stable_shift values to 0 for {var_name}')
            stable_shift_values[nan_stable_shift_values_mask] = 0

        self.layers[shift_var_name] = xr.DataArray(
            data=stable_shift_values,
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: shift_var_name,
                DataVars.DESCRIPTION_ATTR: f'applied {var_name} shift calibrated using pixels over stable or slow surfaces'
            }
        )
        return_vars.append(shift_var_name)

        stable_shift_values = None
        gc.collect()

        if DataVars.STABLE_SHIFT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT]

        # Create 'stable_shift_mask' and 'stable_shift_slow' specific to the data variable
        # (for example, 'vx_stable_shift_mask' for 'vx' data variable).
        for each_attr in [DataVars.STABLE_SHIFT_MASK, DataVars.STABLE_SHIFT_SLOW]:
            shift_var_name = _name_sep.join([var_name, each_attr])
            self.layers[shift_var_name] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(ds, url, var_name, each_attr, DataVars.MISSING_VALUE)
                      for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE],
                attrs={
                    DataVars.UNITS: DataVars.M_Y_UNITS,
                    DataVars.STD_NAME: shift_var_name,
                    DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[each_attr].format(var_name)
                }
            )
            return_vars.append(shift_var_name)

            # If attribute is propagated as cube's vx attribute, delete it
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

        # Return names of new data variables - to be included into "encoding" settings
        # for writing to the file store.
        return return_vars

    def process_m_attributes(self, var_name: str, mid_date_coord):
        """
        Helper method to clean up attributes for M1[12]-related data variables.
        """
        # Process attributes
        # If attribute is propagated as cube's data var attribute, delete it.
        _name_sep = '_'

        # Need to create new DR_TO_VR_FACTOR data variable
        attr_name = f'{var_name}{_name_sep}{DataVars.DR_TO_VR_FACTOR}'

        attr_data = [ITSCube.get_data_var_attr(ds, url, var_name, attr_name, DataVars.MISSING_BYTE)
                     for ds, url in zip(self.ds, self.urls)]

        self.layers[attr_name] = xr.DataArray(
            data=attr_data,
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.STD_NAME: attr_name,
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.DR_TO_VR_FACTOR],
                DataVars.UNITS: DataVars.M_PER_YEAR_PIXEL
            }
        )

        # Remove attributes from the "parent" variable
        if DataVars.DR_TO_VR_FACTOR in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.DR_TO_VR_FACTOR]

        if DataVars.DR_TO_VR_FACTOR_DESCRIPTION in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.DR_TO_VR_FACTOR_DESCRIPTION]

        # Remove scale_factor and offset that come with original M11 and M12 data
        # if any
        if Output.SCALE_FACTOR in self.layers[var_name].encoding:
            del self.layers[var_name].encoding[Output.SCALE_FACTOR]

        if Output.ADD_OFFSET in self.layers[var_name].encoding:
            del self.layers[var_name].encoding[Output.ADD_OFFSET]

        # Return name of new data variable - to be included into "encoding" settings
        # for writing to the file store.
        return attr_name

    def set_grid_mapping_attr(self, var_name: str, ds_grid_mapping_value: str):
        """
        Check on existence of "grid_mapping" attribute for the variable, set it
        if not present.
        """
        if DataVars.GRID_MAPPING in self.layers[var_name].attrs:
            # Attribute is already set, nothing to do
            return

        self.layers[var_name].attrs[DataVars.GRID_MAPPING] = ds_grid_mapping_value

    @staticmethod
    def show_memory_usage(msg: str = ''):
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

        # Write skipped granules info to local file
        with open(ITSCube.SKIPPED_GRANULES_FILE, 'w') as fh:
            json.dump(self.skipped_granules, fh, indent=3)

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
            data_vars={DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords={
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
            attrs={
                CubeOutput.AUTHOR: CubeOutput.Values.AUTHOR
            }
        )

        # Set datacube attribute to capture autoRIFT parameter file
        self.layers.attrs[DataVars.AUTORIFT_PARAMETER_FILE] = self.ds[0].attrs[DataVars.AUTORIFT_PARAMETER_FILE]

        if self.autoRIFTParamFile is None:
            self.autoRIFTParamFile = self.layers.attrs[DataVars.AUTORIFT_PARAMETER_FILE]

        # Make sure all layers have the same parameter file
        all_values = [urlparse(ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE]).path for ds in self.ds]
        unique_values = list(set(all_values))
        if len(unique_values) > 1:
            raise RuntimeError(f"Multiple values for '{DataVars.AUTORIFT_PARAMETER_FILE}' are detected for current {len(self.ds)} layers: {unique_values}")

        # All layers within datacube must have the same autoRIFT parameter file
        if self.autoRIFTParamFile != self.layers.attrs[DataVars.AUTORIFT_PARAMETER_FILE]:
            raise RuntimeError(f"Inconsistent values for '{DataVars.AUTORIFT_PARAMETER_FILE}' are detected: {self.layers.attrs[DataVars.AUTORIFT_PARAMETER_FILE]} for current {len(self.ds)} layers vs. previously detected {self.autoRIFTParamFile}")

        self.layers.attrs[CubeOutput.CONVENTIONS] = CubeOutput.Values.CONVENTIONS
        self.layers.attrs[CubeOutput.DATACUBE_SOFTWARE_VERSION] = ITSCube.Version
        self.layers.attrs[CubeOutput.DATE_CREATED] = self.date_created
        self.layers.attrs[CubeOutput.DATE_UPDATED] = self.date_updated if self.date_updated is not None else self.date_created
        self.layers.attrs[CubeOutput.GDAL_AREA_OR_POINT] = CubeOutput.Values.AREA
        self.layers.attrs[CubeOutput.GEO_POLYGON] = json.dumps(self.polygon_coords)
        self.layers.attrs[CubeOutput.INSTITUTION] = CubeOutput.Values.INSTITUTION
        self.layers.attrs[CubeOutput.LATITUDE] = round(self.center_lon_lat[1], 2)
        self.layers.attrs[CubeOutput.LONGITUDE] = round(self.center_lon_lat[0], 2)
        self.layers.attrs[CubeOutput.PROJ_POLYGON] = json.dumps(self.polygon)
        self.layers.attrs[CubeOutput.PROJECTION] = str(self.projection)
        self.layers.attrs[CubeOutput.S3] = ITSCube.S3

        # Store path to the file with skipped granules (the ones that didn't
        # qualify to make it into the datacube)
        if len(ITSCube.S3):
            # Result datacube is to be stored in S3 bucket, record S3 location of the
            # skipped granules file
            self.layers.attrs[DataVars.SKIPPED_GRANULES] = ITSCube.S3.replace(FileExtension.ZARR, FileExtension.JSON)

        else:
            # Result datacube is to be stored locally, record location of the
            # skipped granules file
            self.layers.attrs[DataVars.SKIPPED_GRANULES] = output_dir.replace(FileExtension.ZARR, FileExtension.JSON)

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

        self.layers.attrs[CubeOutput.TITLE] = CubeOutput.Values.TITLE
        self.layers.attrs[CubeOutput.URL] = ITSCube.URL

        # Set attributes for 'url' data variable
        self.layers[DataVars.URL].attrs[DataVars.STD_NAME] = DataVars.URL
        self.layers[DataVars.URL].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.URL]

        # Set projection information once for the whole datacube
        if is_first_write:
            # Should never happen - just in case as it's a new data format
            if DataVars.MAPPING not in self.ds[0]:
                raise RuntimeError(f"Missing {DataVars.MAPPING} in {self.urls[0]}")

            # Can't copy the whole data variable, as it introduces obscure coordinates.
            # Just copy all attributes for the scalar type of the xr.DataArray.
            # Use latest granule format: 'mapping' data variable for projection info.
            self.layers[DataVars.MAPPING] = xr.DataArray(
                data='',
                attrs=self.ds[0][DataVars.MAPPING].attrs,
                coords={},
                dims=[]
            )

            # Set GeoTransform to correspond to the datacube's tile:
            # format cube's GeoTransform
            new_geo_transform_str = f"{self.grid_x[0] - self.half_x_cell} {self.x_cell} 0 {self.grid_y[0] - self.half_y_cell} 0 {self.y_cell}"
            self.layers[DataVars.MAPPING].attrs['GeoTransform'] = new_geo_transform_str

            twodim_var_coords = [self.grid_y, self.grid_x]
            twodim_var_dims = [Coords.Y, Coords.X]

            # Create ice masks data variables if they exist
            self.land_ice_mask = to_int_type(
                self.land_ice_mask,
                np.uint8,
                DataVars.MISSING_UINT8_VALUE
            )
            self.layers[ShapeFile.LANDICE] = xr.DataArray(
                data=self.land_ice_mask,
                coords=twodim_var_coords,
                dims=twodim_var_dims,
                attrs={
                    DataVars.STD_NAME: ShapeFile.Name[ShapeFile.LANDICE],
                    DataVars.DESCRIPTION_ATTR: ShapeFile.Description[ShapeFile.LANDICE],
                    DataVars.GRID_MAPPING: DataVars.MAPPING,
                    BinaryFlag.VALUES_ATTR: BinaryFlag.VALUES,
                    BinaryFlag.MEANINGS_ATTR: BinaryFlag.MEANINGS[ShapeFile.LANDICE],
                    CubeOutput.URL: self.land_ice_mask_url
                }
            )
            self.land_ice_mask = None
            gc.collect()

            self.floating_ice_mask = to_int_type(
                self.floating_ice_mask,
                np.uint8,
                DataVars.MISSING_UINT8_VALUE
            )
            # Land ice mask exists for the composite
            self.layers[ShapeFile.FLOATINGICE] = xr.DataArray(
                data=self.floating_ice_mask,
                coords=twodim_var_coords,
                dims=twodim_var_dims,
                attrs={
                    DataVars.STD_NAME: ShapeFile.Name[ShapeFile.FLOATINGICE],
                    DataVars.DESCRIPTION_ATTR: ShapeFile.Description[ShapeFile.FLOATINGICE],
                    DataVars.GRID_MAPPING: DataVars.MAPPING,
                    BinaryFlag.VALUES_ATTR: BinaryFlag.VALUES,
                    BinaryFlag.MEANINGS_ATTR: BinaryFlag.MEANINGS[ShapeFile.FLOATINGICE],
                    CubeOutput.URL: self.floating_ice_mask_url
                }
            )
            self.floating_ice_mask = None
            gc.collect()

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
        grid_mapping_values = [ds.mapping.attrs[DataVars.GRID_MAPPING_NAME] for ds in self.ds]
        unique_values = list(set(grid_mapping_values))
        if len(unique_values) > 1:
            raise RuntimeError(f"Multiple '{DataVars.MAPPING}' values are detected for current {len(self.ds)} layers: {unique_values}")
        # Remember the value as all 3D data variables need to have this attribute
        # set with the same value
        ds_grid_mapping_value = DataVars.MAPPING

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [each.drop_vars(DataVars.V) for each in self.ds]
        del v_layers
        gc.collect()

        # Process 'v_error'
        self.layers[DataVars.V_ERROR] = xr.concat(
            [self.get_data_var(ds, DataVars.V_ERROR) for ds in self.ds],
            mid_date_coord
        )
        self.layers[DataVars.V_ERROR].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.V_ERROR]
        self.layers[DataVars.V_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        new_v_vars.append(DataVars.V_ERROR)

        self.set_grid_mapping_attr(DataVars.V_ERROR, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.V_ERROR) if DataVars.V_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'v[xy]' data variables and their attributes
        for each_var in [DataVars.VX, DataVars.VY]:
            self.layers[each_var] = xr.concat([ds[each_var] for ds in self.ds], mid_date_coord)
            self.layers[each_var].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[each_var]
            new_v_vars.append(each_var)
            new_v_vars.extend(self.process_v_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var, ds_grid_mapping_value)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        # Process 'v[ar]' data variables and their attributes
        for each_var in [DataVars.VA, DataVars.VR]:
            self.layers[each_var] = xr.concat([self.get_data_var(ds, each_var) for ds in self.ds], mid_date_coord)
            self.layers[each_var].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[each_var]
            new_v_vars.append(each_var)
            new_v_vars.extend(self.process_v_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var, ds_grid_mapping_value)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        new_vars_zero_missing_value = []
        # Process 'M1[12]' data variables of radar format, if any, and their attributes
        for each_var in [DataVars.M11, DataVars.M12]:
            self.layers[each_var] = xr.concat([self.get_data_var(ds, each_var) for ds in self.ds], mid_date_coord)
            self.layers[each_var].attrs[DataVars.STD_NAME] = DataVars.NAME[each_var]
            self.layers[each_var].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[each_var]
            self.layers[each_var].attrs[DataVars.UNITS] = DataVars.PIXEL_PER_M_YEAR
            new_v_vars.append(each_var)
            new_vars_zero_missing_value.append(self.process_m_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var, ds_grid_mapping_value)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        # Process chip_size_height: dtype=ushort
        # Optical legacy granules might not have chip_size_height set, use
        # chip_size_width instead
        self.layers[DataVars.CHIP_SIZE_HEIGHT] = xr.concat([
                ds.chip_size_height if
                np.ma.masked_equal(ds.chip_size_height.values, ITSCube.CHIP_SIZE_HEIGHT_NO_VALUE).count() != 0 else
                ds.chip_size_width for ds in self.ds
            ],
            mid_date_coord
        )
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.CHIP_SIZE_COORDS] = \
            DataVars.DESCRIPTION[DataVars.CHIP_SIZE_COORDS]
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.DESCRIPTION_ATTR] = \
            DataVars.DESCRIPTION[DataVars.CHIP_SIZE_HEIGHT]

        self.set_grid_mapping_attr(DataVars.CHIP_SIZE_HEIGHT, ds_grid_mapping_value)

        # Report if used chip_size_width in place of chip_size_height
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
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.STD_NAME] = DataVars.NAME[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.DESCRIPTION_ATTR] = DataVars.DESCRIPTION[DataVars.INTERP_MASK]
        self.layers[DataVars.INTERP_MASK].attrs[BinaryFlag.VALUES_ATTR] = BinaryFlag.VALUES
        self.layers[DataVars.INTERP_MASK].attrs[BinaryFlag.MEANINGS_ATTR] = BinaryFlag.MEANINGS[DataVars.INTERP_MASK]

        self.set_grid_mapping_attr(DataVars.INTERP_MASK, ds_grid_mapping_value)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.INTERP_MASK) for ds in self.ds]
        gc.collect()

        for each in DataVars.ImgPairInfo.ALL:
            # Add new variables that correspond to attributes of 'img_pair_info'
            # (only selected ones)
            each_dtype = None
            if each in DataVars.ImgPairInfo.ALL_DTYPE:
                each_dtype = DataVars.ImgPairInfo.ALL_DTYPE[each]

            self.layers[each] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(
                    ds,
                    url,
                    DataVars.ImgPairInfo.NAME,
                    each,
                    to_date=DataVars.ImgPairInfo.CONVERT_TO_DATE[each],
                    data_dtype=each_dtype
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
            encoding_settings = {}

            # ATTN: Set _FillValue for data variables of floating point data type.
            #       Must set 'missing_value' for data variables on int data type,
            #       otherwise xarray just ignores provided dtype if _FillValue is
            #       provided and assumes floating point type.
            for each in [
                DataVars.ImgPairInfo.DATE_DT,
                DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE
            ]:
                encoding_settings[each] = {
                    Output.DTYPE_ATTR: np.float32
                }

            # Settings for variables of "uint8" data type if any variables exist
            ice_mask_vars = [ShapeFile.LANDICE, ShapeFile.FLOATINGICE]
            for each in ice_mask_vars:
                encoding_settings.setdefault(each, {}).update({
                    Output.DTYPE_ATTR: np.uint8,
                    Output.COMPRESSOR_ATTR: compressor,
                    Output.MISSING_VALUE_ATTR: DataVars.MISSING_UINT8_VALUE
                })

            for each in [
                DataVars.INTERP_MASK,
                DataVars.CHIP_SIZE_HEIGHT,
                DataVars.CHIP_SIZE_WIDTH,
                DataVars.FLAG_STABLE_SHIFT,
                DataVars.STABLE_COUNT_SLOW,
                DataVars.STABLE_COUNT_MASK
            ]:
                encoding_settings[each] = {
                    Output.DTYPE_ATTR: DataVars.INT_TYPE[each]
                }

                if each in DataVars.INT_MISSING_VALUE:
                    encoding_settings[each][Output.MISSING_VALUE_ATTR] = DataVars.INT_MISSING_VALUE[each]

            # new_v_vars: ['v', 'v_error', 'vx', 'vx_error', 'vx_error_mask',
            # 'vx_error_modeled', 'vx_error_slow', 'vx_stable_shift',
            # 'vx_stable_shift_mask', 'vx_stable_shift_slow',
            # 'vy', 'vy_error', 'vy_error_mask', 'vy_error_modeled', 'vy_error_slow',
            # 'vy_stable_shift', 'vy_stable_shift_mask', 'vy_stable_shift_slow',
            # 'va', 'va_error', 'va_error_mask', 'va_error_modeled', 'va_error_slow',
            # 'va_stable_shift', 'va_stable_shift_mask', 'va_stable_shift_slow',
            # 'vr', 'vr_error', 'vr_error_mask', 'vr_error_modeled', 'vr_error_slow',
            # 'vr_stable_shift', 'vr_stable_shift_mask', 'vr_stable_shift_slow', 'M11', 'M12']
            for each in new_v_vars:
                # Default to floating point data type and _FillValue attribute for encoding
                missing_value = DataVars.MISSING_VALUE
                missing_value_attr = Output.FILL_VALUE_ATTR

                dtype_value = np.float32

                if each in DataVars.INT_TYPE:
                    missing_value_attr = Output.MISSING_VALUE_ATTR

                    if each in DataVars.INT_MISSING_VALUE:
                        missing_value = DataVars.INT_MISSING_VALUE[each]

                    if each in DataVars.INT_TYPE:
                        dtype_value = DataVars.INT_TYPE[each]

                encoding_settings[each] = {
                    missing_value_attr: missing_value,
                    Output.DTYPE_ATTR: dtype_value
                }

                encoding_settings[each].update(compression)

            # new_vars_zero_missing_value: ['M11_dr_to_vr_factor', 'M12_dr_to_vr_factor']
            for each in new_vars_zero_missing_value:
                encoding_settings[each] = {
                    Output.DTYPE_ATTR: np.float32,
                    Output.FILL_VALUE_ATTR: DataVars.MISSING_BYTE
                }
                encoding_settings[each].update(compression)

            # Explicitly desable _FillValue for some variables: can be set for floating
            # point data variables only.
            # xarray is broken if _FillValue=None is provided along with "chunks"
            # encoding attribute: don't do it.
            # for each in [Coords.MID_DATE,
            #              DataVars.STABLE_COUNT_SLOW,
            #              DataVars.STABLE_COUNT_MASK,
            #              DataVars.AUTORIFT_SOFTWARE_VERSION,
            #              DataVars.ImgPairInfo.DATE_DT,
            #              DataVars.ImgPairInfo.DATE_CENTER,
            #              DataVars.ImgPairInfo.SATELLITE_IMG1,
            #              DataVars.ImgPairInfo.SATELLITE_IMG2,
            #              DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
            #              DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
            #              DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE,
            #              DataVars.ImgPairInfo.MISSION_IMG1,
            #              DataVars.ImgPairInfo.MISSION_IMG2,
            #              DataVars.ImgPairInfo.SENSOR_IMG1,
            #              DataVars.ImgPairInfo.SENSOR_IMG2]:
            #     encoding_settings.setdefault(each, {}).update({Output.FILL_VALUE_ATTR: None})

            # Set units for all datetime objects
            for each in [DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
                         DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
                         DataVars.ImgPairInfo.DATE_CENTER,
                         Coords.MID_DATE]:
                encoding_settings.setdefault(each, {}).update({DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS})

            # Set array size to accomodate maximum length of the sensor
            for each in [DataVars.ImgPairInfo.SATELLITE_IMG1,
                         DataVars.ImgPairInfo.SATELLITE_IMG2]:
                max_sensor_len = max(map(len, self.layers[each].values))
                if max_sensor_len > ITSCube.MAX_SENSOR_LEN:
                    raise RuntimeError(
                        f'"{each}" will be truncated to the current length limit: '
                        f'{ITSCube.MAX_SENSOR_LEN}: {max_sensor_len} length is detected. '
                        'Please update ITSCube.MAX_SENSOR_LEN value.'
                    )

                encoding_settings.setdefault(each, {}).update({Output.DTYPE_ATTR: f'U{ITSCube.MAX_SENSOR_LEN}'})

            # Check for the length limit of the granule_url's
            max_url_len = max(map(len, self.layers[DataVars.URL].values))
            if max_url_len > ITSCube.MAX_GRANULE_URL_LEN:
                raise RuntimeError(
                    f'"{each}" will be truncated to the current length limit: '
                    '{ITSCube.MAX_GRANULE_URL_LEN}: {max_url_len} length is detected.'
                    'Please update ITSCube.MAX_GRANULE_URL_LEN value.')

            encoding_settings.setdefault(DataVars.URL, {}).update({Output.DTYPE_ATTR: f'U{ITSCube.MAX_GRANULE_URL_LEN}'})

            # Determine optimal chunking for the cube
            chunking_settings_3d = (
                min(self.max_number_of_layers, ITSCube.TIME_CHUNK_VALUE),
                ITSCube.X_Y_CHUNK_VALUE,
                ITSCube.X_Y_CHUNK_VALUE
            )

            # Set chunking for writing to the store
            for each in [DataVars.INTERP_MASK,
                         DataVars.CHIP_SIZE_HEIGHT,
                         DataVars.CHIP_SIZE_WIDTH,
                         DataVars.V,
                         DataVars.V_ERROR,
                         DataVars.VA,
                         DataVars.VR,
                         DataVars.VX,
                         DataVars.VY,
                         DataVars.M11,
                         DataVars.M12]:
                encoding_settings.setdefault(each, {})[Output.CHUNKS_ATTR] = chunking_settings_3d

            chunking_settings_1d = min(self.max_number_of_layers, ITSCube.TIME_CHUNK_VALUE_1D)

            for each in [
                DataVars.FLAG_STABLE_SHIFT,
                DataVars.STABLE_COUNT_SLOW,
                DataVars.STABLE_COUNT_MASK,
                'vx_' + DataVars.ERROR,
                'vx_' + DataVars.ERROR_MASK,
                'vx_' + DataVars.ERROR_MODELED,
                'vx_' + DataVars.ERROR_SLOW,
                'vx_' + DataVars.STABLE_SHIFT,
                'vx_' + DataVars.STABLE_SHIFT_SLOW,
                'vx_' + DataVars.STABLE_SHIFT_MASK,
                'vy_' + DataVars.ERROR,
                'vy_' + DataVars.ERROR_MASK,
                'vy_' + DataVars.ERROR_MODELED,
                'vy_' + DataVars.ERROR_SLOW,
                'vy_' + DataVars.STABLE_SHIFT,
                'vy_' + DataVars.STABLE_SHIFT_SLOW,
                'vy_' + DataVars.STABLE_SHIFT_MASK,
                'va_' + DataVars.ERROR,
                'va_' + DataVars.ERROR_MASK,
                'va_' + DataVars.ERROR_MODELED,
                'va_' + DataVars.ERROR_SLOW,
                'va_' + DataVars.STABLE_SHIFT,
                'va_' + DataVars.STABLE_SHIFT_SLOW,
                'va_' + DataVars.STABLE_SHIFT_MASK,
                'vr_' + DataVars.ERROR,
                'vr_' + DataVars.ERROR_MASK,
                'vr_' + DataVars.ERROR_MODELED,
                'vr_' + DataVars.ERROR_SLOW,
                'vr_' + DataVars.STABLE_SHIFT,
                'vr_' + DataVars.STABLE_SHIFT_SLOW,
                'vr_' + DataVars.STABLE_SHIFT_MASK,
                'M11_' + DataVars.DR_TO_VR_FACTOR,
                'M12_' + DataVars.DR_TO_VR_FACTOR,
                DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
                DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
                DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE,
                DataVars.ImgPairInfo.SATELLITE_IMG1,
                DataVars.ImgPairInfo.SATELLITE_IMG2,
                DataVars.ImgPairInfo.MISSION_IMG1,
                DataVars.ImgPairInfo.MISSION_IMG2,
                DataVars.ImgPairInfo.SENSOR_IMG1,
                DataVars.ImgPairInfo.SENSOR_IMG2,
                DataVars.ImgPairInfo.DATE_CENTER,
                DataVars.ImgPairInfo.DATE_DT
            ]:
                # Reset existing encoding settings if any for the data variable
                self.layers[each].encoding = {}
                encoding_settings.setdefault(each, {})[Output.CHUNKS_ATTR] = (chunking_settings_1d)

                if Output.FILL_VALUE_ATTR in self.layers[each].attrs:
                    del self.layers[each].attrs[Output.FILL_VALUE_ATTR]

                # logging.info(f'Encoding for {each}: {encoding_settings[each]}')
                # logging.info(f'each.attrs for {each}: {self.layers[each].attrs}')
                # logging.info(f'each.encoding for {each}: {self.layers[each].encoding}')

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
        Format statistics of the run. Don't display statistics if using
        granules as provided in the input JSON file.
        """
        if ITSCube.USE_GRANULES is not None:
            return

        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_granules[DataVars.SKIP_PROJECTION].values()])

        self.logger.info(f"Skipped granules due to empty data: {len(self.skipped_granules[DataVars.SKIP_EMPTY_DATA])} ({100.0 * len(self.skipped_granules[DataVars.SKIP_EMPTY_DATA])/num_urls}%)")
        self.logger.info(f"Skipped granules due to double mid_date: {len(self.skipped_granules[DataVars.SKIP_DUPLICATE])} ({100.0 * len(self.skipped_granules[DataVars.SKIP_DUPLICATE])/num_urls}%)")
        self.logger.info(f"Skipped granules due to wrong projection: {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        if len(self.skipped_granules[DataVars.SKIP_PROJECTION]):
            self.logger.info(f"Skipped wrong projections: {sorted(self.skipped_granules[DataVars.SKIP_PROJECTION].keys())}")

    def read_dataset(self, url: str):
        """
        Read Dataset from the file and pre-process for the cube layer.
        """
        with xr.open_dataset(url) as ds:
            return self.preprocess_dataset(ds, url)

    def read_s3_dataset(
            self,
            each_url: str,
            s3: s3fs.S3FileSystem,
            total_retries: int = 5,
            num_seconds: int = 15
    ):
        """
        Read Dataset from the S3 bucket and pre-process it for the cube layer.
        Return re-tried exceptions messages, if any, and cube layer information.

        each_url: Granule S3 URL.
        s3: s3fs.S3FileSystem object to access the granule from.
        total_retries: Number of retries in a case of exception
        num_seconds: Number of seconds to sleep between retries.
        """
        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

        num_retries = 0
        got_granule = False

        exception_info = []

        while not got_granule and num_retries < total_retries:
            num_retries += 1
            try:
                with s3.open(s3_path, mode='rb') as fhandle:
                    with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                        results = self.preprocess_dataset(ds, each_url)
                        return exception_info, *results

            except RuntimeError:
                # Re-raise the exception
                raise

            except:
                # Other types of exceptions (like botocore.exceptions.ResponseStreamingError)
                exception_info.append(f'Got exception reading {s3_path}: {sys.exc_info()}')
                if num_retries < total_retries:
                    # Sleep if it's not last attempt
                    exception_info.append(f'Sleeping for {num_seconds} seconds...')
                    time.sleep(num_seconds)

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
    def validate_cube(ds: xr.Dataset, start_date: str, cube_url: str):
        """
        Validate just written to disk datacube. This method is introduced because
        of observed corrupted datacube properties:
        1. Validate X and Y coordinates values: not to include NaN's.
        2. Validate datetime objects of the cube against start_date of the cube.

        This check is introduced to capture corrupted datacubes as early as
        possible in the cube generation.
        """
        logging.info(f"Validating X and Y coordinates for {cube_url}")
        if np.any(np.isnan(ds.x.values)):
            raise RuntimeError(f'Detected NaNs in X: {cube_url} ds.size={ds.sizes}')

        if np.any(np.isnan(ds.y.values)):
            raise RuntimeError(f'Detected NaNs in Y: {cube_url} ds.size={ds.sizes}')

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

    @staticmethod
    def remove_s3_datacube(cube_store: str, skipped_granules_file: str, s3_bucket: str):
        """
        Remove Zarr store and corresponding json file (with records of skipped
        granules for the cube) in S3 if they exists - this is done to replace existing
        cube with newly generated one:
            * at the beginning of the processing if --removeExistingCube command-line
              option is provided
            * at the end of the processing if destination location of created
              cube is in S3 bucket. This is done to avoid lingering Zarr objects
              generated with other settings which will result in different
              "directory" structure of the Zarr store.
        """
        # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
        # resulting in as many error messages as there are files in Zarr store
        # to copy
        env_copy = os.environ.copy()
        if ITSCube.exists(cube_store, s3_bucket):
            cube_s3_path = os.path.join(s3_bucket, cube_store)

            command_line = [
                "awsv2", "s3", "rm", "--recursive",
                cube_s3_path
            ]
            logging.info(f'Removing existing cube {cube_s3_path}: {" ".join(command_line)}')

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to remove original {cube_s3_path}: {command_return.stdout}")

            json_s3_path = os.path.join(s3_bucket, skipped_granules_file)

            command_line = [
                "awsv2", "s3", "rm",
                json_s3_path
            ]
            logging.info(f'Removing existing skipped granules json {json_s3_path}: {" ".join(command_line)}')

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to remove original {json_s3_path}: {command_return.stdout}")

    @staticmethod
    def read_shapefile(shapefile: str):
        """
        Read shape file in with ice mask information required for processing.

        Inputs:
        =======
        shapefile: URL to the shapefile.

        Returns:
        ========
        Object representing the shapefile.
        """
        # Make sure it's S3 URL that is provided
        shape_file = shapefile.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        shape_file = shape_file.replace(ITSCube.SHAPE_PATH_URL, '')
        return gpd.read_file(shape_file)

    @staticmethod
    def read_ice_mask(shapefile_row, column_name, grid_x, grid_y):
        """
        Read ice mask as stored in "column_name" field of the shapefile's row.

        Inputs:
        =======
        found_row: Row from the shape file that corresponds to the datacube's EPSG code
        column_name: Name of the shape file column that represents the land ice mask.
        grid_x: X coordinates of the datacube grid.
        grid_y: Y coordinates of the datacube grid.

        Returns: A tuple of:
                 * None if there is no overlap between land ice mask and datacube polygon,
                 or land ice mask for the same grid as datacube polygon.
                 * URL to the mask file as provided in the shapefile.
        """
        ice_mask_file = shapefile_row[column_name].item()

        ice_mask_file = ice_mask_file.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        ice_mask_file = ice_mask_file.replace(ITSCube.SHAPE_PATH_URL, '')
        logging.info(f'Using {column_name} mask file {ice_mask_file}')

        # Load the mask
        mask_ds = rioxarray.open_rasterio(ice_mask_file)

        # Zoom into cube polygon
        mask_x = (mask_ds.x >= grid_x.min()) & (mask_ds.x <= grid_x.max())
        mask_y = (mask_ds.y >= grid_y.min()) & (mask_ds.y <= grid_y.max())
        mask = (mask_x & mask_y)

        # Allocate xr.DataArray to match cube dimentions: will be empty if
        # no overlap exists with the ice mask, or will be set to overlap with
        # ice mask
        ice_mask = xr.DataArray(
            np.zeros((len(grid_y), len(grid_x))),
            coords={
                Coords.X: grid_x,
                Coords.Y: grid_y
            },
            dims=[Coords.Y, Coords.X]
        )

        if mask.sum().item() == 0:
            # Mask does not overlap with the cube
            logging.info(f'No overlap is detected with {column_name} mask data {ice_mask_file}')

        else:
            cropped_mask_ds = mask_ds.where(mask, drop=True)

            # Populate mask data into cube-size array
            if cropped_mask_ds.ndim == 3:
                # If it's 3d data, it should have first dimension=1: just
                # one layer is expected
                mask_data_sizes = cropped_mask_ds.shape
                if mask_data_sizes[0] != 1:
                    raise RuntimeError(f'Unexpected size for mask data from {ice_mask_file} file: {mask_data_sizes}')

                else:
                    ice_mask.loc[dict(x=cropped_mask_ds.x, y=cropped_mask_ds.y)] = cropped_mask_ds[0]

            else:
                ice_mask.loc[dict(x=ds.x, y=ds.y)] = cropped_mask_ds

        # Store mask as numpy array since all calcuations are done using
        # numpy arrays
        ice = ice_mask.values
        land_ice_coverage = int(np.sum(ice))/(len(grid_x)*len(grid_y))*100
        logging.info(f'Got {column_name} mask for {np.round(land_ice_coverage, 2)}% cells of the datacube')

        return (ice, shapefile_row[column_name].item())


if __name__ == '__main__':
    import argparse
    import warnings
    import sys

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSCube.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=8,
        help='Number of Dask workers to use for parallel processing [%(default)d].'
    )
    parser.add_argument(
        '-r', '--removeExistingCube',
        action='store_true',
        default=False,
        help='Flag to remove existing datacube in S3 bucket, default is to update existing datacube. '
             'This flag is useful when we need to re-create the cube from scratch, though beware of AWS limit of push requests '
             'when multiple datacubes are deleted at the same time.'
    )
    parser.add_argument(
        '-n', '--numberGranules',
        type=int,
        default=None,
        help='Number of ITS_LIVE granules to consider for the cube (due to runtime limitations). '
             'If none is provided, process all found granules.'
    )
    # parser.add_argument(
    #     '-l', '--localPath',
    #     type=str,
    #     default=None,
    #     help='Local path that stores ITS_LIVE granules.'
    # )
    parser.add_argument(
        '-o', '--outputStore',
        type=str,
        default="cubedata.zarr",
        help='Zarr output directory to write cube data to [%(default)s].'
    )
    parser.add_argument(
        '-b', '--outputBucket',
        type=str,
        default='',
        help='S3 bucket to copy Zarr format of the datacube to (for example, s3://its-live-data) [%(default)s].'
    )
    parser.add_argument(
        '-c', '--chunks',
        type=int,
        default=250,
        help='Number of granules to write at a time [%(default)d].'
    )
    parser.add_argument(
        '--targetProjection',
        type=str,
        required=True,
        help='UTM target projection.'
    )
    parser.add_argument(
        '--dimSize',
        type=float,
        default=100000,
        help='Cube dimension in meters [%(default)d].'
    )
    parser.add_argument(
        '-g', '--gridCellSize',
        type=int,
        default=120,
        help='Grid cell size of input ITS_LIVE granules [%(default)d].'
    )
    parser.add_argument(
        '--fivePointsPerPolygonSide',
        action='store_true',
        help='Define 5 points per side before re-projecting granule polygon to longitude/latitude coordinates'
    )
    parser.add_argument(
        '--useGranulesFile',
        type=Path,
        default=None,
        help='Json file that stores a list of ITS_LIVE image velocity granules to build datacube from [%(default)s].'
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
        help='Stop date in YYYY-MM-DD format to pass to search API query to get velocity pair granules. Use "now" if not provided.'
    )
    parser.add_argument(
        '--disableCubeValidation',
        action='store_true',
        default=False,
        help='Disable datetime validation for created datacube. This is to identify corrupted Zarr stores at the time of creation.'
    )
    parser.add_argument(
        '-s', '--shapeFile',
        type=str,
        default='s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp',
        help='Shapefile that stores ice masks per each of the EPSG codes [%(default)s].'
    )
    parser.add_argument(
        '-p', '--pathURLToken',
        type=str,
        default='.s3.amazonaws.com',
        help='Path URL token for each of the input granules to remove for S3 access [%(default)s].'
    )

    # One of --centroid or --polygon options is allowed for the datacube coordinates
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--centroid',
        type=str,
        action='store',
        help='JSON 2-element list for centroid point (x, y) of the datacube in target EPSG code projection. '
             'Polygon vertices are calculated based on the centroid and cube dimension arguments.'
    )
    group.add_argument(
        '--polygon',
        type=str,
        action='store',
        help='JSON list of polygon points ((x1, y1), (x2, y2),... (x1, y1)) to define datacube in target EPSG code projection.'
    )

    args = parser.parse_args()

    # Read shape file with ice masks information in
    ITSCube.SHAPE_FILE = ITSCube.read_shapefile(args.shapeFile)

    # Enforce .zarr file extension for the datacube store
    if not args.outputStore.endswith(FileExtension.ZARR):
        raise RuntimeError(f'Output Zarr store is expected to have {FileExtension.ZARR} extension, got {args.outputStore}')

    ITSCube.NUM_THREADS = args.threads
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks
    ITSCube.CELL_SIZE = args.gridCellSize
    ITSCube.PATH_URL = args.pathURLToken

    if args.useGranulesFile:
        # Check for this option first as another mutually exclusive option has a default value
        ITSCube.USE_GRANULES = json.loads(args.useGranulesFile.read_text())
        logging.info(f'Using {len(ITSCube.USE_GRANULES)} granules as provided in {args.useGranulesFile.name} file')

    if len(args.outputBucket):
        # S3 bucket is provided, format S3 path to the target datacube
        ITSCube.S3 = os.path.join(args.outputBucket, args.outputStore)
        logging.info(f'Cube S3: {ITSCube.S3}')

        # URL is valid only if output S3 bucket is provided
        ITSCube.URL = ITSCube.S3.replace(ITSCube.S3_PREFIX, ITSCube.HTTP_PREFIX)
        url_tokens = urlparse(ITSCube.URL)
        ITSCube.URL = url_tokens._replace(netloc=url_tokens.netloc+ITSCube.PATH_URL).geturl()
        logging.info(f'Cube URL: {ITSCube.URL}')

    else:
        ITSCube.S3 = ''
        ITSCube.URL = ''

    # Set local file path for skipped granules info
    ITSCube.SKIPPED_GRANULES_FILE = args.outputStore.replace(FileExtension.ZARR, FileExtension.JSON)

    if args.removeExistingCube and len(args.outputBucket):
        # Remove Zarr store in S3 if it exists - this is done to replace existing
        # cube with newly generated one
        ITSCube.remove_s3_datacube(args.outputStore, ITSCube.SKIPPED_GRANULES_FILE, args.outputBucket)

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
            (c_x - off, c_y + off)
        )
    else:
        # Polygon for the cube definition is provided
        polygon = json.loads(args.polygon)

    if args.fivePointsPerPolygonSide:
        # Introduce 5 points per each polygon side
        polygon = itslive_utils.add_five_points_to_polygon_side(polygon)

    # Create cube object
    cube = ITSCube(polygon, projection)

    # Record used package versions
    cube.logger.info(f'Command: {sys.argv}')
    cube.logger.info(f'Command args: {args}')
    cube.logger.info(f'{xr.show_versions()}')
    cube.logger.info(f's3fs: {s3fs.__version__}')

    # Parameters for the search granule API
    end_date = datetime.now().strftime('%Y-%m-%d') if args.searchAPIStopDate is None else args.searchAPIStopDate
    API_params = {
        'start': args.searchAPIStartDate,
        'end': end_date,
        'percent_valid_pixels': 1
    }
    cube.logger.info(f'ITS_LIVE API parameters: {API_params}')

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
    remove_original_datacube = False

    try:
        if not args.disableCubeValidation and os.path.exists(args.outputStore):
            with xr.open_zarr(args.outputStore, decode_timedelta=False, consolidated=True) as ds:
                ITSCube.validate_cube(ds, args.searchAPIStartDate, args.outputStore)

            gc.collect()

        if os.path.exists(args.outputStore) and len(args.outputBucket):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy

            # TODO: Might need to make it a command-line option?
            # This should be done only when Zarr chunking of the existing (in S3 bucket)
            # datacube is changed
            if remove_original_datacube:
                # Remove Zarr store in S3 if it exists: updated Zarr, which is stored to the
                # local file system before copying to the S3 bucket, might have different
                # "sub-directory" structure. This will result in original "sub-directories"
                # and "new" ones to co-exist for the same Zarr store. This doubles up
                # the Zarr disk usage in S3 bucket.
                ITSCube.remove_s3_datacube(args.outputStore, ITSCube.SKIPPED_GRANULES_FILE, args.outputBucket)

            env_copy = os.environ.copy()

            # Allow for multiple retries to avoid AWS triggered errors
            for each_input, each_recursive_option, each_validate_flag in zip(
                [args.outputStore, ITSCube.SKIPPED_GRANULES_FILE],
                [True, False],
                [True, False]
            ):
                file_is_copied = False
                num_retries = 0
                command_return = None

                command_line = ["awsv2", "s3", "cp"]

                if each_recursive_option:
                    command_line.append('--recursive')

                command_line.extend([
                    each_input,
                    os.path.join(args.outputBucket, os.path.basename(each_input)),
                    "--acl", "bucket-owner-full-control"
                ])

                logging.info(' '.join(command_line))

                while not file_is_copied and num_retries < ITSCube.NUM_AWS_COPY_RETRIES:
                    logging.info(f"Attempt #{num_retries+1} to copy {each_input} to {args.outputBucket}")

                    command_return = subprocess.run(
                        command_line,
                        env=env_copy,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                    )

                    if command_return.returncode != 0:
                        # Report the whole stdout stream as one logging message
                        logging.warning(f"Failed to copy {each_input} to {args.outputBucket} with returncode={command_return.returncode}: {command_return.stdout}")

                        num_retries += 1
                        # If failed due to AWS SlowDown error, retry
                        if num_retries != ITSCube.NUM_AWS_COPY_RETRIES and \
                                ITSCube.AWS_SLOW_DOWN_ERROR in command_return.stdout.decode('utf-8'):
                            # Sleep if it's not a last attempt to copy
                            time.sleep(ITSCube.AWS_COPY_SLEEP_SECONDS)

                        else:
                            # Don't retry otherwise
                            num_retries = ITSCube.NUM_AWS_COPY_RETRIES

                    else:
                        file_is_copied = True

                if not file_is_copied:
                    raise RuntimeError(f'Failed to copy {each_input} to {args.outputBucket} with command.returncode={command_return.returncode}')

                elif not args.disableCubeValidation:
                    if each_validate_flag:
                        # Validate just copied to S3 datacube
                        s3_in, cube_store, ds_from_zarr, _ = ITSCube.init_input_store(each_input, args.outputBucket, read_skipped_granules=False)
                        ITSCube.validate_cube(ds_from_zarr, args.searchAPIStartDate, os.path.join(args.outputBucket, each_input))

    finally:
        # Remove locally written Zarr store.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        if len(args.outputBucket) and os.path.exists(args.outputStore):
            logging.info(f'Removing local copy of {args.outputStore}')
            shutil.rmtree(args.outputStore)

        # Remove locally skipped granules info file.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        if len(args.outputBucket) and len(ITSCube.SKIPPED_GRANULES_FILE) and \
                os.path.exists(ITSCube.SKIPPED_GRANULES_FILE):
            logging.info(f'Removing local copy of {ITSCube.SKIPPED_GRANULES_FILE}')
            os.unlink(ITSCube.SKIPPED_GRANULES_FILE)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')

    logging.info('Done.')
