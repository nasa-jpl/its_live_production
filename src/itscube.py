"""
ITSCube class creates ITS_LIVE datacube based on target projection,
bounding polygon and datetime period provided by the caller.

Authors: Masha Liukis, Alex Gardner, Mark Fahnestock
"""
from dateutil.parser import parse
from datetime import datetime
import gc
import json
from joblib import Parallel, delayed, parallel_config
import logging
import os
from pathlib import Path
import pyproj
import shutil
import timeit
import zarr
import numpy as np
import pandas as pd
import re
import s3fs
import subprocess
from tqdm import tqdm
import xarray as xr
from urllib.parse import urlparse

# Local modules
import itslive_utils
from grid import Bounds, Grid
from itscube_types import (
    CubeFormat,
    ImgPairInfo,
    Mapping,
    Vars,
    SkippedGranules
)
from itslive_binary_type import BinaryFlag
import aws_utils
import utils
import shapefile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Coordinates attributes for the output store
MID_DATE_ATTRS = {
    Vars.attrs.std_name: utils.Coords.STD_NAME[utils.Coords.MID_DATE],
    Vars.attrs.description: utils.Coords.DESCRIPTION[utils.Coords.MID_DATE]
}
X_ATTRS = {
    Vars.attrs.std_name: utils.Coords.STD_NAME[utils.Coords.X],
    Vars.attrs.description: utils.Coords.DESCRIPTION[utils.Coords.X]
}
Y_ATTRS = {
    Vars.attrs.std_name: utils.Coords.STD_NAME[utils.Coords.Y],
    Vars.attrs.description: utils.Coords.DESCRIPTION[utils.Coords.Y]
}


# Landsat8 filename prefixes to use when we need to remove duplicate
# reprocessed granules for Landsat8/9
# Per Mark comments on Slack:
# "Should keep both prefixes for L9, but there may not be any ‘LO09’ images -
# the O means only optical bands were acquired for that frame (no
# thermal bands), the ‘LC’ means both optical and thermal were acquired.
# We don’t care about thermal, but we have to deal with the file names USGS
# uses."
LANDSAT89_PREFIX = tuple(['LC08', 'LO08', 'LC09', 'LO09'])


class ITSCube:
    """
    Builds ITS_LIVE datacube: velocity pair time series for a spatial region
    and time period.
    """
    # Current ITSCube software version
    Version = '1.0'

    # Number of connections in the connection pool for AWS S3 access, to be
    # used when creating S3FileSystem instance.
    MAX_AWS_CONNECTIONS = 32

    # Number of chunks to backup in parallel if updating
    # the datacube in S3 bucket
    NUM_CHUNKS_TO_BACKUP = 1000

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = 'EPSG:4326'

    S3_PREFIX = 's3://'
    HTTP_PREFIX = 'https://'

    NO_AWS_SIGNING = False

    # Token within granule's HTTP URL that needs to be replaced to get file
    # location within S3 bucket using S3 URL:
    # from 'https://its-live-data.s3.amazonaws.com/file.nc'
    # to
    # 's3://its-live-data/file.nc'
    PATH_URL = utils.PATH_URL

    # By default, it's set to the same value as in utils, but utils.PATH_URL
    # can be set to a different value for some datacubes, so keep its own
    # path URL for the shape files with ice masks to avoid confusion when
    # utils.PATH_URL is set to a different value for some datacubes.
    SHAPE_PATH_URL = utils.PATH_URL

    # STAC catalog S3 URL for the ITS_LIVE granules
    STAC_CATALOG = "s3://its-live-data/test-space/stac"

    # Start and end dates for the catalog search
    START_DATE = '1982-01-01'
    END_DATE = None

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
    CELL_SIZE = 120.0

    # No-value for the chip size height variable in the datacube.
    CHIP_SIZE_HEIGHT_NO_VALUE = 65535

    # Chunking to apply when writing datacube to the Zarr store
    TIME_CHUNK_VALUE = 20000
    X_Y_CHUNK_VALUE = 10

    # Chunking to apply to 1-D data variables when writing datacube to the
    # Zarr store
    TIME_CHUNK_VALUE_1D = 200000

    # ATTN: Character arrays size must be explicitely set before first write:
    # to avoid truncation of the data if first ever written block of data
    # has values of less characters in length than values that would be
    # written in the future blocks of data. This is a requirement of the newer
    # version of zarr library.

    # Maximum length for the satellite value across all used missions
    MAX_SATELLITE_LEN = 2

    # Maximum length for the sensor value across all used missions
    MAX_SENSOR_LEN = 5

    # Maximum length of the granule URL
    MAX_GRANULE_URL_LEN = 1024

    # Token to split image pair filename into two image names
    SPLIT_IMAGES_TOKEN = '_X_'
    IMAGE_TOKEN = '_'

    # If a list of granules to generate datacube from is provided through
    # an input JSON file.
    USE_GRANULES = None

    # Shape file to locate ice masks files that correspond to the datacube's
    # EPSG code
    SHAPE_FILE = None

    # Flag indicating whether to use an existing backup of the datacube, if
    # one exists.
    # Reusing backups can be helpful, but it’s often cumbersome to delete
    # incomplete backups left behind by terminated EC2 jobs in AWS. Therefore,
    # this is made optional, with the default behavior set to *not* reuse an
    # existing backup.
    USE_EXISTING_BACKUP = False

    # Flag if existing datacube should be ignored for the run, to overwrite
    # any existing cube with the same name.
    IGNORE_EXISTING_CUBE = False

    # Grid boundaries for the datacube based on its bounding polygon, to filter
    # each granule's spacial extents by
    GRID_X_MIN = None
    GRID_X_MAX = None
    GRID_Y_MIN = None
    GRID_Y_MAX = None

    # Cube target projection - to consider granules that have data only
    # within the target projection
    PROJECTION = None

    def __init__(self, polygon: tuple, projection: str):
        """
        Initialize object.

        Inputs:
        polygon (tuple): Polygon for the datacube tile.
        projection (str): Projection code for the polygon coordinates.
        """
        self.logger = logging.getLogger("datacube")
        self.logger.info(f"Polygon: {polygon}")
        self.logger.info(f"Projection: {projection}")

        ITSCube.PROJECTION = projection
        self.polygon = polygon

        # All layers are required to have the same autoRIFT parameter file:
        # set it to the parameter file for the fist granule to be appended
        # to the new datacube. Set it as an attribute for existing datacube.
        self.autoRIFTParamFile = None

        # Set min/max x/y values to filter each granule's spacial extents by
        x = Bounds([each[0] for each in polygon])
        y = Bounds([each[1] for each in polygon])

        # Grid for the datacube based on its bounding polygon
        self.grid_x, self.grid_y = Grid.create(x, y, ITSCube.CELL_SIZE)

        self.x_cell = self.grid_x[1] - self.grid_x[0]
        self.y_cell = self.grid_y[1] - self.grid_y[0]

        # Grid cell half sizes
        self.half_x_cell = self.x_cell / 2.0
        self.half_y_cell = self.y_cell / 2.0

        abs_x_size = np.abs(self.half_x_cell)
        abs_y_size = np.abs(self.half_y_cell)

        # Define range for x and y based on grid edges
        ITSCube.GRID_X_MIN = self.grid_x.min() - abs_x_size
        ITSCube.GRID_X_MAX = self.grid_x.max() + abs_x_size

        ITSCube.GRID_Y_MIN = self.grid_y.min() - abs_y_size
        ITSCube.GRID_Y_MAX = self.grid_y.max() + abs_y_size

        # Ensure lonlat output order
        to_lon_lat_transformer = pyproj.Transformer.from_crs(
            f"EPSG:{projection}",
            ITSCube.LON_LAT_PROJECTION,
            always_xy=True
        )

        mid_x = (self.grid_x.min() + self.grid_x.max()) / 2
        mid_y = (self.grid_y.min() + self.grid_y.max()) / 2

        # Convert centroid to lon/lat coordinates
        self.center_lon_lat = to_lon_lat_transformer.transform(mid_x, mid_y)

        # Convert polygon from its target projection to longitude/latitude
        # coordinates which are used by granule search API
        self.polygon_coords = []

        for each in polygon:
            coords = to_lon_lat_transformer.transform(each[0], each[1])
            self.polygon_coords.append(list(coords))

        self.logger.info(
            f"Polygon's longitude/latitude coordinates: {self.polygon_coords}"
        )

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle date, original granules URLs)
        self.ds = []
        self.dates = []
        self.urls = []
        self.num_urls_from_api = None

        # Keep track of skipped granules due to:
        # * no data coverage for the cube
        # * other than target projection
        # * duplicate middle date
        self.skipped_granules = {
            SkippedGranules.empty: [],
            SkippedGranules.duplicate: [],
            SkippedGranules.projection: {}
        }
        # # Keep track of skipped granules due to no data for the polygon of interest
        # self.skipped_empty_granules = []
        # # Keep track of "double" granules with older processing date which are
        # # not included into the cube
        # self.skipped_double_granules = []

        # Constructed cube
        self.layers = None

        # For existing datacubes capture size of unicode strings as newer
        # version of zarr requires dtype to be the same when appending the
        # layers
        self.existing_dtypes = {}

        # Number of layers in the cube - this will be the same as the number
        # of granules if datacube exists. If datacube is being created from
        # scratch, this number will be zero. This number indicates starting
        # index when appending new layers to the existing datacube.
        self.current_cube_layers = 0

        # Dates when datacube was created or updated
        self.date_created = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        self.date_updated = None

        # Number of layers for cube generation based on the searchAPI query return
        self.max_number_of_layers = 0

        # Find corresponding to EPSG ice masks for the cube
        # -------------------------------------------------------------------
        # Land ice mask for the cube
        self.land_ice_mask, \
        self.land_ice_mask_url = shapefile.read_ice_mask(
            ITSCube.SHAPE_FILE, shapefile.LANDICE, self.grid_x, self.grid_y,
            ITSCube.PROJECTION
        )

        # Floating ice coverage for the datacube
        self.floating_ice_mask, \
        self.floating_ice_mask_url = shapefile.read_ice_mask(
            ITSCube.SHAPE_FILE, shapefile.FLOATINGICE, self.grid_x, self.grid_y,
            ITSCube.PROJECTION
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
            SkippedGranules.empty: [],
            SkippedGranules.duplicate: [],
            SkippedGranules.projection: {}
        }

    def request_granules(self, num_granules: int):
        """
        Send request to ITS_LIVE API to get a list of granules to satisfy
        polygon request. Or instead for the testing purposes use a list of
        provided granules through input JSON file.

        Inputs:
        num_granules (int): Number of first granules to examine.
            (ATTN: This is for testing only as a temporary solution to a very
            long time to open remote granules. Should not be used when
            running in production mode.)
        """
        if ITSCube.USE_GRANULES is not None:
            found_urls = ITSCube.USE_GRANULES

            if num_granules:
                found_urls = ITSCube.USE_GRANULES[:num_granules]

                # # Pick S1 or S2 granules to test
                # sentinel_granules = [each for each in ITSCube.USE_GRANULES
                #   if os.path.basename(each)[0] == 'S']
                # found_urls.extend(sentinel_granules[:num_granules])
                self.logger.info(
                    f"Examining only first {len(found_urls)} out of "
                    f"{len(ITSCube.USE_GRANULES)} provided granules"
                )

            self.max_number_of_layers = len(found_urls)
            return found_urls

        self.logger.info(f'Getting granules for the polygon: {self.polygon_coords}')

        roi = {
            "type": "Polygon",
            "coordinates": [self.polygon_coords]
        }

        found_urls = itslive_utils.serverless_search(
            epsg_code=ITSCube.PROJECTION,
            start_date=ITSCube.START_DATE,
            end_date=ITSCube.END_DATE,
            roi=roi
        )
        total_num = len(found_urls)
        self.logger.info(
            f"Total of {total_num} granules returned by searchAPI."
        )

        if total_num == 0:
            self.logger.info(
                "No granules are found, skipping datacube generation or update"
            )

            return found_urls

        self.max_number_of_layers = total_num

        # Number of granules to examine is specified
        # ATTN: just a way to limit number of granules to be considered for the
        #       datacube generation (testing or debugging only).
        if num_granules:
            found_urls = found_urls[:num_granules]
            self.logger.info(
                f"Examining only first {len(found_urls)} out of {total_num} "
                f"found granules"
            )

        # Number of found URL's should report number of granules as returned
        # by searchAPI to provide correct % value for skipped granules if
        # updating the cube
        self.num_urls_from_api = len(found_urls)

        urls, self.skipped_granules[SkippedGranules.duplicate] = \
            ITSCube.skip_duplicate_l89_granules(found_urls)

        # Sort URLs by mid_date extracted from filename for chronological order
        urls = sorted(urls, key=ITSCube.extract_mid_date_from_url)

        # DEBUG: pick only S1 granules to test
        # sentinel_granules = [each for each in urls if
        #                       os.path.basename(each).startswith('S1')]
        # self.logger.info(f'Leaving {len(sentinel_granules)} Sentinel '
        #                   f'granules out of {len(urls)} granules for testing')
        # return sentinel_granules

        return urls

    @staticmethod
    def skip_duplicate_l89_granules(found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s))
        for the same path row granule for Landsat8 and Landsat9 data only.

        Examples of the Landsat image pair filename with one of the images
        from L89 mission group:
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
            if os.path.basename(each)
            .split(ITSCube.SPLIT_IMAGES_TOKEN)[0].startswith(LANDSAT89_PREFIX)
            or os.path.basename(each)
            .split(ITSCube.SPLIT_IMAGES_TOKEN)[1].startswith(LANDSAT89_PREFIX)
        ]

        if len(landsat89_granules) == 0:
            # There are no Landsat8 granules, no need to remove duplicates
            return found_urls, skipped_double_granules

        else:
            # Include non-Landsat89 granules into unique granules to return
            # as they don't need to be searched for duplicates
            granules = list(set(found_urls).difference(landsat89_granules))
            logging.info(f'Number of non-Landsat89 granules: {len(granules)}')

        for each_url in tqdm(
            landsat89_granules, ascii=True,
            desc=f'Skipping duplicate Landsat89 granules out of '
            f'{len(landsat89_granules)} granules...'
        ):
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
                        raise RuntimeError(
                            f'Mismatching IDs for each_url={each_url}: '
                            f'{granule_id} vs. found_url={found_url}: '
                            f'{found_granule_id}')

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

        logging.info(
            f'Keeping {len(granules)} unique granules, skipping '
            f'{len(skipped_double_granules)} Landsat89 granules'
        )

        return granules, skipped_double_granules

    def exclude_processed_granules(
        self,
        found_urls: list,
        cube_ds: xr.Dataset,
        skipped_granules: dict
    ):
        """
        * Exclude granules that are already added to the datacube, also
        all skipped granules in existing datacube (empty data, wrong
        projection, duplicate middle date) from found granules.

        * Identify if any of the skipped double mid_date granules from
        "found_urls" are already existing layers in the datacube. Need to
        mark such layers to be deleted from the datacube - this is disabled
        for now as current v2 cubes have layers with duplicate "mid_date".

        * Identify if current cube layers and remaining found_urls have
        duplicate mid_date - register these for deletion from the datacube
        if they appear as datacube layers.

        Return:
            found_urls (list): List of granules to update datacube with.
            cube_ds (xarray.Dataset): Existing datacube to update.
            skipped_granules (dict): Dictionary of already excluded datacube
                                    layers.
        """
        self.logger.info("Excluding known to datacube granules...")
        self.logger.info(
            f"Got {len(found_urls)} total granules to consider "
            f"({len(set(found_urls))} unique granules)..."
        )

        cube_granules = cube_ds[Vars.url].values.tolist()
        self.logger.info(
            f"Existing datacube granules: {len(cube_granules)} "
            f"({len(set(cube_granules))} unique granules)"
        )

        # New granules to be added to the datacube
        granules = set(found_urls).difference(cube_granules)

        # Check if any of the existing cube layers are not in found_urls
        # (this can happen if the cube is updated with different start/end dates),
        # just report it
        cube_in_found_urls = set(cube_granules).difference(found_urls)
        self.logger.info(
            f"Cube granules not in found_urls: ({len(cube_in_found_urls)})"
        )

        # Log an example of the cube layer that is not present in found_urls
        if len(cube_in_found_urls):
            self.logger.info(
                f"Cube layer not present in found_urls: {list(cube_in_found_urls)[0]}"
            )

        # Check if any of the cube granules not reported in the new found_urls
        # are due to the skipped granules in the datacube because of
        # duplicate mid_date.
        cube_in_skipped_found_urls = set(cube_in_found_urls).difference(
            self.skipped_granules[SkippedGranules.duplicate]
        )
        self.logger.info(
            f"Cube granules not in found_urls and not skipped due to "
            f"double mid_date: ({len(cube_in_skipped_found_urls)})"
        )

        # Log an example of the cube layer that is not present in found_urls
        # and not skipped due to double mid_date
        if len(cube_in_skipped_found_urls):
            self.logger.info(
                f"Example of the cube layer not present in found_urls: "
                f"{list(cube_in_skipped_found_urls)[0]}"
            )

        self.logger.info(
            f"Exclude known cube granules ({len(cube_granules)}): "
            f"{len(granules)} granules remain"
        )

        # Remove known empty granules from found_urls
        self.skipped_granules[SkippedGranules.empty] = \
            skipped_granules[SkippedGranules.empty]

        granules = granules.difference(
            self.skipped_granules[SkippedGranules.empty]
        )
        self.logger.info(
            f"Exclude known empty data granules "
            f"({len(self.skipped_granules[SkippedGranules.empty])}): "
            f"{len(granules)} granules remain"
        )

        # Remove known wrong projection granules (per projection) from
        # found_urls.
        # ATTN: int values get written as strings to json files, so make
        # sure read back in values for the keys are of int type
        for each_key, each_value in skipped_granules[
                SkippedGranules.projection].items():
            self.skipped_granules[SkippedGranules.projection][
                int(each_key)
            ] = each_value

        known_granules = []
        for each in self.skipped_granules[SkippedGranules.projection].values():
            known_granules.extend(each)

        granules = granules.difference(known_granules)
        self.logger.info(
            f"Exclude known wrong projection granules "
            f"({len(known_granules)}): {len(granules)} granules remain"
        )

        # Identify if there are any cube granules that now need to be skipped
        # due to double middle date in "new" found_urls granules
        # (self.skipped_granules[SkippedGranules.duplicate] is set by self.request_granules())
        cube_layers_to_delete = list(
            set(self.skipped_granules[SkippedGranules.duplicate])
                .intersection(cube_granules))
        self.logger.info(
            f"{len(cube_layers_to_delete)} existing datacube layers to "
            f"delete due to duplicate mid_date: {cube_layers_to_delete}"
        )

        # Remove known duplicate middle date granules from found_urls:
        # if cube's skipped granules don't appear in found_urls.skipped_granules
        # for whatever reason (different start/end dates are used for cube update)
        # self.skipped_granules[SkippedGranules.duplicate] is populated by self.request_granules()
        # with skipped granules due to double date in "found_urls"
        cube_skipped_double_granules = skipped_granules[SkippedGranules.duplicate]
        granules = granules.difference(cube_skipped_double_granules)
        self.logger.info(
            f"Removed known cube's duplicate middle date granules "
            f"({len(cube_skipped_double_granules)}): {len(granules)} "
            f"granules remain"
        )

        # Check if there are any granules between existing cube layers and found_urls
        # that have duplicate middle date
        cube_and_found_urls = cube_granules + list(granules)

        _, skipped_landsat_granules = ITSCube.skip_duplicate_l89_granules(
            cube_and_found_urls
        )

        # Check if any of the skipped granules are in the cube
        cube_layers_to_delete.extend(
            list(set(cube_granules).intersection(skipped_landsat_granules))
        )
        self.logger.info(
            f"After (cube_granules+found_urls): total of "
            f"{len(cube_layers_to_delete)} "
            f"existing datacube layers to delete due to duplicate mid_date: "
            f"{cube_layers_to_delete}"
        )

        # Make sure there is only unique granules in the list
        cube_layers_to_delete_set = set(cube_layers_to_delete)
        cube_layers_to_delete = list(cube_layers_to_delete_set)
        self.logger.info(
            f"After (cube_granules+found_urls): total of "
            f"{len(cube_layers_to_delete)} unique existing datacube layers "
            f"to delete due to duplicate mid_date: {cube_layers_to_delete}"
        )

        # ATTN: Disable deletion of any existing cube layers since current
        # v2 cubes have layers with duplicate "mid_date":
        # something to resolve in the future
        if len(cube_layers_to_delete) != 0:
            self.logger.info(
                "WARNING: Ignoring datacube layers to delete due to "
                "duplicate mid_date (for now)..."
            )
            cube_layers_to_delete = []

        # Merge two lists of skipped granules (for existing cube, new list
        # of granules from search API, and duplicate granules b/w cube and
        # new granules)
        cube_skipped_double_granules.extend(
            self.skipped_granules[SkippedGranules.duplicate]
        )
        cube_skipped_double_granules.extend(skipped_landsat_granules)
        self.skipped_granules[SkippedGranules.duplicate] = list(
            set(cube_skipped_double_granules)
        )

        # Skim down found_urls by newly skipped granules
        granules = list(granules.difference(
            self.skipped_granules[SkippedGranules.duplicate]
        ))
        self.logger.info(f"Leaving {len(granules)} granules...")

        return granules, cube_layers_to_delete

    @staticmethod
    def extract_mid_date_from_url(url: str):
        """
        Extract mid_date from granule filename by parsing acquisition dates.
        This method is used to sort granules in chronological order by
        acquisition date by avoiding reading the granule files to get its time
        dimension value.

        Supports multiple sensor filename formats:
        - Landsat: acquisition date at token[3] (format: YYYYMMDD)
        - NISAR: acquisition date+time at token[11] (format: YYYYMMDDTHHMMSS)
        - Sentinel-1: acquisition date+time at token[5] (format: YYYYMMDDTHHMMSS)
        - Sentinel-2: acquisition date+time at token[2] (format: YYYYMMDDTHHMMSS)

        Mid_date is calculated as the average of the two acquisition dates.

        Inputs:
        url (str): URL for the granule.

        Returns:
        Datetime object for sorting purposes.
        """
        from datetime import datetime, timedelta

        # Extract filename from URL
        filename = os.path.basename(url)

        # Split into two images
        images = filename.split(ITSCube.SPLIT_IMAGES_TOKEN)
        if len(images) < 2:
            raise RuntimeError(
                f'Filename does not contain expected split token: '
                f'{ITSCube.SPLIT_IMAGES_TOKEN} in {filename}'
            )

        # Parse first image tokens
        tokens_1 = images[0].split(ITSCube.IMAGE_TOKEN)
        # Parse second image tokens
        tokens_2 = images[1].split(ITSCube.IMAGE_TOKEN)

        # Detect sensor type and extract acquisition dates accordingly
        sensor_prefix = tokens_1[0][:5] if len(tokens_1[0]) >= 5 else tokens_1[0]

        if sensor_prefix == 'NISAR':
            # NISAR: acquisition date+time at token[11]
            # Format: YYYYMMDDTHHMMSS (e.g., 20251120T130632)
            date_token_idx = 11
            date_format = "%Y%m%dT%H%M%S"

        elif sensor_prefix.startswith('S1'):
            # Sentinel-1: acquisition date+time at token[5]
            # Format: YYYYMMDDTHHMMSS (e.g., 20200221T095209)
            date_token_idx = 5
            date_format = "%Y%m%dT%H%M%S"

        elif sensor_prefix.startswith('S2'):
            # Sentinel-2: acquisition date+time at token[2]
            # Format: YYYYMMDDTHHMMSS (e.g., 20181008T190459)
            date_token_idx = 2
            date_format = "%Y%m%dT%H%M%S"

        elif sensor_prefix.startswith('L'):
            # Landsat (LC08, LC09, LE07, LT05, etc.): acquisition date at token[3]
            # Format: YYYYMMDD
            date_token_idx = 3
            date_format = ITSCube.DATE_FORMAT

        else:
            # Unsupported sensor format
            raise ValueError(
                f"Unsupported sensor filename format: {sensor_prefix} in {filename}"
            )

        # Parse acquisition dates using the determined token index and format
        try:
            date_1 = datetime.strptime(tokens_1[date_token_idx], date_format)
            date_2 = datetime.strptime(tokens_2[date_token_idx], date_format)
        except IndexError:
            raise RuntimeError(
                f'Missing expected token at index {date_token_idx} for sensor '
                f'{sensor_prefix} in filename: {filename}'
            )
        except ValueError as e:
            raise RuntimeError(
                f'Invalid date format at token {date_token_idx} for sensor '
                f'{sensor_prefix} in filename {filename}: {e}'
            )

        # Calculate mid_date as average
        mid_date = date_1 + (date_2 - date_1) / 2
        return mid_date

    @staticmethod
    def get_tokens_from_filename(filename):
        """
        Extract processing dates for two images from the filename and
        construct unique identifier for the image pair by removing processing
        dates, percent valid pixels fields and file extension.

        Inputs:
        filename (str): Granule filename to parse.

        Returns:
        url_proc_date_1 (datetime): Processing date for first image.
        url_proc_date_2 (datetime): Processing date for second image.
        id (str): Unique identifier for the image pair.
        """
        files = os.path.basename(filename).split(ITSCube.SPLIT_IMAGES_TOKEN)

        # Get acquisition, processing date, path_row for both images
        # from url and index_url
        url_tokens = os.path.basename(files[0]).split(ITSCube.IMAGE_TOKEN)

        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)

        # Remove processing date from the first image name: don't replace date
        # token with an empty string as acquisition and processing dates can be
        # the same
        id_tokens = url_tokens[:4]
        id_tokens.extend(url_tokens[5:])

        url_tokens = os.path.basename(files[1]).split(ITSCube.IMAGE_TOKEN)
        url_proc_date_2 = datetime.strptime(url_tokens[4],
                                            ITSCube.DATE_FORMAT)

        # Remove processing date and _Pxxx.nc from the second image name
        id_tokens.extend(url_tokens[:4])
        id_tokens.extend(url_tokens[5:8])

        id = ITSCube.IMAGE_TOKEN.join(id_tokens)

        return url_proc_date_1, url_proc_date_2, id

    def add_layer(self, is_empty, layer_projection, mid_date, url, data, msgs):
        """
        Examine the layer if it qualifies to be added as a cube layer and
        add it to the cube if it does. If layer is not added, keep track of the
        reason for it to be skipped.

        Inputs:
        is_empty (bool): Flag indicating whether the layer contains valid
                        data for the region of interest.
        layer_projection (str): Projection code for the layer.
        mid_date (datetime): Middle date for the layer.
        url (str): URL for the layer.
        data (xarray.Dataset): Data for the layer.
        msgs (list): List of messages to log for the layer.
        """
        if len(msgs):
            # Log messages for the layer if there are any
            self.logger.info(f"Messages for {url}: {msgs}")

        if data is not None:
            # "Duplicate" granules are handled apriori for newly constructed
            #  cubes (see self.request_granules() method) and for updated
            #  cubes (see self.exclude_processed_granules() method).
            # print(f"Adding {url} for {mid_date}")
            self.dates.append(mid_date)
            self.ds.append(data)
            self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_granules[SkippedGranules.empty].append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_granules[SkippedGranules.projection].setdefault(
                    layer_projection, []
                ).append(url)

    @staticmethod
    def init_output_store(output_dir: str):
        """
        Initialize output store for the datacube. It removes existing local
        store if it exists already. This method is useful only if create_*
        methods are called directly by the user - to guarantee that datacube
        is created from scratch.

        Inputs:
        output_dir (str): Local directory to write datacube Zarr store to.
        """
        # Remove datacube store if it exists
        if os.path.exists(output_dir):
            logging.info(f"Removing existing {output_dir}")
            shutil.rmtree(output_dir)

    @staticmethod
    def exists(output_dir: str, s3_bucket: str):
        """
        Check if datacube exists. The datacube can be on a local file system or
        in AWS S3 bucket.

        Inputs:
        output_dir (str): Local directory or S3 bucket datacube URL to check
                            for datacube existence.
        s3_bucket (str): AWS S3 bucket if datacube Zarr store resides in the
                            cloud.
        """
        cube_exists = False

        cube_path = os.path.join(s3_bucket, output_dir) if len(s3_bucket) \
                    else output_dir

        # Check if the datacube is in the S3 bucket
        if len(s3_bucket):
            s3 = aws_utils.make_s3fs(no_sign_request=ITSCube.NO_AWS_SIGNING)
            cube_glob = s3.glob(cube_path)
            if len(cube_glob):
                cube_exists = True

        else:
            if os.path.exists(cube_path):
                cube_exists = True

        logging.info(f'{cube_path} exists: {cube_exists is True}')
        return cube_exists

    @staticmethod
    def init_input_store(
        input_dir: str,
        s3_bucket: str,
        backup_bucket: str = None,
        read_skipped_granules: bool = True
    ):
        """
        Read datacube from provided store. The method detects if S3 bucket
        store or local Zarr archive is provided, and reads xarray.Dataset
        from the Zarr store. It also reads skipped granules info from the
        corresponding JSON file if it exists.

        Inputs:
        input_dir (str): Zarr store to read existing datacube from. It can be
                            either local directory or S3 bucket directory.
        s3_bucket (str): AWS S3 bucket if datacube Zarr store resides in the
                            cloud.
        backup_bucket (str): AWS S3 bucket directory to write backup of
                            original datacube and JSON file with skipped
                            granules info to.
        read_skipped_granules (bool): If True, read skipped granules info
                                        from the datacube's corresponding
                                        JSON file.
        """
        ds_from_zarr = None
        s3_in = None
        cube_store = None
        skipped_granules = None

        # This is a workaround for:
        # if other than original datacube s3 location was provided for
        # updated cube, then original skipped granules file attributes
        # will still point to the original datacube location for json file.
        # Need to use the same location as the datacube being updated.
        skipped_granules_file = None

        if len(s3_bucket) == 0:
            # Reading from the local directory, check if datacube store exists
            if ITSCube.exists(input_dir, s3_bucket):
                logging.info(f"Reading existing {input_dir}...")

                # Read dataset in
                ds_from_zarr = xr.open_zarr(input_dir,
                                            decode_timedelta=False,
                                            consolidated=True)

                # Read skipped granules info that corresponds to the cube
                if read_skipped_granules:
                    skipped_granules_file = \
                        ds_from_zarr.attrs[SkippedGranules.name]

                    logging.info(
                        f"Reading existing {skipped_granules_file}..."
                    )

                    with open(skipped_granules_file) as skipped_fh:
                        skipped_granules = json.load(skipped_fh)

        elif ITSCube.exists(input_dir, s3_bucket):
            # If datacube is in the AWS S3 bucket
            cube_path = os.path.join(s3_bucket, input_dir)
            logging.info(f"Reading existing {cube_path}")

            # Open S3FS access to S3 bucket with input datacube
            s3_in = aws_utils.make_s3fs(
                max_connections=ITSCube.MAX_AWS_CONNECTIONS,
                no_sign_request=ITSCube.NO_AWS_SIGNING
            )
            cube_store = s3fs.S3Map(root=cube_path, s3=s3_in, check=False)
            ds_from_zarr = xr.open_dataset(
                cube_store,
                decode_timedelta=False,
                engine='zarr',
                consolidated=True
            )
            logging.info(
                f'Dimensions for existing {cube_path}: {ds_from_zarr.dims}'
            )

            if read_skipped_granules:
                skipped_granules_file = cube_path.replace(
                    utils.File.ext.zarr, utils.File.ext.json
                )
                logging.info(
                    f'Cube stores '
                    f'{ds_from_zarr.attrs[SkippedGranules.name]}, but '
                    f'reading skipped granules from {skipped_granules_file}'
                )

                with s3_in.open(skipped_granules_file, 'r') as skipped_fh:
                    skipped_granules = json.load(skipped_fh)

        if ds_from_zarr is None:
            raise RuntimeError(
                f"Provided input datacube {input_dir} does not exist in "
                f"(s3={s3_bucket})"
            )

        # If backup bucket is provided, copy the skipped cube info to the
        # backup s3 bucket directory
        if backup_bucket is not None and read_skipped_granules:
            # Make sure the backup copy does not exist
            json_file = os.path.basename(skipped_granules_file)
            if ITSCube.exists(json_file, backup_bucket):
                logging.info(
                    f'Backup skipped granules file {skipped_granules_file} '
                    f'already exists in {backup_bucket}, skipping copy'
                )

            else:
                env_copy = os.environ.copy()
                logging.info(
                    f"Copying {skipped_granules_file} to {backup_bucket}"
                )

                command_line = [
                    "aws", "s3", "cp",
                    skipped_granules_file,
                    os.path.join(
                        backup_bucket,
                        os.path.basename(skipped_granules_file)
                    ),
                    "--acl", "bucket-owner-full-control"
                ]

                itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

        # Don't use cube_store - keep it in scope only to guarantee valid
        # file-like access.
        return s3_in, cube_store, ds_from_zarr, skipped_granules

    def create_or_update(
        self,
        output_dir: str,
        output_bucket: str,
        backup_bucket: str,
        num_granules=None
    ):
        """
        Create new or update existing datacube.

        Inputs:
        output_dir (str):
            Local datacube Zarr store to write updated datacube to.
        output_bucket (str):
            AWS S3 bucket if datacube Zarr store resides in the cloud.
        backup_bucket (str):
            AWS S3 bucket directory to write backup of original datacube to.
        num_granules (int):
            Number of first granules to examine. This is used for testing only.
        """
        if ITSCube.IGNORE_EXISTING_CUBE is False and \
                ITSCube.exists(output_dir, output_bucket):
            # Datacube exists, update
            self.update_parallel(
                output_dir,
                output_bucket,
                backup_bucket,
                num_granules
            )

        else:
            # Create new datacube
            self.create_parallel(output_dir, output_bucket, num_granules)

    def update_parallel(
        self,
        output_dir: str,
        output_bucket: str,
        backup_bucket: str,
        num_granules=None
    ):
        """
        Update velocity pair datacube by reading and pre-processing new cube
        layers in parallel.

        Inputs:
        output_dir (str):
            Local datacube Zarr store to write updated datacube to.
        output_bucket (str):
            AWS S3 bucket directory path if datacube Zarr store resides
            in the cloud.
        backup_bucket (str):
            AWS S3 bucket directory path to write backup of original
            datacube to.
        num_granules (int):
            Number of first granules to examine.
            TODO: This is a temporary solution for a very long time to open
                remote granules when testing. Should not be used when
                running the code in production mode.
        """
        self.logger.info(
            f"Updating {os.path.join(output_bucket, output_dir)}"
        )

        # Backup skipped granules info to the backup bucket if provided
        # and open existing datacube (if it exists) to update with new layers
        s3, cube_store_in, cube_ds, skipped_granules = \
            ITSCube.init_input_store(
                output_dir,
                output_bucket,
                backup_bucket
            )

        # Update with number of layers in existing datacube
        self.current_cube_layers = cube_ds.dims[utils.Coords.MID_DATE]

        self.date_updated = self.date_created
        self.date_created = cube_ds.attrs[CubeFormat.date_created]

        # When updating existing datacube we need to know maximum number of
        # characters for unicode string data variables. Newer zarr requires
        # dtypes to be consistent: when appending new layers data has to be
        # formatted to the existing on disk dtype.
        if cube_ds is not None:
            # Set autoRIFT attribute for newly appended layers to what
            # is already in the datacube
            self.autoRIFTParamFile = cube_ds.attrs[
                Vars.attrs.autorift_param_file
            ]

            for each in [
                ImgPairInfo.sensor_img1,
                ImgPairInfo.sensor_img2,
                ImgPairInfo.satellite_img1,
                ImgPairInfo.satellite_img2
            ]:
                # Convert dtype to string rep: '<U3' for dtype('<U3')
                dtype_str = str(cube_ds[each].dtype)

                # Extract how many characters
                match = re.match(r"[<>=|]?U(\d+)", dtype_str)
                if match:
                    self.existing_dtypes[each] = int(match.group(1))
                    self.logger.info(
                        f'Extracted dtype for {each}: '
                        f'{self.existing_dtypes[each]} characters'
                    )

                else:
                    self.logger.warning(f'Could not extract dtype for {each}')

        if s3 is None:
            # If input datacube is on the local filesystem, open S3FS for reading
            # granules from S3 bucket
            s3 = aws_utils.make_s3fs(
                max_connections=ITSCube.MAX_AWS_CONNECTIONS,
                no_sign_request=ITSCube.NO_AWS_SIGNING
            )

        self.clear()

        # This will exclude older Landsat8/9 granules that have duplicate
        # mid_date and will update self.skipped_granules[SkippedGranules.duplicate]
        # with such granules.
        found_urls = self.request_granules(num_granules)
        if len(found_urls) == 0:
            return found_urls

        # Remove already processed granules (granules that are already in
        # the datacube and granules that are skipped due to empty data,
        # wrong projection, duplicate mid_date) from the list of found
        # granules for the datacube update.
        # ATTN: cube_layers_to_delete is set to empty list for now as cubes
        # have uplicate mid_date layers which need to be resolved in the
        # future (there are granules with identical middate in the datacube)
        found_urls, cube_layers_to_delete = self.exclude_processed_granules(
            found_urls,
            cube_ds,
            skipped_granules
        )
        num_cube_layers = len(cube_ds.mid_date.values)

        if len(found_urls) == 0:
            self.logger.info("No granules to update with, exiting.")
            return found_urls

        # Clean up the open store for the dataset
        del cube_store_in
        del cube_ds
        gc.collect()

        # If datacube resides in AWS S3 bucket, create backup of latest
        # data variables chunks and copy that backup locally for the update
        if not os.path.exists(output_dir):
            # Copy datacube locally: bring only datacube metadata files and
            # latest chunk for each of the data variables of the cube.
            # To avoid large runtime overhead for the copy, we need to
            # copy only latest chunks that are going to be updated with
            # new data.
            cube_url = os.path.join(output_bucket, output_dir)
            backup_url = os.path.join(backup_bucket, output_dir)

            # Identify "last" chunks for the cube and back them up to
            # the backup directory in s3 bucket if provided
            if ITSCube.exists(output_dir, backup_bucket) and \
                    ITSCube.USE_EXISTING_BACKUP:
                logging.info(
                    f"Backup {output_dir} already exists in {backup_bucket}, "
                    "skipping backup copy."
                )
            else:
                _ = itslive_utils.backup_datacube_latest_chunks(
                    cube_url,
                    backup_url,
                )

            # Download chunks per just created backup (to guarantee that
            # backup copy is a valid copy) before updating the datacube
            # locally
            env_copy = os.environ.copy()

            logging.info(
                f"Copying latest chunks from backup copy in {backup_bucket}"
                f" locally to {os.path.basename(output_dir)}"
            )

            command_line = [
                "aws", "s3", "cp",
                backup_url,
                os.path.basename(output_dir),
                "--recursive",
                "--acl", "bucket-owner-full-control"
            ]

            logging.info(f"Command line: {command_line}")

            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)
            logging.info("Done copying datacube locally.")

            # Make sure cube's metadata exists in the local directory
            for each_meta in itslive_utils.CUBE_META:
                if not os.path.exists(
                    os.path.join(os.path.basename(output_dir), each_meta)
                ):
                    raise RuntimeError(
                        f"Missing {each_meta} in {os.path.basename(output_dir)}"
                        f" after copying from {backup_url}"
                    )

        elif len(output_bucket):
            # datacube exists on local file system even though S3 bucket for
            # the datacube is provided.
            raise RuntimeError(
                f'Local copy of {output_dir} already exists though '
                f'{output_bucket} is provided, remove datacube first '
                'to avoid data overwrite.'
            )

        # Delete identified layers of the cube if any
        is_first_write = False

        # For now this is disabled by setting cube_layers_to_delete to
        # an empty list
        if len(cube_layers_to_delete):
            # For now we need to disable support for deletion of existing
            # layers.
            # The reason is that current datacubes have duplicate "mid_date"
            # layers which need to be resolved in the future if we need to
            # support deletion of existing layers.
            raise RuntimeError(
                'Deletion of existing datacube layers is not supported, exiting...'
            )

            self.logger.info(
                f"Deleting {len(cube_layers_to_delete)} layers from "
                f"total {num_cube_layers} layers of {output_dir}"
            )

            if len(cube_layers_to_delete) == num_cube_layers:
                # If all layers need to be deleted, just delete the cube and
                # start from the scratch
                is_first_write = True
                self.logger.info(f"Deleting existing {output_dir}")
                shutil.rmtree(output_dir)

            else:
                # Delete identified layers
                ds_from_zarr = xr.open_zarr(
                    output_dir,
                    decode_timedelta=False,
                    consolidated=True
                )

                # Identify layer indices that correspond to granule urls
                layers_bool_flag = ds_from_zarr[Vars.url].isin(cube_layers_to_delete)

                # Drop the layers
                # layers_mid_dates = ds_from_zarr[DataVars.MID_DATE].values[layers_bool_flag.values]
                dropped_ds = ds_from_zarr.drop_isel(mid_date=layers_bool_flag.values)

                tmp_output_dir = f"{output_dir}.original"
                self.logger.info(f"Moving original {output_dir} to {tmp_output_dir}")
                os.renames(output_dir, tmp_output_dir)

                # Write updated datacube to original store location,
                # but at first re-chunk xr.Dataset to avoid errors
                dropped_ds = dropped_ds.chunk(
                    {utils.Coords.MID_DATE: ITSCube.NUM_GRANULES_TO_WRITE}
                )

                self.logger.info(f"Saving updated {output_dir}")
                # Should use already existing encoding attributes for the cube
                dropped_ds.to_zarr(
                    output_dir,
                    # encoding=zarr_to_netcdf.ENCODING_ZARR,
                    consolidated=True
                )

                self.logger.info(f"Removing original {tmp_output_dir}")
                shutil.rmtree(tmp_output_dir)

                ds_from_zarr = None
                dropped_ds = None
                gc.collect()

        start = 0
        num_to_process = len(found_urls)

        # For debugging only:
        # num_to_process = 500

        with parallel_config(
            backend='threading',
            n_jobs=ITSCube.MAX_AWS_CONNECTIONS
        ):
            while num_to_process > 0:
                # How many tasks to process at a time
                num_tasks = min(num_to_process, ITSCube.NUM_GRANULES_TO_WRITE)

                # Run in parallel with joblib
                log_msg = f"Processing {num_tasks} tasks out of " \
                            f"{num_to_process} remaining"
                logging.info(log_msg)

                # results = None
                # with tqdm_joblib(tqdm(desc=log_msg, total=num_tasks)):
                results = Parallel()(
                    delayed(ITSCube.read_s3_dataset)(each_file, s3) for
                    each_file in found_urls[start:start + num_tasks]
                )

                # Sort results by mid_date (index 2) for chronological order
                # Use datetime.min for None mid_dates (will be filtered anyway)
                results = sorted(
                    results,
                    key=lambda x: x[2] if x[2] is not None else np.datetime64(datetime.min)
                )

                for each_ds in results:
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

    def create_parallel(
        self,
        output_dir: str,
        output_bucket: str,
        num_granules=None
    ):
        """
        Create velocity pair datacube by reading and pre-processing cube
        layers in parallel.

        Inputs:
        =======
        output_dir (str):       Directory to write datacube to.
        output_bucket (str):    AWS S3 bucket if datacube Zarr store resides
                                in the cloud.
        num_granules (int):     Number of first granules to examine.
                                ATTN: This is a temporary solution to a very
                                long time to open remote granules. Should not
                                be used in production.
        """
        self.logger.info(
            f"Creating {os.path.join(output_bucket, output_dir)}"
        )

        ITSCube.init_output_store(output_dir)

        self.clear()
        found_urls = self.request_granules(num_granules)
        if len(found_urls) == 0:
            return found_urls

        # Parallelize layer collection
        s3 = aws_utils.make_s3fs(
            max_connections=ITSCube.MAX_AWS_CONNECTIONS,
            no_sign_request=ITSCube.NO_AWS_SIGNING
        )

        is_first_write = True
        start = 0
        num_to_process = len(found_urls)

        with parallel_config(
            backend='threading', n_jobs=ITSCube.MAX_AWS_CONNECTIONS
        ):
            while num_to_process > 0:
                # How many tasks to process at a time
                num_tasks = min(num_to_process, ITSCube.NUM_GRANULES_TO_WRITE)

                # Run in parallel with joblib
                log_msg = f"Processing {num_tasks} tasks out of " \
                            f"{num_to_process} remaining"
                logging.info(log_msg)

                # results = None
                # with tqdm_joblib(tqdm(desc=log_msg, total=num_tasks)):
                results = Parallel()(
                    delayed(ITSCube.read_s3_dataset)(each_file, s3) for
                    each_file in found_urls[start:start + num_tasks]
                )

                # Sort results by mid_date (index 2) for chronological order
                # Use datetime.min for None mid_dates (will be filtered anyway)
                results = sorted(
                    results,
                    key=lambda x: x[2] if x[2] is not None else np.datetime64(datetime.min)
                )

                for each_ds in results:
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

    def get_data_var(
        self,
        ds: xr.Dataset,
        var_name: str,
        index: int = 0,
        data_dtype: str = 'short',
        data_fill_value: int = utils.Missing.value
    ):
        """
        Return xr.DataArray that corresponds to the data variable if it exists
        in the 'ds' dataset, or empty xr.DataArray if it is not present in
        the input dataset 'ds'.
        If requested datatype for output data is not of data's original type,
        convert data to the requested type.
        Empty xr.DataArray assumes the same dimensions as input ds.v data
        array.

        Inputs:
        ds (xarray.Dataset):    The dataset the variable belongs to.
        var_name (str):         Name of the variable to extract.
        data_dtype (str):       Datatype to use for the data variable. Default
                                is 'short'.
        data_fill_value (int):   Value to use for filling empty data array if
                                variable is not present in the input dataset
                                'ds'. Default is utils.Missing.value.
        """
        if var_name in ds:
            _dims = [
                d for d in ds[var_name].dims
                if d != utils.Coords.TIME
            ]

            _coords = {
                k: v for k, v in ds[var_name].coords.items()
                if k != utils.Coords.TIME
            }

            if data_dtype and ds[var_name].dtype != np.dtype(data_dtype):
                # Return data of requested type with corresponding
                # "missing_value".
                # Don't preserve "time" dimension from original granule
                return xr.DataArray(
                    data=utils.to_int_type(
                        ds[var_name].values[0, :, :],
                        data_type=np.dtype(data_dtype),
                        fill_value=data_fill_value
                    ),
                    coords=_coords,
                    dims=_dims,
                    attrs=ds[var_name].attrs
                )

            return ds[var_name]

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: Can't use None as data to create xr.DataArray - won't be able
        # to set dtype='short' in encoding for writing to the file.
        return xr.DataArray(
            data=np.full((len(self.grid_y), len(self.grid_x)),
                            data_fill_value, dtype=np.dtype(data_dtype)),
            coords=[self.grid_y, self.grid_x],
            dims=[utils.Coords.Y, utils.Coords.X]
        )

    def get_data_var_float(
        self,
        ds: xr.Dataset,
        var_name: str,
        data_fill_value: int = utils.Missing.value
    ):
        """
        Return xr.DataArray that corresponds to the data variable of floating
        point datatype if it exists in the 'ds' dataset, or an empty
        xr.DataArray if it is not present in the input dataset 'ds'.
        Empty xr.DataArray assumes the same dimensions as input ds.v data
        array.

        Inputs:
        ds (xarray.Dataset):    The dataset the variable belongs to.
        var_name (str):         Name of the variable to extract.
        data_dtype (str):       Datatype to use for the data variable. Default
                                is 'short'.
        data_fill_value (int):   Value to use for filling empty data array if
                                variable is not present in the input dataset
                                'ds'. Default is utils.Missing.value.
        """
        if var_name in ds:
            return ds[var_name][0, :, :].drop_vars(utils.Coords.TIME)

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: Can't use None as data to create xr.DataArray - won't be able
        # to set dtype='short' in encoding for writing to the file.
        return xr.DataArray(
            data=np.full((len(self.grid_y), len(self.grid_x)),
                            data_fill_value, dtype=np.dtype(np.float32)),
            coords=[self.grid_y, self.grid_x],
            dims=[utils.Coords.Y, utils.Coords.X]
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
        ds (xarray.Dataset): The dataset the variable belongs to.
        ds_url (str): URL of the granule that corresponds to the input "ds"
                        dataset (used for error reporting only).
        var_name (str): Name of the variable to extract attribute for.
        attr_name (str): Name of the attribute to extract value for.
        missing_value: Value to use if attribute is missing for the variable.
                        Default is None, which will result in raising an
                        exception if attribute is missing for the variable.
        to_date (bool): Flag if attribute value should be converted to
                        datetime object. Default is False.
        data_dtype: Datatype to use for the attribute value. Default is
                    np.float32.
        """
        if var_name in ds and attr_name in ds[var_name].attrs:
            # NISAR workaround for some attributes being stored as arrays
            # instead of a single value: take the first element of the array
            # if it has only one element.``
            value = ds[var_name].attrs[attr_name]

            # Check if type has "length"
            # if hasattr(type(value), '__len__') and len(value) == 1:
            #     value = value[0]

            if np.ndim(value) != 0:
                # Not a scalar (int, float, or 0-d numpy array)
                # list, tuple, or numpy array.
                value = np.asarray(value).flat[0]

            # print(f"Read value for {var_name}.{attr_name}: {value}")

            if to_date is True:
                try:
                    tokens = value.split('T')
                    if len(tokens) == 3:
                        # Handle malformed datetime in Sentinel 2 granules:
                        # img_pair_info.acquisition_date_img1 = "20190215T205541T00:00:00"
                        value = tokens[0] + 'T' + tokens[1][0:2] + ':' \
                                + tokens[1][2:4] + ':' + tokens[1][4:6]
                        value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

                    elif len(value) >= 8:
                        value = parse(value)

                except ValueError as exc:
                    raise RuntimeError(
                        f"Error converting {value} to date format '%Y%m%d': "
                        f"{exc} for {var_name}.{attr_name} in {ds_url}"
                    )

            else:
                # Convert value to expected datatype
                if data_dtype:
                    value = data_dtype(value)

            # print(f"Return value for {var_name}.{attr_name}: {value}")
            return value

        if missing_value is None:
            # If missing_value is not provided, attribute is expected to exist always
            raise RuntimeError(
                f"{attr_name} is expected within {var_name} for {ds_url}"
            )

        return missing_value

    @staticmethod
    def get_data_var_binary_attr(
        ds: xr.Dataset,
        ds_url: str,
        var_name: str,
        attr_name: str,
        token: str,
        data_dtype=np.uint8,
        missing_value=None
    ):
        """
        Return attribute for the data variable in data set if it exists,
        or missing_value if it is not present.
        If "missing_value" is set to None, than specified attribute is expected
        to exist for the data variable "var_name" and exception is raised if
        it does not.

        Inputs:
        ds (xarray.Dataset): The dataset the variable belongs to.
        ds_url (str): URL of the granule that corresponds to the input "ds"
                        dataset (used for error reporting only).
        var_name (str): Name of the variable to extract attribute for.
        attr_name (str): Name of the attribute to extract value for.
        token (str): Token to use for the attribute value to convert it to
                        binary value. If token is present in the attribute
                        value, then the attribute value is set to 1,
                        otherwise 0.
        missing_value: Value to use if attribute is missing for the variable.
                        Default is None, which will result in raising an
                        exception if attribute is missing for the variable.
        data_dtype: Datatype to use for the attribute value. Default is
                    np.uint8.
        """
        if var_name in ds and attr_name in ds[var_name].attrs:
            # NISAR workaround for some attributes being stored as arrays
            # instead of a single value: take the first element of the array
            # if it has only one element.``
            value = ds[var_name].attrs[attr_name]

            if np.ndim(value) != 0:
                # Not a scalar (int, float, or 0-d numpy array)
                # list, tuple, or numpy array.
                value = np.asarray(value).flat[0]

            value = data_dtype(value == token)

            # print(f"Return value for {var_name}.{attr_name}: {value}")
            return value

        if missing_value is None:
            # If missing_value is not provided, attribute is expected to exist always
            raise RuntimeError(
                f"{attr_name} is expected within {var_name} for {ds_url}"
            )

        return missing_value

    @staticmethod
    def preprocess_dataset(ds: xr.Dataset, ds_url: str):
        """
        Pre-process ITS_LIVE granule dataset in preparation to be added to
        the datacube.

        Inputs:
        ds (xarray.Dataset):    Dataset to pre-process.
        ds_url (str):           URL that corresponds to the dataset.

        Returns:
        cube_v:     Filtered data array for the layer.
        mid_date:   Middle date that corresponds to the velocity pair (uses
                    "time" dimension value of the ITS_LIVE granule which
                    is guaranteed to be unique across multiple granules).
        empty:      Flag to indicate if dataset does not contain any data for
                    the cube region.
        projection: Source projection for the dataset.
        url:        Original URL for the granule (have to return for parallel
                    processing: no track of inputs for each task, but have
                    output available for each task).
        """
        # Tried to load the whole dataset into memory to avoid penalty for
        # random read access when accessing S3 bucket (?) - does not make any
        # difference.
        # ds.load()

        # Flag if layer data is empty
        empty = False

        # Layer data
        mask_data = None

        # Layer middle date
        mid_date = None

        # Granule's projection
        ds_projection = int(ds.mapping.spatial_epsg)

        # Any messages to log during pre-processing of the dataset, e.g. if
        # dataset does not have expected dimensions
        msgs = []

        # Nisar workaround:
        # *_P000.nc don't have time dimension, skip those granules as we
        # won't be able to assign unique mid_date to them
        if utils.Coords.TIME not in ds.dims:
            msgs.append(
                f'{ds_url=} does not have "time" dimension, skipping...'
            )
            empty = True
            return empty, ds_projection, mid_date, ds_url, mask_data, msgs

        # Consider granules that have data only within the target projection
        if str(ds_projection) == ITSCube.PROJECTION:
            # Use granule's "time" dimension value as middle date for the
            # layer. It's guaranteed to be unique across multiple granules.
            mid_date = ds.time.values[0]

            # Define which points are within target polygon.
            mask_lon = (ds.x >= ITSCube.GRID_X_MIN) & (ds.x <= ITSCube.GRID_X_MAX)
            mask_lat = (ds.y >= ITSCube.GRID_Y_MIN) & (ds.y <= ITSCube.GRID_Y_MAX)
            mask = (mask_lon & mask_lat)
            if mask.values.sum() == 0:
                # One or both masks resulted in no coverage
                mask_data = None
                mid_date = None
                empty = True

            else:
                mask_data = ds.where(mask, drop=True)

                # If it's a valid velocity layer, add it to the cube,
                # and skip granules that have only one cell in cube's polygon
                if np.any(mask_data.v.notnull()) and \
                    len(mask_data.x.values) > 1 and \
                    len(mask_data.y.values) > 1:

                    mask_data = mask_data.load()

                    # Verify that granule is defined on the same grid cell size as
                    # expected output datacube.
                    cell_x_size = np.abs(
                        mask_data.x.values[0] - mask_data.x.values[1]
                    )
                    if cell_x_size != ITSCube.CELL_SIZE:
                        raise RuntimeError(
                            f"Unexpected grid cell size ({cell_x_size}) is "
                            f"detected for {ds_url} vs. expected "
                            f"{ITSCube.CELL_SIZE}"
                        )

                else:
                    # Reset cube back to None as it does not contain any
                    # valid data
                    mask_data = None
                    mid_date = None
                    empty = True

        # Have to return URL for the dataset, which is provided as an input
        # to the method, to track URL per granule in parallel processing
        return empty, ds_projection, mid_date, ds_url, mask_data, msgs

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
            'vx_error': (
                "x_velocity_error",
                "error for velocity component in x direction"
            ),
            'vy_error': (
                "y_velocity_error",
                "error for velocity component in y direction"
            ),
            'va_error': (
                "azimuth_velocity_error",
                "error for velocity in radar azimuth direction"
            ),
            'vr_error': (
                "range_velocity_error",
                "error for velocity in radar range direction"
            ),
            # The following descriptions are the same for all v* data
            # variables
            'error_stationary': (
                None,
                "RMSE over stable surfaces, stationary or slow-flowing " \
                "surfaces with velocity < 15 m/yr identified from an " \
                "external mask"
            ),
            'error_slow': (
                None,
                "RMSE over slowest 25% of retrieved velocities"
            ),
            'error_modeled': (
                None,
                "1-sigma error calculated using a modeled error-dt relationship"
            )
        }

        # Possible attributes for the velocity data variable
        _v_comp_attrs = [
            Vars.postfix.error,
            Vars.postfix.error_mask,
            Vars.postfix.error_modeled,
            Vars.postfix.error_slow
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
            # optical legacy format vs. v[xy].v[xy]_error in radar format as
            # these are the same
            error_data = [
                ITSCube.get_data_var_attr(
                    ds, url, var_name, each_attr, utils.Missing.value
                )
                for ds, url in zip(self.ds, self.urls)
            ]

            error_name_desc = f'{each_attr}{_name_sep}' \
                                f'{Vars.attrs.description}'
            desc_str = None
            if var_name in self.ds[0] and \
                    error_name_desc in self.ds[0][var_name].attrs:
                desc_str = self.ds[0][var_name].attrs[error_name_desc]

            elif each_attr in _attrs:
                # If generic description is provided
                desc_str = _attrs[each_attr][1]

            elif error_name in _attrs:
                # If variable specific description is provided
                desc_str = _attrs[error_name][1]

            else:
                raise RuntimeError(
                    f"Unknown description for {error_name} of {var_name}"
                )

            self.layers[error_name] = xr.DataArray(
                data=error_data,
                coords=[mid_date_coord],
                dims=[utils.Coords.MID_DATE],
                attrs={
                    utils.Units.name: utils.Units.m_y,
                    Vars.attrs.std_name: error_name,
                    Vars.attrs.description: desc_str
                }
            )

            # If attribute is propagated as cube's data var attribute,
            # delete it
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

            # If attribute description is in the var's attributes, remove it
            if error_name_desc in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[error_name_desc]

        # These attributes appear for all v* data variables of the granule,
        # capture it only once if it exists
        for each_attr, each_attr_units in zip(
            [
                Vars.flag_stable_shift,
                Vars.stable_count_mask,
                Vars.stable_count_slow
            ],
            [None, utils.Units.count, utils.Units.count]
        ):
            if var_name in self.ds[0] and \
                    each_attr not in self.layers and \
                    each_attr in self.ds[0][var_name].attrs:
                self.layers[each_attr] = xr.DataArray(
                    data=[
                        ITSCube.get_data_var_attr(
                            ds, url, var_name, each_attr, data_dtype=np.int32
                        )
                        for ds, url in zip(self.ds, self.urls)
                    ],
                    coords=[mid_date_coord],
                    dims=[utils.Coords.MID_DATE],
                    attrs={
                        Vars.attrs.std_name: each_attr,
                        Vars.attrs.description: Vars.description[each_attr]
                    }
                )

                # Set units if appropriate
                if each_attr_units is not None:
                    self.layers[each_attr].attrs[utils.Units.name] = each_attr_units

            # Remove attribute if it made it into datacube as original
            # variable attribute
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

        if Vars.attrs.flag_stable_shift_description in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[Vars.attrs.flag_stable_shift_description]

        # Create 'stable_shift' specific to the data variable,
        # for example, 'vx_stable_shift' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, Vars.postfix.stable_shift])
        stable_shift_values = np.array(
            [
                ITSCube.get_data_var_attr(
                    ds,
                    url,
                    var_name,
                    Vars.postfix.stable_shift,
                    utils.Missing.value
                )
                for ds, url in zip(self.ds, self.urls)
            ]
        )

        # Some of the granules have "stable_shift" attribute set to NaN:
        # set them to zero
        nan_stable_shift_values_mask = np.isnan(stable_shift_values)

        if np.sum(nan_stable_shift_values_mask) > 0:
            self.logger.info(
                f'Setting {np.sum(nan_stable_shift_values_mask)} '
                f'stable_shift values to 0 for {var_name}'
            )
            stable_shift_values[nan_stable_shift_values_mask] = 0

        _desc_str = f'applied {var_name} shift calibrated using pixels ' \
                    f'over stable or slow surfaces'
        self.layers[shift_var_name] = xr.DataArray(
            data=stable_shift_values,
            coords=[mid_date_coord],
            dims=[utils.Coords.MID_DATE],
            attrs={
                utils.Units.name: utils.Units.m_y,
                Vars.attrs.std_name: shift_var_name,
                Vars.attrs.description: _desc_str
            }
        )
        return_vars.append(shift_var_name)

        stable_shift_values = None
        gc.collect()

        if Vars.postfix.stable_shift in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[Vars.postfix.stable_shift]

        # Create 'stable_shift_mask' and 'stable_shift_slow' specific to the
        # data variable
        # (for example, 'vx_stable_shift_mask' for 'vx' data variable).
        for each_attr in [
            Vars.postfix.stable_shift_mask,
            Vars.postfix.stable_shift_slow
        ]:
            shift_var_name = _name_sep.join([var_name, each_attr])
            _desc_str = Vars.description[each_attr].format(var_name)
            self.layers[shift_var_name] = xr.DataArray(
                data=[
                    ITSCube.get_data_var_attr(ds, url, var_name, each_attr,
                                                utils.Missing.value)
                    for ds, url in zip(self.ds, self.urls)
                ],
                coords=[mid_date_coord],
                dims=[utils.Coords.MID_DATE],
                attrs={
                    utils.Units.name: utils.Units.m_y,
                    Vars.attrs.std_name: shift_var_name,
                    Vars.attrs.description: _desc_str
                }
            )
            return_vars.append(shift_var_name)

            # If attribute is propagated as cube's vx attribute, delete it
            if each_attr in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[each_attr]

        # Return names of new data variables - to be included into "encoding"
        # settings for writing to the file store.
        return return_vars

    def process_m_attributes(self, var_name: str, mid_date_coord):
        """
        Helper method to clean up attributes for M1[12]-related data variables.
        """
        # Process attributes
        # If attribute is propagated as cube's data var attribute, delete it.
        _name_sep = '_'

        # Need to create new DR_TO_VR_FACTOR data variable
        attr_name = f'{var_name}{_name_sep}{Vars.postfix.dr_to_vr_factor}'

        attr_data = [
            ITSCube.get_data_var_attr(ds, url, var_name,
                                        Vars.postfix.dr_to_vr_factor,
                                        utils.Missing.byte)
            for ds, url in zip(self.ds, self.urls)
        ]

        _desc_str = Vars.description[Vars.postfix.dr_to_vr_factor]
        self.layers[attr_name] = xr.DataArray(
            data=attr_data,
            coords=[mid_date_coord],
            dims=[utils.Coords.MID_DATE],
            attrs={
                Vars.attrs.std_name: attr_name,
                Vars.attrs.description: _desc_str,
                utils.Units.name: utils.Units.m_per_year_pixel
            }
        )

        # Remove attributes from the "parent" variable
        if Vars.postfix.dr_to_vr_factor in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[Vars.postfix.dr_to_vr_factor]

        if Vars.attrs.dr_to_vr_factor_description in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[Vars.attrs.dr_to_vr_factor_description]

        # Remove scale_factor and offset that come with original M11 and M12 data
        # if any
        if utils.OutputFormat.scale_factor in self.layers[var_name].encoding:
            del self.layers[var_name].encoding[utils.OutputFormat.scale_factor]

        if utils.OutputFormat.add_offset in self.layers[var_name].encoding:
            del self.layers[var_name].encoding[utils.OutputFormat.add_offset]

        # Return name of new data variable - to be included into "encoding" settings
        # for writing to the file store.
        return attr_name

    def set_grid_mapping_attr(self, var_name: str):
        """
        Check on existence of "grid_mapping" attribute for the variable, set
        it if not present.

        Inputs:
        var_name: Name of the variable to set "grid_mapping" attribute for.
        """
        if Mapping.attrs.grid_mapping in self.layers[var_name].attrs:
            # Attribute is already set, nothing to do
            return

        self.layers[var_name].attrs[Mapping.attrs.grid_mapping] = Mapping.name

    def combine_layers(self, output_dir, is_first_write=False):
        """
        Combine selected layers into one xr.Dataset object and write (append)
        it to the Zarr store.

        Inputs:
        output_dir (str):       Zarr store to write datacube to.
        is_first_write (bool):   Flag to indicate if it's the first time writing
                                to the datacube (if True, write to a new
                                datacube, if False, append to existing
                                datacube).
        """
        self.layers = {}
        wrote_layers = False

        # Write skipped granules info to local file
        with open(ITSCube.SKIPPED_GRANULES_FILE, 'w') as fh:
            json.dump(self.skipped_granules, fh, indent=3)

        # Construct xarray to hold layers by concatenating layer objects
        # along 'mid_date' dimension
        self.logger.info(
            f'Combine {len(self.urls)} layers to the {output_dir}...'

        )
        if len(self.ds) == 0:
            self.logger.info('No layers to combine, continue')
            return wrote_layers

        wrote_layers = True

        start_time = timeit.default_timer()
        mid_date_coord = pd.Index(self.dates, name=utils.Coords.MID_DATE)

        self.layers = xr.Dataset(
            data_vars={Vars.url: ([utils.Coords.MID_DATE], self.urls)},
            coords={
                utils.Coords.MID_DATE: (
                    utils.Coords.MID_DATE,
                    self.dates,
                    {
                        Vars.attrs.std_name:
                            utils.Coords.STD_NAME[utils.Coords.MID_DATE],
                        Vars.attrs.description:
                            utils.Coords.DESCRIPTION[utils.Coords.MID_DATE]
                    }
                ),
                utils.Coords.X: (
                    utils.Coords.X,
                    self.grid_x,
                    {
                        Vars.attrs.std_name:
                            utils.Coords.STD_NAME[utils.Coords.X],
                        Vars.attrs.description:
                            utils.Coords.DESCRIPTION[utils.Coords.X]
                    }
                ),
                utils.Coords.Y: (
                    utils.Coords.Y,
                    self.grid_y,
                    {
                        Vars.attrs.std_name:
                            utils.Coords.STD_NAME[utils.Coords.Y],
                        Vars.attrs.description:
                            utils.Coords.DESCRIPTION[utils.Coords.Y]
                    }
                )
            },
            attrs={
                utils.OutputFormat.author: CubeFormat.values[utils.OutputFormat.author]
            }
        )

        # Set datacube attribute to capture autoRIFT parameter file
        if self.autoRIFTParamFile is None:
            # If autoRIFT parameter file is not set (meaning we are generating
            # brand new cube), use the first layer's parameter file
            self.autoRIFTParamFile = \
                self.ds[0].attrs[Vars.attrs.autorift_param_file]

        self.layers.attrs[Vars.attrs.autorift_param_file] = \
            self.autoRIFTParamFile

        # Make sure all layers have the same parameter file
        all_values = [
            urlparse(ds.attrs[Vars.attrs.autorift_param_file]).path for ds in
            self.ds
        ]
        unique_values = list(set(all_values))

        if len(unique_values) > 1:
            raise RuntimeError(
                f"Multiple values for '{Vars.attrs.autorift_param_file}' "
                f"are detected for current {len(self.ds)} layers: "
                f"{unique_values}"
            )

        # All layers within datacube must have the same autoRIFT parameter file
        if os.path.basename(self.autoRIFTParamFile) != \
                os.path.basename(unique_values[0]):
            raise RuntimeError(
                f"Inconsistent values for '{Vars.attrs.autorift_param_file}' "
                f"are detected: {self.layers.attrs[Vars.attrs.autorift_param_file]} "
                f"for current {len(self.ds)} layers vs. previously detected "
                f"{unique_values[0]}"
            )

        self.layers.attrs[utils.OutputFormat.conventions] = \
            CubeFormat.values[utils.OutputFormat.conventions]
        self.layers.attrs[CubeFormat.datacube_software_version] = ITSCube.Version
        self.layers.attrs[CubeFormat.date_created] = self.date_created
        self.layers.attrs[CubeFormat.date_updated] = self.date_updated \
            if self.date_updated is not None else self.date_created
        self.layers.attrs[CubeFormat.gdal_area_or_point] = \
            CubeFormat.values[CubeFormat.gdal_area_or_point]
        self.layers.attrs[CubeFormat.geo_polygon] = json.dumps(self.polygon_coords)
        self.layers.attrs[utils.OutputFormat.institution] = \
            CubeFormat.values[utils.OutputFormat.institution]
        self.layers.attrs[utils.OutputFormat.latitude] = round(self.center_lon_lat[1], 2)
        self.layers.attrs[utils.OutputFormat.longitude] = round(self.center_lon_lat[0], 2)
        self.layers.attrs[CubeFormat.proj_polygon] = json.dumps(self.polygon)
        self.layers.attrs[utils.OutputFormat.projection] = str(ITSCube.PROJECTION)
        self.layers.attrs[utils.OutputFormat.s3] = ITSCube.S3

        # Store path to the file with skipped granules (the ones that didn't
        # qualify to make it into the datacube)
        if len(ITSCube.S3):
            # Result datacube is to be stored in S3 bucket, record S3 location
            # of the skipped granules file
            self.layers.attrs[SkippedGranules.name] = ITSCube.S3.replace(
                utils.File.ext.zarr, utils.File.ext.json
            )

        else:
            # Result datacube is to be stored locally, record location of the
            # skipped granules file
            self.layers.attrs[SkippedGranules.name] = output_dir.replace(
                utils.File.ext.zarr, utils.File.ext.json
            )

        # Set time standard as datacube attributes
        for var_name in [
            ImgPairInfo.time_standard_img1,
            ImgPairInfo.time_standard_img2
        ]:
            self.layers.attrs[var_name] = self.ds[0].img_pair_info.attrs[var_name]

            # Make sure all layers have the same time standard
            all_values = [ds.img_pair_info.attrs[var_name] for ds in self.ds]
            unique_values = list(set(all_values))
            if len(unique_values) > 1:
                raise RuntimeError(
                    f"Multiple values for '{var_name}' are detected for "
                    f"current {len(self.ds)} layers: {unique_values}"
                )

        self.layers.attrs[utils.OutputFormat.title] = \
            CubeFormat.values[utils.OutputFormat.title]
        self.layers.attrs[utils.OutputFormat.url] = ITSCube.URL

        # Set attributes for 'url' data variable
        self.layers[Vars.url].attrs[Vars.attrs.std_name] = Vars.url
        self.layers[Vars.url].attrs[Vars.attrs.description] = Vars.description[Vars.url]

        # Set projection information once for the whole datacube
        if is_first_write:
            # Should never happen - just in case as it's a new data format
            if Mapping.name not in self.ds[0]:
                raise RuntimeError(f"Missing {Mapping.name} in {self.urls[0]}")

            # Can't copy the whole data variable, as it introduces obscure coordinates.
            # Just copy all attributes for the scalar type of the xr.DataArray.
            # Use latest granule format: 'mapping' data variable for projection info.
            self.layers[Mapping.name] = xr.DataArray(
                data='',
                attrs=self.ds[0][Mapping.name].attrs,
                coords={},
                dims=[]
            )

            # Set GeoTransform to correspond to the datacube's tile:
            # format cube's GeoTransform
            new_geo_transform_str = f"{self.grid_x[0] - self.half_x_cell} " \
                f"{self.x_cell} 0 {self.grid_y[0] - self.half_y_cell} " \
                f"0 {self.y_cell}"
            self.layers[Mapping.name].attrs[Mapping.attrs.geo_transform] = \
                new_geo_transform_str

            twodim_var_coords = [self.grid_y, self.grid_x]
            twodim_var_dims = [utils.Coords.Y, utils.Coords.X]

            # Create ice masks data variables if they exist
            self.land_ice_mask = utils.to_int_type(
                self.land_ice_mask,
                np.uint8,
                utils.Missing.u8value
            )
            self.layers[shapefile.LANDICE] = xr.DataArray(
                data=self.land_ice_mask,
                coords=twodim_var_coords,
                dims=twodim_var_dims,
                attrs={
                    Vars.attrs.std_name: shapefile.Name[shapefile.LANDICE],
                    Vars.attrs.description: shapefile.Description[shapefile.LANDICE],
                    Mapping.attrs.grid_mapping: Mapping.name,
                    BinaryFlag.attrs.values: BinaryFlag.values,
                    BinaryFlag.attrs.meanings: BinaryFlag.meanings[shapefile.LANDICE],
                    utils.OutputFormat.url: self.land_ice_mask_url
                }
            )
            self.land_ice_mask = None
            gc.collect()

            self.floating_ice_mask = utils.to_int_type(
                self.floating_ice_mask,
                np.uint8,
                utils.Missing.u8value
            )
            # Land ice mask exists for the composite
            self.layers[shapefile.FLOATINGICE] = xr.DataArray(
                data=self.floating_ice_mask,
                coords=twodim_var_coords,
                dims=twodim_var_dims,
                attrs={
                    Vars.attrs.std_name: shapefile.Name[shapefile.FLOATINGICE],
                    Vars.attrs.description: shapefile.Description[shapefile.FLOATINGICE],
                    Mapping.attrs.grid_mapping: Mapping.name,
                    BinaryFlag.attrs.values: BinaryFlag.values,
                    BinaryFlag.attrs.meanings: BinaryFlag.meanings[shapefile.FLOATINGICE],
                    utils.OutputFormat.url: self.floating_ice_mask_url
                }
            )
            self.floating_ice_mask = None
            gc.collect()

        # ATTN: Assign one data variable at a time to avoid running out of
        #       memory. Delete each variable after it has been processed to
        #       free up the memory.

        # Process 'v' (all formats have v variable - variable's attributes
        # are inherited, so no need to set them manually)
        v_layers = xr.concat(
            [
                each_ds.v[0].drop_vars(utils.Coords.TIME) for each_ds in self.ds
            ],
            mid_date_coord
        )

        self.layers[Vars.v] = v_layers
        self.layers[Vars.v].attrs[Vars.attrs.description] = \
            Vars.description[Vars.v]

        #  Collect names of new data variables for the cube
        new_v_vars = [Vars.v]

        # Make sure grid_mapping attribute has the same value for all layers
        grid_mapping_values = [
            ds.mapping.attrs[Mapping.attrs.grid_mapping_name] for ds
            in self.ds
        ]
        unique_values = list(set(grid_mapping_values))
        if len(unique_values) > 1:
            raise RuntimeError(
                f"Multiple '{Mapping.name}' values are detected for current "
                f"{len(self.ds)} layers: {unique_values}"
            )

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [each.drop_vars(Vars.v) for each in self.ds]
        del v_layers
        gc.collect()

        # Process 'v_error'
        self.layers[Vars.v_error] = xr.concat(
            [self.get_data_var(ds, Vars.v_error) for ds in self.ds],
            mid_date_coord
        )
        self.layers[Vars.v_error].attrs[Vars.attrs.std_name] = Vars.name[Vars.v_error]
        self.layers[Vars.v_error].attrs[Vars.attrs.description] = Vars.description[Vars.v_error]
        self.layers[Vars.v_error].attrs[utils.Units.name] = utils.Units.m_y
        self.set_grid_mapping_attr(Vars.v_error)

        new_v_vars.append(Vars.v_error)

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(Vars.v_error) if Vars.v_error in ds
                    else ds for ds in self.ds]
        gc.collect()

        # Process 'v[xy]' data variables and their attributes
        for each_var in [Vars.vx, Vars.vy]:
            self.layers[each_var] = xr.concat(
                [
                    ds[each_var][0].drop_vars(utils.Coords.TIME) for ds in
                    self.ds
                ],
                mid_date_coord
            )
            self.layers[each_var].attrs[Vars.attrs.description] = Vars.description[each_var]
            new_v_vars.append(each_var)
            new_v_vars.extend(self.process_v_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        # Process 'v[ar]' data variables and their attributes
        for each_var in [Vars.va, Vars.vr]:
            self.layers[each_var] = xr.concat(
                [
                    self.get_data_var(ds, each_var, i) for i, ds in
                    enumerate(self.ds)
                ],
            mid_date_coord)
            self.layers[each_var].attrs[Vars.attrs.description] = Vars.description[each_var]
            new_v_vars.append(each_var)
            new_v_vars.extend(self.process_v_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        new_vars_zero_missing_value = []
        # Process 'M1[12]' data variables of radar format, if any, and their attributes
        for each_var in [Vars.m11, Vars.m12]:
            self.layers[each_var] = xr.concat(
                [
                    self.get_data_var_float(ds, each_var) for ds in self.ds
                ],
                mid_date_coord
            )
            self.layers[each_var].attrs[Vars.attrs.std_name] = Vars.name[each_var]
            self.layers[each_var].attrs[Vars.attrs.description] = Vars.description[each_var]
            self.layers[each_var].attrs[utils.Units.name] = utils.Units.pixel_per_m_year
            new_v_vars.append(each_var)
            new_vars_zero_missing_value.append(self.process_m_attributes(each_var, mid_date_coord))

            self.set_grid_mapping_attr(each_var)

            # Drop data variable as we don't need it anymore - free up memory
            self.ds = [ds.drop_vars(each_var) if each_var in ds else ds for ds in self.ds]
            gc.collect()

        # Process chip_size_height: dtype=ushort
        # Optical legacy granules might not have chip_size_height set, use
        # chip_size_width value instead
        self.layers[Vars.chip_size_height] = xr.concat(
            [
                ds.chip_size_height[0].drop_vars(utils.Coords.TIME) if
                np.ma.masked_equal(
                    ds.chip_size_height[0].values,
                    ITSCube.CHIP_SIZE_HEIGHT_NO_VALUE
                ).count() != 0 else
                ds.chip_size_width[0].drop_vars(utils.Coords.TIME) for ds in self.ds
            ],
            mid_date_coord
        )
        self.layers[Vars.chip_size_height].attrs[Vars.attrs.chip_size_coords] = \
            Vars.description[Vars.attrs.chip_size_coords]
        self.layers[Vars.chip_size_height].attrs[Vars.attrs.description] = \
            Vars.description[Vars.chip_size_height]

        self.set_grid_mapping_attr(Vars.chip_size_height)

        # Report if used chip_size_width in place of chip_size_height
        concat_ind = [
            ind for ind, ds in enumerate(self.ds) if
            np.ma.masked_equal(
                ds.chip_size_height[0].values,
                ITSCube.CHIP_SIZE_HEIGHT_NO_VALUE
            ).count() == 0]
        for each in concat_ind:
            self.logger.warning(
                f'Using chip_size_width in place of chip_size_height for '
                f'{self.urls[each]}'
            )

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(Vars.chip_size_height) for ds in self.ds]
        gc.collect()

        # Process chip_size_width: dtype=ushort
        self.layers[Vars.chip_size_width] = xr.concat(
            [
                ds.chip_size_width[0].drop_vars(utils.Coords.TIME) for ds in
                self.ds
            ],
            mid_date_coord
        )
        self.layers[Vars.chip_size_width].attrs[Vars.attrs.chip_size_coords] = \
            Vars.description[Vars.attrs.chip_size_coords]
        self.layers[Vars.chip_size_width].attrs[Vars.attrs.description] = \
            Vars.description[Vars.chip_size_width]

        self.set_grid_mapping_attr(Vars.chip_size_width)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(Vars.chip_size_width) for ds in self.ds]
        gc.collect()

        # Process interp_mask: dtype=ubyte
        self.layers[Vars.interp_mask] = xr.concat(
            [
                ds.interp_mask[0].drop_vars(utils.Coords.TIME) for ds in self.ds
            ],
            mid_date_coord
        )
        self.layers[Vars.interp_mask].attrs[Vars.attrs.std_name] = \
            Vars.name[Vars.interp_mask]
        self.layers[Vars.interp_mask].attrs[Vars.attrs.description] = \
            Vars.description[Vars.interp_mask]
        self.layers[Vars.interp_mask].attrs[BinaryFlag.attrs.values] = \
            BinaryFlag.values
        self.layers[Vars.interp_mask].attrs[BinaryFlag.attrs.meanings] = \
            BinaryFlag.meanings[Vars.interp_mask]

        self.set_grid_mapping_attr(Vars.interp_mask)

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(Vars.interp_mask) for ds in self.ds]
        gc.collect()

        for each in ImgPairInfo.all:
            # Add new variables that correspond to attributes of
            # 'img_pair_info' (only selected ones)
            each_dtype = None
            if each in ImgPairInfo.allTypes:
                each_dtype = ImgPairInfo.allTypes[each]

            # Flag if value should be converted to date type
            convert_to_date = each in ImgPairInfo.toDate

            self.layers[each] = xr.DataArray(
                data=[ITSCube.get_data_var_attr(
                    ds,
                    url,
                    ImgPairInfo.name,
                    each,
                    to_date=convert_to_date,
                    data_dtype=each_dtype
                ) for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[utils.Coords.MID_DATE],
                attrs={
                    Vars.attrs.std_name: ImgPairInfo.stdName[each],
                    Vars.attrs.description: ImgPairInfo.allDescriptions[each]
                }
            )

            if each in ImgPairInfo.allUnits:
                # Units attribute exists for the variable
                self.layers[each].attrs[utils.Units.name] = ImgPairInfo.allUnits[each]

        for (each, new_each) in zip(
            [ImgPairInfo.flight_direction_img1, ImgPairInfo.flight_direction_img2],
            [Vars.ascending_img1, Vars.ascending_img2]
        ):
            # Add new variables that correspond to flight direction attributes
            # of 'img_pair_info'
            self.layers[new_each] = xr.DataArray(
                data=[ITSCube.get_data_var_binary_attr(
                    ds,
                    url,
                    ImgPairInfo.name,
                    each,
                    ImgPairInfo.ascending,
                    data_dtype=np.uint8,
                    missing_value=utils.Missing.u8value
                ) for ds, url in zip(self.ds, self.urls)],
                coords=[mid_date_coord],
                dims=[utils.Coords.MID_DATE],
                attrs={
                    Vars.attrs.std_name: Vars.name[new_each],
                    Vars.attrs.description: Vars.description[new_each],
                    BinaryFlag.attrs.values: BinaryFlag.values,
                    BinaryFlag.attrs.meanings: BinaryFlag.meanings[new_each]
                }
            )

        # Add new variable that corresponds to autoRIFT_software_version
        self.layers[Vars.autorift_software_version] = xr.DataArray(
            data=[ds.attrs[Vars.autorift_software_version] for ds in self.ds],
            coords=[mid_date_coord],
            dims=[utils.Coords.MID_DATE],
            attrs={
                Vars.attrs.std_name: Vars.autorift_software_version,
                Vars.attrs.description: Vars.description[Vars.autorift_software_version]
            }
        )

        # ATTN: Set attributes for the Dataset coordinates as the very last step:
        # when adding data variables that don't have the same attributes for the
        # coordinates, originally set Dataset coordinates will be wiped out
        self.layers[utils.Coords.MID_DATE].attrs = MID_DATE_ATTRS
        self.layers[utils.Coords.X].attrs = X_ATTRS
        self.layers[utils.Coords.Y].attrs = Y_ATTRS

        time_delta = timeit.default_timer() - start_time
        self.logger.info(
            f"Combined {len(self.urls)} layers (took {time_delta} seconds)"
        )

        compressor = zarr.Blosc(
            cname="lz4", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE
        )
        compression = {"compressor": compressor}

        start_time = timeit.default_timer()

        # New version of Zarr requires all granule_url values to be of the
        # same type
        self.layers[Vars.url] = self.layers[Vars.url].astype(
            f'U{ITSCube.MAX_GRANULE_URL_LEN}'
        )

        # Write to the Zarr store
        if is_first_write:
            encoding_settings = {}

            # Make sure chunking is set to full X and Y extends
            encoding_settings.setdefault(utils.Coords.X, {}).update(
                {
                    utils.OutputFormat.compressor: compressor,
                    utils.OutputFormat.chunks: (len(self.layers.x))
                }
            )
            encoding_settings.setdefault(utils.Coords.Y, {}).update(
                {
                    utils.OutputFormat.compressor: compressor,
                    utils.OutputFormat.chunks: (len(self.layers.y))
                }
            )

            # ATTN: Set _FillValue for data variables of floating point data type.
            #       Must set 'missing_value' for data variables on int data type,
            #       otherwise xarray just ignores provided dtype if _FillValue is
            #       provided and assumes floating point type.
            for each in [
                ImgPairInfo.date_dt,
                ImgPairInfo.roi_valid_percentage
            ]:
                encoding_settings[each] = {
                    utils.OutputFormat.dtype: ImgPairInfo.allTypes[each],
                }

            # Set chunking for 2-d variables
            chunking_settings_2d = (len(self.layers.y), len(self.layers.x))

            # Settings for variables of "uint8" data type if any variables exist
            for each in [shapefile.LANDICE, shapefile.FLOATINGICE]:
                encoding_settings.setdefault(each, {}).update({
                    utils.OutputFormat.dtype: shapefile.Type[each],
                    utils.OutputFormat.compressor: compressor,
                    utils.Missing.name: utils.Missing.u8value,
                    utils.OutputFormat.chunks: chunking_settings_2d
                })

            for each in [
                Vars.interp_mask,
                Vars.chip_size_height,
                Vars.chip_size_width,
                Vars.flag_stable_shift,
                Vars.stable_count_slow,
                Vars.stable_count_mask,
                Vars.ascending_img1,
                Vars.ascending_img2
            ]:
                encoding_settings[each] = {
                    utils.OutputFormat.dtype: Vars.intType[each]
                }

                if each in Vars.intMissingValue:
                    encoding_settings[each][utils.Missing.name] = Vars.intMissingValue[each]

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
                missing_value = utils.Missing.value
                missing_value_attr = utils.OutputFormat.fill_value

                dtype_value = np.float32

                if each in Vars.intType:
                    missing_value_attr = utils.Missing.name
                    dtype_value = Vars.intType[each]

                    if each in Vars.intMissingValue:
                        missing_value = Vars.intMissingValue[each]

                encoding_settings[each] = {
                    missing_value_attr: missing_value,
                    utils.OutputFormat.dtype: dtype_value
                }

                encoding_settings[each].update(compression)

            # new_vars_zero_missing_value: ['M11_dr_to_vr_factor', 'M12_dr_to_vr_factor']
            for each in new_vars_zero_missing_value:
                encoding_settings[each] = {
                    utils.OutputFormat.dtype: np.float32,
                    utils.OutputFormat.fill_value: utils.Missing.byte
                }
                encoding_settings[each].update(compression)

            # Explicitly disable _FillValue for some variables: can be set
            # for floating point data variables only.
            # xarray is broken if _FillValue=None is provided along with
            # "chunks" encoding attribute: don't do it.
            # for each in [utils.Coords.MID_DATE,
            #              Vars.stable_count_slow,
            #              Vars.stable_count_mask,
            #              Vars.autorift_software_version,
            #              ImgPairInfo.date_dt,
            #              ImgPairInfo.date_center,
            #              ImgPairInfo.satellite_img1,
            #              ImgPairInfo.satellite_img2,
            #              ImgPairInfo.acquisition_date_img1,
            #              ImgPairInfo.acquisition_date_img2,
            #              ImgPairInfo.roi_valid_percentage,
            #              ImgPairInfo.mission_img1,
            #              ImgPairInfo.mission_img2,
            #              ImgPairInfo.sensor_img1,
            #              ImgPairInfo.sensor_img2]:
            #     encoding_settings.setdefault(each, {}).update({utils.OutputFormat.fill_value: None})

            # Set units for all datetime objects
            for each in [
                ImgPairInfo.acquisition_date_img1,
                ImgPairInfo.acquisition_date_img2,
                ImgPairInfo.date_center,
                utils.Coords.MID_DATE
            ]:
                encoding_settings.setdefault(each, {}).update(
                    {utils.Units.name: utils.Units.date}
                )

            # Set array size to accomodate maximum length of the satellite
            for each in [
                ImgPairInfo.satellite_img1,
                ImgPairInfo.satellite_img2
            ]:
                max_len = max(map(len, self.layers[each].values))
                if max_len > ITSCube.MAX_SATELLITE_LEN:
                    raise RuntimeError(
                        f'"{each}" will be truncated to the current length '
                        f'limit: {ITSCube.MAX_SATELLITE_LEN}: {max_len} '
                        f'length is detected. Please update '
                        f'ITSCube.MAX_SATELLITE_LEN value to proceed.'
                    )

                # Capture max number of characters allowed for the dtype
                self.existing_dtypes[each] = ITSCube.MAX_SATELLITE_LEN

                # Set encoding
                encoding_settings.setdefault(each, {}).update(
                    {utils.OutputFormat.dtype: f'U{ITSCube.MAX_SATELLITE_LEN}'}
                )

            # Set array size to accomodate maximum length of the sensor
            for each in [
                ImgPairInfo.sensor_img1,
                ImgPairInfo.sensor_img2
            ]:
                max_len = max(map(len, self.layers[each].values))
                if max_len > ITSCube.MAX_SENSOR_LEN:
                    raise RuntimeError(
                        f'"{each}" will be truncated to the current length '
                        f'limit: {ITSCube.MAX_SENSOR_LEN}: {max_len} length '
                        f'is detected. Please update '
                        f'ITSCube.MAX_SENSOR_LEN value to proceed.'
                    )

                # Capture max number of characters allowed for the dtype
                self.existing_dtypes[each] = ITSCube.MAX_SENSOR_LEN

                # Set encoding
                encoding_settings.setdefault(each, {}).update(
                    {utils.OutputFormat.dtype: f'U{ITSCube.MAX_SENSOR_LEN}'}
                )

            # Check for the length limit of the granule_url's
            max_url_len = max(map(len, self.layers[Vars.url].values))
            if max_url_len > ITSCube.MAX_GRANULE_URL_LEN:
                raise RuntimeError(
                    f'"{each}" will be truncated to the current length limit: '
                    f'{ITSCube.MAX_GRANULE_URL_LEN}: {max_url_len} length is '
                    f'detected. Please update ITSCube.MAX_GRANULE_URL_LEN '
                    f'value to proceed.'
                )

            encoding_settings.setdefault(Vars.url, {}).update(
                {utils.OutputFormat.dtype: f'U{ITSCube.MAX_GRANULE_URL_LEN}'}
            )

            # Determine optimal chunking for the cube
            chunking_settings_3d = (
                min(self.max_number_of_layers, ITSCube.TIME_CHUNK_VALUE),
                ITSCube.X_Y_CHUNK_VALUE,
                ITSCube.X_Y_CHUNK_VALUE
            )

            # Set chunking for writing to the store
            for each in [
                Vars.interp_mask,
                Vars.chip_size_height,
                Vars.chip_size_width,
                Vars.v,
                Vars.v_error,
                Vars.va,
                Vars.vr,
                Vars.vx,
                Vars.vy,
                Vars.m11,
                Vars.m12
            ]:
                encoding_settings.setdefault(each, {})[utils.OutputFormat.chunks] = \
                    chunking_settings_3d

            chunking_settings_1d = min(
                self.max_number_of_layers,
                ITSCube.TIME_CHUNK_VALUE_1D
            )

            # Create a list of new variables that need to be written to the
            # store and set encoding attributes for them.
            _vars = []
            for each in [Vars.vx, Vars.vy, Vars.va, Vars.vr]:
                _vars.extend([
                    f'{each}_{Vars.postfix.error}',
                    f'{each}_{Vars.postfix.error_mask}',
                    f'{each}_{Vars.postfix.error_modeled}',
                    f'{each}_{Vars.postfix.error_slow}',
                    f'{each}_{Vars.postfix.stable_shift}',
                    f'{each}_{Vars.postfix.stable_shift_slow}',
                    f'{each}_{Vars.postfix.stable_shift_mask}'
                ])

            for each in [Vars.m11, Vars.m12]:
                _vars.append(f'{each}_{Vars.postfix.dr_to_vr_factor}')

            _vars.extend([
                Vars.flag_stable_shift,
                Vars.stable_count_slow,
                Vars.stable_count_mask,
                Vars.autorift_software_version,
                Vars.url,
                ImgPairInfo.acquisition_date_img1,
                ImgPairInfo.acquisition_date_img2,
                ImgPairInfo.roi_valid_percentage,
                ImgPairInfo.satellite_img1,
                ImgPairInfo.satellite_img2,
                ImgPairInfo.mission_img1,
                ImgPairInfo.mission_img2,
                ImgPairInfo.sensor_img1,
                ImgPairInfo.sensor_img2,
                ImgPairInfo.date_center,
                ImgPairInfo.date_dt,
                utils.Coords.MID_DATE,
                Vars.ascending_img1,
                Vars.ascending_img2
            ])

            for each in _vars:
                # Reset existing encoding settings if any for the data variable
                self.layers[each].encoding = {}
                encoding_settings.setdefault(each, {})[utils.OutputFormat.chunks] = \
                    (chunking_settings_1d)

                if utils.OutputFormat.fill_value in self.layers[each].attrs:
                    del self.layers[each].attrs[utils.OutputFormat.fill_value]

                # logging.info(f'Encoding for {each}: {encoding_settings[each]}')
                # logging.info(f'each.attrs for {each}: {self.layers[each].attrs}')
                # logging.info(f'each.encoding for {each}: {self.layers[each].encoding}')

            self.logger.info(f"Encoding writing to Zarr: {encoding_settings}")

            # This is first write, create Zarr store
            self.layers.to_zarr(output_dir, encoding=encoding_settings, consolidated=True)

        else:
            # New version of Zarr requires all unicode string values to be
            # of the same type (length)
            # Set array size for unicode string to the one stored in Zarr on
            # disk
            for each in [
                ImgPairInfo.sensor_img1,
                ImgPairInfo.sensor_img2,
                ImgPairInfo.satellite_img1,
                ImgPairInfo.satellite_img2
            ]:
                # Number of characters in dtype for the data variable in
                # the existing Zarr store on disk
                num_chars = self.existing_dtypes[each]
                dtype_str = f'U{num_chars}'

                _values = self.layers[each].values
                max_len = max(map(len, _values))
                if max_len > num_chars:
                    # Find which values exceed the current dtype's number
                    # of characters
                    mask = np.char.str_len(_values) > num_chars
                    raise RuntimeError(
                        f'"{each}" would be truncated to the current '
                        f'length limit {num_chars=}: {set(_values[mask])} '
                        f'new values are detected in layers to append.'
                    )

                self.layers[each] = self.layers[each].astype(dtype_str)

            # Append layers to existing Zarr store
            self.layers.to_zarr(
                output_dir,
                append_dim=utils.Coords.MID_DATE,
                consolidated=True
            )

        time_delta = timeit.default_timer() - start_time
        self.logger.info(
            f"Wrote {len(self.urls)} layers to {output_dir} "
            f"(took {time_delta} seconds)"
        )

        # Free up memory
        self.clear_vars()

        # Return a flag if any layers were written to the store
        return wrote_layers

    def format_stats(self):
        """
        Format statistics of the run. Don't display statistics if using
        granules as provided in the input JSON file.
        """
        # Granules list for processing was provided in the input JSON file,
        # so no need to display statistics
        if ITSCube.USE_GRANULES is not None:
            return

        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum(
            [len(each) for each in
                self.skipped_granules[SkippedGranules.projection].values()]
        )

        self.logger.info(
            f"Skipped granules due to empty data: "
            f"{len(self.skipped_granules[SkippedGranules.empty])} "
            f"({100.0 * len(self.skipped_granules[SkippedGranules.empty]) / num_urls}%)"
        )

        self.logger.info(
            f"Skipped granules due to double mid_date: "
            f"{len(self.skipped_granules[SkippedGranules.duplicate])} "
            f"({100.0 * len(self.skipped_granules[SkippedGranules.duplicate]) / num_urls}%)"
        )

        self.logger.info(
            f"Skipped granules due to wrong projection: {sum_projs} "
            f"({100.0 * sum_projs / num_urls}%)"
        )

        if len(self.skipped_granules[SkippedGranules.projection]):
            self.logger.info(
                f"Skipped wrong projections: "
                f"{sorted(self.skipped_granules[SkippedGranules.projection].keys())}"
            )

    @itslive_utils.retry_decorator(max_retries=5)
    @staticmethod
    def read_s3_dataset(
        each_url: str,
        s3: s3fs.S3FileSystem,
    ):
        """
        Read Dataset from the S3 bucket and pre-process it for the cube layer.

        Inputs:
        each_url (str): Granule S3 URL.
        s3 (s3fs.S3FileSystem): S3FileSystem object to access the granule.

        Returns:
        Tuple from preprocess_dataset(): (empty, ds_projection, mid_date, ds_url, mask_data, msgs).
        """
        s3_path = each_url.replace(utils.HTTP_PREFIX, utils.S3_PREFIX)
        s3_path = s3_path.replace(utils.PATH_URL, '')

        with s3.open(s3_path, mode='rb') as fhandle:
            with xr.open_dataset(
                fhandle, engine=ITSCube.NC_ENGINE
            ) as ds:
                return ITSCube.preprocess_dataset(ds, each_url)

    @staticmethod
    def validate_cube(ds: xr.Dataset, start_date: str, cube_url: str):
        """
        Validate just written to the local disk datacube. This method is
        introduced because of observed corrupted datacube properties:
        1. Validate X and Y coordinates values: not to include NaN's.
        2. Validate datetime objects of the cube against start_date of the
            cube.

        This check is introduced to capture corrupted datacubes as early as
        possible in the cube generation.
        """
        logging.info(f"Validating X and Y coordinates for {cube_url}")
        if np.any(np.isnan(ds.x.values)):
            raise RuntimeError(
                f'Detected NaNs in X: {cube_url} ds.size={ds.sizes}'
            )

        if np.any(np.isnan(ds.y.values)):
            raise RuntimeError(
                f'Detected NaNs in Y: {cube_url} ds.size={ds.sizes}'
            )

        # ATTN: This checking assumes that start_date corresponds to the start
        # date of the data used to create the datacube
        start_date = np.datetime64(start_date)
        logging.info(f"Validating datetime objects for {cube_url}")

        values = ds.acquisition_date_img1.values
        if values.min() < start_date:
            raise RuntimeError(
                f"Unexpected acquisition_date_img1: {values.min()}"
            )

        values = ds.acquisition_date_img2.values
        if values.min() < start_date:
            raise RuntimeError(
                f"Unexpected acquisition_date_img2: {values.min()}"
            )

        values = ds.date_center.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected date_center: {values.min()}")

        values = ds.mid_date.values
        if values.min() < start_date:
            raise RuntimeError(f"Unexpected mid_date: {values.min()}")

    @staticmethod
    def remove_s3_datacube(
        cube_store: str,
        skipped_granules_file: str,
        s3_bucket: str
    ):
        """
        Remove Zarr store and corresponding json file (with records of skipped
        granules for the cube) in S3 if they exists - this is done to replace
        existing cube with newly generated one:
            * at the beginning of the processing if --removeExistingCube
            command-line option is provided
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
                "aws", "s3", "rm", "--recursive", "--quiet",
                cube_s3_path
            ]
            logging.info(
                f'Removing existing cube {cube_s3_path}: '
                f'{" ".join(command_line)}'
            )

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(
                    f"Failed to remove original {cube_s3_path}: "
                    f"{command_return.stdout}"
                )

            json_s3_path = os.path.join(s3_bucket, skipped_granules_file)

            command_line = [
                "aws", "s3", "rm", "--quiet",
                json_s3_path
            ]
            logging.info(
                f'Removing existing skipped granules json {json_s3_path}: '
                f'{" ".join(command_line)}'
            )

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )

            if command_return.returncode != 0:
                raise RuntimeError(
                    f"Failed to remove original {json_s3_path}: "
                    f"{command_return.stdout}"
                )


if __name__ == '__main__':
    import argparse
    import warnings
    import sys

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=ITSCube.__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=8,
        help='Number of threads to use for parallel processing [%(default)d].'
    )
    parser.add_argument(
        '-r', '--removeExistingCube',
        action='store_true',
        default=False,
        help='Flag to remove existing datacube in S3 bucket, '
            'default is to update existing datacube. '
            'This flag is useful when we need to re-create the cube from '
            'scratch, though beware of AWS limit of push requests '
            'when multiple datacubes are deleted at the same time.'
    )
    parser.add_argument(
        '-n', '--numberGranules',
        type=int,
        default=None,
        help='Number of ITS_LIVE granules to consider for the cube (due to '
            'runtime limitations). If none is provided, process all found '
            'granules.'
    )
    parser.add_argument(
        '-bc', '--numberBackupChunks',
        type=int,
        default=500,
        help='Number of Zarr chunks to backup in parallel when updating '
            'existing datacube residing in s3 bucket [%(default)d].'
    )
    parser.add_argument(
        '-stacCatalog',
        type=str,
        default='s3://its-live-data/test-space/stac/geoparquet/h3r2',
        help='ITS_LIVE granule STAC catalog to request granules from '
            '[%(default)s].'
    )
    parser.add_argument(
        '-o', '--outputStore',
        type=str,
        default="cubedata.zarr",
        help='Zarr full path to write cube data to [%(default)s].'
    )
    parser.add_argument(
        '-tb', '--targetBucket',
        type=str,
        default=None,
        help='Target s3 full path to write cube data to if it should be other '
            'than original "outputBucket" s3 location [%(default)s]. '
            'This is used when datacubes are being updated and original '
            ' datacubes should be preserved, serving as a temporary cube '
            'location.'
    )
    parser.add_argument(
        '-b', '--outputBucket',
        type=str,
        default='',
        help='S3 bucket to copy Zarr format of the datacube to '
                '(for example, s3://its-live-data) [%(default)s].'
    )
    parser.add_argument(
        '-bb', '--backupBucket',
        type=str,
        default=None,
        help='S3 bucket directory to backup original cube latest chunks to '
            'before any updates (for example, datacubes/backup/YYYY-MM-DD) '
            '[%(default)s].'
    )
    parser.add_argument(
        '-c', '--chunks',
        type=int,
        default=250,
        help='Number of granules to write at a time [%(default)d].'
    )
    parser.add_argument(
        '-e', '--encodingTimeChunk',
        type=int,
        default=20000,
        help='Encoding time chunk size to use when storing Zarr cube [%(default)d].'
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
        help='Define 5 points per side before re-projecting granule polygon '
            'to longitude/latitude coordinates'
    )
    parser.add_argument(
        '--ignoreExistingCube',
        action='store_true',
        default=False,
        help='Ignore existing cube for the run. This is to overwrite '
                'any existing cube with the newly generated one without the '
                'need to remove it manually [%(default)s].'
    )
    parser.add_argument(
        '--useExistingCubeBackup',
        action='store_true',
        help='Use datacube backup copy for the update if it exists already.'
    )
    parser.add_argument(
        '--noAWSSigning',
        action='store_true',
        default=False,
        help='Use no AWS signing for S3 requests. If set, requests will be '
            'unsigned (anon=True) which should be used for public buckets '
            '[%(default)d].'
    )
    parser.add_argument(
        '--useGranulesFile',
        type=str,
        default=None,
        help='Json file that stores a list of ITS_LIVE image velocity granules '
            'to build datacube from [%(default)s].'
    )
    parser.add_argument(
        '--searchAPIStartDate',
        type=lambda s: parse(s).strftime('%Y-%m-%d'),
        default='1982-01-01',
        help='Start date in YYYY-MM-DD format to pass to search API query '
            'to get velocity pair granules [%(default)s]'
    )
    parser.add_argument(
        '--searchAPIStopDate',
        action='store',
        type=lambda s: parse(s).strftime('%Y-%m-%d'),
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Stop date in YYYY-MM-DD format to pass to search API query '
            'to get velocity pair granules. Use "now" if not provided [default: %(default)s]'
    )
    parser.add_argument(
        '--disableCubeValidation',
        action='store_true',
        default=False,
        help='Disable datetime validation for created datacube. '
            'This is to identify corrupted Zarr stores at the time of creation.'
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
        default=utils.PATH_URL,
        help='Path URL token to remove from each of the input granules URLs '
            'to allow S3 access [%(default)s].'
    )

    # One of --centroid or --polygon options is allowed for the datacube coordinates
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--centroid',
        type=str,
        action='store',
        help='JSON 2-element list for centroid point (x, y) of the datacube in '
            'target EPSG code projection. '
            'Polygon vertices are calculated based on the centroid and '
            'cube dimension arguments.'
    )
    group.add_argument(
        '--polygon',
        type=str,
        action='store',
        help='JSON list of polygon points [[x1, y1], [x2, y2],... [x1, y1]] to '
            'define datacube in target EPSG code projection.'
    )

    args = parser.parse_args()

    # Check if target location of the generated/updated datacube is different from original cube location
    target_bucket = args.targetBucket if args.targetBucket is not None \
                    else args.outputBucket
    logging.info(f'Target s3 bucket location is set to {target_bucket}')

    # Enforce .zarr file extension for the datacube store
    if not args.outputStore.endswith(utils.File.ext.zarr):
        raise RuntimeError(
            f'Output Zarr store is expected to have {utils.File.ext.zarr} '
            f'extension, got {args.outputStore}'
        )

    ITSCube.MAX_AWS_CONNECTIONS = args.threads
    ITSCube.NUM_CHUNKS_TO_BACKUP = args.numberBackupChunks
    ITSCube.USE_EXISTING_BACKUP = args.useExistingCubeBackup
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks
    ITSCube.CELL_SIZE = args.gridCellSize
    utils.PATH_URL = args.pathURLToken
    ITSCube.STAC_CATALOG = args.stacCatalog
    ITSCube.START_DATE = args.searchAPIStartDate
    ITSCube.END_DATE = args.searchAPIStopDate
    ITSCube.NO_AWS_SIGNING = args.noAWSSigning
    ITSCube.IGNORE_EXISTING_CUBE = args.ignoreExistingCube
    ITSCube.TIME_CHUNK_VALUE = args.encodingTimeChunk
    ITSCube.TIME_CHUNK_VALUE_1D = args.encodingTimeChunk * 10

    if args.useGranulesFile:
        # Check for this option first as another mutually exclusive option has a default value
        if ITSCube.S3_PREFIX in args.useGranulesFile:
            # File is in s3 bucket
            s3 = s3fs.S3FileSystem(anon=True)
            granules_file = args.useGranulesFile.replace(ITSCube.S3_PREFIX, '')

            with s3.open(granules_file, 'r') as ins3file:
                ITSCube.USE_GRANULES = json.load(ins3file)

        else:
            granules_file = Path(args.useGranulesFile)
            ITSCube.USE_GRANULES = json.loads(granules_file.read_text())

        logging.info(
            f'Using {len(ITSCube.USE_GRANULES)} granules as provided in '
            f'{args.useGranulesFile} file'
        )

    if len(args.outputBucket):
        # S3 bucket is provided, format S3 path to the target datacube
        ITSCube.S3 = os.path.join(args.outputBucket, args.outputStore)
        logging.info(f'Cube S3: {ITSCube.S3}')

        # URL is valid only if output S3 bucket is provided
        ITSCube.URL = ITSCube.S3.replace(
            utils.S3_PREFIX,
            utils.HTTP_PREFIX
        )
        url_tokens = urlparse(ITSCube.URL)
        ITSCube.URL = url_tokens._replace(
            netloc=url_tokens.netloc+utils.PATH_URL
        ).geturl()
        logging.info(f'Cube URL: {ITSCube.URL}')

    else:
        ITSCube.S3 = ''
        ITSCube.URL = ''

    # Set local file path for skipped granules info
    ITSCube.SKIPPED_GRANULES_FILE = args.outputStore.replace(
        utils.File.ext.zarr,
        utils.File.ext.json
    )

    if args.removeExistingCube and len(args.outputBucket):
        # Remove Zarr store in S3 if it exists - this is done to replace existing
        # cube with brand new generated one (to avoid update of the existing in s3 datacube)
        ITSCube.remove_s3_datacube(
            args.outputStore,
            ITSCube.SKIPPED_GRANULES_FILE,
            args.outputBucket
        )

    # Read shape file with ice masks information in
    ITSCube.SHAPE_FILE = shapefile.read_file(args.shapeFile)

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

    cube.create_or_update(
        args.outputStore,
        args.outputBucket,
        args.backupBucket,
        args.numberGranules)

    cube = None
    gc.collect()

    # Debugging only: don't remove local copy of the cube and don't copy to s3
    # sys.exit()

    try:
        if not args.disableCubeValidation and os.path.exists(args.outputStore):
            with xr.open_zarr(
                args.outputStore,
                decode_timedelta=False,
                consolidated=True
            ) as ds:
                ITSCube.validate_cube(ds, args.searchAPIStartDate, args.outputStore)

            gc.collect()

        if len(target_bucket):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy

            # TODO: introduce another CLI option if ever need to do it, for now disable it
            # This should be done only when Zarr chunking of the existing (in S3 bucket)
            # datacube is changed
            # remove_original_datacube = False

            # if remove_original_datacube:
            #     # Remove Zarr store in S3 if it exists: updated Zarr, which is stored to the
            #     # local file system before copying to the S3 bucket, might have different
            #     # "sub-directory" structure. This will result in original "sub-directories"
            #     # and "new" ones to co-exist for the same Zarr store. This doubles up
            #     # the Zarr disk usage in S3 bucket.
            #     ITSCube.remove_s3_datacube(args.outputStore, ITSCube.SKIPPED_GRANULES_FILE, args.outputBucket)

            env_copy = os.environ.copy()

            results_files = None
            if os.path.exists(args.outputStore):
                # Local copy of the datacube exists, specify which files need to copy to the target s3 location
                results_files = [args.outputStore, ITSCube.SKIPPED_GRANULES_FILE]

            elif ITSCube.exists(args.outputStore, args.outputBucket) and \
                    (args.outputBucket != target_bucket):
                # Check if original datacube exists - since local copy
                # doesn't exist, but target s3 location is specified,
                # it's one of the cases:
                # * cube was not generated
                # * it was an update to existing datacube and there were no
                #   new granules to update it with (no local copy
                #   of the cube exists).
                # If target s3 location is other than original s3 location,
                # then just copy the cube to new location
                results_files = [
                    os.path.join(args.outputBucket, args.outputStore),
                    os.path.join(args.outputBucket, ITSCube.SKIPPED_GRANULES_FILE)
                ]

            logging.info(
                f'Identified files to copy to the {target_bucket}: '
                f'{results_files}'
            )

            if results_files is not None:
                # Allow for multiple retries to avoid AWS triggered errors
                for each_input, each_recursive_option, each_validate_flag in zip(
                    results_files,
                    [True, False],  # recursive option for copy
                    [True, False]   # flag if need to validate the store once it's copied over to the s3 target location
                ):
                    command_line = ["aws", "s3", "cp"]

                    if each_recursive_option:
                        command_line.append('--recursive')

                    command_line.extend([
                        each_input,
                        os.path.join(target_bucket, os.path.basename(each_input)),
                        "--acl", "bucket-owner-full-control"
                    ])

                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

                    if not args.disableCubeValidation and each_validate_flag:
                        # Validate just copied to S3 datacube
                        s3_in, cube_store, ds_from_zarr, _ = \
                            ITSCube.init_input_store(
                                os.path.basename(each_input),
                                target_bucket,
                                read_skipped_granules=False
                            )
                        ITSCube.validate_cube(
                            ds_from_zarr,
                            args.searchAPIStartDate,
                            os.path.join(target_bucket, each_input)
                        )

    finally:
        # Remove locally written Zarr store.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        if len(target_bucket) and os.path.exists(args.outputStore):
            logging.info(f'Removing local copy of {args.outputStore}')
            shutil.rmtree(args.outputStore)

        # Remove locally skipped granules info file.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        if len(target_bucket) and len(ITSCube.SKIPPED_GRANULES_FILE) and \
                os.path.exists(ITSCube.SKIPPED_GRANULES_FILE):
            logging.info(f'Removing local copy of {ITSCube.SKIPPED_GRANULES_FILE}')
            os.unlink(ITSCube.SKIPPED_GRANULES_FILE)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')

    logging.info('Done.')
