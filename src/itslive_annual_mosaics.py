"""
ITSLiveMosaics class creates yearly mosaics of ITS_LIVE datacubes for the region.

Command examples
================
* Generate mosaics for ALA region:
 ** use cube definitions from tools/catalog_datacubes_v02.json
 ** use only cubes which center point falls within region polygon as defined by
   aws/regions/Alaska.geojson
 ** create mosaics in 3413 EPSG projection code

python ./itslive_annual_mosaics.py -c tools/catalog_datacubes_v02.json
    --processCubesWithinPolygon aws/regions/Alaska.geojson -e '[3413]'
    --mosaicsEpsgCode 3413 -r Alaska

python ./itslive_annual_mosaics.py -c ~/itslive_data/catalog_v2_Alaska.json
    --processCubesWithinPolygon aws/regions/Alaska.geojson -r ALA
    -e '[3413]' --mosaicsEpsgCode 3413

* Generate mosaics for HMA region:
 ** use cube definitions from tools/catalog_datacubes_v02.json
 ** use only cubes as listed in HMA_datacubes.json (see extract_region_cubes.py
   helper script to extract cubes for the region)
 ** create mosaics in 102027 ESRI projection code
python ./itslive_annual_mosaics.py -c tools/catalog_datacubes_v02.json
    --processCubesFile HMA_datacubes.json -r HMA
    --mosaicsEpsgCode 102027

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Mark Fahnestock (UAF)
"""
import collections
import copy
import datetime
import gc
import json
import logging
import math
import numba as nb
import numpy  as np
import os
from osgeo import osr
from pathlib import Path
import pandas as pd
import s3fs
from shapely import geometry
from shapely.ops import unary_union
import subprocess
import timeit
from tqdm import tqdm
import xarray as xr

# Local imports
from grid import Bounds
import itslive_utils
from itscube_types import \
    Coords, \
    DataVars, \
    BatchVars, \
    CubeJson, \
    FilenamePrefix, \
    annual_mosaics_filename_nc, \
    composite_filename_zarr, \
    summary_mosaics_filename_nc
from itscube import CubeOutputFormat
from itslive_composite import CompDataVars, CompOutputFormat
from reproject_mosaics import main as reproject_main
from reproject_mosaics import ESRICode, ESRICode_Proj4

from itslive_composite import CENTER_DATE


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

class MosaicsOutputFormat:
    """
    Data attributes specific to mosaics datasets.
    """
    COMPOSITES_S3 = 'composites_s3'
    COMPOSITES_URL = 'composites_url'
    COMPOSITES_CREATED = 'composites_created'
    COMPOSITES_UPDATED = 'composites_updated'
    MOSAICS_SOFTWARE_VERSION = 'mosaics_software_version'

    # Mapping of mosaics and composites attributes: some of composites attributes
    # will appear with names specific to mosaic.
    ATTR_MAP = {
        CubeOutputFormat.DATE_CREATED: COMPOSITES_CREATED,
        CubeOutputFormat.DATE_UPDATED: COMPOSITES_UPDATED,
        CubeOutputFormat.S3:           COMPOSITES_S3,
        CubeOutputFormat.URL:          COMPOSITES_URL
    }

    REGION = 'region'
    YEAR = 'year'

    ALL_ATTR = [
        CompOutputFormat.COMPOSITES_SOFTWARE_VERSION,
        CubeOutputFormat.DATACUBE_SOFTWARE_VERSION,
        COMPOSITES_CREATED,
        COMPOSITES_UPDATED,
        CompOutputFormat.DATACUBE_CREATED,
        CompOutputFormat.DATACUBE_S3,
        CompOutputFormat.DATACUBE_UPDATED,
        CompOutputFormat.DATACUBE_URL,
        CubeOutputFormat.GEO_POLYGON,
        CubeOutputFormat.PROJ_POLYGON,
        COMPOSITES_S3,
        COMPOSITES_URL
    ]

    ANNUAL_TITLE = 'ITS_LIVE annual mosaics of image pair velocities'
    STATIC_TITLE = 'ITS_LIVE summary mosaics of image pair velocities'

def repr_composite(composites):
    """
    Representation for the composite.
    """
    composites_repr = {}
    for each_file, each_ds in composites.items():
        composites_repr[each_file] = {
            'x': [np.min(each_ds.x), np.max(each_ds.x)],
            'y': [np.max(each_ds.y), np.min(each_ds.y)]
        }

    return composites_repr

class ITSLiveAnnualMosaics:
    """
    Class to build annual mosaics based on composites for ITS_LIVE datacubes.
    """
    VERSION = '1.0'

    FILE_VERSION = 'v02'

    # "structure" to hold composites information for a single cube
    Composite = collections.namedtuple("Composite", ['s3', 'x', 'y', 'time', 'sensor'])

    # "structure" to store xarray.Dataset and correspsoinding s3_store for loading
    # of the data from AWS S3 bucket
    CompositeS3 = collections.namedtuple("CompositeS3", ['ds', 'ds_store'])

    # "structure" to hold composites information for a single cube
    Mosaic = collections.namedtuple("Mosaic", ['s3', 'x', 'y', 'time', 'sensor'])

    # S3 store location for mosaics
    S3 = ''

    # URL location for mosaics
    URL = ''

    CELL_SIZE = 120

    # Chunk size to use for writing to NetCDF file
    # (otherwise running out of memory if attempting to write the whole dataset to the file)
    CHUNK_SIZE = 5000

    REGION = None

    # Postfixes of the files to write collected region information
    COMPOSITES_CENTERS_FILE = '_composites_center_coords.json'
    COMPOSITES_RANGES_FILE = '_composites_x_y_ranges.json'

    # File to store transformation matrix and index mapping to original grid
    # (generate once per each EPSG and re-use to create all mosaics for the region)
    TRANSFORMATION_MATRIX_FILE = ''

    # Flag to use existing EPSG mosaics files if they exist (useful for multi-EPSG code
    # mosaics when need to pick up from where previous processing stopped)
    USE_EXISTING_FILES = False

    # Key into dictionary of generated mosaics files for the static mosaic (since
    # it does not have a year associated with it - just use its filename token)
    SUMMARY_KEY = '0000'

    # Data variables for summary mosaics
    SUMMARY_VARS = [
        CompDataVars.COUNT0,
        CompDataVars.SLOPE_V,
        CompDataVars.SLOPE_VX,
        CompDataVars.SLOPE_VY,
        CompDataVars.OUTLIER_FRAC,
        CompDataVars.SENSOR_INCLUDE,
        CompDataVars.MAX_DT,
        CompDataVars.V0,
        CompDataVars.VX0,
        CompDataVars.VY0,
        CompDataVars.V0_ERROR,
        CompDataVars.VX0_ERROR,
        CompDataVars.VY0_ERROR,
        CompDataVars.V_AMP,
        CompDataVars.VX_AMP,
        CompDataVars.VY_AMP,
        CompDataVars.V_AMP_ERROR,
        CompDataVars.VX_AMP_ERROR,
        CompDataVars.VY_AMP_ERROR,
        CompDataVars.V_PHASE,
        CompDataVars.VX_PHASE,
        CompDataVars.VY_PHASE
    ]

    # Data variables for annual mosaics
    ANNUAL_VARS = [
        CompDataVars.COUNT,
        DataVars.VX,
        DataVars.VY,
        DataVars.V,
        CompDataVars.V_ERROR,
        CompDataVars.VX_ERROR,
        CompDataVars.VY_ERROR,
    ]

    # Attributes that need to propagate from composites to mosaics
    ALL_ATTR = [
        CompOutputFormat.COMPOSITES_SOFTWARE_VERSION,
        CubeOutputFormat.DATACUBE_SOFTWARE_VERSION,
        CubeOutputFormat.DATE_CREATED,
        CubeOutputFormat.DATE_UPDATED,
        CompOutputFormat.DATACUBE_CREATED,
        CompOutputFormat.DATACUBE_S3,
        CompOutputFormat.DATACUBE_UPDATED,
        CompOutputFormat.DATACUBE_URL,
        CubeOutputFormat.GEO_POLYGON,
        CubeOutputFormat.PROJ_POLYGON,
        CubeOutputFormat.S3,
        CubeOutputFormat.URL
    ]

    def __init__(self, epsg: int, is_dry_run: bool):
        """
        Initialize object.

        epsg: Target EPSG code to create mosaics for.
        grid_size: Grid size for mosaics.
        is_dry_run: Flag to display steps to be taken without actually generating
                    mosaics.
        """
        self.epsg = epsg
        self.grid_size_str = f'{ITSLiveAnnualMosaics.CELL_SIZE:03d}'
        self.is_dry_run = is_dry_run

        # Read datacube composites from S3 bucket
        self.s3 = s3fs.S3FileSystem(anon=True)

        # Identified composites for annual mosaics in form of
        # EPSG: {Y: {X: composite_file}}
        self.composites = {}

        # Opened (composites or mosaics) xr.Dataset objects for the currently
        # processed EPSG code
        self.raw_ds = {}

        # Mapping xr.DataArray
        self.mapping = None

        # "united" coordinates for mosaics
        self.time_coords = []
        self.x_coords = []
        self.y_coords = []
        self.sensor_coords = []

        # Common attributes for all mosaics
        self.attrs = {}

        # Date when mosaics were created/updated
        self.date_created = datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')

    def create(
        self,
        cubes_file: str,
        s3_bucket: str,
        cube_dir: str,
        composite_dir: str,
        mosaics_dir: str
    ):
        """
        Create annual mosaics from datacubes composites for specified region.

        cubes_file: GeoJson catalog file with existing datacubes' Zarr S3 URLs.
        s3_bucket: S3 bucket that stores all data (assumes that all datacubes,
                   composites, and mosaics are stored in the same bucket).
        cube_dir:  Directory path within S3 bucket that stores datacubes.
        composite_dir: Directory path within S3 bucket that stores datacubes' composites.
        mosaics_dir:   Directory path within S3 bucket that stores datacubes' mosaics.
        """
        self.collect_composites(cubes_file, s3_bucket, cube_dir, composite_dir)

        # If it's multi-EPSG region, create mosaics per each EPSG code first, then
        # combine them into target EPSG
        # We don't support re-projection of a single EPSG code mosaics into new EPSG -
        # that can be done manually after such mosaics are created.
        need_to_reproject = (len(self.composites) != 1)

        # Disable write to the S3 bucket if it's multi-EPSG mosaics:
        # create mosaics per each of EPSG, re-project them into target EPSG,
        # then combine them into one mosaic and copy to the target S3 bucket
        # OR
        # if it's a dry run - don't actually push results to the S3 bucket
        copy_to_s3 = not self.is_dry_run
        if need_to_reproject:
            copy_to_s3 = False

        logging.info(f'Copying to AWS S3: {copy_to_s3} (need_to_reproject={need_to_reproject}; dryrun={self.is_dry_run})')

        # Dictionary of mosaics per each EPSG code
        result_files = {}
        for each_epsg in self.composites.keys():
            logging.info(f'Opening annual composites for EPSG={each_epsg}')
            epsg = self.composites[each_epsg]
            result_files[each_epsg] = self.make_mosaics(each_epsg, epsg, s3_bucket, mosaics_dir, not need_to_reproject, copy_to_s3)

            logging.info(f'Created mosaics files: {result_files[each_epsg]}')

        if len(self.composites) > 1:
            # Determine if target mosaics need to be placed into S3 bucket
            copy_to_s3 = not self.is_dry_run
            logging.info(f'Merge mosaics: copy_to_s3={copy_to_s3}')

            # Combine re-projected mosaics per EPSG code into the whole region mosaic
            result_files = self.merge_mosaics(result_files, s3_bucket, mosaics_dir, copy_to_s3)

        # Otherwise it's only mosaics for one projection and we are done
        return result_files

    def collect_composites(
        self,
        cubes_file: str,
        s3_bucket: str,
        cube_dir: str,
        composite_dir: str
    ):
        """
        Collect datacube composites for specified region.
        Iterate over existing cubes that qualify for the search criteria and
        use cube's corresponding composites to create annual mosaics.

        cubes_file: GeoJson catalog file with existing datacubes' Zarr S3 URLs.
        s3_bucket: S3 bucket that stores all data (assumes that all datacubes,
                   composites, and mosaics are stored in the same bucket).
        cube_dir:  Directory path within S3 bucket that stores datacubes.
        composite_dir: Directory path within S3 bucket that stores datacubes' composites.
        """
        logging.info(f'BatchVars.POLYGON_SHAPE: {BatchVars.POLYGON_SHAPE}')

        all_composites = {}
        with open(cubes_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to process
            num_processed = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')
            for each_cube in cubes[CubeJson.FEATURES]:
                # Example of data cube definition in json file
                # "properties": {
                #     "fill-opacity": 0.9848664555858357,
                #     "fill": "red",
                #     "roi_percent_coverage": 1.5133544414164224,
                #     "data_epsg": "EPSG:32718",
                #     "geometry_epsg": {
                #         "type": "Polygon",
                #         "coordinates": [
                #             [
                #                 [
                #                     400000,
                #                     4400000
                #                 ],
                #                 [
                #                     500000,
                #                     4400000
                #                 ],
                #                 [
                #                     500000,
                #                     4500000
                #                 ],
                #                 [
                #                     400000,
                #                     4500000
                #                 ],
                #                 [
                #                     400000,
                #                     4400000
                #                 ]
                #             ]
                #         ]
                #     },
                #     "datacube_exist": 1,
                #     "zarr_url": "http://its-live-data.s3.amazonaws.com/datacubes/v02/S50W070/ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr"
                # }

                # Consider cubes with ROI != 0 only
                properties = each_cube[CubeJson.PROPERTIES]

                roi = properties[CubeJson.ROI_PERCENT_COVERAGE]
                if roi != 0.0:
                    # Format filename for the cube
                    epsg = properties[CubeJson.DATA_EPSG].replace(CubeJson.EPSG_SEPARATOR, '')
                    # Extract int EPSG code
                    epsg_code = epsg.replace(CubeJson.EPSG_PREFIX, '')

                    # Include only specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_GENERATE) and \
                       epsg_code not in BatchVars.EPSG_TO_GENERATE:
                       # logging.info(f'Skipping {epsg_code} which is not in {BatchVars.EPSG_TO_GENERATE}')
                       continue

                    # Exclude specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_EXCLUDE) and \
                       epsg_code in BatchVars.EPSG_TO_EXCLUDE:
                        continue

                    coords = properties[CubeJson.GEOMETRY_EPSG][CubeJson.COORDINATES][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int(x_bounds.middle_point())
                    mid_y = int(y_bounds.middle_point())

                    # Get mid point to the nearest 50
                    # logging.info(f"Mid point: x={mid_x} y={mid_y}")
                    mid_x = int(math.floor(mid_x/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)
                    mid_y = int(math.floor(mid_y/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)
                    # logging.info(f"Mid point at {BatchVars.MID_POINT_RESOLUTION}: x={mid_x} y={mid_y}")

                    # Convert to lon/lat coordinates to format s3 bucket path
                    # for the datacube
                    mid_lon_lat = itslive_utils.transform_coord(
                        epsg_code,
                        BatchVars.LON_LAT_PROJECTION,
                        mid_x, mid_y
                    )

                    if BatchVars.POLYGON_SHAPE and \
                       (not BatchVars.POLYGON_SHAPE.contains(geometry.Point(mid_lon_lat[0], mid_lon_lat[1]))):
                        # logging.info(f"Skipping non-polygon point: {mid_lon_lat}")
                        # Provided polygon does not contain cube's center point
                        continue

                    # Format filename for the cube's composites
                    cube_s3 = properties[CubeJson.URL].replace(
                        BatchVars.HTTP_PREFIX,
                        BatchVars.AWS_PREFIX
                    )

                    # Process specific datacubes only: check for full path of original cube as
                    if len(BatchVars.CUBES_TO_GENERATE) and cube_s3 not in BatchVars.CUBES_TO_GENERATE:
                        # logging.info(f"Skipping as not provided in BatchVars.CUBES_TO_GENERATE")
                        continue

                    logging.info(f'Cube name: {cube_s3}')

                    # Check if cube exists in S3 bucket (should exist, just to be sure)
                    cube_exists = self.s3.ls(cube_s3)
                    if len(cube_exists) == 0:
                        logging.info(f"Datacube {cube_s3} does not exist, skipping.")
                        continue

                    s3_composite_dir = itslive_utils.point_to_prefix(mid_lon_lat[1], mid_lon_lat[0], composite_dir)
                    composite_s3 = os.path.join(s3_bucket, s3_composite_dir, composite_filename_zarr(ITSLiveAnnualMosaics.CELL_SIZE, mid_x, mid_y))
                    logging.info(f'Composite file: {composite_s3}')

                    # Check if composite exists in S3 bucket (should exist, just to be sure)
                    composite_exists = self.s3.ls(composite_s3)
                    if len(composite_exists) == 0:
                        logging.info(f"Datacube composite {composite_s3} does not exist, skipping.")
                        continue

                    if composite_s3 in all_composites:
                        # TODO: For now just issue a warning. Once composites are re-created, enable exception
                        # raise RuntimeError(f'Composite {composite_s3} already exists for {all_composites[composite_s3]} datacube. Check on {cube_s3}!!!')
                        logging.info(f'WARNING_ATTENTION: Composite {composite_s3} already exists for {all_composites[composite_s3]} datacube. Check on {cube_s3}!!!')

                    all_composites[composite_s3] = cube_s3

                    # Update EPSG: Y: X: composite_s3_path nested dictionary
                    epsg_dict = self.composites.setdefault(epsg_code, {})
                    y_dict = epsg_dict.setdefault(mid_y, {})
                    y_dict[mid_x] = composite_s3

                    # Format cube composites filename:
                    # s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3250000_Y250000.zarr
                    num_processed += 1

            logging.info(f"Number of collected composites: {num_processed}")
            logging.info(f'Collected composites: {json.dumps(self.composites, indent=4)}')

            centers_filename = ITSLiveAnnualMosaics.REGION + ITSLiveAnnualMosaics.COMPOSITES_CENTERS_FILE
            logging.info(f'Writing collected composites to {centers_filename}...')
            with open(centers_filename, 'w') as fh:
                json.dump(self.composites, fh, indent=4)

    def make_mosaics(self, epsg_code: str, epsg: dict, s3_bucket: str, mosaics_dir: str, check_for_epsg: bool, copy_to_s3: bool):
        """
        Build annual and static mosaics from collected datacube composites per each EPSG code.
        Store "robust" composite attributes to the dataset if mosaics are not copied
        to the target AWS S3 bucket. Store "robust" mosaics attributes to the standalone
        JSON file if mosaics are to be copied to the target AWS S3 bucket.
        (ATTN: should change the logic to always store to JSON file? - than need
        to add more logic when merging re-projected mosaics stored on local drive as an
        intermittent step).

        epsg_code:      Current EPSG code the mosaics are built for.
        epsg:           Dictionary of center_x->center_y->s3_path_to_composite format to
                        provide composites that should contribute to mosaics.
        s3_bucket:      S3 bucket that stores all data (assumes that all datacubes,
                        composites, and mosaics are stored in the same bucket).
        mosaics_dir:    Directory path within S3 bucket that stores datacubes' mosaics.
        check_for_epsg: Boolean flag to indicate if source data should be of the same EPSG code as
                        requested target EPSG for mosaics.
        copy_to_s3:     Boolean flag to indicate if generated mosaics files should be copied
                        to the target S3 bucket.
        """
        # xarray.Dataset's objects for opened Zarr composites
        self.raw_ds = {}

        # "united" coordinates for mosaics within the same EPSG code
        self.time_coords = []
        self.x_coords = []
        self.y_coords = []
        self.sensor_coords = []

        # Common attributes for all mosaics of currently processed EPSG code
        self.attrs = {}

        gc.collect()

        epsg_code = int(epsg_code)

        # For each y from sorted list of composites center's y's:
        for each_mid_y in sorted(epsg.keys()):
            all_y = epsg[each_mid_y]
            # For each x from sorted list of composites center's x:
            for each_mid_x in sorted(all_y.keys()):
                composite_s3_path = all_y[each_mid_x]

                # TODO: Should preserve s3_store or just reopen the composite when needed?
                s3_store = s3fs.S3Map(root=composite_s3_path, s3=self.s3, check=False)
                ds_from_zarr = xr.open_dataset(s3_store, decode_timedelta=False, engine='zarr', consolidated=True)

                # Make sure processed composite is of the EPSG code being processed:
                # this is to address the problem we introduced by removing EPSG code
                # from composites filenames
                ds_projection = int(ds_from_zarr.attrs['projection'])
                if epsg_code != ds_projection:
                    logging.info(f'WARNING_ATTENTION: ds.projection {ds_projection} ' \
                        f'differs from epsg_code {epsg_code} being processed for {composite_s3_path}, ignoring the file')
                    # For now to be able to test with "combo" composites, don't
                    # include such composite for EPSG
                    continue

                # Make sure all composites are of the target projection if no re-projection is needed
                if check_for_epsg and (ds_projection != self.epsg):
                    raise RuntimeError(f'Expected composites in {self.epsg} projection, got {ds_projection} for {composite_s3_path}.')

                ds_time = [t.astype('M8[ms]').astype('O') for t in ds_from_zarr.time.values]

                # Store open cube's composites and corresponding metadata
                self.raw_ds[composite_s3_path] = ITSLiveAnnualMosaics.Composite(
                    ITSLiveAnnualMosaics.CompositeS3(ds_from_zarr, s3_store),
                    ds_from_zarr.x.values,
                    ds_from_zarr.y.values,
                    [each_t.year for each_t in ds_time],
                    ds_from_zarr.sensor.values.tolist()
                )

                # Collect coordinates
                self.x_coords.append(ds_from_zarr.x.values)
                self.y_coords.append(ds_from_zarr.y.values)
                self.time_coords.append(ds_time)
                self.sensor_coords.append(ds_from_zarr.sensor.values)

                # We had some corrupted composites (EC2 was terminated during copy to S3)
                # that had NaNs in x and y, check for it
                if np.any(np.isnan(self.x_coords[-1])):
                    raise RuntimeError(f'Got nan in x: {composite_s3_path}')

                if np.any(np.isnan(self.y_coords[-1])):
                    raise RuntimeError(f'Got nan in y: {composite_s3_path}')

        # Create merged dataset per each year and for the summary
        self.time_coords = sorted(list(set(np.concatenate(self.time_coords))))

        self.x_coords = sorted(list(set(np.concatenate(self.x_coords))))
        self.y_coords = sorted(list(set(np.concatenate(self.y_coords))))

        logging.info(f'Mosaics grid size: x={len(self.x_coords)} y={len(self.y_coords)}')

        # Compute cell size in x and y dimension
        x_cell = self.x_coords[1] - self.x_coords[0]
        y_cell = self.y_coords[1] - self.y_coords[0]

        self.x_coords = np.arange(self.x_coords[0], self.x_coords[-1]+1, x_cell)
        self.y_coords = np.arange(self.y_coords[0], self.y_coords[-1]+1, y_cell)

        # y coordinate in EPSG is always in ascending order
        self.y_coords = np.flip(self.y_coords)
        y_cell = self.y_coords[1] - self.y_coords[0]

        self.sensor_coords = sorted(list(set(np.concatenate(self.sensor_coords))))
        logging.info(f'Got unique sensor groups: {self.sensor_coords}')

        composites_info = repr_composite(self.raw_ds)
        epsg_range_filename = f'{ITSLiveAnnualMosaics.REGION}_{epsg_code}{ITSLiveAnnualMosaics.COMPOSITES_RANGES_FILE}'
        logging.info(f'Writing collected X/Y info to {epsg_range_filename}...')
        with open(epsg_range_filename, 'w') as fh:
            json.dump(composites_info, fh, indent=4)

        logging.info(json.dumps(composites_info))

        first_ds = self.set_mapping(x_cell, y_cell)

        # Dictionary of generated mosaics:
        # year -> mosaic file
        output_files = {}

        # Create summary mosaic (to store all 2d data variables from all composites)
        output_files[ITSLiveAnnualMosaics.SUMMARY_KEY] = self.create_summary_mosaics(epsg_code, first_ds, s3_bucket, mosaics_dir, copy_to_s3)

        # Force garbage collection as it does not always kick in
        gc.collect()

        # Re-project mosaics if target projection is other than EPSG being processed
        local_dir = None
        if epsg_code != self.epsg:
            mosaics_file = output_files[ITSLiveAnnualMosaics.SUMMARY_KEY]

            # Create sub-directory to store EPSG mosaics to
            local_dir = f'{epsg_code}_reproject_to_{self.epsg}'
            if not os.path.exists(local_dir):
                logging.info(f'Creating EPSG specific directory to write re-projected mosaics to: {local_dir}')
                os.mkdir(local_dir)

            # Append local path to the filename to store mosaics and transformation matrix to
            reproject_mosaics_filename = os.path.join(local_dir, os.path.basename(mosaics_file))
            reproject_matrix_filename = os.path.join(local_dir, ITSLiveAnnualMosaics.TRANSFORMATION_MATRIX_FILE)

            # Use EPSG based path to the filename to store mosaics and transformation matrix to
            reproject_mosaics_filename = os.path.join(local_dir, os.path.basename(mosaics_file))
            reproject_matrix_filename = os.path.join(local_dir, ITSLiveAnnualMosaics.TRANSFORMATION_MATRIX_FILE)

            if ITSLiveAnnualMosaics.USE_EXISTING_FILES and os.path.exists(reproject_mosaics_filename):
                # Mosaic file exists, don't create it
                logging.info(f'Using existing {reproject_mosaics_filename}')

            else:
                logging.info(f'Re-projecting {mosaics_file} to {self.epsg}')
                reproject_main(mosaics_file, reproject_mosaics_filename, self.epsg, reproject_matrix_filename, verbose_flag=True)

                # Force garbage collection as it does not always kick in
                gc.collect()

            # Create corresponding "robust" attributes JSON file for re-projected
            # mosaic if it does not exist
            ITSLiveAnnualMosaics.reproject_attributes(
                epsg_code,
                mosaics_file,
                self.epsg,
                reproject_mosaics_filename
            )

            # Replace output file with re-projected file
            output_files[ITSLiveAnnualMosaics.SUMMARY_KEY] = reproject_mosaics_filename

        # Create annual mosaics
        logging.info(f'Creating annual mosaics for {ITSLiveAnnualMosaics.REGION}')
        for each_year in self.time_coords:
            # Year (as "string" dtype) for the mosaic
            year_token = str(each_year.year)
            output_files[year_token] = self.create_annual_mosaics(epsg_code, first_ds, each_year, s3_bucket, mosaics_dir, copy_to_s3)

            # Force garbage collection as it does not always kick in
            gc.collect()

            if epsg_code != self.epsg:
                mosaics_file = output_files[year_token]

                # Append local path to the filename to store mosaics and transformation matrix to
                reproject_mosaics_filename = os.path.join(local_dir, os.path.basename(mosaics_file))
                reproject_matrix_filename = os.path.join(local_dir, ITSLiveAnnualMosaics.TRANSFORMATION_MATRIX_FILE)

                if ITSLiveAnnualMosaics.USE_EXISTING_FILES and os.path.exists(reproject_mosaics_filename):
                    # Mosaic file exists, don't create it
                    output_files[year_token] = reproject_mosaics_filename
                    logging.info(f'Using existing {reproject_mosaics_filename}')
                    continue

                logging.info(f'Re-projecting {mosaics_file} to {self.epsg}')
                reproject_main(mosaics_file, reproject_mosaics_filename, self.epsg, reproject_matrix_filename, verbose_flag=True)

                # Replace output file with re-projected file
                output_files[year_token] = reproject_mosaics_filename

                # Force garbage collection as it does not always kick in
                gc.collect()

        return output_files

    @staticmethod
    def reproject_attributes(
        mosaics_epsg,
        mosaic_filename,
        target_epsg,
        reproject_mosaic_filename
    ):
        """
        Re-project "proj_polygon" attribute of original mosaic and save it to
        new JSON format file that corresponds to re-projected mosaic.

        Inputs:
        =======
        mosaics_epsg: Source EPSG of original mosaic.
        mosaic_filename: Filename of original mosaic.
        target_epsg: Target EPSG of re-projected mosaic.
        reproject_mosaic_filename: Filename of re-projected mosaic.
        """
        reproject_attrs_filename = ITSLiveAnnualMosaics.filename_nc_to_json(reproject_mosaic_filename)

        if not os.path.exists(reproject_attrs_filename):
            # Original attribute filename
            attrs_filename = ITSLiveAnnualMosaics.filename_nc_to_json(mosaic_filename)

            logging.info(f'Re-projecting attributes from {attrs_filename} to {reproject_attrs_filename}')

            # Read original attributes
            mosaic_attrs = {}
            with open(attrs_filename, 'r') as fh:
                mosaic_attrs = json.load(fh)

            # Re-project proj_polygon of original attribute file to target_epsg
            polygons = mosaic_attrs[CompOutputFormat.PROJ_POLYGON]

            input_projection = osr.SpatialReference()
            input_projection.ImportFromEPSG(mosaics_epsg)

            output_projection = osr.SpatialReference()

            if target_epsg != ESRICode:
                output_projection.ImportFromEPSG(target_epsg)

            else:
                output_projection.ImportFromProj4(ESRICode_Proj4)

            source_to_target_transfer = osr.CoordinateTransformation(input_projection, output_projection)

            # Step through all polygons and transfer coordinates to target projection
            target_polygons = []
            for each in polygons:
                target_polygons.append([list(coord) for coord in source_to_target_transfer.TransformPoints(each)])

            mosaic_attrs[CompOutputFormat.PROJ_POLYGON] = target_polygons

            # Save re-projected attributes to file
            with open(reproject_attrs_filename, 'w') as fh:
                json.dump(mosaic_attrs, fh, indent=4)

    def set_mapping(self, x_cell, y_cell):
        """
        Set mapping data variable for the current mosaics.

        Inputs:
        =======
        x_cell: Cell dimension in X
        y_cell: Cell dimension in Y
        """
        ds_urls = sorted(list(self.raw_ds.keys()))
        # Use "first" dataset to "collect" global attributes
        first_ds = self.raw_ds[ds_urls[0]].s3.ds

        # Create mapping data variable
        self.mapping = xr.DataArray(
            data='',
            attrs=first_ds[DataVars.MAPPING].attrs,
            coords={},
            dims=[]
        )

        # Set GeoTransform to correspond to the mosaic's tile:
        # format GeoTransform
        # Sanity check: check cell size for all mosaics against target cell size
        if ITSLiveAnnualMosaics.CELL_SIZE != x_cell and \
           ITSLiveAnnualMosaics.CELL_SIZE != np.abs(y_cell):
           raise RuntimeError(f'Provided grid cell size {ITSLiveAnnualMosaics.CELL_SIZE} does not correspond to the cell size of dataset: x={x_cell} or y={np.abs(y_cell)}')

        # :GeoTransform = "-3300007.5 120.0 0 300007.5 0 -120.0";
        new_geo_transform_str = f"{self.x_coords[0] - x_cell/2.0} {x_cell} 0 {self.y_coords[0] - y_cell/2.0} 0 {y_cell}"
        logging.info(f'Setting mapping.GeoTransform: {new_geo_transform_str}')
        self.mapping.attrs['GeoTransform'] = new_geo_transform_str

        # Return first of the datasets just to copy attributes
        return first_ds

    def merge_mosaics(self, epsg_mosaics_files: dict, s3_bucket: str, mosaics_dir: str, copy_to_s3: bool):
        """
        Combine re-projected to the target EPSG projection annual and static mosaics
        for the region.
        Apply "average" to overlapping cells due to re-projection of mosaics that
        originate from different EPSG projections (original projections of composites).

        epsg_mosaics_files: Dictionary of re-projected mosaics per EPSG code. All mosaics
                      in this dictionary correspond to the same EPSG target code.
                      It's in the format: {epsg: {'year' or '0000': mosaic_file}}.
        s3_bucket: S3 bucket that stores all data (assumes that all datacubes,
                   composites, and mosaics are stored in the same bucket).
        mosaics_dir: Directory path within S3 bucket that stores datacubes' mosaics.
        copy_to_s3: Flag if result mosaics need to be copied to the AWS S3 bucket.
        """
        # Collect unique year values for all generated mosaics
        all_mosaics_keys = None
        epsg_keys = list(epsg_mosaics_files.keys())

        # Unique "year" tokens for mosaics of the very first listed EPSG code
        all_keys = set(epsg_mosaics_files[epsg_keys[0]])

        # Collect all other unique tokens from the rest of EPSG mosaics
        for i in range(1, len(epsg_keys)):
            all_keys = all_keys.union(epsg_mosaics_files[epsg_keys[i]])

        all_years = sorted(all_keys)
        logging.info(f'Unique year tokens for all generated mosaics: {all_years}')

        self.raw_ds = {}
        # There is no time coordinate in mosaics
        self.time_coords = []

        # "united" coordinates for mosaics within the same EPSG code
        self.x_coords = []
        self.y_coords = []
        self.sensor_coords = []

        # Common attributes for all mosaics for the target EPSG projection
        self.attrs = {}
        gc.collect()

        # Merge static mosaics first, that will define X/Y grid for annual mosaics
        for mosaics_dict in epsg_mosaics_files.values():
            # Get mosaic file corresponding to '0000': must be present for each EPSG "sub_directory"
            mosaic_file = mosaics_dict[ITSLiveAnnualMosaics.SUMMARY_KEY]
            ds_from_nc = xr.open_dataset(mosaic_file)

            # Make sure processed composite is of the EPSG code being processed:
            ds_projection = int(ds_from_nc.attrs['projection'])

            # Make sure EPSG-specific mosaics are of the target projection
            # (re-projection is done at this point)
            if ds_projection != self.epsg:
                raise RuntimeError(f'Expected mosaic in {self.epsg} projection, got {ds_projection} for {mosaic_file}.')

            # Store open mosaics and corresponding metadata
            self.raw_ds[mosaic_file] = ITSLiveAnnualMosaics.Composite(
                ITSLiveAnnualMosaics.CompositeS3(ds_from_nc, None),
                ds_from_nc.x.values,
                ds_from_nc.y.values,
                None,
                ds_from_nc.sensor.values.tolist()
            )

            # Collect coordinates
            self.x_coords.append(ds_from_nc.x.values)
            self.y_coords.append(ds_from_nc.y.values)
            self.sensor_coords.append(ds_from_nc.sensor.values)

        # Remove processed key from the list of known keys
        all_years.remove(ITSLiveAnnualMosaics.SUMMARY_KEY)

        # Create one large dataset
        self.x_coords = sorted(list(set(np.concatenate(self.x_coords))))
        self.y_coords = sorted(list(set(np.concatenate(self.y_coords))))

        self.sensor_coords = sorted(list(set(np.concatenate(self.sensor_coords))))
        logging.info(f'Got unique sensor groups: {self.sensor_coords}')

        logging.info(f'Mosaics grid size: x={len(self.x_coords)} y={len(self.y_coords)}')

        # Compute cell size in x and y dimension
        x_cell = self.x_coords[1] - self.x_coords[0]
        y_cell = self.y_coords[1] - self.y_coords[0]

        self.x_coords = np.arange(self.x_coords[0], self.x_coords[-1]+1, x_cell)
        self.y_coords = np.arange(self.y_coords[0], self.y_coords[-1]+1, y_cell)

        # y coordinate in EPSG is always in ascending order
        self.y_coords = np.flip(self.y_coords)
        y_cell = self.y_coords[1] - self.y_coords[0]

        mosaics_info = repr_composite(self.raw_ds)
        epsg_range_filename = f'{ITSLiveAnnualMosaics.REGION}_{self.epsg}{ITSLiveAnnualMosaics.COMPOSITES_RANGES_FILE}'
        logging.info(f'Writing collected X/Y info to {epsg_range_filename}...')
        with open(epsg_range_filename, 'w') as fh:
            json.dump(mosaics_info, fh, indent=4)

        logging.info(json.dumps(mosaics_info))

        # Set mapping data variable for the target mosaics
        first_ds = self.set_mapping(x_cell, y_cell)

        # Dictionary of generated mosaics:
        # year -> mosaic file
        output_files = {}

        # Merge summary mosaics (to store all 2d data and common 3d variables from all composites)
        output_files[ITSLiveAnnualMosaics.SUMMARY_KEY] = self.merge_summary_mosaics(first_ds, s3_bucket, mosaics_dir, copy_to_s3)

        # Force garbage collection as it does not always kick in
        gc.collect()

        # Create annual mosaics
        logging.info(f'Merging annual mosaics for {ITSLiveAnnualMosaics.REGION}')
        for each_year in all_years:
            self.raw_ds = {}
            gc.collect()

            # Populate raw data
            for mosaics_epsg, mosaics_dict in epsg_mosaics_files.items():
                # Get mosaic file corresponding to '0000': must be present for each EPSG "sub_directory"
                if each_year not in mosaics_dict:
                    logging.info(f'Missing {each_year} from {mosaics_epsg} re-projection')
                    continue

                mosaic_file = mosaics_dict[each_year]
                ds_from_nc = xr.open_dataset(mosaic_file)

                # Make sure processed composite is of the EPSG code being processed:
                ds_projection = int(ds_from_nc.attrs['projection'])

                # Make sure EPSG-specific mosaics are of the target projection
                # (re-projection is already done at this point)
                if ds_projection != self.epsg:
                    raise RuntimeError(f'Expected mosaic in {self.epsg} projection, got {ds_projection} for {mosaic_file}.')

                # Store open mosaics and corresponding metadata
                self.raw_ds[mosaic_file] = ITSLiveAnnualMosaics.Composite(
                    ITSLiveAnnualMosaics.CompositeS3(ds_from_nc, None),
                    ds_from_nc.x.values,
                    ds_from_nc.y.values,
                    None,
                    ds_from_nc.sensor.values.tolist()
                )

            output_files[each_year] = self.merge_annual_mosaics(first_ds, each_year, s3_bucket, mosaics_dir, copy_to_s3)

            # Force garbage collection as it does not always kick in
            gc.collect()

        return output_files

    def merge_summary_mosaics(self, first_ds, s3_bucket, mosaics_dir, copy_to_s3):
        """
        Merge summary mosaics that were generated by re-projection of EPSG mosaics into
        target projection. Store result to NetCDF format file in S3 bucket if provided.
        Store robust attributes that contain information about all composites and datacubes
        contributed to the mosaic into separate json file (for traceability).

        first_ds: xarray.Dataset object that represents any (first) composite dataset.
                  It's used to collect global attributes that are applicable to the
                  mosaics.
        s3_bucket: AWS S3 bucket to place result mosaics file in.
        mosaics_dir: AWS S3 directory to place mosaics in.
        copy_to_s3: Boolean flag to indicate if generated mosaics files should be copied
                    to the target S3 bucket.
        """
        logging.info(f'Merging summary mosaics for {ITSLiveAnnualMosaics.REGION} region')

        # Format filename for the mosaics
        mosaics_filename = summary_mosaics_filename_nc(self.grid_size_str, ITSLiveAnnualMosaics.REGION, ITSLiveAnnualMosaics.FILE_VERSION)

        if ITSLiveAnnualMosaics.USE_EXISTING_FILES and os.path.exists(mosaics_filename):
            attrs_filename = ITSLiveAnnualMosaics.filename_nc_to_json(mosaics_filename)

            if os.path.exists(attrs_filename):
                # Read attributes from existing file
                logging.info(f'Loading attributes from existing {attrs_filename}...')
                with open(attrs_filename, 'r') as fh:
                    self.attrs = json.load(fh)

            else:
                raise RuntimeError(f'Attributes file {attrs_filename} is missing for {mosaics_filename}')

            return mosaics_filename

        # Dataset to represent summary mosaic
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.x_coords,
                    first_ds.x.attrs
                ),
                Coords.Y: (
                    Coords.Y,
                    self.y_coords,
                    first_ds.y.attrs
                ),
                CompDataVars.SENSORS: (
                    CompDataVars.SENSORS,
                    self.sensor_coords,
                    {
                        DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSORS],
                        DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSORS]
                    }
                )
            },
            attrs = {
                CubeOutputFormat.AUTHOR: CubeOutputFormat.Values.AUTHOR,
                CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE: first_ds.attrs[CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE],
                CubeOutputFormat.INSTITUTION: CubeOutputFormat.Values.INSTITUTION,
                MosaicsOutputFormat.REGION: ITSLiveAnnualMosaics.REGION
            }
        )

        ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT] = first_ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT]
        ds.attrs[MosaicsOutputFormat.MOSAICS_SOFTWARE_VERSION] = ITSLiveAnnualMosaics.VERSION
        ds.attrs[CubeOutputFormat.PROJECTION] = self.epsg
        ds.attrs[CubeOutputFormat.TITLE] = MosaicsOutputFormat.STATIC_TITLE
        ds.attrs[CubeOutputFormat.DATE_CREATED] = self.date_created

        # Create sensors_labels = "Band 1: S1A_S1B; Band 2: S2A_S2B; Band 3: L8_L9";
        sensors_labels = [f'Band {index+1}: {self.sensor_coords[index]}' for index in range(len(self.sensor_coords))]
        sensors_labels = f'{"; ".join(sensors_labels)}'
        ds.attrs[CompOutputFormat.SENSORS_LABELS] = sensors_labels

        ds[DataVars.MAPPING] = self.mapping

        # Add dt_max data array to Dataset which is based on union of sensor groups
        # of all composites (in case some of them differ in sensor groups)
        three_coords = [self.sensor_coords, self.y_coords, self.x_coords]
        three_dims = [CompDataVars.SENSORS, Coords.Y, Coords.X]
        three_dims_len = (len(self.sensor_coords), len(self.y_coords), len(self.x_coords))

        two_coords = [self.y_coords, self.x_coords]
        two_dims = [Coords.Y, Coords.X]
        two_dims_len = (len(self.y_coords), len(self.x_coords))

        # Create lists of attributes that correspond to multiple mosaics (already
        # re-projected from multiple EPSGs)
        self.attrs = {key: [] for key in MosaicsOutputFormat.ALL_ATTR}

        for each_file, each_ds in self.raw_ds.items():
            logging.info(f'Collecting attributes from {each_file}')

            attrs_filename = ITSLiveAnnualMosaics.filename_nc_to_json(each_file)

            if os.path.exists(attrs_filename):
                # Read attributes from existing file
                logging.info(f'Loading attributes from existing {attrs_filename}...')
                with open(attrs_filename, 'r') as fh:
                    _attrs = json.load(fh)

                    # Collect attributes
                    for each_attr in self.attrs.keys():
                        self.attrs[each_attr].extend(_attrs[each_attr])

            else:
                raise RuntimeError(f'Missing {attrs_filename} file for {each_file}')

        # Set center point's longitude and latitude for each polygon (if more than one) of the mosaic
        lon = []
        lat = []

        # Set cumulative attributes for the mosaic
        for key, each_value in self.attrs.items():
            # Each value is a list of polygon lists collected over
            # EPSG mosaics, iterate through each list of polygons and unite them
            value = each_value

            if key in [CubeOutputFormat.GEO_POLYGON, CubeOutputFormat.PROJ_POLYGON]:
                # Collect polygons:
                polygons = [geometry.Polygon(json.loads(each_polygon)) for each_polygon in each_value]
                # for each_mosaics_value in value:
                #     polygons.extend([geometry.Polygon(each_polygon) for each_polygon in each_mosaics_value])

                value, geo_polygon = ITSLiveAnnualMosaics.unite_polygons(key, polygons)

                if len(geo_polygon):
                    for each_polygon in geo_polygon:
                        lon.append(Bounds([each[0] for each in each_polygon]).middle_point())
                        lat.append(Bounds([each[1] for each in each_polygon]).middle_point())

            # Just reset to cumulative value
            self.attrs[key] = value

        # Set center point's longitude and latitude for each polygon (if more than one) of the mosaic
        # Reset to 2 digits of precision
        lat = [round(each_lat, 2) for each_lat in lat]
        lon = [round(each_lon, 2) for each_lon in lon]

        ds.attrs[CompOutputFormat.LATITUDE] = json.dumps(lat)
        ds.attrs[CompOutputFormat.LONGITUDE] = json.dumps(lon)

        # Save attributes for the use by annual mosaics
        self.attrs[CompOutputFormat.LATITUDE] = lat
        self.attrs[CompOutputFormat.LONGITUDE] = lon

        _concat_dim_name  = 'new_dim'

        # Concatenate data for each data variable
        for each_var in ITSLiveAnnualMosaics.SUMMARY_VARS:
            data_list = []

            # Concatenated dataset
            concatenated = None

            # Default to 2-d data
            ndim = 2
            _coords = two_coords
            _dims = two_dims
            _dims_len = two_dims_len

            # Step through all datasets and concatenate data in new dimension
            # to be able to compute the average - xr.merge() does not support
            # function to apply on merging
            for each_file, each_ds in self.raw_ds.items():
                logging.info(f'Merging {each_var} from {each_file}')

                if each_var not in ds:
                    # Read shape of the data in dataset, and set dimensions
                    # accordingly
                    ndim = each_ds.s3.ds[each_var].ndim

                    if ndim == 3:
                        # Set for 3-d data
                        _coords = three_coords
                        _dims = three_dims
                        _dims_len = three_dims_len

                    ds[each_var] = xr.DataArray(
                        data=np.full(_dims_len, np.nan),
                        coords=_coords,
                        dims=_dims,
                        attrs=each_ds.s3.ds[each_var].attrs
                    )

                    # Reset sensor_labels if present in data
                    if CompOutputFormat.SENSORS_LABELS in ds[each_var].attrs:
                        ds[each_var].attrs[CompOutputFormat.SENSORS_LABELS] = sensors_labels

                data_list.append(each_ds.s3.ds[each_var])

                if len(data_list) > 1:
                    # Concatenate once we have 2 arrays
                    concatenated = xr.concat(data_list, _concat_dim_name, join="outer")
                    data_list = [concatenated]

                gc.collect()

            # Take average of all overlapping cells
            avg_overlap = concatenated.mean(_concat_dim_name, skipna=True)

            # Set data values in result dataset
            avg_overlap_dims = dict(x=avg_overlap.x.values, y=avg_overlap.y.values)
            if ndim == 3:
                avg_overlap_dims = dict(x=avg_overlap.x.values, y=avg_overlap.y.values, sensor=avg_overlap.sensor.values)

            # Set values for the output dataset
            ds[each_var].loc[avg_overlap_dims] = avg_overlap

            gc.collect()

        logging.info(f'Merged all data.')

        if copy_to_s3:
            ds.attrs[CubeOutputFormat.S3] = os.path.join(s3_bucket, mosaics_dir, mosaics_filename)
            ds.attrs[CubeOutputFormat.URL] = ds.attrs[CubeOutputFormat.S3].replace(BatchVars.AWS_PREFIX, BatchVars.HTTP_PREFIX)

        else:
            # Append local path to the filename to store mosaics to
            mosaics_filename = ITSLiveAnnualMosaics.epsg_mosaics_path(ds_projection, mosaics_filename)

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': ITSLiveAnnualMosaics.CHUNK_SIZE, 'y': ITSLiveAnnualMosaics.CHUNK_SIZE})

        # Write mosaic to NetCDF format file
        ITSLiveAnnualMosaics.summary_mosaic_to_netcdf(ds, self.attrs, s3_bucket, mosaics_dir, mosaics_filename, copy_to_s3)

        return mosaics_filename

    def merge_annual_mosaics(self, first_ds: xr.Dataset, year: str, s3_bucket, mosaics_dir: str, copy_to_s3: bool):
        """
        Merge mosaics for a specific year that were generated by re-projection of
        EPSG mosaics into target projection. Store result to NetCDF format file in
        S3 bucket if provided.
        This method relies on the fact that it's called after static mosaics were
        merged - thus all metadata is set when static mosaics are processed.

        first_ds: xarray.Dataset object that represents any (first) composite dataset.
                  It's used to collect global attributes that are applicable to the
                  mosaics.
        year: Year for the mosaic to create.
        s3_bucket: AWS S3 bucket to place result mosaics file in.
        mosaics_dir: AWS S3 directory to place mosaics in.
        copy_to_s3: Boolean flag to indicate if generated mosaics files should be copied
            to the target S3 bucket.
        """
        logging.info(f'Merging annual mosaics for {ITSLiveAnnualMosaics.REGION} region for {year} year')

        # Format filename for the mosaics
        # mosaics_filename = f'{FilenamePrefix.Mosaics}_{self.grid_size_str}m_{ITSLiveAnnualMosaics.REGION}_{year_date.year}_{ITSLiveAnnualMosaics.FILE_VERSION}.nc'
        mosaics_filename = annual_mosaics_filename_nc(self.grid_size_str, ITSLiveAnnualMosaics.REGION, year, ITSLiveAnnualMosaics.FILE_VERSION)

        # Keep consistent with "year" date of the composites
        mosaic_date = datetime.date(int(year), CENTER_DATE.month, CENTER_DATE.day)

        # Dataset to represent annual mosaic
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.x_coords,
                    first_ds.x.attrs
                ),
                Coords.Y: (
                    Coords.Y,
                    self.y_coords,
                    first_ds.y.attrs
                )
            },
            attrs = {
                CubeOutputFormat.AUTHOR: CubeOutputFormat.Values.AUTHOR,
                CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE: first_ds.attrs[CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE],
                CubeOutputFormat.INSTITUTION: CubeOutputFormat.Values.INSTITUTION,
                MosaicsOutputFormat.REGION: ITSLiveAnnualMosaics.REGION,
                MosaicsOutputFormat.YEAR: mosaic_date.strftime('%d-%b-%Y')
            }
        )

        ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT] = first_ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT]
        ds.attrs[MosaicsOutputFormat.MOSAICS_SOFTWARE_VERSION] = ITSLiveAnnualMosaics.VERSION
        ds.attrs[CubeOutputFormat.PROJECTION] = self.epsg
        ds.attrs[CubeOutputFormat.TITLE] = MosaicsOutputFormat.ANNUAL_TITLE
        ds.attrs[CubeOutputFormat.DATE_CREATED] = self.date_created

        ds[DataVars.MAPPING] = self.mapping

        # Re-use attributes as set by static mosaics
        ds.attrs[CompOutputFormat.LATITUDE] = json.dumps(self.attrs[CompOutputFormat.LATITUDE])
        ds.attrs[CompOutputFormat.LONGITUDE] = json.dumps(self.attrs[CompOutputFormat.LONGITUDE])

        two_coords = [self.y_coords, self.x_coords]
        two_dims = [Coords.Y, Coords.X]
        two_dims_len = (len(self.y_coords), len(self.x_coords))
        _concat_dim_name  = 'new_dim'

        # Concatenate data for each data variable
        for each_var in ITSLiveAnnualMosaics.ANNUAL_VARS:
            data_list = []

            # Concatenated dataset
            concatenated = None

            # Step through all datasets and concatenate data in new dimension
            # to be able to compute the average - xr.merge() does not support
            # function to apply on merging
            for each_file, each_ds in self.raw_ds.items():
                logging.info(f'Merging {each_var} from {each_file}')

                if each_var not in ds:
                    # Create data variable
                    ds[each_var] = xr.DataArray(
                        data=np.full(two_dims_len, np.nan),
                        coords=two_coords,
                        dims=two_dims,
                        attrs=each_ds.s3.ds[each_var].attrs
                    )

                data_list.append(each_ds.s3.ds[each_var])

                if len(data_list) > 1:
                    # Concatenate once we have 2 arrays
                    concatenated = xr.concat(data_list, _concat_dim_name, join="outer")
                    data_list = [concatenated]

                gc.collect()

            # Take average of all overlapping cells
            avg_overlap = concatenated.mean(_concat_dim_name, skipna=True)

            # Set data values in result dataset
            avg_overlap_dims = dict(x=avg_overlap.x.values, y=avg_overlap.y.values)

            # Set values for the output dataset
            ds[each_var].loc[avg_overlap_dims] = avg_overlap

            gc.collect()

        logging.info(f'Merged all data.')

        if copy_to_s3:
            ds.attrs['s3'] = os.path.join(s3_bucket, mosaics_dir, mosaics_filename)
            ds.attrs['url'] = ds.attrs['s3'].replace(BatchVars.AWS_PREFIX, BatchVars.HTTP_PREFIX)

        # Write mosaic to NetCDF format file
        ITSLiveAnnualMosaics.annual_mosaic_to_netcdf(ds, s3_bucket, mosaics_dir, mosaics_filename, copy_to_s3)

        return mosaics_filename

    def create_annual_mosaics(self, ds_projection, first_ds, year_date, s3_bucket, mosaics_dir, copy_to_s3):
        """
        Create mosaics for a specific year and store it to NetCDF format file in
        S3 bucket if provided.

        ds_projection: EPSG projection for the current mosaics.
        first_ds: xarray.Dataset object that represents any (first) composite dataset.
                  It's used to collect global attributes that are applicable to the
                  mosaics.
        year_date: Datetime object for the mosaic to create.
        s3_bucket: AWS S3 bucket to place result mosaics file in.
        mosaics_dir: AWS S3 directory to place mosaics in.
        copy_to_s3: Boolean flag to indicate if generated mosaics files should be copied
            to the target S3 bucket.
        """
        logging.info(f'Creating annual mosaics for {ITSLiveAnnualMosaics.REGION} region for {year_date.year} year')

        # Format filename for the mosaics
        # mosaics_filename = f'{FilenamePrefix.Mosaics}_{self.grid_size_str}m_{ITSLiveAnnualMosaics.REGION}_{year_date.year}_{ITSLiveAnnualMosaics.FILE_VERSION}.nc'
        mosaics_filename = annual_mosaics_filename_nc(self.grid_size_str, ITSLiveAnnualMosaics.REGION, year_date, ITSLiveAnnualMosaics.FILE_VERSION)

        if not copy_to_s3:
            # If need to re-project mosaics, then mosaics is written to local directory first,
            # create path based on EPSG code for the mosaic
            mosaics_filename = ITSLiveAnnualMosaics.epsg_mosaics_path(ds_projection, mosaics_filename)

        if ITSLiveAnnualMosaics.USE_EXISTING_FILES and os.path.exists(mosaics_filename):
            # Mosaic file exists, don't create it
            logging.info(f'Using existing {mosaics_filename}')
            return mosaics_filename

        # Dataset to represent annual mosaic
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.x_coords,
                    first_ds.x.attrs
                ),
                Coords.Y: (
                    Coords.Y,
                    self.y_coords,
                    first_ds.y.attrs
                )
            },
            attrs = {
                MosaicsOutputFormat.AUTHOR: MosaicsOutputFormat.ATTR_VALUES[MosaicsOutputFormat.AUTHOR],
                CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE: first_ds.attrs[CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE],
                MosaicsOutputFormat.INSTITUTION: MosaicsOutputFormat.ATTR_VALUES[MosaicsOutputFormat.INSTITUTION],
                MosaicsOutputFormat.REGION: ITSLiveAnnualMosaics.REGION,
                MosaicsOutputFormat.YEAR: year_date.strftime('%d-%b-%Y')
            }
        )

        ds.attrs[CubeOutputFormat.GDAL_AREA_OR_POINT] = CubeOutputFormat.Values.AREA
        ds.attrs[MosaicsOutputFormat.MOSAICS_SOFTWARE_VERSION] = ITSLiveAnnualMosaics.VERSION
        ds.attrs[CubeOutputFormat.PROJECTION] = str(ds_projection)
        ds.attrs[CubeOutputFormat.TITLE] = MosaicsOutputFormat.ANNUAL_TITLE
        ds.attrs[CubeOutputFormat.DATE_CREATED] = self.date_created

        # Cumulative attributes are already collected by generation of summary mosaic,
        # so longitude and latitude of center points are already computed for the
        # region.
        ds.attrs[CompOutputFormat.LATITUDE] = json.dumps(self.attrs[CompOutputFormat.LATITUDE])
        ds.attrs[CompOutputFormat.LONGITUDE] = json.dumps(self.attrs[CompOutputFormat.LONGITUDE])

        ds[DataVars.MAPPING] = self.mapping

        # Concatenate data for each data variable that has time (year value) dimension
        for each_file, each_ds in self.raw_ds.items():
            if year_date.year in each_ds.time:
                # Composites have data for the year
                year_index = each_ds.time.index(year_date.year)

                for each_var in ITSLiveAnnualMosaics.ANNUAL_VARS:
                    if each_var not in ds:
                        # Create data variable in output dataset
                        ds[each_var] = each_ds.s3.ds[each_var][year_index].load()
                        ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

                    else:
                        # Update data variable in output dataset
                        ds[each_var].loc[dict(x=each_ds.x, y=each_ds.y)] = each_ds.s3.ds[each_var][year_index].load()

            else:
                logging.warning(f'{each_file} does not have data for {year_date.year} year, skipping.')

        if copy_to_s3:
            ds.attrs['s3'] = os.path.join(s3_bucket, mosaics_dir, mosaics_filename)
            ds.attrs['url'] = ds.attrs['s3'].replace(BatchVars.AWS_PREFIX, BatchVars.HTTP_PREFIX)

        # Write mosaic to NetCDF format file
        ITSLiveAnnualMosaics.annual_mosaic_to_netcdf(ds, s3_bucket, mosaics_dir, mosaics_filename, copy_to_s3)

        return mosaics_filename

    @staticmethod
    def epsg_mosaics_path(ds_projection, mosaics_filename):
        """
        Create path to the annual mosaics filename which is based on the EPSG
        code the mosaic is created for.
        """
        # Create sub-directory to store EPSG mosaics to
        local_dir = str(ds_projection)
        if not os.path.exists(local_dir):
            logging.info(f'Creating EPSG specific directory to write mosaics to: {local_dir}')
            os.mkdir(local_dir)

        # Append local path to the filename to store mosaics to
        mosaics_filepath = os.path.join(local_dir, mosaics_filename)

        return mosaics_filepath

    @staticmethod
    def annual_mosaic_to_netcdf(ds: xr.Dataset, s3_bucket: str, bucket_dir: str, filename: str, copy_to_s3: bool):
        """
        Store datacube annual mosaics to NetCDF store.
        """
        target_file = filename

        if copy_to_s3:
            target_file = os.path.join(s3_bucket, bucket_dir, filename)

        logging.info(f'Writing summary mosaics to {target_file}')

        if CompDataVars.TIME in ds:
            # "time" coordinate can propagate into dataset when assigning
            # xr.DataArray values from composites, which have "time" dimension
            del ds[CompDataVars.TIME]


        two_dim_chunks_settings = (ds.y.size, ds.x.size)

        compression = {"zlib": True, "complevel": 2, "shuffle": True}

        # Set encoding
        encoding_settings = {}
        for each in [Coords.X, Coords.Y]:
            encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

        # Settings for "float" data types
        for each in [
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR
            ]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                'dtype': np.float32,
                'chunksizes': two_dim_chunks_settings
            })
            encoding_settings[each].update(compression)

        for each in [CompDataVars.COUNT]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                'dtype': np.short,
                'chunksizes': two_dim_chunks_settings
            })
            encoding_settings[each].update(compression)

        # Write locally
        ds.to_netcdf(f'{filename}', engine=ITSLiveAnnualMosaics.NC_ENGINE, encoding=encoding_settings)

        if copy_to_s3:
            ITSLiveAnnualMosaics.copy_to_s3_bucket(filename, target_file)

    def create_summary_mosaics(self, ds_projection, first_ds, s3_bucket, mosaics_dir, copy_to_s3):
        """
        Create summary mosaics and store it to NetCDF format file in
        S3 bucket if provided.

        ds_projection: EPSG projection for the current mosaics.
        first_ds: xarray.Dataset object that represents any (first) composite dataset.
                  It's used to collect global attributes that are applicable to the
                  mosaics.
        s3_bucket: AWS S3 bucket to place result mosaics file in.
        mosaics_dir: AWS S3 directory to place mosaics in.
        copy_to_s3: Boolean flag to indicate if generated mosaics files should be copied
            to the target S3 bucket.
        """
        logging.info(f'Creating summary mosaics for {ITSLiveAnnualMosaics.REGION} region')

        # Format filename for the mosaics
        mosaics_filename = summary_mosaics_filename_nc(self.grid_size_str, ITSLiveAnnualMosaics.REGION, ITSLiveAnnualMosaics.FILE_VERSION)

        if not copy_to_s3:
            # If need to re-project mosaics, then mosaics is written to local directory first,
            # create path based on EPSG code for the mosaic
            mosaics_filename = ITSLiveAnnualMosaics.epsg_mosaics_path(ds_projection, mosaics_filename)

        if ITSLiveAnnualMosaics.USE_EXISTING_FILES and os.path.exists(mosaics_filename):
            # Mosaic file exists, don't create it
            logging.info(f'Using existing {mosaics_filename}')
            return mosaics_filename

        # Dataset to represent summary mosaic
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.x_coords,
                    first_ds.x.attrs
                ),
                Coords.Y: (
                    Coords.Y,
                    self.y_coords,
                    first_ds.y.attrs
                ),
                CompDataVars.SENSORS: (
                    CompDataVars.SENSORS,
                    self.sensor_coords,
                    {
                        DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSORS],
                        DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSORS]
                    }
                )
            },
            attrs = {
                MosaicsOutputFormat.AUTHOR: MosaicsOutputFormat.ATTR_VALUES[MosaicsOutputFormat.AUTHOR],
                CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE: first_ds.attrs[CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE],
                MosaicsOutputFormat.INSTITUTION: MosaicsOutputFormat.ATTR_VALUES[MosaicsOutputFormat.INSTITUTION],
                MosaicsOutputFormat.REGION: ITSLiveAnnualMosaics.REGION
            }
        )

        ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT] = first_ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT]
        ds.attrs[MosaicsOutputFormat.MOSAICS_SOFTWARE_VERSION] = ITSLiveAnnualMosaics.VERSION
        ds.attrs[CubeOutputFormat.PROJECTION] = ds_projection
        ds.attrs[CubeOutputFormat.TITLE] = MosaicsOutputFormat.STATIC_TITLE
        ds.attrs[CubeOutputFormat.DATE_CREATED] = self.date_created

        # Create sensors_labels = "Band 1: S1A_S1B; Band 2: S2A_S2B; Band 3: L8_L9";
        sensors_labels = [f'Band {index+1}: {self.sensor_coords[index]}' for index in range(len(self.sensor_coords))]
        sensors_labels = f'{"; ".join(sensors_labels)}'
        ds.attrs[CompOutputFormat.SENSORS_LABELS] = sensors_labels

        ds[DataVars.MAPPING] = self.mapping

        # Add dt_max data array to Dataset which is based on union of sensor groups
        # of all composites (in case some of them differ in sensor groups)
        var_coords = [self.sensor_coords, self.y_coords, self.x_coords]
        var_dims = [CompDataVars.SENSORS, Coords.Y, Coords.X]
        sensor_dims = (len(self.sensor_coords), len(self.y_coords), len(self.x_coords))

        # These data variables need to be pre-allocated as their sensor dimension
        # is cumulative over all datasets
        ds[CompDataVars.MAX_DT] = xr.DataArray(
            data=np.full(sensor_dims, np.nan),
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.MAX_DT],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.MAX_DT],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                CompOutputFormat.SENSORS_LABELS: sensors_labels,
                DataVars.UNITS: DataVars.ImgPairInfo.UNITS[DataVars.ImgPairInfo.DATE_DT]
            }
        )

        ds[CompDataVars.SENSOR_INCLUDE] = xr.DataArray(
            data=np.full(sensor_dims, 0, dtype=np.short),
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSOR_INCLUDE],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSOR_INCLUDE],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                CompOutputFormat.SENSORS_LABELS: sensors_labels,
                DataVars.UNITS: DataVars.BINARY_UNITS
            }
        )

        # Create lists of attributes that correspond to multiple composites that
        # contribute to each of the static mosaic
        self.attrs = {key: [] for key in ITSLiveAnnualMosaics.ALL_ATTR}

        # For debugging only to speed up the runtime
        # index = 0

        # Concatenate data for each data variable
        for each_file, each_ds in self.raw_ds.items():
            logging.info(f'Collecting summary data from {each_file}')

            for each_var in ITSLiveAnnualMosaics.SUMMARY_VARS:
                # logging.info(f'Collecting {each_var} from {each_file}')

                if each_var not in ds:
                    # Create data variable in result dataset
                    # This applies only to 2d variables as 3d variables need to
                    # be allocated before this loop
                    ds[each_var] = each_ds.s3.ds[each_var].load()

                    # Set mapping attribute
                    ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

                else:
                    _coords = dict(x=each_ds.x, y=each_ds.y)
                    if each_ds.s3.ds[each_var].ndim == 3:
                        # If it has a sensor dimension, then data variable is already in
                        # dataset and need to specify sensor to avoid an exception
                        # if number of sensors is different in loaded dataset
                        _coords = dict(x=each_ds.x, y=each_ds.y, sensor=each_ds.sensor)

                    # Update data variable in result dataset
                    ds[each_var].loc[_coords] = each_ds.s3.ds[each_var].load()

            # For debugging only to speed up the runtime
            # index += 1

            # Collect "dt_max" and "sensor_flag" values per each sensor group: self.sensor_coords
            for each_group in self.sensor_coords:
                if each_group in each_ds.sensor:
                    sensor_index = each_ds.sensor.index(each_group)

                    for each_var in [CompDataVars.MAX_DT, CompDataVars.SENSOR_INCLUDE]:
                        # logging.info(f'Update {each_var} for {each_group} by {each_file}')
                        ds[each_var].loc[dict(x=each_ds.x, y=each_ds.y, sensor=each_group)] = each_ds.s3.ds[each_var][sensor_index].load()

            # Update attributes
            for each_attr in self.attrs.keys():
                self.attrs[each_attr].append(each_ds.s3.ds.attrs[each_attr])

            # For debugging only to speed up the runtime
            # if index == 1:
            #     break

        # Set center point's longitude and latitude for each polygon (if more than one) of the mosaic
        lon = []
        lat = []

        # Set cumulative attributes for the mosaic
        for each_key, each_value in self.attrs.items():
            key = each_key

            if each_key in MosaicsOutputFormat.ATTR_MAP:
                # Get key name for the mosaic's attribute - some will have different
                # name as it appears in composites attributes
                key = MosaicsOutputFormat.ATTR_MAP[each_key]

            value = each_value

            if each_key in [CubeOutputFormat.GEO_POLYGON, CubeOutputFormat.PROJ_POLYGON]:
                # Join polygons
                polygons = [geometry.Polygon(json.loads(each_polygon)) for each_polygon in each_value]

                value, geo_polygon = ITSLiveAnnualMosaics.unite_polygons(each_key, polygons)

                if len(geo_polygon):
                    for each_polygon in geo_polygon:
                        lon.append(Bounds([each[0] for each in each_polygon]).middle_point())
                        lat.append(Bounds([each[1] for each in each_polygon]).middle_point())

            # Save to be used by annual mosaics
            self.attrs[each_key] = value

        # Set center point's longitude and latitude for each polygon (if more than one) of the mosaic
        # Reset to 2 digits of precision
        lat = [round(each_lat, 2) for each_lat in lat]
        lon = [round(each_lon, 2) for each_lon in lon]

        ds.attrs[CompOutputFormat.LATITUDE] = json.dumps(lat)
        ds.attrs[CompOutputFormat.LONGITUDE] = json.dumps(lon)

        # Save attributes for the use by annual mosaics
        self.attrs[CompOutputFormat.LATITUDE] = lat
        self.attrs[CompOutputFormat.LONGITUDE] = lon

        if copy_to_s3:
            ds.attrs['s3'] = os.path.join(s3_bucket, mosaics_dir, mosaics_filename)
            ds.attrs['url'] = ds.attrs['s3'].replace(BatchVars.AWS_PREFIX, BatchVars.HTTP_PREFIX)

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': ITSLiveAnnualMosaics.CHUNK_SIZE, 'y': ITSLiveAnnualMosaics.CHUNK_SIZE})

        # Write mosaic to NetCDF format file
        ITSLiveAnnualMosaics.summary_mosaic_to_netcdf(ds, self.attrs, s3_bucket, mosaics_dir, mosaics_filename, copy_to_s3)

        return mosaics_filename

    @staticmethod
    def unite_polygons(key, polygons):
        """
        Unite provided polygons into one.

        Inputs:
        =======
        key: Attribute type that corresponds to the polygon being processed.
        polygons: collected "geometry" objects that represent polygons

        Returns:
        ========
        values: List of united polygons coordinates as lists (to be consistent with
                other attributes formats).
        geo_polygon: List of geo polygons external coordinates to compute center
                longitude and latitude for.

        """
        # Unite polygons
        united_polygon = unary_union(polygons)
        values = []
        geo_polygon = []

        all_polygons = [united_polygon]

        if isinstance(united_polygon, geometry.MultiPolygon):
            # There are multiple polygons, collect geometry per polygon
            all_polygons = united_polygon.geoms

        for each_obj in all_polygons:
            # By default coordinates are returned as tuple, convert to "list"
            # datatype to be consistent with other data products
            values.append([list(each) for each in each_obj.exterior.coords])

            if key == CubeOutputFormat.GEO_POLYGON:
                # Collect numeric coordinates to calculate center lon/lat for each polygon
                geo_polygon.append(list(each_obj.exterior.coords))

        return (values, geo_polygon)


    @staticmethod
    def summary_mosaic_to_netcdf(ds: xr.Dataset, mosaics_attrs: dict, s3_bucket: str, bucket_dir: str, filename: str, copy_to_s3: bool):
        """
        Store datacube summary mosaics to NetCDF store and "robust" attributes
        to standalone JSON format file (for traceability).
        """
        target_file = filename

        if copy_to_s3:
            target_file = os.path.join(s3_bucket, bucket_dir, filename)

        logging.info(f'Writing summary mosaics to {target_file}')

        # Set encoding
        encoding_settings = {}
        for each in [Coords.X, Coords.Y, CompDataVars.SENSORS]:
            encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

        # Set dtype for "sensor" dimension to S1 so QGIS can at least see the dimension indices.
        # QGIS does not display even indices if dtype==str.
        # encoding_settings.setdefault(CompDataVars.SENSORS, {}).update({'dtype': 'S1', "zlib": True, "complevel": 2, "shuffle": True})
        encoding_settings.setdefault(CompDataVars.SENSORS, {}).update({'dtype': 'S1'})

        two_dim_chunks_settings = (ds.y.size, ds.x.size)
        three_dim_chunks_settings = (1, ds.y.size, ds.x.size)

        compression = {"zlib": True, "complevel": 2, "shuffle": True}

        # Settings for "float" data types
        for each in [
            CompDataVars.SLOPE_V,
            CompDataVars.SLOPE_VX,
            CompDataVars.SLOPE_VY,
            CompDataVars.V0,
            CompDataVars.VX0,
            CompDataVars.VY0,
            CompDataVars.V0_ERROR,
            CompDataVars.VX0_ERROR,
            CompDataVars.VY0_ERROR,
            CompDataVars.V_AMP,
            CompDataVars.VX_AMP,
            CompDataVars.VY_AMP,
            CompDataVars.V_AMP_ERROR,
            CompDataVars.VX_AMP_ERROR,
            CompDataVars.VY_AMP_ERROR,
            CompDataVars.V_PHASE,
            CompDataVars.VX_PHASE,
            CompDataVars.VY_PHASE
        ]:
            _chunks = two_dim_chunks_settings
            if ds[each].ndim == 3:
                _chunks = three_dim_chunks_settings

            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                'dtype': np.float32,
                'chunksizes': _chunks
            })
            encoding_settings[each].update(compression)

        for each in [Coords.X, Coords.Y]:
            encoding_settings.setdefault(each, {}).update({
                'dtype': np.float32
            })
            encoding_settings[each].update(compression)


        # TODO: Change dtype to np.uint32 once count0 is fixed in all composites
        for each in [
            CompDataVars.COUNT0,
            CompDataVars.MAX_DT,
            CompDataVars.SENSOR_INCLUDE
        ]:
            _chunks = two_dim_chunks_settings

            if ds[each].ndim == 3:
                _chunks = three_dim_chunks_settings

            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                'dtype': np.short,
                'chunksizes': _chunks
            })
            encoding_settings[each].update(compression)


        # Set encoding for CompDataVars.OUTLIER_FRAC
        encoding_settings.setdefault(CompDataVars.OUTLIER_FRAC, {}).update({
            DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
            'dtype': np.float32,
            'chunksizes': two_dim_chunks_settings
        })
        encoding_settings[CompDataVars.OUTLIER_FRAC].update(compression)

        logging.info(f'DS: {ds}')
        logging.info(f'DS encoding: {encoding_settings}')

        # Write locally
        ds.to_netcdf(f'{filename}', engine=ITSLiveAnnualMosaics.NC_ENGINE, encoding=encoding_settings)

        attrs_filename = ITSLiveAnnualMosaics.filename_nc_to_json(filename)

        # Write attributes to local JSON file
        with open(attrs_filename, 'w') as fh:
            json.dump(mosaics_attrs, fh, indent=3)

        ds.to_netcdf(f'{filename}', engine=ITSLiveAnnualMosaics.NC_ENGINE, encoding=encoding_settings)

        if copy_to_s3:
            ITSLiveAnnualMosaics.copy_to_s3_bucket(filename, target_file)

            target_file = ITSLiveAnnualMosaics.filename_nc_to_json(target_file)
            ITSLiveAnnualMosaics.copy_to_s3_bucket(attrs_filename, target_file)

    @staticmethod
    def filename_nc_to_json(filename):
        """
        Convert filename from NetCDF format to JSON.
        """
        return filename.replace('.nc', '.json')

    def copy_to_s3_bucket(local_filename, target_s3_filename):
        """
        Copy local NetCDF file to S3 bucket.
        """
        if os.path.exists(local_filename):
            try:
                # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
                # resulting in as many error messages as there are files in Zarr store
                # to copy
                command_line = [
                    "awsv2", "s3", "cp",
                    local_filename,
                    target_s3_filename,
                    "--acl", "bucket-owner-full-control"
                ]

                logging.info(' '.join(command_line))

                command_return = None
                env_copy = os.environ.copy()

                logging.info(f"Copy {local_filename} to {target_s3_filename}")

                command_return = subprocess.run(
                    command_line,
                    env=env_copy,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

                if command_return.returncode != 0:
                    # Report the whole stdout stream as one logging message
                    raise RuntimeError(f"Failed to copy {local_filename} to {target_s3_filename} with returncode={command_return.returncode}: {command_return.stdout}")

            finally:
                # Remove locally written file
                # This is to eliminate out of disk space failures when the same EC2 instance is
                # being re-used by muliple Batch jobs.
                if os.path.exists(local_filename):
                    logging.info(f"Removing local {local_filename}")
                    os.unlink(local_filename)

def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '--processCubesWithinPolygon',
        type=str,
        action='store',
        default=None,
        help="GeoJSON file that stores polygon the cubes centers should belong to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data',
        help="S3 bucket to store datacubes, composites and mosaics [%(default)s]"
    )
    parser.add_argument(
        '-u', '--urlPath',
        type=str,
        action='store',
        default='http://its-live-data.s3.amazonaws.com',
        help="URL for the store in S3 bucket (to provide for easier download option) [%(default)s]"
    )
    parser.add_argument(
        '-d', '--cubeDir',
        type=str,
        action='store',
        default='datacubes/v02',
        help="S3 directory with datacubes [%(default)s]"
    )
    parser.add_argument(
        '-s', '--compositesDir',
        type=str,
        action='store',
        default='composites/annual/v02',
        help="Destination S3 directory with composites [%(default)s]"
    )
    parser.add_argument(
        '-m', '--mosaicsDir',
        type=str,
        action='store',
        default='mosaics/annual/v02',
        help="Destination S3 directory to store mosaics to [%(default)s]"
    )
    parser.add_argument(
        '-r', '--region',
        type=str,
        action='store',
        required=True,
        help="Region to create annual mosaics for"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to process [%(default)s]"
    )
    parser.add_argument(
        '--excludeEPSG',
        type=str,
        action='store',
        default=None,
        help="JSON list of EPSG codes to exclude from the datacube processing [%(default)s]"
    )
    parser.add_argument(
        '--mosaicsEpsgCode',
        type=int,
        action='store',
        default=None,
        help="Target EPSG code for annual mosaics [%(default)s]"
    )
    parser.add_argument(
        '-g', '--gridCellSize',
        type=int,
        default=120,
        help="Grid cell size of input ITS_LIVE datacube composites [%(default)d]."
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not actually submit any AWS Batch jobs'
    )
    parser.add_argument(
        '-n', '--engine',
        type=str,
        required=False,
        default='h5netcdf',
        help="NetCDF engine to use to store NetCDF data to the file [%(default)s]."
    )
    parser.add_argument(
        '-t', '--transformation_matrix_file',
        default='',
        type=str,
        help='Store transformation matrix to provided file and re-use it to build all mosaics for the same region [%(default)s]'
    )
    parser.add_argument(
        '--use_existing_files',
        action='store_true',
        default=False,
        help='Use existing mosaics files if they exist [%(default)s]. This is to pick up from where previous processing stopped.'
    )

    # One of --processCubes or --processCubesFile options is allowed for the datacube names
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default=None,
        help="JSON list of datacubes filenames to process [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=str,
        action='store',
        default=None,
        help="JSON file that contains a list of filenames for datacubes to process [%(default)s]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    BatchVars.HTTP_PREFIX = args.urlPath
    BatchVars.AWS_PREFIX = args.bucket
    ITSLiveAnnualMosaics.REGION = args.region
    ITSLiveAnnualMosaics.CELL_SIZE = args.gridCellSize
    ITSLiveAnnualMosaics.NC_ENGINE = args.engine
    ITSLiveAnnualMosaics.TRANSFORMATION_MATRIX_FILE = args.transformation_matrix_file
    ITSLiveAnnualMosaics.USE_EXISTING_FILES = args.use_existing_files

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        BatchVars.EPSG_TO_GENERATE = epsg_codes

    epsg_codes = list(map(str, json.loads(args.excludeEPSG))) if args.excludeEPSG is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes to exclude: {epsg_codes}")
        BatchVars.EPSG_TO_EXCLUDE = epsg_codes

    # Make sure there is no overlap in EPSG_TO_GENERATE and EPSG_TO_EXCLUDE
    diff = set(BatchVars.EPSG_TO_GENERATE).intersection(BatchVars.EPSG_TO_EXCLUDE)
    if len(diff):
        raise RuntimeError(f"The same code is specified for BatchVars.EPSG_TO_EXCLUDE={BatchVars.EPSG_TO_EXCLUDE} and BatchVars.EPSG_TO_GENERATE={BatchVars.EPSG_TO_GENERATE}")

    if args.processCubesFile:
        # Check for this option first as another mutually exclusive option has a default value
        with open(args.processCubesFile, 'r') as fhandle:
            BatchVars.CUBES_TO_GENERATE = json.load(fhandle)

        # Replace each path by the datacube s3 path
        BatchVars.CUBES_TO_GENERATE = [each.replace(BatchVars.HTTP_PREFIX, BatchVars.AWS_PREFIX) for each in BatchVars.CUBES_TO_GENERATE if len(each)]
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

    elif args.processCubes:
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubes)
        if len(BatchVars.CUBES_TO_GENERATE):
            logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubes}: {BatchVars.CUBES_TO_GENERATE}")

    if args.processCubesWithinPolygon:
        with open(args.processCubesWithinPolygon, 'r') as fhandle:
            shape_file = json.load(fhandle)
            logging.info(f'Reading region polygon the datacube\'s central point should fall into: {args.processCubesWithinPolygon}')

            shapefile_coords = shape_file[CubeJson.FEATURES][0]['geometry']['coordinates']
            logging.info(f'Got polygon coordinates: {shapefile_coords}')
            line = geometry.LineString(shapefile_coords[0][0])
            BatchVars.POLYGON_SHAPE = geometry.Polygon(line)

            logging.info(f'Set polygon: {BatchVars.POLYGON_SHAPE}')

    return args

if __name__ == '__main__':
    import argparse
    import warnings
    import shutil
    import subprocess
    import sys
    from urllib.parse import urlparse

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    args = parse_args()

    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Command arguments: {args}")

    # Set static data for computation
    mosaics = ITSLiveAnnualMosaics(args.mosaicsEpsgCode, args.dryrun)
    result_files = mosaics.create(
        args.cubeDefinitionFile,
        args.bucket,
        args.cubeDir,
        args.compositesDir,
        args.mosaicsDir
    )

    logging.info(f'Created mosaics files: {json.dumps(result_files, indent=3)}')

    logging.info(f"Done.")
