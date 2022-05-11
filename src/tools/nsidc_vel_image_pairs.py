"""
Script to prepare V1 ITS_LIVE granules to be ingested by NSIDC:

* count: fix units to ‘count’

* vx_err and vy_err: remove 'missing_value' attribute from any data variable that has it

* dt: change 'long_name' to 'error weighted average time separation between image-pairs'

* Add: Conventions = "CF-1.9" to PAT_G0120_0000.nc like data products
* Change: Conventions = "CF-1.9" to velocity image pair products

* Fix "m/y" units to "meter/year" for all variables that the unit is applicable for
* Replace 'binary' units for 'interp_mask' by:
    flag_values = 0UB, 1UB; // ubyte
    flag_meanings = 'measured, interpolated'

* UTM_Projection, Polar_Stereographic: replace by 'mapping' variable
* UTM_Projection: change grid_mapping_name: "universal_transverse_mercator" to
  "transverse_mercator"

* Changing standard_name for vx, vy, and v to:
  “land_ice_surface_x_velocity”, “land_ice_surface_y_velocity” and “land_ice_surface_velocity”

* Add standard_name = 'image_pair_information' to img_pair_info

Rename original file:
LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_X_LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_G0240V01_PXYZ.nc
to
LXSSLLLLPPPRRRYYYYMMDDCCTXX_LXSSLLLLPPPRRRYYYYMMDDCCTX_EEEEE_G0240V01_XYZ.nc
where EEEEE is EPSG code from subdirectory name of original file.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
from botocore.exceptions import ClientError
import collections
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import fsspec
import json
import h5py
import logging
import os
import psutil
import pyproj
import s3fs
import sys
import time
from tqdm import tqdm
import xarray as xr

# Local imports
from itscube_types import DataVars
from mission_info import Encoding


# Date format as it appears in granules filenames of optical format:
# LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc
DATE_FORMAT = "%Y%m%d"

# Date and time format as it appears in granules filenames in radar format:
# S1A_IW_SLC__1SSH_20170221T204710_20170221T204737_015387_0193F6_AB07_X_S1B_IW_SLC__1SSH_20170227T204628_20170227T204655_004491_007D11_6654_G0240V02_P094.nc
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S"

def get_tokens_from_filename(filename):
    """
    Extract acquisition/processing dates and path/row for two images from the
    optical granule filename, or start/end date/time and product unique ID for
    radar granule filename.
    """
    # ATTN: Optical format granules have different file naming convention than radar
    # format granules
    url_files = os.path.basename(filename).split('_X_')

    # Get tokens for the first image name
    url_tokens_1 = url_files[0].split('_')

    if len(url_tokens_1) < 9:
        # Optical format granule
        # Get acquisition/processing dates and path&row for both images
        first_date_1 = datetime.strptime(url_tokens_1[3], DATE_FORMAT)
        second_date_1 = datetime.strptime(url_tokens_1[4], DATE_FORMAT)
        key_1 = url_tokens_1[2]

        # Extract info from second part of the granule's filename
        url_tokens_2 = url_files[1].split('_')
        first_date_2 = datetime.strptime(url_tokens_2[3], DATE_FORMAT)
        second_date_2 = datetime.strptime(url_tokens_2[4], DATE_FORMAT)
        key_2 = url_tokens_2[2]

    else:
        logging.info(f'Unexpected filename format: {filename}')

    # first_date_1, second_date_1, key_1, first_date_2, second_date_2, key_2
    return (url_tokens_1, url_tokens_2)

PS = collections.namedtuple("PM", ['platform', 'sensor'])

class NSIDCPremetFile:
    """
    Class to create premet files for each of the granules in the following format:

    FileName=LC08_L1GT_001111_20140217_20170425_01_T2_X_LC08_L1GT_001111_20131113_20170428_01_T2_G0240V01_P006.nc
    VersionID_local=001
    Begin_date=2013-11-13
    End_date=2017-04-28
    Begin_time=00:00:01.000
    End_time=23:59:59.000
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-8
    AssociatedInstrumentShortName=OLI
    AssociatedSensorShortName=OLI
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-8
    AssociatedInstrumentShortName=TIRS
    AssociatedSensorShortName=TIRS
    """

    # Dictionary of metadata values based on the mission+sensor token
    L8 = 'LC08'
    L7 = 'LE07'
    L5 = 'LT05'
    L4 = 'LT04'

    ShortName = {
        L8: PS('LANDSAT-8', 'OLI'),
        L7: PS('LANDSAT-7', 'ETM+'),
        L5: PS('LANDSAT-5', 'TM'),
        L4: PS('LANDSAT-4', 'TM')
    }

class NSIDCFormat:
    """
    Class to prepare V1 ITS_LIVE data for ingest by NSIDC:
    1. Make V1 ITS_LIVE data CF-1.9 convention compliant.
    2. Generate metadata files required by NSIDC ingest (premet and spacial metadata files
       which are generated per each data product being ingested).
    """
    GRANULES_FILE = 'used_granules_landsat.json'

    # Flag to enable dry run: don't process any granules, just report what would be processed
    DRY_RUN = False

    def __init__(
        self,
        start_index: int=0,
        stop_index: int=-1
    ):
        """
        Initialize the object.
        """
        # S3FS to access files stored in S3 bucket
        self.s3 = s3fs.S3FileSystem(anon=True)

        # Granule files as read from the S3 granule summary file
        self.infiles = None
        logging.info(f"Opening granules file: {NSIDCFormat.GRANULES_FILE}")

        with self.s3.open(NSIDCFormat.GRANULES_FILE, 'r') as ins3file:
            self.infiles = json.load(ins3file)
            logging.info(f"Loaded {len(self.infiles)} granules from '{NSIDCFormat.GRANULES_FILE}'")

        if start_index != 0:
            # Start index is provided for the granule to begin with
            if stop_index != -1:
                self.infiles = self.infiles[start_index:stop_index]

            else:
                self.infiles = self.infiles[start_index:]

            logging.info(f"Starting with granule #{start_index} (stop={stop_index}), remains {len(self.infiles)} granules to fix")

    @staticmethod
    def object_exists(bucket, key: str) -> bool:
        """
        Returns true if file exists in the bucket, False otherwise.
        """
        try:
            bucket.Object(key).load()

        except ClientError:
            return False

        return True

    def __call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spacial
        and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to catalog, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        file_list = []
        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            tasks = [dask.delayed(NSIDCFormat.fix_granule)(target_bucket, target_dir, each, self.s3) for each in self.infiles[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks


    @staticmethod
    def fix_granule(target_bucket: str, target_dir: str, infilewithpath: str, s3):
        """
        Fix granule format:
        1. Rename projection variable to be 'mapping'
        2. Replace all projection attributes to new value of 'mapping'
        3. Remove 'missing_value' attribute
        4. Replace units = "m/y" to units='meter/year'
        5. Change standard_name for vx, vy, and v to:
          “land_ice_surface_x_velocity”, “land_ice_surface_y_velocity” and “land_ice_surface_velocity”
        6. For UTM_Projection: set grid_mapping_name=transverse_mercator
        7. Add standard_name = 'image_pair_information' to img_pair_info
        """
        _missing_value = 'missing_value'
        _meter_year_units = 'meter/year'

        _conventions = 'Conventions'
        _cf_value = 'CF-1.9'

        _transverse_mercator = 'transverse_mercator'

        flag_values = 'flag_values'
        flag_meanings = 'flag_meanings'

        _ocean = 'ocean'
        _ice = 'ice'
        _rock = 'rock'

        binary_flags = '0UB, 1UB'

        _std_name = {
            DataVars.V: 'land_ice_surface_velocity',
            DataVars.VX: 'land_ice_surface_x_velocity',
            DataVars.VY: 'land_ice_surface_y_velocity'
        }

        _binary_meanings = {
            DataVars.INTERP_MASK: 'measured, interpolated',
            _ocean: 'non-ocean, ocean',
            _ice: 'non-ice, ice',
            _rock: 'non-rock, rock',
        }

        _image_pair_info = 'image_pair_information'

        filename_tokens = infilewithpath.split('/')
        directory = '/'.join(filename_tokens[1:-1])
        filename = filename_tokens[-1]

        # Parent subdirectory is EPSG code
        epsg_code = int(filename_tokens[-2])

        # Extract tokens from the filename
        url_tokens_1, url_tokens_2 = get_tokens_from_filename(filename)

        # Format target filename for the granule to be within 80 characters long
        # from
        # LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_X_LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_G0240V01_PXYZ.nc
        # to
        # LXSSLLLLPPPRRRYYYYMMDDCCTXX_LXSSLLLLPPPRRRYYYYMMDDCCTX_EEEEE_G0240V01_XYZ.nc
        new_filename = ''.join(url_tokens_1[:4]) + url_tokens_1[5] + url_tokens_1[6] + '_'
        new_filename += ''.join(url_tokens_2[:4]) + url_tokens_2[5] + url_tokens_2[6]
        new_filename += f'{epsg_code:05d}'
        new_filename += url_tokens_2[7]
        new_filename += url_tokens_2[8]

        msgs = [f'Processing {infilewithpath}: {new_filename}']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_granule = os.path.join(target_dir, new_filename)

        # Store granules under 'landsat8' sub-directory in new S3 bucket
        if NSIDCFormat.object_exists(bucket, bucket_granule):
            msgs.append(f'WARNING: {bucket.name}/{bucket_granule} already exists, skipping granule')
            return msgs

        with fsspec.open(infilewithpath) as fh:
            with xr.open_dataset(fh) as ds:

                ds.attrs[_conventions] = _cf_value

                # Convert keys to list since we will remove some of the variables
                # during iteration
                for each_var in list(ds.keys()):
                    if _missing_value in ds[each_var].attrs:
                        # 3. Remove 'missing_value' attribute
                        del ds[each_var].attrs[_missing_value]

                    if DataVars.UNITS in ds[each_var].attrs and \
                        ds[each_var].attrs[DataVars.UNITS].value == DataVars.M_Y_UNITS:
                        # 4. Replace units = "m/y" to units='meter/year'
                        ds[each_var].attrs[DataVars.UNITS] = _meter_year_units

                    # 5. Change standard_name for vx, vy, and v
                    if each_var in _std_name:
                        ds[each_var].attrs[DataVars.STD_NAME] = _std_name[each_var]

                    # 7. Replace 'binary' units
                    if each_var in [DataVars.INTERP_MASK, _ocean, _ice, _rock] and DataVars.UNITS in ds[each_var].attrs:
                        del ds[each_var].attrs[DataVars.UNITS]
                        ds[each_var].attrs[flag_values] = binary_flags
                        ds[each_var].attrs[flag_meanings] = _binary_meanings[each_var]

                    # 8. Add standard_name = 'image_pair_information' to img_pair_info
                    if each_var == DataVars.ImgPairInfo.NAME:
                        ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.STD_NAME] = _image_pair_info

                    if DataVars.GRID_MAPPING in each_var.attrs:
                        # 2. Replace projection attribute to "mapping"
                        ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

                    elif each_var in [DataVars.UTM_PROJECTION, DataVars.POLAR_STEREOGRAPHIC]:
                        # 6. For UTM_Projection: set grid_mapping_name=transverse_mercator
                        if each_var == DataVars.UTM_PROJECTION:
                            ds[each_var].attrs[DataVars.GRID_MAPPING_NAME] = 'transverse_mercator'

                        # 1. Rename projection variable to 'mapping'
                        ds[DataVars.MAPPING] = ds[each_var].rename(DataVars.MAPPING)

                        # Delete old projection variable
                        del ds[each_var]

                # Write fixed granule to local file
                ds.to_netcdf(new_filename, engine='h5netcdf', encoding = Encoding.LANDSAT_SENTINEL2)

                # TODO: Copy new granule to S3 bucket

                # TODO: Create spacial and premet metadata files

                # TODO: Copy spacial and premet metadata files to S3 bucket

        return msgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description="""
           Fix ITS_LIVE V1 velocity image pairs to be CF compliant for
           ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-catalog_dir',
                        action='store',
                        type=str,
                        default='catalog_geojson/landsat/v01',
                        help='Output path for feature collections [%(default)s]')

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE granules to [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/v01/velocity_image_pair/',
        help='AWS S3 directory that stores processed granules'
    )

    parser.add_argument('-chunk_by',
                        action='store',
                        type=int,
                        default=8,
                        help='Number of granules to process in parallel [%(default)d]')

    parser.add_argument('-granules_file',
                        action='store',
                        type=str,
                        default='used_granules_landsat.json',
                        help='Filename with JSON list of granules [%(default)s], file is stored in  "-catalog_dir"')

    parser.add_argument('-start_index',
                        action='store',
                        type=int,
                        default=0,
                        help="Start index for the granule to fix [%(default)d]. " \
                             "Useful if need to continue previously interrupted process to fix the granules.")

    parser.add_argument('-stop_index',
                        action='store',
                        type=int,
                        default=-1,
                        help="Stop index for the granules to fix [%(default)d]. " \
                             "Usefull if need to split the job between multiple processes.")

    parser.add_argument('-w', '--dask_workers',
                        type=int,
                        default=4,
                        help='Number of Dask parallel workers for processing [%(default)d]')

    parser.add_argument('--dryrun',
                        action='store_true',
                        help='Dry run, do not actually process any granules')

    args = parser.parse_args()

    NSIDCFormat.GRANULES_FILE = os.path.join(args.bucket, args.catalog_dir, args.granules_file)
    NSIDCFormat.DRY_RUN = args.dryrun

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = NSIDCFormat(
        args.start_index,
        args.stop_index
    )
    nsidc_format(
        args.bucket,
        args.target_dir,
        args.chunk_by,
        args.dask_workers
    )

    logging.info('Done.')