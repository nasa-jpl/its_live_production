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
import json
import h5py
import logging
import numpy as np
import os
import pyproj
import re
import s3fs
import sys
import subprocess
from tqdm import tqdm
import xarray as xr

# Local imports
from itscube_types import DataVars

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

class Encoding:
    """
    Encoding settings for writing ITS_LIVE granule to the file
    """
    IMG_PAIR = {
        'interp_mask':      {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_height': {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_width':  {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'v':                {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vx':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vy':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'img_pair_info':    {'_FillValue': None, 'dtype': np.float32},
        'mapping':          {'_FillValue': None, 'dtype': np.float32},
        'x':                {'_FillValue': None},
        'y':                {'_FillValue': None}
    }

PS = collections.namedtuple("PM", ['platform', 'sensor'])

class NSIDCMeta:
    """
    Class to create premet and spacial files for each of the granules.

    Example of premet file:
    =======================
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

    Example of spacial file:
    ========================
    -94.32	71.86
    -99.41	71.67
    -94.69	73.3
    -100.22	73.09
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

    @staticmethod
    def create_premet_file(infile: str, url_tokens_1, url_tokens_2):
        """
        Create premet file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE granule
        url_tokens_1: Parsed out filename tokens that correspond to the first image of the pair
        url_tokens_2: Parsed out filename tokens that correspond to the second image of the pair
        """
        # Get acquisition dates for both images
        begin_date = datetime.strptime(url_tokens_1[3], DATE_FORMAT)
        end_date = datetime.strptime(url_tokens_2[3], DATE_FORMAT)

        sensor1 = url_tokens_1[0]
        if sensor1 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file(): got unexpected mission+sensor {sensor1} for image#1 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        sensor2 = url_tokens_2[0]
        if sensor2 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file() got unexpected mission+sensor {sensor2} for image#2 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={infile}\n')
            fh.write(f'VersionID_local=001\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Hard-code the values as we don't have timestamps captured within img_pair_info
            fh.write("Begin_time=00:00:01.000\n")
            fh.write("End_time=23:59:59.000\n")

            # Append premet with sensor info
            for each_sensor in [sensor1, sensor2]:
                fh.write(f"Container=AssociatedPlatformInstrumentSensor\n")
                fh.write(f"AssociatedPlatformShortName={NSIDCMeta.ShortName[each_sensor].platform}\n")
                fh.write(f"AssociatedInstrumentShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")
                fh.write(f"AssociatedSensorShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")

        return meta_filename

    @staticmethod
    def create_spacial_file(infile: str):
        """
        Create spatial file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE granule
        """
        meta_filename = f'{infile}.spatial'

        with xr.open_dataset(infile, engine='h5netcdf') as ds:
            xvals = ds.x.values
            yvals = ds.y.values
            pix_size_x = xvals[1] - xvals[0]
            pix_size_y = yvals[1] - yvals[0]

            # minval_x, pix_size_x, _, maxval_y, _, pix_size_y = [float(x) for x in ds['mapping'].attrs['GeoTransform'].split()]

            # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
            projection_cf_minx = xvals[0] - pix_size_x/2.0
            projection_cf_maxx = xvals[-1] + pix_size_x/2.0
            projection_cf_miny = yvals[-1] + pix_size_y/2.0 # pix_size_y is negative!
            projection_cf_maxy = yvals[0] - pix_size_y/2.0  # pix_size_y is negative!

            epsgcode = int(ds['mapping'].attrs['spatial_epsg'][0])

            transformer = pyproj.Transformer.from_crs(f"EPSG:{epsgcode}", "EPSG:4326", always_xy=True) # ensure lonlat output order

            # Convert coordinates to long/lat
            ll_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_miny),decimals = 2).tolist()
            lr_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_miny),decimals = 2).tolist()
            ur_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_maxy),decimals = 2).tolist()
            ul_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_maxy),decimals = 2).tolist()


        # Write to spatial file
        with open(meta_filename, 'w') as fh:
            for long, lat in [ul_lonlat, ll_lonlat, ur_lonlat, lr_lonlat]:
                fh.write(f"{long}\t{lat}\n")

        return meta_filename

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

        if start_index != 0 or stop_index != -1:
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

    def no__call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        ATTN: This method implements sequetial processing for debugging purposes only.

        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spatial
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
            for each in self.infiles[start:start+num_tasks]:
                results = NSIDCFormat.fix_granule(target_bucket, target_dir, each, self.s3)
                logging.info("-->".join(results))

            total_num_files -= num_tasks
            start += num_tasks

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
                logging.info("\n-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks

    @staticmethod
    def upload_to_s3(filename: str, target_dir: str, target_bucket: str, s3_client, remove_original_file: bool=True):
        """
        Upload file to the AWS S3 bucket.
        """
        msgs = []
        target_filename = os.path.join(target_dir, filename)

        try:
            msgs.append(f"Uploading {filename} to {target_bucket}/{target_filename}")

            if not NSIDCFormat.DRY_RUN:
                s3_client.upload_file(filename, target_bucket, target_filename)

                if remove_original_file:
                    msgs.append(f"Removing local {filename}")
                    os.unlink(filename)

        except ClientError as exc:
            msgs.append(f"ERROR: {exc}")

        return msgs

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
        7. Replace 'binary' units
        8. Add standard_name = 'image_pair_information' to img_pair_info
        """
        _missing_value = 'missing_value'
        _meter_year_units = 'meter/year'

        _conventions = 'Conventions'
        _cf_value = 'CF-1.8'

        _transverse_mercator = 'transverse_mercator'

        flag_values = 'flag_values'
        flag_meanings = 'flag_meanings'

        _ocean = 'ocean'
        _ice = 'ice'
        _rock = 'rock'

        binary_flags = np.array([0, 1], dtype=np.uint8)

        _std_name = {
            DataVars.V: 'land_ice_surface_velocity',
            DataVars.VX: 'land_ice_surface_x_velocity',
            DataVars.VY: 'land_ice_surface_y_velocity'
        }

        _binary_meanings = {
            DataVars.INTERP_MASK: 'measured interpolated',
            _ocean: 'non-ocean ocean',
            _ice: 'non-ice ice',
            _rock: 'non-rock rock',
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
        new_filename += f'_{epsg_code:05d}_'
        new_filename += url_tokens_2[7]
        new_filename += '_'
        new_filename += url_tokens_2[8]

        logging.info(f'filename: {infilewithpath}')
        logging.info(f'new_filename: {new_filename}')

        msgs = [f'Processing {infilewithpath} into new {new_filename}']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_granule = os.path.join(target_dir, new_filename)

        # Store granules under 'landsat8' sub-directory in new S3 bucket
        if NSIDCFormat.object_exists(bucket, bucket_granule):
            msgs.append(f'WARNING: {bucket.name}/{bucket_granule} already exists, skipping granule')
            return msgs

        s3_client = boto3.client('s3')

        with s3.open(infilewithpath) as fh:
            with xr.open_dataset(fh) as ds:
                ds.attrs[_conventions] = _cf_value

                # Convert keys to list since we will remove some of the variables
                # during iteration
                for each_var in list(ds.keys()):
                    msgs.append(f'Processing {each_var}')
                    if _missing_value in ds[each_var].attrs:
                        # 3. Remove 'missing_value' attribute
                        del ds[each_var].attrs[_missing_value]

                    if DataVars.UNITS in ds[each_var].attrs and \
                        ds[each_var].attrs[DataVars.UNITS] == DataVars.M_Y_UNITS:
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

                        # Reset to data of int type to get rid of dimensionality
                        ds[DataVars.ImgPairInfo.NAME] = xr.DataArray(
                            attrs=ds[each_var].attrs,
                            coords={},
                            dims=[]
                        )

                    if DataVars.GRID_MAPPING in ds[each_var].attrs:
                        # 2. Replace projection attribute to "mapping"
                        ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

                    elif each_var in [DataVars.UTM_PROJECTION, DataVars.POLAR_STEREOGRAPHIC]:
                        # 6. For UTM_Projection: set grid_mapping_name=transverse_mercator
                        if each_var == DataVars.UTM_PROJECTION:
                            ds[each_var].attrs[DataVars.GRID_MAPPING_NAME] = 'transverse_mercator'

                        # 1. Rename projection variable to 'mapping'
                        # Can't copy the whole data variable, as it introduces obscure coordinates.
                        # Just copy all attributes for the scalar type of the xr.DataArray.
                        ds[DataVars.MAPPING] = xr.DataArray(
                            attrs=ds[each_var].attrs,
                            coords={},
                            dims=[]
                        )

                        # Delete old projection variable
                        msgs.append(f'Deleting {each_var}')
                        del ds[each_var]

                # Write fixed granule to local file
                ds.to_netcdf(new_filename, engine='h5netcdf', encoding = Encoding.IMG_PAIR)

                # Copy new granule to S3 bucket
                msgs.extend(NSIDCFormat.upload_to_s3(new_filename, target_dir, target_bucket, s3_client, remove_original_file=False))

        # Create spacial and premet metadata files, and copy them to S3 bucket
        meta_file = NSIDCMeta.create_premet_file(new_filename, url_tokens_1, url_tokens_2)
        msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        meta_file = NSIDCMeta.create_spacial_file(new_filename)
        msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        msgs.append(f"Removing local {new_filename}")
        os.unlink(new_filename)

        return msgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description="""
           Fix ITS_LIVE V1 velocity image pairs to be CF compliant for
           ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-catalog_dir',
        action='store',
        type=str,
        default='catalog_geojson/landsat/v01',
        help='Output path for feature collections [%(default)s]'
    )

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
        help='AWS S3 directory that stores processed granules [%(default)s]'
    )

    parser.add_argument(
        '-chunk_by',
        action='store',
        type=int,
        default=8,
        help='Number of granules to process in parallel [%(default)d]'
    )

    parser.add_argument(
        '-granules_file',
        action='store',
        type=str,
        default='used_granules_landsat.json',
        help='Filename with JSON list of granules [%(default)s], file is stored in  "-catalog_dir"'
    )

    parser.add_argument(
        '-start_index',
        action='store',
        type=int,
        default=0,
        help="Start index for the granule to fix [%(default)d]. " \
             "Useful if need to continue previously interrupted process to fix the granules."
    )

    parser.add_argument(
        '-stop_index',
        action='store',
        type=int,
        default=-1,
        help="Stop index for the granules to fix [%(default)d]. " \
             "Usefull if need to split the job between multiple processes."
    )

    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers for processing [%(default)d]'
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually process any granules'
    )

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
