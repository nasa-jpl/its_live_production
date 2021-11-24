#!/usr/bin/env python
"""
Fix ITS_LIVE S1 V2 granules:

 1. Acquisition attributes (acquisition_img1/2 attribute should be acquisition_date_img1/2)

 2. Fix all references to https://its-live-data.jpl.nasa.gov.s3.amazonaws.com

 3. Recompute S1, S2 and L8 stable shift

 4. Remove vp, vxp, vyp, vp_error layers from S1 layers

 5. Rename v*_error_* and flag_stable_shift attributes

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket.
It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Joe Kennedy
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import copy
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
from dateutil.parser import parse
import json
import logging
import numpy as np
import os
from pathlib import Path
import s3fs
from tqdm import tqdm
import xarray as xr

from mission_info import Encoding
from itscube_types import DataVars

from netcdf_patch_update import main as patch_stable_shift
from netcdf_patch_update import ITSLiveException

# Old S3 bucket tocken to replace
OLD_S3_NAME = '.jpl.nasa.gov'
NEW_S3_NAME = ''


def rename_error_attrs(ds: xr.Dataset):
    """
    Rename v*_error* and flag_stable_shift* attributes of velocity data variables.
    For example, for "vx" data variable:

    vx_error -> error
    vx_error_description -> error_description
    vx_error_mask -> error_mask
    vx_error_mask_description -> error_mask_description
    vx_error_slow -> error_slow
    vx_error_slow_description -> error_slow_description
    vx_error_modeled -> error_modeled
    vx_error_modeled_description -> error_modeled_description

    flag_stable_shift -> stable_shift_flag
    flag_stable_shift_description -> stable_shift_flag_description
    """
    new_attrs = [
        "error",
        "error_description",
        "error_mask",
        "error_mask_description",
        "error_modeled",
        "error_modeled_description",
        "error_slow",
        "error_slow_description"
    ]

    stable_shift_attrs = [
        'stable_shift',
        'flag_stable_shift',
        'flag_stable_shift_description',
        'stable_shift_mask',
        'stable_count_mask',
        'stable_shift_slow',
        'stable_count_slow'
    ]

    stable_shift_new_attrs = {
        'flag_stable_shift': 'stable_shift_flag',
        'flag_stable_shift_description': 'stable_shift_flag_description'
    }

    _rm_keys = ('stable', 'flag', 'grid_mapping')

    # Should be:
    # short vx(y, x) ;
    #     :_FillValue = -32767s ;
    #     :standard_name = "x_velocity" ;
    #     :description = "velocity component in x direction" ;
    #     :units = "m/y" ;
    #     :grid_mapping = "mapping" ;
    #     :error = 15.5 ;
    #     :error_description = "best estimate of x_velocity error: vx_error is populated according to the approach used for the velocity bias correction as indicated in \"stable_shift_flag\"" ;
    #     :error_mask = 15.5 ;
    #     :error_mask_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 m/yr identified from an external mask" ;
    #     :error_modeled = 193.9 ;
    #     :error_modeled_description = "1-sigma error calculated using a modeled error-dt relationship" ;
    #     :error_slow = 15.5 ;
    #     :error_slow_description = "RMSE over slowest 25% of retrieved velocities" ;
    #     :stable_shift = 1.5 ;
    #     :stable_shift_flag = 1LL ;
    #     :stable_shift_flag_description = "flag for applying velocity bias correction: 0 = no correction; 1 = correction from overlapping stable surface mask (stationary or slow-flowing surfaces with velocity < 15 m/yr)(top priority); 2 = correction from slowest 25% of overlapping velocities (second priority)" ;
    #     :stable_shift_mask = 1.5 ;
    #     :stable_count_mask = 272579LL ;
    #     :stable_shift_slow = 1.5 ;
    #     :stable_count_slow = 275326LL ;

    # Rename attributes (and TODO: re-order them to be in alphabetic order)
    for each_var in ['vx', 'vy', 'va', 'vr']:
        if each_var in ds:
            # Save old attributes values
            old_attrs = copy.deepcopy(ds[each_var].attrs)

            for each_key in list(ds[each_var].attrs.keys()):
                if each_key.startswith(_rm_keys):
                    del ds[each_var].attrs[each_key]

            ds[each_var].attrs['grid_mapping'] = old_attrs['grid_mapping']

            # Rename error attributes
            for each_attr in new_attrs:
                old_attr_name = f'{each_var}_{each_attr}'
                ds[each_var].attrs[each_attr] = old_attrs[old_attr_name]
                del ds[each_var].attrs[old_attr_name]

            # Insert (and rename some) stable_shift* attrs
            for each_attr in stable_shift_attrs:
                new_attr = each_attr
                if each_attr in stable_shift_new_attrs:
                    new_attr = stable_shift_new_attrs[each_attr]

                ds[each_var].attrs[new_attr] = old_attrs[each_attr]

    return ds

def fix_all(source_bucket: str, target_bucket: str, granule_url: str, local_dir: str, s3):
    """
    Fix everything in the granule.
    """
    msgs = [f'Processing {granule_url}']

    # get center lat lon
    with s3.open(granule_url) as fhandle:
        with xr.open_dataset(fhandle) as ds:
            img1_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_IMG1]
            img2_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_IMG2]

            # Re-order attributes in alphabetic order
            # String img_pair_info;
            #       :absolute_orbit_number_img1 = "004546";
            #       :absolute_orbit_number_img2 = "004721";
            #       :acquisition_date_img1 = "20170303T15:31:44.753674";
            #       :acquisition_date_img2 = "20170315T15:31:45.105188";
            #       :flight_direction_img1 = "descending";
            #       :flight_direction_img2 = "descending";
            #       :mission_data_take_ID_img1 = "007EB7";
            #       :mission_data_take_ID_img2 = "0083E4";
            #       :mission_img1 = "S";
            #       :mission_img2 = "S";
            #       :product_unique_ID_img1 = "2737";
            #       :product_unique_ID_img2 = "3CBC";
            #       :satellite_img1 = "1B";
            #       :satellite_img2 = "1B";
            #       :sensor_img1 = "C";
            #       :sensor_img2 = "C";
            #       :time_standard_img1 = "UTC";
            #       :time_standard_img2 = "UTC";
            #       :date_center = "20170309T15:31:44.929431";
            #       :date_dt = 12.000004068449075; // double
            #       :latitude = 58.56; // double
            #       :longitude = -138.04; // double
            #       :roi_valid_percentage = 80.2; // double
            old_attrs = copy.deepcopy(ds[DataVars.ImgPairInfo.NAME].attrs)
            for each_key in list(ds[DataVars.ImgPairInfo.NAME].attrs.keys()):
                del ds[DataVars.ImgPairInfo.NAME].attrs[each_key]

            ds[DataVars.ImgPairInfo.NAME].attrs['absolute_orbit_number_img1'] = old_attrs['absolute_orbit_number_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['absolute_orbit_number_img2'] = old_attrs['absolute_orbit_number_img2']

            # 1. Add missing attributes
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1] = img1_datetime
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2] = img2_datetime

            ds[DataVars.ImgPairInfo.NAME].attrs['flight_direction_img1'] = old_attrs['flight_direction_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['flight_direction_img2'] = old_attrs['flight_direction_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['mission_data_take_ID_img1'] = old_attrs['mission_data_take_ID_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['mission_data_take_ID_img2'] = old_attrs['mission_data_take_ID_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['mission_img1'] = old_attrs['mission_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['mission_img2'] = old_attrs['mission_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['product_unique_ID_img1'] = old_attrs['product_unique_ID_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['product_unique_ID_img2'] = old_attrs['product_unique_ID_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['satellite_img1'] = old_attrs['satellite_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['satellite_img2'] = old_attrs['satellite_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['sensor_img1'] = old_attrs['sensor_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['sensor_img2'] = old_attrs['sensor_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['time_standard_img1'] = old_attrs['time_standard_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['time_standard_img2'] = old_attrs['time_standard_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['date_center'] = old_attrs['date_center']
            ds[DataVars.ImgPairInfo.NAME].attrs['date_dt'] = old_attrs['date_dt']
            ds[DataVars.ImgPairInfo.NAME].attrs['latitude'] = old_attrs['latitude']
            ds[DataVars.ImgPairInfo.NAME].attrs['longitude'] = old_attrs['longitude']
            ds[DataVars.ImgPairInfo.NAME].attrs['roi_valid_percentage'] = old_attrs['roi_valid_percentage']

            # 2. Fix reference to old its-live-data.jpl.nasa.gov S3 bucket
            ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE] = ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE].replace(OLD_S3_NAME, NEW_S3_NAME)

            # 3. Recompute stable shift
            try:
                ds = patch_stable_shift(ds, ds_filename = granule_url)

            except ITSLiveException as exc:
                # A granule with ROI=0 is used for cataloging purposes only,
                # skip conversion
                msgs.append(f'WARNING: Skip stable shift corrections for ROI=0: {exc}')

            # 4. Remove vp, vxp, vyp, vp_error layers after stable shift re-calculations
            # as it uses v*p* variables
            del ds[DataVars.VP]
            del ds[DataVars.VP_ERROR]
            del ds[DataVars.VXP]
            del ds[DataVars.VYP]

            # 5. Rename v*_error_* and flag_stable_shift attributes
            ds = rename_error_attrs(ds)

            granule_basename = os.path.basename(granule_url)

            # Re-compute date_center and date_dt (some of the granules didn't have
            # these attributes updated after acquisition datetime was corrected
            # to include time stamp)
            d0 = parse(img1_datetime)
            d1 = parse(img2_datetime)
            date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
            date_dt = np.float64(date_dt_base)
            if date_dt < 0:
                raise Exception('Input image 1 must be older than input image 2')

            date_ct = d0 + (d1 - d0)/2
            date_center = date_ct.strftime(FixSentinel1Granules.DATETIME_FORMAT).rstrip('0')

            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_DT] = date_dt
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER] = date_center

            # Write the granule locally, upload it to the bucket, remove file
            fixed_file = os.path.join(local_dir, granule_basename)
            ds.to_netcdf(fixed_file, engine='h5netcdf', encoding = Encoding.SENTINEL1)

            # Upload corrected granule to the bucket
            s3_client = boto3.client('s3')
            try:
                bucket_granule = granule_url.replace(source_bucket+'/', '')
                msgs.append(f"Uploading {bucket_granule} to {target_bucket}")

                if not FixSentinel1Granules.DRY_RUN:
                    s3_client.upload_file(fixed_file, target_bucket, bucket_granule)

                msgs.append(f"Removing local {fixed_file}")
                os.unlink(fixed_file)

            except ClientError as exc:
                msgs.append(f"ERROR: {exc}")

            # There are corresponding browse and thumbprint images to transfer
            bucket = boto3.resource('s3').Bucket(target_bucket)
            source_ext = '.nc'

            for target_ext in ['.png', '_thumb.png']:
                # It's an extra file to transfer, replace extension
                target_key = bucket_granule.replace(source_ext, target_ext)

                if FixSentinel1Granules.object_exists(bucket, target_key):
                    msgs.append(f'WARNING: {bucket.name}/{target_key} already exists, skipping upload')

                else:
                    source_dict = {'Bucket': source_bucket,
                                   'Key': target_key}

                    msgs.append(f'Copying {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')
                    if not FixSentinel1Granules.DRY_RUN:
                        bucket.copy(source_dict, target_key)

            return msgs


class FixSentinel1Granules:
    """
    Class to fix ITS_LIVE granules (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    # Flag if dry run is requested - print information about to be done actions
    # without actually invoking commands.
    DRY_RUN = False

    # Date and time format used by ITS_LIVE granules
    DATETIME_FORMAT = '%Y%m%dT%H:%M:%S.%f'

    def __init__(self,
        bucket: str,
        bucket_dir: str,
        glob_pattern: dir
    ):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_granules = self.s3.glob(f'{os.path.join(bucket, bucket_dir)}/{glob_pattern}')

        logging.info(f"Number of all granules: {len(self.all_granules)}")

        # Exclude granules previously fixed: the ones that have suffix
        # self.all_granules = [each for each in self.all_granules if 'LC08' in each]
        self.bucket = bucket

    @staticmethod
    def object_exists(bucket, key: str) -> bool:
        try:
            bucket.Object(key).load()

        except ClientError:
            return False

        return True

    def __call__(self, target_bucket: str, local_dir: str, chunk_size: int, num_dask_workers: int, start_index: int):
        """
        Fix ITS_LIVE granules which are stored in AWS S3 bucket.
        """
        num_to_fix = len(self.all_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not FixSentinel1Granules.DRY_RUN and not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(fix_all)(self.bucket, target_bucket, each, local_dir, self.s3) for each in self.all_granules[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=num_dask_workers
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size',
        type=int,
        default=100, help='Number of granules to fix in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data.jpl.nasa.gov',
        help='AWS S3 bucket that stores ITS_LIVE granules to fix'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1/v02',
        help='AWS S3 directory that stores the granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox_sentinel1',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-glob', action='store',
        type=str,
        default='*/*.nc',
        help='Glob pattern for the granule search under "s3://bucket/dir/" [%(default)s]')
    parser.add_argument(
        '-t', '--target_bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store fixed ITS_LIVE granules (under the same "--bucket_dir" directory)'
    )
    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start-granule',
        type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run, do not apply any fixes to the granules stored in AWS S3 bucket'
    )
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    if not os.path.exists(args.local_dir):
        os.mkdir(args.local_dir)

    FixSentinel1Granules.DRY_RUN = args.dry

    fix_attributes = FixSentinel1Granules(
        args.bucket,
        args.bucket_dir,
        args.glob
    )
    fix_attributes(
        args.target_bucket,
        args.local_dir,
        args.chunk_size,
        args.dask_workers,
        args.start_granule
    )


if __name__ == '__main__':
    main()

    logging.info("Done.")
