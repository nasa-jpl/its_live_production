#!/usr/bin/env python
"""
Fix ITS_LIVE L8 V2 granules:

 1. Organize img_pair_info attributes in alphabetic order.

 2. Fix all references to https://its-live-data.jpl.nasa.gov.s3.amazonaws.com

 3. Recompute S1, S2 and L8 stable shift

 4. Rename v*_error_* and flag_stable_shift attributes

 5. Re-compute date_center and date_dt (some of the granules didn't have
    these attributes updated after acquisition datetime was corrected
    to include time stamp)

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
from fix_v2_sentinel_1_granules import rename_error_attrs

# Old S3 bucket tocken to replace
OLD_S3_NAME = '.jpl.nasa.gov'
NEW_S3_NAME = ''

def fix_all(source_bucket: str, target_bucket: str, granule_url: str, local_dir: str, s3):
    """
    Fix everything in the granule.
    """
    msgs = [f'Processing {granule_url}']

    # get center lat lon
    with s3.open(granule_url) as fhandle:
        with xr.open_dataset(fhandle) as ds:
            img1_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1]
            img2_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2]

            # Re-order attributes in alphabetic order
            # String img_pair_info;
            #   :acquisition_date_img1 = "20150130T16:15:20";
            #   :acquisition_date_img2 = "20160202T16:15:25";
            #   :collection_category_img1 = "T2";
            #   :collection_category_img2 = "T2";
            #   :collection_number_img1 = 2.0; // double
            #   :collection_number_img2 = 2.0; // double
            #   :correction_level_img1 = "L1GT";
            #   :correction_level_img2 = "L1GT";
            #   :mission_img1 = "L";
            #   :mission_img2 = "L";
            #   :path_img1 = 14.0; // double
            #   :path_img2 = 14.0; // double
            #   :processing_date_img1 = "20200924";
            #   :processing_date_img2 = "20200907";
            #   :row_img1 = 121.0; // double
            #   :row_img2 = 121.0; // double
            #   :satellite_img1 = 8.0; // double
            #   :satellite_img2 = 8.0; // double
            #   :sensor_img1 = "O";
            #   :sensor_img2 = "C";
            #   :time_standard_img1 = "UTC";
            #   :time_standard_img2 = "UTC";
            #   :date_center = "20150802";
            #   :date_dt = 368.0; // double
            #   :latitude = -81.7; // double
            #   :longitude = -170.09; // double
            #   :roi_valid_percentage = 57.599999999999994; // double

            old_attrs = copy.deepcopy(ds[DataVars.ImgPairInfo.NAME].attrs)
            for each_key in list(ds[DataVars.ImgPairInfo.NAME].attrs.keys()):
                del ds[DataVars.ImgPairInfo.NAME].attrs[each_key]

            # 1. Organize attributes in alpabetical order
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1] = img1_datetime
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2] = img2_datetime

            ds[DataVars.ImgPairInfo.NAME].attrs['collection_category_img1'] = old_attrs['collection_category_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['collection_category_img2'] = old_attrs['collection_category_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['collection_number_img1'] = old_attrs['collection_number_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['collection_number_img2'] = old_attrs['collection_number_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['correction_level_img1'] = old_attrs['correction_level_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['correction_level_img2'] = old_attrs['correction_level_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['mission_img1'] = old_attrs['mission_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['mission_img2'] = old_attrs['mission_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['path_img1'] = old_attrs['path_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['path_img2'] = old_attrs['path_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['processing_date_img1'] = old_attrs['processing_date_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['processing_date_img2'] = old_attrs['processing_date_img2']

            ds[DataVars.ImgPairInfo.NAME].attrs['row_img1'] = old_attrs['row_img1']
            ds[DataVars.ImgPairInfo.NAME].attrs['row_img2'] = old_attrs['row_img2']

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

            # 4. Rename v*_error_* and flag_stable_shift attributes
            ds = rename_error_attrs(ds)

            granule_basename = os.path.basename(granule_url)

            # 5. Re-compute date_center and date_dt (some of the granules didn't have
            # these attributes updated after acquisition datetime was corrected
            # to include time stamp)
            d0 = parse(img1_datetime)
            d1 = parse(img2_datetime)
            date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
            date_dt = np.float64(date_dt_base)
            if date_dt < 0:
                raise Exception('Input image 1 must be older than input image 2')

            date_ct = d0 + (d1 - d0)/2
            date_center = date_ct.strftime(FixLandsat8Granules.DATETIME_FORMAT).rstrip('0')

            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_DT] = date_dt
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER] = date_center

            # Write the granule locally, upload it to the bucket, remove file
            fixed_file = os.path.join(local_dir, granule_basename)
            ds.to_netcdf(fixed_file, engine='h5netcdf', encoding = Encoding.LANDSAT_SENTINEL2)

            # Upload corrected granule to the bucket
            s3_client = boto3.client('s3')
            bucket_granule = granule_url.replace(source_bucket+'/', '')
            try:
                # Store granules under 'landsat8' sub-directory
                bucket_granule = bucket_granule.replace(FixLandsat8Granules.OLD_SUBDIR, FixLandsat8Granules.NEW_SUBDIR)

                msgs.append(f"Uploading {fixed_file} to {target_bucket}/{bucket_granule}")

                if not FixLandsat8Granules.DRY_RUN:
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
                source_key = target_key.replace(FixLandsat8Granules.NEW_SUBDIR, FixLandsat8Granules.OLD_SUBDIR)

                if FixLandsat8Granules.object_exists(bucket, target_key):
                    msgs.append(f'WARNING: {bucket.name}/{target_key} already exists, skipping upload')

                else:
                    source_dict = {'Bucket': source_bucket,
                                   'Key': source_key}

                    msgs.append(f'Copying {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')
                    if not FixLandsat8Granules.DRY_RUN:
                        bucket.copy(source_dict, target_key)

            return msgs


class FixLandsat8Granules:
    """
    Class to fix ITS_LIVE granules (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    # Flag if dry run is requested - print information about to be done actions
    # without actually invoking commands.
    DRY_RUN = False

    OLD_SUBDIR = 'landsat'
    NEW_SUBDIR = 'landsat8'

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

        if not FixLandsat8Granules.DRY_RUN and not os.path.exists(local_dir):
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
        default='velocity_image_pair/landsat/v02',
        help='AWS S3 directory that stores the granules'
    )
    parser.add_argument(
        '-n', '--new_subdir',
        type=str,
        default='velocity_image_pair/landsat8/v02',
        help='AWS S3 directory that stores target granules'
    )

    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox_landsat8',
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
        help='AWS S3 bucket to store fixed ITS_LIVE granules (under the same "--bucket_dir" directory adjusted for "landsat8" sub-directory)'
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

    FixLandsat8Granules.OLD_SUBDIR = args.bucket_dir
    FixLandsat8Granules.NEW_SUBDIR = args.new_subdir

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    if not os.path.exists(args.local_dir):
        os.mkdir(args.local_dir)

    FixLandsat8Granules.DRY_RUN = args.dry

    fix_attributes = FixLandsat8Granules(
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
