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
        "error_slow",
        "error_slow_description",
        "error_modeled",
        "error_modeled_description"
    ]

    stable_shift_new_attrs = {
        'flag_stable_shift': 'stable_shift_flag',
        'flag_stable_shift_description': 'stable_shift_flag_description'
    }

    for each_var in ['vx', 'vy', 'va', 'vr']:
        if each_var in ds:
            for each_attr in new_attrs:
                old_attr = f'{each_var}_{each_attr}'
                ds[each_var].attrs[each_attr] = ds[each_var].attrs[old_attr]
                del ds[each_var].attrs[old_attr]

            # Rename flag_stable_shift* attrs
            for old_attr, new_attr in stable_shift_new_attrs.items():
                ds[each_var].attrs[new_attr] = ds[each_var].attrs[old_attr]
                del ds[each_var].attrs[old_attr]

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

            # 1. Add missing attributes
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1] = img1_datetime
            ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2] = img2_datetime

            # Remove attributes with wrong names
            del ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_IMG1]
            del ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_IMG2]

            # 2. Fix reference to old its-live-data.jpl.nasa.gov S3 bucket
            ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE] = ds.attrs[DataVars.AUTORIFT_PARAMETER_FILE].replace(OLD_S3_NAME, NEW_S3_NAME)

            # 3. Recompute stable shift
            ds = patch_stable_shift(ds, ds_filename = granule_url)

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
