#!/usr/bin/env python
"""
Fix and transfer (with provided "fixed" filename) ITS_LIVE S2 V2 granules from ASF:

 1. Add missing sensor_img1="MSI" and sensor_img2="MSI" attributes for img_pair_info

 2. Fix all references to https://its-live-data-eu.jpl.nasa.gov.s3.amazonaws.com to point
    to https://its-live-data.s3.amazonaws.com

 3. Recompute S1, S2 and L8 stable shift

 4. Rename v*_error_* and flag_stable_shift attributes

To run the script you need to have credentials for:
1. https://urs.earthdata.nasa.gov (register for free if you don't have an account).
Place credentials into the file:
echo 'machine urs.earthdata.nasa.gov login USERNAME password PASSWORD' >& ~/.netrc

Authors: Masha Liukis, Joe Kennedy, Mark Fahnestock
"""
import argparse
import copy
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta
from dateutil.parser import parse
import json
import logging
import os
import pandas as pd
from pathlib import Path

import boto3
import fsspec
import xarray as xr
from botocore.exceptions import ClientError

import hyp3_sdk as sdk
import numpy as np

from mission_info import Encoding
from itscube_types import DataVars

from netcdf_patch_update import main as patch_stable_shift
from netcdf_patch_update import ITSLiveException
from fix_v2_sentinel_1_granules import rename_error_attrs

# Old S3 bucket tocken to replace
OLD_S3_NAME = 'its-live-data-eu.jpl.nasa.gov'
NEW_S3_NAME = 'its-live-data'

SENSOR_NAME = 'MSI'

# Date and time format used by ITS_LIVE granules
DATETIME_FORMAT = '%Y%m%dT%H:%M:%S.%f'

#
# Author: Mark Fahnestock
#
def point_to_prefix(dir_path: str, lat: float, lon: float) -> str:
    """
    Returns a string (for example, N78W124) for directory name based on
    granule centerpoint lat,lon
    """
    NShemi_str = 'N' if lat >= 0.0 else 'S'
    EWhemi_str = 'E' if lon >= 0.0 else 'W'

    outlat = int(10*np.trunc(np.abs(lat/10.0)))
    if outlat == 90: # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80

    outlon = int(10*np.trunc(np.abs(lon/10.0)))

    if outlon >= 180: # if you are at the dateline, back off to the 170 bin
        outlon = 170

    dirstring = os.path.join(dir_path, f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}')
    return dirstring

def fix_all(ds: xr.Dataset, granule_url: str):
    """
    Fix everything in the granule.
    """
    msg = None
    # 1. Add missing attributes: preserve original insert order of the attributes
    # String img_pair_info;
    #       :acquisition_date_img1 = "20190429T21:08:09.";
    #       :acquisition_date_img2 = "20201010T21:08:11.";
    #       :correction_level_img1 = "L1C";
    #       :correction_level_img2 = "L1C";
    #       :mission_img1 = "S";
    #       :mission_img2 = "S";
    #       :satellite_img1 = "2A";
    #       :satellite_img2 = "2A";
    #       :sensor_img1 = "MSI";
    #       :sensor_img2 = "MSI";
    #       :time_standard_img1 = "UTC";
    #       :time_standard_img2 = "UTC";
    #       :date_center = "20200119T21:08:10.";
    #       :date_dt = 530.0000231481481; // double
    #       :latitude = 61.7; // double
    #       :longitude = -144.05; // double
    #       :roi_valid_percentage = 51.0; // double
    old_attrs = copy.deepcopy(ds[DataVars.ImgPairInfo.NAME].attrs)
    for each_key in list(ds[DataVars.ImgPairInfo.NAME].attrs.keys()):
        del ds[DataVars.ImgPairInfo.NAME].attrs[each_key]

    # Re-populate new dictionary
    ds[DataVars.ImgPairInfo.NAME].attrs['acquisition_date_img1'] = old_attrs['acquisition_date_img1']
    ds[DataVars.ImgPairInfo.NAME].attrs['acquisition_date_img2'] = old_attrs['acquisition_date_img2']
    ds[DataVars.ImgPairInfo.NAME].attrs['correction_level_img1'] = old_attrs['correction_level_img1']
    ds[DataVars.ImgPairInfo.NAME].attrs['correction_level_img2'] = old_attrs['correction_level_img2']
    ds[DataVars.ImgPairInfo.NAME].attrs['mission_img1'] = old_attrs['mission_img1']
    ds[DataVars.ImgPairInfo.NAME].attrs['mission_img2'] = old_attrs['mission_img2']
    ds[DataVars.ImgPairInfo.NAME].attrs['satellite_img1'] = old_attrs['satellite_img1']
    ds[DataVars.ImgPairInfo.NAME].attrs['satellite_img2'] = old_attrs['satellite_img2']
    ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG2] = SENSOR_NAME
    ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG1] = SENSOR_NAME
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
        msg = f'WARNING: Skip stable shift corrections for ROI=0: {exc}'

    # 4. Rename v*_error_* and flag_stable_shift attributes
    ds = rename_error_attrs(ds)

    # Re-compute date_center and date_dt (some of the granules didn't have
    # these attributes updated after acquisition datetime was corrected
    # to include time stamp)
    img1_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1]
    img2_datetime = ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2]

    d0 = parse(img1_datetime)
    d1 = parse(img2_datetime)
    date_dt_base = (d1 - d0).total_seconds() / timedelta(days=1).total_seconds()
    date_dt = np.float64(date_dt_base)
    if date_dt < 0:
        raise Exception(f'{granule_url}: input image 1 must be older than input image 2')

    date_ct = d0 + (d1 - d0)/2
    date_center = date_ct.strftime(DATETIME_FORMAT).rstrip('0')

    ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_DT] = date_dt
    ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER] = date_center

    return ds, msg


class ASFTransfer:
    """
    Class to handle ITS_LIVE granule transfer from ASF to ITS_LIVE bucket.
    """
    PROCESSED_JOB_IDS = []

    # Some of original granules included the postfix that needs to be removed
    # from the target filename
    POSTFIX_TO_RM = '_IL_ASF_OD'

    # HyP3 API access
    HYP3_AUTORIFT_API = None
    # Use test environment for the ingest of L8_memlimit_rerun.json
    # HYP3_AUTORIFT_API = 'https://hyp3-test-api.asf.alaska.edu'

    HYP3 = None
    TARGET_BUCKET = None
    TARGET_BUCKET_DIR = None
    PROCESSED_JOBS_FILE = None
    LOCAL_DIR = None
    DRY_RUN = False

    def __init__(self, processed_jobs_file: str):
        self.processed_jobs_file = processed_jobs_file

    def run_sequentially(
        self,
        job_ids_file: str,
        exclude_job_ids_file: str,
        chunks_to_copy: int,
        start_job: int
    ):
        """
        Run the transfer of granules from ASF to ITS_LIVE S3 bucket.

        If provided, don't process job IDs listed in exclude_job_ids_file (
        as previously processed).
        """
        # job_ids = json.loads(job_ids_file.read_text())
        jobs = pd.read_csv(job_ids_file)

        if exclude_job_ids_file is not None:
            exclude_ids = json.loads(exclude_job_ids_file.read_text())

            # Remove exclude_ids from the jobs to process
            # job_ids = list(set(job_ids).difference(exclude_ids))
            jobs = jobs.loc[~jobs.job_id.isin(exclude_ids)].copy()

        # total_num_to_copy = len(job_ids)
        total_num_to_copy = len(jobs)
        num_to_copy = total_num_to_copy - start_job
        start = start_job
        logging.info(f"{num_to_copy} out of {total_num_to_copy} granules to copy...")

        while num_to_copy > 0:
        # while num_to_copy == total_num_to_copy:
            num_tasks = chunks_to_copy if num_to_copy > chunks_to_copy else num_to_copy

            logging.info(f"Starting tasks {start}:{start+num_tasks} out of {total_num_to_copy} total")
            for id, out_name in jobs.iloc[start:start+num_tasks].itertuples(index=False):
                each_result, _ = ASFTransfer.copy_granule(id, out_name)
            # for id in job_ids[start:start+num_tasks]:
            #     each_result, _ = ASFTransfer.copy_granule(id)
                logging.info("-->".join(each_result))
                ASFTransfer.PROCESSED_JOB_IDS.append(id)

            num_to_copy -= num_tasks
            start += num_tasks

        # Store job IDs that were processed
        with open(self.processed_jobs_file , 'w') as outfile:
            json.dump(ASFTransfer.PROCESSED_JOB_IDS, outfile)

    def __call__(
        self,
        job_ids_file: str,
        exclude_job_ids_file: str,
        chunks_to_copy: int,
        start_job: int,
        num_dask_workers: int
    ):
        """
        Run the transfer of granules from ASF to ITS_LIVE S3 bucket.

        If provided, don't process job IDs listed in exclude_job_ids_file (
        as previously processed).
        """
        # job_ids = json.loads(job_ids_file.read_text())
        jobs = pd.read_csv(job_ids_file)

        if exclude_job_ids_file is not None:
            exclude_ids = json.loads(exclude_job_ids_file.read_text())

            # Remove exclude_ids from the jobs to process
            # job_ids = list(set(job_ids).difference(exclude_ids))
            jobs = jobs.loc[~jobs.job_id.isin(exclude_ids)].copy()

        # total_num_to_copy = len(job_ids)
        total_num_to_copy = len(jobs)
        num_to_copy = total_num_to_copy - start_job
        start = start_job
        logging.info(f"{num_to_copy} out of {total_num_to_copy} granules to copy...")

        while num_to_copy > 0:
            num_tasks = chunks_to_copy if num_to_copy > chunks_to_copy else num_to_copy

            logging.info(f"Starting tasks {start}:{start+num_tasks} out of {total_num_to_copy} total")
            # tasks = [dask.delayed(ASFTransfer.copy_granule)(id) for id in job_ids[start:start+num_tasks]]
            tasks = [dask.delayed(ASFTransfer.copy_granule)(id, out_name)
                     for id, out_name in jobs.iloc[start:start+num_tasks].itertuples(index=False)]
            assert len(tasks) == num_tasks
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result, id in results[0]:
                logging.info("-->".join(each_result))
                ASFTransfer.PROCESSED_JOB_IDS.append(id)

            num_to_copy -= num_tasks
            start += num_tasks

        # Store job IDs that were processed
        with open(self.processed_jobs_file , 'w') as outfile:
            json.dump(ASFTransfer.PROCESSED_JOB_IDS, outfile)

    @staticmethod
    def object_exists(bucket, key: str) -> bool:
        try:
            bucket.Object(key).load()

        except ClientError:
            return False

        return True

    @staticmethod
    def copy_granule(job_id, out_name):
        """
        Copy granule from source to target bucket if it does not exist in target
        bucket already.
        """
        job = ASFTransfer.HYP3.get_job_by_id(job_id)
        msgs = [f"Processing {job} url={job.files[0]['url']}"]

        if job.running():
            msgs.append(f'WARNING: Job is still running! Skipping {job}')
            return msgs, job_id

        if job.succeeded():
            granule_url = job.files[0]['url']

            fixed_file = os.path.join(ASFTransfer.LOCAL_DIR, out_name)
            target = None

            source = job.files[0]['s3']['key']
            source_ext = '.nc'
            bucket = boto3.resource('s3').Bucket(ASFTransfer.TARGET_BUCKET)

            # get center lat lon for target directory, and fix attributes
            with fsspec.open(job.files[0]['url']) as f:
                with xr.open_dataset(f) as ds:
                    lat = ds.img_pair_info.latitude[0]
                    lon = ds.img_pair_info.longitude[0]
                    msgs.append(f'Image center (lat, lon): ({lat}, {lon})')

                    target_prefix = point_to_prefix(ASFTransfer.TARGET_BUCKET_DIR, lat, lon)
                    # target = f"{target_prefix}/{job.files[0]['filename']}"
                    target = f"{target_prefix}/{out_name}"

                    # Remove filename postfix which should not make it to the
                    # filename in target bucket
                    if target.endswith(f'{ASFTransfer.POSTFIX_TO_RM}'):
                        target = target.replace(ASFTransfer.POSTFIX_TO_RM, '')

                    if ASFTransfer.object_exists(bucket, target):
                        msgs.append(f'WARNING: {bucket.name}/{target} already exists, skipping upload')

                    else:
                        # Fix granule
                        ds, warning_msg = fix_all(ds, granule_url)
                        if warning_msg is not None:
                            msgs.append(warning_msg)

                        # Write the granule locally, upload it to the bucket, remove file
                        ds.to_netcdf(fixed_file, engine='h5netcdf', encoding = Encoding.LANDSAT_SENTINEL2)

            if os.path.exists(fixed_file):
                # Upload corrected granule to the bucket
                s3_client = boto3.client('s3')
                try:
                    msgs.append(f"Uploading {fixed_file} to {ASFTransfer.TARGET_BUCKET}/{target}")

                    if not ASFTransfer.DRY_RUN:
                        s3_client.upload_file(fixed_file, ASFTransfer.TARGET_BUCKET, target)

                    msgs.append(f"Removing local {fixed_file}")
                    os.unlink(fixed_file)

                except ClientError as exc:
                    msgs.append(f"ERROR: {exc}")
                    return msgs

            # There are corresponding browse and thumbprint images to transfer
            for target_ext in ['.png', '_thumb.png']:
                # It's an extra file to transfer, replace extension
                target_key = target.replace(source_ext, target_ext)
                source_key = source.replace(source_ext, target_ext)

                if ASFTransfer.object_exists(bucket, target_key):
                    msgs.append(f'WARNING: {bucket.name}/{target_key} already exists, skipping {job}')

                else:
                    source_dict = {'Bucket': job.files[0]['s3']['bucket'],
                                   'Key': source_key}

                    msgs.append(f'Copying {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')
                    if not ASFTransfer.DRY_RUN:
                        bucket.copy(source_dict, target_key)

        else:
            msgs.append(f'WARNING: {job} failed!')
            # TODO: handle failures

        return msgs, job_id

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-j', '--job-ids',
        type=Path,
        help='JSON list of HyP3 Job IDs'
    )
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=100,
        help='Number of granules to copy in parallel [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start-job',
        type=int,
        default=0,
        help='Job index to start with (to continue from where the previous run stopped) [%(default)d]'
    )
    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-t', '--target-bucket',
        default='its-live-data',
        help='Upload the autoRIFT products to this AWS bucket'
    )
    parser.add_argument(
        '-d', '--dir',
        default='velocity_image_pair/sentinel2/v02',
        help='Upload the autoRIFT products to this sub-directory of AWS bucket'
    )
    parser.add_argument(
        '-l', '--local-dir',
        type=str,
        default='sandbox_sentinel2',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-u', '--user',
        help='Username for https://urs.earthdata.nasa.gov login'
    )
    parser.add_argument(
        '-p', '--password',
        help='Password for https://urs.earthdata.nasa.gov login'
    )
    parser.add_argument(
        '-o', '--output-job-file',
        type=str,
        default='processed_jobs.json',
        help='File of processed job IDs [%(default)s]'
    )
    parser.add_argument(
        '-e', '--exclude-job-file',
        type=Path,
        default=None,
        help='JSON list of HyP3 Job IDs (previously processed) to exclude from the transfer [%(default)s]'
    )
    parser.add_argument(
        '-a', '--autoRIFT',
        default='https://hyp3-autorift-eu.asf.alaska.edu',
        help='autoRIFT deployment to connect to [%(default)s]'
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        default=False,
        help='Dry run, do not copy fixed granules to AWS S3 bucket'
    )
    parser.add_argument(
        '--enableDebug',
        action='store_true',
        default=False,
        help='Enable debug mode: process jobs sequentially'
    )

    args = parser.parse_args()

    ASFTransfer.HYP3_AUTORIFT_API = args.autoRIFT
    ASFTransfer.HYP3 = sdk.HyP3(ASFTransfer.HYP3_AUTORIFT_API, args.user, args.password)
    ASFTransfer.TARGET_BUCKET = args.target_bucket
    ASFTransfer.TARGET_BUCKET_DIR = args.dir
    ASFTransfer.LOCAL_DIR = args.local_dir
    ASFTransfer.DRY_RUN = args.dry

    if not os.path.exists(ASFTransfer.LOCAL_DIR):
        os.mkdir(ASFTransfer.LOCAL_DIR)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    transfer = ASFTransfer(args.output_job_file)
    if args.enableDebug is True:
        transfer.run_sequentially(
            args.job_ids,
            args.exclude_job_file, # Exclude previously processed job IDs if any
            args.chunk_size,
            args.start_job
        )
    else:
        transfer(
            args.job_ids,
            args.exclude_job_file, # Exclude previously processed job IDs if any
            args.chunk_size,
            args.start_job,
            args.dask_workers
        )

if __name__ == '__main__':
    main()

    logging.info("Done.")
