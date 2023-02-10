#!/usr/bin/env python
"""
Locate ITS_LIVE S3 location for the granules for specific jobIDs.

To run the script you need to have credentials for:
1. https://urs.earthdata.nasa.gov (register for free if you don't have an account).
Place credentials into the file:
echo 'machine urs.earthdata.nasa.gov login USERNAME password PASSWORD' >& ~/.netrc

Authors: Masha Liukis, Joe Kennedy, Mark Fahnestock
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import json
import logging
import os
from pathlib import Path

import boto3
import fsspec
import xarray as xr
from botocore.exceptions import ClientError

import hyp3_sdk as sdk
import numpy as np


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
    if outlat == 90:  # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80

    outlon = int(10*np.trunc(np.abs(lon/10.0)))

    if outlon >= 180:  # if you are at the dateline, back off to the 170 bin
        outlon = 170

    dirstring = os.path.join(dir_path, f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}')
    return dirstring


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
    TARGET_BUCKET_URL = None

    def __init__(self):
        pass

    def __call__(
        self,
        job_ids_file: str,
        num_dask_workers: int
    ):
        """
        Locate granules in ITS_LIVE S3 bucket that correspond to ASF HyP3 jobIDs.
        """
        job_ids = json.loads(job_ids_file.read_text())

        total_num_to_copy = len(job_ids)
        num_to_copy = total_num_to_copy
        logging.info(f"{num_to_copy} out of {total_num_to_copy} granules to locate...")

        chunks_to_copy = 10
        start = 0

        while num_to_copy > 0:
            num_tasks = chunks_to_copy if num_to_copy > chunks_to_copy else num_to_copy

            logging.info(f"Starting tasks {start}:{start+num_tasks} out of {total_num_to_copy} total")
            tasks = [dask.delayed(ASFTransfer.copy_granule)(id) for id in job_ids[start:start+num_tasks]]
            assert len(tasks) == num_tasks
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result, id in results[0]:
                logging.info("-->".join(each_result))

            num_to_copy -= num_tasks
            start += num_tasks

    @staticmethod
    def object_exists(bucket, key: str) -> bool:
        try:
            bucket.Object(key).load()

        except ClientError:
            return False

        return True

    @staticmethod
    def copy_granule(job_id):
        """
        Copy granule from source to target bucket if it does not exist in target
        bucket already.
        """
        job = ASFTransfer.HYP3.get_job_by_id(job_id)
        msgs = []

        # Copy data for jobs that have status=RUNNING - they are not running
        # anymore according to Joe
        # if job.running():
        #     msgs.append(f'WARNING: Job is still running! Skipping {job}')
        #     return msgs, job_id

        if job.running() or job.succeeded():
            # get center lat lon
            with fsspec.open(job.files[0]['url']) as f:
                with xr.open_dataset(f) as ds:
                    lat = ds.img_pair_info.latitude[0]
                    lon = ds.img_pair_info.longitude[0]
                    # msgs.append(f'Image center (lat, lon): ({lat}, {lon})')

            target_prefix = point_to_prefix(ASFTransfer.TARGET_BUCKET_DIR, lat, lon)
            bucket = boto3.resource('s3').Bucket(ASFTransfer.TARGET_BUCKET)
            target = f"{target_prefix}/{job.files[0]['filename']}"

            # Remove filename postfix which should not make it to the
            if target.endswith('_IL_ASF_OD.nc'):
                target = target.replace(ASFTransfer.POSTFIX_TO_RM, '')

            source = job.files[0]['s3']['key']

            # There are corresponding browse and thumbprint images to transfer
            for target_ext in [None]:
                target_key = target

                if ASFTransfer.object_exists(bucket, target_key):
                    granule_url = f'{bucket.name}/{target_key}'.replace(ASFTransfer.TARGET_BUCKET, ASFTransfer.TARGET_BUCKET_URL)
                    msgs.append(f'jobID {job}: {granule_url}')

                else:
                    source_key = source
                    source_dict = {'Bucket': job.files[0]['s3']['bucket'],
                                   'Key': source_key}

                    msgs.append(f'File does not exist, need to copy {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')

        else:
            msgs.append(f'WARNING: {job} failed!')
            # TODO: handle failures

        return msgs, job_id


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--job-ids', type=Path, help='JSON list of HyP3 Job IDs')
    parser.add_argument('-w', '--dask-workers', type=int, default=4, help='Number of Dask parallel workers [%(default)d]')
    parser.add_argument('-t', '--target-bucket', help='Upload the autoRIFT products to this AWS bucket', default='its-live-data')
    parser.add_argument('-b', '--target-bucket-url', help='Upload the autoRIFT products to this AWS bucket', default='https://its-live-data.s3.amazonaws.com')
    parser.add_argument('-d', '--dir', help='Upload the autoRIFT products to this sub-directory of AWS bucket')
    parser.add_argument('-u', '--user', help='Username for https://urs.earthdata.nasa.gov login')
    parser.add_argument('-p', '--password', help='Password for https://urs.earthdata.nasa.gov login')
    parser.add_argument('-a', '--autoRIFT', default='https://hyp3-its-live.asf.alaska.edu', help='autoRIFT deployment to connect to [%(default)s]')

    args = parser.parse_args()

    ASFTransfer.HYP3_AUTORIFT_API = args.autoRIFT
    ASFTransfer.HYP3 = sdk.HyP3(ASFTransfer.HYP3_AUTORIFT_API, args.user, args.password)
    ASFTransfer.TARGET_BUCKET = args.target_bucket
    ASFTransfer.TARGET_BUCKET_DIR = args.dir
    ASFTransfer.TARGET_BUCKET_URL = args.target_bucket_url

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    transfer = ASFTransfer()
    transfer(
        args.job_ids,
        args.dask_workers
    )


if __name__ == '__main__':
    main()

    logging.info("Done.")
