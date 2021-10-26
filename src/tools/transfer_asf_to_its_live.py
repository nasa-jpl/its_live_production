#!/usr/bin/env python
"""
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
import pandas as pd


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

    def __init__(self, processed_jobs_file: str):
        self.processed_jobs_file = processed_jobs_file

    def run_sequentially(
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
        jobs = pd.read_csv(job_ids_file)

        if exclude_job_ids_file is not None:
            exclude_ids = json.loads(exclude_job_ids_file.read_text())

            # Remove exclude_ids from the jobs to process
            jobs = jobs.loc[~jobs.job_id.isin(exclude_ids)].copy()

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
        jobs = pd.read_csv(job_ids_file)

        if exclude_job_ids_file is not None:
            exclude_ids = json.loads(exclude_job_ids_file.read_text())

            # Remove exclude_ids from the jobs to process
            jobs = jobs.loc[~jobs.job_id.isin(exclude_ids)].copy()

        total_num_to_copy = len(jobs)
        num_to_copy = total_num_to_copy - start_job
        start = start_job
        logging.info(f"{num_to_copy} out of {total_num_to_copy} granules to copy...")

        while num_to_copy > 0:
            num_tasks = chunks_to_copy if num_to_copy > chunks_to_copy else num_to_copy

            logging.info(f"Starting tasks {start}:{start+num_tasks} out of {total_num_to_copy} total")
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
        msgs = [f'Processing {job}']

        if job.running():
            msgs.append(f'WARNING: Job is still running! Skipping {job}')
            return msgs, job_id

        if job.succeeded():
            # get center lat lon
            with fsspec.open(job.files[0]['url']) as f:
                with xr.open_dataset(f) as ds:
                    lat = ds.img_pair_info.latitude[0]
                    lon = ds.img_pair_info.longitude[0]
                    msgs.append(f'Image center (lat, lon): ({lat}, {lon})')

            target_prefix = point_to_prefix(ASFTransfer.TARGET_BUCKET_DIR, lat, lon)
            bucket = boto3.resource('s3').Bucket(ASFTransfer.TARGET_BUCKET)
            target = f"{target_prefix}/{out_name}"

            # Remove filename postfix which should not make it to the
            if target.endswith(f'{ASFTransfer.POSTFIX_TO_RM}.nc'):
                target = target.replace(ASFTransfer.POSTFIX_TO_RM, '')

            source = job.files[0]['s3']['key']

            # There are corresponding browse and thumbprint images to transfer
            for target_ext in [None, '.png', '_thumb.png']:
                target_key = target
                source_key = source

                # It's an extra file to transfer, replace extension
                if target_ext is not None:
                    target_key = target_key.replace('.nc', target_ext)
                    source_key = source_key.replace('.nc', target_ext)

                if ASFTransfer.object_exists(bucket, target_key):
                    msgs.append(f'WARNING: {bucket.name}/{target_key} already exists, skipping {job}')

                else:
                    source_dict = {'Bucket': job.files[0]['s3']['bucket'],
                                   'Key': source_key}

                    bucket.copy(source_dict, target_key)
                    msgs.append(f'Copying {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')

        else:
            msgs.append(f'WARNING: {job} failed!')
            # TODO: handle failures

        return msgs, job_id

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--job-ids', type=Path, help='JSON list of HyP3 Job IDs')
    parser.add_argument('-n', '--number-to-copy', type=int, default=100, help='Number of granules to copy in parallel [%(default)d]')
    parser.add_argument('-s', '--start-job', type=int, default=0, help='Job index to start with (to continue from where the previous run stopped) [%(default)d]')
    parser.add_argument('-w', '--dask-workers', type=int, default=4, help='Number of Dask parallel workers [%(default)d]')
    parser.add_argument('-t', '--target-bucket', help='Upload the autoRIFT products to this AWS bucket')
    parser.add_argument('-d', '--dir', help='Upload the autoRIFT products to this sub-directory of AWS bucket')
    parser.add_argument('-u', '--user', help='Username for https://urs.earthdata.nasa.gov login')
    parser.add_argument('-p', '--password', help='Password for https://urs.earthdata.nasa.gov login')
    parser.add_argument('-o', '--output-job-file', type=str, default='processed_jobs.json', help='File of processed job IDs [%(default)s]')
    parser.add_argument('-e', '--exclude-job-file', type=Path, default=None, help='JSON list of HyP3 Job IDs (previously processed) to exclude from the transfer [%(default)s]')
    parser.add_argument('-a', '--autoRIFT', default='https://hyp3-autorift.asf.alaska.edu', help='autoRIFT deployment to connect to [%(default)s]')
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

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    transfer = ASFTransfer(args.output_job_file)
    if args.enableDebug is True:
        transfer.run_sequentially(
            args.job_ids,
            args.exclude_job_file, # Exclude previously processed job IDs if any
            args.number_to_copy,
            args.start_job,
            args.dask_workers
        )
    else:
        transfer(
            args.job_ids,
            args.exclude_job_file, # Exclude previously processed job IDs if any
            args.number_to_copy,
            args.start_job,
            args.dask_workers
        )

if __name__ == '__main__':
    main()

    logging.info("Done.")
