#!/usr/bin/env python
"""
Script to identify datacube composites that need to be re-generated due to the
np.short type used for originally computed "count" and "count0" data variables.
Using np.short datatype to store "count" and "count0" data resulted in data
overflow for some of the composites, and np.uint32 should be used instead.

This script creates three lists of composites to process:
1. Composites that only need to change datatype to np.uint32 for existing "count"
   and "count0" variables as there was no overflow.
2. Composites that should be re-created using SPOT EC2 instances as they are not
   large in size and won't take as long to run.
3. Composites that should be re-created using On-Demand EC2 instance as they are
   too large in size and most likely EC2 instance will get terminated by AWS, so
   don't waste time and just submit them to On-Demand queue.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis (JPL)
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import json
import logging
import numpy as np
import os
import s3fs
import shutil
import subprocess
import xarray as xr
import zarr

from itscube_types import DataVars, Coords
from itslive_composite import CompDataVars, CompOutputFormat


class AnnualCompositesCountOverflow:
    """
    Class to identify ITS_LIVE datacubes composites that have "count" and/or
    "count0" data overflow.

    This script creates three lists of composites to process:
    1. Composites that only need to change datatype to np.uint32 for existing "count"
       and "count0" variables as there was no overflow.
    2. Composites that should be re-created using SPOT EC2 instances as they are not
       large in size and won't take as long to run.
    3. Composites that should be re-created using On-Demand EC2 instance as they are
       too large in size and most likely EC2 instance will get terminated by AWS, so
       don't waste time and just submit them to On-Demand queue.
    """
    S3_PREFIX = 's3://'
    FILENAME_PREFIX = 'composites_for_'
    CHANGE_DTYPE_PREFIX = 'change_dtype.json'
    SPOT_QUEUE_PREFIX = 'spot_queue.json'
    ONDEMAND_QUEUE_PREFIX = 'ondemand_queue.json'

    # Only for composites that need to be re-created: maximum "time" dimension
    # to qualify re-processing for SPOT AWS queue,
    # any number greater than that should be re-processed with On-Demand queue
    MAX_SPOT_TIME_DIM = 140000

    def __init__(self, bucket: str, bucket_dir: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.bucket_dir = bucket_dir

        # Collect names for existing datacubes
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_composites = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            composites = self.s3.ls(each)
            composites = [each for each in composites if each.endswith('.zarr')]
            self.all_composites.extend(composites)

        # Sort the list to guarantee the order of found stores
        self.all_composites.sort()
        logging.info(f"Found number of composites: {len(self.all_composites)}")

        # For debugging only
        # self.all_composites = self.all_composites[:1]
        # logging.info(f"ULRs: {self.all_composites}")

    def no__call__(self, num_dask_workers: int, start_index: int=0):
        """
        Apply fixes.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} composites to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        change_type_composites = []
        spot_queue_composites = []
        ondemand_queue_composites = []

        for each in self.all_composites:
            logging.info(f"Starting {each}")
            msgs, change_type, spot_queue, ondemand_queue, composite_file = \
                AnnualCompositesCountOverflow.all(each, self.s3, AnnualCompositesCountOverflow.MAX_SPOT_TIME_DIM)
            logging.info("\n-->".join(msgs))

            if change_type:
                change_type_composites.append(composite_file)

            elif spot_queue:
                spot_queue_composites.append(composite_file)

            elif ondemand_queue:
                ondemand_queue_composites.append(composite_file)

        # Save lists to json files
        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.CHANGE_DTYPE_PREFIX
        logging.info(f"Writing change_type_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(change_type_composites, output_fhandle, indent=4)

        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.SPOT_QUEUE_PREFIX
        logging.info(f"Writing spot_queue_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(spot_queue_composites, output_fhandle, indent=4)

        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.ONDEMAND_QUEUE_PREFIX
        logging.info(f"Writing ondemand_queue_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(ondemand_queue_composites, output_fhandle, indent=4)

    def __call__(self, num_dask_workers: int, start_index: int=0):
        """
        Identify which re-processing category the datacube should be associated with.

        Inputs:
        num_dask_workers - Number of parallel workers to use for processing.
        start_index - Start index into composites to process (useful if continue after
                      a failure in processing). Default is 0.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} composites to check...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        change_type_composites = []
        spot_queue_composites = []
        ondemand_queue_composites = []

        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(AnnualCompositesCountOverflow.validate)( \
                each, \
                self.s3, \
                AnnualCompositesCountOverflow.MAX_SPOT_TIME_DIM \
            ) for each in self.all_composites[start:start+num_tasks]]

            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                msgs, change_type, spot_queue, ondemand_queue, composite_file = each_result
                logging.info("\n-->".join(msgs))

                if change_type:
                    change_type_composites.append(composite_file)

                elif spot_queue:
                    spot_queue_composites.append(composite_file)

                elif ondemand_queue:
                    ondemand_queue_composites.append(composite_file)

            num_to_fix -= num_tasks
            start += num_tasks

        # Save lists to json files
        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.CHANGE_DTYPE_PREFIX
        logging.info(f"Writing change_type_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(change_type_composites, output_fhandle, indent=4)

        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.SPOT_QUEUE_PREFIX
        logging.info(f"Writing spot_queue_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(spot_queue_composites, output_fhandle, indent=4)

        output_file = AnnualCompositesCountOverflow.FILENAME_PREFIX + AnnualCompositesCountOverflow.ONDEMAND_QUEUE_PREFIX
        logging.info(f"Writing ondemand_queue_composites to the {output_file}...")
        with open(output_file, 'w') as output_fhandle:
            json.dump(ondemand_queue_composites, output_fhandle, indent=4)

    @staticmethod
    def validate(composite_url: str, s3_in, max_dim_threshold):
        """
        Validate composite and determine what kind of processing it should go through.

        Returns:
        msgs: Messages created during Processing
        change_type: Flag if dtype of "count" and "count0" data variables should be just changed to np.uint32
        spot_queue: Flag if composite should be re-created with SPOT queue
        ondemand_queue: Flag if omposite should be re-created with OnDemand queue
        composite_file: Name of the composite store in AWS
        """
        msgs = [f'Processing {composite_url}']

        zarr_store = s3fs.S3Map(root=composite_url, s3=s3_in, check=False)

        change_type, spot_queue, ondemand_queue = False, False, False

        with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            # Check if there are any negative "count0" values
            if np.any(ds[CompDataVars.COUNT0].values < 0):
                # Get S3 URL for corresponding datacube
                datacube_url = ds.attrs[CompOutputFormat.DATECUBE_URL].replace('https:', 's3:')
                datacube_url = datacube_url.replace('.s3.amazonaws.com', '')
                zarr_store = s3fs.S3Map(root=datacube_url, s3=s3_in, check=False)

                with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as datacube_ds:
                    # Find corresponding datacube and check on its size which will
                    # determine AWSs queue for the composite to re-create
                    sizes = datacube_ds.sizes
                    queue_msg = f'Need to re-process due to the datacube size threshold ({sizes}): adding to '

                    if sizes[Coords.MID_DATE] > max_dim_threshold:
                        queue_msg += 'OnDemand queue'
                        ondemand_queue = True

                    else:
                        queue_msg += 'SPOT queue'
                        spot_queue = True

                msgs.append(queue_msg)

            else:
                msgs.append(f'Need to change dtype as no negative values are detected for {CompDataVars.COUNT0}')
                change_type = True

        return (msgs, change_type, spot_queue, ondemand_queue, composite_url)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE annual composites to fix v_error for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='composites/annual/v02',
        help='AWS S3 bucket and directory that store annual composites [%(default)s]'
    )
    parser.add_argument('-w', '--dask-workers', type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start-index',
        type=int,
        default=0,
        help='Index for the start composite to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    check_composite_count0 = AnnualCompositesCountOverflow(args.bucket, args.bucket_dir)
    check_composite_count0(args.dask_workers, args.start_index)

if __name__ == '__main__':
    main()
    logging.info("Done.")
