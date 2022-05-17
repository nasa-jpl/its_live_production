#!/usr/bin/env python
"""
Fix coordinates order within spatial metadata files required by NSIDC data ingest.

Original order, as provided by NSIDC, was not correct:

    [ul_lonlat, ll_lonlat, ur_lonlat, lr_lonlat]

should be changed to:

    [ul_lonlat, ur_lonlat, lr_lonlat, ll_lonlat]

Authors: Masha Liukis
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import dask
from datetime import datetime
from dask.diagnostics import ProgressBar
import json
import logging
import os
import s3fs


class FixSpatialFiles:
    """
    Functor class to fix coordinates order within NSIDC required spatial files.

    Original order, as provided by NSIDC, was not correct:

        [ul_lonlat, ll_lonlat, ur_lonlat, lr_lonlat]

    should be changed to:

        [ul_lonlat, ur_lonlat, lr_lonlat, ll_lonlat]
    """
    # Number of files to fix in parallel
    CHUNK_SIZE = 100

    NUM_DASK_WORKERS = 8

    # Just report what would run - don't actually invoke processing
    DRY_RUN = False

    GLOB = '*.spatial'

    def __init__(self,
        bucket: str,
        bucket_dir: str
    ):
        """
        Initialize functor-like object to fix spatial files.
        """
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_files = self.s3.glob(f'{os.path.join(bucket, bucket_dir)}/{FixSpatialFiles.GLOB}')

        # To guarantee the same order of files if need to pick up the processing
        # in a middle
        self.all_files = sorted(self.all_files)[:2]
        logging.info(f"Number of files: {len(self.all_files)}")

        self.bucket = bucket

    def __call__(self, start_index: int):
        """
        Fix order of listed coordinates.
        """
        num_to_fix = len(self.all_files) - start_index

        start = start_index
        logging.info(f"{num_to_fix} files to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        while num_to_fix > 0:
            num_tasks = FixSpatialFiles.CHUNK_SIZE if num_to_fix > FixSpatialFiles.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixSpatialFiles.fix)(self.bucket, each, self.s3) for each in self.all_files[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=FixSpatialFiles.NUM_DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    def fix(bucket: str, filename: str, s3):
        """
        Fix order of the coordinates.
        """
        all_lines = []
        s3_client = boto3.client('s3')

        filename_tokens = filename.split(os.path.sep)
        file_path = os.path.sep.join(filename_tokens[1:])

        data = s3_client.get_object(Bucket=bucket, Key=file_path)
        for line in data['Body'].iter_lines():
            all_lines.append(line)

        msgs = [f'Processing {filename} from:']
        msgs.append(json.dumps(all_lines, indent=4))

        meta_filename = filename_tokens[-1]

        # Write to local spatial file
        with open(meta_filename, 'w') as fh:
            fh.write(all_lines[0])
            fh.write(all_lines[2])
            fh.write(all_lines[3])
            fh.write(all_lines[1])

        return msgs

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 that stores NSIDC spatial files'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='NSIDC/v01/velocity_image_pair',
        help='AWS S3 directory that stores NSIDC spatial files'
    )
    parser.add_argument(
        '-s', '--start_index',
        type=int,
        default=0,
        help='Index for the start file to process (if previous processing terminated) [%(default)d]'
    )
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    fix_files = FixSpatialFiles(args.bucket, args.bucket_dir)
    fix_files(args.start_granule)

if __name__ == '__main__':
    main()

    logging.info("Done.")
