#!/usr/bin/env python
"""
Script to copy M11/M12 restore data to its-live-project s3 bucket.
"""
import argparse
import boto3
import dask
from dask.diagnostics import ProgressBar
import logging
import pandas as pd

target_bucket = 'its-live-project'

CHUNK_SIZE = 100

def copy_file(source_bucket, source_dir, target_bucket, target_dir):
    # Dask can't pickle boto3.client('s3') object, create one locally
    s3 = boto3.client('s3')

    source_key = f"{source_dir}/conversion_matrices.nc"

    # Construct the target key
    target_key = f"{target_dir}/{source_key}"

    # Copy the file from source to target
    copy_source = {'Bucket': source_bucket, 'Key': source_key}

    msg = ''
    try:
        s3.copy(copy_source, target_bucket, target_key)
        msg = f"Copied {source_key} to s3://{target_bucket}/{target_key}"

    except Exception as e:
        msg = f"Failed to copy {source_key} to s3://{target_bucket}/{target_key}: {e}"

    return msg


def main(args):
    # Initialize S3 client
    s3 = boto3.client('s3')

    # Load the Parquet file
    parquet_file = args.parquet_file
    table = pd.read_parquet(parquet_file)

    num_to_copy = len(table)
    logging.info(f'{num_to_copy} directories to copy...')

    # For debugging
    # num_to_copy = 5

    if num_to_copy <= 0:
        logging.info("Nothing to copy, exiting.")
        return

    # Define the target bucket and directory
    target_dir = args.target_dir

    start = 0
    while num_to_copy > 0:
        num_tasks = CHUNK_SIZE if num_to_copy > CHUNK_SIZE else num_to_copy

        logging.info(f"Starting tasks {start}:{start+num_tasks}")

        tasks = [
            dask.delayed(copy_file)(
                each['cor_s3_bucket'],  # source s3 bucket
                each['job_id'],         # directory with correction matrices file
                target_bucket,          # target s3 bucket to copy file to
                target_dir              # target s3 directory to copy file to
            ) for _, each in table.iloc[start:start+num_tasks].iterrows()
        ]
        results = None

        with ProgressBar():
            # Display progress bar
            results = dask.compute(
                tasks,
                scheduler="processes",
                num_workers=8
            )

        for each_result in results[0]:
            logging.info(each_result)

        num_to_copy -= num_tasks
        start += num_tasks


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--parquet_file',
        help=f'Parquet file that stores s3 location of granules M11/M12 data.'
    )
    parser.add_argument(
        '--target_dir',
        default='fromASF/restore_conversion_matrices/production',
        help=f'Parquet file that stores s3 location of granules M11/M12 data.'
    )
    args = parser.parse_args()

    logging.info(f"Args: {args}")

    main(args)

    logging.info("Done.")

