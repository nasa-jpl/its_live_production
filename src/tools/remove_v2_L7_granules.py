"""
Helper script to remove original granules from the "its-live-data" S3 bucket.
S3 paths are provided through input JSON file.
"""
import boto3
import dask
from dask.diagnostics import ProgressBar
import os
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

S3_PREFIX = 'its-live-data'


def remove_s3_granule(s3_path: str, is_dryrun: bool):
    """
    Remove granule and corresponding *png files from S3 bucket if they exist.
    This is done in preparation to replace buggy granules with newly generated ones.
    """
    # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
    # resulting in as many error messages as there are files in Zarr store
    # to copy
    s3 = boto3.resource('s3')

    # There are corresponding browse and thumbprint images to transfer
    for target_ext in [None, '.png', '_thumb.png']:
        file_path = s3_path

        # It's an extra file to transfer, replace extension
        if target_ext is not None:
            file_path = s3_path.replace('.nc', target_ext)

        obj = s3.Object(S3_PREFIX, file_path)
        # logging.info(obj)

        if not is_dryrun:
            _ = obj.delete()


if __name__ == '__main__':
    import argparse

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-f', '--granulesFile',
        type=str,
        help="Input JSON file that stores a list of granules to remove from S3 bucket."
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not actually remove any granules'
    )

    args = parser.parse_args()

    CHUNK_SIZE = 100
    DASK_WORKERS = 8

    with open(args.granulesFile) as fh:
        all_granules = json.load(fh)

        num_to_fix = len(all_granules)
        logging.info(f"{num_to_fix} granules to remove...")

        start = 0

        while num_to_fix > 0:
            num_tasks = CHUNK_SIZE if num_to_fix > CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [
                dask.delayed(remove_s3_granule)(each, args.dryrun) for each in all_granules[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks, scheduler="processes", num_workers=DASK_WORKERS)

            num_to_fix -= num_tasks
            start += num_tasks

    logging.info("Done.")
