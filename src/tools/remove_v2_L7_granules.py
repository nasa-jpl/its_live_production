"""
Helper script to remove original granules from the "its-live-data" S3 bucket.
S3 paths are provided through input JSON file.
"""
import dask
from dask.diagnostics import ProgressBar
import subprocess
import os
import logging
import s3fs
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

S3_PREFIX = 's3://its-live-data'

ENV_COPY = os.environ.copy()


def exists(s3_path: str):
    """
    Check if granule exists in AWS S3 bucket.
    """
    granule_exists = False

    # Check if the datacube is in the S3 bucket
    s3 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    file_glob = s3.glob(s3_path)
    if len(file_glob):
        granule_exists = True

    return granule_exists


def remove_s3_granule(s3_path: str, is_dryrun: bool):
    """
    Remove granule and corresponding *png files from S3 bucket if they exist.
    This is done in preparation to replace buggy granules with newly generated ones.
    """
    # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
    # resulting in as many error messages as there are files in Zarr store
    # to copy
    granule_path = os.path.join(S3_PREFIX, s3_path)

    msgs = []

    # There are corresponding browse and thumbprint images to transfer
    for target_ext in [None, '.png', '_thumb.png']:
        file_path = granule_path

        # It's an extra file to transfer, replace extension
        if target_ext is not None:
            file_path = granule_path.replace('.nc', target_ext)

        if exists(file_path):
            command_line = [
                "awsv2", "s3", "rm", "--quiet",
                file_path
            ]
            # logging.info(f'{is_dryrun_str}Removing existing {file_path}: {" ".join(command_line)}')

            if not is_dryrun:
                command_return = subprocess.run(
                    command_line,
                    env=ENV_COPY,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                if command_return.returncode != 0:
                    msgs.append(f"ERROR: failed to remove {file_path}: {command_return.stdout}")
        else:
            msgs.append(f'WARNING: {file_path} does not exist, skip removal')

    return msgs


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

            msgs = []
            for each_result in results[0]:
                msgs.extend(each_result)

            if len(msgs):
                logging.info("\n-->".join(msgs))

            num_to_fix -= num_tasks
            start += num_tasks

    logging.info("Done.")
