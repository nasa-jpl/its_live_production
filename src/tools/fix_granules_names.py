#!/usr/bin/env python
"""
Compress ITS_LIVE granules filenames in its-live-data S3 bucket.
Need to remove suffix

Authors: Masha Liukis
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import logging
import os
import subprocess
import s3fs


class FixGranulesNames:
    """
    Class to fix ITS_LIVE granules filenames (that were transferred
    from ASF to ITS_LIVE bucket).
    """
    SUFFIX_TO_REMOVE = '_IL_ASF_OD'

    def __init__(self, bucket: str, glob_pattern: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket}")
        self.all_granules = self.s3.glob(f'{bucket}/{glob_pattern}')
        png_pattern = glob_pattern.replace('.nc', '.png')
        self.all_granules.extend(self.s3.glob(f'{bucket}/{png_pattern}'))

        logging.info(f"Number of files: {len(self.all_granules)}")

        # Each granule has corresponding png files: '.png', '_thumb.png',
        # rename those too
        self.all_granules = [each for each in self.all_granules if FixGranulesNames.SUFFIX_TO_REMOVE in each]
        logging.info(f"Number of {FixGranulesNames.SUFFIX_TO_REMOVE} granules: {len(self.all_granules)}")

    def __call__(self, chunk_size: int, num_dask_workers: int, start_index: int):
        """
        Rename granules in S3 bucket.
        """
        num_to_fix = len(self.all_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixGranulesNames.move)(each) for each in self.all_granules[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def move(granule_url: str):
        """
        Rename granule.
        """
        msgs = [f'Processing {granule_url}']

        env_copy = os.environ.copy()

        if 's3://' not in granule_url:
            granule_url = 's3://' + granule_url
        target_url = granule_url.replace(FixGranulesNames.SUFFIX_TO_REMOVE, '')
        command_line = [
            "aws", "s3", "mv",
            granule_url,
            target_url,
            "--acl", "bucket-owner-full-control"
        ]

        msgs.append(' '.join(command_line))

        command_return = subprocess.run(
            command_line,
            env=env_copy,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if command_return.returncode != 0:
            msgs.append(f"ERROR: Failed to copy {granule_url} to {target_url}: {command_return.stdout}")

        return msgs

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size', type=int,
        default=100, help='Number of granules to fix in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='s3://its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v02',
        help='AWS S3 that stores ITS_LIVE granules to fix attributes for'
    )
    parser.add_argument(
        '-glob', action='store', type=str, default='*/*.nc',
        help='Glob pattern for the granule search under "s3://bucket/dir/" [%(default)s]')

    parser.add_argument('-w', '--dask-workers', type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument('-s', '--start-granule', type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    fix_names = FixGranulesNames(args.bucket, args.glob)
    fix_names(
        args.chunk_size,
        args.dask_workers,
        args.start_granule
    )


if __name__ == '__main__':
    main()

    logging.info("Done.")
