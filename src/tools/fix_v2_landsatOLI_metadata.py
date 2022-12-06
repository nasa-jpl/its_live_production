#!/usr/bin/env python
"""
Fix metadata for newly created GRE Landsat granules:

Was:
interp_mask:
    flag_meanings = "measured, interpolated";

Fixed:
interp_mask:
    flag_meanings = "measured interpolated";

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import copy
import dask
from dask.diagnostics import ProgressBar
import logging
import os
import s3fs
import xarray as xr

from itscube_types import DataVars, BinaryFlag, Coords, Output


class FixGranules:
    # Flag if should do a dry run: don't push any fixed files to the S3 bucket,
    # just write fixed files locally
    DRYRUN = False

    """
    Class to fix metadata for v2 granules, which are part of the first batch of
    production Landsat granules for Greenland that had the wrong metadata in them.
    """
    def __init__(self, bucket: str, bucket_dir: str, glob_pattern: dir):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem()

        # use a glob to list directory
        logging.info(f"Reading {bucket_dir}")
        self.all_granules = self.s3.glob(f'{os.path.join(bucket, bucket_dir)}/{glob_pattern}')
        logging.info(f"Number of granules: {len(self.all_granules)}")

        # Guarantee the order of the granules if need to pick up from where previous
        # processing left off
        self.all_granules = sorted(self.all_granules)

        # For debugging only:
        # self.all_granules = self.all_granules[:2]
        self.bucket = bucket

    def __call__(self, local_dir: str, chunk_size: int, num_dask_workers: int, start_index: int):
        """
        Fix acquisition date and time attributes of ITS_LIVE granules stored
        in the bucket.
        """
        num_to_fix = len(self.all_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixGranules.metadata)(self.bucket, each, local_dir, self.s3) for each in self.all_granules[start:start+num_tasks]]
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
    def metadata(bucket_name: str, granule_url: str, local_dir: str, s3):
        """
        Fix metadata, store it to the local NetCDF file, and copy to the
        destination S3 bucket.
        """
        _new_value = "measured interpolated"

        msgs = [f'Processing {granule_url}']

        # get center lat lon
        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                # Fix metadata
                ds[DataVars.INTERP_MASK].attrs[BinaryFlag.MEANINGS_ATTR] = _new_value

                # Re-use existing in the granule encoding, except for adding
                # _FillValue = None for x, y data variables: otherwise they
                # are set though they should not be.
                ds[Coords.X].encoding[Output.FILL_VALUE_ATTR] = None
                ds[Coords.Y].encoding[Output.FILL_VALUE_ATTR] = None

                granule_basename = os.path.basename(granule_url)

                # Write the granule locally, upload it to the bucket, remove file
                fixed_file = os.path.join(local_dir, granule_basename)
                ds.to_netcdf(fixed_file, engine='h5netcdf')

                # Upload corrected granule to the bucket
                s3_client = boto3.client('s3')
                try:
                    bucket_granule = granule_url.replace(bucket_name+'/', '')
                    msgs.append(f"Uploading {fixed_file} as {bucket_granule} to {bucket_name}")

                    if not FixGranules.DRYRUN:
                        s3_client.upload_file(fixed_file, bucket_name, bucket_granule)

                        msgs.append(f"Removing local {fixed_file}")
                        os.unlink(fixed_file)

                except ClientError as exc:
                    msgs.append(f"ERROR: {exc}")

                return msgs


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
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE granules to fix attributes for'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/landsatOLI-latest',
        help='AWS S3 bucket and directory that store the granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox',
        help='Directory to store fixed granules before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '-glob',
        action='store',
        type=str,
        default='*/*.nc',
        help='Glob pattern for the granule search under "--bucket_dir" [%(default)s]')

    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument('-s', '--start-granule', type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    FixGranules.DRYRUN = args.dryrun

    fix_v2_metadata = FixGranules(args.bucket, args.bucket_dir, args.glob)
    fix_v2_metadata(
        args.local_dir,
        args.chunk_size,
        args.dask_workers,
        args.start_granule
    )


if __name__ == '__main__':
    main()

    logging.info("Done.")
