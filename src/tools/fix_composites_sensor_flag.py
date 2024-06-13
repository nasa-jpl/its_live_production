#!/usr/bin/env python
"""
Update "sensor_flag" data variable in datacube's composites. Originally
generated composites had values of 0 - excluded; 1 - included, missing_value = 255.
This script changes the values to: 0 - included; 1 - excluded, missing_value = 0.
All composites reside in AWS S3 bucket.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket.

Authors: Masha Liukis, Alex Gardner
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import logging
import os
import s3fs
import shutil
import subprocess
import xarray as xr

from itscube_types import DataVars, BinaryFlag, Output
from itslive_composite import CompDataVars


class FixComposites:
    """
    Class to apply fixes to ITS_LIVE datacubes composites:

    * Reverse "sensor_flag" values to: 0 - included; 1 - excluded
    * Use missing_value = 0
    """
    COMPOSITES_TO_GENERATE = []
    S3_PREFIX = 's3://'
    DRY_RUN = False

    def __init__(self, bucket: str, bucket_dir: str, target_bucket_dir: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.bucket_dir = bucket_dir
        self.target_bucket_dir = target_bucket_dir

        # For debugging only
        # FixComposites.COMPOSITES_TO_GENERATE = FixComposites.COMPOSITES_TO_GENERATE[:2]
        # logging.info(f"DEBUG ONLY: test ULRs: {FixComposites.COMPOSITES_TO_GENERATE}")

        # Collect names for existing composites
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_composites = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            files = self.s3.ls(each)
            files = [each_cube for each_cube in files if each_cube.endswith('.zarr')]
            self.all_composites.extend(files)

        # Sort the list to guarantee the order of found Zarr's
        self.all_composites.sort()
        logging.info(f"Found number of composites: {len(self.all_composites)}")

        self.num_to_fix = len(self.all_composites)
        logging.info(f"{self.num_to_fix} composites to fix...")

    def debug__call__(self, local_dir: str, num_dask_workers: int):
        """
        Fix "sensor_flag" of ITS_LIVE composites stored in S3 bucket.
        """
        if self.num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for each in self.all_composites:
            logging.info(f"Starting {each}")
            msgs = FixComposites.all(each, self.bucket, self.bucket_dir, self.target_bucket_dir, local_dir, self.s3)
            logging.info("\n-->".join(msgs))

    def __call__(self, local_dir: str, num_dask_workers: int, start_index: int = 0):
        """
        Fix sensor_flag of ITS_LIVE composites stored in S3 bucket.
        """
        if self.num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        start = start_index

        num_to_fix = self.num_to_fix - start
        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixComposites.all)(
                each,
                self.bucket,
                self.bucket_dir,
                self.target_bucket_dir,
                local_dir,
                self.s3
            ) for each in self.all_composites[start:start+num_tasks]]

            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=num_dask_workers
                )

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def all(composite_url: str, bucket_name: str, bucket_dir: str, target_bucket_dir: str, local_dir: str, s3_in):
        """
        Fix composites and copy them back to S3 bucket.
        """
        msgs = [f'Processing {composite_url}']

        zarr_store = s3fs.S3Map(root=composite_url, s3=s3_in, check=False)
        composite_basename = os.path.basename(composite_url)

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, composite_basename)

        with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            # Fix sensor_flag data:
            curr_attrs = ds[CompDataVars.SENSOR_INCLUDE].attrs
            curr_attrs[DataVars.DESCRIPTION_ATTR] = CompDataVars.DESCRIPTION[CompDataVars.SENSOR_INCLUDE]
            curr_attrs[BinaryFlag.MEANINGS_ATTR] = BinaryFlag.MEANINGS[CompDataVars.SENSOR_INCLUDE]

            # Reverse 0s and 1s in data:
            sensor_flag = xr.where(ds[CompDataVars.SENSOR_INCLUDE] == 1, 0, 1)
            sensor_flag.attrs = curr_attrs
            # xr.where does not preserve encoding of attributes of original data variable
            sensor_flag.encoding = ds[CompDataVars.SENSOR_INCLUDE].encoding

            ds[CompDataVars.SENSOR_INCLUDE] = sensor_flag
            ds[CompDataVars.SENSOR_INCLUDE].attrs = curr_attrs

            # Fix encoding for the sensor_flag
            ds[CompDataVars.SENSOR_INCLUDE].encoding[Output.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

            # Change dtype to np.uint32 for count and count0 data variables
            msgs.append(f"Saving composite to {fixed_file}")

            # logging.info(f"Encoding settings: {encoding_settings}")
            ds.to_zarr(fixed_file, consolidated=True)

        target_url = composite_url.replace(bucket_dir, target_bucket_dir)
        if not target_url.startswith(FixComposites.S3_PREFIX):
            target_url = FixComposites.S3_PREFIX + target_url

        if FixComposites.DRY_RUN:
            msgs.append(f'DRYRUN: copy composite to {target_url}')
            return msgs

        if os.path.exists(fixed_file) and len(target_bucket_dir):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy

            # Enable conversion to NetCDF when the cube is created
            # Convert Zarr to NetCDF and copy to the bucket
            # nc_filename = args.outputStore.replace('.zarr', '.nc')
            # zarr_to_netcdf.main(args.outputStore, nc_filename, ITSCube.NC_ENGINE)
            # ITSCube.show_memory_usage('after Zarr to NetCDF conversion')
            env_copy = os.environ.copy()

            command_line = [
                "aws", "s3", "cp", "--recursive",
                fixed_file,
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
                msgs.append(f"ERROR: Failed to copy {fixed_file} to {target_url}: {command_return.stdout}")

            msgs.append(f"Removing local {fixed_file}")
            shutil.rmtree(fixed_file)

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE datacubes composites to fix "sensor_flag" for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='composites/annual/v2',
        help='AWS S3 bucket and directory that store datacubes composites [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='composites/annual/v2_fixed_sensor_flag',
        help='AWS S3 directory to store fixed datacubes composites [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir', type=str,
        default='sandbox',
        help='Directory to store fixed data before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually copy any data to AWS S3 bucket'
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
    FixComposites.DRY_RUN = args.dryrun

    fix_composites = FixComposites(args.bucket, args.bucket_dir,  args.target_bucket_dir)
    fix_composites(args.local_dir, args.dask_workers, args.start_index)


if __name__ == '__main__':
    main()
    logging.info("Done.")
