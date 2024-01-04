#!/usr/bin/env python
"""
Update "sensor_flag" data variable in datacube's static mosaics. Originally
generated composites/mosaics had values of 0 - excluded; 1 - included, missing_value = 255.
This script changes the values to: 0 - included; 1 - excluded, missing_value = 0.
All mosaics reside in AWS S3 bucket.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket.

Authors: Masha Liukis, Alex Gardner
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import json
import logging
import os
import s3fs
import shutil
import subprocess
import xarray as xr

from itscube_types import DataVars, BinaryFlag, Output
from itslive_composite import CompDataVars
from itslive_annual_mosaics import ITSLiveAnnualMosaics
from itscube import ITSCube


class FixMosaics:
    """
    Class to apply fixes to ITS_LIVE datacubes mosaics:

    * Reverse "sensor_flag" values to: 0 - included; 1 - excluded
    * Use missing_value = 0
    """
    COMPOSITES_TO_GENERATE = []
    S3_PREFIX = 's3://'
    DRY_RUN = False

    ORIGINAL_DIR = 'sandbox_original'
    NC_ENGINE = 'h5netcdf'

    def __init__(self, bucket: str, bucket_dir: str, target_bucket_dir: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.bucket_dir = bucket_dir
        self.target_bucket_dir = target_bucket_dir

        # Collect names for existing composites
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_mosaics = []
        files = self.s3.ls(os.path.join(bucket, bucket_dir))
        files = [each for each in files if each.endswith('0000_v02.nc')]
        self.all_mosaics.extend(files)

        # Sort the list to guarantee the order of found files
        self.all_mosaics.sort()
        logging.info(f"Found number of static mosaics: {len(self.all_mosaics)}")

        # For debugging only
        # self.all_mosaics = [self.all_mosaics[0]]

        self.num_to_fix = len(self.all_mosaics)
        logging.info(f"{self.num_to_fix} mosaics to fix: {json.dumps(self.all_mosaics, indent=4)}")

    def __call__(self, local_dir: str, num_dask_workers: int, start_index: int = 0):
        """
        Fix sensor_flag of ITS_LIVE static mosaics stored in S3 bucket.
        """
        if self.num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        # Create directory to store original mosaics - faster to read the whole thing if locally
        if not os.path.exists(FixMosaics.ORIGINAL_DIR):
            os.mkdir(FixMosaics.ORIGINAL_DIR)

        start = start_index

        num_to_fix = self.num_to_fix - start
        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixMosaics.all)(
                FixMosaics.ORIGINAL_DIR,
                each,
                self.bucket_dir,
                self.target_bucket_dir,
                local_dir
            ) for each in self.all_mosaics[start:start+num_tasks]]

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
    def all(local_original_dir: str, mosaic_url: str, bucket_dir: str, target_bucket_dir: str, local_dir: str):
        """
        Fix static mosaics and copy them back to S3 bucket.
        """
        msgs = [f'Processing {mosaic_url}']

        mosaic_basename = os.path.basename(mosaic_url)

        # Write original mosaic locally, fix it, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, mosaic_basename)

        # Bring original mosaics file locally as it's too slow to read the whole file from S3
        source_url = mosaic_url
        if not source_url.startswith(ITSCube.S3_PREFIX):
            source_url = ITSCube.S3_PREFIX + source_url

        local_original_mosaic = os.path.join(local_original_dir, mosaic_basename)

        command_line = [
            "awsv2", "s3", "cp",
            source_url,
            local_original_mosaic
        ]

        msgs.append(f"Creating local copy of {source_url}: {local_original_mosaic}")
        msgs.append(' '.join(command_line))

        env_copy = os.environ.copy()

        command_return = subprocess.run(
            command_line,
            env=env_copy,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if command_return.returncode != 0:
            raise RuntimeError(f"Failed to copy {source_url} to {local_original_mosaic}: {command_return.stdout}")

        with xr.open_dataset(local_original_mosaic, engine=FixMosaics.NC_ENGINE) as ds:
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
            msgs.append(f"Saving mosaics to {fixed_file}")

            ds.to_netcdf(fixed_file, engine=ITSLiveAnnualMosaics.NC_ENGINE)

        target_url = mosaic_url.replace(bucket_dir, target_bucket_dir)
        if not target_url.startswith(FixMosaics.S3_PREFIX):
            target_url = FixMosaics.S3_PREFIX + target_url

        if FixMosaics.DRY_RUN:
            msgs.append(f'DRYRUN: copy mosaic to {target_url}')
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
            os.unlink(fixed_file)

            msgs.append(f'Removing local {local_original_mosaic}')
            os.unlink(local_original_mosaic)

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
        help='AWS S3 that stores ITS_LIVE static mosaics to fix "sensor_flag" for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='mosaics/annual/v2/netcdf',
        help='AWS S3 bucket and directory that store static mosaics [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='mosaics/annual/v2/netcdf_fixed_sensor_flag',
        help='AWS S3 directory to store fixed static mosaics [%(default)s]'
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
        help='Index for the start mosaics to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    FixMosaics.DRY_RUN = args.dryrun

    fix_composites = FixMosaics(args.bucket, args.bucket_dir,  args.target_bucket_dir)
    fix_composites(args.local_dir, args.dask_workers, args.start_index)


if __name__ == '__main__':
    main()
    logging.info("Done.")
