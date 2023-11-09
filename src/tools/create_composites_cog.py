#!/usr/bin/env python
"""
Create COGs for a specific data variable for all existing ITS_LIVE composites.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UAF)
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
import zarr


class CompositesCOG:
    """
    Class to create COGs for ITS_LIVE datacubes composites.
    """
    S3_PREFIX = 's3://'

    S3_BUCKET = 'its-live-data'
    HTTPS_BUCKET = 'https://its-live-data.s3.amazonaws.com'

    PROJECTION = 'projection'

    DRY_RUN = False

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

        self.all_composites = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            composites = self.s3.ls(each)
            composites = [each for each in composites if each.endswith('.zarr')]
            self.all_composites.extend(composites)

        # For debugging only: to process only specific sub-directory, such as:
        # python ./create_composites_cog.py -d composites/annual/v2/N50W140
        # self.all_composites = [each_cube for each_cube in self.s3.ls(os.path.join(bucket, bucket_dir)) if each_cube.endswith('.zarr')]

        # Sort the list to guarantee the order of found stores
        self.all_composites.sort()
        logging.info(f"Found number of composites: {len(self.all_composites)}")

        # For debugging only
        # self.all_composites = self.all_composites[:1]
        # logging.info(f"ULRs: {self.all_composites}")

    def no__call__(self, var_name: str, local_dir: str, num_dask_workers: int, start_index: int=0):
        """
        Create COG for the "var_name" data variable of composites sequentially.
        """
        num_to_generate = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_generate} composites to fix...")

        if num_to_generate <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for each in self.all_composites:
            logging.info(f"Starting {each}")
            msgs = CompositesCOG.all(var_name, each, self.bucket, self.target_bucket_dir, local_dir, self.s3)
            logging.info("\n-->".join(msgs))

    def __call__(
        self,
        var_name: str,
        local_dir: str,
        num_dask_workers: int,
        start_index: int=0
    ):
        """
        Create COG for the "var_name" data variable of composites using parallel processing.
        """
        num_to_generate = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_generate} composites to process...")

        if num_to_generate <= 0:
            logging.info(f"Nothing to process, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_generate > 0:
            num_tasks = num_dask_workers if num_to_generate > num_dask_workers else num_to_generate

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(CompositesCOG.all)(var_name, each, self.bucket, self.target_bucket_dir, local_dir, self.s3) for each in self.all_composites[start:start+num_tasks]]
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

            num_to_generate -= num_tasks
            start += num_tasks

    @staticmethod
    def all(
        var_name: str,
        composite_url: str,
        bucket_name: str,
        target_bucket_dir: str,
        local_dir: str,
        s3_in
    ):
        """
        Generate COG for the data variable of interest and copy the file to the S3 bucket in
        provided target directory.

        Use CLI command to generate COG. For example:
        gdal_translate -of COG 'ZARR:"/vsicurl/https://its-live-data.s3.amazonaws.com/composites/annual/v2/N50W140/ITS_LIVE_velocity_EPSG3413_120m_X-3350000_Y350000.zarr":/v0' junk_cog.tif -a_srs epsg:3413 -co TILING_SCHEME=GoogleMapsCompatible

        where "3413" for the "epsg:3413" command-line option is the projection attribute of the composite.

        Output COG filename should be formatted:
         s3://its-live-data/composites/annual/v2/cog/v0/ITS_LIVE_velocity_EPSG32735_120m_X750000_Y10050000_v0.tif
        """
        msgs = [f'Processing {composite_url}']

        zarr_store = s3fs.S3Map(root=composite_url, s3=s3_in, check=False)
        composite_basename = os.path.basename(composite_url)
        cog_basename = composite_basename.replace('.zarr', f'_{var_name}.tif')

        # Write COG locally, upload it to the bucket, remove file
        cog_file = os.path.join(local_dir, cog_basename)

        env_copy = os.environ.copy()

        # Extract projection code:
        projection = None
        with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            projection = ds.attrs[CompositesCOG.PROJECTION]

        msgs.append(f"Creating COG for {composite_url}: {cog_file}")

        http_composite = composite_url.replace(CompositesCOG.S3_BUCKET, CompositesCOG.HTTPS_BUCKET)
        command_line = [
            'gdal_translate',
            '-of',
            'COG',
            f'ZARR:"/vsicurl/{http_composite}":/{var_name}',
            cog_file,
            '-a_srs',
            f'epsg:{projection}',
            '-co',
            'TILING_SCHEME=GoogleMapsCompatible'
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
            msgs.append(f"ERROR: Failed to create {cog_basename}: {command_return.stdout}")
            return msgs

        target_cog_url = os.path.join(
            CompositesCOG.S3_PREFIX,
            bucket_name,
            target_bucket_dir,
            var_name,
            cog_basename
        )

        if CompositesCOG.DRY_RUN:
            msgs.append(f'DRYRUN: copy {cog_basename} to {target_cog_url}')
            return msgs

        if os.path.exists(cog_file) and len(bucket_name):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy
            command_line = [
                "aws", "s3", "cp",
                cog_file,
                target_cog_url,
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
                msgs.append(f"ERROR: Failed to copy {cog_basename} to {target_cog_url}: {command_return.stdout}")

            msgs.append(f"Removing local {cog_file}")
            os.unlink(cog_file)

        return msgs

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--variable_name',
        type=str,
        default='v0',
        help='Data variable name to generate COGs for [%(default)s]'
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE annual composites to generate COG for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='composites/annual/v2',
        help='AWS S3 bucket and directory that store annual composites [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir', type=str,
        default='composites/annual/v2/cog',
        # default='test_datacubes/composites/annual/v2/cog_NoGoogleComp',   # Debugging only
        help='AWS S3 directory to store COG files to [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir', type=str,
        default='sandbox',
        help='Directory to store fixed granules before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument('-w', '--dask-workers', type=int,
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
        help='Index for the start datacube to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    CompositesCOG.DRY_RUN = args.dryrun

    fix_composites = CompositesCOG(args.bucket, args.bucket_dir,  args.target_bucket_dir)
    fix_composites(args.variable_name, args.local_dir, args.dask_workers, args.start_index)

if __name__ == '__main__':
    main()
    logging.info("Done.")
