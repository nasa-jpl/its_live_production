#!/usr/bin/env python
"""
Re-compute v_error based on vx and vy components instead of vx0 and vy0 components as
it was done originally. All composites reside in AWS S3 bucket.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis
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
from itslive_composite import CompDataVars


class FixAnnualComposites:
    """
    Class to apply fixes to ITS_LIVE datacubes composites:

    * Re-compute v_error data based on vx and vy data variables instead of
      vx0 and vy0 for unit vector as it was done in composites code originally
    """
    # Suffix to remove in original granule URLs
    SUFFIX_TO_USE = '.nc'
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

        # Collect names for existing datacubes
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_composites = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            cubes = self.s3.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]
            self.all_composites.extend(cubes)

        # Sort the list to guarantee the order of found stores
        self.all_composites.sort()
        logging.info(f"Found number of composites: {len(self.all_composites)}")

        # For debugging only
        # self.all_composites = self.all_composites[:8]
        # logging.info(f"ULRs: {self.all_composites}")

    def debug__call__(self, local_dir: str, num_dask_workers: int, start_index: int=0):
        """
        Fix v_error of ITS_LIVE datacubes' composites stored in S3 bucket.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} composites to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for each in self.all_composites:
            logging.info(f"Starting {each}")
            msgs = FixAnnualComposites.all(each, self.bucket, self.bucket_dir, self.target_bucket_dir, local_dir, self.s3)
            logging.info("\n-->".join(msgs))

    def __call__(self, local_dir: str, num_dask_workers: int, start_index: int=0):
        """
        Fix v_error of ITS_LIVE datacubes' composites stored in S3 bucket.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} datacubes to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixAnnualComposites.all)(each, self.bucket, self.bucket_dir, self.target_bucket_dir, local_dir, self.s3) for each in self.all_composites[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

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

        # Use composite parent directory to format local filename as there are
        # multiple copies of the same composite filename under different sub-directories
        dir_tokens = composite_url.split('/')

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, f'{dir_tokens[-2]}_{composite_basename}')

        with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            sizes = ds.sizes

            # Fix v_error
            vx = ds.vx.values
            vy = ds.vy.values
            v = ds.v.values
            vx_error = ds.vx_error.values
            vy_error = ds.vy_error.values

            valid_mask = np.where(
                (vx != DataVars.MISSING_VALUE) & (vy != DataVars.MISSING_VALUE) &
                (v != DataVars.MISSING_VALUE) & (v != 0)
            )

            v_error = np.full_like(v, DataVars.MISSING_VALUE, dtype=np.float32)
            v_error[valid_mask] = (vx_error[valid_mask]*np.abs(vx[valid_mask]) + vy_error[valid_mask]*np.abs(vy[valid_mask]))/v[valid_mask]

            # Replace v_error variable in dataset
            ds[CompDataVars.V_ERROR] = xr.DataArray(
                data=v_error,
                coords=ds[CompDataVars.V_ERROR].coords,
                dims=ds[CompDataVars.V_ERROR].dims,
                attrs=ds[CompDataVars.V_ERROR].attrs
            )
            msgs.append(f"Saving composite to {fixed_file}")

            # Set encoding
            encoding_settings = {}
            encoding_settings.setdefault(CompDataVars.TIME, {}).update({DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS})

            for each in [CompDataVars.TIME, CompDataVars.SENSORS, Coords.X, Coords.Y]:
                encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

            encoding_settings.setdefault(CompDataVars.SENSORS, {}).update({'dtype': 'str'})

            # Compression for the data
            compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)

            # Settings for "float" data types
            for each in [
                DataVars.VX,
                DataVars.VY,
                DataVars.V,
                CompDataVars.VX_ERROR,
                CompDataVars.VY_ERROR,
                CompDataVars.V_ERROR,
                CompDataVars.VX_AMP_ERROR,
                CompDataVars.VY_AMP_ERROR,
                CompDataVars.V_AMP_ERROR,
                CompDataVars.VX_AMP,
                CompDataVars.VY_AMP,
                CompDataVars.V_AMP,
                CompDataVars.VX_PHASE,
                CompDataVars.VY_PHASE,
                CompDataVars.V_PHASE,
                CompDataVars.OUTLIER_FRAC,
                CompDataVars.VX0,
                CompDataVars.VY0,
                CompDataVars.V0,
                CompDataVars.VX0_ERROR,
                CompDataVars.VY0_ERROR,
                CompDataVars.V0_ERROR,
                CompDataVars.SLOPE_VX,
                CompDataVars.SLOPE_VY,
                CompDataVars.SLOPE_V
                ]:
                encoding_settings.setdefault(each, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                    'dtype': np.float32,
                    'compressor': compressor
                })

            # Settings for "short" datatypes
            for each in [
                CompDataVars.COUNT,
                CompDataVars.COUNT0
            ]:
                encoding_settings.setdefault(each, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                    'dtype': np.short
                })

            # Settings for "max_dt" datatypes
            encoding_settings.setdefault(CompDataVars.MAX_DT, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_POS_VALUE,
                    'dtype': np.short
                })

            # Settings for "sensor_include" datatypes
            encoding_settings.setdefault(CompDataVars.SENSOR_INCLUDE, {}).update({
                    'dtype': np.short
                })

            # Chunking to apply when writing datacube to the Zarr store
            chunks_settings = (1, sizes[Coords.Y], sizes[Coords.X])

            for each in [
                DataVars.VX,
                DataVars.VY,
                DataVars.V,
                CompDataVars.VX_ERROR,
                CompDataVars.VY_ERROR,
                CompDataVars.V_ERROR,
                CompDataVars.MAX_DT
            ]:
                encoding_settings[each].update({
                    'chunks': chunks_settings
                })

            # Chunking to apply when writing datacube to the Zarr store
            chunks_settings = (sizes[Coords.Y], sizes[Coords.X])

            for each in [
                CompDataVars.VX_AMP,
                CompDataVars.VY_AMP,
                CompDataVars.V_AMP,
                CompDataVars.VX_PHASE,
                CompDataVars.VY_PHASE,
                CompDataVars.V_PHASE,
                CompDataVars.VX_AMP_ERROR,
                CompDataVars.VY_AMP_ERROR,
                CompDataVars.V_AMP_ERROR,
                CompDataVars.OUTLIER_FRAC,
                CompDataVars.SENSOR_INCLUDE,
                CompDataVars.VX0,
                CompDataVars.VY0,
                CompDataVars.V0,
                CompDataVars.VX0_ERROR,
                CompDataVars.VY0_ERROR,
                CompDataVars.V0_ERROR,
                CompDataVars.SLOPE_VX,
                CompDataVars.SLOPE_VY,
                CompDataVars.SLOPE_V
                ]:
                encoding_settings[each].update({
                    'chunks': chunks_settings
                })

            # logging.info(f"Encoding settings: {encoding_settings}")
            ds.to_zarr(fixed_file, encoding=encoding_settings, consolidated=True)

        target_url = composite_url.replace(bucket_dir, target_bucket_dir)
        if not target_url.startswith(FixAnnualComposites.S3_PREFIX):
            target_url = FixAnnualComposites.S3_PREFIX + target_url

        if FixAnnualComposites.DRY_RUN:
            msgs.append(f'DRYRUN: copy composite to {target_url}')
            return msgs

        if os.path.exists(fixed_file) and len(bucket_name):
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
        '-b', '--bucket', type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE annual composites to fix v_error for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='composites/annual/v02',
        help='AWS S3 bucket and directory that store annual composites [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir', type=str,
        default='composites/annual/v02_fixed_verror',
        help='AWS S3 directory to store fixed annual composites [%(default)s]'
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
    FixAnnualComposites.DRY_RUN = args.dryrun

    fix_composites = FixAnnualComposites(args.bucket, args.bucket_dir,  args.target_bucket_dir)
    fix_composites(args.local_dir, args.dask_workers, args.start_index)

if __name__ == '__main__':
    main()
    logging.info("Done.")
