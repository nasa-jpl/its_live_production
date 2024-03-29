#!/usr/bin/env python
"""
Apply some fixed to existing ITS_LIVE datacubes that are residing in AWS S3 bucket:

* Fix mapping.GeoTransform to capture origin of the datacube tile (used metdata of
  first granule (each granule has different origin))

* Remove '_IL_ASF_OD' suffix from all granule URLs as used within the datacube
 (granules names are being fixed after the transfer)

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
import json
import logging
import os
import s3fs
import shutil
import subprocess
import xarray as xr
import zarr

from itscube_types import DataVars, Coords
import zarr_to_netcdf


class FixDatacubes:
    """
    Class to apply fixes to ITS_LIVE datacubes:

    * Fix mapping.GeoTransform to capture origin of the datacube tile - not first
      granule as original datacube code was doing (each granule has different origin)

    * Remove '_IL_ASF_OD' suffix from all granule URLs as used within the datacube
     (granules names are being fixed after the transfer)
    """
    # Suffix to remove in original granule URLs
    SUFFIX_TO_REMOVE = '_IL_ASF_OD.nc'
    SUFFIX_TO_USE = '.nc'
    S3_PREFIX = 's3://'
    DASK_CHUNKS = 250
    DRY_RUN = False

    def __init__(self, bucket: str, bucket_dir: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.bucket_dir = bucket_dir

        # Collect names for existing datacubes
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_zarr_datacubes = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            cubes = self.s3.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]
            self.all_zarr_datacubes.extend(cubes)

        logging.info(f"Found number of datacubes: {len(self.all_zarr_datacubes)}")

    def debug__call__(self, local_dir: str, num_dask_workers: int):
        """
        Fix mapping.GeoTransform of ITS_LIVE datacubes stored in S3 bucket.
        Strip suffix from original granules names as appear within 'granule_url'
        data variable and skipped_* datacube attributes.
        """
        num_to_fix = len(self.all_zarr_datacubes)
        start = 0

        logging.info(f"{num_to_fix} datacubes to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for each_cube in self.all_zarr_datacubes:
            logging.info(f"Starting {each_cube}")
            msgs = FixDatacubes.all(each_cube, self.bucket, local_dir, self.s3)
            logging.info("\n-->".join(msgs))

    def __call__(self, local_dir: str, num_dask_workers: int, start_cube: int=0):
        """
        Fix mapping.GeoTransform of ITS_LIVE datacubes stored in S3 bucket.
        Strip suffix from original granules names as appear within 'granule_url'
        data variable and skipped_* datacube attributes.
        """
        num_to_fix = len(self.all_zarr_datacubes) - start_cube
        start = start_cube

        logging.info(f"{num_to_fix} datacubes to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixDatacubes.all)(each, self.bucket, local_dir, self.s3) for each in self.all_zarr_datacubes[start:start+num_tasks]]
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
    def all(cube_url: str, bucket_name: str, local_dir: str, s3_in):
        """
        Fix datacubes and copy them back to S3 bucket.
        """
        msgs = [f'Processing {cube_url}']

        if FixDatacubes.DRY_RUN:
            return msgs

        cube_store = s3fs.S3Map(root=cube_url, s3=s3_in, check=False)
        cube_basename = os.path.basename(cube_url)

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, cube_basename)

        with xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True, chunks={'mid_date': FixDatacubes.DASK_CHUNKS}) as ds:
            # Fix mapping.GeoTransform
            ds_x = ds.x.values
            ds_y = ds.y.values

            x_size = ds_x[1] - ds_x[0]
            y_size = ds_y[1] - ds_y[0]

            half_x_cell = x_size/2.0
            half_y_cell = y_size/2.0

            # Format cube's GeoTransform
            new_geo_transform_str = f"{ds_x[0] - half_x_cell} {x_size} 0 {ds_y[0] - half_y_cell} 0 {y_size}"
            ds.mapping.attrs['GeoTransform'] = new_geo_transform_str

            # Remove not used suffix from original granules filenames
            urls = [each.replace(FixDatacubes.SUFFIX_TO_REMOVE, FixDatacubes.SUFFIX_TO_USE) for each in ds.granule_url.values]
            ds[DataVars.URL] = xr.DataArray(data = urls, coords=[ds.granule_url.mid_date.values], dims=[Coords.MID_DATE])

            # Update cube attributes which use original granule URLs
            attr_data = json.loads(ds.attrs[DataVars.SKIP_EMPTY_DATA])
            attr_data = [each.replace(FixDatacubes.SUFFIX_TO_REMOVE, FixDatacubes.SUFFIX_TO_USE) for each in attr_data]
            # Replace attribute value
            ds.attrs[DataVars.SKIP_EMPTY_DATA] = json.dumps(attr_data)

            attr_data = json.loads(ds.attrs[DataVars.SKIP_DUPLICATE_MID_DATE])
            attr_data = [each.replace(FixDatacubes.SUFFIX_TO_REMOVE, FixDatacubes.SUFFIX_TO_USE) for each in attr_data]
            # Replace attribute value
            ds.attrs[DataVars.SKIP_DUPLICATE_MID_DATE] = json.dumps(attr_data)

            # Skipped wrong projection data is stored in dictionary
            attr_data = json.loads(ds.attrs[DataVars.SKIP_WRONG_PROJECTION])
            for each_key in attr_data:
                values = [each.replace(FixDatacubes.SUFFIX_TO_REMOVE, FixDatacubes.SUFFIX_TO_USE) for each in attr_data[each_key]]
                attr_data[each_key] = values
            # Replace attribute value
            ds.attrs[DataVars.SKIP_WRONG_PROJECTION] = json.dumps(attr_data)

            msgs.append(f"Saving datacube to {fixed_file}")

            ds.to_zarr(fixed_file, encoding=zarr_to_netcdf.ENCODING_ZARR, consolidated=True)

        cube_store = None

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
            target_url = cube_url

            if not target_url.startswith(FixDatacubes.S3_PREFIX):
                target_url = FixDatacubes.S3_PREFIX + cube_url

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


            # Save fixed datacube to NetCDF format file
            nc_fixed_file = fixed_file.replace('.zarr', '.nc')
            msgs.append(f"Saving datacube to {nc_fixed_file}")
            zarr_to_netcdf.main(fixed_file, nc_fixed_file, 'h5netcdf', FixDatacubes.DASK_CHUNKS)

            if os.path.exists(nc_fixed_file) and len(bucket_name):
                # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
                # resulting in as many error messages as there are files in Zarr store
                # to copy
                cube_url_nc = target_url.replace('.zarr', '.nc')
                env_copy = os.environ.copy()
                command_line = [
                    "aws", "s3", "cp",
                    nc_fixed_file,
                    cube_url_nc,
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
                    logging.info(f"ERROR: Failed to copy {nc_fixed_file} to {cube_url_nc}: {command_return.stdout}")

                msgs.append(f"Removing local {nc_fixed_file}")
                os.unlink(nc_fixed_file)

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
        default='its-live-data.jpl.nasa.gov',
        help='AWS S3 that stores ITS_LIVE granules to fix attributes for'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='datacubes/v01',
        help='AWS S3 bucket and directory that store the granules'
    )
    parser.add_argument(
        '-l', '--local_dir', type=str,
        default='sandbox',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument('-w', '--dask-workers', type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument('-c', '--chunks', type=int, default=250,
                        help="Dask chunk size for mid_date coordinate [%(default)d]. " \
                        "This is to handle datacubes that can't fit in memory, and should be read in as Dask arrays.")
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )
    parser.add_argument(
        '-s', '--start-cube',
        type=int,
        default=0,
        help='Index for the start datacube to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    FixDatacubes.DASK_CHUNKS = args.chunks
    FixDatacubes.DRY_RUN = args.dry

    fix_cubes = FixDatacubes(args.bucket, args.bucket_dir)
    fix_cubes(args.local_dir, args.dask_workers, args.start_cube)


if __name__ == '__main__':
    main()
    logging.info("Done.")
