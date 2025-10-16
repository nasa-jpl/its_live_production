"""
Tool to change dtype of data arrays of string type to have maximum
allowed number of characters (5). This is to accomodate changes in
string dtype handling in newer versions of conda packages (zarr, xarray)
when appending new layers to existing cubes.

Newer conda packages we use in itscube.py code now requires that all
values of string data array to conform to the same dtype when appending
new layers to the existing cube. For example, if the existing cube has
dtype='S3' (byte string of max 3 characters), and we try to append
a new layer with dtype='S5' (byte string of max 5 characters), we get
an error. To avoid this, we need to ensure that all string data arrays
have the same dtype with sufficient number of characters to hold all
possible values.
"""
import boto3
import logging
import numpy as np
import os
import xarray as xr
import s3fs
import zarr

import itslive_utils

logging.basicConfig(level=logging.INFO)

# List of variable names to change per dtype
VARS_TO_CHANGE = {
   '<U5': ['sensor_img1', 'sensor_img2'],
   '<U2': ['satellite_img1', 'satellite_img2']
}


def change_string_dtype(
   s3_path: str,
   source_path: str,
   backup_path: str,
   fs: s3fs.S3FileSystem,
   bucket: boto3.resource('s3').Bucket,
) -> None:
   """
   Change the dtype of specific variables of string dtype in a zarr store on
   S3. Delete old xr.DataArray that corresponds to the variable and add new
   one with new dtype.

   The change is done in place, i.e., the original variable is removed
   and replaced with a new one with the specified dtype without changing
   the whole Zarr store.

   Parameters
   ----------
   s3_path : str
      S3 path to the zarr store (e.g., 'bucket/path/to/store.zarr')
   source_path: str
      S3 top level path for all Zarr datacubes
   backup_path: str
      S3 path to the backup location for cube metadata (e.g.,
      's3://bucket/path/to/backup')
   fs : s3fs.S3FileSystem
      An initialized s3fs filesystem object with appropriate credentials.
   bucket: boto3.Bucket
      An initialized boto3 S3 bucket resource object.
   """
   # # Remove 's3://' prefix if present
   # if s3_path.startswith('s3://'):
   #    s3_path = s3_path[5:]

   logging.info(f"Connecting to S3 zarr store: {s3_path}")

   # Create S3Map for zarr
   store = s3fs.S3Map(root=s3_path, s3=fs, check=False)

   # Open the dataset
   logging.info("Opening dataset...")
   ds = xr.open_zarr(store, consolidated=True)

   changed_vars = []
   zarr_group = None

   env_copy = os.environ.copy()
   _, cube_url = itslive_utils.bucket_cube_name_from_url(s3_path)
   target_path = s3_path.replace(source_path, backup_path)
   target_url = cube_url.replace(source_path, backup_path)

   for new_dtype, variable_names in VARS_TO_CHANGE.items():
      for variable_name in variable_names:
         # Get the variable
         var = ds[variable_name]
         logging.info(f"Current dtype of '{variable_name}': {var.dtype}")

         if str(var.dtype) == new_dtype:
            logging.info(
               f"Variable '{variable_name}' already has dtype {new_dtype}. "
               f"Skipping..."
            )

         else:
            logging.info(f"Changing '{variable_name}' to dtype={new_dtype}...")

            # If it's very first data variable to change the dtype, backup
            # cube metadata
            if zarr_group is None:
               logging.info("Backing up cube metadata...")
               for each_meta in itslive_utils.CUBE_META:
                  logging.info(f'Backing up {each_meta}...')
                  itslive_utils.backup_chunk(
                     bucket, cube_url, each_meta, target_url
                  )

            # Back up the entire variable before the dtype change is done
            command_line = ["awsv2", "s3", "cp", "--recursive"]
            command_line.extend([
               os.path.join(s3_path, variable_name),
               os.path.join(target_path, variable_name),
               "--acl", "bucket-owner-full-control"
            ])

            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            changed_vars.append(variable_name)

            var_data = var.values
            converted_data = var_data.astype(new_dtype)

            var_attrs = dict(var.attrs)
            # Add missing _ARRAY_DIMENSIONS attribute: required by xarray
            var_attrs['_ARRAY_DIMENSIONS'] = ['mid_date']

            var_encoding = dict(var.encoding)
            var_encoding['dtype'] = new_dtype

            if zarr_group is None:
               # Open zarr group in write mode
               logging.info("Opening zarr group in write mode...")
               zarr_group = zarr.open_group(store, mode='r+')

            # Delete the old array
            logging.info(
               f"Removing old array '{variable_name}' from zarr group..."
            )
            del zarr_group[variable_name]

            # Create new array with new dtype
            logging.info(f"Creating new array with dtype {new_dtype}...")

            # zarr v 2.18.7 does not have create_array
            new_array = zarr_group.create_dataset(
               variable_name,
               data=converted_data,
               # chunks=var.chunks,  # Preserve original chunking
               dtype=np.dtype(new_dtype),
               overwrite=False
            )

            # Restore attributes
            new_array.attrs.update(var_attrs)
            new_array.encoding = var_encoding

   if len(changed_vars):
      # Re-consolidate metadata
      logging.info("Consolidating metadata...")
      zarr.consolidate_metadata(store)
      logging.info(f"Changed dtype of '{changed_vars}' to {new_dtype}")

   else:
      logging.info("No variables were changed.")


if __name__ == "__main__":
   import argparse

   parser = argparse.ArgumentParser(
      description="Change the dtype of some string variables in a zarr "
                  "datacubes as stored in AWS S3 bucket."
   )
   parser.add_argument(
      "--s3Bucket",
      type=str,
      default="its-live-data",
      help="S3 bucket name that stores Zarr datacubes [%(default)s]"
   )
   parser.add_argument(
      "--sourcePath",
      type=str,
      default="datacubes/v2-updated-october2024",
      help="S3 path to the top level directory that stores Zarr datacubes "
         "[%(default)s]"
   )
   parser.add_argument(
      "--backupPath",
      type=str,
      default="test-space/backup/v2_datacubes_dtype_change",
      help="S3 path to the top level directory to store backup of datacubes "
            "variables that are updated to new dtype [%(default)s]"
   )

   args = parser.parse_args()

   # Set up S3 filesystem
   fs = s3fs.S3FileSystem()

   # List all zarr datacubes in the source path
   all_zarr_datacubes = []
   for each in fs.ls(os.path.join(args.s3Bucket, args.sourcePath)):
      cubes = fs.ls(each)
      cubes = [
         f's3://{each_cube}' for each_cube in cubes if each_cube.endswith('.zarr')
      ]
      all_zarr_datacubes.extend(cubes)

   s3 = boto3.resource('s3')
   s3_bucket = s3.Bucket(args.s3Bucket)

   # Process each zarr datacube
   for each_cube in all_zarr_datacubes:
      change_string_dtype(
         s3_path=each_cube,
         source_path=args.sourcePath,
         backup_path=args.backupPath,
         fs=fs,
         bucket=s3_bucket
      )

   logging.info(f"Done.")
