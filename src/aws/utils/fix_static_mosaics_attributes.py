"""
Script to fix some attributes in NetCDF static mosaics files stored in an
AWS S3 bucket.

Usage:
   python fix_static_mosaics_attributes.py --bucket my-bucket --prefix data/
   --pattern "*RGI19A_0000_V02.1.nc" --dryrun
"""

import argparse
import logging
import fnmatch
from pathlib import Path

import boto3
import xarray as xr
from botocore.exceptions import ClientError

from itscube_types import CompDataVars


FIXED_FILES_DIR = Path('original_mosaics_v2.1_to_fix')


def setup_logging():
   """Configure logging format and level."""
   logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s'
   )


def list_nc_files(s3_client, bucket: str, prefix: str, pattern: str) -> list:
   """
   List all .nc files in the specified S3 bucket and prefix.

   Args:
      s3_client: Boto3 S3 client
      bucket: S3 bucket name
      prefix: Directory prefix (should end with '/' if not empty)
      pattern: File pattern to match (e.g., '*.nc')

   Returns:
      List of S3 object keys matching the pattern
   """
   if prefix and not prefix.endswith('/'):
      prefix += '/'

   matching_files = []
   paginator = s3_client.get_paginator('list_objects_v2')

   try:
      for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
         if 'Contents' in page:
            for obj in page['Contents']:
               filename = Path(obj['Key']).name
               if fnmatch.fnmatch(filename, pattern):
                  matching_files.append(obj['Key'])

   except ClientError as e:
      logging.error(f"Error listing objects in s3://{bucket}/{prefix}: {e}")
      raise

   logging.info(f"Found {len(matching_files)} .nc files in s3://{bucket}/{prefix}")
   return matching_files


def process_nc_file(local_path: str, fixed_file: str):
   """
   Add spatial_ref attribute to the 'mapping' variable in a NetCDF file.

   Args:
      local_path: Path to local NetCDF file
      spatial_ref_value: WKT string for spatial_ref attribute
      fixed_file: Path to save the modified NetCDF file.
   """
   # Open dataset with write permissions
   with xr.open_dataset(local_path, mode='r') as ds:
      # Fix standard name for some variables
      ds[CompDataVars.COUNT].attrs['standard_name'] = "count"
      ds.v.attrs['standard_name'] = "climatological [2014-2024] velocity"
      ds.v_amp.attrs['standard_name'] = "climatological [2014-2024] mean seasonal amplitude"
      ds.v_error.attrs['standard_name'] = "v_velocity_error"
      ds.vx_error.attrs['standard_name'] = "vx_velocity_error"
      ds.vy_error.attrs['standard_name'] = "vy_velocity_error"

      # Save changes to a new file
      logging.info(f"Saving modified file to {fixed_file}")
      ds.to_netcdf(fixed_file, engine='h5netcdf')


def main():
   parser = argparse.ArgumentParser(
      description="Fix some variable attributes for static mosaics files as "
                  "stored in an AWS S3 bucket."

   )
   parser.add_argument(
      '--bucket',
      default='its-live-data',
      help='S3 bucket name (default: %(default)s)'
   )
   parser.add_argument(
      '--prefix',
      default='velocity_mosaic/v2.1/production/post_process/',
      help='S3 directory prefix that stores mosaics NetCDF files (default: %(default)s)'
   )
   parser.add_argument(
      '--targetPrefix',
      default='velocity_mosaic/v2.1/static/',
      help='S3 directory prefix to store corrected mosaics NetCDF files to '
            '(default: %(default)s)'
   )
   parser.add_argument(
      '--dryrun',
      action='store_true',
      help='List files without making changes'
   )
   parser.add_argument(
      '--pattern',
      default='*_0000_V02.1.nc',
      help='File pattern to match (default: [%(default)s])'
   )

   args = parser.parse_args()

   setup_logging()

   # Initialize S3 client
   s3_client = boto3.client('s3')

   # Get list of mosaics files
   nc_files = list_nc_files(s3_client, args.bucket, args.prefix, args.pattern)

   if not nc_files:
      logging.info("No mosaics files found. Exiting.")
      return

   logging.info("Found files:")
   for file_key in nc_files:
      logging.info(f"  {file_key}")

   # Ensure the fixed files directory exists
   FIXED_FILES_DIR.mkdir(parents=True, exist_ok=True)

   # Process each file
   for file_key in nc_files:
      local_file = Path(file_key).name

      logging.info(f'Processing local file: {local_file}')

      # Download file from S3
      logging.info(f"Downloading s3://{args.bucket}/{file_key}")
      local_file_path = FIXED_FILES_DIR / local_file

      s3_client.download_file(args.bucket, file_key, str(local_file_path))

      # Process the file
      process_nc_file(str(local_file_path), str(local_file))

      # Upload modified file back to S3
      target_key = file_key.replace(args.prefix, args.targetPrefix)
      msg = f"uploading {local_file} to s3://{args.bucket}/{target_key}"

      if args.dryrun:
         logging.info(f"Dry run - not {msg}")

      else:
         # Replace file only in non-dryrun mode
         logging.info(msg)
         s3_client.upload_file(str(local_file), args.bucket, target_key)

   logging.info("Done")


if __name__ == "__main__":
   main()
