"""
Script to add 'spatial_ref' attribute to the 'mapping' variable in NetCDF
mosaics files stored in an AWS S3 bucket.

Usage:
   python add_mosaics_mapping_attribute.py --bucket my-bucket --prefix data/
   --dryrun --pattern "ITS_LIVE_velocity_120m_RGI01A*.nc"
"""

import argparse
import logging
import fnmatch
import os
from pathlib import Path

import boto3
import xarray as xr
from botocore.exceptions import ClientError


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

   nc_files = []
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


def process_nc_file(local_path: Path) -> bool:
   """
   Add spatial_ref attribute to the 'mapping' variable in a NetCDF file.

   Args:
      local_path: Path to local NetCDF file
      spatial_ref_value: WKT string for spatial_ref attribute

   Returns:
      True if successful, False otherwise
   """
   try:
      # Open dataset with write permissions
      with xr.open_dataset(local_path, mode='r+') as ds:
         # Add or update the spatial_ref attribute to have the same value as
         # crs_wkt attribute
         ds.mapping.attrs['spatial_ref'] = ds.mapping.attrs['crs_wkt']
         logging.info(f"Added spatial_ref attribute to 'mapping' variable in {local_path.name}")

      return True

   except Exception as e:
      logging.error(f"Error processing {local_path.name}: {e}")
      return False

LOCAL_DIR = 'fix_mosaics_mapping_attribute'

def main():
   parser = argparse.ArgumentParser(
      description="Add spatial_ref attribute to mapping variable in S3 mosaics files"
   )
   parser.add_argument(
      '--bucket',
      default='its-live-data',
      help='S3 bucket name'
   )
   parser.add_argument(
      '--prefix',
      default='velocity_mosaic/v2.1/production/post_process/',
      help='S3 directory prefix that stores mosaics NetCDF files'
   )
   parser.add_argument(
      '--dryrun',
      action='store_true',
      help='List files without making changes'
   )
   parser.add_argument(
      '--pattern',
      default='*.nc',
      help='File pattern to match (default: *.nc)'
   )

   args = parser.parse_args()

   setup_logging()

   # Initialize S3 client
   s3_client = boto3.client('s3')

   # Get list of .nc files
   nc_files = list_nc_files(s3_client, args.bucket, args.prefix, args.pattern)

   if not nc_files:
      logging.info("No .nc files found. Exiting.")
      return

   if args.dryrun:
      logging.info("Dry run mode - would process the following files:")
      for file_key in nc_files:
         logging.info(f"  {file_key}")
      return

   # Process each file
   success_count = 0
   total_count = len(nc_files)

   temp_path = Path(LOCAL_DIR)

   for file_key in nc_files:
      try:
            local_file = temp_path / Path(file_key).name

            # Download file from S3
            logging.info(f"Downloading s3://{args.bucket}/{file_key}")
            s3_client.download_file(args.bucket, file_key, str(local_file))

            # Process the file
            if process_nc_file(local_file):
               # Upload modified file back to S3
               logging.info(f"Uploading modified file to s3://{args.bucket}/{file_key}")
               s3_client.upload_file(str(local_file), args.bucket, file_key)
               success_count += 1

            else:
               logging.warning(f"Skipped {file_key} due to processing error")

      except ClientError as e:
            logging.error(f"AWS error processing {file_key}: {e}")
      except Exception as e:
            logging.error(f"Unexpected error processing {file_key}: {e}")

   logging.info(
      f"Completed processing {total_count} files. "
      f"Successfully updated {success_count} files."
   )


if __name__ == "__main__":
   main()