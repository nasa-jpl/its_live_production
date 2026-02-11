#!/usr/bin/env python3
"""
Script to create GeoJSON FeatureCollection from ITS_LIVE velocity mosaic
.nc files.
Reads .nc files from S3 and extracts geo_polygon from corresponding JSON files.
"""
import argparse
import json
import logging
import numpy as np
import os
import re
import sys
from typing import List, Dict, Any
import boto3
import xarray as xr

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


def list_nc_files(s3_client, bucket: str, prefix: str) -> List[str]:
   """
   List all .nc files in the specified S3 bucket/prefix.

   Args:
      s3_client: Boto3 S3 client
      bucket: S3 bucket name
      prefix: S3 prefix to search

   Returns:
      List of S3 keys for .nc files
   """
   nc_files = []
   paginator = s3_client.get_paginator('list_objects_v2')

   logging.info(f"Scanning s3://{bucket}/{prefix} for .nc files...")

   for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
      if 'Contents' not in page:
         continue

      for obj in page['Contents']:
         key = obj['Key']
         if key.endswith('.nc'):
               nc_files.append(key)

   logging.info(f"Found {len(nc_files)} .nc files")
   return nc_files


def get_coordinates_bbox(coordinates):
   """Get bounding box for polygon(s).

   Args:
      coordinates: List of lists of [lon, lat] polygons coordinates.

   Returns:
      Bounding box coordinates in [min_lon, min_lat, max_lon, max_lat] format.
   """
   all_points = []

   for polygon in coordinates:
      all_points.extend(polygon)

   points = np.array(all_points)
   return [
      points[:, 0].min(),  # min_lon
      points[:, 1].min(),  # min_lat
      points[:, 0].max(),  # max_lon
      points[:, 1].max()   # max_lat
   ]


def get_geo_polygon(s3_client, bucket: str, nc_key: str) -> List[List[List[float]]]:
   """
   Get geo_polygon from the JSON file corresponding to the .nc file.

   Args:
      s3_client: Boto3 S3 client
      bucket: S3 bucket name
      nc_key: S3 key of the .nc file

   Returns:
      Polygon's bounding box coordinates.
   """
   # Replace .nc extension with .json
   json_key = nc_key.replace('.nc', '.json')

   response = s3_client.get_object(Bucket=bucket, Key=json_key)
   json_content = json.loads(response['Body'].read().decode('utf-8'))

   # Extract geo_polygon from JSON
   geo_polygon = json_content['geo_polygon']

   # Return bounding box for the region
   return get_coordinates_bbox(geo_polygon)


def get_projection_epsg(bucket: str, nc_key: str):
   """
   Extract projection EPSG code from .nc file.

   Args:
      s3_client: Boto3 S3 client
      bucket: S3 bucket name
      nc_key: S3 key of the .nc file

   Returns:
      EPSG code
   """
   nc_key_url = os.path.join("s3://", bucket, nc_key)
   logging.info(f'Getting projection for {nc_key_url}...')
   ds = xr.open_dataset(nc_key_url, engine="h5netcdf")
   epsg = ds.attrs['projection']
   logging.info(f'...got {epsg=}')

   return epsg


def create_feature(
   s3_url: str,
   coordinates: List[List[List[float]]],
   epsg: int,
   rgi: str,
   year: str=None) -> Dict[str, Any]:
   """
   Create a GeoJSON Feature for a single .nc file.

   Args:
      s3_url: Full S3 URL to the .nc file
      coordinates: Polygon coordinates
      epsg: EPSG

   Returns:
      GeoJSON Feature dictionary
   """
   start_time = None
   end_time = None

   if year is None:
      # Static mosaics cover a range of 2014-2024 years
      start_time = "01-Jan-2014"
      end_time = "31-Dec-2024"

   else:
      start_time = f"01-Jan-{year}"
      end_time = f"31-Dec-{year}"

   logging.info(f'Setting {start_time=} and {end_time=} for {s3_url}')

   return {
      "type": "Feature",
      "bbox": coordinates,
      "properties": {
         "epsg": epsg,
         "rgi": rgi,
         "startTime": start_time,
         "endTime": end_time,
         "url": s3_url,
      }
   }


def get_region_year_from_filename(filename: str) -> (str, str):
   """
   Extract RGI region and year from the filename.

   Args:
      filename: Name of the .nc file

   Returns:
      Tuple of (RGI region, year)
   """
   match = re.search(r'_(RGI\d+[A-Z])_(\d{4})_', filename)
   if match:
      rgi_region = match.group(1)
      year = match.group(2)
      return rgi_region, year

   raise RuntimeError(f"Could not extract RGI id and year from {filename}")


def create_geojson(s3_client, bucket: str, nc_files: List[str],
                     annual_nc_files: List[str]) -> Dict[str, Any]:
   """
   Create GeoJSON FeatureCollection from list of .nc files.

   Args:
      bucket: S3 bucket name
      nc_files: List of S3 keys for .nc files
      use_unsigned: Whether to use unsigned S3 requests (for public buckets)

   Returns:
      GeoJSON FeatureCollection dictionary
   """
   features = []

   for i, nc_key in enumerate(nc_files, 1):
      logging.info(f"Processing {i}/{len(nc_files)} files...")

      # Extract RGI region and year from the filename
      rgi, _ = get_region_year_from_filename(nc_key)

      # Get polygon coordinates from JSON
      coordinates = get_geo_polygon(s3_client, bucket, nc_key)

      # Get projection/EPSG from .nc file
      epsg = str(get_projection_epsg(bucket, nc_key))

      # Create S3 URL
      s3_url = f"s3://{bucket}/{nc_key}"

      # Create feature
      feature = create_feature(s3_url, coordinates, epsg, rgi)
      features.append(feature)

      # For annual mosaics, use the stored coordinates from static mosaics
      annual_files = [f for f in annual_nc_files if rgi in f]
      for annual_nc_key in annual_files:
         logging.info(f"Processing annual mosaic {annual_nc_key}...")

         # Extract the year from the filename
         _, year = get_region_year_from_filename(annual_nc_key)

         # Create S3 URL
         s3_url_annual = f"s3://{bucket}/{annual_nc_key}"

         # Create feature
         feature_annual = create_feature(s3_url_annual, coordinates, epsg,
                                          rgi, year)
         features.append(feature_annual)


   # Create FeatureCollection
   feature_collection = {
      "type": "FeatureCollection",
      "features": features
   }

   return feature_collection


def main():
   parser = argparse.ArgumentParser(
      description='Create GeoJSON FeatureCollection from ITS_LIVE velocity '
                  'mosaic .nc files'
   )
   parser.add_argument(
      '--bucket',
      default='its-live-data',
      help='S3 bucket name [%(default)s]'
   )
   parser.add_argument(
      '--prefix',
      default='velocity_mosaic/v2.1/static',
      help='S3 prefix to search [%(default)s]'
   )
   parser.add_argument(
      '--annualPrefix',
      default='velocity_mosaic/v2.1/annual',
      help='S3 prefix to search for annual mosaics [%(default)s]'
   )
   parser.add_argument(
      '--output',
      '-o',
      default='mosaics_catalog_v2.1.json',
      help='Output GeoJSON file path [%(default)s]'
   )

   args = parser.parse_args()

   # Configure S3 client for listing
   s3_client = boto3.client('s3')

   # List all static .nc files
   nc_files = list_nc_files(s3_client, args.bucket, args.prefix)

   # List all annual .nc files
   annual_nc_files = list_nc_files(s3_client, args.bucket, args.annualPrefix)

   if not nc_files:
      logging.info("No .nc files found!")
      sys.exit(1)

   # Create GeoJSON
   logging.info("Creating GeoJSON FeatureCollection...")
   geojson = create_geojson(
               s3_client, args.bucket, nc_files, annual_nc_files
            )

   # Write output
   logging.info(f"Writing to {args.output}...")
   with open(args.output, 'w') as f:
      json.dump(geojson, f, indent=3)

   logging.info(f"Wrote {args.output}")

   logging.info("Done.")


if __name__ == '__main__':
   main()