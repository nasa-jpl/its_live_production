#!/usr/bin/env python
"""
Crop ITS_LIVE granules (that are residing in AWS S3 bucket)
to the X/Y range that covers only valid data for the granule.
Replace original granule in the bucket with its cropped version.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.


Authors: Masha Liukis
"""
import argparse
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
import json
import geojson
import logging
import numpy as np
import os
import pyproj
import s3fs
import xarray as xr

from itscube_types import DataVars, Coords, Output
from mission_info import Encoding
from lon_lat_to_dir_prefix import point_to_prefix

mission_encoding = {
    'S1': Encoding.SENTINEL1,
    'S2': Encoding.LANDSAT_SENTINEL2,
    'L':  Encoding.LANDSAT_SENTINEL2
}


class ProcessV2Granules:
    """
    Crops existing V2 granules to the X/Y range that corresponds to valid data.

    Skip the granules that end with "_P000.nc" as those don't have any data to begin with.
    """
    ZERO_PERCENT_COVERAGE = '_P000.nc'

    # Flag to copy zero percent coverage files and related PNG files to the target S3 bucket.
    COPY_ZERO_PERCENT_COVERAGE_FILES = False

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = 'EPSG:4326'

    # Mission for which granules are cropped (to know encoding to use)
    MISSION = None

    # S3 bucket with granules
    BUCKET = 'its-live-data'

    # Source S3 bucket directory
    SOURCE_DIR = None

    # Target S3 bucket directory
    TARGET_DIR = None

    # Local directory to store cropped granules before copying them to the S3 bucket
    LOCAL_DIR = None

    # Number of granules to process in parallel
    CHUNK_SIZE = 100

    # Number of Dask workers for parallel processing
    DASK_WORKERS = 8

    STORE_GRANULE_LIST_FILE = False
    READ_GRANULE_LIST_FILE = False
    GRANULE_LIST_FILE = 'used_granules.json'

    def __init__(self, glob_pattern: dir):
        """
        Initialize object.

        bucket: S3 bucket to store original and cropped granules
        glob_pattern: Glob pattern to use to collect existing granules.
        """
        self.s3 = s3fs.S3FileSystem()
        self.granule_method_to_call = ProcessV2Granules.crop
        self.all_granules = []

        # Use glob to list directory
        if ProcessV2Granules.READ_GRANULE_LIST_FILE:
            granule_filename = os.path.join(
                ProcessV2Granules.BUCKET,
                ProcessV2Granules.TARGET_DIR,
                ProcessV2Granules.GRANULE_LIST_FILE
            )

            with self.s3.open(granule_filename, 'r') as ins3file:
                self.all_granules = json.load(ins3file)
                logging.info(f"Loaded {len(self.all_granules)} granules from '{granule_filename}'")

        else:
            logging.info(f"Reading {ProcessV2Granules.SOURCE_DIR}")
            self.all_granules = self.s3.glob(f'{os.path.join(ProcessV2Granules.BUCKET, ProcessV2Granules.SOURCE_DIR)}/{glob_pattern}')

        logging.info(f"Number of granules: {len(self.all_granules)}")

        if ProcessV2Granules.COPY_ZERO_PERCENT_COVERAGE_FILES is True:
            # Set the method to call for the granules
            self.granule_method_to_call = ProcessV2Granules.copy

            # Copy only zero percent coverage files over to the target bucket as they do not
            # need to be cropped
            self.all_granules = [
                each for each in self.all_granules if each.endswith(ProcessV2Granules.ZERO_PERCENT_COVERAGE)
            ]

        else:
            # Don't process granules that have 0% coverage
            self.all_granules = [
                each for each in self.all_granules if not each.endswith(ProcessV2Granules.ZERO_PERCENT_COVERAGE)
            ]

        # For debugging only: convert first granule
        # self.all_granules = self.all_granules[:1]

        logging.info(f"Number of granules to process: {len(self.all_granules)}")

        if ProcessV2Granules.STORE_GRANULE_LIST_FILE:
            # Store the granule list to the file in S3 target directory
            granule_filename = os.path.join(
                ProcessV2Granules.BUCKET,
                ProcessV2Granules.TARGET_DIR,
                ProcessV2Granules.GRANULE_LIST_FILE
            )
            with self.s3.open(granule_filename, 'w') as outs3file:
                geojson.dump(self.all_granules, outs3file)
                logging.info(f'Stored granule list to {granule_filename}')

    def __call__(self, start_index: int):
        """
        Crop V2 ITS_LIVE granules to X/Y extend of non-None values or
        copy zero percent coverage files over to new location.
        """
        num_to_fix = len(self.all_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(ProcessV2Granules.LOCAL_DIR):
            os.mkdir(ProcessV2Granules.LOCAL_DIR)

        while num_to_fix > 0:
            num_tasks = ProcessV2Granules.CHUNK_SIZE if num_to_fix > ProcessV2Granules.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [
                dask.delayed(self.granule_method_to_call)(
                    each,
                    self.s3,
                ) for each in self.all_granules[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=ProcessV2Granules.DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def crop(granule_url: str, s3: s3fs.S3FileSystem):
        """
        Crop the granule to X/Y extend of non-None "v" values, store
        new granule in NetCDF format to the local directory, and push new granule
        to the target location in S3 bucket.
        """
        msgs = [f'Processing cropping of {granule_url}']

        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                # this will drop X/Y coordinates, so drop non-None values just to get X/Y extends
                xy_ds = ds.where(ds.v.notnull(), drop=True)

                x_values = xy_ds.x.values
                grid_x_min, grid_x_max = x_values.min(), x_values.max()

                y_values = xy_ds.y.values
                grid_y_min, grid_y_max = y_values.min(), y_values.max()

                # Based on X/Y extends, mask original dataset
                mask_lon = (ds.x >= grid_x_min) & (ds.x <= grid_x_max)
                mask_lat = (ds.y >= grid_y_min) & (ds.y <= grid_y_max)
                mask = (mask_lon & mask_lat)

                # No need to check as non-P000 % coverage will have at least 0.5% valid pixels
                # if mask.values.sum() <= 1:
                #     msgs.append('There is not enough data in cropped granule, skip cropping')

                #     # Copy original granule and corresponding *png files to the destination
                #     # location in S3 bucket
                #     msgs.extend(ProcessV2Granules.copy(granule_url, s3))
                # else:
                cropped_ds = ds.where(mask, drop=True)
                cropped_ds = cropped_ds.load()

                # Reset data for DataVars.MAPPING and DataVars.ImgPairInfo.NAME
                # data variables as ds.where() extends data of all data variables
                # to the dimentions of the "mask"
                cropped_ds[DataVars.MAPPING] = ds[DataVars.MAPPING]
                cropped_ds[DataVars.ImgPairInfo.NAME] = ds[DataVars.ImgPairInfo.NAME]

                # Add date when granule was updated
                cropped_ds.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

                # Compute centroid longitude/latitude
                center_x = (grid_x_min + grid_x_max)/2
                center_y = (grid_y_min + grid_y_max)/2

                # Convert to lon/lat coordinates
                projection = ds[DataVars.MAPPING].attrs['spatial_epsg']
                to_lon_lat_transformer = pyproj.Transformer.from_crs(
                    f"EPSG:{projection}",
                    ProcessV2Granules.LON_LAT_PROJECTION,
                    always_xy=True
                )

                # Update centroid information for the granule
                center_lon_lat = to_lon_lat_transformer.transform(center_x, center_y)

                cropped_ds[DataVars.ImgPairInfo.NAME].attrs['latitude'] = round(center_lon_lat[1], 2)
                cropped_ds[DataVars.ImgPairInfo.NAME].attrs['longitude'] = round(center_lon_lat[0], 2)

                # Update mapping.GeoTransform
                x_cell = x_values[1] - x_values[0]
                y_cell = y_values[1] - y_values[0]

                # It was decided to keep all values in GeoTransform center-based
                cropped_ds[DataVars.MAPPING].attrs['GeoTransform'] = f"{x_values[0]} {x_cell} 0 {y_values[0]} 0 {y_cell}"

                # Save to local file
                granule_basename = os.path.basename(granule_url)

                # Write the granule locally, upload it to the bucket, remove file
                fixed_file = os.path.join(ProcessV2Granules.LOCAL_DIR, granule_basename)

                # Set chunking for 2D data variables
                dims = cropped_ds.dims
                num_x = dims[Coords.X]
                num_y = dims[Coords.Y]

                # Compute chunking like AutoRIFT does:
                # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
                chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
                two_dim_chunks_settings = (chunk_lines, num_x)

                granule_encoding = mission_encoding[ProcessV2Granules.MISSION].copy()

                for each_var, each_var_settings in granule_encoding.items():
                    if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
                        each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

                # Get encoding for the file
                cropped_ds.to_netcdf(fixed_file, engine='h5netcdf', encoding=granule_encoding)

                target_prefix = point_to_prefix(ProcessV2Granules.TARGET_DIR, center_lon_lat[1], center_lon_lat[0])

                # Upload corrected granule to the bucket - format sub-directory based on new cropped values
                s3_client = boto3.client('s3')
                try:
                    bucket_granule = os.path.join(target_prefix, granule_basename)
                    msgs.append(f"Uploading to {target_prefix}")

                    s3_client.upload_file(fixed_file, ProcessV2Granules.BUCKET, bucket_granule)

                    # msgs.append(f"Removing local {fixed_file}")
                    os.unlink(fixed_file)

                    # Original granule in S3 bucket
                    source = granule_url.replace(ProcessV2Granules.BUCKET+'/', '')

                    bucket = boto3.resource('s3').Bucket(ProcessV2Granules.BUCKET)

                    # There are corresponding browse and thumbprint images to transfer
                    for target_ext in ['.png', '_thumb.png']:
                        target_key = bucket_granule.replace('.nc', target_ext)

                        source_key = source.replace('.nc', target_ext)

                        source_dict = {
                            'Bucket': ProcessV2Granules.BUCKET,
                            'Key': source_key
                        }

                        bucket.copy(source_dict, target_key)
                        msgs.append(f'Copying {target_ext} to s3')

                except ClientError as exc:
                    msgs.append(f"ERROR: {exc}")

        return msgs

    @staticmethod
    def copy(granule_url: str, s3: s3fs.S3FileSystem):
        """
        Copy the granule and corresponding PNG files to the target location in S3 bucket (without any modifications).
        """
        msgs = [f'Processing copy of {granule_url}']

        # Upload granule to the target directory in the bucket
        source = granule_url.replace(ProcessV2Granules.BUCKET+'/', '')
        target = source.replace(ProcessV2Granules.SOURCE_DIR, ProcessV2Granules.TARGET_DIR)

        bucket = boto3.resource('s3').Bucket(ProcessV2Granules.BUCKET)

        try:
            # There are corresponding browse and thumbprint images to transfer
            for target_ext in [None, '.png', '_thumb.png']:
                target_key = target
                source_key = source

                # It's an extra file to transfer, replace extension
                if target_ext is not None:
                    target_key = target_key.replace('.nc', target_ext)
                    source_key = source_key.replace('.nc', target_ext)

                source_dict = {
                    'Bucket': ProcessV2Granules.BUCKET,
                    'Key': source_key
                }

                bucket.copy(source_dict, target_key)
                msgs.append(f'Copying {source_key if target_ext is None else target_ext} to s3')

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
        default=100,
        help='Number of granules to process in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket that stores original ITS_LIVE V2 granules'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel2-latest',
        help='AWS S3 bucket and directory that store the granules'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel2/v02',
        help='AWS S3 bucket and directory that store the granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-sentinel2',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-glob', action='store',
        type=str,
        default='*/*.nc',
        help='Glob pattern for the granule search under "s3://bucket/bucket_dir/" [%(default)s]')

    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start-granule',
        type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '-m', '--mission',
        type=str,
        default='S2',
        help=f'Mission for the granules to crop. One of {list(mission_encoding.keys())}]'
    )
    parser.add_argument(
        '--zero_coverage_files',
        action='store_true',
        help=f'Copy *{ProcessV2Granules.ZERO_PERCENT_COVERAGE} granules to the target S3 bucket only. '
    )
    parser.add_argument(
        '--read_granule_list',
        action='store_true',
        help=f'Read granule file list from {ProcessV2Granules.GRANULE_LIST_FILE} stored in the target S3 bucket only (to avoid time consuming glob). '
    )
    parser.add_argument(
        '--store_granule_list',
        action='store_true',
        help=f'Collect granule files and store them to the {ProcessV2Granules.GRANULE_LIST_FILE} in the target S3 bucket. '
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    ProcessV2Granules.MISSION = args.mission
    ProcessV2Granules.CHUNK_SIZE = args.chunk_size
    ProcessV2Granules.DASK_WORKERS = args.dask_workers
    ProcessV2Granules.LOCAL_DIR = args.local_dir
    ProcessV2Granules.SOURCE_DIR = args.bucket_dir
    ProcessV2Granules.BUCKET = args.bucket
    ProcessV2Granules.TARGET_DIR = args.target_bucket_dir
    ProcessV2Granules.LOCAL_DIR = args.local_dir
    ProcessV2Granules.COPY_ZERO_PERCENT_COVERAGE_FILES = args.zero_coverage_files
    ProcessV2Granules.STORE_GRANULE_LIST_FILE = args.store_granule_list
    ProcessV2Granules.READ_GRANULE_LIST_FILE = args.read_granule_list

    process_granules = ProcessV2Granules(args.glob)
    process_granules(args.start_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
