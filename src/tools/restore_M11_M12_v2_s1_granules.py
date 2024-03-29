#!/usr/bin/env python
"""
Restore original M11 and M12 values for uncorrected V2 Sentinel-1 granules (that are residing in AWS S3 bucket):
1. Use original granules from s3://its-live-project/velocity_image_pair/sentinel1-latest
2. Crop original granule's M11 and M12 values to valid X/Y extends (since target granule is already cropped)
3. Copy M11 and M12 values into target granule, preserve encoding values
4. Copy granule with restored M11 and M12 into destination s3://its-live-data/velocity_image_pair/sentinel1-m-restored.

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
import s3fs
import xarray as xr

from itscube_types import DataVars, Coords, Output
from mission_info import Encoding

mission_encoding = Encoding.SENTINEL1


class RestoreM11M12Values:
    """
    Restore original M11 and M12 values for uncorrected V2 Sentinel-1 granules (that are residing in AWS S3 bucket):
    1. Use original granules from s3://its-live-project/velocity_image_pair/sentinel1-latest
    2. Crop original granule's M11 and M12 values to valid X/Y extends (since target granule is already cropped)
    3. Copy granule with restored M11 and M12 into destination s3://its-live-data/velocity_image_pair/sentinel1.

    Ask Alex: Skip the granules that end with "_P000.nc" as those don't have any data to begin with.
    """
    ZERO_PERCENT_COVERAGE = '_P000.nc'

    # Flag to restore M11 and M12 for the zero percent coverage files (they don't need to be cropped) and copy
    # related PNG files to the target S3 bucket.
    ZERO_PERCENT_COVERAGE_FILES = False

    # S3 bucket with original granules
    BUCKET = 'its-live-project'

    # Source S3 bucket directory
    SOURCE_DIR = 'velocity_image_pair/sentinel1-latest/v02'

    # S3 bucket to place granules with restored M11 and M12
    TARGET_BUCKET = 'its-live-data'

    # Target S3 bucket directory
    TARGET_DIR = None

    # Destination S3 bucket directory for restored data
    RESTORED_DATA_BUCKET_DIR = None

    # Local directory to store cropped granules before copying them to the S3 bucket
    LOCAL_DIR = 'sandbox'

    # Number of granules to process in parallel
    CHUNK_SIZE = 100

    # Number of Dask workers for parallel processing
    DASK_WORKERS = 8

    STORE_GRANULE_LIST_FILE = False
    READ_GRANULE_LIST_FILE = False
    GRANULE_LIST_FILE = 'used_granules.json'

    def __init__(self, glob_pattern: dir, granules_to_exclude: str):
        """
        Initialize object.

        glob_pattern: Glob pattern to use to collect existing granules.
        granules_to_exclude: Filename with granules to exclude from processing. Some of the granules
            require extra processing, and should be handled separately.
        """
        self.s3 = s3fs.S3FileSystem()

        # Read JSON format file that lists granules for correction - skip those granules as M11 and M12
        # will be restored for those granules during correction.
        # Original file contains "old" paths, so need to replace them with current location of
        # velocity_image_pair/sentinel1/v02/N50W130/S1A_IW_SLC__1SDV_20170303T025410_20170303T025438_015522_0197FC_9167_X_S1A_IW_SLC__1SDV_20170327T025410_20170327T025438_015872_01A27F_DE8E_G0120V02_P072.nc
        self.exclude_granules = []
        with open(granules_to_exclude, 'r') as fh:
            self.exclude_granules = json.load(fh)
            logging.info(f'Got {len(self.exclude_granules)} from {granules_to_exclude}')

            # Replace originally recorded s3 paths with current location (moved due to s3 bucket move and introduction of s3://its-live-project)
            # self.exclude_granules = [
            #     os.path.join(RestoreM11M12Values.BUCKET, each.replace('velocity_image_pair/sentinel1/v02', RestoreM11M12Values.SOURCE_DIR)) for each in self.exclude_granules
            # ]

            # List of target granules as they exist in the target s3 bucket
            self.exclude_granules = [os.path.join(RestoreM11M12Values.TARGET_BUCKET, each) for each in self.exclude_granules]
            logging.info(f"Number of granules to exclude: {len(self.exclude_granules)}")

        self.all_target_granules = []
        self.all_original_granules = []

        # Store listings of original and target granules to the files in case we want to store/load granules lists
        # for subsequent runs
        self.target_granule_filename = os.path.join(
            RestoreM11M12Values.TARGET_BUCKET,
            RestoreM11M12Values.TARGET_DIR,
            RestoreM11M12Values.GRANULE_LIST_FILE
        )

        self.original_granule_filename = os.path.join(
            RestoreM11M12Values.BUCKET,
            RestoreM11M12Values.SOURCE_DIR,
            RestoreM11M12Values.GRANULE_LIST_FILE
        )

        # Use glob to list directory or read the listings from already existing file
        if RestoreM11M12Values.READ_GRANULE_LIST_FILE:
            with self.s3.open(self.target_granule_filename, 'r') as ins3file:
                self.all_target_granules = json.load(ins3file)
                logging.info(f"Loaded {len(self.all_target_granules)} target granules from '{self.target_granule_filename}'")

            with self.s3.open(self.original_granule_filename, 'r') as ins3file:
                self.all_original_granules = json.load(ins3file)
                logging.info(f"Loaded {len(self.all_original_granules)} original granules from '{self.original_granule_filename}'")

        else:
            s3_dir = os.path.join(
                RestoreM11M12Values.TARGET_BUCKET,
                RestoreM11M12Values.TARGET_DIR
            )
            logging.info(f"Reading {s3_dir}")
            self.all_target_granules = self.s3.glob(f'{s3_dir}/{glob_pattern}')

            s3_dir = os.path.join(
                RestoreM11M12Values.BUCKET,
                RestoreM11M12Values.SOURCE_DIR
            )
            logging.info(f"Reading {s3_dir}")
            self.all_original_granules = self.s3.glob(f'{s3_dir}/{glob_pattern}')

        logging.info(f"Number of target granules: {len(self.all_target_granules)}")
        logging.info(f"Number of original granules: {len(self.all_original_granules)}")

        if RestoreM11M12Values.STORE_GRANULE_LIST_FILE:
            # Store the granule list to the file in S3 target directory
            with self.s3.open(self.target_granule_filename, 'w') as outs3file:
                geojson.dump(self.all_target_granules, outs3file)
                logging.info(f'Stored granule list to {self.target_granule_filename}')

            with self.s3.open(self.original_granule_filename, 'w') as outs3file:
                geojson.dump(self.all_original_granules, outs3file)
                logging.info(f'Stored granule list to {self.original_granule_filename}')

        # To be consistent, restore M11 and M12 values and their encoding attributes for the zero coverage granules
        if RestoreM11M12Values.ZERO_PERCENT_COVERAGE_FILES is True:
            logging.info('Will process only zero coverage granules...')

            # Restore M11 and M12 for zero percent coverage files and copy them over to the target bucket as they do not
            # need to be cropped
            self.all_target_granules = [
                each for each in self.all_target_granules if each.endswith(RestoreM11M12Values.ZERO_PERCENT_COVERAGE)
            ]

            self.all_original_granules = [
                each for each in self.all_original_granules if each.endswith(RestoreM11M12Values.ZERO_PERCENT_COVERAGE)
            ]

            logging.info(f'Leaving zero coverage granules for target granules ({len(self.all_target_granules)}) and original granules ({len(self.all_original_granules)})')

        else:
            logging.info('Excluding zero coverage granules...')

            # Don't handle granules that have 0% coverage - only update v.description attribute for those
            self.all_target_granules = [
                each for each in self.all_target_granules if not each.endswith(RestoreM11M12Values.ZERO_PERCENT_COVERAGE)
            ]

            self.all_original_granules = [
                each for each in self.all_original_granules if not each.endswith(RestoreM11M12Values.ZERO_PERCENT_COVERAGE)
            ]

            logging.info(f'Leaving non-zero coverage granules for target granules ({len(self.all_target_granules)}) and original granules ({len(self.all_original_granules)})')

        # Remove corrected granules from the jobs to process
        if len(self.exclude_granules):
            self.all_target_granules = list(set(self.all_target_granules).difference(self.exclude_granules))
            logging.info(f'Excluding corrected granules, leaving number of target granules: {len(self.all_target_granules)} and original granules: {len(self.all_original_granules)} ')

        # Guarantee the same order of granules in case we have to pick up from where previous run stopped
        self.all_target_granules.sort()

        # For debugging only: convert first granule
        # self.all_target_granules = self.all_target_granules[:1]

        logging.info(f"Number of granules to process: {len(self.all_target_granules)}")

    def find_original_granule(self, granule_url: str):
        """Given target granule URL find corresponding original granule in s3://its-live-project bucket. Granule sub-directories
            can be different due to the fact that target granules were cropped to valid X/Y extends and could end up in
            different "10x10 degree grid" subdirectory.

        Args:
            granule_url (str): Target granule URL

        Returns:
            Original granule URL that corresponds to the target URL.
        """
        granule_url_basename = os.path.basename(granule_url)

        found_granule = [each for each in self.all_original_granules if os.path.basename(each) == granule_url_basename]
        if len(found_granule) == 0:
            raise RuntimeError(f'Could not find original granule that corresponds to {granule_url}')

        if len(found_granule) > 1:
            # Should never happen, just to be sure
            raise RuntimeError(f'More than one granule found that corresponds to {granule_url}: {found_granule}')

        return found_granule[0]

    def __call__(self, start_index: int):
        """
        Crop M11 and M12 values from original V2 granules to X/Y extend of non-None values and copy the values
        to already corrected "target" granule.
        """
        num_to_fix = len(self.all_target_granules) - start_index

        start = start_index
        logging.info(f"{num_to_fix} granules to fix...")

        if num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(RestoreM11M12Values.LOCAL_DIR):
            os.mkdir(RestoreM11M12Values.LOCAL_DIR)

        while num_to_fix > 0:
            num_tasks = RestoreM11M12Values.CHUNK_SIZE if num_to_fix > RestoreM11M12Values.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [
                dask.delayed(RestoreM11M12Values.crop_and_restore)(
                    each,
                    self.find_original_granule(each),
                    each.replace(RestoreM11M12Values.TARGET_DIR, RestoreM11M12Values.RESTORED_DATA_BUCKET_DIR),
                    self.s3,
                    RestoreM11M12Values.ZERO_PERCENT_COVERAGE_FILES
                ) for each in self.all_target_granules[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=RestoreM11M12Values.DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def crop_and_restore(
        granule_url: str,
        original_granule_url: str,
        restored_granule_url: str,
        s3: s3fs.S3FileSystem,
        restore_only: bool

    ):
        """
        Crop corrsponding to "granule_url" original granule to X/Y extend of non-None "v" values,
        and restore M11 and M12 values in target granule (which has all metadata already fixed).
        Store new granule in NetCDF format to the local directory, and push new granule
        to the target location in S3 bucket along with corresponding *png files.
        If "restore_only" flag is set to True, it means that processing is done for zero coverage granules.
        Zero coverage granules were not cropped to the valid data X/Y extends and cropping should be
        skipped when restoring M11 and M12 data.

        Args:
            granule_url (str): Target URL to correct.
            original_granule_url (str): Original granule that has correct M11 and M12 values.
            restored_granule_url (str): S3 bucket fullpath filename to store corrected granule to.
            s3 (s3fs.S3FileSystem): s3fs object to access target granules stored in s3 bucket.
            s3_original (s3fs.S3FileSystem): s3fs object to access original granules stored in s3 bucket.
            restore_only (bool): Flag if cropping of the granule should be skipped when restoring M11 and M12
                (applies to zero coverage granules only).
        """
        msgs = [f'Processing {granule_url}']

        msgs.append(f'Opening original {original_granule_url}...')

        ds_target = None
        with s3.open(granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds_target:
                # Load target granule that needs to restore M11 and M12
                ds_target = ds_target.load()

        # Write the granule locally, upload it to the bucket, remove file
        granule_basename = os.path.basename(granule_url)
        fixed_file = os.path.join(RestoreM11M12Values.LOCAL_DIR, granule_basename)

        with s3.open(original_granule_url) as fhandle:
            with xr.open_dataset(fhandle) as ds:
                cropped_ds = ds

                if not restore_only:
                    # Have to crop source data to valid X/Y extend

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

                    cropped_ds = ds.where(mask, drop=True)

                    # We are interested in M11 and M12 only
                    cropped_ds = cropped_ds[[DataVars.M11, DataVars.M12]].load()

                # Update target granule with original M11 and M12 values
                ds_target[DataVars.M11] = cropped_ds[DataVars.M11]
                ds_target[DataVars.M12] = cropped_ds[DataVars.M12]

                # Add date when granule was updated
                ds_target.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

                # Save to local file

                # Set chunking for 2D data variables
                dims = cropped_ds.dims
                num_x = dims[Coords.X]
                num_y = dims[Coords.Y]

                # Compute chunking like AutoRIFT does:
                # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
                chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
                two_dim_chunks_settings = (chunk_lines, num_x)

                granule_encoding = Encoding.SENTINEL1.copy()

                for each_var, each_var_settings in granule_encoding.items():
                    if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
                        each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

                # Preserve encoding attributes for M11 and M12 per original granule
                for each_var in [DataVars.M11, DataVars.M12]:
                    if Output.SCALE_FACTOR not in ds[each_var].encoding:
                        msgs.append(f'ERROR: missing {each_var}:{Output.SCALE_FACTOR} encoding attribute for {original_granule_url}')
                        return msgs

                    if Output.ADD_OFFSET not in ds[each_var].encoding:
                        msgs.append(f'ERROR: missing {each_var}:{Output.ADD_OFFSET} encoding attribute for {original_granule_url}')
                        return msgs

                    granule_encoding[each_var][Output.SCALE_FACTOR] = ds[each_var].encoding[Output.SCALE_FACTOR]
                    granule_encoding[each_var][Output.ADD_OFFSET] = ds[each_var].encoding[Output.ADD_OFFSET]

                # Save to local file
                ds_target.to_netcdf(fixed_file, engine='h5netcdf', encoding=granule_encoding)

        # Upload corrected granule to the bucket - format sub-directory based on new cropped values
        s3_client = boto3.client('s3')
        try:
            # New location for corrected granule
            bucket_granule = restored_granule_url.replace(RestoreM11M12Values.TARGET_BUCKET + '/', '')
            msgs.append(f"Uploading {RestoreM11M12Values.TARGET_BUCKET}/{bucket_granule}")

            s3_client.upload_file(fixed_file, RestoreM11M12Values.TARGET_BUCKET, bucket_granule)

            # msgs.append(f"Removing local {fixed_file}")
            os.unlink(fixed_file)

            # Original granule in S3 bucket
            source = granule_url.replace(RestoreM11M12Values.TARGET_BUCKET+'/', '')

            bucket = boto3.resource('s3').Bucket(RestoreM11M12Values.TARGET_BUCKET)

            # There are corresponding browse and thumbprint images to transfer
            for target_ext in ['.png', '_thumb.png']:
                target_key = bucket_granule.replace('.nc', target_ext)

                source_key = source.replace('.nc', target_ext)

                source_dict = {
                    'Bucket': RestoreM11M12Values.TARGET_BUCKET,
                    'Key': source_key
                }
                # msgs.append(f'Uploading {source_dict} to {target_key}')
                bucket.copy(source_dict, target_key)
                msgs.append(f'Copying {target_ext} to s3')

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
        default='its-live-project',
        help='AWS S3 bucket that stores original ITS_LIVE V2 granules'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1-latest/v02',
        help='AWS directory that store original granules'
    )
    parser.add_argument(
        '--target_bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket that stores target granules to restore M11/M12 data for'
    )
    parser.add_argument(
        '--target_bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1/v02',
        help='AWS S3 directory that store target granules'
    )
    parser.add_argument(
        '--restored_data_bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1-restoredM/v02',
        help='AWS S3 directory that store target granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-s1',
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
        '--read_granule_list',
        action='store_true',
        help=f'Read granule file list from {RestoreM11M12Values.GRANULE_LIST_FILE} stored in the target S3 bucket only (to avoid time consuming glob). '
    )
    parser.add_argument(
        '--store_granule_list',
        action='store_true',
        help=f'Collect granule files and store them to the {RestoreM11M12Values.GRANULE_LIST_FILE} in the target S3 bucket. '
    )
    parser.add_argument(
        '--zero_coverage_files',
        action='store_true',
        help=f'Process *{RestoreM11M12Values.ZERO_PERCENT_COVERAGE} granules only.'
    )
    parser.add_argument(
        '--skip_granules',
        default='S1_to_correct.json',
        help='JSON file with granules to exclude from processing [%(default)s]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    RestoreM11M12Values.CHUNK_SIZE = args.chunk_size
    RestoreM11M12Values.DASK_WORKERS = args.dask_workers
    RestoreM11M12Values.LOCAL_DIR = args.local_dir
    RestoreM11M12Values.SOURCE_DIR = args.bucket_dir
    RestoreM11M12Values.BUCKET = args.bucket
    RestoreM11M12Values.TARGET_BUCKET = args.target_bucket
    RestoreM11M12Values.TARGET_DIR = args.target_bucket_dir
    RestoreM11M12Values.RESTORED_DATA_BUCKET_DIR = args.restored_data_bucket_dir
    RestoreM11M12Values.LOCAL_DIR = args.local_dir
    RestoreM11M12Values.ZERO_PERCENT_COVERAGE_FILES = args.zero_coverage_files
    RestoreM11M12Values.STORE_GRANULE_LIST_FILE = args.store_granule_list
    RestoreM11M12Values.READ_GRANULE_LIST_FILE = args.read_granule_list

    process_granules = RestoreM11M12Values(args.glob, args.skip_granules)
    process_granules(args.start_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
