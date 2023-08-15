#!/usr/bin/env python
"""
Correct percent coverage and crop X/Y range that covers only valid data for L7 ITS_LIVE granules residing in AWS S3 bucket.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Mark Fahnestock, Alex Gardner
"""
import argparse
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
import glob
import logging
import numpy as np
import os
import pandas as pd
import pyproj
import s3fs
import xarray as xr

from itscube_types import DataVars, Coords, Output
from mission_info import Encoding
from lon_lat_to_dir_prefix import point_to_prefix


class ProcessV2Granules:
    """
    Correct percent coverage for L7 ITS_LIVE granules.
    """
    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = 'EPSG:4326'

    # S3 bucket with granules
    BUCKET = 'its-live-data'

    HTTP_BUCKET_URL = 'https://its-live-data.s3.us-west-2.amazonaws.com'

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

    # Disable dry run and copy all corrected files to the target location in S3 bucket
    DRYRUN = False

    def __init__(self, new_p_values_dir: str, glob_pattern: str):
        """
        Initialize object.

        glob_pattern: Glob pattern to use to collect files with new_pvalue.
        """
        self.s3 = s3fs.S3FileSystem()

        self.processed_granules = []

        self.new_pvalue_files = sorted(glob.glob(os.path.join(new_p_values_dir, glob_pattern)))
        logging.info(f'Got {len(self.new_pvalue_files)} files with new pvalue to process')

    def __call__(self, start_index: int = 0, start_granule_first_file: int = 0):
        """
        Correct percent value for the granules. Copy file and corresponding PNG files into
        specified location in S3 bucket.

        start_index: Index into file list to resume processing with. Default is 0.
        """
        num_files = len(self.new_pvalue_files) - start_index
        index = start_index
        logging.info(f"{num_files} files to process...")

        if num_files == 0:
            logging.info("Nothing to fix, exiting.")
            return

        # Create local directory to store corrected files to before copying them to S3 bucket
        if not os.path.exists(ProcessV2Granules.LOCAL_DIR):
            os.mkdir(ProcessV2Granules.LOCAL_DIR)

        num_files_to_process = num_files

        while num_files_to_process > 0:
            current_file = self.new_pvalue_files[index]
            file_data = pd.read_csv(current_file, header=None, delimiter=r"\s+")

            granules_to_fix = len(file_data)
            logging.info(f'Got {index}: {current_file} {granules_to_fix} total granules')
            start_granule = 0

            # Reset start granule for the first processed file only
            if num_files_to_process == num_files:
                start_granule = start_granule_first_file
                granules_to_fix -= start_granule_first_file

            logging.info(f'Processing {index}: {current_file} {granules_to_fix} remaining granules')

            # Process granules within the file
            while granules_to_fix > 0:
                num_tasks = ProcessV2Granules.CHUNK_SIZE if granules_to_fix > ProcessV2Granules.CHUNK_SIZE else granules_to_fix

                logging.info(f"Starting tasks {start_granule}:{start_granule+num_tasks} ({granules_to_fix} left)")

                tasks = [
                    dask.delayed(ProcessV2Granules.correct)(
                        each[0],
                        each[2],
                        each[4],
                        self.s3,
                    ) for _, each in file_data[start_granule:start_granule+num_tasks].iterrows()
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

                granules_to_fix -= num_tasks
                start_granule += num_tasks

            # Update for next file to process
            index += 1
            num_files_to_process -= 1

    @staticmethod
    def correct(granule_url: str, new_pvalue: float, v1_pvalue: float, s3: s3fs.S3FileSystem):
        """
        1. Correct percent valid for the granule in NetCDF format to the local directory:
        * Rename granule
        * Rename corresponding PNG files
        * Update img_pair_info.roi_valid_percentage to int(new_pvalue)
        * Push new granule to the target location in S3 bucket.
        * Copy original PNG files to the target location in S3 bucket using new filenames

        2. Crop granule's X/Y grid only to the valid data extends

        Inputs:
        granule_url - HTTPS URL for the granule to correct.
        new_pvalue - Newly computed percent valid value.
        v1_pvalue - V1 percent valid value.
        s3 - s3fs.S3FileSystem object to access granules in AWS S3 bucket.
        """
        msgs = [f'Processing {granule_url} new_pvalue={new_pvalue} v1_pvalue={v1_pvalue}']

        s3_granule_url = granule_url.replace(ProcessV2Granules.HTTP_BUCKET_URL, ProcessV2Granules.BUCKET)

        open_attempts = 0
        done = False

        while open_attempts < 3 and not done:
            open_attempts += 1

            try:
                with s3.open(s3_granule_url) as fhandle:
                    with xr.open_dataset(fhandle) as ds:
                        # this will drop X/Y coordinates, so drop non-None values just to get X/Y extends
                        xy_ds = ds.where(ds.v.notnull(), drop=True)

                        x_values = xy_ds.x.values
                        y_values = xy_ds.y.values

                        cropped_ds = None
                        center_lon_lat = None

                        if len(x_values) <= 1 or len(y_values) <= 1:
                            msgs.append(f'WARNING Skipping cropping due to invalid X or Y length (len(x) = {len(x_values)}, len(y) = {len(y_values)}) for {granule_url} new_pvalue={new_pvalue} v1_pvalue={v1_pvalue}')
                            cropped_ds = ds.load()

                            # Use current centroid information for the granule
                            center_lon_lat = (
                                float(cropped_ds[DataVars.ImgPairInfo.NAME].attrs['longitude']),
                                float(cropped_ds[DataVars.ImgPairInfo.NAME].attrs['latitude'])
                            )

                        else:
                            grid_x_min, grid_x_max = x_values.min(), x_values.max()
                            grid_y_min, grid_y_max = y_values.min(), y_values.max()

                            # Based on X/Y extends, mask original dataset
                            mask_lon = (ds.x >= grid_x_min) & (ds.x <= grid_x_max)
                            mask_lat = (ds.y >= grid_y_min) & (ds.y <= grid_y_max)
                            mask = (mask_lon & mask_lat)

                            # No need to check as we are changing coverage for non-P000 % coverage
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

                        # Add date when granule was updated
                        cropped_ds.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

                        # Update valid percent coverage
                        pvalue_to_use = float(new_pvalue) if new_pvalue <= v1_pvalue else float(v1_pvalue)
                        cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE] = pvalue_to_use

                        # Save to local file
                        granule_basename = os.path.basename(granule_url)

                        # Replace the pvalue in filename
                        granule_basename_tokens = granule_basename.split('_')
                        granule_basename_tokens[-1] = f'P{int(pvalue_to_use):03d}.nc'

                        new_granule_basename = '_'.join(granule_basename_tokens)

                        # Write the granule locally, upload it to the bucket, remove file
                        fixed_file = os.path.join(ProcessV2Granules.LOCAL_DIR, new_granule_basename)

                        # Set chunking for 2D data variables
                        dims = cropped_ds.dims
                        num_x = dims[Coords.X]
                        num_y = dims[Coords.Y]

                        # Compute chunking like AutoRIFT does:
                        # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
                        chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
                        two_dim_chunks_settings = (chunk_lines, num_x)

                        granule_encoding = Encoding.LANDSAT_SENTINEL2.copy()

                        for each_var, each_var_settings in granule_encoding.items():
                            if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
                                each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

                        # Get encoding for the file
                        cropped_ds.to_netcdf(fixed_file, engine='h5netcdf', encoding=granule_encoding)

                        target_prefix = point_to_prefix(ProcessV2Granules.TARGET_DIR, center_lon_lat[1], center_lon_lat[0])

                        # Upload corrected granule to the bucket - format sub-directory based on new cropped values
                        s3_client = boto3.client('s3')
                        try:
                            bucket_granule = os.path.join(target_prefix, new_granule_basename)
                            msgs.append(f"Uploading to {target_prefix}/{bucket_granule}")

                            if not ProcessV2Granules.DRYRUN:
                                s3_client.upload_file(fixed_file, ProcessV2Granules.BUCKET, bucket_granule)

                                # msgs.append(f"Removing local {fixed_file}")
                                os.unlink(fixed_file)

                            # Original granule in S3 bucket
                            source = s3_granule_url.replace(ProcessV2Granules.BUCKET+'/', '')

                            bucket = boto3.resource('s3').Bucket(ProcessV2Granules.BUCKET)

                            # There are corresponding browse and thumbprint images to transfer
                            for target_ext in ['.png', '_thumb.png']:
                                target_key = bucket_granule.replace('.nc', target_ext)

                                source_key = source.replace('.nc', target_ext)

                                source_dict = {
                                    'Bucket': ProcessV2Granules.BUCKET,
                                    'Key': source_key
                                }

                                msgs.append(f'Copying {target_ext} to S3')

                                if not ProcessV2Granules.DRYRUN:
                                    bucket.copy(source_dict, target_key)

                        except ClientError as exc:
                            msgs.append(f"ERROR: {exc}")

                        done = True
            except ValueError as exc:
                # Retry only network kind of exceptions accesing the file
                if 'h5netcdf' in exc:
                    msgs.append(f'Attempt #{open_attempts}: ValueError exception reading {granule_url}: {exc}')

                else:
                    msgs.append(f'ERROR: {exc}')
                    done = True

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        epilog="""
    Applied corrections:
        * Update img_pair_info.roi_valid_percentage to new value
        * Replace PXXX with PVVV (VVV is v1_pvalid) if new_pvalid >100 or new_pvalid > v1_pvalid
        * Place corrected version and corresponding PNG files into new location in S3 bucket.
    For example, for one of the lines from the text file:
    https://its-live-data.s3.us-west-2.amazonaws.com/velocity_image_pair/landsatOLI-latest/N70W020/LE07_L1TP_001005_20120713_20201020_02_T1_X_LC08_L1TP_001005_20130323_20200913_02_T1_G0120V02_P019.nc  new_pvalid 39.5  v1_pvalid 62
        1. Replace P019 with P039 in filename
        2. Update img_pair_info.roi_valid_percentage = 39.5
        *** If new_pvalid > v1_pvalid then replace PXXX with PVVV where VVV is the v1_pvalid.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help='AWS S3 bucket that stores original ITS_LIVE V2 granules [%(default)s]'
    )
    parser.add_argument(
        '-f', '--text_files',
        type=str,
        help='Directory that stores files with corrected p-valid values per granule'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/landsatOLI-latest',
        help='AWS S3 bucket and directory that store original granules [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='velocity_image_pair/landsatOLI/v02',
        help='AWS S3 bucket and directory that store corrected granules and corresponding PNG files [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-landsat',
        help='Directory to store fixed granules before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '--start_file',
        type=int,
        default=0,
        help='Index for the start file to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '--start_granule',
        type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '-glob', action='store',
        type=str,
        default='new_p_valid_fixes_*.txt',
        help='Glob pattern for the files with " [%(default)s]')
    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually copy any data to AWS S3 bucket [%(default)s]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    ProcessV2Granules.CHUNK_SIZE = args.chunk_size
    ProcessV2Granules.DASK_WORKERS = args.dask_workers
    ProcessV2Granules.LOCAL_DIR = args.local_dir
    ProcessV2Granules.SOURCE_DIR = args.bucket_dir
    ProcessV2Granules.BUCKET = args.bucket
    ProcessV2Granules.TARGET_DIR = args.target_bucket_dir
    ProcessV2Granules.DRYRUN = args.dryrun

    process_granules = ProcessV2Granules(args.text_files, args.glob)
    process_granules(args.start_file, args.start_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
