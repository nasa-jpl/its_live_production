#!/usr/bin/env python
"""
Apply correction to the Vx, Vy, V, M11, and M12 data of the Sentinel1 V2 data.

Reference table, as provided by Joe, contains the following information per each row:
reference        S1A_IW_SLC__1SDV_20170203T162106_20170203T162132_015121_018B9A_1380
secondary        S1A_IW_SLC__1SDV_20170215T162105_20170215T162133_015296_019127_6E1E
v2_s3_bucket     its-live-data
v2_s3_key        velocity_image_pair/sentinel1/v02/N00E020/S1A_IW_SLC__1SDV_20170203T162106_20170203T162132_015121_018B9A_1380_X_S1A_IW_SLC__1SDV_20170215T162105_20170215T162133_015296_019127_6E1E_G0120V02_P099.nc
status           SUCCEEDED
to_correct       False
cor_s3_bucket    hyp3-its-live-contentbucket-s10lg85g5sp4
job_id           a851f0b5-9ae2-43fe-b81d-344eeae67b79

where the correction tifs for a particular file will all be in s3://[cor_s3_bucket]/[job_id]/

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Yang Lei, Alex Gardner, Joe Kennedy
"""
import argparse
import boto3
from datetime import datetime
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
import logging
import numpy as np
import os
import pandas as pd
import rioxarray as rxr
import s3fs
import xarray as xr

from itscube_types import Coords, Output
from mission_info import Encoding

NoDataValue = -32767


def float_to_int(data, nan_mask):
    """
    Replace Nan's with NoDataValue before converting data to np.int16 to avoid Nan->0 replacement
    """
    masked_data = np.where(nan_mask, NoDataValue, data)
    int_data = np.round(np.clip(masked_data, -32768, 32767)).astype(np.int16)

    return int_data


class CorrectV2Granules:
    """
    Apply correction to existing V2 Sentinel1 granules, copy corrected granule and corresponding PNG files
    to the target location in AWS S3 bucket.
    """
    NC_ENGINE = 'h5netcdf'

    # Reference TIF files required for correction
    VX_TIFF_FILE = 'window_rdr_off2vel_x_vec.tif'
    VY_TIFF_FILE = 'window_rdr_off2vel_y_vec.tif'
    WINDOW_TIFF_FILE = 'window_scale_factor.tif'

    # S3 bucket with granules
    BUCKET = 'its-live-data'

    # Source S3 bucket directory
    SOURCE_DIR = 'velocity_image_pair/sentinel1/v02'

    # Target S3 bucket directory
    TARGET_DIR = None

    # Local directory to store corrected granules before copying them to the S3 bucket
    LOCAL_DIR = 'sandbox-correct-S1'

    # Number of granules to process in parallel
    CHUNK_SIZE = 100

    # Number of Dask workers for parallel processing
    DASK_WORKERS = 8

    def __init__(self, granule_table: str):
        """
        Initialize object.

        Inputs:
        =======
        granule_table: File that stores information of granule correction.
        """
        self.s3 = s3fs.S3FileSystem()

        # Use other s3fs object to read reference TIF data
        self.s3_ref = s3fs.S3FileSystem()

        self.table = pd.read_parquet(granule_table)

        # Keep rows with granules that need to be corrected
        self.table = self.table.loc[self.table['to_correct'] == True]

        logging.info(f"Total number of granules to correct: {len(self.table)}")

    def __call__(self, start_index: int = 0, stop_index: int = 0):
        """
        Correct V2 S1 granules and copy corrected files to the target location in S3 bucket.
        """
        num_to_fix = len(self.table)

        if stop_index > 0:
            num_to_fix = stop_index

        num_to_fix -= start_index

        # For debugging only
        # num_to_fix = 3

        start = start_index
        logging.info(f"{num_to_fix} granules to correct...")

        if num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not os.path.exists(CorrectV2Granules.LOCAL_DIR):
            os.mkdir(CorrectV2Granules.LOCAL_DIR)

        while num_to_fix > 0:
            num_tasks = CorrectV2Granules.CHUNK_SIZE if num_to_fix > CorrectV2Granules.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")

            tasks = [
                dask.delayed(CorrectV2Granules.correct)(
                    each['v2_s3_bucket'],
                    each['v2_s3_key'],
                    each['cor_s3_bucket'],
                    each['job_id'],
                    self.s3,
                    self.s3_ref,
                ) for _, each in self.table.iloc[start:start+num_tasks].iterrows()
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=CorrectV2Granules.DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def correct(
        bucket: str,
        granule_path: str,
        ref_bucket: str,
        ref_dir: str,
        s3: s3fs.S3FileSystem,
        s3_ref: s3fs.S3FileSystem
    ):
        """
        Correct S1 data for the granule residing in S3 bucket. Copy corrected granule to the new
        S3 location.

        Inputs:
        =======
        bucket: S3 bucket holding granule for correction.
        granule_path: Granule path within S3 bucket.
        ref_bucket: S3 bucket with reference data for correction.
        ref_dir: S3 directory path that holds reference data for correction.
        s3: s3fs.S3FileSystem object to access data for correction.
        """
        msgs = [f'Processing {granule_path}']

        # Read granule to correct
        ds = None

        # Read the granule in
        with s3.open(os.path.join(bucket, granule_path), mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=CorrectV2Granules.NC_ENGINE) as ds:
                ds = ds.load()

        # Read bands from reference TIF files

        # Initialize mask to correspond to the granule's X/Y extends to crop reference TIF files to
        mask = None

        cropped_vx_tiff = None
        AA = None
        BB = None
        di_to_vr_factor = None

        with s3_ref.open(os.path.join(ref_bucket, ref_dir, CorrectV2Granules.VX_TIFF_FILE), mode='rb') as fhandle:
            with rxr.open_rasterio(fhandle, mask_and_scale=True) as vx_tiff:
                # Crop TIFs to granule's X/Y ranges
                mask_x = (vx_tiff.x >= ds.x.min().item()) & (vx_tiff.x <= ds.x.max().item())
                mask_y = (vx_tiff.y >= ds.y.min().item()) & (vx_tiff.y <= ds.y.max().item())
                mask = (mask_x & mask_y)

                cropped_vx_tiff = vx_tiff.where(mask, drop=True)

                # Get parameters from TIFs
                AA = cropped_vx_tiff[0].values
                BB = cropped_vx_tiff[1].values
                di_to_vr_factor = cropped_vx_tiff[2].values

        cropped_vy_tiff = None
        CC = None
        DD = None
        dj_to_va_factor = None

        with s3_ref.open(os.path.join(ref_bucket, ref_dir, CorrectV2Granules.VY_TIFF_FILE), mode='rb') as fhandle:
            with rxr.open_rasterio(fhandle) as vy_tiff:
                cropped_vy_tiff = vy_tiff.where(mask, drop=True)

                # Get parameters from TIFs
                CC = cropped_vy_tiff[0].values
                DD = cropped_vy_tiff[1].values
                dj_to_va_factor = cropped_vy_tiff[2].values

        cropped_window_tiff = None
        scalefactor_di = None
        scalefactor_dj = None

        with s3_ref.open(os.path.join(ref_bucket, ref_dir, CorrectV2Granules.WINDOW_TIFF_FILE), mode='rb') as fhandle:
            with rxr.open_rasterio(fhandle, mask_and_scale=True) as window_tiff:
                cropped_window_tiff = window_tiff.where(mask, drop=True)

                # Get parameters from TIFs
                scalefactor_di = cropped_window_tiff[0].values  # ai
                scalefactor_dj = cropped_window_tiff[1].values  # aj

        # This is constant value as dj_to_va_factor
        # dr_to_vr_factor = ds.M11.attrs['dr_to_vr_factor']

        di = ds.vr.values / di_to_vr_factor
        dj = ds.va.values / dj_to_va_factor

        # Compute new vx and vy
        vx = AA * di * scalefactor_di + BB * dj * scalefactor_dj
        vy = CC * di * scalefactor_di + DD * dj * scalefactor_dj

        v = np.sqrt(vx**2 + vy**2)

        # Compute new M matrix (M1* elements)
        M11 = DD / (AA * DD - BB * CC) / scalefactor_di
        M12 = -BB / (AA * DD - BB * CC) / scalefactor_di

        # Convert corrected data to int when storing to the file
        nan_mask = np.isnan(vx) | np.isnan(vy)

        # Update granule to include corrected data
        ds['vx'] = xr.DataArray(
            data=float_to_int(vx, nan_mask),
            coords=ds.vx.coords,
            dims=ds.vx.dims,
            attrs=ds.vx.attrs
        )

        ds['vy'] = xr.DataArray(
            data=float_to_int(vy, nan_mask),
            coords=ds.vy.coords,
            dims=ds.vy.dims,
            attrs=ds.vy.attrs
        )

        ds['v'] = xr.DataArray(
            data=float_to_int(v, nan_mask),
            coords=ds.v.coords,
            dims=ds.v.dims,
            attrs=ds.v.attrs
        )

        ds['M11'] = xr.DataArray(
            data=float_to_int(M11, nan_mask),
            coords=ds.M11.coords,
            dims=ds.M11.dims,
            attrs=ds.M11.attrs
        )

        ds['M12'] = xr.DataArray(
            data=float_to_int(M12, nan_mask),
            coords=ds.M12.coords,
            dims=ds.M12.dims,
            attrs=ds.M12.attrs
        )

        # Set the date when the granule was updated
        ds.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

        # Set chunking for 2D data variables
        dims = ds.dims
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

        path_tokens = granule_path.split('/')

        # Use immediate parent directory as a local filename
        fixed_file = os.path.join(CorrectV2Granules.LOCAL_DIR, f'{path_tokens[-2]}_{path_tokens[-1]}')

        msgs.append(f'Writing corrected granule to {fixed_file}')
        ds.to_netcdf(fixed_file, engine='h5netcdf', encoding=granule_encoding)

        # Upload corrected granule to the bucket - format sub-directory based on new cropped values
        s3_client = boto3.client('s3')
        try:
            # Upload granule to the target directory in the bucket
            target = granule_path.replace(CorrectV2Granules.SOURCE_DIR, CorrectV2Granules.TARGET_DIR)

            bucket = boto3.resource('s3').Bucket(CorrectV2Granules.BUCKET)

            msgs.append(f"Uploading to {target}")

            s3_client.upload_file(fixed_file, CorrectV2Granules.BUCKET, target)

            # msgs.append(f"Removing local {fixed_file}")
            os.unlink(fixed_file)

            # There are corresponding browse and thumbprint images to transfer
            bucket = boto3.resource('s3').Bucket(CorrectV2Granules.BUCKET)

            for target_ext in ['.png', '_thumb.png']:
                target_key = target.replace('.nc', target_ext)

                source_key = granule_path.replace('.nc', target_ext)

                source_dict = {
                    'Bucket': CorrectV2Granules.BUCKET,
                    'Key': source_key
                }

                bucket.copy(source_dict, target_key)
                msgs.append(f'Copying {target_ext} to s3')

        except ClientError as exc:
            msgs.append(f"ERROR: {exc}")

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[1],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size',
        type=int,
        default=100,
        help='Number of granules to process in parallel [%(default)d]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1-corrected/v02',
        help='AWS S3 bucket and directory to store corrected granules'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-sentinel1',
        help='Directory to store fixed granules before uploading them to the S3 bucket'
    )
    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '-s', '--start_granule',
        type=int,
        default=0,
        help='Index for the start granule to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '-e', '--stop_granule',
        type=int,
        default=0,
        help='Index for the last granule to process (if splitting processing across multiple EC2s) [%(default)d]'
    )
    parser.add_argument(
        '--granule_table',
        # default='fix_s1_v2_granules/correction_table/sentinel-1-correction-files.parquet',
        default='sentinel-1-correction-files.parquet',
        help='Table that provides the reference data for the granules to correct [%(default)s]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    CorrectV2Granules.CHUNK_SIZE = args.chunk_size
    CorrectV2Granules.DASK_WORKERS = args.dask_workers
    CorrectV2Granules.TARGET_DIR = args.target_bucket_dir
    CorrectV2Granules.LOCAL_DIR = args.local_dir

    process_granules = CorrectV2Granules(args.granule_table)
    process_granules(args.start_granule, args.stop_granule)


if __name__ == '__main__':
    main()

    logging.info("Done.")
