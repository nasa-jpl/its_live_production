#!/usr/bin/env python
"""
Restore NISAR M11 and M12 values and corresponding dr_to_vr_factor attributes
within existing ITS_LIVE datacubes that are residing in the AWS S3 bucket.

* Copy M11 and M12 values from original NISAR granules into corresponding
layers of the existing datacubes.
* Push corrected datacubes back to the S3 bucket by:
    ** Removing original M11 and M12 from s3 bucket as those are of int dtype.
    ** Copy cube's metadata files and newly restored M11/M12 data variables to
    the s3 cube's original location as np.float32 dtype.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Alex Gardner
"""
import argparse
import json
from pathlib import Path
import logging
import numpy as np
import os
import s3fs
import shutil
from joblib import Parallel, delayed, parallel_config
import xarray as xr
import zarr

from itscube_types import Vars, ImgPairInfo
from itslive_binary_type import BinaryFlag
import itslive_utils
import utils

class FixDatacubes:
    """
    Class to apply fixes to ITS_LIVE datacubes:

    * Copy M11 and M12 values from original NISAR granules into corresponding
    layers of the existing datacubes.

    * Push corrected datacubes back to the S3 bucket by:
    ** Removing original M11 and M12 from s3 bucket as those are of int dtype.
    ** Copy cube's metadata files and newly restored M11/M12 data variables to
    the s3 cube's original location as np.float32 dtype.
    """
    DRY_RUN = False

    def __init__(
        self, cubes_to_process, local_original_cube_dir: str, local_dir: str
    ):
        """
        Initialize object.

        Args:
            cubes_to_process: List of s3 urls for cubes to process.
            local_original_cube_dir (str): Local directory to store
                downloaded original datacubes to fix.
            local_dir (str): Local directory to save corrected cubes to.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        self.local_original_cube_dir = local_original_cube_dir
        self.local_dir = local_dir

        self.all_zarr_datacubes = cubes_to_process
        logging.info(f"Number of datacubes to process: {len(self.all_zarr_datacubes)}")

        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        if not os.path.exists(self.local_original_cube_dir):
            os.mkdir(self.local_original_cube_dir)


    def __call__(self, num_threads: int=8, start_cube: int = 0):
        """
        Restore M11 and M12 related data for the ITS_LIVE datacubes stored
        in S3 bucket.
        """
        num_to_process = len(self.all_zarr_datacubes) - start_cube
        start = start_cube

        logging.info(f"{num_to_process} datacubes to fix...")

        if num_to_process <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        with parallel_config(
            backend='threading',
            n_jobs=num_threads
        ):
            while num_to_process > 0:
                # How many tasks to process at a time
                num_tasks = min(num_to_process, num_threads)

                # Run in parallel with joblib
                log_msg = f"Processing {num_tasks} tasks out of " \
                            f"{num_to_process} remaining"
                logging.info(log_msg)

                results = Parallel()(
                    delayed(FixDatacubes.all)(
                        each,
                        self.local_original_cube_dir,
                        self.local_dir,
                        self.s3
                    ) for each in self.all_zarr_datacubes[start:start+num_tasks]
                )

                for each_result in results:
                    logging.info("\n-->".join(each_result))

                num_to_process -= num_tasks
                start += num_tasks

    @staticmethod
    def all(
        cube_url: str,
        local_original_cube_dir: str,
        local_dir: str,
        s3: s3fs.S3FileSystem
    ):
        """
        Fix M11 and M12 related data in datacubes and copy them to S3
        bucket's original location.
        """
        msgs = [f'Processing {cube_url}']

        cube_basename = os.path.basename(cube_url)

        # Copy datacube locally using AWS CLI to take advantage of parallel copy:
        # have to include "max_concurrent_requests" option for the
        # configuration in ~/.aws/config
        # [default]
        # region = us-west-2
        # output = json
        # s3 =
        #    max_concurrent_requests = 100
        #
        env_copy = os.environ.copy()

        local_original_cube = os.path.join(local_original_cube_dir, cube_basename)

        # If previous run already copied the cube, skip the copying from s3
        # which takes long time
        if not os.path.exists(local_original_cube):
            command_line = [
                "aws", "s3", "cp", "--recursive",
                cube_url,
                local_original_cube
            ]

            msgs.append(f"Creating local copy of {cube_url}: {local_original_cube}")
            msgs.append(' '.join(command_line))
            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, cube_basename)

        time_chunks = 250

        # Fill value for new data variables
        ascending_fill_value = Vars.intMissingValue[Vars.ascending_img1]

        with xr.open_dataset(local_original_cube, decode_timedelta=False,
                                engine='zarr', consolidated=True) as ds:
            msgs.append(f'Cube dimensions: {ds.dims}')

            x_values = ds.x.values
            grid_x_min, grid_x_max = x_values.min(), x_values.max()

            y_values = ds.y.values
            grid_y_min, grid_y_max = y_values.min(), y_values.max()

            # Identify RSLC layers within the cube - only those will have
            # M11/M12 data
            # 1. Build boolean mask over mid_date dimension
            #    granule_url is 1D: [mid_date], dtype U1024
            rslc_mask = np.char.find(
                ds['granule_url'].values.astype(str), 'RSLC'
            ) >= 0

            # Number of RSLC layers in the datacube
            num_rslc_layers = np.sum(rslc_mask)
            msgs.append(f'Identified {num_rslc_layers} RSLC layers in the cube')

            ascending_img1 = np.full((len(ds.mid_date)), ascending_fill_value, dtype=np.ubyte)
            ascending_img2 = np.full((len(ds.mid_date)), ascending_fill_value, dtype=np.ubyte)

            if num_rslc_layers:
                mask_i = np.where(rslc_mask == True)

                # Need to load all of M11/M12 data values in order to update
                # them. Otherwise it silently ignores values when updating
                # (xarray bug?)
                for each_var in [
                    Vars.m11, Vars.m12
                ]:
                    _ = ds[each_var].values

                    factor_var = f'{each_var}_{Vars.postfix.dr_to_vr_factor}'
                    _ = ds[factor_var].values

                # If there are no RSLC granules, nothing to do
                for each_index in mask_i[0]:
                    # Read URL of the granule. For example, granules paths will be in the format:
                    # https://its-live-data.s3.amazonaws.com/velocity_image_pair/sentinel1/v02/N70W060/S1A_IW_SLC__1SSH_20160728T113645_20160728T113712_012348_0133B2_74C0_X_S1A_IW_SLC__1SSH_20160809T113646_20160809T113713_012523_013989_2C50_G0120V02_P030.nc
                    granule = str(ds.granule_url[each_index].values)

                    each_granule_s3 = granule.replace('https://', '')
                    each_granule_s3 = each_granule_s3.replace('.s3.amazonaws.com', '')

                    # Open the granule
                    with s3.open(each_granule_s3, mode='rb') as fhandle:
                        with xr.open_dataset(fhandle, engine=utils.NC_ENGINE) as granule_ds:
                            granule_ds = granule_ds.load()

                            msgs.append(f'Granule for index={each_index}: {each_granule_s3};')

                            # Zoom into cube polygon
                            mask_x = (granule_ds.x >= grid_x_min) & (granule_ds.x <= grid_x_max)
                            mask_y = (granule_ds.y >= grid_y_min) & (granule_ds.y <= grid_y_max)
                            mask = (mask_x & mask_y)

                            cropped_ds = granule_ds.where(mask, drop=True)

                            # Restore values in the datacube
                            for each_var in [Vars.m11, Vars.m12]:
                                # # Show current values
                                # m_values = ds[each_var][each_index, :, :].values
                                # print(f'====>before assigning ds {each_var}: m_values.shape={m_values.shape} min={np.nanmin(m_values)} max={np.nanmax(m_values)}')

                                ds[each_var][each_index, :, :].loc[dict(x=cropped_ds.x, y=cropped_ds.y)] = \
                                    cropped_ds[each_var][0, :, :].drop_vars(utils.Coords.TIME)

                                # Restore corresponding dr_to_vr_factor attribute values
                                factor_var = f'{each_var}_{Vars.postfix.dr_to_vr_factor}'
                                ds[factor_var][each_index] = granule_ds[each_var].attrs[Vars.postfix.dr_to_vr_factor]

                                # # Show restored values
                                # m_values = ds[each_var][each_index, :, :].values
                                # print(f'====>assigned ds {each_var}: m_values.shape={m_values.shape} min={np.nanmin(m_values)} max={np.nanmax(m_values)}')

                        # Extract flight direction for both images of the granule
                        ascending_img1[each_index] = granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img1].strip() == ImgPairInfo.ascending
                        ascending_img2[each_index] = granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img2].strip() == ImgPairInfo.ascending

            new_coords = ds[ImgPairInfo.satellite_img1].coords
            new_dims = ds[ImgPairInfo.satellite_img1].dims

            # Add new variables to the datacube - just use existing 1-d data variable coords and dims
            ds[Vars.ascending_img1] = xr.DataArray(
                data=ascending_img1, coords=new_coords, dims=new_dims
            )
            ds[Vars.ascending_img1].attrs = {
                Vars.attrs.std_name: Vars.name[Vars.ascending_img1],
                Vars.attrs.description: Vars.description[Vars.ascending_img1],
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img1]
            }

            ds[Vars.ascending_img2] = xr.DataArray(
                data=ascending_img2, coords=new_coords, dims=new_dims
            )
            ds[Vars.ascending_img2].attrs = {
                Vars.attrs.std_name: Vars.name[Vars.ascending_img2],
                Vars.attrs.description: Vars.description[Vars.ascending_img2],
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img2]
            }

            # Apply chunking settings in the cube, use them as golden standard for all variables
            chunking_1d = (ds[ImgPairInfo.date_dt].encoding[utils.OutputFormat.chunks])
            chunking_2d = (len(ds.y), len(ds.x))
            chunking_3d = ds[Vars.chip_size_height].encoding[utils.OutputFormat.chunks]
            compression_zarr = zarr.Blosc(
                cname="lz4", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE
            )

            # Fix chunking for mid_date, ice masks, autoRIFT_software_version, granule_url,
            # and just to be sure - for x/y (set to the full extend already)
            for each_var in ds:
                if utils.OutputFormat.chunks in ds[each_var].encoding:
                    ds_chunking = ds[each_var].encoding[utils.OutputFormat.chunks]
                    chunking = ()

                    if len(ds_chunking) == 1:
                        chunking = chunking_1d

                    elif len(ds_chunking) == 2:
                        chunking = chunking_2d

                    elif len(ds_chunking) == 3:
                        chunking = chunking_3d

                    ds[each_var].encoding[utils.OutputFormat.chunks] = chunking

                    # Apply the same compression to all data variables
                    ds[each_var].encoding[utils.OutputFormat.compressor] = compression_zarr

            # Change datatype for M11 and M12 to floating point
            ds[Vars.m11].encoding[utils.OutputFormat.dtype] = np.float32
            ds[Vars.m12].encoding[utils.OutputFormat.dtype] = np.float32

            ds[Vars.m11].encoding[utils.Missing.name] = utils.Missing.value
            ds[Vars.m12].encoding[utils.Missing.name] = utils.Missing.value
            msgs.append(f"Saving datacube to {fixed_file}")

            # Re-chunk xr.Dataset to avoid memory errors when writing to the ZARR store
            ds = ds.chunk({utils.Coords.MID_DATE: time_chunks})
            ds.to_zarr(fixed_file, consolidated=True)


        if os.path.exists(fixed_file):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy
            for each_var in [Vars.m11, Vars.m12]:
                # Copy fixed variable and corresponding attribute values
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{each_var}',
                    f'{cube_url}/{each_var}',
                    "--acl", "bucket-owner-full-control"
                ]

                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

                factor_var = f'{each_var}_{Vars.postfix.dr_to_vr_factor}'
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{factor_var}',
                    f'{cube_url}/{factor_var}',
                    "--acl", "bucket-owner-full-control"
                ]

                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            for each_var in [Vars.ascending_img1, Vars.ascending_img2]:
                # Copy fixed variable and corresponding attribute values
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{each_var}',
                    f'{cube_url}/{each_var}',
                    "--acl", "bucket-owner-full-control"
                ]

                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            # Copy datacube meta files
            for each_file in itslive_utils.CUBE_META:
                command_line = [
                    "aws", "s3", "cp",
                    f'{fixed_file}/{each_file}',
                    f'{cube_url}/{each_file}',
                    "--acl", "bucket-owner-full-control"
                ]

                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            msgs.append(f"Removing local {fixed_file}")
            # shutil.rmtree(fixed_file)

            return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=8,
        help='Number of threads to use for parallel processing [%(default)d].'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of filenames to generate [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help='File that contains JSON list of datacubes filenames in s3 bucket '
                'to process [%(default)s].'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-fixed-nisar',
        help='Directory to store fixed datacubes before uploading them to the S3 bucket '
                '(it is much faster to read and write fixed datacubes locally first, then upload them to s3) [%(default)s]'
    )
    parser.add_argument(
        '-o', '--local_original_cube_dir',
        type=str,
        default='sandbox-original-nisar',
        help='Directory to store downloaded original datacubes to '
                '(it is much faster to read and write fixed datacubes locally first, then upload them to s3) [%(default)s]'
    )
    parser.add_argument(
        '-s', '--start_cube',
        type=int,
        default=0,
        help='Index for the start datacube to process (if previous processing terminated) [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit AWS push/pull commands.'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    FixDatacubes.DRY_RUN = args.dryrun

    cubes_to_generate = None
    if args.processCubesFile:
        cubes_to_generate = json.loads(args.processCubesFile.read_text())

    elif args.processCubes:
        cubes_to_generate = json.loads(args.processCubes)

    else:
        raise RuntimeError('Should provide cubes to process.')

    # Make sure all datacubes are unique
    if len(cubes_to_generate) != len(list(set(cubes_to_generate))):
        raise RuntimeError(f"Duplicates datacubes are identified.")

    logging.info(
        f"Found {len(cubes_to_generate)} unique datacubes to generate"
    )

    fix_cubes = FixDatacubes(
        cubes_to_generate,
        args.local_original_cube_dir,
        args.local_dir
    )

    fix_cubes(args.threads, args.start_cube)


if __name__ == '__main__':
    main()
    logging.info("Done.")
