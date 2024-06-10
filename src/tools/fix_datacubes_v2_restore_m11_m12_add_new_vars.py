#!/usr/bin/env python
"""
1. Restore Sentinel-1 M11 and M12 values within existing ITS_LIVE datacubes that are residing in AWS S3 bucket.
    * Copy M11 and M12 values from S1 granules into corresponding layers of the existing datacubes.

2. Add new data variables to existing V2 datacubes:

    * issue #27: Using granules img_pair_info variable, add binary array for 'ascending/descending' per each image
    of the granule:
        ascending1 = [true/false]
        ascending2 = [true/false]

    * issue #28: Add new sensor_uid1 & sensor_uid2 data variables of uint8 type to datacubes.
    Both variables should have description of unique values:
    description: unique sensor id: 4 = Landsat 4, 5 = Landsat 5, 6 = Landsat 6, 7 = Landsat 7, 8 = Landsat 8,
        9 = Landsat 9, 11 = Sentinel 1A, 12 = Sentinel 1B, 21 = Sentinel 2A, 22 = Sentinel 2B

* Re-chunk datacube's mid_date to include the whole dimension to speed up access time to the data.
* Push corrected datacubes back to the S3 bucket.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Alex Gardner
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import logging
import numpy as np
import os
import s3fs
import shutil
import subprocess
import xarray as xr
import zarr

from itslive_composite import SensorExcludeFilter, MissionSensor, Output
from itscube import ITSCube
from itscube_types import DataVars, Coords, BinaryFlag
from sensor_id import SENSORS, all_sensors_description

NC_ENGINE = 'h5netcdf'


class FixDatacubes:
    """
    Class to apply fixes to ITS_LIVE datacubes:

    * Copy M11 and M12 values from S1 granules into corresponding layers of the existing datacubes.
    * Re-chunk mid_date dimenstion to the same chunking as the rest of 1-d data variables.
    * Add new data variables:
        * issue #27: Using granules img_pair_info variable, add binary array for 'ascending/descending' per each image
        of the granule:
        ascending1 = [true/false]
        ascending2 = [true/false]

        * issue #28: Add new sensor_uid1 & sensor_uid2 data variables of uint8 type to datacubes.
        Both variables should have description of unique values:
        description: unique sensor id: 4 = Landsat 4, 5 = Landsat 5, 6 = Landsat 6, 7 = Landsat 7, 8 = Landsat 8,
        9 = Landsat 9, 11 = Sentinel 1A, 12 = Sentinel 1B, 21 = Sentinel 2A, 22 = Sentinel 2B
    * Push corrected datacubes back to the S3 bucket.
    """
    S3_PREFIX = 's3://'
    DRY_RUN = False

    NEW_VAR_NAMES = {
        DataVars.ImgPairInfo.SATELLITE_IMG1: DataVars.SENSOR_UID1,
        DataVars.ImgPairInfo.SATELLITE_IMG2: DataVars.SENSOR_UID2,
    }

    def __init__(
        self,
        bucket: str,
        bucket_dir: str,
        target_bucket_dir: str,
        local_original_cube_dir: str,
        local_dir: str
    ):
        """
        Initialize object.

        Args:
            bucket (str): AWS S3 bucket
            bucket_dir (str): AWS S3 directory that stores datacubes.
            target_bucket_dir (str): AWS S3 directgory to store corrected datacubes.
            local_original_cube_dir (str): Local directory to store downloaded original datacubes to fix.
            local_dir (str): Local directory to save corrected cubes to.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket_dir = bucket_dir
        self.target_bucket_dir = target_bucket_dir

        self.local_original_cube_dir = local_original_cube_dir
        self.local_dir = local_dir

        # Collect names for existing datacubes
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_zarr_datacubes = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            cubes = self.s3.ls(each)
            cubes = [each for each in cubes if each.endswith('.zarr')]
            self.all_zarr_datacubes.extend(cubes)

        # Sort the cubes to guarantee the order
        # (if we need to resume processing from previous interrupted run)
        self.all_zarr_datacubes.sort()

        logging.info(f"Found number of datacubes: {len(self.all_zarr_datacubes)}")

        if not os.path.exists(self.local_dir):
            os.mkdir(self.local_dir)

        if not os.path.exists(self.local_original_cube_dir):
            os.mkdir(self.local_original_cube_dir)

    def debug__call__(self, num_dask_workers: int):
        """
        Fix mapping.GeoTransform of ITS_LIVE datacubes stored in S3 bucket.
        Strip suffix from original granules names as appear within 'granule_url'
        data variable and skipped_* datacube attributes.
        """
        num_to_fix = len(self.all_zarr_datacubes)

        logging.info(f"{num_to_fix} datacubes to fix...")

        if num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        for each_cube in self.all_zarr_datacubes:
            logging.info(f"Starting {each_cube}")
            msgs = FixDatacubes.all(
                each_cube,
                self.bucket_dir,
                self.target_bucket_dir,
                self.local_original_cube_dir,
                self.local_dir,
                self.s3
            )
            logging.info("\n-->".join(msgs))

    def __call__(
        self,
        num_dask_workers: int,
        start_cube: int = 0
    ):
        """
        Apply all the changes to existing V2 datacubes.
        """
        num_to_fix = len(self.all_zarr_datacubes) - start_cube
        start = start_cube

        logging.info(f"{num_to_fix} datacubes to fix...")

        if num_to_fix <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        # For debugging
        num_to_fix = 1

        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [
                dask.delayed(FixDatacubes.all)(
                    each,
                    self.bucket_dir,
                    self.target_bucket_dir,
                    self.local_original_cube_dir,
                    self.local_dir,
                    self.s3
                ) for each in self.all_zarr_datacubes[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=num_dask_workers
                )

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def all(
        cube_url: str,
        cube_bucket_dir: str,
        target_bucket_dir: str,
        local_original_cube_dir: str,
        local_dir: str,
        s3: s3fs.S3FileSystem
    ):
        """
        Fix datacubes and copy them to S3 bucket's new location.

        Args:
        =====
        cube_url (str): Original cube URL in S3 bucket to add new variables to.
        cube_bucket_dir (str): AWS S3 bucket for original datacube.
        target_bucket_dir (str): AWS S3 target bucket for updated datacube.
        local_original_cube_dir (str): Local directory to store copy of the original datacube to update.
        local_dir (str): Local directory to store updated datacube before copying it to the target S3 bucket.
        s3 (s3fs.S3FileSystem): s3fs FileSystem object to access datacubes and granules.
        """
        msgs = [f'Processing {cube_url}']

        # cube_store = s3fs.S3Map(root=cube_url, s3=s3, check=False)
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
        source_url = cube_url
        if not cube_url.startswith(ITSCube.S3_PREFIX):
            source_url = ITSCube.S3_PREFIX + cube_url

        local_original_cube = os.path.join(local_original_cube_dir, cube_basename)
        command_line = [
            "awsv2", "s3", "cp", "--recursive",
            source_url,
            local_original_cube
        ]

        msgs.append(f"Creating local copy of {source_url}: {local_original_cube}")
        msgs.append(' '.join(command_line))

        command_return = subprocess.run(
            command_line,
            env=env_copy,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if command_return.returncode != 0:
            msgs.append(f"ERROR: Failed to copy {source_url} to {local_original_cube}: {command_return.stdout}")

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, cube_basename)

        fill_value = 0
        ascending_fill_value = DataVars.MISSING_UINT8_VALUE

        with xr.open_dataset(local_original_cube, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            # 1. Add unique sensor ID variables per each image in the granule - information is already in the datacube
            for each_var in [
                DataVars.ImgPairInfo.SATELLITE_IMG1,
                DataVars.ImgPairInfo.SATELLITE_IMG2
            ]:
                msgs.append(f'Processing {each_var}...')
                missions_str = ds[each_var].values

                unique_satellite_img = set(ds[each_var].values)
                msgs.append(f'Got {each_var}: {unique_satellite_img}')

                unique_id = np.full((len(missions_str)), fill_value, dtype=np.uint8)

                # Mask all granules that correspond to the unique missionID
                for each_mission in unique_satellite_img:
                    mask = (missions_str == each_mission)
                    msgs.append(f'Identified {np.sum(mask)} layers for {each_mission}')

                    mask_i = np.where(mask == True)

                    sensor_value = SENSORS[each_mission]
                    msgs.append(f'Setting "{each_mission}" masked values (num={np.sum(mask_i)}) to {sensor_value}')
                    unique_id[mask_i[0]] = sensor_value

                # Check if all sensors were masked
                mask = (missions_str == fill_value)
                if np.sum(mask) != 0:
                    msgs.append(f'ERROR: got unprocessed sensors: {set(missions_str[mask[0]].values)}')

                # Add new variable to the datacube
                new_var_name = FixDatacubes.NEW_VAR_NAMES[each_var]
                ds[new_var_name] = xr.DataArray(data=unique_id, coords=ds[each_var].coords, dims=ds[each_var].dims)
                ds[new_var_name].attrs = {
                    DataVars.STD_NAME: DataVars.STANDARD_NAME[new_var_name],
                    DataVars.DESCRIPTION_ATTR: all_sensors_description()
                }

            msgs.append(f'Cube dimensions: {ds.dims}')
            x_values = ds.x.values
            grid_x_min, grid_x_max = x_values.min(), x_values.max()

            y_values = ds.y.values
            grid_y_min, grid_y_max = y_values.min(), y_values.max()

            # Identify S1 layers within the cube
            sensors = ds[DataVars.ImgPairInfo.SATELLITE_IMG1].values
            sensors_str = SensorExcludeFilter.map_sensor_to_group(sensors)

            s1_mask = (sensors_str == MissionSensor.SENTINEL1.mission)
            msgs.append(f'Identified {np.sum(s1_mask)} S1 layers in the cube')

            mask_i = np.where(s1_mask == True)
            num_mid_date = len(ds.mid_date)

            ascending_img1 = np.full((num_mid_date), ascending_fill_value, dtype=np.uint8)
            ascending_img2 = np.full((num_mid_date), ascending_fill_value, dtype=np.uint8)

            # This look does nothing but loading variable values in.
            # Need to load all of M11/M12 data values in order to update them.
            # Otherwise it silently ignores values when updating (xarray bug?)
            for each_var in [DataVars.M11, DataVars.M12]:
                m_values = ds[each_var].values
                msgs.append(f'cube {each_var}: min={np.nanmin(m_values)} max={np.nanmax(m_values)}')

            # If there are no S1 granules, we still want to rechunk 'mid_date' coordinate
            for each_index in mask_i[0]:
                # Read URL of the granule. For example, granules paths will be in the format:
                # https://its-live-data.s3.amazonaws.com/velocity_image_pair/sentinel1/v02/N70W060/S1A_IW_SLC__1SSH_20160728T113645_20160728T113712_012348_0133B2_74C0_X_S1A_IW_SLC__1SSH_20160809T113646_20160809T113713_012523_013989_2C50_G0120V02_P030.nc
                granule = str(ds.granule_url[each_index].values)

                each_granule_s3 = granule.replace('https://', '')
                each_granule_s3 = each_granule_s3.replace('.s3.amazonaws.com', '')
                # If using new temporary location of restored S1 granules
                each_granule_s3 = each_granule_s3.replace('/sentinel1/', '/sentinel1-restoredM/')

                # Open the granule
                with s3.open(each_granule_s3, mode='rb') as fhandle:
                    with xr.open_dataset(fhandle, engine=NC_ENGINE) as granule_ds:
                        granule_ds = granule_ds.load()

                        # 2. Extract flight direction for both images of the granule
                        ascending_img1[each_index] = granule_ds.img_pair_info.attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG1].strip() == DataVars.ImgPairInfo.ASCENDING
                        ascending_img2[each_index] = granule_ds.img_pair_info.attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG2].strip() == DataVars.ImgPairInfo.ASCENDING

                        msgs.append(f'Granule for index={each_index}: {each_granule_s3}; date_updated: {granule_ds.attrs["date_updated"]}')

                        # Zoom into cube polygon
                        mask_x = (granule_ds.x >= grid_x_min) & (granule_ds.x <= grid_x_max)
                        mask_y = (granule_ds.y >= grid_y_min) & (granule_ds.y <= grid_y_max)
                        mask = (mask_x & mask_y)

                        cropped_ds = granule_ds.where(mask, drop=True)

                        # 3. Restore M11/M12 values in the datacube
                        for each_var in [DataVars.M11, DataVars.M12]:
                            # # Show current values
                            # m_values = ds[each_var][each_index, :, :].values
                            # print(f'====>before assigning ds {each_var}: m_values.shape={m_values.shape} min={np.nanmin(m_values)} max={np.nanmax(m_values)}')

                            ds[each_var][each_index, :, :].loc[dict(x=cropped_ds.x, y=cropped_ds.y)] = cropped_ds[each_var]

                            # # Show restored values
                            # m_values = ds[each_var][each_index, :, :].values
                            # print(f'====>assigned ds {each_var}: m_values.shape={m_values.shape} min={np.nanmin(m_values)} max={np.nanmax(m_values)}')

            # ds_encoding = zarr_to_netcdf.ENCODING_ZARR.copy()

            # Add new variables to the datacube - just use existing 1-d data variable coords and dims
            ds[DataVars.ASCENDING_IMG1] = xr.DataArray(data=ascending_img1, coords=ds[DataVars.ImgPairInfo.SATELLITE_IMG1].coords, dims=ds[DataVars.ImgPairInfo.SATELLITE_IMG1].dims)
            ds[DataVars.ASCENDING_IMG1].attrs = {
                DataVars.STD_NAME: DataVars.STANDARD_NAME[DataVars.ASCENDING_IMG1],
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.ASCENDING_IMG1],
                BinaryFlag.VALUES_ATTR: BinaryFlag.VALUES,
                BinaryFlag.MEANINGS_ATTR: BinaryFlag.MEANINGS[DataVars.ASCENDING_IMG1]
            }

            ds[DataVars.ASCENDING_IMG2] = xr.DataArray(data=ascending_img2, coords=ds[DataVars.ImgPairInfo.SATELLITE_IMG1].coords, dims=ds[DataVars.ImgPairInfo.SATELLITE_IMG1].dims)
            ds[DataVars.ASCENDING_IMG2].attrs = {
                DataVars.STD_NAME: DataVars.STANDARD_NAME[DataVars.ASCENDING_IMG2],
                DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.ASCENDING_IMG2],
                BinaryFlag.VALUES_ATTR: BinaryFlag.VALUES,
                BinaryFlag.MEANINGS_ATTR: BinaryFlag.MEANINGS[DataVars.ASCENDING_IMG2]
            }

            # Correct chunking settings in the cube, use them as golden standard for all variables
            chunking_1d = ds[DataVars.ImgPairInfo.DATE_DT].encoding[Output.CHUNKS_ATTR]
            chunking_2d = (len(ds.y), len(ds.x))
            chunking_3d = ds[DataVars.CHIP_SIZE_HEIGHT].encoding[Output.CHUNKS_ATTR]
            compression_zarr = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

            # Fix chunking for mid_date, ice masks, autoRIFT_software_version, granule_url,
            # and just to be sure - for x/y (set to the full extend already)
            for each_var in ds:
                if Output.CHUNKS_ATTR in ds[each_var].encoding:
                    ds_chunking = ds[each_var].encoding[Output.CHUNKS_ATTR]
                    chunking = ()

                    if len(ds_chunking) == 1:
                        chunking = chunking_1d

                    elif len(ds_chunking) == 2:
                        chunking = chunking_2d

                    elif len(ds_chunking) == 3:
                        chunking = chunking_3d

                    ds[each_var].encoding[Output.CHUNKS_ATTR] = chunking

                    # Apply the same compression to all data variables
                    ds[each_var].encoding[Output.COMPRESSOR_ATTR] = compression_zarr

            # Chunking for X and Y are set to full extend by default, set it just to be sure
            ds[Coords.X].encoding[Output.CHUNKS_ATTR] = (len(ds.x))
            ds[Coords.Y].encoding[Output.CHUNKS_ATTR] = (len(ds.y))

            # Change datatype for M11 and M12 to floating point
            ds[DataVars.M11].encoding[Output.DTYPE_ATTR] = np.float32
            ds[DataVars.M12].encoding[Output.DTYPE_ATTR] = np.float32

            # Set encoding attributes for new data variables
            for each_var in [DataVars.SENSOR_UID1, DataVars.SENSOR_UID2]:
                ds[each_var].encoding = {
                    Output.DTYPE_ATTR: np.uint8,
                    Output.COMPRESSOR_ATTR: compression_zarr,
                    Output.CHUNKS_ATTR: chunking_1d
                }

            for each_var in [DataVars.ASCENDING_IMG1, DataVars.ASCENDING_IMG2]:
                ds[each_var].encoding = {
                    Output.MISSING_VALUE_ATTR: DataVars.MISSING_UINT8_VALUE,
                    Output.DTYPE_ATTR: np.uint8,
                    Output.COMPRESSOR_ATTR: compression_zarr,
                    Output.CHUNKS_ATTR: chunking_1d
                }

            msgs.append(f"Saving datacube to {fixed_file}")
            # Re-chunk xr.Dataset to avoid memory errors when writing to the ZARR store
            ds = ds.chunk({Coords.MID_DATE: 250})
            ds.to_zarr(fixed_file, consolidated=True)

        if FixDatacubes.DRY_RUN:
            return msgs

        if os.path.exists(fixed_file):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy
            target_url = cube_url.replace(cube_bucket_dir, target_bucket_dir)

            if not target_url.startswith(FixDatacubes.S3_PREFIX):
                target_url = FixDatacubes.S3_PREFIX + target_url

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

            msgs.append(f"Removing local {fixed_file}")
            shutil.rmtree(fixed_file)

            return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE datacubes to fix chunking and restore M11/M12 values for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='datacubes/v2',
        help='AWS S3 directory that store the datacubes to fix [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir',
        type=str,
        default='datacubes/v2_restored_M11_M12',
        help='AWS S3 directory to store fixed datacubes [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox',
        help='Directory to store fixed datacubes before uploading them to the S3 bucket '
                '(it is much faster to read and write fixed datacubes locally first, then upload them to s3) [%(default)s]'
    )
    parser.add_argument(
        '-o', '--local_original_cube_dir',
        type=str,
        default='sandbox-original',
        help='Directory to store downloaded original datacubes to '
                '(it is much faster to read and write fixed datacubes locally first, then upload them to s3) [%(default)s]'
    )
    parser.add_argument(
        '-w', '--dask-workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
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

    fix_cubes = FixDatacubes(
        args.bucket,
        args.bucket_dir,
        args.target_bucket_dir,
        args.local_original_cube_dir,
        args.local_dir
    )

    fix_cubes(args.dask_workers, args.start_cube)


if __name__ == '__main__':
    main()
    logging.info("Done.")
