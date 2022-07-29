#!/usr/bin/env python
"""
Re-compute amplitude and phase of existing composites using analytical solution by Alex and Chad
as provided in Matlab prototype code.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL)
"""

"""
Original prototype code in Matlab
=================================

function [vx_amp_r,vx_phase_r,vy_amp_r,vy_phase_r] = seasonal_velocity_rotation(theta,vx_amp,vx_phase,vy_amp,vy_phase)
% seasonal_velocity_rotation gives the amplitude and phase of seasonal
% velocity components. (Only the x and y components change when rotation is
% applied, i.e., v_amp and v_phase are unchanged).
%
% Inputs:
% theta (degrees) rotation of the coordinate system.
% vx_amp (m/yr) x component of seasonal amplitude in the original coordinate system.
% vx_phase (doy) day of maximum x velocity in original coordinate system.
% vy_amp (m/yr) y component of seasonal amplitude in the original coordinate system.
% vy_phase (doy) day of maximum y velocity in original coordinate system.
%
% Outputs:
% vx_amp_r (m/yr) x component of seasonal amplitude in the original coordinate system.
% vx_phase_r (doy) day of maximum x velocity in original coordinate system.
% vy_amp_r (m/yr) y component of seasonal amplitude in the original coordinate system.
% vy_phase_r (doy) day of maximum y velocity in original coordinate system.
%
% Written by Alex Gardner and Chad Greene, July 2022.

% Convert phase values from day-of-year to degrees:
vx_phase_deg = vx_phase*360/365.24;
vy_phase_deg = vy_phase*360/365.24;

% Rotation matrix for x component:
A1 =  vx_amp.*cosd(theta);
B1 = -vy_amp.*sind(theta);
vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

% Rotation matrix for y component:
A2 = vx_amp.*sind(theta);
B2 = vy_amp.*cosd(theta);
vy_amp_r   =   hypot(A2.*cosd(vx_phase_deg) + B2.*cosd(vy_phase_deg),  A2.*sind(vx_phase_deg) + B2.*sind(vy_phase_deg));
vy_phase_r = atan2d((A2.*sind(vx_phase_deg) + B2.*sind(vy_phase_deg)),(A2.*cosd(vx_phase_deg) + B2.*(cosd(vy_phase_deg))));

% Make all amplitudes positive (and reverse phase accordingly):
nx = vx_amp_r<0; % indices of negative Ax_r
vx_amp_r(nx) = -vx_amp_r(nx);
vx_phase_r(nx) = vx_phase_r(nx)+180;

ny = vy_amp_r<0; % indices of negative Ay_r
vy_amp_r(ny) = -vy_amp_r(ny);
vy_phase_r(ny) = vy_phase_r(ny)+180;

% Wrap to 360 degrees:
px = vx_phase_r > 0;
vx_phase_r = mod(vx_phase_r, 360);
vx_phase_r((vx_phase_r == 0) & px) = 360;

py = vy_phase_r > 0;
vy_phase_r = mod(vy_phase_r, 360);
vy_phase_r((vy_phase_r == 0) & py) = 360;

% Convert degrees to days:
vx_phase_r = vx_phase_r*365.24/360;
vy_phase_r = vy_phase_r*365.24/360;

end
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import json
import logging
import numpy as np
import os
import s3fs
import shutil
import subprocess
import xarray as xr
import zarr

from itscube_types import DataVars, Coords
from itslive_composite import CompDataVars


class FixAnnualComposites:
    """
    Class to apply fixes to ITS_LIVE datacubes composites:

    * Re-compute v_amp and v_phase data based on analytical solution instead of
      imperical solution as used in original composites code.
    """
    # Suffix to remove in original granule URLs
    SUFFIX_TO_USE = '.nc'
    S3_PREFIX = 's3://'
    DRY_RUN = False

    def __init__(self, bucket: str, bucket_dir: str, target_bucket_dir: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.bucket = bucket
        self.bucket_dir = bucket_dir
        self.target_bucket_dir = target_bucket_dir

        # Collect names for existing datacubes
        logging.info(f"Reading sub-directories of {os.path.join(bucket, bucket_dir)}")

        self.all_composites = []
        for each in self.s3.ls(os.path.join(bucket, bucket_dir)):
            cubes = self.s3.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]
            self.all_composites.extend(cubes)

        # Sort the list to guarantee the order of found stores
        self.all_composites.sort()
        logging.info(f"Found number of composites: {len(self.all_composites)}")

        # For debugging only
        # self.all_composites = self.all_composites[:1]
        # logging.info(f"ULRs: {self.all_composites}")

    def no__call__(self, local_dir: str, num_dask_workers: int, start_index: int=0):
        """
        Apply fixes.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} composites to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        for each in self.all_composites:
            logging.info(f"Starting {each}")
            msgs = FixAnnualComposites.all(each, self.bucket, self.bucket_dir, self.target_bucket_dir, local_dir, self.s3)
            logging.info("\n-->".join(msgs))

    def __call__(self, local_dir: str, num_dask_workers: int, start_index: int=0):
        """
        Apply fixes.
        """
        num_to_fix = len(self.all_composites) - start_index
        start = start_index

        logging.info(f"{num_to_fix} composites to fix...")

        if num_to_fix <= 0:
            logging.info(f"Nothing to fix, exiting.")
            return

        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        while num_to_fix > 0:
            num_tasks = num_dask_workers if num_to_fix > num_dask_workers else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(FixAnnualComposites.all)(each, self.bucket, self.bucket_dir, self.target_bucket_dir, local_dir, self.s3) for each in self.all_composites[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def all(composite_url: str, bucket_name: str, bucket_dir: str, target_bucket_dir: str, local_dir: str, s3_in):
        """
        Fix composites and copy them back to S3 bucket.

        Per Slack chat with Alex on July 12, 2022:
        "we need to rotate the vx/y_amp and vx/y_phase into the direction of v,
        which is defined by vx0 and vy0. If you replace the rotation matrix in
        the sudo code (coordinate projection rotation) by the rotation matrix
        defined by vx0 and vy0 then one of the rotated component is in the
        direction of v0 and the other is perpendicular to v0.
        We only want to retain the component that is in the direction of v0.""
        """
        _two_pi = np.pi * 2

        msgs = [f'Processing {composite_url}']

        zarr_store = s3fs.S3Map(root=composite_url, s3=s3_in, check=False)
        composite_basename = os.path.basename(composite_url)

        # Use composite parent directory to format local filename as there are
        # multiple copies of the same composite filename under different sub-directories
        dir_tokens = composite_url.split('/')

        # Write datacube locally, upload it to the bucket, remove file
        fixed_file = os.path.join(local_dir, f'{dir_tokens[-2]}_{composite_basename}')

        with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
            sizes = ds.sizes
            # Create rotation matrix based on vx0 and vy0

            # Matlab prototype code:
            # % Convert phase values from day-of-year to degrees:
            # vx_phase_deg = vx_phase*360/365.24;
            # vy_phase_deg = vy_phase*360/365.24;
            vx_phase_deg = (ds.vx_phase.values*360/365.24)
            vy_phase_deg = (ds.vy_phase.values*360/365.24)

            # Don't use np.nan values in calculations to avoid warnings
            valid_mask = (~np.isnan(vx_phase_deg)) & (~np.isnan(vy_phase_deg))

            # logging.info(f'Degrees: vx_phase_deg={vx_phase_deg[valid_mask]} vy_phase_deg={vy_phase_deg[valid_mask]}')

            # Convert degrees to radians as numpy trig. functions take angles in radians
            vx_phase_deg = vx_phase_deg*np.pi/180.0
            vy_phase_deg = vy_phase_deg*np.pi/180.0
            # logging.info(f'Radians: vx_phase_deg={vx_phase_deg[valid_mask]} vy_phase_deg={vy_phase_deg[valid_mask]}')

            # Matlab prototype code:
            # % Rotation matrix for x component:
            # A1 =  vx_amp.*cosd(theta);
            # B1 = -vy_amp.*sind(theta);

            # Python code: compute theta rotation angle
            # theta = arctan(vy0/vx0), since sin(theta)=vy0 and cos(theta)=vx0,
            theta = np.full_like(vx_phase_deg, np.nan)
            theta[valid_mask] = np.arctan2(ds.vy0.values[valid_mask], ds.vx0.values[valid_mask])

            if np.any(theta<0):
                # logging.info(f'Got negative theta, converting to positive values')
                mask = (theta<0)
                theta[mask] += _two_pi

            # Find negative values
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            A1 = ds.vx_amp.values*cos_theta
            B1 = ds.vy_amp.values*sin_theta

            # Matlab prototype code:
            # vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
            # vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

            # We want to retain the component only in the direction of v0,
            # which becomes new v_amp and v_phase
            v_amp = np.full_like(vx_phase_deg, np.nan)
            v_phase = np.full_like(vx_phase_deg, np.nan)

            v_amp[valid_mask] = np.hypot(
                A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask]),
                A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask])
            )
            # np.arctan2 returns phase in radians
            v_phase[valid_mask] = np.arctan2(
                A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask]),
                A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask])
            )*180.0/np.pi

            # ??? no need to convert to degrees, as we are going to wrap to 365.25 days
            # at the end
            # *180.0/np.pi

            # Matlab prototype code:
            # % Make all amplitudes positive (and reverse phase accordingly):
            # nx = vx_amp_r<0; % indices of negative Ax_r
            # vx_amp_r(nx) = -vx_amp_r(nx);
            # vx_phase_r(nx) = vx_phase_r(nx)+180;
            mask = v_amp < 0
            v_amp[mask] *= -1.0
            # v_phase[mask] += np.pi
            v_phase[mask] += 180

            # Matlab prototype code:
            # % Wrap to 360 degrees:
            # px = vx_phase_r > 0;
            # vx_phase_r = mod(vx_phase_r, 360);
            # vx_phase_r((vx_phase_r == 0) & px) = 360;
            mask = v_phase > 0
            # v_phase[mask] = np.remainder(v_phase[mask], _two_pi)
            v_phase[mask] = np.remainder(v_phase[mask], 360.0)
            mask = mask & (v_phase == 0)
            # v_phase[mask] = _two_pi
            v_phase[mask] = 360.0

            # Convert all values to positive
            mask = v_phase < 0
            if np.any(mask):
                # logging.info(f'Got negative phase, converting to positive values')
                # v_phase[mask] += _two_pi
                v_phase[mask] = np.remainder(v_phase[mask], -360.0)
                v_phase[mask] += 360.0

            # Matlab prototype code:
            # % Convert degrees to days:
            # vx_phase_r = vx_phase_r*365.24/360;
            # vy_phase_r = vy_phase_r*365.24/360;
            # Python: to be consistent with phase calculations in LSQ fit of composites
            # phase converted such that it reflects the day when value is maximized
            # TODO: confirm with Alex
            # Does not generate the same solution as original composites: because it's
            # shifted by 0.25 again?
            # v_phase = 365.25*((0.25 - v_phase/_two_pi) % 1)

            # Matlab's solution generates the same solution as original composites
            v_phase = v_phase*365.24/360

            # Replace v_amp variable in dataset
            ds[CompDataVars.V_AMP] = xr.DataArray(
                data=v_amp,
                coords=ds[CompDataVars.V_AMP].coords,
                dims=ds[CompDataVars.V_AMP].dims,
                attrs=ds[CompDataVars.V_AMP].attrs
            )

            # Replace v_phase variable in dataset
            ds[CompDataVars.V_PHASE] = xr.DataArray(
                data=v_phase,
                coords=ds[CompDataVars.V_PHASE].coords,
                dims=ds[CompDataVars.V_PHASE].dims,
                attrs=ds[CompDataVars.V_PHASE].attrs
            )
            msgs.append(f"Saving composite to {fixed_file}")

            # Set encoding
            encoding_settings = {}
            encoding_settings.setdefault(CompDataVars.TIME, {}).update({DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS})

            for each in [CompDataVars.TIME, CompDataVars.SENSORS, Coords.X, Coords.Y]:
                encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

            encoding_settings.setdefault(CompDataVars.SENSORS, {}).update({'dtype': 'str'})

            # Compression for the data
            compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)

            # Settings for "float" data types
            for each in [
                DataVars.VX,
                DataVars.VY,
                DataVars.V,
                CompDataVars.VX_ERROR,
                CompDataVars.VY_ERROR,
                CompDataVars.V_ERROR,
                CompDataVars.VX_AMP_ERROR,
                CompDataVars.VY_AMP_ERROR,
                CompDataVars.V_AMP_ERROR,
                CompDataVars.VX_AMP,
                CompDataVars.VY_AMP,
                CompDataVars.V_AMP,
                CompDataVars.VX_PHASE,
                CompDataVars.VY_PHASE,
                CompDataVars.V_PHASE,
                CompDataVars.OUTLIER_FRAC,
                CompDataVars.VX0,
                CompDataVars.VY0,
                CompDataVars.V0,
                CompDataVars.VX0_ERROR,
                CompDataVars.VY0_ERROR,
                CompDataVars.V0_ERROR,
                CompDataVars.SLOPE_VX,
                CompDataVars.SLOPE_VY,
                CompDataVars.SLOPE_V
                ]:
                encoding_settings.setdefault(each, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                    'dtype': np.float32,
                    'compressor': compressor
                })

            # Settings for "short" datatypes
            for each in [
                CompDataVars.COUNT,
                CompDataVars.COUNT0
            ]:
                encoding_settings.setdefault(each, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                    'dtype': np.short
                })

            # Settings for "max_dt" datatypes
            encoding_settings.setdefault(CompDataVars.MAX_DT, {}).update({
                    DataVars.FILL_VALUE_ATTR: DataVars.MISSING_POS_VALUE,
                    'dtype': np.short
                })

            # Settings for "sensor_include" datatypes
            encoding_settings.setdefault(CompDataVars.SENSOR_INCLUDE, {}).update({
                    'dtype': np.short
                })

            # Chunking to apply when writing datacube to the Zarr store
            chunks_settings = (1, sizes[Coords.Y], sizes[Coords.X])

            for each in [
                DataVars.VX,
                DataVars.VY,
                DataVars.V,
                CompDataVars.VX_ERROR,
                CompDataVars.VY_ERROR,
                CompDataVars.V_ERROR,
                CompDataVars.MAX_DT
            ]:
                encoding_settings[each].update({
                    'chunks': chunks_settings
                })

            # Chunking to apply when writing datacube to the Zarr store
            chunks_settings = (sizes[Coords.Y], sizes[Coords.X])

            for each in [
                CompDataVars.VX_AMP,
                CompDataVars.VY_AMP,
                CompDataVars.V_AMP,
                CompDataVars.VX_PHASE,
                CompDataVars.VY_PHASE,
                CompDataVars.V_PHASE,
                CompDataVars.VX_AMP_ERROR,
                CompDataVars.VY_AMP_ERROR,
                CompDataVars.V_AMP_ERROR,
                CompDataVars.OUTLIER_FRAC,
                CompDataVars.SENSOR_INCLUDE,
                CompDataVars.VX0,
                CompDataVars.VY0,
                CompDataVars.V0,
                CompDataVars.VX0_ERROR,
                CompDataVars.VY0_ERROR,
                CompDataVars.V0_ERROR,
                CompDataVars.SLOPE_VX,
                CompDataVars.SLOPE_VY,
                CompDataVars.SLOPE_V
                ]:
                encoding_settings[each].update({
                    'chunks': chunks_settings
                })

            # logging.info(f"Encoding settings: {encoding_settings}")
            ds.to_zarr(fixed_file, encoding=encoding_settings, consolidated=True)

        target_url = composite_url.replace(bucket_dir, target_bucket_dir)
        if not target_url.startswith(FixAnnualComposites.S3_PREFIX):
            target_url = FixAnnualComposites.S3_PREFIX + target_url

        if FixAnnualComposites.DRY_RUN:
            msgs.append(f'DRYRUN: copy composite to {target_url}')
            return msgs

        if os.path.exists(fixed_file) and len(bucket_name):
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy

            # Enable conversion to NetCDF when the cube is created
            # Convert Zarr to NetCDF and copy to the bucket
            # nc_filename = args.outputStore.replace('.zarr', '.nc')
            # zarr_to_netcdf.main(args.outputStore, nc_filename, ITSCube.NC_ENGINE)
            # ITSCube.show_memory_usage('after Zarr to NetCDF conversion')
            env_copy = os.environ.copy()

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='its-live-data',
        help='AWS S3 that stores ITS_LIVE annual composites to fix v_error for [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir', type=str,
        default='composites/annual/v02',
        help='AWS S3 bucket and directory that store annual composites [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_bucket_dir', type=str,
        default='composites/annual/v02_fixed_amp_phase',
        help='AWS S3 directory to store fixed annual composites [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir', type=str,
        default='sandbox',
        help='Directory to store fixed granules before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument('-w', '--dask-workers', type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually copy any data to AWS S3 bucket'
    )
    parser.add_argument(
        '-s', '--start-index',
        type=int,
        default=0,
        help='Index for the start datacube to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    FixAnnualComposites.DRY_RUN = args.dryrun

    fix_composites = FixAnnualComposites(args.bucket, args.bucket_dir,  args.target_bucket_dir)
    fix_composites(args.local_dir, args.dask_workers, args.start_index)

if __name__ == '__main__':
    main()
    logging.info("Done.")
