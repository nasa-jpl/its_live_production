"""
Fix ITS_LIVE S1 V2 granules metadata to confirm to the latest format and NSIDC standards.

1. Fix global attribute:
    :Conventions = "CF-1.8"

2. img_pair_info:
    img_pair_info:standard_name = "image_pair_information"

3. mapping:
    Add "crs_wkt" which is a duplicate of "spatial_ref"
    Rename "spatial_proj4" to "proj4text"
    Remove if present:
        :CoordinateAxisTypes = "GeoX GeoY";
        :CoordinateTransformType = "Projection";

4. For mapping:grid_mapping_name: follow NSIDC/nsidc_vel_iamge_pairs.py:fix_mapping_attrs() fixes related to mapping.

5. Set new std. names:
    vx:standard_name = "land_ice_surface_x_velocity"
    vy:standard_name = "land_ice_surface_y_velocity"
    v:standard_name = "land_ice_surface_velocity"

5a. Change v description to:
    v:description = "velocity magnitude"

6. Replace "m/y" with "meter/year" units:
    vx:units = "meter/year" ;
    vy:units = "meter/year" ;
    v:units = "meter/year" ;
    v_error:units = "meter/year" ;
    vr:units = "meter/year" ;
    va:units = "meter/year" ;

7.	Replace "m/y" with "meter/year" units:
    M11:units = "pixel/(meter/year)" ;
    M12:units = "pixel/(meter/year)" ;

8. Replace "error_mask" with "error_stationary" attribute:
    vx:error_stationary
    vy:error_stationary
    vr:error_stationary
    va:error_stationary

9. Rename "error_mask_description" with "error_stationary_description":
    vx:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    vy:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    vr:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    va:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;

10. Replace "m/y" with "meter/year" in:
    vx:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    vy:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    vr:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
    va:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;

11. Rename "stable_shift_mask" to "stable_shift_stationary":
    vx:stable_shift_stationary
    vy:stable_shift_stationary
    vr:stable_shift_stationary
    va:stable_shift_stationary

12. Rename "stable_count_mask" to "stable_count_stationary":
    vx:stable_count_stationary
    vy:stable_count_stationary
    vr:stable_count_stationary
    va:stable_count_stationary

13.Replace 'binary' units for 'interp_mask' by:
        flag_values = 0UB, 1UB; // ubyte
        flag_meanings = 'measured, interpolated'

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket.
It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Joe Kennedy, Alex Gardner, Chad Greene
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import json
import logging
import numpy as np
import os
import s3fs
import xarray as xr

from mission_info import Encoding
from itscube_types import DataVars
from nsidc_types import Mapping
from nsidc_vel_image_pairs import required_attrs


def fix_metadata(ds: xr.Dataset):
    """
    1. Fix global attribute:
        :Conventions = "CF-1.8"

    2. img_pair_info:
        img_pair_info:standard_name = "image_pair_information"

    3. mapping:
        a. Add "crs_wkt" which is a duplicate of "spatial_ref"
            Adding crs_wkt (redundant with spatial_ref) expands interoperability
            with geolocation tools

        b. Rename "spatial_proj4" to "proj4text"

        c. Remove if present:
            :CoordinateAxisTypes = "GeoX GeoY";
            :CoordinateTransformType = "Projection";


    4. For mapping:grid_mapping_name: follow NSIDC/nsidc_vel_iamge_pairs.py:fix_mapping_attrs() fixes related to mapping.

    5. Set new std. names:
        vx:standard_name = "land_ice_surface_x_velocity"
        vy:standard_name = "land_ice_surface_y_velocity"
        v:standard_name = "land_ice_surface_velocity"

    5a. Change v description to:
        v:description = "velocity magnitude"

    6. Replace "m/y" with "meter/year" units:
        vx:units = "meter/year" ;
        vy:units = "meter/year" ;
        v:units = "meter/year" ;
        v_error:units = "meter/year" ;
        vr:units = "meter/year" ;
        va:units = "meter/year" ;

    7.	Replace "m/y" with "meter/year" units:
        M11:units = "pixel/(meter/year)" ;
        M12:units = "pixel/(meter/year)" ;

    8. Replace "error_mask" with "error_stationary" attribute:
        vx:error_stationary
        vy:error_stationary
        vr:error_stationary
        va:error_stationary

    9. Rename "error_mask_description" with "error_stationary_description":
        vx:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        vy:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        vr:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        va:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;

    10. Replace "m/yr" with "meter/year" in:
        vx:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        vy:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        vr:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;
        va:error_stationary_description = "RMSE over stable surfaces, stationary or slow-flowing surfaces with velocity < 15 meter/year identified from an external mask" ;

    11. Rename "stable_shift_mask" to "stable_shift_stationary":
        vx:stable_shift_stationary
        vy:stable_shift_stationary
        vr:stable_shift_stationary
        va:stable_shift_stationary

    12. Rename "stable_count_mask" to "stable_count_stationary":
        vx:stable_count_stationary
        vy:stable_count_stationary
        vr:stable_count_stationary
        va:stable_count_stationary

    13.Replace 'binary' units for 'interp_mask' by:
        flag_values = 0UB, 1UB; // ubyte
        flag_meanings = 'measured, interpolated'
    """
    _conventions = 'Conventions'
    _cf_value = 'CF-1.8'

    _new_std_name = {
        DataVars.VX: "land_ice_surface_x_velocity",
        DataVars.VY: "land_ice_surface_y_velocity",
        DataVars.V: "land_ice_surface_velocity"
    }

    _new_units = {
        DataVars.VX: DataVars.M_Y_UNITS,
        DataVars.VY: DataVars.M_Y_UNITS,
        DataVars.V: DataVars.M_Y_UNITS,
        DataVars.V_ERROR: DataVars.M_Y_UNITS,
        DataVars.VR: DataVars.M_Y_UNITS,
        DataVars.VA: DataVars.M_Y_UNITS,
        DataVars.M11: DataVars.PIXEL_PER_M_YEAR,
        DataVars.M12: DataVars.PIXEL_PER_M_YEAR
    }

    _old_error_mask = 'error_mask'
    _old_error_mask_description = 'error_mask_description'
    _error_stationary_description = 'error_stationary_description'

    _old_stable_shift_mask = 'stable_shift_mask'
    _old_stable_count_mask = 'stable_count_mask'
    _old_m_y_units = 'm/y'
    _old_m_yr_units = 'm/yr'


    _spatial_epsg = 'spatial_epsg'

    flag_values = 'flag_values'
    flag_meanings = 'flag_meanings'

    # Changes #8-12 will apply to these variables only
    _new_error_mask = [
        DataVars.VX,
        DataVars.VY,
        DataVars.VR,
        DataVars.VA
    ]

    binary_flags = np.array([0, 1], dtype=np.uint8)
    _binary_meanings = {
        DataVars.INTERP_MASK: 'measured interpolated'
    }

    # 1. Fix global attribute:
    ds.attrs[_conventions] = _cf_value

    # 2. img_pair_info:standard_name = "image_pair_information"
    ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.STD_NAME] = "image_pair_information"

    # Adding crs_wkt (redundant with spatial_ref) expands interoperability
    # with geolocation tools:
    # 3a. Add "crs_wkt" which is a duplicate of "spatial_ref"
    mapping_attrs = ds[DataVars.MAPPING].attrs

    # 3c. Remove if present:
    # :CoordinateAxisTypes = "GeoX GeoY";
    # :CoordinateTransformType = "Projection";
    if Mapping.COORDINATE_TRANSFORM_TYPE in mapping_attrs:
        del mapping_attrs[Mapping.COORDINATE_TRANSFORM_TYPE]

    if Mapping.COORDINATE_AXIS_TYPES in mapping_attrs:
        del mapping_attrs[Mapping.COORDINATE_AXIS_TYPES]

    # Adding crs_wkt (redundant with spatial_ref) expands interoperability
    # with geolocation tools:
    if (Mapping.CRS_WKT not in mapping_attrs) and (Mapping.SPATIAL_REF in mapping_attrs):
        mapping_attrs[Mapping.CRS_WKT] = mapping_attrs[Mapping.SPATIAL_REF]

    # 3b. Rename "spatial_proj4" to "proj4text"
    if Mapping.SPATIAL_PROJ4 in mapping_attrs:
        mapping_attrs[Mapping.PROJ4TEXT] = mapping_attrs[Mapping.SPATIAL_PROJ4]
        del mapping_attrs[Mapping.SPATIAL_PROJ4]

    for each_var in list(ds.keys()):
        #  5. Set new std. names:
        if each_var in _new_std_name:
            ds[each_var].attrs[DataVars.STD_NAME] = _new_std_name[each_var]

        # 6. Replace "m/y" with "meter/year" units:
        # 7. Replace "m/y" with "meter/year" units:
        if each_var in _new_units:
            ds[each_var].attrs[DataVars.UNITS] = _new_units[each_var]

        # 8. Replace "error_mask" with "error_stationary" attribute:
        if each_var in _new_error_mask:
            ds[each_var].attrs[DataVars.ERROR_MASK] = ds[each_var].attrs[_old_error_mask]
            del ds[each_var].attrs[_old_error_mask]

            # 9. Rename "error_mask_description" with "error_stationary_description" and
            # 10. Replace 'm/y' with 'meter/year'
            desc_value = ds[each_var].attrs[_old_error_mask_description]
            desc_value = desc_value.replace(_old_m_yr_units, DataVars.M_Y_UNITS)

            ds[each_var].attrs[_error_stationary_description] = desc_value
            del ds[each_var].attrs[_old_error_mask_description]

            # 11. Rename "stable_shift_mask" to "stable_shift_stationary":
            ds[each_var].attrs[DataVars.STABLE_SHIFT_MASK] = ds[each_var].attrs[_old_stable_shift_mask]
            del ds[each_var].attrs[_old_stable_shift_mask]

            # 12. Rename "stable_count_mask" to "stable_count_stationary":
            ds[each_var].attrs[DataVars.STABLE_COUNT_MASK] = ds[each_var].attrs[_old_stable_count_mask]
            del ds[each_var].attrs[_old_stable_count_mask]

    # 13. Replace 'binary' units
    if DataVars.UNITS in ds[DataVars.INTERP_MASK].attrs:
        del ds[DataVars.INTERP_MASK].attrs[DataVars.UNITS]
        ds[DataVars.INTERP_MASK].attrs[flag_values] = binary_flags
        ds[DataVars.INTERP_MASK].attrs[flag_meanings] = _binary_meanings[DataVars.INTERP_MASK]

    epsgcode = int(mapping_attrs[_spatial_epsg])

    # Apply corrections based on the EPSG code
    if epsgcode not in required_attrs and epsgcode != 3031:
        return ds

    if epsgcode in required_attrs:
        for each_attr, each_value in required_attrs[epsgcode].items():
            mapping_attrs[each_attr] = each_value

        mapping_attrs[Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN] = 0.9996
        mapping_attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = 0.0

    if epsgcode == 3031:
        # :latitude_of_origin = -71.0; // double	delete attribute
        # :semi_major_axis = 6378137.0; // double	optional attribute; but if to be included, set to 6378137.0
        # :semi_minor_axis = 6356.752; // double	optional attribute; remove attribute (I can't find the correct value)
        # :standard_parallel = -71.0; // double	required attribute; add attribute and set to -71.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin).
        if Mapping.LATITUDE_OF_ORIGIN in mapping_attrs:
            del mapping_attrs[Mapping.LATITUDE_OF_ORIGIN]

        mapping_attrs[Mapping.STANDARD_PARALLEL] = -71.0

    return ds


def fix_all(source_bucket: str, source_dir: str, target_dir: str, granule_url: str, local_dir: str, s3):
    """
    Fix everything in the granule.
    """
    _v_description = "velocity magnitude"

    msgs = [f'Processing {granule_url}']

    # get center lat lon
    with s3.open(granule_url) as fhandle:
        with xr.open_dataset(fhandle) as ds:
            # Add the date when the granule was updated
            ds.attrs['date_updated'] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

            # Update v.description for all granules - this fix was implemented in autoRIFT post
            # V2 latest campaign
            # 5a. v:description = "velocity magnitude"
            ds[DataVars.V].attrs[DataVars.DESCRIPTION_ATTR] = _v_description

            # Check if granule confirms to the new format already, then fix v.description only
            if DataVars.STABLE_SHIFT_MASK in ds[DataVars.VX].attrs:
                msgs.append('Confirms to new format already.')

            else:
                # Apply all the fixes
                ds = fix_metadata(ds)

            # Write the granule locally, upload it to the bucket, remove local file
            granule_basename = os.path.basename(granule_url)
            fixed_file = os.path.join(local_dir, granule_basename)
            ds.to_netcdf(fixed_file, engine='h5netcdf', encoding=Encoding.SENTINEL1)

            # Upload corrected granule to the bucket
            s3_client = boto3.client('s3')
            try:
                bucket_granule = granule_url.replace(source_bucket+'/', '')
                bucket_granule = bucket_granule.replace(source_dir, target_dir)
                msgs.append(f"Uploading to {source_bucket}/{bucket_granule}")

                if not FixSentinel1Granules.DRY_RUN:
                    s3_client.upload_file(fixed_file, source_bucket, bucket_granule)

                    # msgs.append(f"Removing local {fixed_file}")
                    os.unlink(fixed_file)

            except ClientError as exc:
                msgs.append(f"ERROR: {exc}")

            # No need to copy PNG files for the granule:
            # We are relying on the fact that we only fix the NetCDF granules, but corresponding
            # browse and thumbprint PNG files are already in the target directory.
            # It's easiest to run 'aws s3 sync' on original bucket and then fix granules from
            # the newly synced target into original granule location

            # bucket = boto3.resource('s3').Bucket(target_bucket)
            # source_ext = '.nc'

            # for target_ext in ['.png', '_thumb.png']:
            #     # It's an extra file to transfer, replace extension
            #     target_key = bucket_granule.replace(source_ext, target_ext)

            #     if FixSentinel1Granules.object_exists(bucket, target_key):
            #         msgs.append(f'WARNING: {bucket.name}/{target_key} already exists, skipping upload')

            #     else:
            #         source_dict = {'Bucket': source_bucket,
            #                        'Key': target_key}

            #         msgs.append(f'Copying {source_dict["Bucket"]}/{source_dict["Key"]} to {bucket.name}/{target_key}')
            #         if not FixSentinel1Granules.DRY_RUN:
            #             bucket.copy(source_dict, target_key)

            return msgs


class FixSentinel1Granules:
    """
    Class to fix V2 ITS_LIVE granules that still use older format of the metadata
    as compared to the latest V2 campaign.
    """
    # Flag if dry run is requested - print information about to be done actions
    # without actually invoking commands.
    DRY_RUN = False

    BUCKET = 'its-live-data'
    BUCKET_DIR = None
    TARGET_DIR = None

    CHUNK_SIZE = 100
    NUM_DASK_WORKERS = 8

    def __init__(self, granules_list: list, start_index: int = 0, stop_index: int = -1):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem()
        self.all_granules = granules_list
        logging.info(f"Number of all granules: {len(self.all_granules)}")

        if start_index != 0 or stop_index != -1:
            # Start index is provided for the granule to begin with
            if stop_index != -1:
                self.all_granules = self.all_granules[start_index:stop_index]

            else:
                self.all_granules = self.all_granules[start_index:]

        logging.info(f"Starting with granule #{start_index} (stop={stop_index}), remains {len(self.all_granules)} granules to fix")

        # Exclude granules previously fixed: the ones that have suffix
        # self.all_granules = [each for each in self.all_granules if 'LC08' in each]

    def __call__(self, local_dir: str):
        """
        Fix ITS_LIVE granules which are stored in AWS S3 bucket.
        """
        num_to_fix = len(self.all_granules)

        if num_to_fix == 0:
            logging.info("Nothing to fix, exiting.")
            return

        if not FixSentinel1Granules.DRY_RUN and not os.path.exists(local_dir):
            os.mkdir(local_dir)

        start = 0

        while num_to_fix > 0:
            num_tasks = FixSentinel1Granules.CHUNK_SIZE if num_to_fix > FixSentinel1Granules.CHUNK_SIZE else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks} ({num_to_fix} remaining)")
            tasks = [
                dask.delayed(fix_all)(
                    FixSentinel1Granules.BUCKET,
                    FixSentinel1Granules.BUCKET_DIR,
                    FixSentinel1Granules.TARGET_DIR,
                    each,
                    local_dir,
                    self.s3
                ) for each in self.all_granules[start:start+num_tasks]
            ]

            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=FixSentinel1Granules.NUM_DASK_WORKERS
                )

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size',
        type=int,
        default=100, help='Number of granules to fix in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket that stores ITS_LIVE granules to fix [%(default)s]'
    )
    parser.add_argument(
        '-d', '--bucket_dir',
        type=str,
        default='velocity_image_pair/sentinel1-backup/v02',
        help='AWS S3 directory that stores the granules [%(default)s]'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox_sentinel1',
        help='Directory to store fixed granules before uploading them to the S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '-t', '--target_dir',
        type=str,
        default='velocity_image_pair/sentinel1/v02',
        help='AWS S3 bucket directory to store fixed ITS_LIVE granules [%(default)s]'
    )
    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=8,
        help='Number of Dask parallel workers [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not apply any fixes to the granules stored in AWS S3 bucket'
    )
    parser.add_argument(
        '--granule_list',
        type=str,
        default='used_granules.json',
        help='Read granule file list to avoid time consuming glob of S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '-start_index',
        action='store',
        type=int,
        default=0,
        help="Start index for the granule to fix [%(default)d]. "
            "Useful if need to continue previously interrupted process to fix the granules."
    )
    parser.add_argument(
        '-stop_index',
        action='store',
        type=int,
        default=-1,
        help="Stop index for the granules to fix [%(default)d]. "
            "Usefull if need to split the job between multiple processes."
    )

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    if not os.path.exists(args.local_dir):
        os.mkdir(args.local_dir)

    FixSentinel1Granules.DRY_RUN = args.dryrun
    FixSentinel1Granules.BUCKET = args.bucket
    FixSentinel1Granules.BUCKET_DIR = args.bucket_dir
    FixSentinel1Granules.TARGET_DIR = args.target_dir
    FixSentinel1Granules.CHUNK_SIZE = args.chunk_size
    FixSentinel1Granules.NUM_DASK_WORKERS = args.dask_workers

    # Read in granule file list from S3 file
    granule_list = None
    granule_list_file = os.path.join(args.bucket, args.bucket_dir, args.granule_list)

    logging.info(f"Opening granules file: {granule_list_file}")
    with s3fs.S3FileSystem().open(granule_list_file, 'r') as granule_fhandle:
        granule_list = json.load(granule_fhandle)
        logging.info(f"Loaded {len(granule_list)} granules from '{granule_list_file}'")

    fix_metadata = FixSentinel1Granules(
        granule_list,
        args.start_index,
        args.stop_index
    )
    fix_metadata(args.local_dir)


if __name__ == '__main__':
    main()

    logging.info("Done.")
