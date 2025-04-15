"""This tool is used to generate range-range velocity granules from the given Zarr datacube.

It iterates over all S1 datacube layers and determines which are ascending and descending granules.
It then generates range-range velocity granules for each ascending and descending granule pair.
A pair is qualified for the range-range velocity granule generation if the following conditions are met:

   1. Have granule image pair separations <= 24 days
   2. The mid-dates between ascending and descending granules <= 24 days apart
   3. Large overlap between ascending and descending granules. We consider the overlap
      to be large if the overlap is >= 60% of the granule size for both ascending and descending granules.

Please refer to the github issue https://github.com/nasa-jpl/its_live_production/issues/40 for
more details on the format definition of range-range velocity granules.

If json files that store lists of ascending and descending granuler are provided, the tool will
only generate range-range velocity granules for the granules listed in these json files.
"""
import argparse
import boto3
import datetime
from dateutil.parser import parse
import dask
from dask.diagnostics import ProgressBar
import gc
import s3fs
import json
import logging
import numpy as np
import os
import pandas as pd
import pyproj
import sys
import xarray as xr
import warnings

from mission_info import Encoding
from itslive_composite import SensorExcludeFilter, MissionSensor
from itscube import ITSCube
import grid
from itscube_types import \
   Coords, \
   DataVars, \
   Output, \
   CubeOutput, \
   MappingInstance

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

NC_ENGINE = 'h5netcdf'

ascending = "ascending"
descending = "descending"

# Cell dimensions
x_cell = 120.0
half_x_cell = x_cell/2.0
y_cell = -120.0
half_y_cell = y_cell/2.0


def create_new_granule(x, y):
   """
   Create xr.Dataset to represent new range-range velocity granule.

   Input:
   x: x-coordinates
   y: y-coordinates

   Returns:
   xr.Dataset: New range-range velocity granule.
   """
   granule = xr.Dataset(
      coords={
         'y': (
               'y',
               # sorted(y),
               y,
               {
                  DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                  DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
               }
         ),
         'x': (
               'x',
               # sorted(x),
               x,
               {
                  DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                  DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
               }
         )
      }
   )
   return granule


def compute_velocity_based_on_m11_m12(
   m11_a,
   m12_a,
   m11_d,
   m12_d,
   vr_a,
   vr_d,
   dr_to_vr_factor_a,
   dr_to_vr_factor_d,
   trans1,
   trans2):
   """
   Compute velocity fields in map coordinates based on LOS (slant range)
   measurements of pixel displacement.

   vr_a and vr_b can be scalar values or 2D arrays.
   """
   ysize1, xsize1 = m11_a.shape
   ysize2, xsize2 = m11_d.shape

   W = np.min([trans1[0], trans2[0]])
   N = np.max([trans1[3], trans2[3]])
   E = np.max([trans1[0] + (xsize1 - 1)*trans1[1], trans2[0] + (xsize2 - 1)*trans2[1]])
   S = np.min([trans1[3] + (ysize1 - 1)*trans1[5], trans2[3] + (ysize2 - 1)*trans2[5]])

   trans = [W, trans1[1], 0.0, N, 0.0, trans1[5]]
   xsize = int(np.round((E - W) / trans[1])) + 1
   ysize = int(np.round((S - N) / trans[5])) + 1

   dims = (ysize, xsize)

   vr_1 = np.full(dims, np.nan)
   M11_1 = np.full(dims, np.nan)
   M12_1 = np.full(dims, np.nan)

   vr_2 = np.full(dims, np.nan)
   M11_2 = np.full(dims, np.nan)
   M12_2 = np.full(dims, np.nan)

   x1a = int(np.round((trans1[0] - W) / trans1[1]))
   x1b = x1a + xsize1
   y1a = int(np.round((trans1[3] - N) / trans1[5]))
   y1b = y1a + ysize1

   x2a = int(np.round((trans2[0] - W) / trans2[1]))
   x2b = x2a + xsize2
   y2a = int(np.round((trans2[3] - N) / trans2[5]))
   y2b = y2a + ysize2

   vr_1[y1a:y1b, x1a:x1b] = vr_a
   M11_1[y1a:y1b, x1a:x1b] = m11_a
   M12_1[y1a:y1b, x1a:x1b] = m12_a

   vr_2[y2a:y2b, x2a:x2b] = vr_d
   M11_2[y2a:y2b, x2a:x2b] = m11_d
   M12_2[y2a:y2b, x2a:x2b] = m12_d

   scale_factor = M11_1*M12_2 - M12_1*M11_2
   zero_mask = (scale_factor == 0)
   scale_factor[zero_mask] = np.nan

   vx = (M12_2 * vr_1 / dr_to_vr_factor_a - M12_1 * vr_2 / dr_to_vr_factor_d) / scale_factor
   vy = (-M11_2 * vr_1 / dr_to_vr_factor_a + M11_1 * vr_2 / dr_to_vr_factor_d) / scale_factor

   return vx, vy, xsize, ysize, trans


def get_granule_image_pairs(pd_d_orbits, pd_a_orbits, threshold=0.6, num_days=24):
   """Find qualifying pairs for range-range velocities.

   Args:
      pd_d_orbits (pd.DataFrame): DataFrame containing descending granules.
      pd_a_orbits (pd.DataFrame): DataFrame containing ascending granules.
      threshold (float, optional): Minimum spacial overlap for ascending and descending
         granules to qualify for the range-range pair. Defaults to 0.6 (60%).
      num_days (int, optional): Maximum number of days between date_center of
         ascending and descending granules. Defaults to 24.
   """
   # Find overlap in time
   overlap_df = pd.DataFrame(columns=['d_url', 'a_url', 'xy_overlap_percent', 'x', 'y'])

   max_overlap = 0

   # Step through descending orbits
   for index1, row1 in pd_d_orbits.iterrows():
      d_mid_date = row1[DataVars.ImgPairInfo.DATE_CENTER]

      start_date = d_mid_date - datetime.timedelta(days=num_days)
      end_date = d_mid_date + datetime.timedelta(days=num_days)

      num_x = len(row1['x'])
      num_y = len(row1['y'])

      total_num_xy = num_x * num_y
      # print(f'total area: {total_num_xy}')

      found_time_overlap = pd_a_orbits[
         pd_a_orbits.date_center.apply(
            lambda x: all([pd.Timestamp(start_date) <= x, x <= pd.Timestamp(end_date)])
         )
      ]

      # Find overlap in x/y extends
      if len(found_time_overlap):
         for index2, row2 in found_time_overlap.iterrows():
            # Intersection of descending with ascending
            x_intersection = set(row1['x']).intersection(row2['x'])
            y_intersection = set(row1['y']).intersection(row2['y'])

            total_num_intersection = len(x_intersection) * len(y_intersection)
            total_num_intersection_percent = total_num_intersection/total_num_xy

            if total_num_intersection_percent < threshold:
               continue

            # Intersection of ascending with descending - if ascending granule
            # had bigger coverage, make sure we have maximum coverage based
            # on both granules spatial limits
            num_a_x = len(row2['x'])
            num_a_y = len(row2['y'])

            total_a_num_xy = num_a_x * num_a_y

            x_ad_intersection = set(row2['x']).intersection(row1['x'])
            y_ad_intersection = set(row2['y']).intersection(row1['y'])

            total_ad_num_intersection = len(x_ad_intersection) * len(y_ad_intersection)
            total_ad_num_intersection_percent = total_ad_num_intersection/total_a_num_xy

            # If using only descending/ascending % coverage
            # common_overlap = total_num_intersection_percent
            # If using both overlap thresholds: d->a and a->d
            common_overlap = min(total_num_intersection_percent, total_ad_num_intersection_percent)

            # If using only descending/ascending % coverage
            # max_overlap = max(max_overlap, total_num_intersection_percent)

            # If using both % coverage: descending/ascending and ascending/descending
            max_overlap = max(max_overlap, common_overlap)

            # if len(x_intersection) > 1 and len(y_intersection) > 1 and total_num_intersection_percent > threshold:
            if len(x_intersection) > 1 and len(y_intersection) > 1 and \
               common_overlap > threshold:
               overlap_df = overlap_df.append(
                  {
                     'd_url': row1.url,
                     'a_url': row2.url,
                     'xy_overlap_percent': common_overlap
                  },
                  ignore_index=True
               )

   logging.info(f'Maximum % overlap detected: {max_overlap*100}')
   return overlap_df


def v_error_zero_velocity_fill(vx_error, vy_error):
   """
   This function is derived from the autoRIFT v_error_cal()
   to compute fill value for the cells where computed velocity
   is zero.

   See: https://github.com/nasa-jpl/autoRIFT/blob/249e6d03afa84597091f56dca7f5d8fce37be026/netcdf_output.py#L28
   """
   return np.std(np.sqrt(vx_error**2 + vy_error**2))


def process_pair(bucket: str, target_bucket_dir: str, row: pd.Series, s3: s3fs.S3FileSystem):
   """Generate range-range velocity granules for the given
      ascending and descending granule pair.

   Args:
      bucket (str):  s3 bucket name.
      target_bucket_dir (str): s3 bucket directory.
      row (pd.Series): row containing ascending and descending granule urls and metadata about both granules.
      s3 (s3fs.FileSystem): s3 filesystem object.
   """
   # Don't crop for now
   crop_to_valid_extends = False

   input_granules = {}

   input_granules_names = {
      descending: row.d_url,
      ascending: row.a_url
   }

   msgs = []
   for each_orbit, each_granule in input_granules_names.items():
      msgs.append(f'Opening {each_orbit} {each_granule=}')

      # Load granule into memory
      with s3.open(each_granule, mode='rb') as fhandle:
         with xr.open_dataset(fhandle, engine=NC_ENGINE) as granule_ds:
            input_granules[each_orbit] = granule_ds.load()

   # Descending values
   d_ds_m11 = input_granules[descending].M11.values
   d_ds_m12 = input_granules[descending].M12.values
   d_ds_vr = input_granules[descending].vr.values
   d_dr_to_vr_factor = input_granules[descending].M11.attrs[DataVars.DR_TO_VR_FACTOR]
   d_trans = np.array(str.split(input_granules[descending][MappingInstance.name].GeoTransform)).astype(float)

   # Ascending values
   a_ds_m11 = input_granules[ascending].M11.values
   a_ds_m12 = input_granules[ascending].M12.values
   a_ds_vr = input_granules[ascending].vr.values
   a_dr_to_vr_factor = input_granules[ascending].M11.attrs[DataVars.DR_TO_VR_FACTOR]
   a_trans = np.array(str.split(input_granules[ascending][MappingInstance.name].GeoTransform)).astype(float)

   vx, vy, x_size, y_size, trans = compute_velocity_based_on_m11_m12(
      a_ds_m11, a_ds_m12,
      d_ds_m11, d_ds_m12,
      a_ds_vr, d_ds_vr,
      a_dr_to_vr_factor, d_dr_to_vr_factor,
      a_trans, d_trans
   )

   x_min = trans[0] + half_x_cell
   y_max = trans[3] + half_y_cell

   x_max = x_min + x_cell * x_size
   y_min = y_max + y_cell * y_size

   x_bounds = grid.Bounds(min_value=x_min, max_value=x_max)
   y_bounds = grid.Bounds(min_value=y_min, max_value=y_max)

   # Create grid for new granule
   grid_x, grid_y = grid.Grid.create(x_bounds, y_bounds, x_cell)

   # Ensure lonlat output order when computing centroid long/lat coordinates
   projection = str(int(input_granules[descending][MappingInstance.name].attrs[MappingInstance.attrs.spatial_epsg]))

   to_lon_lat_transformer = pyproj.Transformer.from_crs(
      f"EPSG:{projection}",
      ITSCube.LON_LAT_PROJECTION,
      always_xy=True
   )

   center_x = (grid_x.min() + grid_x.max())/2
   center_y = (grid_y.min() + grid_y.max())/2

   # Convert to lon/lat coordinates
   center_lon_lat = to_lon_lat_transformer.transform(center_x, center_y)

   #     # dt is the time between granule center dates
   #     date_diff = date_center[0] - date_center[1]
   #     # Calculate total days in decimal form
   #     total_days = abs(date_diff.days) + date_diff.seconds / 86400 + date_diff.microseconds / 86400 / 1e6

   #     print(f'total_days: {total_days}')

   two_granules = create_new_granule(grid_x, grid_y)
   two_granules.attrs = input_granules[ascending].attrs

   # Set date_created attribute
   two_granules.attrs[CubeOutput.DATE_CREATED] = datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
   del two_granules.attrs[CubeOutput.DATE_UPDATED]

   two_granules['vx'] = xr.DataArray(
      vx,
      coords=two_granules.coords,
      attrs=input_granules[ascending].vx.attrs
   )
   two_granules['vy'] = xr.DataArray(
      vy,
      coords=two_granules.coords,
      attrs=input_granules[ascending].vy.attrs
   )

   # Reset error_* and stable_shift_* attributes for newly computed vx and vy to Nan
   for each_var in ['vx', 'vy']:
      for each_attr in [
         'error_modeled',
         'error_slow',
         'error_stationary',
         'stable_shift',
         'stable_shift_slow',
         'stable_shift_stationary'
      ]:
         two_granules[each_var].attrs[each_attr] = np.nan

   # Set the mapping data variable
   two_granules[MappingInstance.name] = input_granules[ascending].mapping

   # Set new GeoTransform
   two_granules[MappingInstance.name].attrs[MappingInstance.attrs.geo_transform] = ' '.join([str(each) for each in trans])

   # Crop result granule
   # TODO: crop granule to valid vx values
   cropped_ds = two_granules

   if crop_to_valid_extends:
      xy_ds = two_granules.where(two_granules.vx.notnull(), drop=True)

      x_values = xy_ds.x.values
      grid_x_min, grid_x_max = x_values.min(), x_values.max()

      y_values = xy_ds.y.values
      grid_y_min, grid_y_max = y_values.min(), y_values.max()

      # Based on X/Y extends, mask original dataset
      mask_lon = (two_granules.x >= grid_x_min) & (two_granules.x <= grid_x_max)
      mask_lat = (two_granules.y >= grid_y_min) & (two_granules.y <= grid_y_max)
      mask = (mask_lon & mask_lat)

      cropped_ds = two_granules.where(mask, drop=True)

      # TODO: Recalculate GeoTransform

   # Add interp_mask: take from descending granule
   x_values = two_granules.x.values
   grid_x_min, grid_x_max = x_values.min(), x_values.max()

   y_values = two_granules.y.values
   grid_y_min, grid_y_max = y_values.min(), y_values.max()

   # Based on X/Y extends, mask original granule's interp_mask values
   mask_lon = (input_granules[descending].x >= grid_x_min) & (input_granules[descending].x <= grid_x_max)
   mask_lat = (input_granules[descending].y >= grid_y_min) & (input_granules[descending].y <= grid_y_max)
   mask = (mask_lon & mask_lat)

   cropped_interp_mask = input_granules[descending].interp_mask.where(mask, drop=True)

   # Mask out vx's NaNs in interp_mask
   mask = two_granules.vx.isnull()

   # Apply mask to interp_mask: invert mask to keep non-NaN values
   cropped_interp_mask = cropped_interp_mask.where(~mask)

   # Set new granule's interp_mask to the cropped granule's interp_mask
   two_granules[DataVars.INTERP_MASK] = cropped_interp_mask
   two_granules[DataVars.INTERP_MASK].attrs = input_granules[descending].interp_mask.attrs

   # Attributes for the CHIP_SIZE_HEIGHT
   # Per Alex:
   # We are only using "chip_size_width:range_pixel_size"...
   # so maybe we take the maximum "chip_size_width:range_pixel_size" and
   # use that to populate both "chip_size_height" and chip_size_width"
   max_range_pixel_size = 0.0
   for each_var in [DataVars.CHIP_SIZE_WIDTH, DataVars.CHIP_SIZE_HEIGHT]:
      var_values = [
         input_granules[descending][each_var],
         input_granules[ascending][each_var]
      ]
      # Align the arrays to have the same coordinates
      da1_aligned, da2_aligned = xr.align(*var_values, join="inner")

      # max_da = xr.ufuncs.fmax(da1_aligned, da2_aligned)   # Ignores NaN's
      max_da = xr.ufuncs.maximum(da1_aligned, da2_aligned)  # Propagates NaN's

      # Set variable in new range-range granule
      cropped_ds[each_var] = max_da

      # Copy all existing attributes
      cropped_ds[each_var].attrs = var_values[0].attrs

      # chip_size_width is set to be processed first, so will collect maximum
      # attribute value among ascending and descending granules
      if 'range_pixel_size' in cropped_ds[each_var].attrs:
         max_range_pixel_size = max(
               var_values[0].attrs['range_pixel_size'],
               var_values[1].attrs['range_pixel_size']
         )

         cropped_ds[each_var].attrs['range_pixel_size'] = max_range_pixel_size

      else:
         # Set chip_size_height:azimuth_pixel_size to the same max_range_pixel_size
         cropped_ds[each_var].attrs['azimuth_pixel_size'] = max_range_pixel_size

   # Create img_pair_info data variable
   cropped_ds[DataVars.ImgPairInfo.NAME] = input_granules[descending].img_pair_info
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs = {}

   # Populate "img_pair_info" attributes and collect acquisition dates for both granules (4 images)
   # to format filename for the new granule
   filename_dates = {
      ascending: [],
      descending: []
   }

   # acquisition_date_img1 - Yan's granules have 'acquisition_img1/2'
   # [use minimum acquisition_date]
   date_values = [
      parse(input_granules[descending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1]),
      parse(input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1])
   ]
   # msgs.append(f'Got acquisition_date_img1={date_values}')

   filename_dates[descending].append(date_values[0].strftime('%Y%m%dT%H%M%S'))
   filename_dates[ascending].append(date_values[1].strftime('%Y%m%dT%H%M%S'))

   # Assume first value is minimum
   min_value_index = descending

   if date_values[0] > date_values[1]:
      min_value_index = ascending

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1] = \
      input_granules[min_value_index][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1]

   # acquisition_date_img2
   # [use maximum acquisition_date]
   date_values = [
      parse(input_granules[descending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2]),
      parse(input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2])
   ]
   # msgs.append(f'Got acquisition_date_img2={date_values}')

   filename_dates[descending].append(date_values[0].strftime('%Y%m%dT%H%M%S'))
   filename_dates[ascending].append(date_values[1].strftime('%Y%m%dT%H%M%S'))

   # Assume first value is maximum
   max_value_index = descending

   if date_values[0] < date_values[1]:
      max_value_index = ascending

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2] = \
      input_granules[max_value_index][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2]

   # date_center
   # [average date_center]
   # Find the midpoint
   date1 = parse(input_granules[descending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER])
   date2 = parse(input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER])
   msgs.append(f'Got date_center={date1, date2} (descending, ascending)')

   earlier_date = min(date1, date2)
   later_date = max(date1, date2)
   mid_date = earlier_date + (later_date - earlier_date) / 2

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.DATE_CENTER] = mid_date.strftime('%d-%b-%YT%H:%M:%S.%f')

   # Determine which granule has earlier date_center - that's granule #1
   granule_1 = descending
   granule_2 = ascending

   if earlier_date == date2:
      granule_1 = ascending
      granule_2 = descending

   msgs.append(f'Got {granule_1=} {granule_2=}')

   # date_dt
   # [not sure about this one] - dt between date_center's?

   # roi_valid_percentage
   # [not sure about this one]

   # satellite_img1
   # [just "1"]
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SATELLITE_IMG1] = '1'

   # satellite_img2
   # [just "1"]
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SATELLITE_IMG2] = '1'

   # sensor_img1
   # [granule 1 sensor_img1]
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG1] = \
      input_granules[granule_1][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG1]

   # sensor_img2
   # [granule 2 sensor_img1]
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG2] = \
      input_granules[granule_2][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.SENSOR_IMG1]

   # MISSION_IMG1/2 - always 'S'
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_IMG1] = \
      input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_IMG1]

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_IMG2] = \
      input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_IMG1]

   # flight_direction_img1/2
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG1] = \
      input_granules[descending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG1]

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG2] = \
      input_granules[ascending][DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.FLIGHT_DIRECTION_IMG2]

   # id_img1/2
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ID_IMG1] = input_granules_names[granule_1]
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ID_IMG2] = input_granules_names[granule_2]

   # Compute centroid longitude/latitude
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[CubeOutput.LATITUDE] = round(center_lon_lat[1], 2)
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[CubeOutput.LONGITUDE] = round(center_lon_lat[0], 2)

   ########################################################################
   # Hard-coded values since they are the same for all range-range granules
   ########################################################################
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.PRODUCT_UNIQUE_ID_IMG1] = 'N/A'
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.PRODUCT_UNIQUE_ID_IMG2] = 'N/A'

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_DATA_TAKE_ID_IMG1] = 'N/A'
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.MISSION_DATA_TAKE_ID_IMG2] = 'N/A'

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ABSOLUTE_ORBIT_NUMBER_IMG1] = 'N/A'
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ABSOLUTE_ORBIT_NUMBER_IMG2] = 'N/A'

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.STD_NAME] = 'image_pair_information'

   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.TIME_STANDARD_IMG1] = 'UTC'
   cropped_ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.TIME_STANDARD_IMG2] = 'UTC'

   # Example of the attributes:
   # string img_pair_info:absolute_orbit_number_img1 = "006692" ;
   # string img_pair_info:absolute_orbit_number_img2 = "007042" ;
   # string img_pair_info:acquisition_date_img1 = "20150706T15:31:54.422322" ;
   # string img_pair_info:acquisition_date_img2 = "20150730T15:31:55.977067" ;
   # string img_pair_info:date_center = "20150718T15:31:55.199694" ;
   # img_pair_info:date_dt = 24.0000179947338 ;
   # string img_pair_info:flight_direction_img1 = "descending" ;
   # string img_pair_info:flight_direction_img2 = "descending" ;
   # string img_pair_info:id_img1 = "S1A_IW_SLC__1SSV_20150706T153140_20150706T153207_006692_008F2F_894B" ;
   # string img_pair_info:id_img2 = "S1A_IW_SLC__1SSV_20150730T153142_20150730T153209_007042_00992B_CBD8" ;
   # img_pair_info:latitude = 60.11 ;
   # img_pair_info:longitude = -137.55 ;
   # string img_pair_info:mission_data_take_ID_img1 = "008F2F" ;
   # string img_pair_info:mission_data_take_ID_img2 = "00992B" ;
   # string img_pair_info:mission_img1 = "S" ;
   # string img_pair_info:mission_img2 = "S" ;
   # string img_pair_info:product_unique_ID_img1 = "894B" ;
   # string img_pair_info:product_unique_ID_img2 = "CBD8" ;
   # img_pair_info:roi_valid_percentage = 83.3 ;
   # string img_pair_info:satellite_img1 = "1A" ;
   # string img_pair_info:satellite_img2 = "1A" ;
   # string img_pair_info:sensor_img1 = "C" ;
   # string img_pair_info:sensor_img2 = "C" ;
   # string img_pair_info:standard_name = "image_pair_information" ;
   # string img_pair_info:time_standard_img1 = "UTC" ;
   # string img_pair_info:time_standard_img2 = "UTC" ;

   # Add "v" data variable
   v = np.sqrt(vx**2 + vy**2)
   # There are no numeric values for any of the 'v' attributes, just copy from input granule
   cropped_ds[DataVars.V] = xr.DataArray(
      data=v,
      coords=two_granules.coords,
      attrs=input_granules[descending].v.attrs
   )

   # Compute vx_error and vy_error based on vr.vr_error of input granules.
   # These are scalar values, so populate a_vr_ds and d_vr_ds with that value
   # (to make sure we use the same non-NaN masking as vr data in original granules)
   # Yan's granules have vr.vr_error vs. current S1 granules have vr.error attributes
   # print(
   #    f"Got original granules vr.vr_error: descending={input_granules[descending].vr.attrs['error']} " \
   #    f"ascending={input_granules[ascending].vr.attrs['error']}")

   d_vr_error = input_granules[descending].vr.attrs[DataVars.ERROR]
   a_vr_error = input_granules[ascending].vr.attrs[DataVars.ERROR]

   vx_error, vy_error, x_size, y_size, trans = compute_velocity_based_on_m11_m12(
      a_ds_m11, a_ds_m12,
      d_ds_m11, d_ds_m12,
      a_vr_error, d_vr_error,
      a_dr_to_vr_factor, d_dr_to_vr_factor,
      a_trans, d_trans
   )

   vx_error = np.abs(vx_error)
   vy_error = np.abs(vy_error)

   v_error_rand = v_error_zero_velocity_fill(vx_error, vy_error)
   v_error = np.sqrt((vx_error * vx / v)**2 + (vy_error * vy / v)**2)
   v_error[v == 0] = v_error_rand
   no_data_mask = np.where(np.isnan(vx_error))
   v_error[no_data_mask] = np.nan

   # Add v_error to new dataset
   cropped_ds[DataVars.V_ERROR] = xr.DataArray(
      data=v_error,
      coords=cropped_ds.vx.coords,
      attrs=input_granules[descending].v_error.attrs
   )

   # Write granule to local file
   granule_name = 'S1_' + '_'.join(filename_dates[granule_1]) + '_X_' + '_'.join(filename_dates[granule_2]) + '_range.nc'

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

   msgs.append(f'Writing granule to {granule_name=}')
   cropped_ds.to_netcdf(granule_name, engine=NC_ENGINE, encoding=granule_encoding)

   # Upload granule to S3
   s3_client = boto3.client('s3')

   bucket_granule = os.path.join(target_bucket_dir, granule_name)

   msgs.append(f"Uploading to {os.path.join(bucket, target_bucket_dir)}")
   s3_client.upload_file(granule_name, bucket, bucket_granule)

   os.unlink(granule_name)

   return msgs


def get_granule_info(granule_ds):
   """
   Collect granule information.
   """
   img_pair_attrs = granule_ds.img_pair_info.attrs

   return (
      img_pair_attrs['flight_direction_img1'].strip(),
      {
            DataVars.ImgPairInfo.DATE_CENTER: parse(img_pair_attrs[DataVars.ImgPairInfo.DATE_CENTER]),
            'date_dt': img_pair_attrs['date_dt'],
            'lat': img_pair_attrs['latitude'],
            'lon': img_pair_attrs['longitude'],
            'x': granule_ds.x.values,
            'y': granule_ds.y.values
      }
   )


def process_granule(granule: str, s3=None):
   """
   Extract S1 granule information for range-range velocity calculations.
   """
   # Open the granule
   each_granule_s3 = granule.replace('https://', '')
   each_granule_s3 = each_granule_s3.replace('.s3.amazonaws.com', '')

   with s3.open(each_granule_s3, mode='rb') as fhandle:
      with xr.open_dataset(fhandle, engine=NC_ENGINE) as granule_ds:
         return (granule,) + get_granule_info(granule_ds)


if __name__ == '__main__':
   # Initialize CL arguments parser
   parser = argparse.ArgumentParser(description='Generate range-range velocity granules from the given Zarr datacube.')

   parser.add_argument(
      '-i', '--input',
      type=str,
      help='Path to the input Zarr datacube.'
   )
   parser.add_argument(
      '-chunk_size',
      action='store',
      type=int,
      default=10,
      help='Number of granules to process in parallel [%(default)d]'
   )
   parser.add_argument(
      '-b', '--bucket',
      type=str,
      default='its-live-data',
      help='AWS S3 bucket that stores ITS_LIVE V2 granules [%(default)s]'
   )
   parser.add_argument(
      '-t', '--target_bucket_dir',
      type=str,
      default='test-space/range_range',
      help='AWS S3 bucket and directory to store range-range velocity granules to [%(default)s]'
   )
   parser.add_argument(
      '-w', '--dask-workers',
      type=int,
      default=8,
      help='Number of Dask parallel workers [%(default)d]'
   )

   args = parser.parse_args()

   # Get input zarr cube
   input_cube_name = os.path.basename(args.input)

   # S3 args
   target_bucket = args.bucket
   target_bucket_dir = args.target_bucket_dir

   # Format filename for the ascending granule json file
   ascending_json = f'ascending_orbits_{input_cube_name}.json'
   ascending_orbit = {}

   if os.path.exists(ascending_json):
      with open(ascending_json, 'r') as fh:
         ascending_orbit = json.load(fh)

      logging.info(f'{len(ascending_orbit)} ascending granules are provided.')

   # Format filename for the descending granule json file
   descending_json = f'descending_orbits_{input_cube_name}.json'
   descending_orbit = {}

   if os.path.exists(descending_json):
      with open(descending_json, 'r') as fh:
         descending_orbit = json.load(fh)

      logging.info(f'{len(descending_orbit)} descending granules are provided.')

   if len(ascending_orbit) == 0 or len(descending_orbit) == 0:
      logging.info('No ascending or descending granules are provided. Need to generate them first.')

      # Search for ascending and descending granules in datacube
      # Read local cube
      datacube_zarr = args.input
      logging.info(f'Reading {datacube_zarr} layers...')

      granule_urls = None
      s3 = s3fs.S3FileSystem()

      with xr.open_dataset(datacube_zarr, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
         logging.info(f'Cube dimensions: {ds.dims}')

         # Identify S1 layers within the cube
         sensors = ds[DataVars.ImgPairInfo.SATELLITE_IMG1].values
         sensors_str = SensorExcludeFilter.map_sensor_to_group(sensors)

         s1_mask = (sensors_str == MissionSensor.SENTINEL1.mission)
         total_num_files = np.sum(s1_mask)
         logging.info(f'Identified {total_num_files} S1 layers in the cube')

         mask_i = np.where(s1_mask == True)
         granule_urls = ds.granule_url[mask_i].values
         logging.info(f'Got {len(granule_urls)=} urls: first granule={granule_urls[0]}')

         # Current start index into list of S1 granules to process
         start = 0
         chunk_size = args.dask_workers

         total_num_files = len(granule_urls)
         init_total_files = len(granule_urls)

         while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            # logging.info(f"Starting layers {start}:{start+num_tasks} out of {init_total_files} total layers")
            tasks = [
               dask.delayed(process_granule)(each_granule_url, s3)
               for each_granule_url in granule_urls[start:start+num_tasks]
            ]

            results = None

            with ProgressBar():
               # Display progress bar
               results = dask.compute(
                     tasks,
                     scheduler="processes",
                     num_workers=args.dask_workers
               )

            for each_result in results[0]:
               each_granule, orbit_dir, result_dict = each_result
               orbit = ascending_orbit
               # If orbit is descending than populate corresponding dict
               if orbit_dir == descending:
                  orbit = descending_orbit

               orbit[each_granule] = result_dict

            total_num_files -= num_tasks
            start += num_tasks

            _ = gc.collect()

      # Report number of found granules
      logging.info(f'Got {len(ascending_orbit)=}')
      logging.info(f'Got {len(descending_orbit)=}')

      # Save ascending and descending granules to json files
      # Convert datetime objects to string as json can't serialize datetime,
      # and convert x and y arrays to lists
      for each_key, each_val in ascending_orbit.items():
         each_val[DataVars.ImgPairInfo.DATE_CENTER] = each_val[DataVars.ImgPairInfo.DATE_CENTER].strftime("%Y-%m-%d %H:%M:%S.%f")
         each_val[Coords.X] = list(each_val[Coords.X])
         each_val[Coords.Y] = list(each_val[Coords.Y])

      with open(ascending_json, 'w') as fh:
         json.dump(ascending_orbit, fh, indent=3)

      # Convert datetime objects to string as json can't serialize datetime
      for each_key, each_val in descending_orbit.items():
         each_val[DataVars.ImgPairInfo.DATE_CENTER] = each_val[DataVars.ImgPairInfo.DATE_CENTER].strftime("%Y-%m-%d %H:%M:%S.%f")
         each_val[Coords.X] = list(each_val[Coords.X])
         each_val[Coords.Y] = list(each_val[Coords.Y])

      with open(descending_json, 'w') as fh:
         json.dump(descending_orbit, fh, indent=3)
      logging.info(f'Wrote {ascending_json} and {descending_json} files.')

   # Convert datetime strings back to objects
   for each_key, each_val in ascending_orbit.items():
      each_val[DataVars.ImgPairInfo.DATE_CENTER] = datetime.datetime.strptime(
         each_val[DataVars.ImgPairInfo.DATE_CENTER],
         "%Y-%m-%d %H:%M:%S.%f"
      )

   for each_key, each_val in descending_orbit.items():
      each_val[DataVars.ImgPairInfo.DATE_CENTER] = datetime.datetime.strptime(
         each_val[DataVars.ImgPairInfo.DATE_CENTER],
         "%Y-%m-%d %H:%M:%S.%f"
      )

   # Create pandas.DataFrames for easier lookup
   pd_a_orbits = pd.DataFrame(ascending_orbit)
   pd_a_orbits = pd_a_orbits.transpose()

   # Convert old index into a column
   pd_a_orbits.reset_index(inplace=True)

   # Rename original 'index' column to more meaningful
   pd_a_orbits.rename(columns={'index': 'url'}, inplace=True)

   pd_d_orbits = pd.DataFrame(descending_orbit)
   pd_d_orbits = pd_d_orbits.transpose()

   # Convert old index into a column
   pd_d_orbits.reset_index(inplace=True)
   # Rename original 'index' column to more meaningful
   pd_d_orbits.rename(columns={'index': 'url'}, inplace=True)

   # Find qualifying pairs for range-range velocities
   logging.info('Finding qualifying pairs for range-range velocities...')
   overlap_df = get_granule_image_pairs(pd_d_orbits, pd_a_orbits)

   total_pairs = len(overlap_df)
   # total_pairs = 1  # For testing

   if total_pairs == 0:
      logging.info('No qualifying pairs found. Exiting.')
      sys.exit(1)

   logging.info(f'Found {len(overlap_df)} qualifying pairs for range-range velocities.')

   # Remove http tokens from granules names
   overlap_df['d_url'] = overlap_df['d_url'].str.replace(r'https://|\.s3\.amazonaws.com', '', regex=True)
   overlap_df['a_url'] = overlap_df['a_url'].str.replace(r'https://|\.s3\.amazonaws.com', '', regex=True)

   s3 = s3fs.S3FileSystem(anon=False, skip_instance_cache=True)

   # Current start index into list of pairs to process
   start_index = 0
   chunk_size = args.chunk_size

   while total_pairs > 0:
      num_tasks = chunk_size if total_pairs > chunk_size else total_pairs

      logging.info(f"Starting pairs {start_index}:{start_index+num_tasks} out of {total_pairs} total pairs")
      tasks = [
            dask.delayed(process_pair)(target_bucket, target_bucket_dir, each, s3)
            for _, each in overlap_df.iloc[start_index:start_index+num_tasks].iterrows()
      ]
      results = None

      with ProgressBar():
         # Display progress bar
         results = dask.compute(
            tasks,
            scheduler="processes",
            num_workers=args.dask_workers
         )

      for each_result in results[0]:
            if len(each_result):
               # If there are any messages to report
               logging.info("\n-->".join(each_result))

      total_pairs -= num_tasks
      start_index += num_tasks

      gc.collect()

   logging.info('Done.')
