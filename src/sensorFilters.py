""" sensors module.

Mission and sensor definitions with filtering capabilities for ITS_LIVE data
processing.

Authors:
Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Mark Fahnestock (UAF)
"""
import logging
import os
from itslive_mosaics_types import CompositeVars
import numpy as np
import xarray as xr
from itscube_types import Vars, ImgPairInfo
import sensors


# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


class SensorExcludeFilter:
   """
   Filter to identify sensor groups to exclude based on reference data quality.

   Compares sensor performance across the timeseries at each spatial point of
   the datacube and excludes underperforming sensors where a reference sensor
   provides superior data quality.

   It allows:
   * To remove L8 and S1 (and possibly other missions) data over very narrow
   glaciers where S2 outperforms. It uses S2 data as a reference data.

   * To exclude S2 data in areas of low constrast with very little stable
   terrain (i.e. ice sheet interiors) during second step in the LSQ fit applied
   to all but S2 data).
   """
   # Minimum required number of values in a bin for one sensorgroup to
   # compute statistics.
   MIN_COUNT = 3

   # Longest dt to use for all sensor groups.
   MAX_DT = 64

   # Reference sensor group to compare other sensor groups to.
   # ATTN: this variable serves two purposes and has opposite meaning for two
   # filters it's used in:
   #
   # 1. The first exclude filter (implemented by this SensorExcludeFilter class)
   # is designed to remove L8 and S1 (and possibly other mission) data over
   # very narrow glaciers where S2 outperforms.
   #
   # 2. The second filter (second step in LSQ fit applied to all but S2 data)
   # is designed to exclude S2 data in areas of low contrast with very little
   # stable terrain (i.e. ice sheet interiors)
   REF_SENSOR = sensors.SENTINEL2

   # Multiplier of standard error to use in comparison
   SESCALE = 3

   def __init__(
      self,
      acquisition_start_time,
      acquisition_stop_time,
      sensors_ids,
      sensors_groups
   ):
      """
      Initialize object.

      Inputs:
      =======
      acquisition_start_time: Acquisition datetime (as decimal year) for the
                        first image of each granule in spatial point's
                        timeseries.
      acquisition_stop_time: Acquisition datetime (as decimal year) for the
                        second image of each granule in spatial point's
                        timeseries.
      sensors_ids:      Sensor groups IDs for the spatial point's timeseries
                        (correspond to the datacube layers).
      sensors_groups:   List of identified sensor groups in timeseries
                        (correspond to the datacube layers).
      """
      # Flag if filter should be applied to timeseries
      self.apply = False

      # Flag if there should be second LSQ fit based on all data except for S2
      # (this is done to exclude trouble S2 data from composites:
      # if (amp_all) > (S1+L8_amp) * 2 then use lsqfit_annual output from S1+L8
      # and add S2 to the excluded sensors mask
      self.excludeS2FromLSQ = False

      self.binedges = None

      # Mapping of each sensor to its mission group ID
      self.sensors_ids = sensors_ids

      # Identify if reference sensor group is present in timeseries
      if SensorExcludeFilter.REF_SENSOR in sensors_groups:
         logging.info(
            f'Reference sensor {SensorExcludeFilter.REF_SENSOR.label} '
            'is present'
         )

         # Is there any other than reference (S2) data that would need
         # to be checked against the reference data
         if len(sensors_groups) > 1:
            self.excludeS2FromLSQ = True
            self.apply = True

            # Extract start and end dates that correspond to the sensor
            # group
            mask = np.isin(self.sensors_ids,
                           SensorExcludeFilter.REF_SENSOR.id)

            start_date = np.array(acquisition_start_time)[mask]
            stop_date = np.array(acquisition_stop_time)[mask]

            logging.info(
               f'Identified reference "{SensorExcludeFilter.REF_SENSOR.label}" '
               f'sensor group: start_date={start_date.min().date()} '
               f'end_date={stop_date.max().date()}'
            )
            self.binedges = np.arange(start_date.min().date(),
                                       stop_date.max().date(),
                                       # 73 D is 1/5 of a year
                                       np.timedelta64(73, '[D]'),
                                       dtype="datetime64[D]")
            logging.info(f'Bin edges: {self.binedges}')

         else:
            logging.info(
               'There is no other than '
               f'{SensorExcludeFilter.REF_SENSOR.label} data present, '
               'disable SensorExcludeFilter and 2nd LSQ fit.'
            )

      else:
         logging.info(
            f'Reference sensor {SensorExcludeFilter.REF_SENSOR.label} '
            'is missing, disable SensorExcludeFilter and 2nd LSQ fit.'
         )

   @staticmethod
   def map_sensor_to_group(all_sensors: list):
      """
      Map each of the granule's first sensor to the mission group it belongs to.

      Inputs:
      =======
      all_sensors: Sensor for the first image in all granules of the cube.
      """
      # Map each sensor to its mission group ID
      return np.array([sensors.GROUPS[x] for x in all_sensors])

   @staticmethod
   def identify_sensor_groups(sensors_ids: list):
      """
      Identify unique sensors within provided set and collect mission groups
      these sensors belong to: to know which missions are represented by
      the set.

      Inputs:
      =======
      sensors_ids: List of sensors groups IDs (that correspond to the datacube
         layers).

      Returns:
      ========
      List of mission groups that correspond to provided sensors groups IDs.
      """
      unique_ids = list(set(sensors_ids))

      # Keep values sorted to be consistent
      unique_ids.sort()
      logging.info(
         f'Identified unique sensor groups: '
         f'{[sensors.GROUPS_LABELS[each] for each in unique_ids]}'
      )

      # Identify sensor groups that correspond to their IDs
      return [sensors.ALL_GROUPS[each] for each in unique_ids]

   def __call__(
      self,
      ds_date_dt,
      ds_vx,
      ds_vy,
      ds_mid_date,
      ds_land_ice_mask
   ):
      """
      Invoke filter for the block of spacial points.

      Inputs:
      =======
      ds_date_dt:       Date separation b/w image pairs for spacial points.
      ds_vx:            X component of velocity for the spacial points.
      ds_vy:            Y component of velicity for the spacial points.
      ds_mid_date:      Middle date for the spacial points.
      ds_land_ice_mask: 2km inbuffer land ice mask for spacial points.
         SensorExcludeFilter should only be applied if land_ice 2km
         inbuffer mask == 0.

      Returns:
      ========
      Array of lists of sensors to exclude per each spacial point.
      """
      y_len, x_len, _ = ds_vx.shape
      dims = (y_len, x_len)
      # logging.info(
      #    f'Applying SensorExcludeFilter to the block of spacial points with '
      #    f'dimensions {dims}...'
      # )
      exclude_sensors = np.frompyfunc(set, 0, 1)(np.empty(dims, dtype=object))

      if self.apply:
         # SensorExcludeFilter should only be applied if land_ice 2km
         # inbuffer mask == 0. Find such indices in data.
         if ds_land_ice_mask is not None:
            valid_mask_ind = np.argwhere(ds_land_ice_mask == 0)

            for each_index in valid_mask_ind:
               j_index = each_index[0]
               i_index = each_index[1]

               exclude_sensors[j_index, i_index] = self.iteration(
                  ds_date_dt,
                  ds_vx[j_index, i_index, :],
                  ds_vy[j_index, i_index, :],
                  ds_mid_date
               )

         else:
            # Apply filter to all points
            for j_index in range(0, y_len):
               for i_index in range(0, x_len):
                  exclude_sensors[j_index, i_index] = self.iteration(
                     ds_date_dt,
                     ds_vx[j_index, i_index, :],
                     ds_vy[j_index, i_index, :],
                     ds_mid_date
                  )

      return exclude_sensors

   def iteration(self, ds_date_dt, ds_vx, ds_vy, ds_mid_date):
      """
      Returns list of sensor groups to exclude based on the timeseries for
      the spacial point.

      Inputs:
      =======
      ds_date_dt: date_dt timeseries for spacial point
      ds_sensors: individual sensors timeseries for spacial point
      ds_vx:      vx timeseries
      ds_vy:      vy timeseries
      ds_mid_date: mid_date timeseries

      Outputs:
      ========
      list of sensor groups to exclude for the spacial point.
      """
      sensors_to_exclude = set()

      trimmed_index = ((ds_date_dt <= SensorExcludeFilter.MAX_DT)
                        & (~np.isnan(ds_vx)))

      # If no data left, exit the filter
      if np.sum(trimmed_index) == 0:
         return sensors_to_exclude

      vx = ds_vx[trimmed_index]
      vy = ds_vy[trimmed_index]
      sensor = self.sensors_ids[trimmed_index]
      mid_dates = ds_mid_date[trimmed_index]

      # get unique sensorgroup id's
      sensorgroups = set(sensor)

      if SensorExcludeFilter.REF_SENSOR.id not in sensorgroups:
         return sensors_to_exclude

      # Do this as a dict of dicts so we can use sensor id as index
      bindicts = {
         each_sensor: {
            'vbin': np.nan * np.ones((len(self.binedges) - 1)),
            'vstdbin': np.nan * np.ones((len(self.binedges) - 1)),
            'vcountbin': np.zeros((len(self.binedges) - 1), dtype='int32')
         } for each_sensor in sensorgroups
      }

      # For each sensor group
      for sen in sensorgroups:
         ind = np.isin(sensor, sen)
         vx0 = np.mean(vx[ind])
         vy0 = np.mean(vy[ind])
         sen_mid_dates = mid_dates[ind]
         v0 = np.sqrt(np.power(vx0, 2.0) + np.power(vy0, 2.0))

         uv = np.array([vx0 / v0, vy0 / v0])
         vp = uv.dot(np.vstack((vx[ind], vy[ind])))

         # Do the bin stats here rather than in a separate function -
         # "return" values populate bindicts
         for bin_num, (be_lo, be_hi) in enumerate(zip(self.binedges[:-1],
                                                      self.binedges[1:])):
            bin_ind = (sen_mid_dates >= be_lo) & (sen_mid_dates < be_hi)
            # these are still numpy.array's - .item() returns sigular
            # value instead of array(value)
            num_in_bin = np.sum(bin_ind).item()

            if num_in_bin >= SensorExcludeFilter.MIN_COUNT:
               bindicts[sen]['vcountbin'] = num_in_bin
               bindicts[sen]['vbin'][bin_num] = np.mean(vp[bin_ind])
               bindicts[sen]['vstdbin'][bin_num] = np.std(vp[bin_ind])

      # Check if reference filter made it into the bindicts:
      refsensor = SensorExcludeFilter.REF_SENSOR.id
      if refsensor not in bindicts:
         return sensors_to_exclude

      stats = {each_sensor: {} for each_sensor in sensorgroups}

      for sen in sensorgroups:
         # No need to check on reference sensor
         if sen == refsensor:
            continue

         covalid = (~np.isnan(bindicts[refsensor]['vbin'])) & \
                     (~np.isnan(bindicts[sen]['vbin']))

         if sum(covalid) > 3:
            delta = \
               bindicts[sen]['vbin'][covalid] - \
               bindicts[refsensor]['vbin'][covalid]

            stats[sen]['mean'] = np.mean(delta)
            stats[sen]['se'] = np.std(delta) / np.sqrt((sum(covalid) - 1))

            # TODO: Should use absolute difference for sigma comparison?
            stats[sen]['disagree_with_refsensor'] = \
               (stats[sen]['mean']
               + (stats[sen]['se'] * SensorExcludeFilter.SESCALE)) < 0

            if stats[sen]['disagree_with_refsensor']:
               sensors_to_exclude.add(sen)

      return sensors_to_exclude


class StableShiftFilter:
   """
   Class to implement stable shift filter for the datacube data.
   It excludes granules that don't pass the filter criteria.

   The class is also responsible for excluding all but specific mission group
   granules if such option is provided to the composite generation code. This
   is to isolate granule exclusion to one place (one can't just drop a
   "mid_date" dimension values for the whole cube xr.Dataset since originally
   created cubes don't have unique values for the dimension - to be fixed for
   another run of the datacube generation).

   stable_shift filter prototype code is:

   if (max(abs(vx_stable_shift), abs(vy_stable_shift)) .* date_dt./365.25) >
         threshold
      if stable_shift_flag == 1
         exclude image pair

      else if stable_shift_flag == 2
         vx += vx_stable_shift
         vy += vy_stable_shift
      end
   end

   Explanation:

   1. If the stable_shift is very large, and stable_shift_flag == 1, then we
      exclude the image pair from our composite.
   2. If the stable_shift_flag == 2, then simply remove stable_shift.
      The correction is subtracted in autoRIFT, so we have to add it back.

   The shift is very large over surface we are "not confident"
   (stable_shift_flag=2) about, so we decided to remove the stable_shift
   (reverse it as compared to the granules).
   """
   # Thresholds for stable_shift filter
   THRESHOLD = {
      sensors.LANDSAT45.id: np.inf,
      sensors.LANDSAT7.id: np.inf,
      sensors.LANDSAT89.id: 61.6,
      sensors.SENTINEL1.id: 1.1,
      sensors.SENTINEL2.id: 28.5
   }

   DEC_YEAR_LEN = 365.25

   # If mission group is provided, then only the granules for this group
   # should be included into composites.
   KEEP_MISSION_GROUP = None

   # Optional list of missions to exclude from composites.
   EXCLUDE_MISSION_GROUP = None

   def __init__(self, ds: xr.Dataset):
      """
      Initialize the filter.

      Inputs:
      =======
      ds: xarray.Dataset containing the datacube.
      """
      cube_sensors = ds[ImgPairInfo.satellite_img1].values

      sensor_list = SensorExcludeFilter.map_sensor_to_group(cube_sensors)
      logging.info(f'Total number of sensors in the cube: {sensor_list.size}')

      # Mask of granules that need their vx and vy readjusted by
      # their corresponding stable_shift value
      self.reverse_stable_shift_mask = np.zeros_like(sensor_list, dtype=bool)
      self.num_reverse_stable_shift_mask = 0

      # Mask of granules that need to be included into composite computations
      self.keep_granule_mask = np.ones_like(sensor_list, dtype=bool)
      self.num_exclude_granules = 0

      # stable_shift values that need to be applied to vx and vy: keep only the
      # values that correspond to the granule mask that need the adjustment
      self.vx_stable_shift = None
      self.vy_stable_shift = None

      # Populate threshold vector with values based on the sensor group
      # each image pair belongs to
      self.threshold = np.zeros_like(sensor_list, dtype=float)

      # Step through all mission groups present in the datacube
      for each_group in SensorExcludeFilter.identify_sensor_groups(sensor_list):
         mask = np.isin(sensor_list, each_group.id)

         if StableShiftFilter.KEEP_MISSION_GROUP and \
               each_group.id != StableShiftFilter.KEEP_MISSION_GROUP.id:
            # Disable other than requested mission group
            self.keep_granule_mask[mask] = False
            self.num_exclude_granules += np.sum(mask)
            logging.info(
               f'Need to exclude {np.sum(mask)} granules for '
               f'{sensors.GROUPS_LABELS[each_group.id]} group '
               f'as requested by user.'
            )

         if StableShiftFilter.EXCLUDE_MISSION_GROUP and \
                  each_group.id in StableShiftFilter.EXCLUDE_MISSION_GROUP:
            # Disable requested mission group
            self.keep_granule_mask[mask] = False
            self.num_exclude_granules += np.sum(mask)
            logging.info(
               f'Need to exclude {np.sum(mask)} granules for '
               f'{sensors.GROUPS_LABELS[each_group.id]} group '
               f'as requested by user.'
            )

         # Set threshold for all mission related granules
         self.threshold[mask] = StableShiftFilter.THRESHOLD[each_group.id]

      # Make sure all missions are encountered for when setting the threshold,
      # if not then need to update StableShiftFilter.THRESHOLD
      zero_mask = (self.threshold == 0)
      if np.any(zero_mask):
         # There are non populated missions in the dataset, raise an exception
         unique_values = set(sensor_list[zero_mask])
         raise RuntimeError(
            f'Need to set stable_shift threshold for {unique_values} '
            f'sensor groups in StableShiftFilter.THRESHOLD.'
         )

      # Set filtering masks based on the datacube data and the filter criteria
      self._setMasks(ds)


   def _setMasks(self, cube_ds: xr.Dataset):
      """
      Set up stable_shift filter for the datacube.

      This method identifies which granules need to be excluded from composites
      based on the stable_shift filter criteria and which granules need to have
      their vx and vy values readjusted by their corresponding stable_shift
      value.

      To apply the filter, call the apply() method with vx and vy data as
      inputs, or call the exclude() method with data to exclude granules
      from an input for other than vx and vy data variables.

      Inputs:
      =======
      cube_ds: xarray.Dataset that represents the datacube.
      """
      # Don't need to do anything about va_stable_shift and vr_stable_shift
      date_dt = cube_ds[ImgPairInfo.date_dt].values
      self.vx_stable_shift = cube_ds[Vars.vx_stable_shift].values

      # Some older cubes inherit NaN's from granules for stable_shift
      # attribute, set them to zero
      nan_mask = np.isnan(self.vx_stable_shift)
      self.vx_stable_shift[nan_mask] = 0

      self.vy_stable_shift = cube_ds[Vars.vy_stable_shift].values
      nan_mask = np.isnan(self.vy_stable_shift)
      self.vy_stable_shift[nan_mask] = 0

      max_values = np.maximum(
         np.abs(self.vx_stable_shift),
         np.abs(self.vy_stable_shift)
      ) * date_dt / StableShiftFilter.DEC_YEAR_LEN

      filter_mask = np.greater(max_values, self.threshold)

      if np.any(filter_mask):
         stable_shift = cube_ds[Vars.flag_stable_shift].values

         # ATTN: need to apply stable_shift first, if any, then exclude the
         # granules, if any, as they all use the full dataset length for
         # masking

         # Need to revert stable_shift adjustment if stable_shift == 2
         _mask = (stable_shift == 2) & filter_mask & self.keep_granule_mask
         if np.any(_mask):
            # Add back corresponding stable_shift
            self.reverse_stable_shift_mask[_mask] = True
            self.num_reverse_stable_shift_mask = np.sum(_mask)

            self.vx_stable_shift = self.vx_stable_shift[_mask]
            self.vy_stable_shift = self.vy_stable_shift[_mask]

            # Since vx and vy are 3d data variables, need to reshape
            # the stable_shift values to the same 3d dimensions
            self.vx_stable_shift = self.vx_stable_shift.reshape(
               (self.num_reverse_stable_shift_mask, 1, 1)
            )
            self.vy_stable_shift = self.vy_stable_shift.reshape(
               (self.num_reverse_stable_shift_mask, 1, 1)
            )

            # # Update vx and vy values as we process each chunk of datacube
            # data
            # vx_stable_shift = np.broadcast_to(
            #     self.vx_stable_shift,
            #     (np.sum(self.reverse_stable_shift_mask), x_len, y_len)
            #     )
            # vx[self.reverse_stable_shift_mask] += self.vx_stable_shift
            #
            # # Update vx in dataset
            # ds[DataVars.VX].loc[dict(x=ds.x, y=ds.y, mid_date=ds.mid_date)]
            #  = vx

         # Exclude the granule if stable_shift == 1
         _mask = (stable_shift == 1) & filter_mask & self.keep_granule_mask
         if np.any(_mask):
            self.keep_granule_mask[_mask] = False

            # If only specific mission group is used, then some of the granules
            # might be set to be excluded already. Get the number of
            # total excluded granules in the mask.
            self.num_exclude_granules = np.sum(
               np.isin(self.keep_granule_mask, False)
            ).item()

            # DEBUG: pandas.errors.InvalidIndexError: Reindexing only valid
            # with uniquely valued Index objects:
            # There are duplicates of mid_date values in some datacubes,
            # so can't use xr.Dataset.drop_isel()
            # Solution: to mask each of the data variables required for
            # composite generation by keeping only the values masked by
            # self.keep_granule_mask

            # Remove granules if any
            # result_ds = cube_ds.drop_isel(mid_date=_mask_index)


   def exclude(self, data):
      """
      Exclude granules, if any are detected by the filter, from the data.

      ATTN: We had to introduce this method because of the
      "pandas.errors.InvalidIndexError: Reindexing only valid with uniquely
      valued Index objects" exception we are getting if calling
      xr.Dataset.drop_isel()
      for the datacubes which have layers with duplicates of "mid_date"
      values. Can restore original implementation once all datacubes are
      regenerated with unique "mid_date" values.

      Inputs:
      =======
      data: Data to exclude granules from.

      Returns:
      ========
      Updated or original data if no exclusions are required.
      """
      return_data = data
      if self.num_exclude_granules > 0:
         return_data = data[self.keep_granule_mask]

      return return_data


   def apply(self, vx, vy):
      """
      Apply stable_shift corrections to the datacube's vx and vy variables
      and remove excluded granules if any.

      Inputs:
      =======
      vx: VX data
      vy: VY data

      Returns:
      ========
      Updated vx and vy data or original data if no corrections are required.
      """
      return_vx = vx.copy()
      return_vy = vy.copy()

      if self.num_reverse_stable_shift_mask > 0:
         _, y_len, x_len = vx.shape

         # Update vx and vy values as we process each chunk of datacube data
         vx_stable_shift = np.broadcast_to(
            self.vx_stable_shift,
            (self.num_reverse_stable_shift_mask, y_len, x_len)
         )
         return_vx[self.reverse_stable_shift_mask] += vx_stable_shift

         vy_stable_shift = np.broadcast_to(
            self.vy_stable_shift,
            (self.num_reverse_stable_shift_mask, y_len, x_len)
         )
         return_vy[self.reverse_stable_shift_mask] += vy_stable_shift

      if self.num_exclude_granules > 0:
         # Exclude some of the granules
         return_vx = return_vx[self.keep_granule_mask, :, :]
         return_vy = return_vy[self.keep_granule_mask, :, :]

      return (return_vx, return_vy)


def get_cube_data(cube_ds: xr.Dataset, stable_shift_filter: StableShiftFilter):
   """
   Prepare datacube data for the filter application.

   This method is to be used to prepare datacube data for the filter application
   by keeping only the variables required for the filter application and
   sorting cube layers by date_dt (this is only important when we start applying
   the filters and need to make sure that the data is in the correct
   chronological order when creating composites).

   Inputs:
   =======
   cube_ds             xarray.Dataset: datacube.
   stable_shift_filter  StableShiftFilter: filter that is initialized based on
      the datacube data and is used to filter the datacube for the
      SensorExcludeFilter application.

   Returns:
   ========
   Prepared datacube data for the filter application.
   """
   # Sensor data for the cube's layers: map each sensor to its group ID
   sensors_ids = SensorExcludeFilter.map_sensor_to_group(
      stable_shift_filter.exclude(
         cube_ds[ImgPairInfo.satellite_img1].values
      )
   )
   # Identify sensors groups (L89, S1, S2, etc.) within datacube.
   sensors_groups = SensorExcludeFilter.identify_sensor_groups(sensors_ids)

   # Images acquisition times
   datetime_img1 = [
      t.astype('M8[ms]').astype('O') for t in
      stable_shift_filter.exclude(
         cube_ds[ImgPairInfo.acquisition_date_img1].values
      )
   ]
   datetime_img2 = [
      t.astype('M8[ms]').astype('O') for t in
      stable_shift_filter.exclude(
         cube_ds[ImgPairInfo.acquisition_date_img2].values
      )
   ]

   return  datetime_img1, datetime_img2, sensors_ids, sensors_groups


# A set of datacube variables that are of interest. Update the list if
# the list of variables is other than specified.
CUBE_VARS = [
   Vars.vx,
   Vars.vy,
   Vars.flag_stable_shift,
   Vars.vx_stable_shift,
   Vars.vy_stable_shift,
   CompositeVars.vx_error,
   CompositeVars.vy_error,
   ImgPairInfo.date_dt,
   ImgPairInfo.date_center,
   ImgPairInfo.acquisition_date_img1,
   ImgPairInfo.acquisition_date_img2,
   ImgPairInfo.satellite_img1,
   ImgPairInfo.mission_img1
]

# Dimensions order of the data to guarantee continuous memory in time dimension
# Original data as stored in [time, y, x] dimension order.
CONT_TIME_ORDER = [1, 2, 0]

# Name of the date center variable in the datacube.
DATE_CENTER = 'date_center'

if __name__ == '__main__':
   """Demonstration code on how to use filters defined in this module.
   """
   import argparse
   import s3fs
   import shapefile

   # Command-line arguments parser
   parser = argparse.ArgumentParser(
      description=__doc__.split('\n')[0],
      formatter_class=argparse.RawDescriptionHelpFormatter
   )
   parser.add_argument(
      '-s3', '--s3Bucket',
      type=str,
      default='s3://its-live-data',
      help="s3 bucket that stores datacube [%(default)s]."
   )
   parser.add_argument(
      '-c', '--inputCube',
      type=str,
      default=None,
      help="Input Zarr datacube store to filter [%(default)s]."
   )
   parser.add_argument(
      '-i', '--iCoordinate',
      type=int,
      default=0,
      help="x index for the spatial point to check the filter application "
            "[%(default)s]."
   )
   parser.add_argument(
      '-j', '--jCoordinate',
      type=int,
      default=0,
      help="y index for the spatial point to check the filter application "
            "[%(default)s]."
   )
   parser.add_argument(
      '-b', '--blockSize',
      type=int,
      default=10,
      help="Chunk size for x and y coordinates of the spatial block to check "
            "the filter application [%(default)s]. For example, if i=0, j=0, "
            "and blockSize=10, then the filter will be applied to the 10x10 "
            "spatial block starting at (0,0)."
   )
   parser.add_argument(
      '-s', '--shapeFile',
      type=str,
      default='s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp',
      help="Shapefile that stores ice masks per each of the EPSG codes "
            "[%(default)s]."
   )

   args = parser.parse_args()

   # Test with only ['S1A_S1B'] sensors - nothing to excluded:
   # Total number of sensors in the cube: 556
   # s3://its-live-project/test_datacubes/sensorFilter/ITS_LIVE_vel_EPSG32717_G0120_X750000_Y9750000.zarr

   # Test with ['L8_L9', 'S1A_S1B', 'S2A_S2B'] sensors -
   # Total number of sensors in the cube: 1017
   # StableShiftFilter set to exclude 298 granules
   # s3://its-live-data/datacubes/v2-updated-october2024/S60E090/ITS_LIVE_vel_EPSG3031_G0120_X2750000_Y-350000.zarr
   input_cube = args.inputCube
   logging.info(f"Reading existing {input_cube}...")

   # Read dataset in
   ds = None
   # Keep s3 store open to read the datacube data
   cube_store = None
   if len(args.s3Bucket) == 0:
      # S3 bucket is not provided, read datacube from local path
      logging.info(f"Reading {input_cube} datacube from local path...")
      ds = xr.open_zarr(input_cube, decode_timedelta=False, consolidated=True)

   else:
      # S3 bucket is provided, read datacube from the S3 bucket
      cube_path = os.path.join(args.s3Bucket, input_cube)
      logging.info(f"Reading {cube_path} datacube from S3 bucket...")

      # Open S3FS access to S3 bucket with input datacube
      s3_in = s3fs.S3FileSystem(skip_instance_cache=True)
      cube_store = s3fs.S3Map(root=cube_path, s3=s3_in, check=False)
      ds = xr.open_dataset(cube_store, decode_timedelta=False,
                           engine='zarr', consolidated=True)

   # Keep only variables of interest and sort cube layers by date_dt
   # (this is only important when we start applying the filters and need to
   # make sure that the data is in the correct chronological order when
   # creating composites)
   cube_ds = ds[CUBE_VARS].sortby(ImgPairInfo.date_dt)

   # Load shapefile with ice mask information required for SensorExcludeFilter
   # processing.
   logging.info(f"Loading shapefile {args.shapeFile}...")
   shape_ds = shapefile.read_file(args.shapeFile)

   # Read land ice mask from the shape file
   # ------------------------------------------------------------------------
   x = ds.x.values
   y = ds.y.values

   cube_projection = int(cube_ds.attrs[utils.OutputFormat.projection])
   land_ice_mask, _ = shapefile.read_ice_mask(shape_ds, shapefile.LANDICE_2KM,
                                                x, y, cube_projection)

   # Initialize stable shift filter according to the cube data
   stable_shift_filter = StableShiftFilter(cube_ds)

   logging.info(f'StableShiftFilter set to exclude '
                  f'{stable_shift_filter.num_exclude_granules} granules')

   # As an example, select a spatial block to check the filter application:
   # vx and vy values should be updated according to the filter criteria.
   # If there are no granules to exclude, then the values should be the same
   # as original.
   j = args.jCoordinate
   i = args.iCoordinate
   block = args.blockSize
   logging.info(
      f'Apply filters to the datacube block defined by i={i}:{i+block} and '
      f'j={j}:{j+block}...'
   )

   # This can be a loop over all spatial blocks of the datacube, but for
   # demonstration purposes we are just applying the filter to one spatial
   # block defined by i, j, and block size.
   logging.info(f'Loading vx and vy data for the selected spatial block...')
   vx = cube_ds[Vars.vx].values[:, j:j + block, i:i + block]
   vy = cube_ds[Vars.vy].values[:, j:j + block, i:i + block]

   # Land ice mask is already cropped to the datacube polygon
   land_ice_mask = None if land_ice_mask is None else \
                     land_ice_mask[j:j + block, i:i + block]

   # Apply stable shift filter to vx and vy data for the selected spatial
   # block: besides excluding some of the layers, the vx and vy values will
   # get variable values readjusted by stable_shift value if
   # stable_shift_flag == 2
   logging.info(f'Applying StableShiftFilter to vx and vy...')
   vx, vy = stable_shift_filter.apply(vx, vy)

   # An example of how to apply stable shift filter to other than vx and vy
   # data variables of the cube
   logging.info(f'Apply StableShiftFilter to vx_error and vy_error...')
   vx_error= stable_shift_filter.exclude(cube_ds.vx_error.values)
   vy_error= stable_shift_filter.exclude(cube_ds.vy_error.values)

   #-------------------------------------------------------------------------
   # Now apply SensorExcludeFilter
   cube_data = get_cube_data(cube_ds, stable_shift_filter)

   # Initialize sensor exclusion filter
   sensor_filter = SensorExcludeFilter(*cube_data)

   # Day separation between images
   dt = stable_shift_filter.exclude(
      cube_ds[ImgPairInfo.date_dt].values
   )

   date_center = stable_shift_filter.exclude(cube_ds[DATE_CENTER].values)
   mid_date_len = len(date_center)

   # Transpose data to make it continuous in time
   vxt = np.zeros((block, block, mid_date_len))
   vxt.flat = np.transpose(vx, CONT_TIME_ORDER)

   vyt = np.zeros((block, block, mid_date_len))
   vyt.flat = np.transpose(vy, CONT_TIME_ORDER)

   # Filter returns an array of lists of sensors to exclude per each spacial
   # point in the block.
   exclude_sensors = sensor_filter(dt, vxt, vyt, date_center, land_ice_mask)

   # Each spacial point in the block has a list of sensors to exclude.
   # For demonstration purposes, we are just printing the list of sensors to
   # exclude for the first spacial point in the block only.
   for index, element in np.ndenumerate(exclude_sensors):
      logging.info(
         f'List of sensors to exclude for spacial point with index {index}: '
         f'{element}'
      )

   logging.info('Done.')
