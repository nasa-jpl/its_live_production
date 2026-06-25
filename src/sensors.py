""" sensors module.

   Mission and sensor definitions for ITS_LIVE data processing.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorGroup:
   """
   Data class to represent all unique sensor groups that contribute to the
   ITS_LIVE datasets.
   """
   # List of sensor identifiers belonging to the mission
   sensors: list

   # Group name
   label: str

   # Unique identifier for the mission/sensor group. Used for filtering
   # and processing (faster to compare int values than strings).
   id: int

   # Mission identifier (single letter as in datacube mission_img1 field)
   # 'L' for Landsat, 'S' for Sentinel, 'N' for NISAR
   mission: str


# If datacube contains only numeric sensor values (Landsat8 or Landsat9),
# sensor values are of type float, otherwise sensor values are of string
# type. Need to support both.
LANDSAT45 = SensorGroup(['4.', '5.', '4.0', '5.0', 4.0, 5.0, '4', '5'],
                        'L4_L5', 4, 'L')

LANDSAT7 = SensorGroup(['7.', '7.0', 7.0, '7'], 'L7', 7, 'L')

LANDSAT89 = SensorGroup(['8.', '9.', '8.0', '9.0', 8.0, 9.0, '8', '9'],
                        'L8_L9', 8, 'L')

# ATTN: '1' and '2' are added as a workaround for the stripped
# satellite_img[12] values when Zarr writes first chunk of the datacube
# with less than 2 characters per sensor values
SENTINEL1 = SensorGroup(['1A', '1B', '1C', '1D', '1'], 'S1', 11, 'S')
SENTINEL2 = SensorGroup(['2A', '2B', '2C', '2D', '2'], 'S2', 21, 'S')

# NISAR uses sensor IDs '1' and '2' which overlap with Sentinel '1' and '2'
# Composite (mission, sensor) keys resolve this conflict
NISAR = SensorGroup(['1', '2', 1, 2, 1.0, 2.0], 'NISAR', 31, 'N')

# TODO: update with new missions groups as they become available
# to be included into datacubes
ALL_GROUPS = {
   LANDSAT45.id: LANDSAT45,
   LANDSAT7.id: LANDSAT7,
   LANDSAT89.id: LANDSAT89,
   SENTINEL1.id: SENTINEL1,
   SENTINEL2.id: SENTINEL2,
   NISAR.id: NISAR
}


def _groups():
   """
   Return mapping of (mission, sensor) tuple to corresponding sensor group ID.

   This method builds mapping using composite keys to handle cases where
   multiple missions use the same sensor IDs. Mission IDs are single letters
   as stored in datacube mission_img1 field.

   Returns dictionary with (mission_id, sensor_id) tuple keys:
      {
         ('L', '4.'):  4,   ('L', '4.0'): 4,   ('L', '4'): 4,   ('L', 4.0): 4,
         ('L', '5.'):  4,   ('L', '5.0'): 4,   ('L', '5'): 4,   ('L', 5.0): 4,
         ('L', '7.'):  7,   ('L', '7.0'): 7,   ('L', '7'): 7,   ('L', 7.0): 7,
         ('L', '8.'):  8,   ('L', '8.0'): 8,   ('L', '8'): 8,   ('L', 8.0): 8,
         ('L', '9.'):  8,   ('L', '9.0'): 8,   ('L', '9'): 8,   ('L', 9.0): 8,
         ('S', '1A'): 11,   ('S', '1B'): 11,   ('S', '1C'): 11, ('S', '1'): 11,
         ('S', '2A'): 21,   ('S', '2B'): 21,   ('S', '2C'): 21, ('S', '2'): 21,
         ('N', '1'):  31,   ('N', '2'):  31,
         ...
      }

   Note: Composite keys resolve sensor ID conflicts between missions:
   - ('S', '1') → 11 (Sentinel-1) vs ('N', '1') → 31 (NISAR)
   - ('S', '2') → 21 (Sentinel-2) vs ('N', '2') → 31 (NISAR)
   """
   all_sensors = {}

   for each_group in ALL_GROUPS.values():
      for each_sensor in each_group.sensors:
         all_sensors[(each_group.mission, each_sensor)] = each_group.id

   return all_sensors


def _groups_labels():
   """
   Return mapping of group ID to its corresponding sensor group name
   as stored in the "label" attribute of the group.

   This method builds mapping of the individual sensor to the group
   it belongs to:
      {
         4: 'L4_L5',
         7: 'L7',
         8: 'L8_L9',
         11: 'S1',
         21: 'S2',
         31: 'NISAR'  # TODO: run by Alex to confirm the label
      }
   """
   all_ids = {}
   for each_group in ALL_GROUPS.values():
      all_ids[each_group.id] = each_group.label

   return all_ids


# Mapping using (mission, sensor) composite keys for unique identification
GROUPS = _groups()

# Mapping of mission group to the corresponding string label
GROUPS_LABELS = _groups_labels()
