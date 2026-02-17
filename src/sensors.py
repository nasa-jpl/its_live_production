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


# If datacube contains only numeric sensor values (Landsat8 or Landsat9),
# sensor values are of type float, otherwise sensor values are of string
# type. Need to support both.
LANDSAT45 = SensorGroup(['4.', '5.', '4.0', '5.0', 4.0, 5.0, '4', '5'],
                        'L4_L5', 4)

LANDSAT7 = SensorGroup(['7.', '7.0', 7.0, '7'], 'L7', 7)

LANDSAT89 = SensorGroup(['8.', '9.', '8.0', '9.0', 8.0, 9.0, '8', '9'],
                        'L8_L9', 8)

# ATTN: '1' and '2' are added as a workaround for the stripped
# satellite_img[12] values when Zarr writes first chunk of the datacube
# with less than 2 characters per sensor values
SENTINEL1 = SensorGroup(['1A', '1B', '1'], 'S1A_S1B', 11)
SENTINEL2 = SensorGroup(['2A', '2B', '2'], 'S2A_S2B', 21)

# TODO: update with new missions groups as they become available
# to be included into datacubes
ALL_GROUPS = {
   LANDSAT45.id: LANDSAT45,
   LANDSAT7.id: LANDSAT7,
   LANDSAT89.id: LANDSAT89,
   SENTINEL1.id: SENTINEL1,
   SENTINEL2.id: SENTINEL2
}


def _groups():
   """
   Return mapping of sensor to its corresponding sensor group ID.

   This method builds mapping of the individual sensor to the group ID
   it belongs to:
      {
         '4.':  4,
         '5.':  4,
         4.0:   4,
         5.0:   4,
         '4.0': 4,
         '5.0': 4,
         '7.':  7,
         '7.0': 7,
         7.0:   7,
         '8.':  8,
         '9.':  8,
         8.0:   8,
         9.0:   8,
         '8.0': 8,
         '9.0': 8,
         '1':   11,
         '1A':  11,
         '1B':  11,
         '2':   21,
         '2A':  21,
         '2B':  21
      }
   """
   all_sensors = {}

   for each_group in ALL_GROUPS.values():
      for each_sensor in each_group.sensors:
         all_sensors[each_sensor] = each_group.id

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
         11: 'S1A_S1B',
         21: 'S2A_S2B'
      }
   """
   all_ids = {}
   for each_group in ALL_GROUPS.values():
      all_ids[each_group.id] = each_group.label

   return all_ids


# Mapping of all sensors to the corresponding mission group
GROUPS = _groups()

# Mapping of mission group to the corresponding string label
GROUPS_LABELS = _groups_labels()
