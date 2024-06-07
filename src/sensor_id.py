from dataclasses import dataclass


@dataclass(frozen=True)
class UniqueSensorID:
    """
    Data class to represent all unique sensors that contribute to the ITS_LIVE datasets.

    The class associates each sensor with unique identifier:

    4 = Landsat 4
    5 = Landsat 5
    6 = Landsat 6
    7 = Landsat 7
    8 = Landsat 8
    9 = Landsat 9
    11 = Sentinel 1A
    12 = Sentinel 1B
    21 = Sentinel 2A
    22 = Sentinel 2B
    """
    # Unique ID associated with the mission
    id: int

    # Possible values of mission IDs as they appear in ITS_LIVE granules
    # If datacube contains only numeric sensor values (Landsat8 or Landsat9),
    # sensor values are of type float, otherwise sensor values are of string type
    # ---> support both
    sensor_ids: list

    # Description for the mission
    name: str

    def __str__(self):
        """String representation of the object.

        Returns:
            str: String representation of the object.
        """
        return f'{self.name} = {self.id}'


LANDSAT4 = UniqueSensorID(4, ['4.', '4.0', 4.0, '4'], 'Landsat 4')
LANDSAT5 = UniqueSensorID(5, ['5.', '5.0', 5.0, '5'], 'Landsat 5')
LANDSAT7 = UniqueSensorID(7, ['7.', '7.0', 7.0, '7'], 'Landsat 7')
LANDSAT8 = UniqueSensorID(8, ['8.', '8.0', 8.0, '8'], 'Landsat 8')
LANDSAT9 = UniqueSensorID(9, ['9.', '9.0', 9.0, '9'], 'Landsat 9')

SENTINEL1A = UniqueSensorID(11, ['1A'], 'Sentinel 1A')
SENTINEL1B = UniqueSensorID(12, ['1B'], 'Sentinel 1B')
SENTINEL2A = UniqueSensorID(21, ['2A'], 'Sentinel 2A')
SENTINEL2B = UniqueSensorID(22, ['2B'], 'Sentinel 2B')

# TODO: update with new unique sensors as their granules become supported by ITS_LIVE
#       datacubes
ALL_SENSORS = {
    LANDSAT4.name: LANDSAT4,
    LANDSAT5.name: LANDSAT5,
    LANDSAT7.name: LANDSAT7,
    LANDSAT8.name: LANDSAT8,
    LANDSAT9.name: LANDSAT9,
    SENTINEL1A.name: SENTINEL1A,
    SENTINEL1B.name: SENTINEL1B,
    SENTINEL2A.name: SENTINEL2A,
    SENTINEL2B.name: SENTINEL2B
}

def _sensors():
    """
    Return mapping of sensor to its corresponding unique sensor ID.

    This method builds mapping of the individual sensor to the group
    it belongs to:
        {
            '4.':  4,
            4.0:   4,
            4:     4,
            '4.0': 4,
            '5.':  5,
            5.0:   5,
            '5.0': 5,
            '7.':  7,
            '7.0': 7,
            7.0:   7,
            '8.':  8,
            '9.':  9,
            8.0:   8,
            9.0:   9,
            '8.0': 8,
            '9.0': 9,
            '1A':  11,
            '1B':  12,
            '2A':  21,
            '2B':  22
        }
    """
    all_sensors = {}

    for each_group in ALL_SENSORS.values():
        for each_sensor in each_group.sensor_ids:
            all_sensors[each_sensor] = each_group.id

    return all_sensors

# Mapping of sensor to the unique ID
SENSORS = _sensors()

def all_sensors_description():
    """
    Function that returns description string to include all unique sensors and their meanings.
    """
    common_str = 'unique sensor id: '
    desc = [str(each) for each in ALL_SENSORS.values()]

    return common_str + ', '.join(desc)