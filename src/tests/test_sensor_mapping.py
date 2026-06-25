"""
Unit tests for SensorExcludeFilter.map_sensor_to_group() method.

Tests the composite (mission, sensor) key mapping to resolve conflicts where
multiple missions use the same sensor IDs. Specifically tests NISAR sensors
'1' and '2' which overlap with Sentinel-1 and Sentinel-2 sensor IDs.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sensors
from sensorFilters import SensorExcludeFilter


class TestSensorGroupDataclass:
    """Test cases for SensorGroup dataclass structure."""

    def test_landsat_mission_ids(self):
        """Verify all Landsat missions have mission ID 'L'."""
        assert sensors.LANDSAT45.mission == 'L'
        assert sensors.LANDSAT7.mission == 'L'
        assert sensors.LANDSAT89.mission == 'L'

    def test_sentinel_mission_ids(self):
        """Verify all Sentinel missions have mission ID 'S'."""
        assert sensors.SENTINEL1.mission == 'S'
        assert sensors.SENTINEL2.mission == 'S'

    def test_nisar_mission_id(self):
        """Verify NISAR mission has mission ID 'N'."""
        assert sensors.NISAR.mission == 'N'

    def test_nisar_sensor_ids(self):
        """Verify NISAR uses sensor IDs '1' and '2' (not 'A')."""
        assert '1' in sensors.NISAR.sensors
        assert '2' in sensors.NISAR.sensors

    def test_all_group_ids_are_unique(self):
        """Verify all sensor groups have unique group IDs."""
        group_ids = [group.id for group in sensors.ALL_GROUPS.values()]

        # Check no duplicates
        assert len(group_ids) == len(set(group_ids)), \
            f"Duplicate group IDs found: {[gid for gid in group_ids if group_ids.count(gid) > 1]}"

        # Verify expected group IDs are present
        expected_ids = {4, 7, 8, 11, 21, 31}  # L4_L5, L7, L8_L9, S1, S2, NISAR
        actual_ids = set(group_ids)
        assert actual_ids == expected_ids, \
            f"Group IDs mismatch. Expected: {expected_ids}, Got: {actual_ids}"


class TestCompositeKeyMapping:
    """Test cases for composite key mapping in GROUPS dictionary."""

    def test_groups_dict_uses_tuple_keys(self):
        """Verify GROUPS dictionary uses (mission, sensor) tuple keys."""
        # Check a sample of keys are tuples
        for key in list(sensors.GROUPS.keys())[:5]:
            assert isinstance(key, tuple)
            assert len(key) == 2
            mission, _ = key
            assert isinstance(mission, str)

    def test_landsat_composite_keys(self):
        """Test Landsat sensor mappings with composite keys."""
        assert sensors.GROUPS[('L', '4')] == sensors.LANDSAT45.id
        assert sensors.GROUPS[('L', '5')] == sensors.LANDSAT45.id
        assert sensors.GROUPS[('L', '7')] == sensors.LANDSAT7.id
        assert sensors.GROUPS[('L', '8')] == sensors.LANDSAT89.id
        assert sensors.GROUPS[('L', '9')] == sensors.LANDSAT89.id

    def test_sentinel_composite_keys(self):
        """Test Sentinel sensor mappings with composite keys."""
        assert sensors.GROUPS[('S', '1')] == sensors.SENTINEL1.id
        assert sensors.GROUPS[('S', '1A')] == sensors.SENTINEL1.id
        assert sensors.GROUPS[('S', '1B')] == sensors.SENTINEL1.id
        assert sensors.GROUPS[('S', '2')] == sensors.SENTINEL2.id
        assert sensors.GROUPS[('S', '2A')] == sensors.SENTINEL2.id
        assert sensors.GROUPS[('S', '2B')] == sensors.SENTINEL2.id

    def test_nisar_composite_keys(self):
        """Test NISAR sensor mappings with composite keys."""
        assert sensors.GROUPS[('N', '1')] == sensors.NISAR.id
        assert sensors.GROUPS[('N', '2')] == sensors.NISAR.id


class TestSensorConflictResolution:
    """Test cases for resolving sensor ID conflicts between missions."""

    def test_sensor_1_conflict_resolution(self):
        """Test sensor ID '1' correctly maps to different groups for S vs N."""
        sentinel1_group = sensors.GROUPS[('S', '1')]
        nisar_group = sensors.GROUPS[('N', '1')]

        assert sentinel1_group == 11  # Sentinel-1 group
        assert nisar_group == 31      # NISAR group
        assert sentinel1_group != nisar_group

    def test_sensor_2_conflict_resolution(self):
        """Test sensor ID '2' correctly maps to different groups for S vs N."""
        sentinel2_group = sensors.GROUPS[('S', '2')]
        nisar_group = sensors.GROUPS[('N', '2')]

        assert sentinel2_group == 21  # Sentinel-2 group
        assert nisar_group == 31      # NISAR group
        assert sentinel2_group != nisar_group

    def test_numeric_sensor_variants(self):
        """Test various numeric formats of sensor IDs (string, float, etc.)."""
        # Landsat supports multiple formats
        assert sensors.GROUPS[('L', '8')] == sensors.LANDSAT89.id
        assert sensors.GROUPS[('L', '8.')] == sensors.LANDSAT89.id
        assert sensors.GROUPS[('L', '8.0')] == sensors.LANDSAT89.id
        assert sensors.GROUPS[('L', 8.0)] == sensors.LANDSAT89.id


class TestMapSensorToGroup:
    """Test cases for SensorExcludeFilter.map_sensor_to_group() method."""

    def test_basic_mapping(self):
        """Test basic sensor to group mapping with single mission."""
        sensors_list = ['8', '9', '8']
        missions_list = ['L', 'L', 'L']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        expected = np.array([8, 8, 8])  # All Landsat 8/9
        np.testing.assert_array_equal(result, expected)

    def test_mixed_missions_without_conflicts(self):
        """Test mapping with multiple missions but no sensor ID conflicts."""
        sensors_list = ['8', '1A', '2A', '7']
        missions_list = ['L', 'S', 'S', 'L']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        expected = np.array([8, 11, 21, 7])  # L8_L9, S1, S2, L7
        np.testing.assert_array_equal(result, expected)

    def test_overlapping_sensor_ids_resolved_by_mission(self):
        """Test that overlapping sensor IDs are correctly resolved using mission."""
        # Critical test: Same sensor IDs, different missions
        sensors_list = ['1', '1', '2', '2']
        missions_list = ['S', 'N', 'S', 'N']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        # Sentinel-1, NISAR, Sentinel-2, NISAR
        expected = np.array([11, 31, 21, 31])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_sentinel_and_nisar(self):
        """Test realistic scenario with mixed Sentinel and NISAR granules."""
        sensors_list = ['1A', '1B', '2A', '2B', '1', '2', '1', '2']
        missions_list = ['S', 'S', 'S', 'S', 'N', 'N', 'S', 'N']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        # S1, S1, S2, S2, NISAR, NISAR, S1, NISAR
        expected = np.array([11, 11, 21, 21, 31, 31, 11, 31])
        np.testing.assert_array_equal(result, expected)

    def test_all_missions_combined(self):
        """Test mapping with all mission types present."""
        sensors_list = ['4', '7', '8', '9', '1A', '2B', '1', '2']
        missions_list = ['L', 'L', 'L', 'L', 'S', 'S', 'N', 'N']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        # L4_L5, L7, L8_L9, L8_L9, S1, S2, NISAR, NISAR
        expected = np.array([4, 7, 8, 8, 11, 21, 31, 31])
        np.testing.assert_array_equal(result, expected)

    def test_return_type_is_numpy_array(self):
        """Verify the method returns a numpy array."""
        sensors_list = ['8', '1']
        missions_list = ['L', 'S']

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        assert isinstance(result, np.ndarray)

    def test_empty_lists(self):
        """Test handling of empty sensor/mission lists."""
        sensors_list = []
        missions_list = []

        result = SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        assert len(result) == 0
        assert isinstance(result, np.ndarray)


class TestInvalidInputs:
    """Test cases for invalid inputs and error handling."""

    def test_unknown_mission_sensor_combination(self):
        """Test that unknown (mission, sensor) combination raises KeyError."""
        sensors_list = ['Z']  # Non-existent sensor
        missions_list = ['X']  # Non-existent mission

        with pytest.raises(KeyError):
            SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

    def test_mismatched_list_lengths_sensors_longer(self):
        """Test that mismatched list lengths raises ValueError (sensors > missions)."""
        sensors_list = ['8', '9', '7']
        missions_list = ['L', 'L']  # Only two missions for three sensors

        with pytest.raises(ValueError) as exc_info:
            SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        # Verify error message contains helpful information
        assert 'same length' in str(exc_info.value).lower()
        assert '3 sensors' in str(exc_info.value)
        assert '2 missions' in str(exc_info.value)

    def test_mismatched_list_lengths_missions_longer(self):
        """Test that mismatched list lengths raises ValueError (missions > sensors)."""
        sensors_list = ['8']
        missions_list = ['L', 'L', 'S']  # Three missions for one sensor

        with pytest.raises(ValueError) as exc_info:
            SensorExcludeFilter.map_sensor_to_group(sensors_list, missions_list)

        # Verify error message contains helpful information
        assert 'same length' in str(exc_info.value).lower()
        assert '1 sensors' in str(exc_info.value)
        assert '3 missions' in str(exc_info.value)


class TestGroupsLabels:
    """Test cases for GROUPS_LABELS mapping."""

    def test_all_group_ids_have_labels(self):
        """Verify all group IDs have corresponding labels."""
        for group in sensors.ALL_GROUPS.values():
            assert group.id in sensors.GROUPS_LABELS
            assert sensors.GROUPS_LABELS[group.id] == group.label

    def test_specific_labels(self):
        """Verify specific group ID to label mappings."""
        assert sensors.GROUPS_LABELS[4] == 'L4_L5'
        assert sensors.GROUPS_LABELS[7] == 'L7'
        assert sensors.GROUPS_LABELS[8] == 'L8_L9'
        assert sensors.GROUPS_LABELS[11] == 'S1'
        assert sensors.GROUPS_LABELS[21] == 'S2'
        assert sensors.GROUPS_LABELS[31] == 'NISAR'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
