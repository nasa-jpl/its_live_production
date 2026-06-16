"""
Unit tests for ITSCube.extract_mid_date_from_url() method.

Tests the extraction of mid_date from various sensor filename formats:
- Landsat (LC08, LC09, LE07, LT05, etc.)
- NISAR
- Sentinel-1
- Sentinel-2
"""

import pytest
from datetime import datetime
import sys
import os

# Add parent directory to path to import itscube
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itscube import ITSCube


class TestExtractMidDateFromUrl:
    """Test cases for extract_mid_date_from_url() method."""

    def test_landsat8_filename(self):
        """Test Landsat 8 filename format (LC08)."""
        url = "s3://bucket/LC08_L1GT_007011_20130819_20200912_02_T2_X_LC08_L1GT_007011_20140806_20200911_02_T2_G0120V02_P044.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2013-08-19 and 2014-08-06
        expected = datetime(2013, 8, 19) + (datetime(2014, 8, 6) - datetime(2013, 8, 19)) / 2
        assert result == expected

    def test_landsat9_filename(self):
        """Test Landsat 9 filename format (LC09)."""
        url = "LC09_L1TP_013010_20220101_20220102_02_T1_X_LC09_L1TP_013010_20220201_20220202_02_T1_G0120V02_P044.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2022-01-01 and 2022-02-01
        expected = datetime(2022, 1, 1) + (datetime(2022, 2, 1) - datetime(2022, 1, 1)) / 2
        assert result == expected

    def test_landsat7_filename(self):
        """Test Landsat 7 filename format (LE07)."""
        url = "LE07_L1TP_012010_20130627_20200907_02_T1_X_LE07_L1TP_012010_20130727_20200907_02_T1_G0120V02_P003.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2013-06-27 and 2013-07-27
        expected = datetime(2013, 6, 27) + (datetime(2013, 7, 27) - datetime(2013, 6, 27)) / 2
        assert result == expected

    def test_nisar_filename(self):
        """Test NISAR filename format."""
        url = "NISAR_L1_PR_RSLC_005_149_D_074_2005_QPDH_A_20251120T130632_20251120T130707_X05009_N_F_J_001_X_NISAR_L1_PR_RSLC_006_149_D_074_4005_DHDH_A_20251202T130633_20251202T130707_X05009_N_F_J_001_G0120V02_P095.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2025-11-20 13:06:32 and 2025-12-02 13:06:33
        date1 = datetime(2025, 11, 20, 13, 6, 32)
        date2 = datetime(2025, 12, 2, 13, 6, 33)
        expected = date1 + (date2 - date1) / 2
        assert result == expected

    def test_sentinel1_filename(self):
        """Test Sentinel-1 filename format."""
        url = "S1A_IW_SLC__1SDH_20200221T095209_20200221T095237_031349_039B9A_CBD7_X_S1B_IW_SLC__1SDH_20200227T095113_20200227T095143_020453_026C13_6119_G0120V02_P098.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2020-02-21 09:52:09 and 2020-02-27 09:51:13
        date1 = datetime(2020, 2, 21, 9, 52, 9)
        date2 = datetime(2020, 2, 27, 9, 51, 13)
        expected = date1 + (date2 - date1) / 2
        assert result == expected

    def test_sentinel2_filename(self):
        """Test Sentinel-2 filename format."""
        url = "S2B_MSIL1C_20181008T190459_N0206_R127_T02CMU_20181008T232024_X_S2B_MSIL1C_20190923T190459_N0208_R127_T02CMU_20190923T232238_G0120V02_P000.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2018-10-08 19:04:59 and 2019-09-23 19:04:59
        date1 = datetime(2018, 10, 8, 19, 4, 59)
        date2 = datetime(2019, 9, 23, 19, 4, 59)
        expected = date1 + (date2 - date1) / 2
        assert result == expected

    def test_url_with_full_path(self):
        """Test that method works with full S3 URL path."""
        url = "https://its-live-data.s3.amazonaws.com/velocity_image_pair/landsat/v02/N70W050/LC08_L1TP_001010_20130819_20200912_02_T1_X_LC08_L1TP_001010_20140806_20200911_02_T1_G0120V02_P044.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Should still extract dates correctly
        expected = datetime(2013, 8, 19) + (datetime(2014, 8, 6) - datetime(2013, 8, 19)) / 2
        assert result == expected

    def test_invalid_filename_format(self):
        """Test that invalid filename format raises RuntimeError exception."""
        url = "invalid_filename_without_separator.nc"

        with pytest.raises(
            RuntimeError,
            match="Filename does not contain expected split token"
        ) as _:
            _ = ITSCube.extract_mid_date_from_url(url)

    def test_unsupported_sensor_raises_error(self):
        """Test that unsupported sensor prefix raises ValueError."""
        # Using a fake sensor prefix that doesn't match any known patterns
        url = "FAKE_SENSOR_DATA_20200101_X_FAKE_SENSOR_DATA_20200201_G0120V02_P044.nc"

        # Should raise ValueError for unsupported sensor format
        with pytest.raises(ValueError, match="Unsupported sensor filename format"):
            ITSCube.extract_mid_date_from_url(url)

    def test_landsat5_filename(self):
        """Test Landsat 5 filename format (LT05)."""
        url = "LT05_L1TP_001010_19990101_20200912_02_T1_X_LT05_L1TP_001010_19990201_20200911_02_T1_G0120V02_P044.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 1999-01-01 and 1999-02-01
        expected = datetime(1999, 1, 1) + (datetime(1999, 2, 1) - datetime(1999, 1, 1)) / 2
        assert result == expected

    def test_mixed_sensors_landsat89(self):
        """Test mixed Landsat 8 and 9 filename."""
        url = "LC08_L1TP_001010_20220101_20220102_02_T1_X_LC09_L1TP_001010_20220201_20220202_02_T1_G0120V02_P044.nc"

        result = ITSCube.extract_mid_date_from_url(url)

        # Mid date should be average of 2022-01-01 and 2022-02-01
        expected = datetime(2022, 1, 1) + (datetime(2022, 2, 1) - datetime(2022, 1, 1)) / 2
        assert result == expected

    def test_chronological_ordering(self):
        """Test that different sensor files can be sorted chronologically."""
        urls = [
            "S1A_IW_SLC__1SDH_20200301T095209_20200301T095237_031349_039B9A_CBD7_X_S1B_IW_SLC__1SDH_20200401T095113_20200401T095143_020453_026C13_6119_G0120V02_P098.nc",
            "LC08_L1GT_007011_20200101_20200912_02_T2_X_LC08_L1GT_007011_20200201_20200911_02_T2_G0120V02_P044.nc",
            "S2B_MSIL1C_20200501T190459_N0206_R127_T02CMU_20200501T232024_X_S2B_MSIL1C_20200601T190459_N0208_R127_T02CMU_20200601T232238_G0120V02_P000.nc",
        ]

        # Sort URLs using the extract_mid_date_from_url method
        sorted_urls = sorted(urls, key=ITSCube.extract_mid_date_from_url)

        # Expected order: LC08 (Jan-Feb), S1A (Mar-Apr), S2B (May-Jun)
        assert sorted_urls[0].startswith("LC08")
        assert sorted_urls[1].startswith("S1A")
        assert sorted_urls[2].startswith("S2B")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
