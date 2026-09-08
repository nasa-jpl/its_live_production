"""
Unit tests for virtual ITS_LIVE datacube generation using pytest.

Tests virtual datacube creation using VirtualiZarr and icechunk to verify:
- Virtual datacube generation with specified polygon and projection
- Handling of mixed granule types (radar S1, optical S2, Landsat)
- Correct variable inclusion (v, vx, vy, vr, va, M11, M12)
- Proper handling of radar-specific variables (vr/va) for optical granules
- Error attribute extraction (error_modeled) for all velocity variables
- Icechunk store creation and structure

Authors: Masha Liukis
"""
import json
import math
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pyproj
import pytest
import s3fs
import xarray as xr
import icechunk as ic

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings('ignore', message='resource_tracker: There appear to be.*leaked semaphore')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Suppress shapely.geos deprecation warning
warnings.filterwarnings('ignore', message=".*'shapely.geos' module is deprecated.*")
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyogrio')

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from itscube_types import CubeFormat, SkippedGranules, Vars, ImgPairInfo
import itslive_utils
import utils
from virtual_itslive_cube import build_virtual_cube
from virtual_itslive_cube_per_chunk import (
    load_granules, skipped_granules_path, HTTPS_URL, LON_LAT_PROJECTION,
    PIXEL_SIZE_HALF, S3_URL,
)

# Golden truth for variable encoding, captured from the regular (deep-copy)
# datacube -- see test_variable_encoding_against_golden below.
GOLDEN_ENCODING_PATH = Path(__file__).parent / "data" / "cubeEncoding.json"

# Variables in the golden truth that are passthrough, granule-native 3D/
# coordinate arrays -- NOT newly synthesized by build_virtual_cube()/
# virtual_itslive_cube_per_chunk.py, so their on-disk encoding is whatever
# the source granule already used (native chunk size, native compressor),
# not something this codebase sets. Excluded from the golden-encoding
# comparison so it doesn't spuriously fail on encoding this code never
# touches.
GOLDEN_ENCODING_PASSTHROUGH_VARS = {
    'M11', 'M12', 'v', 'v_error', 'va', 'vr', 'vx', 'vy',
    'chip_size_height', 'chip_size_width', 'interp_mask', 'x', 'y',
}

# The virtual cube keeps each granule's original 'time' coordinate as the
# dimension coordinate instead of renaming it to 'mid_date' like the regular
# deep-copy datacube -- map the golden truth's 'mid_date' key onto the
# virtual cube's 'time' variable for comparison.
GOLDEN_ENCODING_NAME_OVERRIDES = {
    'mid_date': 'time',
}

# Only these encoding keys are compared against the golden truth: chunk
# sizes and compressor clevel/shuffle/blocksize legitimately differ between
# the virtual cube (one shared compressor, granule-native chunk sizes) and
# the deep-copy cube (per-variable-group compressors, rechunked to fixed
# sizes), so only sentinel-value keys and compressor *type* are checked.
GOLDEN_ENCODING_VALUE_KEYS = (
    utils.OutputFormat.fill_value,  # '_FillValue'
    utils.Missing.fill_value,       # 'fill_value' (zarr-level sentinel)
    utils.Missing.name,             # 'missing_value'
)


def _load_golden_encoding():
    """Load the golden-truth encoding map captured from the regular
    (deep-copy) datacube (data/cubeEncoding.json)."""
    with open(GOLDEN_ENCODING_PATH) as f:
        return json.load(f)


def _actual_encoding_map(var):
    """Merge a variable's .attrs and .encoding into a single lookup.

    Depending on how mask_and_scale/zarr-format round-tripping shakes out,
    _FillValue/fill_value/missing_value can surface in either dict; this
    test only cares whether/what value is present, not which xarray bucket
    it lives in.

    Inputs:
    var (xarray.Variable/DataArray): Variable to merge .attrs/.encoding for.
    """
    merged = dict(var.attrs)
    merged.update(var.encoding)
    return merged


def _compressor_cname(var):
    """Extract the compressor codec's short name (e.g. 'lz4') from a
    variable's encoding, tolerating both the zarr v3 'compressors' (list of
    codec objects) and legacy v2 'compressor' (single codec) encoding keys.

    Inputs:
    var (xarray.Variable/DataArray): Variable to extract the compressor
                  codec name from (via its .encoding).
    """
    codecs = var.encoding.get(utils.OutputFormat.compressors)
    if not codecs:
        single = var.encoding.get(utils.OutputFormat.compressor)
        codecs = [single] if single is not None else []

    if not codecs:
        return None

    codec = codecs[0]
    if isinstance(codec, dict):
        return codec.get('cname')
    return getattr(codec, 'cname', None)


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        "output_store": "its_live_cube_subset_m11_m12_s1_s2_landsat.icechunk",
        "projection": "3031",
        "polygon": [
            [-1658887.5, -430072.5],
            [-1597447.5, -430072.5],
            [-1597447.5, -368632.5],
            [-1658887.5, -368632.5],
            [-1658887.5, -430072.5]
        ],
        "granules_file": "virtual_input_39files.json",
        "expected_velocity_vars": [Vars.v, Vars.vx, Vars.vy, Vars.vr, Vars.va],
        "expected_m_vars": [Vars.m11, Vars.m12],
        "expected_error_vars": [
            f"{Vars.vx}_{Vars.postfix.error_modeled}",
            f"{Vars.vy}_{Vars.postfix.error_modeled}",
            f"{Vars.vr}_{Vars.postfix.error_modeled}",
            f"{Vars.va}_{Vars.postfix.error_modeled}"
        ],
        "expected_velocity_attribute_vars": [
            # Error attributes for each velocity component
            f"{Vars.vx}_{Vars.postfix.error}",
            f"{Vars.vx}_{Vars.postfix.error_mask}",
            f"{Vars.vx}_{Vars.postfix.error_modeled}",
            f"{Vars.vx}_{Vars.postfix.error_slow}",
            f"{Vars.vy}_{Vars.postfix.error}",
            f"{Vars.vy}_{Vars.postfix.error_mask}",
            f"{Vars.vy}_{Vars.postfix.error_modeled}",
            f"{Vars.vy}_{Vars.postfix.error_slow}",
            f"{Vars.vr}_{Vars.postfix.error}",
            f"{Vars.vr}_{Vars.postfix.error_mask}",
            f"{Vars.vr}_{Vars.postfix.error_modeled}",
            f"{Vars.vr}_{Vars.postfix.error_slow}",
            f"{Vars.va}_{Vars.postfix.error}",
            f"{Vars.va}_{Vars.postfix.error_mask}",
            f"{Vars.va}_{Vars.postfix.error_modeled}",
            f"{Vars.va}_{Vars.postfix.error_slow}",
            # Stable shift attributes for each velocity component
            f"{Vars.vx}_{Vars.postfix.stable_shift}",
            f"{Vars.vx}_{Vars.postfix.stable_shift_mask}",
            f"{Vars.vx}_{Vars.postfix.stable_shift_slow}",
            f"{Vars.vy}_{Vars.postfix.stable_shift}",
            f"{Vars.vy}_{Vars.postfix.stable_shift_mask}",
            f"{Vars.vy}_{Vars.postfix.stable_shift_slow}",
            f"{Vars.vr}_{Vars.postfix.stable_shift}",
            f"{Vars.vr}_{Vars.postfix.stable_shift_mask}",
            f"{Vars.vr}_{Vars.postfix.stable_shift_slow}",
            f"{Vars.va}_{Vars.postfix.stable_shift}",
            f"{Vars.va}_{Vars.postfix.stable_shift_mask}",
            f"{Vars.va}_{Vars.postfix.stable_shift_slow}",
        ],
        "expected_shared_attribute_vars": [
            Vars.flag_stable_shift,
            Vars.stable_count_mask,
            Vars.stable_count_slow
        ],
        "expected_m_attribute_vars": [
            f"{Vars.m11}_{Vars.postfix.dr_to_vr_factor}",
            f"{Vars.m12}_{Vars.postfix.dr_to_vr_factor}"
        ]
    }


@pytest.fixture(scope="session")
def test_output_dir():
    """Create and clean up test output directory."""
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after all tests - commented out to preserve output for inspection
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def virtual_cube_script():
    """Path to virtual_itslive_cube_per_chunk.py script."""
    return Path(__file__).parent.parent / "virtual_itslive_cube_per_chunk.py"


@pytest.fixture(scope="session")
def granules_file():
    """Path to granules JSON file."""
    granules_path = Path(__file__).parent.parent / "virtual_input_39files.json"
    if not granules_path.exists():
        pytest.skip(f"Granules file not found: {granules_path}")
    return granules_path


@pytest.fixture(scope="session")
def virtual_cube_path(test_output_dir, test_config):
    """Path to the generated virtual datacube."""
    return test_output_dir / test_config["output_store"]


# Virtual Datacube Generation Tests
class TestVirtualDatacubeGeneration:
    """Tests for virtual datacube generation via CLI."""

    @pytest.mark.order(1)
    def test_virtual_datacube_generation_via_cli(
        self, virtual_cube_path, test_config, virtual_cube_script, granules_file
    ):
        """Test virtual datacube generation using command-line interface.

        This test creates a virtual datacube using VirtualiZarr and icechunk
        from mixed granule types (radar S1, optical S2, Landsat). The virtual
        cube uses chunk references (ManifestArrays) without copying pixel data.
        """
        # Clean up any existing store
        if virtual_cube_path.exists():
            shutil.rmtree(virtual_cube_path)

        # Build command-line arguments
        cmd = [
            sys.executable,
            str(virtual_cube_script),
            "--polygon", json.dumps(test_config["polygon"]),
            "--granules-file", str(granules_file),
            "--output-store", str(virtual_cube_path),
            "--projection", test_config["projection"]
        ]

        print(f"\nRunning command: {' '.join(cmd)}")

        # Run the virtual datacube generation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Check return code
        if result.returncode != 0:
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            pytest.fail(f"Virtual datacube generation failed with return code {result.returncode}")

        print(f"\nSTDOUT:\n{result.stdout}")

        # Verify the store was created
        assert virtual_cube_path.exists(), f"Virtual cube store not created at {virtual_cube_path}"
        assert virtual_cube_path.is_dir(), f"Virtual cube store is not a directory: {virtual_cube_path}"

    @pytest.mark.order(2)
    def test_virtual_datacube_structure(self, virtual_cube_path, test_config):
        """Test that the virtual datacube has the expected structure."""
        # Open the icechunk repository
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check dimensions
        assert 'x' in cube.dims, "Missing 'x' dimension"
        assert 'y' in cube.dims, "Missing 'y' dimension"
        assert 'time' in cube.dims, "Missing 'time' dimension"

        print(f"\nDatacube dimensions: {dict(cube.sizes)}")
        print(f"Datacube variables: {list(cube.data_vars)}")

        # Check coordinates. The virtual cube keeps each granule's original
        # 'time' coordinate (unlike the regular deep-copy datacube, which
        # renames it to 'mid_date').
        assert 'x' in cube.coords, "Missing 'x' coordinate"
        assert 'y' in cube.coords, "Missing 'y' coordinate"
        assert 'time' in cube.coords, "Missing 'time' coordinate"

        # Verify data variables
        for var in test_config["expected_velocity_vars"]:
            assert var in cube.data_vars, f"Missing velocity variable: {var}"

        for var in test_config["expected_m_vars"]:
            assert var in cube.data_vars, f"Missing M variable: {var}"

        # Check error_modeled variables
        for var in test_config["expected_error_vars"]:
            assert var in cube.data_vars, f"Missing error variable: {var}"

        # Check img_pair_info variables. build_virtual_cube() promotes each
        # attribute in ImgPairInfo.all to a data variable of the same name,
        # so check against that authoritative list.
        for var_name in ImgPairInfo.all:
            assert var_name in cube.data_vars, f"Missing img_pair_info variable: {var_name}"

        print("\nAll expected variables present in datacube")

    @pytest.mark.order(3)
    def test_velocity_variable_dtypes(self, virtual_cube_path):
        """Test that velocity variables have correct data types."""
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        # Open with mask_and_scale=False to inspect the raw stored dtypes.
        # With masking on (the default), xarray promotes the int16 velocity
        # variables to float32 on read so the fill value can be represented as
        # NaN -- that's a read-time decode, not the stored dtype.
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3,
            mask_and_scale=False
        )

        # Check v, vx, vy dtypes (should be int16)
        for var in [Vars.v, Vars.vx, Vars.vy]:
            assert cube[var].dtype == 'int16', \
                f"{var} should be int16, got {cube[var].dtype}"

        # Check vr, va dtypes (should be int16)
        for var in [Vars.vr, Vars.va]:
            assert cube[var].dtype == 'int16', \
                f"{var} should be int16, got {cube[var].dtype}"

        # Check M11, M12 dtypes (should be float32)
        for var in [Vars.m11, Vars.m12]:
            assert cube[var].dtype == 'float32', \
                f"{var} should be float32, got {cube[var].dtype}"

        print("\nAll velocity variables have correct dtypes")

    @pytest.mark.order(4)
    def test_velocity_variable_attributes(self, virtual_cube_path):
        """Test that velocity variables have correct attributes."""
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check vr attributes
        vr_attrs = cube[Vars.vr].attrs
        assert Vars.attrs.std_name in vr_attrs, f"vr missing {Vars.attrs.std_name} attribute"
        assert vr_attrs[Vars.attrs.std_name] == Vars.name[Vars.vr], \
            f"vr {Vars.attrs.std_name} should be {Vars.name[Vars.vr]}, got {vr_attrs[Vars.attrs.std_name]}"
        assert utils.Units.name in vr_attrs, f"vr missing {utils.Units.name} attribute"
        assert vr_attrs[utils.Units.name] == utils.Units.m_y, \
            f"vr units should be {utils.Units.m_y}, got {vr_attrs[utils.Units.name]}"

        # Check va attributes
        va_attrs = cube[Vars.va].attrs
        assert Vars.attrs.std_name in va_attrs, f"va missing {Vars.attrs.std_name} attribute"
        assert va_attrs[Vars.attrs.std_name] == Vars.name[Vars.va], \
            f"va {Vars.attrs.std_name} should be {Vars.name[Vars.va]}, got {va_attrs[Vars.attrs.std_name]}"
        assert utils.Units.name in va_attrs, f"va missing {utils.Units.name} attribute"
        assert va_attrs[utils.Units.name] == utils.Units.m_y, \
            f"va units should be {utils.Units.m_y}, got {va_attrs[utils.Units.name]}"

        print("\nAll velocity variables have correct attributes")

    @pytest.mark.order(5)
    def test_error_modeled_variables(self, virtual_cube_path):
        """Test that error_modeled variables exist and have correct structure.

        DEPRECATED: This test is superseded by test_velocity_error_attributes.
        Kept for backward compatibility.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check all error_modeled variables
        for var in [Vars.vx, Vars.vy, Vars.vr, Vars.va]:
            error_var = f"{var}_{Vars.postfix.error_modeled}"
            assert error_var in cube.data_vars, f"Missing {error_var}"

            # These are scalar (dims=()) per granule and get stacked along time
            # by combine_by_coords, so in the assembled cube they are 1-D
            # indexed by time.
            assert cube[error_var].dims == ('time',), \
                f"{error_var} should have dims ('time',), got: {cube[error_var].dims}"

            # Check description attribute
            assert Vars.attrs.description in cube[error_var].attrs, \
                f"{error_var} missing {Vars.attrs.description} attribute"

        print("\nAll error_modeled variables present and correctly structured")

    @pytest.mark.order(6)
    def test_velocity_error_attributes(self, virtual_cube_path, test_config):
        """Test that all velocity error attribute variables exist and have correct structure.

        Each velocity variable (vx, vy, vr, va) should have 4 error attributes extracted:
        - error: base error value
        - error_stationary (error_mask): RMSE over stable surfaces
        - error_modeled: 1-sigma error from modeled error-dt relationship
        - error_slow: RMSE over slowest 25% of velocities
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check all error attributes for each velocity variable
        for vel_var in [Vars.vx, Vars.vy, Vars.vr, Vars.va]:
            for error_postfix in [
                Vars.postfix.error,
                Vars.postfix.error_mask,
                Vars.postfix.error_modeled,
                Vars.postfix.error_slow
            ]:
                error_var = f"{vel_var}_{error_postfix}"
                assert error_var in cube.data_vars, f"Missing {error_var}"

                # These are scalar (dims=()) per granule and get stacked along time
                # by combine_by_coords, so in the assembled cube they are 1-D indexed by time
                assert cube[error_var].dims == ('time',), \
                    f"{error_var} should have dims ('time',), got: {cube[error_var].dims}"

                # Check required attributes
                assert Vars.attrs.description in cube[error_var].attrs, \
                    f"{error_var} missing {Vars.attrs.description} attribute"
                assert utils.Units.name in cube[error_var].attrs, \
                    f"{error_var} missing {utils.Units.name} attribute"
                assert cube[error_var].attrs[utils.Units.name] == utils.Units.m_y, \
                    f"{error_var} units should be {utils.Units.m_y}, got {cube[error_var].attrs[utils.Units.name]}"

        print("\nAll velocity error attribute variables present and correctly structured")

    @pytest.mark.order(7)
    def test_velocity_stable_shift_attributes(self, virtual_cube_path, test_config):
        """Test that all velocity stable_shift attribute variables exist and have correct structure.

        Each velocity variable (vx, vy, vr, va) should have 3 stable_shift attributes extracted:
        - stable_shift: base shift calibration value
        - stable_shift_stationary (stable_shift_mask): shift calibrated using stable surfaces
        - stable_shift_slow: shift calibrated using slowest 25%
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check all stable_shift attributes for each velocity variable
        for vel_var in [Vars.vx, Vars.vy, Vars.vr, Vars.va]:
            for shift_postfix in [
                Vars.postfix.stable_shift,
                Vars.postfix.stable_shift_mask,
                Vars.postfix.stable_shift_slow
            ]:
                shift_var = f"{vel_var}_{shift_postfix}"
                assert shift_var in cube.data_vars, f"Missing {shift_var}"

                # These are scalar (dims=()) per granule and get stacked along time
                assert shift_var in cube.data_vars, f"Missing {shift_var}"
                assert cube[shift_var].dims == ('time',), \
                    f"{shift_var} should have dims ('time',), got: {cube[shift_var].dims}"

                # Check required attributes
                assert Vars.attrs.description in cube[shift_var].attrs, \
                    f"{shift_var} missing {Vars.attrs.description} attribute"
                assert utils.Units.name in cube[shift_var].attrs, \
                    f"{shift_var} missing {utils.Units.name} attribute"
                assert cube[shift_var].attrs[utils.Units.name] == utils.Units.m_y, \
                    f"{shift_var} units should be {utils.Units.m_y}, got {cube[shift_var].attrs[utils.Units.name]}"

        print("\nAll velocity stable_shift attribute variables present and correctly structured")

    @pytest.mark.order(8)
    def test_shared_attribute_variables(self, virtual_cube_path, test_config):
        """Test that shared attribute variables exist and have correct structure.

        Shared attributes appear once per granule (not per velocity variable):
        - flag_stable_shift: flag for applying velocity bias correction
        - stable_count_stationary (stable_count_mask): count of valid pixels over stable surfaces
        - stable_count_slow: count of valid pixels over slowest 25%
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        for shared_var in test_config["expected_shared_attribute_vars"]:
            assert shared_var in cube.data_vars, f"Missing shared attribute variable: {shared_var}"

            # These are scalar (dims=()) per granule and get stacked along time
            assert cube[shared_var].dims == ('time',), \
                f"{shared_var} should have dims ('time',), got: {cube[shared_var].dims}"

            # Check required attributes
            assert Vars.attrs.description in cube[shared_var].attrs, \
                f"{shared_var} missing {Vars.attrs.description} attribute"

            # Check units for count variables
            if 'count' in shared_var:
                assert utils.Units.name in cube[shared_var].attrs, \
                    f"{shared_var} missing {utils.Units.name} attribute"
                assert cube[shared_var].attrs[utils.Units.name] == utils.Units.count, \
                    f"{shared_var} units should be {utils.Units.count}, got {cube[shared_var].attrs[utils.Units.name]}"

        print("\nAll shared attribute variables present and correctly structured")

    @pytest.mark.order(9)
    def test_m_variable_attributes(self, virtual_cube_path, test_config):
        """Test that M11 and M12 dr_to_vr_factor attribute variables exist.

        Each M variable (M11, M12) should have a dr_to_vr_factor attribute extracted.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        for m_attr_var in test_config["expected_m_attribute_vars"]:
            assert m_attr_var in cube.data_vars, f"Missing M attribute variable: {m_attr_var}"

            # These are scalar (dims=()) per granule and get stacked along time
            assert cube[m_attr_var].dims == ('time',), \
                f"{m_attr_var} should have dims ('time',), got: {cube[m_attr_var].dims}"

            # Check required attributes
            assert Vars.attrs.description in cube[m_attr_var].attrs, \
                f"{m_attr_var} missing {Vars.attrs.description} attribute"
            assert utils.Units.name in cube[m_attr_var].attrs, \
                f"{m_attr_var} missing {utils.Units.name} attribute"
            assert cube[m_attr_var].attrs[utils.Units.name] == utils.Units.m_per_year_pixel, \
                f"{m_attr_var} units should be {utils.Units.m_per_year_pixel}, got {cube[m_attr_var].attrs[utils.Units.name]}"

        print("\nAll M attribute variables present and correctly structured")

    @pytest.mark.order(10)
    def test_all_velocity_attributes_comprehensive(self, virtual_cube_path, test_config):
        """Comprehensive test to verify all velocity attribute variables are present.

        This test verifies the complete set of extracted velocity attributes against
        the expected list in test_config.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check all velocity attribute variables
        for var in test_config["expected_velocity_attribute_vars"]:
            assert var in cube.data_vars, f"Missing velocity attribute variable: {var}"

        # Check all shared attribute variables
        for var in test_config["expected_shared_attribute_vars"]:
            assert var in cube.data_vars, f"Missing shared attribute variable: {var}"

        # Check all M attribute variables
        for var in test_config["expected_m_attribute_vars"]:
            assert var in cube.data_vars, f"Missing M attribute variable: {var}"

        print("\nComprehensive velocity attribute variables check passed")
        print(f"Total velocity attribute variables: {len(test_config['expected_velocity_attribute_vars'])}")
        print(f"Total shared attribute variables: {len(test_config['expected_shared_attribute_vars'])}")
        print(f"Total M attribute variables: {len(test_config['expected_m_attribute_vars'])}")

    @pytest.mark.order(11)
    def test_datacube_attributes(self, virtual_cube_path, test_config):
        """Test that the datacube has expected global attributes."""
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        # Check required attributes
        required_attrs = [
            'date_created',
            'title',
            'institution',
            'projection',
            'geo_polygon',
            'proj_polygon'
        ]

        for attr in required_attrs:
            assert attr in cube.attrs, f"Missing required attribute: {attr}"

        # Check projection matches input
        assert cube.attrs['projection'] == test_config['projection'], \
            f"Projection should be {test_config['projection']}, got {cube.attrs['projection']}"

        print(f"\nAll required datacube attributes present")
        print(f"Datacube attributes: {list(cube.attrs.keys())}")

    @pytest.mark.order(11)
    def test_datacube_attributes_no_granule_leakage(self, virtual_cube_path, test_config):
        """Verify the datacube's global attributes are exactly the
        canonical set virtual_itslive_cube_per_chunk.py's ``__main__``
        explicitly assigns after ``cube.attrs.clear()`` -- and nothing
        else. build_virtual_cube() seeds each mosaicked dataset's attrs
        from the granules' own global attrs (``attrs=vds.attrs``,
        combined via ``combine_attrs="drop_conflicts"``), so without that
        explicit clear()+rebuild, granule-only attributes (e.g. 'author',
        'source', 'satellite', the injected 'granule_url'/'granule_path')
        would leak straight through into the cube.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3
        )

        expected_keys = {
            utils.OutputFormat.conventions,
            CubeFormat.date_created,
            CubeFormat.gdal_area_or_point,
            CubeFormat.geo_polygon,
            utils.OutputFormat.institution,
            utils.OutputFormat.latitude,
            utils.OutputFormat.longitude,
            CubeFormat.proj_polygon,
            utils.OutputFormat.projection,
            ImgPairInfo.time_standard_img1,
            ImgPairInfo.time_standard_img2,
            utils.OutputFormat.title,
            Vars.attrs.autorift_param_file,
            utils.OutputFormat.s3,
            utils.OutputFormat.url,
            SkippedGranules.name,
        }
        actual_keys = set(cube.attrs.keys())
        assert actual_keys == expected_keys, (
            "Datacube global attributes don't match the canonical set "
            "virtual_itslive_cube_per_chunk.py explicitly assigns -- "
            f"extra: {actual_keys - expected_keys}, "
            f"missing: {expected_keys - actual_keys}"
        )

        # None of a source granule's own global netCDF attributes -- which
        # would otherwise survive into cube.attrs via combine_by_coords'
        # combine_attrs="drop_conflicts" before the script's explicit
        # cube.attrs.clear() -- should have leaked through. (Already
        # implied by the exact-set check above; spelled out explicitly
        # here using the real attribute names a source granule carries.)
        granule_only_attrs = [
            'author', 'date_updated', 'motion_coordinates',
            'motion_detection_method', 'references', 'satellite',
            'scene_pair_type', 'source', 'granule_url', 'granule_path',
        ]
        for attr_name in granule_only_attrs:
            assert attr_name not in cube.attrs, (
                f"Granule-only attribute '{attr_name}' leaked into cube.attrs"
            )

        # Fixed canonical values (CubeFormat.values) -- 'title' in
        # particular is deliberately distinct from any source granule's own
        # 'title' ('autoRIFT surface velocities'), proving it's the cube's
        # own value, not copied through from a granule.
        assert cube.attrs[utils.OutputFormat.conventions] == \
            CubeFormat.values[utils.OutputFormat.conventions]
        assert cube.attrs[CubeFormat.gdal_area_or_point] == \
            CubeFormat.values[CubeFormat.gdal_area_or_point]
        assert cube.attrs[utils.OutputFormat.institution] == \
            CubeFormat.values[utils.OutputFormat.institution]
        assert cube.attrs[utils.OutputFormat.title] == \
            CubeFormat.values[utils.OutputFormat.title]

        # date_created is this run's own batch_job_id timestamp, not
        # copied from any granule's (much older) date_created attribute --
        # check it parses in the expected format and is recent, rather
        # than trusting it merely exists.
        date_created = datetime.strptime(
            cube.attrs[CubeFormat.date_created], '%d-%b-%Y %H:%M:%S'
        )
        assert date_created.year >= 2024, (
            "date_created looks stale for this run's batch_job_id: "
            f"{cube.attrs[CubeFormat.date_created]!r}"
        )

        # autoRIFT_parameter_file is legitimately derived from (a superset
        # of) the granules' own attribute of the same name, normalized to
        # https -- format check only, not an identity check against any
        # one granule.
        assert cube.attrs[Vars.attrs.autorift_param_file].startswith('https://')

        # Local filesystem output (see test_config/virtual_cube_path) ->
        # s3/url left blank, matching the is_s3_output=False branch.
        assert cube.attrs[utils.OutputFormat.s3] == ''
        assert cube.attrs[utils.OutputFormat.url] == ''

        # skipped_granules points at this store's sidecar JSON.
        assert cube.attrs[SkippedGranules.name] == \
            skipped_granules_path(str(virtual_cube_path))

        # time_standard is always UTC for every source ITS_LIVE granule.
        assert cube.attrs[ImgPairInfo.time_standard_img1] == 'UTC'
        assert cube.attrs[ImgPairInfo.time_standard_img2] == 'UTC'

        # geo_polygon/proj_polygon/latitude/longitude: recompute with the
        # same production helpers (itslive_utils.add_five_points_to_
        # polygon_side, pyproj) the script itself uses, applied directly
        # to test_config's --polygon/--projection, rather than trusting
        # the script's own derivation of these values.
        polygon = test_config["polygon"]
        projection = test_config["projection"]
        x_coords = [c[0] for c in polygon]
        y_coords = [c[1] for c in polygon]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        xmin += PIXEL_SIZE_HALF
        xmax -= PIXEL_SIZE_HALF
        ymin += PIXEL_SIZE_HALF
        ymax -= PIXEL_SIZE_HALF
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2

        transformer = pyproj.Transformer.from_crs(
            f"EPSG:{projection}", LON_LAT_PROJECTION, always_xy=True
        )
        augmented_polygon = itslive_utils.add_five_points_to_polygon_side(polygon)
        expected_proj_polygon = [list(pt) for pt in augmented_polygon]
        expected_geo_polygon = [
            list(transformer.transform(pt[0], pt[1])) for pt in augmented_polygon
        ]
        expected_lon, expected_lat = transformer.transform(xmid, ymid)

        actual_proj_polygon = json.loads(cube.attrs[CubeFormat.proj_polygon])
        actual_geo_polygon = json.loads(cube.attrs[CubeFormat.geo_polygon])

        assert np.allclose(actual_proj_polygon, expected_proj_polygon), (
            f"proj_polygon mismatch: expected {expected_proj_polygon}, "
            f"got {actual_proj_polygon}"
        )
        assert np.allclose(actual_geo_polygon, expected_geo_polygon), (
            f"geo_polygon mismatch: expected {expected_geo_polygon}, "
            f"got {actual_geo_polygon}"
        )
        assert cube.attrs[utils.OutputFormat.longitude] == round(expected_lon, 2)
        assert cube.attrs[utils.OutputFormat.latitude] == round(expected_lat, 2)

        print(
            "\nAll datacube global attributes match production values; "
            "no granule-only attributes leaked into cube.attrs"
        )

    @pytest.mark.order(12)
    def test_variable_encoding_against_golden(self, virtual_cube_path):
        """Compare newly-introduced variables' encoding against the golden
        truth captured from the regular (deep-copy) datacube
        (data/cubeEncoding.json).

        Only checks existence/value of _FillValue, fill_value, and
        missing_value, plus the compressor's type (cname) -- not chunk
        sizes or exact compressor clevel/shuffle/blocksize, which
        legitimately differ between the virtual cube (one shared
        compressor, granule-native chunk sizes) and the deep-copy cube
        (per-variable-group compressors, rechunked to fixed sizes). The
        golden truth's 'mid_date' entry is compared against this cube's
        'time' coordinate, since the virtual cube keeps the granules'
        original 'time' coordinate rather than renaming it to 'mid_date'.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        # mask_and_scale=False: this test inspects the raw on-disk
        # encoding/attrs the golden truth was itself captured from, not the
        # CF-decoded representation.
        cube = xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3,
            mask_and_scale=False
        )

        golden = _load_golden_encoding()
        checked_vars = []

        for golden_name, golden_enc in golden.items():
            if golden_name in GOLDEN_ENCODING_PASSTHROUGH_VARS:
                continue

            var_name = GOLDEN_ENCODING_NAME_OVERRIDES.get(golden_name, golden_name)
            assert var_name in cube.variables, \
                f"Expected newly-introduced variable '{var_name}' " \
                f"(golden key '{golden_name}') missing from virtual cube"

            actual = _actual_encoding_map(cube[var_name])

            for key in GOLDEN_ENCODING_VALUE_KEYS:
                if key not in golden_enc:
                    continue

                assert key in actual, \
                    f"{var_name}: missing '{key}' (golden value: {golden_enc[key]!r})"

                golden_value = golden_enc[key]
                actual_value = actual[key]

                if isinstance(golden_value, float) and math.isnan(golden_value):
                    assert isinstance(actual_value, float) and math.isnan(actual_value), \
                        f"{var_name}: expected NaN for '{key}', got {actual_value!r}"
                else:
                    assert actual_value == golden_value, \
                        f"{var_name}: '{key}' mismatch -- expected " \
                        f"{golden_value!r}, got {actual_value!r}"

            golden_cname = golden_enc.get('compressor', {}).get('cname')
            if golden_cname is not None:
                actual_cname = _compressor_cname(cube[var_name])
                assert actual_cname == golden_cname, \
                    f"{var_name}: compressor type mismatch -- expected " \
                    f"cname={golden_cname!r}, got {actual_cname!r}"

            checked_vars.append(var_name)

        print(f"\nChecked encoding against golden truth for {len(checked_vars)} variables:")
        print(sorted(checked_vars))


class TestVirtualCubeValuesAgainstGranule:
    """Verify that the values of data variables newly introduced by
    build_virtual_cube() -- promoted from each granule's img_pair_info and
    velocity-component attributes -- match the corresponding values read
    directly from the original source granule, after round-tripping through
    cropping, time-stacking, and the icechunk write/read.

    Each test computes its "expected" value with the same production helpers
    (utils.get_data_var_attr/get_data_var_binary_attr) that
    virtual_itslive_cube.py itself uses, but applied directly to the fully
    opened source granule rather than the cropped virtual dataset. This
    isolates the thing actually being tested: whether a value survives the
    virtualization/cropping/stacking/icechunk pipeline intact, not whether
    the extraction helpers themselves are correct (already covered by their
    own callers).
    """

    # Epoch used by every GPS-epoch-encoded (time,) variable in the virtual
    # cube -- see utils.Units.gps_epoch_date / set_1d_time_chunk_encoding.
    GPS_EPOCH = pd.Timestamp('1980-01-06T00:00:00')

    @pytest.fixture(scope="class")
    def raw_cube(self, virtual_cube_path):
        """Cube opened with mask_and_scale=False so newly-introduced scalar
        variables read back as their raw stored values -- matching what
        utils.get_data_var_attr()/build_virtual_cube() computed -- rather
        than a CF-decoded/masked representation.
        """
        repo = ic.Repository.open(ic.local_filesystem_storage(str(virtual_cube_path)))
        return xr.open_zarr(
            repo.readonly_session("main").store,
            consolidated=False,
            zarr_format=3,
            mask_and_scale=False,
            decode_times=False,
        )

    @staticmethod
    def _open_granule(granule_url):
        """Open the original granule (S3 NetCDF) read-only, for comparison.

        Inputs:
        granule_url (str): S3 URL of the granule to open (e.g.
                      's3://its-live-data/.../granule.nc').
        """
        s3 = s3fs.S3FileSystem(anon=True)
        with s3.open(granule_url, mode='rb') as fhandle:
            return xr.open_dataset(fhandle, engine=utils.NC_ENGINE).load()

    def _decode_gps_seconds(self, seconds):
        """Convert a GPS-epoch-seconds value (this cube's on-disk time
        encoding) back to a pandas Timestamp for comparison.

        Inputs:
        seconds (float): Number of seconds since GPS_EPOCH, as stored
                      on disk for a (time,) GPS-epoch-encoded variable.
        """
        return self.GPS_EPOCH + pd.to_timedelta(float(seconds), unit='s')

    def _check_img_pair_info(self, cube, ds, t_idx, granule_url):
        """Compare every ImgPairInfo attribute (dates, ascending flags,
        and remaining scalar attributes) promoted to a cube data variable
        against the value read directly from the source granule `ds` at
        time index `t_idx`.

        Inputs:
        cube (xarray.Dataset): Assembled virtual datacube (raw_cube fixture)
                      to check promoted ImgPairInfo variables in.
        ds (xarray.Dataset): Fully opened source granule to read the
                      original img_pair_info attributes from.
        t_idx (int): Time index of `granule_url` within `cube`.
        granule_url (str): S3 URL of the source granule (used for error
                      reporting only).
        """
        for attr in ImgPairInfo.all:
            if attr in ImgPairInfo.toDate:
                expected = utils.get_data_var_attr(
                    ds, granule_url, ImgPairInfo.name, attr, to_date=True
                )
                expected_ts = pd.Timestamp(expected)
                if expected_ts.tzinfo is not None:
                    expected_ts = expected_ts.tz_localize(None)

                actual_ts = self._decode_gps_seconds(cube[attr].values[t_idx])
                delta = abs((actual_ts - expected_ts).total_seconds())
                assert delta < 1e-3, (
                    f"{attr} mismatch for {granule_url}: "
                    f"expected {expected_ts}, got {actual_ts}"
                )
                continue

            attr_dtype = ImgPairInfo.allTypes.get(attr)
            expected = utils.get_data_var_attr(
                ds, granule_url, ImgPairInfo.name, attr, data_dtype=attr_dtype
            )
            actual = cube[attr].values[t_idx]

            if attr in ImgPairInfo.stringType or isinstance(expected, str):
                assert str(actual) == str(expected), (
                    f"{attr} mismatch for {granule_url}: "
                    f"expected {expected!r}, got {actual!r}"
                )
            else:
                assert np.float32(actual) == np.float32(expected), (
                    f"{attr} mismatch for {granule_url}: "
                    f"expected {expected!r}, got {actual!r}"
                )

        for flight_attr, ascending_var in [
            (ImgPairInfo.flight_direction_img1, Vars.ascending_img1),
            (ImgPairInfo.flight_direction_img2, Vars.ascending_img2),
        ]:
            expected = utils.get_data_var_binary_attr(
                ds, granule_url, ImgPairInfo.name, flight_attr, ImgPairInfo.ascending,
                data_dtype=np.uint8, missing_value=utils.Missing.u8value
            )
            actual = cube[ascending_var].values[t_idx]
            assert int(actual) == int(expected), (
                f"{ascending_var} mismatch for {granule_url}: "
                f"expected {expected!r}, got {actual!r}"
            )

    @staticmethod
    def _expected_scalar_attr(granule_attrs, attr_name):
        """Expected value for a scalar (time,) attribute variable written by
        build_virtual_cube()'s generic extraction (error/error_mask/
        error_modeled/error_slow, stable_shift_mask/stable_shift_slow).

        These variables are written with encoding={'_FillValue':
        Missing.value, 'dtype': float32} but *no* explicit NaN handling in
        the production code (unlike stable_shift, which is special-cased) --
        so if the granule attribute is present but NaN, xarray's CF encoder
        converts that NaN to the fill value on write, same as if the
        attribute were absent outright.

        Inputs:
        granule_attrs (dict): .attrs of the source granule's velocity
                      variable (e.g. ds[Vars.vx].attrs) to look up
                      `attr_name` in.
        attr_name (str): Name of the attribute to compute the expected
                      value for (e.g. Vars.postfix.error_modeled).
        """
        if attr_name not in granule_attrs:
            return np.float32(utils.Missing.value)

        value = np.float32(granule_attrs[attr_name])
        return np.float32(utils.Missing.value) if np.isnan(value) else value

    def _check_velocity_attributes(self, cube, ds, t_idx, granule_url):
        """Compare each velocity component's (vx/vy/vr/va) error and
        stable_shift attribute variables, plus the shared attributes
        (flag_stable_shift, stable_count_mask, stable_count_slow) read
        once from vx, against the values read directly from the source
        granule `ds` at time index `t_idx`.

        Inputs:
        cube (xarray.Dataset): Assembled virtual datacube (raw_cube fixture)
                      to check promoted attribute variables in.
        ds (xarray.Dataset): Fully opened source granule to read the
                      original velocity-component attributes from.
        t_idx (int): Time index of `granule_url` within `cube`.
        granule_url (str): S3 URL of the source granule (used for error
                      reporting only).
        """
        for var_name in [Vars.vx, Vars.vy, Vars.vr, Vars.va]:
            has_var = var_name in ds.data_vars
            if var_name in (Vars.vr, Vars.va) and not has_var:
                # Optical granule: no source data for this radar-specific
                # velocity component, so there's no real value to compare.
                continue

            granule_attrs = ds[var_name].attrs

            for each_attr in [
                Vars.postfix.error, Vars.postfix.error_mask,
                Vars.postfix.error_modeled, Vars.postfix.error_slow
            ]:
                cube_var_name = f"{var_name}_{each_attr}"
                expected = self._expected_scalar_attr(granule_attrs, each_attr)
                actual = np.float32(cube[cube_var_name].values[t_idx])
                assert actual == expected, (
                    f"{cube_var_name} mismatch for {granule_url}: "
                    f"expected {expected!r}, got {actual!r}"
                )

            # stable_shift: a NaN attribute value is normalized to 0.
            shift_var_name = f"{var_name}_{Vars.postfix.stable_shift}"
            raw_shift = utils.get_data_var_attr(
                ds, granule_url, var_name, Vars.postfix.stable_shift,
                utils.Missing.value
            )
            expected_shift = np.float32(0) if np.isnan(raw_shift) else np.float32(raw_shift)
            actual_shift = np.float32(cube[shift_var_name].values[t_idx])
            assert actual_shift == expected_shift, (
                f"{shift_var_name} mismatch for {granule_url}: "
                f"expected {expected_shift!r}, got {actual_shift!r}"
            )

            for each_attr in [Vars.postfix.stable_shift_mask, Vars.postfix.stable_shift_slow]:
                shift_var_name = f"{var_name}_{each_attr}"
                expected = self._expected_scalar_attr(granule_attrs, each_attr)
                actual = np.float32(cube[shift_var_name].values[t_idx])
                assert actual == expected, (
                    f"{shift_var_name} mismatch for {granule_url}: "
                    f"expected {expected!r}, got {actual!r}"
                )

        # Shared attributes are read once from vx (present in every granule).
        # Cast expected through the same on-disk integer dtype
        # build_virtual_cube() declares for each attribute (Vars.intType) --
        # not a wider int32 -- so if a granule value ever overflowed that
        # dtype's width (e.g. flag_stable_shift's uint8), it's expected to
        # wrap around here too, matching actual stored behavior. stable_count_
        # mask/stable_count_slow are uint32 in Vars.intType, wide enough for
        # any real pixel count, so no wraparound occurs for those in practice.
        for each_attr in [Vars.flag_stable_shift, Vars.stable_count_mask, Vars.stable_count_slow]:
            raw_expected = utils.get_data_var_attr(
                ds, granule_url, Vars.vx, each_attr, data_dtype=np.int32
            )
            expected = np.array(raw_expected, dtype=np.int64).astype(Vars.intType[each_attr])
            actual = cube[each_attr].values[t_idx]
            assert int(actual) == int(expected), (
                f"{each_attr} mismatch for {granule_url}: "
                f"expected {expected!r}, got {actual!r}"
            )

    def _check_m_attributes(self, cube, ds, t_idx, granule_url):
        """Compare each M variable's (M11/M12) dr_to_vr_factor attribute
        variable against the value read directly from the source granule
        `ds` at time index `t_idx`.

        Inputs:
        cube (xarray.Dataset): Assembled virtual datacube (raw_cube fixture)
                      to check promoted dr_to_vr_factor variables in.
        ds (xarray.Dataset): Fully opened source granule to read the
                      original M11/M12 attributes from.
        t_idx (int): Time index of `granule_url` within `cube`.
        granule_url (str): S3 URL of the source granule (used for error
                      reporting only).
        """
        for var_name in [Vars.m11, Vars.m12]:
            expected = utils.get_data_var_attr(
                ds, granule_url, var_name, Vars.postfix.dr_to_vr_factor,
                utils.Missing.byte
            )
            attr_var_name = f"{var_name}_{Vars.postfix.dr_to_vr_factor}"
            actual = np.float32(cube[attr_var_name].values[t_idx])
            assert actual == np.float32(expected), (
                f"{attr_var_name} mismatch for {granule_url}: "
                f"expected {expected!r}, got {actual!r}"
            )

    def _check_autorift_version(self, cube, ds, t_idx, granule_url):
        """Compare the autorift_software_version data variable against the
        global attribute of the same name on the source granule `ds` at
        time index `t_idx`.

        Inputs:
        cube (xarray.Dataset): Assembled virtual datacube (raw_cube fixture)
                      to check the promoted autorift_software_version
                      variable in.
        ds (xarray.Dataset): Fully opened source granule to read the
                      original autorift_software_version global attribute
                      from.
        t_idx (int): Time index of `granule_url` within `cube`.
        granule_url (str): S3 URL of the source granule (used for error
                      reporting only).
        """
        expected = ds.attrs[Vars.autorift_software_version]
        actual = str(cube[Vars.autorift_software_version].values[t_idx])
        assert actual == str(expected), (
            f"{Vars.autorift_software_version} mismatch for {granule_url}: "
            f"expected {expected!r}, got {actual!r}"
        )

    @pytest.mark.order(13)
    def test_data_variable_values_match_granule(self, raw_cube):
        """For every granule that made it into the cube, re-derive the
        newly-introduced scalar variables straight from the source granule
        and compare against what's stored in the cube at that granule's
        time position.
        """
        cube = raw_cube
        granule_urls = [str(u) for u in cube[Vars.url].values]
        assert len(granule_urls) > 0, "virtual cube has no layers to check"

        for t_idx, granule_url in enumerate(granule_urls):
            ds = self._open_granule(granule_url)
            try:
                self._check_img_pair_info(cube, ds, t_idx, granule_url)
                self._check_velocity_attributes(cube, ds, t_idx, granule_url)
                self._check_m_attributes(cube, ds, t_idx, granule_url)
                self._check_autorift_version(cube, ds, t_idx, granule_url)
            finally:
                ds.close()

        print(
            f"\nVerified newly-introduced variable values against "
            f"{len(granule_urls)} source granules"
        )


class TestBuildVirtualCubeSingleGranule:
    """Regression test for a bug where build_virtual_cube(), given only a
    single granule, left every newly-synthesized per-granule scalar
    attribute variable (error/stable_shift/stable_count/flag_stable_shift,
    img_pair_info attrs, url, ascending_img1/2, autorift_software_version,
    M11/M12 dr_to_vr_factor) dimensionless instead of giving it a 'time'
    dimension.

    Root cause: those variables are created with dims=() and rely on
    xr.combine_by_coords() to stack them along 'time' when granules are
    merged -- but combine_by_coords() on a single-element list returns the
    dataset unchanged (nothing to stack/order), so the dims=() variables
    never gained the intended 'time' dimension. Fixed in build_virtual_cube()
    by using xr.concat() (which forces the 'time' dimension even for one
    dataset) when there's only one granule to combine.
    """

    @pytest.fixture(scope="class")
    def single_granule_cube(self, granules_file):
        """Build a virtual cube from exactly one granule, using the same
        load_granules()/build_virtual_cube() production helpers as the CLI
        entry point, without the CLI subprocess/icechunk-write overhead.
        """
        with open(granules_file) as f:
            granules = json.load(f)

        real_url = granules[0].replace(HTTPS_URL, S3_URL)
        vds_list, missing_granules = load_granules([real_url], "s3://its-live-data")
        assert missing_granules == [], f"Unexpected missing granules: {missing_granules}"
        assert len(vds_list) == 1, "Expected exactly one loaded granule"

        cube, _ = build_virtual_cube(vds_list)
        return cube

    def test_scalar_attribute_variables_have_time_dimension(
        self, single_granule_cube, test_config
    ):
        """Every newly-synthesized per-granule attribute variable must have
        dims == ('time',) with length 1 -- not dims == () (the bug)."""
        cube = single_granule_cube

        checked_vars = (
            test_config["expected_velocity_attribute_vars"]
            + test_config["expected_shared_attribute_vars"]
            + test_config["expected_m_attribute_vars"]
            + list(ImgPairInfo.all)
            + [
                Vars.url,
                Vars.ascending_img1,
                Vars.ascending_img2,
                Vars.autorift_software_version,
            ]
        )

        for var_name in checked_vars:
            assert var_name in cube.data_vars, f"Missing variable: {var_name}"
            assert cube[var_name].dims == ('time',), (
                f"{var_name} should have dims ('time',), got "
                f"{cube[var_name].dims} -- regression of the single-granule "
                f"dimensionless-attribute bug"
            )
            assert cube[var_name].sizes['time'] == 1, (
                f"{var_name} should have time length 1, got "
                f"{cube[var_name].sizes['time']}"
            )

        print(
            f"\nVerified {len(checked_vars)} scalar attribute variables "
            f"have 'time' dimension for a single-granule cube"
        )


class TestLoadGranulesMissingBypass:
    """Tests for load_granules()'s temporary bypass of a known searchAPI/
    catalog bug where some returned granule URLs don't correspond to any
    real S3 object (see _granule_exists in virtual_itslive_cube_per_chunk.py).
    """

    def test_missing_granule_is_skipped_not_raised(self, granules_file):
        """A granule that doesn't exist in S3 is reported back as missing
        instead of aborting the whole load; a real granule in the same call
        still loads normally."""
        with open(granules_file) as f:
            real_granules = json.load(f)

        real_url = real_granules[0].replace(HTTPS_URL, S3_URL)
        bogus_url = f"{S3_URL}velocity_image_pair/does/not/exist_test_only.nc"

        vds_list, missing_granules = load_granules(
            [real_url, bogus_url], "s3://its-live-data"
        )

        assert len(vds_list) == 1, "Only the real granule should load"
        assert missing_granules == [bogus_url], \
            "Bogus granule should be reported as missing, not raised"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
