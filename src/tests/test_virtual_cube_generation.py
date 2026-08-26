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
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
import pytest
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

from itscube_types import Vars, ImgPairInfo
import utils


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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
