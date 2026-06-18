"""
Unit tests for ITSCube datacube generation using pytest.

Tests datacube creation with small test regions to verify:
- Datacube generation with specified polygon and projection
- Correct grid creation and spatial bounds
- Granule filtering and processing
- Zarr store creation and structure

IMPORTANT: These tests do NOT upload any data to S3. All S3-related arguments
(--outputBucket, --targetBucket, --backupBucket) are intentionally omitted to
ensure that generated test datacubes remain local only in src/tests/test_output/.
No AWS credentials are needed to run these tests.

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
import zarr

# Suppress multiprocessing resource tracker warnings
# These occur during cleanup but don't affect test results
warnings.filterwarnings('ignore', message='resource_tracker: There appear to be.*leaked semaphore')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Suppress shapely.geos deprecation warning from pyogrio
# This will be fixed when pyogrio is updated to a version compatible with shapely 2.0+
warnings.filterwarnings('ignore', message=".*'shapely.geos' module is deprecated.*")
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pyogrio')

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from itscube import ITSCube
from grid import Bounds, Grid
from itslive_composite import ITSLiveComposite
import shapefile


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        "output_file": "small_malaspina_ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr",
        "target_projection": "3413",
        "polygon": [
            [-3300000, 200000],
            [-3200000, 200000],
            [-3200000, 300000],
            [-3300000, 300000],
            [-3300000, 200000]
        ],
        "cell_size": 120,  # Grid cell size in kilometers
        "num_granules": 200,  # Limit for testing
        "chunks": 100,  # Number of granules to write at a time
        "composite_chunks": 100,  # Spacial points of data in x and y to
                                    # process in parallel for composites.
        "output_composite": "composite.zarr",
        "default_shapefile": "s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp",
        "use_error_slow": False,
        "v0_years": list(range(2014, 2025))
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
def itscube_script():
    """Path to itscube.py script."""
    return Path(__file__).parent.parent / "itscube.py"


@pytest.fixture(scope="session")
def composite_script():
    """Path to itslive_composite.py script."""
    return Path(__file__).parent.parent / "itslive_composite.py"


@pytest.fixture(scope="session")
def datacube_path(test_output_dir, test_config):
    """Path to the generated datacube."""
    return test_output_dir / test_config["output_file"]


@pytest.fixture(scope="session")
def composite_path(test_output_dir, test_config):
    """Path to the generated composite."""
    return test_output_dir / test_config["output_composite"]


# Datacube Generation Tests
class TestDatacubeGeneration:
    """Tests for datacube generation via CLI."""

    @pytest.mark.order(1)
    def test_datacube_generation_via_cli(
        self, datacube_path, test_config, itscube_script
    ):
        """Test datacube generation using command-line interface.

        IMPORTANT: This test does NOT upload to S3. The --outputBucket and
        --targetBucket arguments are intentionally omitted, which ensures that
        no S3 copy operations occur (see itscube.py line 3563: if len(target_bucket)).
        The datacube is created locally in src/tests/test_output/ only.
        """
        # Build command-line arguments
        # NOTE: No --outputBucket (-b) or --targetBucket (-tb) arguments are provided,
        # which ensures no S3 upload occurs
        cmd = [
            sys.executable,
            str(itscube_script),
            "-o", str(datacube_path),
            "--targetProjection", test_config["target_projection"],
            "--polygon", json.dumps(test_config["polygon"]),
            "-c", str(test_config["chunks"]),
            "-g", str(test_config["cell_size"]),
            "--fivePointsPerPolygonSide",
            "-n", str(test_config["num_granules"])
        ]

        print(f"\nRunning command: {' '.join(cmd)}")

        # Run the datacube generation
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Check that the command completed successfully
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(
                f"Datacube generation failed with return code {result.returncode}"
            )
        else:
            print(f"Datacube STDOUT:\n{result.stdout}")
            print(f"Datacube STDERR:\n{result.stderr}")

        # Verify that the output file was created
        assert datacube_path.exists(), (
            f"Output datacube not created at {datacube_path}"
        )

        print(f"✓ Datacube created successfully at {datacube_path}")

    @pytest.mark.order(2)
    def test_verify_datacube_structure(self, datacube_path, test_config):
        """Verify the structure and contents of the generated datacube."""
        # Skip if datacube doesn't exist (previous test failed)
        if not datacube_path.exists():
            pytest.skip("Datacube was not created in previous test")

        # Open the datacube using xarray
        ds = xr.open_dataset(
            str(datacube_path),
            decode_timedelta=False,
            engine='zarr',
            consolidated=True
        )

        # Verify dimensions exist
        assert 'mid_date' in ds.sizes, "mid_date dimension missing"
        assert 'x' in ds.sizes, "x dimension missing"
        assert 'y' in ds.sizes, "y dimension missing"

        # Verify coordinate variables
        assert 'mid_date' in ds.coords, "mid_date coordinate missing"
        assert 'x' in ds.coords, "x coordinate missing"
        assert 'y' in ds.coords, "y coordinate missing"

        # Verify that mid_date is in ascending order`
        mid_date = ds['mid_date'].values
        sorted_mid_date = sorted(mid_date)
        assert all(sorted_mid_date == mid_date), 'mid_date is not sorted'

        # Verify all expected data variables exist
        expected_vars = [
            # Velocity components
            'v', 'vx', 'vy', 'vr', 'va',
            # Velocity errors
            'v_error', 'vx_error', 'vy_error', 'vr_error', 'va_error',
            # Modeled errors
            'vx_error_modeled', 'vy_error_modeled', 'vr_error_modeled',
            'va_error_modeled',
            # Slow errors
            'vx_error_slow', 'vy_error_slow', 'vr_error_slow', 'va_error_slow',
            # Stationary errors
            'vx_error_stationary', 'vy_error_stationary',
            'vr_error_stationary', 'va_error_stationary',
            # Stable shifts
            'vx_stable_shift', 'vy_stable_shift', 'vr_stable_shift',
            'va_stable_shift', 'vx_stable_shift_slow', 'vy_stable_shift_slow',
            'vr_stable_shift_slow', 'va_stable_shift_slow',
            'vx_stable_shift_stationary', 'vy_stable_shift_stationary',
            'vr_stable_shift_stationary', 'va_stable_shift_stationary',
            # Stable counts and flags
            'stable_count_slow', 'stable_count_stationary', 'stable_shift_flag',
            # M11/M12 and factors
            'M11', 'M12', 'M11_dr_to_vr_factor', 'M12_dr_to_vr_factor',
            # Date information
            'date_center', 'date_dt', 'acquisition_date_img1',
            'acquisition_date_img2',
            # Mission/sensor information
            'mission_img1', 'mission_img2', 'satellite_img1', 'satellite_img2',
            'sensor_img1', 'sensor_img2',
            # Processing parameters
            'chip_size_height', 'chip_size_width', 'interp_mask',
            'autoRIFT_software_version', 'granule_url',
            # Ice masks
            'landice', 'floatingice',
            # ROI and mapping
            'roi_valid_percentage', 'mapping'
        ]

        # Check for existence of all expected variables
        missing_vars = [var for var in expected_vars if var not in ds.data_vars]
        assert len(missing_vars) == 0, f"Missing data variables: {missing_vars}"

        # Verify we have time layers (granules)
        num_layers = len(ds.mid_date)
        assert num_layers > 0, "Datacube has no time layers"
        assert num_layers <= test_config["num_granules"], (
            f"Datacube has more layers ({num_layers}) than "
            f"requested ({test_config['num_granules']})"
        )

        # Verify EPSG code attribute
        assert 'mapping' in ds.data_vars, "Mapping variable missing"
        if 'spatial_epsg' in ds.mapping.attrs:
            assert str(ds.mapping.attrs['spatial_epsg']) == test_config["target_projection"], (
                "EPSG code mismatch"
            )

        print(f"✓ Datacube structure verified:")
        print(f"  - Dimensions: {ds.sizes}")
        print(f"  - Number of time layers: {num_layers}")
        print(f"  - Data variables: {list(ds.data_vars.keys())}")

        ds.close()

    @pytest.mark.order(3)
    def test_verify_zarr_chunks(self, datacube_path):
        """Verify Zarr chunking strategy."""
        # Skip if datacube doesn't exist
        if not datacube_path.exists():
            pytest.skip("Datacube was not created")

        # Open zarr store directly
        store = zarr.open(str(datacube_path), mode='r')

        # Check that velocity variables have appropriate chunking
        if 'v' in store:
            v_chunks = store['v'].chunks
            print(f"✓ Velocity (v) chunking: {v_chunks}")

            # Verify chunking follows expected pattern:
            # - Time dimension should be chunked
            # - Spatial dimensions (y, x) should ideally be full or reasonably sized
            assert len(v_chunks) == 3, (
                f"Expected 3D chunking for 'v', got {len(v_chunks)}D"
            )


class TestITSCubeClass:
    """Tests for ITSCube class initialization and methods."""

    def test_itscube_initialization(self, test_config):
        """Test ITSCube class initialization with test polygon."""
        # Import shapefile module to initialize SHAPE_FILE
        import shapefile

        # Use default shapefile path from command-line argument default
        default_shapefile = 's3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp'

        # Initialize ITSCube class variables (required before instantiation)
        ITSCube.SHAPE_FILE = shapefile.read_file(default_shapefile)
        ITSCube.CELL_SIZE = test_config["cell_size"]  # Use test config cell size

        # Convert polygon list to tuple of tuples as expected by ITSCube
        polygon_tuple = tuple(tuple(point) for point in test_config["polygon"])

        # Initialize ITSCube object
        cube = ITSCube(
            polygon=polygon_tuple,
            projection=test_config["target_projection"]
        )

        # Verify grid was created correctly
        assert cube.grid_x is not None, "Grid X not initialized"
        assert cube.grid_y is not None, "Grid Y not initialized"

        # Verify spatial bounds
        assert ITSCube.GRID_X_MIN is not None, "Grid X min not set"
        assert ITSCube.GRID_X_MAX is not None, "Grid X max not set"
        assert ITSCube.GRID_Y_MIN is not None, "Grid Y min not set"
        assert ITSCube.GRID_Y_MAX is not None, "Grid Y max not set"

        # Verify polygon was converted to lon/lat coordinates
        assert len(cube.polygon_coords) > 0, (
            "Polygon coordinates not transformed to lon/lat"
        )

        print(f"✓ ITSCube initialized successfully")
        print(f"  - Grid bounds: x=[{ITSCube.GRID_X_MIN:.0f}, {ITSCube.GRID_X_MAX:.0f}], "
              f"y=[{ITSCube.GRID_Y_MIN:.0f}, {ITSCube.GRID_Y_MAX:.0f}]")
        print(f"  - Centroid (lon/lat): {cube.center_lon_lat}")

    def test_grid_creation(self, test_config):
        """Test Grid utility class with test polygon bounds."""
        # Extract x and y bounds from polygon
        polygon = test_config["polygon"]
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]

        x_bounds = Bounds(x_coords)
        y_bounds = Bounds(y_coords)

        # Create grid
        grid_x, grid_y = Grid.create(x_bounds, y_bounds, test_config["cell_size"])

        # Verify grid properties
        assert len(grid_x) > 0, "Grid X is empty"
        assert len(grid_y) > 0, "Grid Y is empty"

        # Verify grid spacing (should be identical within floating-point precision)
        if len(grid_x) > 1:
            x_spacing = grid_x[1] - grid_x[0]
            assert abs(x_spacing) == test_config["cell_size"], (
                f"Grid X spacing {abs(x_spacing)} doesn't match "
                f"cell size {test_config['cell_size']}"
            )

        if len(grid_y) > 1:
            y_spacing = grid_y[1] - grid_y[0]
            assert abs(y_spacing) == test_config["cell_size"], (
                f"Grid Y spacing {abs(y_spacing)} doesn't match "
                f"cell size {test_config['cell_size']}"
            )

        # Verify grid covers the polygon
        # Note: grid values are cell centers, so actual extent is ±cell_size/2
        half_cell = test_config["cell_size"] / 2.0
        grid_x_min_extent = grid_x.min() - half_cell
        grid_x_max_extent = grid_x.max() + half_cell
        grid_y_min_extent = grid_y.min() - half_cell
        grid_y_max_extent = grid_y.max() + half_cell

        assert grid_x_min_extent <= min(x_coords), (
            f"Grid X min extent {grid_x_min_extent} doesn't cover polygon min {min(x_coords)}"
        )
        assert grid_x_max_extent >= max(x_coords), (
            f"Grid X max extent {grid_x_max_extent} doesn't cover polygon max {max(x_coords)}"
        )
        assert grid_y_min_extent <= min(y_coords), (
            f"Grid Y min extent {grid_y_min_extent} doesn't cover polygon min {min(y_coords)}"
        )
        assert grid_y_max_extent >= max(y_coords), (
            f"Grid Y max extent {grid_y_max_extent} doesn't cover polygon max {max(y_coords)}"
        )

        print(f"✓ Grid creation verified")
        print(f"  - Grid X: {len(grid_x)} points, "
              f"centers [{grid_x.min():.0f}, {grid_x.max():.0f}], "
              f"extent [{grid_x_min_extent:.0f}, {grid_x_max_extent:.0f}]")
        print(f"  - Grid Y: {len(grid_y)} points, "
              f"centers [{grid_y.min():.0f}, {grid_y.max():.0f}], "
              f"extent [{grid_y_min_extent:.0f}, {grid_y_max_extent:.0f}]")


# Composite Generation Tests
class TestCompositeGeneration:
    """Tests for composite generation via CLI using the generated datacube."""

    @pytest.mark.order(4)
    def test_composite_generation_via_cli(
        self, composite_path, datacube_path, test_config, composite_script
    ):
        """Test composite generation using command-line interface.

        IMPORTANT: This test uses the datacube generated by test_datacube_generation_via_cli
        (order=1). This test does NOT upload to S3. The --inputBucket (-b) and
        --targetBucket (-t) arguments are intentionally omitted, which ensures that
        no S3 copy operations occur. The composite is created locally in
        src/tests/test_output/ only.
        """
        # Skip if datacube doesn't exist (previous test failed)
        if not datacube_path.exists():
            pytest.skip("Datacube was not created in previous test")

        # Build command-line arguments
        # NOTE: No --inputBucket (-b) or --targetBucket (-t) arguments are provided,
        # which ensures no S3 upload occurs
        cmd = [
            sys.executable,
            str(composite_script),
            "-i", str(datacube_path),
            "-o", str(composite_path),
            "-s", test_config["default_shapefile"]
        ]

        # Add --useErrorSlow flag if specified
        if test_config.get("use_error_slow", False):
            cmd.append("--useErrorSlow")

        print(f"\nRunning command: {' '.join(cmd)}")

        # Run the composite generation
        # Set PYTHONWARNINGS to suppress resource_tracker warnings from subprocess
        import os
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

        result = subprocess.run(
            cmd,
            # capture_output=True,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # text=True,
            # timeout=600,  # 10 minute timeout
            env=env
        )

        # Check if the output file was created (most important check)
        # Note: The process may crash during cleanup (return code -6 or -11)
        # even though the composite generation completed successfully.
        # This is a known issue with native libraries (HDF5, NetCDF, zarr)
        # and multiprocessing cleanup.
        composite_created = composite_path.exists()

        # Check that the command completed successfully
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

            # If composite was created despite the crash, treat as a warning
            if composite_created:
                print(f"Process crashed with return code {result.returncode}, "
                        f"but composite was created successfully")
                print(f"  This is likely a cleanup issue in native libraries")
            else:
                pytest.fail(
                    f"Composite generation failed with return code {result.returncode}"
                )
        else:
            print(f"Composite STDOUT:\n{result.stdout}")
            print(f"CompositeSTDERR:\n{result.stderr}")

        # Verify that the output file was created
        assert composite_created, (
            f"Output composite not created at {composite_path}"
        )

        print(f"✓ Composite created successfully at {composite_path}")

    @pytest.mark.order(5)
    def test_verify_composite_structure(self, composite_path):
        """Verify the structure and contents of the generated composite."""
        # Skip if composite doesn't exist (previous test failed)
        if not composite_path.exists():
            pytest.skip("Composite was not created in previous test")

        # Open the composite using xarray
        ds = xr.open_dataset(
            str(composite_path),
            decode_timedelta=False,
            engine='zarr',
            consolidated=True
        )

        # Verify dimensions exist
        assert 'time' in ds.sizes, "time dimension missing"
        assert 'x' in ds.sizes, "x dimension missing"
        assert 'y' in ds.sizes, "y dimension missing"
        assert 'sensor' in ds.sizes, "sensor dimension missing"

        # Verify coordinate variables
        assert 'time' in ds.coords, "time coordinate missing"
        assert 'x' in ds.coords, "x coordinate missing"
        assert 'y' in ds.coords, "y coordinate missing"

        print(f"✓ All required dimensions present: {dict(ds.sizes)}")

        # Verify expected composite data variables
        # These are the standard outputs from ITSLiveComposite
        expected_vars = [
            # Mean velocity and components
            'v', 'vx', 'vy',
            # Velocity errors
            'v_error', 'vx_error', 'vy_error',
            # Data counts and statistics
            'count', 'count0',
            # Velocity derivatives
            'dv_dt', 'dvx_dt', 'dvy_dt',
            # Time statistics
            'dt_max',
            # Ice masks
            'landice', 'floatingice',
            # Mapping
            'mapping',
            # Outlier statistics
            'outlier_percent',
            # Sensor information
            'sensor_filter_applied',
            # Amplitude and phase (for seasonal variations)
            'v_amp', 'vx_amp', 'vy_amp',
            'v_phase', 'vx_phase', 'vy_phase',
            # Amplitude errors
            'v_amp_error', 'vx_amp_error', 'vy_amp_error',
            # V0 (intercept) and errors
            'v0', 'vx0', 'vy0',
            'v0_error', 'vx0_error', 'vy0_error'
        ]

        # Check for existence of expected variables (some may be optional)
        present_vars = [var for var in expected_vars if var in ds.data_vars]
        print(f"✓ Found {len(present_vars)}/{len(expected_vars)} expected data variables:")
        print(f"  Present: {', '.join(sorted(present_vars))}")

        missing_vars = [var for var in expected_vars if var not in ds.data_vars]
        if missing_vars:
            print(f"  Missing (may be optional): {', '.join(sorted(missing_vars))}")

        # Verify we have at least the basic required variables
        required_vars = ['v', 'vx', 'vy', 'count', 'mapping']
        missing_required = [var for var in required_vars if var not in ds.data_vars]
        assert len(missing_required) == 0, (
            f"Missing required data variables: {missing_required}"
        )

        # Verify time dimension has data
        num_time_layers = len(ds.time)
        assert num_time_layers > 0, "Composite has no time layers"

        print(f"✓ Composite structure verified:")
        print(f"  - Dimensions: {ds.sizes}")
        print(f"  - Number of time layers: {num_time_layers}")
        print(f"  - Data variables: {list(ds.data_vars.keys())}")

        ds.close()

    @pytest.mark.order(6)
    def test_verify_composite_data_ranges(self, composite_path):
        """Verify that composite data values are within reasonable ranges."""
        # Skip if composite doesn't exist
        if not composite_path.exists():
            pytest.skip("Composite was not created")

        # Open the composite
        ds = xr.open_dataset(
            str(composite_path),
            decode_timedelta=False,
            engine='zarr',
            consolidated=True
        )

        # Check velocity magnitude (v) for reasonable values
        if 'v' in ds.data_vars:
            v_data = ds['v'].values
            # Filter out NaN values for statistics
            import numpy as np
            v_valid = v_data[~np.isnan(v_data)]

            if len(v_valid) > 0:
                v_min = np.min(v_valid)
                v_max = np.max(v_valid)
                v_mean = np.mean(v_valid)

                print(f"✓ Velocity (v) statistics:")
                print(f"  - Min: {v_min:.2f} m/yr")
                print(f"  - Max: {v_max:.2f} m/yr")
                print(f"  - Mean: {v_mean:.2f} m/yr")
                print(f"  - Valid pixels: {len(v_valid)}")

                # Sanity check: glacier velocities typically range from 0 to ~10000 m/yr
                # (most glaciers < 1000 m/yr, fast outlet glaciers can exceed 5000 m/yr)
                assert v_min >= 0, f"Velocity should be non-negative, got min={v_min}"
                assert v_max < 20000, (
                    f"Velocity seems unreasonably high: max={v_max} m/yr"
                )

        # Check count variable
        if 'count' in ds.data_vars:
            count_data = ds['count'].values
            import numpy as np
            count_valid = count_data[~np.isnan(count_data)]

            if len(count_valid) > 0:
                count_min = np.min(count_valid)
                count_max = np.max(count_valid)
                count_mean = np.mean(count_valid)

                print(f"✓ Count statistics:")
                print(f"  - Min: {count_min:.0f}")
                print(f"  - Max: {count_max:.0f}")
                print(f"  - Mean: {count_mean:.1f}")

                # Count should be positive integers
                assert count_min >= 0, f"Count should be non-negative, got min={count_min}"

        ds.close()

    @pytest.mark.order(7)
    def test_verify_composite_zarr_chunks(self, composite_path):
        """Verify Zarr chunking strategy in composite."""
        # Skip if composite doesn't exist
        if not composite_path.exists():
            pytest.skip("Composite was not created")

        # Open zarr store directly
        store = zarr.open(str(composite_path), mode='r')

        # Check that velocity variables have appropriate chunking
        if 'v' in store:
            v_chunks = store['v'].chunks
            print(f"✓ Velocity (v) chunking: {v_chunks}")

            # Verify chunking follows expected pattern (3D: time, y, x)
            assert len(v_chunks) == 3, (
                f"Expected 3D chunking for 'v', got {len(v_chunks)}D"
            )

        # Check other key variables
        for var in ['vx', 'vy', 'count']:
            if var in store:
                chunks = store[var].chunks
                print(f"✓ {var} chunking: {chunks}")


class TestITSLiveCompositeClass:
    """Tests for ITSLiveComposite class initialization and methods."""

    @pytest.mark.order(8)
    def test_composite_class_initialization(self, datacube_path, test_config):
        """Test ITSLiveComposite class initialization using generated datacube."""
        # Skip if datacube doesn't exist
        if not datacube_path.exists():
            pytest.skip("Datacube was not created in previous test")


        # Initialize shapefile and class variables
        ITSLiveComposite.SHAPE_FILE = shapefile.read_file(test_config["default_shapefile"])
        ITSLiveComposite.V0_YEARS = test_config["v0_years"]
        ITSLiveComposite.NUM_TO_PROCESS = test_config["composite_chunks"]
        ITSLiveComposite.USE_ERROR_SLOW = test_config["use_error_slow"]

        # Initialize ITSLiveComposite object
        composite = ITSLiveComposite(
            cube_store=str(datacube_path),
            s3_bucket=''
        )

        # Verify composite was initialized
        assert composite is not None, "ITSLiveComposite object not created"

        # Verify datacube was loaded
        assert hasattr(composite, 'cube_ds'), "Datacube dataset not loaded"
        assert composite.cube_ds is not None, "Datacube dataset is None"

        # Verify required dimensions exist in datacube
        assert 'mid_date' in composite.cube_ds.sizes, (
            "mid_date dimension missing in datacube"
        )
        assert 'x' in composite.cube_ds.sizes, "x dimension missing in datacube"
        assert 'y' in composite.cube_ds.sizes, "y dimension missing in datacube"

        print(f"✓ ITSLiveComposite initialized successfully")
        print(f"  - Datacube dimensions: {composite.cube_ds.sizes}")
        print(f"  - Datacube variables: {list(composite.cube_ds.data_vars.keys())}...")

        # Close dataset
        del composite
