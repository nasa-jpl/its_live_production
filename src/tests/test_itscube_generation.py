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
from pathlib import Path
import pytest
import xarray as xr
import zarr

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from itscube import ITSCube
from grid import Bounds, Grid


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
        "chunks": 100  # Number of granules to write at a time
    }


@pytest.fixture(scope="session")
def test_output_dir():
    """Create and clean up test output directory."""
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after all tests
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture(scope="session")
def itscube_script():
    """Path to itscube.py script."""
    return Path(__file__).parent.parent / "itscube.py"


@pytest.fixture(scope="session")
def datacube_path(test_output_dir, test_config):
    """Path to the generated datacube."""
    return test_output_dir / test_config["output_file"]


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
            engine='zarr',
            consolidated=True
        )

        # Verify dimensions exist
        assert 'mid_date' in ds.dims, "mid_date dimension missing"
        assert 'x' in ds.dims, "x dimension missing"
        assert 'y' in ds.dims, "y dimension missing"

        # Verify coordinate variables
        assert 'mid_date' in ds.coords, "mid_date coordinate missing"
        assert 'x' in ds.coords, "x coordinate missing"
        assert 'y' in ds.coords, "y coordinate missing"

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
        print(f"  - Dimensions: {dict(ds.dims)}")
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
