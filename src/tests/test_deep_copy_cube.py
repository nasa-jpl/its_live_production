"""
Unit and integration tests for deep_copy_cube.py using pytest.

Tests materializing a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube to verify:
- Pure helper functions (split_vars_by_time, build_encoding,
  resolve_output_store) on synthetic, network-free data
- End-to-end CLI materialization: the 'time' dimension is preserved (not
  renamed), static cube-level variables (mapping/landice/floatingice) are
  written exactly once across batches, chunking matches deep_copy_cube.py's
  own TIME_CHUNK_VALUE/X_Y_CHUNK_VALUE/TIME_CHUNK_VALUE_1D scheme (which
  deliberately diverges from itscube.py's X_Y_CHUNK_VALUE -- 8 vs. 10 -- to
  divide the 61.44km chunk-aligned production grid evenly; this pipeline is
  meant to eventually replace itscube.py's regular datacube generation
  entirely, so it should not depend on it), and materialized pixel values
  match the source virtual cube's own on-demand reads

Authors: Masha Liukis
"""
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
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

import utils
from itscube_types import Vars
import deep_copy_cube as dcc


# ---------------------------------------------------------------------------
# Pure helper-function tests: no network, no icechunk -- synthetic dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cube():
    """Small synthetic virtual-cube-shaped dataset: one 3D (time, y, x) var,
    one 1D (time,) var, one static 2D (y, x) var, and one scalar var --
    enough to exercise split_vars_by_time()/build_encoding() without any
    icechunk/S3 access."""
    time = np.array(
        ['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]'
    )
    y = np.arange(20, dtype='float64')
    x = np.arange(30, dtype='float64')

    return xr.Dataset(
        data_vars={
            Vars.vx: (('time', 'y', 'x'), np.zeros((3, 20, 30), dtype='int16')),
            Vars.url: (('time',), np.array(['a', 'b', 'c'])),
            'landice': (('y', 'x'), np.zeros((20, 30), dtype='uint8')),
            'mapping': ((), ''),
        },
        coords={'time': time, 'y': y, 'x': x},
    )


class TestDeepCopyCubeHelpers:
    """Unit tests for deep_copy_cube.py's pure helper functions."""

    def test_split_vars_by_time(self, synthetic_cube):
        time_vars, static_vars = dcc.split_vars_by_time(synthetic_cube)

        assert set(time_vars) == {Vars.vx, Vars.url}
        assert set(static_vars) == {'landice', 'mapping'}

    def test_build_encoding_3d_var_chunks(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[Vars.vx]['chunks'] == (20000, 10, 10)

    def test_build_encoding_1d_var_chunks(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[Vars.url]['chunks'] == (200000,)

    def test_build_encoding_time_coord_chunk_uses_time_chunk_1d(self, synthetic_cube):
        # The 'time' coordinate itself is fixed to time_chunk_1d too (not the
        # cube's current size), for the same future-growth reason as the
        # other 1D (time,) variables.
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[utils.Coords.TIME]['chunks'] == (200000,)

    def test_build_encoding_static_var_full_extent(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding['landice']['chunks'] == (20, 30)

    def test_build_encoding_coords_full_extent(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[utils.Coords.X]['chunks'] == (30,)
        assert encoding[utils.Coords.Y]['chunks'] == (20,)

    def test_build_encoding_coords_disable_fill_value(self, synthetic_cube):
        # x/y have no missing values; suppress xarray's default _FillValue=NaN.
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[utils.Coords.X][utils.OutputFormat.fill_value] is None
        assert encoding[utils.Coords.Y][utils.OutputFormat.fill_value] is None

    def test_build_encoding_scalar_var_skipped(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert 'mapping' not in encoding

    def test_build_encoding_time_chunk_not_capped_by_cube_size(self, synthetic_cube):
        # synthetic_cube only has 3 layers, but time_chunk (20000) must be
        # used as-is: a Zarr chunk grid is fixed at creation, so capping it at
        # however many layers exist right now would wall in a too-small
        # chunk size forever once the cube grows via a later deep_copy_update.py
        # append.
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert encoding[Vars.vx]['chunks'][0] == 20000

    def test_build_encoding_no_shards_by_default(self, synthetic_cube):
        # xy_shard_multiplier defaults to 1: unsharded.
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=10, time_chunk_1d=200000
        )

        assert 'shards' not in encoding[Vars.vx]

    def test_build_encoding_rejects_shard_multiplier_below_one(self, synthetic_cube):
        with pytest.raises(ValueError, match="xy_shard_multiplier must be >= 1"):
            dcc.build_encoding(
                synthetic_cube, time_chunk=20000, xy_chunk=10,
                time_chunk_1d=200000, xy_shard_multiplier=0
            )

    def test_build_encoding_adds_shards_for_3d_var(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=8,
            time_chunk_1d=200000, xy_shard_multiplier=4
        )

        assert encoding[Vars.vx]['shards'] == (20000, 32, 32)

    def test_build_encoding_shard_time_extent_matches_chunk_time_extent(self, synthetic_cube):
        # Critical invariant: never group multiple time-chunks into one
        # shard, so a shard's time extent must always equal the chunk's.
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=8,
            time_chunk_1d=200000, xy_shard_multiplier=4
        )

        assert encoding[Vars.vx]['shards'][0] == encoding[Vars.vx]['chunks'][0]

    def test_build_encoding_no_shards_for_1d_or_static_vars(self, synthetic_cube):
        encoding = dcc.build_encoding(
            synthetic_cube, time_chunk=20000, xy_chunk=8,
            time_chunk_1d=200000, xy_shard_multiplier=4
        )

        assert 'shards' not in encoding[Vars.url]
        assert 'shards' not in encoding['landice']

    def test_resolve_output_store_removes_existing_local_dir(self, tmp_path):
        existing = tmp_path / "existing.zarr"
        existing.mkdir()
        (existing / "marker.txt").write_text("stale")

        result = dcc.resolve_output_store(str(existing))

        assert result == str(existing)
        assert not existing.exists(), "Pre-existing local output store should be removed"

    def test_resolve_output_store_leaves_s3_path_untouched(self):
        s3_path = "s3://its-live-data/test-space/deep_copy_test.zarr"
        assert dcc.resolve_output_store(s3_path) == s3_path

    def test_resolve_output_store_refuses_existing_s3_path(self, monkeypatch):
        import s3fs

        monkeypatch.setattr(s3fs.S3FileSystem, "exists", lambda self, path: True)

        with pytest.raises(RuntimeError, match="already exists in S3"):
            dcc.resolve_output_store("s3://its-live-data/some/existing.zarr")


class TestDeepCopyCubeLocalStaging:
    """Unit test for the --local-staging-dir option's input validation. No
    network/S3 access: deep_copy_cube() raises before touching the
    (nonexistent) input store or performing any upload."""

    def test_local_staging_dir_rejected_with_local_output_store(self, tmp_path):
        with pytest.raises(ValueError, match="--local-staging-dir only applies"):
            dcc.deep_copy_cube(
                input_store="unused.icechunk",
                output_store=str(tmp_path / "local_output.zarr"),
                bucket_prefix="s3://its-live-data/",
                batch_size=1000,
                time_chunk=20000,
                xy_chunk=10,
                time_chunk_1d=200000,
                local_staging_dir=str(tmp_path / "staging"),
            )


# ---------------------------------------------------------------------------
# Integration tests: build a real (small) virtual cube from S3 granules, then
# deep-copy it via the CLI and verify the materialized output.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_config():
    """Test configuration parameters."""
    return {
        "projection": "3031",
        "polygon": [
            [-1658887.5, -430072.5],
            [-1597447.5, -430072.5],
            [-1597447.5, -368632.5],
            [-1658887.5, -368632.5],
            [-1658887.5, -430072.5]
        ],
        "granules_file": "virtual_input_4files.json",
        "num_granules": 4,
        "virtual_store": "deep_copy_test_source.icechunk",
        "output_store": "deep_copy_test_output.zarr",
        "batch_size": 2,  # smaller than num_granules: exercises the append path
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
def deep_copy_script():
    """Path to deep_copy_cube.py script."""
    return Path(__file__).parent.parent / "deep_copy_cube.py"


@pytest.fixture(scope="session")
def granules_file(test_config):
    """Path to granules JSON file."""
    granules_path = Path(__file__).parent.parent / test_config["granules_file"]
    if not granules_path.exists():
        pytest.skip(f"Granules file not found: {granules_path}")
    return granules_path


@pytest.fixture(scope="session")
def virtual_cube_path(test_output_dir, test_config):
    """Path to the (source) virtual datacube to be deep-copied."""
    return test_output_dir / test_config["virtual_store"]


@pytest.fixture(scope="session")
def deep_copy_cube_path(test_output_dir, test_config):
    """Path to the materialized deep-copy datacube."""
    return test_output_dir / test_config["output_store"]


class TestDeepCopyCubeIntegration:
    """End-to-end tests: build a small virtual cube, deep-copy it via the
    CLI, and verify the materialized output."""

    @pytest.mark.order(1)
    def test_build_source_virtual_cube(
        self, virtual_cube_path, test_config, virtual_cube_script, granules_file
    ):
        """Build the small source virtual datacube that subsequent tests
        deep-copy. Reuses the same polygon/projection as
        test_virtual_cube_generation.py, paired with the smaller 4-granule
        fixture for a fast test."""
        if virtual_cube_path.exists():
            shutil.rmtree(virtual_cube_path)

        cmd = [
            sys.executable,
            str(virtual_cube_script),
            "--polygon", json.dumps(test_config["polygon"]),
            "--granules-file", str(granules_file),
            "--output-store", str(virtual_cube_path),
            "--projection", test_config["projection"]
        ]

        print(f"\nRunning command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            pytest.fail(f"Virtual datacube generation failed with return code {result.returncode}")

        assert virtual_cube_path.exists(), f"Virtual cube store not created at {virtual_cube_path}"

    @pytest.mark.order(2)
    def test_deep_copy_via_cli(
        self, virtual_cube_path, deep_copy_cube_path, deep_copy_script, test_config
    ):
        """Run deep_copy_cube.py against the source virtual cube with a
        batch size smaller than the number of layers, so the run exercises
        both the initial (mode='w') and appended batch-write paths."""
        if deep_copy_cube_path.exists():
            shutil.rmtree(deep_copy_cube_path)

        cmd = [
            sys.executable,
            str(deep_copy_script),
            "--input-store", str(virtual_cube_path),
            "--output-store", str(deep_copy_cube_path),
            "--batch-size", str(test_config["batch_size"]),
        ]

        print(f"\nRunning command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            pytest.fail(f"deep_copy_cube.py failed with return code {result.returncode}")

        assert deep_copy_cube_path.exists(), f"Deep-copy store not created at {deep_copy_cube_path}"

    @pytest.mark.order(3)
    def test_deep_copy_keeps_time_dimension(self, deep_copy_cube_path, test_config):
        """The deep-copy output must keep the virtual cube's 'time' dimension
        as-is (no rename to 'mid_date')."""
        cube = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)

        assert 'time' in cube.dims, "Deep-copy cube should keep the 'time' dimension"
        assert 'mid_date' not in cube.dims, "Deep-copy cube should NOT be renamed to 'mid_date'"
        assert cube.sizes['time'] == test_config["num_granules"]

    @pytest.mark.order(4)
    def test_deep_copy_static_vars_written_once(self, deep_copy_cube_path, virtual_cube_path):
        """Static cube-level variables (mapping/landice/floatingice) must be
        present exactly once (not duplicated by the batched append writes)
        and match the source virtual cube's spatial shape."""
        deep = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)
        virt = dcc.open_virtual_cube(str(virtual_cube_path), 's3://its-live-data/')

        for var in ('mapping', 'landice', 'floatingice'):
            assert var in deep.variables, f"Missing static variable: {var}"
            assert deep[var].shape == virt[var].shape, \
                f"{var} shape mismatch: {deep[var].shape} vs source {virt[var].shape}"
            # No 'time' dimension: confirms it was written once, not appended
            assert 'time' not in deep[var].dims, f"{var} unexpectedly has a 'time' dimension"

    @pytest.mark.order(5)
    def test_deep_copy_chunking_matches_itscube_constants(self, deep_copy_cube_path):
        """Chunking must match deep_copy_cube.py's own TIME_CHUNK_VALUE/
        X_Y_CHUNK_VALUE/TIME_CHUNK_VALUE_1D scheme -- the full fixed values,
        not capped at this test cube's small (4-granule) layer count, so a
        real cube has room to grow via later deep_copy_update.py appends
        without its chunk grid being walled in. Deliberately compares
        against dcc's own module constants, not itscube.py's --
        X_Y_CHUNK_VALUE is intentionally 8 here vs. itscube.py's 10 (see
        module docstring)."""
        cube = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)

        expected_3d_chunks = (
            dcc.TIME_CHUNK_VALUE,
            dcc.X_Y_CHUNK_VALUE,
            dcc.X_Y_CHUNK_VALUE
        )
        assert cube[Vars.vx].encoding['chunks'] == expected_3d_chunks, \
            f"vx chunks {cube[Vars.vx].encoding['chunks']} != expected {expected_3d_chunks}"

        expected_1d_chunks = (dcc.TIME_CHUNK_VALUE_1D,)
        assert cube[Vars.url].encoding['chunks'] == expected_1d_chunks, \
            f"{Vars.url} chunks {cube[Vars.url].encoding['chunks']} != expected {expected_1d_chunks}"

        assert cube['time'].encoding['chunks'] == expected_1d_chunks, \
            f"time chunks {cube['time'].encoding['chunks']} != expected {expected_1d_chunks}"

        assert cube['x'].encoding['chunks'] == (cube.sizes['x'],), \
            "x coordinate should be chunked at full extent"
        assert cube['y'].encoding['chunks'] == (cube.sizes['y'],), \
            "y coordinate should be chunked at full extent"

    @pytest.mark.order(6)
    def test_deep_copy_values_match_source_virtual_cube(
        self, deep_copy_cube_path, virtual_cube_path
    ):
        """Materialized pixel values (and dtype) must agree exactly with the
        source virtual cube's own on-demand (referenced) reads.

        Both sides are read with mask_and_scale=False so the deep copy's raw
        on-disk dtype (int16, preserved by open_virtual_cube's
        mask_and_scale=False) is compared against the source's raw dtype --
        not a float32 promotion from CF decoding on either read."""
        deep = xr.open_zarr(
            str(deep_copy_cube_path), zarr_format=3, consolidated=True,
            mask_and_scale=False
        )
        virt = dcc.open_virtual_cube(str(virtual_cube_path), 's3://its-live-data/')

        deep_vx = deep[Vars.vx].values
        virt_vx = virt[Vars.vx].load().values

        assert deep_vx.dtype == virt_vx.dtype, \
            f"vx dtype mismatch: deep-copy {deep_vx.dtype} vs source {virt_vx.dtype}"
        assert np.array_equal(deep_vx, virt_vx), \
            "vx values differ between deep-copy and source virtual cube"

    @pytest.mark.order(6)
    def test_deep_copy_fill_value_convention(self, deep_copy_cube_path):
        """The deep copy must follow itscube.py's fill convention: integer
        variables use the 'missing_value' attribute (never '_FillValue', which
        would make xarray decode them as float), floating point variables use
        '_FillValue', and the sentinels decode to NaN on a default CF read."""
        raw = xr.open_zarr(
            str(deep_copy_cube_path), zarr_format=3, consolidated=True,
            mask_and_scale=False
        )

        int_var = Vars.vx      # int16 velocity component
        assert raw[int_var].dtype.kind in ('i', 'u')
        assert utils.Missing.name in raw[int_var].attrs, \
            f"{int_var} (int) must carry a '{utils.Missing.name}' attribute"
        assert utils.OutputFormat.fill_value not in raw[int_var].attrs, \
            f"{int_var} (int) must NOT carry a '{utils.OutputFormat.fill_value}' attribute"

        float_var = 'vr_error_stationary'  # float, was the original crash case
        assert raw[float_var].dtype.kind == 'f'
        assert utils.OutputFormat.fill_value in raw[float_var].attrs, \
            f"{float_var} (float) must carry a '{utils.OutputFormat.fill_value}' attribute"

        # Default CF read masks the int sentinels to NaN (promoting to float).
        decoded = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)
        sentinel = raw[int_var].attrs[utils.Missing.name]
        n_sentinel = int((raw[int_var].values == sentinel).sum())
        n_nan = int(np.isnan(decoded[int_var].values).sum())
        assert n_sentinel == n_nan, \
            f"{int_var}: {n_sentinel} sentinels but {n_nan} NaNs after CF decode"

        # x/y coordinates have no missing values: xarray's default
        # _FillValue=NaN on float coordinates is suppressed.
        for coord in (utils.Coords.X, utils.Coords.Y):
            assert utils.OutputFormat.fill_value not in raw[coord].attrs, \
                f"{coord} coordinate should not carry a '{utils.OutputFormat.fill_value}' attribute"

    @pytest.mark.order(7)
    def test_deep_copy_variable_set_matches_source(self, deep_copy_cube_path, virtual_cube_path):
        """The deep copy must carry over the exact same set of data variables
        as the source virtual cube (nothing dropped or added)."""
        deep = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)
        virt = dcc.open_virtual_cube(str(virtual_cube_path), 's3://its-live-data/')

        assert set(deep.data_vars) == set(virt.data_vars), \
            f"Variable set differs.\n  deep-copy only: {set(deep.data_vars) - set(virt.data_vars)}\n" \
            f"  source only: {set(virt.data_vars) - set(deep.data_vars)}"

    @pytest.mark.order(8)
    def test_deep_copy_global_attrs(self, deep_copy_cube_path, test_config):
        """The deep copy should carry over the source cube's global
        attributes and stamp a fresh 'date_updated'."""
        cube = xr.open_zarr(str(deep_copy_cube_path), zarr_format=3, consolidated=True)

        for attr in ['date_created', 'date_updated', 'title', 'institution', 'projection']:
            assert attr in cube.attrs, f"Missing required attribute: {attr}"

        assert cube.attrs['projection'] == test_config['projection']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
