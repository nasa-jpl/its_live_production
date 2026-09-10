"""
Unit and local (no-network) integration tests for deep_copy_cube_per_var.py
using pytest.

Like test_deep_copy_cube_tiled.py, this module never touches S3 or
icechunk: deep_copy_cube_per_var()'s end-to-end tests monkeypatch
open_virtual_cube() to return a synthetic, dask-backed xr.Dataset shaped
like a real virtual cube -- per feedback_no_s3_writing_tests, no network
dependency needed to exercise the per-variable, half-chunk-write mechanics
for real, against a real local zarr v3 store.

Covers:
- _write_var_in_chunks(): the half-chunk region-write boundary math in
  isolation, against a directly-templated local store -- even chunk size
  (write_span divides the chunk evenly, matching the module docstring's
  "bounded, fixed 2x" claim), odd chunk size (write_span desyncs from real
  chunk boundaries, but values must still come out exactly correct),
  num_layers smaller than write_span, chunk_size larger than total_layers,
  and the same logic applied to a 1D (time,) var.
- deep_copy_cube_per_var(): static vars written once, multiple 3D and
  multiple 1D variables all processed correctly in sequence (no
  cross-variable leakage), chunking/sharding matches build_encoding(),
  --num-layers truncation, and an odd --time-chunk-value end to end.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

warnings.filterwarnings('ignore', message='resource_tracker: There appear to be.*leaked semaphore')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import deep_copy_cube_per_var as dcpv


# ---------------------------------------------------------------------------
# _write_var_in_chunks(): pure boundary-math tests against a directly
# templated local store -- no full deep_copy_cube_per_var() run needed.
# ---------------------------------------------------------------------------

Y_SIZE = 8
X_SIZE = 6


def _build_template_and_source(tmp_path, var_name, total_layers, chunk_size, is_3d):
    """Build (a) an in-memory, dask-backed source dataset holding the
    "real" per-layer values _write_var_in_chunks() reads from, and (b) an
    already-templated local zarr v3 store (mode='w', compute=False)
    declaring var_name's shape/dtype/chunks up front but with no pixel data
    written yet -- mirroring deep_copy_cube_per_var()'s own template step,
    so _write_var_in_chunks()'s mode='r+' region writes have a real store
    to land into.

    Dask-chunked at one chunk per layer (full spatial extent for 3D vars):
    to_zarr(compute=False) only defers writing for dask-backed variables
    (a hard requirement, not a graceful degrade -- see
    deep_copy_cube_tiled.py's test fixture for the same reasoning), and
    this deliberately mismatches `chunk_size` the same way a real virtual
    cube's fixed per-granule chunking does, exercising safe_chunks=False.
    """
    if is_3d:
        dims = (utils.Coords.TIME, utils.Coords.Y, utils.Coords.X)
        shape = (total_layers, Y_SIZE, X_SIZE)
        chunks = (chunk_size, Y_SIZE, X_SIZE)
        coords = {
            utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]'),
            utils.Coords.Y: np.arange(Y_SIZE, dtype='float64'),
            utils.Coords.X: np.arange(X_SIZE, dtype='float64'),
        }
        dask_chunks = {utils.Coords.TIME: 1, utils.Coords.Y: Y_SIZE, utils.Coords.X: X_SIZE}
    else:
        dims = (utils.Coords.TIME,)
        shape = (total_layers,)
        chunks = (chunk_size,)
        coords = {utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]')}
        dask_chunks = {utils.Coords.TIME: 1}

    values = (np.arange(int(np.prod(shape))).reshape(shape) % 500).astype('int16')
    source = xr.Dataset({var_name: (dims, values)}, coords=coords)
    source = source.chunk(dask_chunks)

    output_store = str(tmp_path / "template.zarr")
    source.to_zarr(
        output_store, mode='w', compute=False,
        encoding={var_name: {'chunks': chunks}},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    return source, output_store


class TestWriteVarInChunksBoundaries:

    def test_even_chunk_size_no_gaps_or_overlaps(self, tmp_path):
        # chunk_size=6 -> write_span=3, dividing evenly into each 6-layer
        # chunk (exactly 2 writes per chunk, matching the module docstring's
        # "bounded, fixed 2x" claim).
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=12, chunk_size=6, is_3d=True
        )

        dcpv._write_var_in_chunks(source, output_store, 'vx', chunk_size=6, total_layers=12)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_odd_chunk_size_still_produces_correct_values(self, tmp_path):
        # chunk_size=5 -> write_span=2, which does NOT divide evenly into
        # the 5-layer chunk -- writes desync from real chunk boundaries
        # (e.g. the write [4,6) straddles the chunk-0/chunk-1 boundary at
        # layer 5). This means a chunk can be touched by more than the
        # documented 2 writes, but the written VALUES must still be
        # exactly correct.
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=10, chunk_size=5, is_3d=True
        )

        dcpv._write_var_in_chunks(source, output_store, 'vx', chunk_size=5, total_layers=10)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_num_layers_smaller_than_write_span(self, tmp_path):
        # write_span = 10 // 2 = 5, but total_layers=3 -- a single write
        # must cover the whole (short) range, not loop past it.
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=3, chunk_size=10, is_3d=True
        )

        dcpv._write_var_in_chunks(source, output_store, 'vx', chunk_size=10, total_layers=3)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_chunk_size_larger_than_total_layers(self, tmp_path):
        # chunk_size=100 with only 7 real layers: write_span=50, so the
        # single write [0,7) must cover everything in one shot.
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=7, chunk_size=100, is_3d=True
        )

        dcpv._write_var_in_chunks(source, output_store, 'vx', chunk_size=100, total_layers=7)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_1d_var_boundary_math_matches_source(self, tmp_path):
        # Same half-chunk logic applies identically to 1D (time,) vars
        # (time_chunk_1d instead of time_chunk) -- verify it isn't
        # accidentally 3D-only, using an odd chunk_size like the 3D case
        # above.
        source, output_store = _build_template_and_source(
            tmp_path, 'v1d', total_layers=9, chunk_size=5, is_3d=False
        )

        dcpv._write_var_in_chunks(source, output_store, 'v1d', chunk_size=5, total_layers=9)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['v1d'].values, source['v1d'].values)


# ---------------------------------------------------------------------------
# deep_copy_cube_per_var(): local end-to-end, monkeypatched open_virtual_cube.
# ---------------------------------------------------------------------------

NUM_LAYERS = 5
CUBE_Y_SIZE = 16
CUBE_X_SIZE = 24
FILL_VALUE = np.int16(-1)


def _make_synthetic_virtual_cube():
    """A small xr.Dataset shaped like a real virtual cube: two 3D
    (time,y,x) int16 vars, two 1D (time,) vars (one string, one numeric),
    one static 2D (y,x) var, and one scalar var -- enough to exercise
    deep_copy_cube_per_var()'s full per-variable loop (multiple 3D and
    multiple 1D vars processed one after another) without cross-variable
    leakage (e.g. a stale .encoding/.attrs left over from
    _reset_write_encoding, or a gc.collect() dropping a reference too
    early). Dask-chunked at (1, CUBE_Y_SIZE, CUBE_X_SIZE) -- one dask chunk
    per layer, full spatial extent -- matching the real virtual cube's
    fixed per-granule chunk size, same reasoning as
    test_deep_copy_cube_tiled.py's fixture.
    """
    values = np.arange(NUM_LAYERS * CUBE_Y_SIZE * CUBE_X_SIZE, dtype='int64').reshape(
        NUM_LAYERS, CUBE_Y_SIZE, CUBE_X_SIZE
    ) % 500

    ds = xr.Dataset(
        data_vars={
            'vx': (
                ('time', 'y', 'x'), values.astype('int16'),
                {'missing_value': FILL_VALUE}
            ),
            'vy': (
                ('time', 'y', 'x'), (values * 2 % 500).astype('int16'),
                {'missing_value': FILL_VALUE}
            ),
            'url': (('time',), np.array([f'granule_{i}.nc' for i in range(NUM_LAYERS)])),
            'date_dt': (('time',), np.arange(NUM_LAYERS, dtype='float32')),
            'landice': (('y', 'x'), np.ones((CUBE_Y_SIZE, CUBE_X_SIZE), dtype='uint8')),
            'mapping': ((), ''),
        },
        coords={
            utils.Coords.TIME: np.arange(NUM_LAYERS).astype('datetime64[ns]'),
            utils.Coords.Y: np.arange(CUBE_Y_SIZE, dtype='float64'),
            utils.Coords.X: np.arange(CUBE_X_SIZE, dtype='float64'),
        },
    )
    ds = ds.chunk({utils.Coords.TIME: 1, utils.Coords.Y: CUBE_Y_SIZE, utils.Coords.X: CUBE_X_SIZE})
    ds.attrs['date_created'] = '01-Jan-2020 00:00:00'
    ds.attrs['title'] = 'synthetic test cube'
    ds.attrs['institution'] = 'test'
    ds.attrs['projection'] = '3413'

    return ds


@pytest.fixture
def patched_open_virtual_cube(monkeypatch):
    """Monkeypatch deep_copy_cube_per_var's open_virtual_cube so
    deep_copy_cube_per_var() runs against the synthetic cube instead of a
    real icechunk repo/S3 bucket. Returns the synthetic source cube so
    tests can compare against it."""
    source = _make_synthetic_virtual_cube()
    monkeypatch.setattr(dcpv, 'open_virtual_cube', lambda *args, **kwargs: source)
    return source


class TestDeepCopyCubePerVarLocal:

    def test_static_vars_written_once(self, patched_open_virtual_cube, tmp_path):
        output_store = str(tmp_path / "output.zarr")
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=2, xy_chunk=4, time_chunk_1d=100,
        )

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=True)
        assert result['landice'].shape == (CUBE_Y_SIZE, CUBE_X_SIZE)
        assert 'time' not in result['landice'].dims
        np.testing.assert_array_equal(
            result['landice'].values, patched_open_virtual_cube['landice'].values
        )

    def test_multiple_3d_vars_match_source_values(self, patched_open_virtual_cube, tmp_path):
        # time_chunk=2 with NUM_LAYERS=5 forces 3 time-chunks (2,2,1 --
        # ragged last chunk) per variable; vx/vy being processed one after
        # another exercises that neither variable's write leaks into or
        # corrupts the other's.
        output_store = str(tmp_path / "output.zarr")
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=2, xy_chunk=4, time_chunk_1d=100,
        )

        result = xr.open_zarr(
            output_store, zarr_format=3, consolidated=True, mask_and_scale=False
        )
        source = patched_open_virtual_cube

        for var_name in ('vx', 'vy'):
            assert result[var_name].shape == (NUM_LAYERS, CUBE_Y_SIZE, CUBE_X_SIZE)
            np.testing.assert_array_equal(result[var_name].values, source[var_name].values)

    def test_multiple_1d_vars_match_source_values(self, patched_open_virtual_cube, tmp_path):
        output_store = str(tmp_path / "output.zarr")
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=2, xy_chunk=4, time_chunk_1d=100,
        )

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=True)
        source = patched_open_virtual_cube

        for var_name in ('url', 'date_dt'):
            assert result[var_name].shape == (NUM_LAYERS,)
            np.testing.assert_array_equal(result[var_name].values, source[var_name].values)

    def test_chunking_matches_build_encoding(self, patched_open_virtual_cube, tmp_path):
        output_store = str(tmp_path / "output.zarr")
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=2, xy_chunk=4, time_chunk_1d=100,
            xy_shard_multiplier=2,
        )

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=True)
        assert result['vx'].encoding['chunks'] == (2, 4, 4)
        assert result['vx'].encoding['shards'] == (2, 8, 8)
        assert result['url'].encoding['chunks'] == (100,)

    def test_num_layers_truncates_output(self, patched_open_virtual_cube, tmp_path):
        output_store = str(tmp_path / "output.zarr")
        requested_layers = NUM_LAYERS - 2
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=2, xy_chunk=4, time_chunk_1d=100,
            num_layers=requested_layers,
        )

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=True)
        assert result.sizes['time'] == requested_layers
        np.testing.assert_array_equal(
            result['vx'].values,
            patched_open_virtual_cube['vx'].isel(time=slice(0, requested_layers)).values
        )

    def test_odd_time_chunk_still_matches_source(self, patched_open_virtual_cube, tmp_path):
        # time_chunk=3 (odd) with NUM_LAYERS=5 -> write_span=1 for the 3D
        # loop, desyncing from the real 3-layer chunk boundary (chunk 0 is
        # [0,3), chunk 1 is [3,5)) -- covers the same real-world caveat as
        # TestWriteVarInChunksBoundaries.test_odd_chunk_size_..., but
        # through the full public deep_copy_cube_per_var() entry point,
        # for both a 3D and a 1D variable.
        output_store = str(tmp_path / "output.zarr")
        dcpv.deep_copy_cube_per_var(
            input_store="unused", output_store=output_store, bucket_prefix="s3://unused/",
            time_chunk=3, xy_chunk=4, time_chunk_1d=3,
        )

        result = xr.open_zarr(
            output_store, zarr_format=3, consolidated=True, mask_and_scale=False
        )
        source = patched_open_virtual_cube
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)
        np.testing.assert_array_equal(result['url'].values, source['url'].values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
