"""
Unit and local (no-network) integration tests for deep_copy_cube_per_var.py
using pytest.

This module never touches S3 or icechunk: deep_copy_cube_per_var()'s
end-to-end tests monkeypatch open_virtual_cube() to return a synthetic,
dask-backed xr.Dataset shaped like a real virtual cube -- per
feedback_no_s3_writing_tests, no network dependency needed to exercise the
per-variable, half-chunk-write mechanics for real, against a real local
zarr v3 store.

Covers:
- split_time_vars_by_rank(): 3D (time,y,x) vs. 1D (time,) split (moved here
  from the now-removed deep_copy_cube_tiled.py, which needed the same split
  for a different reason -- see the function's own docstring).
- _write_var_3d(): the half-chunk raw-zarr-array write boundary math in
  isolation, against a directly-templated local store -- both an int var
  and a float var (INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS are both 2 as of
  2026-09-16, see their comment for why a whole-chunk write regressed ints
  2.4x in production despite confirmed-clean RAM; kept as separate test
  cases so a future divergence between the two constants stays covered)
  with even chunk size (divides the chunk evenly) and odd chunk size
  (write_span desyncs from real chunk boundaries, but values must still
  come out exactly correct), num_layers smaller than write_span, chunk_size
  larger than total_layers, NaN -> fill_value substitution landing
  correctly on disk, the whole-chunk radar-skip leaving a chunk genuinely
  absent, and the dims-order guard raising rather than silently
  transposing.
- _write_var_1d(): whole-chunk boundary math for 1D (time,) vars, which keep
  the xarray to_zarr(region=...) path (unlike _write_var_3d(), these are not
  split into half-chunk writes -- 1D vars are baked directly into the
  virtual cube rather than referenced through per-granule manifest arrays,
  so RAM/write-amplification never motivated splitting them; see
  _write_var_1d()'s own docstring).
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
import zarr

warnings.filterwarnings('ignore', message='resource_tracker: There appear to be.*leaked semaphore')
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import deep_copy_cube_per_var as dcpv
from itscube_types import ImgPairInfo


# ---------------------------------------------------------------------------
# split_time_vars_by_rank(): pure, synthetic, no network. Ported from
# test_deep_copy_cube_tiled.py when deep_copy_cube_tiled.py was removed and
# the function moved to deep_copy_cube_per_var.py.
# ---------------------------------------------------------------------------

@pytest.fixture
def rank_split_cube():
    """Small dataset with one 3D (time,y,x) var and one 1D (time,) var --
    enough to exercise split_time_vars_by_rank()."""
    return xr.Dataset(
        data_vars={
            'vx': (('time', 'y', 'x'), np.zeros((3, 4, 5), dtype='int16')),
            'url': (('time',), np.array(['a', 'b', 'c'])),
        },
        coords={
            utils.Coords.TIME: np.arange(3).astype('datetime64[ns]'),
            utils.Coords.Y: np.arange(4, dtype='float64'),
            utils.Coords.X: np.arange(5, dtype='float64'),
        },
    )


class TestSplitTimeVarsByRank:

    def test_splits_3d_and_1d_vars(self, rank_split_cube):
        vars_3d, vars_1d = dcpv.split_time_vars_by_rank(
            rank_split_cube, ['vx', 'url']
        )

        assert set(vars_3d) == {'vx'}
        assert set(vars_1d) == {'url'}


# ---------------------------------------------------------------------------
# _write_var_3d() / _write_var_1d(): pure boundary-math tests against a
# directly templated local store -- no full deep_copy_cube_per_var() run
# needed.
# ---------------------------------------------------------------------------

Y_SIZE = 8
X_SIZE = 6


def _build_template_and_source(
    tmp_path, var_name, total_layers, chunk_size, is_3d,
    dtype='int16', fill_value=None
):
    """Build (a) an in-memory, dask-backed source dataset holding the
    "real" per-layer values _write_var_3d()/_write_var_1d() read from, and
    (b) an already-templated local zarr v3 store (mode='w', compute=False)
    declaring var_name's shape/dtype/chunks up front but with no pixel data
    written yet -- mirroring deep_copy_cube_per_var()'s own template step,
    so _write_var_3d()'s raw zarr-array writes (and _write_var_1d()'s
    mode='r+' region writes) have a real store to land into.

    Dask-chunked at one chunk per layer (full spatial extent for 3D vars):
    to_zarr(compute=False) only defers writing for dask-backed variables
    (a hard requirement, not a graceful degrade -- see
    _make_synthetic_virtual_cube()'s fixture below for the same reasoning),
    and this deliberately mismatches `chunk_size` the same way a real
    virtual cube's fixed per-granule chunking does, exercising
    safe_chunks=False.

    Parameters
    ----------
    dtype : str
        'int16' or 'float32' -- the two _write_var_3d() dispatches on via
        INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS (both 2 as of 2026-09-16, see
        their comment in deep_copy_cube_per_var.py). Irrelevant to
        _write_var_1d(), which writes one whole chunk per write regardless
        of dtype.
    fill_value : scalar, optional
        If set, declared as the array's CF fill (missing_value for int,
        _FillValue for float) in the template's own encoding, so a
        mask_and_scale=True re-open decodes it the same way build_encoding()
        would in production.
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

    values = (np.arange(int(np.prod(shape))).reshape(shape) % 500).astype(dtype)
    source = xr.Dataset({var_name: (dims, values)}, coords=coords)
    source = source.chunk(dask_chunks)

    var_encoding = {'chunks': chunks}
    if fill_value is not None:
        np_dtype = np.dtype(dtype)
        if np_dtype.kind == 'f':
            var_encoding['_FillValue'] = np_dtype.type(fill_value)
        else:
            var_encoding['missing_value'] = np_dtype.type(fill_value)

    output_store = str(tmp_path / "template.zarr")
    source.to_zarr(
        output_store, mode='w', compute=False,
        encoding={var_name: var_encoding},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    return source, output_store


class TestWriteVar3dBoundaries:

    def test_int_var_writes_half_chunk_at_a_time(self, tmp_path):
        # int16 -> INT_CHUNK_SPLITS=2 -> write_span=3, dividing evenly into
        # each 6-layer chunk (exactly 2 writes per chunk). Separate test
        # case from the float one below so a future divergence between
        # INT_CHUNK_SPLITS and FLOAT_CHUNK_SPLITS stays covered.
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=12, chunk_size=6, is_3d=True, dtype='int16'
        )

        dcpv._write_var_3d(source, output_store, 'vx', chunk_size=6, total_layers=12)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_float_var_writes_half_chunk_at_a_time(self, tmp_path):
        # float32 -> FLOAT_CHUNK_SPLITS=2 -> write_span=3, dividing evenly
        # into each 6-layer chunk (exactly 2 writes per chunk).
        source, output_store = _build_template_and_source(
            tmp_path, 'M11', total_layers=12, chunk_size=6, is_3d=True, dtype='float32'
        )

        dcpv._write_var_3d(source, output_store, 'M11', chunk_size=6, total_layers=12)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['M11'].values, source['M11'].values)

    def test_odd_chunk_size_still_produces_correct_values(self, tmp_path):
        # float32, chunk_size=5 -> write_span=2, which does NOT divide
        # evenly into the 5-layer chunk -- writes desync from real chunk
        # boundaries (e.g. the write [4,6) straddles the chunk-0/chunk-1
        # boundary at layer 5). The written VALUES must still be exactly
        # correct regardless.
        source, output_store = _build_template_and_source(
            tmp_path, 'M11', total_layers=10, chunk_size=5, is_3d=True, dtype='float32'
        )

        dcpv._write_var_3d(source, output_store, 'M11', chunk_size=5, total_layers=10)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['M11'].values, source['M11'].values)

    def test_num_layers_smaller_than_write_span(self, tmp_path):
        # write_span = 10 // 2 = 5, but total_layers=3 -- a single write
        # must cover the whole (short) range, not loop past it.
        source, output_store = _build_template_and_source(
            tmp_path, 'M11', total_layers=3, chunk_size=10, is_3d=True, dtype='float32'
        )

        dcpv._write_var_3d(source, output_store, 'M11', chunk_size=10, total_layers=3)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['M11'].values, source['M11'].values)

    def test_chunk_size_larger_than_total_layers(self, tmp_path):
        # write_span = 100 // 2 = 50, but only 7 real layers exist: the
        # single write [0,7) must cover everything in one shot, not loop
        # past total_layers.
        source, output_store = _build_template_and_source(
            tmp_path, 'vx', total_layers=7, chunk_size=100, is_3d=True, dtype='int16'
        )

        dcpv._write_var_3d(source, output_store, 'vx', chunk_size=100, total_layers=7)

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)
        np.testing.assert_array_equal(result['vx'].values, source['vx'].values)

    def test_nan_replaced_with_fill_value_on_disk(self, tmp_path):
        # A float batch with real NaNs and a declared fill: the raw on-disk
        # bytes must hold the sentinel, not NaN (see _fill_nan_in_place),
        # while a mask_and_scale=True re-open still decodes back to NaN.
        total_layers, chunk_size = 4, 4
        fill = np.float32(-32767.0)
        values = np.arange(total_layers * Y_SIZE * X_SIZE, dtype='float32').reshape(
            total_layers, Y_SIZE, X_SIZE
        )
        values[1, 0, 0] = np.nan
        source = xr.Dataset(
            {'M11': ((utils.Coords.TIME, utils.Coords.Y, utils.Coords.X), values)},
            coords={
                utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]'),
                utils.Coords.Y: np.arange(Y_SIZE, dtype='float64'),
                utils.Coords.X: np.arange(X_SIZE, dtype='float64'),
            },
        ).chunk({utils.Coords.TIME: 1, utils.Coords.Y: Y_SIZE, utils.Coords.X: X_SIZE})

        output_store = str(tmp_path / "template.zarr")
        source.to_zarr(
            output_store, mode='w', compute=False,
            encoding={'M11': {'chunks': (chunk_size, Y_SIZE, X_SIZE), '_FillValue': fill}},
            zarr_format=3, consolidated=False, safe_chunks=False,
        )

        dcpv._write_var_3d(
            source, output_store, 'M11', chunk_size=chunk_size, total_layers=total_layers,
            fill_value=fill
        )

        raw = zarr.open_group(output_store, mode='r')['M11'][:]
        assert raw[1, 0, 0] == fill, 'NaN must be replaced with the fill sentinel on disk'

        decoded = xr.open_zarr(output_store, zarr_format=3, consolidated=False)['M11'].values
        assert np.isnan(decoded[1, 0, 0]), 'fill sentinel must decode back to NaN'
        finite_mask = np.ones_like(values, dtype=bool)
        finite_mask[1, 0, 0] = False
        np.testing.assert_array_equal(decoded[finite_mask], values[finite_mask])

    def test_radar_skip_leaves_chunk_absent(self, tmp_path):
        # Two chunks: the first has a radar layer (written normally), the
        # second is all-optical (is_radar all False) and must be skipped
        # entirely -- never written, reads back via the declared fill_value.
        total_layers, chunk_size = 4, 2
        fill = np.float32(-32767.0)
        source, output_store = _build_template_and_source(
            tmp_path, 'M11', total_layers=total_layers, chunk_size=chunk_size,
            is_3d=True, dtype='float32', fill_value=fill
        )
        is_radar = np.array([True, True, False, False])

        dcpv._write_var_3d(
            source, output_store, 'M11', chunk_size=chunk_size, total_layers=total_layers,
            fill_value=fill, is_radar=is_radar
        )

        result = xr.open_zarr(output_store, zarr_format=3, consolidated=False)['M11'].values
        np.testing.assert_array_equal(result[:chunk_size], source['M11'].values[:chunk_size])
        assert np.all(np.isnan(result[chunk_size:])), 'skipped chunk must read back as fill/NaN'

    def test_wrong_dim_order_raises_value_error(self, tmp_path):
        # (y, x, time) instead of (time, y, x): must raise rather than
        # silently transpose (which would allocate a full extra copy --
        # the exact cost this function exists to avoid).
        total_layers = 4
        values = np.zeros((Y_SIZE, X_SIZE, total_layers), dtype='int16')
        source = xr.Dataset(
            {'vx': ((utils.Coords.Y, utils.Coords.X, utils.Coords.TIME), values)},
            coords={
                utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]'),
                utils.Coords.Y: np.arange(Y_SIZE, dtype='float64'),
                utils.Coords.X: np.arange(X_SIZE, dtype='float64'),
            },
        )

        with pytest.raises(ValueError, match='expected'):
            dcpv._write_var_3d(source, str(tmp_path / "unused.zarr"), 'vx', chunk_size=2, total_layers=4)


class TestWriteVar1dBoundaries:

    def test_1d_var_whole_chunk_matches_source(self, tmp_path):
        # _write_var_1d() writes one whole real chunk per write (no
        # half-splitting) -- total_layers=9 with chunk_size=5 forces two
        # real chunks (a ragged [0,5) and [5,9)), verifying the boundary
        # math across a whole-chunk write rather than an artificial
        # half-chunk split like the 3D cases above.
        source, output_store = _build_template_and_source(
            tmp_path, 'v1d', total_layers=9, chunk_size=5, is_3d=False
        )

        dcpv._write_var_1d(source, output_store, 'v1d', chunk_size=5, total_layers=9)

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
    (time,y,x) int16 vars, four 1D (time,) vars (one string, one numeric,
    plus the mission/satellite pair _compute_radar_mask() needs), one
    static 2D (y,x) var, and one scalar var -- enough to exercise
    deep_copy_cube_per_var()'s full per-variable loop (multiple 3D and
    multiple 1D vars processed one after another) without cross-variable
    leakage (e.g. a stale .encoding/.attrs left over from
    _reset_write_encoding, or a gc.collect() dropping a reference too
    early). Dask-chunked at (1, CUBE_Y_SIZE, CUBE_X_SIZE) -- one dask chunk
    per layer, full spatial extent -- matching the real virtual cube's
    fixed per-granule chunk size, same reasoning as
    _build_template_and_source()'s fixture above.

    mission_img1/satellite_img1 are set to a fixed, valid Sentinel-2
    (optical) classification for every layer -- deep_copy_cube_per_var()
    calls _compute_radar_mask() unconditionally, and an unrecognized
    (mission, satellite) pair raises KeyError by design (see that
    function's docstring), so these must resolve to a real sensors.py
    group even though this cube has no RADAR_ONLY_VARS (M11/M12/vr/va) for
    the resulting mask to actually gate.
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
            ImgPairInfo.mission_img1: (('time',), np.array(['S'] * NUM_LAYERS)),
            ImgPairInfo.satellite_img1: (('time',), np.array(['2A'] * NUM_LAYERS)),
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
        # TestWriteVar3dBoundaries.test_odd_chunk_size_..., but
        # through the full public deep_copy_cube_per_var() entry point, for
        # a 3D variable. time_chunk_1d=3 with NUM_LAYERS=5 exercises the
        # equivalent whole-chunk boundary (chunk 0 is [0,3), chunk 1 is
        # [3,5)) for the 1D variable.
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
