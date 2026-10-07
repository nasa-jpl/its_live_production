"""
Unit tests for deep_copy_cube_per_var_chunk.py's resumability mechanism
(deep_copy_cube_progress._Progress and its markers under --progress-dir)
using pytest.

This module never touches S3, icechunk, or the `aws` CLI. The S3-writing code
paths are always monkeypatched away -- either by stubbing _upload_chunk()
itself (for tests focused purely on the skip/mark decision) or, for the one
full end-to-end test, by stubbing the lower-level _s3_copy() so
consolidate/delete still run for real against local-only paths. _s3_copy is
imported separately into both deep_copy_cube_per_var_chunk (dcpvc, used by
_upload_chunk() and the skeleton upload) and deep_copy_cube_progress (dcp,
used by _Progress.write_config()), so tests that exercise a full run
monkeypatch it in BOTH modules -- patching only one leaves the other's calls
hitting the real `aws` CLI. Everything resumability-related runs against a
plain in-memory fake filesystem: a set of "existing paths" plus
exists()/touch()/rm()/open(), with no S3 or network semantics at all, never a
real or mocked S3 client. Per feedback_no_s3_writing_tests, what's exercised
here is only OUR branching logic -- which variable/chunk gets skipped vs.
(re)done, which markers get read and written, and which config mismatches
must fail.

Covers:
- _upload_chunk(): a chunk zarr's write_empty_chunks skipped writing (all
    values equal the fill_value) is a no-op, not an _s3_copy() call on a path
    that doesn't exist locally; an actually-written chunk still uploads.
- _Progress marker path layout, including that --progress-dir is namespaced
    by the output store's name so one shared directory can serve many cubes.
- _Progress.validate_config(): every shape-defining parameter mismatch is a
    hard failure; a grown input cube clamps back to the recorded layer count;
    a shrunken cube fails; a shifted first/last 'time' (a granule inserted
    rather than appended) fails.
- _write_var_3d_and_upload()/_write_var_1d_and_upload(): a chunk whose
    marker exists is skipped without any upload; a chunk without one proceeds
    and is marked afterward; a variable whose own marker exists returns
    without checking any per-chunk marker; a radar-skipped chunk is still
    marked done though never uploaded; progress=None disables everything.
- _write_var_3d_and_upload()/_write_var_1d_and_upload()'s `start_layer`
    parameter (used by deep_copy_update_per_var_chunk.py): chunks entirely
    before it are never iterated at all, the one boundary chunk straddling it
    only writes/uploads its `[start_layer, chunk_stop)` portion, and the
    radar-skip check on that boundary chunk is narrowed to the same range
    rather than considering the already-written prefix.
- deep_copy_cube_per_var_chunk(): a pre-existing _SUCCESS short-circuits the
    run before open_virtual_cube() is called; a full run writes run_config
    plus every marker and prunes the per-variable ones unless asked not to;
    and without --progress-dir no markers are touched at all.
- build_encoding()'s 'time' coordinate encoding: units/calendar/dtype are
    pinned explicitly to the GPS-epoch/float64 convention, not left for
    xarray's CF encoder to infer a different one per cube.
- time_collisions.uniquify_granule_times(): catches not just exact duplicate
    'time' values but any that alias to the identical on-disk float64 value
    once CF-encoded (e.g. sub-ULP nanosecond-apart granules), and leaves
    already-unique granules untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import deep_copy_cube_per_var_chunk as dcpvc
import deep_copy_cube_progress as dcp
import time_collisions as tc
from itscube_types import ImgPairInfo, Vars


PROGRESS_DIR = 's3://bucket/progress'
OUTPUT_STORE = 's3://bucket/cubes/mycube.zarr'
# What _Progress.create() derives from the two above.
BASE = f'{PROGRESS_DIR}/mycube.zarr'


class _FakeFS:
    """In-memory stand-in for s3fs.S3FileSystem: a set of paths plus the
    handful of operations _Progress uses. No S3 or network semantics, so
    using it never exercises a real (or mocked) S3 write path -- only our own
    marker-path logic and branching decisions."""

    def __init__(self, existing=()):
        self.paths = set(existing)
        self.contents = {}
        self.exists_calls = []

    def exists(self, path):
        self.exists_calls.append(path)
        return path in self.paths or any(p.startswith(f'{path}/') for p in self.paths)

    def touch(self, path):
        self.paths.add(path)

    def rm(self, path, recursive=False):
        self.paths = {
            p for p in self.paths
            if not (p == path or (recursive and p.startswith(f'{path}/')))
        }

    def open(self, path, mode='r'):
        import io
        return io.StringIO(self.contents[path])

    def write_json(self, path, payload):
        """Test-side helper standing in for a prior attempt having uploaded
        run_config.json (which production writes via _s3_copy, not s3fs)."""
        self.paths.add(path)
        self.contents[path] = json.dumps(payload)


def _make_progress(existing=(), config=None):
    fake_fs = _FakeFS(existing)
    if config is not None:
        fake_fs.write_json(f'{BASE}/{dcp.RUN_CONFIG_NAME}', config)

    return dcp._Progress(fake_fs, BASE), fake_fs


RUN_PARAMS = {
    'time_chunk': 2,
    'xy_chunk': 4,
    'time_chunk_1d': 2,
    'xy_shard_multiplier': 1,
    'num_layers': 0,
}


# ---------------------------------------------------------------------------
# Marker layout.
# ---------------------------------------------------------------------------

class TestProgressPaths:

    def test_base_is_namespaced_by_output_store_name(self, monkeypatch):
        # One shared --progress-dir must not let two cubes' markers collide.
        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: _FakeFS())
        first = dcp._Progress.create(PROGRESS_DIR, 's3://bucket/cubes/cube_a.zarr')
        second = dcp._Progress.create(PROGRESS_DIR, 's3://other/place/cube_b.zarr')

        assert first.base == f'{PROGRESS_DIR}/cube_a.zarr'
        assert second.base == f'{PROGRESS_DIR}/cube_b.zarr'

    def test_create_sets_bucket_owner_acl(self, monkeypatch):
        # Every object this pipeline writes carries this ACL (see _s3_copy).
        captured = {}

        def _fake_fs(*args, **kwargs):
            captured.update(kwargs)
            return _FakeFS()

        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', _fake_fs)
        dcp._Progress.create(PROGRESS_DIR, OUTPUT_STORE)

        assert captured['s3_additional_kwargs'] == {'ACL': dcp.BUCKET_OWNER_ACL}

    def test_marker_paths(self):
        progress, _ = _make_progress()

        assert progress._success_path() == f'{BASE}/_SUCCESS'
        assert progress._var_success_path('vx') == f'{BASE}/vx/_SUCCESS'
        assert progress._chunk_path('vx', 3) == f'{BASE}/vx/3.done'
        assert progress._config_path() == f'{BASE}/run_config.json'
        assert progress._skeleton_success_path() == f'{BASE}/skeleton/_SUCCESS'

    def test_progress_dir_trailing_slash_is_normalized(self, monkeypatch):
        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: _FakeFS())
        progress = dcp._Progress.create(f'{PROGRESS_DIR}/', OUTPUT_STORE)

        assert progress.base == BASE


# ---------------------------------------------------------------------------
# validate_config(): the guard against resuming with markers that no longer
# mean what they meant.
# ---------------------------------------------------------------------------

TIME_VALUES = np.arange(10).astype('datetime64[ns]')


def _recorded_config(total_layers=6, **overrides):
    config = dcp._Progress.build_config(
        OUTPUT_STORE, RUN_PARAMS, total_layers, TIME_VALUES
    )
    config.update(overrides)
    return config


class TestValidateConfig:

    def test_identical_config_returns_recorded_layers(self):
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        current = dcp._Progress.build_config(OUTPUT_STORE, RUN_PARAMS, 6)

        assert progress.validate_config(recorded, current, 6, TIME_VALUES) == 6

    @pytest.mark.parametrize('key,changed', [
        ('time_chunk', 5),
        ('xy_chunk', 8),
        ('time_chunk_1d', 7),
        ('xy_shard_multiplier', 4),
        ('num_layers', 3),
    ])
    def test_changed_shape_parameter_raises(self, key, changed):
        # Markers are keyed on (var, chunk_index), meaningless if any of
        # these changed -- must fail rather than silently corrupt.
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        params = dict(RUN_PARAMS, **{key: changed})
        current = dcp._Progress.build_config(OUTPUT_STORE, params, 6)

        with pytest.raises(RuntimeError, match='disagree with those recorded'):
            progress.validate_config(recorded, current, 6, TIME_VALUES)

    def test_different_output_store_raises(self):
        # Guards against two cubes sharing one progress directory.
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        current = dcp._Progress.build_config('s3://bucket/cubes/other.zarr', RUN_PARAMS, 6)

        with pytest.raises(RuntimeError, match='disagree with those recorded'):
            progress.validate_config(recorded, current, 6, TIME_VALUES)

    def test_grown_input_cube_clamps_to_recorded_layers(self):
        # The output store's shape was frozen by the first attempt, so newly
        # appended layers are ignored rather than treated as an error.
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        current = dcp._Progress.build_config(OUTPUT_STORE, RUN_PARAMS, 9)

        assert progress.validate_config(recorded, current, 9, TIME_VALUES) == 6

    def test_shrunken_input_cube_raises(self):
        # The declared shape could never be filled.
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        current = dcp._Progress.build_config(OUTPUT_STORE, RUN_PARAMS, 4)

        with pytest.raises(RuntimeError, match='fewer than'):
            progress.validate_config(recorded, current, 4, TIME_VALUES)

    def test_shifted_layer_times_raise(self):
        # A granule inserted (not appended) shifts every later layer, so
        # already-written chunks no longer hold what their markers claim.
        progress, _ = _make_progress()
        recorded = _recorded_config(total_layers=6)
        current = dcp._Progress.build_config(OUTPUT_STORE, RUN_PARAMS, 6)
        shifted = np.arange(100, 110).astype('datetime64[ns]')

        with pytest.raises(RuntimeError, match='changed'):
            progress.validate_config(recorded, current, 6, shifted)


# ---------------------------------------------------------------------------
# _write_var_3d_and_upload() / _write_var_1d_and_upload(): resumability
# control flow, against a real local store but a stubbed _upload_chunk().
# ---------------------------------------------------------------------------

Y_SIZE = 4
X_SIZE = 3


def _build_3d_template_and_source(tmp_path, var_name, total_layers, chunk_size, dtype='int16'):
    dims = (utils.Coords.TIME, utils.Coords.Y, utils.Coords.X)
    shape = (total_layers, Y_SIZE, X_SIZE)
    values = (np.arange(int(np.prod(shape))).reshape(shape) % 500).astype(dtype)
    source = xr.Dataset(
        {var_name: (dims, values)},
        coords={
            utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]'),
            utils.Coords.Y: np.arange(Y_SIZE, dtype='float64'),
            utils.Coords.X: np.arange(X_SIZE, dtype='float64'),
        },
    ).chunk({utils.Coords.TIME: 1, utils.Coords.Y: Y_SIZE, utils.Coords.X: X_SIZE})

    local_store = str(tmp_path / 'local.zarr')
    source.to_zarr(
        local_store, mode='w', compute=False,
        encoding={var_name: {'chunks': (chunk_size, Y_SIZE, X_SIZE)}},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    return source, local_store


def _build_1d_template_and_source(tmp_path, var_name, total_layers, chunk_size):
    source = xr.Dataset(
        {var_name: ((utils.Coords.TIME,), np.arange(total_layers))},
        coords={utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]')},
    ).chunk({utils.Coords.TIME: 1})

    local_store = str(tmp_path / 'local.zarr')
    source.to_zarr(
        local_store, mode='w', compute=False,
        encoding={var_name: {'chunks': (chunk_size,)}},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    return source, local_store


# ---------------------------------------------------------------------------
# _upload_chunk(): a chunk zarr's write_empty_chunks skipped writing (all
# values equal the array's fill_value, e.g. an all-optical time range for a
# radar-only derived 1D variable like M11_dr_to_vr_factor) must be a no-op,
# not an `aws s3 cp` on a path that was never created.
# ---------------------------------------------------------------------------

class TestUploadChunk:

    def test_missing_chunk_is_a_noop(self, tmp_path, monkeypatch):
        # compute=False writes only metadata, never any chunk data --
        # standing in for a chunk zarr's write_empty_chunks skipped writing
        # because every value in it equals the array's fill_value.
        _, local_store = _build_1d_template_and_source(
            tmp_path, 'M11_dr_to_vr_factor', total_layers=4, chunk_size=2
        )
        monkeypatch.setattr(
            dcpvc, '_s3_copy',
            lambda *a, **k: pytest.fail('must not attempt to sync a chunk never written locally')
        )

        dcpvc._upload_chunk(local_store, OUTPUT_STORE, 'M11_dr_to_vr_factor', 0)

    def test_existing_chunk_still_uploads(self, tmp_path, monkeypatch):
        source, local_store = _build_1d_template_and_source(
            tmp_path, 'M11_dr_to_vr_factor', total_layers=4, chunk_size=2
        )
        source.isel({utils.Coords.TIME: slice(0, 2)}).to_zarr(
            local_store, mode='r+', region={utils.Coords.TIME: slice(0, 2)},
            zarr_format=3, consolidated=False, safe_chunks=False,
        )
        copied_to = []
        monkeypatch.setattr(dcpvc, '_s3_copy', lambda local_path, s3_path, **k: copied_to.append(s3_path))

        dcpvc._upload_chunk(local_store, OUTPUT_STORE, 'M11_dr_to_vr_factor', 0)

        assert f'{OUTPUT_STORE}/M11_dr_to_vr_factor/c/0' in copied_to


class TestWriteVar3dAndUploadResumability:

    def test_no_progress_disables_resumability(self, tmp_path, monkeypatch):
        # progress=None (the default): every chunk is (re)done every time,
        # no marker ever read or written -- pre-resumability behavior.
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=4, chunk_size=2)

        dcpvc._write_var_3d_and_upload(source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=4)

        assert uploaded == [0, 1]

    def test_chunk_with_existing_marker_is_skipped(self, tmp_path, monkeypatch):
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=4, chunk_size=2)
        progress, _ = _make_progress(existing={f'{BASE}/vx/0.done'})
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=4, progress=progress
        )

        assert uploaded == [1]

    def test_chunk_marked_done_after_upload(self, tmp_path, monkeypatch):
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=4, chunk_size=2)
        progress, fake_fs = _make_progress()
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: None)

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=4, progress=progress
        )

        assert fake_fs.paths == {
            f'{BASE}/vx/0.done',
            f'{BASE}/vx/1.done',
            f'{BASE}/vx/_SUCCESS',
        }

    def test_variable_already_done_skips_every_chunk(self, tmp_path, monkeypatch):
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=4, chunk_size=2)
        progress, fake_fs = _make_progress(existing={f'{BASE}/vx/_SUCCESS'})
        monkeypatch.setattr(
            dcpvc, '_upload_chunk',
            lambda *a: pytest.fail('must not upload when the variable is already done')
        )

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=4, progress=progress
        )

        # Only the variable-level marker was ever checked -- no per-chunk
        # lookups needed, which is the point of the per-variable marker.
        assert fake_fs.exists_calls == [f'{BASE}/vx/_SUCCESS']

    def test_radar_skipped_chunk_still_gets_marked(self, tmp_path, monkeypatch):
        # Chunk 1 (layers 2:4) is all-optical -- never uploaded, but still a
        # resolved chunk, so it still gets a marker.
        source, local_store = _build_3d_template_and_source(
            tmp_path, Vars.m11, total_layers=4, chunk_size=2, dtype='float32'
        )
        progress, fake_fs = _make_progress()
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, Vars.m11, chunk_size=2, total_layers=4,
            is_radar=np.array([True, True, False, False]), progress=progress
        )

        assert uploaded == [0]
        assert f'{BASE}/{Vars.m11}/1.done' in fake_fs.paths
        assert f'{BASE}/{Vars.m11}/_SUCCESS' in fake_fs.paths


class TestWriteVar1dAndUploadResumability:

    def test_no_progress_disables_resumability(self, tmp_path, monkeypatch):
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)

        dcpvc._write_var_1d_and_upload(source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5)

        assert uploaded == [0, 1, 2]

    def test_chunk_with_existing_marker_is_skipped(self, tmp_path, monkeypatch):
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)
        progress, _ = _make_progress(existing={f'{BASE}/url/1.done'})
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_1d_and_upload(
            source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5, progress=progress
        )

        assert uploaded == [0, 2]

    def test_variable_already_done_skips_every_chunk(self, tmp_path, monkeypatch):
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)
        progress, fake_fs = _make_progress(existing={f'{BASE}/url/_SUCCESS'})
        monkeypatch.setattr(
            dcpvc, '_upload_chunk',
            lambda *a: pytest.fail('must not upload when the variable is already done')
        )

        dcpvc._write_var_1d_and_upload(
            source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5, progress=progress
        )

        assert fake_fs.exists_calls == [f'{BASE}/url/_SUCCESS']

    def test_chunks_marked_done_after_upload(self, tmp_path, monkeypatch):
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)
        progress, fake_fs = _make_progress()
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: None)

        dcpvc._write_var_1d_and_upload(
            source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5, progress=progress
        )

        assert fake_fs.paths == {
            f'{BASE}/url/0.done',
            f'{BASE}/url/1.done',
            f'{BASE}/url/2.done',
            f'{BASE}/url/_SUCCESS',
        }


# ---------------------------------------------------------------------------
# start_layer: the parameter deep_copy_update_per_var_chunk.py relies on to
# reuse these two functions for writing only the newly appended layers.
# ---------------------------------------------------------------------------

class TestWriteVar3dAndUploadStartLayer:

    def test_start_layer_skips_chunks_before_it_and_narrows_boundary_chunk(self, tmp_path, monkeypatch):
        # total_layers=6, chunk_size=2 -> chunks [0:2), [2:4), [4:6).
        # start_layer=3 falls inside chunk 1 (chunk_start=2): chunk 0 must
        # never even be iterated, and chunk 1 must only write/upload its
        # [3:4) portion, not its already-written [2:3) prefix.
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=6, chunk_size=2)
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=6, start_layer=3
        )

        assert uploaded == [1, 2]

        target = zarr.open_group(local_store, mode='r', zarr_format=3)['vx']
        # Layers before start_layer were never touched by this call -- still
        # read back as the template's default fill, not the source data.
        np.testing.assert_array_equal(
            target[:3], np.zeros((3, Y_SIZE, X_SIZE), dtype=source['vx'].dtype)
        )
        np.testing.assert_array_equal(target[3:6], source['vx'].values[3:6])

    def test_default_start_layer_matches_no_start_layer_argument(self, tmp_path, monkeypatch):
        # start_layer=0 (the default) must reproduce the exact same behavior
        # as omitting it entirely -- no behavior change for existing callers.
        source, local_store = _build_3d_template_and_source(tmp_path, 'vx', total_layers=4, chunk_size=2)
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=2, total_layers=4, start_layer=0
        )

        assert uploaded == [0, 1]

    def test_start_layer_narrows_radar_skip_check_to_new_portion(self, tmp_path, monkeypatch):
        # Chunk 1 (layers 2:4) has a radar layer at index 2 (the
        # already-written old portion) but not at index 3 (the new portion,
        # start_layer=3) -- the narrowed check must consider only the new
        # portion and skip, where checking the whole chunk would not have.
        source, local_store = _build_3d_template_and_source(
            tmp_path, Vars.m11, total_layers=4, chunk_size=2, dtype='float32'
        )
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, Vars.m11, chunk_size=2, total_layers=4,
            is_radar=np.array([False, False, True, False]), start_layer=3
        )

        assert uploaded == []

    def test_appends_land_despite_stale_consolidated_metadata(self, tmp_path, monkeypatch):
        # Regression: reached via deep_copy_update_per_var_chunk.py, which
        # resizes this array in a store whose root zarr.json still carries
        # creation's CONSOLIDATED block -- so the root advertises the
        # pre-resize length. zarr prefers that block, making the raw
        # `target[start:stop] = values` append past it a SILENT no-op (no
        # exception, chunk still marked done, layers reading back as fill).
        # Hence use_consolidated=False on this function's open_group().
        # See utils/check_stale_consolidated_metadata.py.
        source, local_store = _build_3d_template_and_source(
            tmp_path, 'vx', total_layers=6, chunk_size=8
        )
        # Reproduce an update's metadata state: consolidate while the array
        # is still at its OLD length (3), then grow it to 6. Only the array's
        # own zarr.json learns about the growth; the root's block still says
        # 3. Safe to resize down first -- the template was written
        # compute=False, so no chunk data exists yet to lose.
        live = {'mode': 'r+', 'zarr_format': 3, 'use_consolidated': False}
        zarr.open_group(local_store, **live)['vx'].resize((3, Y_SIZE, X_SIZE))
        zarr.consolidate_metadata(local_store)
        zarr.open_group(local_store, **live)['vx'].resize((6, Y_SIZE, X_SIZE))

        # Guard the premise: without opting out, the store really does look
        # shorter than it is. If zarr ever stops preferring the stale block,
        # this fails loudly instead of leaving the test below vacuous.
        stale_shape = zarr.open_group(local_store, mode='r', zarr_format=3)['vx'].shape
        assert stale_shape == (3, Y_SIZE, X_SIZE), (
            f'expected the stale consolidated block to report the pre-resize '
            f'shape, got {stale_shape} -- premise of this regression no '
            f'longer holds'
        )

        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: None)

        dcpvc._write_var_3d_and_upload(
            source, local_store, OUTPUT_STORE, 'vx', chunk_size=8, total_layers=6
        )

        written = zarr.open_group(
            local_store, mode='r', zarr_format=3, use_consolidated=False
        )['vx']
        np.testing.assert_array_equal(written[:6], source['vx'].values[:6])


class TestWriteVar1dAndUploadStartLayer:

    def test_start_layer_skips_chunks_before_it_and_narrows_boundary_chunk(self, tmp_path, monkeypatch):
        # total_layers=5, chunk_size=2 -> chunks [0:2), [2:4), [4:5).
        # start_layer=3 falls inside chunk 1 (chunk_start=2): chunk 0 must
        # never even be iterated, and chunk 1 must only write/upload its
        # [3:4) portion.
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_1d_and_upload(
            source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5, start_layer=3
        )

        assert uploaded == [1, 2]

        result = xr.open_zarr(local_store, consolidated=False)['url'].values
        np.testing.assert_array_equal(result[:3], np.zeros(3, dtype=source['url'].dtype))
        np.testing.assert_array_equal(result[3:5], source['url'].values[3:5])

    def test_default_start_layer_matches_no_start_layer_argument(self, tmp_path, monkeypatch):
        source, local_store = _build_1d_template_and_source(tmp_path, 'url', total_layers=5, chunk_size=2)
        uploaded = []
        monkeypatch.setattr(dcpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        dcpvc._write_var_1d_and_upload(
            source, local_store, OUTPUT_STORE, 'url', chunk_size=2, total_layers=5, start_layer=0
        )

        assert uploaded == [0, 1, 2]


# ---------------------------------------------------------------------------
# deep_copy_cube_per_var_chunk(): short-circuit, full-run markers, pruning.
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
CUBE_Y_SIZE = 4
CUBE_X_SIZE = 3

# Realistic, day-spaced datetime values for _make_synthetic_virtual_cube()'s
# 'time' coordinate (and every _recorded_config-style call below that must
# agree with it bit-for-bit). np.arange(NUM_LAYERS).astype('datetime64[ns]')
# sits only nanoseconds from the 1970 Unix epoch -- fine under the old
# auto-inferred (cube-relative) time encoding, but once 'time' is pinned to
# float64 seconds since the real (1980) GPS epoch, those nanosecond-level
# gaps round away to the identical float64 value, making every layer look
# like a duplicate. Not a bug in the pinned epoch -- real mid_dates are never
# within nanoseconds of 1970 -- just unrealistic test data.
SYNTHETIC_TIME_VALUES = (
    np.datetime64('2020-01-01') + np.arange(NUM_LAYERS).astype('timedelta64[D]')
).astype('datetime64[ns]')


def _make_synthetic_virtual_cube():
    """Small xr.Dataset shaped like a real virtual cube -- one 3D
    (time,y,x) var, one 1D (time,) var, mission/satellite (needed since
    deep_copy_cube_per_var_chunk() calls _compute_radar_mask()
    unconditionally), and one static (y,x) var. Same construction as
    test_deep_copy_cube_per_var.py's equivalent, scaled down -- this module
    only needs it for marker bookkeeping, not to re-verify write-boundary
    math already covered there."""
    values = np.arange(NUM_LAYERS * CUBE_Y_SIZE * CUBE_X_SIZE, dtype='int64').reshape(
        NUM_LAYERS, CUBE_Y_SIZE, CUBE_X_SIZE
    ) % 500

    ds = xr.Dataset(
        data_vars={
            'vx': (
                ('time', 'y', 'x'), values.astype('int16'),
                {'missing_value': np.int16(-1)}
            ),
            Vars.url: (('time',), np.array([f'granule_{i}.nc' for i in range(NUM_LAYERS)])),
            ImgPairInfo.mission_img1: (('time',), np.array(['S'] * NUM_LAYERS)),
            ImgPairInfo.satellite_img1: (('time',), np.array(['2A'] * NUM_LAYERS)),
            'landice': (('y', 'x'), np.ones((CUBE_Y_SIZE, CUBE_X_SIZE), dtype='uint8')),
        },
        coords={
            utils.Coords.TIME: SYNTHETIC_TIME_VALUES,
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


def _install_local_only_run(monkeypatch, fake_fs):
    """Wire deep_copy_cube_per_var_chunk() up to run entirely locally: a
    synthetic cube instead of an icechunk repo, `fake_fs` for markers, and
    _s3_copy() stubbed out so no `aws` invocation ever happens
    (_upload_chunk() itself still runs for real -- consolidate plus the local
    delete -- against a destination it never reaches). The stub records every
    destination it was handed, since run_config.json is written through
    _s3_copy rather than s3fs (see _Progress.write_config).

    _s3_copy is imported into both dcpvc (used by _upload_chunk() and the
    skeleton upload) and dcp (used by _Progress.write_config()) -- each is a
    separate name binding, so both must be patched or one of them would
    still shell out to the real `aws` CLI."""
    copied_to = []
    monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: fake_fs)
    monkeypatch.setattr(dcpvc, 'open_virtual_cube', lambda *a, **k: _make_synthetic_virtual_cube())
    stub = lambda local_path, s3_path, **kwargs: copied_to.append(s3_path)
    monkeypatch.setattr(dcpvc, '_s3_copy', stub)
    monkeypatch.setattr(dcp, '_s3_copy', stub)
    return copied_to


@pytest.fixture
def local_only_run(monkeypatch):
    """A local-only run against an empty fake filesystem. Returns that
    filesystem so tests can inspect which markers were written."""
    fake_fs = _FakeFS()
    _install_local_only_run(monkeypatch, fake_fs)
    return fake_fs


def _run(tmp_path, **kwargs):
    dcpvc.deep_copy_cube_per_var_chunk(
        input_store='unused', output_store=OUTPUT_STORE, bucket_prefix='s3://unused/',
        time_chunk=2, xy_chunk=4, time_chunk_1d=2,
        local_staging_dir=str(tmp_path / 'local.zarr'),
        **kwargs
    )


class TestDeepCopyCubePerVarChunkStagingValidation:
    """--local-staging-dir input validation. No network: the raise happens
    before the input store is opened or any S3 call is made."""

    def test_s3_staging_dir_rejected_before_any_work(self, tmp_path, monkeypatch):
        # An s3:// --output-store with an s3:// --local-staging-dir passes
        # both pre-existing guards (staging is set, output IS s3), so
        # without this check every chunk would take _upload_chunk()'s
        # "not written locally" branch and the run would publish a
        # skeleton-only store marked _SUCCESS.
        monkeypatch.setattr(
            dcpvc, 'open_virtual_cube',
            lambda *a, **k: pytest.fail('must not open the input cube; validation comes first')
        )

        with pytest.raises(ValueError, match='must be a local filesystem path'):
            dcpvc.deep_copy_cube_per_var_chunk(
                input_store='unused', output_store=OUTPUT_STORE,
                bucket_prefix='s3://unused/', time_chunk=2, xy_chunk=4,
                time_chunk_1d=2,
                local_staging_dir='s3://bucket/scratch/cube.zarr',
            )


class TestDeepCopyCubePerVarChunkResumability:

    def test_existing_success_marker_short_circuits(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS(existing={f'{BASE}/_SUCCESS'})
        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: fake_fs)
        monkeypatch.setattr(
            dcpvc, 'open_virtual_cube',
            lambda *a, **k: pytest.fail('must not open the input cube when already done')
        )

        _run(tmp_path, progress_dir=PROGRESS_DIR)

    def test_full_run_marks_complete_and_prunes(self, tmp_path, local_only_run):
        _run(tmp_path, progress_dir=PROGRESS_DIR)

        # _SUCCESS and the skeleton marker survive the prune (so a retry of
        # an already-finished job still short-circuits, and so the skeleton
        # is never re-derived/re-uploaded even if something did retry);
        # per-variable/per-chunk markers do not.
        assert local_only_run.paths == {f'{BASE}/_SUCCESS', f'{BASE}/skeleton/_SUCCESS'}

    def test_keep_progress_markers_retains_everything(self, tmp_path, local_only_run):
        _run(tmp_path, progress_dir=PROGRESS_DIR, keep_progress_markers=True)

        assert f'{BASE}/_SUCCESS' in local_only_run.paths
        assert f'{BASE}/skeleton/_SUCCESS' in local_only_run.paths
        assert f'{BASE}/vx/_SUCCESS' in local_only_run.paths
        assert f'{BASE}/vx/0.done' in local_only_run.paths
        assert f'{BASE}/{Vars.url}/_SUCCESS' in local_only_run.paths

    def test_no_progress_dir_writes_no_markers(self, tmp_path, local_only_run, monkeypatch):
        # Without --progress-dir the strict refuse-to-overwrite guard is used
        # instead of the permissive resume check, so stub it out (this test
        # is about markers, not that guard).
        monkeypatch.setattr(dcpvc, 'resolve_output_store', lambda path: path)

        _run(tmp_path)

        assert local_only_run.paths == set()

    def test_fresh_run_records_run_config(self, tmp_path, monkeypatch):
        # The config guard is only as good as the config actually landing on
        # the first attempt -- it's written via _s3_copy, not s3fs.
        fake_fs = _FakeFS()
        copied_to = _install_local_only_run(monkeypatch, fake_fs)

        _run(tmp_path, progress_dir=PROGRESS_DIR)

        assert f'{BASE}/{dcp.RUN_CONFIG_NAME}' in copied_to

    def test_skeleton_uploaded_and_marked_done_on_fresh_run(self, tmp_path, monkeypatch):
        # The skeleton's whole-store copy is `_s3_copy(local_store,
        # output_store)` -- its destination is exactly OUTPUT_STORE, unlike
        # every other copy in this run (_upload_chunk() always appends a
        # `{var}/c/{chunk_idx}` or `zarr.json` subpath, and write_config()'s
        # destination is under BASE), so it's uniquely identifiable.
        fake_fs = _FakeFS()
        copied_to = _install_local_only_run(monkeypatch, fake_fs)

        _run(tmp_path, progress_dir=PROGRESS_DIR)

        assert OUTPUT_STORE in copied_to
        assert f'{BASE}/skeleton/_SUCCESS' in fake_fs.paths

    def test_resume_with_skeleton_done_skips_reupload(self, tmp_path, monkeypatch):
        # A prior attempt already fully uploaded the skeleton -- this
        # attempt must not re-derive/re-upload it (that would overwrite
        # time/x/y coordinate data every already-done chunk upload depends
        # on staying put), even though it still needs to rebuild local_store
        # from scratch and still proceeds with whatever chunk work is left.
        recorded = dcp._Progress.build_config(
            OUTPUT_STORE,
            {'time_chunk': 2, 'xy_chunk': 4, 'time_chunk_1d': 2,
             'xy_shard_multiplier': 1, 'num_layers': 0},
            NUM_LAYERS,
            SYNTHETIC_TIME_VALUES
        )
        fake_fs = _FakeFS(existing={f'{BASE}/skeleton/_SUCCESS'})
        fake_fs.write_json(f'{BASE}/{dcp.RUN_CONFIG_NAME}', recorded)
        copied_to = _install_local_only_run(monkeypatch, fake_fs)

        _run(tmp_path, progress_dir=PROGRESS_DIR, keep_progress_markers=True)

        assert OUTPUT_STORE not in copied_to
        # Chunk-level work for not-yet-done vars/chunks still proceeds
        # normally -- only the skeleton re-upload is skipped.
        assert f'{BASE}/vx/_SUCCESS' in fake_fs.paths
        assert f'{BASE}/{Vars.url}/_SUCCESS' in fake_fs.paths
        assert f'{BASE}/skeleton/_SUCCESS' in fake_fs.paths

    def test_duplicate_time_values_raise_before_any_upload(self, tmp_path, monkeypatch):
        # mid_date is microsecond-uniquified by construction -- any collision
        # means some position was never actually written (zarr's raw int
        # fill silently decodes back to a real-looking date). Must fail
        # loudly before the skeleton (or anything else) is ever published or
        # marked done.
        fake_fs = _FakeFS()
        cube = _make_synthetic_virtual_cube()
        duplicated_time = cube[utils.Coords.TIME].values.copy()
        duplicated_time[-1] = duplicated_time[0]
        cube = cube.assign_coords({utils.Coords.TIME: duplicated_time})

        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: fake_fs)
        monkeypatch.setattr(dcpvc, 'open_virtual_cube', lambda *a, **k: cube)
        copied_to = []
        stub = lambda local_path, s3_path, **kwargs: copied_to.append(s3_path)
        monkeypatch.setattr(dcpvc, '_s3_copy', stub)
        monkeypatch.setattr(dcp, '_s3_copy', stub)

        with pytest.raises(RuntimeError, match='duplicate value'):
            _run(tmp_path, progress_dir=PROGRESS_DIR)

        assert OUTPUT_STORE not in copied_to
        assert f'{BASE}/skeleton/_SUCCESS' not in fake_fs.paths

    def test_resume_skips_chunks_a_prior_attempt_finished(self, tmp_path, monkeypatch):
        # The core scenario: a prior attempt recorded its config and finished
        # some of vx's chunks; this attempt must leave those alone and pick up
        # only what's left.
        recorded = dcp._Progress.build_config(
            OUTPUT_STORE,
            {'time_chunk': 2, 'xy_chunk': 4, 'time_chunk_1d': 2,
             'xy_shard_multiplier': 1, 'num_layers': 0},
            NUM_LAYERS,
            SYNTHETIC_TIME_VALUES
        )
        fake_fs = _FakeFS(existing={f'{BASE}/vx/0.done', f'{BASE}/{Vars.url}/_SUCCESS'})
        fake_fs.write_json(f'{BASE}/{dcp.RUN_CONFIG_NAME}', recorded)
        _install_local_only_run(monkeypatch, fake_fs)

        uploaded = []
        monkeypatch.setattr(
            dcpvc, '_upload_chunk',
            lambda local, out, var, idx: uploaded.append((var, idx))
        )

        _run(tmp_path, progress_dir=PROGRESS_DIR, keep_progress_markers=True)

        # NUM_LAYERS=4 with time_chunk=2 gives vx chunks 0 and 1. vx chunk 0
        # was already marked, so only chunk 1 may be re-uploaded, and
        # Vars.url (marked fully done) must be untouched entirely. The
        # cube's other 1D variables were never marked, so they're expected
        # in `uploaded` -- assert on what must/must not appear rather than
        # the exact list.
        assert ('vx', 1) in uploaded
        assert ('vx', 0) not in uploaded
        assert not [entry for entry in uploaded if entry[0] == Vars.url]
        assert f'{BASE}/_SUCCESS' in fake_fs.paths

    def test_resume_with_mismatched_chunk_size_raises(self, tmp_path, monkeypatch):
        recorded = dcp._Progress.build_config(
            OUTPUT_STORE,
            {'time_chunk': 2, 'xy_chunk': 4, 'time_chunk_1d': 2,
             'xy_shard_multiplier': 1, 'num_layers': 0},
            NUM_LAYERS,
            SYNTHETIC_TIME_VALUES
        )
        fake_fs = _FakeFS()
        fake_fs.write_json(f'{BASE}/{dcp.RUN_CONFIG_NAME}', recorded)
        _install_local_only_run(monkeypatch, fake_fs)

        with pytest.raises(RuntimeError, match='disagree with those recorded'):
            dcpvc.deep_copy_cube_per_var_chunk(
                input_store='unused', output_store=OUTPUT_STORE,
                bucket_prefix='s3://unused/',
                # time_chunk=3 instead of the recorded 2.
                time_chunk=3, xy_chunk=4, time_chunk_1d=2,
                local_staging_dir=str(tmp_path / 'local.zarr'),
                progress_dir=PROGRESS_DIR,
            )

    def test_local_progress_dir_rejected(self, tmp_path):
        # A local marker directory would die with the EC2 instance, which
        # defeats the purpose.
        with pytest.raises(ValueError, match='must be an s3:// path'):
            _run(tmp_path, progress_dir=str(tmp_path / 'progress'))


# ---------------------------------------------------------------------------
# 'time' coordinate encoding: build_encoding() must pin units/calendar/
# dtype explicitly, not leave them for xarray's CF encoder to infer.
# ---------------------------------------------------------------------------

class TestTimeCoordinateEncoding:

    def test_units_calendar_dtype_are_pinned(self, tmp_path, local_only_run):
        # Regression: left unset, xarray derives a reference datetime from
        # this cube's own first 'time' value and whatever dtype that
        # happens to produce, instead of one shared convention -- so every
        # deep-copy cube would get a different (epoch, dtype) pair. Must
        # match the source virtual cube's own GPS-epoch/float64 scheme
        # (utils.Units.gps_epoch_date) exactly.
        _run(tmp_path, progress_dir=PROGRESS_DIR, keep_local_staging=True)

        time_array = zarr.open_group(
            str(tmp_path / 'local.zarr'), mode='r', zarr_format=3
        )[utils.Coords.TIME]

        assert time_array.attrs[utils.Units.name] == utils.Units.gps_epoch_date
        assert time_array.attrs[utils.Units.calendar_name] == utils.Units.proleptic_gregorian
        assert time_array.dtype == np.dtype('float64')


# ---------------------------------------------------------------------------
# time_collisions.uniquify_granule_times(): the collision space it must
# catch is bigger than exact datetime64 equality -- any two times that alias
# to the identical on-disk float64 value once CF-encoded collide too.
# ---------------------------------------------------------------------------

class TestTimeCollisionsDeduplication:

    def test_sub_ulp_nanosecond_offsets_in_2018_are_deduplicated(self):
        # At 2018's magnitude (~1.2e9 seconds since the 1980 GPS epoch),
        # float64's precision floor is ~238ns -- granules whose true
        # mid_dates differ by only a few nanoseconds alias to the SAME
        # stored value despite not being literally equal datetime64 values.
        # Granules 0-3 (1-3ns apart) collide; granule 4 (500ns apart, above
        # the ULP) does not.
        base = np.datetime64('2018-06-15T12:00:00.000000000')
        offsets_ns = [0, 1, 2, 3, 500]
        times = np.array(
            [base + np.timedelta64(o, 'ns') for o in offsets_ns], dtype='datetime64[ns]'
        )
        urls = [f'granule_{i}.nc' for i in range(len(offsets_ns))]

        encoded_before = tc.encode_time_values(times)
        assert len(set(encoded_before.tolist())) < len(offsets_ns), (
            'premise of this test no longer holds -- these offsets no '
            'longer alias to the same float64 value at this magnitude'
        )

        datasets = [
            xr.Dataset(
                {'v': (('time',), [1])},
                coords={'time': ('time', [t])},
                attrs={'granule_url': u},
            )
            for u, t in zip(urls, times)
        ]

        taken = set()
        num_bumped = tc.uniquify_granule_times(datasets, taken)

        # granules 1-3 collide with granule 0's aliased value; granule 4
        # never collides, so only 3 of the 5 need moving.
        assert num_bumped == 3
        assert datasets[0]['time'].values[0] == times[0]
        assert datasets[4]['time'].values[0] == times[4]

        final_times = np.array(
            [ds['time'].values[0] for ds in datasets], dtype='datetime64[ns]'
        )
        final_encoded = tc.encode_time_values(final_times)

        # Every granule now has a distinct, stable on-disk encoding, and it
        # matches exactly what was validated during resolution -- no drift
        # between the value checked and the value a real re-encode produces.
        assert len(set(final_encoded.tolist())) == len(offsets_ns)
        assert set(final_encoded.tolist()) == taken


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
