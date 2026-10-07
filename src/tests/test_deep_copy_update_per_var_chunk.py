"""
Unit tests for deep_copy_update_per_var_chunk.py.

Follows the same no-S3/no-network approach as
test_deep_copy_cube_per_var_chunk.py (per feedback_no_s3_writing_tests):
every AWS-CLI-invoking call (itslive_utils.s3_copy_using_subprocess) and
every S3 filesystem access (s3fs.S3FileSystem, always swapped for a
_FakeFS -- a plain in-memory set of paths) is stubbed out. What's exercised
here is our own logic -- resize bookkeeping, the boundary-chunk-merge
decision, the manual CF-datetime encoding for 'time', config validation, and
the update's own orchestration/short-circuit/pruning -- never a real network
call.

Covers:
- _prepare_array_for_update(): resizes the local array and re-uploads its
    metadata; only attempts a boundary-chunk merge when old_total_layers
    isn't an exact multiple of the chunk size.
- _download_boundary_chunk_if_present(): skips when the chunk was never
    written (radar-skip case); downloads recursively for a 3D variable's
    directory-shaped chunk path, as a single file for a 1D/time one.
- _write_time_coord_and_upload(): writes new 'time' values CF-encoded to
    match the store's own recorded units/calendar; a resumed/boundary write
    only touches the new portion, leaving already-written real values
    (restored by a prior merge) untouched; per-chunk resumability markers.
- _validate_update_config(): any mismatch is a hard failure.
- _find_incomplete_update(): finds a transition directory with a recorded
    config but no top-level _SUCCESS (an interrupted attempt); ignores
    completed transitions and ones missing even their config; raises if
    more than one incomplete transition is found.
- deep_copy_update_per_var_chunk(): missing creation run_config.json raises;
    an up-to-date store short-circuits before touching any array; a full run
    prepares+writes every time-indexed array (the 'time' coordinate, every
    1D variable, every 3D variable) and marks/prunes the update's own
    progress; local staging cleanup; an incomplete update is resumed from
    its recorded old_total_layers even when the store's own (corrupted)
    declared shape already claims the target -- the exact bug this module
    was written to fix -- with shrink-raises/grew-warns-and-clamps behavior
    at the boundary.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import deep_copy_update_per_var_chunk as ducpvc
import deep_copy_cube_progress as dcp
from itscube_types import ImgPairInfo


PROGRESS_DIR = 's3://bucket/progress'
OUTPUT_STORE = 's3://bucket/cubes/mycube.zarr'


class _FakeFS:
    """In-memory stand-in for s3fs.S3FileSystem -- see
    test_deep_copy_cube_per_var_chunk.py's identical class for the full
    rationale. No S3 or network semantics at all."""

    def __init__(self, existing=()):
        self.paths = set(existing)
        self.contents = {}

    def exists(self, path):
        return path in self.paths or any(p.startswith(f'{path}/') for p in self.paths)

    def ls(self, path):
        prefix = f'{path.rstrip("/")}/'
        children = {
            p[len(prefix):].split('/')[0]
            for p in self.paths
            if p.startswith(prefix) and len(p) > len(prefix)
        }
        return [f'{prefix}{child}' for child in children]

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
        self.paths.add(path)
        self.contents[path] = json.dumps(payload)


# ---------------------------------------------------------------------------
# _prepare_array_for_update(): resize + metadata re-upload + boundary-chunk
# merge decision.
# ---------------------------------------------------------------------------

def _build_1d_array_store(tmp_path, var_name, total_layers, chunk_size, name='store'):
    path = str(tmp_path / f'{name}.zarr')
    ds = xr.Dataset(
        {var_name: ((utils.Coords.TIME,), np.arange(total_layers))},
        coords={utils.Coords.TIME: np.arange(total_layers).astype('datetime64[ns]')},
    )
    ds.to_zarr(
        path, mode='w',
        encoding={var_name: {'chunks': (chunk_size,)}, utils.Coords.TIME: {'chunks': (chunk_size,)}},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    return path


class TestPrepareArrayForUpdate:

    def test_resizes_and_uploads_metadata_with_no_boundary_chunk_to_merge(self, tmp_path, monkeypatch):
        # old_total_layers=4 is an exact multiple of chunk_size=2 -- the
        # store's last chunk was fully written, so nothing to merge.
        local_store = _build_1d_array_store(tmp_path, 'url', total_layers=4, chunk_size=2)
        uploaded_meta = []
        merge_calls = []
        monkeypatch.setattr(ducpvc, '_upload_var_metadata', lambda *a: uploaded_meta.append(a))
        monkeypatch.setattr(ducpvc, '_download_boundary_chunk_if_present', lambda *a: merge_calls.append(a))

        ducpvc._prepare_array_for_update(
            None, OUTPUT_STORE, local_store, 'url', chunk_size=2,
            old_total_layers=4, new_total_layers=6, extra_dims_sizes=(), is_dir=False
        )

        arr = zarr.open_group(local_store, mode='r', zarr_format=3)['url']
        assert arr.shape == (6,)
        assert uploaded_meta == [(local_store, OUTPUT_STORE, 'url')]
        assert merge_calls == []

    def test_merges_boundary_chunk_when_old_total_layers_is_ragged(self, tmp_path, monkeypatch):
        # old_total_layers=3 is NOT a multiple of chunk_size=2 -- the store's
        # last chunk (index 1) was only partially written and must be merged
        # before anything new gets written into it.
        local_store = _build_1d_array_store(tmp_path, 'url', total_layers=3, chunk_size=2)
        monkeypatch.setattr(ducpvc, '_upload_var_metadata', lambda *a: None)
        merge_calls = []
        monkeypatch.setattr(ducpvc, '_download_boundary_chunk_if_present', lambda *a: merge_calls.append(a))

        ducpvc._prepare_array_for_update(
            'FAKE_S3', OUTPUT_STORE, local_store, 'url', chunk_size=2,
            old_total_layers=3, new_total_layers=5, extra_dims_sizes=(), is_dir=False
        )

        arr = zarr.open_group(local_store, mode='r', zarr_format=3)['url']
        assert arr.shape == (5,)
        # boundary chunk index = old_total_layers // chunk_size = 3 // 2 = 1.
        assert merge_calls == [('FAKE_S3', OUTPUT_STORE, local_store, 'url', 1, False)]

    def test_resizes_3d_array_with_spatial_dims(self, tmp_path, monkeypatch):
        path = str(tmp_path / 'store3d.zarr')
        ds = xr.Dataset(
            {'vx': (('time', 'y', 'x'), np.zeros((4, 3, 2), dtype='int16'))},
            coords={
                utils.Coords.TIME: np.arange(4).astype('datetime64[ns]'),
                utils.Coords.Y: np.arange(3, dtype='float64'),
                utils.Coords.X: np.arange(2, dtype='float64'),
            },
        )
        ds.to_zarr(
            path, mode='w', encoding={'vx': {'chunks': (2, 3, 2)}},
            zarr_format=3, consolidated=False, safe_chunks=False,
        )
        monkeypatch.setattr(ducpvc, '_upload_var_metadata', lambda *a: None)
        monkeypatch.setattr(ducpvc, '_download_boundary_chunk_if_present', lambda *a: None)

        ducpvc._prepare_array_for_update(
            None, OUTPUT_STORE, path, 'vx', chunk_size=2,
            old_total_layers=4, new_total_layers=6, extra_dims_sizes=(3, 2), is_dir=True
        )

        arr = zarr.open_group(path, mode='r', zarr_format=3)['vx']
        assert arr.shape == (6, 3, 2)


# ---------------------------------------------------------------------------
# _download_boundary_chunk_if_present().
# ---------------------------------------------------------------------------

class _FakeS3Exists:
    def __init__(self, existing=()):
        self.existing = set(existing)

    def exists(self, path):
        return path in self.existing


class TestDownloadBoundaryChunkIfPresent:

    def test_skips_download_when_chunk_was_never_written(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(ducpvc.itslive_utils, 's3_copy_using_subprocess', lambda *a, **k: calls.append(a))

        ducpvc._download_boundary_chunk_if_present(
            _FakeS3Exists(), OUTPUT_STORE, str(tmp_path / 'local.zarr'), 'm11', 3, is_dir=True
        )

        assert calls == []

    def test_downloads_recursively_for_a_3d_variable(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(ducpvc.itslive_utils, 's3_copy_using_subprocess', lambda *a, **k: calls.append(a[0]))
        s3_path = f'{OUTPUT_STORE}/vx/c/3'

        ducpvc._download_boundary_chunk_if_present(
            _FakeS3Exists({s3_path}), OUTPUT_STORE, str(tmp_path / 'local.zarr'), 'vx', 3, is_dir=True
        )

        assert len(calls) == 1
        assert calls[0][:3] == ["aws", "s3", "cp"]
        assert "--recursive" in calls[0]
        assert calls[0][-2:] == [s3_path, str(tmp_path / 'local.zarr' / 'vx' / 'c' / '3')]

    def test_downloads_single_file_for_a_1d_variable(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(ducpvc.itslive_utils, 's3_copy_using_subprocess', lambda *a, **k: calls.append(a[0]))
        s3_path = f'{OUTPUT_STORE}/url/c/1'

        ducpvc._download_boundary_chunk_if_present(
            _FakeS3Exists({s3_path}), OUTPUT_STORE, str(tmp_path / 'local.zarr'), 'url', 1, is_dir=False
        )

        assert len(calls) == 1
        assert "--recursive" not in calls[0]
        assert calls[0][-2:] == [s3_path, str(tmp_path / 'local.zarr' / 'url' / 'c' / '1')]


# ---------------------------------------------------------------------------
# _write_time_coord_and_upload(): the manual CF-datetime encoding, and its
# resumability/start_layer behavior.
# ---------------------------------------------------------------------------

def _build_time_store(
    tmp_path, old_total_layers, new_total_layers, chunk_size, consolidated=False
):
    """A real local store whose 'time' array already has old_total_layers of
    real, CF-encoded data (written the normal xarray way, so its boundary
    chunk is genuinely ragged if old_total_layers isn't a chunk_size
    multiple), then resized (metadata-only, matching what
    _prepare_array_for_update() would have already done by the time
    _write_time_coord_and_upload() runs) up to new_total_layers.

    consolidated=True consolidates BEFORE the resize, reproducing the state a
    real update actually runs in: _download_store_skeleton() pulls down the
    consolidated root zarr.json creation uploaded, and nothing rebuilds it
    between the resize and the append. That leaves the root advertising the
    PRE-resize length while each array's own zarr.json carries the new one --
    which zarr resolves in favor of the stale root unless a reader opts out
    (see the regression test below). Defaults to False so every other test
    here keeps exercising the simpler no-stale-metadata case."""
    path = str(tmp_path / 'time_store.zarr')
    old_times = np.array(
        [np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(old_total_layers)]
    )
    ds = xr.Dataset(coords={utils.Coords.TIME: old_times})
    ds.to_zarr(
        path, mode='w', encoding={utils.Coords.TIME: {'chunks': (chunk_size,)}},
        zarr_format=3, consolidated=False, safe_chunks=False,
    )
    if consolidated:
        zarr.consolidate_metadata(path)
    zarr.open_group(
        path, mode='r+', zarr_format=3, use_consolidated=False
    )[utils.Coords.TIME].resize((new_total_layers,))
    return path, old_times


def _read_back_times(local_store):
    return xr.open_zarr(local_store, consolidated=False)[utils.Coords.TIME].values


class TestWriteTimeCoordAndUpload:

    def test_writes_new_values_matching_the_stores_own_encoding(self, tmp_path, monkeypatch):
        local_store, old_times = _build_time_store(tmp_path, old_total_layers=4, new_total_layers=4, chunk_size=2)
        cube = xr.Dataset(coords={utils.Coords.TIME: old_times})
        monkeypatch.setattr(ducpvc, '_upload_chunk', lambda *a: None)

        ducpvc._write_time_coord_and_upload(cube, local_store, OUTPUT_STORE, chunk_size=2, total_layers=4)

        np.testing.assert_array_equal(_read_back_times(local_store), old_times)

    def test_start_layer_preserves_merged_old_data_and_writes_only_new(self, tmp_path, monkeypatch):
        # old_total_layers=3, chunk_size=2: chunk 1 (layers 2:4) is the
        # boundary chunk, ragged -- only layer 2 is real old data. Resized to
        # new_total_layers=5. A prior _prepare_array_for_update() merge would
        # have restored that real old data (already present here, since this
        # store was built with real writes, not a bare template) -- this call
        # must leave it untouched and only fill in the genuinely new [3:5).
        local_store, old_times = _build_time_store(tmp_path, old_total_layers=3, new_total_layers=5, chunk_size=2)
        new_times = np.array([np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(5)])
        cube = xr.Dataset(coords={utils.Coords.TIME: new_times})
        uploaded = []
        monkeypatch.setattr(ducpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        ducpvc._write_time_coord_and_upload(
            cube, local_store, OUTPUT_STORE, chunk_size=2, total_layers=5, start_layer=3
        )

        # Chunk 0 (layers 0:2) is entirely before start_layer -- never
        # iterated, never (re)uploaded. Chunk 1 (2:4, the ragged boundary)
        # and chunk 2 (4:5, entirely new) both get written/uploaded.
        assert uploaded == [1, 2]
        result = _read_back_times(local_store)
        np.testing.assert_array_equal(result[:3], old_times)
        np.testing.assert_array_equal(result[3:5], new_times[3:5])

    def test_chunk_marker_skips_reupload(self, tmp_path, monkeypatch):
        local_store, old_times = _build_time_store(tmp_path, old_total_layers=4, new_total_layers=4, chunk_size=2)
        cube = xr.Dataset(coords={utils.Coords.TIME: old_times})
        progress, fake_fs = _make_progress(existing={f'{BASE}/{utils.Coords.TIME}/0.done'})
        uploaded = []
        monkeypatch.setattr(ducpvc, '_upload_chunk', lambda *a: uploaded.append(a[-1]))

        ducpvc._write_time_coord_and_upload(
            cube, local_store, OUTPUT_STORE, chunk_size=2, total_layers=4, progress=progress
        )

        assert uploaded == [1]
        assert f'{BASE}/{utils.Coords.TIME}/_SUCCESS' in fake_fs.paths

    def test_variable_marker_short_circuits(self, tmp_path, monkeypatch):
        local_store, old_times = _build_time_store(tmp_path, old_total_layers=4, new_total_layers=4, chunk_size=2)
        cube = xr.Dataset(coords={utils.Coords.TIME: old_times})
        progress, _ = _make_progress(existing={f'{BASE}/{utils.Coords.TIME}/_SUCCESS'})
        monkeypatch.setattr(
            ducpvc, '_upload_chunk',
            lambda *a: pytest.fail('must not upload when the variable is already done')
        )

        ducpvc._write_time_coord_and_upload(
            cube, local_store, OUTPUT_STORE, chunk_size=2, total_layers=4, progress=progress
        )

    def test_appends_land_despite_stale_consolidated_metadata(self, tmp_path, monkeypatch):
        # Regression: a real update resizes arrays in a store whose root
        # zarr.json still carries creation's CONSOLIDATED block, so the root
        # advertises the pre-resize length. zarr prefers that block, so an
        # open that doesn't opt out reports the OLD shape -- and the append
        # past it is a SILENT no-op (no exception, chunk still marked done,
        # layers reading back as fill value). Every other test here builds
        # consolidated=False and therefore cannot catch it. See
        # utils/check_stale_consolidated_metadata.py.
        local_store, old_times = _build_time_store(
            tmp_path, old_total_layers=3, new_total_layers=5, chunk_size=8,
            consolidated=True,
        )
        new_times = np.array(
            [np.datetime64('2020-01-01') + np.timedelta64(i, 'D') for i in range(5)]
        )
        cube = xr.Dataset(coords={utils.Coords.TIME: new_times})
        monkeypatch.setattr(ducpvc, '_upload_chunk', lambda *a: None)

        # Guard the premise: without opting out, the store really does look
        # shorter than it is. If zarr ever stops preferring the stale block,
        # this assertion fails and the test below becomes vacuous.
        stale_shape = zarr.open_group(
            local_store, mode='r', zarr_format=3
        )[utils.Coords.TIME].shape
        assert stale_shape == (3,), (
            f'expected the stale consolidated block to report the pre-resize '
            f'length, got {stale_shape} -- premise of this regression no '
            f'longer holds'
        )

        ducpvc._write_time_coord_and_upload(
            cube, local_store, OUTPUT_STORE, chunk_size=8, total_layers=5, start_layer=3
        )

        result = _read_back_times(local_store)
        np.testing.assert_array_equal(result[:3], old_times)
        np.testing.assert_array_equal(result[3:5], new_times[3:5])


BASE = f'{PROGRESS_DIR}/mycube.zarr'


def _make_progress(existing=(), config=None):
    fake_fs = _FakeFS(existing)
    if config is not None:
        fake_fs.write_json(f'{BASE}/{dcp.RUN_CONFIG_NAME}', config)
    return dcp._Progress(fake_fs, BASE), fake_fs


# ---------------------------------------------------------------------------
# _validate_update_config(): hard-fail on any mismatch.
# ---------------------------------------------------------------------------

class TestValidateUpdateConfig:

    def _config(self, **overrides):
        base = ducpvc._build_update_config(
            OUTPUT_STORE, 4, 6,
            {'time_chunk': 2, 'xy_chunk': 4, 'time_chunk_1d': 2, 'xy_shard_multiplier': 1},
        )
        base.update(overrides)
        return base

    def test_identical_config_does_not_raise(self):
        progress, _ = _make_progress()
        ducpvc._validate_update_config(self._config(), self._config(), progress)

    @pytest.mark.parametrize('key,changed', [
        ('time_chunk', 5), ('xy_chunk', 8), ('time_chunk_1d', 7),
        ('xy_shard_multiplier', 4), ('old_total_layers', 3), ('new_total_layers', 7),
        ('output_store', 's3://bucket/cubes/other.zarr'),
    ])
    def test_any_mismatch_raises(self, key, changed):
        progress, _ = _make_progress()
        recorded = self._config()
        current = self._config(**{key: changed})

        with pytest.raises(RuntimeError, match='disagree with those a prior attempt recorded'):
            ducpvc._validate_update_config(recorded, current, progress)


# ---------------------------------------------------------------------------
# _find_incomplete_update(): detects an interrupted attempt that the store's
# own declared shape can no longer reveal on its own (see this module's
# docstring for the mechanics of why).
# ---------------------------------------------------------------------------

class TestFindIncompleteUpdate:

    def test_no_updates_dir_returns_none(self):
        creation_progress, _ = _make_progress()
        assert ducpvc._find_incomplete_update(creation_progress) is None

    def test_completed_transition_is_ignored(self):
        creation_progress, fake_fs = _make_progress()
        transition_base = f'{BASE}/updates/4_to_6'
        fake_fs.write_json(
            f'{transition_base}/{dcp.RUN_CONFIG_NAME}',
            {'old_total_layers': 4, 'new_total_layers': 6}
        )
        fake_fs.touch(f'{transition_base}/{dcp.SUCCESS_MARKER}')

        assert ducpvc._find_incomplete_update(creation_progress) is None

    def test_transition_missing_config_is_ignored(self):
        # write_config() hadn't landed yet either -- nothing to resume from,
        # and nothing on disk yet that could have been corrupted.
        creation_progress, fake_fs = _make_progress()
        fake_fs.touch(f'{BASE}/updates/4_to_6/some_var/0.done')

        assert ducpvc._find_incomplete_update(creation_progress) is None

    def test_incomplete_transition_is_returned(self):
        creation_progress, fake_fs = _make_progress()
        transition_base = f'{BASE}/updates/4_to_6'
        config = {'old_total_layers': 4, 'new_total_layers': 6}
        fake_fs.write_json(f'{transition_base}/{dcp.RUN_CONFIG_NAME}', config)

        result = ducpvc._find_incomplete_update(creation_progress)

        assert result is not None
        old_total_layers, new_total_layers, update_progress, recorded_config = result
        assert (old_total_layers, new_total_layers) == (4, 6)
        assert update_progress.base == transition_base
        assert recorded_config == config

    def test_multiple_incomplete_transitions_raise(self):
        creation_progress, fake_fs = _make_progress()
        fake_fs.write_json(
            f'{BASE}/updates/4_to_6/{dcp.RUN_CONFIG_NAME}',
            {'old_total_layers': 4, 'new_total_layers': 6}
        )
        fake_fs.write_json(
            f'{BASE}/updates/6_to_8/{dcp.RUN_CONFIG_NAME}',
            {'old_total_layers': 6, 'new_total_layers': 8}
        )

        with pytest.raises(RuntimeError, match='At most one update should ever be in flight'):
            ducpvc._find_incomplete_update(creation_progress)


# ---------------------------------------------------------------------------
# deep_copy_update_per_var_chunk(): orchestration, short-circuit, pruning.
# ---------------------------------------------------------------------------

CUBE_Y = 3
CUBE_X = 2

RECORDED_CREATION_CONFIG = {
    'time_chunk': 2, 'xy_chunk': 4, 'time_chunk_1d': 2, 'xy_shard_multiplier': 1,
}


def _build_minimal_output_store(tmp_path, old_total_layers):
    path = str(tmp_path / 'output.zarr')
    ds = xr.Dataset(
        {'url': (('time',), np.arange(old_total_layers))},
        coords={utils.Coords.TIME: np.arange(old_total_layers).astype('datetime64[ns]')},
    )
    ds.to_zarr(path, mode='w', zarr_format=3, consolidated=True)
    return path


def _make_synthetic_cube(new_total_layers):
    values = np.arange(new_total_layers * CUBE_Y * CUBE_X, dtype='int64').reshape(
        new_total_layers, CUBE_Y, CUBE_X
    ) % 500
    ds = xr.Dataset(
        data_vars={
            'vx': (
                ('time', 'y', 'x'), values.astype('int16'),
                {'missing_value': np.int16(-1)}
            ),
            'url': (('time',), np.array([f'granule_{i}.nc' for i in range(new_total_layers)])),
            ImgPairInfo.mission_img1: (('time',), np.array(['S'] * new_total_layers)),
            ImgPairInfo.satellite_img1: (('time',), np.array(['2A'] * new_total_layers)),
            'landice': (('y', 'x'), np.ones((CUBE_Y, CUBE_X), dtype='uint8')),
        },
        coords={
            utils.Coords.TIME: np.arange(new_total_layers).astype('datetime64[ns]'),
            utils.Coords.Y: np.arange(CUBE_Y, dtype='float64'),
            utils.Coords.X: np.arange(CUBE_X, dtype='float64'),
        },
    )
    return ds


def _install_common_mocks(monkeypatch, fake_fs, new_total_layers):
    monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: fake_fs)
    monkeypatch.setattr(ducpvc, 'open_virtual_cube', lambda *a, **k: _make_synthetic_cube(new_total_layers))
    # Stands in for the real `aws s3 cp --recursive --exclude "*/c/*"`: the
    # output store here is already a plain local directory, so a real
    # copytree is both simpler and more faithful than a recording stub --
    # everything downstream (root/var zarr.json opens) needs a real store.
    monkeypatch.setattr(
        ducpvc, '_download_store_skeleton',
        lambda output_store, local_store: shutil.copytree(output_store, local_store)
    )
    # _Progress.write_config() (used for the update's own run_config.json)
    # shells out to the real `aws` CLI via itslive_utils -- never allow that
    # to reach a real network call in these tests. It writes through the AWS
    # CLI, not s3fs, so it never lands in fake_fs.paths -- record its
    # destinations instead, for tests that need to see it happened.
    copied_to = []
    monkeypatch.setattr(
        dcp.itslive_utils, 's3_copy_using_subprocess',
        lambda command_line, *a, **k: copied_to.append(command_line[4])
    )
    return copied_to


class TestProgressRemoveEntirely:
    # _Progress.remove_entirely() (deep_copy_cube_progress.py) is what
    # deep_copy_update_per_var_chunk() calls to delete a completed update
    # transition's directory in full. The property under test here -- that
    # _SUCCESS is the very last thing removed -- is what guarantees a
    # process killed mid-cleanup still leaves the transition reading as
    # "complete" (is_complete() checks only _SUCCESS) rather than
    # "interrupted", so a later _find_incomplete_update() never mistakes
    # already-correct, already-uploaded data for something needing a
    # pointless redo just because its own progress markers got cleaned up
    # first.

    def test_removes_success_marker_last(self, monkeypatch):
        base = f'{PROGRESS_DIR}/output.zarr/updates/4_to_6'
        fake_fs = _FakeFS(existing={
            f'{base}/{dcp.SUCCESS_MARKER}',
            f'{base}/{dcp.RUN_CONFIG_NAME}',
            f'{base}/vx/{dcp.SUCCESS_MARKER}',
            f'{base}/vx/0.done',
        })
        progress = dcp._Progress(fake_fs, base)

        real_remove_prefix = dcp._remove_prefix
        removed_order = []

        def spy_remove_prefix(s3, prefix):
            if prefix == progress._success_path():
                # Right before _SUCCESS itself is removed, everything else
                # must already be gone -- so a kill exactly here still
                # leaves only _SUCCESS behind.
                assert progress._config_path() not in s3.paths
                assert not any(p.startswith(f'{base}/vx') for p in s3.paths)
            removed_order.append(prefix)
            real_remove_prefix(s3, prefix)

        monkeypatch.setattr(dcp, '_remove_prefix', spy_remove_prefix)

        progress.remove_entirely(['vx'])

        assert removed_order[-1] == progress._success_path()
        assert not any(p.startswith(f'{base}/') for p in fake_fs.paths)


class TestDeepCopyUpdatePerVarChunk:

    def test_s3_staging_dir_rejected_before_any_work(self, tmp_path, monkeypatch):
        # Staging must be a real local path -- see
        # deep_copy_cube.validate_local_staging_dir(). Rejected up front,
        # before the output store is even verified to exist.
        monkeypatch.setattr(
            ducpvc, 'verify_output_store_exists',
            lambda *a, **k: pytest.fail('must not touch the output store; validation comes first')
        )

        with pytest.raises(ValueError, match='must be a local filesystem path'):
            ducpvc.deep_copy_update_per_var_chunk(
                input_store='unused', output_store=OUTPUT_STORE,
                bucket_prefix='s3://unused/', progress_dir=PROGRESS_DIR,
                local_staging_dir='s3://bucket/scratch/cube.zarr',
            )

    def test_missing_creation_run_config_raises(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        monkeypatch.setattr(dcp.s3fs, 'S3FileSystem', lambda *a, **k: fake_fs)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)

        with pytest.raises(RuntimeError, match='No run_config.json found'):
            ducpvc.deep_copy_update_per_var_chunk(
                input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
                progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
            )

    def test_already_up_to_date_short_circuits(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=4)
        monkeypatch.setattr(
            ducpvc, '_prepare_array_for_update',
            lambda *a, **k: pytest.fail('must not touch any array when already up to date')
        )

        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
        )

        assert fake_fs.paths == {f'{creation_base}/{dcp.RUN_CONFIG_NAME}'}

    def test_resumes_incomplete_update_despite_corrupted_shape(self, tmp_path, monkeypatch):
        # Reproduces the real bug directly: the output store's own declared
        # shape already claims 6 layers (exactly what happens once
        # _prepare_array_for_update() resizes every array up front and the
        # first chunk upload re-uploads the root zarr.json -- see this
        # module's docstring), but the update never actually finished --
        # there's a leftover updates/4_to_6 dir with a recorded config and
        # no _SUCCESS. Must resume from the recorded old_total_layers=4, not
        # short-circuit on the corrupted shape.
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=6)
        recorded_update_config = ducpvc._build_update_config(output_store, 4, 6, RECORDED_CREATION_CONFIG)
        fake_fs.write_json(f'{creation_base}/updates/4_to_6/{dcp.RUN_CONFIG_NAME}', recorded_update_config)
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=6)

        prepared = []
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: prepared.append(a[3]))
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: None)

        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
        )

        assert prepared  # did NOT short-circuit as "already up to date"
        update_base = f'{creation_base}/updates/4_to_6'
        # remove_entirely() deletes the whole transition directory on
        # success (unless --keep-progress-markers), not just its
        # per-variable/per-chunk markers.
        assert not any(p.startswith(f'{update_base}/') for p in fake_fs.paths)

    def test_uploads_consolidated_root_before_marking_complete(self, tmp_path, monkeypatch):
        # Defensive upload (see its comment in deep_copy_update_per_var_chunk):
        # _upload_chunk() normally refreshes S3's root as a side effect of
        # each chunk, so this stubs every write out entirely -- the one shape
        # of run where no chunk is uploaded and nothing else would push the
        # post-resize root. Ordering is the real property under test: the
        # root must be durable BEFORE _SUCCESS claims the update finished,
        # since _find_incomplete_update() skips completed transitions and so
        # would never come back to repair it.
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=6)

        events = []
        monkeypatch.setattr(
            dcp.itslive_utils, 's3_copy_using_subprocess',
            lambda command_line, *a, **k: events.append(command_line[4])
        )
        real_mark_complete = dcp._Progress.mark_complete

        def _recording_mark_complete(self):
            events.append('mark_complete')
            return real_mark_complete(self)

        monkeypatch.setattr(dcp._Progress, 'mark_complete', _recording_mark_complete)
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: None)

        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
        )

        root_dest = f'{output_store.rstrip("/")}/{ducpvc.ROOT_METADATA_FILE}'
        assert root_dest in events, f'root metadata never uploaded; saw {events}'
        assert events.index(root_dest) < events.index('mark_complete')

    def test_incomplete_update_raises_if_cube_shrank_below_its_target(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=6)
        recorded_update_config = ducpvc._build_update_config(output_store, 4, 6, RECORDED_CREATION_CONFIG)
        fake_fs.write_json(f'{creation_base}/updates/4_to_6/{dcp.RUN_CONFIG_NAME}', recorded_update_config)
        # The virtual cube now resolves to fewer layers (5) than the 6 this
        # incomplete update was already targeting.
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=5)

        with pytest.raises(RuntimeError, match='fewer than the 6'):
            ducpvc.deep_copy_update_per_var_chunk(
                input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
                progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
            )

    def test_incomplete_update_finishes_recorded_target_when_cube_grew_further(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=6)
        recorded_update_config = ducpvc._build_update_config(output_store, 4, 6, RECORDED_CREATION_CONFIG)
        fake_fs.write_json(f'{creation_base}/updates/4_to_6/{dcp.RUN_CONFIG_NAME}', recorded_update_config)
        # The virtual cube grew further (to 8) since the interrupted attempt
        # started -- this run must still only finish the recorded 4->6
        # transition, not chase the live 8.
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=8)

        written_3d = []
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: None)
        # Positional args are (cube, local_store, output_store, var_name,
        # chunk_size, total_layers, ...) -- total_layers is a[5].
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: written_3d.append(a[5]))

        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
        )

        assert written_3d and all(total == 6 for total in written_3d)
        update_base = f'{creation_base}/updates/4_to_6'
        # remove_entirely() deletes the whole transition directory on
        # success (unless --keep-progress-markers), not just its
        # per-variable/per-chunk markers.
        assert not any(p.startswith(f'{update_base}/') for p in fake_fs.paths)

    def test_full_run_prepares_and_writes_every_time_indexed_array(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)
        copied_to = _install_common_mocks(monkeypatch, fake_fs, new_total_layers=6)

        prepared = []
        written_time = []
        written_1d = []
        written_3d = []
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: prepared.append(a[3]))
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: written_time.append(True))
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: written_1d.append(a[3]))
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: written_3d.append(a[3]))

        local_staging_dir = str(tmp_path / 'local.zarr')
        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=local_staging_dir,
        )

        # mission_img1/satellite_img1 are also real, time-indexed 1D data
        # variables in the synthetic cube (needed for _compute_radar_mask()),
        # so they're expected here too -- assert on what must appear rather
        # than the exact list, matching
        # test_deep_copy_cube_per_var_chunk.py's own convention.
        assert prepared[0] == utils.Coords.TIME
        assert 'url' in prepared
        assert 'vx' in prepared
        assert prepared[-1] == 'vx'  # 3D vars are always prepared last.
        assert written_time == [True]
        assert 'url' in written_1d
        assert written_3d == ['vx']

        update_base = f'{creation_base}/updates/4_to_6'
        # run_config.json is written via _s3_copy() (the AWS CLI), not
        # s3fs, so it never lands in fake_fs.paths -- check the stubbed AWS
        # CLI's recorded destinations instead.
        assert f'{update_base}/{dcp.RUN_CONFIG_NAME}' in copied_to
        # remove_entirely() deletes the whole transition directory on
        # success (unless --keep-progress-markers), not just its
        # per-variable/per-chunk markers.
        assert not any(p.startswith(f'{update_base}/') for p in fake_fs.paths)
        assert not os.path.exists(local_staging_dir)

    def test_keep_local_staging_retains_directory(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=6)
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: None)

        local_staging_dir = str(tmp_path / 'local.zarr')
        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=local_staging_dir,
            keep_local_staging=True,
        )

        assert os.path.exists(local_staging_dir)

    def test_keep_progress_markers_retains_update_dir(self, tmp_path, monkeypatch):
        fake_fs = _FakeFS()
        creation_base = f'{PROGRESS_DIR}/output.zarr'
        fake_fs.write_json(f'{creation_base}/{dcp.RUN_CONFIG_NAME}', RECORDED_CREATION_CONFIG)
        output_store = _build_minimal_output_store(tmp_path, old_total_layers=4)
        _install_common_mocks(monkeypatch, fake_fs, new_total_layers=6)
        monkeypatch.setattr(ducpvc, '_prepare_array_for_update', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_time_coord_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_1d_and_upload', lambda *a, **k: None)
        monkeypatch.setattr(ducpvc, '_write_var_3d_and_upload', lambda *a, **k: None)

        ducpvc.deep_copy_update_per_var_chunk(
            input_store='unused', output_store=output_store, bucket_prefix='s3://unused/',
            progress_dir=PROGRESS_DIR, local_staging_dir=str(tmp_path / 'local.zarr'),
            keep_progress_markers=True,
        )

        update_base = f'{creation_base}/updates/4_to_6'
        assert f'{update_base}/{dcp.SUCCESS_MARKER}' in fake_fs.paths

    def test_local_progress_dir_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='must be an s3:// path'):
            ducpvc.deep_copy_update_per_var_chunk(
                input_store='unused', output_store=str(tmp_path / 'output.zarr'),
                bucket_prefix='s3://unused/', progress_dir=str(tmp_path / 'progress'),
                local_staging_dir=str(tmp_path / 'local.zarr'),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
