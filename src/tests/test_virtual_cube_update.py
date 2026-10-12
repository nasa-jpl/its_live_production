"""
Unit tests for virtual_itslive_cube_per_chunk.py's merged create+append
workflow.

virtual_itslive_cube_per_chunk.py's __main__ now auto-detects whether
--output-store already holds a committed cube (ic.Repository.exists()) and
either creates it from scratch or appends new granules to it, using the
exact same CLI flags either way -- so "updating" a cube is just re-running
the creation command against an existing store with a (super-set) granules
file. The old standalone virtual_itslive_cube_per_chunk_update.py is kept
around as a secondary/legacy tool but is no longer what this module tests.

Covers the pure/local-filesystem-only logic used by the append path:
- get_existing_granule_urls(): extracting + normalizing granule URLs from a
  cube's 'granule_url' variable
- _filter_processed_granules(): skipped/existing partitioning (P000
  filtering is no longer a reusable function -- it's an inline list
  comprehension in __main__ -- so it's not unit-tested in isolation here)
- load_skipped_granules(): local-filesystem read path (missing file, present
  file) -- the S3 (boto3) branch is not exercised here

Also covers one end-to-end scenario (TestVirtualCubeUpdateEndToEnd): build a
small local-filesystem virtual cube via the script's CLI, then re-run the
same CLI against the same --output-store with a superset granules file to
append new granules, and verify the result. The icechunk *store* is local
(no S3 write, matching test_deep_copy_update.py's convention), but granule
*reads* still hit the real, public, read-only its-live-data S3 bucket --
same as test_virtual_cube_generation.py -- so this test needs network access
and is slower than the rest of this module.

Not covered: build_virtual_cube_subset()'s internals and the batching loop's
per-batch bookkeeping in isolation -- these are exercised indirectly via the
end-to-end scenario above rather than unit-tested directly, since they're
tightly coupled to real icechunk/virtualizarr state.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import icechunk as ic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itscube_types import Vars
import virtual_itslive_cube_per_chunk as vicpc

HTTPS_URL = vicpc.HTTPS_URL
S3_URL = vicpc.S3_URL

# Same tile/polygon used by test_virtual_cube_generation.py's test_config.
_POLYGON = [
    [-1658887.5, -430072.5],
    [-1597447.5, -430072.5],
    [-1597447.5, -368632.5],
    [-1658887.5, -368632.5],
    [-1658887.5, -430072.5],
]
_PROJECTION = "3031"

# 4 of the 39 granules in _ALL_GRANULES_FILE, used to build the initial cube.
_INITIAL_GRANULES_FILE = Path(__file__).parent.parent / "virtual_input_4files.json"

# All 39 granules for the same tile/polygon (a superset of the 4 above, no
# P000s), fed to the script's second ("update") run as-is -- its own
# _filter_processed_granules() is what's expected to exclude the 4
# already-processed ones, so this exercises that filtering against real repo
# state rather than a hand-picked disjoint list.
_ALL_GRANULES_FILE = Path(__file__).parent.parent / "virtual_input_39files.json"


def _open_cube(store_path):
    """Open a local-filesystem icechunk cube read-only, raw on-disk dtypes.

    Mirrors the inline open pattern used throughout
    test_virtual_cube_generation.py (mask_and_scale=False, zarr_format=3)
    rather than reintroducing a standalone inspect helper -- that's no
    longer part of virtual_itslive_cube_per_chunk.py's merged workflow.

    Inputs:
    store_path (str or Path): path to the local icechunk repository.
    """
    repo = ic.Repository.open(ic.local_filesystem_storage(str(store_path)))
    cube = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False,
    )
    return cube, repo


class TestGetExistingGranuleUrls:
    def test_normalizes_to_s3_and_dedupes(self):
        cube = xr.Dataset({
            Vars.url: (('time',), np.array([
                HTTPS_URL + 'path/to/g1.nc',
                S3_URL + 'path/to/g2.nc',
                HTTPS_URL + 'path/to/g1.nc',
            ]))
        })

        result = vicpc.get_existing_granule_urls(cube)

        assert result == {
            S3_URL + 'path/to/g1.nc',
            S3_URL + 'path/to/g2.nc',
        }


class TestFilterProcessedGranules:
    """_filter_processed_granules() expects `urls` already normalized to
    s3:// and already P000-filtered -- __main__ does both of those via a
    plain list comprehension before calling it (no longer a reusable
    function), so these cases only exercise the skipped/existing
    partitioning that's still this function's job.
    """

    def test_keeps_genuinely_new_granules(self):
        remaining, already_skipped = vicpc._filter_processed_granules(
            [S3_URL + 'a/g1.nc'], skipped=set(), existing=set(), num_p000_skipped=0
        )

        assert remaining == [S3_URL + 'a/g1.nc']
        assert already_skipped == []

    def test_excludes_previously_skipped_granules(self):
        skipped = {S3_URL + 'a/g1.nc'}

        remaining, already_skipped = vicpc._filter_processed_granules(
            [S3_URL + 'a/g1.nc'], skipped=skipped, existing=set(), num_p000_skipped=0
        )

        assert remaining == []
        assert already_skipped == [S3_URL + 'a/g1.nc']

    def test_excludes_granules_already_in_cube_without_marking_skipped(self):
        existing = {S3_URL + 'a/g1.nc'}

        remaining, already_skipped = vicpc._filter_processed_granules(
            [S3_URL + 'a/g1.nc'], skipped=set(), existing=existing, num_p000_skipped=0
        )

        assert remaining == []
        assert already_skipped == []

    def test_mixed_batch_partitions_correctly(self):
        all_urls = [
            S3_URL + 'a/g1.nc',  # genuinely new
            S3_URL + 'a/g3.nc',  # previously skipped
            S3_URL + 'a/g4.nc',  # already in cube
        ]
        skipped = {S3_URL + 'a/g3.nc'}
        existing = {S3_URL + 'a/g4.nc'}

        remaining, already_skipped = vicpc._filter_processed_granules(
            all_urls, skipped, existing, num_p000_skipped=0
        )

        assert remaining == [S3_URL + 'a/g1.nc']
        assert already_skipped == [S3_URL + 'a/g3.nc']

    def test_empty_input(self):
        remaining, already_skipped = vicpc._filter_processed_granules(
            [], skipped=set(), existing=set(), num_p000_skipped=0
        )

        assert remaining == []
        assert already_skipped == []


class TestLoadSkippedGranulesLocal:
    def test_loads_and_normalizes_to_s3(self, tmp_path):
        cube_store = str(tmp_path / "my_cube.icechunk")
        skipped_path = vicpc.skipped_granules_path(cube_store)
        with open(skipped_path, 'w') as f:
            json.dump([HTTPS_URL + 'a/g1.nc', S3_URL + 'a/g2.nc'], f)

        result = vicpc.load_skipped_granules(cube_store)

        assert result == {S3_URL + 'a/g1.nc', S3_URL + 'a/g2.nc'}

    def test_missing_file_raises_runtime_error(self, tmp_path):
        cube_store = str(tmp_path / "missing_cube.icechunk")

        with pytest.raises(RuntimeError):
            vicpc.load_skipped_granules(cube_store)


@pytest.fixture
def virtual_cube_script():
    """Path to virtual_itslive_cube_per_chunk.py -- same script re-invoked
    for both the initial creation run and every "update" run below.
    """
    return Path(__file__).parent.parent / "virtual_itslive_cube_per_chunk.py"


@pytest.fixture
def initial_cube_store(tmp_path, virtual_cube_script):
    """Build a small local virtual cube (4 granules, all with valid data in
    the polygon -- see virtual_itslive_cube_per_chunk.py's own CLI usage
    docstring) via its CLI, for the update tests below to append to. Reads
    real granule data from the public its-live-data S3 bucket; the icechunk
    store itself is local (no S3 write).
    """
    if not _INITIAL_GRANULES_FILE.exists():
        pytest.skip(f"Granules file not found: {_INITIAL_GRANULES_FILE}")

    store_path = tmp_path / "update_test_cube.icechunk"

    cmd = [
        sys.executable, str(virtual_cube_script),
        "--polygon", json.dumps(_POLYGON),
        "--granules-file", str(_INITIAL_GRANULES_FILE),
        "--output-store", str(store_path),
        "--projection", _PROJECTION,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        pytest.fail(
            f"Initial cube creation failed:\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return store_path


class TestVirtualCubeUpdateEndToEnd:
    """Build a small local virtual cube from 4 granules, then re-run the
    same creation CLI against the same --output-store, handing it the
    *full* 39-granule candidate list (a superset of the 4 the cube was built
    from, no P000s) instead of a hand-picked disjoint list -- so
    _filter_processed_granules() is exercised against real repo state, not
    a list curated to avoid overlap. virtual_itslive_cube_per_chunk.py
    auto-detects the existing store (ic.Repository.exists()) and appends
    instead of recreating.

    Not every one of the 39 necessarily has valid data inside the polygon
    (build_virtual_cube_subset() drops those into skipped_granules), so this
    doesn't assert an exact resulting layer count -- only that the cube
    never shrinks, every originally-processed granule survives, and nothing
    outside the candidate list appears.
    """

    def test_update_appends_new_granules_from_full_candidate_list(
        self, initial_cube_store, virtual_cube_script
    ):
        if not _ALL_GRANULES_FILE.exists():
            pytest.skip(f"Granules file not found: {_ALL_GRANULES_FILE}")

        initial_cube, _ = _open_cube(initial_cube_store)
        initial_num_layers = initial_cube.sizes['time']
        initial_urls = vicpc.get_existing_granule_urls(initial_cube)

        all_granules = json.loads(_ALL_GRANULES_FILE.read_text())
        candidate_urls = {g.replace(HTTPS_URL, S3_URL) for g in all_granules}

        cmd = [
            sys.executable, str(virtual_cube_script),
            "--polygon", json.dumps(_POLYGON),
            "--granules-file", str(_ALL_GRANULES_FILE),
            "--output-store", str(initial_cube_store),
            "--projection", _PROJECTION,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            pytest.fail(
                f"Update failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        updated_cube, _ = _open_cube(initial_cube_store)
        updated_urls = vicpc.get_existing_granule_urls(updated_cube)

        print(
            f"\nInitial layers: {initial_num_layers}, "
            f"updated layers: {updated_cube.sizes['time']}, "
            f"candidates: {len(all_granules)}"
        )

        # Nothing already in the cube was lost or duplicated away.
        assert initial_urls <= updated_urls
        assert updated_cube.sizes['time'] >= initial_num_layers

        # Every granule that made it into the cube came from the candidate
        # list handed to the update run (plus whatever was already there).
        assert updated_urls <= (initial_urls | candidate_urls)

        # The 4 initial granules are documented as having valid data in this
        # polygon (see virtual_itslive_cube_per_chunk.py's CLI usage
        # docstring), so re-running the same candidate list is a real
        # no-op update: nothing new to append.
        result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result2.returncode != 0:
            pytest.fail(
                f"Second (no-op) update failed:\nSTDOUT:\n{result2.stdout}\n"
                f"STDERR:\n{result2.stderr}"
            )
        rerun_cube, _ = _open_cube(initial_cube_store)
        assert rerun_cube.sizes['time'] == updated_cube.sizes['time']


class TestNoOpUpdateSkippedGranulesBookkeeping:
    """Regression test for a bug where a no-op update run (candidate list
    contains no genuinely new granules) recorded granules already present
    in the cube into the persisted skipped-granules JSON. A granule can be
    legitimately in the cube's granule_url variable, or legitimately
    skipped, but never both.
    """

    def test_existing_granules_are_not_recorded_as_skipped(
        self, initial_cube_store, virtual_cube_script
    ):
        # Candidate list is exactly the 4 granules the cube was already built
        # from -- _filter_processed_granules() excludes all of them via
        # existing_urls, so the batch loop has nothing to append.
        cmd = [
            sys.executable, str(virtual_cube_script),
            "--polygon", json.dumps(_POLYGON),
            "--granules-file", str(_INITIAL_GRANULES_FILE),
            "--output-store", str(initial_cube_store),
            "--projection", _PROJECTION,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            pytest.fail(
                f"No-op update failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        cube, _ = _open_cube(initial_cube_store)
        existing_urls = vicpc.get_existing_granule_urls(cube)

        skipped_path = Path(vicpc.skipped_granules_path(str(initial_cube_store)))
        assert skipped_path.exists(), "no-op update should still write a skipped-granules file"
        skipped = vicpc.load_skipped_granules(str(initial_cube_store))

        # None of the 4 already-in-cube granules should also appear as skipped.
        assert not (existing_urls & skipped), (
            f"granules recorded as both existing and skipped: {existing_urls & skipped}"
        )
