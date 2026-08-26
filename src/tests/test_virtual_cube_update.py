"""
Unit tests for virtual_itslive_cube_per_chunk_update.py.

Covers the pure/local-filesystem-only logic used by the update workflow:
- get_existing_granule_urls(): extracting + normalizing granule URLs from a
  cube's 'granule_url' variable
- filter_new_granules(): P000/skipped/existing filtering and https->s3
  normalization
- load_skipped_granules(): local-filesystem read path (missing file, present
  file) -- the S3 (boto3) branch is not exercised here

Also covers one end-to-end scenario (TestVirtualCubeUpdateEndToEnd): build a
small local-filesystem virtual cube via the creation script's CLI, then
append new granules to it via the update script's CLI, and verify the
result. The icechunk *store* is local (no S3 write, matching
test_deep_copy_update.py's convention), but granule *reads* still hit the
real, public, read-only its-live-data S3 bucket -- same as
test_virtual_cube_generation.py -- so this test needs network access and is
slower than the rest of this module.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itscube_types import Vars
import virtual_itslive_cube_per_chunk_update as vicu

HTTPS_URL = vicu.HTTPS_URL
S3_URL = vicu.S3_URL

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
# P000s), fed to the update script as-is -- its own filter_new_granules()
# is what's expected to exclude the 4 already-processed ones, so this
# exercises that filtering against real repo state rather than a hand-picked
# disjoint list.
_ALL_GRANULES_FILE = Path(__file__).parent.parent / "virtual_input_39files.json"


class TestGetExistingGranuleUrls:
    def test_normalizes_to_s3_and_dedupes(self):
        cube = xr.Dataset({
            Vars.url: (('time',), np.array([
                HTTPS_URL + 'path/to/g1.nc',
                S3_URL + 'path/to/g2.nc',
                HTTPS_URL + 'path/to/g1.nc',
            ]))
        })

        result = vicu.get_existing_granule_urls(cube)

        assert result == {
            S3_URL + 'path/to/g1.nc',
            S3_URL + 'path/to/g2.nc',
        }


class TestFilterNewGranules:
    def test_keeps_genuinely_new_granules_normalized_to_s3(self):
        filtered, p000 = vicu.filter_new_granules(
            [HTTPS_URL + 'a/g1.nc'], skipped=set(), existing=set()
        )

        assert filtered == [S3_URL + 'a/g1.nc']
        assert p000 == []

    def test_excludes_p000_granules_but_still_reports_them(self):
        filtered, p000 = vicu.filter_new_granules(
            [HTTPS_URL + 'a/g1_P000.nc'], skipped=set(), existing=set()
        )

        assert filtered == []
        assert p000 == [S3_URL + 'a/g1_P000.nc']

    def test_excludes_previously_skipped_granules(self):
        skipped = {S3_URL + 'a/g1.nc'}

        filtered, p000 = vicu.filter_new_granules(
            [HTTPS_URL + 'a/g1.nc'], skipped=skipped, existing=set()
        )

        assert filtered == []
        assert p000 == []

    def test_excludes_granules_already_in_cube(self):
        existing = {S3_URL + 'a/g1.nc'}

        filtered, p000 = vicu.filter_new_granules(
            [HTTPS_URL + 'a/g1.nc'], skipped=set(), existing=existing
        )

        assert filtered == []
        assert p000 == []

    def test_mixed_batch_partitions_correctly(self):
        all_urls = [
            HTTPS_URL + 'a/g1.nc',       # genuinely new
            HTTPS_URL + 'a/g2_P000.nc',  # P000
            HTTPS_URL + 'a/g3.nc',       # previously skipped
            S3_URL + 'a/g4.nc',         # already in cube
        ]
        skipped = {S3_URL + 'a/g3.nc'}
        existing = {S3_URL + 'a/g4.nc'}

        filtered, p000 = vicu.filter_new_granules(all_urls, skipped, existing)

        assert filtered == [S3_URL + 'a/g1.nc']
        assert p000 == [S3_URL + 'a/g2_P000.nc']

    def test_empty_input(self):
        filtered, p000 = vicu.filter_new_granules([], skipped=set(), existing=set())

        assert filtered == []
        assert p000 == []


class TestLoadSkippedGranulesLocal:
    def test_loads_and_normalizes_to_s3(self, tmp_path):
        cube_store = str(tmp_path / "my_cube.icechunk")
        skipped_path = vicu.skipped_granules_path(cube_store)
        with open(skipped_path, 'w') as f:
            json.dump([HTTPS_URL + 'a/g1.nc', S3_URL + 'a/g2.nc'], f)

        result = vicu.load_skipped_granules(cube_store)

        assert result == {S3_URL + 'a/g1.nc', S3_URL + 'a/g2.nc'}

    def test_missing_file_raises_runtime_error(self, tmp_path):
        cube_store = str(tmp_path / "missing_cube.icechunk")

        with pytest.raises(RuntimeError):
            vicu.load_skipped_granules(cube_store)


@pytest.fixture
def initial_cube_store(tmp_path):
    """Build a small local virtual cube (4 granules, all with valid data in
    the polygon -- see virtual_itslive_cube_per_chunk.py's own CLI usage
    docstring) via the creation script's CLI, for the update test below to
    append to. Reads real granule data from the public its-live-data S3
    bucket; the icechunk store itself is local (no S3 write).
    """
    if not _INITIAL_GRANULES_FILE.exists():
        pytest.skip(f"Granules file not found: {_INITIAL_GRANULES_FILE}")

    store_path = tmp_path / "update_test_cube.icechunk"
    script = Path(__file__).parent.parent / "virtual_itslive_cube_per_chunk.py"

    cmd = [
        sys.executable, str(script),
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
    """Build a small local virtual cube from 4 granules, then update it via
    the update script's CLI, handing it the *full* 39-granule candidate list
    (a superset of the 4 the cube was built from, no P000s) instead of a
    hand-picked disjoint list -- so filter_new_granules() is exercised
    against real repo state, not a list curated to avoid overlap.

    Not every one of the 39 necessarily has valid data inside the polygon
    (build_virtual_cube_subset() drops those into skipped_granules), so this
    doesn't assert an exact resulting layer count -- only that the cube
    never shrinks, every originally-processed granule survives, and nothing
    outside the candidate list appears.
    """

    def test_update_appends_new_granules_from_full_candidate_list(self, initial_cube_store):
        if not _ALL_GRANULES_FILE.exists():
            pytest.skip(f"Granules file not found: {_ALL_GRANULES_FILE}")

        initial_cube, _ = vicu.open_virtual_cube(str(initial_cube_store))
        initial_num_layers = initial_cube.sizes['time']
        initial_urls = vicu.get_existing_granule_urls(initial_cube)

        all_granules = json.loads(_ALL_GRANULES_FILE.read_text())
        candidate_urls = {g.replace(HTTPS_URL, S3_URL) for g in all_granules}

        update_script = Path(__file__).parent.parent / "virtual_itslive_cube_per_chunk_update.py"
        cmd = [
            sys.executable, str(update_script),
            "--cube-store", str(initial_cube_store),
            "--granules-file", str(_ALL_GRANULES_FILE),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            pytest.fail(
                f"Update failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        updated_cube, _ = vicu.open_virtual_cube(str(initial_cube_store))
        updated_urls = vicu.get_existing_granule_urls(updated_cube)

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
        rerun_cube, _ = vicu.open_virtual_cube(str(initial_cube_store))
        assert rerun_cube.sizes['time'] == updated_cube.sizes['time']
