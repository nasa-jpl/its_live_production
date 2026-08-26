"""
Unit tests for itslive_utils.py's Zarr v3/shard-aware helpers
(shard_key_v3, identify_datacube_latest_shards).

These exercise real local Zarr v3 stores (no S3, no mocking) -- matching
this repo's convention of not testing S3-writing code paths even mocked
(backup_datacube_latest_shards()'s actual S3 copy work is not tested here;
it reuses the already-covered backup_chunk()/Parallel() machinery).
"""
import os
import sys

import numpy as np
import pytest
import xarray as xr
from zarr.codecs import BloscCodec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itslive_utils

_COMPRESSOR = [BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')]


@pytest.fixture
def small_v3_store(tmp_path):
    """A local Zarr v3 store with one sharded 3D var, one unsharded 1D var,
    one static 2D var, and one scalar var -- enough to exercise every branch
    of identify_datacube_latest_shards() without any network/S3 access.

    'v' is 64x64 with 8x8 inner chunks and 32x32 shards: 2x2 shards/side.
    """
    ds = xr.Dataset(
        {
            'v': (('time', 'y', 'x'), np.zeros((3, 64, 64), dtype='int16')),
            'url': (('time',), np.array(['a', 'b', 'c'])),
            'landice': (('y', 'x'), np.zeros((64, 64), dtype='uint8')),
            'mapping': ((), np.array('')),
        },
        coords={'time': np.arange(3), 'y': np.arange(64), 'x': np.arange(64)}
    )
    store_path = str(tmp_path / "small_v3_store.zarr")
    ds.to_zarr(
        store_path,
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding={
            'v': {'chunks': (3, 8, 8), 'shards': (3, 32, 32), 'compressors': _COMPRESSOR},
            'url': {'chunks': (3,), 'compressors': _COMPRESSOR},
            'landice': {'chunks': (64, 64), 'compressors': _COMPRESSOR},
        }
    )
    return store_path


class TestShardKeyV3:
    def test_shard_key_v3_3d(self):
        assert itslive_utils.shard_key_v3((0, 3, 5)) == 'c/0/3/5'

    def test_shard_key_v3_1d(self):
        assert itslive_utils.shard_key_v3((0,)) == 'c/0'


class TestIdentifyDatacubeLatestShards:
    def test_sharded_3d_var_covers_all_spatial_shards(self, small_v3_store):
        result = itslive_utils.identify_datacube_latest_shards(small_v3_store)

        # 64px / 32px-per-shard = 2 shards/side; only 1 time-shard (3 layers,
        # shard time extent = 3, matching the chunk's time extent).
        assert list(result['v'].ranges[0]) == [0]
        assert list(result['v'].ranges[1]) == [0, 1]
        assert list(result['v'].ranges[2]) == [0, 1]

        # "Last shard" for a 3D var = last time-index, ALL spatial indices --
        # there's only 1 time-shard here, so last == all.
        assert list(result['v'].last_dim_ranges[0]) == [0]
        assert list(result['v'].last_dim_ranges[1]) == [0, 1]
        assert list(result['v'].last_dim_ranges[2]) == [0, 1]

    def test_unsharded_1d_var_falls_back_to_chunks(self, small_v3_store):
        result = itslive_utils.identify_datacube_latest_shards(small_v3_store)

        assert list(result['url'].ranges[0]) == [0]
        assert list(result['url'].last_dim_ranges[0]) == [0]

    def test_static_2d_var_covers_full_extent(self, small_v3_store):
        result = itslive_utils.identify_datacube_latest_shards(small_v3_store)

        assert list(result['landice'].last_dim_ranges[0]) == [0]
        assert list(result['landice'].last_dim_ranges[1]) == [0]

    def test_scalar_var(self, small_v3_store):
        result = itslive_utils.identify_datacube_latest_shards(small_v3_store)

        assert list(result['mapping'].last_dim_ranges[0]) == [0]
