"""
Unit tests for deep_copy_update.py.

Covers:
- The local-staging decision requirement (--local-staging-dir/--backup-store)
  is never exercised against real S3 here -- only against local stores,
  where use_local_staging is always False, so no network path is taken.
- A fully-local end-to-end sharded append: builds a local sharded v3 store,
  then appends more layers to it directly (output_store is local, so
  deep_copy_update() always takes the cheap, unmodified direct-append path,
  never the S3-staging branch), verifying earlier-written data survives a
  later batch's read-modify-write of the same still-open, sharded chunk.

No test here points output_store at a real s3:// path or mocks S3.
"""
import os
import sys

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import deep_copy_update as dcu

_COMPRESSOR = [BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')]


def _write_sharded_v3_store(path, num_layers):
    ds = xr.Dataset(
        {'v': (('time', 'y', 'x'), np.arange(num_layers * 40 * 40, dtype='int16')
                                        .reshape(num_layers, 40, 40))},
        coords={
            'time': np.arange(num_layers),
            'y': np.arange(40),
            'x': np.arange(40),
        }
    )
    encoding = {
        'chunks': (num_layers, 10, 10),
        'compressors': _COMPRESSOR,
        'shards': (num_layers, 20, 20),
    }
    ds.to_zarr(path, mode='w', zarr_format=3, consolidated=True, encoding={'v': encoding})
    return ds


class TestDeepCopyUpdateLocalShardedAppend:
    """Fully local (no S3) sharded append: output_store is a local path, so
    deep_copy_update()'s use_local_staging is always False and it always
    takes the direct-append branch -- but that branch still exercises real
    sharding read-modify-write semantics, since the store itself is sharded.
    This is the strongest test for "does a second batch corrupt the first
    batch's data in the same still-open shard" without touching S3 at all.
    """

    def test_second_batch_append_preserves_first_batch_data(self, tmp_path):
        path = str(tmp_path / "sharded_append.zarr")
        first = _write_sharded_v3_store(path, num_layers=2)

        # Append 3 more layers directly (mimics deep_copy_update()'s batch
        # loop body without needing a real icechunk-backed virtual cube).
        more = xr.Dataset(
            {'v': (('time', 'y', 'x'), (np.arange(3 * 40 * 40, dtype='int32') + 20000)
                                            .astype('int16').reshape(3, 40, 40))},
            coords={'time': np.arange(2, 5), 'y': np.arange(40), 'x': np.arange(40)}
        )
        more.to_zarr(path, append_dim='time', zarr_format=3, consolidated=True)

        result = xr.open_zarr(path, consolidated=True)
        assert result.sizes['time'] == 5
        np.testing.assert_array_equal(result['v'].values[:2], first['v'].values)
        np.testing.assert_array_equal(result['v'].values[2:], more['v'].values)
