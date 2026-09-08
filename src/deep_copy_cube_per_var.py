"""
Materialize a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube, batched by
DATA VARIABLE rather than deep_copy_cube.py's time-only batching or
deep_copy_cube_tiled.py's spatial-tile batching.

Why: deep_copy_cube.py's batch_size (default 2000 layers) is far smaller
than the output store's fixed TIME_CHUNK_VALUE (20000), so every batch after
the first forces a full decompress/merge/recompress of the entire
still-open time-chunk across every spatial chunk/shard AND every variable
simultaneously -- quadratic write amplification, root-caused in
src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md. Matching
batch_size to time_chunk (the wiki's preferred fix) eliminates that, but
needs ~256 GiB RAM to hold every variable's full time-chunk, at full spatial
extent, at once.

deep_copy_cube_tiled.py shrinks that RAM need by tiling the *spatial* extent
instead, but each granule's virtual chunk spans the full spatial grid (no
source-side sub-tiling), so tiling the write does not tile the read: every
tile re-fetches/decompresses each touched granule's full spatial extent,
discarding the unwanted pixels afterward -- real, measured read
amplification.

This script shrinks RAM the other way: keep full x/y extent and full
time-chunk (no write amplification, same mechanism as the wiki's Option 1),
but process ONE DATA VARIABLE AT A TIME instead of all variables at once.
Peak RAM becomes one variable's full-chunk footprint (one variable x
time_chunk layers x full x/y extent) instead of the whole cube's. Unlike
spatial tiling, this does not read-amplify: each granule's per-variable
chunk is already an independently addressable virtual reference, so reading
one variable at a time does not re-fetch bytes another variable's read
already covered.

KNOWN TRADEOFF -- read this before using this script for a production run:
processing variables one at a time forgoes any inter-variable parallelism
in the read path (S3 GETs for variable N+1 don't start until variable N's
full chunk has been written), so total wall-clock time is closer to
(num_variables x per-variable read+write time) than deep_copy_cube.py's more
overlapped batch processing. This is a deliberate RAM-vs-wall-clock
tradeoff -- benchmark with --num-layers on a bounded slice before relying on
this for a full production run.

Usage example:
python src/deep_copy_cube_per_var.py \
   --input-store my_virtual_cube.icechunk \
   --output-store my_deep_copy_cube.zarr
"""
import gc
import logging
import sys
import time
import warnings
from datetime import datetime

import xarray as xr
import zarr
from zarr.errors import UnstableSpecificationWarning

import utils
from itscube_types import CubeFormat
from deep_copy_cube import (
   TIME_CHUNK_VALUE,
   X_Y_CHUNK_VALUE,
   TIME_CHUNK_VALUE_1D,
   XY_SHARD_MULTIPLIER,
   open_virtual_cube,
   split_vars_by_time,
   build_encoding,
   _reset_write_encoding,
   resolve_output_store,
   upload_local_staging_dir,
)
from deep_copy_cube_tiled import split_time_vars_by_rank

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Zarr V3 unstable string dtype warnings, same rationale as
# deep_copy_cube.py.
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)


def _write_var_in_chunks(cube, write_target, var_name, chunk_size, total_layers):
   """Write one data variable's full extent, one complete time-chunk at a
   time, via region writes into an already-templated store.

   Each (variable, time-chunk) pair is exactly one load + one write -- no
   sub-batching within a chunk -- so every write is a complete, aligned
   chunk written exactly once. A partial-chunk write would force the same
   decompress/merge/recompress this script exists to avoid (see module
   docstring).

   Explicitly deletes the loaded batch and forces a gc pass after each
   write: xarray/dask Datasets commonly hold internal reference cycles
   (e.g. task-graph closures), which CPython's refcounting alone won't
   collect promptly -- on a RAM-constrained instance, leaving a finished
   chunk's ~10-20GB array pending cyclic collection until Python gets
   around to it can crowd out the next variable's read/decompress
   footprint.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   write_target : str
      Local path or s3:// URL of the store to write into (already
      templated with full shape/dtype/chunks -- see deep_copy_cube_per_var).
   var_name : str
      Name of the data variable to write.
   chunk_size : int
      Number of layers per time-chunk for this variable (time_chunk for 3D
      variables, time_chunk_1d for 1D variables).
   total_layers : int
      Total number of layers to write (honors --num-layers).
   """
   for start in range(0, total_layers, chunk_size):
      stop = min(start + chunk_size, total_layers)
      logging.info(f'Materializing {var_name} layers {start}:{stop} of {total_layers}')

      batch = cube[[var_name]].isel(
         {utils.Coords.TIME: slice(start, stop)}
      ).drop_vars(
         [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X], errors='ignore'
      ).load()
      _reset_write_encoding(batch)
      batch.to_zarr(
         write_target,
         mode='r+',
         region={utils.Coords.TIME: slice(start, stop)},
         zarr_format=3,
         consolidated=False
      )

      logging.info(f'Wrote {var_name} layers {start}:{stop} of {total_layers} to {write_target}')

      del batch
      gc.collect()


def deep_copy_cube_per_var(
   input_store,
   output_store,
   bucket_prefix,
   time_chunk,
   xy_chunk,
   time_chunk_1d,
   xy_shard_multiplier=1,
   local_staging_dir=None,
   keep_local_staging=False,
   num_layers=0
):
   """Materialize a virtual datacube into a real zarr v3 datacube, one data
   variable at a time, each in exactly time_chunk-sized (3D) or
   time_chunk_1d-sized (1D) increments at full spatial extent -- see this
   module's docstring for the RAM-vs-wall-clock tradeoff this makes relative
   to deep_copy_cube.py and deep_copy_cube_tiled.py.

   Like deep_copy_cube_tiled.py (and unlike deep_copy_cube.py's incremental
   append-based construction), this writes the whole store's
   shape/dtype/chunk-grid up front (mode='w', compute=False) and fills it in
   via region writes (mode='r+') -- here, per (variable, time-chunk) pair
   instead of per (spatial-tile, time-chunk) pair. Static 2D (y,x) vars are
   written once, full extent, matching both existing scripts.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      Path to write the deep-copy zarr store to (s3:// or local).
   bucket_prefix : str
      S3 URL prefix the virtual chunk container resolves granule references
      against (see deep_copy_cube.open_virtual_cube).
   time_chunk : int
      Chunk size along 'time' for 3D variables, and the number of layers
      materialized per write for each 3D variable.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables, and the number of layers
      materialized per write for each 1D variable.
   xy_shard_multiplier : int
      Must be >= 1; 1 (the default) leaves the store unsharded. See
      deep_copy_cube.XY_SHARD_MULTIPLIER for the recommended value to pass
      explicitly.
   local_staging_dir : str, optional
      If set, `output_store` must be an s3:// path. All writes go to this
      local directory first, and the whole store is uploaded to
      `output_store` with a single `aws s3 cp --recursive` at the end. See
      deep_copy_cube.py's parameter of the same name.
   keep_local_staging : bool
      If True, keep `local_staging_dir` after a successful upload instead of
      removing it. Ignored if `local_staging_dir` is not set.
   num_layers : int
      If > 0, only materialize the first `num_layers` layers of the virtual
      cube. 0 (the default) processes every layer.
   """
   if local_staging_dir and not output_store.startswith(utils.S3_PREFIX):
      raise ValueError(
         "--local-staging-dir only applies when --output-store is an s3:// "
         f"path, got {output_store}"
      )

   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   if num_layers > 0 and num_layers < total_layers:
      logging.info(f'Limiting to first {num_layers} of {total_layers} layers (--num-layers)')
      total_layers = num_layers

   if total_layers == 0:
      logging.info(f'{input_store} has no layers, nothing to deep-copy')
      return

   time_vars, static_vars = split_vars_by_time(cube)
   vars_3d, vars_1d = split_time_vars_by_rank(cube, time_vars)
   logging.info(
      f'Batching over {len(vars_1d)} 1D variable(s) '
      f'({time_chunk_1d} layers/write) and {len(vars_3d)} 3D variable(s) '
      f'({time_chunk} layers/write)'
   )

   encoding = build_encoding(
      cube, time_chunk, xy_chunk, time_chunk_1d,
      xy_shard_multiplier
   )

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   resolve_output_store(output_store)
   write_target = resolve_output_store(local_staging_dir) if local_staging_dir else output_store

   # Template: declare every variable's shape/dtype/chunks/encoding up
   # front. compute=False defers writing pixel data for every dask-backed
   # variable; 'time'/'x'/'y' coordinates are eagerly-loaded index variables
   # regardless of chunks=, so they get written for real by this call --
   # needed before any region write below. mode='r+' (used by every later
   # write) requires every variable to already exist, so this template must
   # declare all of them (time_vars + static_vars) up front, sliced to
   # total_layers along 'time' to honor --num-layers.
   #
   # safe_chunks=False: the virtual cube's own per-granule dask chunking
   # (chunk size 1 along 'time') doesn't match encoding's much larger
   # time_chunk, which xarray's dask-parallel-write safety check flags as
   # unsafe. That check exists to prevent multiple dask workers racing on the
   # same zarr chunk during a real parallel write -- it doesn't apply here
   # since compute=False never executes the write at all, only declares
   # metadata; the actual pixel data is written later, per (variable,
   # time-chunk), from already-.load()-ed (non-dask) batches.
   template = xr.merge([
      cube[time_vars].isel({utils.Coords.TIME: slice(0, total_layers)}),
      cube[static_vars]
   ])
   _reset_write_encoding(template)
   # consolidated=False: metadata is consolidated exactly once, after every
   # region write finishes. Consolidating on each write would re-scan/rewrite
   # the whole store's metadata redundantly.
   template.to_zarr(
      write_target,
      mode='w',
      compute=False,
      encoding=encoding,
      zarr_format=3,
      consolidated=False,
      safe_chunks=False
   )
   logging.info(f'Created template store at {write_target}')

   # Static 2D (y,x) vars: written once, full extent, no per-variable
   # chunking -- they have no 'time' dimension to chunk over.
   static_batch = cube[static_vars].load()
   _reset_write_encoding(static_batch)
   static_batch.to_zarr(write_target, mode='r+', zarr_format=3, consolidated=False)
   logging.info(f'Wrote {len(static_vars)} static variable(s) to {write_target}')

   for var_name in vars_1d:
      _write_var_in_chunks(cube, write_target, var_name, time_chunk_1d, total_layers)

   for var_name in vars_3d:
      _write_var_in_chunks(cube, write_target, var_name, time_chunk, total_layers)

   # Consolidate metadata once, now that every region write is done -- instead
   # of re-consolidating (a full-store metadata rescan/rewrite) on each write
   # above. Matches deep_copy_cube.py's intended convention of a consolidated
   # output store, without the per-write cost.
   zarr.consolidate_metadata(write_target)
   logging.info(f'Consolidated metadata at {write_target}')

   if local_staging_dir:
      upload_local_staging_dir(local_staging_dir, output_store, keep_local_staging)

   logging.info(f'Done: deep-copied {total_layers} layers to {output_store}')


if __name__ == '__main__':
   import argparse

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Materialize a virtual ITS_LIVE datacube (icechunk repo built by
      virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube, one
      data variable at a time, each in full time_chunk-sized (3D) or
      time_chunk_1d-sized (1D) writes at full spatial extent -- avoids
      deep_copy_cube.py's batch_size-vs-time_chunk write amplification
      without deep_copy_cube_tiled.py's spatial read-amplification, at the
      cost of losing inter-variable read/write overlap (see this module's
      docstring for the accepted tradeoff).

      Usage example:
      python src/deep_copy_cube_per_var.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr

      # Write a store with sharded 3D variables (--xy-shard-multiplier 4 is
      # the recommended production value, see deep_copy_cube.XY_SHARD_MULTIPLIER):
      python src/deep_copy_cube_per_var.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr \
         --xy-shard-multiplier 4

      # Stage locally, then upload the whole store to S3 in one shot:
      python src/deep_copy_cube_per_var.py \
         --input-store my_virtual_cube.icechunk \
         --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
         --local-staging-dir /local/scratch/my_deep_copy_cube.zarr
      """,
      formatter_class=argparse.RawDescriptionHelpFormatter
   )
   parser.add_argument(
      '--input-store',
      type=str,
      required=True,
      help='Path to the virtual cube icechunk repository (s3:// or local).'
   )
   parser.add_argument(
      '--output-store',
      type=str,
      required=True,
      help='Path to write the deep-copy Zarr v3 store to (s3:// or local).'
   )
   parser.add_argument(
      '--bucket',
      type=str,
      default='s3://its-live-data/',
      help='S3 URL prefix the virtual chunk container resolves granule '
         'references against [%(default)s]'
   )
   parser.add_argument(
      '--time-chunk-value',
      type=int,
      default=TIME_CHUNK_VALUE,
      help='Chunk size along time for 3D (time, y, x) variables, and the '
         'number of layers materialized per write for each 3D variable '
         '[%(default)d].'
   )
   parser.add_argument(
      '--xy-chunk-value',
      type=int,
      default=X_Y_CHUNK_VALUE,
      help='Chunk size along x/y for 3D (time, y, x) variables [%(default)d].'
   )
   parser.add_argument(
      '--time-chunk-value-1d',
      type=int,
      default=TIME_CHUNK_VALUE_1D,
      help='Chunk size for 1D (time,) variables, and the number of layers '
         'materialized per write for each 1D variable [%(default)d].'
   )
   parser.add_argument(
      '--xy-shard-multiplier',
      type=int,
      default=1,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables. A value of 1 '
         f'(the default) disables sharding. Recommended value once sharding '
         f'is enabled: {XY_SHARD_MULTIPLIER} [%(default)d].'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      default=None,
      help='If set, --output-store must be an s3:// path. Write the '
         'deep-copy store to this local directory first, then upload the '
         'whole store to --output-store with a single "aws s3 cp '
         '--recursive" at the end.'
   )
   parser.add_argument(
      '--keep-local-staging',
      action='store_true',
      help='Keep --local-staging-dir after a successful upload instead of '
         'deleting it [%(default)s].'
   )
   parser.add_argument(
      '-n', '--num-layers',
      type=int,
      default=0,
      help='Only materialize the first N layers of the virtual cube '
         '[%(default)d meaning to process all layers].'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   deep_copy_cube_per_var(
      args.input_store,
      args.output_store,
      args.bucket,
      args.time_chunk_value,
      args.xy_chunk_value,
      args.time_chunk_value_1d,
      args.xy_shard_multiplier,
      args.local_staging_dir,
      args.keep_local_staging,
      args.num_layers
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
