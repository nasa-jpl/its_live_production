"""
Materialize a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube, batched
along BOTH 'time' (in exactly time_chunk-sized groups) and spatial tiles,
instead of deep_copy_cube.py's time-only batching at full spatial extent.

Why: deep_copy_cube.py's batch_size (default 2000 layers, full extent) is
far smaller than the output store's fixed TIME_CHUNK_VALUE (20000), so every
batch after the first forces a full decompress/merge/recompress of the
entire still-open time-chunk across every spatial chunk/shard/variable --
quadratic write amplification, root-caused in
src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md. Matching
batch_size to time_chunk (the wiki's preferred fix) eliminates that, but
needs ~256 GiB RAM at full spatial extent. This script gets the same
elimination on a much smaller instance by shrinking the *spatial* extent
per batch instead of the *time* extent: every (tile, time-chunk) batch
writes a complete, final chunk/shard exactly once.

KNOWN TRADEOFF -- read this before using this script for a production run:
each granule's virtual chunk (see virtual_itslive_cube_per_chunk.py) spans
the FULL spatial grid -- there is no source-side sub-tiling. So tiling the
*write* does not tile the *read*: every tile's isel()+load() still fetches
and decompresses each touched granule's full spatial extent, discarding the
unwanted pixels afterward. With N spatial tiles, expect roughly Nx more S3
GETs and decompression CPU time than a full-extent read strategy. This is a
deliberate, accepted tradeoff (write-amplification elimination vs. read
amplification) -- benchmark with --num-layers on a bounded slice before
relying on this for a full production run.

Usage example:
python src/deep_copy_cube_tiled.py \
   --input-store my_virtual_cube.icechunk \
   --output-store my_deep_copy_cube.zarr
"""
import logging
import sys
import time
import warnings
from datetime import datetime

import xarray as xr
from zarr.errors import UnstableSpecificationWarning

import utils
from itscube_types import CubeFormat
from deep_copy_cube import (
   TIME_CHUNK_VALUE,
   X_Y_CHUNK_VALUE,
   TIME_CHUNK_VALUE_1D,
   open_virtual_cube,
   split_vars_by_time,
   build_encoding,
   _reset_write_encoding,
   resolve_output_store,
   upload_local_staging_dir,
)

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Zarr V3 unstable string dtype warnings, same rationale as
# deep_copy_cube.py.
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Spatial tile size (pixels per axis) for batching 3D-variable writes.
# Derived so that a (tile, time_chunk=20000) batch's raw memory footprint
# (tile_pixels * TIME_CHUNK_VALUE * 25 bytes/pixel/layer, measured on the
# current 512x512/62-variable cube schema) is about the same as
# deep_copy_cube.py's current full-extent, batch_size=2000 batch
# (2000 * 262144px * 25B): tile_pixels ~= 2000*262144/20000 ~= 26214,
# tile side ~= 162px. Both production grids (512px@120m, 256px@240m) are
# powers of 2, and a tile must stay a multiple of the shard size (64px --
# xy_chunk=8 * XY_SHARD_MULTIPLIER=8, see below) so a tile boundary never
# splits a shard across two batches. The nearest such value close to 162 is
# 128 (256 overshoots to ~4x the target memory) -- giving 4x4=16 tiles on
# the 512-grid, 2x2=4 tiles on the 256-grid.
XY_TILE_VALUE = 128

# Recommended --xy-shard-multiplier for this script: shard = 64px
# (X_Y_CHUNK_VALUE=8 * 8), per the empirically-fastest-for-reads shard size
# found in src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md's
# multi-trial EC2 retest (Sept 1 2026). Distinct from deep_copy_cube.py's
# own XY_SHARD_MULTIPLIER=4 constant, which predates that finding and is
# left unchanged there (out of scope for this script).
XY_SHARD_MULTIPLIER = 8


def split_time_vars_by_rank(cube, time_vars):
   """Further split split_vars_by_time()'s time_vars into 3D (time,y,x) and
   1D (time,) groups.

   1D vars have no x/y dimension, so this script (which batches writes
   spatially) must write them once per time-chunk rather than once per
   (tile, time-chunk) pair, to avoid redundantly decompressing/recompressing
   their own single (time_chunk_1d,)-sized chunk on every tile.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   time_vars : list of str
      Output of split_vars_by_time()'s first return value.

   Returns
   -------
   tuple of (list of str, list of str)
      (vars_3d, vars_1d) data variable names.
   """
   vars_3d = [v for v in time_vars if len(cube[v].dims) == 3]
   vars_1d = [v for v in time_vars if len(cube[v].dims) != 3]
   return vars_3d, vars_1d


def compute_xy_tiles(cube, xy_tile_size, xy_chunk, xy_shard_multiplier=1):
   """Compute the spatial tiles to batch 3D-variable writes over.

   Each tile is xy_tile_size pixels per side, except for a ragged smaller
   final tile along either axis when the grid size doesn't divide evenly by
   xy_tile_size (allowed -- the write itself stays correct, just uneven).

   Warns (does not raise) if xy_tile_size is not a multiple of the
   effective shard size (xy_chunk * xy_shard_multiplier, or plain xy_chunk
   when xy_shard_multiplier=1): a tile straddling a chunk/shard boundary
   would make that chunk/shard get touched by more than one tile-batch,
   reintroducing the same decompress/recompress write amplification this
   batching scheme exists to eliminate (see
   src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md).

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube (only its 'x'/'y' sizes are used).
   xy_tile_size : int
      Tile size in pixels, per spatial axis.
   xy_chunk : int
      Chunk size along x/y for 3D variables (see build_encoding).
   xy_shard_multiplier : int
      Same meaning as build_encoding's parameter; used here only to compute
      the alignment-warning threshold.

   Returns
   -------
   list of tuple of (slice, slice)
      (y_slice, x_slice) pairs, row-major, covering the full y/x extent.
   """
   shard_size = xy_chunk * xy_shard_multiplier
   if xy_tile_size % shard_size:
      logging.warning(
         f"xy_tile_size={xy_tile_size} is not a multiple of the effective "
         f"shard size {shard_size} (xy_chunk={xy_chunk} * "
         f"xy_shard_multiplier={xy_shard_multiplier}); a tile boundary may "
         f"split a chunk/shard across two batches, reintroducing write "
         f"amplification for that chunk/shard"
      )

   y_size = cube.sizes[utils.Coords.Y]
   x_size = cube.sizes[utils.Coords.X]

   y_starts = range(0, y_size, xy_tile_size)
   x_starts = range(0, x_size, xy_tile_size)

   tiles = []
   for y_start in y_starts:
      y_slice = slice(y_start, min(y_start + xy_tile_size, y_size))
      for x_start in x_starts:
         x_slice = slice(x_start, min(x_start + xy_tile_size, x_size))
         tiles.append((y_slice, x_slice))

   return tiles


def deep_copy_cube_tiled(
   input_store,
   output_store,
   bucket_prefix,
   time_chunk,
   xy_chunk,
   time_chunk_1d,
   xy_tile_size,
   xy_shard_multiplier=1,
   local_staging_dir=None,
   keep_local_staging=False,
   num_layers=0
):
   """Materialize a virtual datacube into a real zarr v3 datacube, batched
   along both 'time' (in exactly time_chunk-sized groups) and spatial tiles,
   so every write is a complete, aligned zarr chunk/shard written exactly
   once -- see this module's docstring for the write-vs-read amplification
   tradeoff this makes.

   Unlike deep_copy_cube.py's incremental append-based construction, this
   writes the whole store's shape/dtype/chunk-grid up front
   (mode='w', compute=False) and fills it in via region writes (mode='r+')
   per (tile, time-chunk). Static 2D (y,x) vars and 1D (time,) vars are
   excluded from the spatial tile loop (see split_time_vars_by_rank) and
   written once each, to avoid redundant partial-chunk rewrites of their
   own (much smaller) single chunks.

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
      materialized per time-chunk batch.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables.
   xy_tile_size : int
      Spatial tile size in pixels per axis for batching 3D-variable writes.
      Should be a multiple of xy_chunk * xy_shard_multiplier (warned, not
      enforced, by compute_xy_tiles) to avoid splitting a chunk/shard across
      two tile-batches.
   xy_shard_multiplier : int
      Must be >= 1; 1 (the default) leaves the store unsharded. See
      XY_SHARD_MULTIPLIER for the recommended value to pass explicitly.
   local_staging_dir : str, optional
      If set, `output_store` must be an s3:// path. All batches are written
      to this local directory first, and the whole store is uploaded to
      `output_store` with a single `aws s3 cp --recursive` after the last
      batch. See deep_copy_cube.py's parameter of the same name.
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
   tiles = compute_xy_tiles(cube, xy_tile_size, xy_chunk, xy_shard_multiplier)
   logging.info(
      f'Batching over {len(tiles)} spatial tile(s) '
      f'({xy_tile_size}px per side) x '
      f'{-(-total_layers // time_chunk)} time-chunk(s) ({time_chunk} layers each)'
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
   # variable (all data variables here, since open_virtual_cube's
   # xr.open_zarr defaults to chunks="auto"); 'time'/'x'/'y' coordinates are
   # eagerly-loaded index variables regardless of chunks=, so they get
   # written for real by this call -- needed before any region write below.
   # mode='r+' (used by every later write) requires every variable to
   # already exist, so this template must declare all of them (time_vars +
   # static_vars) up front, sliced to total_layers along 'time' to honor
   # --num-layers.
   template = xr.merge([
      cube[time_vars].isel({utils.Coords.TIME: slice(0, total_layers)}),
      cube[static_vars]
   ])
   _reset_write_encoding(template)
   template.to_zarr(
      write_target,
      mode='w',
      compute=False,
      encoding=encoding,
      zarr_format=3,
      consolidated=True
   )
   logging.info(f'Created template store at {write_target}')

   # Static 2D (y,x) vars: written once, full extent, no tiling -- avoids
   # partial-chunk rewrites of their single full-extent chunk.
   static_batch = cube[static_vars].load()
   _reset_write_encoding(static_batch)
   static_batch.to_zarr(write_target, mode='r+', zarr_format=3, consolidated=True)
   logging.info(f'Wrote {len(static_vars)} static variable(s) to {write_target}')

   for start in range(0, total_layers, time_chunk):
      stop = min(start + time_chunk, total_layers)
      logging.info(f'Materializing time-chunk {start}:{stop} of {total_layers}')

      # 1D (time,) vars: once per time-chunk, not tiled.
      batch_1d = cube[vars_1d].isel(
         {utils.Coords.TIME: slice(start, stop)}
      ).drop_vars(
         [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X], errors='ignore'
      ).load()
      _reset_write_encoding(batch_1d)
      batch_1d.to_zarr(
         write_target,
         mode='r+',
         region={utils.Coords.TIME: slice(start, stop)},
         zarr_format=3,
         consolidated=True
      )

      # 3D (time,y,x) vars: inner tile loop.
      for y_slice, x_slice in tiles:
         tile_batch = cube[vars_3d].isel({
            utils.Coords.TIME: slice(start, stop),
            utils.Coords.Y: y_slice,
            utils.Coords.X: x_slice,
         }).drop_vars(
            [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X], errors='ignore'
         ).load()
         _reset_write_encoding(tile_batch)
         tile_batch.to_zarr(
            write_target,
            mode='r+',
            region={
               utils.Coords.TIME: slice(start, stop),
               utils.Coords.Y: y_slice,
               utils.Coords.X: x_slice,
            },
            zarr_format=3,
            consolidated=True
         )

      logging.info(f'Wrote time-chunk {start}:{stop} of {total_layers} to {write_target}')

   if local_staging_dir:
      upload_local_staging_dir(local_staging_dir, output_store, keep_local_staging)

   logging.info(f'Done: deep-copied {total_layers} layers to {output_store}')


if __name__ == '__main__':
   import argparse

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Materialize a virtual ITS_LIVE datacube (icechunk repo built by
      virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube,
      batched along both 'time' (in full time_chunk-sized groups) and
      spatial tiles, to avoid deep_copy_cube.py's batch_size-vs-time_chunk
      write amplification (see this module's docstring for the accepted
      read-amplification tradeoff this makes instead).

      Usage example:
      python src/deep_copy_cube_tiled.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr

      # Recommended production shard size (shard=64px):
      python src/deep_copy_cube_tiled.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr \
         --xy-shard-multiplier 8

      # Stage locally, then upload the whole store to S3 in one shot:
      python src/deep_copy_cube_tiled.py \
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
         'number of layers materialized per time-chunk batch [%(default)d].'
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
      help='Chunk size for 1D (time,) variables [%(default)d].'
   )
   parser.add_argument(
      '--xy-tile-value',
      type=int,
      default=XY_TILE_VALUE,
      help='Spatial tile size in pixels per axis for batching 3D-variable '
         'writes [%(default)d].'
   )
   parser.add_argument(
      '--xy-shard-multiplier',
      type=int,
      default=XY_SHARD_MULTIPLIER,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables. A value of 1 '
         f'disables sharding. Recommended value once sharding '
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

   deep_copy_cube_tiled(
      args.input_store,
      args.output_store,
      args.bucket,
      args.time_chunk_value,
      args.xy_chunk_value,
      args.time_chunk_value_1d,
      args.xy_tile_value,
      args.xy_shard_multiplier,
      args.local_staging_dir,
      args.keep_local_staging,
      args.num_layers
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
