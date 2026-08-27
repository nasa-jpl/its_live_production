"""
Update an existing deep-copy ITS_LIVE datacube (plain Zarr store, built by
deep_copy_cube.py) with new layers from its source virtual datacube (icechunk
repo, built by virtual_itslive_cube_per_chunk.py).

Layers are matched by position, not granule identity: the deep-copy cube's
current length along 'time' is read from the output store, and every layer
in the virtual cube at index >= that length is treated as new and appended,
in the order it already has in the virtual cube. This assumes the virtual
cube is only ever appended to (never reordered or backfilled in the middle),
matching virtual_itslive_cube_per_chunk_update.py's own append-only update
model for the *source* icechunk repo.

Usage example:
python src/deep_copy_update.py \
   --input-store my_virtual_cube.icechunk \
   --output-store my_deep_copy_cube.zarr
"""
import logging
import os
import time
from datetime import datetime

import s3fs
import xarray as xr

import itslive_utils
import utils
from itscube_types import CubeFormat
from deep_copy_cube import (
   NUM_GRANULES_TO_WRITE,
   open_virtual_cube,
   split_vars_by_time,
   resolve_output_store,
   upload_local_staging_dir,
   _reset_write_encoding,
)

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


def verify_output_store_exists(output_store):
   """Fail fast, with a clear message, if the deep-copy store to update
   doesn't exist yet.

   Parameters
   ----------
   output_store : str
      Path to the existing deep-copy zarr store (s3:// or local).
   """
   if output_store.startswith(utils.S3_PREFIX):
      s3_path = output_store.replace(utils.S3_PREFIX, '', 1)
      s3 = s3fs.S3FileSystem()

      if not s3.exists(s3_path):
         raise RuntimeError(
            f"Output store {output_store} does not exist in S3; nothing to "
            "update. Use deep_copy_cube.py to create it first."
         )

   elif not os.path.exists(output_store):
      raise RuntimeError(
         f"Output store {output_store} does not exist locally; nothing to "
         "update. Use deep_copy_cube.py to create it first."
      )


def get_current_num_layers(output_store):
   """Read the number of layers the deep-copy store currently has along
   'time'.

   Parameters
   ----------
   output_store : str
      Path to the existing deep-copy zarr store (s3:// or local).

   Returns
   -------
   int
      Current size of the 'time' dimension.
   """
   with xr.open_zarr(output_store, consolidated=True) as ds:
      return ds.sizes[utils.Coords.TIME]


def deep_copy_update(
   input_store, output_store, bucket_prefix, batch_size,
   backup_store=None, local_staging_dir=None, keep_local_staging=False
):
   """Append any layers new to the virtual cube (index >= the deep-copy
   store's current length) onto an existing deep-copy zarr datacube, batched
   along 'time' to bound memory use.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      Path to the existing deep-copy zarr store to update (s3:// or local).
   bucket_prefix : str
      S3 URL prefix the virtual chunk container resolves granule references
      against (see deep_copy_cube.open_virtual_cube).
   batch_size : int
      Number of new layers to materialize and write per batch.
   backup_store : str, optional
      S3 path to back up `output_store`'s latest chunks/shards to before
      staging them locally. Required (and only meaningful) when
      `output_store` is on S3 -- appending means a full read-modify-write of
      every active chunk/shard, so a durable, independently-restorable copy
      is kept in case the final upload back to `output_store` fails partway
      (this has happened before in production; some cubes have hundreds of
      thousands to ~1M granules, making a from-scratch rebuild too costly to
      risk without one).
   local_staging_dir : str, optional
      Local directory to stage the backed-up latest chunks/shards in before
      appending, then upload back to `output_store` from. Required under
      the same condition as `backup_store`. Ignored (and unneeded) for a
      local `output_store`, where appends go directly to `output_store` as
      before -- a local chunk/shard read-modify-write is normal filesystem
      I/O; there's no network round-trip to economize on.
   keep_local_staging : bool
      If True, keep `local_staging_dir` after a successful upload instead of
      removing it. Ignored if local staging wasn't used for this run.
   """
   verify_output_store_exists(output_store)

   current_num_layers = get_current_num_layers(output_store)
   logging.info(f'Existing deep-copy cube {output_store} has {current_num_layers} layers')

   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   if total_layers <= current_num_layers:
      logging.info(
         f'{output_store} is already up to date '
         f'({current_num_layers} layers, virtual cube has {total_layers})'
      )
      return

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   # Static, cube-level variables (mapping/landice/floatingice) were already
   # written once when the deep-copy store was created; append_dim="time"
   # requires every variable in the write to carry the append dimension, so
   # only time-indexed variables are ever appended here.
   time_vars, _ = split_vars_by_time(cube)
   new_cube = cube[time_vars].isel(
      {utils.Coords.TIME: slice(current_num_layers, total_layers)}
   )
   num_new_layers = new_cube.sizes[utils.Coords.TIME]
   logging.info(
      f'Appending {num_new_layers} new layers '
      f'({current_num_layers}:{total_layers}) to {output_store}'
   )

   # Every append does a full read-modify-write of each active chunk/shard.
   # With time_chunk=20000 (see deep_copy_cube.TIME_CHUNK_VALUE), the active
   # time-chunk holds all/most of a cube's layers until it exceeds 20000, so
   # doing this directly against S3, once per batch, would repeatedly
   # re-download/re-upload the same (potentially large) chunk/shard files
   # over the network. A local output_store doesn't have this cost -- a
   # local read-modify-write is plain filesystem I/O -- so only S3 output
   # stages locally first.
   use_local_staging = str(output_store).startswith(utils.S3_PREFIX)

   # Fail loudly instead of silently ignoring these flags: passing either one
   # for a store that doesn't qualify for local staging (a local
   # output_store) previously fell straight through to a direct append with
   # no indication the flags had no effect.
   if (local_staging_dir or backup_store) and not use_local_staging:
      raise ValueError(
         f"--local-staging-dir/--backup-store were given but {output_store} "
         "does not qualify for local staging (requires an s3:// "
         "--output-store) -- these flags would otherwise be silently "
         "ignored. Omit them for this store."
      )

   if use_local_staging:
      if not local_staging_dir or not backup_store:
         raise ValueError(
            f"{output_store} is on S3: --local-staging-dir and "
            "--backup-store are both required so the active chunks/shards "
            "can be backed up and staged locally before appending (see "
            "deep_copy_update()'s docstring)."
         )

      logging.info(f'Backing up latest chunks/shards from {output_store} to {backup_store}')
      # backup_datacube_latest_shards() handles both sharded and unsharded
      # v3 stores (falls back from .shards to .chunks -- see
      # itslive_utils.identify_datacube_latest_shards()).
      itslive_utils.backup_datacube_latest_shards(output_store, backup_store)

      resolve_output_store(local_staging_dir)
      logging.info(f'Restoring backed-up shards from {backup_store} to {local_staging_dir}')
      command_line = ["aws", "s3", "cp", "--recursive", backup_store, local_staging_dir]
      itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())

      write_target = local_staging_dir
   else:
      write_target = output_store

   try:
      for batch_num, start in enumerate(range(0, num_new_layers, batch_size)):
         stop = min(start + batch_size, num_new_layers)
         logging.info(f'Materializing new layers {start}:{stop} of {num_new_layers}')

         batch = new_cube.isel({utils.Coords.TIME: slice(start, stop)}).load()

         # Clear fill attrs so the fill lives only in the store's existing
         # write encoding (see deep_copy_cube._reset_write_encoding); required
         # on every append batch, since the CF-encoder collision would
         # otherwise fire on each write.
         _reset_write_encoding(batch)

         batch.to_zarr(
            write_target,
            append_dim=utils.Coords.TIME,
            zarr_format=3,
            consolidated=True
         )

         logging.info(f'Appended layers {start}:{stop} of {num_new_layers} to {write_target}')

   except Exception:
      if use_local_staging:
         logging.error(
            f'Append failed -- leaving local staging directory '
            f'{local_staging_dir} (and backup at {backup_store}) in place '
            'for manual inspection/retry instead of cleaning them up.'
         )
      raise

   if use_local_staging:
      upload_local_staging_dir(local_staging_dir, output_store, keep_local_staging)

   logging.info(
      f'Done: appended {num_new_layers} new layers to {output_store} '
      f'(now {total_layers} total)'
   )


if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(
      description="""
      Append new layers from a virtual ITS_LIVE datacube (icechunk repo,
      built by virtual_itslive_cube_per_chunk.py) onto an existing deep-copy
      Zarr datacube (built by deep_copy_cube.py). Any layer in the virtual
      cube at index >= the deep-copy cube's current length is treated as new
      and appended.

      Usage example:
      python src/deep_copy_update.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr
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
      help='Path to the existing deep-copy Zarr store to update (s3:// or local).'
   )
   parser.add_argument(
      '--bucket',
      type=str,
      default='s3://its-live-data/',
      help='S3 URL prefix the virtual chunk container resolves granule '
         'references against [%(default)s]'
   )
   parser.add_argument(
      '-b', '--batch-size',
      type=int,
      default=NUM_GRANULES_TO_WRITE,
      help='Number of new layers to materialize and write per batch [%(default)d].'
   )
   parser.add_argument(
      '--backup-store',
      type=str,
      default=None,
      help='S3 path to back up --output-store\'s latest chunks/shards to '
         'before staging them locally. Required when --output-store is on '
         'S3; ignored otherwise.'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      default=None,
      help='Local directory to stage the backed-up latest chunks/shards in '
         'before appending, then upload back to --output-store from. '
         'Required under the same condition as --backup-store; ignored '
         'otherwise.'
   )
   parser.add_argument(
      '--keep-local-staging',
      action='store_true',
      help='Keep --local-staging-dir after a successful upload instead of '
         'deleting it [%(default)s].'
   )

   args = parser.parse_args()

   start_time = time.time()

   deep_copy_update(
      args.input_store,
      args.output_store,
      args.bucket,
      args.batch_size,
      args.backup_store,
      args.local_staging_dir,
      args.keep_local_staging
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
