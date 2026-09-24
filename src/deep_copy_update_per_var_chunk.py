"""
Update an existing deep-copy datacube (deep_copy_cube_per_var_chunk.py) with
new layers appended to its source virtual cube. Layers are matched by
position: everything at index >= the store's current 'time' length is
treated as new, assuming the virtual cube is only ever appended to (matching
virtual_itslive_cube_per_chunk_update.py's own model). Known limitation: if
the creation run capped layers below the source's true count at the time
(--num-layers), that gap is indistinguishable from a real append.

No separate backup step: deep_copy_cube_per_var_chunk.py already uploads one
whole zarr time-chunk at a time, so a failed chunk write just leaves its
`.done` marker unset and a retry redoes only that chunk -- the old S3 chunk
is never overwritten until its replacement fully lands.

--progress-dir must be the SAME directory the creation run used: this update
reads that run's recorded chunking parameters from its run_config.json
(they can't be supplied directly -- they must match the build exactly), and
keeps its own progress markers in a sub-path namespaced by the (old, new)
layer-count transition.

Every time-indexed array is resized to the new layer count up front, in one
pass, before any chunk is (re)written -- mirrors creation's own atomicity of
declaring full shape before any pixel data (resize() is metadata-only, so
cheap and safe to re-run). If the old length wasn't an exact multiple of the
chunk size, the boundary chunk's real bytes are merged in from S3 first --
otherwise resizing would expose zarr's on-disk fill-value padding of that
ragged chunk as if it were legitimate old data (see
_prepare_array_for_update()).

The store's own declared shape is NOT a reliable "is it done" signal once an
update has started: _upload_chunk() re-uploads the *root* zarr.json (which
carries every array's shape) after the very first chunk of ANY variable
lands, so the store looks fully updated the moment one variable starts,
regardless of the rest. _find_incomplete_update() detects an interrupted
attempt and resumes it from its recorded old_total_layers instead of
trusting that shape.

Usage example:
python src/deep_copy_update_per_var_chunk.py \
   --input-store my_virtual_cube.icechunk \
   --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
   --local-staging-dir /local/scratch/my_deep_copy_cube.zarr \
   --progress-dir s3://its-live-data/path/to/deep_copy_progress
"""
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import zarr
from xarray.coding.times import encode_cf_datetime
from zarr.errors import UnstableSpecificationWarning

import itslive_utils
import utils
from itscube_types import CubeFormat
from deep_copy_cube import (
   open_virtual_cube,
   split_vars_by_time,
   build_encoding,
   resolve_output_store,
   split_time_vars_by_rank
)
from deep_copy_cube_per_var_chunk import (
   _compute_radar_mask,
   _upload_chunk,
   _write_var_3d_and_upload,
   _write_var_1d_and_upload,
)
from deep_copy_update import verify_output_store_exists, get_current_num_layers
from deep_copy_cube_progress import _Progress, SUCCESS_MARKER

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


# Suppress Zarr V3 unstable string dtype warnings, same rationale as
# deep_copy_cube.py.
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)


def _download_store_skeleton(output_store, local_store):
   """Download output_store's metadata skeleton (every zarr.json, no
   chunk/shard payloads) into local_store, so later steps have something to
   open in 'r+' mode before any pixel data is staged.

   Args:
      output_store (str): existing s3:// deep-copy store to update.
      local_store (str): local directory to download the skeleton into.
   """
   command_line = [
      "aws", "s3", "cp", "--recursive",
      output_store, local_store,
      "--exclude", "*/c/*",
   ]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


def _upload_var_metadata(local_store, output_store, var_name):
   """Re-upload one variable's (or 'time's) own zarr.json -- unlike
   creation, this update's resize() changes it (the 'shape' field), and
   _upload_chunk() only ever re-uploads the *root* zarr.json.

   Args:
      local_store (str): staging store path.
      output_store (str): final store path.
      var_name (str): variable (or 'time') whose array was just resized.
   """
   var_meta_path = os.path.join(local_store, var_name, 'zarr.json')
   _s3_copy_file(var_meta_path, f'{output_store.rstrip("/")}/{var_name}/zarr.json')


def _s3_copy_file(local_path, s3_path):
   """Upload a single file to S3 (retried, ACL-setting AWS CLI), matching
   deep_copy_cube_progress._s3_copy()'s mechanism -- reimplemented here so
   this module doesn't depend on its internal helper name.

   Args:
      local_path (str): source file.
      s3_path (str): destination s3:// URL.
   """
   command_line = [
      "aws", "s3", "cp", local_path, s3_path,
      "--acl", "bucket-owner-full-control"
   ]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


def _download_boundary_chunk_if_present(s3, output_store, local_store, var_name, chunk_index, is_dir):
   """Download one variable's boundary time-chunk from S3 into local
   staging, if present -- mirrors _upload_chunk()'s path scheme in reverse.
   Absence is legitimate (e.g. a RADAR_ONLY_VARS chunk the creation run
   radar-skipped), not an error: the old portion already reads as
   fill-value and will continue to once resized.

   Args:
      s3 (s3fs.S3FileSystem): used only to check existence.
      output_store (str): final store path.
      local_store (str): staging store path.
      var_name (str): variable name, or 'time'.
      chunk_index (int): this boundary chunk's index along 'time'.
      is_dir (bool): True for a 3D variable's chunk (a directory of spatial
         shards), False for a 1D/'time' chunk (a single file).
   """
   chunk_dir = f'{var_name}/c/{chunk_index}'
   s3_chunk_path = f'{output_store.rstrip("/")}/{chunk_dir}'

   if not s3.exists(s3_chunk_path):
      logging.info(f'{s3_chunk_path} does not exist (never written); nothing to merge')
      return

   local_chunk_path = os.path.join(local_store, chunk_dir)
   command_line = ["aws", "s3", "cp"]
   if is_dir:
      command_line.append("--recursive")
   command_line += [s3_chunk_path, local_chunk_path]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())
   logging.info(f'Merged existing boundary chunk {s3_chunk_path} into {local_chunk_path}')


def _prepare_array_for_update(
   s3, output_store, local_store, var_name, chunk_size, old_total_layers,
   new_total_layers, extra_dims_sizes, is_dir
):
   """Resize one time-indexed array to new_total_layers, re-upload its
   metadata, and -- if old_total_layers wasn't an exact multiple of
   chunk_size -- merge in the existing boundary chunk's real data first
   (see module docstring).

   Args:
      s3 (s3fs.S3FileSystem): used only to check existence.
      output_store (str): final store path.
      local_store (str): staging store path.
      var_name (str): variable name, or 'time'.
      chunk_size (int): this array's 'time' chunk size (time_chunk for a 3D
         variable, time_chunk_1d for a 1D one or 'time').
      old_total_layers (int): store's layer count before this update.
      new_total_layers (int): store's layer count after this update.
      extra_dims_sizes (tuple[int]): array's dims after 'time' -- () for
         1D/'time', (y_size, x_size) for a 3D variable.
      is_dir (bool): see _download_boundary_chunk_if_present().
   """
   group = zarr.open_group(local_store, mode='r+', zarr_format=3)
   array = group[var_name]
   array.resize((new_total_layers,) + extra_dims_sizes)
   _upload_var_metadata(local_store, output_store, var_name)
   logging.info(f'Resized {var_name} to {new_total_layers} layers and re-uploaded its metadata')

   if old_total_layers % chunk_size != 0:
      boundary_chunk_index = old_total_layers // chunk_size
      _download_boundary_chunk_if_present(
         s3, output_store, local_store, var_name, boundary_chunk_index, is_dir
      )


def _write_time_coord_and_upload(
   cube, local_store, output_store, chunk_size, total_layers,
   start_layer=0, progress=None
):
   """Write newly appended 'time' values into local staging, uploading each
   chunk to S3 as soon as it's done -- same resumable shape as
   _write_var_1d_and_upload(), but for 'time' itself.

   Can't reuse _write_var_1d_and_upload(): its _load_batch() drops 'time'
   from the selection (correct for real data variables, but drops the very
   value being written for var_name='time'), and a region write of a
   coordinate-only Dataset is a silent no-op in xarray's zarr backend
   (verified). Instead writes CF-encoded values directly via the zarr array
   API, matching what to_zarr()'s own encoder would have produced -- 'time'
   is always eagerly loaded already, so there's no S3 batch to fetch.

   Args:
      cube (xr.Dataset): the virtual datacube.
      local_store (str): staging store path.
      output_store (str): final store path.
      chunk_size (int): 'time' chunk size (time_chunk_1d).
      total_layers (int): store's new total length.
      start_layer (int): first layer to actually (re)write -- same
         semantics as _write_var_3d_and_upload()'s parameter of the same
         name (deep_copy_cube_per_var_chunk.py); 0 writes every chunk from
         scratch.
      progress (_Progress, optional): enables resumability.
   """
   var_name = utils.Coords.TIME
   if progress is not None and progress.var_is_done(var_name):
      logging.info(f'{var_name}: already fully processed in a previous attempt, skipping')
      return

   time_values = cube[var_name].values
   target = zarr.open_group(local_store, mode='r+', zarr_format=3)[var_name]
   units = target.attrs['units']
   calendar = target.attrs['calendar']

   num_chunks = 0
   num_resumed = 0

   first_chunk_start = (start_layer // chunk_size) * chunk_size
   for chunk_start in range(first_chunk_start, total_layers, chunk_size):
      chunk_stop = min(chunk_start + chunk_size, total_layers)
      chunk_index = chunk_start // chunk_size
      write_start = max(chunk_start, start_layer)
      num_chunks += 1

      if progress is not None and progress.chunk_is_done(var_name, chunk_index):
         logging.info(
            f'Skipping {var_name} chunk {chunk_start}:{chunk_stop} of '
            f'{total_layers} (already done in a previous attempt)'
         )
         num_resumed += 1
         continue

      encoded, _, _ = encode_cf_datetime(
         time_values[write_start:chunk_stop], units=units, calendar=calendar
      )
      target[write_start:chunk_stop] = encoded
      logging.info(f'Wrote {var_name} layers {write_start}:{chunk_stop} of {total_layers} to {local_store}')

      _upload_chunk(local_store, output_store, var_name, chunk_index)
      logging.info(
         f'Uploaded {var_name} chunk {chunk_start}:{chunk_stop} of '
         f'{total_layers} to {output_store}'
      )
      if progress is not None:
         progress.mark_chunk_done(var_name, chunk_index)

   if num_resumed:
      logging.info(
         f'{var_name}: resumed past {num_resumed} of {num_chunks} chunk(s) '
         '(already done in a previous attempt)'
      )

   if progress is not None:
      progress.mark_var_done(var_name)


_UPDATE_CONFIG_KEYS = (
   'output_store', 'old_total_layers', 'new_total_layers',
   'time_chunk', 'xy_chunk', 'time_chunk_1d', 'xy_shard_multiplier',
)


def _build_update_config(output_store, old_total_layers, new_total_layers, creation_config):
   """Assemble this update transition's config dict, for validate/write.

   Args:
      output_store (str): existing s3:// deep-copy store to update.
      old_total_layers (int): store's layer count before this update.
      new_total_layers (int): store's layer count after this update.
      creation_config (dict): creation run's own recorded chunking params.
   """
   return {
      'output_store': output_store,
      'old_total_layers': old_total_layers,
      'new_total_layers': new_total_layers,
      'time_chunk': creation_config['time_chunk'],
      'xy_chunk': creation_config['xy_chunk'],
      'time_chunk_1d': creation_config['time_chunk_1d'],
      'xy_shard_multiplier': creation_config['xy_shard_multiplier'],
   }


def _validate_update_config(recorded, current, update_progress):
   """Hard-fail on any mismatch between this attempt's config and a prior
   attempt's recorded one for the same (old_total_layers, new_total_layers)
   transition. Unlike creation's validate_config(), there's no grow/shrink/
   clamp case here: old/new_total_layers are baked into update_progress's
   own base path, so any mismatch at all means the markers underneath no
   longer mean what they claim.

   Args:
      recorded (dict): prior attempt's persisted config.
      current (dict): this attempt's config, from _build_update_config().
      update_progress (_Progress): used only to name the config path in
         the error message.
   """
   mismatched = [
      (key, recorded.get(key), current.get(key))
      for key in _UPDATE_CONFIG_KEYS
      if recorded.get(key) != current.get(key)
   ]
   if mismatched:
      details = '\n'.join(
         f'  {key}: recorded {r!r}, this attempt {c!r}' for key, r, c in mismatched
      )
      raise RuntimeError(
         f"This update attempt's parameters disagree with those a prior "
         f"attempt recorded at {update_progress._config_path()} for the "
         f"same layer range:\n{details}\n"
         f"Delete {update_progress.base} to restart this update range from "
         f"scratch, or investigate why the parameters differ."
      )


def _find_incomplete_update(creation_progress):
   """Scan {creation_progress.base}/updates/ for a transition directory left
   by an interrupted attempt (has run_config.json but no _SUCCESS) -- see
   module docstring for why the store's own shape can't be trusted to tell
   "done" apart from "interrupted".

   Raises RuntimeError if more than one incomplete transition is found --
   at most one update should ever be in flight at a time.

   Args:
      creation_progress (_Progress): rooted at the creation run's own
         progress directory (not namespaced by any transition).

   Returns:
      tuple of (int, int, _Progress, dict) or None: (old_total_layers,
         new_total_layers, update_progress, recorded_update_config) for the
         one incomplete transition found, or None if updates/ is missing or
         empty (the normal case: a completed transition's directory is
         removed entirely via remove_entirely() rather than kept around,
         unless --keep-progress-markers was used).
   """
   updates_root = f'{creation_progress.base}/updates'
   if not creation_progress.s3.exists(updates_root):
      return None

   incomplete = []
   for entry in creation_progress.s3.ls(updates_root):
      name = os.path.basename(entry.rstrip('/'))
      parts = name.split('_to_')
      if len(parts) != 2 or not all(part.isdigit() for part in parts):
         continue

      candidate = _Progress(creation_progress.s3, f'{updates_root}/{name}')
      if candidate.is_complete():
         continue

      config = candidate.read_config()
      if config is None:
         # write_config() hadn't landed yet either; nothing to resume from.
         continue

      incomplete.append((int(parts[0]), int(parts[1]), candidate, config))

   if not incomplete:
      return None

   if len(incomplete) > 1:
      found = [f'{old}_to_{new}' for old, new, _, _ in incomplete]
      raise RuntimeError(
         f'Found {len(incomplete)} incomplete update transitions under '
         f'{updates_root}: {found}. At most one update should ever be in '
         f'flight at a time -- investigate manually before rerunning.'
      )

   return incomplete[0]


def deep_copy_update_per_var_chunk(
   input_store,
   output_store,
   bucket_prefix,
   progress_dir,
   local_staging_dir,
   keep_local_staging=False,
   num_load_workers=None,
   keep_progress_markers=False,
):
   """Append any layers new to the virtual cube onto an existing deep-copy
   store, resumably and without a separate backup step (see module
   docstring).

   Args:
      input_store (str): virtual cube's icechunk repo (s3:// or local).
      output_store (str): existing deep-copy store to update.
      bucket_prefix (str): S3 prefix the virtual cube's granule references
         resolve against.
      progress_dir (str): the creation run's own --progress-dir -- this
         update's chunking parameters are read from there rather than
         supplied directly, and its own progress markers live in a
         sub-path underneath it.
      local_staging_dir (str): local directory for the skeleton download
         and every write, before syncing back to output_store.
      keep_local_staging (bool): keep local_staging_dir after success
         instead of deleting it.
      num_load_workers (int, optional): thread-pool size per batch
         .load() call (None leaves dask's own default).
      keep_progress_markers (bool): keep this update's own progress
         markers instead of removing them on success.
   """
   if not progress_dir or not progress_dir.startswith(utils.S3_PREFIX):
      raise ValueError(
         f"--progress-dir must be an s3:// path recorded by the creation "
         f"run (deep_copy_cube_per_var_chunk.py --progress-dir) -- this "
         f"update reuses that run's run_config.json for its chunking "
         f"parameters, got {progress_dir!r}"
      )

   verify_output_store_exists(output_store)

   creation_progress = _Progress.create(progress_dir, output_store)
   creation_config = creation_progress.read_config()
   if creation_config is None:
      raise RuntimeError(
         f'No run_config.json found at {creation_progress._config_path()} '
         f'-- the creation run for {output_store} must have used '
         f'--progress-dir (deep_copy_cube_per_var_chunk.py) for this update '
         f'to know the original chunking parameters (time_chunk, xy_chunk, '
         f'time_chunk_1d, xy_shard_multiplier). Rerun creation with '
         f'--progress-dir pointed at {progress_dir} if this store predates it.'
      )

   time_chunk = creation_config['time_chunk']
   xy_chunk = creation_config['xy_chunk']
   time_chunk_1d = creation_config['time_chunk_1d']
   xy_shard_multiplier = creation_config['xy_shard_multiplier']

   cube = open_virtual_cube(input_store, bucket_prefix)
   new_total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {new_total_layers} layers')

   incomplete = _find_incomplete_update(creation_progress)
   if incomplete is not None:
      old_total_layers, target_total_layers, update_progress, recorded_update_config = incomplete
      logging.info(
         f'Found an incomplete update at {update_progress.base} '
         f'({old_total_layers} -> {target_total_layers} layers) left by a '
         f'previous interrupted attempt; resuming/repairing it before '
         f'considering any further layers'
      )
      if new_total_layers < target_total_layers:
         raise RuntimeError(
            f'The virtual cube now has only {new_total_layers} layer(s), '
            f'fewer than the {target_total_layers} an interrupted update at '
            f'{update_progress.base} was already targeting -- delete that '
            f'directory and investigate before rerunning.'
         )
      if new_total_layers > target_total_layers:
         logging.warning(
            f'Virtual cube now has {new_total_layers} layers, more than '
            f'the {target_total_layers} this interrupted update was '
            f'targeting; finishing that transition first. Rerun this '
            f'script afterward to pick up the remaining layers.'
         )
      new_total_layers = target_total_layers
      current_update_config = _build_update_config(
         output_store, old_total_layers, new_total_layers, creation_config
      )
      _validate_update_config(recorded_update_config, current_update_config, update_progress)
   else:
      old_total_layers = get_current_num_layers(output_store)
      logging.info(f'{output_store} currently has {old_total_layers} layers')

      if new_total_layers <= old_total_layers:
         logging.info(
            f'{output_store} is already up to date '
            f'({old_total_layers} layers, virtual cube has {new_total_layers})'
         )
         return

      update_progress = _Progress(
         creation_progress.s3, f'{creation_progress.base}/updates/{old_total_layers}_to_{new_total_layers}'
      )
      current_update_config = _build_update_config(
         output_store, old_total_layers, new_total_layers, creation_config
      )
      recorded_update_config = update_progress.read_config()
      if recorded_update_config is None:
         update_progress.write_config(current_update_config)
      else:
         _validate_update_config(recorded_update_config, current_update_config, update_progress)

   time_vars, _ = split_vars_by_time(cube)
   vars_3d, vars_1d = split_time_vars_by_rank(cube, time_vars)
   logging.info(
      f"Updating {len(vars_1d)} 1D variable(s), {len(vars_3d)} 3D "
      f"variable(s), and the '{utils.Coords.TIME}' coordinate from "
      f'{old_total_layers} to {new_total_layers} layers'
   )

   is_radar = _compute_radar_mask(cube, new_total_layers)
   encoding = build_encoding(cube, time_chunk, xy_chunk, time_chunk_1d, xy_shard_multiplier)

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   local_store = resolve_output_store(local_staging_dir)
   _download_store_skeleton(output_store, local_store)
   logging.info(f'Downloaded metadata skeleton from {output_store} to {local_store}')

   # Set directly rather than via to_zarr(): region writes never touch
   # root-group attrs (verified). _upload_chunk() re-uploads the root
   # zarr.json on the very first chunk below regardless.
   zarr.open_group(local_store, mode='r+', zarr_format=3).attrs[CubeFormat.date_updated] = (
      cube.attrs[CubeFormat.date_updated]
   )

   y_size = cube.sizes[utils.Coords.Y]
   x_size = cube.sizes[utils.Coords.X]

   # Resize/merge every time-indexed array up front, before any chunk is
   # rewritten -- mirrors creation's atomicity, minimizing the window where
   # arrays disagree on 'time' length.
   _prepare_array_for_update(
      creation_progress.s3, output_store, local_store, utils.Coords.TIME, time_chunk_1d,
      old_total_layers, new_total_layers, (), is_dir=False
   )
   for var_name in vars_1d:
      _prepare_array_for_update(
         creation_progress.s3, output_store, local_store, var_name, time_chunk_1d,
         old_total_layers, new_total_layers, (), is_dir=False
      )
   for var_name in vars_3d:
      _prepare_array_for_update(
         creation_progress.s3, output_store, local_store, var_name, time_chunk,
         old_total_layers, new_total_layers, (y_size, x_size), is_dir=True
      )

   _write_time_coord_and_upload(
      cube, local_store, output_store, time_chunk_1d, new_total_layers,
      old_total_layers, update_progress
   )
   for var_name in vars_1d:
      _write_var_1d_and_upload(
         cube, local_store, output_store, var_name, time_chunk_1d,
         new_total_layers, num_load_workers, update_progress,
         start_layer=old_total_layers
      )
   for var_name in vars_3d:
      _write_var_3d_and_upload(
         cube, local_store, output_store, var_name, time_chunk, new_total_layers,
         encoding.get(var_name, {}).get(utils.OutputFormat.fill_value),
         is_radar, num_load_workers, update_progress,
         start_layer=old_total_layers
      )

   update_progress.mark_complete()
   logging.info(f'Marked update complete ({update_progress.base}/{SUCCESS_MARKER})')

   if keep_progress_markers:
      logging.info(f'Keeping every progress marker under {update_progress.base}')
   else:
      update_progress.remove_entirely([utils.Coords.TIME] + vars_1d + vars_3d)

   if keep_local_staging:
      logging.info(f'Keeping local staging directory {local_store}')
   else:
      logging.info(f'Removing local staging directory {local_store}')
      shutil.rmtree(local_store)

   logging.info(
      f'Done: updated {output_store} from {old_total_layers} to '
      f'{new_total_layers} layers'
   )


if __name__ == '__main__':
   import argparse

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Append new layers from a virtual ITS_LIVE datacube (icechunk repo,
      built by virtual_itslive_cube_per_chunk.py) onto an existing deep-copy
      Zarr v3 datacube built by deep_copy_cube_per_var_chunk.py. Any layer in
      the virtual cube at index >= the deep-copy cube's current length is
      treated as new and appended. See this module's docstring for the
      rationale and the resumability model.

      Usage example:
      python src/deep_copy_update_per_var_chunk.py \
         --input-store my_virtual_cube.icechunk \
         --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
         --local-staging-dir /local/scratch/my_deep_copy_cube.zarr \
         --progress-dir s3://its-live-data/path/to/deep_copy_progress
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
      help='s3:// URL of the existing deep-copy Zarr v3 store to update.'
   )
   parser.add_argument(
      '--bucket',
      type=str,
      default='s3://its-live-data/',
      help='S3 URL prefix the virtual chunk container resolves granule '
         'references against [%(default)s]'
   )
   parser.add_argument(
      '--progress-dir',
      type=str,
      required=True,
      help='s3:// directory the creation run (deep_copy_cube_per_var_chunk.py '
         '--progress-dir) recorded its run_config.json under. This update '
         'reads the store\'s chunking parameters from there and keeps its '
         'own progress markers in a sub-path underneath it.'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      required=True,
      help='Local directory the store\'s metadata skeleton is downloaded '
         'into, and every write lands in, before being synced back to '
         '--output-store. Required.'
   )
   parser.add_argument(
      '--keep-local-staging',
      action='store_true',
      help='Keep --local-staging-dir after a successful run instead of '
         'deleting it [%(default)s].'
   )
   parser.add_argument(
      '--num-load-workers',
      type=int,
      default=None,
      help='Thread-pool size for each batch .load() call. Unset (the '
         'default) leaves dask\'s own default in effect (CPU core count).'
   )
   parser.add_argument(
      '--keep-progress-markers',
      action='store_true',
      help='Keep every per-variable/per-chunk progress marker for this '
         'update after a successful run instead of pruning them down to '
         'just _SUCCESS and run_config.json [%(default)s].'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   deep_copy_update_per_var_chunk(
      args.input_store,
      args.output_store,
      args.bucket,
      args.progress_dir,
      args.local_staging_dir,
      args.keep_local_staging,
      args.num_load_workers,
      args.keep_progress_markers,
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
