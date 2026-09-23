"""
Update an existing deep-copy ITS_LIVE datacube built by
deep_copy_cube_per_var_chunk.py with new layers from its source virtual
datacube (icechunk repo, built by virtual_itslive_cube_per_chunk.py).

Layers are matched by position, not granule identity, when the deep-copy
store's current length along 'time' is read from the
output store, and every layer in the virtual cube at index >= that length is
treated as new and appended, in the order it already has in the virtual
cube. This assumes the virtual cube is only ever appended to, matching
virtual_itslive_cube_per_chunk_update.py's own append-only update model for
the *source* icechunk repo. See this module's "Known limitation" note below.

Unlike previous approach, there is no separate backup step here.
deep_copy_cube_per_var_chunk.py already uploads/marks progress one whole
zarr time-chunk at a time (never a bulk final upload), so the "old S3 chunk
stays valid until the new one successfully replaces it" guarantee already
holds without a backup: a failed upload of a chunk this update touches
simply leaves that chunk's `.done` marker unwritten, and a retry redoes
exactly that chunk from the same (still-present) S3 data plus the source
cube -- see deep_copy_cube_progress._Progress and this module's own
docstrings below for the mechanics.

--progress-dir must point at the SAME directory the creation run
(deep_copy_cube_per_var_chunk.py) used: this update reads that run's
recorded run_config.json to learn the store's chunking parameters
(time_chunk, xy_chunk, time_chunk_1d, xy_shard_multiplier), which must match
the original build exactly (they are not user-supplied here). This update's
own progress lives in a distinct sub-path under that same --progress-dir,
namespaced by the (old, new) layer-count pair it's updating between, so a
completed update's markers never collide with the creation run's or with a
later update's.

A store's Zarr v3 arrays are all resized to the new layer count up front, in
one pass, before any chunk gets (re)written -- mirroring
deep_copy_cube_per_var_chunk.py's own creation-time atomicity property of
declaring every array's full shape in a single call before filling in any
pixel data. zarr.Array.resize() is metadata-only (verified: it never touches
existing chunk/shard files), so this is cheap and, on its own, safe to
re-run if interrupted.

If the store's layer count wasn't an exact multiple of a chunk size, its
last ("boundary") zarr time-chunk was only partially filled by the
creation run (or a prior update). Resizing alone would silently expose that
chunk's already-on-disk-but-logically-unwritten padding (zarr pads a
partially written chunk out to its full declared size on disk, using the
array's fill_value) as if it were legitimate old data once the array grows
past it -- so before writing anything new into that chunk, this downloads
the chunk's existing real bytes from S3 into the local staging copy first,
so the read-modify-write cycle that follows only ever touches the genuinely
new portion. See _prepare_array_for_update()'s docstring.

Known limitation (identical to deep_copy_update.py's own): if the creation
run used --num-layers to deliberately cap below the source cube's true
count at that time, layers between that cap and the real historical count
are indistinguishable from genuinely new appended layers.

The store's own declared shape (what get_current_num_layers() reads) is NOT
a reliable signal of how much of an update actually finished, once an
update has started: _prepare_array_for_update() resizes every time-indexed
array to new_total_layers up front, before any new pixel data is written
for any of them, and _upload_chunk() unconditionally re-uploads the *root*
zarr.json (which carries every array's consolidated shape) after the very
first chunk of ANY variable lands. So as soon as one variable's first chunk
is uploaded, the store's consolidated metadata already claims
new_total_layers for every array, including ones untouched so far this
run. An interrupted attempt therefore leaves the store looking "already up
to date" to a naive shape comparison, even though some variables' new-layer
data was never written. _find_incomplete_update() is what lets a rerun
detect (and finish, or repair) that instead of silently no-op'ing -- see
its own docstring.

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
)
from deep_copy_cube_per_var import split_time_vars_by_rank
from deep_copy_cube_per_var_chunk import (
   _compute_radar_mask,
   _upload_chunk,
   _write_var_3d_and_upload,
   _write_var_1d_and_upload,
)
from deep_copy_update import verify_output_store_exists, get_current_num_layers
from deep_copy_cube_progress import _Progress, SUCCESS_MARKER

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


# Suppress Zarr V3 unstable string dtype warnings, same rationale as
# deep_copy_cube.py.
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)


def _download_store_skeleton(output_store, local_store):
   """Download the existing output store's metadata skeleton (root
   zarr.json + every array's own zarr.json + coordinate/static variable
   data) into `local_store`, excluding every chunk/shard payload directory
   -- cheap, and gives every step below something to open in 'r+' mode
   locally before any new pixel data is staged.

   Parameters
   ----------
   output_store : str
      Existing s3:// deep-copy store to update.
   local_store : str
      Local directory to download the skeleton into (already
      resolve_output_store()-cleaned).
   """
   command_line = [
      "aws", "s3", "cp", "--recursive",
      output_store, local_store,
      "--exclude", "*/c/*",
   ]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


def _upload_var_metadata(local_store, output_store, var_name):
   """Re-upload one variable's (or the 'time' coordinate's) own zarr.json.

   _upload_chunk() (deep_copy_cube_per_var_chunk.py) only ever re-uploads
   the *root* zarr.json after a chunk write -- fine for creation, where a
   variable's own metadata never changes after the initial template write,
   but this update's resize() call DOES change it (the array's 'shape'
   field), so that change needs its own upload step.

   Parameters
   ----------
   local_store : str
      Local path of the staging store.
   output_store : str
      Final s3:// destination.
   var_name : str
      Name of the variable (or 'time') whose array was just resized.
   """
   var_meta_path = os.path.join(local_store, var_name, 'zarr.json')
   _s3_copy_file(var_meta_path, f'{output_store.rstrip("/")}/{var_name}/zarr.json')


def _s3_copy_file(local_path, s3_path):
   """Upload a single local file to S3, matching
   deep_copy_cube_progress._s3_copy()'s mechanism (retried, ACL-setting AWS
   CLI invocation) -- imported here as a thin wrapper rather than reused
   directly so this module doesn't depend on deep_copy_cube_progress's
   internal helper name."""
   command_line = [
      "aws", "s3", "cp", local_path, s3_path,
      "--acl", "bucket-owner-full-control"
   ]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


def _download_boundary_chunk_if_present(s3, output_store, local_store, var_name, chunk_index, is_dir):
   """Download one variable's (or 'time' coordinate's) boundary zarr
   time-chunk from S3 into the local staging store, if it exists there --
   mirrors _upload_chunk()'s own path scheme in reverse.

   It may legitimately not exist: e.g. a RADAR_ONLY_VARS variable whose
   boundary chunk was radar-skipped (never written) by the creation run. In
   that case the old portion already reads as fill-value on S3 and will
   continue to once resized -- consistent with never having existed --
   so skipping the download is correct, not an error.

   Parameters
   ----------
   s3 : s3fs.S3FileSystem
      Used only to check existence.
   output_store : str
      Final s3:// destination.
   local_store : str
      Local path of the staging store.
   var_name : str
      Name of the variable or 'time'.
   chunk_index : int
      Index of the boundary chunk along 'time'.
   is_dir : bool
      True for a 3D variable's chunk path (a directory of spatial
      chunks/shards); False for a 1D variable's or 'time's chunk path (a
      single file) -- statically known from the variable's rank, not
      introspected, since a 1D array's chunk path never has anything nested
      under it.
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
   """Resize one time-indexed local array to `new_total_layers` along its
   first axis, re-upload its own zarr.json, and -- if the store's old
   length wasn't an exact multiple of `chunk_size` -- merge the existing
   boundary chunk's real data into the local store first (see this module's
   docstring).

   Parameters
   ----------
   s3 : s3fs.S3FileSystem
   output_store : str
      Final s3:// destination.
   local_store : str
      Local path of the staging store.
   var_name : str
      Name of the variable, or 'time' for the time coordinate.
   chunk_size : int
      Size of this array's zarr chunk along 'time' (time_chunk for a 3D
      variable, time_chunk_1d for a 1D variable or 'time').
   old_total_layers : int
      The store's layer count before this update.
   new_total_layers : int
      The store's layer count after this update.
   extra_dims_sizes : tuple of int
      Sizes of this array's dimensions after 'time' -- () for a 1D array or
      'time', (y_size, x_size) for a 3D variable.
   is_dir : bool
      See _download_boundary_chunk_if_present().
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
   """Write the 'time' coordinate's newly appended values into the local
   staging store, uploading each whole zarr chunk to S3 as soon as it's
   done -- the same resumable, whole-chunk-at-a-time shape as
   deep_copy_cube_per_var_chunk._write_var_1d_and_upload(), but for the
   'time' coordinate itself.

   Can't reuse _write_var_1d_and_upload()/its _load_batch() as-is for
   'time': _load_batch() selects `cube[[var_name]]` and then drops 'time'
   (along with 'y'/'x') from the result -- correct for every real data
   variable (none of which are named 'time'), but for var_name='time' that
   drops the very value being written, leaving nothing to write. A region
   write of a coordinate-only Dataset is also a silent no-op in xarray's
   zarr backend (verified directly against a real store): it only writes
   data variables within the region, never bare index coordinates, even
   though 'time' is squarely within the written region's dims.

   Instead, this writes the raw CF-encoded values directly via the zarr
   array API, using the units/calendar already recorded in the store's own
   (just-resized) 'time' array attrs -- the same values
   `xr.Dataset.to_zarr()`'s CF encoder would have produced, computed here via
   `xr.coding.times.encode_cf_datetime()` instead of going through a
   Dataset write. 'time' is always eagerly loaded already (see
   deep_copy_cube.open_virtual_cube()), so there's no batch to fetch from S3
   in the first place -- every value comes straight from `cube` in memory.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   local_store : str
      Local path of the staging store.
   output_store : str
      Final s3:// destination each chunk gets uploaded to as soon as it's
      done.
   chunk_size : int
      Chunk size for the 'time' coordinate (time_chunk_1d).
   total_layers : int
      Total number of layers ('time' values) the store now covers.
   start_layer : int, optional
      First layer to actually write -- see
      deep_copy_cube_per_var_chunk._write_var_3d_and_upload()'s
      `start_layer` parameter doc for the exact semantics. 0 (the default)
      writes every chunk from scratch.
   progress : _Progress, optional
      Enables resumability, same contract as
      deep_copy_cube_per_var_chunk._write_var_1d_and_upload()'s `progress`.
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
   """Hard-fail on any mismatch between this update attempt's parameters and
   those a prior attempt recorded for the same (old_total_layers,
   new_total_layers) transition.

   Unlike deep_copy_cube_progress._Progress.validate_config() (used by
   creation), there are no grow/shrink/clamp cases to handle here:
   old_total_layers and new_total_layers are both baked into
   `update_progress`'s own base path, so a resumed attempt at the exact same
   transition can only ever see identical values for them -- any mismatch
   at all (on those or the chunking parameters) means the markers
   underneath no longer mean what they claim.

   Parameters
   ----------
   recorded : dict
      This update transition's previously persisted configuration.
   current : dict
      This attempt's configuration, from _build_update_config().
   update_progress : _Progress
      Used only to name the config path in the error message.

   Raises
   ------
   RuntimeError
      On any mismatch.
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
   """Scan `{creation_progress.base}/updates/` for a transition directory
   left behind by an interrupted attempt -- one with a run_config.json but
   no top-level _SUCCESS marker.

   See this module's own docstring for why the store's declared shape can't
   be trusted to tell "up to date" apart from "an update started and got
   interrupted": once that's happened, this scan is the only reliable way
   to find (and resume, from the recorded old_total_layers rather than the
   corrupted live shape) the update that never finished.

   Parameters
   ----------
   creation_progress : _Progress
      Rooted at the creation run's own progress directory (NOT namespaced
      by any transition).

   Returns
   -------
   tuple of (int, int, _Progress, dict), or None
      (old_total_layers, new_total_layers, update_progress,
      recorded_update_config) for the one incomplete transition found, or
      None if `updates/` doesn't exist yet, or every transition under it is
      already marked complete (the expected steady state -- prune_var_markers()
      deliberately keeps _SUCCESS + run_config.json forever, so completed
      transitions accumulate over the store's lifetime).

   Raises
   ------
   RuntimeError
      If more than one incomplete transition is found -- at most one update
      should ever be in flight at a time; more than that means something
      odd happened and needs a human look, not a guess about which to
      resume.
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
   """Append any layers new to the virtual cube (index >= the deep-copy
   store's current length) onto an existing deep-copy zarr datacube built by
   deep_copy_cube_per_var_chunk.py, resumably and without a separate backup
   step -- see this module's own docstring for the rationale.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      s3:// URL of the existing deep-copy store to update.
   bucket_prefix : str
      S3 URL prefix the virtual chunk container resolves granule references
      against (see deep_copy_cube.open_virtual_cube).
   progress_dir : str
      s3:// directory the creation run (deep_copy_cube_per_var_chunk.py)
      recorded its run_config.json under, via its own --progress-dir. This
      update reads the chunking parameters from there (they cannot be
      supplied directly -- they must match the original build exactly) and
      keeps its own progress markers in a sub-path underneath it, namespaced
      by the (old, new) layer-count transition it's updating between.
   local_staging_dir : str
      Local directory the store's metadata skeleton is downloaded into, and
      every write lands in, before being synced back to `output_store`.
   keep_local_staging : bool
      If True, keep `local_staging_dir` after a successful run instead of
      removing it.
   num_load_workers : int, optional
      Thread-pool size for each batch .load() call -- see
      deep_copy_cube_per_var_chunk._load_batch()'s equivalent parameter.
   keep_progress_markers : bool
      If True, keep every per-variable/per-chunk marker for this update
      transition after a successful run instead of pruning them down to
      just _SUCCESS and run_config.json.
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

   # Set directly on the local root zarr.json's own attrs rather than via any
   # to_zarr() call: every write below is either a raw zarr array write or an
   # xarray to_zarr(region=...) write, and region writes never touch
   # Dataset-level (root group) attrs (verified directly). _upload_chunk()
   # re-uploads the root zarr.json after every chunk regardless, so this
   # reaches S3 on the very first chunk uploaded below.
   zarr.open_group(local_store, mode='r+', zarr_format=3).attrs[CubeFormat.date_updated] = (
      cube.attrs[CubeFormat.date_updated]
   )

   y_size = cube.sizes[utils.Coords.Y]
   x_size = cube.sizes[utils.Coords.X]

   # Resize + re-upload metadata + (if needed) merge the boundary chunk for
   # every time-indexed array up front, before any chunk gets (re)written --
   # mirrors deep_copy_cube_per_var_chunk()'s own creation-time atomicity
   # property of declaring every array's full shape in one pass before
   # filling in any pixel data, minimizing the window where the store's
   # arrays disagree on their 'time' length.
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
      update_progress.prune_var_markers([utils.Coords.TIME] + vars_1d + vars_3d)

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
