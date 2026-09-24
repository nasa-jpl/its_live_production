"""
Deep-copy a virtual ITS_LIVE datacube to a real Zarr v3 store.

For an s3:// --output-store: uploads each variable's zarr chunk to S3 as
soon as it's done, instead of writing the whole store locally and uploading
once at the end -- so the S3 store stays valid/browsable throughout the run
and a crash loses only the in-flight chunk. Supersedes
deep_copy_cube_per_var.py's single-upload-at-the-end approach as the
production creation script.

Per (variable, whole zarr chunk): write locally -> consolidate metadata ->
copy that chunk's subtree (`{var}/c/{chunk_idx}/`, which covers every
spatial shard regardless of --xy-shard-multiplier) + root zarr.json to S3 ->
delete the local copy (safe -- chunk_index is never revisited) -> mark done
if --progress-dir resumability is enabled. The S3 copy granularity (whole
chunk) is coarser than the internal write granularity (see
INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS) since a zarr chunk is atomic anyway.
The store's skeleton (root + every array's zarr.json + coordinate/static
data) is uploaded once up front so S3 always has valid metadata to resolve
against, even before the first chunk lands.

For a local --output-store: writes straight into it instead -- no
--local-staging-dir, no per-chunk upload/consolidate, no --progress-dir
(resumability is S3-specific) -- there's no separate publish target to
protect, so the whole point of this script's design doesn't apply; only the
per-(variable, chunk) write loop itself is reused.

Resumable when --progress-dir is given (s3:// output only) -- per-
(variable,chunk) markers let a retry skip whatever a prior attempt
finished, and the chunking parameters are recorded/re-checked so resuming
with mismatched arguments fails loudly instead of silently corrupting the
store. See deep_copy_cube_progress._Progress. Without --progress-dir this
behaves exactly as before, including refusing to overwrite an existing
--output-store.

Usage examples:
python src/deep_copy_cube_per_var_chunk.py \
   --input-store my_virtual_cube.icechunk \
   --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
   --local-staging-dir /local/scratch/my_deep_copy_cube.zarr \
   --progress-dir s3://its-live-data/path/to/deep_copy_progress

# Local output (e.g. for dev/testing) -- writes directly, no staging:
python src/deep_copy_cube_per_var_chunk.py \
   --input-store my_virtual_cube.icechunk \
   --output-store my_deep_copy_cube.zarr
"""
import gc
import logging
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import xarray as xr
import zarr
from zarr.errors import UnstableSpecificationWarning

import itslive_utils
import sensors
import utils
from itscube_types import CubeFormat, ImgPairInfo, Vars
from sensorFilters import SensorExcludeFilter
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
   split_time_vars_by_rank
)
from deep_copy_cube_progress import (
   SUCCESS_MARKER,
   _Progress,
   _log_resume_or_fresh,
   _s3_copy,
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


# Sub-writes per whole zarr time-chunk for a 3D variable, by dtype. MEASURED
# (2026-09-16; see src/wiki/07_Deep_Copy_Time_Chunking_And_Write_
# Amplification.md), not theoretical: write-amplification theory says N=1
# (whole chunk) is always cheapest, but a whole-chunk int write measured
# 2.4x SLOWER than two half-chunk writes at production scale (RAM/swap was
# clean; root cause not isolated). FLOAT_CHUNK_SPLITS also can't drop to 1
# on a 32 GB box regardless -- a float32 whole chunk at 512x512/
# time_chunk=20000 is ~39 GiB. Re-benchmark before changing either value.
INT_CHUNK_SPLITS = 2
FLOAT_CHUNK_SPLITS = 2

# Radar-only (Sentinel-1, NISAR) variables -- optical granules carry
# all-missing placeholders instead (virtual_itslive_cube.py). Equally
# eligible for the whole-chunk skip below on any all-optical time range.
RADAR_ONLY_VARS = {Vars.m11, Vars.m12, Vars.vr, Vars.va}

# Mission groups whose granules are radar (SAR).
RADAR_GROUP_IDS = {sensors.SENTINEL1.id, sensors.NISAR.id}


@itslive_utils.retry_decorator(max_retries=5)
def _load_batch(cube, var_name, start, stop, num_load_workers=None):
   """Materialize `var_name`'s [start:stop) time slice from `cube`, retrying
   on any exception -- icechunk's s3_store() (unlike obstore.S3Store used
   elsewhere in this pipeline) has no retry/backoff of its own.

   Args:
      cube (xr.Dataset): the virtual datacube.
      var_name (str): variable to load.
      start (int): first layer index to load.
      stop (int): one past the last layer index to load.
      num_load_workers (int): dask threaded-scheduler size for this
         .load() call; each granule is one dask task. None leaves
         dask's own default (CPU core count).
   """
   load_kwargs = {'scheduler': 'threads'}
   if num_load_workers is not None:
      load_kwargs['num_workers'] = num_load_workers

   return cube[[var_name]].isel(
      {utils.Coords.TIME: slice(start, stop)}
   ).drop_vars(
      [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X], errors='ignore'
   ).load(**load_kwargs)


def _compute_radar_mask(cube, total_layers):
   """Classify each of `cube`'s first `total_layers` layers as radar or not.

   Classifies via SensorExcludeFilter.map_sensor_to_group() on
   (mission_img1, satellite_img1) rather than inspecting the manifests
   directly, on the assumption every radar-classified granule genuinely
   carries M11/M12/vr/va (true for Sentinel-1; NISAR included in
   RADAR_GROUP_IDS ahead of its planned addition). An unrecognized sensor
   raises KeyError, left unhandled on purpose -- guessing "optical" would
   silently skip real data.

   Args:
      cube (xr.Dataset): the virtual datacube.
      total_layers (int): number of leading layers to classify.

   Returns:
      np.ndarray of bool: True for each layer whose granule is a radar
         (SAR) mission -- the layers for which RADAR_ONLY_VARS carries
         real data rather than an all-missing placeholder.
   """
   sensor_info = cube[[ImgPairInfo.mission_img1, ImgPairInfo.satellite_img1]].isel(
      {utils.Coords.TIME: slice(0, total_layers)}
   ).load()

   group_ids = SensorExcludeFilter.map_sensor_to_group(
      sensor_info[ImgPairInfo.satellite_img1].values,
      sensor_info[ImgPairInfo.mission_img1].values
   )
   return np.isin(group_ids, list(RADAR_GROUP_IDS))


def _fill_nan_in_place(values, fill_value):
   """Replace NaN in `values` with `fill_value` in place -- byte-identical to what
   xarray's CF encoder (CFMaskCoder.encode) would write, but without its
   allocations. On a region write xarray always re-derives encoding from
   the store (_reset_write_encoding() has no effect there) and runs
   `fillna` = `where(notnull(data), data, fill)`, allocating a full bool
   mask + inverse + output copy on every write, for every variable that
   declares a CF fill -- including int variables, where it's a no-op. For a
   10000-layer float32 batch at 512x512 that's ~14.7 GiB of temporaries on
   top of a 9.8 GiB batch: the actual cause of a real M11/M12 OOM (and, via
   a forced-down batch size, a ~2x runtime regression). np.copyto with
   `where=` allocates only the mask.

   No-ops (nothing to substitute): int/uint batches loaded with
   mask_and_scale=False can't contain NaN; fill_value=None means the
   variable declares no CF fill (see deep_copy_cube.NO_FILL_VARS); a NaN
   fill_value already matches.

   Args:
      values (np.ndarray): array to patch in place.
      fill_value (float): CF fill to substitute for NaN; None for a
         variable with no CF fill.
   """
   if fill_value is None or values.dtype.kind != 'f':
      return

   if np.isnan(fill_value):
      return

   np.copyto(values, fill_value, where=np.isnan(values))


def _upload_chunk(local_store, output_store, var_name, chunk_index):
   """Consolidate `local_store`'s local metadata, sync one finished zarr
   chunk (data + refreshed root zarr.json) from `local_store` to
   `output_store`, then delete the local copy.

   `{var_name}/c/{chunk_index}` covers every spatial shard under that
   time-chunk regardless of --xy-shard-multiplier ('time' is always first
   under 'c/'); for a 1D variable the same path is a plain file instead of a
   directory, so the copy/delete both branch on os.path.isdir()
   (`aws s3 cp --recursive` errors on a file source). Delete is safe:
   chunk_index is never revisited by any caller, and _s3_copy() raises
   (rather than returning) on total failure, so a failed upload always
   leaves its local chunk in place to retry from.

   Args:
      local_store (str): local staging store path.
      output_store (str): s3:// destination to sync the chunk to.
      var_name (str): variable the chunk belongs to.
      chunk_index (int): index of the zarr chunk along time.
   """
   chunk_dir = f'{var_name}/c/{chunk_index}'
   local_chunk_path = os.path.join(local_store, chunk_dir)

   if not os.path.exists(local_chunk_path):
      # zarr's write_empty_chunks=False (the zarr-python default) silently
      # skips writing a chunk whose values all equal the array's fill_value
      # -- e.g. an all-optical time range for a radar-only derived variable
      # like M11_dr_to_vr_factor/M12_dr_to_vr_factor (1D, so not covered by
      # RADAR_ONLY_VARS's whole-chunk skip, which only guards the 3D write
      # path). Nothing was written locally, so there's nothing to sync --
      # it already reads back as fill_value via the template's metadata.
      logging.info(
         f'{chunk_dir}: not written locally (all-fill chunk skipped by '
         'zarr write_empty_chunks) -- nothing to upload'
      )
      return

   zarr.consolidate_metadata(local_store)

   chunk_is_dir = os.path.isdir(local_chunk_path)
   _s3_copy(
      local_chunk_path,
      f'{output_store.rstrip("/")}/{chunk_dir}',
      recursive=chunk_is_dir
   )
   _s3_copy(
      os.path.join(local_store, 'zarr.json'),
      f'{output_store.rstrip("/")}/zarr.json',
      recursive=False
   )

   if chunk_is_dir:
      shutil.rmtree(local_chunk_path)
   else:
      os.remove(local_chunk_path)


def _write_var_3d_and_upload(
   cube, local_store, output_store, var_name, chunk_size, total_layers,
   fill_value=None, is_radar=None, num_load_workers=None, progress=None,
   start_layer=0
):
   """Write one 3D (time,y,x) variable in half-chunk pieces via the raw
   zarr array API (not to_zarr()/the CF encoder), uploading each whole
   zarr chunk from `local_store` to `output_store` as soon as it's locally
   complete. Neither optimization changes a byte on disk -- see
   INT_CHUNK_SPLITS and _fill_nan_in_place(): half-chunk writes (measured
   faster than whole-chunk even for ints, despite write-amplification
   theory saying otherwise) and raw writes (skips the CF encoder's
   ~1.5x-batch allocation; safe since the template already fixed
   shape/dtype/chunks/fill_value -- only pixels are left to place).

   Raises ValueError if dims aren't (time,y,x) -- refuses to transpose
   rather than silently allocate a full extra batch copy.

   Args:
      cube (xr.Dataset): the virtual datacube `var_name` is read from.
      local_store (str): local staging store path.
      output_store (str): destination each finished chunk is uploaded to;
         if equal to `local_store` (a local `output_store`), no upload
         happens since `local_store` already IS the final destination.
      var_name (str): the 3D variable to write.
      chunk_size (int): zarr chunk size along time.
      total_layers (int): number of leading layers to write.
      fill_value (float): CF fill substituted for NaN before the raw
         write (see _fill_nan_in_place()); None for a variable with no
         CF fill.
      is_radar (np.ndarray): per-layer radar/optical mask; for a
         RADAR_ONLY_VARS variable, a whole chunk with no True layers is
         skipped entirely (never written by the template either, so it
         reads back as fill_value). None disables the skip.
      num_load_workers (int): forwarded to each batch's _load_batch()
         call.
      progress (_Progress): enables resumability -- this variable's own
         marker is checked first, then each chunk's, before doing that
         chunk's work.
      start_layer (int): used by deep_copy_update_per_var_chunk.py -- the
         outer chunk loop starts at start_layer's own chunk instead of
         chunk 0, and only that one boundary chunk's write/radar-check
         is narrowed to [start_layer, chunk_stop); every other chunk is
         unaffected. 0 (the default) reproduces the original code path
         exactly.
   """
   expected_dims = (utils.Coords.TIME, utils.Coords.Y, utils.Coords.X)
   if cube[var_name].dims != expected_dims:
      raise ValueError(
         f'{var_name} has dims {cube[var_name].dims}, expected '
         f'{expected_dims}; refusing to transpose (would allocate a full '
         f'extra copy of every batch)'
      )

   if progress is not None and progress.var_is_done(var_name):
      logging.info(
         f'{var_name}: already fully processed in a previous attempt, skipping'
      )
      return

   splits = FLOAT_CHUNK_SPLITS if cube[var_name].dtype.kind == 'f' else INT_CHUNK_SPLITS
   write_span = max(1, chunk_size // splits)
   check_radar = is_radar is not None and var_name in RADAR_ONLY_VARS
   num_chunks = 0
   num_skipped = 0
   num_resumed = 0

   # Safe to open once: consolidate_metadata() below only rewrites root
   # zarr.json, never this array's own metadata.
   target = zarr.open_group(local_store, mode='r+', zarr_format=3)[var_name]

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

      if check_radar and not is_radar[write_start:chunk_stop].any():
         logging.info(
            f'Skipping {var_name} chunk {chunk_start}:{chunk_stop} of '
            f'{total_layers} (all-optical, no radar layers present)'
         )
         num_skipped += 1
         if progress is not None:
            progress.mark_chunk_done(var_name, chunk_index)
         continue

      for start in range(write_start, chunk_stop, write_span):
         stop = min(start + write_span, chunk_stop)
         logging.info(f'Materializing {var_name} layers {start}:{stop} of {total_layers}')

         batch = _load_batch(cube, var_name, start, stop, num_load_workers)
         values = batch[var_name].values
         _fill_nan_in_place(values, fill_value)
         target[start:stop, :, :] = values

         logging.info(f'Wrote {var_name} layers {start}:{stop} of {total_layers} to {local_store}')

         del values
         del batch
         gc.collect()

      if local_store != output_store:
         _upload_chunk(local_store, output_store, var_name, chunk_index)
         logging.info(
            f'Uploaded {var_name} chunk {chunk_start}:{chunk_stop} of '
            f'{total_layers} to {output_store}'
         )
      if progress is not None:
         progress.mark_chunk_done(var_name, chunk_index)

   if check_radar and num_skipped:
      logging.info(
         f'{var_name}: skipped {num_skipped} of {num_chunks} chunk(s) '
         f'(all-optical, no radar layers present)'
      )

   if num_resumed:
      logging.info(
         f'{var_name}: resumed past {num_resumed} of {num_chunks} chunk(s) '
         '(already done in a previous attempt)'
      )

   if progress is not None:
      progress.mark_var_done(var_name)


def _write_var_1d_and_upload(
   cube, local_store, output_store, var_name, chunk_size, total_layers,
   num_load_workers=None, progress=None, start_layer=0
):
   """Write one 1D (time,) variable via xarray's to_zarr(region=...), one
   whole zarr chunk per write, uploading each from `local_store` to
   `output_store` as soon as it's done.

   Kept on the xarray/CF-encoding path deliberately, unlike
   _write_var_3d_and_upload(): 1D variables are ~262k times smaller per
   layer, so the encoder's cost is irrelevant, while its correctness
   (datetime/string encoding) very much isn't. Written whole-chunk rather
   than half: every 1D variable's dask chunking is a single chunk spanning
   all layers, not one per granule, so there's no per-granule task-graph to
   blow up and RAM is a non-issue regardless.

   Args:
      cube (xr.Dataset): the virtual datacube `var_name` is read from.
      local_store (str): local staging store path.
      output_store (str): destination each finished chunk is uploaded to;
         if equal to `local_store` (a local `output_store`), no upload
         happens since `local_store` already IS the final destination.
      var_name (str): the 1D variable to write.
      chunk_size (int): zarr chunk size along time.
      total_layers (int): number of leading layers to write.
      num_load_workers (int): same semantics as
         _write_var_3d_and_upload()'s.
      progress (_Progress): same semantics as
         _write_var_3d_and_upload()'s.
      start_layer (int): same semantics as _write_var_3d_and_upload()'s.
   """
   if progress is not None and progress.var_is_done(var_name):
      logging.info(
         f'{var_name}: already fully processed in a previous attempt, skipping'
      )
      return

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

      logging.info(f'Materializing {var_name} layers {write_start}:{chunk_stop} of {total_layers}')

      batch = _load_batch(cube, var_name, write_start, chunk_stop, num_load_workers)
      _reset_write_encoding(batch)
      batch.to_zarr(
         local_store,
         mode='r+',
         region={utils.Coords.TIME: slice(write_start, chunk_stop)},
         zarr_format=3,
         consolidated=False
      )

      logging.info(f'Wrote {var_name} layers {write_start}:{chunk_stop} of {total_layers} to {local_store}')

      del batch
      gc.collect()

      if local_store != output_store:
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


def deep_copy_cube_per_var_chunk(
   input_store,
   output_store,
   bucket_prefix,
   time_chunk,
   xy_chunk,
   time_chunk_1d,
   xy_shard_multiplier=1,
   local_staging_dir=None,
   keep_local_staging=False,
   num_layers=0,
   num_load_workers=None,
   progress_dir=None,
   keep_progress_markers=False
):
   """Materialize `input_store` into a real zarr v3 datacube at
   `output_store`. For an s3:// `output_store`, uploads each whole zarr
   chunk to S3 as soon as it's done (see this module's docstring). For a
   local `output_store`, writes straight into it instead -- no staging dir,
   no per-chunk upload -- since there's no separate publish target to keep
   valid mid-run.

   Resumable when `progress_dir` is given (s3:// output only -- resumability
   relies on S3-specific existence checks that don't apply to a local
   store). Without it, behaves exactly as before resumability existed,
   including refusing to overwrite an existing output store. Fresh-vs-resume
   is auto-detected from the markers themselves, since an AWS Batch retry
   resubmits the identical command line; a completed prior attempt
   short-circuits immediately, an interrupted one skips only what it already
   finished. The parameters that give a chunk marker its meaning are
   recorded and re-checked on resume, so a mismatch fails loudly instead of
   silently corrupting the store -- see deep_copy_cube_progress._Progress.

   Args:
      input_store (str): virtual datacube (icechunk repo) to read.
      output_store (str): destination for the deep-copy store (s3:// or
         local).
      bucket_prefix (str): S3 prefix `input_store`'s granule references
         are resolved against.
      time_chunk (int): zarr chunk size along time for 3D variables.
      xy_chunk (int): zarr chunk size along x/y for 3D variables.
      time_chunk_1d (int): zarr chunk size along time for 1D variables.
      xy_shard_multiplier (int): inner chunks per shard per spatial
         axis; 1 disables sharding.
      local_staging_dir (str): local directory writes land in before
         syncing to `output_store`. Only valid (and required) when
         `output_store` is s3://; must be unset for a local `output_store`.
      keep_local_staging (bool): keep `local_staging_dir` after a
         successful run instead of deleting it.
      num_layers (int): cap on how many layers to process; 0 means all.
      num_load_workers (int): sizes each batch's load thread-pool (see
         _load_batch()).
      progress_dir (str): s3:// directory for resumability progress
         markers; None disables resumability. Requires an s3:// `output_store`.
      keep_progress_markers (bool): keep the progress markers around
         after a successful run instead of pruning them.
   """
   is_s3_output = output_store.startswith(utils.S3_PREFIX)

   if local_staging_dir and not is_s3_output:
      raise ValueError(
         "--local-staging-dir only applies when --output-store is an s3:// "
         f"path, got {output_store}"
      )

   if is_s3_output and not local_staging_dir:
      raise ValueError(
         "--local-staging-dir is required when --output-store is an s3:// path"
      )

   if progress_dir and not is_s3_output:
      raise ValueError(
         "--progress-dir requires an s3:// --output-store -- resumability "
         "relies on S3-specific existence checks (see _log_resume_or_fresh) "
         f"that don't apply to a local store, got --output-store={output_store}"
      )

   if progress_dir and not progress_dir.startswith(utils.S3_PREFIX):
      raise ValueError(
         f"--progress-dir must be an s3:// path so markers survive the EC2 "
         f"instance they were written from (a local directory would be lost "
         f"with the instance, which defeats the point), got {progress_dir}"
      )

   progress = _Progress.create(progress_dir, output_store) if progress_dir else None
   if progress is not None and progress.is_complete():
      logging.info(
         f'{output_store} is already marked complete '
         f'({progress.base}/{SUCCESS_MARKER}); nothing to do'
      )
      return

   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   if num_layers > 0 and num_layers < total_layers:
      logging.info(f'Limiting to first {num_layers} of {total_layers} layers (--num-layers)')
      total_layers = num_layers

   if total_layers == 0:
      logging.info(f'{input_store} has no layers, nothing to deep-copy')
      return

   # Reconcile against (or establish) the recorded run configuration before
   # anything downstream consumes total_layers -- validate_config() can clamp
   # it back to what a prior attempt froze the output store's shape at, and
   # both _compute_radar_mask() and build_encoding() below must see the
   # clamped value.
   if progress is not None:
      run_params = {
         'time_chunk': time_chunk,
         'xy_chunk': xy_chunk,
         'time_chunk_1d': time_chunk_1d,
         'xy_shard_multiplier': xy_shard_multiplier,
         'num_layers': num_layers,
      }
      time_values = cube[utils.Coords.TIME].values
      recorded_config = progress.read_config()

      if recorded_config is None:
         progress.write_config(
            _Progress.build_config(output_store, run_params, total_layers, time_values)
         )
      else:
         total_layers = progress.validate_config(
            recorded_config,
            _Progress.build_config(output_store, run_params, total_layers),
            total_layers,
            time_values
         )

   time_vars, static_vars = split_vars_by_time(cube)
   vars_3d, vars_1d = split_time_vars_by_rank(cube, time_vars)
   logging.info(
      f'Batching over {len(vars_1d)} 1D variable(s) '
      f'({time_chunk_1d} layers/write) and {len(vars_3d)} 3D variable(s) '
      f'({time_chunk} layers/write)'
   )

   # Shared across every RADAR_ONLY_VARS variable -- same missing-ness.
   is_radar = _compute_radar_mask(cube, total_layers)
   logging.info(
      f'{np.count_nonzero(is_radar)} of {total_layers} layers are radar '
      'granules (see RADAR_GROUP_IDS)'
   )

   encoding = build_encoding(
      cube, time_chunk, xy_chunk, time_chunk_1d,
      xy_shard_multiplier
   )

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   # With resumability on, a retry finding an already-populated store is
   # expected, not an error -- the strict refuse-to-overwrite guard would
   # break exactly the case it's meant to protect.
   if progress is not None:
      _log_resume_or_fresh(progress.s3, output_store)
   else:
      resolve_output_store(output_store)

   local_store = resolve_output_store(local_staging_dir) if local_staging_dir else output_store

   # compute=False defers pixel writes for dask-backed variables (time/x/y
   # coords are always written for real here); mode='r+' below requires
   # every variable already declared, hence templating all of them up
   # front. safe_chunks=False: the virtual cube's per-granule (chunk=1)
   # dask chunking doesn't match encoding's larger time_chunk, which
   # xarray's parallel-write safety check would otherwise flag -- moot here
   # since compute=False never executes the write, only declares metadata.
   template = xr.merge([
      cube[time_vars].isel({utils.Coords.TIME: slice(0, total_layers)}),
      cube[static_vars]
   ])
   _reset_write_encoding(template)
   template.to_zarr(
      local_store,
      mode='w',
      compute=False,
      encoding=encoding,
      zarr_format=3,
      consolidated=False,
      safe_chunks=False
   )
   logging.info(f'Created local template store at {local_store}')

   # template's chunk-manifest bookkeeping is non-trivial and unused below
   # (later code re-derives batches from `cube` directly) -- free it now.
   del template
   gc.collect()

   static_batch = cube[static_vars].load()
   _reset_write_encoding(static_batch)
   static_batch.to_zarr(local_store, mode='r+', zarr_format=3, consolidated=False)
   logging.info(f'Wrote {len(static_vars)} static variable(s) to {local_store}')

   if is_s3_output:
      # Skeleton (root+array zarr.json + coord/static data) never changes
      # again, but S3 needs it before any chunk lands so a mid-run reader
      # always finds valid metadata (unwritten chunks read as fill_value).
      # Not needed for a local output_store: it's already the final store,
      # consolidated once at the end instead (see below).
      zarr.consolidate_metadata(local_store)
      _s3_copy(local_store, output_store)
      logging.info(f'Uploaded store skeleton to {output_store}')

   for var_name in vars_1d:
      _write_var_1d_and_upload(
         cube, local_store, output_store, var_name, time_chunk_1d, total_layers,
         num_load_workers, progress
      )

   for var_name in vars_3d:
      # Only floats get a CF fill under build_encoding()'s convention
      # (ints use 'missing_value'; _fill_nan_in_place() no-ops for them).
      _write_var_3d_and_upload(
         cube, local_store, output_store, var_name, time_chunk, total_layers,
         encoding.get(var_name, {}).get(utils.OutputFormat.fill_value),
         is_radar, num_load_workers, progress
      )

   if not is_s3_output:
      # Local output skipped the per-chunk consolidate _upload_chunk() would
      # otherwise have done -- do it once now that every write is in.
      zarr.consolidate_metadata(local_store)
      logging.info(f'Consolidated metadata at {local_store}')

   if progress is not None:
      # Marked complete BEFORE pruning, so an interrupted prune still leaves
      # the store recognizable as finished on the next attempt.
      progress.mark_complete()
      logging.info(f'Marked {output_store} complete ({progress.base}/{SUCCESS_MARKER})')

      if keep_progress_markers:
         logging.info(f'Keeping every progress marker under {progress.base}')
      else:
         progress.prune_var_markers(vars_1d + vars_3d)

   if local_staging_dir:
      if keep_local_staging:
         logging.info(f'Keeping local staging directory {local_store}')
      else:
         logging.info(f'Removing local staging directory {local_store}')
         shutil.rmtree(local_store)

   logging.info(f'Done: deep-copied {total_layers} layers to {output_store}')


if __name__ == '__main__':
   import argparse

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Materialize a virtual ITS_LIVE datacube (icechunk repo built by
      virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube. For
      an s3:// --output-store, uploads each whole zarr chunk to S3 as soon
      as it's written locally. For a local --output-store, writes directly
      into it instead (no staging, no per-chunk upload). See this module's
      docstring for the rationale.

      Usage example:
      python src/deep_copy_cube_per_var_chunk.py \
         --input-store my_virtual_cube.icechunk \
         --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
         --local-staging-dir /local/scratch/my_deep_copy_cube.zarr \
         --progress-dir s3://its-live-data/path/to/deep_copy_progress

      # Local output (e.g. for dev/testing) -- writes directly, no staging:
      python src/deep_copy_cube_per_var_chunk.py \
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
      help='Path to write the deep-copy Zarr v3 store to (s3:// or local). '
         'A local path writes directly, without --local-staging-dir or '
         '--progress-dir (see those flags\' help).'
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
      help='Chunk size along time for 3D (time, y, x) variables. Each write '
         'for a 3D variable materializes half this many layers '
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
      help='Chunk size for 1D (time,) variables [%(default)d].'
   )
   parser.add_argument(
      '--xy-shard-multiplier',
      type=int,
      default=XY_SHARD_MULTIPLIER,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables [%(default)d]. '
         'A value of 1 disables sharding.'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      default=None,
      help='Local directory every write lands in before being synced to '
         '--output-store one whole zarr chunk at a time. Required when '
         '--output-store is s3://; must be omitted for a local '
         '--output-store, which is written to directly.'
   )
   parser.add_argument(
      '--keep-local-staging',
      action='store_true',
      help='Keep --local-staging-dir after a successful run instead of '
         'deleting it [%(default)s].'
   )
   parser.add_argument(
      '-n', '--num-layers',
      type=int,
      default=0,
      help='Only materialize the first N layers of the virtual cube '
         '[%(default)d meaning to process all layers].'
   )
   parser.add_argument(
      '--num-load-workers',
      type=int,
      default=None,
      help='Thread-pool size for each batch .load() call (each granule is '
         'one dask task, chunk size 1 along "time" in the virtual cube, so '
         'this bounds how many granules get fetched/decompressed '
         'concurrently). Unset (the default) leaves dask\'s own default in '
         'effect (CPU core count).'
   )
   parser.add_argument(
      '--progress-dir',
      type=str,
      default=None,
      help='s3:// directory to keep progress markers in, which enables '
         'resuming an interrupted run (e.g. after an AWS Batch spot '
         'termination) instead of rebuilding the whole cube. Kept out of '
         '--output-store on purpose so the published cube carries none of '
         'this bookkeeping; markers go under '
         '{--progress-dir}/{output store name}/, so one shared directory '
         'can serve a whole batch of cubes. Unset (the default) disables '
         'resumability and restores the usual refuse-to-overwrite guard on '
         '--output-store. Requires an s3:// --output-store -- not supported '
         'for a local one.'
   )
   parser.add_argument(
      '--keep-progress-markers',
      action='store_true',
      help='Keep every per-variable/per-chunk progress marker after a '
         'successful run instead of pruning them down to just _SUCCESS and '
         'run_config.json (useful to inspect which chunks were '
         'radar-skipped vs. written). No effect without --progress-dir '
         '[%(default)s].'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   deep_copy_cube_per_var_chunk(
      args.input_store,
      args.output_store,
      args.bucket,
      args.time_chunk_value,
      args.xy_chunk_value,
      args.time_chunk_value_1d,
      args.xy_shard_multiplier,
      args.local_staging_dir,
      args.keep_local_staging,
      args.num_layers,
      args.num_load_workers,
      args.progress_dir,
      args.keep_progress_markers
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
