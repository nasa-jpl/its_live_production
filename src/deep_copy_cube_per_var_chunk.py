"""
EXPERIMENTAL variant of deep_copy_cube_per_var.py: instead of writing the
whole store locally (or to --local-staging-dir) and uploading it all in one
shot at the very end, this script uploads each variable's zarr chunk to its
final S3 destination as soon as that one chunk is done -- so the S3 cube is a
valid, browsable (if incomplete) store throughout the run, not just after it
finishes.

Why: deep_copy_cube_per_var.py's --local-staging-dir already writes locally
first and uploads once at the end (to dodge many small direct-to-S3 writes/
rewrites of the same chunk). That's efficient, but it means the S3 store
doesn't exist in any usable form until the whole run succeeds -- a crash near
the end loses all of it, and there's no way to inspect progress mid-run. This
script keeps the "write locally first" part (still needed to avoid direct-to-
S3 partial-chunk rewrites) but syncs to S3 incrementally, one whole zarr
chunk at a time, instead of once at the end.

Chunk-write granularity vs. copy granularity are deliberately different: the
write sizing from deep_copy_cube_per_var.py is used internally (both 3D int
and 3D float variables in half-time-chunk writes -- see
INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS, and _write_var_3d_and_upload() for why
3D variables bypass xarray's CF encoder), but the S3 copy only happens once
every write making up a given whole zarr chunk has landed locally. A zarr
chunk is atomic anyway (see RADAR_ONLY_VARS's whole-chunk skip), so there's
no reason to sync a half-finished one.

Sequence for each (variable, whole zarr chunk):
1. Load+write the chunk into the local staging store (same as
   deep_copy_cube_per_var.py).
2. zarr.consolidate_metadata() the local store.
3. Copy that chunk's new chunk/shard files (scoped to `{var}/c/{chunk_idx}/`
   -- this correctly captures every spatial chunk/shard under it regardless
   of --xy-shard-multiplier, since 'time' is always the first path segment
   under 'c/') to S3.
4. Copy the freshly consolidated root zarr.json to S3.

Before any of that, the store's skeleton (root zarr.json + every array's own
zarr.json + coordinate/static variable data -- none of which ever changes
after the initial template write, since shape/dtype/chunks/encoding are all
declared up front) is uploaded once, so S3 always has valid metadata to
resolve any chunk path against, even before the first real chunk lands.

NOTE: re-consolidating and re-uploading the root zarr.json after every chunk
is technically redundant here -- the store's schema never changes after the
initial template write, so the very first consolidation is already complete
and correct for the life of the run. Doing it again each time costs one tiny
extra S3 PUT per chunk, which is cheap enough not to bother special-casing.

Skipped (all-optical) chunks (see RADAR_ONLY_VARS) need no copy at all --
nothing was written locally for them either, so there's nothing to sync.

This is a test/benchmark script for comparing incremental-sync wall-clock and
S3 request cost against deep_copy_cube_per_var.py's single-upload-at-the-end
approach -- not yet a production replacement.

Usage example:
python src/deep_copy_cube_per_var_chunk.py \
   --input-store my_virtual_cube.icechunk \
   --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
   --local-staging-dir /local/scratch/my_deep_copy_cube.zarr
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
)
from deep_copy_cube_per_var import split_time_vars_by_rank

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


# Suppress Zarr V3 unstable string dtype warnings, same rationale as
# deep_copy_cube.py.
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)


# Number of equal sub-writes each whole zarr time-chunk is split into for a
# 3D variable, by dtype. In *theory* every inner chunk spans the FULL
# time_chunk extent (chunks are (time_chunk, xy_chunk, xy_chunk)), so
# splitting a chunk's write into N pieces should cost N full-volume
# compressions + N-1 full-volume decompressions regardless of piece size --
# making N=1 (whole chunk) always fastest, with RAM as the only reason to
# split.
#
# MEASURED (2026-09-16, in deep_copy_cube_per_var.py -- see
# INT_CHUNK_SPLITS's comment there for the full writeup) this theory doesn't
# hold at production scale: a whole-chunk (20000-layer) write of an int
# variable (chip_size_height, no CF-encoder cost either way -- see
# _fill_nan_in_place) took 632s, vs 268s total for the same 20000 layers as
# two half-chunk (10000-layer) writes on the same run -- 2.4x SLOWER despite
# strictly less write-side work by the theory above. RAM/swap was confirmed
# clean throughout, so it isn't the RAM tradeoff this constant was designed
# around -- something about a single, much bigger .load() call (dask
# task-graph construction, or S3 request-pattern effects) dominates at this
# scale instead. Root cause not yet isolated; INT_CHUNK_SPLITS is set
# empirically (matching FLOAT_CHUNK_SPLITS) rather than by the theory above
# until it is. Applied here too since this script shares _load_batch()'s
# per-granule fetch mechanism with deep_copy_cube_per_var.py, where the
# measurement was made.
#
# ATTN: re-benchmark before changing either value. If revisiting the RAM
# arithmetic: peak per write is roughly (batch bytes) x ~2 (the extra ~1x
# being the .load() fan-out a virtual/manifest-backed read pays); at the
# production 512x512 grid and time_chunk=20000, a float32 whole chunk is
# ~39 GiB (doesn't fit a 32 GB box) vs ~20 GiB halved -- that RAM ceiling is
# still real and still why FLOAT_CHUNK_SPLITS can't drop to 1 on a 32 GB box.
INT_CHUNK_SPLITS = 2
FLOAT_CHUNK_SPLITS = 2

# Variables present ONLY in radar (Sentinel-1, NISAR) granules -- optical
# (Landsat, Sentinel-2) granules carry all-missing placeholders for these
# instead (see virtual_itslive_cube.py's _add_missing_m11_m12()/
# _add_missing_vr_va()). M11/M12 are float32 and vr/va int16, so they split
# their chunk writes differently (see INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS),
# but all four are equally all-fill-value for every optical time range and so
# equally eligible for the whole-chunk skip below.
RADAR_ONLY_VARS = {Vars.m11, Vars.m12, Vars.vr, Vars.va}

# Mission groups whose granules are radar (SAR).
RADAR_GROUP_IDS = {sensors.SENTINEL1.id, sensors.NISAR.id}


def _s3_copy(local_path, s3_path, recursive=True):
   """Copy a local file or directory to S3 via the AWS CLI, matching
   deep_copy_cube.upload_local_staging_dir()'s mechanism (retried, subprocess-
   based) but scoped to a single path instead of a whole store, so it can be
   called once per chunk without waiting for the whole run to finish.

   Parameters
   ----------
   local_path : str
      Local file or directory to copy.
   s3_path : str
      Destination s3:// URL.
   recursive : bool
      True (default) for a directory copy, False for a single file.
   """
   command_line = ["aws", "s3", "cp"]
   if recursive:
      command_line.append("--recursive")
   command_line += [local_path, s3_path, "--acl", "bucket-owner-full-control"]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())


@itslive_utils.retry_decorator(max_retries=5)
def _load_batch(cube, var_name, start, stop, num_load_workers=None):
   """Materialize one variable's [start:stop) time slice, retrying on any
   exception.

   This is the actual S3 fetch of the granule bytes a virtual chunk
   references, via icechunk's own s3_store() -- which, unlike the
   obstore.S3Store used elsewhere in this pipeline (see
   virtual_itslive_cube_per_chunk.py's RETRY_CONFIG), has no configurable
   retry/backoff of its own, so a transient network blip here would
   otherwise fail the whole run outright.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   var_name : str
      Name of the data variable to load.
   start, stop : int
      Time-slice bounds (see _write_var_3d_and_upload/_write_var_1d_and_upload).
   num_load_workers : int, optional
      Thread-pool size for this .load() call's dask threaded scheduler --
      each granule in [start, stop) is one dask task (chunk size 1 along
      'time' in the virtual cube), so this bounds how many granules get
      fetched/decompressed concurrently. Passed straight through to
      dask.compute() via xr.Dataset.load(**kwargs); None (the default)
      leaves dask's own default in effect (CPU core count).

   Returns
   -------
   xr.Dataset
      The loaded (non-dask) batch for this variable and time slice.
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
   """Determine, for each of the cube's first `total_layers` layers, whether
   that layer's granule is from a radar (SAR) mission -- the layers for
   which RADAR_ONLY_VARS carry real data rather than an all-missing
   placeholder.

   Uses (mission_img1, satellite_img1) via the same
   SensorExcludeFilter.map_sensor_to_group() classification already used
   elsewhere in this codebase (sensorFilters.py), rather than inspecting
   RADAR_ONLY_VARS's own virtual chunk manifests directly, on the assumption
   that every granule this classifier maps to a radar mission group
   genuinely carries M11/M12/vr/va (true today for Sentinel-1; NISAR is
   included in RADAR_GROUP_IDS ahead of its planned addition to these
   datacubes).

   map_sensor_to_group() raises KeyError for any (mission, satellite) pair
   it doesn't recognize -- intentionally left unhandled here: an unknown
   sensor value means we cannot classify that layer, and guessing wrong in
   the "optical" direction would silently skip real data, so failing the
   whole run is the correct behavior rather than a graceful fallback.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   total_layers : int
      Number of layers to classify, from the start of 'time' (honors
      --num-layers).

   Returns
   -------
   np.ndarray
      Boolean array of length total_layers; True where that layer's granule
      is from a radar mission group (see RADAR_GROUP_IDS).
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
   """Replace NaN with `fill_value` in `values`, in place.

   This reproduces exactly what xarray's CF encoder (CFMaskCoder.encode)
   would do on write -- but in place, instead of allocating a whole extra
   copy of the batch.

   Worth spelling out why this is done by hand rather than left to xarray.
   On a region write into an existing store, xarray does NOT use the
   batch's own encoding: backends/zarr.py replaces it wholesale with the
   store's decoded encoding ("vars_with_encoding[vn].encoding =
   existing_vars[vn].encoding"), so _reset_write_encoding() clearing the
   batch's fill has no effect on this path -- the store's _FillValue is
   re-injected and CFMaskCoder runs regardless. It then does
   `data = fillna(data, fill_value)`, i.e. `where(notnull(data), data,
   fill)`, which allocates a full bool mask, its inverse, AND a full-size
   output copy. For a 10000-layer float32 batch at 512x512 that's ~14.7 GiB
   of temporaries on top of a 9.8 GiB batch -- the actual cause of the
   M11/M12 OOM, and (by forcing the batch size down, which multiplies
   partial-chunk read-modify-write passes) of a ~2x runtime regression.
   np.copyto with a boolean `where` allocates only the mask (~2.4 GiB for
   that same batch) and mutates the buffer we're about to discard anyway.

   Doing this here and writing the array with the raw zarr API keeps the
   on-disk result byte-identical to what xarray would have written -- this
   is purely an allocation/CPU optimization, not a format change.

   No-ops when there's nothing to do: an int/uint batch loaded with
   mask_and_scale=False cannot contain NaN at all (xarray's own fillna is
   pure waste there -- it still allocates all three temporaries), a
   fill_value of None means the variable declares no CF fill (see
   deep_copy_cube.NO_FILL_VARS), and a NaN fill_value already matches NaN
   in the data.

   Parameters
   ----------
   values : np.ndarray
      The batch's raw values, mutated in place.
   fill_value : scalar or None
      The variable's CF fill value as declared in the output store (see
      build_encoding()); None if it declares none.
   """
   if fill_value is None or values.dtype.kind != 'f':
      return

   if np.isnan(fill_value):
      return

   np.copyto(values, fill_value, where=np.isnan(values))


def _upload_chunk(local_store, output_store, var_name, chunk_index):
   """Consolidate the local store's metadata and sync one finished zarr chunk
   (plus the refreshed root zarr.json) to its final S3 destination.

   `{var_name}/c/{chunk_index}` captures every spatial chunk/shard under that
   time-chunk regardless of --xy-shard-multiplier, since 'time' is always the
   first path segment under 'c/'. For a 1D ('time',) variable the same path
   shape holds, with no further segments beneath it.

   Parameters
   ----------
   local_store : str
      Local path of the staging store.
   output_store : str
      Final s3:// destination.
   var_name : str
      Name of the variable whose chunk just finished.
   chunk_index : int
      Index of the finished chunk along 'time'.
   """
   zarr.consolidate_metadata(local_store)

   chunk_dir = f'{var_name}/c/{chunk_index}'
   _s3_copy(
      os.path.join(local_store, chunk_dir),
      f'{output_store.rstrip("/")}/{chunk_dir}'
   )
   _s3_copy(
      os.path.join(local_store, 'zarr.json'),
      f'{output_store.rstrip("/")}/zarr.json',
      recursive=False
   )


def _write_var_3d_and_upload(
   cube, local_store, output_store, var_name, chunk_size, total_layers,
   fill_value=None, is_radar=None, num_load_workers=None
):
   """Write one 3D (time, y, x) data variable into the local staging store,
   in half-chunk-sized pieces (see INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS),
   using the raw zarr array API rather than xr.Dataset.to_zarr() --
   uploading each whole zarr chunk to S3 as soon as it's complete locally.

   Two deliberate departures from the obvious xarray implementation, both
   purely for speed/RAM -- neither changes a single byte on disk:

   1. Half-chunk writes for both dtypes. Every inner chunk spans the full
      time_chunk extent, so in theory splitting a chunk's write into N
      pieces costs N full-volume compressions + N-1 full-volume
      decompressions no matter how small the pieces are (see src/wiki/
      07_Deep_Copy_Time_Chunking_And_Write_Amplification.md), making a
      single whole-chunk write (N=1) the cheapest by that model. Measured
      (2026-09-16, in deep_copy_cube_per_var.py) that this doesn't hold for
      int variables at production scale -- see INT_CHUNK_SPLITS's own
      comment -- so both dtypes are kept at N=2 empirically rather than N=1
      for ints as the write-amplification model alone would suggest.
   2. Raw zarr writes. to_zarr(region=...) re-derives encoding from the
      store and runs the CF encoder, which allocates ~1.5x the batch in
      temporaries for every variable that declares a fill -- including int
      variables, where the transform is provably a no-op (see
      _fill_nan_in_place). Writing the array directly skips that; the
      NaN->sentinel substitution the encoder would have done is applied in
      place instead. Safe because the template already declared this array's
      full shape/dtype/chunks/shards/compressors/fill_value/attrs, so
      there's nothing left for xarray to negotiate -- only pixels to place.

   The S3 sync still happens only once a whole zarr chunk is done locally,
   never mid-chunk: a zarr chunk is atomic, so there's no value in syncing a
   half-written one.

   For var_name in RADAR_ONLY_VARS, each whole zarr chunk is checked against
   `is_radar` first: if none of its layers are radar granules the chunk is
   genuinely all-missing (see RADAR_ONLY_VARS's comment) and is skipped
   entirely -- no load, no write, no upload. The template never wrote it
   either (mode='w', compute=False defers all pixel data), so it stays
   absent on disk and reads fall back to the array's fill_value, which
   build_encoding() sets to match the variable's real CF fill -- identical
   to what a written, all-optical chunk would return.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   local_store : str
      Local path of the staging store (already templated with full
      shape/dtype/chunks -- see deep_copy_cube_per_var_chunk).
   output_store : str
      Final s3:// destination each chunk gets uploaded to as soon as it's
      done.
   var_name : str
      Name of the 3D data variable to write.
   chunk_size : int
      Size of the underlying zarr time-chunk for this variable (time_chunk).
   total_layers : int
      Total number of layers to write (honors --num-layers).
   fill_value : scalar, optional
      The variable's CF fill value as declared in the output store, used for
      the in-place NaN substitution (see _fill_nan_in_place). None if it
      declares no fill.
   is_radar : np.ndarray, optional
      Boolean array of length total_layers, True where that layer is a
      radar granule (see _compute_radar_mask). Only consulted when
      var_name is in RADAR_ONLY_VARS; pass None to disable the skip.
   num_load_workers : int, optional
      Passed straight through to _load_batch()'s dask thread-pool size.
      None (the default) leaves dask's own default in effect.

   Raises
   ------
   ValueError
      If the variable's dimension order isn't (time, y, x). Transposing to
      match would silently allocate a full extra copy of the batch -- the
      exact cost this function exists to avoid -- and the order should
      always match, since the store was templated from this same cube.
   """
   expected_dims = (utils.Coords.TIME, utils.Coords.Y, utils.Coords.X)
   if cube[var_name].dims != expected_dims:
      raise ValueError(
         f'{var_name} has dims {cube[var_name].dims}, expected '
         f'{expected_dims}; refusing to transpose (would allocate a full '
         f'extra copy of every batch)'
      )

   splits = FLOAT_CHUNK_SPLITS if cube[var_name].dtype.kind == 'f' else INT_CHUNK_SPLITS
   write_span = max(1, chunk_size // splits)
   check_radar = is_radar is not None and var_name in RADAR_ONLY_VARS
   num_chunks = 0
   num_skipped = 0

   # Opened once per variable, not per write: every write below targets the
   # same array, and the per-chunk consolidate_metadata() only rewrites the
   # root zarr.json, never this array's own metadata or chunk paths.
   target = zarr.open_group(local_store, mode='r+', zarr_format=3)[var_name]

   for chunk_start in range(0, total_layers, chunk_size):
      chunk_stop = min(chunk_start + chunk_size, total_layers)
      chunk_index = chunk_start // chunk_size
      num_chunks += 1

      if check_radar and not is_radar[chunk_start:chunk_stop].any():
         logging.info(
            f'Skipping {var_name} chunk {chunk_start}:{chunk_stop} of '
            f'{total_layers} (all-optical, no radar layers present)'
         )
         num_skipped += 1
         continue

      for start in range(chunk_start, chunk_stop, write_span):
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

      _upload_chunk(local_store, output_store, var_name, chunk_index)
      logging.info(
         f'Uploaded {var_name} chunk {chunk_start}:{chunk_stop} of '
         f'{total_layers} to {output_store}'
      )

   if check_radar and num_skipped:
      logging.info(
         f'{var_name}: skipped {num_skipped} of {num_chunks} chunk(s) '
         f'(all-optical, no radar layers present)'
      )


def _write_var_1d_and_upload(
   cube, local_store, output_store, var_name, chunk_size, total_layers,
   num_load_workers=None
):
   """Write one 1D ('time',) data variable into the local staging store in
   half-chunk-sized region writes, uploading each whole zarr chunk to S3 as
   soon as it's complete locally.

   Unlike _write_var_3d_and_upload(), this keeps xarray's to_zarr(region=...)
   path and all of its CF encoding. 1D variables are ~262k times smaller per
   layer than 3D ones (one value vs a 512x512 grid), so the encoder overhead
   _write_var_3d_and_upload() goes out of its way to avoid is irrelevant here
   -- while the encoding itself very much is not: this set includes datetime
   variables needing CF time encoding (units/calendar/epoch offsets, see
   src/wiki/) and string variables, neither of which can be written correctly
   by dropping raw values into a zarr array.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   local_store : str
      Local path of the staging store.
   output_store : str
      Final s3:// destination each chunk gets uploaded to as soon as it's
      done.
   var_name : str
      Name of the 1D data variable to write.
   chunk_size : int
      Size of the underlying zarr time-chunk for this variable
      (time_chunk_1d). Each write covers half this many layers.
   total_layers : int
      Total number of layers to write (honors --num-layers).
   num_load_workers : int, optional
      Passed straight through to _load_batch()'s dask thread-pool size.
      None (the default) leaves dask's own default in effect.
   """
   write_span = max(1, chunk_size // 2)

   for chunk_start in range(0, total_layers, chunk_size):
      chunk_stop = min(chunk_start + chunk_size, total_layers)
      chunk_index = chunk_start // chunk_size

      for start in range(chunk_start, chunk_stop, write_span):
         stop = min(start + write_span, chunk_stop)
         logging.info(f'Materializing {var_name} layers {start}:{stop} of {total_layers}')

         batch = _load_batch(cube, var_name, start, stop, num_load_workers)
         _reset_write_encoding(batch)
         batch.to_zarr(
            local_store,
            mode='r+',
            region={utils.Coords.TIME: slice(start, stop)},
            zarr_format=3,
            consolidated=False
         )

         logging.info(f'Wrote {var_name} layers {start}:{stop} of {total_layers} to {local_store}')

         del batch
         gc.collect()

      _upload_chunk(local_store, output_store, var_name, chunk_index)
      logging.info(
         f'Uploaded {var_name} chunk {chunk_start}:{chunk_stop} of '
         f'{total_layers} to {output_store}'
      )


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
   num_load_workers=None
):
   """Materialize a virtual datacube into a real zarr v3 datacube on S3, one
   data variable at a time, uploading each whole zarr chunk to S3 as soon as
   it's done rather than uploading the whole store once at the end -- see
   this module's docstring for the rationale and tradeoffs.

   Unlike deep_copy_cube_per_var.py, --output-store must be an s3:// path and
   --local-staging-dir is required: every write always lands locally first,
   then gets synced to S3 chunk by chunk.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      s3:// URL to write the deep-copy zarr store to.
   bucket_prefix : str
      S3 URL prefix the virtual chunk container resolves granule references
      against (see deep_copy_cube.open_virtual_cube).
   time_chunk : int
      Chunk size along 'time' for 3D variables.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables.
   xy_shard_multiplier : int
      Must be >= 1; 1 (the default) leaves the store unsharded. See
      deep_copy_cube.XY_SHARD_MULTIPLIER for the recommended value to pass
      explicitly.
   local_staging_dir : str
      Local directory every write lands in before being synced to
      `output_store` chunk by chunk. Required.
   keep_local_staging : bool
      If True, keep `local_staging_dir` after a successful run instead of
      removing it.
   num_layers : int
      If > 0, only materialize the first `num_layers` layers of the virtual
      cube. 0 (the default) processes every layer.
   num_load_workers : int, optional
      Thread-pool size for each _load_batch() .load() call -- each granule
      in a batch is one dask task (chunk size 1 along 'time' in the virtual
      cube), so this bounds how many granules get fetched/decompressed
      concurrently. None (the default) leaves dask's own default in effect
      (CPU core count).
   """
   if not output_store.startswith(utils.S3_PREFIX):
      raise ValueError(
         f"--output-store must be an s3:// path for this script, got {output_store}"
      )

   if not local_staging_dir:
      raise ValueError("--local-staging-dir is required for this script")

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

   # Computed once and reused for every RADAR_ONLY_VARS variable (M11, M12,
   # vr, va): they all share the same missing-ness (all-optical time ranges),
   # so there's no need to re-derive it per variable.
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

   resolve_output_store(output_store)
   local_store = resolve_output_store(local_staging_dir)

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

   # template holds a lazy (ManifestArray/dask-backed) reference to every
   # time_var across the full total_layers extent -- no pixel data, but the
   # chunk-manifest/task-graph bookkeeping for that many variables x layers
   # is non-trivial, and nothing below needs template again (later code
   # re-derives batches from `cube` directly), so free it now rather than
   # let it linger for the rest of the run.
   del template
   gc.collect()

   # Static 2D (y,x) vars: written once, full extent, no per-variable
   # chunking -- they have no 'time' dimension to chunk over.
   static_batch = cube[static_vars].load()
   _reset_write_encoding(static_batch)
   static_batch.to_zarr(local_store, mode='r+', zarr_format=3, consolidated=False)
   logging.info(f'Wrote {len(static_vars)} static variable(s) to {local_store}')

   # Upload the store's skeleton (root zarr.json + every array's own
   # zarr.json + coordinate/static variable data) once, up front. None of
   # this changes again after this point -- shape/dtype/chunks/encoding are
   # all fixed by the template call above -- but S3 needs it in place before
   # any per-chunk data lands, so any reader hitting the store mid-run
   # always finds valid metadata (unwritten chunks fall back to fill_value,
   # same as skipped RADAR_ONLY_VARS chunks).
   zarr.consolidate_metadata(local_store)
   _s3_copy(local_store, output_store)
   logging.info(f'Uploaded store skeleton to {output_store}')

   for var_name in vars_1d:
      _write_var_1d_and_upload(
         cube, local_store, output_store, var_name, time_chunk_1d, total_layers,
         num_load_workers
      )

   for var_name in vars_3d:
      # The CF fill this variable's array was templated with -- only floats
      # get one under build_encoding()'s convention (ints use the separate
      # 'missing_value' key, and raw ints can't be NaN anyway, so
      # _fill_nan_in_place has nothing to do for them).
      _write_var_3d_and_upload(
         cube, local_store, output_store, var_name, time_chunk, total_layers,
         encoding.get(var_name, {}).get(utils.OutputFormat.fill_value),
         is_radar, num_load_workers
      )

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
      EXPERIMENTAL: materialize a virtual ITS_LIVE datacube (icechunk repo
      built by virtual_itslive_cube_per_chunk.py) into a real Zarr v3
      datacube on S3, one data variable at a time, uploading each whole zarr
      chunk to S3 as soon as it's written locally -- instead of
      deep_copy_cube_per_var.py's single upload at the very end. See this
      module's docstring for the rationale.

      Usage example:
      python src/deep_copy_cube_per_var_chunk.py \
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
      help='s3:// URL to write the deep-copy Zarr v3 store to.'
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
      default=1,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables. A value of 1 '
         f'(the default) disables sharding. Recommended value once sharding '
         f'is enabled: {XY_SHARD_MULTIPLIER} [%(default)d].'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      required=True,
      help='Local directory every write lands in before being synced to '
         '--output-store one whole zarr chunk at a time. Required.'
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
      args.num_load_workers
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
