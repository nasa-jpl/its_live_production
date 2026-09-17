"""
Materialize a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube, batched by
DATA VARIABLE rather than deep_copy_cube.py's time-only batching or a
spatial-tile scheme.

Why: deep_copy_cube.py's batch_size (default 2000 layers) is far smaller
than the output store's fixed TIME_CHUNK_VALUE (20000), so every batch after
the first forces a full decompress/merge/recompress of the entire
still-open time-chunk across every spatial chunk/shard AND every variable
simultaneously -- quadratic write amplification, root-caused in
src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md. Matching
batch_size to time_chunk (the wiki's preferred fix) eliminates that, but
needs ~256 GiB RAM to hold every variable's full time-chunk, at full spatial
extent, at once.

Spatial tiling shrinks that RAM need by tiling the *spatial* extent instead
(tried and retired -- documented as Option 4 in
src/wiki/07_Deep_Copy_Time_Chunking_And_Write_Amplification.md), but each
granule's virtual chunk spans the full spatial grid (no source-side
sub-tiling), so tiling the write does not tile the read: every tile
re-fetches/decompresses each touched granule's full spatial extent,
discarding the unwanted pixels afterward -- real, measured read
amplification.

This script shrinks RAM the other way: keep full x/y extent and full
time-chunk (no unbounded write amplification, same mechanism as the wiki's
Option 1), but process ONE DATA VARIABLE AT A TIME instead of all variables
at once. Peak RAM becomes one variable's chunk footprint instead of the
whole cube's. Unlike spatial tiling, this does not read-amplify: each
granule's per-variable chunk is already an independently addressable virtual
reference, so reading one variable at a time does not re-fetch bytes
another variable's read already covered.

Both 3D variable classes write their whole zarr time-chunk in two
half-chunk-sized pieces (see INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS) rather
than one whole-chunk write. In theory a single whole-chunk write should be
strictly cheaper on the write side -- every inner chunk spans the full
time_chunk extent, so an N-piece write costs N full-volume compressions +
N-1 full-volume decompressions regardless of piece size, making N=1 always
fastest by that model, with RAM as the only reason to split. Measured
(2026-09-16) at production scale, that theory didn't hold for int
variables: going whole-chunk regressed them 2.4x despite confirmed-clean
RAM/swap throughout, so INT_CHUNK_SPLITS is set empirically to match
FLOAT_CHUNK_SPLITS rather than by the write-amplification model above --
see INT_CHUNK_SPLITS's own comment for the measurement and the still-open
question of what actually dominates at that scale. _write_var_3d() also
bypasses xarray's CF encoder (which would otherwise allocate ~1.5x the
batch in temporaries per write, for every variable that declares a fill --
including int ones, where the transform it performs is provably a no-op).

KNOWN TRADEOFFS -- read this before using this script for a production run:
- Processing variables one at a time forgoes any inter-variable parallelism
  in the read path (S3 GETs for variable N+1 don't start until variable N's
  full chunk has been written), so total wall-clock time is closer to
  (num_variables x per-variable read+write time) than deep_copy_cube.py's
  more overlapped batch processing.
- FLOAT_CHUNK_SPLITS is sized for a 32 GB instance and trades peak RAM
  against write amplification (raise it on a smaller box; on a larger one,
  dropping it to 1 removes float's remaining write amplification --
  re-benchmark first given INT_CHUNK_SPLITS's finding above). INT_CHUNK_SPLITS
  is set empirically, not by the same RAM tradeoff -- see its own comment.
- 3D variables are written with the raw zarr array API instead of
  xr.Dataset.to_zarr(region=...), so they get none of xarray's write-time
  validation (dimension/coordinate consistency, region alignment). This is
  safe because the template declares every array's full
  shape/dtype/chunks/shards/compressors/fill_value/attrs up front and the
  region bounds are pure chunk arithmetic -- but it does mean a future
  change to either would fail later and less clearly than it would have
  under xarray. 1D variables deliberately keep the xarray path (they need
  CF datetime/string encoding, and are far too small for the overhead to
  matter).
- M11/M12/vr/va (RADAR_ONLY_VARS) skip a whole zarr chunk entirely -- no
  load, no write -- whenever none of that chunk's layers are radar granules
  (see _compute_radar_mask). Datacubes span optical-only history back to
  1984, well before Sentinel-1 (2014), so a full 20000-layer chunk with no
  radar layers at all is a real, common case, not a corner case. Skipped
  chunks read back via the store's fill_value, which is set to match each
  variable's real CF fill -- correctness-equivalent to writing an
  all-optical chunk, just without paying to write it.
These are deliberate RAM-vs-wall-clock tradeoffs -- benchmark with
--num-layers on a bounded slice before relying on this for a full
production run.

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


# Number of equal sub-writes each whole zarr time-chunk is split into for a
# 3D variable, by dtype. In *theory* every inner chunk spans the FULL
# time_chunk extent (chunks are (time_chunk, xy_chunk, xy_chunk)), so
# splitting a chunk's write into N pieces should cost N full-volume
# compressions + N-1 full-volume decompressions regardless of piece size --
# making N=1 (whole chunk) always fastest, with RAM as the only reason to
# split.
#
# MEASURED (2026-09-16) this theory doesn't hold at production scale: a
# whole-chunk (20000-layer) write of an int variable (chip_size_height, no
# CF-encoder cost either way -- see _fill_nan_in_place) took 632s, vs 268s
# total for the same 20000 layers as two half-chunk (10000-layer) writes on
# the same run -- 2.4x SLOWER despite strictly less write-side work by the
# theory above. RAM/swap was confirmed clean throughout (free -m showed
# ~20 GiB available, 0 swap), so it isn't the RAM tradeoff this constant was
# designed around -- something about a single, much bigger .load() call
# (dask task-graph construction, or S3 request-pattern effects) dominates at
# this scale instead. Root cause not yet isolated; INT_CHUNK_SPLITS is set
# empirically (matching FLOAT_CHUNK_SPLITS) rather than by the theory above
# until it is. M11/M12 (float, FLOAT_CHUNK_SPLITS unchanged at 2 across this
# investigation) got faster in the same run, from the encoder-bypass +
# radar-skip wins alone -- confirming the regression is specific to the
# int/whole-chunk change, not the raw-zarr-write mechanism itself.
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

# Mission groups whose granules are radar (SAR)
RADAR_GROUP_IDS = {sensors.SENTINEL1.id, sensors.NISAR.id}


def split_time_vars_by_rank(cube, time_vars):
   """Further split split_vars_by_time()'s time_vars into 3D (time,y,x) and
   1D (time,) groups.

   The two ranks take entirely different write paths here: 3D variables go
   through _write_var_3d() (raw zarr writes, whole-chunk sized, radar-skip
   aware) while 1D variables go through _write_var_1d() (xarray's
   to_zarr(region=...), which they need for CF datetime/string encoding).
   They also use different time-chunk sizes -- time_chunk vs time_chunk_1d.

   Originally lived in deep_copy_cube_tiled.py, which needed the same split
   for a different reason: 1D vars have no x/y dimension, so its spatial-tile
   loop had to exclude them to avoid redundantly recompressing their single
   (time_chunk_1d,)-sized chunk once per tile.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   time_vars : list of str
      Output of deep_copy_cube.split_vars_by_time()'s first return value.

   Returns
   -------
   tuple of (list of str, list of str)
      (vars_3d, vars_1d) data variable names.
   """
   vars_3d = [v for v in time_vars if len(cube[v].dims) == 3]
   vars_1d = [v for v in time_vars if len(cube[v].dims) != 3]
   return vars_3d, vars_1d


@itslive_utils.retry_decorator(max_retries=5)
def _load_batch(cube, var_name, start, stop):
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
      Time-slice bounds (see _write_var_3d/_write_var_1d).

   Returns
   -------
   xr.Dataset
      The loaded (non-dask) batch for this variable and time slice.
   """
   return cube[[var_name]].isel(
      {utils.Coords.TIME: slice(start, stop)}
   ).drop_vars(
      [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X], errors='ignore'
   ).load()


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
   genuinely carries M11/M12/vr/va (true today; see this function's
   docstring caller for the one-time validation this assumes was done).

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


def _write_var_3d(
   cube, write_target, var_name, chunk_size, total_layers,
   fill_value=None, is_radar=None
):
   """Write one 3D (time, y, x) data variable's full extent into an
   already-templated store, in half-chunk-sized pieces (see
   INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS), using the raw zarr array API rather
   than xr.Dataset.to_zarr().

   Two deliberate departures from the obvious xarray implementation, both
   purely for speed/RAM -- neither changes a single byte on disk:

   1. Half-chunk writes for both dtypes. Every inner chunk spans the full
      time_chunk extent, so in theory splitting a chunk's write into N
      pieces costs N full-volume compressions + N-1 full-volume
      decompressions no matter how small the pieces are (see src/wiki/
      07_Deep_Copy_Time_Chunking_And_Write_Amplification.md), making a
      single whole-chunk write (N=1) the cheapest by that model. Measured
      (2026-09-16) that this doesn't hold for int variables at production
      scale -- see INT_CHUNK_SPLITS's own comment -- so both dtypes are kept
      at N=2 empirically rather than N=1 for ints as the write-amplification
      model alone would suggest.
   2. Raw zarr writes. to_zarr(region=...) re-derives encoding from the
      store and runs the CF encoder, which allocates ~1.5x the batch in
      temporaries for every variable that declares a fill -- including int
      variables, where the transform is provably a no-op (see
      _fill_nan_in_place). Writing the array directly skips that; the
      NaN->sentinel substitution the encoder would have done is applied in
      place instead. Safe because the template already declared this array's
      full shape/dtype/chunks/shards/compressors/fill_value/attrs, so
      there's nothing left for xarray to negotiate -- only pixels to place.

   For var_name in RADAR_ONLY_VARS, each whole zarr chunk is checked against
   `is_radar` first: if none of its layers are radar granules the chunk is
   genuinely all-missing (see RADAR_ONLY_VARS's comment) and is skipped
   entirely -- no load, no write. The template never wrote it either
   (mode='w', compute=False defers all pixel data), so it stays absent on
   disk and reads fall back to the array's fill_value, which
   build_encoding() sets to match the variable's real CF fill -- identical
   to what a written, all-optical chunk would return. The check is at whole-
   chunk granularity because a zarr chunk is atomic; there's no such thing
   as skipping part of one.

   Explicitly deletes the loaded batch and forces a gc pass after each
   write: xarray/dask Datasets commonly hold internal reference cycles
   (e.g. task-graph closures), which CPython's refcounting alone won't
   collect promptly -- on a RAM-constrained instance, leaving a finished
   write's array pending cyclic collection until Python gets around to it
   can crowd out the next write's read/decompress footprint.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   write_target : str
      Local path or s3:// URL of the store to write into (already
      templated with full shape/dtype/chunks -- see deep_copy_cube_per_var).
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
   # same array.
   target = zarr.open_group(write_target, mode='r+', zarr_format=3)[var_name]

   for chunk_start in range(0, total_layers, chunk_size):
      chunk_stop = min(chunk_start + chunk_size, total_layers)
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

         batch = _load_batch(cube, var_name, start, stop)
         values = batch[var_name].values
         _fill_nan_in_place(values, fill_value)
         target[start:stop, :, :] = values

         logging.info(f'Wrote {var_name} layers {start}:{stop} of {total_layers} to {write_target}')

         del values
         del batch
         gc.collect()

   if check_radar and num_skipped:
      logging.info(
         f'{var_name}: skipped {num_skipped} of {num_chunks} chunk(s) '
         f'(all-optical, no radar layers present)'
      )


def _write_var_1d(cube, write_target, var_name, chunk_size, total_layers):
   """Write one 1D ('time',) data variable's full extent into an
   already-templated store, in half-chunk-sized region writes.

   Unlike _write_var_3d(), this keeps xarray's to_zarr(region=...) path and
   all of its CF encoding. 1D variables are ~262k times smaller per layer
   than 3D ones (one value vs a 512x512 grid), so the encoder overhead
   _write_var_3d() goes out of its way to avoid is irrelevant here -- while
   the encoding itself very much is not: this set includes datetime
   variables needing CF time encoding (units/calendar/epoch offsets, see
   src/wiki/) and string variables, neither of which can be written
   correctly by dropping raw values into a zarr array.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.
   write_target : str
      Local path or s3:// URL of the store to write into.
   var_name : str
      Name of the 1D data variable to write.
   chunk_size : int
      Size of the underlying zarr time-chunk for this variable
      (time_chunk_1d). Each write covers half this many layers.
   total_layers : int
      Total number of layers to write (honors --num-layers).
   """
   write_span = max(1, chunk_size // 2)

   for start in range(0, total_layers, write_span):
      stop = min(start + write_span, total_layers)
      logging.info(f'Materializing {var_name} layers {start}:{stop} of {total_layers}')

      batch = _load_batch(cube, var_name, start, stop)
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
   variable at a time, at full spatial extent -- 3D variables in
   half-time_chunk increments (see INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS), 1D
   variables in half-time_chunk_1d increments. See this module's docstring
   for the RAM-vs-wall-clock tradeoffs this makes relative to
   deep_copy_cube.py.

   Unlike deep_copy_cube.py's incremental append-based construction, this
   writes the whole store's shape/dtype/chunk-grid up front (mode='w',
   compute=False) and fills it in per (variable, time-chunk) afterwards.
   Static 2D (y,x) vars are written once, at full extent.

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
      Chunk size along 'time' for 3D variables. Each write for a 3D
      variable materializes half this many layers (see
      INT_CHUNK_SPLITS/FLOAT_CHUNK_SPLITS).
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables. Each write for a 1D variable
      materializes half this many layers.
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
   static_batch.to_zarr(write_target, mode='r+', zarr_format=3, consolidated=False)
   logging.info(f'Wrote {len(static_vars)} static variable(s) to {write_target}')

   for var_name in vars_1d:
      _write_var_1d(cube, write_target, var_name, time_chunk_1d, total_layers)

   for var_name in vars_3d:
      # The CF fill this variable's array was templated with -- only floats
      # get one under build_encoding()'s convention (ints use the separate
      # 'missing_value' key, and raw ints can't be NaN anyway, so
      # _fill_nan_in_place has nothing to do for them).
      _write_var_3d(
         cube, write_target, var_name, time_chunk, total_layers,
         encoding.get(var_name, {}).get(utils.OutputFormat.fill_value),
         is_radar
      )

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
      data variable at a time at full spatial extent -- 3D variables in
      time_chunk/2 writes, 1D variables in time_chunk_1d/2 writes. Avoids
      deep_copy_cube.py's unbounded
      batch_size-vs-time_chunk write amplification and the spatial
      read-amplification a tiled write incurs, at the cost of losing
      inter-variable read/write overlap (see this module's docstring for the
      accepted tradeoffs).

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
      help='Chunk size for 1D (time,) variables. Each write for a 1D '
         'variable materializes half this many layers [%(default)d].'
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
