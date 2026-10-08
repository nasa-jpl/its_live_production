"""
Validate a deep-copied ITS_LIVE Zarr v3 datacube at a given S3 location for
structural/metadata consistency across every variable and layer.

Why: deep_copy_cube_per_var_chunk.py uploads each variable's zarr chunk to
S3 as soon as it's done locally, one `aws s3 cp --recursive` per (variable,
time-chunk) -- see _upload_chunk() there. That copy is NOT atomic across a
sharded chunk's many shard files: a run killed mid-copy (spot termination,
network blip) can leave some shard files on S3 and others missing for the
same chunk. A plain "does the store open" check can't catch this -- Zarr
resolves ANY missing chunk/shard to the array's fill_value with no error,
so a partially-uploaded chunk reads back exactly like a legitimately absent
one (see RADAR_ONLY_VARS's whole-chunk skip in deep_copy_cube_per_var.py).
This script instead lists the actual S3 objects under each array's chunk
grid and cross-checks that against what the array's OWN declared shape/
chunk-grid says should be there.

Checks performed, per data variable/coordinate array in the store:
1. Root zarr.json + consolidated metadata are present, and the consolidated
   member list agrees with a live (non-consolidated) listing of the same
   group -- catches consolidated metadata going stale (e.g. a crash between
   writing a chunk and the zarr.consolidate_metadata() call that follows
   it).
2. Every variable's shape agrees with the shared 'time'/'y'/'x' coordinate
   lengths (no per-variable drift).
3. 'time' is sorted (non-decreasing; duplicate mid_dates are a normal,
   valid state, so this is NOT a strict-increase check) with no NaT
   values (ERROR); a small out-of-order swap is reported as a WARNING, not
   an ERROR -- expected given granules are sorted by a filename-derived
   mid_date that can disagree in precision with the true value stored here
   (see _check_coordinate_sanity()'s docstring). 'x'/'y' have no NaN
   (ERROR).
4. Storage-chunk completeness: for every array with a non-empty chunk grid,
   every expected chunk index (computed from the array's own shape and
   effective storage-chunk shape -- shards, if sharded, else chunks) must be
   either FULLY present (every leaf object under it written) or fully
   ABSENT. A PARTIALLY present chunk (some but not all of its expected leaf
   objects) is always an ERROR, for any variable -- zarr's
   write_empty_chunks=False default (see below) either omits a chunk
   entirely or writes it whole, never partially, so a partial chunk is
   never legitimate and is the direct fingerprint of an interrupted upload.
   A fully ABSENT chunk is graded by how confidently its legitimacy can be
   verified:
   - RADAR_ONLY_VARS (M11/M12/vr/va), their derived variants (e.g.
     vr_error_slow, va_stable_shift_mask, M11_dr_to_vr_factor), and
     EXTRA_RADAR_ONLY_VARS (ascending_img1/ascending_img2 -- see
     _is_radar_only()): cross-checked against mission_img1/satellite_img1
     using the exact same classification deep_copy_cube_per_var(_chunk).py's
     radar-skip uses (see deep_copy_cube_per_var.RADAR_ONLY_VARS/
     RADAR_GROUP_IDS) -- reported as INFO (verified legitimate) or ERROR
     (verified NOT legitimate).
   - every other variable: reported as a WARNING, not an ERROR -- zarr's
     write_empty_chunks=False default (the global default as of zarr-
     python 3.x, see zarr/core/config.py) never writes a chunk whose real
     data is uniformly equal to the array's fill_value, and this script has
     no source data to confirm that one way or the other. Observed (Sep
     2026) on one store for landice/floatingice with fill_value=0 (which
     doubles as the real "no ice here" mask value there) -- but
     deep_copy_cube.build_encoding() mirrors whatever fill value the source
     virtual cube's own array metadata carries, so this is NOT guaranteed
     to be 0 for these two variables on every store (confirmed current
     deep_copy_cube_per_var(_chunk).py output instead carries
     missing_value=fill_value=255 for them, e.g. -- outside their real 0/1
     value range, which would make an ABSENT chunk suspicious rather than
     plausible). Read the array's own fill_value in the finding's message
     before assuming this is benign.

This intentionally never reads variable pixel data (beyond the tiny
coordinate/classification arrays needed for checks 2/3/4 above) -- it's a
structural/metadata check, not a byte-for-byte data validator. A store that
passes every check here is guaranteed to have every declared chunk either
genuinely intact or genuinely (and correctly) skipped; it is NOT a guarantee
against codec-level corruption of an otherwise-complete object.

Usage example:
python src/tools/validate_s3_cube.py \
   --cube-url s3://its-live-data/path/to/my_deep_copy_cube.zarr
"""
import argparse
import itertools
import logging
import random
import sys
from math import ceil

import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr

import itslive_utils
import utils
from itscube_types import ImgPairInfo, Vars
from sensorFilters import SensorExcludeFilter
from deep_copy_cube_per_var_chunk import RADAR_ONLY_VARS, RADAR_GROUP_IDS

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


def _add(findings, level, variable, message):
   findings.append({'level': level, 'variable': variable, 'message': message})


@itslive_utils.retry_decorator(max_retries=5)
def _find(fs, path):
   """List every leaf object under `path`, retrying on transient S3
   errors. Returns an empty list (not an error) if `path` doesn't exist --
   an absent chunk-grid prefix is a normal, expected state (e.g. a
   variable with no chunks written yet, or a legitimately fully-skipped
   RADAR_ONLY_VARS chunk), not a failure to distinguish from a real one."""
   return fs.find(path)


# Radar-only variables that aren't a vr/va/M11/M12 derivative -- flight
# direction ('ascending') only exists in a granule's img_pair_info for a
# radar (SAR) mission (see itscube_types.ImgPairInfo's 'Attributes for
# radar granules' grouping); optical granules carry the 255 missing_value
# placeholder instead, same as RADAR_ONLY_VARS.
EXTRA_RADAR_ONLY_VARS = {Vars.ascending_img1, Vars.ascending_img2}


# Direct per-pixel copies of a granule data variable, present for every
# granule regardless of sensor type.
DIRECT_ARRAY_VARS = (
   Vars.v, Vars.vx, Vars.vy, Vars.v_error,
   Vars.chip_size_height, Vars.chip_size_width, Vars.interp_mask,
)
# Direct per-pixel copies that only exist for radar granules (see
# RADAR_ONLY_VARS above).
RADAR_ARRAY_VARS = (Vars.va, Vars.vr, Vars.m11, Vars.m12)

# Attribute suffixes read off a vx/vy/va/vr granule variable that become
# "{var}_{suffix}" scalar cube variables (see itscube.py's
# process_v_attributes()). 'stable_shift' is handled separately since a NaN
# granule value is expected to read back as 0 in the cube.
V_ERROR_SUFFIXES = (
   Vars.postfix.error, Vars.postfix.error_mask,
   Vars.postfix.error_modeled, Vars.postfix.error_slow,
)
V_SHIFT_SUFFIXES = (Vars.postfix.stable_shift_mask, Vars.postfix.stable_shift_slow)

# Tolerance for comparing CF-decoded (physical) float values between a
# granule and the cube. Deliberately not exact equality: the cube's own
# scale_factor/add_offset is allowed to legitimately differ from the source
# granule's, so this check cares about the physical value being right, not
# byte-for-byte encoding equality.
ARRAY_RTOL = 1e-5
ARRAY_ATOL = 1e-3
SCALAR_RTOL = 1e-5
SCALAR_ATOL = 1e-6



def _is_radar_only(var_name):
   """True if `var_name` is RADAR_ONLY_VARS itself, a derived variant of
   one (e.g. 'vr_error_slow', 'va_stable_shift_mask', 'M11_dr_to_vr_factor')
   -- these carry real data only for radar granules too, since they're
   attributes/derivatives of a radar-only base variable (see itscube.py's
   new_v_vars / virtual_itslive_cube.py's _extract_velocity_attributes) --
   or in EXTRA_RADAR_ONLY_VARS."""
   if var_name in EXTRA_RADAR_ONLY_VARS:
      return True
   return any(
      var_name == base or var_name.startswith(f'{base}_')
      for base in RADAR_ONLY_VARS
   )


def _storage_chunk_shape(array):
   """The shape of one on-disk storage unit for `array`: the shard shape if
   sharded (the S3 object boundary sharding actually writes), else the
   plain chunk shape. Either way this is the granularity that determines
   the 'c/{i}/{j}/...' path structure -- see zarr v3's sharding_indexed
   codec, which nests multiple logical chunks inside one physical shard
   file without adding another path segment."""
   return array.shards if array.shards is not None else array.chunks


def _expected_indices(shape, storage_chunk_shape):
   """Every expected chunk index for an array of `shape` with storage units
   of `storage_chunk_shape`, as a set of index tuples. A 0-d (scalar)
   array has no chunk grid at all (see build_encoding()'s own skip for
   scalar variables like 'mapping') -- represented here as an empty set,
   meaning "nothing to check", not "everything missing"."""
   if len(shape) == 0:
      return set()
   ranges = [range(ceil(s / c)) for s, c in zip(shape, storage_chunk_shape)]
   return set(itertools.product(*ranges))


def _actual_indices(fs, base, var_name):
   """Every chunk index actually present on S3 for `var_name`, as a set of
   index tuples, derived from a flat listing of every leaf object under
   '{var_name}/c/'."""
   prefix = f'{base}/{var_name}/c/'
   keys = _find(fs, f'{base}/{var_name}/c')
   indices = set()
   for key in keys:
      rel = key[len(prefix):] if key.startswith(prefix) else key.rsplit(f'/{var_name}/c/', 1)[-1]
      indices.add(tuple(int(part) for part in rel.split('/')))
   return indices


def _check_chunk_completeness(fs, base, var_name, array, is_radar, radar_status, findings):
   """Check chunk-grid completeness for one array (see module docstring,
   item 4). Appends ERROR/WARNING/INFO findings.

   Parameters
   ----------
   is_radar : np.ndarray or None
      Boolean mask over the full 'time' extent, True where that layer is a
      radar granule (see deep_copy_cube_per_var._compute_radar_mask).
      None if it couldn't be computed (see `radar_status`).
   radar_status : str
      'ok' if `is_radar` is trustworthy, otherwise a short reason
      (e.g. 'mission_img1/satellite_img1 incomplete') used to downgrade an
      otherwise-legitimate-looking radar skip to a WARNING instead of an
      OK, since it can no longer be verified.
   """
   shape = array.shape
   chunk_shape = _storage_chunk_shape(array)
   expected = _expected_indices(shape, chunk_shape)
   if not expected:
      return

   actual = _actual_indices(fs, base, var_name)
   is_radar_var = _is_radar_only(var_name)

   # Group both expected and actual indices by their leading index -- that's
   # the only axis this pipeline's skip/partial-upload logic ever operates
   # on (see RADAR_ONLY_VARS's whole *time*-chunk skip). For every array
   # this pipeline writes, the leading dimension genuinely is 'time' (3D/1D
   # time-indexed variables); for the handful of non-time-indexed arrays
   # (static 2D vars, which are always a single whole-extent chunk) the
   # label below is a harmless misnomer rather than a correctness issue --
   # named from the array's own dimension_names rather than hardcoded so it
   # reads accurately either way.
   dims = array.metadata.dimension_names
   leading_dim = dims[0] if dims else 'axis-0'
   expected_by_time = {}
   for idx in expected:
      expected_by_time.setdefault(idx[0], set()).add(idx)
   actual_by_time = {}
   for idx in actual:
      actual_by_time.setdefault(idx[0], set()).add(idx)

   for time_idx in sorted(expected_by_time):
      expected_set = expected_by_time[time_idx]
      actual_set = actual_by_time.get(time_idx, set())
      missing = expected_set - actual_set
      unexpected = actual_set - expected_set
      if unexpected:
         _add(
            findings, 'WARNING', var_name,
            f'{leading_dim}-chunk {time_idx}: {len(unexpected)} unexpected '
            f'chunk object(s) outside the declared shape (stale data from '
            f'an earlier/different run?): {sorted(unexpected)[:5]}'
         )

      if not missing:
         continue

      if len(missing) < len(expected_set):
         _add(
            findings, 'ERROR', var_name,
            f'{leading_dim}-chunk {time_idx}: PARTIAL '
            f'({len(missing)}/{len(expected_set)} objects missing) -- '
            f'interrupted upload.'
         )
         continue

      # Fully missing for this time_idx: only legitimate for radar-only vars
      # (RADAR_ONLY_VARS and their derivatives, see _is_radar_only()), and
      # only when genuinely all-optical over this chunk's time range.
      chunk_size = chunk_shape[0]
      start, stop = time_idx * chunk_size, min((time_idx + 1) * chunk_size, shape[0])

      if not is_radar_var:
         _add(
            findings, 'WARNING', var_name,
            f'{leading_dim}-chunk {time_idx} ({start}:{stop}) missing -- '
            f'not a radar-only var; plausibly all-fill '
            f'(fill_value={array.metadata.fill_value!r}), unverified.'
         )
      elif radar_status != 'ok':
         _add(
            findings, 'WARNING', var_name,
            f'{leading_dim}-chunk {time_idx} ({start}:{stop}) missing -- '
            f'radar-only var, could not verify (mission/satellite_img1: '
            f'{radar_status}).'
         )
      elif not is_radar[start:stop].any():
         _add(
            findings, 'INFO', var_name,
            f'{leading_dim}-chunk {time_idx} ({start}:{stop}) missing -- '
            f'verified: no radar granules in range.'
         )
      else:
         _add(
            findings, 'ERROR', var_name,
            f'{leading_dim}-chunk {time_idx} ({start}:{stop}) missing but '
            f'{int(np.count_nonzero(is_radar[start:stop]))} radar granule(s) '
            f'in range expect real data.'
         )


def _compute_is_radar(group, total_layers, findings):
   """Load mission_img1/satellite_img1 from the (already deep-copied, real)
   store and classify each of the first `total_layers` layers as radar or
   not -- the same classification deep_copy_cube_per_var(_chunk).py's
   radar-skip uses, computed here from the OUTPUT store directly rather
   than a virtual cube (these two variables are baked-in 1D data, not
   manifest-array references -- see feedback on deep_copy_cube_per_var.py's
   1D-variable handling -- so no icechunk/S3-granule access is needed).

   Returns
   -------
   tuple of (np.ndarray or None, str)
      (is_radar, status). status == 'ok' iff is_radar is trustworthy;
      otherwise is_radar is None and status names the reason (chunk
      completeness for these two variables should be checked separately
      and will have already produced its own ERROR findings if broken).
   """
   for var_name in (ImgPairInfo.mission_img1, ImgPairInfo.satellite_img1):
      if var_name not in group.array_keys():
         return None, f'{var_name} not present in store'

   try:
      mission = group[ImgPairInfo.mission_img1][:total_layers]
      satellite = group[ImgPairInfo.satellite_img1][:total_layers]
      group_ids = SensorExcludeFilter.map_sensor_to_group(satellite, mission)
   except Exception as exc:
      _add(
         findings, 'WARNING', 'mission_img1/satellite_img1',
         f'failed to classify granules for the radar-skip legitimacy check: '
         f'{exc}'
      )
      return None, f'classification failed: {exc}'

   return np.isin(group_ids, list(RADAR_GROUP_IDS)), 'ok'


def _check_consolidated_metadata(cube_url, findings):
   """Compare the store's consolidated member list/shapes/dtypes against a
   live (non-consolidated) listing of the same group -- catches
   consolidated metadata left stale by a crash between writing a chunk and
   the zarr.consolidate_metadata() call that's supposed to follow it.

   Returns
   -------
   zarr.Group or None
      The consolidated-mode group, for reuse by later checks, or None if
      the store couldn't be opened at all (fatal -- callers should stop).
   """
   try:
      consolidated = zarr.open_group(
         cube_url, mode='r', zarr_format=3, storage_options={'anon': True},
         use_consolidated=True
      )
   except Exception as exc:
      _add(findings, 'ERROR', '(root)', f'failed to open store with consolidated metadata: {exc}')
      return None

   try:
      live = zarr.open_group(
         cube_url, mode='r', zarr_format=3, storage_options={'anon': True},
         use_consolidated=False
      )
   except Exception as exc:
      _add(findings, 'ERROR', '(root)', f'failed to open store without consolidated metadata: {exc}')
      return consolidated

   consolidated_keys = set(consolidated.array_keys())
   live_keys = set(live.array_keys())
   if consolidated_keys != live_keys:
      _add(
         findings, 'ERROR', '(root)',
         f'consolidated metadata is stale: consolidated-only='
         f'{sorted(consolidated_keys - live_keys)}, live-only='
         f'{sorted(live_keys - consolidated_keys)}'
      )

   for var_name in consolidated_keys & live_keys:
      c_arr, l_arr = consolidated[var_name], live[var_name]
      if c_arr.shape != l_arr.shape or c_arr.dtype != l_arr.dtype:
         _add(
            findings, 'ERROR', var_name,
            f'consolidated metadata disagrees with live metadata: '
            f'consolidated shape/dtype={c_arr.shape}/{c_arr.dtype} vs. '
            f'live={l_arr.shape}/{l_arr.dtype}'
         )

   return consolidated


def _check_coordinate_sanity(ds, findings):
   """'time' should be sorted (non-decreasing) with no NaT -- NOT strictly
   increasing: multiple granule pairs commonly share the same mid_date, so
   duplicates are a normal, valid state, not corruption. 'x'/'y' must have
   no NaN (see tools/check_cubes_dims_nan.py for the historical version of
   this same check against older zarr v2 datacubes).

   Out-of-order 'time' is reported as a WARNING, not an ERROR: granules are
   sorted at cube-creation time by a filename-derived mid_date (see
   itscube.py's extract_mid_date_from_url()), which can disagree in
   precision with the true, full-precision mid_date stored in 'time' --
   confirmed (Sep 2026) this produces real, small (sub-minute) out-of-order
   swaps between near-tied granules, which is an accepted, expected
   consequence of that sort key choice, not corruption. A large-magnitude
   swap would look identical to this check, so a WARNING (surfaced for a
   human to glance at) rather than silence is still the right call.

   On an out-of-order finding, reports the first offending pair's layer
   indices, their 'time' values, and (when granule_url is present and
   intact) their source granule URLs -- enough to go look at the actual
   granules rather than just knowing *that* something's out of order."""
   time_values = ds[utils.Coords.TIME].values
   if np.any(np.isnat(time_values)):
      _add(findings, 'ERROR', utils.Coords.TIME, 'contains NaT value(s)')
   elif len(time_values) > 1:
      deltas = np.diff(time_values)
      out_of_order = np.where(deltas < np.timedelta64(0, 'ns'))[0]
      if out_of_order.size:
         i = int(out_of_order[0])
         message = (
            f'is not sorted ({out_of_order.size} out-of-order pair(s) total).\n'
            f'  layer {i}:     {time_values[i]!r}\n'
            f'  layer {i + 1}: {time_values[i + 1]!r} (earlier than the previous layer)\n'
            f'  Expected to some degree -- granules are sorted by a '
            f'filename-derived mid_date, which can disagree in precision '
            f'with the true value stored here (see this function\'s docstring).'
         )
         if Vars.url in ds:
            try:
               urls = ds[Vars.url].values
               message += (
                  f'\n  granule_url[{i}]:     {urls[i]!r}\n'
                  f'  granule_url[{i + 1}]: {urls[i + 1]!r}'
               )
            except Exception as exc:
               message += f'\n  (failed to read granule_url for detail: {exc})'
         _add(findings, 'WARNING', utils.Coords.TIME, message)

   for coord_name in (utils.Coords.X, utils.Coords.Y):
      if coord_name in ds.coords and np.any(np.isnan(ds[coord_name].values)):
         _add(findings, 'ERROR', coord_name, 'contains NaN value(s)')


def _check_shape_consistency(group, findings):
   """Every variable's declared shape must agree with the shared
   'time'/'y'/'x' coordinate lengths -- catches a variable templated
   against a different total_layers/xy extent than the rest of the store
   (e.g. a partial re-run with a different --num-layers/--xy-chunk-value
   accidentally pointed at the same output store)."""
   dim_lengths = {}
   for coord_name in (utils.Coords.TIME, utils.Coords.Y, utils.Coords.X):
      if coord_name in group.array_keys():
         dim_lengths[coord_name] = group[coord_name].shape[0]

   for var_name in group.array_keys():
      if var_name in dim_lengths:
         continue

      array = group[var_name]
      dims = array.metadata.dimension_names
      if not dims:
         continue

      for axis, dim_name in enumerate(dims):
         expected_len = dim_lengths.get(dim_name)
         if expected_len is not None and array.shape[axis] != expected_len:
            _add(
               findings, 'ERROR', var_name,
               f"dimension '{dim_name}' has length {array.shape[axis]}, "
               f"expected {expected_len} (from the '{dim_name}' coordinate)"
            )


@itslive_utils.retry_decorator(max_retries=5)
def _open_granule(granule_url, fs):
   """Open one source granule from S3 exactly as ITSCube.read_s3_dataset()
   does (see itscube.py) -- https:// URL -> s3:// path, h5netcdf, default
   (CF-decoding) mask_and_scale. Loaded into memory so the file handle can
   be closed before this returns."""
   s3_path = granule_url.replace(utils.HTTP_PREFIX, utils.S3_PREFIX)
   s3_path = s3_path.replace(utils.PATH_URL, '')
   with fs.open(s3_path, mode='rb') as fhandle:
      with xr.open_dataset(fhandle, engine=utils.NC_ENGINE) as granule_ds:
         return granule_ds.load()


def _crop_and_align_granule(granule_ds, cube_x, cube_y):
   """Crop `granule_ds` to the cube's x/y footprint and reindex onto the
   cube's exact x/y coordinate values, mirroring the bounding-box mask
   ITSCube.preprocess_dataset() applies (itscube.py) before a granule's
   data ever reaches a cube layer. Any cube pixel outside the granule's own
   footprint reindexes to NaN here -- indistinguishable from (and handled
   the same as) a pixel the granule itself reports as missing."""
   x_min, x_max = float(cube_x.min()), float(cube_x.max())
   y_min, y_max = float(cube_y.min()), float(cube_y.max())
   mask = (
      (granule_ds.x >= x_min) & (granule_ds.x <= x_max) &
      (granule_ds.y >= y_min) & (granule_ds.y <= y_max)
   )
   return granule_ds.where(mask, drop=True).reindex(x=cube_x, y=cube_y)


def _values_match(cube_val, granule_val, is_date=False):
   """Compare one cube scalar against the granule value it was derived
   from. Dates are compared as timestamps, exactly -- these are encoded
   losslessly (int64 nanoseconds since epoch, utils.Units.ns_epoch_date),
   so a real mismatch is the only thing that should trip this. Everything
   else is compared numerically with a small tolerance (same cast function
   applied to the same source value should be bit-identical, but
   floor-level float32 round trips are tolerated) or, failing that, by
   string representation."""
   try:
      if pd.isna(cube_val) and pd.isna(granule_val):
         return True
   except (TypeError, ValueError):
      pass

   if is_date:
      try:
         return pd.Timestamp(cube_val) == pd.Timestamp(granule_val)
      except Exception:
         return str(cube_val) == str(granule_val)

   try:
      return np.isclose(
         float(cube_val), float(granule_val),
         rtol=SCALAR_RTOL, atol=SCALAR_ATOL, equal_nan=True
      )
   except (TypeError, ValueError):
      return str(cube_val) == str(granule_val)


def _verify_array_vars(cube_decoded, granule_cropped, idx, var_names, findings):
   """Compare decoded physical values of every array variable in
   `var_names`, for layer `idx`, between the cube and the (already
   cropped/aligned) granule. See ARRAY_RTOL/ARRAY_ATOL for why this
   compares physical values with tolerance rather than raw encoded bytes."""
   for var_name in var_names:
      if var_name not in granule_cropped:
         _add(
            findings, 'WARNING', var_name,
            f'layer {idx}: not present in source granule, skipped data check.'
         )
         continue

      cube_arr = cube_decoded[var_name].values[idx, :, :]
      granule_arr = granule_cropped[var_name].values[0, :, :]

      mismatch = ~np.isclose(
         cube_arr, granule_arr, rtol=ARRAY_RTOL, atol=ARRAY_ATOL, equal_nan=True
      )
      num_mismatch = int(np.count_nonzero(mismatch))
      if num_mismatch:
         abs_diff = np.abs(cube_arr[mismatch] - granule_arr[mismatch])
         _add(
            findings, 'ERROR', var_name,
            f'layer {idx}: {num_mismatch}/{mismatch.size} pixel(s) differ '
            f'from source granule (max abs diff={np.nanmax(abs_diff):.6g}).'
         )


def _verify_time_coordinate(ds, granule_ds, granule_url, idx, findings):
   """Compare the cube's 'time' coordinate value for layer `idx` against
   the source granule's own 'time' coordinate, exactly. The cube's 'time'
   is the granule's own 'time' value carried through unchanged (see
   virtual_itslive_cube.py's `coords={..., "time": vds["time"]}`) -- so
   any difference here is a real bug, not encoding noise.

   Args:
      ds (xr.Dataset): the cube, opened with mask_and_scale=False (decode_
         times still defaults to True, so 'time' is already datetime64).
      granule_ds (xr.Dataset): the source granule, opened with default
         (CF-decoding) options.
      granule_url (str): source granule URL, for error reporting.
      idx (int): cube layer index.
      findings (list of dict): findings list to append to.
   """
   if utils.Coords.TIME not in granule_ds:
      _add(
         findings, 'WARNING', utils.Coords.TIME,
         f"layer {idx}: granule has no '{utils.Coords.TIME}' coordinate, "
         'cannot verify.'
      )
      return

   granule_val = granule_ds[utils.Coords.TIME].values[0]
   cube_val = ds[utils.Coords.TIME].values[idx]
   if pd.Timestamp(cube_val) != pd.Timestamp(granule_val):
      _add(
         findings, 'ERROR', utils.Coords.TIME,
         f'layer {idx}: cube value {cube_val!r} does not exactly match '
         f'source granule value {granule_val!r} ({granule_url}).'
      )


def _verify_img_pair_info_vars(cube_decoded, granule_ds, granule_url, idx, findings):
   """Compare the scalar variables promoted from the granule's
   'img_pair_info' attributes (ImgPairInfo.all, plus the ascending_img1/2
   binary flags and autoRIFT_software_version), re-using the exact
   extraction helpers itscube.py itself calls (utils.get_data_var_attr /
   utils.get_data_var_binary_attr) so this check can't drift from how the
   cube was actually populated."""
   for each in ImgPairInfo.all:
      each_dtype = ImgPairInfo.allTypes.get(each)
      is_date = each in ImgPairInfo.toDate
      try:
         granule_val = utils.get_data_var_attr(
            granule_ds, granule_url, ImgPairInfo.name, each,
            to_date=is_date, data_dtype=each_dtype
         )
      except Exception as exc:
         _add(
            findings, 'WARNING', each,
            f'layer {idx}: could not read from source granule: {exc}'
         )
         continue

      cube_val = cube_decoded[each].values[idx]
      if not _values_match(cube_val, granule_val, is_date=is_date):
         _add(
            findings, 'ERROR', each,
            f'layer {idx}: cube value {cube_val!r} does not match source '
            f'granule value {granule_val!r} ({granule_url}).'
         )

   for flight_attr, cube_var in (
      (ImgPairInfo.flight_direction_img1, Vars.ascending_img1),
      (ImgPairInfo.flight_direction_img2, Vars.ascending_img2),
   ):
      granule_val = utils.get_data_var_binary_attr(
         granule_ds, granule_url, ImgPairInfo.name, flight_attr,
         ImgPairInfo.ascending, data_dtype=np.uint8,
         missing_value=utils.Missing.u8value
      )
      # get_data_var_binary_attr() returns the raw 255 missing_value
      # sentinel for every optical granule (no flight-direction attr --
      # radar-only, see EXTRA_RADAR_ONLY_VARS); cube_decoded is CF-decoded,
      # so the cube's own 255 _FillValue reads back as NaN. Normalize to
      # match, or this flags as a false mismatch.
      if granule_val == utils.Missing.u8value:
         granule_val = np.nan
      cube_val = cube_decoded[cube_var].values[idx]
      if not _values_match(cube_val, granule_val):
         _add(
            findings, 'ERROR', cube_var,
            f'layer {idx}: cube value {cube_val!r} does not match source '
            f'granule value {granule_val!r} ({granule_url}).'
         )

   if Vars.autorift_software_version in granule_ds.attrs:
      granule_val = granule_ds.attrs[Vars.autorift_software_version]
      cube_val = cube_decoded[Vars.autorift_software_version].values[idx]
      if not _values_match(cube_val, granule_val):
         _add(
            findings, 'ERROR', Vars.autorift_software_version,
            f'layer {idx}: cube value {cube_val!r} does not match source '
            f'granule value {granule_val!r} ({granule_url}).'
         )


def _verify_v_attribute_vars(cube_decoded, granule_ds, granule_url, idx, bases, findings):
   """Compare the per-velocity-component scalar variables derived from
   vx/vy/va/vr attributes (error/error_stationary/error_modeled/error_slow,
   stable_shift/stable_shift_stationary/stable_shift_slow), mirroring
   itscube.py's process_v_attributes()."""
   for base in bases:
      for suffix in V_ERROR_SUFFIXES + V_SHIFT_SUFFIXES:
         cube_var = f'{base}_{suffix}'
         if cube_var not in cube_decoded:
            continue

         granule_val = utils.get_data_var_attr(
            granule_ds, granule_url, base, suffix, utils.Missing.value
         )
         cube_val = cube_decoded[cube_var].values[idx]
         if not _values_match(cube_val, granule_val):
            _add(
               findings, 'ERROR', cube_var,
               f'layer {idx}: cube value {cube_val!r} does not match source '
               f'granule value {granule_val!r} ({granule_url}).'
            )

      # 'stable_shift' itself: a NaN granule value is expected to read back
      # as 0 in the cube (see itscube.py's process_v_attributes()).
      cube_var = f'{base}_{Vars.postfix.stable_shift}'
      if cube_var in cube_decoded:
         granule_val = utils.get_data_var_attr(
            granule_ds, granule_url, base, Vars.postfix.stable_shift,
            utils.Missing.value
         )
         if isinstance(granule_val, float) and np.isnan(granule_val):
            granule_val = 0
         cube_val = cube_decoded[cube_var].values[idx]
         if not _values_match(cube_val, granule_val):
            _add(
               findings, 'ERROR', cube_var,
               f'layer {idx}: cube value {cube_val!r} does not match source '
               f'granule value {granule_val!r} ({granule_url}).'
            )

   # Shared-once attributes: captured from whichever of `bases` actually
   # carries them on the granule (first hit wins), see itscube.py's
   # process_v_attributes() "capture only once" loop.
   for attr_name in (Vars.flag_stable_shift, Vars.stable_count_mask, Vars.stable_count_slow):
      if attr_name not in cube_decoded:
         continue

      source_base = next(
         (b for b in bases if b in granule_ds and attr_name in granule_ds[b].attrs),
         None
      )
      if source_base is None:
         continue

      granule_val = utils.get_data_var_attr(
         granule_ds, granule_url, source_base, attr_name, data_dtype=np.int32
      )
      cube_val = cube_decoded[attr_name].values[idx]
      if not _values_match(cube_val, granule_val):
         _add(
            findings, 'ERROR', attr_name,
            f'layer {idx}: cube value {cube_val!r} does not match source '
            f'granule value {granule_val!r} ({granule_url}).'
         )


def _verify_m_attribute_vars(cube_decoded, granule_ds, granule_url, idx, findings):
   """Compare M11/M12_dr_to_vr_factor, mirroring itscube.py's
   process_m_attributes(). Radar-only -- callers should only invoke this for
   a radar-sourced layer."""
   for base in (Vars.m11, Vars.m12):
      cube_var = f'{base}_{Vars.postfix.dr_to_vr_factor}'
      if cube_var not in cube_decoded:
         continue

      granule_val = utils.get_data_var_attr(
         granule_ds, granule_url, base, Vars.postfix.dr_to_vr_factor,
         utils.Missing.byte
      )
      cube_val = cube_decoded[cube_var].values[idx]
      if not _values_match(cube_val, granule_val):
         _add(
            findings, 'ERROR', cube_var,
            f'layer {idx}: cube value {cube_val!r} does not match source '
            f'granule value {granule_val!r} ({granule_url}).'
         )


def _sample_layers_by_sensor_type(total_layers, is_radar, radar_status, num_layers, seed, findings):
   """Pick layers to verify, split evenly between radar and optical so a
   flat random sample (radar is typically a small minority) can't quietly
   skip radar-only variables (M11/M12/vr/va) entirely. `num_layers // 2` of
   each class (minimum 1) when `is_radar` is trustworthy; falls back to one
   flat random sample across every layer, with a WARNING, when it isn't.

   Returns
   -------
   list of int
      Sorted layer indices to verify.
   """
   rng = random.Random(seed)

   if is_radar is None:
      num_layers = min(num_layers, total_layers)
      _add(
         findings, 'WARNING', Vars.url,
         f'sensor type unknown ({radar_status}) -- cannot stratify the '
         'sample by radar/optical, falling back to one flat random sample '
         'across every layer.'
      )
      logging.info(f'Verifying {num_layers} random layer(s) against source granules...')
      return sorted(rng.sample(range(total_layers), k=num_layers))

   radar_pool = np.flatnonzero(is_radar[:total_layers])
   optical_pool = np.flatnonzero(~is_radar[:total_layers])
   num_per_class = max(1, num_layers // 2)

   num_radar = min(num_per_class, len(radar_pool))
   num_optical = min(num_per_class, len(optical_pool))

   if num_radar < num_per_class:
      _add(
         findings, 'WARNING', Vars.url,
         f'only {len(radar_pool)} radar layer(s) in the cube -- sampling '
         f'all of them instead of the requested {num_per_class}.'
      )
   if num_optical < num_per_class:
      _add(
         findings, 'WARNING', Vars.url,
         f'only {len(optical_pool)} optical layer(s) in the cube -- '
         f'sampling all of them instead of the requested {num_per_class}.'
      )

   logging.info(
      f'Verifying {num_radar} random radar layer(s) and {num_optical} '
      'random optical layer(s) against source granules...'
   )
   return sorted(
      rng.sample(list(radar_pool), k=num_radar) +
      rng.sample(list(optical_pool), k=num_optical)
   )


def _verify_random_layers(
   cube_url, ds, is_radar, radar_status, num_layers, seed, findings
):
   """Pick `num_layers` random layers from the cube -- split evenly between
   radar and optical (see _sample_layers_by_sensor_type()) -- and compare
   every variable's value against the original source granule they were
   deep-copied/extracted from (see Vars.url). This is the only check in this
   module that reads pixel/attribute data rather than just structural
   metadata -- expensive (one full granule S3 fetch per sampled layer), so
   callers only reach this when explicitly requested via --verify-granule-data.
   """
   if Vars.url not in ds:
      _add(
         findings, 'WARNING', Vars.url,
         'granule_url variable not present in store -- cannot verify '
         'layers against source granules.'
      )
      return

   total_layers = ds.sizes[utils.Coords.TIME]
   layer_indices = _sample_layers_by_sensor_type(
      total_layers, is_radar, radar_status, num_layers, seed, findings
   )

   cube_decoded = xr.open_dataset(
      cube_url, engine='zarr', zarr_format=3, consolidated=True,
      backend_kwargs={'storage_options': {'anon': True}}
   )
   cube_x = ds[utils.Coords.X]
   cube_y = ds[utils.Coords.Y]
   fs = s3fs.S3FileSystem(anon=True)

   urls = ds[Vars.url].values
   for idx in layer_indices:
      granule_url = str(urls[idx])
      if not granule_url:
         _add(
            findings, 'WARNING', Vars.url,
            f'layer {idx}: empty granule_url, cannot verify.'
         )
         continue

      try:
         granule_ds = _open_granule(granule_url, fs)
      except Exception as exc:
         _add(
            findings, 'WARNING', Vars.url,
            f'layer {idx}: failed to open source granule {granule_url}: {exc}'
         )
         continue

      with granule_ds:
         granule_cropped = _crop_and_align_granule(granule_ds, cube_x, cube_y)

         layer_is_radar = is_radar[idx] if is_radar is not None else None
         array_vars = list(DIRECT_ARRAY_VARS)
         v_bases = [Vars.vx, Vars.vy]
         if layer_is_radar:
            array_vars += list(RADAR_ARRAY_VARS)
            v_bases += [Vars.va, Vars.vr]
         elif layer_is_radar is None:
            _add(
               findings, 'WARNING', Vars.url,
               f'layer {idx}: sensor type unknown ({radar_status}), only '
               f'optical-common variables verified.'
            )

         _verify_array_vars(cube_decoded, granule_cropped, idx, array_vars, findings)
         _verify_time_coordinate(ds, granule_ds, granule_url, idx, findings)
         _verify_img_pair_info_vars(cube_decoded, granule_ds, granule_url, idx, findings)
         _verify_v_attribute_vars(
            cube_decoded, granule_ds, granule_url, idx, v_bases, findings
         )
         if layer_is_radar:
            _verify_m_attribute_vars(cube_decoded, granule_ds, granule_url, idx, findings)

   cube_decoded.close()


def validate_cube(cube_url, verify_granule_data=False, num_random_layers=10, seed=None):
   """Run every check in this module's docstring against `cube_url`.

   Parameters
   ----------
   verify_granule_data (bool): Also sample `num_random_layers` random
      layers and compare every variable's value against the original
      source granule (see _verify_random_layers()). Off by default -- it's
      the only check here that reads pixel data, at the cost of one full
      granule S3 fetch per sampled layer.
   num_random_layers (int): Number of random layers to sample when
      `verify_granule_data` is True.
   seed: Seed for the random layer sample, for a reproducible re-run.
      Default is None (a fresh random sample each run).

   Returns
   -------
   list of dict
      Findings, each {'level': 'ERROR'|'WARNING'|'INFO', 'variable': str,
      'message': str}, most-actionable first (ERRORs are what block calling
      this store production-ready).
   """
   findings = []

   group = _check_consolidated_metadata(cube_url, findings)
   if group is None:
      return findings

   ds = xr.open_dataset(
      cube_url, engine='zarr', zarr_format=3, consolidated=True,
      mask_and_scale=False, backend_kwargs={'storage_options': {'anon': True}}
   )
   _check_coordinate_sanity(ds, findings)
   _check_shape_consistency(group, findings)

   total_layers = group[utils.Coords.TIME].shape[0]
   is_radar, radar_status = _compute_is_radar(group, total_layers, findings)

   base = cube_url.replace(utils.S3_PREFIX, '').rstrip('/')
   fs = s3fs.S3FileSystem(anon=True)
   for var_name in sorted(group.array_keys()):
      if var_name in (utils.Coords.TIME, utils.Coords.X, utils.Coords.Y):
         # Written once, up front, as part of the store's skeleton -- never
         # incrementally chunked/uploaded, so there's no partial-write
         # failure mode to check here beyond the shape/sanity checks above.
         continue

      _check_chunk_completeness(
         fs, base, var_name, group[var_name], is_radar, radar_status, findings
      )

   if verify_granule_data:
      _verify_random_layers(
         cube_url, ds, is_radar, radar_status, num_random_layers, seed, findings
      )

   return findings


def _level_rank(level):
   return {'ERROR': 0, 'WARNING': 1, 'INFO': 2}.get(level, 3)


def main():
   parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter
   )
   parser.add_argument(
      '--cube-url',
      type=str,
      required=True,
      help='s3:// URL of the deep-copy Zarr v3 store to validate.'
   )
   parser.add_argument(
      '--show-info',
      action='store_true',
      help='Also print INFO-level findings (e.g. verified-legitimate '
         'radar skips) [%(default)s].'
   )
   parser.add_argument(
      '--verify-granule-data',
      action='store_true',
      help='Also sample --num-random-layers random layers and compare '
         'every variable (optical and radar) against the original source '
         'granule they were extracted from. Expensive -- fetches a full '
         'granule per sampled layer -- so off by default [%(default)s].'
   )
   parser.add_argument(
      '--num-random-layers',
      type=int,
      default=10,
      help='Number of random layers to sample for --verify-granule-data, '
         'split evenly between radar and optical (half of this value '
         'sampled from each, so radar -- typically a small minority -- '
         'always gets checked) when the cube\'s sensor type per layer can '
         'be classified; falls back to one flat random sample across '
         'every layer otherwise [%(default)s].'
   )
   parser.add_argument(
      '--seed',
      type=int,
      default=None,
      help='Seed for the --verify-granule-data random layer sample, for a '
         'reproducible re-run. Default is an unseeded, fresh sample each run.'
   )
   args = parser.parse_args()

   logging.info(f'Validating {args.cube_url}')
   findings = validate_cube(
      args.cube_url,
      verify_granule_data=args.verify_granule_data,
      num_random_layers=args.num_random_layers,
      seed=args.seed
   )
   findings.sort(key=lambda f: (_level_rank(f['level']), f['variable']))

   counts = {'ERROR': 0, 'WARNING': 0, 'INFO': 0}
   for finding in findings:
      counts[finding['level']] += 1
      if finding['level'] == 'INFO' and not args.show_info:
         continue
      logging.info(f"[{finding['level']}] {finding['variable']}: {finding['message']}")

   logging.info(
      f"Done: {counts['ERROR']} error(s), {counts['WARNING']} warning(s), "
      f"{counts['INFO']} info finding(s)"
   )
   sys.exit(1 if counts['ERROR'] else 0)


if __name__ == '__main__':
   main()
