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
import sys
from math import ceil

import numpy as np
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


def validate_cube(cube_url):
   """Run every check in this module's docstring against `cube_url`.

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
   args = parser.parse_args()

   logging.info(f'Validating {args.cube_url}')
   findings = validate_cube(args.cube_url)
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
