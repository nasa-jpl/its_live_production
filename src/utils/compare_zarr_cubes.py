"""
Compare two deep-copied ITS_LIVE Zarr v3 datacubes ("old" reference vs.
"new") to confirm the new one is a faithful reproduction of the old one --
global/per-variable attributes, on-disk encoding (chunks/shards/compressor/
fill-value), coordinates, and every data value.

There are other two existing datacube-inspection tools:
* tools/validate_s3_cube.py validates a SINGLE cube's structural completeness
(chunk presence, consolidated-metadata staleness)
* compare_zarr_cube_open.py compares open timing and on-disk chunk-layout
metadata across stores.

Both cubes are opened raw (mask_and_scale=False, matching
validate_s3_cube.py's convention) so integer velocity variables compare as
plain integers and floating-point variables carry their on-disk NaN fill
values directly -- comparisons below treat NaN/NaT as equal to NaN/NaT
(not a mismatch), but otherwise require EXACT equality: these are meant to
be byte-for-byte reproductions of the same source data, so no numeric
tolerance is applied.

Data values are compared in batches along 'time' (full y/x extent per
batch, matching deep_copy_cube.py's batch-along-time convention) to stay
usable on 100K+-layer, 145GB+ production cubes (see CLAUDE.md's chunking
notes) -- each batch holds both cubes' slices in memory at once, so
--batch-size defaults lower than deep_copy_cube.py's batch_size.

Usage example:
python src/utils/compare_zarr_cubes.py \
   --old-cube-url s3://its-live-data/path/to/reference_cube.zarr \
   --new-cube-url s3://its-live-data/path/to/regenerated_cube.zarr
"""
import argparse
import logging
import math
import sys

import numpy as np
import xarray as xr

import utils
from itscube_types import CubeFormat

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Global attributes expected to legitimately differ between two generation
# runs of "the same" cube -- not a sign of inconsistency.
IGNORE_GLOBAL_ATTRS = {
   CubeFormat.date_created,
   CubeFormat.date_updated,
   CubeFormat.datacube_software_version,
}

# Keys that carry a variable's fill/missing-value sentinel. Depending on
# how mask_and_scale/zarr-format round-tripping shakes out, this can
# surface in either .attrs or .encoding (see _actual_encoding_map) -- same
# tuple used by tests/test_virtual_cube_generation.py's golden-encoding
# test.
_FILL_VALUE_KEYS = (
   utils.OutputFormat.fill_value,  # '_FillValue'
   utils.Missing.fill_value,       # 'fill_value' (zarr-level sentinel)
   utils.Missing.name,             # 'missing_value'
)

DEFAULT_BATCH_SIZE = 1000


def _add(findings, level, variable, message):
   findings.append({'level': level, 'variable': variable, 'message': message})


def _open_cube(url):
   """Open a deep-copy Zarr v3 datacube raw (mask_and_scale=False), the
   same read convention as validate_s3_cube.py -- so encoding/fill-value
   sentinels are directly comparable rather than hidden behind decoding.
   """
   kwargs = {}
   if url.startswith(utils.S3_PREFIX):
      kwargs['backend_kwargs'] = {'storage_options': {'anon': True}}

   return xr.open_dataset(
      url, engine='zarr', zarr_format=3, consolidated=True,
      mask_and_scale=False, **kwargs
   )


def _scalar_equal(old_value, new_value):
   """True if two scalar attribute/encoding values are equal, treating two
   NaN floats as equal (mirrors test_virtual_cube_generation.py's golden-
   encoding test, which uses the same math.isnan trick for the same
   reason: a real, identical NaN sentinel on both sides must not be
   reported as a mismatch).
   """
   try:
      if isinstance(old_value, float) and isinstance(new_value, float) \
            and math.isnan(old_value) and math.isnan(new_value):
         return True
   except TypeError:
      pass

   return old_value == new_value


def _actual_encoding_map(var):
   """Merge a variable's .attrs and .encoding into a single lookup -- see
   _FILL_VALUE_KEYS's docstring for why. Identical to
   test_virtual_cube_generation.py's helper of the same name.
   """
   merged = dict(var.attrs)
   merged.update(var.encoding)
   return merged


def _compressor_cname(var):
   """Extract the compressor codec's short name (e.g. 'lz4') from a
   variable's encoding, tolerating both zarr v3's 'compressors' (list of
   codec objects) and legacy v2's singular 'compressor' key. Identical to
   test_virtual_cube_generation.py's helper of the same name.
   """
   codecs = var.encoding.get(utils.OutputFormat.compressors)
   if not codecs:
      single = var.encoding.get(utils.OutputFormat.compressor)
      codecs = [single] if single is not None else []

   if not codecs:
      return None

   codec = codecs[0]
   if isinstance(codec, dict):
      return codec.get('cname')

   return getattr(codec, 'cname', None)


def _compare_global_attrs(old_ds, new_ds, findings):
   """Diff dataset-level .attrs (minus IGNORE_GLOBAL_ATTRS): ERROR on any
   key missing from one side or differing in value.
   """
   old_attrs, new_attrs = old_ds.attrs, new_ds.attrs
   keys = (set(old_attrs) | set(new_attrs)) - IGNORE_GLOBAL_ATTRS

   for key in sorted(keys):
      if key not in old_attrs:
         _add(findings, 'ERROR', '(global)', f"attribute '{key}' present in new cube but missing from old")
      elif key not in new_attrs:
         _add(findings, 'ERROR', '(global)', f"attribute '{key}' present in old cube but missing from new")
      elif not _scalar_equal(old_attrs[key], new_attrs[key]):
         _add(findings, 'ERROR', '(global)', f"attribute '{key}' differs: old={old_attrs[key]!r}, new={new_attrs[key]!r}")


def _compare_variable_presence(old_ds, new_ds, findings):
   """Set-diff of data_vars. A variable missing in the new cube is a
   regression (ERROR); a variable that's new is a deliberate addition, not
   a consistency violation (INFO). Returns the sorted list of variable
   names common to both cubes.
   """
   old_vars = set(old_ds.data_vars)
   new_vars = set(new_ds.data_vars)

   for name in sorted(old_vars - new_vars):
      _add(findings, 'ERROR', name, 'present in old cube but missing from new cube')

   for name in sorted(new_vars - old_vars):
      _add(findings, 'INFO', name, 'present in new cube but not in old cube (new addition)')

   return sorted(old_vars & new_vars)


def _compare_coordinates(old_ds, new_ds, findings):
   """x/y: exact equality required -- any mismatch means the two cubes
   cover different grids, which makes a pixel-by-pixel data comparison
   meaningless, so this gates all later value comparisons via the
   returned `grids_match` flag. time: WARNING (not ERROR) if lengths
   differ -- comparisons are then bounded to the overlapping range --
   plus an ERROR at the first differing value within that overlap.

   Returns
   -------
   tuple of (bool, int)
      (grids_match, overlap_time) -- overlap_time is min(len(old.time),
      len(new.time)), the number of layers safe to compare.
   """
   grids_match = True

   for coord_name in (utils.Coords.X, utils.Coords.Y):
      if coord_name not in old_ds.coords or coord_name not in new_ds.coords:
         _add(findings, 'ERROR', coord_name, 'coordinate missing from one of the two cubes')
         grids_match = False
         continue

      old_vals, new_vals = old_ds[coord_name].values, new_ds[coord_name].values
      if old_vals.shape != new_vals.shape or not np.array_equal(old_vals, new_vals):
         _add(
            findings, 'ERROR', coord_name,
            f'coordinate values differ (shapes {old_vals.shape} vs '
            f'{new_vals.shape}) -- grids are not aligned, skipping all '
            f'data-value comparisons'
         )
         grids_match = False

   old_time = old_ds[utils.Coords.TIME].values
   new_time = new_ds[utils.Coords.TIME].values
   overlap = min(len(old_time), len(new_time))

   if len(old_time) != len(new_time):
      _add(
         findings, 'WARNING', utils.Coords.TIME,
         f'layer counts differ: old={len(old_time)}, new={len(new_time)} '
         f'-- comparing overlapping {overlap} layer(s) only'
      )

   if overlap and not np.array_equal(old_time[:overlap], new_time[:overlap]):
      mismatch_idx = np.where(old_time[:overlap] != new_time[:overlap])[0]
      i = int(mismatch_idx[0])
      _add(
         findings, 'ERROR', utils.Coords.TIME,
         f'time values differ starting at layer {i}: '
         f'old={old_time[i]!r}, new={new_time[i]!r}'
      )

   return grids_match, overlap


def _compare_variable_dtype_shape(old_ds, new_ds, var_name, findings):
   """ERROR on dtype mismatch or non-time-dim shape mismatch. Returns
   False to signal "skip the value comparison for this variable" -- a
   shape/dtype mismatch makes elementwise comparison meaningless (or
   crash-prone via isel/broadcasting).
   """
   old_var, new_var = old_ds[var_name], new_ds[var_name]
   ok = True

   if old_var.dtype != new_var.dtype:
      _add(findings, 'ERROR', var_name, f'dtype differs: old={old_var.dtype}, new={new_var.dtype}')
      ok = False

   non_time_old = tuple(s for d, s in zip(old_var.dims, old_var.shape) if d != utils.Coords.TIME)
   non_time_new = tuple(s for d, s in zip(new_var.dims, new_var.shape) if d != utils.Coords.TIME)
   if non_time_old != non_time_new:
      _add(
         findings, 'ERROR', var_name,
         f'non-time shape differs: old={old_var.shape} {old_var.dims}, '
         f'new={new_var.shape} {new_var.dims}'
      )
      ok = False

   return ok


def _compare_variable_encoding(old_ds, new_ds, var_name, findings):
   """On-disk storage-encoding diff: chunk shape, shard shape, compressor
   family, and fill/missing-value sentinel. Deliberately NOT comparing
   exact clevel/shuffle/blocksize -- same "deliberately narrow scope"
   rationale already documented and validated by
   test_virtual_cube_generation.py's golden-encoding test: those can
   legitimately differ by design between pipeline versions.
   """
   old_var, new_var = old_ds[var_name], new_ds[var_name]

   old_chunks = old_var.encoding.get('chunks')
   new_chunks = new_var.encoding.get('chunks')
   if old_chunks != new_chunks:
      _add(findings, 'ERROR', var_name, f'chunks differ: old={old_chunks}, new={new_chunks}')

   old_shards = old_var.encoding.get('shards')
   new_shards = new_var.encoding.get('shards')
   if old_shards != new_shards:
      _add(findings, 'ERROR', var_name, f'shards differ: old={old_shards}, new={new_shards}')

   old_cname = _compressor_cname(old_var)
   new_cname = _compressor_cname(new_var)
   if old_cname != new_cname:
      _add(findings, 'ERROR', var_name, f'compressor family differs: old={old_cname!r}, new={new_cname!r}')

   old_map = _actual_encoding_map(old_var)
   new_map = _actual_encoding_map(new_var)
   for key in _FILL_VALUE_KEYS:
      old_has, new_has = key in old_map, key in new_map
      if old_has and not new_has:
         _add(findings, 'ERROR', var_name, f"'{key}' present in old cube but missing from new")
      elif new_has and not old_has:
         _add(findings, 'ERROR', var_name, f"'{key}' present in new cube but missing from old")
      elif old_has and new_has and not _scalar_equal(old_map[key], new_map[key]):
         _add(findings, 'ERROR', var_name, f"'{key}' differs: old={old_map[key]!r}, new={new_map[key]!r}")


def _compare_variable_attrs(old_ds, new_ds, var_name, findings):
   """Diff the general descriptive .attrs (standard_name, description,
   units, etc.) -- excludes _FILL_VALUE_KEYS, which are checked by
   _compare_variable_encoding instead (they're a storage property, not
   descriptive metadata, and can live in either .attrs or .encoding).
   """
   old_attrs, new_attrs = old_ds[var_name].attrs, new_ds[var_name].attrs
   keys = (set(old_attrs) | set(new_attrs)) - set(_FILL_VALUE_KEYS)

   for key in sorted(keys):
      if key not in old_attrs:
         _add(findings, 'ERROR', var_name, f"attribute '{key}' present in new cube but missing from old")
      elif key not in new_attrs:
         _add(findings, 'ERROR', var_name, f"attribute '{key}' present in old cube but missing from new")
      elif not _scalar_equal(old_attrs[key], new_attrs[key]):
         _add(findings, 'ERROR', var_name, f"attribute '{key}' differs: old={old_attrs[key]!r}, new={new_attrs[key]!r}")


def _mismatch_mask(old_vals, new_vals):
   """Exact-equality mask, treating NaN-vs-NaN (floating) and NaT-vs-NaT
   (datetime64) as a match rather than a mismatch -- these are real,
   identical fill/missing sentinels, not a comparison failure. No numeric
   tolerance is applied otherwise: these are meant to be byte-for-byte
   reproductions of the same source data.
   """
   equal = np.equal(old_vals, new_vals)

   def _nanlike(values):
      if np.issubdtype(values.dtype, np.floating):
         return np.isnan(values)
      if np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(values.dtype, np.timedelta64):
         return np.isnat(values)
      return np.zeros_like(equal, dtype=bool)

   both_missing = _nanlike(old_vals) & _nanlike(new_vals)
   return ~(equal | both_missing)


def _time_batches(n, batch_size):
   for start in range(0, n, batch_size):
      yield start, min(start + batch_size, n)


def _compare_variable_values(old_ds, new_ds, var_name, overlap_time, batch_size, findings):
   """Memory-bounded elementwise value comparison for one variable.
   Time-indexed variables are read in --batch-size-layer batches (full y/x
   extent per batch); static/scalar variables are read in one shot. Emits
   ONE finding for the whole variable (INFO if it matches exactly, ERROR
   with the total mismatch count + first offending index/value pair
   otherwise) rather than one per batch, so a badly-broken variable
   doesn't flood the findings list.
   """
   old_var, new_var = old_ds[var_name], new_ds[var_name]

   if utils.Coords.TIME in old_var.dims:
      batches = list(_time_batches(overlap_time, batch_size))
   else:
      batches = [(None, None)]
   num_batches = len(batches)

   total_mismatches = 0
   total_elements = 0
   first_example = None

   for batch_num, (start, stop) in enumerate(batches, start=1):
      if start is None:
         old_vals = old_var.values
         new_vals = new_var.values
         offset = 0
      else:
         logging.info(
            f"  '{var_name}': comparing layers {start}:{stop} of "
            f'{overlap_time} (batch {batch_num}/{num_batches})'
         )
         old_vals = old_var.isel({utils.Coords.TIME: slice(start, stop)}).values
         new_vals = new_var.isel({utils.Coords.TIME: slice(start, stop)}).values
         offset = start

      mask = _mismatch_mask(old_vals, new_vals)
      total_elements += mask.size
      n = int(mask.sum())
      if not n:
         continue

      total_mismatches += n
      if first_example is None:
         if mask.ndim == 0:
            first_example = ((), old_vals.item(), new_vals.item())
         else:
            idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape))
            global_idx = idx if offset == 0 else (idx[0] + offset,) + idx[1:]
            first_example = (global_idx, old_vals[idx].item(), new_vals[idx].item())

   if total_mismatches:
      global_idx, old_val, new_val = first_example
      message = (
         f'{total_mismatches}/{total_elements} elements differ; first '
         f'mismatch at index {global_idx}: old={old_val!r}, new={new_val!r}'
      )
      _add(findings, 'ERROR', var_name, message)
   else:
      message = f'values match exactly across {total_elements} elements'
      _add(findings, 'INFO', var_name, message)

   logging.info(f"  '{var_name}': {message}")


def _compare_datasets(old_ds, new_ds, var_names=None, batch_size=DEFAULT_BATCH_SIZE, skip_values=False):
   """Run every check above against two already-open xr.Datasets and
   return the findings list. Kept separate from compare_cubes() so tests
   can exercise it directly with small in-memory synthetic datasets --
   no S3/file I/O.
   """
   findings = []

   _compare_global_attrs(old_ds, new_ds, findings)
   common_vars = _compare_variable_presence(old_ds, new_ds, findings)
   grids_match, overlap_time = _compare_coordinates(old_ds, new_ds, findings)

   if var_names:
      requested = set(var_names)
      for name in sorted(requested - set(common_vars)):
         _add(findings, 'WARNING', name, '--vars requested this variable but it is not common to both cubes; skipping')
      common_vars = [name for name in common_vars if name in requested]

   if not grids_match:
      _add(findings, 'ERROR', '(root)', 'x/y grids differ between cubes -- skipping all per-variable checks')
      return findings

   total_vars = len(common_vars)
   logging.info(f'{total_vars} variable(s) to compare across {overlap_time} overlapping time layer(s)')

   for i, var_name in enumerate(common_vars, start=1):
      logging.info(f"Comparing variable '{var_name}' ({i}/{total_vars})")

      ok = _compare_variable_dtype_shape(old_ds, new_ds, var_name, findings)
      _compare_variable_encoding(old_ds, new_ds, var_name, findings)
      _compare_variable_attrs(old_ds, new_ds, var_name, findings)

      if skip_values or not ok:
         continue

      _compare_variable_values(old_ds, new_ds, var_name, overlap_time, batch_size, findings)

   return findings


def compare_cubes(old_url, new_url, var_names=None, batch_size=DEFAULT_BATCH_SIZE, skip_values=False):
   """Open both cubes and run every comparison, returning the findings
   list (see _compare_datasets).
   """
   old_ds = _open_cube(old_url)
   new_ds = _open_cube(new_url)

   return _compare_datasets(old_ds, new_ds, var_names, batch_size, skip_values)


def _level_rank(level):
   return {'ERROR': 0, 'WARNING': 1, 'INFO': 2}.get(level, 3)


def main():
   parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter
   )
   parser.add_argument(
      '--old-cube-url',
      type=str,
      required=True,
      help='s3:// URL or local path of the reference (old) deep-copy Zarr v3 store.'
   )
   parser.add_argument(
      '--new-cube-url',
      type=str,
      required=True,
      help='s3:// URL or local path of the deep-copy Zarr v3 store to check for consistency against --old-cube-url.'
   )
   parser.add_argument(
      '--vars',
      nargs='+',
      default=None,
      dest='var_names',
      help='Restrict comparison to these data variable names '
         '(default: every variable common to both cubes).'
   )
   parser.add_argument(
      '--batch-size',
      type=int,
      default=DEFAULT_BATCH_SIZE,
      help='Number of time layers to read per batch when comparing '
         'time-indexed variables [%(default)s]. Each batch holds both '
         "cubes' slices in memory at once, so keep this lower than a "
         'single-cube tool would for the same variable/extent.'
   )
   parser.add_argument(
      '--skip-values',
      action='store_true',
      help='Skip the (expensive) per-element data-value comparison; '
         'only check structure/encoding/attributes.'
   )
   parser.add_argument(
      '--show-info',
      action='store_true',
      help='Also print INFO-level findings (e.g. exact-match '
         'confirmations, new-variable additions) [%(default)s].'
   )
   args = parser.parse_args()

   logging.info(f'Comparing old={args.old_cube_url} vs new={args.new_cube_url}')
   findings = compare_cubes(
      args.old_cube_url, args.new_cube_url,
      var_names=args.var_names, batch_size=args.batch_size,
      skip_values=args.skip_values
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
