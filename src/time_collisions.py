"""
Guarantee unique 'time' values across a virtual ITS_LIVE datacube.

autoRIFT makes each granule's time unique by adding a 6-digit numeric hash
of the filename as microseconds to the pair's mid-date. With only 10**6
possible jitters, granules sharing a center second (e.g. Sentinel-2 tiles
from one datatake) collide by the birthday bound once enough exist --
observed in production. Not just cosmetic: xr.combine_by_coords silently
drops all but one same-time granule from a batch, with no error.

Resolved here at cube-build time: a colliding granule is moved by +1us
until it reaches a free slot. Times already committed to a cube are never
moved, so append-only updates stay valid.

Identity is the exact value the cube stores on disk, not a freshly-decoded
datetime64 (see existing_time_values()'s raw zarr read) -- new granules are
compared via encode_time_values() (the same CF encoder the writer uses,
verified bit-identical to to_zarr's own output) and existing layers via
their raw stored values. 'time' is int64 nanoseconds-since-epoch
(Units.ns_epoch_date, lossless) as of Oct 2026; before that it was float64
'seconds since GPS epoch' (off by up to ~0.24us on decode, with re-encoding
reproducing the original float only ~94% of the time) -- a cube already
committed under that old scheme keeps using it for every append.
TIME_UNITS/TIME_DTYPE below reflect only the current convention, so
existing_time_values()'s dtype check will raise for an older cube rather
than silently miscomparing.
"""
import logging

import numpy as np
import xarray as xr
import zarr

import utils

# The cube's pinned 'time' encoding (see virtual_itslive_cube_per_chunk.py's
# create path) -- for a cube created after the Oct 2026 int64 migration
# above; an older cube's on-disk dtype won't match.
TIME_UNITS = utils.Units.ns_epoch_date
TIME_CALENDAR = utils.Units.proleptic_gregorian
TIME_DTYPE = np.dtype(utils.Coords.DTYPE[utils.Coords.TIME])

# Size of each bump applied to a colliding time.
STEP_US = 1


def encode_time_values(times):
   """Encode datetime64 values exactly as the cube's writer stores them.

   Args:
      times (array-like of datetime64): decoded times.

   Returns:
      np.ndarray of int64: nanoseconds since epoch (TIME_UNITS) -- bit-
         identical to the values to_zarr()/to_icechunk() write for the
         cube's 'time' coordinate (xarray's encoder is elementwise, so
         this holds regardless of which other times share a write).
   """
   values, _, _ = xr.coding.times.encode_cf_datetime(
      np.asarray(times, dtype='datetime64[ns]'), TIME_UNITS, TIME_CALENDAR,
      dtype=TIME_DTYPE
   )
   return np.asarray(values, dtype=TIME_DTYPE)


def _resolve(values, tie_break, taken, candidate):
   """Claim a unique value for every entry, bumping collisions.

   Entries are processed in (value, tie_break) order, so the outcome is
   deterministic regardless of input order: the first claimant of a value
   keeps it, and each later one tries candidate(i, 1), candidate(i, 2), ...
   until it finds a value not in `taken`.

   Args:
      values (np.ndarray of TIME_DTYPE): each entry's own value.
      tie_break (array-like of str): per-entry tie-breaker (granule URL).
      taken (set of TIME_DTYPE-native Python scalar): values already
         claimed; updated in place.
      candidate (callable): (index, step count) -> TIME_DTYPE value to try.

   Returns:
      np.ndarray of int64: steps applied per entry (0 where unchanged).
   """
   tie_break = np.asarray(tie_break)
   if values.shape != tie_break.shape:
      raise ValueError(
         f'values ({values.shape}) and tie-breakers ({tie_break.shape}) must '
         'have the same shape'
      )
   if not np.isfinite(values).all():
      raise ValueError(f"non-finite '{utils.Coords.TIME}' value(s): {values[~np.isfinite(values)][:3]}")

   steps = np.zeros(values.shape, dtype=np.int64)
   for i in np.lexsort((tie_break, values)):
      # .item(), not float(): TIME_DTYPE is int64 (~1.5e18 ns for a
      # realistic date) -- float() would re-collapse distinct values at
      # that magnitude, reintroducing the precision loss int64 avoids.
      value = values[i].item()
      while value in taken:
         steps[i] += 1
         value = candidate(i, steps[i]).item()
      taken.add(value)

   return steps


def existing_time_values(store):
   """Raw stored 'time' values of an existing cube, as the set new granules
   must avoid.

   Read straight from zarr (not through xarray) so they're the exact
   on-disk values (TIME_DTYPE) -- decode/re-encode doesn't always round-
   trip under the pre-Oct-2026 float64 scheme (see module docstring), and
   raw.tolist() avoids ever casting through float regardless.

   Args:
      store: zarr store holding the cube (e.g. an icechunk session store).

   Returns:
      set of TIME_DTYPE-native Python scalar. Raises RuntimeError if the
         cube already holds duplicate times -- it was built before this
         fix (and likely also lost granules to combine_by_coords), so it
         must be regenerated from scratch; appending can't repair layers
         already committed.
   """
   raw = zarr.open_group(store, mode='r', zarr_format=3, use_consolidated=False)[utils.Coords.TIME][:]
   if raw.dtype != TIME_DTYPE:
      raise RuntimeError(
         f"Expected '{utils.Coords.TIME}' stored as {TIME_DTYPE} "
         f"'{TIME_UNITS}', got {raw.dtype}"
      )

   values = set(raw.tolist())
   if len(values) != raw.size:
      raise RuntimeError(
         f"Existing cube already has {raw.size - len(values)} duplicate "
         f"'{utils.Coords.TIME}' value(s) -- it predates the time-collision "
         f"fix and must be regenerated from scratch rather than appended to."
      )

   return values


def uniquify_granule_times(datasets, taken, url_attr='granule_url'):
   """Make every single-layer granule dataset's time unique, in place.

   Must run on the per-granule datasets BEFORE they're stacked with
   xr.combine_by_coords, which silently drops all but one of any granules
   sharing a time value.

   A moved granule's time is shifted by whole microseconds and keeps its
   original attrs/encoding; untouched granules keep their exact value.

   Args:
      datasets (list of xr.Dataset): single-layer granule datasets, each
         with a length-1 'time' coordinate and `url_attr` in its attrs.
         Entries whose time changes are replaced in the list.
      taken (set of TIME_DTYPE-native Python scalar): encoded time values
         already claimed -- layers already committed to the cube
         (existing_time_values()) plus earlier batches of this run.
         Updated in place with every value this call assigns, so passing
         the same set across batches keeps the whole cube unique.
      url_attr (str): dataset attribute holding the granule URL, used as
         the deterministic tie-breaker.

   Returns:
      int: number of granules whose time was changed.
   """
   if not datasets:
      return 0

   for ds in datasets:
      if ds.sizes.get(utils.Coords.TIME) != 1:
         raise ValueError(
            f"{ds.attrs.get(url_attr, '<unknown>')}: expected exactly one "
            f"'{utils.Coords.TIME}' value per granule, got "
            f"{ds.sizes.get(utils.Coords.TIME)}"
         )

   times = np.array(
      [ds[utils.Coords.TIME].values[0] for ds in datasets], dtype='datetime64[ns]'
   )
   urls = [ds.attrs[url_attr] for ds in datasets]
   step = np.timedelta64(STEP_US, 'us')

   steps = _resolve(
      encode_time_values(times), urls, taken,
      lambda i, k: encode_time_values(times[i:i + 1] + k * step)[0]
   )

   for i in np.nonzero(steps)[0]:
      ds = datasets[i]
      time_var = ds[utils.Coords.TIME]
      attrs = time_var.attrs.copy()
      encoding = time_var.encoding.copy()

      new_time = times[i:i + 1] + steps[i] * step
      ds = ds.assign_coords({utils.Coords.TIME: new_time})
      ds[utils.Coords.TIME].attrs = attrs
      ds[utils.Coords.TIME].encoding = encoding
      datasets[i] = ds

      logging.info(
         f"Time collision: moved {urls[i]} from {times[i]} to {new_time[0]} "
         f"(+{int(steps[i]) * STEP_US} us)"
      )

   return int(np.count_nonzero(steps))


def assert_layers_kept(cube, num_granules, context=''):
   """Hard-fail if stacking lost any granule or left a duplicate time.

   Args:
      cube (xr.Dataset): the stacked datacube.
      num_granules (int): number of granule datasets that went into it.
      context (str): extra text for the error message.
   """
   num_layers = cube.sizes[utils.Coords.TIME]
   if num_layers != num_granules:
      raise RuntimeError(
         f'{context}stacked cube has {num_layers} layer(s) but '
         f'{num_granules} granule(s) went in -- combine_by_coords dropped '
         f'{num_granules - num_layers}, which means duplicate time values '
         f'reached it.'
      )

   if not cube.indexes[utils.Coords.TIME].is_unique:
      raise RuntimeError(f"{context}stacked cube has duplicate '{utils.Coords.TIME}' values")
