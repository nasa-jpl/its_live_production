"""
Build a virtual ITS_LIVE datacube restricted to a bounding box smaller than
the granules' combined extent -- one 512x512-pixel chunk, co-aligned with
every ITS_LIVE granule.

Each cropped granule is handed to virtual_itslive_cube.py's
build_virtual_cube(), reusing its padding / combine_by_coords /
img_pair_info-handling logic unchanged.
"""
import boto3
from dateutil.parser import parse
from datetime import datetime, timedelta
from joblib import Parallel, delayed, parallel_config
import json
import logging
import numpy as np
import pyproj
import shutil
import obstore
from obstore.store import S3Store
import virtualizarr as vz
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
import icechunk as ic
import xarray as xr

from virtual_itslive_cube import (
   _drop_nonfinite_attrs,
   _get_manifestarray_chunks,
   build_virtual_cube,
)
from deep_copy_cube import TIME_CHUNK_VALUE_1D
from time_collisions import (
   assert_layers_kept,
   existing_time_values,
   uniquify_granule_times,
)

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.utils import copy_and_replace_metadata

from zarr.codecs import BloscCodec, BloscShuffle
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.array_spec import ArraySpec, ArrayConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

import itslive_catalog_utils
import itslive_utils
import utils
import shapefile
from itslive_binary_type import BinaryFlag
from itscube_types import (
   CubeFormat,
   ImgPairInfo,
   Mapping,
   Vars,
   SkippedGranules
)
# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress informational warnings for fixed-length UTF32 dtypes (<U2, <U3,
# etc.) -- no stable Zarr V3 spec yet. Filter by category, not message text
# (the message never names the class).
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Grid pixel size in meters
PIXEL_SIZE = 120
PIXEL_SIZE_HALF = PIXEL_SIZE / 2

# Number of threads for parallel processing
MAX_AWS_CONNECTIONS = 8

# Widens obstore's S3Store defaults (max_retries=10, retry_timeout=3min,
# max_backoff=15s) to ride out multi-minute S3 503 "SlowDown" bursts from
# many jobs starting at once. Safe past the 5-minute credential-expiry
# caveat since these requests are anonymous (skip_signature=True).
RETRY_CONFIG = {
   'max_retries': 20,
   'retry_timeout': timedelta(minutes=5),
   'backoff': {
      'init_backoff': timedelta(milliseconds=500),
      'max_backoff': timedelta(seconds=30),
      'base': 2,
   },
}

# Log progress after this many granules complete. Tasks are dispatched to the
# pool continuously (no batch barrier); this only controls how often progress
# is reported.
PROGRESS_LOG_INTERVAL = 100

# String representation of longitude/latitude projection
LON_LAT_PROJECTION = 'EPSG:4326'

HTTPS_URL = 'https://its-live-data.s3.amazonaws.com/'
S3_URL = 's3://its-live-data/'

# P000 granules are placeholder/degenerate pairs (zero-offset pair with itself)
# that never carry usable velocity data; both this script and
# virtual_itslive_cube_per_chunk_update.py filter them out by filename suffix.
# Shared here so the two call sites can't drift out of sync.
P000_SUFFIX = 'P000.nc'


def skipped_granules_path(cube_store):
   """Path to the skipped-granules JSON sidecar for `cube_store`.

   Args:
      cube_store (str): icechunk repository path (S3 or local).

   Returns: sidecar JSON path.
   """
   return cube_store.rstrip('/').rstrip('.icechunk') + '_skippedGranules.json'


def save_skipped_granules(cube_store, skipped_granules):
   """Write the skipped-granules JSON sidecar, normalizing every URL to
   https:// form (de-duplicated) so the same granule can't appear twice
   under different string forms. Shared with
   virtual_itslive_cube_per_chunk_update.py so both scripts write this file
   the same way.

   Args:
      cube_store (str): icechunk repository path (S3 or local).
      skipped_granules (list of str): skipped granule URLs, in s3://,
         https://, or a mix of both.
   """
   skipped_path = skipped_granules_path(cube_store)
   is_s3 = skipped_path.startswith('s3://')

   skipped_granules = list(set(
      url.replace(S3_URL, HTTPS_URL) for url in skipped_granules
   ))

   if is_s3:
      # Write to S3 using boto3
      s3_client = boto3.client('s3', region_name='us-west-2')
      s3_parts = skipped_path.replace('s3://', '').split('/', 1)
      bucket = s3_parts[0]
      key = s3_parts[1] if len(s3_parts) > 1 else ''

      s3_client.put_object(
         Bucket=bucket,
         Key=key,
         Body=json.dumps(skipped_granules, indent=2),
         ContentType='application/json'
      )
   else:
      # Write to local filesystem
      with open(skipped_path, 'w') as f:
         json.dump(skipped_granules, f, indent=2)

   logging.info(f'Saved {len(skipped_granules)} skipped granules to {skipped_path}')


def crop_manifestarray(marr, starts, stops):
   """Chunk-aligned crop to only the chunks covering [starts[i], stops[i])
   per axis -- mirror image of virtual_itslive_cube.py's
   `pad_manifestarray`. Only chunk references move, no pixel data is read.
   Raises ValueError if starts/stops are malformed or not chunk-aligned.

   Args:
      marr (ManifestArray): array to crop.
      starts (sequence of int): per-axis start index; must be a multiple of
         that axis' chunk size.
      stops (sequence of int): per-axis exclusive end index; must be a
         multiple of the chunk size, except it may equal the axis length to
         reach the array's (legitimately partial) last chunk.

   Returns: cropped ManifestArray, same dtype/chunk structure as `marr`.
   """
   shape = marr.shape
   chunks = _get_manifestarray_chunks(marr)
   starts = tuple(int(s) for s in starts)
   stops = tuple(int(s) for s in stops)

   if len(starts) != len(shape) or len(stops) != len(shape):
      raise ValueError(
         f"starts/stops ndim ({len(starts)}/{len(stops)}) != array ndim {len(shape)}"
      )

   for ax, (start, stop, chunk, n) in enumerate(zip(starts, stops, chunks, shape)):
      if not (0 <= start < stop <= n):
         raise ValueError(f"axis {ax}: invalid range [{start}, {stop}] for size {n}")

      if start % chunk != 0:
         raise ValueError(f"axis {ax}: {start=} not a multiple of {chunk=} size")

      if stop % chunk != 0 and stop != n:
         raise ValueError(
               f"axis {ax}: {stop=} is not a multiple of {chunk=} and does not "
               f"reach the array edge ({stop=} != {n}); cannot crop on a chunk boundary"
         )

   # element bounds -> chunk-grid index bounds (ceil-div for the stop side,
   # so a partial trailing chunk that starts before `stop` is still included)
   chunk_starts = [start // chunk for start, chunk in zip(starts, chunks)]
   chunk_stops = [-(-stop // chunk) for stop, chunk in zip(stops, chunks)]

   region = tuple(slice(cs, ce) for cs, ce in zip(chunk_starts, chunk_stops))

   manifest = marr.manifest
   new_paths = manifest._paths[region]
   new_offsets = manifest._offsets[region]
   new_lengths = manifest._lengths[region]

   from virtualizarr.manifests import ChunkManifest

   new_manifest = ChunkManifest.from_arrays(
      paths=new_paths,
      offsets=new_offsets,
      lengths=new_lengths,
      validate_paths=False,  # references already validated in the source manifest
      inlined=manifest._inlined or None,
   )

   new_shape = [b - a for a, b in zip(starts, stops)]
   new_metadata = copy_and_replace_metadata(marr.metadata, new_shape=list(new_shape))

   return ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)


# Chunk data for ITS_LIVE granules lives under this bucket prefix; manifest
# paths are absolute s3:// URLs, but obstore range reads take a bucket-relative
# key.
_BUCKET_PREFIX = 's3://its-live-data/'


def _compute_allfill_chunk_length(chunk_shape, np_dtype, zarr_dtype, fill_value, codecs):
   """Encoded byte length of an all-fill chunk -- a reference length used to
   short-circuit `_cropped_var_has_valid_data`: a chunk whose encoded length
   differs from this CANNOT be all-fill, so its S3 read + decode can be
   skipped.

   Args:
      chunk_shape (tuple): shape of the chunk.
      np_dtype (numpy.dtype): dtype to build the fill array with (from
         `ManifestArray.dtype`).
      zarr_dtype (zarr dtype): dtype for the encode `ArraySpec` (from
         `metadata.dtype`).
      fill_value (scalar): fill value for the array.
      codecs (list): zarr codec pipeline.

   Returns: encoded byte length (int) of an all-fill chunk.
   """
   fill_chunk = np.full(chunk_shape, fill_value, dtype=np_dtype)
   prototype = default_buffer_prototype()
   pipeline = BatchedCodecPipeline.from_codecs(codecs)
   spec = ArraySpec(
      shape=chunk_shape,
      dtype=zarr_dtype,
      fill_value=fill_value,
      config=ArrayConfig.from_dict({}),
      prototype=prototype,
   )
   encoded = sync(pipeline.encode([(prototype.nd_buffer.from_numpy_array(fill_chunk), spec)]))[0]
   return len(encoded.as_numpy_array().tobytes())


def _cropped_var_has_valid_data(marr, netcdf_store, allfill_chunk_len):
   """True if a cropped ManifestArray references any non-fill data.

   Reads only the referenced chunk bytes via S3 byte-range GETs + the
   array's own zarr codec, instead of downloading the whole granule.
   "Valid" means any element differs from the fill value (equivalent to the
   old `np.isnan(cf_decoded_v).all()` test, since v is int16 and
   `_FillValue` is its only NaN source). Missing chunks read as fill and
   are skipped. Short-circuits to True without decoding if a chunk's
   encoded length differs from `allfill_chunk_len` (can't be all-fill).

   Args:
      marr (ManifestArray): already cropped to the target window.
      netcdf_store (obstore.store.S3Store): store to fetch chunk bytes from.
      allfill_chunk_len (int): pre-computed encoded length of an all-fill
         chunk, passed down from build_virtual_cube_subset() to avoid
         recomputing per granule.

   Returns: bool -- True if any referenced chunk has a non-fill value.
   """
   metadata = marr.metadata
   fill = metadata.fill_value
   prototype = default_buffer_prototype()
   pipeline = BatchedCodecPipeline.from_codecs(metadata.codecs)
   chunk_shape = _get_manifestarray_chunks(marr)

   # Per-chunk decode spec: shape is the chunk shape (not the window); dtype,
   # fill value and codecs come from the array metadata.
   spec = ArraySpec(
      shape=chunk_shape,
      dtype=metadata.dtype,
      fill_value=fill,
      config=ArrayConfig.from_dict({}),
      prototype=prototype,
   )

   # allfill_chunk_len is now passed in from build_virtual_cube_subset() (computed once
   # for all granules), so no need to recompute it here.

   manifest = marr.manifest
   for path, offset, length in zip(
      manifest._paths.flat, manifest._offsets.flat, manifest._lengths.flat
   ):
      if not path:
         # Missing chunk -> reads back as fill, no valid data contributed
         continue

      # Assumes 'v' shares chunk size/dtype/fill/codec across all granules
      # (see allfill_chunk_len's computation in build_virtual_cube_subset).
      # If a differently-encoded granule ever violates that, an all-fill
      # chunk there could get a false "has data" positive -- safe-direction
      # (an extra near-empty layer kept, not real data dropped).
      if length != allfill_chunk_len:
         return True

      # Length matches all-fill reference; need to verify by decoding
      key = str(path).replace(_BUCKET_PREFIX, '')
      raw = bytes(obstore.get_range(
         netcdf_store, key, start=int(offset), length=int(length)
      ))
      buffer = prototype.buffer.from_bytes(raw)
      chunk = sync(pipeline.decode([(buffer, spec)]))[0].as_numpy_array()

      if not np.all(chunk == fill):
         return True

   return False


def bbox_to_chunk_aligned_indices(coord, step, chunk_size, bbox_lo, bbox_hi):
   """Map [bbox_lo, bbox_hi] onto chunk-grid-aligned element indices
   [start, stop) into `coord`.

   Args:
      coord (numpy.ndarray): regularly-spaced 1-D coordinate vector (x or y).
      step (float): spacing between coordinate values; may be negative for
         a descending axis (e.g. y).
      chunk_size (int): chunk size along this dimension, in elements.
      bbox_lo (float): lower bound of the target range.
      bbox_hi (float): upper bound of the target range.

   Returns: (start, stop) chunk-aligned indices into `coord`, `stop`
      exclusive, or None if the bbox doesn't overlap `coord` at all.
   """
   n = len(coord)
   p0 = (bbox_lo - coord[0]) / step
   p1 = (bbox_hi - coord[0]) / step
   lo_idx, hi_idx = (p0, p1) if step > 0 else (p1, p0)

   raw_start = int(np.floor(lo_idx))
   raw_stop = int(np.ceil(hi_idx)) + 1  # +1: hi_idx is a pixel *center*
   raw_start = max(raw_start, 0)
   raw_stop = min(raw_stop, n)
   if raw_start >= raw_stop:
      return None

   start = (raw_start // chunk_size) * chunk_size
   stop = min(n, int(np.ceil(raw_stop / chunk_size)) * chunk_size)

   return start, stop


def crop_virtual_dataset_to_bbox(vds, bbox, netcdf_store, allfill_chunk_len):
   """Crop one granule's virtual dataset to the chunk-grid-aligned window
   covering `bbox`.

   Args:
      vds (xr.Dataset): one granule's virtual dataset (x/y/time loaded, data
         vars virtual), as returned by `open_virtual_dataset`.
      bbox (xmin, xmax, ymin, ymax): target region in native x/y units,
         adjusted to cell centers (the bounding polygon is at cell corners).
      netcdf_store (obstore.store.S3Store): store to read granule data from.
      allfill_chunk_len (int): pre-computed encoded length of an all-fill
         'v' chunk, for the valid-data short-circuit.

   Returns:
      tuple of (xr.Dataset or None, str): the cropped dataset, or None if
      this granule doesn't overlap `bbox` or has no valid data in the
      overlap; and the granule URL.
   """
   xmin, xmax, ymin, ymax = bbox
   x = vds["x"].values
   y = vds["y"].values
   dx = float(x[1] - x[0])
   dy = float(y[1] - y[0])

   # pull x/y chunk size off the first virtual data var that has those dims
   x_chunk = y_chunk = None
   for var in vds.data_vars.values():
      data = var.data
      if isinstance(data, ManifestArray):
         dims = var.dims

         # Get chunk shape using helper function
         try:
            chunks = _get_manifestarray_chunks(data)

            if x_chunk is None and "x" in dims:
               x_chunk = chunks[dims.index("x")]
            if y_chunk is None and "y" in dims:
               y_chunk = chunks[dims.index("y")]
         except (AttributeError, IndexError) as e:
            logging.warning(f"Could not get chunks from variable: {e}")
            continue

      if x_chunk is not None and y_chunk is not None:
         break

   if x_chunk is None or y_chunk is None:
      raise ValueError("could not determine x/y chunk size from this dataset's data vars")

   logging.debug(f'Granule x: {x[0]=} {x[-1]}')
   logging.debug(f'Granule y: {y[0]=} {y[-1]}')
   logging.debug(f'Cube polygon: {xmin=} {xmax=} {ymin=} {ymax=}')

   x_range = bbox_to_chunk_aligned_indices(x, dx, x_chunk, xmin, xmax)
   y_range = bbox_to_chunk_aligned_indices(y, dy, y_chunk, ymin, ymax)

   logging.debug(f'{x_range=}')
   logging.debug(f'{y_range=}')

   if x_range is None or y_range is None:
      # Granule doesn't intersect the bounding bbox
      logging.debug(f'{vds.attrs["granule_url"]} does not overlap the polygon')
      return None, vds.attrs["granule_url"]

   logging.debug(f'Updating to {x_range=}')
   logging.debug(f'Updating to {y_range=}')

   x_start, x_stop = x_range
   y_start, y_stop = y_range
   start_by_dim = {"x": x_start, "y": y_start}
   stop_by_dim = {"x": x_stop, "y": y_stop}

   # Crop v to the window first, then check for valid data via v's chunk
   # references (reads only the window's chunk bytes, not the whole granule).
   # The cropped v is reused below so v is only cropped once.
   v_var = vds.data_vars[Vars.v]
   v_starts = [start_by_dim.get(str(d), 0) for d in v_var.dims]
   v_stops = [stop_by_dim.get(str(d), s) for d, s in zip(v_var.dims, v_var.data.shape)]
   cropped_v = crop_manifestarray(v_var.data, v_starts, v_stops)

   if not _cropped_var_has_valid_data(cropped_v, netcdf_store, allfill_chunk_len):
      # Granule does not have any valid data within intersection
      logging.debug(f'{vds.attrs["granule_url"]} does not have valid data within polygon')
      return None, vds.attrs["granule_url"]

   new_vars = {}
   for name, var in vds.data_vars.items():
      data = var.data

      if isinstance(data, ManifestArray):
         if name == Vars.v:
            # Reuse the v ManifestArray already cropped for the valid-data check
            data = cropped_v
         else:
            starts = [start_by_dim.get(str(d), 0) for d in var.dims]
            stops = [stop_by_dim.get(str(d), s) for d, s in zip(var.dims, data.shape)]
            data = crop_manifestarray(data, starts, stops)

      new_vars[name] = xr.Variable(var.dims, data, attrs=var.attrs, encoding=var.encoding)

   new_coords = {
      "x": ("x", x[x_start:x_stop]),
      "y": ("y", y[y_start:y_stop]),
      "time": vds["time"],
   }

   return xr.Dataset(new_vars, coords=new_coords, attrs=vds.attrs), \
      vds.attrs["granule_url"]


def _assert_identical_grids(cropped, bbox):
   """Fail loudly if any cropped granule landed on a different x/y grid than
   the first.

   Granules share a common chunk grid (same posting, same chunk boundaries),
   so chunk-aligned cropping to the same `bbox` should produce *identical*
   x/y coordinates across granules -- not merely overlapping ones needing a
   pad-to-common-grid step. Raises ValueError (rather than letting
   `pad_manifestarray` silently paper over it) if that's ever violated, e.g.
   by an unexpectedly offset granule.

   Args:
      cropped (list of xr.Dataset): cropped virtual datasets; at least one
         element.
      bbox (tuple): the (xmin, xmax, ymin, ymax) bbox used for cropping,
         included in error messages for debugging.
   """
   ref_vds = cropped[0]
   ref_x = ref_vds["x"].values
   ref_y = ref_vds["y"].values
   for vds in cropped[1:]:
      x = vds["x"].values
      y = vds["y"].values
      if x.shape != ref_x.shape or not np.array_equal(x, ref_x):
         raise ValueError(
               f"cropped granules disagree on the x grid for bbox {bbox}: "
               f"expected {ref_x.shape} spanning [{ref_x[0]}, {ref_x[-1]}], "
               f"got {x.shape} spanning [{x[0]}, {x[-1]}]. This should not "
               f"happen if all granules share a common chunk grid -- check "
               f"that assumption for this granule."
         )
      if y.shape != ref_y.shape or not np.array_equal(y, ref_y):
         raise ValueError(
               f"cropped granules disagree on the y grid for bbox {bbox}: "
               f"expected {ref_y.shape} spanning [{ref_y[0]}, {ref_y[-1]}], "
               f"got {y.shape} spanning [{y[0]}, {y[-1]}]. This should not "
               f"happen if all granules share a common chunk grid -- check "
               f"that assumption for this granule."
         )


def build_virtual_cube_subset(vds_list, bbox, netcdf_store, taken_time_values):
   """Build a virtual datacube restricted to `bbox`, smaller than the
   granules' combined extent.

   Each granule's ManifestArrays are cropped (chunk-aligned) to the window
   overlapping `bbox`; non-overlapping granules are dropped. Since granules
   share a common chunk grid, the cropped granules are then required to land
   on an identical x/y grid (`_assert_identical_grids`, ValueError if not)
   before being mosaicked via `build_virtual_cube`, which stacks them on
   time and reuses its dtype/attr/img_pair_info handling.

   Before stacking, every granule's time is made unique against
   `taken_time_values` (see time_collisions.uniquify_granule_times): autoRIFT's
   6-digit filename-hash jitter collides in practice, and combine_by_coords
   silently keeps only one of any granules sharing a time. The stacked cube
   is then checked to hold exactly one layer per granule, as a hard failure.

   Args:
      vds_list (list of xr.Dataset): per-granule virtual datasets, as passed
         to `build_virtual_cube` -- 'x'/'y'/'time' coordinates, virtual
         ManifestArray data variables.
      bbox (tuple of float): target region in the granules' shared x/y
         units, as (xmin, xmax, ymin, ymax).
      netcdf_store (obstore.store.S3Store): object store used to check for
         valid data within the overlap region.
      taken_time_values (set of float): encoded 'time' values (float64
         seconds since GPS epoch, as stored) already claimed by the cube --
         existing layers plus earlier batches of this run. Updated in place
         with this batch's times, so the caller passes the same set to every
         batch.

   Returns:
      tuple of (xr.Dataset or None, str or None, list of str): the cube with
      all cropped granules stacked on time (None if no granule had valid
      data in the bbox); the autorift parameter file path (None if no cube
      was built); and the URLs of skipped granules (no overlap, or no valid
      data in the overlap region).
   """
   if not vds_list:
      logging.info("vds_list is empty -- no granules to build a cube from; no cube built")
      return None, None, []

   logging.info(f'Building cube out of {len(vds_list)} granules')
   cropped = []
   skipped_granules = []

   # All granules share 'v's chunk size/dtype/fill/codec, so this reference
   # length is the same for every granule -- compute it once here instead
   # of re-encoding a full 512x512 array per granule.
   sample_v = vds_list[0].data_vars[Vars.v]
   sample_marr = sample_v.data
   allfill_chunk_len = _compute_allfill_chunk_length(
      _get_manifestarray_chunks(sample_marr),
      sample_marr.dtype,
      sample_marr.metadata.dtype,
      sample_marr.metadata.fill_value,
      sample_marr.metadata.codecs
   )
   logging.debug(f'Computed reference all-fill chunk length: {allfill_chunk_len} bytes')

   # Threads, not processes: cropping is pure obstore range reads + numpy
   # slicing + zarr decode (no h5py), so nothing here is thread-unsafe or
   # GIL-bound, and threads avoid pickling ManifestArray datasets to/from a
   # worker process. return_as="generator_unordered" yields each result as
   # it completes, so a slow granule doesn't stall the rest (arrival order
   # doesn't matter -- build_virtual_cube stacks by time coordinate).
   with parallel_config(
      backend='threading',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      result_stream = Parallel(return_as="generator_unordered")(
         delayed(crop_virtual_dataset_to_bbox)(each_vds, bbox, netcdf_store, allfill_chunk_len)
         for each_vds in vds_list
      )

      total = len(vds_list)
      for done, (cropped_ds, cropped_url) in enumerate(result_stream, start=1):
         if cropped_ds is not None:
            cropped.append(cropped_ds)

         else:
            skipped_granules.append(cropped_url)

         if done % PROGRESS_LOG_INTERVAL == 0 or done == total:
            logging.info(
               f"Cropped {done}/{total} granules "
               f"({len(cropped)} kept, {len(skipped_granules)} skipped)"
            )

   if not cropped:
      # No granule had valid data within the bbox: report and return no cube so
      # the caller can skip the rest of the processing instead of failing.
      logging.info(
         f"No granules overlap bbox {bbox} with valid data; no cube built"
      )
      return None, None, skipped_granules

   logging.info(f'Got {len(cropped)} cropped granules')

   if logging.getLogger().isEnabledFor(logging.DEBUG):
      for i, vds in enumerate(cropped):
         t = vds["time"]
         logging.debug(f"Granule {i}: {t.values=} {t.dtype=} {t.dims=} {t.shape=}")

   _assert_identical_grids(cropped, bbox)

   # Must happen before build_virtual_cube: its combine_by_coords silently
   # drops all but one of any granules sharing a time value. Ties resolve by
   # granule URL, so the result doesn't depend on the unordered arrival
   # order of the cropping pass above.
   num_bumped = uniquify_granule_times(cropped, taken_time_values)
   if num_bumped:
      logging.info(f'Resolved {num_bumped} duplicate time value(s) in this batch')

   logging.info(f'Number of skipped granules: {len(skipped_granules)}')
   # All cropped granules are on identical grids (verified by _assert_identical_grids),
   # so skip the extend_coords() step in build_virtual_cube
   cube, autorift_param_file = build_virtual_cube(cropped, already_aligned=True)

   # Every granule must come out as its own layer -- a mismatch means a
   # duplicate time slipped past uniquify_granule_times and a granule was
   # silently dropped.
   assert_layers_kept(cube, len(cropped), context=f'bbox {bbox}: ')

   return cube, autorift_param_file, skipped_granules


def _granule_exists(granule_url, store, bucket_prefix):
   """Cheap HEAD check for whether `granule_url` exists in S3.

   Temporary bypass for a known searchAPI/catalog bug returning URLs with
   no real S3 object. Runs before the far more expensive
   `open_virtual_dataset()` in `read_virtual_dataset`, so a missing granule
   never triggers that function's fail-fast handling; any other open
   failure still does.

   Args:
      granule_url (str): full s3:// URL to the granule.
      store (obstore.store.S3Store): object store for the granules bucket.
      bucket_prefix (str): s3:// prefix stripped from `granule_url` to get
         the bucket-relative key.

   Returns: bool -- False only on a 404 (`FileNotFoundError`).
   """
   key = granule_url.replace(bucket_prefix, '')
   try:
      obstore.head(store, key)
      return True
   except FileNotFoundError:
      return False


def _check_granule_exists(granule_url, store, bucket_prefix):
   """Parallel-map wrapper pairing `_granule_exists`'s result with its URL,
   since `Parallel(return_as="generator_unordered")` yields out of order
   (mirrors `crop_virtual_dataset_to_bbox`'s `(result, url)` convention for
   the same reason).

   Args:
      granule_url (str): granule URL to check.
      store (obstore.store.S3Store): object store for the granules bucket.
      bucket_prefix (str): s3:// URL prefix stripped to get the
         bucket-relative key.

   Returns: (granule_url, exists) tuple of (str, bool).
   """
   return granule_url, _granule_exists(granule_url, store, bucket_prefix)


def read_virtual_dataset(granule_url, parser, registry):
   """Read one granule into a virtual dataset. Raises RuntimeError
   immediately on failure -- a granule that can't be opened (bad input list,
   broken S3 access, etc.) signals a problem worth stopping the run for, not
   one to silently paper over (see `load_granules`).

   Args:
      granule_url (str): S3 URL to the granule file.
      parser (virtualizarr.parsers.HDFParser): parser for reading
         HDF/NetCDF as a virtual dataset.
      registry (obspec_utils.registry.ObjectStoreRegistry): maps URL
         prefixes to object stores for chunk access.

   Returns: xr.Dataset with 'time'/'y'/'x' loaded, data variables as
      ManifestArrays, and 'granule_url'/'granule_path' set in attrs.
   """
   try:
      v = vz.open_virtual_dataset(
         url=granule_url,
         parser=parser,
         registry=registry,
         loadable_variables=["time", "y", "x", Mapping.name],
         decode_times=True,
      )

      # Remember the granule url
      v.attrs["granule_url"] = granule_url
      v.attrs["granule_path"] = granule_url.replace('s3://its-live-data/', '')

   except Exception as e:
      raise RuntimeError(f'Got exception loading {granule_url=}: {e}')

   return v


def load_granules(granules, bucket):
   """Load granules into virtual datasets in parallel (ManifestArray data
   vars, no pixel data loaded).

   First HEAD-checks each granule (`_granule_exists`'s searchAPI/catalog-bug
   bypass) and drops any that are genuinely missing (404) before calling
   `read_virtual_dataset`, so a missing granule never reaches -- and never
   triggers -- that function's fail-fast RuntimeError. Any other open
   failure still propagates and aborts the whole run.

   Args:
      granules (list of str): S3 URLs or paths to granule files.
      bucket (str): S3 bucket URL storing the granules.

   Returns:
      tuple of (list of xr.Dataset, list of str): virtual datasets (order
      not guaranteed to match input -- collected as-completed; downstream
      stacking keys off the time coordinate, not list position), and the
      subset of `granules` that don't exist in S3, for the caller to record
      in the persistent skipped-granules JSON.
   """
   store = obstore.store.from_url(
      bucket, region="us-west-2", skip_signature=True, retry_config=RETRY_CONFIG,
   )
   registry = ObjectStoreRegistry({bucket: store})
   # Keep 'mapping': loaded as a small 0-dim variable so build_virtual_cube
   # can recover its projection attrs for the cube's CF grid-mapping var.
   parser = HDFParser()

   # HEAD-check existence before the much more expensive open_virtual_dataset()
   # below (see _granule_exists). Threads, not processes: same reasoning as
   # the cropping pass in build_virtual_cube_subset.
   bucket_prefix = bucket.rstrip('/') + '/'
   existing_granules = []
   missing_granules = []
   with parallel_config(backend='threading', n_jobs=MAX_AWS_CONNECTIONS):
      result_stream = Parallel(return_as="generator_unordered")(
         delayed(_check_granule_exists)(each_url, store, bucket_prefix)
         for each_url in granules
      )
      for url, exists in result_stream:
         (existing_granules if exists else missing_granules).append(url)

   if missing_granules:
      logging.warning(
         f"{len(missing_granules)} granules reported by searchAPI are "
         f"missing from S3 (known catalog issue) -- skipping: "
         f"{missing_granules[:5]}{'...' if len(missing_granules) > 5 else ''}"
      )
   granules = existing_granules

   vds_list = []

   # Processes ("loky"), not threads: HDF parsing's thread-safety isn't
   # guaranteed (unlike the h5py-free crop pass in build_virtual_cube_subset).
   # return_as="generator_unordered" gives as-completed progress and throttles
   # dispatch (~2*n_jobs) so huge granule lists don't all queue at once;
   # arrival order doesn't matter since downstream keys off time, not position.
   total = len(granules)
   with parallel_config(
      backend='loky',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      result_stream = Parallel(return_as="generator_unordered")(
         delayed(read_virtual_dataset)(each_file, parser, registry)
         for each_file in granules
      )

      for done, each_ds in enumerate(result_stream, start=1):
         vds_list.append(each_ds)

         if done % PROGRESS_LOG_INTERVAL == 0 or done == total:
            logging.info(f"Loaded {done}/{total} granules")

   return vds_list, missing_granules


# ---------------------------------------------------------------------------
# Append-to-existing-repo functions (merged from
# virtual_itslive_cube_per_chunk_update.py): lets __main__ auto-detect an
# existing --output-store and append instead of recreating, so a job killed
# mid-run can be safely re-submitted with the same arguments.
# ---------------------------------------------------------------------------

def _build_output_storage(store_path):
   """Build the icechunk `Storage` handle for `store_path`.

   Args:
      store_path (str): icechunk repository path (s3:// URL or local).

   Returns: icechunk.Storage, usable with `ic.Repository.exists`/`.open`/
      `.create`.
   """
   if store_path.startswith(utils.S3_PREFIX):
      s3_parts = store_path.replace(utils.S3_PREFIX, '').split('/', 1)
      out_bucket = s3_parts[0]
      prefix = s3_parts[1] if len(s3_parts) > 1 else ''
      logging.info(f'Using icechunk repo on S3: bucket={out_bucket}, prefix={prefix}')
      return ic.s3_storage(bucket=out_bucket, prefix=prefix, region="us-west-2")

   logging.info(f'Using icechunk repo on local filesystem: {store_path}')
   return ic.local_filesystem_storage(store_path)


def open_repo_for_append(store_path, url_prefix):
   """Open a pre-existing icechunk repository to append new granules to.

   Mirrors __main__'s RepositoryConfig/credentials setup for a fresh repo
   (StorageSettings for stronger recovery from transient S3 failures,
   anonymous virtual-chunk-container credentials for the granules bucket),
   but opens rather than creates.

   Args:
      store_path (str): icechunk repository path (s3:// URL or local).
      url_prefix (str): s3:// URL prefix (trailing slash) for the granules
         bucket, used to authorize anonymous virtual-chunk access.

   Returns:
      tuple of (ic.Repository, xr.Dataset): the opened repository and its
      current datacube, read with mask_and_scale=False (raw on-disk dtypes,
      matching every other cube read in this file).
   """
   config = ic.RepositoryConfig.default()

   if store_path.startswith(utils.S3_PREFIX):
      # unsafe_use_metadata/unsafe_use_conditional_update only work with S3
      # storage -- see the create path's local-filesystem branch, which
      # doesn't set config.storage either.
      config.storage = ic.StorageSettings(
         unsafe_use_metadata=True,
         unsafe_use_conditional_update=True
      )

   config.set_virtual_chunk_container(
      ic.VirtualChunkContainer(url_prefix, ic.s3_store(region="us-west-2", anonymous=True))
   )

   repo = ic.Repository.open(
      storage=_build_output_storage(store_path),
      config=config,
      authorize_virtual_chunk_access=ic.containers_credentials(
         {url_prefix: ic.s3_credentials(anonymous=True)}
      ),
   )

   # zarr_format=3, not 2: icechunk repos are natively Zarr V3 metadata;
   # forcing zarr_format=2 raises GroupNotFoundError (no .zgroup/.zarray
   # markers exist in an icechunk store).
   cube = xr.open_zarr(
      repo.readonly_session("main").store,
      consolidated=False,
      zarr_format=3,
      mask_and_scale=False
   )

   return repo, cube


def load_skipped_granules(cube_store):
   """Load a cube's persistent skipped-granules JSON. Raises RuntimeError
   if no such file exists yet at `cube_store`'s expected path.

   Args:
      cube_store (str): icechunk repository path (S3 or local).

   Returns: set of skipped granule URLs, normalized to s3:// form.
   """
   skipped_path = skipped_granules_path(cube_store)
   is_s3 = skipped_path.startswith(utils.S3_PREFIX)

   try:
      if is_s3:
         s3_client = boto3.client('s3', region_name='us-west-2')
         s3_parts = skipped_path.replace(utils.S3_PREFIX, '').split('/', 1)
         bucket = s3_parts[0]
         key = s3_parts[1] if len(s3_parts) > 1 else ''

         response = s3_client.get_object(Bucket=bucket, Key=key)
         content = response['Body'].read().decode('utf-8')
         skipped = json.loads(content)

      else:
         with open(skipped_path, 'r') as f:
            skipped = json.load(f)

      skipped_set = set(url.replace(HTTPS_URL, S3_URL) for url in skipped)
      logging.info(f'Loaded {len(skipped_set)} previously skipped granules from {skipped_path}')

      return skipped_set

   except FileNotFoundError:
      raise RuntimeError(f'No existing skipped granules file at {skipped_path}')

   except boto3.exceptions.botocore.exceptions.ClientError as e:
      error_code = e.response.get('Error', {}).get('Code')
      if error_code in ('NoSuchKey', '404'):
         raise RuntimeError(f'No existing skipped granules file at {skipped_path}')

      # Any other ClientError (permission denied, wrong region, throttling,
      # expired credentials, bucket typo, etc.) is a real problem -- don't
      # mask it behind a misleading "file not found" message
      raise


def get_existing_granule_urls(cube):
   """Extract existing granule URLs from a datacube.

   Args:
      cube (xr.Dataset): the virtual datacube.

   Returns: set of granule URLs already in the cube, normalized to s3://
      form.
   """
   urls = set(str(u).replace(HTTPS_URL, S3_URL) for u in cube[Vars.url].values)
   logging.info(f'Found {len(urls)} existing granules in cube')

   return urls


def _filter_processed_granules(urls, skipped, existing, num_p000_skipped):
   """Drop granules already committed to the cube (if any) or previously
   skipped. `existing` is empty when no repo exists yet, so this also
   covers the no-repo-but-sidecar-exists case. P000 granules are not
   filtered here -- __main__ already excludes them from `urls` before
   this is ever called.

   Args:
      urls (list of str): candidate granule URLs (s3:// form), sorted
         chronologically.
      skipped (set of str): previously skipped granule URLs (s3:// form).
      existing (set of str): granule URLs already in the cube (s3:// form).
      num_p000_skipped (int): P000 granules already excluded from `urls`,
         for the log line only.

   Returns:
      tuple of (list of str, list of str): remaining granules to process
      (`urls` order); and the subset of `urls` found in `skipped` (excludes
      granules dropped for being `existing` -- those are successfully
      processed, not skipped, and must never reach the skipped-granules
      record).
   """
   remaining = []
   already_skipped = []
   already_in_cube = 0
   for url in urls:
      if url in skipped:
         already_skipped.append(url)
      elif url in existing:
         already_in_cube += 1
      else:
         remaining.append(url)

   logging.info(
      f'Filtered out {len(already_skipped)} previously-skipped, '
      f'{num_p000_skipped} P000 skipped, and '
      f'{already_in_cube} already-in-cube granules; {len(remaining)} new '
      'granules remain'
   )
   return remaining, already_skipped


def set_1d_time_chunk_encoding(cube, chunk_size):
   """Set an explicit 'chunks' encoding on every 1-D (time,) variable and
   the 'time' coordinate, overriding xarray/icechunk's default of one chunk
   spanning the whole write (see TIME_CHUNK_VALUE_1D). Must run before the
   cube's first write -- a Zarr chunk grid is fixed at creation.

   Iterates cube.variables (not just data_vars) so 'time' gets the same
   treatment; the 3D ManifestArray variables (chunk fixed at 1, see
   virtual_itslive_cube.py's np.concatenate handling) and 'x'/'y' don't
   match the (TIME,) filter and are left alone.

   Args:
      cube (xr.Dataset): the virtual cube about to be written (first batch
         only).
      chunk_size (int): 'time' chunk size to set.
   """
   for var_name in cube.variables:
      var = cube[var_name]
      if var.dims == (utils.Coords.TIME,):
         var.encoding[utils.OutputFormat.chunks] = (chunk_size,)


if __name__ == "__main__":
   import argparse
   import sys
   import os
   from joblib.externals.loky import get_reusable_executor
   import time

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Build a virtual ITS_LIVE datacube from granules, restricted to a bounding box.

      Usage examples:
      # Using JSON file for granules
      python src/virtual_itslive_cube_per_chunk.py \
         --granules-file granules.json \
         --polygon '[[-1658887.5, -430072.5], [-1597447.5, -430072.5], [-1597447.5, -368632.5], [-1658887.5, -368632.5], [-1658887.5, -430072.5]]' \
         --output-store output.icechunk

      # Using 4 granules with valid data in cube's polygon
      python ./virtual_itslive_cube_per_chunk.py --polygon '[[-1658887.5, -430072.5], [-1597447.5, -430072.5], [-1597447.5, -368632.5], [-1658887.5, -368632.5], [-1658887.5, -430072.5]]' --granules-file virtual_input_4files.json --output-store its_live_cube_subset_m11_m12_s1_s2_landsat.icechunk

      # Using 39 input granules with only 7 having valid data in cube's polygon:
      python ./virtual_itslive_cube_per_chunk.py --polygon '[[-1658887.5, -430072.5], [-1597447.5, -430072.5], [-1597447.5, -368632.5], [-1658887.5, -368632.5], [-1658887.5, -430072.5]]' --granules-file virtual_input_39files.json --output-store its_live_cube_subset_m11_m12_s1_s2_landsat.icechunk

      # Using direct granule list
      python src/virtual_itslive_cube_per_chunk.py \
         --granules granule1.nc granule2.nc granule3.nc \
         --polygon '[[-1658887.5, -430072.5], [-1597447.5, -430072.5], [-1597447.5, -368632.5], [-1658887.5, -368632.5], [-1658887.5, -430072.5]]'
      """
   )

   # Create mutually exclusive group for granules input
   granules_group = parser.add_mutually_exclusive_group(required=True)
   granules_group.add_argument(
      "--granules-file",
      type=str,
      help="Path to JSON file containing a list of granule paths"
   )
   granules_group.add_argument(
      "--use-searchAPI",
      action='store_true',
      default=False,
      help="Use searchAPI to get list of granules for the bounding box"
   )

   parser.add_argument(
      "--polygon",
      type=str,
      required=True,
      help="Bounding polygon as JSON list of [x,y] coordinates: "
         "'[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]' (closed polygon)"
   )
   parser.add_argument(
      "--output-store",
      type=str,
      default="its_live_cube_subset.icechunk",
      help="Path to output icechunk store (default: its_live_cube_subset.icechunk)"
   )
   parser.add_argument(
      "--bucket",
      type=str,
      default="s3://its-live-data",
      help="S3 bucket URL [%(default)s]"
   )
   parser.add_argument(
      "--bucketHTTP",
      type=str,
      default="https://its-live-data.s3.amazonaws.com",
      help="S3 bucket HTTP URL [%(default)s]"
   )
   parser.add_argument(
      '-t', '--threads',
      type=int,
      default=8,
      help='Number of threads to use for parallel processing [%(default)d].'
   )
   parser.add_argument(
      '-n', '--num-granules',
      type=int,
      default=0,
      help='Number of granules to process [%(default)d meaning to process all granules].'
   )
   parser.add_argument(
      '--batch-size',
      type=int,
      default=10000,
      help='Number of granules to load and commit together per icechunk snapshot '
         '[%(default)d]. Granules are sorted chronologically first, then split '
         'into batches of this size to bound memory use for very large granule '
         'lists; the first batch creates the icechunk repo and each subsequent '
         'batch appends to it.'
   )
   parser.add_argument(
      "--start-date",
      type=lambda s: parse(s).strftime('%Y-%m-%d'),
      default='1982-01-01',
      help="Start date for searchAPI query (required with --use-SearchAPI) [%(default)s]"
   )
   parser.add_argument(
      "--end-date",
      type=lambda s: parse(s).strftime('%Y-%m-%d'),
      default=datetime.now().strftime('%Y-%m-%d'),
      help="End date for searchAPI query (required with --use-SearchAPI) [%(default)s]"
   )
   parser.add_argument(
      '--projection',
      type=str,
      required=True,
      help='UTM target projection for the virtual cube and granules it will be'
         'constructed (required with --use-SearchAPI) [%(default)s]'
   )
   parser.add_argument(
      '--searchType',
      choices=['serverless', 'pgstac'],
      default='serverless',
      help='Granule search backend: "serverless" queries the geoparquet '
         'warehouse via duckdb (default), "pgstac" queries the STAC API '
         'via pystac_client [%(default)s].'
   )
   parser.add_argument(
      '--stacCatalog',
      type=str,
      default=None,
      help='Granule catalog location override. For serverless: s3:// path to '
         'the geoparquet warehouse (default: itslive warehouse). For pgstac: '
         'https:// URL of the STAC API (default: https://stac.itslive.cloud).'
   )
   parser.add_argument(
      '-s', '--shapeFile',
      type=str,
      default='s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp',
      help='Shapefile with ice masks information [%(default)s]'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   MAX_AWS_CONNECTIONS = args.threads

   # Not Darwin-only: resource_tracker's subprocess launches via a fresh
   # `sys.executable` (not fork()), re-reading PYTHONWARNINGS at its own
   # startup regardless of platform -- so this benign, self-remediating
   # "leaked folder objects" UserWarning has been observed on Linux too, not
   # just macOS's original semaphore-tracker noise.
   os.environ.setdefault(
      "PYTHONWARNINGS",
      "ignore::UserWarning:multiprocessing.resource_tracker,"
      "ignore::UserWarning:joblib.externals.loky.backend.resource_tracker"
   )

   # Parse bounding polygon from JSON string in UTM coordinates
   polygon = json.loads(args.polygon)

   # Extract bounding box from polygon
   x_coords = [coord[0] for coord in polygon]
   y_coords = [coord[1] for coord in polygon]

   xmin = min(x_coords)
   xmax = max(x_coords)
   ymin = min(y_coords)
   ymax = max(y_coords)

   logging.info(f"Extracted bbox from polygon: {xmin=}, {xmax=}, {ymin=}, {ymax=}")

   # Adjust cube cell edge coordinates to the cell centers (like in granules)
   xmin = xmin + PIXEL_SIZE_HALF
   xmax = xmax - PIXEL_SIZE_HALF
   ymin = ymin + PIXEL_SIZE_HALF
   ymax = ymax - PIXEL_SIZE_HALF

   # Bounding box using cell centers
   bbox = [xmin, xmax, ymin, ymax]

   xmid = (xmin + xmax) / 2
   ymid = (ymin + ymax) / 2

   # Convert UTM coordinates to lon/lat (ensure lonlat output order)
   to_lon_lat_transformer = pyproj.Transformer.from_crs(
      f"EPSG:{args.projection}", LON_LAT_PROJECTION, always_xy=True
   )

   # Introduce 5 points per each polygon side (cell corners)
   polygon = itslive_utils.add_five_points_to_polygon_side(polygon)

   # Convert polygon from its target projection to longitude/latitude
   # coordinates which are used by granule search API
   polygon_coords = []

   for each in polygon:
      coords = to_lon_lat_transformer.transform(each[0], each[1])
      polygon_coords.append(list(coords))

   # Load granules from either JSON file or command-line arguments
   if args.granules_file:
      logging.info(f"Loading granules from {args.granules_file}")
      with open(args.granules_file, 'r') as f:
         granules = json.load(f)

      if not isinstance(granules, list):
         raise ValueError(f"JSON file must contain a list of granule paths")

   elif args.use_searchAPI:
      # Validate that other arguments are provided when using searchAPI
      if not args.start_date or not args.end_date or not args.projection:
         parser.error(
            "--use-searchAPI requires --start-date, --end-date, --projection arguments"
            " (and optional --searchType, --stacCatalog arguments)"
         )

      itslive_catalog_utils.STAC_CATALOG = args.stacCatalog
      itslive_catalog_utils.SEARCH_TYPE = args.searchType

      roi = {
         "type": "Polygon",
         "coordinates": [polygon_coords]
      }

      granules = itslive_catalog_utils.serverless_search(
         epsg_code=args.projection,
         start_date=args.start_date,
         end_date=args.end_date,
         roi=roi
      )
      logging.info(f'Got {len(granules)} granules from searchAPI')

   # Truncate *before* P000 filtering below, so --num-granules reflects a
   # slice of the raw candidate list ("the first N granules") -- the later
   # "Processing" count can come out lower than N if any of that slice is
   # P000.
   if args.num_granules > 0:
      num_granules = args.num_granules
      granules = granules[:num_granules]

   # P000 granules never have usable data; exclude from processing but still
   # record in the persistent skipped-granules JSON below (merged in after
   # build_virtual_cube_subset). P000 URLs are in original "https://" form.
   p000_granules = [each for each in granules if each.endswith(P000_SUFFIX)]

   # The rest of the granules have "s3://"" url
   granules = [
      each.replace(HTTPS_URL, S3_URL) for each in granules \
      if each.endswith(P000_SUFFIX) is False
   ]

   if p000_granules:
      logging.info(f"Excluding {len(p000_granules)} P000 granules")

   # Sort chronologically by mid_date parsed from each filename (cheap --
   # no granule is opened) so both the cube's layers and the batches below
   # come out in time order.
   granules = sorted(granules, key=utils.extract_mid_date_from_url)

   logging.info(f"Processing {len(granules)} granules")

   bucket = args.bucket
   bucketHTTP = args.bucketHTTP

   # "s3://its-live-data/"
   url_prefix = bucket + os.sep
   store_path = args.output_store

   # Determine if output is S3 or local filesystem
   is_s3_output = store_path.startswith(utils.S3_PREFIX)

   # Accumulates skipped granules across all batches; P000 granules were
   # never handed to any batch, so seed with those up front.
   skipped_granules = list(p000_granules)

   # Detect whether the target repo already exists, so this run appends
   # instead of recreating from scratch -- lets a job be safely re-submitted
   # (e.g. after a transient S3 throttling failure) without clobbering
   # already-committed progress.
   repo_exists = ic.Repository.exists(_build_output_storage(store_path))

   # None until either an existing repo is opened below, or the first batch
   # that actually produces a cube creates one; every later batch with data
   # appends to it.
   repo = None
   existing_urls = set()

   # Encoded 'time' values (as stored on disk) already claimed by the cube:
   # seeded from the existing repo's layers when appending, then grown in
   # place by every batch's build_virtual_cube_subset() call, so each new
   # granule's time is unique against both committed layers and earlier
   # batches of this run. Committed layers are never moved -- only new
   # granules get bumped.
   taken_time_values = set()

   if repo_exists:
      repo, existing_cube = open_repo_for_append(store_path, url_prefix)
      logging.info(
         f'Found existing cube with {len(existing_cube.time)} time layers '
         f'at {store_path}; appending new granules to it'
      )
      existing_urls = get_existing_granule_urls(existing_cube)

      # Raises if the existing cube already holds duplicates -- it predates
      # the time-collision fix and must be regenerated from scratch, not
      # appended to. Read raw from the repo, not from existing_cube's
      # decoded times: decode/re-encode doesn't always round-trip to the
      # stored float (see time_collisions.py).
      taken_time_values = existing_time_values(repo.readonly_session("main").store)

   # Load any previously-recorded skipped granules even if the repo itself
   # doesn't exist yet -- e.g. a prior run where every batch found no valid
   # data in the bbox never created a repo, but still wrote the sidecar
   # after each batch (see the "cube is None" branch below). Re-checking
   # those granules' S3 existence/validity on this run would be pure wasted
   # work, so exclude them regardless of repo_exists.
   try:
      existing_skipped = load_skipped_granules(store_path)
   except RuntimeError:
      logging.info(f'No existing skipped-granules file yet for {store_path}')
      existing_skipped = set()

   if existing_skipped or existing_urls:
      # Second return value (already-skipped granules re-found in this
      # run's candidate list) is always a subset of existing_skipped, so
      # unioning it in below would add nothing -- discarded here.
      granules, _ = _filter_processed_granules(
         granules, existing_skipped, existing_urls, len(skipped_granules))
      skipped_granules = list(set(skipped_granules) | existing_skipped)
      save_skipped_granules(store_path, skipped_granules)

   batch_size = args.batch_size
   batches = [granules[i:i + batch_size] for i in range(0, len(granules), batch_size)]
   num_batches = len(batches)

   # Ties every commit made by this run together (see commit metadata below),
   # so a large run's history can be identified as one logical build.
   batch_job_id = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   logging.info(
      f"Split into {num_batches} batch(es) of up to {batch_size} granules "
      f"[batch_job_id={batch_job_id}]"
   )

   netcdf_store = S3Store(
      bucket="its-live-data",
      region="us-west-2",
      skip_signature=True,
   )

   for batch_num, batch_granules in enumerate(batches, start=1):
      logging.info(f'Batch {batch_num}/{num_batches}: loading {len(batch_granules)} granules')

      vds_list, missing_granules = load_granules(batch_granules, bucket)
      if missing_granules:
         logging.warning(
            f'Batch {batch_num}/{num_batches}: {len(missing_granules)} '
            'granules reported by searchAPI are missing from S3 (known '
            'catalog issue) -- skipping'
         )
      skipped_granules.extend(missing_granules)
      logging.info(f'Batch {batch_num}/{num_batches}: parsed {len(vds_list)} datasets')

      cube, autorift_param_file, batch_skipped_granules = \
         build_virtual_cube_subset(vds_list, bbox, netcdf_store, taken_time_values)

      # Record granules build_virtual_cube_subset skipped for this batch, so
      # the persistent skipped-granules JSON reflects every granule that was
      # considered and passed over, not just the ones that made it as far as
      # cropping.
      skipped_granules.extend(batch_skipped_granules)

      if cube is None:
         logging.info(f'Batch {batch_num}/{num_batches}: no valid data, nothing committed')
         save_skipped_granules(store_path, skipped_granules)
         continue

      # Ties every commit from this run together so they can be identified
      # as one logical build (see icechunk repo.ops_log()/ancestry()).
      commit_metadata = {
         "batch_job_id": batch_job_id,
         "batch_index": batch_num,
         "total_batches": num_batches,
         "batch_size": len(batch_granules),
      }

      # Clear granule-specific attrs on every batch (not just the repo-
      # creating one): combine_attrs re-derives cube.attrs fresh per batch,
      # so a later batch whose granules happen to agree on a value would
      # otherwise re-introduce it into the committed cube.
      cube.attrs.clear()

      if repo is None:
         # First batch with data: set up cube-level attributes, ice masks,
         # and create the icechunk repository.
         date_created = batch_job_id

         # Set all datacube attributes matching itscube.py
         cube.attrs[utils.OutputFormat.conventions] = \
            CubeFormat.values[utils.OutputFormat.conventions]
         cube.attrs[CubeFormat.date_created] = date_created
         cube.attrs[CubeFormat.gdal_area_or_point] = \
            CubeFormat.values[CubeFormat.gdal_area_or_point]
         cube.attrs[CubeFormat.geo_polygon] = json.dumps(polygon_coords)
         cube.attrs[utils.OutputFormat.institution] = \
            CubeFormat.values[utils.OutputFormat.institution]

         center_lon_lat = to_lon_lat_transformer.transform(xmid, ymid)
         cube.attrs[utils.OutputFormat.latitude] = round(center_lon_lat[1], 2)
         cube.attrs[utils.OutputFormat.longitude] = round(center_lon_lat[0], 2)

         cube.attrs[CubeFormat.proj_polygon] = json.dumps(polygon)
         cube.attrs[utils.OutputFormat.projection] = str(args.projection)

         # Time standard attributes from this (creation) batch's first granule
         if len(vds_list) > 0:
            first_vds = vds_list[0]
            if ImgPairInfo.name in first_vds.data_vars:
               img_pair_attrs = first_vds[ImgPairInfo.name].attrs
               for var_name in [ImgPairInfo.time_standard_img1, ImgPairInfo.time_standard_img2]:
                  if var_name in img_pair_attrs:
                     cube.attrs[var_name] = img_pair_attrs[var_name]

         cube.attrs[utils.OutputFormat.title] = \
            CubeFormat.values[utils.OutputFormat.title]
         cube.attrs[Vars.attrs.autorift_param_file] = autorift_param_file

         # Set attributes for 'url' data variable
         if Vars.url in cube.data_vars:
            cube[Vars.url].attrs[Vars.attrs.std_name] = Vars.url
            cube[Vars.url].attrs[Vars.attrs.description] = Vars.description[Vars.url]

         logging.info(f"\n{cube}")

         # Set S3 and URL attributes based on output location
         if is_s3_output:
            cube.attrs[utils.OutputFormat.s3] = store_path
            # Convert s3:// to https:// URL
            cube.attrs[utils.OutputFormat.url] = store_path.replace(
               bucket, bucketHTTP
            )

         else:
            cube.attrs[utils.OutputFormat.s3] = ''
            cube.attrs[utils.OutputFormat.url] = ''

         # Set skipped_granules attribute pointing to JSON file location
         skipped_json_path = skipped_granules_path(store_path)
         cube.attrs[SkippedGranules.name] = skipped_json_path

         config = ic.RepositoryConfig.default()

         if is_s3_output:
            # Configure storage settings for stronger recovery from transient
            # S3 failures. Only works with S3 storage -- local filesystem
            # storage doesn't set config.storage and will otherwise hit a
            # "put_opts with opts.attributes not yet implemented" error.
            config.storage = ic.StorageSettings(
               unsafe_use_metadata=True,           # Enable metadata stamping for write-id recovery
               unsafe_use_conditional_update=True  # Enable conditional PUTs to prevent conflicts
            )
         else:
            # Local filesystem storage: clear any stale directory left over
            # from a previous non-repo run (repo_exists was already checked
            # above and is False here, so there's nothing valid to preserve).
            shutil.rmtree(store_path, ignore_errors=True)

         config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(url_prefix, ic.s3_store(region="us-west-2", anonymous=True))
         )

         repo = ic.Repository.create(
            storage=_build_output_storage(store_path),
            config=config,
            authorize_virtual_chunk_access=ic.containers_credentials(
               {url_prefix: ic.s3_credentials(anonymous=True)}
            ),
         )

         # Add land/floating ice mask data variables, matching itscube.py's
         # combine_layers() (only added once, at cube creation -- the update
         # script never touches them again, since they have no 'time' dimension
         # and their already-committed chunks stay valid across every later
         # icechunk snapshot).
         shape_gdp = shapefile.read_file(args.shapeFile)

         # Set compession for encoding
         compressor = BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')
         # Set chunking for 2-d variables
         chunking_settings_2d = (len(cube.y), len(cube.x))
         logging.info(f'Icemasks using {chunking_settings_2d=}')

         for mask_name in [shapefile.LANDICE, shapefile.FLOATINGICE]:
            mask_data, mask_url = shapefile.read_ice_mask(
               shape_gdp, mask_name, cube.x.values, cube.y.values, args.projection
            )
            mask_data = utils.to_int_type(mask_data, np.uint8, utils.Missing.u8value)
            cube[mask_name] = xr.DataArray(
               data=mask_data,
               coords={utils.Coords.Y: cube.y.values, utils.Coords.X: cube.x.values},
               dims=[utils.Coords.Y, utils.Coords.X],
               attrs={
                  Vars.attrs.std_name: shapefile.Name[mask_name],
                  Vars.attrs.description: shapefile.Description[mask_name],
                  Mapping.attrs.grid_mapping: Mapping.name,
                  BinaryFlag.attrs.values: BinaryFlag.values,
                  BinaryFlag.attrs.meanings: BinaryFlag.meanings[mask_name],
                  utils.OutputFormat.url: mask_url
               }
            )
            cube[mask_name].encoding={
                  utils.OutputFormat.dtype: shapefile.Type[mask_name],
                  # icechunk repos are Zarr V3 stores, which use the plural
                  # 'compressors' encoding key (a list of codecs) rather
                  # than V2's singular 'compressor'.
                  utils.OutputFormat.compressors: [compressor],
                  utils.Missing.name: utils.Missing.u8value,
                  # The zarr-level sentinel, separate from any CF attribute,
                  # have it just in case
                  utils.Missing.fill_value: utils.Missing.u8value,
                  utils.OutputFormat.chunks: chunking_settings_2d
            }

         # Fix the 1D (time,) data variables' write chunk size to the fixed
         # TIME_CHUNK_VALUE_1D (imported from deep_copy_cube.py, see its
         # module comment), not whatever this first batch's size happens to
         # be -- leaves room to grow on later appends.
         set_1d_time_chunk_encoding(cube, TIME_CHUNK_VALUE_1D)

         # Fix 'time's CF units/dtype at creation too -- baked in like the
         # chunk size above. Left unset, xarray may infer 'days since
         # 1970-01-01' + int64, which can't hold mid_date's sub-day time-
         # of-day and forces a lossy fallback on a later append. int64
         # nanoseconds since epoch (Units.ns_epoch_date) is lossless --
         # see its docstring (previously float64, ~94% round-trip
         # reproducibility, see time_collisions.py).
         cube['time'].encoding[utils.Units.name] = utils.Units.ns_epoch_date
         cube['time'].encoding[utils.Units.calendar_name] = utils.Units.proleptic_gregorian
         cube['time'].encoding[utils.OutputFormat.dtype] = utils.Coords.DTYPE[utils.Coords.TIME]
         # Explicit None: xarray's CF encoder defaults a datetime
         # variable's _FillValue to NaN, which can't cast to int64 --
         # 'time' is never actually missing anyway.
         cube['time'].encoding[utils.OutputFormat.fill_value] = None

         session = repo.writable_session("main")
         cube_clean = _drop_nonfinite_attrs(cube)

         cube_clean.vz.to_icechunk(session.store)
         snapshot_id = session.commit(
            f"its_live virtual cube subset: create cube (batch {batch_num}/{num_batches}, "
            f"{len(cube.time)} granules)",
            metadata=commit_metadata
         )
         logging.info(f"Batch {batch_num}/{num_batches}: icechunk committed snapshot {snapshot_id=}")

      else:
         # Subsequent batches with data: append along time to the existing repo.
         session = repo.writable_session("main")
         cube_clean = _drop_nonfinite_attrs(cube)
         cube_clean.vz.to_icechunk(session.store, append_dim="time")
         snapshot_id = session.commit(
            f"its_live virtual cube subset: append batch {batch_num}/{num_batches} "
            f"({len(cube.time)} granules)",
            metadata=commit_metadata
         )
         logging.info(f"Batch {batch_num}/{num_batches}: icechunk committed snapshot {snapshot_id=}")

      # Save skipped granules to JSON file with _skippedGranules.json postfix
      # after every committed batch, so progress survives a mid-run failure
      # on a later batch.
      save_skipped_granules(store_path, skipped_granules)

   if repo is not None:
      if len(skipped_granules):
         logging.info(f'Skipped granules (first 10): \n{"\n".join(skipped_granules[:10])}')

      # zarr_format=3: icechunk repos are natively Zarr V3 metadata; forcing
      # zarr_format=2 here raises GroupNotFoundError (no .zgroup/.zarray
      # markers exist in an icechunk store).
      cube_roundtrip = xr.open_zarr(
         repo.readonly_session("main").store,
         consolidated=False,
         zarr_format=3,
         mask_and_scale=False
      )
      logging.info(f"{cube_roundtrip=}")

      # Belt and braces: every committed time must be unique, whatever path
      # (create, append, resumed run) produced it.
      if not cube_roundtrip.indexes[utils.Coords.TIME].is_unique:
         raise RuntimeError(
            f"{store_path} has duplicate '{utils.Coords.TIME}' values after "
            "this run -- the time-collision fix did not hold"
         )

   else:
      logging.info('No cube was created')

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')

   # Not Darwin-only: the same "Error in sys.excepthook" crash (traceback/sys
   # already partially torn down) has also been observed on Linux, on a
   # large multi-batch run. All commits/writes are already durable by this
   # point ('Done' logged above), so unconditionally shutting down the
   # executor and skipping Python's normal (racy) interpreter finalization
   # is safe on any platform.
   get_reusable_executor().shutdown(wait=True, kill_workers=True)
   time.sleep(0.5)  # let resource_tracker's unregister messages land
   os._exit(0)
