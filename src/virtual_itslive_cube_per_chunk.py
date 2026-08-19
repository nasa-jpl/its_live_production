"""
Build a virtual ITS_LIVE datacube restricted to a bounding box that is
*smaller* than the granules' combined extent. Such virtual datacube corresponds
to a single chunk of 512x512 pixels which are (chunk) co-aligned with all
ITS_LIVE granules.

After cropping, each granule is handed to the existing
virtual_itslive_cube.py:build_virtual_cube(), which mosaics the chunk-aligned
cropped grids onto a shared grid and stacks them on time. This reuses all of
the existing padding / combine_by_coords / img_pair_info-handling logic
unchanged.
"""
import boto3
from dateutil.parser import parse
from datetime import datetime
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

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.utils import copy_and_replace_metadata

from zarr.codecs import BloscCodec, BloscShuffle
from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.array_spec import ArraySpec, ArrayConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

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

# Suppress Zarr V3 unstable string dtype warnings
# These are informational - string specs are being finalized in Zarr V3.
# Zarr V3 has no stable spec for the fixed-length UTF32 dtype (<U2, <U3, etc.)
# Must filter by category (the message text is "... does not have a Zarr V3
# specification ...", which never contains the class name).
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Grid pixel size in meters
PIXEL_SIZE = 120
PIXEL_SIZE_HALF = PIXEL_SIZE / 2

# Number of threads for parallel processing
MAX_AWS_CONNECTIONS = 8

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
   """Get path to skipped granules JSON file for a given cube store.

   Parameters
   ----------
   cube_store : str
      Path to icechunk repository (S3 or local).

   Returns
   -------
   str
      Path to skipped granules JSON file.
   """
   return cube_store.rstrip('/').rstrip('.icechunk') + '_skippedGranules.json'


def save_skipped_granules(cube_store, skipped_granules):
   """Save skipped granules list to JSON file.

   Normalizes every URL to https:// form (and de-duplicates) before writing,
   so the file is consistently formatted regardless of whether the caller's
   in-memory set happened to hold s3://, https://, or a mix of both -- the
   same granule shouldn't appear twice under two different string forms.
   Shared between this script (fresh-cube creation) and
   virtual_itslive_cube_per_chunk_update.py (cube updates), so both write the
   skipped-granules JSON the same way.

   Parameters
   ----------
   cube_store : str
      Path to icechunk repository (S3 or local).
   skipped_granules : list of str
      List of skipped granule URLs, in s3:// form, https:// form, or a mix
      of both.
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
   """Chunk-aligned crop: return a new ManifestArray referencing only the
   chunks needed to cover element range [starts[i], stops[i]) on each axis.

   This is the mirror image of `pad_manifestarray` in virtual_itslive_cube.py:
   instead of placing a granule's ManifestArray into a larger grid (nodata
   elsewhere), we slice it down to only the chunks that cover the target bbox.
   Only chunk references are dropped/kept -- no pixel data is read.

   Parameters
   ----------
   marr : ManifestArray
      The array to crop. Only its chunk references are read/moved; no pixel
      data is accessed.
   starts : sequence of int
      Per-axis element indices where the crop region starts. Each value must
      be a multiple of that axis' chunk size.
   stops : sequence of int
      Per-axis element indices where the crop region ends (exclusive). Each
      value must be a multiple of that axis' chunk size, except it may equal
      the axis length to reach the array's edge (where the last chunk is
      legitimately partial).

   Returns
   -------
   ManifestArray
      A new ManifestArray with the same dtype and chunk structure as `marr`,
      but containing only the chunk references needed to cover the specified
      element ranges. The returned array's shape reflects the cropped region.

   Raises
   ------
   ValueError
      If starts/stops have wrong dimensionality, contain invalid ranges, or
      are not aligned to chunk boundaries.
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
   """Compute the encoded length of an all-fill chunk.

   This reference length allows short-circuiting: if a chunk's encoded length
   differs from allfill_chunk_len, it CANNOT be all-fill (must contain valid data),
   so we can skip the expensive S3 read + decode.

   Parameters
   ----------
   chunk_shape : tuple
      Shape of the chunk.
   np_dtype : numpy.dtype
      Numpy data type used to build the fill array (from `ManifestArray.dtype`).
   zarr_dtype : zarr dtype
      Zarr data type for the encode `ArraySpec` (from `metadata.dtype`).
   fill_value : scalar
      Fill value for the array.
   codecs : list
      Zarr codec pipeline.

   Returns
   -------
   int
      The encoded byte length of an all-fill chunk.
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
   """Return True if a cropped ManifestArray references any non-fill data.

   Reads ONLY the chunk bytes the manifest references (a single 512x512 chunk
   for a chunk-aligned tile) via S3 byte-range GETs and decodes them through the
   array's own zarr codec pipeline -- instead of downloading the entire granule
   NetCDF just to inspect one window. It also reuses the virtual dataset already
   opened in load_granules, so the granule file is never opened a second time.

   A chunk counts as "valid" if any element differs from the fill value. This
   matches the previous test `np.isnan(cf_decoded_v).all()`: v is stored as
   int16 whose only NaN-producing value on CF-decode is `_FillValue`, so
   "all fill" is exactly "all NaN after decode". Missing chunks (empty path)
   read back as fill and are skipped.

   Performance optimization: before reading a chunk from S3, checks if its
   encoded length differs from the reference all-fill chunk length. If so,
   the chunk MUST contain valid data (short-circuit to True without decode).

   Parameters
   ----------
   marr : ManifestArray
      A ManifestArray already cropped to the target window (e.g. v cropped via
      crop_manifestarray).
   netcdf_store : obstore.store.S3Store
      Object store used to fetch the referenced chunk byte ranges.
   allfill_chunk_len : int
      Pre-computed encoded byte length of an all-fill chunk. Passed down from
      build_virtual_cube_subset() to avoid recomputing for every granule.

   Returns
   -------
   bool
      True if any referenced chunk contains a non-fill value.
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

      # Length-based short-circuit: if encoded length differs from all-fill
      # reference, this chunk MUST contain valid data (no S3 read needed).
      # This assumes 'v' uses the same chunk size, dtype, fill value, and
      # codec configuration across ALL granules feeding this cube (see the
      # comment at allfill_chunk_len's computation in
      # build_virtual_cube_subset). If that ever doesn't hold -- e.g. a
      # differently-encoded granule from a mission/processing-version not
      # covered by that assumption -- a genuinely all-fill chunk in it could
      # encode to a different byte length than this reference, and this
      # short-circuit would incorrectly return True (treat it as having
      # valid data) instead of falling through to the decode-and-compare
      # check below. That's a safe-direction failure (an extra near-empty
      # layer kept, not real data silently dropped), but it means this
      # optimization is only as safe as that cross-granule uniformity
      # assumption.
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
   """Map a target coordinate range [bbox_lo, bbox_hi] onto chunk-grid-aligned
   element indices [start, stop) into `coord`.

   Parameters
   ----------
   coord : numpy.ndarray
      A regularly-spaced 1-D coordinate vector (e.g., x or y coordinates).
   step : float
      Spacing between coordinate values. May be negative for descending axes
      (e.g., y-axis).
   chunk_size : int
      Size of chunks along this dimension in elements.
   bbox_lo : float
      Lower bound of the target coordinate range.
   bbox_hi : float
      Upper bound of the target coordinate range.

   Returns
   -------
   tuple of (int, int) or None
      A tuple (start, stop) of chunk-aligned element indices into `coord`,
      where `stop` is exclusive. Returns None if the bbox doesn't overlap
      `coord` at all.
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
   """Crop one virtual dataset (real x/y coords, virtual ManifestArray data
   vars) to the chunk-grid-aligned window covering `bbox`.

   Parameters
   ----------
   vds : xr.Dataset
      A single granule's virtual dataset, as returned by
      `open_virtual_dataset` (x/y/time loaded, data vars virtual).
   bbox : (xmin, xmax, ymin, ymax)
      Target region in the dataset's native x/y units. These are adjusted
      for the cell centers based on the datacube bounding polygon which is
      for the cell corners.
   netcdf_store: obstore.store.S3Store
      Object store to access granule data from.
   allfill_chunk_len : int
      Pre-computed encoded byte length of an all-fill chunk for the 'v'
      variable. Used to short-circuit the valid-data check when a chunk's
      length differs from this reference.

   Returns
   -------
   tuple of (xr.Dataset or None, str)
      A tuple containing:
      - The cropped dataset with chunk-aligned spatial bounds, or None if
      this granule doesn't overlap `bbox` or has no valid data in the
      overlap region.
      - The granule URL string from the dataset's attributes.
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
   """Verify every cropped granule landed on exactly the same x/y grid.

   Granules are guaranteed to share a common chunk grid (same posting,
   same chunk boundaries), so chunk-aligned cropping to the same `bbox`
   should produce *identical* x/y coordinate arrays across granules -- not
   merely overlapping ones needing a pad-to-common-grid step. If that's
   ever violated (e.g. an unexpectedly offset granule), fail loudly here
   rather than silently letting `pad_manifestarray` paper over it.

   Parameters
   ----------
   cropped : list of xr.Dataset
      List of cropped virtual datasets. Must have at least one element.
   bbox : tuple
      The bounding box (xmin, xmax, ymin, ymax) used for cropping, included
      in error messages for debugging.

   Raises
   ------
   ValueError
      If any cropped granule has different x or y coordinates than the first
      granule in the list.

   Returns
   -------
   None
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


def build_virtual_cube_subset(vds_list, bbox, netcdf_store):
   """Build a virtual datacube restricted to `bbox`, smaller than the
   granules' combined extent.

   Each granule's ManifestArrays are first cropped (chunk-aligned) to the
   window overlapping `bbox`; granules with no overlap are dropped entirely.
   Since granules are guaranteed to share a common chunk grid, the cropped
   granules are then checked to have landed on an identical x/y grid (a
   hard failure if not) before being mosaicked via `build_virtual_cube`
   -- which stacks them on time and reuses its existing dtype/attr/
   img_pair_info handling.

   Parameters
   ----------
   vds_list : list of xr.Dataset
      Virtual datasets for each granule (as passed to `build_virtual_cube`).
      Each dataset should have 'x', 'y', 'time' coordinates and virtual
      ManifestArray data variables.
   bbox : tuple of (float, float, float, float)
      Target region in the granules' shared x/y units, specified as
      (xmin, xmax, ymin, ymax).
   netcdf_store : obstore.store.S3Store
      S3 object store instance for accessing granule data to check for
      valid data within the overlap region.

   Returns
   -------
   tuple of (xr.Dataset or None, str or None, list of str)
      A tuple containing:
      - The virtual datacube with all cropped granules stacked along the
      time dimension, or None if no granule had valid data in the bbox.
      - The autorift parameter file path, or None if no cube was built.
      - List of URLs for granules that were skipped (no overlap or no valid
      data in overlap region).

   Raises
   ------
   ValueError
      If cropped granules don't share identical x/y grids.
   """
   if not vds_list:
      logging.info("vds_list is empty -- no granules to build a cube from; no cube built")
      return None, None, []

   logging.info(f'Building cube out of {len(vds_list)} granules')
   cropped = []
   skipped_granules = []

   # Compute reference all-fill chunk length once for all granules.
   # All ITS_LIVE granules share the same chunk size (512×512), dtype (int16),
   # fill value (-32767), and codec configuration for the 'v' variable, so
   # allfill_chunk_len is identical across all granules. Computing it once here avoids
   # repeating the encode operation (which requires creating and encoding a
   # full 512×512 array) for every granule.
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

   # Threads ("threading"), not processes: cropping is S3-I/O-bound work whose
   # per-granule path is now pure obstore range reads + numpy manifest slicing +
   # zarr codec decode (no h5py) after the manifest-based valid-data check, so
   # there is no thread-unsafe library on this path and no GIL-bound compute to
   # contend on. Threads also avoid pickling each granule's ManifestArray-bearing
   # dataset to a worker process and the cropped dataset back again.
   #
   # Hand joblib the whole list (no manual batching) so the n_jobs-sized pool
   # stays continuously saturated -- a manual batch barrier would make every
   # batch wait on its slowest granule. return_as="generator_unordered" yields
   # each result as soon as it completes (as-completed), so progress advances
   # smoothly even under skewed granule durations; the ordered generator would
   # stall behind a slow early granule (head-of-line blocking). Arrival order is
   # irrelevant: build_virtual_cube stacks layers by the time coordinate.
   with parallel_config(
      backend='threading',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      # This returns a lazy generator immediately -- no cropping has run yet.
      # The tasks execute as the loop below pulls from result_stream, each
      # result yielded the moment its task finishes.
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

   logging.info(f'Number of skipped granules: {len(skipped_granules)}')
   # All cropped granules are on identical grids (verified by _assert_identical_grids),
   # so skip the extend_coords() step in build_virtual_cube
   return *build_virtual_cube(cropped, already_aligned=True), skipped_granules


def read_virtual_dataset(granule_url, parser, registry):
   """Read granule into virtual dataset.

   Raises immediately on failure (see `load_granules`): a granule that can't
   be opened signals a problem worth stopping the run for (bad input list,
   broken S3 access, etc.) rather than one to silently paper over.

   Parameters
   ----------
   granule_url : str
      S3 URL to the granule file (e.g., 's3://its-live-data/path/to/granule.nc').
   parser : virtualizarr.parsers.HDFParser
      Parser instance for reading HDF/NetCDF files as virtual datasets.
   registry : obspec_utils.registry.ObjectStoreRegistry
      Registry mapping URL prefixes to object store instances for chunk access.

   Returns
   -------
   xr.Dataset
      Virtual dataset with 'time', 'y', 'x' coordinates loaded into memory
      and data variables as ManifestArrays (chunk references only). The dataset
      includes 'granule_url' and 'granule_path' in its attributes.

   Raises
   ------
   RuntimeError
      If the granule cannot be opened as a virtual dataset.
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
   """Load granules into virtual datasets using parallel processing.

   Reads multiple granule files in parallel, converting each into a virtual
   dataset with coordinate data loaded and data variables as ManifestArrays
   (chunk references only, no pixel data loaded).

   Parameters
   ----------
   granules : list of str
      List of S3 URLs or paths to granule files to load.
   bucket : str
      AWS S3 bucket URL (e.g., 's3://its-live-data') that stores the granules.

   Returns
   -------
   list of xr.Dataset
      List of virtual datasets, one per granule. Order is not guaranteed to
      match the input (results are collected as-completed); downstream stacking
      keys off the time coordinate, not list position. Each dataset has 'time',
      'y', 'x' coordinates loaded and data variables as ManifestArrays.

   Raises
   ------
   RuntimeError
      Propagated from `read_virtual_dataset` on the first granule that fails
      to open -- aborts the whole run rather than skipping it.
   """
   store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
   registry = ObjectStoreRegistry({bucket: store})
   # Keep 'mapping' (don't drop it): it's loaded as a small 0-dim variable in
   # read_virtual_dataset so build_virtual_cube can recover its projection attrs
   # to synthesize the cube's CF grid-mapping variable.
   parser = HDFParser()

   vds_list = []

   # Processes ("loky"), not threads: opening a granule as a virtual dataset
   # goes through HDF parsing whose thread-safety is not guaranteed, so keep it
   # in separate worker processes. (The crop pass in build_virtual_cube_subset
   # is h5py-free and uses threads instead.)
   #
   # Hand joblib the whole list (no manual batching) so the n_jobs-sized pool
   # stays continuously saturated; return_as="generator_unordered" yields each
   # result as soon as it completes (as-completed progress, no head-of-line
   # stall behind a slow granule) and throttles dispatch to pre_dispatch
   # (~2*n_jobs), so a million granules don't all get queued at once. Arrival
   # order does not matter: downstream cropping/stacking keys off the time
   # coordinate, not list position.
   total = len(granules)
   with parallel_config(
      backend='loky',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      # This returns a lazy generator immediately -- no granule is opened yet.
      # The tasks execute as the loop below pulls from result_stream, each
      # result yielded the moment its task finishes.
      result_stream = Parallel(return_as="generator_unordered")(
         delayed(read_virtual_dataset)(each_file, parser, registry)
         for each_file in granules
      )

      for done, each_ds in enumerate(result_stream, start=1):
         vds_list.append(each_ds)

         if done % PROGRESS_LOG_INTERVAL == 0 or done == total:
            logging.info(f"Loaded {done}/{total} granules")

   return vds_list


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
      default=100000,
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
      '-s', '--shapeFile',
      type=str,
      default='s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp',
      help='Shapefile with ice masks information [%(default)s]'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   MAX_AWS_CONNECTIONS = args.threads

   if sys.platform == 'darwin':
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
         )

      roi = {
         "type": "Polygon",
         "coordinates": [polygon_coords]
      }

      granules = itslive_utils.serverless_search(
         epsg_code=args.projection,
         start_date=args.start_date,
         end_date=args.end_date,
         roi=roi
      )

   # If testing and want to process only a subset of granules. Truncate
   # *before* filtering P000 granules below, so --num-granules reflects a
   # slice of the raw candidate list (matching what a user would expect when
   # asking for "the first N granules"), and the "Processing" count logged
   # after P000 filtering can come out lower than N if any of that slice
   # turned out to be P000.
   if args.num_granules > 0:
      num_granules = args.num_granules
      granules = granules[:num_granules]

   # P000 granules never have usable data; exclude them from processing but
   # still record them in the skipped-granules JSON below (merged into
   # skipped_granules after build_virtual_cube_subset), so the persistent
   # record reflects every granule that was considered and passed over.
   # P000 granules have original "https://"" url
   p000_granules = [each for each in granules if each.endswith(P000_SUFFIX)]

   # The rest of the granules have "s3://"" url
   granules = [
      each.replace(HTTPS_URL, S3_URL) for each in granules \
      if each.endswith(P000_SUFFIX) is False
   ]

   if p000_granules:
      logging.info(f"Excluding {len(p000_granules)} P000 granules")

   # Sort chronologically by mid_date parsed from each filename -- no granule
   # is opened for this, so it's cheap even for very large granule counts.
   # This both gives the cube's layers a sensible time order and lets the
   # batches below be processed (and appended to the cube) in time order.
   granules = sorted(granules, key=utils.extract_mid_date_from_url)

   logging.info(f"Processing {len(granules)} granules")

   bucket = args.bucket
   bucketHTTP = args.bucketHTTP

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

   # "s3://its-live-data/"
   url_prefix = bucket + os.sep
   store_path = args.output_store

   # Determine if output is S3 or local filesystem
   is_s3_output = store_path.startswith(utils.S3_PREFIX)

   # Accumulates skipped granules across all batches; P000 granules were
   # never handed to any batch, so seed with those up front.
   skipped_granules = list(p000_granules)

   # None until the first batch that actually produces a cube creates the
   # icechunk repo; every later batch with data appends to it.
   repo = None

   for batch_num, batch_granules in enumerate(batches, start=1):
      logging.info(f'Batch {batch_num}/{num_batches}: loading {len(batch_granules)} granules')

      vds_list = load_granules(batch_granules, bucket)
      logging.info(f'Batch {batch_num}/{num_batches}: parsed {len(vds_list)} datasets')

      cube, autorift_param_file, batch_skipped_granules = \
         build_virtual_cube_subset(vds_list, bbox, netcdf_store)

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

      if repo is None:
         # First batch with data: set up cube-level attributes, ice masks,
         # and create the icechunk repository.
         date_created = batch_job_id

         # Set all datacube attributes matching itscube.py
         cube.attrs[utils.OutputFormat.conventions] = \
            CubeFormat.values[utils.OutputFormat.conventions]
         cube.attrs[CubeFormat.datacube_software_version] = '1.0'
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

         # Remove granule specific attributes (may already be absent if
         # combine_attrs="drop_conflicts" dropped it due to differing values
         # across granules)
         cube.attrs.pop('motion_detection_method', None)

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

         if is_s3_output:
            # S3 storage - parse bucket and prefix. Named out_bucket (not
            # `bucket`) so it doesn't shadow the granule-source bucket that
            # later batches' load_granules() calls still need.
            s3_parts = store_path.replace(utils.S3_PREFIX, '').split('/', 1)
            out_bucket = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ''

            logging.info(f'Writing icechunk repo to S3: bucket={out_bucket}, prefix={prefix}')

            # Configure storage settings for stronger recovery from transient S3 failures
            storage_settings = ic.StorageSettings(
               unsafe_use_metadata=True,           # Enable metadata stamping for write-id recovery
               unsafe_use_conditional_update=True  # Enable conditional PUTs to prevent conflicts
            )

            config = ic.RepositoryConfig.default()
            config.storage = storage_settings
            config.set_virtual_chunk_container(
               ic.VirtualChunkContainer(url_prefix, ic.s3_store(region="us-west-2", anonymous=True))
            )

            # Create S3 storage for repository (authenticated write access)
            storage = ic.s3_storage(
               bucket=out_bucket,
               prefix=prefix,
               region="us-west-2"
            )

            repo = ic.Repository.create(
               storage=storage,
               config=config,
               authorize_virtual_chunk_access=ic.containers_credentials(
                  {url_prefix: ic.s3_credentials(anonymous=True)}
               ),
            )
         else:
            # Local filesystem storage
            shutil.rmtree(store_path, ignore_errors=True)

            # Note: Don't enable unsafe_use_metadata for local filesystem - it only
            # works with S3 storage and will cause "put_opts with opts.attributes
            # not yet implemented" error on local filesystem.
            config = ic.RepositoryConfig.default()
            config.set_virtual_chunk_container(
               ic.VirtualChunkContainer(
                  url_prefix,
                  ic.s3_store(region="us-west-2", anonymous=True)
               )
            )
            repo = ic.Repository.create(
               storage=ic.local_filesystem_storage(store_path),
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
                  utils.OutputFormat.compressor: compressor,
                  utils.Missing.name: utils.Missing.u8value,
                  # The zarr-level sentinel, separate from any CF attribute,
                  # have it just in case
                  utils.Missing.fill_value: utils.Missing.u8value,
                  utils.OutputFormat.chunks: chunking_settings_2d
            }

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

      cube_roundtrip = xr.open_zarr(
         repo.readonly_session("main").store,
         consolidated=False,
         zarr_format=3,
         mask_and_scale=False
      )
      logging.info(f"{cube_roundtrip=}")

   else:
      logging.info('No cube was created')

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')

   if sys.platform == 'darwin':
      get_reusable_executor().shutdown(wait=True, kill_workers=True)
      time.sleep(0.5)  # let resource_tracker's unregister messages land

      # macOS-only: works around GDAL/PROJ/loky atexit ordering on Darwin
      # bypass Python's normal interpreter finalization
      os._exit(0)
