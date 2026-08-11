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
from dateutil.parser import parse
from datetime import datetime
from joblib import Parallel, delayed, parallel_config
import gc
import logging
import numpy as np
import pyproj
import xarray as xr
import os
import json
import shutil
import obstore
from obstore.store import S3Store
import virtualizarr as vz
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
import icechunk as ic

from virtual_itslive_cube import _drop_nonfinite_attrs

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.utils import copy_and_replace_metadata

from zarr.core.codec_pipeline import BatchedCodecPipeline
from zarr.core.array_spec import ArraySpec, ArrayConfig
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync

from virtual_itslive_cube import build_virtual_cube
import itslive_utils
import utils
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

# Grid pixel size in meters
PIXEL_SIZE = 120
PIXEL_SIZE_HALF = PIXEL_SIZE / 2

# Number of threads for parallel processing
MAX_AWS_CONNECTIONS = 8
NUM_GRANULES_TO_READ = 100

# String representation of longitude/latitude projection
LON_LAT_PROJECTION = 'EPSG:4326'


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
   shape, chunks = marr.shape, marr.chunks
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


def _cropped_var_has_valid_data(marr, netcdf_store):
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

   Parameters
   ----------
   marr : ManifestArray
      A ManifestArray already cropped to the target window (e.g. v cropped via
      crop_manifestarray).
   netcdf_store : obstore.store.S3Store
      Object store used to fetch the referenced chunk byte ranges.

   Returns
   -------
   bool
      True if any referenced chunk contains a non-fill value.
   """
   metadata = marr.metadata
   fill = metadata.fill_value
   prototype = default_buffer_prototype()
   pipeline = BatchedCodecPipeline.from_codecs(metadata.codecs)

   # Per-chunk decode spec: shape is the chunk shape (not the window); dtype,
   # fill value and codecs come from the array metadata.
   spec = ArraySpec(
      shape=marr.chunks,
      dtype=metadata.dtype,
      fill_value=fill,
      config=ArrayConfig.from_dict({}),
      prototype=prototype,
   )

   manifest = marr.manifest
   for path, offset, length in zip(
      manifest._paths.flat, manifest._offsets.flat, manifest._lengths.flat
   ):
      if not path:
         # Missing chunk -> reads back as fill, no valid data contributed
         continue

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


def crop_virtual_dataset_to_bbox(vds, bbox, netcdf_store):
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
         if x_chunk is None and "x" in dims:
            x_chunk = data.chunks[dims.index("x")]
         if y_chunk is None and "y" in dims:
            y_chunk = data.chunks[dims.index("y")]

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
      logging.info(f'{vds.attrs["granule_url"]} does not overlap the polygon')
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

   if not _cropped_var_has_valid_data(cropped_v, netcdf_store):
      # Granule does not have any valid data within intersection
      logging.info(f'{vds.attrs["granule_url"]} does not have valid data within polygon')
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
   tuple of (xr.Dataset, list of str)
      A tuple containing:
      - The virtual datacube with all cropped granules stacked along the
        time dimension.
      - List of URLs for granules that were skipped (no overlap or no valid
        data in overlap region).

   Raises
   ------
   ValueError
      If no granules overlap the bbox, or if cropped granules don't share
      identical x/y grids.
   """
   logging.info(f'Building cube out of {len(vds_list)} granules')
   cropped = []
   skipped_granules = []

   start = 0
   num_to_process = len(vds_list)

   # Threads ("threading"), not processes: cropping is S3-I/O-bound work whose
   # per-granule path is now pure obstore range reads + numpy manifest slicing +
   # zarr codec decode (no h5py) after the manifest-based valid-data check, so
   # there is no thread-unsafe library on this path and no GIL-bound compute to
   # contend on. Threads also avoid pickling each granule's ManifestArray-bearing
   # dataset to a worker process and the cropped dataset back again.
   with parallel_config(
      backend='threading',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      while num_to_process > 0:
         # How many tasks to process at a time
         num_tasks = min(num_to_process, NUM_GRANULES_TO_READ)

         # Run in parallel with joblib
         log_msg = f"Building virtual cube: processing {num_tasks} tasks out of " \
                     f"{num_to_process} remaining"
         logging.info(log_msg)

         results = Parallel()(
            delayed(crop_virtual_dataset_to_bbox)(each_vds, bbox, netcdf_store) for
            each_vds in vds_list[start:start + num_tasks]
         )

         for each_result in results:
            cropped_ds, cropped_url = each_result
            if cropped_ds is not None:
               cropped.append(cropped_ds)

            else:
               skipped_granules.append(cropped_url)

         num_to_process -= num_tasks
         start += num_tasks

   if not cropped:
      raise ValueError(f"Building virtual cube: no granules overlap bbox {bbox}")

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
   """
   v = None

   try:
      v = vz.open_virtual_dataset(
         url=granule_url,
         parser=parser,
         registry=registry,
         loadable_variables=["time", "y", "x"],
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

   Reads multiple granule files in parallel batches, converting each into
   a virtual dataset with coordinate data loaded and data variables as
   ManifestArrays (chunk references only, no pixel data loaded).

   Parameters
   ----------
   granules : list of str
      List of S3 URLs or paths to granule files to load.
   bucket : str
      AWS S3 bucket URL (e.g., 's3://its-live-data') that stores the granules.

   Returns
   -------
   list of xr.Dataset
      List of virtual datasets, one per granule, with the same order as the
      input granules list. Each dataset has 'time', 'y', 'x' coordinates loaded
      and data variables as ManifestArrays.
   """
   store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
   registry = ObjectStoreRegistry({bucket: store})
   parser = HDFParser(drop_variables=[Mapping.name])

   vds_list = []
   start = 0
   num_to_process = len(granules)

   # Use processes ("loky") instead of threads ("threading") for parallel
   # processing - each process is getting their own copy of the object instance
   # (registry, object store, etc.) that are passed to each of the processes,
   # Using loky (process-based) bypasses the threading-lock contention entirely,
   # by construction, because there is no shared process for a lock to live in.
   with parallel_config(
      backend='loky',
      n_jobs=MAX_AWS_CONNECTIONS
   ):
      while num_to_process > 0:
         # How many tasks to process at a time
         num_tasks = min(num_to_process, NUM_GRANULES_TO_READ)

         # Run in parallel with joblib
         log_msg = f"Processing {num_tasks} tasks out of " \
                     f"{num_to_process} remaining"
         logging.info(log_msg)

         # with tqdm_joblib(tqdm(desc=log_msg, total=num_tasks)):
         results = Parallel()(
            delayed(read_virtual_dataset)(each_file, parser, registry) for
            each_file in granules[start:start + num_tasks]
         )

         for each_ds in results:
            vds_list.append(each_ds)

         num_to_process -= num_tasks
         start += num_tasks

   return vds_list


if __name__ == "__main__":
   import argparse

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
      "--granules",
      nargs="+",
      help="List of granule paths (space-separated)"
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

   args = parser.parse_args()

   MAX_AWS_CONNECTIONS = args.threads

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

   # Adjust cube cell edge coordinates to the cell centers (like granules grids
   # have it)
   xmin = xmin + PIXEL_SIZE_HALF
   xmax = xmax - PIXEL_SIZE_HALF
   ymin = ymin + PIXEL_SIZE_HALF
   ymax = ymax - PIXEL_SIZE_HALF

   bbox = [xmin, xmax, ymin, ymax]

   xmid = (xmin + xmax) / 2
   ymid = (ymin + ymax) / 2

   # Convert UTM coordinates to lon/lat (ensure lonlat output order)
   to_lon_lat_transformer = pyproj.Transformer.from_crs(
      f"EPSG:{args.projection}", LON_LAT_PROJECTION, always_xy=True
   )

   # Introduce 5 points per each polygon side
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
   elif args.granules:
      granules = args.granules

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

   granules = [
      each.replace(
         'https://its-live-data.s3.amazonaws.com/',
         's3://its-live-data/') for each in granules
   ]

   granules = [each for each in granules if not each.endswith('P000.nc')]

   # If testing and want to process only a subset of granules
   if args.num_granules > 0:
      num_granules = args.num_granules
      granules = granules[:num_granules]

   logging.info(f"Processing {len(granules)} granules")

   bucket = args.bucket

   vds_list = load_granules(granules, bucket)
   logging.info(f'Parsed {len(vds_list)} datasets')

   # Look up a cube that overlapps both granules in latest catalog:
   # 1. Run tools/tests/verify_chunk_alignment_granules_datacubes.py
   #     - identifies overlapping datacubes per catalog, pick UTM polygon
   #       coordinates

   # 2. Adjust cube cell edge coordinates for the cells centers:
   # bbox = (xmin, xmax, ymin, ymax)
   # bbox for original 2 Landsat granules
   # bbox = (
   #    -61447.5 + PIXEL_SIZE_HALF, -7.5 - PIXEL_SIZE_HALF,
   #    -983032.5 + PIXEL_SIZE_HALF, -921592.5 - PIXEL_SIZE_HALF)

   # bbox for test case with S1, S2, Landsat granules
   # bbox = (
   #    -1658887.5 + PIXEL_SIZE_HALF, -1597447.5 - PIXEL_SIZE_HALF,
   #    -430072.5 + PIXEL_SIZE_HALF, -368632.5 - PIXEL_SIZE_HALF)

   # Store to load each of the granule's "v" values to check if granule has
   # any valid "v" data in the cube polygon
   netcdf_store = S3Store(
      bucket="its-live-data",
      region="us-west-2",
      skip_signature=True,
   )

   cube, autorift_param_file, skipped_granules = \
      build_virtual_cube_subset(vds_list, bbox, netcdf_store)

   # Add new attributes to the cube
   if cube:
      date_created = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
      cube.attrs[CubeFormat.date_created] = date_created
      cube.attrs[CubeFormat.date_updated] = date_created

      cube.attrs[utils.OutputFormat.title] = \
         CubeFormat.values[utils.OutputFormat.title]

      cube.attrs[CubeFormat.gdal_area_or_point] = \
         CubeFormat.values[CubeFormat.gdal_area_or_point]

      cube.attrs[CubeFormat.geo_polygon] = json.dumps(polygon_coords)

      cube.attrs[utils.OutputFormat.institution] = \
         CubeFormat.values[utils.OutputFormat.institution]

      cube.attrs[Vars.attrs.autorift_param_file] = autorift_param_file

      center_lon_lat = to_lon_lat_transformer.transform(xmid, ymid)
      cube.attrs[utils.OutputFormat.latitude] = round(center_lon_lat[1], 2)
      cube.attrs[utils.OutputFormat.longitude] = round(center_lon_lat[0], 2)

      cube.attrs[CubeFormat.proj_polygon] = json.dumps(polygon)
      cube.attrs[utils.OutputFormat.projection] = str(args.projection)

      # Remove granule specific attributes
      del cube.attrs['motion_detection_method']

      print(f"\n{cube}")

      format_skipped_granules = "\n".join(skipped_granules)
      logging.info(f'Skipped first 10 granules: \n{format_skipped_granules[:10]}')

      url_prefix = "s3://its-live-data/"
      store_path = args.output_store
      shutil.rmtree(store_path, ignore_errors=True)

      config = ic.RepositoryConfig.default()
      config.set_virtual_chunk_container(
         ic.VirtualChunkContainer(url_prefix, ic.s3_store(region="us-west-2", anonymous=True))
      )
      repo = ic.Repository.create(
         storage=ic.local_filesystem_storage(store_path),
         config=config,
         authorize_virtual_chunk_access=ic.containers_credentials(
            {url_prefix: ic.s3_credentials(anonymous=True)}
         ),
      )

      session = repo.writable_session("main")
      cube_clean = _drop_nonfinite_attrs(cube)
      cube_clean.vz.to_icechunk(session.store)
      snapshot_id = session.commit("its_live virtual cube subset: create cube")
      logging.info(f"icechunk committed snapshot: {snapshot_id=}")

      cube_roundtrip = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
      logging.info(f"{cube_roundtrip=}")

      logging.info(f'{cube_roundtrip.mission_img1.values=}')
      logging.info(f'{cube_roundtrip.mission_img2.values=}')
      logging.info(f'{cube_roundtrip.satellite_img1.values=}')
      logging.info(f'{cube_roundtrip.satellite_img2.values=}')
      logging.info(f'{cube_roundtrip.time.values=}')

   else:
      logging.info('No cube was created')

   logging.info('Done')

