"""
Build a virtual ITS_LIVE datacube restricted to a bounding box that is
*smaller* than the granules' combined extent. Such virtual datacube corresponds
to a single chunk of 522x512 pixels which are co-aligned with all ITS_LIVE
granules.

This is the mirror image of `pad_manifestarray` in virtual_itslive_cube.py:
instead of placing a granule's ManifestArray into a larger grid (nodata
elsewhere), we slice it down to only the chunks that cover the target bbox.
Only chunk references are dropped/kept -- no pixel data is read.

After cropping, each granule is handed to the existing `build_virtual_cube`,
which mosaics the chunk-aligned cropped grids onto a shared grid and stacks
them on time. This reuses all of ehe existing padding / combine_by_coords /
img_pair_info-handling logic unchanged.
"""
from joblib import Parallel, delayed, parallel_config
import gc
import logging
import numpy as np
import xarray as xr
from obstore.store import S3Store
import os
import io

from virtualizarr.manifests import ManifestArray
from virtualizarr.manifests.utils import copy_and_replace_metadata

from virtual_itslive_cube import build_virtual_cube

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


def crop_manifestarray(marr, starts, stops):
   """Chunk-aligned crop: return a new ManifestArray referencing only the
   chunks needed to cover element range [starts[i], stops[i]) on each axis.

   Parameters
   ----------
   marr : ManifestArray
      The array to crop. Only its chunk references are read/moved.
   starts, stops : sequence of int
      Per-axis element bounds of the region to keep, `stops` exclusive.
      Both must be multiples of that axis' chunk size, except `stops[i]`
      may equal the axis length to reach the array's edge (where the last
      chunk is legitimately partial).
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


def open_netcdf_from_s3(store, key, engine='h5netcdf'):
   if 'https://' in key:
      key = key.replace('https://its-live-data.s3.amazonaws.com/', '')

   result = obstore.get(store, key)
   buf = io.BytesIO(result.bytes())
   return xr.open_dataset(buf, engine=engine)


def bbox_to_chunk_aligned_indices(coord, step, chunk_size, bbox_lo, bbox_hi):
   """Map a target coordinate range [bbox_lo, bbox_hi] onto chunk-grid-aligned
   element indices [start, stop) into `coord`.

   `coord` is a real, regularly-spaced 1-D coordinate vector with spacing
   `step` (may be negative -- e.g. a descending y-axis). Returns None if
   `[bbox_lo, bbox_hi]` doesn't overlap `coord` at all.
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
   xr.Dataset or None
      The cropped dataset, or None if this granule doesn't overlap `bbox`
      at all (caller should drop it from the cube).
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

   # Check if overlapping region contain any valid data
   with open_netcdf_from_s3(netcdf_store, vds.attrs['granule_path']) as granule_ds:
      # Need to open the granule by loading "v" data - ManifestArray has only
      # chunk references, no actual data so can't mask arrays
      mask_lon = (granule_ds.x >= xmin) & (granule_ds.x <= xmax)
      mask_lat = (granule_ds.y >= ymin) & (granule_ds.y <= ymax)
      mask = (mask_lon & mask_lat)

      mask_data = granule_ds.v.isel(x=slice(*x_range), y=slice(*y_range)).isel(time=0).load()

      if np.isnan(mask_data).all():
         # Granule does not have any valid data within intersection
         logging.info(f'{vds.attrs["granule_url"]} does not have valid data within polygon')
         return None, vds.attrs["granule_url"]

   logging.debug(f'Updating to {x_range=}')
   logging.debug(f'Updating to {y_range=}')

   x_start, x_stop = x_range
   y_start, y_stop = y_range
   start_by_dim = {"x": x_start, "y": y_start}
   stop_by_dim = {"x": x_stop, "y": y_stop}

   new_vars = {}
   for name, var in vds.data_vars.items():
      data = var.data

      if isinstance(data, ManifestArray):
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
   bbox : (xmin, xmax, ymin, ymax)
      Target region in the granules' shared x/y units.
   """
   logging.info(f'Building cube out of {len(vds_list)} granules')
   cropped = []
   skipped_granules = []

   start = 0
   num_to_process = len(vds_list)

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
         log_msg = f"Building virtual cube: processing {num_tasks} tasks out of " \
                     f"{num_to_process} remaining"
         logging.info(log_msg)

         # with tqdm_joblib(tqdm(desc=log_msg, total=num_tasks)):
         results = Parallel()(
            delayed(crop_virtual_dataset_to_bbox)(each_vds, bbox, netcdf_store) for
            each_vds in vds_list[start:start + num_tasks]
         )

         for each_result in results:
            cropped_ds, ds_url = each_result
            if cropped_ds is not None:
               cropped.append(cropped_ds)

            else:
               skipped_granules.append(ds_url)

         num_to_process -= num_tasks
         start += num_tasks

   # for vds in vds_list:
   #    c = crop_virtual_dataset_to_bbox(vds, bbox, netcdf_store)
   #    if c is not None:
   #       cropped.append(c)

   #    else:
   #       skipped_granules.append(vds.attrs["granule_url"])

   if not cropped:
      raise ValueError(f"Building virtual cube: no granules overlaps bbox {bbox}")

   logging.info(f'Got {len(cropped)} cropped granules')

   for i, vds in enumerate(cropped):
      t = vds["time"]
      logging.info(f"Granule {i}: {t.values=} {t.dtype=} {t.dims=} {t.shape=}")

   _assert_identical_grids(cropped, bbox)

   logging.info(f'Number of skipped granules: {len(skipped_granules)}')
   return build_virtual_cube(cropped), skipped_granules


def read_virtual_dataset(granule_url, parser, registry):
   """Read granule into virtual dataset.

   Parameters
   ----------
   granule_url: str
      Granule url to open.
   parser: virtualizarr.parsers.HDFParser
      Parser to open granule url with.
   """
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

   return v


def load_granules(granules, bucket):
   """Load granules into virtual datasets.

   Parameters
   ----------
   granule_url: list(str)
      Granules to load.
   bucket: str
      AWS S3 bucket that stores granules.

   """
   store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
   registry = ObjectStoreRegistry({bucket: store})
   parser = HDFParser(drop_variables=["mapping"])

   vds_list = []
   start = 0
   num_to_process = len(granules)

   # Use processes ("loky") instead of threads ("threading") for parallel
   # processing - each process is getting their own copy of the object instance
   # (registry, object store, etc.) that are passed to each of the processes,
   # Using loky (process-based) bypasses the threading-lock contention entirely,
   # by construction, because there is no shared process for a lock to live in.
   with parallel_config(
      # backend='threading',
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
   import json
   import os
   import shutil

   import obstore
   import virtualizarr as vz
   from virtualizarr.parsers import HDFParser
   from obspec_utils.registry import ObjectStoreRegistry
   import icechunk as ic

   from virtual_itslive_cube import _drop_nonfinite_attrs

   parser = argparse.ArgumentParser(
      description="""
      Build a virtual ITS_LIVE datacube from granules, restricted to a bounding box.

      Usage examples:
      # Using JSON file for granules
      python src/virtual_itslive_cube_per_chunk.py \
         --granules-file granules.json \
         --bbox '[-1658887.5, -1597447.5, -430072.5, -368632.5]' \
         --output-store output.icechunk

      # Using 4 granules with valid data in cube's polygon
      python ./virtual_itslive_cube_per_chunk.py --bbox '[-1658887.5, -1597447.5, -430072.5, -368632.5]' --granules-file virtual_input_4files.json --output-store its_live_cube_subset_m11_m12_s1_s2_landsat.icechunk

      # Using 39 input granules with only 7 having valid data in cube's polygon:
      python ./virtual_itslive_cube_per_chunk.py --bbox '[-1658887.5, -1597447.5, -430072.5, -368632.5]' --granules-file virtual_input_39files.json --output-store its_live_cube_subset_m11_m12_s1_s2_landsat.icechunk

      # Using direct granule list
      python src/virtual_itslive_cube_per_chunk.py \
         --granules granule1.nc granule2.nc granule3.nc \
         --bbox '[-1658887.5, -1597447.5, -430072.5, -368632.5]'
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

   parser.add_argument(
      "--bbox",
      type=str,
      required=True,
      help="Bounding box as JSON list string: '[xmin, xmax, ymin, ymax]'"
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
      help="S3 bucket URL (default: s3://its-live-data)"
   )
   parser.add_argument(
      '-t', '--threads',
      type=int,
      default=8,
      help='Number of threads to use for parallel processing [%(default)d].'
   )

   args = parser.parse_args()
   MAX_AWS_CONNECTIONS = args.threads

   # Load granules from either JSON file or command-line arguments
   if args.granules_file:
      logging.info(f"Loading granules from {args.granules_file}")
      with open(args.granules_file, 'r') as f:
         granules = json.load(f)

      if not isinstance(granules, list):
         raise ValueError(f"JSON file must contain a list of granule paths")
   else:
      granules = args.granules

   granules = [
      each.replace(
         'https://its-live-data.s3.amazonaws.com/',
         's3://its-live-data/') for each in granules
   ]

   logging.info(f"Processing {len(granules)} granules")

   # Parse bounding box from JSON string
   bbox_list = json.loads(args.bbox)

   if not isinstance(bbox_list, list) or len(bbox_list) != 4:
      raise ValueError(f"bbox must be a list with 4 values [xmin, xmax, ymin, ymax], got {bbox_list}")

   # Adjust cube cell edge coordinates for the cell centers
   # bbox_list is [xmin, xmax, ymin, ymax]
   bbox_list[0] = bbox_list[0] + PIXEL_SIZE_HALF  # xmin
   bbox_list[1] = bbox_list[1] - PIXEL_SIZE_HALF  # xmax
   bbox_list[2] = bbox_list[2] + PIXEL_SIZE_HALF  # ymin
   bbox_list[3] = bbox_list[3] - PIXEL_SIZE_HALF  # ymax

   bucket = args.bucket

   # Original granules for testing
   # granules = [
   #    "velocity_image_pair/landsatOLI/v02/S80W170/LC08_L1GT_020121_20231013_20231102_02_T2_X_LC09_L1GT_020121_20231106_20231106_02_T2_G0120V02_P084.nc",
   #    "velocity_image_pair/landsatOLI/v02/S80W170/LC08_L1GT_020120_20201121_20210315_02_T2_X_LC08_L1GT_020120_20210124_20210305_02_T2_G0120V02_P051.nc",
   # ]

   # Now test with S1, Landsat and S2 granules - these all have Nan data
   # for the cube overlap
   # granules = [
   #    "velocity_image_pair/landsatOLI/v02/S70W090/LC08_L1GT_002112_20140208_20201016_02_T2_X_LC08_L1GT_002112_20140312_20200911_02_T2_G0120V02_P062.nc",
   #    "velocity_image_pair/sentinel1/v02/S70W100/S1A_IW_SLC__1SSH_20251223T043633_20251223T043654_062437_07D279_7E42_X_S1A_IW_SLC__1SSH_20260104T043632_20260104T043653_062612_07D939_651F_G0120V02_P030.nc",
   #    "velocity_image_pair/sentinel2/v02/S70W100/S2B_MSIL1C_20181208T151259_N0207_R139_T13CET_20181208T180302_X_S2B_MSIL1C_20190206T151259_N0207_R139_T13CET_20190206T180306_G0120V02_P055.nc"
   # ]

   # Using dev_notebooks/issues/virtualizarr/cube_with_s1_granules.ipynb
   # identified S1, S2, Landsat granules that actually have data in the
   # cube polygon
   # granules = [
   #    "velocity_image_pair/sentinel2/v02/S70W100/S2B_MSIL1C_20210215T151259_N0209_R139_T13CET_20210215T181843_X_S2B_MSIL1C_20220121T151259_N0301_R139_T13CET_20220121T181438_G0120V02_P029.nc",
   #    "velocity_image_pair/sentinel1/v02/S70W100/S1A_IW_SLC__1SSH_20251125T050857_20251125T050919_062029_07C28A_EB80_X_S1A_IW_SLC__1SSH_20251207T050856_20251207T050917_062204_07C968_3C90_G0120V02_P029.nc",
   #    "velocity_image_pair/sentinel1/v02/S70W100/S1C_IW_SLC__1SSH_20251201T050755_20251201T050816_005253_00A6DB_DFED_X_S1A_IW_SLC__1SSH_20251207T050856_20251207T050917_062204_07C968_3C90_G0120V02_P034.nc",
   #    "velocity_image_pair/landsatOLI/v02/S70W100/LE07_L1GT_002113_20121024_20200908_02_T2_X_LE07_L1GT_001113_20121118_20200908_02_T2_G0120V02_P024.nc"
   # ]


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

   bbox = tuple(bbox_list)

   # Store to load each of the granule's "v" values to check if granule has
   # any valid "v" data in the cube polygon
   netcdf_store = S3Store(
      bucket="its-live-data",
      region="us-west-2",
      skip_signature=True,
   )

   cube, skipped_granules = build_virtual_cube_subset(vds_list, bbox, netcdf_store)
   print(f"\n{cube}")

   format_skipped_granules = "\n".join(skipped_granules)
   logging.info(f'Skipped granules: \n{format_skipped_granules}')

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
   snapshot_id = session.commit("its_live virtual cube subset: cropped to bbox")
   logging.info(f"icechunk committed snapshot: {snapshot_id=}")

   cube_roundtrip = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
   logging.info(f"{cube_roundtrip=}")

   logging.info(f'{cube_roundtrip.mission_img1.values=}')
   logging.info(f'{cube_roundtrip.mission_img2.values=}')
   logging.info(f'{cube_roundtrip.satellite_img1.values=}')
   logging.info(f'{cube_roundtrip.satellite_img2.values=}')
   logging.info(f'{cube_roundtrip.time.values=}')
