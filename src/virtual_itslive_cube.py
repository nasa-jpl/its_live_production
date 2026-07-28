import obstore
import os
import virtualizarr as vz
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
import xarray as xr
import zarr
import numpy as np
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH
from virtualizarr.manifests.utils import copy_and_replace_metadata

# Icechunk repo related
import shutil
import icechunk as ic


def pad_manifestarray(marr, new_shape, offsets=None):
   """Smart padding: place a ManifestArray into a larger `new_shape` at the
   given per-axis element `offsets`, filling every other chunk-grid cell with
   nodata chunks (ChunkEntry path="", which reads back as the array's fill
   value). Only chunk references are moved -- no pixel data is read.

   Parameters
   ----------
   marr : ManifestArray
      The array to place. Its existing chunk references are preserved.
   new_shape : sequence of int
      Target shape (same ndim as `marr`, >= its shape on every axis).
   offsets : sequence of int, optional
      Element offset of `marr`'s origin within `new_shape` on each axis.
      Defaults to 0 everywhere (top-left). Each offset must be a multiple of
      that axis' chunk size, and a partial final chunk is only allowed if the
      data reaches the grid edge -- so granules tile without overlapping
      partial chunks.
   """
   new_shape = tuple(int(s) for s in new_shape)
   if offsets is None:
      offsets = (0,) * len(new_shape)
   offsets = tuple(int(o) for o in offsets)
   shape, chunks = marr.shape, marr.chunks

   if len(new_shape) != len(shape):
      raise ValueError(f"new_shape {new_shape} ndim != array ndim {len(shape)}")
   for ax, (off, s, c, n) in enumerate(zip(offsets, shape, chunks, new_shape)):
      if off % c != 0:
         raise ValueError(f"axis {ax}: offset {off} not a multiple of chunk size {c}")
      if off + s > n:
         raise ValueError(f"axis {ax}: offset {off}+size {s} exceeds new size {n}")
      if s % c != 0 and off + s != n:
         raise ValueError(
               f"axis {ax}: size {s} is not a multiple of chunk {c} and does not "
               f"reach the grid edge ({off}+{s} != {n}); cannot tile cleanly"
         )

   old_grid = marr.manifest.shape_chunk_grid
   new_grid = tuple(-(-s // c) for s, c in zip(new_shape, chunks))
   chunk_offsets = [o // c for o, c in zip(offsets, chunks)]

   # build the new chunk-grid arrays, initialised to the nodata sentinel,
   # then drop the existing references in at the chunk offset
   new_paths = np.full(new_grid, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
   new_offsets = np.zeros(new_grid, dtype="uint64")
   new_lengths = np.zeros(new_grid, dtype="uint64")
   region = tuple(slice(co, co + og) for co, og in zip(chunk_offsets, old_grid))
   new_paths[region] = marr.manifest._paths
   new_offsets[region] = marr.manifest._offsets
   new_lengths[region] = marr.manifest._lengths

   new_manifest = ChunkManifest.from_arrays(
      paths=new_paths, offsets=new_offsets, lengths=new_lengths,
      validate_paths=False,  # existing paths already valid, new ones are sentinel
      inlined=marr.manifest._inlined or None,
   )
   # copy_and_replace_metadata should preserve dtype, but let's verify
   new_metadata = copy_and_replace_metadata(marr.metadata, new_shape=list(new_shape))

   # Ensure dtype is preserved (copy_and_replace_metadata should do this, but be explicit)
   if hasattr(marr.metadata, 'dtype') and hasattr(new_metadata, 'dtype'):
      if new_metadata.dtype != marr.metadata.dtype:
         print(f"Warning: pad_manifestarray dtype changed from {marr.metadata.dtype} to {new_metadata.dtype}")

   return ManifestArray(metadata=new_metadata, chunkmanifest=new_manifest)


def _union_axis(starts, ends, step):
   """Regular coordinate vector covering [starts..ends] at spacing `step`."""
   if step > 0:
      lo, hi = min(starts), max(ends)
   else:
      lo, hi = max(starts), min(ends)
   n = round((hi - lo) / step) + 1
   return lo + np.arange(n) * step


def extend_coords(vds_list):
   """Compute the common (union) x/y coordinate vectors covering all granules,
   plus each granule's per-axis element offset into that grid.

   The granules' x/y coordinates must be loaded (real values) and share a
   common posting and lattice. Returns ``(x_union, y_union, offsets)`` where
   ``offsets[i]`` is ``{"x": ix, "y": iy}`` for granule ``i``.

   ``time`` is not extended here -- each granule keeps its own time value and
   ``combine_by_coords`` stacks/orders them along the time dimension.
   """
   xs = [vds["x"].values for vds in vds_list]
   ys = [vds["y"].values for vds in vds_list]
   dx = float(xs[0][1] - xs[0][0])
   dy = float(ys[0][1] - ys[0][0])
   for x, y in zip(xs, ys):
      if not (np.isclose(x[1] - x[0], dx) and np.isclose(y[1] - y[0], dy)):
         raise ValueError("all datasets must share the same x/y posting")

   x_union = _union_axis([x[0] for x in xs], [x[-1] for x in xs], dx)
   y_union = _union_axis([y[0] for y in ys], [y[-1] for y in ys], dy)
   offsets = [
      {"x": round((x[0] - x_union[0]) / dx), "y": round((y[0] - y_union[0]) / dy)}
      for x, y in zip(xs, ys)
   ]
   return x_union, y_union, offsets


def _add_missing_m11_m12(new_vars, vds, x_union, y_union):
   """Add M11 and M12 data variables as ManifestArrays with missing chunks if not
   present in the granule.

   M11 and M12 are conversion matrix elements for radar data (see Eq. A18 in
   https://www.mdpi.com/2072-4292/13/4/749). Optical granules don't have these
   variables, so we create ManifestArrays with all-missing chunks that read as
   fill value. This ensures consistency when combining optical and radar granules.

   Parameters
   ----------
   new_vars : dict
      Dictionary to add M11/M12 variables to
   vds : xr.Dataset
      Virtual dataset for the granule
   x_union : np.ndarray
      Union x coordinates for the datacube
   y_union : np.ndarray
      Union y coordinates for the datacube
   """
   # M11 and M12 metadata based on itscube.py and itscube_types.py
   m_vars_info = {
      'M11': {
         'standard_name': 'conversion_matrix_element_11',
         'description': 'conversion matrix element (1st row, 1st column) that can be '
                        'multiplied with vx to give range pixel displacement dr (see '
                        'Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)',
         'units': 'pixel/(meter/year)'
      },
      'M12': {
         'standard_name': 'conversion_matrix_element_12',
         'description': 'conversion matrix element (1st row, 2nd column) that can be '
                        'multiplied with vy to give range pixel displacement dr (see '
                        'Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)',
         'units': 'pixel/(meter/year)'
      }
   }

   for m_var_name, m_var_attrs in m_vars_info.items():
      if m_var_name not in vds.data_vars:
         # Create ManifestArray with missing chunks for optical granules
         # Use chunk size matching other 2D variables in the granule (typically 512x512 or similar)
         # Default to 512 if we can't determine from existing variables
         chunk_y = chunk_x = 512

         # Try to get chunk size from an existing 2D ManifestArray variable
         for var in vds.data_vars.values():
            if isinstance(var.data, ManifestArray) and len(var.dims) == 2:
               dims = var.dims
               if 'y' in dims and 'x' in dims:
                  y_idx = dims.index('y')
                  x_idx = dims.index('x')
                  chunk_y = var.data.chunks[y_idx]
                  chunk_x = var.data.chunks[x_idx]
                  break

         shape = (len(y_union), len(x_union))
         chunks = (chunk_y, chunk_x)

         # Create chunk grid filled with MISSING_CHUNK_PATH
         from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH
         chunk_grid_shape = (-(-shape[0] // chunks[0]), -(-shape[1] // chunks[1]))

         paths = np.full(chunk_grid_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
         offsets = np.zeros(chunk_grid_shape, dtype="uint64")
         lengths = np.zeros(chunk_grid_shape, dtype="uint64")

         manifest = ChunkManifest.from_arrays(
            paths=paths,
            offsets=offsets,
            lengths=lengths,
            validate_paths=False
         )

         # Create array metadata with proper fill value for float32
         fill_value = np.float32(np.nan)

         # Create Zarr array metadata for float32 with fill value
         from zarr.core.metadata import ArrayV3Metadata
         from zarr.core.chunk_grids import RegularChunkGrid
         from zarr.core.common import ChunkCoords

         metadata = ArrayV3Metadata(
            shape=shape,
            data_type=np.dtype('float32'),
            chunk_grid=RegularChunkGrid(chunk_shape=ChunkCoords(chunks)),
            chunk_key_encoding={'name': 'default', 'separator': '/'},
            fill_value=fill_value,
            codecs=[]
         )

         manifest_array = ManifestArray(metadata=metadata, chunkmanifest=manifest)

         new_vars[m_var_name] = xr.Variable(
            dims=('y', 'x'),
            data=manifest_array,
            attrs={
               'standard_name': m_var_attrs['standard_name'],
               'description': m_var_attrs['description'],
               'units': m_var_attrs['units'],
               '_FillValue': fill_value
            }
         )
         print(f"Added {m_var_name} as ManifestArray with missing chunks (not present in granule)")


def build_virtual_cube(vds_list):
   """Mosaic virtual datasets onto their common x/y grid and stack along time.

   Three steps:
   1. ``extend_coords``    -> union x/y grid + each granule's offset into it
   2. ``pad_manifestarray``-> drop each granule's data at its offset on the
                              union grid (nodata everywhere else)
   3. ``combine_by_coords``-> order/stack the granules along time

   Data variables stay virtual (ManifestArray); x/y/time are real, indexed
   coordinates. No pixel data is read.
   """
   x_union, y_union, offsets = extend_coords(vds_list)
   sizes = {"x": len(x_union), "y": len(y_union)}

   placed = []
   for vds, off in zip(vds_list, offsets):
      new_vars = {}
      for name, var in vds.data_vars.items():
         # Skip img_pair_info - we'll extract mission_img1 from it instead
         if name == "img_pair_info":
            # Add "mission_img*" and "satellite_img*"" (img_pair_info attributes):
            # have to add all of the attributes that represent cube's data
            # variables. This is just to proof a concept that we can
            # introduce attributes as new variables in the virtual dataset.
            for attr_name in [
               "mission_img1", "mission_img2",
               "satellite_img1", "satellite_img2"
            ]:
               attr_value = vds["img_pair_info"].attrs.get(attr_name, "")
               new_vars[attr_name] = xr.Variable(
                  dims=(),
                  data=np.array(attr_value),
                  attrs={"description": attr_name}
               )
            continue  # Skip adding img_pair_info itself to the cube

         # Extract vx attributes as scalar data variables (e.g., error_modeled)
         if name == 'vx':
            attr_value = vds["vx"].attrs.get("error_modeled", -32767)
            print(f'Getting vx.error_modeled {attr_value}')
            new_vars["vx_error_modeled"] = xr.Variable(
               dims=(),
               data=np.array(attr_value),
               attrs={"description": "vx_error_modeled"}
            )
            # Continue to add vx itself (don't skip like img_pair_info)

         data = var.data
         original_dtype = var.dtype  # Capture original dtype BEFORE any operations
         if isinstance(data, ManifestArray):
               new_shape = [sizes.get(str(d), s) for d, s in zip(var.dims, data.shape)]
               var_offsets = [off.get(str(d), 0) for d in var.dims]
               data = pad_manifestarray(data, new_shape, var_offsets)
               # Verify dtype is preserved in ManifestArray
               if data.dtype != original_dtype:
                  print(f"Warning: {name} dtype changed from {original_dtype} to {data.dtype} during padding")
         # Create variable with original dtype preserved
         new_vars[name] = xr.Variable(var.dims, data, attrs=var.attrs, encoding=var.encoding)

      # Add M11 and M12 if not present in granule (optical granules don't have these)
      # _add_missing_m11_m12(new_vars, vds, x_union, y_union)

      placed.append(xr.Dataset(
         new_vars,
         coords={"x": ("x", x_union), "y": ("y", y_union), "time": vds["time"]},
         attrs=vds.attrs,
      ))

   # Use compat="override" but log dtype changes
   # Print dtype info before combining
   print("\nDtypes before combine_by_coords:")
   for var_name in placed[0].data_vars:
      dtypes = [ds[var_name].dtype for ds in placed if var_name in ds.data_vars]
      if len(set(str(d) for d in dtypes)) > 1:
         print(f"  {var_name}: {dtypes} (MISMATCH)")
      else:
         print(f"  {var_name}: {dtypes[0]}")

   result = xr.combine_by_coords(
      placed, coords="minimal", compat="override", join="override",
      combine_attrs="drop_conflicts",
   )

   # Check dtypes after combining
   print("\nDtypes after combine_by_coords:")
   for var_name in result.data_vars:
      print(f"  {var_name}: {result[var_name].dtype}")

   return result

# icechunk metadata is strict JSON and cannot represent NaN/inf, but ITS_LIVE
# stores NaN in some attributes (e.g. 'stable_shift_stationary'); drop those.
def _drop_nonfinite_attrs(ds):
   ds = ds.copy()

   def clean(attrs):
      keep = {}
      for k, v in attrs.items():
            arr = np.asarray(v)
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
               continue
            keep[k] = v
      return keep

   ds.attrs = clean(ds.attrs)
   for var in ds.variables.values():
      var.attrs = clean(var.attrs)
   return ds


if __name__ == "__main__":
   bucket = "s3://its-live-data"
   # key = "test-space/virtual-cubes"
   # key = "test-space/virtual-cubes/fromJoe/fillValue"
   key = 'velocity_image_pair/landsatOLI/v02/S80W170'

   granules = [
      # to confirm Joe's fix to cropped granule to change dtype of img_pair_info
      # 'LC08_L1TP_009011_20200703_20200913_02_T1_X_LC08_L1TP_009011_20200820_20200905_02_T1_G0120V02_P078_cropped.nc'
      "LC08_L1GT_020121_20231013_20231102_02_T2_X_LC09_L1GT_020121_20231106_20231106_02_T2_G0120V02_P084.nc",
      "LC08_L1GT_020120_20201121_20210315_02_T2_X_LC08_L1GT_020120_20210124_20210305_02_T2_G0120V02_P051.nc"
   ]

   store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
   registry = ObjectStoreRegistry({bucket: store})
   # parser = HDFParser(drop_variables=["mapping", "img_pair_info"])
   parser = HDFParser(drop_variables=["mapping"])

   vds = []

   for granule in granules:
      vds.append(
         vz.open_virtual_dataset(
            url=os.path.join(bucket, key, granule),
            parser=parser,
            registry=registry,
            loadable_variables=["time", "y", "x"],
            decode_times=True,
         )
      )

   print(f"{vds}")


   # mosaic both granules onto their common grid and stack along time
   cube = build_virtual_cube(vds)
   print(f"\n{cube}")

   # # Display img_pair_info attributes for each time step in the cube
   # if "img_pair_info" in cube.data_vars:
   #    print(f"\nimg_pair_info attributes in cube:")
   #    for i, time_val in enumerate(cube.time.values):
   #       print(f"\nTime step {i} ({time_val}):")
   #       img_pair_info_at_time = cube["img_pair_info"].isel(time=i)
   #       for attr_name, attr_value in img_pair_info_at_time.attrs.items():
   #          print(f"  {attr_name}: {attr_value}")


   # the referenced chunk data lives on this public, anonymous S3 bucket
   url_prefix = "s3://its-live-data/"
   store_path = "its_live_cube.icechunk"

   # start fresh so this cell is re-runnable
   shutil.rmtree(store_path, ignore_errors=True)

   # register a virtual chunk container so icechunk knows how to read the s3 refs
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

   # write the cube: only the virtual references are stored, no pixel data is copied
   session = repo.writable_session("main")

   # Print dtypes before writing to icechunk
   print("\nDtypes before writing to icechunk:")
   for var_name in cube.data_vars:
      var = cube[var_name]
      dtype = var.dtype
      if isinstance(var.data, ManifestArray):
         manifest_dtype = var.data.dtype
         print(f"  {var_name}: xr.Variable dtype={dtype}, ManifestArray dtype={manifest_dtype}")
      else:
         print(f"  {var_name}: dtype={dtype} (not ManifestArray)")

   cube_clean = _drop_nonfinite_attrs(cube)

   # Write using vz.to_icechunk
   cube_clean.vz.to_icechunk(session.store)

   snapshot_id = session.commit("its_live virtual cube: 2 granules mosaicked + stacked on time")
   print("committed snapshot", snapshot_id)

   # reopen from the committed store
   cube_roundtrip = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
   print(f'{cube_roundtrip=}')