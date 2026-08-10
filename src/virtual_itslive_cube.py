import obstore
import os
import virtualizarr as vz
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
from obspec_utils.registry import ObjectStoreRegistry
import xarray as xr
import zarr
import numpy as np
import logging
from virtualizarr.manifests import ChunkManifest, ManifestArray
from virtualizarr.manifests.manifest import MISSING_CHUNK_PATH
from virtualizarr.manifests.utils import copy_and_replace_metadata
from zarr.core.metadata.v3 import ArrayV3Metadata

# Icechunk repo related
import shutil
import icechunk as ic

import utils
from itslive_binary_type import BinaryFlag
from itscube_types import (
   ImgPairInfo,
   Vars
)

HTTP_PREFIX = 'http://'
HTTPS_PREFIX = 'https://'


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
      paths=new_paths,
      offsets=new_offsets,
      lengths=new_lengths,
      validate_paths=False,  # existing paths already valid, new ones are sentinel
      inlined=marr.manifest._inlined or None,
   )
   # copy_and_replace_metadata should preserve dtype, but let's verify
   new_metadata = copy_and_replace_metadata(
      marr.metadata, new_shape=list(new_shape)
   )

   # Ensure dtype is preserved (copy_and_replace_metadata should do this,
   # but be explicit)
   if hasattr(marr.metadata, 'dtype') and hasattr(new_metadata, 'dtype'):
      if new_metadata.dtype != marr.metadata.dtype:
         logging.info(
            f"Warning: pad_manifestarray dtype changed from {marr.metadata.dtype} to {new_metadata.dtype}")

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


def copy_and_replace_metadata_dtype(old_metadata, new_shape, new_dtype, new_fill_value, codecs):
   """Like virtualizarr's `copy_and_replace_metadata`, but also overrides
   dtype, fill_value, and codec configuration.

   `copy_and_replace_metadata` has no `new_dtype` param -- it always inherits
   `data_type` (and codecs) from `old_metadata` -- so changing dtype has to go
   through zarr's `ArrayV3Metadata` dict round-trip directly.

   Parameters
   ----------
   codecs : sequence of codec dicts
      Used verbatim as the new metadata's codecs. Always required: the
      physical on-disk itemsize doesn't necessarily match `new_dtype`'s
      itemsize (e.g. packed/scaled storage), so there's no dtype-derived
      fallback to guess from -- the caller must supply a codec spec known
      to be correct (see `_HARDCODED_M_VAR_PLACEHOLDER_CODECS`).
   """
   metadata_dict = old_metadata.to_dict().copy()
   metadata_dict["shape"] = tuple(int(s) for s in new_shape)
   metadata_dict["data_type"] = new_dtype        # e.g. "float32"
   metadata_dict["fill_value"] = new_fill_value  # e.g. float("nan")
   metadata_dict["codecs"] = codecs

   return ArrayV3Metadata.from_dict(metadata_dict)


# Hard-coded codecs for synthesized M11/M12 placeholders (used only for
# granules missing real M11/M12; granules that already have real data are
# left untouched and never go through this path). Captured directly from a
# granule confirmed to have REAL M11/M12 data, so it's the exact,
# known-correct codec structure for this variable -- not an approximation
# derived from an unrelated variable like `vx` (which is packed int16 and
# would need patching).
#
# IMPORTANT: this is the dict form zarr's ArrayV3Metadata.from_dict()
# expects (i.e. what `.metadata.to_dict()["codecs"]` produces), not the
# parsed codec objects you see printed in logs (e.g. `Shuffle(...)` reprs).
# If it ever needs to be re-captured (e.g. a schema change), run this on a
# granule where M11 is real, not synthesized:
#
#   print(vds["M11"].data.metadata.to_dict()["codecs"])
#
# and paste the result below verbatim.
_HARDCODED_M_VAR_PLACEHOLDER_CODECS = (
   {'name': 'bytes', 'configuration': {'endian': 'little'}},
   {'name': 'numcodecs.shuffle', 'configuration': {'elementsize': 4}},
   {'name': 'numcodecs.zlib', 'configuration': {'level': 2}},
)


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
   reference_codecs : dict
      Mapping {"M11": codecs, "M12": codecs}, resolved once per cube by
      `_resolve_placeholder_codecs` (see `_HARDCODED_M_VAR_PLACEHOLDER_CODECS`).
      Used verbatim for any placeholder synthesized here. Real M11/M12 data
      (when present in `vds`) is untouched and never reaches this codepath.
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

   if 'M11' in vds.data_vars:
      # Variable is already in the dataset, nothing to do
      return

   for m_var_name, m_var_attrs in m_vars_info.items():
      # Create ManifestArray with missing chunks for optical granules
      # Use an existing 3D ManifestArray variable (time, y, x) as a template
      # Granules have 3D variables with single-valued time dimension
      template_var = None
      for var in vds.data_vars.values():
         if isinstance(var.data, ManifestArray) and len(var.dims) == 3:
            dims = var.dims
            if 'time' in dims and 'y' in dims and 'x' in dims:
               template_var = var
               break

      if template_var is None:
         # No suitable ManifestArray found, skip M11/M12 for this granule
         logging.info(f"Warning: No 3D ManifestArray template found for {m_var_name}, skipping")
         continue

      # Get shape and chunks from template - create 3D (time, y, x) just like other variables
      # This ensures M11/M12 can be concatenated along time dimension later
      time_idx = template_var.dims.index('time')
      y_idx = template_var.dims.index('y')
      x_idx = template_var.dims.index('x')

      # Create 3D shape with time=1 (single time slice)
      shape = (1, len(y_union), len(x_union))

      # Get chunk sizes from template (time, y, x)
      chunk_time = template_var.data.chunks[time_idx]
      chunk_y = template_var.data.chunks[y_idx]
      chunk_x = template_var.data.chunks[x_idx]
      chunks = (chunk_time, chunk_y, chunk_x)

      # Create 3D chunk grid filled with MISSING_CHUNK_PATH (all missing chunks)
      chunk_grid_shape = (
         -(-shape[0] // chunks[0]),  # time chunks
         -(-shape[1] // chunks[1]),  # y chunks
         -(-shape[2] // chunks[2])   # x chunks
      )

      paths = np.full(chunk_grid_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType())
      offsets = np.zeros(chunk_grid_shape, dtype="uint64")
      lengths = np.zeros(chunk_grid_shape, dtype="uint64")

      manifest = ChunkManifest.from_arrays(
         paths=paths,
         offsets=offsets,
         lengths=lengths,
         validate_paths=False
      )

      # Create metadata by copying from template and updating shape and dtype.
      # M11/M12 must be float32 to match radar granules, regardless of
      # whichever dtype `template_var` happens to have (it's only used here
      # for its 3D (time, y, x) shape/chunking) -- so force the dtype.
      #
      # NOTE: only set via the zarr-level `fill_value` (new_fill_value
      # below), NOT also as an xr.Variable `encoding={'_FillValue': ...}`.
      # virtualizarr's to_icechunk writer writes whatever's in `encoding`
      # verbatim into the zarr array's raw attributes, bypassing xarray's
      # own FillValueCoder.encode() step (which base64-encodes floats) --
      # so a raw float lands in `attributes["_FillValue"]`. On read,
      # xr.open_zarr's FillValueCoder.decode() then chokes on that raw
      # float, expecting the base64-encoded string its own writer would
      # have produced (TypeError: "expected str or bytes ... got float").
      # The zarr-level fill_value alone is sufficient: xarray's zarr
      # backend reads `_FillValue` directly from `zarr_array.fill_value`
      # (no decode step) when `use_zarr_fill_value_as_mask=True`, or
      # exposes it as `encoding["fill_value"]` otherwise -- either way,
      # no FillValueCoder round-trip is involved when no separate
      # `_FillValue` attribute is present.
      fill_value = np.float32(-32767.0)

      new_metadata = copy_and_replace_metadata_dtype(
         template_var.data.metadata,
         new_shape=list(shape),
         new_dtype="float32",
         new_fill_value=float(fill_value),
         codecs=_HARDCODED_M_VAR_PLACEHOLDER_CODECS,
      )

      manifest_array = ManifestArray(metadata=new_metadata, chunkmanifest=manifest)

      new_vars[m_var_name] = xr.Variable(
         dims=('time', 'y', 'x'),
         data=manifest_array,
         attrs={
            'standard_name': m_var_attrs['standard_name'],
            'description': m_var_attrs['description'],
            'units': m_var_attrs['units'],
         },
         # _FillValue belongs in encoding, not attrs, per xarray's CF
         # convention -- attrs is inert metadata, encoding is what
         # actually participates in xarray's encode/decode machinery.
         # encoding={'_FillValue': fill_value},

      )

      logging.info(f"Added {m_var_name} as 3D ManifestArray (time, y, x) with missing chunks (not present in granule)")


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

   # Collect only a subset of data variables in the virtual datacube
   _vars = [ImgPairInfo.name, Vars.v, Vars.vx, Vars.vy, Vars.m11, Vars.m12]

   placed = []

   # Only one value as virtual cube attribute should be preserved, but need
   # to confirm that all layers have the same parameter file.
   autorift_param_files = []

   for vds, off in zip(vds_list, offsets):
      new_vars = {}
      for name, var in vds.data_vars.items():
         if name not in _vars:
            # Skip data variable if it's not going to be in a virtual cube
            continue

         # Skip img_pair_info but extract its attributes into new data variables
         # within virtual datacube
         if name == ImgPairInfo.name:
            for attr in ImgPairInfo.all:
               # Add new variables that correspond to selected attributes of
               # 'img_pair_info'
               attr_dtype = None
               if attr in ImgPairInfo.allTypes:
                  attr_dtype = ImgPairInfo.allTypes[attr]

               # Flag if value should be converted to date type
               convert_to_date = attr in ImgPairInfo.toDate

               new_var_attrs = {
                  Vars.attrs.std_name: ImgPairInfo.stdName[attr],
                  Vars.attrs.description: ImgPairInfo.allDescriptions[attr]
               }
               if attr in ImgPairInfo.allUnits:
                  # Units attribute exists for new variable
                  new_var_attrs[utils.Units.name] = ImgPairInfo.allUnits[attr]

               new_vars[attr] = xr.Variable(
                  dims=(),
                  data=utils.get_data_var_attr(
                        vds,
                        vds.attrs[Vars.url],
                        ImgPairInfo.name,
                        attr,
                        to_date=convert_to_date,
                        data_dtype=attr_dtype
                     ),
                  attrs=new_var_attrs
               )

            for (each, new_each) in zip(
                  [ImgPairInfo.flight_direction_img1, ImgPairInfo.flight_direction_img2],
                  [Vars.ascending_img1, Vars.ascending_img2]
            ):
                  # Add new variables that correspond to flight direction attributes
                  # of 'img_pair_info'
                  new_vars[new_each] = xr.Variable(
                     data=utils.get_data_var_binary_attr(
                        vds,
                        vds.attrs[Vars.url],
                        ImgPairInfo.name,
                        each,
                        ImgPairInfo.ascending,
                        data_dtype=np.uint8,
                        missing_value=utils.Missing.u8value
                     ),
                     dims=(),
                     attrs={
                        Vars.attrs.std_name: Vars.name[new_each],
                        Vars.attrs.description: Vars.description[new_each],
                        BinaryFlag.attrs.values: BinaryFlag.values,
                        BinaryFlag.attrs.meanings: BinaryFlag.meanings[new_each]
                     }
                  )

            continue  # Skip adding img_pair_info itself to the cube

         # Add autoRIFT_software_version
         new_vars[Vars.autorift_software_version] = xr.Variable(
               data=vds.attrs[Vars.autorift_software_version],
               dims=(),
               attrs={
                  Vars.attrs.std_name: Vars.autorift_software_version,
                  Vars.attrs.description: Vars.description[Vars.autorift_software_version]
               }
         )

         # Remember autoRIFT_parameter_file value - must be the same across
         # all cube layers
         autorift_param_files.append(vds.attrs[Vars.attrs.autorift_param_file])

         # Process 'v[xy]' data variables and their attributes
         # if name in [Vars.vx, Vars.vy]:


         # Extract "v*"'s attributes as scalar data variables (e.g., error_modeled)

         if name == 'vx':
            attr_value = vds["vx"].attrs.get("error_modeled", -32767)
            logging.info(f'Getting vx.error_modeled {attr_value}')
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
                  raise RuntimeError(
                     f"{name} dtype changed from {original_dtype} to "
                     f"{data.dtype} during padding"
                  )

         # Create variable with original dtype preserved
         new_vars[name] = xr.Variable(
            var.dims, data, attrs=var.attrs, encoding=var.encoding
         )

      # Add source granule url for the layer in the cube
      new_vars[Vars.url] = xr.Variable(
         dims=("time",),
         data=np.array([vds.attrs[Vars.url]],
                        dtype=np.dtypes.StringDType()),
         attrs={"description": "source granule URL for this time step"},
      )

      # Add M11 and M12 if not present in granule
      # (optical granules don't have these)
      _add_missing_m11_m12(new_vars, vds, x_union, y_union)

      placed.append(xr.Dataset(
         new_vars,
         coords={"x": ("x", x_union), "y": ("y", y_union), "time": vds["time"]},
         attrs=vds.attrs,
      ))

   result = xr.combine_by_coords(
      placed, coords="minimal", compat="override", join="override",
      combine_attrs="drop_conflicts", data_vars="all"
   )

   # Some param files have 'http' and some 'https' - remove them before comparison
   autorift_param_files = [
      each.replace(HTTPS_PREFIX, '').replace(HTTP_PREFIX, '') \
      for each in autorift_param_files
   ]
   unique_values = list(set(autorift_param_files))
   if len(unique_values) > 1:
      raise RuntimeError(
         f"Multiple values for '{Vars.attrs.autorift_param_file}' "
         f"are detected for {len(vds_list)} granules: "
         f"{unique_values} (one value is expected)"
      )

   return result, HTTPS_PREFIX + unique_values[0]


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
   key = 'velocity_image_pair/landsatOLI/v02/S80W170'

   granules = [
      "LC08_L1GT_020121_20231013_20231102_02_T2_X_LC09_L1GT_020121_20231106_20231106_02_T2_G0120V02_P084.nc",
      "LC08_L1GT_020120_20201121_20210315_02_T2_X_LC08_L1GT_020120_20210124_20210305_02_T2_G0120V02_P051.nc"
   ]

   store = obstore.store.from_url(bucket, region="us-west-2", skip_signature=True)
   registry = ObjectStoreRegistry({bucket: store})
   parser = HDFParser(drop_variables=["mapping"])

   vds = []

   for granule in granules:
      vds.append(
         vz.open_virtual_dataset(
            url=os.path.join(bucket, key, granule),
            parser=parser,
            registry=registry,
            loadable_variables=["time", "y", "x", "v"],
            decode_times=True,
         )
      )
      v_min = np.nanmin(vds[-1]["v"].values)
      v_max = np.nanmax(vds[-1]["v"].values)
      logging.info(f'Values of {granule=} v before combine_by_coords: {v_min=} {v_max=}')


   logging.info(f"{vds}")


   # mosaic both granules onto their common grid and stack along time
   cube, autorift_param_file = build_virtual_cube(vds)

   cube.attrs[Vars.attrs.autorift_param_file] = autorift_param_file

   logging.info(f"\n{cube}")

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
   logging.info("\nDtypes before writing to icechunk:")
   for var_name in cube.data_vars:
      var = cube[var_name]
      dtype = var.dtype
      if isinstance(var.data, ManifestArray):
         manifest_dtype = var.data.dtype
         logging.info(f"  {var_name}: xr.Variable dtype={dtype}, ManifestArray dtype={manifest_dtype}")
      else:
         logging.info(f"  {var_name}: dtype={dtype} (not ManifestArray)")

   cube_clean = _drop_nonfinite_attrs(cube)

   # Write using vz.to_icechunk
   cube_clean.vz.to_icechunk(session.store)

   snapshot_id = session.commit("its_live virtual cube: 2 granules mosaicked + stacked on time")
   logging.info("committed snapshot", snapshot_id)

   # reopen from the committed store
   cube_roundtrip = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
   logging.info(f'{cube_roundtrip=}')
