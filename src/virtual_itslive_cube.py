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
from zarr.core.dtype import parse_data_type
from zarr.codecs import BloscCodec

# Icechunk repo related
import shutil
import icechunk as ic

import utils
from itslive_binary_type import BinaryFlag
from itscube_types import (
   ImgPairInfo,
   Mapping,
   Vars
)

# Suppress Zarr V3 unstable string dtype warnings for fixed-length UTF32 dtypes.
# These are informational - string specs are being finalized in Zarr V3.
# Zarr V3 has no stable spec for the fixed-length UTF32 dtype (<U2, <U3, etc.)
# Must filter by category (the message text is "... does not have a Zarr V3
# specification ...", which never contains the class name).
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Some granules have autoRIFT param. file with 'http://' and some with 'https://',
# strip these to compare the values across the granules (should be the same)
HTTP_PREFIX = 'http://'
HTTPS_PREFIX = 'https://'

# Compressor applied to every newly synthesized 1-D (time,) data variable in
# the virtual cube (img_pair_info-derived attrs, url, etc.) -- these are real
# (non-virtual) arrays written through the normal to_zarr encoding path, so
# unlike the 3D vx/vy/... ManifestArrays (whose bytes are never re-encoded,
# just referenced), they need an explicit compressor or zarr falls back to
# its own default per batch.
NEW_VARS_COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')


def _get_manifestarray_chunks(marr):
   """Get chunk shape from ManifestArray, handling different API versions.

   Parameters
   ----------
   marr : ManifestArray
      The ManifestArray to get chunks from.

   Returns
   -------
   tuple of int
      Chunk shape for each dimension.

   Raises
   ------
   AttributeError
      If chunks cannot be determined from the ManifestArray.
   """
   # Try different API versions
   if hasattr(marr, 'chunks'):
      return marr.chunks
   elif hasattr(marr, 'metadata'):
      if hasattr(marr.metadata, 'chunk_shape'):
         return marr.metadata.chunk_shape
      elif hasattr(marr.metadata, 'chunk_grid'):
         # Zarr v3 chunk grid
         return tuple(marr.metadata.chunk_grid.chunk_shape)

   raise AttributeError(
      f"Cannot determine chunks from ManifestArray. "
      f"Available attributes: {dir(marr)}"
   )


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
   shape = marr.shape
   chunks = _get_manifestarray_chunks(marr)

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


def copy_and_replace_metadata_dtype(old_metadata, new_shape, new_dtype,
                                    new_fill_value, codecs):
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
      to be correct (see `_M_VAR_PLACEHOLDER_CODECS`).
   """
   metadata_dict = old_metadata.to_dict().copy()
   metadata_dict["shape"] = tuple(int(s) for s in new_shape)
   # `data_type` must be the zarr V3 JSON form (a string like "float32" for
   # stable dtypes, or a dict for e.g. fixed-length UTF32), NOT a numpy dtype
   # object. Normalize whatever `new_dtype` is (numpy dtype, string, or JSON)
   # through zarr's parser so callers can pass any of them.
   metadata_dict["data_type"] = parse_data_type(
      new_dtype, zarr_format=3
   ).to_json(zarr_format=3)
   metadata_dict["fill_value"] = new_fill_value  # e.g. float("nan")
   metadata_dict["codecs"] = codecs

   return ArrayV3Metadata.from_dict(metadata_dict)


# Hard-coded codecs for synthesized M11/M12 placeholders (used only for
# granules missing real M11/M12; granules that already have real data are
# left untouched and never go through this path). Captured directly from a
# granule confirmed to have real M11/M12 data, so it's the exact,
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
_M_VAR_PLACEHOLDER_CODECS = (
   {'name': 'bytes', 'configuration': {'endian': 'little'}},
   {'name': 'numcodecs.shuffle', 'configuration': {'elementsize': 4}},
   {'name': 'numcodecs.zlib', 'configuration': {'level': 2}},
)

# Codecs for synthesized vr/va placeholders (used only for
# granules missing real vr/va; granules that already have real data are
# left untouched and never go through this path). Captured directly from a
# radar granule confirmed to have REAL vr/va data, so it's the exact,
# known-correct codec structure for these variables.
#
# IMPORTANT: this is the dict form zarr's ArrayV3Metadata.from_dict()
# expects (i.e. what `.metadata.to_dict()["codecs"]` produces), not the
def _add_missing_m11_m12(new_vars, vds, x_union, y_union):
   """Add M11 and M12 data variables as ManifestArrays with missing chunks
   if not present in the granule.

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
      `_resolve_placeholder_codecs` (see `_M_VAR_PLACEHOLDER_CODECS`).
      Used verbatim for any placeholder synthesized here. Real M11/M12 data
      (when present in `vds`) is untouched and never reaches this codepath.
   """
   # M11 and M12 metadata based on itscube.py and itscube_types.py
   m_vars_info = {
      Vars.m11: {
         Vars.attrs.std_name: Vars.name[Vars.m11],
         Vars.attrs.description: Vars.description[Vars.m11],
         utils.Units.name: utils.Units.pixel_per_m_year
      },
      Vars.m12: {
         Vars.attrs.std_name: Vars.name[Vars.m12],
         Vars.attrs.description: Vars.description[Vars.m12],
         utils.Units.name: utils.Units.pixel_per_m_year
      }
   }

   if Vars.m11 in vds.data_vars:
      # Variable is already in the dataset, nothing to do
      return

   # Use 'v' as template since it's present in all granules (both optical and radar)
   template_var = vds.data_vars[Vars.v]

   # Get dimension indices
   time_idx = template_var.dims.index('time')
   y_idx = template_var.dims.index('y')
   x_idx = template_var.dims.index('x')

   # Get chunk sizes from template
   template_chunks = _get_manifestarray_chunks(template_var.data)
   chunk_time = template_chunks[time_idx]
   chunk_y = template_chunks[y_idx]
   chunk_x = template_chunks[x_idx]
   chunks = (chunk_time, chunk_y, chunk_x)

   for m_var_name, m_var_attrs in m_vars_info.items():
      # Create ManifestArray with missing chunks for optical granules
      # Create 3D shape with time=1 (single time slice)
      shape = (1, len(y_union), len(x_union))

      # Create 3D chunk grid filled with MISSING_CHUNK_PATH (all missing chunks)
      chunk_grid_shape = (
         -(-shape[0] // chunks[0]),  # time chunks
         -(-shape[1] // chunks[1]),  # y chunks
         -(-shape[2] // chunks[2])   # x chunks
      )

      paths = np.full(
         chunk_grid_shape, MISSING_CHUNK_PATH, dtype=np.dtypes.StringDType()
      )
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
      new_metadata = copy_and_replace_metadata_dtype(
         template_var.data.metadata,
         new_shape=list(shape),
         new_dtype="float32",
         new_fill_value=float(utils.Missing.value),
         codecs=_M_VAR_PLACEHOLDER_CODECS,
      )

      manifest_array = ManifestArray(metadata=new_metadata, chunkmanifest=manifest)

      new_vars[m_var_name] = xr.Variable(
         dims=('time', 'y', 'x'),
         data=manifest_array,
         attrs=m_var_attrs,
         # _FillValue belongs in encoding, not attrs, per xarray's CF
         # convention -- attrs is inert metadata, encoding is what
         # actually participates in xarray's encode/decode machinery.
         # encoding={'_FillValue': fill_value},
      )

      logging.debug(
         f"Added {m_var_name} as 3D ManifestArray (time, y, x) with missing "
         "chunks (not present in granule)"
      )

      # Promote the dr_to_vr_factor variable for this synthesized M variable
      # (resolves to the missing value since m_var_name is not in the granule),
      # keeping the promoted variable set identical to real radar granules.
      _extract_m_attributes(new_vars, vds, m_var_name)


def _extract_shared_velocity_attributes(new_vars, vds):
   """Extract the per-granule attributes shared identically across all v*
   variables: flag_stable_shift, stable_count_mask, stable_count_slow.

   Every granule is guaranteed to carry vx and vy (and to have these
   attributes set on them), so read from vx directly -- matching itscube.py's
   combine_layers(), which processes vx first (order [vx, vy] then [va, vr]
   in process_v_attributes()) -- instead of iterating vx/vy/va/vr and
   depending on whichever variable happens to appear first in the granule's
   on-disk NetCDF variable order.

   Call once per granule, not once per velocity variable.

   Parameters
   ----------
   new_vars : dict
      Dictionary to add the shared attribute variables to.
   vds : xr.Dataset
      Virtual dataset for the granule.
   """
   for each_attr, each_attr_units in zip(
      [Vars.flag_stable_shift, Vars.stable_count_mask, Vars.stable_count_slow],
      [None, utils.Units.count, utils.Units.count]
   ):
      attr_value = utils.get_data_var_attr(
         vds, vds.attrs[Vars.url], Vars.vx, each_attr,
         data_dtype=np.int32
      )

      new_vars[each_attr] = xr.Variable(
         dims=(),
         data=np.array(attr_value),
         attrs={
            Vars.attrs.std_name: each_attr,
            Vars.attrs.description: Vars.description[each_attr]
         },
         # Match itscube.py's combine_layers() encoding convention: fixes
         # this variable's on-disk dtype at creation instead of letting zarr
         # infer it (and a default fill value) from whatever raw int32 values
         # happen to appear in a given batch.
         encoding={utils.OutputFormat.dtype: Vars.intType[each_attr]}
      )

      if each_attr_units is not None:
         new_vars[each_attr].attrs[utils.Units.name] = each_attr_units

      logging.debug(f'Extracted shared attribute {each_attr}: {attr_value}')


def _extract_velocity_attributes(new_vars, vds, var_name):
   """Extract all attributes from a velocity variable and create scalar
   variables to represent them in the datacube.

   Extracts the same attributes that itscube.py process_v_attributes() extracts:
   - Error attributes: error, error_mask, error_modeled, error_slow
   - Stable shift attributes: stable_shift, stable_shift_mask, stable_shift_slow
   - Shared attributes (once per granule): flag_stable_shift, stable_count_mask, stable_count_slow

   In the regular datacube, these become 1-D arrays indexed by mid_date. In the
   virtual cube, they're scalar variables per granule (dims=()).

   Parameters
   ----------
   new_vars : dict
      Dictionary to add attribute variables to
   vds : xr.Dataset
      Virtual dataset for the granule
   var_name : str
      Name of the velocity variable (vx, vy, vr, va)
   """
   _name_sep = '_'

   # Error attributes for velocity variables (one per velocity component)
   _v_comp_attrs = [
      Vars.postfix.error,
      Vars.postfix.error_mask,
      Vars.postfix.error_modeled,
      Vars.postfix.error_slow
   ]

   # Extract error attributes
   for each_attr in _v_comp_attrs:
      error_var_name = f'{var_name}{_name_sep}{each_attr}'

      # Get attribute value from the granule if it exists
      if var_name in vds.data_vars and each_attr in vds[var_name].attrs:
         attr_value = vds[var_name].attrs[each_attr]

      else:
         attr_value = utils.Missing.value

      # Get description for the attribute - use canonical Vars.errorAttrs
      # (not per-granule text) to ensure identical descriptions across all
      # granules, matching itscube.py behavior
      desc_str = None

      if each_attr in Vars.errorAttrs:
         # If generic description is provided (e.g., 'error_slow')
         desc_str = Vars.errorAttrs[each_attr][1]

      elif error_var_name in Vars.errorAttrs:
         # If variable specific description is provided (e.g., 'vr_error')
         desc_str = Vars.errorAttrs[error_var_name][1]

      else:
         raise RuntimeError(
            f"Unknown description for {error_var_name} of {var_name}"
         )

      # Create scalar variable for this attribute
      new_vars[error_var_name] = xr.Variable(
         dims=(),
         data=np.array(attr_value),
         attrs={
            utils.Units.name: utils.Units.m_y,
            Vars.attrs.std_name: error_var_name,
            Vars.attrs.description: desc_str
         },
         # Match itscube.py's combine_layers() encoding convention (its
         # new_v_vars default branch: float32 with _FillValue=Missing.value)
         # so this variable's dtype/fill is fixed at creation rather than
         # flipping between int64/float64 depending on whether this batch's
         # granules had a real value or fell back to the missing sentinel.
         encoding={
            utils.OutputFormat.fill_value: utils.Missing.value,
            utils.OutputFormat.dtype: np.float32
         }
      )

      logging.debug(f'Extracted {error_var_name}: {attr_value}')

   # Extract stable_shift (specific to each velocity variable).
   # Read via get_data_var_attr so the value is coerced to float32 (and any
   # array-wrapped NISAR attribute is flattened to a scalar) before the NaN
   # check, matching itscube.py process_v_attributes(). Reading the raw attr
   # here could hand np.isnan a non-float or a length-1 array and crash.
   shift_var_name = _name_sep.join([var_name, Vars.postfix.stable_shift])

   stable_shift_value = utils.get_data_var_attr(
      vds, vds.attrs[Vars.url], var_name, Vars.postfix.stable_shift,
      utils.Missing.value
   )

   # Some granules have "stable_shift" attribute set to NaN: set it to zero
   if np.isnan(stable_shift_value):
      logging.info(f'Setting NaN stable_shift to 0 for {var_name}')
      stable_shift_value = np.float32(0)

   _desc_str = f'applied {var_name} shift calibrated using pixels over stable or slow surfaces'
   new_vars[shift_var_name] = xr.Variable(
      dims=(),
      data=np.array(stable_shift_value),
      attrs={
         utils.Units.name: utils.Units.m_y,
         Vars.attrs.std_name: shift_var_name,
         Vars.attrs.description: _desc_str
      },
      encoding={
         utils.OutputFormat.fill_value: utils.Missing.value,
         utils.OutputFormat.dtype: np.float32
      }
   )

   logging.debug(f'Extracted {shift_var_name}: {stable_shift_value}')

   # Extract stable_shift_mask and stable_shift_slow
   for each_attr in [Vars.postfix.stable_shift_mask, Vars.postfix.stable_shift_slow]:
      shift_var_name = _name_sep.join([var_name, each_attr])

      if var_name in vds.data_vars and each_attr in vds[var_name].attrs:
         attr_value = vds[var_name].attrs[each_attr]
      else:
         attr_value = utils.Missing.value

      _desc_str = Vars.description[each_attr].format(var_name)
      new_vars[shift_var_name] = xr.Variable(
         dims=(),
         data=np.array(attr_value),
         attrs={
            utils.Units.name: utils.Units.m_y,
            Vars.attrs.std_name: shift_var_name,
            Vars.attrs.description: _desc_str
         },
         encoding={
            utils.OutputFormat.fill_value: utils.Missing.value,
            utils.OutputFormat.dtype: np.float32
         }
      )

      logging.debug(f'Extracted {shift_var_name}: {attr_value}')


def _extract_m_attributes(new_vars, vds, var_name):
   """Promote an M11/M12 variable's dr_to_vr_factor attribute into its own
   scalar variable, mirroring itscube.py process_m_attributes().

   Works for both real M11/M12 (reads the attribute) and synthesized
   placeholders (var_name not in vds.data_vars -> uses the missing value),
   so the promoted variable set stays identical across radar and optical
   granules and build_virtual_cube()'s variable-set check does not fail on
   mixed cubes.

   NOTE: unlike the regular datacube, scale_factor/add_offset are intentionally
   NOT stripped from M11/M12 here. Virtual cubes reference the raw, packed
   on-disk chunks, so those encoding keys are required for xarray to decode the
   referenced data correctly on read.

   Parameters
   ----------
   new_vars : dict
      Dictionary to add the dr_to_vr_factor variable to
   vds : xr.Dataset
      Virtual dataset for the granule
   var_name : str
      Name of the M variable (M11, M12)
   """
   attr_name = f'{var_name}_{Vars.postfix.dr_to_vr_factor}'

   # Missing.byte matches itscube.py process_m_attributes() default; absent for
   # synthesized placeholders (optical granules).
   attr_value = utils.get_data_var_attr(
      vds, vds.attrs[Vars.url], var_name, Vars.postfix.dr_to_vr_factor,
      utils.Missing.byte
   )

   new_vars[attr_name] = xr.Variable(
      dims=(),
      data=np.array(attr_value),
      attrs={
         Vars.attrs.std_name: attr_name,
         Vars.attrs.description: Vars.description[Vars.postfix.dr_to_vr_factor],
         utils.Units.name: utils.Units.m_per_year_pixel
      },
      # Match itscube.py's combine_layers() encoding convention (its
      # new_vars_zero_missing_value branch: float32 with _FillValue=Missing.byte=0.0).
      encoding={
         utils.OutputFormat.dtype: np.float32,
         utils.OutputFormat.fill_value: utils.Missing.byte
      }
   )

   logging.debug(f'Extracted {attr_name}: {attr_value}')


def _add_missing_vr_va(new_vars, vds, x_union, y_union):
   """Add vr and va data variables as ManifestArrays with missing chunks if not
   present in the granule.

   vr (range velocity) and va (azimuth velocity) are radar-specific variables.
   Optical granules don't have these variables, so we create ManifestArrays with
   all-missing chunks that read as fill value. This ensures consistency when
   combining optical and radar granules.

   Parameters
   ----------
   new_vars : dict
      Dictionary to add vr/va variables to
   vds : xr.Dataset
      Virtual dataset for the granule
   x_union : np.ndarray
      Union x coordinates for the datacube
   y_union : np.ndarray
      Union y coordinates for the datacube
   """
   # vr and va metadata based on itscube_types.py
   radar_vars_info = {
      Vars.vr: {
         Vars.attrs.std_name: Vars.name[Vars.vr],
         Vars.attrs.description: Vars.description[Vars.vr],
         utils.Units.name: utils.Units.m_y
      },
      Vars.va: {
         Vars.attrs.std_name: Vars.name[Vars.va],
         Vars.attrs.description: Vars.description[Vars.va],
         utils.Units.name: utils.Units.m_y
      }
   }

   if Vars.vr in vds.data_vars and Vars.va in vds.data_vars:
      # Variables are already in the dataset, nothing to do
      return

   logging.debug(f'Adding va and vr to the {vds.attrs['granule_url']}...')

   # Use 'v' as template since it's present in all granules (both optical and radar)
   template_var = vds.data_vars[Vars.v]

   # Get dimension indices
   time_idx = template_var.dims.index('time')
   y_idx = template_var.dims.index('y')
   x_idx = template_var.dims.index('x')

   # Get chunk sizes from template
   template_chunks = _get_manifestarray_chunks(template_var.data)
   chunk_time = template_chunks[time_idx]
   chunk_y = template_chunks[y_idx]
   chunk_x = template_chunks[x_idx]
   chunks = (chunk_time, chunk_y, chunk_x)

   for var_name, var_attrs in radar_vars_info.items():
      # Create ManifestArray with missing chunks for optical granules
      # Create 3D shape with time=1 (single time slice)
      shape = (1, len(y_union), len(x_union))

      # Create 3D chunk grid filled with MISSING_CHUNK_PATH
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

      # vr and va use int16, same as v, so reuse v's metadata directly
      manifest_array = ManifestArray(metadata=template_var.data.metadata, chunkmanifest=manifest)

      new_vars[var_name] = xr.Variable(
         dims=('time', 'y', 'x'),
         data=manifest_array,
         attrs=var_attrs,
      )

      logging.debug(f"Added {var_name} as 3D ManifestArray (time, y, x) with missing chunks (not present in granule)")

      # Promote the FULL set of velocity attribute variables for this
      # synthesized component (error/error_mask/error_modeled/error_slow,
      # stable_shift/_mask/_slow, plus the shared flag/count attributes),
      # exactly as a real radar granule would via _extract_velocity_attributes.
      # Since var_name is not in vds.data_vars, every attribute resolves to its
      # missing value. This keeps the promoted variable set identical between
      # optical and radar granules so build_virtual_cube()'s variable-set check
      # (and combine_by_coords) does not fail on mixed cubes. The shared
      # attributes were already added while processing vx/vy, so
      # _extract_velocity_attributes skips re-adding them.
      _extract_velocity_attributes(new_vars, vds, var_name)


def _extra_var_cf_attrs(name):
   """CF attributes to stamp on the extra 3D data variables (v_error,
   chip_size_height/width, interp_mask) so the virtual cube's metadata matches
   the regular datacube's normalized output (see itscube.py combine_layers).
   These are merged ON TOP of the granule's own attrs -- the granule attrs are
   kept because they carry the encoding needed to decode the referenced chunks.

   Returns {} for any other variable (so merging is a no-op for v/vx/vy/...).

   NOTE (chip_size_height): the regular cube substitutes chip_size_width per
   pixel where chip_size_height == CHIP_SIZE_HEIGHT_NO_VALUE. That fallback
   requires reading pixel data and cannot be replicated on a ManifestArray, so
   the virtual cube references the granule's chip_size_height as-is.
   """
   if name == Vars.v_error:
      return {
         Vars.attrs.std_name: Vars.name[Vars.v_error],
         Vars.attrs.description: Vars.description[Vars.v_error],
         utils.Units.name: utils.Units.m_y,
         Mapping.attrs.grid_mapping: Mapping.name,
      }

   if name in (Vars.chip_size_height, Vars.chip_size_width):
      return {
         Vars.attrs.chip_size_coords: Vars.description[Vars.attrs.chip_size_coords],
         Vars.attrs.description: Vars.description[name],
         Mapping.attrs.grid_mapping: Mapping.name,
      }

   if name == Vars.interp_mask:
      return {
         Vars.attrs.std_name: Vars.name[Vars.interp_mask],
         Vars.attrs.description: Vars.description[Vars.interp_mask],
         BinaryFlag.attrs.values: BinaryFlag.values,
         BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.interp_mask],
         Mapping.attrs.grid_mapping: Mapping.name,
      }

   return {}


def build_virtual_cube(vds_list, already_aligned=False):
   """Mosaic virtual datasets onto their common x/y grid and stack along time.

   Three steps:
   1. ``extend_coords``    -> union x/y grid + each granule's offset into it
                              (skipped if already_aligned=True)
   2. ``pad_manifestarray``-> drop each granule's data at its offset on the
                              union grid (nodata everywhere else)
                              (no-op if already_aligned=True, offsets are all zeros)
   3. ``combine_by_coords``-> order/stack the granules along time

   Data variables stay virtual (ManifestArray); x/y/time are real, indexed
   coordinates. No pixel data is read.

   Parameters
   ----------
   vds_list : list of xr.Dataset
      Virtual datasets to combine. Each must have x, y, time coordinates.
   already_aligned : bool, optional
      If True, assumes all datasets in vds_list are already on identical x/y
      grids (same coordinates, same extents). Skips extend_coords() and uses
      the first dataset's x/y directly. Default is False.
      Use True when granules have been pre-cropped to identical grids (e.g.,
      via crop_virtual_dataset_to_bbox in virtual_itslive_cube_per_chunk.py).

   Returns
   -------
   tuple of (xr.Dataset, str)
      The combined virtual datacube and the autorift parameter file path.
   """
   if already_aligned:
      # All granules are on identical grids - use first granule's x/y
      # and set all offsets to zero
      x_union = vds_list[0]["x"].values
      y_union = vds_list[0]["y"].values
      offsets = [{"x": 0, "y": 0}] * len(vds_list)
      logging.info(
         f"Using already-aligned grids: x={len(x_union)}, y={len(y_union)}"
      )

   else:
      # Compute union grid and offsets
      x_union, y_union, offsets = extend_coords(vds_list)

   sizes = {"x": len(x_union), "y": len(y_union)}

   # Collect only a subset of data variables in the virtual datacube.
   # v_error/chip_size_height/chip_size_width/interp_mask are present in every
   # granule, so they just need to be kept (and their CF attrs normalized via
   # _extra_var_cf_attrs) -- no missing-chunk placeholder is required.
   _vars = [
      ImgPairInfo.name, Vars.v, Vars.vx, Vars.vy, Vars.vr, Vars.va,
      Vars.m11, Vars.m12, Vars.v_error, Vars.chip_size_height,
      Vars.chip_size_width, Vars.interp_mask
   ]

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

               value = utils.get_data_var_attr(
                  vds, vds.attrs[Vars.url], ImgPairInfo.name, attr,
                  to_date=convert_to_date, data_dtype=attr_dtype
               )
               if isinstance(value, str):
                  # Use fixed-length string dtype if defined, otherwise variable-length
                  value = np.array(value, dtype=ImgPairInfo.stringType.get(attr, np.dtypes.StringDType()))

               # Match itscube.py's combine_layers() encoding convention (for
               # non-date attrs) so this variable's on-disk dtype/fill is
               # fixed at creation instead of being inferred per-batch from
               # whatever raw values happen to appear (see itscube.py's
               # encoding_settings).
               new_var_encoding = {}
               if attr_dtype is not None:
                  new_var_encoding[utils.OutputFormat.dtype] = attr_dtype
               if convert_to_date:
                  # Explicit units/calendar/dtype instead of itscube.py's
                  # bare 'days since 1970-01-01' (no dtype): these values
                  # carry a sub-day time-of-day component (acquisition
                  # timestamps and their midpoint), which 'days' units can't
                  # represent as int64 -- and since a Zarr array's CF
                  # encoding is fixed at creation and reused by every later
                  # append (see set_1d_time_chunk_encoding's docstring in
                  # virtual_itslive_cube_per_chunk.py), leaving dtype
                  # unset would let whichever batch creates the store
                  # silently decide int64 vs. float64, and any later batch
                  # that doesn't fit would hit xarray's
                  # "Times can't be serialized faithfully to int64..."
                  # UserWarning on every append. float64 seconds since the
                  # GPS epoch (matches the cube's 'time' coordinate) avoids
                  # that fallback entirely.
                  new_var_encoding[utils.Units.name] = utils.Units.gps_epoch_date
                  new_var_encoding[utils.Units.calendar_name] = utils.Units.proleptic_gregorian
                  new_var_encoding[utils.OutputFormat.dtype] = 'float64'

               new_vars[attr] = xr.Variable(
                  dims=(),
                  data=value,
                  attrs=new_var_attrs,
                  encoding=new_var_encoding
               )

            for (each, new_each) in zip(
                  [
                     ImgPairInfo.flight_direction_img1,
                     ImgPairInfo.flight_direction_img2
                  ],
                  [
                     Vars.ascending_img1,
                     Vars.ascending_img2
                  ]
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
                     },
                     # Match itscube.py's combine_layers() encoding convention
                     # (Vars.intType/Vars.intMissingValue: uint8, missing_value=255)
                     # so this stays uint8 instead of silently widening to
                     # int64 when a batch's fallback (raw Python int 255) mixes
                     # with the real uint8 values from get_data_var_binary_attr.
                     encoding={
                        utils.OutputFormat.dtype: np.uint8,
                        utils.Missing.name: utils.Missing.u8value
                     }
                  )

            continue  # Skip adding img_pair_info itself to the cube

         # Add autoRIFT_software_version (only if it was not added already)
         if Vars.autorift_software_version not in new_vars:
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

         # Process velocity data variables and their attributes
         if name in [Vars.vx, Vars.vy, Vars.vr, Vars.va]:
            # Extract velocity attributes (error, error_mask, error_modeled,
            # error_slow, stable_shift, stable_shift_mask, stable_shift_slow)
            # to match "deep copy" datacube behavior
            # (see itscube.py process_v_attributes()).
            # Only process if variable exists in original granule
            # (not synthetic placeholder)
            if name in vds.data_vars:
               _extract_velocity_attributes(new_vars, vds, name)

            # Shared attributes (flag_stable_shift, stable_count_mask,
            # stable_count_slow) are identical across vx/vy/vr/va -- read
            # once from vx, deterministically, rather than from whichever of
            # vx/vy/vr/va this loop happens to visit first (see
            # _extract_shared_velocity_attributes docstring).
            if name == Vars.vx:
               _extract_shared_velocity_attributes(new_vars, vds)

         # Promote M11/M12 dr_to_vr_factor attribute into its own variable to
         # match itscube.py process_m_attributes(). Real M11/M12 reach this path
         # (they're in _vars); optical granules that lack M11/M12 get the same
         # variable synthesized in _add_missing_m11_m12().
         if name in [Vars.m11, Vars.m12]:
            _extract_m_attributes(new_vars, vds, name)

         data = var.data
         if isinstance(data, ManifestArray):
               new_shape = [sizes.get(str(d), s) for d, s in zip(var.dims, data.shape)]
               var_offsets = [off.get(str(d), 0) for d in var.dims]

               # Skip the manifest rebuild when the array already fills the
               # target grid (already_aligned granules: shape unchanged and all
               # offsets zero). pad_manifestarray would just reconstruct a
               # byte-identical manifest, so reuse `data` as-is.
               if new_shape != list(data.shape) or any(var_offsets):
                  data = pad_manifestarray(data, new_shape, var_offsets)

         # Create variable with original dtype preserved. Merge normalized CF
         # attrs for the extra 3D vars on top of the granule attrs (no-op for
         # everything else).
         var_attrs = dict(var.attrs)
         var_attrs.update(_extra_var_cf_attrs(name))
         new_vars[name] = xr.Variable(
            var.dims, data, attrs=var_attrs, encoding=var.encoding
         )

      # Add source granule url for the layer in the cube
      # Use fixed-length string dtype to prevent truncation issues during updates
      new_vars[Vars.url] = xr.Variable(
         dims=("time",),
         data=np.array([vds.attrs[Vars.url]],
                        dtype=Vars.stringType[Vars.url]),
         attrs={"description": "source granule URL for this time step"},
      )

      # Add M11 and M12 if not present in granule
      # (optical granules don't have these)
      _add_missing_m11_m12(new_vars, vds, x_union, y_union)

      # Add vr and va if not present in granule
      # (optical granules don't have these radar-specific variables)
      _add_missing_vr_va(new_vars, vds, x_union, y_union)

      placed.append(xr.Dataset(
         new_vars,
         coords={"x": ("x", x_union), "y": ("y", y_union), "time": vds["time"]},
         attrs=vds.attrs,
      ))

   var_sets = [frozenset(ds.data_vars) for ds in placed]
   if len(set(var_sets)) > 1:
      from collections import Counter
      counts = Counter(var_sets)

      # find the minority set(s) and report which granule(s) have them
      majority = counts.most_common(1)[0][0]
      for ds, vs in zip(placed, var_sets):
         if vs != majority:
            raise ValueError(
               f"{ds.attrs.get(Vars.url, '<unknown>')}: variable set differs "
               f"from majority.\n  extra: {vs - majority}\n  missing: {majority - vs}"
            )

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

   # Synthesize the CF grid-mapping variable once for the whole cube (added to
   # the combined result, not per granule, to avoid combine_by_coords conflicts).
   # The HDF parser no longer drops 'mapping', so its projection attrs travel
   # with each (loaded, 0-dim) granule variable; copy them from the first
   # granule that has it and override GeoTransform with one computed for this
   # tile's grid (the granule's own GeoTransform describes the full granule, not
   # this tile). Mirrors itscube.py combine_layers().
   mapping_source = next(
      (each for each in vds_list if Mapping.name in each.variables), None
   )
   if mapping_source is not None:
      x_cell = float(x_union[1] - x_union[0])
      y_cell = float(y_union[1] - y_union[0])
      geo_transform = (
         f"{x_union[0] - x_cell / 2.0} {x_cell} 0 "
         f"{y_union[0] - y_cell / 2.0} 0 {y_cell}"
      )
      mapping_attrs = dict(mapping_source[Mapping.name].attrs)
      mapping_attrs[Mapping.attrs.geo_transform] = geo_transform
      result[Mapping.name] = xr.DataArray(
         data='', attrs=mapping_attrs, coords={}, dims=[]
      )

   # Apply the shared compressor to every newly synthesized 1-D (time,)
   # variable (img_pair_info-derived attrs, url, ascending_img1/2, etc.) and
   # to the 'time' coordinate itself -- it's just as real/non-virtual an
   # array as those variables, so iterate result.variables (data_vars +
   # coords), not just result.data_vars. Excludes the 3D vx/vy/vr/va/m11/m12
   # ManifestArrays (dims include y/x, and their bytes are virtual
   # references, never re-encoded through this path) and scalar vars like
   # 'mapping' (dims=()).
   for var_name in result.variables:
      if result[var_name].dims == (utils.Coords.TIME,):
         result[var_name].encoding[utils.OutputFormat.compressors] = [NEW_VARS_COMPRESSOR]

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
   parser = HDFParser()

   vds = []

   for granule in granules:
      vds.append(
         vz.open_virtual_dataset(
            url=os.path.join(bucket, key, granule),
            parser=parser,
            registry=registry,
            loadable_variables=["time", "y", "x", "v", Mapping.name],
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
   # zarr_format=3, not 2: icechunk repos are natively Zarr V3 metadata;
   # forcing zarr_format=2 raises GroupNotFoundError.
   cube_roundtrip = xr.open_zarr(repo.readonly_session("main").store, consolidated=False, zarr_format=3)
   logging.info(f'{cube_roundtrip=}')
