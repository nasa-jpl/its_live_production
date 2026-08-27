"""
Materialize a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube whose data
variables are physically copied out of the referenced granules, chunked the
same way itscube.py chunks a regular datacube (TIME_CHUNK_VALUE,
X_Y_CHUNK_VALUE, TIME_CHUNK_VALUE_1D).
"""
from datetime import datetime
import logging
import os
import shutil
import sys
import time

import icechunk as ic
import numpy as np
import s3fs
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

import itslive_utils
import utils
from itscube_types import CubeFormat, Vars

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Zarr V3 unstable string dtype warnings for fixed-length UTF32
# dtypes, same rationale as virtual_itslive_cube_per_chunk.py.
import warnings
from zarr.errors import UnstableSpecificationWarning
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Values below match ITSCube's current defaults in itscube.py, kept as local
# constants rather than importing ITSCube: this virtual-cube -> deep-copy
# pipeline is meant to eventually replace itscube.py's regular datacube
# generation entirely, so it should not depend on it.

# Granules are written to the file in chunks to avoid out of memory issues.
# Number of granules to write to the file at a time.
# Default value is optimized for the EC2 instance with 32Gb of RAM.
NUM_GRANULES_TO_WRITE = 2000

# Chunking to apply when writing datacube to the Zarr store for 3-d variables.
# X_Y_CHUNK_VALUE=8 (rather than itscube.py's 10) divides both production
# grid sizes evenly (512px @ 120m -> 64 chunks/side, 256px @ 240m -> 32
# chunks/side, for the 61.44km chunk-aligned catalog), since 512 and 256 are
# powers of 2 -- no partial trailing chunk on either grid. With
# XY_SHARD_MULTIPLIER=4 those per-side chunk counts also divide evenly into
# shards (64/4=16 and 32/4=8 shards/side respectively).
TIME_CHUNK_VALUE = 20000
X_Y_CHUNK_VALUE = 8

# Chunking to apply to 1-D data variables when writing datacube to the
# Zarr store
TIME_CHUNK_VALUE_1D = 200000

# Recommended number of X_Y_CHUNK_VALUE-sized inner chunks to group into one
# shard file, per spatial axis, for sharded 3D (time,y,x) variables. 4 groups
# chunks into 32x32px shards -- on the 61.44km chunk-aligned production grid
# (512px @ 120m / 256px @ 240m) with X_Y_CHUNK_VALUE=8, that divides both
# grids evenly (16 and 8 shards/side, from 64 and 32 chunks/side
# respectively), cutting per-variable object count 16x (4,096 chunks -> 256
# shards on the 512px grid) while keeping worst-case shard rewrite size in
# the tens-of-MB range.
#
# NOT wired as the implicit default for build_encoding()/deep_copy_cube()'s
# xy_shard_multiplier parameter (that defaults to 1, i.e. sharding off), so
# the most basic no-args call stays unsharded. Pass this value explicitly via
# --xy-shard-multiplier to enable sharding.
#
# A shard's extent along 'time' is always exactly one inner chunk's times
# extent -- see build_encoding() -- so this constant only ever affects
# spatial shard size, never how many time-chunks get grouped together
# (grouping across time would force a shard rewrite that touches historical,
# already-finalized time periods on every future append, defeating
# incremental updates).
XY_SHARD_MULTIPLIER = 4

# Compressor for the deep-copy zarr v3 store. Same cname/clevel/shuffle as
# itscube.py's `zarr.Blosc` compressor; that class was removed from
# zarr-python 3.x (this pipeline's zarr version), so zarr.codecs.BloscCodec is
# used here instead. V3 arrays require the plural 'compressors' encoding key
# (a list of codecs) rather than v2's singular 'compressor' key.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')
COMPRESSOR_KEY = 'compressors'

# Variables whose virtual-cube attrs carry no _FillValue/missing_value at all
# -- utils.get_data_var_binary_attr() (the helper that synthesizes these two
# in virtual_itslive_cube.py's build_virtual_cube()) only uses its
# missing_value parameter as a fallback *data* value, never as an attached
# fill attribute -- but itscube.py hardcodes a fill for them regardless (see
# Vars.intMissingValue in itscube_types.py). Applied in build_encoding() only
# when the variable's own attrs have no fill at all, matching itscube.py's
# convention instead of silently omitting the fill.
MISSING_VALUE_OVERRIDES = {
   Vars.ascending_img1: utils.Missing.u8value,
   Vars.ascending_img2: utils.Missing.u8value,
}


def open_virtual_cube(store_path, bucket_prefix):
   """Open a virtual datacube's icechunk repository read-only.

   Parameters
   ----------
   store_path : str
      Path to the icechunk repository (s3:// or local).
   bucket_prefix : str
      S3 URL prefix (e.g. 's3://its-live-data/') that the virtual chunk
      container resolves references against. Read anonymously, matching
      virtual_itslive_cube_per_chunk.py.

   Returns
   -------
   xr.Dataset
      The virtual cube, with data variables backed by ManifestArray chunk
      references (no pixel data loaded yet). The 'time' dimension is left
      as-is -- no renaming to 'mid_date'.
   """
   config = ic.RepositoryConfig.default()
   config.set_virtual_chunk_container(
      ic.VirtualChunkContainer(bucket_prefix, ic.s3_store(region="us-west-2", anonymous=True))
   )
   credentials = ic.containers_credentials(
      {bucket_prefix: ic.s3_credentials(anonymous=True)}
   )

   if store_path.startswith(utils.S3_PREFIX):
      s3_parts = store_path.replace(utils.S3_PREFIX, '').split('/', 1)
      storage = ic.s3_storage(
         bucket=s3_parts[0],
         prefix=s3_parts[1] if len(s3_parts) > 1 else '',
         region="us-west-2"
      )
   else:
      storage = ic.local_filesystem_storage(store_path)

   repo = ic.Repository.open(
      storage=storage,
      config=config,
      authorize_virtual_chunk_access=credentials,
   )

   # mask_and_scale=False preserves the granules' raw on-disk dtypes (int16
   # stays int16 rather than being CF-decoded to float32 with NaN fills), so
   # the deep copy materializes the same dtypes the virtual cube references
   # instead of doubling storage/memory with a float32 promotion. Matches
   # virtual_itslive_cube_per_chunk_update.py's reads.
   #
   # zarr_format=3, not 2: icechunk repositories are natively Zarr V3
   # (zarr.json metadata, no .zgroup/.zarray) -- forcing zarr_format=2 here
   # makes zarr-python look for V2-only markers that don't exist in an
   # icechunk store and fails with GroupNotFoundError. This is unrelated to
   # this pipeline's *output* store format (see deep_copy_cube()'s to_zarr
   # calls below, which correctly write a real, non-icechunk store at
   # zarr_format=2).
   return xr.open_zarr(
      repo.readonly_session("main").store,
      consolidated=False,
      zarr_format=3,
      mask_and_scale=False
   )


def split_vars_by_time(cube):
   """Split the cube's data variables into per-layer ('time'-indexed) and
   static (cube-level, no 'time' dimension) variables.

   Static variables are things like 'mapping', 'landice', 'floatingice' --
   added once at cube creation (see src/wiki/02_Implementation_Details.md,
   "Ice Mask Variables") and never appended to on updates.

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube.

   Returns
   -------
   tuple of (list of str, list of str)
      (time_vars, static_vars) data variable names.
   """
   time_vars = [v for v in cube.data_vars if utils.Coords.TIME in cube[v].dims]
   static_vars = [v for v in cube.data_vars if utils.Coords.TIME not in cube[v].dims]
   return time_vars, static_vars


def build_encoding(
   cube, time_chunk, xy_chunk, time_chunk_1d,
   xy_shard_multiplier=1
):
   """Build the zarr v3 encoding dict for the deep-copy store.

   Sets chunking + compressor per variable, plus the fill value under the
   itscube.py convention (see itscube.py's encoding block, ~lines 2246-2313):
   integer / unsigned-integer variables use the 'missing_value' attribute,
   floating point variables use '_FillValue'. (xarray ignores a requested int
   dtype and assumes float if '_FillValue' is set on an int variable, so ints
   must use 'missing_value'.) dtypes themselves are already correct on the
   virtual cube and are left untouched. Verified (Aug 2026) that a float's
   '_FillValue' masks correctly on read under v3's default
   use_zarr_fill_value_as_mask behavior, even though it's stored as a
   base64-encoded attribute in zarr.json rather than the array-level
   fill_value (a cosmetic xarray-serialization artifact, not a masking bug).

   Chunking follows itscube.py's scheme, except time-chunk sizes are always
   the full fixed value rather than capped at the cube's current layer
   count: a Zarr array's chunk grid is fixed at creation and can't be
   widened on a later append, so capping at however many layers the cube
   happens to have right now would wall in a too-small chunk size forever
   once a cube grows past it via deep_copy_update.py.
   - 3D (time, y, x) variables: (time_chunk, xy_chunk, xy_chunk)
   - 1D (time,) variables and the 'time' coordinate itself: (time_chunk_1d,)
   - static 2D (y, x) variables (landice/floatingice): full extent
   - x/y coordinates: full extent
   - 0-d variables (mapping): no chunk encoding

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube. Opened with mask_and_scale=False, so each
      variable's granule-inherited fill value sits in its attrs (as
      '_FillValue'); this function reads it from there and re-keys it into the
      write encoding. _strip_fill_attrs() then clears it from attrs so the fill
      lives only in the encoding dict (a fill present in both attrs and
      encoding collides in to_zarr's CF encoder).
   time_chunk : int
      Chunk size along 'time' for 3D variables.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables.
   xy_shard_multiplier : int
      Number of xy_chunk-sized inner chunks grouped into one shard per
      spatial axis, for 3D (time,y,x) variables only. Must be >= 1; 1 (the
      default) omits the 'shards' encoding key entirely (unsharded). See
      XY_SHARD_MULTIPLIER for the recommended value to pass explicitly.
      Raises ValueError if < 1.

   Returns
   -------
   dict
      Per-variable/coordinate encoding dict for xr.Dataset.to_zarr().
   """
   if xy_shard_multiplier < 1:
      raise ValueError(
         f"xy_shard_multiplier must be >= 1 (1 disables sharding), "
         f"got {xy_shard_multiplier}"
      )

   encoding = {}

   for coord_name in (utils.Coords.X, utils.Coords.Y):
      if coord_name in cube.coords:
         encoding[coord_name] = {
            'chunks': (cube.sizes[coord_name],),
            COMPRESSOR_KEY: [COMPRESSOR],
            # Suppress xarray's default _FillValue=NaN on these float
            # coordinates (they have no missing values). itscube.py avoids
            # this combination due to an old xarray bug where _FillValue=None
            # alongside 'chunks' encoding broke the write; verified fixed in
            # the xarray/zarr versions this pipeline uses (August 2026).
            # Verified clean (no stray attrs) under zarr v3 too.
            utils.OutputFormat.fill_value: None,
         }

   # The 'time' coordinate needs an explicit chunk size too: left unset, it
   # falls back to zarr's auto-chunker for a dimension that's appended across
   # batches, which came out as chunks=(1,) -- one chunk PER LAYER.
   # xr.open_zarr() eagerly loads dimension coordinates (time/x/y) to build
   # their pandas index on open, so chunks=(1,) meant one S3 GET per layer
   # just to open the cube (measured: ~35,000 GETs, ~167s of open time on a
   # real cube) -- independent of consolidated metadata, which only covers
   # metadata reads, never chunk data. Fixed to time_chunk_1d (not the
   # cube's current total_layers) so it stays a single chunk for as long as
   # possible as the cube grows via later appends -- same reasoning as the
   # other 1D (time,) variables (see the chunking-scheme note above).
   if utils.Coords.TIME in cube.coords:
      encoding[utils.Coords.TIME] = {
         'chunks': (time_chunk_1d,),
         COMPRESSOR_KEY: [COMPRESSOR],
      }

   for var_name in cube.data_vars:
      var = cube[var_name]
      dims = var.dims

      if len(dims) == 0:
         # Scalar variable (e.g. 'mapping'): no chunk encoding needed.
         continue

      is_3d = False
      if utils.Coords.TIME in dims:
         if len(dims) == 3:
            is_3d = True
            chunks = (time_chunk, xy_chunk, xy_chunk)
         else:
            chunks = (time_chunk_1d,)
      else:
         # Static 2D (y, x) variable: full extent, matching itscube.py.
         chunks = tuple(cube.sizes[d] for d in dims)

      var_encoding = {
         'chunks': chunks,
         COMPRESSOR_KEY: [COMPRESSOR],
      }

      if is_3d and xy_shard_multiplier > 1:
         # Shard's 'time' extent is pinned to exactly one inner chunk's time
         # extent -- never group multiple time-chunks into one shard (see
         # XY_SHARD_MULTIPLIER). Only x/y are grouped, by xy_shard_multiplier
         # inner chunks per axis.
         xy_shard_size = xy_chunk * xy_shard_multiplier
         var_encoding['shards'] = (chunks[0], xy_shard_size, xy_shard_size)

         # The XY_SHARD_MULTIPLIER divisibility argument only holds for the
         # chunk-aligned production grids (512px/256px). On any other grid a
         # spatial extent not divisible by the shard size leaves a ragged,
         # partially-filled trailing shard -- valid in Zarr v3, and the write
         # stays correct, but it wastes space and undercuts the object-count
         # reduction sharding is meant to buy. Warn rather than fail.
         for spatial_dim in dims[1:]:
            dim_size = cube.sizes[spatial_dim]
            if dim_size % xy_shard_size:
               logging.warning(
                  f"{var_name}: '{spatial_dim}' size {dim_size} is not a "
                  f"multiple of shard size {xy_shard_size} (xy_chunk="
                  f"{xy_chunk} * xy_shard_multiplier={xy_shard_multiplier}); "
                  f"trailing shard will be partially filled"
               )

      # Re-key the granule-inherited fill (in attrs due to mask_and_scale=False)
      # into the write encoding: 'missing_value' for int/uint, '_FillValue' for
      # float. datetime ('M') / string ('U') variables carry no numeric fill.
      fill = var.attrs.get(
         utils.OutputFormat.fill_value, var.attrs.get(utils.Missing.name)
      )
      if fill is None:
         # No inherited fill at all (e.g. ascending_img1/img2 -- see
         # MISSING_VALUE_OVERRIDES); None here if the variable genuinely has
         # none (itscube.py agrees, e.g. flag_stable_shift).
         fill = MISSING_VALUE_OVERRIDES.get(var_name)
      elif np.isnan(fill):
         # Some granule-native float variables (M11/M12) carry no explicit
         # missing-value fill and default to NaN on the source granule.
         # itscube.py never trusts that and always hardcodes the standard
         # ITS_LIVE numeric fill for these (see itscube.py's "new_v_vars"
         # encoding block, ~lines 2296-2313); match that convention here
         # instead of writing NaN into a freshly materialized store.
         fill = utils.Missing.value

      if fill is not None:
         if var.dtype.kind in ('i', 'u'):
            var_encoding[utils.Missing.name] = var.dtype.type(fill)
         elif var.dtype.kind == 'f':
            var_encoding[utils.OutputFormat.fill_value] = var.dtype.type(fill)

      encoding[var_name] = var_encoding

   return encoding


def _reset_write_encoding(ds):
   """Clear inherited fill attrs and inherited .encoding from every variable
   of a batch in place, so build_encoding()'s explicit dict is the sole
   source of truth for the write.

   Opening the virtual cube with mask_and_scale=False (to preserve int dtypes)
   leaves the granule-inherited fill in each variable's attrs, while the zarr
   backend also carries a fill in encoding; to_zarr's CF encoder then refuses
   to reconcile the two ("Key '_FillValue' already exists in attrs..."). Once
   build_encoding() has captured the fill into the write encoding, clearing
   the attrs copy here makes that encoding the single source of truth. Also
   clears coordinate fills (x/y/time inherit a spurious '_FillValue' too).

   Separately, each variable's own .encoding (populated when the virtual cube
   was opened at zarr_format=3) still carries the full V3 pipeline --
   'serializer', 'compressors', 'filters', 'shards', 'dtype',
   'preferred_chunks', 'fill_value' -- inherited from how the *source*
   granules happen to be chunked/compressed. to_zarr() merges that inherited
   encoding with the explicit `encoding=` dict passed to it, so those stale
   values would otherwise survive the merge and fight with
   build_encoding()'s deliberately-chosen chunking/compression for this
   store. Clearing .encoding entirely removes that leftover pipeline so only
   build_encoding()'s dict applies.

   Parameters
   ----------
   ds : xr.Dataset
      A batch about to be written; mutated in place.
   """
   for var in ds.variables:
      ds[var].attrs.pop(utils.OutputFormat.fill_value, None)
      ds[var].attrs.pop(utils.Missing.name, None)
      ds[var].encoding = {}


def resolve_output_store(output_store):
   """Prepare the output location for a fresh zarr v3 store write.

   For a local path, remove any pre-existing directory first (mirrors
   ITSCube.init_output_store). For an s3:// path, refuse to proceed if
   something already exists there (mirrors ITSCube.exists()) -- unlike the
   local case, there's no single directory to just remove, so silently
   writing over an existing S3 store risks leaving orphaned chunks from a
   differently-shaped previous store. Authenticated write access is expected
   to come from the ambient AWS credential chain (not anonymous, unlike the
   *input* virtual chunk container).

   Parameters
   ----------
   output_store : str
      Local path or s3:// URL for the deep-copy zarr v3 store.

   Returns
   -------
   str
      The (possibly cleaned-up) output store path.
   """
   if output_store.startswith(utils.S3_PREFIX):
      s3_path = output_store.replace(utils.S3_PREFIX, '', 1)
      s3 = s3fs.S3FileSystem()

      if s3.exists(s3_path):
         raise RuntimeError(
            f"Output store {output_store} already exists in S3; refusing "
            "to overwrite. Remove it first if this is intentional."
         )

   elif os.path.exists(output_store):
      logging.info(f"Removing existing {output_store}")
      shutil.rmtree(output_store)

   return output_store


def upload_local_staging_dir(local_staging_dir, output_store, keep_local_staging):
   """Upload a local zarr store to its final S3 destination in one shot, then
   remove the local copy.

   Uses the AWS CLI (via itslive_utils.s3_copy_using_subprocess), matching
   the "write locally, then upload" convention already used elsewhere in this
   codebase (e.g. tools/fix_datacubes_v2_restore_m11_m12_add_new_vars.py) --
   a single recursive `aws s3 cp` uploads each chunk file exactly once,
   instead of the interleaved per-batch S3 writes (and repeated partial
   rewrites of the same small chunks) that direct-to-S3 writing incurs.

   Parameters
   ----------
   local_staging_dir : str
      Local directory the deep-copy store was written to.
   output_store : str
      Final s3:// destination for the deep-copy store.
   keep_local_staging : bool
      If False (default), remove `local_staging_dir` after a successful
      upload.
   """
   logging.info(f'Uploading local staging directory {local_staging_dir} to {output_store}')

   command_line = [
      "aws", "s3", "cp", "--recursive",
      local_staging_dir,
      output_store,
      "--acl", "bucket-owner-full-control"
   ]
   itslive_utils.s3_copy_using_subprocess(command_line, os.environ.copy())

   if keep_local_staging:
      logging.info(f'Keeping local staging directory {local_staging_dir}')
   else:
      logging.info(f'Removing local staging directory {local_staging_dir}')
      shutil.rmtree(local_staging_dir)


def deep_copy_cube(
   input_store,
   output_store,
   bucket_prefix,
   batch_size,
   time_chunk,
   xy_chunk,
   time_chunk_1d,
   xy_shard_multiplier=1,
   local_staging_dir=None,
   keep_local_staging=False
):
   """Materialize a virtual datacube into a real zarr v3 datacube, batched
   along 'time' to bound memory use.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      Path to write the deep-copy zarr store to (s3:// or local).
   bucket_prefix : str
      S3 URL prefix the virtual chunk container resolves granule references
      against (see open_virtual_cube).
   batch_size : int
      Number of layers to materialize and write per batch.
   time_chunk : int
      Chunk size along 'time' for 3D variables.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables.
   xy_shard_multiplier : int
      Must be >= 1; 1 (the default) leaves the store unsharded (see
      build_encoding/XY_SHARD_MULTIPLIER for the recommended value to pass
      explicitly). Raises ValueError if < 1.
   local_staging_dir : str, optional
      If set, `output_store` must be an s3:// path. All batches are written
      to this local directory first, and the whole store is uploaded to
      `output_store` with a single `aws s3 cp --recursive` after the last
      batch, instead of writing every batch directly to S3. Local writes
      avoid per-request network latency and the repeated partial rewrites
      of small zarr chunks that direct-to-S3 batched writes incur.
   keep_local_staging : bool
      If True, keep `local_staging_dir` after a successful upload instead of
      removing it. Ignored if `local_staging_dir` is not set.
   """
   if local_staging_dir and not output_store.startswith(utils.S3_PREFIX):
      raise ValueError(
         "--local-staging-dir only applies when --output-store is an s3:// "
         f"path, got {output_store}"
      )

   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   if total_layers == 0:
      # Should never happen in practice -- virtual cubes are guaranteed to
      # have at least one layer -- but log clearly rather than silently
      # writing nothing if it ever does.
      logging.info(f'{input_store} has no layers, nothing to deep-copy')
      return

   time_vars, static_vars = split_vars_by_time(cube)
   encoding = build_encoding(
      cube, time_chunk, xy_chunk, time_chunk_1d,
      xy_shard_multiplier
   )

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   # Fail fast if the final S3 destination already exists, even when staging
   # locally first -- otherwise that check would only happen after every
   # batch has already been materialized and written locally.
   resolve_output_store(output_store)

   # Batches are written to write_target; local_staging_dir (when set) is
   # cleaned up here the same way resolve_output_store cleans up a local
   # output_store, so a pre-existing staging directory from a failed prior
   # run doesn't get silently merged into.
   write_target = resolve_output_store(local_staging_dir) if local_staging_dir else output_store

   for batch_num, start in enumerate(range(0, total_layers, batch_size)):
      stop = min(start + batch_size, total_layers)
      logging.info(f'Materializing layers {start}:{stop} of {total_layers}')

      batch = cube[time_vars].isel({utils.Coords.TIME: slice(start, stop)})

      if batch_num == 0:
         # Static cube-level variables (landice/floatingice/mapping) are
         # written once, with the first batch: to_zarr's append_dim requires
         # every variable in the dataset to carry the append dimension, so
         # they can't be included in later batches.
         batch = xr.merge([batch, cube[static_vars]])

      batch = batch.load()

      # Clear fill attrs so the fill lives only in the write encoding (see
      # _reset_write_encoding); required on every batch, including appends, since
      # the CF-encoder collision would otherwise fire on each write.
      _reset_write_encoding(batch)

      if batch_num == 0:
         batch.to_zarr(
            write_target,
            mode='w',
            encoding=encoding,
            zarr_format=3,
            consolidated=True
         )
      else:
         batch.to_zarr(
            write_target,
            append_dim=utils.Coords.TIME,
            zarr_format=3,
            consolidated=True
         )

      logging.info(f'Wrote layers {start}:{stop} of {total_layers} to {write_target}')

   # # Consolidate metadata once, after all batches are written, instead of
   # # re-consolidating on every batch (matches itscube.py's convention of a
   # # consolidated output store, without the redundant per-batch cost).
   # zarr.consolidate_metadata(write_target)

   if local_staging_dir:
      upload_local_staging_dir(local_staging_dir, output_store, keep_local_staging)

   logging.info(f'Done: deep-copied {total_layers} layers to {output_store}')


if __name__ == '__main__':
   import argparse

   start_time = time.time()

   parser = argparse.ArgumentParser(
      description="""
      Materialize a virtual ITS_LIVE datacube (icechunk repo built by
      virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube,
      chunked the same way itscube.py chunks a regular datacube.

      Usage example:
      python src/deep_copy_cube.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr

      # Write a store with sharded 3D variables (--xy-shard-multiplier 4 is
      # the recommended production value, see XY_SHARD_MULTIPLIER):
      python src/deep_copy_cube.py \
         --input-store my_virtual_cube.icechunk \
         --output-store my_deep_copy_cube.zarr \
         --xy-shard-multiplier 4

      # Stage locally, then upload the whole store to S3 in one shot
      # (much faster than writing directly to S3 batch-by-batch):
      python src/deep_copy_cube.py \
         --input-store my_virtual_cube.icechunk \
         --output-store s3://its-live-data/path/to/my_deep_copy_cube.zarr \
         --local-staging-dir /local/scratch/my_deep_copy_cube.zarr
      """,
      formatter_class=argparse.RawDescriptionHelpFormatter
   )
   parser.add_argument(
      '--input-store',
      type=str,
      required=True,
      help='Path to the virtual cube icechunk repository (s3:// or local).'
   )
   parser.add_argument(
      '--output-store',
      type=str,
      required=True,
      help='Path to write the deep-copy Zarr v3 store to (s3:// or local).'
   )
   parser.add_argument(
      '--bucket',
      type=str,
      default='s3://its-live-data/',
      help='S3 URL prefix the virtual chunk container resolves granule '
         'references against [%(default)s]'
   )
   parser.add_argument(
      '-b', '--batch-size',
      type=int,
      default=NUM_GRANULES_TO_WRITE,
      help='Number of layers to materialize and write per batch [%(default)d].'
   )
   parser.add_argument(
      '--time-chunk-value',
      type=int,
      default=TIME_CHUNK_VALUE,
      help='Chunk size along time for 3D (time, y, x) variables [%(default)d].'
   )
   parser.add_argument(
      '--xy-chunk-value',
      type=int,
      default=X_Y_CHUNK_VALUE,
      help='Chunk size along x/y for 3D (time, y, x) variables [%(default)d].'
   )
   parser.add_argument(
      '--time-chunk-value-1d',
      type=int,
      default=TIME_CHUNK_VALUE_1D,
      help='Chunk size for 1D (time,) variables [%(default)d].'
   )
   parser.add_argument(
      '--xy-shard-multiplier',
      type=int,
      default=1,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables. A value of 1 '
         f'(the default) disables sharding. Recommended value once sharding '
         f'is enabled: {XY_SHARD_MULTIPLIER} [%(default)d].'
   )
   parser.add_argument(
      '--local-staging-dir',
      type=str,
      default=None,
      help='If set, --output-store must be an s3:// path. Write the '
         'deep-copy store to this local directory first, then upload the '
         'whole store to --output-store with a single "aws s3 cp '
         '--recursive" at the end, instead of writing every batch directly '
         'to S3. Much faster for S3 output, since it avoids per-request '
         'network latency and repeated partial rewrites of small zarr '
         'chunks. Requires enough local disk to hold the full deep-copy '
         'store.'
   )
   parser.add_argument(
      '--keep-local-staging',
      action='store_true',
      help='Keep --local-staging-dir after a successful upload instead of '
         'deleting it [%(default)s].'
   )

   args = parser.parse_args()
   logging.info(f'Command: {sys.argv}')
   logging.info(f'Using command-line arguments: {args}')

   deep_copy_cube(
      args.input_store,
      args.output_store,
      args.bucket,
      args.batch_size,
      args.time_chunk_value,
      args.xy_chunk_value,
      args.time_chunk_value_1d,
      args.xy_shard_multiplier,
      args.local_staging_dir,
      args.keep_local_staging
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
