"""
Materialize a virtual ITS_LIVE datacube (icechunk repo, built by
virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube whose data
variables and correspondign attributes are physically copied out of the
referenced granules, and chunked the same way we used to chunk Zarr v2
datacubes (TIME_CHUNK_VALUE, X_Y_CHUNK_VALUE, TIME_CHUNK_VALUE_1D).

This is a a new pipeline that replaces itscube.py's Zarr v2 datacube
generation.
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
from itscube_types import CubeFormat, ImgPairInfo, Vars

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Zarr V3 unstable string dtype warnings for fixed-length UTF32
# dtypes, same rationale as virtual_itslive_cube_per_chunk.py.
import warnings
from zarr.errors import UnstableSpecificationWarning, ZarrUserWarning
warnings.filterwarnings('ignore', category=UnstableSpecificationWarning)

# Consolidated metadata isn't part of the Zarr V3 spec yet, but every store
# here is written zarr_format=3, consolidated=True deliberately (fast
# single-request opens); the warning fires on every consolidate/open and
# adds nothing actionable.
warnings.filterwarnings('ignore', category=ZarrUserWarning, message='Consolidated metadata')

# Default batch size, tuned for a 32GB-RAM EC2 instance.
NUM_GRANULES_TO_WRITE = 2000

# X_Y_CHUNK_VALUE=8 (vs itscube.py's 10): divides production grid size
# of 512px@120m evenly with no ragged trailing chunk, and those
# per-side chunk counts also divide evenly into XY_SHARD_MULTIPLIER shards.
TIME_CHUNK_VALUE = 20000

# Also imported by virtual_itslive_cube_per_chunk.py, so the virtual-cube
# writer and this deep-copy pipeline can't drift out of sync on the 1-D
# 'time' chunk size (see that module's set_1d_time_chunk_encoding()).
TIME_CHUNK_VALUE_1D = 200000
X_Y_CHUNK_VALUE = 8


# Recommended xy_shard_multiplier for 3D variables: 64x64px shards, dividing
# 120 production grid evenly, cutting object count 64x per variable. The
# --xy-shard-multiplier CLI flag defaults to this; build_encoding()/
# deep_copy_cube() themselves still default to 1 (off) for direct callers
# that don't pass it explicitly. A shard's 'time' extent is always exactly
# one chunk (never grouped across time), since grouping across time would
# force rewriting historical, already-finalized shards on every future
# append.
XY_SHARD_MULTIPLIER = 8

# zarr.Blosc (itscube.py's compressor) was removed in zarr-python 3.x; same
# cname/clevel/shuffle via BloscCodec instead. V3 uses the plural
# 'compressors' key (a list), not v2's singular 'compressor'.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')
COMPRESSOR_KEY = 'compressors'

# Variables whose virtual-cube attrs carry no fill at all, but which
# itscube.py hardcodes a fill for regardless (see Vars.intMissingValue).
MISSING_VALUE_OVERRIDES = {
   Vars.ascending_img1: utils.Missing.u8value,
   Vars.ascending_img2: utils.Missing.u8value,
}

# Variables that encode with dtype only, no fill at all -- every
# granule always carries a real value for these, so a fill would be
# meaningless. build_encoding() must override any fill these inherit from
# the virtual cube's attrs/encoding (e.g. a granule-native _FillValue=NaN).
NO_FILL_VARS = {
   ImgPairInfo.date_dt,
   ImgPairInfo.roi_valid_percentage,
   Vars.flag_stable_shift,
   Vars.stable_count_slow,
   Vars.stable_count_mask,
}


def split_time_vars_by_rank(cube, time_vars):
   """Split into 3D (time,y,x) and 1D (time,) groups -- callers take
   different write paths per rank (raw zarr writes vs to_zarr(region=...)
   for CF datetime/string encoding) and use different time-chunk sizes
   (time_chunk vs time_chunk_1d).

   Args:
      cube (xr.Dataset): the virtual datacube time_vars belongs to.
      time_vars (list of str): variable names, from split_vars_by_time().
   """
   vars_3d = [v for v in time_vars if len(cube[v].dims) == 3]
   vars_1d = [v for v in time_vars if len(cube[v].dims) != 3]

   return vars_3d, vars_1d


@itslive_utils.retry_decorator(max_retries=5)
def open_virtual_cube(store_path, bucket_prefix):
   """Open an icechunk repository read-only, resolving its virtual chunk
   references.

   retry_decorator: unlike the granule-reading obstore.S3Store elsewhere in
   this pipeline, icechunk's own s3_store() has no built-in retry/backoff,
   so a transient blip resolving repo/manifest metadata would otherwise fail
   outright.

   Args:
      store_path (str): icechunk repository path (s3:// or local).
      bucket_prefix (str): S3 URL prefix the virtual chunk container
         resolves granule references against.
   """
   config = ic.RepositoryConfig.default()
   config.set_virtual_chunk_container(
      ic.VirtualChunkContainer(
         bucket_prefix, ic.s3_store(region="us-west-2", anonymous=True)
      )
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

   # mask_and_scale=False preserves granules' raw dtypes (int16 stays int16
   # instead of being CF-decoded to float32+NaN), avoiding a storage-doubling
   # promotion. zarr_format=3 because icechunk repos are natively V3 --
   # forcing 2 makes zarr-python look for nonexistent V2 markers and raises
   # GroupNotFoundError.
   return xr.open_zarr(
      repo.readonly_session("main").store,
      consolidated=False,
      zarr_format=3,
      mask_and_scale=False
   )


def split_vars_by_time(cube):
   """Split data variables into per-layer ('time'-indexed) vs static
   (cube-level, e.g. 'mapping'/'landice'/'floatingice' -- added once at
   creation, never appended to).

   Args:
      cube (xr.Dataset): the virtual datacube.

   Returns:
      tuple of (list of str, list of str): (time_vars, static_vars).
   """
   time_vars = [v for v in cube.data_vars if utils.Coords.TIME in cube[v].dims]
   static_vars = [v for v in cube.data_vars if utils.Coords.TIME not in cube[v].dims]

   return time_vars, static_vars


def build_encoding(
   cube, time_chunk, xy_chunk, time_chunk_1d, xy_shard_multiplier=1
):
   """Build the zarr v3 encoding dict for `cube`'s deep-copy store: chunking +
   compressor per variable, plus fill value under itscube.py's convention
   (int/uint -> 'missing_value', float -> '_FillValue' -- xarray assumes
   float if '_FillValue' is set on an int variable). Verified a float
   '_FillValue' still masks correctly on read despite being a base64
   zarr.json attribute rather than the array-level fill_value.

   Time-chunk sizes are always the full fixed value, never capped to the
   cube's current layer count: a zarr chunk grid is fixed at creation and
   can't widen on a later append, so capping now would wall in a too-small
   chunk size forever once the cube grows via deep_copy_update.py.

   Args:
      cube (xr.Dataset): the virtual datacube.
      time_chunk (int): 'time' chunk size for 3D (time,y,x) variables.
      xy_chunk (int): 'x'/'y' chunk size for 3D variables.
      time_chunk_1d (int): 'time' chunk size for 1D (time,) variables and
         the 'time' coordinate itself.
      xy_shard_multiplier (int): must be >= 1; 1 (default) omits 'shards'
         entirely (unsharded) -- see XY_SHARD_MULTIPLIER for the
         recommended explicit value.

   Returns: encoding dictionary.
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
            # Suppress xarray's default _FillValue=NaN (no missing values on
            # these coords) and the separate zarr-level array fill_value
            # (zarr v3 requires one; null it explicitly rather than take
            # zarr's dtype default).
            utils.OutputFormat.fill_value: None,
            utils.Missing.fill_value: None,
         }

   # 'time' needs an explicit chunk size: left unset, zarr's auto-chunker
   # picks chunks=(1,) for an appended dimension -- one S3 GET per layer just
   # to open the cube (measured: ~35,000 GETs, ~167s on a real cube), since
   # xr.open_zarr() eagerly loads dimension coordinates. time_chunk_1d (not
   # the cube's current total_layers) keeps it a single chunk as long as
   # possible across later appends.
   if utils.Coords.TIME in cube.coords:
      encoding[utils.Coords.TIME] = {
         'chunks': (time_chunk_1d,),
         COMPRESSOR_KEY: [COMPRESSOR],
         utils.Missing.fill_value: None,
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
         # Placeholder zarr-level fill_value; overridden below wherever a
         # real CF fill is computed, so it's a meaningful sentinel rather
         # than zarr's arbitrary dtype-zero default.
         utils.Missing.fill_value: None,
      }

      if is_3d and xy_shard_multiplier > 1:
         # Shard's 'time' extent is always one inner chunk -- see
         # XY_SHARD_MULTIPLIER; only x/y get grouped.
         xy_shard_size = xy_chunk * xy_shard_multiplier
         var_encoding['shards'] = (chunks[0], xy_shard_size, xy_shard_size)

         # Divides evenly only on the chunk-aligned production grids; warn
         # (don't fail) on a ragged trailing shard elsewhere.
         for spatial_dim in dims[1:]:
            dim_size = cube.sizes[spatial_dim]
            if dim_size % xy_shard_size:
               logging.warning(
                  f"{var_name}: '{spatial_dim}' size {dim_size} is not a "
                  f"multiple of shard size {xy_shard_size} (xy_chunk="
                  f"{xy_chunk} * xy_shard_multiplier={xy_shard_multiplier}); "
                  f"trailing shard will be partially filled"
               )

      if var_name in NO_FILL_VARS:
         # Float dtypes need an explicit fill_value=None override -- an
         # absent key still lets xarray/zarr re-inject _FillValue=NaN on
         # write (verified). Int/uint don't get that default.
         if var.dtype.kind == 'f':
            var_encoding[utils.OutputFormat.fill_value] = None
         encoding[var_name] = var_encoding
         continue

      # Re-key the granule-inherited fill (in attrs, since mask_and_scale=False)
      # into the write encoding: 'missing_value' for int/uint, '_FillValue'
      # for float.
      fill = var.attrs.get(
         utils.OutputFormat.fill_value, var.attrs.get(utils.Missing.name)
      )
      if fill is None and var.dtype.kind in ('i', 'u', 'f'):
         # M11/M12: no CF fill attribute, but the zarr-level array fill
         # still surfaces as var.encoding['fill_value'] (verified). Numeric
         # dtypes only -- a string var's encoding fill is '', which would
         # crash the np.isnan() check below.
         fill = var.encoding.get(utils.Missing.fill_value)
      if fill is None:
         # Genuinely no fill anywhere (e.g. ascending_img1/img2 -- see
         # MISSING_VALUE_OVERRIDES).
         fill = MISSING_VALUE_OVERRIDES.get(var_name)
      elif np.isnan(fill):
         # M11/M12-style variables default to NaN on the source granule;
         # itscube.py always hardcodes the standard ITS_LIVE fill instead.
         fill = utils.Missing.value

      if fill is not None:
         # Mirror into the zarr-level array fill_value too (a real sentinel,
         # not zarr's arbitrary dtype-zero default).
         if var.dtype.kind in ('i', 'u'):
            var_encoding[utils.Missing.name] = var.dtype.type(fill)
            var_encoding[utils.Missing.fill_value] = var.dtype.type(fill)
         elif var.dtype.kind == 'f':
            var_encoding[utils.OutputFormat.fill_value] = var.dtype.type(fill)
            var_encoding[utils.Missing.fill_value] = var.dtype.type(fill)

      encoding[var_name] = var_encoding

   return encoding


def _reset_write_encoding(ds):
   """Clear each variable's inherited fill attrs and .encoding in place, so
   build_encoding()'s explicit dict is the sole source of truth.

   mask_and_scale=False leaves a granule-inherited fill in attrs while zarr
   also carries one in .encoding; to_zarr's CF encoder refuses to reconcile
   both ("Key '_FillValue' already exists in attrs..."). Each variable's
   .encoding also still carries the *source* granules' chunking/compression
   pipeline, which to_zarr() would otherwise merge in and fight with
   build_encoding()'s chosen settings for this store.

   Args:
      ds (xr.Dataset): dataset whose variables get their fill attrs and
         encoding cleared in place.
   """
   for var in ds.variables:
      ds[var].attrs.pop(utils.OutputFormat.fill_value, None)
      ds[var].attrs.pop(utils.Missing.name, None)
      ds[var].encoding = {}


def resolve_output_store(output_store):
   """Prepare a fresh zarr v3 store write location: remove a pre-existing
   local directory, but refuse an existing s3:// path outright -- there's
   no single directory to remove, so silently overwriting risks leaving
   orphaned chunks from a differently-shaped previous store.

   Args:
      output_store (str): local path or s3:// URL to prepare.
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


def upload_local_staging_dir(
   local_staging_dir, output_store, keep_local_staging
):
   """Upload a local zarr store to S3 in one recursive `aws s3 cp`, then
   remove the local copy unless kept -- avoids the interleaved per-batch S3
   writes (and repeated partial rewrites of small chunks) that direct-to-S3
   writing incurs.

   Args:
      local_staging_dir (str): local directory to upload.
      output_store (str): s3:// destination.
      keep_local_staging (bool): if True, keep local_staging_dir after
         upload instead of deleting it.
   """
   logging.info(
      f'Uploading local staging directory {local_staging_dir} to {output_store}'
   )

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
   keep_local_staging=False,
   num_layers=0
):
   """Materialize a virtual datacube into a real zarr v3 datacube, batched
   along 'time' to bound memory use.

   Args:
      input_store (str): virtual cube icechunk repository path (s3:// or
         local).
      output_store (str): destination zarr v3 store (s3:// or local).
      bucket_prefix (str): S3 URL prefix the virtual chunk container
         resolves granule references against (see open_virtual_cube()).
      batch_size (int): number of 'time' layers materialized/written per
         batch.
      time_chunk (int): forwarded to build_encoding().
      xy_chunk (int): forwarded to build_encoding().
      time_chunk_1d (int): forwarded to build_encoding().
      xy_shard_multiplier (int): forwarded to build_encoding().
      local_staging_dir (str, optional): if given (requires an s3://
         `output_store`), writes every batch locally and uploads the whole
         store in one shot at the end -- see upload_local_staging_dir().
      keep_local_staging (bool): keep local_staging_dir after upload
         instead of deleting it.
      num_layers (int): caps how many layers to process, e.g. for
         benchmarking on a bounded subset; 0 (default) processes all.
   """
   if local_staging_dir and not output_store.startswith(utils.S3_PREFIX):
      raise ValueError(
         "--local-staging-dir only applies when --output-store is an s3:// "
         f"path, got {output_store}"
      )

   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   if num_layers > 0 and num_layers < total_layers:
      logging.info(f'Limiting to first {num_layers} of {total_layers} layers (--num-layers)')
      total_layers = num_layers

   if total_layers == 0:
      logging.info(f'{input_store} has no layers, nothing to deep-copy')
      return

   time_vars, static_vars = split_vars_by_time(cube)
   encoding = build_encoding(
      cube, time_chunk, xy_chunk, time_chunk_1d,
      xy_shard_multiplier
   )

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   # Fail fast on an existing S3 destination even when staging locally, so
   # the check doesn't wait until after every batch is already written.
   resolve_output_store(output_store)

   # Also cleans up a stale local_staging_dir left by a failed prior run.
   write_target = resolve_output_store(local_staging_dir) if local_staging_dir else output_store

   for batch_num, start in enumerate(range(0, total_layers, batch_size)):
      stop = min(start + batch_size, total_layers)
      logging.info(f'Materializing layers {start}:{stop} of {total_layers}')

      batch = cube[time_vars].isel({utils.Coords.TIME: slice(start, stop)})

      if batch_num == 0:
         # Static vars written once here: to_zarr's append_dim requires
         # every variable to carry it, so they can't appear in later batches.
         batch = xr.merge([batch, cube[static_vars]])

      batch = batch.load()
      _reset_write_encoding(batch)

      if batch_num == 0:
         batch.to_zarr(
            write_target,
            mode='w',
            encoding=encoding,
            zarr_format=3,
            consolidated=False
         )
      else:
         batch.to_zarr(
            write_target,
            append_dim=utils.Coords.TIME,
            zarr_format=3,
            consolidated=False
         )

      logging.info(f'Wrote layers {start}:{stop} of {total_layers} to {write_target}')

   # Once at the end, not per batch -- avoids a full metadata rescan/rewrite
   # on every append.
   zarr.consolidate_metadata(write_target)
   logging.info(f'Consolidated metadata at {write_target}')

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
      default=XY_SHARD_MULTIPLIER,
      help='Number of --xy-chunk-value-sized inner chunks grouped into one '
         'shard per spatial axis, for 3D (time,y,x) variables [%(default)d]. '
         'A value of 1 disables sharding.'
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
   parser.add_argument(
      '-n', '--num-layers',
      type=int,
      default=0,
      help='Only materialize the first N layers of the virtual cube (e.g. '
         'for benchmarking on a bounded subset) [%(default)d meaning to '
         'process all layers].'
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
      args.keep_local_staging,
      args.num_layers
   )

   elapsed_time = time.time() - start_time
   logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')
   logging.info('Done')
