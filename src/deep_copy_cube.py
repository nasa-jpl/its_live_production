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

import icechunk as ic
import xarray as xr
from zarr.codecs import BloscCodec

import utils
from itscube_types import CubeFormat

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

S3_PREFIX = 's3://'

# Granules are written to the file in chunks to avoid out of memory issues.
# Number of granules to write to the file at a time.
NUM_GRANULES_TO_WRITE = 1000

# Chunking to apply when writing datacube to the Zarr store for 3-d variables
TIME_CHUNK_VALUE = 20000
X_Y_CHUNK_VALUE = 10

# Chunking to apply to 1-D data variables when writing datacube to the
# Zarr store
TIME_CHUNK_VALUE_1D = 200000

# Compressor for the deep-copy zarr v3 store; same one already used for ice
# mask variables in virtual_itslive_cube_per_chunk.py.
COMPRESSOR = BloscCodec(cname="lz4", clevel=1, shuffle='bitshuffle')


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

   if store_path.startswith(S3_PREFIX):
      s3_parts = store_path.replace(S3_PREFIX, '').split('/', 1)
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

   return xr.open_zarr(
      repo.readonly_session("main").store, consolidated=False, zarr_format=3
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


def build_encoding(cube, total_layers, time_chunk, xy_chunk, time_chunk_1d):
   """Build the zarr v3 encoding dict for the deep-copy store.

   Only sets chunking + compressor per variable: dtype/fill_value are already
   correct on the virtual cube (see src/wiki dtype-preservation notes), so
   they're left untouched. Chunking follows itscube.py's scheme:
   - 3D (time, y, x) variables: (min(total_layers, time_chunk), xy_chunk, xy_chunk)
   - 1D (time,) variables: (min(total_layers, time_chunk_1d),)
   - static 2D (y, x) variables (landice/floatingice): full extent
   - x/y coordinates: full extent
   - 0-d variables (mapping): no chunk encoding

   Parameters
   ----------
   cube : xr.Dataset
      The virtual datacube (used only for each variable's dims/sizes).
   total_layers : int
      Total number of layers along 'time', to cap chunk sizes the same way
      itscube.py does (min(max_number_of_layers, TIME_CHUNK_VALUE)).
   time_chunk : int
      Chunk size along 'time' for 3D variables.
   xy_chunk : int
      Chunk size along 'x'/'y' for 3D variables.
   time_chunk_1d : int
      Chunk size for 1D ('time',) variables.

   Returns
   -------
   dict
      Per-variable/coordinate encoding dict for xr.Dataset.to_zarr().
   """
   encoding = {}

   for coord_name in (utils.Coords.X, utils.Coords.Y):
      if coord_name in cube.coords:
         encoding[coord_name] = {
            'chunks': (cube.sizes[coord_name],),
            'compressors': [COMPRESSOR],
         }

   for var_name in cube.data_vars:
      dims = cube[var_name].dims

      if len(dims) == 0:
         # Scalar variable (e.g. 'mapping'): no chunk encoding needed.
         continue

      if utils.Coords.TIME in dims:
         if len(dims) == 3:
            chunks = (min(total_layers, time_chunk), xy_chunk, xy_chunk)
         else:
            chunks = (min(total_layers, time_chunk_1d),)
      else:
         # Static 2D (y, x) variable: full extent, matching itscube.py.
         chunks = tuple(cube.sizes[d] for d in dims)

      encoding[var_name] = {
         'chunks': chunks,
         'compressors': [COMPRESSOR],
      }

   return encoding


def resolve_output_store(output_store):
   """Prepare the output location for a fresh zarr v3 store write.

   For a local path, remove any pre-existing directory first (mirrors
   ITSCube.init_output_store). For an s3:// path, return it as-is --
   xarray/zarr resolve 's3://...' store URLs via fsspec, with authenticated
   write access expected to come from the ambient AWS credential chain (not
   anonymous, unlike the *input* virtual chunk container).

   Parameters
   ----------
   output_store : str
      Local path or s3:// URL for the deep-copy zarr v3 store.

   Returns
   -------
   str
      The (possibly cleaned-up) output store path.
   """
   if not output_store.startswith(S3_PREFIX) and os.path.exists(output_store):
      logging.info(f"Removing existing {output_store}")
      shutil.rmtree(output_store)

   return output_store


def deep_copy_cube(
   input_store,
   output_store,
   bucket_prefix,
   batch_size,
   time_chunk,
   xy_chunk,
   time_chunk_1d
):
   """Materialize a virtual datacube into a real zarr v3 datacube, batched
   along 'time' to bound memory use.

   Parameters
   ----------
   input_store : str
      Path to the virtual cube's icechunk repository (s3:// or local).
   output_store : str
      Path to write the deep-copy zarr v3 store to (s3:// or local).
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
   """
   cube = open_virtual_cube(input_store, bucket_prefix)
   total_layers = cube.sizes[utils.Coords.TIME]
   logging.info(f'Opened virtual cube {input_store}: {total_layers} layers')

   time_vars, static_vars = split_vars_by_time(cube)
   encoding = build_encoding(cube, total_layers, time_chunk, xy_chunk, time_chunk_1d)

   cube.attrs[CubeFormat.date_updated] = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

   output_store = resolve_output_store(output_store)

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

      if batch_num == 0:
         batch.to_zarr(
            output_store,
            mode='w',
            encoding=encoding,
            zarr_format=3,
            consolidated=True
         )
      else:
         batch.to_zarr(
            output_store,
            append_dim=utils.Coords.TIME,
            zarr_format=3,
            consolidated=True
         )

      logging.info(f'Wrote layers {start}:{stop} of {total_layers} to {output_store}')

   logging.info(f'Done: deep-copied {total_layers} layers to {output_store}')


if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(
      description="""
      Materialize a virtual ITS_LIVE datacube (icechunk repo built by
      virtual_itslive_cube_per_chunk.py) into a real Zarr v3 datacube, chunked
      the same way itscube.py chunks a regular datacube.

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

   args = parser.parse_args()

   deep_copy_cube(
      args.input_store,
      args.output_store,
      args.bucket,
      args.batch_size,
      args.time_chunk_value,
      args.xy_chunk_value,
      args.time_chunk_value_1d
   )

   logging.info('Done')
