#!/usr/bin/env python
"""
Restore NISAR M11 and M12 values and corresponding dr_to_vr_factor attributes
within existing ITS_LIVE datacubes that are residing in the AWS S3 bucket.

* Copy M11 and M12 values from original NISAR granules into corresponding
layers of the existing datacubes.
* Push corrected datacubes back to the S3 bucket by:
    ** Removing original M11 and M12 from s3 bucket as those are of int dtype.
    ** Copy cube's metadata files and newly restored M11/M12 data variables to
    the s3 cube's original location as np.float32 dtype.

ATTN: This script should run from AWS EC2 instance to have fast access to the S3
bucket. It takes 2 seconds to upload the file to the S3 bucket from EC2 instance
vs. 1.5 minutes to upload the file from laptop to the S3 bucket.

Authors: Masha Liukis, Alex Gardner
"""
import argparse
import json
from pathlib import Path
import logging
import numpy as np
import os
import s3fs
import shutil
from joblib import Parallel, delayed, parallel_config
import xarray as xr
import zarr

from itscube_types import Vars, ImgPairInfo
from itslive_binary_type import BinaryFlag
import itslive_utils
import utils


class GranuleFilter:
    """
    Supported granule filter modes for selecting which layers within a
    datacube carry M11/M12 data and need restoration.

    RSLC: match granule URLs that contain the substring 'RSLC' (NISAR).
    S1:   match granule URLs whose basename starts with 'S1' (Sentinel-1).
    """
    RSLC = 'RSLC'
    S1 = 'S1'
    ALL = [RSLC, S1]


# Result container for one granule's fetched data.
# All fields are plain Python/numpy — no xarray, no shared state.
class GranuleResult:
    __slots__ = (
        'index',
        'ascending_img1', 'ascending_img2',
        # M11/M12 fields — None when granule is not RSLC/S1
        'm11_data', 'm12_data',
        'm11_factor', 'm12_factor',
        'x_coords', 'y_coords',
        # Integer positions into the cube's x/y axes (resolved once, reused on write)
        'x_idx', 'y_idx',
    )

    def __init__(self, index: int):
        self.index = index
        self.ascending_img1 = Vars.intMissingValue[Vars.ascending_img1]
        self.ascending_img2 = Vars.intMissingValue[Vars.ascending_img2]
        self.m11_data = None
        self.m12_data = None
        self.m11_factor = None
        self.m12_factor = None
        self.x_coords = None
        self.y_coords = None
        self.x_idx = None
        self.y_idx = None


class FixDatacubes:
    """
    Class to apply fixes to ITS_LIVE datacubes:

    * Copy M11 and M12 values from original NISAR granules into corresponding
    layers of the existing datacubes.

    * Push corrected datacubes back to the S3 bucket by:
    ** Removing original M11 and M12 from s3 bucket as those are of int dtype.
    ** Copy cube's metadata files and newly restored M11/M12 data variables to
    the s3 cube's original location as np.float32 dtype.
    """
    DRY_RUN = False

    # Number of time layers to process in one in-memory chunk.
    # At 1000 layers × 833 × 834 × float32 ≈ 2.6 GiB for M11+M12 combined,
    # well within 60 GB RAM even with other variables resident.
    TIME_CHUNK_SIZE = 1000

    def __init__(
        self, cubes_to_process, local_original_cube_dir: str, local_dir: str
    ):
        """
        Initialize object.

        Args:
            cubes_to_process: List of s3 urls for cubes to process.
            local_original_cube_dir (str): Local directory to store
                downloaded original datacubes to fix.
            local_dir (str): Local directory to save corrected cubes to.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        self.local_original_cube_dir = local_original_cube_dir
        self.local_dir = local_dir

        self.all_zarr_datacubes = cubes_to_process
        logging.info(f"Number of datacubes to process: {len(self.all_zarr_datacubes)}")

        os.makedirs(self.local_dir, exist_ok=True)
        os.makedirs(self.local_original_cube_dir, exist_ok=True)

    def __call__(
        self,
        num_threads: int = 8,
        start_cube: int = 0,
        granule_filter: str = GranuleFilter.RSLC,
    ):
        """
        Restore M11 and M12 related data for the ITS_LIVE datacubes stored
        in S3 bucket. Outer loop over cubes runs serially; inner loop over
        granules within each cube is parallelized.

        Args:
            num_threads (int): Number of parallel threads for granule fetching.
            start_cube (int): Index of the first cube to process (allows
                resuming after a failed run).
            granule_filter (str): Which granule type carries M11/M12 data.
                Must be one of GranuleFilter.ALL. Defaults to GranuleFilter.RSLC.
        """
        num_to_process = len(self.all_zarr_datacubes) - start_cube

        logging.info(f"{num_to_process} datacubes to fix (filter={granule_filter})...")

        if num_to_process <= 0:
            logging.info("Nothing to fix, exiting.")
            return

        for cube_url in self.all_zarr_datacubes[start_cube:]:
            msgs = FixDatacubes.all(
                cube_url,
                self.local_original_cube_dir,
                self.local_dir,
                self.s3,
                num_threads,
                granule_filter,
            )
            logging.info("\n-->".join(msgs))

    @staticmethod
    def fetch_one_granule(
        index: int,
        granule_url: str,
        cube_x: np.ndarray,
        cube_y: np.ndarray,
        is_target: bool,
        s3: s3fs.S3FileSystem,
    ) -> GranuleResult:
        """
        Fetch one granule from S3 and extract all data needed to update the
        cube. This function is pure: it reads from S3 and returns a
        GranuleResult with plain numpy arrays. It never touches the shared
        in-memory cube dataset.

        This is the function that runs in parallel threads. Since it only
        performs S3 I/O and local numpy ops, it is safe to call concurrently
        for any number of granules — threads never share writable state.

        Args:
            index (int): Time index of this granule in the datacube.
            granule_url (str): HTTP(S) URL of the source granule NetCDF file.
            cube_x (np.ndarray): Sorted x-coordinate array of the datacube.
            cube_y (np.ndarray): Sorted y-coordinate array of the datacube.
            is_target (bool): Whether this granule matches the filter and
                carries M11/M12 data.
            s3 (s3fs.S3FileSystem): Shared s3fs handle (thread-safe for reads).

        Returns:
            GranuleResult populated with extracted data.
        """
        result = GranuleResult(index)

        granule_basename = os.path.basename(granule_url)
        if not granule_basename.startswith('S1') and \
            not granule_basename.startswith('NISAR'):
                # Only radar data of Sentinel-1 and NISAR missions have
                # flight direction information in img_pair_info attributes.
                # For other missions, we skip fetching the granule
                return result

        granule_s3 = granule_url.replace('https://', '').replace('.s3.amazonaws.com', '')

        with s3.open(granule_s3, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=utils.NC_ENGINE) as granule_ds:
                granule_ds = granule_ds.load()

        logging.info(f'[{index}] Fetched {granule_s3}')

        # Flight direction flags — present for all granules in radar layers
        result.ascending_img1 = np.uint8(
            granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img1].strip()
            == ImgPairInfo.ascending
        )
        result.ascending_img2 = np.uint8(
            granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img2].strip()
            == ImgPairInfo.ascending
        )

        if not is_target:
            return result

        # Crop the granule to the cube's spatial bounding box
        grid_x_min, grid_x_max = cube_x.min(), cube_x.max()
        grid_y_min, grid_y_max = cube_y.min(), cube_y.max()

        mask_x = (granule_ds.x >= grid_x_min) & (granule_ds.x <= grid_x_max)
        mask_y = (granule_ds.y >= grid_y_min) & (granule_ds.y <= grid_y_max)
        cropped_ds = granule_ds.where(mask_x & mask_y, drop=True)

        if cropped_ds.x.size == 0 or cropped_ds.y.size == 0:
            logging.warning(f'[{index}] Granule has no overlap with cube: {granule_s3}')
            return result

        result.x_coords = cropped_ds.x.values
        result.y_coords = cropped_ds.y.values

        # Resolve cropped coordinates to integer positions in the cube axes.
        # np.clip guards against floating-point edge cases where a coordinate
        # sits exactly at (or epsilon beyond) the cube boundary and searchsorted
        # returns len(axis) instead of len(axis)-1.
        result.x_idx = np.clip(
            np.searchsorted(cube_x, result.x_coords), 0, len(cube_x) - 1
        )
        result.y_idx = np.clip(
            np.searchsorted(cube_y, result.y_coords), 0, len(cube_y) - 1
        )

        # Extract M11/M12 as plain numpy arrays — drop time dim (shape: [y, x])
        result.m11_data = cropped_ds[Vars.m11].isel(time=0).drop_vars(utils.Coords.TIME).values
        result.m12_data = cropped_ds[Vars.m12].isel(time=0).drop_vars(utils.Coords.TIME).values

        result.m11_factor = granule_ds[Vars.m11].attrs[Vars.postfix.dr_to_vr_factor]
        result.m12_factor = granule_ds[Vars.m12].attrs[Vars.postfix.dr_to_vr_factor]

        return result

    @staticmethod
    def apply_result(
        result: GranuleResult,
        chunk_offset: int,
        m11_chunk: np.ndarray,
        m12_chunk: np.ndarray,
        factor_m11_chunk: np.ndarray,
        factor_m12_chunk: np.ndarray,
        ascending_img1: np.ndarray,
        ascending_img2: np.ndarray,
    ) -> None:
        """
        Write one GranuleResult into the current in-memory chunk arrays.

        Runs serially after all parallel fetches for a chunk complete.
        Each result has a unique time index so writes never conflict.
        Direct numpy indexing is used throughout — no xarray, no pandas.

        Args:
            result (GranuleResult): Fetched data for one granule.
            chunk_offset (int): Absolute time index of the first layer in the
                current chunk. Used to convert result.index (absolute) to a
                position within the chunk arrays (relative).
            m11_chunk (np.ndarray): In-memory M11 slice for the current chunk,
                shape (chunk_size, ny, nx). Modified in place.
            m12_chunk (np.ndarray): In-memory M12 slice, same shape. Modified
                in place.
            factor_m11_chunk (np.ndarray): 1-D dr_to_vr_factor array for M11,
                length chunk_size. Modified in place.
            factor_m12_chunk (np.ndarray): 1-D dr_to_vr_factor array for M12.
                Modified in place.
            ascending_img1 (np.ndarray): Full-length 1-D output array for
                flight direction flag, image 1. Indexed by result.index.
            ascending_img2 (np.ndarray): Full-length 1-D output array for
                flight direction flag, image 2. Indexed by result.index.
        """
        i_abs = result.index
        i_rel = i_abs - chunk_offset  # position within the current chunk

        ascending_img1[i_abs] = result.ascending_img1
        ascending_img2[i_abs] = result.ascending_img2

        if result.m11_data is None:
            # Non-target granule: nothing more to do
            return

        # np.ix_ builds the open mesh for fancy indexing into the [y, x] slice
        yx_idx = np.ix_(result.y_idx, result.x_idx)

        m11_chunk[i_rel][yx_idx] = result.m11_data
        m12_chunk[i_rel][yx_idx] = result.m12_data

        factor_m11_chunk[i_rel] = result.m11_factor
        factor_m12_chunk[i_rel] = result.m12_factor

    @staticmethod
    def _build_encoding(ds: xr.Dataset) -> dict:
        """
        Build the encoding dict for all variables in ds, using the chunking
        already present in the original cube as the golden standard and
        applying lz4/BITSHUFFLE compression uniformly.

        M11 and M12 are re-typed to float32. ascending_img1/img2 are typed
        as ubyte. All other variables keep their existing dtype.

        Args:
            ds (xr.Dataset): The lazily-opened original cube dataset, whose
                .encoding attributes carry the on-disk chunking to preserve.

        Returns:
            Dict mapping variable name -> encoding kwargs for to_zarr().
        """
        chunking_1d = ds[ImgPairInfo.date_dt].encoding[utils.OutputFormat.chunks]
        chunking_2d = (len(ds.y), len(ds.x))
        chunking_3d = ds[Vars.chip_size_height].encoding[utils.OutputFormat.chunks]
        compression_zarr = zarr.Blosc(
            cname="lz4", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE
        )

        encoding = {}

        for var in ds:
            var_enc = dict(ds[var].encoding)  # copy so we don't mutate the dataset
            var_enc.pop('source', None)        # xarray internal — not valid for to_zarr

            if utils.OutputFormat.chunks in var_enc:
                ndim = len(var_enc[utils.OutputFormat.chunks])
                if ndim == 1:
                    var_enc[utils.OutputFormat.chunks] = chunking_1d
                elif ndim == 2:
                    var_enc[utils.OutputFormat.chunks] = chunking_2d
                elif ndim == 3:
                    var_enc[utils.OutputFormat.chunks] = chunking_3d

            var_enc[utils.OutputFormat.compressor] = compression_zarr
            encoding[var] = var_enc

        # M11/M12: promote to float32 with a proper fill value
        for m_var in [Vars.m11, Vars.m12]:
            encoding.setdefault(m_var, {})[utils.OutputFormat.dtype] = np.float32
            encoding[m_var][utils.Missing.name] = utils.Missing.value

        # ascending_img1/img2: explicit ubyte dtype
        for asc_var in [Vars.ascending_img1, Vars.ascending_img2]:
            encoding.setdefault(asc_var, {})[utils.OutputFormat.dtype] = np.uint8
            encoding[asc_var][utils.Missing.name] = utils.Missing.u8value
            # New 1-D variables — assign the same 1-D chunking
            encoding[asc_var][utils.OutputFormat.chunks] = chunking_1d
            encoding[asc_var][utils.OutputFormat.compressor] = compression_zarr

        return encoding

    @staticmethod
    def all(
        cube_url: str,
        local_original_cube_dir: str,
        local_dir: str,
        s3: s3fs.S3FileSystem,
        num_threads: int = 8,
        granule_filter: str = GranuleFilter.RSLC,
    ) -> list[str]:
        """
        Fix M11 and M12 related data in one datacube and copy it to the S3
        bucket's original location.

        Processing strategy for large cubes that exceed available RAM:
          - The time axis is processed in chunks of TIME_CHUNK_SIZE layers.
          - For each chunk:
              1. Granules are fetched from S3 in parallel (I/O-bound).
              2. M11/M12/factor data is assembled into small numpy arrays
                 (one chunk at a time) and written to the fixed Zarr store.
          - The first chunk creates the output Zarr store via xr.Dataset.to_zarr
            (using the original cube's encoding as the golden standard).
            Subsequent chunks write directly via zarr.open_group to avoid
            ever allocating the full time-axis arrays in memory.
          - ascending_img1/img2 are accumulated across all chunks in full-
            length 1-D arrays (cheap: ubyte, num_layers bytes each) and
            written in a single pass at the end.

        Args:
            cube_url (str): S3 URL of the datacube Zarr store to fix.
            local_original_cube_dir (str): Local path to cache the downloaded
                original cube.
            local_dir (str): Local path to write the fixed cube to.
            s3 (s3fs.S3FileSystem): Shared s3fs handle used for granule reads.
            num_threads (int): Number of parallel threads for granule fetching.
            granule_filter (str): Which granule type carries M11/M12 data.
                GranuleFilter.RSLC matches URLs containing 'RSLC' (NISAR).
                GranuleFilter.S1 matches URLs whose basename starts with 'S1'
                (Sentinel-1).

        Returns:
            List of log message strings.
        """
        msgs = [f'Processing {cube_url}']
        env_copy = os.environ.copy()

        cube_basename = os.path.basename(cube_url)
        local_original_cube = os.path.join(local_original_cube_dir, cube_basename)

        # Download the cube locally if not already cached from a prior run
        if not os.path.exists(local_original_cube):
            command_line = [
                "aws", "s3", "cp", "--recursive",
                cube_url,
                local_original_cube
            ]
            msgs.append(f"Creating local copy of {cube_url}: {local_original_cube}")
            msgs.append(' '.join(command_line))
            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

        fixed_file = os.path.join(local_dir, cube_basename)
        ascending_fill_value = Vars.intMissingValue[Vars.ascending_img1]
        factor_m11 = f'{Vars.m11}_{Vars.postfix.dr_to_vr_factor}'
        factor_m12 = f'{Vars.m12}_{Vars.postfix.dr_to_vr_factor}'

        # Open the original cube lazily — no large arrays loaded yet.
        # We keep this handle open for the full duration so we can read
        # metadata, coordinates, and small variables without re-opening.
        with xr.open_dataset(
            local_original_cube,
            decode_timedelta=False,
            engine='zarr',
            consolidated=True
        ) as ds:
            msgs.append(f'Cube dimensions: {ds.dims}')
            logging.info(msgs[-1])

            num_layers = len(ds.mid_date)

            # Snapshot cube spatial axes as plain numpy arrays once —
            # passed into worker threads to avoid repeated xarray access
            cube_x = ds.x.values.copy()
            cube_y = ds.y.values.copy()

            # Build a boolean mask over the time dimension identifying which
            # layers carry M11/M12 data and need restoration.
            granule_urls_str = ds['granule_url'].values.astype(str)

            if granule_filter == GranuleFilter.RSLC:
                target_mask = np.char.find(granule_urls_str, 'RSLC') >= 0
            elif granule_filter == GranuleFilter.S1:
                basenames = np.array([os.path.basename(u) for u in granule_urls_str])
                target_mask = np.char.startswith(basenames, 'S1')
            else:
                raise ValueError(
                    f"Unknown granule_filter={granule_filter!r}. "
                    f"Must be one of {GranuleFilter.ALL}."
                )

            num_target_layers = int(target_mask.sum())
            msgs.append(
                f'Identified {num_target_layers} {granule_filter} layers in {cube_basename}'
            )
            logging.info(msgs[-1])

            # ------------------------------------------------------------------
            # Build encoding once from the original cube's on-disk settings.
            # This preserves the original chunking exactly and applies the new
            # compression and dtype overrides for M11/M12/ascending.
            # ------------------------------------------------------------------
            encoding = FixDatacubes._build_encoding(ds)

            # ------------------------------------------------------------------
            # Full-length 1-D arrays for ascending flags — cheap to hold in
            # memory (ubyte, ~100 KB for 111k layers), accumulated across all
            # chunks and written in a single pass at the end.
            # ------------------------------------------------------------------
            ascending_img1 = np.full(num_layers, ascending_fill_value, dtype=np.uint8)
            ascending_img2 = np.full(num_layers, ascending_fill_value, dtype=np.uint8)

            # ------------------------------------------------------------------
            # Chunked processing loop
            # ------------------------------------------------------------------
            chunk_size = FixDatacubes.TIME_CHUNK_SIZE
            is_first_chunk = True

            for chunk_start in range(0, num_layers, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_layers)
                actual_chunk_size = chunk_end - chunk_start

                msgs.append(
                    f'Processing time chunk [{chunk_start}:{chunk_end}] '
                    f'({actual_chunk_size} layers) of {num_layers} total'
                )
                logging.info(msgs[-1])

                # Load M11/M12 and their factor arrays for this chunk only.
                # shape: (actual_chunk_size, ny, nx) for 3-D vars
                #        (actual_chunk_size,) for 1-D factor vars
                if num_target_layers:
                    m11_chunk  = ds[Vars.m11].isel(mid_date=slice(chunk_start, chunk_end)).values
                    m12_chunk  = ds[Vars.m12].isel(mid_date=slice(chunk_start, chunk_end)).values
                    factor_m11_chunk = ds[factor_m11].isel(mid_date=slice(chunk_start, chunk_end)).values
                    factor_m12_chunk = ds[factor_m12].isel(mid_date=slice(chunk_start, chunk_end)).values
                else:
                    ny, nx = len(cube_y), len(cube_x)
                    m11_chunk  = np.full((actual_chunk_size, ny, nx), utils.Missing.value, dtype=np.float32)
                    m12_chunk  = np.full((actual_chunk_size, ny, nx), utils.Missing.value, dtype=np.float32)
                    factor_m11_chunk = np.full(actual_chunk_size, utils.Missing.value, dtype=np.float32)
                    factor_m12_chunk = np.full(actual_chunk_size, utils.Missing.value, dtype=np.float32)

                # Build task list for this chunk
                tasks = [
                    (
                        i,
                        granule_urls_str[i],
                        cube_x,
                        cube_y,
                        bool(target_mask[i]),
                        s3,
                    )
                    for i in range(chunk_start, chunk_end)
                ]

                # --------------------------------------------------------------
                # Parallel fetch phase for this chunk
                # --------------------------------------------------------------
                with parallel_config(backend='threading', n_jobs=num_threads):
                    results: list[GranuleResult] = Parallel()(
                        delayed(FixDatacubes.fetch_one_granule)(*task)
                        for task in tasks
                    )

                # --------------------------------------------------------------
                # Serial apply phase for this chunk
                # --------------------------------------------------------------
                for result in results:
                    FixDatacubes.apply_result(
                        result,
                        chunk_start,
                        m11_chunk,
                        m12_chunk,
                        factor_m11_chunk,
                        factor_m12_chunk,
                        ascending_img1,
                        ascending_img2,
                    )

                # --------------------------------------------------------------
                # Write this chunk to the output Zarr store.
                #
                # First chunk: write the full dataset via xr.Dataset.to_zarr
                # with all encoding settings.  We build a lightweight slice
                # dataset from the original ds using isel — this avoids loading
                # any large variable in full; xarray streams each variable
                # from the original store while writing.  We then overwrite
                # M11/M12/factors in that slice with the corrected chunk data.
                #
                # Subsequent chunks: write only the changed variables directly
                # via zarr.open_group with region slicing, bypassing xarray
                # entirely.  This avoids ever holding the full time-axis arrays
                # in RAM.
                # --------------------------------------------------------------
                if is_first_chunk:
                    # Slice the full dataset to this chunk along mid_date.
                    # All non-time variables (x, y, mapping, …) are included
                    # automatically; xarray streams them from the original store.
                    chunk_ds = ds.isel(mid_date=slice(chunk_start, chunk_end))

                    # Attach ascending vars to the chunk dataset (they don't
                    # exist yet in the original cube)
                    new_coords = chunk_ds[ImgPairInfo.satellite_img1].coords
                    new_dims   = chunk_ds[ImgPairInfo.satellite_img1].dims

                    chunk_ds[Vars.ascending_img1] = xr.DataArray(
                        data=ascending_img1[chunk_start:chunk_end],
                        coords=new_coords, dims=new_dims
                    )
                    chunk_ds[Vars.ascending_img1].attrs = {
                        Vars.attrs.std_name: Vars.name[Vars.ascending_img1],
                        Vars.attrs.description: Vars.description[Vars.ascending_img1],
                        BinaryFlag.attrs.values: BinaryFlag.values,
                        BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img1],
                    }
                    chunk_ds[Vars.ascending_img2] = xr.DataArray(
                        data=ascending_img2[chunk_start:chunk_end],
                        coords=new_coords, dims=new_dims
                    )
                    chunk_ds[Vars.ascending_img2].attrs = {
                        Vars.attrs.std_name: Vars.name[Vars.ascending_img2],
                        Vars.attrs.description: Vars.description[Vars.ascending_img2],
                        BinaryFlag.attrs.values: BinaryFlag.values,
                        BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img2],
                    }

                    # Overwrite M11/M12/factors in the chunk dataset with
                    # corrected numpy arrays before writing to disk.
                    # Wrap in xr.DataArray to preserve coordinates.
                    chunk_ds[Vars.m11] = xr.DataArray(
                        data=m11_chunk,
                        coords=chunk_ds[Vars.m11].coords,
                        dims=chunk_ds[Vars.m11].dims,
                        attrs=chunk_ds[Vars.m11].attrs,
                    )
                    chunk_ds[Vars.m12] = xr.DataArray(
                        data=m12_chunk,
                        coords=chunk_ds[Vars.m12].coords,
                        dims=chunk_ds[Vars.m12].dims,
                        attrs=chunk_ds[Vars.m12].attrs,
                    )
                    chunk_ds[factor_m11] = xr.DataArray(
                        data=factor_m11_chunk,
                        coords=chunk_ds[factor_m11].coords,
                        dims=chunk_ds[factor_m11].dims,
                        attrs=chunk_ds[factor_m11].attrs,
                    )
                    chunk_ds[factor_m12] = xr.DataArray(
                        data=factor_m12_chunk,
                        coords=chunk_ds[factor_m12].coords,
                        dims=chunk_ds[factor_m12].dims,
                        attrs=chunk_ds[factor_m12].attrs,
                    )

                    msgs.append(f"Creating output Zarr store: {fixed_file}")
                    logging.info(msgs[-1])

                    chunk_ds.to_zarr(
                        fixed_file,
                        encoding=encoding,
                        consolidated=False,  # consolidate once at the very end
                        mode='w',
                    )
                    is_first_chunk = False

                else:
                    # All subsequent chunks: write via zarr directly to avoid
                    # allocating full-length arrays.
                    t_slice = slice(chunk_start, chunk_end)

                    out_store = zarr.open_group(fixed_file, mode='r+')
                    out_store[Vars.m11][t_slice, :, :]  = m11_chunk
                    out_store[Vars.m12][t_slice, :, :]  = m12_chunk
                    out_store[factor_m11][t_slice]       = factor_m11_chunk
                    out_store[factor_m12][t_slice]       = factor_m12_chunk
                    out_store[Vars.ascending_img1][t_slice] = ascending_img1[chunk_start:chunk_end]
                    out_store[Vars.ascending_img2][t_slice] = ascending_img2[chunk_start:chunk_end]

                # Free chunk arrays before loading the next chunk
                del m11_chunk, m12_chunk, factor_m11_chunk, factor_m12_chunk, results

            # ------------------------------------------------------------------
            # Consolidate metadata once after all chunks are written
            # ------------------------------------------------------------------
            msgs.append(f"Consolidating metadata for {fixed_file}")
            logging.info(msgs[-1])
            zarr.consolidate_metadata(fixed_file)

        # --------------------------------------------------------------------
        # Upload fixed variables and metadata back to S3
        # --------------------------------------------------------------------
        if os.path.exists(fixed_file):
            for each_var in [Vars.m11, Vars.m12]:
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{each_var}',
                    f'{cube_url}/{each_var}',
                    "--acl", "bucket-owner-full-control",
                ]
                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

                factor_var = f'{each_var}_{Vars.postfix.dr_to_vr_factor}'
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{factor_var}',
                    f'{cube_url}/{factor_var}',
                    "--acl", "bucket-owner-full-control",
                ]
                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            for each_var in [Vars.ascending_img1, Vars.ascending_img2]:
                command_line = [
                    "aws", "s3", "cp", "--recursive",
                    f'{fixed_file}/{each_var}',
                    f'{cube_url}/{each_var}',
                    "--acl", "bucket-owner-full-control",
                ]
                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            for each_file in itslive_utils.CUBE_META:
                command_line = [
                    "aws", "s3", "cp",
                    f'{fixed_file}/{each_file}',
                    f'{cube_url}/{each_file}',
                    "--acl", "bucket-owner-full-control",
                ]
                msgs.append(' '.join(command_line))
                if not FixDatacubes.DRY_RUN:
                    itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

            msgs.append(f"Removing local {fixed_file}")
            # shutil.rmtree(fixed_file)

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=8,
        help='Number of threads for parallel granule fetching within each '
             'cube [%(default)d].'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of filenames to generate [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help='File that contains JSON list of datacubes filenames in s3 bucket '
             'to process [%(default)s].'
    )
    parser.add_argument(
        '-l', '--local_dir',
        type=str,
        default='sandbox-fixed-nisar',
        help='Directory to store fixed datacubes before uploading them to the '
             'S3 bucket [%(default)s]'
    )
    parser.add_argument(
        '-o', '--local_original_cube_dir',
        type=str,
        default='sandbox-original-nisar',
        help='Directory to store downloaded original datacubes [%(default)s]'
    )
    parser.add_argument(
        '-s', '--start_cube',
        type=int,
        default=0,
        help='Index for the start datacube to process (if previous processing '
             'terminated) [%(default)d]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit AWS push/pull commands.'
    )
    parser.add_argument(
        '--granuleFilter',
        type=str,
        default=GranuleFilter.RSLC,
        choices=GranuleFilter.ALL,
        help=(
            'Which granule type carries M11/M12 data and needs restoration. '
            f'{GranuleFilter.RSLC}: match URLs containing \'RSLC\' (NISAR, default). '
            f'{GranuleFilter.S1}: match URLs whose basename starts with \'S1\' (Sentinel-1). '
            '[%(default)s]'
        )
    )

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
    )

    logging.info(f"Args: {args}")
    FixDatacubes.DRY_RUN = args.dryrun

    cubes_to_generate = None
    if args.processCubesFile:
        cubes_to_generate = json.loads(args.processCubesFile.read_text())
    elif args.processCubes:
        cubes_to_generate = json.loads(args.processCubes)
    else:
        raise RuntimeError('Should provide cubes to process.')

    if len(cubes_to_generate) != len(set(cubes_to_generate)):
        raise RuntimeError("Duplicate datacubes are identified.")

    logging.info(f"Found {len(cubes_to_generate)} unique datacubes to generate")

    fix_cubes = FixDatacubes(
        cubes_to_generate,
        args.local_original_cube_dir,
        args.local_dir
    )

    fix_cubes(args.threads, args.start_cube, args.granuleFilter)


if __name__ == '__main__':
    main()
    logging.info("Done.")