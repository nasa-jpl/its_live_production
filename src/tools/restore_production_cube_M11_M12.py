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
        # M11/M12 fields — None when granule is not RSLC
        'm11_data', 'm12_data',
        'm11_factor', 'm12_factor',
        'x_coords', 'y_coords',
        # Integer positions into the cube's x/y axes (resolved once, reused on write)
        'x_idx', 'y_idx',
    )

    def __init__(self, index: int):
        self.index = index
        self.ascending_img1 = None
        self.ascending_img2 = None
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
        is_rslc: bool,
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
            is_rslc (bool): Whether this granule is an RSLC granule that
                carries M11/M12 data.
            s3 (s3fs.S3FileSystem): Shared s3fs handle (thread-safe for reads).

        Returns:
            GranuleResult populated with extracted data.
        """
        result = GranuleResult(index)

        granule_s3 = granule_url.replace('https://', '').replace('.s3.amazonaws.com', '')

        with s3.open(granule_s3, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=utils.NC_ENGINE) as granule_ds:
                granule_ds = granule_ds.load()

        logging.info(f'[{index}] Fetched {granule_s3}')

        # Flight direction flags — present for all granules
        result.ascending_img1 = np.ubyte(
            granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img1].strip()
            == ImgPairInfo.ascending
        )
        result.ascending_img2 = np.ubyte(
            granule_ds.img_pair_info.attrs[ImgPairInfo.flight_direction_img2].strip()
            == ImgPairInfo.ascending
        )

        if not is_rslc:
            return result

        # Crop the granule to the cube's spatial bounding box
        grid_x_min, grid_x_max = cube_x.min(), cube_x.max()
        grid_y_min, grid_y_max = cube_y.min(), cube_y.max()

        mask_x = (granule_ds.x >= grid_x_min) & (granule_ds.x <= grid_x_max)
        mask_y = (granule_ds.y >= grid_y_min) & (granule_ds.y <= grid_y_max)
        cropped_ds = granule_ds.where(mask_x & mask_y, drop=True)

        if cropped_ds.x.size == 0 or cropped_ds.y.size == 0:
            logging.warning(f'[{index}] RSLC granule has no overlap with cube: {granule_s3}')
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
        ds: xr.Dataset,
        result: GranuleResult,
        m11_values: np.ndarray,
        m12_values: np.ndarray,
        ascending_img1: np.ndarray,
        ascending_img2: np.ndarray,
    ) -> None:
        """
        Write one GranuleResult into the shared in-memory arrays.

        This function runs serially after all parallel fetches complete.
        Writes are safe because each result has a unique time index.

        Direct numpy indexing is used throughout — no xarray label alignment,
        no pandas overhead.

        Args:
            ds (xr.Dataset): The fully loaded in-memory datacube dataset.
            result (GranuleResult): Fetched data for one granule.
            m11_values (np.ndarray): Pre-extracted m11 backing array
                (shape: [time, y, x]), writable view of ds[Vars.m11].values.
            m12_values (np.ndarray): Pre-extracted m12 backing array,
                same shape as m11_values.
            ascending_img1 (np.ndarray): 1-D output array for flight
                direction flag, image 1.
            ascending_img2 (np.ndarray): 1-D output array for flight
                direction flag, image 2.
        """
        i = result.index
        ascending_img1[i] = result.ascending_img1
        ascending_img2[i] = result.ascending_img2

        if result.m11_data is None:
            # Non-RSLC granule: nothing more to do
            return

        # np.ix_ builds the open mesh for fancy indexing into [y, x] slice
        yx_idx = np.ix_(result.y_idx, result.x_idx)

        m11_values[i][yx_idx] = result.m11_data
        m12_values[i][yx_idx] = result.m12_data

        factor_m11 = f'{Vars.m11}_{Vars.postfix.dr_to_vr_factor}'
        factor_m12 = f'{Vars.m12}_{Vars.postfix.dr_to_vr_factor}'
        ds[factor_m11].values[i] = result.m11_factor
        ds[factor_m12].values[i] = result.m12_factor

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

        The inner loop over granules is fully parallelized:
          1. All granules are fetched from S3 concurrently (I/O-bound,
             threads spend most of their time waiting on the network).
          2. Results are applied to the shared in-memory dataset serially.
             Each result targets a unique time index, so step 2 is fast
             (pure numpy, no I/O) and does not benefit from parallelism.

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
            # The mask logic depends on the chosen granule_filter:
            #   RSLC — substring match anywhere in the URL (NISAR granules)
            #   S1   — basename starts with 'S1' (Sentinel-1 granules)
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

            # Keep the internal name generic so the rest of the method is
            # filter-agnostic.
            rslc_mask = target_mask
            num_rslc_layers = int(rslc_mask.sum())
            msgs.append(
                f'Identified {num_rslc_layers} {granule_filter} layers in {cube_basename}'
            )
            logging.info(msgs[-1])

            # Force-load M11, M12, and their factor variables into memory so
            # that subsequent index assignments write to numpy arrays, not
            # back to the Zarr store on disk.
            # This is required: without .values the assignment silently no-ops
            # (xarray defers writes to the backing store).
            if num_rslc_layers:
                m11_values = ds[Vars.m11].values        # shape: (time, y, x)
                m12_values = ds[Vars.m12].values
                factor_m11 = f'{Vars.m11}_{Vars.postfix.dr_to_vr_factor}'
                factor_m12 = f'{Vars.m12}_{Vars.postfix.dr_to_vr_factor}'
                _ = ds[factor_m11].values               # shape: (time,)
                _ = ds[factor_m12].values
            else:
                m11_values = None
                m12_values = None

            # Pre-build the task list: granule URL + metadata needed by workers
            granule_urls = ds['granule_url'].values.astype(str)
            tasks = [
                (
                    i,
                    granule_urls[i],
                    cube_x,
                    cube_y,
                    bool(rslc_mask[i]),
                    s3,
                )
                for i in range(num_layers)
            ]

            # ----------------------------------------------------------------
            # Parallel fetch phase
            # All S3 network I/O happens here. Workers return GranuleResult
            # objects containing only plain numpy arrays — no shared state is
            # mutated during this phase.
            # ----------------------------------------------------------------
            msgs.append(f'Fetching {num_layers} granules with {num_threads} threads...')
            logging.info(msgs[-1])

            with parallel_config(backend='threading', n_jobs=num_threads):
                results: list[GranuleResult] = Parallel()(
                    delayed(FixDatacubes.fetch_one_granule)(*task)
                    for task in tasks
                )

            # ----------------------------------------------------------------
            # Serial apply phase
            # Write extracted data back into the shared in-memory dataset.
            # This loop is O(num_layers) numpy scalar/slice assignments —
            # fast enough that parallelizing it would add overhead, not remove it.
            # ----------------------------------------------------------------
            ascending_img1 = np.full(num_layers, ascending_fill_value, dtype=np.ubyte)
            ascending_img2 = np.full(num_layers, ascending_fill_value, dtype=np.ubyte)

            for result in results:
                FixDatacubes.apply_result(
                    ds,
                    result,
                    m11_values,
                    m12_values,
                    ascending_img1,
                    ascending_img2,
                )

            # ----------------------------------------------------------------
            # Attach new ascending_img1 / ascending_img2 data variables
            # ----------------------------------------------------------------
            new_coords = ds[ImgPairInfo.satellite_img1].coords
            new_dims = ds[ImgPairInfo.satellite_img1].dims

            ds[Vars.ascending_img1] = xr.DataArray(
                data=ascending_img1, coords=new_coords, dims=new_dims
            )
            ds[Vars.ascending_img1].attrs = {
                Vars.attrs.std_name: Vars.name[Vars.ascending_img1],
                Vars.attrs.description: Vars.description[Vars.ascending_img1],
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img1],
            }

            ds[Vars.ascending_img2] = xr.DataArray(
                data=ascending_img2, coords=new_coords, dims=new_dims
            )
            ds[Vars.ascending_img2].attrs = {
                Vars.attrs.std_name: Vars.name[Vars.ascending_img2],
                Vars.attrs.description: Vars.description[Vars.ascending_img2],
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[Vars.ascending_img2],
            }

            # ----------------------------------------------------------------
            # Apply chunking and compression settings
            # Use encoding from the existing cube as the golden standard
            # ----------------------------------------------------------------
            chunking_1d = ds[ImgPairInfo.date_dt].encoding[utils.OutputFormat.chunks]
            chunking_2d = (len(ds.y), len(ds.x))
            chunking_3d = ds[Vars.chip_size_height].encoding[utils.OutputFormat.chunks]
            compression_zarr = zarr.Blosc(
                cname="lz4", clevel=1, shuffle=zarr.Blosc.BITSHUFFLE
            )

            for each_var in ds:
                if utils.OutputFormat.chunks in ds[each_var].encoding:
                    ds_chunking = ds[each_var].encoding[utils.OutputFormat.chunks]
                    ndim = len(ds_chunking)

                    if ndim == 1:
                        chunking = chunking_1d
                    elif ndim == 2:
                        chunking = chunking_2d
                    elif ndim == 3:
                        chunking = chunking_3d
                    else:
                        chunking = ds_chunking  # passthrough for unexpected dims

                    ds[each_var].encoding[utils.OutputFormat.chunks] = chunking
                    ds[each_var].encoding[utils.OutputFormat.compressor] = compression_zarr

            # Change datatype for M11 and M12 to float32 in encoding
            ds[Vars.m11].encoding[utils.OutputFormat.dtype] = np.float32
            ds[Vars.m12].encoding[utils.OutputFormat.dtype] = np.float32
            ds[Vars.m11].encoding[utils.Missing.name] = utils.Missing.value
            ds[Vars.m12].encoding[utils.Missing.name] = utils.Missing.value

            ds[Vars.ascending_img1].encoding[utils.OutputFormat.dtype] = np.ubyte
            ds[Vars.ascending_img2].encoding[utils.OutputFormat.dtype] = np.ubyte

            msgs.append(f"Saving datacube to {fixed_file}")
            logging.info(msgs[-1])

            ds.to_zarr(fixed_file, consolidated=True)

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