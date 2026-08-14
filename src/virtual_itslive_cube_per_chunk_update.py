"""
Open, inspect, and update an existing virtual ITS_LIVE datacube stored in icechunk repository.

This script can:
1. Inspect an existing virtual datacube (read-only mode)
2. Update an existing virtual datacube with new granules (update mode)

The update workflow:
- Queries searchAPI for granules in the datacube's polygon
- Loads skipped granules JSON file for the icechunk store
- Filters out already-processed granules (in cube's granule_url variable)
- Filters out previously-skipped granules
- Updates the cube with remaining new granules

Usage examples:

# Inspect from S3
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store s3://its-live-data/test-space/virtual-cubes/icechunk/my_cube.icechunk

# Inspect from local filesystem
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store my_cube.icechunk

# Update from searchAPI (S3 or local)
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store s3://its-live-data/test-space/virtual-cubes/icechunk/my_cube.icechunk \
    --use-searchAPI \
    --start-date 2024-01-01 \
    --end-date 2024-12-31

# Update from granules file (S3 or local)
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store my_cube.icechunk \
    --granules-file new_granules.json
"""
import argparse
import logging
import xarray as xr
import icechunk as ic
from datetime import datetime
import json
import numpy as np
import pyproj
import s3fs
import boto3
from dateutil.parser import parse

from itscube_types import (
    CubeFormat,
    ImgPairInfo,
    Vars,
    SkippedGranules
)
import utils
import itslive_utils

# Import functions from the creation script
from virtual_itslive_cube_per_chunk import (
    load_granules,
    build_virtual_cube_subset,
    HTTPS_URL,
    S3_URL,
    PIXEL_SIZE_HALF,
    LON_LAT_PROJECTION
)
from virtual_itslive_cube import _drop_nonfinite_attrs
from obstore.store import S3Store

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress Zarr V3 unstable string dtype warnings
import warnings
warnings.filterwarnings('ignore', message='.*UnstableSpecificationWarning.*')

# ManifestArray is used to distinguish granule-backed virtual variables (whose
# dtype must not be relabeled) from in-memory numpy-backed variables during the
# dtype-matching step in the update workflow.
from virtualizarr.manifests import ManifestArray

# Constants
PIXEL_SIZE = 120  # meters


def open_virtual_cube(cube_store_path):
    """Open an existing virtual datacube from icechunk repository.

    Parameters
    ----------
    cube_store_path : str
        Path to icechunk repository. Can be:
        - S3 path: 's3://bucket/prefix/cube.icechunk'
        - Local path: 'path/to/cube.icechunk'

    Returns
    -------
    tuple of (xr.Dataset, ic.Repository)
        The opened datacube and its repository handle.
    """
    is_s3 = cube_store_path.startswith('s3://')

    if is_s3:
        # Parse S3 path
        s3_parts = cube_store_path.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ''

        logging.info(f'Opening icechunk repo from S3: bucket={bucket}, prefix={prefix}')

        # Open S3 storage
        storage = ic.s3_storage(
            bucket=bucket,
            prefix=prefix,
            region="us-west-2"
        )
        repo = ic.Repository.open(storage)
    else:
        logging.info(f'Opening icechunk repo from local filesystem: {cube_store_path}')
        repo = ic.Repository.open(ic.local_filesystem_storage(cube_store_path))

    # Read datacube from main branch.
    # mask_and_scale=False preserves raw on-disk dtypes (int16 stays int16
    # rather than being CF-decoded to float32 with NaN fills), reflecting the
    # true virtual representation of the cube.
    cube = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False
    )

    return cube, repo


def print_cube_info(cube):
    """Print comprehensive information about the virtual datacube.

    Parameters
    ----------
    cube : xr.Dataset
        The datacube to inspect.
    """
    print("\n" + "="*80)
    print("VIRTUAL DATACUBE INFORMATION")
    print("="*80)

    # Basic structure
    print("\n--- STRUCTURE ---")
    print(f"Dimensions: {dict(cube.sizes)}")
    print(f"Number of time layers: {len(cube.time)}")
    print(
        f"Spatial extent: x=[{cube.x.values[0]:.1f}, {cube.x.values[-1]:.1f}], "
        f"y=[{cube.y.values[0]:.1f}, {cube.y.values[-1]:.1f}]"
    )

    # Data variables
    print("\n--- DATA VARIABLES ---")
    velocity_vars = [v for v in [Vars.v, Vars.vx, Vars.vy, Vars.vr, Vars.va] if v in cube.data_vars]
    print(f"Velocity variables ({len(velocity_vars)}): {velocity_vars}")

    m_vars = [v for v in [Vars.m11, Vars.m12] if v in cube.data_vars]
    print(f"M variables ({len(m_vars)}): {m_vars}")

    # Count velocity attribute variables
    error_vars = [v for v in cube.data_vars if 'error' in v]
    shift_vars = [v for v in cube.data_vars if 'stable_shift' in v]
    print(f"Error attribute variables: {len(error_vars)}")
    print(f"Stable shift attribute variables: {len(shift_vars)}")

    # img_pair_info variables
    img_pair_vars = []
    for attr in ImgPairInfo.all:
        if attr in cube.data_vars:
            img_pair_vars.append(attr)
    print(f"Image pair info variables ({len(img_pair_vars)}): {img_pair_vars[:5]}...")

    print(f"\nTotal data variables: {len(cube.data_vars)}")

    # Global attributes
    print("\n--- GLOBAL ATTRIBUTES ---")
    key_attrs = [
        CubeFormat.date_created,
        CubeFormat.datacube_software_version,
        utils.OutputFormat.projection,
        utils.OutputFormat.latitude,
        utils.OutputFormat.longitude,
        utils.OutputFormat.s3,
        utils.OutputFormat.url,
        SkippedGranules.name
    ]

    for attr in key_attrs:
        if attr in cube.attrs:
            value = cube.attrs[attr]
            # Truncate long values
            if isinstance(value, str) and len(value) > 60:
                value = value[:60] + "..."
            print(f"{attr}: {value}")

    # Time range
    print("\n--- TIME COVERAGE ---")
    time_values = cube.time.values
    print(f"First layer: {time_values[0]}")
    print(f"Last layer: {time_values[-1]}")

    # Sensor breakdown
    if 'mission_img1' in cube.data_vars:
        missions = cube.mission_img1.values
        unique_missions = set(str(m) for m in missions)
        print(f"\nMissions represented: {sorted(unique_missions)}")

        # Count by sensor type
        s1_count = sum(1 for m in missions if str(m).startswith('S1'))
        s2_count = sum(1 for m in missions if str(m).startswith('S2'))
        landsat_count = sum(1 for m in missions if str(m).startswith('L'))
        print(f"Sentinel-1 (radar): {s1_count} layers")
        print(f"Sentinel-2 (optical): {s2_count} layers")
        print(f"Landsat (optical): {landsat_count} layers")

    # Data types
    print("\n--- DATA TYPES ---")
    for var in [Vars.v, Vars.vx, Vars.vy, Vars.vr, Vars.va, Vars.m11, Vars.m12]:
        if var in cube.data_vars:
            print(f"{var}: {cube[var].dtype}")

    print("\n" + "="*80)
    print("\nFull Dataset:")
    print(cube)
    print("="*80 + "\n")


def skipped_granules_path(cube_store):
    """Get path to skipped granules JSON file for a given cube store.

    Parameters
    ----------
    cube_store : str
        Path to icechunk repository (S3 or local).

    Returns
    -------
    str
        Path to skipped granules JSON file.
    """
    return cube_store.rstrip('/').rstrip('.icechunk') + '_skippedGranules.json'


def load_skipped_granules(cube_store):
    """Load previously skipped granules from JSON file.

    Parameters
    ----------
    cube_store : str
        Path to icechunk repository (S3 or local).

    Returns
    -------
    set of str
        Set of skipped granule URLs (normalized to s3:// form).
    """
    skipped_path = skipped_granules_path(cube_store)
    is_s3 = skipped_path.startswith('s3://')

    try:
        if is_s3:
            # Read from S3 using boto3
            s3_client = boto3.client('s3', region_name='us-west-2')
            s3_parts = skipped_path.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1] if len(s3_parts) > 1 else ''

            response = s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            skipped = json.loads(content)

        else:
            # Read from local filesystem
            with open(skipped_path, 'r') as f:
                skipped = json.load(f)

        # Normalize to S3 URL format
        skipped_set = set(url.replace(HTTPS_URL, S3_URL) for url in skipped)
        logging.info(f'Loaded {len(skipped_set)} previously skipped granules from {skipped_path}')

        return skipped_set

    except (FileNotFoundError, boto3.exceptions.botocore.exceptions.ClientError) as e:
        # Empty skipped granules granules will exist for any existing
        # virtual cube
        raise RuntimeError(f'No existing skipped granules file at {skipped_path}')


def get_existing_granule_urls(cube):
    """Extract existing granule URLs from a datacube.

    Parameters
    ----------
    cube : xr.Dataset
        The virtual datacube.

    Returns
    -------
    set of str
        Set of granule URLs already in the cube (normalized to s3:// form).
    """
    urls = set(str(u).replace(HTTPS_URL, S3_URL) for u in cube[Vars.url].values)
    logging.info(f'Found {len(urls)} existing granules in cube')

    return urls


def filter_new_granules(all_urls, skipped, existing):
    """Filter granule list to only new granules.

    Parameters
    ----------
    all_urls : list of str
        All candidate granule URLs.
    skipped : set of str
        Previously skipped granule URLs.
    existing : set of str
        Granule URLs already in the cube.

    Returns
    -------
    list of str
        Filtered list of new granules to process.
    """
    # Normalize to S3 format and filter
    filtered = []
    for url in all_urls:
        url_s3 = url.replace(HTTPS_URL, S3_URL)

        # Skip P000 granules
        if url_s3.endswith('P000.nc'):
            continue

        # Skip if previously skipped or already in cube
        if url_s3 in skipped:
            logging.debug(f'Skipping previously-skipped granule: {url_s3}')
            continue

        if url_s3 in existing:
            logging.debug(f'Skipping existing granule: {url_s3}')
            continue

        filtered.append(url_s3)

    logging.info(f'After filtering: {len(filtered)} new granules to process')
    return filtered


def open_repo_for_update(cube_store):
    """Open an existing icechunk repository for updating.

    Parameters
    ----------
    cube_store : str
        Path to icechunk repository (S3 or local).

    Returns
    -------
    tuple of (ic.Repository, xr.Dataset)
        The repository handle and current datacube.
    """
    is_s3 = cube_store.startswith('s3://')
    url_prefix = "s3://its-live-data/"

    if is_s3:
        # Parse S3 path
        s3_parts = cube_store.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        prefix = s3_parts[1] if len(s3_parts) > 1 else ''

        logging.info(f'Opening icechunk repo from S3 for update: bucket={bucket}, prefix={prefix}')

        # Configure storage settings for stronger recovery
        storage_settings = ic.StorageSettings(
            unsafe_use_metadata=True,
            unsafe_use_conditional_update=True
        )

        config = ic.RepositoryConfig.default()
        config.storage = storage_settings
        config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(url_prefix, ic.s3_store(region="us-west-2", anonymous=True))
        )

        storage = ic.s3_storage(
            bucket=bucket,
            prefix=prefix,
            region="us-west-2"
        )

        repo = ic.Repository.open(
            storage=storage,
            config=config,
            authorize_virtual_chunk_access=ic.containers_credentials(
                {url_prefix: ic.s3_credentials(anonymous=True)}
            ),
        )
    else:
        logging.info(f'Opening icechunk repo from local filesystem for update: {cube_store}')

        config = ic.RepositoryConfig.default()
        config.set_virtual_chunk_container(
            ic.VirtualChunkContainer(
                url_prefix,
                ic.s3_store(region="us-west-2", anonymous=False)
            )
        )

        repo = ic.Repository.open(
            storage=ic.local_filesystem_storage(cube_store),
            config=config,
            authorize_virtual_chunk_access=ic.containers_credentials(
                {url_prefix: ic.s3_credentials(anonymous=True)}
            ),
        )

    # Read current datacube.
    # mask_and_scale=False is REQUIRED: it disables CF decoding so integer
    # variables (int16 with scale_factor/_FillValue) keep their raw on-disk
    # dtype instead of being promoted to float32 with NaN fills. The new cube
    # is built from raw int16 ManifestArrays, so the existing cube must be read
    # raw too for the append dtypes to match (and to reflect the true virtual
    # representation).
    cube = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False
    )

    return repo, cube


def save_skipped_granules(cube_store, skipped_granules):
    """Save skipped granules list to JSON file.

    Parameters
    ----------
    cube_store : str
        Path to icechunk repository (S3 or local).
    skipped_granules : list of str
        List of skipped granule URLs (in https:// form).
    """
    skipped_path = skipped_granules_path(cube_store)
    is_s3 = skipped_path.startswith('s3://')

    if is_s3:
        # Write to S3 using boto3
        s3_client = boto3.client('s3', region_name='us-west-2')
        s3_parts = skipped_path.replace('s3://', '').split('/', 1)
        bucket = s3_parts[0]
        key = s3_parts[1] if len(s3_parts) > 1 else ''

        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(skipped_granules, indent=2),
            ContentType='application/json'
        )
    else:
        # Write to local filesystem
        with open(skipped_path, 'w') as f:
            json.dump(skipped_granules, f, indent=2)

    logging.info(f'Saved {len(skipped_granules)} skipped granules to {skipped_path}')


def main():
    parser = argparse.ArgumentParser(
        description="""
        Open, inspect, and update an existing virtual ITS_LIVE datacube from icechunk repository.

        This script can:
        1. Inspect an existing virtual datacube (read-only mode)
        2. Update an existing virtual datacube with new granules (update mode)

        Usage examples:

        # Inspect from S3
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store s3://its-live-data/test-space/virtual-cubes/icechunk/my_cube.icechunk

        # Inspect from local filesystem
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store my_cube.icechunk

        # Update from searchAPI
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store my_cube.icechunk \\
            --use-searchAPI \\
            --start-date 2024-01-01 \\
            --end-date 2024-12-31

        # Update from granules file
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store my_cube.icechunk \\
            --granules-file new_granules.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--cube-store',
        type=str,
        required=True,
        help='Path to icechunk repository (S3 or local). '
            'S3 format: s3://bucket/prefix/cube.icechunk'
    )

    # Mutually exclusive group for granules input (optional - if not provided, inspect only)
    granules_group = parser.add_mutually_exclusive_group(required=False)
    granules_group.add_argument(
        "--granules-file",
        type=str,
        help="Path to JSON file containing a list of granule paths (for update mode)"
    )
    granules_group.add_argument(
        "--use-searchAPI",
        action='store_true',
        default=False,
        help="Use searchAPI to get list of granules (for update mode)"
    )

    parser.add_argument(
        "--start-date",
        type=lambda s: parse(s).strftime('%Y-%m-%d'),
        default='1982-01-01',
        help="Start date for searchAPI query (required with --use-searchAPI) [%(default)s]"
    )
    parser.add_argument(
        "--end-date",
        type=lambda s: parse(s).strftime('%Y-%m-%d'),
        default=datetime.now().strftime('%Y-%m-%d'),
        help="End date for searchAPI query (required with --use-searchAPI) [%(default)s]"
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
        '--show-variables',
        action='store_true',
        help='Show complete list of all data variables'
    )

    parser.add_argument(
        '--show-attributes',
        action='store_true',
        help='Show all global attributes'
    )

    args = parser.parse_args()

    # Determine if this is an update operation or inspect-only
    update_mode = args.granules_file or args.use_searchAPI

    try:
        if update_mode:
            # ============================================================
            # UPDATE MODE: Add new granules to existing cube
            # ============================================================
            logging.info("UPDATE MODE: Adding new granules to existing cube")

            # Open existing repo and cube
            repo, cube = open_repo_for_update(args.cube_store)
            logging.info(f"Opened existing cube with {len(cube.time)} time layers")

            # Extract polygon and projection from cube attributes
            proj_polygon = json.loads(cube.attrs[CubeFormat.proj_polygon])
            geo_polygon = json.loads(cube.attrs[CubeFormat.geo_polygon])
            projection = cube.attrs[utils.OutputFormat.projection]

            logging.info(f"Cube projection: {projection}")

            # Compute bbox from proj_polygon (adjust to cell centers)
            x_coords = [coord[0] for coord in proj_polygon]
            y_coords = [coord[1] for coord in proj_polygon]
            xmin = min(x_coords) + PIXEL_SIZE_HALF
            xmax = max(x_coords) - PIXEL_SIZE_HALF
            ymin = min(y_coords) + PIXEL_SIZE_HALF
            ymax = max(y_coords) - PIXEL_SIZE_HALF
            bbox = [xmin, xmax, ymin, ymax]

            logging.info(f"Bbox for cropping: {bbox}")

            # Get candidate granules
            if args.granules_file:
                logging.info(f"Loading granules from {args.granules_file}")
                with open(args.granules_file, 'r') as f:
                    granules = json.load(f)

            elif args.use_searchAPI:
                # Use searchAPI with ROI from cube attributes
                roi = {
                    "type": "Polygon",
                    "coordinates": [geo_polygon]
                }

                logging.info(f"Querying searchAPI for {projection} from {args.start_date} to {args.end_date}")
                granules = itslive_utils.serverless_search(
                    epsg_code=projection,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    roi=roi
                )

            logging.info(f"Found {len(granules)} candidate granules")

            # Apply num_granules limit if specified
            if args.num_granules > 0:
                granules = granules[:args.num_granules]
                logging.info(f'Leaving {args.num_granules} to process')

            # Filter granules
            skipped_set = load_skipped_granules(args.cube_store)
            existing_set = get_existing_granule_urls(cube)
            new_granules = filter_new_granules(
                granules,
                skipped_set,
                existing_set
            )

            if not new_granules:
                logging.info("No new granules to process - cube is up to date")
                return

            logging.info(f"Processing {len(new_granules)} new granules")

            # Build cube from new granules
            netcdf_store = S3Store(
                bucket="its-live-data",
                region="us-west-2",
                skip_signature=True,
            )

            vds_list = load_granules(new_granules, args.bucket)
            logging.info(f'Loaded {len(vds_list)} virtual datasets')

            new_cube, autorift_param_file, run_skipped = build_virtual_cube_subset(
                vds_list, bbox, netcdf_store
            )

            if new_cube is None:
                logging.warning("No valid data in new granules - no update performed")
                # Still update skipped granules file
                all_skipped = list(skipped_set.union(set(s.replace(HTTPS_URL, S3_URL) for s in run_skipped)))
                all_skipped_https = [s.replace(S3_URL, HTTPS_URL) for s in all_skipped]
                save_skipped_granules(args.cube_store, all_skipped_https)

                return

            logging.info(f"Built cube from {len(new_cube.time)} new granules")

            # Match dtypes with existing cube before appending.
            #
            # Two distinct cases:
            #  1. In-memory (numpy-backed) variables -- granule_url, mission_img*,
            #     etc. These are real arrays built in build_virtual_cube, so
            #     astype() genuinely re-allocates and converts. Safe to convert
            #     (after a truncation guard for fixed-length strings).
            #  2. Granule-backed ManifestArray variables -- v, vx, vy, M11, etc.
            #     Their chunk bytes are encoded on-disk for the granule's dtype
            #     (int16). Relabeling the metadata dtype does NOT re-encode the
            #     bytes, so a genuine dtype change (e.g. int16 -> float32) would
            #     decode garbage. If these don't already match, the existing cube
            #     was built with incompatible dtype definitions and must be
            #     regenerated -- fail loudly rather than silently corrupt.
            for var_name in new_cube.data_vars:
                if var_name not in cube.data_vars:
                    continue

                existing_dtype = cube[var_name].dtype
                new_dtype = new_cube[var_name].dtype
                if existing_dtype == new_dtype:
                    continue

                is_manifest = isinstance(new_cube[var_name].data, ManifestArray)

                # String <-> string mismatch (fixed-length UTF32): numpy-backed,
                # safe to astype after guarding against truncation.
                if np.issubdtype(existing_dtype, np.str_) and np.issubdtype(new_dtype, np.str_):
                    if existing_dtype.kind == 'U':
                        existing_max_len = existing_dtype.itemsize // 4  # UTF32: 4 bytes/char
                        new_values = new_cube[var_name].values
                        max_new_len = max((len(str(v)) for v in new_values.flat), default=0)
                        if max_new_len > existing_max_len:
                            raise ValueError(
                                f"Cannot append: {var_name} has values longer ({max_new_len} chars) "
                                f"than existing cube's fixed-length string ({existing_max_len} chars). "
                                f"This would truncate data. Please regenerate the existing cube with "
                                f"larger string lengths (Vars.stringType / ImgPairInfo.stringType)."
                            )
                    logging.debug(f"Converting {var_name} dtype from {new_dtype} to {existing_dtype}")
                    new_cube[var_name] = new_cube[var_name].astype(existing_dtype)
                    continue

                # Non-string mismatch.
                if is_manifest:
                    raise ValueError(
                        f"Cannot append {var_name}: existing cube stores it as {existing_dtype}, "
                        f"but the new virtual (granule-backed) data is {new_dtype}. The on-disk "
                        f"chunk bytes are encoded for {new_dtype}, so relabeling to {existing_dtype} "
                        f"would decode incorrectly. The existing cube was built with incompatible "
                        f"dtype definitions and must be regenerated to match the current code."
                    )

                logging.debug(f"Converting {var_name} dtype from {new_dtype} to {existing_dtype}")
                new_cube[var_name] = new_cube[var_name].astype(existing_dtype)

            # Append to existing cube (dtypes are now compatible)
            session = repo.writable_session("main")
            new_cube_clean = _drop_nonfinite_attrs(new_cube)

            try:
                logging.info("Attempting to append new granules with append_dim='time'")
                new_cube_clean.vz.to_icechunk(session.store, append_dim="time")
                snapshot_id = session.commit(f"its_live virtual cube subset: append {len(new_cube.time)} new granules")
                logging.info(f"Successfully appended {len(new_cube.time)} granules, snapshot: {snapshot_id}")

            except TypeError as e:
                if "append_dim" in str(e):
                    raise RuntimeError(f"append_dim not supported by virtualizarr: {e}")
                else:
                    raise

            # Update skipped granules file
            all_skipped = list(skipped_set.union(set(s.replace(HTTPS_URL, S3_URL) for s in run_skipped)))
            all_skipped_https = [s.replace(S3_URL, HTTPS_URL) for s in all_skipped]
            save_skipped_granules(args.cube_store, all_skipped_https)

            # Verify and report (raw dtypes, matching how the cube was read for append)
            updated_cube = xr.open_zarr(
                repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
                mask_and_scale=False
            )
            logging.info(f"Update complete. Cube now has {len(updated_cube.time)} time layers")
            print_cube_info(updated_cube)

        else:
            # ============================================================
            # INSPECT MODE: Read-only inspection
            # ============================================================
            logging.info("INSPECT MODE: Read-only inspection")

            # Open the virtual datacube
            cube, repo = open_virtual_cube(args.cube_store)

            # Print comprehensive information
            print_cube_info(cube)

            # Optional: show all variables
            if args.show_variables:
                print("\n--- ALL DATA VARIABLES ---")
                for var in sorted(cube.data_vars):
                    dims = cube[var].dims
                    shape = cube[var].shape
                    dtype = cube[var].dtype
                    print(f"{var:40s} {str(dims):20s} {str(shape):20s} {dtype}")

            # Optional: show all attributes
            if args.show_attributes:
                print("\n--- ALL GLOBAL ATTRIBUTES ---")
                for attr in sorted(cube.attrs.keys()):
                    value = cube.attrs[attr]
                    # Truncate long values
                    if isinstance(value, str) and len(value) > 80:
                        value = value[:80] + "..."
                    print(f"{attr}: {value}")

            logging.info("Successfully opened and inspected virtual datacube")

    except Exception as e:
        logging.error(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
