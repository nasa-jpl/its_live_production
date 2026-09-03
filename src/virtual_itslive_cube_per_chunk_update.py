"""
Open, inspect, and update an existing virtual ITS_LIVE datacube stored in
icechunk repository.

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
import os
import time
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
import virtual_itslive_cube_per_chunk
from virtual_itslive_cube_per_chunk import (
    load_granules,
    build_virtual_cube_subset,
    skipped_granules_path,
    save_skipped_granules,
    HTTPS_URL,
    S3_URL,
    P000_SUFFIX,
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
    # zarr_format=3, not 2: icechunk repos are natively Zarr V3 metadata;
    # forcing zarr_format=2 raises GroupNotFoundError (no .zgroup/.zarray
    # markers exist in an icechunk store).
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

    except FileNotFoundError:
        # Empty skipped granules file will exist for any existing virtual cube
        raise RuntimeError(f'No existing skipped granules file at {skipped_path}')

    except boto3.exceptions.botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code in ('NoSuchKey', '404'):
            # Object genuinely doesn't exist -- empty skipped granules file
            # will exist for any existing virtual cube
            raise RuntimeError(f'No existing skipped granules file at {skipped_path}')

        # Any other ClientError (permission denied, wrong region, throttling,
        # expired credentials, bucket typo, etc.) is a real problem -- don't
        # mask it behind a misleading "file not found" message
        raise


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
    tuple of (list of str, list of str)
        - Filtered list of new granules to process.
        - P000 granule URLs excluded from `all_urls` (never carry usable
          data), so the caller can record them in the persistent
          skipped-granules JSON alongside `skipped`/`existing`-filtered ones.
    """
    # Normalize to S3 format and filter
    filtered = []
    p000_granules = []
    for url in all_urls:
        url_s3 = url.replace(HTTPS_URL, S3_URL)

        # Skip P000 granules -- excluded from processing, but recorded by the
        # caller so the persistent skipped-granules JSON reflects every
        # granule considered, not just the ones that made it past this filter.
        if url_s3.endswith(P000_SUFFIX):
            p000_granules.append(url_s3)
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
    if p000_granules:
        logging.info(f'Excluding {len(p000_granules)} P000 new granules')

    return filtered, p000_granules


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
                ic.s3_store(region="us-west-2", anonymous=True)
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
    # zarr_format=3, not 2: icechunk repos are natively Zarr V3 metadata;
    # forcing zarr_format=2 raises GroupNotFoundError (no .zgroup/.zarray
    # markers exist in an icechunk store).
    cube = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        zarr_format=3,
        mask_and_scale=False
    )

    return repo, cube


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
        '--searchType',
        choices=['serverless', 'pgstac'],
        default='serverless',
        help='Granule search backend: "serverless" queries the geoparquet '
            'warehouse via duckdb (default), "pgstac" queries the STAC API '
            'via pystac_client [%(default)s].'
    )
    parser.add_argument(
        '--stacCatalog',
        type=str,
        default=None,
        help='Granule catalog location override. For serverless: s3:// path to '
            'the geoparquet warehouse (default: itslive warehouse). For pgstac: '
            'https:// URL of the STAC API (default: https://stac.itslive.cloud).'
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
        '--batch-size',
        type=int,
        default=10000,
        help='Number of new granules to load and commit together per icechunk '
            'snapshot [%(default)d]. New granules are sorted chronologically '
            'first, then split into batches of this size to bound memory use '
            'for very large update runs; each batch appends to the existing cube.'
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

    start_time = time.time()

    # load_granules/build_virtual_cube_subset (imported from
    # virtual_itslive_cube_per_chunk) read their pool size from that module's
    # own MAX_AWS_CONNECTIONS global, not from any argument -- set it there so
    # --threads actually takes effect instead of always using that module's
    # hardcoded default.
    virtual_itslive_cube_per_chunk.MAX_AWS_CONNECTIONS = args.threads

    # This shares the same joblib/loky-based load_granules/
    # build_virtual_cube_subset machinery as virtual_itslive_cube_per_chunk.py
    # (imported above), so a large multi-batch update run can hit the same
    # benign/self-remediating "resource_tracker: There appear to be N leaked
    # folder objects to clean up at shutdown" UserWarning. Must be set before
    # any Parallel() call below: the resource_tracker's own subprocess is
    # launched via a fresh `sys.executable` invocation (not fork()), so it
    # re-reads PYTHONWARNINGS from the environment at its own startup.
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::UserWarning:multiprocessing.resource_tracker,"
        "ignore::UserWarning:joblib.externals.loky.backend.resource_tracker"
    )

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

            granules = []

            # Get candidate granules
            if args.granules_file:
                logging.info(f"Loading granules from {args.granules_file}")
                with open(args.granules_file, 'r') as f:
                    granules = json.load(f)

            elif args.use_searchAPI:
                # Validate that other arguments are provided when using searchAPI
                if not args.start_date or not args.end_date:
                    parser.error(
                        "--use-searchAPI requires --start-date, --end-date arguments"
                        " (and optional --searchType, --stacCatalog arguments)"
                    )

                itslive_utils.STAC_CATALOG = args.stacCatalog
                itslive_utils.SEARCH_TYPE = args.searchType

                # Use searchAPI with ROI from cube attributes
                roi = {
                    "type": "Polygon",
                    "coordinates": [geo_polygon]
                }

                # TODO: might need to switch to use earthcatalog
                # granules = itslive_utils.earthcatalog_search(
                granules = itslive_utils.serverless_search(
                    epsg_code=projection,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    # polygon=roi
                    roi=roi
                )

            logging.info(f"Found {len(granules)} candidate granules")

            # Apply num_granules limit if specified
            if args.num_granules > 0:
                granules = granules[:args.num_granules]
                logging.info(f'Leaving {args.num_granules} to process')

            # Filter granules: returned sets have s3:// url's
            skipped_set = load_skipped_granules(args.cube_store)
            existing_set = get_existing_granule_urls(cube)
            new_granules, p000_granules = filter_new_granules(
                granules,
                skipped_set,
                existing_set
            )

            if not new_granules:
                logging.info("No new granules to process - cube is up to date")

                # Only record newly-discovered P000 granules here -- unioning
                # the raw `granules` candidate list would also record granules
                # already present in the cube (existing_set) as "skipped",
                # even though they're genuinely in the cube.
                all_skipped = list(skipped_set.union(p000_granules))
                save_skipped_granules(args.cube_store, all_skipped)
                return

            logging.info(f"Processing {len(new_granules)} new granules")

            if p000_granules:
                logging.info(
                    f'Adding {len(p000_granules)} P000.nc new granules to '
                    'skipped granules'
                )
                skipped_set = skipped_set.union(p000_granules)

            # Sort new granules chronologically by mid_date parsed from the
            # filename -- no granule is opened for this, so it's cheap even
            # for very large lists. Matches virtual_itslive_cube_per_chunk.py's
            # ordering so batches are appended in time order.
            new_granules = sorted(new_granules, key=utils.extract_mid_date_from_url)

            batch_size = args.batch_size
            batches = [
                new_granules[i:i + batch_size]
                for i in range(0, len(new_granules), batch_size)
            ]
            num_batches = len(batches)

            # Ties every commit made by this run together (see commit_metadata
            # below), so a large update run's history can be identified as one
            # logical update.
            batch_job_id = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

            logging.info(
                f"Split into {num_batches} batch(es) of up to {batch_size} "
                f"granules [batch_job_id={batch_job_id}]"
            )

            netcdf_store = S3Store(
                bucket="its-live-data",
                region="us-west-2",
                skip_signature=True,
            )

            total_appended = 0

            for batch_num, batch_granules in enumerate(batches, start=1):
                logging.info(
                    f'Batch {batch_num}/{num_batches}: loading '
                    f'{len(batch_granules)} granules'
                )

                vds_list, missing_granules = load_granules(batch_granules, args.bucket)
                if missing_granules:
                    logging.warning(
                        f'Batch {batch_num}/{num_batches}: {len(missing_granules)} '
                        'granules reported by searchAPI are missing from S3 (known '
                        'catalog issue) -- skipping'
                    )
                    skipped_set = skipped_set.union(
                        set(s.replace(HTTPS_URL, S3_URL) for s in missing_granules)
                    )
                logging.info(
                    f'Batch {batch_num}/{num_batches}: loaded '
                    f'{len(vds_list)} virtual datasets'
                )

                new_cube, autorift_param_file, run_skipped = build_virtual_cube_subset(
                    vds_list, bbox, netcdf_store
                )

                # Record whatever this batch skipped regardless of whether it
                # produced a cube, so the persistent skipped-granules JSON
                # reflects every granule considered across the whole run.
                skipped_set = skipped_set.union(
                    set(s.replace(HTTPS_URL, S3_URL) for s in run_skipped)
                )

                if new_cube is None:
                    logging.warning(
                        f"Batch {batch_num}/{num_batches}: no valid data in "
                        "this batch - nothing appended"
                    )
                    save_skipped_granules(args.cube_store, list(skipped_set))
                    continue

                logging.info(
                    f"Batch {batch_num}/{num_batches}: built cube from "
                    f"{len(new_cube.time)} new granules"
                )

                # Variable-set consistency check: the dtype-matching loop below only
                # walks new_cube.data_vars, so a variable present in one cube but not
                # the other would silently skip past it and append_dim="time" would
                # then try to stack two datasets with different schemas. Fail loudly
                # here instead -- this usually means the cube-building code changed
                # between when the existing cube was built and now, and the existing
                # cube needs to be regenerated to match.
                #
                # Only compare *time-indexed* variables: static, cube-level
                # variables with no 'time' dimension (mapping, landice,
                # floatingice) are set once at cube creation and never touched
                # again on update -- new_cube legitimately never carries them, and
                # their already-committed chunks stay valid across every later
                # icechunk snapshot without needing to match new_cube's schema.
                existing_vars = {v for v in cube.data_vars if 'time' in cube[v].dims}
                new_vars = {v for v in new_cube.data_vars if 'time' in new_cube[v].dims}
                if existing_vars != new_vars:
                    raise ValueError(
                        f"Cannot append: new cube's data variables differ from the "
                        f"existing cube's.\n  missing from new cube: {existing_vars - new_vars}\n"
                        f"  extra in new cube: {new_vars - existing_vars}"
                    )

                # x/y grid-identity check: `bbox` above is re-derived each run from
                # the existing cube's stored proj_polygon attribute, so grid
                # alignment between new_cube and the existing cube is only implicit.
                # Verify it explicitly (mirrors _assert_identical_grids() in
                # virtual_itslive_cube_per_chunk.py, which does the analogous check
                # across granules within a single build) so a bbox-derivation drift
                # (e.g. float round-tripping through JSON) fails loudly here instead
                # of surfacing as a low-level zarr/icechunk error, or worse, a silent
                # spatial misalignment.
                existing_x, new_x = cube.x.values, new_cube.x.values
                existing_y, new_y = cube.y.values, new_cube.y.values
                if existing_x.shape != new_x.shape or not np.array_equal(existing_x, new_x):
                    raise ValueError(
                        f"Cannot append: new cube's x grid does not match the existing "
                        f"cube's. existing: {existing_x.shape} spanning "
                        f"[{existing_x[0]}, {existing_x[-1]}], new: {new_x.shape} "
                        f"spanning [{new_x[0]}, {new_x[-1]}]."
                    )
                if existing_y.shape != new_y.shape or not np.array_equal(existing_y, new_y):
                    raise ValueError(
                        f"Cannot append: new cube's y grid does not match the existing "
                        f"cube's. existing: {existing_y.shape} spanning "
                        f"[{existing_y[0]}, {existing_y[-1]}], new: {new_y.shape} "
                        f"spanning [{new_y[0]}, {new_y[-1]}]."
                    )

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

                # Ties every commit from this run together so they can be
                # identified as one logical update (see icechunk
                # repo.ops_log()/ancestry()), matching
                # virtual_itslive_cube_per_chunk.py's creation-batch metadata.
                commit_metadata = {
                    "batch_job_id": batch_job_id,
                    "batch_index": batch_num,
                    "total_batches": num_batches,
                    "batch_size": len(batch_granules),
                }

                try:
                    logging.info(
                        f"Batch {batch_num}/{num_batches}: attempting to append "
                        "new granules with append_dim='time'"
                    )
                    new_cube_clean.vz.to_icechunk(session.store, append_dim="time")
                    snapshot_id = session.commit(
                        f"its_live virtual cube subset: append {len(new_cube.time)} "
                        f"new granules (batch {batch_num}/{num_batches})",
                        metadata=commit_metadata
                    )
                    logging.info(
                        f"Batch {batch_num}/{num_batches}: successfully appended "
                        f"{len(new_cube.time)} granules, snapshot: {snapshot_id}"
                    )

                except TypeError as e:
                    if "append_dim" in str(e):
                        raise RuntimeError(f"append_dim not supported by virtualizarr: {e}")
                    else:
                        raise

                total_appended += len(new_cube.time)

                # Save skipped granules to JSON file after every committed
                # batch, so progress survives a mid-run failure on a later
                # batch.
                save_skipped_granules(args.cube_store, list(skipped_set))

            if total_appended == 0:
                logging.info("No batch produced any appended data - no update performed")
                return

            # Verify and report (raw dtypes, matching how the cube was read for append)
            # zarr_format=3, not 2: icechunk repos are natively Zarr V3
            # metadata; forcing zarr_format=2 raises GroupNotFoundError.
            updated_cube = xr.open_zarr(
                repo.readonly_session("main").store,
                consolidated=False,
                zarr_format=3,
                mask_and_scale=False
            )
            logging.info(
                f"Update complete. Cube now has {len(updated_cube.time)} time layers "
                f"({total_appended} appended across {num_batches} batch(es))"
            )
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

    finally:
        elapsed_time = time.time() - start_time
        logging.info(f'Total runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)')


if __name__ == "__main__":
    main()

    # Mirrors virtual_itslive_cube_per_chunk.py's end-of-run cleanup: this
    # script drives the same joblib/loky-based load_granules/
    # build_virtual_cube_subset machinery (imported above), so a large
    # multi-batch update run can hit the same background-thread/tokio-runtime
    # race with Python's interpreter finalization ("Error in sys.excepthook"
    # with a blank "Original exception was:" body) after everything has
    # already completed and committed successfully. Only reached if main()
    # returned normally (it re-raises on failure), so this never masks a
    # genuine error with a bypassed exit(0).
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor().shutdown(wait=True, kill_workers=True)
    time.sleep(0.5)  # let resource_tracker's unregister messages land
    os._exit(0)
