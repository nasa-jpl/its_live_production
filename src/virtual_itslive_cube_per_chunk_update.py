"""
Open and inspect an existing virtual ITS_LIVE datacube stored in icechunk repository.

This script reads a virtual datacube from an icechunk repository (local or S3),
displays its structure, and can be extended to update the datacube with new granules.

Usage examples:

# Open from S3
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store s3://its-live-data/test-space/virtual-cubes/icechunk/my_cube.icechunk

# Open from local filesystem
python virtual_itslive_cube_per_chunk_update.py \
    --cube-store my_cube.icechunk
"""
import argparse
import logging
import xarray as xr
import icechunk as ic
from datetime import datetime
import json

from itscube_types import (
    CubeFormat,
    ImgPairInfo,
    Vars,
    SkippedGranules
)
import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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

    # Read datacube from main branch
    cube = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        zarr_format=3
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
    print(f"Dimensions: {dict(cube.dims)}")
    print(f"Number of time layers: {len(cube.time)}")
    print(f"Spatial extent: x=[{cube.x.values[0]:.1f}, {cube.x.values[-1]:.1f}], "
          f"y=[{cube.y.values[0]:.1f}, {cube.y.values[-1]:.1f}]")

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
        CubeFormat.date_updated,
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


def main():
    parser = argparse.ArgumentParser(
        description="""
        Open and inspect an existing virtual ITS_LIVE datacube from icechunk repository.

        This script reads a virtual datacube stored in an icechunk repository (local or S3)
        and displays comprehensive information about its structure, variables, and metadata.

        Usage examples:

        # Open from S3
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store s3://its-live-data/test-space/virtual-cubes/icechunk/my_cube.icechunk

        # Open from local filesystem
        python virtual_itslive_cube_per_chunk_update.py \\
            --cube-store my_cube.icechunk
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

    try:
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
        logging.error(f"Failed to open virtual datacube: {e}")
        raise


if __name__ == "__main__":
    main()
