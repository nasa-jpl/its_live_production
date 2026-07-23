#!/usr/bin/env python3
"""
Verify that chunk alignment is consistent between granules and datacubes.

This script:
1. Reads sample granules from test data directory
2. Finds overlapping datacube from the chunk-aligned catalog
3. Verifies EPSG codes match
4. Verifies x and y grid coordinates align

To run:
python ./verify_chunk_alignment_granules_datacubes.py ../datacube_catalog_chunk_aligned_July20.json --all
"""
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import grid module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import aws_utils
from test_real_chunk_alignment import (
    read_granule_metadata,
    find_overlapping_datacube
)
from chunk_aligned_utils import (
    get_alignment_info, CHUNK_SIZE, PIXEL_SIZE
)

CHUNK_LEN = CHUNK_SIZE * PIXEL_SIZE


GRANULES = [
    # 's3://its-live-data/velocity_image_pair/landsatOLI/v02/S80W170/LC08_L1GT_012122_20141028_20200910_02_T2_X_LC08_L1GT_012122_20141113_20201016_02_T2_G0120V02_P082.nc'
    's3://its-live-data/velocity_image_pair/landsatOLI/v02/S80W170/LC08_L1GT_020121_20231013_20231102_02_T2_X_LC09_L1GT_020121_20231106_20231106_02_T2_G0120V02_P084.nc',
    's3://its-live-data/velocity_image_pair/landsatOLI/v02/S80W170/LC08_L1GT_020120_20201121_20210315_02_T2_X_LC08_L1GT_020120_20210124_20210305_02_T2_G0120V02_P051.nc'
]

s3 = None

def check_coordinate_alignment(granule_meta, cube_meta, tolerance=0.00001):
    """
    Check if granule coordinates align with cube grid in the overlap region.

    The granule doesn't need to cover the entire datacube. We only check that
    in the overlapping region, the x and y coordinates match.

    Returns:
        dict with alignment results
    """
    # Convert to numpy arrays if they are xarray objects
    g_x = np.array(granule_meta['x'])
    g_y = np.array(granule_meta['y'])

    # Cube grid is already adjusted for the cell centers
    c_x_min, c_x_max = cube_meta['x_min'], cube_meta['x_max']
    c_y_min, c_y_max = cube_meta['y_min'], cube_meta['y_max']

    # Find overlap region
    overlap_x_min = max(granule_meta['x_min'], c_x_min)
    overlap_x_max = min(granule_meta['x_max'], c_x_max)
    overlap_y_min = max(granule_meta['y_min'], c_y_min)
    overlap_y_max = min(granule_meta['y_max'], c_y_max)

    # Filter granule coordinates to overlap region
    g_x_overlap = g_x[(g_x >= overlap_x_min) & (g_x <= overlap_x_max)]

    # print(f'{g_x_overlap=}')
    g_y_overlap = g_y[(g_y >= overlap_y_min) & (g_y <= overlap_y_max)]

    if len(g_x_overlap) == 0 or len(g_y_overlap) == 0:
        return {
            'overlap_exists': False,
            'x_in_overlap': 0,
            'y_in_overlap': 0,
        }

    cube_x = np.arange(c_x_min, c_x_max + 1, PIXEL_SIZE )
    cube_y = np.arange(c_y_min, c_y_max + 1, PIXEL_SIZE )
    print(f"{cube_x[:5]=}")
    print(f"{cube_y[:5]=}")

    cube_y = np.arange(c_y_min, c_y_max + 1, PIXEL_SIZE )

    # Check alignment IN OVERLAP REGION ONLY
    x_aligned = []
    x_not_aligned = []

    for gx in g_x_overlap:
        # Calculate distance to all cube grid points
        distances = cube_x - gx
        # Find minimum absolute distance
        min_abs_dist = np.abs(distances).min()
        if min_abs_dist < tolerance:
            x_aligned.append(gx)
        else:
            # Store: granule coord, nearest cube coord, signed distance
            # Positive distance: cube point is ahead of granule point
            # Negative distance: cube point is behind granule point
            min_idx = np.abs(distances).argmin()
            nearest_cube_x = cube_x[min_idx]
            signed_dist = distances[min_idx]
            x_not_aligned.append((gx, nearest_cube_x, signed_dist))

    y_aligned = []
    y_not_aligned = []

    for gy in g_y_overlap:
        # Calculate distance to all cube grid points
        distances = cube_y - gy
        # Find minimum absolute distance
        min_abs_dist = np.abs(distances).min()
        if min_abs_dist < tolerance:
            y_aligned.append(gy)
        else:
            # Store: granule coord, nearest cube coord, signed distance
            # Positive distance: cube point is ahead of granule point
            # Negative distance: cube point is behind granule point
            min_idx = np.abs(distances).argmin()
            nearest_cube_y = cube_y[min_idx]
            signed_dist = distances[min_idx]
            y_not_aligned.append((gy, nearest_cube_y, signed_dist))

    return {
        'overlap_exists': True,
        'overlap_region': {
            'x': [overlap_x_min, overlap_x_max],
            'y': [overlap_y_min, overlap_y_max]
        },
        'x_in_overlap': len(g_x_overlap),
        'x_aligned': len(x_aligned),
        'x_not_aligned': len(x_not_aligned),
        'x_not_aligned_samples': x_not_aligned[:5],  # First 5 examples
        'x_alignment_pct': 100 * len(x_aligned) / len(g_x_overlap) if len(g_x_overlap) > 0 else 0,
        'y_in_overlap': len(g_y_overlap),
        'y_aligned': len(y_aligned),
        'y_not_aligned': len(y_not_aligned),
        'y_not_aligned_samples': y_not_aligned[:5],
        'y_alignment_pct': 100 * len(y_aligned) / len(g_y_overlap) if len(g_y_overlap) > 0 else 0,
    }


def verify_granule(granule_path, catalog_path):
    """Verify chunk alignment for a single granule."""
    print(f"\n{'='*80}")
    print(f"Granule: {Path(granule_path).name}")
    print(f"{'='*80}")

    # Read granule metadata
    try:
        granule_meta = read_granule_metadata(granule_path, s3)

    except Exception as e:
        print(f"Error reading granule {granule_path.name}: {e}")
        return False

    print(f"\nGranule metadata:")
    print(f"  EPSG: {granule_meta['epsg']}")
    print(f"  X range: [{granule_meta['x_min']:.1f}, {granule_meta['x_max']:.1f}]")
    print(f"  Y range: [{granule_meta['y_min']:.1f}, {granule_meta['y_max']:.1f}]")
    print(f"  X points: {len(granule_meta['x'])}, spacing: {granule_meta['x_spacing']:.1f}m")
    print(f"  Y points: {len(granule_meta['y'])}, spacing: {granule_meta['y_spacing']:.1f}m")

    # Find overlapping datacubes - cube cooordinates are already adjusted for the
    # cell centers
    overlapping_cubes = find_overlapping_datacube(granule_meta, catalog_path)

    if not overlapping_cubes:
        print(f"\nNo overlapping datacube found!")
        return False

    print(f"\n✓ Found {len(overlapping_cubes)} overlapping datacube(s)")
    print(f"\nOverlapping datacubes:")
    for i, cube_meta in enumerate(overlapping_cubes, 1):
        print(f"  {i}. {cube_meta['cube_id']}")

    all_aligned = True

    for i, cube_meta in enumerate(overlapping_cubes, 1):
        print(f"\n{'='*80}")
        print(f"Datacube {i}: {cube_meta['cube_id']}")
        print(f"{'='*80}")
        print(f"  Catalog index: {cube_meta['catalog_index']}")
        print(f"  EPSG: {cube_meta['epsg']}")
        print(f"  Datacube bounds per cell centers:")
        print(f"    X: [{cube_meta['x_min']:.1f}, {cube_meta['x_max']:.1f}]")
        print(f"    Y: [{cube_meta['y_min']:.1f}, {cube_meta['y_max']:.1f}]")

        cube_x = cube_meta['x_max']-cube_meta['x_min']+PIXEL_SIZE
        cube_y = cube_meta['y_max']-cube_meta['y_min']+PIXEL_SIZE

        print(f"  Size: {cube_x:.1f}m × {cube_y:.1f}m (dims: {cube_x/CHUNK_LEN} x {cube_y/CHUNK_LEN})")

        # Display ROI coverage if available
        if cube_meta['roi_coverage'] is not None:
            print(f"  ROI coverage: {cube_meta['roi_coverage']:.2f}%")

        # Display polygon information
        print(f"\n  Datacube polygon (EPSG projection per catalog cell corners):")
        if cube_meta['epsg_polygon']:
            epsg_coords = cube_meta['epsg_polygon']['coordinates'][0]
            print(f"    Coordinates (X, Y):")
            for i, coord in enumerate(epsg_coords):
                print(f"      {i+1}. ({coord[0]:.1f}, {coord[1]:.1f})")

        # Check chunk alignment multiples FIRST (this is a fundamental property)
        print(f"\n  Checking chunk alignment (multiples of {CHUNK_SIZE * PIXEL_SIZE}m)...")
        chunk_alignment = verify_chunk_alignment_multiples(granule_meta, cube_meta)

        print(f"\n  Datacube chunk alignment:")
        print(f"    Original bounds: [{cube_meta['x_min']:.1f}, {cube_meta['y_min']:.1f}, {cube_meta['x_max']:.1f}, {cube_meta['y_max']:.1f}]")
        print(f"    Aligned bounds:  [{chunk_alignment['cube_aligned_bounds'][0]:.1f}, {chunk_alignment['cube_aligned_bounds'][1]:.1f}, {chunk_alignment['cube_aligned_bounds'][2]:.1f}, {chunk_alignment['cube_aligned_bounds'][3]:.1f}]")
        print(f"    Padding (pixels): {chunk_alignment['cube_padding']}")

        if chunk_alignment['cube_aligned']:
            print(f"    ✅ All datacube boundaries are chunk-aligned (multiples of {chunk_alignment['chunk_boundary']:.0f}m)")
        else:
            print(f"    FAIL: Datacube boundaries are NOT chunk-aligned (multiples of {chunk_alignment['chunk_boundary']:.0f}m))")

        print(f"\n  Granule chunk alignment:")
        print(f"    Original bounds: [{granule_meta['x_min']:.1f}, {granule_meta['y_min']:.1f}, {granule_meta['x_max']:.1f}, {granule_meta['y_max']:.1f}]")
        print(f"    Aligned bounds:  [{chunk_alignment['granule_aligned_bounds'][0]:.1f}, {chunk_alignment['granule_aligned_bounds'][1]:.1f}, {chunk_alignment['granule_aligned_bounds'][2]:.1f}, {chunk_alignment['granule_aligned_bounds'][3]:.1f}]")
        print(f"    Padding (pixels): {chunk_alignment['granule_padding']}")

        if chunk_alignment['c_vs_g_x_alignment'] == 0:
            print(f"    ✅ Granule chunk X boundaries are aligned with cube: {chunk_alignment['c_vs_g_x_alignment']:.0f}")
        else:
            print(f"    FAIL: Granule chunk X boundaries are NOT aligned with cube: {chunk_alignment['c_vs_g_x_alignment']:.0f}")

        if chunk_alignment['c_vs_g_y_alignment'] == 0:
            print(f"    ✅ Granule chunk Y boundaries are aligned with cube: {chunk_alignment['c_vs_g_y_alignment']:.0f}")
        else:
            print(f"    FAIL: Granule chunk Y boundaries are NOT aligned with cube: {chunk_alignment['c_vs_g_y_alignment']:.0f}")

        # Require chunk alignment for success
        if not (chunk_alignment['cube_aligned'] and chunk_alignment['granule_aligned']):
            print(f"\n  FAIL: CHUNK ALIGNMENT FAILED")
            all_aligned = False
            # Continue to show coordinate alignment anyway
        else:
            print(f"\n  ✅ CHUNK ALIGNMENT VERIFIED")

        # Check coordinate alignment
        alignment = check_coordinate_alignment(granule_meta, cube_meta)

        if not alignment['overlap_exists']:
            print(f"\n  ⚠ No coordinate overlap detected")
            all_aligned = False
            continue

        print(f"\n  Overlap region:")
        print(f"    X: [{alignment['overlap_region']['x'][0]:.1f}, {alignment['overlap_region']['x'][1]:.1f}]")
        print(f"    Y: [{alignment['overlap_region']['y'][0]:.1f}, {alignment['overlap_region']['y'][1]:.1f}]")

        print(f"\n  Coordinate alignment in overlap region:")
        print(f"    X: {alignment['x_aligned']}/{alignment['x_in_overlap']} points aligned ({alignment['x_alignment_pct']:.1f}%)")
        print(f"    Y: {alignment['y_aligned']}/{alignment['y_in_overlap']} points aligned ({alignment['y_alignment_pct']:.1f}%)")

        if alignment['x_not_aligned'] > 0:
            print(f"\n    X misalignment examples (granule_x, nearest_cube_x, signed_distance):")
            for granule_x, cube_x, dist in alignment['x_not_aligned_samples'][:3]:
                print(f"      {granule_x:.1f}m → {cube_x:.1f}m (distance: {dist:+.3f}m)")

        if alignment['y_not_aligned'] > 0:
            print(f"\n    Y misalignment examples (granule_y, nearest_cube_y, signed_distance):")
            for granule_y, cube_y, dist in alignment['y_not_aligned_samples'][:3]:
                print(f"      {granule_y:.1f}m → {cube_y:.1f}m (distance: {dist:+.3f}m)")

        if alignment['x_alignment_pct'] == 100 and alignment['y_alignment_pct'] == 100:
            print(f"\n  ✅ PERFECT ALIGNMENT - All coordinates in overlap region match!")

        elif alignment['x_alignment_pct'] >= 95 and alignment['y_alignment_pct'] >= 95:
            print(f"\n  ✓ Good alignment (>95%)")
        else:
            print(f"\n  FAIL: POOR ALIGNMENT - Coordinates do not match datacube grid")
            all_aligned = False

    return all_aligned


def verify_chunk_alignment_multiples(granule_meta, cube_meta):
    """
    Verify that aligned coordinates are multiples of chunk size.

    ITS_LIVE uses a standard grid with a 7.5m offset (half of Landsat 8 Band 8's 15m pixel size)
    applied via Grid.create(). This means coordinates are typically: N * pixel_size + 7.5m

    This test:
    1. Aligns datacube coordinates using get_alignment_info()
    2. Aligns granule coordinates using get_alignment_info()
    3. Verifies that aligned coordinates are at chunk boundaries (accounting for the 7.5m offset)

    Args:
        granule_meta: Dictionary with granule metadata (x_min, x_max, y_min, y_max)
        cube_meta: Dictionary with datacube metadata (x_min, x_max, y_min, y_max)
        chunk_size: Number of pixels per chunk (default: 512)
        pixel_size: Pixel size in meters (default: 120)

    Returns:
        dict with verification results including:
        - cube_aligned: bool, whether datacube boundaries are chunk-aligned
        - granule_aligned: bool, whether granule boundaries are chunk-aligned
        - chunk_boundary: float, the chunk boundary size in meters (61440m)
        - cube_misalignments: dict with any misalignment values
        - granule_misalignments: dict with any misalignment values
    """
    chunk_boundary = CHUNK_SIZE * PIXEL_SIZE  # 61440m
    grid_spacing = chunk_boundary

    # 1. Align datacube coordinates
    cube_aligned_bounds, cube_padding, _, _ = get_alignment_info(
        cube_meta['x_min'],
        cube_meta['y_min'],
        cube_meta['x_max'],
        cube_meta['y_max'],
        grid_spacing
    )
    cube_aligned = all(each == 0 for each in cube_padding)

    # 2. Align granule coordinates
    granule_aligned_bounds, granule_padding, _, _ = get_alignment_info(
        granule_meta['x_min'],
        granule_meta['y_min'],
        granule_meta['x_max'],
        granule_meta['y_max'],
        grid_spacing
    )

    granule_aligned = all(each == 0 for each in granule_padding)

    # Check for chunk alignment between the cube and granule
    c_vs_g_x_alignment = (cube_meta['x_min'] - granule_meta['x_min']) % chunk_boundary
    c_vs_g_y_alignment = (cube_meta['y_min'] - granule_meta['y_min']) % chunk_boundary

    return {
        'chunk_boundary': chunk_boundary,
        'cube_aligned': cube_aligned,
        'cube_aligned_bounds': cube_aligned_bounds,
        'cube_padding': cube_padding,
        'granule_aligned': granule_aligned,
        'granule_aligned_bounds': granule_aligned_bounds,
        'granule_padding': granule_padding,
        'c_vs_g_x_alignment': c_vs_g_x_alignment,
        'c_vs_g_y_alignment': c_vs_g_y_alignment
    }


def main():
    """Main verification script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify chunk alignment between granules and chunk-aligned datacube catalog',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Verify granules from S3 (default)
        python %(prog)s catalog.json

        # Verify granules from local directory
        python %(prog)s catalog.json --granule-dir data/chunkAlignedGranules

        # Verify specific granule URL
        python %(prog)s catalog.json --granule-url s3://bucket/path/to/granule.nc

        # Process all granules instead of just the first one
        python %(prog)s catalog.json --all
        """
    )

    parser.add_argument(
        'catalog',
        type=str,
        help='Path to chunk-aligned datacube catalog GeoJSON file'
    )

    parser.add_argument(
        '--granule-dir',
        type=str,
        default=None,
        help='Directory containing granule .nc files to verify (overrides default S3 granules)'
    )

    parser.add_argument(
        '--granule-url',
        type=str,
        default=None,
        help='Single granule S3 URL or local path to verify (overrides granule-dir and default S3 granules)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.001,
        help='Tolerance in meters for coordinate alignment checks [%(default)s]'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all granules instead of just the first one'
    )

    args = parser.parse_args()

    # Validate catalog path
    catalog_path = args.catalog
    if not Path(catalog_path).exists():
        print(f"Error: Catalog not found: {catalog_path}")
        return 1

    # Determine granule sources
    if args.granule_url:
        # Single granule URL specified
        granule_files = [args.granule_url]
        print(f"Verifying single granule: {args.granule_url}")

    elif args.granule_dir:
        # Local directory specified
        granule_dir = Path(args.granule_dir)
        if not granule_dir.exists():
            print(f"Error: Granule directory not found: {granule_dir}")
            return 1

        granule_files = list(granule_dir.glob("*.nc"))
        if not granule_files:
            print(f"Error: No .nc files found in {granule_dir}")
            return 1

        granule_files = [str(f) for f in sorted(granule_files)]
        print(f"Found {len(granule_files)} granule(s) in {granule_dir}")

    else:
        # Use default S3 granules from GRANULES constant
        granule_files = GRANULES
        print(f"Using default S3 granules: {len(granule_files)} granule(s)")

    print(f"Catalog: {catalog_path}")

    with open(catalog_path) as f:
        catalog = json.load(f)

    results = []
    # Process all granules or just the first one
    granules_to_process = granule_files if args.all else granule_files[:1]

    if not args.all and len(granule_files) > 1:
        print(f"Processing only the first granule (use --all to process all {len(granule_files)} granules)")

    for granule_file in granules_to_process:
        result = verify_granule(str(granule_file), catalog)
        # Extract filename for display
        granule_name = Path(granule_file).name if '/' in granule_file or '\\' in granule_file else granule_file
        results.append((granule_name, result))

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, result in results:
        status = "✓ PASS" if result else "FAIL: FAIL"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    s3 = aws_utils.make_s3fs()
    sys.exit(main())
