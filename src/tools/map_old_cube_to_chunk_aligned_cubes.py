"""
Find chunk-aligned (new) datacubes that overlap a given old-format ITS_LIVE
datacube.

Given a target "old" 100x100km datacube (identified by its 'zarr_url' from
the old-format catalog, e.g.
catalog_with_composites_updateNumGranules_v2.1.26_2026.json) and the chunk-
aligned "new" datacube catalog (e.g.
datacube_catalog_chunk_aligned_July21_no.y.min.json), report every new cube
whose bounding box overlaps the old cube's bounding box (same EPSG), ranked
by overlap area (largest first).

Usage example:

python find_new_cubes_per_old_cube.py \
    --old-catalog catalog_with_composites_updateNumGranules_v2.1.26_2026.json \
    --new-catalog datacube_catalog_chunk_aligned_July21_no.y.min.json \
    --old-cube ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr
"""
import argparse
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def load_catalog(path):
    """Load a GeoJSON FeatureCollection catalog from disk."""
    with open(path) as fh:
        return json.load(fh)


def get_bbox(feature):
    """Extract a feature's projected bounding box.

    Both the old-format and chunk-aligned catalogs store an axis-aligned
    rectangle in 'properties.geometry_epsg' (cell-corner coordinates in the
    cube's own EPSG projection).

    Returns
    -------
    tuple
        (epsg, x_min, x_max, y_min, y_max)
    """
    coords = feature['properties']['geometry_epsg']['coordinates'][0]
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    epsg = feature['properties']['epsg']
    return epsg, min(x_coords), max(x_coords), min(y_coords), max(y_coords)


def find_old_cube(old_catalog, cube_identifier):
    """Find the old-catalog feature whose 'zarr_url' contains
    cube_identifier (a datacube filename or distinctive substring of it).

    Parameters
    ----------
    old_catalog : dict
        Old-format catalog GeoJSON FeatureCollection.
    cube_identifier : str
        Substring to match against each feature's 'zarr_url' (typically the
        datacube's *.zarr filename).

    Returns
    -------
    dict
        The matching GeoJSON feature.
    """
    matches = [
        feature for feature in old_catalog['features']
        if cube_identifier in feature['properties'].get('zarr_url', '')
    ]

    if len(matches) == 0:
        raise RuntimeError(f"No old cube found matching '{cube_identifier}'")

    if len(matches) > 1:
        urls = [m['properties']['zarr_url'] for m in matches]
        raise RuntimeError(
            f"Multiple old cubes match '{cube_identifier}': {urls}"
        )

    return matches[0]


def overlap_area(bbox_a, bbox_b):
    """Return the overlap area of two axis-aligned bounding boxes.

    Parameters
    ----------
    bbox_a, bbox_b : tuple
        (x_min, x_max, y_min, y_max), in the same projected units.

    Returns
    -------
    float
        Overlap area in squared projected units (e.g. m^2), or 0.0 if the
        boxes don't overlap.
    """
    a_x_min, a_x_max, a_y_min, a_y_max = bbox_a
    b_x_min, b_x_max, b_y_min, b_y_max = bbox_b

    overlap_x = max(0.0, min(a_x_max, b_x_max) - max(a_x_min, b_x_min))
    overlap_y = max(0.0, min(a_y_max, b_y_max) - max(a_y_min, b_y_min))

    return overlap_x * overlap_y


def find_overlapping_new_cubes(old_bbox, old_epsg, new_catalog):
    """Find every new-catalog cube that overlaps the old cube's bounding box.

    Parameters
    ----------
    old_bbox : tuple
        (x_min, x_max, y_min, y_max) of the target old cube.
    old_epsg : int
        EPSG code of the target old cube. Only new cubes with the same EPSG
        are considered.
    new_catalog : dict
        Chunk-aligned catalog GeoJSON FeatureCollection.

    Returns
    -------
    list of dict
        One entry per overlapping new cube, sorted by 'overlap_area'
        (largest first). Each entry has 'cube_id', 'epsg', bounding box
        fields, 'overlap_area', and 'overlap_percent_of_old_cube' (percent
        of the *old* cube's area covered by this new cube).
    """
    old_x_min, old_x_max, old_y_min, old_y_max = old_bbox
    old_area = (old_x_max - old_x_min) * (old_y_max - old_y_min)

    results = []

    for feature in new_catalog['features']:
        epsg, x_min, x_max, y_min, y_max = get_bbox(feature)

        if epsg != old_epsg:
            continue

        area = overlap_area(old_bbox, (x_min, x_max, y_min, y_max))
        if area <= 0.0:
            continue

        results.append({
            'cube_id': feature['properties']['cube_id'],
            'epsg': epsg,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'overlap_area': area,
            'overlap_percent_of_old_cube': 100.0 * area / old_area
        })

    results.sort(key=lambda r: r['overlap_area'], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Find chunk-aligned (new) datacubes overlapping a given old-'
            'format ITS_LIVE datacube, ranked by overlap area.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:
        python %(prog)s \\
            --old-catalog catalog_with_composites_updateNumGranules_v2.1.26_2026.json \\
            --new-catalog datacube_catalog_chunk_aligned_July21_no.y.min.json \\
            --old-cube ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr
        """
    )
    parser.add_argument(
        '--old-catalog',
        type=str,
        required=True,
        help='Path to the old-format datacube catalog GeoJSON file.'
    )
    parser.add_argument(
        '--new-catalog',
        type=str,
        required=True,
        help='Path to the chunk-aligned (new) datacube catalog GeoJSON file.'
    )
    parser.add_argument(
        '--old-cube',
        type=str,
        required=True,
        help="Target old datacube's filename (or distinctive substring of "
             "its 'zarr_url'), e.g. "
             "ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional path to write the overlapping new cubes as JSON.'
    )

    args = parser.parse_args()

    old_catalog = load_catalog(args.old_catalog)
    new_catalog = load_catalog(args.new_catalog)

    old_feature = find_old_cube(old_catalog, args.old_cube)
    old_epsg, old_x_min, old_x_max, old_y_min, old_y_max = get_bbox(old_feature)
    old_bbox = (old_x_min, old_x_max, old_y_min, old_y_max)

    logging.info(f"Target old cube: {old_feature['properties']['zarr_url']}")
    logging.info(
        f"EPSG: {old_epsg}, bounds: X=[{old_x_min}, {old_x_max}], "
        f"Y=[{old_y_min}, {old_y_max}]"
    )

    overlapping = find_overlapping_new_cubes(old_bbox, old_epsg, new_catalog)

    if not overlapping:
        logging.info("No overlapping new cubes found.")
        return

    total_percent = sum(
        cube['overlap_percent_of_old_cube'] for cube in overlapping
    )

    logging.info(
        f"Found {len(overlapping)} overlapping new cube(s) "
        f"(total coverage: {total_percent:.1f}% of old cube area):"
    )
    for rank, cube in enumerate(overlapping, start=1):
        logging.info(
            f"  {rank}. {cube['cube_id']}: "
            f"X=[{cube['x_min']}, {cube['x_max']}], "
            f"Y=[{cube['y_min']}, {cube['y_max']}], "
            f"overlap_area={cube['overlap_area']:.0f} m^2 "
            f"({cube['overlap_percent_of_old_cube']:.1f}% of old cube)"
        )

    if args.output:
        with open(args.output, 'w') as fh:
            json.dump(
                {
                    'old_cube': old_feature['properties']['zarr_url'],
                    'old_cube_bbox': {
                        'epsg': old_epsg,
                        'x_min': old_x_min,
                        'x_max': old_x_max,
                        'y_min': old_y_min,
                        'y_max': old_y_max
                    },
                    'overlapping_new_cubes': overlapping
                },
                fh,
                indent=2
            )
        logging.info(f"Wrote results to {args.output}")


if __name__ == '__main__':
    main()
