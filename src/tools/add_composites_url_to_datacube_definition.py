"""
Script to add existing composites URLs to the datacube catalog geojson file.

python ./add_composites_url_to_datacube_definition.py -g ../tools/catalog_v2.1.json
    -o catalog_composites_v2.1.json
"""
import argparse
import json
import logging
import os
import sys
import s3fs

from itslive_mosaics_types import GeoJsonVars


HTTP_PREFIX = 'https://its-live-data.s3.amazonaws.com'
S3_PREFIX = 's3://its-live-data'


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-g', '--cubeCatalog',
        type=str,
        action='store',
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '-a', '--compositesDir',
        type=str,
        action='store',
        default='https://its-live-data.s3.amazonaws.com/composites/annual/v2-updated-september2025',
        help="S3 URL to the composites [%(default)s]."
    )
    parser.add_argument(
        '-c', '--cubesDir',
        type=str,
        action='store',
        default='https://its-live-data.s3.amazonaws.com/datacubes/v2-updated-october2024',
        help="S3 URL to the composites [%(default)s]."
    )
    parser.add_argument(
        '-o', '--targetCatalog',
        type=str,
        action='store',
        default='catalog_with_composites_v2.1.json',
        help="GeoJson file that stores existing composites URLs [%(default)s]."
    )
    parser.add_argument(
        '--updateGranuleCountOnly',
        action='store_true',
        help='Update the number of granules for each cube in the catalog'
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f'Parser arguments: {args}')

    s3 = s3fs.S3FileSystem(anon=True)

    with open(args.cubeCatalog, 'r') as fhandle:
        cubes = json.load(fhandle)

        num_existing_cubes = len(cubes[GeoJsonVars.features])
        num_existing_composites = 0

        for each_cube in cubes[GeoJsonVars.features]:
            properties = each_cube[GeoJsonVars.properties]

            if args.updateGranuleCountOnly:
                cube_url = properties[GeoJsonVars.url]
                # Update number of granules in existing cube if it changed since
                # some of the cubes were updated with new granules after the
                # catalog was created.
                # Write number of granules for existing cube
                cube_s3_url_meta = os.path.join(
                    cube_url.replace(HTTP_PREFIX, S3_PREFIX),
                    '.zmetadata'
                )

                # Open the cube's metadata to get number of
                # granules
                with s3.open(cube_s3_url_meta, 'r') as fh:
                    meta = json.load(fh)
                    each_cube[GeoJsonVars.properties][GeoJsonVars.granule_count] = \
                        meta['metadata']['mid_date/.zarray']['shape'][0]

                    logging.info(
                        f"Number of granules: "
                        f"{each_cube[GeoJsonVars.properties][GeoJsonVars.granule_count]} "
                        f"for {cube_url}"
                    )

                # No need to update anything else in the catalog
                continue

            # Replace HTTP URL with HTTPS - fixes access using python and
            # julia
            properties[GeoJsonVars.url] = properties[GeoJsonVars.url].replace(
                'http://', 'https://'
            )

            cube_path, cube_filename = os.path.split(properties[GeoJsonVars.url])

            logging.info(f'Cube URL original: {properties[GeoJsonVars.url]}')
            logging.info(f'Cube split: {cube_path=} {cube_filename=}')

            # ITS_LIVE_vel_EPSG32717_G0120_X750000_Y9850000.zarr

            # Format composite name
            composite_url = cube_path.replace(args.cubesDir, args.compositesDir)
            logging.info(f'Composite URL {composite_url=}')

            # ITS_LIVE_velocity_EPSG3031_120m_X-50000_Y-950000.zarr
            composite_filename = cube_filename.replace('_vel_', '_velocity_')
            composite_filename = composite_filename.replace('_G0120_', '_120m_')
            logging.info(f'Composite file {composite_filename=}')

            composite_url = os.path.join(composite_url, composite_filename)
            logging.info(f'Composite path {composite_url=}')

            s3_composite_url = composite_url.replace(HTTP_PREFIX, S3_PREFIX)
            logging.info(f'Cube composite name: {s3_composite_url}')

            # Check if composite exists in S3 bucket as not all composites
            # are most likely generated
            if not s3.exists(s3_composite_url):
                logging.info(
                    f"Composite {s3_composite_url} does not exist, setting "
                    "path to an empty string "
                )
                composite_url = ''

            else:
                num_existing_composites += 1

            properties[GeoJsonVars.composite_url] = composite_url

        logging.info(f'Number of catalog cubes {num_existing_cubes=}')
        logging.info(f'Number of catalog composites {num_existing_composites=}')

        # Write updated cube information to the new json file
        logging.info(f"Writing updated catalog to the {args.targetCatalog}...")
        with open(args.targetCatalog, 'w') as output_fhandle:
            json.dump(cubes, output_fhandle, indent=4)

    logging.info("Done")
