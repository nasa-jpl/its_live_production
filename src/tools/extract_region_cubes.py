"""
Script to extract datacubes for the region of interest. It builds a list of
datacubes to be used for the annual mosaics creation for a particular region.

It accepts geojson file with datacube definitions:

python ./extract_region_cubes.py -c ../aws/regions/catalog_v02_regions.json -o HMA_datacubes.json --region HMA

python ./extract_region_cubes.py -c ../aws/regions/catalog_v02_rgi.geojson -o Greenland_datacubes.json --rgi_code 5
"""
import argparse
import json
import logging
import sys
import os

from itscube_types import CubeJson

if __name__ == '__main__':

    # Set up logging
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--cube_file',
        type=str,
        action='store',
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '-o', '--output_file',
        type=str,
        action='store',
        default=None,
        required=True,
        help="Output file to store extracted datacubes for the region of interest [%(default)s]"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--region',
        type=str,
        action='store',
        default=None,
        help="Region to extract datacubes for [%(default)s]"
    )
    group.add_argument(
        '--rgi_code',
        type=str,
        action='store',
        default=None,
        help="JSON list of RGI codes to extract datacubes for [%(default)s]."
    )
    group.add_argument(
        '--region_id',
        type=str,
        action='store',
        default=None,
        help="Region ID (introduced in V2 data) to extract datacubes for [%(default)s]"
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    cubes_to_generate = []

    attr_name = None
    attr_value = None
    if args.region:
        # Region ID is provided: previous catalog had "region" attribute
        # currently replaced with "region_id" (there is no M_ID or RGI_CODE
        # attributes anymore)
        attr_name = CubeJson.REGION
        attr_value = args.region
        logging.info(f"Generating list for {attr_name}: {attr_value}")

    # elif args.region_id:
    #     # Region ID is provided: introduced in final V2 catalog
    #     attr_name = CubeJson.REGION_ID
    #     attr_value = args.region_id
    #     logging.info(f"Generating list for {attr_name}: {attr_value}")

    # elif args.rgi_code:
    #     # RGI code is provided
    #     attr_name = CubeJson.RGI_CODE
    #     attr_value = json.loads(args.rgi_code)
    #     logging.info(f"Generating list for {attr_name}: {attr_value}")

    logging.info(f'Filtering for {attr_name }')
    with open(args.cube_file, 'r') as fhandle:
        cubes = json.load(fhandle)

        logging.info(f'Total number of datacubes: {len(cubes["features"])}')
        for each_cube in cubes[CubeJson.FEATURES]:
            # Example of data cube definition in "aws/regions/catalog_v02_regions.json: file
            # { "type": "Feature",
            #   "properties": {
            #       "fill-opacity": 0.98486645558583574,
            #       "fill": "red", "roi_percent_coverage": 1.5133544414164224,
            #       "data_epsg": "EPSG:32718",
            #       "geometry_epsg": {
            #           "type": "Polygon",
            #           "coordinates": [ [ [ 400000, 4400000 ], [ 500000, 4400000 ], [ 500000, 4500000 ], [ 400000, 4500000 ], [ 400000, 4400000 ] ] ]
            #       },
            #       "datacube_exist": 1,
            #       "zarr_url": "http://its-live-data.s3.amazonaws.com/datacubes/v02/S50W070/ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr",
            #       "region": "PAT"
            #   },
            #   "geometry": { "type": "Polygon", "coordinates": [ [ [ -76.411339, -50.54338 ], [ -75.0, -50.551932 ], [ -75.0, -49.652543 ], [ -76.385169, -49.644257 ], [ -76.411339, -50.54338 ] ] ] }
            # }

            # Example of cube definition from aws/update_V2_datacubes_2025/catalog_v02.1_region_id.json
            # { "type": "Feature",
            # "properties": {
            #   "fill-opacity": 0.9594690217023184,
            #   "fill": "red",
            #   "roi_percent_coverage": 4.0530978297681619,
            #   "epsg": 3413,
            #   "geometry_epsg": {
            #       "type": "Polygon",
            #       "coordinates": [ [ [ -2800000, 600000 ], [ -2700000, 600000 ], [ -2700000, 700000 ], [ -2800000, 700000 ], [ -2800000, 600000 ] ] ]
            #   },
            #   "datacube_exist": 1,
            #   "zarr_url": "http://its-live-data.s3.amazonaws.com/datacubes/v2-updated-october2024/N60W140/ITS_LIVE_vel_EPSG3413_G0120_X-2750000_Y650000.zarr",
            #   "granule_count": 14890,
            #   "region_name": "Alaska",
            #   "region_id": "RGI01A" },
            #   "geometry": {
            #       "type": "Polygon",
            #       "coordinates": [ [ [ -147.094757, 64.002998 ], [ -147.528808, 64.862384 ], [ -149.534455, 64.656283 ], [ -149.036243, 63.804526 ], [ -147.094757, 64.002998 ] ] ]
            #    }
            # },
            # OR
            # can specify RGI_CODE as provided in aws/regions/catalog_v02_rgi.geojson under "propertites"
            # "RGI_CODE": 5

            # Start the Batch job for each cube with ROI != 0
            properties = each_cube[CubeJson.PROPERTIES]

            if attr_name in properties:
                if attr_name and \
                        ((isinstance(attr_value, list) and properties[attr_name] in attr_value) or \
                        (not isinstance(attr_value, list) and properties[attr_name] == attr_value)):
                    cubes_to_generate.append(properties[CubeJson.URL])
                    logging.info(
                        f'{properties[CubeJson.URL]}: '
                        f'region={properties[attr_name]}'
                    )

    logging.info(f'Number of cubes for {attr_name}={attr_value}: {len(cubes_to_generate)}')

    logging.info(f'Writing found datacubes to {args.output_file}')

    with open(args.output_file, 'w') as fh:
        json.dump(cubes_to_generate, fh, indent=3)

    logging.info(f"Done")
