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

from itscube_types import CubeJson

if __name__ == '__main__':

    # Set up logging
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
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

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    cubes_to_generate = []

    attr_name = None
    attr_value = None
    if args.region:
        # Region ID is provided
        attr_name = CubeJson.REGION
        attr_value = args.region
        logging.info(f"Generating list for {attr_name}: {attr_value}")

    elif args.rgi_code:
        # RGI code is provided
        attr_name = CubeJson.RGI_CODE
        attr_value = json.loads(args.rgi_code)
        logging.info(f"Generating list for {attr_name}: {attr_value}")

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

    logging.info(f'Number of cubes for {attr_name}={attr_value}: {len(cubes_to_generate)}')

    logging.info(f'Writing found datacubes to {args.output_file}')

    with open(args.output_file, 'w') as fh:
        json.dump(cubes_to_generate, fh, indent=3)

    logging.info(f"Done")
