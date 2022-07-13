"""
Script to check for nan's in datacubes x and y coordinates.

It accepts geojson file with datacube definitions:

python ./check_cubes_dims_nan.py -c catalog_datacubes_v02_July12.2022.json -o datacubes_no_x_y_values.json
"""
import argparse
import json
import logging
import numpy as np
import s3fs
import sys
import xarray as xr

from itscube_types import CubeJson
from itscube import ITSCube

HTTP_PREFIX = 'http://'

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
        help="Output file to store identified datacubes [%(default)s]"
    )
    parser.add_argument(
        '-s', '--start_index',
        type=int,
        action='store',
        default=0,
        help="Start index into datacube list to process [%(default)s]"
    )

    logging.info(f"Command-line arguments: {sys.argv}")

    args = parser.parse_args()
    logging.info(f"Command args: {args}")

    found_cubes = []
    start_index = args.start_index

    s3_in = s3fs.S3FileSystem(anon=True)

    with open(args.cube_file, 'r') as fhandle:
        cubes = json.load(fhandle)

        logging.info(f'Total number of datacubes: {len(cubes["features"])}')
        for each_cube in cubes[CubeJson.FEATURES][start_index:]:
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

            cube_s3_url = properties[CubeJson.URL].replace(ITSCube.PATH_URL, '')
            cube_s3_url = cube_s3_url.replace(HTTP_PREFIX, ITSCube.S3_PREFIX)

            # Read X and Y values for the cube, and check if either one of them has NaN's
            store = s3fs.S3Map(root=cube_s3_url, s3=s3_in, check=False)

            logging.info(f"Checking {cube_s3_url}")
            with xr.open_dataset(store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
                if np.any(np.isnan(ds.x.values)):
                    logging.info(f'WARNING: Got nan in x: {cube_s3_url} ds.size={ds.sizes}')
                    found_cubes.append(cube_s3_url)

                if np.any(np.isnan(ds.y.values)):
                    logging.info(f'WARNING: Got nan in y: {cube_s3_url} ds.size={ds.sizes}')

                    if cube_s3_url not in found_cubes:
                        found_cubes.append(cube_s3_url)

    logging.info(f'Number of found cubes with nan x or y: {len(found_cubes)}')

    logging.info(f'Writing found datacubes to {args.output_file}')

    with open(args.output_file, 'w') as fh:
        json.dump(found_cubes, fh, indent=3)

    logging.info(f"Done")
