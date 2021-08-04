"""
Script to add existing datacube NetCDF format S3 URL paths to the global
datacube definition GeoJson file.

It accepts geojson file with datacube definitions and a list of existing NetCDF URLs.
"""
import json
import logging
import math
import os

from grid import Bounds
import itslive_utils


class CubeJson:
    """
    Variables names within GeoJson cube definition file.
    """
    FEATURES = 'features'
    PROPERTIES = 'properties'
    DATA_EPSG = 'data_epsg'
    GEOMETRY_EPSG = 'geometry_epsg'
    COORDINATES = 'coordinates'
    ROI_PERCENT_COVERAGE = 'roi_percent_coverage'
    EPSG_SEPARATOR = ':'
    EPSG_PREFIX = 'EPSG'
    NC_URL = 'nc_url'


class DataCubeGlobalDefinition:
    """
    Class to manage global datacube definition GeoJson file.

    TODO: Not to rely on external piece of information (list of NC URLs for
    datacubes), the sript can just "glob" for expected datacube S3 URL to check
    if it exists or not.
    """
    # HTTP URL for the datacube full path
    HTTP_PREFIX = ''

    FILENAME_PREFIX = 'ITS_LIVE_vel'
    MID_POINT_RESOLUTION = 50.0

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    # List of EPSG codes to generate datacubes for. If this list is empty,
    # then generate all ROI!=0 datacubes.
    EPSG_TO_UPDATE = []

    def __init__(self, grid_size: int):
        """
        Initialize object.
        """
        self.grid_size_str = f'{grid_size:04d}'
        self.grid_size = grid_size

    def __call__(self, cube_file: str, datacube_nc_file: str, output_file: str):
        """
        Insert datacube NetCDF S3 URL into datacube definition GeoJson and write
        it to provided output file.
        """
        # List of datacubes that had their URL updated
        num_cubes = 0

        # Read existing datacube URLs
        cubes_urls = None
        with open(datacube_nc_file, 'r') as fhandle:
            cubes_urls = [line.rstrip() for line in fhandle]

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to generate
            num_jobs = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')
            for each_cube in cubes[CubeJson.FEATURES]:
                # Example of data cube definition in json file
                # "properties": {
                #     "fill-opacity": 1.0,
                #     "fill": "red",
                #     "roi_percent_coverage": 0.0,
                #     "data_epsg": "EPSG:32701",
                #     "geometry_epsg": {
                #         "type": "Polygon",
                #         "coordinates": [
                #             [
                #                 [
                #                     100000,
                #                     7100000
                #                 ],
                #                 [
                #                     200000,
                #                     7100000
                #                 ],
                #                 [
                #                     200000,
                #                     7200000
                #                 ],
                #                 [
                #                     100000,
                #                     7200000
                #                 ],
                #                 [
                #                     100000,
                #                     7100000
                #                 ]
                #             ]
                #         ]
                #     }
                # }

                # Start the Batch job for each cube with ROI != 0
                properties = each_cube[CubeJson.PROPERTIES]

                roi = properties[CubeJson.ROI_PERCENT_COVERAGE]
                if roi != 0.0:
                    # Format filename for the cube
                    epsg = properties[CubeJson.DATA_EPSG].replace(CubeJson.EPSG_SEPARATOR, '')
                    # Extract int EPSG code
                    epsg_code = epsg.replace(CubeJson.EPSG_PREFIX, '')

                    if len(DataCubeGlobalDefinition.EPSG_TO_UPDATE) and \
                       epsg_code not in DataCubeGlobalDefinition.EPSG_TO_UPDATE:
                        continue

                    coords = properties[CubeJson.GEOMETRY_EPSG][CubeJson.COORDINATES][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int((x_bounds.min + x_bounds.max)/2)
                    mid_y = int((y_bounds.min + y_bounds.max)/2)

                    # Get mid point to the nearest 50
                    logging.info(f"Mid point: x={mid_x} y={mid_y}")
                    mid_x = int(math.floor(mid_x/DataCubeGlobalDefinition.MID_POINT_RESOLUTION)*DataCubeGlobalDefinition.MID_POINT_RESOLUTION)
                    mid_y = int(math.floor(mid_y/DataCubeGlobalDefinition.MID_POINT_RESOLUTION)*DataCubeGlobalDefinition.MID_POINT_RESOLUTION)
                    logging.info(f"Mid point at {DataCubeGlobalDefinition.MID_POINT_RESOLUTION}: x={mid_x} y={mid_y}")

                    # Convert to lon/lat coordinates to format s3 bucket path
                    # for the datacube
                    mid_lon_lat = itslive_utils.transform_coord(
                        epsg_code,
                        DataCubeGlobalDefinition.LON_LAT_PROJECTION,
                        mid_x, mid_y
                    )

                    cube_filename = f"{DataCubeGlobalDefinition.FILENAME_PREFIX}_{epsg}_G{self.grid_size_str}_X{mid_x}_Y{mid_y}.nc"
                    logging.info(f'Cube name: {cube_filename}')

                    cube_url = [each for each in cubes_urls if cube_filename in each]
                    if len(cube_url):
                        # The datacube NetCDF exists, update GeoJson
                        each_cube[CubeJson.PROPERTIES][CubeJson.NC_URL] = cube_url[0]
                        num_cubes += 1

            logging.info(f"Number of updated entries: {num_cubes}")

            # Write job info to the json file
            logging.info(f"Writing updated datacube info to the {output_file}...")
            with open(output_file, 'w') as output_fhandle:
                json.dump(cubes, output_fhandle, indent=4)

            return

def main(
    cube_definition_file: str,
    datacube_nc_file: str,
    grid_size: int,
    output_file: str):
    """
    Driver to update general datacube definition file with existing
    datacube NetCDF S3 URLs.
    """
    update_urls = DataCubeGlobalDefinition(grid_size)
    update_urls(cube_definition_file, datacube_nc_file, output_file)


if __name__ == '__main__':
    import argparse
    import warnings
    import sys
    warnings.filterwarnings('ignore')

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
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '-n', '--ncFile',
        type=str,
        action='store',
        default=None,
        help="File with NetCDF S3 URLs for existing datacubes [%(default)s]"
    )
    parser.add_argument(
        '-g', '--gridSize',
        type=int,
        action='store',
        default=120,
        help="Grid size for the data cube [%(default)d]"
    )
    parser.add_argument(
        '-o', '--outputFile',
        type=str,
        action='store',
        default='datacube_batch_jobs.json',
        help="File to capture updated general datacube definition information [%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to generate [%(default)s]"
    )

    args = parser.parse_args()

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        DataCubeGlobalDefinition.EPSG_TO_UPDATE = epsg_codes

    main(
        args.cubeDefinitionFile,
        args.ncFile,
        args.gridSize,
        args.outputFile)

    logging.info(f"Done")
