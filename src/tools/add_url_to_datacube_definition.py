"""
Script to add existing datacube Zarr format S3 URL paths to the global
datacube definition GeoJson file.

It accepts geojson file with datacube definitions and S3 bucket and top level
path of existing datacubes.
"""
import copy
import json
import logging
import math
import os
from pathlib import Path
import s3fs

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
    URL = 'zarr_url'
    EXIST_FLAG = 'datacube_exist'


class DataCubeGlobalDefinition:
    """
    Class to manage global datacube definition GeoJson file.

    TODO: Not to rely on external piece of information (list of NC URLs for
    datacubes), the sript can just "glob" for expected datacube S3 URL to check
    if it exists or not.
    """
    FILENAME_PREFIX = 'ITS_LIVE_vel'
    MID_POINT_RESOLUTION = 50.0

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    # List of EPSG codes to generate datacubes for. If this list is empty,
    # then generate all ROI!=0 datacubes.
    EPSG_TO_UPDATE = []

    # Filename to save found datacubes URLs to
    FOUND_CUBES_FILE = 'found_cubes.json'

    AWS_PREFIX = 'its-live-data'
    URL_PREFIX = 'http://its-live-data.s3.amazonaws.com'

    # Path of datacubes top level directory in S3 bucket
    CUBES_S3_PATH = None

    # Flag to disable reduced catalog geojson
    DISABLE_REDUCED_CATALOG = False

    # List of datacube filenames to include into catalog.
    CUBES_TO_INCLUDE = []

    def __init__(self, grid_size: int):
        """
        Initialize object.
        """
        self.grid_size_str = f'{grid_size:04d}'
        self.grid_size = grid_size

        # Collect existing datacubes in Zarr format
        s3_out = s3fs.S3FileSystem(anon=True)

        self.all_cubes = []
        for each in s3_out.ls(DataCubeGlobalDefinition.CUBES_S3_PATH):
            cubes = s3_out.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]
            self.all_cubes.extend(cubes)

        if len(self.all_cubes):
            # Write down all found datacubes to the file
            with open(DataCubeGlobalDefinition.FOUND_CUBES_FILE , 'w') as outfile:
                json.dump(self.all_cubes, outfile, indent=4)

        self.all_cubes = [each.replace(DataCubeGlobalDefinition.AWS_PREFIX, DataCubeGlobalDefinition.URL_PREFIX) for each in self.all_cubes]
        logging.info(f'Number of datacubes in Zarr format: {len(self.all_cubes)}')

    def __call__(self, cube_file: str, output_file: str):
        """
        Insert datacube Zarr S3 URL into datacube definition GeoJson and write
        it to provided output file.
        """
        # List of datacubes that had their URL updated
        num_cubes = 0

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to generate
            num_jobs = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')

            if output_file is None:
                # If output file is not provided, just report how many datacubes exist
                return

            # Output only cubes that have Zarr store created for them.
            output_cubes = cubes

            # If need to create reduced catalog of datacubes
            if not DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG:
                output_cubes = copy.deepcopy(cubes)
                output_cubes[CubeJson.FEATURES] = []

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

                # Default: datacube does not exist
                each_cube[CubeJson.PROPERTIES][CubeJson.EXIST_FLAG] = 0

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

                    cube_filename = f"{DataCubeGlobalDefinition.FILENAME_PREFIX}_{epsg}_G{self.grid_size_str}_X{mid_x}_Y{mid_y}.zarr"
                    logging.info(f'Cube name: {cube_filename}')

                    if cube_filename not in DataCubeGlobalDefinition.CUBES_TO_INCLUDE:
                        logging.info(f'Skipping cube filename: {cube_filename}')
                        continue

                    cube_url = [each for each in self.all_cubes if cube_filename in each]
                    if len(cube_url):
                        # The datacube in Zarr format exists, update GeoJson
                        each_cube[CubeJson.PROPERTIES][CubeJson.URL] = cube_url[0]
                        each_cube[CubeJson.PROPERTIES][CubeJson.EXIST_FLAG] = 1
                        num_cubes += 1

                        # If constructing reduced catalog, append cube to the result catalog
                        if not DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG:
                            output_cubes[CubeJson.FEATURES].append(each_cube)

            logging.info(f"Number of updated entries: {num_cubes}")

            # Write job info to the json file
            logging.info(f"Writing updated datacube info to the {output_file}...")
            with open(output_file, 'w') as output_fhandle:
                json.dump(output_cubes, output_fhandle, indent=4)

            return

def main(
    cube_definition_file: str,
    grid_size: int,
    output_file: str):
    """
    Driver to update general datacube definition file with existing
    datacube NetCDF S3 URLs.
    """
    update_urls = DataCubeGlobalDefinition(grid_size)
    update_urls(cube_definition_file, output_file)


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
        default=None,
        help="File to capture updated general datacube definition information [%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to generate [%(default)s]"
    )
    parser.add_argument(
        '--disableReducedCatalog',
        action='store_true',
        default=False,
        help="Flag to disable reduced (list only the cubes for which Zarr store exists) catalog generation. Default is to generate reduced catalog."
    )
    parser.add_argument(
        '--includeCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains a list of S3 URLs for datacube to include into catalog [%(default)s]."
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='its-live-data/datacubes/v02',
        help="Destination S3 bucket and directory for the datacubes [%(default)s]"
    )

    args = parser.parse_args()

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        DataCubeGlobalDefinition.EPSG_TO_UPDATE = epsg_codes

    DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG = args.disableReducedCatalog
    DataCubeGlobalDefinition.CUBES_S3_PATH = args.bucketDir

    DataCubeGlobalDefinition.CUBES_TO_INCLUDE = [os.path.basename(each) for each in args.includeCubesFile.read_text().split('\n')]
    if len(DataCubeGlobalDefinition.CUBES_TO_INCLUDE):
        logging.info(f"Number of datacubes for catalog: {len(DataCubeGlobalDefinition.CUBES_TO_INCLUDE)}")


    main(
        args.cubeDefinitionFile,
        args.gridSize,
        args.outputFile)

    logging.info(f"Done")
