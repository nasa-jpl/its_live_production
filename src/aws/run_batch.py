"""
Script to drive Batch processing for datacube generation at AWS.

It accepts geojson file with datacube definitions and submits one AWS Batch job
per each datacube which has ROI (region of interest) != 0.
"""
import boto3
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


class DataCubeBatch:
    """
    Class to manage one Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # HTTP URL for the datacube full path
    HTTP_PREFIX = ''

    FILENAME_PREFIX = 'ITS_LIVE_vel'
    MID_POINT_RESOLUTION = 50.0

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    # List of EPSG codes to generate datacubes for. If this list is empty,
    # then generate all ROI!=0 datacubes.
    EPSG_TO_GENERATE = []

    # List of datacube filenames to generate if only specific datacubes should be generated.
    # If an empty list then generate all qualifying datacubes.
    CUBES_TO_GENERATE = []

    # Number of granules to process in parallel at a time (to avoid out of memory
    # failures)
    PARALLEL_GRANULES = 250

    def __init__(self, grid_size: int, batch_job: str, batch_queue: str, is_dry_run: bool):
        """
        Initialize object.
        """
        self.grid_size_str = f'{grid_size:04d}'
        self.grid_size = grid_size
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

    def __call__(self, cube_file: str, s3_bucket: str, bucket_dir_path: str, job_file: str, num_cubes: int):
        """
        Submit Batch jobs to AWS.
        """
        # List of submitted datacube Batch jobs and AWS response
        jobs = []

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to generate
            num_jobs = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')
            for each_cube in cubes[CubeJson.FEATURES]:
                if num_cubes is not None and num_jobs == num_cubes:
                    # Number of datacubes to generate is provided,
                    # stop if they have been generated
                    break

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
                    # Submit AWS Batch to generate the cube
                    # Format filename for the cube
                    epsg = properties[CubeJson.DATA_EPSG].replace(CubeJson.EPSG_SEPARATOR, '')
                    # Extract int EPSG code
                    epsg_code = epsg.replace(CubeJson.EPSG_PREFIX, '')

                    if len(DataCubeBatch.EPSG_TO_GENERATE) and \
                       epsg_code not in DataCubeBatch.EPSG_TO_GENERATE:
                        continue

                    coords = properties[CubeJson.GEOMETRY_EPSG][CubeJson.COORDINATES][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int((x_bounds.min + x_bounds.max)/2)
                    mid_y = int((y_bounds.min + y_bounds.max)/2)

                    # Get mid point to the nearest 50
                    logging.info(f"Mid point: x={mid_x} y={mid_y}")
                    mid_x = int(math.floor(mid_x/DataCubeBatch.MID_POINT_RESOLUTION)*DataCubeBatch.MID_POINT_RESOLUTION)
                    mid_y = int(math.floor(mid_y/DataCubeBatch.MID_POINT_RESOLUTION)*DataCubeBatch.MID_POINT_RESOLUTION)
                    logging.info(f"Mid point at {DataCubeBatch.MID_POINT_RESOLUTION}: x={mid_x} y={mid_y}")

                    # Convert to lon/lat coordinates to format s3 bucket path
                    # for the datacube
                    mid_lon_lat = itslive_utils.transform_coord(
                        epsg_code,
                        DataCubeBatch.LON_LAT_PROJECTION,
                        mid_x, mid_y
                    )
                    bucket_dir = itslive_utils.point_to_prefix(mid_lon_lat[1], mid_lon_lat[0], bucket_dir_path)
                    if len(DataCubeBatch.PATH_TOKEN) and DataCubeBatch.PATH_TOKEN not in bucket_dir:
                        # A way to pick specific 10x10 grid cell for the datacube
                        logging.info(f"Skipping non-{DataCubeBatch.PATH_TOKEN}")
                        continue

                    cube_filename = f"{DataCubeBatch.FILENAME_PREFIX}_{epsg}_G{self.grid_size_str}_X{mid_x}_Y{mid_y}.zarr"
                    logging.info(f'Cube name: {cube_filename}')

                    # Hack to run specific jobs
                    # to test s3fs problem: 'ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr'
                    if len(DataCubeBatch.CUBES_TO_GENERATE) and cube_filename not in DataCubeBatch.CUBES_TO_GENERATE:
                        logging.info(f"Skipping non-{DataCubeBatch.CUBES_TO_GENERATE}")
                        continue

                    cube_params = {
                        'outputStore': cube_filename,
                        'outputBucket': os.path.join(s3_bucket, bucket_dir),
                        'targetProjection': epsg_code,
                        'polygon': json.dumps(coords),
                        'gridCellSize': str(self.grid_size),
                        'chunks': str(DataCubeBatch.PARALLEL_GRANULES)
                    }
                    logging.info(f'Cube params: {cube_params}')

                    # Submit AWS Batch job
                    response = None
                    if self.is_dry_run is False:
                        response = DataCubeBatch.CLIENT.submit_job(
                            jobName=cube_filename.replace('.zarr', ''),
                            jobQueue=self.batch_queue,
                            jobDefinition=self.batch_job,
                            parameters=cube_params,
                            # containerOverrides={
                            #     'vcpus': 123,
                            #     'memory': ,
                            #     'command': [
                            #         'string',
                            #     ],
                            #     'environment': [
                            #         {
                            #             'name': 'string',
                            #             'value': 'string'
                            #         },
                            #     ]
                            # },
                            retryStrategy={
                                'attempts': 1
                            },
                            timeout={
                                'attemptDurationSeconds': 86400
                            }
                        )

                        logging.info(f"Response: {response}")

                    num_jobs += 1
                    jobs.append({
                        'filename': os.path.join(DataCubeBatch.HTTP_PREFIX, bucket_dir, cube_filename),
                        's3_filename': os.path.join(s3_bucket, bucket_dir, cube_filename),
                        'roi_percent': roi,
                        'aws_params': cube_params,
                        'aws': {'queue': self.batch_queue,
                                'job_definition': self.batch_job,
                                'response': response
                                }
                    })

            logging.info(f"Number of batch jobs submitted: {num_jobs}")

            # Write job info to the json file
            logging.info(f"Writing AWS job info to the {job_file}...")
            with open(job_file, 'w') as output_fhandle:
                json.dump(jobs, output_fhandle, indent=4)

            return

def main(
    dry_run: bool,
    cube_definition_file: str,
    grid_size: int,
    batch_job: str,
    batch_queue: str,
    s3_bucket: str,
    bucket_dir: str,
    output_job_file: str,
    number_of_cubes: int):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0
    run_batch = DataCubeBatch(
        grid_size,
        batch_job,
        batch_queue,
        dry_run
    )
    run_batch(cube_definition_file, s3_bucket, bucket_dir, output_job_file, number_of_cubes)


if __name__ == '__main__':
    import argparse
    import sys
    import warnings

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
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data.jpl.nasa.gov',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-u', '--urlPath',
        type=str,
        action='store',
        default='http://its-live-data.jpl.nasa.gov.s3.amazonaws.com',
        help="URL for the datacube store in S3 bucket (to provide for easier download option) [%(default)s]"
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='datacubes/v1',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-g', '--gridSize',
        type=int,
        action='store',
        default=120,
        help="Grid size for the data cube [%(default)d]"
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-create-30Gb:1',
        help="AWS Batch job definition to use [%(default)s]"
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        default='datacube-convert-4vCPU-32GB',
        help="AWS Batch job queue to use [%(default)s]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='datacube_batch_jobs.json',
        help="File to capture submitted datacube AWS Batch jobs [%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to generate [%(default)s]"
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )
    parser.add_argument(
        '-n', '--numberOfCubes',
        type=int,
        action='store',
        default=-1,
        help="Number of datacubes to generate [%(default)d]. If left at default value, then generate all qualifying datacubes."
    )
    parser.add_argument(
        '-p', '--parallelGranules',
        type=int,
        default=250,
        help="Number of granules to process in parallel at one time [%(default)d]."
    )
    parser.add_argument(
        '-t', '--pathToken',
        type=str,
        default='',
        help="Path token to be present in datacube S3 target path in order for the datacube to be generated [%(default)s]."
    )
    parser.add_argument(
        '--processCubes',
        type=str,
        default='[]',
        help="JSON list of datacube filenames to generate [%(default)s]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        DataCubeBatch.EPSG_TO_GENERATE = epsg_codes

    DataCubeBatch.PARALLEL_GRANULES = args.parallelGranules
    DataCubeBatch.HTTP_PREFIX       = args.urlPath
    DataCubeBatch.PATH_TOKEN        = args.pathToken
    DataCubeBatch.CUBES_TO_GENERATE = json.loads(args.processCubes)

    main(
        args.dry,
        args.cubeDefinitionFile,
        args.gridSize,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.bucket,
        args.bucketDir,
        args.outputJobFile,
        args.numberOfCubes
    )

    logging.info(f"Done")
