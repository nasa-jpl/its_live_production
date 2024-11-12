"""
Script to drive Batch processing for datacube update at AWS.

It accepts geojson file with datacube definitions (as generated by the
add_url_to_datacube_definition.py) and submits one AWS Batch job per each datacube.
"""
import argparse
import boto3
import json
import logging
import math
import os
from pathlib import Path, PurePath
import s3fs
import sys
from shapely import geometry
import time

from grid import Bounds
import itslive_utils
from itscube_types import BatchVars, CubeJson, FilenamePrefix


class DataCubeBatch:
    """
    Class to manage Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # Number of granules to process in parallel at a time (to avoid out of memory
    # failures)
    PARALLEL_GRANULES = 1000

    # Number of Dask threads to generate datacube
    NUM_DASK_THREADS = 8

    def __init__(self, grid_size: int, batch_job: str, batch_queue: str, is_dry_run: bool):
        """
        Initialize object.
        """
        self.grid_size_str = f'{grid_size:04d}'
        self.grid_size = grid_size
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

        self.s3 = s3fs.S3FileSystem(anon=True)

    def __call__(
        self,
        cube_file: str,
        s3_bucket: str,
        bucket_dir_path: str,
        target_bucket_dir_path: str,
        job_file: str,
        num_cubes: int,
        cube_start_index: int=0
    ):
        """
        Submit Batch jobs to update existing ITS_LIVE datacubes to AWS.
        """
        # List of submitted datacube Batch jobs and AWS response
        jobs = []

        # List of submitted datacubes for processing
        jobs_files = []

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to generate
            num_jobs = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')

            logging.info(f'Starting with cube index={cube_start_index}')

            # Number of skipped cubes
            num_skipped = 0

            for current_cube_index, each_cube in enumerate(cubes[CubeJson.FEATURES][cube_start_index:]):
                if num_cubes is not None and current_cube_index == num_cubes:
                    # Number of datacubes to iterate through is provided,
                    # stop if they have been iterated through
                    logging.info(f'Reached number of cubes to process: {num_cubes}')
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
                    # Extract int EPSG code
                    epsg_code = str(properties[CubeJson.EPSG])
                    # Format EPSG code as it appears in the datacube filename
                    epsg = f'{CubeJson.EPSG_PREFIX}{epsg_code}'

                    # Include only specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_GENERATE) and \
                            epsg_code not in BatchVars.EPSG_TO_GENERATE:
                        continue

                    # Exclude specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_EXCLUDE) and \
                            epsg_code in BatchVars.EPSG_TO_EXCLUDE:
                        continue

                    http_cube_filename = PurePath(properties[CubeJson.URL])
                    cube_filename = http_cube_filename.name
                    logging.info(f'Cube name: {cube_filename}')

                    bucket_dir = str(PurePath(*(http_cube_filename.parts[2:-1])))
                    if len(BatchVars.PATH_TOKEN) and BatchVars.PATH_TOKEN not in bucket_dir:
                        # A way to pick specific 10x10 grid cell for the datacube
                        logging.info(f"Skipping non-{BatchVars.PATH_TOKEN}")
                        continue

                    # Hack to run specific jobs
                    # to test s3fs problem: 'ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr'
                    if len(BatchVars.CUBES_TO_GENERATE) and cube_filename not in BatchVars.CUBES_TO_GENERATE:
                        logging.info(f"Skipping as not provided in BatchVars.CUBES_TO_GENERATE")
                        num_skipped += 1
                        continue

                    if len(BatchVars.CUBES_TO_EXCLUDE) and cube_filename in BatchVars.CUBES_TO_EXCLUDE:
                        logging.info(f"Skipping as provided in BatchVars.CUBES_TO_EXCLUDE")
                        num_skipped += 1
                        continue

                    # Excluded everything if anything is requested, can count qualified cubes from this point on
                    # starting with initial value of -1, so will have at 0 for the very first cube after
                    # this increment
                    num_jobs += 1

                    target_bucket_dir_s3 = target_bucket_dir_path

                    if target_bucket_dir_path is not None:
                        target_bucket_dir_s3 = bucket_dir.replace(bucket_dir_path, target_bucket_dir_path)

                    # Read into json list and then dump into string to strip all identation characters
                    coords = properties[CubeJson.GEOMETRY_EPSG][CubeJson.COORDINATES][0]

                    cube_params = {
                        'outputStore': cube_filename,
                        'outputBucket': os.path.join(s3_bucket, bucket_dir),
                        'targetProjection': epsg_code,
                        'polygon': json.dumps(coords),
                        'gridCellSize': str(self.grid_size),
                        'chunks': str(DataCubeBatch.PARALLEL_GRANULES),
                        'threads': str(DataCubeBatch.NUM_DASK_THREADS)
                    }

                    if target_bucket_dir_path is not None:
                        cube_params['targetBucket'] = os.path.join(s3_bucket, target_bucket_dir_s3)

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
                                'attempts': 2
                            },
                            timeout={
                                # 'attemptDurationSeconds': 86400
                                # Change to 48 hours (172800)
                                # Change to 72 hours (259200) for very large cubes
                                # Change to 4 days (345600) to support very large cubes
                                'attemptDurationSeconds': 345600
                            }
                        )

                        logging.info(f"Response: {response}")

                        # Does not really work - AWS piles up the jobs in the queue,
                        # then starts a whole bunch at once anyway
                        # # Sleep for 30 seconds to make sure that all AWS Batch jobs
                        # # are not started at the same time
                        # time.sleep(30)

                    # [
                    #     {
                    #         "filename": "https://its-live-data.s3.amazonaws.com/datacubes/v2/S50W070/ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr",
                    #         "s3_filename": "s3://its-live-data/datacubes/v2/S50W070/ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr",
                    #         "roi_percent": 1.5133544414164224,
                    #         "aws_params": {
                    #             "outputStore": "ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr",
                    #             "outputBucket": "s3://its-live-data/datacubes/v2/S50W070",
                    #             "targetProjection": "32718",
                    #             "polygon": "[[400000, 4400000], [500000, 4400000], [500000, 4500000], [400000, 4500000], [400000, 4400000]]",
                    #             "gridCellSize": "120",
                    #             "chunks": "1000",
                    #             "numThreads": "16",
                    #             "targetBucket": "s3://its-live-data/datacubes/v2-updated-10012024/S50W070"
                    #         },
                    #         "aws": {
                    #             "queue": "datacube-spot-16vCPU-128GB",
                    #             "job_definition": "arn:aws:batch:us-west-2:849259517355:job-definition/datacube-update-128Gb:1",
                    #             "response": null
                    #         }
                    #     }
                    # ]
                    jobs.append({
                        'filename': os.path.join(BatchVars.HTTP_PREFIX, bucket_dir, cube_filename),
                        's3_filename': os.path.join(s3_bucket, bucket_dir, cube_filename),
                        'roi_percent': roi,
                        'aws_params': cube_params,
                        'aws': {'queue': self.batch_queue,
                                'job_definition': self.batch_job,
                                'response': response
                                }
                    })

                    jobs_files.append(os.path.join(s3_bucket, bucket_dir, cube_filename))

            logging.info(f"Number of cubes submitted to AWS: {num_jobs}")
            logging.info(f"Number of cubes skipped: {num_skipped}")

            # Write job info to the json file
            logging.info(f"Writing AWS job info to the {job_file}...")
            with open(job_file, 'w') as output_fhandle:
                json.dump(jobs, output_fhandle, indent=4)

            # Write job files to the json file
            job_files_file = f'filenames_{job_file}'
            logging.info(f"Writing jobs output files to the {job_files_file}...")
            with open(job_files_file, 'w') as output_fhandle:
                json.dump(jobs_files, output_fhandle, indent=4)

            return

def main(
    dry_run: bool,
    cube_definition_file: str,
    grid_size: int,
    batch_job: str,
    batch_queue: str,
    s3_bucket: str,
    bucket_dir: str,
    target_bucket_dir: str,
    output_job_file: str,
    number_of_cubes: int,
    cube_start_index: int):
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
    run_batch(cube_definition_file, s3_bucket, bucket_dir, target_bucket_dir, output_job_file, number_of_cubes, cube_start_index)

def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """
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
        default='s3://its-live-data',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-u', '--urlPath',
        type=str,
        action='store',
        default='https://its-live-data.s3.amazonaws.com',
        help="URL for the datacube store in S3 bucket (to provide for easier download option) [%(default)s]"
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='datacubes/v2',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-td', '--targetBucketDir',
        type=str,
        action='store',
        default=None,
        help="Target destination S3 bucket for the datacubes being updated [%(default)s]. Use this option "
            "if datacubes are being updated and their target destination should be other than original cubes s3 location."
    )
    parser.add_argument(
        '-g', '--gridSize',
        type=int,
        action='store',
        default=120,
        help="Grid size for the data cube [%(default)d]"
    )
    parser.add_argument(
        '--numThreads',
        type=int,
        action='store',
        # default=16,
        default=8,
        help="Number of threads to use for the datacube generation [%(default)d]"
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        # default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-create-64Gb:2',
        # default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-update-128Gb:1',   # Update datacubes with 2000 granules in parallel
        default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-update-64Gb:1',  # Update datacubes with 1000 granules in parallel
        help="AWS Batch job definition to use [%(default)s]"
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        # default='datacube-ondemand-8vCPU-64GB',
        default='datacube-spot-8vCPU-64GB',
        # default='datacube-spot-16vCPU-128GB',
        # default='datacube-ondemand-16vCPU-128GB',
        help="AWS Batch job queue to use [%(default)s]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='datacube_update_batch_jobs.json',
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
        '--dryrun',
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
        '-s', '--cubeStartIndex',
        type=int,
        action='store',
        default=0,
        help="Start index for the datacubes to process [%(default)d]."
    )
    parser.add_argument(
        '-p', '--parallelGranules',
        type=int,
        default=1000,
        help="Number of granules to process in parallel at one time [%(default)d]."
    )
    parser.add_argument(
        '-t', '--pathToken',
        type=str,
        default='',
        help="Path token to be present in datacube S3 target path in order for the datacube to be generated [%(default)s]."
    )
    parser.add_argument(
        '--processCubesWithinPolygon',
        type=str,
        action='store',
        default=None,
        help="GeoJSON file that stores polygon the cubes centers should belong to [%(default)s]."
    )
    parser.add_argument(
        '--excludeCubesFile',
        type=Path,
        default=None,
        help="Json file that stores a list of datacubes to exclude from processing [%(default)s]."
    )
    parser.add_argument(
        '--excludeEPSG',
        type=str,
        action='store',
        default=None,
        help="JSON list of EPSG codes to exclude from the datacube generation [%(default)s]"
    )

    # One of --processCubes or --processCubesFile options is allowed for the datacube names
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of filenames to generate [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of filenames for datacube to generate [%(default)s]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Parsed out command-line arguments: {args}")

    DataCubeBatch.PARALLEL_GRANULES = args.parallelGranules
    DataCubeBatch.NUM_DASK_THREADS = args.numThreads
    BatchVars.HTTP_PREFIX           = args.urlPath
    BatchVars.PATH_TOKEN            = args.pathToken

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        BatchVars.EPSG_TO_GENERATE = epsg_codes

    epsg_codes = list(map(str, json.loads(args.excludeEPSG))) if args.excludeEPSG is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes to exclude: {epsg_codes}")
        BatchVars.EPSG_TO_EXCLUDE = epsg_codes

    # Make sure there is no overlap in EPSG_TO_GENERATE and EPSG_TO_EXCLUDE
    diff = set(BatchVars.EPSG_TO_GENERATE).intersection(BatchVars.EPSG_TO_EXCLUDE)
    if len(diff):
        raise RuntimeError(f"The same code is specified for BatchVars.EPSG_TO_EXCLUDE={BatchVars.EPSG_TO_EXCLUDE} and BatchVars.EPSG_TO_GENERATE={BatchVars.EPSG_TO_GENERATE}")

    if args.processCubesFile:
        # Check for this option first as another mutually exclusive option has a default value
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubesFile.read_text())
        # Replace each path by the datacube basename
        BatchVars.CUBES_TO_GENERATE = [os.path.basename(each) for each in BatchVars.CUBES_TO_GENERATE if len(each)]
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

        # Make sure all datacubes are unique
        BatchVars.CUBES_TO_GENERATE = list(set(BatchVars.CUBES_TO_GENERATE))
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} unique datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

    elif args.processCubes:
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubes)
        if len(BatchVars.CUBES_TO_GENERATE):
            logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubes}: {BatchVars.CUBES_TO_GENERATE}")

    if args.processCubesWithinPolygon:
        with open(args.processCubesWithinPolygon, 'r') as fhandle:
            shape_file = json.load(fhandle)

            logging.info(f'Reading region polygon the datacube\'s central point should fall into: {args.processCubesWithinPolygon}')
            shapefile_coords = shape_file[CubeJson.FEATURES][0]['geometry']['coordinates']
            logging.info(f'Got polygon coordinates: {shapefile_coords}')
            line = geometry.LineString(shapefile_coords[0][0])
            BatchVars.POLYGON_SHAPE = geometry.Polygon(line)

    if args.excludeCubesFile:
        BatchVars.CUBES_TO_EXCLUDE = json.loads(args.excludeCubesFile.read_text())

        # Replace each path by the datacube basename
        BatchVars.CUBES_TO_EXCLUDE = [os.path.basename(each) for each in BatchVars.CUBES_TO_EXCLUDE if len(each)]
        logging.info(f"Found {len(BatchVars.CUBES_TO_EXCLUDE)} of datacubes to exclude per {args.excludeCubesFile}: {BatchVars.CUBES_TO_EXCLUDE}")

    return args


if __name__ == '__main__':

    args = parse_args()

    # Check if target datacube location should be other than original cubes location - this is onlly when
    # updating datacubes
    main(
        args.dryrun,
        args.cubeDefinitionFile,
        args.gridSize,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.bucket,
        args.bucketDir,
        args.targetBucketDir,
        args.outputJobFile,
        args.numberOfCubes,
        args.cubeStartIndex
    )

    logging.info(f"Done")
