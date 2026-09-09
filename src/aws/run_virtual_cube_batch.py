"""
Script to drive Batch processing for virtual (icechunk) datacube generation at AWS.

It accepts geojson file with chunk-aligned datacube definitions and submits one AWS
Batch job per each datacube which has ROI (region of interest) != 0. Each job runs
virtual_itslive_cube_per_chunk.py to build a virtual icechunk datacube for the cube's
bounding box.
"""
import argparse
import boto3
import json
import logging
import math
import os
from pathlib import Path
import s3fs
import sys
import time
from shapely import geometry

from grid import Bounds
import itslive_utils
from itscube_types import BatchVars
import utils
from itslive_mosaics_types import GeoJsonVars


class VirtualDataCubeBatch:
    """
    Class to manage Batch job submission for virtual datacube generation at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # Pixel size (grid cell size) used by virtual_itslive_cube_per_chunk.py, hardcoded
    # there (no CLI override) as its PIXEL_SIZE module constant. Needed here only to
    # format the datacube filename consistently with the deep-copy naming convention.
    PIXEL_SIZE = 120

    # Number of threads to use for parallel processing within each job
    NUM_THREADS = 16

    # Number of granules to load and commit together per icechunk snapshot
    BATCH_SIZE = 10000

    # Pace job submission to AWS Batch to avoid many jobs starting their
    # granule-loading burst within the same 1-2 minutes and overwhelming S3
    # with concurrent HEAD/GET requests (observed to cause widespread 503
    # "SlowDown" throttling -- see src/aws/batch_logs/virtual_cubes/09082026).
    # Sleep SLEEP_DURATION_SEC after every SLEEP_AFTER_NUM_JOBS jobs submitted.
    SLEEP_AFTER_NUM_JOBS = 300
    SLEEP_DURATION_SEC = 120

    def __init__(self, batch_job: str, batch_queue: str, is_dry_run: bool):
        """
        Initialize object.
        """
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

        self.s3 = s3fs.S3FileSystem(anon=True)

    def __call__(
        self,
        cube_file: str,
        s3_bucket: str,
        bucket_dir_path: str,
        job_file: str,
        num_cubes: int
    ):
        """
        Submit Batch jobs to AWS.
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

            for each_cube in cubes[GeoJsonVars.features]:
                if num_cubes is not None and num_jobs == num_cubes:
                    # Number of datacubes to generate is provided,
                    # stop if they have been generated
                    logging.info(f'Reached number of cubes to process: {num_cubes}')
                    break

                # Example of data cube definition in json file
                # "properties": {
                #     "fill-opacity": 0.923583984375,
                #     "fill": "red",
                #     "cube_id": "ITS_LIVE_velocity_EPSG32717_61440m_X675832_Y9768967",
                #     "roi_percent_coverage": 7.6416015625,
                #     "epsg": 32717,
                #     "geometry_epsg": {
                #         "type": "Polygon",
                #         "coordinates": [
                #             [
                #                 [675832.5, 9768967.5],
                #                 [737272.5, 9768967.5],
                #                 [737272.5, 9830407.5],
                #                 [675832.5, 9830407.5],
                #                 [675832.5, 9768967.5]
                #             ]
                #         ]
                #     }
                # }

                # Start the Batch job for each cube with ROI != 0
                properties = each_cube[GeoJsonVars.properties]

                roi = properties[GeoJsonVars.roi_percent_coverage]
                if roi != 0.0:
                    # Submit AWS Batch to generate the virtual cube
                    epsg_code = str(properties[GeoJsonVars.epsg])
                    epsg = GeoJsonVars.epsg_prefix + epsg_code

                    # Include only specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_GENERATE) and \
                            epsg_code not in BatchVars.EPSG_TO_GENERATE:
                        continue

                    # Exclude specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_EXCLUDE) and \
                            epsg_code in BatchVars.EPSG_TO_EXCLUDE:
                        continue

                    coords = properties[GeoJsonVars.geometry_epsg][GeoJsonVars.coordinates][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int((x_bounds.min + x_bounds.max)/2)
                    mid_y = int((y_bounds.min + y_bounds.max)/2)

                    # Get mid point to the nearest 50, same as for deep-copy datacubes,
                    # to keep filenames consistent between the two conventions
                    mid_x = int(math.floor(mid_x/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)
                    mid_y = int(math.floor(mid_y/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)

                    cube_filename = utils.File.datacube_filename_icechunk(
                        epsg, VirtualDataCubeBatch.PIXEL_SIZE, mid_x, mid_y
                    )
                    logging.info(f'Cube name: {cube_filename}')

                    # A way to run specific jobs only
                    if len(BatchVars.CUBES_TO_GENERATE) and cube_filename not in BatchVars.CUBES_TO_GENERATE:
                        logging.info("Skipping as not provided in BatchVars.CUBES_TO_GENERATE")
                        continue

                    if len(BatchVars.CUBES_TO_EXCLUDE) and cube_filename in BatchVars.CUBES_TO_EXCLUDE:
                        logging.info("Skipping as provided in BatchVars.CUBES_TO_EXCLUDE")
                        continue

                    if BatchVars.POLYGON_SHAPE:
                        mid_lon_lat = itslive_utils.transform_coord(
                            epsg_code,
                            BatchVars.LON_LAT_PROJECTION,
                            mid_x, mid_y
                        )

                        if not BatchVars.POLYGON_SHAPE.contains(
                                geometry.Point(mid_lon_lat[0], mid_lon_lat[1])
                        ):
                            logging.info(f"Skipping non-polygon point: {mid_lon_lat}")
                            # Provided polygon does not contain cube's center point
                            continue

                    target_s3_path = os.path.join(s3_bucket, bucket_dir_path, cube_filename)

                    # Work around to make sure there are no partially generated cubes from
                    # previously failed runs
                    if self.s3.exists(target_s3_path):
                        logging.info(
                            f"Datacube {target_s3_path} exists, skipping datacube generation."
                        )
                        continue

                    cube_params = {
                        'outputStore': target_s3_path,
                        'projection': epsg_code,
                        'polygon': json.dumps(coords),
                        'threads': str(VirtualDataCubeBatch.NUM_THREADS),
                        'batchSize': str(VirtualDataCubeBatch.BATCH_SIZE),
                    }

                    logging.info(f'Cube params: {cube_params}')

                    # Submit AWS Batch job
                    response = None
                    if self.is_dry_run is False:
                        # Aws job name can't include '.' character, remove
                        # file extension
                        response = VirtualDataCubeBatch.CLIENT.submit_job(
                            jobName=cube_filename.replace(utils.File.ext.icechunk, ''),
                            jobQueue=self.batch_queue,
                            jobDefinition=self.batch_job,
                            parameters=cube_params,
                            retryStrategy={
                                'attempts': 1
                            },
                            timeout={
                                # Change to 14 days to support very large cubes
                                'attemptDurationSeconds': 1209600
                            }
                        )

                        logging.info(f"Response: {response}")

                    num_jobs += 1
                    logging.info(f'Submitted {num_jobs} to AWS')

                    if not self.is_dry_run and \
                            num_jobs % VirtualDataCubeBatch.SLEEP_AFTER_NUM_JOBS == 0:
                        logging.info(
                            f'Submitted {num_jobs} jobs so far; sleeping '
                            f'{VirtualDataCubeBatch.SLEEP_DURATION_SEC}s to avoid '
                            'overwhelming S3 with concurrent job start-up requests'
                        )
                        time.sleep(VirtualDataCubeBatch.SLEEP_DURATION_SEC)

                    jobs.append({
                        's3_filename': target_s3_path,
                        'roi_percent': roi,
                        'aws_params': cube_params,
                        'aws': {'queue': self.batch_queue,
                                'job_definition': self.batch_job,
                                'response': response
                                }
                    })

                    jobs_files.append(target_s3_path)

            logging.info(f"Number of batch jobs submitted: {num_jobs}")

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
    batch_job: str,
    batch_queue: str,
    s3_bucket: str,
    bucket_dir: str,
    output_job_file: str,
    number_of_cubes: int
):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0
    run_batch = VirtualDataCubeBatch(
        batch_job,
        batch_queue,
        dry_run
    )
    run_batch(cube_definition_file, s3_bucket, bucket_dir, output_job_file, number_of_cubes)


def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """
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
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        required=True,
        help="GeoJson file that stores chunk-aligned cube polygon definitions."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data',
        help="Destination S3 bucket for the virtual datacubes [%(default)s]"
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='datacubes/spatial/v2.2',
        help="Destination S3 directory for the virtual datacubes [%(default)s]. All "
            "generated icechunk repositories are placed directly in this single "
            "directory."
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        default='virtual-datacube-32Gb',
        help="AWS Batch job definition to use for virtual datacube generation"
        "[%(default)s]."
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        default='datacube-spot-4vCPU-32GB',
        help="AWS Batch job queue to use for virtual datacube generation [%(default)s]."
    )
    parser.add_argument(
        '--numThreads',
        type=int,
        action='store',
        default=16,
        help="Number of threads to use for the virtual datacube generation [%(default)d]"
    )
    parser.add_argument(
        '--batchSize',
        type=int,
        action='store',
        default=10000,
        help="Number of granules to load and commit together per icechunk snapshot "
            "[%(default)d]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='virtual_datacube_batch_jobs.json',
        help="File to capture submitted virtual datacube AWS Batch jobs [%(default)s]"
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
        help="Number of datacubes to generate [%(default)d]. If left at "
            "default value, then generate all qualifying datacubes."
    )
    parser.add_argument(
        '--excludeCubesFile',
        type=Path,
        nargs='+',
        default=None,
        help="One or more json files, each storing a list of datacubes to exclude from "
            "processing [%(default)s]. Lists from all files are combined and de-duplicated."
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

    VirtualDataCubeBatch.NUM_THREADS = args.numThreads
    VirtualDataCubeBatch.BATCH_SIZE = args.batchSize

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

    if args.excludeCubesFile:
        exclude_cubes = []
        for each_file in args.excludeCubesFile:
            cubes_from_file = json.loads(each_file.read_text())
            logging.info(f"Found {len(cubes_from_file)} of datacubes to exclude per {each_file}")
            exclude_cubes.extend(cubes_from_file)

        # Replace each path by the datacube basename
        exclude_cubes = [os.path.basename(each) for each in exclude_cubes if len(each)]

        # Make sure all datacubes are unique
        BatchVars.CUBES_TO_EXCLUDE = list(set(exclude_cubes))
        logging.info(
            f"Found {len(BatchVars.CUBES_TO_EXCLUDE)} unique datacubes to exclude from "
            f"{len(args.excludeCubesFile)} file(s): {BatchVars.CUBES_TO_EXCLUDE}"
        )

    return args


if __name__ == '__main__':

    args = parse_args()

    main(
        args.dryrun,
        args.cubeDefinitionFile,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.bucket,
        args.bucketDir,
        args.outputJobFile,
        args.numberOfCubes
    )

    logging.info(f"Done")
