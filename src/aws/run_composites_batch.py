"""
Script to drive Batch processing for datacube's annual composites generation at AWS.

It accepts shape file with region definition and generates annual composites for
the datacubes which centers fall within the region.

Caller can provide a set of datacubes to generate composites for. Then only these
datacubes are used.

If no shape file is provided, then it generates annual composites for all
existing datacubes as found in the source S3 bucket.

python ./run_composites_batch.py -c ../tools/dataJson/datacubes_v9.json
        --epsgCode '[3413]' -f rerun_final_composite.json
        --processCubes '["ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr"]'
"""
import argparse
import boto3
import json
import logging
import math
import os
from pathlib import Path
from shapely import geometry
import sys
import s3fs

from grid import Bounds
from itscube_types import BatchVars
import itslive_utils
import utils
from itslive_mosaics_types import GeoJsonVars


# Define AWS resources for different number of layers in the datacube and for
# composites creation with slow_error enabled/disabled.

# Previously used AWS Batch job definitions:
# This job definition has 64GB memory and 8 vCPU and should be used
# for RGI05A (Greenland) and RGI19A (Antarctica) only - see explanation
# in the composites code
# default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-64Gb:1',
# default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-64Gb:1',
# default = 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-256Gb:2',
# default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-128Gb:1',
# Use compute optimized env. to re-run composites for ANT and Greenland
# default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-CO-32Gb:1',

# Previously used AWS Batch queues:
# default = 'datacube-ondemand-32vCPU-256GB',

# Up to 150000 use SPOT EC2 instances
# NUM_LAYERS_SPOT = 100_000.   # to force to use ONDEMAND instances
NUM_LAYERS_SPOT = 300_000
# Above 150000 and below 500000 use ONDEMAND EC2 instances with 64 Gb of memory
# Above 500000 use ONDEMAND EC2 instances with 128 Gb of memory
# NUM_LAYERS_ONDEMAND = 500_000
NUM_LAYERS_ONDEMAND = 750_000

# 09/25/2025: Change SPOT vs. OnDemand queues and layers thresholds
def get_aws_disable_slow_error_config(num_layers):
    """
    Get AWS Batch job definition and queue based on the number of layers for
    composites generation with slow_error disabled.
    """
    if num_layers <= NUM_LAYERS_SPOT:
        return {
            'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-32Gb:2',
            'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-spot-4vCPU-32GB',
            # 'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-4vCPU-32GB'
        }
    else:  # num_layers <= NUM_LAYERS_ONDEMAND:
        # return {
        #     'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-64Gb:1',
        #     'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-manual-8vCPU-64GB'
        # }
        return {
            'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-32Gb:2',
            'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-4vCPU-32GB',
        }
    # else:
    #     # Anything above NUM_LAYERS_ONDEMAND
    #     # return {
    #     #     'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-128Gb:1',
    #     #     'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-16vCPU-128GB'
    #     # }
    #     return {
    #         'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-disableSlowError-64Gb:1',
    #         'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-manual-8vCPU-64GB'
    #     }

def get_aws_enable_slow_error_config(num_layers):
    """
    Get AWS Batch job definition and queue based on the number of layers for
    composites generation with slow_error enabled.
    """
    if num_layers <= NUM_LAYERS_SPOT:
        return {
            'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-32Gb:1',
            'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-spot-4vCPU-32GB'
        }
    # elif
    else:   #num_layers <= NUM_LAYERS_ONDEMAND:
        # return {
        #     'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-64Gb:1',
        #     'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-manual-8vCPU-64GB'
        # }
        return {
            'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-32Gb:1',
            'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-4vCPU-32GB'
        }
    # else:
    #     # Anything above NUM_LAYERS_ONDEMAND
    #     # return {
    #     #     'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-128Gb:1',
    #     #     'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-16vCPU-128GB'
    #     # }
    #     return {
    #         'jobDefinition': 'arn:aws:batch:us-west-2:849259517355:job-definition/datacube-annual-composites-64Gb:1',
    #         'jobQueue': 'arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-manual-8vCPU-64GB'
    #     }


class DataCubeCompositeBatch:
    """
    Class to manage one Batch job submission at AWS for datacube composite generation.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # Chunk size in X and Y dimensions to load data in at one time: [:, 100, 100]
    X_Y_CHUNK = 100

    CUBE_NUM_LAYERS_THRESHOLD = 200000

    COMPOSITES_TO_GENERATE = []

    # AWS Batch resources to use based on the error_slow usage
    AWS_RESOURCES = get_aws_disable_slow_error_config

    def __init__(self, grid_size: int, is_dry_run: bool):
        """
        Initialize object.
        """
        self.grid_size_str = f'{grid_size:04d}'
        self.grid_size = grid_size
        self.is_dry_run = is_dry_run

        self.s3 = s3fs.S3FileSystem(anon=True)

    def __call__(
        self,
        cube_file: str,
        s3_bucket: str,
        bucket_dir_path: str,
        output_bucket_dir: str,
        job_file: str,
        num_cubes: int
    ):
        """
        Submit Batch jobs to AWS.
        """
        # List of submitted datacube composite Batch jobs and AWS response
        jobs = []

        # List of submitted datacubes for processing
        jobs_files = []

        num_existing_composites = 0

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of composites to generate
            num_jobs = 0
            num_spot_jobs = 0
            num_ondemand_jobs = 0

            spot_queues = set()
            ondemand_queues = set()

            logging.info(f'Total number of datacubes: {len(cubes["features"])}')
            for each_cube in cubes[GeoJsonVars.features]:
                if num_cubes is not None and num_jobs == num_cubes:
                    # Number of datacubes to generate is provided,
                    # stop if they have been generated
                    logging.info(f'Reached number of cubes to process: {num_cubes}')
                    break

                # Example of data cube definition in json file
                # "properties": {
                #     "fill-opacity": 1.0,
                #     "fill": "red",
                #     "roi_percent_coverage": 0.0,
                # Old format:
                #     "data_epsg": "EPSG:32701",
                #  New format:
                #     "epsg": 32701,
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
                properties = each_cube[GeoJsonVars.properties]

                roi = properties[GeoJsonVars.roi_percent_coverage]
                if roi != 0.0:
                    # Submit AWS Batch to generate the cube composite
                    # Convert int EPSG code to string
                    epsg_code = str(properties[GeoJsonVars.epsg])
                    # Format filename for the cube
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

                    # Get mid point to the nearest 50
                    logging.info(f"Mid point: x={mid_x} y={mid_y}")
                    mid_x = int(
                        math.floor(mid_x/BatchVars.MID_POINT_RESOLUTION) *
                        BatchVars.MID_POINT_RESOLUTION
                    )
                    mid_y = int(
                        math.floor(mid_y/BatchVars.MID_POINT_RESOLUTION) *
                        BatchVars.MID_POINT_RESOLUTION
                    )
                    logging.info(
                        f"Mid point at {BatchVars.MID_POINT_RESOLUTION}: "
                        f"x={mid_x} y={mid_y}"
                    )

                    # Convert to lon/lat coordinates to format s3 bucket path
                    # for the datacube
                    mid_lon_lat = itslive_utils.transform_coord(
                        epsg_code,
                        BatchVars.LON_LAT_PROJECTION,
                        mid_x, mid_y
                    )

                    if BatchVars.POLYGON_SHAPE and \
                        not BatchVars.POLYGON_SHAPE.contains(
                                geometry.Point(mid_lon_lat[0], mid_lon_lat[1])
                            ):
                        logging.info(f"Skipping non-polygon point: {mid_lon_lat}")
                        # Provided polygon does not contain cube's center point
                        continue

                    bucket_dir = itslive_utils.point_to_prefix(
                        mid_lon_lat[1],
                        mid_lon_lat[0],
                        bucket_dir_path
                    )
                    if len(BatchVars.PATH_TOKEN) and \
                            BatchVars.PATH_TOKEN not in bucket_dir:
                        # A way to pick specific 10x10 grid cell for the datacube
                        logging.info(f"Skipping non-{BatchVars.PATH_TOKEN}")
                        continue

                    cube_filename = utils.File.datacube_filename_zarr(
                        epsg,
                        self.grid_size,
                        mid_x, mid_y
                    )
                    logging.info(f'Cube name: {cube_filename}')

                    # Process specific datacubes only
                    if len(BatchVars.CUBES_TO_GENERATE) and \
                            cube_filename not in BatchVars.CUBES_TO_GENERATE:
                        # logging.info(f"Skipping {cube_filename} as not provided in BatchVars.CUBES_TO_GENERATE")
                        continue

                    if len(BatchVars.CUBES_TO_EXCLUDE) and \
                            cube_filename in BatchVars.CUBES_TO_EXCLUDE:
                        logging.info(
                            "Skipping as provided in BatchVars.CUBES_TO_EXCLUDE"
                        )
                        continue

                    # Check if datacube exists in S3 bucket as not all cubes
                    # are most likely generated
                    cube_exists = self.s3.ls(
                        os.path.join(s3_bucket, bucket_dir, cube_filename)
                    )
                    if len(cube_exists) == 0:
                        logging.info(
                            f"Datacube "
                            f"{os.path.join(s3_bucket, bucket_dir, cube_filename)} "
                            "does not exist, skipping composite."
                        )
                        continue

                    # Format cube composites filename:
                    # s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_velocity_120m_X-3250000_Y250000.zarr
                    # composite_filename =f"{FilenamePrefix.Composites}_{int(self.grid_size_str):03d}m_X{mid_x}_Y{mid_y}.zarr"
                    composite_filename = utils.File.composite_filename_zarr(
                        epsg_code,
                        self.grid_size,
                        mid_x, mid_y
                    )
                    logging.info(f'Cube composite name: {composite_filename}')

                    # Check if list of specific composites was provided
                    if len(DataCubeCompositeBatch.COMPOSITES_TO_GENERATE) and \
                            composite_filename not in \
                            DataCubeCompositeBatch.COMPOSITES_TO_GENERATE:
                        continue

                    composite_dir = bucket_dir.replace(
                        bucket_dir_path, output_bucket_dir
                    )
                    logging.info(f'Cube composite S3 directory: {composite_dir}')

                    # Work around to process only non-existent composites (if they
                    # failed from prevoius run or were not generated to begin with)
                    # TODO: make a command-line option
                    composite_exists = self.s3.exists(
                        os.path.join(s3_bucket, composite_dir, composite_filename)
                    )
                    if composite_exists:
                        num_existing_composites += 1
                        logging.info(
                            f"Composite "
                            f"{os.path.join(composite_dir, composite_filename)} "
                            "exists, skipping composite generation."
                        )
                        continue

                    cube_params = {
                        'inputCube': cube_filename,
                        'inputBucket': os.path.join(s3_bucket, bucket_dir),
                        'outputStore': composite_filename,
                        'targetBucket': os.path.join(s3_bucket, composite_dir),
                        'chunkSize': str(DataCubeCompositeBatch.X_Y_CHUNK)
                    }
                    logging.info(f'Cube params: {cube_params}')

                    # Record how many layers are in the datacube to process
                    num_layers = properties[GeoJsonVars.granule_count]
                    aws_resources = DataCubeCompositeBatch.AWS_RESOURCES(num_layers)

                    queue = aws_resources['jobQueue']
                    batch_job = aws_resources['jobDefinition']

                    num_retries = 1
                    if 'spot' in queue:
                        # Re-try SPOT failures, but not OnDemand ones
                        num_retries = 2
                        num_spot_jobs += 1
                        spot_queues.add(queue)

                    else:
                        num_ondemand_jobs += 1
                        ondemand_queues.add(queue)

                    # Submit AWS Batch job
                    response = None
                    if self.is_dry_run is False:
                        # Use datacube filename as that is guaranteed to be unique - it uses
                        # EPSG code as part of the filename. There are multiple composites with the same
                        # filename but in different target directories
                        response = DataCubeCompositeBatch.CLIENT.submit_job(
                            jobName='composite_'+cube_filename.replace('.zarr', ''),
                            jobQueue=queue,
                            jobDefinition=batch_job,
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
                                'attempts': num_retries
                            },
                            # Set timeout to 72 hours
                            timeout={
                                # Change to 4 days (345600) to support very large cubes
                                # Change to 6 days (518400)
                                # Change to 9 days (777600)
                                # Change to 14 days (1209600)
                                # Change to 21 days (1814400)
                                # Change to 42 days (3628800)
                                'attemptDurationSeconds': 3628800
                            }
                        )

                        logging.info(f"Response: {response}")

                    num_jobs += 1

                    jobs.append({
                        'filename': os.path.join(BatchVars.HTTP_PREFIX, composite_dir, composite_filename),
                        's3_filename': os.path.join(s3_bucket, composite_dir, composite_filename),
                        'roi_percent': roi,
                        'cube_layers': num_layers,
                        'aws_params': cube_params,
                        'aws': {'queue': queue,
                                'job_definition': batch_job,
                                'response': response
                                },
                        'job_params': cube_params
                    })

                    jobs_files.append(os.path.join(s3_bucket, bucket_dir, cube_filename))

                else:
                    # Report the cube as being skipped due to the ROI==0 if
                    # it's one of the requested datacubes
                    epsg = properties[GeoJsonVars.data_epsg].replace(
                        GeoJsonVars.epsg_separator, ''
                    )
                    coords = properties[GeoJsonVars.geometry_epsg][GeoJsonVars.coordinates][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int((x_bounds.min + x_bounds.max)/2)
                    mid_y = int((y_bounds.min + y_bounds.max)/2)

                    cube_filename = utils.File.datacube_filename_zarr(
                        epsg, self.grid_size, mid_x, mid_y
                    )

                    if len(BatchVars.CUBES_TO_GENERATE) and \
                            cube_filename in BatchVars.CUBES_TO_GENERATE:
                        logging.info(f'ROI=0 is detected for {cube_filename}')

            logging.info(
                f"Number of batch jobs submitted: {num_jobs} "
                f"({num_spot_jobs} {spot_queues}, "
                f"{num_ondemand_jobs} {ondemand_queues}"
                f" jobs)"
            )

            # Write job info to the json file
            logging.info(f"Writing AWS job info to the {job_file}...")
            with open(job_file, 'w') as output_fhandle:
                json.dump(jobs, output_fhandle, indent=4)

            # Write job files to the json file
            job_files_file = f'filenames_{job_file}'
            logging.info(f"Writing jobs output files to the {job_files_file}...")
            with open(job_files_file, 'w') as output_fhandle:
                json.dump(jobs_files, output_fhandle, indent=4)

            logging.info(f'Number of existing composites: {num_existing_composites}')
            return


def main(
    dry_run: bool,
    cube_definition_file: str,
    grid_size: int,
    s3_bucket: str,
    bucket_dir: str,
    output_bucket_dir: str,
    output_job_file: str,
    number_of_cubes: int
):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0
    run_batch = DataCubeCompositeBatch(
        grid_size,
        dry_run
    )
    run_batch(
        cube_definition_file,
        s3_bucket,
        bucket_dir,
        output_bucket_dir,
        output_job_file,
        number_of_cubes
    )


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
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '--processCubesWithinPolygon',
        type=str,
        action='store',
        default=None,
        help="GeoJSON file that stores polygon the cubes centers should belong to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data',
        help="Destination S3 bucket for the datacubes composites [%(default)s]"
    )
    parser.add_argument(
        '-u', '--urlPath',
        type=str,
        action='store',
        default='https://its-live-data.s3.amazonaws.com',
        help="URL for the store in S3 bucket (to provide for easier download option) [%(default)s]"
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='datacubes/v2-updated-october2024',
        help="S3 directory for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-o', '--outputBucketDir',
        type=str,
        action='store',
        default='composites/annual/v2-updated-september2025',
        help="Destination S3 directory for the composites [%(default)s]"
    )
    parser.add_argument(
        '-g', '--gridSize',
        type=int,
        action='store',
        default=120,
        help="Grid size for the data cube [%(default)d]"
    )
    parser.add_argument(
        '-f', '--outputJobFile',
        type=str,
        action='store',
        default='annual_composite_batch_jobs.json',
        help="File to capture submitted composites AWS Batch jobs [%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to process [%(default)s]"
    )
    parser.add_argument(
        '--excludeEPSG',
        type=str,
        action='store',
        default=None,
        help="JSON list of EPSG codes to exclude from the datacube processing [%(default)s]"
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
        help="Number of datacubes to process [%(default)d]. If left at default value, then process all qualifying datacubes."
    )
    parser.add_argument(
        '-s', '--chunkSize',
        type=int,
        default=10,
        help="Size of x and y dimensions for the chunk of spacial points to process at one time [%(default)d]."
    )
    parser.add_argument(
        '-t', '--pathToken',
        type=str,
        default='',
        help="Path token to be present in datacube S3 target path in order for the datacube to be processed [%(default)s]."
    )
    parser.add_argument(
        '--excludeCubesFile',
        type=Path,
        default=None,
        help="Json file that stores a list of datacubes to exclude from processing [%(default)s]."
    )

    # One of --processCubes or --processCubesFile options is allowed for the datacube names
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of filenames to process [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of filenames for datacube to process [%(default)s]."
    )
    group.add_argument(
        '--processCompositesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of filenames for composites to process [%(default)s]."
    )
    parser.add_argument(
        '--useErrorSlow',
        action='store_true',
        help="Enable use of valid v[xy]_error_slow instead of v[xy]_error "
                "values. Use this option for RGI05 (Greenland) or "
                "RGI19 (Antarctica) composites only for now."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f'Parser arguments: {args}')

    BatchVars.HTTP_PREFIX = args.urlPath
    BatchVars.PATH_TOKEN = args.pathToken
    DataCubeCompositeBatch.X_Y_CHUNK = args.chunkSize

    DataCubeCompositeBatch.AWS_RESOURCES = get_aws_enable_slow_error_config if args.useErrorSlow else get_aws_disable_slow_error_config

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        BatchVars.EPSG_TO_GENERATE = epsg_codes

    epsg_codes = list(
        map(str, json.loads(args.excludeEPSG))
    ) if args.excludeEPSG is not None else None

    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes to exclude: {epsg_codes}")
        BatchVars.EPSG_TO_EXCLUDE = epsg_codes

    # Make sure there is no overlap in EPSG_TO_GENERATE and EPSG_TO_EXCLUDE
    diff = set(BatchVars.EPSG_TO_GENERATE).intersection(BatchVars.EPSG_TO_EXCLUDE)
    if len(diff):
        raise RuntimeError(
            f"The same code is specified for BatchVars.EPSG_TO_EXCLUDE="
            f"{BatchVars.EPSG_TO_EXCLUDE} and BatchVars.EPSG_TO_GENERATE="
            f"{BatchVars.EPSG_TO_GENERATE}"
        )

    if args.processCubesFile:
        # Check for this option first as another mutually exclusive
        # option has a default value
        BatchVars.CUBES_TO_GENERATE = json.loads(
            args.processCubesFile.read_text()
        )

        # Replace each path by the datacube basename, and make sure
        # cubefilenames are unique
        BatchVars.CUBES_TO_GENERATE = list(
            set([os.path.basename(each) for each in BatchVars.CUBES_TO_GENERATE if len(each)])
        )

        logging.info(
            f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to "
            f"generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}"
        )

    elif args.processCompositesFile:
        DataCubeCompositeBatch.COMPOSITES_TO_GENERATE = json.loads(
            args.processCompositesFile.read_text()
        )
        DataCubeCompositeBatch.COMPOSITES_TO_GENERATE = [
            os.path.basename(each) for each in
            DataCubeCompositeBatch.COMPOSITES_TO_GENERATE if len(each)
        ]

        if len(DataCubeCompositeBatch.COMPOSITES_TO_GENERATE):
            logging.info(
                f"Found {len(DataCubeCompositeBatch.COMPOSITES_TO_GENERATE)} "
                f"of datacubes to generate from {args.processCompositesFile}: "
                f"{DataCubeCompositeBatch.COMPOSITES_TO_GENERATE}"
            )

    elif args.processCubes:
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubes)
        if len(BatchVars.CUBES_TO_GENERATE):
            logging.info(
                f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to "
                f"generate from {args.processCubes}: "
                f"{BatchVars.CUBES_TO_GENERATE}"
            )

    if args.processCubesWithinPolygon:
        with open(args.processCubesWithinPolygon, 'r') as fhandle:
            shape_file = json.load(fhandle)

            logging.info(
                f"Reading region polygon the datacube's central point "
                f"should fall into: {args.processCubesWithinPolygon}"
            )
            shapefile_coords = shape_file[GeoJsonVars.features][0]['geometry']['coordinates']
            logging.info(f'Got polygon coordinates: {shapefile_coords}')

            line = geometry.LineString(shapefile_coords[0][0])
            BatchVars.POLYGON_SHAPE = geometry.Polygon(line)
            logging.info(f'Set polygon: {BatchVars.POLYGON_SHAPE}')

    if args.excludeCubesFile:
        BatchVars.CUBES_TO_EXCLUDE = json.loads(args.excludeCubesFile.read_text())

        # Replace each path by the datacube basename
        BatchVars.CUBES_TO_EXCLUDE = [
            os.path.basename(each) for each in BatchVars.CUBES_TO_EXCLUDE if
            len(each)
        ]
        logging.info(
            f"Found {len(BatchVars.CUBES_TO_EXCLUDE)} of datacubes to "
            f"exclude per {args.excludeCubesFile}: "
            f"{BatchVars.CUBES_TO_EXCLUDE}"
        )

    return args


if __name__ == '__main__':

    args = parse_args()

    main(
        args.dryrun,
        args.cubeDefinitionFile,
        args.gridSize,
        args.bucket,
        args.bucketDir,
        args.outputBucketDir,
        args.outputJobFile,
        args.numberOfCubes
    )

    logging.info("Done")
