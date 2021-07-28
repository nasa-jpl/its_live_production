"""
Script to drive AWS Batch processing for datacube conversion from Zarr to NetCDF format.

It collects all Zarr datacubes in S3 bucket directory and submits one AWS Batch job
per each datacube conversion to NetCDF format.
"""
import boto3
import json
import logging
import math
import os
import s3fs

from grid import Bounds
import itslive_utils


class DataCubeConversionBatch:
    """
    Class to manage Batch job submissions at AWS for each datacube conversion
    (one job per datacube).
    """
    # File extensions for the datacube
    ZARR_EXT = '.zarr'
    NC_EXT    = '.nc'
    S3_PREFIX = 's3://'

    CLIENT = boto3.client('batch', region_name='us-west-2')

    def __init__(self, batch_job: str, batch_queue: str, is_dry_run: bool):
        """
        Initialize object.
        """
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

    def __call__(self, s3_bucket: str, bucket_dir_path: str, job_file: str):
        """
        Submit Batch jobs to AWS.
        """
        # List of submitted datacube Batch jobs and AWS response
        jobs = []

        # Collect all datacubes in Zarr format
        s3_out = s3fs.S3FileSystem(anon=True)
        all_datacubes = []

        for each in s3_out.ls(os.path.join(s3_bucket, bucket_dir_path)):
            # List subdirectories of the top-level path: its-live-data.jpl.nasa.gov/datacubes/v01
            # ATTN: don't use s3_out.glob as it is very slow and does not work on dirs
            cubes = s3_out.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]

            all_datacubes.extend(cubes)

        logging.info(f'Total number of datacubes: {len(all_datacubes)}')
        if len(all_datacubes) == 0:
            logging.info(f"No Zarr datacubes are found in {os.path.join(s3_bucket, bucket_dir_path)}.")

        # Number of cubes to generate
        num_jobs = 0

        for each_cube in all_datacubes:
            # Submit AWS Batch to convert the cube to NetCDF format
            logging.info(f'Cube: {each_cube}')

            # Local name for the NetCDF format datacube
            cube_path, cube_filename = os.path.split(each_cube)

            # Hack to create long running job - to test s3fs issue
            # if cube_filename != 'ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr':
            #     continue

            if DataCubeConversionBatch.S3_PREFIX not in each_cube:
                # Prepend an S3 bucket prefix (for conversion script to use)
                each_cube = f"{DataCubeConversionBatch.S3_PREFIX}{each_cube}"

            cube_params = {
                'inputFile': each_cube,
                'outputFile': cube_filename.replace(DataCubeConversionBatch.ZARR_EXT, DataCubeConversionBatch.NC_EXT),
                'outputBucket': cube_path
            }
            logging.info(f'Cube params: {cube_params}')

            # Submit AWS Batch job
            response = None
            if self.is_dry_run is False:
                response = DataCubeConversionBatch.CLIENT.submit_job(
                    jobName='toNetCDF_' * cube_filename,
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
                        'attemptDurationSeconds': 21600
                    }
                )

                logging.info(f"Response: {response}")

            num_jobs += 1
            jobs.append({
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
    batch_job: str,
    batch_queue: str,
    s3_bucket: str,
    bucket_dir: str,
    output_job_file: str):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube conversion
    run_batch = DataCubeConversionBatch(batch_job, batch_queue, dry_run)
    run_batch(s3_bucket, bucket_dir, output_job_file)


if __name__ == '__main__':
    import argparse
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
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data.jpl.nasa.gov',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        action='store',
        default='datacubes/v01',
        help="Destination S3 bucket for the datacubes [%(default)s]"
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-terraform:1',
        help="AWS Batch job definition to use [%(default)s]"
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        default='datacube-terraform',
        help="AWS Batch job queue to use [%(default)s]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='datacube_convert_batch_jobs.json',
        help="File to capture submitted datacube AWS Batch jobs [%(default)s]"
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )

    args = parser.parse_args()

    main(
        args.dry,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.bucket,
        args.bucketDir,
        args.outputJobFile
    )

    logging.info(f"Done")
