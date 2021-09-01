"""
Script to submit one Batch job to generate one datacube on AWS.
"""
import boto3
import logging
import os


class DataCubeBatch:
    """
    Class to manage one Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # Number of granules to process in parallel at a time (to avoid out of memory
    # failures)
    PARALLEL_GRANULES = 250

    def __init__(self, batch_job: str, batch_queue: str, is_dry_run: bool):
        """
        Initialize object.
        """
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

    def __call__(self, cube_params: dict):
        """
        Submit one datacube Batch job to AWS.
        """
        logging.info(f'Cube params: {cube_params}')

        # Submit AWS Batch job
        response = None
        if self.is_dry_run is False:
            response = DataCubeBatch.CLIENT.submit_job(
                jobName=cube_params['outputStore'].replace('.zarr', ''),
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

        job = {
            'aws_params': cube_params,
            'aws': {
                'queue': self.batch_queue,
                'job_definition': self.batch_job,
                'response': response
            }
        }

        logging.info(f"Batch job submitted: {job}")
        return


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
        default='test_datacube/v01',
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
        default='arn:aws:batch:us-west-2:849259517355:job-definition/datacube-create-centroid-30Gb:1',
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
        '-o', '--outputFilename',
        type=str,
        action='store',
        help="Filename for the datacube Zarr output store."
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        help="EPSG code for the datacube to generate [%(default)s]"
    )
    parser.add_argument(
        '--dry',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )
    parser.add_argument(
        '-p', '--parallelGranules',
        type=int,
        default=250,
        help="Number of granules to process in parallel at one time [%(default)d]."
    )
    parser.add_argument(
        '--centroid',
        type=str,
        required=True,
        help="JSON 2-element list for centroid point (x, y) of the datacube in target EPSG code projection. "
             "Polygon vertices are calculated based on the centroid and cube dimension arguments."
    )
    parser.add_argument(
        '--dimSize',
        type=int,
        default=100000,
        help="Cube dimension in meters [%(default)d]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    # "cube_params": {
    #   "outputStore": "ITS_LIVE_vel_EPSG3413_G0120_X150000_Y-950000.zarr",
    #   "outputBucket": "s3://its-live-data.jpl.nasa.gov/datacubes/v01/N80W030",
    #   "targetProjection": "3413",
    #   "centroid": "[100000, -1000000]",
    #   "gridCellSize": "120",
    #   "chunks": "250"
    # },
    cube_params = {
        'outputStore':      args.outputFilename,
        'outputBucket':     os.path.join(args.bucket, args.bucketDir),
        'targetProjection': args.epsgCode,
        'centroid':         args.centroid,
        'dimSize':          str(args.dimSize),
        'gridCellSize':     str(args.gridSize),
        'chunks':           str(args.parallelGranules)
    }
    logging.info(f'Cube params: {cube_params}')

    DataCubeBatch.PARALLEL_GRANULES = args.parallelGranules

    # Submit Batch job to AWS for each datacube which has ROI!=0
    batch_job = DataCubeBatch(args.batchJobDefinition, args.batchJobQueue, args.dry)
    batch_job(cube_params)

    logging.info(f"Done")
