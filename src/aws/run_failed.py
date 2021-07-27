"""
Script to re-run failed Batch jobs for the datacube generation at AWS.

It accepts JSON file that lists submitted jobs and a file with job IDs that need to be
re-run.
"""
import boto3
import json
import logging
import os
from pathlib import Path


class AWS:
    """
    Variables names within JSON datacube job information file.
    """
    PARAMS         = 'aws_params'
    AWS            = 'aws'
    JOB_ID         = 'jobId'
    S3_FILENAME    = 's3_filename'
    QUEUE          = 'queue'
    JOB_DEFINITION = 'job_definition'
    RESPONSE       = 'response'


class DataCubeBatchReprocess:
    """
    Class to manage one Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')


    def __init__(self, batch_job: str, batch_queue: str, is_dry_run: bool = False):
        """
        Initialize object.
        """
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

    def __call__(self, job_file: str, fail_job_file: str, output_job_file: str):
        """
        Re-submit failed jobs (as listed in fail_job_file) to AWS Batch.
        """
        # List of submitted datacube Batch jobs and AWS response
        jobs = []

        # Read all submitted jobs info
        job_info = json.loads(job_file.read_text())

        # Read IDs of failed jobs
        failed_ids = fail_job_file.read_text().split('\n')
        logging.info(f"Failed IDs: {failed_ids}")
        num_jobs = 0

        for each_job_info in job_info:
            # Format example for the job information file
            # {
            #     "filename": "http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/datacubes/v01/N50W120/ITS_LIVE_vel_EPSG3413_G0120_X-3850000_Y-950000.zarr",
            #     "s3_filename": "s3://its-live-data.jpl.nasa.gov/datacubes/v01/N50W120/ITS_LIVE_vel_EPSG3413_G0120_X-3850000_Y-950000.zarr",
            #     "roi_percent": 67.0798931817625,
            #     "aws_params": {
            #         "outputStore": "ITS_LIVE_vel_EPSG3413_G0120_X-3850000_Y-950000.zarr",
            #         "outputBucket": "s3://its-live-data.jpl.nasa.gov/datacubes/v01/N50W120",
            #         "targetProjection": "3413",
            #         "polygon": "[[-3900000, -1000000], [-3800000, -1000000], [-3800000, -900000], [-3900000, -900000], [-3900000, -1000000]]",
            #         "gridCellSize": "120",
            #         "chunks": "250"
            #     },
            #     "aws": {
            #         "queue": "datacube-r5d",
            #         "job_definition": "arn:aws:batch:us-west-2:849259517355:job-definition/datacube-v01:3",
            #         "response": {
            #             "ResponseMetadata": {
            #                 "RequestId": "2cd5a16e-ba89-4d2a-aa87-11ce39e6ca2b",
            #                 "HTTPStatusCode": 200,
            #                 "HTTPHeaders": {
            #                     "date": "Fri, 23 Jul 2021 19:03:34 GMT",
            #                     "content-type": "application/json",
            #                     "content-length": "162",
            #                     "connection": "keep-alive",
            #                     "x-amzn-requestid": "2cd5a16e-ba89-4d2a-aa87-11ce39e6ca2b",
            #                     "access-control-allow-origin": "*",
            #                     "x-amz-apigw-id": "C7_U9FBSPHcFkTQ=",
            #                     "access-control-expose-headers": "X-amzn-errortype,X-amzn-requestid,X-amzn-errormessage,X-amzn-trace-id,X-amz-apigw-id,date",
            #                     "x-amzn-trace-id": "Root=1-60fb1285-6497e68565eb1dca05acf929"
            #                 },
            #                 "RetryAttempts": 0
            #             },
            #             "jobArn": "arn:aws:batch:us-west-2:849259517355:job/c1a4dc18-f939-4574-baec-f77fd4b8168b",
            #             "jobName": "datacube_v01",
            #             "jobId": "c1a4dc18-f939-4574-baec-f77fd4b8168b"
            #         }
            #     }
            # },
            # logging.info(f"Keys: {list(each_job_info['aws']['response'].keys())}")

            logging.info(f"Checking for {each_job_info[AWS.AWS][AWS.RESPONSE][AWS.JOB_ID]}")

            if each_job_info[AWS.AWS][AWS.RESPONSE][AWS.JOB_ID] in failed_ids:
                # Extract cube parameters from previous run
                cube_params = each_job_info[AWS.PARAMS]
                logging.info(f'Re-submitting job {each_job_info[AWS.AWS][AWS.RESPONSE][AWS.JOB_ID]} for {each_job_info[AWS.S3_FILENAME]}')

                # Hack to create long running job - to test s3fs issue
                # if cube_filename != 'ITS_LIVE_vel_EPSG3413_G0120_X-550000_Y-1450000.zarr':
                #     continue

                # Submit AWS Batch job
                response = None
                if self.is_dry_run is False:
                    cube_filename = os.path.basename(each_job_info[AWS.S3_FILENAME])
                    cube_filename = cube_filename.replace('.zarr', '')
                    response = DataCubeBatchReprocess.CLIENT.submit_job(
                        jobName=cube_filename,
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

                # Overwrite runtime info for just submitted job
                each_job_info[AWS.AWS]['queue'] = self.batch_queue
                each_job_info[AWS.AWS]['job_definition'] = self.batch_job
                each_job_info[AWS.AWS][AWS.RESPONSE] = response

                num_jobs += 1
                jobs.append(each_job_info)

        logging.info(f"Number of batch jobs submitted: {num_jobs}")

        # Write job info to the json file
        logging.info(f"Writing AWS job info to the {output_job_file}...")
        with open(output_job_file, 'w') as output_fhandle:
            json.dump(jobs, output_fhandle, indent=4)

        return

def main(
    dry_run: bool,
    job_file: str,
    failed_job_file: str,
    batch_job: str,
    batch_queue: str,
    output_job_file: str):
    """
    Driver to re-submit multiple Batch jobs to AWS.
    """
    run_batch = DataCubeBatchReprocess(
        batch_job,
        batch_queue,
        dry_run
    )
    run_batch(job_file, failed_job_file, output_job_file)


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
        '-j', '--jobFile',
        type=Path,
        help="JSON file that stores job definitions from previous run"
    )
    parser.add_argument(
        '-f', '--failedJobFile',
        type=Path,
        help="List of failed AWS Batch job IDs"
    )
    parser.add_argument(
        '-d', '--batchJobDefinition',
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
        default='datacube_batch_reprocess_jobs.json',
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
        args.jobFile,
        args.failedJobFile,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.outputJobFile
    )

    logging.info(f"Done")
