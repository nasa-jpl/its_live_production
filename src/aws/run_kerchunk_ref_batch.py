"""
Script to drive Batch processing for velocity image pairs kerchunk references generation at AWS.

It accepts a top level file that lists all catalog geojsons for the mission and submits one AWS Batch job
per each catalog geojson.
"""
import argparse
import boto3
import fsspec
import json
import logging
import os
import sys


class KerchunkRefBatch:
    """
    Class to manage Batch job submission at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    def __init__(self, catalog_list_file: str, batch_job: str, batch_queue: str, is_dry_run: bool):
        """Initialize object.

        Args:
            catalog_list_file (str): AWS S3 URL to the file that lists all catalog geojsons.
            batch_job (str): AWS Batch job to use for processing.
            batch_queue (str): AWS Batch queue to use for processing.
            is_dry_run (bool): Flag if run the sript in "dry" mode meaning without submitting job to AWS.
        """
        self.catalog_list_file = catalog_list_file
        self.batch_job = batch_job
        self.batch_queue = batch_queue
        self.is_dry_run = is_dry_run

        self.catalogs = None
        with fsspec.open(catalog_list_file, mode='r', anon=True) as f:
            self.catalogs = f.readlines()

        logging.info(f'Got {len(self.catalogs)} to process from {catalog_list_file}')

        logging.info(f'First catalog: {self.catalogs[0]}')
        logging.info(f'Last catalog: {self.catalogs[-1]}')

    def __call__(self, target_bucket_dir: str, job_file: str, start_index: int = 0):
        """Submit Batch jobs to AWS.

        Args:
            target_bucket_dir (str): AWS S3 bucket and top level directory to store kerchunk references to.
            job_file (str): File to keep trac of all submitted jobs.
            start_index (int): Start index into catalogs list to begin processing with. Default is 0.
        """
        # List of submitted catalog geojson Batch jobs and AWS response
        jobs = []

        # Number of submitted jobs
        num_jobs = 0

        num_to_process = len(self.catalogs)

        # For debugging only
        # num_to_process = 1

        for index in range(start_index, num_to_process):
            catalog_params = {
                'catalog_file': self.catalog_list_file,
                'start_index': str(index),
                'out_dir': target_bucket_dir
            }
            logging.info(f'Job params for {index} job_name={os.path.basename(self.catalogs[index]).split(".")[0]}: {catalog_params}')

            # Submit AWS Batch job
            response = None
            if self.is_dry_run is False:
                response = KerchunkRefBatch.CLIENT.submit_job(
                    jobName=os.path.basename(self.catalogs[index]).split('.')[0],
                    jobQueue=self.batch_queue,
                    jobDefinition=self.batch_job,
                    parameters=catalog_params,
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
                        # 'attemptDurationSeconds': 86400  # 24 hours
                        # Change to 48 hours (172800)
                        # Change to 72 hours (259200) for very large cubes
                        # Change to 4 days (345600) to support very large cubes
                        'attemptDurationSeconds': 86400
                    }
                )

                logging.info(f"Response: {response}")

                # Does not really work - AWS piles up the jobs in the queue,
                # then starts a whole bunch at once anyway
                # # Sleep for 30 seconds to make sure that all AWS Batch jobs
                # # are not started at the same time
                # time.sleep(30)

            num_jobs += 1

            jobs.append({
                'catalog_geojson': self.catalogs[index],
                'aws_params': catalog_params,
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
    catalog_list_file: str,
    batch_job: str,
    batch_queue: str,
    target_bucket_dir: str,
    job_file: str,
    start_index: int
):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0
    run_batch = KerchunkRefBatch(catalog_list_file, batch_job, batch_queue, dry_run)
    run_batch(target_bucket_dir, job_file, start_index)


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
        '-c', '--catalogListFile',
        type=str,
        action='store',
        default=None,
        help="AWS S3 URL for the file that stores all catalog geojsons [%(default)s]."
    )
    parser.add_argument(
        '-s', '--catalogStartIndex',
        type=int,
        action='store',
        default=0,
        help="Index into catalog geojson list to begin processing for [%(default)s]."
    )
    parser.add_argument(
        '-t', '--targetS3Dir',
        type=str,
        action='store',
        default='s3://its-live-project/velocity_image_pair/kerchunk_refs',
        help="Destination S3 bucket for the kerchunk references [%(default)s]. It should point to the mission specific"
                "sub-directory of this default top-level directory."
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        default='arn:aws:batch:us-west-2:849259517355:job-definition/kerchunk-ref-30Gb:1',
        help="AWS Batch job definition to use [%(default)s]"
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        default='arn:aws:batch:us-west-2:849259517355:job-queue/datacube-ondemand-4vCPU-32GB',
        help="AWS Batch job queue to use [%(default)s]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='kerchunk_ref_batch_jobs.json',
        help="File to capture submitted kerchunk AWS Batch jobs [%(default)s]"
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Parsed out command-line arguments: {args}")

    return args


if __name__ == '__main__':

    args = parse_args()

    main(
        args.dryrun,
        args.catalogListFile,
        args.batchJobDefinition,
        args.batchJobQueue,
        args.targetS3Dir,
        args.outputJobFile,
        args.catalogStartIndex
    )

    logging.info("Done")
