"""
Script to prepare V2 ITS_LIVE granules to be ingested by NSIDC:

* Generate metadata file
* Generate spacial info file

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import dask
from dask.diagnostics import ProgressBar
import gc
import json
import logging
import os
import s3fs
import sys
import time

# Local imports
import nsidc_meta_files

# Date format as it appears in granules filenames of optical format:
# LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc
DATE_FORMAT = "%Y%m%d"

# Date and time format as it appears in granules filenames in radar format:
# S1A_IW_SLC__1SSH_20170221T204710_20170221T204737_015387_0193F6_AB07_X_S1B_IW_SLC__1SSH_20170227T204628_20170227T204655_004491_007D11_6654_G0240V02_P094.nc
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S"

# Number of retries when encountering AWS S3 download/upload error
_NUM_AWS_COPY_RETRIES = 3

# Number of seconds between retries to access AWS S3 bucket
_AWS_COPY_SLEEP_SECONDS = 3


class NSIDCFormat:
    """
    Class to prepare V2 ITS_LIVE data for ingest by NSIDC. It requires generation of
    2 metadata files required by NSIDC ingest: premet and spacial metadata files
    which are generated per each data product being ingested.
    """
    GRANULES_FILE = 'used_granules.json'

    # Flag to enable dry run: don't process any granules, just report what would be processed
    DRY_RUN = False

    # Token to be present in granule's path - this is to process only selected granules,
    # such as belonging to the same EPSG code
    PATH_TOKEN = ''

    def __init__(
        self,
        start_index: int = 0,
        stop_index: int = -1,
        local_file_list=None
    ):
        """
        Initialize the object.

        Arguments:
        start_index - Start index into the list of granules to process. Default is 0.
        stop_index - Stop index into the list of granules to process. Default is -1
            meaning to process to the end of the list.
        local_file_list - List of granules to process. This is a "hack" to run
            the script for specific set of granules such as sample set for NSIDC.
            Default is None.
        """
        # S3FS to access files stored in S3 bucket
        self.s3 = s3fs.S3FileSystem(anon=True)

        self.infiles = local_file_list

        if local_file_list is None:
            # Granule files as read from the S3 granule summary file (in catalog geojsons directory)
            self.infiles = None
            logging.info(f"Opening granules file: {NSIDCFormat.GRANULES_FILE}")

            with self.s3.open(NSIDCFormat.GRANULES_FILE, 'r') as ins3file:
                self.infiles = json.load(ins3file)
                logging.info(f"Loaded {len(self.infiles)} granules from '{NSIDCFormat.GRANULES_FILE}'")

        if start_index != 0 or stop_index != -1:
            # Start index is provided for the granule to begin with
            if stop_index != -1:
                self.infiles = self.infiles[start_index:stop_index]

            else:
                self.infiles = self.infiles[start_index:]

        logging.info(f"Starting with granule #{start_index} (stop={stop_index}), remains {len(self.infiles)} granules to process")

        if len(NSIDCFormat.PATH_TOKEN):
            # Leave only granules with provided token in their path
            self.infiles = [each for each in self.infiles if NSIDCFormat.PATH_TOKEN in each]
            logging.info(f'Leaving granules with {NSIDCFormat.PATH_TOKEN} in their path: {len(self.infiles)}')

    @staticmethod
    def object_exists(bucket, key: str) -> bool:
        """
        Returns true if file exists in the bucket, False otherwise.
        """
        try:
            bucket.Object(key).load()

        except ClientError:
            return False

        return True

    def __call__(self, target_bucket, chunk_size, num_dask_workers):
        """
        Prepare ITS_LIVE granules by creating corresponding NSIDC meta files (spacial
        and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info("Nothing to process, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            tasks = [
                dask.delayed(NSIDCFormat.process_granule)(target_bucket, each, self.s3)
                for each in self.infiles[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=num_dask_workers
                )

            for each_result in results[0]:
                if len(each_result):
                    # If there are any messages to report
                    logging.info("\n-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks

            gc.collect()

    @staticmethod
    def upload_to_s3(
        filename: str,
        target_dir: str,
        target_bucket: str,
        s3_client,
        remove_original_file: bool = True
    ):
        """
        Upload file to the AWS S3 bucket. If dryrun is enabled,
        just report the action.

        Inputs:
        filename: Filename to be uploaded.
        target_dir: Target directory in the S3 bucket.
        target_bucket: Target bucket in the S3.
        s3_client: S3 client object.
        remove_original_file: Flag to remove the original file after uploading. Default is True.
        """
        msgs = []
        target_filename = os.path.join(target_dir, filename)

        try:
            msg = ""
            if NSIDCFormat.DRY_RUN:
                msg = "DRYRUN: "
                msgs.append(f"{msg}Uploading {filename} to {target_bucket}/{target_filename}")

            if not NSIDCFormat.DRY_RUN:
                s3_client.upload_file(filename, target_bucket, target_filename)

                if remove_original_file:
                    # msgs.append(f"Removing local {filename}")
                    os.unlink(filename)

        except ClientError as exc:
            msgs.append(f"ERROR: {exc}")

        return msgs

    @staticmethod
    def process_granule(target_bucket: str, infilewithpath: str, s3: s3fs.S3FileSystem):
        """
        Create corresponding metadata files for the granule as required by NSIDC.

        Inputs:
        target_bucket: Target AWS S3 bucket to copy metadata files to to.
        infilewithpath: Path to input granule file.
        """
        filename_tokens = infilewithpath.split('/')
        granule_directory = '/'.join(filename_tokens[1:-1])

        msgs = []

        # Automatically handle boto exceptions on file upload to s3 bucket
        files_are_copied = False
        num_retries = 0

        while not files_are_copied and num_retries < _NUM_AWS_COPY_RETRIES:
            try:
                meta_files = nsidc_meta_files.create_nsidc_meta_files(infilewithpath, s3)

                s3_client = boto3.client('s3')

                for each_file in meta_files:
                    # Place metadata files into the same s3 directory as the granule
                    NSIDCFormat.upload_to_s3(each_file, granule_directory, target_bucket, s3_client)

                files_are_copied = True

            except:
                msgs.append(f"Try #{num_retries + 1} exception processing {infilewithpath}: {sys.exc_info()}")
                num_retries += 1

                if num_retries < _NUM_AWS_COPY_RETRIES:
                    # Possible to have some other types of failures that are not related to AWS SlowDown,
                    # retry the copy for any kind of failure
                    # and _AWS_SLOW_DOWN_ERROR in command_return.stdout.decode('utf-8'):

                    # Sleep if it's not a last attempt to copy
                    time.sleep(_AWS_COPY_SLEEP_SECONDS)

                else:
                    # Don't retry, trigger an exception
                    num_retries = _NUM_AWS_COPY_RETRIES

        return msgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
           Prepare ITS_LIVE V2 velocity image pairs for ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-catalog_dir',
        action='store',
        type=str,
        default='catalog_geojson/landsatOLI/v02/',
        help='s3 path to the file (see "-granules_file" command-line option) that lists all granules to be ingested by NSIDC [%(default)s]'
    )

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE granules to [%(default)s]'
    )

    parser.add_argument(
        '-chunk_by',
        action='store',
        type=int,
        default=8,
        help='Number of granules to process in parallel [%(default)d]'
    )

    parser.add_argument(
        '-granules_file',
        action='store',
        type=str,
        default='used_granules.json',
        help='Filename with JSON list of granules [%(default)s], file is stored in  "-catalog_dir"'
    )

    parser.add_argument(
        '-start_index',
        action='store',
        type=int,
        default=0,
        help="Start index for the granule to fix [%(default)d]. "
             "Useful if need to continue previously interrupted process to prepare the granules, "
             "or need to split the load across multiple processes."
    )

    parser.add_argument(
        '-stop_index',
        action='store',
        type=int,
        default=-1,
        help="Stop index for the granules to process [%(default)d]. Usefull if need to split the job between multiple processes."
    )

    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers for processing [%(default)d]'
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually process any granules'
    )

    parser.add_argument(
        '--use_granule_file',
        action='store',
        type=str,
        default=None,
        help='Use provided file with granules to process [%(default)s]. This is used only if some of the granules need to be regenerated.'
    )

    parser.add_argument(
        '-t', '--path_token',
        type=str,
        default='',
        help="Optional path token to be present in granule's S3 target path in order for the granule to be processed [%(default)s]."
    )

    args = parser.parse_args()

    NSIDCFormat.GRANULES_FILE = os.path.join(args.bucket, args.catalog_dir, args.granules_file)
    NSIDCFormat.DRY_RUN = args.dryrun
    NSIDCFormat.PATH_TOKEN = args.path_token

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    # If file with granules to process is provided, just use it
    infiles = None
    if args.use_granule_file:
        with open(args.use_granule_file, 'r') as fh:
            infiles = json.load(fh)
            logging.info(f"Loaded {len(infiles)} granules from '{args.use_granule_file}'")

    nsidc_format = NSIDCFormat(
        args.start_index,
        args.stop_index,
        infiles
    )
    nsidc_format(
        args.bucket,
        args.chunk_by,
        args.dask_workers
    )

    logging.info('Done.')
