"""
Script to collect a list of some deleted by accident V1 ITS_LIVE fixed granules that were prepared
to be ingested by NSIDC. Feed this list to nsidc_vel_image_pairs.py to recreate the files.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
from botocore.exceptions import ClientError
import collections
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import json
import logging
import numpy as np
import os
import pyproj
import s3fs
import sys
import xarray as xr

# Local imports
from itscube_types import DataVars
from nsidc_vel_image_pairs import NSIDCFormat, get_tokens_from_filename


class FindGranulesToProcess:
    """
    Class to find deleted by accident fixed v1 granules for ingest by NSIDC:
    either fixed granule or any of its metadata files are missing in the target directory.
    The granule is from one of 32624, 32625 or 32626 EPSG projections.
    """
    EPSG_CODES = ['32624', '32625', '32626']

    # Flag to enable dry run: don't process any granules, just report what would be processed
    DRY_RUN = False

    def __init__(self):
        """
        Initialize the object.
        """
        # S3FS to access files stored in S3 bucket
        s3 = s3fs.S3FileSystem(anon=True)

        # Granule files as read from the S3 granule summary file
        self.infiles = None
        logging.info(f"Opening granules file: {NSIDCFormat.GRANULES_FILE}")

        with s3.open(NSIDCFormat.GRANULES_FILE, 'r') as ins3file:
            self.infiles = json.load(ins3file)
            logging.info(f"Loaded {len(self.infiles)} granules from '{NSIDCFormat.GRANULES_FILE}'")

        # Keep only granules of selected EPGS codes
        self.infiles = [each for each in self.infiles if each.split('/')[-2] in FindGranulesToProcess.EPSG_CODES]
        logging.info(f"Keeping {len(self.infiles)} granules that correspond to '{FindGranulesToProcess.EPSG_CODES}'")

    def no__call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        ATTN: This method implements sequetial processing for debugging purposes only.

        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spatial
        and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to catalog, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        file_list = []
        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            for each in self.infiles[start:start+num_tasks]:
                results = NSIDCFormat.fix_granule(target_bucket, target_dir, each, self.s3)
                logging.info("-->".join(results))

            total_num_files -= num_tasks
            start += num_tasks

    def __call__(self, target_bucket, target_dir, chunk_size=100, num_dask_workers=4):
        """
        Collect ITS_LIVE granules that need to be reprocessed.
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to catalog, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        file_list = []
        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            tasks = [dask.delayed(FindGranulesToProcess.check_granule)(target_bucket, target_dir, each) for each in self.infiles[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_file in results[0]:
                # logging.info("\n-->".join(each_result))
                if each_file:
                    file_list.append(each_file)

            total_num_files -= num_tasks
            start += num_tasks

        logging.info(f'Found {len(file_list)} granules to reprocess')

        json_file = 'nsidc_granules_to_fix.json'
        with open(json_file, 'w') as fh:
            json.dump(file_list, fh, indent=3)

        logging.info(f'Wrote files to fix to {json_file}')

    @staticmethod
    def check_granule(target_bucket: str, target_dir: str, infilewithpath: str):
        """
        Fix granule format and create corresponding metadata files as required by NSIDC.
        """
        filename_tokens = infilewithpath.split('/')
        directory = '/'.join(filename_tokens[1:-1])
        filename = filename_tokens[-1]

        # Parent subdirectory is EPSG code
        epsg_code = int(filename_tokens[-2])

        # Extract tokens from the filename
        url_tokens_1, url_tokens_2 = get_tokens_from_filename(filename)

        # Format target filename for the granule to be within 80 characters long
        # from
        # LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_X_LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_G0240V01_PXYZ.nc
        # to
        # LXSSLLLLPPPRRRYYYYMMDDCCTXX_LXSSLLLLPPPRRRYYYYMMDDCCTX_EEEEE_G0240V01_XYZ.nc
        new_filename = ''.join(url_tokens_1[:4]) + url_tokens_1[5] + url_tokens_1[6] + '_'
        new_filename += ''.join(url_tokens_2[:4]) + url_tokens_2[5] + url_tokens_2[6]
        new_filename += f'_{epsg_code:05d}_'
        new_filename += url_tokens_2[7]
        new_filename += '_'
        new_filename += url_tokens_2[8]

        # logging.info(f'filename: {infilewithpath}')
        # logging.info(f'new_filename: {new_filename}')
        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_granule = os.path.join(target_dir, new_filename)

        # Store granules under 'landsat8' sub-directory in new S3 bucket
        if not NSIDCFormat.object_exists(bucket, bucket_granule):
            # New granule does not exist, register it
            return infilewithpath

        # Check if corresponding metadata files exist:
        meta_filename = f'{bucket_granule}.premet'

        if not NSIDCFormat.object_exists(bucket, meta_filename):
            # Meta file does not exist, register it
            return infilewithpath

        meta_filename = f'{bucket_granule}.spatial'

        if not NSIDCFormat.object_exists(bucket, meta_filename):
            # Meta file does not exist, register it
            return infilewithpath

        # All three files exist, return None to register
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description="""
           Find ITS_LIVE V1 velocity image pairs granules that need to be convered
           to CF compliant format for ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-catalog_dir',
        action='store',
        type=str,
        default='catalog_geojson/landsat/v01',
        help='Output path for feature collections [%(default)s]'
    )

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE granules to [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/v01/velocity_image_pair/',
        help='AWS S3 directory that stores processed granules [%(default)s]'
    )

    parser.add_argument(
        '-granules_file',
        action='store',
        type=str,
        default='used_granules_landsat.json',
        help='Filename with JSON list of granules [%(default)s], file is stored in  "-catalog_dir"'
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually process any granules'
    )

    args = parser.parse_args()

    NSIDCFormat.GRANULES_FILE = os.path.join(args.bucket, args.catalog_dir, args.granules_file)
    NSIDCFormat.DRY_RUN = args.dryrun

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = FindGranulesToProcess()
    nsidc_format(args.bucket, args.target_dir)

    logging.info('Done.')
