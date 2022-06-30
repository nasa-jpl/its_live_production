"""
Script to fix AUTHORITY[\"EPSG\",\"102027\"] string to AUTHORITY[\"ESRI\",\"102027\"]
within mapping's "spatial_ref" attribute of V1 HMA ITS_LIVE mosaics (with EPGS code of 102027)
to be ingested by NSIDC.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
import logging
import os
import s3fs
import sys
import xarray as xr

# Local imports
from itscube_types import DataVars
from nsidc_vel_image_pairs import NSIDCMeta, NSIDCFormat, get_attr_value
from nsidc_mosaics import Encoding, NSIDCMosaicFormat


class FixESRIMosaics:
    """
    Class to fix AUTHORITY[\"EPSG\",\"102027\"] string to AUTHORITY[\"ESRI\",\"102027\"]
    within mapping's "spatial_ref" attribute of V1 ITS_LIVE HMA mosaics for ingest by NSIDC.
    This is additional fixes to the "mapping" data variable that were not in place
    when nsidc_mosaics.py script was run against V1 annual mosaics.
    """
    GLOB_PATTERN = 'HMA*.nc'

    SPATIAL_REF = 'spatial_ref'
    SPATIAL_EPSG = 'spatial_epsg'

    def __init__(self, s3_bucket: str, s3_dir: str):
        """
        Initialize the object.

        Inputs:
        =======
        s3_dir: Directory in AWS S3 bucket that stores mosaics files.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        # Granule files as read from the S3 granule summary file
        glob_pattern = os.path.join(s3_bucket, s3_dir, FixESRIMosaics.GLOB_PATTERN)
        logging.info(f"Glob mosaics: {glob_pattern}")

        self.infiles = self.s3.glob(f'{glob_pattern}')

        logging.info(f"Got {len(self.infiles)} files")

    def __call__(self, target_bucket: dir, target_dir: dir):
        """
        Fix ITS_LIVE mosaics's mapping "spatial_ref" attribute
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to process, exiting.")
            return

        file_list = []
        start = 0
        while total_num_files > 0:
            logging.info(f"Starting mosaics {start} out of {init_total_files} total files")
            results = FixESRIMosaics.fix_file(target_bucket, target_dir, self.infiles[start], self.s3)
            logging.info("\n-->".join(results))

        start += 1
        total_num_files -= 1

    @staticmethod
    def fix_file(target_bucket: str, target_dir: str, infilewithpath: str, s3):
        """
        Fix "mapping" attributes for ingest by NSIDC.
        """
        filename_tokens = infilewithpath.split(os.path.sep)
        directory = os.path.sep.join(filename_tokens[1:-1])

        filename = filename_tokens[-1]

        # Extract tokens from the filename
        tokens = filename.split('_')
        year_str = tokens[-1].replace('.nc', '')
        year = int(year_str)

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Processing {infilewithpath} into new format']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_file = os.path.join(target_dir, filename)

        s3_client = boto3.client('s3')

        file_path = os.path.sep.join(filename_tokens[1:])
        local_file = filename + '.local'

        # Download file locally - takes too long to read the whole mosaic file
        # from S3 in order for it to write fixed dataset locally
        logging.info(f"Copying {infilewithpath} locally to {local_file}...")
        s3_client.download_file(target_bucket, file_path, local_file)

        with xr.open_dataset(local_file, engine='h5netcdf') as ds:
            mapping = ds[DataVars.MAPPING]
            epsgcode = int(get_attr_value(mapping.attrs[FixESRIMosaics.SPATIAL_EPSG]))

            msgs.extend(
                FixESRIMosaics.process_nc_file(
                    ds,
                    filename,
                    Encoding.MOSAICS,
                    epsgcode,
                    NSIDCMosaicFormat.CHUNK_SIZE
                )
            )

        # Remove local copy of the file: don't remove for now - save as original file backup
        # msgs.append(f"Removing original local {local_file}")
        # os.unlink(local_file)

        # Copy new granule to S3 bucket
        msgs.extend(
            NSIDCFormat.upload_to_s3(filename, target_dir, target_bucket, s3_client, remove_original_file=True)
        )

        return msgs

    @staticmethod
    def process_nc_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        epsg_code: int,
        chunk_size: int
    ):
        """
        Fix "mapping" attribute.
        """
        _epsg = 'EPSG'
        _esri = 'ESRI'

        msgs = []
        if epsg_code != NSIDCFormat.ESRI_CODE:
            msgs.append(f'Unexpected EPSG code: {epsg_code} for {new_filename}')

        # Extra fixes are required to the mapping data variable
        ds[DataVars.MAPPING].attrs[FixESRIMosaics.SPATIAL_REF] = ds[DataVars.MAPPING].attrs[FixESRIMosaics.SPATIAL_REF].replace(_epsg, _esri)

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        logging.info(f'Saving fixed file to {new_filename}')
        ds.to_netcdf(new_filename, engine='h5netcdf', encoding = encoding_params)

        return msgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description="""
           Fix ITS_LIVE V1 mosaics to be CF compliant for ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE granules to [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/v01/mosaics/annual',
        help='AWS S3 directory that stores processed mosaics [%(default)s]'
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually process any granules'
    )

    args = parser.parse_args()

    NSIDCFormat.DRY_RUN = args.dryrun

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = FixESRIMosaics(args.bucket, args.target_dir)

    nsidc_format(args.bucket, args.target_dir)

    logging.info('Done.')
