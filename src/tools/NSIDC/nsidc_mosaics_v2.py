"""
Script to prepare V2 ITS_LIVE mosaics to be ingested by NSIDC: see nsidc_vel_image_pairs_v2.py
for the details.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
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
from nsidc_vel_image_pairs import NSIDCFormat
from nsidc_vel_image_pairs_v2 import NSIDCMeta


class NSIDCMosaicsMeta:
    """
    Class to create premet and spacial files for each of the mosaics.

    Example of premet file:
    =======================
    FileName=PAT_G0120_0000.nc
    VersionID_local=001
    Begin_date=2019-01-01
    End_date=2019-12-31
    Begin_time=00:00:01.000
    End_time=23:59:59.000
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-8
    AssociatedInstrumentShortName=OLI
    AssociatedSensorShortName=OLI
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-7
    AssociatedInstrumentShortName=ETM+
    AssociatedSensorShortName=ETM+
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-4
    AssociatedInstrumentShortName=TM
    AssociatedSensorShortName=TM
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-5
    AssociatedInstrumentShortName=TM
    AssociatedSensorShortName=TM

    Example of spatial file:
    ========================
    -94.32	71.86
    -99.41	71.67
    -94.69	73.3
    -100.22	73.09
    """

    @staticmethod
    def create_premet_file(infile: str, year: int = 0):
        """
        Create premet file that corresponds to the input mosaics file.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE granule
        url_tokens_1: Parsed out filename tokens that correspond to the first image of the pair
        url_tokens_2: Parsed out filename tokens that correspond to the second image of the pair
        """
        # Hard-code values for static mosaics
        start_year = 2014
        stop_year = 2022
        if year != 0:
            start_year = year
            stop_year = year

        # This is annual mosaic, set start/end dates for the year
        begin_date = datetime(start_year, 1, 1)
        end_date = datetime(stop_year, 12, 31)

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={infile}\n')
            fh.write(f'VersionID_local=002\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Hard-code the values for annual and static mosaics
            fh.write("Begin_time=00:00:01.000\n")
            fh.write("End_time=23:59:59.000\n")

            # Append premet with sensor info
            for each_sensor in [
                NSIDCMeta.LC9,
                NSIDCMeta.LO9,
                NSIDCMeta.LC8,
                NSIDCMeta.LO8,
                NSIDCMeta.L7,
                NSIDCMeta.L5,
                NSIDCMeta.L4,
                NSIDCMeta.S1A,
                NSIDCMeta.S1B,
                NSIDCMeta.S2A,
                NSIDCMeta.S2B,
            ]:
                fh.write(f"Container=AssociatedPlatformInstrumentSensor\n")
                fh.write(f"AssociatedPlatformShortName={NSIDCMeta.ShortName[each_sensor].platform}\n")
                fh.write(f"AssociatedInstrumentShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")
                fh.write(f"AssociatedSensorShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")

        return meta_filename

    @staticmethod
    def create_spatial_file(ds: xr.Dataset, infile: str):
        """
        Create spatial file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        ds: xarray.Dataset object that represents the granule.
        infile: Filename of the input ITS_LIVE granule
        """
        meta_filename = f'{infile}.spatial'

        xvals = ds.x.values
        yvals = ds.y.values
        pix_size_x = xvals[1] - xvals[0]
        pix_size_y = yvals[1] - yvals[0]

        # minval_x, pix_size_x, _, maxval_y, _, pix_size_y = [float(x) for x in ds['mapping'].attrs['GeoTransform'].split()]

        # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
        projection_cf_minx = xvals[0] - pix_size_x/2.0
        projection_cf_maxx = xvals[-1] + pix_size_x/2.0
        projection_cf_miny = yvals[-1] + pix_size_y/2.0 # pix_size_y is negative!
        projection_cf_maxy = yvals[0] - pix_size_y/2.0  # pix_size_y is negative!

        epsgcode = ds[DataVars.MAPPING].attrs['spatial_epsg']
        epsgcode_str = f'EPSG:{epsgcode}'

        if epsgcode == NSIDCFormat.ESRI_CODE:
            epsgcode_str = f'ESRI:{epsgcode}'

        transformer = pyproj.Transformer.from_crs(epsgcode_str, "EPSG:4326", always_xy=True) # ensure lonlat output order

        # Convert coordinates to long/lat
        ll_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_miny),decimals = 2).tolist()
        lr_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_miny),decimals = 2).tolist()
        ur_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_maxy),decimals = 2).tolist()
        ul_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_maxy),decimals = 2).tolist()

        # Write to spatial file
        with open(meta_filename, 'w') as fh:
            for long, lat in [ul_lonlat, ur_lonlat, lr_lonlat, ll_lonlat]:
                fh.write(f"{long}\t{lat}\n")

        return meta_filename

class NSIDCMosaicFormat:
    """
    Class to prepare V2 ITS_LIVE mosaics for ingest by NSIDC.
    It generates metadata files required by NSIDC ingest (premet and spacial metadata files
    which are generated per each data product being ingested).
    """
    # Pattern to collect input files in AWS S3 bucket
    GLOB_PATTERN = '*.nc'

    def __init__(self, s3_bucket: str, s3_dir: str):
        """
        Initialize the object.

        Inputs:
        =======
        s3_dir: Directory in AWS S3 bucket that stores mosaics files.
        start_index: Start index into file list to process.
        stop_index: Stop index into file list to process.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)
        self.s3_bucket = s3_bucket
        self.s3_dir = s3_dir

        # Granule files as read from the S3 granule summary file
        glob_pattern = os.path.join(s3_bucket, s3_dir, NSIDCMosaicFormat.GLOB_PATTERN)
        logging.info(f"Glob mosaics: {glob_pattern}")

        self.infiles = self.s3.glob(f'{glob_pattern}')

        logging.info(f"Got {len(self.infiles)} files to process")

    def __call__(self):
        """
        Create NSIDC meta files (spatial and premet) for ITS_LIVE v2 mosaics.
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to process, exiting.")
            return

        # Current start index into list of files to process
        start = 0

        file_list = []
        while start < total_num_files:
            logging.info(f"Starting {self.infiles[start]} {start} out of {init_total_files} total files")
            results = NSIDCMosaicFormat.process_file(self.s3_bucket, self.s3_dir, self.infiles[start], self.s3)
            logging.info("\n-->".join(results))

            start += 1

    @staticmethod
    def process_file(target_bucket: str, target_dir: str, infilewithpath: str, s3: s3fs.S3FileSystem):
        """
        Fix granule format and create corresponding metadata files as required by NSIDC.
        """
        filename_tokens = infilewithpath.split(os.path.sep)
        directory = os.path.sep.join(filename_tokens[1:-1])

        filename = filename_tokens[-1]

        # Extract tokens from the filename
        tokens = filename.split('_')
        year = int(tokens[-2])

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Processing {infilewithpath}']

        s3_client = boto3.client('s3')

        with s3.open(infilewithpath, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=NSIDCMeta.NC_ENGINE) as ds:
                # Create spacial and premet metadata files, and copy them to S3 bucket
                meta_file = NSIDCMosaicsMeta.create_premet_file(ds, filename, year)

                msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

                meta_file = NSIDCMosaicsMeta.create_spatial_file(filename)
                msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

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
        help='AWS S3 bucket to store ITS_LIVE data products to [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/velocity_mosaic_sample/v2/static',
        help='AWS S3 directory that stores input mosaics [%(default)s]'
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

    nsidc_format = NSIDCMosaicFormat(args.bucket, args.target_dir)
    nsidc_format()

    logging.info('Done.')
