"""
Script to prepare V1 ITS_LIVE mosaics to be ingested by NSIDC: see nsidc_vel_image_pairs.py
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
from nsidc_vel_image_pairs import NSIDCMeta, NSIDCFormat, get_attr_value

class Encoding:
    """
    Encoding settings for writing ITS_LIVE mosaics to the NetCDF format file.
    """
    MOSAICS = {
        'v':                {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vx':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vy':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vx_err':           {'_FillValue': 0.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
        'vy_err':           {'_FillValue': 0.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
        'v_err':            {'_FillValue': 0.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
        'date':             {'_FillValue': 0.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
        'dt':               {'_FillValue': 0.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
        'count':            {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_max':    {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'ocean':            {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'rock':             {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'ice':              {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'mapping':          {'_FillValue': None, 'dtype': np.float32},
        'x':                {'_FillValue': None},
        'y':                {'_FillValue': None}
    }

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
        start_year = 1985
        stop_year = 2018
        if year != 0:
            start_year = year
            stop_year = year

        # This is annual mosaic, set start/end dates for the year
        begin_date = datetime(start_year, 1, 1)
        end_date = datetime(stop_year, 12, 31)

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={infile}\n')
            fh.write(f'VersionID_local=001\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Hard-code the values for annual and static mosaics
            fh.write("Begin_time=00:00:01.000\n")
            fh.write("End_time=23:59:59.000\n")

            # Append premet with sensor info
            for each_sensor in [
                NSIDCMeta.L8,
                NSIDCMeta.L7,
                NSIDCMeta.L5,
                NSIDCMeta.L4
            ]:
                fh.write(f"Container=AssociatedPlatformInstrumentSensor\n")
                fh.write(f"AssociatedPlatformShortName={NSIDCMeta.ShortName[each_sensor].platform}\n")
                fh.write(f"AssociatedInstrumentShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")
                fh.write(f"AssociatedSensorShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")

        return meta_filename

    @staticmethod
    def create_spatial_file(infile: str):
        """
        Create spatial file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE granule
        """
        meta_filename = f'{infile}.spatial'

        with xr.open_dataset(infile, engine='h5netcdf') as ds:
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

            epsgcode = int(get_attr_value(ds['mapping'].attrs['spatial_epsg']))
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
    Class to prepare V1 ITS_LIVE mosaics for ingest by NSIDC:
    1. Make V1 ITS_LIVE data CF-1.8 convention compliant.
    2. Generate metadata files required by NSIDC ingest (premet and spacial metadata files
       which are generated per each data product being ingested).
    """
    # Pattern to collect input files in AWS S3 bucket
    GLOB_PATTERN = '*.nc'

    # Re-chunk input mosaic to speed up read of the data
    CHUNK_SIZE = 1000

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

        # Granule files as read from the S3 granule summary file
        glob_pattern = os.path.join(s3_bucket, s3_dir, NSIDCMosaicFormat.GLOB_PATTERN)
        logging.info(f"Glob mosaics: {glob_pattern}")

        self.infiles = self.s3.glob(f'{glob_pattern}')

        logging.info(f"Got {len(self.infiles)} files")

    def __call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        ATTN: This method implements sequential processing for debugging purposes only.

        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spatial
        and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to process, exiting.")
            return

        # Current start index into list of files to process
        start = 0

        file_list = []
        while total_num_files > 0:
            logging.info(f"Starting {self.infiles[start]} {start} out of {init_total_files} total files")
            results = NSIDCMosaicFormat.fix_file(target_bucket, target_dir, self.infiles[start], self.s3)
            logging.info("\n-->".join(results))

            total_num_files -= 1
            start += 1

    def no__call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spacial
        and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info(f"Nothing to process, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        file_list = []
        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            tasks = [dask.delayed(NSIDCMosaicFormat.fix_file)(target_bucket, target_dir, each, self.s3) for each in self.infiles[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks

    @staticmethod
    def fix_file(target_bucket: str, target_dir: str, infilewithpath: str, s3):
        """
        Fix granule format and create corresponding metadata files as required by NSIDC.
        """
        filename_tokens = infilewithpath.split(os.path.sep)
        directory = os.path.sep.join(filename_tokens[1:-1])

        filename = filename_tokens[-1]

        # Extract tokens from the filename
        # Get tokens for the first image name
        tokens = filename.split('_')
        year_str = tokens[-1].replace('.nc', '')
        year = int(year_str)

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Processing {infilewithpath} into new format']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_file = os.path.join(target_dir, filename)

        if NSIDCFormat.object_exists(bucket, bucket_file):
            msgs.append(f'WARNING: {bucket.name}/{bucket_file} already exists, skipping file')
            return msgs

        s3_client = boto3.client('s3')

        file_path = os.path.sep.join(filename_tokens[1:])
        local_file = filename + '.local'

        # Download file locally - takes too long to read the whole mosaic file
        # from S3 in order for it to write fixed dataset locally
        s3_client.download_file(target_bucket, file_path, local_file)

        with xr.open_dataset(local_file, engine='h5netcdf') as ds:
            mapping = None

            if DataVars.UTM_PROJECTION in ds:
                mapping = ds[DataVars.UTM_PROJECTION]

            elif DataVars.POLAR_STEREOGRAPHIC in ds:
                mapping = ds[DataVars.POLAR_STEREOGRAPHIC]

            epsgcode = int(get_attr_value(mapping.attrs['spatial_epsg']))

            msgs.extend(
                NSIDCFormat.process_nc_file(
                    ds,
                    filename,
                    Encoding.MOSAICS,
                    epsgcode,
                    NSIDCMosaicFormat.CHUNK_SIZE
                )
            )

        # Remove local copy of the file
        msgs.append(f"Removing local {local_file}")
        os.unlink(local_file)

        # Copy new granule to S3 bucket
        msgs.extend(
            NSIDCFormat.upload_to_s3(filename, target_dir, target_bucket, s3_client, remove_original_file=False)
        )

        # Create spacial and premet metadata files, and copy them to S3 bucket
        meta_file = NSIDCMosaicsMeta.create_premet_file(filename, year)
        msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        meta_file = NSIDCMosaicsMeta.create_spatial_file(filename)
        msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        msgs.append(f"Removing local {filename}")
        os.unlink(filename)

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
        '-source_dir',
        type=str,
        default='velocity_mosaic/landsat/v00.0/annual',
        help='AWS S3 directory that stores input mosaics [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/v01/mosaics/annual',
        help='AWS S3 directory that stores processed mosaics [%(default)s]'
    )

    parser.add_argument(
        '-chunk_by',
        action='store',
        type=int,
        default=4,
        help='Number of granules to process in parallel [%(default)d]'
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

    args = parser.parse_args()

    NSIDCFormat.DRY_RUN = args.dryrun

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = NSIDCMosaicFormat(args.bucket, args.source_dir)

    nsidc_format(
        args.bucket,
        args.target_dir,
        args.chunk_by,
        args.dask_workers
    )

    logging.info('Done.')
