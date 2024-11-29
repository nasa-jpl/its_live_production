"""
Script to prepare V2 ITS_LIVE granules to be ingested by NSIDC:

* Generate metadata file
* Generate spacial info file

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import collections
import dask
from dask.diagnostics import ProgressBar
from dateutil.parser import parse
from datetime import datetime
import gc
import json
import logging
import numpy as np
import os
import pyproj
import s3fs
import xarray as xr

from itscube_types import DataVars

# Date format as it appears in granules filenames of optical format:
# LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc
DATE_FORMAT = "%Y%m%d"

# Date and time format as it appears in granules filenames in radar format:
# S1A_IW_SLC__1SSH_20170221T204710_20170221T204737_015387_0193F6_AB07_X_S1B_IW_SLC__1SSH_20170227T204628_20170227T204655_004491_007D11_6654_G0240V02_P094.nc
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S"


def get_tokens_from_filename(filename):
    """
    Extract acquisition/processing dates and path/row for two images from the
    optical granule filename, or start/end date/time and product unique ID for
    radar granule filename.
    """
    # ATTN: Optical format granules have different file naming convention than radar
    # format granules
    url_files = os.path.basename(filename).split('_X_')

    # Get tokens for the first image
    url_tokens_1 = url_files[0].split('_')

    # Extract info from second part of the granule's filename: corresponds to the second image
    url_tokens_2 = url_files[1].split('_')

    return (url_tokens_1, url_tokens_2)


# Collection to represent the mission and sensor combo
PlatformSensor = collections.namedtuple("PM", ['platform', 'sensor'])


class NSIDCMeta:
    """
    Class to create premet and spacial files for each of the granules.

    Example of premet file:
    =======================
    FileName=LC08_L1GT_001111_20140217_20170425_01_T2_X_LC08_L1GT_001111_20131113_20170428_01_T2_G0240V01_P006.nc
    VersionID_local=001
    Begin_date=2013-11-13
    End_date=2017-04-28
    Begin_time=00:00:01.000
    End_time=23:59:59.000
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-8
    AssociatedInstrumentShortName=OLI
    AssociatedSensorShortName=OLI
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=LANDSAT-8
    AssociatedInstrumentShortName=TIRS
    AssociatedSensorShortName=TIRS

    Example of spatial file:
    ========================
    -94.32	71.86
    -99.41	71.67
    -94.69	73.3
    -100.22	73.09
    """

    # Dictionary of metadata values based on the mission+sensor token
    # Optical data:
    LC9 = 'LC09'
    LO9 = 'LO09'
    LC8 = 'LC08'
    LO8 = 'LO08'
    L7 = 'LE07'
    L5 = 'LT05'
    L4 = 'LT04'
    S2A = 'S2A'
    S2B = 'S2B'

    # Radar data:
    S1A = 'S1A'
    S1B = 'S1B'

    ShortName = {
        LC9: PlatformSensor('LANDSAT-9', 'OLI'),
        LO9: PlatformSensor('LANDSAT-9', 'OLI'),
        LC8: PlatformSensor('LANDSAT-8', 'OLI'),
        LO8: PlatformSensor('LANDSAT-8', 'OLI'),
        L7: PlatformSensor('LANDSAT-7', 'ETM+'),
        L5: PlatformSensor('LANDSAT-5', 'TM'),
        L4: PlatformSensor('LANDSAT-4', 'TM'),
        S1A: PlatformSensor('SENTINEL-1', 'Sentinel-1A'),
        S1B: PlatformSensor('SENTINEL-1', 'Sentinel-1B'),
        S2A: PlatformSensor('SENTINEL-2', 'Sentinel-2A'),
        S2B: PlatformSensor('SENTINEL-2', 'Sentinel-2B')
    }

    NC_ENGINE = 'h5netcdf'

    @staticmethod
    def create_premet_file(ds: xr.Dataset, infile: str, url_tokens_1, url_tokens_2):
        """
        Create premet file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        ds: xarray.Dataset object that represents the granule.
        infile: Filename of the input ITS_LIVE granule
        url_tokens_1: Parsed out filename tokens that correspond to the first image of the pair
        url_tokens_2: Parsed out filename tokens that correspond to the second image of the pair
        """
        sensor1 = url_tokens_1[0]
        if sensor1 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file(): got unexpected mission+sensor {sensor1} for image#1 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        sensor2 = url_tokens_2[0]
        if sensor2 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file() got unexpected mission+sensor {sensor2} for image#2 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        # Get acquisition dates for both images
        begin_date=parse(ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1])
        end_date=parse(ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2])

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={infile}\n')
            fh.write('VersionID_local=002\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Extract time stamps
            fh.write(f'Begin_time={begin_date.strftime("%H:%M:%S")}.{begin_date.microsecond // 1000:03d}\n')
            fh.write(f'End_time={end_date.strftime("%H:%M:%S")}.{end_date.microsecond // 1000:03d}\n')

            # Append premet with sensor info
            for each_sensor in [sensor1, sensor2]:
                fh.write("Container=AssociatedPlatformInstrumentSensor\n")
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
        infile: Basename of the granule.
        """
        meta_filename = f'{infile}.spatial'

        xvals = ds.x.values
        yvals = ds.y.values
        pix_size_x = xvals[1] - xvals[0]
        pix_size_y = yvals[1] - yvals[0]

        epsgcode = ds[DataVars.MAPPING].attrs['spatial_epsg']

        # minval_x, pix_size_x, _, maxval_y, _, pix_size_y = [float(x) for x in ds['mapping'].attrs['GeoTransform'].split()]

        # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
        projection_cf_minx = xvals[0] - pix_size_x/2.0
        projection_cf_maxx = xvals[-1] + pix_size_x/2.0
        projection_cf_miny = yvals[-1] + pix_size_y/2.0  # pix_size_y is negative!
        projection_cf_maxy = yvals[0] - pix_size_y/2.0   # pix_size_y is negative!

        transformer = pyproj.Transformer.from_crs(f"EPSG:{epsgcode}", "EPSG:4326", always_xy=True)  # ensure lonlat output order

        # Convert coordinates to long/lat
        ll_lonlat = np.round(transformer.transform(projection_cf_minx, projection_cf_miny), decimals=2).tolist()
        lr_lonlat = np.round(transformer.transform(projection_cf_maxx, projection_cf_miny), decimals=2).tolist()
        ur_lonlat = np.round(transformer.transform(projection_cf_maxx, projection_cf_maxy), decimals=2).tolist()
        ul_lonlat = np.round(transformer.transform(projection_cf_minx, projection_cf_maxy), decimals=2).tolist()

        # Write to spatial file
        with open(meta_filename, 'w') as fh:
            for long, lat in [ul_lonlat, ur_lonlat, lr_lonlat, ll_lonlat]:
                fh.write(f"{long}\t{lat}\n")

        return meta_filename


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
        stop_index - Stop index into the list of granules to process. Default is -1 meaning to process to the end of the list.
        local_file_list - List of granules to process. This is a "hack" to run the script for specific set of granules such as sample set for NSIDC.
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

    def __call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
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
                dask.delayed(NSIDCFormat.process_granule)(target_bucket, target_dir, each, self.s3)
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
                logging.info("\n-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks

            gc.collect()

    @staticmethod
    def upload_to_s3(filename: str, target_dir: str, target_bucket: str, s3_client, remove_original_file: bool = True):
        """
        Upload file to the AWS S3 bucket.
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
        granule_filename = filename_tokens[-1]

        # Extract tokens from the filename
        url_tokens_1, url_tokens_2 = get_tokens_from_filename(granule_filename)

        msgs = [f'Processing {infilewithpath}']

        s3_client = boto3.client('s3')

        with s3.open(infilewithpath, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=NSIDCMeta.NC_ENGINE) as ds:
                # Create spacial and premet metadata files locally, then copy them to S3 bucket
                meta_file = NSIDCMeta.create_premet_file(ds, granule_filename, url_tokens_1, url_tokens_2)

                # ATTN: Place metadata files into the same directory as granules
                msgs.extend(NSIDCFormat.upload_to_s3(meta_file, granule_directory, target_bucket, s3_client))

                meta_file = NSIDCMeta.create_spatial_file(ds, granule_filename)
                # ATTN: This is for sample dataset to be tested by NSIDC only: places meta file in the same s3 directory as granule
                msgs.extend(NSIDCFormat.upload_to_s3(meta_file, granule_directory, target_bucket, s3_client))

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
                "Useful if need to continue previously interrupted process to prepare the granules, or need to split the load across "
                "multiple processes."
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
