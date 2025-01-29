"""
Script to prepare V2 ITS_LIVE mosaics to be ingested by NSIDC: see nsidc_vel_image_pairs_v2.py
for the details.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
from datetime import datetime
import logging
import numpy as np
import os
import pyproj
import s3fs
import xarray as xr

# Local imports
from itscube_types import DataVars, Output, CompDataVars, BinaryFlag, ShapeFile
from itslive_composite import MissionSensor
from nsidc_vel_image_pairs import NSIDCFormat
from nsidc_vel_image_pairs_v2 import NSIDCMeta

# NetCDF attributes and data variables names
SCALE_FACTOR_AT_PROJECTION_ORIGIN = 'scale_factor_at_projection_origin'


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
            fh.write('VersionID_local=002\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Hard-code the values for annual and static mosaics
            fh.write("Begin_time=00:00:01.000\n")
            fh.write("End_time=23:59:59.000\n")

            already_written_platform_sensor = []
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
                # Some of the platform/sensor short names are the same across mission/sensor combos (LC9 vs. LO9),
                # so keep only unique combos in the metadata file
                each_sensor_str = f'{NSIDCMeta.ShortName[each_sensor].platform}_{NSIDCMeta.ShortName[each_sensor].sensor}'

                if each_sensor_str not in already_written_platform_sensor:
                    already_written_platform_sensor.append(each_sensor_str)

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
        projection_cf_miny = yvals[-1] + pix_size_y/2.0  # pix_size_y is negative!
        projection_cf_maxy = yvals[0] - pix_size_y/2.0   # pix_size_y is negative!

        epsgcode = ds[DataVars.MAPPING].attrs['spatial_epsg']
        epsgcode_str = f'EPSG:{epsgcode}'

        if epsgcode == NSIDCFormat.ESRI_CODE:
            epsgcode_str = f'ESRI:{epsgcode}'

        transformer = pyproj.Transformer.from_crs(epsgcode_str, "EPSG:4326", always_xy=True)  # ensure lonlat output order

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


class NSIDCMosaicFormat:
    """
    Class to prepare V2 ITS_LIVE mosaics for ingest by NSIDC.
    It generates metadata files required by NSIDC ingest (premet and spacial metadata files
    which are generated per each data product being ingested).
    """
    # Pattern to collect input files in AWS S3 bucket
    GLOB_PATTERN = '*.nc'

    # Re-chunk input mosaic to speed up read of the data
    CHUNK_SIZE = 1000

    LOCAL_FIX_DIR = 'fixed_mosaics'

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

        if not os.path.exists(NSIDCMosaicFormat.LOCAL_FIX_DIR):
            os.mkdir(NSIDCMosaicFormat.LOCAL_FIX_DIR)

    def __call__(self, target_dir):
        """
        Create NSIDC meta files (spatial and premet) for ITS_LIVE v2 mosaics.

        Inputs:
        target_dir: Directory in AWS S3 bucket to store fixed mosaics to.
        """
        total_num_files = len(self.infiles)

        if total_num_files <= 0:
            logging.info("Nothing to process, exiting.")
            return

        # Current start index into list of files to process
        start = 0

        while start < total_num_files:
            logging.info(f"Starting {self.infiles[start]} {start} out of {total_num_files} total files")
            results = NSIDCMosaicFormat.process_file(
                self.s3_bucket,
                self.infiles[start],
                self.s3,
                target_dir
            )
            logging.info("\n-->".join(results))

            start += 1

    @staticmethod
    def process_file(target_bucket: str, infilewithpath: str, s3: s3fs.S3FileSystem, target_dir: str):
        """
        Fix granule format and create corresponding metadata files as required by NSIDC.
        Place fixed granule and metadata files in the target S3 bucket.
        """
        filename_tokens = infilewithpath.split(os.path.sep)
        # directory = os.path.sep.join(filename_tokens[1:-1])

        filename = filename_tokens[-1]

        # Extract tokens from the filename
        tokens = filename.split('_')
        year = int(tokens[-2])

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Processing {infilewithpath}']

        s3_client = boto3.client('s3')

        file_path = os.path.sep.join(filename_tokens[1:])
        local_file = filename + '.local'

        # Download file locally - takes too long to read the whole mosaic file
        # from S3 in order for it to write fixed dataset locally
        s3_client.download_file(target_bucket, file_path, local_file)

        with xr.open_dataset(local_file, engine=NSIDCMeta.NC_ENGINE) as ds:
            # Fix metadata for the dataset
            fixed_ds, fixed_ds_msgs = NSIDCMosaicFormat.process_nc_file(ds)
            msgs.extend(fixed_ds_msgs)

            # Write fixed granule to local file
            # Convert dataset to Dask dataset not to run out of memory while writing to the file
            fixed_ds = fixed_ds.chunk(chunks={
                'x': NSIDCMosaicFormat.CHUNK_SIZE,
                'y': NSIDCMosaicFormat.CHUNK_SIZE
            })

            # Write fixed granule to local file
            new_filename = os.path.join(NSIDCMosaicFormat.LOCAL_FIX_DIR, filename)
            fixed_ds.to_netcdf(new_filename, engine='h5netcdf')

            msgs.extend(NSIDCFormat.upload_to_s3(new_filename, target_dir, target_bucket, s3_client))

            # Create spacial and premet metadata files, and copy them to S3 bucket
            meta_file = NSIDCMosaicsMeta.create_premet_file(filename, year)

            msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

            meta_file = NSIDCMosaicsMeta.create_spatial_file(ds, filename)
            msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        return msgs

    @staticmethod
    def process_nc_file(
        ds
    ):
        """
        See github issue #38 for details on the changes required for the file
        to be ingested by NSIDC.

        Inputs:
        ds: xarray.Dataset object that represents the NetCDF file content (granule
            or mosaic).
        """
        msgs = []
        _cf_value = 'CF-1.8'

        # Add "conversion" attribute to the dataset
        ds.attrs[Output.CONVENTIONS] = _cf_value

        # Change value of the "title" attribute
        ds.attrs[Output.TITLE] = 'MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface ' \
            'Velocities, Version 2'

        # Add "citation" attribute to the dataset
        ds.attrs[Output.CITATION] = 'Gardner, A. S., Fahnestock, M., Greene, C. A., ' \
            'Kennedy, J. H., Liukis, M., & Scambos, T. 2024. ' \
            'MEaSUREs ITS_LIVE Regional Glacier and Ice Sheet Surface Velocities, ' \
            'Version 2 [Indicate subset used]. Boulder, Colorado USA. ' \
            'NASA National Snow and Ice Data Center Distributed Active Archive Center. ' \
            'https//:doi.org/10.5067/JQ6337239C96. [Date accessed]."'

        # Add "publisher_name" attribute to the dataset
        ds.attrs[Output.PUBLISHER_NAME] = 'NASA National Snow and Ice Data Center Distributed Active Archive Center'

        # These are not bounding polygon - centers of polygons for multi-EPSG mosaics
        # Replace latitude and longitude attributes with geospatial_bounds and
        # geospatial_bounds_crs.
        # Example:
        # :latitude = "[59.42, 60.11, 62.72]"; and
        # :longitude = "[-146.05, -159.66, -130.11]";
        # with
        # :geospatial_bounds = "POLYGON((lat1 lon1, lat2 lon2,...))"; and
        # :geospatial_bounds_crs= "EPSG:4326";
        # lat_values = json.loads(ds.attrs[Output.LATITUDE])
        # long_values = json.loads(ds.attrs[Output.LONGITUDE])

        # # Create POLYGON string for geospatial_bounds attribute:
        # # https://wiki.esipfed.org/Attribute_Convention_for_Data_Discovery_1-3#Global_Attributes:~:text=110.29%2C%2040.26%20%2D111.29))%27.-,geospatial_bounds_crs,-The%20coordinate%20reference

        # Remove latitude and longitude attributes - not used by any tools
        del ds.attrs[Output.LATITUDE]
        del ds.attrs[Output.LONGITUDE]

        # Change datatype for mapping.scale_factor_at_projection_origin to float
        if SCALE_FACTOR_AT_PROJECTION_ORIGIN in ds.mapping.attrs:
            ds.mapping.attrs[SCALE_FACTOR_AT_PROJECTION_ORIGIN] = float(ds.mapping.attrs[SCALE_FACTOR_AT_PROJECTION_ORIGIN])

        # Change 'floatingice' attributes
        ds[ShapeFile.FLOATINGICE].attrs[BinaryFlag.VALUES_ATTR] = BinaryFlag.VALUES
        ds[ShapeFile.LANDICE].attrs[BinaryFlag.VALUES_ATTR] = BinaryFlag.VALUES

        ds[ShapeFile.FLOATINGICE].attrs[Output.REFERENCES] = ds[ShapeFile.FLOATINGICE].attrs[Output.URL]
        del ds[ShapeFile.FLOATINGICE].attrs[Output.URL]

        ds[ShapeFile.LANDICE].attrs[Output.REFERENCES] = ds[ShapeFile.LANDICE].attrs[Output.URL]
        del ds[ShapeFile.LANDICE].attrs[Output.URL]

        # Changes for the static or annual mosaics
        if CompDataVars.SENSORS in ds:
            # This is static mosiac

            # Change "sensor" dimension
            # * Change dtype to ubyte
            # * Add new attribute: flag_values = 1B, 2B, 3B, 4B, 5B; // ubyte
            # * Add new attribute: flag_meanings = "L4_L5 L7 L8_L9 S1A_S1B S2A_S2B";`
            sensor_description = []
            sensor_values = []
            for each in ds.sensor.values:
                sensor_values.append(MissionSensor.SENSOR_DIMENSION_MAPPING[each])
                sensor_description.append(each)

            sensor_description = ' '.join(sensor_description)
            msgs.append(f'Got sensors: {sensor_values=} corresponding to {sensor_description=} ')

            sensor_array = np.array(sensor_values, dtype=np.ubyte)

            # This won't overwrite input "ds" object
            ds = ds.assign_coords(sensor=sensor_array)

            # Set attributes for the sensor dimension
            ds[CompDataVars.SENSORS].attrs[BinaryFlag.VALUES_ATTR] = sensor_array
            ds[CompDataVars.SENSORS].attrs[BinaryFlag.MEANINGS_ATTR] = sensor_description

            # Change 'count' attributes
            ds[Output.COUNT].attrs[DataVars.COMMENT] = ds[Output.COUNT].attrs[DataVars.NOTE]
            del ds[Output.COUNT].attrs[DataVars.NOTE]

            ds[Output.COUNT].attrs[DataVars.STD_NAME] = 'number_of_observations'

            # Change 'dv*_dt' attributes
            ds[CompDataVars.SLOPE_V].attrs[DataVars.STD_NAME] = 'trend [2014-2022] in v'
            ds[CompDataVars.SLOPE_VX].attrs[DataVars.STD_NAME] = 'trend [2014-2022] in vx'
            ds[CompDataVars.SLOPE_VY].attrs[DataVars.STD_NAME] = 'trend [2014-2022] in vy'

            # Change 'v' attributes
            ds[DataVars.V].attrs[DataVars.STD_NAME] = 'climatological velocity [2014-2022]'
            ds[DataVars.V].attrs[DataVars.DESCRIPTION_ATTR] = 'determined by taking the hypotenuse of vx and vy. ' \
                'The climatology uses a time-intercept of January 1, 2018.'

            # Change 'v_amp' attributes
            ds[CompDataVars.V_AMP].attrs[DataVars.STD_NAME] = 'climatological [2014-2022] mean seasonal amplitude'

            # Change 'v_amp_error' attributes
            ds[CompDataVars.V_AMP_ERROR].attrs[DataVars.STD_NAME] = 'v_amp error'

            # Change 'v_error' attributes
            ds[CompDataVars.V_ERROR].attrs[DataVars.STD_NAME] = 'v error'

        else:
            # This is annual mosaic

            # Change 'v' attributes
            ds[DataVars.V].attrs[DataVars.STD_NAME] = 'mean annual velocity'

            # Change 'v_error' attributes
            ds[CompDataVars.V_ERROR].attrs[DataVars.STD_NAME] = 'v error'
            ds[CompDataVars.VX_ERROR].attrs[DataVars.STD_NAME] = 'vx error'
            ds[CompDataVars.VY_ERROR].attrs[DataVars.STD_NAME] = 'vy error'

        return ds, msgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        Prepare ITS_LIVE V2 mosaics for ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE data products to [%(default)s]'
    )

    parser.add_argument(
        '-source_dir',
        type=str,
        default='NSIDC/velocity_mosaic_sample/v2/static',
        help='AWS S3 directory that stores input mosaics [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/velocity_mosaic_sample/v2/static/fixed',
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

    nsidc_format = NSIDCMosaicFormat(args.bucket, args.source_dir)
    nsidc_format(args.target_dir)

    logging.info('Done.')
