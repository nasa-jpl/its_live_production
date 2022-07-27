"""
Script to fix various mapping attributes for HMA mosaics for ingest by NSIDC.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
import logging
import numpy as np
import os
import s3fs
import sys
import xarray as xr

# Local imports
from itscube_types import DataVars
from nsidc_vel_image_pairs import NSIDCMeta, NSIDCFormat, get_attr_value
from nsidc_mosaics import Encoding, NSIDCMosaicFormat


class Mapping:
    """
    Class to define mapping's attributes.
    """
    COORDINATE_AXIS_TYPES = 'CoordinateAxisTypes'
    COORDINATE_TRANSFORM_TYPE = 'CoordinateTransformType'
    LONGITUDE_OF_CENTRAL_MERIDIAN = 'longitude_of_central_meridian'
    LATITUDE_OF_PROJECTION_ORIGIN = 'latitude_of_projection_origin'
    FALSE_EASTING = 'false_easting'
    FALSE_NORTHING = 'false_northing'
    STANDARD_PARALLEL = 'standard_parallel'
    SPATIAL_PROJ4 = 'spatial_proj4'
    PROJ4TEXT = 'proj4text'
    SPATIAL_REF = 'spatial_ref'
    CRS_WKT = 'crs_wkt'
    LONGITUDE_OF_CENTRAL_MERIDIAN = 'longitude_of_central_meridian'
    FALSE_EASTING = 'false_easting'
    FALSE_NORTHING = 'false_northing'
    STANDARD_PARALLEL = 'standard_parallel'
    SEMI_MAJOR_AXIS = 'semi_major_axis'
    SEMI_MINOR_AXIS ='semi_minor_axis'
    SCALE_FACTOR_AT_CENTRAL_MERIDIAN = 'scale_factor_at_central_meridian'

class FixMosaicsMapping:
    """
    Class to fix various mapping attributes of V1 ITS_LIVE mosaics for ingest by NSIDC
    (as required by NSIDC after mosaics were submitted to them for ingest).
    This is additional fixes to the "mapping" data variable that were not in place
    when nsidc_mosaics.py script was run against V1 annual mosaics.
    There's a method per each type of the mosaics as fixes vary based on the region
    the mosaics correspond to.

    HMA mosaics:
    float mapping;
        Remove :CoordinateAxisTypes = "GeoX GeoY";
        Remove :CoordinateTransformType = "Projection";
        Rename spatial_proj4 to proj4text :proj4text = "+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
        :spatial_ref = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
        Add (this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools.) :crs_wkt = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
        Add required attribute; add attribute and set to 95.0 :longitude_of_central_meridian = 95.0;
        Add required attribute; add attribute and set to 30.0 :latitude_of_projection_origin = 30.0; // double
        Add required attribute; add attribute and set to 0.0 :false_easting = 0.0; //double
        Add required attribute; add attribute and set to 0.0 :false_northing = 0.0; //double
        Add required attribute; add attribute and set to 15.0 and 65.0 :standard_parallel = 15.0, 65.0; // double

    ANT mosaics:
    float mapping;
        Change value to -90 :latitude_of_projection_origin = -90.0; // double
        Optional attribute; but if to be included, set to 6378137.0 :semi_major_axis = 6378137.0; // double
        Optional attribute; remove attribute (I can't find the correct value) :semi_minor_axis = 6356.752; // double
        Add new: this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools. :crs_wkt = "PROJCS["WGS 84 / Antarctic Polar Stereographic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-71],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3031"]]";
        Edit attribute name from spatial_proj4 to proj4text :proj4text = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
        Required attribute; add attribute and set to -71.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin). :standard_parallel = -71.0; // double

    PAT mosaics:
        Remove :CoordinateAxisTypes = "GeoX GeoY";
        Remove :CoordinateTransformType = "Projection";
        Rename spatial_proj4 to proj4text :proj4text = "+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
        :spatial_ref = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
        Add (this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools.) :crs_wkt = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
        Required attribute; add attribute and set to 0.9996 :scale_factor_at_central_meridian = 0.9996;
        Required attribute; add attribute and set to -75.0 :longitude_of_central_meridian = -75.0
        Required attribute;  add attribute and set to 0.0 :latitude_of_projection_origin = 0.0;
        Optional; add attribute and set to 500000 :false_easting = 500000;
        Optional; add attribute and set to 10000000 :false_northing = 10000000;

    ALA, CAN, GRE, ICE, SRA mosaics:
        Change attribute value and set to 90.0 :latitude_of_projection_origin = 90.0; // double
        Optional attribute; but if to be included, set to 6378137.0 :semi_major_axis = 6378137.0; // double
        Remove :semi_minor_axis = 6356.752; // double
        Add new: this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools. :crs_wkt = "PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3413"]]";
        Edit attribute name from spatial_proj4 to proj4text :proj4text = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
        Required attribute; add attribute and set to 70.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin) :standard_parallel = 70.0; // double
    """
    REST_OF_REGIONS = 'ALA_CAN_GRE_ICE_SRA'

    def __init__(self, s3_bucket: str, s3_dir: str, region: str):
        """
        Initialize the object.

        Inputs:
        =======
        s3_bucket: AWS S3 bucket that stores mosaics files.
        s3_dir: Directory in AWS S3 bucket that stores mosaics files.
        """
        glob_pattern_str = f'{region}_*.nc'


        fix_method_dict = {
            'HMA': FixMosaicsMapping.process_HMA_file,
            'ANT': FixMosaicsMapping.process_ANT_file,
            'PAT': FixMosaicsMapping.process_PAT_file,
            FixMosaicsMapping.REST_OF_REGIONS: FixMosaicsMapping.process_rest_of_regions_file
        }

        self.fix_method = fix_method_dict[region]

        self.s3 = s3fs.S3FileSystem(anon=True)

        all_regions = [region]
        if region == FixMosaicsMapping.REST_OF_REGIONS:
            # Split the region into keywords, and glob files per each region
            all_regions = FixMosaicsMapping.REST_OF_REGIONS.split('_')

        self.infiles = []

        for each_region in all_regions:
            # Mosaics files that correspond to the glob pattern based on
            # the selected region
            glob_pattern_str = f'{each_region}_*.nc'
            glob_pattern = os.path.join(s3_bucket, s3_dir, glob_pattern_str)
            logging.info(f"Glob mosaics: {glob_pattern}")

            self.infiles.extend(self.s3.glob(f'{glob_pattern}'))

        # For debugging only - process one file
        # self.infiles = self.infiles[:1]
        logging.info(f"Got {len(self.infiles)} files")
        logging.info(f'Got mosaics files: {self.infiles}')

    def __call__(self, target_bucket: dir, target_dir: dir):
        """
        Fix ITS_LIVE mosaics mapping's various attributes

        Inputs:
        =======
        target_bucket: AWS S3 bucket to store fixed mosaics files.
        target_dir: Directory in AWS S3 bucket to store fixed mosaics files.
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
            results = FixMosaicsMapping.fix_file(target_bucket, target_dir, self.infiles[start], self.s3, self.fix_method)
            logging.info("\n-->".join(results))

            start += 1
            total_num_files -= 1

    @staticmethod
    def fix_file(target_bucket: str, target_dir: str, infilewithpath: str, s3, fix_method):
        """
        Fix "mapping" attributes for ingest by NSIDC.
        """
        filename_tokens = infilewithpath.split(os.path.sep)
        filename = filename_tokens[-1]

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Fixing mapping of {infilewithpath}']

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
            msgs.extend(
                fix_method(
                    ds,
                    filename,
                    Encoding.MOSAICS,
                    NSIDCMosaicFormat.CHUNK_SIZE
                )
            )

        # Remove local copy of the file
        msgs.append(f"Removing original local {local_file}")
        os.unlink(local_file)

        # Copy new granule to S3 bucket
        msgs.extend(
            NSIDCFormat.upload_to_s3(filename, target_dir, target_bucket, s3_client, remove_original_file=True)
        )

        return msgs

    @staticmethod
    def process_HMA_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        chunk_size: int
    ):
        """
        Fix "mapping" attributes of "ds" Dataset:
            Remove :CoordinateAxisTypes = "GeoX GeoY";
            Remove :CoordinateTransformType = "Projection";
            Rename spatial_proj4 to proj4text :proj4text = "+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
            :spatial_ref = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
            Add (this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools.) :crs_wkt = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
            Add required attribute; add attribute and set to 95.0 :longitude_of_central_meridian = 95.0;
            Add required attribute; add attribute and set to 30.0 :latitude_of_projection_origin = 30.0; // double
            Add required attribute; add attribute and set to 0.0 :false_easting = 0.0; //double
            Add required attribute; add attribute and set to 0.0 :false_northing = 0.0; //double
            Add required attribute; add attribute and set to 15.0 and 65.0 :standard_parallel = 15.0, 65.0; // double

        Inputs:
        =======
        ds - xr.Dataset object
        new_filename - Local filename to store fixed dataset to
        encoding_params - Encoding settings for the dataset to store to the file
        chunk_size - Chunk size to use when storing dataset to the file.
        """
        msgs = []

        # Extra fixes are required by NSIDC to the mapping data variable
        mapping = ds[DataVars.MAPPING]

        del mapping.attrs[Mapping.COORDINATE_AXIS_TYPES]
        del mapping.attrs[Mapping.COORDINATE_TRANSFORM_TYPE]

        # Edit attribute name from spatial_proj4 to proj4text
        mapping.attrs[Mapping.PROJ4TEXT] = mapping.attrs[Mapping.SPATIAL_PROJ4]
        del mapping.attrs[Mapping.SPATIAL_PROJ4]

        # Add attributes
        mapping.attrs[Mapping.CRS_WKT] = mapping.attrs[Mapping.SPATIAL_REF]
        mapping.attrs[Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN] = 95.0
        mapping.attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = 30.0
        mapping.attrs[Mapping.FALSE_EASTING] = 0.0
        mapping.attrs[Mapping.FALSE_NORTHING] = 0.0
        mapping.attrs[Mapping.STANDARD_PARALLEL] = np.array([15.0, 65.0], dtype=np.float32)

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        logging.info(f'Saving fixed file to local {new_filename}')
        ds.to_netcdf(new_filename, engine='h5netcdf', encoding = encoding_params)

        return msgs

    @staticmethod
    def process_PAT_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        chunk_size: int
    ):
        """
        Fix "mapping" attributes of "ds" Dataset:
            Remove :CoordinateAxisTypes = "GeoX GeoY";
            Remove :CoordinateTransformType = "Projection";
            Rename spatial_proj4 to proj4text :proj4text = "+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
            :spatial_ref = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
            Add (this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools.) :crs_wkt = "PROJCS["Asia_North_Lambert_Conformal_Conic",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",95],PARAMETER["Standard_Parallel_1",15],PARAMETER["Standard_Parallel_2",65],PARAMETER["Latitude_Of_Origin",30],UNIT["Meter",1],AUTHORITY["ESRI","102027"]]";
            Required attribute; add attribute and set to 0.9996 :scale_factor_at_central_meridian = 0.9996;
            Required attribute; add attribute and set to -75.0 :longitude_of_central_meridian = -75.0
            Required attribute;  add attribute and set to 0.0 :latitude_of_projection_origin = 0.0;
            Optional; add attribute and set to 500000 :false_easting = 500000;
            Optional; add attribute and set to 10000000 :false_northing = 10000000;

        Inputs:
        =======
        ds - xr.Dataset object
        new_filename - Local filename to store fixed dataset to
        encoding_params - Encoding settings for the dataset to store to the file
        chunk_size - Chunk size to use when storing dataset to the file.
        """
        msgs = []

        # Extra fixes are required by NSIDC to the mapping data variable
        mapping = ds[DataVars.MAPPING]

        del mapping.attrs[Mapping.COORDINATE_AXIS_TYPES]
        del mapping.attrs[Mapping.COORDINATE_TRANSFORM_TYPE]

        # Edit attribute name from spatial_proj4 to proj4text
        mapping.attrs[Mapping.PROJ4TEXT] = mapping.attrs[Mapping.SPATIAL_PROJ4]
        del mapping.attrs[Mapping.SPATIAL_PROJ4]

        # Add attributes
        mapping.attrs[Mapping.CRS_WKT] = mapping.attrs[Mapping.SPATIAL_REF]
        mapping.attrs[Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN] = 0.9996
        mapping.attrs[Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN] = -75.0
        mapping.attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = 0.0
        mapping.attrs[Mapping.FALSE_EASTING] = 500000.0
        mapping.attrs[Mapping.FALSE_NORTHING] = 10000000.0

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        logging.info(f'Saving fixed file to local {new_filename}')
        ds.to_netcdf(new_filename, engine='h5netcdf', encoding = encoding_params)

        return msgs

    @staticmethod
    def process_ANT_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        chunk_size: int
    ):
        """
        Fix "mapping" attributes of "ds" Dataset:
            Change value to -90 :latitude_of_projection_origin = -90.0; // double
            Optional attribute; but if to be included, set to 6378137.0 :semi_major_axis = 6378137.0; // double
            Optional attribute; remove attribute (I can't find the correct value) :semi_minor_axis = 6356.752; // double
            Add new: this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools. :crs_wkt = "PROJCS["WGS 84 / Antarctic Polar Stereographic",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-71],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3031"]]";
            Edit attribute name from spatial_proj4 to proj4text :proj4text = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
            Required attribute; add attribute and set to -71.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin). :standard_parallel = -71.0; // double

        Inputs:
        =======
        ds - xr.Dataset object
        new_filename - Local filename to store fixed dataset to
        encoding_params - Encoding settings for the dataset to store to the file
        chunk_size - Chunk size to use when storing dataset to the file.
        """
        msgs = []

        # Extra fixes are required by NSIDC to the mapping data variable
        mapping = ds[DataVars.MAPPING]

        # Edit attribute name from spatial_proj4 to proj4text
        mapping.attrs[Mapping.PROJ4TEXT] = mapping.attrs[Mapping.SPATIAL_PROJ4]
        del mapping.attrs[Mapping.SPATIAL_PROJ4]

        # Remove attributes
        del mapping.attrs[Mapping.SEMI_MINOR_AXIS]

        # Add attributes
        mapping.attrs[Mapping.CRS_WKT] = mapping.attrs[Mapping.SPATIAL_REF]
        mapping.attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = -90.0

        # Optional attribute; but if to be included, set to 6378137.0
        mapping.attrs[Mapping.SEMI_MAJOR_AXIS] = 6378137.0
        mapping.attrs[Mapping.STANDARD_PARALLEL] = -71.0

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        logging.info(f'Saving fixed file to local {new_filename}')
        ds.to_netcdf(new_filename, engine='h5netcdf', encoding = encoding_params)

        return msgs

    @staticmethod
    def process_rest_of_regions_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        chunk_size: int
    ):
        """
        Fix "mapping" attributes of "ds" Dataset:
            Change attribute value and set to 90.0 :latitude_of_projection_origin = 90.0; // double
            Optional attribute; but if to be included, set to 6378137.0 :semi_major_axis = 6378137.0; // double
            Remove :semi_minor_axis = 6356.752; // double
            Add new: this is a suggestion only, so optional! adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools. :crs_wkt = "PROJCS["WGS 84 / NSIDC Sea Ice Polar Stereographic North",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3413"]]";
            Edit attribute name from spatial_proj4 to proj4text :proj4text = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
            Required attribute; add attribute and set to 70.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin) :standard_parallel = 70.0; // double

        Inputs:
        =======
        ds - xr.Dataset object
        new_filename - Local filename to store fixed dataset to
        encoding_params - Encoding settings for the dataset to store to the file
        chunk_size - Chunk size to use when storing dataset to the file.
        """
        msgs = []

        # Extra fixes are required by NSIDC to the mapping data variable
        mapping = ds[DataVars.MAPPING]

        # Edit attribute name from spatial_proj4 to proj4text
        mapping.attrs[Mapping.PROJ4TEXT] = mapping.attrs[Mapping.SPATIAL_PROJ4]
        del mapping.attrs[Mapping.SPATIAL_PROJ4]

        # Remove attributes
        del mapping.attrs[Mapping.SEMI_MINOR_AXIS]

        # Add attributes
        mapping.attrs[Mapping.CRS_WKT] = mapping.attrs[Mapping.SPATIAL_REF]
        mapping.attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = 90.0

        # Optional attribute; but if to be included, set to 6378137.0
        mapping.attrs[Mapping.SEMI_MAJOR_AXIS] = 6378137.0
        mapping.attrs[Mapping.STANDARD_PARALLEL] = 70.0

        # Convert dataset to Dask dataset not to run out of memory while writing to the file
        ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        logging.info(f'Saving fixed file to local {new_filename}')
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
        '-source_dir',
        type=str,
        default='NSIDC/v01/mosaics_prev/annual',
        help='AWS S3 directory that stores mosaics to fix [%(default)s]'
    )

    parser.add_argument(
        '-region',
        type=str,
        required=True,
        choices=['HMA', 'ANT', 'PAT', FixMosaicsMapping.REST_OF_REGIONS],
        help='Region code for which to fix mosaics files'
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

    nsidc_format = FixMosaicsMapping(args.bucket, args.source_dir, args.region)

    nsidc_format(args.bucket, args.target_dir)

    logging.info('Done.')
