"""
Script to prepare V1 ITS_LIVE granules to be ingested by NSIDC:

* count: fix units to ‘count’

* vx_err and vy_err: remove 'missing_value' attribute from any data variable that has it

* dt: change 'long_name' to 'error weighted average time separation between image-pairs'

* Add: Conventions = "CF-1.8" to PAT_G0120_0000.nc like data products
* Change: Conventions = "CF-1.8" to velocity image pair products

* Fix "m/y" units to "meter/year" for all variables that the unit is applicable for
* Replace 'binary' units for 'interp_mask' by:
    flag_values = 0UB, 1UB; // ubyte
    flag_meanings = 'measured, interpolated'

* UTM_Projection, Polar_Stereographic: replace by 'mapping' variable
* UTM_Projection: change grid_mapping_name: "universal_transverse_mercator" to
  "transverse_mercator"

* Changing standard_name for vx, vy, and v to:
  “land_ice_surface_x_velocity”, “land_ice_surface_y_velocity” and “land_ice_surface_velocity”

* Add standard_name = 'image_pair_information' to img_pair_info

August 5, 2022: Additional changes are requested by NSIDC to "mapping" data variable based on EPSG code:

* Remove if present:
    :CoordinateAxisTypes = "GeoX GeoY";
    :CoordinateTransformType = "Projection";

* Change attribute name from spatial_proj4 to proj4text

* Adding crs_wkt (redundant with spatial_ref) expands interoperability with geolocation tools


Rename original file:
LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_X_LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX_G0240V01_PXYZ.nc
to
LXSSLLLLPPPRRRYYYYMMDDCCTXX_LXSSLLLLPPPRRRYYYYMMDDCCTX_EEEEE_G0240V01_XYZ.nc
where EEEEE is EPSG code from subdirectory name of original file.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""
import argparse
import boto3
from botocore.exceptions import ClientError
import collections
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import gc
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
from nsidc_types import Mapping


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

    # Get tokens for the first image name
    url_tokens_1 = url_files[0].split('_')

    if len(url_tokens_1) < 9:
        # Optical format granule
        # Get acquisition/processing dates and path&row for both images
        first_date_1 = datetime.strptime(url_tokens_1[3], DATE_FORMAT)
        second_date_1 = datetime.strptime(url_tokens_1[4], DATE_FORMAT)
        key_1 = url_tokens_1[2]

        # Extract info from second part of the granule's filename
        url_tokens_2 = url_files[1].split('_')
        first_date_2 = datetime.strptime(url_tokens_2[3], DATE_FORMAT)
        second_date_2 = datetime.strptime(url_tokens_2[4], DATE_FORMAT)
        key_2 = url_tokens_2[2]

    else:
        logging.info(f'Unexpected filename format: {filename}')

    # first_date_1, second_date_1, key_1, first_date_2, second_date_2, key_2
    return (url_tokens_1, url_tokens_2)

def get_attr_value(h5_attr: str):
    """
    Extract value of the hd5 data variable attribute.
    """
    value = None
    if isinstance(h5_attr, str):
        value = h5_attr

    elif isinstance(h5_attr, bytes):
        value = h5_attr.decode('utf-8')  # h5py returns byte values, turn into byte characters

    elif h5_attr.shape == ():
        value = h5_attr

    else:
        value = h5_attr[0] # h5py returns lists of numbers - all 1 element lists here, so dereference to number

    return value

class Encoding:
    """
    Encoding settings for writing ITS_LIVE granule to the file
    """
    IMG_PAIR = {
        'interp_mask':      {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_height': {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'chip_size_width':  {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
        'v':                {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vx':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'vy':               {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
        'img_pair_info':    {'_FillValue': None, 'dtype': np.float32},
        'mapping':          {'_FillValue': None, 'dtype': np.float32},
        'x':                {'_FillValue': None},
        'y':                {'_FillValue': None}
    }

PS = collections.namedtuple("PM", ['platform', 'sensor'])

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
    L8 = 'LC08'
    L7 = 'LE07'
    L5 = 'LT05'
    L4 = 'LT04'

    ShortName = {
        L8: PS('LANDSAT-8', 'OLI'),
        L7: PS('LANDSAT-7', 'ETM+'),
        L5: PS('LANDSAT-5', 'TM'),
        L4: PS('LANDSAT-4', 'TM')
    }

    @staticmethod
    def create_premet_file(infile: str, url_tokens_1, url_tokens_2):
        """
        Create premet file that corresponds to the input image pair velocity granule.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE granule
        url_tokens_1: Parsed out filename tokens that correspond to the first image of the pair
        url_tokens_2: Parsed out filename tokens that correspond to the second image of the pair
        """
        # Get acquisition dates for both images
        begin_date = datetime.strptime(url_tokens_1[3], DATE_FORMAT)
        end_date = datetime.strptime(url_tokens_2[3], DATE_FORMAT)

        sensor1 = url_tokens_1[0]
        if sensor1 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file(): got unexpected mission+sensor {sensor1} for image#1 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        sensor2 = url_tokens_2[0]
        if sensor2 not in NSIDCMeta.ShortName:
            raise RuntimeError(f'create_premet_file() got unexpected mission+sensor {sensor2} for image#2 of {infile}: one of {list(NSIDCMeta.ShortName.keys())} is supported.')

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={infile}\n')
            fh.write(f'VersionID_local=001\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Hard-code the values as we don't have timestamps captured within img_pair_info
            fh.write("Begin_time=00:00:01.000\n")
            fh.write("End_time=23:59:59.000\n")

            # Append premet with sensor info
            for each_sensor in [sensor1, sensor2]:
                fh.write(f"Container=AssociatedPlatformInstrumentSensor\n")
                fh.write(f"AssociatedPlatformShortName={NSIDCMeta.ShortName[each_sensor].platform}\n")
                fh.write(f"AssociatedInstrumentShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")
                fh.write(f"AssociatedSensorShortName={NSIDCMeta.ShortName[each_sensor].sensor}\n")

        return meta_filename

    @staticmethod
    def create_spatial_file(infile: str, epsgcode: int):
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

            transformer = pyproj.Transformer.from_crs(f"EPSG:{epsgcode}", "EPSG:4326", always_xy=True) # ensure lonlat output order

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

# Map of required attributes per EPSG code
# :spatial_epsg = 32622.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -51.0 // double	required attribute; add attribute and set to -51.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0
#
# :spatial_epsg = 32623.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -45.0 // double	required attribute; add attribute and set to -45.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000
# :false_northing = 0; // double	optional; add attribute and set to 0
#
# :spatial_epsg = 32624.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -39.0 // double	required attribute; add attribute and set to -39.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0
#
# :spatial_epsg = 32625.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -33.0 // double	required attribute; add attribute and set to -33.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0
#
# :spatial_epsg = 32626.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -27.0 // double	required attribute; add attribute and set to -27.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32627.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -21.0 // double	required attribute; add attribute and set to -21.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32628.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -15.0 // double	required attribute; add attribute and set to -15.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32640.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 57.0 // double	required attribute; add attribute and set to 57.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32641.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 63.0 // double	required attribute; add attribute and set to 63.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32605.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -153.0 // double	required attribute; add attribute and set to -153.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32606.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -147.0 // double	required attribute; add attribute and set to -147.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32607.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -141.0 // double	required attribute; add attribute and set to -141.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32608.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -135.0 // double	required attribute; add attribute and set to -135.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32609.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -129.0 // double	required attribute; add attribute and set to -129.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32610.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -123.0 // double	required attribute; add attribute and set to -123.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32611.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -117.0 // double	required attribute; add attribute and set to -117.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32615.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -93.0 // double	required attribute; add attribute and set to -93.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32616.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -87.0 // double	required attribute; add attribute and set to -87.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32617.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -81.0 // double	required attribute; add attribute and set to -81.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32618.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -75.0 // double	required attribute; add attribute and set to -75.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32619.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -69.0 // double	required attribute; add attribute and set to -69.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32620.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -63.0 // double	required attribute; add attribute and set to -63.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32621.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -57.0 // double	required attribute; add attribute and set to -57.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32632.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 9.0 // double	required attribute; add attribute and set to 9.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32633.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 15.0 // double	required attribute; add attribute and set to 15.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32634.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 21.0 // double	required attribute; add attribute and set to 21.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32635.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 27.0 // double	required attribute; add attribute and set to 27.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32639.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 51.0 // double	required attribute; add attribute and set to 51.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32642.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 69.0 // double	required attribute; add attribute and set to 69.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32643.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 75.0 // double	required attribute; add attribute and set to 75.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32644.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 81.0 // double	required attribute; add attribute and set to 81.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32645.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 87.0 // double	required attribute; add attribute and set to 87.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32646.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 93.0 // double	required attribute; add attribute and set to 93.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32647.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 99.0 // double	required attribute; add attribute and set to 99.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32648.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 105.0 // double	required attribute; add attribute and set to 105.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 0; // double	optional; add attribute and set to 0.0

# :spatial_epsg = 32718.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -75.0 // double	required attribute; add attribute and set to -75.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 10000000; // double	optional; add attribute and set to 10000000.0

# :spatial_epsg = 32719.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -69.0 // double	required attribute; add attribute and set to -69.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 10000000; // double	optional; add attribute and set to 10000000.0

# :spatial_epsg = 32720.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -63.0 // double	required attribute; add attribute and set to -63.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 10000000; // double	optional; add attribute and set to 10000000.0

# :spatial_epsg = 32721.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = -57.0 // double	required attribute; add attribute and set to -57.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 10000000; // double	optional; add attribute and set to 10000000.0

# :spatial_epsg = 32759.0; // double
# :scale_factor_at_central_meridian = 0.9996; // double	required attribute; add attribute and set to 0.9996
# :longitude_of_central_meridian = 171.0 // double	required attribute; add attribute and set to 171.0
# :latitude_of_projection_origin = 0.0; // double	required attribute;  add attribute and set to 0.0
# :false_easting = 500000.0; // double	optional; add attribute and set to 500000.0
# :false_northing = 10000000; // double	optional; add attribute and set to 10000000.0

required_attrs = {
    32622: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -51.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32623: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -45.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32624: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -39.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32625: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -33.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32626: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -27.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32627: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -21.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32628: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -15.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32640: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 57.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32641: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 63.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32605: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -153.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32606: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -147.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32607: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -141.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32608: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -135.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32609: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -129.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32610: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -123.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32611: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -117.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32615: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -93.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32616: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -87.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32617: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -81.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32618: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -75.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32619: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -69.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32620: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -63.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32621: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -57.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32632: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 9.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32633: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 15.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32634: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 21.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32635: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 27.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32639: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 51.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32642: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 69.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32643: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 75.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32644: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 81.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32645: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 87.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32646: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 93.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32647: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 99.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32648: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 105.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0
    },
    32718: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -75.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0
    },
    32719: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -69.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0
    },
    32720: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -63.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0
    },
    32721: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -57.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0
    },
    32759: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 171.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0
    }
}

def fix_mapping_attrs(mapping, epsgcode: int):
    """
    Additional fixes to dataset's mapping variable attributes.

    mapping: Mapping data variable of the dataset.
    """
    attrs = mapping.attrs
    if Mapping.COORDINATE_TRANSFORM_TYPE in attrs:
        del attrs[Mapping.COORDINATE_TRANSFORM_TYPE]

    if Mapping.COORDINATE_AXIS_TYPES in attrs:
        del attrs[Mapping.COORDINATE_AXIS_TYPES]

    # Edit attribute name from spatial_proj4 to proj4text
    if Mapping.SPATIAL_PROJ4 in attrs:
        attrs[Mapping.PROJ4TEXT] = attrs[Mapping.SPATIAL_PROJ4]
        del attrs[Mapping.SPATIAL_PROJ4]

    # Optional attribute; but if to be included, set to 6378137.0
    attrs[Mapping.SEMI_MAJOR_AXIS] = 6378137.0

    # Optional attribute; remove attribute (Danica Linda Cantarero@NSIDC: I can't find the correct value)
    if Mapping.SEMI_MINOR_AXIS in attrs:
        del attrs[Mapping.SEMI_MINOR_AXIS]

    # Adding crs_wkt (redundant with spatial_ref) expands interoperability
    # with geolocation tools:
    if (Mapping.CRS_WKT not in attrs) and (Mapping.SPATIAL_REF in attrs):
        attrs[Mapping.CRS_WKT] = attrs[Mapping.SPATIAL_REF]

    # Apply corrections based on the EPSG code
    if epsgcode not in required_attrs and epsgcode != 3031:
        return f'No extra mapping attributes need to be set for {epsgcode}'

    if epsgcode in required_attrs:
        for each_attr, each_value in required_attrs[epsgcode].items():
            attrs[each_attr] = each_value

        attrs[Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN] = 0.9996
        attrs[Mapping.LATITUDE_OF_PROJECTION_ORIGIN] = 0.0

    if epsgcode == 3031:
    # :latitude_of_origin = -71.0; // double	delete attribute
    # :semi_major_axis = 6378137.0; // double	optional attribute; but if to be included, set to 6378137.0
    # :semi_minor_axis = 6356.752; // double	optional attribute; remove attribute (I can't find the correct value)
    # :standard_parallel = -71.0; // double	required attribute; add attribute and set to -71.0 (standard_parallel is aka latitude_of_origin, but is not the same as latitude_of_projection_origin).
        if Mapping.LATITUDE_OF_ORIGIN in attrs:
            del attrs[Mapping.LATITUDE_OF_ORIGIN]

        attrs[Mapping.STANDARD_PARALLEL] = -71.0

    return f'Set extra required mapping attributes for {epsgcode}'

class NSIDCFormat:
    """
    Class to prepare V1 ITS_LIVE data for ingest by NSIDC:
    1. Make V1 ITS_LIVE data CF-1.8 convention compliant.
    2. Generate metadata files required by NSIDC ingest (premet and spacial metadata files
       which are generated per each data product being ingested).
    """
    GRANULES_FILE = 'used_granules_landsat.json'

    # Flag to enable dry run: don't process any granules, just report what would be processed
    DRY_RUN = False

    # ESRI code that requires extra fixes to the mapping data variable
    ESRI_CODE = 102027

    # Flag if corresponding NSIDC metadata files need to be created for the granules
    CREATE_META_FILES = False

    # Token to be present in granule's path - this is to process only selected granules,
    # such as belonging to the same EPSG code
    PATH_TOKEN = ''

    def __init__(
        self,
        start_index: int=0,
        stop_index: int=-1,
        use_local_file = None
    ):
        """
        Initialize the object.
        """
        # S3FS to access files stored in S3 bucket
        self.s3 = s3fs.S3FileSystem(anon=True)

        self.local_file = use_local_file

        # If file with granules to process is provided, just use it
        if self.local_file:
            with open(self.local_file, 'r') as fh:
                self.infiles = json.load(fh)
                logging.info(f"Loaded {len(self.infiles)} granules from '{self.local_file}'")

        else:
            # Granule files as read from the S3 granule summary file
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

        logging.info(f"Starting with granule #{start_index} (stop={stop_index}), remains {len(self.infiles)} granules to fix")

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
                results = NSIDCFormat.fix_granule(target_bucket, target_dir, each, self.s3, NSIDCFormat.CREATE_META_FILES)
                logging.info("-->".join(results))

            total_num_files -= num_tasks
            start += num_tasks

    def __call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        Fix ITS_LIVE granules and create corresponding NSIDC meta files (spacial
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
            tasks = [
                dask.delayed(NSIDCFormat.fix_granule)(target_bucket, target_dir, each, self.s3, self.local_file, NSIDCFormat.CREATE_META_FILES)
                for each in self.infiles[start:start+num_tasks]
            ]
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

            gc.collect()

    @staticmethod
    def upload_to_s3(filename: str, target_dir: str, target_bucket: str, s3_client, remove_original_file: bool=True):
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
                    msgs.append(f"Removing local {filename}")
                    os.unlink(filename)

        except ClientError as exc:
            msgs.append(f"ERROR: {exc}")

        return msgs

    @staticmethod
    def fix_granule(target_bucket: str, target_dir: str, infilewithpath: str, s3, use_local_file, create_meta_files):
        """
        Fix granule format and create corresponding metadata files as required by NSIDC.

        Inputs:
        target_bucket: Target AWS S3 bucket to copy fixed granule to.
        target_dir: Directory in AWS S3 bucket to copy fixed granule to.
        infilewithpath: Path to input granule file.

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

        logging.info(f'filename: {infilewithpath}')
        logging.info(f'new_filename: {new_filename}')

        msgs = [f'Processing {infilewithpath} into new {new_filename}']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_granule = os.path.join(target_dir, new_filename)

        # Check if fixed granules already exists - ignore the check if granule
        # is provided from local file. Granules are provided from local file only
        # if these granules need to be re-generated.
        # This was a hack to re-process only specific granules :)
        # if use_local_file is None and NSIDCFormat.object_exists(bucket, bucket_granule):
        #     msgs.append(f'WARNING: {bucket.name}/{bucket_granule} already exists, skipping granule')
        #     return msgs

        s3_client = boto3.client('s3')

        # Read granule from S3
        with s3.open(infilewithpath) as fh:
            with xr.open_dataset(fh) as ds:
                msgs.extend(
                    NSIDCFormat.process_nc_file(
                        ds,
                        new_filename,
                        Encoding.IMG_PAIR,
                        epsg_code
                    )
                )

        # Copy new granule to S3 bucket
        msgs.extend(
            NSIDCFormat.upload_to_s3(new_filename, target_dir, target_bucket, s3_client, remove_original_file=False)
        )

        if create_meta_files:
            # Create spacial and premet metadata files, and copy them to S3 bucket
            meta_file = NSIDCMeta.create_premet_file(new_filename, url_tokens_1, url_tokens_2)
            msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

            meta_file = NSIDCMeta.create_spatial_file(new_filename, epsg_code)
            msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

        # Leave the file if dry run to examine the file
        if not NSIDCFormat.DRY_RUN:
            msgs.append(f"Removing local {new_filename}")
            os.unlink(new_filename)

        return msgs

    @staticmethod
    def process_nc_file(
        ds,
        new_filename: str,
        encoding_params: dict,
        epsg_code: int,
        chunk_size: int=0
    ):
        """
        Fix granule format:
        1. Rename projection variable to be 'mapping'
        2. Replace all projection attributes to new value of 'mapping'
        3. Remove 'missing_value' attribute
        4. Replace units = "m/y" to units='meter/year'
        5. Change standard_name for vx, vy, and v to:
          “land_ice_surface_x_velocity”, “land_ice_surface_y_velocity” and “land_ice_surface_velocity”
        6. For UTM_Projection: set grid_mapping_name=transverse_mercator
        7. Replace 'binary' units
        8. Add standard_name = 'image_pair_information' to img_pair_info
        9. Mosaics only: update long_name = 'error weighted average time separation between image-pairs' for dt
        10. Mosaics only: fix count's units='count'
        """
        _missing_value = 'missing_value'
        _meter_year_units = 'meter/year'

        _conventions = 'Conventions'
        _cf_value = 'CF-1.8'

        _transverse_mercator = 'transverse_mercator'

        flag_values = 'flag_values'
        flag_meanings = 'flag_meanings'

        _ocean = 'ocean'
        _ice = 'ice'
        _rock = 'rock'

        _dt = 'dt'
        _dt_info = 'error weighted average time separation between image-pairs'

        _count = 'count'

        _utm_zone_number = 'utm_zone_number'
        _lambert_conformal_conic = 'lambert_conformal_conic'

        binary_flags = np.array([0, 1], dtype=np.uint8)

        _std_name = {
            DataVars.V: 'land_ice_surface_velocity',
            DataVars.VX: 'land_ice_surface_x_velocity',
            DataVars.VY: 'land_ice_surface_y_velocity'
        }

        _binary_meanings = {
            DataVars.INTERP_MASK: 'measured interpolated',
            _ocean: 'non-ocean ocean',
            _ice: 'non-ice ice',
            _rock: 'non-rock rock',
        }

        _image_pair_info = 'image_pair_information'

        msgs = []

        ds.attrs[_conventions] = _cf_value

        # Convert keys to list since we will remove some of the variables
        # during iteration
        for each_var in list(ds.keys()):
            msgs.append(f'Processing {each_var}')
            if _missing_value in ds[each_var].attrs:
                # 3. Remove 'missing_value' attribute
                del ds[each_var].attrs[_missing_value]

            if DataVars.UNITS in ds[each_var].attrs and \
                ds[each_var].attrs[DataVars.UNITS] == DataVars.M_Y_UNITS:
                # 4. Replace units = "m/y" to units='meter/year'
                ds[each_var].attrs[DataVars.UNITS] = _meter_year_units

            # 5. Change standard_name for vx, vy, and v
            if each_var in _std_name:
                ds[each_var].attrs[DataVars.STD_NAME] = _std_name[each_var]

            # 7. Replace 'binary' units
            if each_var in [DataVars.INTERP_MASK, _ocean, _ice, _rock] and DataVars.UNITS in ds[each_var].attrs:
                del ds[each_var].attrs[DataVars.UNITS]
                ds[each_var].attrs[flag_values] = binary_flags
                ds[each_var].attrs[flag_meanings] = _binary_meanings[each_var]

            # 8. Add standard_name = 'image_pair_information' to img_pair_info
            if each_var == DataVars.ImgPairInfo.NAME:
                ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.STD_NAME] = _image_pair_info

                # Reset to data of int type to get rid of dimensionality
                ds[DataVars.ImgPairInfo.NAME] = xr.DataArray(
                    attrs=ds[each_var].attrs,
                    coords={},
                    dims=[]
                )

            # 9. Fix long_name for dt
            if each_var == _dt:
                ds[each_var].attrs['long_name'] = _dt_info

            # 10. Fix count's units='_count'
            if each_var == _count and DataVars.UNITS in ds[each_var].attrs:
                ds[each_var].attrs[DataVars.UNITS] = _count

            if DataVars.GRID_MAPPING in ds[each_var].attrs:
                # 2. Replace projection attribute to "mapping"
                ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

            elif each_var in [DataVars.UTM_PROJECTION, DataVars.POLAR_STEREOGRAPHIC]:
                # 6. For UTM_Projection: set grid_mapping_name=transverse_mercator
                if each_var == DataVars.UTM_PROJECTION:
                    ds[each_var].attrs[DataVars.GRID_MAPPING_NAME] = 'transverse_mercator'

                # 1. Rename projection variable to 'mapping'
                # Can't copy the whole data variable, as it introduces obscure coordinates.
                # Just copy all attributes for the scalar type of the xr.DataArray.
                ds[DataVars.MAPPING] = xr.DataArray(
                    attrs=ds[each_var].attrs,
                    coords={},
                    dims=[]
                )

                if epsg_code == NSIDCFormat.ESRI_CODE:
                    # Extra fixes are required to the mapping data variable
                    del ds[DataVars.MAPPING].attrs[_utm_zone_number]
                    ds[DataVars.MAPPING].attrs[DataVars.GRID_MAPPING_NAME] = _lambert_conformal_conic

                # Delete old projection variable
                msgs.append(f'Deleting {each_var}')
                del ds[each_var]

        # Additional fixes to "mapping"
        msgs.append(fix_mapping_attrs(ds[DataVars.MAPPING], epsg_code))

        if chunk_size:
            # Convert dataset to Dask dataset not to run out of memory while writing to the file
            ds = ds.chunk(chunks={'x': chunk_size, 'y': chunk_size})

        # Write fixed granule to local file
        ds.to_netcdf(new_filename, engine='h5netcdf', encoding = encoding_params)

        return msgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description="""
           Fix ITS_LIVE V1 velocity image pairs to be CF compliant for
           ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '-catalog_dir',
        action='store',
        type=str,
        default='catalog_geojson/landsatOLI/v01',
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
        default='used_granules_landsat.json',
        help='Filename with JSON list of granules [%(default)s], file is stored in  "-catalog_dir"'
    )

    parser.add_argument(
        '-start_index',
        action='store',
        type=int,
        default=0,
        help="Start index for the granule to fix [%(default)d]. " \
             "Useful if need to continue previously interrupted process to fix the granules."
    )

    parser.add_argument(
        '-stop_index',
        action='store',
        type=int,
        default=-1,
        help="Stop index for the granules to fix [%(default)d]. " \
             "Usefull if need to split the job between multiple processes."
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
        '--create_meta_files',
        action='store_true',
        help='Flag to enable generation of NSIDC required metadata files. Default is False.'
    )

    parser.add_argument(
        '-t', '--path_token',
        type=str,
        default='',
        help="Path token to be present in granule S3 target path in order for the granule to be processed [%(default)s]."
    )

    args = parser.parse_args()

    NSIDCFormat.GRANULES_FILE = os.path.join(args.bucket, args.catalog_dir, args.granules_file)
    NSIDCFormat.DRY_RUN = args.dryrun
    NSIDCFormat.CREATE_META_FILES = args.create_meta_files
    NSIDCFormat.PATH_TOKEN = args.path_token

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = NSIDCFormat(
        args.start_index,
        args.stop_index,
        args.use_granule_file
    )
    nsidc_format(
        args.bucket,
        args.target_dir,
        args.chunk_by,
        args.dask_workers
    )

    logging.info('Done.')
