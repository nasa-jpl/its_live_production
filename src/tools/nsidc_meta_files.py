"""
Class to prepare V2 ITS_LIVE granules to be ingested by NSIDC:

* Generate metadata file
* Generate spacial info file

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""
import boto3
import collections
from dateutil.parser import parse
import numpy as np
import os
import pyproj
import s3fs
import xarray as xr

from itscube_types import DataVars


def get_sensor_tokens_from_filename(filename: str):
   """
   Extract sensor tokens for two images from the granule
   filename. The filename is expected to have the following format:
   <image1_tokens>_X_<image2_tokens>_<granule_tokens>.nc.
   """
   url_files = os.path.basename(filename).split('_X_')

   # Get tokens for the first image
   url_tokens_1 = url_files[0].split('_')

   # Extract info from second part of the granule's filename: corresponds to the second image
   url_tokens_2 = url_files[1].split('_')

   # Return sensor tokens for both images
   return (url_tokens_1[0], url_tokens_2[0])


# Collection to represent the mission and sensor combo
PlatformSensor = collections.namedtuple("PM", ['platform', 'sensor'])


class NSIDCMeta:
   """
   Class to create premet and spacial files for v2 ITS_LIVE granule.

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

   PREMET_EXT = '.premet'
   SPATIAL_EXT = '.spatial'

   @staticmethod
   def create_premet_file(ds: xr.Dataset, infile: str):
      """
      Create premet file that corresponds to the input image pair velocity granule.

      Inputs
      ======
      ds: xarray.Dataset object that represents the granule.
      infile: Filename of the input ITS_LIVE granule
      """
      # Extract tokens from the filename
      sensor1, sensor2 = get_sensor_tokens_from_filename(infile)

      if sensor1 not in NSIDCMeta.ShortName:
         raise RuntimeError(
            f'create_premet_file(): got unexpected mission+sensor '
            f'{sensor1} for image#1 of {infile}: one of '
            f'{list(NSIDCMeta.ShortName.keys())} is supported.'
         )

      if sensor2 not in NSIDCMeta.ShortName:
         raise RuntimeError(
            f'create_premet_file() got unexpected mission+sensor '
            f'{sensor2} for image#2 of {infile}: one of '
            f'{list(NSIDCMeta.ShortName.keys())} is supported.'
         )

      # Get acquisition dates for both images
      begin_date = parse(ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1])
      end_date = parse(ds[DataVars.ImgPairInfo.NAME].attrs[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2])

      meta_filename = f'{infile}{NSIDCMeta.PREMET_EXT}'
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
      meta_filename = f'{infile}{NSIDCMeta.SPATIAL_EXT}'

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


def upload_to_s3(
   filename: str,
   target_dir: str,
   target_bucket: str,
   s3_client=None,
   remove_original_file: bool = True
):
   """
   Upload file to the AWS S3 bucket, and remove local file if requested.

   Inputs:
   filename: Filename to be uploaded.
   target_dir: Target directory in the S3 bucket.
   target_bucket: Target bucket in the S3.
   s3_client: S3 client object.
   remove_original_file: Flag to remove the original file after uploading. Default is True.
   """
   target_filename = os.path.join(target_dir, filename)

   if s3_client is None:
      s3_client = boto3.client('s3')

   s3_client.upload_file(filename, target_bucket, target_filename)

   if remove_original_file:
      os.unlink(filename)


def create_nsidc_meta_files(infilewithpath: str, s3: s3fs.S3FileSystem = None):
   """
   Create premet and spatial files for the input ITS_LIVE granule.

   Inputs
   ======
   infilewithpath: Filename of the input ITS_LIVE granule.
   s3: S3FileSystem object to access the S3 bucket. Default is None.

   Returns:
   ========
   A list of premet and spatial filenames generated locally.
   """
   granule_filename = infilewithpath

   if os.path.sep in infilewithpath:
      filename_tokens = infilewithpath.split('/')
      granule_filename = filename_tokens[-1]

   # Extract tokens from the filename
   sensor1, sensor2 = get_sensor_tokens_from_filename(granule_filename)

   if s3 is None:
      s3 = s3fs.S3FileSystem(anon=False)

   # Open the granule
   with s3.open(infilewithpath, mode='rb') as fhandle:
      with xr.open_dataset(fhandle, engine=NSIDCMeta.NC_ENGINE) as ds:
         # Create premet file
         premet_file = NSIDCMeta.create_premet_file(ds, granule_filename)

         # Create spatial file
         spatial_file = NSIDCMeta.create_spatial_file(ds, granule_filename)

   return premet_file, spatial_file
