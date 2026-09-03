"""
Utility variables and functions for ITS_LIVE processing.
"""
from dataclasses import dataclass
from dateutil.parser import parse
from datetime import datetime
import numpy as np
import os
import xarray as xr

S3_PREFIX = 's3://'
HTTP_PREFIX = 'https://'

# Token within granule's HTTP URL that needs to be replaced to get file
# location within S3 bucket using S3 URL:
# from 'https://its-live-data.s3.amazonaws.com/file.nc'
# to
# 's3://its-live-data/file.nc'
PATH_URL = ".s3.amazonaws.com"

# Engine to read xarray data into from NetCDF filecompression
NC_ENGINE = 'h5netcdf'

# Token to split image pair filename into two image names
SPLIT_IMAGES_TOKEN = '_X_'
IMAGE_TOKEN = '_'

# Date format as it appears in granules filenames:
# (LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc)
DATE_FORMAT = "%Y%m%d"


@dataclass(frozen=True)
class CoordsInfo:
   """
   Coordinates for the ITS_LIVE datasets.
   """
   # Granule's "time" dimension within datacubes
   MID_DATE: str = 'mid_date'
   TIME: str = 'time'
   SENSORS: str = 'sensor'
   X: str = 'x'
   Y: str = 'y'

   STD_NAME = {
      MID_DATE: "image_pair_center_date_with_time_separation",
      X: "projection_x_coordinate",
      Y: "projection_y_coordinate",
      TIME: 'time',
      SENSORS: 'sensors',
   }

   DESCRIPTION = {
      MID_DATE:   "midpoint of image 1 and image 2 acquisition date and time "
                  "with granule's centroid longitude and latitude as "
                  "microseconds",
      X:          "x coordinate of projection",
      Y:          "y coordinate of projection",
      TIME:       "time",
      SENSORS:    "combinations of unique sensors and missions that are "
                  "grouped together for date_dt filtering",
   }

Coords = CoordsInfo()


# Former FileExtension
@dataclass(frozen=True)
class FileExtInfo:
   """
   File extensions used by ITS_LIVE data sets.
   """
   zarr: str = '.zarr'
   json: str = '.json'
   nc: str = '.nc'
   icechunk: str = '.icechunk'


# Former FilenamePrefix
@dataclass(frozen=True)
class FilenamePrefix:
   """
   Filename prefixes used by ITS_LIVE data products.
   """
   datacube: str = 'ITS_LIVE_vel'
   composites: str = 'ITS_LIVE_velocity'
   mosaics: str = 'ITS_LIVE_velocity'


@dataclass(frozen=True)
class FileInfo:
   ext: FileExtInfo = FileExtInfo()
   prefix: FilenamePrefix = FilenamePrefix()

   mosaicsSummaryKey: str = '0000'

   @staticmethod
   def datacube_filename(
      epsg_format: str,
      grid_size: int,
      mid_x: int,
      mid_y: int
   ):
      """
      Format filename for the datacube as:
      ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000
      """
      return f"{FileInfo.prefix.datacube}_{epsg_format}_G{int(grid_size):04d}_X{mid_x}_Y{mid_y}"

   # Former datacube_filename_zarr
   @staticmethod
   def datacube_filename_icechunk(
      epsg_format: str,
      grid_size: int,
      mid_x: int,
      mid_y: int
   ):
      """
      Format filename for the virtual datacube icechunk repository as:
      ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr
      """
      name = FileInfo.datacube_filename(epsg_format, grid_size, mid_x, mid_y)
      return f"{name}{FileInfo.ext.zarr}"

   # Former datacube_filename_zarr
   @staticmethod
   def datacube_filename_zarr(
      epsg_format: str,
      grid_size: int,
      mid_x: int,
      mid_y: int
   ):
      """
      Format filename for the datacube as:
      ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr
      """
      name = FileInfo.datacube_filename(epsg_format, grid_size, mid_x, mid_y)
      return f"{name}{FileInfo.ext.zarr}"

   # Former composite_filename_zarr
   @staticmethod
   def composite_filename_zarr(
      epsg_format: int,
      grid_size: str,
      mid_x: int,
      mid_y: int
   ):
      """
      Format filename for the datacube's composite. For example,
      ITS_LIVE_velocity_EPSG3413_120m_X-3250000_Y250000.zarr.

      Inputs:
      =======
      epsg_format: String representation of the EPSG code in "EPSGXXXXX"
                     format.
      grid_size: Grid size
      mid_x: X coordinate of datacube centroid
      mid_y: Y coordinate of datacube centroid
      """
      return f"{FileInfo.prefix.composites}_EPSG{epsg_format}" \
            f"_{int(grid_size):03d}m_X{mid_x}_Y{mid_y}{FileInfo.ext.zarr}"

   # Former annual_mosaics_filename_nc
   @staticmethod
   def annual_mosaics_filename_nc(
      grid_size: str,
      region: str,
      year_date,
      version: str
   ):
      """
      Format filename for the annual mosaics of the region:
      ITS_LIVE_velocity_120m_ALA_2013_v02.nc

      Inputs:
      =======
      grid_size: Size of the grid cell (assumes the same in X and Y dimentions)
      region: Region for which mosaic file is created.
      year_date: Year for which mosaic file is created. Can be a string or a
                  datetime object.
      """
      year_value = year_date

      if not isinstance(year_value, str):
         # Provided as datetime object, extract year value
         year_value = year_date.year

      return f"{FileInfo.prefix.mosaics}_{grid_size}m_{region}_{year_value}_" \
            f"{version}{FileInfo.ext.nc}"

   # Former get_corresponding_static_mosaics_filename
   @staticmethod
   def static_mosaics_filename(
      year_date,
      annual_mosaics_filename: str
   ):
      """
      Get filename for static mosaics filename that is based on existing annual
      mosaics for the region.
      Given "ITS_LIVE_velocity_120m_ALA_2013_v02.nc" filename it will generate
      "ITS_LIVE_velocity_120m_ALA_0000_v02.nc"

      Inputs:
      =======
      year_date: Year for which mosaic file is created. Can be a string or a
                  datetime object.
      annual_mosaics_filename: Name of the annual mosaics filename.
      """
      year_value = year_date

      if not isinstance(year_value, str):
         # Provided as datetime object, extract year value
         year_value = year_date.year

      return annual_mosaics_filename.replace(
         f'{year_value}', FileInfo.mosaicsSummaryKey
      )

   # Former summary_mosaics_filename_nc
   @staticmethod
   def summary_mosaics_filename_nc(
      grid_size: str,
      region: str,
      version: str
   ):
      """
      Format filename for the summary mosaics of the region:
      ITS_LIVE_velocity_120m_ALA_0000_v02.nc
      """
      return f"{FileInfo.prefix.mosaics}_{grid_size}m_{region}_" \
               f"{FileInfo.mosaicsSummaryKey}_{version}{FileInfo.ext.nc}"

File = FileInfo()


@dataclass(frozen=True)
class UnitsInfo:
   """Units related information.
   """
   # Attribute name for units in the data variables
   name: str = 'units'

   m_y: str = 'meter/year'
   m_y2: str = 'meter/year^2'
   m: str = 'm'
   count: str = 'count'
   binary: str = 'binary'
   percent: str = 'percent'
   day_of_year: str = 'day of year'
   pixel_per_m_year: str = 'pixel/(meter/year)'
   m_per_year_pixel: str = 'meter/(year*pixel)'
   date: str = 'days since 1970-01-01'
   days: str = 'days'

   # Attribute name for calendar in datetime-encoded data variables
   calendar_name: str = 'calendar'

   # GPS epoch, used as the reference date for datetime-valued variables
   # (the cube's 'time' coordinate and the img_pair_info-derived
   # acquisition_date_img1/acquisition_date_img2/date_center variables) so
   # they can be encoded as float64 seconds without any resolution loss --
   # see virtual_itslive_cube_per_chunk.py and virtual_itslive_cube.py.
   gps_epoch_date: str = 'seconds since 1980-01-06T00:00:00+00:00'
   proleptic_gregorian: str = 'proleptic_gregorian'

Units = UnitsInfo()


@dataclass(frozen=True)
class MissingInfo:
   """
   Class to represent missing value information for the data variables of
   integer datatype in the ITS_LIVE data products.
   """
   # Attribute name for missing value in integer data variables in Zarr
   # format. For NetCDF format, the missing value is set using the
   # "_FillValue" attribute.
   name: str = 'missing_value'
   # The zarr-level sentinel, separate from any CF attribute.
   fill_value: str = 'fill_value'

   # Missing values for different data types
   # Missing (FillValue) values for data variables
   byte = 0.0          # missing value for byte type, ex MISSING_BYTE
   value = -32767      # missing value for int type. ex MISSING_VALUE
   uvalue = 32767      # missing value for uint type, ex MISSING_POS_VALUE
   u8value = 255       # missing value for uint8 type, ex MISSING_UINT8_VALUE

Missing = MissingInfo()


# Former Output
@dataclass(frozen=True)
class OutputFormatInfo:
   """
   Class to represent output format information for the ITS_LIVE data
   products.
   """
   # Standard attributes for the output format
   dtype: str = 'dtype'
   # Zarr v2 encoding key: a single compressor codec. Zarr v3 stores (e.g.
   # icechunk repos) use the plural 'compressors' key instead, whose value is
   # a list of codecs -- see OutputFormatInfo.compressors below and
   # deep_copy_cube.py's COMPRESSOR_KEY.
   compressor: str = 'compressor'
   compressors: str = 'compressors'
   # For the floating point types in Zarr format, and any datatype in NetCDF
   # format.
   # Integer types in Zarr format use 'missing_value' attribute instead of
   # '_FillValue' to specify the missing value.
   fill_value: str = '_FillValue'
   chunks: str = 'chunks'
   chunksizes: str = 'chunksizes'

   # These encoding attributes are for M11 and M12 variables in radar granules
   scale_factor: str = 'scale_factor'
   add_offset: str = 'add_offset'

   # Global attributes
   conventions: str = 'Conventions'
   institution: str = 'institution'

   title: str = 'title'
   author: str = 'author'
   citation: str = 'citation'
   latitude: str = 'latitude'
   longitude: str = 'longitude'

   count: str = 'count'
   url: str = 'url'
   s3: str = 's3'
   references: str = 'references'
   projection: str = 'projection'
   publisher_name: str = 'publisher_name'

OutputFormat = OutputFormatInfo()


def to_int_type(
   data, data_type=np.uint16, fill_value=Missing.uvalue
):
   """
   Convert data to requested integer datatype. "fill_value" must correspond
   to the "data_type" to replace NaNs with corresponding to the datatype
   missing_value:
   -32767 for int16/32
   32767 for uint16/32
   etc.

   Inputs:
   =======
   data: Data to convert to new datatype to. It can be of numpy.ndarray or
         xarray.DataArray data type.
   data_type: numpy data type to convert data to. Default is np.uint16.
   fill_value: value to replace NaN's with before conversion to integer type.
   """
   # Replace NaN's with fill_values as it will store garbage for NaN's
   _mask = np.isnan(data)
   data[_mask] = fill_value

   # Mask Inf's with maximum value for the target dtype
   _mask = np.isinf(data)
   data[_mask] = np.iinfo(data_type).max

   # Round to nearest int value
   int_data = np.rint(data).astype(data_type)

   return int_data


def parse_time(value: str, var_name: str = None, attr_name: str = None,
               ds_url: str = None):
   """Parse string value into datetime object.

   Inputs:
   =======
   value: str
      String representation of the datetime.
   var_name: str
      Name of the variable for which value is parsed (for error reporting
      only). Default is None.
   attr_name: str
      Name of the variable attribute for which value is parsed (for error
      reporting only). Default is None.
   ds_url: str
      URL of the file for which value is parsed (for error reporting only).
      Default is None.
   """
   try:
      tokens = value.split('T')
      if len(tokens) == 3:
         # Handle malformed datetime in Sentinel 2 granules:
         # img_pair_info.acquisition_date_img1 = "20190215T205541T00:00:00"
         value = tokens[0] + 'T' + tokens[1][0:2] + ':' \
                  + tokens[1][2:4] + ':' + tokens[1][4:6]
         value = datetime.strptime(value, '%Y%m%dT%H:%M:%S')

      elif len(value) >= 8:
         value = parse(value)

      return value

   except ValueError as exc:
      raise RuntimeError(
         f"Error converting {value} to date format '%Y%m%d': "
         f"{exc} for {var_name}.{attr_name} in {ds_url}"
      )


def get_data_var_binary_attr(
   ds: xr.Dataset,
   ds_url: str,
   var_name: str,
   attr_name: str,
   token: str,
   data_dtype=np.uint8,
   missing_value=None
):
   """
   Return attribute for the data variable in data set if it exists,
   or missing_value if it is not present.
   If "missing_value" is set to None, than specified attribute is expected
   to exist for the data variable "var_name" and exception is raised if
   it does not.

   Inputs:
   ds (xarray.Dataset): The dataset the variable belongs to.
   ds_url (str): URL of the granule that corresponds to the input "ds"
                  dataset (used for error reporting only).
   var_name (str): Name of the variable to extract attribute for.
   attr_name (str): Name of the attribute to extract value for.
   token (str): Token to use for the attribute value to convert it to
                  binary value. If token is present in the attribute
                  value, then the attribute value is set to 1,
                  otherwise 0.
   missing_value: Value to use if attribute is missing for the variable.
                  Default is None, which will result in raising an
                  exception if attribute is missing for the variable.
   data_dtype: Datatype to use for the attribute value. Default is
               np.uint8.
   """
   if var_name in ds and attr_name in ds[var_name].attrs:
      # NISAR workaround for some attributes being stored as arrays
      # instead of a single value: take the first element of the array
      # if it has only one element.``
      value = ds[var_name].attrs[attr_name]

      if np.ndim(value) != 0:
            # Not a scalar (int, float, or 0-d numpy array)
            # list, tuple, or numpy array.
            value = np.asarray(value).flat[0]

      value = data_dtype(value == token)

      # print(f"Return value for {var_name}.{attr_name}: {value}")
      return value

   if missing_value is None:
      # If missing_value is not provided, attribute is expected to exist always
      raise RuntimeError(
            f"{attr_name} is expected within {var_name} for {ds_url}"
      )

   return missing_value


def get_data_var_attr(
   ds: xr.Dataset,
   ds_url: str,
   var_name: str,
   attr_name: str,
   missing_value=None,
   to_date=False,
   data_dtype=np.float32
):
   """
   Return attribute for the data variable in data set if it exists,
   or missing_value if it is not present.
   If "missing_value" is set to None, than specified attribute is expected
   to exist for the data variable "var_name" and exception is raised if
   it does not.

   Inputs:
   ds (xarray.Dataset): The dataset the variable belongs to.
   ds_url (str): URL of the granule that corresponds to the input "ds"
                  dataset (used for error reporting only).
   var_name (str): Name of the variable to extract attribute for.
   attr_name (str): Name of the attribute to extract value for.
   missing_value: Value to use if attribute is missing for the variable.
                  Default is None, which will result in raising an
                  exception if attribute is missing for the variable.
   to_date (bool): Flag if attribute value should be converted to
                  datetime object. Default is False.
   data_dtype: Datatype to use for the attribute value. Default is
               np.float32.
   """
   if var_name in ds and attr_name in ds[var_name].attrs:
      # NISAR workaround for some attributes being stored as arrays
      # instead of a single value: take the first element of the array
      # if it has only one element.``
      value = ds[var_name].attrs[attr_name]

      # Check if type has "length"
      # if hasattr(type(value), '__len__') and len(value) == 1:
      #     value = value[0]

      if np.ndim(value) != 0:
            # Not a scalar (int, float, or 0-d numpy array)
            # list, tuple, or numpy array.
            value = np.asarray(value).flat[0]

      # print(f"Read value for {var_name}.{attr_name}: {value}")
      if to_date is True:
            value = parse_time(value, var_name, attr_name, ds_url)

      else:
            # Convert value to expected datatype
            if data_dtype and \
               not isinstance(data_dtype, np.dtypes.StringDType):
               value = data_dtype(value)

      # print(f"Return value for {var_name}.{attr_name}: {value}")
      return value

   if missing_value is None:
      # If missing_value is not provided, attribute is expected to exist always
      raise RuntimeError(
            f"{attr_name} is expected within {var_name} for {ds_url}"
      )

   return missing_value


def extract_mid_date_from_url(url: str):
   """
   Extract mid_date from granule filename by parsing acquisition dates.
   This method is used to sort granules in chronological order by
   acquisition date by avoiding reading the granule files to get its time
   dimension value.

   Supports multiple sensor filename formats:
   - Landsat: acquisition date at token[3] (format: YYYYMMDD)
   - NISAR: acquisition date+time at token[11] (format: YYYYMMDDTHHMMSS)
   - Sentinel-1: acquisition date+time at token[5] (format: YYYYMMDDTHHMMSS)
   - Sentinel-2: acquisition date+time at token[2] (format: YYYYMMDDTHHMMSS)

   Mid_date is calculated as the average of the two acquisition dates.

   Inputs:
   url (str): URL for the granule.

   Returns:
   Datetime object for sorting purposes.
   """
   from datetime import datetime, timedelta

   # Extract filename from URL
   filename = os.path.basename(url)

   # Split into two images
   images = filename.split(SPLIT_IMAGES_TOKEN)
   if len(images) < 2:
      raise RuntimeError(
            f'Filename does not contain expected split token: '
            f'{SPLIT_IMAGES_TOKEN} in {filename}'
      )

   # Parse first image tokens
   tokens_1 = images[0].split(IMAGE_TOKEN)
   # Parse second image tokens
   tokens_2 = images[1].split(IMAGE_TOKEN)

   # Detect sensor type and extract acquisition dates accordingly
   sensor_prefix = tokens_1[0][:5] if len(tokens_1[0]) >= 5 else tokens_1[0]

   if sensor_prefix == 'NISAR':
      # NISAR: acquisition date+time at token[11]
      # Format: YYYYMMDDTHHMMSS (e.g., 20251120T130632)
      date_token_idx = 11
      date_format = "%Y%m%dT%H%M%S"

   elif sensor_prefix.startswith('S1'):
      # Sentinel-1: acquisition date+time at token[5]
      # Format: YYYYMMDDTHHMMSS (e.g., 20200221T095209)
      date_token_idx = 5
      date_format = "%Y%m%dT%H%M%S"

   elif sensor_prefix.startswith('S2'):
      # Sentinel-2: acquisition date+time at token[2]
      # Format: YYYYMMDDTHHMMSS (e.g., 20181008T190459)
      date_token_idx = 2
      date_format = "%Y%m%dT%H%M%S"

   elif sensor_prefix.startswith('L'):
      # Landsat (LC08, LC09, LE07, LT05, etc.): acquisition date at token[3]
      # Format: YYYYMMDD
      date_token_idx = 3
      date_format = DATE_FORMAT

   else:
      # Unsupported sensor format
      raise ValueError(
            f"Unsupported sensor filename format: {sensor_prefix} in {filename}"
      )

   # Parse acquisition dates using the determined token index and format
   try:
      date_1 = datetime.strptime(tokens_1[date_token_idx], date_format)
      date_2 = datetime.strptime(tokens_2[date_token_idx], date_format)
   except IndexError:
      raise RuntimeError(
            f'Missing expected token at index {date_token_idx} for sensor '
            f'{sensor_prefix} in filename: {filename}'
      )
   except ValueError as e:
      raise RuntimeError(
            f'Invalid date format at token {date_token_idx} for sensor '
            f'{sensor_prefix} in filename {filename}: {e}'
      )

   # Calculate mid_date as average
   mid_date = date_1 + (date_2 - date_1) / 2
   return mid_date


def get_tokens_from_filename(filename):
   """
   Extract processing dates for two images from the filename and
   construct unique identifier for the image pair by removing processing
   dates, percent valid pixels fields and file extension.

   Inputs:
   filename (str): Granule filename to parse.

   Returns:
   url_proc_date_1 (datetime): Processing date for first image.
   url_proc_date_2 (datetime): Processing date for second image.
   id (str): Unique identifier for the image pair.
   """
   files = os.path.basename(filename).split(SPLIT_IMAGES_TOKEN)

   # Get acquisition, processing date, path_row for both images
   # from url and index_url
   url_tokens = os.path.basename(files[0]).split(IMAGE_TOKEN)

   url_proc_date_1 = datetime.strptime(url_tokens[4], DATE_FORMAT)

   # Remove processing date from the first image name: don't replace date
   # token with an empty string as acquisition and processing dates can be
   # the same
   id_tokens = url_tokens[:4]
   id_tokens.extend(url_tokens[5:])

   url_tokens = os.path.basename(files[1]).split(IMAGE_TOKEN)
   url_proc_date_2 = datetime.strptime(url_tokens[4], DATE_FORMAT)

   # Remove processing date and _Pxxx.nc from the second image name
   id_tokens.extend(url_tokens[:4])
   id_tokens.extend(url_tokens[5:8])

   id = IMAGE_TOKEN.join(id_tokens)

   return url_proc_date_1, url_proc_date_2, id

