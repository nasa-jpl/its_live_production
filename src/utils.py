"""
Utility variables and functions for ITS_LIVE processing.
"""
from dataclasses import dataclass
import numpy as np

S3_PREFIX = 's3://'
HTTP_PREFIX = 'https://'
PATH_URL = ".s3.amazonaws.com"

NC_ENGINE = 'h5netcdf'


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
      return f"{FileInfo.prefix.datacube}_{epsg_format}_G{grid_size:04d}_" \
               f"X{mid_x}_Y{mid_y}{FileInfo.ext.zarr}"

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
   compressor: str = 'compressor'
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
