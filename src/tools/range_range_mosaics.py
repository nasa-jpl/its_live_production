"""This tool is used to generate range-range velocity granules
based on 12-day collection of the granules for provided polygon which defines
a region of interest.

The tool is reading polygon from the Geojson file (to support long lists of
coordinates).

The tool:
1. Submits a query to the STAC catalog for ascending granules for the time
   period and geographical region.
2. Submits a query to the STAC catalog for descending granules for the time
   period and geographical region.
3. Merges each of ascending and descending granules collections into a
   single mosaic, using minimum "v_error" masking for overlapping regions.
4. It populates dr_to_vr_factor attribute for each granule in the collection
   by populating it with the value of the dr_to_vr_factor attribute from M11.
4a. TODO: has an option to apply average of M11.attrs['dr_to_vr_factor'] thus
   using a scalar in range-range calculations.

Please refer to the github issue
https://github.com/nasa-jpl/its_live_production/issues/40 for more details
on the format definition of range-range velocity granules.
"""
import argparse
import datetime
from dateutil.parser import parse
import gc
import s3fs
import json
import logging
import numpy as np
from pystac_client import Client
import sys
import time
import xarray as xr
import warnings

import grid
from itscube_types import \
   Coords, \
   DataVars, \
   Output, \
   MappingInstance

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Date format string for STAC queries
DATE_FORMAT = '%Y-%m-%d'

# Encoding for S1 mosaics
SENTINEL1_ENCODING = {
   'M11':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
   'M12':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
   'v':                      {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'vx':                     {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'vy':                     {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'v_error':                {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'vr':                     {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   DataVars.DR_TO_VR_FACTOR: {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
   'x':                      {'_FillValue': None},
   'y':                      {'_FillValue': None}
}

RANGE_RANGE_ENCODING = {
   'v':                      {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'vx':                     {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'vy':                     {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
   'x':                      {'_FillValue': None},
   'y':                      {'_FillValue': None}
}

NC_ENGINE = 'h5netcdf'

ascending = "ascending"
descending = "descending"

# Name of new dimension to use when concatenating multiple composites into one xr.DataArray
CONCAT_DIM_NAME = 'new_dim'

# Cell dimensions
x_cell = 120.0
half_x_cell = x_cell/2.0
y_cell = -120.0
half_y_cell = y_cell/2.0

# Expected cell size
CELL_SIZE = 120


def load_polygon(granule_file: str):
   """
   Load polygon coordinates as loaded from the Geojson file.
   """
   granules_info = None

   with open(granule_file, 'r') as fh:
      granules_info = json.load(fh)

   return granules_info['features'][0]['geometry']['coordinates']


def search_stac(
   stac_catalog="https://stac.itslive.cloud",
   page_size=100,
   filter_list=[],
   **kwargs
):
   """
   Returns list of found granule URLs for the given STAC catalog.
   """
   catalog = Client.open(stac_catalog)
   search_kwargs = {
      "collections": ["itslive-granules"],
      "limit": page_size,
      **kwargs
   }

   def build_cql2_filter(filters_list):
      if not filters_list:
         return None
      return filters_list[0] if len(filters_list) == 1 else {"op": "and", "args": filters_list}
   if filter_list:
      filters = build_cql2_filter(filter_list)
      search_kwargs["filter"] = build_cql2_filter(filters)
      search_kwargs["filter_lang"] = "cql2-json"

   search = catalog.search(**search_kwargs)
   logging.info(f'Search STAC catalog: {search.get_parameters()}')

   hrefs = []
   pages_count = 0
   for page in search.pages():
      pages_count += 1
      for item in page:
         if kwargs.get("debug"):
            logging.info(f"fetching page {pages_count}")
         for asset in item.assets.values():
            if "data" in asset.roles and asset.href.endswith(".nc"):
               hrefs.append(asset.href)
      time.sleep(0.1)  # we can remove this one, just to avoid overwhelming the server

   logging.info(f"Requested pages: {pages_count}")
   return hrefs


def create_new_granule(x, y):
   """
   Create xr.Dataset to represent new range-range velocity granule.

   Input:
   x: x-coordinates
   y: y-coordinates

   Returns:
   xr.Dataset: New range-range velocity granule.
   """
   granule = xr.Dataset(
      coords={
         'y': (
               'y',
               # sorted(y),
               y,
               {
                  DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                  DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
               }
         ),
         'x': (
               'x',
               # sorted(x),
               x,
               {
                  DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                  DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
               }
         )
      }
   )
   return granule


def compute_velocity_based_on_m11_m12(
   m11_a,
   m12_a,
   m11_d,
   m12_d,
   vr_a,
   vr_d,
   dr_to_vr_factor_a,
   dr_to_vr_factor_d,
   trans1,
   trans2):
   """
   Compute velocity fields in map coordinates based on LOS (slant range)
   measurements of pixel displacement.

   vr_a and vr_b can be scalar values or 2D arrays.
   """
   ysize1, xsize1 = m11_a.shape
   ysize2, xsize2 = m11_d.shape

   W = np.min([trans1[0], trans2[0]])
   N = np.max([trans1[3], trans2[3]])
   E = np.max([trans1[0] + (xsize1 - 1)*trans1[1], trans2[0] + (xsize2 - 1)*trans2[1]])
   S = np.min([trans1[3] + (ysize1 - 1)*trans1[5], trans2[3] + (ysize2 - 1)*trans2[5]])

   trans = [W, trans1[1], 0.0, N, 0.0, trans1[5]]
   xsize = int(np.round((E - W) / trans[1])) + 1
   ysize = int(np.round((S - N) / trans[5])) + 1

   dims = (ysize, xsize)

   vr_1 = np.full(dims, np.nan)
   M11_1 = np.full(dims, np.nan)
   M12_1 = np.full(dims, np.nan)
   dr_to_vr_1 = np.full(dims, np.nan)

   vr_2 = np.full(dims, np.nan)
   M11_2 = np.full(dims, np.nan)
   M12_2 = np.full(dims, np.nan)
   dr_to_vr_2 = np.full(dims, np.nan)

   x1a = int(np.round((trans1[0] - W) / trans1[1]))
   x1b = x1a + xsize1
   y1a = int(np.round((trans1[3] - N) / trans1[5]))
   y1b = y1a + ysize1

   x2a = int(np.round((trans2[0] - W) / trans2[1]))
   x2b = x2a + xsize2
   y2a = int(np.round((trans2[3] - N) / trans2[5]))
   y2b = y2a + ysize2

   vr_1[y1a:y1b, x1a:x1b] = vr_a
   dr_to_vr_1[y1a:y1b, x1a:x1b] = dr_to_vr_factor_a
   M11_1[y1a:y1b, x1a:x1b] = m11_a
   M12_1[y1a:y1b, x1a:x1b] = m12_a

   vr_2[y2a:y2b, x2a:x2b] = vr_d
   dr_to_vr_2[y2a:y2b, x2a:x2b] = dr_to_vr_factor_d
   M11_2[y2a:y2b, x2a:x2b] = m11_d
   M12_2[y2a:y2b, x2a:x2b] = m12_d

   scale_factor = M11_1*M12_2 - M12_1*M11_2
   zero_mask = (scale_factor == 0)
   scale_factor[zero_mask] = np.nan

   vx = (M12_2 * vr_1 / dr_to_vr_1 - M12_1 * vr_2 / dr_to_vr_2) / scale_factor
   vy = (-M11_2 * vr_1 / dr_to_vr_1 + M11_1 * vr_2 / dr_to_vr_2) / scale_factor

   return vx, vy, xsize, ysize, trans


def v_error_zero_velocity_fill(vx_error, vy_error):
   """
   This function is derived from the autoRIFT v_error_cal()
   to compute fill value for the cells where computed velocity
   is zero.

   See: https://github.com/nasa-jpl/autoRIFT/blob/249e6d03afa84597091f56dca7f5d8fce37be026/netcdf_output.py#L28
   """
   return np.std(np.sqrt(vx_error**2 + vy_error**2))


def set_mapping(raw_ds, x_cell, y_cell, x_coords, y_coords):
   """
   Set mapping data variable for the current mosaics.

   Inputs:
   =======
   raw_ds: Dictionary of all granules for the mosaic.
   x_cell: Cell dimension in X
   y_cell: Cell dimension in Y
   x_coords: X grid coordinates
   y_coords: Y grid coordinates
   """
   ds_urls = sorted(list(raw_ds.keys()))
   # Use "first" dataset to "collect" global attributes
   first_ds = raw_ds[ds_urls[0]]

   # Create mapping data variable
   mapping = xr.DataArray(
      data='',
      attrs=first_ds[DataVars.MAPPING].attrs,
      coords={},
      dims=[]
   )

   # Set GeoTransform to correspond to the mosaic's tile:
   # format GeoTransform
   # Sanity check: check cell size for all mosaics against target cell size
   if CELL_SIZE != x_cell and CELL_SIZE != np.abs(y_cell):
      raise RuntimeError(f'Provided grid cell size {CELL_SIZE} does not correspond to the cell size of dataset: x={x_cell} or y={np.abs(y_cell)}')

   # :GeoTransform = "-3300007.5 120.0 0 300007.5 0 -120.0";
   new_geo_transform_str = f"{x_coords[0] - x_cell/2.0} {x_cell} 0 {y_coords[0] - y_cell/2.0} 0 {y_cell}"
   logging.info(f'Setting mapping.GeoTransform: {new_geo_transform_str}')
   mapping.attrs['GeoTransform'] = new_geo_transform_str

   # Return first of the datasets just to copy attributes
   return first_ds, mapping


def build_mosaics(granules, orbit_dir):
   """Build mosaics for the given granules and save it to the NetCDF file.

   Args:
      granules (list): list of granule URLs.
      orbit_dir (str): orbit direction, either 'ascending' or 'descending'.
   """
   # "united" coordinates for mosaics
   x_coords = []
   y_coords = []
   raw_ds = {}

   logging.info(f'Building mosaics for {orbit_dir} granules...')

   dr_to_vr_factor_var = DataVars.DR_TO_VR_FACTOR

   # Collect dr_to_vr_factor to compute average
   scalar_factors = []

   # Step through all granules and concatenate data in new dimension

   # Iterate over granules to introduce dr_to_vr_factor raster to each
   # of granules based on the M11 attribute
   for each in granules:
      each_s3 = each.replace('https://its-live-data.s3.amazonaws.com', 's3://its-live-data')

      with s3.open(each_s3, mode='rb') as fhandle:
         with xr.open_dataset(fhandle, engine=NC_ENGINE) as ds:
            logging.info(f'{each_s3=}')
            x_coords.append(ds.x.values)
            y_coords.append(ds.y.values)

            # Load only the data variables we need
            raw_ds[each_s3] = ds[['vx', 'vy', 'v', 'M11', 'M12',
                                    'vr', 'v_error', 'mapping']].load()
            logging.info(f'Loaded {list(raw_ds[each_s3].keys())} variables')

            # Add "dr_to_vr_factor" raster to each of them as it's a scalar and will
            # need to be set based on the minimum v_error
            scalar_factors.append(float(ds.M11.attrs['dr_to_vr_factor']))

            logging.info(f'Adding {dr_to_vr_factor_var=} data var with {ds.M11.attrs["dr_to_vr_factor"]}')

            # Create the mask: each_ds.M11.attrs['dr_to_vr_factor'] where 'M11' is not NaN, np.nan where it is NaN
            valid_mask = xr.where(~np.isnan(ds["M11"]), float(ds.M11.attrs['dr_to_vr_factor']), np.nan)

            # Add the mask to the dataset
            valid_mask.name = dr_to_vr_factor_var
            raw_ds[each_s3][dr_to_vr_factor_var] = valid_mask
            raw_ds[each_s3][dr_to_vr_factor_var].attrs[DataVars.GRID_MAPPING] = MappingInstance.name

      # Garbage collection
      gc.collect()

   gc.collect()

   logging.info(f'Mean dr_to_vr_factor for {orbit_dir}: {np.mean(scalar_factors)}')

   # merge coordinates
   x_coords = sorted(list(set(np.concatenate(x_coords))))
   y_coords = sorted(list(set(np.concatenate(y_coords))))

   # Compute cell size in x and y dimension
   x_cell = x_coords[1] - x_coords[0]
   y_cell = y_coords[1] - y_coords[0]

   x_coords = np.arange(x_coords[0], x_coords[-1]+1, x_cell)
   y_coords = np.arange(y_coords[0], y_coords[-1]+1, y_cell)

   # y coordinate in EPSG is always in ascending order
   y_coords = np.flip(y_coords)
   y_cell = y_coords[1] - y_coords[0]

   first_ds, mapping = set_mapping(raw_ds, x_cell, y_cell, x_coords, y_coords)

   # Dataset to represent mosaic for the given orbit direction
   ds = xr.Dataset(
      coords={
         Coords.X: (
               Coords.X,
               x_coords,
               first_ds[Coords.X].attrs
         ),
         Coords.Y: (
               Coords.Y,
               y_coords,
               first_ds[Coords.Y].attrs
         )
      },
      attrs={
         'range_range_mosaics': f'Greenland {orbit_dir}'
      }
   )

   ds[MappingInstance.name] = mapping

   ds_var = 'v_error'

   two_coords = [y_coords, x_coords]
   two_dims = [Coords.Y, Coords.X]
   two_dims_len = (len(y_coords), len(x_coords))

   data_list = []

   # Concatenated dataset
   concatenated = None

   # Allocate new data variable for the mask
   logging.info(f'Creating {ds_var=} overlap based on minimum values')

   # 1. Step through all granules and concatenate v_error data in new dimension
   for each_file, each_ds in raw_ds.items():
      logging.info(f'Merging {ds_var} from {each_file}')

      if ds_var not in ds:
         # Add variable to the result dataset
         ds[ds_var] = xr.DataArray(
               data=np.full(two_dims_len, np.nan),
               coords=two_coords,
               dims=two_dims,
               attrs=each_ds[ds_var].attrs
         )

      logging.info(f'-->appending to overlap: '
                     f'{np.nanmin(each_ds[ds_var].values)=} '
                     f'{np.nanmax(each_ds[ds_var].values)=}')
      data_list.append(each_ds[ds_var])

      if len(data_list) > 1:
         # Concatenate once we have 2 arrays
         concatenated = xr.concat(data_list, CONCAT_DIM_NAME, join="outer")
         data_list = [concatenated.min(CONCAT_DIM_NAME, skipna=True, keep_attrs=True)]

      gc.collect()

   max_overlap = data_list[0]

   # 1a. Take minimum of all overlapping cells
   logging.info(f'Adding minimum for {ds_var} to dataset')
   # max_overlap = concatenated.min(CONCAT_DIM_NAME, skipna=True, keep_attrs=True)

   # Set data values in result dataset
   max_overlap_dims = dict(x=max_overlap.x.values, y=max_overlap.y.values)

   # Add min v_error for the output dataset
   ds[ds_var].loc[max_overlap_dims] = max_overlap

   del max_overlap
   gc.collect()

   # Concatenate data for each data variable:
   # 2. Merge all other data variables based on the v_error mask: pick
   # values from the mosaics which match v_error value
   _first = '_first'
   _second = '_second'

   two_coords = [y_coords, x_coords]
   two_dims = [Coords.Y, Coords.X]
   two_dims_len = (len(y_coords), len(x_coords))

   # Name of the variable that should be used for masking of the rest of the variables
   mask_var = 'v_error'

   dr_to_vr_factor = []

   for each_var in ['M11', 'M12', 'vr', 'vx', 'vy', 'v', dr_to_vr_factor_var]:
      data_list = []
      mask_list = []

      # Concatenated dataset
      concatenated = None

      for each_file, each_ds in raw_ds.items():
         logging.info(f'Merging {each_var=} from {each_file}')

         if each_var not in ds:
            # Add variable to the result dataset:
            # all vars are 2-d data
            ds[each_var] = xr.DataArray(
               data=np.full(two_dims_len, np.nan),
               coords=two_coords,
               dims=two_dims,
               attrs=each_ds[each_var].attrs
            )

         # Collect dr_to_vr_factor from M11
         if each_var == 'M11':
            dr_to_vr_factor.append(each_ds.M11.attrs['dr_to_vr_factor'])

         data_list.append(each_ds[each_var])
         # Have to match v_error values in mosaics mask which is constructed
         # of min v_error values in overlapping areas
         mask_list.append(each_ds[mask_var])

         if len(data_list) > 1:
            # Merge once we have 2 mosaics
            first_var = f'{_first}{each_var}'
            second_var = f'{_second}{each_var}'
            first_mask = f'{_first}{mask_var}'
            second_mask = f'{_second}{mask_var}'

            # Have to insert into dataset to make sure all coordinates are
            # equal before masking against min v_error
            # ATTN: ds[mask_var] is the mask
            ds[first_var] = data_list[0]
            ds[second_var] = data_list[1]

            ds[first_mask] = mask_list[0]
            ds[second_mask] = mask_list[1]

            # Pick values that correspond to the minimum v_error values
            concatenated = ds[second_var].where(ds[mask_var] ==
                              ds[second_mask], other=ds[first_var])

            # Delete temporary datasets
            del ds[first_var]
            del ds[second_var]
            del ds[first_mask]
            del ds[second_mask]
            gc.collect()

            data_list = [concatenated]

            # Update min v_error for the "concatenated" to be compared with
            # next granule to merge
            verror_concatenated = xr.concat(mask_list,
                                    CONCAT_DIM_NAME,
                                    join="outer")
            mask_list = [verror_concatenated.min(
                              CONCAT_DIM_NAME,
                              skipna=True
                           )
                        ]
            del verror_concatenated
            gc.collect()

         # Done with variable in the granule dataset, free up memory
         del each_ds[each_var]
         gc.collect()

      if concatenated is not None:
         logging.info(f'Done merging values for {each_var} based on '
                        'min v_error mask')

      # Set data values in result dataset
      overlap_dims = dict(x=concatenated.x.values, y=concatenated.y.values)

      # Set values for the output dataset
      ds[each_var].loc[overlap_dims] = concatenated

      gc.collect()

   # Save mosaics to the file
   filename = f'{orbit_dir}_mosaics.nc'

   # Set chunking for 2D data variables
   dims = ds.dims
   num_x = dims[Coords.X]
   num_y = dims[Coords.Y]

   # Compute chunking like AutoRIFT does:
   # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
   chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
   two_dim_chunks_settings = (chunk_lines, num_x)

   granule_encoding = SENTINEL1_ENCODING.copy()

   for each_var, each_var_settings in granule_encoding.items():
      if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
         each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

   ds.to_netcdf(filename, engine=NC_ENGINE, encoding=granule_encoding)

   return np.mean(scalar_factors), ds


if __name__ == '__main__':
   # Initialize CL arguments parser
   parser = argparse.ArgumentParser(
      description='Generate range-range '
                  'velocity mosaics for the given region and time interval.'
   )

   parser.add_argument(
      '-p', '--polygonGeoJSON',
      type=str,
      help='Path to the input Geojson file storing region polygon.'
   )
   parser.add_argument(
      '-a', '--ascendingNetCDF',
      type=str,
      default=None,
      help='Path to the input file storing ascending granules mosaics [%(default)s].'
   )
   parser.add_argument(
      '-d', '--descendingNetCDF',
      type=str,
      default=None,
      help='Path to the input file storing descending granules mosaics [%(default)s].'
   )
   parser.add_argument(
      '-f', '--ascendingFactor',
      type=float,
      default=None,
      help='Ascending dr_to_vr_factor to use for range-range computations [%(default)s].'
         ' (the value should be used only if ascendingNetCDF is provided).'
   )
   parser.add_argument(
      '--useFactorRaster',
      action='store_true',
      default=False,
      help='Use dr_to_vr_factor raster as stored in the ascending/descending'
            'mosaics for the range-range computations. [%(default)s]')
   parser.add_argument(
      '--startDate',
      type=lambda s: parse(s),
      help='Start date for search API query to get velocity pair '
            'granules [%(default)s]'
   )
   parser.add_argument(
      '--numDays',
      type=int,
      default=12,
      help='Number of days for the granule search interval [%(default)s]'
   )
   parser.add_argument(
      '--epsgString',
      type=str,
      default='EPSG:3413',
      help='EPSG for granules [%(default)s]'
   )
   parser.add_argument(
      '-chunk_size',
      action='store',
      type=int,
      default=10,
      help='Number of granules to process in parallel [%(default)d]'
   )
   parser.add_argument(
      '-w', '--dask-workers',
      type=int,
      default=8,
      help='Number of Dask parallel workers [%(default)d]'
   )

   args = parser.parse_args()

   # Load polygon
   coordinates = load_polygon(args.polygonGeoJSON)
   logging.info(f'Got {len(coordinates)} coordinates from {args.polygonGeoJSON}')

   if len(coordinates) == 0:
      logging.error(f'No coordinates found in {args.polygonGeoJSON}')
      sys.exit(1)

   # Query for all ascending granules
   all_granules = {}

   start_date = args.startDate
   end_date = start_date + datetime.timedelta(days=12)
   logging.info(f'Got {start_date=} {end_date=}')

   logging.info(f'Querying for ascending granules between {start_date} and '
                  f'{end_date}')

   roi = {
      "type": "Polygon",
      "coordinates": [coordinates]
   }

   date_str = f"{start_date.strftime(DATE_FORMAT)}/{end_date.strftime(DATE_FORMAT)}"
   logging.info(f'Got {date_str=}')

   # Full list of properties can be found by inspecting a STAC item from the collection
   # e.g. https://stac.itslive.cloud/collections/itslive-granules/items/S2B_MSIL1C_20250410T133729_N0511_R067_T33XXK_20250410T171508_X_S2A_MSIL1C_20250417T133731_N0511_R067_T33XXK_20250417T205237_G0120V02_P054
   items = {}
   for each_orbit_direction in [ascending, descending]:
      # Format filters for the STAC query
      filters = [
         {"op": ">=", "args": [{"property": "percent_valid_pixels"}, 1]},
         {"op": "=", "args": [{"property": "proj:code"}, args.epsgString]},
         {"op": "=", "args": [{"property": "sat:orbit_state"}, each_orbit_direction]},
      ]

      items[each_orbit_direction] = search_stac(
                                       datetime=date_str,
                                       intersects=roi,
                                       filter_list=filters,
                                       page_size=2000
                                    )

   # Report number of found granules
   logging.info(f'Got {len(items[ascending])=}')
   logging.info(f'Got {len(items[descending])=}')

   # Create mosaics for ascending and descending granules
   s3 = s3fs.S3FileSystem(anon=False, skip_instance_cache=True)

   # This is to save on interrupted workflow (due to out of memory issues)
   # to load already created ascending granules mosaics
   asc_ds = None
   asc_factor = None

   var_list = ['vx', 'vy', 'v', 'M11', 'M12', 'vr', 'dr_to_vr_factor', 'mapping']

   if args.ascendingNetCDF:
      # Use provided ascending granules
      logging.info(f'Using provided ascending granules {args.ascendingNetCDF}')
      with xr.open_dataset(args.ascendingNetCDF, engine=NC_ENGINE) as ids:
         asc_ds = ids[var_list].load()
         logging.info(f'Got {list(asc_ds.keys())} variables from dataset.')

      if args.ascendingFactor:
         asc_factor = args.ascendingFactor
         logging.info(f'Got {asc_factor=}')

      else:
         # Compute average dr_to_vr_factor from the raster
         asc_factor = np.nanmean(asc_ds['dr_to_vr_factor'].values)
         logging.info(f'Computed {asc_factor=} from dr_to_vr_factor raster')

   else:
      # Create ascending granules mosaics
      asc_factor, asc_ds = build_mosaics(items[ascending], ascending)

   if args.descendingNetCDF:
      # Use provided descending granules
      logging.info(f'Using provided descending granules {args.descendingNetCDF}')
      with xr.open_dataset(args.descendingNetCDF, engine=NC_ENGINE) as ids:
         des_ds = ids[var_list].load()
         logging.info(f'Got {list(des_ds.keys())} variables from dataset.')

      if 'dr_to_vr_factor' in des_ds:
         des_factor = np.nanmean(des_ds['dr_to_vr_factor'].values)
         logging.info(f'Got {des_factor=} from dr_to_vr_factor raster')

      else:
         # Compute average dr_to_vr_factor from the raster
         des_factor = np.nanmean(des_ds['dr_to_vr_factor'].values)
         logging.info(f'Computed {des_factor=} from dr_to_vr_factor raster')

   else:
      # Create descending granules mosaics
      des_factor, des_ds = build_mosaics(items[descending], descending)

   # Build range-range mosaics based on ascending and descending mosaics
   d_ds_m11 = des_ds['M11'].values
   d_ds_m12 = des_ds['M12'].values
   d_ds_vr = des_ds['vr'].values

   logging.info(f'MIN/MAX vr_descending: {np.nanmin(d_ds_vr)}, {np.nanmax(d_ds_vr)}')
   d_dr_to_vr_factor = des_factor

   d_trans = np.array(str.split(des_ds['mapping'].GeoTransform)).astype(float)

   a_ds_m11 = asc_ds['M11'].values
   a_ds_m12 = asc_ds['M12'].values
   a_ds_vr = asc_ds['vr'].values
   logging.info(f'MIN/MAX vr_ascending: {np.nanmin(a_ds_vr)}, {np.nanmax(a_ds_vr)}')

   a_dr_to_vr_factor = asc_factor
   a_trans = np.array(str.split(asc_ds['mapping'].GeoTransform)).astype(float)

   if args.useFactorRaster:
      # Use dr_to_vr_factor raster from the descending granules
      a_dr_to_vr_factor = asc_ds['dr_to_vr_factor'].values
      d_dr_to_vr_factor = des_ds['dr_to_vr_factor'].values
      logging.info(f'Using dr_to_vr_factor rasters.')

   vx, vy, x_size, y_size, trans = compute_velocity_based_on_m11_m12(
      a_ds_m11, a_ds_m12,
      d_ds_m11, d_ds_m12,
      a_ds_vr, d_ds_vr,
      a_dr_to_vr_factor, d_dr_to_vr_factor,
      a_trans, d_trans
   )

   x_min = trans[0] + half_x_cell
   y_max = trans[3] + half_y_cell

   x_max = x_min + x_cell * x_size
   y_min = y_max + y_cell * y_size

   x_bounds = grid.Bounds(min_value=x_min, max_value=x_max)
   y_bounds = grid.Bounds(min_value=y_min, max_value=y_max)

   # Create grid for new granule
   grid_x, grid_y = grid.Grid.create(x_bounds, y_bounds, x_cell)

   rr_mosaic = create_new_granule(grid_x, grid_y)

   rr_mosaic[DataVars.VX] = xr.DataArray(
      vx,
      coords=rr_mosaic.coords,
      attrs=des_ds.vx.attrs
   )
   rr_mosaic[DataVars.VY] = xr.DataArray(
      vy,
      coords=rr_mosaic.coords,
      attrs=des_ds.vy.attrs
   )
   rr_mosaic['mapping'] = asc_ds.mapping

   # Add "v" data variable
   v = np.sqrt(vx**2 + vy**2)
   # There are no numeric values for any of the 'v' attributes, just copy from input granule
   rr_mosaic[DataVars.V] = xr.DataArray(
      data=v,
      coords=rr_mosaic.coords,
      attrs=des_ds.v.attrs
   )

   # Crop to valid extends
   xy_ds = rr_mosaic.where(rr_mosaic.vx.notnull(), drop=True)

   x_values = xy_ds.x.values
   grid_x_min, grid_x_max = x_values.min(), x_values.max()

   y_values = xy_ds.y.values
   grid_y_min, grid_y_max = y_values.min(), y_values.max()

   # Based on X/Y extends, mask original dataset
   mask_lon = (rr_mosaic.x >= grid_x_min) & (rr_mosaic.x <= grid_x_max)
   mask_lat = (rr_mosaic.y >= grid_y_min) & (rr_mosaic.y <= grid_y_max)
   mask = (mask_lon & mask_lat)

   cropped_ds = rr_mosaic.where(mask, drop=True)

   # Recalculate the mapping for the cropped dataset
   x_values = cropped_ds.x.values
   y_values = cropped_ds.y.values

   # Update mapping.GeoTransform
   x_cell = x_values[1] - x_values[0]
   y_cell = y_values[1] - y_values[0]

   # It was decided to keep all values in GeoTransform center-based
   cropped_ds[DataVars.MAPPING].attrs['GeoTransform'] = f"{x_values[0]} {x_cell} 0 {y_values[0]} 0 {y_cell}"

   # Save range-range mosaic to the file
   filename = 'rr_mosaics.nc'
   logging.info(f'Saving range-range mosaic to {filename}')

   # Set chunking for 2D data variables
   dims = cropped_ds.dims
   num_x = dims[Coords.X]
   num_y = dims[Coords.Y]

   # Compute chunking like AutoRIFT does:
   # https://github.com/ASFHyP3/hyp3-autorift/blob/develop/hyp3_autorift/vend/netcdf_output.py#L410-L411
   chunk_lines = np.min([np.ceil(8192/num_y)*128, num_y])
   two_dim_chunks_settings = (chunk_lines, num_x)

   for each_var, each_var_settings in RANGE_RANGE_ENCODING.items():
      if each_var_settings[Output.FILL_VALUE_ATTR] is not None:
         each_var_settings[Output.CHUNKSIZES_ATTR] = two_dim_chunks_settings

   cropped_ds.to_netcdf(filename, engine=NC_ENGINE, encoding=RANGE_RANGE_ENCODING)

   logging.info('Done.')
