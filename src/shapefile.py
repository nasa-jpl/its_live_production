"""
Shapefile processing utilities for ITS_LIVE data.

This module provides functions to read and process shapefiles containing ice
mask information, which are used in the SensorExcludeFilter for filtering
ITS_LIVE data based on land ice coverage.

Note: Heavy geospatial dependencies (geopandas, rioxarray, xarray) are
imported lazily to allow lightweight imports of constants.
"""
import logging
import numpy as np

import itslive_utils
import utils

# Set up logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)

# Variables names specific to the ITS_LIVE shapefiles.
EPSG = 'epsg'
LANDICE_2KM = 'landice_2km'
LANDICE = 'landice'
FLOATINGICE = 'floatingice'

Name = {
   LANDICE: 'land_ice_mask',
   FLOATINGICE: 'floating_ice_mask',
}
Description = {
   LANDICE: 'land ice mask, 0 = non-land-ice, 1 = land-ice',
   FLOATINGICE: 'floating ice mask, 0 = non-floating-ice, 1 = floating-ice',
}
Type = {
   LANDICE: np.uint8,
   FLOATINGICE: np.uint8,
}
MissingValue = {
   LANDICE: utils.Missing.u8value,
   FLOATINGICE: utils.Missing.u8value,
}

@itslive_utils.retry_decorator()
def read_file(shapeFile: str):
   """
   Read shape file in with ice masks information required for processing.

   Inputs:
   =======
   shapeFile: URL to the shapefile.

   Returns:
   ========
   Object representing the shapefile.
   """
   import geopandas as gpd

   # Make sure it's S3 URL that is provided
   shape_file = shapeFile.replace(utils.HTTP_PREFIX, utils.S3_PREFIX)
   shape_file = shape_file.replace(utils.PATH_URL, '')
   return gpd.read_file(shape_file)


@itslive_utils.retry_decorator()
def read_ice_mask(shapefile_gdp, mask_name, grid_x, grid_y, projection):
   """
   Read ice mask as stored in "column_name" field of the shapefile's row.

   Inputs:
   =======
   shapefile_gdp:  Geopandas object that represents shape file contents.
   mask_name:     Name of the mask to be read, e.g. "landice_2km" or
                  "floatingice".
   grid_x:        X coordinates of the datacube grid.
   grid_y:        Y coordinates of the datacube grid.
   projection:    Projection of the datacube, used to find the right row in
                  the shapefile.

   Returns:
   A tuple of: None if there is no overlap between ice mask and datacube
               polygon, or ice mask cropped to the provided grid.
               URL to the mask file as provided in the shapefile.
   """
   import rioxarray
   import xarray as xr

   row = shapefile_gdp.loc[shapefile_gdp[EPSG] == int(projection)]
   if len(row) != 1:
      raise RuntimeError(f'Expected one entry for EPSG {projection} in '
                           f'shapefile, got {len(row)} rows.')

   ice_mask_file = row[mask_name].item()

   ice_mask_file = ice_mask_file.replace(utils.HTTP_PREFIX, utils.S3_PREFIX)
   ice_mask_file = ice_mask_file.replace(utils.PATH_URL, '')
   logging.info(f'Using {mask_name} mask file {ice_mask_file}')

   # Load the mask
   mask_ds = rioxarray.open_rasterio(ice_mask_file)

   # Zoom into cube polygon
   mask_x = (mask_ds.x >= grid_x.min()) & (mask_ds.x <= grid_x.max())
   mask_y = (mask_ds.y >= grid_y.min()) & (mask_ds.y <= grid_y.max())
   mask = (mask_x & mask_y)

   # Allocate xr.DataArray to match cube dimentions: will be empty if
   # no overlap exists with the ice mask, or will be set to overlap with
   # ice mask
   ice_mask = xr.DataArray(
      np.zeros((len(grid_y), len(grid_x))),
      coords={
         utils.Coords.X: grid_x,
         utils.Coords.Y: grid_y
      },
      dims=[utils.Coords.Y, utils.Coords.X]
   )

   if mask.sum().item() == 0:
      # Mask does not overlap with the cube
      logging.info(
            f'No overlap is detected with {mask_name} mask data '
            f'{ice_mask_file}'
      )

   else:
      cropped_mask_ds = mask_ds.where(mask, drop=True)

      # Populate mask data into cube-size array
      if cropped_mask_ds.ndim == 3:
         # If it's 3d data, it should have first dimension=1: just
         # one layer is expected
         mask_data_sizes = cropped_mask_ds.shape
         if mask_data_sizes[0] != 1:
            raise RuntimeError(
               f'Unexpected size for {mask_name} mask data from '
               f'{ice_mask_file} file: {mask_data_sizes}'
            )

         else:
            ice_mask.loc[
               dict(x=cropped_mask_ds.x, y=cropped_mask_ds.y)
            ] = cropped_mask_ds[0]

      else:
         ice_mask.loc[dict(x=cropped_mask_ds.x, y=cropped_mask_ds.y)] = \
            cropped_mask_ds

   # Store mask as numpy array since all calcuations are done using
   # numpy arrays
   ice = ice_mask.values
   land_ice_coverage = int(np.sum(ice))/(len(grid_x)*len(grid_y))*100
   logging.info(
      f'Got {mask_name} mask for {np.round(land_ice_coverage, 2)}% '
      f'cells of the datacube'
   )

   return (ice, ice_mask_file)
