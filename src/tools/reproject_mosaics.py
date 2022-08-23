"""
Reprojection tool for ITS_LIVE mosaics to new target projection.

Examples:
$ python reproject_mosaics.py -i input_filename -p target_projection -o output_filename

    Reproject "input_filename" into 'target_projection' and output new mosaic into
'output_filename' in NetCDF format.

$ python ./reproject_mosaics.py -i  ITS_LIVE_velocity_120m_HMA_2015_v02.nc -o reproject_ITS_LIVE_velocity_120m_HMA_2015_v02.nc -p 102027

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Yang Lei (Caltech)
"""
import argparse
from datetime import datetime
import gc
import logging
import math
import numpy as np
import os
from osgeo import osr, gdalnumeric, gdal
from tqdm import tqdm
import xarray as xr

from grid import Grid, Bounds
from itscube_types import Coords, DataVars
from itslive_composite import CompDataVars

# GDAL: enable exceptions and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

spatial_ref_3031 = "PROJCS[\"WGS 84 / Antarctic Polar Stereographic\"," \
    "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\"," \
    "6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]]," \
    "AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0," \
    "AUTHORITY[\"EPSG\",\"8901\"]],UNIT[“degree\",0.0174532925199433," \
    "AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]," \
    "PROJECTION[\"Polar_Stereographic\"],PARAMETER[\"latitude_of_origin\",-71]," \
    "PARAMETER[\"central_meridian\",0],PARAMETER[\"false_easting\",0]," \
    "PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]" \
    ",AXIS[\"Easting\",NORTH],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"3031\"]]"

spatial_ref_3413 = "PROJCS[\"WGS 84 / NSIDC Sea Ice Polar Stereographic North\"," \
    "\"GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563," \
    "AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]]" \
    ",PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]]," \
    "UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]]," \
    "AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Polar_Stereographic\"]," \
    "PARAMETER[\"latitude_of_origin\",70],PARAMETER[\"central_meridian\",-45]," \
    "PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0]," \
    "UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",SOUTH]," \
    "AXIS[\"Northing\",SOUTH],AUTHORITY[\"EPSG\",\"3413\"]]"

spatial_ref_102027 = "PROJCS[\"Asia_North_Lambert_Conformal_Conic\",GEOGCS[\"GCS_WGS_1984\"," \
    "DATUM[\"WGS_1984\",SPHEROID[\"WGS_1984\",6378137,298.257223563]]," \
    "PRIMEM[\"Greenwich\",0],UNIT[\"Degree\",0.017453292519943295]]," \
    "PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"False_Easting\",0]," \
    "PARAMETER[\"False_Northing\",0],PARAMETER[\"Central_Meridian\",95]," \
    "PARAMETER[\"Standard_Parallel_1\",15],PARAMETER[\"Standard_Parallel_2\",65]," \
    "PARAMETER[\"Latitude_Of_Origin\",30],UNIT[\"Meter\",1],AUTHORITY[\"ESRI\",\"102027\"]]"

PROJECTION_ATTR = 'projection'

# Non-EPSG projection that can be provided on output
ESRICode = 102027

# last: ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
# ESRICode_Proj4 = '+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

class MosaicsReproject:
    """
    Class to re-project static and annual ITS_LIVE mosaics into a new target projection.

    The following steps must be taken to re-project ITS_LIVE data to the new
    projection:

    1. Compute bounding box for input granule in original P_in projection ("ij" naming convention)
    2. Re-project P_in bounding box to P_out projection ("xy" naming convention)
    3. Compute grid in P_out projection based on its bounding bbox: (x0, y0)
    4. Project each cell center in P_out grid to original P_in projection: (i0, j0)
    5. Add unit length (240m) to x of (i0, j0) and project to P_out: (x1, y1),
       compute x_unit vector based on (x0, y0) and (x1, y1)
    6. Add unit length (240m) to y of (i0, j0) and project to P_out: (x2, y2),
       compute y_unit vector based on (x0, y0) and (x2, y2)
    7. In Geogrid code, set normal = (0, 0, 1)
    8. Compute transformation matrix using Geogrid equations amd unit vectors
       x_unit and y_unit per each cell of grid in output projection
    9. Re-project v* values: gdal.warp(original_granule, P_out_grid) --> P_out_v*
       Apply tranformation matrix to P_out_v* per cell to get "true" v value in
       output projection
    """
    # Number of years: the same time period used to convert v(elocity) component
    # to corresponding d(isplacement), and use the same time period in
    # transformation matrix computations.
    # Set to one year.
    TIME_DELTA = 1

    V_ERROR_ATTRS = {
        DataVars.STD_NAME:         DataVars.NAME[DataVars.V_ERROR],
        DataVars.DESCRIPTION_ATTR: DataVars.DESCRIPTION[DataVars.V_ERROR],
        DataVars.UNITS:            DataVars.M_Y_UNITS
    }

    # Have to provide error threshold for gdal.warp to avoid large values
    # being introduced by warping.
    WARP_ET = 1e-5

    # Use virtual memory format to avoid writing warped dataset to the file
    WARP_FORMAT = 'vrt'

    # Flag to enable verbose reporting
    VERBOSE = False

    # Flag to enable computation of debug/validation variables (map distortion
    # in X and Y dimensions, v_error)
    COMPUTE_DEBUG_VARS = False

    # Filename to store transformation matrix as numpy array to npy binary file.
    # Large regions, such as HMA, contain over 93M points and it takes ~3.5 hours
    # to build the transformation matrix.
    # Since all mosaics use the same transformation matrix, build it once.
    TRANSFORMATION_MATRIX_FILE = None

    def __init__(self, data, output_projection: int):
        """
        Initialize object.
        """
        self.ds = data
        self.input_file = None
        if isinstance(data, str):
            # Filename for the dataset is provided, read it in
            self.input_file = data
            self.ds = xr.open_dataset(data, mask_and_scale=False)
            self.ds.load()

            logging.info(f'Grid in P_in: num_x={len(self.ds.x)} num_y={len(self.ds.y)}')

        # Input and output projections
        self.ij_epsg = int(self.ds.mapping.spatial_epsg)
        self.xy_epsg = output_projection

        # Initialize
        self.ij_epsg_str = 'EPSG'
        self.xy_epsg_str = 'EPSG'
        # Google search says GDAL and QGIS use EPSG
        if self.xy_epsg == 102027:
            self.xy_epsg_str = 'ESRI'

        self.reproject = True
        if self.ij_epsg == self.xy_epsg:
            logging.info("Done: original data is in the target {self.xy_epsg} projection already.")
            self.reproject = False

        else:
            logging.info(f"Reprojecting from {self.ij_epsg} to {self.xy_epsg}")

            # Grid spacing
            self.x_size = self.ds.x.values[1] - self.ds.x.values[0]
            self.y_size = self.ds.y.values[1] - self.ds.y.values[0]

            self.i_limits = Bounds(self.ds.x.values)
            self.j_limits = Bounds(self.ds.y.values)
            logging.info(f"P_in: x: {self.i_limits} y: {self.j_limits}")

            # Placeholders for:
            # bounding box in output projection
            self.x0_bbox = None
            self.y0_bbox = None

            # grid coordinates in output projection
            self.x0_grid = None
            self.y0_grid = None

            # Sensor dimension if present (static mosaics only)
            self.sensors = None
            if CompDataVars.SENSORS in self.ds:
                self.sensors = self.ds[CompDataVars.SENSORS].values

            self.xy_central_meridian = None

            # Indices for original cells that correspond to the re-projected cells:
            # to find corresponding values
            self.original_ij_index = None

            # Transformation matrix to rotate warped velocity components (vx* and vy*)
            # in output projection taking distortion factor into consideration
            self.transformation_matrix = None

            # GDAL options to use for warping to new output grid
            self.warp_options = None

            # A "pointer" to the method to invoke to process the mosaic: static or annual
            self.mosaic_function = None

    def bounding_box(self):
        """
        Identify bounding box for original dataset.
        """
        # ATTN: Assuming that X and Y cell spacings are the same
        assert np.abs(self.x_size) == np.abs(self.y_size), \
            f"Cell dimensions differ: x={np.abs(self.x_size)} y={np.abs(self.y_size)}"

        center_off_X = self.x_size/2
        center_off_Y = self.y_size/2

        # Compute cell boundaries as ITS_LIVE grid stores x/y for the cell centers
        xmin = self.i_limits.min - center_off_X
        xmax = self.i_limits.max + center_off_X

        # Y coordinate calculations are based on the fact that dy < 0
        ymin = self.j_limits.min + center_off_Y
        ymax = self.j_limits.max - center_off_Y

        return Grid.bounding_box(
            Bounds(min_value=xmin, max_value=xmax),
            Bounds(min_value=ymin, max_value=ymax),
            self.x_size
        )

    def __call__(self, output_file: str = None):
        """
        Run reprojection of ITS_LIVE mosaic into target projection.

        This methods warps X and Y components of v* velocities, and
        adjusts them by rotation for new projection.
        """
        if not self.reproject:
            logging.info(f'Nothing to do.')

        # Flag if v0 is present in the mosaic, which indicates it's static mosaic
        is_static_mosaic = (CompDataVars.V0 in self.ds)
        if is_static_mosaic:
            self.create_transformation_matrix(CompDataVars.VX0, CompDataVars.VY0, CompDataVars.V0)
            self.mosaic_function = self.reproject_static_mosaic

        else:
            self.create_transformation_matrix(DataVars.VX, DataVars.VY, DataVars.V)
            self.mosaic_function = self.reproject_annual_mosaic

        # outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS

        # MISSING_BYTE      = 0.0
        # MISSING_UBYTE     = 0.0
        # MISSING_VALUE     = -32767.0
        # MISSING_POS_VALUE = 32767.0
        self.warp_options = gdal.WarpOptions(
            # format='netCDF',
            format=MosaicsReproject.WARP_FORMAT,
            outputBounds=(self.x0_bbox.min, self.y0_bbox.min, self.x0_bbox.max, self.y0_bbox.max),
            xRes=self.x_size,
            yRes=self.y_size,
            srcSRS=f'{self.ij_epsg_str}:{self.ij_epsg}',
            dstSRS=f'{self.xy_epsg_str}:{self.xy_epsg}',
            srcNodata=DataVars.MISSING_BYTE,
            dstNodata=DataVars.MISSING_BYTE,
            resampleAlg=gdal.GRA_NearestNeighbour,
            errorThreshold=MosaicsReproject.WARP_ET
        )

        self.mosaic_function(output_file)

    def set_mapping(self, reproject_ds):
        """
        Set mapping data variable and related attributes for the result dataset.

        Inputs:
        =======
        reproject_ds - Dataset to set mapping for.
        """
        # Change projection attribute of re-projected mosaic:
        reproject_ds.attrs[PROJECTION_ATTR] = self.xy_epsg

        # Add projection data variable
        proj_attrs = None
        if self.xy_epsg == 3031:
            proj_attrs={
                DataVars.GRID_MAPPING_NAME: 'polar_stereographic',
                'straight_vertical_longitude_from_pole': 0,
                'latitude_of_projection_origin': -90.0,
                'latitude_of_origin': -71.0,
                'scale_factor_at_projection_origin': 1,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'semi_major_axis': 6378.137,
                'semi_minor_axis': 6356.752,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_3031,
                'spatial_proj4': "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            }

        elif self.xy_epsg == 3413:
            proj_attrs={
                DataVars.GRID_MAPPING_NAME: 'polar_stereographic',
                'straight_vertical_longitude_from_pole': -45,
                'latitude_of_projection_origin': 90.0,
                'latitude_of_origin': 70.0,
                'scale_factor_at_projection_origin': 1,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'semi_major_axis': 6378.137,
                'semi_minor_axis': 6356.752,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_3413,
                'spatial_proj4': "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            }

        elif self.xy_epsg == 102027:
            proj_attrs={
                DataVars.GRID_MAPPING_NAME: 'lambert_conformal_conic',
                'CoordinateTransformType': 'Projection',
                'standard_parallel': (15.0, 65.0),
                'latitude_of_projection_origin': 30.0,
                'longitude_of_central_meridian': 95.0,
                'CoordinateAxisTypes': 'GeoX GeoY',
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_102027,
                'spatial_proj4': ESRICode_Proj4
            }

        else:
            zone, spacial_ref_value = self.spatial_ref_32x()

            proj_attrs={
                DataVars.GRID_MAPPING_NAME: 'universal_transverse_mercator',
                'utm_zone_number': zone,
                'semi_major_axis': 6378137,
                'inverse_flattening': 298.257223563,
                'CoordinateTransformType': 'Projection',
                'CoordinateAxisTypes': 'GeoX GeoY',
                'crs_wkt': spacial_ref_value,
                'spatial_proj4': f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            }

        reproject_ds[DataVars.MAPPING] = xr.DataArray(
            data='',
            coords={},
            dims=[],
            attrs=proj_attrs
        )

        # if self.xy_epsg == 102027:
        #     reproject_ds[DataVars.MAPPING].attrs['spatial_esri'] = self.xy_epsg
        #
        # else:
        #    reproject_ds[DataVars.MAPPING].attrs['spatial_epsg'] = self.xy_epsg
        reproject_ds[DataVars.MAPPING].attrs['spatial_epsg'] = self.xy_epsg

        # Format GeoTransform attribute:
        # x top left (cell left most boundary), grid size, 0, y top left (cell upper most boundary), 0, -grid size
        half_x_cell = self.x_size/2.0
        half_y_cell = self.y_size/2.0
        reproject_ds[DataVars.MAPPING].attrs['GeoTransform'] = f"{self.x0_grid[0] - half_x_cell} {self.x_size} 0 {self.y0_grid[0] - half_y_cell} 0 {self.y_size}"

    def reproject_static_mosaic(self, output_file: str):
        """
        Reproject static mosaic to new projection.

        output_file: Output file to write reprojected data to.
        """
        # Compute v0 and v0_error and their X and Y components
        vx0, vy0, v0, vx0_error, vy0_error, v0_error = self.reproject_velocity(
            CompDataVars.VX0,
            CompDataVars.VY0,
            CompDataVars.V0,
            CompDataVars.VX0_ERROR,
            CompDataVars.VY0_ERROR,
            CompDataVars.V0_ERROR
        )

        # Create new granule in target projection
        ds_coords=[
            (CompDataVars.SENSORS, self.sensors, self.ds.sensor.attrs),
            (Coords.Y, self.y0_grid, self.ds.y.attrs),
            (Coords.X, self.x0_grid, self.ds.x.attrs),
        ]

        ds_coords_2d=[
            (Coords.Y, self.y0_grid, self.ds.y.attrs),
            (Coords.X, self.x0_grid, self.ds.x.attrs)
        ]

        reproject_ds = xr.Dataset(
            data_vars={
                CompDataVars.V0: xr.DataArray(
                    data=v0,
                    coords=ds_coords_2d,
                    attrs=self.ds[CompDataVars.V0].attrs
                )
            },
            coords={
                Coords.Y: (Coords.Y, self.y0_grid, self.ds[Coords.Y].attrs),
                Coords.X: (Coords.X, self.x0_grid, self.ds[Coords.X].attrs),
                CompDataVars.SENSORS: (CompDataVars.SENSORS, self.sensors, self.ds[CompDataVars.SENSORS].attrs)
            },
            attrs=self.ds.attrs
        )
        v0 = None
        gc.collect()

        reproject_ds[CompDataVars.V0_ERROR] = xr.DataArray(
            data=v0_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V0_ERROR].attrs
        )

        v0_error = None
        gc.collect()

        reproject_ds[CompDataVars.VX0_ERROR] = xr.DataArray(
            data=vx0_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX0_ERROR].attrs
        )

        vx0_error = None
        gc.collect()

        reproject_ds[CompDataVars.VY0_ERROR] = xr.DataArray(
            data=vy0_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY0_ERROR].attrs
        )

        vy0_error = None
        gc.collect()

        # Re-project variables that depend on direction of unit flow vector [vx0, vy0]

        # This is memory hungry function as it reads all the variables that
        # need to be re-projected, but it saves on access to transformation matrix -
        # should probably re-consider the approach.
        dvx_dt, \
        dvy_dt, \
        dv_dt, \
        vx_amp, \
        vy_amp, \
        v_amp, \
        vx_amp_error, \
        vy_amp_error, \
        v_amp_error, \
        vx_phase, \
        vy_phase, \
        v_phase = self.reproject_static_vars(vx0, vy0)

        # No more need for vx0 and vy0,
        reproject_ds[CompDataVars.VX0] = xr.DataArray(
            data=vx0,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX0].attrs
        )
        vx0 = None
        gc.collect()

        reproject_ds[CompDataVars.VY0] = xr.DataArray(
            data=vy0,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY0].attrs
        )
        vy0 = None
        gc.collect()

        reproject_ds[CompDataVars.SLOPE_VX] = xr.DataArray(
            data=dvx_dt,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.SLOPE_VX].attrs
        )
        dvx_dt = None
        gc.collect()

        reproject_ds[CompDataVars.SLOPE_VY] = xr.DataArray(
            data=dvy_dt,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.SLOPE_VY].attrs
        )
        dvy_dt = None
        gc.collect()

        reproject_ds[CompDataVars.SLOPE_V] = xr.DataArray(
            data=dv_dt,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.SLOPE_V].attrs
        )
        dv_dt = None
        gc.collect()

        reproject_ds[CompDataVars.VX_AMP] = xr.DataArray(
            data=vx_amp,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_AMP].attrs
        )
        vx_amp = None
        gc.collect()

        reproject_ds[CompDataVars.VY_AMP] = xr.DataArray(
            data=vy_amp,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_AMP].attrs
        )
        vy_amp = None
        gc.collect()

        reproject_ds[CompDataVars.V_AMP] = xr.DataArray(
            data=v_amp,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_AMP].attrs
        )
        v_amp = None
        gc.collect()

        reproject_ds[CompDataVars.VX_AMP_ERROR] = xr.DataArray(
            data=vx_amp_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_AMP_ERROR].attrs
        )
        vx_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.VY_AMP_ERROR] = xr.DataArray(
            data=vy_amp_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_AMP_ERROR].attrs
        )
        vy_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.V_AMP_ERROR] = xr.DataArray(
            data=v_amp_error,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_AMP_ERROR].attrs
        )
        v_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.VX_PHASE] = xr.DataArray(
            data=vx_phase,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_PHASE].attrs
        )
        vx_phase = None
        gc.collect()

        reproject_ds[CompDataVars.VY_PHASE] = xr.DataArray(
            data=vy_phase,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_PHASE].attrs
        )
        vy_phase = None
        gc.collect()

        reproject_ds[CompDataVars.V_PHASE] = xr.DataArray(
            data=v_phase,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_PHASE].attrs
        )
        v_phase = None
        gc.collect()

        # Warp "count0" variable
        reproject_ds[CompDataVars.COUNT0] = xr.DataArray(
            data=self.warp_var(CompDataVars.COUNT0, self.warp_options),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.COUNT0].attrs
        )

        self.set_mapping(reproject_ds)

        # Warp "dt_max" variable: per each sensor dimension
        warp_data = self.warp_var(CompDataVars.MAX_DT, self.warp_options)

        if warp_data.ndim == 2:
            # If warped data is 2d array
            _y_dim, _x_dim = warp_data.shape

            # Convert to 3d array as MAX_DT is 3d data (has sensor dimension)
            warp_data = warp_data.reshape((1, _y_dim, _x_dim))

        if MosaicsReproject.VERBOSE:
            _values = self.ds[CompDataVars.MAX_DT].values
            verbose_mask = np.isfinite(_values)
            logging.info(f"Original {CompDataVars.MAX_DT}:  min={np.nanmin(_values[verbose_mask])} max={np.nanmax(_values[verbose_mask])}")

            verbose_mask = np.isfinite(warp_data)
            logging.info(f"Warped {CompDataVars.MAX_DT}:  min={np.nanmin(warp_data[verbose_mask])} max={np.nanmax(warp_data[verbose_mask])}")

        reproject_ds[CompDataVars.MAX_DT] = xr.DataArray(
            data=warp_data,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.MAX_DT].attrs
        )

        # Warp "outlier_frac" variable: per each sensor dimension
        reproject_ds[CompDataVars.OUTLIER_FRAC] = xr.DataArray(
            data=self.warp_var(CompDataVars.OUTLIER_FRAC, self.warp_options),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.OUTLIER_FRAC].attrs
        )

        # Warp "sensor_flag" variable: per each sensor dimension
        if CompDataVars.SENSOR_INCLUDE in self.ds:
            # This is workaround for missing variable in original mosaics code
            # so can test the code with originally generated small test sets
            warp_data = self.warp_var(CompDataVars.SENSOR_INCLUDE, self.warp_options)

            if warp_data.ndim == 2:
                # If warped data is 2d array
                _y_dim, _x_dim = warp_data.shape

                # Convert to 3d array as MAX_DT is 3d data (has sensor dimension)
                warp_data = warp_data.reshape((1, _y_dim, _x_dim))

            reproject_ds[CompDataVars.SENSOR_INCLUDE] = xr.DataArray(
                data=warp_data,
                coords=ds_coords,
                attrs=self.ds[CompDataVars.SENSOR_INCLUDE].attrs
            )

            if MosaicsReproject.VERBOSE:
                _values = self.ds[CompDataVars.SENSOR_INCLUDE].values
                verbose_mask = np.isfinite(_values)
                logging.info(f"Original {CompDataVars.SENSOR_INCLUDE}:  min={np.nanmin(_values[verbose_mask])} max={np.nanmax(_values[verbose_mask])}")

                verbose_mask = np.isfinite(warp_data)
                logging.info(f"gdal.warp(): Original {CompDataVars.SENSOR_INCLUDE}:  min={np.nanmin(warp_data[verbose_mask])} max={np.nanmax(warp_data[verbose_mask])}")

        MosaicsReproject.write_static_to_netCDF(reproject_ds, output_file)

    def reproject_annual_mosaic(self, output_file):
        """
        Reproject annual mosaic to new projection.

        output_file: Output file to write reprojected data to.
        """
        # Compute new v, v_error and their components
        vx, vy, v, vx_error, vy_error, v_error = self.reproject_velocity(
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR
        )

        v_error_verify = None
        if MosaicsReproject.COMPUTE_DEBUG_VARS:
            # Compute re-projection verification for v_error:
            # ATTN: This is just a sanity check for scaled v_error: they should be the same
            valid_mask = np.where(
                (vx != DataVars.MISSING_VALUE) & (vy != DataVars.MISSING_VALUE) &
                (v != DataVars.MISSING_VALUE) & (v != 0)
            )

            v_error_verify = np.full_like(v_error, DataVars.MISSING_VALUE, dtype=np.float32)
            v_error_verify[valid_mask] = (vx_error[valid_mask]*np.abs(vx[valid_mask]) + vy_error[valid_mask]*np.abs(vy[valid_mask]))/v[valid_mask]

        # Create new granule in target projection
        ds_coords=[
            (Coords.Y, self.y0_grid, self.ds.y.attrs),
            (Coords.X, self.x0_grid, self.ds.x.attrs)
        ]

        reproject_ds = xr.Dataset(
            data_vars={
                DataVars.VX: xr.DataArray(
                    data=vx,
                    coords=ds_coords,
                    attrs=self.ds[DataVars.VX].attrs
                )
            },
            coords={
                Coords.Y: (Coords.Y, self.y0_grid, self.ds[Coords.Y].attrs),
                Coords.X: (Coords.X, self.x0_grid, self.ds[Coords.X].attrs),
            },
            attrs=self.ds.attrs
        )

        vx = None
        gc.collect()

        reproject_ds[DataVars.VY] = xr.DataArray(
            data=vy,
            coords=ds_coords,
            attrs=self.ds[DataVars.VY].attrs
        )

        vy = None
        gc.collect()

        reproject_ds[DataVars.V] = xr.DataArray(
            data=v,
            coords=ds_coords,
            attrs=self.ds[DataVars.V].attrs
        )

        v = None
        gc.collect()

        reproject_ds[CompDataVars.V_ERROR] = xr.DataArray(
            data=v_error,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.V_ERROR].attrs
        )

        v_error = None
        gc.collect()

        # Add vx_error to dataset
        reproject_ds[CompDataVars.VX_ERROR] = xr.DataArray(
            data=vx_error,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.VX_ERROR].attrs
        )

        vx_error = None
        gc.collect()

        # Add vy_error to dataset
        reproject_ds[CompDataVars.VY_ERROR] = xr.DataArray(
            data=vy_error,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.VY_ERROR].attrs
        )

        vy_error = None
        gc.collect()

        if MosaicsReproject.COMPUTE_DEBUG_VARS:
            # Add debug v_error to dataset just to compare to already computed v_error
            reproject_ds[CompDataVars.V_ERROR+'_verify'] = xr.DataArray(
                data=v_error_verify,
                coords=ds_coords,
                attrs=self.ds[CompDataVars.V_ERROR].attrs
            )

            v_error_verify = None
            gc.collect()

        self.set_mapping(reproject_ds)

        # Compute x and y distortion maps for the dataset if enabled
        if MosaicsReproject.COMPUTE_DEBUG_VARS:
            vx_xunit, vy_xunit, vx_yunit, vy_yunit = self.get_distortion_for_debugging(
                DataVars.VX,
                DataVars.VY
            )

            # Distortion in X direction
            reproject_ds['vx_xunit'] = xr.DataArray(
                data=vx_xunit,
                coords=ds_coords,
                attrs=self.ds[DataVars.VX].attrs
            )
            vx_xunit = None
            gc.collect()

            reproject_ds['vy_xunit'] = xr.DataArray(
                data=vy_xunit,
                coords=ds_coords,
                attrs=self.ds[DataVars.VY].attrs
            )
            vy_xunit = None
            gc.collect()

            # Distortion in Y direction
            reproject_ds['vx_yunit'] = xr.DataArray(
                data=vx_yunit,
                coords=ds_coords,
                attrs=self.ds[DataVars.VX].attrs
            )
            vx_yunit = None
            gc.collect()

            reproject_ds['vy_yunit'] = xr.DataArray(
                data=vy_yunit,
                coords=ds_coords,
                attrs=self.ds[DataVars.VY].attrs
            )
            vy_yunit = None
            gc.collect()

        # Warp "count" variable
        reproject_ds[CompDataVars.COUNT] = xr.DataArray(
            data=self.warp_var(CompDataVars.COUNT, self.warp_options),
            coords=ds_coords,
            attrs=self.ds[CompDataVars.COUNT].attrs
        )

        MosaicsReproject.write_annual_to_netCDF(reproject_ds, output_file)

    @staticmethod
    def write_annual_to_netCDF(ds, output_file: str):
        """
        Write dataset to the netCDF format file.
        """
        if output_file is None:
            # Output filename is not provided, don't write to the file
            return

        encoding_settings = {}
        compression = {"zlib": True, "complevel": 2, "shuffle": True}

        # Disable FillValue for coordinates
        for each in [Coords.X, Coords.Y]:
            encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: None}

        vars = [DataVars.V, DataVars.VX, DataVars.VY,
            CompDataVars.VX_ERROR, CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR]

        if MosaicsReproject.COMPUTE_DEBUG_VARS:
            # Handle debug variables, if any, automatically
            debug_vars = [CompDataVars.V_ERROR+'_verify', 'vx_xunit', 'vy_xunit', 'vx_yunit', 'vy_yunit']

            for each in debug_vars:
                if each in ds:
                    vars.append(each)

        # Explicitly set dtype for some variables
        for each in vars:
            encoding_settings[each] = {
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                'dtype': np.float32
            }
            encoding_settings[each].update(compression)

            if DataVars.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[DataVars.FILL_VALUE_ATTR]

        # Set encoding for 'count' data variable
        encoding_settings[CompDataVars.COUNT] = {
            DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
            'dtype': np.uint32
        }
        encoding_settings[CompDataVars.COUNT].update(compression)

        if DataVars.FILL_VALUE_ATTR in ds[CompDataVars.COUNT].attrs:
            del ds[CompDataVars.COUNT].attrs[DataVars.FILL_VALUE_ATTR]

        logging.info(f'Enconding for {output_file}: {encoding_settings}')

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding = encoding_settings)

    @staticmethod
    def write_static_to_netCDF(ds, output_file: str):
        """
        Write static mosaic dataset to the netCDF format file.
        """
        if output_file is None:
            # Output filename is not provided, don't write to the file
            return

        encoding_settings = {}
        compression = {"zlib": True, "complevel": 2, "shuffle": True}

        # Disable FillValue for coordinates
        for each in [Coords.X, Coords.Y, CompDataVars.SENSORS]:
            encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: None}

        vars = [
            CompDataVars.VX_AMP_ERROR,
            CompDataVars.VY_AMP_ERROR,
            CompDataVars.V_AMP_ERROR,
            CompDataVars.VX_AMP,
            CompDataVars.VY_AMP,
            CompDataVars.V_AMP,
            CompDataVars.VX_PHASE,
            CompDataVars.VY_PHASE,
            CompDataVars.V_PHASE,
            CompDataVars.OUTLIER_FRAC,
            CompDataVars.VX0,
            CompDataVars.VY0,
            CompDataVars.V0,
            CompDataVars.VX0_ERROR,
            CompDataVars.VY0_ERROR,
            CompDataVars.V0_ERROR,
            CompDataVars.SLOPE_VX,
            CompDataVars.SLOPE_VY,
            CompDataVars.SLOPE_V
        ]

        # Explicitly set dtype for some variables
        for each in vars:
            encoding_settings[each] = {
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                'dtype': np.float32
            }
            encoding_settings[each].update(compression)

            if DataVars.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[DataVars.FILL_VALUE_ATTR]

        # Set encoding for 'count0' data variable
        encoding_settings[CompDataVars.COUNT0] = {
            DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
            'dtype': np.uint32
        }
        encoding_settings[CompDataVars.COUNT0].update(compression)

        if DataVars.FILL_VALUE_ATTR in ds[CompDataVars.COUNT0].attrs:
            del ds[CompDataVars.COUNT0].attrs[DataVars.FILL_VALUE_ATTR]

        # Settings for "sensor_include" datatypes
        if CompDataVars.SENSOR_INCLUDE in ds:
            # This is workaround for missing variable in original mosaics code
            # so can test the code with originally generated small test sets
            encoding_settings.setdefault(CompDataVars.SENSOR_INCLUDE, {}).update({
                    'dtype': np.short
                })
            encoding_settings[CompDataVars.SENSOR_INCLUDE].update(compression)

            if DataVars.FILL_VALUE_ATTR in ds[CompDataVars.SENSOR_INCLUDE].attrs:
                del ds[CompDataVars.SENSOR_INCLUDE].attrs[DataVars.FILL_VALUE_ATTR]

        # Settings for "max_dt" datatypes
        encoding_settings[CompDataVars.MAX_DT] = {
            DataVars.FILL_VALUE_ATTR: DataVars.MISSING_POS_VALUE,
            'dtype': np.short
        }

        if DataVars.FILL_VALUE_ATTR in ds[CompDataVars.MAX_DT].attrs:
            del ds[CompDataVars.MAX_DT].attrs[DataVars.FILL_VALUE_ATTR]

        logging.info(f'Enconding for {output_file}: {encoding_settings}')

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding = encoding_settings)

    def warp_var(self, var: str, warp_options: gdal.WarpOptions):
        """
        Warp variable into new projection.
        """
        np_ds = gdal.Warp('', f'NETCDF:"{self.input_file}":{var}', options=warp_options).ReadAsArray()

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(np_ds)
            logging.info(f"Warped {var}:  min={np.nanmin(np_ds[verbose_mask])} max={np.nanmax(np_ds[verbose_mask])}")

        return np_ds

    def reproject_velocity(self,
        vx_var: str,
        vy_var: str,
        v_var: str,
        vx_error_var: str,
        vy_error_var: str,
        v_error_var: str,
    ):
        """
        Re-project variable's X and Y components, compute its magnitude and
        error if required.

        vx_var: name of the variable's X component
        vy_var: name of the variable's Y component
        v_var:  name of the variable
        vx_error_var: name of X component of error
        vy_error_var: name of Y component of error
        v_error_var: name of the error variable
        """
        # Read X component of variable
        _vx = self.ds[vx_var].values
        _vx[_vx==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of variable
        _vy = self.ds[vy_var].values
        _vy[_vy==DataVars.MISSING_VALUE] = np.nan

        # Read original velocity values
        _v = self.ds[v_var].values
        _v[_v==DataVars.MISSING_VALUE] = np.nan

        # Read original error values in
        _v_error = self.ds[v_error_var].values
        _v_error[_v_error==DataVars.MISSING_VALUE] = np.nan

        # Read X component of v_error
        _vx_error = self.ds[vx_error_var].values
        _vx_error[_vx_error==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of the error
        _vy_error = self.ds[vy_error_var].values
        _vy_error[_vy_error==DataVars.MISSING_VALUE] = np.nan

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        # Allocate output data
        vx = np.full((num_y, num_x), np.nan, dtype=np.float32)
        vy = np.full((num_y, num_x), np.nan, dtype=np.float32)
        v = np.full((num_y, num_x), np.nan, dtype=np.float32)

        v_error = np.full((num_y, num_x), np.nan, dtype=np.float32)
        vx_error = np.full((num_y, num_x), np.nan, dtype=np.float32)
        vy_error = np.full((num_y, num_x), np.nan, dtype=np.float32)

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(_vx)
            logging.info(f"reproject_velocity: Original {vx_var}:  min={np.nanmin(_vx[verbose_mask])} max={np.nanmax(_vx[verbose_mask])}")

            verbose_mask = np.isfinite(_vy)
            logging.info(f"reproject_velocity: Original {vy_var}:  min={np.nanmin(_vy[verbose_mask])} max={np.nanmax(_vy[verbose_mask])}")

            verbose_mask = np.isfinite(_v)
            logging.info(f"reproject_velocity: Original {v_var}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

            verbose_mask = np.isfinite(_vx_error)
            logging.info(f"reproject_velocity: Original {vx_error_var}: min={np.nanmin(_vx_error[verbose_mask])} max={np.nanmax(_vx_error[verbose_mask])}")

            verbose_mask = np.isfinite(_vy_error)
            logging.info(f"reproject_velocity: Original {vy_error_var}: min={np.nanmin(_vy_error[verbose_mask])} max={np.nanmax(_vy_error[verbose_mask])}")

            verbose_mask = np.isfinite(_v_error)
            logging.info(f"reproject_velocity: Original {v_error_var}: min={np.nanmin(_v_error[verbose_mask])} max={np.nanmax(_v_error[verbose_mask])}")

        for y in tqdm(range(num_y), ascii=True, desc=f"Re-projecting {vx_var}, {vy_var}, {vx_error_var}, {vy_error_var}..."):
            for x in range(num_x):
                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                t_matrix = self.transformation_matrix[y, x]
                if np.isscalar(t_matrix):
                    continue

                # Look up original cell in input ij-projection
                i, j = self.original_ij_index[y, x]

                # Re-project velocity variables
                dv = np.array([_vx[j, i], _vy[j, i]])

                # Some points get NODATA for vx but valid vy and v.v.
                if not np.any(np.isnan(dv)):
                    # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                    xy_v = np.matmul(t_matrix, dv)

                    vx[y, x] = xy_v[0]
                    vy[y, x] = xy_v[1]

                    # # Compute v: sqrt(vx^2 + vy^2)
                    v[y, x] = np.sqrt(xy_v[0]**2 + xy_v[1]**2)

                    # Look up original velocity value to compute the scale factor
                    # for v_error: scale_factor = v_old / v_new
                    v_ij_value = _v[j, i]

                    scale_factor = 1.0
                    if v_ij_value != 0:
                        scale_factor = v[y, x]/v_ij_value

                    elif v[y, x] != 0:
                        # Set re-projected v to zero - non-zero vx and vy values are
                        # introduced by warping (we don't warp input values anymore though - still need it?)
                        vx[y, x] = 0
                        vy[y, x] = 0
                        v[y, x] = 0

                    if v_ij_value and (not np.any(np.isnan(_v_error[j, i]))):
                        # Apply scale factor to the error value
                        v_error[y, x] = _v_error[j, i]*scale_factor

                        # For debugging only:
                        # Track large differences in v_error values in case they happen. If observed,
                        # most likely need to reduce error threshold for the gdal.warp()
                        # if np.abs(v_error[y, x] - _v_error[j, i]) > 100:
                        #     logging.warning(f"Computed {v_error_var}={v_error[y, x]} vs. {v_error_var}_in={_v_error[j, i]}")
                        #     logging.info(f"--->indices: i={i} j={j} vs. x={x} y={y}")
                        #     logging.info(f"--->{v_var}: {v_var}_in={v_ij_value} {v_var}_out={v[y, x]}")
                        #     logging.info(f"--->in:     {vx_var}={_vx[j, i]} {vy_var}={_vy[j, i]}")
                        #     logging.info(f"--->out:    {vx_var}={vx[y, x]} {vy_var}={vy[y, x]}")
                        #     logging.info(f"--->transf_matrix: {t_matrix}")

                    dv = [_vx_error[j, i], _vy_error[j, i]]

                    # If any of the values is NODATA, don't re-project, leave them as NODATA
                    if not np.any(np.isnan(dv)):
                        # vx_error and vy_error must be positive:
                        # use absolute values of transformation matrix to avoid
                        # negative re-projected vx_error and vy_error values
                        vx_error[y, x], vy_error[y, x] = np.matmul(np.abs(t_matrix), dv)

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(vx)
            logging.info(f"reproject_velocity: Re-projected {vx_var}:  min={np.nanmin(vx[verbose_mask])} max={np.nanmax(vx[verbose_mask])}")

            verbose_mask = np.isfinite(vy)
            logging.info(f"reproject_velocity: Re-projected {vy_var}:  min={np.nanmin(vy[verbose_mask])} max={np.nanmax(vy[verbose_mask])}")

            verbose_mask = np.isfinite(v)
            logging.info(f"reproject_velocity: Re-projected {v_var}:  min={np.nanmin(v[verbose_mask])} max={np.nanmax(v[verbose_mask])}")

            verbose_mask = np.isfinite(vx_error)
            logging.info(f"reproject_velocity: Re-projected {vx_error_var}:  min={np.nanmin(vx_error[verbose_mask])} max={np.nanmax(vx_error[verbose_mask])}")

            verbose_mask = np.isfinite(vy_error)
            logging.info(f"reproject_velocity: Re-projected {vy_error_var}:  min={np.nanmin(vy_error[verbose_mask])} max={np.nanmax(vy_error[verbose_mask])}")

            verbose_mask = np.isfinite(v_error)
            logging.info(f"reproject_velocity: Re-projected {v_error_var}:  min={np.nanmin(v_error[verbose_mask])} max={np.nanmax(v_error[verbose_mask])}")

        # Replace np.nan with DataVars.MISSING_VALUE
        MosaicsReproject.replace_nan_by_missing_value(vx)
        MosaicsReproject.replace_nan_by_missing_value(vy)
        MosaicsReproject.replace_nan_by_missing_value(v)
        MosaicsReproject.replace_nan_by_missing_value(v_error)
        MosaicsReproject.replace_nan_by_missing_value(vx_error)
        MosaicsReproject.replace_nan_by_missing_value(vy_error)

        return (vx, vy, v, vx_error, vy_error, v_error)

    def reproject_static_vars(
        self,
        vx0: np.ndarray,
        vy0: np.ndarray
    ):
        """
        Re-project:
        * dv_dt's X and Y components, and compute dv_dt based on re-projected unit flow vector.
        * v_amp's X and Y components, and compute v_amp based on re-projected unit flow vector.
        * v_amp_error's X and Y components (compute v_amp_error outside of this method as need to
            compute re-projected v_phase first).
        * v_phase's X and Y components, and compute v_phase in direction of re-projected unit flow.

        Inputs:
        =======
        vx0: numpy array that stores re-projected vx0 data, used to compute unit flow vector
        vy0: numpy array that stores re-projected vy0 data, used to compute unit flow vector

        Outputs:
        ========
        dvx_dt
        dvy_dt
        dv_dt
        vx_amp
        vy_amp
        v_amp
        vx_amp_error
        vy_amp_error
        v_amp_error
        vx_phase
        vy_phase
        v_phase

        Example:
        dvx_dt, \
        dvy_dt, \
        dv_dt, \
        vx_amp, \
        vy_amp, \
        v_amp, \
        vx_amp_error, \
        vy_amp_error, \
        v_amp_error, \
        vx_phase, \
        vy_phase, \
        v_phase = self.reproject_static_vars(vx0, vy0)
        """
        # Compute flow unit vector
        v0 = np.sqrt(vx0**2 + vy0**2) # velocity magnitude
        uv_x = vx0/v0 # unit flow vector in x direction
        uv_y = vy0/v0 # unit flow vector in y direction

        # Read X component of dv_dt
        _dvx_dt = self.ds[CompDataVars.SLOPE_VX].values
        _dvx_dt[_dvx_dt==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of dv_dt
        _dvy_dt = self.ds[CompDataVars.SLOPE_VY].values
        _dvy_dt[_dvy_dt==DataVars.MISSING_VALUE] = np.nan

        # Read X component of v_phase
        _vx_phase = self.ds[CompDataVars.VX_PHASE].values
        _vx_phase[_vx_phase==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of v_phase
        _vy_phase = self.ds[CompDataVars.VY_PHASE].values
        _vy_phase[_vy_phase==DataVars.MISSING_VALUE] = np.nan

        # Read X component of v_amp
        _vx_amp = self.ds[CompDataVars.VX_AMP].values
        _vx_amp[_vx_amp==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of v_amp
        _vy_amp = self.ds[CompDataVars.VY_AMP].values
        _vy_amp[_vy_amp==DataVars.MISSING_VALUE] = np.nan

        # Read X component of v_amp_error
        _vx_amp_error = self.ds[CompDataVars.VX_AMP_ERROR].values
        _vx_amp_error[_vx_amp_error==DataVars.MISSING_VALUE] = np.nan

        # Read Y component of v_amp_error
        _vy_amp_error = self.ds[CompDataVars.VY_AMP_ERROR].values
        _vy_amp_error[_vy_amp_error==DataVars.MISSING_VALUE] = np.nan

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(_dvx_dt)
            logging.info(f"reproject_static_vars: Original {CompDataVars.SLOPE_VX}:  min={np.nanmin(_dvx_dt[verbose_mask])} max={np.nanmax(_dvx_dt[verbose_mask])}")

            verbose_mask = np.isfinite(_dvy_dt)
            logging.info(f"reproject_static_vars: Original {CompDataVars.SLOPE_VY}:  min={np.nanmin(_dvy_dt[verbose_mask])} max={np.nanmax(_dvy_dt[verbose_mask])}")

            verbose_mask = np.isfinite(_vx_phase)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VX_PHASE}:  min={np.nanmin(_vx_phase[verbose_mask])} max={np.nanmax(_vx_phase[verbose_mask])}")

            verbose_mask = np.isfinite(_vy_phase)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VY_PHASE}:  min={np.nanmin(_vy_phase[verbose_mask])} max={np.nanmax(_vy_phase[verbose_mask])}")

            verbose_mask = np.isfinite(_vx_amp)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VX_AMP}:  min={np.nanmin(_vx_amp[verbose_mask])} max={np.nanmax(_vx_amp[verbose_mask])}")

            verbose_mask = np.isfinite(_vy_amp)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VY_AMP}:  min={np.nanmin(_vy_amp[verbose_mask])} max={np.nanmax(_vy_amp[verbose_mask])}")

            verbose_mask = np.isfinite(_vx_amp_error)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VX_AMP_ERROR}:  min={np.nanmin(_vx_amp_error[verbose_mask])} max={np.nanmax(_vx_amp_error[verbose_mask])}")

            verbose_mask = np.isfinite(_vy_amp_error)
            logging.info(f"reproject_static_vars: Original {CompDataVars.VY_AMP_ERROR}:  min={np.nanmin(_vy_amp_error[verbose_mask])} max={np.nanmax(_vy_amp_error[verbose_mask])}")

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)
        xy_dims = (num_y, num_x)

        # TODO: may be too many variables are allocated at the same time - break it up
        # into multiple functions (don't re-project all at once)

        # Allocate output data for dv_dt variables
        dvx_dt = np.full(xy_dims, np.nan, dtype=np.float32)
        dvy_dt = np.full(xy_dims, np.nan, dtype=np.float32)
        dv_dt = np.full(xy_dims, np.nan, dtype=np.float32)

        # Allocate output data for v_amp variables
        vx_amp = np.full(xy_dims, np.nan, dtype=np.float32)
        vy_amp = np.full(xy_dims, np.nan, dtype=np.float32)
        v_amp = np.full(xy_dims, np.nan, dtype=np.float32)

        # Allocate output data for v_amp_error variables
        vx_amp_error = np.full(xy_dims, np.nan, dtype=np.float32)
        vy_amp_error = np.full(xy_dims, np.nan, dtype=np.float32)
        v_amp_error = np.full(xy_dims, np.nan, dtype=np.float32)

        # Allocate output data for v_phase variables
        vx_phase = np.full(xy_dims, np.nan, dtype=np.float32)
        vy_phase = np.full(xy_dims, np.nan, dtype=np.float32)
        v_phase = np.full(xy_dims, np.nan, dtype=np.float32)

        if MosaicsReproject.VERBOSE:
            # Read original dv_dt values
            _v = self.ds[CompDataVars.SLOPE_V].values
            _v[_v==DataVars.MISSING_VALUE] = np.nan

            verbose_mask = np.isfinite(_v)

            # Report min and max values for the error variable
            logging.info(f"reproject_static_vars: Original {CompDataVars.SLOPE_V}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

            # Read original v_phase values
            _v = None
            _v = self.ds[CompDataVars.V_PHASE].values
            _v[_v==DataVars.MISSING_VALUE] = np.nan

            verbose_mask = np.isfinite(_v)

            # Report min and max values for the error variable
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_PHASE}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

        for y in tqdm(range(num_y), ascii=True, desc=f"Re-projecting {CompDataVars.SLOPE_V}, {CompDataVars.V_AMP}, {CompDataVars.V_AMP_ERROR}, {CompDataVars.V_PHASE}..."):
            for x in range(num_x):
                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                t_matrix = self.transformation_matrix[y, x]
                if np.isscalar(t_matrix):
                    continue

                # Look up original cell in input ij-projection
                i, j = self.original_ij_index[y, x]

                # Re-project dv_dt's X and Y components
                dv = np.array([_dvx_dt[j, i], _dvy_dt[j, i]])

                # Some points get NODATA for vx but valid vy and v.
                if np.all(np.isfinite(dv)):
                    # Apply transformation matrix to (dvx_dt, dvy_dt) vector
                    dvx_dt[y, x], dvy_dt[y, x] = np.matmul(t_matrix, dv)

                # Re-project v_amp's X and Y components
                dv = np.array([_vx_amp[j, i], _vy_amp[j, i]])

                # Some points get NODATA for vx but valid vy and v.
                if np.all(np.isfinite(dv)):
                    # Apply transformation matrix to (vx, vy) values
                    vx_amp[y, x], vy_amp[y, x] = np.matmul(np.abs(t_matrix), dv)

                # Re-project v_amp_error's components
                dv = [_vx_amp_error[j, i], _vy_amp_error[j, i]]

                # If any of the values is NODATA, don't re-project, leave them as NODATA
                if np.all(np.isfinite(dv)):
                    # vx_error and vy_error must be positive:
                    # use absolute values of transformation matrix to avoid
                    # negative re-projected vx_error and vy_error values
                    vx_amp_error[y, x], vy_amp_error[y, x] = np.matmul(np.abs(t_matrix), dv)

                # Re-project v_phase's components
                dv = [_vx_phase[j, i], _vy_phase[j, i]]

                # If any of the values is NODATA, don't re-project, leave them as NODATA
                if np.all(np.isfinite(dv)):
                    vx_phase[y, x], vy_phase[y, x] = np.matmul(np.abs(t_matrix), dv)

        # No need for some of original data, cleanup
        _dvx_dt = None
        _dvy_dt = None
        _vx_amp = None
        _vy_amp = None
        gc.collect()

        # Compute dv_dt: flow acceleration in direction of unit flow vector
        dv_dt = dvx_dt * uv_x
        dv_dt += dvy_dt * uv_y

        # Wrap components of v_amp and v_phase to make sure they are in valid ranges
        vx_phase, vx_amp = MosaicsReproject.wrap_amp_phase(vx_phase, vx_amp)
        vy_phase, vy_amp = MosaicsReproject.wrap_amp_phase(vy_phase, vy_amp)

        # Compute v_phase and v_amp using analytical solution
        v_phase, v_amp = MosaicsReproject.seasonal_velocity_rotation(vx0, vy0, vx_phase, vy_phase, vx_amp, vy_amp)

        # Compute v_amp_error using scale factor b/w old and newly re-projected v_amp values
        # (don't project v_amp_error in direction of unit flow vector
        # like in composites)
        # Scale the "v_amp_error" as new "v_amp" is computed now
        _v_amp = self.ds[CompDataVars.V_AMP].values
        _v_amp[_v_amp==DataVars.MISSING_VALUE] = np.nan

        _v_amp_error = self.ds[CompDataVars.V_AMP_ERROR].values
        _v_amp_error[_v_amp_error==DataVars.MISSING_VALUE] = np.nan

        if MosaicsReproject.VERBOSE:
            # Report min and max values for the error variable
            verbose_mask = np.isfinite(_v_amp)
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_AMP}: min={np.nanmin(_v_amp[verbose_mask])} max={np.nanmax(_v_amp[verbose_mask])}")

            verbose_mask = np.isfinite(_v_amp_error)
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_AMP_ERROR}: min={np.nanmin(_v_amp_error[verbose_mask])} max={np.nanmax(_v_amp_error[verbose_mask])}")

        for y in tqdm(range(num_y), ascii=True, desc=f"Scaling {CompDataVars.V_AMP_ERROR}..."):
            for x in range(num_x):
                if not np.isfinite(v_amp[y, x]):
                    continue

                # Look up original cell in input ij-projection
                i, j = self.original_ij_index[y, x]

                # Look up original velocity value to compute the scale factor
                # for v_error: scale_factor = v_old / v_new
                v_ij_value = _v_amp[j, i]

                scale_factor = 1.0
                if v_ij_value != 0:
                    scale_factor = v_amp[y, x]/v_ij_value

                if v_ij_value and (not np.any(np.isnan(_v_amp_error[j, i]))):
                    # Apply scale factor to the error value
                    v_amp_error[y, x] = _v_amp_error[j, i]*scale_factor

        if MosaicsReproject.VERBOSE:
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_VX}:  min={np.nanmin(dvx_dt)} max={np.nanmax(dvx_dt)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_VY}:  min={np.nanmin(dvy_dt)} max={np.nanmax(dvy_dt)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_AMP}:  min={np.nanmin(vx_amp)} max={np.nanmax(vx_amp)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_AMP}:  min={np.nanmin(vy_amp)} max={np.nanmax(vy_amp)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_AMP_ERROR}:  min={np.nanmin(vx_amp_error)} max={np.nanmax(vx_amp_error)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_AMP_ERROR}:  min={np.nanmin(vy_amp_error)} max={np.nanmax(vy_amp_error)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_PHASE}:  min={np.nanmin(vx_phase)} max={np.nanmax(vx_phase)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_PHASE}:  min={np.nanmin(vy_phase)} max={np.nanmax(vy_phase)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_V}:  min={np.nanmin(dv_dt)} max={np.nanmax(dv_dt)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_PHASE}:  min={np.nanmin(v_phase)} max={np.nanmax(v_phase)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_AMP}:  min={np.nanmin(v_amp)} max={np.nanmax(v_amp)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_AMP_ERROR}:  min={np.nanmin(vy_amp_error)} max={np.nanmax(vy_amp_error)}")

        # Replace np.nan with DataVars.MISSING_VALUE
        MosaicsReproject.replace_nan_by_missing_value(dvx_dt)
        MosaicsReproject.replace_nan_by_missing_value(dvy_dt)
        MosaicsReproject.replace_nan_by_missing_value(dv_dt)
        MosaicsReproject.replace_nan_by_missing_value(vx_amp)
        MosaicsReproject.replace_nan_by_missing_value(vy_amp)
        MosaicsReproject.replace_nan_by_missing_value(v_amp)
        MosaicsReproject.replace_nan_by_missing_value(vx_amp_error)
        MosaicsReproject.replace_nan_by_missing_value(vy_amp_error)
        MosaicsReproject.replace_nan_by_missing_value(v_amp_error)
        MosaicsReproject.replace_nan_by_missing_value(vx_phase)
        MosaicsReproject.replace_nan_by_missing_value(vy_phase)
        MosaicsReproject.replace_nan_by_missing_value(v_phase)

        return (dvx_dt, dvy_dt, dv_dt, vx_amp, vy_amp, v_amp, vx_amp_error, vy_amp_error, v_amp_error, vx_phase, vy_phase, v_phase)

    @staticmethod
    def wrap_amp_phase(v_phase_days, v_amp):
        """
        Wrap phase and amplitude to be within valid ranges.
        """
        # Convert phase from days to degrees
        v_phase = v_phase_days*360/365.24

        mask = v_amp < 0
        v_amp[mask] *= -1.0
        v_phase[mask] += 180

        # Matlab prototype code:
        # % Wrap to 360 degrees:
        # px = vx_phase_r > 0;
        # vx_phase_r = mod(vx_phase_r, 360);
        # vx_phase_r((vx_phase_r == 0) & px) = 360;
        mask = v_phase > 0
        # v_phase[mask] = np.remainder(v_phase[mask], _two_pi)
        v_phase[mask] = np.remainder(v_phase[mask], 360.0)
        mask = mask & (v_phase == 0)
        # v_phase[mask] = _two_pi
        v_phase[mask] = 360.0

        # New in Python: convert all values to positive
        mask = v_phase < 0
        if np.any(mask):
            # logging.info(f'Got negative phase, converting to positive values')
            # v_phase[mask] += _two_pi
            v_phase[mask] = np.remainder(v_phase[mask], -360.0)
            v_phase[mask] += 360.0

        # Matlab prototype code:
        # % Convert degrees to days:
        # vx_phase_r = vx_phase_r*365.24/360;
        # vy_phase_r = vy_phase_r*365.24/360;
        # Composites code does:
        # v_phase = 365.25*((0.25 - phase_rad/_two_pi) % 1),
        # and since vx_phase and vy_phase are already shifted by 0.25 in original projection,
        # so we don't need to do it after rotation in direction of v0

        # Convert phase back to the day of the year:
        v_phase = v_phase*365.24/360

        return v_phase, v_amp

    @staticmethod
    def replace_nan_by_missing_value(data):
        """
        Replace np.nan with DataVars.MISSING_VALUE.

        Inputs:
        =======
        data: numpy.ndarray to replace the values of.
        """
        mask = np.isnan(data)
        data[mask] = DataVars.MISSING_VALUE

    @staticmethod
    def seasonal_velocity_rotation(vx0, vy0, vx_phase, vy_phase, vx_amp, vy_amp):
        """
        Rotate v_phase and v_amp in the direction of v which is defined by vx0 and vy0.

        % seasonal_velocity_rotation gives the amplitude and phase of seasonal
        % velocity components. (Only the x and y components change when rotation is
        % applied, i.e., v_amp and v_phase are unchanged).
        %
        % Inputs:
        % theta (degrees) rotation of the coordinate system.
        % vx_amp (m/yr) x component of seasonal amplitude in the original coordinate system.
        % vx_phase (doy) day of maximum x velocity in original coordinate system.
        % vy_amp (m/yr) y component of seasonal amplitude in the original coordinate system.
        % vy_phase (doy) day of maximum y velocity in original coordinate system.
        %
        % Outputs:
        % vx_amp_r (m/yr) x component of seasonal amplitude in the original coordinate system.
        % vx_phase_r (doy) day of maximum x velocity in original coordinate system.
        % vy_amp_r (m/yr) y component of seasonal amplitude in the original coordinate system.
        % vy_phase_r (doy) day of maximum y velocity in original coordinate system.
        %
        % Written (in Matlab) by Alex Gardner and Chad Greene, July 2022.

        Returns:
        v_phase (doy) - day of maximum velocity in original coordinate system
        v_amp (m/yr) - seasonal amplitude in the original coordinate system
        """
        _two_pi = np.pi * 2

        # Matlab prototype code:
        # % Convert phase values from day-of-year to degrees:
        # vx_phase_deg = vx_phase*360/365.24;
        # vy_phase_deg = vy_phase*360/365.24;
        # TODO: avoid conversion to degrees - go from day-of-year to radians
        vx_phase_deg = vx_phase/365.24
        vy_phase_deg = vy_phase/365.24

        # Don't use np.nan values in calculations to avoid warnings
        valid_mask = (~np.isnan(vx_phase_deg)) & (~np.isnan(vy_phase_deg))

        # logging.info(f'Degrees: vx_phase_deg={vx_phase_deg[valid_mask]} vy_phase_deg={vy_phase_deg[valid_mask]}')

        # Convert degrees to radians as numpy trig. functions take angles in radians
        vx_phase_deg *= _two_pi
        vy_phase_deg *= _two_pi
        # logging.info(f'Radians: vx_phase_deg={vx_phase_deg[valid_mask]} vy_phase_deg={vy_phase_deg[valid_mask]}')

        # Matlab prototype code:
        # % Rotation matrix for x component:
        # A1 =  vx_amp.*cosd(theta);
        # B1 = -vy_amp.*sind(theta);

        # New in Python code: compute theta rotation angle
        # theta = arctan(vy0/vx0), since sin(theta)=vy0 and cos(theta)=vx0,
        theta = np.full_like(vx_phase_deg, np.nan)
        theta[valid_mask] = np.arctan2(vy0[valid_mask], vx0[valid_mask])

        if np.any(theta<0):
            # logging.info(f'Got negative theta, converting to positive values')
            mask = (theta<0)
            theta[mask] += _two_pi

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # New in Python: assume clockwise rotation by theta as we need to align
        # vector with v0 direction. Therefore  use clockwise transformation matrix
        # for rotation, not counter-clockwise as in Matlab prototype code.
        A1 = vx_amp*cos_theta
        B1 = vy_amp*sin_theta

        # Matlab prototype code:
        # vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
        # vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

        # We want to retain the component only in the direction of v0,
        # which becomes new v_amp and v_phase
        v_amp = np.full_like(vx_phase_deg, np.nan)
        v_phase = np.full_like(vx_phase_deg, np.nan)

        v_amp[valid_mask] = np.hypot(
            A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask]),
            A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask])
        )
        # np.arctan2 returns phase in radians, convert to degrees
        v_phase[valid_mask] = np.arctan2(
            A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask]),
            A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask])
        )*180.0/np.pi

        # Matlab prototype code:
        # % Make all amplitudes positive (and reverse phase accordingly):
        # nx = vx_amp_r<0; % indices of negative Ax_r
        # vx_amp_r(nx) = -vx_amp_r(nx);
        # vx_phase_r(nx) = vx_phase_r(nx)+180;
        mask = v_amp < 0
        v_amp[mask] *= -1.0
        # v_phase[mask] += np.pi
        v_phase[mask] += 180

        # Matlab prototype code:
        # % Wrap to 360 degrees:
        # px = vx_phase_r > 0;
        # vx_phase_r = mod(vx_phase_r, 360);
        # vx_phase_r((vx_phase_r == 0) & px) = 360;
        mask = v_phase > 0
        # v_phase[mask] = np.remainder(v_phase[mask], _two_pi)
        v_phase[mask] = np.remainder(v_phase[mask], 360.0)
        mask = mask & (v_phase == 0)
        # v_phase[mask] = _two_pi
        v_phase[mask] = 360.0

        # New in Python: convert all values to positive
        mask = v_phase < 0
        if np.any(mask):
            # logging.info(f'Got negative phase, converting to positive values')
            # v_phase[mask] += _two_pi
            v_phase[mask] = np.remainder(v_phase[mask], -360.0)
            v_phase[mask] += 360.0

        # Matlab prototype code:
        # % Convert degrees to days:
        # vx_phase_r = vx_phase_r*365.24/360;
        # vy_phase_r = vy_phase_r*365.24/360;
        # Composites code does:
        # v_phase = 365.25*((0.25 - phase_rad/_two_pi) % 1),
        # and since vx_phase and vy_phase are already shifted by 0.25 in original projection,
        # so we don't need to do it after rotation in direction of v0
        # Convert phase to the day of the year:
        v_phase = v_phase*365.24/360

        return v_phase, v_amp

    def get_distortion_for_debugging(self, vx_var: str, vy_var: str):
        """
        Get distortion in X and Y dimensions for the variables. This is for
        debugging purposes only. It replaces values of vx(vy) variables with 1(0),
        then vx(vy) with 0(1) to get reprojection distortion map for each coordinate.

        vx_var: name of the variable's X component
        vy_var: name of the variable's Y component
        """
        # Get distortion in x dimension:
        logging.info(f'Get distortion for {vx_var}')
        np_vx = self.ds[vx_var].values

        np_vx[np_vx==DataVars.MISSING_VALUE] = np.nan
        np_vx[~np.isnan(np_vx)] = 1.0

        # Warp y component
        np_vy = self.ds[vy_var].values
        np_vy[np_vy==DataVars.MISSING_VALUE] = np.nan

        np_vy[~np.isnan(np_vy)] = 0.0

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        vx = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        vy = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)

        # TODO: make use of parallel processing as cells are independent to speed up
        #       the processing
        for y_index in tqdm(range(num_y), ascii=True, desc=f"Re-projecting X unit {vx_var}, {vy_var}..."):
            for x_index in range(num_x):
                # Get values corresponding to the cell in input projection
                if np.isscalar(self.original_ij_index[y_index, x_index]):
                    # There is no corresponding cell in input projection
                    continue

                v_i, v_j = self.original_ij_index[y_index, x_index]
                dv = np.array([
                    np_vx[v_j, v_i],
                    np_vy[v_j, v_i]
                ])

                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                if not np.isscalar(self.transformation_matrix[y_index, x_index]) and \
                   not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                    # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                    xy_v = np.matmul(self.transformation_matrix[y_index, x_index], dv)

                    vx[y_index, x_index] = xy_v[0]
                    vy[y_index, x_index] = xy_v[1]

        masked_np = np.ma.masked_equal(vx, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vx_var}:  min={np.nanmin(masked_np)} max={np.nanmax(masked_np)}")

        masked_np = np.ma.masked_equal(vy, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vy_var}:  min={np.nanmin(masked_np)} max={np.nanmax(masked_np)}")

        logging.info(f'Get distortion for {vy_var}')
        # Get distortion in y dimension: replace x values with 0, y values with 1
        # Remember vx distortion
        vx_xunit = np.array(vx)
        vy_xunit = np.array(vy)

        np_vx[~np.isnan(np_vx)] = 0.0

        # Warp y component
        np_vy[~np.isnan(np_vy)] = 1.0

        vx = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        vy = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)

        # TODO: make use of parallel processing as cells are independent to speed up
        #       the processing
        for y_index in tqdm(range(num_y), ascii=True, desc=f"Re-projecting Y unit {vx_var}, {vy_var}..."):
            for x_index in range(num_x):
                if np.isscalar(self.original_ij_index[y_index, x_index]):
                    # There is no corresponding cell in input projection
                    continue

                # Get values corresponding to the cell in input projection
                v_i, v_j = self.original_ij_index[y_index, x_index]
                dv = np.array([
                    np_vx[v_j, v_i],
                    np_vy[v_j, v_i]
                ])

                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                if not np.isscalar(self.transformation_matrix[y_index, x_index]) and \
                   not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                    # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                    xy_v = np.matmul(self.transformation_matrix[y_index, x_index], dv)

                    vx[y_index, x_index] = xy_v[0]
                    vy[y_index, x_index] = xy_v[1]

        masked_np = np.ma.masked_equal(vx, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vx_var}:  min={np.nanmin(masked_np)} max={np.nanmax(masked_np)}")

        masked_np = np.ma.masked_equal(vy, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vy_var}:  min={np.nanmin(masked_np)} max={np.nanmax(masked_np)}")

        return (vx_xunit, vy_xunit, vx, vy)

    def create_transformation_matrix(self, vx_var, vy_var, v_var):
        """
        This method creates transformation matrix for each point of the grid.

        vx_var - Name of the X component of the variable to decide if there is
                 data in the cell.
        vy_var - Name of the Y component of the variable to decide if there is
                 data in the cell.
        v_var - Name of the variable to decide if there is data in the cell.
        """
        logging.info(f'Creating trasformation matrix...')

        # Project the bounding box into output projection
        input_projection = osr.SpatialReference()
        input_projection.ImportFromEPSG(self.ij_epsg)

        output_projection = osr.SpatialReference()

        if self.xy_epsg != ESRICode:
            output_projection.ImportFromEPSG(self.xy_epsg)

        else:
            output_projection.ImportFromProj4(ESRICode_Proj4)

        self.xy_central_meridian = output_projection.GetProjParm("central_meridian")

        ij_to_xy_transfer = osr.CoordinateTransformation(input_projection, output_projection)
        xy_to_ij_transfer = osr.CoordinateTransformation(output_projection, input_projection)

        # Compute bounding box in source projection
        ij_x_bbox, ij_y_bbox = self.bounding_box()
        logging.info(f"P_in bounding box:  x: {ij_x_bbox} y: {ij_y_bbox}")
        # Re-project bounding box to output projection
        points_in = np.array([
            [ij_x_bbox.min, ij_y_bbox.max],
            [ij_x_bbox.max, ij_y_bbox.max],
            [ij_x_bbox.max, ij_y_bbox.min],
            [ij_x_bbox.min, ij_y_bbox.min]
        ])

        # logging.info(f'Input points: {points_in}')
        points_out = ij_to_xy_transfer.TransformPoints(points_in)

        bbox_out_x = Bounds([each[0] for each in points_out])
        bbox_out_y = Bounds([each[1] for each in points_out])

        # Get corresponding bounding box in output projection based on edge points of
        # bounding polygon in P_in projection
        self.x0_bbox, self.y0_bbox = Grid.bounding_box(bbox_out_x, bbox_out_y, self.x_size)
        logging.info(f"P_out bounding box: x: {self.x0_bbox} y: {self.y0_bbox}")

        # Output grid will be used as input to the gdal.warp() and to identify
        # corresponding grid cells in original P_in projection when computing
        # transformation matrix
        self.x0_grid, self.y0_grid = Grid.create(self.x0_bbox, self.y0_bbox, self.x_size)
        logging.info(f"Grid in P_out: num_x={len(self.x0_grid)} num_y={len(self.y0_grid)}")
        # logging.info(f"Cell centers in P_out: x_min={self.x0_grid[0]} x_max={self.x0_grid[-1]} y_max={self.y0_grid[0]} y_min={self.y0_grid[-1]}")

        # Read transformation matrix if it exists
        if os.path.exists(MosaicsReproject.TRANSFORMATION_MATRIX_FILE):
            # Transformation matrix exists, just load it.
            # There is no security concern about "allow_pickle" as this code
            # runs only manually to generate annual mosaics
            logging.info(f'Loading {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')
            npzfile = np.load(MosaicsReproject.TRANSFORMATION_MATRIX_FILE, allow_pickle=True)
            self.transformation_matrix = npzfile['transformation_matrix']
            self.original_ij_index = npzfile['original_ij_index']
            logging.info(f'Loaded transformation_matrix and original_ij_index from {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')

            # Make sure matrix dimensions correspond to the target grid
            if self.transformation_matrix.shape != (len(self.y0_grid), len(self.x0_grid)):
                raise RuntimeError(f'Unexpected shape of transformation matrix: {self.transformation_matrix.shape}' \
                                    'vs. expected {(len(self.y0_grid), len(self.x0_grid))}')

            return

        xy0_points = MosaicsReproject.dims_to_grid(self.x0_grid, self.y0_grid)

        # Get corresponding to xy0_points in original projection
        ij0_points = xy_to_ij_transfer.TransformPoints(xy0_points)

        logging.info(f'Got list of points in original projection...')

        # Calculate x unit vector: add unit length to ij0_points.x
        # TODO: possible optimization - just use already transformed points when
        #       computing bounding box in target projection as it's only by one cell shift in x dimension
        ij_unit = np.array(ij0_points)
        ij_unit[:, 0] += self.x_size
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())

        num_xy0_points = len(xy0_points)

        # Compute X unit vector based on xy0_points, xy_points
        # in output projection
        xunit_v = np.zeros((num_xy0_points, 3))

        logging.info(f'Creating unit vectors...')

        # Compute unit vector for each cell of the output grid
        for index in range(num_xy0_points):
            xunit_v[index] = np.array(xy_points[index]) - np.array(xy0_points[index])
            # xunit_v[index] /= np.linalg.norm(xunit_v[index])
            xunit_v[index] /= self.x_size

        # Calculate Y unit vector: add unit length to ij0_points.y
        ij_unit = np.array(ij0_points)
        ij_unit[:, 1] += self.y_size
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())

        yunit_v = np.zeros((num_xy0_points, 3))

        # Compute Y unit vector based on xy0_points, xy_points
        # in output projection
        for index in range(num_xy0_points):
            yunit_v[index] = np.array(xy_points[index]) - np.array(xy0_points[index])
            # yunit_v[index] /= np.linalg.norm(yunit_v[index])
            yunit_v[index] /= np.abs(self.y_size)

        # Compute transformation matrix per cell
        self.transformation_matrix = np.full((num_xy0_points), DataVars.MISSING_VALUE, dtype=object)
        # self.transformation_matrix.fill(DataVars.MISSING_VALUE)

        # For each re-projected cell store indices of corresponding cells in
        # original projection
        self.original_ij_index = np.zeros((num_xy0_points), dtype=object)

        # Counter of how many points don't have transformation matrix
        no_value_counter = 0

        # scale_factor_x = self.x_size/MosaicsReproject.TIME_DELTA
        # scale_factor_y = self.y_size/MosaicsReproject.TIME_DELTA
        # DEBUG: try to remove scale factor and creating T matrix in terms of
        # displacement
        scale_factor_x = 1.0
        scale_factor_y = 1.0

        # Local normal vector
        # normal = np.array([0.0, 0.0, 1.0])

        # e = normal[2]*scale_factor_y
        # f = normal[2]*scale_factor_x

        num_i = len(self.ds.x.values)
        num_j = len(self.ds.y.values)

        # debug = False

        # For each point on the output grid:
        logging.info(f'Populating transformation matrix...')

        for each_index in tqdm(range(num_xy0_points), ascii=True, desc="Creating transformation matrix..."):
            # Find corresponding point in source P_in projection
            ij_point = ij0_points[each_index]

            # Find indices for the original point on its grid
            x_index = (ij_point[0] - ij_x_bbox.min) / self.x_size
            y_index = (ij_point[1] - ij_y_bbox.max) / self.y_size

            if (x_index < 0) or (y_index < 0) or \
               (x_index >= num_i) or (x_index < 0) or \
               (y_index >= num_j) or (y_index < 0):
                no_value_counter += 1
                # logging.info('Skipping out of range point')
                continue

            x_index = int(x_index)
            y_index = int(y_index)

            # Make sure we found correct indices for the point on original grid
            # if debug:
            #     x_min = self.ds.x.values[x_index]-self.x_size/2
            #     x_max = self.ds.x.values[x_index+1]+self.x_size/2 if (x_index+1) < num_i else ij_x_bbox.max
            #     y_min = self.ds.y.values[y_index+1]+self.y_size/2 if (y_index+1) < num_j else ij_y_bbox.min
            #     y_max = self.ds.y.values[y_index]-self.y_size/2
            #
            #     assert ij_point[0] >= x_min and ij_point[0] <= x_max, \
            #            f"Invalid X index={x_index} is found for {ij_point}: x_max={x_max} diff/size={(ij_point[0] - ij_x_bbox.min) / self.x_size}"
            #     assert ij_point[1] <= y_max and ij_point[1] >= y_min, \
            #            f"Invalid Y index={y_index} is found for {ij_point}: y_min={y_min} y_max={y_max} (max={ij_y_bbox.max}) diff/size={(ij_point[1] - ij_y_bbox.max) / self.y_size}"

            self.original_ij_index[each_index] = [x_index, y_index]

            # Check if velocity=NODATA_VALUE for original point -->
            # don't compute the matrix
            v_value = self.ds[v_var].isel(y=y_index, x=x_index).values

            if np.isnan(v_value) or v_value.item() == DataVars.MISSING_VALUE:
                no_value_counter += 1
                continue

            # Double check that vx and vy are valid for the point in original projection
            vx_value = self.ds[vx_var].isel(y=y_index, x=x_index).values
            vy_value = self.ds[vy_var].isel(y=y_index, x=x_index).values

            if np.isnan(vx_value) or np.isnan(vy_value) or \
               (vx_value.item() == DataVars.MISSING_VALUE) or \
               (vy_value.item() == DataVars.MISSING_VALUE):
                raise RuntimeError(f"Mix of invalid vx={vx_value} and vy={vy_value} in original projection")

            xunit = xunit_v[each_index]
            yunit = yunit_v[each_index]

            # See (A9)-(A15) in Yang's autoRIFT paper:
            # a = normal[2]*yunit[0]-normal[0]*yunit[2]
            # b = normal[2]*yunit[1]-normal[1]*yunit[2]
            # c = normal[2]*xunit[0]-normal[0]*xunit[2]
            # d = normal[2]*xunit[1]-normal[1]*xunit[2]
            # Since normal[0]=normal[1]=0, remove not necessary second multiplication,
            # and remove "normal[2]*" since normal[2]=1
            # a = yunit[0]
            # b = yunit[1]
            # c = xunit[0]
            # d = xunit[1]
            #
            # self.transformation_matrix[each_index] = np.array([[-b*f, d*e], [a*f, -c*e]])
            # self.transformation_matrix[each_index] /= (a*d - b*c)
            self.transformation_matrix[each_index] = np.array(
                [[-yunit[1]*scale_factor_x, xunit[1]*scale_factor_y],
                 [yunit[0]*scale_factor_x, -xunit[0]*scale_factor_y]]
            )
            self.transformation_matrix[each_index] /= (yunit[0]*xunit[1] - yunit[1]*xunit[0])

        # Reshape transformation matrix and original cell indices into 2D matrix: (y, x)
        self.transformation_matrix = self.transformation_matrix.reshape((len(self.y0_grid), len(self.x0_grid)))
        self.original_ij_index = self.original_ij_index.reshape((len(self.y0_grid), len(self.x0_grid)))
        logging.info(f"Number of points with no transformation matrix: {no_value_counter} out of {num_xy0_points} points ({no_value_counter/num_xy0_points*100.0}%)")

        #  transformation matrix and mapping to original ij index for output grid to
        # numpy archive - don't need to calculate these every time need to re-project each
        # of the annual and static mosaics for the same region.
        logging.info(f'Saving transformation_matrix and original_ij_index arrays to {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')

        np.savez(
            MosaicsReproject.TRANSFORMATION_MATRIX_FILE,
            transformation_matrix=self.transformation_matrix,
            original_ij_index=self.original_ij_index
        )
        logging.info(f'Saved data to {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')

    def spatial_ref_32x(self):
        """
        Format spatial_ref attribute value for the UTM_Projection.
        """
        epsg = math.floor(self.xy_epsg/100)*100
        zone = self.xy_epsg - epsg
        hemisphere = None
        # We only worry about the following EPSG and zone:
        # 32600 + zone in the northern hemisphere
        # 32700 + zone in the southern hemisphere
        if epsg == 32700:
            hemisphere = 'S'

        elif epsg == 32600:
            hemisphere = 'N'

        else:
            raise RuntimeError(f"Unsupported target projection {self.xy_epsg} is provided.")

        return zone, f"PROJCS[\"WGS 84 / UTM zone {zone}{hemisphere}\"," \
            "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\"," \
            "6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]]," \
            "AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0," \
            "AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433," \
            "AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]," \
            "PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0]," \
            f"PARAMETER[\"central_meridian\",{self.xy_central_meridian}]," \
            "PARAMETER[\"scale_factor\",0.9996]," \
            "PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0]," \
            "UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]]," \
            "AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]," \
            f"AUTHORITY[\"EPSG\",\"{self.xy_epsg}\"]]"

    @staticmethod
    def dims_to_grid(x, y):
        """
        Convert x, y dimensions of the dataset into numpy grid array in (y, x) order.
        """
        # Use z=0 as osr.CoordinateTransformation.TransformPoints() returns 3d point coordinates,
        # so have to keep grip points defined in 3D.
        x_1, y_1 = np.meshgrid(x, y)
        z_1 = np.zeros_like(x_1)

        return list(zip(x_1.flatten(), y_1.flatten(), z_1.flatten()))

def main(input_file: str, output_file: str, output_proj: int, matrix_file: str, verbose_flag: bool, compute_debug_vars: bool=False):
    """
    Main function of the module to be able to invoke the code from
    another Python module.
    """
    MosaicsReproject.VERBOSE = verbose_flag
    MosaicsReproject.COMPUTE_DEBUG_VARS = compute_debug_vars
    MosaicsReproject.TRANSFORMATION_MATRIX_FILE = matrix_file

    reproject = MosaicsReproject(input_file, output_proj)
    reproject(output_file)

    logging.info(f'Done re-projection of {input_file}')


if __name__ == '__main__':
    """
    Re-project ITS_LIVE mosaic (static or annual) to the target projection.
    """
    import sys

    parser = argparse.ArgumentParser(description='Re-project ITS_LIVE static or annual mosaics to new projection.')
    parser.add_argument(
        '-i', '--input',
        dest='input_file',
        type=str,
        required=True,
        help='Input file name for ITS_LIVE mosaic')
    parser.add_argument(
        '-p', '--projection',
        dest='output_proj',
        type=int,
        required=True,
        help='Target projection')
    parser.add_argument(
        '-o', '--output',
        dest='output_file',
        type=str,
        default=None,
        required=False,
        help='Output filename to store re-projected mosaic in target projection')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output about re-projection [False]'
    )
    parser.add_argument(
        '-d', '--compute_debug_vars',
        action='store_true',
        help='Enable computation of validation variables, done to assist debugging only [False]'
    )
    parser.add_argument(
        '-m', '--transformation_matrix_file',
        default='transformation_matrix.npy',
        type=str,
        help='Store transformation matrix to provided file and re-use it to build all mosaics for the same region [%(default)s]'
    )

    command_args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f'Command args: {command_args}')

    main(
        command_args.input_file,
        command_args.output_file,
        command_args.output_proj,
        command_args.transformation_matrix_file,
        command_args.verbose,
        command_args.compute_debug_vars
    )

    logging.info('Done.')
