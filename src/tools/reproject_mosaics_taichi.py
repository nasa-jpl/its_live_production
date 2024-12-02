"""
Reprojection tool for ITS_LIVE mosaics to new target projection.

Examples:
$ python reproject_mosaics.py -i input_filename -p target_projection -o output_filename

    Reproject "input_filename" into 'target_projection' and output new mosaic into
'output_filename' in NetCDF format.

$ python ./reproject_mosaics.py -i  ITS_LIVE_velocity_120m_HMA_2015_v02.nc -o reproject_ITS_LIVE_velocity_120m_HMA_2015_v02.nc -p 102027

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Yang Lei (Caltech)
"""
import taichi as ti
ti.init(arch=ti.cpu)

import argparse
import gc
import logging
import math
import numpy as np
import os
from osgeo import osr, gdal
from tqdm import tqdm
import timeit
import xarray as xr

from grid import Grid, Bounds
from itscube_types import Coords, DataVars, Output, CompDataVars, to_int_type, ShapeFile
from nsidc_types import Mapping

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

# This is for testing of the reprojection to EPSG:8859 - to verify that HMA reprojected mosaic
# to ESRI 102027 is correct.
spatial_ref_8859 = 'PROJCRS["WGS 84 / Equal Earth Asia-Pacific",' \
    'BASEGEOGCRS["WGS 84", DATUM["World Geodetic System 1984", ' \
    'ELLIPSOID["WGS 84",6378137,298.257223563, LENGTHUNIT["metre",1]]], ' \
    'PRIMEM["Greenwich",0, ANGLEUNIT["degree",0.0174532925199433]], ID["EPSG",4326]], ' \
    'CONVERSION["Asia-Pacific Equal Earth", METHOD["Equal Earth"], ' \
    'PARAMETER["Longitude of natural origin",150, ANGLEUNIT["degree",0.0174532925199433]], ' \
    'PARAMETER["False easting",0, LENGTHUNIT["metre",1]], ' \
    'PARAMETER["False northing",0, LENGTHUNIT["metre",1]], ID["EPSG",8859]], ' \
    'CS[Cartesian,2], AXIS["easting (X)",east, ORDER[1], LENGTHUNIT["metre",1]], ' \
    'AXIS["northing (Y)",north, ORDER[2], LENGTHUNIT["metre",1]], ID["EPSG",8859]]'

PROJECTION_ATTR = 'projection'

# Non-EPSG projection that can be provided on output
ESRICode = 102027

# last: ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
# ESRICode_Proj4 = '+proj=lcc +lat_1=15 +lat_2=65 +lat_0=30 +lon_0=95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'

required_mapping_attributes = {
    32610: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -123.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    },
    32632: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 9.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    },
    32638: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 45.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    },
    32645: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 87.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 0.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    },
    32718: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: -75.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    },
    32759: {
        Mapping.LONGITUDE_OF_CENTRAL_MERIDIAN: 171.0,
        Mapping.LATITUDE_OF_PROJECTION_ORIGIN: 0.0,
        Mapping.FALSE_EASTING: 500000.0,
        Mapping.FALSE_NORTHING: 10000000.0,
        Mapping.SCALE_FACTOR_AT_CENTRAL_MERIDIAN: 0.9996
    }
}

@ti.data_oriented
class TiUnitVector:
    """
    Class to compute unit vectors for transformation matrix. We are interested only
    in 2D unit vector as we ignore the Z component for the transformation.

    This is taichi (Python high performance computing package) implementation of
    the unit vector.
    """
    # Vector size of the unit vector
    SIZE = 2

    # Custom compile-time type to represent a unit vector as you can't define one
    # at runtime within the function
    VECTOR = ti.types.vector(SIZE, ti.f32)

    """
    """
    def __init__(self, n: int):
        # self.vector = ti.field(dtype=float, shape=(n, TiUnitVector.SIZE))
        self.data = ti.Vector.field(TiUnitVector.SIZE, dtype=ti.f32, shape=(n))
        self.dataLength = n

    @ti.kernel
    def compute(self, xy: ti.types.ndarray(), xy0: ti.types.ndarray()):
        """
        Compute 2d unit vector for each cell of the grid.

        Inputs:
        =======
        xy: Re-projected to original grid cell points (this is 3d vectors because
            re-projection code generates 3d output).
        xy0: Original grid cell points (this is 3d vectors) (this is 3d vectors because
            re-projection code generates 3d output).
        cell_size: Size of the grid cell (either X or Y dimension).
        """
        for i in range(self.dataLength):
            # Can't slice taichi arrays, have to pass values of the array slice explicitely
            # as separate values
            self.data[i][0] = xy[i, 0] - xy0[i, 0]
            self.data[i][1] = xy[i, 1] - xy0[i, 1]


@ti.data_oriented
class TiTransformMatrix:
    """
    Class to represent transformation matrix between two projections.

    This is taichi (Python high performance computing package) implementation of
    the unit vector.

    taichi supports aliases for default datatypes: int=ti.i32, float=ti.f32,
    so just use aliases.
    """
    def __init__(self, n: int):
        # Declares a matrix field of n-elements, each of its elements being a 2x2 matrix
        self.data = ti.Matrix.field(n=TiUnitVector.SIZE, m=TiUnitVector.SIZE, dtype=ti.f32, shape=(n))
        self.fill_data()

        self.angle = ti.field(dtype=ti.f32, shape=(n))
        # Initialize values
        _values = np.full(n, DataVars.MISSING_VALUE, dtype=np.float32)
        self.angle.from_numpy(_values)

        self.scale = ti.Vector.field(TiUnitVector.SIZE, dtype=ti.f32, shape=(n))
        # Initialize values
        _values = np.full((n, TiUnitVector.SIZE), DataVars.MISSING_VALUE, dtype=np.float32)
        self.scale.from_numpy(_values)

    @ti.kernel
    def fill_data(self):
        """
        Fill matrix data with initial values.
        """
        for i in ti.grouped(self.data):
            # self.data[i] is a 2x2 matrix
            self.data[i] = [
                [DataVars.MISSING_VALUE, DataVars.MISSING_VALUE],
                [DataVars.MISSING_VALUE, DataVars.MISSING_VALUE]
            ]


    @ti.kernel
    def compute(
        self,
        xunit_v: ti.template(),
        yunit_v: ti.template(),
        valid_indices: ti.types.ndarray(),
        original_ij_index: ti.types.ndarray(),
        v_all_values: ti.types.ndarray()
    ):
        """
        Compute transformation matrix for provided valid indices.
        """
        for i in valid_indices:
            # Find corresponding point in source P_in projection
            x_index = original_ij_index[i, 0]
            y_index = original_ij_index[i, 1]

            # Check if velocity is valid for the cell, if not then
            # don't compute the matrix
            v_value = v_all_values[y_index, x_index]

            if v_value != DataVars.MISSING_VALUE:
                xunit = xunit_v[i]
                yunit = yunit_v[i]

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
                denom_value = (yunit[0]*xunit[1] - yunit[1]*xunit[0])
                # self.data[each_index] is a 2x2 matrix
                self.data[i] = [
                    [-yunit[1]/denom_value, xunit[1]/denom_value],
                    [yunit[0]/denom_value, -xunit[0]/denom_value]
                ]

                # Compute angle and scale
                self.angle[i] = ti.atan2(self.data[i][1, 0], self.data[i][0, 0])

                theta_cos = ti.cos(self.angle[i])
                self.scale[i][0] = self.data[i][0, 0]/theta_cos
                self.scale[i][1] = self.data[i][1, 1]/theta_cos

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

    # Compression settings for storing data to NetCDF file
    COMPRESSION = {"zlib": True, "complevel": 2, "shuffle": True}

    NC_ENGINE = 'h5netcdf'

    INVALID_CELL_INDEX = -1

    def __init__(self, data, output_projection: int):
        """
        Initialize object.
        """
        self.ds = data
        self.input_file = None
        if isinstance(data, str):
            # Filename for the dataset is provided, read it in
            self.input_file = data
            self.ds = xr.open_dataset(data, engine=MosaicsReproject.NC_ENGINE, decode_timedelta=False)
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
            self.transformation_matrix_angle = None
            self.transformation_matrix_scale = None

            # Lists of valid cell indices for which transformation matrix is available
            self.valid_cell_indices_x = None
            self.valid_cell_indices_y = None

            # GDAL options to use for warping to new output grid
            self.warp_options_uint8 = None
            self.warp_options_uint16 = None
            self.warp_options_uint16_zero_missing_value = None
            self.warp_options_uint32 = None

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
            logging.info('Nothing to do.')

        # Flag if v0 is present in the mosaic, which indicates it's static mosaic
        is_static_mosaic = (CompDataVars.V0 in self.ds)
        if is_static_mosaic:
            self.create_transformation_matrix(CompDataVars.VX0, CompDataVars.VY0, CompDataVars.V0)
            self.mosaic_function = self.reproject_static_mosaic

        else:
            self.create_transformation_matrix(DataVars.VX, DataVars.VY, DataVars.V)
            self.mosaic_function = self.reproject_annual_mosaic

        # outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS

        self.warp_options_uint8 = gdal.WarpOptions(
            # format='netCDF',
            format=MosaicsReproject.WARP_FORMAT,
            outputBounds=(self.x0_bbox.min, self.y0_bbox.min, self.x0_bbox.max, self.y0_bbox.max),
            xRes=self.x_size,
            yRes=self.y_size,
            srcSRS=f'{self.ij_epsg_str}:{self.ij_epsg}',
            dstSRS=f'{self.xy_epsg_str}:{self.xy_epsg}',
            srcNodata=DataVars.MISSING_UINT8_VALUE,
            dstNodata=DataVars.MISSING_UINT8_VALUE,
            resampleAlg=gdal.GRA_NearestNeighbour,
            errorThreshold=MosaicsReproject.WARP_ET
        )

        self.warp_options_uint8_zero_missing_value = gdal.WarpOptions(
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

        self.warp_options_uint16 = gdal.WarpOptions(
            # format='netCDF',
            format=MosaicsReproject.WARP_FORMAT,
            outputBounds=(self.x0_bbox.min, self.y0_bbox.min, self.x0_bbox.max, self.y0_bbox.max),
            xRes=self.x_size,
            yRes=self.y_size,
            srcSRS=f'{self.ij_epsg_str}:{self.ij_epsg}',
            dstSRS=f'{self.xy_epsg_str}:{self.xy_epsg}',
            srcNodata=DataVars.MISSING_POS_VALUE,
            dstNodata=DataVars.MISSING_POS_VALUE,
            resampleAlg=gdal.GRA_NearestNeighbour,
            errorThreshold=MosaicsReproject.WARP_ET
        )

        self.warp_options_uint16_zero_missing_value = gdal.WarpOptions(
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

        self.warp_options_uint32 = gdal.WarpOptions(
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
            proj_attrs = {
                DataVars.GRID_MAPPING_NAME: 'polar_stereographic',
                'straight_vertical_longitude_from_pole': 0,
                'latitude_of_projection_origin': -90.0,
                'latitude_of_origin': -71.0,
                'scale_factor_at_projection_origin': 1,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_3031,
                'proj4text': "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            }

        elif self.xy_epsg == 3413:
            proj_attrs = {
                DataVars.GRID_MAPPING_NAME: 'polar_stereographic',
                'straight_vertical_longitude_from_pole': -45,
                'latitude_of_projection_origin': 90.0,
                'latitude_of_origin': 70.0,
                'scale_factor_at_projection_origin': 1,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_3413,
                'proj4text': "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            }

        elif self.xy_epsg == 102027:
            proj_attrs = {
                DataVars.GRID_MAPPING_NAME: 'lambert_conformal_conic',
                'CoordinateTransformType': 'Projection',
                'standard_parallel': (15.0, 65.0),
                'latitude_of_projection_origin': 30.0,
                'longitude_of_central_meridian': 95.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_102027,
                'proj4text': ESRICode_Proj4
            }
        elif self.xy_epsg == 8859:
            # gdalsrsinfo EPSG:8859

            # PROJ.4 : +proj=eqearth +lon_0=150 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs

            # OGC WKT2:2018 :
            # PROJCRS["WGS 84 / Equal Earth Asia-Pacific",
            #     BASEGEOGCRS["WGS 84",
            #         DATUM["World Geodetic System 1984",
            #             ELLIPSOID["WGS 84",6378137,298.257223563,
            #                 LENGTHUNIT["metre",1]]],
            #         PRIMEM["Greenwich",0,
            #             ANGLEUNIT["degree",0.0174532925199433]],
            #         ID["EPSG",4326]],
            #     CONVERSION["Equal Earth Asia-Pacific",
            #         METHOD["Equal Earth",
            #             ID["EPSG",1078]],
            #         PARAMETER["Longitude of natural origin",150,
            #             ANGLEUNIT["degree",0.0174532925199433],
            #             ID["EPSG",8802]],
            #         PARAMETER["False easting",0,
            #             LENGTHUNIT["metre",1],
            #             ID["EPSG",8806]],
            #         PARAMETER["False northing",0,
            #             LENGTHUNIT["metre",1],
            #             ID["EPSG",8807]]],
            #     CS[Cartesian,2],
            #         AXIS["(E)",east,
            #             ORDER[1],
            #             LENGTHUNIT["metre",1]],
            #         AXIS["(N)",north,
            #             ORDER[2],
            #             LENGTHUNIT["metre",1]],
            #     USAGE[
            #         SCOPE["Very small scale equal-area mapping - Asia-Pacific-centred."],
            #         AREA["World centred on Asia-Pacific."],
            #         BBOX[-90,-29.99,90,-30.01]],
            #     ID["EPSG",8859]]
            proj_attrs = {
                DataVars.GRID_MAPPING_NAME: 'equal_earth',
                'longitude_of_projection_origin': 150.0,
                'false_easting': 0.0,
                'false_northing': 0.0,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                'crs_wkt': spatial_ref_8859,
                'proj4text': '+proj=eqearth +lon_0=150 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
            }

        else:
            # Example of mapping for EPGS=32632:
            # string mapping ;
            #     string mapping:GeoTransform = "200032.5 120.0 0 5317327.5 0 -120.0" ;
            #     string mapping:crs_wkt = "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]" ;
            #     mapping:false_easting = 500000. ;
            #     mapping:false_northing = 0. ;
            #     string mapping:grid_mapping_name = "universal_transverse_mercator" ;
            #     mapping:inverse_flattening = 298.257223563 ;
            #     mapping:latitude_of_projection_origin = 0. ;
            #     mapping:longitude_of_central_meridian = 9. ;
            #     string mapping:proj4text = "+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs" ;
            #     mapping:scale_factor_at_central_meridian = 0.9996 ;
            #     mapping:semi_major_axis = 6378137. ;
            #     mapping:spatial_epsg = 32632LL ;
            #     string mapping:spatial_ref = "PROJCS[\"WGS 84 / UTM zone 32N\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"32632\"]]" ;
            #     mapping:utm_zone_number = 32. ;

            zone, spacial_ref_value = self.spatial_ref_32x()

            proj_attrs = {
                DataVars.GRID_MAPPING_NAME: 'universal_transverse_mercator',
                'utm_zone_number': zone,
                'semi_major_axis': 6378137.0,
                'inverse_flattening': 298.257223563,
                Mapping.CRS_WKT: spacial_ref_value,
                Mapping.SPATIAL_REF: spacial_ref_value,
                Mapping.PROJ4TEXT: f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
            }

            if self.xy_epsg in required_mapping_attributes:
                for each_attr, each_value in required_mapping_attributes[self.xy_epsg].items():
                    proj_attrs[each_attr] = each_value

            else:
                raise RuntimeError(f'Missing definition of mapping attributes for EPSG={self.xy_epsg}: please update required_mapping_attributes')

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
        reproject_ds[DataVars.MAPPING].attrs[Mapping.SPACIAL_EPSG] = self.xy_epsg

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
        ds_coords = [
            (CompDataVars.SENSORS, self.sensors, self.ds.sensor.attrs),
            (Coords.Y, self.y0_grid, self.ds.y.attrs),
            (Coords.X, self.x0_grid, self.ds.x.attrs),
        ]

        ds_coords_2d = [
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

        reproject_ds[CompDataVars.VX0] = xr.DataArray(
            data=vx0,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX0].attrs
        )

        reproject_ds[CompDataVars.VY0] = xr.DataArray(
            data=vy0,
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY0].attrs
        )

        # Convert v0 error variables to uint16 type
        reproject_ds[CompDataVars.V0_ERROR] = xr.DataArray(
            data=to_int_type(v0_error),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V0_ERROR].attrs
        )

        v0_error = None
        gc.collect()

        reproject_ds[CompDataVars.VX0_ERROR] = xr.DataArray(
            data=to_int_type(vx0_error),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX0_ERROR].attrs
        )

        vx0_error = None
        gc.collect()

        reproject_ds[CompDataVars.VY0_ERROR] = xr.DataArray(
            data=to_int_type(vy0_error),
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

        # No more need for vx0 and vy0, delete them
        vx0 = None
        gc.collect()

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
            data=to_int_type(vx_amp),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_AMP].attrs
        )
        vx_amp = None
        gc.collect()

        reproject_ds[CompDataVars.VY_AMP] = xr.DataArray(
            data=to_int_type(vy_amp),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_AMP].attrs
        )
        vy_amp = None
        gc.collect()

        reproject_ds[CompDataVars.V_AMP] = xr.DataArray(
            data=to_int_type(v_amp),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_AMP].attrs
        )
        v_amp = None
        gc.collect()

        reproject_ds[CompDataVars.VX_AMP_ERROR] = xr.DataArray(
            data=to_int_type(vx_amp_error),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_AMP_ERROR].attrs
        )
        vx_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.VY_AMP_ERROR] = xr.DataArray(
            data=to_int_type(vy_amp_error),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_AMP_ERROR].attrs
        )
        vy_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.V_AMP_ERROR] = xr.DataArray(
            data=to_int_type(v_amp_error),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_AMP_ERROR].attrs
        )
        v_amp_error = None
        gc.collect()

        reproject_ds[CompDataVars.VX_PHASE] = xr.DataArray(
            data=to_int_type(vx_phase),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VX_PHASE].attrs
        )
        vx_phase = None
        gc.collect()

        reproject_ds[CompDataVars.VY_PHASE] = xr.DataArray(
            data=to_int_type(vy_phase),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.VY_PHASE].attrs
        )
        vy_phase = None
        gc.collect()

        reproject_ds[CompDataVars.V_PHASE] = xr.DataArray(
            data=to_int_type(v_phase),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.V_PHASE].attrs
        )
        v_phase = None
        gc.collect()

        # Warp "count0" variable
        warp_data = self.warp_var(CompDataVars.COUNT0, self.warp_options_uint32)
        reproject_ds[CompDataVars.COUNT0] = xr.DataArray(
            data=to_int_type(warp_data, np.uint32, DataVars.MISSING_BYTE),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.COUNT0].attrs
        )
        warp_data = None
        gc.collect()

        self.set_mapping(reproject_ds)

        # Warp "dt_max" variable: per each sensor dimension
        warp_data = self.warp_var(CompDataVars.MAX_DT, self.warp_options_uint16_zero_missing_value)

        if warp_data.ndim == 2:
            # If warped data is 2d array
            _y_dim, _x_dim = warp_data.shape

            # Convert to 3d array as MAX_DT is 3d data (has sensor dimension)
            warp_data = warp_data.reshape((1, _y_dim, _x_dim))

        if MosaicsReproject.VERBOSE:
            _values = self.ds[CompDataVars.MAX_DT].values
            verbose_mask = np.isfinite(_values)
            if np.sum(verbose_mask) == 0:
                logging.info(f'Original {CompDataVars.MAX_DT}: no valid data')

            else:
                logging.info(f"Original {CompDataVars.MAX_DT}:  min={np.nanmin(_values[verbose_mask])} max={np.nanmax(_values[verbose_mask])}")

            verbose_mask = np.isfinite(warp_data) & (warp_data != DataVars.MISSING_POS_VALUE)
            if np.sum(verbose_mask) == 0:
                logging.info(f'Warped {CompDataVars.MAX_DT}: no valid data')

            else:
                logging.info(f"Warped {CompDataVars.MAX_DT}:  min={np.nanmin(warp_data[verbose_mask])} max={np.nanmax(warp_data[verbose_mask])}")

        reproject_ds[CompDataVars.MAX_DT] = xr.DataArray(
            data=to_int_type(warp_data, fill_value=DataVars.MISSING_BYTE),
            coords=ds_coords,
            attrs=self.ds[CompDataVars.MAX_DT].attrs
        )
        warp_data = None
        gc.collect()

        # Warp "landice" variable (check for variable existence if older mosaics)
        if ShapeFile.LANDICE in self.ds:
            is_binary_data = True
            warp_data = self.warp_var(
                ShapeFile.LANDICE,
                self.warp_options_uint8_zero_missing_value,
                is_binary_data
            )
            reproject_ds[ShapeFile.LANDICE] = xr.DataArray(
                data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_BYTE),
                coords=ds_coords_2d,
                attrs=self.ds[ShapeFile.LANDICE].attrs
            )
            warp_data = None
            gc.collect()

        if ShapeFile.FLOATINGICE in self.ds:
            is_binary_data = True
            warp_data = self.warp_var(
                ShapeFile.FLOATINGICE,
                self.warp_options_uint8_zero_missing_value,
                is_binary_data
            )
            reproject_ds[ShapeFile.FLOATINGICE] = xr.DataArray(
                data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_BYTE),
                coords=ds_coords_2d,
                attrs=self.ds[ShapeFile.FLOATINGICE].attrs
            )
            warp_data = None
            gc.collect()

        # Warp "outlier_frac" variable: per each sensor dimension
        warp_data = self.warp_var(CompDataVars.OUTLIER_FRAC, self.warp_options_uint8)
        reproject_ds[CompDataVars.OUTLIER_FRAC] = xr.DataArray(
            data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_UINT8_VALUE),
            coords=ds_coords_2d,
            attrs=self.ds[CompDataVars.OUTLIER_FRAC].attrs
        )
        warp_data = None
        gc.collect()

        # Warp "sensor_flag" variable: per each sensor dimension
        if CompDataVars.SENSOR_INCLUDE in self.ds:
            is_binary_data = True
            # This is workaround for missing variable in original mosaics code
            # so can test the code with originally generated small test sets
            warp_data = self.warp_var(
                CompDataVars.SENSOR_INCLUDE,
                self.warp_options_uint8,
                is_binary_data
            )

            if warp_data.ndim == 2:
                # If warped data is 2d array
                _y_dim, _x_dim = warp_data.shape

                # Convert to 3d array as SENSOR_INCLUDE is 3d data (has sensor dimension)
                warp_data = warp_data.reshape((1, _y_dim, _x_dim))

            reproject_ds[CompDataVars.SENSOR_INCLUDE] = xr.DataArray(
                data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_UINT8_VALUE),
                coords=ds_coords,
                attrs=self.ds[CompDataVars.SENSOR_INCLUDE].attrs
            )

            if MosaicsReproject.VERBOSE:
                _values = self.ds[CompDataVars.SENSOR_INCLUDE].values
                verbose_mask = np.isfinite(_values)
                logging.info(f"Original {CompDataVars.SENSOR_INCLUDE}:  min={np.nanmin(_values[verbose_mask])} max={np.nanmax(_values[verbose_mask])}")

                verbose_mask = np.isfinite(warp_data)
                logging.info(f"gdal.warp(): Original {CompDataVars.SENSOR_INCLUDE}:  min={np.nanmin(warp_data[verbose_mask])} max={np.nanmax(warp_data[verbose_mask])}")

            warp_data = None
            gc.collect()

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
            valid_mask = np.where((~np.isnan(vx)) & (~np.isnan(vy)) & (~np.isnan(v)) & (v != 0))

            v_error_verify = np.full_like(v_error, np.nan, dtype=np.float32)
            v_error_verify[valid_mask] = (vx_error[valid_mask]*np.abs(vx[valid_mask]) + vy_error[valid_mask]*np.abs(vy[valid_mask]))/v[valid_mask]

        # Create new granule in target projection
        ds_coords = [
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
            data=to_int_type(v_error),
            coords=ds_coords,
            attrs=self.ds[CompDataVars.V_ERROR].attrs
        )

        v_error = None
        gc.collect()

        # Add vx_error to dataset
        reproject_ds[CompDataVars.VX_ERROR] = xr.DataArray(
            data=to_int_type(vx_error),
            coords=ds_coords,
            attrs=self.ds[CompDataVars.VX_ERROR].attrs
        )

        vx_error = None
        gc.collect()

        # Add vy_error to dataset
        reproject_ds[CompDataVars.VY_ERROR] = xr.DataArray(
            data=to_int_type(vy_error),
            coords=ds_coords,
            attrs=self.ds[CompDataVars.VY_ERROR].attrs
        )

        vy_error = None
        gc.collect()

        # Warp "landice" variable (check for variable existence in older mosaics)
        if ShapeFile.LANDICE in self.ds:
            is_binary_data = True
            warp_data = self.warp_var(
                ShapeFile.LANDICE,
                self.warp_options_uint8_zero_missing_value,
                is_binary_data
            )
            reproject_ds[ShapeFile.LANDICE] = xr.DataArray(
                data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_BYTE),
                coords=ds_coords,
                attrs=self.ds[ShapeFile.LANDICE].attrs
            )
            warp_data = None
            gc.collect()

        # Warp "floatingice" variable (check for variable existence in older mosaics)
        if ShapeFile.FLOATINGICE in self.ds:
            is_binary_data = True
            warp_data = self.warp_var(
                ShapeFile.FLOATINGICE,
                self.warp_options_uint8_zero_missing_value,
                is_binary_data
            )
            reproject_ds[ShapeFile.FLOATINGICE] = xr.DataArray(
                data=to_int_type(warp_data, np.uint8, fill_value=DataVars.MISSING_BYTE),
                coords=ds_coords,
                attrs=self.ds[ShapeFile.FLOATINGICE].attrs
            )
            warp_data = None
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
        warp_data = self.warp_var(CompDataVars.COUNT, self.warp_options_uint32)
        reproject_ds[CompDataVars.COUNT] = xr.DataArray(
            data=to_int_type(warp_data, np.uint32, DataVars.MISSING_BYTE),
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

        # Disable FillValue for coordinates
        for each in [Coords.X, Coords.Y]:
            encoding_settings[each] = {Output.FILL_VALUE_ATTR: None}

        float_vars = [
            DataVars.V,
            DataVars.VX,
            DataVars.VY
        ]

        int_vars = [
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR
        ]

        if MosaicsReproject.COMPUTE_DEBUG_VARS:
            # Handle debug variables, if any, automatically
            debug_vars = [CompDataVars.V_ERROR+'_verify', 'vx_xunit', 'vy_xunit', 'vx_yunit', 'vy_yunit']

            for each in debug_vars:
                if each in ds:
                    float_vars.append(each)

        two_dim_chunks_settings = (ds.y.size, ds.x.size)

        # Explicitly set dtype for some variables
        for each in float_vars:
            encoding_settings[each] = {
                Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                Output.DTYPE_ATTR: np.float32,
                Output.CHUNKSIZES_ATTR: two_dim_chunks_settings
            }
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

        # Explicitly set dtype and missing_value for some variables
        for each in int_vars:
            encoding_settings[each] = {
                Output.MISSING_VALUE_ATTR: DataVars.MISSING_POS_VALUE,
                Output.DTYPE_ATTR: np.uint16,
                Output.CHUNKSIZES_ATTR: two_dim_chunks_settings
            }
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

            if Output.MISSING_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.MISSING_VALUE_ATTR]

        # Settings for variable of "uint8" data type
        for each in [
            ShapeFile.LANDICE,
            ShapeFile.FLOATINGICE
        ]:
            # Support older mosaics which might not have these variables
            if each not in ds:
                continue

            encoding_settings.setdefault(each, {}).update({
                Output.DTYPE_ATTR: np.uint8,
                Output.MISSING_VALUE_ATTR: DataVars.MISSING_BYTE,
                Output.CHUNKSIZES_ATTR: two_dim_chunks_settings
            })
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

            if Output.MISSING_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.MISSING_VALUE_ATTR]

        # Set encoding for 'count' data variable
        encoding_settings[CompDataVars.COUNT] = {
            Output.MISSING_VALUE_ATTR: DataVars.MISSING_BYTE,
            Output.DTYPE_ATTR: np.uint32,
            Output.CHUNKSIZES_ATTR: two_dim_chunks_settings
        }
        encoding_settings[CompDataVars.COUNT].update(MosaicsReproject.COMPRESSION)

        if Output.FILL_VALUE_ATTR in ds[CompDataVars.COUNT].attrs:
            del ds[CompDataVars.COUNT].attrs[Output.FILL_VALUE_ATTR]

        if Output.MISSING_VALUE_ATTR in ds[CompDataVars.COUNT].attrs:
            del ds[CompDataVars.COUNT].attrs[Output.MISSING_VALUE_ATTR]

        logging.info(f'Enconding for {output_file}: {encoding_settings}')

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding_settings)

    @staticmethod
    def write_static_to_netCDF(ds, output_file: str):
        """
        Write static mosaic dataset to the netCDF format file.
        """
        if output_file is None:
            # Output filename is not provided, don't write to the file
            return

        encoding_settings = {}

        # Disable FillValue for coordinates
        for each in [Coords.X, Coords.Y, CompDataVars.SENSORS]:
            encoding_settings[each] = {Output.FILL_VALUE_ATTR: None}

        two_dim_chunks_settings = (ds.y.size, ds.x.size)
        three_dim_chunks_settings = (1, ds.y.size, ds.x.size)

        float_vars = [
            CompDataVars.VX0,
            CompDataVars.VY0,
            CompDataVars.V0,
            CompDataVars.SLOPE_VX,
            CompDataVars.SLOPE_VY,
            CompDataVars.SLOPE_V
        ]

        int_vars = [
            CompDataVars.VX_AMP_ERROR,
            CompDataVars.VY_AMP_ERROR,
            CompDataVars.V_AMP_ERROR,
            CompDataVars.VX_AMP,
            CompDataVars.VY_AMP,
            CompDataVars.V_AMP,
            CompDataVars.VX_PHASE,
            CompDataVars.VY_PHASE,
            CompDataVars.V_PHASE,
            CompDataVars.VX0_ERROR,
            CompDataVars.VY0_ERROR,
            CompDataVars.V0_ERROR,
            CompDataVars.MAX_DT
        ]

        # Explicitly set dtype for some variables
        for each in float_vars:
            _chunks = two_dim_chunks_settings

            if ds[each].ndim == 3:
                _chunks = three_dim_chunks_settings

            encoding_settings[each] = {
                Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                Output.DTYPE_ATTR: np.float32,
                Output.CHUNKSIZES_ATTR: _chunks
            }
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

        for each in int_vars:
            _chunks = two_dim_chunks_settings

            if ds[each].ndim == 3:
                _chunks = three_dim_chunks_settings

            _missing_value = DataVars.MISSING_POS_VALUE
            if each in [
                CompDataVars.MAX_DT
            ]:
                _missing_value = DataVars.MISSING_BYTE

            encoding_settings[each] = {
                Output.MISSING_VALUE_ATTR: _missing_value,
                Output.DTYPE_ATTR: np.uint16,
                Output.CHUNKSIZES_ATTR: _chunks
            }
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

            if Output.MISSING_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.MISSING_VALUE_ATTR]

        # Settings for variable of "uint8" data type
        for each in [
            CompDataVars.OUTLIER_FRAC,
            CompDataVars.SENSOR_INCLUDE,
            ShapeFile.LANDICE,
            ShapeFile.FLOATINGICE
        ]:
            if each not in ds:
                continue

            _chunks = two_dim_chunks_settings

            if ds[each].ndim == 3:
                _chunks = three_dim_chunks_settings

            # ice masks should use 0 as missing_value
            _missing_value = DataVars.MISSING_UINT8_VALUE
            if each in [
                ShapeFile.LANDICE,
                ShapeFile.FLOATINGICE
            ]:
                _missing_value = DataVars.MISSING_BYTE

            encoding_settings.setdefault(each, {}).update({
                Output.DTYPE_ATTR: np.uint8,
                Output.MISSING_VALUE_ATTR: _missing_value,
                Output.CHUNKSIZES_ATTR: _chunks
            })
            encoding_settings[each].update(MosaicsReproject.COMPRESSION)

            if Output.FILL_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.FILL_VALUE_ATTR]

            if Output.MISSING_VALUE_ATTR in ds[each].attrs:
                del ds[each].attrs[Output.MISSING_VALUE_ATTR]

        # Set encoding for 'count0' data variable
        encoding_settings[CompDataVars.COUNT0] = {
            Output.MISSING_VALUE_ATTR: DataVars.MISSING_BYTE,
            Output.DTYPE_ATTR: np.uint32,
            Output.CHUNKSIZES_ATTR: two_dim_chunks_settings
        }
        encoding_settings[CompDataVars.COUNT0].update(MosaicsReproject.COMPRESSION)

        if Output.FILL_VALUE_ATTR in ds[CompDataVars.COUNT0].attrs:
            del ds[CompDataVars.COUNT0].attrs[Output.FILL_VALUE_ATTR]

        if Output.MISSING_VALUE_ATTR in ds[CompDataVars.COUNT0].attrs:
            del ds[CompDataVars.COUNT0].attrs[Output.MISSING_VALUE_ATTR]

        logging.info(f'Enconding for {output_file}: {encoding_settings}')

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding=encoding_settings)

    def warp_var(self, var: str, warp_options: gdal.WarpOptions, is_binary_data: bool = False):
        """
        Warp variable into new projection.

        Inputs:
        =======
        var: Name of the variable to warp
        warp_options: gdal.WarpOptions object to use for warping
        is_binary_data: Flag if the data is binary, and should enforce 0|1 values
            for warped data.
        """
        np_ds = gdal.Warp('', f'NETCDF:"{self.input_file}":{var}', options=warp_options).ReadAsArray()

        if is_binary_data:
            # Make sure the mask is of 0/1 values
            warp_data_mask = (np_ds > 0) & (np_ds != DataVars.MISSING_UINT8_VALUE)
            np_ds[warp_data_mask] = 1

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(np_ds)
            # In case of the binary data missing_value=255,
            # so take this into account when reporting min/max values
            if is_binary_data:
                verbose_mask &= (np_ds != DataVars.MISSING_UINT8_VALUE)

            logging.info(f"Warped {var}:  min={np.nanmin(np_ds[verbose_mask])} max={np.nanmax(np_ds[verbose_mask])}")

        return np_ds

    def reproject_velocity(
        self,
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
        _vx[_vx == DataVars.MISSING_VALUE] = np.nan

        # Read Y component of variable
        _vy = self.ds[vy_var].values
        _vy[_vy == DataVars.MISSING_VALUE] = np.nan

        # Read original velocity values
        _v = self.ds[v_var].values
        _v[_v == DataVars.MISSING_VALUE] = np.nan

        # Read original error values in
        _v_error = self.ds[v_error_var].values
        _v_error[_v_error == DataVars.MISSING_POS_VALUE] = np.nan

        # Read X component of v_error
        _vx_error = self.ds[vx_error_var].values
        _vx_error[_vx_error == DataVars.MISSING_POS_VALUE] = np.nan

        # Read Y component of the error
        _vy_error = self.ds[vy_error_var].values
        _vy_error[_vy_error == DataVars.MISSING_POS_VALUE] = np.nan

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
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {vx_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {vx_var}:  min={np.nanmin(_vx[verbose_mask])} max={np.nanmax(_vx[verbose_mask])}")

            verbose_mask = np.isfinite(_vy)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {vy_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {vy_var}:  min={np.nanmin(_vy[verbose_mask])} max={np.nanmax(_vy[verbose_mask])}")

            verbose_mask = np.isfinite(_v)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {v_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {v_var}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

            verbose_mask = np.isfinite(_vx_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {vx_error_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {vx_error_var}: min={np.nanmin(_vx_error[verbose_mask])} max={np.nanmax(_vx_error[verbose_mask])}")

            verbose_mask = np.isfinite(_vy_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {vy_error_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {vy_error_var}: min={np.nanmin(_vy_error[verbose_mask])} max={np.nanmax(_vy_error[verbose_mask])}")

            verbose_mask = np.isfinite(_v_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Original {v_error_var}: no valid data")

            else:
                logging.info(f"reproject_velocity: Original {v_error_var}: min={np.nanmin(_v_error[verbose_mask])} max={np.nanmax(_v_error[verbose_mask])}")

        # debug_y = 7915
        # debug_x = 3701

        # debug_esri_index = 54748893
        # debug_esri_x = 4269
        # debug_esri_y = 7752

        for y, x in tqdm(
            zip(self.valid_cell_indices_y, self.valid_cell_indices_x),
            ascii=True,
            desc=f"Re-projecting {vx_var}, {vy_var}, {vx_error_var}, {vy_error_var}..."
        ):
            t_matrix = self.transformation_matrix[y, x]

            # Look up original cell in input ij-projection
            i, j = self.original_ij_index[y, x]

            # Re-project velocity variables
            dv = [_vx[j, i], _vy[j, i]]

            # if y == debug_esri_y and x == debug_esri_x:
            #     logging.info(f'--->DEBUG_HMA (reproject {vx_var}, {vy_var}: i={i} j={j} based on x={x} y={y}')
            #     logging.info(f'--->DEBUG_HMA (reproject {vx_var}, {vy_var}: t_matrix={t_matrix}')
            #     logging.info(f'--->DEBUG_HMA (reproject {vx_var}, {vy_var}: dv={dv}')

            # Some points get NODATA for vx but valid vy and vx.
            if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
                # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                xy_v = np.matmul(t_matrix, dv)

                vx[y, x] = xy_v[0]
                vy[y, x] = xy_v[1]
                # if y == debug_esri_y and x == debug_esri_x:
                #     logging.info(f'--->DEBUG_HMA (reproject {vx_var}, {vy_var}: reprojected vx={vx[y, x]} vy={vy[y, x]}')

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

                # if v_ij_value and (not np.any(np.isnan(_v_error[j, i]))):
                if v_ij_value and (not math.isnan(_v_error[j, i])):
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
                # if not np.any(np.isnan(dv)):
                if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
                    # vx_error and vy_error must be positive:
                    # use absolute values of transformation matrix to avoid
                    # negative re-projected vx_error and vy_error values
                    vx_error[y, x], vy_error[y, x] = np.matmul(np.abs(t_matrix), dv)

        if MosaicsReproject.VERBOSE:
            verbose_mask = np.isfinite(vx)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {vx_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {vx_var}:  min={np.nanmin(vx[verbose_mask])} max={np.nanmax(vx[verbose_mask])}")

            verbose_mask = np.isfinite(vy)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {vy_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {vy_var}:  min={np.nanmin(vy[verbose_mask])} max={np.nanmax(vy[verbose_mask])}")

            verbose_mask = np.isfinite(v)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {v_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {v_var}:  min={np.nanmin(v[verbose_mask])} max={np.nanmax(v[verbose_mask])}")

            verbose_mask = np.isfinite(vx_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {vx_error_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {vx_error_var}:  min={np.nanmin(vx_error[verbose_mask])} max={np.nanmax(vx_error[verbose_mask])}")

            verbose_mask = np.isfinite(vy_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {vy_error_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {vy_error_var}:  min={np.nanmin(vy_error[verbose_mask])} max={np.nanmax(vy_error[verbose_mask])}")

            verbose_mask = np.isfinite(v_error)
            if np.sum(verbose_mask) == 0:
                logging.info(f"reproject_velocity: Re-projected {v_error_var}: no valid values")

            else:
                logging.info(f"reproject_velocity: Re-projected {v_error_var}:  min={np.nanmin(v_error[verbose_mask])} max={np.nanmax(v_error[verbose_mask])}")

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
        v0 = np.sqrt(vx0**2 + vy0**2)  # velocity magnitude
        uv_x = vx0/v0  # unit flow vector in x direction
        uv_y = vy0/v0  # unit flow vector in y direction

        # Read X component of dv_dt
        _dvx_dt = self.ds[CompDataVars.SLOPE_VX].values

        # Read Y component of dv_dt
        _dvy_dt = self.ds[CompDataVars.SLOPE_VY].values

        # Read X component of v_phase
        _vx_phase = self.ds[CompDataVars.VX_PHASE].values

        # Read Y component of v_phase
        _vy_phase = self.ds[CompDataVars.VY_PHASE].values

        # Read v_phase
        # _v_phase = self.ds[CompDataVars.V_PHASE].values
        _v_amp = self.ds[CompDataVars.V_AMP].values

        # Read X component of v_amp
        _vx_amp = self.ds[CompDataVars.VX_AMP].values

        # Read Y component of v_amp
        _vy_amp = self.ds[CompDataVars.VY_AMP].values

        # Read X component of v_amp_error
        _vx_amp_error = self.ds[CompDataVars.VX_AMP_ERROR].values

        # Read Y component of v_amp_error
        _vy_amp_error = self.ds[CompDataVars.VY_AMP_ERROR].values

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
            verbose_mask = np.isfinite(_v)

            # Report min and max values for the error variable
            logging.info(f"reproject_static_vars: Original {CompDataVars.SLOPE_V}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

            # Read original v_phase values
            _v = None
            _v = self.ds[CompDataVars.V_PHASE].values
            verbose_mask = np.isfinite(_v)

            # Report min and max values for the error variable
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_PHASE}: min={np.nanmin(_v[verbose_mask])} max={np.nanmax(_v[verbose_mask])}")

        # for y in tqdm(range(num_y), ascii=True, desc=f"Re-projecting {CompDataVars.SLOPE_V}, {CompDataVars.V_AMP}, {CompDataVars.V_AMP_ERROR}, {CompDataVars.V_PHASE}..."):
        #     for x in range(num_x):
        for y, x in tqdm(
            zip(self.valid_cell_indices_y, self.valid_cell_indices_x),
            ascii=True,
            desc=f"Re-projecting {CompDataVars.SLOPE_V}, {CompDataVars.V_AMP}, {CompDataVars.V_AMP_ERROR}, {CompDataVars.V_PHASE}..."
        ):
            t_matrix = self.transformation_matrix[y, x]

            # Look up original cell in input ij-projection
            i, j = self.original_ij_index[y, x]

            # Re-project dv_dt's X and Y components
            dv = [_dvx_dt[j, i], _dvy_dt[j, i]]

            # Some points get NODATA for vx but valid vy and v.
            # if np.all(np.isfinite(dv)):
            if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
                # Apply transformation matrix to (dvx_dt, dvy_dt) vector
                dvx_dt[y, x], dvy_dt[y, x] = np.matmul(t_matrix, dv)

            # Populate v_amp's and v_phase's X and Y components with original values,
            # they will be overwritten by reprojection
            vx_phase[y, x] = _vx_phase[j, i]
            vy_phase[y, x] = _vy_phase[j, i]
            vx_amp[y, x] = _vx_amp[j, i]
            vy_amp[y, x] = _vy_amp[j, i]

            # dv = [_vx_amp[j, i], _vy_amp[j, i]]

            # # Some points get NODATA for vx but valid vy and v.
            # # if np.all(np.isfinite(dv)):
            # if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
            #     # Apply transformation matrix to (vx, vy) values
            #     vx_amp[y, x], vy_amp[y, x] = np.matmul(np.abs(t_matrix), dv)

            # Re-project v_amp_error's components
            dv = [_vx_amp_error[j, i], _vy_amp_error[j, i]]

            # If any of the values is NODATA, don't re-project, leave them as NODATA
            # if np.all(np.isfinite(dv)):
            if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
                # vx_error and vy_error must be positive:
                # use absolute values of transformation matrix to avoid
                # negative re-projected vx_error and vy_error values
                vx_amp_error[y, x], vy_amp_error[y, x] = np.matmul(np.abs(t_matrix), dv)

            # # Re-project v_phase's components
            # dv = [_vx_phase[j, i], _vy_phase[j, i]]

            # # If any of the values is NODATA, don't re-project, leave them as NODATA
            # # if np.all(np.isfinite(dv)):
            # if (not math.isnan(dv[0])) and (not math.isnan(dv[1])):
            #     # vx_phase[y, x], vy_phase[y, x] = np.matmul(np.abs(t_matrix), dv)
            #     vx_phase[y, x], vy_phase[y, x] = np.matmul(t_matrix, dv)

        # DEBUG: indices into the problem cell
        # dx=369
        # dy=683
        # di=197
        # dj=726

        # logging.info(f'self.transformation_matrix[y, x]')
        # logging.info(f'GOT_ERROR v_amp_error>200: x={dx} y={dy} _vx_amp={_vx_amp[dj, di]} _vy_amp={_vy_amp[dj, di]}')
        # logging.info(f'GOT_ERROR v_amp_error>200: x={dx} y={dy} _vx_phase={_vx_phase[dj, di]} _vy_phase={_vy_phase[dj, di]}')
        # logging.info(f'GOT_ERROR v_amp_error>200: x={dx} y={dy} _v_phase={_v_phase[dj, di]} _v_amp={_v_amp[dj, di]}')


        # No need for some of original data, cleanup
        _dvx_dt = None
        _dvy_dt = None
        _vx_amp = None
        _vy_amp = None
        gc.collect()

        # Compute dv_dt: flow acceleration in direction of unit flow vector
        dv_dt = dvx_dt * uv_x
        dv_dt += dvy_dt * uv_y

        # Was for point with rotation matrix of positive B1
        # dx=468
        # dy=372
        # di=226
        # dj=368

        # Rotate v_phase and v_amp using analytical solution:
        # - theta is rotation matrix as derived from the transformation matrix
        # - vx_amp and vy_amp components are scaled by factors as derived from the transformation matrix
        vx_phase_r, vy_phase_r, vx_amp_r, vy_amp_r = MosaicsReproject.seasonal_velocity_rotation(
            self.transformation_matrix_angle,
            vx_phase, vy_phase,
            self.transformation_matrix_scale[:, :, 0]*vx_amp,
            self.transformation_matrix_scale[:, :, 1]*vy_amp
        )

        # Now rotate in the flow direction determined by vx0 and vy0
        v_phase, v_amp = MosaicsReproject.seasonal_velocity_rotation_x_term(vx0, vy0, vx_phase_r, vy_phase_r, vx_amp_r, vy_amp_r)

        # Compute v_amp_error using scale factor b/w old and newly re-projected v_amp values
        # (don't project v_amp_error in direction of unit flow vector
        # like in composites)
        # Scale the "v_amp_error" as new "v_amp" is computed now
        _v_amp = self.ds[CompDataVars.V_AMP].values
        _v_amp_error = self.ds[CompDataVars.V_AMP_ERROR].values

        if MosaicsReproject.VERBOSE:
            # Report min and max values for the error variable
            verbose_mask = np.isfinite(_v_amp)
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_AMP}: min={np.nanmin(_v_amp[verbose_mask])} max={np.nanmax(_v_amp[verbose_mask])}")

            verbose_mask = np.isfinite(_v_amp_error)
            logging.info(f"reproject_static_vars: Original {CompDataVars.V_AMP_ERROR}: min={np.nanmin(_v_amp_error[verbose_mask])} max={np.nanmax(_v_amp_error[verbose_mask])}")

        # for y in tqdm(range(num_y), ascii=True, desc=f"Scaling {CompDataVars.V_AMP_ERROR}..."):
        #     for x in range(num_x):
        for y, x in tqdm(
            zip(self.valid_cell_indices_y, self.valid_cell_indices_x),
            ascii=True,
            desc=f"Scaling {CompDataVars.V_AMP_ERROR}..."
        ):
            v_amp_value = v_amp[y, x]
            if math.isnan(v_amp_value):
                continue

            # Look up original cell in input ij-projection
            i, j = self.original_ij_index[y, x]

            # Look up original velocity value to compute the scale factor
            # for v_error: scale_factor = v_old / v_new
            v_ij_value = _v_amp[j, i]

            scale_factor = 1.0
            if v_ij_value != 0:
                scale_factor = v_amp_value/v_ij_value

            # if v_ij_value and (not np.any(np.isnan(_v_amp_error[j, i]))):
            if (not math.isnan(v_ij_value)) and (not math.isnan(_v_amp_error[j, i])):
                # Apply scale factor to the error value
                v_amp_error[y, x] = _v_amp_error[j, i]*scale_factor

            if math.isnan(v_amp_error[y, x]) and ~math.isnan(v_amp[y, x]):
                logging.info(f'GOT_NAN_VAMP_ERROR: x={x} y={y} v_amp[y, x]={v_amp_value} v_ij_value={v_ij_value}')

        if MosaicsReproject.VERBOSE:
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_VX}:  min={np.nanmin(dvx_dt)} max={np.nanmax(dvx_dt)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_VY}:  min={np.nanmin(dvy_dt)} max={np.nanmax(dvy_dt)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.SLOPE_V}:  min={np.nanmin(dv_dt)} max={np.nanmax(dv_dt)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_AMP}:  min={np.nanmin(vx_amp)} max={np.nanmax(vx_amp)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_AMP}:  min={np.nanmin(vy_amp)} max={np.nanmax(vy_amp)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_AMP}:  min={np.nanmin(v_amp)} max={np.nanmax(v_amp)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_AMP_ERROR}:  min={np.nanmin(vx_amp_error)} max={np.nanmax(vx_amp_error)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_AMP_ERROR}:  min={np.nanmin(vy_amp_error)} max={np.nanmax(vy_amp_error)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_AMP_ERROR}:  min={np.nanmin(v_amp_error)} max={np.nanmax(v_amp_error)}")

            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VX_PHASE}:  min={np.nanmin(vx_phase)} max={np.nanmax(vx_phase)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.VY_PHASE}:  min={np.nanmin(vy_phase)} max={np.nanmax(vy_phase)}")
            logging.info(f"reproject_static_vars: Re-projected {CompDataVars.V_PHASE}:  min={np.nanmin(v_phase)} max={np.nanmax(v_phase)}")

        return (dvx_dt, dvy_dt, dv_dt, vx_amp, vy_amp, v_amp, vx_amp_error, vy_amp_error, v_amp_error, vx_phase, vy_phase, v_phase)

    @staticmethod
    def wrap_amp_phase(v_phase, v_amp):
        """
        Wrap phase and amplitude to be within valid ranges.

        Args:
        v_phase: Input phase in degrees.
        v_amp: Input amplitude.

        Ouputs:
        v_phase, v_amp: Wrap-corrected input values.

        """
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
    def seasonal_velocity_rotation(theta, vx_phase, vy_phase, vx_amp, vy_amp):
        """
        Rotate v_phase and v_amp given "theta" rotation angle and already scaled
        vx_amp and vy_amp (according to the transformation matrix).

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
        vx_phase_r (doy) - day of maximum velocity in original coordinate system
        vy_phase_r (doy) - day of maximum velocity in original coordinate system
        vx_amp_r (m/yr) - x component of seasonal amplitude in the original coordinate system
        vy_amp_r (m/yr) - y component of seasonal amplitude in the original coordinate system
        """
        _two_pi = np.pi * 2

        # Matlab prototype code:
        # % Convert phase values from day-of-year to degrees:
        # vx_phase_deg = vx_phase*360/365.24;
        # vy_phase_deg = vy_phase*360/365.24;
        # Avoid conversion to degrees - go from day-of-year to radians

        vx_phase_deg = vx_phase/365.24
        vy_phase_deg = vy_phase/365.24

        # Don't use np.nan values in calculations to avoid warnings
        valid_mask = (~np.isnan(vx_phase_deg)) & (~np.isnan(vy_phase_deg))

        # Convert degrees to radians as numpy trig. functions take angles in radians
        # Explanation: if skipping *360.0 in vx_phase_deg, vy_phase_deg above,
        # then to convert to radians: *np.pi/180 --> 360*np.py/180 = 2*np.pi
        vx_phase_deg *= _two_pi
        vy_phase_deg *= _two_pi

        # Matlab prototype code:
        # % Rotation matrix for x component:
        # A1 =  vx_amp.*cosd(theta);
        # B1 = -vy_amp.*sind(theta);

        if np.any(theta < 0):
            # logging.info(f'Got negative theta, converting to positive values')
            mask = (theta < 0)
            theta[mask] += _two_pi

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # New in Python: assume clockwise rotation by theta as we need to align
        # vector with v0 direction. Therefore  use clockwise transformation matrix
        # for rotation, not counter-clockwise as in Matlab prototype code.
        A1 = vx_amp*cos_theta
        B1 = -vy_amp*sin_theta
        # Matlab prototype:
        # B1 = -vy_amp*sin_theta

        # Matlab prototype:
        # A2 = vx_amp*sin_theta
        # B2 = vy_amp*cos_theta
        A2 = vx_amp*sin_theta
        B2 = vy_amp*cos_theta

        # Matlab prototype code:
        # % Rotation matrix for x component:
        # vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
        # vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

        # % Rotation matrix for y component:
        # A2 = vx_amp.*sind(theta);
        # B2 = vy_amp.*cosd(theta);
        # vy_amp_r   =   hypot(A2.*cosd(vx_phase_deg) + B2.*cosd(vy_phase_deg),  A2.*sind(vx_phase_deg) + B2.*sind(vy_phase_deg));
        # vy_phase_r = atan2d((A2.*sind(vx_phase_deg) + B2.*sind(vy_phase_deg)),(A2.*cosd(vx_phase_deg) + B2.*(cosd(vy_phase_deg))));

        # Allocate arrays
        vx_amp_r = np.full_like(vx_phase_deg, np.nan)
        vy_amp_r = np.full_like(vx_phase_deg, np.nan)
        vx_phase_r = np.full_like(vx_phase_deg, np.nan)
        vy_phase_r = np.full_like(vx_phase_deg, np.nan)

        vx_amp_r[valid_mask] = np.hypot(
            A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask]),
            A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask])
        )
        # np.arctan2 returns phase in radians, convert to degrees
        vx_phase_r[valid_mask] = np.arctan2(
            A1[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.sin(vy_phase_deg[valid_mask]),
            A1[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B1[valid_mask]*np.cos(vy_phase_deg[valid_mask])
        )*180.0/np.pi

        vy_amp_r[valid_mask] = np.hypot(
            A2[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B2[valid_mask]*np.cos(vy_phase_deg[valid_mask]),
            A2[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B2[valid_mask]*np.sin(vy_phase_deg[valid_mask])
        )
        # np.arctan2 returns phase in radians, convert to degrees
        vy_phase_r[valid_mask] = np.arctan2(
            A2[valid_mask]*np.sin(vx_phase_deg[valid_mask]) + B2[valid_mask]*np.sin(vy_phase_deg[valid_mask]),
            A2[valid_mask]*np.cos(vx_phase_deg[valid_mask]) + B2[valid_mask]*np.cos(vy_phase_deg[valid_mask])
        )*180.0/np.pi

        vx_phase_r, vx_amp_r = MosaicsReproject.wrap_amp_phase(vx_phase_r, vx_amp_r)
        vy_phase_r, vy_amp_r = MosaicsReproject.wrap_amp_phase(vy_phase_r, vy_amp_r)

        return vx_phase_r, vy_phase_r, vx_amp_r, vy_amp_r

    @staticmethod
    def seasonal_velocity_rotation_x_term(vx0, vy0, vx_phase, vy_phase, vx_amp, vy_amp):
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
        vx_phase_deg = vx_phase/365.24
        vy_phase_deg = vy_phase/365.24

        # Don't use np.nan values in calculations to avoid warnings
        valid_mask = (~np.isnan(vx_phase_deg)) & (~np.isnan(vy_phase_deg))

        # Convert degrees to radians as numpy trig. functions take angles in radians
        # Explanation: if skipping *360.0 in vx_phase_deg, vy_phase_deg above,
        # then to convert to radians: *np.pi/180 --> 360*np.py/180 = 2*np.pi
        vx_phase_deg *= _two_pi
        vy_phase_deg *= _two_pi

        # Matlab prototype code:
        # % Rotation matrix for x component:
        # A1 =  vx_amp.*cosd(theta);
        # B1 = -vy_amp.*sind(theta);

        # New in Python code: compute theta rotation angle
        # theta = arctan(vy0/vx0), since sin(theta)=vy0 and cos(theta)=vx0,
        theta = np.full_like(vx_phase_deg, np.nan)
        theta[valid_mask] = np.arctan2(vy0[valid_mask], vx0[valid_mask])

        if np.any(theta < 0):
            # logging.info(f'Got negative theta, converting to positive values')
            mask = (theta < 0)
            theta[mask] += _two_pi

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # New in Python: assume clockwise rotation by theta as we need to align
        # vector with v0 direction. Therefore  use clockwise transformation matrix
        # for rotation, not counter-clockwise as in Matlab prototype code when rotating
        # both X and Y components.
        A1 = vx_amp*cos_theta
        B1 = vy_amp*sin_theta

        # Matlab WAY
        # B1 = -vy_amp*sin_theta

        # Matlab prototype code:
        # vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
        # vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

        # We want to retain the component only in the direction of v0,
        # which becomes new v_amp and v_phase (see original itslive_composite.py code)
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

        v_phase, v_amp = MosaicsReproject.wrap_amp_phase(v_phase, v_amp)

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

        np_vx[np_vx == DataVars.MISSING_VALUE] = np.nan
        np_vx[~np.isnan(np_vx)] = 1.0

        # Warp y component
        np_vy = self.ds[vy_var].values
        np_vy[np_vy == DataVars.MISSING_VALUE] = np.nan

        np_vy[~np.isnan(np_vy)] = 0.0

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        vx = np.full((num_y, num_x), np.nan, dtype=np.float32)
        vy = np.full((num_y, num_x), np.nan, dtype=np.float32)

        # TODO: make use of parallel processing as cells are independent to speed up
        #       the processing
        for y_index, x_index in tqdm(
            zip(self.valid_cell_indices_y, self.valid_cell_indices_x),
            ascii=True,
            desc=f"Re-projecting X unit {vx_var}, {vy_var}..."
        ):
            # Get values corresponding to the cell in input projection
            v_i, v_j = self.original_ij_index[y_index, x_index]
            dv = [np_vx[v_j, v_i], np_vy[v_j, v_i]]

            # Some points get NODATA for vx but valid vy
            if not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                t_matrix = self.transformation_matrix[y_index, x_index]

                # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                xy_v = np.matmul(t_matrix, dv)

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

        vx = np.full((num_y, num_x), np.nan, dtype=np.float32)
        vy = np.full((num_y, num_x), np.nan, dtype=np.float32)

        # TODO: make use of parallel processing as cells are independent to speed up
        #       the processing
        # for y_index in tqdm(range(num_y), ascii=True, desc=f"Re-projecting Y unit {vx_var}, {vy_var}..."):
        #     for x_index in range(num_x):
        for y_index, x_index in tqdm(
            zip(self.valid_cell_indices_y, self.valid_cell_indices_x),
            ascii=True,
            desc=f"Re-projecting Y unit {vx_var}, {vy_var}..."
        ):
            # Get values corresponding to the cell in input projection
            v_i, v_j = self.original_ij_index[y_index, x_index]
            dv = [np_vx[v_j, v_i], np_vy[v_j, v_i]]

            # Some points get NODATA for vx but valid vy and vice versa
            if not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                t_matrix = self.transformation_matrix[y_index, x_index]
                # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                xy_v = np.matmul(t_matrix, dv)

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
        logging.info(f'Creating trasformation matrix based on {vx_var}, {vy_var}, {v_var}...')

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
            self.transformation_matrix_angle=npzfile['transformation_matrix_angle']
            self.transformation_matrix_scale=npzfile['transformation_matrix_scale']
            self.original_ij_index = npzfile['original_ij_index']
            logging.info(f'Loaded transformation_matrix, angle, scale and original_ij_index from {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')

            self.valid_cell_indices_y = npzfile['valid_cell_indices_y']
            self.valid_cell_indices_x = npzfile['valid_cell_indices_x']
            logging.info(f'Loaded valid grid cell indices from {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')

            # Make sure matrix dimensions correspond to the target grid
            if self.transformation_matrix.shape != (len(self.y0_grid), len(self.x0_grid), TiUnitVector.SIZE, TiUnitVector.SIZE):
                raise RuntimeError(
                    f'Unexpected shape of transformation matrix: {self.transformation_matrix.shape}'
                    f'vs. expected {(len(self.y0_grid), len(self.x0_grid), TiUnitVector.SIZE, TiUnitVector.SIZE)}'
                )

            return

        # Returns a list of (x, y) pairs for the whole grid
        xy0_points = MosaicsReproject.dims_to_grid(self.x0_grid, self.y0_grid)

        # Get corresponding to xy0_points in original projection
        ij0_points = xy_to_ij_transfer.TransformPoints(xy0_points)
        xy0_points = np.array(xy0_points, dtype=np.float32)

        logging.info('Got list of points in original projection...')

        # Calculate x unit vector: add unit length to ij0_points.x
        # TODO: possible optimization - just use already transformed points when
        #       computing bounding box in target projection as it's only by one cell shift in x dimension
        # ij_unit = np.array(ij0_points)
        # ij_unit[:, 0] += self.x_size
        # xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())
        ij_unit = [[each[0] + 1, each[1]] for each in ij0_points]
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit)
        # logging.info(f'Type of xy_points: {type(xy_points)}') # list
        xy_points = np.array(xy_points, dtype=np.float32)

        num_xy0_points = len(xy0_points)

        # Compute X unit vector based on xy0_points, xy_points
        # in output projection
        logging.info('Creating X unit vectors...')
        start_time = timeit.default_timer()

        # unit_vectors = TiUnitVectors(num_xy0_points)
        xunit_v = TiUnitVector(num_xy0_points)
        # unit_vectors.compute_xunit(xy_points, xy0_points, self.x_size)
        # unit_vector.compute(xy_points, xy0_points, self.x_size)
        xunit_v.compute(xy_points, xy0_points)

        # xunit_v = unit_vector.vector.to_numpy()
        logging.info(f'Computed xunit (took {timeit.default_timer() - start_time} seconds)')

        # # Compute unit vector for each cell of the output grid
        # for index in range(num_xy0_points):
        #     xunit_v[index] = np.array(xy_points[index]) - np.array(xy0_points[index])
        #     # xunit_v[index] /= np.linalg.norm(xunit_v[index])
        #     xunit_v[index] /= self.x_size

        logging.info('Creating Y unit vectors...')
        # Calculate Y unit vector: add unit length to ij0_points.y
        # ij_unit = np.array(ij0_points)
        # ij_unit[:, 1] += self.y_size
        # xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())
        ij_unit = [[each[0], each[1] + 1] for each in ij0_points]
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit)
        xy_points = np.array(xy_points, dtype=np.float32)

        # yunit_v = np.zeros((num_xy0_points, 3))
        start_time = timeit.default_timer()
        yunit_v = TiUnitVector(num_xy0_points)
        # unit_vector.compute(xy_points, xy0_points, np.abs(self.y_size))
        yunit_v.compute(xy_points, xy0_points)
        # yunit_v = unit_vector.vector.to_numpy()
        logging.info(f'Computed yunit (took {timeit.default_timer() - start_time} seconds)')

        # Compute transformation matrix per cell

        # Fill out array of "original" indices (indices in input projection)
        # (x,y indices per each cell) with -1 (invalid cell index)
        self.original_ij_index = np.full((num_xy0_points, 2), MosaicsReproject.INVALID_CELL_INDEX, dtype=np.int32)

        # Counter of how many points don't have transformation matrix
        no_value_counter = 0

        # scale_factor_x = self.x_size/MosaicsReproject.TIME_DELTA
        # scale_factor_y = self.y_size/MosaicsReproject.TIME_DELTA
        # remove scale factor when creating T matrix - do not do it in
        # terms of displacement
        # scale_factor_x = 1.0
        # scale_factor_y = 1.0

        # Local normal vector - we don't use it here
        # normal = np.array([0.0, 0.0, 1.0])

        # e = normal[2]*scale_factor_y
        # f = normal[2]*scale_factor_x

        num_i = len(self.ds.x.values)
        num_j = len(self.ds.y.values)

        # For each point on the output grid:
        logging.info('Populating transformation matrix...')

        # Convert list of points to numpy array
        np_ij_points = np.array(ij0_points)

        # Find indices for the original point on its grid
        x_index_all = (np_ij_points[:, 0] - ij_x_bbox.min) / self.x_size
        y_index_all = (np_ij_points[:, 1] - ij_y_bbox.max) / self.y_size

        invalid_mask = (x_index_all < 0) | (y_index_all < 0) | \
            (x_index_all >= num_i) | (y_index_all >= num_j)

        no_value_counter = np.sum(invalid_mask)
        logging.info(f'No value counter = {no_value_counter} (out of {num_xy0_points}) after setting original ij indices')

        # Set original indices for each valid point
        # self.original_ij_index[~invalid_mask, :] = np.vectorize(lambda x, y: [int(x), int(y)], otypes='O')(x_index_all[~invalid_mask], y_index_all[~invalid_mask])

        self.original_ij_index[~invalid_mask, 0] = x_index_all[~invalid_mask].astype(int)
        self.original_ij_index[~invalid_mask, 1] = y_index_all[~invalid_mask].astype(int)

        v_all_values = self.ds[v_var].values

        # Replace nan's with MISSING_VALUE since taichi does not support np.isnan()
        v_all_values_mask = np.isnan(v_all_values)
        v_all_values[v_all_values_mask] = DataVars.MISSING_VALUE

        # xunit_v = xunit_v.data.to_numpy()
        # yunit_v = yunit_v.data.to_numpy()

        # Get indices of cells with valid original_ij_index
        valid_indices, = np.where(~invalid_mask)

        # TODO: check if all cells are excluded from computations
        logging.info('Creating transformation matrix...')
        t1 = timeit.default_timer()

        # vx_all_values = self.ds[vx_var].values
        # vy_all_values = self.ds[vy_var].values

        # debug_y = 7915
        # debug_x = 3701
        # debug_esri_index = 54748893
        # debug_esri_x = 4269
        # debug_esri_y = 7752

        use_taichi = False
        if use_taichi is True:
            transform_matrix = TiTransformMatrix(num_xy0_points)

            # TODO:
            # investigate why taichi code segfaults for HMA: 32642/ITS_LIVE_velocity_120m_HMA_2020_v02.nc -p 102027
            # but not ANT: 32724/ITS_LIVE_velocity_120m_ANT_2020_v02.nc -p 3031
            transform_matrix.compute(xunit_v.data, yunit_v.data, valid_indices, self.original_ij_index, v_all_values)

            self.transformation_matrix = transform_matrix.data.to_numpy()
            self.transformation_matrix_angle = transform_matrix.angle.to_numpy()
            self.transformation_matrix_scale = transform_matrix.scale.to_numpy()

        else:
            self.transformation_matrix = np.full(
                (num_xy0_points, TiUnitVector.SIZE, TiUnitVector.SIZE),
                DataVars.MISSING_VALUE,
                dtype=np.float32
            )
            # Rotation and scale factors based on computed transformation matrix
            self.transformation_matrix_angle = np.full(
                num_xy0_points,
                DataVars.MISSING_VALUE,
                dtype=np.float32
            )
            self.transformation_matrix_scale = np.full(
                (num_xy0_points, 2),
                DataVars.MISSING_VALUE,
                dtype=np.float32
            )

            xunit_v = xunit_v.data.to_numpy()
            yunit_v = yunit_v.data.to_numpy()

            for i in tqdm(valid_indices, ascii=True, desc="Creating transformation matrix..."):
                # Find corresponding point in source P_in projection
                x_index, y_index = self.original_ij_index[i]

                # Check if velocity is valid for the cell, if not then
                # don't compute the matrix
                v_value = v_all_values[y_index, x_index]

                if v_value != DataVars.MISSING_VALUE:
                    xunit = xunit_v[i]
                    yunit = yunit_v[i]

                    # if debug_esri_index == i:
                    #     logging.info(f'DEBUG_HMA ({i}): xunit={xunit}')
                    #     logging.info(f'DEBUG_HMA ({i}): yunit={yunit}')

                    #     logging.info(f'DEBUG_HMA ({i}): Original index: x_index={x_index}, y_index={y_index}, v_value={v_value}, debug_esri_index={debug_esri_index}')
                    #     logging.info(f'DEBUG_HMA ({i}): vx_all_values={vx_all_values[y_index, x_index]}, vy_all_values={vy_all_values[y_index, x_index]}')

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

                    # Yan's code:
                    # denominator = (xunit[0])*(yunit[1])-(yunit[0])*(xunit[1])
                    # which translates to negative of the denominater in the paper:
                    # denominator = c*b - a*d
                    # raster1a[jj] = (yunit[1])/((xunit[0])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                    # raster1b[jj] = -(xunit[1])/((xunit[0])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                    # raster2a[jj] = -(yunit[0])/((xunit[0])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                    # raster2b[jj] = (xunit[0])/((normal[2]*xunit[0]-normal[0]*xunit[2])*(normal[2]*yunit[1]-normal[1]*yunit[2])-(normal[2]*yunit[0]-normal[0]*yunit[2])*(normal[2]*xunit[1]-normal[1]*xunit[2]));
                    # Yan's code has all factors as negatives of paper's A, B, C, D values, so it's all consistent with this code
                    # self.transformation_matrix[i] is a 2x2 matrix
                    denom = (yunit[0]*xunit[1] - yunit[1]*xunit[0])
                    t = [
                        [-yunit[1]/denom, xunit[1]/denom],
                        [yunit[0]/denom, -xunit[0]/denom]
                    ]
                    self.transformation_matrix[i] = t

                    # if debug_esri_index == i:
                    #     logging.info(f'DEBUG_HMA ({i}): Transformation matrix: {t}')

                    # Extract rotation angle and scale factors for X and Y - to be used
                    # to re-project amplitude and phase as can't apply transformation matrix directly,
                    # have to use seasonal_velocity_rotation() to reproject v[xy]_amp and v[xy]_phase
                    # M = [
                    #       [cos(theta) * ScaleX, -sin(theta) * ScaleY],
                    #       [ScaleX * sin(theta), ScaleY * cos(theta)]
                    # ]
                    self.transformation_matrix_angle[i] = np.arctan2(t[1][0], t[0][0])

                    theta_cos = np.cos(self.transformation_matrix_angle[i])

                    # Store scale factor for X and Y components:
                    self.transformation_matrix_scale[i] = [
                        t[0][0] / theta_cos,
                        t[1][1] / theta_cos
                    ]

                else:
                    no_value_counter += 1

        logging.info(f'Created transformation matrix, took {timeit.default_timer() - t1} seconds')

        # Reshape transformation matrix and original cell indices into 2D matrix: (y, x)
        self.transformation_matrix = self.transformation_matrix.reshape(
            (len(self.y0_grid), len(self.x0_grid), TiUnitVector.SIZE, TiUnitVector.SIZE)
        )

        # logging.info(f'DEBUG_HMA (x={debug_esri_x} y={debug_esri_y}) matrix from reshaped array: {self.transformation_matrix[debug_esri_y, debug_esri_x]}')

        self.transformation_matrix_angle = self.transformation_matrix_angle.reshape(
            (len(self.y0_grid), len(self.x0_grid))
        )
        self.transformation_matrix_scale = self.transformation_matrix_scale.reshape(
            (len(self.y0_grid), len(self.x0_grid), TiUnitVector.SIZE)
        )

        self.original_ij_index = self.original_ij_index.reshape(
            (len(self.y0_grid), len(self.x0_grid), TiUnitVector.SIZE)
        )

        # logging.info(f'DEBUG_HMA (x={debug_esri_x} y={debug_esri_y}) original_ij_index from reshaped array: {self.original_ij_index[debug_esri_y, debug_esri_x]}')

        logging.info(f"Number of points with no transformation matrix: {no_value_counter} out of {num_xy0_points} points ({no_value_counter/num_xy0_points*100.0}%)")

        # TODO: Collect indices of cells with valid transformation matrix - to avoid look up
        # later during re-projection
        logging.info('Getting valid cells indices for transformation matrix...')
        t1 = timeit.default_timer()
        self.valid_cell_indices_y, self.valid_cell_indices_x = np.where(self.transformation_matrix[:, :, 0, 0] != DataVars.MISSING_VALUE)
        logging.info(f'Got valid cells indices for transformation matrix, took {timeit.default_timer() - t1} seconds')

        #  transformation matrix and mapping to original ij index for output grid to
        # numpy archive - don't need to calculate these every time need to re-project each
        # of the annual and static mosaics for the same region.
        logging.info(f'Saving transformation_matrix and related info arrays to {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}')
        t1 = timeit.default_timer()

        np.savez(
            MosaicsReproject.TRANSFORMATION_MATRIX_FILE,
            transformation_matrix=self.transformation_matrix,
            transformation_matrix_angle=self.transformation_matrix_angle,
            transformation_matrix_scale=self.transformation_matrix_scale,
            original_ij_index=self.original_ij_index,
            valid_cell_indices_x=self.valid_cell_indices_x,
            valid_cell_indices_y=self.valid_cell_indices_y
        )
        logging.info(f'Saved data to {MosaicsReproject.TRANSFORMATION_MATRIX_FILE}, took {timeit.default_timer() - t1} seconds')

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


def main(input_file: str, output_file: str, output_proj: int, matrix_file: str, verbose_flag: bool, compute_debug_vars: bool = False):
    """
    Main function of the module to be able to invoke the code from
    another Python module.
    """
    start_time = timeit.default_timer()
    MosaicsReproject.VERBOSE = verbose_flag
    MosaicsReproject.COMPUTE_DEBUG_VARS = compute_debug_vars
    MosaicsReproject.TRANSFORMATION_MATRIX_FILE = matrix_file

    logging.info(
        f'reproject_mosaics: verbose={MosaicsReproject.VERBOSE}, '
        f'compute_debug={MosaicsReproject.COMPUTE_DEBUG_VARS}, '
        f'matrix_file={MosaicsReproject.TRANSFORMATION_MATRIX_FILE}'
    )

    reproject = MosaicsReproject(input_file, output_proj)
    reproject(output_file)

    logging.info(f'Done re-projection of {input_file} (took {timeit.default_timer() - start_time} seconds)')


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
        default='transformation_matrix.npz',
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
