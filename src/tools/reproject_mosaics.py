"""
Reprojection tool for ITS_LIVE mosaics to new target projection.

Examples:
$ python reproject.py -i input_filename -p target_projection -o output_filename

    Reproject "input_filename" into 'target_projection' and output new mosaic into
'output_filename' in NetCDF format.

$ python ./reproject.py -i  ITS_LIVE_velocity_120m_HMA_2015_v02.nc -o reproject_ITS_LIVE_velocity_120m_HMA_2015_v02.nc -p 102027
"""
import argparse
from datetime import datetime
import gc
import logging
import math
import numpy as np
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
ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
# ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'
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

            # New error values for V anv VP components (in new projection)
            self.stable_rmse = {}

            self.xy_central_meridian = None

            # Indices for original cells that correspond to the re-projected cells:
            # to find corresponding values
            self.original_ij_index = None

            # Transformation matrix to rotate warped velocity components (vx* and vy*)
            # in output projection
            self.transformation_matrix = None

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
        self.create_transformation_matrix()

        # Check if v0 is present in mosaic, then it's static mosaic
        if CompDataVars.V0 in self.ds:
            self.reproject_static_mosaic(output_file)

        else:
            self.reproject_annual_mosaic(output_file)

    def reproject_static_mosaic(self, output_file: str):
        """
        Reproject static mosaic to new projection.

        output_file: Output file to write reprojected data to.
        """
        raise RuntimeError(f'reproject_static_mosaic(): not implemented')

    def reproject_annual_mosaic(self, output_file):
        """
        Reproject annual mosaic to new projection.

        output_file: Output file to write reprojected data to.
        """
        # Re-project: vx, vy, v, v_error
        v_error_np = self.ds[CompDataVars.V_ERROR]

        masked_np = np.ma.masked_equal(v_error_np, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Original v_error:  min={masked_np.min()} max={masked_np.max()}")

        # outputBounds --- output bounds as (minX, minY, maxX, maxY) in target SRS
        warp_options = gdal.WarpOptions(
            format=MosaicsReproject.WARP_FORMAT,   # Use virtual memory format to avoid writing warped dataset to the file
            outputBounds=(self.x0_bbox.min, self.y0_bbox.min, self.x0_bbox.max, self.y0_bbox.max),
            xRes=self.x_size,
            yRes=self.y_size,
            srcSRS=f'{self.ij_epsg_str}:{self.ij_epsg}',
            dstSRS=f'{self.xy_epsg_str}:{self.xy_epsg}',
            srcNodata=DataVars.MISSING_VALUE,
            dstNodata=DataVars.MISSING_VALUE,
            resampleAlg=gdal.GRA_NearestNeighbour,
            errorThreshold=MosaicsReproject.WARP_ET
        )

        # Compute new vx, vy and v
        vx, vy, v, v_error = self.reproject_velocity(
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.V_ERROR,
            warp_options
        )

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

        # vx = None
        gc.collect()

        reproject_ds[DataVars.VY] = xr.DataArray(
            data=vy,
            coords=ds_coords,
            attrs=self.ds[DataVars.VY].attrs
        )

        # vy = None
        gc.collect()

        reproject_ds[DataVars.V] = xr.DataArray(
            data=v,
            coords=ds_coords,
            attrs=self.ds[DataVars.V].attrs
        )

        # v = None
        gc.collect()

        reproject_ds[CompDataVars.V_ERROR] = xr.DataArray(
            data=v_error,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.V_ERROR].attrs
        )

        v_error = None
        gc.collect()

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

        if self.xy_epsg == 102027:
            reproject_ds[DataVars.MAPPING].attrs['spatial_esri'] = self.xy_epsg

        else:
            reproject_ds[DataVars.MAPPING].attrs['spatial_epsg'] = self.xy_epsg

        # Format GeoTransform attribute:
        # x top left (cell left most boundary), grid size, 0, y top left (cell upper most boundary), 0, -grid size
        half_x_cell = self.x_size/2.0
        half_y_cell = self.y_size/2.0
        reproject_ds[DataVars.MAPPING].attrs['GeoTransform'] = f"{self.x0_grid[0] - half_x_cell} {self.x_size} 0 {self.y0_grid[0] - half_y_cell} 0 {self.y_size}"

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        # Re-project vx_error and vy_error: apply transformation matrix per each cell
        vx_error = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        vy_error = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        v_error = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)

        vx_error_ij = self.ds[CompDataVars.VX_ERROR].values
        vy_error_ij = self.ds[CompDataVars.VY_ERROR].values

        vx_error_ij *= MosaicsReproject.TIME_DELTA/self.x_size
        vy_error_ij *= MosaicsReproject.TIME_DELTA/np.abs(self.y_size)

        for y_index in tqdm(range(num_y), ascii=True, desc=f"Re-projecting {CompDataVars.VX_ERROR}, {CompDataVars.VY_ERROR}..."):
            for x_index in range(num_x):
                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                if not np.isscalar(self.transformation_matrix[y_index, x_index]):
                    # Look up original error values
                    v_i, v_j = self.original_ij_index[y_index, x_index]
                    v_ij = [vx_error_ij[v_j, v_i], vy_error_ij[v_j, v_i]]

                    # If any of the values is NODATA, don't re-project, leave them as NODATA
                    if not np.any(np.isnan(v_ij)) and \
                       not np.isscalar(self.transformation_matrix[y_index, x_index]):
                        # Apply transformation matrix to (vx, vy) values
                        xy_v = np.matmul(self.transformation_matrix[y_index, x_index], v_ij)

                        vx_error[y_index, x_index] = xy_v[0]
                        vy_error[y_index, x_index] = xy_v[1]

                        # ATTN: That should be a sanity check vs. v_error
                        v_error[y_index, x_index] = xy_v[0]*np.abs(vx[y_index, x_index])/v[y_index, x_index] + \
                            xy_v[1]*np.abs(vy[y_index, x_index])/v[y_index, x_index]

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

        # Add debug v_error to dataset just to compare to already computed v_error
        reproject_ds[CompDataVars.V_ERROR+'_verify'] = xr.DataArray(
            data=v_error,
            coords=ds_coords,
            attrs=self.ds[CompDataVars.V_ERROR].attrs
        )

        v_error = None
        gc.collect()

        # Warp "count" variable
        warp_options = gdal.WarpOptions(
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

        # All formats have the following data variables
        reproject_ds[CompDataVars.COUNT] = xr.DataArray(
            data=self.warp_var(CompDataVars.COUNT, warp_options),
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

        # Explicitly set dtype for some variables
        for each in [DataVars.V, DataVars.VX, DataVars.VY,
            CompDataVars.VX_ERROR, CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR, CompDataVars.V_ERROR+'_verify']:
            if each in ds:
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
            'dtype': np.short
        }
        encoding_settings[CompDataVars.COUNT].update(compression)

        if DataVars.FILL_VALUE_ATTR in ds[CompDataVars.COUNT].attrs:
            del ds[CompDataVars.COUNT].attrs[DataVars.FILL_VALUE_ATTR]


        logging.info(f'Enconding for {output_file}: {encoding_settings}')

        # write re-projected data to the file
        ds.to_netcdf(output_file, engine="h5netcdf", encoding = encoding_settings)

    def warp_var(self, var: str, warp_options: gdal.WarpOptions):
        """
        Warp variable into new projection.
        """
        np_ds = gdal.Warp('', f'NETCDF:"{self.input_file}":{var}', options=warp_options).ReadAsArray()
        logging.info(f"Read with GDAL {var}.shape = {np_ds.shape}")

        return np_ds

    def reproject_velocity(self,
        vx_var: str,
        vy_var: str,
        v_var: str,
        v_error_var: str,   # error in original projection
        warp_options: gdal.WarpOptions,
    ):
        """
        Re-project variable's X and Y components, compute its magnitude and
        error if required.

        warp_options: GDAL options to use for warping
        vx_var: name of the variable's X component
        vy_var: name of the variable's Y component
        v_var:  name of the variable
        v_error_var: name of the error variable
        """
        v_error_np = self.ds[v_error_var].values

        masked_np = np.ma.masked_equal(v_error_np, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Original v_error:  min={masked_np.min()} max={masked_np.max()}")

        # Warp x component
        # dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vx_var}')
        _vx = self.ds[vx_var].values
        # _vx = _vx.astype(type(self.x_size))
        _vx[_vx==DataVars.MISSING_VALUE] = np.nan
        logging.info(f"Original {vx_var}:  min={np.nanmin(_vx)} max={np.nanmax(_vx)}")
        # dataset = None

        np_vx = gdal.Warp('', f'NETCDF:"{self.input_file}":{vx_var}', options=warp_options).ReadAsArray()
        masked_np = np.ma.masked_equal(np_vx, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Warped {vx_var}:  min={masked_np.min()} max={masked_np.max()}")

        np_vx = np_vx.astype(np.float32)
        np_vx[np_vx==DataVars.MISSING_VALUE] = np.nan

        # Warp y component
        # dataset = gdal.Open(f'NETCDF:"{self.input_file}":{vy_var}')
        _vy = self.ds[vy_var].values
        # _vy = _vy.astype(type(self.y_size))
        _vy[_vy==DataVars.MISSING_VALUE] = np.nan
        logging.info(f"Original {vy_var}:  min={np.nanmin(_vy)} max={np.nanmax(_vy)}")
        # dataset = None

        # dataset=gdal.Translate('temp_vy.nc', dataset, **kwargs)
        # vy_ds = gdal.Warp('', dataset, options=warp_options)
        np_vy = gdal.Warp('', f'NETCDF:"{self.input_file}":{vy_var}', options=warp_options).ReadAsArray()
        masked_np = np.ma.masked_equal(np_vy, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Warped {vy_var}:  min={masked_np.min()} max={masked_np.max()}")

        np_vy = np_vy.astype(np.float32)
        np_vy[np_vy==DataVars.MISSING_VALUE] = np.nan

        # Convert velocity components to displacement (per transformation matrix requirement)
        # (displacement values are in pixel units)
        np_vx *= MosaicsReproject.TIME_DELTA/self.x_size
        np_vy *= MosaicsReproject.TIME_DELTA/np.abs(self.y_size)

        # Number of X and Y points in the output grid
        num_x = len(self.x0_grid)
        num_y = len(self.y0_grid)

        vx = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        vy = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        v = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)
        v_error = np.full((num_y, num_x), DataVars.MISSING_VALUE, dtype=np.float32)

        # TODO: make use of parallel processing as cells are independent to speed up
        #       the processing
        for y_index in tqdm(range(num_y), ascii=True, desc=f"Re-projecting {vx_var}, {vy_var}..."):
            for x_index in range(num_x):
                dv = np.array([
                    np_vx[y_index, x_index],
                    np_vy[y_index, x_index]
                ])

                # There is no transformation matrix available for the point -->
                # keep it as NODATA
                if not np.isscalar(self.transformation_matrix[y_index, x_index]) and \
                   not np.any(np.isnan(dv)):  # some warped points get NODATA for vx but valid vy
                    # Apply transformation matrix to (vx, vy) values converted to pixel displacement
                    xy_v = np.matmul(self.transformation_matrix[y_index, x_index], dv)

                    vx[y_index, x_index] = xy_v[0]
                    vy[y_index, x_index] = xy_v[1]

                    # Compute v: sqrt(vx^2 + vy^2)
                    v[y_index, x_index] = np.sqrt(xy_v[0]**2 + xy_v[1]**2)

                    # Look up original velocity value to compute the scale factor
                    # for v_error: scale_factor = v_old / v_new
                    v_i, v_j = self.original_ij_index[y_index, x_index]
                    v_ij_value = self.ds[v_var].isel(y=v_j, x=v_i).values.item()

                    scale_factor = 1.0
                    if v_ij_value != 0:
                        scale_factor = v[y_index, x_index]/v_ij_value

                    else:
                        if v[y_index, x_index] != 0:
                            # Set re-projected v to zero - non-zero vx and vy values are
                            # introduced by warping
                            vx[y_index, x_index] = 0
                            vy[y_index, x_index] = 0
                            v[y_index, x_index] = 0

                    if v_ij_value != DataVars.MISSING_VALUE and v_error_np[v_j, v_i] != DataVars.MISSING_VALUE:
                        v_error[y_index, x_index] = v_error_np[v_j, v_i]*scale_factor

                        # Track large differences in v_error values in case they happen. If observed,
                        # most likely need to reduce error threshold for the gdal.warp()
                        if np.abs(v_error[y_index, x_index] - v_error_np[v_j, v_i]) > 100:
                            logging.warning(f"Computed {v_var}_error={v_error[y_index, x_index]}: {v_var}_error_old={v_error_np[v_j, v_i]}")
                            logging.info(f"--->indices: i={v_i} j={v_j} vs. x={x_index} y={y_index}")
                            logging.info(f"--->v:       {v_var}_new={v[y_index, x_index]} {v_var}_old={v_ij_value}")
                            vx_value = self.ds[vx_var].isel(y=v_j, x=v_i).values.item()
                            vy_value = self.ds[vy_var].isel(y=v_j, x=v_i).values.item()
                            logging.info(f"--->old:     {vx_var}={vx_value} {vy_var}={vy_value}")
                            logging.info(f"--->new:     {vx_var}={vx[y_index, x_index]} {vy_var}={vy[y_index, x_index]}")
                            logging.info(f"--->warped_dv={dv*self.x_size}: original_{vx_var}={_vx[v_j, v_i]} original_{vy_var}={_vy[v_j, v_i]}")
                            logging.info(f"--->transf_matrix: {self.transformation_matrix[y_index, x_index]}")

        masked_np = np.ma.masked_equal(vx, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vx_var}:  min={masked_np.min()} max={masked_np.max()}")

        masked_np = np.ma.masked_equal(vy, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Rotated {vy_var}:  min={masked_np.min()} max={masked_np.max()}")

        masked_np = np.ma.masked_equal(v_error, DataVars.MISSING_VALUE, copy=False)
        logging.info(f"Scaled {v_error_var}:  min={masked_np.min()} max={masked_np.max()}")

        return (vx, vy, v, v_error)

    def create_transformation_matrix(self):
        """
        This method creates transformation matrix for each point of the grid.
        """
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

        xy0_points = MosaicsReproject.dims_to_grid(self.x0_grid, self.y0_grid)

        # Get corresponding to xy0_points in original projection
        ij0_points = xy_to_ij_transfer.TransformPoints(xy0_points)

        # Calculate x unit vector: add unit length to ij0_points.x
        ij_unit = np.array(ij0_points)
        ij_unit[:, 0] += self.x_size
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())

        num_xy0_points = len(xy0_points)

        # Compute X unit vector based on xy0_points, xy_points
        # in output projection
        xunit_v = np.zeros((num_xy0_points, 3))

        # Compute unit vector for each cell of the output grid
        for index in range(num_xy0_points):
            xunit_v[index] = np.array(xy_points[index]) - np.array(xy0_points[index])
            xunit_v[index] /= np.linalg.norm(xunit_v[index])

        # Calculate Y unit vector: add unit length to ij0_points.y
        ij_unit = np.array(ij0_points)
        ij_unit[:, 1] += self.y_size
        # ij_unit[:, 1] += np.abs(self.y_size)
        xy_points = ij_to_xy_transfer.TransformPoints(ij_unit.tolist())

        yunit_v = np.zeros((num_xy0_points, 3))

        # Compute Y unit vector based on xy0_points, xy_points
        # in output projection
        for index in range(num_xy0_points):
            yunit_v[index] = np.array(xy_points[index]) - np.array(xy0_points[index])
            yunit_v[index] /= np.linalg.norm(yunit_v[index])

        # Local normal vector
        normal = np.array([0.0, 0.0, 1.0])

        # Compute transformation matrix per cell
        self.transformation_matrix = np.full((num_xy0_points), DataVars.MISSING_VALUE, dtype=object)
        # self.transformation_matrix.fill(DataVars.MISSING_VALUE)

        # For each re-projected cell store indices of corresponding cells in
        # original projection
        self.original_ij_index = np.zeros((num_xy0_points), dtype=object)

        # Counter of how many points don't have transformation matrix
        no_value_counter = 0

        scale_factor_x = self.x_size/MosaicsReproject.TIME_DELTA
        scale_factor_y = self.y_size/MosaicsReproject.TIME_DELTA

        # e = normal[2]*scale_factor_y
        # f = normal[2]*scale_factor_x

        num_i = len(self.ds.x.values)
        num_j = len(self.ds.y.values)

        # debug = False

        # For each point on the output grid:
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
            v_value = self.ds.v.isel(y=y_index, x=x_index).values

            if np.isnan(v_value) or v_value.item() == DataVars.MISSING_VALUE:
                no_value_counter += 1
                continue

            # Double check that vx and vy are valid for the point in original projection
            vx_value = self.ds.vx.isel(y=y_index, x=x_index).values
            vy_value = self.ds.vy.isel(y=y_index, x=x_index).values

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
        # Use z=0 as osr.CoordinateTransformation.TransformPoints() returns 3d point coordinates
        grid = np.zeros((len(x)*len(y), 3))

        num_row = 0
        for each_y in y:
            for each_x in x:
                grid[num_row][0] = each_x
                grid[num_row][1] = each_y
                num_row += 1

        return grid


if __name__ == '__main__':
    """
    Re-project ITS_LIVE mosaic (static or annual) to the target projection.
    """
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

    command_args = parser.parse_args()

    reproject = MosaicsReproject(command_args.input_file, command_args.output_proj)
    reproject(command_args.output_file)

    logging.info('Done.')
