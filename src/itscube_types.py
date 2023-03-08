"""
Classes that define data variables and attributes for the ITS_LIVE data sets:
datacubes, composites, and mosaics.
"""
import numpy as np


class ShapeFile:
    """
    Variables names specific to the ITS_LIVE shapefiles.
    """
    EPSG = 'epsg'
    LANDICE_2KM = 'landice_2km'
    LANDICE = 'landice'
    FLOATINGICE = 'floatingice'

    Name = {
        LANDICE: 'land ice mask',
        FLOATINGICE: 'floating ice mask',
    }
    Description = {
        LANDICE: 'land ice mask, 1 = land-ice, 0 = non-land-ice',
        FLOATINGICE: 'floating ice mask, 1 = floating-ice, 0 = non-floating-ice',
    }


class Output:
    """
    Attributes specific to the output store format (Zarr or NetCDF)
    """
    DTYPE_ATTR = 'dtype'
    COMPRESSOR_ATTR = 'compressor'
    # For the floating point types in Zarr format, any datatype in NetCDF format
    FILL_VALUE_ATTR = '_FillValue'
    # For integer types in Zarr format
    MISSING_VALUE_ATTR = 'missing_value'
    CHUNKS_ATTR = 'chunks'
    CHUNKSIZES_ATTR = 'chunksizes'

    # These encoding attributes are for M11 and M12 variables in radar granules
    SCALE_FACTOR = 'scale_factor'
    ADD_OFFSET = 'add_offset'


class CubeOutput:
    """
    Class to represent attributes and their values for xr.Dataset that represents
    a datacube.
    """
    # Attributes
    GDAL_AREA_OR_POINT = 'GDAL_AREA_OR_POINT'
    PROJ_POLYGON = 'proj_polygon'
    GEO_POLYGON = 'geo_polygon'
    DATACUBE_SOFTWARE_VERSION = 'datacube_software_version'
    DATE_CREATED = 'date_created'
    DATE_UPDATED = 'date_updated'
    INSTITUTION = 'institution'
    PROJECTION = 'projection'
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'
    S3 = 's3'
    URL = 'url'
    TITLE = 'title'
    AUTHOR = 'author'
    CONVENTIONS = 'Conventions'

    class Values:
        """
        Attribute values.
        """
        AREA = 'Area'
        INSTITUTION = 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology'
        TITLE = 'ITS_LIVE datacube of image pair velocities'
        AUTHOR = 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)'
        CONVENTIONS = 'CF-1.8'


class CompOutput:
    """
    Class to represent attributes for the output format of the composites data.
    """
    COMPOSITES_SOFTWARE_VERSION = 'composites_software_version'
    DATACUBE_AUTORIFT_PARAMETER_FILE = 'datacube_autoRIFT_parameter_file'
    SENSORS_LABELS = 'sensors_labels'

    DATACUBE_CREATED = 'datacube_created'
    DATACUBE_UPDATED = 'datacube_updated'
    DATACUBE_S3 = 'datacube_s3'
    DATACUBE_URL = 'datacube_url'

    class Values:
        TITLE = 'ITS_LIVE annual composites of image pair velocities'


class Coords:
    """
    Coordinates for the data cube.
    """
    # For original datacube
    MID_DATE = 'mid_date'
    X = 'x'
    Y = 'y'

    STD_NAME = {
        MID_DATE: "image_pair_center_date_with_time_separation",
        X: "projection_x_coordinate",
        Y: "projection_y_coordinate"
    }

    DESCRIPTION = {
        MID_DATE: "midpoint of image 1 and image 2 acquisition date and time "
                  "with granule's centroid longitude and latitude as microseconds",
        X: "x coordinate of projection",
        Y: "y coordinate of projection"
    }


class FileExtension:
    """
    File extensions used by datacube related files.
    """
    ZARR = '.zarr'
    JSON = '.json'


class DataVars:
    """
    Data variables for the data cube.
    """
    # Granule attributes
    AUTORIFT_SOFTWARE_VERSION = 'autoRIFT_software_version'
    AUTORIFT_PARAMETER_FILE = 'autoRIFT_parameter_file'

    # Datacube variable and its attributes to store skipped granule information
    SKIPPED_GRANULES = 'skipped_granules'
    SKIP_EMPTY_DATA = 'skipped_empty_data'
    SKIP_DUPLICATE = 'skipped_duplicate_middle_date'
    SKIP_PROJECTION = 'skipped_wrong_projection'

    # Attributes that appear for multiple data variables
    DESCRIPTION_ATTR = 'description'  # v, vx, vy
    GRID_MAPPING = 'grid_mapping'  # v, vx, vy - store only one per cube
    GRID_MAPPING_NAME = 'grid_mapping_name'  # New format: attribute to store grid mapping

    # Store only one per cube (attributes in vx, vy)
    # Per Yang: generally yes, though for vxp and vyp it was calculated again
    # but the number should not change quite a bit. so it should be okay to
    # use a single value for all variables
    STABLE_COUNT_SLOW = 'stable_count_slow'
    STABLE_COUNT_MASK = 'stable_count_stationary'

    # Attributes for vx, vy, vr, va
    # FLAG_STABLE_SHIFT = 'flag_stable_shift' # Old granule format: In Radar and updated Optical formats
    FLAG_STABLE_SHIFT = 'stable_shift_flag'  # In Radar and updated Optical formats
    FLAG_STABLE_SHIFT_DESCRIPTION = 'stable_shift_flag_description'
    STABLE_SHIFT = 'stable_shift'
    STABLE_SHIFT_SLOW = 'stable_shift_slow'
    STABLE_SHIFT_MASK = 'stable_shift_stationary'

    # These data variables names are created at runtime: based on "stable_shift"
    # attribute of vx and vy variables
    VX_STABLE_SHIFT = 'vx_stable_shift'
    VY_STABLE_SHIFT = 'vy_stable_shift'

    STD_NAME = 'standard_name'
    NOTE = 'note'

    UNITS = 'units'
    M_Y_UNITS = 'meter/year'
    M_Y2_UNITS = 'meter/year^2'
    M_UNITS = 'm'
    COUNT_UNITS = 'count'
    BINARY_UNITS = 'binary'
    PERCENT_UNITS = 'percent'
    DAY_OF_YEAR_UNITS = 'day of year'
    PIXEL_PER_M_YEAR = 'pixel/(meter/year)'
    M_PER_YEAR_PIXEL = 'meter/(year*pixel)'

    # Original data variables and their attributes per ITS_LIVE granules.
    V = 'v'
    VX = 'vx'
    VY = 'vy'
    V_ERROR = 'v_error'

    # Radar data variables to preserve in datacube
    VA = 'va'
    VR = 'vr'
    M11 = 'M11'
    M12 = 'M12'
    # Attributes for M1* data
    DR_TO_VR_FACTOR = 'dr_to_vr_factor'
    DR_TO_VR_FACTOR_DESCRIPTION = 'dr_to_vr_factor_description'

    # Postfix to format velocity specific attributes, such as
    # vx_error, vx_error_mask, vx_error_modeled, vx_error_slow,
    # vx_error_description, vx_error_mask_description, vx_error_modeled_description,
    # vx_error_slow_description
    ERROR_DESCRIPTION = 'description'
    ERROR = 'error'
    ERROR_MASK = 'error_stationary'
    ERROR_MODELED = 'error_modeled'
    ERROR_SLOW = 'error_slow'

    CHIP_SIZE_HEIGHT = 'chip_size_height'
    CHIP_SIZE_WIDTH = 'chip_size_width'
    # Attributes
    CHIP_SIZE_COORDS = 'chip_size_coordinates'

    INTERP_MASK = 'interp_mask'

    # Specific to the datacube
    URL = 'granule_url'

    # Data variable specific to the epsg code:
    # * Polar_Stereographic when epsg code of 3031 or 3413
    # * UTM_Projection when epsg code of 326** or 327**
    POLAR_STEREOGRAPHIC = 'Polar_Stereographic'
    UTM_PROJECTION = 'UTM_Projection'
    MAPPING = 'mapping'  # New format

    # Missing (FillValue) values for data variables
    MISSING_BYTE = 0.0
    MISSING_VALUE = -32767
    MISSING_POS_VALUE = 32767
    MISSING_UINT8_VALUE = 255

    # Standard name for variables to use
    NAME = {
        INTERP_MASK: 'interpolated_value_mask',
        VA: 'azimuth_velocity',
        VR: 'range_velocity',
        V_ERROR: 'velocity_error',
        M11: 'conversion_matrix_element_11',
        M12: 'conversion_matrix_element_12',
    }

    # Map of variables with integer data type
    INT_TYPE = {
        INTERP_MASK: np.ubyte,
        CHIP_SIZE_HEIGHT: np.uint16,
        CHIP_SIZE_WIDTH: np.uint16,
        FLAG_STABLE_SHIFT: np.uint8,
        STABLE_COUNT_SLOW: np.uint16,
        STABLE_COUNT_MASK: np.uint16,
        V: np.int16,
        VX: np.int16,
        VY: np.int16,
        V_ERROR: np.int16,
        VA: np.int16,
        VR: np.int16,
        M11: np.int16,
        M12: np.int16
    }

    # Missing value for data variables of integer data type
    INT_MISSING_VALUE = {
        INTERP_MASK: MISSING_BYTE,
        CHIP_SIZE_HEIGHT: MISSING_BYTE,
        CHIP_SIZE_WIDTH: MISSING_BYTE,
        V: MISSING_VALUE,
        VX: MISSING_VALUE,
        VY: MISSING_VALUE,
        V_ERROR: MISSING_VALUE,
        VA: MISSING_VALUE,
        VR: MISSING_VALUE,
        M11: MISSING_VALUE,
        M12: MISSING_VALUE
    }

    # Description strings for all data variables and some
    # of their attributes.
    DESCRIPTION = {
        V: "velocity magnitude",
        VX: "velocity component in x direction",
        VY: "velocity component in y direction",

        STABLE_COUNT_SLOW: "number of valid pixels over slowest 25% of ice",
        STABLE_COUNT_MASK: "number of valid pixels over stationary or slow-flowing surfaces",

        STABLE_SHIFT_SLOW:
            "{} shift calibrated using valid pixels over slowest 25% of retrieved velocities",
        STABLE_SHIFT_MASK:
            "{} shift calibrated using valid pixels over stable surfaces, "
            " stationary or slow-flowing surfaces with velocity < 15 m/yr identified from an external mask",

        # These descriptions are based on Radar granule format. Have to set them
        # manually since there are no Radar format granules are available for
        # processing just yet (otherwise these attributes would be automatically
        # picked up from the granules).
        VA: "velocity in radar azimuth direction",
        VR: "velocity in radar range direction",
        M11: "conversion matrix element (1st row, 1st column) that can be multiplied with vx to give range pixel displacement dr (see Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)",
        M12: "conversion matrix element (1st row, 2nd column) that can be multiplied with vy to give range pixel displacement dr (see Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)",
        DR_TO_VR_FACTOR: "multiplicative factor that converts slant range pixel displacement dr to slant range velocity vr",
        V_ERROR: "velocity magnitude error",
        INTERP_MASK: "light interpolation mask",
        CHIP_SIZE_COORDS:
            "Optical data: chip_size_coordinates = "
            "'image projection geometry: width = x, height = y'. Radar data: "
            "chip_size_coordinates = 'radar geometry: width = range, height = azimuth'",
        CHIP_SIZE_HEIGHT: "height of search template (chip)",
        CHIP_SIZE_WIDTH:  "width of search template (chip)",
        FLAG_STABLE_SHIFT:
            "flag for applying velocity bias correction: "
            "0 = no correction; "
            "1 = correction from overlapping stable surface mask (stationary or "
            "slow-flowing surfaces with velocity < 15 m/yr)(top priority); "
            "2 = correction from slowest 25% of overlapping velocities (second priority)",
        URL: "original granule URL",
        AUTORIFT_SOFTWARE_VERSION: "version of autoRIFT software",
        SKIPPED_GRANULES: "skipped granules during datacube construction"
    }

    class ImgPairInfo:
        """
        Class to represent attributes of the "img_pair_info" data variable,
        which become new data variables in the datacube to represent these
        attributes for all layers in the datacube.
        """
        NAME = 'img_pair_info'

        DATE_UNITS = 'days since 1970-01-01'

        # Attributes
        MISSION_IMG1 = 'mission_img1'
        SENSOR_IMG1 = 'sensor_img1'
        SATELLITE_IMG1 = 'satellite_img1'
        MISSION_IMG2 = 'mission_img2'
        SENSOR_IMG2 = 'sensor_img2'
        SATELLITE_IMG2 = 'satellite_img2'
        DATE_DT = 'date_dt'
        # Rename mid_date to date_center as they are the same, don't collect this
        DATE_CENTER = 'date_center'
        ROI_VALID_PERCENTAGE = 'roi_valid_percentage'
        LONGITUDE = 'longitude'
        LATITUDE = 'latitude'

        # New format defines these attributes, make them datacube attributes
        TIME_STANDARD_IMG1 = 'time_standard_img1'
        TIME_STANDARD_IMG2 = 'time_standard_img2'

        ACQUISITION_DATE_IMG1 = 'acquisition_date_img1'
        ACQUISITION_DATE_IMG2 = 'acquisition_date_img2'

        # ATTN: Sentinel-2 granules are using satellite_img1 and satellite_img2 instead
        # of sensor_img1 and sensor_img2
        ALL = [
            ACQUISITION_DATE_IMG1,
            ACQUISITION_DATE_IMG2,
            MISSION_IMG1,
            MISSION_IMG2,
            SATELLITE_IMG1,
            SATELLITE_IMG2,
            SENSOR_IMG1,
            SENSOR_IMG2,
            DATE_CENTER,
            DATE_DT,
            ROI_VALID_PERCENTAGE
        ]

        ALL_DTYPE = {
            DATE_DT: np.float32,
            ROI_VALID_PERCENTAGE: np.float32,
            MISSION_IMG1: str,
            MISSION_IMG2: str,
            SATELLITE_IMG1: str,
            SATELLITE_IMG2: str,
            SENSOR_IMG1: str,
            SENSOR_IMG2: str
        }

        # Description strings for data variables.
        DESCRIPTION = {
            MISSION_IMG1: "id of the mission that acquired image 1",
            SENSOR_IMG1: "id of the sensor that acquired image 1",
            SATELLITE_IMG1: "id of the satellite that acquired image 1",
            ACQUISITION_DATE_IMG1: "acquisition date and time of image 1",
            MISSION_IMG2: "id of the mission that acquired image 2",
            SENSOR_IMG2: "id of the sensor that acquired image 2",
            SATELLITE_IMG2: "id of the satellite that acquired image 2",
            ACQUISITION_DATE_IMG2: "acquisition date and time of image 2",
            DATE_DT: "time separation between acquisition of image 1 and image 2",
            DATE_CENTER: "midpoint of image 1 and image 2 acquisition date",
            ROI_VALID_PERCENTAGE: "percentage of pixels with a valid velocity "
                                  "estimate determined for the intersection of the full image "
                                  "pair footprint and the region of interest (roi) that defines "
                                  "where autoRIFT tried to estimate a velocity",
        }

        # Flag if data variable values are to be converted to the date objects
        CONVERT_TO_DATE = {
            MISSION_IMG1: False,
            SENSOR_IMG1: False,
            SATELLITE_IMG1: False,
            ACQUISITION_DATE_IMG1: True,
            MISSION_IMG2: False,
            SENSOR_IMG2:  False,
            SATELLITE_IMG2: False,
            ACQUISITION_DATE_IMG2: True,
            DATE_DT: False,
            DATE_CENTER: True,
            ROI_VALID_PERCENTAGE: False,
        }

        STD_NAME = {
            MISSION_IMG1: "image1_mission",
            SENSOR_IMG1: "image1_sensor",
            SATELLITE_IMG1: "image1_satellite",
            ACQUISITION_DATE_IMG1: "image1_acquition_date",
            MISSION_IMG2: "image2_mission",
            SENSOR_IMG2: "image2_sensor",
            SATELLITE_IMG2: "image2_satellite",
            ACQUISITION_DATE_IMG2: "image2_acquition_date",
            DATE_DT: "image_pair_time_separation",
            DATE_CENTER: "image_pair_center_date",
            ROI_VALID_PERCENTAGE: "region_of_interest_valid_pixel_percentage",
        }

        UNITS = {
            DATE_DT: 'days',
            # ACQUISITION_DATE_IMG1: DATE_UNITS,
            # ACQUISITION_DATE_IMG2: DATE_UNITS,
            # DATE_CENTER: DATE_UNITS
        }


class CompDataVars:
    """
    Data variables and their descriptions to write annual composites to Zarr or
    NetCDF output store.
    """
    TIME = 'time'
    SENSORS = 'sensor'

    VX_ERROR = 'vx_error'
    VY_ERROR = 'vy_error'
    V_ERROR = 'v_error'
    VX_AMP_ERROR = 'vx_amp_error'
    VY_AMP_ERROR = 'vy_amp_error'
    V_AMP_ERROR = 'v_amp_error'
    VX_AMP = 'vx_amp'
    VY_AMP = 'vy_amp'
    V_AMP = 'v_amp'
    VX_PHASE = 'vx_phase'
    VY_PHASE = 'vy_phase'
    V_PHASE = 'v_phase'
    COUNT = 'count'
    MAX_DT = 'dt_max'
    OUTLIER_FRAC = 'outlier_percent'
    SENSOR_INCLUDE = 'sensor_flag'
    VX0 = 'vx0'
    VY0 = 'vy0'
    V0 = 'v0'
    COUNT0 = 'count0'
    VX0_ERROR = 'vx0_error'
    VY0_ERROR = 'vy0_error'
    V0_ERROR = 'v0_error'
    SLOPE_VX = 'dvx_dt'
    SLOPE_VY = 'dvy_dt'
    SLOPE_V = 'dv_dt'

    STD_NAME = {
        DataVars.VX: 'land_ice_surface_x_velocity',
        DataVars.VY: 'land_ice_surface_y_velocity',
        DataVars.V: 'velocity',
        VX_ERROR: 'x_velocity_error',
        VY_ERROR: 'y_velocity_error',
        V_ERROR: 'velocity_error',
        VX_AMP_ERROR: 'vx_amplitude_error',
        VY_AMP_ERROR: 'vy_amplitude_error',
        V_AMP_ERROR: 'v_amplitude_error',
        VX_AMP: 'vx_amplitude',
        VY_AMP: 'vy_amplitude',
        V_AMP: 'v_amplitude',
        VX_PHASE: 'vx_phase',
        VY_PHASE: 'vy_phase',
        V_PHASE: 'v_phase',
        SENSORS: 'sensors',
        TIME: 'time',
        COUNT: 'count',
        MAX_DT: 'dt_maximum',
        SENSOR_INCLUDE: 'sensor_flag',
        OUTLIER_FRAC: 'outlier_percent',
        VX0: 'climatological_x_velocity',
        VY0: 'climatological_y_velocity',
        V0: 'climatological_velocity',
        COUNT0: 'count0',
        VX0_ERROR: 'vx0_velocity_error',
        VY0_ERROR: 'vy0_velocity_error',
        V0_ERROR: 'v0_velocity_error',
        SLOPE_VX: 'dvx_dt',
        SLOPE_VY: 'dvy_dt',
        SLOPE_V: 'dv_dt'
    }

    DESCRIPTION = {
        DataVars.VX: 'mean annual velocity of sinusoidal fit to vx',
        DataVars.VY: 'mean annual velocity of sinusoidal fit to vy',
        DataVars.V: 'mean annual velocity determined by taking the hypotenuse of vx and vy',
        TIME: 'time',
        VX_ERROR: 'error weighted error for vx',
        VY_ERROR: 'error weighted error for vy',
        V_ERROR: 'error weighted error for v',
        VX_AMP_ERROR: 'error for vx_amp',
        VY_AMP_ERROR: 'error for vy_amp',
        V_AMP_ERROR: 'error for v_amp',
        VX_AMP: f'climatological [%i-%i] mean seasonal amplitude of sinusoidal fit to vx',
        VY_AMP: f'climatological [%i-%i] mean seasonal amplitude in sinusoidal fit in vy',
        V_AMP: f'climatological [%i-%i] mean seasonal amplitude in the direction of mean flow as defined by vx0 and vy0',
        VX_PHASE: f'climatological [%i-%i] day of seasonal maximum velocity of sinusoidal fit to vx',
        VY_PHASE: f'climatological [%i-%i] day of seasonal maximum velocity of sinusoidal fit to vy',
        V_PHASE: f'day of maximum climatological [%i-%i] seasonal velocity determined from sinusoidal fit to vx and vy',
        COUNT: 'number of image pairs used in error weighted least squares fit',
        MAX_DT: 'maximum allowable time separation between image pair acquisitions included in error weighted least squares fit',
        SENSOR_INCLUDE: 'flag = 1 if sensor group (see sensor variable) is included, flag = 0 if sensor group is excluded',
        OUTLIER_FRAC: f'percentage of data identified as outliers and excluded from the climatological [%i-%i] error weighted least squares fit',
        SENSORS: 'combinations of unique sensors and missions that are grouped together for date_dt filtering',
        VX0: f'climatological [%i-%i] vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values. The climatology uses a time-intercept of January 1, %i.',
        VY0: f'climatological [%i-%i] vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values. The climatology uses a time-intercept of January 1, %i.',
        V0: f'climatological [%i-%i] v determined by taking the hypotenuse of vx0 and vy0. The climatology uses a time-intercept of January 1, %i.',
        COUNT0: f'number of image pairs used for climatological [%i-%i] means',
        VX0_ERROR: 'error for vx0',
        VY0_ERROR: 'error for vy0',
        V0_ERROR: 'error for v0',
        SLOPE_VX: f'trend [%i-%i] in vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values',
        SLOPE_VY: f'trend [%i-%i] in vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values',
        SLOPE_V: f'trend [%i-%i] in v determined by projecting dvx_dt and dvy_dt onto the unit flow vector defined by vx0 and vy0'
    }


class BinaryFlag:
    """
    Class to store output format attributes and their values for the binary masking.
    """
    # Standard attributes for the output format
    VALUES_ATTR = 'flag_values'
    MEANINGS_ATTR = 'flag_meanings'

    # Binary mask values
    VALUES = np.array([0, 1], dtype=np.uint8)

    # Binary mask meanings
    MEANINGS = {
        DataVars.INTERP_MASK: 'measured interpolated',
        ShapeFile.LANDICE: 'non-ice ice',
        ShapeFile.FLOATINGICE: 'non-ice ice',
        CompDataVars.SENSOR_INCLUDE: 'excluded included'
    }


class BatchVars:
    """
    Variables that are common to all AWS Batch processing for the ITS_LIVE project.
    """
    # List of EPSG codes to generate data products for. If this list is empty,
    # then generate all data products.
    EPSG_TO_GENERATE = []

    # List of EPSG codes to exclude from data product generation.
    # If this list is empty, then generate don't apply EPSG exclusion filter.
    EPSG_TO_EXCLUDE = []

    # List of datacubes filenames to generate/consider if only specific
    # datacubes should be generated/considered.
    # If an empty list then generate/consider all qualifying datacubes.
    CUBES_TO_GENERATE = []

    # List of datacube filenames to exclude from processing. This is handy
    # when some of the cubes were already processed.
    CUBES_TO_EXCLUDE = []

    # Generate data products which centers fall within provided polygon
    POLYGON_SHAPE = None

    # A way to pick specific 10x10 grid cell for the datacube
    PATH_TOKEN = None

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    MID_POINT_RESOLUTION = 50.0

    # Default AWS S3 bucket for the data
    AWS_PREFIX = 's3://its-live-data'

    # HTTP URL for the datacube/composite/mosaics full path in S3 bucket
    HTTP_PREFIX = 'http://its-live-data.s3.amazonaws.com'


class CubeJson:
    """
    Variables names within GeoJson cube definition file.
    """
    FEATURES = 'features'
    PROPERTIES = 'properties'
    DATA_EPSG = 'data_epsg'
    EPSG = 'epsg'
    GEOMETRY_EPSG = 'geometry_epsg'
    COORDINATES = 'coordinates'
    ROI_PERCENT_COVERAGE = 'roi_percent_coverage'
    EPSG_SEPARATOR = ':'
    EPSG_PREFIX = 'EPSG'
    URL = 'zarr_url'
    EXIST_FLAG = 'datacube_exist'
    REGION = 'region'
    RGI_CODE = 'RGI_CODE'
    DIRECTORY = 'directory'

    # String token to use in filenames to specify EPSG code for the data
    EPSG_TOKEN = 'EPSG'


class FilenamePrefix:
    """
    Filename prefixes used by ITS_LIVE data products.
    """
    Datacube = 'ITS_LIVE_vel'
    Composites = 'ITS_LIVE_velocity'
    Mosaics = 'ITS_LIVE_velocity'


# Define attributes for coordinates of composites and annual mosaics
# ATTN: this is done to set coordinates attributes of the xr.Dataset before
# saving it to the file - adding some data variables to the xr.Dataaset wipes
# out coordinates attributes (xarray bug?)
TIME_ATTRS = {
    DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.TIME],
    DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.TIME]
}
SENSORS_ATTRS = {
    DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSORS],
    DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSORS]
}
X_ATTRS = {
    DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
    DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X],
    DataVars.UNITS: DataVars.M_UNITS
}
Y_ATTRS = {
    DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
    DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y],
    DataVars.UNITS: DataVars.M_UNITS
}


def datacube_filename_zarr(epsg_format: str, grid_size: int, mid_x: int, mid_y: int):
    """
    Format filename for the datacube:
    ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-2650000.zarr
    """
    return f"{FilenamePrefix.Datacube}_{epsg_format}_G{grid_size:04d}_X{mid_x}_Y{mid_y}.zarr"


def composite_filename_zarr(epsg_format: int, grid_size: str, mid_x: int, mid_y: int):
    """
    Format filename for the datacube's composite:
    ITS_LIVE_velocity_EPSG3413_120m_X-3250000_Y250000.zarr

    Inputs:
    =======
    epsg_format: String representation of the EPSG code in "EPSGXXXXX" format
    grid_size: Grid size
    mid_x: X coordinate of datacube centroid
    mid_y: Y coordinate of datacube centroid
    """
    return f"{FilenamePrefix.Composites}_{CubeJson.EPSG_TOKEN}{epsg_format}_{int(grid_size):03d}m_X{mid_x}_Y{mid_y}.zarr"


def annual_mosaics_filename_nc(grid_size: str, region: str, year_date, version: str):
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

    return f"{FilenamePrefix.Mosaics}_{grid_size}m_{region}_{year_value}_{version}.nc"


def summary_mosaics_filename_nc(grid_size: str, region: str, version: str):
    """
    Format filename for the summary mosaics of the region:
    ITS_LIVE_velocity_120m_ALA_2013_v02.nc
    """
    return f"{FilenamePrefix.Mosaics}_{grid_size}m_{region}_0000_{version}.nc"


def to_int_type(data, data_type=np.uint16, fill_value=DataVars.MISSING_POS_VALUE):
    """
    Convert data to requested integer datatype. "fill_value" must correspond
    to the "data_type" to replace NaNs with corresponding to the datatype missing_value:
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
    # Replace NaN's with zero's as it will store garbage for NaN's
    _mask = np.isnan(data)
    data[_mask] = fill_value

    # Round to nearest int value
    int_data = np.rint(data).astype(data_type)

    return int_data
