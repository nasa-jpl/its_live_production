"""
Classes that define data variables and attributes for the ITS_LIVE data sets:
datacubes, composites, and mosaics.
"""
from dataclasses import dataclass
import numpy as np

import utils


# @dataclass(frozen=True)
class Output:
    """
    Attributes specific to the output store format (Zarr or NetCDF)
    """
    DTYPE_ATTR = 'dtype'
    COMPRESSOR_ATTR = 'compressor'
    # For the floating point types in Zarr format, and any datatype in NetCDF
    # format.
    # Integer types in Zarr format use 'missing_value' attribute instead of
    # '_FillValue' to specify the missing value.
    FILL_VALUE_ATTR = '_FillValue'
    CHUNKS_ATTR = 'chunks'
    CHUNKSIZES_ATTR = 'chunksizes'

    # These encoding attributes are for M11 and M12 variables in radar granules
    SCALE_FACTOR = 'scale_factor'
    ADD_OFFSET = 'add_offset'

    # Global attributes
    LATITUDE = 'latitude'
    LONGITUDE = 'longitude'

    COUNT = 'count'
    URL = 'url'
    REFERENCES = 'references'
    PROJECTION = 'projection'
    PUBLISHER_NAME = 'publisher_name'


# Former CubeOutput
@dataclass(frozen=True)
class CubeOutputInfo:
    """
    Class to represent attributes and their values for xr.Dataset that
    represents a datacube.
    """
    # Attributes
    gdal_area_or_point: str = 'GDAL_AREA_OR_POINT'
    proj_polygon: str = 'proj_polygon'
    geo_polygon: str = 'geo_polygon'
    datacube_software_version: str = 'datacube_software_version'
    date_created: str = 'date_created'
    date_updated: str = 'date_updated'
    geospatial_bounds: str = 'geospatial_bounds'

    # Attribute values.
    values = {
        gdal_area_or_point: 'Area',
        utils.OutputFormat.institution: 'NASA Jet Propulsion Laboratory (JPL), ' \
                        'California Institute of Technology',
        utils.OutputFormat.title: 'ITS_LIVE datacube of image pair velocities',
        utils.OutputFormat.author: 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)',
        utils.OutputFormat.conventions: 'CF-1.8'
    }

CubeFormat = CubeOutputInfo()


@dataclass(frozen=True)
class MappingAttributes:
    """Class to represent attributes of the "mapping" data variable that
    exists in ITS_LIVE granules and is propagated to the datacube as a data
    variable with these attributes.
    """
    spatial_epsg: str = 'spatial_epsg'
    geo_transform: str = 'GeoTransform'
    grid_mapping: str = 'grid_mapping'
    grid_mapping_name: str = 'grid_mapping_name'


@dataclass(frozen=True)
class MappingInstance:
    """Class to represent the "mapping" data variable.
    """
    name: str = 'mapping'
    attrs: MappingAttributes = MappingAttributes()

# Former MappingInstance
Mapping = MappingInstance()


@dataclass(frozen=True)
class SkippedGranulesInfo:
    """
    Attributes for skipped granules information.
    """
    name: str = 'skipped_granules'
    empty: str = 'skipped_empty_data'
    duplicate: str = 'skipped_duplicate_middle_date'
    projection: str = 'skipped_wrong_projection'

SkippedGranules = SkippedGranulesInfo()


@dataclass(frozen=True)
class VarsAttributes:
    """
    Class to represent attributes of the data variables in the datacube.
    """
    std_name: str = 'standard_name'
    description: str = 'description'
    autorift_param_file: str = 'autoRIFT_parameter_file'
    chip_size_coords: str = 'chip_size_coordinates'
    flag_stable_shift_description: str = 'stable_shift_flag_description'

    # M11/M12 attributes
    dr_to_vr_factor_description: str = 'dr_to_vr_factor_description'

    note: str = 'note'


@dataclass(frozen=True)
class VarsPostfix:
    """
    Class to represent postfixes of the data variables in the datacube.
    These will be used to create data variables names specific to velocity
    components. For example, 'vx_stable_shift', 'vy_stable_shift',
    'va_stable_shift', and 'vr_stable_shift'.
    """
    #  These will be used to create data variables specific to velocity
    # components, for example, 'vx_stable_shift', 'vy_stable_shift',
    # 'va_stable_shift', and 'vr_stable_shift'
    stable_shift: str = 'stable_shift'
    stable_shift_slow: str = 'stable_shift_slow'
    stable_shift_mask: str = 'stable_shift_stationary'

    # Postfix to format velocity specific attributes, such as
    # vx_error, vx_error_stationary, vx_error_modeled, vx_error_slow.
    error: str = 'error'
    error_mask: str = 'error_stationary'
    error_modeled: str = 'error_modeled'
    error_slow: str = 'error_slow'

    # Attributes for M1* data
    dr_to_vr_factor: str = 'dr_to_vr_factor'


# Former DataVars
@dataclass(frozen=True)
class VarsInfo:
    """
    Class to represent data variable names and common attributes in the
    datacube.
    """
    attrs: VarsAttributes = VarsAttributes()
    postfix: VarsPostfix = VarsPostfix()

    # Variables names
    v: str = 'v'
    vx: str = 'vx'
    vy: str = 'vy'
    v_error: str = 'v_error'

    # Radar data variables to preserve in datacube
    va: str = 'va'
    vr: str = 'vr'
    m11: str = 'M11'
    m12: str = 'M12'

    chip_size_height: str = 'chip_size_height'
    chip_size_width: str = 'chip_size_width'
    interp_mask: str = 'interp_mask'
    flag_stable_shift: str = 'stable_shift_flag'

    # Store only one per cube (attributes in vx, vy)
    # Per Yang: generally yes, though for vxp and vyp it was calculated again
    # but the number should not change quite a bit. so it should be okay to
    # use a single value for all variables
    stable_count_slow: str = 'stable_count_slow'
    stable_count_mask: str = 'stable_count_stationary'

    # These names are created at runtime: based on "stable_shift"
    # attribute of vx and vy variables, needed for processing
    vx_stable_shift: str = f'{vx}_{postfix.stable_shift}'
    vy_stable_shift: str = f'{vy}_{postfix.stable_shift}'

    autorift_software_version: str = 'autoRIFT_software_version'

    # New data variables that to be added to already generated V2 datacubes
    ascending_img1: str = 'ascending_img1'
    ascending_img2: str = 'ascending_img2'

    # Specific to the datacube
    url: str = 'granule_url'

    # Standard name for variables to use
    name = {
        interp_mask: 'interpolated_value_mask',
        va: 'azimuth_velocity',
        vr: 'range_velocity',
        v_error: 'velocity_error',
        m11: 'conversion_matrix_element_11',
        m12: 'conversion_matrix_element_12',
        ascending_img1: 'image1_ascending_orbit',
        ascending_img2: 'image2_ascending_orbit',
    }

    # Description strings for all data variables and some
    # of their attributes.
    description = {
        v:  "velocity magnitude",
        vx: "velocity component in x direction",
        vy: "velocity component in y direction",

        stable_count_slow: "number of valid pixels over slowest 25% of ice",
        stable_count_mask: "number of valid pixels over stationary or "
                            "slow-flowing surfaces",

        postfix.stable_shift_slow:
            "{} shift calibrated using valid pixels over slowest 25% of "
            "retrieved velocities",
        postfix.stable_shift_mask:
            "{} shift calibrated using valid pixels over stable surfaces, "
            " stationary or slow-flowing surfaces with velocity < 15 m/yr "
            "identified from an external mask",

        # These descriptions are based on Radar granule format. Have to set them
        # manually since there are no Radar format granules are available for
        # processing just yet (otherwise these attributes would be automatically
        # picked up from the granules).
        va: "velocity in radar azimuth direction",
        vr: "velocity in radar range direction",
        m11: "conversion matrix element (1st row, 1st column) that can be "
            "multiplied with vx to give range pixel displacement dr (see "
            "Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)",
        m12: "conversion matrix element (1st row, 2nd column) that can be "
            "multiplied with vy to give range pixel displacement dr (see "
            "Eq. A18 in https://www.mdpi.com/2072-4292/13/4/749)",
        postfix.dr_to_vr_factor: "multiplicative factor that converts slant "
            "range pixel displacement dr to slant range velocity vr",
        v_error: "velocity magnitude error",
        interp_mask: "light interpolation mask",
        attrs.chip_size_coords:
            "Optical data: chip_size_coordinates = "
            "'image projection geometry: width = x, height = y'. Radar data: "
            "chip_size_coordinates = 'radar geometry: width = range, "
            "height = azimuth'",
        chip_size_height: "height of search template (chip)",
        chip_size_width:  "width of search template (chip)",
        flag_stable_shift:
            "flag for applying velocity bias correction: "
            "0 = no correction; "
            "1 = correction from overlapping stable surface mask (stationary "
            "or slow-flowing surfaces with velocity < 15 m/yr)(top priority); "
            "2 = correction from slowest 25% of overlapping velocities "
            "(second priority)",
        url: "original granule URL",
        autorift_software_version: "version of autoRIFT software",
        SkippedGranules.name: "skipped granules during datacube construction",
        ascending_img1: 'true = ascending orbit, false = descending orbit',
        ascending_img2: 'true = ascending orbit, false = descending orbit'
    }

    # Map of variables with integer data type
    intType = {
        interp_mask: np.ubyte,
        ascending_img1: np.uint8,
        ascending_img2: np.uint8,
        chip_size_height: np.uint16,
        chip_size_width: np.uint16,
        flag_stable_shift: np.uint8,
        stable_count_slow: np.uint16,
        stable_count_mask: np.uint16,
        v: np.int16,
        vx: np.int16,
        vy: np.int16,
        v_error: np.int16,
        va: np.int16,
        vr: np.int16,
    }

    # Missing value for data variables of integer data type
    intMissingValue = {
        interp_mask: utils.Missing.byte,
        ascending_img1: utils.Missing.u8value,
        ascending_img2: utils.Missing.u8value,
        chip_size_height: utils.Missing.byte,
        chip_size_width: utils.Missing.byte,
        v: utils.Missing.value,
        vx: utils.Missing.value,
        vy: utils.Missing.value,
        v_error: utils.Missing.value,
        va: utils.Missing.value,
        vr: utils.Missing.value,
    }

    # Standard names and descriptions for velocity error data variables
    # Each entry is a tuple of (std_name, description)
    errorAttrs = {
        'vx_error': (
            "x_velocity_error",
            "error for velocity component in x direction"
        ),
        'vy_error': (
            "y_velocity_error",
            "error for velocity component in y direction"
        ),
        'va_error': (
            "azimuth_velocity_error",
            "best estimate of azimuth_velocity error: va_error is populated "
            "according to the approach used for the velocity bias correction "
            "as indicated in \"stable_shift_flag\""
        ),
        'vr_error': (
            "range_velocity_error",
            "best estimate of range_velocity error: vr_error is populated "
            "according to the approach used for the velocity bias correction "
            "as indicated in \"stable_shift_flag\""
        ),
        # The following descriptions are the same for all v* data
        # variables
        'error_stationary': (
            None,
            "RMSE over stable surfaces, stationary or slow-flowing "
            "surfaces with velocity < 15 m/yr identified from an "
            "external mask"
        ),
        'error_slow': (
            None,
            "RMSE over slowest 25% of retrieved velocities"
        ),
        'error_modeled': (
            None,
            "1-sigma error calculated using a modeled error-dt relationship"
        )
    }

Vars = VarsInfo()


@dataclass(frozen=True)
class ImgPairInfoVars:
    """
    Class to represent the "img_pair_info" data variable, attributes of which
    become new data variables in the datacube to represent these attributes
    for all layers in the datacube.
    """
    name: str = 'img_pair_info'

    # Attributes of the "img_pair_info" data variable, which become new data
    # variables within the datacube
    mission_img1: str    = 'mission_img1'
    mission_img2: str    = 'mission_img2'
    sensor_img1: str     = 'sensor_img1'
    sensor_img2: str     = 'sensor_img2'
    satellite_img1: str  = 'satellite_img1'
    satellite_img2: str  = 'satellite_img2'
    date_dt: str         = 'date_dt'
    date_center: str     = 'date_center' # Rename mid_date to date_center as
                                        # they are the same, don't collect this
    acquisition_date_img1: str = 'acquisition_date_img1'
    acquisition_date_img2: str = 'acquisition_date_img2'
    roi_valid_percentage: str  = 'roi_valid_percentage'

    # New format defines these attributes, make them datacube attributes
    time_standard_img1: str = 'time_standard_img1'
    time_standard_img2: str = 'time_standard_img2'

    flight_direction_img1: str = 'flight_direction_img1'
    flight_direction_img2: str = 'flight_direction_img2'
    ascending: str = 'ascending'

    # Were in the old DataVars.ImgPairInfo format to support range-range
    # granules, keep them for now...
    # FLIGHT_DIRECTION_IMG1 = 'flight_direction_img1'
    # FLIGHT_DIRECTION_IMG2 = 'flight_direction_img2'
    # ASCENDING = 'ascending'
    # DESCENDING = 'descending'
    # Attributes for radar granules
    # ABSOLUTE_ORBIT_NUMBER_IMG1 = 'absolute_orbit_number_img1'
    # ABSOLUTE_ORBIT_NUMBER_IMG2 = 'absolute_orbit_number_img2'
    # ID_IMG1 = 'id_img1'
    # ID_IMG2 = 'id_img2'
    # PRODUCT_UNIQUE_ID_IMG1 = 'product_unique_ID_img1'
    # PRODUCT_UNIQUE_ID_IMG2 = 'product_unique_ID_img2'
    # MISSION_DATA_TAKE_ID_IMG1 = 'mission_data_take_ID_img1'
    # MISSION_DATA_TAKE_ID_IMG2 = 'mission_data_take_ID_img2'

    # Variables in datacube that correspond to attributes of "img_pair_info"
    # data variable in the granules
    all = [
        acquisition_date_img1,
        acquisition_date_img2,
        mission_img1,
        mission_img2,
        satellite_img1,
        satellite_img2,
        sensor_img1,
        sensor_img2,
        date_center,
        date_dt,
        roi_valid_percentage
    ]

    # Data types for the variables in datacube that correspond to attributes
    # of "img_pair_info" data variable in the granules
    # ATTN: Sentinel-2 granules are using satellite_img1 and satellite_img2 instead
    # of sensor_img1 and sensor_img2
    allTypes = {
        date_dt: np.float32,
        roi_valid_percentage: np.float32,
        mission_img1: np.dtypes.StringDType(),
        mission_img2: np.dtypes.StringDType(),
        satellite_img1: np.dtypes.StringDType(),
        satellite_img2: np.dtypes.StringDType(),
        sensor_img1: np.dtypes.StringDType(),
        sensor_img2: np.dtypes.StringDType()
    }

    # Units for the variables in datacube that correspond to attributes of
    # "img_pair_info" data variable in the granules
    allUnits = {
        date_dt: utils.Units.days
    }

    # Description strings for data variables.
    allDescriptions = {
        mission_img1: "id of the mission that acquired image 1",
        mission_img2: "id of the mission that acquired image 2",
        sensor_img1: "id of the sensor that acquired image 1",
        sensor_img2: "id of the sensor that acquired image 2",
        satellite_img1: "id of the satellite that acquired image 1",
        satellite_img2: "id of the satellite that acquired image 2",
        acquisition_date_img1: "acquisition date and time of image 1",
        acquisition_date_img2: "acquisition date and time of image 2",
        date_dt: "time separation between acquisition of image 1 and image 2",
        date_center: "midpoint of image 1 and image 2 acquisition date",
        roi_valid_percentage:
            "percentage of pixels with a valid velocity "
            "estimate determined for the intersection of the full image "
            "pair footprint and the region of interest (roi) that defines "
            "where autoRIFT tried to estimate a velocity",
    }

    # Data variables which values to be stored as date objects
    toDate = {
        acquisition_date_img1,
        acquisition_date_img2,
        date_center
    }

    stdName = {
        mission_img1: "image1_mission",
        mission_img2: "image2_mission",
        sensor_img1: "image1_sensor",
        sensor_img2: "image2_sensor",
        satellite_img1: "image1_satellite",
        satellite_img2: "image2_satellite",
        acquisition_date_img1: "image1_acquition_date",
        acquisition_date_img2: "image2_acquition_date",
        date_dt: "image_pair_time_separation",
        date_center: "image_pair_center_date",
        roi_valid_percentage: "region_of_interest_valid_pixel_percentage"
    }

ImgPairInfo = ImgPairInfoVars()


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
