import copy
from datetime import datetime, timedelta
import gc
import glob
import os
import shutil
import timeit

# Extra installed packages
import dask
# from dask.distributed import Client, performance_report
from dask.diagnostics import ProgressBar
import numpy  as np
import pandas as pd
import s3fs
from tqdm import tqdm
import xarray as xr

# Local modules
from itslive import itslive_ui
from grid import Bounds


class Coords:
    """
    Coordinates for the data cube.
    """
    MID_DATE = 'mid_date'
    X = 'x'
    Y = 'y'


class DataVars:
    """
    Data variables for the data cube.
    """
    # Attributes that appear for multiple data variables
    MISSING_VALUE_ATTR = 'missing_value'
    FILL_VALUE_ATTR    = '_FillValue'
    DESCRIPTION  = 'description'  # v, vx, vy
    GRID_MAPPING = 'grid_mapping' # v, vx, vy - store only one
    STABLE_COUNT = 'stable_count' # vx, vy    - store only one
    STABLE_SHIFT = 'stable_shift' # vx, vy
    FLAG_STABLE_SHIFT_MEANINGS = 'flag_stable_shift_meanings' # vx, vy

    # Optical Legacy format only:
    STABLE_SHIFT_APPLIED = 'stable_shift_applied' # vx, vy - remove from attributes
    STABLE_APPLY_DATE = 'stable_apply_date' # vx, vy - remove from attributes

    STD_NAME = 'standard_name'
    UNITS = 'units'
    M_Y_UNITS = 'm/y'

    # Original data variables and their attributes per ITS_LIVE granules.
    V = 'v'
    # Attributes
    MAP_SCALE_CORRECTED = 'map_scale_corrected'

    VX = 'vx'
    # Attributes
    VX_ERROR          = 'vx_error'          # In Radar and updated Optical formats
    FLAG_STABLE_SHIFT = 'flag_stable_shift' # In Radar and updated Optical formats
    STABLE_RMSE       = 'stable_rmse'       # In Optical legacy format only

    VY                = 'vy'
    # Attributes
    VY_ERROR          = 'vy_error'          # In Radar and updated Optical formats

    CHIP_SIZE_HEIGHT  = 'chip_size_height'
    CHIP_SIZE_WIDTH   = 'chip_size_width'
    # Attributes
    CHIP_SIZE_COORDS  = 'chip_size_coordinates'

    INTERP_MASK      = 'interp_mask'
    V_ERROR          = 'v_error' # Optical and Radar only
    VA               = 'va'
    VP               = 'vp'
    VP_ERROR         = 'vp_error'
    VR               = 'vr'
    VX               = 'vx'
    VXP              = 'vxp'
    VYP              = 'vyp'
    # Added for the datacube
    URL = 'url'

    # Missing values for data variables
    MISSING_BYTE      = 0.0
    MISSING_VALUE     = -32767.0
    MISSING_POS_VALUE = 32767.0

    V_DESCRIPTION_STR = 'velocity magnitude'
    VX_DESCRIPTION_STR = "velocity component in x direction"
    VY_DESCRIPTION_STR = "velocity component in y direction"

    # These descriptions are based on Radar granule format. Have to set them
    # manually since there are no Radar format granules are available for
    # processing yet (otherwise these attributes would be automatically picked up
    # from the granules).
    VA_DESCRIPTION = "velocity in radar azimuth direction"
    VR_DESCRIPTION = "velocity in radar range direction"
    VP_DESCRIPTION = "velocity magnitude determined by projecting radar " \
        "range measurements onto an a priori flow vector. Where projected " \
        "errors are larger than those determined from range and azimuth " \
        "measurements, unprojected v estimates are used"
    VP_ERROR_DESCRIPTION = "velocity magnitude error determined by projecting " \
        "radar range measurements onto an a priori flow vector. " \
        "Where projected errors are larger than those determined from range " \
        "and azimuth measurements, unprojected v_error estimates are used"
    VXP_DESCRIPTION = "x-direction velocity determined by projecting radar " \
        "range measurements onto an a priori flow vector. Where projected " \
        "errors are larger than those determined from range and azimuth " \
        "measurements, unprojected vx estimates are used"
    VYP_DESCRIPTION = "y-direction velocity determined by projecting radar " \
        "range measurements onto an a priori flow vector. Where projected " \
        "errors are larger than those determined from range and azimuth " \
        "measurements, unprojected vy estimates are used"

    V_ERROR_DESCRIPTION = 'velocity magnitude error'
    INTERP_MASK_DESCRIPTION = "light interpolation mask"
    CHIP_SIZE_COORDS_STR = "Optical data: chip_size_coordinates = " \
        "'image projection geometry: width = x, height = y'. Radar data: " \
        "chip_size_coordinates = 'radar geometry: width = range, height = azimuth'"
    CHIP_SIZE_HEIGHT_STR = "height of search window"
    CHIP_SIZE_WIDTH_STR  = "width of search window"
    FLAG_STABLE_SHIFT_MEANINGS_STR = \
        "flag for applying velocity bias correction over stable surfaces " \
        "(stationary or slow-flowing surfaces with velocity < 15 m/yr): " \
        "0 = there is no stable surface available and no correction is applied; " \
        "1 = there are stable surfaces and velocity bias is corrected;"


class ITSCube:
    """
    Class to build ITS_LIVE cube: time series of velocity pairs within a
    polygon of interest.
    """
    # Number of threads for parallel processing
    NUM_THREADS = 4

    # Dask scheduler for parallel processing
    DASK_SCHEDULER = "processes"

    # String representation of longitude/latitude projection
    LON_LAT_PROJECTION = '4326'

    S3_PREFIX = 's3://'
    HTTP_PREFIX = 'http://'

    # Token within granule's URL that needs to be removed to get file location within S3 bucket:
    # if URL is of the 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0/32628/file.nc' format,
    # S3 bucket location of the file is 's3://its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0/32628/file.nc'
    PATH_URL = ".s3.amazonaws.com"

    # Engine to read xarray data into from NetCDF file
    NC_ENGINE = 'h5netcdf'

    # Date format as it appears in granules filenames:
    # (LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc)
    DATE_FORMAT = "%Y%m%d"

    # Granules are written to the file in chunks to avoid out of memory issues.
    # Number of granules to write to the file at a time.
    NUM_GRANULES_TO_WRITE = 1000

    def __init__(self, polygon: tuple, projection: str):
        """
        Initialize object.

        polygon: tuple
            Polygon for the tile.
        projection: str
            Projection in which polygon is defined.
        """
        self.projection = projection

        # Set min/max x/y values to filter region by
        self.x = Bounds([each[0] for each in polygon])
        self.y = Bounds([each[1] for each in polygon])

        # Convert polygon from its target projection to longitude/latitude coordinates
        # which are used by granule search API
        self.polygon_coords = []

        for each in polygon:
            coords = itslive_ui.transform_coord(
                projection,
                ITSCube.LON_LAT_PROJECTION,
                each[0], each[1]
            )
            self.polygon_coords.extend(coords)

        print(f"Longitude/latitude coords for polygon: {self.polygon_coords}")

        # Lists to store filtered by region/start_date/end_date velocity pairs
        # and corresponding metadata (middle dates (+ date separation in days as milliseconds),
        # original granules URLs)
        self.ds = []

        self.dates = []
        self.urls = []
        self.num_urls_from_api = None

        # Keep track of skipped granules due to the other than target projection
        self.skipped_proj_granules = {}
        # Keep track of skipped granules due to no data for the polygon of interest
        self.skipped_empty_granules = []
        # Keep track of "double" granules with older processing date which are
        # not included into the cube
        self.skipped_double_granules = []

        # Constructed cube
        self.layers = None

    def clear_vars(self):
        """
        Clear current set of cube layers.
        """
        self.ds = None

        self.layers = None
        self.dates = []
        self.urls = []

        gc.collect()

        self.ds = []

    def clear(self):
        """
        Reset all internal data structures.
        """
        self.clear_vars()

        self.num_urls_from_api = None
        self.skipped_proj_granules = {}
        self.skipped_empty_granules = []
        self.skipped_double_granules = []

    def request_granules(self, api_params: dict, num_granules: int):
        """
        Send request to ITS_LIVE API to get a list of granules to satisfy polygon request.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        # Append polygon information to API's parameters
        params = copy.deepcopy(api_params)
        params['polygon'] = ",".join([str(each) for each in self.polygon_coords])

        start_time = timeit.default_timer()
        found_urls = [each['url'] for each in itslive_ui.get_granule_urls(params)]
        total_num = len(found_urls)
        time_delta = timeit.default_timer() - start_time
        print(f"Number of found by API granules: {total_num} (took {time_delta} seconds)")

        if len(found_urls) == 0:
            raise RuntimeError(f"No granules are found for the search API parameters: {params}")

        # Number of granules to examine is specified
        # TODO: just a workaround for now as it's very slow to examine all granules
        #       sequentially at this point.
        if num_granules:
            found_urls = found_urls[:num_granules]
            print(f"Examining only first {len(found_urls)} out of {total_num} found granules")

        return self.skip_duplicate_granules(found_urls)

    def skip_duplicate_granules(self, found_urls):
        """
        Skip duplicate granules (the ones that have earlier processing date(s)).
        """
        self.num_urls_from_api = len(found_urls)

        # Need to remove duplicate granules for the middle date: some granules
        # have newer processing date, keep those.
        url_mid_dates = []
        keep_urls = []
        self.skipped_double_granules = []

        for each_url in tqdm(found_urls, ascii=True, desc='Skipping duplicate granules...'):
            # Extract acquisition and processing dates
            url_acq_1, url_proc_1, url_acq_2, url_proc_2 = \
                ITSCube.get_dates_from_filename(each_url)

            day_separation = (url_acq_1 - url_acq_2).days
            mid_date = url_acq_2 + timedelta(days=day_separation/2, milliseconds=day_separation)

            # There is a granule for the mid_date already, check which processing
            # time is newer, keep the one with newer processing date
            if mid_date in url_mid_dates:
                index = url_mid_dates.index(mid_date)
                found_url = keep_urls[index]

                found_acq_1, found_proc_1, found_acq_2, found_proc_2 = \
                    ITSCube.get_dates_from_filename(found_url)

                # It is allowed for the same image pair only
                if url_acq_1 != found_acq_1 or \
                    url_acq_2 != found_acq_2:
                    raise RuntimeError(f"Found duplicate granule for {mid_date} that differs in acquisition time: {url_acq_1} != {found_acq_1} or {url_acq_2} != {found_acq_2} ({each_url} vs. {found_url})")

                if url_proc_1 >= found_proc_1 and \
                   url_proc_2 >= found_proc_2:
                    # Replace the granule with newer processed one
                    keep_urls[index] = each_url
                    self.skipped_double_granules.append(found_url)

                else:
                    # New granule has older processing date, don't include
                    self.skipped_double_granules.append(each_url)

            else:
                # This is new mid_date, append information
                url_mid_dates.append(mid_date)
                keep_urls.append(each_url)

        print (f"Keeping {len(keep_urls)} unique granules")
        return keep_urls

    @staticmethod
    def get_dates_from_filename(filename):
        """
        Extract acquisition and processing dates for two images from the filename.
        """
        # Get acquisition and processing date for both images from url and index_url
        url_tokens = os.path.basename(filename).split('_')
        url_acq_date_1 = datetime.strptime(url_tokens[3], ITSCube.DATE_FORMAT)
        url_proc_date_1 = datetime.strptime(url_tokens[4], ITSCube.DATE_FORMAT)
        url_acq_date_2 = datetime.strptime(url_tokens[11], ITSCube.DATE_FORMAT)
        url_proc_date_2 = datetime.strptime(url_tokens[12], ITSCube.DATE_FORMAT)

        return url_acq_date_1, url_proc_date_1, url_acq_date_2, url_proc_date_2

    def add_layer(self, is_empty, layer_projection, mid_date, url, data):
        """
        Examine the layer if it qualifies to be added as a cube layer.
        """

        if data is not None:
            # TODO: Handle "duplicate" granules for the mid_date if concatenating
            #       to existing cube.
            #       "Duplicate" granules are handled apriori for newly constructed
            #       cubes (see self.request_granules() method).
            # print(f"Adding {url} for {mid_date}")
            self.dates.append(mid_date)
            self.ds.append(data)
            self.urls.append(url)

        else:
            if is_empty:
                # Layer does not contain valid data for the region
                self.skipped_empty_granules.append(url)

            else:
                # Layer corresponds to other than target projection
                self.skipped_proj_granules.setdefault(layer_projection, []).append(url)

    def create(self, api_params: dict, output_dir: str, num_granules=None):
        """
        Create velocity cube.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules.
                  Should not be used when running the code in production mode.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()

        found_urls = self.request_granules(api_params, num_granules)

        # Open S3FS access to S3 bucket with input granules
        s3 = s3fs.S3FileSystem(anon=True)

        is_first_write = True
        for each_url in tqdm(found_urls, ascii=True, desc='Reading and processing S3 granules'):
            s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
            s3_path = s3_path.replace(ITSCube.PATH_URL, '')

            with s3.open(s3_path, mode='rb') as fhandle:
                with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                    results = self.preprocess_dataset(ds, each_url)
                    self.add_layer(*results)

                    # Check if need to write to the file accumulated number of granules
                    if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                        self.combine_layers(output_dir, is_first_write)
                        is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        # Report statistics for skipped granules
        self.format_stats()

        return found_urls

    def create_parallel(self, api_params: dict, output_dir: str, num_granules=None):
        """
        Create velocity cube by reading and pre-processing cube layers in parallel.

        api_params: dict
            Search API required parameters.
        num_granules: int
            Number of first granules to examine.
            TODO: This is a temporary solution to a very long time to open remote granules. Should not be used
                  when running the code at AWS.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()
        found_urls = self.request_granules(api_params, num_granules)

        # Parallelize layer collection
        s3 = s3fs.S3FileSystem(anon=True)

        # In order to enable Dask profiling, need to create Dask client for
        # processing: using "processes" or "threads" scheduler
        # processes_scheduler = True if ITSCube.DASK_SCHEDULER == 'processes' else False
        # client = Client(processes=processes_scheduler, n_workers=ITSCube.NUM_THREADS)
        # # Use client to collect profile information
        # client.profile(filename=f"dask-profile-{num_granules}-parallel.html")
        is_first_write = True
        start = 0
        num_to_process = len(found_urls)

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            tasks = [dask.delayed(self.read_s3_dataset)(each_file, s3) for each_file in found_urls[start:start+num_tasks]]
            print(f"Processing {len(tasks)} tasks")

            results = None
            with ProgressBar():
                # If to collect performance report (need to define global Client - see above)
                # with performance_report(filename=f"dask-report-{num_granules}.html"):
                #     results = dask.compute(tasks)
                results = dask.compute(
                    tasks,
                    scheduler=ITSCube.DASK_SCHEDULER,
                    num_workers=ITSCube.NUM_THREADS
                )

            del tasks
            gc.collect()

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            del results
            gc.collect()

            self.combine_layers(output_dir, is_first_write)

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()
        return found_urls

    @staticmethod
    def ds_to_netcdf(ds: xr.Dataset, filename: str):
        """
        Write datacube xarray.Dataset to the NetCDF file.
        """
        if ds is not None:
            ds.to_netcdf(filename, engine=ITSCube.NC_ENGINE, unlimited_dims=(Coords.MID_DATE))

        else:
            raise RuntimeError(f"Datacube data does not exist.")

    def create_from_local_no_api(self, output_dir: str, dirpath='data'):
        """
        Create velocity cube by accessing local data stored in "dirpath" directory.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        found_urls = self.skip_duplicate_granules(found_urls)
        is_first_write = True

        # Number of granules to examine is specified (it's very slow to examine all granules sequentially)
        for each_url in tqdm(found_urls, ascii=True, desc='Processing local granules'):
            with xr.open_dataset(each_url) as ds:
                results = self.preprocess_dataset(ds, each_url)
                self.add_layer(*results)
                # Check if need to write to the file accumulated number of granules
                if len(self.urls) == ITSCube.NUM_GRANULES_TO_WRITE:
                    self.combine_layers(output_dir, is_first_write)
                    is_first_write = False

        # Check if there are remaining layers to be written to the file
        if len(self.urls):
            self.combine_layers(output_dir, is_first_write)

        self.format_stats()

        return found_urls

    def create_from_local_parallel_no_api(self, output_dir: str, dirpath='data'):
        """
        Create velocity cube from local data stored in "dirpath" in parallel.

        dirpath: str
            Directory that stores granules files. Default is 'data' sub-directory
            accessible from the directory the code is running from.
        """
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        self.clear()
        found_urls = glob.glob(dirpath + os.sep + '*.nc')
        found_urls = self.skip_duplicate_granules(found_urls)

        num_to_process = len(found_urls)

        is_first_write = True
        start = 0
        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSCube.NUM_GRANULES_TO_WRITE if num_to_process > ITSCube.NUM_GRANULES_TO_WRITE else num_to_process
            print("NUM to process: ", num_tasks)

            tasks = [dask.delayed(self.read_dataset)(each_file) for each_file in found_urls[start:start+num_tasks]]
            assert len(tasks) == num_tasks
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler=ITSCube.DASK_SCHEDULER,
                                       num_workers=ITSCube.NUM_THREADS)

            for each_ds in results[0]:
                self.add_layer(*each_ds)

            self.combine_layers(output_dir, is_first_write)

            if start == 0:
                is_first_write = False

            num_to_process -= num_tasks
            start += num_tasks

        self.format_stats()

        return found_urls

    def get_data_var(self, ds: xr.Dataset, var_name: str):
        """
        Return xr.DataArray that corresponds to the data variable if it exists
        in the 'ds' dataset, or empty xr.DataArray if it is not present in the 'ds'.
        Empty xr.DataArray assumes the same dimensions as ds.v data array.
        """

        if var_name in ds:
            return ds[var_name]

        # Create empty array as it is not provided in the granule,
        # use the same coordinates as for any cube's data variables.
        # ATTN: This assumes that self.layers already contains 'v' data variable.
        return xr.DataArray(
            data=None,
            coords=[self.layers.v.coords[Coords.Y], self.layers.v.coords[Coords.X]],
            dims=[Coords.Y, Coords.X]
        )

    def get_data_var_attr(self, ds: xr.Dataset, var_name: str, attr_name: str, missing_value: int = None):
        """
        Return a list of attributes for the data variable in data set if it exists,
        or missing_value if it is not present.
        If missing_value is set to None, than attribute is expected to exist for
        the data variable "var_name".
        """
        if var_name in ds and attr_name in ds[var_name].attrs:
            return ds[var_name].attrs[attr_name][0]

        if missing_value is None:
            # If missing_value is not provided, attribute is expected to exist always
            raise RuntimeError(f"{attr_name} is expected within {var_name} for {ds}")

        return missing_value

    def preprocess_dataset(self, ds: xr.Dataset, ds_url: str):
        """
        Pre-process ITS_LIVE dataset in preparation for the cube layer.

        ds: xarray dataset
            Dataset to pre-process.
        ds_url: str
            URL that corresponds to the dataset.

        Returns:
        cube_v:     Filtered data array for the layer.
        mid_date:   Middle date that corresponds to the velicity pair (uses date
                    separation as milliseconds)
        empty:      Flag to indicate if dataset does not contain any data for
                    the cube region.
        projection: Source projection for the dataset.
        url:        Original URL for the granule (have to return for parallel
                    processing: no track of inputs for each task, but have output
                    available for each task).
        """
        # Try to load the whole dataset into memory to avoid penalty for random read access
        # when accessing S3 bucket (?)
        # ds.load()

        # Flag if layer data is empty
        empty = False

        # Layer data
        mask_data = None

        # Layer middle date
        mid_date = None

        # Consider granules with data only within target projection
        if str(int(ds.UTM_Projection.spatial_epsg)) == self.projection:
            mid_date = datetime.strptime(ds.img_pair_info.date_center, '%Y%m%d')

            # Add date separation in days as milliseconds for the middle date
            # (avoid resolution issues for layers with the same middle date).
            mid_date += timedelta(milliseconds=int(ds.img_pair_info.date_dt))

            # Define which points are within target polygon.
            mask_lon = (ds.x >= self.x.min) & (ds.x <= self.x.max)
            mask_lat = (ds.y >= self.y.min) & (ds.y <= self.y.max)
            mask_data = ds.where(mask_lon & mask_lat, drop=True)

            # Another way to filter:
            # cube_v = ds.v.sel(x=slice(self.x.min, self.x.max),y=slice(self.y.max, self.y.min)).copy()

            # If it's a valid velocity layer, add it to the cube.
            if np.any(mask_data.v.notnull()):
                mask_data.load()

            else:
                # Reset cube back to None as it does not contain any valid data
                mask_data = None
                mid_date = None
                empty = True

        # Have to return URL for the dataset, which is provided as an input to the method,
        # to track URL per granule in parallel processing
        return empty, int(ds.UTM_Projection.spatial_epsg), mid_date, ds_url, \
            mask_data

    def process_v_attributes(self, var_name: str, is_first_write: bool, mid_date_coord):
        """
        Helper method to clean up attributes for v-related data variables.
        """
        _stable_rmse_vars = [DataVars.VX, DataVars.VY]

        # Dictionary of attributes values for new v*_error data variables:
        # std_name, description
        _attrs = {
            'vx_error': ("x_velocity_error", "error for velocity component in x direction"),
            'vy_error': ("y_velocity_error", "error for velocity component in y direction"),
            'va_error': ("azimuth_velocity_error", "error for velocity in radar azimuth direction"),
            'vr_error': ("range_velocity_error", "error for velocity in radar range direction"),
            'vxp_error': ("projected_x_velocity_error", "error for x-direction velocity determined by projecting radar range measurements onto an a priori flow vector"),
            'vyp_error': ("projected_y_velocity_error", "error for y-direction velocity determined by projecting radar range measurements onto an a priori flow vector")
        }

        # Process attributes
        if DataVars.STABLE_APPLY_DATE in self.layers[var_name].attrs:
            # Remove optical legacy attribute if it propagated to the cube data
            del self.layers[var_name].attrs[DataVars.STABLE_APPLY_DATE]

        # If attribute is propagated as cube's data var attribute, delete it.
        # These attributes were collected based on 'v' data variable
        if DataVars.GRID_MAPPING in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.GRID_MAPPING]
        if DataVars.MAP_SCALE_CORRECTED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.MAP_SCALE_CORRECTED]
        if DataVars.STABLE_SHIFT_APPLIED in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT_APPLIED]

        _name_sep = '_'
        error_name = _name_sep.join([var_name, 'error'])

        # Special care must be taken of v[xy].stable_rmse in
        # optical legacy format vs. v[xy].v[xy]_error in radar format as these
        # are the same
        error_data = None
        if var_name in _stable_rmse_vars:
            error_data = [
                self.get_data_var_attr(ds, var_name, error_name, DataVars.MISSING_VALUE) if error_name in ds[var_name].attrs else
                self.get_data_var_attr(ds, var_name, DataVars.STABLE_RMSE, DataVars.MISSING_VALUE) for ds in self.ds
            ]

            # If attribute is propagated as cube's data var attribute, delete it
            if DataVars.STABLE_RMSE in self.layers[var_name].attrs:
                del self.layers[var_name].attrs[DataVars.STABLE_RMSE]

        else:
            error_data = [self.get_data_var_attr(ds, var_name, error_name, DataVars.MISSING_VALUE) for ds in self.ds]

        self.layers[error_name] = xr.DataArray(
            data=error_data,
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE],
            attrs={
                DataVars.UNITS: DataVars.M_Y_UNITS,
                DataVars.STD_NAME: _attrs[error_name][0],
                DataVars.DESCRIPTION: _attrs[error_name][1]
            }
        )

        # If attribute is propagated as cube's data var attribute, delete it
        if error_name in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[error_name]

        # This attribute appears for all v* data variables, capture it only once
        if DataVars.STABLE_COUNT not in self.layers:
            self.layers[DataVars.STABLE_COUNT] = xr.DataArray(
                data=[self.get_data_var_attr(ds, var_name, DataVars.STABLE_COUNT) for ds in self.ds],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE]
            )
        if DataVars.STABLE_COUNT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_COUNT]

        # This flag appears for vx and vy data variables, capture it only once.
        # "stable_shift_applied" was incorrectly set in the optical legacy dataset
        # and should be set to the no data value
        if DataVars.FLAG_STABLE_SHIFT not in self.layers:
            missing_stable_shift_value = 0.0
            self.layers[DataVars.FLAG_STABLE_SHIFT] = xr.DataArray(
                data=[self.get_data_var_attr(ds, var_name, DataVars.FLAG_STABLE_SHIFT, missing_stable_shift_value) for ds in self.ds],
                coords=[mid_date_coord],
                dims=[Coords.MID_DATE]
            )
            # Set flag meaning description
            self.layers[DataVars.FLAG_STABLE_SHIFT].attrs[DataVars.FLAG_STABLE_SHIFT_MEANINGS] = DataVars.FLAG_STABLE_SHIFT_MEANINGS_STR

        # Create data variable specific 'stable_shift' data variable,
        # for example, 'vx_stable_shift' for 'vx' data variable
        shift_var_name = _name_sep.join([var_name, DataVars.STABLE_SHIFT])
        self.layers[shift_var_name] = xr.DataArray(
            data=[self.get_data_var_attr(ds, var_name, DataVars.STABLE_SHIFT, DataVars.MISSING_VALUE) for ds in self.ds],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE]
        )
        # If attribute is propagated as cube's vx attribute, delete it
        if DataVars.STABLE_SHIFT in self.layers[var_name].attrs:
            del self.layers[var_name].attrs[DataVars.STABLE_SHIFT]

        # Return names of new data variables - to be included into "encoding" settings
        # for writing to the file store.
        return [error_name, shift_var_name]

    def combine_layers(self, output_dir, is_first_write=False):
        """
        Combine selected layers into one xr.Dataset object and write (append) it
        to the Zarr store.
        """
        self.layers = {}

        # Construct xarray to hold layers by concatenating layer objects along 'mid_date' dimension
        print(f'Combine {len(self.urls)} layers to the {output_dir}...')
        start_time = timeit.default_timer()
        mid_date_coord = pd.Index(self.dates, name=Coords.MID_DATE)

        v_layers = xr.concat([each_ds.v for each_ds in self.ds], mid_date_coord)

        now_date = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        self.layers = xr.Dataset(
            data_vars = {DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords = {
                Coords.MID_DATE: self.dates,
                Coords.X: v_layers.coords[Coords.X],
                Coords.Y: v_layers.coords[Coords.Y]
            },
            attrs = {
                'title': 'ITS_LIVE datacube of image_pair velocities',
                'author': 'ITS_LIVE, a NASA MeASUREs project (its-live.jpl.nasa.gov)',
                'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                'date_created': now_date,
                'date_updated': now_date,
                'GDAL_AREA_OR_POINT': 'Area',
                'projection': str(self.projection)
            }
        )

        # ATTN: Assign one data variable at a time to avoid running out of memory.
        #       Delete each variable after it has been processed to free up the
        #       memory.

        # Process 'v'
        self.layers[DataVars.V] = v_layers
        self.layers[DataVars.V].attrs[DataVars.DESCRIPTION] = DataVars.V_DESCRIPTION_STR
        new_v_vars = [DataVars.V]

        # Collect 'v' attributes: these repeat for v, vx, vy, keep only one copy
        # per datacube
        self.layers[DataVars.GRID_MAPPING] = xr.DataArray(
            data=[ds.v.attrs[DataVars.GRID_MAPPING] for ds in self.ds],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE]
        )
        # If attribute is propagated as cube's v attribute, delete it
        if DataVars.GRID_MAPPING in self.layers[DataVars.V].attrs:
            del self.layers[DataVars.V].attrs[DataVars.GRID_MAPPING]

        # Create new data var to store map_scale_corrected v's attribute
        self.layers[DataVars.MAP_SCALE_CORRECTED] = xr.DataArray(
            data=[self.get_data_var_attr(ds, DataVars.V, DataVars.MAP_SCALE_CORRECTED, DataVars.MISSING_BYTE) for ds in self.ds],
            coords=[mid_date_coord],
            dims=[Coords.MID_DATE]
        )
        if is_first_write:
            # Set missing_value only on first write to the disk store, otherwise
            # will get "ValueError: failed to prevent overwriting existing key missing_value in attrs."
            self.layers[DataVars.MAP_SCALE_CORRECTED].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

        # If attribute is propagated as cube's v attribute, delete it
        if DataVars.MAP_SCALE_CORRECTED in self.layers[DataVars.V].attrs:
            del self.layers[DataVars.V].attrs[DataVars.MAP_SCALE_CORRECTED]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [each.drop_vars(DataVars.V) for each in self.ds]
        del v_layers
        gc.collect()

        # Process 'vx'
        self.layers[DataVars.VX] = xr.concat([ds.vx for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VX].attrs[DataVars.DESCRIPTION] = DataVars.VX_DESCRIPTION_STR
        new_v_vars.append(DataVars.VX)
        new_v_vars.extend(self.process_v_attributes(DataVars.VX, is_first_write, mid_date_coord))
        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VX) for ds in self.ds]
        gc.collect()

        # Process 'vy'
        self.layers[DataVars.VY] = xr.concat([ds.vy for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VY].attrs[DataVars.DESCRIPTION] = DataVars.VY_DESCRIPTION_STR
        new_v_vars.append(DataVars.VY)
        new_v_vars.extend(self.process_v_attributes(DataVars.VY, is_first_write, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.VY) for ds in self.ds]
        gc.collect()

        # Process 'va'
        self.layers[DataVars.VA] = xr.concat([self.get_data_var(ds, DataVars.VA) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VA].attrs[DataVars.DESCRIPTION] = DataVars.VA_DESCRIPTION
        new_v_vars.append(DataVars.VA)
        new_v_vars.extend(self.process_v_attributes(DataVars.VA, is_first_write, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VA) if DataVars.VA in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vr'
        self.layers[DataVars.VR] = xr.concat([self.get_data_var(ds, DataVars.VR) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VR].attrs[DataVars.DESCRIPTION] = DataVars.VR_DESCRIPTION
        new_v_vars.append(DataVars.VR)
        new_v_vars.extend(self.process_v_attributes(DataVars.VR, is_first_write, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VR) if DataVars.VR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vxp'
        self.layers[DataVars.VXP] = xr.concat([self.get_data_var(ds, DataVars.VXP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VXP].attrs[DataVars.DESCRIPTION] = DataVars.VXP_DESCRIPTION
        new_v_vars.append(DataVars.VXP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VXP, is_first_write, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VXP) if DataVars.VXP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vyp'
        self.layers[DataVars.VYP] = xr.concat([self.get_data_var(ds, DataVars.VYP) for ds in self.ds], mid_date_coord)
        self.layers[DataVars.VYP].attrs[DataVars.DESCRIPTION] = DataVars.VYP_DESCRIPTION
        new_v_vars.append(DataVars.VYP)
        new_v_vars.extend(self.process_v_attributes(DataVars.VYP, is_first_write, mid_date_coord))

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VYP) if DataVars.VYP in ds else ds for ds in self.ds]
        gc.collect()

        # Process chip_size_height
        self.layers[DataVars.CHIP_SIZE_HEIGHT] = xr.concat([ds.chip_size_height for ds in self.ds], mid_date_coord)
        if is_first_write:
            self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

        # Collect 'chip_size_height' attributes
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.CHIP_SIZE_COORDS] = DataVars.CHIP_SIZE_COORDS_STR
        self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.DESCRIPTION] = DataVars.CHIP_SIZE_HEIGHT_STR
        # If attribute is propagated as cube's data attribute, delete it
        if DataVars.GRID_MAPPING in self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs:
            del self.layers[DataVars.CHIP_SIZE_HEIGHT].attrs[DataVars.GRID_MAPPING]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_HEIGHT) for ds in self.ds]
        gc.collect()

        # Process chip_size_width
        self.layers[DataVars.CHIP_SIZE_WIDTH] = xr.concat([ds.chip_size_width for ds in self.ds], mid_date_coord)
        if is_first_write:
            self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

        # Collect 'chip_size_width' attributes
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.CHIP_SIZE_COORDS] = DataVars.CHIP_SIZE_COORDS_STR
        self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.DESCRIPTION] = DataVars.CHIP_SIZE_WIDTH_STR
        # If attribute is propagated as cube's data attribute, delete it
        if DataVars.GRID_MAPPING in self.layers[DataVars.CHIP_SIZE_WIDTH].attrs:
            del self.layers[DataVars.CHIP_SIZE_WIDTH].attrs[DataVars.GRID_MAPPING]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.CHIP_SIZE_WIDTH) for ds in self.ds]
        gc.collect()

        # Process interp_mask
        self.layers[DataVars.INTERP_MASK] = xr.concat([ds.interp_mask for ds in self.ds], mid_date_coord)
        self.layers[DataVars.INTERP_MASK].attrs[DataVars.DESCRIPTION] = DataVars.INTERP_MASK_DESCRIPTION

        if is_first_write:
            self.layers[DataVars.INTERP_MASK].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_BYTE

        # If attribute is propagated as cube's data attribute, delete it as
        # it's a data variable on its own.
        if DataVars.GRID_MAPPING in self.layers[DataVars.INTERP_MASK].attrs:
            del self.layers[DataVars.INTERP_MASK].attrs[DataVars.GRID_MAPPING]

        # Drop data variable as we don't need it anymore - free up memory
        self.ds = [ds.drop_vars(DataVars.INTERP_MASK) for ds in self.ds]
        gc.collect()

        # Process 'v_error'
        self.layers[DataVars.V_ERROR] = xr.concat([self.get_data_var(ds, DataVars.V_ERROR) for ds in self.ds] , mid_date_coord)
        new_v_vars.append(DataVars.V_ERROR)

        # Collect 'v_error' attributes
        # If attribute is propagated as cube's data attribute, delete it as
        # it's a data variable on its own.
        if DataVars.GRID_MAPPING in self.layers[DataVars.V_ERROR].attrs:
            del self.layers[DataVars.V_ERROR].attrs[DataVars.GRID_MAPPING]

        self.layers[DataVars.V_ERROR].attrs[DataVars.DESCRIPTION] = DataVars.V_ERROR_DESCRIPTION
        self.layers[DataVars.V_ERROR].attrs[DataVars.STD_NAME] = 'velocity_error'
        self.layers[DataVars.V_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.V_ERROR) if DataVars.V_ERROR in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vp'
        self.layers[DataVars.VP] = xr.concat([self.get_data_var(ds, DataVars.VP) for ds in self.ds] , mid_date_coord)
        new_v_vars.append(DataVars.VP)

        # Collect 'vp' attributes
        # If attribute is propagated as cube's data attribute, delete it as
        # it's a data variable on its own.
        if DataVars.GRID_MAPPING in self.layers[DataVars.VP].attrs:
            del self.layers[DataVars.VP].attrs[DataVars.GRID_MAPPING]

        self.layers[DataVars.VP].attrs[DataVars.DESCRIPTION] = DataVars.VP_DESCRIPTION
        self.layers[DataVars.VP].attrs[DataVars.STD_NAME] = 'projected_velocity'
        self.layers[DataVars.VP].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP) if DataVars.VP in ds else ds for ds in self.ds]
        gc.collect()

        # Process 'vp_error'
        self.layers[DataVars.VP_ERROR] = xr.concat([self.get_data_var(ds, DataVars.VP_ERROR) for ds in self.ds] , mid_date_coord)
        new_v_vars.append(DataVars.VP_ERROR)

        # Collect 'vp' attributes
        # If attribute is propagated as cube's data attribute, delete it as
        # it's a data variable on its own.
        if DataVars.GRID_MAPPING in self.layers[DataVars.VP].attrs:
            del self.layers[DataVars.VP].attrs[DataVars.GRID_MAPPING]

        self.layers[DataVars.VP_ERROR].attrs[DataVars.DESCRIPTION] = DataVars.VP_ERROR_DESCRIPTION
        self.layers[DataVars.VP_ERROR].attrs[DataVars.STD_NAME] = 'projected_velocity_error'
        self.layers[DataVars.VP_ERROR].attrs[DataVars.UNITS] = DataVars.M_Y_UNITS

        # Drop data variable as we don't need it anymore - free up memory
        # Drop only from datasets that have it
        self.ds = [ds.drop_vars(DataVars.VP) if DataVars.VP in ds else ds for ds in self.ds]
        gc.collect()


        time_delta = timeit.default_timer() - start_time
        print(f"Combined {len(self.urls)} layers (took {time_delta} seconds)")

        start_time = timeit.default_timer()
        # Write to the Zarr store
        if is_first_write:
            # ATTN: Must set _FillValue attribute for each data variable that has
            #       its missing_value attribute set
            encoding_settings = {DataVars.MAP_SCALE_CORRECTED: {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE},
                                 DataVars.INTERP_MASK: {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE},
                                 DataVars.CHIP_SIZE_HEIGHT: {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE},
                                 DataVars.CHIP_SIZE_WIDTH: {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE},
                                 DataVars.V_ERROR: {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE}}
            for each in new_v_vars:
                encoding_settings[each] = {DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE}

                # Set missing_value only on first write to the disk store, otherwise
                # will get "ValueError: failed to prevent overwriting existing key missing_value in attrs."
                if DataVars.MISSING_VALUE_ATTR not in self.layers[each].attrs:
                    self.layers[each].attrs[DataVars.MISSING_VALUE_ATTR] = DataVars.MISSING_VALUE

            # This is first write, create Zarr store
            self.layers.to_zarr(output_dir, encoding = encoding_settings)

        else:
            # Append layers to existing Zarr store
            self.layers.to_zarr(output_dir, append_dim=Coords.MID_DATE)

        time_delta = timeit.default_timer() - start_time
        print(f"Wrote {len(self.urls)} layers to {output_dir} (took {time_delta} seconds)")

        # Free up memory
        self.clear_vars()

        # No need to sort data by date as we will be appending layers to the datacubes

    def format_stats(self):
        """
        Format statistics of the run.
        """
        num_urls = self.num_urls_from_api
        # Total number of skipped granules due to wrong projection
        sum_projs = sum([len(each) for each in self.skipped_proj_granules.values()])

        print( "Skipped granules:")
        print(f"      empty data       : {len(self.skipped_empty_granules)} ({100.0 * len(self.skipped_empty_granules)/num_urls}%)")
        print(f"      wrong projection : {sum_projs} ({100.0 * sum_projs/num_urls}%)")
        print(f"      double mid_date  : {len(self.skipped_double_granules)} ({100.0 * len(self.skipped_double_granules)/num_urls}%)")
        if len(self.skipped_proj_granules):
            print(f"      wrong projections: {sorted(self.skipped_proj_granules.keys())}")

    def read_dataset(self, url: str):
        """
        Read Dataset from the file and pre-process for the cube layer.
        """
        with xr.open_dataset(url) as ds:
            return self.preprocess_dataset(ds, url)

    def read_s3_dataset(self, each_url: str, s3):
        """
        Read Dataset from the S3 bucket and pre-process for the cube layer.
        """
        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

        with s3.open(s3_path, mode='rb') as fhandle:
            with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as ds:
                return self.preprocess_dataset(ds, each_url)

    @staticmethod
    def plot(cube, variable, boundaries: tuple = None):
        """
        Plot cube's layers data. All layers share the same x/y coordinate labels.
        There is an option to display only a subset of layers by specifying
        start and end index through "boundaries" input parameter.
        """
        if boundaries is not None:
            start, end = boundaries
            cube[variable][start:end].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)

        else:
            cube[variable].plot(
                x=Coords.X,
                y=Coords.Y,
                col=Coords.MID_DATE,
                col_wrap=5,
                levels=100)


if __name__ == '__main__':
    # Since port forwarding is not working on EC2 to run jupyter lab for now,
    # allow to run test case from itscube.ipynb in standalone mode
    import argparse
    import warnings
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSCube.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='number of threads to use for parallel processing.')
    parser.add_argument('-s', '--scheduler', type=str, default="processes",
                        help="Dask scheduler to use. One of ['threads', 'processes'] (effective only when -p option is specified).")
    parser.add_argument('-p', '--parallel', action='store_true',
                        help='enable parallel processing')
    parser.add_argument('-n', '--numberGranules', type=int, required=False, default=None,
                        help="number of ITS_LIVE granules to consider for the cube (due to runtime limitations). "
                             " If none is provided, process all found granules.")
    parser.add_argument('-l', '--localPath', type=str, default=None,
                        help='Local path that stores ITS_LIVE granules.')
    parser.add_argument('-o', '--outputDir', type=str, default="cubedata.zarr",
                        help="Zarr output directory to write cube data to. Default is 'cubedata.zarr'.")
    parser.add_argument('-c', '--chunks', type=int, default=1000,
                        help="Number of granules to write at a time. Default is 1000.")

    args = parser.parse_args()
    ITSCube.NUM_THREADS = args.threads
    ITSCube.DASK_SCHEDULER = args.scheduler
    ITSCube.NUM_GRANULES_TO_WRITE = args.chunks

    # Test Case from itscube.ipynb:
    # =============================
    # Create polygon as a square around the centroid in target '32628' UTM projection
    # Projection for the polygon coordinates
    projection = '32628'

    # Centroid for the tile in target projection
    c_x, c_y = (487462, 9016243)

    # Offset in meters (1 pixel=240m): 100 km square (with offset=50km)
    off = 50000
    polygon = (
        (c_x - off, c_y + off),
        (c_x + off, c_y + off),
        (c_x + off, c_y - off),
        (c_x - off, c_y - off),
        (c_x - off, c_y + off))
    print("Polygon: ", polygon)

    # Create cube object
    cube = ITSCube(polygon, projection)

    # Parameters for the search granule API
    API_params = {
        'start'               : '1984-01-01',
        'end'                 : '2021-01-01',
        'percent_valid_pixels': 1
    }

    skipped_projs = {}
    if not args.parallel:
        # Process ITS_LIVE granules sequentially, look at provided number of granules only
        print("Processing granules sequentially...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_no_api(args.outputDir, args.localPath)

        else:
            cube.create(API_params, args.outputDir, args.numberGranules)

    else:
        # Process ITS_LIVE granules in parallel, look at 100 first granules only
        print("Processing granules in parallel...")
        if args.localPath:
            # Granules are downloaded locally
            cube.create_from_local_parallel_no_api(args.outputDir, args.localPath)

        else:
            cube.create_parallel(API_params, args.outputDir, args.numberGranules)

    # Write cube data to the NetCDF file
    # cube.to_netcdf('test_v_cube.nc')
