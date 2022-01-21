"""
ITSLiveComposite class creates yearly composites of ITS_LIVE datacubes with data
within the same target projection, bounding polygon and datetime period
specified at the time the datacube was constructed/updated.

Authors: Masha Liukis, Alex Gardner, Chad Green
"""
import collections
import copy
import dask
from dask.diagnostics import ProgressBar
import datetime
from enum import IntEnum, unique
import gc
import glob
import logging
import numpy  as np
import os
import pandas as pd
import s3fs
import shutil
import timeit
from tqdm import tqdm
import xarray as xr

# Local imports
from itscube import ITSCube
from itscube_types import Coords, DataVars

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

# Define edges of dt bins
DT_EDGE = [0, 32, 64, 128, 256, np.inf]
DT_EDGE_LEN = len(DT_EDGE)-1

# Filter parameters for dt bins (default: 2 - TODO: ask Alex):
# used to determine if dt means are significantly different
DTBIN_MAD_THRESH = 1

# Filter parameters for lsq fit for outlier rejections
MAD_THRESH = 3
MAD_FILTER_ITERATIONS = 3

# Scalar relation between MAD and STD
MAD_STD_RATIO = 1.4826

DTBIN_RATIO = DTBIN_MAD_THRESH * MAD_STD_RATIO

def madFunction(x):
    """
    Compute median absolute deviation (MAD).
    """
    return (np.fabs(x - x.median(dim=Coords.MID_DATE))).median()

def medianMadFunction(x):
    """
    Compute median and median absolute deviation (MAD) for the vector x.
    """
    xmed = 0
    xmad = 0

    if x.size:
        xmed = np.median(x)
        xmad = np.median(np.fabs(x - xmed))

    return [xmed, xmad]

def decimal_year(x):
    """
    Return decimal year representation of the datetime object.

    Reference: https://newbedev.com/how-to-convert-python-datetime-dates-to-decimal-float-years
    """
    return x.year + float(x.toordinal() - datetime.date(x.year, 1, 1).toordinal()) / (datetime.date(x.year+1, 1, 1).toordinal() - datetime.date(x.year, 1, 1).toordinal())

class VariablesInfo:
    """
    Variables information to store in Zarr.
    """
    STD_NAME = {
        DataVars.VX: "x_velocity",
        DataVars.VY: "y_velocity",
    }

    DESCRIPTION = {
        DataVars.VX: "error weighted velocity component in x direction",
        DataVars.VY: "error weighted velocity component in y direction",
    }

class CompositeVariable:
    """
    Class to hold values for v, vx and vy components of the variables.
    """
    def __init__(self, dims: list, name: str):
        """
        Initialize data variables to hold results.
        """
        self.name = name
        self.v = np.full(dims, np.nan)
        self.vx = np.full(dims, np.nan)
        self.vy = np.full(dims, np.nan)


class ITSLiveComposite:
    """
    CLass to build composites for ITS_LIVE datacubes.

    Configurable parameters of the mosaics:
    * Time interval
    * Filters
    * TODO
    """
    # Only the following datacube variables are needed for composites/mosaics
    VARS = [
        DataVars.VX,
        DataVars.VY,
        'vx_error',
        'vy_error',
        DataVars.ImgPairInfo.DATE_DT,
        Coords.MID_DATE,
        DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
        DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
        DataVars.FLAG_STABLE_SHIFT,
        DataVars.ImgPairInfo.SATELLITE_IMG1,
        DataVars.ImgPairInfo.MISSION_IMG1
    ]
    # % maltab can't read in 'mission_img1' but this needs to be implemented in python verson

    # Define edges of dt bins
    DT_EDGE = [0, 32, 64, 128, 256, np.inf]
    DT_EDGE_LEN = len(DT_EDGE)-1

    # Filter parameters for dt bins (default: 2 - TODO: ask Alex):
    # used to determine if dt means are significantly different
    DTBIN_MAD_THRESH = 1

    # Filter parameters for lsq fit for outlier rejections
    MAD_THRESH = 3
    MAD_FILTER_ITERATIONS = 3

    # Scalar relation between MAD and STD
    MAD_STD_RATIO = 1.4826

    DTBIN_RATIO = DTBIN_MAD_THRESH * MAD_STD_RATIO

    # Systematic error based on level of co-registration
    CO_REGISTRATION_ERROR = {
        0: 100,
        1: 5,
        2: 20
    }

    # Threshold for invalid velocity component value: value must be greater than threshold
    V_COMPONENT_THRESHOLD = 25000

    # Store generic cube metadata as static data as these are the same for the whole cube
    YEARS = None
    DATE_DT = None

    START_DECIMAL_YEAR = None
    STOP_DECIMAL_YEAR  = None

    # Dimensions that correspond to the currently processed datacube chunk
    START_X = None
    STOP_X  = None
    START_Y = None
    STOP_Y  =  None

    X_LEN = None
    Y_LEN = None
    MID_DATE_LEN = None
    YEARS_LEN = None

    # Minimum number of non-NAN values in the data to proceceed with processing
    NUM_VALID_POINTS = 5

    TWO_PI = np.pi * 2

    # ------------------
    # Dask related data
    # ------------------
    # Number of X coordinates to process in one "chunk" with Dask parallel scheduler
    NUM_TO_PROCESS = 200

    # Keep it constant static (so there is no command-line option for it to overwrite):
    DASK_SCHEDULER = 'threads'

    # Number of Dask processes to run in parallel
    NUM_DASK_THREADS = 4

    def __init__(self, cube_store: str, s3_bucket: str):
        """
        Initialize composites.
        """
        # Don't need to know skipped granules information for the purpose of composites
        read_skipped_granules_flag = False
        self.s3, self.cube_store_in, cube_ds, _ = ITSCube.init_input_store(
            cube_store,
            s3_bucket,
            read_skipped_granules_flag
        )
        # If reading NetCDF data cube
        # cube_ds = xr.open_dataset(cube_store, decode_timedelta=False)

        # Read in only specific data variables
        logging.info(f"Read only variables of interest from datacube...")
        # Need to sort data by dt to be able to filter with np.searchsorted()
        # (relies on date_dt vector being sorted)
        # self.data = cube_ds[ITSLiveComposite.VARS].sortby(DataVars.ImgPairInfo.DATE_DT)
        # cube_ds = cube_ds[ITSLiveComposite.VARS].sortby(DataVars.ImgPairInfo.DATE_DT)
        cube_ds = cube_ds.sortby(DataVars.ImgPairInfo.DATE_DT)

        self.data = cube_ds[[DataVars.VX, DataVars.VY]]
        logging.info(f"Done reading only variables of interest from datacube...")

        # Add systematic error based on level of co-registration
        # Load Dask arrays before being able to modify their values
        logging.info(f"Add systematic error based on level of co-registration...")
        self.vx_error = cube_ds.vx_error.values
        self.vy_error = cube_ds.vy_error.values

        for value, error in ITSLiveComposite.CO_REGISTRATION_ERROR.items():
            mask = (cube_ds[DataVars.FLAG_STABLE_SHIFT] == value)
            self.vx_error[mask] += error
            self.vy_error[mask] += error

        # TODO: introduce a method to determine mosaics granularity

        # Images acquisition times and middle_date of each layer as datetime.datetime objects
        acq_datetime_img1 = [t.astype('M8[ms]').astype('O') for t in cube_ds[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1].values]
        acq_datetime_img2 = [t.astype('M8[ms]').astype('O') for t in cube_ds[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2].values]

        # Compute decimal year representation for start and end dates of each velocity pair
        ITSLiveComposite.START_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img1])
        ITSLiveComposite.STOP_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img2])

        # Define time boundaries of composites
        # y1 = floor(min(yr(:,1))):floor(max(yr(:,2)));
        start_year = int(np.floor(np.min(ITSLiveComposite.START_DECIMAL_YEAR)))
        stop_year = int(np.floor(np.max(ITSLiveComposite.STOP_DECIMAL_YEAR)))

        # Years to generate mosaics for
        ITSLiveComposite.YEARS = list(range(start_year, stop_year+1))
        logging.info(f'Years for composite: {ITSLiveComposite.YEARS}')

        # Day separation between images (sorted per cube.sortby() call above)
        ITSLiveComposite.DATE_DT = cube_ds[DataVars.ImgPairInfo.DATE_DT].load()

        # Remember datacube dimensions
        self.cube_sizes = cube_ds.sizes

        # These data members will be set for each block of data being currently
        # processed ---> have to change the logic if want to parallelize blocks
        ITSLiveComposite.X_LEN = self.cube_sizes[Coords.X]
        ITSLiveComposite.Y_LEN = self.cube_sizes[Coords.Y]
        ITSLiveComposite.MID_DATE_LEN = self.cube_sizes[Coords.MID_DATE]
        ITSLiveComposite.YEARS_LEN = len(ITSLiveComposite.YEARS)

        # Allocate memory for composite outputs
        dims = (ITSLiveComposite.YEARS_LEN, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)

        self.error = CompositeVariable(dims, 'error')
        self.count = CompositeVariable(dims, 'count')
        self.amplitude = CompositeVariable(dims, 'amplitude')
        self.phase = CompositeVariable(dims, 'phase')
        self.sigma = CompositeVariable(dims, 'sigma')
        self.mean = CompositeVariable(dims, 'mean')

        dims = (ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        self.outlier_fraction = np.full(dims, np.nan)

        # Sensor data for the cube's layers
        self.sensors = cube_ds[DataVars.ImgPairInfo.SATELLITE_IMG1].load()

        # Identify unique sensors within datacube
        self.unique_sensors = list(set(self.sensors.values))
        dims = (len(self.unique_sensors), ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        self.max_dt = np.full(dims, np.nan)

    def create(self, output_store: str, s3_bucket: str):
        """
        Create datacube composite: cube time mean values.
        """
        # Loop through cube in chunks to minimize memory footprint
        x_start = 0
        x_num_to_process = ITSLiveComposite.X_LEN
        initial_y_num_to_process = ITSLiveComposite.Y_LEN

        while x_num_to_process > 0:
            # How many tasks to process at a time
            x_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if x_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else x_num_to_process

            y_num_to_process = initial_y_num_to_process
            y_start = 0

            while y_num_to_process > 0:
                y_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if y_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else y_num_to_process

                self.cube_time_mean(y_start, y_start+y_num_tasks, x_start, x_start+x_num_tasks)

                y_num_to_process -= y_num_tasks
                y_start += y_num_tasks


            x_num_to_process -= x_num_tasks
            x_start += x_num_tasks

        # Save data to Zarr store
        # self.to_zarr(output_store, s3_bucket)

        logging.info(f"Done.")

    def cube_time_mean(self, start_y, stop_y, start_x, stop_x):
        """
        Compute time average for the datacube [:, :, start_x:stop_index] coordinates.
        Update corresponding entries in output data variables.
        """
        logging.info(f"Processing datacube[:, {start_y}:{stop_y}, {start_x}:{stop_x}] coordinates out of [{self.cube_sizes[Coords.MID_DATE]}, {self.cube_sizes[Coords.Y]}, {self.cube_sizes[Coords.X]}]")

        # Set current block length for the X dimension
        ITSLiveComposite.X_LEN = stop_x - start_x
        ITSLiveComposite.START_X = start_x
        ITSLiveComposite.STOP_X = stop_x

        # Set current block length for the Y dimension
        ITSLiveComposite.Y_LEN = stop_y - start_y
        ITSLiveComposite.START_Y = start_y
        ITSLiveComposite.STOP_Y = stop_y

        # TODO:
        # % form unique sensor_satellite ID to pass to "cubetimemean"... not
        # % done in matlab verison because can't read in 'mission_img1'

        # % convert to singles (Float32) to reduce memory footprint
        # for k = 1:length(vars)
        #     if any(strcmp({'vx','vy','vx_error','vy_error'}, vars{k}))
        #         data.(vars{k}) = single(data.(vars{k}));
        #     end
        # end
        # fprintf('finished [%.1fmin]\n', (now-t0)*24*60)

        # % convert to matlab date
        # data.mid_date = data.mid_date + datenum(1970,1,1);

        #     % compute time averages
        #     [v_amp(x0(i):x0(i+1)-1,:,:), vx_amp(x0(i):x0(i+1)-1,:,:), vy_amp(x0(i):x0(i+1)-1,:,:), ...
        #     v_phase(x0(i):x0(i+1)-1,:,:), vx_phase(x0(i):x0(i+1)-1,:,:), vy_phase(x0(i):x0(i+1)-1,:,:), ...
        #     v(x0(i):x0(i+1)-1,:,:), vx(x0(i):x0(i+1)-1,:,:), vy(x0(i):x0(i+1)-1,:,:), v_err(x0(i):x0(i+1)-1,:,:), ...
        #     vx_err(x0(i):x0(i+1)-1,:,:), vy_err(x0(i):x0(i+1)-1,:,:), v_sigma(x0(i):x0(i+1)-1,:,:), ...
        #     vx_sigma(x0(i):x0(i+1)-1,:,:), vy_sigma(x0(i):x0(i+1)-1,:,:), v_cnt(x0(i):x0(i+1)-1,:,:), ...
        #     vx_cnt(x0(i):x0(i+1)-1,:,:), vy_cnt(x0(i):x0(i+1)-1,:,:), maxdt(x0(i):x0(i+1)-1,:), ...
        #     outlier_frac(x0(i):x0(i+1)-1,:)] = ...
        #     cubetimemean(data.vx,data.vy,data.vx_error,data.vy_error,data.mid_date,data.date_dt,data.satellite_img1,yrs);
        # end

        # Start timer
        start_time = timeit.default_timer()

        # ----- FILTER DATA -----
        # Filter data based on locations where means of various dts are
        # statistically different and mad deviations from a running meadian
        logging.info(f'Filtering data based on dt binned medians...')

        # Initialize variables
        dims = (ITSLiveComposite.MID_DATE_LEN, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        vx_invalid = np.full(dims, False)
        vy_invalid = np.full(dims, False)

        # Loop for each unique sensor (those groupings image pairs that can be
        # expected to have different temporal decorelation)
        dims = (len(self.unique_sensors), ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        vx_maxdt = np.full(dims, np.nan)
        vy_maxdt = np.full(dims, np.nan)

        # ATTN: Using native xarray functionality is much slower,
        # convert data to numpy types and use numpy only
        logging.info(f'Loading vx[:, {start_y}:{stop_y}, {start_x}:{stop_x}]...')
        vx = self.data.vx[:, start_y:stop_y, start_x:stop_x].astype(float).values
        logging.info(f'Loaded vx[:, {start_y}:{stop_y}, {start_x}:{stop_x}]...')

        logging.info(f'Loading vy[:, {start_y}:{stop_y}, {start_x}:{stop_x}]...')
        vy = self.data.vy[:, start_y:stop_y, start_x:stop_x].astype(float).values
        logging.info(f'Loaded vy[:, {start_y}:{stop_y}, {start_x}:{stop_x}]...')

        for i in range(len(self.unique_sensors)):
            logging.info(f'Filtering dt for sensor "{self.unique_sensors[i]}" ({i+1} out of {len(self.unique_sensors)})')
            # Find which layers correspond to each sensor
            mask = (self.sensors == self.unique_sensors[i]).values

            # Filter current block's variables
            # TODO: Don't drop variables when masking - won't work on return assignment
            #       for cubes with multiple sensors
            dt_masked = ITSLiveComposite.DATE_DT.values[mask]

            logging.info(f'Filtering vx...')
            # vx_invalid[mask, :, :], vx_maxdt[i, :, :] = ITSLiveComposite.cube_filter(vx.where(mask), dt_masked)
            vx_invalid[mask, :, :], vx_maxdt[i, :, :] = ITSLiveComposite.cube_filter(vx[mask, ...], dt_masked)

            logging.info(f'Filtering vy...')
            # vy_invalid[mask, :, :], vy_maxdt[i, :, :] = ITSLiveComposite.cube_filter(vy.where(mask), dt_masked)
            vy_invalid[mask, :, :], vy_maxdt[i, :, :] = ITSLiveComposite.cube_filter(vy[mask, ...], dt_masked)

            # Get maximum value along sensor dimension: concatenate maxdt
            # for vx and vy in new dimension
            self.max_dt[i, start_y:stop_y, start_x:stop_x] = np.nanmax(np.stack((vx_maxdt[i, :, :], vy_maxdt[i, :, :]), axis=0), axis=0)

        # Load data to avoid NotImplemented exception when invoked on Dask arrays
        logging.info(f'Compute invalid mask...')

        invalid = \
            vx_invalid | vy_invalid \
            | (np.fabs(vx) > ITSLiveComposite.V_COMPONENT_THRESHOLD) \
            | (np.fabs(vy) > ITSLiveComposite.V_COMPONENT_THRESHOLD)

        # Mask data
        logging.info(f'Mask invalid entries for vx...')
        vx[invalid] = np.nan

        logging.info(f'Mask invalid entries for vy...')
        vy[invalid] = np.nan

        invalid = np.nansum(invalid, axis=0)
        invalid = np.divide(invalid, np.sum(np.isnan(vx), 0) + invalid)

        logging.info(f'Finished filtering ({timeit.default_timer() - start_time} seconds)')

        # %% Least-squares fits to detemine amplitude, phase and annual means
        logging.info(f'Find vx annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        _ = ITSLiveComposite.cubelsqfit2(
            vx,
            self.vx_error,
            self.amplitude.vx,
            self.phase.vx,
            self.mean.vx,
            self.error.vx,
            self.sigma.vx,
            self.count.vx
        )
        logging.info(f'Finished vx LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find vy annual means using LSQ fit... ')
        start_time = timeit.default_timer()
        _ = ITSLiveComposite.cubelsqfit2(
            vy,
            self.vy_error,
            self.amplitude.vy,
            self.phase.vy,
            self.mean.vy,
            self.error.vy,
            self.sigma.vy,
            self.count.vy
        )
        logging.info(f'Finished vy LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find v annual means using LSQ fit... ')
        start_time = timeit.default_timer()
        # Need to project velocity onto a unit flow vector to avoid biased (Change from Rician to Normal distribution)
        vxm = np.nanmedian(self.mean.vx[:, start_y:stop_y, start_x:stop_x], axis=0)
        vym = np.nanmedian(self.mean.vy[:, start_y:stop_y, start_x:stop_x], axis=0)
        theta = np.arctan2(vxm, vym)

        # Explicitly expand the value
        stheta = np.tile(np.sin(theta).astype(float), (ITSLiveComposite.MID_DATE_LEN, 1, 1))
        ctheta = np.tile(np.cos(theta).astype(float), (ITSLiveComposite.MID_DATE_LEN, 1, 1))

        theta = None
        vx = vx*stheta + vy*ctheta

        # Don't need variables anymore
        stheta = None
        ctheta = None

        # Now only np.abs(vxm) and np.abs(vym) are used, reset the variables
        vxm = np.fabs(vxm)
        vym = np.fabs(vym)

        # Expand dimensions of vectors
        vy = self.vx_error.reshape((len(self.vx_error), 1, 1))
        vy_expand = self.vy_error.reshape((len(self.vy_error), 1, 1))

        vy = (
            np.tile(vy, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN))*vxm + \
            np.tile(vy_expand, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN))*vym
        ) / np.sqrt(np.power(vxm, 2) + np.power(vym, 2))

        voutlier = ITSLiveComposite.cubelsqfit2(
            vx,
            vy,
            self.amplitude.v,
            self.phase.v,
            self.mean.v,
            self.error.v,
            self.sigma.v,
            self.count.v
        )
        logging.info(f'Finished v LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        # Because velocities have been projected onto a mean flow direction they can be negative
        self.mean.v = np.fabs(self.mean.v)

        # outlier = invalid + voutlier*(1-invalid)
        self.outlier_fraction[start_y:stop_y, start_x:stop_x] = invalid + voutlier*(1-invalid)

    def to_zarr(self, output_store: str, s3_bucket: str):
        """
        Store datacube annual composite to the Zarr store.
        """
        ds = xr.Dataset(
            data_vars = {DataVars.URL: ([Coords.MID_DATE], self.urls)},
            coords = {
                Coords.TIME: (
                    Coords.TIME,
                    ITSLiveComposite.YEARS,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.TIME],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.TIME]
                    }
                ),
                Coords.X: (
                    Coords.X,
                    self.grid_x,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X]
                    }
                ),
                Coords.Y: (
                    Coords.Y,
                    self.grid_y,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y]
                    }
                )
            },
            attrs = {
                'author': 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)'
            }
        )

        ds.attrs['institution'] = 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology'
        ds.attrs['title'] = 'autoRIFT surface velocities'
        # TODO: Add any other attributes

        # Add data as variables

    @staticmethod
    def cube_filter(data, dt):
    # def cube_filter(data, bin_dt, dt):
        """
        Filter data cube by dt (date separation) between the images.
        """
        # %dtbin_mad_thresh: used to determine in dt means are significantly different
        # %dtbin_mad_thresh = 2*1.4826;

        # check if populations overlap (use first, smallest dt, bin as reference)

        # Initialize output
        invalid = np.full_like(data, False)

        dims = (ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        maxdt = np.full(dims, np.nan)

        # Loop through all spacial points
        # ATTN: debugging only - process 2 x cells only
        for j in tqdm(range(0, ITSLiveComposite.Y_LEN), ascii=True, desc='cube_dt_filter: y'):
        # for j in tqdm(range(0, 1), ascii=True, desc='cube_filter (debug_len=1): y'):
            # Process X coordinates in parallel
            # Pass to the dask only the data it needs: data.isel() by X and Y

            # Sequential processing for debugging
            # iter_i, iter_j, iter_maxdt, iter_invalid = ITSLiveComposite.cube_filter_iteration(data[:, j, i], dt, 0, j)
            # maxdt[iter_j, iter_i] = iter_maxdt
            # invalid[:, iter_j, iter_i] = iter_invalid

            tasks = [dask.delayed(ITSLiveComposite.cube_filter_iteration)(data[:, j, i], dt, i, j) for i in range(0, ITSLiveComposite.X_LEN)]

            results = dask.compute(
                tasks,
                threads_per_worker=1,
                scheduler=ITSLiveComposite.DASK_SCHEDULER,
                num_workers=ITSLiveComposite.NUM_DASK_THREADS
            )

            # Process results
            for each_output in results[0]:
                iter_i, iter_j, iter_maxdt, iter_invalid = each_output
                maxdt[iter_j, iter_i] = iter_maxdt
                invalid[:, iter_j, iter_i] = iter_invalid

        return invalid, maxdt

    @staticmethod
    def cube_filter_iteration(x0_in, dt, i, j):
        """
        Filter one spacial point by dt (date separation) between the images.
        This method is introduced to be able to parallelize cube_filter() method
        using Dask scheduler.
        """
        # Output data variables
        maxdt = np.nan
        invalid = np.full_like(dt, False)

        # x0_is_null = x0_in.isnull()
        x0_is_null = np.isnan(x0_in)
        if np.all(x0_is_null):
            # No data to process
            return (i, j, maxdt, invalid)

        # Filter NAN values out
        # logging.info(f'Before mask filter: type(x0)={type(x0_in)}')
        mask = ~x0_is_null
        # x0 = x0_in.where(mask, drop=True).values
        # x0_dt = dt.where(mask, drop=True).values
        x0 = x0_in[mask]
        x0_dt = dt[mask]

        # x0_dt = dt_.where(mask, drop=True)
        # np_digitize = np.digitize(x0_dt.values, ITSLiveComposite.DT_EDGE, right=False)
        # index_var = xr.IndexVariable(Coords.MID_DATE, np_digitize)
        # groups = x0.groupby(index_var)

        # Group data values by identified bins "manually":
        # since data is sorted by date_dt, we can identify index boundaries
        # for each bin within the "date_dt" vector
        # logging.info(f'Before searchsorted')
        bin_index = np.searchsorted(x0_dt, ITSLiveComposite.DT_EDGE).tolist()

        # logging.info(f'Before init xmed xmad')
        num_bins = ITSLiveComposite.DT_EDGE_LEN
        xmed = np.zeros(ITSLiveComposite.DT_EDGE_LEN)
        xmad = np.zeros(ITSLiveComposite.DT_EDGE_LEN)

        # for i in range(0, ITSLiveComposite.DT_EDGE_LEN):
        for bin_i in range(0, num_bins):
            xmed[bin_i], xmad[bin_i] = np.apply_along_axis(medianMadFunction, 0, x0[bin_index[bin_i]:bin_index[bin_i+1]])

        # Check if populations overlap (use first, smallest dt, bin as reference)
        # logging.info(f'Before min/max bound')
        std_dev = xmad * ITSLiveComposite.DTBIN_RATIO
        minBound = xmed - std_dev
        maxBound = xmed + std_dev

        exclude = (minBound > maxBound[0]) | (maxBound < minBound[0])

        if np.any(exclude):
            # logging.info(f"Got exclude: i={i} j={j}")
            maxdt = np.take(ITSLiveComposite.DT_EDGE, exclude).min()
            invalid = dt > maxdt

        return (i, j, maxdt, invalid)

    @staticmethod
    def cubelsqfit2(
        v,
        v_err_data,
        amplitude,
        phase,
        mean,
        error,
        sigma,
        count
    ):
        """
        Cube LSQ fit (why 2 in name - 2 iterations?)

        Populate: [amp, phase, mean, err, sigma, cnt]

        Return: outlier_frac
        """
        #
        # Get start and end dates
        # tt = [t - dt/2, t+dt/2];
        #
        # Initialize output
        dims = (ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        outlier_frac = np.full(dims, np.nan)

        # This is only done for generic parfor "slicing" may not be needed when
        # recoded
        v_err = v_err_data
        if v_err_data.ndim != v.ndim:
            # Expand vector to 3-d array
            reshape_v_err = v_err_data.reshape((v_err_data.size, 1, 1))
            v_err = np.tile(reshape_v_err, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN))

        for j in tqdm(range(0, ITSLiveComposite.Y_LEN), ascii=True, desc='cubelsqfit2: y'):
        # for j in tqdm(range(0, 1), ascii=True, desc='cubelsqfit2: y (debug)'):
            tasks = [dask.delayed(ITSLiveComposite.cubelsqfit2_iteration)(v[:, j, i], v_err[:, j, i], i, j) for i in range(0, ITSLiveComposite.X_LEN)]

            results = dask.compute(
                tasks,
                threads_per_worker=1,
                scheduler=ITSLiveComposite.DASK_SCHEDULER,
                num_workers=ITSLiveComposite.NUM_DASK_THREADS
            )

            # Process results
            for each_output in results[0]:
                iter_i, iter_j, skip_flag, results = each_output

                if skip_flag is False:
                    # Annual phase and amplitude for processed years
                    global_i = iter_i + ITSLiveComposite.START_X
                    global_j = iter_j + ITSLiveComposite.START_Y

                    ind, \
                    amplitude[ind, global_j, global_i], \
                    phase[ind, global_j, global_i], \
                    sigma[ind, global_j, global_i], \
                    mean[ind, global_j, global_i], \
                    error[ind, global_j, global_i], \
                    count[ind, global_j, global_i], \
                    outlier_frac[iter_j, iter_i] = results

        return outlier_frac

    @staticmethod
    def cubelsqfit2_iteration(x1, x_err1, i, j):
        """
        cubelsqfit2 processing of one spacial point - to be able to parallelize
        the processing with dask.
        """
        # Flag if computations were skipped
        skip_flag = True

        mask = ~np.isnan(x1)
        if np.sum(mask) < ITSLiveComposite.NUM_VALID_POINTS:
            # Skip the point, return no computed results
            return (i, j, skip_flag, ())

        skip_flag = False
        return (i, j, skip_flag, ITSLiveComposite.itslive_lsqfit_annual(x1, x_err1))

    @staticmethod
    def itslive_lsqfit_annual(v, v_err):
        # Populates [A,ph,A_err,t_int,v_int,v_int_err,N_int,outlier_frac] data
        # variables.
        # Computes the amplitude and phase of seasonal velocity
        # variability, and also gives interanual variability.
        #
        # From original Matlab code:
        # %% Syntax
        # %
        # %  [A,ph] = itslive_sinefit_lsq(t,v,v_err)
        # %  [A,ph,A_err,t_int,v_int,v_int_err,N_int] = itslive_sinefit_lsq(t,v,v_err)
        # %
        # %% Description
        # %
        # % [A,ph] = itslive_sinefit_lsq(t,v,v_err) gives the amplitude and phase of a seasonal
        # % cycle for a single-pixel itslive time series. Times t are Nx2, velocities v
        # % are Nx1 and velocity error v_err are Nx1.
        # %
        # % [A,ph,A_err,t_int,v_int,v_int_err,N_int] = itslive_sinefit_lsq(t,v,v_err)
        # % also returns the standard deviation of amplitude residuals A_err. Outputs
        # % t_int and v_int describe interannual velocity variability, and can then
        # % be used to reconstruct a continuous time series, as shown below. Output
        # % Output N_int is the number of image pairs that contribute to the annual mean
        # % v_int of each year. The output |v_int_err| is a formal estimate of error
        # % in the v_int.
        # %
        # %% Example
        # %
        # % load byrd_test_data
        # %
        # % [A,ph,A_err,t_int,v_int] =  itslive_sinefit_lsq(t,v,v_err);
        # %
        # % ti = min(t(:,1)):max(t(:,2));
        # % vi = interp1(t_int,v_int,ti,'pchip') + sineval([A ph],ti);
        # %
        # % itslive_tsplot(t,v,v_err,'datenum','inliers','thresh',50)
        # % plot(ti,vi,'linewidth',1)
        # %
        # %% Author Info
        # % Chad A. Greene, Jan 2020.
        # %
        # % See also itslive_interannual.
        #

        # Ensure we're starting with finite data
        isf_mask   = np.isfinite(v) & np.isfinite(v_err)
        start_year = ITSLiveComposite.START_DECIMAL_YEAR[isf_mask]
        stop_year  = ITSLiveComposite.STOP_DECIMAL_YEAR[isf_mask]

        v     = v[isf_mask]
        v_err = v_err[isf_mask]

        #
        # %% Detrend
        #
        # % % tm = mean(t,2);
        # % % polyOrder = ceil((max(tm)-min(tm))/(4*365.25)); % order of polynomial for detrending
        # % % [pv,S,mu] = polyfitw(tm,v,polyOrder,1./v_err.^2);
        # % % v = v - polyval(pv,tm,S,mu);
        #

        # dt in years
        dyr = stop_year - start_year

        # yr = decyear(t)

        # Weights for velocities
        w_v = 1/v_err**2

        # Weights (correspond to displacement error, not velocity error):
        w_d = 1/(v_err*dyr)  # Not squared because the p= line below would then have to include sqrt(w) on both accounts

        # Observed displacement in meters
        d_obs = v*dyr

        #
        # %% Initial outlier detection and removal:
        # % % % I've tried this entire function with or without this section, and it
        # % % % doesn't seem to make much difference. It's actually fine to do
        # % % % the whole analysis without detrending at all, but I'm leaving this here
        # % % % for now anyways.
        # % %
        # % % % Define outliers as anything that's more than 10 standard deviations away from detrended vals.
        # % % outliers = abs(v) > 10*std(v);
        # % %
        # % % % Remove them!
        # % % yr = yr(~outliers,:);
        # % % dyr = dyr(~outliers);
        # % % d_obs = d_obs(~outliers);
        # % % w_d = w_d(~outliers);
        # % % w_v = w_v(~outliers);
        #

        # Make matrix of percentages of years corresponding to each displacement measurement
        y_min = int(np.floor(start_year.min()))
        y_max = int(np.floor(stop_year.max())) + 1
        y1 = np.arange(y_min, y_max)

        M = np.zeros((len(dyr), len(y1)))

        # Loop through each year:
        for k in range(len(y1)):
            # Set all measurements that begin before the first day of the year and end after the last
            # day of the year to 1:
            ind = np.logical_and(start_year <= y1[k], stop_year >= (y1[k] + 1))
            M[ind, k] = 1

            # Within year:
            ind = np.logical_and(start_year >= y1[k], stop_year < (y1[k] + 1))
            M[ind, k] = dyr[ind]

            # Started before the beginning of the year and ends during the year:
            ind = np.logical_and(start_year < y1[k], np.logical_and(stop_year >= y1[k], stop_year < (y1[k] + 1)))
            M[ind, k] = stop_year[ind] - y1[k]

            # Started during the year and ends the next year:
            ind = np.logical_and(start_year >= y1[k], np.logical_and(start_year < (y1[k] + 1), stop_year >= (y1[k]+1)))
            M[ind, k] = (y1[k] + 1) - start_year[ind]

        # Filter sum of each column
        hasdata = M.sum(axis=0) > 0
        y1 = y1[hasdata]
        M = M[:, hasdata]

        #
        # First LSQ iteration
        # Iterative mad filter
        totalnum = len(start_year)
        for i in range(0, ITSLiveComposite.MAD_FILTER_ITERATIONS):
            # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
            D = np.array([
                (np.cos(ITSLiveComposite.TWO_PI*start_year) - np.cos(ITSLiveComposite.TWO_PI*stop_year))/ITSLiveComposite.TWO_PI,\
                (np.sin(ITSLiveComposite.TWO_PI*stop_year) - np.sin(ITSLiveComposite.TWO_PI*start_year))/ITSLiveComposite.TWO_PI]).T

            # Add M: a different constant for each year (annual mean)
            D_M = np.concatenate([D, M], axis=1)

            # Add ones: constant offset for all data (effectively the mean velocity)
            D = np.column_stack([D_M, np.ones(len(dyr))])

            # Make numpy happy: have all data 2D
            np_w_d = np.copy(w_d).reshape((len(w_d), 1))

            # These both are of xarray type, so can multiply them
            # w_d*d_obs

            # TODO: Change later to have consistent iterations (use second iteration approach)
            # Solve for coefficients of each column in the Vandermonde:
            p = np.linalg.lstsq(np_w_d * D, w_d*d_obs, rcond=None)[0]

            # Find and remove outliers
            # d_model = sum(bsxfun(@times,D,p'),2); % modeled displacements (m)
            d_model = (D * p).sum(axis=1)  # modeled displacements (m)

            # Divide by dt to avoid penalizing long dt [asg]
            d_resid = np.abs(d_obs - d_model)/dyr

            # Robust standard deviation of errors, using median absolute deviation
            d_sigma = np.median(d_resid, axis=0)*ITSLiveComposite.MAD_STD_RATIO

            outliers = d_resid > (ITSLiveComposite.MAD_THRESH * d_sigma)

            # Remove outliers
            non_outlier_mask = ~outliers
            start_year = start_year[non_outlier_mask]
            stop_year = stop_year[non_outlier_mask]
            dyr = dyr[non_outlier_mask]
            d_obs = d_obs[non_outlier_mask]
            w_d = w_d[non_outlier_mask]
            w_v = w_v[non_outlier_mask]
            M = M[non_outlier_mask]

            # Remove no-data columns from M
            hasdata = M.sum(axis=0) > 1
            y1 = y1[hasdata]
            M = M[:, hasdata]

            outliers_fraction = outliers.sum() / totalnum
            if outliers_fraction < 0.01 and (i+1) != ITSLiveComposite.MAD_FILTER_ITERATIONS:
                # There are less than 1% outliers, skip the rest of iterations
                # if it's not the last iteration
                # logging.info(f'{outliers_fraction*100}% ({outliers.sum()} out of {totalnum}) outliers, done with first LSQ loop after {i+1} iterations')
                break

        # Matlab: outlier_frac = length(yr)./totalnum;
        outlier_frac = (totalnum - len(start_year))/totalnum

        #
        # Second LSQ iteration
        #
        # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
        # D = [(cos(2*pi*yr(:,1)) - cos(2*pi*yr(:,2)))./(2*pi).*(M>0) (sin(2*pi*yr(:,2)) - sin(2*pi*yr(:,1)))./(2*pi).*(M>0) M];
        M_pos = M > 0
        D = np.block([
            ((np.cos(ITSLiveComposite.TWO_PI*start_year) - np.cos(ITSLiveComposite.TWO_PI*stop_year))/ITSLiveComposite.TWO_PI).reshape((len(M_pos), 1)) * M_pos,\
            ((np.sin(ITSLiveComposite.TWO_PI*stop_year) - np.sin(ITSLiveComposite.TWO_PI*start_year))/ITSLiveComposite.TWO_PI).reshape((len(M_pos), 1)) * M_pos])

        # D = np.block([
        #     ((np.cos(ITSLiveComposite.TWO_PI*start_year) - np.cos(ITSLiveComposite.TWO_PI*stop_year))/ITSLiveComposite.TWO_PI) * M_pos,\
        #     ((np.sin(ITSLiveComposite.TWO_PI*stop_year) - np.sin(ITSLiveComposite.TWO_PI*start_year))/ITSLiveComposite.TWO_PI]) * M_pos])

        # Add M: a different constant for each year (annual mean)
        D = np.concatenate([D, M], axis=1)

        # Make numpy happy: have all data 2D
        np_w_d = np.copy(w_d).reshape((len(w_d), 1))

        # Solve for coefficients of each column in the Vandermonde:
        p = np.linalg.lstsq(np_w_d * D, w_d*d_obs, rcond=None)[0]

        # Postprocess
        #
        # Convert coefficients to amplitude and phase of a single sinusoid:
        Nyrs = len(y1)
        # Amplitude of sinusoid from trig identity a*sin(t) + b*cos(t) = d*sin(t+phi), where d=hypot(a,b) and phi=atan2(b,a).
        A = np.hypot(p[0:Nyrs], p[Nyrs:2*Nyrs])

        # phase in radians
        ph_rad = np.arctan2(p[Nyrs:2*Nyrs], p[0:Nyrs])

        # phase converted such that it reflects the day when value is maximized
        ph = 365.25*((0.25 - ph_rad/ITSLiveComposite.TWO_PI) % 1)

        # Goodness of fit:
        d_model = (D * p).sum(axis=1)  # modeled displacements (m)

        # A_err is the *velocity* (not displacement) error, which is the displacement error divided by the weighted mean dt:
        A_err = np.full_like(A, np.nan)

        for k in range(Nyrs):
            ind = M[:, k] > 0
            # asg replaced call to wmean
            A_err[k] = np.sqrt(np.cov(d_obs[ind]-d_model[ind], aweights=w_d[ind])) / (w_d[ind]*dyr[ind]).sum(axis=0) / dyr[ind].sum()

        v_int = p[2*Nyrs:]

        # % Number of equivalent image pairs per year: (1 image pair equivalent means a full year of data. It takes about 23 16-day image pairs to make 1 year equivalent image pair.)
        N_int = np.sum(M>0)
        v_int = p[2*Nyrs:]

        # Reshape array to have the same number of dimensions as M for multiplication
        w_v = w_v.reshape((1, w_v.shape[0]))

        # logging.info(f'w_v.shape={w_v.shape}')
        # logging.info(f'M.type={M} M.shape={M.shape}')
        v_int_err = 1/np.sqrt((w_v@M).sum(axis=0))
        # logging.info(f"v_int_err: {v_int_err}")
        # logging.info(f"v_int_err.shape: {v_int_err.shape}")

        # Identify year's indices to assign return values to in "final" composite
        # variables
        _, _, ind = np.intersect1d(ITSLiveComposite.YEARS, y1, return_indices=True)

        # On return: amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1, outlier_fraction
        return [ind, A, ph, A_err, v_int, v_int_err, N_int, outlier_frac]

if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSLiveComposite.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default=4,
        help='Number of Dask workers to use for parallel processing [%(default)d].'
    )
    parser.add_argument(
        '-c', '--daskChunkSize',
        type=int,
        default=100,
        help='Number of X coordinates to process in parallel with Dask [%(default)d]. ' \
             'This should be the size of chunking used within the cube to optimize IO of the datacube data.'
    )
    parser.add_argument(
        '-i', '--inputStore',
        type=str,
        default=None,
        help="Input Zarr datacube store to generate mosaics for [%(default)s]."
    )
    parser.add_argument(
        '-o', '--outputStore',
        type=str,
        default="cube_composite.zarr",
        help="Zarr output directory to write composite data to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        default='',
        help="S3 bucket to copy Zarr format of the input datacube from, and to store cube composite to [%(default)s]."
    )
    args = parser.parse_args()

    # Set static data for computation
    ITSLiveComposite.NUM_TO_PROCESS = args.daskChunkSize
    ITSLiveComposite.NUM_DASK_THREADS = args.threads

    # s3://its-live-data/test_datacubes/AGU2021/S70W100/ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.zarr
    # AGU21_ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.nc

    mosaics = ITSLiveComposite(args.inputStore, args.bucket)
    mosaics.create(args.outputStore, args.bucket)
