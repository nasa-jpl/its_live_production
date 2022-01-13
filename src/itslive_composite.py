"""
ITSLiveCubeMosaic class creates yearly composites of ITS_LIVE datacubes based on target projection,
bounding polygon and datetime period provided by the caller.

Authors: Masha Liukis, Alex Gardner, Chad Green
"""
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
from itscube_types import Coords, DataVars


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

def madFunction(x):
    """
    Compute median absolute deviation (MAD).
    """
    return (np.fabs(x - x.median(dim=Coords.MID_DATE))).median()

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

    # Filter parameters for dt bins (default: 2 - TODO: ask Alex):
    # used to determine if dt means are significantly different
    DTBIN_MAD_THRESH = 1

    # Filter parameters for lsq fit for outlier rejections
    MAD_THRESH = 3
    MAD_FILTER_ITERATIONS = 3

    # Scalar relation between MAD and STD
    MAD_STD_RATIO = 1.4826

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
    STOP_X = None
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
    # don't want to use 'threads' to avoid Python’s Global Interpreter Lock (GIL)
    # since we are processing Python's objects
    DASK_SCHEDULER = 'processes'

    # Number of Dask processes to run in parallel
    NUM_THREADS = 4

    def __init__(self, cube_store: str, s3_bucket: str):
        """
        Initialize mosaics object.
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
        self.data = cube_ds[ITSLiveComposite.VARS]
        # self.data.load()

        # Add systematic error based on level of co-registration
        # Load Dask arrays before being able to modify their values
        self.data.vx_error.load()
        self.data.vy_error.load()

        for value, error in ITSLiveComposite.CO_REGISTRATION_ERROR.items():
            mask = (self.data.flag_stable_shift == value)
            self.data.vx_error[mask] += error
            self.data.vy_error[mask] += error

        # Sort cube data by datetime
        # self.data = self.data.sortby(Coords.MID_DATE)

        # TODO: introduce a method to determine mosaics granularity

        # Images acquisition times and middle_date of each layer as datetime.datetime objects
        acq_datetime_img1 = [t.astype('M8[ms]').astype('O') for t in self.data[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1].values]
        acq_datetime_img2 = [t.astype('M8[ms]').astype('O') for t in self.data[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2].values]

        # Compute decimal year representation for start and end dates of each velocity pair
        ITSLiveComposite.START_DECIMAL_YEAR = [decimal_year(each) for each in acq_datetime_img1]
        ITSLiveComposite.STOP_DECIMAL_YEAR = [decimal_year(each) for each in acq_datetime_img2]

        # Define time boundaries of composites
        # y1 = floor(min(yr(:,1))):floor(max(yr(:,2)));
        start_year = int(np.floor(min(ITSLiveComposite.START_DECIMAL_YEAR)))
        stop_year = int(np.floor(max(ITSLiveComposite.STOP_DECIMAL_YEAR)))

        # Years to generate mosaics for
        ITSLiveComposite.YEARS = list(range(start_year, stop_year+1))
        logging.info(f'Years for composite: {ITSLiveComposite.YEARS}')

        # Day separation between images
        ITSLiveComposite.DATE_DT = self.data[DataVars.ImgPairInfo.DATE_DT].load()

        # Initialize variable to hold results
        cube_sizes = self.data.sizes
        ITSLiveComposite.X_LEN = cube_sizes[Coords.X]
        ITSLiveComposite.Y_LEN = cube_sizes[Coords.Y]
        ITSLiveComposite.MID_DATE_LEN = cube_sizes[Coords.MID_DATE]
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

        # Identify unique sensors within datacube
        self.unique_sensors = list(set(self.data[DataVars.ImgPairInfo.SATELLITE_IMG1].values))
        dims = (len(self.unique_sensors), ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        self.max_dt = np.full(dims, np.nan)

    def create(self, output_store: str, s3_bucket: str):
        """
        Create datacube composite: cube time mean values.

        In original Matlab code:
        cubetimemean(vx,vy,vx_err,vy_err,t,dt,sensor,yrs)
            where:
                vx = data.vx;
                vy = data.vy;
                vx_err = data.vx_error;
                vy_err = data.vy_error;
                t = data.mid_date;
                dt = data.date_dt;
                sensor = data.satellite_img1; % needs to be updated
        """
        # Loop through cube in chunks to minimize memory footprint
        start = 0
        num_to_process = ITSLiveComposite.X_LEN

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = ITSLiveComposite.NUM_TO_PROCESS if num_to_process > ITSLiveComposite.NUM_TO_PROCESS else num_to_process
            self.cube_time_mean(start, start+num_tasks)

            num_to_process -= num_tasks
            start += num_tasks

        # TODO: Save data to Zarr store

        # Save data to Zarr store
        # self.to_zarr(output_store, s3_bucket, maxdt, outlier_frac)

        logging.info(f"Done.")

    def cube_time_mean(self, start_x, stop_x):
        """
        Compute time average for the datacube [:, :, start_x:stop_index] coordinates.
        Update corresponding entries in output data variables.
        """
        logging.info(f"Processing datacube[:, :, {start_x}:{stop_x}] coordinates out of [:, :, {self.data.x.size}]")

        # Set current length for the X dimension
        ITSLiveComposite.X_LEN = stop_x - start_x
        ITSLiveComposite.START_X = start_x
        ITSLiveComposite.STOP_X = stop_x

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
        # maxdt    = np.full(dims, np.nan)

        for i in range(len(self.unique_sensors)):
            # Find which layers correspond to each sensor
            mask = (self.data[DataVars.ImgPairInfo.SATELLITE_IMG1] == self.unique_sensors[i])

            logging.info(f'Filtering vx...')
            vx_invalid[mask, :, :], vx_maxdt[i, :, :] = ITSLiveComposite.cube_filter(self.data.vx[:, :, start_x:stop_x].where(mask), ITSLiveComposite.DATE_DT.where(mask))

            logging.info(f'Filtering vy...')
            vy_invalid[mask, :, :], vy_maxdt[i, :, :] = ITSLiveComposite.cube_filter(self.data.vy[:, :, start_x:stop_x].where(mask), ITSLiveComposite.DATE_DT.where(mask))

            # Get maximum value along sensor dimension: concatenate maxdt
            #  for vx and vy in new dimension
            self.max_dt[i, :, start_x:stop_x] = np.nanmax(np.stack((vx_maxdt[i, :, :], vy_maxdt[i, :, :]), axis=0), axis=0)

        # self.max_dt[:, :, start_x:stop_x] = maxdt

        # Load data to avoid NotImplemented exception when invoked on Dask arrays
        logging.info(f'Compute invalid mask...')
        # self.data.vx.load()
        # self.data.vy.load()
        # In case when no boolean values are inserted into the masking array,
        # ensure the array type is still boolean (will be float32 type otherwise)
        # ATTN: bool(np.nan) == True
        # vx_invalid = vx_invalid.astype(bool)
        # vy_invalid = vx_invalid.astype(bool)

        invalid = \
            vx_invalid | vy_invalid \
            | (np.abs(self.data.vx[:, :, start_x:stop_x]) > ITSLiveComposite.V_COMPONENT_THRESHOLD) \
            | (np.abs(self.data.vy[:, :, start_x:stop_x]) > ITSLiveComposite.V_COMPONENT_THRESHOLD)

        # Mask data
        logging.info(f'Mask invalid entries for vx...')
        vx = self.data.vx[:, :, start_x:stop_x].where(~invalid)

        logging.info(f'Mask invalid entries for vy...')
        vy = self.data.vy[:, :, start_x:stop_x].where(~invalid)

        invalid = np.nansum(invalid, axis=0)
        invalid = np.divide(invalid, np.sum(self.data.vx[:, :, start_x:stop_x].isnull(), 0) + invalid)

        logging.info(f'Finished filtering ({timeit.default_timer() - start_time} seconds)')

        # %% Least-squares fits to detemine amplitude, phase and annual means
        logging.info(f'Find vx annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        # vxamp, vxphase, vxmean, vxerr, vxsigma, vxcnt, _ = ITSLiveComposite.cubelsqfit2(vx, self.data.vx_error)
        _ = ITSLiveComposite.cubelsqfit2(
            vx,
            self.data.vx_error,
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
        # vyamp, vyphase, vymean, vyerr, vysigma, vycnt, _ = ITSLiveComposite.cubelsqfit2(vy, self.data.vy_error)
        _ = ITSLiveComposite.cubelsqfit2(
            vy,
            self.data.vy_error,
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
        vxm = np.nanmedian(self.mean.vx[:, :, start_x:stop_x], axis=0)
        vym = np.nanmedian(self.mean.vy[:, :, start_x:stop_x], axis=0)
        theta = np.arctan2(vxm, vym)

        # vx_sizes = vx.sizes
        # vx_time_dim = vx_sizes[Coords.MID_DATE]
        # vx_x_dim = vx_sizes[Coords.X]
        # vx_y_dim = vx_sizes[Coords.Y]

        # Explicitly expand the value
        stheta = np.tile(np.sin(theta).astype('float32'), (ITSLiveComposite.MID_DATE_LEN, 1, 1))
        ctheta = np.tile(np.cos(theta).astype('float32'), (ITSLiveComposite.MID_DATE_LEN, 1, 1))

        theta = None
        gc.collect()

        vx = vx*stheta + vy*ctheta

        stheta = None
        ctheta = None
        # Call Python's garbage collector
        gc.collect()

        # Now only np.abs(vxm) and np.abs(vym) are used, reset the variables
        vxm = np.abs(vxm)
        vym = np.abs(vym)

        # Expand dimensions of vectors
        vy = self.data.vx_error.expand_dims(Coords.Y)
        vy = vy.expand_dims(Coords.X)
        # Revert dimensions order: (mid_date, y, x)
        vy = vy.transpose()

        vy_expand = self.data.vy_error.expand_dims(Coords.Y)
        vy_expand = vy_expand.expand_dims(Coords.X)
        # Revert dimensions order: (mid_date, y, x)
        vy_expand = vy_expand.transpose()

        vy = (
            np.tile(vy, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN))*vxm + \
            np.tile(vy_expand, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN))*vym
        ) / np.sqrt(np.power(vxm, 2) + np.power(vym, 2))

        logging.info(f"vy.shape: {vy.shape}")

        # np.tile outputs np.ndarray type object, convert it back to xr.DataArray
        # type as expected by cubelsqfit2() method
        vy = xr.DataArray(
            data=vy,
            dims=(Coords.MID_DATE, Coords.Y, Coords.X),
            coords=[vx.mid_date.values, vx.y.values, vx.x.values]
        )

        # # Expand error data to be 3-d array
        # ITSCube.show_memory_usage(f'before vx expand')
        # vy = self.data.vx_error.expand_dims(Coords.Y)
        # vy = vy.expand_dims(Coords.X)
        # # Revert dimensions order: (mid_date, y, x)
        # vy = vy.transpose()
        #
        # vy = np.tile(
        #         vy,
        #         (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        #     )*vxm
        # logging.info(f"vy.shape: {vy.shape}")
        #
        # ITSCube.show_memory_usage(f'before vy expand')
        #
        # # Expand error data to be 3-d array
        # vy_expand = self.data.vy_error.expand_dims(Coords.Y)
        # vy_expand = vy_expand.expand_dims(Coords.X)
        # # Revert dimensions order: (mid_date, y, x)
        # vy_expand = vy_expand.transpose()
        # ITSCube.show_memory_usage(f'before vy += np.tile')
        #
        # vy += np.tile(
        #         vy_expand,
        #         (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        #     )*vym
        #
        # ITSCube.show_memory_usage(f'before vy /=')
        # vy /= np.sqrt(np.power(vxm, 2) + np.power(vym, 2))

        # vamp, vphase, vmean, verr, vsigma, vcnt, voutlier = ITSLiveComposite.cubelsqfit2(vx, vy)
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
        self.mean.v = np.abs(self.mean.v)

        # outlier = invalid + voutlier*(1-invalid)
        self.outlier_fraction[:, start_x:stop_x] = invalid + voutlier*(1-invalid)

    def to_zarr(self, output_store: str, s3_bucket: str, maxdt, outlier_frac):
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
        """
        Filter data cube by dt (date separation) between the images.
        """
        # %dtbin_mad_thresh: used to determine in dt means are significantly different
        # %dtbin_mad_thresh = 2*1.4826;

        # check if populations overlap (use first, smallest dt, bin as reference)
        # Matlab
        # [m, n, ~] = size(data);
        # data_dims = data.sizes
        # x_dim = data_dims[Coords.X]
        # y_dim = data_dims[Coords.Y]
        # date_dim = data_dims[Coords.MID_DATE]
        # dims = (date_dim, y_dim, x_dim)

        # Initialize output and functions
        # invalid = false(size(x));
        # maxdt = nan([m,n]);
        # madFun = @(x) median(abs(x - median(x))); % replaced on chad's suggestion
        dims = (ITSLiveComposite.MID_DATE_LEN, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        invalid = np.full(dims, False)

        dims = (ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)
        maxdt = np.full(dims, np.nan)
        # madFun = @(x) median(abs(x - median(x))); % replaced on chad's suggestion

        # Loop through all spacial points
        # ATTN: debugging only - process 2 x cells only
        # for j in tqdm(range(0, y_dim), ascii=True, desc='cube_filter: y'):
        #     for i in range(0, x_dim):
        for j in tqdm(range(0, 2), ascii=True, desc='cube_filter: y'):
            for i in tqdm(range(0, 2), ascii=True, desc='cube_filter: x'):
                # TODO: parallelize
                # Select by X, Y
                x0 = data.isel(x=i, y=j)
                # Since we are dealing with Dask arrays, load data in memory,
                # otherwise "groupby" functionality does not work
                x0 = x0.load()

                if np.all(x0.isnull()):
                    continue

                # Filter NAN values out
                mask = ~x0.isnull()
                x0 = x0.where(mask, drop=True)
                x0_dt = dt.where(mask, drop=True)

                np_digitize = np.digitize(x0_dt.values, ITSLiveComposite.DT_EDGE, right=False)
                index_var = xr.IndexVariable(Coords.MID_DATE, np_digitize)
                groups = x0.groupby(index_var)

                # Are means significantly different for various dt groupings?
                median = groups.median()
                xmad = groups.map(madFunction)

                # Check if populations overlap (use first, smallest dt, bin as reference)
                std_dev = xmad * ITSLiveComposite.DTBIN_MAD_THRESH * ITSLiveComposite.MAD_STD_RATIO
                minBound = median - std_dev
                maxBound = median + std_dev

                exclude = (minBound > maxBound[0]) | (maxBound < minBound[0])

                if np.any(exclude):
                    maxdt[j, i] = np.take(ITSLiveComposite.DT_EDGE, np.take(np_digitize, exclude)).min()
                    invalid[j, i] = dt > maxdt[j, i]

        return invalid, maxdt

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
        if len(v_err_data.sizes) != len(v.sizes):
            # Expand vector to 3-d array
            reshape_v_err = v_err_data.expand_dims(Coords.Y)
            reshape_v_err = reshape_v_err.expand_dims(Coords.X)
            # Revert dimensions order: (mid_date, y, x)
            reshape_v_err = reshape_v_err.transpose()

            v_err = xr.DataArray(
                np.tile(reshape_v_err, (1, ITSLiveComposite.Y_LEN, ITSLiveComposite.X_LEN)),
                dims=(Coords.MID_DATE, Coords.Y, Coords.X),
                coords={Coords.MID_DATE: v.mid_date.values, Coords.Y: v.y.values, Coords.X: v.x.values}
            )

        # for j in range(ITSLiveComposite.Y_LEN):
        #     for i in range(ITSLiveComposite.X_LEN):
        for j in range(1):
            for i in range(1):
                # TODO: parallelize
                x1 = v.isel(x=i, y=j)
                x_err1 = v_err.isel(x=i, y=j)

                mask = x1.notnull()
                if mask.sum() < ITSLiveComposite.NUM_VALID_POINTS:
                    continue

                # Annual phase and amplitude for processed years
                # amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1, outlier_frac[j, i] = ITSLiveComposite.itslive_lsqfit_annual(x1, x_err1)
                global_i = i + ITSLiveComposite.START_X

                ind, \
                amplitude[ind, j, global_i], \
                phase[ind, j, global_i], \
                sigma[ind, j, global_i], \
                mean[ind, j, global_i], \
                error[ind, j, global_i], \
                count[ind, j, global_i], \
                outlier_frac[j, i] = ITSLiveComposite.itslive_lsqfit_annual(x1, x_err1)

                # _, _, ind = np.intersect1d(ITSLiveComposite.YEARS, t_int1, return_indices=True)
                #
                # xmean[ind, j, i] = xmean1
                # err[ind, j, i] = err1
                # sigma[ind, j, i] = sigma1
                # amp[ind, j, i] = amp1
                # phase[ind, j, i] = phase1
                # cnt[ind, j, i] = cnt1

        # return (amp, phase, xmean, err, sigma, cnt, outlier_frac)
        return outlier_frac

    # def displacement(start_date, end_date, weights):
    #     """
    #     Displacement Vandermonde matrix: (these are displacements! not velocities,
    #     so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
    #     """
    #     return [(cos(2*pi*yr(:,1)) - cos(2*pi*yr(:,2)))./(2*pi) (sin(2*pi*yr(:,2)) - sin(2*pi*yr(:,1)))./(2*pi) M ones(size(dyr))];


    @staticmethod
    def itslive_lsqfit_annual(v, v_err):
        # Returns: [A,ph,A_err,t_int,v_int,v_int_err,N_int,outlier_frac]
        # % itslive_sinefit_lsq computes the amplitude and phase of seasonal velocity
        # % variability, and also gives computes interanual variability.
        # %
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
        start_year = np.array(ITSLiveComposite.START_DECIMAL_YEAR)[isf_mask]
        stop_year  = np.array(ITSLiveComposite.STOP_DECIMAL_YEAR)[isf_mask]

        v     = v.where(isf_mask, drop=True)
        v_err = v_err.where(isf_mask, drop=True)

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
            np_w_d = w_d.values
            np_w_d = np_w_d.reshape((len(w_d), 1))

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
            non_outlier_mask = ~outliers.values
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

            outliers_fraction = outliers.values.sum() / totalnum
            if outliers_fraction < 0.01:
                # There are less than 1% outliers, skip the rest of iterations
                logging.info(f'{outliers_fraction*100}% ({outliers.values.sum()} out of {totalnum}) outliers, done with first LSQ loop after {i+1} iterations')
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
        np_w_d = w_d.values
        np_w_d = np_w_d.reshape((len(w_d), 1))

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
        w_v = w_v.values.reshape((1, w_v.shape[0]))
        # logging.info(f'w_v.shape={w_v.shape}')
        # logging.info(f'M.type={M} M.shape={M.shape}')
        v_int_err = 1/np.sqrt((w_v@M).sum(axis=0))
        # logging.info(f"v_int_err: {v_int_err}")
        # logging.info(f"v_int_err.shape: {v_int_err.shape}")

        # Identify year's indices to assign return values to in "global" variables
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
        default=200,
        help='Number of X coordinates to process in parallel with Dask [%(default)d].'
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

    # s3://its-live-data/test_datacubes/AGU2021/S70W100/ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.zarr
    # AGU21_ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.nc

    mosaics = ITSLiveComposite(args.inputStore, args.bucket)
    mosaics.create(args.outputStore, args.bucket)
