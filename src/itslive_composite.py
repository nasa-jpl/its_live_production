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
from itscube import ITSCube
from itscube_types import Coords, DataVars


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

class Components:
    """
    Class to hold values for v, vx and vy data variables.
    """
    def __init__(self, dims: list, name: str):
        """
        Initialize data variables to hold results.
        """
        self.name = name
        self.value = np.full(dims, np.nan)
        self.x_comp = np.full(dims, np.nan)
        self.y_comp = np.full(dims, np.nan)

    def __str__(self):
        """
        String representation of the object.
        """
        return f"{self.name}: v_size={self.v.shape} vx_size={self.vx.shape} vy_size={self.vy.shape}"

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
    ACQ_DATETIME_IMG1 = None
    ACQ_DATETIME_IMG2 = None
    YEARS = None
    DATE_DT = None
    MIDDLE_DATE = None

    START_DECIMAL_YEAR = None
    STOP_DECIMAL_YEAR  = None

    # Minimum number of non-NAN values in the data to proceceed with processing
    NUM_VALID_POINTS = 5

    TWO_PI = np.pi * 2

    def __init__(self, cube_store: str, s3_bucket: str):
        """
        Initialize mosaics object.
        """
        self.s3, self.cube_store_in, cube_ds = ITSCube.init_input_store(cube_store, s3_bucket)
        # If reading NetCDF data cube
        # cube_ds = xr.open_dataset(cube_store, decode_timedelta=False)

        # Read in only specific data variables
        self.data = cube_ds[ITSLiveComposite.VARS]
        # self.data.load()

        # Sort cube data by datetime
        # self.data = self.data.sortby(Coords.MID_DATE)

        # TODO: introduce a method to determine mosaics granularity
        self.dates = [t.astype('M8[ms]').astype('O') for t in self.data[Coords.MID_DATE].values]

        # Images acquisition times and middle_date of each layer as datetime.datetime objects
        ITSLiveComposite.ACQ_DATETIME_IMG1 = [t.astype('M8[ms]').astype('O') for t in self.data[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1].values]
        ITSLiveComposite.ACQ_DATETIME_IMG2 = [t.astype('M8[ms]').astype('O') for t in self.data[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2].values]
        ITSLiveComposite.MIDDLE_DATE = [t.astype('M8[ms]').astype('O') for t in self.data[Coords.MID_DATE].values]

        # Compute decimal year representation for start and end dates of each velocity pair
        ITSLiveComposite.START_DECIMAL_YEAR = [decimal_year(each) for each in ITSLiveComposite.ACQ_DATETIME_IMG1]
        ITSLiveComposite.STOP_DECIMAL_YEAR = [decimal_year(each) for each in ITSLiveComposite.ACQ_DATETIME_IMG2]

        # Define time edges of composites
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
        self.x_len = cube_sizes[Coords.X]
        self.y_len = cube_sizes[Coords.Y]
        self.date_len = len(ITSLiveComposite.YEARS)
        dims = (self.x_len, self.y_len, self.date_len)

        self.v = Components(dims, 'v')
        self.error = Components(dims, 'error')
        self.count = Components(dims, 'count')
        self.amplitude = Components(dims, 'amplitude')
        self.phase = Components(dims, 'phase')
        self.sigma = Components(dims, 'sigma')

        dims = (self.x_len, self.y_len)
        self.outlier_frac = np.full(dims, np.nan)
        self.maxdt = np.full(dims, np.nan)

    def create(self, output_cube_store: str, s3_bucket: str):
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
        # TODO: loop through cube to minimize memory footprint
        # number of slices to read
        # x0 = 1:floor(sz(1)/ns):sz(1);
        # x0(end) = sz(1)+1;

        # % ML: To speed up trouble shooting, use one iteration only
        # % for i = 1:(length(x0)-1)
        # for i = 1:1

            # %% compute time averages
            # fprintf('------------------ CHUNK %.0f of %.0f --------------\n ', i, length(x0)-1)
            #
            # % read in data
            # fprintf('reading velocity data from %s ... ', fileName)
            #
            # % define subset to read
            # S.x =[x0(i),x0(i+1)-x0(i),1];
            # S.y =[1,sz(2),1];
            # S.mid_date =[1,sz(3),1];
            #
            # % read data
            # data = ncstruct(datacube, S, vars{:});
            # data

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

        # self.cube_time_mean(self.data, self.dates, self.years)

        # Start timer
        start_time = timeit.default_timer()

        # ----- FILTER DATA -----
        # Filter data based on locations where means of various dts are
        # statistically different and mad deviations from a running meadian
        logging.info(f'Filtering data based on dt binned medians...')

        # Initialize variables
        data_dims = self.data[DataVars.VX].sizes
        x_dim = data_dims[Coords.X]
        y_dim = data_dims[Coords.Y]
        date_dim = data_dims[Coords.MID_DATE]

        dims = (x_dim, y_dim, date_dim)
        vx_invalid = np.full(dims, False)
        vy_invalid = np.full(dims, False)

        # Loop for each unique sensor (those groupings image pairs that can be
        # expected to have different temporal decorelation)
        unique_sensors = list(set(self.data[DataVars.ImgPairInfo.SATELLITE_IMG1].values))

        num_sensors = len(unique_sensors)
        dims = (x_dim, y_dim, num_sensors)

        vx_maxdt = np.full(dims, np.nan)
        vy_maxdt = np.full(dims, np.nan)
        maxdt    = np.full(dims, np.nan)

        for i in range(len(unique_sensors)):
            # Find which layers correspond to each sensor
            mask = (self.data[DataVars.ImgPairInfo.SATELLITE_IMG1] == unique_sensors[i])
            vx_invalid[:, :, mask], vx_maxdt[:, :, i] = ITSLiveComposite.cube_filter(self.data.vx.where(mask), ITSLiveComposite.DATE_DT.where(mask))
            vy_invalid[:, :, mask], vy_maxdt[:, :, i] = ITSLiveComposite.cube_filter(self.data.vy.where(mask), ITSLiveComposite.DATE_DT.where(mask))

            # Get maximum value along sensor dimension: concatenate maxdt
            #  for vx and vy in new dimension
            maxdt[:, :, i] = np.nanmax(np.stack((vx_maxdt[:, :, i], vy_maxdt[:, :, i]), axis=2), axis=2)

        invalid = vx_invalid | vy_invalid | np.abs(data.vx) > ITSLiveComposite.V_COMPONENT_THRESHOLD | np.abs(data.vy) > ITSLiveComposite.V_COMPONENT_THRESHOLD

        # Mask data
        # TODO: not sure if this assignment will work
        self.data.vx[invalid] = np.nan
        self.data.vy[invalid] = np.nan

        invalid = np.nansum(invalid, axis=2)
        invalid = np.divide(invalid, np.sum(self.data.vx.isnull(), 2) + invalid)

        logging.info(f'Finished ({timeit.default_timer() - start_time})')
        #
        # fprintf('finished [%.1fmin]\n', (now-t0)*24*60)
        #
        # TODO before calling cubelsqfit2():
        # Add systematic error based on level of co-registration
        # Load Dask arrays before being able to modify their values
        self.data.vx_error.load()
        self.data.vy_error.load()

        for value, error in ITSLiveComposite.CO_REGISTRATION_ERROR.items():
            mask = (self.data.flag_stable_shift == value)
            self.data.vx_error[mask] += error
            self.data.vy_error[mask] += error

        # %% Least-squares fits to detemine amplitude, phase and annual means
        logging.info(f'Find vx annual means using LSQ fit ... ')
        start_time = timeit.default_timer()
        vxamp, vxphase, vxmean, vxerr, vxsigma, vxcnt, _ = ITSLiveComposite.cubelsqfit2(self.data.vx, self.data.vx_error, self.dates)
        logging.info(f'Finished vx (took {timeit.default_timer() - start_time} seconds)')

        #
        # fprintf('find vy annual means using lsg fit ... ')
        # [vyamp, vyphase, vymean, vyerr, vysigma, vycnt, ~] = cubelsqfit2(vy, vy_err, t, dt, yrs, mad_thresh,mad_filt_iterations);
        # fprintf('finished [%.1fmin]\n', (now-t0)*24*60)
        #
        # fprintf('find v annual means using lsg fit ... ')
        # % need to project velocity onto a unit flow vector to avoid biased (Change from Rician to Normal distribution)
        # vxm = median(vxmean,3,'omitnan');
        # vym = median(vymean,3,'omitnan');
        # theta = atan2(vxm, vym);
        #
        # % free up RAM by overwriting variables
        # % vx = rotated v
        # % vy = rotated v_err
        #
        # stheta = repmat(single(sin(theta)),[1,1,size(vx,3)]); % explicit expansion
        # ctheta = repmat(single(cos(theta)),[1,1,size(vx,3)]); % explicit expansion
        # vx = vx.*stheta + vy.*ctheta;
        #
        # vy = (repmat(reshape(vx_err, [1,1,length(vx_err)]),[size(vx,1), size(vx,2), 1]).*abs(vxm) + ...
        #     repmat(reshape(vy_err, [1,1,length(vy_err)]),[size(vx,1), size(vx,2), 1]).*abs(vym)) ./ sqrt(vxm.^2 + vym.^2); % explicit expansion
        #
        # [vamp, vphase, vmean, verr, vsigma, vcnt, voutlier] = cubelsqfit2(vx, vy, t, dt, yrs, mad_thresh, mad_filt_iterations);
        # vmean = abs(vmean); % because velocityies have been projected onto a mean flow direction they can be negative
        #
        # outlier = invalid + voutlier.*(1-invalid);
        #
        # fprintf('finished [%.1fmin]\n', (now-t0)*24*60)
        #
        # end

        # TODO: Save data to Zarr store

        # %% Save data to a netcdf
        # epsg = ncreadatt(datacube, 'mapping', 'spatial_epsg');
        # data = ncstruct(datacube, 'x', 'y', 'mid_date');
        # cubesave(outFileName, data.x, data.y, yrs, v_amp, vx_amp, vy_amp, v_phase, vx_phase, vy_phase, v, vx, vy, v_err, vx_err, vy_err, v_sigma, vx_sigma, vy_sigma, v_cnt, maxdt, outlier_frac, epsg)
        # delete(gcp) % close workers
        # fprintf('%s\ntime to build annual cube = %.1f min\n\n',fileName,(now-t0)*24*60)

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
        data_dims = data.sizes
        x_dim = data_dims[Coords.X]
        y_dim = data_dims[Coords.Y]
        date_dim = data_dims[Coords.MID_DATE]

        # Initialize output and functions
        # invalid = false(size(x));
        # maxdt = nan([m,n]);
        # madFun = @(x) median(abs(x - median(x))); % replaced on chad's suggestion
        dims = (x_dim, y_dim, date_dim)
        invalid = np.full(dims, False)

        dims = (x_dim, y_dim)
        maxdt = np.full(dims, np.nan)
        # madFun = @(x) median(abs(x - median(x))); % replaced on chad's suggestion

        # Loop through all spacial points
        # for i in range(0, x_dim):
        # ATTN: debugging only - process 2 x cells only
        for i in tqdm(range(0, 2), ascii=True, desc='cube_filter: x'):
            # for j in tqdm(range(0, y_dim), ascii=True, desc='cube_filter: y'):
            for j in tqdm(range(0, 2), ascii=True, desc='cube_filter: y'):
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
                    maxdt[i, j] = np.take(ITSLiveComposite.DT_EDGE, np.take(np_digitize, exclude)).min()
                    invalid[i, j] = dt > maxdt[i, j]

        return invalid, maxdt

    @staticmethod
    def cubelsqfit2(x, x_err, t, dt, years):
        """
        Cube LSQ fit (why 2 - 2 iterations?)

        Returns: [amp, phase, xmean, err, sigma, cnt, outlier_frac]
        """
        #
        # Get start and end dates
        # tt = [t - dt/2, t+dt/2];
        #
        # % initialize
        data_dims = x.sizes
        x_dim = data_dims[Coords.X]
        y_dim = data_dims[Coords.Y]
        year_dim = len(ITSLiveComposite.YEARS)
        dims = (x_dim, y_dim, year_dim)

        xmean = np.full(dims, np.nan)
        err = np.full(dims, np.nan)
        cnt = np.full(dims, np.nan)
        amp = np.full(dims, np.nan)
        phase = np.full(dims, np.nan)
        sigma = np.full(dims, np.nan)

        dims = (x_dim, y_dim)
        outlier_frac = np.full(dims, np.nan)

        # This is only done for generic parfor "slicing" may not be needed when
        # recoded
        if len(x_err.sizes) != len(x.sizes):
            # Populate the whole array with provided single value per cell
            reshape_x_err = x_err.values[:, np.newaxis, np.newaxis]
            x_err = xr.DataArray(
                np.tile(reshape_x_err, (y_dim, x_dim)),
                dims=(Coords.MID_DATE, Coords.Y, Coords.X),
                coords={Coords.MID_DATE: self.data.mid_date.values, Coords.Y: self.data.y.values, Coords.X: self.data.x.values}
            )

        for i in range(x_dim):
            for j in range(y_dim):
                # TODO: parallelize
                x1 = x.isel(x=i, y=j)
                x_err1 = x_err.isel(x=i, y=j)

                # xmean0 = xmean(i,:,:);
                # cnt0 = cnt(i,:,:);
                # err0 = err(i,:,:);
                # sigma0 = sigma(i,:,:);
                #
                # amp0 = amp(i,:,:);
                # phase0 = phase(i,:,:);
                # outlier_frac0 = outlier_frac(i,:);

                mask = x1.nonnull()
                if mask.sum() < ITSLiveComposite.NUM_VALID_POINTS:
                    continue

                # % single phase and amplitude
                # %[amp0(1,j), phase0(1,j), sigma1, t_int1, xmean1, err1, cnt1, outlier_frac0(1,j)] = ...
                # %    itslive_lsqfit(tt(idx,:),x1(idx),x_err1(idx), mad_thres);

                # Annual phase and amplitude
                amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1, outlier_frac0[i, j] = itslive_lsqfit_annual(x1, x_err1)

                _, _, ind = np.intersect1d(years, t_int1, return_indices=True)

                xmean[i, j, ind] = xmean1
                err[i, j, ind] = err1
                sigma[i, j, ind] = sigma1
                amp[i, j, ind] = amp1
                phase[i, j, ind] = phase1
                cnt[i, j, ind] = cnt1

        return (amp, phase, xmean, err, sigma, cnt, outlier_frac)

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
        y_1 = int(np.floor(start_year.min()))
        y_2 = int(np.floor(stop_year.max())) + 1
        y1 = np.arange(y_1, y_2)

        M = np.zeros((len(dyr), len(y1)))

        # Loop through each year:
        for k in range(len(y1)):
            # Set all measurements that begin before the first day of the year and end after the last
            # day of the year to 1:
            ind = start_year <= y1[k] & stop_year >= (y1[k] + 1)
            M[ind, k] = 1

            # Within year:
            ind = start_year >= y1[k] & stop_year < (y1[k] + 1)
            M[ind, k] = dyr[ind]

            # Started before the beginning of the year and ends during the year:
            ind = start_year < y1[k] & stop_year >= y1[k] & stop_year < (y1[k] + 1)
            M[ind, k] = stop_year[ind] - y1[k]

            # Started during the year and ends the next year:
            ind = start_year >= y1[k] & start_date < (y1[k] + 1) & stop_year >= (y1[k]+1)
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
            p = p.linalg.lstsq(np_w_d * A, w_d*d_obs, rcond=None)[0]

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
                # print(f'{outliers_fraction*100}% ({np.sum(outliers)} out of {totalnum}) outliers, done with first LSQ iteration')
                break

        outlier_frac = (totalnum - length(yr))/totalnum

        #
        # Second LSQ iteration
        #
        # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
        # D = [(cos(2*pi*yr(:,1)) - cos(2*pi*yr(:,2)))./(2*pi).*(M>0) (sin(2*pi*yr(:,2)) - sin(2*pi*yr(:,1)))./(2*pi).*(M>0) M];
        M_pos = M > 0
        D = np.block([
            ((np.cos(two_pi*start_year) - np.cos(two_pi*stop_year))/two_pi).reshape((len(M_pos), 1)) * M_pos,\
            ((np.sin(two_pi*stop_year) - np.sin(two_pi*start_year))/two_pi).reshape((len(M_pos), 1)) * M_pos])

        # D = np.block([
        #     ((np.cos(ITSLiveComposite.TWO_PI*start_year) - np.cos(ITSLiveComposite.TWO_PI*stop_year))/ITSLiveComposite.TWO_PI) * M_pos,\
        #     ((np.sin(ITSLiveComposite.TWO_PI*stop_year) - np.sin(ITSLiveComposite.TWO_PI*start_year))/ITSLiveComposite.TWO_PI]) * M_pos])

        # Add M: a different constant for each year (annual mean)
        D = np.concatenate([D, M], axis=1)

        # Make numpy happy: have all data 2D
        np_w_d = w_d.values
        np_w_d = np_w_d.reshape((len(w_d), 1))

        # These both are of xarray type, so can multiply them
        # w_d*d_obs

        # Solve for coefficients of each column in the Vandermonde:
        p = p.linalg.lstsq(np_w_d * D, w_d*d_obs, rcond=None)[0]

        # Postprocess
        #
        # Convert coefficients to amplitude and phase of a single sinusoid:
        Nyrs = len(y1)
        # Amplitude of sinusoid from trig identity a*sin(t) + b*cos(t) = d*sin(t+phi), where d=hypot(a,b) and phi=atan2(b,a).
        A = np.hypot(p[0:Nyrs], p[Nyrs:2*Nyrs])

        # phase in radians
        ph_rad = np.arctan2(p[Nyrs:2*Nyrs], p[0:Nyrs])

        # phase converted such that it reflects the day when value is maximized
        ph = 365.25*((0.25 - ph_rad/two_pi) % 1)

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

        v_int_err =  1/np.sqrt(w_v*M.sum())
        print(f"v_int_err: {v_int_err}")
        v_int_err.shape

        return [A, ph, A_err, y1, v_int, v_int_err, N_int, outlier_frac]

if __name__ == '__main__':
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSLiveComposite.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument(
    #     '-t', '--threads',
    #     type=int, default=4,
    #     help='Number of Dask workers to use for parallel processing [%(default)d].'
    # )
    # parser.add_argument(
    #     '-s', '--scheduler',
    #     type=str,
    #     default="processes",
    #     help="Dask scheduler to use. One of ['threads', 'processes'] (effective only when --parallel option is specified) [%(default)s]."
    # )
    # parser.add_argument(
    #     '-p', '--parallel',
    #     action='store_true',
    #     default=False,
    #     help='Enable parallel processing, default is to process all cube spacial points in parallel'
    # )
    parser.add_argument(
        '-i', '--inputStore',
        type=str,
        default=None,
        help="Input Zarr datacube store to generate mosaics for [%(default)s]."
    )
    parser.add_argument(
        '-o', '--outputStore',
        type=str,
        default="cube_mosaics.zarr",
        help="Zarr output directory to write mosaics data to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--outputBucket',
        type=str,
        default='',
        help="S3 bucket to copy Zarr format of the datacube to [%(default)s]."
    )
    args = parser.parse_args()

    # s3://its-live-data/test_datacubes/AGU2021/S70W100/ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.zarr
    # AGU21_ITS_LIVE_vel_EPSG3031_G0120_X-1550000_Y-450000.nc

    mosaics = ITSLiveComposite(args.inputStore, args.outputBucket)
    mosaics.create(args.outputStore, args.outputBucket)
