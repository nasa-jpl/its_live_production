"""
ITSLiveComposite class creates yearly composites of ITS_LIVE datacubes with data
within the same target projection, bounding polygon and datetime period as
specified at the time the datacube was constructed/updated.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Green (JPL)

Jet Propulsion Laboratory, California Institute of Technology, Pasadena, California
March 21, 2022
"""
import collections
import copy
import datetime
import gc
import logging
import numba as nb
import numpy  as np
import os
import pandas as pd
import s3fs
from scipy import ndimage
import timeit
from tqdm import tqdm
import xarray as xr
import zarr

# Local imports
from itscube import ITSCube
from itscube_types import Coords, DataVars

class CompDataVars:
    """
    Data variables and their descriptions to write to Zarr or NetCDF output store.
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
    OUTLIER_FRAC = 'outlier_frac'
    VX0 = 'vx0'
    VY0 = 'vy0'
    V0  = 'v0'
    COUNT0 = 'count0'
    VX0_ERROR = 'vx0_error'
    VY0_ERROR = 'vy0_error'
    V0_ERROR  = 'v0_error'
    SLOPE_VX  = 'dvx_dt'
    SLOPE_VY  = 'dvy_dt'
    SLOPE_V   = 'dv_dt'

    STD_NAME = {
        DataVars.VX: 'x_velocity',
        DataVars.VY: 'y_velocity',
        DataVars.V:  'velocity',
        VX_ERROR: 'x_velocity_error',
        VY_ERROR: 'y_velocity_error',
        V_ERROR:  'velocity_error',
        VX_AMP_ERROR: 'vx_amplitude_error',
        VY_AMP_ERROR: 'vy_amplitude_error',
        V_AMP_ERROR:  'v_amplitude_error',
        VX_AMP: 'vx_amplitude',
        VY_AMP: 'vy_amplitude',
        V_AMP:  'v_amplitude',
        VX_PHASE: 'vx_phase',
        VY_PHASE: 'vy_phase',
        V_PHASE:  'v_phase',
        SENSORS: 'sensors',
        TIME: 'time',
        COUNT: 'count',
        MAX_DT: 'dt_maximum',
        OUTLIER_FRAC: 'outlier_fraction',
        VX0: 'climatological_x_velocity',
        VY0: 'climatological_y_velocity',
        V0: 'climatological_velocity',
        COUNT0: 'count0',
        VX0_ERROR: 'vx0_velocity_error',
        VY0_ERROR: 'vy0_velocity_error',
        V0_ERROR: 'v0_velocity_error',
        SLOPE_VX: 'dvx_dt',
        SLOPE_VY: 'dvy_dt',
        SLOPE_V:  'dv_dt'
    }

    DESCRIPTION = {
        DataVars.VX:  'mean annual velocity of sinusoidal fit to vx',
        DataVars.VY:  'mean annual velocity of sinusoidal fit to vy',
        DataVars.V:   'mean annual velocity of sinusoidal fit to v',
        TIME:         'time',
        VX_ERROR:     'error weighted error for vx',
        VY_ERROR:     'error weighted error for vy',
        V_ERROR:      'error weighted error for v',
        VX_AMP_ERROR: 'error for vx_amp',
        VY_AMP_ERROR: 'error for vy_amp',
        V_AMP_ERROR:  'error for v_amp',
        VX_AMP:       'climatological mean seasonal amplitude of sinusoidal fit to vx',
        VY_AMP:       'climatological mean seasonal amplitude in sinusoidal fit in vy',
        V_AMP:        'climatological mean seasonal amplitude of sinusoidal fit to v',
        VX_PHASE:     'day of maximum velocity of sinusoidal fit to vx',
        VY_PHASE:     'day of maximum velocity of sinusoidal fit to vy',
        V_PHASE:      'day of maximum velocity of sinusoidal fit to v',
        COUNT:        'number of image pairs used in error weighted least squares fit',
        MAX_DT:       'maximum allowable time separation between image pair acquisitions included in error weighted least squares fit',
        OUTLIER_FRAC: 'fraction of data identified as outliers and excluded from error weighted least squares fit',
        SENSORS:      'combinations of unique sensors and missions that are grouped together for date_dt filtering',
        VX0:          'climatological vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values. The climatology is arbitrarily fixed to a y-intercept of July 2, 2017.',
        VY0:          'climatological vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values. The climatology is arbitrarily fixed to a y-intercept of July 2, 2017.',
        V0:           'climatological v determined by taking the hypotenuse of vx0 and vy0. The climatology is arbitrarily fixed to a y-intercept of July 2, 2017.',
        COUNT0:       'number of image pairs used for climatological means',
        VX0_ERROR:    'error for vx0',
        VY0_ERROR:    'error for vy0',
        V0_ERROR:     'error for v0',
        SLOPE_VX:     'trend in vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values',
        SLOPE_VY:     'trend in vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values',
        SLOPE_V:      'trend in v determined by projecting dvx_dt and dvy_dt onto the unit flow vector defined by vx0 and vy0'
    }

class CompOutputFormat:
    """
    Class to represent attrubutes for the output format of the data.
    """
    GDAL_AREA_OR_POINT = 'GDAL_AREA_OR_POINT'
    COMPOSITES_SOFTWARE_VERSION = 'composites_software_version'
    DATACUBE_AUTORIFT_PARAMETER_FILE = 'datacube_autoRIFT_parameter_file'
    DATACUBE_SOFTWARE_VERSION = 'datacube_software_version'
    DATE_CREATED = 'date_created'
    DATE_UPDATED = 'date_updated'
    DATECUBE_CREATED = 'datecube_created'
    DATECUBE_S3 = 'datecube_s3'
    DATECUBE_UPDATED = 'datecube_updated'
    DATECUBE_URL = 'datecube_url'
    PROJECTION = 'projection'
    SENSORS_LABELS = 'sensors_labels'


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

def decimal_year(dt):
    start_year = datetime.datetime(year=dt.year, month=1, day=1)
    year_part = dt - start_year
    year_length = (
        datetime.datetime(year=dt.year, month=12, day=31, hour=23, minute=59, second=59) - \
        start_year
    )
    return dt.year + year_part / year_length

@nb.jit(nopython=True)
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

# @nb.jit(nopython=True)
def cube_filter_iteration(x_in, y_in, dt, mad_std_ratio):
    """
    Filter one spacial point by dt (date separation) between the images.
    """
    # Filter parameters for dt bins (default: 2 - TODO: ask Alex):
    # used to determine if dt means are significantly different
    _dtbin_mad_thresh = 0.67

    _dtbin_ratio = _dtbin_mad_thresh * mad_std_ratio

    _dt_edge = np.array([0, 32, 64, 128, 256, np.inf])
    _num_bins = len(_dt_edge)-1

    _dt_median_flow = np.array([16, 32, 64, 128, 256])

    # Skip dt filter for slow moving areas
    _min_v0_threshold = 5

    # Output data variables
    maxdt = np.nan
    # Make numba happy - use np.bool_ type
    invalid = np.zeros_like(dt, dtype=np.bool_)

    x0_is_null = np.isnan(x_in)
    if np.all(x0_is_null):
        # No data to process
        return (maxdt, invalid)

    # Project vx and vy onto the median flow vector for dt <= 16.
    # If there is no data, then for dt <= 32
    ind = None
    valid = ~x0_is_null # Number of valid points
    num_valid = valid.sum()

    for each_dt in _dt_median_flow:
        ind = (dt <= each_dt) & valid

        dt_fraction = ind.sum()/num_valid
        if dt_fraction >= 0.2:
            break

    # Make numba happy
    ind = ind.astype(np.bool_)

    vx0 = np.median(x_in[ind])
    vy0 = np.median(y_in[ind])
    v0 = np.sqrt(vx0**2 + vy0**2)

    if v0 <= _min_v0_threshold:
        maxdt = np.inf
        return (maxdt, invalid)

    uv_x = vx0/v0 # unit flow vector
    uv_y = vy0/v0
    x0_in = x_in*uv_x + y_in*uv_y # projected flow vectors

    # Filter NAN values out
    # logging.info(f'Before mask filter: type(x0)={type(x0_in)}')
    x0_is_null = np.isnan(x0_in)
    if np.all(x0_is_null):
        # No data to process
        return (maxdt, invalid)

    mask = ~x0_is_null
    x0 = x0_in[mask]
    x0_dt = dt[mask]

    # Group data values by identified bins "manually":
    # since data is sorted by date_dt, we can identify index boundaries
    # for each bin within the "date_dt" vector
    # logging.info(f'Before searchsorted')
    bin_index = np.searchsorted(x0_dt, _dt_edge)

    xmed = []
    xmad = []

    # Collect indices for bins that represent current x0_dt
    dt_bin_indices = []

    for bin_i in range(0, _num_bins):
        # if bin_index[bin_i] and bin_index[bin_i+1] are the same, there are no values for the bin, skip it
        if bin_index[bin_i] != bin_index[bin_i+1]:
            bin_xmed, bin_xmad = medianMadFunction(x0[bin_index[bin_i]:bin_index[bin_i+1]])
            xmed.append(bin_xmed)
            xmad.append(bin_xmad)
            dt_bin_indices.append(bin_i)

    # Check if populations overlap (use first, smallest dt, bin as reference)
    # logging.info(f'Before min/max bound')
    std_dev = np.array(xmad) * _dtbin_ratio
    xmed = np.array(xmed)
    minBound = xmed - std_dev
    maxBound = xmed + std_dev

    exclude = (minBound > maxBound[0]) | (maxBound < minBound[0])

    if np.any(exclude):
        dt_bin_indices = np.array(dt_bin_indices)[exclude]
        maxdt = _dt_edge[dt_bin_indices].min()
        invalid = dt > maxdt

    return (maxdt, invalid)

# @nb.jit(nopython=True, parallel=True)
def cube_filter(data_x, data_y, dt, mad_std_ratio):
    """
    Filter data cube by dt (date separation) between the images.
    """
    # Initialize output
    # Make numba happy - use np.bool_ type
    invalid = np.zeros_like(data_x, dtype=np.bool_)

    y_len, x_len, _ = data_x.shape
    dims = (y_len, x_len)
    maxdt = np.full(dims, np.nan)

    # Loop through all spacial points
    # for j_index in nb.prange(y_len):
    #     for i_index in nb.prange(x_len):
    for j_index in range(0, y_len):
        for i_index in range(0, x_len):
            maxdt[j_index, i_index], invalid[j_index, i_index, :] = cube_filter_iteration(
                data_x[j_index, i_index, :],
                data_y[j_index, i_index, :],
                dt,
                mad_std_ratio
            )

    return invalid, maxdt

@nb.jit(nopython=True)
def weighted_std(values, weights):
    """
    Return weighted standard deviation.

    Reference: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

@nb.jit(nopython=True)
def create_M(y1, start_year, stop_year, dyr):
    """
    Make matrix of percentages of years corresponding to each displacement measurement
    """
    M = np.zeros((len(dyr), len(y1)))

    # Loop through each year:
    for k in range(len(y1)):
        # Set all measurements that begin before the first day of the year and end after the last
        # day of the year to 1:
        y1_value = y1[k]
        y1_next_value = y1_value + 1

        ind = np.logical_and(start_year <= y1_value, stop_year >= y1_next_value)
        M[ind, k] = 1

        # Within year:
        ind = np.logical_and(start_year >= y1_value, stop_year < y1_next_value)
        M[ind, k] = dyr[ind]

        # Started before the beginning of the year and ends during the year:
        ind = np.logical_and(start_year < y1_value, np.logical_and(stop_year >= y1_value, stop_year < y1_next_value))
        M[ind, k] = stop_year[ind] - y1_value

        # Started during the year and ends the next year:
        ind = np.logical_and(start_year >= y1_value, np.logical_and(start_year < y1_next_value, stop_year >= y1_next_value))
        M[ind, k] = y1_next_value - start_year[ind]

    return M

@nb.jit(nopython=True)
def itslive_lsqfit_iteration(start_year, stop_year, M, w_d, d_obs):
    _two_pi = np.pi * 2

    #
    # LSQ fit iteration
    #
    # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
    # D = [(cos(2*pi*yr(:,1)) - cos(2*pi*yr(:,2)))./(2*pi).*(M>0) (sin(2*pi*yr(:,2)) - sin(2*pi*yr(:,1)))./(2*pi).*(M>0) M];
    D = np.stack( \
            ((np.cos(_two_pi*start_year) - np.cos(_two_pi*stop_year))/_two_pi, \
             (np.sin(_two_pi*stop_year) - np.sin(_two_pi*start_year))/_two_pi), axis=-1)

    # Add M: a different constant for each year (annual mean)
    D = np.concatenate((D, M), axis=1)

    # Make numpy happy: have all data 2D
    # w_d.reshape((len(w_d), 1))

    # Solve for coefficients of each column in the Vandermonde:
    p = np.linalg.lstsq(w_d.reshape((len(w_d), 1)) * D, w_d*d_obs)[0]

    # Goodness of fit:
    d_model = (D * p).sum(axis=1)  # modeled displacements (m)

    return (p, d_model)

@nb.jit(nopython=True)
def itersect_years(all_years, select_years):
    """
    Get indices of "select_years" into "all_years" array.
    This is to replace built-in numpy.intersect1d() which does not work with
    numba.
    """
    lookup_table = {v:i for i, v in enumerate(all_years)}
    return np.array([lookup_table[each] for each in select_years])

@nb.jit(nopython=True)
def init_lsq_fit1(v_input, v_err_input, start_dec_year, stop_dec_year, dec_dt, all_years, M_input):
    """
    Initialize variables for LSQ fit.
    """
    # start_time = timeit.default_timer()
    # logging.info(f"Start init of itslive_lsqfit_annual")

    # Ensure we're starting with finite data
    isf_mask   = np.isfinite(v_input) & np.isfinite(v_err_input)
    start_year = start_dec_year[isf_mask]
    stop_year  = stop_dec_year[isf_mask]
    # dt in decimal years
    dyr = dec_dt[isf_mask]

    v_in = v_input[isf_mask]
    v_err_in = v_err_input[isf_mask]
    M_in = M_input[isf_mask]

    totalnum = len(start_year)

    # Sort arrays based on the mid_date
    mid_date = start_year + (stop_year - start_year)/2.0
    sort_indices = np.argsort(mid_date)

    # Sort inputs
    start_year = start_year[sort_indices]
    stop_year = stop_year[sort_indices]
    dyr = dyr[sort_indices]

    v_in = v_in[sort_indices]
    v_err_in = v_err_in[sort_indices]
    M_in = M_in[sort_indices]

    return (start_year, stop_year, v_in, v_err_in, dyr, totalnum, M_in)

@nb.jit(nopython=True)
def init_lsq_fit2(v_median, v_input, v_err_input, start_dec_year, stop_dec_year, dec_dt, all_years, M_input, mad_thresh, mad_std_ratio):
    """
    Initialize variables for LSQ fit.
    """
    # Remove outliers based on MAD filter for v, subtract from v to get residual
    v_residual = np.abs(v_input - v_median)

    # Take median of residual, multiply median of residual * 1.4826 = sigma
    v_sigma = np.median(v_residual)*mad_std_ratio

    non_outlier_mask  = ~(v_residual > (2.0 * mad_thresh * v_sigma))

    # remove ouliers from v_in, v_error_in, start_dec_year, stop_dec_year
    start_year = start_dec_year[non_outlier_mask]
    stop_year = stop_dec_year[non_outlier_mask]
    dyr = dec_dt[non_outlier_mask]
    v_in = v_input[non_outlier_mask]
    v_err_in = v_err_input[non_outlier_mask]
    M_in = M_input[non_outlier_mask]

    # Weights for velocities
    w_v = 1/(v_err_in**2)

    # Weights (correspond to displacement error, not velocity error):
    w_d = 1/(v_err_in*dyr)  # Not squared because the p= line below would then have to include sqrt(w) on both accounts
    # logging.info(f"w_d.shape: {w_d.shape}")

    # Observed displacement in meters
    d_obs = v_in*dyr
    # logging.info(f"d_obs.shape: {d_obs.shape}")

    # logging.info(f'Finished init of itslive_lsqfit_annual ({timeit.default_timer() - start_time} seconds)')
    # start_time = timeit.default_timer()
    # logging.info(f"Start building M")

    # Make matrix of percentages of years corresponding to each displacement measurement
    y_min = int(np.floor(start_year.min()))
    y_max = int(np.floor(stop_year.max())) + 1
    y1 = np.arange(y_min, y_max)

    # Reduce M matrix to the years considered for the spacial point
    year_indices = np.searchsorted(all_years, y1)
    M_in = M_in[:, year_indices]

    return (start_year, stop_year, v_in, v_err_in, dyr, w_v, w_d, d_obs, y1, M_in)

# Don't compile the whole function with numba - runs a bit slower (why???)
# @nb.jit(nopython=True)
def itslive_lsqfit_annual(
    v_input,
    v_err_input,
    start_dec_year,
    stop_dec_year,
    dec_dt,
    all_years,
    M_input,
    mad_std_ratio,
    mean,  # outputs to populate
    error,
    count):
    # Populates [A,ph,A_err,t_int,v_int,v_int_err,N_int,outlier_frac] data
    # variables.
    # Computes the amplitude and phase of seasonal velocity
    # variability, and also gives interannual variability.
    #
    # From original Matlab code:
    # % [A,ph,A_err,t_int,v_int,v_int_err,N_int] = itslive_sinefit_lsq(t,v,v_err)
    # % also returns the standard deviation of amplitude residuals A_err. Outputs
    # % t_int and v_int describe interannual velocity variability, and can then
    # % be used to reconstruct a continuous time series, as shown below. Output
    # % Output N_int is the number of image pairs that contribute to the annual mean
    # % v_int of each year. The output |v_int_err| is a formal estimate of error
    # % in the v_int.
    # %
    # %% Author Info
    # % Chad A. Greene, Jan 2020.
    # %
    _two_pi = np.pi * 2

    # Filter parameters for lsq fit for outlier rejections
    _mad_thresh = 6
    _mad_filter_iterations = 1

    # Apply MAD filter to input v
    _mad_kernel_size = 15

    init_runtime = timeit.default_timer()

    start_year, stop_year, v, v_err, dyr, totalnum, M = init_lsq_fit1(
        v_input, v_err_input, start_dec_year, stop_dec_year, dec_dt, all_years, M_input
    )
    init_runtime1 = timeit.default_timer() - init_runtime

    # Compute outside of numba-compiled code as numba does not support a lot of scipy
    # functionality
    # Apply 15-point moving median to v, subtract from v to get residual
    init_runtime = timeit.default_timer()
    v_median = ndimage.median_filter(v, _mad_kernel_size)
    init_runtime2 = timeit.default_timer() - init_runtime

    init_runtime = timeit.default_timer()
    start_year, stop_year, v, v_err, dyr, w_v, w_d, d_obs, y1, M = init_lsq_fit2(
        v_median, v, v_err, start_year, stop_year, dyr, all_years, M, _mad_thresh, mad_std_ratio
    )
    init_runtime3 = timeit.default_timer() - init_runtime

    # Filter sum of each column
    # WAS: hasdata = M.sum(axis=0) > 0
    hasdata = M.sum(axis=0) >= 1.0
    y1 = y1[hasdata]
    M = M[:, hasdata]

    # logging.info(f'Finished building M and filter by M ({timeit.default_timer() - start_time} seconds)')
    # start_time = timeit.default_timer()
    # logging.info(f"Start 1st iteration of LSQ")

    #
    # LSQ iterations
    # Iterative mad filter
    p = None
    d_model = None

    # Last iteration of LSQ should always skip the outlier filter
    last_iteration = _mad_filter_iterations - 1

    iter_runtime = 0
    for i in range(0, _mad_filter_iterations):
        # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
        # p, d_model = itslive_lsqfit_iteration(start_year, stop_year, M, w_d, d_obs, dyr)
        runtime = timeit.default_timer()
        p, d_model = itslive_lsqfit_iteration(start_year, stop_year, M, w_d, d_obs)
        iter_runtime += (timeit.default_timer() - runtime)

        # Divide by dt to avoid penalizing long dt [asg]
        d_resid = np.abs(d_obs - d_model)/dyr

        # Robust standard deviation of errors, using median absolute deviation
        d_sigma = np.median(d_resid)*mad_std_ratio

        outliers = d_resid > (_mad_thresh * d_sigma)
        if np.all(outliers):
            # All are outliers, return from the function
            return 1.0

        if (outliers.sum() / totalnum) < 0.01 and i != last_iteration:
            # There are less than 1% outliers, skip the rest of iterations
            # if it's not the last iteration
            # logging.info(f'{outliers_fraction*100}% ({outliers.sum()} out of {totalnum}) outliers, done with first LSQ loop after {i+1} iterations')
            break

        if i < last_iteration:
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

            if not np.any(hasdata):
                # Since we are throwing away everything, report all as outliers
                return 1.0

            y1 = y1[hasdata]
            M = M[:, hasdata]

    # logging.info(f'Size of p:{p.shape}')

    # ATTN: Matlab had it probably wrong, but confirm on output: outlier_frac = length(yr)./totalnum;
    outlier_frac = (totalnum - len(start_year))/totalnum

    # Convert coefficients to amplitude and phase of a single sinusoid:
    Nyrs = len(y1)

    # Amplitude of sinusoid from trig identity a*sin(t) + b*cos(t) = d*sin(t+phi), where d=hypot(a,b) and phi=atan2(b,a).
    # WAS: A = np.hypot(p[0:Nyrs], p[Nyrs:2*Nyrs])
    A = np.hypot(p[0], p[1])

    # phase in radians
    # ph_rad = np.arctan2(p[Nyrs:2*Nyrs], p[0:Nyrs])
    ph_rad = np.arctan2(p[1], p[0])

    # phase converted such that it reflects the day when value is maximized
    ph = 365.25*((0.25 - ph_rad/_two_pi) % 1)

    # A_err is the *velocity* (not displacement) error, which is the displacement error divided by the weighted mean dt:
    # WAS: A_err = np.full_like(A, np.nan)
    A_err = np.full((Nyrs), np.nan)

    for k in range(Nyrs):
        ind = M[:, k] > 0

        # asg replaced call to wmean
        A_err[k] = weighted_std(d_obs[ind]-d_model[ind], w_d[ind]) / ((w_d[ind]*dyr[ind]).sum() / w_d[ind].sum())

    # Compute climatology amplitude error based on annual values
    amp_error = np.sqrt((A_err**2).sum())/(Nyrs-1)

    # WAS: v_int = p[2*Nyrs:]
    v_int = p[2:]

    # Number of equivalent image pairs per year: (1 image pair equivalent means a full year of data. It takes about 23 16-day image pairs to make 1 year equivalent image pair.)
    N_int = (M>0).sum(axis=0)

    # Reshape array to have the same number of dimensions as M for multiplication
    w_v = w_v.reshape((1, w_v.shape[0]))

    v_int_err = 1/np.sqrt((w_v@M).sum(axis=0))

    # Identify year's indices to assign return values to in "final" composite
    # variables
    ind = itersect_years(all_years, y1)

    # logging.info(f'Finished post-process ({timeit.default_timer() - start_time} seconds)')
    # start_time = timeit.default_timer()

    # On return: amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1, outlier_fraction
    # amplitude[ind] = A
    # phase[ind] = ph
    # sigma[ind] = A_err
    mean[ind] = v_int
    error[ind] = v_int_err
    count[ind] = N_int

    offset, slope, se = weighted_linear_fit(y1, mean[ind], error[ind])

    return A, amp_error, ph, offset, slope, se, outlier_frac, init_runtime1, init_runtime2, init_runtime3, iter_runtime

# @nb.jit(nopython=True)
def annual_magnitude(
    vx0,
    vy0,
    vx_fit,
    vy_fit,
    vx_fit_err,
    vy_fit_err,
    vx_fit_count,
    vy_fit_count,
    vx_fit_outlier_frac,
    vy_fit_outlier_frac,
    # v_fit, # outputs
    # v_fit_err,
    # v_fit_count
):
    """
    Computes and returns the annual mean, error, count, and outlier fraction
    from component values projected on the unit flow vector defined by vx0 and vy0.

    Inputs:
        vx0: mean flow in x direction
        vy0: mean flow in y direction
        vx_fit: annual mean flow in x direction
        vy_fit: annual mean flow in y direction
        vx_fit_err: error in annual mean flow in x direction
        vy_fit_err: error in annual mean flow in y direction
        vx_fit_count: number of values used to determine annual mean flow in x direction
        vy_fit_count: number of values used to determine annual mean flow in y direction
        vx_fit_outlier_frac: fraction of data identified as outliers and removed
            when calculating annual mean flow in x direction
        vy_fit_outlier_frac: fraction of data identified as outliers and removed
            when calculating annual mean flow in y direction

    Outputs:
        self.mean.v[start_y:stop_y, start_x:stop_x, :]
        self.error.v[start_y:stop_y, start_x:stop_x, :]
        self.count.v[start_y:stop_y, start_x:stop_x, :]
        voutlier
    """
    # solve for velocity magnitude
    v_fit = np.sqrt(vx_fit**2 + vy_fit**2) # velocity magnitude

    y_len, x_len, years_len = v_fit.shape
    expand_dims = (y_len, x_len, years_len)

    vx0_exp = np.broadcast_to(vx0.reshape((y_len, x_len, 1)), expand_dims)
    vy0_exp = np.broadcast_to(vy0.reshape((y_len, x_len, 1)), expand_dims)

    # logging.info(f'vx0_exp: {vx0_exp.shape} vy0_exp: {vy0_exp.shape}')
    uv_x = vx0_exp/v_fit # unit flow vector
    uv_y = vy0_exp/v_fit

    v_fit_err = np.abs(vx_fit_err) * np.abs(uv_x) # flow acceleration in direction of unit flow vector, take absolute values
    v_fit_err += np.abs(vy_fit_err) * np.abs(uv_y)

    v_fit_count = np.ceil((vx_fit_count + vy_fit_count) / 2)
    v_fit_outlier_frac = (vx_fit_outlier_frac + vy_fit_outlier_frac) / 2

    return v_fit, v_fit_err, v_fit_count, v_fit_outlier_frac

@nb.jit(nopython=True, parallel=True)
def climatology_magnitude(
    vx0,
    vy0,
    dvx_dt,
    dvy_dt,
    vx_amp,
    vy_amp,
    vx_amp_err,
    vy_amp_err,
    vx_phase,
    vy_phase,
    vx_se,
    vy_se
):
    """
    Computes and populates the mean, trend, seasonal amplitude, error in seasonal amplitude,
    and seasonal phase from component values projected on the unit flow  vector defined by vx0 and vy0

    Input:
    ======
    vx0: mean flow in x direction
    vy0: mean flow in y direction
    dvx_dt: trend in flow in x direction
    dvy_dt: trend in flow in y direction
    vx_amp: seasonal amplitude in x direction
    vy_amp: seasonal amplitude in y direction
    vx_amp_err: error in seasonal amplitude in x direction
    vy_amp_err: error in seasonal amplitude in y direction
    vx_phase: seasonal phase in x direction [day of maximum flow]
    vy_phase: seasonal phase in y direction [day of maximum flow]
    vx_trend:
    vy_trend:

    Correlation to actual inputs:
    =============================
    self.offset.vx[start_y:stop_y, start_x:stop_x],
    self.offset.vy[start_y:stop_y, start_x:stop_x],
    self.slope.vx[start_y:stop_y, start_x:stop_x],
    self.slope.vy[start_y:stop_y, start_x:stop_x],
    self.amplitude.vx[start_y:stop_y, start_x:stop_x],
    self.amplitude.vy[start_y:stop_y, start_x:stop_x],
    self.sigma.vx[start_y:stop_y, start_x:stop_x],
    self.sigma.vy[start_y:stop_y, start_x:stop_x],
    self.phase.vx[start_y:stop_y, start_x:stop_x],
    self.phase.vy[start_y:stop_y, start_x:stop_x],
    self.trend.vx[start_y:stop_y, start_x:stop_x],
    self.trend.vy[start_y:stop_y, start_x:stop_x]

    Output:
    =======
    v
    dv_dt
    v_amp
    v_amp_err
    v_phase
    v_trend

    Correlation to actual outputs:
    =============================
    self.offset.v[start_y:stop_y, start_x:stop_x]
    self.slope.v[start_y:stop_y, start_x:stop_x]
    self.amplitude.v[start_y:stop_y, start_x:stop_x]
    self.sigma.v[start_y:stop_y, start_x:stop_x]
    self.phase.v[start_y:stop_y, start_x:stop_x]
    self.trend.v[start_y:stop_y, start_x:stop_x]
    """
    _two_pi = np.pi * 2

    # solve for velcity magnitude and acceleration
    # [do this using vx and vy as to not bias the result due to the Rician distribution of v]
    v = np.sqrt(vx0**2 + vy0**2) # velocity magnitude
    uv_x = vx0/v # unit flow vector in x direction
    uv_y = vy0/v # unit flow vector in y direction

    dv_dt = dvx_dt * uv_x # flow acceleration in direction of unit flow vector
    dv_dt += dvy_dt * uv_y

    v_amp_err = np.abs(vx_amp_err) * np.abs(uv_x) # flow acceleration in direction of unit flow vector, take absolute values
    v_amp_err += np.abs(vy_amp_err) * np.abs(uv_y)

    # solve for amplitude and phase in unit flow direction
    t0 = np.arange(0, 1+0.1, 0.1)

    # Design matrix for LSQ fit
    D = np.stack((np.cos(t0 * _two_pi), np.sin(t0 * _two_pi)), axis=-1)
    # logging.info(f'D: {D}')

    # Step through all spacial points
    y_len, x_len = vx_amp.shape

    v_se = np.full_like(vx_se, np.nan)
    v_se = vx_se * np.abs(uv_x)
    v_se += vy_se * np.abs(uv_y)

    v_amp = np.full_like(vx_amp, np.nan)
    v_phase = np.full_like(vx_phase, np.nan)

    # for j in range(0, y_len):
    #   for i in range(0, x_len):
    for j in nb.prange(y_len):
        for i in nb.prange(x_len):
            # Skip [y, x] point if unit vector value is nan
            if np.isnan(uv_x[j, i]) or np.isnan(uv_y[j, i]):
                continue

            vx_sin = vx_amp[j, i] * np.sin((t0 + (-vx_phase[j, i]/365.25 + 0.25))*_two_pi)  # must convert phase to fraction of a year and adjust from peak to phase
            vy_sin = vy_amp[j, i] * np.sin((t0 + (-vy_phase[j, i]/365.25 + 0.25))*_two_pi)  # must convert phase to fraction of a year and adjust from peak to phase

            v_sin  = vx_sin * uv_x[j, i]  # seasonality in direction of unit flow vector
            v_sin += vy_sin * uv_y[j, i]

            # Solve for coefficients of each column:
            a1, a2 = np.linalg.lstsq(D, v_sin)[0]

            v_amp[j, i] = np.hypot(a1, a2) # amplitude of sinusoid from trig identity a*sin(t) + b*cos(t) = d*sin(t+phi), where d=hypot(a,b) and phi=atan2(b,a).
            phase_rad = np.arctan2(a1, a2) # phase in radians
            v_phase[j, i] = 365.25*((0.25 - phase_rad/_two_pi) % 1) # phase converted such that it reflects the day when value is maximized

    return v, dv_dt, v_amp, v_amp_err, v_phase, v_se

def weighted_linear_fit(t, v, v_err, datetime0=datetime.datetime(2017, 7, 2)):
    """
    Returns the offset, slope, and error for a weighted linear fit to v with an intercept of datetime0.

   - t: date of input estimates
   - v: estimates
   - v_err: estimate errors
   - datetime0: model intercept
   """
    yr = np.array([decimal_year(datetime.datetime(each, 7, 2)) for each in t])
    yr0 = decimal_year(datetime0)
    yr = yr - yr0

    # weights for velocities:
    w_v = 1 / (v_err**2)

    # create design matrix
    D = np.ones((len(yr), 2))
    D[:, 1] = yr

    # Solve for coefficients of each column in the Vandermonde:
    valid = ~np.isnan(v)
    w_v = w_v[valid]
    D = D[valid, :]

    # Julia: offset, slope = (w_v[valid].*D[valid,:]) \ (w_v[valid].*v[valid]);
    offset, slope = np.linalg.lstsq(w_v.reshape((len(w_v), 1)) * D, w_v*v[valid])[0]
    # offset = p[0]
    # slope = p[1]

    # Julia: error = sqrt(sum(v_err[valid].^2))/(sum(valid)-1)
    error = np.sqrt((v_err[valid]**2).sum())/(valid.sum()-1)

    return offset, slope, error

class CompositeVariable:
    """
    Class to hold values for v, vx and vy components of the variables.
    """
    CONT_IN_X = (2, 0, 1)

    def __init__(self, dims: list, name: str):
        """
        Initialize data variables to hold results.
        """
        self.name = name
        self.v = np.full(dims, np.nan)
        self.vx = np.full(dims, np.nan)
        self.vy = np.full(dims, np.nan)

    def transpose(self, dims=CONT_IN_X):
        """
        dims: a tuple of dimension indices for new data layout, i.e. if original
              dimension indices are [y, x, t], then to get [t, y, x] dimensions,
              tuple has to be (2, 0, 1).

        Traspose data variables to new dimensions.
        This is used to switch from continuous memory layout approach (for
        time dimension calculations) to end result data access ([time, y,x]).
        """
        self.v = self.v.transpose(dims)
        self.vx = self.vx.transpose(dims)
        self.vy = self.vy.transpose(dims)

# Currently processed datacube chunk
Chunk = collections.namedtuple("Chunk", ['start_x', 'stop_x', 'x_len', 'start_y', 'stop_y', 'y_len'])

class MissionSensor:
    """
    Mission and sensor combos that should be grouped during filtering by date_dt.
    Group together:
    Sentinel1: 1A and 1B sensors
    Sentinel2: 2A and 2B sensors
    Landsat8 and Landsat9
    """
    # Tuple to keep mission, sensors and string representation of the grouped
    # mission/sensors information as to be written to the Zarr composites store
    # filter
    MSTuple = collections.namedtuple("MissionSensorTuple", ['mission', 'sensors', 'sensors_label'])

    LANDSAT1 = MSTuple('L', ['8.', '9.'], 'L8_L9')
    # If datacube contains only numeric sensor values (Landsat8 or Landsat9),
    # sensor values are of type float, otherwise sensor values are of string type
    # ---> support both
    LANDSAT2 = MSTuple('L', [8.0, 9.0], 'L8_L9')
    LANDSAT3 = MSTuple('L', ['8.0', '9.0'], 'L8_L9')
    SENTINEL1 = MSTuple('S', ['1A', '1B'], 'S1A_S1B')
    SENTINEL2 = MSTuple('S', ['2A', '2B'], 'S2A_S2B')

    # TODO: update with granules information as new missions granules are added
    GROUPS = {
        '8.':  LANDSAT1,
        '9.':  LANDSAT1,
        8.0:   LANDSAT2,
        9.0:   LANDSAT2,
        '8.0': LANDSAT3,
        '9.0': LANDSAT3,
        '1A':  SENTINEL1,
        '1B':  SENTINEL1,
        '2A':  SENTINEL2,
        '2B':  SENTINEL2
    }

class ITSLiveComposite:
    """
    CLass to build annual composites for ITS_LIVE datacubes.
    """
    VERSION = '1.0'

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

    # S3 store location for the Zarr composite
    S3 = ''
    # URL location of the Zarr composite
    URL = ''

    # Scalar relation between MAD and STD
    MAD_STD_RATIO = 1.4826

    # Systematic error based on level of co-registration
    CO_REGISTRATION_ERROR = {
        0: 100,
        1: 5,
        2: 20
    }

    # Chad: put a governor on v and v_amp: NaN-out any values over 20,000 m/yr
    # for the annual composites.
    V_AMP_LIMIT = 10000

    # Threshold for invalid velocity component value: value must be greater than threshold
    V_LIMIT = 20000

    # Store generic cube metadata as static data as these are the same for the whole cube
    YEARS = None
    DATE_DT = None

    START_DECIMAL_YEAR = None
    STOP_DECIMAL_YEAR  = None
    DECIMAL_DT = None
    M = None

    # Dimensions that correspond to the currently processed datacube chunk
    CHUNK = None
    MID_DATE_LEN = None
    YEARS_LEN = None

    # Number of X and Y coordinates to load from the datacube at any given time,
    # and to process in one "chunk"
    NUM_TO_PROCESS = 100

    # Dimensions order of the data to guarantee continuous memory in time dimension
    # Original data as stored in [time, y, x] dimension order.
    CONT_TIME_ORDER = [1, 2, 0]

    def __init__(self, cube_store: str, s3_bucket: str):
        """
        Initialize composites.
        """
        # Don't need to know skipped granules information for the purpose of composites
        read_skipped_granules_flag = False
        self.s3, self.cube_store_in, self.cube_ds, _ = ITSCube.init_input_store(
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
        # Store "shallow" version of the cube for carrying over some of the metadata
        # when writing composites to the Zarr store
        cube_ds = self.cube_ds[ITSLiveComposite.VARS].sortby(DataVars.ImgPairInfo.DATE_DT)
        self.data = cube_ds[[DataVars.VX, DataVars.VY]]

        # Add systematic error based on level of co-registration
        # Load Dask arrays before being able to modify their values
        logging.info(f"Add systematic error based on level of co-registration...")
        self.vx_error = cube_ds.vx_error.astype(np.float32).values
        self.vy_error = cube_ds.vy_error.astype(np.float32).values

        for value, error in ITSLiveComposite.CO_REGISTRATION_ERROR.items():
            mask = (cube_ds[DataVars.FLAG_STABLE_SHIFT] == value)
            self.vx_error[mask] += error
            self.vy_error[mask] += error

        # Images acquisition times and middle_date of each layer as datetime.datetime objects
        acq_datetime_img1 = [t.astype('M8[ms]').astype('O') for t in cube_ds[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1].values]
        acq_datetime_img2 = [t.astype('M8[ms]').astype('O') for t in cube_ds[DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2].values]

        # Compute decimal year representation for start and end dates of each velocity pair
        ITSLiveComposite.START_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img1])
        ITSLiveComposite.STOP_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img2])
        ITSLiveComposite.DECIMAL_DT = ITSLiveComposite.STOP_DECIMAL_YEAR - ITSLiveComposite.START_DECIMAL_YEAR

        # logging.info('DEBUG: Reading date values from Matlab files')
        # Read Matlab values instead of generating them internally
        # with open('/Users/mliukis/Documents/ITS_LIVE/source/github-mliukis/itslive/src/cubesForAlex/start_dates.txt','r') as fh:
        #     ITSLiveComposite.START_DECIMAL_YEAR = np.array([float(each) for each in fh.readlines()[0].rstrip().split(' ')])
        #
        # with open('/Users/mliukis/Documents/ITS_LIVE/source/github-mliukis/itslive/src/cubesForAlex/end_dates.txt','r') as fh:
        #     ITSLiveComposite.STOP_DECIMAL_YEAR = np.array([float(each) for each in fh.readlines()[0].rstrip().split(' ')])

        # TODO: introduce a method to determine composites granularity.
        #       Right now we are generating annual composites only

        # Define time boundaries of composites
        start_year = int(np.floor(np.min(ITSLiveComposite.START_DECIMAL_YEAR)))
        stop_year = int(np.floor(np.max(ITSLiveComposite.STOP_DECIMAL_YEAR)))

        # Years to generate mosaics for
        ITSLiveComposite.YEARS = np.array(range(start_year, stop_year+1))
        ITSLiveComposite.YEARS_LEN = ITSLiveComposite.YEARS.size
        logging.info(f'Years for composite: {ITSLiveComposite.YEARS.tolist()}')

        # Create M matrix for the cube:
        start_time = timeit.default_timer()
        ITSLiveComposite.M = create_M(
            ITSLiveComposite.YEARS,
            ITSLiveComposite.START_DECIMAL_YEAR,
            ITSLiveComposite.STOP_DECIMAL_YEAR,
            ITSLiveComposite.DECIMAL_DT
        )
        logging.info(f'Computed M (took {timeit.default_timer() - start_time} seconds)')

        # Day separation between images (sorted per cube.sortby() call above)
        ITSLiveComposite.DATE_DT = cube_ds[DataVars.ImgPairInfo.DATE_DT].load()

        # Remember datacube dimensions
        self.cube_sizes = cube_ds.sizes

        ITSLiveComposite.MID_DATE_LEN = self.cube_sizes[Coords.MID_DATE]

        # These data members will be set for each block of data being currently
        # processed ---> have to change the logic if want to parallelize blocks
        x_len = self.cube_sizes[Coords.X]
        y_len = self.cube_sizes[Coords.Y]

        # Allocate memory for composite outputs
        dims = (y_len, x_len, ITSLiveComposite.YEARS_LEN)

        self.error = CompositeVariable(dims, 'error')
        self.count = CompositeVariable(dims, 'count')
        # WAS: self.amplitude = CompositeVariable(dims, 'amplitude')
        # WAS: self.phase = CompositeVariable(dims, 'phase')
        self.mean = CompositeVariable(dims, 'mean')

        dims = (y_len, x_len)
        self.outlier_fraction = np.full(dims, np.nan)
        self.amplitude = CompositeVariable(dims, 'amplitude')
        self.sigma = CompositeVariable(dims, 'sigma')
        self.phase = CompositeVariable(dims, 'phase')
        self.offset = CompositeVariable(dims, 'offset')
        self.slope = CompositeVariable(dims, 'slope')
        self.trend = CompositeVariable(dims, 'trend')

        # Sensor data for the cube's layers
        self.sensors = cube_ds[DataVars.ImgPairInfo.SATELLITE_IMG1].values
        self.missions = cube_ds[DataVars.ImgPairInfo.MISSION_IMG1].values

        # Identify unique sensors within datacube.
        # ATTN: if the same sensor is present for multiple missions,
        # this WON'T WORK - need to identify unique
        # mission and sensor pairs present in the cube
        unique_sensors = list(set(self.sensors))
        # Keep values sorted to be consistent
        unique_sensors.sort()
        logging.info(f'Identified unique sensors: {unique_sensors}')

        # Make sure each identified sensor is listed in the MissionSensor.GROUPS
        for each in unique_sensors:
            if each not in MissionSensor.GROUPS:
                raise RuntimeError(f'Unknown sensor {each} is detected. " \
                    f"Sensor value must be listed in MissionSensor.GROUPS ({MissionSensor.GROUPS}) " \
                    f"to identify the group it belongs to for "date_dt" filtering.')

        # Identify unique sensor groups
        self.sensors_groups = []
        collected_sensors = []
        for each in unique_sensors:
            if each not in collected_sensors:
                self.sensors_groups.append(MissionSensor.GROUPS[each])
                collected_sensors.extend(self.sensors_groups[-1].sensors)

        dims = (y_len, x_len, len(self.sensors_groups))
        self.max_dt = np.full(dims, np.nan)

        # Date when composites were created
        self.date_created = datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        self.date_updated = self.date_created

        # TODO: take care of self.date_updated when support for composites updates
        # is implemented

    def create(self, output_store: str, s3_bucket: str):
        """
        Create datacube composite: cube time mean values.
        """
        # Loop through cube in chunks to minimize memory footprint
        x_start = 0
        x_num_to_process = self.cube_sizes[Coords.X]

        # For debugging only
        # x_start = 200
        # x_num_to_process = self.cube_sizes[Coords.X] - x_start
        # x_num_to_process = 100

        while x_num_to_process > 0:
            # How many tasks to process at a time
            x_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if x_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else x_num_to_process

            y_start = 0
            y_num_to_process = self.cube_sizes[Coords.Y]

            # For debugging only
            # y_start = 300
            # y_num_to_process = self.cube_sizes[Coords.Y] - y_start
            # y_num_to_process = 100

            while y_num_to_process > 0:
                y_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if y_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else y_num_to_process

                self.cube_time_mean(x_start, x_num_tasks, y_start, y_num_tasks)

                y_num_to_process -= y_num_tasks
                y_start += y_num_tasks

            x_num_to_process -= x_num_tasks
            x_start += x_num_tasks

        # Save data to Zarr store
        self.to_zarr(output_store, s3_bucket)

    def cube_time_mean(self, start_x, num_x, start_y, num_y):
        """
        Compute time average for the datacube [:, :, start_x:stop_index] coordinates.
        Update corresponding entries in output data variables.
        """
        # Set current block length for the X and Y dimensions
        stop_y = start_y + num_y
        stop_x = start_x + num_x
        ITSLiveComposite.Chunk = Chunk(start_x, stop_x, num_x, start_y, stop_y, num_y)

        # % convert to singles (Float32) to reduce memory footprint
        # for k = 1:length(vars)
        #     if any(strcmp({'vx','vy','vx_error','vy_error'}, vars{k}))
        #         data.(vars{k}) = single(data.(vars{k}));
        #     end
        # end
        # fprintf('finished [%.1fmin]\n', (now-t0)*24*60)

        # Start timer
        start_time = timeit.default_timer()

        # ----- FILTER DATA -----
        # Filter data based on locations where means of various dts are
        # statistically different and mad deviations from a running meadian
        logging.info(f'Filter data based on dt binned medians...')

        # Initialize variables
        dims = (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN)
        v_invalid = np.full(dims, False)

        # Loop for each unique sensor (those groupings image pairs that can be
        # expected to have different temporal decorelation)

        # ATTN: don't use native xarray functionality is much slower,
        # convert data to numpy types and use numpy only
        logging.info(f'Loading vx[:, {start_y}:{stop_y}, {start_x}:{stop_x}] out of [{self.cube_sizes[Coords.MID_DATE]}, {self.cube_sizes[Coords.Y]}, {self.cube_sizes[Coords.X]}]...')
        vx_org = self.data.vx[:, start_y:stop_y, start_x:stop_x].astype(np.float32).values

        # Transpose data to make it continuous in time
        vx = np.zeros((ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN))
        vx.flat = np.transpose(vx_org, ITSLiveComposite.CONT_TIME_ORDER)

        logging.info(f'Loading vy[:, {start_y}:{stop_y}, {start_x}:{stop_x}] out of [{self.cube_sizes[Coords.MID_DATE]}, {self.cube_sizes[Coords.Y]}, {self.cube_sizes[Coords.X]}]...')
        vy_org = self.data.vy[:, start_y:stop_y, start_x:stop_x].astype(np.float32).values

        vy = np.zeros((ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN))
        vy.flat = np.transpose(vy_org, ITSLiveComposite.CONT_TIME_ORDER)

        for i in range(len(self.sensors_groups)):
            sensor_group = self.sensors_groups[i]
            logging.info(f'Filtering dt for sensors "{sensor_group.sensors}" ({i+1} out ' \
                f'of {len(self.sensors_groups)} sensor groups)')

            # Find which layers correspond to each sensor group
            mask = np.zeros((ITSLiveComposite.MID_DATE_LEN), dtype=np.bool_)

            for each in sensor_group.sensors:
                # logging.info(f'Update mask with {each} as part of the sensor group')
                mask |= (self.sensors == each)

            # Filter current block's variables
            # TODO: Don't drop variables when masking - won't work on return assignment
            #       for cubes with multiple sensors
            dt_masked = ITSLiveComposite.DATE_DT.values[mask]

            logging.info(f'Filtering projected v onto median flow vector...')
            start_time = timeit.default_timer()
            v_invalid[:, :, mask], self.max_dt[start_y:stop_y, start_x:stop_x, i] = cube_filter(vx[..., mask], vy[..., mask], dt_masked, ITSLiveComposite.MAD_STD_RATIO)
            logging.info(f'Filtered projected v (took {timeit.default_timer() - start_time} seconds)')

        # Load data to avoid NotImplemented exception when invoked on Dask arrays
        logging.info(f'Compute invalid mask...')
        start_time = timeit.default_timer()

        invalid = v_invalid | (np.hypot(vx, vy) > ITSLiveComposite.V_LIMIT)

        # Mask data
        vx[invalid] = np.nan
        vy[invalid] = np.nan

        invalid = np.nansum(invalid, axis=2)
        invalid = np.divide(invalid, np.sum(np.isnan(vx), 2) + invalid)

        logging.info(f'Finished filtering with invalid mask ({timeit.default_timer() - start_time} seconds)')

        # %% Least-squares fits to detemine amplitude, phase and annual means
        logging.info(f'Find vx annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        # Transform vx data to make time series continuous in memory: [y, x, t]
        vx_outlier = ITSLiveComposite.cubelsqfit2(
            vx,
            self.vx_error,
            self.amplitude.vx,
            self.phase.vx,
            self.mean.vx,
            self.error.vx,
            self.sigma.vx,
            self.count.vx,
            self.offset.vx,
            self.slope.vx,
            self.trend.vx
        )
        logging.info(f'Finished vx LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find vy annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        vy_outlier = ITSLiveComposite.cubelsqfit2(
            vy,
            self.vy_error,
            self.amplitude.vy,
            self.phase.vy,
            self.mean.vy,
            self.error.vy,
            self.sigma.vy,
            self.count.vy,
            self.offset.vy,
            self.slope.vy,
            self.trend.vy
        )
        logging.info(f'Finished vy LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find annual magnitude... ')
        start_time = timeit.default_timer()

        self.mean.v[start_y:stop_y, start_x:stop_x, :], \
        self.error.v[start_y:stop_y, start_x:stop_x, :], \
        self.count.v[start_y:stop_y, start_x:stop_x, :], \
        voutlier = \
        annual_magnitude(
            self.offset.vx[start_y:stop_y, start_x:stop_x],
            self.offset.vy[start_y:stop_y, start_x:stop_x],
            self.mean.vx[start_y:stop_y, start_x:stop_x, :],
            self.mean.vy[start_y:stop_y, start_x:stop_x, :],
            self.error.vx[start_y:stop_y, start_x:stop_x, :],
            self.error.vy[start_y:stop_y, start_x:stop_x, :],
            self.count.vx[start_y:stop_y, start_x:stop_x, :],
            self.count.vy[start_y:stop_y, start_x:stop_x, :],
            vx_outlier,
            vy_outlier,
        )
        logging.info(f'Finished annual magnitude (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find climatology magnitude...')
        start_time = timeit.default_timer()

        self.offset.v[start_y:stop_y, start_x:stop_x], \
        self.slope.v[start_y:stop_y, start_x:stop_x], \
        self.amplitude.v[start_y:stop_y, start_x:stop_x], \
        self.sigma.v[start_y:stop_y, start_x:stop_x], \
        self.phase.v[start_y:stop_y, start_x:stop_x], \
        self.trend.v[start_y:stop_y, start_x:stop_x] = \
        climatology_magnitude(
            self.offset.vx[start_y:stop_y, start_x:stop_x],
            self.offset.vy[start_y:stop_y, start_x:stop_x],
            self.slope.vx[start_y:stop_y, start_x:stop_x],
            self.slope.vy[start_y:stop_y, start_x:stop_x],
            self.amplitude.vx[start_y:stop_y, start_x:stop_x],
            self.amplitude.vy[start_y:stop_y, start_x:stop_x],
            self.sigma.vx[start_y:stop_y, start_x:stop_x],
            self.sigma.vy[start_y:stop_y, start_x:stop_x],
            self.phase.vx[start_y:stop_y, start_x:stop_x],
            self.phase.vy[start_y:stop_y, start_x:stop_x],
            self.trend.vx[start_y:stop_y, start_x:stop_x],
            self.trend.vy[start_y:stop_y, start_x:stop_x]
        )
        logging.info(f'Finished climatology magnitude (took {timeit.default_timer() - start_time} seconds)')

        # Nan out invalid values
        # WAS: invalid_mask = (self.mean.v > ITSLiveComposite.V_LIMIT) | (self.amplitude.v > ITSLiveComposite.V_AMP_LIMIT)
        invalid_mask = (self.mean.v > ITSLiveComposite.V_LIMIT)
        self.mean.v[invalid_mask] = np.nan

        invalid_mask = (self.amplitude.v > ITSLiveComposite.V_AMP_LIMIT)
        self.amplitude.v[invalid_mask] = np.nan

        # outlier = invalid + voutlier*(1-invalid)
        self.outlier_fraction[start_y:stop_y, start_x:stop_x] = invalid + voutlier*(1-invalid)

    def to_zarr(self, output_store: str, s3_bucket: str):
        """
        Store datacube annual composite to the Zarr store.
        """
        logging.info(f'Writing composites to {output_store}')

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

        # Convert years to datetime objects to represent the center of calendar year
        ITSLiveComposite.YEARS = [datetime.datetime(each, 7, 2) for each in ITSLiveComposite.YEARS]
        logging.info(f"Converted years to datetime objs: {ITSLiveComposite.YEARS}")

        # Create list of sensors groups labels
        sensors_labels = [each.sensors_label for each in self.sensors_groups]
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.cube_ds.x.values,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.X],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.X],
                        DataVars.UNITS: DataVars.M_UNITS
                    }
                ),
                Coords.Y: (
                    Coords.Y,
                    self.cube_ds.y.values,
                    {
                        DataVars.STD_NAME: Coords.STD_NAME[Coords.Y],
                        DataVars.DESCRIPTION_ATTR: Coords.DESCRIPTION[Coords.Y],
                        DataVars.UNITS: DataVars.M_UNITS
                    }
                ),
                CompDataVars.TIME: (
                    CompDataVars.TIME,
                    ITSLiveComposite.YEARS,
                    {
                        DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.TIME],
                        DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.TIME]
                    }
                ),
                CompDataVars.SENSORS: (
                    CompDataVars.SENSORS,
                    sensors_labels,
                    {
                        DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSORS],
                        DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSORS]
                    }
                )
            },
            attrs = {
                'author': 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)'
            }
        )

        ds.attrs['composites_software_version'] = ITSLiveComposite.VERSION
        ds.attrs['date_created'] = self.date_created
        ds.attrs['date_updated'] = self.date_updated

        # To support old format datacubes for testing
        # TODO: remove once done testing with old cubes (to compare to Matlab)
        if 's3' in self.cube_ds.attrs:
            ds.attrs['datecube_s3'] = self.cube_ds.attrs['s3']
            ds.attrs['datecube_url'] = self.cube_ds.attrs['url']

        ds.attrs['datecube_created'] = self.cube_ds.attrs['date_created']
        ds.attrs['datecube_updated'] = self.cube_ds.attrs['date_updated']
        ds.attrs['datacube_software_version'] = self.cube_ds.attrs['datacube_software_version']
        ds.attrs['datacube_autoRIFT_parameter_file'] = self.cube_ds.attrs['autoRIFT_parameter_file']

        ds.attrs['GDAL_AREA_OR_POINT'] = 'Area'

        # To support old format datacubes for testing
        # TODO: remove once done testing with old cubes (to compare to Matlab)
        if 'geo_polygon' in self.cube_ds.attrs:
            ds.attrs['geo_polygon']  = self.cube_ds.attrs['geo_polygon']
            ds.attrs['proj_polygon'] = self.cube_ds.attrs['proj_polygon']

        ds.attrs['institution'] = 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology'
        ds.attrs['latitude']  = self.cube_ds.attrs['latitude']
        ds.attrs['longitude'] = self.cube_ds.attrs['longitude']
        # ds.attrs['proj_polygon'] = self.cube_ds.attrs['proj_polygon']
        ds.attrs['projection'] = self.cube_ds.attrs['projection']
        ds.attrs['s3'] = ITSLiveComposite.S3
        ds.attrs['url'] = ITSLiveComposite.URL
        ds.attrs['institution'] = 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology'
        ds.attrs['title'] = 'ITS_LIVE annual composites of image_pair velocities'

        # Add data as variables
        ds[DataVars.MAPPING] = self.cube_ds[DataVars.MAPPING]

        years_coord = pd.Index(ITSLiveComposite.YEARS, name=CompDataVars.TIME)
        var_coords = [years_coord, self.cube_ds.y.values, self.cube_ds.x.values]
        var_dims = [CompDataVars.TIME, Coords.Y, Coords.X]

        twodim_var_coords = [self.cube_ds.y.values, self.cube_ds.x.values]
        twodim_var_dims = [Coords.Y, Coords.X]

        self.mean.transpose()
        self.error.transpose()
        self.count.transpose()

        ds[DataVars.V] = xr.DataArray(
            data=self.mean.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[DataVars.V],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[DataVars.V],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.mean.v = None
        gc.collect()

        ds[CompDataVars.V_ERROR] = xr.DataArray(
            data=self.error.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.error.v = None
        gc.collect()

        ds[DataVars.VX] = xr.DataArray(
            data=self.mean.vx,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[DataVars.VX],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[DataVars.VX],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.mean.vx = None
        gc.collect()

        ds[CompDataVars.VX_ERROR] = xr.DataArray(
            data=self.error.vx,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.error.vx = None
        gc.collect()

        ds[DataVars.VY] = xr.DataArray(
            data=self.mean.vy,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[DataVars.VY],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[DataVars.VY],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.mean.vy = None
        gc.collect()

        ds[CompDataVars.VY_ERROR] = xr.DataArray(
            data=self.error.vy,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.error.vy = None
        gc.collect()

        ds[CompDataVars.V_AMP] = xr.DataArray(
            data=self.amplitude.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V_AMP],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V_AMP],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.amplitude.v = None
        gc.collect()

        ds[CompDataVars.V_AMP_ERROR] = xr.DataArray(
            data=self.sigma.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V_AMP_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V_AMP_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.sigma.v = None
        gc.collect()

        ds[CompDataVars.V_PHASE] = xr.DataArray(
            data=self.phase.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V_PHASE],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V_PHASE],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.DAY_OF_YEAR_UNITS
            }
        )
        self.phase.v = None
        gc.collect()

        ds[CompDataVars.VX_AMP] = xr.DataArray(
            data=self.amplitude.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX_AMP],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX_AMP],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.amplitude.vx = None
        gc.collect()

        ds[CompDataVars.VX_AMP_ERROR] = xr.DataArray(
            data=self.sigma.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX_AMP_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX_AMP_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.sigma.vx = None
        gc.collect()

        ds[CompDataVars.VX_PHASE] = xr.DataArray(
            data=self.phase.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX_PHASE],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX_PHASE],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.DAY_OF_YEAR_UNITS
            }
        )
        self.phase.vx = None
        gc.collect()

        ds[CompDataVars.VY_AMP] = xr.DataArray(
            data=self.amplitude.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY_AMP],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY_AMP],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.amplitude.vy = None
        gc.collect()

        ds[CompDataVars.VY_AMP_ERROR] = xr.DataArray(
            data=self.sigma.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY_AMP_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY_AMP_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.sigma.vy = None
        gc.collect()

        ds[CompDataVars.VY_PHASE] = xr.DataArray(
            data=self.phase.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY_PHASE],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY_PHASE],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.DAY_OF_YEAR_UNITS
            }
        )
        self.phase.vy = None
        gc.collect()

        count0 = self.count.v.sum(axis=0)

        ds[CompDataVars.COUNT] = xr.DataArray(
            data=self.count.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.COUNT],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.COUNT],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.COUNT_UNITS
            }
        )
        self.count.v = None
        gc.collect()

        # Add max_dt (per sensor)
        # Use "group" label for each of the sensors used to filter data
        sensor_coord = pd.Index(sensors_labels, name=CompDataVars.SENSORS)
        var_coords = [sensor_coord, self.cube_ds.y.values, self.cube_ds.x.values]
        var_dims = [CompDataVars.SENSORS, Coords.Y, Coords.X]

        self.max_dt = self.max_dt.transpose(CompositeVariable.CONT_IN_X)

        ds[CompDataVars.MAX_DT] = xr.DataArray(
            data=self.max_dt,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.MAX_DT],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.MAX_DT],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.ImgPairInfo.UNITS[DataVars.ImgPairInfo.DATE_DT]
            }
        )
        self.max_dt = None
        gc.collect()

        ds[CompDataVars.OUTLIER_FRAC] = xr.DataArray(
            data=self.outlier_fraction,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.OUTLIER_FRAC],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.OUTLIER_FRAC],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.FRACTION_UNITS
            }
        )
        self.outlier_fraction = None
        gc.collect()

        ds[CompDataVars.VX0] = xr.DataArray(
            data=self.offset.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX0],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX0],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.offset.vx = None
        gc.collect()

        ds[CompDataVars.VY0] = xr.DataArray(
            data=self.offset.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY0],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY0],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.offset.vy = None
        gc.collect()

        ds[CompDataVars.V0] = xr.DataArray(
            data=self.offset.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V0],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V0],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.offset.v = None
        gc.collect()

        ds[CompDataVars.VX0_ERROR] = xr.DataArray(
            data=self.trend.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.trend.vx = None
        gc.collect()

        ds[CompDataVars.VY0_ERROR] = xr.DataArray(
            data=self.trend.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.trend.vy = None
        gc.collect()

        ds[CompDataVars.V0_ERROR] = xr.DataArray(
            data=self.trend.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.trend.v = None
        gc.collect()

        ds[CompDataVars.SLOPE_V] = xr.DataArray(
            data=self.slope.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SLOPE_V],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SLOPE_V],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y2_UNITS
            }
        )
        self.slope.v = None
        gc.collect()

        ds[CompDataVars.SLOPE_VX] = xr.DataArray(
            data=self.slope.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SLOPE_VX],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SLOPE_VX],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y2_UNITS
            }
        )
        self.slope.vx = None
        gc.collect()

        ds[CompDataVars.SLOPE_VY] = xr.DataArray(
            data=self.slope.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SLOPE_VY],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SLOPE_VY],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y2_UNITS
            }
        )
        self.slope.vy = None
        gc.collect()

        ds[CompDataVars.COUNT0] = xr.DataArray(
            data=count0,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.COUNT0],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.COUNT0],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.COUNT_UNITS
            }
        )
        count0 = None
        gc.collect()

        # ATTN: Set attributes for the Dataset coordinates as the very last step:
        # when adding data variables that don't have the same attributes for the
        # coordinates, originally set Dataset coordinates will be wiped out
        # (xarray bug?)
        ds[Coords.X].attrs = X_ATTRS
        ds[Coords.Y].attrs = Y_ATTRS
        ds[CompDataVars.TIME].attrs = TIME_ATTRS
        ds[CompDataVars.SENSORS].attrs = SENSORS_ATTRS

        # Set encoding
        encoding_settings = {}
        encoding_settings.setdefault(CompDataVars.TIME, {}).update({DataVars.UNITS: DataVars.ImgPairInfo.DATE_UNITS})

        for each in [CompDataVars.TIME, CompDataVars.SENSORS, Coords.X, Coords.Y]:
            encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

        encoding_settings.setdefault(CompDataVars.SENSORS, {}).update({'dtype': 'str'})

        # Compression for the data
        compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)

        # Settings for "float" data types
        for each in [
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR,
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
            ]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                # 'dtype': 'float',
                'dtype': np.float,
                'compressor': compressor
            })

        # Settings for "short" datatypes
        for each in [CompDataVars.COUNT, CompDataVars.MAX_DT]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                'dtype': 'short'
            })

        # Chunking to apply when writing datacube to the Zarr store
        chunks_settings = (1, self.cube_sizes[Coords.Y], self.cube_sizes[Coords.X])

        for each in [
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR,
            CompDataVars.VX_AMP_ERROR,
            CompDataVars.VY_AMP_ERROR,
            CompDataVars.V_AMP_ERROR
        ]:
            encoding_settings[each].update({
                'chunks': chunks_settings
            })

        # Chunking to apply when writing datacube to the Zarr store
        chunks_settings = (self.cube_sizes[Coords.Y], self.cube_sizes[Coords.X])

        for each in [
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
            ]:
            encoding_settings[each].update({
                'chunks': chunks_settings
            })

        logging.info(f"Encoding settings: {encoding_settings}")
        ds.to_zarr(output_store, encoding=encoding_settings, consolidated=True)

    @staticmethod
    def cubelsqfit2(
        v,
        v_err_data,
        amplitude,
        phase,
        mean,
        error,
        sigma,
        count,
        offset,
        slope,
        se
    ):
        """
        Cube LSQ fit with 2 iterations.

        Populate: [amp, phase, mean, err, sigma, cnt]

        Return: outlier_frac
        """
        # Minimum number of non-NAN values in the data to proceed with LSQ fit
        _num_valid_points = 5

        # Initialize output
        start_time = timeit.default_timer()
        dims = (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len)
        outlier_frac = np.full(dims, np.nan)

        # This is only done for generic parfor "slicing" may not be needed when
        # recoded
        v_err = v_err_data
        if v_err_data.ndim != v.ndim:
            # Expand vector to 3-d array
            reshape_v_err = v_err_data.reshape((1,1,v_err_data.size))
            # v_err = np.tile(reshape_v_err, (1, ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len))
            v_err = np.broadcast_to(reshape_v_err, (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, v_err_data.size))

        init_time1 = 0
        init_time2 = 0
        init_time3 = 0
        lsq_time = 0

        # for j in tqdm(range(0, 1), ascii=True, desc='cubelsqfit2: y (debug)'):
        for j in tqdm(range(0, ITSLiveComposite.Chunk.y_len), ascii=True, desc='cubelsqfit2: y'):
            for i in range(0, ITSLiveComposite.Chunk.x_len):
                # mask = ~np.isnan(v[:, j, i])
                mask = ~np.isnan(v[j, i, :])
                if mask.sum() < _num_valid_points:
                    # Skip the point, return no outliers
                    continue

                global_i = i + ITSLiveComposite.Chunk.start_x
                global_j = j + ITSLiveComposite.Chunk.start_y

                amplitude[global_j, global_i], \
                sigma[global_j, global_i], \
                phase[global_j, global_i], \
                offset[global_j, global_i], \
                slope[global_j, global_i], \
                se[global_j, global_i], \
                outlier_frac[j, i], \
                init_runtime1, \
                init_runtime2, \
                init_runtime3, \
                lsq_runtime = itslive_lsqfit_annual(
                    v[j, i, :],
                    v_err[j, i, :],
                    ITSLiveComposite.START_DECIMAL_YEAR,
                    ITSLiveComposite.STOP_DECIMAL_YEAR,
                    ITSLiveComposite.DECIMAL_DT,
                    ITSLiveComposite.YEARS,
                    ITSLiveComposite.M,
                    ITSLiveComposite.MAD_STD_RATIO,
                    mean[global_j, global_i, :],
                    error[global_j, global_i, :],
                    count[global_j, global_i, :]
                )
                init_time1 += init_runtime1
                init_time2 += init_runtime2
                init_time3 += init_runtime3
                lsq_time += lsq_runtime

        logging.info(f'Init_time1: {init_time1} sec, Init_time2: {init_time2} sec, Init_time3: {init_time3} sec, lsq_time: {lsq_time} seconds')
        return outlier_frac


if __name__ == '__main__':
    import argparse
    import warnings
    import shutil
    import subprocess
    import sys
    from urllib.parse import urlparse


    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=ITSLiveComposite.__doc__.split('\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-c', '--chunkSize',
        type=int,
        default=100,
        help='Number of X and Y coordinates to process in parallel with Dask. ' \
             'This should be multiples of the size of chunking used within the cube to optimize data reads [%(default)d].'
    )
    parser.add_argument(
        '-i', '--inputCube',
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
        '-b', '--inputBucket',
        type=str,
        default='',
        help="S3 bucket with input datacube Zarr store [%(default)s]."
    )
    parser.add_argument(
        '-t', '--targetBucket',
        type=str,
        default='',
        help="S3 bucket to store cube composite in Zarr format to [%(default)s]."
    )
    args = parser.parse_args()

    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Command arguments: {args}")

    # Set static data for computation
    ITSLiveComposite.NUM_TO_PROCESS = args.chunkSize

    if len(args.targetBucket):
        ITSLiveComposite.S3 = os.path.join(args.targetBucket, args.outputStore)
        logging.info(f'Composite S3: {ITSLiveComposite.S3}')

        # URL is valid only if output S3 bucket is provided
        ITSLiveComposite.URL = ITSLiveComposite.S3.replace(ITSCube.S3_PREFIX, ITSCube.HTTP_PREFIX)
        url_tokens = urlparse(ITSLiveComposite.URL)
        ITSLiveComposite.URL = url_tokens._replace(netloc=url_tokens.netloc+ITSCube.PATH_URL).geturl()
        logging.info(f'Composite URL: {ITSLiveComposite.URL}')

    mosaics = ITSLiveComposite(args.inputCube, args.inputBucket)
    mosaics.create(args.outputStore, args.targetBucket)

    # Copy generated composites to the S3 bucket if provided
    if os.path.exists(args.outputStore) and len(args.targetBucket):
        try:
            # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
            # resulting in as many error messages as there are files in Zarr store
            # to copy
            command_line = [
                "awsv2", "s3", "cp", "--recursive",
                args.outputStore,
                os.path.join(args.targetBucket, os.path.basename(args.outputStore)),
                "--acl", "bucket-owner-full-control"
            ]

            logging.info(' '.join(command_line))

            file_is_copied = False
            num_retries = 0
            command_return = None
            env_copy = os.environ.copy()

            while not file_is_copied and num_retries < ITSCube.NUM_AWS_COPY_RETRIES:
                logging.info(f"Attempt #{num_retries+1} to copy {args.outputStore} to {args.targetBucket}")

                command_return = subprocess.run(
                    command_line,
                    env=env_copy,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

                if command_return.returncode != 0:
                    # Report the whole stdout stream as one logging message
                    logging.warning(f"Failed to copy {args.outputStore} to {args.targetBucket} with returncode={command_return.returncode}: {command_return.stdout}")

                    num_retries += 1
                    # If failed due to AWS SlowDown error, retry
                    if num_retries != ITSCube.NUM_AWS_COPY_RETRIES and \
                       ITSCube.AWS_SLOW_DOWN_ERROR in command_return.stdout.decode('utf-8'):
                        # Sleep if it's not a last attempt to copy
                        time.sleep(ITSCube.AWS_COPY_SLEEP_SECONDS)

                    else:
                        # Don't retry otherwise
                        num_retries = ITSCube.NUM_AWS_COPY_RETRIES

                else:
                    file_is_copied = True

            if not file_is_copied:
                raise RuntimeError(f"Failed to copy {args.outputStore} to {args.targetBucket} with command.returncode={command_return.returncode}")

        finally:
            # Remove locally written Zarr store.
            # This is to eliminate out of disk space failures when the same EC2 instance is
            # being re-used by muliple Batch jobs.
            if os.path.exists(args.outputStore):
                logging.info(f"Removing local copy of {args.outputStore}")
                shutil.rmtree(args.outputStore)

    logging.info("Done.")
