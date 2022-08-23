"""
ITSLiveComposite class creates yearly composites of ITS_LIVE datacubes with data
within the same target projection, bounding polygon and datetime period as
specified at the time the datacube was constructed/updated.

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Mark Fahnestock (UAF)

Jet Propulsion Laboratory, California Institute of Technology, Pasadena, California
March 21, 2022
"""
import collections
import copy
import datetime
import gc
import json
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
    SENSOR_INCLUDE = 'sensor_flag'
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
        SENSOR_INCLUDE: 'sensor_flag',
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
        DataVars.VX:    'mean annual velocity of sinusoidal fit to vx',
        DataVars.VY:    'mean annual velocity of sinusoidal fit to vy',
        DataVars.V:     'mean annual velocity of sinusoidal fit to v',
        TIME:           'time',
        VX_ERROR:       'error weighted error for vx',
        VY_ERROR:       'error weighted error for vy',
        V_ERROR:        'error weighted error for v',
        VX_AMP_ERROR:   'error for vx_amp',
        VY_AMP_ERROR:   'error for vy_amp',
        V_AMP_ERROR:    'error for v_amp',
        VX_AMP:         'climatological mean seasonal amplitude of sinusoidal fit to vx',
        VY_AMP:         'climatological mean seasonal amplitude in sinusoidal fit in vy',
        V_AMP:          'climatological mean seasonal amplitude of sinusoidal fit to v',
        VX_PHASE:       'day of maximum velocity of sinusoidal fit to vx',
        VY_PHASE:       'day of maximum velocity of sinusoidal fit to vy',
        V_PHASE:        'day of maximum velocity of sinusoidal fit to v',
        COUNT:          'number of image pairs used in error weighted least squares fit',
        MAX_DT:         'maximum allowable time separation between image pair acquisitions included in error weighted least squares fit',
        SENSOR_INCLUDE: 'flag = 1 if sensor group (see sensor variable) is included, flag = 0 if sensor group is excluded',
        OUTLIER_FRAC:   'fraction of data identified as outliers and excluded from error weighted least squares fit',
        SENSORS:        'combinations of unique sensors and missions that are grouped together for date_dt filtering',
        VX0:            'climatological vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values. The climatology is arbitrarily fixed to a y-intercept of July 2, 2019.',
        VY0:            'climatological vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values. The climatology is arbitrarily fixed to a y-intercept of July 2, 2019.',
        V0:             'climatological v determined by taking the hypotenuse of vx0 and vy0. The climatology is arbitrarily fixed to a y-intercept of July 2, 2019.',
        COUNT0:         'number of image pairs used for climatological means',
        VX0_ERROR:      'error for vx0',
        VY0_ERROR:      'error for vy0',
        V0_ERROR:       'error for v0',
        SLOPE_VX:       'trend in vx determined by a weighted least squares line fit, described by an offset and slope, to mean annual vx values',
        SLOPE_VY:       'trend in vy determined by a weighted least squares line fit, described by an offset and slope, to mean annual vy values',
        SLOPE_V:        'trend in v determined by projecting dvx_dt and dvy_dt onto the unit flow vector defined by vx0 and vy0'
    }

class CompOutputFormat:
    """
    Class to represent attributes for the output format of the data.
    """
    GDAL_AREA_OR_POINT = 'GDAL_AREA_OR_POINT'
    COMPOSITES_SOFTWARE_VERSION = 'composites_software_version'
    DATACUBE_AUTORIFT_PARAMETER_FILE = 'datacube_autoRIFT_parameter_file'
    DATACUBE_SOFTWARE_VERSION = 'datacube_software_version'
    DATECUBE_CREATED = 'datecube_created'
    DATECUBE_S3 = 'datecube_s3'
    DATECUBE_UPDATED = 'datecube_updated'
    DATECUBE_URL = 'datecube_url'
    PROJECTION = 'projection'
    PROJ_POLYGON = 'proj_polygon'
    GEO_POLYGON = 'geo_polygon'
    SENSORS_LABELS = 'sensors_labels'

    # Composites specific attributes
    DATE_CREATED = 'date_created'
    DATE_UPDATED = 'date_updated'
    S3 = 's3'
    URL = 'url'

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

@nb.jit(nopython=True)
def create_projected_velocity(x_in, y_in, dt):
    """
    Project vx and vy onto the median flow vector for the given spacial point.

    Inputs:
    =======
    x_in: x component of the velocity vector.
    y_in: y component of the velocity vector.
    dt:   Day separation vector.

    Return:
    =======
    Projected velocity onto median flow vector.
    """
    # Bin edges to use for the median flow vector
    _dt_median_flow = np.array([16, 32, 64, 128, 256, np.inf])

    # Minimum number of points for dt_median_flow bin
    _min_ref_unit_count = 50

    # Skip dt filter for slow moving areas
    _min_v0_threshold = 50

    x0_in = np.full_like(x_in, np.nan)

    x0_is_null = np.isnan(x_in)
    if np.all(x0_is_null):
        # No data to process
        return x0_in

    # Project vx and vy onto the median flow vector for dt <= 16;
    # if there is no data, then for dt <= 32, etc.
    ind = None
    valid = ~x0_is_null # Number of valid points

    for each_dt in _dt_median_flow:
        ind = (dt <= each_dt) & valid

        # Are there enough points?
        if ind.sum() >= _min_ref_unit_count:
            break

    if ind.sum() == 0:
        # No data to process
        return x0_in

    # Make numba happy
    ind = ind.astype(np.bool_)

    vx0 = np.median(x_in[ind])
    vy0 = np.median(y_in[ind])
    v0 = np.sqrt(vx0**2 + vy0**2)

    if v0 <= _min_v0_threshold:
        # maxdt should be set to np.inf
        x0_in = np.full_like(x_in, np.inf)
        return x0_in

    uv_x = vx0/v0 # unit flow vector
    uv_y = vy0/v0
    x0_in = x_in*uv_x + y_in*uv_y # projected flow vectors

    return x0_in

# @nb.jit(nopython=True)
def cube_filter_iteration(vp, dt, mad_std_ratio):
    """
    Filter one spacial point by dt (date separation) between the images.

    Inputs:
    =======
    vp_in: Projected velocity to median flow unit vector.
    dt_in: Day separation vector.

    Return: a tuple of
    maxdt:   Maximum dt as determined by the filter.
    invalid: Mask for invalid values of the input vector based on maxdt.
    """
    # Filter parameters for dt bins:
    # used to determine if dt means are significantly different
    _dtbin_mad_thresh = 0.67

    _dtbin_ratio = _dtbin_mad_thresh * mad_std_ratio

    _dt_edge = np.array([0, 16, 32, 64, 128, 256, np.inf])
    _num_bins = len(_dt_edge)-1

    # Minumum number of points for the reference bin
    _min_ref_bin_count = 50

    # Output data variables
    maxdt = np.nan

    # Make numba happy - use np.bool_ type
    invalid = np.zeros_like(dt, dtype=np.bool_)

    # There is no valid projected velocity vector
    if np.all(np.isnan(vp)):
        return (np.nan, invalid)

    if np.any(np.isinf(vp)):
        return (np.inf, invalid)

    x0_is_null = np.isnan(vp)
    if np.all(x0_is_null):
        # No data to process
        return (maxdt, invalid)

    mask = ~x0_is_null
    x0 = vp[mask]
    x0_dt = dt[mask]

    # Group data values by identified bins "manually":
    # since data is sorted by date_dt, we can identify index boundaries
    # for each bin within the "date_dt" vector
    bin_index = np.searchsorted(x0_dt, _dt_edge)

    # Don't know ahead of time how many valid (start != end) bins will be collected,
    # so don't pre-allocate lists
    xmed = []
    xmad = []
    count = []

    # Collect indices for bins that represent current x0_dt
    dt_bin_indices = []

    for bin_i in range(0, _num_bins):
        # if bin_index[bin_i] and bin_index[bin_i+1] are the same, there are no values for the bin, skip it
        if bin_index[bin_i] != bin_index[bin_i+1]:
            bin_xmed, bin_xmad = medianMadFunction(x0[bin_index[bin_i]:bin_index[bin_i+1]])
            count.append(bin_index[bin_i+1] - bin_index[bin_i] + 1)
            xmed.append(bin_xmed)
            xmad.append(bin_xmad)
            dt_bin_indices.append(bin_i)

    # Check if populations overlap (use first, smallest dt, bin as reference)
    # logging.info(f'Before min/max bound')
    std_dev = np.array(xmad) * _dtbin_ratio
    xmed = np.array(xmed)

    minBound = xmed - std_dev
    maxBound = xmed + std_dev

    # Find first valid bin with minimum acceptable number of points
    ref_index, = np.where(np.array(count) >= _min_ref_bin_count)

    # If no such valid bin exists, just consider first bin where maxBound != 0
    if ref_index.size == 0:
        ref_index, = np.where(maxBound != 0)

    # Not enough data to proceed
    if ref_index.size == 0:
        return (maxdt, invalid)

    ref_index = ref_index[0]

    exclude = (minBound > maxBound[ref_index]) | (maxBound < minBound[ref_index])

    if np.any(exclude):
        dt_bin_indices = np.array(dt_bin_indices)[exclude]
        maxdt = _dt_edge[dt_bin_indices].min()
        invalid = dt > maxdt

    return (maxdt, invalid)

# @nb.jit(nopython=True, parallel=True)
def cube_filter(vp, dt, mad_std_ratio, current_sensor_group, exclude_sensor_groups):
    """
    Filter data cube by dt (date separation) between the images.

    Input:
    ======
    vp:          Velocity projected to median flow unit vector.
    dt:          Day separation vector.
    mad_std_ratio: Scalar relation between MAD and STD

    Return:
    =======
    invalid: Mask for invalid values.
    maxdt:   Maximum date separation.
    sensor_include: Mask for included sensors.
    """
    # Initialize output
    y_len, x_len, t_len = vp.shape
    dims = (y_len, x_len)
    maxdt = np.full(dims, np.nan)
    sensor_include = np.ones(dims)

    # dims = (y_len, x_len, np.sum(sensor_mask))
    invalid = np.zeros_like(vp, dtype=np.bool_)

    # Loop through all spacial points
    # for j_index in nb.prange(y_len):
    #     for i_index in nb.prange(x_len):
    for j_index in range(0, y_len):
        for i_index in range(0, x_len):
            # Check if filter should be skipped due to exclude_sensor_groups
            if exclude_sensor_groups[j_index, i_index] and \
               current_sensor_group in exclude_sensor_groups[j_index, i_index]:
                # logging.info(f'j={j_index} i={i_index}: skipping {current_sensor_group} due to exclude_groups={exclude_sensor_groups[j_index, i_index]}')
                invalid[j_index, i_index, :] = True
                sensor_include[j_index, i_index] = 0
                continue

            maxdt[j_index, i_index], invalid[j_index, i_index, :] = cube_filter_iteration(
                vp[j_index, i_index],
                dt,
                mad_std_ratio
            )
            # logging.info(f'DEBUG: j={j_index} i={i_index} after cube_filter: maxdt={maxdt[j_index, i_index]}')

    return invalid, maxdt, sensor_include

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

# Disable numba as its wrapper for lstsq does not support "rcond" input parameter for LSQ fit
# @nb.jit(nopython=True)
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
    # logging.info(f'DEBUG: LSQ fit: D={D}')
    D = np.concatenate((D, M), axis=1)

    # Make numpy happy: have all data 2D
    # w_d.reshape((len(w_d), 1))

    # Solve for coefficients of each column in the Vandermonde:
    p = np.linalg.lstsq(w_d.reshape((len(w_d), 1)) * D, w_d*d_obs, rcond=None)[0]

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

    Return:
    results_valid: Boolean flag set to True if results are valid, False otherwise meaning that
                   further computation should be skipped. Computations should be
                   skipped if identified data validity mask is empty which results
                   in no data to be processed.
                   This flag has to be introduced in order to use numba compilation
                   otherwise numba-compiled code fails when using empty mask (pure
                   Python code does not).
    start_year, stop_year, v_in, v_err_in, dyr, totalnum, M_in: Filtered by data validity mask
                    and sorted by mid_date all input data variables.
    """
    # start_time = timeit.default_timer()
    # logging.info(f"Start init of itslive_lsqfit_annual: M_input.shape={M_input.shape}")

    # Ensure we're starting with finite data
    isf_mask = np.isfinite(v_input) & np.isfinite(v_err_input)
    results_valid = np.any(isf_mask)

    if not results_valid:
        # All results will be ignored, but they must match in type to valid returned
        # results to keep numba happy, so just return input-like data
        # Can't use input variables as they are read-only which makes numba unhappy
        dy_out = np.zeros_like(start_dec_year)

        return (results_valid, dy_out, dy_out, np.zeros_like(v_input), np.zeros_like(v_err_input), np.zeros_like(dec_dt), 0, np.zeros_like(M_input))

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

    return (results_valid, start_year, stop_year, v_in, v_err_in, dyr, totalnum, M_in)

@nb.jit(nopython=True)
def init_lsq_fit2(v_median, v_input, v_err_input, start_dec_year, stop_dec_year, dec_dt, all_years, M_input, mad_thresh, mad_std_ratio):
    """
    Initialize variables for LSQ fit.

    Return:
    results_valid: Boolean flag set to True if results are valid, False otherwise meaning that
                   further computation should be skipped. Computations should be
                   skipped if identified data validity mask is empty which results
                   in no data to be processed.
                   This flag has to be introduced in order to use numba compilation
                   otherwise numba-compiled code fails when using empty mask (pure
                   Python code does not).
    start_year, stop_year, v_in, v_err_in, dyr, w_v, w_d, d_obs, y1, M_in: Filtered by data
                   validity mask and pre-processed for LSQ fit input data variables.
    """
    _num_valid_points = 30

    # Remove outliers based on MAD filter for v, subtract from v to get residual
    v_residual = np.abs(v_input - v_median)

    # Take median of residual, multiply median of residual * 1.4826 = sigma
    v_sigma = np.median(v_residual)*mad_std_ratio

    non_outlier_mask  = ~(v_residual > (2.0 * mad_thresh * v_sigma))

    # If less than _num_valid_points don't do the fit
    results_valid = (np.sum(non_outlier_mask) >= _num_valid_points)
    # results_valid = np.any(non_outlier_mask)

    if not results_valid:
        # All results will be ignored, but they must match in type to valid returned
        # results to keep numba happy.
        # Can't use input variables as they are read-only which makes numba unhappy
        v_out = np.zeros_like(v_input)
        v_err_out = np.zeros_like(v_err_input)
        dy_out = np.zeros_like(start_dec_year)

        return (results_valid, dy_out, dy_out, v_out, v_err_out, np.zeros_like(dec_dt), v_err_out, v_err_out.astype(np.float64), v_out, np.arange(1, 2), np.zeros_like(M_input))

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

    return (results_valid, start_year, stop_year, v_in, v_err_in, dyr, w_v, w_d, d_obs, y1, M_in)

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
    count,
    v_limit
):
    # Populates [A,ph,A_err,t_int,v_int,v_int_err,N_int,count_image_pairs] data
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

    results_valid, start_year, stop_year, v, v_err, dyr, totalnum, M = init_lsq_fit1(
        v_input, v_err_input, start_dec_year, stop_dec_year, dec_dt, all_years, M_input
    )
    # Capture runtimes of specific processing steps
    init_runtime1 = timeit.default_timer() - init_runtime
    init_runtime2 = 0
    init_runtime3 = 0
    iter_runtime = 0

    if not results_valid:
        # There is no data to process, exit
        return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])

    # Compute outside of numba-compiled code as numba does not support a lot of scipy
    # functionality
    # Apply 15-point moving median to v, subtract from v to get residual
    init_runtime = timeit.default_timer()
    v_median = ndimage.median_filter(v, _mad_kernel_size)
    init_runtime2 = timeit.default_timer() - init_runtime

    init_runtime = timeit.default_timer()
    results_valid, start_year, stop_year, v, v_err, dyr, w_v, w_d, d_obs, y1, M = init_lsq_fit2(
        v_median, v, v_err, start_year, stop_year, dyr, all_years, M, _mad_thresh, mad_std_ratio
    )
    init_runtime3 = timeit.default_timer() - init_runtime

    if not results_valid:
        # There is no data to process, exit
        return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])

    # Filter sum of each column
    hasdata = M.sum(axis=0) > 0
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

    for i in range(0, _mad_filter_iterations):
        # Displacement Vandermonde matrix: (these are displacements! not velocities, so this matrix is just the definite integral wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
        runtime = timeit.default_timer()
        p, d_model = itslive_lsqfit_iteration(start_year, stop_year, M, w_d, d_obs)
        iter_runtime += (timeit.default_timer() - runtime)

        # Original code:
        # # Divide by dt to avoid penalizing long dt [asg]
        # d_resid = np.abs(d_obs - d_model)/dyr
        #
        # # Robust standard deviation of errors, using median absolute deviation
        # d_sigma = np.median(d_resid)*mad_std_ratio
        #
        # outliers = d_resid > (_mad_thresh * d_sigma)
        # if np.all(outliers):
        #     # All are outliers, return from the function
        #     results_valid = False
        #     return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])
        #
        # if (outliers.sum() / totalnum) < 0.01 and i != last_iteration:
        #     # There are less than 1% outliers, skip the rest of iterations
        #     # if it's not the last iteration
        #     # logging.info(f'{outliers_fraction*100}% ({outliers.sum()} out of {totalnum}) outliers, done with first LSQ loop after {i+1} iterations')
        #     break

        if i < last_iteration:
            # Divide by dt to avoid penalizing long dt [asg]
            d_resid = np.abs(d_obs - d_model)/dyr

            # Robust standard deviation of errors, using median absolute deviation
            d_sigma = np.median(d_resid)*mad_std_ratio

            outliers = d_resid > (_mad_thresh * d_sigma)
            if np.all(outliers):
                # All are outliers, return from the function
                results_valid = False
                return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])

            if (outliers.sum() / totalnum) < 0.01:
                # There are less than 1% outliers, skip the rest of iterations
                # if it's not the last iteration
                # logging.info(f'{outliers_fraction*100}% ({outliers.sum()} out of {totalnum}) outliers, done with first LSQ loop after {i+1} iterations')
                break

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
                results_valid = False
                return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])

            y1 = y1[hasdata]
            M = M[:, hasdata]

    # logging.info(f'Size of p:{p.shape}')

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

    # Number of image pairs used
    count_image_pairs = M.shape[0]

    # Reshape array to have the same number of dimensions as M for multiplication
    w_v = w_v.reshape((1, w_v.shape[0]))

    v_int_err = 1/np.sqrt((w_v@M).sum(axis=0))

    # Identify year's indices to assign return values to in "final" composite
    # variables
    ind = itersect_years(all_years, y1)

    # logging.info(f'Finished post-process ({timeit.default_timer() - start_time} seconds)')
    # start_time = timeit.default_timer()

    # On return: amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1
    # amplitude[ind] = A
    # phase[ind] = ph
    # sigma[ind] = A_err
    mean[ind] = v_int
    error[ind] = v_int_err
    count[ind] = N_int

    # logging.info(f'DEBUG: LSQ fit error: {error}')
    offset, slope, se = weighted_linear_fit(y1, mean[ind], error[ind])

    if offset >= v_limit:
        # Since it's invalid v0, report all output as invalid
        results_valid = False
        return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [])

    return (results_valid, init_runtime1, init_runtime2, init_runtime3, iter_runtime, [A, amp_error, ph, offset, slope, se, count_image_pairs])

@nb.jit(nopython=True)
def annual_magnitude(
    vx_fit,
    vy_fit,
    vx_fit_err,
    vy_fit_err,
    vx_fit_count,
    vy_fit_count,
    # v_fit, # outputs
    # v_fit_err,
    # v_fit_count
):
    """
    Computes and returns the annual mean, error, count, and outlier fraction
    from component values projected on the unit flow vector defined by vx0 and vy0.

    Inputs:
        vx_fit: annual mean flow in x direction
        vy_fit: annual mean flow in y direction
        vx_fit_err: error in annual mean flow in x direction
        vy_fit_err: error in annual mean flow in y direction
        vx_fit_count: number of values used to determine annual mean flow in x direction
        vy_fit_count: number of values used to determine annual mean flow in y direction

    Outputs:
        self.mean.v[start_y:stop_y, start_x:stop_x, :]
        self.error.v[start_y:stop_y, start_x:stop_x, :]
        self.count.v[start_y:stop_y, start_x:stop_x, :]
    """
    # solve for velocity magnitude
    v_fit = np.sqrt(vx_fit**2 + vy_fit**2) # velocity magnitude

    uv_x = vx_fit/v_fit # unit flow vector
    uv_y = vy_fit/v_fit

    v_fit_err = np.abs(vx_fit_err) * np.abs(uv_x) # flow acceleration in direction of unit flow vector, take absolute values
    v_fit_err += np.abs(vy_fit_err) * np.abs(uv_y)

    v_fit_count = np.ceil((vx_fit_count + vy_fit_count) / 2)

    return v_fit, v_fit_err, v_fit_count

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
    vx_se:
    vy_se:

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
    self.std_error.vx[start_y:stop_y, start_x:stop_x],
    self.std_error.vy[start_y:stop_y, start_x:stop_x]

    Output:
    =======
    v
    dv_dt
    v_amp
    v_amp_err
    v_phase
    v_se

    Correlation to actual outputs:
    =============================
    self.offset.v[start_y:stop_y, start_x:stop_x]
    self.slope.v[start_y:stop_y, start_x:stop_x]
    self.amplitude.v[start_y:stop_y, start_x:stop_x]
    self.sigma.v[start_y:stop_y, start_x:stop_x]
    self.phase.v[start_y:stop_y, start_x:stop_x]
    self.std_error.v[start_y:stop_y, start_x:stop_x]
    """
    _two_pi = np.pi * 2

    # solve for velocity magnitude and acceleration
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

def weighted_linear_fit(t, v, v_err, datetime0=datetime.datetime(2019, 7, 2)):
    """
    Returns the offset, slope, and error for a weighted linear fit to v with an intercept of datetime0.

   - t: date (year) of input estimates
   - v: estimates
   - v_err: estimate errors
   - datetime0: model intercept
   """
    yr = np.array([decimal_year(datetime.datetime(each, 7, 2)) for each in t])
    yr0 = decimal_year(datetime0)
    yr = yr - yr0

    # Per Chad:
    # In the data testing Matlab script I posted, you may notice I added a step
    # because in a few grid cells we were getting crazy velocities where, say,
    # there were only v measurements in 2013 and 2014, and that meant we were
    # extrapolating to get to 2019.5.
    # To minimize the influence of such cases, we should
    # * Only calculate the slope in grid cells that contain at least one valid
    #   measurement before 2019 and at least one valid measurement after 2019.
    #   That will constrain the values of v0 by ensuring we’re interpolating
    #   between good measurements.
    # * Wherever Condition 1 is not met, fill v0 with the weighted mean velocity
    #   of whatever measurements are available.
    # * Wherever Condition 1 is not met, fill dv_dt with NaN.
    # If there is no data before or after datetime0.year, then return NaN's
    valid = (~np.isnan(v)) & (~np.isnan(v_err))

    if valid.sum() == 0:
        # There are no valid entries
        return np.nan, np.nan, np.nan

    # weights for velocities:
    w_v = 1 / (v_err**2)
    w_v = w_v[valid]

    before_datetime0 = (yr < 0)
    after_datetime0 = (yr >= 0)

    # Is there data on both sides of datatime0:
    interpolate_data = np.any(valid & before_datetime0) and np.any(valid & after_datetime0)
    if not interpolate_data:
        # There is no valid data on both ends of the datetime0, populate:
        # v0 (offset):   with weighted mean of whatever values are available
        # dv_dt (slope): with NaN
        offset = np.average(v[valid], weights=w_v)
        slope = np.nan
        error = np.sqrt((v_err[valid]**2).sum())/(valid.sum()-1)

        return offset, slope, error

    # Normalize the weights per Chad's suggestion before LSQ fit:
    w_v = np.sqrt(w_v/np.mean(w_v))

    # create design matrix
    D = np.ones((len(yr), 2))
    D[:, 1] = yr

    # Solve for coefficients of each column in the Vandermonde:
    # w_v = w_v[valid]
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
    # Index order for data to be continuous in X dimension
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

    # If datacube contains only numeric sensor values (Landsat8 or Landsat9),
    # sensor values are of type float, otherwise sensor values are of string type
    # ---> support both
    LANDSAT45 = MSTuple('L45', ['4.', '5.', '4.0', '5.0', 4.0, 5.0], 'L4_L5')
    LANDSAT89 = MSTuple('L89', ['8.', '9.', '8.0', '9.0', 8.0, 9.0], 'L8_L9')
    LANDSAT7  = MSTuple('L7', ['7.', '7.0', 7.0], 'L7')

    SENTINEL1 = MSTuple('S1', ['1A', '1B'], 'S1A_S1B')
    SENTINEL2 = MSTuple('S2', ['2A', '2B'], 'S2A_S2B')

    # TODO: update with new missions groups as their granules are added
    # to the datacubes
    ALL_GROUPS = {
        LANDSAT45.mission: LANDSAT45,
        LANDSAT7.mission: LANDSAT7,
        LANDSAT89.mission: LANDSAT89,
        SENTINEL1.mission: SENTINEL1,
        SENTINEL2.mission: SENTINEL2
    }

    # Mapping of sensor to the group
    GROUPS = {}

    # Mapping of sensor to the group label
    GROUPS_MISSIONS = {}

    @staticmethod
    def _groups():
        """
        Return mapping of sensor to its corresponding sensor group.

        This method builds mapping of the individual sensor to the group
        it belongs to:
            {
                '4.':  LANDSAT45,
                '5.':  LANDSAT45,
                4.0:   LANDSAT45,
                5.0:   LANDSAT45,
                '4.0': LANDSAT45,
                '4.0': LANDSAT45,
                '7.':  LANDSAT7,
                '7.0': LANDSAT7,
                7.0:   LANDSAT7,
                '8.':  LANDSAT89,
                '9.':  LANDSAT89,
                8.0:   LANDSAT89,
                9.0:   LANDSAT89,
                '8.0': LANDSAT89,
                '9.0': LANDSAT89,
                '1A':  SENTINEL1,
                '1B':  SENTINEL1,
                '2A':  SENTINEL2,
                '2B':  SENTINEL2
            }
        """
        all_sensors = {}

        for each_group in MissionSensor.ALL_GROUPS.values():
            for each_sensor in each_group.sensors:
                all_sensors[each_sensor] = each_group

        return all_sensors

    @staticmethod
    def _groups_missions():
        """
        Return mapping of sensor to its corresponding sensor group name.

        This method builds mapping of the individual sensor to the group
        it belongs to:
            {
                '4.':  'L45',
                '5.':  'L45',
                4.0:   'L45',
                5.0:   'L45',
                '4.0': 'L45',
                '4.0': 'L45',
                '7.':  'L7',
                '7.0': 'L7',
                7.0:   'L7',
                '8.':  'L89',
                '9.':  'L89',
                8.0:   'L89',
                9.0:   'L89',
                '8.0': 'L89',
                '9.0': 'L89',
                '1A':  'S1',
                '1B':  'S1',
                '2A':  'S2',
                '2B':  'S2'
            }
        """
        all_sensors = {}
        for each_group in MissionSensor.ALL_GROUPS.values():
            for each_sensor in each_group.sensors:
                # Use homogeneous type as keys (numba allows for key values of the same type only)
                all_sensors[str(each_sensor)] = each_group.mission

        return all_sensors

# Initialize static data of the class
MissionSensor.GROUPS = MissionSensor._groups()
MissionSensor.GROUPS_MISSIONS = MissionSensor._groups_missions()

class SensorExcludeFilter:
    """
    This class represents filter to identify sensor groups to exclude based
    on the timeseries for a single spacial point of the datacube.
    """
    # Min required values in bin for one sensorgroup to compute stats
    MIN_COUNT=3

    # Longest dt to use for all sensor groups
    MAX_DT = 64

    # Sensor group to compare others to
    REF_SENSOR = MissionSensor.SENTINEL2

    # Multiplier of standard error to use in comparison
    SESCALE = 3

    def __init__(
        self,
        acquisition_start_time,
        acquisition_stop_time,
        sensors,
        sensors_groups
    ):
        """
        Initialize object.

        Inputs:
        =======
        acquisition_start_time - Acquisition datetime for the first image of timeseries.
        acquisition_stop_time - Acquisition datetime for the second image of timeseries.
        sensors - Sensors for the timeseries.
        sensors_groups - List of identified sensor groups in timeseries.
        """
        # Flag if filter should be applies to timeseries
        self.apply = False
        self.binedges = None
        self.groups = sensors_groups

        # Map each sensor to its mission group
        # Use homogeneous type as keys (numba allows for key values of the same type only)
        self.sensors_str = np.array([MissionSensor.GROUPS_MISSIONS[str(x)] for x in sensors])

        # Identify if reference sensor group is present in timeseries
        if SensorExcludeFilter.REF_SENSOR in sensors_groups:
            logging.info(f'Reference sensor {SensorExcludeFilter.REF_SENSOR.mission} is present')
            self.apply = True

            # mask = np.zeros((len(sensors)), dtype=np.bool_)

            # Extract start and end dates that correspond to the sensor group
            mask = (self.sensors_str == SensorExcludeFilter.REF_SENSOR.mission)
            # for each in SensorExcludeFilter.REF_SENSOR.sensors:
            #     # logging.info(f'Update mask with {each} as part of the sensor group')
            #     mask |= (sensors == each)

            start_date = np.array(acquisition_start_time)[mask]
            stop_date = np.array(acquisition_stop_time)[mask]

            logging.info(f'Identified reference "{SensorExcludeFilter.REF_SENSOR.mission}" sensor group: start_date={start_date.min().date()} end_date={stop_date.max().date()}')
            self.binedges = np.arange(
                start_date.min().date(),
                stop_date.max().date(),
                np.timedelta64(73,'[D]'),  # 73 D is 1/5 of a year
                dtype="datetime64[D]"
            )
            logging.info(f'Bin edges: {self.binedges}')

        else:
            logging.info(f'Reference sensor {SensorExcludeFilter.REF_SENSOR.mission} is missing, disable SensorExclude filter.')

    def __call__(self, ds_date_dt, ds_vx, ds_vy, ds_mid_date):
        """
        Invoke filter for the block of spacial points.
        """
        y_len, x_len, _ = ds_vx.shape
        dims = (y_len, x_len)
        exclude_sensors = np.frompyfunc(list, 0, 1)(np.empty(dims, dtype=object))

        if self.apply:
            for j_index in range(0, y_len):
                for i_index in range(0, x_len):
                    exclude_sensors[j_index, i_index] = self.iteration(
                        ds_date_dt,
                        ds_vx[j_index, i_index, :],
                        ds_vy[j_index, i_index, :],
                        ds_mid_date
                    )

                    # logging.info(f'Excluded sensors: {exclude_sensors[j_index, i_index]}')

        return exclude_sensors

    def iteration(self, ds_date_dt, ds_vx, ds_vy, ds_mid_date, plot=False):
        """
        Returns list of sensor groups to exclude based on the timeseries for
        the spacial point.

        Inputs:
        =======
        ds_date_dt: date_dt timeseries for spacial point
        ds_sensors: individual sensors timeseries for spacial point
        ds_vx:      vx timeseries
        ds_vy:      vy timeseries
        ds_mid_date: mid_date timeseries

        Returns list of sensor groups to exclude.
        """
        #     # trim data to redude computations
        #     dtind = dt .<= dtmax;
        #     valid = .~ismissing.(vx)
        #     ind = dtind .& valid
        #
        sensors_to_exclude = []

        # logging.info(f'Num valid points: {np.sum((~np.isnan(ds_vx)))}')
        trimmed_index = ((ds_date_dt <= SensorExcludeFilter.MAX_DT) & (~np.isnan(ds_vx)))
        # logging.info(f'Num valid points after max_dt: {np.sum(trimmed_index)}')

        # If no data left, exit the filter
        if np.sum(trimmed_index) == 0:
            return sensors_to_exclude

        vx = ds_vx[trimmed_index]
        vy = ds_vy[trimmed_index]
        sensor = self.sensors_str[trimmed_index]
        dt = ds_date_dt[trimmed_index]
        mid_dates = ds_mid_date[trimmed_index]

        #
        #     # determine sensor group ids
        #     id, sensorgroups = ItsLive.sensorgroup(sensor)
        #     numsg = length(sensorgroups)
        #  ItsLive.sensorgroup returns indicies, etc - here just use sengrp_from_satellite_dict to directly map to sensorgroup

        # sensor needs to be a numpy array to vectorize comparisons below
        # sensor = np.array([sengrp_from_satellite_dict[x] for x in satellite.values])
        # logging.info(f'Existing groups: {MissionSensor.GROUPS}')
        # sensor = np.array([MissionSensor.GROUPS[x].mission for x in satellite])

        # get unique sensorgroup names
        sensorgroups = set(sensor)

        if SensorExcludeFilter.REF_SENSOR.mission not in sensorgroups:
            return sensors_to_exclude

        #
        #     # convert date to decimal year
        #     decyear = ItsLive.decimalyear(mid_date)
        # skipped this since we are using datetime64's for the comparison

        #     # initialize veriables
        #     vbin = fill(NaN, (numsg,length(binedges)-1))
        # do this as a dict of dicts so we can use sensor name as index
        bindicts = {
            sen: {
                'vbin':      np.nan * np.ones((len(self.binedges)-1)),
                'vstdbin':   np.nan * np.ones((len(self.binedges)-1)),
                'vcountbin': np.zeros((len(self.binedges)-1), dtype='int32')
            }  for sen in sensorgroups
        }

        #
        #     # loop for each sensor group
        #     for sg = 1:numsg
        #         ind = id .== sg;
        #         vx0 = mean(vx[ind])
        #         vy0 = mean(vy[ind])
        #         v0 = sqrt.(vx0.^2 .+ vy0.^2);
        #         uv = vcat(vx0/v0, vy0/v0)
        #         vp = hcat(vx[ind],vy[ind]) * uv # flow acceleration in direction of unit flow vector
        #
        #         vbin[sg,:], vstdbin[sg,:], vcountbin[sg,:], bincenters = ItsLive.binstats(decyear[ind], vp; binedges)
        #     end
        for sen in sensorgroups:
            ind = sen == sensor
            vx0 = np.mean(vx[ind])
            vy0 = np.mean(vy[ind])
            sen_mid_dates = mid_dates[ind]
            v0 = np.sqrt(np.power(vx0, 2.0) + np.power(vy0, 2.0))

            uv = np.array([vx0 / v0, vy0 / v0])
            vp = uv.dot(np.vstack((vx[ind], vy[ind])))

            # do the bin stats here rather than in a separate function - "return" values populate bindicts
            for bin_num, (be_lo, be_hi) in enumerate(zip(self.binedges[:-1], self.binedges[1:])):
                bin_ind = (sen_mid_dates >= be_lo) & (sen_mid_dates < be_hi)
                num_in_bin = np.sum(bin_ind).item() # these are still xarray DataArrays - .item() returns sigular value instead of array(value)

                if num_in_bin >= SensorExcludeFilter.MIN_COUNT:
                    bindicts[sen]['vcountbin'] = num_in_bin
                    bindicts[sen]['vbin'][bin_num] = np.mean(vp[bin_ind])
                    bindicts[sen]['vstdbin'][bin_num] = np.std(vp[bin_ind])

        # Check if reference filter made it into the bindicts:
        refsensor = SensorExcludeFilter.REF_SENSOR.mission
        if refsensor not in bindicts:
            return sensors_to_exclude

        # if plot:
        #     # Not the best practice, but done for debugging only, so it's OK
        #     from matplotlib import pyplot as plt
        #
        #     # Plotting is for debugging purposes only
        #     # bin centers as datetime64's
        #     bincenters = self.binedges[:-1] + ((self.binedges[1:] - self.binedges[:-1]) / 2.0)
        #
        #     bin_colors = {'S1': 'r', 'S2': 'b', 'L89': 'g'}
        #     plt.figure(figsize=(7, 7))
        #     for sen in bindicts.keys():
        #         plt.plot(bincenters, bindicts[sen]['vbin'], f'o{bin_colors[sen]}-', label=sen)
        #     plt.legend()
        #     plt.ion()
        #     plt.show()

        #
        #     # check if Setinel-2 mean is different from Landsat-8
        #     # calculate mean (m) and standard error (s)
        #     m = fill(NaN, numsg)
        #     s = fill(NaN, numsg)
        #     for sg = 1:numsg
        #         covalid = .~isnan.(vbin[id_refsensor,:]) .& .~isnan.(vbin[sg,:])
        #         if sum(covalid) > 3
        #             delta =   vbin[sg,covalid] - vbin[id_refsensor,covalid];
        #             m[sg,1] = mean(delta)
        #             s[sg,1] = std(delta)/sqrt((sum(covalid)-1))
        #         end
        #     end
        #
        #     # check if the mean difference + se < zero,
        #     # if < zero then id_refsensor has a sginificanlty faster mean
        #     disagree = (m .+ (s.*sescale)) .< 0

        stats = {sen: {} for sen in sensorgroups}

        for sen in sensorgroups:
            covalid = (~np.isnan(bindicts[refsensor]['vbin'])) & (~np.isnan(bindicts[sen]['vbin']))

            if sum(covalid) > 3:
                delta = bindicts[sen]['vbin'][covalid] - bindicts[refsensor]['vbin'][covalid]
                stats[sen]['mean'] = np.mean(delta)
                stats[sen]['se'] = np.std(delta)/np.sqrt((sum(covalid)-1))
                stats[sen]['disagree_with_refsensor'] =  (stats[sen]['mean'] + (stats[sen]['se'] * SensorExcludeFilter.SESCALE)) < 0
                if stats[sen]['disagree_with_refsensor']:
                    sensors_to_exclude.append(sen)

        return sensors_to_exclude

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
        DataVars.ImgPairInfo.MISSION_IMG1,
        DataVars.URL
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
        logging.info(f'Datacube sizes: {cube_ds.sizes}')

        # Exclude "faulty" S2 data: filenames containing '23WPN'
        logging.info(f'Excluding S2 data for 23WPN...')
        url_values = cube_ds[DataVars.URL].values
        sel_indices = [index for index, each in enumerate(url_values) if (not os.path.basename(each).startswith('S2') or (os.path.basename(each).startswith('S2') and not('23WPN' in each)))]

        logging.info(f'Leaving {len(sel_indices)} layers...')
        cube_ds = cube_ds.isel(mid_date=sel_indices)
        logging.info(f'Datacube sizes after S2 exclusion: {cube_ds.sizes}')

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
        self.count_image_pairs = CompositeVariable(dims, 'count_image_pairs')
        self.amplitude = CompositeVariable(dims, 'amplitude')
        self.sigma = CompositeVariable(dims, 'sigma')
        self.phase = CompositeVariable(dims, 'phase')
        self.offset = CompositeVariable(dims, 'offset')
        self.slope = CompositeVariable(dims, 'slope')
        self.std_error = CompositeVariable(dims, 'std_error')

        # Sensor data for the cube's layers
        self.sensors = cube_ds[DataVars.ImgPairInfo.SATELLITE_IMG1].values
        self.mid_date = cube_ds[Coords.MID_DATE].values

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
        # Step through each unique sensor and collect sensor group it belongs together
        for each in unique_sensors:
            if each not in collected_sensors:
                self.sensors_groups.append(MissionSensor.GROUPS[each])
                collected_sensors.extend(self.sensors_groups[-1].sensors)

        dims = (y_len, x_len, len(self.sensors_groups))
        self.max_dt = np.full(dims, np.nan)
        # np.bool_ - to make numba happy
        self.sensor_include = np.ones(dims)

        # Date when composites were created
        self.date_created = datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        self.date_updated = self.date_created

        # Initialize sensor exclusion filter
        self.sensor_filter = SensorExcludeFilter(
            acq_datetime_img1,
            acq_datetime_img2,
            self.sensors,
            self.sensors_groups
        )

        # TODO: take care of self.date_updated when support for composites updates
        # is implemented

    def create(self, output_store: str, s3_bucket: str):
        """
        Create datacube composite: cube time mean values.
        """
        # Loop through cube in chunks to minimize memory footprint
        # x_index: 331
        # y_index: 796
        x_start = 0
        x_num_to_process = self.cube_sizes[Coords.X]

        # For debugging only
        # ======================
        # Alex's Julia code
        # datacubes/v02/N60W040/ITS_LIVE_vel_EPSG3413_G0120_X150000_Y-2650000.zarr
        # x_start = 118  # good point
        # x_start = 265  # bad point

        # GRE huge v0 values
        # x_start = 512
        # ======================

        x_num_to_process = self.cube_sizes[Coords.X] - x_start
        # For debugging only
        # ======================
        # x_num_to_process = 1

        while x_num_to_process > 0:
            # How many tasks to process at a time
            x_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if x_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else x_num_to_process

            y_start = 0
            y_num_to_process = self.cube_sizes[Coords.Y]

            # For debugging only
            # ======================
            # Alex's Julia code
            # y_start = 428  # good point
            # y_start = 432  # bad point

            # GRE huge v0 values
            # y_start = 32

            y_num_to_process = self.cube_sizes[Coords.Y] - y_start
            # For debugging only
            # ======================
            # y_num_to_process = 1

            while y_num_to_process > 0:
                y_num_tasks = ITSLiveComposite.NUM_TO_PROCESS if y_num_to_process > ITSLiveComposite.NUM_TO_PROCESS else y_num_to_process

                self.cube_time_mean(x_start, x_num_tasks, y_start, y_num_tasks)
                gc.collect()

                y_num_to_process -= y_num_tasks
                y_start += y_num_tasks

            x_num_to_process -= x_num_tasks
            x_start += x_num_tasks

        # Save data to Zarr store
        self.to_zarr(output_store, s3_bucket)

    @staticmethod
    # @nb.jit(nopython=True, parallel=True)
    def project_v_to_median_flow(ds_vx, ds_vy, ds_date_dt, ds_sensors_str, exclude_sensors):
        """
        Project valid velocity values to median flow unit vector.

        Inputs:
        =======
        ds_vx: 3d block of vx values.
        ds_vy: 3d block of vy values.
        ds_date_dt: day separation for velocity image pairs.
        ds_sensors: Current sensors for the datacube.
        exclude_sensors: 2d "map" of sensors to exclude from calculations (one list per each [y, x] point).
        """
        vp = np.full_like(ds_vx, np.nan)

        dims = ds_vx.shape
        y_len = dims[0]
        x_len = dims[1]

        # for j_index in nb.prange(y_len):
        #     for i_index in nb.prange(x_len):
        for j_index in range(0, y_len):
            for i_index in range(0, x_len):
                # Exclude all identified invalid sensor groups per [y, x] point
                exclude_mask = np.zeros((len(ds_sensors_str)), dtype=np.bool_)

                for each in exclude_sensors[j_index, i_index]:
                    # logging.info(f'DEBUG: j={j_index} i={i_index}: exclude {each} as part of {exclude_sensors[j_index, i_index]}')
                    exclude_mask |= (ds_sensors_str == each)
                    # logging.info(f'exclude_mask.sum={exclude_mask.sum()}')

                include_mask = ~exclude_mask
                x_in = ds_vx[j_index, i_index, include_mask]
                y_in = ds_vy[j_index, i_index, include_mask]
                dt = ds_date_dt[include_mask]
                vp[j_index, i_index, include_mask] = create_projected_velocity(x_in, y_in, dt)

        return vp

    def cube_time_mean(self, start_x, num_x, start_y, num_y):
        """
        Compute time average for the datacube [:, :, start_x:stop_index] coordinates.
        Update corresponding entries in output data variables.
        """
        # Set current block length for the X and Y dimensions
        stop_y = start_y + num_y
        stop_x = start_x + num_x
        ITSLiveComposite.Chunk = Chunk(start_x, stop_x, num_x, start_y, stop_y, num_y)
        ITSCube.show_memory_usage(f'before cube_time_mean(): start_x={start_x} start_y={start_y}')

        # Start timer
        start_time = timeit.default_timer()

        # ----- FILTER DATA -----
        # Filter data based on locations where means of various dts are
        # statistically different and mad deviations from a running meadian
        logging.info(f'Filter data based on dt binned medians...')

        # Initialize variables
        dims = (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN)

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

        # Call filter to exclude sensors if any
        logging.info(f'Sensor exclude filter...')
        start_time = timeit.default_timer()
        exclude_sensors = self.sensor_filter(ITSLiveComposite.DATE_DT.values, vx, vy, self.mid_date)
        logging.info(f'Finished sensor exclude filter ({timeit.default_timer() - start_time} seconds)')

        # Project valid (excluding sensors) v onto median flow vector:
        # take into account exclude_sensors for each spacial point
        v_invalid = np.full(dims, False)

        # Count all valid points before any filters are applied
        count_mask = ~np.isnan(vx)
        count0_vx = count_mask.sum(axis=2)

        start_time = timeit.default_timer()
        logging.info(f'Project velocity to median flow unit vector...')
        # Project velocity to median flow unit vector using only valid sensors
        vp = ITSLiveComposite.project_v_to_median_flow(
            vx,
            vy,
            ITSLiveComposite.DATE_DT.values,
            self.sensor_filter.sensors_str,
            exclude_sensors
        )
        logging.info(f'Done with velocity projection to median flow unit vector (took {timeit.default_timer() - start_time} seconds)')

        mask = np.isfinite(vp)
        logging.info(f'Number of isfinite values: {np.sum(mask[0, 0, :])} out of {vp.shape}')

        # DEBUG only: store vp to CSV file
        # logging.info(f'vp.size={vp.shape}')
        # filename = f'good_vp.csv'
        # np.savetxt(filename, vp[0, 0, :], delimiter=',')

        # Apply dt filter: step through all sensors groups
        for i, sensor_group in enumerate(self.sensors_groups):
            logging.info(f'Filtering dt for sensors of "{sensor_group.mission}" ({i+1} out ' \
                f'of {len(self.sensors_groups)} sensor groups)')

            # Find which layers correspond to the sensor group
            mask = (self.sensor_filter.sensors_str == sensor_group.mission)

            # Filter current block's variables
            logging.info(f'Start dt filter for projected v using {sensor_group.mission} sensors...')
            start_time = timeit.default_timer()

            v_invalid[:, :, mask], \
            self.max_dt[start_y:stop_y, start_x:stop_x, i], \
            self.sensor_include[start_y:stop_y, start_x:stop_x, i] = cube_filter(
                vp[..., mask],
                ITSLiveComposite.DATE_DT.values[mask],
                ITSLiveComposite.MAD_STD_RATIO,
                sensor_group.mission,
                exclude_sensors
            )
            logging.info(f'Done with dt filter for projected v (took {timeit.default_timer() - start_time} seconds)')

        # Load data to avoid NotImplemented exception when invoked on Dask arrays
        logging.info(f'Compute invalid mask...')
        start_time = timeit.default_timer()

        invalid = v_invalid | (np.hypot(vx, vy) > ITSLiveComposite.V_LIMIT)

        # Mask data
        vx[invalid] = np.nan
        vy[invalid] = np.nan

        # plot = True
        # if plot:
        #     # Not the best practice, but done for debugging only, so leave it
        #     from matplotlib import pyplot as plt
        #
        #     vp[invalid] = np.nan
        #
        #     # Plotting is for debugging purposes only
        #     plt.figure(figsize=(7, 7))
        #     plt.plot(self.mid_date, np.hypot(vx[0, 0, :], vy[0, 0, :]), 'xg', label='hypot(vx, vy)')
        #     plt.plot(self.mid_date, vp[0, 0, :], 'xr', label='vp')
        #     plt.legend()
        #     plt.ion()
        #     plt.show()

        invalid = np.nansum(invalid, axis=2)
        invalid = np.divide(invalid, np.sum(np.isnan(vx), 2) + invalid)

        logging.info(f'Finished filtering with invalid mask ({timeit.default_timer() - start_time} seconds)')

        # %% Least-squares fits to detemine amplitude, phase and annual means
        logging.info(f'Find vx annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        # filename = f'good_vx.csv'
        # np.savetxt(filename, vx[0, 0, :], delimiter=',')
        #
        # filename = f'good_start_dec_year.csv'
        # np.savetxt(filename, ITSLiveComposite.START_DECIMAL_YEAR, delimiter=',')
        #
        # filename = f'good_stop_dec_year.csv'
        # np.savetxt(filename, ITSLiveComposite.STOP_DECIMAL_YEAR, delimiter=',')
        #
        # filename = f'good_dec_dt.csv'
        # np.savetxt(filename, ITSLiveComposite.DECIMAL_DT, delimiter=',')

        filename = f'good_mid_date.csv'
        debug_dates = [str(t.astype('M8[ms]').astype('O')) for t in self.mid_date]
        with open(filename, 'w') as debug_fh:
            debug_fh.write('\n'.join(debug_dates))

        # logging.info(f'DEBUG:  Before LSQ fit: vx: min={np.nanmin(vx)} max={np.nanmax(vx)}')
        # Transform vx data to make time series continuous in memory: [y, x, t]
        ITSLiveComposite.cubelsqfit2(
            vx,
            self.vx_error,
            self.amplitude.vx,
            self.phase.vx,
            self.mean.vx,
            self.error.vx,
            self.sigma.vx,
            self.count.vx,
            self.count_image_pairs.vx,
            self.offset.vx,
            self.slope.vx,
            self.std_error.vx
        )
        logging.info(f'Finished vx LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        # Outlier fraction is based on vx data (count for vx and v are identical to vx's count)
        self.outlier_fraction[start_y:stop_y, start_x:stop_x] = 1 - (self.count_image_pairs.vx[start_y:stop_y, start_x:stop_x] / count0_vx)

        logging.info(f'Find vy annual means using LSQ fit... ')
        start_time = timeit.default_timer()

        filename = f'good_vy.csv'
        np.savetxt(filename, vy[0, 0, :], delimiter=',')

        # logging.info(f'DEBUG:  Before LSQ fit: vy: min={np.nanmin(vy)} max={np.nanmax(vy)}')
        ITSLiveComposite.cubelsqfit2(
            vy,
            self.vy_error,
            self.amplitude.vy,
            self.phase.vy,
            self.mean.vy,
            self.error.vy,
            self.sigma.vy,
            self.count.vy,
            self.count_image_pairs.vy,
            self.offset.vy,
            self.slope.vy,
            self.std_error.vy
        )
        logging.info(f'Finished vy LSQ fit (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find annual magnitude... ')
        start_time = timeit.default_timer()

        self.mean.v[start_y:stop_y, start_x:stop_x, :], \
        self.error.v[start_y:stop_y, start_x:stop_x, :], \
        self.count.v[start_y:stop_y, start_x:stop_x, :] = annual_magnitude(
            self.mean.vx[start_y:stop_y, start_x:stop_x, :],
            self.mean.vy[start_y:stop_y, start_x:stop_x, :],
            self.error.vx[start_y:stop_y, start_x:stop_x, :],
            self.error.vy[start_y:stop_y, start_x:stop_x, :],
            self.count.vx[start_y:stop_y, start_x:stop_x, :],
            self.count.vy[start_y:stop_y, start_x:stop_x, :],
        )
        logging.info(f'Finished annual magnitude (took {timeit.default_timer() - start_time} seconds)')

        logging.info(f'Find climatology magnitude...')
        start_time = timeit.default_timer()

        self.offset.v[start_y:stop_y, start_x:stop_x], \
        self.slope.v[start_y:stop_y, start_x:stop_x], \
        self.amplitude.v[start_y:stop_y, start_x:stop_x], \
        self.sigma.v[start_y:stop_y, start_x:stop_x], \
        self.phase.v[start_y:stop_y, start_x:stop_x], \
        self.std_error.v[start_y:stop_y, start_x:stop_x] = \
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
            self.std_error.vx[start_y:stop_y, start_x:stop_x],
            self.std_error.vy[start_y:stop_y, start_x:stop_x]
        )
        logging.info(f'Finished climatology magnitude (took {timeit.default_timer() - start_time} seconds)')

        # Nan out invalid values
        # WAS: invalid_mask = (self.mean.v > ITSLiveComposite.V_LIMIT) | (self.amplitude.v > ITSLiveComposite.V_AMP_LIMIT)
        invalid_mask = (self.mean.v > ITSLiveComposite.V_LIMIT)
        self.mean.v[invalid_mask] = np.nan

        invalid_mask = (self.amplitude.v > ITSLiveComposite.V_AMP_LIMIT)
        self.amplitude.v[invalid_mask] = np.nan

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

        sensors_labels_attr = [f'Band {index+1}: {sensors_labels[index]}' for index in range(len(sensors_labels))]
        sensors_labels_attr = f'{", ".join(sensors_labels_attr)}'

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
                CompOutputFormat.SENSORS_LABELS: sensors_labels_attr,
                DataVars.UNITS: DataVars.ImgPairInfo.UNITS[DataVars.ImgPairInfo.DATE_DT]
            }
        )
        self.max_dt = None
        gc.collect()

        self.sensor_include = self.sensor_include.transpose(CompositeVariable.CONT_IN_X)

        ds[CompDataVars.SENSOR_INCLUDE] = xr.DataArray(
            data=self.sensor_include,
            coords=var_coords,
            dims=var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.SENSOR_INCLUDE],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.SENSOR_INCLUDE],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                CompOutputFormat.SENSORS_LABELS: sensors_labels_attr,
                DataVars.UNITS: DataVars.BINARY_UNITS
            }
        )
        self.sensor_include = None
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
            data=self.std_error.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VX0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VX0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.std_error.vx = None
        gc.collect()

        ds[CompDataVars.VY0_ERROR] = xr.DataArray(
            data=self.std_error.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.VY0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.VY0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.std_error.vy = None
        gc.collect()

        ds[CompDataVars.V0_ERROR] = xr.DataArray(
            data=self.std_error.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.V0_ERROR],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.V0_ERROR],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.UNITS: DataVars.M_Y_UNITS
            }
        )
        self.std_error.v = None
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
            data=self.count_image_pairs.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                DataVars.STD_NAME: CompDataVars.STD_NAME[CompDataVars.COUNT0],
                DataVars.DESCRIPTION_ATTR: CompDataVars.DESCRIPTION[CompDataVars.COUNT0],
                DataVars.GRID_MAPPING: DataVars.MAPPING,
                DataVars.NOTE: f'{CompDataVars.COUNT0} often does not equal the sum of annual counts ({CompDataVars.COUNT}) as a single image pair can contribute to the least squares fit for multiple years',
                DataVars.UNITS: DataVars.COUNT_UNITS
            }
        )
        self.count_image_pairs = None
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
                'dtype': np.float32,
                'compressor': compressor
            })

        # Settings for "short" datatypes
        for each in [
            CompDataVars.COUNT,
            CompDataVars.COUNT0
        ]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_BYTE,
                'dtype': np.uint32
            })

        # Settings for "max_dt" datatypes
        encoding_settings.setdefault(CompDataVars.MAX_DT, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_POS_VALUE,
                'dtype': np.short
            })

        # Settings for "sensor_include" datatypes
        encoding_settings.setdefault(CompDataVars.SENSOR_INCLUDE, {}).update({
                'dtype': np.short
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
            CompDataVars.MAX_DT
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
            CompDataVars.VX_AMP_ERROR,
            CompDataVars.VY_AMP_ERROR,
            CompDataVars.V_AMP_ERROR,
            CompDataVars.OUTLIER_FRAC,
            CompDataVars.SENSOR_INCLUDE,
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
        count_image_pairs,
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
        # dims = (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len)
        # outlier_frac = np.full(dims, np.nan)

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
                mask = ~np.isnan(v[j, i, :])
                if mask.sum() < _num_valid_points:
                    # Skip the point, return no outliers
                    continue

                global_i = i + ITSLiveComposite.Chunk.start_x
                global_j = j + ITSLiveComposite.Chunk.start_y

                results_valid, init_runtime1, init_runtime2, init_runtime3, lsq_runtime, results = \
                itslive_lsqfit_annual(
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
                    count[global_j, global_i, :],
                    ITSLiveComposite.V_LIMIT,
                )

                init_time1 += init_runtime1
                init_time2 += init_runtime2
                init_time3 += init_runtime3
                lsq_time += lsq_runtime

                if not results_valid:
                    continue

                amplitude[global_j, global_i], \
                sigma[global_j, global_i], \
                phase[global_j, global_i], \
                offset[global_j, global_i], \
                slope[global_j, global_i], \
                se[global_j, global_i], \
                count_image_pairs[global_j, global_i] = results

        logging.info(f'Init_time1: {init_time1} sec, Init_time2: {init_time2} sec, Init_time3: {init_time3} sec, lsq_time: {lsq_time} seconds')
        return


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

    if os.path.exists(args.outputStore):
        output_size = subprocess.run(['du', '-skh', args.outputStore], capture_output=True, text=True).stdout.split()[0]
        logging.info(f'Size of {args.outputStore}: {output_size}')

    else:
        logging.info(f'{args.outputStore} is not created.')

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
