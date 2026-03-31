"""
ITSLiveComposite class creates yearly and mean composites of ITS_LIVE
datacubes with data within the same target projection, bounding polygon
and datetime period as specified at the time of the datacube generation.

Authors:
Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Mark Fahnestock (UAF)

Jet Propulsion Laboratory, California Institute of Technology, Pasadena,
California
September 10, 2025
"""
import collections
import datetime
from dateutil.parser import parse
import gc
from joblib import Parallel, delayed
import json
import logging
import numba as nb
import numpy as np
import os
import pandas as pd
from scipy import ndimage
import timeit
import xarray as xr
import zarr

# Local imports
from itscube import ITSCube
import sensors
import sensorFilters
from itscube_types import CubeFormat, ImgPairInfo, Mapping, Vars
from itslive_mosaics_types import TIME_ATTRS, SENSORS_ATTRS, X_ATTRS, Y_ATTRS
from itslive_mosaics_types import CompositeVars
from itslive_binary_type import BinaryFlag
import itslive_utils
import aws_utils
import shapefile
import utils

# Intercept date used for a weighted linear fit
CENTER_DATE = datetime.datetime(2018, 1, 1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Bin edges for dt_max filter
DT_EDGE = np.array([0, 16, 32, 64, 128, 256, np.inf])

# Bin edges for the median flow vector
DT_MEDIAN_FLOW = np.array([16, 32, 64, 128, 256, np.inf])

# For convenience
TWO_PI = np.pi * 2


def decimal_year(dt):
    """
    Convert datetime to decimal year.

    Input:
    ======
    dt: datetime object.

    Returns:
    =======
    Decimal year.
    """
    start_year = datetime.datetime(year=dt.year, month=1, day=1)
    year_part = dt - start_year
    year_length = (
        datetime.datetime(
            year=dt.year,
            month=12,
            day=31,
            hour=23,
            minute=59,
            second=59
        ) - start_year
    )
    return dt.year + year_part / year_length


@nb.jit(nopython=True)
def medianMadFunction(x):
    """
    Compute median and median absolute deviation (MAD) for the vector x.

    Inputs:
    =======
    x: Input vector.

    Returns:
    =======
    Median and median absolute deviation (MAD) of the input vector.
    """
    xmed = 0
    xmad = 0

    if x.size:
        xmed = np.median(x)
        xmad = np.median(np.fabs(x - xmed))

    return [xmed, xmad]


@nb.jit(nopython=True)
def create_projected_velocity(
    x_in, y_in, dt, dt_median_flow, min_ref_unit_count=50,
    min_v0_threshold=50
):
    """
    Project vx and vy onto the median flow vector for the given spacial point.

    Inputs:
    =======
    x_in: x component of the velocity vector.
    y_in: y component of the velocity vector.
    dt:   Day separation vector.
    dt_median_flow: Bin edges for the median flow vector.
    min_ref_unit_count: Minimum number of points for dt_median_flow bin.
        Default is 50.
    min_v0_threshold: v0 threshold for slow moving areas. Those will be
        skipped for dt_max filter.

    Return:
    =======
    Projected velocity onto median flow vector.
    """
    x0_in = np.full_like(x_in, np.nan)

    x0_is_null = np.isnan(x_in)
    if np.all(x0_is_null):
        # No data to process
        return x0_in

    # Project vx and vy onto the median flow vector for dt <= 16;
    # if there is no data, then for dt <= 32, etc.
    ind = None
    valid = ~x0_is_null  # Number of valid points

    for each_dt in dt_median_flow:
        ind = (dt <= each_dt) & valid

        # Are there enough points?
        if ind.sum() >= min_ref_unit_count:
            break

    if ind.sum() == 0:
        # No data to process
        return x0_in

    # Make numba happy
    ind = ind.astype(np.bool_)

    vx0 = np.median(x_in[ind])
    vy0 = np.median(y_in[ind])
    v0 = np.sqrt(vx0**2 + vy0**2)

    # Skip dt filter for slow moving areas
    if v0 <= min_v0_threshold:
        # maxdt will be set to np.nan after vp>=20000 masking
        # (see dt_max_filter())
        x0_in = np.full_like(x_in, np.inf)
        return x0_in

    uv_x = vx0 / v0  # unit flow vector
    uv_y = vy0 / v0
    x0_in = x_in * uv_x + y_in * uv_y  # projected flow vector

    return x0_in


@nb.jit(nopython=True)
def dt_max_filter_iteration(vp, dt, dt_edge, min_ref_bin_count=50):
    """
    Filter one spacial point by dt (date separation) between the images.

    Inputs:
    =======
    vp: Projected velocity to median flow unit vector.
    dt: Day separation vector.
    dt_edge: Bin boundaries.
    min_ref_bin_count: Minimum number of points for the reference bin.
        Default is 50.

    Returns:
    ========
    maxdt:      Maximum dt as determined by the filter.
    is_invalid: Mask for invalid values of the input vector based on dt_max
        filter. These values will be excluded from the composites.
    """
    _num_bins = len(dt_edge) - 1

    # Make numba happy - use np.bool_ type
    is_invalid = np.zeros(len(dt), dtype=np.bool_)

    # There is no valid projected velocity vector:
    # should never be the case of all values being as np.inf
    # since we mask vp > 20000 before dt_max filter
    if np.all(np.isnan(vp)) or np.all(np.isinf(vp)):
        return np.nan, is_invalid

    mask = ~np.isnan(vp)
    x0 = vp[mask]
    x0_dt = dt[mask]

    # Group data values by identified bins "manually":
    # since data is sorted by date_dt, we can identify index boundaries
    # for each bin within the "date_dt" vector
    bin_index = np.searchsorted(x0_dt, dt_edge)

    # Collect indices for bins that represent current x0_dt

    # Pre-allocate with maximum possible size
    xmed = np.empty(_num_bins)
    xmad = np.empty(_num_bins)
    count = np.empty(_num_bins, dtype=np.int32)
    dt_bin_indices = np.empty(_num_bins, dtype=np.int32)
    valid_bins = 0

    for bin_i in range(0, _num_bins):
        # if bin_index[bin_i] and bin_index[bin_i+1] are the same,
        # there are no values for the bin, skip it
        start_idx = bin_index[bin_i]
        end_idx = bin_index[bin_i + 1]

        if start_idx != end_idx:
            bin_data = x0[start_idx:end_idx]
            xmed[valid_bins], xmad[valid_bins] = medianMadFunction(bin_data)
            count[valid_bins] = end_idx - start_idx + 1
            dt_bin_indices[valid_bins] = bin_i
            valid_bins += 1

    # Check if populations overlap (use first, smallest dt, bin as reference)

    # Trim arrays to actual size
    xmed = xmed[:valid_bins]
    std_dev = xmad[:valid_bins]
    count = count[:valid_bins]
    dt_bin_indices = dt_bin_indices[:valid_bins]

    # Calculate bounds
    minBound = xmed - std_dev
    maxBound = xmed + std_dev

    # Find first valid bin with minimum acceptable number of points
    ref_index, = np.where(count >= min_ref_bin_count)

    # If no such valid bin exists, just consider first bin where
    if ref_index.size == 0:
        ref_index, = np.where(maxBound != 0)

    # Not enough data to proceed
    if ref_index.size == 0:
        return np.nan, is_invalid

    ref_index = ref_index[0]

    exclude = (minBound > maxBound[ref_index]) | (maxBound < minBound[ref_index])

    maxdt = np.nan

    if np.any(exclude):
        dt_bin_indices = dt_bin_indices[exclude]
        maxdt = dt_edge[dt_bin_indices].min()
        is_invalid = dt > maxdt

    return maxdt, is_invalid


@nb.jit(nopython=True, parallel=True)
def dt_max_filter_parallel(vp, dt, dt_edge, min_ref_bin_count=50):
    """
    Parallelized core computation for dt_max filtering of the data.
    This handles the numerical computation only.

    Inputs:
    =======
    vp: Projected velocity to median flow unit vector.
    dt: Day separation vector.
    dt_edge: Bin boundaries.
    min_ref_bin_count: Minimum number of points for the reference bin.
        Default is 50.
    """
    y_len, x_len, t_len = vp.shape
    maxdt = np.full((y_len, x_len), np.nan)
    invalid = np.zeros((y_len, x_len, t_len), dtype=np.bool_)

    # Parallel loop over spatial dimensions
    for j_index in nb.prange(y_len):
        for i_index in range(x_len):
            maxdt[j_index, i_index], invalid[j_index, i_index, :] = \
                dt_max_filter_iteration(
                    vp[j_index, i_index],
                    dt,
                    dt_edge,
                    min_ref_bin_count
                )

    return invalid, maxdt


def dt_max_filter(
    vp, dt, current_sensor_group, exclude_sensor_groups, min_ref_bin_count=50
):
    """
    Filter data cube by dt (date separation).

    Inputs:
    =======
    vp: Projected velocity to median flow unit vector.
    dt: Day separation vector.
    current_sensor_group: Sensor group of the current datacube.
    exclude_sensor_groups: Sensor groups to exclude from the analysis.
    min_ref_bin_count: Minimum number of points for the reference bin.
        Default is 50.

    Returns:
    ========
    invalid: Array indicating invalid points based on dt_max filter which
        will be excluded from the composites.
    maxdt: Maximum time separation for each valid cell.
    sensor_include: Array indicating which sensors are included.
    """
    y_len, x_len, t_len = vp.shape

    # First pass: identify cells to exclude (vectorized where possible)
    sensor_include = np.ones((y_len, x_len))
    exclude_mask = np.zeros((y_len, x_len), dtype=np.bool_)

    # Check exclusions
    for j in range(y_len):
        for i in range(x_len):
            if len(exclude_sensor_groups[j, i]) > 0 and \
                    current_sensor_group in exclude_sensor_groups[j, i]:
                exclude_mask[j, i] = True
                sensor_include[j, i] = 0

    # Create working copy for cells that need processing
    vp_work = vp.copy()
    vp_work[exclude_mask, :] = np.nan  # Mark excluded cells

    # Run parallelized computation
    invalid, maxdt = dt_max_filter_parallel(
        vp_work, dt, DT_EDGE, min_ref_bin_count
    )

    # Apply exclusion mask to results
    invalid[exclude_mask, :] = True
    maxdt[exclude_mask] = np.nan

    return invalid, maxdt, sensor_include


@nb.jit(nopython=True)
def weighted_std(values, weights):
    """
    Return weighted standard deviation.

    Reference: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    Inputs:
    =======
    values: Array of values.
    weights: Array of weights.

    Returns:
    ========
    Weighted standard deviation of the input values.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)


@nb.jit(nopython=True)
def create_M(y1, start_year, stop_year, dyr):
    """
    Make matrix of percentages of years corresponding to each displacement
    measurement.

    Inputs:
    =======
    y1: Array of years.
    start_year: Decimal year corresponding to the start date.
    stop_year: Decimal year corresponding to the stop date.
    dyr: Array of year fractions for each displacement measurement.

    Returns:
    ========
    M: Matrix of percentages of years corresponding to each displacement
    measurement.
    """
    M = np.zeros((len(dyr), len(y1)))

    # Parallel loop through each year:
    for k in nb.prange(len(y1)):
        # Set all measurements that begin before the first day of the year
        # and end after the last day of the year to 1:
        y1_value = y1[k]
        y1_next_value = y1_value + 1

        ind = np.logical_and(
            start_year <= y1_value,
            stop_year >= y1_next_value
        )
        M[ind, k] = 1

        # Within year:
        ind = np.logical_and(
            start_year >= y1_value,
            stop_year < y1_next_value
        )
        M[ind, k] = dyr[ind]

        # Started before the beginning of the year and ends during the year:
        ind = np.logical_and(
            start_year < y1_value,
            np.logical_and(stop_year >= y1_value, stop_year < y1_next_value)
        )
        M[ind, k] = stop_year[ind] - y1_value

        # Started during the year and ends the next year:
        ind = np.logical_and(
            start_year >= y1_value,
            np.logical_and(
                start_year < y1_next_value, stop_year >= y1_next_value
            )
        )
        M[ind, k] = y1_next_value - start_year[ind]

    return M


@nb.jit(nopython=True)
def create_D_components(start_decimal_year, stop_decimal_year):
    """
    Precompute D matrix components for LSQ fit.

    Inputs:
    =======
    start_decimal_year: Decimal year corresponding to the start date.
    stop_decimal_year: Decimal year corresponding to the stop date.

    Returns:
    ========
    D_cos, D_sin: Precomputed D matrix components for LSQ fit.
    """
    # Pre-compute trigonometric terms using vectorized computations
    years = np.column_stack((start_decimal_year, stop_decimal_year))

    cos_terms = np.cos(TWO_PI * years)
    sin_terms = np.sin(TWO_PI * years)

    # Pre-compute D matrix components
    # cos_start - cos_stop
    D_cos = (cos_terms[:, 0] - cos_terms[:, 1]) / TWO_PI
    # sin_stop - sin_start
    D_sin = (sin_terms[:, 1] - sin_terms[:, 0]) / TWO_PI

    return D_cos, D_sin


# ATTN: if using "rcond=None" to lstsq(), need to disable numba as its wrapper
# for lstsq does not support "rcond=None" input parameter for LSQ fit
# Getting convergence failures with numba enabled for some RGI05A cubes
@nb.jit(nopython=True)
def itslive_lsqfit_iteration(var_name, d_cos, d_sin, M, w_d, d_obs):
    """
    LSQ fit iteration for a single spacial point of the datacube.
    """
    #
    # LSQ fit iteration
    #
    # Displacement Vandermonde matrix: (these are displacements! not velocities,
    # so this matrix is just the definite integral wrt time of
    # a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
    # D = [(cos(2*pi*yr(:,1)) - cos(2*pi*yr(:,2)))./(2*pi).*(M>0)
    #      (sin(2*pi*yr(:,2)) - sin(2*pi*yr(:,1)))./(2*pi).*(M>0) M];
    D = np.stack((d_cos, d_sin), axis=-1)
    D = np.concatenate((D, M), axis=1)

    # Solve for coefficients of each column in the Vandermonde:
    # p = np.linalg.lstsq(w_d.reshape((len(w_d), 1)) * D, w_d*d_obs, rcond=None)[0]
    p = np.linalg.lstsq(
        w_d.reshape((len(w_d), 1)) * D, w_d * d_obs, rcond=1e-15
    )[0]

    # Goodness of fit: modeled displacements (m)
    d_model = D @ p

    return (p, d_model)


@nb.jit(nopython=True)
def itersect_years(all_years, select_years):
    """
    Get indices of "select_years" into "all_years" array.
    This is to replace built-in numpy.intersect1d() which does not work with
    numba.

    Inputs:
    =======
    all_years: Array of all years.
    select_years: Tuple of years to select. Have to use tuple as python's
        lists are not supported by numba.
    """
    lookup_table = {v: i for i, v in enumerate(all_years)}
    return np.array(
        [lookup_table[each] for each in select_years if each in lookup_table]
    )


@nb.jit(nopython=True)
def init_lsq_fit1(
    v_input, v_err_input, d_cos, d_sin, start_dec_year, stop_dec_year,
    dec_dt, M_input
):
    """
    First step of variables initialization for LSQ fit.

    Inputs:
    =======
    v_input: Velocity input vector.
    v_err_input: Velocity error input vector.
    d_cos: Precomputed D matrix cos component for LSQ fit.
    d_sin: Precomputed D matrix sin component for LSQ fit.
    start_dec_year: Decimal year corresponding to the start date.
    stop_dec_year: Decimal year corresponding to the stop date.
    dec_dt: Time step in decimal years.
    M_input: Input matrix.

    Returns:
    ========
    results_valid: Boolean flag set to True if results are valid, False
        otherwise meaning that further computation should be skipped.
        Computations should be skipped if identified data validity mask is
        empty which results in no data to be processed.
        This flag has to be introduced in order to use numba compilation
        otherwise numba-compiled code fails when using empty mask (pure
        Python code does not).
    d_cos, d_sin, start_year, stop_year, v_in, v_err_in, dyr, totalnum, M_in:
        Filtered by data validity mask and sorted by mid_date all input data
        variables.
    """
    # Ensure we're starting with finite data
    isf_mask = np.isfinite(v_input) & np.isfinite(v_err_input)
    results_valid = np.any(isf_mask)

    if not results_valid:
        # All results will be ignored, but they must match in type to valid
        # returned esults to keep numba happy, so just return input-like data
        # Can't use input variables as they are read-only which makes numba
        # unhappy
        dy_out = np.zeros_like(start_dec_year)

        # Return dummy data
        return (
            results_valid,
            dy_out,
            dy_out,
            dy_out,
            dy_out,
            np.zeros_like(v_input),
            np.zeros_like(v_err_input),
            np.zeros_like(dec_dt),
            0,
            np.zeros_like(M_input)
        )

    d_cos_filtered = d_cos[isf_mask]
    d_sin_filtered = d_sin[isf_mask]
    start_year = start_dec_year[isf_mask]
    stop_year = stop_dec_year[isf_mask]
    # dt in decimal years
    dyr = dec_dt[isf_mask]

    v_in = v_input[isf_mask]
    v_err_in = v_err_input[isf_mask]
    M_in = M_input[isf_mask]

    totalnum = len(start_year)

    # Sort arrays based on the mid_date
    mid_date = start_year + (stop_year - start_year) / 2.0
    sort_indices = np.argsort(mid_date)

    # Sort inputs
    d_cos_filtered = d_cos_filtered[sort_indices]
    d_sin_filtered = d_sin_filtered[sort_indices]
    start_year = start_year[sort_indices]
    stop_year = stop_year[sort_indices]
    dyr = dyr[sort_indices]

    v_in = v_in[sort_indices]
    v_err_in = v_err_in[sort_indices]
    M_in = M_in[sort_indices]

    return (
        results_valid,
        d_cos_filtered,
        d_sin_filtered,
        start_year,
        stop_year,
        v_in,
        v_err_in,
        dyr,
        totalnum,
        M_in
    )


@nb.jit(nopython=True)
def init_lsq_fit2(
    v_median, v_input, v_err_input, d_cos, d_sin,
    start_dec_year, stop_dec_year, dec_dt, all_years,
    M_input, mad_thresh, mad_std_ratio, sigma
):
    """
    Second step of variables initialization for the LSQ fit.

    Inputs:
    =======
    v_median: Median filtered velocity input vector.
    v_input: Velocity input vector.
    v_err_input: Velocity error input vector.
    d_cos: Precomputed D matrix cos component for LSQ fit.
    d_sin: Precomputed D matrix sin component for LSQ fit.
    start_dec_year: Decimal year corresponding to the start date.
    stop_dec_year: Decimal year corresponding to the stop date.
    dec_dt: Time step in decimal years.
    all_years: Array of all years.
    M_input: Input M matrix.
    mad_thresh: Threshold for the MAD filter.
    mad_std_ratio: Ratio to convert MAD to standard deviation (approximately 1.4826).
    sigma: Sigma multiplier for the MAD filter.

    Returns:
    ========
    results_valid: Boolean flag set to True if results are valid, False
        otherwise meaning that further computation should be skipped.
        Computations should be skipped if identified data validity mask is
        empty which results in no data to be processed.
        This flag has to be introduced in order to use numba compilation,
        otherwise numba-compiled code fails when using empty mask (pure
        Python code does not).
    d_cos_filtered, d_sin_filtered, start_year, stop_year, v_in, v_err_in,
        dyr, w_v, w_d, d_obs, y1, M_in: Filtered by data validity mask and
        pre-processed for LSQ fit input data variables.
    """
    _num_valid_points = 30

    # Remove outliers based on MAD filter for v: subtract from v to get
    # residual
    v_residual = np.abs(v_input - v_median)

    # Take median of residual, multiply median of residual * 1.4826 = sigma
    v_sigma = np.median(v_residual) * mad_std_ratio

    non_outlier_mask = ~(v_residual > (sigma * mad_thresh * v_sigma))

    # If less than _num_valid_points don't do the fit: not enough observations
    results_valid = (np.sum(non_outlier_mask) >= _num_valid_points)

    if not results_valid:
        # All results will be ignored, but they must match in type to valid returned
        # results to keep numba happy.
        # Can't use input variables as they are read-only which makes numba unhappy
        v_out = np.zeros_like(v_input)
        v_err_out = np.zeros_like(v_err_input)
        dy_out = np.zeros_like(start_dec_year)

        return (
            results_valid,
            dy_out,
            dy_out,
            dy_out,
            dy_out,
            v_out,
            v_err_out,
            np.zeros_like(dec_dt),
            v_err_out,
            v_err_out.astype(np.float64),
            v_out,
            np.arange(1, 2),
            np.zeros_like(M_input)
        )

    # remove ouliers from v_in, v_error_in, start_dec_year, stop_dec_year,
    # d_cos, d_sin, dec_dt, M_input
    d_cos_filtered = d_cos[non_outlier_mask]
    d_sin_filtered = d_sin[non_outlier_mask]

    start_year = start_dec_year[non_outlier_mask]
    stop_year = stop_dec_year[non_outlier_mask]
    dyr = dec_dt[non_outlier_mask]

    v_in = v_input[non_outlier_mask]
    v_err_in = v_err_input[non_outlier_mask]
    M_in = M_input[non_outlier_mask]

    # Weights for velocities
    w_v = 1 / (v_err_in**2)

    # Weights (correspond to displacement error, not velocity error):
    # Matlab comment: Not squared because the p= line below would then have to include
    # sqrt(w) on both accounts
    w_d = 1 / (v_err_in * dyr)
    # logging.info(f"w_d.shape: {w_d.shape}")

    # Observed displacement in meters
    d_obs = v_in * dyr
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

    return (
        results_valid, d_cos_filtered, d_sin_filtered, start_year, stop_year,
        v_in, v_err_in, dyr, w_v, w_d, d_obs, y1, M_in
    )


@nb.jit(nopython=True)
def create_v0_years_mask(start_year, stop_year, v0_years_start, v0_years_end):
    """
    Create a mask based on the median date which falls within v0_years.

    Inputs:
    =======
    start_year: Decimal year corresponding to the start date.
    stop_year: Decimal year corresponding to the stop date.
    v0_years_start: Start year within which middle date should fall into.
    v0_years_end: Stop year within which middle date should fall into.
    """
    #  Reduce number of image pairs only to the provided range:
    # v0_years[0] <= mid_date < v0_years[-1]+1
    mid_date = start_year + (stop_year - start_year) / 2.0

    v0_year_mask = (mid_date >= v0_years_start) & (mid_date < (v0_years_end + 1))
    return v0_year_mask


def itslive_lsqfit_annual(
    var_name,
    v_input,
    v_err_input,
    d_cos_input,
    d_sin_input,
    start_dec_year,
    stop_dec_year,
    dec_dt,
    all_years,
    M_input,
    mad_std_ratio,
    v0_years,
    center_date,
    mean,  # outputs to populate
    error,
    count,
):
    """
    Computes the amplitude and phase of seasonal velocity variability,
    and also gives interannual variability.

    From original Matlab code:
    % [A,ph,A_err,t_int,v_int,v_int_err,N_int] = itslive_sinefit_lsq(t,v,v_err)
    % also returns the standard deviation of amplitude residuals A_err. Outputs
    % t_int and v_int describe interannual velocity variability, and can then
    % be used to reconstruct a continuous time series, as shown below. Output
    % Output N_int is the number of image pairs that contribute to the annual mean
    % v_int of each year. The output |v_int_err| is a formal estimate of error
    % in the v_int.
    %
    %% Author Info
    % Chad A. Greene, Jan 2020.
    %

    Inputs/Outputs:
    =======================================================================
    var_name: Name of the variable being processed (vx or vy).
    v_input: Velocity input vector.
    v_err_input: Velocity error input vector.
    d_cos_input: Precomputed D matrix component for LSQ fit.
    d_sin_input: Precomputed D matrix component for LSQ fit.
    start_dec_year: Decimal year corresponding to the start date.
    stop_dec_year: Decimal year corresponding to the stop date.
    dec_dt: Day separation vector in decimal years.
    all_years: Array of all years.
    M_input: Matrix of percentages of years corresponding to each displacement
        measurement.
    mad_std_ratio: Ratio to convert MAD to standard deviation (approximately
        1.4826 for normal distribution).
    center_date: Date to use as the intercept for the weighted linear fit.
    mean: Output data variable to populate with the mean velocity.
    error: Output data variable to populate with the velocity error.
    count: Output data variable to populate with the count of image pairs.
    """
    # Filter parameters for lsq fit for outlier rejections
    _mad_thresh = 6
    _mad_filter_iterations = 1

    # Apply MAD filter to input v
    _mad_kernel_size = 15

    results_valid = True

    results_valid, \
        d_cos_1, \
        d_sin_1, \
        start_year_1, \
        stop_year_1, \
        v_1, \
        v_err_1, \
        dyr_1, \
        totalnum, \
        M_1 = init_lsq_fit1(
            v_input, v_err_input, d_cos_input, d_sin_input,
            start_dec_year, stop_dec_year, dec_dt, M_input
        )

    empty_results = []

    if not results_valid:
        # There is no data to process, exit
        return (results_valid, empty_results)

    # Compute outside of numba-compiled code as numba does not support a lot of scipy
    # functionality
    # Apply 15-point moving median to v, subtract from v to get residual
    v_median = ndimage.median_filter(v_1, _mad_kernel_size)

    # "Bandaid" solution to the LSQ fit convergence exception we get randomly - seems to depend
    # on the platform (and possibly some package versions???) the composites are being generated on.
    lsq_fit_converged = False
    max_number_attempts = 10
    number_of_attempts = 0
    sigma = 2.0
    sigma_delta = 0.05

    p = None
    d_model = None

    results_valid = True
    start_year = None
    stop_year = None
    v = None
    v_err = None
    dyr = None
    w_v = None
    w_d = None
    d_obs = None
    y1 = None
    M = None
    d_cos = None
    d_sin = None

    while (lsq_fit_converged is False) and (number_of_attempts < max_number_attempts):
        try:
            results_valid, \
                d_cos, \
                d_sin, \
                start_year, \
                stop_year, \
                v, v_err, dyr, \
                w_v, w_d, d_obs, y1, M = init_lsq_fit2(
                    v_median, v_1, v_err_1, d_cos_1, d_sin_1,
                    start_year_1, stop_year_1, dyr_1, all_years,
                    M_1, _mad_thresh, mad_std_ratio, sigma
                )

            if not results_valid:
                # There is no data to process, exit
                return (results_valid, empty_results)

            # Filter sum of each column
            hasdata = M.sum(axis=0) > 0
            y1 = y1[hasdata]
            M = M[:, hasdata]

            # LSQFit iterations

            # Last iteration of LSQFit should always skip the outlier filter
            last_iteration = _mad_filter_iterations - 1

            for i in range(0, _mad_filter_iterations):
                # Displacement Vandermonde matrix: (these are displacements!
                # not velocities, so this matrix is just the definite integral
                # wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
                p, d_model = itslive_lsqfit_iteration(
                    var_name, d_cos, d_sin, M, w_d, d_obs
                )

                if i < last_iteration:
                    # Divide by dt to avoid penalizing long dt [asg]
                    d_resid = np.abs(d_obs - d_model) / dyr

                    # Robust standard deviation of errors, using median absolute deviation
                    d_sigma = np.median(d_resid) * mad_std_ratio

                    outliers = d_resid > (_mad_thresh * d_sigma)
                    if np.all(outliers):
                        # All are outliers, return from the function
                        results_valid = False
                        return (results_valid, empty_results)

                    if (outliers.sum() / totalnum) < 0.01:
                        # There are less than 1% outliers, skip the rest of
                        # iterations if it's not the last iteration
                        break

                    # Remove outliers
                    non_outlier_mask = ~outliers
                    d_cos = d_cos[non_outlier_mask]
                    d_sin = d_sin[non_outlier_mask]
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
                        return (results_valid, empty_results)

                    y1 = y1[hasdata]
                    M = M[:, hasdata]

            lsq_fit_converged = True

        except np.linalg.LinAlgError:
            number_of_attempts += 1
            logging.info(
                f'Got np.linalg.LinAlgError exception using sigma={sigma}, '
                f'increment sigma by {sigma_delta}, retry #{number_of_attempts}...'
            )
            sigma += sigma_delta
            time.sleep(5)

            if number_of_attempts == max_number_attempts:
                # Re-raise exception once achieved maximum number of retries
                raise

        except ValueError as exc:
            # numba raises ValueError exception when LSQ fit does not converge
            if "Internal algorithm failed to converge." not in str(exc):
                # Re-raise exception if it's not the LSQ fit convergence one
                raise

            number_of_attempts += 1
            logging.info(
                f'Got ValueError exception "{exc}" using sigma={sigma}, '
                f'increment sigma by {sigma_delta}, retry #{number_of_attempts}...'
            )
            sigma += sigma_delta
            time.sleep(5)

            if number_of_attempts == max_number_attempts:
                # Re-raise exception once achieved maximum number of retries
                raise

    # WAS: v_int = p[2*Nyrs:]
    v_int = p[2:]

    # Number of equivalent image pairs per year:
    # (1 image pair equivalent means a full year of data.
    # It takes about 23 16-day image pairs to make 1 year equivalent image pair.)
    N_int = (M > 0).sum(axis=0)

    # Reshape array to have the same number of dimensions as M for multiplication
    w_v = w_v.reshape((1, w_v.shape[0]))

    v_int_err = 1 / np.sqrt((w_v @ M).sum(axis=0))

    # Identify year's indices to assign return values to in "final" composite
    # variables
    ind = itersect_years(all_years, tuple(y1))

    # On return: amp1, phase1, sigma1, t_int1, xmean1, err1, cnt1
    # amplitude[ind] = A
    # phase[ind] = ph
    # sigma[ind] = A_err

    mean[ind] = v_int
    error[ind] = v_int_err
    count[ind] = N_int

    offset, slope, se = np.nan, np.nan, np.nan

    # Reduce input data to specified years to compute climatological values
    v0_ind = itersect_years(y1, tuple(v0_years))

    if v0_ind.size != 0:
        yr = np.array(
            [
                decimal_year(
                    datetime.datetime(each, center_date.month, center_date.day)
                ) for each in y1[v0_ind]
            ]
        )
        yr0 = decimal_year(center_date)
        yr = yr - yr0

        offset, slope, se = weighted_linear_fit(
            yr, mean[ind][v0_ind], error[ind][v0_ind]
        )

    # If there is more than 1 iterations for LSQ fit invoked above, then all
    # data vars (start_year, stop_year, dyr, etc.)
    # might be reduced by "non_outlier_mask" mask in last iteration.
    # Therefore, the v0_year_mask must be applied to the
    # initial values of these data variables. Confirm with Alex that it's
    # the case. For now just raise an exception if more than 1 iterations
    # are required.
    if _mad_filter_iterations > 1:
        raise RuntimeError(
            f'_mad_filter_iterations={_mad_filter_iterations}: need to '
            f'apply v0_years mask to original values of start_year, '
            f'stop_year, dyr, etc. for next LSQ fit as these values might '
            f'have been reduced by "non_outlier_mask" above.'
        )

    #  Reduce number of image pairs only to the provided range:
    # v0_years[0] <= mid_date < v0_years[-1]+1
    _v0_year_mask = create_v0_years_mask(
        start_year, stop_year, v0_years[0], v0_years[-1]
    )

    d_cos = d_cos[_v0_year_mask]
    d_sin = d_sin[_v0_year_mask]
    start_year = start_year[_v0_year_mask]
    stop_year = stop_year[_v0_year_mask]
    dyr = dyr[_v0_year_mask]
    d_obs = d_obs[_v0_year_mask]
    w_d = w_d[_v0_year_mask]
    M = M[_v0_year_mask]

    # Filter sum of each column
    hasdata = M.sum(axis=0) > 0
    y1 = y1[hasdata]
    M = M[:, hasdata]

    count_image_pairs = np.nan
    A, ph, amp_error = np.nan, np.nan, np.nan

    if np.any(hasdata):
        # Last iteration of LSQ should always skip the outlier filter
        last_iteration = _mad_filter_iterations - 1

        for i in range(0, _mad_filter_iterations):
            # Displacement Vandermonde matrix: (these are displacements!
            # not velocities, so this matrix is just the definite integral
            # wrt time of a*sin(2*pi*yr)+b*cos(2*pi*yr)+c.
            p, d_model = itslive_lsqfit_iteration(
                var_name, d_cos, d_sin, M, w_d, d_obs
            )

            if i < last_iteration:
                # Divide by dt to avoid penalizing long dt [asg]
                d_resid = np.abs(d_obs - d_model) / dyr

                # Robust standard deviation of errors, using median
                # absolute deviation
                d_sigma = np.median(d_resid) * mad_std_ratio

                outliers = d_resid > (_mad_thresh * d_sigma)
                if np.all(outliers):
                    # All are outliers, return from the function
                    results_valid = False
                    return (results_valid, empty_results)

                if (outliers.sum() / totalnum) < 0.01:
                    # There are less than 1% outliers, skip the rest of
                    #  iterations if it's not the last iteration
                    break

                # Remove outliers
                non_outlier_mask = ~outliers
                d_cos = d_cos[non_outlier_mask]
                d_sin = d_sin[non_outlier_mask]
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
                    # Since we are throwing away everything,
                    # report all as outliers
                    results_valid = False
                    return (results_valid, empty_results)

                y1 = y1[hasdata]
                M = M[:, hasdata]

        # logging.info(
        #   f'Reducing count_image_pairs from {count_image_pairs} to '
        #   f'{M[_v0_year_mask, :].shape[0]}'
        # )
        count_image_pairs = M.shape[0]

        # Either v0_years are not provided or second LSQ fit was not invoked
        # when v0_years are provided.
        # Convert coefficients to amplitude and phase of a single sinusoid:
        Nyrs = len(y1)

        # Amplitude of sinusoid from trig identity
        # a*sin(t) + b*cos(t) = d*sin(t+phi),
        # where d=hypot(a,b) and phi=atan2(b,a).
        # WAS: A = np.hypot(p[0:Nyrs], p[Nyrs:2*Nyrs])
        A = np.hypot(p[0], p[1])

        # phase in radians
        # ph_rad = np.arctan2(p[Nyrs:2*Nyrs], p[0:Nyrs])
        ph_rad = np.arctan2(p[1], p[0])

        # phase converted such that it reflects the day when value is maximized
        ph = 365.25 * ((0.25 - ph_rad / TWO_PI) % 1)

        # A_err is the *velocity* (not displacement) error, which is the
        # displacement error divided by the weighted mean dt:
        # WAS: A_err = np.full_like(A, np.nan)
        A_err = np.full((Nyrs), np.nan)

        for k in range(Nyrs):
            ind = M[:, k] > 0

            # asg replaced call to wmean
            _w_d_ind = w_d[ind]
            A_err[k] = weighted_std(
                d_obs[ind] - d_model[ind], _w_d_ind
            ) / ((_w_d_ind * dyr[ind]).sum() / _w_d_ind.sum())

        # Compute climatology amplitude error based on annual values
        amp_error = np.sqrt((A_err**2).sum()) / (Nyrs - 1)

    return (
        results_valid,
        [A, amp_error, ph, offset, slope, se, count_image_pairs]
    )


@nb.jit(nopython=True)
def annual_magnitude(
    vx_fit,
    vy_fit,
    vx_fit_err,
    vy_fit_err,
    vx_fit_count,
    vy_fit_count
):
    """
    Computes and returns the annual mean, error, count, and outlier fraction
    from component values projected on the unit flow vector defined by vx0 and
    vy0.

    Inputs:
    -------
    vx_fit: annual mean flow in x direction
    vy_fit: annual mean flow in y direction
    vx_fit_err: error in annual mean flow in x direction
    vy_fit_err: error in annual mean flow in y direction
    vx_fit_count: number of values used to determine annual mean flow in x direction
    vy_fit_count: number of values used to determine annual mean flow in y direction

    Outputs:
    --------
    self.mean.v[start_y:stop_y, start_x:stop_x, :]
    self.error.v[start_y:stop_y, start_x:stop_x, :]
    self.count.v[start_y:stop_y, start_x:stop_x, :]

    Outputs map to:
        * v_fit
        * v_fit_err
        * v_fit_count

    """
    # solve for velocity magnitude
    v_fit = np.sqrt(vx_fit**2 + vy_fit**2)  # velocity magnitude

    # Compute v_fit_error like autoRIFT does:
    # V_error = np.sqrt((vx_error * VX / V)**2 + (vy_error * VY / V)**2)
    v_fit_err = (vx_fit_err * vx_fit)**2
    v_fit_err += (vy_fit_err * vy_fit)**2
    v_fit_err = np.sqrt(v_fit_err)
    v_fit_err /= np.abs(v_fit)

    v_fit_count = np.ceil((vx_fit_count + vy_fit_count) / 2)

    return v_fit, v_fit_err, v_fit_count


# No need for numba as all is done in Numpy internally
# @nb.jit(nopython=True, parallel=True)
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
    vy_se,
    v_limit
):
    """
    Computes and populates the mean, trend, seasonal amplitude, error in
    seasonal amplitude, and seasonal phase from component values projected
    on the unit flow vector defined by vx0 and vy0.

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
    vx_se: standard error in x direction
    vy_se: standard error in y direction
    v_limit: maximum limit for the flow magnitude

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
    # solve for velocity magnitude and acceleration
    # [do this using vx and vy as to not bias the result due to the Rician
    # distribution of v]
    v = np.sqrt(vx0**2 + vy0**2)  # velocity magnitude

    invalid_mask = (v >= v_limit)
    if np.sum(invalid_mask) > 0:
        # Since it's invalid v0, report all output as invalid
        v[invalid_mask] = np.nan
        vx0[invalid_mask] = np.nan
        vy0[invalid_mask] = np.nan

    uv_x = vx0 / v  # unit flow vector in x direction
    uv_y = vy0 / v  # unit flow vector in y direction

    dv_dt = dvx_dt * uv_x  # flow acceleration in direction of unit flow vector
    dv_dt += dvy_dt * uv_y

    # flow acceleration in direction of unit flow vector, take absolute values
    v_amp_err = np.abs(vx_amp_err) * np.abs(uv_x)
    v_amp_err += np.abs(vy_amp_err) * np.abs(uv_y)

    v_se = np.full_like(vx_se, np.nan)
    v_se = vx_se * np.abs(uv_x)
    v_se += vy_se * np.abs(uv_y)

    # Analytical solution for amplitude and phase
    # -------------------------------------------
    # Per Slack chat with Alex on July 12, 2022:
    # "we need to rotate the vx/y_amp and vx/y_phase into the direction of v,
    # which is defined by vx0 and vy0. If you replace the rotation matrix in
    # the sudo [Matlab] code (coordinate projection rotation) by the rotation matrix
    # defined by vx0 and vy0 then one of the rotated component is in the
    # direction of v0 and the other is perpendicular to v0.
    # We only want to retain the component that is in the direction of v0."
    vx_phase_rad = vx_phase / 365.25
    vy_phase_rad = vy_phase / 365.25

    # Convert degrees to radians as numpy trig. functions take angles in radians
    vx_phase_rad *= TWO_PI
    vy_phase_rad *= TWO_PI

    # Don't use np.nan values in calculations to avoid warnings
    valid_mask = (~np.isnan(vx_phase_rad)) & (~np.isnan(vy_phase_rad))

    # Compute theta rotation angle
    # theta = arctan(vy0/vx0), since sin(theta)=vy0 and cos(theta)=vx0,
    # can't just use vy0 and vx0 values instead of sin/cos as they are not normalized
    theta = np.full_like(vx_phase_rad, np.nan)
    theta[valid_mask] = np.arctan2(vy0[valid_mask], vx0[valid_mask])

    mask = (theta < 0)
    if np.any(mask):
        # logging.info(f'Got negative theta, converting to positive values')
        theta[mask] += TWO_PI

    # Find negative values
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    A1 = vx_amp * cos_theta
    B1 = vy_amp * sin_theta

    # Matlab prototype code:
    # vx_amp_r   =   hypot(A1.*cosd(vx_phase_deg) + B1.*cosd(vy_phase_deg),  A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg));
    # vx_phase_r = atan2d((A1.*sind(vx_phase_deg) + B1.*sind(vy_phase_deg)),(A1.*cosd(vx_phase_deg) + B1.*(cosd(vy_phase_deg))));

    # We want to retain the component only in the direction of v0,
    # which becomes new v_amp and v_phase
    v_amp = np.full_like(vx_amp, np.nan)
    v_phase = np.full_like(vx_phase, np.nan)

    v_amp[valid_mask] = np.hypot(
        A1[valid_mask] * np.cos(vx_phase_rad[valid_mask]) + \
            B1[valid_mask] * np.cos(vy_phase_rad[valid_mask]),
        A1[valid_mask] * np.sin(vx_phase_rad[valid_mask]) + \
            B1[valid_mask]*np.sin(vy_phase_rad[valid_mask])
    )
    # np.arctan2 returns phase in radians, convert to degrees
    v_phase[valid_mask] = np.arctan2(
        A1[valid_mask] * np.sin(vx_phase_rad[valid_mask]) + \
            B1[valid_mask]*np.sin(vy_phase_rad[valid_mask]),
        A1[valid_mask] * np.cos(vx_phase_rad[valid_mask]) + \
            B1[valid_mask] * np.cos(vy_phase_rad[valid_mask])
    ) * 180.0 / np.pi

    mask = v_amp < 0
    v_amp[mask] *= -1.0
    v_phase[mask] += 180

    mask = v_phase > 0
    v_phase[mask] = np.remainder(v_phase[mask], 360.0)
    mask = mask & (v_phase == 0)
    v_phase[mask] = 360.0

    # Convert all values to positive
    mask = v_phase < 0
    if np.any(mask):
        # logging.info(f'Got negative phase, converting to positive values')
        v_phase[mask] = np.remainder(v_phase[mask], -360.0)
        v_phase[mask] += 360.0

    # Since vx_phase and vy_phase are already shifted by 0.25 in original projection,
    # so we don't need to do it after rotation in direction of v0

    # Convert phase to the day of the year
    v_phase = v_phase * 365.25 / 360

    return v, dv_dt, v_amp, v_amp_err, v_phase, v_se


@nb.jit(nopython=True)
def weighted_linear_fit(yr, v, v_err):
    """
    Returns the offset, slope, and error for a weighted linear fit to v with
    an intercept of datetime0.

    Inputs:
    =======
    yr: decimal years offset by the CENTER_DATE
    v: estimates
    v_err: estimate errors

    Returns:
    ========
    offset: value of v at CENTER_DATE
    slope: rate of change of v with time
    error: formal estimate of error in slope
    """
    # yr = np.array([decimal_year(datetime.datetime(each, CENTER_DATE.month, CENTER_DATE.day)) for each in t])
    # yr0 = decimal_year(datetime0)
    # yr = yr - yr0

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
        # error = np.sqrt((v_err[valid]**2).sum())/(valid.sum()-1)

        error = np.nan
        if valid.sum() == 1:
            error = np.nan

        else:
            error = np.sqrt((v_err[valid]**2).sum()) / (valid.sum() - 1)

        return offset, slope, error

    # Normalize the weights per Chad's suggestion before LSQ fit:
    w_v = np.sqrt(w_v / np.mean(w_v))

    # create design matrix
    D = np.ones((len(yr), 2))
    D[:, 1] = yr

    # Solve for coefficients of each column in the Vandermonde:
    # w_v = w_v[valid]
    D = D[valid, :]

    # Julia: offset, slope = (w_v[valid].*D[valid,:]) \ (w_v[valid].*v[valid]);
    offset, slope = np.linalg.lstsq(
        w_v.reshape((len(w_v), 1)) * D, w_v * v[valid]
    )[0]
    # offset = p[0]
    # slope = p[1]

    # Julia: error = sqrt(sum(v_err[valid].^2))/(sum(valid)-1)
    error = np.sqrt((v_err[valid]**2).sum()) / (valid.sum() - 1)

    return offset, slope, error


class CompositeVariable:
    """
    Class to hold values for v, vx and vy components of the variables.
    """
    # Index order for data to be continuous in X dimension: [t, y, x]
    # since original order is [y, x, t]
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

    def to_uint16(self):
        """
        Convert data to uint16 datatype to store to output file.
        """
        self.v = utils.to_int_type(self.v)
        self.vx = utils.to_int_type(self.vx)
        self.vy = utils.to_int_type(self.vy)


# Currently processed datacube chunk
Chunk = collections.namedtuple(
    "Chunk",
    ['start_x', 'stop_x', 'x_len', 'start_y', 'stop_y', 'y_len']
)


class ITSLiveComposite:
    """
    Class to build annual and mean composites for ITS_LIVE datacubes.
    """
    VERSION = '1.0'

    # Flag is valid v[xy]_error_slow should be used in place of v[xy]_error
    USE_ERROR_SLOW = False

    # Only the following datacube variables are needed for composites/mosaics
    VARS = [
        Vars.vx,
        Vars.vy,
        Vars.flag_stable_shift,
        Vars.vx_stable_shift,
        Vars.vy_stable_shift,
        CompositeVars.vx_error,
        CompositeVars.vy_error,
        ImgPairInfo.date_dt,
        ImgPairInfo.date_center,
        ImgPairInfo.acquisition_date_img1,
        ImgPairInfo.acquisition_date_img2,
        ImgPairInfo.satellite_img1,
        ImgPairInfo.mission_img1
        # Vars.url  # for debugging only
    ]

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

    # Threshold for invalid velocity component value: value must be greater
    # than threshold
    V_LIMIT = 20000

    # Store generic cube metadata as static data as these are the same for
    # the whole cube
    YEARS = None
    DATE_DT = None

    START_DECIMAL_YEAR = None
    STOP_DECIMAL_YEAR = None
    DECIMAL_DT = None
    M = None
    D_COS = None
    D_SIN = None

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

    # Scale factor for amplitude comparison b/w LSQ fit using all data and
    # LSQ fit excluding S2 data
    LSQ_AMP_SCALE = 2

    # minimum difference in amplitude between LSQ fit results before removing S2 data
    LSQ_MIN_AMP_DIFF = 2

    # Shape file to locate ice masks files that correspond to the composite's EPSG code
    SHAPE_FILE = None

    # A list of years to include data for in computations of v*0 (offset), dv*_dt (slope) and v*0_error (std_error)
    # (done right after LSQ fit).
    # This flag is used for debugging purposes to understand the data.
    V0_YEARS = []

    def __init__(self, cube_store: str, s3_bucket: str):
        """
        Initialize composites.

        Inputs:
        =======
        cube_store: Location of the datacube to process.
        s3_bucket: S3 bucket that stores the datacube.
        """
        # Don't need to know skipped granules information for the purpose of composites
        read_skipped_granules_flag = False
        self.s3, self.cube_store_in, self.cube_ds, _ = ITSCube.init_input_store(
            cube_store,
            s3_bucket,
            read_skipped_granules=read_skipped_granules_flag
        )

        cube_projection = int(self.cube_ds.attrs[utils.OutputFormat.projection])

        # Find corresponding to EPSG ice masks files for the cube
        # Read land ice mask to be used for processing
        self.land_ice_mask, _ = shapefile.read_ice_mask(
            ITSLiveComposite.SHAPE_FILE, shapefile.LANDICE_2KM,
            self.cube_ds.x, self.cube_ds.y, cube_projection
        )

        # This is land ice coverage for the datacube
        # If landice and floating ice masks are provided by the datacube, just use them.
        # Otherwise, to support datacubes without ice masks, read them in and store
        # within composite.
        self.land_ice_mask_composite = None
        self.land_ice_mask_composite_url = None

        if shapefile.LANDICE in self.cube_ds:
            self.land_ice_mask_composite = \
                self.cube_ds[shapefile.LANDICE].values
            self.land_ice_mask_composite_url = \
                self.cube_ds[shapefile.LANDICE].attrs[utils.OutputFormat.url]

        else:
            self.land_ice_mask_composite, \
                self.land_ice_mask_composite_url = shapefile.read_ice_mask(
                    ITSLiveComposite.SHAPE_FILE, shapefile.LANDICE,
                    self.cube_ds.x, self.cube_ds.y, cube_projection
                )

        # This is floating ice coverage for the datacube
        self.floating_ice_mask_composite = None
        self.floating_ice_mask_composite_url = None

        if shapefile.FLOATINGICE in self.cube_ds:
            self.floating_ice_mask_composite = \
                self.cube_ds[shapefile.FLOATINGICE].values
            self.floating_ice_mask_composite_url = \
                self.cube_ds[shapefile.FLOATINGICE].attrs[utils.OutputFormat.url]

        else:
            self.floating_ice_mask_composite, \
                self.floating_ice_mask_composite_url = shapefile.read_ice_mask(
                    ITSLiveComposite.SHAPE_FILE, shapefile.FLOATINGICE,
                    self.cube_ds.x, self.cube_ds.y, cube_projection
                )

        # Read in only specific data variables
        # Need to sort data by dt to be able to filter with np.searchsorted()
        # (relies on date_dt vector being sorted)
        # Store "shallow" version of the cube for carrying over some of the metadata
        # when writing composites to the Zarr store
        cube_ds = self.cube_ds[ITSLiveComposite.VARS].sortby(
            ImgPairInfo.date_dt
        )
        logging.info(f'Original datacube sizes: {cube_ds.sizes}')

        # Setup StableShiftFilter: revert stable_shift offset and/or exclude
        # some granules.
        # Create valid granule mask and "need to adjust vx/vy" mask based on
        # the stable_shift filter
        start_time = timeit.default_timer()
        self.stable_shift_filter = sensorFilters.StableShiftFilter(cube_ds)

        # Remember datacube dimensions
        sizes = cube_ds.sizes

        # Update cube sizes with excluded granules
        self.cube_sizes = {
            utils.Coords.MID_DATE: sizes[utils.Coords.MID_DATE] - \
                self.stable_shift_filter.num_exclude_granules,
            utils.Coords.X: sizes[utils.Coords.X],
            utils.Coords.Y: sizes[utils.Coords.Y]
        }
        logging.info(f'Datacube sizes after StableShiftFilter: {self.cube_sizes}')

        ITSLiveComposite.MID_DATE_LEN = self.cube_sizes[utils.Coords.MID_DATE]

        # Need to keep original datacube dimensions to revert stable_shift, if any.
        # Then remove any granules for these data variables if any are identified
        # by the StableShiftFilter.
        self.data = cube_ds[[
            Vars.vx,
            Vars.vy
        ]]

        # From this point on initialize all data based on "reduced" by
        # StableShiftFilter datacube.
        # Only vx and vy data need to be read in full, reversed stable_shift
        # adjustment if any, and then reduced to the same size as reduced
        # cube_ds by removing granules as identified by the StableShiftFilter,
        # if any.

        # Add systematic error based on level of co-registration
        # Load Dask arrays before being able to modify their values
        logging.info("Add systematic error based on level of co-registration...")
        self.vx_error = self.stable_shift_filter.exclude(cube_ds.vx_error.values)

        # Note: we discovered that when there is very little stationary ground
        # (i.e. Greenland and Antarctica) the errors can be much too small
        # leading to poor composites. We therefore replaced the error with
        # slow error which does a better job at capturing true error at these
        # locations.
        if ITSLiveComposite.USE_ERROR_SLOW:
            # Replace vx_error with valid vx_error_slow
            vx_error_slow = self.stable_shift_filter.exclude(
                cube_ds.vx_error_slow.values
            )
            mask = ~np.isnan(vx_error_slow)
            logging.info(
                f'Replacing vx_error with vx_error_slow for {np.sum(mask)} '
                'values...'
            )
            self.vx_error[mask] = vx_error_slow[mask]

        self.vy_error = self.stable_shift_filter.exclude(cube_ds.vy_error.values)

        if ITSLiveComposite.USE_ERROR_SLOW:
            # Replace vy_error with valid vx_error_slow
            vy_error_slow = self.stable_shift_filter.exclude(
                cube_ds.vy_error_slow.values
            )
            mask = ~np.isnan(vy_error_slow)
            logging.info(
                f'Replacing vy_error with vy_error_slow for {np.sum(mask)} '
                'values'
            )
            self.vy_error[mask] = vy_error_slow[mask]

        stable_shift_values = self.stable_shift_filter.exclude(
            cube_ds[Vars.flag_stable_shift]
        )
        # NOTE V3: a code is written as a simple summation of errors. It might
        # be better to add it as a root sum of squares:
        # sqrt(v[xy]_error**2 + error**2). Something to consider for v3.
        for value, error in ITSLiveComposite.CO_REGISTRATION_ERROR.items():
            # mask = (stable_shift_values == value)
            mask = np.isin(stable_shift_values, value)
            self.vx_error[mask] += error
            self.vy_error[mask] += error

        # Re-size error arrays to the dimension of the velocity arrays
        self.vx_error = self.vx_error.reshape(1, 1, -1)
        self.vy_error = self.vy_error.reshape(1, 1, -1)

        # Images acquisition times and middle_date of each layer as datetime.datetime objects
        acq_datetime_img1 = [
            t.astype('M8[ms]').astype('O') for t in
            self.stable_shift_filter.exclude(
                cube_ds[ImgPairInfo.acquisition_date_img1].values
            )
        ]
        acq_datetime_img2 = [
            t.astype('M8[ms]').astype('O') for t in
            self.stable_shift_filter.exclude(
                cube_ds[ImgPairInfo.acquisition_date_img2].values
            )
        ]

        # Compute decimal year representation for start and end dates of each velocity pair
        ITSLiveComposite.START_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img1])
        ITSLiveComposite.STOP_DECIMAL_YEAR = np.array([decimal_year(each) for each in acq_datetime_img2])
        ITSLiveComposite.DECIMAL_DT = ITSLiveComposite.STOP_DECIMAL_YEAR - ITSLiveComposite.START_DECIMAL_YEAR

        # logging.info('DEBUG: Reading date values from Matlab files')
        # Read Matlab values instead of generating them internally: proves that slight
        # variation in date can cause deviation in Matlab vs. Python results
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
        ITSLiveComposite.YEARS = np.array(range(start_year, stop_year + 1))
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
        logging.info(
            f'Computed M (took {timeit.default_timer() - start_time} seconds)')

        start_time = timeit.default_timer()
        ITSLiveComposite.D_COS, ITSLiveComposite.D_SIN = create_D_components(
            ITSLiveComposite.START_DECIMAL_YEAR,
            ITSLiveComposite.STOP_DECIMAL_YEAR
        )
        logging.info(
            'Computed D matrix components '
            f'(took {timeit.default_timer() - start_time} seconds)'
        )

        # Day separation between images (sorted per cube.sortby() call above)
        ITSLiveComposite.DATE_DT = self.stable_shift_filter.exclude(
            cube_ds[ImgPairInfo.date_dt].values
        )

        # These data members will be set for each block of data being currently
        # processed ---> have to change the logic if want to parallelize blocks
        x_len = self.cube_sizes[utils.Coords.X]
        y_len = self.cube_sizes[utils.Coords.Y]

        # Allocate memory for composite outputs
        years_dims = (y_len, x_len, ITSLiveComposite.YEARS_LEN)

        self.error = CompositeVariable(years_dims, 'error')
        self.count = CompositeVariable(years_dims, 'count')
        self.mean = CompositeVariable(years_dims, 'mean')

        dims = (y_len, x_len)
        self.outlier_fraction = np.full(dims, np.nan)
        self.count_image_pairs = CompositeVariable(dims, 'count_image_pairs')
        self.amplitude = CompositeVariable(dims, 'amplitude')
        self.sigma = CompositeVariable(dims, 'sigma')
        self.phase = CompositeVariable(dims, 'phase')
        self.offset = CompositeVariable(dims, 'offset')
        self.slope = CompositeVariable(dims, 'slope')
        self.std_error = CompositeVariable(dims, 'std_error')

        # Sensor data for the cube's layers: map each sensor to its group ID
        self.sensors = sensorFilters.SensorExcludeFilter.map_sensor_to_group(
            self.stable_shift_filter.exclude(
                cube_ds[ImgPairInfo.satellite_img1].values
            )
        )
        # Identify sensors groups (L89, S1, S2, etc.) within datacube.
        self.sensors_groups = sensorFilters.SensorExcludeFilter.identify_sensor_groups(
            self.sensors
        )

        # Use true "date_center" value for processing since "mid_date" has been
        # adjusted by milliseconds to guarantee uniqueness of the values so we
        # can manipulate the whole xr.Dataset based on "mid_date" dimension
        self.date_center = self.stable_shift_filter.exclude(
            cube_ds[ImgPairInfo.date_center].values
        )

        sensor_dims = (y_len, x_len, len(self.sensors_groups))
        self.max_dt = np.full(sensor_dims, np.nan)
        self.sensor_include = np.ones(sensor_dims)

        # Date when composites were created
        self.date_created = datetime.datetime.now().strftime('%d-%b-%Y %H:%M:%S')
        self.date_updated = self.date_created

        # Initialize sensor exclusion filter
        self.sensor_filter = sensorFilters.SensorExcludeFilter(
            acq_datetime_img1,
            acq_datetime_img2,
            self.sensors,
            self.sensors_groups
        )

        if self.sensor_filter.excludeS2FromLSQ:
            # Need a 2 step LSQ fit: including S2 data and excluding S2 data,
            # allocate memory to store results of second LSQ fit
            self.excludeS2_error = CompositeVariable(years_dims, 'error')
            self.excludeS2_count = CompositeVariable(years_dims, 'count')
            self.excludeS2_mean = CompositeVariable(years_dims, 'mean')

            self.excludeS2_count_image_pairs = CompositeVariable(dims, 'count_image_pairs')
            self.excludeS2_amplitude = CompositeVariable(dims, 'amplitude')
            self.excludeS2_sigma = CompositeVariable(dims, 'sigma')
            self.excludeS2_phase = CompositeVariable(dims, 'phase')
            self.excludeS2_offset = CompositeVariable(dims, 'offset')
            self.excludeS2_slope = CompositeVariable(dims, 'slope')
            self.excludeS2_std_error = CompositeVariable(dims, 'std_error')

        # TODO: take care of self.date_updated when support for composites updates
        # is implemented

    def create(self, output_store: str):
        """
        Create datacube composite: cube time mean values.

        Inputs:
        =======
        output_store: Location to store the composite Zarr store.
        """
        # Loop through cube in chunks to minimize memory footprint
        x_start = 0
        x_num_to_process = self.cube_sizes[utils.Coords.X]
        # For validation/debugging only (RGI12A):
        # python ./new_composite.py -i ITS_LIVE_vel_EPSG32638_G0120_X350000_Y4750000.zarr
        # -o ITS_LIVE_velocity_EPSG32638_120m_X350000_Y475000_new_composite.zarr
        # -b s3://its-live-data/datacubes/v2-updated-october2024/N40E040
        # -t s3://its-live-data/test-space/composites-optimize-Sep04.2025
        # --disableErrorSlowUse --chunkSize 10 |& tee
        # ITS_LIVE_velocity_EPSG32638_120m_X350000_Y475000_new_composite_noprint.zarr.log

        logging.info(
            f"Processing cube size: [{self.cube_sizes[utils.Coords.MID_DATE]}, "
            f"{self.cube_sizes[utils.Coords.Y]}, {self.cube_sizes[utils.Coords.X]}]..."
        )
        while x_num_to_process > 0:
            # How many tasks to process at a time
            x_num_tasks = ITSLiveComposite.NUM_TO_PROCESS \
                if x_num_to_process > ITSLiveComposite.NUM_TO_PROCESS \
                else x_num_to_process

            y_start = 0
            y_num_to_process = self.cube_sizes[utils.Coords.Y]

            while y_num_to_process > 0:
                y_num_tasks = ITSLiveComposite.NUM_TO_PROCESS \
                    if y_num_to_process > ITSLiveComposite.NUM_TO_PROCESS \
                    else y_num_to_process

                self.cube_time_mean(x_start, x_num_tasks, y_start, y_num_tasks)
                gc.collect()

                y_num_to_process -= y_num_tasks
                y_start += y_num_tasks

            x_num_to_process -= x_num_tasks
            x_start += x_num_tasks

        # Save data to Zarr store
        self.to_zarr(output_store)

    @staticmethod
    def project_v_to_median_flow(
        ds_vx, ds_vy, ds_date_dt, ds_sensors_ids, exclude_sensors
    ):
        """
        Project valid velocity values to median flow unit vector.

        Inputs:
        =======
        ds_vx: 3d block of vx values.
        ds_vy: 3d block of vy values.
        ds_date_dt: day separation for velocity image pairs.
        ds_sensors_ids: Current sensor groups IDs for the datacube.
        exclude_sensors: 2d "map" of sensors group IDs to exclude from
            calculations (one set per each [y, x] point).
        """
        vp = np.full_like(ds_vx, np.nan)

        y_len, x_len, _ = ds_vx.shape

        for j_index in range(0, y_len):
            for i_index in range(0, x_len):
                # Exclude all identified invalid sensor groups per [y, x] point
                # exclude_mask = np.zeros((len(ds_sensors_ids)), dtype=np.bool_)
                exclude_set = exclude_sensors[j_index, i_index]

                # Use use list() for implicit conversion of set to the np.array
                include_mask = ~np.isin(ds_sensors_ids, list(exclude_set))

                x_in = ds_vx[j_index, i_index, include_mask]
                y_in = ds_vy[j_index, i_index, include_mask]
                dt = ds_date_dt[include_mask]
                vp[j_index, i_index, include_mask] = create_projected_velocity(
                    x_in, y_in, dt, DT_MEDIAN_FLOW
                )

        return vp

    def cube_time_mean(self, start_x, num_x, start_y, num_y):
        """
        Compute time average for the datacube
        [:, start_y:start_y + num_y, start_x:start_x + num_x] coordinates.
        Update corresponding entries in output data variables for the
        composite.

        Inputs:
        -------
        start_x: Starting index for the X dimension.
        num_x: Number of X slices to include.
        start_y: Starting index for the Y dimension.
        num_y: Number of Y slices to include.
        """
        # Set current block length for the X and Y dimensions
        stop_y = start_y + num_y
        stop_x = start_x + num_x
        ITSLiveComposite.Chunk = Chunk(start_x, stop_x, num_x, start_y, stop_y, num_y)

        # Start timer
        start_time = timeit.default_timer()

        # ----- FILTER DATA -----
        # Filter data based on locations where means of various dts are
        # statistically different and mad deviations from a running meadian

        # Initialize variables
        dims = (ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN)

        # Loop for each unique sensor (those groupings image pairs that can be
        # expected to have different temporal decorelation)

        # ATTN: don't use native xarray functionality is much slower,
        # convert data to numpy types and use numpy only
        logging.info(f'Loading [:, {start_y}:{stop_y}, {start_x}:{stop_x}]')
        vx_org = self.data.vx[:, start_y:stop_y, start_x:stop_x].values
        if vx_org.dtype != np.float32:
            vx_org = vx_org.astype(np.float32)

        vy_org = self.data.vy[:, start_y:stop_y, start_x:stop_x].values
        if vy_org.dtype != np.float32:
            vy_org = vy_org.astype(np.float32)

        # Reverse stable_shift and exclude granules if any are identified by the
        # StableShiftFilter
        vx_org, vy_org = self.stable_shift_filter.apply(vx_org, vy_org)

        # Transpose data to make it continuous in time
        vx = np.zeros((ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN))
        vx.flat = np.transpose(vx_org, ITSLiveComposite.CONT_TIME_ORDER)

        vy = np.zeros((ITSLiveComposite.Chunk.y_len, ITSLiveComposite.Chunk.x_len, ITSLiveComposite.MID_DATE_LEN))
        vy.flat = np.transpose(vy_org, ITSLiveComposite.CONT_TIME_ORDER)

        # Call filter to exclude sensors if any
        land_ice_mask = None if self.land_ice_mask is None else \
                        self.land_ice_mask[start_y:stop_y, start_x:stop_x]

        exclude_sensors = self.sensor_filter(
            ITSLiveComposite.DATE_DT,
            vx,
            vy,
            self.date_center,
            land_ice_mask
        )

        # Project valid (excluding sensors) v onto median flow vector:
        # take into account exclude_sensors for each spacial point
        v_invalid = np.full(dims, False)

        # Count all valid points before any filters are applied
        # Count should be based on middle date for each image pair falling within v0_years only

        #  Reduce number of image pairs only to the provided range: v0_years[0] <= mid_date < v0_years[-1]+1
        _v0_year_mask = create_v0_years_mask(
            ITSLiveComposite.START_DECIMAL_YEAR,
            ITSLiveComposite.STOP_DECIMAL_YEAR,
            ITSLiveComposite.V0_YEARS[0],
            ITSLiveComposite.V0_YEARS[-1]
        )

        count_mask = ~np.isnan(vx[..., _v0_year_mask])
        count0_vx = count_mask.sum(axis=2)

        copy_vx = None
        if self.sensor_filter.excludeS2FromLSQ:
            # Need to save original vx values before any filters are applied
            # if second LSQ fit iteration will be invoked
            copy_vx = vx.copy()

        # Note for v3:
        # Project velocity to median flow unit vector using only valid sensors: this is
        # pre-processing step for the dt_max filter, not used anywhere else.
        # Note: make it part of the dt_max_filter(), rename to dt_max_filter().
        vp = ITSLiveComposite.project_v_to_median_flow(
            vx,
            vy,
            ITSLiveComposite.DATE_DT,
            self.sensor_filter.sensors_ids,
            exclude_sensors
        )

        # TODO for v3: exclude v > 20000 right before any analysis (before SensorExcludeFilter)
        # filter vp against the same v limit
        vp_invalid_mask = (vp > ITSLiveComposite.V_LIMIT)
        vp[vp_invalid_mask] = np.nan

        # DEBUG only: store vp to CSV file
        # logging.info(f'vp.size={vp.shape}')
        # filename = f'good_vp.csv'
        # np.savetxt(filename, vp[0, 0, :], delimiter=',')

        # Apply dt filter: step through all sensors groups
        for i, sensor_group in enumerate(self.sensors_groups):
            # Find which layers correspond to the sensor group
            # mask = (self.sensor_filter.sensors_str == sensor_group.mission)
            mask = np.isin(self.sensor_filter.sensors_ids, sensor_group.id)

            # Filter current block's variables
            v_invalid[:, :, mask], \
                self.max_dt[start_y:stop_y, start_x:stop_x, i], \
                self.sensor_include[start_y:stop_y, start_x:stop_x, i] = \
                dt_max_filter(
                    vp[..., mask],
                    ITSLiveComposite.DATE_DT[mask],
                    sensor_group.id,
                    exclude_sensors
                )

        # Note for v3: exclude v > 20000 right before any analysis (before SensorExcludeFilter)
        invalid = v_invalid | (np.hypot(vx, vy) > ITSLiveComposite.V_LIMIT)

        # Mask data
        vx[invalid] = np.nan
        vy[invalid] = np.nan

        # logging.info(f'DEBUG:  Before LSQ fit: vx: min={np.nanmin(vx)} max={np.nanmax(vx)}')
        # Transform vx data to make time series continuous in memory: [y, x, t]
        cubelsqfit2(
            'vx',
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

        # logging.info(f'DEBUG:  Before LSQ fit: vy: min={np.nanmin(vy)} max={np.nanmax(vy)}')
        cubelsqfit2(
            'vy',
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
                self.std_error.vy[start_y:stop_y, start_x:stop_x],
                ITSLiveComposite.V_LIMIT
            )

        if self.sensor_filter.excludeS2FromLSQ:
            # The 2nd LSQ S2 filter should only be applied where land_ice_2km_inbuff == 1
            run_lsq_fit = True

            if self.land_ice_mask is not None:
                # Apply mask if it's available for the cube:
                # Alex: The SensorExcludeFilter should only be applied if landice_2km_inbuff == 0 and
                #       the 2nd LSQ S2 filter should only be applied where landice_2km_inbuff == 1
                mask = (self.land_ice_mask[start_y:stop_y, start_x:stop_x] == 1)

                if np.sum(mask) == 0:
                    # There are no cells to apply 2nd LSQ fit to
                    run_lsq_fit = False

                else:
                    vx[~mask] = np.nan
                    vy[~mask] = np.nan

            if run_lsq_fit:
                # Need to compare to LSQ fit excluding all S2 data: to see if
                # S2 contains "faulty" data
                mission_index = self.sensors_groups.index(
                    sensorFilters.SensorExcludeFilter.REF_SENSOR
                )

                # Find which layers correspond to the sensor group
                # mask = (self.sensor_filter.sensors_str == SensorExcludeFilter.REF_SENSOR.mission)
                mask = np.isin(
                    self.sensor_filter.sensors_ids,
                    sensorFilters.SensorExcludeFilter.REF_SENSOR.id
                )
                # logging.info(f'DEBUG: total number of valid S2 points: {np.sum(~np.isnan(vx[:, :, mask]))}')

                # Exclude S2 data from current block's variables
                vx[:, :, mask] = np.nan
                vy[:, :, mask] = np.nan

                # Exclude S2 granules from total number of granules
                copy_vx[:, :, mask] = np.nan

                # logging.info(f'DEBUG: Excluded S2 {self.sensors[mask]}')
                # logging.info(f'DEBUG: left total valid vx points: {np.sum(~np.isnan(vx))}')

                # %% Least-squares fits to detemine amplitude, phase and annual means

                # logging.info(f'DEBUG:  Before LSQ fit: vx: min={np.nanmin(vx)} max={np.nanmax(vx)}')
                # Transform vx data to make time series continuous in memory: [y, x, t]
                cubelsqfit2(
                    'vx_exclS2',
                    vx,
                    self.vx_error,
                    self.excludeS2_amplitude.vx,
                    self.excludeS2_phase.vx,
                    self.excludeS2_mean.vx,
                    self.excludeS2_error.vx,
                    self.excludeS2_sigma.vx,
                    self.excludeS2_count.vx,
                    self.excludeS2_count_image_pairs.vx,
                    self.excludeS2_offset.vx,
                    self.excludeS2_slope.vx,
                    self.excludeS2_std_error.vx
                )

                # logging.info(f'DEBUG:  Before LSQ fit: vy: min={np.nanmin(vy)} max={np.nanmax(vy)}')
                cubelsqfit2(
                    'vy_exclS2',
                    vy,
                    self.vy_error,
                    self.excludeS2_amplitude.vy,
                    self.excludeS2_phase.vy,
                    self.excludeS2_mean.vy,
                    self.excludeS2_error.vy,
                    self.excludeS2_sigma.vy,
                    self.excludeS2_count.vy,
                    self.excludeS2_count_image_pairs.vy,
                    self.excludeS2_offset.vy,
                    self.excludeS2_slope.vy,
                    self.excludeS2_std_error.vy
                )

                self.excludeS2_offset.v[start_y:stop_y, start_x:stop_x], \
                    self.excludeS2_slope.v[start_y:stop_y, start_x:stop_x], \
                    self.excludeS2_amplitude.v[start_y:stop_y, start_x:stop_x], \
                    self.excludeS2_sigma.v[start_y:stop_y, start_x:stop_x], \
                    self.excludeS2_phase.v[start_y:stop_y, start_x:stop_x], \
                    self.excludeS2_std_error.v[start_y:stop_y, start_x:stop_x] = \
                    climatology_magnitude(
                        self.excludeS2_offset.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_offset.vy[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_slope.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_slope.vy[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_amplitude.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_amplitude.vy[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_sigma.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_sigma.vy[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_phase.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_phase.vy[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_std_error.vx[start_y:stop_y, start_x:stop_x],
                        self.excludeS2_std_error.vy[start_y:stop_y, start_x:stop_x],
                        ITSLiveComposite.V_LIMIT
                    )

                # Check if there are any values that satisfy:
                # if (amp_all) > (S1+L8_amp) * 2 and (amp_all) - (S1+L8_amp) > 5)
                # then use lsqfit_annual output from S1+L8 and add S2 to the excluded sensors mask
                amp_mask = (
                    self.amplitude.v[start_y:stop_y, start_x:stop_x] > \
                    (self.excludeS2_amplitude.v[start_y:stop_y, start_x:stop_x] * ITSLiveComposite.LSQ_AMP_SCALE)
                ) & \
                    (
                        (
                            self.amplitude.v[start_y:stop_y, start_x:stop_x] - \
                            self.excludeS2_amplitude.v[start_y:stop_y, start_x:stop_x]
                        ) > ITSLiveComposite.LSQ_MIN_AMP_DIFF
                    )

                if np.sum(amp_mask) > 0:
                    # Use results from LSQ fit when excluding S2 for the spacial points
                    # where (amp_all) > (S1+L8_amp) * 2

                    # Re-compute the mask for valid count which now excludes S2 data
                    # count_mask = ~np.isnan(copy_vx)
                    # count0_vx = count_mask.sum(axis=2)
                    # logging.info(f'Second LSQ fit count based on copy_vx: {count0_vx}')

                    # Set output data to results of 2nd LSQ fit
                    self.amplitude.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_amplitude.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.amplitude.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_amplitude.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.amplitude.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_amplitude.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.phase.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_phase.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.phase.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_phase.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.phase.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_phase.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.mean.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_mean.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.mean.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_mean.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.mean.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_mean.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.error.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_error.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.error.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_error.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.error.v[start_y:stop_y, start_x:stop_x] = self.excludeS2_error.v[start_y:stop_y, start_x:stop_x]

                    self.sigma.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_sigma.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.sigma.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_sigma.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.sigma.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_sigma.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.count.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.count.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.count.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.count_image_pairs.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count_image_pairs.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    # Don't really use vy and v components of count_image_pairs, just to be complete:
                    self.count_image_pairs.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count_image_pairs.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    # This is not even computed, so no need to update anything
                    # self.count_image_pairs.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_count_image_pairs.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.offset.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_offset.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.offset.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_offset.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.offset.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_offset.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.slope.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_slope.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.slope.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_slope.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.slope.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_slope.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    self.std_error.vx[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_std_error.vx[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.std_error.vy[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_std_error.vy[start_y:stop_y, start_x:stop_x][amp_mask]
                    self.std_error.v[start_y:stop_y, start_x:stop_x][amp_mask] = self.excludeS2_std_error.v[start_y:stop_y, start_x:stop_x][amp_mask]

                    # Update self.sensor_include[start_y:stop_y, start_x:stop_x, i] to exclude S2 data
                    self.sensor_include[start_y:stop_y, start_x:stop_x, mission_index][amp_mask] = 0

                    # Re-set max_dt to NaNs
                    self.max_dt[start_y:stop_y, start_x:stop_x, mission_index][amp_mask] = np.nan

                    # Update total granule count only for the cells that are
                    # updated by the 2nd LSQ fit calculations
                    count_mask = ~np.isnan(copy_vx)
                    count0_vx[amp_mask] = count_mask.sum(axis=2)[amp_mask]

        # Some of the cells will have total granule count = 0, exclude these from
        # the assignment
        nonzero_count_mask = ~(count0_vx == 0)

        self.outlier_fraction[start_y:stop_y, start_x:stop_x][nonzero_count_mask] = 1 - (self.count_image_pairs.vx[start_y:stop_y, start_x:stop_x][nonzero_count_mask] / count0_vx[nonzero_count_mask])

        # Sanity check: all reported fractions should be positive
        positive_outlier_mask = (self.outlier_fraction[start_y:stop_y, start_x:stop_x] < 0.0)
        if np.sum(positive_outlier_mask) > 0:
            raise RuntimeError(f'Negative outlier fraction is detected: {self.outlier_fraction[start_y:stop_y, start_x:stop_x][positive_outlier_mask]} for indices={np.where(self.outlier_fraction[start_y:stop_y, start_x:stop_x] < 0.0)}')

        self.mean.v[start_y:stop_y, start_x:stop_x, :], \
            self.error.v[start_y:stop_y, start_x:stop_x, :], \
            self.count.v[start_y:stop_y, start_x:stop_x, :] = \
            annual_magnitude(
                self.mean.vx[start_y:stop_y, start_x:stop_x, :],
                self.mean.vy[start_y:stop_y, start_x:stop_x, :],
                self.error.vx[start_y:stop_y, start_x:stop_x, :],
                self.error.vy[start_y:stop_y, start_x:stop_x, :],
                self.count.vx[start_y:stop_y, start_x:stop_x, :],
                self.count.vy[start_y:stop_y, start_x:stop_x, :],
            )

        # Nan out invalid values
        invalid_mask = (self.mean.v > ITSLiveComposite.V_LIMIT)
        self.mean.v[invalid_mask] = np.nan
        self.mean.vx[invalid_mask] = np.nan
        self.mean.vy[invalid_mask] = np.nan

        invalid_mask = (self.amplitude.v > ITSLiveComposite.V_AMP_LIMIT)
        self.amplitude.v[invalid_mask] = np.nan
        self.amplitude.vx[invalid_mask] = np.nan
        self.amplitude.vy[invalid_mask] = np.nan

    def to_zarr(self, output_store: str):
        """
        Store datacube  composite to the Zarr store.

        Inputs:
        =======
        output_store: Location to store the composite Zarr store.
        """
        logging.info(f'Writing composites to {output_store}')

        # Convert years to datetime objects to represent the center of calendar year
        ITSLiveComposite.YEARS = [datetime.datetime(each, CENTER_DATE.month, CENTER_DATE.day) for each in ITSLiveComposite.YEARS]
        logging.info(f"Converted years to datetime objs: {ITSLiveComposite.YEARS}")

        # Create list of sensors groups labels
        sensors_labels = [each.label for each in self.sensors_groups]

        sensors_labels_attr = [f'Band {i+1}: {sensors_labels[i]}' for i in range(len(sensors_labels))]
        sensors_labels_attr = f'{", ".join(sensors_labels_attr)}'

        ds = xr.Dataset(
            coords={
                utils.Coords.X: (
                    utils.Coords.X,
                    self.cube_ds.x.values,
                    X_ATTRS
                ),
                utils.Coords.Y: (
                    utils.Coords.Y,
                    self.cube_ds.y.values,
                    Y_ATTRS
                ),
                utils.Coords.TIME: (
                    utils.Coords.TIME,
                    ITSLiveComposite.YEARS,
                    TIME_ATTRS
                ),
                utils.Coords.SENSORS: (
                    utils.Coords.SENSORS,
                    sensors_labels,
                    SENSORS_ATTRS
                )
            },
            attrs={
                utils.OutputFormat.author: CubeFormat.values[utils.OutputFormat.author]
            }
        )

        ds.attrs[CompositeVars.attrs.composites_software_version] = \
            ITSLiveComposite.VERSION
        ds.attrs[CubeFormat.date_created] = self.date_created
        ds.attrs[CubeFormat.date_updated] = self.date_updated

        # To support old format datacubes for testing
        # TODO: remove check for existence once done testing with old cubes
        # (to compare to Matlab)
        if utils.OutputFormat.s3 in self.cube_ds.attrs:
            ds.attrs[CompositeVars.attrs.datacube_s3] = \
                self.cube_ds.attrs[utils.OutputFormat.s3]
            ds.attrs[CompositeVars.attrs.datacube_url] = \
                self.cube_ds.attrs[utils.OutputFormat.url]

        ds.attrs[CompositeVars.attrs.datacube_created] = \
            self.cube_ds.attrs[CubeFormat.date_created]
        ds.attrs[CompositeVars.attrs.datacube_updated] = \
            self.cube_ds.attrs[CubeFormat.date_updated]
        ds.attrs[CubeFormat.datacube_software_version] = \
            self.cube_ds.attrs[CubeFormat.datacube_software_version]
        ds.attrs[CompositeVars.attrs.datacube_autorift_parameter_file] = \
            self.cube_ds.attrs[Vars.attrs.autorift_param_file]

        ds.attrs[CubeFormat.gdal_area_or_point] = \
            CubeFormat.values[CubeFormat.gdal_area_or_point]

        # To support old format datacubes for testing
        # TODO: remove once done testing with old cubes (to compare to Matlab)
        if CubeFormat.geo_polygon in self.cube_ds.attrs:
            ds.attrs[CubeFormat.geo_polygon] = \
                self.cube_ds.attrs[CubeFormat.geo_polygon]
            ds.attrs[CubeFormat.proj_polygon] = \
                self.cube_ds.attrs[CubeFormat.proj_polygon]

        ds.attrs[utils.OutputFormat.institution] = \
            CubeFormat.values[utils.OutputFormat.institution]
        ds.attrs[utils.OutputFormat.latitude] = \
            self.cube_ds.attrs[utils.OutputFormat.latitude]
        ds.attrs[utils.OutputFormat.longitude] = \
            self.cube_ds.attrs[utils.OutputFormat.longitude]
        ds.attrs[utils.OutputFormat.projection] = \
            self.cube_ds.attrs[utils.OutputFormat.projection]
        ds.attrs[utils.OutputFormat.s3] = ITSLiveComposite.S3
        ds.attrs[utils.OutputFormat.url] = ITSLiveComposite.URL
        ds.attrs[utils.OutputFormat.title] = \
            CubeFormat.values[utils.OutputFormat.title]

        # Add data as variables
        ds[Mapping.name] = self.cube_ds[Mapping.name]

        years_coord = pd.Index(ITSLiveComposite.YEARS, name=utils.Coords.TIME)
        var_coords = [years_coord, self.cube_ds.y.values, self.cube_ds.x.values]
        var_dims = [utils.Coords.TIME, utils.Coords.Y, utils.Coords.X]

        twodim_var_coords = [self.cube_ds.y.values, self.cube_ds.x.values]
        twodim_var_dims = [utils.Coords.Y, utils.Coords.X]

        self.land_ice_mask_composite = utils.to_int_type(
            self.land_ice_mask_composite,
            np.uint8,
            utils.Missing.byte
        )
        # Land ice mask exists for the composite
        ds[shapefile.LANDICE] = xr.DataArray(
            data=self.land_ice_mask_composite,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: shapefile.Name[shapefile.LANDICE],
                Vars.attrs.description: shapefile.Description[shapefile.LANDICE],
                Mapping.attrs.grid_mapping: Mapping.name,
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[shapefile.LANDICE],
                utils.OutputFormat.url: self.land_ice_mask_composite_url
            }
        )
        self.land_ice_mask_composite = None
        gc.collect()

        self.floating_ice_mask_composite = utils.to_int_type(
            self.floating_ice_mask_composite,
            np.uint8,
            utils.Missing.byte
        )
        # Land ice mask exists for the composite
        ds[shapefile.FLOATINGICE] = xr.DataArray(
            data=self.floating_ice_mask_composite,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: shapefile.Name[shapefile.FLOATINGICE],
                Vars.attrs.description: shapefile.Description[shapefile.FLOATINGICE],
                Mapping.attrs.grid_mapping: Mapping.name,
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[shapefile.FLOATINGICE],
                utils.OutputFormat.url: self.floating_ice_mask_composite_url
            }
        )
        self.floating_ice_mask_composite = None
        gc.collect()

        self.mean.transpose()
        self.error.transpose()
        self.count.transpose()

        # Convert data to output desired datatype
        self.error.to_uint16()       # v_error
        self.amplitude.to_uint16()
        self.sigma.to_uint16()       # amp. error
        self.phase.to_uint16()
        self.std_error.to_uint16()   # v0_error

        # Only these components are used in output, no need to convert the rest
        # of components
        self.count.v = utils.to_int_type(
            self.count.v,
            np.uint32,
            utils.Missing.byte
        )
        self.count_image_pairs.vx = utils.to_int_type(
            self.count_image_pairs.vx,
            np.uint32,
            utils.Missing.byte
        )

        ds[Vars.v] = xr.DataArray(
            data=self.mean.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[Vars.v],
                Vars.attrs.description: CompositeVars.description[Vars.v],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.mean.v = None
        gc.collect()

        ds[CompositeVars.v_error] = xr.DataArray(
            data=self.error.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.v_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.error.v = None
        gc.collect()

        ds[Vars.vx] = xr.DataArray(
            data=self.mean.vx,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[Vars.vx],
                Vars.attrs.description: CompositeVars.description[Vars.vx],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.mean.vx = None
        gc.collect()

        ds[CompositeVars.vx_error] = xr.DataArray(
            data=self.error.vx,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.error.vx = None
        gc.collect()

        ds[Vars.vy] = xr.DataArray(
            data=self.mean.vy,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[Vars.vy],
                Vars.attrs.description: CompositeVars.description[Vars.vy],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.mean.vy = None
        gc.collect()

        ds[CompositeVars.vy_error] = xr.DataArray(
            data=self.error.vy,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.error.vy = None
        gc.collect()

        ds[CompositeVars.v_amp] = xr.DataArray(
            data=self.amplitude.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v_amp] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Vars.attrs.description: CompositeVars.description[CompositeVars.v_amp] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.amplitude.v = None
        gc.collect()

        ds[CompositeVars.v_amp_error] = xr.DataArray(
            data=self.sigma.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v_amp_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.v_amp_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.sigma.v = None
        gc.collect()

        ds[CompositeVars.v_phase] = xr.DataArray(
            data=self.phase.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v_phase],
                Vars.attrs.description: CompositeVars.description[CompositeVars.v_phase] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.day_of_year
            }
        )
        self.phase.v = None
        gc.collect()

        ds[CompositeVars.vx_amp] = xr.DataArray(
            data=self.amplitude.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx_amp],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx_amp] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.amplitude.vx = None
        gc.collect()

        ds[CompositeVars.vx_amp_error] = xr.DataArray(
            data=self.sigma.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx_amp_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx_amp_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.sigma.vx = None
        gc.collect()

        ds[CompositeVars.vx_phase] = xr.DataArray(
            data=self.phase.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx_phase],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx_phase] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.day_of_year
            }
        )
        self.phase.vx = None
        gc.collect()

        ds[CompositeVars.vy_amp] = xr.DataArray(
            data=self.amplitude.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy_amp],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy_amp] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.amplitude.vy = None
        gc.collect()

        ds[CompositeVars.vy_amp_error] = xr.DataArray(
            data=self.sigma.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy_amp_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy_amp_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.sigma.vy = None
        gc.collect()

        ds[CompositeVars.vy_phase] = xr.DataArray(
            data=self.phase.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy_phase],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy_phase] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.day_of_year
            }
        )
        self.phase.vy = None
        gc.collect()

        ds[CompositeVars.count] = xr.DataArray(
            data=self.count.v,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.count],
                Vars.attrs.description: CompositeVars.description[CompositeVars.count],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.count
            }
        )
        self.count.v = None
        gc.collect()

        # Add max_dt (per sensor)
        # Use "group" label for each of the sensors used to filter data
        sensor_coord = pd.Index(sensors_labels, name=utils.Coords.SENSORS)
        var_coords = [sensor_coord, self.cube_ds.y.values, self.cube_ds.x.values]
        var_dims = [utils.Coords.SENSORS, utils.Coords.Y, utils.Coords.X]

        self.max_dt = self.max_dt.transpose(CompositeVariable.CONT_IN_X)
        self.max_dt = utils.to_int_type(self.max_dt)

        ds[CompositeVars.max_dt] = xr.DataArray(
            data=self.max_dt,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.max_dt],
                Vars.attrs.description: CompositeVars.description[CompositeVars.max_dt],
                Mapping.attrs.grid_mapping: Mapping.name,
                CompositeVars.attrs.sensors_labels: sensors_labels_attr,
                utils.Units.name: utils.Units.days
            }
        )
        self.max_dt = None
        gc.collect()

        self.sensor_include = self.sensor_include.transpose(CompositeVariable.CONT_IN_X)

        # Flip values: 0 - include; 1 - exclude (decision made at the time mosaics were created)
        mask_zeros = self.sensor_include == 0
        mask_ones = self.sensor_include == 1

        self.sensor_include[mask_zeros] = 1
        self.sensor_include[mask_ones] = 0

        self.sensor_include = utils.to_int_type(
            self.sensor_include,
            np.uint8,
            utils.Missing.byte
        )

        ds[CompositeVars.sensor_include] = xr.DataArray(
            data=self.sensor_include,
            coords=var_coords,
            dims=var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.sensor_include],
                Vars.attrs.description: CompositeVars.description[CompositeVars.sensor_include],
                Mapping.attrs.grid_mapping: Mapping.name,
                BinaryFlag.attrs.values: BinaryFlag.values,
                BinaryFlag.attrs.meanings: BinaryFlag.meanings[CompositeVars.sensor_include],
                CompositeVars.attrs.sensors_labels: sensors_labels_attr
            }
        )
        self.sensor_include = None
        gc.collect()

        # Convert to percent and use uint8 datatype
        self.outlier_fraction *= 100

        self.outlier_fraction = utils.to_int_type(
            self.outlier_fraction,
            np.uint8,
            utils.Missing.u8value
        )

        ds[CompositeVars.outlier_frac] = xr.DataArray(
            data=self.outlier_fraction,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.outlier_frac],
                Vars.attrs.description: CompositeVars.description[CompositeVars.outlier_frac] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.percent
            }
        )
        self.outlier_fraction = None
        gc.collect()

        ds[CompositeVars.vx0] = xr.DataArray(
            data=self.offset.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx0],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx0] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1], CENTER_DATE.year),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.offset.vx = None
        gc.collect()

        ds[CompositeVars.vy0] = xr.DataArray(
            data=self.offset.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy0],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy0] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1], CENTER_DATE.year),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.offset.vy = None
        gc.collect()

        ds[CompositeVars.v0] = xr.DataArray(
            data=self.offset.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v0] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Vars.attrs.description: CompositeVars.description[CompositeVars.v0] \
                    %(CENTER_DATE.year),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.offset.v = None
        gc.collect()

        ds[CompositeVars.vx0_error] = xr.DataArray(
            data=self.std_error.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vx0_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vx0_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.std_error.vx = None
        gc.collect()

        ds[CompositeVars.vy0_error] = xr.DataArray(
            data=self.std_error.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.vy0_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.vy0_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.std_error.vy = None
        gc.collect()

        ds[CompositeVars.v0_error] = xr.DataArray(
            data=self.std_error.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.v0_error],
                Vars.attrs.description: CompositeVars.description[CompositeVars.v0_error],
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y
            }
        )
        self.std_error.v = None
        gc.collect()

        ds[CompositeVars.slope_v] = xr.DataArray(
            data=self.slope.v,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.slope_v],
                Vars.attrs.description: CompositeVars.description[CompositeVars.slope_v] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y2
            }
        )
        self.slope.v = None
        gc.collect()

        ds[CompositeVars.slope_vx] = xr.DataArray(
            data=self.slope.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.slope_vx],
                Vars.attrs.description: CompositeVars.description[CompositeVars.slope_vx] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y2
            }
        )
        self.slope.vx = None
        gc.collect()

        ds[CompositeVars.slope_vy] = xr.DataArray(
            data=self.slope.vy,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.slope_vy],
                Vars.attrs.description: CompositeVars.description[CompositeVars.slope_vy] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                utils.Units.name: utils.Units.m_y2
            }
        )
        self.slope.vy = None
        gc.collect()

        ds[CompositeVars.count0] = xr.DataArray(
            data=self.count_image_pairs.vx,
            coords=twodim_var_coords,
            dims=twodim_var_dims,
            attrs={
                Vars.attrs.std_name: CompositeVars.name[CompositeVars.count0],
                Vars.attrs.description: CompositeVars.description[CompositeVars.count0] \
                    %(ITSLiveComposite.V0_YEARS[0], ITSLiveComposite.V0_YEARS[-1]),
                Mapping.attrs.grid_mapping: Mapping.name,
                Vars.attrs.note: f'{CompositeVars.count0} may not equal the sum of annual counts, as a single image pair can contribute to the least squares fit for multiple years',
                utils.Units.name: utils.Units.count
            }
        )
        self.count_image_pairs = None
        gc.collect()

        # ATTN: Set attributes for the Dataset coordinates as the very last step:
        # when adding data variables that don't have the same attributes for the
        # coordinates, originally set Dataset coordinates attributes will be wiped out
        # (xarray bug?)
        ds[utils.Coords.X].attrs = X_ATTRS
        ds[utils.Coords.Y].attrs = Y_ATTRS
        ds[utils.Coords.TIME].attrs = TIME_ATTRS
        ds[utils.Coords.SENSORS].attrs = SENSORS_ATTRS

        # Set encoding
        encoding_settings = {}

        # Compression for the data
        compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)

        encoding_settings.setdefault(utils.Coords.TIME, {}).update(
            {utils.Units.name: utils.Units.date}
        )

        # Don't set fill_value for the coordinate variables
        for each in [
            utils.Coords.TIME,
            utils.Coords.SENSORS,
            utils.Coords.X,
            utils.Coords.Y
        ]:
            encoding_settings.setdefault(each, {}).update(
                {
                    utils.OutputFormat.compressor: compressor
                }
            )

        encoding_settings.setdefault(utils.Coords.SENSORS, {}).update(
            {utils.OutputFormat.dtype: 'str'}
        )

        # Newer xarray versions set "fill_value" as attribute instead of
        # an encoding parameter. Also, setting "_FillValue" attribute will
        # enable automatic conversion of "fill_value" to Nans when reading
        # the data back in.

        # Settings for variables of "float" data type
        for each in [
            Vars.vx,
            Vars.vy,
            Vars.v,
            CompositeVars.vx0,
            CompositeVars.vy0,
            CompositeVars.v0,
            CompositeVars.slope_vx,
            CompositeVars.slope_vy,
            CompositeVars.slope_v
        ]:
            encoding_settings.setdefault(each, {}).update({
                utils.OutputFormat.dtype: np.float32,
                utils.OutputFormat.compressor: compressor
            })

            ds[each].attrs[CompositeVars.attrs.fill_value_attr] = utils.Missing.value
            ds[each].attrs[utils.OutputFormat.fill_value] = utils.Missing.value

            # No need to set "missing_value" attribute for floating point data
            # as it has _FillValue set for encoding.
            # ds[each].attrs[utils.Missing.name] = utils.Missing.value

        # Don't provide _FillValue for int types as it will avoid datatype
        # specification for the variable (according to xarray support,
        # _FillValue is used for floating point datatypes only)

        # Settings for variables of "uint16" data type
        for each in [
            CompositeVars.vx_error,
            CompositeVars.vy_error,
            CompositeVars.v_error,
            CompositeVars.vx_amp_error,
            CompositeVars.vy_amp_error,
            CompositeVars.v_amp_error,
            CompositeVars.vx_amp,
            CompositeVars.vy_amp,
            CompositeVars.v_amp,
            CompositeVars.vx_phase,
            CompositeVars.vy_phase,
            CompositeVars.v_phase,
            CompositeVars.vx0_error,
            CompositeVars.vy0_error,
            CompositeVars.v0_error,
            CompositeVars.max_dt
        ]:
            encoding_settings.setdefault(each, {}).update({
                utils.OutputFormat.dtype: np.uint16,
                utils.OutputFormat.compressor: compressor,
            })

            ds[each].attrs[CompositeVars.attrs.fill_value_attr] = utils.Missing.uvalue
            ds[each].attrs[utils.OutputFormat.fill_value] = utils.Missing.uvalue

        # Settings for variables of "uint8" data type
        for each in [
            CompositeVars.outlier_frac
        ]:
            encoding_settings.setdefault(each, {}).update({
                utils.OutputFormat.dtype: np.uint8,
                utils.OutputFormat.compressor: compressor,
            })

            ds[each].attrs[CompositeVars.attrs.fill_value_attr] = utils.Missing.u8value
            ds[each].attrs[utils.OutputFormat.fill_value] = utils.Missing.u8value

        # Variables that have missing_value = 0
        for each in [
            CompositeVars.sensor_include,
            shapefile.LANDICE,
            shapefile.FLOATINGICE
        ]:
            encoding_settings.setdefault(each, {}).update({
                utils.OutputFormat.dtype: np.uint8,
                utils.OutputFormat.compressor: compressor,
            })

            ds[each].attrs[CompositeVars.attrs.fill_value_attr] = utils.Missing.byte
            ds[each].attrs[utils.OutputFormat.fill_value] = utils.Missing.byte

        # NOTE: === this is relative to older versions of xarray ===
        # Settings for variables of "uint32" data type
        # Don't provide _FillValue as it will avoid datatype specification for the
        # variable (according to xarray support, _FillValue is used for floating point
        # datatypes only)
        for each in [
            CompositeVars.count,
            CompositeVars.count0
        ]:
            encoding_settings.setdefault(each, {}).update({
                utils.OutputFormat.dtype: np.uint32,
                utils.OutputFormat.compressor: compressor,
            })

            ds[each].attrs[CompositeVars.attrs.fill_value_attr] = utils.Missing.byte
            ds[each].attrs[utils.OutputFormat.fill_value] = utils.Missing.byte

        # Chunking to apply when writing datacube to the Zarr store
        chunks_settings = (1, self.cube_sizes[utils.Coords.Y], self.cube_sizes[utils.Coords.X])

        for each in [
            Vars.vx,
            Vars.vy,
            Vars.v,
            CompositeVars.vx_error,
            CompositeVars.vy_error,
            CompositeVars.v_error,
            CompositeVars.max_dt
        ]:
            encoding_settings[each].update({
                utils.OutputFormat.chunks: chunks_settings
            })

        # Chunking to apply when writing datacube to the Zarr store
        chunks_settings = (self.cube_sizes[utils.Coords.Y], self.cube_sizes[utils.Coords.X])

        for each in [
            CompositeVars.vx_amp,
            CompositeVars.vy_amp,
            CompositeVars.v_amp,
            CompositeVars.vx_phase,
            CompositeVars.vy_phase,
            CompositeVars.v_phase,
            CompositeVars.vx_amp_error,
            CompositeVars.vy_amp_error,
            CompositeVars.v_amp_error,
            CompositeVars.outlier_frac,
            CompositeVars.sensor_include,
            CompositeVars.vx0,
            CompositeVars.vy0,
            CompositeVars.v0,
            CompositeVars.vx0_error,
            CompositeVars.vy0_error,
            CompositeVars.v0_error,
            CompositeVars.slope_vx,
            CompositeVars.slope_vy,
            CompositeVars.slope_v,
            shapefile.LANDICE,
            shapefile.FLOATINGICE
        ]:
            encoding_settings[each].update({
                utils.OutputFormat.chunks: chunks_settings
            })

        logging.info(f"Encoding settings: {encoding_settings=}")

        ds.to_zarr(output_store, encoding=encoding_settings, consolidated=True)


def cubelsqfit2(
    var_name,
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
    se,
    num_valid_points=5
):
    """
    Cube LSQ fit with 2 iterations.

    Populates [amplitude, phase, mean, error, sigma, count]

    Performs a cube least squares fit with two iterations over a 3D data array,
    using joblib for parallel computation.

    This function fits a model to each pixel's time series in a data cube,
    skipping locations with insufficient valid data.
    The results are stored in the provided output arrays.

    Parameters
    ----------
    var_name : str
        Name of the variable being fitted.
    v : np.ndarray
        3D array of input data values (e.g., velocity) with shape (y, x, t).
    v_err_data : np.ndarray
        Array of error estimates for the input data. Can be 1D or 3D.
    amplitude : np.ndarray
        Output array to store fitted amplitude values.
    phase : np.ndarray
        Output array to store fitted phase values.
    mean : np.ndarray
        Output array to store fitted mean values.
    error : np.ndarray
        Output array to store error estimates for the fit.
    sigma : np.ndarray
        Output array to store fitted sigma values.
    count : np.ndarray
        Output array to store the count of valid data points used in the fit.
    count_image_pairs : np.ndarray
        Output array to store the count of image pairs used in the fit.
    offset : np.ndarray
        Output array to store fitted offset values.
    slope : np.ndarray
        Output array to store fitted slope values.
    se : np.ndarray
        Output array to store standard error of the fit.
    num_valid_points : int
        Minimum number of valid points required for the fit.

    Returns
    -------
    None
        Results are written in-place to the provided output arrays.

    Notes
    -----
    - Requires a minimum number of valid (non-NaN) data points to perform the fit.
    - Uses joblib for parallel computation across spatial dimensions.
    - Only updates output arrays for locations where the fit is valid.
    """
    v_err = np.broadcast_to(
        v_err_data,
        (
            ITSLiveComposite.Chunk.y_len,
            ITSLiveComposite.Chunk.x_len,
            v_err_data.size
        )
    )

    # Define processing function
    def process_point(j, i):
        global_i = i + ITSLiveComposite.Chunk.start_x
        global_j = j + ITSLiveComposite.Chunk.start_y

        results_valid, results = itslive_lsqfit_annual(
            var_name,
            v[j, i, :],
            v_err[j, i, :],
            ITSLiveComposite.D_COS,
            ITSLiveComposite.D_SIN,
            ITSLiveComposite.START_DECIMAL_YEAR,
            ITSLiveComposite.STOP_DECIMAL_YEAR,
            ITSLiveComposite.DECIMAL_DT,
            ITSLiveComposite.YEARS,
            ITSLiveComposite.M,
            ITSLiveComposite.MAD_STD_RATIO,
            ITSLiveComposite.V0_YEARS,
            CENTER_DATE,
            mean[global_j, global_i, :],
            error[global_j, global_i, :],
            count[global_j, global_i, :],
        )

        return (global_j, global_i, results_valid, results)

    # Get valid points for processing
    valid_points = []
    for j in range(ITSLiveComposite.Chunk.y_len):
        for i in range(ITSLiveComposite.Chunk.x_len):
            mask = ~np.isnan(v[j, i, :])
            if mask.sum() >= num_valid_points:
                valid_points.append((j, i))

    # Run in parallel with joblib
    all_results = Parallel(n_jobs=-1, backend='threading')(
        delayed(process_point)(j, i)
        for j, i in valid_points
    )

    # Process results
    for item in all_results:
        global_j, global_i, results_valid, results = item

        if results_valid:
            # Update global results only if they are reported to be valid.
            amplitude[global_j, global_i], \
            sigma[global_j, global_i], \
            phase[global_j, global_i], \
            offset[global_j, global_i], \
            slope[global_j, global_i], \
            se[global_j, global_i], \
            count_image_pairs[global_j, global_i] = results
            # if using "processes" for parallel processing instead of "threads",
            # the following will not work as the arrays are not shared memory.
            # Will need to return populated data arrays from the function instead
            # and then combine them outside of the function.
            # mean[global_j, global_i, :], \
            # error[global_j, global_i, :], \
            # count[global_j, global_i, :] = results

    return


if __name__ == '__main__':
    import argparse
    import warnings
    import shutil
    import subprocess
    import sys
    import time
    from urllib.parse import urlparse

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=ITSLiveComposite.__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunkSize',
        type=int,
        default=100,
        help='Number of X and Y coordinates to process in parallel. '
            'This should be multiples of the size of chunking used within '
            'the cube to optimize data reads [%(default)d].'
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
        help="S3 bucket directory to store cube composite in Zarr format to "
            "[%(default)s]. For example, "
            "s3://its-live-data/composites/v2/S70W100"
    )
    parser.add_argument(
        '-bb', '--backupBucket',
        type=str,
        default='',
        help="S3 bucket directory to backup original composites to before "
            "new composites are generated[%(default)s]. For example, "
            "s3://its-live-data/composites/v2/backup/S70W100"
    )
    parser.add_argument(
        '--noAWSSigning',
        action='store_true',
        default=False,
        help='Use no AWS signing for S3 requests. If set, requests will be '
            'unsigned (anon=True) which should be used for public buckets '
            '[%(default)d].'
    )
    parser.add_argument(
        '-s', '--shapeFile',
        type=str,
        default='s3://its-live-data/autorift_parameters/v001/autorift_landice_0120m.shp',
        help="Shapefile that stores ice masks per each of the EPSG codes [%(default)s]."
    )

    # Add optional group of mission include/exclude options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--missionGroup',
        type=str,
        default=None,
        help=f"Mission group ID to create composites for [%(default)s]. "
            f"One of {list(sensors.ALL_GROUPS.keys())}."
    )
    group.add_argument(
        '--excludeMissionGroup',
        type=lambda s: json.loads(s),
        default=None,
        help=f"JSON list of mission groups IDs to exclude from composites "
            f"[%(default)s]. One of {list(sensors.ALL_GROUPS.keys())}."
    )
    parser.add_argument(
        '--v0Years',
        type=str,
        default=str(list(range(2014, 2025))),
        help="Years to base computations of climotological data on "
            "[%(default)s]. It's a JSON list."
    )
    parser.add_argument(
        '--interceptDate',
        type=str,
        default='2018/01/01',
        help="Intercept date used for weighted linear fit [%(default)s]."
    )
    parser.add_argument(
        '--disableErrorSlowUse',
        action='store_false',
        help="Disable use of valid v[xy]_error_slow instead of v[xy]_error "
            "values [False]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Command arguments: {args}")
    logging.info(f"EC2 instance type: {aws_utils.get_instance_type()}")

    ITSCube.NO_AWS_SIGNING = args.noAWSSigning

    # If original composite exists and backup s3 location is provided, copy
    # existing composite to backup location before it gets overwritten by the
    # new composite
    if ITSCube.exists(args.outputStore, args.targetBucket) and \
        len(args.backupBucket):
        # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
        # resulting in as many error messages as there are files in Zarr store
        # to copy
        existing_url = os.path.join(args.targetBucket, args.outputStore)
        backup_url = existing_url.replace(args.targetBucket, args.backupBucket)
        logging.info(
            f'Backing up existing composite from {existing_url} to '
            f'{backup_url}'
        )
        command_line = [
            "aws", "s3", "cp", "--recursive",
            existing_url,
            backup_url,
            "--acl", "bucket-owner-full-control"
        ]

        env_copy = os.environ.copy()
        itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

    # Set static data for computation
    ITSLiveComposite.NUM_TO_PROCESS = args.chunkSize
    ITSLiveComposite.USE_ERROR_SLOW = args.disableErrorSlowUse
    logging.info(f'Use error_slow: {ITSLiveComposite.USE_ERROR_SLOW}')

    if ITSLiveComposite.USE_ERROR_SLOW:
        # Extend variables to load for processing
        ITSLiveComposite.VARS.append(f'{Vars.vx}_{Vars.postfix.error_slow}')
        ITSLiveComposite.VARS.append(f'{Vars.vy}_{Vars.postfix.error_slow}')

    # Read shape file with ice masks information in
    ITSLiveComposite.SHAPE_FILE = shapefile.read_file(args.shapeFile)

    if args.missionGroup:
        # Mission group is provided
        sensorFilters.StableShiftFilter.KEEP_MISSION_GROUP = \
            sensors.ALL_GROUPS[args.missionGroup]

    elif args.excludeMissionGroup:
        sensorFilters.StableShiftFilter.EXCLUDE_MISSION_GROUP = [
            sensors.ALL_GROUPS[each] for each in args.excludeMissionGroup
        ]

    ITSLiveComposite.V0_YEARS = json.loads(args.v0Years)
    CENTER_DATE = parse(args.interceptDate)

    logging.info(f'Got interceptDate: {CENTER_DATE}')

    if len(args.targetBucket):
        ITSLiveComposite.S3 = os.path.join(args.targetBucket, args.outputStore)
        logging.info(f'Composite S3: {ITSLiveComposite.S3}')

        # URL is valid only if output S3 bucket is provided
        ITSLiveComposite.URL = ITSLiveComposite.S3.replace(utils.S3_PREFIX,
                                                            utils.HTTP_PREFIX)
        url_tokens = urlparse(ITSLiveComposite.URL)
        ITSLiveComposite.URL = url_tokens._replace(netloc=url_tokens.netloc + \
                                                    utils.PATH_URL).geturl()
        logging.info(f'Composite URL: {ITSLiveComposite.URL}')

    mosaics = ITSLiveComposite(args.inputCube, args.inputBucket)
    mosaics.create(args.outputStore)

    if os.path.exists(args.outputStore):
        output_size = subprocess.run(
            ['du', '-skh', args.outputStore],
            capture_output=True,
            text=True
        ).stdout.split()[0]
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
                "aws", "s3", "cp", "--recursive",
                args.outputStore,
                os.path.join(args.targetBucket, os.path.basename(args.outputStore)),
                "--acl", "bucket-owner-full-control"
            ]

            env_copy = os.environ.copy()
            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

        finally:
            # Remove locally written Zarr store.
            # This is to eliminate out of disk space failures when the same EC2 instance is
            # being re-used by muliple Batch jobs.
            if os.path.exists(args.outputStore):
                logging.info(f"Removing local copy of {args.outputStore}")
                shutil.rmtree(args.outputStore)

    logging.info("Done.")
