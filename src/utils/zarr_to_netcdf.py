"""
Script to convert Zarr store to the NetCDF format file.

Usage:
python zarr_to_netcdf.py -i ZarrStoreName -o NetCDFFileName

Convert Zarr data stored in ZarrStoreName to the NetCDF file NetCDFFileName.
"""

import argparse
import timeit
import warnings
import xarray as xr

from itscube_types import Coords, DataVars


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(epilog='\n'.join(__doc__.split('\n')[1:]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input Zarr store directory.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="NetCDF filename to store data to.")
    parser.add_argument('-e', '--engine', type=str, required=False, default='h5netcdf',
                        help="NetCDF engine to use to store NetCDF data to the file.")

    args = parser.parse_args()

    start_time = timeit.default_timer()
    # Don't decode time delta's as it does some internal conversion based on
    # provided units
    ds_zarr = xr.open_zarr(args.input, decode_timedelta=False)

    time_delta = timeit.default_timer() - start_time
    print(f"Read Zarr {args.input} (took {time_delta} seconds)")

    compression = {"zlib": True, "complevel": 2, "shuffle": True}
    encoding = {}

    encoding = {
        # 'map_scale_corrected':       {'_FillValue': 0.0, 'dtype': 'byte'},
        'interp_mask':               {'_FillValue': 0.0, 'dtype': 'ubyte'},
        'flag_stable_shift':         {'dtype': 'long'},
        'chip_size_height':          {'_FillValue': 0.0, 'dtype': 'ushort'},
        'chip_size_width':           {'_FillValue': 0.0, 'dtype': 'ushort'},
        'v_error':                   {'_FillValue': -32767.0, 'dtype': 'short'},
        'v':                         {'_FillValue': -32767.0, 'dtype': 'short'},
        'vx':                        {'_FillValue': -32767.0, 'dtype': 'short'},
        'vx_error':                  {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vx_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy':                        {'_FillValue': -32767.0, 'dtype': 'short'},
        'vy_error':                  {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vy_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'va':                        {'_FillValue': -32767.0, 'dtype': 'short'},
        'va_error':                  {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'va_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr':                        {'_FillValue': -32767.0, 'dtype': 'short'},
        'vr_error':                  {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vr_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp':                       {'_FillValue': -32767.0, 'dtype': 'short'},
        'vxp_error':                 {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double'},
        'vyp':                       {'_FillValue': -32767.0, 'dtype': 'short'},
        'vyp_error':                 {'_FillValue': -32767.0, 'dtype': 'double'},
        'vyp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double'},
        'vyp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double'},
        'vyp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double'},
        'vyp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double'},
        'vxp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double'},
        'vp':                        {'_FillValue': -32767.0, 'dtype': 'short'},
        'vp_error':                  {'_FillValue': -32767.0, 'dtype': 'short'},
        'acquisition_date_img1':     {'units': 'days since 1970-01-01'},
        'acquisition_date_img2':     {'units': 'days since 1970-01-01'},
        'date_center':               {'_FillValue': None, 'units': 'days since 1970-01-01'},
        'mid_date':                  {'_FillValue': None, 'units': 'days since 1970-01-01'},
        'autoRIFT_software_version': {'_FillValue': None},
        'stable_count_slow':         {'_FillValue': None, 'dtype': 'long'},
        'stable_count_mask':         {'_FillValue': None, 'dtype': 'long'},
        'date_dt':                   {'_FillValue': None},
        'x':                         {'_FillValue': None},
        'y':                         {'_FillValue': None}
    }

    if 'stable_count' in ds_zarr:
        encoding['stable_count'] = {'_FillValue': None, 'dtype': 'long'}

    encode_data_vars = [
        'v',
        'v_error',
        'vx',
        'vx_error',
        'vx_error_mask',
        'vx_error_modeled',
        'vx_error_slow',
        'vx_stable_shift',
        'vx_stable_shift_slow',
        'vx_stable_shift_mask',
        'flag_stable_shift',
        'vy',
        'vy_error',
        'vy_error_mask',
        'vy_error_modeled',
        'vy_error_slow',
        'vy_stable_shift',
        'vy_stable_shift_slow',
        'vy_stable_shift_mask',
        'chip_size_height',
        'chip_size_width',
        'interp_mask',
        'va',
        'va_error',
        'va_error_mask',
        'va_error_modeled',
        'va_error_slow',
        'va_stable_shift',
        'va_stable_shift_slow',
        'va_stable_shift_mask',
        'vp',
        'vp_error',
        'vr',
        'vr_error',
        'vr_error_mask',
        'vr_error_modeled',
        'vr_error_slow',
        'vr_stable_shift',
        'vr_stable_shift_slow',
        'vr_stable_shift_mask',
        'vxp',
        'vxp_error',
        'vxp_error_mask',
        'vxp_error_modeled',
        'vxp_error_slow',
        'vxp_stable_shift',
        'vxp_stable_shift_slow',
        'vxp_stable_shift_mask',
        'vyp',
        'vyp_error',
        'vyp_error_mask',
        'vyp_error_modeled',
        'vyp_error_slow',
        'vyp_stable_shift',
        'vyp_stable_shift_slow',
        'vyp_stable_shift_mask',
        'mission_img1',
        'sensor_img1',
        'satellite_img1',
        'acquisition_date_img1',
        'mission_img2',
        'sensor_img2',
        'satellite_img2',
        'acquisition_date_img2',
        'date_dt',
        'date_center',
        'roi_valid_percentage',
        'autoRIFT_software_version'
    ]

    if DataVars.MAP_SCALE_CORRECTED in ds_zarr:
        encode_data_vars.append(DataVars.MAP_SCALE_CORRECTED)

    # Set up compression for each of the data variables
    for each in encode_data_vars:
        encoding.setdefault(each, {}).update(compression)

    start_time = timeit.default_timer()
    ds_zarr.to_netcdf(
        args.output,
        engine=args.engine,
        encoding = encoding
    )
    time_delta = timeit.default_timer() - start_time
    print(f"Wrote dataset to NetCDF file {args.output} (took {time_delta} seconds)")
