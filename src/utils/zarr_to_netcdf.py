"""
Script to convert Zarr store to the NetCDF format file.

Usage:
python zarr_to_netcdf.py -i ZarrStoreName -o NetCDFFileName

Convert Zarr data stored in ZarrStoreName to the NetCDF file NetCDFFileName.
"""

import argparse
import logging
import os
import psutil
import s3fs
import sys
import subprocess
import timeit
import warnings
import xarray as xr
import zarr


# Encoding settings for Zarr format
COMPRESSION_ZARR = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

ENCODING_ZARR = {
    # 'map_scale_corrected':       {'_FillValue': 0.0, 'dtype': 'byte'},
    'interp_mask':               {'_FillValue': 0.0, 'dtype': 'ubyte'},
    'flag_stable_shift':         {'_FillValue': 0, 'dtype': 'long'},
    'chip_size_height':          {'_FillValue': 0.0, 'dtype': 'ushort'},
    'chip_size_width':           {'_FillValue': 0.0, 'dtype': 'ushort'},
    'stable_count_slow':         {'_FillValue': None, 'dtype': 'long'},
    'stable_count_mask':         {'_FillValue': None, 'dtype': 'long'},
    'v_error':                   {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'v':                         {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vx':                        {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vx_error':                  {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vx_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy':                        {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vy_error':                  {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vy_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va':                        {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'va_error':                  {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'va_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr':                        {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vr_error':                  {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vr_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp':                       {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vxp_error':                 {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vxp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp':                       {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vyp_error':                 {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vyp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double', 'compressor': COMPRESSION_ZARR},
    'vp':                        {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vp_error':                  {'_FillValue': -32767.0, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'acquisition_date_img1':     {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'acquisition_date_img2':     {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'date_center':               {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'mid_date':                  {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'roi_valid_percentage':      {'_FillValue': None},
    'satellite_img1':            {'_FillValue': None},
    'satellite_img2':            {'_FillValue': None},
    'mission_img1':              {'_FillValue': None},
    'mission_img2':              {'_FillValue': None},
    'sensor_img1':               {'_FillValue': None},
    'sensor_img2':               {'_FillValue': None},
    'autoRIFT_software_version': {'_FillValue': None},
    'date_dt':                   {'_FillValue': None}
}

# Encoding settings for NetCDF format
ENCODING = {
    # 'map_scale_corrected':       {'_FillValue': 0.0, 'dtype': 'byte'},
    'interp_mask':               {'_FillValue': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    'flag_stable_shift':         {'_FillValue': 0, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    'chip_size_height':          {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
    'chip_size_width':           {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
    'v_error':                   {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'v':                         {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vx':                        {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_error':                  {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy':                        {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_error':                  {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va':                        {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'va_error':                  {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'va_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr':                        {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_error':                  {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_error_mask':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_error_modeled':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_error_slow':             {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_stable_shift':           {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_stable_shift_slow':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_stable_shift_mask':      {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp':                       {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_error':                 {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vxp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp':                       {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_error':                 {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_error_mask':            {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_error_modeled':         {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_error_slow':            {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_stable_shift':          {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_stable_shift_slow':     {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vyp_stable_shift_mask':     {'_FillValue': -32767.0, 'dtype': 'double', "zlib": True, "complevel": 2, "shuffle": True},
    'vp':                        {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vp_error':                  {'_FillValue': -32767.0, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'acquisition_date_img1':     {'_FillValue': None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    'acquisition_date_img2':     {'_FillValue': None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    'roi_valid_percentage':      {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'satellite_img1':            {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'satellite_img2':            {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'mission_img1':              {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'mission_img2':              {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'sensor_img1':               {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'sensor_img2':               {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'date_center':               {'_FillValue': None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    'mid_date':                  {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'autoRIFT_software_version': {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'stable_count_slow':         {'_FillValue': None, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    'stable_count_mask':         {'_FillValue': None, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    'date_dt':                   {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'x':                         {'_FillValue': 0},
    'y':                         {'_FillValue': 0}
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def show_memory_usage(msg: str=''):
    """
    Display current memory usage.
    """
    _GB = 1024 * 1024 * 1024
    usage = psutil.virtual_memory()

    # Use standard logging to be able to use the method without ITSCube object
    memory_msg = 'Memory '
    if len(msg):
        memory_msg += msg

    logging.info(f"{memory_msg}: total={usage.total/_GB}Gb used={usage.used/_GB}Gb available={usage.available/_GB}Gb")

def convert(ds_zarr: xr.Dataset, output_file: str, nc_engine: str):
    """
    Store datacube to NetCDF format file.
    """
    # if 'stable_count' in ds_zarr:
    #     ENCODING['stable_count'] = {'_FillValue': None, 'dtype': 'long'}
    #
    # if 'map_scale_corrected' in ds_zarr:
    #     ENCODE_DATA_VARS.append('map_scale_corrected')
    #
    # # Set up compression for each of the data variables
    # for each in ENCODE_DATA_VARS:
    #     ENCODING.setdefault(each, {}).update(compression)

    start_time = timeit.default_timer()
    show_memory_usage('before to_netcdf()')
    ds_zarr.to_netcdf(
        output_file,
        engine=nc_engine,
        encoding=ENCODING
    )
    show_memory_usage('after to_netcdf()')

    time_delta = timeit.default_timer() - start_time
    logging.info(f"Wrote dataset to NetCDF file {output_file} (took {time_delta} seconds)")

def main(input_file: str, output_file: str, nc_engine: str, chunks_size: int):
    """
    Convert datacube Zarr store to NetCDF format file.
    """
    start_time = timeit.default_timer()

    ds_zarr = None
    s3_in = None
    # Open Zarr store as Dask array to allow for stream write to NetCDF
    dask_chunks = {'mid_date': chunks_size}

    show_memory_usage('before open Zarr()')

    if 's3:' not in input_file:
        # If reading local Zarr store, check if datacube store exists
        if os.path.exists(input_file):
            # Read dataset in
            ds_zarr = xr.open_zarr(input_file, decode_timedelta=False, consolidated=True, chunks=dask_chunks)

        else:
            raise RuntimeError(f"Input datacube {input_file} does not exist.")

    else:
        # When datacube is in the AWS S3 bucket, check if it exists.
        s3_in = s3fs.S3FileSystem(anon=True)

        file_list = s3_in.glob(input_file)
        if len(file_list) != 0:
            # Access datacube in S3 bucket
            cube_store = s3fs.S3Map(root=input_file, s3=s3_in, check=False)
            ds_zarr = xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True, chunks=dask_chunks)

        else:
            raise RuntimeError(f"Input datacube {input_file} does not exist.")


    # Don't decode time delta's as it does some internal conversion based on
    # provided units
    time_delta = timeit.default_timer() - start_time
    logging.info(f"Read Zarr {input_file} (took {time_delta} seconds)")

    convert(ds_zarr, output_file, nc_engine)

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
    parser.add_argument('-b', '--outputBucket', type=str, default="",
                        help="S3 bucket to copy datacube in NetCDF format to [%(default)s].")
    parser.add_argument('-c', '--chunks', type=int, default=250,
                        help="Dask chunk size for mid_date coordinate [%(default)d]. " \
                        "This is to handle datacubes that can't fit in memory, and should be read as Dask arrays.")

    args = parser.parse_args()
    logging.info(f"Args: {sys.argv}")

    main(args.input, args.output, args.engine, args.chunks)

    if os.path.exists(args.output) and len(args.outputBucket):
        # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
        # resulting in as many error messages as there are files in Zarr store
        # to copy

        # Copy NetCDF file to the bucket
        env_copy = os.environ.copy()
        command_line = [
            "aws", "s3", "cp",
            args.output,
            os.path.join(args.outputBucket, os.path.basename(args.output)),
            "--acl", "bucket-owner-full-control"
        ]

        logging.info(' '.join(command_line))

        command_return = subprocess.run(
            command_line,
            env=env_copy,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        if command_return.returncode != 0:
            logging.error(f"Failed to copy {args.output} to {args.outputBucket}: {command_return.stdout}")

        # Remove locally written NetCDF file if target location is AWS S3 bucket.
        # This is to eliminate out of disk space failures when the same EC2 instance is
        # being re-used by muliple Batch jobs.
        logging.info(f"Removing local copy of {args.output}")
        os.remove(args.output)

    logging.info("Done.")
