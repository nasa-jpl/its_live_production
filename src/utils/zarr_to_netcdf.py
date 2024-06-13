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


from itscube_types import DataVars, ShapeFile, Coords

# Encoding settings for Zarr format
COMPRESSION_ZARR = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

ENCODING_ZARR = {
    DataVars.INTERP_MASK:                       {'_FillValue': 0.0, 'dtype': 'ubyte', 'compressor': COMPRESSION_ZARR},
    DataVars.FLAG_STABLE_SHIFT:                 {'_FillValue': None, 'dtype': 'long', 'compressor': COMPRESSION_ZARR},
    DataVars.CHIP_SIZE_HEIGHT:                  {'_FillValue': 0.0, 'dtype': 'ushort', 'compressor': COMPRESSION_ZARR},
    DataVars.CHIP_SIZE_WIDTH:                   {'_FillValue': 0.0, 'dtype': 'ushort', 'compressor': COMPRESSION_ZARR},
    DataVars.STABLE_COUNT_SLOW:                 {'_FillValue': None, 'dtype': 'long', 'compressor': COMPRESSION_ZARR},
    DataVars.STABLE_COUNT_MASK:                 {'_FillValue': None, 'dtype': 'long', 'compressor': COMPRESSION_ZARR},
    DataVars.V_ERROR:                           {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    DataVars.V:                                 {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    DataVars.VX:                                {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.ERROR:                     {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.ERROR_MASK:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.ERROR_MODELED:             {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.ERROR_SLOW:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.STABLE_SHIFT:              {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.STABLE_SHIFT_SLOW:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vx_' + DataVars.STABLE_SHIFT_MASK:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.VY:                                {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.ERROR:                     {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.ERROR_MASK:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.ERROR_MODELED:             {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.ERROR_SLOW:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.STABLE_SHIFT:              {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.STABLE_SHIFT_SLOW:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vy_' + DataVars.STABLE_SHIFT_MASK:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.VA:                                {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.ERROR:                     {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.ERROR_MASK:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.ERROR_MODELED:             {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.ERROR_SLOW:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.STABLE_SHIFT:              {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.STABLE_SHIFT_SLOW:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'va_' + DataVars.STABLE_SHIFT_MASK:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.VR:                                {'_FillValue': -32767, 'dtype': 'short', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.ERROR:                     {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.ERROR_MASK:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.ERROR_MODELED:             {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.ERROR_SLOW:                {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.STABLE_SHIFT:              {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.STABLE_SHIFT_SLOW:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'vr_' + DataVars.STABLE_SHIFT_MASK:         {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'M11_' + DataVars.DR_TO_VR_FACTOR:          {'_FillValue': 0.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    'M12_' + DataVars.DR_TO_VR_FACTOR:          {'_FillValue': 0.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.M11:                               {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.M12:                               {'_FillValue': -32767.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1: {'_FillValue': None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2: {'_FillValue': None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.DATE_CENTER:           {'_FillValue': None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    Coords.MID_DATE:                            {'_FillValue': None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    Coords.X:                                   {'compressor': COMPRESSION_ZARR},
    Coords.Y:                                   {'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE:  {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SATELLITE_IMG1:        {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SATELLITE_IMG2:        {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.MISSION_IMG1:          {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.MISSION_IMG2:          {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SENSOR_IMG1:           {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SENSOR_IMG2:           {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.AUTORIFT_SOFTWARE_VERSION:         {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.DATE_DT:               {'_FillValue': None, 'compressor': COMPRESSION_ZARR},
    DataVars.URL:                               {'_FillValue': None, 'compressor': COMPRESSION_ZARR}
}

# Encoding settings for NetCDF format
ENCODING = {
    # 'map_scale_corrected':       {'_FillValue': 0.0, 'dtype': 'byte'},
    'interp_mask':                      {'missing_value': 0.0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    'stable_shift_flag':                {'_FillValue': None, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    'chip_size_height':                 {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
    'chip_size_width':                  {'_FillValue': 0.0, 'dtype': 'ushort', "zlib": True, "complevel": 2, "shuffle": True},
    'v_error':                          {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'v':                                {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vx':                               {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR:             {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_MASK:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_MODELED:     {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_SLOW:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT:      {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT_SLOW: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT_MASK: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy':                               {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR:             {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_MASK:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_MODELED:     {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_SLOW:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT:      {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT_SLOW: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT_MASK: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va':                               {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR:             {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_MASK:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_MODELED:     {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_SLOW:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT:      {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT_SLOW: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT_MASK: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr':                               {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR:             {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_MASK:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_MODELED:     {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_SLOW:        {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT:      {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT_SLOW: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT_MASK: {'_FillValue': -32767.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'M11_' + DataVars.DR_TO_VR_FACTOR:  {'_FillValue': 0.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'M12_' + DataVars.DR_TO_VR_FACTOR:  {'_FillValue': 0.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'M11':                              {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'M12':                              {'_FillValue': -32767, 'dtype': 'short', "zlib": True, "complevel": 2, "shuffle": True},
    'acquisition_date_img1':            {'_FillValue': None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    'acquisition_date_img2':            {'_FillValue': None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    'roi_valid_percentage':             {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'satellite_img1':                   {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'satellite_img2':                   {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'mission_img1':                     {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'mission_img2':                     {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'sensor_img1':                      {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'sensor_img2':                      {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'date_center':                      {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'mid_date':                         {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'autoRIFT_software_version':        {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.STABLE_COUNT_SLOW:         {'_FillValue': None, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.STABLE_COUNT_MASK:         {'_FillValue': None, 'dtype': 'long', "zlib": True, "complevel": 2, "shuffle": True},
    ShapeFile.LANDICE:                  {'missing_value': 255, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    ShapeFile.FLOATINGICE:              {'missing_value': 255, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    'date_dt':                          {'_FillValue': None, "zlib": True, "complevel": 2, "shuffle": True},
    'x':                                {'_FillValue': None},
    'y':                                {'_FillValue': None}
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
    dask_chunks = {'mid_date': chunks_size, 'x': 10, 'y': 10}

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
