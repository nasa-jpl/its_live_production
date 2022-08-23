"""
Script to convert composite's Zarr store to the NetCDF format file.

Usage:
python composites_to_netcdf.py -i ZarrStoreName -o NetCDFFileName

Convert Zarr composites data stored in ZarrStoreName to the NetCDF file NetCDFFileName.
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
import numpy as np

# Encoding settings for NetCDF format
ENCODING = {
    'v':                         {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx':                        {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy':                        {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx_error':                  {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy_error':                  {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v_error':                   {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx_amp_error':              {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy_amp_error':              {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v_amp_error':               {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx_amp':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy_amp':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v_amp':                     {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx_phase':                  {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy_phase':                  {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v_phase':                   {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'outlier_frac':              {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx0':                       {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy0':                       {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v0':                        {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vx0_error':                 {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'vy0_error':                 {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'v0_error':                  {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'dvx_dt':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'dvy_dt':                    {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'dv_dt':                     {'_FillValue': -32767.0, 'dtype': np.float32, "zlib": True, "complevel": 2, "shuffle": True},
    'count':                     {'_FillValue': 0, 'dtype': np.uint32},
    'count0':                    {'_FillValue': 0, 'dtype': np.uint32},
    'sensor_flag':               {'dtype': np.short},
    'dt_max':                    {'_FillValue': 0, 'dtype': np.short},
    'time':                      {'_FillValue': None, 'units': 'days since 1970-01-01'},
    'sensor':                    {'_FillValue': None, 'dtype': 'S1'},
    'x':                         {'_FillValue': None},
    'y':                         {'_FillValue': None}
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

def convert(ds_zarr: xr.Dataset, output_file: str, nc_engine: str):
    """
    Store datacube to NetCDF format file.
    """
    start_time = timeit.default_timer()

    # Workaround for QGIS: add an attribute to map band index to the band label:
    sensors = ds_zarr.sensor.values
    sensors_labels = [f'Band {index+1}: {sensors[index]}' for index in range(len(sensors))]
    ds_zarr.attrs['sensors_labels'] = f'{"; ".join(sensors_labels)}'

    ds_zarr.to_netcdf(
        output_file,
        engine=nc_engine,
        encoding=ENCODING
    )

    time_delta = timeit.default_timer() - start_time
    logging.info(f"Wrote dataset to NetCDF file {output_file} (took {time_delta} seconds)")

def main(input_file: str, output_file: str, nc_engine: str, chunks_size: int):
    """
    Convert datacube composites Zarr store to NetCDF format file.
    """
    start_time = timeit.default_timer()

    ds_zarr = None
    s3_in = None
    # Open Zarr store as Dask array to allow for stream write to NetCDF
    dask_chunks = {'x': chunks_size, 'y': chunks_size}

    if 's3:' not in input_file:
        # If reading local Zarr store, check if composites store exists
        if os.path.exists(input_file):
            # Read dataset in
            ds_zarr = xr.open_zarr(input_file, decode_timedelta=False, consolidated=True, chunks=dask_chunks)

        else:
            raise RuntimeError(f"Input composites file '{input_file}' does not exist.")

    else:
        # When datacube is in the AWS S3 bucket, check if it exists.
        s3_in = s3fs.S3FileSystem(anon=True)

        file_list = s3_in.glob(input_file)
        if len(file_list) != 0:
            # Access datacube in S3 bucket
            ds_store = s3fs.S3Map(root=input_file, s3=s3_in, check=False)
            ds_zarr = xr.open_dataset(ds_store, decode_timedelta=False, engine='zarr', consolidated=True, chunks=dask_chunks)

        else:
            raise RuntimeError(f"Input composites file '{input_file}' does not exist.")

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
                        help="S3 bucket to copy composites in NetCDF format to [%(default)s].")
    parser.add_argument('-c', '--chunks', type=int, default=100,
                        help="Dask chunk size for x and y coordinates [%(default)d]. " \
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
