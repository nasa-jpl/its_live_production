"""
Script to convert elevation NetCDF file to Zarr store.

Usage:
python elevation_to_zarr.py -o ZarrStoreName -i NetCDFFileName

Convert NetCDF elevation to Zarr format file.
"""

import argparse
import logging
import os
import sys
import subprocess
import timeit
import warnings
import xarray as xr
import zarr
import shutil
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


# Encoding settings for NetCDF format
compressor = zarr.Blosc(cname="zlib", clevel=2, shuffle=1)
compression = {"compressor": compressor}

ENCODING = {
    'dh':           {'_FillValue': -32767.0, 'dtype': np.short},
    'h':            {'_FillValue': -32767.0, 'dtype': np.short},
    'rmse':         {'_FillValue': -32767.0, 'dtype': np.short},
    'quality_flag': {'_FillValue': 0, 'dtype': 'ubyte'},
    'basin':        {'_FillValue': 0, 'dtype': 'ubyte'},
    'time':         {'_FillValue': None, 'units': 'days since 1950-01-01'},
    'x':            {'_FillValue': None},
    'y':            {'_FillValue': None}
}

# Chunking to use for reading data from NetCDF store
CHUNKS={'x': 10, 'y': 10, 'time': -1}

def main(input_file: str, output_file: str):
    """
    Convert elevation data from NetCDF to Zarr format file.
    """
    start_time = timeit.default_timer()

    encoding_settings = ENCODING
    with xr.open_dataset(input_file, chunks=CHUNKS, engine='h5netcdf') as ds:
        ds_sizes = ds.sizes
        logging.info(f'Opened input {input_file}: {ds_sizes}')

        # Update compression for each of the vars:
        for each in ['dh', 'h', 'rmse', 'quality_flag', 'basin']:
            encoding_settings[each].update(compression)

        # Set chunking for 3d vars
        chunks = (ds_sizes['time'], 10, 10)
        for each in ['dh', 'rmse', 'quality_flag']:
            encoding_settings[each]['chunks'] = chunks

        chunks = (10, 10)
        for each in ['h', 'basin']:
            encoding_settings[each]['chunks'] = chunks

        logging.info(f"Saving dataset to {output_file}...")
        start_time = timeit.default_timer()
        ds.to_zarr(output_file, encoding=ENCODING, consolidated=True)

        time_delta = timeit.default_timer() - start_time
        logging.info(f"Wrote dataset to Zarr store {output_file} (took {time_delta} seconds)")

        # Open Zarr store as Dask array to allow for stream write to NetCDF
        # chunks = {'x': chunks_size, 'y': chunks_size}

    # Don't decode time delta's as it does some internal conversion based on
    # provided units
    time_delta = timeit.default_timer() - start_time
    logging.info(f" NetCDF {input_file} (took {time_delta} seconds)")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    parser = argparse.ArgumentParser(epilog='\n'.join(__doc__.split('\n')[1:]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input Zarr store directory.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="NetCDF filename to store data to.")
    parser.add_argument('-b', '--outputBucket', type=str, default="",
                        help="S3 bucket to copy composites in NetCDF format to [%(default)s].")

    args = parser.parse_args()
    logging.info(f"Args: {sys.argv}")

    main(args.input, args.output)

    if os.path.exists(args.output) and len(args.outputBucket):
        # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
        # resulting in as many error messages as there are files in Zarr store
        # to copy

        # Copy NetCDF file to the bucket
        env_copy = os.environ.copy()
        command_line = [
            "aws", "s3", "cp", "--recursive",
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
        shutil.rmtree(args.output)

    logging.info("Done.")
