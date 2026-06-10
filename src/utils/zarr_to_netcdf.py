"""
Script to convert Zarr store to the NetCDF format file.

Usage:
python zarr_to_netcdf.py -i ZarrStoreName -o NetCDFFileName

Convert Zarr data stored in ZarrStoreName to the NetCDF file NetCDFFileName.
"""

import argparse
import logging
import numpy as np
import os
import psutil
import s3fs
import sys
import subprocess
import timeit
import warnings
import xarray as xr
import zarr

import utils
import shapefile
from itscube_types import DataVars, Output, Vars

# Encoding settings for Zarr format
COMPRESSION_ZARR = zarr.Blosc(cname='zlib', clevel=2, shuffle=1)

ENCODING_ZARR = {
    Vars.interp_mask:                       {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.interp_mask], 'compressor': COMPRESSION_ZARR},
    Vars.flag_stable_shift:                 {Output.MISSING_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.flag_stable_shift], 'compressor': COMPRESSION_ZARR},
    Vars.chip_size_height:                  {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.chip_size_height], 'compressor': COMPRESSION_ZARR},
    Vars.chip_size_width:                   {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.chip_size_width], 'compressor': COMPRESSION_ZARR},
    Vars.stable_count_slow:                 {Output.MISSING_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.stable_count_slow], 'compressor': COMPRESSION_ZARR},
    Vars.stable_count_mask:                 {Output.MISSING_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.stable_count_mask], 'compressor': COMPRESSION_ZARR},
    Vars.v_error:                           {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.v_error], 'compressor': COMPRESSION_ZARR},
    Vars.v:                                 {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.v], 'compressor': COMPRESSION_ZARR},
    Vars.vx:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.vx], 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.error}':       {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.error_mask}':  {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.error_modeled}':     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.error_slow}':        {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.stable_shift}':      {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.stable_shift_slow}': {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vx}_{Vars.postfix.stable_shift_mask}': {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.vy:                                      {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.vy], 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.error}':           {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.error_mask}':      {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.error_modeled}':   {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.error_slow}':      {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.stable_shift}':    {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.stable_shift_slow}': {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vy}_{Vars.postfix.stable_shift_mask}': {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.va:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.va], 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.error}':                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.error_mask}':                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.error_modeled}':             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.error_slow}':                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.stable_shift}':              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.stable_shift_slow}':         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.va}_{Vars.postfix.stable_shift_mask}':         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.vr:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.vr], 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.error}':                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.error_mask}':                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.error_modeled}':             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.error_slow}':                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.stable_shift}':              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.stable_shift_slow}':         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.vr}_{Vars.postfix.stable_shift_mask}':         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.m11}_{Vars.postfix.dr_to_vr_factor}':          {Output.FILL_VALUE_ATTR: 0.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    f'{Vars.m12}_{Vars.postfix.dr_to_vr_factor}':          {Output.FILL_VALUE_ATTR: 0.0, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.m11:                               {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.m12:                               {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', 'compressor': COMPRESSION_ZARR},
    Vars.ImgPairInfo.ACQUISITION_DATE_IMG1: {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    Vars.ImgPairInfo.ACQUISITION_DATE_IMG2: {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    Vars.ImgPairInfo.DATE_CENTER:           {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    utils.Coords.MID_DATE:                      {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', 'compressor': COMPRESSION_ZARR},
    utils.Coords.X:                             {'compressor': COMPRESSION_ZARR},
    utils.Coords.Y:                             {'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE:  {Output.FILL_VALUE_ATTR: None, 'dtype': np.float32, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SATELLITE_IMG1:        {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SATELLITE_IMG2:        {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.MISSION_IMG1:          {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.MISSION_IMG2:          {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SENSOR_IMG1:           {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.SENSOR_IMG2:           {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    Vars.autorift_software_version:             {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR},
    DataVars.ImgPairInfo.DATE_DT:               {Output.FILL_VALUE_ATTR: None, 'dtype': np.float32, 'compressor': COMPRESSION_ZARR},
    DataVars.URL:                               {Output.FILL_VALUE_ATTR: None, 'compressor': COMPRESSION_ZARR}
}

# Encoding settings for NetCDF format
ENCODING = {
    # 'map_scale_corrected':       {Output.FILL_VALUE_ATTR: 0.0, 'dtype': 'byte'},
    Vars.interp_mask:                       {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.interp_mask], "zlib": True, "complevel": 2, "shuffle": True},
    Vars.flag_stable_shift:                 {Output.MISSING_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.flag_stable_shift], "zlib": True, "complevel": 2, "shuffle": True},
    Vars.chip_size_height:                  {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.chip_size_height], "zlib": True, "complevel": 2, "shuffle": True},
    Vars.chip_size_width:                   {Output.MISSING_VALUE_ATTR: 0.0, 'dtype': DataVars.INT_TYPE[Vars.chip_size_width], "zlib": True, "complevel": 2, "shuffle": True},
    Vars.v_error:                           {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[Vars.v_error], "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.V:                                 {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[DataVars.V], "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.VX:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[DataVars.VX], "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR:                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_MASK:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_MODELED:             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.ERROR_SLOW:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT:              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT_SLOW:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vx_' + DataVars.STABLE_SHIFT_MASK:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.VY:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[DataVars.VY], "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR:                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_MASK:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_MODELED:             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.ERROR_SLOW:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT:              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT_SLOW:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vy_' + DataVars.STABLE_SHIFT_MASK:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.VA:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[DataVars.VA], "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR:                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_MASK:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_MODELED:             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.ERROR_SLOW:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT:              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT_SLOW:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'va_' + DataVars.STABLE_SHIFT_MASK:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.VR:                                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': DataVars.INT_TYPE[DataVars.VR], "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR:                     {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_MASK:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_MODELED:             {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.ERROR_SLOW:                {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT:              {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT_SLOW:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'vr_' + DataVars.STABLE_SHIFT_MASK:         {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'M11_' + DataVars.DR_TO_VR_FACTOR:          {Output.FILL_VALUE_ATTR: 0.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    'M12_' + DataVars.DR_TO_VR_FACTOR:          {Output.FILL_VALUE_ATTR: 0.0, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.M11:                               {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.M12:                               {Output.FILL_VALUE_ATTR: DataVars.MISSING_VALUE, 'dtype': 'float', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1: {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2: {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.ROI_VALID_PERCENTAGE:  {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.SATELLITE_IMG1:        {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.SATELLITE_IMG2:        {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.MISSION_IMG1:          {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.MISSION_IMG2:          {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.SENSOR_IMG1:           {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.SENSOR_IMG2:           {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.DATE_CENTER:           {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    utils.Coords.MID_DATE:                            {Output.FILL_VALUE_ATTR: None, 'units': 'days since 1970-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.AUTORIFT_SOFTWARE_VERSION:         {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    Vars.stable_count_slow:                 {Output.FILL_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.stable_count_slow], "zlib": True, "complevel": 2, "shuffle": True},
    Vars.stable_count_mask:                 {Output.FILL_VALUE_ATTR: None, 'dtype': DataVars.INT_TYPE[Vars.stable_count_mask], "zlib": True, "complevel": 2, "shuffle": True},
    shapefile.LANDICE:                           {Output.MISSING_VALUE_ATTR: 255, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    shapefile.FLOATINGICE:                       {Output.MISSING_VALUE_ATTR: 255, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.ImgPairInfo.DATE_DT:               {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    utils.Coords.X:                                   {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True},
    utils.Coords.Y:                                   {Output.FILL_VALUE_ATTR: None, "zlib": True, "complevel": 2, "shuffle": True}
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
    #     ENCODING['stable_count'] = {Output.FILL_VALUE_ATTR: None, 'dtype': 'long'}
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
    parser = argparse.ArgumentParser(
        epilog='\n'.join(__doc__.split('\n')[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="Input Zarr store directory.")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="NetCDF filename to store data to.")
    parser.add_argument('-e', '--engine', type=str, required=False,
                        default='h5netcdf',
                        help="NetCDF engine to use to store NetCDF data to "
                                "the file.")
    parser.add_argument('-b', '--outputBucket', type=str, default="",
                        help="S3 bucket to copy datacube in NetCDF format to "
                                "[%(default)s].")
    parser.add_argument('-c', '--chunks', type=int, default=250,
                        help="Dask chunk size for mid_date coordinate "
                                "[%(default)d]. This is to handle datacubes that "
                                "can't fit in memory, and should be read as "
                                "Dask arrays.")

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
