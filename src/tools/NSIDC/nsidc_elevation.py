"""
Script to prepare V1 ITS_LIVE elevation to be ingested by NSIDC:

0. Set Conventions=CF-1.8 to be consistent with other v01 data sets.

1. Remove "none" units if present for the data variable.

2. Add a "comment" attribute to the "time" data variable to indicate missing data
   for 1990-1992 years.

3. Change standard_name from/to:
    h:            height
    dh:           height_change
    rmse:         height_change_rmse
    basin:        glacier_basin
    quality flag: quality_flag

4. Replace 'description' by 'flag_values' and 'flag_meanings' for quality_flag
   data variable.

5. Add missing description (long_name) for quality_flag, dh

6. Fix various attributes of "mapping" variable:
    - Set semi_major_axis = 6378137
    - Delete "semi_minor_axis" attribute
    - Change attribute name from spatial_proj4 to proj4text
    - Add redundant to spatial_ref additional "crs_wkt" attribute (expands interoperability
      with geolocation tools)
    - 6a. Replace:
    :latitude_of_origin = -71.0; // double
      with:
    :standard_parallel = -71.0;

7. Set quality_flag to "0" (no data) for time coordinates that correspond to
   missing 1990-1992 data.

8. Enable compression.

9. Rename some of the data variables:
    h:     height,
    dh:    height_change,
    rmse:  height_change_rmse,
    basin: glacier_basin

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Mark Fahnestock (UFA)
"""

import argparse
import boto3
import dask
from dask.diagnostics import ProgressBar
import logging
import numpy as np
import os
import pyproj
import s3fs
import xarray as xr

# Local imports
from itscube_types import Coords, DataVars, Output
from nsidc_types import Mapping
from nsidc_vel_image_pairs import NSIDCFormat, get_attr_value, PlatformSensor


class Vars:
    """
    Variable names for elevation data.
    """
    TIME = 'time'
    QUALITY_FLAG = 'quality_flag'

    # Old names
    H = 'h'
    DH = 'dh'
    RMSE = 'rmse'
    BASIN = 'basin'

    # New names
    HEIGHT = 'height'
    HEIGHT_CHANGE = 'height_change'
    HEIGHT_CHANGE_RMSE = 'height_change_rmse'
    GLACIER_BASIN = 'glacier_basin'

    # Mapping between old and new variable names
    OldToNewMap = {
        H: HEIGHT,
        DH: HEIGHT_CHANGE,
        RMSE: HEIGHT_CHANGE_RMSE,
        BASIN: GLACIER_BASIN
    }


# Compression settings to add to h, dh, rmse, basin, and quality_flag
ElevationCompression = {
    "zlib": True, "complevel": 2, "shuffle": True
}


# Encoding settings for writing ITS_LIVE elevation to the NetCDF format file.
ElevationEncoding = {
    Vars.DH:           {'_FillValue': -32767.0, 'dtype': np.short, "zlib": True, "complevel": 2, "shuffle": True},
    Vars.H:            {'_FillValue': -32767.0, 'dtype': np.short, "zlib": True, "complevel": 2, "shuffle": True},
    Vars.RMSE:         {'_FillValue': -32767.0, 'dtype': np.short, "zlib": True, "complevel": 2, "shuffle": True},
    Vars.QUALITY_FLAG: {'_FillValue': 0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    Vars.BASIN:        {'_FillValue': 0, 'dtype': 'ubyte', "zlib": True, "complevel": 2, "shuffle": True},
    Vars.TIME:         {'_FillValue': None, 'units': 'days since 1950-01-01', "zlib": True, "complevel": 2, "shuffle": True},
    DataVars.MAPPING:  {'_FillValue': None, 'dtype': np.float32},
    Coords.X:          {'_FillValue': None},
    Coords.Y:          {'_FillValue': None}
}


class NSIDCElevationMeta:
    """
    Class to create premet and spacial files for each of the elevation files.

    Example of premet file:
    =======================
    FileName=ANT_G1920V01_GroundedIceHeight.nc
    VersionID_local=001
    Begin_date=1985-04-17
    End_date=2020-12-16
    Begin_time=11:10:41.601
    End_time=17:49:16.640
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=GEOSAT
    AssociatedInstrumentShortName=RA
    AssociatedSensorShortName=RA
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=ERS-1
    AssociatedInstrumentShortName=RA
    AssociatedSensorShortName=RA
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=ERS-2
    AssociatedInstrumentShortName=RA
    AssociatedSensorShortName=RA
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=ENVISAT
    AssociatedInstrumentShortName=RA-2
    AssociatedSensorShortName=RA-2
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=CRYOSAT-2
    AssociatedInstrumentShortName=SIRAL
    AssociatedSensorShortName=SIRAL
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=ICESat
    AssociatedInstrumentShortName=GLAS
    AssociatedSensorShortName=GLAS
    Container=AssociatedPlatformInstrumentSensor
    AssociatedPlatformShortName=ICESat-2
    AssociatedInstrumentShortName=ATLAS
    AssociatedSensorShortName=ATLAS

    Example of spatial file:
    ========================
    cat ../../elevationData/fromNSIDC/ANT_G1920V01_GroundedIceHeight.nc.spatial
    -134.9998   -54.681
    135.0   -54.6812


    # Structure:
    # WesternMostLongitude NorthernMostLatitude
    # EasternMostLongitude SouthernMostLatitude
    """
    # Data version
    VERSION = '001'

    # Dictionary of metadata values based on the mission+sensor token
    GEOSAT = 'GEOSAT'
    ERS1 = 'ERS-1'
    ERS2 = 'ERS-2'
    ENVISAT = 'ENVISAT'
    CRYOSAT2 = 'CRYOSAT-2'
    ICESAT = 'ICESat'
    ICESAT2 = 'ICESat-2'

    SHORT_NAME = {
        GEOSAT:   PlatformSensor(GEOSAT, 'RA'),
        ERS1:     PlatformSensor(ERS1, 'RA'),
        ERS2:     PlatformSensor(ERS2, 'RA'),
        ENVISAT:  PlatformSensor(ENVISAT, 'RA-2'),
        CRYOSAT2: PlatformSensor(CRYOSAT2, 'SIRAL'),
        ICESAT:   PlatformSensor(ICESAT, 'GLAS'),
        ICESAT2:  PlatformSensor(ICESAT2, 'ATLAS')
    }

    # Attributes
    COMMENT = 'comment'

    @staticmethod
    def create_premet_file(infile: str, time_vals: np.array):
        """
        Create premet file that corresponds to the input elevation file.

        Inputs
        ======
        infile: Filename of the input ITS_LIVE elevation file
        time_vals: Datetime values that correspond to the elevation.
        """
        begin_date = time_vals[0]
        end_date = time_vals[-1]

        # All missions to list in premet file
        missions = sorted(NSIDCElevationMeta.SHORT_NAME.keys())

        meta_filename = f'{infile}.premet'
        with open(meta_filename, 'w') as fh:
            fh.write(f'FileName={os.path.basename(infile)}\n')
            fh.write(f'VersionID_local={NSIDCElevationMeta.VERSION}\n')
            fh.write(f'Begin_date={begin_date.strftime("%Y-%m-%d")}\n')
            fh.write(f'End_date={end_date.strftime("%Y-%m-%d")}\n')
            # Display only 3 digits of microseconds
            fh.write(f'Begin_time={begin_date.strftime("%H:%M:%S.%f")[:-3]}\n')
            fh.write(f'End_time={end_date.strftime("%H:%M:%S.%f")[:-3]}\n')

            # Append premet with sensor info
            for each_sensor in missions:
                fh.write("Container=AssociatedPlatformInstrumentSensor\n")
                fh.write(f"AssociatedPlatformShortName={NSIDCElevationMeta.SHORT_NAME[each_sensor].platform}\n")
                fh.write(f"AssociatedInstrumentShortName={NSIDCElevationMeta.SHORT_NAME[each_sensor].sensor}\n")
                fh.write(f"AssociatedSensorShortName={NSIDCElevationMeta.SHORT_NAME[each_sensor].sensor}\n")

        return meta_filename

    @staticmethod
    def create_spatial_file(infile: str):
        """
        Create spatial file that corresponds to the input elevation in the
        following format:

        # WesternMostLongitude NorthernMostLatitude
        # EasternMostLongitude SouthernMostLatitude

        Inputs
        ======
        infile: Filename of the input ITS_LIVE elevation file

        Per Ryan Weber at NSIDC:
        "This spatial file is a bit of an odd case because it uses an Antarctic Polar Stereographic projection.
        In order to correct for this projection we need to manually input most of the values.
        For the sample NetCDF (ANT_G1920V01_GroundedIceHeight.nc), the coordinates in the spatial file will look like:

        -180       -54.68
        180         -90

        Where the only coordinate that needs to be extracted from the NetCDF is the maximum latitude, in this case -54.68.
        If all your files cover the same area then the above can be used in all the spatial files.
        Basically, by using -180 to 180 for longitude and -90 for the southern
        latitude, we’re forcing the projection to wrap the pole.

        For this data set we decided on a rectangle coordinate system which only
        requires the 2 corner points. We use the 4+ points when using a polygon
        coordinate system. So for this data set we will just have the 2 points.
        "
        """
        meta_filename = f'{infile}.spatial'

        with xr.open_dataset(infile, engine='h5netcdf') as ds:
            xvals = ds.x.values
            yvals = ds.y.values
            pix_size_x = xvals[1] - xvals[0]
            pix_size_y = yvals[1] - yvals[0]

            # minval_x, pix_size_x, _, maxval_y, _, pix_size_y = [float(x) for x in ds['mapping'].attrs['GeoTransform'].split()]

            # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
            projection_cf_minx = xvals[0] - pix_size_x/2.0
            projection_cf_maxx = xvals[-1] + pix_size_x/2.0
            projection_cf_miny = yvals[-1] + pix_size_y/2.0  # pix_size_y is negative!
            projection_cf_maxy = yvals[0] - pix_size_y/2.0  # pix_size_y is negative!

            epsgcode = int(get_attr_value(ds['mapping'].attrs['spatial_epsg']))
            epsgcode_str = f'EPSG:{epsgcode}'

            if epsgcode == NSIDCFormat.ESRI_CODE:
                epsgcode_str = f'ESRI:{epsgcode}'

            transformer = pyproj.Transformer.from_crs(epsgcode_str, "EPSG:4326", always_xy=True)  # ensure lonlat output order

            # Convert coordinates to long/lat
            # low right coordinate
            lr_lonlat = np.round(transformer.transform(projection_cf_maxx, projection_cf_miny), decimals=3).tolist()
            # upper left coordinate
            ul_lonlat = np.round(transformer.transform(projection_cf_minx, projection_cf_maxy), decimals=3).tolist()

        # Write to spatial file
        with open(meta_filename, 'w') as fh:
            for long, lat in [ul_lonlat, lr_lonlat]:
                fh.write(f"{long}\t{lat}\n")

        is_valid = True

        # Raise exception if south and north points have the same values
        if projection_cf_miny == projection_cf_maxy:
            is_valid = False
            logging.info(f'WARNING: Update spatial points as identical values are detected for southern and northern points of the polygon (pole wrap around issue most likely): '
                         f'ul_lonlat={ul_lonlat} lr_lonlat={lr_lonlat} in local file {meta_filename}. Then copy updated file to the destination S3 bucket.')

        return is_valid, meta_filename


class NSIDCElevationFormat:
    """
    Class to prepare V1 ITS_LIVE elevation for ingest by NSIDC:
    1. Make V1 ITS_LIVE data CF-1.8 convention compliant.
    2. Generate metadata files required by NSIDC ingest (premet and spacial metadata files
       which are generated per each data product being ingested).
    """
    # Pattern to collect input files in AWS S3 bucket
    GLOB_PATTERN = '*.nc'

    # Re-chunk input to speed up read of the data
    CHUNK_SIZE = 108

    def __init__(self, s3_bucket: str, s3_dir: str):
        """
        Initialize the object.

        Inputs:
        =======
        s3_bucket: AWS S3 bucket that stores input files.
        s3_dir: Directory in AWS S3 bucket that stores original files.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        # Elevation files as read from the S3 bucket's directory
        glob_pattern = os.path.join(s3_bucket, s3_dir, NSIDCElevationFormat.GLOB_PATTERN)
        logging.info(f"Glob elevation files: {glob_pattern}")

        self.infiles = self.s3.glob(f'{glob_pattern}')

        logging.info(f"Got {len(self.infiles)} files")

    def __call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        ATTN: This method implements sequential processing for debugging purposes only.

        Fix ITS_LIVE elevation data according to NSIDC standard and create
        corresponding NSIDC meta files (spatial and premet).
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info("Nothing to process, exiting.")
            return

        # Current start index into list of files to process
        start = 0

        while total_num_files > 0:
            logging.info(f"Starting {self.infiles[start]} {start} out of {init_total_files} total files")
            results = NSIDCElevationFormat.fix_file(
                target_bucket,
                target_dir,
                self.infiles[start],
                self.s3,
                NSIDCElevationFormat.CHUNK_SIZE
            )
            logging.info("\n-->".join(results))

            total_num_files -= 1
            start += 1

    def no__call__(self, target_bucket, target_dir, chunk_size, num_dask_workers):
        """
        Fix ITS_LIVE elevation and create corresponding NSIDC meta files (spacial
        and premet) in parallel.
        Not used for now as we only have 1 elevation file to fix.
        """
        total_num_files = len(self.infiles)
        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info("Nothing to process, exiting.")
            return

        # Current start index into list of granules to process
        start = 0

        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting {start}:{start+num_tasks} out of {init_total_files} total files")
            tasks = [
                dask.delayed(NSIDCElevationFormat.fix_file)(
                    target_bucket,
                    target_dir,
                    each,
                    self.s3,
                    NSIDCElevationFormat.CHUNK_SIZE
                ) for each in self.infiles[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("\n-->".join(each_result))

            total_num_files -= num_tasks
            start += num_tasks

    @staticmethod
    def fix_file(target_bucket: str, target_dir: str, infilewithpath: str, s3, chunk_size: int):
        """
        Fix data format and create corresponding metadata files as required by NSIDC.
        """
        # Update convention to 1.8 just to be consistent with other v01 data products
        _comment = 'comment'
        _time_comment_value = 'The arrays in the time steps between 1990 and 1992 are empty as no input data were available'

        _conventions = 'Conventions'
        _cf_value = 'CF-1.8'

        _flag_values = 'flag_values'
        _flag_meanings = 'flag_meanings'

        _latitude_of_origin = 'latitude_of_origin'

        _binary_flags = np.array([0, 1, 2, 3], dtype=np.uint8)
        _binary_meanings = 'no_data high_quality_data low_quality_data pole_hole'

        # Could identify empty data at runtime - for now we just know that these
        # years don't have any data
        _empty_data_t = ('1990', '1991', '1992')

        # New values for standard_name attribute of corresponding data variables
        _std_name = {
            Vars.H:            Vars.HEIGHT,
            Vars.DH:           Vars.HEIGHT_CHANGE,
            Vars.RMSE:         Vars.HEIGHT_CHANGE_RMSE,
            Vars.BASIN:        Vars.GLACIER_BASIN,
            Vars.QUALITY_FLAG: Vars.QUALITY_FLAG
        }

        _desc = {
            Vars.QUALITY_FLAG: 'a high level metric of overall data quality',
            Vars.DH: 'change in ice sheet elevation [m] relative to December 16, 2013',
            Vars.H: 'heights are from GLO30 DGED circa 2010-2015, see source'
        }

        filename_tokens = infilewithpath.split(os.path.sep)
        filename = filename_tokens[-1]
        local_file = filename + '.local'

        logging.info(f'Filename: {infilewithpath}')
        msgs = [f'Processing {infilewithpath} into new format']

        bucket = boto3.resource('s3').Bucket(target_bucket)
        bucket_file = os.path.join(target_dir, filename)

        if NSIDCFormat.object_exists(bucket, bucket_file):
            msgs.append(f'WARNING: {bucket.name}/{bucket_file} already exists, skipping file generation')

        else:
            s3_client = boto3.client('s3')

            file_path = os.path.sep.join(filename_tokens[1:])

            # Download file locally - takes too long to read the whole mosaic file
            # from S3 in order for it to write fixed dataset locally
            s3_client.download_file(target_bucket, file_path, local_file)

            with xr.open_dataset(local_file, engine='h5netcdf') as ds:
                sizes = ds.sizes
                logging.info(f'Dataset sizes: {sizes}')

                # To be consistent with other v01 data sets:
                ds.attrs[_conventions] = _cf_value

                # 2. Add comment to "time" about missing data
                ds[Vars.TIME].attrs[_comment] = _time_comment_value

                # If ran from multiple threads, use a copy of the settings
                # enc_settings = copy.deepcopy(ElevationEncoding)

                # Convert keys to list since we will remove some of the variables
                # during iteration
                for each_var in list(ds.keys()):
                    msgs.append(f'Processing {each_var}')

                    # Copy original chunking size for the variable
                    # if Output.CHUNKSIZES_ATTR in ds[each_var].encoding:
                    #     enc_settings[each_var][Output.CHUNKSIZES_ATTR] = ds[each].encoding[Output.CHUNKSIZES_ATTR]

                    # 1. Remove "none" units
                    if DataVars.UNITS in ds[each_var].attrs and \
                            ds[each_var].attrs[DataVars.UNITS] == "none":
                        # Remove "none" units
                        del ds[each_var].attrs[DataVars.UNITS]

                    # 3. Change standard_name for some data variables
                    if each_var in _std_name:
                        ds[each_var].attrs[DataVars.STD_NAME] = _std_name[each_var]

                    # 4. Replace 'description' by 'flag_values' and 'flag_meanings'
                    #    for "quality_flag"
                    if each_var == Vars.QUALITY_FLAG:
                        del ds[each_var].attrs[DataVars.DESCRIPTION_ATTR]

                        ds[each_var].attrs[_flag_values] = _binary_flags
                        ds[each_var].attrs[_flag_meanings] = _binary_meanings

                        # 7. Set quality_flag to "0" (no data) for time coordinates that correspond to
                        #    missing 1990-1992 data.
                        time_values = ds.time.values

                        empty_data = [(index, t) for index, t in enumerate(time_values) if str(t).startswith(_empty_data_t)]
                        logging.info(f'Got {len(empty_data)} entries for empty data in time dimension')

                        empty_array = np.full((sizes[Coords.Y], sizes[Coords.X]), 0)

                        for each in empty_data:
                            i = each[0]
                            logging.info(f'Setting {Vars.QUALITY_FLAG} to no data for {each} ...')
                            ds[Vars.QUALITY_FLAG][i, :, :] = empty_array

                        non_empty_data = [(index, t) for index, t in enumerate(time_values) if not str(t).startswith(_empty_data_t)]
                        logging.info(f'Got {len(non_empty_data)} entries for empty data in time dimension')

                        for each in non_empty_data:
                            index = each[0]
                            logging.info(f'Fixing {Vars.QUALITY_FLAG} values for {each} ...')

                            values = ds[Vars.QUALITY_FLAG][index, :, :].values

                            # Set NaNs to zero
                            nan_mask = np.isnan(values)
                            values[nan_mask] = 0

                            # Get masks for values of 2 and 3, switch the values:
                            # from current
                            #   "pole hole = 2, low-quality = 3"
                            # to
                            #   "2 = low quality data, 3 = pole hole"
                            #
                            # (
                            #   according to the "description" attribute and the paper:
                            #   0 = no data, 1 = high quality data, 2 = low quality data, 3 = pole hole
                            # )
                            two_mask = (values == 2)
                            three_mask = (values == 3)
                            values[two_mask] = 3
                            values[three_mask] = 2

                            # Reset to fixed values
                            ds[Vars.QUALITY_FLAG][index, :, :] = values

                    # 5. Add missing description of the var:
                    if each_var in _desc:
                        ds[each_var].attrs[DataVars.DESCRIPTION_ATTR] = _desc[each_var]

                    # 6. Fix various attributes of "mapping" variable
                    if each_var == DataVars.MAPPING:
                        # Optional attribute; but if to be included, set to 6378137.0
                        ds[each_var].attrs[Mapping.SEMI_MAJOR_AXIS] = 6378137.0

                        # Optional attribute; remove attribute (Danica Linda Cantarero@NSIDC: I can't find the correct value)
                        if Mapping.SEMI_MINOR_AXIS in ds[each_var].attrs:
                            del ds[each_var].attrs[Mapping.SEMI_MINOR_AXIS]

                        # Edit attribute name from spatial_proj4 to proj4text
                        if Mapping.SPATIAL_PROJ4 in ds[each_var].attrs:
                            ds[each_var].attrs[Mapping.PROJ4TEXT] = ds[each_var].attrs[Mapping.SPATIAL_PROJ4]
                            del ds[each_var].attrs[Mapping.SPATIAL_PROJ4]

                        # Adding crs_wkt (redundant with spatial_ref) expands interoperability
                        # with geolocation tools:
                        if (Mapping.CRS_WKT not in ds[each_var].attrs) and (Mapping.SPATIAL_REF in ds[each_var].attrs):
                            ds[each_var].attrs[Mapping.CRS_WKT] = ds[each_var].attrs[Mapping.SPATIAL_REF]

                        # - 6a. Replace:
                        # :latitude_of_origin = -71.0; // double
                        #   with:
                        # :standard_parallel = -71.0;
                        del ds[each_var].attrs[_latitude_of_origin]
                        ds[each_var].attrs[Mapping.STANDARD_PARALLEL] = -71.0

                    # 9. Rename some of the data variables
                    if each_var in Vars.OldToNewMap:
                        new_name = Vars.OldToNewMap[each_var]
                        ds[new_name] = ds[each_var]
                        del ds[each_var]

                # # Original mapping data variable of string type - make it
                # del ds[DataVars.MAPPING]
                ds[DataVars.MAPPING] = xr.DataArray(
                    attrs=ds[DataVars.MAPPING].attrs,
                    coords={},
                    dims=[]
                )

                # Convert dataset to Dask dataset not to run out of memory while writing to the file
                ds = ds.chunk(chunks={Coords.X: chunk_size, Coords.Y: chunk_size, Vars.TIME: 1})

                # 8. Set compression
                for each_var in [
                    Vars.HEIGHT,
                    Vars.HEIGHT_CHANGE,
                    Vars.HEIGHT_CHANGE_RMSE,
                    Vars.GLACIER_BASIN,
                    Vars.QUALITY_FLAG
                ]:
                    ds[each_var].encoding.update(ElevationCompression)

                # Set _FillValue to None explicitly for x, y, time and mapping
                # to avoid xarray adding it at write time
                for each_var in [Vars.TIME, Coords.X, Coords.Y, DataVars.MAPPING]:
                    ds[each_var].encoding[Output.FILL_VALUE_ATTR] = None

                # # Set chunking for 3d vars, and add NC packing variables
                # # (scale_factor and add_offset) if present in original data
                # chunks = (1, chunk_size, chunk_size)
                #
                # for each in [Vars.DH, Vars.RMSE, Vars.QUALITY_FLAG]:
                #     enc_settings[each][Output.CHUNKSIZES_ATTR] = chunks
                #
                #     # Copy scale_factor and add_offset if they are present in
                #     # original data
                #     if _scale_factor in ds[each].encoding:
                #         enc_settings[each][_scale_factor] = ds[each].encoding[_scale_factor]
                #         enc_settings[each][_add_offset] = ds[each].encoding[_add_offset]
                #
                # chunks = (chunk_size, chunk_size)
                # for each in [Vars.H, Vars.BASIN]:
                #     enc_settings[each][Output.CHUNKSIZES_ATTR] = chunks

                msgs.append(f'Writing to {filename}')
                # msgs.append(f'Encoding: {enc_settings}')

                # Write fixed elevation to local file
                # ds.to_netcdf(filename, engine='h5netcdf', encoding = enc_settings)
                ds.to_netcdf(filename, engine='h5netcdf')

                # Copy new granule to S3 bucket
                msgs.extend(
                    NSIDCFormat.upload_to_s3(filename, target_dir, target_bucket, s3_client, remove_original_file=False)
                )

                # Remove local copy of the file
                msgs.append(f"Removing local {local_file}")
                os.unlink(local_file)

                # Time values as datatime objects
                time_values = [t.astype('M8[ms]').astype('O') for t in ds[Vars.TIME].values]

                # Create spacial and premet metadata files, and copy them to S3 bucket
                meta_file = NSIDCElevationMeta.create_premet_file(filename, time_values)
                msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

                # Spatial file can have pole wrap around issues, check for that
                is_valid, meta_file = NSIDCElevationMeta.create_spatial_file(filename)
                if is_valid:
                    # Upload to S3 only if it's a valid file, otherwise - fix the file
                    # manually and push it to S3 manually
                    msgs.extend(NSIDCFormat.upload_to_s3(meta_file, target_dir, target_bucket, s3_client))

                msgs.append(f"Removing locally fixed {filename}")
                os.unlink(filename)

        return msgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
           Fix ITS_LIVE V1 elevation to be CF compliant for ingestion by NSIDC.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-bucket',
        type=str,
        default='its-live-data',
        help='AWS S3 bucket to store ITS_LIVE granules to [%(default)s]'
    )

    parser.add_argument(
        '-source_dir',
        type=str,
        default='elevation/v01',
        help='AWS S3 directory that stores input elevation files in NetCDF format [%(default)s]'
    )

    parser.add_argument(
        '-target_dir',
        type=str,
        default='NSIDC/v01/elevation-latest',
        help='AWS S3 directory that stores processed elevation files [%(default)s]'
    )

    parser.add_argument(
        '-chunk_by',
        action='store',
        type=int,
        default=4,
        help='Number of elevation files to process in parallel [%(default)d]'
    )

    parser.add_argument(
        '-w', '--dask_workers',
        type=int,
        default=4,
        help='Number of Dask parallel workers for processing [%(default)d]'
    )

    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually process any elevation files'
    )

    args = parser.parse_args()

    NSIDCFormat.DRY_RUN = args.dryrun

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')

    nsidc_format = NSIDCElevationFormat(args.bucket, args.source_dir)

    nsidc_format(
        args.bucket,
        args.target_dir,
        args.chunk_by,
        args.dask_workers
    )

    logging.info('Done.')
