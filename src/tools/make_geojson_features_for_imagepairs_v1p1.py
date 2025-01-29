"""
Script to generate catalog geojson file for ITS_LIVE granule dataset.

Authors: Mark Fahnestock, Masha Liukis
"""

import argparse
import boto3
import dask
from dask.diagnostics import ProgressBar
from datetime import datetime
import geojson
import h5py
import json
import logging
import numpy as np
import os
# import psutil
import pyproj
import s3fs
import time
from tqdm import tqdm
from shapely.geometry import mapping

# Local imports
import nsidc_meta_files


# Date format as it appears in granules filenames of optical format:
# LC08_L1TP_011002_20150821_20170405_01_T1_X_LC08_L1TP_011002_20150720_20170406_01_T1_G0240V01_P038.nc
DATE_FORMAT = "%Y%m%d"

# Date and time format as it appears in granules filenames or radar format:
# S1A_IW_SLC__1SSH_20170221T204710_20170221T204737_015387_0193F6_AB07_X_S1B_IW_SLC__1SSH_20170227T204628_20170227T204655_004491_007D11_6654_G0240V02_P094.nc
DATE_TIME_FORMAT = "%Y%m%dT%H%M%S"

# Number of retries when encountering AWS S3 download/upload error
_NUM_AWS_COPY_RETRIES = 3

# Number of seconds between retries to access AWS S3 bucket
_AWS_COPY_SLEEP_SECONDS = 3


def get_tokens_from_filename(filename):
    """
    Extract acquisition/processing dates and path/row for two images from the
    optical granule filename, or start/end date/time and product unique ID for
    radar granule filename.
    """
    # Optical format granules have different file naming convention than radar
    # format granules
    is_optical = True
    url_files = os.path.basename(filename).split('_X_')

    # Get tokens for the first image name
    url_tokens = url_files[0].split('_')

    if len(url_tokens) < 9:
        # Optical format granule
        # Get acquisition/processing dates and path&row for both images
        first_date_1 = datetime.strptime(url_tokens[3], DATE_FORMAT)
        second_date_1 = datetime.strptime(url_tokens[4], DATE_FORMAT)
        key_1 = url_tokens[2]

        url_tokens = url_files[1].split('_')
        first_date_2 = datetime.strptime(url_tokens[3], DATE_FORMAT)
        second_date_2 = datetime.strptime(url_tokens[4], DATE_FORMAT)
        key_2 = url_tokens[2]

    else:
        # Radar format granule
        # Get start/end date/time and product unique ID for both images
        is_optical = False

        url_tokens = url_files[0].split('_')
        # Start date and time
        first_date_1 = datetime.strptime(url_tokens[-5], DATE_TIME_FORMAT)
        # Stop date and time
        second_date_1 = datetime.strptime(url_tokens[-4], DATE_TIME_FORMAT)
        # Product unique identifier
        key_1 = url_tokens[-1]

        # Get tokens for the second image name: there are two extra tokens
        # at the end of the filename which are specific to ITS_LIVE filename
        url_tokens = url_files[1].split('_')
        # Start date and time
        first_date_2 = datetime.strptime(url_tokens[-7], DATE_TIME_FORMAT)
        # Stop date and time
        second_date_2 = datetime.strptime(url_tokens[-6], DATE_TIME_FORMAT)
        # Product unique identifier
        key_2 = url_tokens[-3]

    return is_optical, first_date_1, second_date_1, key_1, first_date_2, second_date_2, key_2


def skip_duplicate_granules(found_urls: list):
    """
    Skip duplicate granules (the ones that have earlier processing date(s)).
    """
    # Need to remove duplicate granules for the middle date: some granules
    # have newer processing date, keep those.
    keep_urls = {}
    skipped_double_granules = []

    for each_url in tqdm(found_urls, ascii=True, desc='Skipping duplicate granules...'):
        # Extract acquisition and processing dates for optical granule,
        # start/end date/time and product unique ID for radar granule
        is_optical, url_acq_1, url_proc_1, key_1, url_acq_2, url_proc_2, key_2 = \
            get_tokens_from_filename(each_url)

        if is_optical:
            # Acquisition time and path/row of images should be identical for
            # duplicate granules
            granule_id = '_'.join([
                url_acq_1.strftime(DATE_FORMAT),
                key_1,
                url_acq_2.strftime(DATE_FORMAT),
                key_2
            ])

        else:
            # Start/stop date/time of both images
            granule_id = '_'.join([
                url_acq_1.strftime(DATE_TIME_FORMAT),
                url_proc_1.strftime(DATE_TIME_FORMAT),
                url_acq_2.strftime(DATE_TIME_FORMAT),
                url_proc_2.strftime(DATE_TIME_FORMAT),
            ])

        # There is a granule for the mid_date already:
        # * For radar granule: issue a warning reporting product unique ID for duplicate granules
        # * For optical granule: check which processing time is newer,
        #                        keep the one with newer processing date
        if granule_id in keep_urls:
            if not is_optical:
                # Radar format granule, just issue a warning
                all_urls = ' '.join(keep_urls[granule_id])
                logging.info(f"WARNING: multiple granules are detected for {each_url}: {all_urls}")
                keep_urls[granule_id].append(each_url)
                continue

            # Process optical granule
            # Flag if newly found URL should be kept
            keep_found_url = False

            for found_url in keep_urls[granule_id]:
                # Check already found URLs for processing time
                _, _, found_proc_1, _, _, found_proc_2, _ = \
                    get_tokens_from_filename(found_url)

                # If both granules have identical processing time,
                # keep them both - granules might be in different projections,
                # any other than target projection will be handled later
                if url_proc_1 == found_proc_1 and \
                   url_proc_2 == found_proc_2:
                    keep_urls[granule_id].append(each_url)
                    keep_found_url = True
                    break

            # There are no "identical" (same acquision and processing times)
            # granules to "each_url", check if new granule has newer processing dates
            if not keep_found_url:
                # Check if any of the found URLs have older processing time
                # than newly found URL
                remove_urls = []
                for found_url in keep_urls[granule_id]:
                    # Check already found URL for processing time
                    _, _, found_proc_1, _, _, found_proc_2, _ = \
                        get_tokens_from_filename(found_url)

                    if url_proc_1 >= found_proc_1 and \
                       url_proc_2 >= found_proc_2:
                        # The granule will need to be replaced with a newer
                        # processed one
                        remove_urls.append(found_url)

                    elif url_proc_1 > found_proc_1:
                        # There are few cases when proc_1 is newer in
                        # each_url and proc_2 is newer in found_url, then
                        # keep the granule with newer proc_1
                        remove_urls.append(found_url)

                if len(remove_urls):
                    # Some of the URLs need to be removed due to newer
                    # processed granule
                    logging.info(f"Skipping {remove_urls} in favor of new {each_url}")
                    skipped_double_granules.extend(remove_urls)

                    # Remove older processed granules based on dates for "each_url"
                    keep_urls[granule_id][:] = [each for each in keep_urls[granule_id] if each not in remove_urls]
                    # Add new granule with newer processing date
                    keep_urls[granule_id].append(each_url)

                else:
                    # New granule has older processing date, don't include
                    logging.info(f"Skipping new {each_url} in favor of {keep_urls[granule_id]}")
                    skipped_double_granules.append(each_url)

        else:
            # This is a granule for new ID, append it to URLs to keep
            keep_urls.setdefault(granule_id, []).append(each_url)

    granules = []
    for each in keep_urls.values():
        granules.extend(each)

    logging.info(f"Keeping {len(granules)} unique granules, skipping {len(skipped_double_granules)} granules")
    return granules, skipped_double_granules


# class memtracker:

#     def __init__(self, include_time=True):
#         self.output_time = include_time
#         if include_time:
#             self.start_time = time.time()
#         self.process = psutil.Process()
#         self.startrss = self.process.memory_info().rss
#         self.startvms = self.process.memory_info().vms

#     def meminfo(self, message):
#         if self.output_time:
#             time_elapsed_seconds = time.time() - self.start_time
#             print(f'{message:<30}:  time: {time_elapsed_seconds:8.2f} seconds    mem_percent {self.process.memory_percent()} ' +
#                     f'delrss={self.process.memory_info().rss - self.startrss:16,}    ' +
#                     f'delvms={self.process.memory_info().vms - self.startvms:16,}',
#                     flush=True)
#         else: # don't output time
#             print(f'{message:<30}:  delrss={self.process.memory_info().rss - self.startrss:16,}   mem_percent {self.process.memory_percent()} ' +
#                     f'delvms={self.process.memory_info().vms - self.startvms:16,}',
#                     flush=True)

# mt = memtracker()

class GranuleCatalog:
    """
    Class to build ITS_LIVE granule catalog in geojson format for ingest by
    the webside DB.
    """
    FIVE_POINTS_PER_SIDE = True
    DATA_VERSION = None
    EXCLUDE_GRANULES_FILE = None
    REMOVE_DUPLICATE_GRANULES = False

    def __init__(self, granules_file: str, features_per_file: int, catalog_dir: str, start_index: int = 0):
        """
        Initialize the object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        # read in granule file list from S3 file
        self.infiles = None
        logging.info(f"Opening granules file: {granules_file}")
        with self.s3.open(granules_file, 'r') as ins3file:
            self.infiles = json.load(ins3file)
            logging.info(f"Loaded {len(self.infiles)} granules from '{granules_file}'")

        if GranuleCatalog.EXCLUDE_GRANULES_FILE is not None:
            # Exclude known granules from new catalog geojson files
            exclude_files = []
            exclude_file_path = os.path.join(catalog_dir, GranuleCatalog.EXCLUDE_GRANULES_FILE)
            logging.info(f"Opening file with granules to exclude: {exclude_file_path}")

            with self.s3.open(exclude_file_path, 'r') as fhandle:
                exclude_files = json.load(fhandle)
                logging.info(f"Loaded {len(exclude_files)} granules from '{exclude_file_path}' to exclude ")

                self.infiles = list(set(self.infiles).difference(exclude_files))
                logging.info(f"{len(self.infiles)} new granules to catalog")

        # Sort self.infiles to guarantee the same order of granules if have to pick the processing
        # somewhere in a middle (previous processing failed due to some exception)
        self.infiles = sorted(self.infiles)

        if start_index != 0:
            # Start index is provided for the granule to begin with
            self.infiles = self.infiles[start_index:]
            logging.info(f"Starting with granule #{start_index}, remains {len(self.infiles)} granules to catalog")

        self.features_per_file = features_per_file
        self.catalog_dir = catalog_dir

    def create(self, chunk_size, num_dask_workers, granules_dir, file_start_index=0):
        """
        Create catalog geojson file.
        """
        # read in granule file list from S3 file
        total_num_files = len(self.infiles)
        # mt.meminfo(f'Working on {total_num_files} total files')

        init_total_files = total_num_files

        if total_num_files <= 0:
            logging.info("Nothing to catalog, exiting.")
            return

        start = 0                              # Current start index into global list
        read_num_files = 0                     # Number of read files within the block
        block_start = file_start_index         # Current start index for the block to write to file
        cum_read_num_files = file_start_index  # Cumulative number of processed granules

        base_dir = os.path.basename(granules_dir)

        feature_list = []
        while total_num_files > 0:
            num_tasks = chunk_size if total_num_files > chunk_size else total_num_files

            logging.info(f"Starting granules {start}:{start+num_tasks} out of {init_total_files} total granules")
            tasks = [
                dask.delayed(GranuleCatalog.image_pair_feature_from_path)(each, self.s3) for
                each in self.infiles[start:start+num_tasks]
            ]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=num_dask_workers
                )

            feature_list.extend(results[0])

            total_num_files -= num_tasks
            read_num_files += num_tasks
            cum_read_num_files += num_tasks
            start += num_tasks

            # Check if need to write to the file:
            if read_num_files >= self.features_per_file or total_num_files <= 0:
                # Use sub-directory name of input path as base for output filename
                featureColl = geojson.FeatureCollection(feature_list)
                outfilename = f'imgpair_{base_dir}_{block_start}_{cum_read_num_files-1}.json'

                with s3_out.open(f'{self.catalog_dir}/{outfilename}', 'w') as outf:
                    # ATTN: Use shapely.geometry.mapping to write geojson to the file.
                    # Using geojson.dump() raises ValueError: Out of range float values are not JSON compliant: nan
                    # for dictionaries with nan's (newly introduced stable_shift
                    # can be set to NaN)
                    # WAS: geojson.dump(featureColl, outf)
                    json.dump(mapping(featureColl), outf)

                # mt.meminfo(f'wrote {args.catalog_dir}/{outfilename}')
                logging.info(f'Wrote {self.catalog_dir}/{outfilename}')
                feature_list = []
                featureColl = None

                read_num_files = 0
                block_start = cum_read_num_files

    @staticmethod
    def get_h5_attribute_value(h5_attr):
        """
        Extract value of the hd5 data variable attribute.
        """
        value = None
        if isinstance(h5_attr, str):
            value = h5_attr

        elif isinstance(h5_attr, bytes):
            value = h5_attr.decode('utf-8')  # h5py returns byte values, turn into byte characters

        elif h5_attr.shape == ():
            value = h5_attr

        else:
            value = h5_attr[0]  # h5py returns lists of numbers - all 1 element lists here, so dereference to number

        return value

    @staticmethod
    def image_pair_feature_from_path(infilewithpath: str, s3: s3fs.S3FileSystem):
        """Generate geojson feature for the image pair, and create NSIDC metadata files.
           Publish metadata files to the target directory in S3 bucket.

        Args:
            infilewithpath (str): Fullpath to the granule file in S3 bucket.
            s3 (s3fs.S3FileSystem): s3fs object to access S3 bucket.

        Returns:
            geojson.Feature: Geojson feature for the granule.

        Raises:
            RuntimeError: If an error occurs while processing the granule file.
        """
        filename_tokens = infilewithpath.split('/')
        directory = '/'.join(filename_tokens[1:-1])
        filename = filename_tokens[-1]
        target_bucket = filename_tokens[0]

        data_version = GranuleCatalog.DATA_VERSION
        if data_version is None:
            data_version = filename_tokens[-2]

        stable_shift_value = np.nan
        v_error_max = np.nan

        with s3.open(f"s3://{infilewithpath}", "rb") as ins3:
            inh5 = h5py.File(ins3, mode='r')
            # netCDF4/HDF5 cf 1.6 has x and y vectors of array pixel CENTERS
            xvals = np.array(inh5.get('x'))
            yvals = np.array(inh5.get('y'))

            # Extract projection variable
            projection_cf = None
            if 'mapping' in inh5:
                projection_cf = inh5['mapping']

            elif 'UTM_Projection' in inh5:
                projection_cf = inh5['UTM_Projection']

            elif 'Polar_Stereographic' in inh5:
                projection_cf = inh5['Polar_Stereographic']

            imginfo_attrs = inh5['img_pair_info'].attrs
            # turn hdf5 img_pair_info attrs into a python dict to save below
            img_pair_info_dict = {}
            try:
                for k in imginfo_attrs.keys():
                    img_pair_info_dict[k] = GranuleCatalog.get_h5_attribute_value(imginfo_attrs[k])

            except Exception as exc:
                raise RuntimeError(f'Error processing {infilewithpath}: img_pair_info.{k}: {imginfo_attrs[k]} type={type(imginfo_attrs[k])} exc={exc} ({imginfo_attrs})')

            minval_x, pix_size_x, rot_x_ignored, maxval_y, rot_y_ignored, pix_size_y = [float(x) for x in projection_cf.attrs['GeoTransform'].split()]

            epsgcode = int(GranuleCatalog.get_h5_attribute_value(projection_cf.attrs['spatial_epsg']))

            # Maximum v error
            v_error_max = max(
                GranuleCatalog.get_h5_attribute_value(inh5['vx'].attrs['error']),
                GranuleCatalog.get_h5_attribute_value(inh5['vy'].attrs['error'])
            )
            vx_stable_shift = GranuleCatalog.get_h5_attribute_value(inh5['vx'].attrs['stable_shift'])
            vy_stable_shift = GranuleCatalog.get_h5_attribute_value(inh5['vy'].attrs['stable_shift'])

            if not np.isnan(vx_stable_shift) and not np.isnan(vy_stable_shift):
                # stable_shift = rms of vx.stable_shift and vy.stable_shift
                stable_shift_value = np.sqrt((vx_stable_shift**2 + vy_stable_shift**2)/2)

            inh5.close()

        # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
        projection_cf_minx = xvals[0] - pix_size_x/2.0
        projection_cf_maxx = xvals[-1] + pix_size_x/2.0
        projection_cf_miny = yvals[-1] + pix_size_y/2.0  # pix_size_y is negative!
        projection_cf_maxy = yvals[0] - pix_size_y/2.0   # pix_size_y is negative!

        transformer = pyproj.Transformer.from_crs(f"EPSG:{epsgcode}", "EPSG:4326", always_xy=True)  # ensure lonlat output order

        ll_lonlat = np.round(transformer.transform(projection_cf_minx, projection_cf_miny), decimals=7).tolist()
        lr_lonlat = np.round(transformer.transform(projection_cf_maxx, projection_cf_miny), decimals=7).tolist()
        ur_lonlat = np.round(transformer.transform(projection_cf_maxx, projection_cf_maxy), decimals=7).tolist()
        ul_lonlat = np.round(transformer.transform(projection_cf_minx, projection_cf_maxy), decimals=7).tolist()

        # find center lon lat for inclusion in feature (to determine lon lat grid cell directory)
        #     projection_cf_centerx = (xvals[0] + xvals[-1])/2.0
        #     projection_cf_centery = (yvals[0] + yvals[-1])/2.0
        center_lonlat = np.round(transformer.transform((xvals[0] + xvals[-1])/2.0, (yvals[0] + yvals[-1])/2.0), decimals=7).tolist()

        if GranuleCatalog.FIVE_POINTS_PER_SIDE:
            fracs = [0.25, 0.5, 0.75]
            polylist = []  # ring in counterclockwise order

            polylist.append(ll_lonlat)
            dx = projection_cf_maxx - projection_cf_minx
            dy = projection_cf_miny - projection_cf_miny
            for frac in fracs:
                polylist.append(np.round(transformer.transform(projection_cf_minx + (frac * dx), projection_cf_miny + (frac * dy)), decimals=7).tolist())

            polylist.append(lr_lonlat)
            dx = projection_cf_maxx - projection_cf_maxx
            dy = projection_cf_maxy - projection_cf_miny
            for frac in fracs:
                polylist.append(np.round(transformer.transform(projection_cf_maxx + (frac * dx), projection_cf_miny + (frac * dy)), decimals=7).tolist())

            polylist.append(ur_lonlat)
            dx = projection_cf_minx - projection_cf_maxx
            dy = projection_cf_maxy - projection_cf_maxy
            for frac in fracs:
                polylist.append(np.round(transformer.transform(projection_cf_maxx + (frac * dx), projection_cf_maxy + (frac * dy)), decimals=7).tolist())

            polylist.append(ul_lonlat)
            dx = projection_cf_minx - projection_cf_minx
            dy = projection_cf_miny - projection_cf_maxy
            for frac in fracs:
                polylist.append(np.round(transformer.transform(projection_cf_minx + (frac * dx), projection_cf_maxy + (frac * dy)), decimals=7).tolist())

            polylist.append(ll_lonlat)

        else:
            # only the corner points
            polylist = [ll_lonlat, lr_lonlat, ur_lonlat, ul_lonlat, ll_lonlat]

        poly = geojson.Polygon([polylist])

        middate = img_pair_info_dict['date_center']
        deldays = img_pair_info_dict['date_dt']
        percent_valid_pix = img_pair_info_dict['roi_valid_percentage']

        feat = geojson.Feature(
            geometry=poly,
            properties={
                'filename': filename,
                'directory': directory,
                'middate': middate,
                'deldays': deldays,
                'percent_valid_pix': percent_valid_pix,
                'center_lonlat': center_lonlat,
                'data_epsg': epsgcode,
                # date_deldays_strrep is a string version of center date and time interval
                # that will sort by date and then by interval length (shorter intervals first) -
                # relies on "string" comparisons by byte
                'date_deldays_strrep': img_pair_info_dict['date_center'] + f"{img_pair_info_dict['date_dt']:07.1f}".replace('.', ''),
                'img_pair_info_dict': img_pair_info_dict,
                'v_error_max': v_error_max,
                'stable_shift': stable_shift_value,
                'version': data_version
            }
        )

        # Create NSIDC metadata files and copy them to the
        # granule directory in S3 bucket

        # Automatically handle aws exceptions on file read and upload from/to s3 bucket
        files_are_copied = False
        num_retries = 0

        while not files_are_copied and num_retries < _NUM_AWS_COPY_RETRIES:
            try:
                meta_files = nsidc_meta_files.create_nsidc_meta_files(infilewithpath, s3)

                s3_client = boto3.client('s3')

                for each_file in meta_files:
                    # Place metadata files into the same s3 directory as the granule
                    nsidc_meta_files.upload_to_s3(each_file, directory, target_bucket, s3_client)

                files_are_copied = True

            except:
                # msgs.append(f"Try #{num_retries + 1} exception processing {infilewithpath}: {sys.exc_info()}")
                num_retries += 1

                if num_retries < _NUM_AWS_COPY_RETRIES:
                    # Retry the copy for any kind of failure

                    # Sleep if it's not a last attempt to copy
                    time.sleep(_AWS_COPY_SLEEP_SECONDS)

                else:
                    # Don't retry, trigger an exception
                    num_retries = _NUM_AWS_COPY_RETRIES
                    raise

        return (feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""make_geojson_features_for_imagepairs_v1.py

        Produces output geojson FeatureCollection for each nn image_pairs from a directory.
        v1 adds 5 points per side to geom (so 3 interior and the two corners from v0)
        and the ability to stop the chunks (in addition to the start allowed in v0)
        so that the code can be run on a range of chunks.
        """,
        epilog="""
    There are two steps to create geojson catalogs:
    1. Create a list of granules to be used for catalog generation. The file that stores
        URLs of such granules is placed in the destination S3 bucket.
    2. Create geojson catalogs using a list of granules as generated by step #1. The list of
        granules is read from the file stored in the destination S3 bucket.""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-granule_dir',
                        action='store',
                        type=str,
                        default='its-live-data/velocity_image_pair/landsatOLI/v02',
                        help='S3 path to tile catalog directories (not including the grid code for zone of tile) [%(default)s]')

    parser.add_argument('-catalog_dir',
                        action='store',
                        type=str,
                        default='its-live-data/catalog_geojson/landsatOLI/v02',
                        help='Output path for feature collections [%(default)s]')

    parser.add_argument('-chunk_by',
                        action='store',
                        type=int,
                        default=1000,
                        help='Chunk feature collections to have chunk_by features each [%(default)d]')

    parser.add_argument('-features_per_file',
                        action='store',
                        type=int,
                        default=20000,
                        help='Number of features to store per the file [%(default)d]')

    parser.add_argument('-skipped_granules_file',
                        action='store',
                        type=str,
                        default='skipped_granules.json',
                        help='Filename to keep track of skipped duplicate granules [%(default)s], file is stored in "-catalog_dir"')

    parser.add_argument('-catalog_granules_file',
                        action='store',
                        type=str,
                        default='used_granules.json',
                        help='Filename to keep track of granules used for the geojson catalog [%(default)s], file is stored in  "-catalog_dir"')

    parser.add_argument('-exclude_granules_file',
                        action='store',
                        type=str,
                        default=None,
                        help='Name of the file with granules to exclude from the geojson catalog [%(default)s], file is stored in  "-catalog_dir"')

    parser.add_argument('-file_start_index',
                        action='store',
                        type=int,
                        default=0,
                        help="Start index to use when formatting first catalog geojson filename [%(default)d]. "
                                "Usefull if adding new granules to existing set of catalog geojson files.")

    parser.add_argument('-granule_start_index',
                        action='store',
                        type=int,
                        default=0,
                        help="Start index for the granule to begin cataloging with [%(default)d]. "
                                "Useful if continuing interrupted process cataloging the files.")

    parser.add_argument('-c', '--create_catalog_list',
                        action='store_true',
                        help='build a list of granules for catalog generation [%(default)s], '
                                'otherwise read the list of granules from catalog_granules_file')

    parser.add_argument('-glob',
                        action='store',
                        type=str,
                        default='*/*.nc',
                        help='Glob pattern for the granule search under "base_dir_s3fs" [%(default)s]')

    parser.add_argument('-five_points_per_side', action='store_true',
                        help='Define 5 points per side before re-projecting granule polygon to longitude/latitude coordinates')

    parser.add_argument('-remove-duplicate-granules', action='store_true',
                        help='Remove duplicate granules based on processing date within granule filename. '
                            'This option should be used for Landsat8 data only.')

    parser.add_argument('-data_version',
                        default=None,
                        type=str,
                        help='Data version to be recorded for each granule [%(default)s]. '
                            'If none is provided, immediate parent directory of the granule is used as its version.')

    parser.add_argument('-w', '--dask_workers', type=int,
                        default=4,
                        help='Number of Dask parallel workers [%(default)d]')

    args = parser.parse_args()

    GranuleCatalog.FIVE_POINTS_PER_SIDE = args.five_points_per_side
    GranuleCatalog.DATA_VERSION = args.data_version
    GranuleCatalog.EXCLUDE_GRANULES_FILE = args.exclude_granules_file
    GranuleCatalog.REMOVE_DUPLICATE_GRANULES = args.remove_duplicate_granules

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f'Command-line args: {args}')
    s3_out = s3fs.S3FileSystem()

    granules_dir = args.granule_dir

    if not args.create_catalog_list:
        catalog = GranuleCatalog(
            os.path.join(args.catalog_dir, args.catalog_granules_file),
            args.features_per_file,
            args.catalog_dir,
            args.granule_start_index
        )

        catalog.create(
            args.chunk_by,
            args.dask_workers,
            granules_dir,
            args.file_start_index
        )

    else:
        # Check if catalog_granules_file exists - report to avoid overwrite
        granules_file = s3_out.glob(os.path.join(args.catalog_dir, args.catalog_granules_file))
        if len(granules_file):
            raise RuntimeError(f"{os.path.join(args.catalog_dir, args.catalog_granules_file)} already exists.")

        # Create a list of granules to catalog and store it in S3 bucket
        # use a glob to list directory
        logging.info("Creating a list of granules to catalog")

        logging.info(f"Glob {granules_dir}/{args.glob}")
        infilelist = s3_out.glob(f'{granules_dir}/{args.glob}')

        logging.info(f"Got {len(infilelist)} granules")

        # check for '_P' in filename - filters out temp.nc files that can be left by bad transfers
        # also skips txt file placeholders for 000 Pct (all invalid) pairs
        infiles = [x for x in infilelist if '_P' in x and 'txt' not in x]

        # Skip duplicate granules (the same middle date, but different processing date) for L8/L9 data only
        # if GranuleCatalog.REMOVE_DUPLICATE_GRANULES:
        #     infiles, skipped_granules = ITSCube.skip_duplicate_l89_granules(infiles)

        #     granule_filename = os.path.join(args.catalog_dir, args.skipped_granules_file)
        #     with s3_out.open(granule_filename, 'w') as outf:
        #         geojson.dump(skipped_granules, outf)

        #     logging.info(f"Wrote skipped granules to '{granule_filename}'")

        #     # ATTN: If any of the skipped granules are already cataloged by
        #     # previous catalog generation, need to exclude them from existing
        #     # catalogs
        #     if GranuleCatalog.EXCLUDE_GRANULES_FILE is not None and len(skipped_granules):
        #         logging.info(f'WARNING: Need to check on exlusion of skipped granules from existing catalogs')

        # Write all unique granules to the file
        granule_filename = os.path.join(args.catalog_dir, args.catalog_granules_file)
        with s3_out.open(granule_filename, 'w') as outf:
            geojson.dump(infiles, outf)

        logging.info(f"Wrote catalog granules to '{granule_filename}'")

    logging.info('Done.')
