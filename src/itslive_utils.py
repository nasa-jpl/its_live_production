import boto3
import json
import requests
import pyproj
import numpy as np
import os
import logging
from rtree import index
import time
import sys
import subprocess
import zipfile

from grid import Bounds
import boto3
from rtree import index

BASE_URL = 'https://nsidc.org/apps/itslive-search/velocities/urls'
# BASE_URL = 'https://staging.nsidc.org/apps/itslive-search/velocities/urls'

# Number of 'aws s3 cp' retries in case of a failure
_NUM_AWS_COPY_RETRIES = 20

# Number of seconds to sleep between 'aws s3 cp' retries
_AWS_COPY_SLEEP_SECONDS = 60

# An error generated by AWS when PUT/GET request rate exceeds 3500
_AWS_SLOW_DOWN_ERROR = "An error occurred (SlowDown) when calling"


def get_min_lon_lat_max_lon_lat(coordinates: list):
    """
    Compute longitude and latitude extends for provided coordinates list.
    The coordinates are given in [longitude, latitude] order.

    Args:
    coordinates: list of lists - list of coordinates in [longitude, latitude] order.

    Returns: tuple of (min_lon, min_lat, max_lon, max_lat).
    """
    longitudes = [coord[0] for coord in coordinates]
    latitudes = [coord[1] for coord in coordinates]

    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    return (min_lon, min_lat, max_lon, max_lat)


def download_rtree_from_s3(s3_bucket, s3_key, local_path: str = None):
    """
    Download R-tree index file from AWS S3 bucket.

    Args:
    - s3_bucket: Name of the S3 bucket.
    - s3_key: Key of the R-tree file in the S3 bucket.
    - local_path: Local path to save the downloaded file. If None is provided,
            the file will be saved in the current directory with the same name as the S3 key.

    Returns: R-tree index object.
    """
    logging.info(f'Reading R-tree index from s3://{s3_bucket}/{s3_key}')

    for each_extension in ['.dat', '.idx']:
        s3_key_with_ext = s3_key + each_extension
        s3_client = boto3.resource('s3')

        local_file = local_path
        if local_file is None:
            local_file = os.path.basename(s3_key_with_ext)

        logging.info(f'Downloading {s3_key_with_ext} from {s3_bucket} to {local_file}')
        s3_client.Bucket(s3_bucket).download_file(s3_key_with_ext, local_file)

    # Open local version of the R-tree index
    return index.Index(os.path.basename(s3_key))


def query_rtree(rtree_idx, query_box, epsg_code, min_percent_valid_pix=1):
    """
    Query the R-tree for all files overlapping with the query bounding box.
    The query returns only files with at least min_percent_valid_pix valid pixels and
    with the same EPSG code as the query box.

    Args:
    - rtree_idx: R-tree index.
    - query_box: Bounding box to query (min_lon, min_lat, max_lon, max_lat)
    - epsg_code: Original EPSG code of the longitude, latitude coordinates in the query box.
    - min_percent_valid_pix: Minimum percentage of valid pixels in the granule to be considered.
        Default is 1%.

    Returns:
    - List of files names whose extents overlap with the query box.
    """
    # Query the R-tree for files that intersect with the query bounding box
    overlapping_files = list(rtree_idx.intersection(query_box, objects=True))

    # Return file names that overlap the query box and have at least 1% valid pixels
    return [item.object[0] for item in overlapping_files if item.object[1] >= min_percent_valid_pix and item.object[2] == epsg_code]


def s3_copy_using_subprocess(command_line: list, env_copy: dict, is_quiet: bool = True):
    """Copy file to/from aws s3 bucket.

    Args:
    command_line (list): List tokens for the command-line to invoke.
    env_copy (dict): Dictionary of environment variables set for the compute environment.
    is_quiet (bool): Flag if using "quiet" mode to reduce output clutter. Default is True.

    Raises:
        RuntimeError: Failure to copy the store if NUM_AWS_COPY_RETRIES attempts failed.
    """
    _quiet_flag = "--quiet"

    if is_quiet and _quiet_flag not in command_line:
        command_line.append(_quiet_flag)

    logging.info(f'aws s3 command: {" ".join(command_line)}')

    file_is_copied = False
    num_retries = 0
    command_return = None

    while not file_is_copied and num_retries < _NUM_AWS_COPY_RETRIES:
        logging.info(f"Attempt #{num_retries+1} to invoke: {' '.join(command_line)}")

        command_return = subprocess.run(
            command_line,
            env=env_copy,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        if command_return.returncode != 0:
            # Report the whole stdout stream as one logging message
            logging.warning(f"Failed to invoke: {' '.join(command_line)} with returncode={command_return.returncode}: {command_return.stdout}")

            num_retries += 1
            # If failed due to AWS SlowDown error, retry
            if num_retries != _NUM_AWS_COPY_RETRIES:
                # Possible to have some other types of failures that are not related to AWS SlowDown,
                # retry the copy for any kind of failure
                # and _AWS_SLOW_DOWN_ERROR in command_return.stdout.decode('utf-8'):

                # Sleep if it's not a last attempt to copy
                time.sleep(_AWS_COPY_SLEEP_SECONDS)

            else:
                # Don't retry, trigger an exception
                num_retries = _NUM_AWS_COPY_RETRIES

        else:
            file_is_copied = True

    if not file_is_copied:
        raise RuntimeError(f"Failed to invoke {' '.join(command_line)} with command.returncode={command_return.returncode}")


def transform_coord(proj1, proj2, lon, lat):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    # Convert coordinates
    return pyproj.transform(proj1, proj2, lon, lat)


def get_granule_urls(params):
    # Allow for longer query time from searchAPI: 10 minutes
    resp = requests.get(BASE_URL, params=params, verify=False, timeout=500)
    return resp.json()


def get_granule_urls_compressed(params, total_retries=1, num_seconds=30):
    """
    Request granules URLs with ZIP compression enabled, save the stream to the ZIP file,
    and retrieve JSON information from the archive.

    params: request parameters
    total_retries: number of retries to query searchAPI in a case of exception.
                    Default is 1.
    num_seconds: number of seconds to sleep between retries to query searchAPI.
                    Default is 30 seconds.
    """
    # Format request URL:
    url = f'{BASE_URL}?'
    for each_key, each_value in params.items():
        url += f'{each_key}={each_value}&'

    # Add compression option
    url += 'compressed=true&'

    # Add requested granules version (TODO: should be configurable on startup?)
    url += 'version=2'

    # Get rid of all single quotes if any in URL
    url = url.replace("'", "")

    num_retries = 0
    got_granules = False
    data = []

    # Save response to local file:
    local_path = 'searchAPI_urls.json.zip'

    logging.info(f'Submitting searchAPI request with url={url}')

    while not got_granules and num_retries < total_retries:
        # Get list of granules:
        try:
            logging.info(f"Getting granules from searchAPI: #{num_retries+1} attempt")
            num_retries += 1

            resp = requests.get(url, stream=True, timeout=500)

            logging.info(f'Saving searchAPI response to {local_path}')
            with open(local_path, 'wb') as fh:
                for chunk in resp.iter_content(10240, decode_unicode=False):
                    _ = fh.write(chunk)

            # Unzip the file
            with zipfile.ZipFile(local_path, 'r') as fh:
                zip_json_file = fh.namelist()[0]
                logging.info(f'Extracting {zip_json_file}')

                with fh.open(zip_json_file) as fh_json:
                    data = json.load(fh_json)

                    got_granules = True

        except:
            # If failed due to response truncation or searchAPI not being able to respond
            # (too many requests at the same time?)
            logging.info(f'Got exception: {sys.exc_info()}')
            if num_retries < total_retries:
                # Sleep if it's not last attempt
                logging.info(f'Sleeping between searchAPI attempts for {num_seconds} seconds...')
                time.sleep(num_seconds)

        finally:
            # Clean up local file
            if os.path.exists(local_path):
                # Remove local file
                logging.info(f'Removing {local_path}')
                os.unlink(local_path)

    if not got_granules:
        raise RuntimeError("Failed to get granules from searchAPI.")

    return data


def get_granule_urls_streamed(params, total_retries=1, num_seconds=30):
    """
    Use streamed retrieval of the response from URL request.

    params: request parameters
    total_retries: number of retries to query searchAPI in a case of exception.
                    Default is 1.
    num_seconds: number of seconds to sleep between retries to query searchAPI.
                    Default is 30 seconds.
    """
    token = ']['

    # Format request URL:
    url = f'{BASE_URL}?'
    for each_key, each_value in params.items():
        url += f'{each_key}={each_value}&'

    # Add requested granules version (TODO: should be configurable on startup?)
    url += 'version=2'

    # Get rid of all single quotes if any in URL
    url = url.replace("'", "")

    num_retries = 0
    got_granules = False
    data = []

    # Save response to local file:
    local_path = 'searchAPI_urls.json'

    logging.info(f'Submitting searchAPI request with url={url}')

    while not got_granules and num_retries < total_retries:
        # Get list of granules:
        try:
            logging.info(f"Getting granules from searchAPI: #{num_retries+1} attempt")
            num_retries += 1

            resp = requests.get(url, stream=True, timeout=500)

            logging.info(f'Saving searchAPI response to {local_path}')
            with open(local_path, 'a') as fh:
                for chunk in resp.iter_content(10240, decode_unicode=True):
                    _ = fh.write(chunk)

            # Read data from local file
            data = ''
            with open(local_path) as fh:
                data = fh.readline()

            # if multiple json strings are returned,  then possible to see '][' within
            # the string, replace it by ','
            if token in data:
                logging.info('Got multiple json variables within the same string (len(data)={len(data)})')
                data = data.replace(token, ',')

                logging.info('Merged multiple json variables into one list (len(data)={len(data)})')

            data = json.loads(data)
            got_granules = True

        except:
            # If failed due to response truncation or searchAPI not being able to respond
            # (too many requests at the same time?)
            logging.info(f'Got exception: {sys.exc_info()}')
            if num_retries < total_retries:
                # Sleep if it's not last attempt
                logging.info(f'Sleeping between searchAPI attempts for {num_seconds} seconds...')
                time.sleep(num_seconds)

        finally:
            # Clean up local file
            if os.path.exists(local_path):
                # Remove local file
                logging.info(f'Removing {local_path}')
                os.unlink(local_path)

    if not got_granules:
        raise RuntimeError("Failed to get granules from searchAPI.")

    return data


#
# Author: Mark Fahnestock
#
def point_to_prefix(lat: float, lon: float, dir_path: str = None) -> str:
    """
    Returns a string (for example, N78W124) for directory name based on
    granule centerpoint lat,lon
    """
    NShemi_str = 'N' if lat >= 0.0 else 'S'
    EWhemi_str = 'E' if lon >= 0.0 else 'W'

    outlat = int(10*np.trunc(np.abs(lat/10.0)))
    if outlat == 90:  # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80

    outlon = int(10*np.trunc(np.abs(lon/10.0)))

    if outlon >= 180:  # if you are at the dateline, back off to the 170 bin
        outlon = 170

    dirstring = f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}'
    if dir_path is not None:
        dirstring = os.path.join(dir_path, dirstring)

    return dirstring


#
# Author: Mark Fahnestock, Masha Liukis
#
def add_five_points_to_polygon_side(polygon):
    """
    Define 5 points per each polygon side. This is done before re-projecting
    polygon to longitude/latitude coordinates.
    This function assumes rectangular polygon where min/max x/y define all
    4 polygon vertices.

    polygon: list of lists
        List of polygon vertices.
    """
    fracs = [0.25, 0.5, 0.75]
    polylist = []  # closed ring of polygon points

    # Determine min/max x/y values for the polygon
    x = Bounds([each[0] for each in polygon])
    y = Bounds([each[1] for each in polygon])

    polylist.append((x.min, y.min))
    dx = x.max - x.min
    dy = y.min - y.min
    for frac in fracs:
        polylist.append((x.min + frac * dx, y.min + frac * dy))

    polylist.append((x.max, y.min))
    dx = x.max - x.max
    dy = y.max - y.min
    for frac in fracs:
        polylist.append((x.max + frac * dx, y.min + frac * dy))

    polylist.append((x.max, y.max))
    dx = x.min - x.max
    dy = y.max - y.max
    for frac in fracs:
        polylist.append((x.max + frac * dx, y.max + frac * dy))

    polylist.append((x.min, y.max))
    dx = x.min - x.min
    dy = y.min - y.max
    for frac in fracs:
        polylist.append((x.min + frac * dx, y.max + frac * dy))

    polylist.append((x.min, y.min))

    return polylist
