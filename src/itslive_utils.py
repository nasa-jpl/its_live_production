from asyncio import all_tasks

import boto3
from botocore.config import Config
import collections
import functools
import itertools
from joblib import Parallel, delayed, parallel_config
import json
import logging
import math
import numpy as np
import os
import pyproj
import random
import time
from tqdm import tqdm
from shapely.geometry import shape, box
import s3fs
import subprocess
import zarr

# Local imports
from grid import Bounds

import itslive
from itslive.search import EQ, GTE

# Number of 'aws s3 cp' retries in case of a failure
_NUM_AWS_COPY_RETRIES = 5

# Number of seconds to sleep between 'aws s3 cp' retries
_AWS_COPY_SLEEP_SECONDS = 60

# Metadata files that exist in the datacube root directory and each
# data variable sub-directory.
CUBE_META = ['.zattrs', '.zgroup', '.zmetadata']
VAR_META = ['.zarray', '.zattrs']

# Extension for the file that contains chunk information
# for each data variable in the datacube
# (ranges for each dimension and last chunk ranges)
# The file is created when the datacube is backed up
# and is used to restore the datacube from the backup.
# The file is created in the same directory as the datacube
# and is named <datacube_name>.chunks.json.
CHUNKS_FILE_EXTENSION = '.chunks.json'


def timing_decorator(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        msg = (
            f"Function {func.__name__}() executed in "
            f"{elapsed_time:.1f}s ({elapsed_time/60:.2f} min)"
        )
        logging.info(msg)

        return result
    return wrapper


def retry_decorator(
    max_retries=3,
    base_delay=1.0,
    backoff=2.0,
    jitter=True
):
    """
    Decorator to retry a function on any exception. This is most useful for
    functions that perform network calls, such as AWS S3 operations, which
    can fail due to transient issues.

    Args:
        max_retries (int): Number of retry attempts.
        base_delay (float): Initial delay between retries.
        backoff (float): Backoff multiplier between retries.
        jitter (bool): Whether to add random jitter to the delay.

    Usage:
        @retry(max_retries=3)
        def my_func(): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    sleep_time = random.uniform(0, delay) if jitter else delay

                    logging.info(
                        f"[Retry {attempt}] {func.__name__}(): {type(e).__name__}: {e} — "
                        f"retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                    delay *= backoff
        return wrapper
    return decorator


# Collection to record chunk information for each data variable in
# existing Zarr store:
# - ranges: list of ranges for each dimension
# - last_dim_ranges: tuple of chunk indices for the last dimension chunk
#   For example, for 3D chunks, it will be a tuple of ranges for 3 indices with the
#  first index being the last chunk index in the first dimension (mid_date for datacubes).
#  For 1D chunks, it will be one tuple to represent one index range.
ZarrChunk = collections.namedtuple(
    "ZarrChunk",
    ['ranges', 'last_dim_ranges']
)


def to_serializable(obj: dict) -> dict:
    """
    Serialize dictionary of ZarrChunk( variable chunk information
    within datacube) for JSON serialization.
    It converts tuples to lists  since JSON doesn't support tuples.

    Args:
        obj (dict): Dictionary of ZarrChunk objects.
    Returns:
        dict: Dictionary with serialized ZarrChunk objects.
    """
    output = {}
    for each_key, each_value in obj.items():
        output[each_key] = {
            'ranges': [[each[0], each[-1]] for each in each_value.ranges],
            'last_dim_ranges': [[each[0], each[-1]] for each in each_value.last_dim_ranges]
        }

    return output


def bucket_cube_name_from_url(source_url: str) -> str:
    """Extract bucket name and file URL from the given datacube URL.

    Args:
        source_url (str): AWS S3 URL of the datacube in Zarr format.

    Returns:
        str: Tuple of bucket name and file URL.
    """
    # Get rid of 's3://' prefix
    source_url = source_url.replace('s3://', '')

    # Split bucket name and file URL
    bucket_name, file_url = source_url.split('/', 1)
    logging.info(f'{bucket_name=} {file_url=}')

    return bucket_name, file_url


def download_chunk(bucket_name, s3_path, each_chunk, local_path):
    """Helper function to download Zarr chunk from S3.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_path (str): Path to the datacube or its variable in S3.
        each_chunk (str): Key of the chunk to download.
        local_path (str): Local path to save the downloaded chunk to.
    """
    # logging.info(f'Downloading {s3_key=} to {local_key}')

    # Initialize S3 bucket to copy files from
    s3_bucket = boto3.resource('s3').Bucket(bucket_name)

    s3_bucket.download_file(
        os.path.join(s3_path, each_chunk),
        os.path.join(local_path, each_chunk)
    )


@retry_decorator(max_retries=5)
def backup_chunk(bucket, source_path, filename, target_path):
    """Helper function to copy Zarr chunk stored in S3 bucket
    from one location to another. This is used to create a backup
    of the datacube in S3.

    Args:
        bucket (boto3.resources.factory.s3.Bucket): S3 bucket resource.
        source_path (str): Path to the datacube or its variable in S3.
        filename (str): Name of the file to copy.
        target_path (str): Target path to copy the chunk to.
    """
    copy_source = {
        'Bucket': bucket.name,
        'Key': os.path.join(source_path, filename)
    }
    bucket.copy(copy_source, os.path.join(target_path, filename))


@timing_decorator
def identify_datacube_latest_chunks(bucket_url: str):
    """
    Identify metadata files and latest chunks for each of the data
    variables for the given datacube s3 URL.

    Args:
        bucket_url (str): Name of the S3 bucket and full path to the
            datacube in Zarr format. Must start with 's3://'.

    Returns:
        Map of data variable to the ranges for existing data chunks,
        and the last chunk ranges for each data variable.
    """
    store = zarr.open_consolidated(
        store=bucket_url,
        mode='r'
    )

    # Identify last chunk for each data variable in the zarr store
    last_chunk_map = {}

    for var_name in store:
        var_obj = store[var_name]
        shape = np.array(var_obj.shape)
        chunk_shape = np.array(var_obj.chunks)

        # Compute total number of chunks along each axis: zero-based indexing
        num_chunks = (shape + chunk_shape - 1) // chunk_shape - 1
        logging.info(f'{var_name=} got {num_chunks=}')

        if len(num_chunks) == 0:
            # For variables with no chunking, just set last chunk to '0'
            # as it exists
            last_chunk_map[var_name] = ZarrChunk([range(0, 1)], [range(0, 1)])
            logging.info(
                f'No chunking for {var_name=}, setting last chunk to {last_chunk_map[var_name]}'
            )

        else:
            # Generate all prior indices per dimension
            dim_ranges = [range(0, idx + 1) for idx in num_chunks]

            # last_chunk_key = ".".join(map(str, num_chunks))
            logging.info(f'{var_name=} {dim_ranges=}')

            # Keep only last chunk for the mid_date dimension
            last_mid_date = num_chunks[0]
            last_chunks = [range(last_mid_date, last_mid_date+1)]

            # There are only 1D, 2D or 3D variables, nothing to do for 1D variables
            if len(num_chunks) == 2:
                # For 2D data variables, have to download all chunks since
                # it corresponds to the whole x/y spatial coverage
                last_chunks = dim_ranges

            elif len(num_chunks) == 3:
                # For 3D data variables, have to download all chunks that correspond
                # to the last chunk in first dimension (mid_date for datacubes)

                # Get list of all chunks that correspond to the last mid_date
                # dimension
                last_chunks.extend(
                    [
                        dim_ranges[1],
                        dim_ranges[2]
                    ]
                )

            logging.info(f'{var_name=}: {dim_ranges=} {last_chunks=}')

            last_chunk_map[var_name] = ZarrChunk(
                dim_ranges,
                last_chunks
            )

    return last_chunk_map


@timing_decorator
def backup_datacube_latest_chunks(
    bucket_url: str,
    backup_url: str,
    num_threads: int = 32
):
    """
    Create backup of metadata files and latest chunks for each of the data
    variables from the given datacube s3 URL.

    Args:
        bucket_url (str): s3 bucket path for the datacube to backup.
        backup_url (str): s3 bucket path for the datacube to backup latest
            Zarr chunks to.
        num_threads (int): Number of threads to use for parallel processing.
            Default is 32.

    Returns:
        Map of data variable to the ranges for existing data chunks,
        and the last chunk ranges for each data variable.
    """
    logging.info(f'Backing up {bucket_url} to {backup_url}...')

    # Identify last chunk for each data variable in the cube
    last_chunk_map = identify_datacube_latest_chunks(bucket_url)

    # Isolate bucket name and file path from the given S3 URLs
    bucket_name, source_url = bucket_cube_name_from_url(bucket_url)
    _, target_url = bucket_cube_name_from_url(backup_url)

    # Save identified chunks to the local file
    local_filename = os.path.basename(source_url) + CHUNKS_FILE_EXTENSION
    logging.info(f'Saving identified chunks to {local_filename}')
    with open(local_filename, 'w') as fhandle:
        json.dump(
            to_serializable(last_chunk_map),
            fhandle,
            indent=3
        )

    s3 = boto3.resource('s3', config=Config(max_pool_connections=num_threads))
    s3_bucket = s3.Bucket(bucket_name)

    # Upload chunk information file to the backup path
    logging.info(f'Backup {local_filename} to {target_url+CHUNKS_FILE_EXTENSION}')
    s3_bucket.upload_file(local_filename, target_url + CHUNKS_FILE_EXTENSION)

    # Backup metadata files
    for each_meta in CUBE_META:
        # Backup the file
        logging.info(
            f'Backup cube {each_meta} to {target_url}'
        )
        backup_chunk(s3_bucket, source_url, each_meta, target_url)

    # Build flat list of all (var, chunk) copy tasks across all variables,
    # and corresponding variable metadata files to copy.
    all_tasks = [
        (each_var, ".".join(map(str, each_chunk)))
        for each_var, each_chunk_info in last_chunk_map.items()
        for each_chunk in itertools.product(*each_chunk_info.last_dim_ranges)
    ] + [
        (each_var, each_meta)
        for each_var in last_chunk_map
        for each_meta in VAR_META
    ]

    logging.info(
        f"Backing up {len(all_tasks)} chunks across {len(last_chunk_map)} "
        "variables..."
    )

    with parallel_config(backend='threading', n_jobs=num_threads):
        Parallel()(
            delayed(backup_chunk)(
                s3_bucket,
                os.path.join(source_url, each_var),
                each_chunk_key,
                os.path.join(target_url, each_var),
            )
            for each_var, each_chunk_key in tqdm(all_tasks,
                                                    desc='Backing up chunks',
                                                    total=len(all_tasks)
                                                )
        )

    return last_chunk_map


@timing_decorator
def serverless_search(
    epsg_code: str,
    start_date: str,
    end_date: str,
    roi: dict,
    percent_valid_pixels: float = 1.0,
):
    """Get list of granules using itslive Python package.

    Returns:
        list(str): Found list of granule URLs.

    For example, this query should return all the granules that intersect
    with the provided polygon and have 100% valid pixels in EPSG:32717
    projection between 1982-01-01 and 2026-03-04:

    urls = itslive.velocity_pairs.find(
        engine="duckdb",
        geojson={
            "type": "Polygon",
            "coordinates": [[
                [-79.20094379386568, -2.7128288679416928],
                [-78.97615089345577, -2.7124718244728148],
                [-78.75138943360363, -2.7120728681102473],
                [-78.52666289912634, -2.711632029553526],
                [-78.301974772136,   -2.7111493427123956],
                [-78.30245463061306, -2.485223527607242],
                [-78.30289263885614, -2.259296852744831],
                [-78.3032888305625,  -2.0333693961602926],
                [-78.30364323620105, -1.8074412359243268],
                [-78.52819277641638, -1.8077629004547089],
                [-78.75278060638821, -1.8080566770534239],
                [-78.97740325463644, -1.8083225431391514],
                [-79.20205724699625, -1.8085604782683422],
                [-79.20182073376273, -2.0346286492152004],
                [-79.20155633443191, -2.260696153747212],
                [-79.20126402864005, -2.486762917947902],
                [-79.20094379386568, -2.7128288679416928],
            ]]
        },
        start="1982-01-01",
        end="2026-03-04",
        filters={
            "percent_valid_pixels": GTE(1.0),
            "proj:code": EQ("EPSG:32717"),
        }
    )
    """
    return itslive.velocity_pairs.find(
        engine="duckdb",
        geojson=roi,
        start=start_date,
        end=end_date,
        filters={
            "percent_valid_pixels": GTE(percent_valid_pixels),
            "proj:code": EQ(f"EPSG:{epsg_code}"),
        }
    )


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

    # if is_quiet and _quiet_flag not in command_line:
    #     command_line.append(_quiet_flag)

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
            logging.warning(
                f"Failed to invoke: {' '.join(command_line)} with "
                f"returncode={command_return.returncode}: "
                f"{command_return.stdout}"
            )

            num_retries += 1
            # If failed due to AWS SlowDown error, retry
            if num_retries < _NUM_AWS_COPY_RETRIES:
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
        raise RuntimeError(
            f"Failed to invoke {' '.join(command_line)} with "
            f"command.returncode={command_return.returncode}"
        )


def transform_coord(proj1, proj2, lon, lat):
    """Transform coordinates from proj1 to proj2 (EPSG num)."""
    # Set full EPSG projection strings
    proj1 = pyproj.Proj("+init=EPSG:"+proj1)
    proj2 = pyproj.Proj("+init=EPSG:"+proj2)
    # Convert coordinates
    return pyproj.transform(proj1, proj2, lon, lat)


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
