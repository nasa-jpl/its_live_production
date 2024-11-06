"""Script to build rtree of all V2 ITS_LIVE granules.
Script to build rtree of all V2 ITS_LIVE granules.

This is to bypass searchAPI when building/updating datacubes to query existing
collection for the existing granules that overlap with any given datacube.
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import gc
import json
import logging
import os
from rtree import index
import s3fs
import timeit

from itslive_utils import query_rtree

# AWS S3 locations of catalog geojsons that need to be parsed for
# to create R-tree of granules
S3_GRANULES_DIRS = [
    'its-live-data/catalog_geojson/landsatOLI/v02/',
    'its-live-data/catalog_geojson/sentinel1/v02/',
    # 'its-live-data/catalog_geojson/sentinel2/v02/'
]

# Glob pattern to match the granule JSON files in the S3 directories.
S3_GRANULES_GLOB_PATTERN = 'imgpair*.json'

# Number of catalogs to process concurrently.
NUM_CATALOGS_TO_PROCESS = 16

# AWS S3 URL prefix for accessing granule files.
AWS_PREFIX = 'https://its-live-data.s3.amazonaws.com'


def read_catalog(filename: str, s3_read: s3fs.S3FileSystem):
    """Read catalog geojson file and extract granules to build R-tree.

    Args:
        filename (str): Catalog geojson file name
        s3_read (s3fs.S3FileSystem): s3fs object to read the file
    """
    data = []
    with s3_read.open(filename, 'r') as fhandle:
        json_data = json.load(fhandle)

        for each_granule in json_data['features']:
            # Extract required fields
            coordinates = each_granule['geometry']['coordinates'][0]

            aws_s3_path = os.path.join(
                AWS_PREFIX,
                each_granule['properties']['directory'],
                each_granule['properties']['filename']
            )

            # Compute longitude and latitude extends for each granule based on coordinates
            # given in [longitude, latitude] order of coordinates list of lists, use it as R-tree index
            longitudes = [coord[0] for coord in coordinates]
            latitudes = [coord[1] for coord in coordinates]

            min_lon, max_lon = min(longitudes), max(longitudes)
            min_lat, max_lat = min(latitudes), max(latitudes)

            percent_valid_pix = each_granule['properties']['percent_valid_pix']
            data.append([(min_lon, min_lat, max_lon, max_lat), (aws_s3_path, percent_valid_pix)])

    return data

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    # Add command-line argument parser
    parser = argparse.ArgumentParser(
        description='Build an R-tree from all ITS_LIVE v2 image velocity pairs available for processing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-rtree_file_path',
        type=str,
        default='its_live_rtree',
        help='Path to the output rtree file'
    )
    parser.add_argument(
        '-num_threads',
        type=int,
        default=16,
        help='Number of threads to use for processing [%(default)d]'
    )
    parser.add_argument(
        '--run_example',
        action='store_true',
        help='Run the example code to query the R-tree after building it'
    )

    args = parser.parse_args()
    NUM_CATALOGS_TO_PROCESS = args.num_threads

    # List to hold the extracted data
    data = []

    # Create an R-tree index
    idx = index.Index(args.rtree_file_path)

    # Index into R-tree
    i = 0

    s3_read = s3fs.S3FileSystem()

    # Capture how long it takes to crawl all the directories
    crawl_start_time = timeit.default_timer()

    for each_s3_dir in S3_GRANULES_DIRS:
        logging.info(f'Processing {each_s3_dir}...')

        # Iterate over all JSON files in the directory
        all_catalogs = s3_read.glob(f'{each_s3_dir}/{S3_GRANULES_GLOB_PATTERN}')
        logging.info(f'Got {len(all_catalogs)} catalogs...')

        start = 0
        num_to_process = len(all_catalogs)

        while num_to_process > 0:
            # How many tasks to process at a time
            num_tasks = NUM_CATALOGS_TO_PROCESS if num_to_process > NUM_CATALOGS_TO_PROCESS else num_to_process
            tasks = [dask.delayed(read_catalog)(each_file, s3_read) for each_file in all_catalogs[start:start+num_tasks]]
            logging.info(f"Processing {len(tasks)} tasks out of {num_to_process} remaining")

            results = None
            with ProgressBar():
                results = dask.compute(
                    tasks,
                    scheduler="processes",
                    num_workers=NUM_CATALOGS_TO_PROCESS
                )

            del tasks
            gc.collect()

            for each_result in results[0]:
                for each_granule in each_result:
                    bounding_box, objects = each_granule
                    # logging.info(f'Got bounding box: {bounding_box} with objects: {objects}')
                    idx.insert(i, bounding_box, obj=objects)
                    i += 1

            num_to_process -= num_tasks
            start += num_tasks

        logging.info(f'Collected {i} granules after crawling {each_s3_dir}')

    # Save the R-tree index to a file:
    # creates two files with .dat and .idx extensions because it uses the R-tree spatial index structure,
    # which requires two separate files to store the index data.
    # .idx file: This file contains the index nodes of the R-tree. It stores the structure of the tree,
    #   including the bounding boxes and pointers to the child nodes or data entries.
    # .dat file: This file contains the actual data entries (e.g., the bounding boxes and associated data)
    # that are indexed by the R-tree.
    # The separation of the index structure and the data entries into two files allows for efficient querying
    # and updating of the spatial index. The .idx file is used to quickly navigate the tree structure,
    # while the .dat file is used to access the actual data entries.
    idx.close()
    logging.info(f'Wrote R-tree index to {args.rtree_file_path}')

    time_delta = timeit.default_timer() - crawl_start_time
    logging.info(f'Took {time_delta} seconds to crawl all directories')

    if args.run_example:
        logging.info('Running example code to query the R-tree...')
        # Example of reading back the R-tree index and quering it for files that
        # overlap with a given bounding box

        # Reading back r-tree
        idx_back = index.Rtree(args.rtree_file_path)

        start_time = timeit.default_timer()
        # Search the tree
        query_box = (82.007672, 42.016033, 82.788207, 42.006526, 83.568262, 41.991717, 84.347626, 41.971618, 85.126093, 41.946244, 85.138266, 42.133812, 85.150556, 42.321371, 85.162964, 42.508921, 85.17549, 42.696462, 84.387752, 42.722508, 83.599075, 42.743139, 82.809677, 42.758341, 82.019776, 42.768101, 82.016706, 42.580093, 82.013666, 42.39208, 82.010655, 42.20406, 82.007672, 42.016033)
        box_lon = query_box[::2]
        box_lat = query_box[1::2]

        min_lon, max_lon = min(box_lon), max(box_lon)
        min_lat, max_lat = min(box_lat), max(box_lat)
        logging.info(f'Got bounding box: lon={(min_lon, max_lon)} lat={(min_lat, max_lat)}')

        query_results = query_rtree(idx_back, (min_lon, min_lat, max_lon, max_lat))

        time_delta = timeit.default_timer() - start_time
        logging.info(f'Got {len(query_results)} granules for the bounding box (took {time_delta} seconds)')

        logging.info(f'First granule from the query: {query_results[0]}')

