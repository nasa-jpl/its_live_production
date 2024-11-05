"""Script to build rtree of all V2 ITS_LIVE granules.

This is to bypass searchAPI when building/updating datacubes to query existing
collection for the existing granules that overlap with any given datacube.
"""
import argparse
import json
import logging
from rtree import index
import s3fs


# AWS S3 locations of catalog geojsons that need to be parsed for
# to create R-tree of granules
S3_GRANULES_DIRS = [
    'its-live-data/catalog_geojson/landsatOLI/v02/',
    'its-live-data/catalog_geojson/sentinel1/v02/',
    # 'its-live-data/catalog_geojson/sentinel2/v02/'
]

S3_GRANULES_GLOB_PATTERN = 'imgpair*.json'


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

    args = parser.parse_args()

    # List to hold the extracted data
    data = []

    # Create an R-tree index
    idx = index.Index(args.rtree_file_path)

    # Index into R-tree
    i = 0

    s3_read = s3fs.S3FileSystem()

    for each_s3_dir in S3_GRANULES_DIRS:
        logging.info(f'Processing {each_s3_dir}...')

        # Iterate over all JSON files in the directory
        all_catalogs = s3_read.glob(f'{each_s3_dir}/{S3_GRANULES_GLOB_PATTERN}')
        logging.info(f'Got {len(all_catalogs)} catalogs...')

        for filename in all_catalogs:
            # if filename.startswith('imgpair') and filename.endswith('.json'):
            #     file_path = os.path.join(json_dir, filename)
            logging.info(f'Reading {filename}...')

            with s3_read.open(filename, 'r') as fhandle:
                json_data = json.load(fhandle)

                for each_granule in json_data['features']:
                    # Extract required fields
                    coordinates = each_granule['geometry']['coordinates'][0]

                    aws_s3_path = each_granule['properties']['directory'] + each_granule['properties']['filename']

                    # Compute longitude and latitude extends for each granule based on coordinates
                    # given in [longitude, latitude] order of coordinates list of lists, use it as R-tree index
                    longitudes = [coord[0] for coord in coordinates]
                    latitudes = [coord[1] for coord in coordinates]

                    min_lon, max_lon = min(longitudes), max(longitudes)
                    min_lat, max_lat = min(latitudes), max(latitudes)

                    idx.insert(i, (min_lon, min_lat, max_lon, max_lat), obj=aws_s3_path)
                    # Does not work - accepts only (min_lon, min_lat, max_lon, max_lat) as index
                    # idx.insert(i, flat_coordinates, obj=aws_s3_path)
                    i += 1

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

    # Example of reading back the R-tree index and quering it for files that ove
    # lap with a given bounding box

    # Reading back r-tree
    # idx_back = index.Rtree(args.rtree_file_path)

    # # Search the tree
    # query_box = (82.007672, 42.016033, 82.788207, 42.006526, 83.568262, 41.991717, 84.347626, 41.971618, 85.126093, 41.946244, 85.138266, 42.133812, 85.150556, 42.321371, 85.162964, 42.508921, 85.17549, 42.696462, 84.387752, 42.722508, 83.599075, 42.743139, 82.809677, 42.758341, 82.019776, 42.768101, 82.016706, 42.580093, 82.013666, 42.39208, 82.010655, 42.20406, 82.007672, 42.016033)
    # box_lon = query_box[::2]
    # box_lat = query_box[1::2]

    # min_lon, max_lon = min(box_lon), max(box_lon)
    # min_lat, max_lat = min(box_lat), max(box_lat)
    # logging.info(f'Got bounding box: lon={(min_lon, max_lon)} lat={(min_lat, max_lat)}')

    # query_results = query_rtree(idx_back, (min_lon, min_lat, max_lon, max_lat))

    # logging.info(f'Got results: {len(query_results)}')
