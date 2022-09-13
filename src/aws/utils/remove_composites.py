"""
Helper script to remove original Zarr composites stores from S3 bucket. S3 paths are provided
through input Json file.
The script also sorts datacubes into two queues to use AWS OnDemand and SPOT EC2
instances to generate new composites.
"""
import subprocess
import os
import logging
import s3fs
import json
import xarray as xr

from itscube_types import Coords

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

# Filenames to store datacubes per AWS queue type
SPOT_QUEUE_PREFIX = 'composites_spot_queue.json'
ONDEMAND_QUEUE_PREFIX = 'composites_ondemand_queue.json'

# Group datacubes by AWS Batch queue
QUEUE = {
    SPOT_QUEUE_PREFIX:     [],
    ONDEMAND_QUEUE_PREFIX: []
}

# Only for composites that need to be re-created: maximum "time" dimension
# to qualify re-processing for SPOT AWS queue,
# any number greater than that should be re-processed with On-Demand queue
MAX_SPOT_TIME_DIM = 140000

HTTP_PREFIX = 'http://its-live-data.s3.amazonaws.com'
S3_PREFIX = 's3://its-live-data'

CUBE_PATH = '/datacubes/'
COMPOSITE_PATH = '/composites/annual/'

ENV_COPY = os.environ.copy()


def exists(s3_path: str):
    """
    Check if Zarr store exists in AWS S3 bucket.
    """
    zarr_exists = False

    # Check if the datacube is in the S3 bucket
    s3 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    cube_glob = s3.glob(s3_path)
    if len(cube_glob):
        zarr_exists = True

    return zarr_exists

def remove_s3_composite(s3_path: str, is_dryrun: bool, s3_in):
    """
    Remove Zarr store from S3 if it exists - this is done to replace existing
    composites with newly generated one.
    Also identify which AWS Batch queue the cube should be processed with: OnDemand
    or SPOT.
    """
    # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
    # resulting in as many error messages as there are files in Zarr store
    # to copy
    is_dryrun_str = 'DRYRUN: ' if is_dryrun else ''

    zarr_store = s3fs.S3Map(root=s3_path, s3=s3_in, check=False)

    queue_name = SPOT_QUEUE_PREFIX

    with xr.open_dataset(zarr_store, decode_timedelta=False, engine='zarr', consolidated=True) as datacube_ds:
        # Check the size of the cube
        sizes = datacube_ds.sizes
        logging.info(f'{s3_path} dimensions: {sizes}')

        if sizes[Coords.MID_DATE] > MAX_SPOT_TIME_DIM:
            queue_name = ONDEMAND_QUEUE_PREFIX

        # Get S3 URL for corresponding composite
        composite_path, composite_filename = os.path.split(s3_path)
        composite_path = composite_path.replace(CUBE_PATH, COMPOSITE_PATH)
        name_tokens = composite_filename.split('_')

        # Cube name
        # s3://its-live-data/datacubes/v02/N30E060/ITS_LIVE_vel_EPSG32642_G0120_X350000_Y4250000.zarr
        # Composite name
        # s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_velocity_120m_X-3250000_Y250000.zarr
        new_tokens = ['ITS_LIVE_velocity_120m']
        new_tokens.extend(name_tokens[5:])
        composite_filename = '_'.join(new_tokens)

        composite_s3_path = os.path.join(composite_path, composite_filename)

        if exists(composite_s3_path):
            command_line = [
                "awsv2", "s3", "rm", "--recursive", "--quiet",
                composite_s3_path
            ]
            logging.info(f'{is_dryrun_str}Removing existing Zarr store {composite_s3_path}: {" ".join(command_line)}')

            if not is_dryrun:
                command_return = subprocess.run(
                    command_line,
                    env=ENV_COPY,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )
                if command_return.returncode != 0:
                    raise RuntimeError(f"Failed to remove original {composite_s3_path}: {command_return.stdout}")
        else:
            logging.info(f'WARNING: composite {composite_s3_path} does not exist, skip removal')

    return (queue_name, s3_path)

if __name__ == '__main__':
    import argparse

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-c', '--cubesFile',
        type=str,
        default=None,
        help="Input Json file that stores a list of datacubes to remove composites for [%(default)s]."
    )
    parser.add_argument(
        '-x', '--excludeCubesFile',
        type=str,
        default=None,
        help="Json file that stores a list of datacubes to exclude from processing [%(default)s]."
    )
    parser.add_argument(
        '-p', '--queueFilePrefix',
        type=str,
        default='Region',
        help="Filename prefix for AWS Batch queue files to store datacubes to [%(default)s]."
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not actually remove any datacubes'
    )

    args = parser.parse_args()

    exclude_list = []
    if args.excludeCubesFile:
        # Initialize a list of datacubes to exclude (that were already processed)
        with open(args.excludeCubesFile) as fh:
            exclude_list = json.load(fh)

            # Replace http paths (if any) with s3 paths
            exclude_list = [each.replace(HTTP_PREFIX, S3_PREFIX) for each in exclude_list]

    s3 = s3fs.S3FileSystem(anon=True)

    with open(args.cubesFile) as fh:
        output_dirs = json.load(fh)

        # Replace http paths (if any) with s3 paths
        output_dirs = [each.replace(HTTP_PREFIX, S3_PREFIX) for each in output_dirs]

        for each in output_dirs:
            if each in exclude_list:
                logging.info(f'Skipping excluded file: {each}')
                continue

            if not exists(each):
                logging.info(f"Datacube does not exist: {each}, skipping.")

            queue_name, cube_filename = remove_s3_composite(each, args.dryrun, s3)

            QUEUE[queue_name].append(cube_filename)

    # Save AWS Batch queues to JSON files
    output_file = args.queueFilePrefix + SPOT_QUEUE_PREFIX
    logging.info(f"Writing spot queue cubes to the {output_file}...")
    with open(output_file, 'w') as output_fhandle:
        json.dump(QUEUE[SPOT_QUEUE_PREFIX], output_fhandle, indent=4)

    output_file = args.queueFilePrefix + ONDEMAND_QUEUE_PREFIX
    logging.info(f"Writing ondemand queue cubes to the {output_file}...")
    with open(output_file, 'w') as output_fhandle:
        json.dump(QUEUE[ONDEMAND_QUEUE_PREFIX], output_fhandle, indent=4)

    logging.info("Done.")
