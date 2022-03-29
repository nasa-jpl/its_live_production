"""
Helper script to remove original datacubes from S3 bucket. S3 paths are provided
through input Json file.
"""

import subprocess
import os
import logging
import s3fs
import json


# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

def exists(cube_path: str):
    """
    Check if datacube exists. The datacube can be on a local file system or
    in AWS S3 bucket.
    """
    cube_exists = False

    # Check if the datacube is in the S3 bucket
    s3 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    cube_glob = s3.glob(cube_path)
    if len(cube_glob):
        cube_exists = True

    return cube_exists

def remove_s3_datacube(cube_s3_path: str, is_dryrun: bool):
    """
    Remove Zarr store and corresponding json file (with records of skipped
    granules for the cube) in S3 if they exists - this is done to replace existing
    cube with newly generated one.
    """
    # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
    # resulting in as many error messages as there are files in Zarr store
    # to copy
    env_copy = os.environ.copy()
    if exists(cube_s3_path):
        command_line = [
            "awsv2", "s3", "rm", "--recursive", "--quiet",
            cube_s3_path
        ]
        logging.info(f'Removing existing cube {cube_s3_path}: {" ".join(command_line)}')

        if not is_dryrun:
            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to remove original {cube_s3_path}: {command_return.stdout}")

            json_s3_path = cube_s3_path.replace('.zarr', '.json')

            command_line = [
                "awsv2", "s3", "rm", "--quiet",
                json_s3_path
            ]
            logging.info(f'Removing existing skipped granules json {json_s3_path}: {" ".join(command_line)}')

            command_return = subprocess.run(
                command_line,
                env=env_copy,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            if command_return.returncode != 0:
                raise RuntimeError(f"Failed to remove original {json_s3_path}: {command_return.stdout}")

    else:
        logging.info(f"Cube does not exist: {cube_s3_path}")

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
        help="Input Json file that stores a list of datacubes to remove [%(default)s]."
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not actually remove any datacubes'
    )
    args = parser.parse_args()

    with open(args.cubesFile) as fh:
        output_dirs = json.load(fh)

        for each in output_dirs:
            remove_s3_datacube(each, args.dryrun)

    logging.info("Done.")
