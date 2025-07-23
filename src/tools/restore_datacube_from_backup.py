#!/usr/bin/env python
"""
Restore datacube to the backup copy of it.
This tool is intended to restore updated datacube for which s3 copy to
original location failed due to the EC2 instance termination during AWS
Batch processing.

This tool will:
1. Read "updated" datacube from the S3 bucket
2. Read metadata from the backup copy of the datacube
3. Identify newly added chunks for all data variables in updated datacube
4. Remove newly added chunks from the updated datacube
5. Copy backup copy of the datacube to the original s3 location

Authors: Masha Liukis
"""
import aioboto3
import asyncio
import argparse
import itertools
import json
import logging
import os
from pathlib import Path

import itslive_utils
from itscube_types import FileExtension

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


async def delete_objects(bucket, s3_var_path, keys):
    if RestoreDatacubeFromBackup.DRY_RUN:
        chunks = [
            os.path.join(s3_var_path, ".".join(map(str, key))) for key in keys
        ]
        logging.info(f"Deleting {len(keys)} objects: {chunks[0]}...{chunks[-1]}")

    if not RestoreDatacubeFromBackup.DRY_RUN:
        await bucket.delete_objects(
            Delete={
                'Objects': [
                    {'Key': os.path.join(s3_var_path, ".".join(map(str, key)))} for key in keys
                ],
                'Quiet': True
            }
        )


async def delete_batch(bucket_name, s3_var_path, keys_to_delete):

    session = aioboto3.Session()
    async with session.resource("s3") as s3:
        bucket = await s3.Bucket(bucket_name)

        # Delete in batches of 1000 (API limit)
        for i in range(0, len(keys_to_delete), 1000):
            batch = keys_to_delete[i:i + 1000]
            await delete_objects(bucket, s3_var_path, batch)


class RestoreDatacubeFromBackup:
    """
    Restore datacube to the backup copy of it.
    """
    DRY_RUN = False

    def __init__(
        self,
        datacubes: list,
        bucket_dir: str,
        backup_dir: str
    ):
        """
        Initialize object.

        Args:
            datacubes (list): List of datacubes to process.
            bucket_dir (str): AWS S3 directory that store the datacubes to
                restore.
            backup_dir (str): AWS S3 directory to store the backup copy of
                the datacubes.
        """
        self.bucket_dir = bucket_dir
        self.backup_dir = backup_dir
        self.datacubes = datacubes

        # Collect names for existing datacubes
        logging.info(f"Got {len(datacubes)=} datacubes to process")

    def __call__(self):
        """
        Identify newly added chunks for all data variables and remove them
        from the updated datacube. Copy backup copy of the datacube to the
        original s3 location.
        """
        logging.info("Restoring cubes to their backup copy...")

        num_to_fix = len(self.datacubes)

        if num_to_fix <= 0:
            logging.info("Nothing to restore, exiting.")
            return

        for each_cube in self.datacubes:
            logging.info(f"Starting {each_cube}")
            msgs = RestoreDatacubeFromBackup.restore(
                each_cube,
                self.bucket_dir,
                self.backup_dir
            )
            logging.info("\n-->".join(msgs))

    @staticmethod
    def restore(
        cube_url: str,
        bucket_dir: str,
        backup_dir: str
    ):
        """
        Restore datacube to the backup copy of it.

        Args:
        cube_url (str): Original cube URL in S3 bucket to add new variables to.
        bucket_dir (str): AWS S3 directory that store the datacubes to
            restore.
        backup_dir (str): AWS S3 directory to store the backup copy of
            the datacubes.
        s3 (s3fs.S3FileSystem): s3fs FileSystem object to access datacubes and granules.
        """
        msgs = [f'Processing {cube_url}']

        bucket_name, source_url = itslive_utils.bucket_cube_name_from_url(cube_url)

        # Format backup cube URL based on the cube URL
        backup_url = cube_url.replace(bucket_dir, backup_dir)

        # Identify chunks for the datacube to restore
        msgs.append(f'Identifying chunks for {cube_url}...')
        cube_chunks = itslive_utils.identify_datacube_latest_chunks(
            cube_url
        )

        msgs.append(f'Identifying chunks for {backup_url}...')
        backup_chunks = itslive_utils.identify_datacube_latest_chunks(
            backup_url
        )

        # Identify chunks that are not in the backup copy of the datacube
        new_chunks = {}

        for each_var, each_info in cube_chunks.items():
            # Identify chunks that are not in the backup copy of the datacube
            # Look up the same variable chunking in backup copy
            # Consider only 1D or 3D variables as those are growing
            # in mid_date dimension
            c_chunks = each_info.ranges
            b_chunks = backup_chunks[each_var].ranges

            # Identify chunks that are not in the backup copy of the datacube
            if len(c_chunks) == 2:
                # Skip 2D variables as they are not growing in
                # mid_date dimension
                continue

            diff = sorted(set(c_chunks[0]) - set(b_chunks[0]))

            if len(diff) == 0:
                # No new chunks
                logging.info(f'No new chunks for {each_var}')
                continue

            else:
                new_chunks[each_var] = [range(diff[0], diff[-1] + 1)]

                # Add ranges for the rest of dimension(s)
                new_chunks[each_var].extend(c_chunks[1:])

                msgs.append(
                    f'Got {diff} new chunks for {each_var}: {new_chunks[each_var]}'
                )

        # Now remove new chunks from the datacube
        if len(new_chunks):
            # Backup latest chunks and metadata files for each data variable
            for each_var, each_chunk_info in new_chunks.items():
                s3_var_path = os.path.join(source_url, each_var)

                logging.info(f'Removing {each_var}...')

                # Step through Cartesian values of the last dimension ranges
                chunk_iterator = itertools.product(*each_chunk_info)

                for chunks in iter(
                    lambda: list(
                        itertools.islice(
                            chunk_iterator,
                            1000)
                        ),
                    []
                ):
                    asyncio.run(
                        delete_batch(bucket_name, s3_var_path, chunks)
                    )
                    logging.info(f'Deleted {len(chunks)} chunks for {each_var}')

        return msgs


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--inputCubes',
        type=str,
        action='store',
        default=None,
        help="JSON list of datacubes to process [%(default)s]."
    )
    group.add_argument(
        '--inputCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of datacube to process [%(default)s]."
    )
    parser.add_argument(
        '-d', '--bucketDir',
        type=str,
        default='/datacubes/v2-updated-october2024',
        help='AWS S3 directory that store the datacubes to restore [%(default)s]'
    )
    parser.add_argument(
        '-t', '--backupDir',
        type=str,
        default='/test-space/backup/v2_datacubes/20250522',
        help='AWS S3 directory with datacubes backups to restore from [%(default)s]'
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit AWS S3 push commands.'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")
    RestoreDatacubeFromBackup.DRY_RUN = args.dryrun

    cubes = []
    if args.inputCubes:
        cubes = json.loads(args.inputCubes)

    elif args.inputCubesFile:
        cubes = json.loads(args.inputCubesFile.read_text())

    if len(cubes) == 0:
        raise RuntimeError('No cubes to restore are provided.')

    restore_cubes = RestoreDatacubeFromBackup(
        cubes,
        args.bucketDir,
        args.backupDir
    )

    restore_cubes()

    env_copy = os.environ.copy()
    # Copy datacube backup to the original location in s3 bucket. This
    # is done as a last step to overwrite latest chunks at the time of the
    # datacube backup creation.
    for each_cube in cubes:
        backup_url = each_cube.replace(args.bucketDir, args.backupDir)

        logging.info(f"Copying {each_cube} from {backup_url}...")

        command_line = [
            "awsv2", "s3", "cp",
            backup_url,
            each_cube,
            "--recursive",
            "--acl", "bucket-owner-full-control"
        ]

        logging.info(f"Command line: {command_line}")

        if not RestoreDatacubeFromBackup.DRY_RUN:
            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)

        # Restore json file with skipped granules
        skipped_granules_file = each_cube.replace(
            FileExtension.ZARR, FileExtension.JSON
        )
        file_url = backup_url.replace(FileExtension.ZARR, FileExtension.JSON)

        logging.info(f"Copying {file_url} to {skipped_granules_file}")

        command_line = [
            "awsv2", "s3", "cp",
            file_url,
            skipped_granules_file,
            "--acl", "bucket-owner-full-control"
        ]

        logging.info(f"Command line: {command_line}")

        if not RestoreDatacubeFromBackup.DRY_RUN:
            itslive_utils.s3_copy_using_subprocess(command_line, env_copy)


if __name__ == '__main__':
    main()
    logging.info("Done.")
