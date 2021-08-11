#!/usr/bin/env python
"""
Validate datetime objects in ITS_LIVE datacubes as stored in its-live-data S3 bucket.

Authors: Masha Liukis
"""
import argparse
import dask
from dask.diagnostics import ProgressBar
import logging
import os
import s3fs
import xarray as xr


class ValidateDatacubes:
    """
    Class to validate ITS_LIVE datacubes residing in S3 bucket.
    Due to the use of Zarr version 2.8.3, some datacubes ended up with wrong
    datetime objects when written to the Zarr store.
    Need to validate that datetimes stored in the datacubes are as expected and
    identify datacubes that need to be re-generated with Zarr version 2.6.1.
    """

    def __init__(self, bucket: str):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

        self.all_datacubes = []
        for each in self.s3.ls('its-live-data.jpl.nasa.gov/datacubes/v01/'):
            cubes = self.s3.ls(each)
            cubes = [each_cube for each_cube in cubes if each_cube.endswith('.zarr')]
            self.all_datacubes.extend(cubes)

        logging.info(f"Number of cubes: {len(self.all_datacubes)}")

        # Each granule has corresponding png files: '.png', '_thumb.png',
        # rename those too
        # self.all_granules = [each for each in self.all_granules if FixGranulesNames.SUFFIX_TO_REMOVE in each]
        # logging.info(f"Number of {FixGranulesNames.SUFFIX_TO_REMOVE} granules: {len(self.all_granules)}")

    def __call__(self, chunk_size: int, num_dask_workers: int):
        """
        Validate each cube in S3 bucket.
        """
        num_to_fix = len(self.all_datacubes)

        start = 0
        logging.info(f"{num_to_fix} cubes to validate...")

        if num_to_fix <= 0:
            logging.info(f"No datacubes, exiting.")
            return

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(ValidateDatacubes.run)(each) for each in self.all_datacubes[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("-->".join(each_result))

            num_to_fix -= num_tasks
            start += num_tasks

    @staticmethod
    def run(cube_url: str, s3_in):
        """
        Validate datacube
        """
        msgs = [f'Processing {cube_url}']

        try:
            #
            cube_store = s3fs.S3Map(root=cube_url, s3=s3_in, check=False)
            with xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True, chunks={'mid_date': 250}) as ds:
                mid_dates = [np.datetime_as_string(t, unit='s') for t in ds.mid_date.values]
                date_center = [np.datetime_as_string(t, unit='s') for t in ds.date_center.values]
                if mid_dates != date_center:
                    msgs.append(f"ERROR: mismatching mid_date and date_center for {cube_url}")
                else:
                    msgs.append(f"Equal mid_date and date_center for {cube_url}")

        except OverflowError as exc:
            msgs.append(f"EXCEPTION: processing {cube_url}: {exc}")

        return msgs

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--chunk_size', type=int,
        default=10, help='Number of datacubes to validate in parallel [%(default)d]'
    )
    parser.add_argument(
        '-b', '--bucket', type=str,
        default='s3://its-live-data.jpl.nasa.gov/datacubes/v01',
        help='AWS S3 that stores ITS_LIVE datacubes to validate'
    )
    parser.add_argument('-w', '--dask-workers', type=int,
        default=4,
        help='Number of Dask parallel workers [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    validate_datacubes = ValidateDatacubes(args.bucket)
    validate_datacubes(args.chunk_size, args.dask_workers)


if __name__ == '__main__':
    main()

    logging.info("Done.")
