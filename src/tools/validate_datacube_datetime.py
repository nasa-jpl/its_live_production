#!/usr/bin/env python
"""
Validate datetime objects in ITS_LIVE datacubes as stored in its-live-data S3 bucket.

Authors: Masha Liukis
"""
import argparse
import dask
import datetime
from dask.diagnostics import ProgressBar
import logging
import numpy as np
import os
import s3fs
import xarray as xr

from itscube import ITSCube
from itscube_types import DataVars


class ValidateDatacubes:
    """
    Class to validate ITS_LIVE datacubes residing in S3 bucket.
    Due to the use of Zarr version 2.8.3, some datacubes ended up with wrong
    datetime objects when written to the Zarr store.
    Need to validate that datetimes stored in the datacubes are as expected and
    identify datacubes that need to be re-generated with Zarr version 2.6.1.
    """
    SUFFIX_TO_REMOVE = '_IL_ASF_OD'

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

        logging.info(f"Number of cubes to validate: {len(self.all_datacubes)}")


    def __call__(self, chunk_size: int, num_dask_workers: int, start_index: int=0):
        """
        Validate each cube in S3 bucket.
        """
        num_to_fix = len(self.all_datacubes) - start_index

        start = start_index
        logging.info(f"{num_to_fix} cubes to validate starting with {start_index}...")

        if num_to_fix <= 0:
            logging.info(f"No datacubes, exiting.")
            return

        while num_to_fix > 0:
            num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

            logging.info(f"Starting tasks {start}:{start+num_tasks}")
            tasks = [dask.delayed(ValidateDatacubes.run)(each, self.s3) for each in self.all_datacubes[start:start+num_tasks]]
            results = None

            with ProgressBar():
                # Display progress bar
                results = dask.compute(tasks,
                                       scheduler="processes",
                                       num_workers=num_dask_workers)

            for each_result in results[0]:
                logging.info("\n--->".join(each_result))

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
            with xr.open_dataset(cube_store, decode_timedelta=False, engine='zarr', consolidated=True) as ds:
                # Make sure mid_date and date_center agree at date() level
                mid_dates_str = [np.datetime_as_string(t, unit='m') for t in ds.mid_date.values]
                date_center_str = [np.datetime_as_string(t, unit='m') for t in ds.date_center.values]

                if mid_dates_str != date_center_str:
                    msgs.append(f"ERROR: mismatching mid_date and date_center for {cube_url}: ")
                    # Show which values mis-match
                    for each_mid_date, each_date_center in zip(mid_dates_str, date_center_str):
                        if each_mid_date != each_date_center:
                            msgs.append(f"mid_date {each_mid_date} vs. date_center {each_date_center}")

                else:
                    msgs.append(f"Equal mid_date and date_center for {cube_url}, validate per each of {len(mid_dates_str)} layers: ")

                    granule_urls = ds.granule_url.values

                    # Validate each layer's datetime against the one as stored in the datacube
                    date_center = [t.astype('M8[ms]').astype('O') for t in ds.date_center.values]
                    acq_date_img1 = [t.astype('M8[ms]').astype('O') for t in ds.acquisition_date_img1.values]
                    acq_date_img2 = [t.astype('M8[ms]').astype('O') for t in ds.acquisition_date_img2.values]

                    for index, each_url in enumerate(granule_urls):
                        # Read each granule in and compare to the value in datacube
                        s3_path = each_url.replace(ITSCube.HTTP_PREFIX, ITSCube.S3_PREFIX)
                        s3_path = s3_path.replace(ITSCube.PATH_URL, '')

                        file_list = s3_in.glob(s3_path)
                        if len(file_list) == 0:
                            # Granule was renamed already, use new name:
                            s3_path = s3_path.replace('_IL_ASF_OD', '')

                        # msgs.append(f"Opening {s3_path}...")
                        with s3_in.open(s3_path, mode='rb') as fhandle:
                            with xr.open_dataset(fhandle, engine=ITSCube.NC_ENGINE) as granule_ds:
                                granule_date_center = ITSCube.get_data_var_attr(
                                        granule_ds,
                                        each_url,
                                        DataVars.ImgPairInfo.NAME,
                                        DataVars.ImgPairInfo.DATE_CENTER,
                                        to_date=True
                                    )

                                granule_acq_date_img1 = ITSCube.get_data_var_attr(
                                        granule_ds,
                                        each_url,
                                        DataVars.ImgPairInfo.NAME,
                                        DataVars.ImgPairInfo.ACQUISITION_DATE_IMG1,
                                        to_date=True
                                    )

                                granule_acq_date_img2 = ITSCube.get_data_var_attr(
                                        granule_ds,
                                        each_url,
                                        DataVars.ImgPairInfo.NAME,
                                        DataVars.ImgPairInfo.ACQUISITION_DATE_IMG2,
                                        to_date=True
                                    )

                                # Compare at date() and hour level as there's a discrepancy of a second when np.datetime64 is converted to datetime
                                if date_center[index].date() != granule_date_center.date() or date_center[index].hour != granule_date_center.hour:
                                    msgs.append(f"ERROR: date_center[{index}]: cube's {date_center[index]} vs. {granule_date_center}")

                                if acq_date_img1[index].date() != granule_acq_date_img1.date() or acq_date_img1[index].hour != granule_acq_date_img1.hour:
                                    msgs.append(f"ERROR: acq_date_img1[{index}]: cube's {acq_date_img1[index]} vs. {granule_acq_date_img1}")

                                if acq_date_img2[index].date() != granule_acq_date_img2.date() or acq_date_img2[index].hour != granule_acq_date_img2.hour:
                                    msgs.append(f"ERROR: acq_date_img2[{index}]: cube's {acq_date_img2[index]} vs. {granule_acq_date_img2}")

        except OverflowError as exc:
            msgs.append(f"EXCEPTION: processing {cube_url}: {exc}")

        except Exception as exc:
            msgs.append("UNEXPECTED_EXCEPTION: processing {cube_url}: {exc}")

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
    parser.add_argument(
        '-s', '--start-index',
        type=int,
        default=0,
        help='Index for the start datacube to process (if previous processing terminated) [%(default)d]'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Args: {args}")

    validate_datacubes = ValidateDatacubes(args.bucket)
    validate_datacubes(args.chunk_size, args.dask_workers, args.start_index)


if __name__ == '__main__':
    main()

    logging.info("Done.")
