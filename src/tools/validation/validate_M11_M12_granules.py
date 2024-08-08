import logging
import os
import dask
from dask.diagnostics import ProgressBar

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

import s3fs
import xarray as xr
import json
from tqdm import tqdm

NC_ENGINE = 'h5netcdf'

bucket = 's3://its-live-data'

validation_dir = 'velocity_image_pair/sentinel1-restoredM/validation/v02/N60W040'
restored_dir = 'velocity_image_pair/sentinel1-restoredM/v02/N60W040'

def collect_info(granule_path, s3):
    """Collect granule differences for original and newly reconstructed M11/M12 workflow"""

    # Open both granules and compute diff(M11) and diff(M12)
    with s3.open(granule_path, mode='rb') as each_fhandle:
            with xr.open_dataset(each_fhandle, engine=NC_ENGINE) as each_ds:
                each_ds = each_ds.load()

                # Open corresponding original granule
                each_orig_path = granule_path.replace('validation/', '')
                # print(f'--->Opening original {each_orig_path}')

                with s3.open(each_orig_path, mode='rb') as each_orig_fhandle:
                        with xr.open_dataset(each_orig_fhandle, engine=NC_ENGINE) as each_orig_ds:
                            each_orig_ds = each_orig_ds.load()

                            # compute M* diffs
                            each_M11 = each_ds.M11 - each_orig_ds.M11
                            each_M12 = each_ds.M12 - each_orig_ds.M12

                            _mean_M11 = each_M11.mean().item()
                            _mean_M12 = each_M12.mean().item()

                            _std_M11 = each_M11.std().item()
                            _std_M12 = each_M12.std().item()

                            _max_M11 = abs(each_M11.max().item())
                            _max_M12 = abs(each_M12.max().item())

                            return (_mean_M11, _mean_M12, _std_M11, _std_M12, _max_M11, _max_M12)



if __name__ == '__main__':
    s3 = s3fs.S3FileSystem(anon=True)

    print(f'Glob of {bucket}: {validation_dir}')
    val_granules = s3.glob(f'{os.path.join(bucket, validation_dir)}/*.nc')

    num_to_fix = len(val_granules)
    # num_to_fix = 10
    chunk_size = 16
    num_dask_workers = 8
    start = 0

    mean_M11 = []
    std_M11 = []
    max_M11 = []

    mean_M12 = []
    std_M12 = []
    max_M12 = []

    while num_to_fix > 0:
        num_tasks = chunk_size if num_to_fix > chunk_size else num_to_fix

        logging.info(f"Starting tasks {start}:{start+num_tasks}")
        tasks = [dask.delayed(collect_info)(each, s3) for each in val_granules[start:start+num_tasks]]
        results = None

        with ProgressBar():
            # Display progress bar
            results = dask.compute(
                tasks,
                scheduler="processes",
                num_workers=num_dask_workers
            )

        for each_result in results[0]:
            print(f'Got results: {each_result}')
            each_mean_M11, each_mean_M12, each_std_M11, each_std_M12, each_max_M11, each_max_M12 = each_result

            # Collect results
            mean_M11.append(each_mean_M11)
            mean_M12.append(each_mean_M12)
            std_M11.append(each_std_M11)
            std_M12.append(each_std_M12)
            max_M11.append(each_max_M11)
            max_M12.append(each_max_M12)

        num_to_fix -= num_tasks
        start += num_tasks


    # for each_path in tqdm(val_granules, desc='Reading validation granules'):
    #     # print(f'Opening {each_path}')

    #     # Open both granules and compute diff(M11) and diff(M12)
    #     with s3.open(each_path, mode='rb') as each_fhandle:
    #             with xr.open_dataset(each_fhandle, engine=NC_ENGINE) as each_ds:
    #                 each_ds = each_ds.load()

    #                 # Open corresponding original granule
    #                 each_orig_path = each_path.replace('validation/', '')
    #                 # print(f'--->Opening original {each_orig_path}')

    #                 with s3.open(each_orig_path, mode='rb') as each_orig_fhandle:
    #                         with xr.open_dataset(each_orig_fhandle, engine=NC_ENGINE) as each_orig_ds:
    #                             each_orig_ds = each_orig_ds.load()

    #                             # compute M* diffs
    #                             each_M11 = each_ds.M11 - each_orig_ds.M11
    #                             each_M12 = each_ds.M12 - each_orig_ds.M12

    #                             mean_M11.append(each_M11.mean().item())
    #                             mean_M12.append(each_M12.mean().item())

    #                             std_M11.append(each_M11.std().item())
    #                             std_M12.append(each_M12.std().item())

    #                             max_M11.append(abs(each_M11.max().item()))
    #                             max_M12.append(abs(each_M12.max().item()))


    # Store values to the files
    results = {
        "mean_M11": mean_M11,
        "std_M11": std_M11,
        "max_M11": max_M11,
        "mean_M12": mean_M12,
        "std_M12": std_M12,
        "max_M12": max_M12
    }

    output_file = 'validation_results.json'
    print(f'Writing results to the file {output_file}')
    with open(output_file, 'w') as fhandle:
        json.dump(results, fhandle, indent=4)

    print('Done.')