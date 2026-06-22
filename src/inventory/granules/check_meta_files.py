"""
Check existence of meta files (.nc.premet, .nc.spatial, .stac.json)
for .nc files missing .png files in S3 and extract date_updated attribute.

This is to reconcile already moved png files, that were dated with
pre-cutoff date of November 1, 2025, to the temporary s3 location, with
original NetCDF granule and corresponding metadata files in original S3
location.

It was verified that S1 granules have both of the png files missing in the
input parquet file (see jupyter notebook):
    tokens = missing_png_s1_df['missing_extensions'].str.split(',')

    # Count exact occurrences of '.png' per row
    png_count = tokens.apply(lambda lst: lst.count('.png'))
    sum(png_count)
        >484041
"""

import argparse
import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from joblib import Parallel, delayed
from tqdm import tqdm
import xarray as xr


# Meta file extensions to check
META_EXTENSIONS = [
    '.nc.premet',
    '.nc.spatial',
    '.stac.json',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Check existence of meta files (.nc.premet, .nc.spatial, '
            '.stac.json) for .nc files missing .png files in S3'
        )
    )
    parser.add_argument(
        'incomplete_file',
        type=str,
        help='Path to the incomplete parquet file',
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default='its-live-data',
        help='S3 bucket name [%(default)s]',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='velocity_image_pair/sentinel1/v02',
        help=(
            'Filter .nc files by this prefix [%(default)s]'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers for S3 checks [%(default)s]',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file to save meta file check results',
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        default='2025-11-01',
        help=(
            'Cutoff date in YYYY-MM-DD format to compare '
            'against date_updated [%(default)s]'
        ),
    )
    return parser.parse_args()


def check_s3_file_exists(s3_client, bucket, key):
    """Check if a file exists in S3 using head_object."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False

        # For other errors, log and return False
        logging.warning(f'Error checking {key}: {e}')
        return False


def get_date_updated_from_nc(bucket, nc_key):
    """
    Extract date_updated or date_created global attribute from .nc file.
    Returns the date value or None if not found/error.
    Tries date_updated first, then falls back to date_created.
    """
    try:
        s3_path = f's3://{bucket}/{nc_key}'
        with xr.open_dataset(s3_path, engine='h5netcdf') as ds:
            # Try date_updated first, then date_created
            date_updated = ds.attrs.get('date_updated', None)
            if date_updated is None:
                date_updated = ds.attrs.get('date_created', None)

            return date_updated

    except Exception as e:
        logging.warning(
            f'Could not read date_updated/date_created from '
            f'{nc_key}: {e}'
        )
        return None


def check_meta_files_for_nc(bucket, nc_key):
    """
    Check existence of meta files for a given .nc file.
    Returns dict with nc_key and existence status for each meta file.

    Creates its own S3 client for thread safety.
    """
    # Create S3 client for this thread
    s3_client = boto3.client('s3')

    # Remove .nc extension to get base key
    base_key = nc_key[:-3]

    result = {'nc_key': nc_key}

    # TODO: Extract date_updated from .nc file (disabled for performance)
    # result['date_updated'] = get_date_updated_from_nc(bucket, nc_key)
    result['date_updated'] = None

    for ext in META_EXTENSIONS:
        meta_key = base_key + ext
        exists = check_s3_file_exists(s3_client, bucket, meta_key)
        # Use column name like 'has_nc.premet'
        result[f'has_{ext[1:]}'] = exists

    return result


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    # Suppress verbose s3fs/fsspec logging
    logging.getLogger('s3fs').setLevel(logging.WARNING)
    logging.getLogger('fsspec').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    args = parse_args()

    # Read the incomplete parquet file
    logger.info(f'Reading incomplete file: {args.incomplete_file}')
    df = pd.read_parquet(args.incomplete_file)
    logger.info(f'Total rows in incomplete file: {len(df)}')

    # Filter for rows with .png in missing_extensions
    logger.info('Filtering for rows with .png in missing_extensions')
    df = df[df['missing_extensions'].str.contains('.png', regex=False)]
    logger.info(f'Rows with missing .png: {len(df)}')

    # Filter for specified prefix
    logger.info(f'Filtering for prefix: {args.prefix}')
    df = df[df['nc_key'].str.startswith(args.prefix)]
    logger.info(f'Rows after prefix filter: {len(df)}')

    if len(df) == 0:
        logger.warning('No files to process after filtering')
        return

    # Get list of .nc keys to check
    nc_keys = df['nc_key'].tolist()
    logger.info(f'Checking meta files for {len(nc_keys)} .nc files')

    # Process files in parallel using joblib
    logger.info(
        f'Starting parallel processing with {args.workers} workers'
    )
    results = Parallel(n_jobs=args.workers, backend='threading')(
        delayed(check_meta_files_for_nc)(args.bucket, nc_key)
        for nc_key in tqdm(nc_keys, desc='Checking S3 files')
    )

    # Convert results to DataFrame
    logger.info('Creating results DataFrame')
    results_df = pd.DataFrame(results)

    # Convert date_updated to datetime for comparison
    # (Currently disabled - all dates are None)
    logger.info('Processing date_updated and cutoff comparison')
    results_df['date_updated_dt'] = pd.to_datetime(
        results_df['date_updated'], errors='coerce'
    )
    cutoff_date = pd.Timestamp(args.cutoff_date)

    # Add pre-cutoff indicator (will be None/False since date extraction disabled)
    results_df['is_pre_cutoff'] = (
        results_df['date_updated_dt'] < cutoff_date
    )

    # Add summary columns
    meta_cols = [f'has_{ext[1:]}' for ext in META_EXTENSIONS]
    results_df['all_meta_exist'] = (
        results_df[meta_cols].all(axis=1)
    )
    results_df['missing_meta_count'] = (
        (~results_df[meta_cols]).sum(axis=1)
    )

    # Drop temporary datetime column before saving
    results_df.drop(columns=['date_updated_dt'], inplace=True)

    # Save results
    logger.info(f'Saving results to: {args.output}')
    results_df.to_parquet(args.output)

    # Print summary statistics
    logger.info('\n=== Meta File Check Summary ===')
    logger.info(f'Total .nc files checked: {len(results_df)}')
    logger.info(
        f'Files with all meta files present: '
        f'{results_df["all_meta_exist"].sum()}'
    )
    logger.info(
        f'Files missing at least one meta file: '
        f'{(~results_df["all_meta_exist"]).sum()}'
    )

    logger.info(
        f'\nPre-cutoff files (date_updated < {args.cutoff_date}): '
        f'{results_df["is_pre_cutoff"].sum()}'
    )
    logger.info(
        f'Post-cutoff or no date_updated: '
        f'{(~results_df["is_pre_cutoff"]).sum()}'
    )

    logger.info('\nMissing meta file breakdown:')
    for col in meta_cols:
        missing_count = (~results_df[col]).sum()
        logger.info(f'  Missing {col}: {missing_count}')

    logger.info('\nDistribution by missing meta file count:')
    logger.info(
        f'\n{results_df["missing_meta_count"].value_counts().sort_index()}'
    )

    # Show sample of files missing all meta files
    all_missing = results_df[results_df['missing_meta_count'] == 3]
    if len(all_missing) > 0:
        logger.info(
            f'\nSample of {min(5, len(all_missing))} files '
            f'missing all meta files:'
        )
        logger.info(f'\n{all_missing["nc_key"].head(5).to_string(index=False)}')


if __name__ == '__main__':
    main()
