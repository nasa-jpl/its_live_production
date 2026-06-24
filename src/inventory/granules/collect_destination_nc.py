"""
Collect .nc files and their associated meta files from a destination S3 prefix.

This script lists all .nc files at the destination S3 location and saves them
to a parquet file for analysis or comparison with source manifests.
"""

import argparse
import logging
import threading
import pandas as pd
import boto3
from tqdm import tqdm


# Meta file extensions to check (in addition to .nc file)
META_EXTENSIONS = [
    '.nc.premet',
    '.nc.spatial',
    '.stac.json',
    '_thumb.png',
    '.png',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Collect .nc files and their meta files from a destination '
            'S3 prefix and save to parquet'
        )
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
        default='test-space/velocity_image_pair/sentinel1/pre11012025/manifest-2026-06-10',
        help='S3 prefix to scan for .nc files [%(default)s]',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file to save collected .nc files',
    )

    return parser.parse_args()


_SESSION = boto3.session.Session(
    region_name='us-west-2',
)

# Thread-local storage for S3 clients
_thread_local = threading.local()


def get_s3_client():
    """
    Get or create a thread-local S3 client with connection pooling.
    Each thread reuses its client across multiple calls.
    """
    if not hasattr(_thread_local, 's3_client'):
        _thread_local.s3_client = _SESSION.client(
            's3',
            config=boto3.session.Config(
                max_pool_connections=50
            )
        )
    return _thread_local.s3_client


def collect_keys_from_s3(bucket, prefix, logger):
    """
    List all keys under the specified S3 prefix using paginator.
    Returns a list of dicts with key and last_modified_date.
    """
    s3_client = get_s3_client()
    keys_data = []
    paginator = s3_client.get_paginator('list_objects_v2')

    logger.info(f'Scanning S3 bucket: {bucket}')
    logger.info(f'Prefix: {prefix}')

    page_count = 0
    for page in tqdm(
        paginator.paginate(Bucket=bucket, Prefix=prefix),
        desc='Scanning S3 pages'
    ):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys_data.append({
                    'key': obj['Key'],
                    'last_modified_date': obj['LastModified'],
                    'size': obj['Size']
                })
            page_count += 1
            if page_count % 100 == 0:
                logger.info(f'  Scanned {page_count} pages, {len(keys_data)} keys so far')

    logger.info(f'  Total pages scanned: {page_count}')
    logger.info(f'  Total keys found: {len(keys_data)}')

    return keys_data


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    # Suppress verbose AWS logging
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    args = parse_args()

    logger.info('\n=== Configuration ===')
    logger.info(f'Bucket: {args.bucket}')
    logger.info(f'Prefix: {args.prefix}')
    logger.info(f'Output: {args.output}\n')

    # Collect all keys from S3
    keys_data = collect_keys_from_s3(args.bucket, args.prefix, logger)

    if not keys_data:
        logger.warning('No keys found at the specified prefix')
        return

    # Create DataFrame
    logger.info('Creating DataFrame from collected keys')
    df = pd.DataFrame(keys_data)
    # Save all keys to parquet
    logger.info(f'\nSaving all keys to: {args.output}')
    df.to_parquet(args.output, index=False)

    # Parse metadata from keys
    # Filter for .nc files
    nc_files = df[df['key'].str.endswith('.nc')].copy()
    logger.info(f'Found {len(nc_files)} .nc files')

    # Save .nc files to separate parquet
    if len(nc_files) > 0:
        nc_output = args.output.replace('.parquet', '_nc_only.parquet')
        nc_files.to_parquet(nc_output, index=False)
        logger.info(f'Saved {len(nc_files)} .nc files to: {nc_output}')

    # Summary statistics
    logger.info('\n=== Summary ===')
    logger.info(f'Total keys collected: {len(df)}')
    logger.info(f'Total .nc files: {len(nc_files)}')


if __name__ == '__main__':
    main()
