"""
Move .nc files and their associated meta files (.nc.premet, .nc.spatial,
.stac.json) from the incomplete parquet report to a new S3 location.

Based on check_meta_files.py but performs moves instead of just checking.
"""

import argparse
import logging
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from joblib import Parallel, delayed
from tqdm import tqdm


# Meta file extensions to move (in addition to .nc file)
META_EXTENSIONS = [
    '.nc.premet',
    '.nc.spatial',
    '.stac.json',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Move .nc files and their meta files (.nc.premet, '
            '.nc.spatial, .stac.json) to a new S3 location'
        )
    )
    parser.add_argument(
        'incomplete_file',
        type=str,
        help='Path to the incomplete parquet file that lists .nc files and their '
        'missing associated meta files (nc files that miss png files should be '
        'processed)',
    )
    parser.add_argument(
        'manifest_file',
        type=str,
        help='Path to the manifest parquet file that lists .nc files and their associated meta files'
    )
    parser.add_argument(
        '--mission',
        type=str,
        default='sentinel1',
        help='Mission to process [%(default)s]',
    )
    parser.add_argument(
        '--source-bucket',
        type=str,
        default='its-live-data',
        help='Source S3 bucket name [%(default)s]',
    )
    parser.add_argument(
        '--destination-bucket',
        type=str,
        help=(
            'Destination S3 bucket. If not specified, '
            'files are moved within the same bucket'
        ),
    )
    parser.add_argument(
        '--destination-prefix',
        type=str,
        default='test-space/velocity_image_pair/sentinel1/pre11012025/manifest-2026-06-10',
        help=('Destination S3 prefix [%(default)s]'),
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='velocity_image_pair/sentinel1/v02',
        help='Filter .nc files by this prefix [%(default)s]',
    )
    parser.add_argument(
        '--split-on',
        type=str,
        default='sentinel1/',
        help=(
            'Split source keys on this string to get relative path '
            '[%(default)s]'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=100,
        help='Number of parallel workers [%(default)s]',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print operations without executing them',
    )
    parser.add_argument(
        '--copy-only',
        action='store_true',
        help='Copy files without deleting the originals',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of .nc files to process (for testing)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='move_meta_processed_results.parquet',
        help='Output file to save processed file results to. [%(default)s]',
    )

    return parser.parse_args()


_SESSION = boto3.session.Session(
    region_name='us-west-2',
)

def file_exists_in_s3(bucket, key):
    """
    Check if a file exists in S3.
    Returns True if exists, False otherwise.
    """
    s3_client = _SESSION.client('s3')
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def move_file(
    source_bucket, source_key, dest_bucket, dest_key,
    copy_only=False, dry_run=False
):
    """
    Move or copy a single file in S3.
    Returns dict with success status and details.
    """
    if dry_run:
        action = 'COPY' if copy_only else 'MOVE'
        return {
            'success': True,
            'source_key': source_key,
            'dest_key': dest_key,
            'action': action,
            'dry_run': True
        }

    # Create S3 client for this thread
    s3_client = _SESSION.client('s3')

    try:
        # Copy the file
        copy_source = {'Bucket': source_bucket, 'Key': source_key}
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key
        )

        # Delete original if moving (not just copying)
        if not copy_only:
            s3_client.delete_object(
                Bucket=source_bucket,
                Key=source_key
            )

        return {
            'success': True,
            'source_key': source_key,
            'dest_key': dest_key
        }

    except ClientError as e:
        return {
            'success': False,
            'source_key': source_key,
            'dest_key': dest_key,
            'error': str(e)
        }


def process_nc_and_meta_files(
    source_bucket, nc_key, dest_bucket, dest_prefix, split_on, existing_keys,
    copy_only=False, dry_run=False
):
    """
    Move .nc file and all its associated meta files.
    Returns dict with overall status and individual file results.
    """
    # Split on specified string to get relative path
    key_parts = nc_key.split(split_on, 1)

    if len(key_parts) != 2:
        return {
            'success': False,
            'nc_key': nc_key,
            'error': f'Missing "{split_on}" in key',
            'files_moved': 0
        }

    relative_path = key_parts[1]
    dest_prefix_clean = dest_prefix.rstrip('/')

    # Get base key (remove .nc extension)
    base_key = nc_key[:-3]
    base_relative_path = relative_path[:-3]

    # List of all files to move: .nc + meta files
    files_to_move = [
        (nc_key, f'{dest_prefix_clean}/{relative_path}')
    ]

    # Add meta files
    for ext in META_EXTENSIONS:
        source_key = base_key + ext

        # Check if metadata file exists in the existing keys set
        if source_key in existing_keys:
            dest_key = f'{dest_prefix_clean}/{base_relative_path}{ext}'
            files_to_move.append((source_key, dest_key))

    # Check which files already exist at destination and filter them out
    files_to_actually_move = []
    files_skipped = 0

    for source_key, dest_key in files_to_move:
        if not dry_run and file_exists_in_s3(dest_bucket, dest_key):
            files_skipped += 1
        else:
            files_to_actually_move.append((source_key, dest_key))

    # If all files already exist, return early
    if files_skipped == len(files_to_move):
        return {
            'success': True,
            'nc_key': nc_key,
            'files_moved': 0,
            'files_skipped': files_skipped,
            'total_files': len(files_to_move),
            'all_skipped': True
        }

    # Move only files that don't already exist
    results = []
    for source_key, dest_key in files_to_actually_move:
        result = move_file(
            source_bucket, source_key, dest_bucket, dest_key,
            copy_only=copy_only, dry_run=dry_run
        )
        results.append(result)

    # Check if all succeeded
    all_success = all(r['success'] for r in results)
    files_moved = sum(1 for r in results if r['success'])

    return {
        'success': all_success,
        'nc_key': nc_key,
        'files_moved': files_moved,
        'files_skipped': files_skipped,
        'total_files': len(files_to_move),
        'all_skipped': False,
        'file_results': results
    }


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

    logger.info(f'Listing all keys under prefix: {args.prefix}')

    # Read the manifest file (only needed columns)
    logger.info(f'Reading manifest file: {args.manifest_file}')
    df = pd.read_parquet(
        args.manifest_file,
        columns=['key', 'mission']
    )

    logger.info(f'Total rows in manifest: {len(df)}')

    # Use categorical dtype for mission to reduce memory usage
    df['mission'] = df['mission'].astype('category')
    is_target_mission = df['mission'] == args.mission

    logger.info(
        f'Keeping only {is_target_mission.sum()} {args.mission} '
        f'files'
    )
    df = df[is_target_mission]
    logger.info(f'Rows after filtering: {len(df)}')
    existing_keys = set(df['key'].astype(str))

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

    # Apply limit if specified
    if args.limit:
        df = df.head(args.limit)
        logger.info(f'Limited to {args.limit} files for testing')

    if len(df) == 0:
        logger.warning('No files to process after filtering')
        return

    # Get list of .nc keys to process
    nc_keys = df['nc_key'].tolist()

    dest_bucket = args.destination_bucket or args.source_bucket

    logger.info(f'\n=== Configuration ===')
    logger.info(f'Source bucket: {args.source_bucket}')
    logger.info(f'Destination bucket: {dest_bucket}')
    logger.info(f'Destination prefix: {args.destination_prefix}')
    logger.info(f'Split on: {args.split_on}')
    logger.info(f'Parallel workers: {args.workers}')
    logger.info(f'Files to process: {len(nc_keys)} .nc files')
    logger.info(
        f'Expected total files: {len(nc_keys) * (1 + len(META_EXTENSIONS))}'
    )

    if args.dry_run:
        logger.info('\n*** DRY RUN MODE - No files will be modified ***\n')

    action = 'Copying' if args.copy_only else 'Moving'
    logger.info(f'\n{action} files...\n')

    # Process files in parallel using joblib
    results = Parallel(n_jobs=args.workers, backend='threading')(
        delayed(process_nc_and_meta_files)(
            args.source_bucket,
            nc_key,
            dest_bucket,
            args.destination_prefix,
            args.split_on,
            existing_keys,
            copy_only=args.copy_only,
            dry_run=args.dry_run
        )
        for nc_key in tqdm(nc_keys, desc='Processing .nc files')
    )

    # Convert results to DataFrame
    logger.info('Creating results DataFrame')
    results_df = pd.DataFrame(results)

    # Save results
    logger.info(f'Saving results to: {args.output}')
    results_df.to_parquet(args.output)

    # Analyze results
    success_count = sum(1 for r in results if r['success'])
    failure_count = sum(1 for r in results if not r['success'])
    all_skipped_count = sum(1 for r in results if r.get('all_skipped', False))
    total_files_moved = sum(r['files_moved'] for r in results)
    total_files_skipped = sum(r.get('files_skipped', 0) for r in results)

    # Print failures
    failures = [r for r in results if not r['success']]
    if failures:
        logger.info('\nFailed .nc file groups:')
        for f in failures[:10]:  # Show first 10 failures
            logger.info(
                f"  {f['nc_key']}: {f.get('error', 'Unknown')} "
                f"({f['files_moved']}/{f['total_files']} files moved)"
            )
        if len(failures) > 10:
            logger.info(f'  ... and {len(failures) - 10} more')

    # Print summary
    logger.info('\n=== Summary ===')
    logger.info(f'Total .nc file groups: {len(results)}')
    logger.info(f'Successful groups: {success_count}')
    logger.info(f'Groups with all files skipped: {all_skipped_count}')
    logger.info(f'Failed groups: {failure_count}')
    logger.info(f'Total individual files moved: {total_files_moved}')
    logger.info(f'Total individual files skipped: {total_files_skipped}')
    logger.info(
        f'Expected total files: '
        f'{len(nc_keys) * (1 + len(META_EXTENSIONS))}'
    )


if __name__ == '__main__':
    main()
