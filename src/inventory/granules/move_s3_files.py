import argparse
from pathlib import Path

import boto3
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Move S3 files identified in parquet inventory to '
            'a new location'
        )
    )
    parser.add_argument(
        'parquet_file',
        type=str,
        help='Path to parquet file with files to move'
    )
    parser.add_argument(
        '--destination-prefix',
        type=str,
        default='test-space/velocity_image_pair/sentinel1/pre11012025/manifest-2026-06-10',
        help=(
            'Destination S3 prefix (e.g., archived/velocity_image_pair)'
        )
    )
    parser.add_argument(
        '--destination-bucket',
        type=str,
        help=(
            'Destination S3 bucket. If not specified, '
            'files are moved within the same bucket'
        )
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print operations without executing them'
    )
    parser.add_argument(
        '--copy-only',
        action='store_true',
        help='Copy files without deleting the originals'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=100,
        help='Number of parallel jobs to run (default: 100)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    return parser.parse_args()


_SESSION = boto3.session.Session(
    region_name='us-west-2',
)


def process_file(
    source_bucket, source_key, dest_bucket, dest_prefix,
    copy_only=False, dry_run=False
):
    """Process a single file (move or copy) in S3."""
    # Create a new S3 client for this thread
    s3_client = _SESSION.client('s3')

    # Split on 'sentinel1/' and keep everything after it
    key_parts = source_key.split('sentinel1/', 1)

    if len(key_parts) != 2:
        return {
            'success': False,
            'source_key': source_key,
            'error': 'Missing "sentinel1/" in key'
        }

    relative_path = key_parts[1]
    # Strip trailing slashes from prefix to avoid double slashes
    dest_prefix_clean = dest_prefix.rstrip('/')
    dest_key = f'{dest_prefix_clean}/{relative_path}'

    if dry_run:
        action = 'COPY' if copy_only else 'MOVE'
        return {
            'success': True,
            'source_key': source_key,
            'dest_key': dest_key,
            'action': action,
            'dry_run': True
        }

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

    except Exception as e:
        return {
            'success': False,
            'source_key': source_key,
            'dest_key': dest_key,
            'error': str(e)
        }


def main():
    args = parse_args()

    # Read the parquet file
    print(f'Reading parquet file: {args.parquet_file}')
    df = pd.read_parquet(args.parquet_file)

    # Limit files if requested (for testing)
    if args.limit:
        df = df.head(args.limit)
        print(f'Limited to {args.limit} files for testing')

    print(f'Found {len(df)} files to process')

    # Get source bucket from the DataFrame
    source_bucket = df['bucket'].iloc[0]
    dest_bucket = args.destination_bucket or source_bucket

    print(f'Source bucket: {source_bucket}')
    print(f'Destination bucket: {dest_bucket}')
    print(f'Destination prefix: {args.destination_prefix}')
    print(f'Parallel jobs: {args.n_jobs}')

    if args.dry_run:
        print('\n*** DRY RUN MODE - No files will be modified ***\n')

    action = 'Copying' if args.copy_only else 'Moving'
    print(f'\n{action} files...\n')

    # Process files in parallel
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(
            source_bucket,
            row['key'],
            dest_bucket,
            args.destination_prefix,
            copy_only=args.copy_only,
            dry_run=args.dry_run
        )
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # Count successes and failures
    success_count = sum(1 for r in results if r['success'])
    failure_count = sum(1 for r in results if not r['success'])

    # For testing purposes, print successes and failures
    # successes = [r for r in results if r['success']]
    # if successes:
    #     print('\nSucceeded files:')
    #     for s in successes:  # Show first 10 successes
    #         print(
    #             f"  {s['source_key']} -> {s['dest_key']}"
    #         )

    # Print failures
    failures = [r for r in results if not r['success']]
    if failures:
        print('\nFailed files:')
        for f in failures:  # Show first 10 failures
            print(
                f"  {f['source_key']}: {f.get('error', 'Unknown')}"
            )
        # if len(failures) > 10:
        #     print(f'  ... and {len(failures) - 10} more')

    # Print summary
    print(f'\n--- Summary ---')
    print(f'Total files: {len(df)}')
    print(f'Successful: {success_count}')
    print(f'Failed: {failure_count}')


if __name__ == '__main__':
    main()
