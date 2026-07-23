#!/usr/bin/env python3
"""
Collect first granule matching pattern from each subdirectory in S3.

This script scans s3://its-live-data/velocity_image_pair/landsatOLI/v02/
and collects the first .nc file matching the specified pattern from each subdirectory.

Uses batched pagination to efficiently process directories with millions of files,
stopping as soon as the first matching file is found (or after checking 10,000 files
per subdirectory to avoid excessive runtime).

Key Features:
    - Configurable pattern matching via --pattern argument (e.g., P099, P055, P082)
    - Dynamic output filename: granules_{pattern}.json
    - Batched S3 pagination for efficiency (default: 1000 files per request)
    - Maximum 10,000 files checked per subdirectory before moving on
    - Proper logging with configurable verbosity

Arguments:
    --output OUTPUT_FILE    Write results to JSON file (default: granules_PXXX.json)
    --pattern PXXX         Pattern to match in filenames (e.g., P099, P055) (default: P099)
    --limit N              Limit to first N subdirectories (for testing)
    --batch-size SIZE      Number of files to fetch per S3 request (default: 1000)
    --verbose              Enable debug logging
    --base-path PATH       S3 base path to scan (default: s3://its-live-data/velocity_image_pair/landsatOLI/v02/)

Usage Examples:
    # Collect P099 granules (default pattern)
    python collect_p099_granules.py --verbose

    # Collect P055 granules
    python collect_p099_granules.py --pattern P055 --verbose

    # Collect P082 granules with custom output file
    python collect_p099_granules.py --pattern P082 --output my_granules.json --verbose

    # Test with first 5 subdirectories only
    python collect_p099_granules.py --pattern P055 --limit 5 --verbose

    # Use larger batches for faster processing (more memory)
    python collect_p099_granules.py --pattern P099 --batch-size 5000 --verbose

    # Collect LandsatOLI granules:
    python ./collect_p099_granules.py --verbose --output landsatOLI_P055_granules.json --pattern P055 |& tee landsatOLI_P055_granules.json.txt

    # Collect S1 granules (no granules with P055 exist, use P097):
    python ./collect_p099_granules.py --verbose --output s1_P097_granules.json --pattern P097 --base-path s3://its-live-data/velocity_image_pair/sentinel1/v02/ |& tee s1_P097_granules.json.txt

    # Collect S2 granules (no granules with P055 exist, use P097):
    python ./collect_p099_granules.py --verbose --output s2_P097_granules.json --pattern P097 --base-path s3://its-live-data/velocity_image_pair/sentinel2/v02/ |& tee s2_P097_granules.json.txt

Output Format:
    JSON array of S3 URLs:
    [
      "s3://its-live-data/velocity_image_pair/landsatOLI/v02/S80W170/file1_P099.nc",
      "s3://its-live-data/velocity_image_pair/landsatOLI/v02/N40W120/file2_P099.nc"
    ]
"""
import argparse
import json
import logging
import sys
from pathlib import Path
import boto3

# Add parent directory to path to import aws_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import aws_utils

# Set up logging
logger = logging.getLogger(__name__)


def find_first_matching_in_subdir(s3_client, bucket, prefix, pattern, batch_size=1000, max_files=10000):
    """
    Find first file matching pattern in a subdirectory using batched pagination.

    Stops as soon as the first matching file is found, or after checking max_files.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: S3 prefix (subdirectory path)
        pattern: Pattern to match (e.g., 'P099', 'P055')
        batch_size: Number of files to fetch per request
        max_files: Maximum number of files to check before giving up (default: 10000)

    Returns:
        str or None: S3 path to first matching file, or None if not found
    """
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
        PaginationConfig={'PageSize': batch_size}
    )

    files_checked = 0
    subdir_name = prefix.rstrip('/').split('/')[-1]
    target_suffix = f"_{pattern}.nc"

    try:
        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in page['Contents']:
                files_checked += 1
                key = obj['Key']

                # Skip if it's a directory marker
                if key.endswith('/'):
                    continue

                # Check if it ends with the target pattern
                if key.endswith(target_suffix):
                    logger.debug(f"  {subdir_name}: found after checking {files_checked} files")
                    return f"s3://{bucket}/{key}"

                # Stop if we've checked enough files
                if files_checked >= max_files:
                    logger.debug(f"  {subdir_name}: not found after checking {files_checked} files (limit reached)")
                    return None

        logger.debug(f"  {subdir_name}: no {pattern} files found (checked {files_checked} files)")
        return None

    except Exception as e:
        logger.error(f"  Error in {subdir_name}: {e}")
        return None


def collect_matching_granules(base_path, pattern, limit=None, batch_size=1000):
    """
    Collect first file matching pattern from each subdirectory.

    Args:
        base_path: S3 path to scan (e.g., 's3://its-live-data/velocity_image_pair/landsatOLI/v02/')
        pattern: Pattern to match (e.g., 'P099', 'P055')
        limit: Maximum number of subdirectories to process (None for all)
        batch_size: Files per S3 request (default: 1000)

    Returns:
        list: S3 paths to collected granules
    """
    # Use s3fs to list directories
    s3 = aws_utils.make_s3fs()

    # Use boto3 client for efficient file listing
    s3_client = boto3.client('s3')

    # Normalize base path
    if base_path.startswith('s3://'):
        base_path = base_path[5:]
    base_path = base_path.rstrip('/')

    # Extract bucket and prefix
    parts = base_path.split('/', 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''

    logger.info(f"Scanning subdirectories in s3://{base_path}/")

    # List all subdirectories
    try:
        subdirs = s3.ls(base_path)
    except Exception as e:
        logger.error(f"Error listing {base_path}: {e}")
        return []

    # Filter to directories only
    subdirs = [d for d in subdirs if s3.isdir(d)]

    logger.info(f"Found {len(subdirs)} subdirectories")

    if limit:
        subdirs = subdirs[:limit]
        logger.info(f"Limited to first {limit} subdirectories")

    collected = []

    for i, subdir in enumerate(subdirs, 1):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(subdirs)}...")

        # Convert subdir path to prefix (remove bucket name)
        subdir_prefix = subdir.split('/', 1)[1] if '/' in subdir else subdir
        if not subdir_prefix.endswith('/'):
            subdir_prefix += '/'

        # Find first matching file using batched pagination
        # Limit to first 10,000 files to avoid processing subdirs with millions of files
        first_file = find_first_matching_in_subdir(
            s3_client,
            bucket,
            subdir_prefix,
            pattern,
            batch_size=batch_size,
            max_files=10000
        )

        if first_file:
            collected.append(first_file)

    logger.info(f"Collected {len(collected)} granules from {len(subdirs)} subdirectories")

    return collected


def main():
    parser = argparse.ArgumentParser(
        description='Collect first granule matching pattern from each subdirectory in S3'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file (default: granules_PXXX.json based on pattern)'
    )
    parser.add_argument(
        '--pattern',
        default='P099',
        help='Pattern to match in filenames (e.g., P099, P055) [%(default)s]'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit to first N subdirectories'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of files to fetch per S3 request [%(default)d]'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--base-path',
        default='s3://its-live-data/velocity_image_pair/landsatOLI/v02/',
        help='S3 base path to scan[%(default)s]'
    )

    args = parser.parse_args()

    # Set default output filename based on pattern if not specified
    if args.output is None:
        args.output = f'granules_{args.pattern}.json'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set this script's logger level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Suppress verbose boto3/botocore/s3fs logs
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('s3fs').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    # Collect granules
    granules = collect_matching_granules(
        args.base_path,
        args.pattern,
        limit=args.limit,
        batch_size=args.batch_size
    )

    # Write to output file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(granules, f, indent=2)

    logger.info(f"Wrote {len(granules)} granule paths to {output_path}")

    # Log first few as preview
    if granules:
        logger.info("First 5 granules:")
        for granule in granules[:5]:
            logger.info(f"  {granule}")
        if len(granules) > 5:
            logger.info(f"  ... and {len(granules) - 5} more")


if __name__ == '__main__':
    main()
