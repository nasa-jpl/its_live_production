#!/bin/bash
# Script to remove original non-slow-error composites from the s3://its-live-data/composites/annual/v2.

# Enable command echoing
set -x

# Specify the S3 bucket location for slow_error composites
slow_error_bucket="s3://its-live-data/composites/annual/v2_slow_error"

# Specify the S3 bucket location for original composites that need to be backed up
original_composites_bucket="s3://its-live-data/composites/annual/v2"

# Iterate over static mosaics in the S3 bucket location
for each_dir in $(awsv2 s3 ls "$slow_error_bucket"/ | grep '\/' | awk '{ print $2 }' | sed 's:/$::'); do
    # List each of the subdirectories for the composite names: will have '/' as part of each subdirectory name
    s3_dir="$slow_error_bucket/$each_dir"

    # Iterate over composites in zarr format
    for each_zarr in $(awsv2 s3 ls "$s3_dir"/ | grep zarr | awk '{ print $2 }' | sed 's:/$::'); do
        # Copy each of the original zarr composites, that were replaced by slow_error version composite, to the backup directory
        original_filename="$original_composites_bucket/$each_dir/$each_zarr"

        # Remove original RGI05A and RGI19A composites that were re-generated with slow_error
        awsv2 s3 rm --dryrun $original_filename --recursive
    done
done

echo 'Done'