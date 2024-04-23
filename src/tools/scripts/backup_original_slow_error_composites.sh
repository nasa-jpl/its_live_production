#!/bin/bash
# Script to back up original non-slow-error composites to the s3://its-live-project/data_backup.

# Enable command echoing
set -x

# Specify the S3 bucket location for slow_error composites
slow_error_bucket="s3://its-live-data/composites/annual/v2_slow_error"

# Specify the S3 bucket location for original composites that need to be backed up
original_composites_bucket="s3://its-live-data/composites/annual/v2"

# Target S3 bucket location
target_bucket="s3://its-live-project/data_backup/composites/original_RGI05A_RGI19A_from_its-live-data"

# Iterate over static mosaics in the S3 bucket location
for each_dir in $(awsv2 s3 ls "$slow_error_bucket"/ | grep '\/' | awk '{ print $2 }' | sed 's:/$::'); do
    # List each of the subdirectories for the composite names: will have '/' as part of each subdirectory name
    s3_dir="$slow_error_bucket/$each_dir"

    # Iterate over composites in zarr format
    for each_zarr in $(awsv2 s3 ls "$s3_dir"/ | grep zarr | awk '{ print $2 }' | sed 's:/$::'); do
        # Copy each of the original zarr composites, that were replaced by slow_error version composite, to the backup directory
        original_filename="$original_composites_bucket/$each_dir/$each_zarr"
        target_filename="$target_bucket/$each_dir/$each_zarr"

        # Backup original RGI05A and RGI19A composites that were re-generated with slow_error
        awsv2 s3 cp $original_filename $target_filename --recursive --acl bucket-owner-full-control
    done
done

echo 'Done'