#!/bin/bash
# Script to create COGs for all 2D data variables of annual mosaics

# Enable command echoing
set -x

# Specify the S3 bucket location
bucket="s3://its-live-data/velocity_mosaic/v2/annual"

# Target S3 bucket location
target_bucket="s3://its-live-data/velocity_mosaic/v2/annual/cog"

# List of 2d variables for static mosaics
variables=(
    "count"
    "v"
    "vx"
    "vy"
    "v_error"
    "vx_error"
    "vy_error"
)

# Iterate over static mosaics in the S3 bucket location
for filename in $(awsv2 s3 ls "$bucket"/ | grep .nc | grep RGI10A | grep -v 2023 | awk '{print $NF}'); do
    # Copy file locally
    awsv2 s3 cp "$bucket/$filename" "$filename"

    # Iterate over variables
    for var in "${variables[@]}"; do
        # Format output filename
        output_filename="${filename/.nc/_$var.tif}"

        # Call gdal_translate for each file and variable
        gdal_translate -of COG -co "BIGTIFF=YES" NETCDF:\"$filename\":"$var" "$output_filename"
        awsv2 s3 cp "$output_filename" "$target_bucket/$output_filename"
        rm "$output_filename"
    done

    rm "$filename"
done
