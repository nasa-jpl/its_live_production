#!/bin/bash
# Script to create COGs for all 2D data variables of static mosaics

# Enable command echoing
set -x

# Specify the S3 bucket location
bucket="s3://its-live-data/mosaics/annual/v2_slow_error"

# Target S3 bucket location
target_bucket="s3://its-live-data/velocity_mosaic/v2/static/cog"

# List of 2d variables for static mosaics
variables=(
    "landice"
    "floatingice"
    "count"
    "dv_dt"
    "dvx_dt"
    "dvy_dt"
    "outlier_percent"
    "v"
    "vx"
    "vy"
    "v_amp"
    "vx_amp"
    "vy_amp"
    "v_amp_error"
    "vx_amp_error"
    "vy_amp_error"
    "v_error"
    "vx_error"
    "vy_error"
    "v_phase"
    "vx_phase"
    "vy_phase"
)

# Iterate over static mosaics in the S3 bucket location
for filename in $(awsv2 s3 ls "$bucket"/ | grep .nc | grep 0000 | awk '{print $NF}'); do
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
