#!/bin/bash
# Script to create COGs for 'v' data variable of static mosaics in EPSG 3031 projection.

# Enable command echoing
set -x

# Specify the S3 bucket target location
target_bucket="s3://its-live-data/velocity_mosaic/v2/static/cog_global/epsg_3031"

# S3 bucket location for mosaics COGs
bucket="s3://its-live-data/velocity_mosaic/v2/static/cog"

# List of 2d variables for static mosaics
variables=(
    "v"
)

# Number of available CPUs for processing: kh9-vm.12xlarge
num_threads=46

# Iterate over variables
for var in "${variables[@]}"; do
    # Array of static mosaics TIF files for the variable, keep only selected mosaics: RGI 17, 18, 19
    downloaded_files=$(awsv2 s3 ls "$bucket"/ | grep v02_"$var".tif | grep -E 'RGI1[789]' | awk '{print $NF}')

    reprojected_files=()
    for filename in ${downloaded_files[@]}; do
        # Download the file
        awsv2 s3 cp "$bucket/$filename" "$filename"

        # Format output filename
        output_filename=$(echo "$filename" | sed 's/\.tif/_3031\.tif/')
        reprojected_files+=("$output_filename")

        # Reproject to the Google Maps projection (epgs:3031)
        if [[ $output_filename == *"RGI19A"* ]]; then
            # echo "Filename includes the token 'RGI19A'."
            # The COG is already in the epsg:3031 projection, just copy original file to expected filename
            cp "$filename" "$output_filename"
            # gdalwarp -of COG -co "BIGTIFF=YES" "$filename" -t_srs epsg:3031 "$output_filename" -r bilinear -wm 9000 -multi -wo "NUM_THREADS=$num_threads" -te -180 -87 180 -60 -te_srs epsg:3031
        else
            # echo "Filename does not include the token 'RGI19A'."
            gdalwarp -of COG -co "BIGTIFF=YES" "$filename" -t_srs epsg:3031 "$output_filename" -r bilinear -wm 9000 -multi -wo "NUM_THREADS=$num_threads"
        fi

        # Remove local copy of original file
        rm "$filename"
    done

    # Output filename for the global mosaic file
    first_file=$(echo "$downloaded_files" | head -n1)
    global_filename=$(echo "$first_file" | sed 's/_RGI[0-9][0-9]A//')
    global_filename="${global_filename/.tif/.vrt}"

    # Call gdalbuild to create global mosaics for the variable
    gdalbuildvrt "$global_filename" *3031.tif

    # Copy all contributing TIFs to the VRT to the target S3 destination
    for filename in ${reprojected_files[@]}; do
        # Upload the file
        awsv2 s3 cp "$filename" "$target_bucket/$filename"

        # Remove local copy of the file
        rm "$filename"
    done

    awsv2 s3 cp "$global_filename" "$target_bucket/$global_filename"
    rm "$global_filename"
done
