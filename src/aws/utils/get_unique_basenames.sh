#!/bin/bash

FAILED_DIR="failed"

# Check if a command-line argument is provided for FAILED_DIR
if [ -n "$1" ]; then
    FAILED_DIR="$1"
fi

# List all files in the failed directory starting with ITS_LIVE_vel prefix
failed_files=$(find "$FAILED_DIR" -type f -name "ITS_LIVE_vel*.log")

# Use a set to store unique basenames
declare -A unique_basenames

# Iterate over each failed file
for failed_file in $failed_files; do
    # Extract the base filename without the timestamp and extension
    base_filename=$(basename "$failed_file" | sed -E 's/(ITS_LIVE_vel_EPSG[0-9]+_G[0-9]+_X-?[0-9]+_Y-?[0-9]+\.zarr).*/\1/')
    # echo "base_filename: $base_filename"

    # Extract number of granules in the cube
    num_cube_granules=$(grep 'Existing datacube granules:' $failed_file | awk '/Existing datacube granules:/ {print $(NF-3)}')

    # Extract number of granules to update the cube with
    num_granules=$(grep 'Leaving' $failed_file | grep 'granules' | awk '/Leaving/ {print $(NF-1)}')
    # echo "num_granules: $num_granules"

    # Add the base filename to the set and number of granules
    unique_basenames["$base_filename"]="$num_cube_granules $num_granules"
    # unique_basenames["$base_filename"]=$num_granules
done

# Print unique basenames followed by the number of granules: number of cube granules and number of granules to update with
echo "Unique basenames and their corresponding number of granules:"
for basename in "${!unique_basenames[@]}"; do
    echo "$basename: ${unique_basenames[$basename]}"
done
