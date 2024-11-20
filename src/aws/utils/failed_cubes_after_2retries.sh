#!/bin/bash

FAILED_DIR="failed"

# Check if a command-line argument is provided for FAILED_DIR
if [ -n "$1" ]; then
    FAILED_DIR="$1"
fi

# Directory to move successfully processed files - processed OK on second retry
OK_FAILED_DIR="${FAILED_DIR}_ok"
mkdir -p "$OK_FAILED_DIR"

# Directory paths
DONE_DIR="done"

# List all files in the failed directory starting with ITS_LIVE_vel prefix
failed_files=$(find "$FAILED_DIR" -type f -name "ITS_LIVE_vel*.log")

# Iterate over each failed file
for failed_file in $failed_files; do
    # Extract the base filename without the timestamp and extension
    base_filename=$(basename "$failed_file" | sed -E 's/(ITS_LIVE_vel_EPSG[0-9]+_G[0-9]+_X-?[0-9]+_Y-?[0-9]+\.zarr).*/\1/')
    echo "Base filename: $base_filename"

    # Check if a corresponding file exists in the done directory
    if find "$DONE_DIR" -type f -name "$base_filename*" | grep -q .; then
        echo "Found corresponding file in 'done' folder for: $failed_file"
        # Move the failed file to the OK_FAILED_DIR
        echo "Moving $failed_file to $OK_FAILED_DIR"
        mv "$failed_file" "$OK_FAILED_DIR"
    else
        echo "No corresponding file in 'done' folder for: $failed_file"
    fi
done