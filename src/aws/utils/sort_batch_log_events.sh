#!/bin/bash
#
# This script sorts events within each exported Batch log stream in
# chronological order and renames log files according to the datacube
# it corresponds to.
# AWS does not export events within log stream in chronological order as of
# September 2021.
#
# To run the script, execute the following command from the top-level directory
# that stores log streams for all exported Batch jobs:
#
# find . -type d -exec ~/sort_batch_log_events.sh \{\} \;
#
# Later on useful commands to sort logs out:
# ATTN: should automate these commands?
#
# grep Traceback *log_group
# ==========================
# mkdir failed_to_copy
# >find . -type f -exec grep -qiF 'Failed to copy' {} \; -exec mv {} failed_to_copy \;

# mkdir zero_granules_by_searchAPI
# >grep 'Number of found by API granules: 0' *log | wc -l
# >find . -type f -exec grep -qiF 'Number of found by API granules: 0' {} \; -exec mv {} zero_granules_by_searchAPI \;

# mkdir done
# >find . -type f -exec grep -qiF 'Done.' {} \; -exec mv {} done \;
# >find . -type f -exec grep -l "Done\." {} \; | xargs -I {} mv {} done

# Search only top level directory:
# find . -maxdepth 1 -type f

# Jobs that failed to get granules from the searchAPI
# mkdir failed_searchAPI

# find . -type f -exec grep -qiF 'RuntimeError: Failed to get granules from searchAPI.' {} \; -exec mv {} failed_searchAPI \;
# Get the cubes names in JSON format
# grep 'Cube S3: ' *log| awk -F' ' '{print $1}' >> failed_searchAPI.txt
# cat failed_searchAPI.txt | sed 's/\.zarr.*/\.zarr",/' >> failed_searchAPI_list.json.tmp
# cat failed_searchAPI_list.json.tmp | sed 's/ITS/"ITS/' >> failed_searchAPI_list.json

# find . -type f -exec grep -qiF 'Killed' {} \; -exec mv {} killed \;

# Extract number of granules as returned by the searchAPI from all log files in the directory,
# sort them by the number of granules
# find . -type f -name '*.log' -exec awk '/Number of found by API granules: [0-9]*/ {print $14, FILENAME}' {} + | sort -k1,1n |& tee done_number_of_granules.txt

# Keep only filenames from "number_of_granules log_filename"
# cat filename.txt | sed 's/.* \.\///' | awk -F'/' '{print $NF}'

# Format logs of datacubes to the JSON list of datacubes:
# cat failed_number_of_granules_list.txt | sed 's/\.zarr.*/\.zarr",/' >> failed_number_of_granules_list.json.tmp
# cat failed_number_of_granules_list.json.tmp | sed 's/ITS/"ITS/' >> failed_number_of_granules_list.json

# Jobs that created cubes successfully
# mkdir wrote_cubes
# >find done -type f -exec grep -qiF 'Wrote' {} \; -exec mv {} wrote_cubes \;

# mkdir failed_to_rm_original_cube
# >find . -type f -exec grep -qiF 'Failed to remove original' {} \; -exec mv {} failed_to_rm_original_cube \;

# How many logs contain ValueError due to new datetime format including microseconds
# mkdir value_error
# >find . -type f -exec grep -qiF 'raise ValueError' {} \; -exec mv {} value_error \;

# How many jobs wrote any layers
# >grep -l Wrote *log | wc -l

# List S3 URLs for cubes that failed to copy
# grep Creating failed_to_copy/* | awk -F'Creating ' '{print $NF}' >> failed_to_copy.txt

# Find maximum number of time series points for all datacubes listed in file
# grep '0:100, 0:100' done/*log >> done.txt:
# ` done/ITS_LIVE_vel_EPSG3031_G0120_X-1050000_Y-1050000.zarr.log:2022-06-25T02:35:24.708Z 06/25/2022 02:35:24 AM - INFO - Loading vx[:, 0:100, 0:100] out of [6240, 833, 833]...`
# grep zarr done.txt| awk -F'out of' '{print $2}' | awk -F, '{print $1}' | sed 's/\[//' | sort -nr | head -1

# COMPOSITES:
# find . -type d -exec ~/sort_composites_batch_log_events.sh \{\} \; |& tee sort_logs.log
# find . -maxdepth 1 -type f -exec grep -l "Done\." {} \; | xargs -I {} mv {} done
# grep -l LinAlgError ITS_LIVE_vel_EPSG3* | xargs -I {}  mv {} LinAlgError/
# grep -l MemoryError *log | xargs -I {}  mv {}  MemoryError/

# See how many layers in original cube
# grep -o 'Total number of sensors in the cube: ([0-9]*,)' *log

# Get number of layers in each input datacube, and sort files by that number - to isolate large cubes to be
# processed with more powerful instances
#   grep -o 'Total number of sensors in the cube: ([0-9]*,)' *log | sed -r 's/Total number of sensors in the cube: \(([0-9]*),\)/ \1/' | sort -t ':' -k 2,2n

echo $1
# Actual exported Batch log file archive
FILE="$1/000000.gz"

# Filename for sorted log stream with "Completed" progress bars removed
NEW_FILE="$1/000000_sorted_log_compact.txt"

if test -f $FILE; then
  # Sort the logs
  echo $FILE
  find $FILE -exec zcat {} + | sed -r 's/^[0-9]+/\x0&/' | sort -z | strings | grep -v Completed >> $NEW_FILE

  # Extract datacube filename the log stream corresponds to
  # CUBE_NAME=$(grep Creating $NEW_FILE | awk -F/ '{print $NF}')
  CUBE_NAME=$(grep 'Cube S3: ' $NEW_FILE | awk -F/ '{print $NF}')
  CUBE_TIME=$(grep 'Cube S3: ' $NEW_FILE | awk -F' ' '{print $1}')

  # CUBE_NAME=$(grep 'Cube S3:' $NEW_FILE | awk -F/ '{print $NF}')

  # When processing logs for composites
  # CUBE_NAME=$(grep 'Reading existing' $NEW_FILE | awk -F/ '{print $NF}')
  # CUBE_TIME=$(grep 'Reading existing' $NEW_FILE | awk -F' ' '{print $1}')

  # Move sorted log file to the base directory as datacube.log file, add timestamp if there are multiple logs
  # for the same datacube - multiple jobs re-issued for the same cube due to failure
  CUBE_LOG_FILE=${CUBE_NAME}_${CUBE_TIME}.log
  echo "Moving log file ${NEW_FILE} to ${CUBE_LOG_FILE}"
  (mv $NEW_FILE $CUBE_LOG_FILE)
fi
