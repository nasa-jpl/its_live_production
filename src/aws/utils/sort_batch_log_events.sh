#!/bin/bash
#
# This script sorts events within each exported Batch log stream in
# chronological order and renames log files according to the datacube
# it corresponds to.
# AWS does not export events within log s`tream in chronological order as of
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
# >find . -type f -exec grep -qiF 'Done' {} \; -exec mv {} done \;

# Jobs that created cubes successfully
# mkdir wrote_cubes
# >find done -type f -exec grep -qiF 'Wrote' {} \; -exec mv {} wrote_cubes \;

# How many logs contain ValueError due to new datetime format including microseconds
# mkdir value_error
# >find . -type f -exec grep -qiF 'raise ValueError' {} \; -exec mv {} value_error \;

# How many jobs wrote any layers
# >grep -l Wrote *log | wc -l

# List S3 URLs for cubes that failed to copy
# grep Creating failed_to_copy/* | awk -F'Creating ' '{print $NF}' >> failed_to_copy.txt

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
  CUBE_NAME=$(grep 'Cube S3:' $NEW_FILE | awk -F/ '{print $NF}')

  # Move sorted log file to the base directory as datacube.log file
  CUBE_LOG_FILE=${CUBE_NAME}.log
  echo "Moving log file ${NEW_FILE} to ${CUBE_LOG_FILE}"
  (mv $NEW_FILE $CUBE_LOG_FILE)
fi
