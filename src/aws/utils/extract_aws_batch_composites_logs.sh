#!/bin/bash
#
# This script extracts AWS logs events in chronological order within each
# exported Batch log stream  and renames log files according to the
# datacube it corresponds to.
# AWS does not export events within log stream in chronological order as of
# September 2021.
#
# To run the script, execute the following command from the top-level directory
# that stores log streams for all exported Batch jobs:
#
# find . -type d -exec ~/extract_aws_batch_composites_logs.sh \{\} \; |& tee extract_composites_logs.log
#

echo $1
# Actual exported Batch log file archive
FILE="$1/000000.gz"

# Filename for sorted log stream with "Completed" progress bars removed
NEW_FILE="$1/000000_sorted_log_compact.txt"

mkdir -p itslive_logs

if test -f $FILE; then
  # Sort the logs
  echo $FILE
  find $FILE -exec zcat {} + | sed -r 's/^[0-9]+/\x0&/' | sort -z | strings | grep -v Completed >> $NEW_FILE
  ls -lh $NEW_FILE

  CUBE_NAME=$(grep 'Reading existing' $NEW_FILE | awk -F/ '{print $NF}')
  CUBE_TIME=$(grep 'Reading existing' $NEW_FILE | awk -F' ' '{print $1}')

  echo "Cube name: $CUBE_NAME"
  echo "Cube time: $CUBE_TIME"

  # Move sorted log file to the base directory as datacube.log file, add timestamp if there are multiple logs
  # for the same datacube - multiple jobs re-issued for the same cube due to failure
  CUBE_LOG_FILE=${CUBE_NAME}_${CUBE_TIME}.log
  echo "Moving log file \"${NEW_FILE}\" to \"itslive_logs/${CUBE_LOG_FILE}\""
  mv $NEW_FILE itslive_logs/${CUBE_LOG_FILE}
fi

# Sort out all successfully processed jobs
cd itslive_logs
mkdir -p done

# Move logs with "Done." message to a separate directory
find . -type f -exec grep -l "Done\." {} \; | xargs -I {} mv {} done
