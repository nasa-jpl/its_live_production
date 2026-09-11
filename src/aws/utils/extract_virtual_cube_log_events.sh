#!/bin/bash
#
# Extracts the exported Batch log stream for a virtual datacube job (as
# exported by CloudWatch Logs to S3), sorts its records back into
# chronological order (CloudWatch does not export log stream events in
# chronological order), and renames it after the icechunk store it
# corresponds to.
#
# To run the script, execute the following command from the top-level
# directory that stores log streams for all exported Batch jobs, e.g.:
#
# find src/aws/batch_logs/virtual_cubes/09082026 -mindepth 1 -maxdepth 1 -type d -exec src/aws/utils/extract_virtual_cube_log_events.sh \{\} \;
#
# Output logs are written to itslive_logs/ (created relative to the
# current working directory) as <icechunk_store_basename>_<timestamp>.log
#
# Each exported line is prefixed with an ISO-8601 UTC timestamp
# (e.g. "2026-09-08T22:48:03.930Z ..."), so a plain (stable) lexicographic
# sort on whole lines restores chronological order.

echo "$1"

# Actual exported Batch log file archive
FILE="$1/000000.gz"

mkdir -p itslive_logs

if test -f "$FILE"; then
  # Identify which icechunk cube this job built by extracting the
  # --output-store argument value from the "Command:" log line
  COMMAND_LINE=$(gunzip -c "$FILE" | grep -m 1 -- "--output-store")
  CUBE_STORE=$(echo "$COMMAND_LINE" | sed -E "s/.*'--output-store', '([^']*)'.*/\1/")
  CUBE_NAME=$(basename "$CUBE_STORE")
  CUBE_TIME=$(echo "$COMMAND_LINE" | awk '{print $1}')

  if [ -z "$CUBE_NAME" ]; then
    echo "WARNING: could not determine cube name for $1, skipping"
    exit 0
  fi

  CUBE_LOG_FILE="itslive_logs/${CUBE_NAME}_${CUBE_TIME}.log"
  echo "Extracting log for cube \"${CUBE_NAME}\" to \"${CUBE_LOG_FILE}\" (sorted chronologically)"
  # Sort by the leading timestamp field only (-k1,1); -s (stable) preserves
  # the original relative order of lines that share the same timestamp,
  # e.g. the lines of a single multi-line log message
  gunzip -c "$FILE" | sort -s -k1,1 -o "$CUBE_LOG_FILE"
fi
