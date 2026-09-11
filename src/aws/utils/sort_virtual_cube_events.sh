#!/bin/bash
#
# Sorts the per-cube logs produced by extract_virtual_cube_log_events.sh
# into subfolders by outcome:
#
# - done/: runs that reached the final "... - INFO - Done" log line
#   (successful completion; may or may not have written a cube, e.g. 0
#   granules found)
# - inprogress/: runs with no error Traceback and no "Done" marker - the
#   job was still running (loading granules) when the CloudWatch log
#   export snapshot was taken
# - everything left in the top-level directory failed with an error
#   Traceback (as of Sep 2026, almost entirely S3 "503 Service
#   Unavailable" throttling while loading Landsat OLI granules)
#
# Run from the directory that contains itslive_logs/, e.g.:
#
# src/aws/utils/sort_virtual_cube_events.sh src/aws/batch_logs/virtual_cubes/09082026/itslive_logs

LOGS_DIR="${1:-itslive_logs}"
TOTAL=$(find "$LOGS_DIR" -maxdepth 1 -name '*.log' | wc -l)

mkdir -p "$LOGS_DIR/done"
find "$LOGS_DIR" -maxdepth 1 -name '*.log' -exec grep -lE "INFO - Done$" {} + | xargs -I{} mv {} "$LOGS_DIR/done/"

mkdir -p "$LOGS_DIR/inprogress"
find "$LOGS_DIR" -maxdepth 1 -name '*.log' -exec grep -L "Traceback (most recent call last)\|obstore.exceptions.GenericError\|RuntimeError: Got exception loading granule_url" {} + | xargs -I{} mv {} "$LOGS_DIR/inprogress/"

echo "Done: $(find "$LOGS_DIR/done" -maxdepth 1 -name '*.log' | wc -l) / $TOTAL total"
echo "In progress: $(find "$LOGS_DIR/inprogress" -maxdepth 1 -name '*.log' | wc -l) / $TOTAL total"
echo "Failed (remaining): $(find "$LOGS_DIR" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l) / $TOTAL total"
