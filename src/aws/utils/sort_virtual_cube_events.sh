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
#   - inprogress/done/: of those, the ones whose cube (matched by
#     <icechunk_store_name>, ignoring the trailing _<timestamp>) shows up
#     in done/ too -- i.e. a later attempt actually finished the cube, so
#     this inprogress log is just a stale snapshot of a run that was later
#     terminated (e.g. an EC2/spot interruption) and successfully retried,
#     not a cube still stuck
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

# An inprogress run's cube name is its filename with the trailing
# _<timestamp>.log stripped (everything up to and including ".icechunk");
# cross-reference that against done/ to find inprogress snapshots whose
# cube was later finished by a different (successful) attempt -- these
# correspond to runs cut short by an earlier EC2 termination.
mkdir -p "$LOGS_DIR/inprogress/done"
DONE_CUBE_NAMES=$(find "$LOGS_DIR/done" -maxdepth 1 -name '*.log' -exec basename {} \; | sed -E 's/^(.*\.icechunk)_.*\.log$/\1/' | sort -u)
find "$LOGS_DIR/inprogress" -maxdepth 1 -name '*.log' | while IFS= read -r f; do
  cube_name=$(basename "$f" | sed -E 's/^(.*\.icechunk)_.*\.log$/\1/')
  if grep -Fxq "$cube_name" <<< "$DONE_CUBE_NAMES"; then
    mv "$f" "$LOGS_DIR/inprogress/done/"
  fi
done

INPROGRESS_DONE=$(find "$LOGS_DIR/inprogress/done" -maxdepth 1 -name '*.log' | wc -l)
INPROGRESS_STUCK=$(find "$LOGS_DIR/inprogress" -maxdepth 1 -name '*.log' | wc -l)

echo "Done: $(find "$LOGS_DIR/done" -maxdepth 1 -name '*.log' | wc -l) / $TOTAL total"
echo "In progress: $((INPROGRESS_DONE + INPROGRESS_STUCK)) / $TOTAL total"
echo "  ...of which later finished (inprogress/done/, earlier EC2 termination): $INPROGRESS_DONE"
echo "  ...still stuck (left in inprogress/): $INPROGRESS_STUCK"
echo "Failed (remaining): $(find "$LOGS_DIR" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l) / $TOTAL total"
