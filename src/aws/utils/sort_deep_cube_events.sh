#!/bin/bash
#
# Sorts the per-cube logs produced by extract_deep_copy_cube_log_events.sh
# into subfolders by outcome:
#
# - done/: runs that reached the final "... - INFO - Done" log line
#   (successful completion, store uploaded and marked complete)
# - terminated/: runs cut short by a spot interruption or OOM kill
#   ("... Terminated" from a SIGTERM, or "... Killed python
#   deep_copy_cube_per_var_chunk.py" from the OOM killer) -- no Python
#   exception, so these are expected to resume cleanly from --progress-dir
#   markers on retry
#   - terminated/done/: of those, the ones whose cube (matched by
#     <zarr_store_name>, ignoring the trailing _<timestamp>) shows up in
#     done/ too -- i.e. a later attempt actually finished the cube, so
#     this terminated log is just a stale retry, not a cube still stuck
# - inprogress/: runs with no error Traceback and no "Done"/terminated
#   marker -- the job was still running when the CloudWatch log export
#   snapshot was taken
# - everything left in the top-level directory failed with an error
#   Traceback (as of Oct 2026, entirely the duplicate-"time"-coordinate
#   RuntimeError raised after materializing a cube locally)
#
# Run from the directory that contains itslive_logs/, e.g.:
#
# src/aws/utils/sort_deep_cube_events.sh src/aws/batch_logs/deep_copy_cubes/10012026/itslive_logs

LOGS_DIR="${1:-itslive_logs}"
TOTAL=$(find "$LOGS_DIR" -maxdepth 1 -name '*.log' | wc -l)

mkdir -p "$LOGS_DIR/done"
find "$LOGS_DIR" -maxdepth 1 -name '*.log' -exec grep -lE "INFO - Done$" {} + | xargs -I{} mv {} "$LOGS_DIR/done/"

mkdir -p "$LOGS_DIR/terminated"
find "$LOGS_DIR" -maxdepth 1 -name '*.log' -exec grep -lE "Terminated$|Killed +python" {} + | xargs -I{} mv {} "$LOGS_DIR/terminated/"

# A terminated run's cube name is its filename with the trailing
# _<timestamp>.log stripped (everything up to and including ".zarr");
# cross-reference that against done/ to find terminated attempts whose
# cube was later finished by a different (successful) attempt.
mkdir -p "$LOGS_DIR/terminated/done"
DONE_CUBE_NAMES=$(find "$LOGS_DIR/done" -maxdepth 1 -name '*.log' -exec basename {} \; | sed -E 's/^(.*\.zarr)_.*\.log$/\1/' | sort -u)
find "$LOGS_DIR/terminated" -maxdepth 1 -name '*.log' | while IFS= read -r f; do
  cube_name=$(basename "$f" | sed -E 's/^(.*\.zarr)_.*\.log$/\1/')
  if grep -Fxq "$cube_name" <<< "$DONE_CUBE_NAMES"; then
    mv "$f" "$LOGS_DIR/terminated/done/"
  fi
done

mkdir -p "$LOGS_DIR/inprogress"
find "$LOGS_DIR" -maxdepth 1 -name '*.log' -exec grep -L "Traceback (most recent call last)" {} + | xargs -I{} mv {} "$LOGS_DIR/inprogress/"

TERMINATED_DONE=$(find "$LOGS_DIR/terminated/done" -maxdepth 1 -name '*.log' | wc -l)
TERMINATED_STUCK=$(find "$LOGS_DIR/terminated" -maxdepth 1 -name '*.log' | wc -l)

echo "Done: $(find "$LOGS_DIR/done" -maxdepth 1 -name '*.log' | wc -l) / $TOTAL total"
echo "Terminated (spot/OOM, resumable): $((TERMINATED_DONE + TERMINATED_STUCK)) / $TOTAL total"
echo "  ...of which later finished (terminated/done/): $TERMINATED_DONE"
echo "  ...still stuck (left in terminated/): $TERMINATED_STUCK"
echo "In progress: $(find "$LOGS_DIR/inprogress" -maxdepth 1 -name '*.log' | wc -l) / $TOTAL total"
echo "Failed (remaining): $(find "$LOGS_DIR" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l) / $TOTAL total"
