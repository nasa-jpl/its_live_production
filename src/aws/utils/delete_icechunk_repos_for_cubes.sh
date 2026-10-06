#!/bin/bash
# Remove the icechunk repos that match the ".zarr" datacube names in an input
# JSON file, e.g.:
#
#   ./delete_icechunk_repos_for_cubes.sh ../deep_copy_fixedDuplicateTime.json
#   ./delete_icechunk_repos_for_cubes.sh --force ../deep_copy_fixedDuplicateTime.json
#
# We keep _skippedGranules.json files as those can be used to exclude already
# known skipped granules from previous processing to optimize the runtime.
#
# Datacube names map to icechunk repos/skipped-granules files like:
#   ITS_LIVE_vel_EPSG3413_G0120_X30700_Y-952350.zarr
#     -> s3://its-live-data/datacubes/spatial/v2.2/ITS_LIVE_vel_EPSG3413_G0120_X30700_Y-952350.icechunk
#
# Without --force, this only prints what WOULD be deleted (dry run).
# Pass --force to actually delete from S3.
set -euo pipefail

S3_PREFIX="s3://its-live-data/datacubes/spatial/v2.2"

FORCE=0
if [ "${1:-}" == "--force" ]; then
   FORCE=1
   shift
fi

if [ $# -eq 0 ]; then
   echo "Usage: $0 [--force] <cubes.json>"
   exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
   echo "File not found: $INPUT_FILE"
   exit 1
fi

if [ "$FORCE" -eq 0 ]; then
   echo "DRY RUN - no files will be deleted. Pass --force to actually delete."
   echo
fi

jq -r '.[]' "$INPUT_FILE" | while read -r cube_name; do
   base_name="${cube_name%.zarr}"
   repo_path="${S3_PREFIX}/${base_name}.icechunk"

   if aws s3 ls "${repo_path}/" >/dev/null 2>&1; then
      if [ "$FORCE" -eq 1 ]; then
         echo "Deleting repo:    ${repo_path}"
         aws s3 rm --recursive "${repo_path}/"
      else
         echo "Would delete repo:    ${repo_path}"
      fi
   else
      echo "Repo not found, skipping: ${repo_path}"
   fi
done
