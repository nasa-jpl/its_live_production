#!/bin/bash
set -euo pipefail

# === DEFAULTS ===
PROFILE="saml-pub"
QUEUES=()  # Will hold array of queue names

# === PARSE COMMAND-LINE ARGUMENTS ===
while [[ $# -gt 0 ]]; do
   case $1 in
      --queue)
         # Support both --queue q1 --queue q2 AND --queue q1,q2,q3
         if [[ "$2" == *,* ]]; then
               IFS=',' read -ra NEW_QUEUES <<< "$2"
               QUEUES+=("${NEW_QUEUES[@]}")
         else
               QUEUES+=("$2")
         fi
         shift 2
         ;;
      --profile)
         PROFILE="$2"
         shift 2
         ;;
      -*)
         echo "❌ Unknown option: $1"
         echo "Usage: $0 [--queue <name>] [--queue <name2>] [--profile <name>] <files.json>"
         exit 1
         ;;
      *)
         break
         ;;
   esac
done

# === FALLBACK: Use default queue if none provided ===
if [ ${#QUEUES[@]} -eq 0 ]; then
   QUEUES=(
      "datacube-spot-4vCPU-32GB"
      "datacube-ondemand-4vCPU-32GB"
      "datacube-spot-8vCPU-64GB"
      "datacube-ondemand-manual-8vCPU-64GB"
      "datacube-ondemand-16vCPU-128GB"
   )
fi

# === INPUT VALIDATION ===
if [ $# -eq 0 ]; then
   echo "❌ Usage: $0 [--queue <name> [--queue <name2>]] [--profile <name>] <files.json>"
   echo "  Example: $0 --queue q1 --queue q2 files.json"
   echo "  Or:      $0 --queue q1,q2,q3 files.json"
   exit 1
fi

INPUT_FILE="$1"
if [ ! -f "$INPUT_FILE" ]; then
   echo "❌ File not found: $INPUT_FILE"
   exit 1
fi

# === FETCH RUNNING JOBS FROM ALL QUEUES ===
echo "🔍 Profile: $PROFILE"
echo "🔍 Checking queues: ${QUEUES[*]}"
echo "🔍 Fetching running jobs from AWS Batch..."

# Collect all running job names from all queues
ALL_RUNNING_JOBS=()
for QUEUE in "${QUEUES[@]}"; do
   echo "  → Querying queue: $QUEUE"
   # Get job names from this queue and append to array
   JOBS_IN_QUEUE=$(awsv2 --profile "$PROFILE" batch list-jobs \
      --job-queue "$QUEUE" \
      --job-status RUNNING \
      --query 'jobSummaryList[*].jobName' \
      --output json)

   # Merge into master list (jq handles deduplication later if needed)
   if [ "$JOBS_IN_QUEUE" != "[]" ]; then
      ALL_RUNNING_JOBS+=("$JOBS_IN_QUEUE")
      # Show files that ARE running
      echo "$JOBS_IN_QUEUE" | jq -r --slurpfile files "$INPUT_FILE" '
      $files[0][] as $path |
      ($path | split("/") | last | sub("\\.zarr$"; "")) as $basename |
      .[] | select(. == "composite_\($basename)") |
      "▶️  \(. ) ← \($path)"
      ' || echo "  (none)"
   fi
done

# Combine all job lists into a single JSON array
if [ ${#ALL_RUNNING_JOBS[@]} -eq 0 ]; then
   COMBINED_JOBS="[]"
else
   # Use jq to flatten all arrays into one
   COMBINED_JOBS=$(printf '%s\n' "${ALL_RUNNING_JOBS[@]}" | jq -s 'add | unique')
fi

echo
echo "✅ FILES CURRENTLY BEING PROCESSED:"
echo "===================================="

# Show files that ARE running
echo "$COMBINED_JOBS" | jq -r --slurpfile files "$INPUT_FILE" '
$files[0][] as $path |
($path | split("/") | last | sub("\\.zarr$"; "")) as $basename |
.[] | select(. == "composite_\($basename)") |
"▶️  \(. ) ← \($path)"
' || echo "  (none)"

echo
echo "🟡 FILES NOT BEING PROCESSED (available to start):"
echo "=================================================="

# Show files that are NOT running
echo "$COMBINED_JOBS" | jq -r --slurpfile files "$INPUT_FILE" '
. as $running |
$files[0][] as $path |
($path | split("/") | last | sub("\\.zarr$"; "")) as $basename |
"composite_\($basename)" as $expected_job |
select( $running | index($expected_job) | not ) |
"⏹️  Not running: \($path)"
' || echo "  (all files are being processed!)"

# === SUMMARY ===
TOTAL_FILES=$(jq length "$INPUT_FILE")
RUNNING_COUNT=$(echo "$COMBINED_JOBS" | jq -r --slurpfile files "$INPUT_FILE" '
[ $files[0][] as $path |
   ($path | split("/") | last | sub("\\.zarr$"; "")) as $basename |
   .[] | select(. == "composite_\($basename)") ] | length
')
NOT_RUNNING_COUNT=$((TOTAL_FILES - RUNNING_COUNT))

echo
echo "📊 SUMMARY:"
echo "=========== "
echo "Total files in $INPUT_FILE     : $TOTAL_FILES"
echo "Currently being processed      : $RUNNING_COUNT"
echo "Not being processed (available): $NOT_RUNNING_COUNT"