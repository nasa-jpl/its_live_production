#!/bin/bash

awsv2 s3api list-objects-v2 \
--bucket its-live-data \
--prefix "velocity_mosaic/v2.1/production/post_process/ITS_LIVE_velocity_120m_" \
--query 'Contents[].Key' \
--output json | jq -r '.[]' | head -5


echo "🔍 Listing and filtering NetCDF files..."

awsv2 s3api list-objects-v2 \
--bucket its-live-data \
--prefix "velocity_mosaic/v2.1/production/post_process/ITS_LIVE_velocity_120m_" \
--query 'Contents[].Key' \
--output json |
jq -r '.[]' |
while IFS= read -r key; do
# Skip if not .nc (shouldn't happen, but safe)
[[ "$key" != *.nc ]] && continue

filename=$(basename "$key")
echo "📄 File: $filename"

# Extract the 4-digit year between RGI??A_ and _V
if [[ $filename =~ RGI[0-9]{2}A_([0-9]{4})_V ]]; then
   year=${BASH_REMATCH[1]}
   echo "   → Extracted year: $year"

   # Skip 0000 and years outside 1982–2024
   if [[ $year == "0000" ]]; then
   echo "   → Skipping placeholder year (0000)"
   elif (( year >= 1982 && year <= 2024 )); then
   echo "   → ✅ MATCH: Copying $filename"
   awsv2 s3 cp "s3://its-live-data/$key" "s3://its-live-data/velocity_mosaic/v2.1/annual/$filename" < /dev/null
   else
   echo "   → 🚫 Skipping (year out of range): $year"
   fi
else
   echo "   → ❌ No year pattern match"
fi

echo "────────────────────────────"
done

echo "✅ Done."