#!/bin/bash

# Helper script to delete datacubes as listed in the input file.
#
# The script expects S3 URLs:
# % head eightAdditionalComposites.txt
# s3://its-live-data/composites/annual/v02/N50W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3350000_Y150000.zarr
# s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3250000_Y150000.zarr
# s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3150000_Y150000.zarr
# s3://its-live-data/composites/annual/v02/N50W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3350000_Y250000.zarr
# s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3150000_Y250000.zarr
# s3://its-live-data/composites/annual/v02/N50W140/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3350000_Y350000.zarr
# s3://its-live-data/composites/annual/v02/N60W140/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3250000_Y350000.zarr
# s3://its-live-data/composites/annual/v02/N60W140/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3150000_Y350000.zarr

filename="$1"
while read -r line; do
    name="$line"
    echo "============== Procesing $name"

    dir_name="$(dirname $name)"
    base_filename="$(basename $name .zarr)"

    original=${dir_name}/${base_filename}.nc

    new_filename=${dir_name}/${base_filename}_original.nc
    echo "--->Moving original $original to $new_filename"
    aws s3 mv $original $new_filename

    new_filename=${dir_name}/${base_filename}.nc
    local_filename=${base_filename}.nc
    echo "--->Converting $name to nc format: $local_filename"
    python ./utils/composites_to_netcdf.py -i $name -o $local_filename

    echo "--->Copying $local_filename to $new_filename"
    aws s3 cp $local_filename $newfilename
done < "$filename"

echo "Done."
