#!/bin/bash

# Helper script to delete datacubes as listed in the input file.
#
# The script expects datacubes S3 URLs:
# % head ../../landsat_failed_to_copy.txt
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N60W140/ITS_LIVE_vel_EPSG3413_G0120_X-2750000_Y650000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70W060/ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-1050000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70W050/ITS_LIVE_vel_EPSG3413_G0120_X-350000_Y-1950000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N50W120/ITS_LIVE_vel_EPSG3413_G0120_X-3750000_Y-550000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70W030/ITS_LIVE_vel_EPSG3413_G0120_X250000_Y-1250000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70W020/ITS_LIVE_vel_EPSG3413_G0120_X550000_Y-1450000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70W020/ITS_LIVE_vel_EPSG3413_G0120_X550000_Y-1550000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/N70E090/ITS_LIVE_vel_EPSG3413_G0120_X750000_Y950000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/S80W090/ITS_LIVE_vel_EPSG3031_G0120_X-1050000_Y-50000.zarr
# s3://its-live-data.jpl.nasa.gov/datacubes/v01/S80W080/ITS_LIVE_vel_EPSG3031_G0120_X-1050000_Y150000.zarr

filename="$1"
while read -r line; do
    name="$line"
    echo "Deleting - $name"
    # aws s3 ls $name
    aws s3 rm --recursive $name
done < "$filename"
