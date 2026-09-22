"""
Extract the first time layer from an ITS_LIVE velocity datacube stored as
Zarr on S3, and save it to a local NetCDF file.

This tool was created to be able to validate zarr cube grid in QGIS when
verifying new catalog geojson vs. granules chunk-aligned grid.

The script opens the "deep copy" (already-materialized) Zarr store directly.
"""
import argparse
import logging

import s3fs
import xarray as xr

logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_first_layer(zarr_url, output_path, anon=True):
   """Open a datacube Zarr store on S3, select time index 0, write to NetCDF.

   Parameters
   ----------
   zarr_url : str
      s3:// URL to the datacube .zarr store.
   output_path : str
      Local path for the output .nc file.
   anon : bool
      Whether to access S3 anonymously (its-live-data is a public bucket).
   """
   fs = s3fs.S3FileSystem(anon=anon)
   store = s3fs.S3Map(root=zarr_url, s3=fs, check=False)

   logging.info(f"Opening {zarr_url}")
   dask_chunks = {'mid_date': 250, 'x': 10, 'y': 10}
   ds = xr.open_zarr(store, decode_timedelta=False, consolidated=True, chunks=dask_chunks)
   logging.info(f"Full cube:\n{ds}")

   # First time layer only, keeping "time" as a length-1 dimension
   # (use ds.isel(time=0, drop=True) instead if you want it dropped)
   layer = ds.isel(mid_date=slice(0, 1))
   logging.info(f"Selected layer:\n{layer}")

   logging.info(f"Writing to {output_path}")
   layer.to_netcdf(output_path)
   logging.info("Done.")


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description=__doc__)
   parser.add_argument(
      "--zarr-url",
      type=str,
      default=(
         "s3://its-live-data/datacubes/v2-updated-october2024/S70W100/"
         "ITS_LIVE_vel_EPSG3031_G0120_X-1450000_Y-450000.zarr"
      ),
      help="S3 URL of the source Zarr datacube",
   )
   parser.add_argument(
      "--output",
      type=str,
      default="its_live_first_layer.nc",
      help="Output NetCDF file path",
   )
   args = parser.parse_args()

   extract_first_layer(args.zarr_url, args.output)
