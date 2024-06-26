"""
Post-processing for UTS_LIVE V2 mosaics.

The script masks out the regions (sets values to Nan's), that are specified
in the Polygons of the shapefile, for all data variables of the mosaic.

Command examples
================
* Mask out high noise areas of the RGI01A mosaics:

python ./itslive_annual_mosaics_postprocessing.py
    -s high_noise_areas_RGI01A.shp
    -m ITS_LIVE_velocity_120m_RGI01A_*_v02.nc
    -b s3://its-live-data/
    -t s3://its-live-data/mosaics/annual/v2/netcdf
    -d mosaics/annual/v2/postprocess

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Greene (JPL), Mark Fahnestock (UAF)
"""
import json
import logging
import numpy as np
import os
from osgeo import osr
import s3fs
from shapely.geometry import Polygon, Point
import subprocess
import xarray as xr

# Local imports
from grid import Bounds, Grid

from itscube import ITSCube
from itscube_types import Coords, BatchVars, DataVars, ShapeFile, CompDataVars
from itslive_annual_mosaics import ITSLiveAnnualMosaics, MosaicsOutputFormat

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Non-EPSG projection that can be provided on input
ESRICode = 102027
ESRICode_Proj4 = '+proj=lcc +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs +type=crs'


class ITSLiveAnnualMosaicsPostProcess:
    """
    Class to apply post-processing to the annual ITS_LIVE mosaics.
    """
    DRYRUN = False

    SPATIAL_EPSG_ATTR = 'spatial_epsg'

    MASK_VAR = 'data'

    def __init__(self, shapefile: str, bucket: str, mosaics_regex: str):
        """Initialize post-processing.

        Args:
            shapefile (str): Shapefile that contains polygons definitions.
            bucket (str): S3 bucket that contains mosaics files.
            mosaics_regex (str): Regex to match mosaics (by the region) within the S3 bucket.

        Raises:
            RuntimeError: No mosaics found
            RuntimeError: No geometry Polygons are provided in shapefile
        """
        self.s3 = s3fs.S3FileSystem()

        self.mosaics_files = self.s3.glob(os.path.join(bucket, mosaics_regex))
        if len(self.mosaics_files) == 0:
            raise RuntimeError(f"No mosaics found: {os.path.join(bucket, mosaics_regex)}.")

        logging.info(f'Found {len(self.mosaics_files)} mosaics files: {json.dumps(self.mosaics_files, indent=3)}')

        # Read shapefile
        self.mask = ITSCube.read_shapefile(shapefile)

        if len(self.mask.geometry) == 0:
            raise RuntimeError(f'No geometry Polygons are provided in {shapefile} shapefile: {self.mask}')

        else:
            logging.info(f'{shapefile} file contains {len(self.mask.geometry)} polygons to mask out')

        # To initialize by reading one of the mosaics files
        self.epsg = None
        self.ij_to_xy_transfer = None
        self.grid_spacing = None
        self.grid_x_min = None
        self.grid_x_max = None
        self.grid_y_min = None
        self.grid_y_max = None
        self.mask_ds = None

        # Load any mosaics file in and extract X/Y coordinates and EPSG code
        with self.s3.open(self.mosaics_files[0], mode='rb') as s3_file_obj:
            with xr.open_dataset(s3_file_obj, engine=ITSCube.NC_ENGINE) as ds:
                # Mosaic's EPSG
                self.epsg = int(ds.mapping.attrs[ITSLiveAnnualMosaicsPostProcess.SPATIAL_EPSG_ATTR])

                output_projection = osr.SpatialReference()

                if self.epsg != ESRICode:
                    output_projection.ImportFromEPSG(self.epsg)

                else:
                    output_projection.ImportFromProj4(ESRICode_Proj4)

                input_projection = osr.SpatialReference()
                input_projection.ImportFromEPSG(int(BatchVars.LON_LAT_PROJECTION))
                input_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


                # Initialize transfer from lon/lat to mosaic's EPSG
                self.ij_to_xy_transfer = osr.CoordinateTransformation(input_projection, output_projection)

                # Mosaics X/Y coordinates
                x_coords = ds.x.values
                self.grid_spacing = x_coords[1] - x_coords[0]

                half_cell_size = np.abs(self.grid_spacing/2.0)

                # Mosaics range for x and y based on grid edges
                self.grid_x_min = x_coords.min() - half_cell_size
                self.grid_x_max = x_coords.max() + half_cell_size

                y_coords = ds.y.values
                self.grid_y_min = y_coords.min() - half_cell_size
                self.grid_y_max = y_coords.max() + half_cell_size

                logging.info(f'Mosaics x bounds: [{self.grid_x_min}, {self.grid_x_max}]')
                logging.info(f'Mosaics y bounds: [{self.grid_y_min}, {self.grid_y_max}]')

                # Create xarray mask that represents all polygons
                self.mask_ds = xr.Dataset(
                    coords={
                        Coords.X: (
                            Coords.X,
                            ds.x.values,
                            ds[Coords.X].attrs
                        ),
                        Coords.Y: (
                            Coords.Y,
                            ds.y.values,
                            ds[Coords.Y].attrs
                        )
                    }
                )

    def __call__(self, target_bucket: str,  target_bucket_dir):
        """
        Apply post-processing to identified mosaics files.

        Args:
            target_bucket (str): Target S3 bucket location to place post-processed mosaics into.
        """
        # Load all shapefile's polygons in and populate mask with grid pixels to exclude from
        # mosaics
        for index, each_polygon in enumerate(self.mask.geometry):
            logging.info(f'Processing polygon #{index}...')

            # Convert lon/lat to x/y in EPSG
            polygon_coords_array = np.array(each_polygon.exterior.coords)
            polygon_out = self.ij_to_xy_transfer.TransformPoints(polygon_coords_array)

            coordinates_without_zeros = [(x, y) for x, y, _ in polygon_out]
            target_polygon = Polygon(coordinates_without_zeros)

            out_x = Bounds([each[0] for each in coordinates_without_zeros])
            out_y = Bounds([each[1] for each in coordinates_without_zeros])

            logging.info(f'-->x bounds: {out_x}')
            logging.info(f'-->y bounds: {out_y}')

            # x0_bbox, y0_bbox = Grid.bounding_box(out_x, out_y, self.grid_spacing)

            # Create grid that corresponds to the Polygon
            mask_x_grid, mask_y_grid = Grid.create(out_x, out_y, self.grid_spacing)

            x_grid, y_grid = np.meshgrid(mask_x_grid, mask_y_grid)

            # Check which grid points are inside the polygon
            grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

            points_inside_polygon = np.array([target_polygon.contains(Point(x, y)) for x, y in grid_points])
            points_inside_polygon = points_inside_polygon.reshape(len(mask_y_grid), len(mask_x_grid))

            polygon_dims = {'x': mask_x_grid, 'y': mask_y_grid}
            polygon_mask = xr.DataArray(points_inside_polygon, dims=('y', 'x'), coords=polygon_dims)

            # Define which points are within mosaic X/Y ranges
            mask_x = (polygon_mask.x >= self.grid_x_min) & (polygon_mask.x <= self.grid_x_max)
            mask_y = (polygon_mask.y >= self.grid_y_min) & (polygon_mask.y <= self.grid_y_max)
            mask = (mask_x & mask_y)

            if mask.values.sum() == 0:
                # One or both masks resulted in no coverage
                logging.info('Skipping polygon since it does not overlap with mosaics')

            else:
                # Reduce polygon to the mosaics coverage
                polygon_mask = polygon_mask.where(mask, drop=True)

                if ITSLiveAnnualMosaicsPostProcess.MASK_VAR not in self.mask_ds:
                    self.mask_ds[ITSLiveAnnualMosaicsPostProcess.MASK_VAR] = polygon_mask

                else:
                    # Update polygon dimensions since it was cropped to the mosaics region
                    polygon_dims = {'x': polygon_mask.x.values, 'y': polygon_mask.y.values}

                    self.mask_ds[ITSLiveAnnualMosaicsPostProcess.MASK_VAR].loc[polygon_dims] = polygon_mask

                self.mask_ds[ITSLiveAnnualMosaicsPostProcess.MASK_VAR] = self.mask_ds[ITSLiveAnnualMosaicsPostProcess.MASK_VAR].fillna(0).astype(bool)

            # For debugging: plot final mask
            # self.mask_ds.data.plot(x='x', y='y')

        copy_file_to_s3 = not ITSLiveAnnualMosaicsPostProcess.DRYRUN

        if ITSLiveAnnualMosaicsPostProcess.MASK_VAR not in self.mask_ds:
            logging.info('Polygons do not overlap mosaics, just copy original mosaics to the target S3 destination')

            _s3_prefix = 's3://'

            # There is no polygon area that overlaps mosaics, just copy source mosaics
            # to its target destination
            for each_file in self.mosaics_files:
                each_s3_path = each_file

                if not each_file.startswith(_s3_prefix):
                    each_s3_path = f'{_s3_prefix}{each_file}'

                target_file = os.path.basename(each_file)
                target_file = os.path.join(target_bucket, target_bucket_dir, target_file)

                logging.info(f'Copying {each_s3_path} to {target_file}')

                if copy_file_to_s3:
                    command_line = [
                        "awsv2", "s3", "cp",
                        each_s3_path,
                        target_file,
                        "--acl", "bucket-owner-full-control"
                    ]

                    logging.info(' '.join(command_line))

                    command_return = None
                    env_copy = os.environ.copy()

                    logging.info(f"Copy {each_s3_path} to {target_file}")

                    command_return = subprocess.run(
                        command_line,
                        env=env_copy,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                    )

                    if command_return.returncode != 0:
                        # Report the whole stdout stream as one logging message
                        raise RuntimeError(f"Failed to copy {each_s3_path} to {target_file} with returncode={command_return.returncode}: {command_return.stdout}")

            return

        # Mask out all variables data based on created mask
        for each_file in self.mosaics_files:
            with self.s3.open(each_file, mode='rb') as s3_file_obj:
                with xr.open_dataset(s3_file_obj, engine=ITSCube.NC_ENGINE, decode_timedelta=False) as ds:
                    logging.info(f'Masking values for {each_file}...')
                    basename_file = os.path.basename(each_file)

                    ds = ds.load()

                    for each_var in ds.keys():
                        if each_var not in [DataVars.MAPPING, ShapeFile.FLOATINGICE, ShapeFile.LANDICE]:
                            logging.info(f'--->{each_var}')

                            other_value = np.nan
                            if each_var in [CompDataVars.SENSOR_INCLUDE]:
                                # Use binary flag's "include" value for masked out polygons:
                                # per Alex: "1 = excluded by the filtering algorithm... not the masking"
                                other_value = 0

                            ds[each_var] = ds[each_var].where(~self.mask_ds[ITSLiveAnnualMosaicsPostProcess.MASK_VAR], other=other_value)

                    # Change missing_value for CompDataVars.SENSOR_INCLUDE

                    # Data variables which dtype should be uint8 in final mosaics with
                    # fill value = 255
                    MosaicsOutputFormat.UINT8_TYPES = [
                        CompDataVars.OUTLIER_FRAC
                    ]

                    # Data variables which dtype should be uint8 in final mosaics with
                    # fill value = 0
                    MosaicsOutputFormat.UINT8_TYPES_ZERO_MISSING_VALUE = [
                        ShapeFile.LANDICE,
                        ShapeFile.FLOATINGICE,
                        CompDataVars.SENSOR_INCLUDE
                    ]

                    if ITSLiveAnnualMosaics.SUMMARY_KEY in basename_file:
                        # This is a summary mosaic
                        ITSLiveAnnualMosaics.summary_mosaic_to_netcdf(
                            ds,
                            mosaics_attrs={},
                            s3_bucket=target_bucket,
                            bucket_dir=target_bucket_dir,
                            filename=basename_file,
                            copy_to_s3=copy_file_to_s3
                        )

                    else:
                        # This is annual mosaic
                        ITSLiveAnnualMosaics.annual_mosaic_to_netcdf(
                            ds,
                            s3_bucket=target_bucket,
                            bucket_dir=target_bucket_dir,
                            filename=basename_file,
                            copy_to_s3=copy_file_to_s3
                        )


def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-s', '--shapeFile',
        type=str,
        action='store',
        default=None,
        help="Shapefile that stores polygon areas to mask out in mosaics [%(default)s]."
    )
    parser.add_argument(
        '-m', '--mosaicsRegex',
        type=str,
        action='store',
        default=None,
        help="File regex to match mosaics in S3 bucket location [%(default)s]."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data/mosaics/annual/v2/netcdf',
        help="S3 bucket location that stores mosaics files for post-processing [%(default)s]."
    )
    parser.add_argument(
        '-t', '--targetBucket',
        type=str,
        action='store',
        default='s3://its-live-data',
        help="S3 bucket to store post-processed mosaics files [%(default)s]"
    )
    parser.add_argument(
        '-d', '--targetBucketDir',
        type=str,
        action='store',
        default='mosaics/annual/v2/postprocessing',
        help="S3 bucket location to store post-processed mosaics files [%(default)s]"
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not copy mosaics to AWS S3 bucket'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import argparse
    import warnings
    import sys

    warnings.filterwarnings('ignore')

    logging.info(f"Command-line arguments: {sys.argv}")

    # Parse command-line arguments
    args = parse_args()
    logging.info(f"Command arguments: {args}")

    # Set static data for processing
    ITSLiveAnnualMosaicsPostProcess.DRYRUN = args.dryrun

    postProcess = ITSLiveAnnualMosaicsPostProcess(args.shapeFile, args.bucket, args.mosaicsRegex)
    postProcess(args.targetBucket, args.targetBucketDir)

    logging.info(f'Processed mosaics files: {json.dumps(postProcess.mosaics_files, indent=3)}')
    logging.info("Done.")
