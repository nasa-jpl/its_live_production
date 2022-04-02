"""
ITSLiveMosaics class creates yearly mosaics of ITS_LIVE datacubes for the region.

Command example:
python ./itslive_annual_mosaics.py -c tools/catalog_datacubes_v02.json
    --processCubesWithinPolygon aws/regions/Alaska.geojson -e '[3413]'
    --mosaicsEpsgCode 3413 -r Alaska

Authors: Masha Liukis (JPL), Alex Gardner (JPL), Chad Green (JPL), Mark Fahnestock (UAF)
"""
import collections
import copy
import datetime
import gc
import json
import logging
import math
import numba as nb
import numpy  as np
import os
from pathlib import Path
import s3fs
from shapely import geometry
import timeit
from tqdm import tqdm
import xarray as xr

# Local imports
from grid import Bounds
import itslive_utils
# from itscube import ITSCube
from itscube_types import Coords, DataVars, BatchVars, CubeJson, FilenamePrefix
from itslive_composite import CompDataVars, CompOutputFormat

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

class MosaicsOutputFormat:
    """
    Data attributes specific to mosaics datasets.
    """
    COMPOSITES_S3 = 'composites_s3'
    COMPOSITES_URL = 'composites_url'
    COMPOSITES_CREATED = 'composites_created'
    COMPOSITES_UPDATED = 'composites_updated'
    MOSAICS_SOFTWARE_VERSION = 'mosaics_software_version'

    # Mapping of mosaics and composites attributes: some of composites attributes
    # will appear with names specific to mosaic.
    ATTR_MAP = {
        CompOutputFormat.DATE_CREATED: COMPOSITES_CREATED,
        CompOutputFormat.DATE_UPDATED: COMPOSITES_UPDATED,
        CompOutputFormat.S3:           COMPOSITES_S3,
        CompOutputFormat.URL:          COMPOSITES_URL
    }


class ITSLiveAnnualMosaics:
    """
    CLass to build annual mosaics based on composites for ITS_LIVE datacubes.
    """
    VERSION = '1.0'

    # "structure" to hold composites information for a single cube
    Composite = collections.namedtuple("Composite", ['s3', 'x', 'y', 'time', 'sensor'])

    # "structure" to store xarray.Dataset and correspsoinding s3_store for loading
    # of the data from AWS S3 bucket
    CompositeS3 = collections.namedtuple("CompositeS3", ['ds', 'ds_store'])

    # S3 store location for mosaics
    S3 = ''

    # URL location for mosaics
    URL = ''

    CELL_SIZE = 120

    REGION = None

    def __init__(self, epsg: str, grid_size: int, is_dry_run: bool):
        """
        Initialize object.

        epsg: Target EPSG code to create mosaics for.
        grid_size: Grid size for mosaics.
        is_dry_run: Flag to display steps to be taken without actually generating
                    mosaics.
        """
        self.epsg_str = epsg
        self.grid_size_str = f'{grid_size:04d}'
        self.is_dry_run = is_dry_run

        # Read datacube composites from S3 bucket
        self.s3 = s3fs.S3FileSystem(anon=True)

        # Identified composites for annual mosaics in form of
        # EPSG: {Y: {X: composite_file}}
        self.composites = {}

        # Opened composites xr.Dataset objects
        self.composites_ds = {}

        # Mapping xr.DataArray
        self.mapping = None

        # "united" coordinates for mosaics
        self.time_coords = []
        self.x_coords = []
        self.y_coords = []
        self.sensor_coords = []

    def create(
        self,
        cubes_file: str,
        s3_bucket: str,
        cube_dir: str,
        composite_dir: str,
        mosaics_dir: str
    ):
        """
        Create annual mosaics from datacubes composites for specified region.

        cubes_file: GeoJson catalog file with existing datacubes' Zarr S3 URLs.
        s3_bucket: S3 bucket that stores all data (assumes that all datacubes,
                   composites, and mosaics are stored in the same bucket).
        cube_dir:  Directory path within S3 bucket that stores datacubes.
        composite_dir: Directory path within S3 bucket that stores datacubes' composites.
        mosaics_dir:   Directory path within S3 bucket that stores datacubes' mosaics.
        """
        self.collect_composites(cubes_file, s3_bucket, cube_dir, composite_dir)

        self.make_mosaics(s3_bucket, mosaics_dir)

        logging.info(f"Done.")
        return

    def collect_composites(
        self,
        cubes_file: str,
        s3_bucket: str,
        cube_dir: str,
        composite_dir: str
    ):
        """
        Collect datacube composites for specified region.
        Iterate over existing cubes that qualify for the search criteria and
        use cube's corresponding composites to create annual mosaics.

        cubes_file: GeoJson catalog file with existing datacubes' Zarr S3 URLs.
        s3_bucket: S3 bucket that stores all data (assumes that all datacubes,
                   composites, and mosaics are stored in the same bucket).
        cube_dir:  Directory path within S3 bucket that stores datacubes.
        composite_dir: Directory path within S3 bucket that stores datacubes' composites.
        """
        with open(cubes_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to process
            num_processed = 0
            logging.info(f'Total number of datacubes: {len(cubes["features"])}')
            for each_cube in cubes[CubeJson.FEATURES]:
                # Example of data cube definition in json file
                # "properties": {
                #     "fill-opacity": 0.9848664555858357,
                #     "fill": "red",
                #     "roi_percent_coverage": 1.5133544414164224,
                #     "data_epsg": "EPSG:32718",
                #     "geometry_epsg": {
                #         "type": "Polygon",
                #         "coordinates": [
                #             [
                #                 [
                #                     400000,
                #                     4400000
                #                 ],
                #                 [
                #                     500000,
                #                     4400000
                #                 ],
                #                 [
                #                     500000,
                #                     4500000
                #                 ],
                #                 [
                #                     400000,
                #                     4500000
                #                 ],
                #                 [
                #                     400000,
                #                     4400000
                #                 ]
                #             ]
                #         ]
                #     },
                #     "datacube_exist": 1,
                #     "zarr_url": "http://its-live-data.s3.amazonaws.com/datacubes/v02/S50W070/ITS_LIVE_vel_EPSG32718_G0120_X450000_Y4450000.zarr"
                # }

                # Consider cubes with ROI != 0 only
                properties = each_cube[CubeJson.PROPERTIES]

                roi = properties[CubeJson.ROI_PERCENT_COVERAGE]
                if roi != 0.0:
                    # Format filename for the cube
                    epsg = properties[CubeJson.DATA_EPSG].replace(CubeJson.EPSG_SEPARATOR, '')
                    # Extract int EPSG code
                    epsg_code = epsg.replace(CubeJson.EPSG_PREFIX, '')

                    # Include only specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_GENERATE) and \
                       epsg_code not in BatchVars.EPSG_TO_GENERATE:
                       logging.info(f'Skipping {epsg_code} which is not in {BatchVars.EPSG_TO_GENERATE}')
                       continue

                    # Exclude specific EPSG code(s) if specified
                    if len(BatchVars.EPSG_TO_EXCLUDE) and \
                       epsg_code in BatchVars.EPSG_TO_EXCLUDE:
                        continue

                    coords = properties[CubeJson.GEOMETRY_EPSG][CubeJson.COORDINATES][0]
                    x_bounds = Bounds([each[0] for each in coords])
                    y_bounds = Bounds([each[1] for each in coords])

                    mid_x = int((x_bounds.min + x_bounds.max)/2)
                    mid_y = int((y_bounds.min + y_bounds.max)/2)

                    # Get mid point to the nearest 50
                    logging.info(f"Mid point: x={mid_x} y={mid_y}")
                    mid_x = int(math.floor(mid_x/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)
                    mid_y = int(math.floor(mid_y/BatchVars.MID_POINT_RESOLUTION)*BatchVars.MID_POINT_RESOLUTION)
                    logging.info(f"Mid point at {BatchVars.MID_POINT_RESOLUTION}: x={mid_x} y={mid_y}")

                    # Convert to lon/lat coordinates to format s3 bucket path
                    # for the datacube
                    mid_lon_lat = itslive_utils.transform_coord(
                        epsg_code,
                        BatchVars.LON_LAT_PROJECTION,
                        mid_x, mid_y
                    )

                    if BatchVars.POLYGON_SHAPE and \
                       (not BatchVars.POLYGON_SHAPE.contains(geometry.Point(mid_lon_lat[0], mid_lon_lat[1]))):
                        logging.info(f"Skipping non-polygon point: {mid_lon_lat}")
                        # Provided polygon does not contain cube's center point
                        continue

                    bucket_dir = itslive_utils.point_to_prefix(mid_lon_lat[1], mid_lon_lat[0], cube_dir)

                    # cube_filename = f"{BatchVars.FILENAME_PREFIX}_{epsg}_G{self.grid_size_str}_X{mid_x}_Y{mid_y}.zarr"
                    cube_filename = os.path.basename(properties[CubeJson.URL])
                    logging.info(f'Cube name: {cube_filename}')

                    # Process specific datacubes only
                    if len(BatchVars.CUBES_TO_GENERATE) and cube_filename not in BatchVars.CUBES_TO_GENERATE:
                        logging.info(f"Skipping as not provided in BatchVars.CUBES_TO_GENERATE")
                        continue

                    # Format filename for the cube's composites
                    cube_s3 = properties[CubeJson.URL].replace(
                        BatchVars.HTTP_PREFIX,
                        BatchVars.AWS_PREFIX
                    )

                    # Check if cube exists in S3 bucket (should exist, just to be sure)
                    cube_exists = self.s3.ls(cube_s3)
                    if len(cube_exists) == 0:
                        logging.info(f"Datacube {cube_s3} does not exist, skipping.")
                        continue

                    composite_s3 = cube_s3.replace(cube_dir, composite_dir)

                    composite_s3 = composite_s3.replace(FilenamePrefix.Datacube, FilenamePrefix.Composites)
                    logging.info(f'Composite file: {composite_s3}')

                    # Check if composite exists in S3 bucket (should exist, just to be sure)
                    composite_exists = self.s3.ls(composite_s3)
                    if len(composite_exists) == 0:
                        logging.info(f"Datacube composite {composite_s3} does not exist, skipping.")
                        continue

                    # Update EPSG: Y: X: composite_s3_path nested dictionary
                    epsg_dict = self.composites.setdefault(epsg_code, {})
                    y_dict = epsg_dict.setdefault(mid_y, {})
                    y_dict[mid_x] = composite_s3

                    # Format cube composites filename:
                    # s3://its-live-data/composites/annual/v02/N60W130/ITS_LIVE_vel_annual_EPSG3413_G0120_X-3250000_Y250000.zarr
                    logging.info(f'Cube composite name: {composite_s3}')

                    # if self.is_dry_run is False:
                    num_processed += 1

            logging.info(f"Number of collected composites: {num_processed}")
            logging.info(f'Collected composites: {json.dumps(self.composites, indent=4)}')

    def make_mosaics(self, s3_bucket: str, mosaics_dir: str):
        """
        Build annual mosaics from collected datacube composites.

        mosaics_dir: Directory path within S3 bucket that stores datacubes' mosaics.
        """
        # xarray.Dataset's objects for opened Zarr composites
        self.composites_ds = {}

        for each_epsg in self.composites.keys():
            logging.info(f'Opening annual composites for EPSG={each_epsg}')
            epsg = self.composites[each_epsg]

            # For each y from sorted list of composites center's y's:
            for each_mid_y in sorted(epsg.keys()):
                all_y = epsg[each_mid_y]
                # For each x from sorted list of composites center's x:
                for each_mid_x in sorted(all_y.keys()):
                    composite_s3_path = all_y[each_mid_x]

                    # TODO: Should preserve s3_store or just reopen the composite when needed?
                    s3_store = s3fs.S3Map(root=composite_s3_path, s3=self.s3, check=False)
                    ds_from_zarr = xr.open_dataset(s3_store, decode_timedelta=False, engine='zarr', consolidated=True)

                    ds_time = [t.astype('M8[ms]').astype('O') for t in ds_from_zarr.time.values]
                    # Store open cube's composites and corresponding metadata
                    self.composites_ds[composite_s3_path] = ITSLiveAnnualMosaics.Composite(
                        ITSLiveAnnualMosaics.CompositeS3(ds_from_zarr, s3_store),
                        ds_from_zarr.x.values,
                        ds_from_zarr.y.values,
                        [each_t.year for each_t in ds_time],
                        ds_from_zarr.sensor.values
                    )

                    # Collect coordinates
                    self.x_coords.append(ds_from_zarr.x.values)
                    self.y_coords.append(ds_from_zarr.y.values)
                    self.time_coords.append(ds_time)
                    self.sensor_coords.append(ds_from_zarr.sensor.values)

        # Create one large dataset per each year
        self.time_coords = sorted(list(set(np.concatenate(self.time_coords))))
        self.x_coords = sorted(list(set(np.concatenate(self.x_coords))))
        self.y_coords = sorted(list(set(np.concatenate(self.y_coords))))
        # y coordinate in EPSG is always in ascending order
        self.y_coords = np.flip(self.y_coords)
        self.sensor_coords = sorted(list(set(np.concatenate(self.sensor_coords))))

        composites_urls = sorted(list(self.composites_ds.keys()))
        # Use first composite to "collect" global attributes
        first_ds = self.composites_ds[composites_urls[0]].s3.ds

        # Create mapping data variable
        self.mapping = xr.DataArray(
            data='',
            attrs=first_ds[DataVars.MAPPING].attrs,
            coords={},
            dims=[]
        )

        # Set GeoTransform to correspond to the mosaic's tile:
        # format GeoTransform
        x_cell = self.x_coords[1] - self.x_coords[0]
        y_cell = self.y_coords[1] - self.y_coords[0]

        # TODO: Check cell size for all mosaics against target cell size?
        if ITSLiveAnnualMosaics.CELL_SIZE != x_cell and \
           ITSLiveAnnualMosaics.CELL_SIZE != np.abs(y_cell):
           raise RuntimeError(f'Provided mosaics grid cell size {ITSLiveAnnualMosaics.CELL_SIZE} does not correspond to the cell size of composites: x={x_cell} or y={np.abs(y_cell)}')

        # :GeoTransform = "-3300007.5 120.0 0 300007.5 0 -120.0";
        new_geo_transform_str = f"{self.x_coords[0] - x_cell/2.0} {x_cell} 0 {self.y_coords[0] - y_cell/2.0} 0 {y_cell}"
        logging.info(f'Setting mapping.GeoTransform: {new_geo_transform_str}')
        self.mapping.attrs['GeoTransform'] = new_geo_transform_str

        # Create annual mosaics
        for each_year in tqdm(self.time_coords, ascii=True, desc=f'Creating annual mosaics for {ITSLiveAnnualMosaics.REGION}'):
            self.create_annual_mosaics(first_ds, each_year, s3_bucket, mosaics_dir)

        # TODO: create summary mosaic (to store all 2d data variables from all composites)

    def create_annual_mosaics(self, first_ds, year_date, s3_bucket, mosaics_dir):
        """
        Create mosaics for a specific year and store it to NetCDF format file in
        S3 bucket if provided.

        first_ds: xarray.Dataset object that represents any (first) composite dataset.
                  It's used to collect global attributes that are applicable to the
                  mosaics.
        year_date: Datetime object for the mosaic to create.
        s3_bucket: AWS S3 bucket to place result mosaics file in.
        mosaics_dir: AWS S3 directory to place mosaics in.
        """
        # Dataset to represent annual mosaic
        ds = xr.Dataset(
            coords = {
                Coords.X: (
                    Coords.X,
                    self.x_coords,
                    first_ds.x.attrs
                ),
                Coords.Y: (
                    Coords.Y,
                    self.y_coords,
                    first_ds.y.attrs
                )
            },
            attrs = {
                'author': 'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)',
                CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE: first_ds.attrs[CompOutputFormat.DATACUBE_AUTORIFT_PARAMETER_FILE],
                'institution': 'NASA Jet Propulsion Laboratory (JPL), California Institute of Technology',
                'region': ITSLiveAnnualMosaics.REGION,
                'year': year_date.strftime('%d-%b-%Y')
            }
        )

        ds.attrs[CompOutputFormat.GDAL_AREA_OR_POINT] = "Area"

        ds[DataVars.MAPPING] = self.mapping

        # Each composite dataset has global attributes
        # // global attributes:
        # :GDAL_AREA_OR_POINT = "Area";
        # :composites_software_version = "1.0";
        # :datacube_autoRIFT_parameter_file = "http://its-live-data.s3.amazonaws.com/autorift_parameters/v001/autorift_landice_0120m.shp";
        # :datacube_software_version = "1.0";
        # :date_created = "25-Mar-2022 05:28:31";
        # :date_updated = "25-Mar-2022 05:28:31";
        # :datecube_created = "03-Mar-2022 05:06:36";
        # :datecube_s3 = "s3://its-live-data/datacubes/v02/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr";
        # :datecube_updated = "03-Mar-2022 05:06:36";
        # :datecube_url = "https://its-live-data.s3.amazonaws.com/datacubes/v02/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y250000.zarr";

        # TODO: if you can get all of the bounding polygons then dissolve using geopandas: https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html
        # :geo_polygon = "[[-138.46822925891712, 60.147756216807174], [-138.49463892725333, 60.36348555837141], [-138.52145337692224, 60.57941645681651], [-138.54868197390445, 60.79554754863715], [-138.57633437499737, 61.01187745791771], [-139.02199017702011, 60.9975093905528], [-139.46715906138928, 60.981460800912316], [-139.911788172506, 60.96373494822363], [-140.35582504285517, 60.944335424185965], [-140.31454566994478, 60.72858919085008], [-140.27389595735178, 60.513033250506986], [-140.23386164721643, 60.29766916958488], [-140.19442890773482, 60.082498496408085], [-139.7636416907262, 60.101242884393535], [-139.33231398318853, 60.11836944082792], [-138.90049374238188, 60.13387487895878], [-138.46822925891712, 60.147756216807174]]";
        # :institution = "NASA Jet Propulsion Laboratory (JPL), California Institute of Technology";
        # :latitude = "60.55";
        # :longitude = "-139.40";

        # TODO: if you can get all of the bounding polygons then dissolve using geopandas: https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html
        # :proj_polygon = "[[-3300000, 200000], [-3275000.0, 200000.0], [-3250000.0, 200000.0], [-3225000.0, 200000.0], [-3200000, 200000], [-3200000.0, 225000.0], [-3200000.0, 250000.0], [-3200000.0, 275000.0], [-3200000, 300000], [-3225000.0, 300000.0], [-3250000.0, 300000.0], [-3275000.0, 300000.0], [-3300000, 300000], [-3300000.0, 275000.0], [-3300000.0, 250000.0], [-3300000.0, 225000.0], [-3300000, 200000]]";
        # :projection = "3413";
        # :s3 = "";
        # :title = "ITS_LIVE annual composites of image_pair velocities";
        # :url = "";
        # :sensors_labels = "Band 1: S1A_S1B; Band 2: S2A_S2B; Band 3: L8_L9";

        # TODO: Create lists of attributes that correspond to multiple composites that
        # contribute to each of the annual mosaic
        all_attributes = {
            CompOutputFormat.COMPOSITES_SOFTWARE_VERSION: [],
            CompOutputFormat.DATACUBE_SOFTWARE_VERSION: [],
            CompOutputFormat.DATE_CREATED: [],
            CompOutputFormat.DATE_UPDATED: [],
            CompOutputFormat.DATECUBE_CREATED: [],
            CompOutputFormat.DATECUBE_S3: [],
            CompOutputFormat.DATECUBE_UPDATED: [],
            CompOutputFormat.DATECUBE_URL: [],
            CompOutputFormat.GEO_POLYGON: [],
            CompOutputFormat.PROJ_POLYGON: []
        }

        # Concatenate data for each data variable that has time (year value) dimension
        for each_file, each_ds in self.composites_ds.items():
            if year_date.year in each_ds.time:
                # Composites have data for the year
                year_index = each_ds.time.index(year_date.year)

                for each_var in [
                    CompDataVars.COUNT,
                    DataVars.VX,
                    DataVars.VY,
                    DataVars.V,
                    CompDataVars.V_ERROR,
                    CompDataVars.VX_ERROR,
                    CompDataVars.VY_ERROR,
                    ]:

                    if each_var not in ds:
                        ds[each_var] = each_ds.s3.ds[each_var][year_index].load()
                        # Each data variable in original composites has mapping attribute set
                        # already, so it will inherit attributes from original dataset
                        # ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

                    else:
                        ds[each_var].loc[dict(x=each_ds.x, y=each_ds.y)] = each_ds.s3.ds[each_var][year_index].load()

                # Update attributes
                for each_attr in all_attributes.keys():
                    all_attributes[each_attr].append(each_ds.s3.ds.attrs[each_attr])

            else:
                logging.info(f'{each_file} does not have data for {year_date.year} year, skipping.')

        # Set "mapping" attribute
        for each_var in [
            CompDataVars.COUNT,
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.V_ERROR,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            ]:

            ds[each_var].attrs[DataVars.GRID_MAPPING] = DataVars.MAPPING

        # Set cumulative attributes for the mosaic
        for each_key, each_value in all_attributes.items():
            key = each_key

            if each_key in MosaicsOutputFormat.ATTR_MAP:
                # Get key name for the mosaic's attribute
                key = MosaicsOutputFormat.ATTR_MAP[each_key]

            # if each_key in [CompOutputFormat.GEO_POLYGON, CompOutputFormat.PROJ_POLYGON]:
                # Join polygons
            # else:
            value = json.dumps(each_value)
            ds.attrs[key] = value

        # TODO: Format filename for the mosaics
        mosaics_filename = f'{FilenamePrefix.Mosaics}_EPSG{self.epsg_str}_G{self.grid_size_str}_{ITSLiveAnnualMosaics.REGION}_{year_date.year}.nc'

        # Write mosaic to NetCDF format file
        ITSLiveAnnualMosaics.to_netcdf(ds, s3_bucket, mosaics_dir, mosaics_filename)

    @staticmethod
    def to_netcdf(ds: xr.Dataset, s3_bucket: str, bucket_dir: str, filename: str):
        """
        Store datacube annual mosaics to NetCDF store.
        """
        target_file = os.path.join(s3_bucket, bucket_dir, filename)
        logging.info(f'Writing mosaics to {target_file}')

        # Set encoding
        encoding_settings = {}
        for each in [Coords.X, Coords.Y]:
            encoding_settings.setdefault(each, {}).update({DataVars.FILL_VALUE_ATTR: None})

        # Settings for "float" data types
        for each in [
            DataVars.VX,
            DataVars.VY,
            DataVars.V,
            CompDataVars.VX_ERROR,
            CompDataVars.VY_ERROR,
            CompDataVars.V_ERROR,
            CompDataVars.COUNT
            ]:
            encoding_settings.setdefault(each, {}).update({
                DataVars.FILL_VALUE_ATTR: DataVars.MISSING_VALUE,
                # 'dtype': 'float',
                'dtype': np.float,
                "zlib": True, "complevel": 2, "shuffle": True
            })

        if self.dryrun is False:
            # Write locally
            ds.to_netcdf(f'{filename}', engine=ITSLiveAnnualMosaics.NC_ENGINE, encoding=encoding_settings)

            # TODO: Copy to S3 bucket

def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """

    # Command-line arguments parser
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     epilog=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        default=None,
        help="GeoJson file that stores cube polygon definitions [%(default)s]."
    )
    parser.add_argument(
        '--processCubesWithinPolygon',
        type=str,
        action='store',
        default=None,
        help="GeoJSON file that stores polygon the cubes centers should belong to [%(default)s]."
    )
    parser.add_argument(
        '-b', '--bucket',
        type=str,
        action='store',
        default='s3://its-live-data',
        help="S3 bucket to store datacubes, composites and mosaics [%(default)s]"
    )
    parser.add_argument(
        '-u', '--urlPath',
        type=str,
        action='store',
        default='http://its-live-data.s3.amazonaws.com',
        help="URL for the store in S3 bucket (to provide for easier download option) [%(default)s]"
    )
    parser.add_argument(
        '-d', '--cubeDir',
        type=str,
        action='store',
        default='datacubes/v02',
        help="S3 directory with datacubes [%(default)s]"
    )
    parser.add_argument(
        '-s', '--compositesDir',
        type=str,
        action='store',
        default='composites/annual/v02',
        help="Destination S3 directory with composites [%(default)s]"
    )
    parser.add_argument(
        '-m', '--mosaicsDir',
        type=str,
        action='store',
        default='mosaics/annual/v02',
        help="Destination S3 directory to store mosaics to [%(default)s]"
    )
    parser.add_argument(
        '-r', '--region',
        type=str,
        action='store',
        required=True,
        help="Region to create annual mosaics for"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to process [%(default)s]"
    )
    parser.add_argument(
        '--excludeEPSG',
        type=str,
        action='store',
        default=None,
        help="JSON list of EPSG codes to exclude from the datacube processing [%(default)s]"
    )
    parser.add_argument(
        '--mosaicsEpsgCode',
        type=int,
        action='store',
        default=None,
        help="Target EPSG code for annual mosaics [%(default)s]"
    )
    parser.add_argument(
        '-g', '--gridCellSize',
        type=int,
        default=120,
        help="Grid cell size of input ITS_LIVE datacube composites [%(default)d]."
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        default=False,
        help='Dry run, do not actually submit any AWS Batch jobs'
    )

    # One of --processCubes or --processCubesFile options is allowed for the datacube names
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of datacubes filenames to process [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of filenames for datacubes to process [%(default)s]."
    )
    parser.add_argument(
        '-n', '--engine',
        type=str,
        required=False,
        default='h5netcdf',
        help="NetCDF engine to use to store NetCDF data to the file [%(default)s]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")

    BatchVars.HTTP_PREFIX = args.urlPath
    BatchVars.AWS_PREFIX = args.bucket
    ITSLiveAnnualMosaics.REGION = args.region
    ITSLiveAnnualMosaics.CELL_SIZE = args.gridCellSize
    ITSLiveAnnualMosaics.NC_ENGINE = args.engine

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        BatchVars.EPSG_TO_GENERATE = epsg_codes

    epsg_codes = list(map(str, json.loads(args.excludeEPSG))) if args.excludeEPSG is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes to exclude: {epsg_codes}")
        BatchVars.EPSG_TO_EXCLUDE = epsg_codes

    # Make sure there is no overlap in EPSG_TO_GENERATE and EPSG_TO_EXCLUDE
    diff = set(BatchVars.EPSG_TO_GENERATE).intersection(BatchVars.EPSG_TO_EXCLUDE)
    if len(diff):
        raise RuntimeError(f"The same code is specified for BatchVars.EPSG_TO_EXCLUDE={BatchVars.EPSG_TO_EXCLUDE} and BatchVars.EPSG_TO_GENERATE={BatchVars.EPSG_TO_GENERATE}")

    if args.processCubesFile:
        # Check for this option first as another mutually exclusive option has a default value
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubesFile.read_text())
        # Replace each path by the datacube basename
        BatchVars.CUBES_TO_GENERATE = [os.path.basename(each) for each in BatchVars.CUBES_TO_GENERATE if len(each)]
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

    elif args.processCubes:
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubes)
        if len(BatchVars.CUBES_TO_GENERATE):
            logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubes}: {BatchVars.CUBES_TO_GENERATE}")

    if args.processCubesWithinPolygon:
        with open(args.processCubesWithinPolygon, 'r') as fhandle:
            shape_file = json.load(fhandle)
            logging.info(f'Reading region polygon the datacube\'s central point should fall into: {args.processCubesWithinPolygon}')

            shapefile_coords = shape_file[CubeJson.FEATURES][0]['geometry']['coordinates']
            logging.info(f'Got polygon coordinates: {shapefile_coords}')
            line = geometry.LineString(shapefile_coords[0][0])
            BatchVars.POLYGON_SHAPE = geometry.Polygon(line)

            logging.info(f'Set polygon: {BatchVars.POLYGON_SHAPE}')

    return args

if __name__ == '__main__':
    import argparse
    import warnings
    import shutil
    import subprocess
    import sys
    from urllib.parse import urlparse

    warnings.filterwarnings('ignore')

    # Command-line arguments parser
    args = parse_args()

    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Command arguments: {args}")

    # Set static data for computation
    mosaics = ITSLiveAnnualMosaics(args.mosaicsEpsgCode, args.gridCellSize, args.dryrun)
    mosaics.create(
        args.cubeDefinitionFile,
        args.bucket,
        args.cubeDir,
        args.compositesDir,
        args.mosaicsDir
    )

    # # Copy generated composites to the S3 bucket if provided
    # if os.path.exists(args.outputStore) and len(args.targetBucket):
    #     try:
    #         # Use "subprocess" as s3fs.S3FileSystem leaves unclosed connections
    #         # resulting in as many error messages as there are files in Zarr store
    #         # to copy
    #         command_line = [
    #             "awsv2", "s3", "cp", "--recursive",
    #             args.outputStore,
    #             os.path.join(args.targetBucket, os.path.basename(args.outputStore)),
    #             "--acl", "bucket-owner-full-control"
    #         ]
    #
    #         logging.info(' '.join(command_line))
    #
    #         file_is_copied = False
    #         num_retries = 0
    #         command_return = None
    #         env_copy = os.environ.copy()
    #
    #         while not file_is_copied and num_retries < ITSCube.NUM_AWS_COPY_RETRIES:
    #             logging.info(f"Attempt #{num_retries+1} to copy {args.outputStore} to {args.targetBucket}")
    #
    #             command_return = subprocess.run(
    #                 command_line,
    #                 env=env_copy,
    #                 check=False,
    #                 stdout=subprocess.PIPE,
    #                 stderr=subprocess.STDOUT
    #             )
    #
    #             if command_return.returncode != 0:
    #                 # Report the whole stdout stream as one logging message
    #                 logging.warning(f"Failed to copy {args.outputStore} to {args.targetBucket} with returncode={command_return.returncode}: {command_return.stdout}")
    #
    #                 num_retries += 1
    #                 # If failed due to AWS SlowDown error, retry
    #                 if num_retries != ITSCube.NUM_AWS_COPY_RETRIES and \
    #                    ITSCube.AWS_SLOW_DOWN_ERROR in command_return.stdout.decode('utf-8'):
    #                     # Sleep if it's not a last attempt to copy
    #                     time.sleep(ITSCube.AWS_COPY_SLEEP_SECONDS)
    #
    #                 else:
    #                     # Don't retry otherwise
    #                     num_retries = ITSCube.NUM_AWS_COPY_RETRIES
    #
    #             else:
    #                 file_is_copied = True
    #
    #         if not file_is_copied:
    #             raise RuntimeError(f"Failed to copy {args.outputStore} to {args.targetBucket} with command.returncode={command_return.returncode}")
    #
    #     finally:
    #         # Remove locally written Zarr store.
    #         # This is to eliminate out of disk space failures when the same EC2 instance is
    #         # being re-used by muliple Batch jobs.
    #         if os.path.exists(args.outputStore):
    #             logging.info(f"Removing local copy of {args.outputStore}")
    #             shutil.rmtree(args.outputStore)
