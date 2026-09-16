"""
Script to add existing datacube S3 locations to the global datacube
definition GeoJson file.

Each catalog entry already carries an `icechunk_filename` property
(precomputed by define_cube_polygons_chunk_aligned.py using the same
midpoint/rounding convention as run_virtual_cube_batch.py). This script:

- Derives the deep-copy (Zarr) cube's filename from `icechunk_filename` by
  swapping the '.icechunk' extension for '.zarr', prepends the provided
  deep-copy cubes location, and checks whether that Zarr v3 store actually
  exists there. If found, records its HTTPS URL, granule count (read from
  the store's root zarr.json -- Zarr v3 embeds consolidated metadata there
  instead of a separate .zmetadata file), and sets the existing
  `datacube_exist` flag.
- Prepends the provided icechunk repo location to `icechunk_filename` and
  checks whether that path exists in S3. If found, records its S3 URL and
  sets the `icechunk_exist` flag.

It accepts a geojson file with datacube definitions and the two S3 locations
(deep-copy cubes directory, icechunk repos directory).
"""
import json
import logging
import s3fs

from itslive_mosaics_types import GeoJsonVars
from utils import File


def _join_s3_url(location: str, filename: str) -> str:
    """Join a s3:// directory URL (with or without trailing slash) and a
    filename into a full s3:// URL.
    """
    return location.rstrip('/') + '/' + filename


class DataCubeGlobalDefinition:
    """
    Class to manage global datacube definition GeoJson file: detects which
    cubes already exist (deep-copy Zarr and/or virtual icechunk) at the
    provided S3 locations and records their URLs.
    """
    # List of EPSG codes to generate datacubes for. If this list is empty,
    # then generate all ROI!=0 datacubes.
    EPSG_TO_UPDATE = []

    # Flag to disable reduced catalog geojson
    DISABLE_REDUCED_CATALOG = False

    # List of icechunk_filename values to include into catalog.
    CUBES_TO_INCLUDE = []

    HTTP_PREFIX = 'https://its-live-data.s3.amazonaws.com'
    S3_PREFIX = 's3://its-live-data'

    # Flat S3 directory that holds deep-copy (Zarr) datacubes, e.g.
    # 's3://its-live-data/test-space/virtual-cubes/zarr/forMark/'.
    ZARR_CUBES_LOCATION = None

    # Flat S3 directory that holds virtual (icechunk) datacube repos, e.g.
    # 's3://its-live-data/datacubes/spatial/v2.2/'.
    ICECHUNK_CUBES_LOCATION = None

    # If True, the reduced catalog only keeps cubes for which the deep-copy
    # (Zarr) cube exists, ignoring icechunk existence. If False (default),
    # a cube is kept if either store was found.
    REDUCED_CATALOG_ZARR_ONLY = False

    def __init__(self):
        """
        Initialize object.
        """
        self.s3 = s3fs.S3FileSystem(anon=True)

    def _check_zarr_cube(self, icechunk_filename: str):
        """Check whether the deep-copy Zarr cube derived from
        `icechunk_filename` exists at ZARR_CUBES_LOCATION.

        Returns
        -------
        tuple of (bool, str or None, int or None)
            (exists, https URL if it exists, granule count if it exists)
        """
        zarr_filename = icechunk_filename.replace(File.ext.icechunk, File.ext.zarr)
        zarr_s3_url = _join_s3_url(
            DataCubeGlobalDefinition.ZARR_CUBES_LOCATION, zarr_filename
        )

        if not self.s3.exists(zarr_s3_url):
            return False, None, None

        zarr_https_url = zarr_s3_url.replace(
            DataCubeGlobalDefinition.S3_PREFIX,
            DataCubeGlobalDefinition.HTTP_PREFIX
        )

        # Zarr v3 stores embed consolidated metadata directly in the root
        # zarr.json (no separate .zmetadata file) -- read granule count from
        # the 'mid_date' array's shape there.
        granule_count = None
        with self.s3.open(f"{zarr_s3_url}/zarr.json", 'r') as fh:
            root_meta = json.load(fh)

        mid_date_meta = root_meta.get('consolidated_metadata', {}) \
            .get('metadata', {}).get('mid_date')
        if mid_date_meta:
            granule_count = mid_date_meta['shape'][0]
        else:
            logging.warning(
                f"{zarr_s3_url}/zarr.json has no consolidated 'mid_date' "
                "metadata; cannot determine granule count"
            )

        return True, zarr_https_url, granule_count

    def _check_icechunk_cube(self, icechunk_filename: str):
        """Check whether the virtual icechunk repo named `icechunk_filename`
        exists at ICECHUNK_CUBES_LOCATION, via a plain S3 existence check
        (no icechunk repo open needed).

        Returns
        -------
        tuple of (bool, str or None)
            (exists, S3 URL if it exists)
        """
        icechunk_s3_url = _join_s3_url(
            DataCubeGlobalDefinition.ICECHUNK_CUBES_LOCATION, icechunk_filename
        )

        if not self.s3.exists(icechunk_s3_url):
            return False, None

        return True, icechunk_s3_url

    def __call__(self, cube_file: str, output_file: str):
        """
        Detect existing datacubes and write their URLs into the datacube
        definition GeoJson, then write result to provided output file.
        """
        # List of datacubes that had their info updated
        num_zarr_cubes = 0
        num_icechunk_cubes = 0

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            logging.info(f'Total number of datacubes: {len(cubes["features"])}')

            # If need to create reduced catalog of datacubes (only cubes for
            # which at least one of the two stores was found)
            output_cubes = cubes
            if not DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG:
                output_cubes = dict(cubes)
                output_cubes[GeoJsonVars.features] = []

            for each_cube in cubes[GeoJsonVars.features]:
                properties = each_cube[GeoJsonVars.properties]

                roi = properties[GeoJsonVars.roi_percent_coverage]

                # Default: neither store exists for this cube
                properties[GeoJsonVars.exist_flag] = 0
                properties[GeoJsonVars.icechunk_exist_flag] = 0

                if roi == 0.0:
                    continue

                icechunk_filename = properties.get(GeoJsonVars.icechunk_filename)
                if not icechunk_filename:
                    logging.warning(
                        f"Cube {properties.get('cube_id')} has no "
                        f"'{GeoJsonVars.icechunk_filename}' property; skipping"
                    )
                    continue

                epsg_code = str(properties[GeoJsonVars.epsg])
                if len(DataCubeGlobalDefinition.EPSG_TO_UPDATE) and \
                        epsg_code not in DataCubeGlobalDefinition.EPSG_TO_UPDATE:
                    continue

                if len(DataCubeGlobalDefinition.CUBES_TO_INCLUDE) and \
                        icechunk_filename not in DataCubeGlobalDefinition.CUBES_TO_INCLUDE:
                    logging.info(f'Skipping cube: {icechunk_filename}')
                    continue

                found_any = False

                zarr_exists, zarr_url, granule_count = self._check_zarr_cube(icechunk_filename)
                if zarr_exists:
                    logging.info(f'Found deep-copy cube: {zarr_url}')
                    properties[GeoJsonVars.url] = zarr_url
                    properties[GeoJsonVars.exist_flag] = 1
                    if granule_count is not None:
                        properties[GeoJsonVars.granule_count] = granule_count
                    num_zarr_cubes += 1
                    found_any = True

                icechunk_exists, icechunk_url = self._check_icechunk_cube(icechunk_filename)
                if icechunk_exists:
                    logging.info(f'Found icechunk cube: {icechunk_url}')
                    properties[GeoJsonVars.icechunk_url] = icechunk_url
                    properties[GeoJsonVars.icechunk_exist_flag] = 1
                    num_icechunk_cubes += 1
                    found_any = True

                keep_in_reduced = zarr_exists \
                    if DataCubeGlobalDefinition.REDUCED_CATALOG_ZARR_ONLY \
                    else found_any

                if keep_in_reduced and not DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG:
                    output_cubes[GeoJsonVars.features].append(each_cube)

            logging.info(f"Number of found deep-copy cubes: {num_zarr_cubes}")
            logging.info(f"Number of found icechunk cubes: {num_icechunk_cubes}")

            logging.info(f"Writing updated datacube info to the {output_file}...")
            with open(output_file, 'w') as output_fhandle:
                json.dump(output_cubes, output_fhandle, indent=4)

            return


def main(
    cube_definition_file: str,
    zarr_cubes_location: str,
    icechunk_cubes_location: str,
    output_file: str
):
    """
    Driver to update general datacube definition file with existing
    datacube S3 locations.
    """
    DataCubeGlobalDefinition.ZARR_CUBES_LOCATION = zarr_cubes_location
    DataCubeGlobalDefinition.ICECHUNK_CUBES_LOCATION = icechunk_cubes_location

    update_urls = DataCubeGlobalDefinition()
    update_urls(cube_definition_file, output_file)


if __name__ == '__main__':
    import argparse
    import warnings
    import sys
    import os
    from pathlib import Path
    warnings.filterwarnings('ignore')

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        required=True,
        help="GeoJson file that stores cube polygon definitions, with each "
                "feature's properties already carrying 'icechunk_filename' "
                "(see define_cube_polygons_chunk_aligned.py)."
    )
    parser.add_argument(
        '--zarrCubesLocation',
        type=str,
        action='store',
        required=True,
        help="Flat S3 directory that holds deep-copy (Zarr) datacubes, e.g. "
                "'s3://its-live-data/test-space/virtual-cubes/zarr/forMark/'."
    )
    parser.add_argument(
        '--icechunkCubesLocation',
        type=str,
        action='store',
        default='s3://its-live-data/datacubes/spatial/v2.2',
        help="Flat S3 directory that holds virtual (icechunk) datacube "
                "repos [%(default)s]."
    )
    parser.add_argument(
        '-o', '--outputFile',
        type=str,
        action='store',
        default=None,
        help="File to capture updated general datacube definition information "
                "[%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes "
                "to generate [%(default)s]"
    )
    parser.add_argument(
        '--disableReducedCatalog',
        action='store_true',
        default=False,
        help="Flag to disable reduced (list only the cubes for which a "
                "store exists) catalog generation. Default is to generate "
                "reduced catalog."
    )
    parser.add_argument(
        '--reducedCatalogZarrOnly',
        action='store_true',
        default=False,
        help="When generating the reduced catalog, keep only cubes for "
                "which the deep-copy (Zarr) cube exists, ignoring icechunk "
                "existence. Default is to keep a cube if either store was "
                "found. Ignored if --disableReducedCatalog is set."
    )
    parser.add_argument(
        '--includeCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains a list of icechunk_filename values for "
                "datacubes to include into catalog [%(default)s]."
    )

    args = parser.parse_args()

    logging.info(f"Command-line arguments: {sys.argv}")

    epsg_codes = list(map(str, json.loads(args.epsgCode))) \
        if args.epsgCode is not None else None

    if epsg_codes and len(epsg_codes):
        logging.info(
            f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes"
        )
        DataCubeGlobalDefinition.EPSG_TO_UPDATE = epsg_codes

    DataCubeGlobalDefinition.DISABLE_REDUCED_CATALOG = args.disableReducedCatalog
    DataCubeGlobalDefinition.REDUCED_CATALOG_ZARR_ONLY = args.reducedCatalogZarrOnly

    if args.includeCubesFile is not None:
        DataCubeGlobalDefinition.CUBES_TO_INCLUDE = [
            os.path.basename(each) for each in
            args.includeCubesFile.read_text().split('\n') if len(each)
        ]

    if len(DataCubeGlobalDefinition.CUBES_TO_INCLUDE):
        logging.info(
            f"Number of datacubes for catalog: "
            f"{len(DataCubeGlobalDefinition.CUBES_TO_INCLUDE)}"
        )

    main(
        args.cubeDefinitionFile,
        args.zarrCubesLocation,
        args.icechunkCubesLocation,
        args.outputFile
    )

    logging.info("Done")
