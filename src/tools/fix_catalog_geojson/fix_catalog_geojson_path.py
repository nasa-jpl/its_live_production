#!/usr/bin/env python
"""
Replace directory path string with new value for all entries in catalog geojson file:

"properties": {"filename": "LC08_L1GT_139041_20130731_20200912_02_T2_X_LC08_L1TP_139041_20130917_20200912_02_T1_G0120V02_P000.nc",
               "directory": "velocity_image_pair/landsat8/v02/N20E080", ...}


Authors: Masha Liukis
"""
import argparse
import json
import glob
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--inputDir', type=str,
        default=None,
        help='Directory that stores catalog geojson files.'
    )
    parser.add_argument(
        '-o', '--outputDir', type=str,
        default=None,
        help='Directory that stores catalog geojson files.'
    )
    parser.add_argument(
        '-g', '--glob', type=str,
        default="imgpair_v02*.json",
        help='Glob pattern to search for catalog geojson files.'
    )
    parser.add_argument(
        '--inputPath', type=str,
        default="landsat8",
        help='Original directory token to replace in all catalog geojson files.'
    )
    parser.add_argument(
        '--outputPath', type=str,
        default="landsatOLI",
        help='Directory token to replace with in all catalog geojson files.'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Command-line args: {args}")
    logging.info(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")

    Path(args.outputDir).mkdir(parents=True, exist_ok=True)

    glob_pattern = os.path.join(args.inputDir, args.glob)
    logging.info(f'Searching for input files with pattern: {glob_pattern}')

    for each_file in glob.glob(glob_pattern):
        logging.info(f"Processing {each_file}: replace '{args.inputPath}' with '{args.outputPath}'")

        ds = json.loads(Path(each_file).read_text())

        # Set version string for each granule
        for each_granule in ds['features']:
            each_granule['properties']['directory'] = each_granule['properties']['directory'].replace(args.inputPath, args.outputPath)

        output_file = os.path.join(args.outputDir, os.path.basename(each_file))
        logging.info(f"Writing {output_file}")
        with open(output_file, 'w') as fh:
            json.dump(ds, fh)


if __name__ == '__main__':
    main()
    logging.info("Done.")
