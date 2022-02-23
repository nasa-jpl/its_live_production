#!/usr/bin/env python
"""
Add data version string to all entries in catalog geojson file.

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
        default="imgpr_v02*",
        help='Glob pattern to search for catalog geojson files.'
    )
    parser.add_argument(
        '-v', '--version', type=str,
        default="v02",
        help='Version string to add to the catalog geojson files.'
    )

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logging.info(f"Command-line args: {args}")
    logging.info(f"Conda environment: {os.environ['CONDA_DEFAULT_ENV']}")

    Path(args.outputDir).mkdir(parents=True, exist_ok=True)

    for each_file in glob.glob(os.path.join(args.inputDir, args.glob)):
        logging.info(f"Processing {each_file}")

        ds = json.loads(Path(each_file).read_text())

        # Set version string for each granule
        for each_granule in ds['features']:
            each_granule['properties']['version'] = args.version

        output_file = os.path.join(args.outputDir, os.path.basename(each_file))
        logging.info(f"Writing {output_file}")
        with open(output_file, 'w') as fh:
            json.dump(ds, fh)


if __name__ == '__main__':
    main()
    logging.info("Done.")
