#!/usr/bin/env python
"""
Replace directory path string with new value for all entries in the list of granules
(skipped or used).

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
        default="used_granules*.json",
        help='Glob pattern to search for used or skipped granules files.'
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
        logging.info(f"There are {len(ds)} granules paths to fix")

        # Set version string for each granule
        for each_index in range(len(ds)):
            ds[each_index] = ds[each_index].replace(args.inputPath, args.outputPath)

        output_file = os.path.join(args.outputDir, os.path.basename(each_file))
        logging.info(f"Writing {output_file}")
        with open(output_file, 'w') as fh:
            json.dump(ds, fh)


if __name__ == '__main__':
    main()
    logging.info("Done.")
