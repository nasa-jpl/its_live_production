#!/usr/bin/env python3
"""
Script to compare two cube definition JSON files and identify cubes present
in one but missing in another. The script identifies cubes by their
geometry_epsg coordinates which represent the standardized cube boundaries.
"""
import json
import argparse
from typing import Set, Dict, List, Tuple

from itscube_types import CubeJson


def load_json_file(filepath: str) -> Dict:
   """Load and parse JSON file."""
   try:
      with open(filepath, 'r', encoding='utf-8') as f:
         return json.load(f)

   except FileNotFoundError:
      raise FileNotFoundError(f"File not found: {filepath}")

   except json.JSONDecodeError as e:
      raise ValueError(f"Invalid JSON in file {filepath}: {e}")


def get_cube_info(data: Dict) -> Dict[Tuple, Dict]:
   """
   Extract cube coordinates and associated metadata.
   Returns a dictionary mapping coordinate tuples to feature data.
   """
   # Stores cubes by their EPSG code and coordinate tuples
   cube_info = {}

   if 'features' not in data:
      raise ValueError("Invalid file format: missing 'features' key")

   for i, feature in enumerate(data['features']):
      try:
         properties = feature['properties']

         # Use only cubes with roi_percent_coverage > 0
         if properties.get('roi_percent_coverage', 0) <= 0:
            continue

         # Use epsg code and geometry_epsg coordinates as the cube identifier
         epsg_coords = properties['geometry_epsg']['coordinates'][0]
         coord_tuple = tuple(tuple(point) for point in epsg_coords)

         epsg_code = None
         if CubeJson.EPSG in properties:
            epsg_code = properties[CubeJson.EPSG]

         elif CubeJson.DATA_EPSG in properties:
            epsg_code = properties[CubeJson.DATA_EPSG].replace(CubeJson.EPSG_SEPARATOR, '')
            epsg_code = int(epsg_code.replace(CubeJson.EPSG_PREFIX, ''))

         # Store relevant information about the cube
         epsg_cubes = cube_info.setdefault(epsg_code, {})
         epsg_cubes[coord_tuple] = feature
         # {
         #       'index': i,
         #       'original_geometry': feature['geometry']['coordinates'],
         #       'properties': feature['properties'],
         #       'epsg_coords': epsg_coords
         # }
      except (KeyError, IndexError, TypeError) as e:
         print(f"Warning: Skipping malformed feature at index {i} - {e}")
         continue

   return cube_info


def compare_cube_files(file1_path: str, file2_path: str) -> Dict:
   """
   Compare two cube definition files and identify differences.

   Returns:
      Dict with keys: 'only_in_file1', 'only_in_file2', 'common', 'file1_info', 'file2_info'
   """
   print(f"Loading {file1_path}...")
   data1 = load_json_file(file1_path)
   cube_info1 = get_cube_info(data1)

   print(f"Loading {file2_path}...")
   data2 = load_json_file(file2_path)
   cube_info2 = get_cube_info(data2)

   only_in_file1 = []
   only_in_file2 = []
   num_common = 0

   # Per each epsg code, compare the sets of coordinate tuples
   for each_epsg in cube_info1.keys():
      print(f'\nComparing EPSG: {each_epsg}')

      coords1 = set(cube_info1[each_epsg].keys())

      if each_epsg in cube_info2:
         coords2 = set(cube_info2[each_epsg].keys())

         keys_in_file1 = coords1 - coords2

         # Collect all features only in file1 for this EPSG
         for each_key in keys_in_file1:
            only_in_file1.append(cube_info1[each_epsg][each_key])

         print(f'Appended {len(keys_in_file1)} cubes only in file1: {len(only_in_file1)=} total.')

         keys_in_file2 = coords2 - coords1
         # Collect all features only in file2 for this EPSG
         for each_key in keys_in_file2:
            only_in_file2.append(cube_info2[each_epsg][each_key])

         print(f'Appended {len(keys_in_file2)} cubes only in file2: {len(only_in_file2)=} total.')

         common = coords1 & coords2
         num_common += len(common)

         print(f"EPSG: {each_epsg} summary")
         print(f"Cubes in {file1_path}: {len(coords1)}")
         print(f"Cubes in {file2_path}: {len(coords2)}")
         print(f"Cubes only in {file1_path}: {len(keys_in_file1)}")
         print(f"Cubes only in {file2_path}: {len(keys_in_file2)}")
         print(f"Common cubes: {len(common)}")

      else:
         print(f"EPSG: {each_epsg} is only in {file1_path} with {len(cube_info1[each_epsg])} cubes.")
         for each_key in cube_info1[each_epsg]:
            only_in_file1.append(cube_info1[each_epsg][each_key])

   print(f"\nComparison Results:")
   print(f"Common cubes: {num_common}")
   print(f"Cubes only in {file1_path}: {len(only_in_file1)}")
   print(f"Cubes only in {file2_path}: {len(only_in_file2)}")

   return {
      'only_in_file1': only_in_file1,
      'only_in_file2': only_in_file2,
      'common': num_common
   }


def save_differences_to_file(
   comparison_result: Dict,
   output_file: str
):
   """Save identified missing cubes to a JSON file."""
   output_cubes = {"type": "FeatureCollection", "features": []}

   output_cubes['features'] = comparison_result['only_in_file1']

   with open(output_file, 'w', encoding='utf-8') as fh:
      json.dump(output_cubes, fh, indent=4)

   print(f"\nMissing cubes are saved to: {output_file}")


def main():
   parser = argparse.ArgumentParser(
      description="Compare two cube definition JSON files and identify differences",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
python cube_compare.py file1.json file2.json
python cube_compare.py file1.json file2.json --verbose
python cube_compare.py file1.json file2.json --output differences.json
      """
   )

   parser.add_argument(
      '--file1', help='First catalog JSON file path'
   )
   parser.add_argument(
      '--file2', help='Second catalog JSON file path'
   )
   parser.add_argument(
      '--output', '-o', metavar='OUTPUT_FILE',
      default=None,
      help='Save difference of comparison results to JSON file'
   )

   args = parser.parse_args()

   try:
      comparison_result = compare_cube_files(args.file1, args.file2)

      if args.output:
         save_differences_to_file(comparison_result, args.output)

   except Exception as e:
      print(f"Error: {e}")
      return 1

   return 0


if __name__ == "__main__":
   exit(main())
