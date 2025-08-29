#!/usr/bin/env python3
"""
Script to compare log files with catalog entries and identify missing files.
"""

import json
import os
import re
from pathlib import Path
from typing import Set, List, Dict, Any, Union

def search_missing_files_in_subdirectories(
   missing_files: Set[str],
   search_directory: str = "update_V2_datacubes_2025") -> Dict[str, List[str]]:
   """
   Search for filenames*.json files in nested subdirectories and check if they contain missing files.

   Args:
      missing_files: Set of zarr filenames to search for
      search_directory: Root directory to search in

   Returns:
      Dictionary mapping found filenames to list of missing files they contain
   """
   found_files = {}

   if not os.path.exists(search_directory):
      print(f"Warning: Search directory '{search_directory}' not found.")
      return found_files

   # Search for all filenames*.json files recursively
   pattern = os.path.join(search_directory, "**", "filenames*.json")
   json_files = glob.glob(pattern, recursive=True)

   print(f"\nSearching for missing files in {len(json_files)} filenames*.json files...")

   for json_file in json_files:
      try:
         # Read the JSON file
         with open(json_file, 'r', encoding='utf-8') as f:
            file_zarr_names = json.load(f)

         # Extract zarr filenames from this file
         file_zarr_names = set(file_zarr_names)

         # Check for intersection with missing files
         found_missing = file_zarr_names.intersection(missing_files)

         if found_missing:
            relative_path = os.path.relpath(json_file, search_directory)
            found_files[relative_path] = sorted(list(found_missing))
            print(f"  Found {len(found_missing)} missing files in: {relative_path}")

      except Exception as e:
            print(f"  Warning: Could not process {json_file}: {e}")

   return found_files

def extract_zarr_filename_from_log(log_filename: str) -> str:
   """
   Extract the zarr filename from a log filename.

   Example:
   Input: "ITS_LIVE_vel_EPSG3413_G0120_X-150000_Y-2250000.zarr_2025-06-04T18:31:44.407Z.log"
   Output: "ITS_LIVE_vel_EPSG3413_G0120_X-150000_Y-2250000.zarr"
   """
   # Use regex to extract everything before the timestamp pattern
   pattern = r'^(.+\.zarr)_\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\.log$'
   match = re.match(pattern, log_filename)

   if match:
      return match.group(1)
   else:
      # Fallback: try to find .zarr and extract up to that point
      zarr_index = log_filename.find('.zarr')
      if zarr_index != -1:
         return log_filename[:zarr_index + 5]  # +5 to include '.zarr'
      else:
         raise ValueError(f"Could not extract zarr filename from: {log_filename}")


def extract_zarr_filename_from_url(zarr_url: str) -> str:
   """
   Extract the zarr filename from a zarr URL.

   Example:
   Input: "s3://its-live-data/velocity_image_pair/v02/N70W070/ITS_LIVE_vel_EPSG3031_G0120_X2650000_Y-350000.zarr"
   Output: "ITS_LIVE_vel_EPSG3031_G0120_X2650000_Y-350000.zarr"
   """
   # Extract filename from URL path
   filename = Path(zarr_url).name
   return filename


def read_json_file(filepath: str) -> Union[Dict, List]:
   """Read and parse a JSON file."""
   with open(filepath, 'r', encoding='utf-8') as f:
      return json.load(f)


def get_log_zarr_filenames(failed_logs: List[str], ok_logs: List[str]) -> Set[str]:
   """Extract zarr filenames from log file lists and remove duplicates."""
   failed_zarr_filenames = set()

   # Process failed log files
   for log_file in failed_logs:
      try:
         zarr_name = extract_zarr_filename_from_log(log_file)
         failed_zarr_filenames.add(zarr_name)
      except ValueError as e:
         print(f"Warning: {e}")

   print(f'Got {len(failed_zarr_filenames)} unique cubes from {len(failed_logs)} failed logs')

   # Process ok log files
   ok_zarr_filenames = set()
   ok_duplicates = set()

   for log_file in ok_logs:
      try:
         zarr_name = extract_zarr_filename_from_log(log_file)
         if zarr_name in ok_zarr_filenames:
            ok_duplicates.add(zarr_name)

         ok_zarr_filenames.add(zarr_name)

      except ValueError as e:
         print(f"Warning: {e}")

   print(f'Got {len(ok_zarr_filenames)} unique cubes from {len(ok_logs)} ok logs')
   print(f'Got {len(ok_duplicates)} duplicates cubes from ok logs')

   if len(ok_duplicates):
      for filename in ok_duplicates:
         print(f"  - {filename}")

   # Find cubes that are in failed logs but are listed in OK logs (processed
   # eventually OK)
   totally_failed_cubes = [x for x in failed_zarr_filenames if x not in ok_zarr_filenames]
   print(f'Got {len(totally_failed_cubes)} not reprocessed failed cubes ')
   if len(totally_failed_cubes):
      for filename in totally_failed_cubes:
         print(f"  - {filename}")

      # Store these cubes to the json file
      output_file = 'cubes_to_reprocess_aug.2025.json'
      with open(output_file, 'w', encoding='utf-8') as f:
         json.dump(sorted(list(totally_failed_cubes)), f, indent=2)

      print(f"Not reprocessed files are saved to: {output_file}")


   ok_zarr_filenames.update(failed_zarr_filenames)
   return ok_zarr_filenames


def get_catalog_zarr_filenames(catalog_data: Dict[str, Any]) -> Set[str]:
   """Extract zarr filenames from catalog data (GeoJSON FeatureCollection format)."""
   zarr_filenames = set()

   # Check if it's a GeoJSON FeatureCollection
   if catalog_data.get('type') == 'FeatureCollection' and 'features' in catalog_data:
      features = catalog_data['features']

      for feature in features:
         if (feature.get('properties') and
               'zarr_url' in feature['properties'] and
               feature['properties']['zarr_url']):

               try:
                  zarr_name = extract_zarr_filename_from_url(feature['properties']['zarr_url'])
                  zarr_filenames.add(zarr_name)
               except Exception as e:
                  print(f"Warning: Could not extract filename from {feature['properties']['zarr_url']}: {e}")
   else:
      # Fallback for list format (original structure)
      if isinstance(catalog_data, list):
         for entry in catalog_data:
               if 'zarr_url' in entry:
                  try:
                     zarr_name = extract_zarr_filename_from_url(entry['zarr_url'])
                     zarr_filenames.add(zarr_name)
                  except Exception as e:
                     print(f"Warning: Could not extract filename from {entry.get('zarr_url', 'N/A')}: {e}")
      else:
         print("Error: Catalog data format not recognized. Expected GeoJSON FeatureCollection or list format.")

   return zarr_filenames


def main():
   """Main function to execute the comparison."""
   # File paths
   log_dir = "../dev_notebooks/debug_cubes_updates"
   failed_logs_file = os.path.join(log_dir, "failed_log_files.json")
   ok_logs_file = os.path.join(log_dir, "ok_log_files.json")
   catalog_file = "../aws/update_V2_datacubes_2025/datacube_update_05222025/catalog_v02_from_s3.json"

   # Read JSON files
   print("Reading JSON files...")
   failed_logs_data = read_json_file(failed_logs_file)
   ok_logs_data = read_json_file(ok_logs_file)

   # Read catalog from project resources or local file
   print("Reading catalog from project resources...")
   catalog_data = read_json_file(catalog_file)

   # Check if files were read successfully
   if failed_logs_data is None or ok_logs_data is None or catalog_data is None:
      print("Error: Could not read one or more input files.")
      return

   if not isinstance(catalog_data, dict) or catalog_data.get('type') != 'FeatureCollection':
      # Check if it's the old list format for backward compatibility
      if not isinstance(catalog_data, list):
         print("Error: catalog_v02_from_s3.json should contain a GeoJSON FeatureCollection or list of entries.")
         return

   # Extract zarr filenames from log files
   print("Extracting zarr filenames from log files...")
   log_zarr_filenames = get_log_zarr_filenames(failed_logs_data, ok_logs_data)
   print(f"Found {len(log_zarr_filenames)} unique zarr files in log data.")

   # Extract zarr filenames from catalog
   print("Extracting zarr filenames from catalog...")
   if isinstance(catalog_data, dict) and catalog_data.get('type') == 'FeatureCollection':
      print(f"Found GeoJSON FeatureCollection with {len(catalog_data.get('features', []))} features.")

   catalog_zarr_filenames = get_catalog_zarr_filenames(catalog_data)
   print(f"Found {len(catalog_zarr_filenames)} zarr files in catalog.")

   # Find files in catalog that are not in log files
   missing_files = catalog_zarr_filenames - log_zarr_filenames

   # Display results
   print(f"\nComparison Results:")
   print(f"Files in log data: {len(log_zarr_filenames)}")
   print(f"Files in catalog: {len(catalog_zarr_filenames)}")
   print(f"Files in catalog but NOT in log data: {len(missing_files)}")

   if missing_files:
      print("\nMissing files (in catalog but not in logs):")
      for filename in sorted(missing_files):
         print(f"  - {filename}")

      # Optionally save missing files to a JSON file
      output_file = "missing_files.json"
      with open(output_file, 'w', encoding='utf-8') as f:
         json.dump(sorted(list(missing_files)), f, indent=2)
      print(f"\nMissing files saved to: {output_file}")

   else:
      print("\nAll catalog files are present in the log data.")

   # Additional statistics
   common_files = catalog_zarr_filenames & log_zarr_filenames
   only_in_logs = log_zarr_filenames - catalog_zarr_filenames

   print(f"\nAdditional Statistics:")
   print(f"Files present in both: {len(common_files)}")
   print(f"Files only in logs (not in catalog): {len(only_in_logs)}")

   if only_in_logs:
      print("\nFiles in logs but NOT in catalog ({len(only_in_logs=)}:)")
      for filename in sorted(list(only_in_logs)):
         print(f"  - {filename}")

   # Final step: Search for missing files in subdirectories
   if missing_files:
      print(f"\n{'='*60}")
      print("Searching for missing files in subdirectories...")
      print(f"{'='*60}")

      found_files = search_missing_files_in_subdirectories(missing_files)

      if found_files:
         print(f"\nSUMMARY: Found {sum(len(files) for files in found_files.values())} missing files across {len(found_files)} JSON files:")
         print("-" * 40)

         for json_file, files_found in found_files.items():
               print(f"\n{json_file=}:")
               for zarr_file in files_found:
                  print(f"  - {zarr_file}")

         # # Save detailed results
         # save_search_results(found_files)

         # Update missing files list (remove found ones)
         all_found = set()
         for files_list in found_files.values():
               all_found.update(files_list)

         still_missing = missing_files - all_found

         print(f"\nFINAL STATUS:")
         print(f"  Total missing files: {len(missing_files)}")
         print(f"  Found in subdirectories: {len(all_found)}")
         print(f"  Still missing: {len(still_missing)}")

         if still_missing:
               print(f"\nFiles still not found anywhere:")
               for filename in sorted(list(still_missing)):
                  print(f"  - {filename}")

               # Save the truly missing files
               filename = "truly_missing_files_aug2025.json"
               with open(filename, 'w', encoding='utf-8') as f:
                  json.dump(sorted(list(still_missing)), f, indent=2)

               print(f"\nTruly missing files saved to: {filename}")
      else:
         print("No missing files were found in any filenames*.json files in subdirectories.")

   else:
      print(f"\n{'='*60}")
      print("No missing files to search for - all catalog files found in logs")
      print(f"{'='*60}")

if __name__ == "__main__":
   main()