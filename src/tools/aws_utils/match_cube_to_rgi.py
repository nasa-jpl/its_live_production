#!/usr/bin/env python3
"""
Script to extract RGI_CODE values from catalog_v02.json for files listed in missing_files.json
"""

import json
import os
from urllib.parse import urlparse
from typing import Dict, List, Set


def load_json_file(filepath: str) -> dict:
   """Load and parse a JSON file."""
   try:
      with open(filepath, 'r') as f:
         return json.load(f)
   except FileNotFoundError:
      print(f"Error: File '{filepath}' not found.")
      return {}
   except json.JSONDecodeError as e:
      print(f"Error: Invalid JSON in '{filepath}': {e}")
      return {}


def extract_filename_from_url(url: str) -> str:
   """Extract filename from zarr_url."""
   return os.path.basename(urlparse(url).path)


def build_catalog_mapping(catalog_data: dict) -> Dict[str, int]:
   """
   Build a mapping from filename to RGI_CODE from catalog data.

   Args:
      catalog_data: The loaded catalog_v02.json data

   Returns:
      Dictionary mapping filename to RGI_CODE
   """
   filename_to_rgi = {}

   if 'features' not in catalog_data:
      print("Error: 'features' key not found in catalog data")
      return filename_to_rgi

   for feature in catalog_data['features']:
      if 'properties' not in feature:
         continue

      properties = feature['properties']

      if 'zarr_url' not in properties or 'RGI_CODE' not in properties:
         continue

      zarr_url = properties['zarr_url']
      rgi_code = properties['RGI_CODE']

      # Extract filename from URL
      filename = extract_filename_from_url(zarr_url)

      if filename:
         filename_to_rgi[filename] = rgi_code

   return filename_to_rgi


def extract_rgi_codes_for_missing_files(catalog_file: str, missing_files_file: str) -> Dict[str, int]:
   """
   Extract RGI_CODE values for files listed in missing_files.json.

   Args:
      catalog_file: Path to catalog_v02.json
      missing_files_file: Path to missing_files.json

   Returns:
      Dictionary mapping missing filename to RGI_CODE
   """
   # Load both files
   catalog_data = load_json_file(catalog_file)
   missing_files_data = load_json_file(missing_files_file)

   if not catalog_data or not isinstance(missing_files_data, list):
      return {}

   # Build filename to RGI_CODE mapping
   filename_to_rgi = build_catalog_mapping(catalog_data)

   # Extract RGI codes for missing files
   missing_file_rgi_codes = {}
   files_not_found = []

   for missing_file in missing_files_data:
      if missing_file in filename_to_rgi:
         missing_file_rgi_codes[missing_file] = filename_to_rgi[missing_file]
      else:
         files_not_found.append(missing_file)

   # Print summary
   print(f"Total missing files: {len(missing_files_data)}")
   print(f"Found RGI_CODE for: {len(missing_file_rgi_codes)} files")
   print(f"Could not find RGI_CODE for: {len(files_not_found)} files")

   if files_not_found:
      print(f"\nFiles not found in catalog (first 10):")
      for file in files_not_found:
         print(f"  - {file}")

   return missing_file_rgi_codes


def save_results(results: Dict[str, int], output_file: str):
   """Save results to a JSON file."""
   try:
      with open(output_file, 'w') as f:
         json.dump(results, f, indent=2)
      print(f"\nResults saved to: {output_file}")
   except Exception as e:
      print(f"Error saving results to '{output_file}': {e}")


def print_rgi_code_summary(results: Dict[str, int]):
   """Print summary of RGI codes found."""
   if not results:
      print("No results to summarize.")
      return

   # Count files by RGI_CODE
   rgi_counts = {}
   for rgi_code in results.values():
      rgi_counts[rgi_code] = rgi_counts.get(rgi_code, 0) + 1

   print(f"\nRGI_CODE distribution:")
   for rgi_code in sorted(rgi_counts.keys()):
      count = rgi_counts[rgi_code]
      print(f"  RGI_CODE {rgi_code}: {count} files")


def main():
   """Main function to run the extraction."""
   catalog_file = "../aws/final_V2_datacubes_10.02.2023/from-s3/catalog_v02.json"
   missing_files_file = "missing_files.json"
   output_file = "missing_files_rgi_codes_aug2025.json"

   print("Extracting RGI_CODE values for missing files...")
   print(f"Catalog file: {catalog_file}")
   print(f"Missing files: {missing_files_file}")
   print("-" * 50)

   # Extract RGI codes
   results = extract_rgi_codes_for_missing_files(catalog_file, missing_files_file)

   if results:
      # Print RGI code summary
      print_rgi_code_summary(results)

      # Save results to file
      save_results(results, output_file)

      # Print first few results as example
      for i, (filename, rgi_code) in enumerate(list(results.items())):
         print(f"  {filename} -> RGI_CODE: {rgi_code}")
   else:
      print("No RGI_CODE values were extracted.")


if __name__ == "__main__":
   main()