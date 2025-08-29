#!/usr/bin/env python3
"""
Script to parse log files from nested 'done' subdirectories.
Extracts runtime and granule information from log files.
"""
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List


def parse_timestamp(line: str) -> Optional[datetime]:
   """
   Extract and parse timestamp from a log line.
   Expected format: "2025-07-03T22:40:41.307Z 07/03/2025 10:40:41 PM"

   Args:
      line: Log line containing timestamp

   Returns:
      datetime object if timestamp found, None otherwise
   """
   # Pattern to match ISO format timestamp at the beginning of line
   pattern = r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)'
   match = re.match(pattern, line.strip())

   if match:
      timestamp_str = match.group(1)
      try:
         # Parse ISO format timestamp (Z means UTC)
         return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
      except ValueError:
         return None

   return None


def extract_granules(line: str) -> Optional[int]:
   """
   Extract number of granules from a log line.
   Expected format: "... - INFO - Leaving 2505 granules..."

   Args:
      line: Log line potentially containing granule information

   Returns:
      Number of granules if found, None otherwise
   """
   if not ( ('Leaving ' in line) and ('granules' in line) ):
      return None

   # Pattern to match "Leaving X granules" where X is a number
   pattern = r'Leaving\s+(\d+)\s+granules'
   match = re.search(pattern, line, re.IGNORECASE)

   if match:
      return int(match.group(1))

   return None


def process_log_file(file_path: Path) -> Tuple[float, int]:
   """
   Process a single log file to extract runtime and granules.

   Args:
      file_path: Path to the log file

   Returns:
      Tuple of (runtime_hours, granules_count)
   """
   runtime_hours = 0.0
   granules = 0
   first_timestamp = None
   last_timestamp = None

   try:
      with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
         lines = f.readlines()

         if not lines:
               return runtime_hours, granules

         # Find first timestamp
         for line in lines:
               timestamp = parse_timestamp(line)
               if timestamp:
                  first_timestamp = timestamp
                  break

         # Find last timestamp and look for granules throughout the file
         for line in lines:
               # Check for timestamp
               timestamp = parse_timestamp(line)
               if timestamp:
                  last_timestamp = timestamp

               # Check for granules information
               granule_count = extract_granules(line)
               if granule_count:
                  granules = max(granules, granule_count)  # Take the maximum if multiple found

         # Calculate runtime if we have both timestamps
         if first_timestamp and last_timestamp:
               runtime_seconds = (last_timestamp - first_timestamp).total_seconds()
               runtime_hours = runtime_seconds / 3600

   except Exception as e:
      print(f"Error processing {file_path}: {e}")

   return runtime_hours, granules


def find_done_directories(root_path: str) -> List[Path]:
   """
   Find all 'done' subdirectories under the given root path.

   Args:
      root_path: Root directory to search from

   Returns:
      List of Path objects pointing to 'done' directories
   """
   done_dirs = []
   root = Path(root_path)

   # Walk through directory tree
   for path in root.rglob('*'):
      if path.is_dir() and path.name == 'done':
         done_dirs.append(path)

   return done_dirs


def main():
   """Main function to process all log files."""
   # Get current directory or specify a custom root path
   root_directory = input("Enter the root directory path (or press Enter for current directory): ").strip()
   if not root_directory:
      root_directory = os.getcwd()

   if not os.path.exists(root_directory):
      print(f"Error: Directory '{root_directory}' does not exist.")
      return

   print(f"\nSearching for 'done' directories in: {root_directory}")
   print("-" * 60)

   # Find all done directories
   done_dirs = find_done_directories(root_directory)

   if not done_dirs:
      print("No 'done' directories found.")
      return

   print(f"Found {len(done_dirs)} 'done' directories:")
   for dir_path in done_dirs:
      print(f"  - {dir_path}")

   print("\nProcessing log files...")
   print("-" * 60)

   total_runtime_hours = 0.0
   total_granules = 0
   processed_files = 0

   all_log_files = []
   # Process each done directory
   for done_dir in done_dirs:
      # Find all .log files in the done directory
      log_files = list(done_dir.glob('*.log'))
      all_log_files.extend( [each.name for each in log_files] )

      for log_file in log_files:
         runtime, granules = process_log_file(log_file)

         if runtime > 0 or granules > 0:
               print(f"\nFile: {log_file.name}")
               print(f"  Directory: {done_dir}")
               print(f"  Runtime: {runtime:.2f} hours")
               print(f"  Granules: {granules}")

               total_runtime_hours += runtime
               total_granules += granules
               processed_files += 1

   # Print summary
   print("\n" + "=" * 60)
   print("SUMMARY")
   print("=" * 60)
   print(f"Total log files processed: {processed_files}")
   print(f"Total runtime: {total_runtime_hours:.2f} hours ({total_runtime_hours/24:.2f} days)")
   print(f"Total granules processed: {total_granules:,}")

   if processed_files > 0:
      print(f"Average runtime per job: {total_runtime_hours/processed_files:.2f} hours")
      if total_granules > 0 and total_runtime_hours > 0:
         print(f"Average processing rate: {total_granules/total_runtime_hours:.1f} granules/hour")

   # Save log files to the json list
   file_list_file = 'log_files.json'
   print(f'Saving all found log files to the {file_list_file}')
   with open(file_list_file, 'w') as fh:
      json.dump(all_log_files, fh, indent=3)

if __name__ == "__main__":
   main()