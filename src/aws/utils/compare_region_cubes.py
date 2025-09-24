import json
import os
import argparse

def load_urls_from_json(filepath):
   """Load a list of URLs from a JSON file."""
   with open(filepath, 'r') as f:
      data = json.load(f)
      if not isinstance(data, list):
         raise ValueError(f"JSON file {filepath} must contain a list of strings.")
      return data

def compare_url_lists(file1_path, file2_path):
   """Compare basenames of URLs from two JSON files and report differences."""
   urls1 = load_urls_from_json(file1_path)
   urls2 = load_urls_from_json(file2_path)

   # Create mapping from basename to full URL for each list
   basename_to_url1 = {os.path.basename(url): url for url in urls1}
   basename_to_url2 = {os.path.basename(url): url for url in urls2}

   basenames1 = set(basename_to_url1.keys())
   basenames2 = set(basename_to_url2.keys())

   only_in_file1 = basenames1 - basenames2
   only_in_file2 = basenames2 - basenames1

   print(f"Found {len(only_in_file1)} files in {file1_path} not in {file2_path}:")
   for bn in sorted(only_in_file1):
      print(f"  {basename_to_url1[bn]}")

   print(f"\nFound {len(only_in_file2)} files in {file2_path} not in {file1_path}:")
   for bn in sorted(only_in_file2):
      print(f"  {basename_to_url2[bn]}")

   if not only_in_file1 and not only_in_file2:
      print("\n✅ Both files contain the same basenames.")

def main():
   parser = argparse.ArgumentParser(description="Compare basenames of URLs in two JSON files.")
   parser.add_argument("-f1", help="Path to first JSON file (list of URLs)")
   parser.add_argument("-f2", help="Path to second JSON file (list of URLs)")
   args = parser.parse_args()

   compare_url_lists(args.f1, args.f2)

if __name__ == "__main__":
   main()