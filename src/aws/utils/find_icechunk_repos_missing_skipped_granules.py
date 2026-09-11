#!/usr/bin/env python3
"""
Identify icechunk datacube repositories under an S3 prefix (default:
s3://its-live-data/datacubes/spatial/v2.2/) that don't have a corresponding
"<repo>_skippedGranules.json" sibling file.

Every successful run of virtual_itslive_cube_per_chunk.py writes this JSON
file (even when the skipped-granules list is empty), so a repo missing it
indicates the build never got far enough to write it -- e.g. it crashed
before completing its first batch, or the repo predates that code path.

Both the ".icechunk/" repo "directories" and the flat "*_skippedGranules.json"
files live side by side directly under the prefix, so a single
Delimiter='/' listing pass is enough -- no need to descend into any repo's
internal chunk/manifest structure.
"""
import argparse
import boto3

DEFAULT_BUCKET = 'its-live-data'
DEFAULT_PREFIX = 'datacubes/spatial/v2.2/'

ICECHUNK_SUFFIX = '.icechunk'
SKIPPED_GRANULES_SUFFIX = '_skippedGranules.json'


def find_repos_missing_skipped_granules(bucket, prefix):
   """List icechunk repos under bucket/prefix that have no matching
   "<repo>_skippedGranules.json" sibling file.

   Parameters
   ----------
   bucket : str
      S3 bucket name.
   prefix : str
      S3 prefix to list (must end with '/'), one level above the
      "<name>.icechunk/" repo directories.

   Returns
   -------
   Tuple[List[str], Set[str], Set[str]]
      Sorted list of repo names missing their skipped-granules file, the
      full set of repo names found, and the full set of repo names that do
      have a skipped-granules file.
   """
   if not prefix.endswith('/'):
      prefix += '/'

   s3 = boto3.client('s3', region_name='us-west-2')
   paginator = s3.get_paginator('list_objects_v2')

   repo_names = set()
   skipped_names = set()

   for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
      for common_prefix in page.get('CommonPrefixes', []):
         name = common_prefix['Prefix'][len(prefix):].rstrip('/')
         if name.endswith(ICECHUNK_SUFFIX):
            repo_names.add(name[:-len(ICECHUNK_SUFFIX)])

      for obj in page.get('Contents', []):
         name = obj['Key'][len(prefix):]
         if name.endswith(SKIPPED_GRANULES_SUFFIX):
            skipped_names.add(name[:-len(SKIPPED_GRANULES_SUFFIX)])

   missing = sorted(repo_names - skipped_names)

   return missing, repo_names, skipped_names


def main():
   parser = argparse.ArgumentParser(
      description=(
         'Find icechunk repos under an S3 prefix that have no matching '
         '"<repo>_skippedGranules.json" sibling file.'
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
python find_icechunk_repos_missing_skipped_granules.py
python find_icechunk_repos_missing_skipped_granules.py --output missing.txt
      """
   )

   parser.add_argument(
      '--bucket', default=DEFAULT_BUCKET,
      help=f'S3 bucket to scan (default: {DEFAULT_BUCKET})'
   )
   parser.add_argument(
      '--prefix', default=DEFAULT_PREFIX,
      help=f'S3 prefix to scan (default: {DEFAULT_PREFIX})'
   )
   parser.add_argument(
      '--output', '-o', metavar='OUTPUT_FILE', default=None,
      help='Save s3:// URIs of repos missing their skipped-granules file to this text file'
   )

   args = parser.parse_args()

   prefix = args.prefix if args.prefix.endswith('/') else args.prefix + '/'

   missing, repo_names, skipped_names = find_repos_missing_skipped_granules(
      args.bucket, prefix
   )

   missing_uris = [
      f's3://{args.bucket}/{prefix}{name}{ICECHUNK_SUFFIX}' for name in missing
   ]

   print(f'Repos found:                {len(repo_names)}')
   print(f'Repos with skippedGranules: {len(skipped_names & repo_names)}')
   print(f'Repos missing the file:    {len(missing)}')
   print()

   for uri in missing_uris:
      print(uri)

   if args.output:
      with open(args.output, 'w', encoding='utf-8') as fh:
         fh.write('\n'.join(missing_uris) + ('\n' if missing_uris else ''))

      print(f'\nSaved {len(missing_uris)} URIs to {args.output}')

   return 0


if __name__ == '__main__':
   exit(main())
