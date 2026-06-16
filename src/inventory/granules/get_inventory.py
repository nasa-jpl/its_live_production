import argparse
import json
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Download and process S3 bucket inventory for '
            'ITS_LIVE velocity granules'
        )
    )
    parser.add_argument(
        '--mission',
        type=str,
        default=None,
        help=(
            'Mission to filter for removal '
            '(e.g., sentinel1, sentinel2, landsat8). '
            'If not specified, no granules are being filtered by cutoff date.'
        )
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        default='2025-11-01',
        help=(
            'Cutoff date in YYYY-MM-DD format. '
            'Granules before this date will be marked for removal. '
            'Default: 2025-11-01'
        )
    )
    return parser.parse_args()


_SESSION = boto3.session.Session(
    profile_name='opendata-its-live',
    region_name='us-west-2',
)

MANIFEST_DATE = '2026-06-10T01-00Z'
MANIFEST_INFO = {
    'Bucket': 'pds-buckets-its-live-logbucket-70tr3aw5f2op',
    'Key': (
        f'inventory/velocity_image_pair/its-live-data/'
        f'VelocityGranuleInventory/{MANIFEST_DATE}/manifest.json'
    ),
}


def main():
    args = parse_args()

    output_file = f'velocity_manifest_{MANIFEST_DATE}.parquet'

    # Check if consolidated parquet file exists from previous run
    if Path(output_file).exists():
        print(f'Loading existing manifest: {output_file}')
        df = pd.read_parquet(output_file)

    else:
        print(f'Downloading and processing inventory files...')
        s3_client = _SESSION.client('s3')

        response = s3_client.get_object(**MANIFEST_INFO)
        manifest = json.loads(response['Body'].read().decode('utf-8'))

        to_concat = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for ff in tqdm(manifest['files']):
                manifest_part = f'{tmpdir}/{Path(ff["key"]).name}'

                s3_client.download_file(
                    Bucket=MANIFEST_INFO['Bucket'],
                    Key=ff['key'],
                    Filename=manifest_part,
                )

                df_part = pd.read_parquet(manifest_part)
                df_part[['mission', 'version', 'cube']] = (
                    df_part.key.str.split('/', expand=True).iloc[:, 1:4]
                )
                df_part['type'] = df_part.key.str.split('.').str[-1]
                df_part.loc[
                    df_part.key.str.contains('_thumb'), 'type'
                ] = 'thumb'
                to_concat.append(df_part.copy())

        df = pd.concat(to_concat, ignore_index=True)

        # Save full inventory to Parquet for references
        df.to_parquet(output_file)
        print(f'Saved consolidated manifest to: {output_file}')
        del to_concat

    print(f'Unique mission values: {df.mission.unique()}')

    # Only perform cutoff filtering if mission is provided
    if args.mission:
        print(f'Filtering for mission: {args.mission}')
        df = df[df['mission'] == args.mission]

        if len(df) == 0:
            print(f'No granules found for mission: {args.mission}')
            return

        # Use command-line arguments for cutoff date
        cutoff_date = pd.Timestamp(args.cutoff_date, tz='UTC')
        df['last_modified_date'] = pd.to_datetime(
            df['last_modified_date']
        )

        # Granules before cutoff - to be removed from catalog
        before_cutoff_df = df[df['last_modified_date'] < cutoff_date]
        before_cutoff_df = before_cutoff_df.sort_values(
            'last_modified_date', ascending=True
        )
        last_date_before_cutoff = (
            before_cutoff_df.iloc[-1]['last_modified_date']
        )

        print(
            f'Granules with {last_date_before_cutoff=} < {cutoff_date} '
            f'(to be removed from catalog): {len(before_cutoff_df)}'
        )

        # Save granules to remove to a separate Parquet file
        before_cutoff_file = (
            f'velocity_manifest_{MANIFEST_DATE}_{args.mission}_'
            f'before_cutoff_{cutoff_date.date()}.parquet'
        )
        before_cutoff_df.to_parquet(before_cutoff_file)
        print(f'Saved granules to remove to: {before_cutoff_file}')

        # Granules on/after cutoff - to be included in catalog
        after_cutoff_df = df[df['last_modified_date'] >= cutoff_date]

        # Sort oldest first (newest as last entry)
        after_cutoff_df = after_cutoff_df.sort_values(
            'last_modified_date', ascending=True
        )
        first_date_after_cutoff = (
            after_cutoff_df.iloc[0]['last_modified_date']
        )

        print(
            f'Granules with {first_date_after_cutoff=} >= '
            f'{cutoff_date} (to be included in catalog): '
            f'{len(after_cutoff_df)}'
        )
    else:
        print(
            'No mission specified - processing all missions '
            'without cutoff filtering'
        )
        print(f'Total granules: {len(df)}')


if __name__ == '__main__':
    main()

