import argparse
import gc
import logging
import pandas as pd

# Define expected associated file extensions
# (must check longer extensions first to avoid partial matches)
expected_extensions = [
    '_thumb.png',
    '.png',
    '.nc.premet',
    '.nc.spatial',
    '.stac.json',
]


all_extensions = expected_extensions + [ '.nc' ]



# Extract base key by removing known extensions
def get_base_key(key):
    for ext in all_extensions:
        if key.endswith(ext):
            return key[:-len(ext)]
    return key


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Check completeness of ITS_LIVE velocity granules by '
            'verifying associated files exist for each .nc file'
        )
    )
    parser.add_argument(
        'manifest_file',
        type=str,
        help='Path to the velocity manifest parquet file',
    )
    parser.add_argument(
        '--mission',
        type=str,
        default='sentinel1',
        help='Mission to filter out that fall before the cutoff-date [%(default)s]',
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        default='2025-11-01',
        help=(
            'Cutoff date in YYYY-MM-DD format. '
            'Files before this date will be excluded [%(default)s]'
        ),
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help=('Output file to save completeness report to.')
    )
    return parser.parse_args()


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    # Read the manifest file (only needed columns)
    logger.info(f'Reading manifest file: {args.manifest_file}')
    df = pd.read_parquet(
        args.manifest_file,
        columns=['key', 'mission', 'last_modified_date']
    )

    logger.info(f'Total rows in manifest: {len(df)}')

    # Use categorical dtype for mission to reduce memory usage
    df['mission'] = df['mission'].astype('category')

    # Convert last_modified_date to datetime
    df['last_modified_date'] = pd.to_datetime(df['last_modified_date'])
    cutoff_date = pd.Timestamp(args.cutoff_date, tz='UTC')

    # Exclude only the specified mission's pre-cutoff files
    # Keep: 1) all other missions (any date)
    #       2) specified mission post-cutoff files
    logger.info(
        f'Excluding {args.mission} files before {cutoff_date}, '
        f'keeping all other missions and {args.mission} post-cutoff'
    )
    is_target_mission = df['mission'] == args.mission
    is_before_cutoff = df['last_modified_date'] < cutoff_date
    exclude_mask = is_target_mission & is_before_cutoff

    logger.info(
        f'Excluding {exclude_mask.sum()} {args.mission} '
        f'files before cutoff'
    )
    df = df[~exclude_mask]
    logger.info(f'Rows after filtering: {len(df)}')

    # Identify all .nc files in remaining data
    nc_files = df[df['key'].str.endswith('.nc')].copy()
    logger.info(f'Found {len(nc_files)} .nc files')

    # Show .nc files by mission
    logger.info('\n.nc files by mission:')
    logger.info(f'\n{nc_files["mission"].value_counts()}')

    # Build a flat key lookup set — far more memory-efficient than
    # groupby().apply(set).to_dict() which creates nested Python sets
    logger.info('Building key lookup set...')
    all_keys_set = set(df['key'].values)
    logger.info(f'Built lookup set with {len(all_keys_set)} keys')

    # Free the full DataFrame — no longer needed
    del df
    gc.collect()
    logger.info('Released full DataFrame from memory')

    logger.info(
        f'Processing {len(nc_files)} .nc files for completeness...'
    )

    # Create results DataFrame starting with nc_files
    results_df = nc_files[['key', 'last_modified_date']]
    results_df.rename(columns={'key': 'nc_key'}, inplace=True)

    # Extract base_key (remove .nc extension)
    results_df['base_key'] = results_df['nc_key'].str[:-3]

    # Get file count for each base_key using flat set membership checks
    logger.info('Counting files for each base_key...')
    results_df['file_count'] = results_df['base_key'].map(
        lambda bk: 1 + sum(
            bk + ext in all_keys_set for ext in expected_extensions
        )
    )

    # Check completeness (expect 6 files: 1 .nc + 5 associated)
    results_df['is_complete'] = results_df['file_count'] == 6

    logger.info('Checking missing extensions for incomplete files...')
    # For incomplete files, check which extensions are missing
    incomplete_mask = ~results_df['is_complete']
    incomplete_base_keys = results_df.loc[incomplete_mask, 'base_key']

    def check_missing_extensions(base_key):
        """Check which extensions are missing for a base_key."""
        missing = [
            ext for ext in expected_extensions
            if base_key + ext not in all_keys_set
        ]
        return ','.join(missing) if missing else ''

    # Only check incomplete files
    results_df['missing_extensions'] = ''
    if incomplete_mask.sum() > 0:
        results_df.loc[incomplete_mask, 'missing_extensions'] = (
            incomplete_base_keys.apply(check_missing_extensions)
        )

    # Drop temporary base_key column
    results_df.drop(columns=['base_key'], inplace=True)

    logger.info('Completeness check finished.')

    # Save results to file
    output_file = args.output

    results_df.to_parquet(output_file)
    logger.info(f'\nSaved completeness report to: {output_file}')

    # Save incomplete files to separate parquet file
    incomplete = results_df[~results_df['is_complete']]
    if len(incomplete) > 0:
        incomplete_file = output_file.replace(
            '.parquet', '_incomplete.parquet'
        )
        incomplete.to_parquet(incomplete_file)
        logger.info(
            f'Saved {len(incomplete)} incomplete files to: '
            f'{incomplete_file}'
        )

        logger.info(f'\nShowing first 10 incomplete files:')
        logger.info(
            f'\n{incomplete[["nc_key", "file_count", "missing_extensions"]].head(10).to_string(index=False)}'
        )

    # Summary statistics
    logger.info('\n=== Completeness Summary ===')
    logger.info(f'Total .nc files: {len(results_df)}')
    logger.info(
        f'Complete files (6 files total): '
        f'{results_df["is_complete"].sum()}'
    )
    logger.info(
        f'Incomplete files: '
        f'{(~results_df["is_complete"]).sum()}'
    )

    logger.info('\nMissing file breakdown:')
    # Count occurrences of each extension in missing_extensions
    for ext in expected_extensions:
        missing_count = results_df['missing_extensions'].str.contains(
            ext.replace('.', r'\.'), regex=True
        ).sum()
        logger.info(f'  Missing {ext}: {missing_count}')

    # Show distribution by file count
    logger.info('\nDistribution by total file count:')
    logger.info(
        f'\n{results_df["file_count"].value_counts().sort_index()}'
    )


if __name__ == '__main__':
    main()
