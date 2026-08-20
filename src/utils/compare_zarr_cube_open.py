"""
Diagnose slow-to-open Zarr datacubes by comparing open timing and on-disk
chunk layout across one or more stores (e.g. a fast itscube.py cube vs. a
slow deep_copy_cube.py cube).

For each --cube LABEL PATH given, this reports:
- xr.open_zarr() timing under consolidated={None, True, False}, to reveal
    whether consolidated metadata is actually being used on open.
- Per-variable zarr-level fill_value, chunk shape, and the number of chunk
    objects actually written to disk (vs. the theoretical total) for a few
    key variables, to reveal whether all-missing chunks are being physically
    written instead of skipped.
- Total chunk objects written across every array in the store.

Usage:
python compare_zarr_cube_open.py \
    --cube "itscube (fast)" /path/to/itscube_cube.zarr \
    --cube "deep_copy (slow)" s3://its-live-data/path/to/deep_copy_cube.zarr

--vars can be used to pick which data variables get the detailed
per-chunk report (default: v, vx, vy).
"""

import argparse
import logging
import sys
import timeit

import s3fs
import xarray as xr
import zarr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def open_zarr_group(path: str):
    """Open `path` (local or s3://) as a zarr.Group for direct metadata
    inspection (fill_value, chunks, nchunks_initialized).

    Parameters
    ----------
    path : str
        Local directory path or s3:// URL of the Zarr store.

    Returns
    -------
    zarr.Group
    """
    if path.startswith('s3://'):
        s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=path, s3=s3, check=False)
        return zarr.open(store, mode='r')

    return zarr.open(path, mode='r')


def report_open_timing(label: str, path: str):
    """Time xr.open_zarr() under consolidated={None, True, False} and log
    dataset shape on success, or the exception on failure.

    A wide gap between consolidated=True and consolidated=None/False (the
    latter as slow as a full listing) means the default open path isn't
    using consolidated metadata.
    """
    for consolidated in (None, True, False):
        start_time = timeit.default_timer()
        try:
            ds = xr.open_zarr(path, consolidated=consolidated)
            elapsed = timeit.default_timer() - start_time
            logging.info(
                f'[{label}] open(consolidated={consolidated}): '
                f'{elapsed:.2f}s, {len(ds.data_vars)} vars, '
                f'time={ds.sizes.get("time", ds.sizes.get("mid_date"))}'
            )
        except Exception as exc:
            elapsed = timeit.default_timer() - start_time
            logging.info(
                f'[{label}] open(consolidated={consolidated}): '
                f'FAILED after {elapsed:.2f}s - {type(exc).__name__}: {exc}'
            )


def report_chunk_layout(label: str, path: str, var_names):
    """Log zarr-level fill_value/chunks/nchunks_initialized for each of
    `var_names` present in the store, plus the total chunk objects written
    across every array in the store.

    A fill_value of None/0 on an integer variable that itscube.py hardcodes
    a missing_value for (see MISSING_VALUE_OVERRIDES in deep_copy_cube.py)
    means all-missing chunks can't be recognized as empty and get written
    anyway, inflating the object count -- the total below is the tell.
    """
    group = open_zarr_group(path)

    total_chunks = 0
    total_arrays = 0
    for name, array in group.arrays():
        total_arrays += 1
        # nchunks_initialized is the actual number of chunk objects present
        # on disk/S3 (skipped/empty chunks are never written); nchunks is
        # the theoretical maximum for the array's shape/chunks.
        try:
            written = array.nchunks_initialized
            theoretical = array.nchunks
        except Exception:
            written = theoretical = None

        if written is not None:
            total_chunks += written

        if name in var_names:
            logging.info(
                f'[{label}] {name}: fill_value={array.metadata.fill_value}, '
                f'chunks={array.chunks}, nchunks_initialized='
                f'{written}/{theoretical}'
            )

    logging.info(
        f'[{label}] TOTAL chunk objects written across {total_arrays} '
        f'arrays: {total_chunks}'
    )


def main(cubes, var_names):
    """Run the open-timing and chunk-layout report for every (label, path)
    pair in `cubes`.
    """
    for label, path in cubes:
        logging.info(f'=== {label}: {path} ===')
        report_open_timing(label, path)
        report_chunk_layout(label, path, var_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--cube',
        nargs=2,
        metavar=('LABEL', 'PATH'),
        action='append',
        dest='cubes',
        required=True,
        help='A label and a Zarr store path (local or s3://) to inspect. '
            'Repeat --cube for each store to compare.'
    )
    parser.add_argument(
        '--vars',
        nargs='+',
        default=['v', 'vx', 'vy'],
        help='Data variable names to report per-chunk detail for '
            '[%(default)s].'
    )

    args = parser.parse_args()
    logging.info(f'Command: {sys.argv}')

    main(args.cubes, set(args.vars))

    logging.info('Done')
