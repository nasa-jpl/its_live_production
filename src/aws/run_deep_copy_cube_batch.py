"""
Script to drive Batch processing for deep-copy (Zarr v3) datacube generation
at AWS.

It accepts the same geojson file of chunk-aligned datacube definitions that
run_virtual_cube_batch.py consumes, and submits one AWS Batch job per each
datacube which has ROI (region of interest) != 0 and whose source virtual
(icechunk) datacube already exists in S3. Each job runs
deep_copy_cube_per_var_chunk.py to materialize that icechunk repo into a real
Zarr v3 datacube.

Each catalog entry already carries an `icechunk_filename` property
(precomputed by define_cube_polygons_chunk_aligned.py using the same
midpoint/rounding convention as run_virtual_cube_batch.py). This script
derives the deep-copy cube's own filename from it by swapping the
'.icechunk' extension for '.zarr' -- the same convention
src/tools/add_url_to_datacube_definition.py uses to detect existing
deep-copy cubes -- and checks the source icechunk repo actually exists
before submitting a job for it.

Before submitting, each source icechunk repo is opened read-only to read
its layer count (same open_virtual_cube() deep_copy_cube_per_var_chunk.py
itself uses once the job is running). Cubes whose layer count exceeds
DeepCopyCubeBatch.NUM_LAYERS_64GB_THRESHOLD are routed to the 64Gb batch
queue *and* its paired, larger-memory job definition instead of the
default ones -- a job definition's container memory limit is enforced
regardless of which instance the queue happens to place it on, so the
queue alone wouldn't actually get a large cube any more RAM. Same
size-tiered idea as run_composites_batch.py's
get_aws_disable_slow_error_config().
"""
import argparse
import boto3
import json
import logging
import os
from pathlib import Path
import s3fs
import sys

from deep_copy_cube import open_virtual_cube
from itscube_types import BatchVars
from itslive_mosaics_types import GeoJsonVars
import utils
from utils import File


def _join_s3_url(location: str, filename: str) -> str:
    """Join a s3:// directory URL (with or without trailing slash) and a
    filename into a full s3:// URL.
    """
    return location.rstrip('/') + '/' + filename


class DeepCopyCubeBatch:
    """
    Class to manage Batch job submission for deep-copy (Zarr v3) datacube
    generation at AWS.
    """
    CLIENT = boto3.client('batch', region_name='us-west-2')

    # Thread-pool size per batch .load() call in
    # deep_copy_cube_per_var_chunk.py (--num-load-workers); matches the job
    # definition's allocated vCPUs (see src/aws/cdk/cubes/env/app.py
    # JOB_VCPUS).
    NUM_LOAD_WORKERS = 32

    # Cubes whose source icechunk repo has more layers than this are routed
    # to batch_queue_64gb instead of batch_queue -- same size-tiered-queue
    # idea as run_composites_batch.py's NUM_LAYERS_ONDEMAND threshold.
    NUM_LAYERS_64GB_THRESHOLD = 800_000

    def __init__(
        self,
        batch_job: str,
        batch_job_64gb: str,
        batch_queue: str,
        batch_queue_64gb: str,
        icechunk_cubes_location: str,
        zarr_cubes_location: str,
        bucket_prefix: str,
        progress_dir: str,
        is_dry_run: bool
    ):
        """
        Initialize object.
        """
        self.batch_job = batch_job
        self.batch_job_64gb = batch_job_64gb
        self.batch_queue = batch_queue
        self.batch_queue_64gb = batch_queue_64gb
        self.icechunk_cubes_location = icechunk_cubes_location
        self.zarr_cubes_location = zarr_cubes_location
        self.bucket_prefix = bucket_prefix
        self.progress_dir = progress_dir
        self.is_dry_run = is_dry_run

        self.s3 = s3fs.S3FileSystem(anon=True)

    def _check_icechunk_cube_exists(self, icechunk_filename: str):
        """Check whether the virtual icechunk repo named `icechunk_filename`
        exists at self.icechunk_cubes_location, via a plain S3 existence
        check (no icechunk repo open needed) -- same check
        add_url_to_datacube_definition.py's _check_icechunk_cube() uses.

        Returns
        -------
        tuple of (bool, str)
            (exists, S3 URL checked)
        """
        icechunk_s3_url = _join_s3_url(self.icechunk_cubes_location, icechunk_filename)
        return self.s3.exists(icechunk_s3_url), icechunk_s3_url

    def _pick_batch_resources(self, icechunk_s3_url: str):
        """Open the source icechunk repo read-only and pick the batch queue
        and job definition based on its layer count: the 64Gb pair if it
        exceeds NUM_LAYERS_64GB_THRESHOLD, the default pair otherwise. Both
        must change together -- a job definition's container memory limit
        is enforced regardless of which instance the queue places it on.

        Returns
        -------
        tuple of (str, str, int)
            (batch_queue, batch_job_definition, num_workers)
        """
        cube = open_virtual_cube(icechunk_s3_url, self.bucket_prefix)
        total_layers = cube.sizes[utils.Coords.TIME]

        queue, job_definition, num_workers = self.batch_queue, self.batch_job, \
            DeepCopyCubeBatch.NUM_LOAD_WORKERS
        if total_layers > DeepCopyCubeBatch.NUM_LAYERS_64GB_THRESHOLD:
            queue, job_definition, num_workers = self.batch_queue_64gb, \
                self.batch_job_64gb, DeepCopyCubeBatch.NUM_LOAD_WORKERS * 2

        logging.info(
            f'{icechunk_s3_url}: {total_layers} layers, using '
            f'queue={queue}, job_definition={job_definition}'
        )
        return queue, job_definition, num_workers

    def __call__(
        self,
        cube_file: str,
        job_file: str,
        num_cubes: int
    ):
        """
        Submit Batch jobs to AWS.
        """
        # List of submitted datacube Batch jobs and AWS response
        jobs = []

        # List of submitted datacubes for processing
        jobs_files = []

        with open(cube_file, 'r') as fhandle:
            cubes = json.load(fhandle)

            # Number of cubes to generate
            num_jobs = 0
            logging.info(f'Total number of datacubes: {len(cubes[GeoJsonVars.features])}')

            for each_cube in cubes[GeoJsonVars.features]:
                if num_cubes is not None and num_jobs == num_cubes:
                    # Number of datacubes to generate is provided,
                    # stop if they have been generated
                    logging.info(f'Reached number of cubes to process: {num_cubes}')
                    break

                properties = each_cube[GeoJsonVars.properties]

                roi = properties[GeoJsonVars.roi_percent_coverage]
                epsg_code = str(properties[GeoJsonVars.epsg])

                # Include only specific EPSG code(s) if specified
                if len(BatchVars.EPSG_TO_GENERATE) and \
                        epsg_code not in BatchVars.EPSG_TO_GENERATE:
                    continue

                # Exclude specific EPSG code(s) if specified
                if len(BatchVars.EPSG_TO_EXCLUDE) and \
                        epsg_code in BatchVars.EPSG_TO_EXCLUDE:
                    continue

                icechunk_filename = properties.get(GeoJsonVars.icechunk_filename)
                if not icechunk_filename:
                    raise RuntimeError(
                        f"Cube {properties.get('cube_id')} has no "
                        f"'{GeoJsonVars.icechunk_filename}' property; skipping"
                    )

                # Derive the deep-copy cube's filename from the virtual
                # (icechunk) one, same convention as
                # add_url_to_datacube_definition.py._check_zarr_cube().
                zarr_filename = icechunk_filename.replace(File.ext.icechunk, File.ext.zarr)
                logging.info(f'Cube name: {zarr_filename}')

                # A way to run specific jobs only
                if len(BatchVars.CUBES_TO_GENERATE) and zarr_filename not in BatchVars.CUBES_TO_GENERATE:
                    logging.info(f"Skipping {zarr_filename} as not provided in BatchVars.CUBES_TO_GENERATE")
                    continue

                if len(BatchVars.CUBES_TO_EXCLUDE) and zarr_filename in BatchVars.CUBES_TO_EXCLUDE:
                    logging.info(f"Skipping {zarr_filename}  as provided in BatchVars.CUBES_TO_EXCLUDE")
                    continue

                icechunk_exists, icechunk_s3_url = self._check_icechunk_cube_exists(icechunk_filename)
                if not icechunk_exists:
                    logging.warning(
                        f'Icechunk repo does not exist at {icechunk_s3_url}; '
                        f'skipping deep-copy for {zarr_filename}'
                    )
                    continue

                output_s3_url = _join_s3_url(self.zarr_cubes_location, zarr_filename)

                batch_queue, batch_job_definition, num_workers = self._pick_batch_resources(icechunk_s3_url)

                cube_params = {
                    'inputStore': icechunk_s3_url,
                    'outputStore': output_s3_url,
                    'localStagingDir': zarr_filename,
                    'progressDir': self.progress_dir,
                    'numLoadWorkers': str(num_workers),
                }

                logging.info(f'grep : {cube_params}')

                # Submit AWS Batch job
                response = None
                if self.is_dry_run is False:
                    # Aws job name can't include '.' character, remove
                    # file extension
                    response = DeepCopyCubeBatch.CLIENT.submit_job(
                        jobName=zarr_filename.replace(File.ext.zarr, ''),
                        jobQueue=batch_queue,
                        jobDefinition=batch_job_definition,
                        parameters=cube_params,
                        timeout={
                            # Change to 14 days to support very large cubes
                            'attemptDurationSeconds': 1209600
                        }

                    )

                    logging.info(f"Response: {response}")

                num_jobs += 1
                logging.info(f'Submitted {num_jobs} to AWS')

                jobs.append({
                    's3_filename': output_s3_url,
                    'roi_percent': roi,
                    'aws_params': cube_params,
                    'aws': {
                        'queue': batch_queue,
                        'job_definition': batch_job_definition,
                        'response': response
                    }
                })

                jobs_files.append(output_s3_url)

            logging.info(f"Number of batch jobs submitted: {num_jobs}")

            # Write job info to the json file
            logging.info(f"Writing AWS job info to the {job_file}...")
            with open(job_file, 'w') as output_fhandle:
                json.dump(jobs, output_fhandle, indent=4)

            # Write job files to the json file
            job_files_file = f'filenames_{job_file}'
            logging.info(f"Writing jobs output files to the {job_files_file}...")
            with open(job_files_file, 'w') as output_fhandle:
                json.dump(jobs_files, output_fhandle, indent=4)

            return


def main(
    dry_run: bool,
    cube_definition_file: str,
    batch_job: str,
    batch_job_64gb: str,
    batch_queue: str,
    batch_queue_64gb: str,
    icechunk_cubes_location: str,
    zarr_cubes_location: str,
    bucket_prefix: str,
    progress_dir: str,
    output_job_file: str,
    number_of_cubes: int
):
    """
    Driver to submit multiple Batch jobs to AWS.
    """
    # Submit Batch job to AWS for each datacube which has ROI!=0 and an
    # existing source icechunk repo
    run_batch = DeepCopyCubeBatch(
        batch_job,
        batch_job_64gb,
        batch_queue,
        batch_queue_64gb,
        icechunk_cubes_location,
        zarr_cubes_location,
        bucket_prefix,
        progress_dir,
        dry_run
    )
    run_batch(cube_definition_file, output_job_file, number_of_cubes)


def parse_args():
    """
    Create command-line argument parser and parse arguments.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Command-line arguments parser
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-c', '--cubeDefinitionFile',
        type=str,
        action='store',
        required=True,
        help="GeoJson file that stores chunk-aligned cube polygon definitions, with "
            "each feature's properties carrying 'icechunk_filename' (see "
            "define_cube_polygons_chunk_aligned.py)."
    )
    parser.add_argument(
        '--icechunkCubesLocation',
        type=str,
        action='store',
        default='s3://its-live-data/datacubes/spatial/v2.2',
        help="Flat S3 directory that holds the source virtual (icechunk) datacube "
            "repos [%(default)s]."
    )
    parser.add_argument(
        '--zarrCubesLocation',
        type=str,
        action='store',
        default="s3://its-live-data/datacubes/timeseries/v2.2/",
        help="Flat S3 directory to place the generated deep-copy (Zarr) datacubes in "
            "[%(default)s]."
    )
    parser.add_argument(
        '--progressDir',
        type=str,
        action='store',
        default="s3://its-live-data/test-space/datacubes/progress/v2.2/",
        help="s3:// directory for deep_copy_cube_per_var_chunk.py's --progress-dir "
            "resumability markers. One directory can serve a whole batch of cubes -- "
            "markers are namespaced per output store underneath it [%(default)s]."
    )
    parser.add_argument(
        '-j', '--batchJobDefinition',
        type=str,
        action='store',
        default='its-live-deep-copy-job',
        help="AWS Batch job definition to use for deep-copy datacube generation "
        "[%(default)s]."
    )
    parser.add_argument(
        '-q', '--batchJobQueue',
        type=str,
        action='store',
        default='its-live-deep-copy-queue',
        help="AWS Batch job queue to use for deep-copy datacube generation "
            "when the source icechunk repo's layer count is at or below "
            f"{DeepCopyCubeBatch.NUM_LAYERS_64GB_THRESHOLD} [%(default)s]."
    )
    parser.add_argument(
        '--batchJobQueue64Gb',
        type=str,
        action='store',
        default='its-live-deep-copy-64Gb-queue',
        help="AWS Batch job queue to use instead of --batchJobQueue when the "
            "source icechunk repo's layer count exceeds "
            f"{DeepCopyCubeBatch.NUM_LAYERS_64GB_THRESHOLD} [%(default)s]."
    )
    parser.add_argument(
        '--batchJobDefinition64Gb',
        type=str,
        action='store',
        default='its-live-deep-copy-64Gb-job',
        help="AWS Batch job definition to use instead of "
            "--batchJobDefinition when the source icechunk repo's layer "
            "count exceeds "
            f"{DeepCopyCubeBatch.NUM_LAYERS_64GB_THRESHOLD} -- must be paired "
            "with --batchJobQueue64Gb since a job definition's container "
            "memory limit is enforced regardless of which instance the "
            "queue places it on [%(default)s]."
    )
    parser.add_argument(
        '--bucketPrefix',
        type=str,
        action='store',
        default='s3://its-live-data/',
        help="S3 URL prefix the source icechunk repo's virtual chunk "
            "container resolves granule references against -- same role as "
            "deep_copy_cube_per_var_chunk.py's --bucket, needed to open each "
            "repo and read its layer count [%(default)s]."
    )
    parser.add_argument(
        '--numLoadWorkers',
        type=int,
        action='store',
        default=32,
        help="Thread-pool size per batch .load() call within each job "
            "(deep_copy_cube_per_var_chunk.py's --num-load-workers) [%(default)d]"
    )
    parser.add_argument(
        '-o', '--outputJobFile',
        type=str,
        action='store',
        default='deep_copy_datacube_batch_jobs.json',
        help="File to capture submitted deep-copy datacube AWS Batch jobs [%(default)s]"
    )
    parser.add_argument(
        '-e', '--epsgCode',
        type=str,
        action='store',
        default=None,
        help="JSON list to specify EPSG codes of interest for the datacubes to generate [%(default)s]"
    )
    parser.add_argument(
        '--dryrun',
        action='store_true',
        help='Dry run, do not actually submit any AWS Batch jobs'
    )
    parser.add_argument(
        '-n', '--numberOfCubes',
        type=int,
        action='store',
        default=-1,
        help="Number of datacubes to generate [%(default)d]. If left at "
            "default value, then generate all qualifying datacubes."
    )
    parser.add_argument(
        '--excludeCubesFile',
        type=Path,
        nargs='+',
        default=None,
        help="One or more json files, each storing a list of datacubes to exclude from "
            "processing [%(default)s]. Lists from all files are combined and de-duplicated."
    )
    parser.add_argument(
        '--excludeEPSG',
        type=str,
        action='store',
        default=None,
        help="JSON list of EPSG codes to exclude from the datacube generation [%(default)s]"
    )

    # One of --processCubes or --processCubesFile options is allowed for the datacube names
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--processCubes',
        type=str,
        action='store',
        default='[]',
        help="JSON list of filenames to generate [%(default)s]."
    )
    group.add_argument(
        '--processCubesFile',
        type=Path,
        action='store',
        default=None,
        help="File that contains JSON list of filenames for datacube to generate [%(default)s]."
    )

    args = parser.parse_args()
    logging.info(f"Command-line arguments: {sys.argv}")
    logging.info(f"Parsed out command-line arguments: {args}")

    DeepCopyCubeBatch.NUM_LOAD_WORKERS = args.numLoadWorkers

    epsg_codes = list(map(str, json.loads(args.epsgCode))) if args.epsgCode is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes: {epsg_codes}, ignoring all other EPGS codes")
        BatchVars.EPSG_TO_GENERATE = epsg_codes

    epsg_codes = list(map(str, json.loads(args.excludeEPSG))) if args.excludeEPSG is not None else None
    if epsg_codes and len(epsg_codes):
        logging.info(f"Got EPSG codes to exclude: {epsg_codes}")
        BatchVars.EPSG_TO_EXCLUDE = epsg_codes

    # Make sure there is no overlap in EPSG_TO_GENERATE and EPSG_TO_EXCLUDE
    diff = set(BatchVars.EPSG_TO_GENERATE).intersection(BatchVars.EPSG_TO_EXCLUDE)
    if len(diff):
        raise RuntimeError(f"The same code is specified for BatchVars.EPSG_TO_EXCLUDE={BatchVars.EPSG_TO_EXCLUDE} and BatchVars.EPSG_TO_GENERATE={BatchVars.EPSG_TO_GENERATE}")

    if args.processCubesFile:
        # Check for this option first as another mutually exclusive option has a default value
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubesFile.read_text())
        # Replace each path by the datacube basename
        BatchVars.CUBES_TO_GENERATE = [os.path.basename(each) for each in BatchVars.CUBES_TO_GENERATE if len(each)]
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

        # Make sure all datacubes are unique
        BatchVars.CUBES_TO_GENERATE = list(set(BatchVars.CUBES_TO_GENERATE))
        logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} unique datacubes to generate from {args.processCubesFile}: {BatchVars.CUBES_TO_GENERATE}")

    elif args.processCubes:
        BatchVars.CUBES_TO_GENERATE = json.loads(args.processCubes)
        if len(BatchVars.CUBES_TO_GENERATE):
            logging.info(f"Found {len(BatchVars.CUBES_TO_GENERATE)} of datacubes to generate from {args.processCubes}: {BatchVars.CUBES_TO_GENERATE}")

    if args.excludeCubesFile:
        exclude_cubes = []
        for each_file in args.excludeCubesFile:
            cubes_from_file = json.loads(each_file.read_text())
            logging.info(f"Found {len(cubes_from_file)} of datacubes to exclude per {each_file}")
            exclude_cubes.extend(cubes_from_file)

        # Replace each path by the datacube basename
        exclude_cubes = [os.path.basename(each) for each in exclude_cubes if len(each)]

        # Make sure all datacubes are unique
        BatchVars.CUBES_TO_EXCLUDE = list(set(exclude_cubes))
        logging.info(
            f"Found {len(BatchVars.CUBES_TO_EXCLUDE)} unique datacubes to exclude from "
            f"{len(args.excludeCubesFile)} file(s): {BatchVars.CUBES_TO_EXCLUDE}"
        )

    return args


if __name__ == '__main__':

    args = parse_args()

    main(
        args.dryrun,
        args.cubeDefinitionFile,
        args.batchJobDefinition,
        args.batchJobDefinition64Gb,
        args.batchJobQueue,
        args.batchJobQueue64Gb,
        args.icechunkCubesLocation,
        args.zarrCubesLocation,
        args.bucketPrefix,
        args.progressDir,
        args.outputJobFile,
        args.numberOfCubes
    )

    logging.info(f"Done")
