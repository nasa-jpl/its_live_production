# AWS Batch Job Submission

The ITS_LIVE deep-copy cube pipeline submits AWS Batch jobs via two scripts:

- **Virtual datacube** -- `run_virtual_cube_batch.py`, which submits one Batch job
  per chunk-aligned datacube definition to build a virtual (icechunk) datacube
  (`virtual_itslive_cube_per_chunk.py`).
- **Deep-copy datacube** -- `run_deep_copy_cube_batch.py`, which submits one Batch
  job per datacube definition to materialize an existing virtual datacube into a
  real Zarr v3 store (`deep_copy_cube_per_var_chunk.py`).

## Submitting virtual datacube jobs

```bash
python src/aws/run_virtual_cube_batch.py
```

- `-c/--cubeDefinitionFile` defaults to
  `s3://its-live-data/datacubes/definition/datacubes_v2.2.json`; pass a local path
  to override it.
- `-o/--outputJobFile` (default `virtual_datacube_batch_jobs.json`) captures every
  submitted job (cube name, AWS Batch params/response); a second file,
  `filenames_<outputJobFile>`, captures just the list of cube S3 paths processed.
- `--dryrun` submits nothing, just reports what would be submitted.

## Submitting deep-copy datacube jobs

```bash
python src/aws/run_deep_copy_cube_batch.py
```

- `-c/--cubeDefinitionFile` defaults to
  `s3://its-live-data/datacubes/definition/datacubes_v2.2.json`; pass a local path
  to override it.
- `-o/--outputJobFile` (default `deep_copy_datacube_batch_jobs.json`) captures every
  submitted job (cube name, AWS Batch params/response); a second file,
  `filenames_<outputJobFile>`, captures just the list of cube S3 paths processed.
- `--dryrun` submits nothing, just reports what would be submitted.
