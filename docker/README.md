# Docker Images

The ITS_LIVE deep-copy cube pipeline uses two Docker images, each driven by its own
build script:

- **Virtual datacube** -- driven by `build_virtual_datacube.sh`, which builds from
  `Dockerfile_virtual_datacube` an icechunk-backed virtual datacube image from
  granules (`virtual_itslive_cube_per_chunk.py`).
- **Deep-copy datacube** -- driven by `build_deep_copy_cube.sh`, which builds from
  `Dockerfile_deep_copy_cube` an image that materializes a virtual datacube into a
  real Zarr v3 store, creating or updating it (`deep_copy_cube_per_var_chunk.py`).

## Prerequisites

- Docker installed and running.
- Run the build scripts from the **repository root**, or set `WORKSPACE` to the repo
  root explicitly -- both scripts default `WORKSPACE` to one directory up from the
  script's own location (`docker/..`), but copy source files via `${WORKSPACE}/src/...`
  paths that assume that root.

## Building the virtual datacube image

```bash
./docker/build_virtual_datacube.sh [TAG]
```

- `TAG` defaults to `$USER` if omitted.

## Building the deep-copy datacube image

```bash
./docker/build_deep_copy_cube.sh [TAG]
```

- `TAG` defaults to `$USER` if omitted.

## Pushing to ECR

The CDK app (`src/aws/cdk/cubes/app.py`) expects these images at:
- `<account-id>.dkr.ecr.us-west-2.amazonaws.com/its-live-icechunk-datacube:<tag>`
  (virtual datacube)
- `<account-id>.dkr.ecr.us-west-2.amazonaws.com/its-live-deep-copy-datacube:<tag>`
  (deep-copy datacube)
