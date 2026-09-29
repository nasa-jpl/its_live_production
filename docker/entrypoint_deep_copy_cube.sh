#!/bin/bash --login
set -e

PROGRAM_DIR=/home/conda/itslive
export PYTHONPATH=$PYTHONPATH:${PROGRAM_DIR}

python /home/conda/itslive/deep_copy_cube_per_var_chunk.py "$@"
