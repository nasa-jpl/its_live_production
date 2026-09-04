#!/bin/bash --login
set -e

PROGRAM_DIR=/home/conda/itslive
export PYTHONPATH=$PYTHONPATH:${PROGRAM_DIR}

python /home/conda/itslive/virtual_itslive_cube_per_chunk.py "$@"
