#!/bin/bash

################################################################################
# Build GPXPy Python library on simcl1n1 or simcl1n2
#
# Some system specific notes:
# - uses module cuda/12.0.1 and clang/17.0.1
# - requires setup of spack environment gpxpy
# - uses Clang as compiler for C, C++, and CUDA
################################################################################

# Exit on error (non-zero status).
# Print each command before execution.
set -ex

# Load modules
module load clang/17.0.1
if $1 == "-DGPXPY_WITH_CUDA=ON"; then
    module load cuda/12.0.1
fi

./compile_gpxpy_python.sh "$@"
