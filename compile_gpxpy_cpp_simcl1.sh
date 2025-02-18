#!/bin/bash

################################################################################
# Build GPXPy C++ library on simcl1n1 or simcl1n2
# - uses module cuda/12.2.2 and clang/17.0.1
# - assumes NVIDIA A30 GPU with compute capability 8.0
################################################################################

# Exit on error (non-zero status).
# Print each command before execution.
set -ex

# Load modules
module load clang/17.0.1
if $1 == "-DGPXPY_WITH_CUDA=ON"; then
    module load cuda/12.0.1
fi

./compile_gpxpy_cpp.sh "$@"
