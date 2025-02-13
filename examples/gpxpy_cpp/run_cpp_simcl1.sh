#!/bin/bash

################################################################################
# Run C++ test code on simcl1n1 or simcl1n2
################################################################################

# Exit on error (non-zero status).
# Print each command before executing it.
set -ex

# Load modules
module load clang/17.0.1
module load cuda/12.2.2

./run_cpp.sh "$@"
