#!/bin/bash

################################################################################
# Run C++ test code
################################################################################

# Exit on error (non-zero status).
# Print each command before executing it.
set -ex

# Configure {{{

# Load spack environment
source $HOME/spack/share/spack/setup-env.sh
spack env activate gpxpy

# Set cmake command
export CMAKE_COMMAND=$(which cmake)

# Configure MKL
export MKL_CONFIG='-DMKL_ARCH=intel64 -DMKL_LINK=dynamic -DMKL_INTERFACE_FULL=intel_lp64 -DMKL_THREADING=sequential'

# Get CUDA architecture
export CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

export CUDA_FLAGS="
    --cuda-gpu-arch=sm_${CUDA_ARCH}
"

export CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=Release
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON
    -DCMAKE_C_COMPILER=$(which clang)
    -DCMAKE_CXX_COMPILER=$(which clang++)
    -DCMAKE_CUDA_COMPILER=$(which clang++)
    -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME}
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
    ${MKL_CONFIG}
"

# }}}

# Compile {{{

# Reset build directory
rm -rf build && mkdir build && cd build

# Copy apex.conf if exists
if [ -f ../apex.conf ]; then
    cp ../apex.conf .
fi

# Configure project
$CMAKE_COMMAND .. $CMAKE_OPTIONS "$@"

# Build project
make -j VERBOSE=1 all

# }}}

# Run {{{

./gpxpy_cpp

# }}}
