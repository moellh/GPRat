#!/bin/bash

# Build GPXPy Python library

# Exit on error (non-zero status).
# Print each command before execution.
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

export CMAKE_OPTIONS="
    -DCMAKE_BUILD_TYPE=Debug
    -DHPX_WITH_DYNAMIC_HPX_MAIN=ON
    -DPYTHON_LIBRARY_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
    -DPYTHON_EXECUTABLE=$(which python3)
    -DCMAKE_C_COMPILER=$(which clang)
    -DCMAKE_CXX_COMPILER=$(which clang++)
    -DCMAKE_CUDA_COMPILER=$(which clang++)
    -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME}
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1
    ${MKL_CONFIG}"

# }}}

# Compile {{{

# Reset build directory
rm -rf build_gpxpy && mkdir build_gpxpy && cd build_gpxpy

# Configure project
$CMAKE_COMMAND .. ${CMAKE_OPTIONS} "$@"

# Build project
make -j all

# }}}
