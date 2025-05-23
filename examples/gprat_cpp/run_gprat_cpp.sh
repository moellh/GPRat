#!/bin/bash
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.
#set -x  # Print each command before executing it.

################################################################################
# Configurations
################################################################################

if [[ -z "$1" ]]; then
    echo "Input parameter is missing. Using default: Run computations on CPU"
elif [[ "$2" == "gpu" ]]; then
    use_gpu="--use_gpu"
elif [[ "$2" != "cpu" ]]; then
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

if command -v spack &> /dev/null; then
    echo "Spack command found, checking for environments..."

    # Check if the gprat_cpu_gcc environment exists
    if spack env list | grep -q "gprat_cpu_gcc"; then
        echo "Found gprat_cpu_gcc environment, activating it."
        module load gcc/14.1.0
        spack env activate gprat_cpu_gcc
        GPRAT_WITH_CUDA=OFF # whether GPRAT_WITH_CUDA is ON of OFF is irrelevant for this example

    # Check if the gprat_gpu_clang environment exists
    elif spack env list | grep -q "gprat_gpu_clang"; then
        echo "Found gprat_gpu_clang environment, activating it."
        module load clang/17.0.1
        module load cuda/12.0.1
        spack env activate gprat_gpu_clang
        GPRAT_WITH_CUDA=ON
    else
        echo "Neither gprat_cpu_gcc nor gprat_gpu_clang environments found. Building example without Spack."
    fi
else
    echo "Spack command not found. Building example without Spack."
    # Assuming that Spack is not required on given system
fi

# Configure APEX
export APEX_SCREEN_OUTPUT=0
export APEX_DISABLE=1

################################################################################
# Compile code
################################################################################

rm -rf build && mkdir build && cd build

# Configure the project
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DGPRat_DIR=./lib/cmake/GPRat \
         -DGPRAT_WITH_CUDA=${GPRAT_WITH_CUDA}

# Build the project
make -j

################################################################################
# Run code
################################################################################

./gprat_cpp $use_gpu
