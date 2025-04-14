#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################

# Release:	release-linux
# Debug:	dev-linux
export PRESET=release-linux

# Bindings
if [[ "$1" == "python" ]]
then
	export BINDINGS=ON
	export INSTALL_DIR=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	export BINDINGS=OFF
	export INSTALL_DIR=$(pwd)/examples/gprat_cpp
else
    echo "Please specify first input parameter: python/cpp"
    exit 1
fi

if [[ -z "$2" ]]; then
    echo "Second parameter is missing. Using default: cpu"
    cpu=1
    gpu=0
elif [[ "$2" == "cpu" ]]; then
    cpu=1
    gpu=0
elif [[ "$2" == "gpu" ]]; then
    cpu=0
    gpu=1
else
    echo "Please specify input parameter: cpu/gpu"
    exit 1
fi

if [[ $cpu -eq 1 ]]; then
    # Load GCC compiler
    module load gcc/14.1.0

    # Activate spack environment
    spack env activate gprat_cpu_gcc

    cmake --preset $PRESET \
        -DGPRAT_BUILD_BINDINGS=$BINDINGS \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
        -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
        -DGPRAT_WITH_CUDA=OFF

elif [[ $gpu -eq 1 ]]; then
    # Load Clang compiler and CUDA library
    module load clang/17.0.1
    module load cuda/12.0.1

    # Activate spack environment
    spack env activate gprat_gpu_clang

    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

    cmake --preset $PRESET \
        -DGPRAT_BUILD_BINDINGS=$BINDINGS \
        -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
        -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
        -DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
	-DCMAKE_C_COMPILER=$(which clang) \
	-DCMAKE_CXX_COMPILER=$(which clang++) \
	-DCMAKE_CUDA_COMPILER=$(which clang++) \
	-DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGPRAT_WITH_CUDA=ON
fi

################################################################################
# Compile code
################################################################################

cmake --build --preset $PRESET -- -j
cmake --install build/$PRESET

cd build/$PRESET
ctest --output-on-failure --no-tests=ignore -C Release -j 2
