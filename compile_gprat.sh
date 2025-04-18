#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
################################################################################
set -ex  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################

# Release: release-linux
# Debug: dev-linux
# Release for GPU: release-linux-gpu
# Debug for GPU: dev-linux-gpu
preset=release-linux-gpu

# Bindings
if [[ "$1" == "python" ]]
then
	bindings=ON
	install_dir=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	bindings=OFF
	install_dir=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

if [[ $preset == "release-linux" || $preset == "dev-linux" ]]; then
    # Load GCC compiler
    module load gcc/14.1.0

    # Activate spack environment
    spack env activate gprat_cpu_gcc

    cmake --preset $preset \
	-DGPRAT_BUILD_BINDINGS=$bindings \
	-DCMAKE_INSTALL_PREFIX=$install_dir \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF

elif [[ $preset == "release-linux-gpu" || $preset == "dev-linux-gpu" ]]; then
    # Load Clang compiler and CUDA library
    module load clang/17.0.1
    module load cuda/12.0.1

    # Activate spack environment
    spack env activate gprat_gpu_clang

    cuda_arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F '.' '{print $1$2}')

    cmake --preset $preset \
	-DGPRAT_BUILD_BINDINGS=$bindings \
	-DCMAKE_INSTALL_PREFIX=$install_dir \
	-DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
	-DGPRAT_ENABLE_FORMAT_TARGETS=OFF \
        -DCMAKE_C_COMPILER=$(which clang) \
        -DCMAKE_CXX_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_COMPILER=$(which clang++) \
        -DCMAKE_CUDA_FLAGS=--cuda-path=${CUDA_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch
fi

################################################################################
# Compile code
################################################################################

cmake --build --preset $preset -- -j
cmake --install build/$preset

cd build/$preset
ctest --output-on-failure --no-tests=ignore -C Release -j 2
