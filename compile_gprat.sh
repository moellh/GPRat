#!/bin/bash
# $1: python/cpp
# $2: cpu/gpu
################################################################################
set -e  # Exit immediately if a command exits with a non-zero status.

################################################################################
# Configurations
################################################################################

# Release: release-linux
# Debug: dev-linux
# Release for GPU: release-linux-gpu
# Debug for GPU: dev-linux-gpu
export preset=release-linux

# Bindings
if [[ "$1" == "python" ]]
then
	bindings=ON
	install_dir=$(pwd)/examples/gprat_python
elif [[ "$1" == "cpp" ]]
then
	export bindings=OFF
	export install_dir=$(pwd)/examples/gprat_cpp
else
    echo "Please specify input parameter: python/cpp"
    exit 1
fi

if [[ $preset == "release-linux" || $preset == "dev-linux" ]]; then
    # Load GCC compiler
    module load gcc/14.1.0

    # Activate spack environment
    spack env activate gprat_cpu_gcc

elif [[ $preset == "release-linux-gpu" || $preset == "dev-linux-gpu" ]]; then
    # Load Clang compiler and CUDA library
    module load clang/17.0.1
    module load cuda/12.0.1

    # Activate spack environment
    spack env activate gprat_gpu_clang
fi

################################################################################
# Compile code
################################################################################

cmake --preset $preset \
    -DGPRAT_BUILD_BINDINGS=$bindings \
    -DCMAKE_INSTALL_PREFIX=$install_dir \
    -DHPX_IGNORE_BOOST_COMPATIBILITY=ON \
    -DGPRAT_ENABLE_FORMAT_TARGETS=OFF
cmake --build --preset $preset -- -j
cmake --install build/$preset

cd build/$preset
ctest --output-on-failure --no-tests=ignore -C Release -j 2
