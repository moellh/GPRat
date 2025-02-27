cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(gpxpy VERSION 0.1.0 LANGUAGES CXX CUDA)

# Option for GPU support with CUDA/cuBLAS
option(GPXPY_WITH_CUDA "Enable GPU support with CUDA/cuBLAS" OFF)
# Pass variable to C++ code
add_compile_definitions(GPXPY_WITH_CUDA=$<BOOL:${GPXPY_WITH_CUDA}>)

option(GPRAT_CHOLESKY_STEPS "Synchronizes between steps of cholesky function" OFF)
add_compile_definitions(GPRAT_CHOLESKY_STEPS=$<BOOL:${GPRAT_CHOLESKY_STEPS}>)

option(GPRAT_ASSEMBLY_ONLY "Only computes assembly of cholesky function" OFF)
add_compile_definitions(GPRAT_ASSEMBLY_ONLY=$<BOOL:${GPRAT_ASSEMBLY_ONLY}>)

option(GPRAT_PREDICT_STEPS "Synchronizes between steps of predict function" OFF)
add_compile_definitions(GPRAT_PREDICT_STEPS=$<BOOL:${GPRAT_PREDICT_STEPS}>)

option(GPRAT_PREDICT_UNCER_STEPS "Synchronizes between steps of predict_with_uncertainty function" OFF)
add_compile_definitions(GPRAT_PREDICT_UNCER_STEPS=$<BOOL:${GPRAT_PREDICT_UNCER_STEPS}>)

option(GPRAT_PREDICT_FULL_COV_STEPS "Synchronizes between steps of predict_with_full_cov function" OFF)
add_compile_definitions(GPRAT_PREDICT_FULL_COV_STEPS=$<BOOL:${GPRAT_PREDICT_FULL_COV_STEPS}>)

find_package(HPX REQUIRED)
find_package(MKL CONFIG REQUIRED)

# Include dir
include_directories(${HPX_INCLUDE_DIRS})
include_directories(${MKL_INCLUDE_DIRS})

# Source files
set(SOURCE_FILES
    src/adapter_mkl.cpp
    src/gp_algorithms_cpu.cpp
    src/gp_functions.cpp
    src/gp_kernel.cpp
    src/gp_hyperparameters.cpp
    src/gp_optimizer_cpu.cpp
    src/gp_uncertainty_cpu.cpp
    src/gpxpy_c.cpp
    src/target.cpp
    src/tiled_algorithms_cpu.cpp
    src/utils_c.cpp
)

if(GPXPY_WITH_CUDA)
    list(APPEND SOURCE_FILES
        src/gp_optimizer_gpu.cu
        src/gp_uncertainty_gpu.cu
        src/adapter_cublas.cu
        src/gp_algorithms_gpu.cu
        src/tiled_algorithms_gpu.cu
        src/cuda_kernels.cu
    )
endif()

# Headers (to be installed)
set(HEADER_FILES
    include/adapter_mkl.hpp
    include/gp_algorithms_cpu.hpp
    include/gp_functions.hpp
    include/apex_utils.hpp
    include/gp_kernels.hpp
    include/gp_hyperparameters.hpp
    include/gp_optimizer_cpu.hpp
    include/gp_uncertainty_cpu.hpp
    include/gpxpy_c.hpp
    include/target.hpp
    include/tiled_algorithms_cpu.hpp
    include/utils_c.hpp
)

if(GPXPY_WITH_CUDA)
    list(APPEND HEADER_FILES
        include/cuda_utils.cuh
        include/cuda_kernels.cuh
        include/gp_optimizer_gpu.cuh
        include/gp_uncertainty_gpu.cu
        include/adapter_cublas.cuh
        include/gp_algorithms_gpu.cuh
        include/tiled_algorithms_gpu.cuh
    )
endif()

# Add library
add_library(gpxpy SHARED ${SOURCE_FILES})

# Link HPX libraries
target_link_libraries(
    gpxpy
    PUBLIC HPX::hpx
    HPX::wrap_main
    HPX::iostreams_component
    MKL::mkl_intel_lp64
    MKL::mkl_core
    MKL::MKL
    MKL::mkl_sequential
    CUDA::cublas
    CUDA::cusolver
    atomic)

# Include directories
target_include_directories(
  gpxpy PRIVATE include/ ${HPX_INCLUDE_DIRS}
                $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)

# Set the RPATH to include the MKL library paths
set_target_properties(gpxpy PROPERTIES INSTALL_RPATH "${MKL_LIBRARIES}")

# Set default installation prefix if not specified by user
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  get_filename_component(PARENT_DIR "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
  set(CMAKE_INSTALL_PREFIX
      "${PARENT_DIR}/examples/gpxpy_cpp/install_cpp"
      CACHE PATH "Installation Directory" FORCE)
endif()

# Install
install(TARGETS gpxpy DESTINATION lib)

# Create base directory
install(DIRECTORY include/ DESTINATION include)
