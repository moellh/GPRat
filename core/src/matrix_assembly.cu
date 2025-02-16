#include "matrix_assembly.cuh"

#include "cuda_runtime.h"
#include "cuda_utils.cuh"
#include "matrix.hpp"

// Generate tiles {{{

// TODO: maybe remove later because of bad naming (look at calling code)
double *
gen_tile_zeros(
    std::size_t n_entries,
    gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;
    cudaStream_t stream = gpu.next_stream();
    check_cuda_error(cudaMalloc(&d_tile, n_entries * sizeof(double)));
    check_cuda_error(cudaMemsetAsync(d_tile, 0, n_entries * sizeof(double), stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    return d_tile;
}

double *
gen_matrix_tile_with_zeros(
    std::size_t n_tile_size,
    gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;
    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));
    check_cuda_error(cudaMemsetAsync(d_tile, 0, n_tile_size * sizeof(double), stream));
    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

// }}}

// Assemble tiles {{{

// TODO: maybe remove later because of bad naming (look at calling code)
std::vector<hpx::shared_future<double *>>
assemble_tiles_with_zeros(std::size_t n_tile_size, std::size_t n_tiles, gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        tiles[i] = hpx::async(&gen_tile_zeros, n_tile_size, std::ref(gpu));
    }
    return tiles;
}

tiled_matrix
assemble_matrix_tiles_with_zeros(
    std::size_t n_tile_size,
    std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    tiled_matrix tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        tiles[i] = hpx::async(&gen_tile_zeros, n_tile_size, std::ref(gpu));
    }
    return tiles;
}

// }}}
