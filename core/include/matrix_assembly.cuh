#ifndef ASSEMBLE_MATRIX_CUH
#define ASSEMBLE_MATRIX_CUH

#include "target.hpp"
#include <hpx/future.hpp>
#include <vector>

// Generate tiles {{{

double *
gen_tile_zeros(
    std::size_t n_entries,
    gpxpy::CUDA_GPU &gpu);

// }}}

// Assemble tiled matrices or vectors {{{

std::vector<hpx::shared_future<double *>>
assemble_tiles_with_zeros(
    std::size_t n_tile_size,
    std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu);

// }}}

#endif
