#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <hpx/future.hpp>

using tiled_matrix = std::vector<hpx::shared_future<double *>>;
using tiled_vector = std::vector<hpx::shared_future<double *>>;
using d_vector = double *;
using h_vector = std::vector<double>;
using fd_vector = hpx::shared_future<double *>;
using fh_vector = hpx::shared_future<std::vector<double>>;

#endif
