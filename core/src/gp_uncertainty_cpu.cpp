#include "gp_uncertainty_cpu.hpp"

#include <cmath>
#include <vector>

std::vector<double>
diag_posterior(
    const std::vector<double> &A,
    const std::vector<double> &B,
    std::size_t M)
{
    std::vector<double> tile;
    tile.reserve(M);

    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i] - B[i]);
    }

    return tile;
}

std::vector<double>
diag_tile(
    const std::vector<double> &A,
    std::size_t M)
{
    std::vector<double> tile;
    tile.reserve(M);

    for (std::size_t i = 0; i < M; ++i)
    {
        tile.push_back(A[i * M + i]);
    }

    return tile;
}
