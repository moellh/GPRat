#include "gp_algorithms_gpu.cuh"

#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"
#include "gp_kernels.hpp"
#include "gp_optimizer_gpu.cuh"
#include "target.hpp"
#include "tiled_algorithms_gpu.cuh"
#include <cuda_runtime.h>
#include <hpx/algorithm.hpp>
#include <hpx/async_cuda/cuda_exception.hpp>

using hpx::experimental::for_loop;

namespace gpu
{

// Kernel function to compute covariance
__global__ void
gen_tile_covariance_kernel(double *d_tile,
                           const double *d_input,
                           const std::size_t n_tile_size,
                           const std::size_t n_regressors,
                           const std::size_t tile_row,
                           const std::size_t tile_column,
                           const gpxpy_hyper::SEKParams sek_params)
{
    // Compute the global indices of the thread
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            int offset = -n_regressors + 1 + k;
            int i_local = i_global + offset;
            int j_local = j_global + offset;

            double z_ik = (i_local >= 0) ? d_input[i_local] : 0.0;
            double z_jk = (j_local >= 0) ? d_input[j_local] : 0.0;
            distance += (z_ik - z_jk) * (z_ik - z_jk);
        }

        // Compute the covariance value
        double covariance = sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        // Add noise variance if diagonal
        if (i_global == j_global)
        {
            covariance += sek_params.noise_variance;
        }

        d_tile[i * n_tile_size + j] = covariance;
    }
}

double *
gen_tile_covariance(const double *d_input,
                    const std::size_t tile_row,
                    const std::size_t tile_column,
                    const std::size_t n_tile_size,
                    const std::size_t n_regressors,
                    const gpxpy_hyper::SEKParams sek_params,
                    gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * n_tile_size * sizeof(double)));
    gen_tile_covariance_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void
gen_tile_full_prior_covariance_kernel(double *d_tile,
                                      const double *d_input,
                                      const std::size_t n_tile_size,
                                      const std::size_t n_regressors,
                                      const std::size_t tile_row,
                                      const std::size_t tile_column,
                                      const gpxpy_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + j;

        double distance = 0.0;
        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            int offset = -n_regressors + 1 + k;
            int i_local = i_global + offset;
            int j_local = j_global + offset;

            double z_ik = (i_local >= 0) ? d_input[i_local] : 0.0;
            double z_jk = (j_local >= 0) ? d_input[j_local] : 0.0;
            distance += (z_ik - z_jk) * (z_ik - z_jk);
        }

        double covariance = sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        d_tile[i * n_tile_size + j] = covariance;
    }
}

double *
gen_tile_full_prior_covariance(const double *d_input,
                               const std::size_t tile_row,
                               const std::size_t tile_colums,
                               const std::size_t n_tile_size,
                               const std::size_t n_regressors,
                               const gpxpy_hyper::SEKParams sek_params,
                               gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * n_tile_size * sizeof(double)));
    gen_tile_full_prior_covariance_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_colums, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void
gen_tile_prior_covariance_kernel(double *d_tile,
                                 const double *d_input,
                                 const std::size_t n_tile_size,
                                 const std::size_t n_regressors,
                                 const std::size_t tile_row,
                                 const std::size_t tile_column,
                                 const gpxpy_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_tile_size)
    {
        std::size_t i_global = n_tile_size * tile_row + i;
        std::size_t j_global = n_tile_size * tile_column + i;

        double distance = 0.0;
        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            int offset = -n_regressors + 1 + k;
            int i_local = i_global + offset;
            int j_local = j_global + offset;

            double z_ik = (i_local >= 0) ? d_input[i_local] : 0.0;
            double z_jk = (j_local >= 0) ? d_input[j_local] : 0.0;
            distance += (z_ik - z_jk) * (z_ik - z_jk);
        }

        double covariance = sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        d_tile[i] = covariance;
    }
}

double *
gen_tile_prior_covariance(const double *d_input,
                          const std::size_t tile_row,
                          const std::size_t tile_column,
                          const std::size_t n_tile_size,
                          const std::size_t n_regressors,
                          const gpxpy_hyper::SEKParams sek_params,
                          gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(256);
    dim3 n_blocks((n_tile_size + 255) / 256);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));
    gen_tile_prior_covariance_kernel<<<n_blocks, threads_per_block, 0, stream>>>(d_tile, d_input, n_tile_size, n_regressors, tile_row, tile_column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

__global__ void
gen_tile_cross_covariance_kernel(double *d_tile,
                                 const double *d_row_input,
                                 const double *d_col_input,
                                 const std::size_t n_row_tile_size,
                                 const std::size_t n_column_tile_size,
                                 const std::size_t tile_row,
                                 const std::size_t tile_column,
                                 const std::size_t n_regressors,
                                 const gpxpy_hyper::SEKParams sek_params)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_row_tile_size && j < n_column_tile_size)
    {
        std::size_t i_global = n_row_tile_size * tile_row + i;
        std::size_t j_global = n_column_tile_size * tile_column + j;

        double z_ik = 0.0;
        double z_jk = 0.0;
        double distance = 0.0;

        for (std::size_t k = 0; k < n_regressors; ++k)
        {
            int offset = -n_regressors + 1 + k;
            int i_local = i_global + offset;
            int j_local = j_global + offset;

            if (i_local >= 0)
            {
                z_ik = d_row_input[i_local];
            }
            if (j_local >= 0)
            {
                z_jk = d_col_input[j_local];
            }
            distance += (z_ik - z_jk) * (z_ik - z_jk);
        }

        double covariance = sek_params.vertical_lengthscale * exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));
        d_tile[i * n_column_tile_size + j] = covariance;
    }
}

double *
gen_tile_cross_covariance(const double *d_row_input,
                          const double *d_col_input,
                          const std::size_t tile_row,
                          const std::size_t tile_column,
                          const std::size_t n_row_tile_size,
                          const std::size_t n_column_tile_size,
                          const std::size_t n_regressors,
                          const gpxpy_hyper::SEKParams sek_params,
                          gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;

    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_column_tile_size + 15) / 16, (n_row_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    check_cuda_error(cudaMalloc(&d_tile, n_row_tile_size * n_column_tile_size * sizeof(double)));
    gen_tile_cross_covariance_kernel<<<n_blocks, threads_per_block, 0, stream>>>(d_tile, d_row_input, d_col_input, n_row_tile_size, n_column_tile_size, tile_row, tile_column, n_regressors, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

hpx::shared_future<double *>
gen_tile_cross_cov_T(
    std::size_t n_row_tile_size,
    std::size_t n_column_tile_size,
    const hpx::shared_future<double *> f_cross_covariance_tile,
    gpxpy::CUDA_GPU &gpu)
{
    double *transposed;
    check_cuda_error(cudaMalloc(&transposed, n_row_tile_size * n_column_tile_size * sizeof(double)));
    double *d_cross_covariance_tile = f_cross_covariance_tile.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((n_column_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_row_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose<<<n_blocks, threads_per_block, 0, stream>>>(transposed, d_cross_covariance_tile, n_row_tile_size, n_column_tile_size);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(transposed);
}

__global__ void
gen_tile_output_kernel(double *tile,
                       const double *output,
                       std::size_t row,
                       std::size_t n_tile_size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_tile_size)
    {
        std::size_t i_global = n_tile_size * row + i;
        tile[i] = output[i_global];
    }
}

double *
gen_tile_output(const std::size_t row,
                const std::size_t n_tile_size,
                const double *d_output,
                gpxpy::CUDA_GPU &gpu)
{
    dim3 threads_per_block(256);
    dim3 n_blocks((n_tile_size + 255) / 256);

    cudaStream_t stream = gpu.next_stream();

    double *d_tile;
    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));

    gen_tile_output_kernel<<<n_blocks, threads_per_block>>>(d_tile, d_output, row, n_tile_size);

    check_cuda_error(cudaStreamSynchronize(stream));

    return d_tile;
}

double *gen_tile_zeros(std::size_t n_tile_size, gpxpy::CUDA_GPU &gpu)
{
    double *d_tile;
    cudaStream_t stream = gpu.next_stream();
    check_cuda_error(cudaMalloc(&d_tile, n_tile_size * sizeof(double)));
    check_cuda_error(cudaMemsetAsync(d_tile, 0, n_tile_size * sizeof(double), stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    return d_tile;
}

double compute_error_norm(std::size_t n_tiles,
                          std::size_t n_tile_size,
                          const std::vector<double> &b,
                          const std::vector<std::vector<double>> &tiles)
{
    double error = 0.0;
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        auto a = tiles[k];
        for (std::size_t i = 0; i < n_tile_size; i++)
        {
            std::size_t i_global = n_tile_size * k + i;
            // ||a - b||_2
            error += (b[i_global] - a[i]) * (b[i_global] - a[i]);
        }
    }
    return sqrt(error);
}

std::vector<hpx::shared_future<double *>>
assemble_tiled_covariance_matrix(const double *d_training_input,
                                 const std::size_t n_tiles,
                                 const std::size_t n_tile_size,
                                 const std::size_t n_regressors,
                                 const gpxpy_hyper::SEKParams sek_params,
                                 gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_tiles(n_tiles * n_tiles);

    // clang-format off
    for_loop(hpx::execution::par, 0, n_tiles, [&](std::size_t tile_row)
    {
        for_loop(hpx::execution::par, 0, tile_row + 1, [&](std::size_t tile_column)
        {
            d_tiles[tile_row * n_tiles + tile_column] = hpx::async(&gen_tile_covariance, d_training_input, tile_row, tile_column, n_tile_size, n_regressors, sek_params, std::ref(gpu));
        });
    });
    // clang-format on

    return d_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_alpha_tiles(const double *d_output,
                     const std::size_t n_tiles,
                     const std::size_t n_tile_size,
                     gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> alpha_tiles(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(hpx::annotated_function(&gen_tile_output, "assemble_tiled_alpha"), i, n_tile_size, d_output, std::ref(gpu));
    }

    return alpha_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_cross_covariance_tiles(const double *d_test_input,
                                const double *d_training_input,
                                const std::size_t m_tiles,
                                const std::size_t n_tiles,
                                const std::size_t m_tile_size,
                                const std::size_t n_tile_size,
                                const std::size_t n_regressors,
                                const gpxpy_hyper::SEKParams sek_params,
                                gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> cross_covariance_tiles;
    cross_covariance_tiles.resize(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            cross_covariance_tiles[i * n_tiles + j] = hpx::async(hpx::annotated_function(&gen_tile_cross_covariance, "assemble_pred"), d_test_input, d_training_input, i, j, m_tile_size, n_tile_size, n_regressors, sek_params, std::ref(gpu));
        }
    }
    return cross_covariance_tiles;
}

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

std::vector<hpx::shared_future<double *>>
assemble_prior_K_tiles(const double *d_test_input,
                       const std::size_t m_tiles,
                       const std::size_t m_tile_size,
                       const std::size_t n_regressors,
                       const gpxpy_hyper::SEKParams sek_params,
                       gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles;
    d_prior_K_tiles.resize(m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        d_prior_K_tiles[i] = hpx::async(
            &gen_tile_prior_covariance,
            d_test_input,
            i,
            i,
            m_tile_size,
            n_regressors,
            sek_params,
            std::ref(gpu));
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_prior_K_tiles_full(const double *d_test_input,
                            const std::size_t m_tiles,
                            const std::size_t m_tile_size,
                            const std::size_t n_regressors,
                            const gpxpy_hyper::SEKParams sek_params,
                            gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_prior_K_tiles(m_tiles * m_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            d_prior_K_tiles[i * m_tiles + j] = hpx::async(
                &gen_tile_full_prior_covariance,
                d_test_input,
                i,
                j,
                m_tile_size,
                n_regressors,
                sek_params,
                std::ref(gpu));

            if (i != j)
            {
                d_prior_K_tiles[j * m_tiles + i] = hpx::dataflow(
                    &gen_tile_grad_l_trans,
                    m_tile_size,
                    d_prior_K_tiles[i * m_tiles + j],
                    std::ref(gpu));
            }
        }
    }
    return d_prior_K_tiles;
}

std::vector<hpx::shared_future<double *>>
assemble_t_cross_covariance_tiles(
    const std::vector<hpx::shared_future<double *>> &d_cross_covariance_tiles,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    gpxpy::CUDA_GPU &gpu)
{
    std::vector<hpx::shared_future<double *>> d_t_cross_covariance_tiles(m_tiles * n_tiles);
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            d_t_cross_covariance_tiles[j * m_tiles + i] = hpx::dataflow(
                &gen_tile_cross_cov_T,
                m_tile_size,
                n_tile_size,
                d_cross_covariance_tiles[i * n_tiles + j],
                std::ref(gpu));
        }
    }
    return d_t_cross_covariance_tiles;
}

std::vector<double>
copy_tiled_vector_to_host_vector(
    std::vector<hpx::shared_future<double *>> &d_tiles,
    std::size_t n_tile_size,
    std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    std::vector<double> h_vector(n_tiles * n_tile_size);
    std::vector<cudaStream_t> streams(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        streams[i] = gpu.next_stream();
        check_cuda_error(cudaMemcpyAsync(h_vector.data() + i * n_tile_size, d_tiles[i].get(), n_tile_size * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
    }
    gpu.sync_streams(streams);
    return h_vector;
}

std::vector<std::vector<double>>
move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    std::vector<std::vector<double>> h_tiles(n_tiles * n_tiles);

    std::vector<cudaStream_t> streams(n_tiles * (n_tiles + 1) / 2);
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            streams[i] = gpu.next_stream();
            h_tiles[i * n_tiles + j].resize(n_tile_size * n_tile_size);
            check_cuda_error(cudaMemcpyAsync(h_tiles[i * n_tiles + j].data(), d_tiles[i * n_tiles + j].get(), n_tile_size * n_tile_size * sizeof(double), cudaMemcpyDeviceToHost, streams[i]));
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
    gpu.sync_streams(streams);

    return h_tiles;
}

void free_lower_tiled_matrix(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            check_cuda_error(cudaFree(d_tiles[i * n_tiles + j].get()));
        }
    }
}

hpx::shared_future<std::vector<double>>
predict(const std::vector<double> &h_training_input,
        const std::vector<double> &h_training_output,
        const std::vector<double> &h_test_input,
        const std::size_t n_tiles,
        const std::size_t n_tile_size,
        const std::size_t m_tiles,
        const std::size_t m_tile_size,
        const std::size_t n_regressors,
        const gpxpy_hyper::SEKParams sek_params,
        gpxpy::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    auto d_tiles = assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);
    auto alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);
    auto cross_covariance_tiles = assemble_cross_covariance_tiles(d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);
    auto prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_tiles, alpha_tiles, n_tile_size, n_tiles, gpu);

    prediction_tiled(cross_covariance_tiles, alpha_tiles, prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(prediction_tiles, m_tile_size, m_tiles, gpu);

    free_lower_tiled_matrix(d_tiles, n_tiles);
    free(alpha_tiles);
    free(cross_covariance_tiles);
    free(prediction_tiles);
    destroy(cusolver);

    gpu.destroy();

    return hpx::make_ready_future(prediction);
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_uncertainty(const std::vector<double> &h_training_input,
                         const std::vector<double> &h_training_output,
                         const std::vector<double> &h_test_input,
                         const std::size_t n_tiles,
                         const std::size_t n_tile_size,
                         const std::size_t m_tiles,
                         const std::size_t m_tile_size,
                         const std::size_t n_regressors,
                         const gpxpy_hyper::SEKParams sek_params,
                         gpxpy::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles = assemble_prior_K_tiles(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predictions
    prediction_tiled(d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    posterior_covariance_tiled(d_t_cross_covariance_tiles, d_prior_inter_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predicition uncertainty
    prediction_uncertainty_tiled(d_prior_K_tiles, d_prior_inter_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full = copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free(d_prior_K_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prior_inter_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    return hpx::make_ready_future(std::vector<std::vector<double>>{ prediction, pred_var_full });
}

hpx::shared_future<std::vector<std::vector<double>>>
predict_with_full_cov(const std::vector<double> &h_training_input,
                      const std::vector<double> &h_training_output,
                      const std::vector<double> &h_test_input,
                      int n_tiles,
                      int n_tile_size,
                      int m_tiles,
                      int m_tile_size,
                      int n_regressors,
                      gpxpy_hyper::SEKParams sek_params,
                      gpxpy::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    double *d_training_output = copy_to_device(h_training_output, gpu);
    double *d_test_input = copy_to_device(h_test_input, gpu);

    // Assemble tiled covariance matrix on GPU.
    auto d_K_tiles = assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    auto d_alpha_tiles = assemble_alpha_tiles(d_training_output, n_tiles, n_tile_size, gpu);

    auto d_prior_K_tiles = assemble_prior_K_tiles_full(d_test_input, m_tiles, m_tile_size, n_regressors, sek_params, gpu);

    auto d_cross_covariance_tiles = assemble_cross_covariance_tiles(d_test_input, d_training_input, m_tiles, n_tiles, m_tile_size, n_tile_size, n_regressors, sek_params, gpu);

    auto d_t_cross_covariance_tiles = assemble_t_cross_covariance_tiles(d_cross_covariance_tiles, n_tiles, m_tiles, n_tile_size, m_tile_size, gpu);

    // Assemble placeholder matrix for diag(K_MxN * (K^-1_NxN * K_NxM))
    auto d_prior_inter_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    auto d_prediction_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    // Assemble placeholder for uncertainty
    auto d_prediction_uncertainty_tiles = assemble_tiles_with_zeros(m_tile_size, m_tiles, gpu);

    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_K_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);
    backward_solve_tiled(d_K_tiles, d_alpha_tiles, n_tile_size, n_tiles, gpu);

    // Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
    forward_solve_KcK_tiled(d_K_tiles, d_t_cross_covariance_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predictions
    prediction_tiled(d_cross_covariance_tiles, d_alpha_tiles, d_prediction_tiles, m_tile_size, n_tile_size, n_tiles, m_tiles, gpu);

    // posterior covariance matrix - (K_MxN * K^-1_NxN) * K_NxM
    full_cov_tiled(d_t_cross_covariance_tiles, d_prior_K_tiles, n_tile_size, m_tile_size, n_tiles, m_tiles, gpu);

    // Compute predicition uncertainty
    pred_uncer_tiled(d_prior_K_tiles, d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    // Get predictions and uncertainty to return them
    std::vector<double> prediction = copy_tiled_vector_to_host_vector(d_prediction_tiles, m_tile_size, m_tiles, gpu);
    std::vector<double> pred_var_full = copy_tiled_vector_to_host_vector(d_prediction_uncertainty_tiles, m_tile_size, m_tiles, gpu);

    check_cuda_error(cudaFree(d_training_input));
    check_cuda_error(cudaFree(d_training_output));
    check_cuda_error(cudaFree(d_test_input));
    free_lower_tiled_matrix(d_K_tiles, n_tiles);
    free(d_alpha_tiles);
    free_lower_tiled_matrix(d_prior_K_tiles, m_tiles);
    free(d_cross_covariance_tiles);
    free(d_t_cross_covariance_tiles);
    free(d_prior_inter_tiles);
    free(d_prediction_tiles);
    free(d_prediction_uncertainty_tiles);
    destroy(cusolver);

    gpu.destroy();

    return hpx::make_ready_future(std::vector<std::vector<double>>{ prediction, pred_var_full });
}

hpx::shared_future<double>
compute_loss(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             const std::size_t n_tiles,
             const std::size_t n_tile_size,
             const std::size_t n_regressors,
             const gpxpy_hyper::SEKParams sek_params,
             gpxpy::CUDA_GPU &gpu)
{
    /* // declare data structures
    // tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    hpx::shared_future<double> loss_value;

    //////////////////////////////////////////////////////////////////////////////
    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }

    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }

    //////////////////////////////////////////////////////////////////////////////
    // Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    // Compute loss
    compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
    // Return loss
    return loss_value; */
    return hpx::shared_future<double>();
}

hpx::shared_future<std::vector<double>>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         const std::size_t n_tiles,
         const std::size_t n_tile_size,
         const std::size_t n_regressors,
         const gpxpy_hyper::SEKParams &sek_params,
         const std::vector<bool> trainable_params,
         const gpxpy_hyper::AdamParams &adam_params,
         gpxpy::CUDA_GPU &gpu)
{
    /* // declaretiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // data holder for computed loss values
    std::vector<double> losses;
    losses.resize(adam_params.opt_iter);
    //////////////////////////////////////////////////////////////////////////////
    // Assemble beta1_t and beta2_t
    beta1_T.resize(adam_params.opt_iter);
    for (int i = 0; i < adam_params.opt_iter; i++)
    {
        beta1_T[i] = hpx::async(
            hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, adam_params.beta1);
    }
    beta2_T.resize(adam_params.opt_iter);
    for (int i = 0; i < adam_params.opt_iter; i++)
    {
        beta2_T[i] = hpx::async(
            hpx::annotated_function(&gen_beta_T, "assemble_tiled"), i + 1, adam_params.beta2);
    }
    // Assemble first and second momemnt vectors: m_T and v_T
    m_T.resize(3);
    v_T.resize(3);
    for (int i = 0; i < 3; i++)
    {
        m_T[i] = hpx::async(
            hpx::annotated_function(&gen_moment, "assemble_tiled"));
        v_T[i] = hpx::async(
            hpx::annotated_function(&gen_moment, "assemble_tiled"));
    }

    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_y"), i, n_tile_size, training_output);
    }

    // Perform optimization
    for (int iter = 0; iter < adam_params.opt_iter; iter++)
    {
        // Assemble covariance matrix vector, derivative of covariance
        // matrix vector w.r.t. to vertical lengthscale and derivative of
        // covariance matrix vector w.r.t. to lengthscale
        K_tiles.resize(n_tiles * n_tiles);
        grad_v_tiles.resize(n_tiles * n_tiles);
        grad_l_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j <= i; j++)
            {
                hpx::shared_future<std::vector<double>> cov_dists =
                    hpx::async(
                        hpx::annotated_function(&compute_cov_dist_vec,
                                                "assemble_cov_dist"),
                        i,
                        j,
                        n_tile_size,
                        n_regressors,
                        sek_params,
                        training_input);

                K_tiles[i * n_tiles + j] = hpx::dataflow(
                    hpx::annotated_function(
                        hpx::unwrapping(&gen_tile_covariance_opt),
                        "assemble_K"),
                    i,
                    j,
                    n_tile_size,
                    n_regressors,
                    sek_params,
                    cov_dists);

                grad_v_tiles[i * n_tiles + j] =
                    hpx::dataflow(hpx::annotated_function(
                                      hpx::unwrapping(&gen_tile_grad_v),
                                      "assemble_gradv"),
                                  i,
                                  j,
                                  n_tile_size,
                                  n_regressors,
                                  sek_params,
                                  cov_dists);

                grad_l_tiles[i * n_tiles + j] =
                    hpx::dataflow(hpx::annotated_function(
                                      hpx::unwrapping(&gen_tile_grad_l),
                                      "assemble_gradl"),
                                  i,
                                  j,
                                  n_tile_size,
                                  n_regressors,
                                  sek_params,
                                  cov_dists);

                if (i != j)
                {
                    grad_v_tiles[j * n_tiles + i] = hpx::dataflow(
                        hpx::annotated_function(
                            hpx::unwrapping(&gen_tile_grad_v_trans),
                            "assemble_gradv_t"),
                        n_tile_size,
                        grad_v_tiles[i * n_tiles + j]);

                    grad_l_tiles[j * n_tiles + i] = hpx::dataflow(
                        hpx::annotated_function(
                            hpx::unwrapping(&gen_tile_grad_l_trans),
                            "assemble_gradl_t"),
                        n_tile_size,
                        grad_l_tiles[i * n_tiles + j]);
                }
            }
        }
        // Assemble placeholder matrix for K^-1 * (I - y*y^T*K^-1)
        grad_K_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j < n_tiles; j++)
            {
                grad_K_tiles[i * n_tiles + j] =
                    hpx::async(hpx::annotated_function(&gen_tile_identity,
                                                       "assemble_tiled"),
                               i,
                               j,
                               n_tile_size);
            }
        }
        // Assemble alpha
        alpha_tiles.resize(n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            alpha_tiles[i] = hpx::async(
                hpx::annotated_function(&gen_tile_zeros, "assemble_tiled"),
                n_tile_size);
        }
        // Assemble placeholder matrix for K^-1
        grad_I_tiles.resize(n_tiles * n_tiles);
        for (std::size_t i = 0; i < n_tiles; i++)
        {
            for (std::size_t j = 0; j < n_tiles; j++)
            {
                grad_I_tiles[i * n_tiles + j] = hpx::async(
                    hpx::annotated_function(&gen_tile_identity,
                                            "assemble_identity_matrix"),
                    i,
                    j,
                    n_tile_size);
            }
        }

        //////////////////////////////////////////////////////////////////////////////
        // Cholesky decomposition
        right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);
        // Compute K^-1 through L*L^T*X = I
        forward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
        backward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

        // Triangular solve K_NxN * alpha = y
        // forward_solve_tiled(grad_I_tiles, alpha_tiles, n_tile_size,
        // n_tiles); backward_solve_tiled(grad_I_tiles, alpha_tiles,
        // n_tile_size, n_tiles);

        // inv(K)*y
        compute_gemm_of_invK_y(target.cublas_executors, grad_I_tiles, y_tiles, alpha_tiles, n_tile_size, n_tiles);

        // Compute loss
        compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);
        losses[iter] = loss_value.get();

        // Compute I-y*y^T*inv(K) -> NxN matrix
        // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles,
        // n_tile_size, n_tiles);

        // Compute K^-1 *(I - y*y^T*K^-1)
        // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
        // n_tile_size, n_tiles, n_tiles);
        // backward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
        // n_tile_size, n_tiles, n_tiles);

        // Update the hyperparameters
        if (trainable_params[0])
        {  // lengthscale
            sek_params.lengthscale = update_lengthscale(grad_I_tiles, grad_l_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
        }
        if (trainable_params[1])
        {  // vertical_lengthscale
            sek_params.vertical_lengthscale = update_vertical_lengthscale(grad_I_tiles, grad_v_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
        }
        if (trainable_params[2])
        {  // noise_variance
            sek_params.noise_variance = update_noise_variance(grad_I_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);
        }
    }
    // Update hyperparameter attributes in Gaussian process model
    // Return losses
    return hpx::async([losses]()
                      { return losses; }); */
    return hpx::shared_future<std::vector<double>>();
}

hpx::shared_future<double>
optimize_step(const std::vector<double> &training_input,
              const std::vector<double> &training_output,
              const std::size_t n_tiles,
              const std::size_t n_tile_size,
              const std::size_t n_regressors,
              const std::size_t iter,
              const gpxpy_hyper::SEKParams &sek_params,
              const std::vector<bool> trainable_params,
              const gpxpy_hyper::AdamParams &adam_params,
              gpxpy::CUDA_GPU &gpu)
{
    /* // declare tiled future data structures
    std::vector<hpx::shared_future<std::vector<double>>> K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_v_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_l_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_K_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> grad_I_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> alpha_tiles;
    std::vector<hpx::shared_future<std::vector<double>>> y_tiles;
    // data holders for Adam
    std::vector<hpx::shared_future<double>> m_T;
    std::vector<hpx::shared_future<double>> v_T;
    std::vector<hpx::shared_future<double>> beta1_T;
    std::vector<hpx::shared_future<double>> beta2_T;
    // data holder for loss
    hpx::shared_future<double> loss_value;
    // make shared future
    for (std::size_t i = 0; i < 3; i++)
    {
        hpx::shared_future<double> m =
            hpx::make_ready_future(adam_params.M_T[i]);  //.share();
        m_T.push_back(m);
        hpx::shared_future<double> v =
            hpx::make_ready_future(adam_params.V_T[i]);  //.share();
        v_T.push_back(v);
    }

    // Assemble beta1_t and beta2_t
    beta1_T.resize(1);
    beta1_T[0] =
        hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"),
                   iter + 1,
                   adam_params.beta1);

    beta2_T.resize(1);
    beta2_T[0] =
        hpx::async(hpx::annotated_function(&gen_beta_T, "assemble_tiled"),
                   iter + 1,
                   adam_params.beta1);

    // Assemble covariance matrix vector
    K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_covariance,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to vertical
    // lengthscale
    grad_v_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_v_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_grad_v, "assemble_tiled"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Assemble derivative of covariance matrix vector w.r.t. to lengthscale
    grad_l_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_l_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_grad_l, "assemble_tiled"),
                i,
                j,
                n_tile_size,
                n_regressors,
                sek_params,
                training_input);
        }
    }
    // Assemble matrix that will be multiplied with derivates
    grad_K_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_K_tiles[i * n_tiles + j] =
                hpx::async(hpx::annotated_function(&gen_tile_identity,
                                                   "assemble_tiled"),
                           i,
                           j,
                           n_tile_size);
        }
    }
    // Assemble alpha
    alpha_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        alpha_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble y
    y_tiles.resize(n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        y_tiles[i] = hpx::async(
            hpx::annotated_function(&gen_tile_output, "assemble_tiled"), i, n_tile_size, training_output);
    }
    // Assemble placeholder matrix for K^-1
    grad_I_tiles.resize(n_tiles * n_tiles);
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            grad_I_tiles[i * n_tiles + j] = hpx::async(
                hpx::annotated_function(&gen_tile_identity,
                                        "assemble_identity_matrix"),
                i,
                j,
                n_tile_size);
        }
    }

    // Cholesky decomposition
    right_looking_cholesky_tiled(target.cublas_executors, K_tiles, n_tile_size, n_tiles);

    // Triangular solve K_NxN * alpha = y
    forward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);
    backward_solve_tiled(target.cublas_executors, K_tiles, alpha_tiles, n_tile_size, n_tiles);

    // Compute K^-1 through L*L^T*X = I
    forward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);
    backward_solve_tiled_matrix(target.cublas_executors, K_tiles, grad_I_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // Compute loss
    compute_loss_tiled(target.cublas_executors, K_tiles, alpha_tiles, y_tiles, loss_value, n_tile_size, n_tiles);

    // // Fill I-y*y^T*inv(K)
    // update_grad_K_tiled(grad_K_tiles, y_tiles, alpha_tiles, n_tile_size,
    // n_tiles);

    // // Compute K^-1 * (I-y*y^T*K^-1)
    // forward_solve_tiled_matrix(K_tiles, grad_K_tiles, n_tile_size,
    // n_tile_size, n_tiles, n_tiles); backward_solve_tiled_matrix(K_tiles,
    // grad_K_tiles, n_tile_size, n_tile_size, n_tiles, n_tiles);

    // Update the hyperparameters
    if (trainable_params[0])
    {  // lengthscale
        sek_params.lengthscale = update_lengthscale(grad_I_tiles, grad_l_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
    }
    if (trainable_params[1])
    {  // vertical_lengthscale
        sek_params.vertical_lengthscale = update_vertical_lengthscale(grad_I_tiles, grad_v_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, 0);
    }
    if (trainable_params[2])
    {  // noise_variance
        sek_params.noise_variance = update_noise_variance(grad_I_tiles, alpha_tiles, sek_params, adam_params, n_tile_size, n_tiles, m_T, v_T, beta1_T, beta2_T, iter);
    }

    // Update hyperparameter attributes (first and second moment) for Adam
    for (std::size_t i = 0; i < 3; i++)
    {
        adam_params.M_T[i] = m_T[i].get();
        adam_params.V_T[i] = v_T[i].get();
    }

    // Return loss value
    double loss = loss_value.get();
    return hpx::async([loss]()
                      { return loss; }); */
    return hpx::shared_future<double>();
}

hpx::shared_future<std::vector<std::vector<double>>>
cholesky(const std::vector<double> &h_training_input,
         const std::size_t n_tiles,
         const std::size_t n_tile_size,
         const std::size_t n_regressors,
         const gpxpy_hyper::SEKParams sek_params,
         gpxpy::CUDA_GPU &gpu)
{
    gpu.create();

    double *d_training_input = copy_to_device(h_training_input, gpu);
    // Assemble tiled covariance matrix on GPU.
    std::vector<hpx::shared_future<double *>> d_tiles = assemble_tiled_covariance_matrix(d_training_input, n_tiles, n_tile_size, n_regressors, sek_params, gpu);

    // Compute Tiled Cholesky decomposition on device
    cusolverDnHandle_t cusolver = create_cusolver_handle();
    right_looking_cholesky_tiled(d_tiles, n_tile_size, n_tiles, gpu, cusolver);

    // Copy tiled matrix to host
    std::vector<std::vector<double>> h_tiles = move_lower_tiled_matrix_to_host(d_tiles, n_tile_size, n_tiles, gpu);

    cudaFree(d_training_input);
    destroy(cusolver);
    gpu.destroy();

    return hpx::make_ready_future(h_tiles);
}

}  // end of namespace gpu
