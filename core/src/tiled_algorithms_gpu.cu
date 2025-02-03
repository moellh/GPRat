#include "tiled_algorithms_gpu.cuh"

#include "adapter_cublas.cuh"
#include <hpx/algorithm.hpp>

using hpx::experimental::for_loop;

namespace gpu
{

// Tiled Cholesky Algorithm ------------------------------------------------ {{{

void right_looking_cholesky_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const cusolverDnHandle_t &cusolver,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    int cublas_counter = 0;
    auto next_cublas = [&]()
    { return cublas_handles[cublas_counter++ % gpu.n_streams]; };

    // clang-format off
    for_loop(hpx::execution::seq, 0, n_tiles, [&](size_t k)
    {
        cusolverDnSetStream(cusolver, gpu.next_stream());

        // POTRF
        ft_tiles[k * n_tiles + k] = hpx::dataflow(&potrf, cusolver, ft_tiles[k * n_tiles + k], n_tile_size);

        for_loop(hpx::execution::par, k + 1, n_tiles, [&](size_t m)
        {
            cublasHandle_t cublas = next_cublas();
            cublasSetStream(cublas, gpu.next_stream());

            // TRSM
            ft_tiles[m * n_tiles + k] = hpx::dataflow(&trsm, cublas, ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], n_tile_size, n_tile_size, Blas_trans, Blas_right);
        });

        for_loop(hpx::execution::par, k + 1, n_tiles, [&](size_t m)
        {
            cublasHandle_t cublas = next_cublas();
            cublasSetStream(cublas, gpu.next_stream());

            // SYRK
            ft_tiles[m * n_tiles + m] = hpx::dataflow(&syrk, cublas,
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[m * n_tiles + m],
                    n_tile_size);

            for_loop(hpx::execution::par, k + 1, m, [&](size_t n)
            {
                cublasHandle_t cublas = next_cublas();
                cublasSetStream(cublas, gpu.next_stream());

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(&gemm, cublas, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], n_tile_size, n_tile_size, n_tile_size, Blas_no_trans, Blas_trans);
            });
        });
    });
    // clang-format on
}

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    int cublas_counter = 0;
    auto next_cublas = [&]()
    { return cublas_handles[cublas_counter++ % gpu.n_streams]; };

    // clang-format off
    for_loop(hpx::execution::seq, 0, n_tiles, [&](std::size_t k)
    {
        cublasHandle_t cublas = next_cublas();
        cublasSetStream(cublas, gpu.next_stream());

        // TRSM: Solve L * x = a
        ft_rhs[k] = hpx::dataflow(
            &trsv,
            cublas,
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            n_tile_size,
            Blas_no_trans);

        for_loop(hpx::execution::seq, k + 1, n_tiles, [&](std::size_t m)
        {
            cublasHandle_t cublas = next_cublas();
            cublasSetStream(cublas, gpu.next_stream());

            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                Blas_substract,
                Blas_no_trans);
        });
    });
    // clang-format on
}

void backward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    int cublas_counter = 0;
    auto next_cublas = [&]()
    { return cublas_handles[cublas_counter++ % gpu.n_streams]; };

    for (int k = n_tiles - 1; k >= 0; k--)  // int instead of std::size_t for last comparison
    {
        cublasHandle_t cublas = next_cublas();
        cublasSetStream(cublas, gpu.next_stream());

        // TRSM: Solve L^T * x = a
        ft_rhs[k] = hpx::dataflow(
            &trsv,
            cublas,
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            n_tile_size,
            Blas_trans);
        for (int m = k - 1; m >= 0; m--)  // int instead of std::size_t for last comparison
        {
            cublasHandle_t cublas = next_cublas();
            cublasSetStream(cublas, gpu.next_stream());

            // GEMV:b = b - A^T * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                ft_tiles[k * n_tiles + m],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                Blas_substract,
                Blas_trans);
        }
    }
}

// Tiled Triangular Solve Algorithms for matrices (K * X = B)
void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

// }}} ---------------------------- end of Tiled Triangular Solve Algorithms

void forward_solve_KcK_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void compute_gemm_of_invK_y(
    std::vector<hpx::shared_future<double *>> &ft_invK,
    std::vector<hpx::shared_future<double *>> &ft_y,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    hpx::shared_future<double> &loss,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

// Tiled Prediction
void prediction_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t N_row,
    const std::size_t N_col,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    // TODO: nextup
}

void posterior_covariance_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void full_cov_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void prediction_uncertainty_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_inter,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void pred_uncer_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

void update_grad_K_tiled_mkl(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::vector<hpx::shared_future<double *>> &ft_v1,
    const std::vector<hpx::shared_future<double *>> &ft_v2,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{ }

/**
 * NOTE: not in header -> TODO: write documentation
 */
static double
update_hyperparameter(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    double &hyperparameter,  // lengthscale or vertical-lengthscale
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    int param_idx,  // 0 for lengthscale, 1 for vertical-lengthscale
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    return 0.0;
}

double update_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        0,
        gpu,
        cublas_handles);
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_gradparam,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.vertical_lengthscale,
        sek_params,
        adam_params,
        n_tile_size,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        1,
        gpu,
        cublas_handles);
}

double update_noise_variance(
    const std::vector<hpx::shared_future<double *>> &ft_invK,
    const std::vector<hpx::shared_future<double *>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles)
{
    return 0;
}

}  // namespace gpu
