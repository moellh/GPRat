#include "tiled_algorithms_gpu.cuh"

#include "adapter_cublas.hpp"
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
        ft_tiles[k * n_tiles + k] = hpx::dataflow(&potrf, cusolver, ft_tiles[k * n_tiles + k], n_tile_size).get();

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
            ft_tiles[m * n_tiles + m] = hpx::dataflow(&syrk, cublas, ft_tiles[m * n_tiles + m], ft_tiles[m * n_tiles + k], n_tile_size);

            for_loop(hpx::execution::par, k + 1, m, [&](size_t n)
            {
                cublasHandle_t cublas = next_cublas();
                cublasSetStream(cublas, gpu.next_stream());

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(&gemm_cholesky, cublas, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], n_tile_size);
            });
        });
    });
    // clang-format on
}

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{ }

void backward_solve_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Triangular Solve Algorithms for matrices (K * X = B)
void forward_solve_tiled_matrix(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

void backward_solve_tiled_matrix(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// }}} ---------------------------- end of Tiled Triangular Solve Algorithms

// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
// Tiled Triangular Solve Algorithms for Matrices (K * X = B)
void forward_solve_KcK_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

void compute_gemm_of_invK_y(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Loss
void compute_loss_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    hpx::shared_future<double> &loss,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Tiled Prediction
void prediction_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Diagonal of Posterior Covariance Matrix
void posterior_covariance_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter_tiles,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Diagonal of Posterior Covariance Matrix
void full_cov_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{ }

// Tiled Prediction Uncertainty
void prediction_uncertainty_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{ }

// Tiled Prediction Uncertainty
void pred_uncer_tiled(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{ }

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(
    std::vector<cublas_executor> cublas,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v1,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v2,
    std::size_t N,
    std::size_t n_tiles)
{ }

// Perform a gradient scent step for selected hyperparameter using Adam
// algorithm
static double
update_hyperparameter(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    double &hyperparameter,  // lengthscale or vertical-lengthscale
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter,
    int param_idx)  // 0 for lengthscale, 1 for vertical-lengthscale
{
    return 0.0;
}

double update_lengthscale(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.lengthscale,
        sek_params,
        adam_params,
        N,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        0);
}

double update_vertical_lengthscale(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &
        ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{
    return update_hyperparameter(
        ft_invK,
        ft_gradparam,
        ft_alpha,
        sek_params.vertical_lengthscale,
        sek_params,
        adam_params,
        N,
        n_tiles,
        m_T,
        v_T,
        beta1_T,
        beta2_T,
        iter,
        1);
}

double update_noise_variance(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    gpxpy_hyper::SEKParams sek_params,
    gpxpy_hyper::AdamParams adam_params,
    std::size_t N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    int iter)
{
    return 0;
}

}  // namespace gpu
