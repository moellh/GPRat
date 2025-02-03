#ifndef TILED_ALGORITHMS_GPU_H
#define TILED_ALGORITHMS_GPU_H

#include "gp_optimizer_cpu.hpp"
#include "target.hpp"
#include <cusolverDn.h>
#include <hpx/modules/async_cuda.hpp>
#include <vector>

namespace gpu
{

// Tiled Cholesky Algorithm -------------------------------------------- {{{

/**
 * @brief Perform right-looking Cholesky decomposition.
 *
 * @param n_streams Number of CUDA streams.
 * @param ft_tiles Matrix represented as a vector of tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Size of the matrix.
 * @param n_tiles Number of tiles.
 */
void right_looking_cholesky_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::size_t N,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const cusolverDnHandle_t &cusolver,
    const std::vector<cublasHandle_t> &cublas_handles);

// }}} ------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms ----------------------------------- {{{

void forward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

void backward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// Tiled Triangular Solve Algorithms for matrices (K * X = B)
void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// }}} ---------------------------- end of Tiled Triangular Solve Algorithms

// Triangular solve A_M,N * K_NxN = K_MxN -> A_MxN = K_MxN * K^-1_NxN
// Tiled Triangular Solve Algorithms for Matrices (K * X = B)
void forward_solve_KcK_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

void compute_gemm_of_invK_y(
    std::vector<hpx::shared_future<double *>> &ft_invK,
    std::vector<hpx::shared_future<double *>> &ft_y,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// Tiled Loss
void compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    hpx::shared_future<double> &loss,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

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
    const std::vector<cublasHandle_t> &cublas_handles);

// Tiled Diagonal of Posterior Covariance Matrix
void posterior_covariance_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// Tiled Diagonal of Posterior Covariance Matrix
void full_cov_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handl);

// Tiled Prediction Uncertainty
void prediction_uncertainty_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_inter,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// Tiled Prediction Uncertainty
void pred_uncer_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::vector<hpx::shared_future<double *>> &ft_v1,
    const std::vector<hpx::shared_future<double *>> &ft_v2,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu,
    const std::vector<cublasHandle_t> &cublas_handles);

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
    const std::vector<cublasHandle_t> &cublas_handles);

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
    const std::vector<cublasHandle_t> &cublas_handles);

// Update noise variance using gradient decent + Adam
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
    const std::vector<cublasHandle_t> &cublas_handles);

}  // end of namespace gpu

#endif  // end of TILED_ALGORITHMS_GPU_H
