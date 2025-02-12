#ifndef GP_ALGORITHMS_GPU_H
#define GP_ALGORITHMS_GPU_H

#include "gp_hyperparameters.hpp"
#include "gp_kernels.hpp"
#include "target.hpp"
#include <hpx/future.hpp>
#include <vector>

namespace gpu
{

/**
 * @brief Generate a tile of the prior covariance matrix.
 */
double *
gen_tile_covariance(const double *d_input,
                    const std::size_t tile_row,
                    const std::size_t tile_column,
                    const std::size_t n_tile_size,
                    const std::size_t n_regressors,
                    const gpxpy_hyper::SEKParams sek_params,
                    gpxpy::CUDA_GPU &gpu);

/**
 * @brief Generate a tile of the prior covariance matrix.
 */
double *
gen_tile_prior_covariance(const double *d_input,
                          const std::size_t tile_row,
                          const std::size_t tile_column,
                          const std::size_t n_tile_size,
                          const std::size_t n_regressors,
                          const gpxpy_hyper::SEKParams sek_params,
                          gpxpy::CUDA_GPU &gpu);

/**
 * @brief Generate a tile of the cross-covariance matrix.
 */
double *
gen_tile_cross_covariance(const double *d_row_input,
                          const double *d_col_input,
                          const std::size_t tile_row,
                          const std::size_t tile_column,
                          const std::size_t n_row_tile_size,
                          const std::size_t n_column_tile_size,
                          const std::size_t n_regressors,
                          const gpxpy_hyper::SEKParams sek_params,
                          gpxpy::CUDA_GPU &gpu);

/**
 * @brief Generate a tile of the cross-covariance matrix.
 */
hpx::shared_future<double *>
gen_tile_cross_cov_T(
    std::size_t n_row_tile_size,
    std::size_t n_column_tile_size,
    const hpx::shared_future<double *> f_cross_covariance_tile,
    gpxpy::CUDA_GPU &gpu);

/**
 * @brief Generate a tile containing the output observations.
 */
double *
gen_tile_output(const std::size_t row,
                const std::size_t n_tile_size,
                const double* d_output,
                gpxpy::CUDA_GPU &gpu);

/**
 * @brief Compute the total 2-norm error.
 */
double
compute_error_norm(const std::size_t n_tiles,
                   const std::size_t n_tile_size,
                   const std::vector<double> &b,
                   const std::vector<std::vector<double>> &tiles);

/**
 * @brief Generate an empty tile
 */
double *gen_tile_zeros(std::size_t n_tile_size, gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<std::vector<double>>
predict(const std::vector<double> &training_input,
        const std::vector<double> &training_output,
        const std::vector<double> &test_input,
        const std::size_t n_tiles,
        const std::size_t n_tile_size,
        const std::size_t m_tiles,
        const std::size_t m_tile_size,
        const std::size_t n_regressors,
        const gpxpy_hyper::SEKParams sek_params,
        gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<std::vector<std::vector<double>>>
predict_with_uncertainty(const std::vector<double> &training_input,
                         const std::vector<double> &training_output,
                         const std::vector<double> &test_input,
                         const std::size_t n_tiles,
                         const std::size_t n_tile_size,
                         const std::size_t m_tiles,
                         const std::size_t m_tile_size,
                         const std::size_t n_regressors,
                         const gpxpy_hyper::SEKParams sek_params,
                         gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<std::vector<std::vector<double>>>
predict_with_full_cov(const std::vector<double> &training_input,
                      const std::vector<double> &training_output,
                      const std::vector<double> &test_input,
                      const int n_tiles,
                      const int n_tile_size,
                      const int m_tiles,
                      const int m_tile_size,
                      const int n_regressors,
                      const gpxpy_hyper::SEKParams sek_params,
                      gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<double>
compute_loss(const std::vector<double> &training_input,
             const std::vector<double> &training_output,
             const std::size_t n_tiles,
             const std::size_t n_tile_size,
             const std::size_t n_regressors,
             const gpxpy_hyper::SEKParams sek_params,
             gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<std::vector<double>>
optimize(const std::vector<double> &training_input,
         const std::vector<double> &training_output,
         const std::size_t n_tiles,
         const std::size_t n_tile_size,
         const std::size_t n_regressors,
         const gpxpy_hyper::SEKParams &sek_params,
         const std::vector<bool> trainable_params,
         const gpxpy_hyper::AdamParams &adam_params,
         gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
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
              gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
std::vector<hpx::shared_future<double *>>
assemble_tiled_covariance_matrix(const double *d_training_input,
                                 const std::size_t n_tiles,
                                 const std::size_t n_tile_size,
                                 const std::size_t n_regressors,
                                 const gpxpy_hyper::SEKParams sek_params,
                                 gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
std::vector<std::vector<double>>
move_lower_tiled_matrix_to_host(
    const std::vector<hpx::shared_future<double *>> &d_tiles,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu);

/**
 * @brief TODO: documentation
 */
hpx::shared_future<std::vector<std::vector<double>>>
cholesky(const std::vector<double> &training_input,
         const std::size_t n_tiles,
         const std::size_t n_tile_size,
         const std::size_t n_regressors,
         const gpxpy_hyper::SEKParams sek_params,
         gpxpy::CUDA_GPU &gpu);

}  // end of namespace gpu

#endif  // end of GP_ALGORITHMS_GPU_H
