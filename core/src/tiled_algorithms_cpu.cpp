#include "../include/tiled_algorithms_cpu.hpp"

#include "../include/adapter_mkl.hpp"
#include "gp_uncertainty_cpu.hpp"
#include "gp_optimizer_cpu.hpp"
#include <cmath>
#include <hpx/future.hpp>

namespace cpu
{

// Tiled Cholesky Algorithm {{{

void right_looking_cholesky_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::size_t N,
    std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // POTRF: Compute Cholesky factor L
        ft_tiles[k * n_tiles + k] =
            hpx::dataflow(hpx::annotated_function(&potrf, "cholesky potrf"),
                          ft_tiles[k * n_tiles + k],
                          N);

        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // TRSM:  Solve X * L^T = A
            ft_tiles[m * n_tiles + k] = hpx::dataflow(
                hpx::annotated_function(&trsm, "cholesky trsm"),
                ft_tiles[k * n_tiles + k],
                ft_tiles[m * n_tiles + k],
                N,
                N,
                Blas_trans,
                Blas_right);
        }
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // SYRK:  A = A - B * B^T
            ft_tiles[m * n_tiles + m] = hpx::dataflow(
                hpx::annotated_function(&syrk, "cholesky syrk"),
                ft_tiles[m * n_tiles + m],
                ft_tiles[m * n_tiles + k],
                N);
            for (std::size_t n = k + 1; n < m; n++)
            {
                // GEMM: C = C - A * B^T
                ft_tiles[m * n_tiles + n] = hpx::dataflow(
                    hpx::annotated_function(&gemm, "cholesky gemm"),
                    ft_tiles[m * n_tiles + k],
                    ft_tiles[n * n_tiles + k],
                    ft_tiles[m * n_tiles + n],
                    N,
                    N,
                    N,
                    Blas_no_trans,
                    Blas_trans);
            }
        }
    }
}

// }}} ------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms ----------------------------------- {{{

void forward_solve_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        // TRSM: Solve L * x = a
        ft_rhs[k] = hpx::dataflow(
            hpx::annotated_function(&trsv, "triangular_solve_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_no_trans);
        for (std::size_t m = k + 1; m < n_tiles; m++)
        {
            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                hpx::annotated_function(&gemv, "triangular_solve_tiled"),
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_no_trans);
        }
    }
}

void backward_solve_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t n_tiles)
{
    for (int k = n_tiles - 1; k >= 0; k--)  // int instead of std::size_t for last comparison
    {
        // TRSM: Solve L^T * x = a
        ft_rhs[k] = hpx::dataflow(
            hpx::annotated_function(&trsv, "triangular_solve_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            N,
            Blas_trans);
        for (int m = k - 1; m >= 0; m--)  // int instead of std::size_t for last comparison
        {
            // GEMV:b = b - A^T * a
            ft_rhs[m] = hpx::dataflow(
                hpx::annotated_function(&gemv, "triangular_solve_tiled"),
                ft_tiles[k * n_tiles + m],
                ft_rhs[k],
                ft_rhs[m],
                N,
                N,
                Blas_substract,
                Blas_trans);
        }
    }
}

void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < n_tiles; k++)
        {
            // TRSM: solve L * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                hpx::annotated_function(&trsm,
                                        "triangular_solve_tiled_matrix"),
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_no_trans,
                Blas_left);
            for (std::size_t m = k + 1; m < n_tiles; m++)
            {
                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    hpx::annotated_function(
                        &gemm, "triangular_solve_tiled_matrix"),
                    ft_tiles[m * n_tiles + k],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (int c = 0; c < m_tiles; c++)
    {
        for (int k = n_tiles - 1; k >= 0;
             k--)  // int instead of std::size_t for last comparison
        {
            // TRSM: solve L^T * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                hpx::annotated_function(&trsm,
                                        "triangular_solve_tiled_matrix"),
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                N,
                M,
                Blas_trans,
                Blas_left);
            for (int m = k - 1; m >= 0;
                 m--)  // int instead of std::size_t for last comparison
            {
                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    hpx::annotated_function(
                        &gemm, "triangular_solve_tiled_matrix"),
                    ft_tiles[k * n_tiles + m],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    N,
                    M,
                    N,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

// }}} -------------------------------- end of Tiled Triangular Solve
// Algorithms

void forward_solve_KcK_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t r = 0; r < m_tiles; r++)
    {
        for (std::size_t c = 0; c < n_tiles; c++)
        {
            // TRSM: solve L * X = A
            ft_rhs[c * m_tiles + r] = hpx::dataflow(
                hpx::annotated_function(&trsm,
                                        "triangular_solve_tiled_matrix_KK"),
                ft_tiles[c * n_tiles + c],
                ft_rhs[c * m_tiles + r],
                N,
                M,
                Blas_no_trans,
                Blas_left);
            for (std::size_t m = c + 1; m < n_tiles; m++)
            {
                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + r] = hpx::dataflow(
                    hpx::annotated_function(
                        &gemm, "triangular_solve_tiled_matrix_KK"),
                    ft_tiles[m * n_tiles + c],
                    ft_rhs[c * m_tiles + r],
                    ft_rhs[m * m_tiles + r],
                    N,
                    M,
                    N,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

void compute_gemm_of_invK_y(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::size_t N,
    std::size_t n_tiles)
{
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            ft_alpha[i] = hpx::dataflow(
                hpx::annotated_function(&gemv, "prediction_tiled"),
                ft_invK[i * n_tiles + j],
                ft_y[j],
                ft_alpha[i],
                N,
                N,
                Blas_add,
                Blas_no_trans);
        }
    }
}

void compute_loss_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_y,
    hpx::shared_future<double> &loss,
    std::size_t N,
    std::size_t n_tiles)
{
    std::vector<hpx::shared_future<double>> loss_tiled;
    loss_tiled.resize(n_tiles);
    for (std::size_t k = 0; k < n_tiles; k++)
    {
        loss_tiled[k] = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&compute_loss),
                                    "loss_tiled"),
            ft_tiles[k * n_tiles + k],
            ft_alpha[k],
            ft_y[k],
            N);
    }

    loss = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&add_losses), "loss_tiled"),
        loss_tiled,
        N,
        n_tiles);
}

void prediction_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_rhs,
    std::size_t N_row,
    std::size_t N_col,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t k = 0; k < m_tiles; k++)
    {
        for (std::size_t m = 0; m < n_tiles; m++)
        {
            ft_rhs[k] = hpx::dataflow(
                hpx::annotated_function(&gemv, "prediction_tiled"),
                ft_tiles[k * n_tiles + m],
                ft_vector[m],
                ft_rhs[k],
                N_row,
                N_col,
                Blas_add,
                Blas_no_trans);
        }
    }
}

void posterior_covariance_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter_tiles,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles;
             ++n)
        {  // Compute inner product to obtain diagonal elements of
           // (K_MxN * (K^-1_NxN * K_NxM))
            ft_inter_tiles[i] = hpx::dataflow(
                hpx::annotated_function(&dot_diag_syrk, "posterior_tiled"),
                ft_tCC_tiles[n * m_tiles + i],
                ft_inter_tiles[i],
                N,
                M);
        }
    }
}

void full_cov_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::size_t N,
    std::size_t M,
    std::size_t n_tiles,
    std::size_t m_tiles)
{
    for (std::size_t c = 0; c < m_tiles; c++)
    {
        for (std::size_t k = 0; k < m_tiles; k++)
        {
            for (std::size_t m = 0; m < n_tiles; m++)
            {
                // GEMM:  C = C - A^T * B
                ft_priorK[c * m_tiles + k] = hpx::dataflow(
                    hpx::annotated_function(
                        &gemm, "triangular_solve_tiled_matrix"),
                    ft_tCC_tiles[m * m_tiles + c],
                    ft_tCC_tiles[m * m_tiles + k],
                    ft_priorK[c * m_tiles + k],
                    N,
                    M,
                    M,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

void prediction_uncertainty_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_inter,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&diag_posterior),
                                    "uncertainty_tiled"),
            ft_priorK[i],
            ft_inter[i],
            M);
    }
}

void pred_uncer_tiled(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_priorK,
    std::vector<hpx::shared_future<std::vector<double>>> &ft_vector,
    std::size_t M,
    std::size_t m_tiles)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&diag_tile),
                                    "uncertainty_tiled"),
            ft_priorK[i * m_tiles + i],
            M);
    }
}

void update_grad_K_tiled_mkl(
    std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v1,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_v2,
    std::size_t N,
    std::size_t n_tiles)

{
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        for (std::size_t j = 0; j < n_tiles; j++)
        {
            ft_tiles[i * n_tiles + j] = hpx::dataflow(
                hpx::annotated_function(&ger, "gradient_tiled"),
                ft_tiles[i * n_tiles + j],
                ft_v1[i],
                ft_v2[j],
                N);
        }
    }
}

// Perform a gradient scent step for selected hyperparameter using Adam
// algorithm
static double
update_hyperparameter(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_gradparam,
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
    /// part 1: trace(inv(K)*grad_param)
    std::vector<hpx::shared_future<std::vector<double>>> diag_tiles;
    diag_tiles.resize(n_tiles);
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        diag_tiles[d] =
            hpx::async(hpx::annotated_function(&gen_tile_zeros_diag,
                                               "assemble_tiled"),
                       N);
    }

    // Compute diagonal elements of inv(K) * grad_hyperparam
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            diag_tiles[i] = hpx::dataflow(
                hpx::annotated_function(&dot_diag_gemm,
                                        "grad_left_tiled"),
                ft_invK[i * n_tiles + j],
                ft_gradparam[j * n_tiles + i],
                diag_tiles[i],
                N,
                N);
        }
    }

    // compute trace(inv(K) * grad_hyperparam)
    hpx::shared_future<double> grad_left =
        hpx::make_ready_future(0.0)
            .share();
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        grad_left = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&sum_gradleft),
                                    "grad_left_tiled"),
            diag_tiles[j],
            grad_left);
    }

    /// part 2: alpha^T * grad_param * alpha
    std::vector<hpx::shared_future<std::vector<double>>> inter_alpha;
    inter_alpha.resize(n_tiles);
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        inter_alpha[d] =
            hpx::async(hpx::annotated_function(&gen_tile_zeros_diag,
                                               "assemble_tiled"),
                       N);
    }

    for (std::size_t k = 0; k < n_tiles; k++)
    {
        for (std::size_t m = 0; m < n_tiles; m++)
        {
            inter_alpha[k] = hpx::dataflow(
                hpx::annotated_function(&gemv, "prediction_tiled"),
                ft_gradparam[k * n_tiles + m],
                ft_alpha[m],
                inter_alpha[k],
                N,
                N,
                Blas_add,
                Blas_no_trans);
        }
    }

    hpx::shared_future<double> grad_right =
        hpx::make_ready_future(0.0).share();
    for (std::size_t j = 0; j < n_tiles;
         ++j)
    {  // Compute inner product to obtain diagonal elements of
       // (K_MxN * (K^-1_NxN * K_NxM))
        grad_right = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&sum_gradright),
                                    "grad_right_tiled"),
            inter_alpha[j],
            ft_alpha[j],
            grad_right,
            N);
    }

    /// part 3: update parameter

    // compute gradient = grad_left + grad_r
    hpx::shared_future<double> gradient = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&compute_gradient),
                                "gradient_tiled"),
        grad_left,
        grad_right,
        N,
        n_tiles);

    // transform hyperparameter to unconstrained form
    hpx::shared_future<double> unconstrained_param = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&to_unconstrained),
                                "gradient_tiled"),
        hyperparameter,
        false);
    // update moments
    m_T[param_idx] = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&update_first_moment),
                                "gradient_tiled"),
        gradient,
        m_T[param_idx],
        adam_params.beta1);
    v_T[param_idx] = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&update_second_moment),
                                "gradient_tiled"),
        gradient,
        v_T[param_idx],
        adam_params.beta2);
    // update unconstrained parameter
    hpx::shared_future<double> updated_param =
        hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&update_param),
                                              "gradient_tiled"),
                      unconstrained_param,
                      sek_params,
                      adam_params,
                      m_T[param_idx],
                      v_T[param_idx],
                      beta1_T,
                      beta2_T,
                      iter);
    // transform hyperparameter to constrained form
    return hpx::dataflow(
               hpx::annotated_function(hpx::unwrapping(&to_constrained),
                                       "gradient_tiled"),
               updated_param,
               false)
        .get();
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
    // part1: compute trace(inv(K) * grad_hyperparam)
    hpx::shared_future<double> grad_left =
        hpx::make_ready_future(0.0).share();
    for (std::size_t j = 0; j < n_tiles; ++j)
    {
        grad_left = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&sum_noise_gradleft),
                                    "grad_left_tiled"),
            ft_invK[j * n_tiles + j],
            grad_left,
            sek_params,
            N,
            n_tiles);
    }

    /// part 2: alpha^T * grad_param * alpha
    hpx::shared_future<double> grad_right =
        hpx::make_ready_future(0.0).share();
    for (std::size_t j = 0; j < n_tiles;
         ++j)
    {  // Compute inner product to obtain diagonal elements of
       // (K_MxN * (K^-1_NxN * K_NxM))
        grad_right = hpx::dataflow(
            hpx::annotated_function(hpx::unwrapping(&sum_noise_gradright),
                                    "grad_right_tiled"),
            ft_alpha[j],
            grad_right,
            sek_params,
            N);
    }

    /// part 3: update parameter
    hpx::shared_future<double> gradient = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&compute_gradient),
                                "gradient_tiled"),
        grad_left,
        grad_right,
        N,
        n_tiles);
    // transform hyperparameter to unconstrained form
    hpx::shared_future<double> unconstrained_param = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&to_unconstrained),
                                "gradient_tiled"),
        sek_params.noise_variance,
        true);
    // update moments
    m_T[2] = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&update_first_moment),
                                "gradient_tiled"),
        gradient,
        m_T[2],
        adam_params.beta1);
    v_T[2] = hpx::dataflow(
        hpx::annotated_function(hpx::unwrapping(&update_second_moment),
                                "gradient_tiled"),
        gradient,
        v_T[2],
        adam_params.beta2);
    // update unconstrained parameter
    hpx::shared_future<double> updated_param =
        hpx::dataflow(hpx::annotated_function(
                          hpx::unwrapping(&update_param), "gradient_tiled"),
                      unconstrained_param,
                      sek_params,
                      adam_params,
                      m_T[2],
                      v_T[2],
                      beta1_T,
                      beta2_T,
                      iter);
    // transform hyperparameter to constrained form
    return hpx::dataflow(hpx::annotated_function(
                             hpx::unwrapping(&to_constrained),
                             "gradient_tiled"),
                         updated_param,
                         true)
        .get();
}

}  // end of namespace cpu
