#include "tiled_algorithms_gpu.cuh"

#include "adapter_cublas.cuh"
#include "gp_uncertainty_gpu.cuh"
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
    const cusolverDnHandle_t &cusolver)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        cudaStream_t stream = gpu.next_stream();
        cusolverDnSetStream(cusolver, stream);

        // POTRF
        ft_tiles[k * n_tiles + k] = hpx::dataflow(hpx::annotated_function(&potrf, "Cholesky POTRF"), cusolver, stream, ft_tiles[k * n_tiles + k], n_tile_size);

        // NOTE: The result is immediately needed by TRSM. Also TRSM may throw
        // an error otherwise.
        ft_tiles[k * n_tiles + k].get();

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM
            ft_tiles[m * n_tiles + k] = hpx::dataflow(&trsm, cublas, stream, ft_tiles[k * n_tiles + k], ft_tiles[m * n_tiles + k], n_tile_size, n_tile_size, Blas_trans, Blas_right);
        }

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // SYRK
            ft_tiles[m * n_tiles + m] = hpx::dataflow(&syrk, cublas, stream, ft_tiles[m * n_tiles + k], ft_tiles[m * n_tiles + m], n_tile_size);

            for (std::size_t n = k + 1; n < m; ++n)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM
                ft_tiles[m * n_tiles + n] = hpx::dataflow(&gemm, cublas, stream, ft_tiles[m * n_tiles + k], ft_tiles[n * n_tiles + k], ft_tiles[m * n_tiles + n], n_tile_size, n_tile_size, n_tile_size, Blas_no_trans, Blas_trans);
            }
        }
    }
}

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

void forward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t k = 0; k < n_tiles; ++k)
    {
        auto [cublas, stream] = gpu.next_cublas_handle();

        // TRSM: Solve L * x = a
        ft_rhs[k] = hpx::dataflow(
            &trsv,
            cublas,
            stream,
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            n_tile_size,
            Blas_no_trans);

        for (std::size_t m = k + 1; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // GEMV: b = b - A * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_tiles[m * n_tiles + k],
                ft_rhs[k],
                ft_rhs[m],
                n_tile_size,
                n_tile_size,
                Blas_substract,
                Blas_no_trans);
        }
    }
}

void backward_solve_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    // NOTE: The loops traverse backwards. Its last comparisons require the
    // usage negative numbers. Therefore they use signed int instead of the
    // unsigned std::size_t.

    for (int k = n_tiles - 1; k >= 0; k--)
    {
        auto [cublas, stream] = gpu.next_cublas_handle();

        // TRSM: Solve L^T * x = a
        ft_rhs[k] = hpx::dataflow(
            &trsv,
            cublas,
            stream,
            ft_tiles[k * n_tiles + k],
            ft_rhs[k],
            n_tile_size,
            Blas_trans);

        for (int m = k - 1; m >= 0; m--)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // GEMV: b = b - A^T * a
            ft_rhs[m] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
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

void forward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    // clang-format off
    for_loop(hpx::execution::seq, 0, m_tiles, [&](std::size_t c)
    {
        for_loop(hpx::execution::seq, 0, n_tiles, [&](std::size_t k)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM: solve L * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                n_tile_size,
                m_tile_size,
                Blas_no_trans,
                Blas_left);

            for_loop(hpx::execution::seq, k + 1, n_tiles, [&](std::size_t m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[m * n_tiles + k],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    Blas_no_trans,
                    Blas_no_trans);
            });
        });
    });
    // clang-format on
}

void backward_solve_tiled_matrix(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    // clang-format off
    for_loop(hpx::execution::seq, 0, m_tiles, [&](std::size_t c)
    {
        for_loop(hpx::execution::seq, 0, n_tiles, [&](std::size_t k)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM: solve L^T * X = A
            ft_rhs[k * m_tiles + c] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[k * n_tiles + k],
                ft_rhs[k * m_tiles + c],
                n_tile_size,
                m_tile_size,
                Blas_trans,
                Blas_left);

            for_loop(hpx::execution::seq, 0, k, [&](std::size_t m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM: C = C - A^T * B
                ft_rhs[m * m_tiles + c] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[k * n_tiles + m],
                    ft_rhs[k * m_tiles + c],
                    ft_rhs[m * m_tiles + c],
                    n_tile_size,
                    m_tile_size,
                    n_tile_size,
                    Blas_trans,
                    Blas_no_trans);
            });
        });
    });
    // clang-format on
}

// }}} ---------------------------- end of Tiled Triangular Solve Algorithms

void forward_solve_KcK_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t n_tile_size,
    const std::size_t M,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t r = 0; r < m_tiles; ++r)
    {
        for (std::size_t c = 0; c < n_tiles; ++c)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // TRSM: solve L * X = A
            ft_rhs[c * m_tiles + r] = hpx::dataflow(
                &trsm,
                cublas,
                stream,
                ft_tiles[c * n_tiles + c],
                ft_rhs[c * m_tiles + r],
                n_tile_size,
                M,
                Blas_no_trans,
                Blas_left);

            for (std::size_t m = c + 1; m < n_tiles; ++m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM: C = C - A * B
                ft_rhs[m * m_tiles + r] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tiles[m * n_tiles + c],
                    ft_rhs[c * m_tiles + r],
                    ft_rhs[m * m_tiles + r],
                    n_tile_size,
                    M,
                    n_tile_size,
                    Blas_no_trans,
                    Blas_no_trans);
            }
        }
    }
}

void compute_gemm_of_invK_y(
    std::vector<hpx::shared_future<double *>> &ft_invK,
    std::vector<hpx::shared_future<double *>> &ft_y,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_alpha[i] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
                ft_invK[i * n_tiles + j],
                ft_y[j],
                ft_alpha[i],
                n_tile_size,
                n_tile_size,
                Blas_add,
                Blas_no_trans);
        }
    }
}

void compute_loss_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_alpha,
    std::vector<hpx::shared_future<double *>> &ft_y,
    hpx::shared_future<double> &loss,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    // std::vector<hpx::shared_future<double>> loss_tiled;
    // loss_tiled.resize(n_tiles);
    // for (std::size_t k = 0; k < n_tiles; k++)
    // {
    //     loss_tiled[k] = hpx::dataflow(
    //         hpx::annotated_function(hpx::unwrapping(&compute_loss),
    //                                 "loss_tiled"),
    //         ft_tiles[k * n_tiles + k],
    //         ft_alpha[k],
    //         ft_y[k],
    //         n_tile_size);
    // }
    //
    // loss = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&add_losses), "loss_tiled"),
    //     loss_tiled,
    //     n_tile_size,
    //     n_tiles);

    // TODO: requires GPU implementation of compute loss and add_losses in gp_optimizer
}

// Tiled Prediction
void prediction_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    std::vector<hpx::shared_future<double *>> &ft_rhs,
    const std::size_t N_row,
    const std::size_t N_col,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t k = 0; k < m_tiles; ++k)
    {
        for (std::size_t m = 0; m < n_tiles; ++m)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_rhs[k] = hpx::dataflow(
                &gemv,
                cublas,
                stream,
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
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_inter_tiles,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; ++i)
    {
        for (std::size_t n = 0; n < n_tiles; ++n)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            // Compute inner product to obtain diagonal elements of
            // (K_MxN * (K^-1_NxN * K_NxM))
            ft_inter_tiles[i] = hpx::dataflow(
                &dot_diag_syrk,
                cublas,
                stream,
                ft_tCC_tiles[n * m_tiles + i],
                ft_inter_tiles[i],
                n_tile_size,
                m_tile_size);
        }
    }
}

void full_cov_tiled(
    std::vector<hpx::shared_future<double *>> &ft_tCC_tiles,
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    const std::size_t n_tile_size,
    const std::size_t m_tile_size,
    const std::size_t n_tiles,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t c = 0; c < m_tiles; ++c)
    {
        for (std::size_t k = 0; k < m_tiles; ++k)
        {
            for (std::size_t m = 0; m < n_tiles; ++m)
            {
                auto [cublas, stream] = gpu.next_cublas_handle();

                // GEMM:  C = C - A^T * B
                ft_priorK[c * m_tiles + k] = hpx::dataflow(
                    &gemm,
                    cublas,
                    stream,
                    ft_tCC_tiles[m * m_tiles + c],
                    ft_tCC_tiles[m * m_tiles + k],
                    ft_priorK[c * m_tiles + k],
                    n_tile_size,
                    m_tile_size,
                    m_tile_size,
                    Blas_trans,
                    Blas_no_trans);
            }
        }
    }
}

void prediction_uncertainty_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_inter,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(
            &diag_posterior,
            ft_priorK[i],
            ft_inter[i],
            m_tile_size,
            std::ref(gpu));
    }
}

void pred_uncer_tiled(
    std::vector<hpx::shared_future<double *>> &ft_priorK,
    std::vector<hpx::shared_future<double *>> &ft_vector,
    const std::size_t m_tile_size,
    const std::size_t m_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < m_tiles; i++)
    {
        ft_vector[i] = hpx::dataflow(
            &diag_tile,
            ft_priorK[i * m_tiles + i],
            m_tile_size,
            std::ref(gpu));
    }
}

void update_grad_K_tiled_mkl(
    std::vector<hpx::shared_future<double *>> &ft_tiles,
    const std::vector<hpx::shared_future<double *>> &ft_v1,
    const std::vector<hpx::shared_future<double *>> &ft_v2,
    const std::size_t n_tile_size,
    const std::size_t n_tiles,
    gpxpy::CUDA_GPU &gpu)
{
    for (std::size_t i = 0; i < n_tiles; ++i)
    {
        for (std::size_t j = 0; j < n_tiles; ++j)
        {
            auto [cublas, stream] = gpu.next_cublas_handle();

            ft_tiles[i * n_tiles + j] = hpx::dataflow(
                &ger,
                cublas,
                stream,
                ft_tiles[i * n_tiles + j],
                ft_v1[i],
                ft_v2[j],
                n_tile_size);
        }
    }
}

/**
 * TODO: not in header -> write documentation
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
    gpxpy::CUDA_GPU &gpu)
{
    // int cublas_counter = 0;
    // auto next_cublas = [&]()
    // { return cublas_handles[cublas_counter++ % gpu.n_streams]; };
    //
    // /// part 1: trace(inv(K)*grad_param)
    // std::vector<hpx::shared_future<std::vector<double>>> diag_tiles;
    // diag_tiles.resize(n_tiles);
    // for (std::size_t d = 0; d < n_tiles; d++)
    // {
    //     diag_tiles[d] =
    //         hpx::async(hpx::annotated_function(&gen_tile_zeros_diag,
    //                                            "assemble_tiled"),
    //                    n_tile_size);
    // }
    //
    // // Compute diagonal elements of inv(K) * grad_hyperparam
    // for (std::size_t i = 0; i < n_tiles; ++i)
    // {
    //     for (std::size_t j = 0; j < n_tiles; ++j)
    //     {
    //         diag_tiles[i] = hpx::dataflow(
    //             hpx::annotated_function(&dot_diag_gemm,
    //                                     "grad_left_tiled"),
    //             ft_invK[i * n_tiles + j],
    //             ft_gradparam[j * n_tiles + i],
    //             diag_tiles[i],
    //             n_tile_size,
    //             n_tile_size);
    //     }
    // }
    //
    // // compute trace(inv(K) * grad_hyperparam)
    // hpx::shared_future<double> grad_left =
    //     hpx::make_ready_future(0.0)
    //         .share();
    // for (std::size_t j = 0; j < n_tiles; ++j)
    // {
    //     grad_left = hpx::dataflow(
    //         hpx::annotated_function(hpx::unwrapping(&sum_gradleft),
    //                                 "grad_left_tiled"),
    //         diag_tiles[j],
    //         grad_left);
    // }
    //
    // /// part 2: alpha^T * grad_param * alpha
    // std::vector<hpx::shared_future<std::vector<double>>> inter_alpha;
    // inter_alpha.resize(n_tiles);
    // for (std::size_t d = 0; d < n_tiles; d++)
    // {
    //     inter_alpha[d] =
    //         hpx::async(hpx::annotated_function(&gen_tile_zeros_diag,
    //                                            "assemble_tiled"),
    //                    n_tile_size);
    // }
    //
    // for (std::size_t k = 0; k < n_tiles; k++)
    // {
    //     for (std::size_t m = 0; m < n_tiles; m++)
    //     {
    //         cublasHandle_t cublas = next_cublas();
    //         cublasSetStream(cublas, gpu.next_stream());
    //
    //         inter_alpha[k] = hpx::dataflow(
    //             &gemv,
    //             cublas,
    //             ft_gradparam[k * n_tiles + m],
    //             ft_alpha[m],
    //             inter_alpha[k],
    //             n_tile_size,
    //             n_tile_size,
    //             Blas_add,
    //             Blas_no_trans);
    //     }
    // }
    //
    // hpx::shared_future<double> grad_right =
    //     hpx::make_ready_future(0.0).share();
    // for (std::size_t j = 0; j < n_tiles;
    //      ++j)
    // {  // Compute inner product to obtain diagonal elements of
    //    // (K_MxN * (K^-1_NxN * K_NxM))
    //     grad_right = hpx::dataflow(
    //         hpx::annotated_function(hpx::unwrapping(&sum_gradright),
    //                                 "grad_right_tiled"),
    //         inter_alpha[j],
    //         ft_alpha[j],
    //         grad_right,
    //         n_tile_size);
    // }
    //
    // /// part 3: update parameter
    //
    // // compute gradient = grad_left + grad_r
    // hpx::shared_future<double> gradient = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&compute_gradient),
    //                             "gradient_tiled"),
    //     grad_left,
    //     grad_right,
    //     n_tile_size,
    //     n_tiles);
    //
    // // transform hyperparameter to unconstrained form
    // hpx::shared_future<double> unconstrained_param = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&to_unconstrained),
    //                             "gradient_tiled"),
    //     hyperparameter,
    //     false);
    // // update moments
    // m_T[param_idx] = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&update_first_moment),
    //                             "gradient_tiled"),
    //     gradient,
    //     m_T[param_idx],
    //     adam_params.beta1);
    // v_T[param_idx] = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&update_second_moment),
    //                             "gradient_tiled"),
    //     gradient,
    //     v_T[param_idx],
    //     adam_params.beta2);
    // // update unconstrained parameter
    // hpx::shared_future<double> updated_param =
    //     hpx::dataflow(hpx::annotated_function(hpx::unwrapping(&update_param),
    //                                           "gradient_tiled"),
    //                   unconstrained_param,
    //                   sek_params,
    //                   adam_params,
    //                   m_T[param_idx],
    //                   v_T[param_idx],
    //                   beta1_T,
    //                   beta2_T,
    //                   iter);
    // // transform hyperparameter to constrained form
    // return hpx::dataflow(
    //            hpx::annotated_function(hpx::unwrapping(&to_constrained),
    //                                    "gradient_tiled"),
    //            updated_param,
    //            false)
    //     .get();

    // TODO: requires GPU implementation of included functions
    return 0;
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
    gpxpy::CUDA_GPU &gpu)
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
        gpu);
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
    gpxpy::CUDA_GPU &gpu)
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
        gpu);
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
    gpxpy::CUDA_GPU &gpu)
{
    // // part1: compute trace(inv(K) * grad_hyperparam)
    // hpx::shared_future<double> grad_left =
    //     hpx::make_ready_future(0.0).share();
    // for (std::size_t j = 0; j < n_tiles; ++j)
    // {
    //     grad_left = hpx::dataflow(
    //         hpx::annotated_function(hpx::unwrapping(&sum_noise_gradleft),
    //                                 "grad_left_tiled"),
    //         ft_invK[j * n_tiles + j],
    //         grad_left,
    //         sek_params,
    //         n_tile_size,
    //         n_tiles);
    // }
    //
    // /// part 2: alpha^T * grad_param * alpha
    // hpx::shared_future<double> grad_right =
    //     hpx::make_ready_future(0.0).share();
    // for (std::size_t j = 0; j < n_tiles;
    //      ++j)
    // {  // Compute inner product to obtain diagonal elements of
    //    // (K_MxN * (K^-1_NxN * K_NxM))
    //     grad_right = hpx::dataflow(
    //         hpx::annotated_function(hpx::unwrapping(&sum_noise_gradright),
    //                                 "grad_right_tiled"),
    //         ft_alpha[j],
    //         grad_right,
    //         sek_params,
    //         n_tile_size);
    // }
    //
    // /// part 3: update parameter
    // hpx::shared_future<double> gradient = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&compute_gradient),
    //                             "gradient_tiled"),
    //     grad_left,
    //     grad_right,
    //     n_tile_size,
    //     n_tiles);
    // // transform hyperparameter to unconstrained form
    // hpx::shared_future<double> unconstrained_param = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&to_unconstrained),
    //                             "gradient_tiled"),
    //     sek_params.noise_variance,
    //     true);
    // // update moments
    // m_T[2] = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&update_first_moment),
    //                             "gradient_tiled"),
    //     gradient,
    //     m_T[2],
    //     adam_params.beta1);
    // v_T[2] = hpx::dataflow(
    //     hpx::annotated_function(hpx::unwrapping(&update_second_moment),
    //                             "gradient_tiled"),
    //     gradient,
    //     v_T[2],
    //     adam_params.beta2);
    // // update unconstrained parameter
    // hpx::shared_future<double> updated_param =
    //     hpx::dataflow(hpx::annotated_function(
    //                       hpx::unwrapping(&update_param), "gradient_tiled"),
    //                   unconstrained_param,
    //                   sek_params,
    //                   adam_params,
    //                   m_T[2],
    //                   v_T[2],
    //                   beta1_T,
    //                   beta2_T,
    //                   iter);
    // // transform hyperparameter to constrained form
    // return hpx::dataflow(hpx::annotated_function(
    //                          hpx::unwrapping(&to_constrained),
    //                          "gradient_tiled"),
    //                      updated_param,
    //                      true)
    //     .get();

    // TODO: requires GPU implementation of included functions
    return 0;
}

}  // namespace gpu
