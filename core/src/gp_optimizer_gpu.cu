#include "gp_optimizer_gpu.cuh"

#include "adapter_cublas.cuh"
#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"

namespace gpu
{

double to_constrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(1.0 + exp(parameter)) + 1e-6;
    }
    else
    {
        return log(1.0 + exp(parameter));
    }
}

double
to_unconstrained(const double parameter, bool noise)
{
    if (noise)
    {
        return log(exp(parameter - 1e-6) - 1.0);
    }
    else
    {
        return log(exp(parameter) - 1.0);
    }
}

double
compute_sigmoid(const double parameter)
{
    return 1.0 / (1.0 + exp(-parameter));
}

__global__ void
compute_cov_dist_vec_kernel(
    double *d_tile,
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
        double covariance = exp(-0.5 * distance / (sek_params.lengthscale * sek_params.lengthscale));

        d_tile[i * n_tile_size + j] = covariance;
    }
}

hpx::shared_future<double *>
compute_cov_dist_vec(
    hpx::shared_future<double *> f_cov_dists,
    std::size_t row,
    std::size_t column,
    std::size_t n_tile_size,
    std::size_t n_regressors,
    gpxpy_hyper::SEKParams sek_params,
    const double *d_training_input,
    gpxpy::CUDA_GPU &gpu)
{
    dim3 threads_per_block(16, 16);
    dim3 n_blocks((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();

    double *d_tile = f_cov_dists.get();
    compute_cov_dist_vec_kernel<<<n_blocks, threads_per_block, gpu.shared_memory_size, stream>>>(d_tile, d_training_input, n_tile_size, n_regressors, row, column, sek_params);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(d_tile);
}

__global__ void
compute_tile_covariance_kernel(
    double *tile,
    std::size_t row,
    std::size_t col,
    std::size_t n_tile_size,
    gpxpy_hyper::SEKParams sek_params,
    const double *cov_dists)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n_tile_size && j < n_tile_size)
    {
        std::size_t i_global = n_tile_size * row + i;
        std::size_t j_global = n_tile_size * col + j;

        double covariance = sek_params.vertical_lengthscale * exp(cov_dists[i * n_tile_size + j]);
        if (i_global == j_global)
        {
            covariance += sek_params.noise_variance;
        }
        tile[i * n_tile_size + j] = covariance;
    }
}

hpx::shared_future<double *>
compute_tile_covariance_opt(
    hpx::shared_future<double *> f_tile,
    std::size_t row,
    std::size_t col,
    std::size_t n_tile_size,
    std::size_t n_regressors,
    gpxpy_hyper::SEKParams sek_params,
    const hpx::shared_future<double *> f_cov_dists,
    gpxpy::CUDA_GPU &gpu)
{
    std::size_t i_global, j_global;
    double *d_tile = f_tile.get();
    double *d_cov_dists = f_cov_dists.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((n_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    compute_tile_covariance_kernel<<<n_blocks, threads_per_block, 0, stream>>>(d_tile, row, col, n_tile_size, sek_params, d_cov_dists);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(d_tile);
}

__global__ void
compute_tile_grad_v_kernel(
    double *tile,
    const double *cov_dists,
    std::size_t N,
    double *hyperparam)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        tile[i * N + j] = exp(cov_dists[i * N + j]) * *hyperparam;
    }
}

hpx::shared_future<double *>
compute_tile_grad_v(
    hpx::shared_future<double *> f_tile,
    std::size_t row,
    std::size_t col,
    std::size_t N,
    std::size_t n_regressors,
    gpxpy_hyper::SEKParams sek_params,
    const hpx::shared_future<double *> cov_dists,
    gpxpy::CUDA_GPU &gpu)
{
    cudaStream_t stream = gpu.next_stream();

    // Initialize tile
    double *d_tile = f_tile.get();
    double *d_cov_dists = cov_dists.get();

    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.vertical_lengthscale, false));
    double *d_hyperparam;
    check_cuda_error(cudaMemcpyAsync(d_hyperparam, &hyperparam_der, sizeof(double), cudaMemcpyHostToDevice, stream));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    compute_tile_grad_v_kernel<<<gridDim, blockDim, gpu.shared_memory_size, stream>>>(
        d_tile,
        d_cov_dists,
        N,
        d_hyperparam);

    check_cuda_error(cudaStreamSynchronize(stream));
    return hpx::make_ready_future(d_tile);
}

__global__ void
compute_tile_grad_l_kernel(
    double *tile,
    const double *cov_dists,
    std::size_t N,
    double *lengthscale,
    double *vertical_lengthscale,
    double *hyperparam)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        tile[i * N + j] = -2.0 * ((*vertical_lengthscale) / (*lengthscale)) * cov_dists[i * N + j] * exp(cov_dists[i * N + j]) * (*hyperparam);
    }
}

hpx::shared_future<double *>
compute_tile_grad_l(
    hpx::shared_future<double *> f_tile,
    std::size_t row,
    std::size_t col,
    std::size_t n_tile_size,
    std::size_t n_regressors,
    gpxpy_hyper::SEKParams sek_params,
    const hpx::shared_future<double *> cov_dists,
    gpxpy::CUDA_GPU &gpu)
{
    cudaStream_t stream = gpu.next_stream();

    double *d_tile = f_tile.get();
    double hyperparam_der = compute_sigmoid(to_unconstrained(sek_params.lengthscale, false));
    double *d_hyperparam;
    check_cuda_error(cudaMemcpyAsync(d_hyperparam, &hyperparam_der, sizeof(double), cudaMemcpyHostToDevice, stream));
    double *d_lengthscale;
    check_cuda_error(cudaMemcpyAsync(d_lengthscale, &sek_params.lengthscale, sizeof(double), cudaMemcpyHostToDevice, stream));
    double *d_vertical_lengthscale;
    check_cuda_error(cudaMemcpyAsync(d_vertical_lengthscale, &sek_params.vertical_lengthscale, sizeof(double), cudaMemcpyHostToDevice, stream));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_tile_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    compute_tile_grad_l_kernel<<<gridDim, blockDim, gpu.shared_memory_size, stream>>>(
        d_tile,
        cov_dists.get(),
        n_tile_size,
        d_lengthscale,
        d_vertical_lengthscale,
        d_hyperparam);
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(d_tile);
}

hpx::shared_future<double *>
compute_transpose(
    std::size_t N,
    const hpx::shared_future<double *> f_tile,
    const hpx::shared_future<double *> f_tile_trans,
    gpxpy::CUDA_GPU &gpu)
{
    double *d_tile = f_tile.get();
    double *d_tile_trans = f_tile_trans.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose<<<n_blocks, threads_per_block, 0, stream>>>(d_tile_trans, d_tile, N, N);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(d_tile_trans);
}

hpx::shared_future<double *>
gen_tile_grad_l_trans(
    std::size_t N,
    const hpx::shared_future<double *> f_grad_l_tile,
    gpxpy::CUDA_GPU &gpu)
{
    double *transposed;
    check_cuda_error(cudaMalloc(&transposed, N * N * sizeof(double)));
    double *d_grad_l_tile = f_grad_l_tile.get();

    cudaStream_t stream = gpu.next_stream();
    dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose<<<n_blocks, threads_per_block, 0, stream>>>(transposed, d_grad_l_tile, N, N);

    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(transposed);
}

double gen_beta_T(int t, double beta)
{
    return pow(beta, t);
}

__global__ void
add_log_squared_K_diag(
    double *K_diag_tile,
    double *alpha_tile,
    double *y_tile,
    double *loss,
    std::size_t N)
{
    for (std::size_t i = 0; i < N; i++)
    {
        *loss += log(K_diag_tile[i * N + i] * K_diag_tile[i * N + i]);
    }
}

hpx::shared_future<double>
compute_loss(
    const hpx::shared_future<double *> &K_diag_tile,
    const hpx::shared_future<double *> &alpha_tile,
    const hpx::shared_future<double *> &y_tile,
    std::size_t N,
    gpxpy::CUDA_GPU &gpu)
{
    auto [cublas, stream] = gpu.next_cublas_handle();

    hpx::shared_future<double *> d_loss = dot(cublas, stream, y_tile, alpha_tile, N);
    add_log_squared_K_diag<<<1, 1, 0, stream>>>(K_diag_tile.get(), alpha_tile.get(), y_tile.get(), d_loss.get(), N);

    double h_loss;
    check_cuda_error(cudaMemcpyAsync(&h_loss, d_loss.get(), sizeof(double), cudaMemcpyDeviceToHost, stream));
    check_cuda_error(cudaFree(d_loss.get()));
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(h_loss);
}

hpx::shared_future<double>
add_losses(
    const std::vector<hpx::shared_future<double>> &losses,
    std::size_t n_tile_size,
    std::size_t n_tiles)
{
    // Add the squared difference to the error
    double l = 0.0;
    for (std::size_t i = 0; i < n_tiles; i++)
    {
        l += losses[i].get();
    }
    l += n_tile_size * n_tiles * log(2.0 * M_PI);

    return hpx::make_ready_future(0.5 * l / (n_tile_size * n_tiles));
}

double compute_gradient(const double &grad_l,
                        const double &grad_r,
                        std::size_t N,
                        std::size_t n_tiles)
{
    double grad = 0.0;
    grad = 1.0 / (2.0 * N * n_tiles) * (grad_l - grad_r);

    return std::move(grad);
}

double compute_gradient_noise(const std::vector<std::vector<double>> &ft_tiles,
                              double noise_variance,
                              std::size_t N,
                              std::size_t n_tiles)
{
    // Initialize tile
    double trace = 0.0;
    double hyperparam_der =
        compute_sigmoid(to_unconstrained(noise_variance, true));
    for (std::size_t d = 0; d < n_tiles; d++)
    {
        auto tile = ft_tiles[d * n_tiles + d];
        for (std::size_t i = 0; i < N; ++i)
        {
            trace += (tile[i * N + i] * hyperparam_der);
        }
    }
    trace = 1.0 / (2.0 * N * n_tiles) * trace;
    return std::move(trace);
}

double
update_first_moment(const double &gradient, double m_T, const double &beta_1)
{
    return beta_1 * m_T + (1.0 - beta_1) * gradient;
}

double
update_second_moment(const double &gradient, double v_T, const double &beta_2)
{
    return beta_2 * v_T + (1.0 - beta_2) * gradient * gradient;
}

hpx::shared_future<double> update_param(const double unconstrained_hyperparam,
                                        gpxpy_hyper::SEKParams sek_params,
                                        gpxpy_hyper::AdamParams adam_params,
                                        double m_T,
                                        double v_T,
                                        const std::vector<double> beta1_T,
                                        const std::vector<double> beta2_T,
                                        int iter)
{
    double alpha_T =
        sek_params.noise_variance * sqrt(1.0 - beta2_T[iter]) / (1.0 - beta1_T[iter]);
    return hpx::make_ready_future(unconstrained_hyperparam - alpha_T * m_T / (sqrt(v_T) + adam_params.epsilon));
}

__global__ void
fill_identity_kernel(
    double *tile,
    std::size_t N,
    std::size_t row,
    std::size_t col)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
    {
        std::size_t i_global = N * row + i;
        std::size_t j_global = N * col + j;
        tile[i * N + j] = (i_global == j_global) ? 1.0 : 0.0;
    }
}

hpx::shared_future<double *>
compute_tile_identity(
    hpx::shared_future<double *> f_tile,
    std::size_t tile_row,
    std::size_t tile_column,
    std::size_t n_tile_size,
    gpxpy::CUDA_GPU &gpu)
{
    double *d_tile = f_tile.get();
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n_tile_size + 15) / 16, (n_tile_size + 15) / 16);

    cudaStream_t stream = gpu.next_stream();
    fill_identity_kernel<<<gridDim, blockDim, gpu.shared_memory_size, stream>>>(
        d_tile,
        n_tile_size,
        tile_row,
        tile_column);
    check_cuda_error(cudaStreamSynchronize(stream));

    return hpx::make_ready_future(d_tile);
}

std::vector<double> gen_tile_zeros_diag(std::size_t N)
{
    // Initialize tile
    std::vector<double> tile;
    tile.resize(N);
    std::fill(tile.begin(), tile.end(), 0.0);
    return std::move(tile);
}

double gen_moment()
{
    double z = 0.0;
    return z;
}

double sum_gradleft(const std::vector<double> &diagonal, double grad)
{
    grad += std::reduce(diagonal.begin(), diagonal.end());
    return grad;
}

double sum_gradright(const std::vector<double> &inter_alpha,
                     const std::vector<double> &alpha,
                     double grad,
                     std::size_t N)
{
    // grad += dot(inter_alpha, alpha, N);
    // return grad;
    return 0.0;
}

double sum_noise_gradleft(const std::vector<double> &ft_invK,
                          double grad,
                          gpxpy_hyper::SEKParams sek_params,
                          std::size_t N,
                          std::size_t n_tiles)
{
    double noise_der =
        compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    for (std::size_t i = 0; i < N; ++i)
    {
        grad += (ft_invK[i * N + i] * noise_der);
    }
    return std::move(grad);
}

double sum_noise_gradright(const std::vector<double> &alpha,
                           double grad,
                           gpxpy_hyper::SEKParams sek_params,
                           std::size_t N)
{
    // double noise_der =
    //     compute_sigmoid(to_unconstrained(sek_params.noise_variance, true));
    // grad += (noise_der * dot(alpha, alpha, N));
    // return grad;
    return 0.0;
}

}  // namespace gpu
