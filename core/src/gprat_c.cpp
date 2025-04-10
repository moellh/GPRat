#include "gprat_c.hpp"

#include "cpu/gp_functions.hpp"
#include "utils_c.hpp"
#include <cstdio>

#ifdef GPRAT_WITH_CUDA
#include "gpu/gp_functions.hpp"
#endif

// namespace for GPRat library entities
namespace gprat
{

GP_data::GP_data(const std::string &f_path, int n, int n_reg) :
    file_path(f_path),
    n_samples(n),
    n_regressors(n_reg)
{
    data = utils::load_data(f_path, n, n_reg - 1);
}

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    n_regressors(n_regressors),
    sek_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2]),
    trainable_params(trainable_bool)
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    n_regressors(n_regressors),
    sek_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2]),
    trainable_params(trainable_bool),
    target(std::make_shared<CPU>())
{ }

GP::GP(std::vector<double> input,
       std::vector<double> output,
       int n_tiles,
       int n_tile_size,
       int n_regressors,
       std::vector<double> kernel_hyperparams,
       std::vector<bool> trainable_bool,
       int gpu_id,
       int n_streams) :
    _training_input(input),
    _training_output(output),
    _n_tiles(n_tiles),
    _n_tile_size(n_tile_size),
    n_regressors(n_regressors),
    sek_params(kernel_hyperparams[0], kernel_hyperparams[1], kernel_hyperparams[2]),
    trainable_params(trainable_bool),
    target(std::make_shared<CUDA_GPU>(CUDA_GPU(gpu_id, n_streams)))
{ }

std::string GP::repr() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "Kernel_Params: [lengthscale=" << sek_params.lengthscale << ", vertical_lengthscale="
        << sek_params.vertical_lengthscale << ", noise_variance=" << sek_params.noise_variance
        << ", n_regressors=" << n_regressors << "], Trainable_Params: [trainable_params l=" << trainable_params[0]
        << ", trainable_params v=" << trainable_params[1] << ", trainable_params n=" << trainable_params[2]
        << "], Target: [" << target->repr() << "], n_tiles=" << _n_tiles << ", n_tile_size=" << _n_tile_size;
    return oss.str();
}

std::vector<double> GP::get_training_input() const { return _training_input; }

std::vector<double> GP::get_training_output() const { return _training_output; }

std::vector<double> GP::predict(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       return gpu::predict(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target));
                   }
                   else
                   {
                       return cpu::predict(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params);
                   }
#else
                   return cpu::predict(
                       training_input,
                       training_output,
                       test_input,
                       n_tiles,
                       n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors,
                       sek_params);
#endif
               })
        .get();
}

std::vector<std::vector<double>>
GP::predict_with_uncertainty(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       return gpu::predict_with_uncertainty(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target));
                   }
                   else
                   {
                       return cpu::predict_with_uncertainty(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params);
                   }
#else
                   return cpu::predict_with_uncertainty(
                       training_input,
                       training_output,
                       test_input,
                       n_tiles,
                       n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors,
                       sek_params);
#endif
               })
        .get();
}

std::vector<std::vector<double>>
GP::predict_with_full_cov(const std::vector<double> &test_input, int m_tiles, int m_tile_size)
{
    return hpx::async(
               [this, &test_input, m_tiles, m_tile_size]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       return gpu::predict_with_full_cov(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target));
                   }
                   else
                   {
                       return cpu::predict_with_full_cov(
                           training_input,
                           training_output,
                           test_input,
                           n_tiles,
                           n_tile_size,
                           m_tiles,
                           m_tile_size,
                           n_regressors,
                           sek_params);
                   }
#else
                   return cpu::predict_with_full_cov(
                       training_input,
                       training_output,
                       test_input,
                       n_tiles,
                       n_tile_size,
                       m_tiles,
                       m_tile_size,
                       n_regressors,
                       sek_params);
#endif
               })
        .get();
}

std::vector<double> GP::optimize(const gprat_hyper::AdamParams &adam_params)
{
    return hpx::async(
               [this, &adam_params]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       std::err << "GP::optimze_step has not been implemented for the GPU.\n"
                                << "Instead, this operation executes the CPU implementation." << std::endl;
                   }
                   return cpu::optimize(
                       training_input,
                       training_output,
                       n_tiles,
                       n_tile_size,
                       n_regressors,
                       sek_params,
                       trainable_params,
                       adam_params);
#else
                   return cpu::optimize(
                       training_input,
                       training_output,
                       n_tiles,
                       n_tile_size,
                       n_regressors,
                       sek_params,
                       trainable_params,
                       adam_params);
#endif
               })
        .get();
}

double GP::optimize_step(gprat_hyper::AdamParams &adam_params, int iter)
{
    return hpx::async(
               [this, &adam_params, iter]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       std::err << "GP::optimze_step has not been implemented for the GPU.\n"
                                << "Instead, this operation executes the CPU implementation." << std::endl;
                   }
                   return cpu::optimize_step(
                       training_input,
                       training_output,
                       n_tiles,
                       n_tile_size,
                       n_regressors,
                       iter,
                       sek_params,
                       trainable_params,
                       adam_params);
#else
                   return cpu::optimize_step(
                       training_input,
                       training_output,
                       n_tiles,
                       n_tile_size,
                       n_regressors,
                       iter,
                       sek_params,
                       trainable_params,
                       adam_params);
#endif
               })
        .get();
}

double GP::calculate_loss()
{
    return hpx::async(
               [this]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       return gpu::compute_loss(
                           training_input,
                           training_output,
                           n_tiles,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target));
                   }
                   else
                   {
                       return cpu::compute_loss(
                           training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
                   }
#else
                   return cpu::compute_loss(
                       training_input, training_output, n_tiles, n_tile_size, n_regressors, sek_params);
#endif
               })
        .get();
}

std::vector<std::vector<double>> GP::cholesky()
{
    return hpx::async(
               [this]()
               {
#ifdef GPRAT_WITH_CUDA
                   if (target->is_gpu())
                   {
                       return gpu::cholesky(
                           training_input,
                           n_tiles,
                           n_tile_size,
                           n_regressors,
                           sek_params,
                           *std::dynamic_pointer_cast<gprat::CUDA_GPU>(target));
                   }
                   else
                   {
                       return cpu::cholesky(training_input, n_tiles, n_tile_size, n_regressors, sek_params);
                   }
#else
                   return cpu::cholesky(training_input, n_tiles, n_tile_size, n_regressors, sek_params);
#endif
               })
        .get();
}

}  // namespace gprat
