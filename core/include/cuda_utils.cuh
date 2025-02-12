#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <hpx/algorithm.hpp>
#include <stdexcept>
#include <target.hpp>
#include <vector>
#include <hpx/async_cuda/cuda_exception.hpp>

#define BLOCK_SIZE 16

using hpx::cuda::experimental::check_cuda_error;

class not_compiled_with_cuda_exception : public std::runtime_error
{
  public:
    not_compiled_with_cuda_exception() :
        std::runtime_error("CUDA is not available because GPXPY has been compiled without CUDA.")
    { }
};

inline double *copy_to_device(const std::vector<double> &h_vector, gpxpy::CUDA_GPU &gpu)
{
    double *d_vector;
    check_cuda_error(cudaMalloc(&d_vector, h_vector.size() * sizeof(double)));
    cudaStream_t stream = gpu.next_stream();
    check_cuda_error(cudaMemcpyAsync(d_vector, h_vector.data(), h_vector.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    check_cuda_error(cudaStreamSynchronize(stream));
    return d_vector;
}

inline cusolverDnHandle_t create_cusolver_handle()
{
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    return handle;
}

inline std::vector<cublasHandle_t> create_cublas_handles(const int n_handles)
{
    std::vector<cublasHandle_t> handles(n_handles);
    for (int i = 0; i < n_handles; i++)
    {
        cublasCreate_v2(&handles[i]);
    }
    return handles;
}

inline void destroy(cusolverDnHandle_t handle)
{
    cusolverDnDestroy(handle);
}

inline void destroy(std::vector<cublasHandle_t> &handles)
{
    // clang-format off
    hpx::experimental::for_loop(hpx::execution::par, 0, handles.size(), [&](int i)
    {
        cublasDestroy_v2(handles[i]);
    });
    // clang-format on
}

inline void
free(std::vector<hpx::shared_future<double *>> &vector)
{
    for (auto &ptr : vector)
    {
        check_cuda_error(cudaFree(ptr.get()));
    }
}

#endif  // end of CUDA_UTILS_H
