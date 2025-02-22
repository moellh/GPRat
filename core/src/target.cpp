#include "target.hpp"

#include "cuda_utils.cuh"
#include <iostream>

#if GPXPY_WITH_CUDA
using hpx::cuda::experimental::check_cuda_error;
#endif

namespace gpxpy
{

CPU::CPU() { }

bool CPU::is_cpu() { return true; }

bool CPU::is_gpu() { return false; }

CUDA_GPU::CUDA_GPU(int id, int n_streams) :
    id(id),
    n_streams(n_streams),
    streams(),
    i_stream(0),
    shared_memory_size(0)
{
#if GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (id >= deviceCount)
    {
        throw std::runtime_error("Requested GPU device is not available.");
    }
#else
    throw not_compiled_with_cuda_exception();
#endif
}

bool CUDA_GPU::is_cpu() { return false; }

bool CUDA_GPU::is_gpu() { return true; }

#if GPXPY_WITH_CUDA
void CUDA_GPU::create()
{
    streams = std::vector<cudaStream_t>(n_streams);
    cublas_handles = std::vector<cublasHandle_t>(n_streams);
    for (size_t i = 0; i < streams.size(); ++i)
    {
        check_cuda_error(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        cublasCreate(&cublas_handles[i]);
    }
}

void CUDA_GPU::destroy()
{
    for (size_t i = 0; i < streams.size(); ++i)
    {
        check_cuda_error(cudaStreamDestroy(streams[i]));
        cublasDestroy(cublas_handles[i]);
    }
}

cudaStream_t CUDA_GPU::next_stream()
{
    return streams[i_stream++ % n_streams];
}

void CUDA_GPU::sync_streams(std::vector<cudaStream_t> &subset_of_streams)
{
    if (subset_of_streams.size() < streams.size())
    {
        for (cudaStream_t &stream : subset_of_streams)
        {
            check_cuda_error(cudaStreamSynchronize(stream));
        }
    }
    else
    {
        for (cudaStream_t &stream : streams)
        {
            check_cuda_error(cudaStreamSynchronize(stream));
        }
    }
}

std::pair<cublasHandle_t, cudaStream_t> CUDA_GPU::next_cublas_handle()
{
    std::size_t i = i_stream++;
    cublasHandle_t cublas = cublas_handles[i % n_streams];
    cudaStream_t stream = streams[i % n_streams];
    cublasSetStream(cublas, stream);

    return std::make_pair(cublas, stream);
}

#endif

CPU get_cpu() { return CPU(); }

CUDA_GPU get_gpu(int id, int n_executors)
{
#if GPXPY_WITH_CUDA
    return CUDA_GPU(id, n_executors);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

CUDA_GPU get_gpu()
{
#if GPXPY_WITH_CUDA
    return CUDA_GPU(0, 1);
#else
    throw not_compiled_with_cuda_exception();
#endif
}

void print_available_gpus()
{
#if GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        // clang-format off
        std::cout
            << "Device " << i << ": " << deviceProp.name << "\n"
            << "  Total Global Memory: " << deviceProp.totalGlobalMem << "\n"
            << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << "\n"
            << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n"
            << "  Total Constant Memory: " << deviceProp.totalConstMem << "\n"
            << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n"
            << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << "\n"
            << "  Clock Rate: " << deviceProp.clockRate << " kHz\n"
            << "  Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz\n"
            << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        // clang-format on
    }
#else
    std::cout
        << "CUDA is not available - There are no GPUs available. You can only "
           "`get_cpu()` to utilize the CPU for computation."
        << std::endl;
#endif
}

int gpu_count()
{
#if GPXPY_WITH_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
#else
    std::cout
        << "CUDA is not available - There are no GPUs available. You can only "
           "use `get_cpu()` to utilize the CPU for computation."
        << std::endl;
    return 0;
#endif
}

}  // namespace gpxpy
