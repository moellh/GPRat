#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

__global__ void
transpose(double *transposed, double *original, std::size_t width, std::size_t height);

#endif  // CUDA_KERNELS_H
