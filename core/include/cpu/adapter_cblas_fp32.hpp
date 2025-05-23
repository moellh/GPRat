#ifndef CPU_ADAPTER_CBLAS_FP32_H
#define CPU_ADAPTER_CBLAS_FP32_H

#include <hpx/future.hpp>
#include <vector>
using vector_future = hpx::shared_future<std::vector<float>>;

// Constants that are compatible with CBLAS
typedef enum BLAS_TRANSPOSE { Blas_no_trans = 111, Blas_trans = 112 } BLAS_TRANSPOSE;

typedef enum BLAS_SIDE { Blas_left = 141, Blas_right = 142 } BLAS_SIDE;

typedef enum BLAS_ALPHA { Blas_add = 1, Blas_substract = -1 } BLAS_ALPHA;

// typedef enum BLAS_UPLO { Blas_upper = 121,
//                          Blas_lower = 122 } BLAS_UPLO;

// typedef enum BLAS_ORDERING { Blas_row_major = 101,
//                              Blas_col_major = 102 } BLAS_ORDERING;

// BLAS level 3 operations

/**
 * @brief FP32 In-place Cholesky decomposition of A
 * @param f_A matrix to be factorized
 * @param N matrix dimension
 * @return factorized, lower triangular matrix f_L
 */
vector_future potrf(vector_future f_A, const int N);

/**
 * @brief FP32 In-place solve L(^T) * X = A or X * L(^T) = A where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_A right hand side matrix
 * @param N first dimension
 * @param M second dimension
 * @return solution matrix f_X
 */
vector_future trsm(vector_future f_L,
                   vector_future f_A,
                   const int N,
                   const int M,
                   const BLAS_TRANSPOSE transpose_L,
                   const BLAS_SIDE side_L);

/**
 * @brief FP32 Symmetric rank-k update: A = A - B * B^T
 * @param f_A Base matrix
 * @param f_B Symmetric update matrix
 * @param N matrix dimension
 * @return updated matrix f_A
 */
vector_future syrk(vector_future f_A, vector_future f_B, const int N);

/**
 * @brief FP32 General matrix-matrix multiplication: C = C - A(^T) * B(^T)
 * @param f_C Base matrix
 * @param f_B Right update matrix
 * @param f_A Left update matrix
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @param K third matrix dimension
 * @param transpose_A transpose left matrix
 * @param transpose_B transpose right matrix
 * @return updated matrix f_X
 */
vector_future
gemm(vector_future f_A,
     vector_future f_B,
     vector_future f_C,
     const int N,
     const int M,
     const int K,
     const BLAS_TRANSPOSE transpose_A,
     const BLAS_TRANSPOSE transpose_B);

// BLAS level 2 operations

/**
 * @brief FP32 In-place solve L(^T) * x = a where L lower triangular
 * @param f_L Cholesky factor matrix
 * @param f_a right hand side vector
 * @param N matrix dimension
 * @param transpose_L transpose Cholesky factor
 * @return solution vector f_x
 */
vector_future trsv(vector_future f_L, vector_future f_a, const int N, const BLAS_TRANSPOSE transpose_L);

/**
 * @brief FP32 General matrix-vector multiplication: b = b - A(^T) * a
 * @param f_A update matrix
 * @param f_a update vector
 * @param f_b base vector
 * @param N matrix dimension
 * @param alpha add or substract update to base vector
 * @param transpose_A transpose update matrix
 * @return updated vector f_b
 */
vector_future gemv(vector_future f_A,
                   vector_future f_a,
                   vector_future f_b,
                   const int N,
                   const int M,
                   const BLAS_ALPHA alpha,
                   const BLAS_TRANSPOSE transpose_A);

/**
 * @brief FP32 Vector update with diagonal SYRK: r = r + diag(A^T * A)
 * @param f_A update matrix
 * @param f_r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector f_r
 */
vector_future dot_diag_syrk(vector_future f_A, vector_future f_r, const int N, const int M);

/**
 * @brief FP32 Vector update with diagonal GEMM: r = r + diag(A * B)
 * @param f_A first update matrix
 * @param f_B second update matrix
 * @param f_r base vector
 * @param N first matrix dimension
 * @param M second matrix dimension
 * @return updated vector f_r
 */
vector_future dot_diag_gemm(vector_future f_A, vector_future f_B, vector_future f_r, const int N, const int M);

// BLAS level 1 operations

/**
 * @brief FP32 AXPY: y - x
 * @param f_y left vector
 * @param f_x right vector
 * @param N vector length
 * @return y - x
 */
vector_future axpy(vector_future f_y, vector_future f_x, const int N);

/**
 * @brief FP32 Dot product: a * b
 * @param f_a left vector
 * @param f_b right vector
 * @param N vector length
 * @return f_a * f_b
 */
float dot(std::vector<float> a, std::vector<float> b, const int N);

#endif  // end of CPU_ADAPTER_CBLAS_FP32_H
