#ifndef TILED_ALGORITHMS_CPU
#define TILED_ALGORITHMS_CPU

#include <hpx/future.hpp>

using Tiled_matrix = std::vector<hpx::shared_future<std::vector<double>>>;
using Tiled_vector = std::vector<hpx::shared_future<std::vector<double>>>;

// Tiled Cholesky Algorithm ------------------------------------------------ {{{

/**
 * @brief Perform right-looking tiled Cholesky decomposition.
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles, containing the
 *        covariance matrix, afterwards the Cholesky decomposition.
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void right_looking_cholesky_tiled(Tiled_matrix &ft_tiles, int N, std::size_t n_tiles);

// }}} ----------------------------------------- end of Tiled Cholesky Algorithm

// Tiled Triangular Solve Algorithms --------------------------------------- {{{

/**
 * @brief Perform tiled forward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void forward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, int N, std::size_t n_tiles);

/**
 * @brief Perform tiled backward triangular matrix-vector solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side vector, afterwards containing the tiled solution vector
 * @param N Tile size per dimension.
 * @param n_tiles Number of tiles per dimension.
 */
void backward_solve_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_rhs, int N, std::size_t n_tiles);

/**
 * @brief Perform tiled forward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void forward_solve_tiled_matrix(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_rhs, int N, int M, std::size_t n_tiles, std::size_t m_tiles);

/**
 * @brief Perform tiled backward triangular matrix-matrix solve.
 *
 * @param ft_tiles Tiled triangular matrix represented as a vector of futurized tiles.
 * @param ft_rhs Tiled right-hand side matrix, afterwards containing the tiled solution matrix.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void backward_solve_tiled_matrix(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_rhs, int N, int M, std::size_t n_tiles, std::size_t m_tiles);

// }}} -------------------------------- end of Tiled Triangular Solve Algorithms

/**
 * @brief Perform tiled matrix-vector multiplication
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector represented as a vector of futurized tiles.
 * @param ft_rhsTiled solution represented as a vector of futurized tiles.
 * @param N_row Tile size of first dimension.
 * @param N_col Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void matrix_vector_tiled(Tiled_matrix &ft_tiles,
                         Tiled_vector &ft_vector,
                         Tiled_vector &ft_rhs,
                         int N_row,
                         int N_col,
                         std::size_t n_tiles,
                         std::size_t m_tiles);

// // Tiled Diagonal of Posterior Covariance Matrix
// void posterior_covariance_tiled(std::vector<hpx::shared_future<std::vector<double>>> &ft_tCC_tiles,
//                                 std::vector<hpx::shared_future<std::vector<double>>> &ft_inter_tiles,
//                                 int N,
//                                 int M,
//                                 std::size_t n_tiles,
//                                 std::size_t m_tiles);
/**
 * @brief Perform tiled symmetric k-rank update on diagonal tiles
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector holding the diagonal tile results
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void symmetric_matrix_matrix_diagonal_tiled(
    Tiled_matrix &ft_tiles, Tiled_vector &ft_vector, int N, int M, std::size_t n_tiles, std::size_t m_tiles);

/**
 * @brief Perform tiled symmetric k-rank update (ft_tiles^T * ft_tiles)
 *
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_result Tiled matrix holding the result of the computationi.
 * @param N Tile size of first dimension.
 * @param M Tile size of second dimension.
 * @param n_tiles Number of tiles in first dimension.
 * @param m_tiles Number of tiles in second dimension.
 */
void symmetric_matrix_matrix_tiled(
    Tiled_matrix &ft_tiles, Tiled_matrix &ft_result, int N, int M, std::size_t n_tiles, std::size_t m_tiles);

/**
 * @brief Compute the difference between two tiled vectors
 * @param ft_minuend Tiled vector that is being subtracted from.
 * @param ft_subtrahend Tiled vector that is being subtracted.
 * @param ft_difference Tiled vector that contains the result of the substraction.
 * @param M Tile size dimension.
 * @param m_tiles Number of tiles.
 */
void vector_difference_tiled(Tiled_vector &ft_minuend, Tiled_vector &ft_substrahend, int M, std::size_t m_tiles);

/**
 * @brief Extract the tiled diagonals of a tiled matrix
 * @param ft_tiles Tiled matrix represented as a vector of futurized tiles.
 * @param ft_vector Tiled vector containing the diagonals of the matrix tiles
 * @param M Tile size per dimension.
 * @param m_tiles Number of tiles per dimension.
 */
void matrix_diagonal_tiled(Tiled_matrix &ft_tiles, Tiled_vector &ft_vector, int M, std::size_t m_tiles);

// Tiled Loss
void compute_loss_tiled(Tiled_matrix &ft_tiles,
                        Tiled_vector &ft_alpha,
                        Tiled_vector &ft_y,
                        hpx::shared_future<double> &loss,
                        int N,
                        std::size_t n_tiles);

// Compute I-y*y^T*inv(K)
void update_grad_K_tiled_mkl(std::vector<hpx::shared_future<std::vector<double>>> &ft_tiles,
                             const std::vector<hpx::shared_future<std::vector<double>>> &ft_v1,
                             const std::vector<hpx::shared_future<std::vector<double>>> &ft_v2,
                             int N,
                             std::size_t n_tiles);

// Perform a gradient scent step for selected hyperparameter using Adam
// algorithm
void update_hyperparameter(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_gradparam,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::vector<double> &hyperparameters,
    int N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    std::size_t iter,
    std::size_t param_idx);
// Update noise variance using gradient decent + Adam
void update_noise_variance(
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_invK,
    const std::vector<hpx::shared_future<std::vector<double>>> &ft_alpha,
    std::vector<double> &hyperparameters,
    int N,
    std::size_t n_tiles,
    std::vector<hpx::shared_future<double>> &m_T,
    std::vector<hpx::shared_future<double>> &v_T,
    const std::vector<hpx::shared_future<double>> &beta1_T,
    const std::vector<hpx::shared_future<double>> &beta2_T,
    std::size_t iter);

#endif
