#ifndef QUANT_TRADING_MATRIX_UTILS_HPP
#define QUANT_TRADING_MATRIX_UTILS_HPP

/**
 * @file matrix_utils.hpp
 * @brief Matrix operations and linear algebra utilities using Eigen
 *
 * Provides common matrix operations for quantitative finance:
 * - Covariance and correlation matrix computation
 * - Cholesky decomposition
 * - Safe matrix inversion with singularity detection
 * - Eigenvalue decomposition utilities
 */

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <stdexcept>
#include <cmath>

namespace quant::core {

/**
 * @brief Compute sample mean of each column
 *
 * @param data Matrix where each column is a variable's observations
 * @return Row vector of means
 */
inline Eigen::RowVectorXd column_means(const Eigen::MatrixXd& data) {
    return data.colwise().mean();
}

/**
 * @brief Compute sample covariance matrix from data
 *
 * Uses Bessel's correction (n-1 denominator) for unbiased estimation.
 *
 * @param data Matrix where each column is a variable's observations
 * @param ddof Delta degrees of freedom (default 1 for sample covariance)
 * @return Covariance matrix
 * @throws std::invalid_argument if not enough observations
 */
inline Eigen::MatrixXd compute_covariance(const Eigen::MatrixXd& data, int ddof = 1) {
    Eigen::Index n = data.rows();
    Eigen::Index p = data.cols();

    if (n <= ddof) {
        throw std::invalid_argument("Not enough observations for covariance computation");
    }

    // Center the data
    Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();

    // Compute covariance: (1/(n-ddof)) * X'X
    return (centered.transpose() * centered) / static_cast<double>(n - ddof);
}

/**
 * @brief Compute sample covariance matrix from return vectors
 *
 * @param returns Matrix where each column is an asset's return series
 * @return Covariance matrix of returns
 */
inline Eigen::MatrixXd returns_covariance(const Eigen::MatrixXd& returns) {
    return compute_covariance(returns, 1);
}

/**
 * @brief Convert covariance matrix to correlation matrix
 *
 * corr_ij = cov_ij / (std_i * std_j)
 *
 * @param cov Covariance matrix (must be square)
 * @return Correlation matrix with 1s on diagonal
 * @throws std::invalid_argument if matrix has zero variance on diagonal
 */
inline Eigen::MatrixXd covariance_to_correlation(const Eigen::MatrixXd& cov) {
    if (cov.rows() != cov.cols()) {
        throw std::invalid_argument("Covariance matrix must be square");
    }

    Eigen::Index n = cov.rows();
    Eigen::VectorXd std_devs = cov.diagonal().array().sqrt();

    // Check for zero variance
    for (Eigen::Index i = 0; i < n; ++i) {
        if (std_devs(i) < 1e-12) {
            throw std::invalid_argument("Covariance matrix has zero variance on diagonal");
        }
    }

    // Compute correlation: corr_ij = cov_ij / (std_i * std_j)
    Eigen::MatrixXd corr(n, n);
    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            corr(i, j) = cov(i, j) / (std_devs(i) * std_devs(j));
        }
    }

    return corr;
}

/**
 * @brief Convert correlation matrix back to covariance matrix
 *
 * @param corr Correlation matrix
 * @param std_devs Vector of standard deviations
 * @return Covariance matrix
 */
inline Eigen::MatrixXd correlation_to_covariance(const Eigen::MatrixXd& corr,
                                                  const Eigen::VectorXd& std_devs) {
    if (corr.rows() != corr.cols()) {
        throw std::invalid_argument("Correlation matrix must be square");
    }
    if (corr.rows() != std_devs.size()) {
        throw std::invalid_argument("Standard deviations vector size must match matrix dimensions");
    }

    Eigen::Index n = corr.rows();
    Eigen::MatrixXd cov(n, n);

    for (Eigen::Index i = 0; i < n; ++i) {
        for (Eigen::Index j = 0; j < n; ++j) {
            cov(i, j) = corr(i, j) * std_devs(i) * std_devs(j);
        }
    }

    return cov;
}

/**
 * @brief Compute condition number of a matrix
 *
 * The condition number is the ratio of largest to smallest singular value.
 * A large condition number indicates numerical instability.
 *
 * @param matrix Input matrix
 * @return Condition number (ratio of largest to smallest singular value)
 */
inline double condition_number(const Eigen::MatrixXd& matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
    Eigen::VectorXd singular_values = svd.singularValues();

    if (singular_values.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double max_sv = singular_values(0);
    double min_sv = singular_values(singular_values.size() - 1);

    if (min_sv < 1e-15) {
        return std::numeric_limits<double>::infinity();
    }

    return max_sv / min_sv;
}

/**
 * @brief Check if matrix is positive definite
 *
 * Uses Cholesky decomposition to check - if it succeeds, matrix is PD.
 *
 * @param matrix Matrix to check (must be square and symmetric)
 * @return true if positive definite
 */
inline bool is_positive_definite(const Eigen::MatrixXd& matrix) {
    if (matrix.rows() != matrix.cols()) {
        return false;
    }

    // Check symmetry
    if (!matrix.isApprox(matrix.transpose(), 1e-10)) {
        return false;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    return llt.info() == Eigen::Success;
}

/**
 * @brief Safe matrix inversion with singularity detection
 *
 * @param matrix Matrix to invert
 * @param tolerance Condition number tolerance (default 1e10)
 * @return Inverted matrix
 * @throws std::runtime_error if matrix is singular or ill-conditioned
 */
inline Eigen::MatrixXd safe_invert(const Eigen::MatrixXd& matrix, double tolerance = 1e10) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Cannot invert non-square matrix");
    }

    double cond = condition_number(matrix);
    if (cond > tolerance) {
        throw std::runtime_error("Matrix is singular or ill-conditioned (condition number: " +
                                 std::to_string(cond) + ")");
    }

    return matrix.inverse();
}

/**
 * @brief Compute Cholesky decomposition: A = LL^T
 *
 * @param matrix Positive definite matrix
 * @return Lower triangular Cholesky factor L
 * @throws std::runtime_error if matrix is not positive definite
 */
inline Eigen::MatrixXd cholesky_decomposition(const Eigen::MatrixXd& matrix) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Cholesky requires square matrix");
    }

    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky decomposition failed - matrix not positive definite");
    }

    return llt.matrixL();
}

/**
 * @brief Make a matrix positive definite by adjusting eigenvalues
 *
 * If a matrix has negative or zero eigenvalues, this function
 * shifts them to ensure the result is positive definite.
 *
 * @param matrix Input matrix (should be symmetric)
 * @param min_eigenvalue Minimum eigenvalue to enforce (default 1e-8)
 * @return Positive definite matrix
 */
inline Eigen::MatrixXd make_positive_definite(const Eigen::MatrixXd& matrix,
                                               double min_eigenvalue = 1e-8) {
    if (matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    // Make symmetric
    Eigen::MatrixXd sym = (matrix + matrix.transpose()) / 2.0;

    // Eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sym);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed");
    }

    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors();

    // Adjust negative eigenvalues
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_eigenvalue) {
            eigenvalues(i) = min_eigenvalue;
        }
    }

    // Reconstruct matrix: V * D * V^T
    return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

/**
 * @brief Solve linear system Ax = b using Cholesky decomposition
 *
 * More numerically stable than direct inversion when A is positive definite.
 *
 * @param A Positive definite matrix
 * @param b Right-hand side vector
 * @return Solution vector x
 */
inline Eigen::VectorXd solve_positive_definite(const Eigen::MatrixXd& A,
                                                const Eigen::VectorXd& b) {
    Eigen::LLT<Eigen::MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Matrix not positive definite");
    }
    return llt.solve(b);
}

/**
 * @brief Compute exponentially weighted covariance matrix
 *
 * More recent observations receive higher weight.
 *
 * @param returns Matrix of returns (each column is an asset)
 * @param lambda Decay factor (typical values: 0.94-0.97)
 * @return Exponentially weighted covariance matrix
 */
inline Eigen::MatrixXd ewma_covariance(const Eigen::MatrixXd& returns, double lambda = 0.94) {
    if (lambda <= 0.0 || lambda >= 1.0) {
        throw std::invalid_argument("Lambda must be in (0, 1)");
    }

    Eigen::Index n = returns.rows();
    Eigen::Index p = returns.cols();

    if (n < 2) {
        throw std::invalid_argument("Need at least 2 observations");
    }

    // Initialize with first observation's outer product
    Eigen::RowVectorXd mean = returns.colwise().mean();
    Eigen::MatrixXd centered = returns.rowwise() - mean;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(p, p);

    // Compute EWMA covariance
    double weight_sum = 0.0;
    for (Eigen::Index t = 0; t < n; ++t) {
        double weight = std::pow(lambda, n - 1 - t);
        weight_sum += weight;
        cov += weight * centered.row(t).transpose() * centered.row(t);
    }

    return cov / weight_sum;
}

}  // namespace quant::core

#endif  // QUANT_TRADING_MATRIX_UTILS_HPP
