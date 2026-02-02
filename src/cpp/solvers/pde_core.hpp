#ifndef QUANT_TRADING_PDE_CORE_HPP
#define QUANT_TRADING_PDE_CORE_HPP

/**
 * @file pde_core.hpp
 * @brief Core infrastructure for PDE solvers
 *
 * Provides grid classes, boundary conditions, and time-stepping utilities
 * for finite difference PDE solvers.
 *
 * Reference: Duffy, D.J. (2006). "Finite Difference Methods in Financial Engineering"
 */

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace quant::solvers {

/**
 * @brief 1D spatial grid for PDE discretization.
 *
 * Supports both uniform and logarithmic spacing. Log-spacing is
 * particularly useful for asset price grids where we need more
 * resolution near the strike price.
 */
class Grid1D {
public:
    /**
     * Create spatial grid.
     *
     * @param x_min Minimum value
     * @param x_max Maximum value
     * @param n_points Number of grid points
     * @param use_log_space If true, grid is uniform in log(x)
     */
    Grid1D(double x_min, double x_max, size_t n_points, bool use_log_space = false)
        : log_space_(use_log_space), n_points_(n_points) {

        if (n_points < 3) {
            throw std::invalid_argument("Grid1D requires at least 3 points");
        }
        if (x_min >= x_max) {
            throw std::invalid_argument("Grid1D: x_min must be less than x_max");
        }
        if (use_log_space && x_min <= 0) {
            throw std::invalid_argument("Grid1D: log-space requires x_min > 0");
        }

        x_.resize(n_points);
        dx_.resize(n_points - 1);

        if (use_log_space) {
            double log_min = std::log(x_min);
            double log_max = std::log(x_max);
            double log_step = (log_max - log_min) / (n_points - 1);

            for (size_t i = 0; i < n_points; ++i) {
                x_[i] = std::exp(log_min + i * log_step);
            }
        } else {
            double step = (x_max - x_min) / (n_points - 1);
            for (size_t i = 0; i < n_points; ++i) {
                x_[i] = x_min + i * step;
            }
        }

        // Compute spacings
        for (size_t i = 0; i < n_points - 1; ++i) {
            dx_[i] = x_[i + 1] - x_[i];
        }
    }

    size_t size() const { return n_points_; }
    double operator[](size_t i) const { return x_[i]; }
    double dx(size_t i) const { return dx_[std::min(i, n_points_ - 2)]; }
    const Eigen::VectorXd& points() const { return x_; }
    bool is_log_space() const { return log_space_; }

    /**
     * Get uniform step in the appropriate coordinate system.
     * For log-space grids, returns step in log coordinates.
     * For uniform grids, returns step in physical coordinates.
     */
    double uniform_step() const {
        if (log_space_) {
            return std::log(x_[n_points_ - 1] / x_[0]) / (n_points_ - 1);
        }
        return (x_[n_points_ - 1] - x_[0]) / (n_points_ - 1);
    }

    double min() const { return x_[0]; }
    double max() const { return x_[n_points_ - 1]; }

    /**
     * Find index of grid point closest to x.
     */
    size_t find_index(double x) const {
        if (x <= x_[0]) return 0;
        if (x >= x_[n_points_ - 1]) return n_points_ - 1;

        // Binary search
        size_t lo = 0, hi = n_points_ - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (x_[mid] <= x) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Return closer point
        return (x - x_[lo] < x_[hi] - x) ? lo : hi;
    }

    /**
     * Linear interpolation at point x.
     */
    double interpolate(double x, const Eigen::VectorXd& values) const {
        if (x <= x_[0]) return values[0];
        if (x >= x_[n_points_ - 1]) return values[n_points_ - 1];

        size_t i = find_index(x);
        if (i == 0) i = 1;
        if (i >= n_points_ - 1) i = n_points_ - 2;

        double t = (x - x_[i - 1]) / (x_[i] - x_[i - 1]);
        return (1 - t) * values[i - 1] + t * values[i];
    }

private:
    Eigen::VectorXd x_;
    Eigen::VectorXd dx_;
    bool log_space_;
    size_t n_points_;
};


/**
 * @brief 2D spatial grid for 2D PDEs (e.g., Heston model).
 *
 * Combines two 1D grids for the two spatial dimensions.
 * Uses row-major ordering for linear indexing: linear = i * ny + j
 */
class Grid2D {
public:
    Grid2D(const Grid1D& x_grid, const Grid1D& y_grid)
        : x_grid_(x_grid), y_grid_(y_grid) {}

    size_t nx() const { return x_grid_.size(); }
    size_t ny() const { return y_grid_.size(); }
    size_t size() const { return nx() * ny(); }

    const Grid1D& x_grid() const { return x_grid_; }
    const Grid1D& y_grid() const { return y_grid_; }

    /// Convert (i,j) 2D index to linear index
    size_t to_linear(size_t i, size_t j) const {
        return i * ny() + j;
    }

    /// Convert linear index to (i,j)
    std::pair<size_t, size_t> to_2d(size_t linear) const {
        return {linear / ny(), linear % ny()};
    }

    /// Get x value at index i
    double x(size_t i) const { return x_grid_[i]; }

    /// Get y value at index j
    double y(size_t j) const { return y_grid_[j]; }

private:
    Grid1D x_grid_;
    Grid1D y_grid_;
};


/**
 * @brief Time-stepping scheme enumeration.
 */
enum class TimeScheme {
    Explicit,       ///< Forward Euler (conditionally stable, CFL condition)
    Implicit,       ///< Backward Euler (unconditionally stable, 1st order)
    CrankNicolson,  ///< Crank-Nicolson (unconditionally stable, 2nd order)
    ADI             ///< Alternating Direction Implicit (for 2D problems)
};


/**
 * @brief Boundary condition type enumeration.
 */
enum class BoundaryType {
    Dirichlet,  ///< Fixed value: V(boundary) = g
    Neumann,    ///< Fixed derivative: dV/dx(boundary) = g
    Linear      ///< Linear extrapolation from interior
};


/**
 * @brief Abstract boundary condition interface.
 */
class BoundaryCondition {
public:
    virtual ~BoundaryCondition() = default;

    /**
     * Get boundary value at a given point and time.
     *
     * @param x Boundary point
     * @param t Current time
     * @return Boundary value or derivative
     */
    virtual double value(double x, double t) const = 0;

    /**
     * Get boundary type.
     */
    virtual BoundaryType type() const = 0;
};


/**
 * @brief Dirichlet boundary condition: V(x_boundary) = g(t)
 */
class DirichletBC : public BoundaryCondition {
public:
    /**
     * Constant Dirichlet condition.
     */
    explicit DirichletBC(double value) : value_func_([value](double, double) { return value; }) {}

    /**
     * Time-dependent Dirichlet condition.
     */
    explicit DirichletBC(std::function<double(double, double)> value_func)
        : value_func_(std::move(value_func)) {}

    double value(double x, double t) const override { return value_func_(x, t); }
    BoundaryType type() const override { return BoundaryType::Dirichlet; }

private:
    std::function<double(double, double)> value_func_;
};


/**
 * @brief Neumann boundary condition: dV/dx(x_boundary) = g(t)
 */
class NeumannBC : public BoundaryCondition {
public:
    explicit NeumannBC(double derivative) : deriv_(derivative) {}

    double value(double x, double t) const override { return deriv_; }
    BoundaryType type() const override { return BoundaryType::Neumann; }

private:
    double deriv_;
};


/**
 * @brief Linear extrapolation boundary condition.
 *
 * Extrapolates linearly from the two interior points.
 */
class LinearBC : public BoundaryCondition {
public:
    LinearBC() = default;

    double value(double x, double t) const override { return 0.0; }  // Not used directly
    BoundaryType type() const override { return BoundaryType::Linear; }
};


/**
 * @brief Check CFL stability condition for explicit schemes.
 *
 * For diffusion equation: dt <= dx^2 / (2 * D)
 * For advection-diffusion: dt <= min(dx/|v|, dx^2/(2*D))
 *
 * @param dt Time step
 * @param dx Spatial step
 * @param diffusion Diffusion coefficient
 * @param advection Advection velocity (optional)
 * @return true if CFL condition is satisfied
 */
inline bool check_cfl_condition(double dt, double dx, double diffusion, double advection = 0.0) {
    double cfl_diffusion = (dx * dx) / (2.0 * std::abs(diffusion) + 1e-14);
    double cfl_advection = (advection != 0.0) ? dx / std::abs(advection) : 1e10;
    return dt <= std::min(cfl_diffusion, cfl_advection);
}


/**
 * @brief Compute suggested time step based on CFL condition.
 *
 * @param dx Spatial step
 * @param diffusion Diffusion coefficient
 * @param cfl_factor Safety factor (default 0.9)
 * @return Suggested time step
 */
inline double compute_stable_dt(double dx, double diffusion, double cfl_factor = 0.9) {
    return cfl_factor * (dx * dx) / (2.0 * std::abs(diffusion) + 1e-14);
}


/**
 * @brief Build tridiagonal matrix for 1D diffusion operator.
 *
 * Discretizes: d²V/dx² using central differences
 *
 * @param grid 1D spatial grid
 * @param diffusion Diffusion coefficient (can vary with x)
 * @return Sparse tridiagonal matrix
 */
inline Eigen::SparseMatrix<double> build_diffusion_matrix(
    const Grid1D& grid,
    const std::function<double(double)>& diffusion) {

    size_t n = grid.size();
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * n);

    for (size_t i = 1; i < n - 1; ++i) {
        double dx_m = grid[i] - grid[i - 1];
        double dx_p = grid[i + 1] - grid[i];
        double dx_avg = 0.5 * (dx_m + dx_p);
        double D = diffusion(grid[i]);

        double coef_m = D / (dx_m * dx_avg);
        double coef_p = D / (dx_p * dx_avg);
        double coef_c = -(coef_m + coef_p);

        triplets.emplace_back(i, i - 1, coef_m);
        triplets.emplace_back(i, i, coef_c);
        triplets.emplace_back(i, i + 1, coef_p);
    }

    // Boundary rows (identity for now, will be modified by BCs)
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(n - 1, n - 1, 1.0);

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}


/**
 * @brief Build tridiagonal matrix for 1D advection operator.
 *
 * Discretizes: v * dV/dx using upwind differences for stability
 *
 * @param grid 1D spatial grid
 * @param velocity Advection velocity (can vary with x)
 * @return Sparse tridiagonal matrix
 */
inline Eigen::SparseMatrix<double> build_advection_matrix(
    const Grid1D& grid,
    const std::function<double(double)>& velocity) {

    size_t n = grid.size();
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(2 * n);

    for (size_t i = 1; i < n - 1; ++i) {
        double v = velocity(grid[i]);
        double dx_m = grid[i] - grid[i - 1];
        double dx_p = grid[i + 1] - grid[i];

        if (v >= 0) {
            // Upwind: use backward difference
            triplets.emplace_back(i, i - 1, -v / dx_m);
            triplets.emplace_back(i, i, v / dx_m);
        } else {
            // Downwind: use forward difference
            triplets.emplace_back(i, i, v / dx_p);
            triplets.emplace_back(i, i + 1, -v / dx_p);
        }
    }

    // Boundary rows
    triplets.emplace_back(0, 0, 1.0);
    triplets.emplace_back(n - 1, n - 1, 1.0);

    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}


/**
 * @brief Solve tridiagonal system Ax = b using Thomas algorithm.
 *
 * O(n) algorithm for tridiagonal systems.
 *
 * @param lower Lower diagonal
 * @param diag Main diagonal
 * @param upper Upper diagonal
 * @param rhs Right-hand side
 * @return Solution vector
 */
inline Eigen::VectorXd solve_tridiagonal(
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& diag,
    const Eigen::VectorXd& upper,
    const Eigen::VectorXd& rhs) {

    size_t n = diag.size();
    Eigen::VectorXd c_prime(n);
    Eigen::VectorXd d_prime(n);
    Eigen::VectorXd x(n);

    // Forward sweep
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for (size_t i = 1; i < n; ++i) {
        double m = diag[i] - lower[i - 1] * c_prime[i - 1];
        c_prime[i] = (i < n - 1) ? upper[i] / m : 0.0;
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / m;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    return x;
}

}  // namespace quant::solvers

#endif  // QUANT_TRADING_PDE_CORE_HPP
