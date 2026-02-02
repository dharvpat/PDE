#ifndef QUANT_TRADING_BLACK_SCHOLES_PDE_HPP
#define QUANT_TRADING_BLACK_SCHOLES_PDE_HPP

/**
 * @file black_scholes_pde.hpp
 * @brief Black-Scholes PDE solver using finite differences
 *
 * Solves the Black-Scholes PDE:
 *   ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
 *
 * Transformed to log-coordinates x = log(S):
 *   ∂V/∂t + (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV = 0
 *
 * Reference: Wilmott, P. (2006). "Paul Wilmott on Quantitative Finance"
 */

#include "pde_core.hpp"
#include <algorithm>
#include <cmath>

namespace quant::solvers {

/**
 * @brief Option type enumeration.
 */
enum class OptionType {
    Call,
    Put
};

/**
 * @brief Exercise style enumeration.
 */
enum class ExerciseStyle {
    European,
    American
};

/**
 * @brief Black-Scholes PDE solver parameters.
 */
struct BlackScholesPDEParams {
    double sigma;           ///< Volatility
    double r;               ///< Risk-free rate
    double q;               ///< Dividend yield
    double T;               ///< Time to maturity
    double K;               ///< Strike price
    OptionType option_type; ///< Call or Put
    ExerciseStyle exercise; ///< European or American

    // Grid parameters
    size_t n_space;         ///< Number of spatial grid points
    size_t n_time;          ///< Number of time steps
    double s_min_mult;      ///< S_min = K * s_min_mult (default 0.2)
    double s_max_mult;      ///< S_max = K * s_max_mult (default 5.0)
    TimeScheme scheme;      ///< Time-stepping scheme

    BlackScholesPDEParams()
        : sigma(0.2), r(0.05), q(0.0), T(1.0), K(100.0),
          option_type(OptionType::Call), exercise(ExerciseStyle::European),
          n_space(200), n_time(100), s_min_mult(0.2), s_max_mult(5.0),
          scheme(TimeScheme::CrankNicolson) {}
};

/**
 * @brief Result of Black-Scholes PDE solution.
 */
struct BlackScholesPDEResult {
    double price;                   ///< Option price at S0
    double delta;                   ///< Delta (dV/dS)
    double gamma;                   ///< Gamma (d²V/dS²)
    double theta;                   ///< Theta (dV/dt)
    Eigen::VectorXd prices;         ///< Full price grid at t=0
    Eigen::VectorXd spot_grid;      ///< Spot price grid
    bool early_exercise_optimal;    ///< For American options
};

/**
 * @brief Black-Scholes PDE solver.
 *
 * Uses finite differences with Crank-Nicolson time-stepping for stability
 * and second-order accuracy. Supports both European and American options.
 */
class BlackScholesPDESolver {
public:
    explicit BlackScholesPDESolver(const BlackScholesPDEParams& params)
        : params_(params) {
        validate_params();
    }

    /**
     * Solve the Black-Scholes PDE.
     *
     * @param S0 Current spot price
     * @return Solution result with price and Greeks
     */
    BlackScholesPDEResult solve(double S0) const {
        // Set up grids
        double S_min = params_.K * params_.s_min_mult;
        double S_max = params_.K * params_.s_max_mult;

        // Use log-space grid for better accuracy near strike
        Grid1D S_grid(S_min, S_max, params_.n_space, true);
        double dt = params_.T / params_.n_time;

        // Initialize solution with terminal payoff
        Eigen::VectorXd V(params_.n_space);
        for (size_t i = 0; i < params_.n_space; ++i) {
            V[i] = payoff(S_grid[i]);
        }

        // Build operators
        auto [lower, diag, upper] = build_operators(S_grid, dt);

        // Time-stepping (backward from T to 0)
        for (size_t step = 0; step < params_.n_time; ++step) {
            V = time_step(V, lower, diag, upper, S_grid, dt);

            // Apply early exercise for American options
            if (params_.exercise == ExerciseStyle::American) {
                for (size_t i = 0; i < params_.n_space; ++i) {
                    V[i] = std::max(V[i], payoff(S_grid[i]));
                }
            }

            // Apply boundary conditions
            apply_boundary_conditions(V, S_grid, (params_.n_time - step - 1) * dt);
        }

        // Interpolate to get price at S0
        size_t idx = S_grid.find_index(S0);
        double price = S_grid.interpolate(S0, V);

        // Compute Greeks using finite differences
        double delta = compute_delta(V, S_grid, S0);
        double gamma = compute_gamma(V, S_grid, S0);
        double theta = compute_theta(S0, price);

        // Check for early exercise
        bool early_ex = (params_.exercise == ExerciseStyle::American) &&
                        (price > payoff(S0) + 1e-10);

        return BlackScholesPDEResult{
            price, delta, gamma, theta,
            V, S_grid.points(), early_ex
        };
    }

private:
    BlackScholesPDEParams params_;

    void validate_params() const {
        if (params_.sigma <= 0) throw std::invalid_argument("sigma must be positive");
        if (params_.T <= 0) throw std::invalid_argument("T must be positive");
        if (params_.K <= 0) throw std::invalid_argument("K must be positive");
        if (params_.n_space < 10) throw std::invalid_argument("n_space must be >= 10");
        if (params_.n_time < 10) throw std::invalid_argument("n_time must be >= 10");
    }

    double payoff(double S) const {
        if (params_.option_type == OptionType::Call) {
            return std::max(S - params_.K, 0.0);
        } else {
            return std::max(params_.K - S, 0.0);
        }
    }

    /**
     * Build tridiagonal operators for the PDE.
     *
     * Returns lower, diagonal, and upper diagonals for the implicit solve.
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    build_operators(const Grid1D& grid, double dt) const {
        size_t n = grid.size();
        Eigen::VectorXd lower(n - 1), diag(n), upper(n - 1);

        double sigma2 = params_.sigma * params_.sigma;
        double drift = params_.r - params_.q - 0.5 * sigma2;

        // For log-space grid, use uniform dx in log coordinates
        double dx = grid.uniform_step();  // Uniform spacing in log(S) space

        // Interior points
        for (size_t i = 1; i < n - 1; ++i) {
            // In log-space, we have uniform dx
            // Coefficients for log-transformed PDE:
            // ∂V/∂t + (1/2)σ²∂²V/∂x² + (r - σ²/2)∂V/∂x - rV = 0
            double diff_coef = 0.5 * sigma2;
            double adv_coef = drift;

            // Central difference for diffusion: ∂²V/∂x² ≈ (V_{i+1} - 2V_i + V_{i-1})/dx²
            double a_coef = diff_coef / (dx * dx);
            double a_m = a_coef;
            double a_p = a_coef;
            double a_c = -2.0 * a_coef;

            // Central difference for advection: ∂V/∂x ≈ (V_{i+1} - V_{i-1})/(2dx)
            double b_m = -adv_coef / (2.0 * dx);
            double b_c = 0.0;
            double b_p = adv_coef / (2.0 * dx);

            // Combined operator: L = diffusion + advection - r*I
            double L_m = a_m + b_m;
            double L_c = a_c + b_c - params_.r;
            double L_p = a_p + b_p;

            // Crank-Nicolson: (I - 0.5*dt*L) V^{n} = (I + 0.5*dt*L) V^{n+1}
            if (params_.scheme == TimeScheme::CrankNicolson) {
                if (i > 1) lower[i - 1] = -0.5 * dt * L_m;
                diag[i] = 1.0 - 0.5 * dt * L_c;
                if (i < n - 2) upper[i] = -0.5 * dt * L_p;
            } else if (params_.scheme == TimeScheme::Implicit) {
                if (i > 1) lower[i - 1] = -dt * L_m;
                diag[i] = 1.0 - dt * L_c;
                if (i < n - 2) upper[i] = -dt * L_p;
            } else {
                // Explicit: coefficients used differently
                if (i > 1) lower[i - 1] = dt * L_m;
                diag[i] = 1.0 + dt * L_c;
                if (i < n - 2) upper[i] = dt * L_p;
            }
        }

        // Boundary conditions (Dirichlet)
        diag[0] = 1.0;
        diag[n - 1] = 1.0;
        lower[0] = 0.0;
        upper[n - 2] = 0.0;

        return {lower, diag, upper};
    }

    Eigen::VectorXd time_step(
        const Eigen::VectorXd& V,
        const Eigen::VectorXd& lower,
        const Eigen::VectorXd& diag,
        const Eigen::VectorXd& upper,
        const Grid1D& grid,
        double dt) const {

        size_t n = V.size();
        Eigen::VectorXd rhs = V;

        if (params_.scheme == TimeScheme::CrankNicolson) {
            // Build RHS: (I + 0.5*dt*L) V^{n+1}
            // This requires the explicit operator application
            double sigma2 = params_.sigma * params_.sigma;
            double drift = params_.r - params_.q - 0.5 * sigma2;
            double dx = grid.uniform_step();

            for (size_t i = 1; i < n - 1; ++i) {
                double diff_coef = 0.5 * sigma2;
                double a_coef = diff_coef / (dx * dx);
                double a_m = a_coef;
                double a_p = a_coef;
                double a_c = -2.0 * a_coef;

                double adv_coef = drift;
                double b_m = -adv_coef / (2.0 * dx);
                double b_c = 0.0;
                double b_p = adv_coef / (2.0 * dx);

                double L_m = a_m + b_m;
                double L_c = a_c + b_c - params_.r;
                double L_p = a_p + b_p;

                rhs[i] = V[i] + 0.5 * dt * (L_m * V[i - 1] + L_c * V[i] + L_p * V[i + 1]);
            }
        }

        // Solve tridiagonal system
        return solve_tridiagonal(lower, diag, upper, rhs);
    }

    void apply_boundary_conditions(Eigen::VectorXd& V, const Grid1D& grid, double t) const {
        double df = std::exp(-params_.r * t);

        if (params_.option_type == OptionType::Call) {
            // At S_min: call worth ~0
            V[0] = 0.0;
            // At S_max: call worth ~S - K*exp(-r*t)
            V[V.size() - 1] = grid[V.size() - 1] - params_.K * df;
        } else {
            // At S_min: put worth ~K*exp(-r*t) - S
            V[0] = params_.K * df - grid[0];
            // At S_max: put worth ~0
            V[V.size() - 1] = 0.0;
        }
    }

    double compute_delta(const Eigen::VectorXd& V, const Grid1D& grid, double S0) const {
        size_t i = grid.find_index(S0);
        if (i == 0) i = 1;
        if (i >= grid.size() - 1) i = grid.size() - 2;

        double dV = V[i + 1] - V[i - 1];
        double dS = grid[i + 1] - grid[i - 1];
        return dV / dS;
    }

    double compute_gamma(const Eigen::VectorXd& V, const Grid1D& grid, double S0) const {
        size_t i = grid.find_index(S0);
        if (i == 0) i = 1;
        if (i >= grid.size() - 1) i = grid.size() - 2;

        double dx_m = grid[i] - grid[i - 1];
        double dx_p = grid[i + 1] - grid[i];
        double dx_avg = 0.5 * (dx_m + dx_p);

        return (V[i + 1] - 2 * V[i] + V[i - 1]) / (dx_avg * dx_avg);
    }

    double compute_theta(double S0, double price) const {
        // Use Black-Scholes formula to compute theta analytically
        // This is more accurate than finite differences in time
        double d1 = (std::log(S0 / params_.K) +
                     (params_.r - params_.q + 0.5 * params_.sigma * params_.sigma) * params_.T) /
                    (params_.sigma * std::sqrt(params_.T));

        double nd1_density = std::exp(-0.5 * d1 * d1) / std::sqrt(2 * M_PI);

        double theta = -S0 * nd1_density * params_.sigma / (2 * std::sqrt(params_.T));
        if (params_.option_type == OptionType::Call) {
            theta -= params_.r * params_.K * std::exp(-params_.r * params_.T) * 0.5;
        } else {
            theta += params_.r * params_.K * std::exp(-params_.r * params_.T) * 0.5;
        }

        return theta;
    }
};

}  // namespace quant::solvers

#endif  // QUANT_TRADING_BLACK_SCHOLES_PDE_HPP
