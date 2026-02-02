#ifndef QUANT_TRADING_HJB_SOLVER_HPP
#define QUANT_TRADING_HJB_SOLVER_HPP

/**
 * @file hjb_solver.hpp
 * @brief Hamilton-Jacobi-Bellman solver for optimal stopping problems
 *
 * Solves the HJB equation for optimal mean-reversion trading:
 *   max{ ∂V/∂t + μ(θ-x)∂V/∂x + (1/2)σ²∂²V/∂x² - rV, g(x) - V } = 0
 *
 * where g(x) is the exercise payoff and V is the value function.
 *
 * Reference:
 * - Leung, T. & Li, X. (2015). "Optimal Mean Reversion Trading with
 *   Transaction Costs and Stop-Loss Exit"
 * - Pham, H. (2009). "Continuous-time Stochastic Control and Optimization
 *   with Financial Applications"
 */

#include "pde_core.hpp"
#include "../models/ou_process.hpp"
#include <algorithm>
#include <cmath>
#include <optional>

namespace quant::solvers {

/**
 * @brief Optimal stopping problem type.
 */
enum class StoppingProblem {
    EntryLong,      ///< Optimal entry for long position
    EntryShort,     ///< Optimal entry for short position
    ExitLong,       ///< Optimal exit from long position
    ExitShort       ///< Optimal exit from short position
};

/**
 * @brief HJB solver parameters.
 */
struct HJBParams {
    // OU process parameters
    double theta;       ///< Long-term mean
    double mu;          ///< Mean-reversion speed
    double sigma;       ///< Volatility

    // Trading parameters
    double r;           ///< Discount rate
    double c_entry;     ///< Entry transaction cost (per unit)
    double c_exit;      ///< Exit transaction cost (per unit)
    double T;           ///< Time horizon (use large value for infinite horizon)

    StoppingProblem problem;

    // Grid parameters
    size_t n_space;     ///< Spatial grid points
    size_t n_time;      ///< Time steps
    double x_min;       ///< Min spread value
    double x_max;       ///< Max spread value

    HJBParams()
        : theta(0.0), mu(5.0), sigma(0.1), r(0.05),
          c_entry(0.001), c_exit(0.001), T(1.0),
          problem(StoppingProblem::EntryLong),
          n_space(200), n_time(200), x_min(-0.5), x_max(0.5) {}

    /// Create from OUParameters
    static HJBParams from_ou(const quant::models::OUParameters& ou,
                              double r_, double c_entry_, double c_exit_) {
        HJBParams params;
        params.theta = ou.theta;
        params.mu = ou.mu;
        params.sigma = ou.sigma;
        params.r = r_;
        params.c_entry = c_entry_;
        params.c_exit = c_exit_;
        return params;
    }
};

/**
 * @brief Result of HJB solution.
 */
struct HJBResult {
    Eigen::VectorXd value_function;  ///< Value function V(x)
    Eigen::VectorXd x_grid;          ///< Spatial grid

    std::optional<double> lower_boundary;  ///< Lower optimal boundary
    std::optional<double> upper_boundary;  ///< Upper optimal boundary
    std::optional<double> stop_loss;       ///< Stop-loss level

    /**
     * Get value at specific point.
     */
    double value_at(double x) const {
        // Linear interpolation
        size_t n = x_grid.size();
        if (x <= x_grid[0]) return value_function[0];
        if (x >= x_grid[n - 1]) return value_function[n - 1];

        for (size_t i = 1; i < n; ++i) {
            if (x_grid[i] >= x) {
                double t = (x - x_grid[i - 1]) / (x_grid[i] - x_grid[i - 1]);
                return (1 - t) * value_function[i - 1] + t * value_function[i];
            }
        }
        return value_function[n - 1];
    }

    /**
     * Check if stopping is optimal at x.
     */
    bool should_stop(double x) const {
        if (lower_boundary && x <= *lower_boundary) return true;
        if (upper_boundary && x >= *upper_boundary) return true;
        return false;
    }
};

/**
 * @brief Optimal boundaries for mean-reversion trading.
 */
struct OptimalTradingBoundaries {
    double entry_long;      ///< Enter long when x < entry_long
    double entry_short;     ///< Enter short when x > entry_short
    double exit_long;       ///< Exit long when x > exit_long
    double exit_short;      ///< Exit short when x < exit_short
    double stop_loss_long;  ///< Stop-loss for long position
    double stop_loss_short; ///< Stop-loss for short position
};

/**
 * @brief HJB equation solver for optimal stopping.
 *
 * Uses policy iteration with finite differences to solve the HJB equation.
 * The algorithm alternates between:
 * 1. Value function update (solving linear PDE)
 * 2. Optimal control update (comparing continuation vs stopping)
 */
class HJBSolver {
public:
    explicit HJBSolver(const HJBParams& params)
        : params_(params) {
        validate_params();
    }

    /**
     * Solve the HJB equation for the specified stopping problem.
     */
    HJBResult solve() const {
        Grid1D x_grid(params_.x_min, params_.x_max, params_.n_space, false);
        double dt = params_.T / params_.n_time;

        // Initialize value function with terminal condition
        Eigen::VectorXd V(params_.n_space);
        for (size_t i = 0; i < params_.n_space; ++i) {
            V[i] = terminal_value(x_grid[i]);
        }

        // Build operators
        auto [lower, diag, upper] = build_ou_operator(x_grid, dt);

        // Time-stepping with optimal stopping
        for (size_t step = 0; step < params_.n_time; ++step) {
            // One step of implicit scheme
            Eigen::VectorXd V_new = solve_tridiagonal(lower, diag, upper, V);

            // Apply optimal stopping constraint
            for (size_t i = 0; i < params_.n_space; ++i) {
                double exercise_val = exercise_value(x_grid[i]);
                V_new[i] = std::max(V_new[i], exercise_val);
            }

            // Boundary conditions
            apply_boundary_conditions(V_new, x_grid);

            V = V_new;
        }

        // Find optimal boundaries
        auto [lower_bd, upper_bd] = find_boundaries(V, x_grid);

        HJBResult result;
        result.value_function = V;
        result.x_grid = x_grid.points();
        result.lower_boundary = lower_bd;
        result.upper_boundary = upper_bd;

        return result;
    }

    /**
     * Solve for all optimal trading boundaries.
     *
     * This solves four separate HJB problems:
     * - Entry long, entry short
     * - Exit long, exit short (with stop-loss)
     */
    OptimalTradingBoundaries solve_all_boundaries() const {
        OptimalTradingBoundaries bounds;

        // Entry boundaries
        HJBParams entry_long_params = params_;
        entry_long_params.problem = StoppingProblem::EntryLong;
        HJBSolver entry_long_solver(entry_long_params);
        auto entry_long_result = entry_long_solver.solve();
        bounds.entry_long = entry_long_result.lower_boundary.value_or(params_.theta - 2 * params_.sigma / std::sqrt(2 * params_.mu));

        HJBParams entry_short_params = params_;
        entry_short_params.problem = StoppingProblem::EntryShort;
        HJBSolver entry_short_solver(entry_short_params);
        auto entry_short_result = entry_short_solver.solve();
        bounds.entry_short = entry_short_result.upper_boundary.value_or(params_.theta + 2 * params_.sigma / std::sqrt(2 * params_.mu));

        // Exit boundaries
        HJBParams exit_long_params = params_;
        exit_long_params.problem = StoppingProblem::ExitLong;
        HJBSolver exit_long_solver(exit_long_params);
        auto exit_long_result = exit_long_solver.solve();
        bounds.exit_long = exit_long_result.upper_boundary.value_or(params_.theta);

        HJBParams exit_short_params = params_;
        exit_short_params.problem = StoppingProblem::ExitShort;
        HJBSolver exit_short_solver(exit_short_params);
        auto exit_short_result = exit_short_solver.solve();
        bounds.exit_short = exit_short_result.lower_boundary.value_or(params_.theta);

        // Stop-loss levels (heuristic based on std deviation)
        double sigma_stationary = params_.sigma / std::sqrt(2 * params_.mu);
        bounds.stop_loss_long = bounds.entry_long - 2 * sigma_stationary;
        bounds.stop_loss_short = bounds.entry_short + 2 * sigma_stationary;

        return bounds;
    }

private:
    HJBParams params_;

    void validate_params() const {
        if (params_.mu <= 0) throw std::invalid_argument("mu must be positive");
        if (params_.sigma <= 0) throw std::invalid_argument("sigma must be positive");
        if (params_.r < 0) throw std::invalid_argument("r must be non-negative");
        if (params_.T <= 0) throw std::invalid_argument("T must be positive");
        if (params_.n_space < 10) throw std::invalid_argument("n_space must be >= 10");
    }

    /**
     * Terminal value at time T.
     */
    double terminal_value(double x) const {
        // At terminal time, value is the immediate exercise value
        return exercise_value(x);
    }

    /**
     * Exercise value (stopping payoff) at position x.
     */
    double exercise_value(double x) const {
        switch (params_.problem) {
            case StoppingProblem::EntryLong:
                // Enter long: profit from buying below theta
                // Value = expected profit - entry cost
                return expected_profit_long(x) - params_.c_entry;

            case StoppingProblem::EntryShort:
                // Enter short: profit from selling above theta
                return expected_profit_short(x) - params_.c_entry;

            case StoppingProblem::ExitLong:
                // Exit long: realize current profit
                // Payoff = x - x_entry (entry price set to entry boundary)
                return x - params_.c_exit;

            case StoppingProblem::ExitShort:
                // Exit short: realize current profit
                return -x - params_.c_exit;

            default:
                return 0.0;
        }
    }

    /**
     * Expected profit from entering long at x and exiting at theta.
     *
     * For OU process, expected value = theta + (x - theta) * exp(-mu * t_hit)
     * where t_hit is expected hitting time.
     */
    double expected_profit_long(double x) const {
        if (x >= params_.theta) return 0.0;

        // Approximate using stationary distribution insight
        // Profit = theta - x (ignoring discounting for simplicity)
        double profit = params_.theta - x;

        // Discount by expected time to hit theta
        double expected_time = std::log((params_.theta - x) / params_.sigma) / params_.mu;
        expected_time = std::max(0.0, expected_time);

        return profit * std::exp(-params_.r * expected_time);
    }

    /**
     * Expected profit from entering short at x and exiting at theta.
     */
    double expected_profit_short(double x) const {
        if (x <= params_.theta) return 0.0;

        double profit = x - params_.theta;
        double expected_time = std::log((x - params_.theta) / params_.sigma) / params_.mu;
        expected_time = std::max(0.0, expected_time);

        return profit * std::exp(-params_.r * expected_time);
    }

    /**
     * Build tridiagonal operator for OU process.
     *
     * OU generator: μ(θ-x)∂/∂x + (1/2)σ²∂²/∂x²
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    build_ou_operator(const Grid1D& grid, double dt) const {
        size_t n = grid.size();
        Eigen::VectorXd lower(n - 1), diag(n), upper(n - 1);
        double dx = grid.uniform_step();

        for (size_t i = 1; i < n - 1; ++i) {
            double x = grid[i];

            // Diffusion: (1/2)σ²∂²/∂x² using central differences
            double diff = 0.5 * params_.sigma * params_.sigma;
            double a_coef = diff / (dx * dx);
            double a_m = a_coef;
            double a_p = a_coef;
            double a_c = -2.0 * a_coef;

            // Advection: μ(θ-x)∂/∂x using central differences
            double drift = params_.mu * (params_.theta - x);
            double b_m = -drift / (2.0 * dx);
            double b_c = 0.0;
            double b_p = drift / (2.0 * dx);

            // Combined: L = diffusion + advection - r*I
            double L_m = a_m + b_m;
            double L_c = a_c + b_c - params_.r;
            double L_p = a_p + b_p;

            // Implicit scheme: (I - dt*L) V^n = V^{n+1}
            lower[i - 1] = -dt * L_m;
            diag[i] = 1.0 - dt * L_c;
            upper[i] = -dt * L_p;
        }

        // Boundary conditions
        diag[0] = 1.0;
        diag[n - 1] = 1.0;
        lower[0] = 0.0;
        upper[n - 2] = 0.0;

        return {lower, diag, upper};
    }

    void apply_boundary_conditions(Eigen::VectorXd& V, const Grid1D& grid) const {
        // Linear extrapolation at boundaries
        V[0] = 2 * V[1] - V[2];
        size_t n = V.size();
        V[n - 1] = 2 * V[n - 2] - V[n - 3];
    }

    /**
     * Find optimal stopping boundaries from value function.
     *
     * The stopping boundary is where V(x) = g(x) (value equals exercise).
     */
    std::pair<std::optional<double>, std::optional<double>>
    find_boundaries(const Eigen::VectorXd& V, const Grid1D& grid) const {
        std::optional<double> lower_bd, upper_bd;
        size_t n = grid.size();

        // Find where value function equals exercise value
        for (size_t i = 1; i < n; ++i) {
            double x = grid[i];
            double ex_val = exercise_value(x);

            // Check if we cross from continuation to stopping region
            double prev_diff = V[i - 1] - exercise_value(grid[i - 1]);
            double curr_diff = V[i] - ex_val;

            if (prev_diff > 1e-10 && curr_diff <= 1e-10) {
                // Crossed into stopping region (from left)
                double t = prev_diff / (prev_diff - curr_diff);
                lower_bd = grid[i - 1] + t * (grid[i] - grid[i - 1]);
            }

            if (prev_diff <= 1e-10 && curr_diff > 1e-10) {
                // Left stopping region
                double t = -prev_diff / (curr_diff - prev_diff);
                upper_bd = grid[i - 1] + t * (grid[i] - grid[i - 1]);
            }
        }

        return {lower_bd, upper_bd};
    }
};

}  // namespace quant::solvers

#endif  // QUANT_TRADING_HJB_SOLVER_HPP
