#ifndef QUANT_TRADING_HESTON_PDE_HPP
#define QUANT_TRADING_HESTON_PDE_HPP

/**
 * @file heston_pde.hpp
 * @brief Heston stochastic volatility PDE solver using ADI method
 *
 * Solves the 2D Heston PDE:
 *   ∂V/∂t + (1/2)vS²∂²V/∂S² + ρσvS∂²V/∂S∂v + (1/2)σ²v∂²V/∂v²
 *         + rS∂V/∂S + κ(θ-v)∂V/∂v - rV = 0
 *
 * Uses the Alternating Direction Implicit (ADI) method for efficient
 * and stable time-stepping of the 2D problem.
 *
 * Reference:
 * - In 't Hout, K.J. & Foulon, S. (2010). "ADI finite difference schemes for
 *   option pricing in the Heston model with correlation"
 * - Craig, I.J.D. & Sneyd, A.D. (1988). "An alternating-direction implicit
 *   scheme for parabolic equations with mixed derivatives"
 */

#include "pde_core.hpp"
#include "../models/heston.hpp"
#include <algorithm>
#include <cmath>

namespace quant::solvers {

/**
 * @brief Heston PDE solver parameters.
 */
struct HestonPDEParams {
    // Model parameters
    double kappa;   ///< Mean-reversion speed
    double theta;   ///< Long-term variance
    double sigma;   ///< Volatility of variance
    double rho;     ///< Correlation
    double v0;      ///< Initial variance
    double r;       ///< Risk-free rate
    double q;       ///< Dividend yield

    // Option parameters
    double T;       ///< Time to maturity
    double K;       ///< Strike price
    OptionType option_type;
    ExerciseStyle exercise;

    // Grid parameters
    size_t n_spot;      ///< Spot grid points
    size_t n_vol;       ///< Variance grid points
    size_t n_time;      ///< Time steps
    double s_min_mult;  ///< S_min = K * s_min_mult
    double s_max_mult;  ///< S_max = K * s_max_mult
    double v_max;       ///< Maximum variance

    HestonPDEParams()
        : kappa(2.0), theta(0.04), sigma(0.3), rho(-0.7), v0(0.04),
          r(0.05), q(0.0), T(1.0), K(100.0),
          option_type(OptionType::Call), exercise(ExerciseStyle::European),
          n_spot(100), n_vol(50), n_time(100),
          s_min_mult(0.2), s_max_mult(5.0), v_max(1.0) {}

    /// Create from HestonParameters struct
    static HestonPDEParams from_model(const quant::models::HestonParameters& p,
                                       double r_, double q_, double T_, double K_) {
        HestonPDEParams params;
        params.kappa = p.kappa;
        params.theta = p.theta;
        params.sigma = p.sigma;
        params.rho = p.rho;
        params.v0 = p.v0;
        params.r = r_;
        params.q = q_;
        params.T = T_;
        params.K = K_;
        return params;
    }
};

/**
 * @brief Result of Heston PDE solution.
 */
struct HestonPDEResult {
    double price;       ///< Option price at (S0, v0)
    double delta;       ///< dV/dS
    double gamma;       ///< d²V/dS²
    double vega;        ///< dV/dv (sensitivity to variance)
    double theta;       ///< dV/dt

    Eigen::MatrixXd prices;     ///< Full price surface V(S,v)
    Eigen::VectorXd spot_grid;  ///< Spot grid
    Eigen::VectorXd vol_grid;   ///< Variance grid
};

/**
 * @brief Heston PDE solver using ADI method.
 *
 * The ADI method splits the 2D operator into two 1D sweeps per time step,
 * maintaining unconditional stability while being computationally efficient.
 *
 * We use the Craig-Sneyd ADI scheme for handling the mixed derivative term.
 */
class HestonPDESolver {
public:
    explicit HestonPDESolver(const HestonPDEParams& params)
        : params_(params) {
        validate_params();
    }

    /**
     * Solve the Heston PDE.
     *
     * @param S0 Current spot price
     * @return Solution result with price and Greeks
     */
    HestonPDEResult solve(double S0) const {
        // Set up grids
        double S_min = params_.K * params_.s_min_mult;
        double S_max = params_.K * params_.s_max_mult;
        double v_min = 1e-6;  // Small positive to avoid singularity
        double v_max = params_.v_max;

        Grid1D S_grid(S_min, S_max, params_.n_spot, true);  // Log-space for spot
        Grid1D v_grid(v_min, v_max, params_.n_vol, false);  // Uniform for variance
        Grid2D grid(S_grid, v_grid);

        double dt = params_.T / params_.n_time;

        // Initialize with terminal payoff
        Eigen::MatrixXd V(params_.n_spot, params_.n_vol);
        for (size_t i = 0; i < params_.n_spot; ++i) {
            double payoff_val = payoff(S_grid[i]);
            for (size_t j = 0; j < params_.n_vol; ++j) {
                V(i, j) = payoff_val;
            }
        }

        // Time-stepping (backward from T to 0)
        for (size_t step = 0; step < params_.n_time; ++step) {
            V = adi_step(V, S_grid, v_grid, dt);

            // Apply early exercise for American options
            if (params_.exercise == ExerciseStyle::American) {
                for (size_t i = 0; i < params_.n_spot; ++i) {
                    double payoff_val = payoff(S_grid[i]);
                    for (size_t j = 0; j < params_.n_vol; ++j) {
                        V(i, j) = std::max(V(i, j), payoff_val);
                    }
                }
            }

            // Apply boundary conditions
            apply_boundary_conditions(V, S_grid, v_grid,
                                      (params_.n_time - step - 1) * dt);
        }

        // Interpolate to get price at (S0, v0)
        double price = interpolate_2d(V, S_grid, v_grid, S0, params_.v0);

        // Compute Greeks
        double delta = compute_delta(V, S_grid, v_grid, S0);
        double gamma = compute_gamma(V, S_grid, v_grid, S0);
        double vega = compute_vega(V, S_grid, v_grid, S0);
        double theta = compute_theta(V, S_grid, v_grid, S0, price, dt);

        return HestonPDEResult{
            price, delta, gamma, vega, theta,
            V, S_grid.points(), v_grid.points()
        };
    }

private:
    HestonPDEParams params_;

    void validate_params() const {
        if (params_.kappa <= 0) throw std::invalid_argument("kappa must be positive");
        if (params_.theta <= 0) throw std::invalid_argument("theta must be positive");
        if (params_.sigma <= 0) throw std::invalid_argument("sigma must be positive");
        if (std::abs(params_.rho) >= 1) throw std::invalid_argument("|rho| must be < 1");
        if (params_.v0 <= 0) throw std::invalid_argument("v0 must be positive");
        if (params_.T <= 0) throw std::invalid_argument("T must be positive");
        if (params_.K <= 0) throw std::invalid_argument("K must be positive");
    }

    double payoff(double S) const {
        if (params_.option_type == OptionType::Call) {
            return std::max(S - params_.K, 0.0);
        } else {
            return std::max(params_.K - S, 0.0);
        }
    }

    /**
     * Perform one ADI time step using Craig-Sneyd scheme.
     *
     * The scheme is:
     *   Y^0 = V^n + dt * F^0(V^n)
     *   Y^1 = Y^0 + θ*dt*(A_1(Y^1) - A_1(V^n))
     *   Y^2 = Y^1 + θ*dt*(A_2(Y^2) - A_2(V^n))
     *   V^{n+1} = Y^2 + (0.5-θ)*dt*(F^0(Y^2) - F^0(V^n))
     *
     * where θ = 0.5 for Crank-Nicolson-like stability.
     */
    Eigen::MatrixXd adi_step(const Eigen::MatrixXd& V,
                              const Grid1D& S_grid,
                              const Grid1D& v_grid,
                              double dt) const {
        size_t nS = S_grid.size();
        size_t nv = v_grid.size();
        double theta_adi = 0.5;  // ADI parameter

        // Step 1: Explicit evaluation of the full operator
        Eigen::MatrixXd F0 = compute_full_operator(V, S_grid, v_grid);
        Eigen::MatrixXd Y0 = V + dt * F0;

        // Step 2: Implicit in S-direction
        Eigen::MatrixXd Y1 = Y0;
        for (size_t j = 1; j < nv - 1; ++j) {
            double v = v_grid[j];
            auto [lower, diag, upper] = build_S_operator(S_grid, v, dt * theta_adi);
            Eigen::VectorXd rhs = Y0.col(j);

            // Subtract explicit part that was already included
            Eigen::VectorXd A1_Vn = apply_S_operator(V.col(j), S_grid, v);
            rhs -= theta_adi * dt * A1_Vn;

            Y1.col(j) = solve_tridiagonal(lower, diag, upper, rhs);
        }

        // Step 3: Implicit in v-direction
        Eigen::MatrixXd Y2 = Y1;
        for (size_t i = 1; i < nS - 1; ++i) {
            double S = S_grid[i];
            auto [lower, diag, upper] = build_v_operator(v_grid, S, dt * theta_adi);
            Eigen::VectorXd rhs = Y1.row(i).transpose();

            // Subtract explicit part
            Eigen::VectorXd A2_Vn = apply_v_operator(V.row(i).transpose(), v_grid, S);
            rhs -= theta_adi * dt * A2_Vn;

            Y2.row(i) = solve_tridiagonal(lower, diag, upper, rhs).transpose();
        }

        // Step 4: Correction step
        Eigen::MatrixXd F0_Y2 = compute_full_operator(Y2, S_grid, v_grid);
        Eigen::MatrixXd Vnew = Y2 + (0.5 - theta_adi) * dt * (F0_Y2 - F0);

        return Vnew;
    }

    /**
     * Compute the full PDE operator F(V) = L_S V + L_v V + L_Sv V - rV
     */
    Eigen::MatrixXd compute_full_operator(const Eigen::MatrixXd& V,
                                           const Grid1D& S_grid,
                                           const Grid1D& v_grid) const {
        size_t nS = S_grid.size();
        size_t nv = v_grid.size();
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(nS, nv);

        for (size_t i = 1; i < nS - 1; ++i) {
            for (size_t j = 1; j < nv - 1; ++j) {
                double S = S_grid[i];
                double v = v_grid[j];

                double dS_m = S_grid[i] - S_grid[i - 1];
                double dS_p = S_grid[i + 1] - S_grid[i];
                double dv_m = v_grid[j] - v_grid[j - 1];
                double dv_p = v_grid[j + 1] - v_grid[j];

                // Second derivatives
                double V_SS = (V(i + 1, j) - V(i, j)) / dS_p - (V(i, j) - V(i - 1, j)) / dS_m;
                V_SS /= 0.5 * (dS_m + dS_p);

                double V_vv = (V(i, j + 1) - V(i, j)) / dv_p - (V(i, j) - V(i, j - 1)) / dv_m;
                V_vv /= 0.5 * (dv_m + dv_p);

                // First derivatives (central differences)
                double V_S = (V(i + 1, j) - V(i - 1, j)) / (dS_m + dS_p);
                double V_v = (V(i, j + 1) - V(i, j - 1)) / (dv_m + dv_p);

                // Mixed derivative
                double V_Sv = (V(i + 1, j + 1) - V(i + 1, j - 1) - V(i - 1, j + 1) + V(i - 1, j - 1));
                V_Sv /= (dS_m + dS_p) * (dv_m + dv_p);

                // PDE operator
                result(i, j) = 0.5 * v * S * S * V_SS
                             + params_.rho * params_.sigma * v * S * V_Sv
                             + 0.5 * params_.sigma * params_.sigma * v * V_vv
                             + (params_.r - params_.q) * S * V_S
                             + params_.kappa * (params_.theta - v) * V_v
                             - params_.r * V(i, j);
            }
        }

        return result;
    }

    /**
     * Build tridiagonal operator for S-direction.
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    build_S_operator(const Grid1D& S_grid, double v, double dt_theta) const {
        size_t n = S_grid.size();
        Eigen::VectorXd lower(n - 1), diag(n), upper(n - 1);

        lower.setZero();
        diag.setOnes();
        upper.setZero();

        for (size_t i = 1; i < n - 1; ++i) {
            double S = S_grid[i];
            double dS_m = S_grid[i] - S_grid[i - 1];
            double dS_p = S_grid[i + 1] - S_grid[i];
            double dS_avg = 0.5 * (dS_m + dS_p);

            // Diffusion: 0.5 * v * S^2 * d²/dS²
            double diff = 0.5 * v * S * S;
            double a_m = diff / (dS_m * dS_avg);
            double a_p = diff / (dS_p * dS_avg);
            double a_c = -(a_m + a_p);

            // Advection: (r-q) * S * d/dS
            double adv = (params_.r - params_.q) * S;
            double b_m = -adv / (dS_m + dS_p);
            double b_p = adv / (dS_m + dS_p);

            // Combined with time stepping
            lower[i - 1] = -dt_theta * (a_m + b_m);
            diag[i] = 1.0 - dt_theta * a_c;
            upper[i] = -dt_theta * (a_p + b_p);
        }

        // Boundaries
        diag[0] = 1.0;
        diag[n - 1] = 1.0;

        return {lower, diag, upper};
    }

    /**
     * Apply S-direction operator (without time stepping).
     */
    Eigen::VectorXd apply_S_operator(const Eigen::VectorXd& V,
                                      const Grid1D& S_grid, double v) const {
        size_t n = S_grid.size();
        Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

        for (size_t i = 1; i < n - 1; ++i) {
            double S = S_grid[i];
            double dS_m = S_grid[i] - S_grid[i - 1];
            double dS_p = S_grid[i + 1] - S_grid[i];
            double dS_avg = 0.5 * (dS_m + dS_p);

            double diff = 0.5 * v * S * S;
            double V_SS = (V[i + 1] - V[i]) / dS_p - (V[i] - V[i - 1]) / dS_m;
            V_SS /= dS_avg;

            double adv = (params_.r - params_.q) * S;
            double V_S = (V[i + 1] - V[i - 1]) / (dS_m + dS_p);

            result[i] = diff * V_SS + adv * V_S;
        }

        return result;
    }

    /**
     * Build tridiagonal operator for v-direction.
     */
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd>
    build_v_operator(const Grid1D& v_grid, double S, double dt_theta) const {
        size_t n = v_grid.size();
        Eigen::VectorXd lower(n - 1), diag(n), upper(n - 1);

        lower.setZero();
        diag.setOnes();
        upper.setZero();

        for (size_t j = 1; j < n - 1; ++j) {
            double v = v_grid[j];
            double dv_m = v_grid[j] - v_grid[j - 1];
            double dv_p = v_grid[j + 1] - v_grid[j];
            double dv_avg = 0.5 * (dv_m + dv_p);

            // Diffusion: 0.5 * sigma^2 * v * d²/dv²
            double diff = 0.5 * params_.sigma * params_.sigma * v;
            double a_m = diff / (dv_m * dv_avg);
            double a_p = diff / (dv_p * dv_avg);
            double a_c = -(a_m + a_p);

            // Advection: kappa * (theta - v) * d/dv
            double adv = params_.kappa * (params_.theta - v);
            double b_m, b_p;
            if (adv >= 0) {
                b_m = -adv / dv_m;
                b_p = 0;
            } else {
                b_m = 0;
                b_p = -adv / dv_p;
            }
            double b_c = -b_m - b_p;

            // Combined with reaction and time stepping
            lower[j - 1] = -dt_theta * (a_m + b_m);
            diag[j] = 1.0 - dt_theta * (a_c + b_c - params_.r);
            upper[j] = -dt_theta * (a_p + b_p);
        }

        // Boundaries
        diag[0] = 1.0;
        diag[n - 1] = 1.0;

        return {lower, diag, upper};
    }

    /**
     * Apply v-direction operator (without time stepping).
     */
    Eigen::VectorXd apply_v_operator(const Eigen::VectorXd& V,
                                      const Grid1D& v_grid, double S) const {
        size_t n = v_grid.size();
        Eigen::VectorXd result = Eigen::VectorXd::Zero(n);

        for (size_t j = 1; j < n - 1; ++j) {
            double v = v_grid[j];
            double dv_m = v_grid[j] - v_grid[j - 1];
            double dv_p = v_grid[j + 1] - v_grid[j];
            double dv_avg = 0.5 * (dv_m + dv_p);

            double diff = 0.5 * params_.sigma * params_.sigma * v;
            double V_vv = (V[j + 1] - V[j]) / dv_p - (V[j] - V[j - 1]) / dv_m;
            V_vv /= dv_avg;

            double adv = params_.kappa * (params_.theta - v);
            double V_v = (V[j + 1] - V[j - 1]) / (dv_m + dv_p);

            result[j] = diff * V_vv + adv * V_v - params_.r * V[j];
        }

        return result;
    }

    void apply_boundary_conditions(Eigen::MatrixXd& V,
                                    const Grid1D& S_grid,
                                    const Grid1D& v_grid,
                                    double t) const {
        size_t nS = S_grid.size();
        size_t nv = v_grid.size();
        double df = std::exp(-params_.r * t);

        // S = S_min boundary
        for (size_t j = 0; j < nv; ++j) {
            if (params_.option_type == OptionType::Call) {
                V(0, j) = 0.0;
            } else {
                V(0, j) = params_.K * df - S_grid[0];
            }
        }

        // S = S_max boundary
        for (size_t j = 0; j < nv; ++j) {
            if (params_.option_type == OptionType::Call) {
                V(nS - 1, j) = S_grid[nS - 1] - params_.K * df;
            } else {
                V(nS - 1, j) = 0.0;
            }
        }

        // v = v_min boundary (use linearity)
        for (size_t i = 0; i < nS; ++i) {
            V(i, 0) = 2 * V(i, 1) - V(i, 2);
        }

        // v = v_max boundary (use linearity)
        for (size_t i = 0; i < nS; ++i) {
            V(i, nv - 1) = 2 * V(i, nv - 2) - V(i, nv - 3);
        }
    }

    double interpolate_2d(const Eigen::MatrixXd& V,
                          const Grid1D& S_grid,
                          const Grid1D& v_grid,
                          double S, double v) const {
        // Bilinear interpolation
        size_t i = S_grid.find_index(S);
        size_t j = v_grid.find_index(v);

        if (i == 0) i = 1;
        if (i >= S_grid.size() - 1) i = S_grid.size() - 2;
        if (j == 0) j = 1;
        if (j >= v_grid.size() - 1) j = v_grid.size() - 2;

        double t_S = (S - S_grid[i - 1]) / (S_grid[i] - S_grid[i - 1]);
        double t_v = (v - v_grid[j - 1]) / (v_grid[j] - v_grid[j - 1]);

        t_S = std::max(0.0, std::min(1.0, t_S));
        t_v = std::max(0.0, std::min(1.0, t_v));

        return (1 - t_S) * (1 - t_v) * V(i - 1, j - 1)
             + t_S * (1 - t_v) * V(i, j - 1)
             + (1 - t_S) * t_v * V(i - 1, j)
             + t_S * t_v * V(i, j);
    }

    double compute_delta(const Eigen::MatrixXd& V,
                         const Grid1D& S_grid,
                         const Grid1D& v_grid,
                         double S0) const {
        size_t j = v_grid.find_index(params_.v0);
        size_t i = S_grid.find_index(S0);
        if (i == 0) i = 1;
        if (i >= S_grid.size() - 1) i = S_grid.size() - 2;

        return (V(i + 1, j) - V(i - 1, j)) / (S_grid[i + 1] - S_grid[i - 1]);
    }

    double compute_gamma(const Eigen::MatrixXd& V,
                         const Grid1D& S_grid,
                         const Grid1D& v_grid,
                         double S0) const {
        size_t j = v_grid.find_index(params_.v0);
        size_t i = S_grid.find_index(S0);
        if (i == 0) i = 1;
        if (i >= S_grid.size() - 1) i = S_grid.size() - 2;

        double dS_m = S_grid[i] - S_grid[i - 1];
        double dS_p = S_grid[i + 1] - S_grid[i];
        double dS_avg = 0.5 * (dS_m + dS_p);

        return (V(i + 1, j) - 2 * V(i, j) + V(i - 1, j)) / (dS_avg * dS_avg);
    }

    double compute_vega(const Eigen::MatrixXd& V,
                        const Grid1D& S_grid,
                        const Grid1D& v_grid,
                        double S0) const {
        size_t i = S_grid.find_index(S0);
        size_t j = v_grid.find_index(params_.v0);
        if (j == 0) j = 1;
        if (j >= v_grid.size() - 1) j = v_grid.size() - 2;

        // dV/dv, then convert to vega (dV/d(sigma)) using chain rule
        double dV_dv = (V(i, j + 1) - V(i, j - 1)) / (v_grid[j + 1] - v_grid[j - 1]);
        // Approximate: vega ~ 2*sqrt(v0)*T * dV/dv
        return 2 * std::sqrt(params_.v0) * params_.T * dV_dv;
    }

    double compute_theta(const Eigen::MatrixXd& V,
                         const Grid1D& S_grid,
                         const Grid1D& v_grid,
                         double S0, double price, double dt) const {
        // Theta computed from PDE: theta = -F(V) at current price
        size_t i = S_grid.find_index(S0);
        size_t j = v_grid.find_index(params_.v0);

        Eigen::MatrixXd F = compute_full_operator(V, S_grid, v_grid);
        return -F(i, j);
    }
};

}  // namespace quant::solvers

#endif  // QUANT_TRADING_HESTON_PDE_HPP
