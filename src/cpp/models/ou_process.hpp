#ifndef QUANT_TRADING_OU_PROCESS_HPP
#define QUANT_TRADING_OU_PROCESS_HPP

/**
 * @file ou_process.hpp
 * @brief Ornstein-Uhlenbeck mean-reverting process implementation
 *
 * Reference: Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with
 * Transaction Costs and Stop-Loss Exit." Journal of Industrial and Management
 * Optimization.
 *
 * The OU process is defined by the SDE:
 *   dX_t = μ(θ - X_t)dt + σ dB_t
 *
 * Key properties:
 * - Mean-reverting to long-term level θ
 * - Mean-reversion speed controlled by μ
 * - Half-life of reversion: t_half = ln(2)/μ
 */

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef QUANT_USE_EIGEN
#include <Eigen/Dense>
#endif

namespace quant::models {

/**
 * @brief Ornstein-Uhlenbeck process parameters
 *
 * Parameters:
 * - theta: Long-term mean (equilibrium level)
 * - mu: Mean-reversion speed (> 0 for mean-reverting)
 * - sigma: Instantaneous volatility
 */
struct OUParameters {
    double theta;  ///< Long-term mean (equilibrium level)
    double mu;     ///< Mean-reversion speed
    double sigma;  ///< Volatility

    /// Default constructor
    OUParameters() : theta(0.0), mu(1.0), sigma(0.1) {}

    /// Parameterized constructor
    OUParameters(double theta_, double mu_, double sigma_)
        : theta(theta_), mu(mu_), sigma(sigma_) {}

    /**
     * @brief Half-life of mean reversion (in same units as time)
     *
     * The half-life is the time it takes for the expected value to move
     * halfway from its current position to the long-term mean.
     *
     * t_half = ln(2) / μ
     */
    double half_life() const {
        if (mu <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return std::log(2.0) / mu;
    }

    /**
     * @brief Check if process is mean-reverting
     *
     * The process is mean-reverting if μ > 0.
     * If μ ≤ 0, the process is either non-stationary (μ = 0, random walk)
     * or explosive (μ < 0).
     */
    bool is_mean_reverting() const noexcept { return mu > 0.0; }

    /**
     * @brief Stationary (long-run) variance of the process
     *
     * For a stationary OU process, the variance converges to:
     *   Var_∞ = σ² / (2μ)
     */
    double stationary_variance() const {
        if (mu <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return (sigma * sigma) / (2.0 * mu);
    }

    /**
     * @brief Stationary (long-run) standard deviation
     */
    double stationary_std() const { return std::sqrt(stationary_variance()); }

    /**
     * @brief Validate parameters
     * @return true if parameters are valid for a well-defined process
     */
    bool is_valid() const noexcept { return sigma > 0.0; }

    /**
     * @brief Validate and throw if invalid
     */
    void validate() const {
        if (sigma <= 0.0) {
            throw std::invalid_argument("OU: sigma must be positive, got " +
                                        std::to_string(sigma));
        }
    }

    /// String representation
    std::string to_string() const {
        return "OUParameters(theta=" + std::to_string(theta) + ", mu=" + std::to_string(mu) +
               ", sigma=" + std::to_string(sigma) + ", half_life=" + std::to_string(half_life()) +
               ")";
    }
};

/**
 * @brief MLE fitting result
 */
struct OUFitResult {
    OUParameters params;       ///< Estimated parameters
    double log_likelihood;     ///< Log-likelihood at optimum
    double aic;                ///< Akaike Information Criterion
    double bic;                ///< Bayesian Information Criterion
    size_t n_observations;     ///< Number of observations used
    bool converged = true;     ///< Whether optimization converged
    std::string message;       ///< Additional information
};

/**
 * @brief Ornstein-Uhlenbeck process utilities
 *
 * Provides:
 * - MLE parameter estimation from time series
 * - Log-likelihood computation
 * - Simulation of OU paths
 * - Transition density calculations
 */
class OUProcess {
public:
    /**
     * @brief Fit OU process to time series using Maximum Likelihood Estimation
     *
     * For equally-spaced observations, the OU process has an exact discrete-time
     * representation as an AR(1) process, enabling closed-form MLE:
     *
     *   X_{t+dt} = θ(1 - e^{-μdt}) + e^{-μdt} X_t + ε_t
     *
     * where ε_t ~ N(0, σ²(1-e^{-2μdt})/(2μ))
     *
     * Reference: See design doc Section 2.3
     *
     * @param prices Vector of observed values (e.g., log prices or spreads)
     * @param dt Time increment between observations (default 1/252 for daily)
     * @return OUFitResult with estimated parameters and diagnostics
     */
    static OUFitResult fit_mle(const std::vector<double>& prices, double dt = 1.0 / 252.0);

#ifdef QUANT_USE_EIGEN
    /**
     * @brief Fit OU process using Eigen vector
     */
    static OUFitResult fit_mle(const Eigen::VectorXd& prices, double dt = 1.0 / 252.0);
#endif

    /**
     * @brief Compute log-likelihood of observed data under OU model
     *
     * The log-likelihood is:
     *   LL = -n/2 * log(2π) - n/2 * log(σ²_ε) - (1/2σ²_ε) Σ(X_{t+1} - μ_t)²
     *
     * where μ_t and σ²_ε are the conditional mean and variance.
     *
     * @param prices Observed time series
     * @param params OU parameters
     * @param dt Time increment
     * @return Log-likelihood value
     */
    static double log_likelihood(const std::vector<double>& prices, const OUParameters& params,
                                 double dt = 1.0 / 252.0);

#ifdef QUANT_USE_EIGEN
    static double log_likelihood(const Eigen::VectorXd& prices, const OUParameters& params,
                                 double dt = 1.0 / 252.0);
#endif

    /**
     * @brief Conditional mean of X_{t+dt} given X_t
     *
     * E[X_{t+dt} | X_t] = θ + (X_t - θ) * e^{-μdt}
     */
    static double conditional_mean(double x_t, const OUParameters& params, double dt);

    /**
     * @brief Conditional variance of X_{t+dt} given X_t
     *
     * Var[X_{t+dt} | X_t] = σ²(1 - e^{-2μdt}) / (2μ)
     */
    static double conditional_variance(const OUParameters& params, double dt);

    /**
     * @brief Transition density p(X_{t+dt} | X_t)
     *
     * The transition density is Gaussian with:
     *   mean = conditional_mean(x_t, params, dt)
     *   variance = conditional_variance(params, dt)
     */
    static double transition_density(double x_next, double x_t, const OUParameters& params,
                                     double dt);

    /**
     * @brief Simulate OU process path using exact discretization
     *
     * Uses the exact solution:
     *   X_{t+dt} = θ + (X_t - θ)e^{-μdt} + σ√((1-e^{-2μdt})/(2μ)) * Z
     *
     * where Z ~ N(0,1).
     *
     * @param params OU parameters
     * @param x0 Initial value
     * @param T Total time horizon
     * @param n_steps Number of time steps
     * @param seed Random seed
     * @return Simulated path (n_steps + 1 values including x0)
     */
    static std::vector<double> simulate(const OUParameters& params, double x0, double T,
                                        size_t n_steps, unsigned int seed = 42);

#ifdef QUANT_USE_EIGEN
    static Eigen::VectorXd simulate_eigen(const OUParameters& params, double x0, double T,
                                          size_t n_steps, unsigned int seed = 42);
#endif

    /**
     * @brief Compute optimal trading boundaries for mean-reversion strategy
     *
     * Based on Leung & Li (2015), computes optimal entry and exit thresholds
     * for a mean-reversion trading strategy.
     *
     * @param params OU parameters
     * @param transaction_cost Round-trip transaction cost (as fraction)
     * @param risk_free_rate Risk-free rate
     * @return Tuple of (entry_lower, entry_upper, exit_target)
     */
    static std::tuple<double, double, double> optimal_boundaries(const OUParameters& params,
                                                                 double transaction_cost,
                                                                 double risk_free_rate);

private:
    /// Helper for MLE: compute sample statistics needed for estimation
    struct SampleStats {
        double sum_x;       // Σ X_t
        double sum_x_next;  // Σ X_{t+1}
        double sum_xx;      // Σ X_t²
        double sum_xx_next; // Σ X_{t+1}²
        double sum_x_xnext; // Σ X_t * X_{t+1}
        size_t n;           // Number of transitions
    };

    static SampleStats compute_sample_stats(const std::vector<double>& prices);
};

}  // namespace quant::models

#endif  // QUANT_TRADING_OU_PROCESS_HPP
