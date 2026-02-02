#include "ou_process.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace quant::models {

namespace {

/// Small number for numerical comparisons
constexpr double EPSILON = 1e-12;

/// Pi constant
constexpr double PI = 3.14159265358979323846;

/// Log of 2*pi for log-likelihood computation
constexpr double LOG_2PI = 1.8378770664093453;

}  // anonymous namespace

OUProcess::SampleStats OUProcess::compute_sample_stats(const std::vector<double>& prices) {
    SampleStats stats{};
    stats.n = prices.size() - 1;

    if (stats.n == 0) {
        return stats;
    }

    for (size_t i = 0; i < stats.n; ++i) {
        double x = prices[i];
        double x_next = prices[i + 1];

        stats.sum_x += x;
        stats.sum_x_next += x_next;
        stats.sum_xx += x * x;
        stats.sum_xx_next += x_next * x_next;
        stats.sum_x_xnext += x * x_next;
    }

    return stats;
}

OUFitResult OUProcess::fit_mle(const std::vector<double>& prices, double dt) {
    OUFitResult result;
    result.n_observations = prices.size();

    if (prices.size() < 3) {
        result.converged = false;
        result.message = "Need at least 3 observations for MLE";
        return result;
    }

    // Compute sample statistics
    SampleStats stats = compute_sample_stats(prices);
    double n = static_cast<double>(stats.n);

    // Sample means
    double mean_x = stats.sum_x / n;
    double mean_x_next = stats.sum_x_next / n;

    // Sample variances and covariance
    double var_x = (stats.sum_xx / n) - mean_x * mean_x;
    double var_x_next = (stats.sum_xx_next / n) - mean_x_next * mean_x_next;
    double cov_x_xnext = (stats.sum_x_xnext / n) - mean_x * mean_x_next;

    // Handle edge case: zero variance (constant series)
    if (var_x < EPSILON) {
        result.converged = false;
        result.message = "Variance is too small - data may be constant";
        result.params.theta = mean_x;
        result.params.mu = 0.0;
        result.params.sigma = 0.0;
        return result;
    }

    // MLE estimation using AR(1) regression
    // X_{t+dt} = a + b * X_t + epsilon
    // where:
    //   b = e^{-μdt}
    //   a = θ(1 - b)

    // OLS estimates
    double b_hat = cov_x_xnext / var_x;

    // Ensure b_hat is in valid range for mean-reverting process
    // b should be in (0, 1) for mean reversion
    if (b_hat >= 1.0) {
        // Non-mean-reverting or explosive - cap at slight mean reversion
        b_hat = 0.9999;
        result.message = "Process appears non-mean-reverting; mu estimate may be unreliable";
    } else if (b_hat <= 0.0) {
        // Strongly mean-reverting or oscillatory
        b_hat = 0.0001;
        result.message = "Process appears strongly mean-reverting; mu estimate may be unreliable";
    }

    // Recover μ from b = e^{-μdt}
    double mu_hat = -std::log(b_hat) / dt;

    // Recover θ from a = θ(1 - b)
    double a_hat = mean_x_next - b_hat * mean_x;
    double theta_hat;
    if (std::abs(1.0 - b_hat) > EPSILON) {
        theta_hat = a_hat / (1.0 - b_hat);
    } else {
        // b ≈ 1 means very slow mean reversion; use sample mean as theta estimate
        theta_hat = (mean_x + mean_x_next) / 2.0;
    }

    // Estimate σ from residual variance
    // Var[ε] = σ²(1 - e^{-2μdt}) / (2μ)
    // Residual variance from regression
    double residual_var = var_x_next - b_hat * b_hat * var_x;
    residual_var = std::max(residual_var, EPSILON);  // Ensure positive

    // Solve for σ
    double sigma_hat;
    if (mu_hat > EPSILON) {
        // σ² = 2μ * Var[ε] / (1 - e^{-2μdt})
        double exp_factor = 1.0 - std::exp(-2.0 * mu_hat * dt);
        if (exp_factor > EPSILON) {
            double sigma_sq = 2.0 * mu_hat * residual_var / exp_factor;
            sigma_hat = std::sqrt(sigma_sq);
        } else {
            // Very small mu, use approximation: Var[ε] ≈ σ²dt
            sigma_hat = std::sqrt(residual_var / dt);
        }
    } else {
        // mu ≈ 0, process is approximately random walk: Var[ε] ≈ σ²dt
        sigma_hat = std::sqrt(residual_var / dt);
    }

    // Store parameters
    result.params.theta = theta_hat;
    result.params.mu = mu_hat;
    result.params.sigma = sigma_hat;

    // Compute log-likelihood at MLE
    result.log_likelihood = log_likelihood(prices, result.params, dt);

    // Compute information criteria
    // AIC = -2*LL + 2*k (k=3 parameters)
    // BIC = -2*LL + k*log(n)
    result.aic = -2.0 * result.log_likelihood + 2.0 * 3.0;
    result.bic = -2.0 * result.log_likelihood + 3.0 * std::log(n);

    result.converged = true;
    return result;
}

#ifdef QUANT_USE_EIGEN
OUFitResult OUProcess::fit_mle(const Eigen::VectorXd& prices, double dt) {
    std::vector<double> prices_vec(prices.data(), prices.data() + prices.size());
    return fit_mle(prices_vec, dt);
}
#endif

double OUProcess::conditional_mean(double x_t, const OUParameters& params, double dt) {
    // E[X_{t+dt} | X_t] = θ + (X_t - θ) * e^{-μdt}
    double exp_mu_dt = std::exp(-params.mu * dt);
    return params.theta + (x_t - params.theta) * exp_mu_dt;
}

double OUProcess::conditional_variance(const OUParameters& params, double dt) {
    // Var[X_{t+dt} | X_t] = σ²(1 - e^{-2μdt}) / (2μ)
    if (params.mu < EPSILON) {
        // μ ≈ 0: Var ≈ σ²dt (Brownian motion limit)
        return params.sigma * params.sigma * dt;
    }

    double exp_factor = 1.0 - std::exp(-2.0 * params.mu * dt);
    return (params.sigma * params.sigma * exp_factor) / (2.0 * params.mu);
}

double OUProcess::transition_density(double x_next, double x_t, const OUParameters& params,
                                     double dt) {
    double mean = conditional_mean(x_t, params, dt);
    double var = conditional_variance(params, dt);

    if (var < EPSILON) {
        // Degenerate case: return very high density if x_next ≈ mean, else 0
        return (std::abs(x_next - mean) < EPSILON) ? 1e10 : 0.0;
    }

    double std_dev = std::sqrt(var);
    double z = (x_next - mean) / std_dev;

    // Gaussian PDF
    return std::exp(-0.5 * z * z) / (std_dev * std::sqrt(2.0 * PI));
}

double OUProcess::log_likelihood(const std::vector<double>& prices, const OUParameters& params,
                                 double dt) {
    if (prices.size() < 2) {
        return -std::numeric_limits<double>::infinity();
    }

    size_t n = prices.size() - 1;
    double cond_var = conditional_variance(params, dt);

    if (cond_var < EPSILON) {
        return -std::numeric_limits<double>::infinity();
    }

    double log_var = std::log(cond_var);
    double sum_sq_resid = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double mean = conditional_mean(prices[i], params, dt);
        double resid = prices[i + 1] - mean;
        sum_sq_resid += resid * resid;
    }

    // LL = -n/2 * log(2π) - n/2 * log(σ²) - (1/2σ²) * Σ(residuals)²
    double ll = -0.5 * n * LOG_2PI - 0.5 * n * log_var - 0.5 * sum_sq_resid / cond_var;

    return ll;
}

#ifdef QUANT_USE_EIGEN
double OUProcess::log_likelihood(const Eigen::VectorXd& prices, const OUParameters& params,
                                 double dt) {
    std::vector<double> prices_vec(prices.data(), prices.data() + prices.size());
    return log_likelihood(prices_vec, params, dt);
}
#endif

std::vector<double> OUProcess::simulate(const OUParameters& params, double x0, double T,
                                        size_t n_steps, unsigned int seed) {
    std::vector<double> path(n_steps + 1);
    path[0] = x0;

    if (n_steps == 0 || T <= 0.0) {
        return path;
    }

    double dt = T / static_cast<double>(n_steps);

    // Pre-compute constants for exact simulation
    double exp_mu_dt = std::exp(-params.mu * dt);
    double std_dev = std::sqrt(conditional_variance(params, dt));

    // Random number generator
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (size_t i = 0; i < n_steps; ++i) {
        // Exact simulation: X_{t+dt} = θ + (X_t - θ)e^{-μdt} + std_dev * Z
        double z = normal(rng);
        path[i + 1] = params.theta + (path[i] - params.theta) * exp_mu_dt + std_dev * z;
    }

    return path;
}

#ifdef QUANT_USE_EIGEN
Eigen::VectorXd OUProcess::simulate_eigen(const OUParameters& params, double x0, double T,
                                          size_t n_steps, unsigned int seed) {
    std::vector<double> path_vec = simulate(params, x0, T, n_steps, seed);
    Eigen::VectorXd path(path_vec.size());
    for (size_t i = 0; i < path_vec.size(); ++i) {
        path(i) = path_vec[i];
    }
    return path;
}
#endif

std::tuple<double, double, double> OUProcess::optimal_boundaries(const OUParameters& params,
                                                                 double transaction_cost,
                                                                 double /* risk_free_rate */) {
    // Simplified optimal boundary computation based on Leung & Li (2015)
    // For a full implementation, this would solve the HJB equation numerically
    // Note: risk_free_rate would be used in the full HJB solution

    // Entry boundaries: enter when price deviates significantly from mean
    // Exit at mean (take profit) or at stop-loss

    double theta = params.theta;

    // Stationary standard deviation
    double stat_std = params.stationary_std();

    // Heuristic boundaries based on stationary distribution
    // Entry when price is 1-2 standard deviations from mean
    double entry_threshold = 1.5 * stat_std;

    // Adjust for transaction costs: need larger deviation to overcome costs
    double cost_adjustment = transaction_cost / stat_std;
    entry_threshold += cost_adjustment * stat_std;

    double entry_lower = theta - entry_threshold;
    double entry_upper = theta + entry_threshold;

    // Exit target: close to theta (take profit)
    // Small buffer to avoid frequent trading
    double exit_target = theta;

    return std::make_tuple(entry_lower, entry_upper, exit_target);
}

}  // namespace quant::models
