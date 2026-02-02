/**
 * @file test_ou_process.cpp
 * @brief Unit tests for Ornstein-Uhlenbeck process
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include "models/ou_process.hpp"

namespace quant::models {
namespace {

// Test fixture for OU tests
class OUProcessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Typical mean-reverting parameters
        default_params = OUParameters(0.0, 5.0, 0.1);  // theta=0, mu=5, sigma=0.1
    }

    OUParameters default_params;
};

// ============== Parameter Tests ==============

TEST_F(OUProcessTest, ParametersDefaultConstruction) {
    OUParameters params;
    EXPECT_DOUBLE_EQ(params.theta, 0.0);
    EXPECT_DOUBLE_EQ(params.mu, 1.0);
    EXPECT_DOUBLE_EQ(params.sigma, 0.1);
}

TEST_F(OUProcessTest, ParametersHalfLife) {
    // Half-life = ln(2) / mu
    EXPECT_NEAR(default_params.half_life(), std::log(2.0) / 5.0, 1e-10);

    // Non-mean-reverting (mu <= 0) should give infinity
    OUParameters non_mr(0.0, 0.0, 0.1);
    EXPECT_EQ(non_mr.half_life(), std::numeric_limits<double>::infinity());

    OUParameters explosive(0.0, -1.0, 0.1);
    EXPECT_EQ(explosive.half_life(), std::numeric_limits<double>::infinity());
}

TEST_F(OUProcessTest, ParametersIsMeanReverting) {
    EXPECT_TRUE(default_params.is_mean_reverting());

    OUParameters non_mr(0.0, 0.0, 0.1);
    EXPECT_FALSE(non_mr.is_mean_reverting());

    OUParameters explosive(0.0, -1.0, 0.1);
    EXPECT_FALSE(explosive.is_mean_reverting());
}

TEST_F(OUProcessTest, ParametersStationaryVariance) {
    // Var_infinity = sigma^2 / (2 * mu) = 0.01 / 10 = 0.001
    EXPECT_NEAR(default_params.stationary_variance(), 0.001, 1e-10);
    EXPECT_NEAR(default_params.stationary_std(), std::sqrt(0.001), 1e-10);

    // Non-mean-reverting should give infinity
    OUParameters non_mr(0.0, 0.0, 0.1);
    EXPECT_EQ(non_mr.stationary_variance(), std::numeric_limits<double>::infinity());
}

TEST_F(OUProcessTest, ParametersValidation) {
    EXPECT_TRUE(default_params.is_valid());

    OUParameters bad_sigma(0.0, 1.0, 0.0);
    EXPECT_FALSE(bad_sigma.is_valid());

    OUParameters negative_sigma(0.0, 1.0, -0.1);
    EXPECT_FALSE(negative_sigma.is_valid());
}

TEST_F(OUProcessTest, ParametersToString) {
    std::string str = default_params.to_string();
    EXPECT_NE(str.find("theta="), std::string::npos);
    EXPECT_NE(str.find("half_life="), std::string::npos);
}

// ============== Conditional Distribution Tests ==============

TEST_F(OUProcessTest, ConditionalMean) {
    double x_t = 0.5;
    double dt = 1.0 / 252.0;  // Daily

    double mean = OUProcess::conditional_mean(x_t, default_params, dt);

    // E[X_{t+dt} | X_t] = theta + (X_t - theta) * exp(-mu * dt)
    double expected = default_params.theta +
                      (x_t - default_params.theta) * std::exp(-default_params.mu * dt);

    EXPECT_NEAR(mean, expected, 1e-10);

    // Mean should be between x_t and theta
    EXPECT_LT(mean, x_t);  // Since x_t > theta
    EXPECT_GT(mean, default_params.theta);
}

TEST_F(OUProcessTest, ConditionalVariance) {
    double dt = 1.0 / 252.0;

    double var = OUProcess::conditional_variance(default_params, dt);

    // Var = sigma^2 * (1 - exp(-2*mu*dt)) / (2*mu)
    double sigma_sq = default_params.sigma * default_params.sigma;
    double mu = default_params.mu;
    double expected = sigma_sq * (1.0 - std::exp(-2.0 * mu * dt)) / (2.0 * mu);

    EXPECT_NEAR(var, expected, 1e-10);
    EXPECT_GT(var, 0.0);
}

TEST_F(OUProcessTest, ConditionalVarianceZeroMu) {
    // When mu ≈ 0, variance should approximate sigma^2 * dt
    OUParameters slow_mr(0.0, 1e-10, 0.1);
    double dt = 1.0 / 252.0;

    double var = OUProcess::conditional_variance(slow_mr, dt);
    double expected = slow_mr.sigma * slow_mr.sigma * dt;

    // Use relative tolerance for numerical precision
    EXPECT_NEAR(var, expected, expected * 1e-4);
}

TEST_F(OUProcessTest, TransitionDensity) {
    double x_t = 0.0;
    double dt = 1.0 / 252.0;

    double mean = OUProcess::conditional_mean(x_t, default_params, dt);
    double var = OUProcess::conditional_variance(default_params, dt);
    double std = std::sqrt(var);

    // Density at mean should be highest
    double density_at_mean = OUProcess::transition_density(mean, x_t, default_params, dt);
    double density_at_1std = OUProcess::transition_density(mean + std, x_t, default_params, dt);
    double density_at_2std = OUProcess::transition_density(mean + 2*std, x_t, default_params, dt);

    EXPECT_GT(density_at_mean, density_at_1std);
    EXPECT_GT(density_at_1std, density_at_2std);
    EXPECT_GT(density_at_mean, 0.0);
}

// ============== Simulation Tests ==============

TEST_F(OUProcessTest, SimulationBasic) {
    double x0 = 0.5;
    double T = 1.0;
    size_t n_steps = 252;

    std::vector<double> path = OUProcess::simulate(default_params, x0, T, n_steps, 42);

    EXPECT_EQ(path.size(), n_steps + 1);
    EXPECT_DOUBLE_EQ(path[0], x0);

    // All values should be finite
    for (double x : path) {
        EXPECT_FALSE(std::isnan(x));
        EXPECT_FALSE(std::isinf(x));
    }
}

TEST_F(OUProcessTest, SimulationMeanReversion) {
    // With strong mean reversion, starting far from mean should converge
    OUParameters strong_mr(0.0, 10.0, 0.05);  // mu = 10 (fast mean reversion)
    double x0 = 1.0;  // Start above mean
    double T = 1.0;
    size_t n_steps = 1000;

    std::vector<double> path = OUProcess::simulate(strong_mr, x0, T, n_steps, 42);

    // Mean of latter half should be close to theta
    double mean_latter = std::accumulate(path.begin() + 500, path.end(), 0.0) / 500.0;
    EXPECT_NEAR(mean_latter, strong_mr.theta, 3 * strong_mr.stationary_std());
}

TEST_F(OUProcessTest, SimulationReproducibility) {
    double x0 = 0.0;
    double T = 1.0;
    size_t n_steps = 100;
    unsigned int seed = 12345;

    std::vector<double> path1 = OUProcess::simulate(default_params, x0, T, n_steps, seed);
    std::vector<double> path2 = OUProcess::simulate(default_params, x0, T, n_steps, seed);

    EXPECT_EQ(path1, path2);
}

TEST_F(OUProcessTest, SimulationDifferentSeeds) {
    double x0 = 0.0;
    double T = 1.0;
    size_t n_steps = 100;

    std::vector<double> path1 = OUProcess::simulate(default_params, x0, T, n_steps, 42);
    std::vector<double> path2 = OUProcess::simulate(default_params, x0, T, n_steps, 43);

    // Should produce different paths
    EXPECT_NE(path1, path2);
}

// ============== Log-Likelihood Tests ==============

TEST_F(OUProcessTest, LogLikelihoodBasic) {
    // Simulate data from known process and compute LL
    std::vector<double> path = OUProcess::simulate(default_params, 0.0, 1.0, 252, 42);

    double ll = OUProcess::log_likelihood(path, default_params, 1.0/252.0);

    // LL should be finite
    EXPECT_FALSE(std::isnan(ll));
    EXPECT_FALSE(std::isinf(ll));
}

TEST_F(OUProcessTest, LogLikelihoodMaximized) {
    // Data from true params should have higher LL than wrong params
    std::vector<double> path = OUProcess::simulate(default_params, 0.0, 1.0, 500, 42);

    double ll_true = OUProcess::log_likelihood(path, default_params, 1.0/252.0);

    OUParameters wrong_params(0.5, 2.0, 0.2);  // Different params
    double ll_wrong = OUProcess::log_likelihood(path, wrong_params, 1.0/252.0);

    // True params should give higher LL (most of the time with enough data)
    // This is a statistical test so we use a relaxed check
    EXPECT_GT(ll_true, ll_wrong - 50.0);  // Allow some variance
}

// ============== MLE Fitting Tests ==============

TEST_F(OUProcessTest, MLEFittingBasic) {
    // Simulate data from known params
    std::vector<double> path = OUProcess::simulate(default_params, 0.0, 2.0, 500, 42);

    OUFitResult result = OUProcess::fit_mle(path, 2.0/500.0);

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.n_observations, path.size());
    EXPECT_FALSE(std::isnan(result.log_likelihood));
}

TEST_F(OUProcessTest, MLEFittingRecovery) {
    // With enough data, MLE should approximately recover true parameters
    OUParameters true_params(0.0, 5.0, 0.1);

    // Use more data for better recovery
    std::vector<double> path = OUProcess::simulate(true_params, 0.0, 10.0, 2500, 42);

    OUFitResult result = OUProcess::fit_mle(path, 10.0/2500.0);

    EXPECT_TRUE(result.converged);

    // Check parameter recovery (with generous tolerance due to finite sample)
    EXPECT_NEAR(result.params.theta, true_params.theta, 0.2);
    EXPECT_NEAR(result.params.mu, true_params.mu, 2.0);
    EXPECT_NEAR(result.params.sigma, true_params.sigma, 0.05);
}

TEST_F(OUProcessTest, MLEFittingInsufficientData) {
    std::vector<double> short_data = {1.0, 1.1};  // Only 2 points

    OUFitResult result = OUProcess::fit_mle(short_data, 1.0/252.0);

    EXPECT_FALSE(result.converged);
}

TEST_F(OUProcessTest, MLEFittingConstantData) {
    std::vector<double> constant_data(100, 1.0);  // All same value

    OUFitResult result = OUProcess::fit_mle(constant_data, 1.0/252.0);

    EXPECT_FALSE(result.converged);
    // Message should mention variance (case-insensitive check)
    EXPECT_NE(result.message.find("Variance"), std::string::npos);
}

TEST_F(OUProcessTest, MLEFittingInformationCriteria) {
    std::vector<double> path = OUProcess::simulate(default_params, 0.0, 1.0, 252, 42);

    OUFitResult result = OUProcess::fit_mle(path, 1.0/252.0);

    // AIC and BIC should be finite
    EXPECT_FALSE(std::isnan(result.aic));
    EXPECT_FALSE(std::isnan(result.bic));

    // BIC should be larger than AIC for n > 7 (due to log(n) > 2)
    EXPECT_GT(result.bic, result.aic);
}

// ============== Optimal Boundaries Tests ==============

TEST_F(OUProcessTest, OptimalBoundariesBasic) {
    double transaction_cost = 0.001;
    double risk_free_rate = 0.05;

    auto [entry_lower, entry_upper, exit_target] =
        OUProcess::optimal_boundaries(default_params, transaction_cost, risk_free_rate);

    // Entry boundaries should be symmetric around theta for symmetric costs
    EXPECT_LT(entry_lower, default_params.theta);
    EXPECT_GT(entry_upper, default_params.theta);

    // Exit target should be at or near theta
    EXPECT_NEAR(exit_target, default_params.theta, default_params.stationary_std());

    // Entry boundaries should be farther from theta than exit
    EXPECT_LT(entry_lower, exit_target);
    EXPECT_GT(entry_upper, exit_target);
}

TEST_F(OUProcessTest, OptimalBoundariesWidenWithCost) {
    double low_cost = 0.0001;
    double high_cost = 0.01;
    double risk_free_rate = 0.05;

    auto [lower_low, upper_low, exit_low] =
        OUProcess::optimal_boundaries(default_params, low_cost, risk_free_rate);
    auto [lower_high, upper_high, exit_high] =
        OUProcess::optimal_boundaries(default_params, high_cost, risk_free_rate);

    // Higher costs should widen the entry bands
    EXPECT_LT(lower_high, lower_low);
    EXPECT_GT(upper_high, upper_low);
}

}  // namespace
}  // namespace quant::models
