/**
 * @file test_heston.cpp
 * @brief Unit tests for Heston stochastic volatility model
 */

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "models/heston.hpp"

namespace quant::models {
namespace {

// Test fixture for Heston tests
class HestonTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Typical equity market parameters
        default_params = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04);
    }

    HestonParameters default_params;
};

// ============== Parameter Tests ==============

TEST_F(HestonTest, ParametersDefaultConstruction) {
    HestonParameters params;
    EXPECT_DOUBLE_EQ(params.kappa, 2.0);
    EXPECT_DOUBLE_EQ(params.theta, 0.04);
    EXPECT_DOUBLE_EQ(params.sigma, 0.3);
    EXPECT_DOUBLE_EQ(params.rho, -0.7);
    EXPECT_DOUBLE_EQ(params.v0, 0.04);
}

TEST_F(HestonTest, ParametersValidation) {
    EXPECT_TRUE(default_params.is_valid());

    // Invalid parameters
    HestonParameters bad_kappa(-1.0, 0.04, 0.3, -0.7, 0.04);
    EXPECT_FALSE(bad_kappa.is_valid());

    HestonParameters bad_theta(2.0, -0.04, 0.3, -0.7, 0.04);
    EXPECT_FALSE(bad_theta.is_valid());

    HestonParameters bad_sigma(2.0, 0.04, 0.0, -0.7, 0.04);
    EXPECT_FALSE(bad_sigma.is_valid());

    HestonParameters bad_rho(2.0, 0.04, 0.3, 1.5, 0.04);
    EXPECT_FALSE(bad_rho.is_valid());

    HestonParameters bad_v0(2.0, 0.04, 0.3, -0.7, -0.01);
    EXPECT_FALSE(bad_v0.is_valid());
}

TEST_F(HestonTest, FellerCondition) {
    // Feller satisfied: 2 * 2.0 * 0.04 = 0.16 >= 0.3^2 = 0.09
    EXPECT_TRUE(default_params.is_feller_satisfied());

    // Feller violated: 2 * 1.0 * 0.02 = 0.04 < 0.5^2 = 0.25
    HestonParameters bad_feller(1.0, 0.02, 0.5, -0.7, 0.04);
    EXPECT_FALSE(bad_feller.is_feller_satisfied());
}

TEST_F(HestonTest, ParametersValidateThrows) {
    HestonParameters bad_params(-1.0, 0.04, 0.3, -0.7, 0.04);
    EXPECT_THROW(bad_params.validate(), std::invalid_argument);
}

TEST_F(HestonTest, ParametersToString) {
    std::string str = default_params.to_string();
    EXPECT_NE(str.find("kappa="), std::string::npos);
    EXPECT_NE(str.find("feller=OK"), std::string::npos);
}

// ============== Model Construction Tests ==============

TEST_F(HestonTest, ModelConstruction) {
    EXPECT_NO_THROW(HestonModel model(default_params));
}

TEST_F(HestonTest, ModelConstructionInvalidParams) {
    HestonParameters bad_params(-1.0, 0.04, 0.3, -0.7, 0.04);
    EXPECT_THROW(HestonModel model(bad_params), std::invalid_argument);
}

TEST_F(HestonTest, ModelSetParameters) {
    HestonModel model(default_params);

    HestonParameters new_params(3.0, 0.05, 0.4, -0.5, 0.05);
    EXPECT_NO_THROW(model.set_parameters(new_params));

    EXPECT_DOUBLE_EQ(model.parameters().kappa, 3.0);
}

// ============== Characteristic Function Tests ==============

TEST_F(HestonTest, CharacteristicFunctionAtZero) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    // φ(0) should equal 1 (characteristic function at u=0)
    std::complex<double> phi = model.characteristic_function(
        std::complex<double>(0.0, 0.0), T, S0, r, q);

    EXPECT_NEAR(phi.real(), 1.0, 1e-10);
    EXPECT_NEAR(phi.imag(), 0.0, 1e-10);
}

TEST_F(HestonTest, CharacteristicFunctionZeroMaturity) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double T = 0.0;
    double r = 0.05;
    double q = 0.02;

    std::complex<double> u(1.0, 0.0);
    std::complex<double> phi = model.characteristic_function(u, T, S0, r, q);

    // At T=0, φ(u) = exp(iu * log(S0))
    std::complex<double> expected = std::exp(std::complex<double>(0.0, 1.0) * u * std::log(S0));

    EXPECT_NEAR(phi.real(), expected.real(), 1e-10);
    EXPECT_NEAR(phi.imag(), expected.imag(), 1e-10);
}

TEST_F(HestonTest, CharacteristicFunctionNumericalStability) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    // Test with various u values - should not produce NaN or Inf
    std::vector<std::complex<double>> u_values = {
        {0.1, 0.0}, {1.0, 0.0}, {5.0, 0.0}, {10.0, 0.0},
        {0.0, -0.5}, {1.0, -1.0}, {5.0, -2.0}
    };

    for (const auto& u : u_values) {
        std::complex<double> phi = model.characteristic_function(u, T, S0, r, q);
        EXPECT_FALSE(std::isnan(phi.real())) << "NaN at u=" << u;
        EXPECT_FALSE(std::isnan(phi.imag())) << "NaN at u=" << u;
        EXPECT_FALSE(std::isinf(phi.real())) << "Inf at u=" << u;
        EXPECT_FALSE(std::isinf(phi.imag())) << "Inf at u=" << u;
    }
}

// ============== Option Pricing Tests ==============

TEST_F(HestonTest, PriceCallOption) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    double call_price = model.price_option(K, T, S0, r, q, true);

    // Price should be positive
    EXPECT_GT(call_price, 0.0);

    // Price should be less than spot (for calls)
    EXPECT_LT(call_price, S0);

    // ATM call with typical params should be roughly 5-15% of spot
    EXPECT_GT(call_price, 3.0);
    EXPECT_LT(call_price, 20.0);
}

TEST_F(HestonTest, PricePutOption) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    double put_price = model.price_option(K, T, S0, r, q, false);

    // Price should be positive
    EXPECT_GT(put_price, 0.0);

    // Price should be less than strike
    EXPECT_LT(put_price, K);
}

TEST_F(HestonTest, PutCallParity) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    double call_price = model.price_option(K, T, S0, r, q, true);
    double put_price = model.price_option(K, T, S0, r, q, false);

    // Put-Call Parity: C - P = S*exp(-qT) - K*exp(-rT)
    double expected_diff = S0 * std::exp(-q * T) - K * std::exp(-r * T);
    double actual_diff = call_price - put_price;

    EXPECT_NEAR(actual_diff, expected_diff, 0.5);  // Allow some numerical error
}

TEST_F(HestonTest, PriceZeroMaturity) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K_itm = 90.0;
    double K_otm = 110.0;
    double T = 0.0;
    double r = 0.05;
    double q = 0.02;

    // ITM call: max(S0 - K, 0) = 10
    double call_itm = model.price_option(K_itm, T, S0, r, q, true);
    EXPECT_NEAR(call_itm, 10.0, 1e-6);

    // OTM call: max(S0 - K, 0) = 0
    double call_otm = model.price_option(K_otm, T, S0, r, q, true);
    EXPECT_NEAR(call_otm, 0.0, 1e-6);
}

TEST_F(HestonTest, PriceOptionInvalidInputs) {
    HestonModel model(default_params);

    EXPECT_THROW(model.price_option(-100.0, 1.0, 100.0, 0.05, 0.02, true),
                 std::invalid_argument);  // negative strike
    EXPECT_THROW(model.price_option(100.0, 1.0, -100.0, 0.05, 0.02, true),
                 std::invalid_argument);  // negative spot
    EXPECT_THROW(model.price_option(100.0, -1.0, 100.0, 0.05, 0.02, true),
                 std::invalid_argument);  // negative maturity
}

TEST_F(HestonTest, PriceMultipleOptions) {
    HestonModel model(default_params);

    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    std::vector<double> maturities = {1.0};  // Single maturity
    double S0 = 100.0;
    double r = 0.05;
    double q = 0.02;

    std::vector<double> prices = model.price_options(strikes, maturities, S0, r, q, true);

    EXPECT_EQ(prices.size(), strikes.size());

    // Prices should be monotonically decreasing with strike for calls
    for (size_t i = 1; i < prices.size(); ++i) {
        EXPECT_LT(prices[i], prices[i - 1]);
    }
}

// ============== Greeks Tests ==============

TEST_F(HestonTest, GreeksDelta) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    PricingResult result = model.price_option_with_greeks(K, T, S0, r, q, true);

    EXPECT_TRUE(result.greeks_computed);

    // Call delta should be between 0 and 1
    EXPECT_GT(result.greeks.delta, 0.0);
    EXPECT_LT(result.greeks.delta, 1.0);

    // ATM call delta should be around 0.5
    EXPECT_GT(result.greeks.delta, 0.3);
    EXPECT_LT(result.greeks.delta, 0.7);
}

TEST_F(HestonTest, GreeksGamma) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    PricingResult result = model.price_option_with_greeks(K, T, S0, r, q, true);

    // Gamma should be positive for both calls and puts
    EXPECT_GT(result.greeks.gamma, 0.0);
}

// ============== Implied Volatility Tests ==============

TEST_F(HestonTest, ImpliedVolatilityRoundTrip) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double K = 100.0;
    double T = 1.0;
    double r = 0.05;
    double q = 0.02;

    // Get Heston implied vol
    double implied_vol = model.implied_volatility(K, T, S0, r, q, true);

    // Should be positive and reasonable
    EXPECT_GT(implied_vol, 0.05);
    EXPECT_LT(implied_vol, 1.0);

    // Should be close to sqrt(v0) for ATM short-dated options
    double expected_vol = std::sqrt(default_params.v0);
    EXPECT_NEAR(implied_vol, expected_vol, 0.1);
}

TEST_F(HestonTest, ImpliedVolatilitySmile) {
    HestonModel model(default_params);

    double S0 = 100.0;
    double T = 0.5;
    double r = 0.05;
    double q = 0.02;

    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    std::vector<double> vols;

    for (double K : strikes) {
        double vol = model.implied_volatility(K, T, S0, r, q, true);
        vols.push_back(vol);
        EXPECT_GT(vol, 0.0);
        EXPECT_LT(vol, 2.0);
    }

    // With negative rho (-0.7), we generally expect higher vol for lower strikes (skew)
    // This is a soft check - the smile shape depends on all parameters
    // At minimum, implied vols should vary across strikes (not constant)
    double vol_range = *std::max_element(vols.begin(), vols.end()) -
                       *std::min_element(vols.begin(), vols.end());
    EXPECT_GT(vol_range, 0.001);  // Some variation in the smile
}

}  // namespace
}  // namespace quant::models
