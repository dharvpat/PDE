/**
 * @file test_sabr.cpp
 * @brief Unit tests for SABR volatility model
 */

#include <gtest/gtest.h>
#include <cmath>
#include "models/sabr.hpp"

namespace quant::models {
namespace {

// Test fixture for SABR tests
class SABRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Typical equity parameters
        default_params = SABRParameters(0.2, 0.5, -0.3, 0.4);
    }

    SABRParameters default_params;
};

// ============== Parameter Tests ==============

TEST_F(SABRTest, ParametersDefaultConstruction) {
    SABRParameters params;
    EXPECT_DOUBLE_EQ(params.alpha, 0.2);
    EXPECT_DOUBLE_EQ(params.beta, 0.5);
    EXPECT_DOUBLE_EQ(params.rho, -0.3);
    EXPECT_DOUBLE_EQ(params.nu, 0.4);
}

TEST_F(SABRTest, ParametersValidation) {
    EXPECT_TRUE(default_params.is_valid());

    // Invalid parameters
    SABRParameters bad_alpha(0.0, 0.5, -0.3, 0.4);
    EXPECT_FALSE(bad_alpha.is_valid());

    SABRParameters bad_beta(0.2, 1.5, -0.3, 0.4);
    EXPECT_FALSE(bad_beta.is_valid());

    SABRParameters bad_rho(0.2, 0.5, 1.5, 0.4);
    EXPECT_FALSE(bad_rho.is_valid());

    SABRParameters bad_nu(0.2, 0.5, -0.3, -0.1);
    EXPECT_FALSE(bad_nu.is_valid());
}

TEST_F(SABRTest, ParametersValidateThrows) {
    SABRParameters bad_params(0.0, 0.5, -0.3, 0.4);
    EXPECT_THROW(bad_params.validate(), std::invalid_argument);
}

TEST_F(SABRTest, ParametersToString) {
    std::string str = default_params.to_string();
    EXPECT_NE(str.find("alpha="), std::string::npos);
    EXPECT_NE(str.find("beta="), std::string::npos);
}

// ============== Model Construction Tests ==============

TEST_F(SABRTest, ModelConstruction) {
    EXPECT_NO_THROW(SABRModel model(0.5));
}

TEST_F(SABRTest, ModelConstructionInvalidBeta) {
    EXPECT_THROW(SABRModel model(-0.1), std::invalid_argument);
    EXPECT_THROW(SABRModel model(1.5), std::invalid_argument);
}

TEST_F(SABRTest, ModelSetBeta) {
    SABRModel model(0.5);
    EXPECT_NO_THROW(model.set_beta(0.7));
    EXPECT_DOUBLE_EQ(model.beta(), 0.7);

    EXPECT_THROW(model.set_beta(1.5), std::invalid_argument);
}

// ============== ATM Volatility Tests ==============

TEST_F(SABRTest, ATMVolatilityBasic) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    double atm_vol = model.atm_volatility(forward, maturity, alpha, rho, nu);

    // Should be positive
    EXPECT_GT(atm_vol, 0.0);

    // For beta=0.5, ATM vol should be roughly alpha / sqrt(F)
    double approx_vol = alpha / std::pow(forward, 0.5);
    EXPECT_NEAR(atm_vol, approx_vol, 0.05);  // Allow some correction term difference
}

TEST_F(SABRTest, ATMVolatilityZeroMaturity) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 0.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    double atm_vol = model.atm_volatility(forward, maturity, alpha, rho, nu);

    // At T=0, correction term vanishes
    double expected = alpha / std::pow(forward, 0.5);
    EXPECT_NEAR(atm_vol, expected, 1e-6);
}

TEST_F(SABRTest, ATMVolatilityInvalidInputs) {
    SABRModel model(0.5);

    EXPECT_THROW(model.atm_volatility(-100.0, 1.0, 0.2, -0.3, 0.4), std::invalid_argument);
    EXPECT_THROW(model.atm_volatility(100.0, -1.0, 0.2, -0.3, 0.4), std::invalid_argument);
    EXPECT_THROW(model.atm_volatility(100.0, 1.0, 0.0, -0.3, 0.4), std::invalid_argument);
    EXPECT_THROW(model.atm_volatility(100.0, 1.0, 0.2, 1.5, 0.4), std::invalid_argument);
    EXPECT_THROW(model.atm_volatility(100.0, 1.0, 0.2, -0.3, -0.1), std::invalid_argument);
}

// ============== Implied Volatility Tests ==============

TEST_F(SABRTest, ImpliedVolatilityATM) {
    SABRModel model(0.5);

    double forward = 100.0;
    double strike = 100.0;  // ATM
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    double vol = model.implied_volatility(strike, forward, maturity, alpha, rho, nu);
    double atm_vol = model.atm_volatility(forward, maturity, alpha, rho, nu);

    // Should match ATM formula
    EXPECT_NEAR(vol, atm_vol, 1e-6);
}

TEST_F(SABRTest, ImpliedVolatilityPositive) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    std::vector<double> strikes = {80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0};

    for (double K : strikes) {
        double vol = model.implied_volatility(K, forward, maturity, alpha, rho, nu);
        EXPECT_GT(vol, 0.0) << "Negative vol at strike " << K;
        EXPECT_LT(vol, 5.0) << "Unreasonably high vol at strike " << K;
    }
}

TEST_F(SABRTest, ImpliedVolatilitySmile) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;  // Negative rho creates downward skew
    double nu = 0.4;

    double vol_low = model.implied_volatility(80.0, forward, maturity, alpha, rho, nu);
    double vol_atm = model.implied_volatility(100.0, forward, maturity, alpha, rho, nu);
    double vol_high = model.implied_volatility(120.0, forward, maturity, alpha, rho, nu);

    // With negative rho, lower strikes should have higher vol (skew)
    EXPECT_GT(vol_low, vol_atm);

    // Vol smile: both wings should eventually be higher than ATM (for high nu)
    // This test may depend on parameter choices
}

TEST_F(SABRTest, ImpliedVolatilitySmoothness) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    // Check that volatility changes smoothly
    double prev_vol = model.implied_volatility(80.0, forward, maturity, alpha, rho, nu);

    for (double K = 81.0; K <= 120.0; K += 1.0) {
        double vol = model.implied_volatility(K, forward, maturity, alpha, rho, nu);

        // Vol should not change by more than 10% between adjacent strikes
        double change = std::abs(vol - prev_vol) / prev_vol;
        EXPECT_LT(change, 0.1) << "Large vol jump at strike " << K;

        prev_vol = vol;
    }
}

TEST_F(SABRTest, ImpliedVolatilityInvalidInputs) {
    SABRModel model(0.5);

    EXPECT_THROW(model.implied_volatility(-100.0, 100.0, 1.0, 0.2, -0.3, 0.4),
                 std::invalid_argument);
    EXPECT_THROW(model.implied_volatility(100.0, -100.0, 1.0, 0.2, -0.3, 0.4),
                 std::invalid_argument);
}

// ============== Vectorized Tests ==============

TEST_F(SABRTest, ImpliedVolatilitiesVector) {
    SABRModel model(0.5);

    std::vector<double> strikes = {80.0, 90.0, 100.0, 110.0, 120.0};
    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    std::vector<double> vols = model.implied_volatilities(strikes, forward, maturity,
                                                          alpha, rho, nu);

    EXPECT_EQ(vols.size(), strikes.size());

    // Compare to individual calls
    for (size_t i = 0; i < strikes.size(); ++i) {
        double expected = model.implied_volatility(strikes[i], forward, maturity,
                                                   alpha, rho, nu);
        EXPECT_NEAR(vols[i], expected, 1e-10);
    }
}

// ============== Beta Variations ==============

TEST_F(SABRTest, ImpliedVolatilityBetaZero) {
    // Beta = 0: Normal model
    SABRModel model(0.0);

    double forward = 100.0;
    double strike = 100.0;
    double maturity = 1.0;
    double alpha = 20.0;  // Higher alpha for normal model
    double rho = -0.3;
    double nu = 0.4;

    double vol = model.implied_volatility(strike, forward, maturity, alpha, rho, nu);

    EXPECT_GT(vol, 0.0);
    // For beta=0, ATM vol should be roughly alpha / F
    EXPECT_NEAR(vol, alpha / forward, 0.05);
}

TEST_F(SABRTest, ImpliedVolatilityBetaOne) {
    // Beta = 1: Lognormal model
    SABRModel model(1.0);

    double forward = 100.0;
    double strike = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    double vol = model.implied_volatility(strike, forward, maturity, alpha, rho, nu);

    EXPECT_GT(vol, 0.0);
    // For beta=1, ATM vol should be roughly alpha
    EXPECT_NEAR(vol, alpha, 0.05);
}

// ============== Sensitivities Tests ==============

TEST_F(SABRTest, VolatilitySensitivities) {
    SABRModel model(0.5);

    double strike = 100.0;
    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.4;

    auto [d_alpha, d_rho, d_nu] = model.volatility_sensitivities(
        strike, forward, maturity, alpha, rho, nu);

    // d_sigma/d_alpha should be positive (more alpha = more vol)
    EXPECT_GT(d_alpha, 0.0);

    // d_sigma/d_nu should be positive at wings (more vol-of-vol = more smile)
    // For ATM it depends on the exact formula

    // Sensitivities should be finite
    EXPECT_FALSE(std::isnan(d_alpha));
    EXPECT_FALSE(std::isnan(d_rho));
    EXPECT_FALSE(std::isnan(d_nu));
}

TEST_F(SABRTest, NumericalStabilitySmallZ) {
    SABRModel model(0.5);

    double forward = 100.0;
    double maturity = 1.0;
    double alpha = 0.2;
    double rho = -0.3;
    double nu = 0.001;  // Very small nu -> small z

    // Near ATM with small nu should not produce numerical issues
    for (double K = 99.0; K <= 101.0; K += 0.1) {
        double vol = model.implied_volatility(K, forward, maturity, alpha, rho, nu);
        EXPECT_FALSE(std::isnan(vol));
        EXPECT_FALSE(std::isinf(vol));
        EXPECT_GT(vol, 0.0);
    }
}

}  // namespace
}  // namespace quant::models
