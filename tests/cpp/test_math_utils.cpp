/**
 * @file test_math_utils.cpp
 * @brief Unit tests for math_utils functions
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "math_utils.hpp"

namespace {

// Test mean calculation
TEST(MathUtilsTest, MeanBasic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(quant::mean(data), 3.0);
}

TEST(MathUtilsTest, MeanSingleElement) {
    std::vector<double> data = {42.0};
    EXPECT_DOUBLE_EQ(quant::mean(data), 42.0);
}

TEST(MathUtilsTest, MeanEmptyThrows) {
    std::vector<double> data;
    EXPECT_THROW(quant::mean(data), std::invalid_argument);
}

// Test variance calculation
TEST(MathUtilsTest, VarianceBasic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    // Sample variance = sum((x - mean)^2) / (n-1)
    // = (4 + 1 + 0 + 1 + 4) / 4 = 2.5
    EXPECT_DOUBLE_EQ(quant::variance(data), 2.5);
}

TEST(MathUtilsTest, VariancePopulation) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    // Population variance = sum((x - mean)^2) / n
    // = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    EXPECT_DOUBLE_EQ(quant::variance(data, 0), 2.0);
}

TEST(MathUtilsTest, VarianceSingleElementThrows) {
    std::vector<double> data = {42.0};
    EXPECT_THROW(quant::variance(data), std::invalid_argument);
}

// Test standard deviation
TEST(MathUtilsTest, StdDevBasic) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(quant::std_dev(data), std::sqrt(2.5));
}

// Test normal CDF
TEST(MathUtilsTest, NormCdfZero) {
    // Phi(0) = 0.5
    EXPECT_NEAR(quant::norm_cdf(0.0), 0.5, 1e-10);
}

TEST(MathUtilsTest, NormCdfPositive) {
    // Phi(1.96) ≈ 0.975
    EXPECT_NEAR(quant::norm_cdf(1.96), 0.975, 0.001);
}

TEST(MathUtilsTest, NormCdfNegative) {
    // Phi(-1.96) ≈ 0.025
    EXPECT_NEAR(quant::norm_cdf(-1.96), 0.025, 0.001);
}

TEST(MathUtilsTest, NormCdfSymmetry) {
    // Phi(x) + Phi(-x) = 1
    double x = 1.5;
    EXPECT_NEAR(quant::norm_cdf(x) + quant::norm_cdf(-x), 1.0, 1e-10);
}

// Test normal PDF
TEST(MathUtilsTest, NormPdfZero) {
    // phi(0) = 1/sqrt(2*pi) ≈ 0.3989
    EXPECT_NEAR(quant::norm_pdf(0.0), 0.3989422804, 1e-8);
}

TEST(MathUtilsTest, NormPdfSymmetry) {
    // phi(x) = phi(-x)
    double x = 1.5;
    EXPECT_DOUBLE_EQ(quant::norm_pdf(x), quant::norm_pdf(-x));
}

TEST(MathUtilsTest, NormPdfDecreases) {
    // PDF decreases as |x| increases
    EXPECT_GT(quant::norm_pdf(0.0), quant::norm_pdf(1.0));
    EXPECT_GT(quant::norm_pdf(1.0), quant::norm_pdf(2.0));
}

} // namespace
