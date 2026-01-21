#ifndef QUANT_TRADING_MATH_UTILS_HPP
#define QUANT_TRADING_MATH_UTILS_HPP

/**
 * @file math_utils.hpp
 * @brief Core mathematical utilities for quantitative trading system
 *
 * Provides foundational numerical operations used across the system:
 * - Statistical functions (mean, variance, covariance)
 * - Special functions (error function, normal CDF)
 * - Linear algebra helpers
 */

#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>

namespace quant {

/**
 * @brief Compute the mean of a vector
 * @param data Input vector
 * @return Arithmetic mean
 */
double mean(const std::vector<double>& data);

/**
 * @brief Compute the variance of a vector
 * @param data Input vector
 * @param ddof Delta degrees of freedom (default 1 for sample variance)
 * @return Sample variance
 */
double variance(const std::vector<double>& data, int ddof = 1);

/**
 * @brief Compute the standard deviation of a vector
 * @param data Input vector
 * @param ddof Delta degrees of freedom (default 1 for sample std)
 * @return Sample standard deviation
 */
double std_dev(const std::vector<double>& data, int ddof = 1);

/**
 * @brief Standard normal cumulative distribution function
 * @param x Point at which to evaluate
 * @return P(Z <= x) where Z ~ N(0,1)
 */
double norm_cdf(double x);

/**
 * @brief Standard normal probability density function
 * @param x Point at which to evaluate
 * @return PDF value at x
 */
double norm_pdf(double x);

} // namespace quant

#endif // QUANT_TRADING_MATH_UTILS_HPP
