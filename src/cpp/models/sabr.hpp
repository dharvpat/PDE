#ifndef QUANT_TRADING_SABR_HPP
#define QUANT_TRADING_SABR_HPP

/**
 * @file sabr.hpp
 * @brief SABR stochastic volatility model implementation
 *
 * Reference: Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002).
 * "Managing smile risk." Wilmott Magazine, September, 84-108.
 *
 * The SABR model describes forward rate dynamics:
 *   dF_t = σ_t F_t^β dW_t^F
 *   dσ_t = ν σ_t dW_t^σ
 *   dW_t^F · dW_t^σ = ρ dt
 */

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef QUANT_USE_EIGEN
#include <Eigen/Dense>
#endif

namespace quant::models {

/**
 * @brief SABR model parameters
 *
 * Parameters:
 * - alpha: Initial volatility level (ATM vol for beta=1)
 * - beta: CEV exponent (backbone parameter, typically 0.5 for equity)
 * - rho: Correlation between forward and volatility
 * - nu: Volatility of volatility
 *
 * Constraints:
 * - alpha > 0
 * - beta ∈ [0, 1]
 * - |rho| < 1
 * - nu ≥ 0
 */
struct SABRParameters {
    double alpha;  ///< Initial volatility
    double beta;   ///< CEV exponent (0 = normal, 0.5 = equity, 1 = lognormal)
    double rho;    ///< Correlation between forward and volatility
    double nu;     ///< Volatility of volatility

    /// Default constructor with typical equity parameters
    SABRParameters() : alpha(0.2), beta(0.5), rho(-0.3), nu(0.4) {}

    /// Parameterized constructor
    SABRParameters(double alpha_, double beta_, double rho_, double nu_)
        : alpha(alpha_), beta(beta_), rho(rho_), nu(nu_) {}

    /**
     * @brief Validate all parameters are in acceptable ranges
     * @return true if all parameters are valid
     */
    bool is_valid() const noexcept {
        return alpha > 0.0 && beta >= 0.0 && beta <= 1.0 && std::abs(rho) < 1.0 && nu >= 0.0;
    }

    /**
     * @brief Validate parameters and throw if invalid
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (alpha <= 0.0) {
            throw std::invalid_argument("SABR: alpha must be positive, got " +
                                        std::to_string(alpha));
        }
        if (beta < 0.0 || beta > 1.0) {
            throw std::invalid_argument("SABR: beta must be in [0, 1], got " +
                                        std::to_string(beta));
        }
        if (std::abs(rho) >= 1.0) {
            throw std::invalid_argument("SABR: |rho| must be < 1, got " + std::to_string(rho));
        }
        if (nu < 0.0) {
            throw std::invalid_argument("SABR: nu must be non-negative, got " +
                                        std::to_string(nu));
        }
    }

    /// String representation for logging
    std::string to_string() const {
        return "SABRParameters(alpha=" + std::to_string(alpha) + ", beta=" + std::to_string(beta) +
               ", rho=" + std::to_string(rho) + ", nu=" + std::to_string(nu) + ")";
    }
};

/**
 * @brief SABR volatility model
 *
 * Implements the Hagan et al. (2002) asymptotic formula for implied volatility.
 * This provides a fast, closed-form approximation to the SABR implied volatility
 * that is accurate for most practical purposes.
 *
 * Key features:
 * - Fast evaluation: ~100 nanoseconds per implied vol calculation
 * - Accurate for strikes not too far from ATM
 * - Handles special cases (ATM, beta=0, beta=1)
 */
class SABRModel {
public:
    /**
     * @brief Construct SABR model with fixed beta
     * @param beta CEV exponent (default 0.5 for equity)
     */
    explicit SABRModel(double beta = 0.5);

    /// Get beta parameter
    double beta() const noexcept { return beta_; }

    /// Set beta parameter
    void set_beta(double beta);

    /**
     * @brief Compute SABR implied volatility using Hagan asymptotic formula
     *
     * For non-ATM strikes (K ≠ F):
     *   σ_impl = α * (z/χ(z)) * correction_factor
     *
     * where:
     *   z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
     *   χ(z) = ln((√(1-2ρz+z²) + z - ρ) / (1-ρ))
     *
     * Reference: Hagan et al. (2002), Equation (2.17a)
     *
     * @param strike Strike price K
     * @param forward Forward price F
     * @param maturity Time to maturity T (years)
     * @param alpha Initial volatility α
     * @param rho Correlation ρ
     * @param nu Vol of vol ν
     * @return Black-Scholes implied volatility
     */
    double implied_volatility(double strike, double forward, double maturity, double alpha,
                              double rho, double nu) const;

    /**
     * @brief Compute implied volatility with SABRParameters struct
     */
    double implied_volatility(double strike, double forward, double maturity,
                              const SABRParameters& params) const {
        return implied_volatility(strike, forward, maturity, params.alpha, params.rho, params.nu);
    }

    /**
     * @brief Compute ATM implied volatility (simpler formula)
     *
     * At the money (K = F), the formula simplifies to:
     *   σ_ATM = α / F^(1-β) * [1 + (correction_terms) * T]
     *
     * @param forward Forward price F
     * @param maturity Time to maturity T
     * @param alpha Initial volatility α
     * @param rho Correlation ρ
     * @param nu Vol of vol ν
     * @return ATM implied volatility
     */
    double atm_volatility(double forward, double maturity, double alpha, double rho,
                          double nu) const;

    /**
     * @brief Compute implied volatilities for multiple strikes (vectorized)
     *
     * @param strikes Vector of strike prices
     * @param forward Forward price
     * @param maturity Time to maturity
     * @param alpha Initial volatility
     * @param rho Correlation
     * @param nu Vol of vol
     * @return Vector of implied volatilities
     */
    std::vector<double> implied_volatilities(const std::vector<double>& strikes, double forward,
                                             double maturity, double alpha, double rho,
                                             double nu) const;

#ifdef QUANT_USE_EIGEN
    /**
     * @brief Compute implied volatilities using Eigen vectors
     */
    Eigen::VectorXd implied_volatilities_eigen(const Eigen::VectorXd& strikes, double forward,
                                               double maturity, double alpha, double rho,
                                               double nu) const;
#endif

    /**
     * @brief Compute volatility smile sensitivities
     *
     * Returns partial derivatives of implied volatility with respect to
     * SABR parameters, useful for calibration and risk management.
     *
     * @param strike Strike price
     * @param forward Forward price
     * @param maturity Time to maturity
     * @param alpha Initial volatility
     * @param rho Correlation
     * @param nu Vol of vol
     * @return Tuple of (d_sigma/d_alpha, d_sigma/d_rho, d_sigma/d_nu)
     */
    std::tuple<double, double, double> volatility_sensitivities(double strike, double forward,
                                                                double maturity, double alpha,
                                                                double rho, double nu) const;

private:
    double beta_;  ///< CEV exponent (fixed for model instance)

    /**
     * @brief χ(z) function from Hagan formula
     *
     * χ(z) = ln((√(1-2ρz+z²) + z - ρ) / (1-ρ))
     *
     * Handles small z via Taylor expansion for numerical stability.
     */
    double chi_function(double z, double rho) const;

    /**
     * @brief Compute z variable for non-ATM case
     *
     * z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
     */
    double compute_z(double strike, double forward, double alpha, double nu) const;

    /**
     * @brief Compute correction factor for non-ATM formula
     */
    double compute_correction_factor(double strike, double forward, double maturity, double alpha,
                                     double rho, double nu) const;
};

}  // namespace quant::models

#endif  // QUANT_TRADING_SABR_HPP
