#ifndef QUANT_TRADING_HESTON_HPP
#define QUANT_TRADING_HESTON_HPP

/**
 * @file heston.hpp
 * @brief Heston stochastic volatility model implementation
 *
 * Reference: Heston, S.L. (1993). "A closed-form solution for options with
 * stochastic volatility with applications to bond and currency options."
 * Review of Financial Studies, 6(2), 327-343.
 *
 * The Heston model describes asset price dynamics with stochastic volatility:
 *   dS_t = μS_t dt + √v_t S_t dW_t^S
 *   dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
 *   dW_t^S · dW_t^v = ρ dt
 */

#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef QUANT_USE_EIGEN
#include <Eigen/Dense>
#endif

namespace quant::models {

/**
 * @brief Heston stochastic volatility model parameters
 *
 * All parameters must satisfy certain constraints for the model to be valid:
 * - kappa > 0: Mean-reversion speed must be positive
 * - theta > 0: Long-term variance must be positive
 * - sigma > 0: Volatility of variance must be positive
 * - |rho| < 1: Correlation must be in (-1, 1)
 * - v0 > 0: Initial variance must be positive
 *
 * Additionally, the Feller condition (2κθ ≥ σ²) ensures variance stays positive.
 */
struct HestonParameters {
    double kappa;  ///< Mean-reversion speed of variance
    double theta;  ///< Long-term variance mean
    double sigma;  ///< Volatility of variance ("vol of vol")
    double rho;    ///< Correlation between asset returns and variance (typically < 0)
    double v0;     ///< Initial variance

    /// Default constructor with typical equity market parameters
    HestonParameters()
        : kappa(2.0), theta(0.04), sigma(0.3), rho(-0.7), v0(0.04) {}

    /// Parameterized constructor
    HestonParameters(double kappa_, double theta_, double sigma_, double rho_, double v0_)
        : kappa(kappa_), theta(theta_), sigma(sigma_), rho(rho_), v0(v0_) {}

    /**
     * @brief Check Feller condition: 2κθ ≥ σ²
     *
     * The Feller condition ensures that the variance process cannot reach zero,
     * which is important for numerical stability and theoretical soundness.
     *
     * @return true if Feller condition is satisfied
     */
    bool is_feller_satisfied() const noexcept {
        return 2.0 * kappa * theta >= sigma * sigma;
    }

    /**
     * @brief Validate all parameters are in acceptable ranges
     * @return true if all parameters are valid
     */
    bool is_valid() const noexcept {
        return kappa > 0.0 && theta > 0.0 && sigma > 0.0 && std::abs(rho) < 1.0 && v0 > 0.0;
    }

    /**
     * @brief Validate parameters and throw if invalid
     * @throws std::invalid_argument if any parameter is invalid
     */
    void validate() const {
        if (kappa <= 0.0) {
            throw std::invalid_argument("Heston: kappa must be positive, got " +
                                        std::to_string(kappa));
        }
        if (theta <= 0.0) {
            throw std::invalid_argument("Heston: theta must be positive, got " +
                                        std::to_string(theta));
        }
        if (sigma <= 0.0) {
            throw std::invalid_argument("Heston: sigma must be positive, got " +
                                        std::to_string(sigma));
        }
        if (std::abs(rho) >= 1.0) {
            throw std::invalid_argument("Heston: |rho| must be < 1, got " + std::to_string(rho));
        }
        if (v0 <= 0.0) {
            throw std::invalid_argument("Heston: v0 must be positive, got " + std::to_string(v0));
        }
    }

    /// String representation for logging and debugging
    std::string to_string() const {
        return "HestonParameters(kappa=" + std::to_string(kappa) + ", theta=" +
               std::to_string(theta) + ", sigma=" + std::to_string(sigma) +
               ", rho=" + std::to_string(rho) + ", v0=" + std::to_string(v0) +
               ", feller=" + (is_feller_satisfied() ? "OK" : "VIOLATED") + ")";
    }
};

/**
 * @brief Option Greeks structure
 */
struct OptionGreeks {
    double delta;  ///< ∂V/∂S
    double gamma;  ///< ∂²V/∂S²
    double vega;   ///< ∂V/∂σ (using √v0 as proxy)
    double theta;  ///< -∂V/∂T
    double rho;    ///< ∂V/∂r
};

/**
 * @brief Option pricing result with price and optional Greeks
 */
struct PricingResult {
    double price;
    OptionGreeks greeks;
    bool greeks_computed = false;
};

/**
 * @brief Heston stochastic volatility model
 *
 * Implements characteristic function-based option pricing using the
 * Carr-Madan (1999) FFT approach for efficient computation.
 *
 * Key features:
 * - Closed-form characteristic function (Heston 1993, Equation 17)
 * - FFT-based option pricing for vectorized strike computation
 * - Numerical integration via Gauss-Laguerre quadrature for single options
 */
class HestonModel {
public:
    /**
     * @brief Construct Heston model with given parameters
     * @param params Model parameters
     * @throws std::invalid_argument if parameters are invalid
     */
    explicit HestonModel(const HestonParameters& params);

    /// Get current parameters
    const HestonParameters& parameters() const noexcept { return params_; }

    /// Update parameters
    void set_parameters(const HestonParameters& params);

    /**
     * @brief Heston characteristic function φ(u, T)
     *
     * Implements Equation 17 from Heston (1993):
     *   φ(u) = exp(C(u,T) + D(u,T)v₀ + iu·log(S₀))
     *
     * where:
     *   d = √[(ρσiu - κ)² + σ²(iu + u²)]
     *   g = (κ - ρσiu - d) / (κ - ρσiu + d)
     *   C = (r-q)iuT + (κθ/σ²)[(κ - ρσiu - d)T - 2log((1-ge^(-dT))/(1-g))]
     *   D = [(κ - ρσiu - d)/σ²][(1-e^(-dT))/(1-ge^(-dT))]
     *
     * @param u Complex frequency variable
     * @param T Time to maturity (years)
     * @param S0 Current spot price
     * @param r Risk-free rate (continuous)
     * @param q Dividend yield (continuous)
     * @return Complex value of characteristic function
     */
    std::complex<double> characteristic_function(std::complex<double> u, double T, double S0,
                                                 double r, double q) const;

    /**
     * @brief Price a European option using numerical integration
     *
     * Uses the Carr-Madan (1999) approach with Gauss-Laguerre quadrature
     * for single option pricing.
     *
     * @param strike Strike price K
     * @param maturity Time to maturity T (years)
     * @param spot Current spot price S0
     * @param rate Risk-free rate r
     * @param dividend Dividend yield q
     * @param is_call true for call option, false for put
     * @return Option price
     */
    double price_option(double strike, double maturity, double spot, double rate, double dividend,
                        bool is_call = true) const;

    /**
     * @brief Price option with Greeks computation
     *
     * Computes price and Greeks using finite difference approximations.
     *
     * @param strike Strike price K
     * @param maturity Time to maturity T (years)
     * @param spot Current spot price S0
     * @param rate Risk-free rate r
     * @param dividend Dividend yield q
     * @param is_call true for call option, false for put
     * @return PricingResult with price and Greeks
     */
    PricingResult price_option_with_greeks(double strike, double maturity, double spot, double rate,
                                           double dividend, bool is_call = true) const;

    /**
     * @brief Price multiple options (vectorized)
     *
     * More efficient than repeated calls to price_option when
     * pricing many options with the same underlying parameters.
     *
     * @param strikes Vector of strike prices
     * @param maturities Vector of maturities (must match strikes size or be single value)
     * @param spot Current spot price
     * @param rate Risk-free rate
     * @param dividend Dividend yield
     * @param is_call true for calls, false for puts
     * @return Vector of option prices
     */
    std::vector<double> price_options(const std::vector<double>& strikes,
                                      const std::vector<double>& maturities, double spot,
                                      double rate, double dividend, bool is_call = true) const;

#ifdef QUANT_USE_EIGEN
    /**
     * @brief Price multiple options using Eigen vectors
     */
    Eigen::VectorXd price_options_eigen(const Eigen::VectorXd& strikes,
                                        const Eigen::VectorXd& maturities, double spot, double rate,
                                        double dividend, bool is_call = true) const;
#endif

    /**
     * @brief Compute implied volatility from Heston model price
     *
     * Uses Newton-Raphson iteration to find Black-Scholes implied vol
     * that matches the Heston price.
     *
     * @param strike Strike price
     * @param maturity Time to maturity
     * @param spot Current spot price
     * @param rate Risk-free rate
     * @param dividend Dividend yield
     * @param is_call true for call, false for put
     * @return Implied volatility
     */
    double implied_volatility(double strike, double maturity, double spot, double rate,
                              double dividend, bool is_call = true) const;

private:
    HestonParameters params_;

    /// Integration parameters for Gauss-Laguerre quadrature
    static constexpr int NUM_INTEGRATION_POINTS = 64;
    static constexpr double INTEGRATION_ALPHA = 0.75;  // Damping parameter for Carr-Madan

    /// Intermediate values for characteristic function
    struct CFIntermediates {
        std::complex<double> d;
        std::complex<double> g;
        std::complex<double> C;
        std::complex<double> D;
    };

    /// Compute intermediate values for characteristic function
    CFIntermediates compute_cf_intermediates(std::complex<double> u, double T) const;

    /// Price single option using numerical integration
    double price_option_integration(double strike, double maturity, double spot, double rate,
                                    double dividend, bool is_call) const;

    /// Black-Scholes price for implied vol calculation
    static double black_scholes_price(double spot, double strike, double rate, double dividend,
                                      double maturity, double vol, bool is_call);

    /// Black-Scholes vega for implied vol Newton-Raphson
    static double black_scholes_vega(double spot, double strike, double rate, double dividend,
                                     double maturity, double vol);
};

}  // namespace quant::models

#endif  // QUANT_TRADING_HESTON_HPP
