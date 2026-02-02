#include "sabr.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace quant::models {

namespace {

/// Small number for numerical comparisons
constexpr double EPSILON = 1e-10;

/// Threshold for ATM detection (relative to forward)
constexpr double ATM_THRESHOLD = 1e-6;

}  // anonymous namespace

SABRModel::SABRModel(double beta) : beta_(beta) {
    if (beta < 0.0 || beta > 1.0) {
        throw std::invalid_argument("SABR: beta must be in [0, 1], got " + std::to_string(beta));
    }
}

void SABRModel::set_beta(double beta) {
    if (beta < 0.0 || beta > 1.0) {
        throw std::invalid_argument("SABR: beta must be in [0, 1], got " + std::to_string(beta));
    }
    beta_ = beta;
}

double SABRModel::chi_function(double z, double rho) const {
    // χ(z) = ln((√(1-2ρz+z²) + z - ρ) / (1-ρ))
    // Reference: Hagan et al. (2002), below Equation (2.17a)

    // Handle small z using Taylor expansion for numerical stability
    // For |z| << 1: χ(z) ≈ z + (ρ/2)z² + ((2ρ²-1)/6)z³ + ...
    if (std::abs(z) < EPSILON) {
        return z * (1.0 + 0.5 * rho * z + (2.0 * rho * rho - 1.0) / 6.0 * z * z);
    }

    double sqrt_term = std::sqrt(1.0 - 2.0 * rho * z + z * z);
    double numerator = sqrt_term + z - rho;
    double denominator = 1.0 - rho;

    // Handle edge case where rho is very close to 1
    if (std::abs(denominator) < EPSILON) {
        // When rho → 1, χ(z) → z / √(1-2z+z²) = z / |1-z|
        if (z < 1.0) {
            return z / (1.0 - z);
        } else {
            return z / (z - 1.0);
        }
    }

    // Ensure numerator is positive to avoid log of negative
    if (numerator <= 0.0) {
        numerator = EPSILON;
    }

    return std::log(numerator / denominator);
}

double SABRModel::compute_z(double strike, double forward, double alpha, double nu) const {
    // z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
    // Reference: Hagan et al. (2002), below Equation (2.17a)

    if (nu < EPSILON || alpha < EPSILON) {
        return 0.0;
    }

    double log_fk = std::log(forward / strike);
    double fk_mid = std::sqrt(forward * strike);
    double fk_power = std::pow(fk_mid, 1.0 - beta_);

    return (nu / alpha) * fk_power * log_fk;
}

double SABRModel::compute_correction_factor(double strike, double forward, double maturity,
                                            double alpha, double rho, double nu) const {
    // Correction factor from Hagan formula (second line of Equation 2.17a)
    // [1 + (correction_terms) * T]

    double fk_mid = std::sqrt(forward * strike);
    double one_minus_beta = 1.0 - beta_;
    double fk_power = std::pow(fk_mid, one_minus_beta);

    // First correction term: (1-β)² * α² / (24 * (FK)^(1-β))
    double term1 = (one_minus_beta * one_minus_beta / 24.0) * (alpha * alpha) /
                   (fk_power * fk_power);

    // Second correction term: ρβνα / (4 * (FK)^((1-β)/2))
    double term2 = (rho * beta_ * nu * alpha) / (4.0 * fk_power);

    // Third correction term: (2 - 3ρ²)ν² / 24
    double term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu;

    return 1.0 + (term1 + term2 + term3) * maturity;
}

double SABRModel::atm_volatility(double forward, double maturity, double alpha, double rho,
                                 double nu) const {
    // ATM volatility formula (K = F)
    // σ_ATM = α / F^(1-β) * [1 + correction_terms * T]
    // Reference: Hagan et al. (2002), Equation (2.18)

    // Input validation
    if (forward <= 0.0) {
        throw std::invalid_argument("SABR: forward must be positive");
    }
    if (alpha <= 0.0) {
        throw std::invalid_argument("SABR: alpha must be positive");
    }
    if (std::abs(rho) >= 1.0) {
        throw std::invalid_argument("SABR: |rho| must be < 1");
    }
    if (nu < 0.0) {
        throw std::invalid_argument("SABR: nu must be non-negative");
    }
    if (maturity < 0.0) {
        throw std::invalid_argument("SABR: maturity must be non-negative");
    }

    double one_minus_beta = 1.0 - beta_;
    double f_power = std::pow(forward, one_minus_beta);

    // Base ATM vol
    double sigma_atm = alpha / f_power;

    // Correction terms for ATM case
    // Term 1: (1-β)² * α² / (24 * F^(2(1-β)))
    double term1 = (one_minus_beta * one_minus_beta / 24.0) * alpha * alpha /
                   (f_power * f_power);

    // Term 2: ρβνα / (4 * F^(1-β))
    double term2 = (rho * beta_ * nu * alpha) / (4.0 * f_power);

    // Term 3: (2 - 3ρ²)ν² / 24
    double term3 = ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu;

    double correction = 1.0 + (term1 + term2 + term3) * maturity;

    return sigma_atm * correction;
}

double SABRModel::implied_volatility(double strike, double forward, double maturity, double alpha,
                                     double rho, double nu) const {
    // Input validation
    if (strike <= 0.0) {
        throw std::invalid_argument("SABR: strike must be positive");
    }
    if (forward <= 0.0) {
        throw std::invalid_argument("SABR: forward must be positive");
    }
    if (alpha <= 0.0) {
        throw std::invalid_argument("SABR: alpha must be positive");
    }
    if (std::abs(rho) >= 1.0) {
        throw std::invalid_argument("SABR: |rho| must be < 1");
    }
    if (nu < 0.0) {
        throw std::invalid_argument("SABR: nu must be non-negative");
    }
    if (maturity < 0.0) {
        throw std::invalid_argument("SABR: maturity must be non-negative");
    }

    // Handle zero maturity
    if (maturity < EPSILON) {
        // Return instantaneous Black-Scholes vol
        double fk_mid = std::sqrt(forward * strike);
        return alpha / std::pow(fk_mid, 1.0 - beta_);
    }

    // Check for ATM case
    double log_fk = std::log(forward / strike);
    if (std::abs(log_fk) < ATM_THRESHOLD) {
        return atm_volatility(forward, maturity, alpha, rho, nu);
    }

    // Non-ATM case: Full Hagan formula
    // Reference: Hagan et al. (2002), Equation (2.17a)

    double one_minus_beta = 1.0 - beta_;
    double fk_mid = std::sqrt(forward * strike);
    double fk_power = std::pow(fk_mid, one_minus_beta);

    // Compute z and χ(z)
    double z = compute_z(strike, forward, alpha, nu);
    double chi_z = chi_function(z, rho);

    // z/χ(z) ratio (handles z → 0 limit)
    double z_over_chi;
    if (std::abs(z) < EPSILON) {
        z_over_chi = 1.0;
    } else {
        z_over_chi = z / chi_z;
    }

    // Numerator correction for log(F/K) terms
    // [1 + (1-β)²/24 * ln²(F/K) + (1-β)⁴/1920 * ln⁴(F/K)]
    double log_fk_sq = log_fk * log_fk;
    double numerator_correction = 1.0 + (one_minus_beta * one_minus_beta / 24.0) * log_fk_sq +
                                  (std::pow(one_minus_beta, 4) / 1920.0) * log_fk_sq * log_fk_sq;

    // Denominator: (FK)^((1-β)/2) * [1 + (1-β)²/24 * ln²(F/K) + ...]
    double denominator = fk_power * numerator_correction;

    // Base implied vol without time correction
    double sigma_base = (alpha / denominator) * z_over_chi;

    // Add time-dependent correction
    double correction_factor = compute_correction_factor(strike, forward, maturity, alpha, rho, nu);

    return sigma_base * correction_factor;
}

std::vector<double> SABRModel::implied_volatilities(const std::vector<double>& strikes,
                                                    double forward, double maturity, double alpha,
                                                    double rho, double nu) const {
    std::vector<double> vols(strikes.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < strikes.size(); ++i) {
        vols[i] = implied_volatility(strikes[i], forward, maturity, alpha, rho, nu);
    }

    return vols;
}

#ifdef QUANT_USE_EIGEN
Eigen::VectorXd SABRModel::implied_volatilities_eigen(const Eigen::VectorXd& strikes,
                                                      double forward, double maturity,
                                                      double alpha, double rho, double nu) const {
    Eigen::VectorXd vols(strikes.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (Eigen::Index i = 0; i < strikes.size(); ++i) {
        vols(i) = implied_volatility(strikes(i), forward, maturity, alpha, rho, nu);
    }

    return vols;
}
#endif

std::tuple<double, double, double> SABRModel::volatility_sensitivities(double strike,
                                                                       double forward,
                                                                       double maturity,
                                                                       double alpha, double rho,
                                                                       double nu) const {
    // Compute sensitivities using finite differences
    const double eps_alpha = alpha * 0.001;
    const double eps_rho = 0.001;
    const double eps_nu = std::max(nu * 0.001, 0.0001);

    // d_sigma/d_alpha
    double vol_alpha_up = implied_volatility(strike, forward, maturity, alpha + eps_alpha, rho, nu);
    double vol_alpha_down =
        implied_volatility(strike, forward, maturity, alpha - eps_alpha, rho, nu);
    double d_alpha = (vol_alpha_up - vol_alpha_down) / (2.0 * eps_alpha);

    // d_sigma/d_rho
    double rho_up = std::min(rho + eps_rho, 0.999);
    double rho_down = std::max(rho - eps_rho, -0.999);
    double vol_rho_up = implied_volatility(strike, forward, maturity, alpha, rho_up, nu);
    double vol_rho_down = implied_volatility(strike, forward, maturity, alpha, rho_down, nu);
    double d_rho = (vol_rho_up - vol_rho_down) / (rho_up - rho_down);

    // d_sigma/d_nu
    double vol_nu_up = implied_volatility(strike, forward, maturity, alpha, rho, nu + eps_nu);
    double vol_nu_down =
        implied_volatility(strike, forward, maturity, alpha, rho, std::max(nu - eps_nu, 0.0));
    double d_nu = (vol_nu_up - vol_nu_down) / (2.0 * eps_nu);

    return std::make_tuple(d_alpha, d_rho, d_nu);
}

}  // namespace quant::models
