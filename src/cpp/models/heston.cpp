#include "heston.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace quant::models {

namespace {

/// Pi constant
constexpr double PI = 3.14159265358979323846;

/// Standard normal CDF
double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

/// Standard normal PDF
double norm_pdf(double x) {
    constexpr double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

}  // anonymous namespace

HestonModel::HestonModel(const HestonParameters& params) : params_(params) {
    params_.validate();
}

void HestonModel::set_parameters(const HestonParameters& params) {
    params.validate();
    params_ = params;
}

HestonModel::CFIntermediates HestonModel::compute_cf_intermediates(std::complex<double> u,
                                                                   double T) const {
    CFIntermediates result;

    const double kappa = params_.kappa;
    const double theta = params_.theta;
    const double sigma = params_.sigma;
    const double rho = params_.rho;

    const std::complex<double> i(0.0, 1.0);
    const std::complex<double> sigma2 = sigma * sigma;

    // d = sqrt[(ρσiu - κ)² + σ²(iu + u²)]
    // Equation (17b) in Heston (1993)
    std::complex<double> xi = kappa - rho * sigma * i * u;
    result.d = std::sqrt(xi * xi + sigma2 * (i * u + u * u));

    // g = (κ - ρσiu - d) / (κ - ρσiu + d)
    // Equation (17c) in Heston (1993)
    result.g = (xi - result.d) / (xi + result.d);

    // Exponential term for numerical stability
    std::complex<double> exp_minus_dT = std::exp(-result.d * T);

    // C = (κθ/σ²) * [(κ - ρσiu - d)T - 2log((1 - ge^(-dT))/(1 - g))]
    // Equation (17d) in Heston (1993)
    std::complex<double> term1 = (xi - result.d) * T;
    std::complex<double> term2 = -2.0 * std::log((1.0 - result.g * exp_minus_dT) / (1.0 - result.g));
    result.C = (kappa * theta / sigma2) * (term1 + term2);

    // D = [(κ - ρσiu - d)/σ²] * [(1 - e^(-dT))/(1 - ge^(-dT))]
    // Equation (17e) in Heston (1993)
    result.D = ((xi - result.d) / sigma2) * ((1.0 - exp_minus_dT) / (1.0 - result.g * exp_minus_dT));

    return result;
}

std::complex<double> HestonModel::characteristic_function(std::complex<double> u, double T,
                                                          double S0, double r, double q) const {
    // Handle edge case: T = 0
    if (T <= 0.0) {
        return std::exp(std::complex<double>(0.0, 1.0) * u * std::log(S0));
    }

    const std::complex<double> i(0.0, 1.0);

    // Compute intermediate values
    CFIntermediates cf = compute_cf_intermediates(u, T);

    // Forward drift term: (r - q) * i * u * T
    std::complex<double> drift_term = (r - q) * i * u * T;

    // Full characteristic function: exp(C + D*v0 + iu*log(S0) + drift)
    // Equation (17) in Heston (1993)
    return std::exp(cf.C + cf.D * params_.v0 + i * u * std::log(S0) + drift_term);
}

double HestonModel::price_option_integration(double strike, double maturity, double spot,
                                             double rate, double dividend, bool is_call) const {
    // Handle edge case: zero maturity
    if (maturity <= 0.0) {
        double intrinsic = is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
        return intrinsic;
    }

    // Use modified Carr-Madan integrand with damping parameter alpha
    // Reference: Carr, P., & Madan, D. (1999). "Option valuation using the fast Fourier transform"
    const double alpha = INTEGRATION_ALPHA;
    const double log_strike = std::log(strike);
    const double discount = std::exp(-rate * maturity);

    // Integration using trapezoidal rule with adaptive refinement
    auto integrand = [&](double v) -> double {
        if (v < 1e-10) return 0.0;

        std::complex<double> u(v, -(alpha + 1.0));
        std::complex<double> phi = characteristic_function(u, maturity, spot, rate, dividend);

        // Modified characteristic function for call pricing
        std::complex<double> numerator = std::exp(-std::complex<double>(0.0, 1.0) * v * log_strike);
        std::complex<double> denominator(alpha * alpha + alpha - v * v,
                                         (2.0 * alpha + 1.0) * v);

        std::complex<double> result = numerator * phi / denominator;
        return result.real();
    };

    // Numerical integration using extended trapezoidal rule
    // Use more points for better accuracy
    const int n_points = 1024;
    const double du = 0.01;  // Integration step
    double integral = 0.0;

    // Trapezoidal rule
    integral = 0.5 * integrand(0.0);
    for (int i = 1; i < n_points; ++i) {
        double v = i * du;
        integral += integrand(v);
    }
    integral *= du;

    // Call price from Carr-Madan formula
    double call_price = (std::exp(-alpha * log_strike) / PI) * discount * integral;

    // Ensure non-negative price
    call_price = std::max(call_price, 0.0);

    if (is_call) {
        return call_price;
    } else {
        // Put-call parity: P = C - S*exp(-qT) + K*exp(-rT)
        double put_price = call_price - spot * std::exp(-dividend * maturity) + strike * discount;
        return std::max(put_price, 0.0);
    }
}

double HestonModel::price_option(double strike, double maturity, double spot, double rate,
                                 double dividend, bool is_call) const {
    // Input validation
    if (strike <= 0.0) {
        throw std::invalid_argument("Strike must be positive");
    }
    if (spot <= 0.0) {
        throw std::invalid_argument("Spot must be positive");
    }
    if (maturity < 0.0) {
        throw std::invalid_argument("Maturity must be non-negative");
    }

    return price_option_integration(strike, maturity, spot, rate, dividend, is_call);
}

PricingResult HestonModel::price_option_with_greeks(double strike, double maturity, double spot,
                                                    double rate, double dividend,
                                                    bool is_call) const {
    PricingResult result;
    result.price = price_option(strike, maturity, spot, rate, dividend, is_call);

    // Compute Greeks using finite differences
    const double eps_spot = spot * 0.001;
    const double eps_rate = 0.0001;
    const double eps_time = 1.0 / 365.0;
    const double eps_vol = 0.001;

    // Delta: ∂V/∂S
    double price_up = price_option(strike, maturity, spot + eps_spot, rate, dividend, is_call);
    double price_down = price_option(strike, maturity, spot - eps_spot, rate, dividend, is_call);
    result.greeks.delta = (price_up - price_down) / (2.0 * eps_spot);

    // Gamma: ∂²V/∂S²
    result.greeks.gamma = (price_up - 2.0 * result.price + price_down) / (eps_spot * eps_spot);

    // Rho: ∂V/∂r
    double price_rate_up = price_option(strike, maturity, spot, rate + eps_rate, dividend, is_call);
    double price_rate_down =
        price_option(strike, maturity, spot, rate - eps_rate, dividend, is_call);
    result.greeks.rho = (price_rate_up - price_rate_down) / (2.0 * eps_rate);

    // Theta: -∂V/∂T
    if (maturity > eps_time) {
        double price_later = price_option(strike, maturity - eps_time, spot, rate, dividend, is_call);
        result.greeks.theta = (price_later - result.price) / eps_time;
    } else {
        result.greeks.theta = 0.0;
    }

    // Vega: ∂V/∂σ (using v0 as proxy)
    HestonParameters params_up = params_;
    HestonParameters params_down = params_;
    params_up.v0 += eps_vol;
    params_down.v0 -= eps_vol;

    HestonModel model_up(params_up);
    HestonModel model_down(params_down);
    double price_vol_up = model_up.price_option(strike, maturity, spot, rate, dividend, is_call);
    double price_vol_down =
        model_down.price_option(strike, maturity, spot, rate, dividend, is_call);
    result.greeks.vega = (price_vol_up - price_vol_down) / (2.0 * eps_vol);

    result.greeks_computed = true;
    return result;
}

std::vector<double> HestonModel::price_options(const std::vector<double>& strikes,
                                               const std::vector<double>& maturities, double spot,
                                               double rate, double dividend, bool is_call) const {
    size_t n = strikes.size();
    if (n == 0) {
        return {};
    }

    bool single_maturity = (maturities.size() == 1);
    if (!single_maturity && maturities.size() != n) {
        throw std::invalid_argument("Maturities must have size 1 or match strikes size");
    }

    std::vector<double> prices(n);

// OpenMP parallelization if available
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (size_t i = 0; i < n; ++i) {
        double T = single_maturity ? maturities[0] : maturities[i];
        prices[i] = price_option(strikes[i], T, spot, rate, dividend, is_call);
    }

    return prices;
}

#ifdef QUANT_USE_EIGEN
Eigen::VectorXd HestonModel::price_options_eigen(const Eigen::VectorXd& strikes,
                                                 const Eigen::VectorXd& maturities, double spot,
                                                 double rate, double dividend, bool is_call) const {
    Eigen::Index n = strikes.size();
    if (n == 0) {
        return Eigen::VectorXd();
    }

    bool single_maturity = (maturities.size() == 1);
    if (!single_maturity && maturities.size() != n) {
        throw std::invalid_argument("Maturities must have size 1 or match strikes size");
    }

    Eigen::VectorXd prices(n);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (Eigen::Index i = 0; i < n; ++i) {
        double T = single_maturity ? maturities(0) : maturities(i);
        prices(i) = price_option(strikes(i), T, spot, rate, dividend, is_call);
    }

    return prices;
}
#endif

double HestonModel::black_scholes_price(double spot, double strike, double rate, double dividend,
                                        double maturity, double vol, bool is_call) {
    if (maturity <= 0.0) {
        return is_call ? std::max(spot - strike, 0.0) : std::max(strike - spot, 0.0);
    }

    double forward = spot * std::exp((rate - dividend) * maturity);
    double discount = std::exp(-rate * maturity);
    double sqrt_t = std::sqrt(maturity);
    double vol_sqrt_t = vol * sqrt_t;

    double d1 = (std::log(forward / strike) + 0.5 * vol * vol * maturity) / vol_sqrt_t;
    double d2 = d1 - vol_sqrt_t;

    if (is_call) {
        return spot * std::exp(-dividend * maturity) * norm_cdf(d1) - strike * discount * norm_cdf(d2);
    } else {
        return strike * discount * norm_cdf(-d2) - spot * std::exp(-dividend * maturity) * norm_cdf(-d1);
    }
}

double HestonModel::black_scholes_vega(double spot, double strike, double rate, double dividend,
                                       double maturity, double vol) {
    if (maturity <= 0.0 || vol <= 0.0) {
        return 0.0;
    }

    double forward = spot * std::exp((rate - dividend) * maturity);
    double sqrt_t = std::sqrt(maturity);
    double vol_sqrt_t = vol * sqrt_t;

    double d1 = (std::log(forward / strike) + 0.5 * vol * vol * maturity) / vol_sqrt_t;

    return spot * std::exp(-dividend * maturity) * sqrt_t * norm_pdf(d1);
}

double HestonModel::implied_volatility(double strike, double maturity, double spot, double rate,
                                       double dividend, bool is_call) const {
    // Get Heston price
    double target_price = price_option(strike, maturity, spot, rate, dividend, is_call);

    // Handle edge cases
    if (maturity <= 0.0) {
        return 0.0;
    }

    // Newton-Raphson iteration to find implied vol
    double vol = std::sqrt(params_.v0);  // Initial guess from current variance
    const double tol = 1e-8;
    const int max_iter = 100;

    for (int iter = 0; iter < max_iter; ++iter) {
        double bs_price = black_scholes_price(spot, strike, rate, dividend, maturity, vol, is_call);
        double vega = black_scholes_vega(spot, strike, rate, dividend, maturity, vol);

        if (vega < 1e-12) {
            // Vega too small, adjust vol and continue
            vol *= 1.5;
            continue;
        }

        double diff = bs_price - target_price;
        if (std::abs(diff) < tol) {
            return vol;
        }

        double new_vol = vol - diff / vega;

        // Ensure vol stays positive and bounded
        new_vol = std::max(0.001, std::min(5.0, new_vol));
        vol = new_vol;
    }

    return vol;  // Return best estimate even if not fully converged
}

}  // namespace quant::models
