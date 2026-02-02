/**
 * @file heston_bindings.cpp
 * @brief pybind11 bindings for Heston stochastic volatility model
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include "models/heston.hpp"

namespace py = pybind11;
using namespace quant::models;

void bind_heston(py::module_& m) {
    // ============== Greeks struct ==============
    py::class_<OptionGreeks>(m, "OptionGreeks",
        R"doc(
        Option Greeks (sensitivities).

        Attributes:
            delta: Sensitivity to underlying price (dV/dS)
            gamma: Second derivative to underlying price (d²V/dS²)
            vega: Sensitivity to volatility (dV/dσ)
            theta: Time decay (-dV/dT)
            rho: Sensitivity to interest rate (dV/dr)
        )doc")
        .def(py::init<>())
        .def_readwrite("delta", &OptionGreeks::delta)
        .def_readwrite("gamma", &OptionGreeks::gamma)
        .def_readwrite("vega", &OptionGreeks::vega)
        .def_readwrite("theta", &OptionGreeks::theta)
        .def_readwrite("rho", &OptionGreeks::rho)
        .def("__repr__", [](const OptionGreeks& g) {
            return "OptionGreeks(delta=" + std::to_string(g.delta) +
                   ", gamma=" + std::to_string(g.gamma) +
                   ", vega=" + std::to_string(g.vega) +
                   ", theta=" + std::to_string(g.theta) +
                   ", rho=" + std::to_string(g.rho) + ")";
        });

    // ============== PricingResult struct ==============
    py::class_<PricingResult>(m, "PricingResult",
        R"doc(
        Option pricing result with price and optional Greeks.

        Attributes:
            price: Option price
            greeks: OptionGreeks object with sensitivities
            greeks_computed: Whether Greeks were computed
        )doc")
        .def(py::init<>())
        .def_readwrite("price", &PricingResult::price)
        .def_readwrite("greeks", &PricingResult::greeks)
        .def_readwrite("greeks_computed", &PricingResult::greeks_computed);

    // ============== HestonParameters struct ==============
    py::class_<HestonParameters>(m, "HestonParameters",
        R"doc(
        Heston model parameters.

        The Heston model describes stochastic volatility dynamics:
            dS_t = μS_t dt + √v_t S_t dW_t^S
            dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
            dW_t^S · dW_t^v = ρ dt

        Attributes:
            kappa: Mean-reversion speed of variance (must be > 0)
            theta: Long-term variance mean (must be > 0)
            sigma: Volatility of variance, "vol of vol" (must be > 0)
            rho: Correlation between asset and variance (-1 < rho < 1)
            v0: Initial variance (must be > 0)

        Reference:
            Heston, S.L. (1993). "A closed-form solution for options
            with stochastic volatility." Review of Financial Studies, 6(2).
        )doc")
        .def(py::init<>(),
             "Construct with default parameters (typical equity market)")
        .def(py::init<double, double, double, double, double>(),
             py::arg("kappa"),
             py::arg("theta"),
             py::arg("sigma"),
             py::arg("rho"),
             py::arg("v0"),
             "Construct with specified parameters")
        .def_readwrite("kappa", &HestonParameters::kappa,
                       "Mean-reversion speed of variance")
        .def_readwrite("theta", &HestonParameters::theta,
                       "Long-term variance mean")
        .def_readwrite("sigma", &HestonParameters::sigma,
                       "Volatility of variance (vol of vol)")
        .def_readwrite("rho", &HestonParameters::rho,
                       "Correlation between asset and variance")
        .def_readwrite("v0", &HestonParameters::v0,
                       "Initial variance")
        .def("is_feller_satisfied", &HestonParameters::is_feller_satisfied,
             R"doc(
             Check if Feller condition (2κθ >= σ²) is satisfied.

             The Feller condition ensures variance cannot reach zero,
             which is important for numerical stability.

             Returns:
                 True if Feller condition is satisfied
             )doc")
        .def("is_valid", &HestonParameters::is_valid,
             "Check if all parameters are in valid ranges")
        .def("validate", &HestonParameters::validate,
             "Validate parameters and raise ValueError if invalid")
        .def("__repr__", &HestonParameters::to_string);

    // ============== HestonModel class ==============
    py::class_<HestonModel>(m, "HestonModel",
        R"doc(
        Heston stochastic volatility model for option pricing.

        Implements characteristic function-based option pricing using the
        Carr-Madan (1999) FFT approach for efficient computation.

        Example:
            >>> params = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> model = HestonModel(params)
            >>> price = model.price_option(100.0, 1.0, 100.0, 0.05, 0.02, True)
        )doc")
        .def(py::init<const HestonParameters&>(),
             py::arg("params"),
             "Initialize Heston model with given parameters")
        .def("parameters", &HestonModel::parameters,
             py::return_value_policy::reference_internal,
             "Get current model parameters")
        .def("set_parameters", &HestonModel::set_parameters,
             py::arg("params"),
             "Update model parameters")
        .def("characteristic_function", &HestonModel::characteristic_function,
             py::arg("u"),
             py::arg("T"),
             py::arg("S0"),
             py::arg("r"),
             py::arg("q"),
             R"doc(
             Heston characteristic function φ(u, T).

             Implements Equation 17 from Heston (1993).

             Args:
                 u: Complex frequency variable
                 T: Time to maturity (years)
                 S0: Current spot price
                 r: Risk-free rate
                 q: Dividend yield

             Returns:
                 Complex value of characteristic function
             )doc")
        .def("price_option", &HestonModel::price_option,
             py::arg("strike"),
             py::arg("maturity"),
             py::arg("spot"),
             py::arg("rate"),
             py::arg("dividend"),
             py::arg("is_call") = true,
             R"doc(
             Price a European option using numerical integration.

             Uses the Carr-Madan (1999) approach with numerical
             integration for single option pricing.

             Args:
                 strike: Strike price K
                 maturity: Time to maturity T (years)
                 spot: Current spot price S0
                 rate: Risk-free rate r
                 dividend: Dividend yield q
                 is_call: True for call option, False for put

             Returns:
                 Option price

             Raises:
                 ValueError: If strike, spot, or maturity are non-positive
             )doc")
        .def("price_option_with_greeks", &HestonModel::price_option_with_greeks,
             py::arg("strike"),
             py::arg("maturity"),
             py::arg("spot"),
             py::arg("rate"),
             py::arg("dividend"),
             py::arg("is_call") = true,
             R"doc(
             Price option with Greeks computation.

             Computes price and Greeks using finite difference approximations.

             Args:
                 strike: Strike price K
                 maturity: Time to maturity T (years)
                 spot: Current spot price S0
                 rate: Risk-free rate r
                 dividend: Dividend yield q
                 is_call: True for call option, False for put

             Returns:
                 PricingResult with price and Greeks
             )doc")
        .def("price_options", &HestonModel::price_options,
             py::arg("strikes"),
             py::arg("maturities"),
             py::arg("spot"),
             py::arg("rate"),
             py::arg("dividend"),
             py::arg("is_call") = true,
             R"doc(
             Price multiple options (vectorized).

             More efficient than repeated calls to price_option when
             pricing many options with the same underlying parameters.

             Args:
                 strikes: List of strike prices
                 maturities: List of maturities (must match strikes size or be single value)
                 spot: Current spot price
                 rate: Risk-free rate
                 dividend: Dividend yield
                 is_call: True for calls, False for puts

             Returns:
                 List of option prices
             )doc")
        .def("implied_volatility", &HestonModel::implied_volatility,
             py::arg("strike"),
             py::arg("maturity"),
             py::arg("spot"),
             py::arg("rate"),
             py::arg("dividend"),
             py::arg("is_call") = true,
             R"doc(
             Compute Black-Scholes implied volatility from Heston price.

             Uses Newton-Raphson iteration to find the Black-Scholes implied vol
             that matches the Heston price.

             Args:
                 strike: Strike price
                 maturity: Time to maturity
                 spot: Current spot price
                 rate: Risk-free rate
                 dividend: Dividend yield
                 is_call: True for call, False for put

             Returns:
                 Implied volatility
             )doc");
}
