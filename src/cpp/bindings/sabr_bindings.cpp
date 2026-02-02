/**
 * @file sabr_bindings.cpp
 * @brief pybind11 bindings for SABR volatility model
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "models/sabr.hpp"

namespace py = pybind11;
using namespace quant::models;

void bind_sabr(py::module_& m) {
    // ============== SABRParameters struct ==============
    py::class_<SABRParameters>(m, "SABRParameters",
        R"doc(
        SABR model parameters.

        The SABR model describes forward rate dynamics:
            dF_t = σ_t F_t^β dW_t^F
            dσ_t = ν σ_t dW_t^σ
            dW_t^F · dW_t^σ = ρ dt

        Attributes:
            alpha: Initial volatility level (ATM vol for beta=1)
            beta: CEV exponent (0 = normal, 0.5 = equity, 1 = lognormal)
            rho: Correlation between forward and volatility (-1 < rho < 1)
            nu: Volatility of volatility (vol of vol)

        Reference:
            Hagan, P.S., et al. (2002). "Managing smile risk."
            Wilmott Magazine, September, 84-108.
        )doc")
        .def(py::init<>(),
             "Construct with default parameters (typical equity)")
        .def(py::init<double, double, double, double>(),
             py::arg("alpha"),
             py::arg("beta"),
             py::arg("rho"),
             py::arg("nu"),
             "Construct with specified parameters")
        .def_readwrite("alpha", &SABRParameters::alpha,
                       "Initial volatility level")
        .def_readwrite("beta", &SABRParameters::beta,
                       "CEV exponent (backbone parameter)")
        .def_readwrite("rho", &SABRParameters::rho,
                       "Correlation between forward and volatility")
        .def_readwrite("nu", &SABRParameters::nu,
                       "Volatility of volatility")
        .def("is_valid", &SABRParameters::is_valid,
             "Check if all parameters are in valid ranges")
        .def("validate", &SABRParameters::validate,
             "Validate parameters and raise ValueError if invalid")
        .def("__repr__", &SABRParameters::to_string);

    // ============== SABRModel class ==============
    py::class_<SABRModel>(m, "SABRModel",
        R"doc(
        SABR volatility model.

        Implements the Hagan et al. (2002) asymptotic formula for implied
        volatility. Provides fast, closed-form approximation accurate for
        most practical purposes.

        Key features:
            - Fast evaluation: ~100 nanoseconds per implied vol calculation
            - Accurate for strikes not too far from ATM
            - Handles special cases (ATM, beta=0, beta=1)

        Example:
            >>> model = SABRModel(beta=0.5)  # Equity
            >>> vol = model.implied_volatility(
            ...     strike=105.0, forward=100.0, maturity=1.0,
            ...     alpha=0.2, rho=-0.3, nu=0.4
            ... )
        )doc")
        .def(py::init<double>(),
             py::arg("beta") = 0.5,
             R"doc(
             Construct SABR model with fixed beta.

             Args:
                 beta: CEV exponent (default 0.5 for equity)
                       0 = normal model
                       0.5 = typical equity
                       1 = lognormal model
             )doc")
        .def_property("beta",
                      &SABRModel::beta,
                      &SABRModel::set_beta,
                      "CEV exponent (backbone parameter)")
        .def("implied_volatility",
             py::overload_cast<double, double, double, double, double, double>(
                 &SABRModel::implied_volatility, py::const_),
             py::arg("strike"),
             py::arg("forward"),
             py::arg("maturity"),
             py::arg("alpha"),
             py::arg("rho"),
             py::arg("nu"),
             R"doc(
             Compute SABR implied volatility using Hagan asymptotic formula.

             For non-ATM strikes (K != F):
                 σ_impl = α * (z/χ(z)) * correction_factor

             where:
                 z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
                 χ(z) = ln((√(1-2ρz+z²) + z - ρ) / (1-ρ))

             Reference: Hagan et al. (2002), Equation (2.17a)

             Args:
                 strike: Strike price K
                 forward: Forward price F
                 maturity: Time to maturity T (years)
                 alpha: Initial volatility α
                 rho: Correlation ρ
                 nu: Vol of vol ν

             Returns:
                 Black-Scholes implied volatility
             )doc")
        .def("implied_volatility",
             py::overload_cast<double, double, double, const SABRParameters&>(
                 &SABRModel::implied_volatility, py::const_),
             py::arg("strike"),
             py::arg("forward"),
             py::arg("maturity"),
             py::arg("params"),
             "Compute implied volatility with SABRParameters struct")
        .def("atm_volatility", &SABRModel::atm_volatility,
             py::arg("forward"),
             py::arg("maturity"),
             py::arg("alpha"),
             py::arg("rho"),
             py::arg("nu"),
             R"doc(
             Compute ATM implied volatility (simpler formula).

             At the money (K = F), the formula simplifies to:
                 σ_ATM = α / F^(1-β) * [1 + (correction_terms) * T]

             Args:
                 forward: Forward price F
                 maturity: Time to maturity T
                 alpha: Initial volatility α
                 rho: Correlation ρ
                 nu: Vol of vol ν

             Returns:
                 ATM implied volatility
             )doc")
        .def("implied_volatilities", &SABRModel::implied_volatilities,
             py::arg("strikes"),
             py::arg("forward"),
             py::arg("maturity"),
             py::arg("alpha"),
             py::arg("rho"),
             py::arg("nu"),
             R"doc(
             Compute implied volatilities for multiple strikes (vectorized).

             Args:
                 strikes: List of strike prices
                 forward: Forward price
                 maturity: Time to maturity
                 alpha: Initial volatility
                 rho: Correlation
                 nu: Vol of vol

             Returns:
                 List of implied volatilities
             )doc")
        .def("volatility_sensitivities", &SABRModel::volatility_sensitivities,
             py::arg("strike"),
             py::arg("forward"),
             py::arg("maturity"),
             py::arg("alpha"),
             py::arg("rho"),
             py::arg("nu"),
             R"doc(
             Compute volatility smile sensitivities.

             Returns partial derivatives of implied volatility with respect to
             SABR parameters, useful for calibration and risk management.

             Args:
                 strike: Strike price
                 forward: Forward price
                 maturity: Time to maturity
                 alpha: Initial volatility
                 rho: Correlation
                 nu: Vol of vol

             Returns:
                 Tuple of (d_sigma/d_alpha, d_sigma/d_rho, d_sigma/d_nu)
             )doc");
}
