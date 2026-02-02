/**
 * @file quant_cpp.cpp
 * @brief Main pybind11 module combining all C++ bindings
 *
 * This creates the 'quant_cpp' Python extension module that provides
 * high-performance C++ implementations of quantitative finance models.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of binding functions
void bind_heston(py::module_& m);
void bind_sabr(py::module_& m);
void bind_ou_process(py::module_& m);
void init_pde_bindings(py::module_& m);

/**
 * @brief Main Python module definition
 *
 * Creates submodules for each model category:
 * - heston: Heston stochastic volatility model
 * - sabr: SABR volatility model
 * - ou: Ornstein-Uhlenbeck process
 */
PYBIND11_MODULE(quant_cpp, m) {
    m.doc() = R"doc(
        Quantitative Trading C++ Extension Module

        High-performance C++ implementations of quantitative finance models,
        accessible from Python with zero-copy NumPy array support.

        Submodules:
            heston: Heston stochastic volatility model for option pricing
            sabr: SABR volatility model for smile calibration
            ou: Ornstein-Uhlenbeck mean-reverting process
            solvers: PDE solvers for option pricing (Black-Scholes, Heston, HJB)

        Example:
            >>> from quant_trading.cpp import quant_cpp
            >>> # Heston option pricing
            >>> params = quant_cpp.heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> model = quant_cpp.heston.HestonModel(params)
            >>> price = model.price_option(100.0, 1.0, 100.0, 0.05, 0.02, True)

            >>> # SABR implied volatility
            >>> sabr = quant_cpp.sabr.SABRModel(beta=0.5)
            >>> vol = sabr.implied_volatility(105.0, 100.0, 1.0, 0.2, -0.3, 0.4)

            >>> # OU process fitting
            >>> result = quant_cpp.ou.OUProcess.fit_mle(prices, dt=1.0/252.0)

        Performance:
            C++ implementations provide 10-100x speedups compared to pure Python
            for computationally intensive operations like:
            - Heston characteristic function evaluation
            - FFT-based option pricing
            - Monte Carlo simulation
            - MLE parameter estimation

        References:
            - Heston (1993): "A closed-form solution for options with stochastic volatility"
            - Hagan et al. (2002): "Managing smile risk"
            - Leung & Li (2015): "Optimal Mean Reversion Trading"
    )doc";

    // Create submodules
    py::module_ heston_module = m.def_submodule("heston",
        R"doc(
        Heston stochastic volatility model.

        Implements characteristic function-based option pricing using the
        Carr-Madan (1999) FFT approach.

        Classes:
            HestonParameters: Model parameters (kappa, theta, sigma, rho, v0)
            HestonModel: Option pricing and Greeks computation
            OptionGreeks: Delta, gamma, vega, theta, rho
            PricingResult: Price with optional Greeks

        Example:
            >>> from quant_trading.cpp.quant_cpp import heston
            >>> params = heston.HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
            >>> model = heston.HestonModel(params)
            >>> price = model.price_option(strike=100, maturity=1.0, spot=100,
            ...                            rate=0.05, dividend=0.02, is_call=True)
        )doc");

    py::module_ sabr_module = m.def_submodule("sabr",
        R"doc(
        SABR volatility model.

        Implements the Hagan et al. (2002) asymptotic formula for fast
        implied volatility computation.

        Classes:
            SABRParameters: Model parameters (alpha, beta, rho, nu)
            SABRModel: Implied volatility calculation

        Example:
            >>> from quant_trading.cpp.quant_cpp import sabr
            >>> model = sabr.SABRModel(beta=0.5)
            >>> vol = model.implied_volatility(
            ...     strike=105, forward=100, maturity=1.0,
            ...     alpha=0.2, rho=-0.3, nu=0.4
            ... )
        )doc");

    py::module_ ou_module = m.def_submodule("ou",
        R"doc(
        Ornstein-Uhlenbeck mean-reverting process.

        Provides MLE parameter estimation, simulation, and optimal
        trading boundary computation.

        Classes:
            OUParameters: Process parameters (theta, mu, sigma)
            OUFitResult: MLE fitting results
            OUProcess: Static methods for fitting, simulation, etc.

        Example:
            >>> from quant_trading.cpp.quant_cpp import ou
            >>> # Fit parameters from price data
            >>> result = ou.OUProcess.fit_mle(prices, dt=1.0/252.0)
            >>> print(f"Half-life: {result.params.half_life():.1f} days")
            >>>
            >>> # Compute optimal trading boundaries
            >>> lower, upper, exit = ou.OUProcess.optimal_boundaries(
            ...     result.params, transaction_cost=0.001, risk_free_rate=0.05
            ... )
        )doc");

    // Bind all models to their respective submodules
    bind_heston(heston_module);
    bind_sabr(sabr_module);
    bind_ou_process(ou_module);

    // Bind PDE solvers (creates solvers submodule)
    init_pde_bindings(m);

    // Add version information
    m.attr("__version__") = "0.1.0";
}
