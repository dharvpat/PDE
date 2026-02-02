/**
 * @file ou_bindings.cpp
 * @brief pybind11 bindings for Ornstein-Uhlenbeck process
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "models/ou_process.hpp"

namespace py = pybind11;
using namespace quant::models;

void bind_ou_process(py::module_& m) {
    // ============== OUParameters struct ==============
    py::class_<OUParameters>(m, "OUParameters",
        R"doc(
        Ornstein-Uhlenbeck process parameters.

        The OU process is defined by the SDE:
            dX_t = μ(θ - X_t)dt + σ dB_t

        Key properties:
            - Mean-reverting to long-term level θ
            - Mean-reversion speed controlled by μ
            - Half-life of reversion: t_half = ln(2)/μ

        Attributes:
            theta: Long-term mean (equilibrium level)
            mu: Mean-reversion speed (> 0 for mean-reverting)
            sigma: Instantaneous volatility

        Reference:
            Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with
            Transaction Costs and Stop-Loss Exit."
        )doc")
        .def(py::init<>(),
             "Construct with default parameters")
        .def(py::init<double, double, double>(),
             py::arg("theta"),
             py::arg("mu"),
             py::arg("sigma"),
             "Construct with specified parameters")
        .def_readwrite("theta", &OUParameters::theta,
                       "Long-term mean (equilibrium level)")
        .def_readwrite("mu", &OUParameters::mu,
                       "Mean-reversion speed")
        .def_readwrite("sigma", &OUParameters::sigma,
                       "Instantaneous volatility")
        .def("half_life", &OUParameters::half_life,
             R"doc(
             Half-life of mean reversion (in same units as time).

             The half-life is the time it takes for the expected value to move
             halfway from its current position to the long-term mean.

             t_half = ln(2) / μ

             Returns:
                 Half-life in time units, or infinity if mu <= 0
             )doc")
        .def("is_mean_reverting", &OUParameters::is_mean_reverting,
             R"doc(
             Check if process is mean-reverting.

             The process is mean-reverting if μ > 0.
             If μ <= 0, the process is either non-stationary (μ = 0, random walk)
             or explosive (μ < 0).

             Returns:
                 True if process is mean-reverting
             )doc")
        .def("stationary_variance", &OUParameters::stationary_variance,
             R"doc(
             Stationary (long-run) variance of the process.

             For a stationary OU process, the variance converges to:
                 Var_∞ = σ² / (2μ)

             Returns:
                 Stationary variance, or infinity if not mean-reverting
             )doc")
        .def("stationary_std", &OUParameters::stationary_std,
             "Stationary (long-run) standard deviation")
        .def("is_valid", &OUParameters::is_valid,
             "Check if parameters are valid (sigma > 0)")
        .def("validate", &OUParameters::validate,
             "Validate parameters and raise ValueError if invalid")
        .def("__repr__", &OUParameters::to_string);

    // ============== OUFitResult struct ==============
    py::class_<OUFitResult>(m, "OUFitResult",
        R"doc(
        MLE fitting result for Ornstein-Uhlenbeck process.

        Attributes:
            params: Estimated OUParameters
            log_likelihood: Log-likelihood at optimum
            aic: Akaike Information Criterion
            bic: Bayesian Information Criterion
            n_observations: Number of observations used
            converged: Whether optimization converged
            message: Additional information or error message
        )doc")
        .def(py::init<>())
        .def_readwrite("params", &OUFitResult::params)
        .def_readwrite("log_likelihood", &OUFitResult::log_likelihood)
        .def_readwrite("aic", &OUFitResult::aic)
        .def_readwrite("bic", &OUFitResult::bic)
        .def_readwrite("n_observations", &OUFitResult::n_observations)
        .def_readwrite("converged", &OUFitResult::converged)
        .def_readwrite("message", &OUFitResult::message)
        .def("__repr__", [](const OUFitResult& r) {
            return "OUFitResult(converged=" + std::string(r.converged ? "True" : "False") +
                   ", params=" + r.params.to_string() +
                   ", log_likelihood=" + std::to_string(r.log_likelihood) + ")";
        });

    // ============== OUProcess class (static methods) ==============
    py::class_<OUProcess>(m, "OUProcess",
        R"doc(
        Ornstein-Uhlenbeck process utilities.

        Provides:
            - MLE parameter estimation from time series
            - Log-likelihood computation
            - Simulation of OU paths
            - Transition density calculations
            - Optimal trading boundaries

        All methods are static - no instance creation needed.

        Example:
            >>> prices = [100.0, 100.5, 99.8, 100.2, ...]
            >>> result = OUProcess.fit_mle(prices, dt=1.0/252.0)
            >>> print(f"Half-life: {result.params.half_life():.2f} days")
        )doc")
        .def_static("fit_mle",
                    py::overload_cast<const std::vector<double>&, double>(&OUProcess::fit_mle),
                    py::arg("prices"),
                    py::arg("dt") = 1.0 / 252.0,
                    R"doc(
                    Fit OU process to time series using Maximum Likelihood Estimation.

                    For equally-spaced observations, the OU process has an exact discrete-time
                    representation as an AR(1) process, enabling closed-form MLE.

                    Args:
                        prices: List of observed values (e.g., log prices or spreads)
                        dt: Time increment between observations (default 1/252 for daily)

                    Returns:
                        OUFitResult with estimated parameters and diagnostics

                    Example:
                        >>> import numpy as np
                        >>> prices = list(np.cumsum(np.random.randn(252)))
                        >>> result = OUProcess.fit_mle(prices)
                        >>> print(result.params)
                    )doc")
        .def_static("log_likelihood",
                    py::overload_cast<const std::vector<double>&, const OUParameters&, double>(
                        &OUProcess::log_likelihood),
                    py::arg("prices"),
                    py::arg("params"),
                    py::arg("dt") = 1.0 / 252.0,
                    R"doc(
                    Compute log-likelihood of observed data under OU model.

                    The log-likelihood is:
                        LL = -n/2 * log(2π) - n/2 * log(σ²_ε) - (1/2σ²_ε) Σ(X_{t+1} - μ_t)²

                    Args:
                        prices: Observed time series
                        params: OU parameters
                        dt: Time increment

                    Returns:
                        Log-likelihood value
                    )doc")
        .def_static("conditional_mean", &OUProcess::conditional_mean,
                    py::arg("x_t"),
                    py::arg("params"),
                    py::arg("dt"),
                    R"doc(
                    Conditional mean of X_{t+dt} given X_t.

                    E[X_{t+dt} | X_t] = θ + (X_t - θ) * e^{-μdt}

                    Args:
                        x_t: Current value
                        params: OU parameters
                        dt: Time increment

                    Returns:
                        Expected value at time t+dt
                    )doc")
        .def_static("conditional_variance", &OUProcess::conditional_variance,
                    py::arg("params"),
                    py::arg("dt"),
                    R"doc(
                    Conditional variance of X_{t+dt} given X_t.

                    Var[X_{t+dt} | X_t] = σ²(1 - e^{-2μdt}) / (2μ)

                    Args:
                        params: OU parameters
                        dt: Time increment

                    Returns:
                        Conditional variance
                    )doc")
        .def_static("transition_density", &OUProcess::transition_density,
                    py::arg("x_next"),
                    py::arg("x_t"),
                    py::arg("params"),
                    py::arg("dt"),
                    R"doc(
                    Transition density p(X_{t+dt} | X_t).

                    The transition density is Gaussian with conditional mean and variance.

                    Args:
                        x_next: Value at time t+dt
                        x_t: Value at time t
                        params: OU parameters
                        dt: Time increment

                    Returns:
                        Probability density
                    )doc")
        .def_static("simulate", &OUProcess::simulate,
                    py::arg("params"),
                    py::arg("x0"),
                    py::arg("T"),
                    py::arg("n_steps"),
                    py::arg("seed") = 42,
                    R"doc(
                    Simulate OU process path using exact discretization.

                    Uses the exact solution:
                        X_{t+dt} = θ + (X_t - θ)e^{-μdt} + σ√((1-e^{-2μdt})/(2μ)) * Z

                    where Z ~ N(0,1).

                    Args:
                        params: OU parameters
                        x0: Initial value
                        T: Total time horizon
                        n_steps: Number of time steps
                        seed: Random seed for reproducibility

                    Returns:
                        Simulated path (n_steps + 1 values including x0)
                    )doc")
        .def_static("optimal_boundaries", &OUProcess::optimal_boundaries,
                    py::arg("params"),
                    py::arg("transaction_cost"),
                    py::arg("risk_free_rate"),
                    R"doc(
                    Compute optimal trading boundaries for mean-reversion strategy.

                    Based on Leung & Li (2015), computes optimal entry and exit thresholds
                    for a mean-reversion trading strategy.

                    Args:
                        params: OU parameters
                        transaction_cost: Round-trip transaction cost (as fraction)
                        risk_free_rate: Risk-free rate

                    Returns:
                        Tuple of (entry_lower, entry_upper, exit_target)

                    Example:
                        >>> params = OUParameters(100.0, 5.0, 2.0)
                        >>> lower, upper, exit = OUProcess.optimal_boundaries(
                        ...     params, transaction_cost=0.001, risk_free_rate=0.05
                        ... )
                        >>> print(f"Enter long below {lower:.2f}, short above {upper:.2f}")
                    )doc");
}
