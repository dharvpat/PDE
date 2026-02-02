"""
Ornstein-Uhlenbeck mean-reverting process (Python wrapper over C++).

This module provides a Pythonic interface to the high-performance
C++ OU process implementation.

Reference:
    Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with
    Transaction Costs and Stop-Loss Exit." Journal of Industrial and
    Management Optimization.

Example:
    >>> from quant_trading.models import OUProcess, OUParameters
    >>> # Fit parameters from price data
    >>> result = OUProcess.fit_mle(prices, dt=1.0/252.0)
    >>> print(f"Half-life: {result.params.half_life():.1f} days")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math

import numpy as np

# Try to import C++ bindings
try:
    from ..cpp import quant_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


@dataclass
class OUParameters:
    """
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
    """
    theta: float
    mu: float
    sigma: float

    def half_life(self) -> float:
        """
        Half-life of mean reversion (in same units as time).

        The half-life is the time it takes for the expected value to move
        halfway from its current position to the long-term mean.

        Returns:
            Half-life in time units, or infinity if mu <= 0
        """
        if self.mu <= 0:
            return float("inf")
        return math.log(2) / self.mu

    def is_mean_reverting(self) -> bool:
        """
        Check if process is mean-reverting.

        The process is mean-reverting if μ > 0.
        """
        return self.mu > 0

    def stationary_variance(self) -> float:
        """
        Stationary (long-run) variance of the process.

        For a stationary OU process, the variance converges to:
            Var_∞ = σ² / (2μ)

        Returns:
            Stationary variance, or infinity if not mean-reverting
        """
        if self.mu <= 0:
            return float("inf")
        return (self.sigma ** 2) / (2 * self.mu)

    def stationary_std(self) -> float:
        """Stationary (long-run) standard deviation."""
        return math.sqrt(self.stationary_variance())

    def is_valid(self) -> bool:
        """Check if parameters are valid (sigma > 0)."""
        return self.sigma > 0

    def validate(self) -> None:
        """Validate parameters and raise ValueError if invalid."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
        }


@dataclass
class OUFitResult:
    """
    MLE fitting result for Ornstein-Uhlenbeck process.

    Attributes:
        params: Estimated OUParameters
        log_likelihood: Log-likelihood at optimum
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        n_observations: Number of observations used
        converged: Whether optimization converged
        message: Additional information or error message
    """
    params: OUParameters
    log_likelihood: float
    aic: float
    bic: float
    n_observations: int
    converged: bool
    message: str = ""


class OUProcess:
    """
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
    """

    @staticmethod
    def _check_cpp() -> None:
        """Check if C++ extensions are available."""
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ extensions not available. "
                "Please build the package with: pip install -e ."
            )

    @staticmethod
    def fit_mle(
        prices: Union[List[float], np.ndarray],
        dt: float = 1.0 / 252.0,
    ) -> OUFitResult:
        """
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
            >>> # Simulate some mean-reverting data
            >>> prices = list(np.cumsum(np.random.randn(252)))
            >>> result = OUProcess.fit_mle(prices)
            >>> print(f"Estimated half-life: {result.params.half_life():.1f}")
        """
        OUProcess._check_cpp()

        prices_list = list(prices)
        cpp_result = quant_cpp.ou.OUProcess.fit_mle(prices_list, dt)

        params = OUParameters(
            theta=cpp_result.params.theta,
            mu=cpp_result.params.mu,
            sigma=cpp_result.params.sigma,
        )

        return OUFitResult(
            params=params,
            log_likelihood=cpp_result.log_likelihood,
            aic=cpp_result.aic,
            bic=cpp_result.bic,
            n_observations=cpp_result.n_observations,
            converged=cpp_result.converged,
            message=cpp_result.message,
        )

    @staticmethod
    def log_likelihood(
        prices: Union[List[float], np.ndarray],
        params: OUParameters,
        dt: float = 1.0 / 252.0,
    ) -> float:
        """
        Compute log-likelihood of observed data under OU model.

        The log-likelihood is:
            LL = -n/2 * log(2π) - n/2 * log(σ²_ε) - (1/2σ²_ε) Σ(X_{t+1} - μ_t)²

        Args:
            prices: Observed time series
            params: OU parameters
            dt: Time increment

        Returns:
            Log-likelihood value
        """
        OUProcess._check_cpp()

        prices_list = list(prices)
        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        return quant_cpp.ou.OUProcess.log_likelihood(prices_list, cpp_params, dt)

    @staticmethod
    def conditional_mean(
        x_t: float,
        params: OUParameters,
        dt: float,
    ) -> float:
        """
        Conditional mean of X_{t+dt} given X_t.

        E[X_{t+dt} | X_t] = θ + (X_t - θ) * e^{-μdt}

        Args:
            x_t: Current value
            params: OU parameters
            dt: Time increment

        Returns:
            Expected value at time t+dt
        """
        OUProcess._check_cpp()

        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        return quant_cpp.ou.OUProcess.conditional_mean(x_t, cpp_params, dt)

    @staticmethod
    def conditional_variance(
        params: OUParameters,
        dt: float,
    ) -> float:
        """
        Conditional variance of X_{t+dt} given X_t.

        Var[X_{t+dt} | X_t] = σ²(1 - e^{-2μdt}) / (2μ)

        Args:
            params: OU parameters
            dt: Time increment

        Returns:
            Conditional variance
        """
        OUProcess._check_cpp()

        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        return quant_cpp.ou.OUProcess.conditional_variance(cpp_params, dt)

    @staticmethod
    def transition_density(
        x_next: float,
        x_t: float,
        params: OUParameters,
        dt: float,
    ) -> float:
        """
        Transition density p(X_{t+dt} | X_t).

        The transition density is Gaussian with conditional mean and variance.

        Args:
            x_next: Value at time t+dt
            x_t: Value at time t
            params: OU parameters
            dt: Time increment

        Returns:
            Probability density
        """
        OUProcess._check_cpp()

        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        return quant_cpp.ou.OUProcess.transition_density(x_next, x_t, cpp_params, dt)

    @staticmethod
    def simulate(
        params: OUParameters,
        x0: float,
        T: float,
        n_steps: int,
        seed: int = 42,
    ) -> np.ndarray:
        """
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
            NumPy array with simulated path (n_steps + 1 values including x0)
        """
        OUProcess._check_cpp()

        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        path = quant_cpp.ou.OUProcess.simulate(cpp_params, x0, T, n_steps, seed)
        return np.array(path)

    @staticmethod
    def optimal_boundaries(
        params: OUParameters,
        transaction_cost: float,
        risk_free_rate: float,
    ) -> Tuple[float, float, float]:
        """
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
            >>> params = OUParameters(theta=100.0, mu=5.0, sigma=2.0)
            >>> lower, upper, exit = OUProcess.optimal_boundaries(
            ...     params, transaction_cost=0.001, risk_free_rate=0.05
            ... )
            >>> print(f"Enter long below {lower:.2f}, short above {upper:.2f}")
        """
        OUProcess._check_cpp()

        cpp_params = quant_cpp.ou.OUParameters(params.theta, params.mu, params.sigma)
        return quant_cpp.ou.OUProcess.optimal_boundaries(
            cpp_params, transaction_cost, risk_free_rate
        )

    @staticmethod
    def generate_trading_signals(
        prices: Union[List[float], np.ndarray],
        params: OUParameters,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.05,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Generate trading signals based on optimal boundaries.

        Args:
            prices: Price series
            params: OU parameters
            transaction_cost: Round-trip transaction cost
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary with:
                - 'signals': Array of signals (-1 = short, 0 = neutral, 1 = long)
                - 'entry_lower': Lower entry boundary
                - 'entry_upper': Upper entry boundary
                - 'exit_target': Exit target
        """
        prices = np.asarray(prices)
        lower, upper, exit_target = OUProcess.optimal_boundaries(
            params, transaction_cost, risk_free_rate
        )

        signals = np.zeros(len(prices))
        position = 0  # Current position

        for i, price in enumerate(prices):
            if position == 0:  # No position
                if price < lower:
                    position = 1  # Go long
                elif price > upper:
                    position = -1  # Go short
            elif position == 1:  # Long position
                if price >= exit_target:
                    position = 0  # Exit
            elif position == -1:  # Short position
                if price <= exit_target:
                    position = 0  # Exit

            signals[i] = position

        return {
            "signals": signals,
            "entry_lower": lower,
            "entry_upper": upper,
            "exit_target": exit_target,
        }
