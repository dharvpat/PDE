"""
Ornstein-Uhlenbeck Process Fitter.

Implements parameter estimation and optimal trading boundary computation
for mean-reverting processes based on Leung & Li (2015).

The OU process:
    dX_t = μ(θ - X_t)dt + σ dB_t

Parameters:
    θ (theta): Long-term mean (equilibrium level)
    μ (mu): Mean-reversion speed
    σ (sigma): Instantaneous volatility

Optimal Trading:
    - Entry boundaries: [L*, U*] determined by optimal stopping
    - Exit conditions: Take-profit at target or stop-loss
    - Half-life: t_half = ln(2)/μ

Performance target: <1 second for 500 data points

Reference:
    Leung, T., & Li, X. (2015). "Optimal mean reversion trading with
    transaction costs and stop-loss exit." International Journal of
    Theoretical and Applied Finance, 18(03), 1550020.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from scipy import optimize, stats

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OUParameters:
    """Ornstein-Uhlenbeck process parameters."""

    theta: float  # Long-term mean
    mu: float  # Mean-reversion speed
    sigma: float  # Instantaneous volatility

    @property
    def half_life(self) -> float:
        """
        Calculate half-life of mean reversion.

        Time for deviation from mean to decay by half.
        t_half = ln(2) / μ
        """
        if self.mu <= 0:
            return float("inf")
        return np.log(2) / self.mu

    @property
    def stationary_variance(self) -> float:
        """
        Calculate stationary (long-run) variance.

        Var_∞ = σ² / (2μ)
        """
        if self.mu <= 0:
            return float("inf")
        return (self.sigma ** 2) / (2 * self.mu)

    @property
    def stationary_std(self) -> float:
        """Calculate stationary standard deviation."""
        return np.sqrt(self.stationary_variance)

    def __post_init__(self):
        """Validate parameters."""
        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "theta": self.theta,
            "mu": self.mu,
            "sigma": self.sigma,
            "half_life": self.half_life,
            "stationary_variance": self.stationary_variance,
        }


@dataclass
class OptimalBoundaries:
    """Optimal entry/exit boundaries for mean-reversion trading."""

    entry_lower: float  # Enter long below this
    entry_upper: float  # Enter short above this
    exit_long: float  # Exit long position
    exit_short: float  # Exit short position
    stop_loss_long: Optional[float] = None
    stop_loss_short: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            "entry_lower": self.entry_lower,
            "entry_upper": self.entry_upper,
            "exit_long": self.exit_long,
            "exit_short": self.exit_short,
            "stop_loss_long": self.stop_loss_long,
            "stop_loss_short": self.stop_loss_short,
        }


@dataclass
class OUFitResult:
    """Result from OU parameter estimation."""

    params: OUParameters
    boundaries: Optional[OptimalBoundaries]
    log_likelihood: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    n_observations: int
    fit_time: float
    success: bool
    message: str
    residual_stats: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "params": self.params.to_dict(),
            "boundaries": self.boundaries.to_dict() if self.boundaries else None,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "n_observations": self.n_observations,
            "fit_time": self.fit_time,
            "success": self.success,
            "message": self.message,
            "residual_stats": self.residual_stats,
            "timestamp": self.timestamp.isoformat(),
        }


class OUFitter:
    """
    Ornstein-Uhlenbeck process parameter estimator.

    Uses Maximum Likelihood Estimation (MLE) to fit OU parameters
    to time series data, with optional computation of optimal
    trading boundaries.

    Features:
        - MLE parameter estimation with analytical gradients
        - Half-life and stationary distribution calculation
        - Optimal entry/exit boundary computation
        - Residual diagnostics and model validation

    Example:
        >>> fitter = OUFitter()
        >>> result = fitter.fit(spread_series, dt=1/252)
        >>> print(f"Half-life: {result.params.half_life:.1f} days")
        >>> print(f"Enter long below: {result.boundaries.entry_lower:.2f}")

    Reference:
        Leung & Li (2015) "Optimal mean reversion trading"
    """

    # Default parameter bounds for numerical stability
    DEFAULT_BOUNDS = {
        "theta": (-1000.0, 1000.0),
        "mu": (0.001, 100.0),
        "sigma": (0.0001, 100.0),
    }

    def __init__(
        self,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        db_session=None,
    ):
        """
        Initialize OU fitter.

        Args:
            bounds: Parameter bounds {param_name: (lower, upper)}
            db_session: Optional database session for storing results
        """
        self.bounds = {**self.DEFAULT_BOUNDS, **(bounds or {})}
        self.db_session = db_session
        self._cached_params: Dict[str, OUParameters] = {}

        logger.info(f"Initialized OUFitter with bounds={self.bounds}")

    def _log_likelihood(
        self,
        params: Tuple[float, float, float],
        X: np.ndarray,
        dt: float,
    ) -> float:
        """
        Compute negative log-likelihood for MLE.

        For discretized OU process:
            X_{t+dt} | X_t ~ N(μ_cond, σ²_cond)

        where:
            μ_cond = θ + (X_t - θ) * exp(-μ * dt)
            σ²_cond = σ² / (2μ) * (1 - exp(-2μ * dt))

        Args:
            params: (theta, mu, sigma)
            X: Time series values
            dt: Time step

        Returns:
            Negative log-likelihood (for minimization)
        """
        theta, mu, sigma = params

        # Conditional mean and variance
        exp_mu_dt = np.exp(-mu * dt)
        mu_cond = theta + (X[:-1] - theta) * exp_mu_dt
        var_cond = (sigma ** 2) / (2 * mu) * (1 - np.exp(-2 * mu * dt))

        if var_cond <= 0:
            return 1e10  # Invalid parameters

        # Log-likelihood
        residuals = X[1:] - mu_cond
        n = len(residuals)

        ll = -0.5 * n * np.log(2 * np.pi * var_cond)
        ll -= 0.5 * np.sum(residuals ** 2) / var_cond

        return -ll  # Return negative for minimization

    def _analytical_mle(
        self,
        X: np.ndarray,
        dt: float,
    ) -> Tuple[float, float, float]:
        """
        Compute analytical MLE estimates for OU parameters.

        Uses the exact discrete-time MLE formulas.

        Args:
            X: Time series values
            dt: Time step

        Returns:
            (theta, mu, sigma) MLE estimates
        """
        n = len(X) - 1
        X_t = X[:-1]
        X_tp1 = X[1:]

        # Sums needed for MLE
        S_x = np.sum(X_t)
        S_y = np.sum(X_tp1)
        S_xx = np.sum(X_t ** 2)
        S_xy = np.sum(X_t * X_tp1)
        S_yy = np.sum(X_tp1 ** 2)

        # MLE for theta and exp(-mu*dt)
        denom = n * S_xx - S_x ** 2
        if abs(denom) < 1e-10:
            # Degenerate case
            theta = np.mean(X)
            a = 0.5
        else:
            a = (S_xy - S_x * S_y / n) / (S_xx - S_x ** 2 / n)
            a = np.clip(a, 0.001, 0.999)  # Ensure valid
            theta = (S_y - a * S_x) / (n * (1 - a))

        # Mean reversion speed
        mu = -np.log(a) / dt

        # Volatility estimate
        residuals = X_tp1 - theta - (X_t - theta) * a
        var_residual = np.var(residuals, ddof=1)
        sigma_sq = 2 * mu * var_residual / (1 - a ** 2)
        sigma = np.sqrt(max(sigma_sq, 1e-10))

        return theta, mu, sigma

    def fit(
        self,
        X: np.ndarray,
        dt: float = 1.0 / 252,
        compute_boundaries: bool = True,
        transaction_cost: float = 0.001,
        method: str = "analytical",
        pair_name: Optional[str] = None,
    ) -> OUFitResult:
        """
        Fit OU parameters to time series data.

        Args:
            X: Time series values (e.g., spread between two assets)
            dt: Time step in years (default: 1 trading day)
            compute_boundaries: Whether to compute optimal trading boundaries
            transaction_cost: Transaction cost as fraction for boundary computation
            method: Estimation method ('analytical' or 'numerical')
            pair_name: Optional pair identifier for caching

        Returns:
            OUFitResult with estimated parameters and diagnostics

        Example:
            >>> spread = price_A - beta * price_B
            >>> result = fitter.fit(spread, dt=1/252)
        """
        import time

        start_time = time.time()
        X = np.asarray(X)
        n = len(X)

        if n < 20:
            logger.warning(f"Short time series ({n} points), estimates may be unstable")

        logger.info(f"Fitting OU model to {n} observations")

        # Estimate parameters
        if method == "analytical":
            theta, mu, sigma = self._analytical_mle(X, dt)

            # Refine with numerical optimization if needed
            if mu < 0.01 or mu > 50:
                method = "numerical"

        if method == "numerical":
            # Use analytical as starting point
            theta_init, mu_init, sigma_init = self._analytical_mle(X, dt)

            result = optimize.minimize(
                self._log_likelihood,
                x0=[theta_init, mu_init, sigma_init],
                args=(X, dt),
                method="L-BFGS-B",
                bounds=[
                    self.bounds["theta"],
                    self.bounds["mu"],
                    self.bounds["sigma"],
                ],
            )
            theta, mu, sigma = result.x
            success = result.success
        else:
            success = True

        # Create parameters object
        try:
            params = OUParameters(theta=theta, mu=mu, sigma=sigma)
        except ValueError as e:
            logger.error(f"Invalid parameters estimated: {e}")
            return OUFitResult(
                params=OUParameters(theta=np.mean(X), mu=0.1, sigma=np.std(X)),
                boundaries=None,
                log_likelihood=float("-inf"),
                aic=float("inf"),
                bic=float("inf"),
                n_observations=n,
                fit_time=time.time() - start_time,
                success=False,
                message=str(e),
            )

        # Compute log-likelihood and information criteria
        neg_ll = self._log_likelihood((theta, mu, sigma), X, dt)
        log_likelihood = -neg_ll
        k = 3  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n - 1) - 2 * log_likelihood

        # Compute residuals and diagnostics
        exp_mu_dt = np.exp(-mu * dt)
        expected = theta + (X[:-1] - theta) * exp_mu_dt
        residuals = X[1:] - expected
        std_residuals = residuals / params.stationary_std

        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "skewness": float(stats.skew(residuals)),
            "kurtosis": float(stats.kurtosis(residuals)),
            "ljung_box_p": self._ljung_box_test(residuals),
        }

        # Compute optimal boundaries
        boundaries = None
        if compute_boundaries:
            boundaries = self.compute_optimal_boundaries(
                params=params,
                transaction_cost=transaction_cost,
            )

        fit_time = time.time() - start_time

        result = OUFitResult(
            params=params,
            boundaries=boundaries,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            n_observations=n,
            fit_time=fit_time,
            success=success,
            message="Fit successful" if success else "Optimization did not converge",
            residual_stats=residual_stats,
        )

        logger.info(
            f"OU fit completed in {fit_time:.3f}s: "
            f"theta={theta:.4f}, mu={mu:.4f}, sigma={sigma:.4f}, "
            f"half-life={params.half_life:.1f} days"
        )

        # Cache results
        if pair_name:
            self._cached_params[pair_name] = params

        # Store in database
        if self.db_session and pair_name:
            self._store_fit_result(pair_name, result)

        return result

    def compute_optimal_boundaries(
        self,
        params: OUParameters,
        transaction_cost: float = 0.001,
        stop_loss_mult: float = 2.0,
    ) -> OptimalBoundaries:
        """
        Compute optimal entry/exit boundaries for mean-reversion trading.

        Based on the optimal stopping framework from Leung & Li (2015).
        Uses approximate analytical solutions for the boundary problem.

        Args:
            params: OU parameters
            transaction_cost: Round-trip transaction cost as fraction
            stop_loss_mult: Stop-loss as multiple of stationary std

        Returns:
            OptimalBoundaries with entry/exit levels
        """
        theta = params.theta
        sigma_stat = params.stationary_std
        mu = params.mu

        # Approximate optimal entry boundaries
        # Entry when deviation exceeds threshold accounting for transaction cost
        # Based on approximate solution from Leung & Li (2015)
        c = transaction_cost * abs(theta) if abs(theta) > 1 else transaction_cost

        # Entry threshold: balance expected profit vs transaction cost
        # Simplified approximation for symmetric case
        entry_threshold = sigma_stat * np.sqrt(2 * c * mu / (params.sigma ** 2) + 0.5)
        entry_threshold = max(entry_threshold, 0.5 * sigma_stat)

        # Exit at mean (simplified - could use more sophisticated exit rule)
        exit_threshold = 0.1 * sigma_stat

        # Stop-loss levels
        stop_loss_threshold = stop_loss_mult * sigma_stat

        boundaries = OptimalBoundaries(
            entry_lower=theta - entry_threshold,
            entry_upper=theta + entry_threshold,
            exit_long=theta + exit_threshold,
            exit_short=theta - exit_threshold,
            stop_loss_long=theta - stop_loss_threshold,
            stop_loss_short=theta + stop_loss_threshold,
        )

        logger.debug(
            f"Computed boundaries: entry=[{boundaries.entry_lower:.2f}, "
            f"{boundaries.entry_upper:.2f}], "
            f"exit=[{boundaries.exit_short:.2f}, {boundaries.exit_long:.2f}]"
        )

        return boundaries

    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """
        Perform Ljung-Box test for autocorrelation.

        Args:
            residuals: Model residuals
            lags: Number of lags to test

        Returns:
            p-value (high = no significant autocorrelation)
        """
        n = len(residuals)
        if n < lags + 10:
            return 1.0  # Not enough data

        acf = np.correlate(residuals, residuals, mode="full")
        acf = acf[n - 1:] / acf[n - 1]  # Normalize

        # Ljung-Box statistic
        lb_stat = n * (n + 2) * np.sum(acf[1:lags + 1] ** 2 / (n - np.arange(1, lags + 1)))

        # Chi-squared p-value
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)

        return float(p_value)

    def simulate(
        self,
        params: OUParameters,
        n_steps: int,
        dt: float = 1.0 / 252,
        X0: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate OU process paths.

        Uses exact discretization:
            X_{t+dt} = θ + (X_t - θ)e^{-μdt} + σ√((1-e^{-2μdt})/(2μ)) * Z

        Args:
            params: OU parameters
            n_steps: Number of time steps
            dt: Time step
            X0: Initial value (default: theta)
            seed: Random seed

        Returns:
            Array of simulated values
        """
        if seed is not None:
            np.random.seed(seed)

        if X0 is None:
            X0 = params.theta

        X = np.zeros(n_steps + 1)
        X[0] = X0

        exp_mu_dt = np.exp(-params.mu * dt)
        std_dt = params.sigma * np.sqrt(
            (1 - np.exp(-2 * params.mu * dt)) / (2 * params.mu)
        )

        for t in range(n_steps):
            X[t + 1] = (
                params.theta
                + (X[t] - params.theta) * exp_mu_dt
                + std_dt * np.random.randn()
            )

        return X

    def test_stationarity(
        self,
        X: np.ndarray,
        significance: float = 0.05,
    ) -> Dict[str, any]:
        """
        Test for stationarity using ADF test.

        Args:
            X: Time series
            significance: Significance level

        Returns:
            Dict with test results
        """
        from scipy import stats as sp_stats

        n = len(X)

        # Simple ADF test approximation
        # Regress ΔX_t on X_{t-1}
        dX = np.diff(X)
        X_lag = X[:-1]

        # OLS regression
        n_reg = len(dX)
        X_mat = np.column_stack([np.ones(n_reg), X_lag])
        coeffs = np.linalg.lstsq(X_mat, dX, rcond=None)[0]

        rho = coeffs[1]
        residuals = dX - X_mat @ coeffs
        se = np.sqrt(np.sum(residuals ** 2) / (n_reg - 2))
        se_rho = se / np.sqrt(np.sum((X_lag - np.mean(X_lag)) ** 2))

        adf_stat = rho / se_rho

        # Approximate critical values (from MacKinnon)
        critical_values = {
            0.01: -3.43,
            0.05: -2.86,
            0.10: -2.57,
        }

        is_stationary = adf_stat < critical_values.get(significance, -2.86)

        return {
            "adf_statistic": float(adf_stat),
            "critical_value": critical_values.get(significance, -2.86),
            "is_stationary": is_stationary,
            "rho": float(rho),
        }

    def _store_fit_result(self, pair_name: str, result: OUFitResult) -> None:
        """Store fit result in database."""
        try:
            from ..database.models import ModelParameter

            param_record = ModelParameter(
                model_type="OU",
                underlying=pair_name,
                parameters=result.to_dict(),
                fit_quality=result.log_likelihood,
                calibration_date=result.timestamp.date(),
            )
            self.db_session.add(param_record)
            self.db_session.commit()
            logger.info(f"Stored OU fit for {pair_name}")

        except ImportError:
            logger.warning("Database models not available, skipping storage")
        except Exception as e:
            logger.error(f"Failed to store fit result: {e}")
            self.db_session.rollback()

    @staticmethod
    def generate_synthetic_data(
        theta: float = 0.0,
        mu: float = 5.0,
        sigma: float = 0.2,
        n_points: int = 500,
        dt: float = 1.0 / 252,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate synthetic OU process data for testing.

        Args:
            theta: Long-term mean
            mu: Mean-reversion speed
            sigma: Volatility
            n_points: Number of data points
            dt: Time step
            seed: Random seed

        Returns:
            Array of simulated values
        """
        params = OUParameters(theta=theta, mu=mu, sigma=sigma)
        fitter = OUFitter()
        return fitter.simulate(params, n_points, dt=dt, seed=seed)
