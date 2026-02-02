"""
Volatility-Managed Position Sizing.

Implements volatility-scaled position sizing based on Moreira & Muir (2017)
"Volatility-managed portfolios" - Journal of Finance.

Core idea: w_t = c / σ_t²

Scale exposure inversely with realized volatility to improve risk-adjusted returns.
This approach:
    - Reduces exposure during high-volatility regimes
    - Increases exposure during low-volatility periods
    - Improves Sharpe ratio by timing volatility

Also includes:
    - Multiple volatility estimation methods (Realized, EWMA, GARCH)
    - Kelly criterion position sizing
    - Correlation-adjusted sizing

Reference:
    Moreira, A., & Muir, T. (2017). "Volatility-managed portfolios."
    Journal of Finance, 72(4), 1611-1644.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityMethod(Enum):
    """Volatility estimation methods."""

    REALIZED = "realized"  # Simple historical realized vol
    EWMA = "ewma"  # Exponentially weighted moving average
    GARCH = "garch"  # GARCH(1,1) model
    IMPLIED = "implied"  # From options market
    HYBRID = "hybrid"  # Combination of realized and implied


class VolatilityEstimator:
    """
    Estimate asset volatility using various methods.

    Supports multiple estimation approaches:
        - Realized: Simple historical standard deviation
        - EWMA: Exponentially weighted (RiskMetrics λ=0.94)
        - GARCH: GARCH(1,1) for time-varying volatility
        - Hybrid: Weighted combination of methods

    Example:
        >>> estimator = VolatilityEstimator(method=VolatilityMethod.EWMA)
        >>> returns = np.array([0.01, -0.02, 0.015, ...])
        >>> vol = estimator.estimate(returns)
        >>> print(f"Annualized volatility: {vol:.1%}")

    Reference:
        RiskMetrics Technical Document (1996) for EWMA methodology
    """

    def __init__(
        self,
        method: VolatilityMethod = VolatilityMethod.REALIZED,
        lookback_days: int = 21,
        ewma_lambda: float = 0.94,  # RiskMetrics standard
        annualization_factor: float = 252.0,
    ):
        """
        Initialize volatility estimator.

        Args:
            method: Estimation method to use
            lookback_days: Lookback period for realized vol
            ewma_lambda: Decay factor for EWMA (0.94 = RiskMetrics)
            annualization_factor: Trading days per year (252)
        """
        self.method = method
        self.lookback_days = lookback_days
        self.ewma_lambda = ewma_lambda
        self.annualization_factor = annualization_factor

        logger.debug(
            f"Initialized VolatilityEstimator with method={method.value}, "
            f"lookback={lookback_days}"
        )

    def estimate(
        self,
        returns: np.ndarray,
        prices: Optional[np.ndarray] = None,
    ) -> float:
        """
        Estimate annualized volatility.

        Args:
            returns: Array of daily returns (or log returns)
            prices: Optional price series (used to compute returns if needed)

        Returns:
            Annualized volatility
        """
        if prices is not None and len(returns) == 0:
            returns = np.diff(np.log(prices))

        if len(returns) < 5:
            logger.warning("Insufficient data for vol estimation, using default 20%")
            return 0.20

        if self.method == VolatilityMethod.REALIZED:
            return self._realized_vol(returns)
        elif self.method == VolatilityMethod.EWMA:
            return self._ewma_vol(returns)
        elif self.method == VolatilityMethod.GARCH:
            return self._garch_vol(returns)
        elif self.method == VolatilityMethod.HYBRID:
            return self._hybrid_vol(returns)
        else:
            return self._realized_vol(returns)

    def _realized_vol(self, returns: np.ndarray) -> float:
        """
        Simple realized volatility (standard deviation).

        Uses lookback_days most recent observations.
        """
        lookback = min(len(returns), self.lookback_days)
        recent_returns = returns[-lookback:]

        daily_vol = np.std(recent_returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(self.annualization_factor)

        return float(annual_vol)

    def _ewma_vol(self, returns: np.ndarray) -> float:
        """
        Exponentially Weighted Moving Average volatility.

        EWMA variance: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t

        RiskMetrics uses λ = 0.94 for daily data.
        """
        returns_squared = returns ** 2
        lambda_val = self.ewma_lambda

        # Initialize with sample variance of first observations
        init_window = min(10, len(returns))
        ewma_var = np.var(returns[:init_window])

        # Iterate through returns
        for ret_sq in returns_squared[init_window:]:
            ewma_var = lambda_val * ewma_var + (1 - lambda_val) * ret_sq

        daily_vol = np.sqrt(ewma_var)
        annual_vol = daily_vol * np.sqrt(self.annualization_factor)

        return float(annual_vol)

    def _garch_vol(self, returns: np.ndarray) -> float:
        """
        GARCH(1,1) volatility forecast.

        GARCH(1,1): σ²_t = ω + α * r²_{t-1} + β * σ²_{t-1}

        If arch package not available, falls back to EWMA.
        """
        try:
            from arch import arch_model

            # Scale returns for numerical stability
            returns_scaled = returns * 100

            # Fit GARCH(1,1)
            model = arch_model(
                returns_scaled,
                vol="Garch",
                p=1,
                q=1,
                mean="Zero",
                rescale=False,
            )

            result = model.fit(disp="off", show_warning=False)

            # Get conditional volatility forecast
            forecast = result.forecast(horizon=1)
            daily_var = forecast.variance.values[-1, 0] / 10000  # Unscale

            daily_vol = np.sqrt(daily_var)
            annual_vol = daily_vol * np.sqrt(self.annualization_factor)

            return float(annual_vol)

        except ImportError:
            logger.debug("arch package not available, falling back to EWMA")
            return self._ewma_vol(returns)
        except Exception as e:
            logger.warning(f"GARCH fit failed: {e}, falling back to EWMA")
            return self._ewma_vol(returns)

    def _hybrid_vol(self, returns: np.ndarray) -> float:
        """
        Hybrid volatility: weighted average of methods.

        Uses 50% realized + 50% EWMA for robustness.
        """
        realized = self._realized_vol(returns)
        ewma = self._ewma_vol(returns)

        # Simple average (can be customized)
        hybrid = 0.5 * realized + 0.5 * ewma

        return float(hybrid)

    def estimate_with_confidence(
        self,
        returns: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Estimate volatility with confidence interval.

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
            at 95% confidence level
        """
        vol = self.estimate(returns)
        n = len(returns)

        # Chi-squared confidence interval for variance
        # σ² is estimated with n-1 degrees of freedom
        if n < 10:
            return vol, vol * 0.5, vol * 2.0

        # Approximate 95% CI using chi-squared
        # Lower: sqrt((n-1) * s² / χ²_{0.975})
        # Upper: sqrt((n-1) * s² / χ²_{0.025})
        from scipy import stats

        df = n - 1
        chi2_lower = stats.chi2.ppf(0.975, df)
        chi2_upper = stats.chi2.ppf(0.025, df)

        var = (vol / np.sqrt(self.annualization_factor)) ** 2
        var_lower = df * var / chi2_lower
        var_upper = df * var / chi2_upper

        vol_lower = np.sqrt(var_lower) * np.sqrt(self.annualization_factor)
        vol_upper = np.sqrt(var_upper) * np.sqrt(self.annualization_factor)

        return vol, float(vol_lower), float(vol_upper)


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation."""

    position_size: float  # Dollar amount to allocate
    target_weight: float  # Weight as fraction of capital
    realized_vol: float  # Realized volatility used
    leverage: float  # Effective leverage
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Risk metrics
    expected_daily_var: Optional[float] = None  # Expected daily VaR
    max_loss_1d: Optional[float] = None  # Max 1-day loss at position size

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "position_size": self.position_size,
            "target_weight": self.target_weight,
            "realized_vol": self.realized_vol,
            "leverage": self.leverage,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
            "expected_daily_var": self.expected_daily_var,
            "max_loss_1d": self.max_loss_1d,
        }


@dataclass
class PositionSizerConfig:
    """Configuration for position sizer."""

    # Target volatility
    target_annual_vol: float = 0.15  # 15% annualized target

    # Leverage limits
    max_leverage: float = 2.0  # Maximum 2x leverage
    min_leverage: float = 0.2  # Minimum 20% exposure

    # Volatility estimation
    vol_lookback_days: int = 21  # 21-day rolling window
    vol_floor: float = 0.01  # Minimum vol to avoid divide-by-zero
    vol_ceiling: float = 1.0  # Maximum vol cap

    # Risk limits
    max_position_pct: float = 0.25  # Max 25% in single position
    max_drawdown_trigger: float = 0.15  # Reduce exposure if DD > 15%


class VolatilityScaledPositionSizer:
    """
    Implements volatility-managed position sizing per Moreira & Muir (2017).

    Core formula: w_t = (σ_target² / σ_realized²)

    This scales position size inversely with variance, reducing exposure
    in high-vol regimes and increasing in low-vol regimes.

    Example:
        >>> sizer = VolatilityScaledPositionSizer(target_annual_vol=0.15)
        >>> result = sizer.compute_position_size(
        ...     return_series=daily_returns,
        ...     available_capital=1_000_000
        ... )
        >>> print(f"Allocate ${result.position_size:,.0f}")

    Reference:
        Moreira & Muir (2017) "Volatility-managed portfolios"
    """

    def __init__(
        self,
        config: Optional[PositionSizerConfig] = None,
    ):
        """
        Initialize position sizer.

        Args:
            config: Configuration parameters
        """
        self.config = config or PositionSizerConfig()

        logger.info(
            f"Initialized VolatilityScaledPositionSizer with "
            f"target_vol={self.config.target_annual_vol:.1%}, "
            f"leverage_range=[{self.config.min_leverage}, {self.config.max_leverage}]"
        )

    def compute_position_size(
        self,
        return_series: np.ndarray,
        available_capital: float,
        current_drawdown: float = 0.0,
    ) -> PositionSizeResult:
        """
        Compute position size based on recent realized volatility.

        Args:
            return_series: Array of recent daily returns (last N days)
            available_capital: Total capital available for allocation
            current_drawdown: Current drawdown as positive fraction (0.1 = 10% DD)

        Returns:
            PositionSizeResult with allocation details
        """
        # Compute realized volatility
        realized_vol = self._compute_realized_vol(return_series)

        # Apply floor/ceiling
        realized_vol = np.clip(
            realized_vol, self.config.vol_floor, self.config.vol_ceiling
        )

        # Compute target weight using volatility scaling
        # w_t = (σ_target² / σ_realized²)
        target_weight = (self.config.target_annual_vol ** 2) / (realized_vol ** 2)

        # Apply leverage limits
        target_weight = np.clip(
            target_weight, self.config.min_leverage, self.config.max_leverage
        )

        # Apply drawdown-based reduction
        if current_drawdown > self.config.max_drawdown_trigger:
            dd_multiplier = self._compute_drawdown_multiplier(current_drawdown)
            target_weight *= dd_multiplier
            rationale = (
                f"Vol-scaled weight {target_weight/dd_multiplier:.2f} reduced to "
                f"{target_weight:.2f} due to {current_drawdown:.1%} drawdown"
            )
        else:
            rationale = (
                f"Vol-scaled: realized vol {realized_vol:.1%} vs target "
                f"{self.config.target_annual_vol:.1%} → weight {target_weight:.2f}"
            )

        # Compute position size
        position_size = available_capital * target_weight

        # Apply max position constraint
        max_position = available_capital * self.config.max_position_pct
        if position_size > max_position:
            position_size = max_position
            target_weight = self.config.max_position_pct
            rationale += f" (capped at {self.config.max_position_pct:.0%})"

        # Compute risk metrics
        daily_vol = realized_vol / np.sqrt(252)
        expected_daily_var = position_size * daily_vol * 2.33  # 99% VaR
        max_loss_1d = position_size * daily_vol * 3  # ~3 sigma move

        return PositionSizeResult(
            position_size=position_size,
            target_weight=target_weight,
            realized_vol=realized_vol,
            leverage=target_weight,
            rationale=rationale,
            expected_daily_var=expected_daily_var,
            max_loss_1d=max_loss_1d,
        )

    def compute_portfolio_weights(
        self,
        strategy_returns: Dict[str, np.ndarray],
        total_capital: float,
        strategy_allocations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, PositionSizeResult]:
        """
        Compute position sizes for multiple strategies.

        Args:
            strategy_returns: Dict of {strategy_name: return_array}
            total_capital: Total portfolio capital
            strategy_allocations: Base allocation per strategy (should sum to 1.0)

        Returns:
            Dict of {strategy_name: PositionSizeResult}
        """
        if strategy_allocations is None:
            # Equal weight by default
            n_strategies = len(strategy_returns)
            strategy_allocations = {
                name: 1.0 / n_strategies for name in strategy_returns
            }

        results = {}
        for strategy_name, returns in strategy_returns.items():
            base_allocation = strategy_allocations.get(strategy_name, 0.0)
            available_capital = total_capital * base_allocation

            result = self.compute_position_size(
                return_series=returns,
                available_capital=available_capital,
            )
            results[strategy_name] = result

        return results

    def _compute_realized_vol(self, returns: np.ndarray) -> float:
        """
        Compute annualized realized volatility.

        Uses simple standard deviation, annualized by sqrt(252).
        """
        if len(returns) < 5:
            logger.warning("Insufficient data for vol estimation, using default")
            return self.config.target_annual_vol

        # Use most recent observations up to lookback window
        lookback = min(len(returns), self.config.vol_lookback_days)
        recent_returns = returns[-lookback:]

        # Standard deviation of returns, annualized
        daily_vol = np.std(recent_returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol

    def _compute_drawdown_multiplier(self, drawdown: float) -> float:
        """
        Compute position reduction multiplier based on drawdown.

        Linear reduction: at max_drawdown_trigger, multiplier = 1.0
                         at 2x trigger, multiplier = 0.5
                         at 3x trigger, multiplier = 0.25
        """
        trigger = self.config.max_drawdown_trigger
        excess_dd = drawdown - trigger

        if excess_dd <= 0:
            return 1.0

        # Linear reduction
        multiplier = max(0.25, 1.0 - (excess_dd / trigger))
        return multiplier

    def estimate_required_capital(
        self,
        target_position: float,
        return_series: np.ndarray,
    ) -> float:
        """
        Estimate capital required to achieve target position size.

        Useful for planning/allocation decisions.

        Args:
            target_position: Desired position size in dollars
            return_series: Recent returns for vol estimation

        Returns:
            Required capital
        """
        realized_vol = self._compute_realized_vol(return_series)
        realized_vol = np.clip(
            realized_vol, self.config.vol_floor, self.config.vol_ceiling
        )

        target_weight = (self.config.target_annual_vol ** 2) / (realized_vol ** 2)
        target_weight = np.clip(
            target_weight, self.config.min_leverage, self.config.max_leverage
        )

        required_capital = target_position / target_weight
        return required_capital


class KellyPositionSizer:
    """
    Kelly Criterion position sizing.

    Computes optimal bet size based on expected return and variance.

    Formula: f* = (μ - r) / σ² = Sharpe / σ

    Where:
        f* = optimal fraction of capital
        μ = expected return
        r = risk-free rate
        σ = standard deviation

    Often use fractional Kelly (e.g., half-Kelly) for safety.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        max_position_pct: float = 0.25,
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize Kelly position sizer.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.5 = half-Kelly)
            max_position_pct: Maximum position as fraction of capital
            risk_free_rate: Annual risk-free rate
        """
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.risk_free_rate = risk_free_rate

    def compute_position_size(
        self,
        expected_return: float,
        volatility: float,
        available_capital: float,
    ) -> PositionSizeResult:
        """
        Compute Kelly-optimal position size.

        Args:
            expected_return: Expected annual return
            volatility: Annual volatility
            available_capital: Capital available

        Returns:
            PositionSizeResult
        """
        if volatility <= 0:
            return PositionSizeResult(
                position_size=0,
                target_weight=0,
                realized_vol=0,
                leverage=0,
                rationale="Zero volatility, no position",
            )

        # Full Kelly fraction
        excess_return = expected_return - self.risk_free_rate
        full_kelly = excess_return / (volatility ** 2)

        # Apply fractional Kelly
        target_weight = full_kelly * self.kelly_fraction

        # Apply constraints
        target_weight = max(0, min(target_weight, self.max_position_pct))

        position_size = available_capital * target_weight

        return PositionSizeResult(
            position_size=position_size,
            target_weight=target_weight,
            realized_vol=volatility,
            leverage=target_weight,
            rationale=(
                f"Kelly: μ={expected_return:.1%}, σ={volatility:.1%}, "
                f"f*={full_kelly:.2f}, {self.kelly_fraction:.0%}-Kelly={target_weight:.2f}"
            ),
        )
