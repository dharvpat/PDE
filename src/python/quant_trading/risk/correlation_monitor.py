"""
Correlation and Cointegration Monitor.

Monitors rolling correlation and cointegration for mean-reversion strategies.
Detects when statistical relationships break down, requiring position adjustment.

Alerts when:
    - Cointegration test fails (Engle-Granger p-value > 0.05)
    - Correlation drops below threshold
    - Half-life of mean reversion increases significantly
    - Spread behavior becomes non-stationary

Reference:
    - Engle, R. F., & Granger, C. W. (1987). "Co-integration and error
      correction: representation, estimation, and testing." Econometrica.
    - Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs
      trading: Performance of a relative-value arbitrage rule."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None  # type: ignore
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status for correlation/cointegration."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class CointegrationResult:
    """Result from cointegration test."""

    is_cointegrated: bool
    p_value: float
    test_statistic: float
    critical_values: Dict[str, float]  # {'1%': x, '5%': y, '10%': z}
    hedge_ratio: float  # Beta coefficient
    residual_std: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "is_cointegrated": self.is_cointegrated,
            "p_value": self.p_value,
            "test_statistic": self.test_statistic,
            "critical_values": self.critical_values,
            "hedge_ratio": self.hedge_ratio,
            "residual_std": self.residual_std,
        }


@dataclass
class CorrelationHealth:
    """Health assessment for a pair/spread."""

    pair_name: str
    status: HealthStatus
    current_correlation: float
    historical_correlation: float
    correlation_change: float
    cointegration: Optional[CointegrationResult]
    current_half_life: float
    historical_half_life: float
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pair_name": self.pair_name,
            "status": self.status.value,
            "current_correlation": self.current_correlation,
            "historical_correlation": self.historical_correlation,
            "correlation_change": self.correlation_change,
            "cointegration": self.cointegration.to_dict() if self.cointegration else None,
            "current_half_life": self.current_half_life,
            "historical_half_life": self.historical_half_life,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CorrelationMonitorConfig:
    """Configuration for correlation monitoring."""

    # Correlation thresholds
    min_correlation: float = 0.7  # Minimum acceptable correlation
    correlation_drop_threshold: float = 0.15  # Alert if correlation drops by 15%

    # Cointegration thresholds
    cointegration_p_value: float = 0.05  # P-value threshold for cointegration

    # Half-life thresholds
    max_half_life_days: float = 90.0  # Max acceptable half-life
    half_life_increase_threshold: float = 1.5  # Alert if half-life increases 50%

    # Rolling window sizes
    short_window: int = 21  # Short-term correlation (1 month)
    long_window: int = 63  # Long-term correlation (3 months)
    cointegration_window: int = 252  # 1 year for cointegration test

    # Monitoring
    check_frequency_days: int = 1  # How often to check


class CorrelationMonitor:
    """
    Monitor rolling correlation and cointegration for mean-reversion strategies.

    Detects when statistical relationships break down:
        - Cointegration failure (spread no longer mean-reverting)
        - Correlation breakdown (assets moving independently)
        - Half-life degradation (mean-reversion slowing)

    Example:
        >>> monitor = CorrelationMonitor()
        >>> health = monitor.check_pair_health(
        ...     pair_name="SPY-IWM",
        ...     asset1_prices=spy_prices,
        ...     asset2_prices=iwm_prices,
        ...     current_ou_params=ou_result.params
        ... )
        >>> if health.status == HealthStatus.CRITICAL:
        ...     print("Consider closing position!")

    Reference:
        Engle & Granger (1987) "Co-integration and error correction"
    """

    def __init__(
        self,
        config: Optional[CorrelationMonitorConfig] = None,
    ):
        """
        Initialize correlation monitor.

        Args:
            config: Configuration parameters
        """
        self.config = config or CorrelationMonitorConfig()
        self._historical_metrics: Dict[str, Dict] = {}

        logger.info(
            f"Initialized CorrelationMonitor with min_correlation="
            f"{self.config.min_correlation}"
        )

    def check_pair_health(
        self,
        pair_name: str,
        asset1_prices: np.ndarray,
        asset2_prices: np.ndarray,
        current_ou_params: Optional[Dict] = None,
    ) -> CorrelationHealth:
        """
        Comprehensive health check for a trading pair.

        Args:
            pair_name: Identifier for the pair (e.g., "SPY-IWM")
            asset1_prices: Price series for first asset
            asset2_prices: Price series for second asset
            current_ou_params: Current OU parameters if available

        Returns:
            CorrelationHealth with assessment and warnings
        """
        warnings = []

        # Compute returns
        returns1 = np.diff(np.log(asset1_prices))
        returns2 = np.diff(np.log(asset2_prices))

        # Compute correlations
        current_corr = self._compute_rolling_correlation(
            returns1, returns2, self.config.short_window
        )
        historical_corr = self._compute_rolling_correlation(
            returns1, returns2, self.config.long_window
        )

        correlation_change = current_corr - historical_corr

        # Check correlation
        if current_corr < self.config.min_correlation:
            warnings.append(
                f"Correlation {current_corr:.2f} below minimum "
                f"{self.config.min_correlation}"
            )

        if abs(correlation_change) > self.config.correlation_drop_threshold:
            warnings.append(
                f"Correlation changed by {correlation_change:+.2f} "
                f"(from {historical_corr:.2f} to {current_corr:.2f})"
            )

        # Cointegration test
        cointegration = None
        if len(asset1_prices) >= self.config.cointegration_window:
            cointegration = self._test_cointegration(
                asset1_prices[-self.config.cointegration_window:],
                asset2_prices[-self.config.cointegration_window:],
            )

            if not cointegration.is_cointegrated:
                warnings.append(
                    f"Cointegration test failed: p-value {cointegration.p_value:.3f}"
                )

        # Half-life check
        current_half_life = 30.0  # Default
        historical_half_life = 30.0

        if current_ou_params:
            current_half_life = current_ou_params.get("half_life", 30.0)
            if hasattr(current_ou_params, "half_life"):
                current_half_life = current_ou_params.half_life * 252  # Convert to days

        # Get historical half-life from cache
        if pair_name in self._historical_metrics:
            historical_half_life = self._historical_metrics[pair_name].get(
                "half_life", current_half_life
            )

        # Check half-life
        if current_half_life > self.config.max_half_life_days:
            warnings.append(
                f"Half-life {current_half_life:.1f} days exceeds maximum "
                f"{self.config.max_half_life_days}"
            )

        if historical_half_life > 0:
            hl_ratio = current_half_life / historical_half_life
            if hl_ratio > self.config.half_life_increase_threshold:
                warnings.append(
                    f"Half-life increased by {(hl_ratio - 1) * 100:.0f}% "
                    f"({historical_half_life:.1f} → {current_half_life:.1f} days)"
                )

        # Determine overall status
        status = self._determine_status(warnings, cointegration, current_corr)

        # Update historical metrics cache
        self._historical_metrics[pair_name] = {
            "correlation": current_corr,
            "half_life": current_half_life,
            "timestamp": datetime.utcnow(),
        }

        return CorrelationHealth(
            pair_name=pair_name,
            status=status,
            current_correlation=current_corr,
            historical_correlation=historical_corr,
            correlation_change=correlation_change,
            cointegration=cointegration,
            current_half_life=current_half_life,
            historical_half_life=historical_half_life,
            warnings=warnings,
        )

    def _compute_rolling_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        window: int,
    ) -> float:
        """Compute rolling correlation using recent data."""
        if len(returns1) < window or len(returns2) < window:
            window = min(len(returns1), len(returns2))

        if window < 5:
            return 0.0

        recent1 = returns1[-window:]
        recent2 = returns2[-window:]

        correlation = np.corrcoef(recent1, recent2)[0, 1]
        return float(correlation)

    def _test_cointegration(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray,
    ) -> CointegrationResult:
        """
        Perform Engle-Granger cointegration test.

        Steps:
        1. Regress prices1 on prices2 to get hedge ratio
        2. Compute residuals (spread)
        3. Test residuals for stationarity (ADF test)
        """
        # Step 1: OLS regression to find hedge ratio
        # prices1 = alpha + beta * prices2 + epsilon
        X = np.column_stack([np.ones(len(prices2)), prices2])
        coeffs, _, _, _ = np.linalg.lstsq(X, prices1, rcond=None)
        alpha, beta = coeffs

        # Step 2: Compute residuals
        residuals = prices1 - alpha - beta * prices2
        residual_std = np.std(residuals)

        # Step 3: ADF test on residuals
        adf_stat, p_value, critical_values = self._adf_test(residuals)

        is_cointegrated = p_value < self.config.cointegration_p_value

        return CointegrationResult(
            is_cointegrated=is_cointegrated,
            p_value=p_value,
            test_statistic=adf_stat,
            critical_values=critical_values,
            hedge_ratio=float(beta),
            residual_std=float(residual_std),
        )

    def _adf_test(
        self,
        series: np.ndarray,
        max_lags: int = 10,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Returns:
            Tuple of (test_statistic, p_value, critical_values)
        """
        n = len(series)
        if n < max_lags + 10:
            max_lags = max(1, n - 10)

        # Difference series
        diff_series = np.diff(series)
        lagged_series = series[:-1]

        # Build regression matrix with lags
        y = diff_series[max_lags:]
        X = lagged_series[max_lags:]

        # Add lagged differences
        X_full = [np.ones(len(y)), X]
        for lag in range(1, max_lags + 1):
            X_full.append(diff_series[max_lags - lag:-lag] if lag < max_lags else diff_series[:len(y)])

        X_matrix = np.column_stack(X_full[:2])  # Simplified: just constant and lagged level

        # OLS
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_matrix, y, rcond=None)

            # Compute standard error of rho coefficient
            if len(residuals) > 0:
                mse = residuals[0] / (len(y) - 2)
            else:
                mse = np.var(y - X_matrix @ coeffs)

            XtX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
            se_rho = np.sqrt(mse * XtX_inv[1, 1])

            # Test statistic
            rho = coeffs[1]
            adf_stat = rho / se_rho if se_rho > 0 else 0

        except np.linalg.LinAlgError:
            adf_stat = 0

        # Critical values (MacKinnon 1994 approximations for n > 250)
        critical_values = {
            "1%": -3.43,
            "5%": -2.86,
            "10%": -2.57,
        }

        # Approximate p-value
        if adf_stat < critical_values["1%"]:
            p_value = 0.01
        elif adf_stat < critical_values["5%"]:
            p_value = 0.05
        elif adf_stat < critical_values["10%"]:
            p_value = 0.10
        else:
            # Linear interpolation for p > 0.10
            p_value = min(1.0, 0.10 + (adf_stat - critical_values["10%"]) * 0.1)

        return float(adf_stat), float(p_value), critical_values

    def _determine_status(
        self,
        warnings: List[str],
        cointegration: Optional[CointegrationResult],
        correlation: float,
    ) -> HealthStatus:
        """Determine overall health status from individual checks."""
        # Failed: cointegration failed
        if cointegration and not cointegration.is_cointegrated:
            return HealthStatus.FAILED

        # Critical: correlation very low
        if correlation < self.config.min_correlation * 0.7:
            return HealthStatus.CRITICAL

        # Critical: multiple warnings
        if len(warnings) >= 3:
            return HealthStatus.CRITICAL

        # Warning: some issues
        if len(warnings) >= 1:
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    def check_all_pairs(
        self,
        pairs_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        ou_params: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, CorrelationHealth]:
        """
        Check health for all tracked pairs.

        Args:
            pairs_data: Dict of {pair_name: (asset1_prices, asset2_prices)}
            ou_params: Dict of {pair_name: ou_params}

        Returns:
            Dict of {pair_name: CorrelationHealth}
        """
        ou_params = ou_params or {}
        results = {}

        for pair_name, (prices1, prices2) in pairs_data.items():
            health = self.check_pair_health(
                pair_name=pair_name,
                asset1_prices=prices1,
                asset2_prices=prices2,
                current_ou_params=ou_params.get(pair_name),
            )
            results[pair_name] = health

        # Log summary
        unhealthy = [name for name, h in results.items() if h.status != HealthStatus.HEALTHY]
        if unhealthy:
            logger.warning(f"Unhealthy pairs: {unhealthy}")

        return results

    def get_position_recommendations(
        self,
        health: CorrelationHealth,
        current_position_size: float,
    ) -> Dict:
        """
        Get position sizing recommendations based on health.

        Args:
            health: Correlation health assessment
            current_position_size: Current position size

        Returns:
            Dict with recommendation
        """
        if health.status == HealthStatus.FAILED:
            return {
                "action": "close",
                "target_size": 0,
                "reason": "Cointegration failed - relationship broken",
                "urgency": "high",
            }

        if health.status == HealthStatus.CRITICAL:
            return {
                "action": "reduce",
                "target_size": current_position_size * 0.25,
                "reason": "Critical health status - reduce exposure",
                "urgency": "high",
            }

        if health.status == HealthStatus.WARNING:
            return {
                "action": "reduce",
                "target_size": current_position_size * 0.5,
                "reason": "Warning status - consider reducing",
                "urgency": "normal",
            }

        return {
            "action": "maintain",
            "target_size": current_position_size,
            "reason": "Healthy - no change needed",
            "urgency": "none",
        }
