"""
Value at Risk (VaR) and Conditional VaR (CVaR) Calculator.

Implements multiple VaR calculation methods:
    - Parametric (delta-normal): Assumes normal distribution
    - Historical simulation: Uses empirical distribution
    - Monte Carlo simulation: Generates correlated scenarios

Also provides:
    - CVaR/Expected Shortfall for tail risk measurement
    - Component VaR for risk attribution
    - Stress testing with historical crisis scenarios
    - VaR backtesting (Kupiec test)

Mathematical Background:
    VaR_α: Loss that will not be exceeded with probability α (e.g., 95%)
    VaR_α = -quantile(returns, 1-α)

    CVaR_α: Expected loss given that loss exceeds VaR
    CVaR_α = E[L | L > VaR_α]

Reference:
    - Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk"
    - RiskMetrics Technical Document (1996)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None  # type: ignore
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VaRMethod(Enum):
    """VaR calculation methods."""

    PARAMETRIC = "parametric"  # Delta-normal method
    HISTORICAL = "historical"  # Historical simulation
    MONTE_CARLO = "monte_carlo"  # Monte Carlo simulation


@dataclass
class VaRResult:
    """
    VaR calculation result.

    Attributes:
        var_95: 95% Value at Risk (1-day, as positive number)
        var_99: 99% Value at Risk
        cvar_95: 95% Conditional VaR (Expected Shortfall)
        cvar_99: 99% Conditional VaR
        method: Method used for calculation
        time_horizon: Time horizon in days
        portfolio_value: Portfolio value used
        component_var: VaR contribution by position
        timestamp: Calculation timestamp
    """

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    method: str
    time_horizon: int = 1
    portfolio_value: float = 0.0
    component_var: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def var_95_pct(self) -> float:
        """VaR as percentage of portfolio."""
        if self.portfolio_value > 0:
            return self.var_95 / self.portfolio_value
        return 0.0

    @property
    def var_99_pct(self) -> float:
        """VaR as percentage of portfolio."""
        if self.portfolio_value > 0:
            return self.var_99 / self.portfolio_value
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "var_95_pct": self.var_95_pct,
            "var_99_pct": self.var_99_pct,
            "method": self.method,
            "time_horizon": self.time_horizon,
            "portfolio_value": self.portfolio_value,
            "component_var": self.component_var,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StressTestResult:
    """
    Stress test result.

    Attributes:
        scenario_name: Name of the scenario
        scenario_pnl: P&L under the scenario
        scenario_pnl_pct: P&L as percentage of portfolio
        positions_affected: Positions with P&L impact
        timestamp: Test timestamp
    """

    scenario_name: str
    scenario_pnl: float
    scenario_pnl_pct: float
    positions_affected: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_pnl": self.scenario_pnl,
            "scenario_pnl_pct": self.scenario_pnl_pct,
            "positions_affected": self.positions_affected,
            "timestamp": self.timestamp.isoformat(),
        }


class VaRCalculator:
    """
    Calculate Value at Risk using multiple methods.

    Supports parametric, historical, and Monte Carlo approaches
    with configurable confidence levels and time horizons.

    Example:
        >>> calculator = VaRCalculator(method=VaRMethod.PARAMETRIC)
        >>>
        >>> # Calculate portfolio VaR
        >>> result = calculator.calculate(
        ...     position_values={'SPY': 45000, 'TLT': 10000},
        ...     historical_returns=returns_df,
        ... )
        >>>
        >>> print(f"95% VaR (1-day): ${result.var_95:,.0f}")
        >>> print(f"95% CVaR: ${result.cvar_95:,.0f}")

    Reference:
        Jorion (2007) "Value at Risk"
    """

    def __init__(
        self,
        method: VaRMethod = VaRMethod.PARAMETRIC,
        confidence_levels: Tuple[float, float] = (0.95, 0.99),
        time_horizon: int = 1,
        n_simulations: int = 10000,
    ):
        """
        Initialize VaR calculator.

        Args:
            method: Calculation method
            confidence_levels: Confidence levels (default: 95%, 99%)
            time_horizon: Time horizon in days
            n_simulations: Number of Monte Carlo simulations
        """
        self.method = method
        self.confidence_levels = confidence_levels
        self.time_horizon = time_horizon
        self.n_simulations = n_simulations

        logger.info(
            f"Initialized VaRCalculator with method={method.value}, "
            f"horizon={time_horizon} days"
        )

    def calculate(
        self,
        position_values: Dict[str, float],
        historical_returns: np.ndarray,
        asset_ids: Optional[List[str]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> VaRResult:
        """
        Calculate portfolio VaR and CVaR.

        Args:
            position_values: Dict of {asset_id: market_value}
            historical_returns: 2D array of returns (rows=time, cols=assets)
                               or 1D array for single asset
            asset_ids: List of asset IDs matching columns in returns
            correlation_matrix: Optional correlation matrix (computed if not provided)

        Returns:
            VaRResult with VaR and CVaR metrics
        """
        # Handle input formats
        if isinstance(historical_returns, np.ndarray):
            if historical_returns.ndim == 1:
                historical_returns = historical_returns.reshape(-1, 1)

        n_assets = historical_returns.shape[1] if historical_returns.ndim > 1 else 1

        if asset_ids is None:
            asset_ids = list(position_values.keys())

        # Ensure we have returns for all positions
        if n_assets != len(asset_ids):
            logger.warning(
                f"Asset count mismatch: {n_assets} returns vs {len(asset_ids)} positions"
            )

        # Get position values in order
        values = np.array([position_values.get(aid, 0.0) for aid in asset_ids[:n_assets]])
        portfolio_value = np.sum(np.abs(values))

        if self.method == VaRMethod.PARAMETRIC:
            return self._parametric_var(values, historical_returns, asset_ids, portfolio_value)
        elif self.method == VaRMethod.HISTORICAL:
            return self._historical_var(values, historical_returns, asset_ids, portfolio_value)
        elif self.method == VaRMethod.MONTE_CARLO:
            return self._monte_carlo_var(values, historical_returns, asset_ids, portfolio_value)
        else:
            return self._parametric_var(values, historical_returns, asset_ids, portfolio_value)

    def _parametric_var(
        self,
        position_values: np.ndarray,
        returns: np.ndarray,
        asset_ids: List[str],
        portfolio_value: float,
    ) -> VaRResult:
        """
        Parametric VaR (delta-normal method).

        Assumes returns are normally distributed.
        VaR_α = -μ + σ * Z_α * sqrt(time_horizon)

        For portfolio: σ_p = sqrt(w' * Σ * w)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using simplified VaR")
            return self._simplified_var(position_values, returns, asset_ids, portfolio_value)

        n_assets = len(position_values)

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Estimate parameters
        mean_returns = np.mean(returns, axis=0)  # Daily mean
        cov_matrix = np.cov(returns, rowvar=False)  # Daily covariance

        # Handle single asset case
        if n_assets == 1:
            cov_matrix = np.array([[cov_matrix]]) if cov_matrix.ndim == 0 else cov_matrix.reshape(1, 1)
            mean_returns = np.array([mean_returns]) if mean_returns.ndim == 0 else mean_returns

        # Portfolio return parameters
        portfolio_mean = np.dot(position_values, mean_returns[:n_assets])

        # Ensure cov_matrix is 2D and correct size
        if cov_matrix.ndim < 2:
            cov_matrix = np.array([[cov_matrix]])
        cov_matrix = cov_matrix[:n_assets, :n_assets]

        portfolio_var = np.dot(position_values, np.dot(cov_matrix, position_values))
        portfolio_std = np.sqrt(max(0, portfolio_var))

        # Scale to time horizon
        portfolio_mean_scaled = portfolio_mean * self.time_horizon
        portfolio_std_scaled = portfolio_std * np.sqrt(self.time_horizon)

        # Calculate VaR (as positive number representing potential loss)
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        var_95 = -portfolio_mean_scaled + z_95 * portfolio_std_scaled
        var_99 = -portfolio_mean_scaled + z_99 * portfolio_std_scaled

        # CVaR (analytical for normal distribution)
        # CVaR_α = σ * φ(Z_α) / (1-α) - μ
        cvar_95 = portfolio_std_scaled * stats.norm.pdf(z_95) / 0.05 - portfolio_mean_scaled
        cvar_99 = portfolio_std_scaled * stats.norm.pdf(z_99) / 0.01 - portfolio_mean_scaled

        # Component VaR (marginal contribution)
        component_var = self._compute_component_var(
            position_values, cov_matrix, portfolio_std, asset_ids, var_95
        )

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            method="parametric",
            time_horizon=self.time_horizon,
            portfolio_value=portfolio_value,
            component_var=component_var,
        )

    def _historical_var(
        self,
        position_values: np.ndarray,
        returns: np.ndarray,
        asset_ids: List[str],
        portfolio_value: float,
    ) -> VaRResult:
        """
        Historical VaR using empirical distribution.

        Uses actual historical returns without distribution assumptions.
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        n_assets = min(len(position_values), returns.shape[1])

        # Compute historical portfolio P&L for each day
        portfolio_returns = returns[:, :n_assets] @ position_values[:n_assets]

        # Scale to time horizon (simple scaling)
        portfolio_returns_scaled = portfolio_returns * np.sqrt(self.time_horizon)

        # Sort returns (losses are negative)
        sorted_returns = np.sort(portfolio_returns_scaled)

        # VaR is negative of the α-quantile
        n = len(sorted_returns)
        var_95_idx = max(0, int(n * 0.05) - 1)
        var_99_idx = max(0, int(n * 0.01) - 1)

        var_95 = -sorted_returns[var_95_idx]
        var_99 = -sorted_returns[var_99_idx]

        # CVaR is average of losses beyond VaR
        losses_beyond_95 = sorted_returns[:var_95_idx + 1]
        losses_beyond_99 = sorted_returns[:var_99_idx + 1]

        cvar_95 = -np.mean(losses_beyond_95) if len(losses_beyond_95) > 0 else var_95
        cvar_99 = -np.mean(losses_beyond_99) if len(losses_beyond_99) > 0 else var_99

        # Component VaR (approximate)
        component_var = {}
        for i, aid in enumerate(asset_ids[:n_assets]):
            asset_returns = returns[:, i] * position_values[i]
            # Correlation with portfolio losses
            if len(asset_returns) > 0 and np.std(portfolio_returns) > 0:
                corr = np.corrcoef(asset_returns, portfolio_returns)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                component_var[aid] = abs(corr) * var_95 * abs(position_values[i]) / portfolio_value
            else:
                component_var[aid] = 0.0

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            method="historical",
            time_horizon=self.time_horizon,
            portfolio_value=portfolio_value,
            component_var=component_var,
        )

    def _monte_carlo_var(
        self,
        position_values: np.ndarray,
        returns: np.ndarray,
        asset_ids: List[str],
        portfolio_value: float,
    ) -> VaRResult:
        """
        Monte Carlo VaR via simulation.

        Generates correlated return scenarios using multivariate normal.
        """
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        n_assets = min(len(position_values), returns.shape[1])

        # Estimate parameters
        mean_returns = np.mean(returns[:, :n_assets], axis=0)
        cov_matrix = np.cov(returns[:, :n_assets], rowvar=False)

        # Handle single asset
        if n_assets == 1:
            cov_matrix = np.array([[cov_matrix]]) if cov_matrix.ndim == 0 else cov_matrix.reshape(1, 1)
            mean_returns = np.array([mean_returns]) if mean_returns.ndim == 0 else mean_returns

        # Scale to time horizon
        mean_scaled = mean_returns * self.time_horizon
        cov_scaled = cov_matrix * self.time_horizon

        # Ensure positive semi-definite
        try:
            # Add small regularization if needed
            cov_scaled = cov_scaled + np.eye(n_assets) * 1e-8

            # Generate correlated random returns
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.multivariate_normal(
                mean_scaled,
                cov_scaled,
                size=self.n_simulations,
            )
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix not positive definite, using diagonal")
            variances = np.diag(cov_scaled)
            simulated_returns = np.random.normal(
                mean_scaled,
                np.sqrt(variances),
                size=(self.n_simulations, n_assets),
            )

        # Compute portfolio P&L for each scenario
        portfolio_pnl = simulated_returns @ position_values[:n_assets]

        # Sort P&L
        sorted_pnl = np.sort(portfolio_pnl)

        # VaR
        var_95_idx = int(self.n_simulations * 0.05)
        var_99_idx = int(self.n_simulations * 0.01)

        var_95 = -sorted_pnl[var_95_idx]
        var_99 = -sorted_pnl[var_99_idx]

        # CVaR
        cvar_95 = -np.mean(sorted_pnl[:var_95_idx]) if var_95_idx > 0 else var_95
        cvar_99 = -np.mean(sorted_pnl[:var_99_idx]) if var_99_idx > 0 else var_99

        # Component VaR
        component_var = {}
        for i, aid in enumerate(asset_ids[:n_assets]):
            asset_pnl = simulated_returns[:, i] * position_values[i]
            corr = np.corrcoef(asset_pnl, portfolio_pnl)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            component_var[aid] = abs(corr) * var_95 * abs(position_values[i]) / portfolio_value

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            method="monte_carlo",
            time_horizon=self.time_horizon,
            portfolio_value=portfolio_value,
            component_var=component_var,
        )

    def _simplified_var(
        self,
        position_values: np.ndarray,
        returns: np.ndarray,
        asset_ids: List[str],
        portfolio_value: float,
    ) -> VaRResult:
        """Simplified VaR when scipy is not available."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        n_assets = min(len(position_values), returns.shape[1])
        portfolio_returns = returns[:, :n_assets] @ position_values[:n_assets]

        std = np.std(portfolio_returns)
        mean = np.mean(portfolio_returns)

        # Use 1.65 for 95% and 2.33 for 99% (normal approximation)
        var_95 = -mean + 1.65 * std * np.sqrt(self.time_horizon)
        var_99 = -mean + 2.33 * std * np.sqrt(self.time_horizon)

        # Approximate CVaR
        cvar_95 = var_95 * 1.2
        cvar_99 = var_99 * 1.2

        return VaRResult(
            var_95=max(0, var_95),
            var_99=max(0, var_99),
            cvar_95=max(0, cvar_95),
            cvar_99=max(0, cvar_99),
            method="simplified",
            time_horizon=self.time_horizon,
            portfolio_value=portfolio_value,
            component_var={},
        )

    def _compute_component_var(
        self,
        position_values: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_std: float,
        asset_ids: List[str],
        total_var: float,
    ) -> Dict[str, float]:
        """
        Compute component VaR (marginal contribution to VaR).

        Component VaR_i = (w_i * (Σw)_i / σ_p) * VaR_total
        """
        component_var = {}

        if portfolio_std <= 0:
            return {aid: 0.0 for aid in asset_ids}

        n = len(position_values)
        for i, aid in enumerate(asset_ids[:n]):
            # Marginal contribution: (Σw)_i / σ_p
            marginal = np.dot(cov_matrix[i, :n], position_values) / portfolio_std
            # Component VaR
            component_var[aid] = marginal * abs(position_values[i]) / np.sum(np.abs(position_values))

        # Normalize to sum to total VaR
        total_component = sum(abs(v) for v in component_var.values())
        if total_component > 0:
            for aid in component_var:
                component_var[aid] = component_var[aid] / total_component * total_var

        return component_var


class StressTester:
    """
    Stress testing and scenario analysis.

    Applies predefined or custom stress scenarios to portfolio
    and calculates potential P&L impact.

    Example:
        >>> stress_tester = StressTester()
        >>>
        >>> # Apply 2008 crisis scenario
        >>> result = stress_tester.apply_scenario(
        ...     portfolio={'SPY': 45000, 'TLT': 10000},
        ...     scenario_name='2008_financial_crisis'
        ... )
        >>>
        >>> print(f"Crisis P&L: ${result.scenario_pnl:,.0f}")

    Reference:
        Standard stress testing practices (Basel III)
    """

    def __init__(self):
        """Initialize stress tester with historical crisis scenarios."""
        # Historical crisis scenarios (approximate returns during crisis periods)
        self.scenarios: Dict[str, Dict[str, float]] = {
            "2008_financial_crisis": {
                "SPY": -0.38,  # S&P 500 peak-to-trough
                "QQQ": -0.42,  # Nasdaq
                "IWM": -0.40,  # Russell 2000
                "TLT": 0.25,   # Long-term treasuries rallied
                "GLD": 0.05,   # Gold modest gain
                "HYG": -0.25,  # High yield bonds
                "VIX": 3.50,   # VIX spike (multiplier)
            },
            "2020_covid_crash": {
                "SPY": -0.34,  # March 2020 selloff
                "QQQ": -0.28,
                "IWM": -0.42,
                "TLT": 0.15,
                "GLD": 0.08,
                "HYG": -0.20,
                "VIX": 4.00,
            },
            "1987_black_monday": {
                "SPY": -0.22,  # Single day drop
                "QQQ": -0.22,
                "IWM": -0.25,
            },
            "2011_euro_crisis": {
                "SPY": -0.20,
                "TLT": 0.15,
                "GLD": 0.12,
            },
            "2022_rate_hike": {
                "SPY": -0.25,
                "QQQ": -0.33,
                "TLT": -0.30,  # Bonds fell with rate hikes
                "GLD": -0.05,
            },
            "vol_spike_20pct": {
                # Uniform volatility shock
                "SPY": -0.10,
                "QQQ": -0.12,
                "IWM": -0.11,
                "TLT": -0.03,
            },
            "correlation_breakdown": {
                # Scenario where diversification fails
                "SPY": -0.15,
                "QQQ": -0.15,
                "IWM": -0.15,
                "TLT": -0.10,  # Bonds also fall
                "GLD": -0.05,
            },
        }

        logger.info(f"Initialized StressTester with {len(self.scenarios)} scenarios")

    def add_scenario(self, name: str, shocks: Dict[str, float]) -> None:
        """
        Add a custom stress scenario.

        Args:
            name: Scenario name
            shocks: Dict of {asset_id: return_shock}
        """
        self.scenarios[name] = shocks
        logger.info(f"Added stress scenario: {name}")

    def apply_scenario(
        self,
        portfolio: Dict[str, float],
        scenario_name: str,
    ) -> StressTestResult:
        """
        Apply a named stress scenario to portfolio.

        Args:
            portfolio: Dict of {asset_id: market_value}
            scenario_name: Name of scenario to apply

        Returns:
            StressTestResult with P&L impact
        """
        if scenario_name not in self.scenarios:
            logger.warning(f"Scenario {scenario_name} not found")
            return StressTestResult(
                scenario_name=scenario_name,
                scenario_pnl=0.0,
                scenario_pnl_pct=0.0,
            )

        scenario = self.scenarios[scenario_name]
        return self.apply_custom_scenario(portfolio, scenario, scenario_name)

    def apply_custom_scenario(
        self,
        portfolio: Dict[str, float],
        shocks: Dict[str, float],
        scenario_name: str = "custom",
    ) -> StressTestResult:
        """
        Apply custom shock scenario to portfolio.

        Args:
            portfolio: Dict of {asset_id: market_value}
            shocks: Dict of {asset_id: return_shock}
            scenario_name: Name for the scenario

        Returns:
            StressTestResult with P&L impact
        """
        total_pnl = 0.0
        positions_affected = {}
        portfolio_value = sum(abs(v) for v in portfolio.values())

        for asset_id, market_value in portfolio.items():
            if asset_id in shocks:
                shock = shocks[asset_id]
                position_pnl = market_value * shock
                total_pnl += position_pnl
                positions_affected[asset_id] = position_pnl

        pnl_pct = total_pnl / portfolio_value if portfolio_value > 0 else 0.0

        return StressTestResult(
            scenario_name=scenario_name,
            scenario_pnl=total_pnl,
            scenario_pnl_pct=pnl_pct,
            positions_affected=positions_affected,
        )

    def run_all_scenarios(
        self,
        portfolio: Dict[str, float],
    ) -> List[StressTestResult]:
        """
        Run all predefined stress scenarios.

        Args:
            portfolio: Dict of {asset_id: market_value}

        Returns:
            List of StressTestResult for each scenario
        """
        results = []

        for scenario_name in self.scenarios:
            result = self.apply_scenario(portfolio, scenario_name)
            results.append(result)

        # Sort by P&L (worst first)
        results.sort(key=lambda x: x.scenario_pnl)

        return results

    def get_worst_case(
        self,
        portfolio: Dict[str, float],
    ) -> StressTestResult:
        """
        Get worst-case scenario result.

        Args:
            portfolio: Dict of {asset_id: market_value}

        Returns:
            StressTestResult for worst scenario
        """
        results = self.run_all_scenarios(portfolio)
        return results[0] if results else StressTestResult(
            scenario_name="none",
            scenario_pnl=0.0,
            scenario_pnl_pct=0.0,
        )

    def summary_report(
        self,
        portfolio: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Generate stress test summary report.

        Args:
            portfolio: Dict of {asset_id: market_value}

        Returns:
            Dict with summary statistics
        """
        results = self.run_all_scenarios(portfolio)
        portfolio_value = sum(abs(v) for v in portfolio.values())

        pnls = [r.scenario_pnl for r in results]

        return {
            "portfolio_value": portfolio_value,
            "num_scenarios": len(results),
            "worst_case": {
                "scenario": results[0].scenario_name if results else "none",
                "pnl": results[0].scenario_pnl if results else 0.0,
                "pnl_pct": results[0].scenario_pnl_pct if results else 0.0,
            },
            "best_case": {
                "scenario": results[-1].scenario_name if results else "none",
                "pnl": results[-1].scenario_pnl if results else 0.0,
                "pnl_pct": results[-1].scenario_pnl_pct if results else 0.0,
            },
            "average_pnl": np.mean(pnls) if pnls else 0.0,
            "median_pnl": np.median(pnls) if pnls else 0.0,
            "scenarios": [r.to_dict() for r in results],
        }


class VaRBacktester:
    """
    VaR model backtesting using Kupiec test.

    Tests whether the actual number of VaR breaches is consistent
    with the expected number given the confidence level.

    Example:
        >>> backtester = VaRBacktester()
        >>> result = backtester.kupiec_test(
        ...     var_estimates=var_history,
        ...     actual_returns=returns,
        ...     confidence_level=0.95
        ... )
        >>> print(f"Model valid: {result['is_valid']}")
    """

    def kupiec_test(
        self,
        var_estimates: np.ndarray,
        actual_pnl: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Perform Kupiec proportion of failures (POF) test.

        H0: The model is correctly calibrated (true failure rate = 1 - confidence)
        H1: The model is mis-calibrated

        Args:
            var_estimates: Array of VaR estimates (positive numbers)
            actual_pnl: Array of actual P&L (negative = loss)
            confidence_level: VaR confidence level

        Returns:
            Dict with test results
        """
        n = len(var_estimates)
        expected_failure_rate = 1 - confidence_level

        # Count breaches (when loss exceeds VaR)
        breaches = np.sum(-actual_pnl > var_estimates)
        observed_failure_rate = breaches / n if n > 0 else 0

        # Kupiec test statistic (likelihood ratio)
        p = expected_failure_rate
        p_hat = observed_failure_rate

        if p_hat == 0:
            p_hat = 1e-10
        if p_hat == 1:
            p_hat = 1 - 1e-10

        # LR = -2 * log[(1-p)^(n-x) * p^x / (1-p_hat)^(n-x) * p_hat^x]
        x = breaches
        lr_stat = -2 * (
            (n - x) * np.log((1 - p) / (1 - p_hat)) +
            x * np.log(p / p_hat)
        )

        # Chi-squared critical value (1 df)
        if SCIPY_AVAILABLE:
            p_value = 1 - stats.chi2.cdf(lr_stat, 1)
            critical_value = stats.chi2.ppf(0.95, 1)
        else:
            p_value = 0.05 if lr_stat > 3.84 else 0.10
            critical_value = 3.84

        is_valid = lr_stat < critical_value

        return {
            "n_observations": n,
            "n_breaches": int(breaches),
            "expected_breaches": n * expected_failure_rate,
            "observed_failure_rate": observed_failure_rate,
            "expected_failure_rate": expected_failure_rate,
            "lr_statistic": lr_stat,
            "critical_value": critical_value,
            "p_value": p_value,
            "is_valid": is_valid,
            "assessment": "Model accepted" if is_valid else "Model rejected",
        }
