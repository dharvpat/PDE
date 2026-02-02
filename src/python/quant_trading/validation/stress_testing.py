"""
Monte Carlo Stress Testing for Trading Strategies.

Provides stress testing capabilities including:
- Historical scenario replay
- Monte Carlo simulation
- Extreme event analysis
- Tail risk assessment

Reference: Section 13.3 of design-doc.md
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ScenarioType(Enum):
    """Types of stress scenarios."""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    REVERSE = "reverse"  # Find scenario that causes target loss


@dataclass
class MarketScenario:
    """Definition of a market stress scenario."""

    name: str
    description: str
    scenario_type: ScenarioType
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    market_shocks: Dict[str, float] = field(default_factory=dict)
    volatility_multiplier: float = 1.0
    correlation_shift: float = 0.0
    duration_days: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "scenario_type": self.scenario_type.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "market_shocks": self.market_shocks,
            "volatility_multiplier": self.volatility_multiplier,
            "correlation_shift": self.correlation_shift,
            "duration_days": self.duration_days,
        }


@dataclass
class StressTestResult:
    """Result of a stress test."""

    scenario: MarketScenario
    portfolio_return: float
    max_drawdown: float
    days_to_recovery: Optional[int]
    var_95: float
    var_99: float
    cvar_95: float
    sharpe_during_stress: float
    worst_day: float
    best_day: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario": self.scenario.to_dict(),
            "portfolio_return": self.portfolio_return,
            "max_drawdown": self.max_drawdown,
            "days_to_recovery": self.days_to_recovery,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "sharpe_during_stress": self.sharpe_during_stress,
            "worst_day": self.worst_day,
            "best_day": self.best_day,
            "details": self.details,
        }


# =============================================================================
# Pre-defined Historical Scenarios
# =============================================================================

HISTORICAL_SCENARIOS = {
    "2008_financial_crisis": MarketScenario(
        name="2008 Financial Crisis",
        description="Global financial crisis triggered by subprime mortgage collapse",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2008, 9, 15),  # Lehman bankruptcy
        end_date=date(2009, 3, 9),  # Market bottom
        market_shocks={
            "SPY": -0.50,  # S&P 500 peak to trough
            "VIX": 3.5,  # VIX went from ~20 to ~80
            "HYG": -0.25,  # High yield bonds
            "TLT": 0.15,  # Treasury bonds (flight to safety)
        },
        volatility_multiplier=3.0,
        correlation_shift=0.3,  # Correlations spike in crisis
        duration_days=126,
    ),

    "2020_covid_crash": MarketScenario(
        name="2020 COVID Crash",
        description="Rapid market decline due to COVID-19 pandemic",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2020, 2, 19),
        end_date=date(2020, 3, 23),
        market_shocks={
            "SPY": -0.34,
            "VIX": 4.0,  # VIX went from ~15 to ~82
            "USO": -0.70,  # Oil collapse
            "HYG": -0.20,
        },
        volatility_multiplier=4.0,
        correlation_shift=0.4,
        duration_days=23,
    ),

    "2010_flash_crash": MarketScenario(
        name="2010 Flash Crash",
        description="Rapid intraday decline and recovery on May 6, 2010",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2010, 5, 6),
        end_date=date(2010, 5, 6),
        market_shocks={
            "SPY": -0.09,  # Intraday drop
            "VIX": 0.5,
        },
        volatility_multiplier=5.0,  # Extreme intraday volatility
        correlation_shift=0.5,
        duration_days=1,
    ),

    "2017_low_volatility": MarketScenario(
        name="2017 Low Volatility",
        description="Persistent low volatility environment",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2017, 1, 1),
        end_date=date(2017, 12, 31),
        market_shocks={
            "SPY": 0.20,  # Strong bull market
            "VIX": -0.5,  # VIX consistently low
        },
        volatility_multiplier=0.5,
        correlation_shift=-0.1,
        duration_days=252,
    ),

    "2022_rate_shock": MarketScenario(
        name="2022 Rate Shock",
        description="Rapid interest rate increases by Federal Reserve",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2022, 1, 1),
        end_date=date(2022, 10, 12),
        market_shocks={
            "SPY": -0.25,
            "TLT": -0.35,  # Long-term bonds crushed
            "QQQ": -0.35,  # Tech hit hard
            "VIX": 1.5,
        },
        volatility_multiplier=1.5,
        correlation_shift=0.2,
        duration_days=200,
    ),

    "2011_debt_ceiling": MarketScenario(
        name="2011 Debt Ceiling Crisis",
        description="US debt ceiling standoff and S&P downgrade",
        scenario_type=ScenarioType.HISTORICAL,
        start_date=date(2011, 7, 22),
        end_date=date(2011, 8, 8),
        market_shocks={
            "SPY": -0.17,
            "VIX": 2.0,
            "TLT": 0.05,  # Paradoxical flight to Treasuries
        },
        volatility_multiplier=2.0,
        correlation_shift=0.25,
        duration_days=12,
    ),
}


class StressTestEngine:
    """Engine for running stress tests on trading strategies."""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        self.scenarios = dict(HISTORICAL_SCENARIOS)

    def add_scenario(self, scenario: MarketScenario) -> None:
        """Add a custom scenario."""
        self.scenarios[scenario.name] = scenario

    def run_historical_scenario(
        self,
        strategy_returns: np.ndarray,
        scenario_name: str,
        portfolio_value: float = 1_000_000,
    ) -> StressTestResult:
        """
        Run a historical stress scenario.

        Applies historical shock to strategy returns and calculates impact.

        Args:
            strategy_returns: Daily returns of the strategy
            scenario_name: Name of the scenario to run
            portfolio_value: Initial portfolio value

        Returns:
            StressTestResult with scenario impact
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = self.scenarios[scenario_name]
        duration = scenario.duration_days

        # Apply volatility scaling
        stressed_returns = strategy_returns.copy()
        if len(stressed_returns) > duration:
            # Scale volatility for scenario period
            period_returns = stressed_returns[:duration]
            scaled_returns = period_returns * scenario.volatility_multiplier

            # Apply market shock proportionally
            avg_shock = np.mean(list(scenario.market_shocks.values()))
            shock_per_day = avg_shock / duration
            scaled_returns = scaled_returns + shock_per_day

            stressed_returns[:duration] = scaled_returns
        else:
            avg_shock = np.mean(list(scenario.market_shocks.values()))
            stressed_returns = strategy_returns * scenario.volatility_multiplier
            stressed_returns = stressed_returns + (avg_shock / len(stressed_returns))

        return self._calculate_stress_metrics(stressed_returns, scenario, portfolio_value)

    def run_monte_carlo_stress(
        self,
        strategy_returns: np.ndarray,
        n_simulations: int = 10000,
        shock_magnitude: float = 0.20,
        portfolio_value: float = 1_000_000,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo stress simulations.

        Args:
            strategy_returns: Historical returns for calibration
            n_simulations: Number of simulations
            shock_magnitude: Magnitude of shocks to apply
            portfolio_value: Initial portfolio value

        Returns:
            Dictionary with Monte Carlo results
        """
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        n_days = 21  # One month

        # Simulate stressed scenarios
        results = {
            "max_drawdowns": [],
            "total_returns": [],
            "worst_days": [],
            "var_95": [],
            "cvar_95": [],
        }

        for _ in range(n_simulations):
            # Generate path with stressed parameters
            shock = self.rng.choice([-1, 1]) * shock_magnitude
            stressed_mean = mean_return + shock / n_days
            stressed_std = std_return * (1 + abs(shock))

            sim_returns = self.rng.normal(stressed_mean, stressed_std, n_days)

            # Calculate metrics
            cum_returns = np.cumprod(1 + sim_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns / running_max - 1
            max_dd = np.min(drawdowns)

            results["max_drawdowns"].append(max_dd)
            results["total_returns"].append(cum_returns[-1] - 1)
            results["worst_days"].append(np.min(sim_returns))
            results["var_95"].append(np.percentile(sim_returns, 5))
            results["cvar_95"].append(np.mean(sim_returns[sim_returns <= np.percentile(sim_returns, 5)]))

        # Calculate statistics
        return {
            "n_simulations": n_simulations,
            "shock_magnitude": shock_magnitude,
            "max_drawdown_mean": np.mean(results["max_drawdowns"]),
            "max_drawdown_5th_percentile": np.percentile(results["max_drawdowns"], 5),
            "max_drawdown_1st_percentile": np.percentile(results["max_drawdowns"], 1),
            "total_return_mean": np.mean(results["total_returns"]),
            "total_return_5th_percentile": np.percentile(results["total_returns"], 5),
            "worst_day_mean": np.mean(results["worst_days"]),
            "worst_day_1st_percentile": np.percentile(results["worst_days"], 1),
            "var_95_mean": np.mean(results["var_95"]),
            "cvar_95_mean": np.mean(results["cvar_95"]),
            "probability_loss_gt_10pct": np.mean(np.array(results["total_returns"]) < -0.10),
            "probability_loss_gt_20pct": np.mean(np.array(results["total_returns"]) < -0.20),
        }

    def run_all_historical_scenarios(
        self,
        strategy_returns: np.ndarray,
        portfolio_value: float = 1_000_000,
    ) -> List[StressTestResult]:
        """
        Run all defined historical scenarios.

        Args:
            strategy_returns: Daily returns of the strategy
            portfolio_value: Initial portfolio value

        Returns:
            List of StressTestResult for each scenario
        """
        results = []
        for scenario_name in self.scenarios:
            result = self.run_historical_scenario(
                strategy_returns, scenario_name, portfolio_value
            )
            results.append(result)
        return results

    def reverse_stress_test(
        self,
        strategy_returns: np.ndarray,
        target_loss: float = 0.25,
        max_iterations: int = 1000,
    ) -> MarketScenario:
        """
        Find scenario that would cause target loss.

        Reverse stress testing per Basel requirements.

        Args:
            strategy_returns: Historical returns for calibration
            target_loss: Target loss percentage (e.g., 0.25 for 25%)
            max_iterations: Maximum optimization iterations

        Returns:
            MarketScenario that would cause target loss
        """
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)

        # Binary search for shock magnitude
        low_shock = 0.0
        high_shock = 2.0  # 200% volatility increase

        for _ in range(max_iterations):
            mid_shock = (low_shock + high_shock) / 2

            # Simulate with this shock
            stressed_returns = strategy_returns * (1 + mid_shock)
            stressed_returns = stressed_returns - mid_shock * std_return

            cum_return = np.prod(1 + stressed_returns[:21]) - 1  # 1 month

            if abs(cum_return - (-target_loss)) < 0.01:
                break
            elif cum_return < -target_loss:
                high_shock = mid_shock
            else:
                low_shock = mid_shock

        return MarketScenario(
            name=f"Reverse Stress ({target_loss*100:.0f}% loss)",
            description=f"Scenario causing {target_loss*100:.0f}% portfolio loss",
            scenario_type=ScenarioType.REVERSE,
            market_shocks={"portfolio": -target_loss},
            volatility_multiplier=1 + mid_shock,
            duration_days=21,
        )

    def _calculate_stress_metrics(
        self,
        returns: np.ndarray,
        scenario: MarketScenario,
        portfolio_value: float,
    ) -> StressTestResult:
        """Calculate stress test metrics from returns."""
        # Total return
        total_return = np.prod(1 + returns) - 1

        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        max_drawdown = np.min(drawdowns)

        # Days to recovery
        if max_drawdown < 0:
            dd_idx = np.argmin(drawdowns)
            recovery_idx = np.where(cum_returns[dd_idx:] >= running_max[dd_idx])[0]
            days_to_recovery = recovery_idx[0] if len(recovery_idx) > 0 else None
        else:
            days_to_recovery = 0

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = np.mean(returns[returns <= var_95])

        # Sharpe during stress
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        return StressTestResult(
            scenario=scenario,
            portfolio_return=total_return,
            max_drawdown=max_drawdown,
            days_to_recovery=days_to_recovery,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            sharpe_during_stress=sharpe,
            worst_day=np.min(returns),
            best_day=np.max(returns),
            details={
                "n_days": len(returns),
                "portfolio_value_end": portfolio_value * (1 + total_return),
                "dollar_loss": portfolio_value * total_return if total_return < 0 else 0,
            },
        )


class TailRiskAnalyzer:
    """Analyze tail risk of trading strategies."""

    def __init__(self):
        pass

    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence_levels: List[float] = [0.95, 0.99],
    ) -> Dict[str, float]:
        """
        Calculate Expected Shortfall (CVaR) at various confidence levels.

        Args:
            returns: Array of returns
            confidence_levels: Confidence levels to calculate

        Returns:
            Dictionary with ES values
        """
        results = {}

        for level in confidence_levels:
            percentile = (1 - level) * 100
            var = np.percentile(returns, percentile)
            es = np.mean(returns[returns <= var])
            results[f"var_{int(level*100)}"] = var
            results[f"es_{int(level*100)}"] = es

        return results

    def extreme_value_analysis(
        self,
        returns: np.ndarray,
        threshold_percentile: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Perform Extreme Value Theory analysis on tail losses.

        Args:
            returns: Array of returns
            threshold_percentile: Percentile for threshold

        Returns:
            Dictionary with EVT analysis
        """
        threshold = np.percentile(returns, threshold_percentile)
        exceedances = returns[returns <= threshold]

        if len(exceedances) < 10:
            return {"error": "Insufficient tail observations"}

        # Fit Generalized Pareto Distribution parameters (simplified)
        excess = threshold - exceedances
        scale = np.mean(excess)  # ML estimate for exponential
        shape = 0.0  # Exponential tail assumption

        return {
            "threshold": threshold,
            "n_exceedances": len(exceedances),
            "excess_mean": np.mean(excess),
            "excess_std": np.std(excess),
            "scale_parameter": scale,
            "shape_parameter": shape,
            "tail_index": 1 / (1 + shape) if shape > -1 else np.inf,
            "expected_max_loss_1y": threshold - scale * np.log(252),
        }

    def drawdown_analysis(
        self,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Comprehensive drawdown analysis.

        Args:
            returns: Array of returns

        Returns:
            Dictionary with drawdown analysis
        """
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1

        # Find drawdown periods
        in_drawdown = drawdowns < 0
        drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
        drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]

        if len(drawdown_starts) == 0:
            return {
                "max_drawdown": 0.0,
                "avg_drawdown": 0.0,
                "n_drawdowns": 0,
            }

        # Calculate drawdown durations
        if len(drawdown_ends) < len(drawdown_starts):
            drawdown_ends = np.append(drawdown_ends, len(returns) - 1)

        durations = drawdown_ends[:len(drawdown_starts)] - drawdown_starts

        return {
            "max_drawdown": np.min(drawdowns),
            "max_drawdown_idx": np.argmin(drawdowns),
            "avg_drawdown": np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0,
            "n_drawdowns": len(drawdown_starts),
            "avg_drawdown_duration": np.mean(durations) if len(durations) > 0 else 0,
            "max_drawdown_duration": np.max(durations) if len(durations) > 0 else 0,
            "time_underwater_pct": np.mean(in_drawdown) * 100,
            "calmar_ratio": (np.prod(1 + returns) ** (252/len(returns)) - 1) / abs(np.min(drawdowns)) if np.min(drawdowns) != 0 else np.inf,
        }
