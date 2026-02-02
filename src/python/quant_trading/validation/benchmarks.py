"""
Benchmark Comparison Utilities for Strategy Validation.

Compare strategy performance against standard benchmarks:
- Buy-and-hold S&P 500
- 60/40 Stock/Bond portfolio
- Pure momentum factor
- Risk-free rate

Reference: Section 13.2 of design-doc.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BenchmarkType(Enum):
    """Types of benchmarks."""
    BUY_AND_HOLD = "buy_and_hold"
    SIXTY_FORTY = "sixty_forty"
    MOMENTUM = "momentum"
    RISK_FREE = "risk_free"
    EQUAL_WEIGHT = "equal_weight"
    CUSTOM = "custom"


@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""

    benchmark_name: str
    benchmark_type: BenchmarkType
    strategy_metrics: Dict[str, float]
    benchmark_metrics: Dict[str, float]
    relative_metrics: Dict[str, float]
    outperformance: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type.value,
            "strategy_metrics": self.strategy_metrics,
            "benchmark_metrics": self.benchmark_metrics,
            "relative_metrics": self.relative_metrics,
            "outperformance": self.outperformance,
            "details": self.details,
        }


@dataclass
class ComparisonReport:
    """Comprehensive benchmark comparison report."""

    strategy_name: str
    strategy_returns: np.ndarray
    benchmark_results: List[BenchmarkResult]
    overall_ranking: int  # 1 = best
    summary: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "benchmark_results": [r.to_dict() for r in self.benchmark_results],
            "overall_ranking": self.overall_ranking,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


class Benchmark:
    """Base class for benchmarks."""

    def __init__(self, name: str, benchmark_type: BenchmarkType):
        self.name = name
        self.benchmark_type = benchmark_type

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculate benchmark returns."""
        raise NotImplementedError


class BuyAndHoldBenchmark(Benchmark):
    """Buy and hold market benchmark (e.g., S&P 500)."""

    def __init__(self, name: str = "S&P 500 Buy-and-Hold"):
        super().__init__(name, BenchmarkType.BUY_AND_HOLD)

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return market returns directly."""
        return market_returns


class SixtyFortyBenchmark(Benchmark):
    """60/40 Stock/Bond portfolio benchmark."""

    def __init__(
        self,
        stock_weight: float = 0.60,
        bond_weight: float = 0.40,
        name: str = "60/40 Portfolio",
    ):
        super().__init__(name, BenchmarkType.SIXTY_FORTY)
        self.stock_weight = stock_weight
        self.bond_weight = bond_weight

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        bond_returns: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculate 60/40 portfolio returns."""
        if bond_returns is None:
            # Approximate bond returns
            # Assume bonds have ~1/3 volatility of stocks and low correlation
            bond_returns = market_returns * 0.3 + np.random.normal(0, 0.002, len(market_returns))

        return self.stock_weight * market_returns + self.bond_weight * bond_returns


class MomentumBenchmark(Benchmark):
    """Pure momentum factor benchmark."""

    def __init__(
        self,
        lookback_period: int = 252,
        holding_period: int = 21,
        name: str = "Momentum Factor",
    ):
        super().__init__(name, BenchmarkType.MOMENTUM)
        self.lookback_period = lookback_period
        self.holding_period = holding_period

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculate momentum strategy returns."""
        n = len(market_returns)
        momentum_returns = np.zeros(n)

        for i in range(self.lookback_period, n, self.holding_period):
            # Calculate momentum signal
            past_return = np.prod(1 + market_returns[i-self.lookback_period:i]) - 1

            # Long if positive momentum, flat otherwise
            if past_return > 0:
                end_idx = min(i + self.holding_period, n)
                momentum_returns[i:end_idx] = market_returns[i:end_idx]

        return momentum_returns


class RiskFreeBenchmark(Benchmark):
    """Risk-free rate benchmark."""

    def __init__(
        self,
        annual_rate: float = 0.04,
        name: str = "Risk-Free Rate",
    ):
        super().__init__(name, BenchmarkType.RISK_FREE)
        self.annual_rate = annual_rate
        self.daily_rate = (1 + annual_rate) ** (1/252) - 1

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return constant risk-free rate."""
        return np.full(len(market_returns), self.daily_rate)


class EqualWeightBenchmark(Benchmark):
    """Equal-weight portfolio benchmark."""

    def __init__(self, name: str = "Equal Weight"):
        super().__init__(name, BenchmarkType.EQUAL_WEIGHT)

    def calculate_returns(
        self,
        market_returns: np.ndarray,
        asset_returns: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Calculate equal-weight portfolio returns."""
        if asset_returns is None:
            return market_returns

        # asset_returns shape: (n_observations, n_assets)
        if asset_returns.ndim == 1:
            return asset_returns

        n_assets = asset_returns.shape[1]
        weights = np.ones(n_assets) / n_assets
        return np.dot(asset_returns, weights)


class BenchmarkComparator:
    """Compare strategy against multiple benchmarks."""

    def __init__(self, benchmarks: Optional[List[Benchmark]] = None):
        """
        Initialize comparator with benchmarks.

        Args:
            benchmarks: List of Benchmark objects. If None, uses defaults.
        """
        if benchmarks is None:
            self.benchmarks = [
                BuyAndHoldBenchmark(),
                SixtyFortyBenchmark(),
                MomentumBenchmark(),
                RiskFreeBenchmark(),
            ]
        else:
            self.benchmarks = benchmarks

    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark."""
        self.benchmarks.append(benchmark)

    def compare(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray,
        strategy_name: str = "Strategy",
        **kwargs: Any,
    ) -> ComparisonReport:
        """
        Compare strategy against all benchmarks.

        Args:
            strategy_returns: Strategy return series
            market_returns: Market return series (for benchmarks)
            strategy_name: Name of the strategy
            **kwargs: Additional data for benchmarks (e.g., bond_returns)

        Returns:
            ComparisonReport with all comparisons
        """
        strategy_metrics = self._calculate_metrics(strategy_returns)
        results = []

        for benchmark in self.benchmarks:
            benchmark_returns = benchmark.calculate_returns(market_returns, **kwargs)
            benchmark_metrics = self._calculate_metrics(benchmark_returns)
            relative_metrics = self._calculate_relative_metrics(
                strategy_metrics, benchmark_metrics
            )

            # Determine outperformance (based on Sharpe ratio)
            outperformance = strategy_metrics["sharpe_ratio"] > benchmark_metrics["sharpe_ratio"]

            result = BenchmarkResult(
                benchmark_name=benchmark.name,
                benchmark_type=benchmark.benchmark_type,
                strategy_metrics=strategy_metrics,
                benchmark_metrics=benchmark_metrics,
                relative_metrics=relative_metrics,
                outperformance=outperformance,
            )
            results.append(result)

        # Calculate overall ranking
        all_sharpes = [strategy_metrics["sharpe_ratio"]] + [
            r.benchmark_metrics["sharpe_ratio"] for r in results
        ]
        ranking = sorted(range(len(all_sharpes)), key=lambda i: -all_sharpes[i])
        strategy_rank = ranking.index(0) + 1

        # Generate summary
        n_outperformed = sum(1 for r in results if r.outperformance)
        summary = {
            "n_benchmarks": len(results),
            "n_outperformed": n_outperformed,
            "outperformance_rate": n_outperformed / len(results) if results else 0,
            "strategy_sharpe": strategy_metrics["sharpe_ratio"],
            "avg_benchmark_sharpe": np.mean([r.benchmark_metrics["sharpe_ratio"] for r in results]),
            "best_benchmark": max(results, key=lambda r: r.benchmark_metrics["sharpe_ratio"]).benchmark_name if results else None,
        }

        return ComparisonReport(
            strategy_name=strategy_name,
            strategy_returns=strategy_returns,
            benchmark_results=results,
            overall_ranking=strategy_rank,
            summary=summary,
        )

    def _calculate_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(returns) == 0:
            return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        total_return = np.prod(1 + returns) - 1

        # Max drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        max_drawdown = np.min(drawdowns)

        # Volatility
        volatility = std_return * np.sqrt(252)

        # Win rate
        win_rate = np.mean(returns > 0)

        return {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "annualized_return": (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "win_rate": win_rate,
            "avg_daily_return": mean_return,
            "n_observations": len(returns),
        }

    def _calculate_relative_metrics(
        self,
        strategy: Dict[str, float],
        benchmark: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate relative performance metrics."""
        return {
            "sharpe_difference": strategy["sharpe_ratio"] - benchmark["sharpe_ratio"],
            "return_difference": strategy["total_return"] - benchmark["total_return"],
            "drawdown_difference": strategy["max_drawdown"] - benchmark["max_drawdown"],
            "volatility_ratio": strategy["volatility"] / benchmark["volatility"] if benchmark["volatility"] > 0 else 0,
            "information_ratio": (strategy["annualized_return"] - benchmark["annualized_return"]) / abs(strategy["volatility"] - benchmark["volatility"]) if abs(strategy["volatility"] - benchmark["volatility"]) > 0.01 else 0,
        }


class AlphaCalculator:
    """Calculate strategy alpha relative to benchmarks."""

    def __init__(self, risk_free_rate: float = 0.04):
        """
        Initialize alpha calculator.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/252) - 1

    def calculate_capm_alpha(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate CAPM alpha (Jensen's alpha).

        Args:
            strategy_returns: Strategy returns
            market_returns: Market benchmark returns

        Returns:
            Dictionary with alpha and related metrics
        """
        excess_strategy = strategy_returns - self.daily_rf
        excess_market = market_returns - self.daily_rf

        # Calculate beta
        covariance = np.cov(excess_strategy, excess_market)[0, 1]
        market_variance = np.var(excess_market)
        beta = covariance / market_variance if market_variance > 0 else 0

        # Calculate alpha
        expected_return = self.daily_rf + beta * (np.mean(market_returns) - self.daily_rf)
        alpha = np.mean(strategy_returns) - expected_return

        # Annualize
        annualized_alpha = (1 + alpha) ** 252 - 1

        # Calculate R-squared
        predicted_returns = self.daily_rf + beta * excess_market
        ss_res = np.sum((strategy_returns - predicted_returns) ** 2)
        ss_tot = np.sum((strategy_returns - np.mean(strategy_returns)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            "alpha_daily": alpha,
            "alpha_annualized": annualized_alpha,
            "beta": beta,
            "r_squared": r_squared,
            "tracking_error": np.std(strategy_returns - predicted_returns) * np.sqrt(252),
        }

    def calculate_fama_french_alpha(
        self,
        strategy_returns: np.ndarray,
        market_returns: np.ndarray,
        smb_returns: Optional[np.ndarray] = None,
        hml_returns: Optional[np.ndarray] = None,
        mom_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate Fama-French alpha.

        Args:
            strategy_returns: Strategy returns
            market_returns: Market excess returns
            smb_returns: Small minus Big factor returns
            hml_returns: High minus Low factor returns
            mom_returns: Momentum factor returns

        Returns:
            Dictionary with alpha and factor exposures
        """
        excess_strategy = strategy_returns - self.daily_rf

        # Build factor matrix
        factors = [market_returns - self.daily_rf]
        factor_names = ["mkt"]

        if smb_returns is not None:
            factors.append(smb_returns)
            factor_names.append("smb")
        if hml_returns is not None:
            factors.append(hml_returns)
            factor_names.append("hml")
        if mom_returns is not None:
            factors.append(mom_returns)
            factor_names.append("mom")

        X = np.column_stack(factors)

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X_with_const, excess_strategy, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {"error": "Regression failed"}

        alpha = coeffs[0]
        betas = coeffs[1:]

        # Calculate R-squared
        predicted = X_with_const @ coeffs
        ss_res = np.sum((excess_strategy - predicted) ** 2)
        ss_tot = np.sum((excess_strategy - np.mean(excess_strategy)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        result = {
            "alpha_daily": alpha,
            "alpha_annualized": (1 + alpha) ** 252 - 1,
            "r_squared": r_squared,
        }

        for i, name in enumerate(factor_names):
            result[f"beta_{name}"] = betas[i]

        return result


def generate_benchmark_report(
    strategy_returns: np.ndarray,
    market_returns: np.ndarray,
    strategy_name: str = "Strategy",
    bond_returns: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive benchmark comparison report.

    Args:
        strategy_returns: Strategy return series
        market_returns: Market (S&P 500) returns
        strategy_name: Name of the strategy
        bond_returns: Optional bond returns for 60/40 benchmark

    Returns:
        Dictionary with full comparison report
    """
    comparator = BenchmarkComparator()
    comparison = comparator.compare(
        strategy_returns,
        market_returns,
        strategy_name,
        bond_returns=bond_returns,
    )

    alpha_calc = AlphaCalculator()
    alpha_metrics = alpha_calc.calculate_capm_alpha(strategy_returns, market_returns)

    return {
        "comparison": comparison.to_dict(),
        "alpha_metrics": alpha_metrics,
        "validation_passed": comparison.overall_ranking == 1,
    }
