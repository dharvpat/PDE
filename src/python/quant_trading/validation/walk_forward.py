"""
Walk-Forward Validation for Trading Strategies.

Implements proper walk-forward analysis to prevent look-ahead bias:
- Rolling window optimization
- Anchored walk-forward
- Purged cross-validation
- Out-of-sample testing

Reference: Section 13.2 of design-doc.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class WalkForwardType(Enum):
    """Types of walk-forward analysis."""
    ROLLING = "rolling"
    ANCHORED = "anchored"
    EXPANDING = "expanding"


@dataclass
class WalkForwardWindow:
    """Definition of a walk-forward window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    window_id: int

    @property
    def train_size(self) -> int:
        """Size of training window."""
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        """Size of test window."""
        return self.test_end - self.test_start


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward window."""

    window: WalkForwardWindow
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    optimized_params: Dict[str, Any]
    test_returns: np.ndarray
    test_signals: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_id": self.window.window_id,
            "train_start": self.window.train_start,
            "train_end": self.window.train_end,
            "test_start": self.window.test_start,
            "test_end": self.window.test_end,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "optimized_params": self.optimized_params,
        }


@dataclass
class WalkForwardReport:
    """Comprehensive walk-forward analysis report."""

    strategy_name: str
    walk_forward_type: WalkForwardType
    total_windows: int
    results: List[WalkForwardResult]
    aggregated_metrics: Dict[str, float]
    all_oos_returns: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_name": self.strategy_name,
            "walk_forward_type": self.walk_forward_type.value,
            "total_windows": self.total_windows,
            "results": [r.to_dict() for r in self.results],
            "aggregated_metrics": self.aggregated_metrics,
            "timestamp": self.timestamp.isoformat(),
        }


class WalkForwardOptimizer:
    """Walk-forward optimization framework."""

    def __init__(
        self,
        train_period: int,
        test_period: int,
        walk_forward_type: WalkForwardType = WalkForwardType.ROLLING,
        purge_gap: int = 0,
        embargo_period: int = 0,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_period: Number of observations in training period
            test_period: Number of observations in test period
            walk_forward_type: Type of walk-forward analysis
            purge_gap: Gap between train and test to prevent leakage
            embargo_period: Additional gap after test for rolling windows
        """
        self.train_period = train_period
        self.test_period = test_period
        self.walk_forward_type = walk_forward_type
        self.purge_gap = purge_gap
        self.embargo_period = embargo_period

    def generate_windows(self, n_observations: int) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Args:
            n_observations: Total number of observations

        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        window_id = 0

        if self.walk_forward_type == WalkForwardType.ROLLING:
            # Rolling window: fixed train size, moves forward
            start = 0
            while start + self.train_period + self.purge_gap + self.test_period <= n_observations:
                train_start = start
                train_end = start + self.train_period
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_period

                windows.append(WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                ))
                window_id += 1
                start += self.test_period + self.embargo_period

        elif self.walk_forward_type == WalkForwardType.ANCHORED:
            # Anchored: train always starts from beginning, expands
            train_start = 0
            train_end = self.train_period

            while train_end + self.purge_gap + self.test_period <= n_observations:
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_period

                windows.append(WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                ))
                window_id += 1
                train_end = test_end

        elif self.walk_forward_type == WalkForwardType.EXPANDING:
            # Expanding: train grows, test fixed
            train_end = self.train_period

            while train_end + self.purge_gap + self.test_period <= n_observations:
                test_start = train_end + self.purge_gap
                test_end = test_start + self.test_period

                windows.append(WalkForwardWindow(
                    train_start=0,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    window_id=window_id,
                ))
                window_id += 1
                train_end += self.test_period

        return windows

    def run(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray],
        optimize_fn: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, Any]],
        evaluate_fn: Callable[[np.ndarray, Optional[np.ndarray], Dict[str, Any]], Tuple[np.ndarray, Dict[str, float]]],
        strategy_name: str = "Strategy",
    ) -> WalkForwardReport:
        """
        Run walk-forward optimization.

        Args:
            returns: Full array of returns
            features: Optional features/signals array
            optimize_fn: Function to optimize parameters on train data
                         Takes (train_returns, train_features) -> params dict
            evaluate_fn: Function to evaluate strategy on test data
                         Takes (test_returns, test_features, params) -> (signals, metrics)
            strategy_name: Name of the strategy

        Returns:
            WalkForwardReport with all results
        """
        windows = self.generate_windows(len(returns))
        results = []
        all_oos_returns = []

        for window in windows:
            # Extract train data
            train_returns = returns[window.train_start:window.train_end]
            train_features = features[window.train_start:window.train_end] if features is not None else None

            # Optimize on train data
            optimized_params = optimize_fn(train_returns, train_features)

            # Calculate train metrics
            train_signals, train_metrics = evaluate_fn(train_returns, train_features, optimized_params)

            # Extract test data
            test_returns = returns[window.test_start:window.test_end]
            test_features = features[window.test_start:window.test_end] if features is not None else None

            # Evaluate on test data
            test_signals, test_metrics = evaluate_fn(test_returns, test_features, optimized_params)

            # Store result
            result = WalkForwardResult(
                window=window,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                optimized_params=optimized_params,
                test_returns=test_returns,
                test_signals=test_signals,
            )
            results.append(result)
            all_oos_returns.extend(test_returns.tolist())

        # Aggregate metrics
        all_oos_returns = np.array(all_oos_returns)
        aggregated = self._aggregate_metrics(results, all_oos_returns)

        return WalkForwardReport(
            strategy_name=strategy_name,
            walk_forward_type=self.walk_forward_type,
            total_windows=len(windows),
            results=results,
            aggregated_metrics=aggregated,
            all_oos_returns=all_oos_returns,
        )

    def _aggregate_metrics(
        self,
        results: List[WalkForwardResult],
        all_oos_returns: np.ndarray,
    ) -> Dict[str, float]:
        """Aggregate metrics across all windows."""
        # Collect all test metrics
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0) for r in results]
        train_sharpes = [r.train_metrics.get("sharpe_ratio", 0) for r in results]

        # Calculate overall OOS performance
        if len(all_oos_returns) > 0 and np.std(all_oos_returns) > 0:
            overall_sharpe = np.mean(all_oos_returns) / np.std(all_oos_returns) * np.sqrt(252)
            cum_return = np.prod(1 + all_oos_returns) - 1

            # Calculate max drawdown
            cum_returns = np.cumprod(1 + all_oos_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns / running_max - 1
            max_drawdown = np.min(drawdowns)
        else:
            overall_sharpe = 0
            cum_return = 0
            max_drawdown = 0

        return {
            "oos_sharpe_mean": np.mean(test_sharpes),
            "oos_sharpe_std": np.std(test_sharpes),
            "oos_sharpe_min": np.min(test_sharpes),
            "oos_sharpe_max": np.max(test_sharpes),
            "is_sharpe_mean": np.mean(train_sharpes),
            "is_oos_sharpe_ratio": np.mean(test_sharpes) / np.mean(train_sharpes) if np.mean(train_sharpes) != 0 else 0,
            "overall_oos_sharpe": overall_sharpe,
            "overall_oos_return": cum_return,
            "overall_max_drawdown": max_drawdown,
            "n_positive_oos_windows": sum(1 for s in test_sharpes if s > 0),
            "pct_positive_oos_windows": sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) if test_sharpes else 0,
        }


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation.

    Implements proper time-series cross-validation with:
    - Purging: Remove observations that could cause leakage
    - Embargo: Additional gap between train and test
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ):
        """
        Initialize purged k-fold.

        Args:
            n_splits: Number of folds
            purge_gap: Number of observations to purge between train/test
            embargo_pct: Percentage of test size to add as embargo
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        n_observations: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.

        Args:
            n_observations: Total number of observations

        Returns:
            List of (train_indices, test_indices) tuples
        """
        fold_size = n_observations // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_observations
            embargo_size = int(fold_size * self.embargo_pct)

            test_indices = np.arange(test_start, test_end)

            # Train indices: all except test and purge/embargo zones
            train_indices = []

            # Before test fold (with purge)
            if test_start > 0:
                train_end_before = max(0, test_start - self.purge_gap)
                train_indices.extend(range(0, train_end_before))

            # After test fold (with embargo)
            if test_end < n_observations:
                train_start_after = min(n_observations, test_end + embargo_size)
                train_indices.extend(range(train_start_after, n_observations))

            train_indices = np.array(train_indices)

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


class OutOfSampleValidator:
    """
    Validate strategy out-of-sample performance.

    Implements the validation requirements from Section 13.2:
    - In-sample period: 2018-2021
    - Out-of-sample period: 2022-2024
    - Validate: OOS Sharpe >= 50% of IS Sharpe
    """

    def __init__(
        self,
        is_start_idx: int,
        is_end_idx: int,
        oos_start_idx: int,
        oos_end_idx: int,
        min_oos_sharpe_ratio: float = 0.5,
    ):
        """
        Initialize out-of-sample validator.

        Args:
            is_start_idx: Start index of in-sample period
            is_end_idx: End index of in-sample period
            oos_start_idx: Start index of out-of-sample period
            oos_end_idx: End index of out-of-sample period
            min_oos_sharpe_ratio: Minimum OOS/IS Sharpe ratio required
        """
        self.is_start = is_start_idx
        self.is_end = is_end_idx
        self.oos_start = oos_start_idx
        self.oos_end = oos_end_idx
        self.min_oos_sharpe_ratio = min_oos_sharpe_ratio

    def validate(
        self,
        returns: np.ndarray,
        features: Optional[np.ndarray],
        optimize_fn: Callable[[np.ndarray, Optional[np.ndarray]], Dict[str, Any]],
        evaluate_fn: Callable[[np.ndarray, Optional[np.ndarray], Dict[str, Any]], Tuple[np.ndarray, Dict[str, float]]],
    ) -> Dict[str, Any]:
        """
        Run out-of-sample validation.

        Args:
            returns: Full array of returns
            features: Optional features array
            optimize_fn: Optimization function
            evaluate_fn: Evaluation function

        Returns:
            Dictionary with validation results
        """
        # Extract IS and OOS data
        is_returns = returns[self.is_start:self.is_end]
        oos_returns = returns[self.oos_start:self.oos_end]

        is_features = features[self.is_start:self.is_end] if features is not None else None
        oos_features = features[self.oos_start:self.oos_end] if features is not None else None

        # Optimize on IS data
        params = optimize_fn(is_returns, is_features)

        # Evaluate on IS data
        is_signals, is_metrics = evaluate_fn(is_returns, is_features, params)

        # Evaluate on OOS data with FROZEN parameters
        oos_signals, oos_metrics = evaluate_fn(oos_returns, oos_features, params)

        # Calculate validation metrics
        is_sharpe = is_metrics.get("sharpe_ratio", 0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        sharpe_ratio = oos_sharpe / is_sharpe if is_sharpe != 0 else 0

        validation_passed = sharpe_ratio >= self.min_oos_sharpe_ratio

        return {
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "sharpe_ratio": sharpe_ratio,
            "min_required_ratio": self.min_oos_sharpe_ratio,
            "validation_passed": validation_passed,
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "optimized_params": params,
            "is_n_observations": len(is_returns),
            "oos_n_observations": len(oos_returns),
            "degradation_pct": (1 - sharpe_ratio) * 100,
        }


def calculate_performance_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate common performance metrics.

    Args:
        returns: Array of returns

    Returns:
        Dictionary with performance metrics
    """
    if len(returns) == 0:
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

    # Annualized Sharpe ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

    # Total return
    total_return = np.prod(1 + returns) - 1

    # Max drawdown
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    max_drawdown = np.min(drawdowns)

    # Win rate
    win_rate = np.mean(returns > 0)

    # Profit factor
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = np.sum(gains) / abs(np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else np.inf

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino = mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Calmar ratio
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

    return {
        "sharpe_ratio": sharpe,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "volatility": std_return * np.sqrt(252),
        "n_observations": len(returns),
    }
