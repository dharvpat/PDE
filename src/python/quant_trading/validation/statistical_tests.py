"""
Statistical Testing Utilities for Strategy Validation.

Provides statistical tests for:
- Strategy performance significance
- Return distribution analysis
- Overfitting detection
- Out-of-sample validation

Reference: Section 13 of design-doc.md
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import scipy, use mock if not available
try:
    from scipy import stats
    from scipy.stats import ttest_1samp, ttest_ind, wilcoxon, mannwhitneyu
    from scipy.stats import jarque_bera, shapiro, normaltest
    from scipy.stats import spearmanr, pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TestResult(Enum):
    """Result of a statistical test."""
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    INCONCLUSIVE = "inconclusive"


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""

    test_name: str
    statistic: float
    p_value: float
    result: TestResult
    confidence_level: float
    interpretation: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "result": self.result.value,
            "confidence_level": self.confidence_level,
            "interpretation": self.interpretation,
            "details": self.details,
        }


class StrategyStatisticalTests:
    """Statistical tests for trading strategy validation."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical tests.

        Args:
            confidence_level: Confidence level for tests (default 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def test_returns_significance(
        self,
        returns: np.ndarray,
        benchmark_mean: float = 0.0,
    ) -> StatisticalTestResult:
        """
        Test if strategy returns are significantly different from benchmark.

        Uses one-sample t-test to determine if mean return differs from benchmark.

        Args:
            returns: Array of strategy returns
            benchmark_mean: Benchmark mean return to test against

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("returns_significance")

        # Perform t-test
        statistic, p_value = ttest_1samp(returns, benchmark_mean)

        result = TestResult.SIGNIFICANT if p_value < self.alpha else TestResult.NOT_SIGNIFICANT

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        interpretation = (
            f"Mean return ({mean_return:.4f}) is "
            f"{'significantly' if result == TestResult.SIGNIFICANT else 'not significantly'} "
            f"different from {benchmark_mean:.4f} (p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Returns Significance (t-test)",
            statistic=statistic,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "mean_return": mean_return,
                "std_return": std_return,
                "benchmark_mean": benchmark_mean,
                "n_observations": len(returns),
            },
        )

    def test_sharpe_significance(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualization_factor: float = 252,
    ) -> StatisticalTestResult:
        """
        Test if Sharpe ratio is significantly greater than zero.

        Uses the Lo (2002) corrected standard error for Sharpe ratio.

        Args:
            returns: Array of strategy returns
            risk_free_rate: Risk-free rate (annualized)
            annualization_factor: Factor to annualize returns

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("sharpe_significance")

        n = len(returns)
        if n < 10:
            return StatisticalTestResult(
                test_name="Sharpe Ratio Significance",
                statistic=0.0,
                p_value=1.0,
                result=TestResult.INCONCLUSIVE,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for Sharpe ratio test",
                details={"n_observations": n},
            )

        # Calculate Sharpe ratio
        daily_rf = risk_free_rate / annualization_factor
        excess_returns = returns - daily_rf
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        sharpe = (mean_excess / std_excess) * np.sqrt(annualization_factor)

        # Lo (2002) standard error with autocorrelation correction
        # Simplified version without autocorrelation
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n) * np.sqrt(annualization_factor)

        # z-statistic
        z_stat = sharpe / se_sharpe
        p_value = 1 - stats.norm.cdf(z_stat)  # One-tailed test

        result = TestResult.SIGNIFICANT if p_value < self.alpha else TestResult.NOT_SIGNIFICANT

        interpretation = (
            f"Sharpe ratio ({sharpe:.2f}) is "
            f"{'significantly' if result == TestResult.SIGNIFICANT else 'not significantly'} "
            f"greater than 0 (p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Sharpe Ratio Significance",
            statistic=z_stat,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "sharpe_ratio": sharpe,
                "standard_error": se_sharpe,
                "n_observations": n,
            },
        )

    def test_returns_normality(
        self,
        returns: np.ndarray,
    ) -> StatisticalTestResult:
        """
        Test if returns follow a normal distribution.

        Uses Jarque-Bera test which checks skewness and kurtosis.

        Args:
            returns: Array of strategy returns

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("returns_normality")

        statistic, p_value = jarque_bera(returns)

        result = TestResult.NOT_SIGNIFICANT if p_value < self.alpha else TestResult.SIGNIFICANT

        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        interpretation = (
            f"Returns are {'non-normal' if result == TestResult.NOT_SIGNIFICANT else 'approximately normal'} "
            f"(skew={skewness:.2f}, kurt={kurtosis:.2f}, p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Returns Normality (Jarque-Bera)",
            statistic=statistic,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "skewness": skewness,
                "kurtosis": kurtosis,
                "n_observations": len(returns),
            },
        )

    def test_strategy_comparison(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        paired: bool = True,
    ) -> StatisticalTestResult:
        """
        Test if two strategies have significantly different performance.

        Args:
            returns_a: Returns of strategy A
            returns_b: Returns of strategy B
            paired: Whether observations are paired (same time periods)

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("strategy_comparison")

        if paired and len(returns_a) == len(returns_b):
            # Paired test (Wilcoxon signed-rank for non-normal data)
            statistic, p_value = wilcoxon(returns_a, returns_b)
            test_name = "Strategy Comparison (Wilcoxon)"
        else:
            # Unpaired test
            statistic, p_value = mannwhitneyu(returns_a, returns_b)
            test_name = "Strategy Comparison (Mann-Whitney U)"

        result = TestResult.SIGNIFICANT if p_value < self.alpha else TestResult.NOT_SIGNIFICANT

        mean_a = np.mean(returns_a)
        mean_b = np.mean(returns_b)

        interpretation = (
            f"Strategy A (mean={mean_a:.4f}) vs Strategy B (mean={mean_b:.4f}): "
            f"{'Significant' if result == TestResult.SIGNIFICANT else 'No significant'} difference "
            f"(p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "mean_a": mean_a,
                "mean_b": mean_b,
                "std_a": np.std(returns_a),
                "std_b": np.std(returns_b),
            },
        )

    def test_information_coefficient(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> StatisticalTestResult:
        """
        Test if prediction-actual correlation (IC) is significant.

        Uses Spearman rank correlation which is robust to outliers.

        Args:
            predictions: Predicted values/signals
            actuals: Actual realized values

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("information_coefficient")

        correlation, p_value = spearmanr(predictions, actuals)

        result = TestResult.SIGNIFICANT if p_value < self.alpha else TestResult.NOT_SIGNIFICANT

        interpretation = (
            f"Information Coefficient ({correlation:.4f}) is "
            f"{'significantly' if result == TestResult.SIGNIFICANT else 'not significantly'} "
            f"different from 0 (p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Information Coefficient (Spearman)",
            statistic=correlation,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "ic": correlation,
                "n_observations": len(predictions),
            },
        )

    def test_regime_stability(
        self,
        is_returns: np.ndarray,
        oos_returns: np.ndarray,
    ) -> StatisticalTestResult:
        """
        Test if strategy performance is stable across regimes (IS vs OOS).

        Uses two-sample test to compare in-sample and out-of-sample performance.

        Args:
            is_returns: In-sample returns
            oos_returns: Out-of-sample returns

        Returns:
            StatisticalTestResult with test outcome
        """
        if not SCIPY_AVAILABLE:
            return self._mock_test_result("regime_stability")

        # Use Welch's t-test (doesn't assume equal variances)
        statistic, p_value = ttest_ind(is_returns, oos_returns, equal_var=False)

        # We want NO significant difference for stability
        result = TestResult.SIGNIFICANT if p_value >= self.alpha else TestResult.NOT_SIGNIFICANT

        is_sharpe = np.mean(is_returns) / np.std(is_returns) * np.sqrt(252)
        oos_sharpe = np.mean(oos_returns) / np.std(oos_returns) * np.sqrt(252)
        degradation = 1 - (oos_sharpe / is_sharpe) if is_sharpe != 0 else 0

        interpretation = (
            f"Performance {'stable' if result == TestResult.SIGNIFICANT else 'degraded'} across regimes. "
            f"IS Sharpe: {is_sharpe:.2f}, OOS Sharpe: {oos_sharpe:.2f} "
            f"(degradation: {degradation*100:.1f}%, p={p_value:.4f})"
        )

        return StatisticalTestResult(
            test_name="Regime Stability (Welch's t-test)",
            statistic=statistic,
            p_value=p_value,
            result=result,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            details={
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "degradation": degradation,
                "is_mean": np.mean(is_returns),
                "oos_mean": np.mean(oos_returns),
            },
        )

    def _mock_test_result(self, test_name: str) -> StatisticalTestResult:
        """Return mock result when scipy is not available."""
        return StatisticalTestResult(
            test_name=test_name,
            statistic=0.0,
            p_value=1.0,
            result=TestResult.INCONCLUSIVE,
            confidence_level=self.confidence_level,
            interpretation="scipy not available for statistical testing",
            details={"scipy_available": False},
        )


class OverfittingDetector:
    """Detect potential overfitting in trading strategies."""

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level

    def deflated_sharpe_ratio(
        self,
        sharpe_observed: float,
        n_trials: int,
        n_observations: int,
        expected_max_sharpe: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate Deflated Sharpe Ratio per Bailey and Lopez de Prado (2014).

        Adjusts Sharpe ratio for multiple testing (data snooping).

        Args:
            sharpe_observed: Observed Sharpe ratio
            n_trials: Number of strategy variations tested
            n_observations: Number of return observations
            expected_max_sharpe: Expected max Sharpe under null

        Returns:
            Dictionary with DSR and probability of overfitting
        """
        if not SCIPY_AVAILABLE:
            return {
                "deflated_sharpe": sharpe_observed,
                "probability_overfit": 0.5,
                "haircut": 0.0,
            }

        # Estimate expected max Sharpe under null if not provided
        if expected_max_sharpe is None:
            # Bailey-Lopez de Prado formula
            gamma = 0.5772156649  # Euler-Mascheroni constant
            expected_max_sharpe = (
                (1 - gamma) * stats.norm.ppf(1 - 1/n_trials) +
                gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))
            )

        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe_observed**2) / n_observations)

        # Deflated Sharpe Ratio
        dsr = stats.norm.cdf(
            (sharpe_observed - expected_max_sharpe) / se_sharpe
        )

        # Probability of overfitting
        prob_overfit = 1 - dsr

        # Haircut (% reduction in expected performance)
        haircut = 1 - (sharpe_observed - expected_max_sharpe) / sharpe_observed if sharpe_observed > 0 else 0

        return {
            "deflated_sharpe": dsr,
            "probability_overfit": prob_overfit,
            "expected_max_sharpe": expected_max_sharpe,
            "haircut": max(0, min(1, haircut)),
            "n_trials": n_trials,
            "n_observations": n_observations,
        }

    def probability_of_backtest_overfitting(
        self,
        is_sharpes: np.ndarray,
        oos_sharpes: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate Probability of Backtest Overfitting (PBO).

        Per Bailey et al. (2014) - "The Probability of Backtest Overfitting"

        Args:
            is_sharpes: In-sample Sharpe ratios for different strategy configs
            oos_sharpes: Out-of-sample Sharpe ratios for same configs

        Returns:
            Dictionary with PBO and related metrics
        """
        if len(is_sharpes) != len(oos_sharpes):
            raise ValueError("IS and OOS Sharpe arrays must have same length")

        n = len(is_sharpes)

        # Find best IS strategy
        best_is_idx = np.argmax(is_sharpes)
        best_is_sharpe = is_sharpes[best_is_idx]
        corresponding_oos = oos_sharpes[best_is_idx]

        # Count how many OOS > best IS's OOS
        n_better = np.sum(oos_sharpes > corresponding_oos)
        pbo = n_better / n

        # Rank correlation between IS and OOS
        if SCIPY_AVAILABLE:
            rank_corr, _ = spearmanr(is_sharpes, oos_sharpes)
        else:
            rank_corr = np.corrcoef(is_sharpes, oos_sharpes)[0, 1]

        return {
            "pbo": pbo,
            "is_oos_correlation": rank_corr,
            "best_is_sharpe": best_is_sharpe,
            "best_is_oos_sharpe": corresponding_oos,
            "n_strategies": n,
            "interpretation": "Low" if pbo < 0.25 else "Medium" if pbo < 0.5 else "High",
        }

    def combinatorial_purged_cross_validation_score(
        self,
        returns: np.ndarray,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 0,
    ) -> Dict[str, float]:
        """
        Estimate strategy performance using CPCV.

        Combinatorial Purged Cross-Validation prevents information leakage.

        Args:
            returns: Array of returns
            n_splits: Number of splits
            n_test_splits: Number of test splits per combination
            purge_gap: Gap between train and test to prevent leakage

        Returns:
            Dictionary with CPCV metrics
        """
        from itertools import combinations

        n = len(returns)
        split_size = n // n_splits
        split_indices = [list(range(i * split_size, (i + 1) * split_size))
                        for i in range(n_splits)]

        # Handle remainder
        if n % n_splits > 0:
            split_indices[-1].extend(range(n_splits * split_size, n))

        # Generate all test combinations
        test_combos = list(combinations(range(n_splits), n_test_splits))
        sharpes = []

        for test_splits in test_combos:
            train_splits = [i for i in range(n_splits) if i not in test_splits]

            # Get train and test indices with purging
            test_idx = []
            for ts in test_splits:
                test_idx.extend(split_indices[ts])

            train_idx = []
            for ts in train_splits:
                # Purge observations near test set
                split_start = split_indices[ts][0]
                split_end = split_indices[ts][-1]

                # Check proximity to any test split
                min_test_start = min(split_indices[t][0] for t in test_splits)
                max_test_end = max(split_indices[t][-1] for t in test_splits)

                if split_end < min_test_start - purge_gap or split_start > max_test_end + purge_gap:
                    train_idx.extend(split_indices[ts])

            if len(train_idx) > 10 and len(test_idx) > 10:
                test_returns = returns[test_idx]
                sharpe = np.mean(test_returns) / np.std(test_returns) * np.sqrt(252)
                sharpes.append(sharpe)

        if not sharpes:
            return {"cpcv_sharpe": 0.0, "cpcv_std": 0.0, "n_combinations": 0}

        return {
            "cpcv_sharpe": np.mean(sharpes),
            "cpcv_std": np.std(sharpes),
            "cpcv_min": np.min(sharpes),
            "cpcv_max": np.max(sharpes),
            "n_combinations": len(sharpes),
        }


class BootstrapAnalysis:
    """Bootstrap methods for confidence intervals and significance."""

    def __init__(self, n_bootstrap: int = 1000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.RandomState(random_state)

    def sharpe_confidence_interval(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence interval for Sharpe ratio.

        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Dictionary with Sharpe ratio and CI bounds
        """
        n = len(returns)
        sharpes = []

        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(returns, size=n, replace=True)
            sharpe = np.mean(sample) / np.std(sample) * np.sqrt(252)
            sharpes.append(sharpe)

        sharpes = np.array(sharpes)
        alpha = 1 - confidence_level
        lower = np.percentile(sharpes, alpha / 2 * 100)
        upper = np.percentile(sharpes, (1 - alpha / 2) * 100)
        point_estimate = np.mean(returns) / np.std(returns) * np.sqrt(252)

        return {
            "sharpe_ratio": point_estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": confidence_level,
            "bootstrap_mean": np.mean(sharpes),
            "bootstrap_std": np.std(sharpes),
        }

    def max_drawdown_confidence_interval(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calculate bootstrap CI for maximum drawdown.

        Args:
            returns: Array of returns
            confidence_level: Confidence level

        Returns:
            Dictionary with max drawdown and CI bounds
        """
        def calculate_max_drawdown(rets: np.ndarray) -> float:
            """Calculate maximum drawdown from returns."""
            cum_returns = np.cumprod(1 + rets)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns / running_max - 1
            return np.min(drawdowns)

        n = len(returns)
        max_dds = []

        for _ in range(self.n_bootstrap):
            sample = self.rng.choice(returns, size=n, replace=True)
            max_dd = calculate_max_drawdown(sample)
            max_dds.append(max_dd)

        max_dds = np.array(max_dds)
        alpha = 1 - confidence_level
        lower = np.percentile(max_dds, alpha / 2 * 100)
        upper = np.percentile(max_dds, (1 - alpha / 2) * 100)
        point_estimate = calculate_max_drawdown(returns)

        return {
            "max_drawdown": point_estimate,
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence_level": confidence_level,
            "bootstrap_mean": np.mean(max_dds),
            "bootstrap_std": np.std(max_dds),
        }
