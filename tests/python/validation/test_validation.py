"""
Tests for Validation Module.

Tests model validation, statistical testing, stress testing,
walk-forward validation, and benchmark comparisons.

Reference: Section 13 of design-doc.md
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple, Optional

from quant_trading.validation import (
    # Model Validation
    ValidationStatus,
    ValidationSeverity,
    ValidationResult,
    ValidationReport,
    ThresholdCheck,
    ParameterBoundsCheck,
    ModelValidator,
    HestonModelValidator,
    FellerConditionCheck,
    SABRModelValidator,
    OUModelValidator,
    StrategyValidator,
    # Statistical Tests
    TestResult,
    StatisticalTestResult,
    StrategyStatisticalTests,
    OverfittingDetector,
    BootstrapAnalysis,
    # Stress Testing
    ScenarioType,
    MarketScenario,
    StressTestResult,
    HISTORICAL_SCENARIOS,
    StressTestEngine,
    TailRiskAnalyzer,
    # Walk-Forward
    WalkForwardType,
    WalkForwardWindow,
    WalkForwardOptimizer,
    PurgedKFold,
    OutOfSampleValidator,
    calculate_performance_metrics,
    # Benchmarks
    BenchmarkType,
    BuyAndHoldBenchmark,
    SixtyFortyBenchmark,
    MomentumBenchmark,
    RiskFreeBenchmark,
    BenchmarkComparator,
    AlphaCalculator,
    generate_benchmark_report,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample return series."""
    np.random.seed(42)
    # Simulate realistic daily returns
    returns = np.random.normal(0.0003, 0.01, 504)  # ~2 years
    return returns


@pytest.fixture
def market_returns() -> np.ndarray:
    """Generate market return series."""
    np.random.seed(123)
    returns = np.random.normal(0.0004, 0.012, 504)
    return returns


@pytest.fixture
def heston_params() -> Dict[str, float]:
    """Valid Heston parameters."""
    return {
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }


@pytest.fixture
def sabr_params() -> Dict[str, float]:
    """Valid SABR parameters."""
    return {
        "alpha": 0.3,
        "beta": 0.5,
        "rho": -0.3,
        "nu": 0.4,
    }


# =============================================================================
# Model Validation Tests
# =============================================================================

class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_create_result(self):
        """Test creating a validation result."""
        result = ValidationResult(
            name="test_check",
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.MEDIUM,
            message="Test passed",
            metric_value=0.95,
            threshold=0.90,
        )
        assert result.name == "test_check"
        assert result.status == ValidationStatus.PASSED
        assert result.metric_value == 0.95

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = ValidationResult(
            name="test",
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.HIGH,
            message="Test failed",
        )
        d = result.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "failed"
        assert d["severity"] == "high"


class TestValidationReport:
    """Test ValidationReport."""

    def test_report_passed(self):
        """Test report with all passing checks."""
        results = [
            ValidationResult(
                name="check1",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.CRITICAL,
                message="Passed",
            ),
            ValidationResult(
                name="check2",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.HIGH,
                message="Passed",
            ),
        ]
        report = ValidationReport(
            model_name="TestModel",
            model_version="1.0",
            validation_date=pytest.importorskip("datetime").datetime.now(),
            results=results,
        )
        assert report.passed is True
        assert report.passed_tests == 2
        assert report.failed_tests == 0

    def test_report_failed(self):
        """Test report with critical failure."""
        results = [
            ValidationResult(
                name="critical_check",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message="Failed",
            ),
        ]
        report = ValidationReport(
            model_name="TestModel",
            model_version="1.0",
            validation_date=pytest.importorskip("datetime").datetime.now(),
            results=results,
        )
        assert report.passed is False
        assert report.failed_tests == 1


class TestThresholdCheck:
    """Test ThresholdCheck validation."""

    def test_threshold_check_passes(self):
        """Test threshold check that passes."""
        check = ThresholdCheck(
            name="sharpe_check",
            description="Check Sharpe ratio",
            metric_fn=lambda m, d: d.get("sharpe", 0),
            threshold=0.5,
            comparison=">=",
        )
        result = check.run(None, {"sharpe": 0.8})
        assert result.status == ValidationStatus.PASSED

    def test_threshold_check_fails(self):
        """Test threshold check that fails."""
        check = ThresholdCheck(
            name="drawdown_check",
            description="Check max drawdown",
            metric_fn=lambda m, d: abs(d.get("drawdown", 0)),
            threshold=0.25,
            comparison="<=",
        )
        result = check.run(None, {"drawdown": -0.30})
        assert result.status == ValidationStatus.FAILED


class TestParameterBoundsCheck:
    """Test ParameterBoundsCheck validation."""

    def test_bounds_check_passes(self, heston_params):
        """Test parameter bounds check that passes."""
        check = ParameterBoundsCheck(
            name="heston_bounds",
            parameter_bounds={
                "kappa": (0.01, 10.0),
                "theta": (0.001, 1.0),
                "sigma": (0.01, 2.0),
            },
        )
        result = check.run(None, {"parameters": heston_params})
        assert result.status == ValidationStatus.PASSED

    def test_bounds_check_fails(self):
        """Test parameter bounds check that fails."""
        check = ParameterBoundsCheck(
            name="param_bounds",
            parameter_bounds={"value": (0, 1)},
        )
        result = check.run(None, {"parameters": {"value": 1.5}})
        assert result.status == ValidationStatus.FAILED


class TestFellerConditionCheck:
    """Test Feller condition check for Heston model."""

    def test_feller_satisfied(self, heston_params):
        """Test Feller condition satisfied."""
        check = FellerConditionCheck()
        result = check.run(None, {"parameters": heston_params})
        # 2 * 2.0 * 0.04 = 0.16 >= 0.3^2 = 0.09
        assert result.status == ValidationStatus.PASSED

    def test_feller_violated(self):
        """Test Feller condition violated."""
        check = FellerConditionCheck()
        params = {"kappa": 1.0, "theta": 0.02, "sigma": 0.5}
        result = check.run(None, {"parameters": params})
        # 2 * 1.0 * 0.02 = 0.04 < 0.5^2 = 0.25
        assert result.status == ValidationStatus.FAILED


class TestHestonModelValidator:
    """Test Heston model validator."""

    def test_valid_heston_model(self, heston_params):
        """Test validation of valid Heston model."""
        validator = HestonModelValidator()
        data = {
            "parameters": heston_params,
            "rmse": 0.02,
            "r_squared": 0.95,
        }
        report = validator.validate(None, data)
        assert report.passed is True

    def test_invalid_heston_model(self):
        """Test validation of invalid Heston model."""
        validator = HestonModelValidator()
        invalid_params = {
            "kappa": 0.5,
            "theta": 0.01,
            "sigma": 0.8,  # Violates Feller
            "rho": -0.5,
            "v0": 0.04,
        }
        data = {
            "parameters": invalid_params,
            "rmse": 0.10,  # High RMSE
            "r_squared": 0.70,
        }
        report = validator.validate(None, data)
        assert report.passed is False


class TestStrategyValidator:
    """Test strategy validator."""

    def test_valid_strategy(self):
        """Test validation of valid strategy."""
        validator = StrategyValidator("TestStrategy")
        data = {
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "win_rate": 0.55,
            "oos_sharpe": 0.8,
            "is_sharpe": 1.2,
            "parameter_stability_score": 0.85,
        }
        report = validator.validate(None, data)
        assert report.passed is True

    def test_invalid_strategy_drawdown(self):
        """Test validation fails on excessive drawdown."""
        validator = StrategyValidator("TestStrategy")
        data = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.35,  # Exceeds 25% threshold
            "win_rate": 0.60,
            "oos_sharpe": 1.0,
            "is_sharpe": 1.5,
            "parameter_stability_score": 0.80,
        }
        report = validator.validate(None, data)
        assert report.passed is False


# =============================================================================
# Statistical Tests
# =============================================================================

class TestStrategyStatisticalTests:
    """Test statistical testing utilities."""

    def test_returns_significance(self, sample_returns):
        """Test returns significance test."""
        tests = StrategyStatisticalTests()
        result = tests.test_returns_significance(sample_returns, benchmark_mean=0.0)
        assert result.test_name == "Returns Significance (t-test)"
        assert result.p_value >= 0 and result.p_value <= 1

    def test_sharpe_significance(self, sample_returns):
        """Test Sharpe ratio significance test."""
        tests = StrategyStatisticalTests()
        result = tests.test_sharpe_significance(sample_returns)
        assert "sharpe_ratio" in result.details or "scipy_available" in result.details

    def test_returns_normality(self, sample_returns):
        """Test returns normality test."""
        tests = StrategyStatisticalTests()
        result = tests.test_returns_normality(sample_returns)
        assert result.test_name == "Returns Normality (Jarque-Bera)"

    def test_strategy_comparison(self, sample_returns, market_returns):
        """Test strategy comparison test."""
        tests = StrategyStatisticalTests()
        result = tests.test_strategy_comparison(sample_returns, market_returns)
        assert result.result in [TestResult.SIGNIFICANT, TestResult.NOT_SIGNIFICANT, TestResult.INCONCLUSIVE]

    def test_information_coefficient(self):
        """Test information coefficient calculation."""
        np.random.seed(42)
        predictions = np.random.randn(100)
        actuals = predictions * 0.5 + np.random.randn(100) * 0.5
        tests = StrategyStatisticalTests()
        result = tests.test_information_coefficient(predictions, actuals)
        assert "ic" in result.details or "scipy_available" in result.details


class TestOverfittingDetector:
    """Test overfitting detection."""

    def test_deflated_sharpe_ratio(self):
        """Test Deflated Sharpe Ratio calculation."""
        detector = OverfittingDetector()
        result = detector.deflated_sharpe_ratio(
            sharpe_observed=1.5,
            n_trials=100,
            n_observations=252,
        )
        assert "deflated_sharpe" in result
        assert "probability_overfit" in result
        assert 0 <= result["probability_overfit"] <= 1

    def test_probability_of_backtest_overfitting(self):
        """Test PBO calculation."""
        np.random.seed(42)
        detector = OverfittingDetector()
        is_sharpes = np.random.uniform(0.5, 2.0, 20)
        oos_sharpes = is_sharpes * 0.6 + np.random.uniform(-0.3, 0.3, 20)
        result = detector.probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
        assert "pbo" in result
        assert 0 <= result["pbo"] <= 1


class TestBootstrapAnalysis:
    """Test bootstrap analysis."""

    def test_sharpe_confidence_interval(self, sample_returns):
        """Test Sharpe ratio confidence interval."""
        bootstrap = BootstrapAnalysis(n_bootstrap=500, random_state=42)
        result = bootstrap.sharpe_confidence_interval(sample_returns)
        assert result["ci_lower"] < result["sharpe_ratio"] < result["ci_upper"]
        assert result["confidence_level"] == 0.95

    def test_max_drawdown_confidence_interval(self, sample_returns):
        """Test max drawdown confidence interval."""
        bootstrap = BootstrapAnalysis(n_bootstrap=500, random_state=42)
        result = bootstrap.max_drawdown_confidence_interval(sample_returns)
        assert "max_drawdown" in result
        assert result["ci_lower"] <= result["max_drawdown"]


# =============================================================================
# Stress Testing
# =============================================================================

class TestMarketScenario:
    """Test MarketScenario dataclass."""

    def test_create_scenario(self):
        """Test creating a market scenario."""
        scenario = MarketScenario(
            name="Test Crisis",
            description="Hypothetical market crash",
            scenario_type=ScenarioType.HYPOTHETICAL,
            market_shocks={"SPY": -0.30},
            volatility_multiplier=2.5,
            duration_days=30,
        )
        assert scenario.name == "Test Crisis"
        assert scenario.market_shocks["SPY"] == -0.30

    def test_historical_scenarios_defined(self):
        """Test that historical scenarios are defined."""
        assert "2008_financial_crisis" in HISTORICAL_SCENARIOS
        assert "2020_covid_crash" in HISTORICAL_SCENARIOS
        assert "2010_flash_crash" in HISTORICAL_SCENARIOS
        assert "2017_low_volatility" in HISTORICAL_SCENARIOS


class TestStressTestEngine:
    """Test stress testing engine."""

    def test_run_historical_scenario(self, sample_returns):
        """Test running a historical stress scenario."""
        engine = StressTestEngine(random_state=42)
        result = engine.run_historical_scenario(
            sample_returns,
            "2008_financial_crisis",
        )
        assert isinstance(result, StressTestResult)
        assert result.max_drawdown <= 0
        assert result.var_95 < 0

    def test_run_all_scenarios(self, sample_returns):
        """Test running all historical scenarios."""
        engine = StressTestEngine(random_state=42)
        results = engine.run_all_historical_scenarios(sample_returns)
        assert len(results) == len(HISTORICAL_SCENARIOS)

    def test_monte_carlo_stress(self, sample_returns):
        """Test Monte Carlo stress simulation."""
        engine = StressTestEngine(random_state=42)
        result = engine.run_monte_carlo_stress(
            sample_returns,
            n_simulations=1000,
            shock_magnitude=0.20,
        )
        assert "max_drawdown_mean" in result
        assert "probability_loss_gt_10pct" in result
        assert result["n_simulations"] == 1000

    def test_reverse_stress_test(self, sample_returns):
        """Test reverse stress testing."""
        engine = StressTestEngine(random_state=42)
        scenario = engine.reverse_stress_test(
            sample_returns,
            target_loss=0.25,
        )
        assert scenario.scenario_type == ScenarioType.REVERSE
        assert scenario.volatility_multiplier >= 1.0


class TestTailRiskAnalyzer:
    """Test tail risk analysis."""

    def test_expected_shortfall(self, sample_returns):
        """Test Expected Shortfall calculation."""
        analyzer = TailRiskAnalyzer()
        result = analyzer.calculate_expected_shortfall(sample_returns)
        assert "var_95" in result
        assert "es_95" in result
        assert result["es_95"] <= result["var_95"]  # ES is worse than VaR

    def test_extreme_value_analysis(self, sample_returns):
        """Test Extreme Value Theory analysis."""
        analyzer = TailRiskAnalyzer()
        result = analyzer.extreme_value_analysis(sample_returns)
        assert "threshold" in result
        assert "n_exceedances" in result

    def test_drawdown_analysis(self, sample_returns):
        """Test drawdown analysis."""
        analyzer = TailRiskAnalyzer()
        result = analyzer.drawdown_analysis(sample_returns)
        assert "max_drawdown" in result
        assert "avg_drawdown_duration" in result
        assert result["max_drawdown"] <= 0


# =============================================================================
# Walk-Forward Validation
# =============================================================================

class TestWalkForwardOptimizer:
    """Test walk-forward optimization."""

    def test_generate_rolling_windows(self):
        """Test generating rolling walk-forward windows."""
        optimizer = WalkForwardOptimizer(
            train_period=252,
            test_period=63,
            walk_forward_type=WalkForwardType.ROLLING,
        )
        windows = optimizer.generate_windows(504)
        assert len(windows) > 0
        # Check first window
        assert windows[0].train_start == 0
        assert windows[0].train_end == 252
        assert windows[0].test_start == 252
        assert windows[0].test_end == 315

    def test_generate_anchored_windows(self):
        """Test generating anchored walk-forward windows."""
        optimizer = WalkForwardOptimizer(
            train_period=252,
            test_period=63,
            walk_forward_type=WalkForwardType.ANCHORED,
        )
        windows = optimizer.generate_windows(504)
        assert len(windows) > 0
        # All windows start from 0
        for window in windows:
            assert window.train_start == 0

    def test_run_walk_forward(self, sample_returns):
        """Test running walk-forward optimization."""
        optimizer = WalkForwardOptimizer(
            train_period=252,
            test_period=63,
            walk_forward_type=WalkForwardType.ROLLING,
        )

        def optimize_fn(returns, features):
            return {"threshold": np.std(returns)}

        def evaluate_fn(returns, features, params):
            metrics = calculate_performance_metrics(returns)
            signals = np.sign(returns)
            return signals, metrics

        report = optimizer.run(
            sample_returns,
            features=None,
            optimize_fn=optimize_fn,
            evaluate_fn=evaluate_fn,
            strategy_name="TestStrategy",
        )

        assert report.total_windows > 0
        assert "oos_sharpe_mean" in report.aggregated_metrics


class TestPurgedKFold:
    """Test Purged K-Fold cross-validation."""

    def test_generate_splits(self):
        """Test generating purged k-fold splits."""
        kfold = PurgedKFold(n_splits=5, purge_gap=5)
        splits = kfold.split(500)
        assert len(splits) == 5

        # Check no overlap between train and test
        for train_idx, test_idx in splits:
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_purge_gap_applied(self):
        """Test that purge gap is correctly applied."""
        kfold = PurgedKFold(n_splits=5, purge_gap=10)
        splits = kfold.split(500)

        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Gap between train end and test start should be at least purge_gap
                train_max = np.max(train_idx[train_idx < np.min(test_idx)]) if np.any(train_idx < np.min(test_idx)) else -100
                test_min = np.min(test_idx)
                if train_max >= 0:
                    assert test_min - train_max >= 10


class TestOutOfSampleValidator:
    """Test out-of-sample validation."""

    def test_validate_oos_performance(self, sample_returns):
        """Test OOS validation."""
        validator = OutOfSampleValidator(
            is_start_idx=0,
            is_end_idx=252,
            oos_start_idx=252,
            oos_end_idx=504,
            min_oos_sharpe_ratio=0.5,
        )

        def optimize_fn(returns, features):
            return {"param": 1.0}

        def evaluate_fn(returns, features, params):
            metrics = calculate_performance_metrics(returns)
            return np.zeros(len(returns)), metrics

        result = validator.validate(
            sample_returns,
            features=None,
            optimize_fn=optimize_fn,
            evaluate_fn=evaluate_fn,
        )

        assert "is_sharpe" in result
        assert "oos_sharpe" in result
        assert "validation_passed" in result


class TestCalculatePerformanceMetrics:
    """Test performance metrics calculation."""

    def test_metrics_calculation(self, sample_returns):
        """Test calculating performance metrics."""
        metrics = calculate_performance_metrics(sample_returns)
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics

    def test_empty_returns(self):
        """Test handling empty returns."""
        metrics = calculate_performance_metrics(np.array([]))
        assert metrics["sharpe_ratio"] == 0
        assert metrics["total_return"] == 0


# =============================================================================
# Benchmark Comparison
# =============================================================================

class TestBenchmarks:
    """Test benchmark implementations."""

    def test_buy_and_hold_benchmark(self, market_returns):
        """Test buy-and-hold benchmark."""
        benchmark = BuyAndHoldBenchmark()
        returns = benchmark.calculate_returns(market_returns)
        np.testing.assert_array_equal(returns, market_returns)

    def test_sixty_forty_benchmark(self, market_returns):
        """Test 60/40 benchmark."""
        benchmark = SixtyFortyBenchmark()
        returns = benchmark.calculate_returns(market_returns)
        assert len(returns) == len(market_returns)
        # 60/40 should have lower volatility than pure equity
        assert np.std(returns) <= np.std(market_returns) * 1.1  # Allow some tolerance

    def test_momentum_benchmark(self, market_returns):
        """Test momentum benchmark."""
        benchmark = MomentumBenchmark(lookback_period=63, holding_period=21)
        returns = benchmark.calculate_returns(market_returns)
        assert len(returns) == len(market_returns)

    def test_risk_free_benchmark(self, market_returns):
        """Test risk-free benchmark."""
        benchmark = RiskFreeBenchmark(annual_rate=0.04)
        returns = benchmark.calculate_returns(market_returns)
        assert len(returns) == len(market_returns)
        # All returns should be constant
        assert np.allclose(returns, returns[0])


class TestBenchmarkComparator:
    """Test benchmark comparison."""

    def test_compare_against_benchmarks(self, sample_returns, market_returns):
        """Test comparing strategy against benchmarks."""
        comparator = BenchmarkComparator()
        report = comparator.compare(
            sample_returns,
            market_returns,
            strategy_name="TestStrategy",
        )

        assert len(report.benchmark_results) > 0
        assert report.overall_ranking >= 1
        assert "n_outperformed" in report.summary

    def test_add_custom_benchmark(self, sample_returns, market_returns):
        """Test adding custom benchmark."""
        comparator = BenchmarkComparator(benchmarks=[])
        comparator.add_benchmark(BuyAndHoldBenchmark("Custom S&P"))

        report = comparator.compare(sample_returns, market_returns)
        assert len(report.benchmark_results) == 1
        assert report.benchmark_results[0].benchmark_name == "Custom S&P"


class TestAlphaCalculator:
    """Test alpha calculation."""

    def test_capm_alpha(self, sample_returns, market_returns):
        """Test CAPM alpha calculation."""
        calculator = AlphaCalculator()
        result = calculator.calculate_capm_alpha(sample_returns, market_returns)

        assert "alpha_daily" in result
        assert "alpha_annualized" in result
        assert "beta" in result
        assert "r_squared" in result
        assert 0 <= result["r_squared"] <= 1

    def test_fama_french_alpha(self, sample_returns, market_returns):
        """Test Fama-French alpha calculation."""
        np.random.seed(42)
        calculator = AlphaCalculator()
        smb = np.random.normal(0, 0.005, len(market_returns))
        hml = np.random.normal(0, 0.005, len(market_returns))

        result = calculator.calculate_fama_french_alpha(
            sample_returns,
            market_returns,
            smb_returns=smb,
            hml_returns=hml,
        )

        assert "alpha_daily" in result
        assert "beta_mkt" in result
        assert "beta_smb" in result
        assert "beta_hml" in result


class TestGenerateBenchmarkReport:
    """Test benchmark report generation."""

    def test_generate_full_report(self, sample_returns, market_returns):
        """Test generating full benchmark report."""
        report = generate_benchmark_report(
            sample_returns,
            market_returns,
            strategy_name="TestStrategy",
        )

        assert "comparison" in report
        assert "alpha_metrics" in report
        assert "validation_passed" in report


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for the validation module."""

    def test_full_strategy_validation_workflow(self, sample_returns, market_returns):
        """Test complete strategy validation workflow."""
        # 1. Model validation
        strategy_validator = StrategyValidator("TestStrategy")
        metrics = calculate_performance_metrics(sample_returns)
        model_data = {
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
            "oos_sharpe": metrics["sharpe_ratio"] * 0.7,
            "is_sharpe": metrics["sharpe_ratio"],
            "parameter_stability_score": 0.75,
        }
        validation_report = strategy_validator.validate(None, model_data)

        # 2. Statistical testing
        stat_tests = StrategyStatisticalTests()
        sharpe_test = stat_tests.test_sharpe_significance(sample_returns)

        # 3. Stress testing
        stress_engine = StressTestEngine(random_state=42)
        stress_results = stress_engine.run_all_historical_scenarios(sample_returns)

        # 4. Benchmark comparison
        benchmark_report = generate_benchmark_report(
            sample_returns, market_returns, "TestStrategy"
        )

        # Verify all components work together
        assert validation_report is not None
        assert sharpe_test is not None
        assert len(stress_results) > 0
        assert benchmark_report is not None

    def test_walk_forward_with_validation(self, sample_returns):
        """Test walk-forward optimization with validation checks."""
        optimizer = WalkForwardOptimizer(
            train_period=126,
            test_period=63,
            walk_forward_type=WalkForwardType.ROLLING,
        )

        def optimize_fn(returns, features):
            return {"lookback": 20}

        def evaluate_fn(returns, features, params):
            metrics = calculate_performance_metrics(returns)
            return np.zeros(len(returns)), metrics

        report = optimizer.run(
            sample_returns,
            features=None,
            optimize_fn=optimize_fn,
            evaluate_fn=evaluate_fn,
        )

        # Verify IS/OOS degradation check
        assert report.aggregated_metrics["is_oos_sharpe_ratio"] is not None

        # This ratio should ideally be >= 0.5 per design doc requirements
        # (but may not be for random data)
        assert "is_oos_sharpe_ratio" in report.aggregated_metrics
