"""
Validation Module for Quantitative Trading System.

Provides comprehensive validation capabilities:
- Model validation per Fed SR 11-7 guidelines
- Statistical significance testing
- Monte Carlo stress testing
- Walk-forward validation
- Benchmark comparisons

Reference: Section 13 of design-doc.md
"""

from quant_trading.validation.model_validation import (
    ValidationStatus,
    ValidationSeverity,
    ValidationResult,
    ValidationReport,
    ValidationCheck,
    ThresholdCheck,
    ParameterBoundsCheck,
    ModelValidator,
    HestonModelValidator,
    FellerConditionCheck,
    SABRModelValidator,
    OUModelValidator,
    StrategyValidator,
)

from quant_trading.validation.statistical_tests import (
    TestResult,
    StatisticalTestResult,
    StrategyStatisticalTests,
    OverfittingDetector,
    BootstrapAnalysis,
)

from quant_trading.validation.stress_testing import (
    ScenarioType,
    MarketScenario,
    StressTestResult,
    HISTORICAL_SCENARIOS,
    StressTestEngine,
    TailRiskAnalyzer,
)

from quant_trading.validation.walk_forward import (
    WalkForwardType,
    WalkForwardWindow,
    WalkForwardResult,
    WalkForwardReport,
    WalkForwardOptimizer,
    PurgedKFold,
    OutOfSampleValidator,
    calculate_performance_metrics,
)

from quant_trading.validation.benchmarks import (
    BenchmarkType,
    BenchmarkResult,
    ComparisonReport,
    Benchmark,
    BuyAndHoldBenchmark,
    SixtyFortyBenchmark,
    MomentumBenchmark,
    RiskFreeBenchmark,
    EqualWeightBenchmark,
    BenchmarkComparator,
    AlphaCalculator,
    generate_benchmark_report,
)

__all__ = [
    # Model Validation
    "ValidationStatus",
    "ValidationSeverity",
    "ValidationResult",
    "ValidationReport",
    "ValidationCheck",
    "ThresholdCheck",
    "ParameterBoundsCheck",
    "ModelValidator",
    "HestonModelValidator",
    "FellerConditionCheck",
    "SABRModelValidator",
    "OUModelValidator",
    "StrategyValidator",
    # Statistical Tests
    "TestResult",
    "StatisticalTestResult",
    "StrategyStatisticalTests",
    "OverfittingDetector",
    "BootstrapAnalysis",
    # Stress Testing
    "ScenarioType",
    "MarketScenario",
    "StressTestResult",
    "HISTORICAL_SCENARIOS",
    "StressTestEngine",
    "TailRiskAnalyzer",
    # Walk-Forward
    "WalkForwardType",
    "WalkForwardWindow",
    "WalkForwardResult",
    "WalkForwardReport",
    "WalkForwardOptimizer",
    "PurgedKFold",
    "OutOfSampleValidator",
    "calculate_performance_metrics",
    # Benchmarks
    "BenchmarkType",
    "BenchmarkResult",
    "ComparisonReport",
    "Benchmark",
    "BuyAndHoldBenchmark",
    "SixtyFortyBenchmark",
    "MomentumBenchmark",
    "RiskFreeBenchmark",
    "EqualWeightBenchmark",
    "BenchmarkComparator",
    "AlphaCalculator",
    "generate_benchmark_report",
]
