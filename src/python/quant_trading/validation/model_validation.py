"""
Model Validation Framework for Quantitative Trading System.

Provides comprehensive model validation per Fed SR 11-7 guidelines:
- Conceptual soundness review
- Developmental evidence testing
- Outcomes analysis
- Ongoing monitoring

Reference: Section 13 of design-doc.md
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    NOT_RUN = "not_run"


class ValidationSeverity(Enum):
    """Severity level of validation findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    model_name: str
    model_version: str
    validation_date: datetime
    results: List[ValidationResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all critical/high severity tests passed."""
        for result in self.results:
            if result.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]:
                if result.status == ValidationStatus.FAILED:
                    return False
        return True

    @property
    def total_tests(self) -> int:
        """Total number of tests run."""
        return len([r for r in self.results if r.status != ValidationStatus.NOT_RUN])

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return len([r for r in self.results if r.status == ValidationStatus.PASSED])

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return len([r for r in self.results if r.status == ValidationStatus.FAILED])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "validation_date": self.validation_date.isoformat(),
            "passed": self.passed,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class ValidationCheck(ABC):
    """Base class for validation checks."""

    def __init__(
        self,
        name: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.MEDIUM,
    ):
        self.name = name
        self.description = description
        self.severity = severity

    @abstractmethod
    def run(self, model: Any, data: Dict[str, Any]) -> ValidationResult:
        """Run the validation check."""
        pass


class ThresholdCheck(ValidationCheck):
    """Check if a metric meets a threshold."""

    def __init__(
        self,
        name: str,
        description: str,
        metric_fn: Callable[[Any, Dict[str, Any]], float],
        threshold: float,
        comparison: str = ">=",
        severity: ValidationSeverity = ValidationSeverity.MEDIUM,
    ):
        super().__init__(name, description, severity)
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.comparison = comparison

    def run(self, model: Any, data: Dict[str, Any]) -> ValidationResult:
        """Run threshold check."""
        try:
            value = self.metric_fn(model, data)

            if self.comparison == ">=":
                passed = value >= self.threshold
            elif self.comparison == "<=":
                passed = value <= self.threshold
            elif self.comparison == ">":
                passed = value > self.threshold
            elif self.comparison == "<":
                passed = value < self.threshold
            elif self.comparison == "==":
                passed = np.isclose(value, self.threshold)
            else:
                passed = False

            status = ValidationStatus.PASSED if passed else ValidationStatus.FAILED
            message = f"{self.name}: {value:.4f} {self.comparison} {self.threshold} - {'PASS' if passed else 'FAIL'}"

            return ValidationResult(
                name=self.name,
                status=status,
                severity=self.severity,
                message=message,
                metric_value=value,
                threshold=self.threshold,
            )
        except Exception as e:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.FAILED,
                severity=self.severity,
                message=f"Error running check: {str(e)}",
            )


class ParameterBoundsCheck(ValidationCheck):
    """Check if model parameters are within valid bounds."""

    def __init__(
        self,
        name: str,
        parameter_bounds: Dict[str, Tuple[float, float]],
        severity: ValidationSeverity = ValidationSeverity.HIGH,
    ):
        super().__init__(name, "Check parameter bounds", severity)
        self.parameter_bounds = parameter_bounds

    def run(self, model: Any, data: Dict[str, Any]) -> ValidationResult:
        """Run parameter bounds check."""
        params = data.get("parameters", {})
        violations = []

        for param, (lower, upper) in self.parameter_bounds.items():
            if param in params:
                value = params[param]
                if value < lower or value > upper:
                    violations.append(f"{param}={value:.4f} outside [{lower}, {upper}]")

        if violations:
            return ValidationResult(
                name=self.name,
                status=ValidationStatus.FAILED,
                severity=self.severity,
                message=f"Parameter violations: {', '.join(violations)}",
                details={"violations": violations},
            )

        return ValidationResult(
            name=self.name,
            status=ValidationStatus.PASSED,
            severity=self.severity,
            message="All parameters within bounds",
        )


class ModelValidator:
    """Comprehensive model validator."""

    def __init__(self, model_name: str, model_version: str = "1.0"):
        self.model_name = model_name
        self.model_version = model_version
        self.checks: List[ValidationCheck] = []

    def add_check(self, check: ValidationCheck) -> None:
        """Add a validation check."""
        self.checks.append(check)

    def add_threshold_check(
        self,
        name: str,
        description: str,
        metric_fn: Callable[[Any, Dict[str, Any]], float],
        threshold: float,
        comparison: str = ">=",
        severity: ValidationSeverity = ValidationSeverity.MEDIUM,
    ) -> None:
        """Add a threshold-based check."""
        self.checks.append(ThresholdCheck(
            name=name,
            description=description,
            metric_fn=metric_fn,
            threshold=threshold,
            comparison=comparison,
            severity=severity,
        ))

    def validate(
        self,
        model: Any,
        data: Dict[str, Any],
    ) -> ValidationReport:
        """Run all validation checks and generate report."""
        results = []

        for check in self.checks:
            result = check.run(model, data)
            results.append(result)

        # Generate summary
        summary = {
            "total_checks": len(results),
            "passed": len([r for r in results if r.status == ValidationStatus.PASSED]),
            "failed": len([r for r in results if r.status == ValidationStatus.FAILED]),
            "warnings": len([r for r in results if r.status == ValidationStatus.WARNING]),
            "critical_failures": len([
                r for r in results
                if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.CRITICAL
            ]),
        }

        # Generate recommendations
        recommendations = []
        for result in results:
            if result.status == ValidationStatus.FAILED:
                if result.severity == ValidationSeverity.CRITICAL:
                    recommendations.append(f"CRITICAL: Address {result.name} immediately")
                elif result.severity == ValidationSeverity.HIGH:
                    recommendations.append(f"HIGH: Review and fix {result.name}")

        return ValidationReport(
            model_name=self.model_name,
            model_version=self.model_version,
            validation_date=datetime.now(),
            results=results,
            summary=summary,
            recommendations=recommendations,
        )


# =============================================================================
# Specialized Model Validators
# =============================================================================

class HestonModelValidator(ModelValidator):
    """Validator for Heston stochastic volatility model."""

    def __init__(self, model_version: str = "1.0"):
        super().__init__("Heston", model_version)
        self._setup_checks()

    def _setup_checks(self) -> None:
        """Set up Heston-specific validation checks."""
        # Feller condition: 2*kappa*theta >= sigma^2
        self.add_check(FellerConditionCheck())

        # Parameter bounds
        self.add_check(ParameterBoundsCheck(
            name="heston_parameter_bounds",
            parameter_bounds={
                "kappa": (0.01, 10.0),
                "theta": (0.001, 1.0),
                "sigma": (0.01, 2.0),
                "rho": (-1.0, 0.0),  # Usually negative for equity
                "v0": (0.001, 1.0),
            },
            severity=ValidationSeverity.HIGH,
        ))

        # Calibration quality
        self.add_threshold_check(
            name="calibration_rmse",
            description="Check calibration RMSE is acceptable",
            metric_fn=lambda m, d: d.get("rmse", 1.0),
            threshold=0.05,
            comparison="<=",
            severity=ValidationSeverity.HIGH,
        )

        self.add_threshold_check(
            name="calibration_r_squared",
            description="Check R-squared of calibration",
            metric_fn=lambda m, d: d.get("r_squared", 0.0),
            threshold=0.90,
            comparison=">=",
            severity=ValidationSeverity.MEDIUM,
        )


class FellerConditionCheck(ValidationCheck):
    """Check Feller condition for Heston model."""

    def __init__(self):
        super().__init__(
            name="feller_condition",
            description="Check 2*kappa*theta >= sigma^2 for variance process",
            severity=ValidationSeverity.CRITICAL,
        )

    def run(self, model: Any, data: Dict[str, Any]) -> ValidationResult:
        """Check Feller condition."""
        params = data.get("parameters", {})

        kappa = params.get("kappa", 0)
        theta = params.get("theta", 0)
        sigma = params.get("sigma", 0)

        feller_lhs = 2 * kappa * theta
        feller_rhs = sigma ** 2
        satisfied = feller_lhs >= feller_rhs

        return ValidationResult(
            name=self.name,
            status=ValidationStatus.PASSED if satisfied else ValidationStatus.FAILED,
            severity=self.severity,
            message=f"Feller: 2κθ={feller_lhs:.4f} {'≥' if satisfied else '<'} σ²={feller_rhs:.4f}",
            metric_value=feller_lhs - feller_rhs,
            details={
                "feller_lhs": feller_lhs,
                "feller_rhs": feller_rhs,
                "margin": feller_lhs - feller_rhs,
            },
        )


class SABRModelValidator(ModelValidator):
    """Validator for SABR volatility model."""

    def __init__(self, model_version: str = "1.0"):
        super().__init__("SABR", model_version)
        self._setup_checks()

    def _setup_checks(self) -> None:
        """Set up SABR-specific validation checks."""
        # Parameter bounds
        self.add_check(ParameterBoundsCheck(
            name="sabr_parameter_bounds",
            parameter_bounds={
                "alpha": (0.001, 2.0),
                "beta": (0.0, 1.0),
                "rho": (-1.0, 1.0),
                "nu": (0.001, 2.0),
            },
            severity=ValidationSeverity.HIGH,
        ))

        # Calibration quality
        self.add_threshold_check(
            name="smile_fit_rmse",
            description="Check smile fit RMSE",
            metric_fn=lambda m, d: d.get("rmse", 1.0),
            threshold=0.02,
            comparison="<=",
            severity=ValidationSeverity.MEDIUM,
        )


class OUModelValidator(ModelValidator):
    """Validator for Ornstein-Uhlenbeck mean reversion model."""

    def __init__(self, model_version: str = "1.0"):
        super().__init__("OU", model_version)
        self._setup_checks()

    def _setup_checks(self) -> None:
        """Set up OU-specific validation checks."""
        # Parameter bounds
        self.add_check(ParameterBoundsCheck(
            name="ou_parameter_bounds",
            parameter_bounds={
                "theta": (-np.inf, np.inf),  # Mean can be any value
                "mu": (0.001, 100.0),  # Mean reversion speed must be positive
                "sigma": (0.001, 10.0),
            },
            severity=ValidationSeverity.HIGH,
        ))

        # Half-life check (reasonable for trading)
        self.add_threshold_check(
            name="half_life_days",
            description="Check half-life is in tradeable range",
            metric_fn=lambda m, d: d.get("half_life", 0) / (1/252),  # Convert to days
            threshold=180,
            comparison="<=",
            severity=ValidationSeverity.MEDIUM,
        )

        # Stationarity check
        self.add_threshold_check(
            name="adf_pvalue",
            description="Check ADF test for stationarity",
            metric_fn=lambda m, d: d.get("adf_pvalue", 1.0),
            threshold=0.05,
            comparison="<",
            severity=ValidationSeverity.HIGH,
        )


class StrategyValidator(ModelValidator):
    """Validator for trading strategies."""

    def __init__(self, strategy_name: str, model_version: str = "1.0"):
        super().__init__(strategy_name, model_version)
        self._setup_checks()

    def _setup_checks(self) -> None:
        """Set up strategy validation checks."""
        # Performance checks
        self.add_threshold_check(
            name="sharpe_ratio",
            description="Check Sharpe ratio meets target",
            metric_fn=lambda m, d: d.get("sharpe_ratio", 0),
            threshold=0.5,
            comparison=">=",
            severity=ValidationSeverity.HIGH,
        )

        self.add_threshold_check(
            name="max_drawdown",
            description="Check max drawdown within limits",
            metric_fn=lambda m, d: abs(d.get("max_drawdown", 1.0)),
            threshold=0.25,
            comparison="<=",
            severity=ValidationSeverity.CRITICAL,
        )

        self.add_threshold_check(
            name="win_rate",
            description="Check win rate is acceptable",
            metric_fn=lambda m, d: d.get("win_rate", 0),
            threshold=0.45,
            comparison=">=",
            severity=ValidationSeverity.MEDIUM,
        )

        # Out-of-sample validation
        self.add_threshold_check(
            name="oos_sharpe_ratio",
            description="Check out-of-sample Sharpe >= 50% of in-sample",
            metric_fn=lambda m, d: d.get("oos_sharpe", 0) / max(d.get("is_sharpe", 1), 0.01),
            threshold=0.5,
            comparison=">=",
            severity=ValidationSeverity.CRITICAL,
        )

        # Overfitting checks
        self.add_threshold_check(
            name="parameter_stability",
            description="Check parameter stability across time periods",
            metric_fn=lambda m, d: d.get("parameter_stability_score", 0),
            threshold=0.7,
            comparison=">=",
            severity=ValidationSeverity.HIGH,
        )
