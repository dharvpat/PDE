"""
Model Diagnostics Monitoring for Quantitative Trading System.

Provides comprehensive model monitoring including:
- Calibration quality tracking
- Model drift detection
- Parameter stability analysis
- Forecast accuracy metrics
- Backtesting vs live performance comparison
- Model degradation alerts
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of models being monitored."""
    HESTON = "heston"
    SABR = "sabr"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"
    FACTOR_MODEL = "factor_model"
    SIGNAL_MODEL = "signal_model"
    RISK_MODEL = "risk_model"
    EXECUTION_MODEL = "execution_model"
    CUSTOM = "custom"


class DiagnosticStatus(Enum):
    """Status of model diagnostics."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class CalibrationMetrics:
    """Metrics from model calibration."""

    model_name: str
    model_type: ModelType
    rmse: float
    mae: float
    r_squared: float
    max_error: float
    num_points: int
    calibration_time_seconds: float
    parameters: Dict[str, float]
    parameter_bounds_satisfied: bool
    convergence_achieved: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "rmse": self.rmse,
            "mae": self.mae,
            "r_squared": self.r_squared,
            "max_error": self.max_error,
            "num_points": self.num_points,
            "calibration_time_seconds": self.calibration_time_seconds,
            "parameters": self.parameters,
            "parameter_bounds_satisfied": self.parameter_bounds_satisfied,
            "convergence_achieved": self.convergence_achieved,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DriftMetrics:
    """Metrics for model drift detection."""

    model_name: str
    psi: float  # Population Stability Index
    kl_divergence: float  # KL divergence from baseline
    ks_statistic: float  # Kolmogorov-Smirnov statistic
    ks_pvalue: float
    feature_drift: Dict[str, float]  # Per-feature drift scores
    drift_detected: bool
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "psi": self.psi,
            "kl_divergence": self.kl_divergence,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
            "feature_drift": self.feature_drift,
            "drift_detected": self.drift_detected,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ParameterStability:
    """Parameter stability analysis."""

    model_name: str
    parameter_name: str
    current_value: float
    historical_mean: float
    historical_std: float
    z_score: float
    percentile: float
    is_stable: bool
    trend: str  # "increasing", "decreasing", "stable"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "z_score": self.z_score,
            "percentile": self.percentile,
            "is_stable": self.is_stable,
            "trend": self.trend,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ForecastAccuracy:
    """Forecast accuracy metrics."""

    model_name: str
    horizon: str  # e.g., "1d", "5d", "1m"
    mae: float
    rmse: float
    mape: float  # Mean Absolute Percentage Error
    direction_accuracy: float  # % correct direction predictions
    hit_rate: float  # % of forecasts within confidence interval
    information_coefficient: float  # Correlation between forecast and actual
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "direction_accuracy": self.direction_accuracy,
            "hit_rate": self.hit_rate,
            "information_coefficient": self.information_coefficient,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BacktestComparison:
    """Comparison of backtest vs live performance."""

    model_name: str
    strategy_name: str
    backtest_sharpe: float
    live_sharpe: float
    sharpe_ratio: float  # live / backtest
    backtest_return: float
    live_return: float
    return_ratio: float
    backtest_volatility: float
    live_volatility: float
    backtest_max_dd: float
    live_max_dd: float
    degradation_detected: bool
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "strategy_name": self.strategy_name,
            "backtest_sharpe": self.backtest_sharpe,
            "live_sharpe": self.live_sharpe,
            "sharpe_ratio": self.sharpe_ratio,
            "backtest_return": self.backtest_return,
            "live_return": self.live_return,
            "return_ratio": self.return_ratio,
            "backtest_volatility": self.backtest_volatility,
            "live_volatility": self.live_volatility,
            "backtest_max_dd": self.backtest_max_dd,
            "live_max_dd": self.live_max_dd,
            "degradation_detected": self.degradation_detected,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


@dataclass
class ModelDiagnosticReport:
    """Complete diagnostic report for a model."""

    model_name: str
    model_type: ModelType
    status: DiagnosticStatus
    calibration: Optional[CalibrationMetrics]
    drift: Optional[DriftMetrics]
    parameter_stability: List[ParameterStability]
    forecast_accuracy: List[ForecastAccuracy]
    backtest_comparison: Optional[BacktestComparison]
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type.value,
            "status": self.status.value,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "drift": self.drift.to_dict() if self.drift else None,
            "parameter_stability": [p.to_dict() for p in self.parameter_stability],
            "forecast_accuracy": [f.to_dict() for f in self.forecast_accuracy],
            "backtest_comparison": self.backtest_comparison.to_dict() if self.backtest_comparison else None,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class CalibrationMonitor:
    """Monitor model calibration quality."""

    def __init__(
        self,
        rmse_warning_threshold: float = 0.03,
        rmse_critical_threshold: float = 0.05,
        r_squared_warning_threshold: float = 0.90,
        r_squared_critical_threshold: float = 0.80,
        max_calibration_time: float = 60.0,
    ):
        self.rmse_warning = rmse_warning_threshold
        self.rmse_critical = rmse_critical_threshold
        self.r2_warning = r_squared_warning_threshold
        self.r2_critical = r_squared_critical_threshold
        self.max_calibration_time = max_calibration_time
        self._history: Dict[str, List[CalibrationMetrics]] = {}
        self._max_history = 100

    def record_calibration(
        self,
        model_name: str,
        model_type: ModelType,
        predicted: np.ndarray,
        actual: np.ndarray,
        parameters: Dict[str, float],
        calibration_time: float,
        parameter_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        convergence_achieved: bool = True,
    ) -> CalibrationMetrics:
        """Record calibration results."""
        # Calculate metrics
        errors = predicted - actual
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        max_error = float(np.max(np.abs(errors)))

        # R-squared
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Check parameter bounds
        bounds_satisfied = True
        if parameter_bounds:
            for param, value in parameters.items():
                if param in parameter_bounds:
                    low, high = parameter_bounds[param]
                    if value < low or value > high:
                        bounds_satisfied = False
                        break

        metrics = CalibrationMetrics(
            model_name=model_name,
            model_type=model_type,
            rmse=rmse,
            mae=mae,
            r_squared=r_squared,
            max_error=max_error,
            num_points=len(actual),
            calibration_time_seconds=calibration_time,
            parameters=parameters,
            parameter_bounds_satisfied=bounds_satisfied,
            convergence_achieved=convergence_achieved,
        )

        # Store in history
        if model_name not in self._history:
            self._history[model_name] = []
        self._history[model_name].append(metrics)
        if len(self._history[model_name]) > self._max_history:
            self._history[model_name] = self._history[model_name][-self._max_history:]

        return metrics

    def get_status(self, metrics: CalibrationMetrics) -> DiagnosticStatus:
        """Determine calibration status."""
        if metrics.rmse >= self.rmse_critical or metrics.r_squared <= self.r2_critical:
            return DiagnosticStatus.CRITICAL
        elif metrics.rmse >= self.rmse_warning or metrics.r_squared <= self.r2_warning:
            return DiagnosticStatus.WARNING
        elif not metrics.parameter_bounds_satisfied or not metrics.convergence_achieved:
            return DiagnosticStatus.WARNING
        else:
            return DiagnosticStatus.HEALTHY

    def get_history(self, model_name: str) -> List[CalibrationMetrics]:
        """Get calibration history for a model."""
        return self._history.get(model_name, [])


class DriftDetector:
    """Detect model and data drift."""

    def __init__(
        self,
        psi_warning_threshold: float = 0.1,
        psi_critical_threshold: float = 0.25,
        ks_alpha: float = 0.05,
    ):
        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.ks_alpha = ks_alpha
        self._baselines: Dict[str, np.ndarray] = {}

    def set_baseline(self, model_name: str, baseline_data: np.ndarray) -> None:
        """Set baseline distribution for comparison."""
        self._baselines[model_name] = baseline_data

    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI = sum((actual% - expected%) * ln(actual% / expected%))
        """
        # Create bins from expected distribution
        _, bin_edges = np.histogram(expected, bins=n_bins)

        # Calculate proportions
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        # Avoid division by zero
        expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
        actual_pct = (actual_counts + 1) / (len(actual) + n_bins)

        # PSI calculation
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def calculate_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """Calculate KL divergence between two distributions."""
        # Create histograms
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        p_hist, _ = np.histogram(p, bins=bins, density=True)
        q_hist, _ = np.histogram(q, bins=bins, density=True)

        # Add small constant to avoid log(0)
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon

        # Normalize
        p_hist = p_hist / p_hist.sum()
        q_hist = q_hist / q_hist.sum()

        return float(np.sum(p_hist * np.log(p_hist / q_hist)))

    def detect_drift(
        self,
        model_name: str,
        current_data: np.ndarray,
        feature_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> DriftMetrics:
        """Detect drift in model inputs or outputs."""
        baseline = self._baselines.get(model_name)

        if baseline is None or len(current_data) == 0:
            return DriftMetrics(
                model_name=model_name,
                psi=0.0,
                kl_divergence=0.0,
                ks_statistic=0.0,
                ks_pvalue=1.0,
                feature_drift={},
                drift_detected=False,
            )

        # Calculate PSI
        psi = self.calculate_psi(baseline, current_data)

        # Calculate KL divergence
        kl_div = self.calculate_kl_divergence(baseline, current_data)

        # Kolmogorov-Smirnov test
        from scipy import stats
        ks_stat, ks_pvalue = stats.ks_2samp(baseline, current_data)

        # Feature-level drift
        feature_drift = {}
        if feature_data:
            for feature_name, feature_values in feature_data.items():
                baseline_key = f"{model_name}_{feature_name}"
                if baseline_key in self._baselines:
                    feature_psi = self.calculate_psi(
                        self._baselines[baseline_key],
                        feature_values,
                    )
                    feature_drift[feature_name] = feature_psi

        # Determine if drift detected
        drift_detected = (
            psi >= self.psi_warning or
            ks_pvalue < self.ks_alpha
        )

        return DriftMetrics(
            model_name=model_name,
            psi=psi,
            kl_divergence=kl_div,
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            feature_drift=feature_drift,
            drift_detected=drift_detected,
        )

    def get_status(self, metrics: DriftMetrics) -> DiagnosticStatus:
        """Determine drift status."""
        if metrics.psi >= self.psi_critical:
            return DiagnosticStatus.CRITICAL
        elif metrics.psi >= self.psi_warning or metrics.drift_detected:
            return DiagnosticStatus.WARNING
        else:
            return DiagnosticStatus.HEALTHY


class ParameterStabilityAnalyzer:
    """Analyze parameter stability over time."""

    def __init__(
        self,
        z_score_warning: float = 2.0,
        z_score_critical: float = 3.0,
        min_history: int = 20,
    ):
        self.z_score_warning = z_score_warning
        self.z_score_critical = z_score_critical
        self.min_history = min_history
        self._history: Dict[str, Dict[str, List[Tuple[datetime, float]]]] = {}

    def record_parameters(
        self,
        model_name: str,
        parameters: Dict[str, float],
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record parameter values."""
        timestamp = timestamp or datetime.now()

        if model_name not in self._history:
            self._history[model_name] = {}

        for param_name, value in parameters.items():
            if param_name not in self._history[model_name]:
                self._history[model_name][param_name] = []
            self._history[model_name][param_name].append((timestamp, value))

            # Trim history
            if len(self._history[model_name][param_name]) > 500:
                self._history[model_name][param_name] = \
                    self._history[model_name][param_name][-500:]

    def analyze_stability(
        self,
        model_name: str,
        current_parameters: Dict[str, float],
    ) -> List[ParameterStability]:
        """Analyze stability of current parameters."""
        results = []

        model_history = self._history.get(model_name, {})

        for param_name, current_value in current_parameters.items():
            history = model_history.get(param_name, [])

            if len(history) < self.min_history:
                results.append(ParameterStability(
                    model_name=model_name,
                    parameter_name=param_name,
                    current_value=current_value,
                    historical_mean=current_value,
                    historical_std=0.0,
                    z_score=0.0,
                    percentile=50.0,
                    is_stable=True,
                    trend="stable",
                ))
                continue

            # Extract values
            values = np.array([v for _, v in history])

            # Calculate statistics
            mean = float(np.mean(values))
            std = float(np.std(values))
            z_score = (current_value - mean) / std if std > 0 else 0.0

            # Calculate percentile
            percentile = float(np.sum(values <= current_value) / len(values) * 100)

            # Determine trend using linear regression
            if len(values) >= 10:
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                if slope > std * 0.1:
                    trend = "increasing"
                elif slope < -std * 0.1:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Determine stability
            is_stable = abs(z_score) <= self.z_score_warning

            results.append(ParameterStability(
                model_name=model_name,
                parameter_name=param_name,
                current_value=current_value,
                historical_mean=mean,
                historical_std=std,
                z_score=z_score,
                percentile=percentile,
                is_stable=is_stable,
                trend=trend,
            ))

        return results

    def get_status(self, stability_list: List[ParameterStability]) -> DiagnosticStatus:
        """Determine overall parameter stability status."""
        if not stability_list:
            return DiagnosticStatus.UNKNOWN

        max_z = max(abs(s.z_score) for s in stability_list)

        if max_z >= self.z_score_critical:
            return DiagnosticStatus.CRITICAL
        elif max_z >= self.z_score_warning or any(not s.is_stable for s in stability_list):
            return DiagnosticStatus.WARNING
        else:
            return DiagnosticStatus.HEALTHY


class ForecastAccuracyTracker:
    """Track model forecast accuracy."""

    def __init__(
        self,
        direction_warning_threshold: float = 0.52,
        direction_critical_threshold: float = 0.48,
        ic_warning_threshold: float = 0.03,
        ic_critical_threshold: float = 0.01,
    ):
        self.direction_warning = direction_warning_threshold
        self.direction_critical = direction_critical_threshold
        self.ic_warning = ic_warning_threshold
        self.ic_critical = ic_critical_threshold
        self._forecasts: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    def record_forecast(
        self,
        model_name: str,
        horizon: str,
        forecast: float,
        actual: float,
    ) -> None:
        """Record a forecast and its actual outcome."""
        if model_name not in self._forecasts:
            self._forecasts[model_name] = {}
        if horizon not in self._forecasts[model_name]:
            self._forecasts[model_name][horizon] = []

        self._forecasts[model_name][horizon].append((forecast, actual))

        # Trim history
        if len(self._forecasts[model_name][horizon]) > 1000:
            self._forecasts[model_name][horizon] = \
                self._forecasts[model_name][horizon][-1000:]

    def calculate_accuracy(
        self,
        model_name: str,
        horizon: str,
    ) -> Optional[ForecastAccuracy]:
        """Calculate forecast accuracy metrics."""
        if model_name not in self._forecasts:
            return None
        if horizon not in self._forecasts[model_name]:
            return None

        pairs = self._forecasts[model_name][horizon]
        if len(pairs) < 20:
            return None

        forecasts = np.array([f for f, _ in pairs])
        actuals = np.array([a for _, a in pairs])

        # MAE and RMSE
        errors = forecasts - actuals
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        # MAPE (avoiding division by zero)
        nonzero_mask = actuals != 0
        if np.any(nonzero_mask):
            mape = float(np.mean(np.abs(errors[nonzero_mask] / actuals[nonzero_mask])) * 100)
        else:
            mape = 0.0

        # Direction accuracy
        forecast_direction = np.sign(forecasts)
        actual_direction = np.sign(actuals)
        direction_accuracy = float(np.mean(forecast_direction == actual_direction))

        # Hit rate (within 1 std of historical errors)
        error_std = np.std(errors)
        hit_rate = float(np.mean(np.abs(errors) <= error_std))

        # Information coefficient (rank correlation)
        from scipy import stats
        ic, _ = stats.spearmanr(forecasts, actuals)
        ic = float(ic) if not np.isnan(ic) else 0.0

        return ForecastAccuracy(
            model_name=model_name,
            horizon=horizon,
            mae=mae,
            rmse=rmse,
            mape=mape,
            direction_accuracy=direction_accuracy,
            hit_rate=hit_rate,
            information_coefficient=ic,
        )

    def get_status(self, accuracy: ForecastAccuracy) -> DiagnosticStatus:
        """Determine forecast accuracy status."""
        if accuracy.direction_accuracy <= self.direction_critical or \
           accuracy.information_coefficient <= self.ic_critical:
            return DiagnosticStatus.CRITICAL
        elif accuracy.direction_accuracy <= self.direction_warning or \
             accuracy.information_coefficient <= self.ic_warning:
            return DiagnosticStatus.WARNING
        else:
            return DiagnosticStatus.HEALTHY


class BacktestLiveComparator:
    """Compare backtest to live performance."""

    def __init__(
        self,
        sharpe_degradation_warning: float = 0.7,
        sharpe_degradation_critical: float = 0.5,
        return_degradation_warning: float = 0.6,
        return_degradation_critical: float = 0.4,
    ):
        self.sharpe_warning = sharpe_degradation_warning
        self.sharpe_critical = sharpe_degradation_critical
        self.return_warning = return_degradation_warning
        self.return_critical = return_degradation_critical

    def compare(
        self,
        model_name: str,
        strategy_name: str,
        backtest_returns: np.ndarray,
        live_returns: np.ndarray,
        risk_free_rate: float = 0.02,
    ) -> BacktestComparison:
        """Compare backtest to live performance."""
        period_start = datetime.now() - timedelta(days=len(live_returns))
        period_end = datetime.now()

        # Calculate metrics for backtest
        bt_sharpe = self._calculate_sharpe(backtest_returns, risk_free_rate)
        bt_return = float(np.prod(1 + backtest_returns) - 1)
        bt_vol = float(np.std(backtest_returns) * np.sqrt(252))
        bt_max_dd = self._calculate_max_drawdown(backtest_returns)

        # Calculate metrics for live
        live_sharpe = self._calculate_sharpe(live_returns, risk_free_rate)
        live_return = float(np.prod(1 + live_returns) - 1)
        live_vol = float(np.std(live_returns) * np.sqrt(252))
        live_max_dd = self._calculate_max_drawdown(live_returns)

        # Ratios
        sharpe_ratio = live_sharpe / bt_sharpe if bt_sharpe != 0 else 0
        return_ratio = live_return / bt_return if bt_return != 0 else 0

        # Detect degradation
        degradation_detected = (
            sharpe_ratio < self.sharpe_warning or
            return_ratio < self.return_warning
        )

        return BacktestComparison(
            model_name=model_name,
            strategy_name=strategy_name,
            backtest_sharpe=bt_sharpe,
            live_sharpe=live_sharpe,
            sharpe_ratio=sharpe_ratio,
            backtest_return=bt_return,
            live_return=live_return,
            return_ratio=return_ratio,
            backtest_volatility=bt_vol,
            live_volatility=live_vol,
            backtest_max_dd=bt_max_dd,
            live_max_dd=live_max_dd,
            degradation_detected=degradation_detected,
            period_start=period_start,
            period_end=period_end,
        )

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252))

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        return float(np.max(drawdowns))

    def get_status(self, comparison: BacktestComparison) -> DiagnosticStatus:
        """Determine comparison status."""
        if comparison.sharpe_ratio < self.sharpe_critical or \
           comparison.return_ratio < self.return_critical:
            return DiagnosticStatus.CRITICAL
        elif comparison.sharpe_ratio < self.sharpe_warning or \
             comparison.return_ratio < self.return_warning:
            return DiagnosticStatus.WARNING
        else:
            return DiagnosticStatus.HEALTHY


class ModelDiagnosticsEngine:
    """Main engine for model diagnostics."""

    def __init__(self):
        self.calibration_monitor = CalibrationMonitor()
        self.drift_detector = DriftDetector()
        self.stability_analyzer = ParameterStabilityAnalyzer()
        self.accuracy_tracker = ForecastAccuracyTracker()
        self.backtest_comparator = BacktestLiveComparator()
        self._registered_models: Dict[str, ModelType] = {}

    def register_model(self, model_name: str, model_type: ModelType) -> None:
        """Register a model for monitoring."""
        self._registered_models[model_name] = model_type

    def record_calibration(
        self,
        model_name: str,
        predicted: np.ndarray,
        actual: np.ndarray,
        parameters: Dict[str, float],
        calibration_time: float,
        **kwargs,
    ) -> CalibrationMetrics:
        """Record calibration results."""
        model_type = self._registered_models.get(model_name, ModelType.CUSTOM)
        metrics = self.calibration_monitor.record_calibration(
            model_name=model_name,
            model_type=model_type,
            predicted=predicted,
            actual=actual,
            parameters=parameters,
            calibration_time=calibration_time,
            **kwargs,
        )
        self.stability_analyzer.record_parameters(model_name, parameters)
        return metrics

    def check_drift(
        self,
        model_name: str,
        current_data: np.ndarray,
        feature_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> DriftMetrics:
        """Check for model drift."""
        return self.drift_detector.detect_drift(model_name, current_data, feature_data)

    def set_drift_baseline(self, model_name: str, baseline_data: np.ndarray) -> None:
        """Set drift detection baseline."""
        self.drift_detector.set_baseline(model_name, baseline_data)

    def record_forecast(
        self,
        model_name: str,
        horizon: str,
        forecast: float,
        actual: float,
    ) -> None:
        """Record forecast and actual."""
        self.accuracy_tracker.record_forecast(model_name, horizon, forecast, actual)

    def generate_report(self, model_name: str) -> ModelDiagnosticReport:
        """Generate complete diagnostic report for a model."""
        model_type = self._registered_models.get(model_name, ModelType.CUSTOM)

        # Get latest calibration
        cal_history = self.calibration_monitor.get_history(model_name)
        calibration = cal_history[-1] if cal_history else None

        # Get current parameters for stability analysis
        if calibration:
            stability = self.stability_analyzer.analyze_stability(
                model_name,
                calibration.parameters,
            )
        else:
            stability = []

        # Get forecast accuracy for common horizons
        accuracy_list = []
        for horizon in ["1d", "5d", "1m"]:
            acc = self.accuracy_tracker.calculate_accuracy(model_name, horizon)
            if acc:
                accuracy_list.append(acc)

        # Determine overall status
        issues = []
        recommendations = []

        if calibration:
            cal_status = self.calibration_monitor.get_status(calibration)
            if cal_status == DiagnosticStatus.CRITICAL:
                issues.append(f"Calibration RMSE ({calibration.rmse:.4f}) exceeds critical threshold")
                recommendations.append("Review market data quality and recalibrate")
            elif cal_status == DiagnosticStatus.WARNING:
                issues.append(f"Calibration quality degraded (R²={calibration.r_squared:.4f})")
                recommendations.append("Monitor calibration closely")

        if stability:
            stab_status = self.stability_analyzer.get_status(stability)
            unstable_params = [s.parameter_name for s in stability if not s.is_stable]
            if unstable_params:
                issues.append(f"Unstable parameters: {', '.join(unstable_params)}")
                recommendations.append("Investigate regime change or data issues")

        for acc in accuracy_list:
            acc_status = self.accuracy_tracker.get_status(acc)
            if acc_status == DiagnosticStatus.CRITICAL:
                issues.append(f"Forecast accuracy critical for {acc.horizon} horizon")
                recommendations.append(f"Review {acc.horizon} forecast model")

        # Determine overall status
        if any("CRITICAL" in issue.upper() or "critical" in issue.lower() for issue in issues):
            overall_status = DiagnosticStatus.CRITICAL
        elif issues:
            overall_status = DiagnosticStatus.WARNING
        else:
            overall_status = DiagnosticStatus.HEALTHY

        return ModelDiagnosticReport(
            model_name=model_name,
            model_type=model_type,
            status=overall_status,
            calibration=calibration,
            drift=None,  # Would need current data
            parameter_stability=stability,
            forecast_accuracy=accuracy_list,
            backtest_comparison=None,  # Would need backtest data
            issues=issues,
            recommendations=recommendations,
        )


# Singleton instance
_diagnostics_engine: Optional[ModelDiagnosticsEngine] = None


def get_diagnostics_engine() -> ModelDiagnosticsEngine:
    """Get or create the diagnostics engine singleton."""
    global _diagnostics_engine
    if _diagnostics_engine is None:
        _diagnostics_engine = ModelDiagnosticsEngine()
    return _diagnostics_engine
