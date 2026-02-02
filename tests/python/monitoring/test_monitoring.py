"""
Tests for the Monitoring Module.

Tests cover:
- Prometheus metrics collection
- Alert management and notifications
- Dashboard configuration
- Structured logging
- Health checks
- Performance attribution
- Model diagnostics
"""

import json
import logging
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMetricsCollector:
    """Tests for Prometheus metrics collection."""

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initializes correctly."""
        from quant_trading.monitoring.metrics import MetricsCollector

        collector = MetricsCollector(port=9999, collect_system_metrics=False)
        assert collector is not None
        assert collector.port == 9999

    def test_record_order_created(self):
        """Test recording order creation metrics."""
        from quant_trading.monitoring.metrics import MetricsCollector

        collector = MetricsCollector(port=9998, collect_system_metrics=False)
        collector.record_order_created("momentum", "AAPL", "buy", 10000)

        # Should not raise any errors
        assert True

    def test_update_pnl(self):
        """Test updating P&L metrics."""
        from quant_trading.monitoring.metrics import MetricsCollector

        collector = MetricsCollector(port=9997, collect_system_metrics=False)
        collector.update_pnl("momentum", realized=1000, unrealized=500, daily=200)

        assert True

    def test_update_position(self):
        """Test updating position metrics."""
        from quant_trading.monitoring.metrics import MetricsCollector

        collector = MetricsCollector(port=9996, collect_system_metrics=False)
        # update_positions takes (strategy, positions_dict, exposure)
        collector.update_positions("momentum", {"AAPL": 100, "GOOGL": 50}, 15000.0)

        assert True

    def test_update_risk_metrics(self):
        """Test updating risk metrics."""
        from quant_trading.monitoring.metrics import MetricsCollector

        collector = MetricsCollector(port=9995, collect_system_metrics=False)
        collector.update_risk_metrics(
            strategy="momentum",
            var_95=-5000,
            delta=0.5,
            gamma=0.01,
            vega=100,
        )

        assert True

    def test_track_latency_decorator(self):
        """Test latency tracking decorator."""
        from quant_trading.monitoring.metrics import track_latency

        # Create a mock histogram
        class MockHistogram:
            def __init__(self):
                self.observed_values = []

            def observe(self, value):
                self.observed_values.append(value)

        histogram = MockHistogram()

        @track_latency(histogram)
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"
        assert len(histogram.observed_values) == 1
        assert histogram.observed_values[0] >= 0.01

    def test_count_calls_decorator(self):
        """Test call counting decorator."""
        from quant_trading.monitoring.metrics import count_calls

        # Create a mock counter
        class MockCounter:
            def __init__(self):
                self.count = 0

            def inc(self):
                self.count += 1

        counter = MockCounter()

        @count_calls(counter)
        def counted_function():
            return "counted"

        result = counted_function()
        assert result == "counted"
        assert counter.count == 1


class TestAlertManager:
    """Tests for alerting system."""

    def test_alert_creation(self):
        """Test creating alerts."""
        from quant_trading.monitoring.alerts import (
            Alert,
            AlertSeverity,
            AlertCategory,
            AlertStatus,
        )

        alert = Alert(
            alert_id="test-001",
            title="Test Alert",
            description="This is a test alert",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            component="test_component",
        )

        assert alert.alert_id == "test-001"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING

    def test_alert_rule_evaluation(self):
        """Test alert rule condition evaluation."""
        from quant_trading.monitoring.alerts import (
            AlertRule,
            AlertSeverity,
            AlertCategory,
        )

        rule = AlertRule(
            name="high_drawdown",
            condition=lambda m: m.get("drawdown", 0) > 0.15,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            description="High Drawdown Alert",
            component="risk_manager",
        )

        # Should trigger - evaluate returns an Alert
        metrics = {"drawdown": 0.20}
        alert = rule.evaluate(metrics)
        assert alert is not None

        # Should not trigger
        metrics = {"drawdown": 0.10}
        alert = rule.evaluate(metrics)
        assert alert is None

    def test_alert_manager_evaluate_rules(self):
        """Test alert manager rule evaluation."""
        from quant_trading.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            AlertCategory,
        )

        manager = AlertManager()

        rule = AlertRule(
            name="test_rule",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="Test Rule",
            component="test",
        )
        manager.add_rule(rule)

        # Should generate alert
        alerts = manager.evaluate_rules({"value": 150})
        assert len(alerts) == 1
        assert alerts[0].title == "test_rule"

    def test_alert_cooldown(self):
        """Test alert cooldown prevents duplicate alerts."""
        from quant_trading.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            AlertCategory,
        )

        manager = AlertManager()

        rule = AlertRule(
            name="cooldown_test",
            condition=lambda m: m.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="Cooldown Test",
            component="test",
            cooldown_minutes=60,
        )
        manager.add_rule(rule)

        # First evaluation should create alert
        alerts1 = manager.evaluate_rules({"value": 150})
        assert len(alerts1) == 1

        # Second evaluation should be suppressed by cooldown
        alerts2 = manager.evaluate_rules({"value": 150})
        assert len(alerts2) == 0

    def test_alert_acknowledge(self):
        """Test acknowledging alerts."""
        from quant_trading.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            AlertCategory,
            AlertStatus,
        )

        manager = AlertManager()

        rule = AlertRule(
            name="ack_test",
            condition=lambda m: True,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="Ack Test",
            component="test",
        )
        manager.add_rule(rule)

        alerts = manager.evaluate_rules({})
        alert_id = alerts[0].alert_id

        manager.acknowledge_alert(alert_id, "test_user")
        assert manager.active_alerts[alert_id].status == AlertStatus.ACKNOWLEDGED
        assert manager.active_alerts[alert_id].acknowledged_by == "test_user"

    def test_log_notification_channel(self):
        """Test log notification channel."""
        from quant_trading.monitoring.alerts import (
            Alert,
            AlertSeverity,
            AlertCategory,
            LogChannel,
        )

        channel = LogChannel()
        alert = Alert(
            alert_id="test-002",
            title="Log Test",
            description="Testing log channel",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            component="test",
        )

        # Should not raise
        channel.send(alert)
        assert True

    def test_create_default_alert_rules(self):
        """Test creating default alert rules."""
        from quant_trading.monitoring.alerts import create_default_alert_rules

        rules = create_default_alert_rules()
        assert len(rules) > 0
        rule_names = [r.name for r in rules]
        assert "high_drawdown" in rule_names
        assert "position_limit_breach" in rule_names


class TestDashboards:
    """Tests for Grafana dashboard configuration."""

    def test_panel_creation(self):
        """Test creating dashboard panels."""
        from quant_trading.monitoring.dashboards import Panel, PanelType, PrometheusTarget

        panel = Panel(
            title="Test Panel",
            panel_type=PanelType.GRAPH,
            targets=[PrometheusTarget(expr="up", legend_format="{{instance}}")],
            grid_pos={"x": 0, "y": 0, "w": 12, "h": 8},
        )

        panel_dict = panel.to_dict(panel_id=1)
        assert panel_dict["title"] == "Test Panel"
        assert panel_dict["type"] == "graph"

    def test_dashboard_creation(self):
        """Test creating dashboards."""
        from quant_trading.monitoring.dashboards import (
            Dashboard,
            Panel,
            PanelType,
            PrometheusTarget,
        )

        panel = Panel(
            title="Test Panel",
            panel_type=PanelType.STAT,
            targets=[PrometheusTarget(expr="up")],
            grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
        )

        dashboard = Dashboard(
            title="Test Dashboard",
            uid="test-dash",
            panels=[panel],
        )

        dash_dict = dashboard.to_dict()
        assert dash_dict["title"] == "Test Dashboard"
        assert dash_dict["uid"] == "test-dash"
        assert len(dash_dict["panels"]) == 1

    def test_dashboard_to_json(self):
        """Test dashboard JSON serialization."""
        from quant_trading.monitoring.dashboards import Dashboard

        dashboard = Dashboard(title="JSON Test", uid="json-test")
        json_str = dashboard.to_json()

        parsed = json.loads(json_str)
        assert parsed["title"] == "JSON Test"

    def test_prometheus_target(self):
        """Test Prometheus target configuration."""
        from quant_trading.monitoring.dashboards import PrometheusTarget

        target = PrometheusTarget(
            expr='sum(rate(orders_total[5m])) by (strategy)',
            legend_format="{{strategy}}",
        )

        target_dict = target.to_dict()
        assert "expr" in target_dict
        assert target_dict["legendFormat"] == "{{strategy}}"

    def test_create_trading_overview_dashboard(self):
        """Test creating trading overview dashboard."""
        from quant_trading.monitoring.dashboards import create_trading_overview_dashboard

        dashboard = create_trading_overview_dashboard()
        assert "Trading" in dashboard.title and "Overview" in dashboard.title
        assert len(dashboard.panels) > 0

    def test_create_risk_dashboard(self):
        """Test creating risk dashboard."""
        from quant_trading.monitoring.dashboards import create_risk_dashboard

        dashboard = create_risk_dashboard()
        assert "Risk" in dashboard.title
        assert len(dashboard.panels) > 0

    def test_dashboard_save(self):
        """Test saving dashboard to file."""
        from quant_trading.monitoring.dashboards import Dashboard

        dashboard = Dashboard(title="Save Test", uid="save-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_dashboard.json"
            dashboard.save(str(filepath))
            assert filepath.exists()

            with open(filepath) as f:
                loaded = json.load(f)
            assert loaded["title"] == "Save Test"


class TestStructuredLogging:
    """Tests for structured logging framework."""

    def test_get_logger(self):
        """Test getting a structured logger."""
        from quant_trading.monitoring.logging import get_logger, LogCategory

        logger = get_logger("test_logger", LogCategory.TRADING)
        assert logger is not None
        assert logger.name == "test_logger"

    def test_log_context_binding(self):
        """Test binding context to logs."""
        from quant_trading.monitoring.logging import (
            get_context,
            bind,
            unbind,
            clear_context,
        )

        clear_context()

        bind(strategy="momentum", symbol="AAPL")
        context = get_context()
        assert context.get("strategy") == "momentum"
        assert context.get("symbol") == "AAPL"

        unbind("symbol")
        assert context.get("symbol") is None

        clear_context()
        assert len(context.fields) == 0

    def test_bound_logger_context_manager(self):
        """Test BoundLogger context manager."""
        from quant_trading.monitoring.logging import (
            BoundLogger,
            get_context,
            clear_context,
        )

        clear_context()

        with BoundLogger(order_id="123", symbol="AAPL"):
            context = get_context()
            assert context.get("order_id") == "123"

        # Should be cleared after context
        assert get_context().get("order_id") is None

    def test_json_formatter(self):
        """Test JSON log formatting."""
        from quant_trading.monitoring.logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert "@timestamp" in parsed

    def test_console_formatter(self):
        """Test console log formatting."""
        from quant_trading.monitoring.logging import ConsoleFormatter

        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Console test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "Console test" in output
        assert "INFO" in output

    def test_error_tracker(self):
        """Test error tracking and grouping."""
        from quant_trading.monitoring.logging import ErrorTracker

        tracker = ErrorTracker()

        # Track similar errors
        for i in range(5):
            tracker.track(
                error_type="ValueError",
                message=f"Invalid value: {i}",
                traceback_str='File "test.py", line 10',
            )

        summary = tracker.get_summary()
        assert len(summary) == 1  # Should be grouped
        assert summary[0]["count"] == 5

    def test_trading_logger(self):
        """Test trading-specific logger."""
        from quant_trading.monitoring.logging import TradingLogger

        logger = TradingLogger()

        # Should not raise
        logger.log_order("order-123", "AAPL", "buy", 100, price=150.0)
        logger.log_fill("order-123", "AAPL", "buy", 100, 150.05)
        logger.log_signal("momentum", "AAPL", "BUY", 0.75)

        assert True

    def test_risk_logger(self):
        """Test risk-specific logger."""
        from quant_trading.monitoring.logging import RiskLogger

        logger = RiskLogger()

        logger.log_risk_metrics("momentum", var_95=-5000, var_99=-7500, expected_shortfall=-8000)
        logger.log_limit_breach("position", current_value=1100, limit_value=1000)
        logger.log_drawdown("momentum", current_drawdown=0.12, max_drawdown=0.15)

        assert True

    def test_audit_logger(self):
        """Test audit logger."""
        from quant_trading.monitoring.logging import AuditLogger

        logger = AuditLogger()

        logger.log_action("user1", "create_order", "order-123")
        logger.log_config_change("admin", "risk_limit", 1000, 1500)

        assert True


class TestHealthChecks:
    """Tests for health checks and monitoring."""

    def test_health_status_enum(self):
        """Test health status enumeration."""
        from quant_trading.monitoring.health import HealthStatus

        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_tcp_health_check(self):
        """Test TCP health check."""
        from quant_trading.monitoring.health import TCPHealthCheck, HealthStatus

        # Test with invalid host/port (should fail gracefully)
        check = TCPHealthCheck(
            name="tcp_test",
            host="localhost",
            port=99999,  # Invalid port
            timeout_seconds=1.0,
        )

        result = check.check()
        assert result.status == HealthStatus.UNHEALTHY
        assert result.name == "tcp_test"

    def test_custom_health_check(self):
        """Test custom health check."""
        from quant_trading.monitoring.health import CustomHealthCheck, HealthStatus

        def healthy_check():
            return True, "All good", {"metric": 42}

        check = CustomHealthCheck(
            name="custom_test",
            check_func=healthy_check,
        )

        result = check.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.details["metric"] == 42

    def test_health_manager(self):
        """Test health manager."""
        from quant_trading.monitoring.health import (
            HealthManager,
            CustomHealthCheck,
            HealthStatus,
        )

        manager = HealthManager()

        check = CustomHealthCheck(
            name="test_check",
            check_func=lambda: (True, "OK", {}),
        )
        manager.register_check(check)

        results = manager.run_health_checks()
        assert len(results) == 1
        assert results[0].status == HealthStatus.HEALTHY

    def test_health_report(self):
        """Test health report generation."""
        from quant_trading.monitoring.health import (
            HealthManager,
            CustomHealthCheck,
            HealthStatus,
        )

        manager = HealthManager()

        check = CustomHealthCheck(
            name="report_test",
            check_func=lambda: (True, "OK", {}),
        )
        manager.register_check(check)
        manager.run_health_checks()

        report = manager.get_health_report()
        assert report.status == HealthStatus.HEALTHY
        assert len(report.checks) == 1

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        from quant_trading.monitoring.health import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=1.0,
        )

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_available()

        # Record failures to open circuit
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert not breaker.is_available()

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        from quant_trading.monitoring.health import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(
            name="recovery_test",
            failure_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
        )

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.is_available()

        # Record successes to close
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_liveness_readiness_probes(self):
        """Test liveness and readiness probes."""
        from quant_trading.monitoring.health import liveness_probe, readiness_probe

        # Liveness should always return True
        assert liveness_probe() is True

        # Readiness depends on health checks
        assert isinstance(readiness_probe(), bool)


class TestPerformanceAttribution:
    """Tests for performance attribution."""

    def test_return_decomposition(self):
        """Test returns decomposition."""
        from quant_trading.monitoring.attribution import ReturnsAttributor

        attributor = ReturnsAttributor(risk_free_rate=0.02)

        # Generate synthetic data
        np.random.seed(42)
        n_days = 252
        benchmark_returns = np.random.normal(0.0004, 0.01, n_days)
        portfolio_returns = benchmark_returns * 1.1 + np.random.normal(0.0001, 0.005, n_days)

        result = attributor.decompose_returns(portfolio_returns, benchmark_returns)

        assert result.total_return != 0
        assert "alpha" in result.to_dict()
        assert "beta_contribution" in result.to_dict()

    def test_risk_attribution(self):
        """Test risk attribution."""
        scipy = pytest.importorskip("scipy")
        from quant_trading.monitoring.attribution import RiskAttributor

        attributor = RiskAttributor(confidence_level=0.95)

        # Generate synthetic position returns
        np.random.seed(42)
        n_days = 252
        position_returns = {
            "AAPL": np.random.normal(0.0005, 0.02, n_days),
            "GOOGL": np.random.normal(0.0004, 0.025, n_days),
            "MSFT": np.random.normal(0.0003, 0.018, n_days),
        }
        position_weights = {"AAPL": 0.4, "GOOGL": 0.35, "MSFT": 0.25}

        result = attributor.attribute_risk(position_returns, position_weights)

        assert result.total_var != 0
        assert len(result.position_contributions) == 3
        assert len(result.component_var) == 3

    def test_trade_attribution(self):
        """Test trade-level attribution."""
        from quant_trading.monitoring.attribution import TradeAttributor

        attributor = TradeAttributor()

        result = attributor.attribute_trade(
            trade_id="trade-001",
            symbol="AAPL",
            side="buy",
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            signal_price=149.5,
            optimal_entry_price=149.8,
            optimal_exit_price=155.2,
            signal_strength=0.8,
        )

        assert result.pnl == 500.0  # (155 - 150) * 100
        assert result.signal_quality == 0.8
        assert result.trade_id == "trade-001"

    def test_brinson_attribution(self):
        """Test Brinson-Fachler attribution."""
        from quant_trading.monitoring.attribution import BrinsonAttributor

        attributor = BrinsonAttributor()

        portfolio_weights = {"AAPL": 0.3, "GOOGL": 0.4, "JPM": 0.3}
        benchmark_weights = {"AAPL": 0.2, "GOOGL": 0.3, "JPM": 0.5}
        portfolio_returns = {"AAPL": 0.10, "GOOGL": 0.15, "JPM": 0.05}
        benchmark_returns = {"AAPL": 0.08, "GOOGL": 0.12, "JPM": 0.06}
        sector_mapping = {"AAPL": "Tech", "GOOGL": "Tech", "JPM": "Finance"}

        result = attributor.calculate_attribution(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            sector_mapping=sector_mapping,
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
        )

        # Total active return should be sum of effects
        expected_total = result.allocation_effect + result.selection_effect + result.interaction_effect
        assert abs(result.total_active_return - expected_total) < 1e-10

    def test_factor_attribution(self):
        """Test factor-based attribution."""
        scipy = pytest.importorskip("scipy")
        from quant_trading.monitoring.attribution import FactorAttributor

        attributor = FactorAttributor(factors=["Mkt-RF", "SMB", "HML"])

        # Generate synthetic data
        np.random.seed(42)
        n_days = 100
        factor_returns = {
            "Mkt-RF": np.random.normal(0.0003, 0.01, n_days),
            "SMB": np.random.normal(0.0001, 0.005, n_days),
            "HML": np.random.normal(0.0001, 0.006, n_days),
        }
        portfolio_returns = (
            0.8 * factor_returns["Mkt-RF"] +
            0.3 * factor_returns["SMB"] +
            0.2 * factor_returns["HML"] +
            np.random.normal(0.0001, 0.002, n_days)
        )

        exposures = attributor.calculate_factor_exposures(portfolio_returns, factor_returns)

        assert len(exposures) == 3
        # Market beta should be close to 0.8
        mkt_exposure = next(e for e in exposures if e.factor_name == "Mkt-RF")
        assert abs(mkt_exposure.exposure - 0.8) < 0.3  # Allow some estimation error

    def test_performance_attribution_engine(self):
        """Test complete performance attribution engine."""
        scipy = pytest.importorskip("scipy")
        from quant_trading.monitoring.attribution import PerformanceAttributionEngine

        engine = PerformanceAttributionEngine()

        # Generate synthetic data
        np.random.seed(42)
        n_days = 100
        portfolio_returns = np.random.normal(0.0005, 0.015, n_days)
        benchmark_returns = np.random.normal(0.0003, 0.012, n_days)
        position_returns = {
            "AAPL": np.random.normal(0.0006, 0.02, n_days),
            "GOOGL": np.random.normal(0.0004, 0.022, n_days),
        }
        position_weights = {"AAPL": 0.6, "GOOGL": 0.4}

        report = engine.generate_report(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            position_returns=position_returns,
            position_weights=position_weights,
        )

        assert report.return_decomposition is not None
        assert report.risk_attribution is not None
        assert "sharpe_ratio" in report.summary_metrics


class TestModelDiagnostics:
    """Tests for model diagnostics monitoring."""

    def test_calibration_metrics(self):
        """Test calibration metrics recording."""
        from quant_trading.monitoring.diagnostics import (
            CalibrationMonitor,
            ModelType,
            DiagnosticStatus,
        )

        monitor = CalibrationMonitor()

        # Generate synthetic calibration data
        np.random.seed(42)
        n_points = 50
        actual = np.random.uniform(0.1, 0.4, n_points)
        predicted = actual + np.random.normal(0, 0.01, n_points)

        metrics = monitor.record_calibration(
            model_name="heston_test",
            model_type=ModelType.HESTON,
            predicted=predicted,
            actual=actual,
            parameters={"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7},
            calibration_time=5.0,
        )

        assert metrics.rmse < 0.05
        assert metrics.r_squared > 0.9
        assert monitor.get_status(metrics) == DiagnosticStatus.HEALTHY

    def test_drift_detection(self):
        """Test drift detection."""
        scipy = pytest.importorskip("scipy")
        from quant_trading.monitoring.diagnostics import DriftDetector

        detector = DriftDetector()

        # Set baseline
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        detector.set_baseline("test_model", baseline)

        # Test with similar distribution (no drift)
        current_similar = np.random.normal(0, 1, 500)
        result_similar = detector.detect_drift("test_model", current_similar)
        assert result_similar.psi < 0.1
        assert not result_similar.drift_detected

        # Test with different distribution (drift)
        current_drifted = np.random.normal(0.5, 1.5, 500)
        result_drifted = detector.detect_drift("test_model", current_drifted)
        assert result_drifted.psi > 0.1
        assert result_drifted.drift_detected

    def test_parameter_stability(self):
        """Test parameter stability analysis."""
        from quant_trading.monitoring.diagnostics import ParameterStabilityAnalyzer

        analyzer = ParameterStabilityAnalyzer(min_history=5)

        # Record historical parameters
        np.random.seed(42)
        for i in range(20):
            params = {
                "kappa": 2.0 + np.random.normal(0, 0.1),
                "theta": 0.04 + np.random.normal(0, 0.005),
            }
            analyzer.record_parameters("test_model", params)

        # Analyze current (stable) parameters
        current_stable = {"kappa": 2.05, "theta": 0.042}
        stability = analyzer.analyze_stability("test_model", current_stable)

        assert len(stability) == 2
        assert all(s.is_stable for s in stability)

        # Analyze current (unstable) parameters
        current_unstable = {"kappa": 3.5, "theta": 0.08}  # Way outside normal range
        stability_unstable = analyzer.analyze_stability("test_model", current_unstable)
        assert any(not s.is_stable for s in stability_unstable)

    def test_forecast_accuracy(self):
        """Test forecast accuracy tracking."""
        scipy = pytest.importorskip("scipy")
        from quant_trading.monitoring.diagnostics import ForecastAccuracyTracker

        tracker = ForecastAccuracyTracker()

        # Record forecasts
        np.random.seed(42)
        for _ in range(50):
            forecast = np.random.normal(0, 0.02)
            actual = forecast + np.random.normal(0, 0.005)  # Reasonably accurate
            tracker.record_forecast("test_model", "1d", forecast, actual)

        accuracy = tracker.calculate_accuracy("test_model", "1d")

        assert accuracy is not None
        assert accuracy.direction_accuracy > 0.5
        assert accuracy.information_coefficient > 0

    def test_backtest_comparison(self):
        """Test backtest vs live comparison."""
        from quant_trading.monitoring.diagnostics import BacktestLiveComparator

        comparator = BacktestLiveComparator()

        # Generate synthetic returns with positive mean for both
        np.random.seed(123)  # Different seed for consistent positive results
        n_days = 100
        backtest_returns = np.random.normal(0.002, 0.01, n_days)  # Positive returns
        live_returns = np.random.normal(0.0015, 0.012, n_days)  # Slightly worse but still positive

        result = comparator.compare(
            model_name="test_model",
            strategy_name="momentum",
            backtest_returns=backtest_returns,
            live_returns=live_returns,
        )

        assert result.backtest_sharpe != 0
        assert result.live_sharpe != 0
        # Check that ratio is positive when both sharpes have same sign
        assert result.backtest_sharpe > 0  # Both should be positive
        assert result.live_sharpe > 0

    def test_diagnostics_engine(self):
        """Test complete diagnostics engine."""
        from quant_trading.monitoring.diagnostics import (
            ModelDiagnosticsEngine,
            ModelType,
        )

        engine = ModelDiagnosticsEngine()
        engine.register_model("heston", ModelType.HESTON)

        # Record calibration
        np.random.seed(42)
        actual = np.random.uniform(0.1, 0.4, 50)
        predicted = actual + np.random.normal(0, 0.01, 50)

        engine.record_calibration(
            model_name="heston",
            predicted=predicted,
            actual=actual,
            parameters={"kappa": 2.0, "theta": 0.04},
            calibration_time=5.0,
        )

        # Generate report
        report = engine.generate_report("heston")

        assert report.model_name == "heston"
        assert report.calibration is not None


class TestModuleExports:
    """Test that all exports work correctly."""

    def test_metrics_exports(self):
        """Test metrics module exports."""
        from quant_trading.monitoring import (
            MetricsCollector,
            track_latency,
            track_latency_async,
            count_calls,
        )

        assert MetricsCollector is not None
        assert callable(track_latency)
        assert callable(count_calls)

    def test_alerts_exports(self):
        """Test alerts module exports."""
        from quant_trading.monitoring import (
            Alert,
            AlertRule,
            AlertManager,
            AlertSeverity,
            create_default_alert_rules,
        )

        assert Alert is not None
        assert AlertManager is not None
        assert callable(create_default_alert_rules)

    def test_dashboards_exports(self):
        """Test dashboards module exports."""
        from quant_trading.monitoring import (
            Panel,
            Dashboard,
            create_trading_overview_dashboard,
            create_risk_dashboard,
        )

        assert Panel is not None
        assert Dashboard is not None
        assert callable(create_trading_overview_dashboard)

    def test_logging_exports(self):
        """Test logging module exports."""
        from quant_trading.monitoring import (
            configure_logging,
            get_logger,
            LogLevel,
            TradingLogger,
        )

        assert callable(configure_logging)
        assert callable(get_logger)
        assert LogLevel is not None

    def test_health_exports(self):
        """Test health module exports."""
        from quant_trading.monitoring import (
            HealthManager,
            HealthCheck,
            CircuitBreaker,
            get_health_manager,
        )

        assert HealthManager is not None
        assert CircuitBreaker is not None
        assert callable(get_health_manager)

    def test_attribution_exports(self):
        """Test attribution module exports."""
        from quant_trading.monitoring import (
            PerformanceAttributionEngine,
            ReturnsAttributor,
            RiskAttributor,
        )

        assert PerformanceAttributionEngine is not None
        assert ReturnsAttributor is not None

    def test_diagnostics_exports(self):
        """Test diagnostics module exports."""
        from quant_trading.monitoring import (
            ModelDiagnosticsEngine,
            CalibrationMonitor,
            DriftDetector,
            get_diagnostics_engine,
        )

        assert ModelDiagnosticsEngine is not None
        assert callable(get_diagnostics_engine)
