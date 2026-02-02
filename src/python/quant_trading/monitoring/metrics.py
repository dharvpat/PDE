"""
Metrics Collection and Prometheus Integration.

This module provides standardized metrics collection across the trading system:
- System metrics (CPU, memory, disk)
- Application metrics (latency, throughput, errors)
- Business metrics (P&L, trades, signals, positions)
- Risk metrics (VaR, Greeks, drawdown)
- Data quality metrics

All metrics are exposed via Prometheus format for scraping by monitoring systems.

References:
    - Prometheus best practices: https://prometheus.io/docs/practices/naming/
    - Trading system monitoring: Design doc Section 6
"""

import logging
import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

logger = logging.getLogger(__name__)


# Try to import prometheus_client, fall back to mock if not installed
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        Info,
        CollectorRegistry,
        REGISTRY,
        generate_latest,
        start_http_server,
        push_to_gateway,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, using mock metrics")


# =============================================================================
# Mock Prometheus Classes (for testing without prometheus_client)
# =============================================================================

if not PROMETHEUS_AVAILABLE:
    class MockMetric:
        """Mock metric for testing."""
        def __init__(self, name, description, labelnames=None, **kwargs):
            self.name = name
            self.description = description
            self.labelnames = labelnames or []
            self._value = 0
            self._labels = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = MockMetric(self.name, self.description)
            return self._labels[key]

        def inc(self, amount=1):
            self._value += amount

        def dec(self, amount=1):
            self._value -= amount

        def set(self, value):
            self._value = value

        def observe(self, value):
            self._value = value

        def info(self, value):
            self._value = value

    Counter = Gauge = Histogram = Summary = Info = MockMetric
    REGISTRY = None

    def start_http_server(port):
        logger.info(f"Mock metrics server on port {port}")

    def generate_latest(registry=None):
        return b""


# =============================================================================
# System Metrics
# =============================================================================

# Overall system health (0-1 scale)
system_health = Gauge(
    'trading_system_health',
    'Overall system health score (0=unhealthy, 1=healthy)'
)

# Component status
component_status = Gauge(
    'trading_component_status',
    'Component status (1=healthy, 0=unhealthy)',
    ['component', 'instance']
)

# System resources
cpu_usage_percent = Gauge(
    'trading_cpu_usage_percent',
    'CPU usage percentage'
)

memory_usage_bytes = Gauge(
    'trading_memory_usage_bytes',
    'Memory usage in bytes'
)

memory_usage_percent = Gauge(
    'trading_memory_usage_percent',
    'Memory usage percentage'
)

disk_usage_percent = Gauge(
    'trading_disk_usage_percent',
    'Disk usage percentage',
    ['mount_point']
)

# Process metrics
process_uptime_seconds = Gauge(
    'trading_process_uptime_seconds',
    'Process uptime in seconds'
)

thread_count = Gauge(
    'trading_thread_count',
    'Number of active threads'
)

open_file_descriptors = Gauge(
    'trading_open_file_descriptors',
    'Number of open file descriptors'
)


# =============================================================================
# Trading Metrics
# =============================================================================

# Orders
orders_total = Counter(
    'trading_orders_total',
    'Total number of orders',
    ['strategy', 'symbol', 'side', 'order_type', 'status']
)

orders_created = Counter(
    'trading_orders_created_total',
    'Total orders created',
    ['strategy', 'symbol', 'side']
)

orders_filled = Counter(
    'trading_orders_filled_total',
    'Total orders filled',
    ['strategy', 'symbol', 'side']
)

orders_rejected = Counter(
    'trading_orders_rejected_total',
    'Total orders rejected',
    ['strategy', 'symbol', 'reason']
)

orders_cancelled = Counter(
    'trading_orders_cancelled_total',
    'Total orders cancelled',
    ['strategy', 'symbol']
)

# Order values
order_value_total = Counter(
    'trading_order_value_total',
    'Total order value in dollars',
    ['strategy', 'side']
)

# P&L
realized_pnl = Gauge(
    'trading_realized_pnl_total',
    'Total realized P&L in dollars',
    ['strategy']
)

unrealized_pnl = Gauge(
    'trading_unrealized_pnl_current',
    'Current unrealized P&L in dollars',
    ['strategy']
)

daily_pnl = Gauge(
    'trading_daily_pnl',
    'Daily P&L in dollars',
    ['strategy', 'date']
)

cumulative_pnl = Gauge(
    'trading_cumulative_pnl',
    'Cumulative P&L in dollars',
    ['strategy']
)

# Positions
open_positions_count = Gauge(
    'trading_open_positions_count',
    'Number of open positions',
    ['strategy']
)

total_exposure = Gauge(
    'trading_total_exposure_dollars',
    'Total portfolio exposure in dollars',
    ['strategy', 'asset_class']
)

position_size = Gauge(
    'trading_position_size',
    'Position size in shares/contracts',
    ['strategy', 'symbol']
)

# Signals
signals_generated = Counter(
    'trading_signals_generated_total',
    'Total signals generated',
    ['strategy', 'signal_type', 'direction']
)

signals_acted_on = Counter(
    'trading_signals_acted_on_total',
    'Signals that resulted in orders',
    ['strategy', 'signal_type']
)

signal_strength = Gauge(
    'trading_signal_strength',
    'Current signal strength (-1 to 1)',
    ['strategy', 'symbol']
)


# =============================================================================
# Risk Metrics
# =============================================================================

# VaR
portfolio_var = Gauge(
    'trading_portfolio_var',
    'Portfolio Value at Risk',
    ['confidence_level', 'time_horizon']
)

# Greeks
portfolio_delta = Gauge(
    'trading_portfolio_delta',
    'Total portfolio delta exposure',
    ['strategy']
)

portfolio_gamma = Gauge(
    'trading_portfolio_gamma',
    'Total portfolio gamma exposure',
    ['strategy']
)

portfolio_vega = Gauge(
    'trading_portfolio_vega',
    'Total portfolio vega exposure',
    ['strategy']
)

portfolio_theta = Gauge(
    'trading_portfolio_theta',
    'Total portfolio theta exposure',
    ['strategy']
)

# Drawdown
max_drawdown_percent = Gauge(
    'trading_max_drawdown_percent',
    'Maximum drawdown from peak (%)',
    ['strategy']
)

current_drawdown_percent = Gauge(
    'trading_current_drawdown_percent',
    'Current drawdown from peak (%)',
    ['strategy']
)

# Sharpe
rolling_sharpe = Gauge(
    'trading_rolling_sharpe_ratio',
    'Rolling Sharpe ratio',
    ['strategy', 'window']
)

# Volatility
portfolio_volatility = Gauge(
    'trading_portfolio_volatility',
    'Portfolio volatility (annualized)',
    ['strategy']
)

# Concentration
position_concentration = Gauge(
    'trading_position_concentration',
    'Position concentration (HHI)',
    ['strategy']
)


# =============================================================================
# Latency Metrics
# =============================================================================

# Signal generation latency
signal_latency = Histogram(
    'trading_signal_generation_latency_seconds',
    'Time to generate trading signal',
    ['strategy'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Order submission latency
order_submission_latency = Histogram(
    'trading_order_submission_latency_seconds',
    'Time to submit order to broker',
    ['broker'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Order fill latency
order_fill_latency = Histogram(
    'trading_order_fill_latency_seconds',
    'Time from order submission to fill',
    ['broker', 'order_type'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
)

# Data ingestion latency
data_ingestion_latency = Histogram(
    'trading_data_ingestion_latency_seconds',
    'Time to ingest market data',
    ['source', 'data_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

# Model calibration latency
calibration_latency = Histogram(
    'trading_calibration_latency_seconds',
    'Time to calibrate model',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

# API response latency
api_latency = Histogram(
    'trading_api_latency_seconds',
    'API endpoint response time',
    ['endpoint', 'method'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)


# =============================================================================
# Data Quality Metrics
# =============================================================================

# Data gaps
data_gaps_detected = Counter(
    'trading_data_gaps_detected_total',
    'Number of data gaps detected',
    ['symbol', 'frequency']
)

# Validation failures
data_validation_failures = Counter(
    'trading_data_validation_failures_total',
    'Number of data validation failures',
    ['symbol', 'check_type']
)

# Data freshness
data_age_seconds = Gauge(
    'trading_data_age_seconds',
    'Age of latest data in seconds',
    ['symbol', 'data_type']
)

# Data completeness
data_completeness_percent = Gauge(
    'trading_data_completeness_percent',
    'Data completeness percentage',
    ['symbol', 'date']
)


# =============================================================================
# Model Metrics
# =============================================================================

# Calibration quality
calibration_rmse = Gauge(
    'trading_calibration_rmse',
    'Model calibration RMSE',
    ['model', 'symbol']
)

# Parameter values
model_parameter = Gauge(
    'trading_model_parameter',
    'Model parameter value',
    ['model', 'parameter']
)

# Prediction accuracy
model_prediction_error = Histogram(
    'trading_model_prediction_error',
    'Model prediction error distribution',
    ['model'],
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
)


# =============================================================================
# Utility Functions
# =============================================================================

def track_latency(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to track function execution latency.

    Args:
        metric: Histogram metric to record latency
        labels: Optional labels for the metric

    Example:
        @track_latency(signal_latency, {'strategy': 'vol_arb'})
        def generate_signal():
            # Signal generation logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(elapsed)
                else:
                    metric.observe(elapsed)
        return wrapper
    return decorator


def track_latency_async(metric: Histogram, labels: Optional[Dict[str, str]] = None):
    """Async version of track_latency decorator."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(elapsed)
                else:
                    metric.observe(elapsed)
        return wrapper
    return decorator


def count_calls(counter: Counter, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.

    Args:
        counter: Counter metric
        labels: Optional labels

    Example:
        @count_calls(orders_created, {'strategy': 'vol_arb'})
        def create_order():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if labels:
                counter.labels(**labels).inc()
            else:
                counter.inc()
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Metrics Collector Class
# =============================================================================

class MetricsCollector:
    """
    Central metrics collector for trading system.

    Provides a unified interface for recording all system metrics and manages
    the Prometheus HTTP server for metric scraping.

    Example:
        collector = MetricsCollector()
        collector.start_server(port=8000)

        # Record order
        collector.record_order_created('vol_arb', 'SPY', 'BUY')

        # Update P&L
        collector.update_pnl('vol_arb', realized=1500, unrealized=300)
    """

    def __init__(
        self,
        port: int = 9090,
        collect_system_metrics: bool = True,
        system_metrics_interval: float = 15.0
    ):
        """
        Initialize metrics collector.

        Args:
            port: Port for Prometheus metrics endpoint
            collect_system_metrics: Whether to auto-collect system metrics
            system_metrics_interval: Interval between system metric collections
        """
        self.port = port
        self.collect_system_metrics = collect_system_metrics
        self.system_metrics_interval = system_metrics_interval

        self._server_started = False
        self._system_metrics_thread: Optional[threading.Thread] = None
        self._running = False
        self._start_time = time.time()

    def start_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if not self._server_started:
            start_http_server(self.port)
            self._server_started = True
            logger.info(f"Metrics server started on port {self.port}")

            if self.collect_system_metrics:
                self._start_system_metrics_collection()

    def stop(self) -> None:
        """Stop metrics collection."""
        self._running = False
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5)
        logger.info("Metrics collector stopped")

    def _start_system_metrics_collection(self) -> None:
        """Start background thread for system metrics."""
        self._running = True
        self._system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics_loop,
            daemon=True
        )
        self._system_metrics_thread.start()
        logger.info("System metrics collection started")

    def _collect_system_metrics_loop(self) -> None:
        """Background loop to collect system metrics."""
        while self._running:
            try:
                self._update_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            time.sleep(self.system_metrics_interval)

    def _update_system_metrics(self) -> None:
        """Update system resource metrics."""
        # CPU
        cpu_usage_percent.set(psutil.cpu_percent())

        # Memory
        mem = psutil.virtual_memory()
        memory_usage_bytes.set(mem.used)
        memory_usage_percent.set(mem.percent)

        # Disk
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage_percent.labels(mount_point=partition.mountpoint).set(
                    usage.percent
                )
            except PermissionError:
                pass

        # Process
        process = psutil.Process()
        process_uptime_seconds.set(time.time() - self._start_time)
        thread_count.set(threading.active_count())

        try:
            open_file_descriptors.set(process.num_fds())
        except AttributeError:
            # Windows doesn't have num_fds
            pass

    # =========================================================================
    # Order Metrics
    # =========================================================================

    def record_order_created(
        self,
        strategy: str,
        symbol: str,
        side: str,
        value: Optional[float] = None
    ) -> None:
        """Record order creation."""
        orders_created.labels(
            strategy=strategy,
            symbol=symbol,
            side=side
        ).inc()

        if value:
            order_value_total.labels(strategy=strategy, side=side).inc(value)

    def record_order_filled(
        self,
        strategy: str,
        symbol: str,
        side: str,
        fill_latency: Optional[float] = None
    ) -> None:
        """Record order fill."""
        orders_filled.labels(
            strategy=strategy,
            symbol=symbol,
            side=side
        ).inc()

        if fill_latency:
            order_fill_latency.labels(
                broker='default',
                order_type='market'
            ).observe(fill_latency)

    def record_order_rejected(
        self,
        strategy: str,
        symbol: str,
        reason: str
    ) -> None:
        """Record order rejection."""
        orders_rejected.labels(
            strategy=strategy,
            symbol=symbol,
            reason=reason
        ).inc()

    def record_order_cancelled(
        self,
        strategy: str,
        symbol: str
    ) -> None:
        """Record order cancellation."""
        orders_cancelled.labels(
            strategy=strategy,
            symbol=symbol
        ).inc()

    # =========================================================================
    # P&L Metrics
    # =========================================================================

    def update_pnl(
        self,
        strategy: str,
        realized: float,
        unrealized: float,
        daily: Optional[float] = None
    ) -> None:
        """Update P&L metrics."""
        realized_pnl.labels(strategy=strategy).set(realized)
        unrealized_pnl.labels(strategy=strategy).set(unrealized)
        cumulative_pnl.labels(strategy=strategy).set(realized + unrealized)

        if daily is not None:
            today = date.today().isoformat()
            daily_pnl.labels(strategy=strategy, date=today).set(daily)

    # =========================================================================
    # Position Metrics
    # =========================================================================

    def update_positions(
        self,
        strategy: str,
        count: int,
        exposure: float,
        asset_class: str = "equity"
    ) -> None:
        """Update position metrics."""
        open_positions_count.labels(strategy=strategy).set(count)
        total_exposure.labels(
            strategy=strategy,
            asset_class=asset_class
        ).set(exposure)

    def update_position_size(
        self,
        strategy: str,
        symbol: str,
        size: float
    ) -> None:
        """Update individual position size."""
        position_size.labels(strategy=strategy, symbol=symbol).set(size)

    # =========================================================================
    # Signal Metrics
    # =========================================================================

    def record_signal_generated(
        self,
        strategy: str,
        signal_type: str,
        direction: str,
        strength: Optional[float] = None,
        symbol: Optional[str] = None
    ) -> None:
        """Record signal generation."""
        signals_generated.labels(
            strategy=strategy,
            signal_type=signal_type,
            direction=direction
        ).inc()

        if strength is not None and symbol:
            signal_strength.labels(strategy=strategy, symbol=symbol).set(strength)

    def record_signal_acted_on(
        self,
        strategy: str,
        signal_type: str
    ) -> None:
        """Record signal that resulted in order."""
        signals_acted_on.labels(
            strategy=strategy,
            signal_type=signal_type
        ).inc()

    # =========================================================================
    # Risk Metrics
    # =========================================================================

    def update_risk_metrics(
        self,
        strategy: str,
        var_95: Optional[float] = None,
        delta: Optional[float] = None,
        gamma: Optional[float] = None,
        vega: Optional[float] = None,
        theta: Optional[float] = None,
        current_drawdown: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        sharpe: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> None:
        """Update risk metrics."""
        if var_95 is not None:
            portfolio_var.labels(
                confidence_level="95",
                time_horizon="1d"
            ).set(var_95)

        if delta is not None:
            portfolio_delta.labels(strategy=strategy).set(delta)
        if gamma is not None:
            portfolio_gamma.labels(strategy=strategy).set(gamma)
        if vega is not None:
            portfolio_vega.labels(strategy=strategy).set(vega)
        if theta is not None:
            portfolio_theta.labels(strategy=strategy).set(theta)

        if current_drawdown is not None:
            current_drawdown_percent.labels(strategy=strategy).set(current_drawdown)
        if max_drawdown is not None:
            max_drawdown_percent.labels(strategy=strategy).set(max_drawdown)

        if sharpe is not None:
            rolling_sharpe.labels(strategy=strategy, window="30d").set(sharpe)

        if volatility is not None:
            portfolio_volatility.labels(strategy=strategy).set(volatility)

    # =========================================================================
    # Data Quality Metrics
    # =========================================================================

    def record_data_gap(
        self,
        symbol: str,
        frequency: str
    ) -> None:
        """Record data gap detection."""
        data_gaps_detected.labels(symbol=symbol, frequency=frequency).inc()

    def record_validation_failure(
        self,
        symbol: str,
        check_type: str
    ) -> None:
        """Record data validation failure."""
        data_validation_failures.labels(
            symbol=symbol,
            check_type=check_type
        ).inc()

    def update_data_freshness(
        self,
        symbol: str,
        data_type: str,
        age_seconds: float
    ) -> None:
        """Update data freshness metric."""
        data_age_seconds.labels(symbol=symbol, data_type=data_type).set(age_seconds)

    # =========================================================================
    # Model Metrics
    # =========================================================================

    def update_calibration_quality(
        self,
        model: str,
        symbol: str,
        rmse: float
    ) -> None:
        """Update model calibration quality."""
        calibration_rmse.labels(model=model, symbol=symbol).set(rmse)

    def update_model_parameter(
        self,
        model: str,
        parameter: str,
        value: float
    ) -> None:
        """Update model parameter value."""
        model_parameter.labels(model=model, parameter=parameter).set(value)

    # =========================================================================
    # System Health
    # =========================================================================

    def update_system_health(self, health_score: float) -> None:
        """
        Update overall system health score.

        Args:
            health_score: Health score from 0 (unhealthy) to 1 (healthy)
        """
        system_health.set(health_score)

    def set_component_status(
        self,
        component: str,
        is_healthy: bool,
        instance: str = "default"
    ) -> None:
        """
        Set component health status.

        Args:
            component: Component name
            is_healthy: True if healthy
            instance: Instance identifier
        """
        component_status.labels(
            component=component,
            instance=instance
        ).set(1 if is_healthy else 0)

    # =========================================================================
    # Latency Recording
    # =========================================================================

    def record_signal_latency(
        self,
        strategy: str,
        latency_seconds: float
    ) -> None:
        """Record signal generation latency."""
        signal_latency.labels(strategy=strategy).observe(latency_seconds)

    def record_order_submission_latency(
        self,
        broker: str,
        latency_seconds: float
    ) -> None:
        """Record order submission latency."""
        order_submission_latency.labels(broker=broker).observe(latency_seconds)

    def record_data_ingestion_latency(
        self,
        source: str,
        data_type: str,
        latency_seconds: float
    ) -> None:
        """Record data ingestion latency."""
        data_ingestion_latency.labels(
            source=source,
            data_type=data_type
        ).observe(latency_seconds)

    def record_calibration_latency(
        self,
        model: str,
        latency_seconds: float
    ) -> None:
        """Record model calibration latency."""
        calibration_latency.labels(model=model).observe(latency_seconds)

    def record_api_latency(
        self,
        endpoint: str,
        method: str,
        latency_seconds: float
    ) -> None:
        """Record API response latency."""
        api_latency.labels(endpoint=endpoint, method=method).observe(latency_seconds)

    # =========================================================================
    # Utility
    # =========================================================================

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY).decode('utf-8')
        return ""

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics."""
        return {
            'server_port': self.port,
            'server_started': self._server_started,
            'uptime_seconds': time.time() - self._start_time,
            'system_metrics_enabled': self.collect_system_metrics
        }
