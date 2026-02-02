"""
Data Quality Monitoring Module.

This module provides comprehensive data quality monitoring:
- Real-time data quality metrics
- Anomaly detection and alerting
- Data freshness tracking
- Gap detection and reporting
- Provider health monitoring
- Metric aggregation and dashboards

Designed for production trading systems requiring high data integrity.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of data quality alerts."""
    DATA_GAP = "data_gap"
    STALE_DATA = "stale_data"
    INVALID_PRICE = "invalid_price"
    MISSING_SYMBOL = "missing_symbol"
    PROVIDER_DOWN = "provider_down"
    HIGH_LATENCY = "high_latency"
    ANOMALOUS_VALUE = "anomalous_value"
    VALIDATION_FAILURE = "validation_failure"
    RATE_LIMIT = "rate_limit"
    CONNECTION_LOST = "connection_lost"


@dataclass
class DataQualityAlert:
    """Represents a data quality alert."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    provider: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()


@dataclass
class DataQualityMetric:
    """Single data quality metric observation."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    provider: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class MetricAggregator:
    """
    Aggregates metrics over time windows.

    Calculates statistics like mean, std, min, max, percentiles.
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize aggregator.

        Args:
            window_size: Number of observations to keep
        """
        self.window_size = window_size
        self._values: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        self._count = 0
        self._sum = 0.0
        self._sum_sq = 0.0

    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the aggregator."""
        if len(self._values) == self.window_size:
            # Remove oldest value from running stats
            old_value = self._values[0]
            self._sum -= old_value
            self._sum_sq -= old_value ** 2

        self._values.append(value)
        self._timestamps.append(timestamp or datetime.now())
        self._sum += value
        self._sum_sq += value ** 2
        self._count = min(self._count + 1, self.window_size)

    @property
    def mean(self) -> float:
        """Calculate mean."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        if self._count < 2:
            return 0.0
        variance = (self._sum_sq / self._count) - (self.mean ** 2)
        return np.sqrt(max(0, variance))

    @property
    def min(self) -> float:
        """Get minimum value."""
        if not self._values:
            return 0.0
        return min(self._values)

    @property
    def max(self) -> float:
        """Get maximum value."""
        if not self._values:
            return 0.0
        return max(self._values)

    def percentile(self, p: float) -> float:
        """Calculate percentile."""
        if not self._values:
            return 0.0
        return float(np.percentile(list(self._values), p))

    def get_stats(self) -> Dict[str, float]:
        """Get all statistics."""
        return {
            'count': self._count,
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'p50': self.percentile(50),
            'p95': self.percentile(95),
            'p99': self.percentile(99)
        }


class SymbolHealthTracker:
    """Tracks health metrics for a single symbol."""

    def __init__(
        self,
        symbol: str,
        stale_threshold_seconds: float = 60.0,
        expected_frequency_seconds: float = 1.0
    ):
        """
        Initialize tracker.

        Args:
            symbol: Symbol being tracked
            stale_threshold_seconds: Time before data is considered stale
            expected_frequency_seconds: Expected time between updates
        """
        self.symbol = symbol
        self.stale_threshold = stale_threshold_seconds
        self.expected_frequency = expected_frequency_seconds

        self.last_update: Optional[datetime] = None
        self.last_price: Optional[float] = None
        self.update_count = 0
        self.error_count = 0
        self.gap_count = 0

        self._latency_aggregator = MetricAggregator(window_size=1000)
        self._price_aggregator = MetricAggregator(window_size=1000)
        self._update_intervals: deque = deque(maxlen=100)

    def record_update(
        self,
        price: float,
        timestamp: datetime,
        latency_ms: Optional[float] = None
    ) -> List[DataQualityAlert]:
        """
        Record a data update for this symbol.

        Args:
            price: Current price
            timestamp: Data timestamp
            latency_ms: End-to-end latency in milliseconds

        Returns:
            List of any alerts triggered
        """
        alerts = []
        now = datetime.now()

        # Check for gaps
        if self.last_update:
            interval = (timestamp - self.last_update).total_seconds()
            self._update_intervals.append(interval)

            if interval > self.stale_threshold:
                self.gap_count += 1
                alerts.append(DataQualityAlert(
                    alert_type=AlertType.DATA_GAP,
                    severity=AlertSeverity.WARNING,
                    message=f"Data gap detected for {self.symbol}: {interval:.1f}s",
                    symbol=self.symbol,
                    metadata={'gap_seconds': interval}
                ))

        # Check for anomalous prices
        if self.last_price and price > 0:
            price_change_pct = abs(price - self.last_price) / self.last_price * 100
            if price_change_pct > 10:  # 10% move
                alerts.append(DataQualityAlert(
                    alert_type=AlertType.ANOMALOUS_VALUE,
                    severity=AlertSeverity.WARNING,
                    message=f"Large price move for {self.symbol}: {price_change_pct:.1f}%",
                    symbol=self.symbol,
                    metadata={
                        'old_price': self.last_price,
                        'new_price': price,
                        'change_pct': price_change_pct
                    }
                ))

        # Record metrics
        if latency_ms:
            self._latency_aggregator.add(latency_ms, timestamp)

            if latency_ms > 1000:  # 1 second latency
                alerts.append(DataQualityAlert(
                    alert_type=AlertType.HIGH_LATENCY,
                    severity=AlertSeverity.WARNING,
                    message=f"High latency for {self.symbol}: {latency_ms:.0f}ms",
                    symbol=self.symbol,
                    metadata={'latency_ms': latency_ms}
                ))

        self._price_aggregator.add(price, timestamp)

        # Update state
        self.last_update = timestamp
        self.last_price = price
        self.update_count += 1

        return alerts

    def record_error(self, error_type: str, message: str) -> DataQualityAlert:
        """Record an error for this symbol."""
        self.error_count += 1
        return DataQualityAlert(
            alert_type=AlertType.VALIDATION_FAILURE,
            severity=AlertSeverity.ERROR,
            message=f"Error for {self.symbol}: {message}",
            symbol=self.symbol,
            metadata={'error_type': error_type}
        )

    def is_stale(self) -> bool:
        """Check if data is stale."""
        if not self.last_update:
            return True
        age = (datetime.now() - self.last_update).total_seconds()
        return age > self.stale_threshold

    def get_health_score(self) -> float:
        """
        Calculate health score from 0 (bad) to 1 (good).

        Based on:
        - Update frequency vs expected
        - Error rate
        - Latency
        - Gap count
        """
        if self.update_count == 0:
            return 0.0

        # Frequency score (1.0 if meeting expected frequency)
        if self._update_intervals:
            avg_interval = np.mean(list(self._update_intervals))
            freq_score = min(1.0, self.expected_frequency / max(avg_interval, 0.001))
        else:
            freq_score = 0.5

        # Error rate score
        error_rate = self.error_count / self.update_count
        error_score = max(0, 1.0 - error_rate * 10)  # Each 10% error rate reduces score by 1.0

        # Latency score
        latency_stats = self._latency_aggregator.get_stats()
        p95_latency = latency_stats.get('p95', 100)
        latency_score = max(0, 1.0 - p95_latency / 1000)  # 1s latency = 0 score

        # Gap penalty
        gap_penalty = min(0.5, self.gap_count * 0.1)

        # Combined score
        score = (freq_score + error_score + latency_score) / 3 - gap_penalty
        return max(0, min(1, score))

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics for this symbol."""
        return {
            'symbol': self.symbol,
            'last_update': self.last_update,
            'last_price': self.last_price,
            'update_count': self.update_count,
            'error_count': self.error_count,
            'gap_count': self.gap_count,
            'is_stale': self.is_stale(),
            'health_score': self.get_health_score(),
            'latency': self._latency_aggregator.get_stats(),
            'price': self._price_aggregator.get_stats()
        }


class ProviderHealthTracker:
    """Tracks health metrics for a data provider."""

    def __init__(
        self,
        provider_name: str,
        connection_timeout_seconds: float = 30.0
    ):
        """
        Initialize tracker.

        Args:
            provider_name: Name of the provider
            connection_timeout_seconds: Time before connection is considered failed
        """
        self.provider_name = provider_name
        self.connection_timeout = connection_timeout_seconds

        self.is_connected = False
        self.last_connection_time: Optional[datetime] = None
        self.last_disconnection_time: Optional[datetime] = None
        self.last_message_time: Optional[datetime] = None

        self.connection_count = 0
        self.disconnection_count = 0
        self.message_count = 0
        self.error_count = 0
        self.rate_limit_count = 0

        self._latency_aggregator = MetricAggregator(window_size=1000)
        self._throughput_tracker: deque = deque(maxlen=60)  # Per-second counts
        self._current_second_count = 0
        self._current_second = 0

    def record_connection(self) -> None:
        """Record a successful connection."""
        self.is_connected = True
        self.last_connection_time = datetime.now()
        self.connection_count += 1
        logger.info(f"Provider {self.provider_name} connected")

    def record_disconnection(self, reason: str = "") -> DataQualityAlert:
        """Record a disconnection."""
        self.is_connected = False
        self.last_disconnection_time = datetime.now()
        self.disconnection_count += 1
        logger.warning(f"Provider {self.provider_name} disconnected: {reason}")

        return DataQualityAlert(
            alert_type=AlertType.CONNECTION_LOST,
            severity=AlertSeverity.ERROR,
            message=f"Provider {self.provider_name} disconnected: {reason}",
            provider=self.provider_name,
            metadata={'reason': reason}
        )

    def record_message(self, latency_ms: Optional[float] = None) -> None:
        """Record a received message."""
        now = datetime.now()
        self.last_message_time = now
        self.message_count += 1

        if latency_ms:
            self._latency_aggregator.add(latency_ms)

        # Update throughput tracking
        current_second = int(time.time())
        if current_second != self._current_second:
            self._throughput_tracker.append(self._current_second_count)
            self._current_second = current_second
            self._current_second_count = 1
        else:
            self._current_second_count += 1

    def record_error(self, error_type: str) -> None:
        """Record an error."""
        self.error_count += 1

    def record_rate_limit(self) -> DataQualityAlert:
        """Record a rate limit hit."""
        self.rate_limit_count += 1
        return DataQualityAlert(
            alert_type=AlertType.RATE_LIMIT,
            severity=AlertSeverity.WARNING,
            message=f"Rate limit hit for {self.provider_name}",
            provider=self.provider_name
        )

    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        if not self.is_connected:
            return False

        if self.last_message_time:
            age = (datetime.now() - self.last_message_time).total_seconds()
            if age > self.connection_timeout:
                return False

        return True

    def get_throughput(self) -> float:
        """Get average messages per second."""
        if not self._throughput_tracker:
            return 0.0
        return np.mean(list(self._throughput_tracker))

    def get_health_score(self) -> float:
        """Calculate provider health score."""
        if not self.is_connected:
            return 0.0

        # Connection stability
        if self.connection_count > 0:
            reconnect_rate = self.disconnection_count / self.connection_count
            stability_score = max(0, 1.0 - reconnect_rate)
        else:
            stability_score = 0.0

        # Error rate
        if self.message_count > 0:
            error_rate = self.error_count / self.message_count
            error_score = max(0, 1.0 - error_rate * 100)
        else:
            error_score = 0.5

        # Latency score
        latency_stats = self._latency_aggregator.get_stats()
        p95_latency = latency_stats.get('p95', 100)
        latency_score = max(0, 1.0 - p95_latency / 500)

        return (stability_score + error_score + latency_score) / 3

    def get_metrics(self) -> Dict[str, Any]:
        """Get all provider metrics."""
        return {
            'provider': self.provider_name,
            'is_connected': self.is_connected,
            'is_healthy': self.is_healthy(),
            'health_score': self.get_health_score(),
            'last_connection': self.last_connection_time,
            'last_message': self.last_message_time,
            'connection_count': self.connection_count,
            'disconnection_count': self.disconnection_count,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'rate_limit_count': self.rate_limit_count,
            'throughput': self.get_throughput(),
            'latency': self._latency_aggregator.get_stats()
        }


AlertHandler = Callable[[DataQualityAlert], None]


class DataQualityMonitor:
    """
    Central data quality monitoring system.

    Coordinates symbol and provider tracking, alert management, and reporting.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 60.0,
        alert_cooldown_seconds: float = 300.0
    ):
        """
        Initialize monitor.

        Args:
            stale_threshold_seconds: Default stale data threshold
            alert_cooldown_seconds: Minimum time between same alerts
        """
        self.stale_threshold = stale_threshold_seconds
        self.alert_cooldown = alert_cooldown_seconds

        self._symbol_trackers: Dict[str, SymbolHealthTracker] = {}
        self._provider_trackers: Dict[str, ProviderHealthTracker] = {}
        self._alerts: List[DataQualityAlert] = []
        self._alert_handlers: List[AlertHandler] = []
        self._alert_history: Dict[str, datetime] = {}  # For cooldown

        # Metric storage
        self._metrics: Dict[str, MetricAggregator] = defaultdict(MetricAggregator)

    def register_symbol(
        self,
        symbol: str,
        expected_frequency_seconds: float = 1.0
    ) -> None:
        """Register a symbol for monitoring."""
        self._symbol_trackers[symbol] = SymbolHealthTracker(
            symbol=symbol,
            stale_threshold_seconds=self.stale_threshold,
            expected_frequency_seconds=expected_frequency_seconds
        )
        logger.info(f"Registered symbol for monitoring: {symbol}")

    def register_provider(
        self,
        provider_name: str,
        connection_timeout_seconds: float = 30.0
    ) -> None:
        """Register a provider for monitoring."""
        self._provider_trackers[provider_name] = ProviderHealthTracker(
            provider_name=provider_name,
            connection_timeout_seconds=connection_timeout_seconds
        )
        logger.info(f"Registered provider for monitoring: {provider_name}")

    def add_alert_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler callback."""
        self._alert_handlers.append(handler)

    def _emit_alert(self, alert: DataQualityAlert) -> None:
        """Emit an alert to all handlers."""
        # Check cooldown
        alert_key = f"{alert.alert_type.value}:{alert.symbol}:{alert.provider}"
        last_alert = self._alert_history.get(alert_key)

        if last_alert:
            elapsed = (datetime.now() - last_alert).total_seconds()
            if elapsed < self.alert_cooldown:
                return  # Skip due to cooldown

        self._alert_history[alert_key] = datetime.now()
        self._alerts.append(alert)

        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def record_data_update(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        provider: Optional[str] = None,
        latency_ms: Optional[float] = None
    ) -> None:
        """
        Record a data update.

        Args:
            symbol: Symbol that was updated
            price: Current price
            timestamp: Data timestamp
            provider: Provider name
            latency_ms: End-to-end latency
        """
        # Track symbol metrics
        if symbol not in self._symbol_trackers:
            self.register_symbol(symbol)

        alerts = self._symbol_trackers[symbol].record_update(
            price=price,
            timestamp=timestamp,
            latency_ms=latency_ms
        )

        for alert in alerts:
            alert.provider = provider
            self._emit_alert(alert)

        # Track provider metrics
        if provider and provider in self._provider_trackers:
            self._provider_trackers[provider].record_message(latency_ms)

        # Record general metrics
        self._metrics['update_count'].add(1, timestamp)
        if latency_ms:
            self._metrics['latency_ms'].add(latency_ms, timestamp)

    def record_provider_connection(self, provider_name: str) -> None:
        """Record a provider connection."""
        if provider_name not in self._provider_trackers:
            self.register_provider(provider_name)
        self._provider_trackers[provider_name].record_connection()

    def record_provider_disconnection(
        self,
        provider_name: str,
        reason: str = ""
    ) -> None:
        """Record a provider disconnection."""
        if provider_name in self._provider_trackers:
            alert = self._provider_trackers[provider_name].record_disconnection(reason)
            self._emit_alert(alert)

    def record_error(
        self,
        error_type: str,
        message: str,
        symbol: Optional[str] = None,
        provider: Optional[str] = None
    ) -> None:
        """Record an error."""
        self._metrics['error_count'].add(1)

        if symbol and symbol in self._symbol_trackers:
            alert = self._symbol_trackers[symbol].record_error(error_type, message)
            alert.provider = provider
            self._emit_alert(alert)

        if provider and provider in self._provider_trackers:
            self._provider_trackers[provider].record_error(error_type)

    def check_staleness(self) -> List[DataQualityAlert]:
        """Check all symbols for staleness."""
        alerts = []
        for symbol, tracker in self._symbol_trackers.items():
            if tracker.is_stale():
                alert = DataQualityAlert(
                    alert_type=AlertType.STALE_DATA,
                    severity=AlertSeverity.WARNING,
                    message=f"Stale data for {symbol}",
                    symbol=symbol,
                    metadata={'last_update': tracker.last_update}
                )
                alerts.append(alert)
                self._emit_alert(alert)
        return alerts

    def check_provider_health(self) -> List[DataQualityAlert]:
        """Check all providers for health issues."""
        alerts = []
        for provider_name, tracker in self._provider_trackers.items():
            if not tracker.is_healthy():
                alert = DataQualityAlert(
                    alert_type=AlertType.PROVIDER_DOWN,
                    severity=AlertSeverity.ERROR,
                    message=f"Provider unhealthy: {provider_name}",
                    provider=provider_name,
                    metadata={
                        'is_connected': tracker.is_connected,
                        'last_message': tracker.last_message_time
                    }
                )
                alerts.append(alert)
                self._emit_alert(alert)
        return alerts

    def get_symbol_health(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get health metrics for a symbol."""
        tracker = self._symbol_trackers.get(symbol)
        return tracker.get_metrics() if tracker else None

    def get_provider_health(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get health metrics for a provider."""
        tracker = self._provider_trackers.get(provider_name)
        return tracker.get_metrics() if tracker else None

    def get_all_symbol_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all symbols."""
        return {
            symbol: tracker.get_metrics()
            for symbol, tracker in self._symbol_trackers.items()
        }

    def get_all_provider_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health metrics for all providers."""
        return {
            name: tracker.get_metrics()
            for name, tracker in self._provider_trackers.items()
        }

    def get_system_health_score(self) -> float:
        """Calculate overall system health score."""
        scores = []

        # Symbol health
        for tracker in self._symbol_trackers.values():
            scores.append(tracker.get_health_score())

        # Provider health
        for tracker in self._provider_trackers.values():
            scores.append(tracker.get_health_score())

        if not scores:
            return 1.0

        return np.mean(scores)

    def get_active_alerts(self) -> List[DataQualityAlert]:
        """Get all unresolved alerts."""
        return [a for a in self._alerts if not a.resolved]

    def get_alert_summary(self) -> Dict[str, int]:
        """Get count of alerts by type."""
        summary: Dict[str, int] = defaultdict(int)
        for alert in self.get_active_alerts():
            summary[alert.alert_type.value] += 1
        return dict(summary)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health_score': self.get_system_health_score(),
            'symbols': {
                'count': len(self._symbol_trackers),
                'healthy': sum(
                    1 for t in self._symbol_trackers.values()
                    if t.get_health_score() > 0.8
                ),
                'stale': sum(
                    1 for t in self._symbol_trackers.values()
                    if t.is_stale()
                )
            },
            'providers': {
                'count': len(self._provider_trackers),
                'healthy': sum(
                    1 for t in self._provider_trackers.values()
                    if t.is_healthy()
                ),
                'connected': sum(
                    1 for t in self._provider_trackers.values()
                    if t.is_connected
                )
            },
            'alerts': {
                'total': len(self._alerts),
                'active': len(self.get_active_alerts()),
                'by_type': self.get_alert_summary()
            },
            'metrics': {
                name: agg.get_stats()
                for name, agg in self._metrics.items()
            }
        }


class DataQualityReporter:
    """Generates data quality reports."""

    def __init__(self, monitor: DataQualityMonitor):
        """
        Initialize reporter.

        Args:
            monitor: Data quality monitor instance
        """
        self.monitor = monitor

    def generate_symbol_report(
        self,
        symbol: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Generate detailed report for a symbol."""
        health = self.monitor.get_symbol_health(symbol)
        if not health:
            return {'error': f'Symbol {symbol} not found'}

        report = {
            'symbol': symbol,
            'generated_at': datetime.now().isoformat(),
            'health': health,
            'status': 'healthy' if health['health_score'] > 0.8 else 'degraded'
        }

        # Add relevant alerts
        symbol_alerts = [
            a for a in self.monitor.get_active_alerts()
            if a.symbol == symbol
        ]
        report['active_alerts'] = [
            {
                'type': a.alert_type.value,
                'severity': a.severity.value,
                'message': a.message,
                'timestamp': a.timestamp.isoformat()
            }
            for a in symbol_alerts
        ]

        return report

    def generate_provider_report(self, provider_name: str) -> Dict[str, Any]:
        """Generate detailed report for a provider."""
        health = self.monitor.get_provider_health(provider_name)
        if not health:
            return {'error': f'Provider {provider_name} not found'}

        report = {
            'provider': provider_name,
            'generated_at': datetime.now().isoformat(),
            'health': health,
            'status': 'healthy' if health['is_healthy'] else 'unhealthy'
        }

        # Add relevant alerts
        provider_alerts = [
            a for a in self.monitor.get_active_alerts()
            if a.provider == provider_name
        ]
        report['active_alerts'] = [
            {
                'type': a.alert_type.value,
                'severity': a.severity.value,
                'message': a.message,
                'timestamp': a.timestamp.isoformat()
            }
            for a in provider_alerts
        ]

        return report

    def generate_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report."""
        dashboard = self.monitor.get_dashboard_data()

        # Add detailed breakdowns
        dashboard['symbol_health'] = self.monitor.get_all_symbol_health()
        dashboard['provider_health'] = self.monitor.get_all_provider_health()

        # Add all active alerts
        dashboard['active_alerts'] = [
            {
                'type': a.alert_type.value,
                'severity': a.severity.value,
                'message': a.message,
                'symbol': a.symbol,
                'provider': a.provider,
                'timestamp': a.timestamp.isoformat()
            }
            for a in self.monitor.get_active_alerts()
        ]

        return dashboard

    def to_dataframe(self) -> pd.DataFrame:
        """Convert symbol health to DataFrame."""
        records = []
        for symbol, health in self.monitor.get_all_symbol_health().items():
            records.append({
                'symbol': symbol,
                'health_score': health['health_score'],
                'is_stale': health['is_stale'],
                'update_count': health['update_count'],
                'error_count': health['error_count'],
                'gap_count': health['gap_count'],
                'last_update': health['last_update'],
                'last_price': health['last_price']
            })
        return pd.DataFrame(records)
