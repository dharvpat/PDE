"""
Health Checks and Synthetic Monitoring for Quantitative Trading System.

Provides comprehensive health monitoring including:
- Component health checks (databases, APIs, services)
- Synthetic monitoring (canary tests, probes)
- Liveness and readiness probes
- Dependency health tracking
- Circuit breaker integration
"""

import asyncio
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class HealthCheck(ABC):
    """Base class for health checks."""

    def __init__(
        self,
        name: str,
        timeout_seconds: float = 5.0,
        critical: bool = True,
    ):
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.critical = critical

    @abstractmethod
    def check(self) -> HealthCheckResult:
        """Execute the health check."""
        pass

    def _timed_check(self, check_func: Callable[[], Tuple[HealthStatus, str, Dict[str, Any]]]) -> HealthCheckResult:
        """Execute check with timing."""
        start = time.time()
        try:
            status, message, details = check_func()
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                latency_ms=latency_ms,
                details=details,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)},
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""

    def __init__(
        self,
        name: str,
        connection_factory: Callable[[], Any],
        query: str = "SELECT 1",
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds, critical=True)
        self.connection_factory = connection_factory
        self.query = query

    def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            conn = self.connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(self.query)
                result = cursor.fetchone()
                cursor.close()
                return (
                    HealthStatus.HEALTHY,
                    "Database connection successful",
                    {"query_result": str(result)},
                )
            finally:
                conn.close()

        return self._timed_check(_check)


class TCPHealthCheck(HealthCheck):
    """Health check for TCP connectivity."""

    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.host = host
        self.port = port

    def check(self) -> HealthCheckResult:
        """Check TCP connectivity."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout_seconds)
            try:
                sock.connect((self.host, self.port))
                return (
                    HealthStatus.HEALTHY,
                    f"TCP connection to {self.host}:{self.port} successful",
                    {"host": self.host, "port": self.port},
                )
            finally:
                sock.close()

        return self._timed_check(_check)


class HTTPHealthCheck(HealthCheck):
    """Health check for HTTP endpoints."""

    def __init__(
        self,
        name: str,
        url: str,
        expected_status: int = 200,
        timeout_seconds: float = 10.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, timeout_seconds)
        self.url = url
        self.expected_status = expected_status
        self.headers = headers or {}

    def check(self) -> HealthCheckResult:
        """Check HTTP endpoint."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            import urllib.request
            import urllib.error

            request = urllib.request.Request(self.url, headers=self.headers)
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
                status_code = response.getcode()
                if status_code == self.expected_status:
                    return (
                        HealthStatus.HEALTHY,
                        f"HTTP check successful: {status_code}",
                        {"url": self.url, "status_code": status_code},
                    )
                else:
                    return (
                        HealthStatus.DEGRADED,
                        f"Unexpected status code: {status_code}",
                        {"url": self.url, "status_code": status_code, "expected": self.expected_status},
                    )
            except urllib.error.HTTPError as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"HTTP error: {e.code}",
                    {"url": self.url, "status_code": e.code, "error": str(e)},
                )
            except urllib.error.URLError as e:
                return (
                    HealthStatus.UNHEALTHY,
                    f"URL error: {str(e)}",
                    {"url": self.url, "error": str(e)},
                )

        return self._timed_check(_check)


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity."""

    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.host = host
        self.port = port
        self.password = password

    def check(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            try:
                import redis
                client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    socket_timeout=self.timeout_seconds,
                )
                info = client.ping()
                if info:
                    return (
                        HealthStatus.HEALTHY,
                        "Redis connection successful",
                        {"host": self.host, "port": self.port},
                    )
                else:
                    return (
                        HealthStatus.UNHEALTHY,
                        "Redis ping failed",
                        {"host": self.host, "port": self.port},
                    )
            except ImportError:
                return (
                    HealthStatus.UNKNOWN,
                    "Redis client not installed",
                    {},
                )

        return self._timed_check(_check)


class RabbitMQHealthCheck(HealthCheck):
    """Health check for RabbitMQ connectivity."""

    def __init__(
        self,
        name: str,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        timeout_seconds: float = 5.0,
    ):
        super().__init__(name, timeout_seconds)
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def check(self) -> HealthCheckResult:
        """Check RabbitMQ connectivity."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            try:
                import pika
                credentials = pika.PlainCredentials(self.username, self.password)
                parameters = pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    socket_timeout=self.timeout_seconds,
                )
                connection = pika.BlockingConnection(parameters)
                connection.close()
                return (
                    HealthStatus.HEALTHY,
                    "RabbitMQ connection successful",
                    {"host": self.host, "port": self.port},
                )
            except ImportError:
                return (
                    HealthStatus.UNKNOWN,
                    "Pika client not installed",
                    {},
                )

        return self._timed_check(_check)


class MemoryHealthCheck(HealthCheck):
    """Health check for system memory."""

    def __init__(
        self,
        name: str = "memory",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        super().__init__(name, critical=True)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check(self) -> HealthCheckResult:
        """Check system memory usage."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            try:
                import psutil
                memory = psutil.virtual_memory()
                usage = memory.percent / 100.0

                details = {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_percent": memory.percent,
                }

                if usage >= self.critical_threshold:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"Memory usage critical: {memory.percent:.1f}%",
                        details,
                    )
                elif usage >= self.warning_threshold:
                    return (
                        HealthStatus.DEGRADED,
                        f"Memory usage high: {memory.percent:.1f}%",
                        details,
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        f"Memory usage normal: {memory.percent:.1f}%",
                        details,
                    )
            except ImportError:
                return (
                    HealthStatus.UNKNOWN,
                    "psutil not installed",
                    {},
                )

        return self._timed_check(_check)


class DiskHealthCheck(HealthCheck):
    """Health check for disk space."""

    def __init__(
        self,
        name: str = "disk",
        path: str = "/",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        super().__init__(name, critical=True)
        self.path = path
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check(self) -> HealthCheckResult:
        """Check disk space."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            try:
                import psutil
                disk = psutil.disk_usage(self.path)
                usage = disk.percent / 100.0

                details = {
                    "path": self.path,
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3),
                    "used_percent": disk.percent,
                }

                if usage >= self.critical_threshold:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"Disk usage critical: {disk.percent:.1f}%",
                        details,
                    )
                elif usage >= self.warning_threshold:
                    return (
                        HealthStatus.DEGRADED,
                        f"Disk usage high: {disk.percent:.1f}%",
                        details,
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        f"Disk usage normal: {disk.percent:.1f}%",
                        details,
                    )
            except ImportError:
                return (
                    HealthStatus.UNKNOWN,
                    "psutil not installed",
                    {},
                )

        return self._timed_check(_check)


class CPUHealthCheck(HealthCheck):
    """Health check for CPU usage."""

    def __init__(
        self,
        name: str = "cpu",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        super().__init__(name, critical=False)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check(self) -> HealthCheckResult:
        """Check CPU usage."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                usage = cpu_percent / 100.0

                details = {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_percent": cpu_percent,
                }

                if usage >= self.critical_threshold:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"CPU usage critical: {cpu_percent:.1f}%",
                        details,
                    )
                elif usage >= self.warning_threshold:
                    return (
                        HealthStatus.DEGRADED,
                        f"CPU usage high: {cpu_percent:.1f}%",
                        details,
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        f"CPU usage normal: {cpu_percent:.1f}%",
                        details,
                    )
            except ImportError:
                return (
                    HealthStatus.UNKNOWN,
                    "psutil not installed",
                    {},
                )

        return self._timed_check(_check)


class CustomHealthCheck(HealthCheck):
    """Custom health check using a callable."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Tuple[bool, str, Dict[str, Any]]],
        timeout_seconds: float = 5.0,
        critical: bool = False,
    ):
        super().__init__(name, timeout_seconds, critical)
        self.check_func = check_func

    def check(self) -> HealthCheckResult:
        """Execute custom check."""
        def _check() -> Tuple[HealthStatus, str, Dict[str, Any]]:
            success, message, details = self.check_func()
            status = HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY
            return status, message, details

        return self._timed_check(_check)


@dataclass
class SyntheticTestResult:
    """Result of a synthetic test."""

    name: str
    success: bool
    latency_ms: float
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class SyntheticTest(ABC):
    """Base class for synthetic tests."""

    def __init__(self, name: str, timeout_seconds: float = 30.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def run(self) -> SyntheticTestResult:
        """Execute the synthetic test."""
        pass


class OrderFlowSyntheticTest(SyntheticTest):
    """Synthetic test for order flow."""

    def __init__(
        self,
        name: str = "order_flow",
        create_order_func: Optional[Callable[[], Any]] = None,
        cancel_order_func: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(name)
        self.create_order_func = create_order_func
        self.cancel_order_func = cancel_order_func

    def run(self) -> SyntheticTestResult:
        """Run order flow test."""
        start = time.time()
        try:
            if not self.create_order_func or not self.cancel_order_func:
                return SyntheticTestResult(
                    name=self.name,
                    success=True,
                    latency_ms=0,
                    message="Order flow functions not configured - skipping",
                )

            # Create test order
            order = self.create_order_func()
            order_id = getattr(order, 'order_id', str(order))

            # Cancel test order
            cancelled = self.cancel_order_func(order_id)

            latency_ms = (time.time() - start) * 1000

            if cancelled:
                return SyntheticTestResult(
                    name=self.name,
                    success=True,
                    latency_ms=latency_ms,
                    message="Order flow test successful",
                    details={"order_id": order_id},
                )
            else:
                return SyntheticTestResult(
                    name=self.name,
                    success=False,
                    latency_ms=latency_ms,
                    message="Failed to cancel test order",
                    details={"order_id": order_id},
                )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return SyntheticTestResult(
                name=self.name,
                success=False,
                latency_ms=latency_ms,
                message=f"Order flow test failed: {str(e)}",
                details={"error": str(e)},
            )


class DataFeedSyntheticTest(SyntheticTest):
    """Synthetic test for data feed."""

    def __init__(
        self,
        name: str = "data_feed",
        get_quote_func: Optional[Callable[[str], Any]] = None,
        test_symbols: Optional[List[str]] = None,
    ):
        super().__init__(name)
        self.get_quote_func = get_quote_func
        self.test_symbols = test_symbols or ["SPY", "QQQ"]

    def run(self) -> SyntheticTestResult:
        """Run data feed test."""
        start = time.time()
        try:
            if not self.get_quote_func:
                return SyntheticTestResult(
                    name=self.name,
                    success=True,
                    latency_ms=0,
                    message="Data feed function not configured - skipping",
                )

            results = {}
            for symbol in self.test_symbols:
                try:
                    quote = self.get_quote_func(symbol)
                    results[symbol] = {
                        "success": True,
                        "has_price": hasattr(quote, 'price') or isinstance(quote, (int, float)),
                    }
                except Exception as e:
                    results[symbol] = {
                        "success": False,
                        "error": str(e),
                    }

            latency_ms = (time.time() - start) * 1000
            all_success = all(r["success"] for r in results.values())

            return SyntheticTestResult(
                name=self.name,
                success=all_success,
                latency_ms=latency_ms,
                message="Data feed test successful" if all_success else "Some symbols failed",
                details={"results": results},
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return SyntheticTestResult(
                name=self.name,
                success=False,
                latency_ms=latency_ms,
                message=f"Data feed test failed: {str(e)}",
                details={"error": str(e)},
            )


class ModelCalibrationSyntheticTest(SyntheticTest):
    """Synthetic test for model calibration."""

    def __init__(
        self,
        name: str = "model_calibration",
        calibrate_func: Optional[Callable[[], Any]] = None,
        max_latency_seconds: float = 60.0,
    ):
        super().__init__(name, timeout_seconds=max_latency_seconds)
        self.calibrate_func = calibrate_func
        self.max_latency_seconds = max_latency_seconds

    def run(self) -> SyntheticTestResult:
        """Run model calibration test."""
        start = time.time()
        try:
            if not self.calibrate_func:
                return SyntheticTestResult(
                    name=self.name,
                    success=True,
                    latency_ms=0,
                    message="Calibration function not configured - skipping",
                )

            result = self.calibrate_func()
            latency_ms = (time.time() - start) * 1000
            latency_seconds = latency_ms / 1000

            details = {
                "latency_seconds": latency_seconds,
                "max_latency_seconds": self.max_latency_seconds,
            }

            if hasattr(result, 'rmse'):
                details["rmse"] = result.rmse
            if hasattr(result, 'r_squared'):
                details["r_squared"] = result.r_squared

            if latency_seconds > self.max_latency_seconds:
                return SyntheticTestResult(
                    name=self.name,
                    success=False,
                    latency_ms=latency_ms,
                    message=f"Calibration too slow: {latency_seconds:.1f}s > {self.max_latency_seconds}s",
                    details=details,
                )

            return SyntheticTestResult(
                name=self.name,
                success=True,
                latency_ms=latency_ms,
                message="Model calibration successful",
                details=details,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return SyntheticTestResult(
                name=self.name,
                success=False,
                latency_ms=latency_ms,
                message=f"Model calibration failed: {str(e)}",
                details={"error": str(e)},
            )


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    synthetic_tests: List[SyntheticTestResult]
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "checks": [c.to_dict() for c in self.checks],
            "synthetic_tests": [t.to_dict() for t in self.synthetic_tests],
            "summary": {
                "total_checks": len(self.checks),
                "healthy_checks": len([c for c in self.checks if c.status == HealthStatus.HEALTHY]),
                "total_tests": len(self.synthetic_tests),
                "passed_tests": len([t for t in self.synthetic_tests if t.success]),
            },
        }


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_calls += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = datetime.now()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def is_available(self) -> bool:
        """Check if calls should be allowed."""
        state = self.state  # Triggers recovery check
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            return self._half_open_calls < self.half_open_max_calls

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time.isoformat() if self._last_failure_time else None,
        }


class HealthManager:
    """Central manager for health checks and synthetic monitoring."""

    def __init__(
        self,
        check_interval_seconds: float = 60.0,
        synthetic_test_interval_seconds: float = 300.0,
    ):
        self.check_interval = check_interval_seconds
        self.synthetic_test_interval = synthetic_test_interval_seconds
        self._health_checks: Dict[str, HealthCheck] = {}
        self._synthetic_tests: Dict[str, SyntheticTest] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._last_results: Dict[str, HealthCheckResult] = {}
        self._last_test_results: Dict[str, SyntheticTestResult] = {}
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._test_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check."""
        self._health_checks[check.name] = check

    def register_synthetic_test(self, test: SyntheticTest) -> None:
        """Register a synthetic test."""
        self._synthetic_tests[test.name] = test

    def register_circuit_breaker(self, breaker: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        self._circuit_breakers[breaker.name] = breaker

    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self._circuit_breakers.get(name)

    def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        results = []
        for name, check in self._health_checks.items():
            try:
                result = check.check()
                with self._lock:
                    self._last_results[name] = result
                results.append(result)
                logger.debug(f"Health check '{name}': {result.status.value}")
            except Exception as e:
                result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check error: {str(e)}",
                )
                with self._lock:
                    self._last_results[name] = result
                results.append(result)
                logger.error(f"Health check '{name}' error: {e}")
        return results

    def run_synthetic_tests(self) -> List[SyntheticTestResult]:
        """Run all synthetic tests."""
        results = []
        for name, test in self._synthetic_tests.items():
            try:
                result = test.run()
                with self._lock:
                    self._last_test_results[name] = result
                results.append(result)
                logger.debug(f"Synthetic test '{name}': {'passed' if result.success else 'failed'}")
            except Exception as e:
                result = SyntheticTestResult(
                    name=name,
                    success=False,
                    latency_ms=0,
                    message=f"Test error: {str(e)}",
                )
                with self._lock:
                    self._last_test_results[name] = result
                results.append(result)
                logger.error(f"Synthetic test '{name}' error: {e}")
        return results

    def get_health_report(self) -> HealthReport:
        """Get complete health report."""
        with self._lock:
            checks = list(self._last_results.values())
            tests = list(self._last_test_results.values())

        # Determine overall status
        if not checks:
            overall_status = HealthStatus.UNKNOWN
        elif any(c.status == HealthStatus.UNHEALTHY and self._health_checks.get(c.name, HealthCheck("", critical=True)).critical for c in checks):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in checks):
            overall_status = HealthStatus.DEGRADED
        elif all(c.status == HealthStatus.HEALTHY for c in checks):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN

        return HealthReport(
            status=overall_status,
            timestamp=datetime.now(),
            checks=checks,
            synthetic_tests=tests,
        )

    def is_healthy(self) -> bool:
        """Quick check if system is healthy."""
        report = self.get_health_report()
        return report.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def is_ready(self) -> bool:
        """Check if system is ready to serve requests."""
        # Run critical checks only
        for name, check in self._health_checks.items():
            if check.critical:
                result = check.check()
                if result.status == HealthStatus.UNHEALTHY:
                    return False
        return True

    def start_background_monitoring(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True

        def health_check_worker():
            while self._running:
                try:
                    self.run_health_checks()
                except Exception as e:
                    logger.error(f"Health check worker error: {e}")
                time.sleep(self.check_interval)

        def synthetic_test_worker():
            while self._running:
                try:
                    self.run_synthetic_tests()
                except Exception as e:
                    logger.error(f"Synthetic test worker error: {e}")
                time.sleep(self.synthetic_test_interval)

        self._check_thread = threading.Thread(target=health_check_worker, daemon=True)
        self._test_thread = threading.Thread(target=synthetic_test_worker, daemon=True)

        self._check_thread.start()
        self._test_thread.start()

        logger.info("Background health monitoring started")

    def stop_background_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        if self._test_thread:
            self._test_thread.join(timeout=5.0)
        logger.info("Background health monitoring stopped")


# Default health manager instance
_health_manager: Optional[HealthManager] = None


def get_health_manager() -> HealthManager:
    """Get or create the default health manager."""
    global _health_manager
    if _health_manager is None:
        _health_manager = HealthManager()
    return _health_manager


def register_default_checks() -> None:
    """Register default system health checks."""
    manager = get_health_manager()
    manager.register_check(MemoryHealthCheck())
    manager.register_check(DiskHealthCheck())
    manager.register_check(CPUHealthCheck())


# Convenience decorators
def with_circuit_breaker(breaker_name: str):
    """Decorator to wrap function with circuit breaker."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            manager = get_health_manager()
            breaker = manager.get_circuit_breaker(breaker_name)
            if breaker is None:
                return func(*args, **kwargs)

            if not breaker.is_available():
                raise RuntimeError(f"Circuit breaker '{breaker_name}' is open")

            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise

        return wrapper
    return decorator


def health_check_endpoint() -> Dict[str, Any]:
    """Return health check data for HTTP endpoint."""
    manager = get_health_manager()
    report = manager.get_health_report()
    return report.to_dict()


def liveness_probe() -> bool:
    """Liveness probe - is the process running."""
    return True


def readiness_probe() -> bool:
    """Readiness probe - is the service ready to accept traffic."""
    return get_health_manager().is_ready()
