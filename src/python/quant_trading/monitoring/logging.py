"""
Structured Logging Framework for Quantitative Trading System.

Provides centralized, structured logging with support for:
- JSON-formatted log output
- Contextual fields (strategy, symbol, order_id, etc.)
- Log aggregation compatibility (ELK, Splunk)
- Performance metrics extraction
- Error tracking and grouping
- Async log handlers
"""

import json
import logging
import sys
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Set, Union
import hashlib


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogCategory(Enum):
    """Categories for log classification."""
    TRADING = "trading"
    RISK = "risk"
    DATA = "data"
    MODEL = "model"
    EXECUTION = "execution"
    SYSTEM = "system"
    AUDIT = "audit"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"


@dataclass
class LogContext:
    """Thread-local context for structured logging."""

    fields: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Set a context field."""
        self.fields[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context field."""
        return self.fields.get(key, default)

    def remove(self, key: str) -> None:
        """Remove a context field."""
        self.fields.pop(key, None)

    def clear(self) -> None:
        """Clear all context fields."""
        self.fields.clear()

    def copy(self) -> Dict[str, Any]:
        """Return a copy of context fields."""
        return self.fields.copy()


class ContextVar:
    """Thread-local context variable storage."""

    def __init__(self):
        self._local = threading.local()

    def get(self) -> LogContext:
        """Get the current context."""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext()
        return self._local.context

    def set(self, context: LogContext) -> None:
        """Set the current context."""
        self._local.context = context


# Global context variable
_context_var = ContextVar()


def get_context() -> LogContext:
    """Get the current logging context."""
    return _context_var.get()


def bind(**kwargs) -> None:
    """Bind fields to the current logging context."""
    context = get_context()
    for key, value in kwargs.items():
        context.set(key, value)


def unbind(*keys: str) -> None:
    """Remove fields from the current logging context."""
    context = get_context()
    for key in keys:
        context.remove(key)


def clear_context() -> None:
    """Clear all fields from the current logging context."""
    get_context().clear()


class BoundLogger:
    """Context manager for temporarily binding log context."""

    def __init__(self, **kwargs):
        self.bindings = kwargs
        self.previous_values: Dict[str, Any] = {}

    def __enter__(self) -> 'BoundLogger':
        context = get_context()
        for key, value in self.bindings.items():
            if key in context.fields:
                self.previous_values[key] = context.fields[key]
            context.set(key, value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        context = get_context()
        for key in self.bindings:
            if key in self.previous_values:
                context.set(key, self.previous_values[key])
            else:
                context.remove(key)


@dataclass
class StructuredLogRecord:
    """Structured log record with all fields."""

    timestamp: datetime
    level: str
    message: str
    logger_name: str
    category: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Tracing fields
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Source location
    filename: Optional[str] = None
    lineno: Optional[int] = None
    func_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "@timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "message": self.message,
            "logger": self.logger_name,
        }

        if self.category:
            result["category"] = self.category

        if self.context:
            result["context"] = self.context

        if self.exception:
            result["exception"] = self.exception

        if self.extra:
            result.update(self.extra)

        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.parent_span_id:
            result["parent_span_id"] = self.parent_span_id

        if self.filename:
            result["source"] = {
                "file": self.filename,
                "line": self.lineno,
                "function": self.func_name,
            }

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        include_context: bool = True,
        include_source: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.include_context = include_context
        self.include_source = include_source
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build structured record
        structured = StructuredLogRecord(
            timestamp=datetime.fromtimestamp(record.created),
            level=record.levelname,
            message=record.getMessage(),
            logger_name=record.name,
        )

        # Add category if present
        if hasattr(record, 'category'):
            structured.category = record.category

        # Add context
        if self.include_context:
            structured.context = get_context().copy()

        # Add extra fields from record
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'category',
        }

        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith('_'):
                structured.extra[key] = value

        # Add static extra fields
        structured.extra.update(self.extra_fields)

        # Add exception info
        if record.exc_info:
            structured.exception = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add source location
        if self.include_source:
            structured.filename = record.filename
            structured.lineno = record.lineno
            structured.func_name = record.funcName

        # Add trace context if present
        if hasattr(record, 'trace_id'):
            structured.trace_id = record.trace_id
        if hasattr(record, 'span_id'):
            structured.span_id = record.span_id
        if hasattr(record, 'parent_span_id'):
            structured.parent_span_id = record.parent_span_id

        return structured.to_json()


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def __init__(
        self,
        use_colors: bool = True,
        include_context: bool = True,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        super().__init__()
        self.use_colors = use_colors
        self.include_context = include_context
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        timestamp = datetime.fromtimestamp(record.created).strftime(self.timestamp_format)
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, '')
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()

        # Build output
        parts = [f"{timestamp} {level_str} [{record.name}] {message}"]

        # Add context
        if self.include_context:
            context = get_context().copy()
            if context:
                context_str = " ".join(f"{k}={v}" for k, v in context.items())
                parts.append(f"  | {context_str}")

        # Add exception
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))

        return "\n".join(parts)


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler with buffering."""

    def __init__(
        self,
        handler: logging.Handler,
        buffer_size: int = 1000,
        flush_interval: float = 1.0,
    ):
        super().__init__()
        self.handler = handler
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.queue: Queue = Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        """Add record to queue."""
        try:
            self.queue.put_nowait(record)
        except Exception:
            # Queue full, drop record
            pass

    def _worker(self) -> None:
        """Background worker to process log records."""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=self.flush_interval)
                self.handler.emit(record)
            except Empty:
                continue
            except Exception:
                pass

    def close(self) -> None:
        """Stop the async handler."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

        # Flush remaining records
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                self.handler.emit(record)
            except Empty:
                break

        self.handler.close()
        super().close()


class RotatingFileHandler(logging.Handler):
    """File handler with rotation based on size."""

    def __init__(
        self,
        filepath: Union[str, Path],
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        encoding: str = "utf-8",
    ):
        super().__init__()
        self.filepath = Path(filepath)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self._lock = threading.Lock()
        self._stream: Optional[Any] = None
        self._open_stream()

    def _open_stream(self) -> None:
        """Open the log file stream."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.filepath, 'a', encoding=self.encoding)

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        if self._stream is None:
            return False
        try:
            return self.filepath.stat().st_size >= self.max_bytes
        except FileNotFoundError:
            return False

    def _rotate(self) -> None:
        """Perform log rotation."""
        if self._stream:
            self._stream.close()

        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.filepath}.{i}")
            dst = Path(f"{self.filepath}.{i + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

        # Rename current to .1
        if self.filepath.exists():
            backup = Path(f"{self.filepath}.1")
            if backup.exists():
                backup.unlink()
            self.filepath.rename(backup)

        self._open_stream()

    def emit(self, record: logging.LogRecord) -> None:
        """Write log record to file."""
        try:
            with self._lock:
                if self._should_rotate():
                    self._rotate()

                if self._stream:
                    msg = self.format(record)
                    self._stream.write(msg + "\n")
                    self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Close the file handler."""
        with self._lock:
            if self._stream:
                self._stream.close()
                self._stream = None
        super().close()


@dataclass
class ErrorGroup:
    """Grouped error information for tracking."""

    error_hash: str
    error_type: str
    message_pattern: str
    count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    sample_traceback: Optional[str] = None
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    max_occurrences: int = 10


class ErrorTracker:
    """Track and group errors for analysis."""

    def __init__(self, max_groups: int = 1000):
        self.max_groups = max_groups
        self.groups: Dict[str, ErrorGroup] = {}
        self._lock = threading.Lock()

    def _compute_hash(self, error_type: str, message: str, traceback_str: str) -> str:
        """Compute a hash for error grouping."""
        # Normalize message by removing numbers and specific values
        import re
        normalized = re.sub(r'\d+', 'N', message)
        normalized = re.sub(r'0x[0-9a-fA-F]+', 'ADDR', normalized)

        # Extract key frames from traceback
        frames = []
        for line in traceback_str.split('\n'):
            if 'File "' in line:
                frames.append(line.strip())

        key = f"{error_type}:{normalized}:{':'.join(frames[:5])}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def track(
        self,
        error_type: str,
        message: str,
        traceback_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorGroup:
        """Track an error occurrence."""
        error_hash = self._compute_hash(error_type, message, traceback_str)
        now = datetime.now()

        with self._lock:
            if error_hash not in self.groups:
                if len(self.groups) >= self.max_groups:
                    # Remove oldest group
                    oldest = min(self.groups.values(), key=lambda g: g.last_seen or now)
                    del self.groups[oldest.error_hash]

                self.groups[error_hash] = ErrorGroup(
                    error_hash=error_hash,
                    error_type=error_type,
                    message_pattern=message[:200],
                    first_seen=now,
                    sample_traceback=traceback_str,
                )

            group = self.groups[error_hash]
            group.count += 1
            group.last_seen = now

            if len(group.occurrences) < group.max_occurrences:
                group.occurrences.append({
                    "timestamp": now.isoformat(),
                    "message": message,
                    "context": context or {},
                })

            return group

    def get_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all error groups."""
        with self._lock:
            return [
                {
                    "hash": g.error_hash,
                    "type": g.error_type,
                    "pattern": g.message_pattern,
                    "count": g.count,
                    "first_seen": g.first_seen.isoformat() if g.first_seen else None,
                    "last_seen": g.last_seen.isoformat() if g.last_seen else None,
                }
                for g in sorted(
                    self.groups.values(),
                    key=lambda g: g.count,
                    reverse=True,
                )
            ]

    def get_group(self, error_hash: str) -> Optional[ErrorGroup]:
        """Get a specific error group."""
        return self.groups.get(error_hash)


class StructuredLogger:
    """Main structured logger interface."""

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        category: Optional[LogCategory] = None,
    ):
        self.name = name
        self.category = category
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        self._error_tracker: Optional[ErrorTracker] = None

    def set_error_tracker(self, tracker: ErrorTracker) -> None:
        """Set error tracker for this logger."""
        self._error_tracker = tracker

    def _log(
        self,
        level: LogLevel,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        """Internal logging method."""
        extra = dict(kwargs)
        if self.category:
            extra['category'] = self.category.value

        self._logger.log(
            level.value,
            message,
            exc_info=exc_info,
            extra=extra,
        )

        # Track errors
        if level in (LogLevel.ERROR, LogLevel.CRITICAL) and self._error_tracker:
            if exc_info:
                tb = ''.join(traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__))
                self._error_tracker.track(
                    error_type=type(exc_info).__name__,
                    message=str(exc_info),
                    traceback_str=tb,
                    context=get_context().copy(),
                )

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        exc_info = sys.exc_info()[1]
        self._log(LogLevel.ERROR, message, exc_info=exc_info, **kwargs)

    def bind(self, **kwargs) -> BoundLogger:
        """Return a context manager that binds fields."""
        return BoundLogger(**kwargs)


class LoggingConfig:
    """Configuration for the logging system."""

    def __init__(
        self,
        level: LogLevel = LogLevel.INFO,
        json_output: bool = True,
        console_output: bool = True,
        file_output: Optional[str] = None,
        include_context: bool = True,
        include_source: bool = True,
        use_async: bool = False,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        self.level = level
        self.json_output = json_output
        self.console_output = console_output
        self.file_output = file_output
        self.include_context = include_context
        self.include_source = include_source
        self.use_async = use_async
        self.extra_fields = extra_fields or {}


class LoggingManager:
    """Central manager for logging configuration."""

    _instance: Optional['LoggingManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'LoggingManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._loggers: Dict[str, StructuredLogger] = {}
        self._handlers: List[logging.Handler] = []
        self._error_tracker = ErrorTracker()
        self._config: Optional[LoggingConfig] = None

    def configure(self, config: LoggingConfig) -> None:
        """Configure the logging system."""
        self._config = config

        # Clear existing handlers
        root = logging.getLogger()
        for handler in self._handlers:
            root.removeHandler(handler)
            handler.close()
        self._handlers.clear()

        # Set root level
        root.setLevel(config.level.value)

        # Create formatters
        json_formatter = JsonFormatter(
            include_context=config.include_context,
            include_source=config.include_source,
            extra_fields=config.extra_fields,
        )
        console_formatter = ConsoleFormatter(
            include_context=config.include_context,
        )

        # Console handler
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            if config.json_output:
                console_handler.setFormatter(json_formatter)
            else:
                console_handler.setFormatter(console_formatter)

            if config.use_async:
                console_handler = AsyncLogHandler(console_handler)

            root.addHandler(console_handler)
            self._handlers.append(console_handler)

        # File handler
        if config.file_output:
            file_handler = RotatingFileHandler(config.file_output)
            file_handler.setFormatter(json_formatter)

            if config.use_async:
                file_handler = AsyncLogHandler(file_handler)

            root.addHandler(file_handler)
            self._handlers.append(file_handler)

    def get_logger(
        self,
        name: str,
        category: Optional[LogCategory] = None,
    ) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in self._loggers:
            logger = StructuredLogger(
                name=name,
                level=self._config.level if self._config else LogLevel.INFO,
                category=category,
            )
            logger.set_error_tracker(self._error_tracker)
            self._loggers[name] = logger

        return self._loggers[name]

    def get_error_summary(self) -> List[Dict[str, Any]]:
        """Get error tracking summary."""
        return self._error_tracker.get_summary()

    def shutdown(self) -> None:
        """Shutdown the logging system."""
        for handler in self._handlers:
            handler.close()
        self._handlers.clear()


# Convenience functions
def configure_logging(
    level: LogLevel = LogLevel.INFO,
    json_output: bool = True,
    console_output: bool = True,
    file_output: Optional[str] = None,
    **kwargs,
) -> None:
    """Configure the logging system with common settings."""
    config = LoggingConfig(
        level=level,
        json_output=json_output,
        console_output=console_output,
        file_output=file_output,
        **kwargs,
    )
    LoggingManager().configure(config)


def get_logger(name: str, category: Optional[LogCategory] = None) -> StructuredLogger:
    """Get a structured logger."""
    return LoggingManager().get_logger(name, category)


# Trading-specific logging utilities
class TradingLogger(StructuredLogger):
    """Logger with trading-specific convenience methods."""

    def __init__(self, name: str = "trading"):
        super().__init__(name, category=LogCategory.TRADING)

    def log_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "market",
        **kwargs,
    ) -> None:
        """Log order creation."""
        with self.bind(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
        ):
            self.info(f"Order created: {side} {quantity} {symbol}", **kwargs)

    def log_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        **kwargs,
    ) -> None:
        """Log order fill."""
        with self.bind(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
        ):
            self.info(f"Order filled: {side} {quantity} {symbol} @ {fill_price}", **kwargs)

    def log_signal(
        self,
        strategy: str,
        symbol: str,
        signal_type: str,
        strength: float,
        **kwargs,
    ) -> None:
        """Log trading signal."""
        with self.bind(
            strategy=strategy,
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
        ):
            self.info(f"Signal generated: {signal_type} {symbol} strength={strength:.3f}", **kwargs)

    def log_position_update(
        self,
        symbol: str,
        quantity: float,
        avg_price: float,
        unrealized_pnl: float,
        **kwargs,
    ) -> None:
        """Log position update."""
        with self.bind(
            symbol=symbol,
            quantity=quantity,
            avg_price=avg_price,
            unrealized_pnl=unrealized_pnl,
        ):
            self.info(f"Position update: {symbol} qty={quantity} pnl={unrealized_pnl:.2f}", **kwargs)


class RiskLogger(StructuredLogger):
    """Logger with risk-specific convenience methods."""

    def __init__(self, name: str = "risk"):
        super().__init__(name, category=LogCategory.RISK)

    def log_risk_metrics(
        self,
        strategy: str,
        var_95: float,
        var_99: float,
        expected_shortfall: float,
        **kwargs,
    ) -> None:
        """Log risk metrics."""
        with self.bind(
            strategy=strategy,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
        ):
            self.info(f"Risk metrics: VaR95={var_95:.2f} VaR99={var_99:.2f} ES={expected_shortfall:.2f}", **kwargs)

    def log_limit_breach(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        **kwargs,
    ) -> None:
        """Log limit breach."""
        with self.bind(
            limit_type=limit_type,
            current_value=current_value,
            limit_value=limit_value,
        ):
            self.warning(f"Limit breach: {limit_type} current={current_value:.2f} limit={limit_value:.2f}", **kwargs)

    def log_drawdown(
        self,
        strategy: str,
        current_drawdown: float,
        max_drawdown: float,
        **kwargs,
    ) -> None:
        """Log drawdown update."""
        with self.bind(
            strategy=strategy,
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
        ):
            level = LogLevel.WARNING if current_drawdown > 0.15 else LogLevel.INFO
            self._log(level, f"Drawdown: {strategy} current={current_drawdown:.2%} max={max_drawdown:.2%}", **kwargs)


class AuditLogger(StructuredLogger):
    """Logger for audit trail."""

    def __init__(self, name: str = "audit"):
        super().__init__(name, category=LogCategory.AUDIT)

    def log_action(
        self,
        user: str,
        action: str,
        resource: str,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Log user action."""
        with self.bind(
            user=user,
            action=action,
            resource=resource,
            details=details or {},
        ):
            self.info(f"Audit: {user} performed {action} on {resource}", **kwargs)

    def log_config_change(
        self,
        user: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        **kwargs,
    ) -> None:
        """Log configuration change."""
        with self.bind(
            user=user,
            config_key=config_key,
            old_value=old_value,
            new_value=new_value,
        ):
            self.info(f"Config change: {config_key} changed from {old_value} to {new_value} by {user}", **kwargs)


# Default loggers
trading_logger = TradingLogger()
risk_logger = RiskLogger()
audit_logger = AuditLogger()
