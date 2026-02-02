"""
Data Backfilling and Recovery Module.

This module provides data recovery and backfilling capabilities:
- Gap detection in time-series data
- Automated backfilling from multiple sources
- Data reconciliation and validation
- Recovery from data corruption
- Historical data reconstruction

Essential for maintaining complete, accurate time-series datasets.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GapType(Enum):
    """Types of data gaps."""
    MISSING_DAY = "missing_day"
    MISSING_BARS = "missing_bars"
    PARTIAL_DAY = "partial_day"
    STALE_DATA = "stale_data"
    CORRUPT_DATA = "corrupt_data"


class RecoveryStatus(Enum):
    """Status of recovery operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class BackfillPriority(Enum):
    """Priority levels for backfill requests."""
    CRITICAL = 1  # Real-time gaps, immediate backfill
    HIGH = 2      # Recent data gaps (< 7 days)
    NORMAL = 3    # Historical gaps
    LOW = 4       # Nice-to-have data


@dataclass
class DataGap:
    """Represents a gap in time-series data."""
    symbol: str
    gap_type: GapType
    start_time: datetime
    end_time: datetime
    expected_bars: int = 0
    actual_bars: int = 0
    detected_at: datetime = field(default_factory=datetime.now)
    priority: BackfillPriority = BackfillPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gap_duration(self) -> timedelta:
        """Get duration of the gap."""
        return self.end_time - self.start_time

    @property
    def missing_bars(self) -> int:
        """Get number of missing bars."""
        return max(0, self.expected_bars - self.actual_bars)


@dataclass
class BackfillRequest:
    """Request to backfill data."""
    request_id: str
    symbol: str
    start_date: date
    end_date: date
    frequency: str  # '1min', '1h', '1d'
    priority: BackfillPriority
    source: Optional[str] = None  # Preferred data source
    status: RecoveryStatus = RecoveryStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    bars_requested: int = 0
    bars_received: int = 0
    error_message: Optional[str] = None

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.bars_requested == 0:
            return 0.0
        return self.bars_received / self.bars_requested * 100


@dataclass
class RecoveryResult:
    """Result of a recovery operation."""
    request: BackfillRequest
    success: bool
    bars_recovered: int = 0
    bars_validated: int = 0
    validation_errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class GapDetector:
    """
    Detects gaps in time-series data.

    Analyzes data for missing periods, partial days, and anomalies.
    """

    def __init__(
        self,
        trading_calendar: Optional[Any] = None,
        expected_bars_per_day: int = 390  # 6.5 hours * 60 minutes
    ):
        """
        Initialize gap detector.

        Args:
            trading_calendar: Trading calendar for holiday awareness
            expected_bars_per_day: Expected bars per trading day
        """
        self._calendar = trading_calendar
        self._expected_bars = expected_bars_per_day

    def detect_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        frequency: str = '1min'
    ) -> List[DataGap]:
        """
        Detect gaps in time-series data.

        Args:
            df: DataFrame with timestamp index
            symbol: Symbol being analyzed
            frequency: Expected data frequency

        Returns:
            List of detected gaps
        """
        if df.empty:
            return []

        gaps = []

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Detect missing days
        gaps.extend(self._detect_missing_days(df, symbol))

        # Detect intraday gaps
        if frequency in ['1min', '5min', '15min', '30min', '1h']:
            gaps.extend(self._detect_intraday_gaps(df, symbol, frequency))

        # Detect partial days
        gaps.extend(self._detect_partial_days(df, symbol))

        return gaps

    def _detect_missing_days(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[DataGap]:
        """Detect completely missing trading days."""
        gaps = []

        if len(df) < 2:
            return gaps

        # Get unique dates
        dates = pd.Series(df.index.date).unique()

        # Check for gaps between consecutive dates
        for i in range(len(dates) - 1):
            current = dates[i]
            next_date = dates[i + 1]

            # Check each day in between
            check_date = current + timedelta(days=1)
            while check_date < next_date:
                # Skip weekends
                if check_date.weekday() < 5:
                    # Check if it's a trading day (if calendar available)
                    is_trading_day = True
                    if self._calendar:
                        is_trading_day = self._calendar.is_trading_day(check_date)

                    if is_trading_day:
                        gaps.append(DataGap(
                            symbol=symbol,
                            gap_type=GapType.MISSING_DAY,
                            start_time=datetime.combine(check_date, datetime.min.time()),
                            end_time=datetime.combine(check_date, datetime.max.time()),
                            expected_bars=self._expected_bars,
                            actual_bars=0,
                            priority=BackfillPriority.HIGH if (
                                date.today() - check_date
                            ).days < 7 else BackfillPriority.NORMAL
                        ))

                check_date += timedelta(days=1)

        return gaps

    def _detect_intraday_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        frequency: str
    ) -> List[DataGap]:
        """Detect gaps within trading days."""
        gaps = []

        # Parse frequency to timedelta
        freq_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1)
        }

        expected_interval = freq_map.get(frequency, timedelta(minutes=1))
        max_gap = expected_interval * 3  # Allow up to 3x expected interval

        # Check gaps between consecutive timestamps
        for i in range(len(df) - 1):
            current_time = df.index[i]
            next_time = df.index[i + 1]

            # Skip overnight gaps
            if current_time.date() != next_time.date():
                continue

            gap_duration = next_time - current_time

            if gap_duration > max_gap:
                expected = int(gap_duration / expected_interval)
                gaps.append(DataGap(
                    symbol=symbol,
                    gap_type=GapType.MISSING_BARS,
                    start_time=current_time,
                    end_time=next_time,
                    expected_bars=expected,
                    actual_bars=1,
                    priority=BackfillPriority.HIGH
                ))

        return gaps

    def _detect_partial_days(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[DataGap]:
        """Detect days with significantly fewer bars than expected."""
        gaps = []

        # Count bars per day
        bars_per_day = df.groupby(df.index.date).size()
        threshold = self._expected_bars * 0.8  # 80% of expected

        for day, count in bars_per_day.items():
            if count < threshold:
                gaps.append(DataGap(
                    symbol=symbol,
                    gap_type=GapType.PARTIAL_DAY,
                    start_time=datetime.combine(day, datetime.min.time()),
                    end_time=datetime.combine(day, datetime.max.time()),
                    expected_bars=self._expected_bars,
                    actual_bars=count,
                    priority=BackfillPriority.NORMAL
                ))

        return gaps


class DataValidator:
    """
    Validates recovered/backfilled data.

    Ensures data quality and consistency before insertion.
    """

    def __init__(self):
        """Initialize validator."""
        self._rules: List[Callable[[pd.DataFrame], Tuple[bool, str]]] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default validation rules."""
        self._rules = [
            self._validate_no_nulls,
            self._validate_ohlc_consistency,
            self._validate_price_bounds,
            self._validate_volume_positive,
            self._validate_timestamps_ordered
        ]

    def _validate_no_nulls(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for null values."""
        null_counts = df.isnull().sum()
        if null_counts.any():
            cols_with_nulls = null_counts[null_counts > 0].index.tolist()
            return False, f"Null values in columns: {cols_with_nulls}"
        return True, ""

    def _validate_ohlc_consistency(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate OHLC relationships."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return True, ""  # Skip if not OHLC data

        # High >= all others, Low <= all others
        invalid = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )

        if invalid.any():
            count = invalid.sum()
            return False, f"OHLC consistency violations: {count} bars"

        return True, ""

    def _validate_price_bounds(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Check for unrealistic prices."""
        price_cols = ['open', 'high', 'low', 'close', 'price']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    return False, f"Non-positive prices in {col}"
                if (df[col] > 1e6).any():
                    return False, f"Unrealistically high prices in {col}"
        return True, ""

    def _validate_volume_positive(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Ensure volume is non-negative."""
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                return False, "Negative volume values"
        return True, ""

    def _validate_timestamps_ordered(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Ensure timestamps are ordered."""
        if not df.index.is_monotonic_increasing:
            return False, "Timestamps not in ascending order"
        return True, ""

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for rule in self._rules:
            try:
                is_valid, error = rule(df)
                if not is_valid:
                    errors.append(error)
            except Exception as e:
                errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors


class BackfillManager:
    """
    Manages data backfilling operations.

    Coordinates gap detection, data retrieval, validation, and storage.
    """

    def __init__(
        self,
        gap_detector: Optional[GapDetector] = None,
        validator: Optional[DataValidator] = None,
        max_concurrent_requests: int = 5
    ):
        """
        Initialize backfill manager.

        Args:
            gap_detector: Gap detector instance
            validator: Data validator instance
            max_concurrent_requests: Maximum concurrent backfill requests
        """
        self._gap_detector = gap_detector or GapDetector()
        self._validator = validator or DataValidator()
        self._max_concurrent = max_concurrent_requests

        self._pending_requests: List[BackfillRequest] = []
        self._active_requests: Dict[str, BackfillRequest] = {}
        self._completed_requests: List[BackfillRequest] = []
        self._request_counter = 0

        # Data source handlers
        self._sources: Dict[str, Callable] = {}

    def register_source(
        self,
        name: str,
        handler: Callable[[str, date, date, str], pd.DataFrame]
    ) -> None:
        """
        Register a data source.

        Args:
            name: Source name
            handler: Function to fetch data (symbol, start, end, freq) -> DataFrame
        """
        self._sources[name] = handler
        logger.info(f"Registered backfill source: {name}")

    def detect_gaps(
        self,
        df: pd.DataFrame,
        symbol: str,
        frequency: str = '1min'
    ) -> List[DataGap]:
        """Detect gaps in data."""
        return self._gap_detector.detect_gaps(df, symbol, frequency)

    def create_backfill_request(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: str = '1d',
        priority: BackfillPriority = BackfillPriority.NORMAL,
        source: Optional[str] = None
    ) -> BackfillRequest:
        """
        Create a backfill request.

        Args:
            symbol: Symbol to backfill
            start_date: Start date
            end_date: End date
            frequency: Data frequency
            priority: Request priority
            source: Preferred data source

        Returns:
            Created backfill request
        """
        self._request_counter += 1
        request = BackfillRequest(
            request_id=f"BF-{self._request_counter:06d}",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            priority=priority,
            source=source
        )

        self._pending_requests.append(request)
        self._pending_requests.sort(key=lambda r: r.priority.value)

        logger.info(f"Created backfill request: {request.request_id}")
        return request

    def create_requests_from_gaps(
        self,
        gaps: List[DataGap],
        frequency: str = '1min'
    ) -> List[BackfillRequest]:
        """
        Create backfill requests from detected gaps.

        Args:
            gaps: List of data gaps
            frequency: Data frequency

        Returns:
            List of created requests
        """
        requests = []

        # Group gaps by symbol and consolidate
        by_symbol: Dict[str, List[DataGap]] = {}
        for gap in gaps:
            if gap.symbol not in by_symbol:
                by_symbol[gap.symbol] = []
            by_symbol[gap.symbol].append(gap)

        for symbol, symbol_gaps in by_symbol.items():
            # Consolidate nearby gaps
            consolidated = self._consolidate_gaps(symbol_gaps)

            for gap in consolidated:
                request = self.create_backfill_request(
                    symbol=symbol,
                    start_date=gap.start_time.date(),
                    end_date=gap.end_time.date(),
                    frequency=frequency,
                    priority=gap.priority
                )
                requests.append(request)

        return requests

    def _consolidate_gaps(
        self,
        gaps: List[DataGap],
        max_gap_days: int = 3
    ) -> List[DataGap]:
        """Consolidate nearby gaps into single requests."""
        if not gaps:
            return []

        sorted_gaps = sorted(gaps, key=lambda g: g.start_time)
        consolidated = [sorted_gaps[0]]

        for gap in sorted_gaps[1:]:
            last = consolidated[-1]
            days_between = (gap.start_time - last.end_time).days

            if days_between <= max_gap_days:
                # Merge gaps
                last.end_time = gap.end_time
                last.expected_bars += gap.expected_bars
                last.actual_bars += gap.actual_bars
                last.priority = BackfillPriority(
                    min(last.priority.value, gap.priority.value)
                )
            else:
                consolidated.append(gap)

        return consolidated

    def execute_request(
        self,
        request: BackfillRequest,
        store_callback: Optional[Callable[[pd.DataFrame], None]] = None
    ) -> RecoveryResult:
        """
        Execute a backfill request.

        Args:
            request: Backfill request to execute
            store_callback: Function to store recovered data

        Returns:
            Recovery result
        """
        start_time = datetime.now()
        request.status = RecoveryStatus.IN_PROGRESS
        request.started_at = start_time
        self._active_requests[request.request_id] = request

        try:
            # Try each source until successful
            sources_to_try = (
                [request.source] if request.source
                else list(self._sources.keys())
            )

            df = None
            for source_name in sources_to_try:
                if source_name not in self._sources:
                    continue

                try:
                    logger.info(f"Trying source {source_name} for {request.request_id}")
                    handler = self._sources[source_name]
                    df = handler(
                        request.symbol,
                        request.start_date,
                        request.end_date,
                        request.frequency
                    )

                    if df is not None and not df.empty:
                        logger.info(f"Retrieved {len(df)} bars from {source_name}")
                        break
                except Exception as e:
                    logger.warning(f"Source {source_name} failed: {e}")
                    continue

            if df is None or df.empty:
                request.status = RecoveryStatus.FAILED
                request.error_message = "No data retrieved from any source"
                return RecoveryResult(
                    request=request,
                    success=False,
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )

            # Validate data
            is_valid, errors = self._validator.validate(df)

            request.bars_requested = request.expected_bars if hasattr(request, 'expected_bars') else len(df)
            request.bars_received = len(df)

            if not is_valid:
                logger.warning(f"Validation errors: {errors}")
                # Attempt to clean data
                df = self._clean_data(df, errors)
                is_valid, errors = self._validator.validate(df)

            # Store data if valid
            if is_valid or len(errors) < 3:
                if store_callback:
                    store_callback(df)
                    logger.info(f"Stored {len(df)} bars for {request.symbol}")

                request.status = RecoveryStatus.COMPLETED
                request.completed_at = datetime.now()

                return RecoveryResult(
                    request=request,
                    success=True,
                    bars_recovered=len(df),
                    bars_validated=len(df) if is_valid else len(df) - len(errors),
                    validation_errors=errors,
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )
            else:
                request.status = RecoveryStatus.PARTIAL
                request.error_message = f"Validation failed: {errors}"

                return RecoveryResult(
                    request=request,
                    success=False,
                    bars_recovered=0,
                    validation_errors=errors,
                    duration_seconds=(datetime.now() - start_time).total_seconds()
                )

        except Exception as e:
            logger.error(f"Backfill request failed: {e}")
            request.status = RecoveryStatus.FAILED
            request.error_message = str(e)

            return RecoveryResult(
                request=request,
                success=False,
                duration_seconds=(datetime.now() - start_time).total_seconds()
            )
        finally:
            if request.request_id in self._active_requests:
                del self._active_requests[request.request_id]
            self._completed_requests.append(request)

    def _clean_data(
        self,
        df: pd.DataFrame,
        errors: List[str]
    ) -> pd.DataFrame:
        """Attempt to clean data based on validation errors."""
        df = df.copy()

        # Remove rows with nulls
        df = df.dropna()

        # Fix OHLC consistency
        if any('OHLC' in e for e in errors):
            # Ensure high >= all others
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            # Ensure low <= all others
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

        # Remove negative volumes
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]

        # Sort by timestamp
        df = df.sort_index()

        return df

    def process_pending(
        self,
        max_requests: Optional[int] = None,
        store_callback: Optional[Callable] = None
    ) -> List[RecoveryResult]:
        """
        Process pending backfill requests.

        Args:
            max_requests: Maximum requests to process
            store_callback: Function to store data

        Returns:
            List of recovery results
        """
        results = []
        count = 0
        max_count = max_requests or self._max_concurrent

        while self._pending_requests and count < max_count:
            request = self._pending_requests.pop(0)
            result = self.execute_request(request, store_callback)
            results.append(result)
            count += 1

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current backfill status."""
        return {
            'pending': len(self._pending_requests),
            'active': len(self._active_requests),
            'completed': len(self._completed_requests),
            'pending_requests': [
                {
                    'id': r.request_id,
                    'symbol': r.symbol,
                    'priority': r.priority.name,
                    'dates': f"{r.start_date} to {r.end_date}"
                }
                for r in self._pending_requests[:10]
            ],
            'active_requests': [
                {
                    'id': r.request_id,
                    'symbol': r.symbol,
                    'progress': r.progress
                }
                for r in self._active_requests.values()
            ]
        }


class DataReconciler:
    """
    Reconciles data between sources.

    Compares data from multiple sources to identify discrepancies.
    """

    def __init__(self, tolerance: float = 0.01):
        """
        Initialize reconciler.

        Args:
            tolerance: Price tolerance for differences (1% default)
        """
        self.tolerance = tolerance

    def reconcile(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        name1: str = "source1",
        name2: str = "source2"
    ) -> Dict[str, Any]:
        """
        Reconcile data from two sources.

        Args:
            source1: First DataFrame
            source2: Second DataFrame
            name1: Name for first source
            name2: Name for second source

        Returns:
            Reconciliation report
        """
        report = {
            'sources': [name1, name2],
            'matching': True,
            'discrepancies': []
        }

        # Align indexes
        common_index = source1.index.intersection(source2.index)

        if len(common_index) == 0:
            report['matching'] = False
            report['error'] = "No overlapping timestamps"
            return report

        s1 = source1.loc[common_index]
        s2 = source2.loc[common_index]

        # Compare price columns
        price_cols = ['open', 'high', 'low', 'close', 'price']

        for col in price_cols:
            if col in s1.columns and col in s2.columns:
                diff = (s1[col] - s2[col]).abs()
                pct_diff = diff / s1[col].replace(0, np.nan)

                issues = pct_diff > self.tolerance
                if issues.any():
                    report['matching'] = False
                    report['discrepancies'].append({
                        'column': col,
                        'count': int(issues.sum()),
                        'max_diff': float(pct_diff.max()),
                        'timestamps': list(common_index[issues][:5])
                    })

        # Compare volumes
        if 'volume' in s1.columns and 'volume' in s2.columns:
            vol_diff = (s1['volume'] - s2['volume']).abs()
            vol_pct = vol_diff / s1['volume'].replace(0, np.nan)

            issues = vol_pct > self.tolerance
            if issues.any():
                report['discrepancies'].append({
                    'column': 'volume',
                    'count': int(issues.sum()),
                    'max_diff': float(vol_pct.max())
                })

        # Add summary stats
        report['summary'] = {
            'common_rows': len(common_index),
            'source1_only': len(source1.index.difference(source2.index)),
            'source2_only': len(source2.index.difference(source1.index)),
            'total_discrepancies': sum(
                d['count'] for d in report['discrepancies']
            )
        }

        return report

    def select_best(
        self,
        sources: Dict[str, pd.DataFrame],
        priority: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Select best data from multiple sources.

        Args:
            sources: Dictionary of source_name -> DataFrame
            priority: Ordered list of preferred sources

        Returns:
            Best available data
        """
        if not sources:
            return pd.DataFrame()

        if priority is None:
            priority = list(sources.keys())

        # Start with highest priority source
        result = None
        for source_name in priority:
            if source_name in sources:
                df = sources[source_name]
                if result is None:
                    result = df.copy()
                else:
                    # Fill gaps from this source
                    missing = result.index.difference(df.index)
                    if len(missing) > 0:
                        result = pd.concat([result, df.loc[missing]])

        return result.sort_index() if result is not None else pd.DataFrame()
