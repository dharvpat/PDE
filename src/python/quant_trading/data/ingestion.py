"""
Data Ingestion Pipeline.

Orchestrates data ingestion from multiple sources with:
- Validation on ingestion
- Duplicate detection
- Gap filling and interpolation
- Database storage
- Logging and monitoring
- Fault tolerance and recovery

Reference: Design doc Section 5.1-5.3
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

from .providers import (
    DataProvider,
    DataProviderFactory,
    DataFrequency,
    DataProviderError,
    RateLimitExceeded,
)
from .validation import (
    DataValidationPipeline,
    ValidationResult,
    DataQuality,
)

logger = logging.getLogger(__name__)


class IngestionStatus(Enum):
    """Status of ingestion job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class IngestionResult:
    """Result of a data ingestion job."""
    symbol: str
    status: IngestionStatus
    rows_ingested: int = 0
    rows_skipped: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    validation_result: Optional[ValidationResult] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate job duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "status": self.status.value,
            "rows_ingested": self.rows_ingested,
            "rows_skipped": self.rows_skipped,
            "duration_seconds": self.duration_seconds,
            "validation": self.validation_result.to_dict() if self.validation_result else None,
            "error": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class IngestionConfig:
    """Configuration for data ingestion."""
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff: float = 2.0

    # Validation settings
    skip_validation: bool = False
    reject_bad_data: bool = True
    mark_suspect_data: bool = True

    # Gap filling
    fill_gaps: bool = True
    max_gap_to_fill: int = 5  # Maximum bars to forward fill

    # Deduplication
    check_duplicates: bool = True
    update_existing: bool = False

    # Batch settings
    batch_size: int = 10000
    parallel_symbols: int = 1  # No parallelism by default


class DataIngestionPipeline:
    """
    Orchestrate data ingestion from multiple sources.

    Example:
        >>> provider = DataProviderFactory.create('yahoo')
        >>> pipeline = DataIngestionPipeline(provider=provider)
        >>>
        >>> # Ingest historical data
        >>> results = pipeline.ingest_historical(
        ...     symbols=["SPY", "QQQ", "TLT"],
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31"
        ... )
        >>>
        >>> for result in results:
        ...     print(f"{result.symbol}: {result.status.value}")
    """

    def __init__(
        self,
        provider: DataProvider,
        db_session: Optional[Any] = None,
        config: Optional[IngestionConfig] = None,
        validator: Optional[DataValidationPipeline] = None,
    ):
        """
        Initialize ingestion pipeline.

        Args:
            provider: Data provider instance
            db_session: SQLAlchemy session for persistence
            config: Ingestion configuration
            validator: Data validator
        """
        self.provider = provider
        self.db_session = db_session
        self.config = config or IngestionConfig()
        self.validator = validator or DataValidationPipeline()

        # Ingestion stats
        self._stats = {
            "total_symbols": 0,
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
        }

        # Callbacks
        self._on_symbol_complete: Optional[Callable] = None
        self._on_error: Optional[Callable] = None

    def set_callbacks(
        self,
        on_complete: Optional[Callable[[IngestionResult], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> None:
        """Set callback functions for ingestion events."""
        self._on_symbol_complete = on_complete
        self._on_error = on_error

    def ingest_historical(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> List[IngestionResult]:
        """
        Ingest historical data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            List of IngestionResult for each symbol
        """
        results = []
        self._stats["total_symbols"] = len(symbols)

        logger.info(
            f"Starting historical ingestion for {len(symbols)} symbols "
            f"from {start_date} to {end_date}"
        )

        for symbol in symbols:
            result = self._ingest_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
            )

            results.append(result)

            if result.status == IngestionStatus.COMPLETED:
                self._stats["successful"] += 1
                self._stats["total_rows"] += result.rows_ingested
            else:
                self._stats["failed"] += 1

            if self._on_symbol_complete:
                self._on_symbol_complete(result)

        logger.info(
            f"Ingestion complete: {self._stats['successful']}/{self._stats['total_symbols']} "
            f"symbols, {self._stats['total_rows']} rows"
        )

        return results

    def _ingest_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency,
    ) -> IngestionResult:
        """
        Ingest data for a single symbol with retry logic.

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            frequency: Data frequency

        Returns:
            IngestionResult
        """
        result = IngestionResult(
            symbol=symbol,
            status=IngestionStatus.PENDING,
            start_time=datetime.now(),
        )

        retries = 0
        delay = self.config.retry_delay_seconds

        while retries <= self.config.max_retries:
            try:
                logger.info(f"Fetching {symbol} from {start_date} to {end_date}")

                # Fetch data
                df = self.provider.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                )

                if df.empty:
                    result.status = IngestionStatus.FAILED
                    result.error_message = "No data returned"
                    result.end_time = datetime.now()
                    return result

                # Validate data
                if not self.config.skip_validation:
                    validation = self.validator.validate_market_data(df)
                    result.validation_result = validation

                    if validation.quality == DataQuality.BAD and self.config.reject_bad_data:
                        result.status = IngestionStatus.FAILED
                        result.error_message = f"Data quality BAD: {len(validation.issues)} issues"
                        result.end_time = datetime.now()
                        return result

                # Clean data
                df = self._clean_data(df)

                # Fill gaps if configured
                if self.config.fill_gaps:
                    df = self._fill_gaps(df)

                # Check duplicates
                if self.config.check_duplicates and self.db_session:
                    df, skipped = self._remove_duplicates(df, symbol)
                    result.rows_skipped = skipped

                # Store in database
                if self.db_session:
                    self._store_data(df, symbol, frequency)

                result.rows_ingested = len(df)
                result.status = IngestionStatus.COMPLETED
                result.end_time = datetime.now()

                logger.info(f"Successfully ingested {len(df)} bars for {symbol}")
                return result

            except RateLimitExceeded as e:
                logger.warning(f"Rate limit for {symbol}: {e}")
                if retries < self.config.max_retries:
                    time.sleep(delay * 10)  # Longer wait for rate limits
                    retries += 1
                else:
                    result.status = IngestionStatus.FAILED
                    result.error_message = f"Rate limit exceeded: {e}"

            except DataProviderError as e:
                logger.warning(f"Provider error for {symbol}: {e}")
                if retries < self.config.max_retries:
                    time.sleep(delay)
                    delay *= self.config.retry_backoff
                    retries += 1
                else:
                    result.status = IngestionStatus.FAILED
                    result.error_message = str(e)

            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                if self._on_error:
                    self._on_error(symbol, e)
                result.status = IngestionStatus.FAILED
                result.error_message = str(e)
                break

        result.end_time = datetime.now()
        return result

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        df = df.copy()

        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Remove rows with all NaN prices
        price_cols = ['open', 'high', 'low', 'close']
        price_cols = [c for c in price_cols if c in df.columns]
        if price_cols:
            df = df.dropna(subset=price_cols, how='all')

        # Sort by index
        df = df.sort_index()

        # Remove timezone
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill small gaps in data."""
        if len(df) < 2:
            return df

        df = df.copy()

        # Forward fill small gaps
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].ffill(limit=self.config.max_gap_to_fill)

        # Volume: fill with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)

        return df

    def _remove_duplicates(
        self, df: pd.DataFrame, symbol: str
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove data that already exists in database.

        Returns:
            Tuple of (filtered DataFrame, count of skipped rows)
        """
        if not self.db_session:
            return df, 0

        # Query existing timestamps for this symbol
        from ..database.models import MarketPrice

        existing = self.db_session.query(MarketPrice.time).filter(
            MarketPrice.symbol == symbol,
            MarketPrice.time >= df.index.min(),
            MarketPrice.time <= df.index.max(),
        ).all()

        existing_times = {row[0] for row in existing}

        if not existing_times:
            return df, 0

        # Filter out existing
        original_len = len(df)
        mask = ~df.index.isin(existing_times)
        df = df[mask]
        skipped = original_len - len(df)

        if skipped > 0:
            logger.debug(f"Skipped {skipped} duplicate rows for {symbol}")

        return df, skipped

    def _store_data(
        self, df: pd.DataFrame, symbol: str, frequency: DataFrequency
    ) -> None:
        """Store data in TimescaleDB."""
        if not self.db_session:
            logger.warning("No database session, skipping storage")
            return

        from ..database.models import MarketPrice

        # Prepare batch insert
        records = []
        for timestamp, row in df.iterrows():
            record = MarketPrice(
                time=timestamp,
                symbol=symbol,
                price=row.get('close'),
                volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                bid=row.get('bid'),
                ask=row.get('ask'),
                data_quality='good',
            )
            records.append(record)

        # Batch insert
        try:
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
            logger.debug(f"Stored {len(records)} records for {symbol}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Database error storing {symbol}: {e}")
            raise

    def ingest_options_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
    ) -> IngestionResult:
        """
        Ingest options chain data.

        Args:
            underlying: Underlying symbol
            expiration: Specific expiration date (None for all)

        Returns:
            IngestionResult
        """
        result = IngestionResult(
            symbol=underlying,
            status=IngestionStatus.PENDING,
            start_time=datetime.now(),
        )

        try:
            logger.info(f"Fetching options chain for {underlying}")

            # Fetch options
            df = self.provider.get_options_chain(
                symbol=underlying,
                expiration=expiration,
            )

            if df.empty:
                result.status = IngestionStatus.FAILED
                result.error_message = "No options data returned"
                result.end_time = datetime.now()
                return result

            # Validate
            if not self.config.skip_validation:
                # Get spot price for validation
                try:
                    quote = self.provider.get_quote(underlying)
                    spot = quote.get('last', 0)
                except Exception:
                    spot = None

                validation = self.validator.validate_options_data(df, spot_price=spot)
                result.validation_result = validation

            # Store in database
            if self.db_session:
                self._store_options(df, underlying)

            result.rows_ingested = len(df)
            result.status = IngestionStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(f"Successfully ingested {len(df)} options for {underlying}")

        except Exception as e:
            logger.error(f"Error ingesting options for {underlying}: {e}")
            result.status = IngestionStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()

        return result

    def _store_options(self, df: pd.DataFrame, underlying: str) -> None:
        """Store options data in database."""
        if not self.db_session:
            return

        from ..database.models import OptionQuote

        records = []
        for _, row in df.iterrows():
            record = OptionQuote(
                time=row.get('timestamp', datetime.now()),
                underlying=underlying,
                expiration=row.get('expiration'),
                strike=row.get('strike'),
                option_type=row.get('option_type'),
                bid=row.get('bid'),
                ask=row.get('ask'),
                last=row.get('last'),
                volume=row.get('volume'),
                open_interest=row.get('open_interest'),
                implied_vol=row.get('implied_vol'),
                delta=row.get('delta'),
                gamma=row.get('gamma'),
                vega=row.get('vega'),
                theta=row.get('theta'),
            )
            records.append(record)

        try:
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset ingestion statistics."""
        self._stats = {
            "total_symbols": 0,
            "successful": 0,
            "failed": 0,
            "total_rows": 0,
        }


class IncrementalIngestion:
    """
    Handles incremental/real-time data ingestion.

    Maintains state of last ingested timestamp per symbol
    and only fetches new data.
    """

    def __init__(
        self,
        pipeline: DataIngestionPipeline,
        state_file: Optional[str] = None,
    ):
        """
        Initialize incremental ingestion.

        Args:
            pipeline: Data ingestion pipeline
            state_file: File to persist state (JSON)
        """
        self.pipeline = pipeline
        self.state_file = state_file
        self._last_timestamps: Dict[str, datetime] = {}

        if state_file:
            self._load_state()

    def _load_state(self) -> None:
        """Load state from file."""
        import json
        import os

        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._last_timestamps = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.items()
                    }
                logger.info(f"Loaded state for {len(self._last_timestamps)} symbols")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save state to file."""
        import json

        if not self.state_file:
            return

        try:
            data = {
                k: v.isoformat() for k, v in self._last_timestamps.items()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def update(
        self,
        symbols: List[str],
        lookback_days: int = 5,
    ) -> List[IngestionResult]:
        """
        Update data for symbols incrementally.

        Args:
            symbols: Symbols to update
            lookback_days: Default lookback if no prior state

        Returns:
            List of IngestionResult
        """
        results = []
        end_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in symbols:
            # Determine start date
            last_ts = self._last_timestamps.get(symbol)
            if last_ts:
                start_date = (last_ts + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_date = (
                    datetime.now() - timedelta(days=lookback_days)
                ).strftime('%Y-%m-%d')

            # Skip if up to date
            if last_ts and last_ts.date() >= datetime.now().date():
                logger.debug(f"{symbol} is up to date")
                continue

            # Ingest
            result = self.pipeline._ingest_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency=DataFrequency.DAILY,
            )

            results.append(result)

            # Update state on success
            if result.status == IngestionStatus.COMPLETED:
                self._last_timestamps[symbol] = datetime.now()

        self._save_state()
        return results

    def get_last_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get last ingested timestamp for symbol."""
        return self._last_timestamps.get(symbol)

    def set_last_timestamp(self, symbol: str, timestamp: datetime) -> None:
        """Set last ingested timestamp for symbol."""
        self._last_timestamps[symbol] = timestamp
        self._save_state()
