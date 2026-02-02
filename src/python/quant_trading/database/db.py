"""
Database access layer for the quantitative trading system.

Provides TimeSeriesDB class with:
- Connection pooling using SQLAlchemy Engine
- Session management with automatic commit/rollback
- Batch insert methods for high throughput
- Retry logic for transient failures
- Query helper methods for common patterns

Reference: Design doc Section 5.3 (Storage Schema)
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Any, Dict, Generator, List, Optional, Union, TYPE_CHECKING
import logging
import time

from sqlalchemy import create_engine, and_, or_, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, InterfaceError

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

from .models import (
    Base,
    MarketPrice,
    OptionQuote,
    ModelParameter,
    Signal,
    Position,
    PositionUpdate,
)

logger = logging.getLogger(__name__)


def retry_on_db_error(max_retries: int = 3, delay: float = 0.5):
    """
    Decorator for retrying database operations on transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds (with exponential backoff)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, InterfaceError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2**attempt)
                        logger.warning(
                            f"Database error (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {sleep_time:.1f}s..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Database error after {max_retries} attempts: {e}")
                        raise
            raise last_error

        return wrapper

    return decorator


class TimeSeriesDB:
    """
    Database access layer for time-series data.

    Provides high-level methods for querying and inserting data
    with connection pooling and retry logic.

    Example:
        >>> db = TimeSeriesDB("postgresql://user:pass@localhost/quant_trading_db")
        >>> db.insert_market_prices([{"time": now, "symbol": "SPY", "price": 450.0}])
        >>> prices = db.get_market_prices("SPY", start_time)
    """

    def __init__(
        self,
        connection_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        echo: bool = False,
    ):
        """
        Initialize database connection.

        Args:
            connection_url: PostgreSQL connection string
                Format: postgresql://user:pass@host:port/database
            pool_size: Max connections in pool (default: 20)
            max_overflow: Max connections beyond pool_size (default: 10)
            pool_pre_ping: Verify connections before use (default: True)
            echo: Log SQL statements (default: False)
        """
        self.connection_url = connection_url
        self.engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
            echo=echo,
        )
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Initialized TimeSeriesDB with pool_size={pool_size}")

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide transactional scope around series of operations.

        Automatically commits on success and rolls back on exception.

        Example:
            >>> with db.session_scope() as session:
            ...     session.add(MarketPrice(...))
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_tables(self):
        """Create all tables defined in models (for testing/setup)."""
        Base.metadata.create_all(self.engine)
        logger.info("Created database tables")

    def drop_tables(self):
        """Drop all tables (for testing/cleanup)."""
        Base.metadata.drop_all(self.engine)
        logger.warning("Dropped all database tables")

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    @retry_on_db_error()
    def insert_market_prices(self, prices: List[Dict[str, Any]]) -> int:
        """
        Bulk insert market prices.

        Args:
            prices: List of dicts with keys: time, symbol, price, volume, bid, ask

        Returns:
            Number of rows inserted
        """
        if not prices:
            return 0

        with self.session_scope() as session:
            session.bulk_insert_mappings(MarketPrice, prices)

        logger.info(f"Inserted {len(prices)} market prices")
        return len(prices)

    @retry_on_db_error()
    def get_market_prices(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve market prices for symbol in time range.

        Args:
            symbol: Ticker symbol
            start_time: Start of time range
            end_time: End of time range (default: now)

        Returns:
            DataFrame with columns: time, symbol, price, volume, bid, ask
        """
        if end_time is None:
            end_time = datetime.utcnow()

        with self.session_scope() as session:
            query = (
                session.query(MarketPrice)
                .filter(
                    and_(
                        MarketPrice.symbol == symbol,
                        MarketPrice.time >= start_time,
                        MarketPrice.time <= end_time,
                    )
                )
                .order_by(MarketPrice.time)
            )
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to dict inside session scope to avoid detached instance error
            data = [r.to_dict() for r in results]
            return pd.DataFrame(data)

    @retry_on_db_error()
    def get_latest_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest price for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with price data or None if not found
        """
        with self.session_scope() as session:
            result = (
                session.query(MarketPrice)
                .filter(MarketPrice.symbol == symbol)
                .order_by(MarketPrice.time.desc())
                .first()
            )

            # Convert to dict inside session scope to avoid detached instance error
            if result:
                return result.to_dict()
            return None

    # =========================================================================
    # Options Data Methods
    # =========================================================================

    @retry_on_db_error()
    def insert_option_quotes(self, quotes: List[Dict[str, Any]]) -> int:
        """
        Bulk insert option quotes.

        Args:
            quotes: List of dicts with option quote data

        Returns:
            Number of rows inserted
        """
        if not quotes:
            return 0

        with self.session_scope() as session:
            session.bulk_insert_mappings(OptionQuote, quotes)

        logger.info(f"Inserted {len(quotes)} option quotes")
        return len(quotes)

    @retry_on_db_error()
    def get_option_chain(
        self,
        underlying: str,
        expiration: Union[date, datetime],
        quote_time: Optional[datetime] = None,
        lookback_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Retrieve full option chain for underlying at expiration.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            quote_time: Quote time (default: latest)
            lookback_minutes: How far back to look for latest quotes

        Returns:
            DataFrame with option chain
        """
        if isinstance(expiration, datetime):
            expiration = expiration.date()

        with self.session_scope() as session:
            query = session.query(OptionQuote).filter(
                and_(
                    OptionQuote.underlying == underlying,
                    OptionQuote.expiration == expiration,
                )
            )

            if quote_time:
                query = query.filter(OptionQuote.time == quote_time)
            else:
                cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)
                query = query.filter(OptionQuote.time >= cutoff)

            query = query.order_by(OptionQuote.strike, OptionQuote.option_type)
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to dict inside session scope to avoid detached instance error
            data = [r.to_dict() for r in results]
            return pd.DataFrame(data)

    @retry_on_db_error()
    def get_options_for_calibration(
        self,
        underlying: str,
        expiration: Union[date, datetime],
        min_moneyness: float = 0.8,
        max_moneyness: float = 1.2,
        min_volume: int = 10,
    ) -> pd.DataFrame:
        """
        Get filtered options for model calibration.

        Filters out illiquid options and extreme strikes.

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            min_moneyness: Minimum moneyness ratio (default: 0.8)
            max_moneyness: Maximum moneyness ratio (default: 1.2)
            min_volume: Minimum volume filter (default: 10)

        Returns:
            DataFrame with filtered options
        """
        chain = self.get_option_chain(underlying, expiration)

        if chain.empty:
            return chain

        # Get latest underlying price
        latest = self.get_latest_price(underlying)
        if not latest:
            return chain

        spot = latest["price"]

        # Filter by moneyness
        chain["moneyness"] = chain["strike"] / spot
        filtered = chain[
            (chain["moneyness"] >= min_moneyness)
            & (chain["moneyness"] <= max_moneyness)
        ]

        # Filter by volume
        if "volume" in filtered.columns:
            filtered = filtered[
                (filtered["volume"] >= min_volume) | (filtered["volume"].isna())
            ]

        return filtered

    # =========================================================================
    # Model Parameters Methods
    # =========================================================================

    @retry_on_db_error()
    def store_model_parameters(
        self,
        model_type: str,
        underlying: str,
        parameters: Dict[str, float],
        fit_quality: Dict[str, Any],
        maturity: Optional[Union[date, datetime]] = None,
        converged: bool = True,
        calibration_time_ms: Optional[int] = None,
        n_iterations: Optional[int] = None,
    ) -> None:
        """
        Store calibrated model parameters.

        Args:
            model_type: 'heston', 'sabr', or 'ou'
            underlying: Underlying symbol
            parameters: Model parameter dict
            fit_quality: Fit quality metrics dict
            maturity: Maturity date (for SABR), None for Heston/OU
            converged: Whether calibration converged
            calibration_time_ms: Calibration time in milliseconds
            n_iterations: Number of optimization iterations
        """
        if isinstance(maturity, datetime):
            maturity = maturity.date()
        elif maturity is None:
            # Use placeholder date for models without maturity (required for SQLite)
            maturity = date(1970, 1, 1)

        param = ModelParameter(
            time=datetime.utcnow(),
            model_type=model_type,
            underlying=underlying,
            maturity=maturity,
            parameters=parameters,
            fit_quality=fit_quality,
            converged=converged,
            calibration_time_ms=calibration_time_ms,
            n_iterations=n_iterations,
        )

        with self.session_scope() as session:
            session.add(param)

        logger.info(f"Stored {model_type} parameters for {underlying}")

    @retry_on_db_error()
    def get_latest_model_parameters(
        self,
        model_type: str,
        underlying: str,
        maturity: Optional[Union[date, datetime]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve latest model parameters.

        Args:
            model_type: 'heston', 'sabr', or 'ou'
            underlying: Underlying symbol
            maturity: Maturity date (for SABR)

        Returns:
            Dict with 'parameters', 'fit_quality', 'time', 'converged'
            or None if not found
        """
        if isinstance(maturity, datetime):
            maturity = maturity.date()
        elif maturity is None:
            # Use placeholder date for models without maturity
            maturity = date(1970, 1, 1)

        with self.session_scope() as session:
            query = (
                session.query(ModelParameter)
                .filter(
                    and_(
                        ModelParameter.model_type == model_type,
                        ModelParameter.underlying == underlying,
                        ModelParameter.maturity == maturity,
                    )
                )
                .order_by(ModelParameter.time.desc())
            )
            result = query.first()

            # Convert to dict inside session scope to avoid detached instance error
            if result:
                return result.to_dict()
            return None

    @retry_on_db_error()
    def get_model_parameters_history(
        self,
        model_type: str,
        underlying: str,
        maturity: Optional[Union[date, datetime]] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get historical model parameters for analysis.

        Args:
            model_type: Model type
            underlying: Underlying symbol
            maturity: Maturity date (optional)
            days: Number of days of history

        Returns:
            DataFrame with parameter history
        """
        if isinstance(maturity, datetime):
            maturity = maturity.date()
        elif maturity is None:
            # Use placeholder date for models without maturity
            maturity = date(1970, 1, 1)

        start_time = datetime.utcnow() - timedelta(days=days)

        with self.session_scope() as session:
            query = (
                session.query(ModelParameter)
                .filter(
                    and_(
                        ModelParameter.model_type == model_type,
                        ModelParameter.underlying == underlying,
                        ModelParameter.maturity == maturity,
                        ModelParameter.time >= start_time,
                    )
                )
                .order_by(ModelParameter.time)
            )
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Convert to dict inside session scope to avoid detached instance error
            data = [r.to_dict() for r in results]
            return pd.DataFrame(data)

    # =========================================================================
    # Signals Methods
    # =========================================================================

    @retry_on_db_error()
    def insert_signal(
        self,
        strategy: str,
        signal_type: str,
        signal_strength: float,
        underlying: Optional[str] = None,
        rationale: Optional[str] = None,
        metadata: Optional[Dict] = None,
        expected_return: Optional[float] = None,
        expected_risk: Optional[float] = None,
    ) -> None:
        """
        Insert trading signal.

        Args:
            strategy: Strategy name
            signal_type: Signal type ('entry_long', 'entry_short', 'exit', 'hold', 'reduce')
            signal_strength: Confidence level (0.0 to 1.0)
            underlying: Asset or spread identifier
            rationale: Human-readable explanation
            metadata: Strategy-specific metadata
            expected_return: Expected return if signal acted upon
            expected_risk: Expected risk/volatility
        """
        signal = Signal(
            time=datetime.utcnow(),
            strategy=strategy,
            underlying=underlying or "",
            signal_type=signal_type,
            signal_strength=Decimal(str(signal_strength)),
            rationale=rationale,
            signal_metadata=metadata,
            expected_return=Decimal(str(expected_return)) if expected_return else None,
            expected_risk=Decimal(str(expected_risk)) if expected_risk else None,
        )

        with self.session_scope() as session:
            session.add(signal)

        logger.info(f"Inserted signal: {strategy} {signal_type} {underlying}")

    @retry_on_db_error()
    def get_latest_signals(
        self,
        strategy: Optional[str] = None,
        lookback_minutes: int = 60,
        min_strength: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get latest signals.

        Args:
            strategy: Strategy name (None for all)
            lookback_minutes: How far back to look
            min_strength: Minimum signal strength

        Returns:
            List of signal dicts
        """
        cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)

        with self.session_scope() as session:
            query = session.query(Signal).filter(
                and_(
                    Signal.time >= cutoff,
                    Signal.signal_strength >= min_strength,
                )
            )

            if strategy:
                query = query.filter(Signal.strategy == strategy)

            results = query.order_by(Signal.time.desc()).all()

            # Convert to dict inside session scope to avoid detached instance error
            return [r.to_dict() for r in results]

    @retry_on_db_error()
    def get_actionable_signals(
        self,
        min_strength: float = 0.6,
        lookback_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Get actionable signals (entry signals with high confidence).

        Args:
            min_strength: Minimum signal strength (default: 0.6)
            lookback_minutes: How far back to look

        Returns:
            List of actionable signal dicts
        """
        cutoff = datetime.utcnow() - timedelta(minutes=lookback_minutes)

        with self.session_scope() as session:
            results = (
                session.query(Signal)
                .filter(
                    and_(
                        Signal.time >= cutoff,
                        Signal.signal_strength >= min_strength,
                        Signal.signal_type.in_(["entry_long", "entry_short"]),
                    )
                )
                .order_by(Signal.signal_strength.desc())
                .all()
            )

            # Convert to dict inside session scope to avoid detached instance error
            return [r.to_dict() for r in results]

    # =========================================================================
    # Positions Methods
    # =========================================================================

    @retry_on_db_error()
    def create_position(
        self,
        strategy: str,
        underlying: str,
        direction: str,
        quantity: float,
        entry_price: float,
        entry_commission: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Create new position.

        Args:
            strategy: Strategy name
            underlying: Asset symbol
            direction: 'long' or 'short'
            quantity: Position size
            entry_price: Entry price
            entry_commission: Entry commission
            metadata: Additional metadata

        Returns:
            Position ID (UUID as string)
        """
        position = Position(
            opened_at=datetime.utcnow(),
            strategy=strategy,
            underlying=underlying,
            direction=direction,
            quantity=Decimal(str(quantity)),
            entry_price=Decimal(str(entry_price)),
            entry_commission=(
                Decimal(str(entry_commission)) if entry_commission else None
            ),
            position_metadata=metadata,
        )

        with self.session_scope() as session:
            session.add(position)
            session.flush()
            position_id = str(position.position_id)

        logger.info(
            f"Created position {position_id}: {strategy} {direction} {underlying}"
        )
        return position_id

    @retry_on_db_error()
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_commission: float = 0.0,
    ) -> float:
        """
        Close position and calculate PnL.

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_commission: Exit commission

        Returns:
            Realized PnL

        Raises:
            ValueError: If position not found or already closed
        """
        with self.session_scope() as session:
            position = (
                session.query(Position)
                .filter(Position.position_id == position_id)
                .first()
            )

            if not position:
                raise ValueError(f"Position {position_id} not found")

            if not position.is_open:
                raise ValueError(f"Position {position_id} already closed")

            # Calculate realized PnL
            entry = float(position.entry_price)
            qty = float(position.quantity)

            if position.direction == "long":
                pnl = (exit_price - entry) * qty
            else:
                pnl = (entry - exit_price) * qty

            # Subtract commissions
            total_commission = float(position.entry_commission or 0) + exit_commission
            pnl -= total_commission

            # Update position
            position.closed_at = datetime.utcnow()
            position.exit_price = Decimal(str(exit_price))
            position.exit_commission = Decimal(str(exit_commission))
            position.realized_pnl = Decimal(str(pnl))

            # Add audit record
            audit = PositionUpdate(
                position_id=position.position_id,
                field_name="closed_at",
                old_value=None,
                new_value=str(position.closed_at),
                updated_by="system",
            )
            session.add(audit)

        logger.info(f"Closed position {position_id}: PnL = ${pnl:.2f}")
        return pnl

    @retry_on_db_error()
    def update_position_price(
        self,
        position_id: str,
        current_price: float,
    ) -> float:
        """
        Update position with current market price and calculate unrealized PnL.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Unrealized PnL
        """
        with self.session_scope() as session:
            position = (
                session.query(Position)
                .filter(Position.position_id == position_id)
                .first()
            )

            if not position:
                raise ValueError(f"Position {position_id} not found")

            if not position.is_open:
                raise ValueError(f"Position {position_id} is closed")

            # Calculate unrealized PnL
            unrealized = position.calculate_unrealized_pnl(current_price)

            position.current_price = Decimal(str(current_price))
            position.unrealized_pnl = Decimal(str(unrealized))

        return unrealized

    @retry_on_db_error()
    def get_open_positions(
        self,
        strategy: Optional[str] = None,
        underlying: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get all open positions.

        Args:
            strategy: Filter by strategy (optional)
            underlying: Filter by underlying (optional)

        Returns:
            List of position dicts
        """
        with self.session_scope() as session:
            query = session.query(Position).filter(Position.closed_at.is_(None))

            if strategy:
                query = query.filter(Position.strategy == strategy)
            if underlying:
                query = query.filter(Position.underlying == underlying)

            results = query.order_by(Position.opened_at.desc()).all()

            # Convert to dict inside session scope to avoid detached instance error
            return [r.to_dict() for r in results]

    @retry_on_db_error()
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position dict or None if not found
        """
        with self.session_scope() as session:
            result = (
                session.query(Position)
                .filter(Position.position_id == position_id)
                .first()
            )

            # Convert to dict inside session scope to avoid detached instance error
            if result:
                return result.to_dict()
            return None

    @retry_on_db_error()
    def get_positions_summary(
        self,
        strategy: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for positions.

        Args:
            strategy: Filter by strategy (optional)
            days: Number of days to include

        Returns:
            Dict with summary statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with self.session_scope() as session:
            query = session.query(Position).filter(Position.opened_at >= cutoff)

            if strategy:
                query = query.filter(Position.strategy == strategy)

            results = query.all()

            if not results:
                return {
                    "total_positions": 0,
                    "open_positions": 0,
                    "closed_positions": 0,
                    "total_pnl": 0.0,
                    "win_rate": 0.0,
                    "avg_pnl": 0.0,
                }

            # Process results inside session scope to avoid detached instance error
            open_positions = [p for p in results if p.is_open]
            closed_positions = [p for p in results if not p.is_open]

            total_pnl = sum(float(p.realized_pnl or 0) for p in closed_positions)
            winning = [p for p in closed_positions if float(p.realized_pnl or 0) > 0]
            win_rate = len(winning) / len(closed_positions) if closed_positions else 0

            return {
                "total_positions": len(results),
                "open_positions": len(open_positions),
                "closed_positions": len(closed_positions),
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "avg_pnl": total_pnl / len(closed_positions) if closed_positions else 0.0,
            }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @retry_on_db_error()
    def refresh_materialized_views(self) -> None:
        """Refresh all materialized views."""
        with self.engine.connect() as conn:
            conn.execute(text("SELECT refresh_all_materialized_views()"))
            conn.commit()
        logger.info("Refreshed materialized views")

    @retry_on_db_error()
    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> List[Any]:
        """
        Execute raw SQL query.

        Args:
            sql: SQL query string
            params: Query parameters (optional)

        Returns:
            List of result rows
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = result.fetchall()
        return rows

    def health_check(self) -> Dict[str, Any]:
        """
        Check database health and connectivity.

        Returns:
            Dict with health status
        """
        try:
            start = time.time()
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            latency_ms = (time.time() - start) * 1000

            # Get connection pool stats
            pool = self.engine.pool
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "pool_size": pool.size(),
                "pool_checkedin": pool.checkedin(),
                "pool_checkedout": pool.checkedout(),
                "pool_overflow": pool.overflow(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
