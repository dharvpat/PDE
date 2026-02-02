"""
SQLAlchemy ORM models for the quantitative trading database.

Defines models for:
- MarketPrice: Equity tick data
- OptionQuote: Options chain with Greeks
- ModelParameter: Calibrated model parameters (Heston, SABR, OU)
- Signal: Trading signals
- Position: Position tracking with PnL
- PositionUpdate: Audit trail for position changes

Reference: Design doc Section 5.3 (Storage Schema)
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Optional, Any, List
import uuid

from sqlalchemy import (
    Column,
    String,
    DateTime,
    Date,
    Numeric,
    Integer,
    BigInteger,
    Boolean,
    Text,
    ForeignKey,
    CheckConstraint,
    Index,
    JSON,
    TypeDecorator,
)
from sqlalchemy.dialects.postgresql import JSONB as PostgresJSONB, UUID as PostgresUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship


class JSONB(TypeDecorator):
    """
    Cross-database compatible JSONB type.

    Uses PostgreSQL JSONB when available, falls back to standard JSON
    for other databases (SQLite, etc.).
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgresJSONB())
        else:
            return dialect.type_descriptor(JSON())


class UUID(TypeDecorator):
    """
    Cross-database compatible UUID type.

    Uses PostgreSQL UUID when available, falls back to String(36)
    for other databases (SQLite, etc.).
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgresUUID(as_uuid=True))
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        # For SQLite, convert UUID to string
        if hasattr(value, 'hex'):
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        # For SQLite, keep as string
        return value

Base = declarative_base()


class MarketPrice(Base):
    """
    Market price tick data.

    Stores equity price data with TimescaleDB hypertable optimization.
    Time-partitioned by day for efficient time-range queries.

    Attributes:
        time: Timestamp of the price observation
        symbol: Ticker symbol (e.g., 'SPY', 'AAPL')
        price: Last traded price
        volume: Trading volume
        bid: Best bid price
        ask: Best ask price
        exchange: Exchange code
        data_quality: Data quality flag ('good', 'suspect', 'bad')
    """

    __tablename__ = "market_prices"

    time: datetime = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol: str = Column(String(16), primary_key=True, nullable=False)
    price: Decimal = Column(Numeric(12, 4), nullable=False)
    volume: int = Column(BigInteger)
    bid: Optional[Decimal] = Column(Numeric(12, 4))
    ask: Optional[Decimal] = Column(Numeric(12, 4))
    exchange: Optional[str] = Column(String(16))
    data_quality: str = Column(String(16), default="good")

    __table_args__ = (
        CheckConstraint("price > 0", name="check_price_positive"),
        CheckConstraint("volume >= 0", name="check_volume_non_negative"),
        CheckConstraint("bid > 0 OR bid IS NULL", name="check_bid_positive"),
        CheckConstraint("ask > 0 OR ask IS NULL", name="check_ask_positive"),
        CheckConstraint(
            "data_quality IN ('good', 'suspect', 'bad')",
            name="check_data_quality_valid",
        ),
        Index("idx_market_prices_symbol_time", "symbol", "time"),
    )

    @property
    def mid_price(self) -> Optional[float]:
        """Compute mid price from bid and ask."""
        if self.bid is not None and self.ask is not None:
            return float((self.bid + self.ask) / 2)
        return None

    @property
    def spread(self) -> Optional[float]:
        """Compute bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return float(self.ask - self.bid)
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        """Compute bid-ask spread in basis points."""
        mid = self.mid_price
        if mid and mid > 0 and self.spread is not None:
            return (self.spread / mid) * 10000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "symbol": self.symbol,
            "price": float(self.price) if self.price else None,
            "volume": self.volume,
            "bid": float(self.bid) if self.bid else None,
            "ask": float(self.ask) if self.ask else None,
            "exchange": self.exchange,
            "data_quality": self.data_quality,
        }

    def __repr__(self) -> str:
        return f"<MarketPrice({self.symbol} @ {self.time}: ${self.price})>"


class OptionQuote(Base):
    """
    Options chain quote data.

    Stores options pricing data including Greeks and implied volatility.
    Used for model calibration and volatility surface analysis.

    Attributes:
        time: Timestamp of the quote
        underlying: Underlying symbol
        expiration: Option expiration date
        strike: Strike price
        option_type: 'call' or 'put'
        bid: Bid price
        ask: Ask price
        last: Last traded price
        volume: Daily volume
        open_interest: Open interest
        implied_vol: Implied volatility
        delta, gamma, vega, theta, rho: Option Greeks
    """

    __tablename__ = "option_quotes"

    time: datetime = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    underlying: str = Column(String(16), primary_key=True, nullable=False)
    expiration: date = Column(Date, primary_key=True, nullable=False)
    strike: Decimal = Column(Numeric(10, 2), primary_key=True, nullable=False)
    option_type: str = Column(String(4), primary_key=True, nullable=False)

    # Prices
    bid: Optional[Decimal] = Column(Numeric(10, 4))
    ask: Optional[Decimal] = Column(Numeric(10, 4))
    last: Optional[Decimal] = Column(Numeric(10, 4))
    volume: Optional[int] = Column(Integer)
    open_interest: Optional[int] = Column(Integer)

    # Greeks
    implied_vol: Optional[Decimal] = Column(Numeric(6, 4))
    delta: Optional[Decimal] = Column(Numeric(6, 4))
    gamma: Optional[Decimal] = Column(Numeric(8, 6))
    vega: Optional[Decimal] = Column(Numeric(8, 4))
    theta: Optional[Decimal] = Column(Numeric(8, 4))
    rho: Optional[Decimal] = Column(Numeric(8, 4))

    __table_args__ = (
        CheckConstraint("option_type IN ('call', 'put')", name="check_option_type"),
        CheckConstraint("strike > 0", name="check_strike_positive"),
        CheckConstraint("bid >= 0 OR bid IS NULL", name="check_bid_non_negative"),
        CheckConstraint("ask >= 0 OR ask IS NULL", name="check_ask_non_negative"),
        CheckConstraint("volume >= 0 OR volume IS NULL", name="check_volume_non_neg"),
        CheckConstraint(
            "open_interest >= 0 OR open_interest IS NULL",
            name="check_oi_non_negative",
        ),
        CheckConstraint(
            "(implied_vol >= 0 AND implied_vol <= 5.0) OR implied_vol IS NULL",
            name="check_iv_range",
        ),
        CheckConstraint(
            "(delta >= -1 AND delta <= 1) OR delta IS NULL",
            name="check_delta_range",
        ),
        CheckConstraint("gamma >= 0 OR gamma IS NULL", name="check_gamma_non_negative"),
        Index(
            "idx_option_quotes_calibration", "underlying", "expiration", "time", "strike"
        ),
        Index("idx_option_quotes_strike", "underlying", "time", "strike", "option_type"),
    )

    @property
    def mid_price(self) -> Optional[float]:
        """Compute mid price from bid and ask."""
        if self.bid is not None and self.ask is not None:
            return float((self.bid + self.ask) / 2)
        return None

    @property
    def bid_ask_spread(self) -> Optional[float]:
        """Compute relative bid-ask spread."""
        mid = self.mid_price
        if mid and mid > 0 and self.bid is not None and self.ask is not None:
            return float((self.ask - self.bid) / mid)
        return None

    @property
    def days_to_expiry(self) -> Optional[int]:
        """Calculate days until expiration."""
        if self.expiration and self.time:
            return (self.expiration - self.time.date()).days
        return None

    @property
    def is_call(self) -> bool:
        """Check if this is a call option."""
        return self.option_type == "call"

    @property
    def moneyness(self) -> Optional[float]:
        """
        Calculate moneyness (K/S approximation using strike).
        Note: This is a proxy without the underlying price.
        """
        return float(self.strike) if self.strike else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "underlying": self.underlying,
            "expiration": self.expiration,
            "strike": float(self.strike) if self.strike else None,
            "option_type": self.option_type,
            "bid": float(self.bid) if self.bid else None,
            "ask": float(self.ask) if self.ask else None,
            "mid_price": self.mid_price,
            "implied_vol": float(self.implied_vol) if self.implied_vol else None,
            "delta": float(self.delta) if self.delta else None,
            "gamma": float(self.gamma) if self.gamma else None,
            "vega": float(self.vega) if self.vega else None,
            "theta": float(self.theta) if self.theta else None,
            "rho": float(self.rho) if self.rho else None,
            "volume": self.volume,
            "open_interest": self.open_interest,
        }

    def __repr__(self) -> str:
        return (
            f"<OptionQuote({self.underlying} {self.strike} {self.option_type} "
            f"exp:{self.expiration})>"
        )


class ModelParameter(Base):
    """
    Calibrated model parameters.

    Stores parameters for Heston, SABR, and OU models with fit quality metrics.
    Uses JSONB for flexible parameter storage.

    Attributes:
        time: Calibration timestamp
        model_type: 'heston', 'sabr', or 'ou'
        underlying: Underlying symbol
        maturity: Maturity date (for SABR), None for Heston/OU
        parameters: Model parameters as JSONB
        fit_quality: Fit quality metrics as JSONB
        calibration_time_ms: Calibration duration in milliseconds
        n_iterations: Number of optimization iterations
        converged: Whether calibration converged

    Parameter Structures:
        Heston: {kappa, theta, sigma, rho, v0}
        SABR: {alpha, beta, rho, nu}
        OU: {theta, mu, sigma}

    Fit Quality Structures:
        Common: {rmse, r_squared}
        Heston: {feller_satisfied}
        OU: {log_likelihood, aic, bic}
    """

    __tablename__ = "model_parameters"

    time: datetime = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    model_type: str = Column(String(16), primary_key=True, nullable=False)
    underlying: str = Column(String(16), primary_key=True, nullable=False)
    maturity: Optional[date] = Column(Date, primary_key=True, default=None)

    parameters: Dict = Column(JSONB, nullable=False)
    fit_quality: Dict = Column(JSONB, nullable=False)

    calibration_time_ms: Optional[int] = Column(Integer)
    n_iterations: Optional[int] = Column(Integer)
    converged: Optional[bool] = Column(Boolean)

    __table_args__ = (
        CheckConstraint(
            "model_type IN ('heston', 'sabr', 'ou')",
            name="check_model_type_valid",
        ),
        Index(
            "idx_model_params_latest", "model_type", "underlying", "maturity", "time"
        ),
    )

    def get_param(self, key: str) -> Optional[float]:
        """Get parameter value by key."""
        if self.parameters and key in self.parameters:
            return float(self.parameters[key])
        return None

    def get_fit_metric(self, metric: str) -> Optional[float]:
        """Get fit quality metric by name."""
        if self.fit_quality and metric in self.fit_quality:
            return float(self.fit_quality[metric])
        return None

    @property
    def is_valid_fit(self) -> bool:
        """Check if this is a valid calibration result."""
        if not self.converged:
            return False
        rmse = self.get_fit_metric("rmse")
        if rmse is not None and rmse > 0.1:  # 10% RMSE threshold
            return False
        return True

    @property
    def feller_satisfied(self) -> Optional[bool]:
        """Check Feller condition for Heston model."""
        if self.model_type != "heston":
            return None
        return self.fit_quality.get("feller_satisfied", None)

    def to_heston_params(self) -> Optional[Dict[str, float]]:
        """Extract Heston parameters if model_type is 'heston'."""
        if self.model_type != "heston" or not self.parameters:
            return None
        return {
            "kappa": float(self.parameters.get("kappa", 0)),
            "theta": float(self.parameters.get("theta", 0)),
            "sigma": float(self.parameters.get("sigma", 0)),
            "rho": float(self.parameters.get("rho", 0)),
            "v0": float(self.parameters.get("v0", 0)),
        }

    def to_sabr_params(self) -> Optional[Dict[str, float]]:
        """Extract SABR parameters if model_type is 'sabr'."""
        if self.model_type != "sabr" or not self.parameters:
            return None
        return {
            "alpha": float(self.parameters.get("alpha", 0)),
            "beta": float(self.parameters.get("beta", 0)),
            "rho": float(self.parameters.get("rho", 0)),
            "nu": float(self.parameters.get("nu", 0)),
        }

    def to_ou_params(self) -> Optional[Dict[str, float]]:
        """Extract OU parameters if model_type is 'ou'."""
        if self.model_type != "ou" or not self.parameters:
            return None
        return {
            "theta": float(self.parameters.get("theta", 0)),
            "mu": float(self.parameters.get("mu", 0)),
            "sigma": float(self.parameters.get("sigma", 0)),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "model_type": self.model_type,
            "underlying": self.underlying,
            "maturity": self.maturity,
            "parameters": self.parameters,
            "fit_quality": self.fit_quality,
            "calibration_time_ms": self.calibration_time_ms,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
        }

    def __repr__(self) -> str:
        return f"<ModelParameter({self.model_type} for {self.underlying} @ {self.time})>"


class Signal(Base):
    """
    Trading signals.

    Stores signals generated by trading strategies with confidence levels
    and strategy-specific metadata.

    Attributes:
        time: Signal generation timestamp
        strategy: Strategy name (e.g., 'vol_arb', 'mean_reversion')
        underlying: Asset or spread identifier
        signal_type: Signal type ('entry_long', 'entry_short', 'exit', 'hold', 'reduce')
        signal_strength: Confidence level (0.0 to 1.0)
        metadata: Strategy-specific metadata as JSONB
        rationale: Human-readable explanation
        expected_return: Expected return if signal acted upon
        expected_risk: Expected risk/volatility
    """

    __tablename__ = "signals"

    time: datetime = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    strategy: str = Column(String(32), primary_key=True, nullable=False)
    underlying: Optional[str] = Column(String(32), primary_key=True, default="")

    signal_type: str = Column(String(16), nullable=False)
    signal_strength: Decimal = Column(Numeric(4, 3), nullable=False)

    signal_metadata: Optional[Dict] = Column("metadata", JSONB)
    rationale: Optional[str] = Column(Text)
    expected_return: Optional[Decimal] = Column(Numeric(8, 4))
    expected_risk: Optional[Decimal] = Column(Numeric(8, 4))

    __table_args__ = (
        CheckConstraint(
            "signal_type IN ('entry_long', 'entry_short', 'exit', 'hold', 'reduce')",
            name="check_signal_type_valid",
        ),
        CheckConstraint(
            "signal_strength >= 0.0 AND signal_strength <= 1.0",
            name="check_signal_strength_range",
        ),
        Index("idx_signals_strategy_time", "strategy", "time"),
        Index("idx_signals_underlying_time", "underlying", "time"),
        Index("idx_signals_type", "signal_type", "time"),
    )

    @property
    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal."""
        return self.signal_type in ("entry_long", "entry_short")

    @property
    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal."""
        return self.signal_type == "exit"

    @property
    def is_actionable(self) -> bool:
        """Check if signal is actionable (strength > 0.6)."""
        return float(self.signal_strength) > 0.6

    @property
    def expected_sharpe(self) -> Optional[float]:
        """Calculate expected Sharpe ratio from return/risk."""
        if self.expected_return is not None and self.expected_risk is not None:
            risk = float(self.expected_risk)
            if risk > 0:
                return float(self.expected_return) / risk
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "strategy": self.strategy,
            "underlying": self.underlying,
            "signal_type": self.signal_type,
            "signal_strength": float(self.signal_strength),
            "metadata": self.signal_metadata,
            "rationale": self.rationale,
            "expected_return": (
                float(self.expected_return) if self.expected_return else None
            ),
            "expected_risk": float(self.expected_risk) if self.expected_risk else None,
        }

    def __repr__(self) -> str:
        return (
            f"<Signal({self.strategy} {self.signal_type} {self.underlying} "
            f"strength={self.signal_strength})>"
        )


class Position(Base):
    """
    Trading positions with PnL tracking.

    Tracks open and closed positions with entry/exit prices, commissions,
    and realized/unrealized PnL calculations.

    Attributes:
        position_id: Unique position identifier (UUID)
        opened_at: Position open timestamp
        closed_at: Position close timestamp (None if open)
        updated_at: Last update timestamp
        strategy: Strategy name
        underlying: Asset symbol
        direction: 'long' or 'short'
        quantity: Position size
        entry_price: Entry price
        exit_price: Exit price (None if open)
        current_price: Current market price for unrealized PnL
        realized_pnl: Realized PnL (after close)
        unrealized_pnl: Unrealized PnL (while open)
        entry_commission: Entry commission
        exit_commission: Exit commission
        delta, gamma, vega, theta: Position Greeks (for options)
        metadata: Additional metadata as JSONB
    """

    __tablename__ = "positions"

    position_id = Column(
        UUID(), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    opened_at: datetime = Column(DateTime(timezone=True), nullable=False)
    closed_at: Optional[datetime] = Column(DateTime(timezone=True))
    updated_at: datetime = Column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now()
    )

    strategy: str = Column(String(32), nullable=False)
    underlying: str = Column(String(32), nullable=False)
    direction: str = Column(String(8), nullable=False)
    quantity: Decimal = Column(Numeric(12, 2), nullable=False)

    entry_price: Decimal = Column(Numeric(12, 4), nullable=False)
    exit_price: Optional[Decimal] = Column(Numeric(12, 4))
    current_price: Optional[Decimal] = Column(Numeric(12, 4))

    realized_pnl: Optional[Decimal] = Column(Numeric(12, 2))
    unrealized_pnl: Optional[Decimal] = Column(Numeric(12, 2))

    entry_commission: Optional[Decimal] = Column(Numeric(10, 2))
    exit_commission: Optional[Decimal] = Column(Numeric(10, 2))

    delta: Optional[Decimal] = Column(Numeric(8, 4))
    gamma: Optional[Decimal] = Column(Numeric(8, 6))
    vega: Optional[Decimal] = Column(Numeric(8, 4))
    theta: Optional[Decimal] = Column(Numeric(8, 4))

    position_metadata: Optional[Dict] = Column("metadata", JSONB)

    # Relationship to position updates
    updates = relationship(
        "PositionUpdate",
        back_populates="position",
        order_by="PositionUpdate.updated_at.desc()",
    )

    __table_args__ = (
        CheckConstraint(
            "direction IN ('long', 'short')",
            name="check_direction_valid",
        ),
        CheckConstraint(
            "closed_at IS NULL OR closed_at >= opened_at",
            name="check_close_after_open",
        ),
        CheckConstraint(
            "exit_price IS NULL OR closed_at IS NOT NULL",
            name="check_exit_price_requires_close",
        ),
        Index("idx_positions_opened_at", "opened_at"),
        Index("idx_positions_strategy", "strategy", "opened_at"),
        Index("idx_positions_underlying", "underlying", "opened_at"),
        Index(
            "idx_positions_active",
            "strategy",
            "underlying",
            postgresql_where="closed_at IS NULL",
        ),
    )

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.closed_at is None

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.direction == "long"

    @property
    def total_pnl(self) -> float:
        """Calculate total PnL (realized + unrealized)."""
        realized = float(self.realized_pnl) if self.realized_pnl else 0.0
        unrealized = float(self.unrealized_pnl) if self.unrealized_pnl else 0.0
        return realized + unrealized

    @property
    def total_commission(self) -> float:
        """Calculate total commission."""
        entry = float(self.entry_commission) if self.entry_commission else 0.0
        exit_comm = float(self.exit_commission) if self.exit_commission else 0.0
        return entry + exit_comm

    @property
    def holding_period_days(self) -> Optional[float]:
        """Calculate holding period in days."""
        if self.opened_at is None:
            return None
        end_time = self.closed_at if self.closed_at else datetime.utcnow()
        return (end_time - self.opened_at).total_seconds() / 86400

    @property
    def return_pct(self) -> Optional[float]:
        """Calculate return percentage."""
        if self.entry_price and float(self.entry_price) > 0:
            entry = float(self.entry_price) * float(self.quantity)
            if entry > 0:
                return (self.total_pnl / entry) * 100
        return None

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at given price."""
        if self.is_long:
            return (current_price - float(self.entry_price)) * float(self.quantity)
        else:
            return (float(self.entry_price) - current_price) * float(self.quantity)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position_id": str(self.position_id),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "strategy": self.strategy,
            "underlying": self.underlying,
            "direction": self.direction,
            "quantity": float(self.quantity) if self.quantity else None,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "current_price": float(self.current_price) if self.current_price else None,
            "realized_pnl": float(self.realized_pnl) if self.realized_pnl else None,
            "unrealized_pnl": (
                float(self.unrealized_pnl) if self.unrealized_pnl else None
            ),
            "total_pnl": self.total_pnl,
            "total_commission": self.total_commission,
            "is_open": self.is_open,
            "metadata": self.position_metadata,
        }

    def __repr__(self) -> str:
        status = "OPEN" if self.is_open else "CLOSED"
        return (
            f"<Position({self.strategy} {self.direction} {self.underlying} "
            f"{status} PnL=${self.total_pnl:.2f})>"
        )


class PositionUpdate(Base):
    """
    Audit trail for position changes.

    Tracks all modifications to positions for compliance and debugging.

    Attributes:
        update_id: Unique update identifier
        position_id: Reference to position
        updated_at: Update timestamp
        field_name: Name of field that was updated
        old_value: Previous value (as string)
        new_value: New value (as string)
        updated_by: User or system that made the update
    """

    __tablename__ = "position_updates"

    update_id: int = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(
        UUID(),
        ForeignKey("positions.position_id"),
        nullable=False,
    )
    updated_at: datetime = Column(DateTime(timezone=True), default=func.now())
    field_name: str = Column(String(64), nullable=False)
    old_value: Optional[str] = Column(Text)
    new_value: Optional[str] = Column(Text)
    updated_by: Optional[str] = Column(String(64))

    # Relationship back to position
    position = relationship("Position", back_populates="updates")

    __table_args__ = (
        Index("idx_position_updates_position", "position_id", "updated_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_id": self.update_id,
            "position_id": str(self.position_id),
            "updated_at": self.updated_at,
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "updated_by": self.updated_by,
        }

    def __repr__(self) -> str:
        return (
            f"<PositionUpdate({self.field_name}: {self.old_value} -> {self.new_value})>"
        )
