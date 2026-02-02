"""
Order classes and enums for the Order Management System.

Provides:
    - OrderStatus: Order lifecycle states
    - OrderType: Types of orders (market, limit, etc.)
    - OrderSide: Buy/sell direction
    - TimeInForce: Order validity duration
    - Order: Full order object with lifecycle tracking
    - Fill: Individual fill record

Reference:
    FIX Protocol 4.4 for order message specification
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order lifecycle states (FIX-compatible)."""

    PENDING = "PENDING"  # Created but not submitted
    VALIDATING = "VALIDATING"  # Undergoing pre-trade checks
    SUBMITTED = "SUBMITTED"  # Sent to broker/exchange
    ACKNOWLEDGED = "ACKNOWLEDGED"  # Broker acknowledged receipt
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partial execution
    FILLED = "FILLED"  # Fully executed
    CANCELLING = "CANCELLING"  # Cancel request pending
    CANCELLED = "CANCELLED"  # Successfully cancelled
    REJECTED = "REJECTED"  # Rejected by broker/exchange
    EXPIRED = "EXPIRED"  # Time-in-force expired
    REPLACED = "REPLACED"  # Order was modified/replaced
    SUSPENDED = "SUSPENDED"  # Order is suspended


class OrderType(Enum):
    """Order types."""

    MARKET = "MARKET"  # Execute at current market price
    LIMIT = "LIMIT"  # Execute at limit price or better
    STOP = "STOP"  # Becomes market when stop price hit
    STOP_LIMIT = "STOP_LIMIT"  # Becomes limit when stop hit
    TRAILING_STOP = "TRAILING_STOP"  # Stop that trails price
    TRAILING_STOP_LIMIT = "TRAILING_STOP_LIMIT"  # Trailing with limit
    MARKET_ON_CLOSE = "MOC"  # Execute at market close
    LIMIT_ON_CLOSE = "LOC"  # Limit order at close
    ICEBERG = "ICEBERG"  # Display only portion
    TWAP = "TWAP"  # Time-weighted average price
    VWAP = "VWAP"  # Volume-weighted average price
    PEG = "PEG"  # Pegged to market


class OrderSide(Enum):
    """Order side/direction."""

    BUY = "BUY"  # Long entry
    SELL = "SELL"  # Long exit
    SHORT = "SHORT"  # Short entry (borrow and sell)
    COVER = "COVER"  # Short exit (buy to cover)


class TimeInForce(Enum):
    """Time in force specifications."""

    DAY = "DAY"  # Valid for trading day only
    GTC = "GTC"  # Good till cancelled
    IOC = "IOC"  # Immediate or cancel (partial OK)
    FOK = "FOK"  # Fill or kill (all or nothing)
    GTD = "GTD"  # Good till date
    OPG = "OPG"  # At the opening
    CLS = "CLS"  # At the close
    GTX = "GTX"  # Good till crossing (for extended hours)


class OrderCapacity(Enum):
    """Order capacity (regulatory requirement)."""

    AGENCY = "AGENCY"  # Acting as agent for client
    PRINCIPAL = "PRINCIPAL"  # Trading for own account
    RISKLESS_PRINCIPAL = "RISKLESS_PRINCIPAL"  # Back-to-back


@dataclass
class Fill:
    """
    Individual fill/execution record.

    Attributes:
        fill_id: Unique fill identifier
        order_id: Parent order ID
        timestamp: Fill timestamp
        quantity: Filled quantity
        price: Execution price
        venue: Execution venue
        commission: Commission for this fill
        fees: Exchange/regulatory fees
        liquidity: Added/removed liquidity indicator
        contra_broker: Contra party broker ID
    """

    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    quantity: float = 0.0
    price: float = 0.0
    venue: str = ""
    commission: float = 0.0
    fees: float = 0.0
    liquidity: str = ""  # "ADD" or "REMOVE"
    contra_broker: str = ""

    @property
    def notional_value(self) -> float:
        """Calculate notional value of fill."""
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        """Total execution cost (commission + fees)."""
        return self.commission + self.fees

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "quantity": self.quantity,
            "price": self.price,
            "venue": self.venue,
            "commission": self.commission,
            "fees": self.fees,
            "liquidity": self.liquidity,
            "notional_value": self.notional_value,
        }


@dataclass
class Order:
    """
    Order object tracking full lifecycle.

    Supports parent-child relationships for order splitting and
    tracks all fills for partial execution.

    Example:
        >>> order = Order(
        ...     symbol="SPY",
        ...     side=OrderSide.BUY,
        ...     order_type=OrderType.LIMIT,
        ...     quantity=1000,
        ...     price=450.00,
        ...     time_in_force=TimeInForce.DAY,
        ... )
        >>> print(f"Order {order.order_id}: {order.side.value} {order.quantity} {order.symbol}")
    """

    # Identity
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_order_id: str = ""  # Client-assigned ID
    broker_order_id: Optional[str] = None  # Broker-assigned ID
    exchange_order_id: Optional[str] = None  # Exchange-assigned ID

    # Order specification
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None  # Limit price
    stop_price: Optional[float] = None  # Stop/trigger price
    trailing_amount: Optional[float] = None  # For trailing stops
    trailing_percent: Optional[float] = None  # Trailing % instead of amount
    display_quantity: Optional[float] = None  # For iceberg orders

    # Validity
    time_in_force: TimeInForce = TimeInForce.DAY
    expire_time: Optional[datetime] = None  # For GTD orders

    # Status
    status: OrderStatus = OrderStatus.PENDING

    # Relationships
    strategy_id: str = "default"
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    linked_order_ids: List[str] = field(default_factory=list)  # OCO, bracket

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    first_fill_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    last_updated_at: datetime = field(default_factory=datetime.utcnow)

    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)

    # Costs
    commission: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0

    # Routing
    venue: Optional[str] = None  # Target venue
    routing_strategy: Optional[str] = None  # e.g., "SMART", "VWAP"
    order_capacity: OrderCapacity = OrderCapacity.PRINCIPAL

    # Risk/compliance
    account_id: str = "default"
    risk_check_passed: bool = False
    compliance_id: Optional[str] = None

    # Rejection
    reject_reason: Optional[str] = None
    reject_code: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.client_order_id:
            self.client_order_id = self.order_id

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.status == OrderStatus.PARTIALLY_FILLED

    @property
    def is_active(self) -> bool:
        """Check if order is active (can still execute)."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.VALIDATING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def is_working(self) -> bool:
        """Check if order is working at exchange."""
        return self.status in [
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return max(0.0, self.quantity - self.filled_quantity)

    @property
    def fill_rate(self) -> float:
        """Get fill rate as percentage."""
        if self.quantity > 0:
            return (self.filled_quantity / self.quantity) * 100
        return 0.0

    @property
    def notional_value(self) -> float:
        """Estimate notional value of order."""
        price = self.price or self.avg_fill_price or 0.0
        return self.quantity * price

    @property
    def filled_notional(self) -> float:
        """Notional value of filled portion."""
        return self.filled_quantity * self.avg_fill_price

    @property
    def total_cost(self) -> float:
        """Total execution cost."""
        return self.commission + self.fees + abs(self.slippage)

    @property
    def cost_per_share(self) -> float:
        """Cost per share executed."""
        if self.filled_quantity > 0:
            return self.total_cost / self.filled_quantity
        return 0.0

    @property
    def has_children(self) -> bool:
        """Check if order has child orders."""
        return len(self.child_order_ids) > 0

    @property
    def is_child(self) -> bool:
        """Check if order is a child order."""
        return self.parent_order_id is not None

    def add_fill(self, fill: Fill) -> None:
        """
        Add a fill to the order.

        Updates filled quantity, average price, and status.

        Args:
            fill: Fill record to add
        """
        fill.order_id = self.order_id
        self.fills.append(fill)

        # Update filled quantity
        self.filled_quantity += fill.quantity

        # Update average fill price (volume-weighted)
        total_value = sum(f.quantity * f.price for f in self.fills)
        if self.filled_quantity > 0:
            self.avg_fill_price = total_value / self.filled_quantity

        # Update costs
        self.commission += fill.commission
        self.fees += fill.fees

        # Update timestamps
        if self.first_fill_at is None:
            self.first_fill_at = fill.timestamp

        self.last_updated_at = datetime.utcnow()

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

        logger.debug(
            f"Order {self.order_id}: fill {fill.quantity}@{fill.price}, "
            f"filled {self.filled_quantity}/{self.quantity}"
        )

    def cancel(self, reason: str = "") -> bool:
        """
        Mark order as cancelled.

        Args:
            reason: Cancellation reason

        Returns:
            True if order was cancellable
        """
        if not self.is_active:
            return False

        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        self.last_updated_at = datetime.utcnow()
        if reason:
            self.notes = f"Cancelled: {reason}"

        logger.info(f"Order {self.order_id} cancelled: {reason}")
        return True

    def reject(self, reason: str, code: Optional[str] = None) -> None:
        """
        Mark order as rejected.

        Args:
            reason: Rejection reason
            code: Rejection code
        """
        self.status = OrderStatus.REJECTED
        self.reject_reason = reason
        self.reject_code = code
        self.last_updated_at = datetime.utcnow()

        logger.warning(f"Order {self.order_id} rejected: {reason}")

    def add_child(self, child_order_id: str) -> None:
        """Add child order ID."""
        if child_order_id not in self.child_order_ids:
            self.child_order_ids.append(child_order_id)

    def calculate_slippage(self, benchmark_price: float) -> float:
        """
        Calculate slippage vs benchmark price.

        Args:
            benchmark_price: Benchmark price (e.g., arrival price)

        Returns:
            Slippage in dollars (positive = unfavorable)
        """
        if self.filled_quantity == 0:
            return 0.0

        if self.side in [OrderSide.BUY, OrderSide.COVER]:
            # For buys, paying more than benchmark is unfavorable
            slippage = (self.avg_fill_price - benchmark_price) * self.filled_quantity
        else:
            # For sells, receiving less than benchmark is unfavorable
            slippage = (benchmark_price - self.avg_fill_price) * self.filled_quantity

        self.slippage = slippage
        return slippage

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "strategy_id": self.strategy_id,
            "parent_order_id": self.parent_order_id,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "remaining_quantity": self.remaining_quantity,
            "commission": self.commission,
            "fees": self.fees,
            "slippage": self.slippage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "reject_reason": self.reject_reason,
            "venue": self.venue,
            "n_fills": len(self.fills),
        }

    def clone(self, new_quantity: Optional[float] = None) -> "Order":
        """
        Create a copy of this order with a new ID.

        Args:
            new_quantity: Override quantity (optional)

        Returns:
            New Order object
        """
        return Order(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            quantity=new_quantity if new_quantity is not None else self.quantity,
            price=self.price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            strategy_id=self.strategy_id,
            parent_order_id=self.order_id,
            venue=self.venue,
            account_id=self.account_id,
            metadata=self.metadata.copy(),
        )

    def __repr__(self) -> str:
        """String representation."""
        price_str = f"@{self.price}" if self.price else ""
        return (
            f"Order({self.order_id[:8]}... {self.side.value} {self.quantity} "
            f"{self.symbol} {self.order_type.value}{price_str} [{self.status.value}])"
        )
