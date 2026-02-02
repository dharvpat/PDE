"""
Broker Gateway abstraction and adapters.

Provides:
    - BrokerGateway: Abstract interface for broker integration
    - SimulatedBroker: Paper trading broker for testing
    - BrokerAdapter: Base class for broker-specific implementations
    - Account/Position management

Reference:
    - FIX Protocol 4.4 for order messaging
    - Interactive Brokers API patterns
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .order import Fill, Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Broker connection status."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class AccountType(Enum):
    """Account types."""

    CASH = "CASH"
    MARGIN = "MARGIN"
    IRA = "IRA"
    PAPER = "PAPER"


@dataclass
class AccountInfo:
    """
    Broker account information.

    Attributes:
        account_id: Unique account identifier
        account_type: Type of account
        currency: Base currency
        buying_power: Available buying power
        cash: Cash balance
        equity: Total equity value
        margin_used: Margin currently in use
        maintenance_margin: Required maintenance margin
        day_trades_remaining: Day trades remaining (PDT rule)
    """

    account_id: str
    account_type: AccountType = AccountType.MARGIN
    currency: str = "USD"
    buying_power: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    maintenance_margin: float = 0.0
    day_trades_remaining: int = 3
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def available_margin(self) -> float:
        """Calculate available margin."""
        return max(0, self.equity - self.maintenance_margin)

    @property
    def margin_utilization(self) -> float:
        """Calculate margin utilization percentage."""
        if self.equity > 0:
            return self.margin_used / self.equity
        return 0.0


@dataclass
class BrokerPosition:
    """
    Position as reported by broker.

    Attributes:
        symbol: Asset symbol
        quantity: Position quantity (positive=long, negative=short)
        avg_cost: Average cost basis
        market_value: Current market value
        unrealized_pnl: Unrealized P&L
        realized_pnl_today: Today's realized P&L
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0


@dataclass
class BrokerQuote:
    """
    Real-time quote from broker.

    Attributes:
        symbol: Asset symbol
        bid: Best bid price
        ask: Best ask price
        bid_size: Bid size
        ask_size: Ask size
        last: Last trade price
        volume: Today's volume
    """

    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0


class BrokerGateway(ABC):
    """
    Abstract broker gateway interface.

    Defines the contract for broker integrations.
    All broker adapters must implement these methods.

    Example:
        >>> gateway = IBKRGateway(config)
        >>> gateway.connect()
        >>> broker_order_id = gateway.submit_order(order)
        >>> gateway.cancel_order(broker_order_id)
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """
        Submit order to broker.

        Args:
            order: Order to submit

        Returns:
            Broker-assigned order ID

        Raises:
            BrokerError: If submission fails
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel order at broker.

        Args:
            broker_order_id: Broker-assigned order ID

        Returns:
            True if cancel request accepted
        """
        pass

    @abstractmethod
    def modify_order(
        self,
        broker_order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """
        Modify order at broker.

        Args:
            broker_order_id: Broker-assigned order ID
            quantity: New quantity
            price: New price

        Returns:
            True if modification accepted
        """
        pass

    @abstractmethod
    def get_order_status(self, broker_order_id: str) -> Optional[OrderStatus]:
        """Get order status from broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Get all positions."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[BrokerQuote]:
        """Get current quote for symbol."""
        pass

    def register_fill_callback(
        self,
        callback: Callable[[str, Fill], None],
    ) -> None:
        """
        Register callback for fill notifications.

        Args:
            callback: Function called with (order_id, fill)
        """
        pass

    def register_status_callback(
        self,
        callback: Callable[[str, OrderStatus], None],
    ) -> None:
        """
        Register callback for order status updates.

        Args:
            callback: Function called with (order_id, status)
        """
        pass


class BrokerError(Exception):
    """Exception raised for broker errors."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.code = code


class SimulatedBroker(BrokerGateway):
    """
    Simulated broker for paper trading and testing.

    Simulates realistic order execution with:
    - Configurable latency
    - Slippage modeling
    - Partial fills
    - Order rejection scenarios

    Example:
        >>> broker = SimulatedBroker(
        ...     initial_cash=100000,
        ...     latency_ms=50,
        ...     slippage_bps=5,
        ... )
        >>> broker.connect()
        >>> order_id = broker.submit_order(order)
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        latency_ms: float = 50.0,
        slippage_bps: float = 5.0,
        fill_probability: float = 0.98,
        partial_fill_probability: float = 0.1,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_cash: Starting cash balance
            latency_ms: Simulated order latency
            slippage_bps: Average slippage in basis points
            fill_probability: Probability of order filling
            partial_fill_probability: Probability of partial fill
            commission_per_share: Commission per share
            min_commission: Minimum commission
        """
        self.initial_cash = initial_cash
        self.latency_ms = latency_ms
        self.slippage_bps = slippage_bps
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        # State
        self._connected = False
        self._account = AccountInfo(
            account_id="SIM_" + str(uuid.uuid4())[:8],
            account_type=AccountType.PAPER,
            cash=initial_cash,
            equity=initial_cash,
            buying_power=initial_cash * 4,  # 4x margin
        )
        self._positions: Dict[str, BrokerPosition] = {}
        self._orders: Dict[str, Order] = {}
        self._quotes: Dict[str, BrokerQuote] = {}

        # Callbacks
        self._fill_callbacks: List[Callable] = []
        self._status_callbacks: List[Callable] = []

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"SimulatedBroker initialized with ${initial_cash:,.0f}")

    def connect(self) -> bool:
        """Connect to simulated broker."""
        time.sleep(self.latency_ms / 1000)  # Simulate connection time
        self._connected = True
        logger.info("SimulatedBroker connected")
        return True

    def disconnect(self) -> None:
        """Disconnect from simulated broker."""
        self._connected = False
        logger.info("SimulatedBroker disconnected")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def submit_order(self, order: Order) -> str:
        """Submit order to simulated broker."""
        if not self._connected:
            raise BrokerError("Not connected to broker")

        # Simulate latency
        time.sleep(self.latency_ms / 1000)

        with self._lock:
            # Generate broker order ID
            broker_order_id = "SIM_" + str(uuid.uuid4())[:12]
            order.broker_order_id = broker_order_id
            self._orders[broker_order_id] = order

            # Simulate order processing
            self._process_order(order)

        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel order."""
        if not self._connected:
            raise BrokerError("Not connected to broker")

        time.sleep(self.latency_ms / 1000)

        with self._lock:
            if broker_order_id not in self._orders:
                return False

            order = self._orders[broker_order_id]
            if order.is_active:
                order.status = OrderStatus.CANCELLED
                self._notify_status(broker_order_id, OrderStatus.CANCELLED)
                return True

        return False

    def modify_order(
        self,
        broker_order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """Modify order."""
        if not self._connected:
            raise BrokerError("Not connected to broker")

        time.sleep(self.latency_ms / 1000)

        with self._lock:
            if broker_order_id not in self._orders:
                return False

            order = self._orders[broker_order_id]
            if not order.is_working:
                return False

            if quantity is not None:
                order.quantity = quantity
            if price is not None:
                order.price = price

            return True

    def get_order_status(self, broker_order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        with self._lock:
            if broker_order_id in self._orders:
                return self._orders[broker_order_id].status
        return None

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        with self._lock:
            self._update_account()
            return self._account

    def get_positions(self) -> List[BrokerPosition]:
        """Get all positions."""
        with self._lock:
            return list(self._positions.values())

    def get_quote(self, symbol: str) -> Optional[BrokerQuote]:
        """Get quote for symbol."""
        with self._lock:
            return self._quotes.get(symbol)

    def set_quote(self, symbol: str, quote: BrokerQuote) -> None:
        """Set quote for symbol (for simulation)."""
        with self._lock:
            self._quotes[symbol] = quote
            # Update position market value
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos.market_value = pos.quantity * quote.mid
                pos.unrealized_pnl = pos.market_value - (pos.quantity * pos.avg_cost)

    def register_fill_callback(
        self,
        callback: Callable[[str, Fill], None],
    ) -> None:
        """Register fill callback."""
        self._fill_callbacks.append(callback)

    def register_status_callback(
        self,
        callback: Callable[[str, OrderStatus], None],
    ) -> None:
        """Register status callback."""
        self._status_callbacks.append(callback)

    def _process_order(self, order: Order) -> None:
        """Process an order (simulate execution)."""
        # Check if we should reject
        if np.random.random() > self.fill_probability:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "Simulated rejection"
            self._notify_status(order.broker_order_id, OrderStatus.REJECTED)
            return

        # Get quote
        quote = self._quotes.get(order.symbol)
        if not quote:
            # Create synthetic quote
            base_price = order.price or 100.0
            quote = BrokerQuote(
                symbol=order.symbol,
                bid=base_price * 0.9999,
                ask=base_price * 1.0001,
                last=base_price,
            )

        # Determine fill price
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            base_price = quote.ask
            slippage = base_price * (self.slippage_bps / 10000)
            fill_price = base_price + slippage
        else:
            base_price = quote.bid
            slippage = base_price * (self.slippage_bps / 10000)
            fill_price = base_price - slippage

        # Check limit price
        if order.order_type == OrderType.LIMIT:
            if order.side in [OrderSide.BUY, OrderSide.COVER]:
                if fill_price > order.price:
                    # Would fill above limit, don't execute
                    order.status = OrderStatus.SUBMITTED
                    return
            else:
                if fill_price < order.price:
                    # Would fill below limit, don't execute
                    order.status = OrderStatus.SUBMITTED
                    return

        # Determine fill quantity
        if np.random.random() < self.partial_fill_probability:
            fill_qty = order.quantity * np.random.uniform(0.3, 0.9)
        else:
            fill_qty = order.quantity

        # Calculate commission
        commission = max(fill_qty * self.commission_per_share, self.min_commission)

        # Create fill
        fill = Fill(
            order_id=order.order_id,
            timestamp=datetime.utcnow(),
            quantity=fill_qty,
            price=fill_price,
            venue="SIMULATED",
            commission=commission,
        )

        # Update order
        order.add_fill(fill)

        # Update position
        self._update_position(order.symbol, fill, order.side)

        # Update account
        self._update_account_for_fill(fill, order.side)

        # Notify
        self._notify_fill(order.broker_order_id, fill)
        self._notify_status(order.broker_order_id, order.status)

    def _update_position(
        self,
        symbol: str,
        fill: Fill,
        side: OrderSide,
    ) -> None:
        """Update position after fill."""
        if symbol not in self._positions:
            self._positions[symbol] = BrokerPosition(symbol=symbol)

        pos = self._positions[symbol]

        if side in [OrderSide.BUY, OrderSide.COVER]:
            # Buying: add to position
            old_value = pos.quantity * pos.avg_cost
            new_value = fill.quantity * fill.price
            pos.quantity += fill.quantity
            if pos.quantity > 0:
                pos.avg_cost = (old_value + new_value) / pos.quantity
        else:
            # Selling: reduce position
            if pos.quantity > 0:
                # Closing long
                realized = (fill.price - pos.avg_cost) * fill.quantity
                pos.realized_pnl_today += realized
            pos.quantity -= fill.quantity
            if pos.quantity < 0:
                # Went short
                pos.avg_cost = fill.price

        pos.last_updated = datetime.utcnow()

        # Remove if flat
        if abs(pos.quantity) < 0.01:
            del self._positions[symbol]

    def _update_account_for_fill(self, fill: Fill, side: OrderSide) -> None:
        """Update account after fill."""
        trade_value = fill.quantity * fill.price

        if side in [OrderSide.BUY, OrderSide.COVER]:
            self._account.cash -= trade_value + fill.commission
        else:
            self._account.cash += trade_value - fill.commission

        self._update_account()

    def _update_account(self) -> None:
        """Recalculate account values."""
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        self._account.equity = self._account.cash + positions_value
        self._account.buying_power = self._account.equity * 4
        self._account.last_updated = datetime.utcnow()

    def _notify_fill(self, broker_order_id: str, fill: Fill) -> None:
        """Notify fill callbacks."""
        for callback in self._fill_callbacks:
            try:
                callback(broker_order_id, fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _notify_status(self, broker_order_id: str, status: OrderStatus) -> None:
        """Notify status callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(broker_order_id, status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")


class BrokerConnectionManager:
    """
    Manage broker connections with automatic reconnection.

    Handles:
    - Connection monitoring
    - Automatic reconnection
    - Heartbeat checking
    - Connection failover
    """

    def __init__(
        self,
        gateway: BrokerGateway,
        reconnect_delay_seconds: float = 5.0,
        max_reconnect_attempts: int = 10,
        heartbeat_interval_seconds: float = 30.0,
    ):
        """
        Initialize connection manager.

        Args:
            gateway: Broker gateway to manage
            reconnect_delay_seconds: Delay between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts
            heartbeat_interval_seconds: Heartbeat check interval
        """
        self.gateway = gateway
        self.reconnect_delay = reconnect_delay_seconds
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval_seconds

        self.status = ConnectionStatus.DISCONNECTED
        self._reconnect_count = 0
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        # Callbacks
        self._status_callbacks: List[Callable[[ConnectionStatus], None]] = []

    def start(self) -> bool:
        """
        Start connection and monitoring.

        Returns:
            True if initial connection successful
        """
        self.status = ConnectionStatus.CONNECTING
        self._notify_status()

        if self.gateway.connect():
            self.status = ConnectionStatus.CONNECTED
            self._reconnect_count = 0
            self._notify_status()

            # Start monitoring thread
            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_connection,
                daemon=True,
            )
            self._monitor_thread.start()

            logger.info("Broker connection established")
            return True
        else:
            self.status = ConnectionStatus.ERROR
            self._notify_status()
            return False

    def stop(self) -> None:
        """Stop connection and monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        self.gateway.disconnect()
        self.status = ConnectionStatus.DISCONNECTED
        self._notify_status()

        logger.info("Broker connection stopped")

    def register_status_callback(
        self,
        callback: Callable[[ConnectionStatus], None],
    ) -> None:
        """Register connection status callback."""
        self._status_callbacks.append(callback)

    def _monitor_connection(self) -> None:
        """Monitor connection health and reconnect if needed."""
        while not self._stop_event.is_set():
            time.sleep(self.heartbeat_interval)

            if not self.gateway.is_connected():
                logger.warning("Connection lost, attempting reconnect")
                self._reconnect()

    def _reconnect(self) -> None:
        """Attempt to reconnect."""
        self.status = ConnectionStatus.RECONNECTING
        self._notify_status()

        while (
            self._reconnect_count < self.max_reconnect_attempts
            and not self._stop_event.is_set()
        ):
            self._reconnect_count += 1
            logger.info(
                f"Reconnection attempt {self._reconnect_count}/"
                f"{self.max_reconnect_attempts}"
            )

            time.sleep(self.reconnect_delay)

            if self.gateway.connect():
                self.status = ConnectionStatus.CONNECTED
                self._reconnect_count = 0
                self._notify_status()
                logger.info("Reconnection successful")
                return

        self.status = ConnectionStatus.ERROR
        self._notify_status()
        logger.error("Max reconnection attempts reached")

    def _notify_status(self) -> None:
        """Notify status callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
