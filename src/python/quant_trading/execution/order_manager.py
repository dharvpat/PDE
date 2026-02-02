"""
Order Management System (OMS) for order lifecycle management.

Provides:
    - OrderManager: Central order lifecycle management
    - Order validation with risk checks
    - State machine transitions
    - Order persistence and history
    - Real-time order monitoring

Reference:
    FIX Protocol order state machine
    SEC Rule 606 order handling requirements
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .order import (
    Fill,
    Order,
    OrderCapacity,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

logger = logging.getLogger(__name__)


# Valid state transitions
VALID_TRANSITIONS: Dict[OrderStatus, Set[OrderStatus]] = {
    OrderStatus.PENDING: {
        OrderStatus.VALIDATING,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.VALIDATING: {
        OrderStatus.SUBMITTED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
    },
    OrderStatus.SUBMITTED: {
        OrderStatus.ACKNOWLEDGED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
    },
    OrderStatus.ACKNOWLEDGED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLING,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.REPLACED,
    },
    OrderStatus.PARTIALLY_FILLED: {
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
        OrderStatus.CANCELLING,
        OrderStatus.CANCELLED,
    },
    OrderStatus.CANCELLING: {
        OrderStatus.CANCELLED,
        OrderStatus.FILLED,  # Race condition
        OrderStatus.PARTIALLY_FILLED,  # Partial fill before cancel
    },
    OrderStatus.FILLED: set(),  # Terminal
    OrderStatus.CANCELLED: set(),  # Terminal
    OrderStatus.REJECTED: set(),  # Terminal
    OrderStatus.EXPIRED: set(),  # Terminal
    OrderStatus.REPLACED: set(),  # Terminal (replaced by new order)
    OrderStatus.SUSPENDED: {
        OrderStatus.SUBMITTED,
        OrderStatus.CANCELLED,
    },
}


@dataclass
class ValidationResult:
    """Result of order validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


@dataclass
class OrderEvent:
    """Event emitted on order state changes."""

    event_type: str  # "created", "submitted", "filled", "cancelled", etc.
    order: Order
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class OrderValidator:
    """
    Validates orders before submission.

    Checks:
        - Order field validity
        - Risk limits (position size, exposure)
        - Market hours
        - Symbol tradability
    """

    def __init__(
        self,
        max_order_size: float = 100000.0,
        max_order_value: float = 1000000.0,
        allowed_symbols: Optional[Set[str]] = None,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        check_market_hours: bool = False,
    ):
        """
        Initialize order validator.

        Args:
            max_order_size: Maximum order quantity
            max_order_value: Maximum order notional value
            allowed_symbols: Set of allowed symbols (None = all)
            market_open: Market open time
            market_close: Market close time
            check_market_hours: Whether to validate market hours
        """
        self.max_order_size = max_order_size
        self.max_order_value = max_order_value
        self.allowed_symbols = allowed_symbols
        self.market_open = market_open
        self.market_close = market_close
        self.check_market_hours = check_market_hours

    def validate(
        self,
        order: Order,
        current_positions: Optional[Dict[str, float]] = None,
        current_exposure: float = 0.0,
        max_exposure: float = float("inf"),
    ) -> ValidationResult:
        """
        Validate an order.

        Args:
            order: Order to validate
            current_positions: Current positions by symbol
            current_exposure: Current total exposure
            max_exposure: Maximum allowed exposure

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Basic field validation
        if not order.symbol:
            errors.append("Symbol is required")

        if order.quantity <= 0:
            errors.append("Quantity must be positive")

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                errors.append(f"{order.order_type.value} order requires valid price")

        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                errors.append(f"{order.order_type.value} order requires stop price")

        # Size limits
        if order.quantity > self.max_order_size:
            errors.append(
                f"Order size {order.quantity} exceeds max {self.max_order_size}"
            )

        if order.notional_value > self.max_order_value:
            errors.append(
                f"Order value ${order.notional_value:,.0f} exceeds max "
                f"${self.max_order_value:,.0f}"
            )

        # Symbol check
        if self.allowed_symbols and order.symbol not in self.allowed_symbols:
            errors.append(f"Symbol {order.symbol} not in allowed list")

        # Market hours check
        if self.check_market_hours:
            current_time = datetime.now().time()
            if not (self.market_open <= current_time <= self.market_close):
                if order.order_type == OrderType.MARKET:
                    errors.append("Market orders not allowed outside market hours")
                else:
                    warnings.append("Order submitted outside regular market hours")

        # Exposure check
        if order.price:
            order_exposure = order.quantity * order.price
            if current_exposure + order_exposure > max_exposure:
                errors.append(
                    f"Order would exceed max exposure "
                    f"(current: ${current_exposure:,.0f}, "
                    f"order: ${order_exposure:,.0f}, max: ${max_exposure:,.0f})"
                )

        # Warnings for risky orders
        if order.order_type == OrderType.MARKET and order.quantity > 10000:
            warnings.append("Large market order may have significant market impact")

        if order.time_in_force == TimeInForce.GTC:
            warnings.append("GTC orders remain active until explicitly cancelled")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class OrderManager:
    """
    Central order lifecycle management.

    Manages order creation, validation, submission, and state transitions.
    Integrates with risk management and broker gateways.

    Example:
        >>> om = OrderManager()
        >>>
        >>> # Create and submit order
        >>> order = om.create_order(
        ...     symbol="SPY",
        ...     side=OrderSide.BUY,
        ...     quantity=100,
        ...     order_type=OrderType.LIMIT,
        ...     price=450.00,
        ... )
        >>> result = om.submit_order(order)
        >>>
        >>> # Check status
        >>> status = om.get_order_status(order.order_id)
    """

    def __init__(
        self,
        validator: Optional[OrderValidator] = None,
        broker_gateway: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        persist_orders: bool = False,
    ):
        """
        Initialize order manager.

        Args:
            validator: Order validator
            broker_gateway: Broker integration
            risk_manager: Risk manager for pre-trade checks
            persist_orders: Whether to persist orders to storage
        """
        self.validator = validator or OrderValidator()
        self.broker_gateway = broker_gateway
        self.risk_manager = risk_manager
        self.persist_orders = persist_orders

        # Order storage
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)

        # Active orders (working at exchange)
        self.active_orders: Dict[str, Order] = {}

        # Event handlers
        self.event_handlers: List[Callable[[OrderEvent], None]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "orders_created": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "orders_rejected": 0,
            "total_filled_value": 0.0,
            "total_commission": 0.0,
        }

        logger.info("OrderManager initialized")

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        strategy_id: str = "default",
        **kwargs,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Asset symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            stop_price: Stop price
            time_in_force: Time in force
            strategy_id: Strategy identifier
            **kwargs: Additional order attributes

        Returns:
            Created Order object
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            strategy_id=strategy_id,
            **kwargs,
        )

        with self._lock:
            self.orders[order.order_id] = order
            self.orders_by_symbol[symbol].append(order.order_id)
            self.orders_by_strategy[strategy_id].append(order.order_id)
            self.stats["orders_created"] += 1

        self._emit_event(OrderEvent(
            event_type="created",
            order=order,
        ))

        logger.debug(f"Created order: {order}")

        return order

    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """
        Submit an order for execution.

        Validates the order and sends to broker gateway.

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Check state transition
            if not self._can_transition(order.status, OrderStatus.VALIDATING):
                return False, f"Cannot submit order in {order.status.value} state"

            # Validate
            order.status = OrderStatus.VALIDATING
            validation = self.validator.validate(order)

            if not validation.is_valid:
                order.reject("; ".join(validation.errors))
                self.stats["orders_rejected"] += 1
                self._emit_event(OrderEvent(
                    event_type="rejected",
                    order=order,
                    details={"errors": validation.errors},
                ))
                return False, f"Validation failed: {validation.errors}"

            # Log warnings
            for warning in validation.warnings:
                logger.warning(f"Order {order.order_id}: {warning}")

            # Risk check if risk manager available
            if self.risk_manager:
                risk_result = self._check_risk(order)
                if not risk_result[0]:
                    order.reject(risk_result[1])
                    self.stats["orders_rejected"] += 1
                    return False, risk_result[1]

            order.risk_check_passed = True

            # Submit to broker
            if self.broker_gateway:
                try:
                    broker_order_id = self.broker_gateway.submit_order(order)
                    order.broker_order_id = broker_order_id
                except Exception as e:
                    order.reject(str(e))
                    self.stats["orders_rejected"] += 1
                    return False, f"Broker submission failed: {e}"

            # Update state
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            order.last_updated_at = datetime.utcnow()

            self.active_orders[order.order_id] = order
            self.stats["orders_submitted"] += 1

        self._emit_event(OrderEvent(
            event_type="submitted",
            order=order,
        ))

        logger.info(f"Submitted order: {order}")

        return True, "Order submitted"

    def cancel_order(self, order_id: str, reason: str = "") -> Tuple[bool, str]:
        """
        Cancel an active order.

        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if order_id not in self.orders:
                return False, "Order not found"

            order = self.orders[order_id]

            if not order.is_active:
                return False, f"Cannot cancel order in {order.status.value} state"

            # PENDING orders can be cancelled directly
            if order.status == OrderStatus.PENDING:
                order.cancel(reason)
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.stats["orders_cancelled"] += 1
                self._emit_event(OrderEvent(
                    event_type="cancelled",
                    order=order,
                    details={"reason": reason},
                ))
                logger.info(f"Cancelled pending order {order_id}: {reason}")
                return True, "Order cancelled"

            # Check transition for non-pending orders
            if not self._can_transition(order.status, OrderStatus.CANCELLING):
                return False, f"Cannot cancel from {order.status.value} state"

            order.status = OrderStatus.CANCELLING

            # Send cancel to broker
            if self.broker_gateway and order.broker_order_id:
                try:
                    self.broker_gateway.cancel_order(order.broker_order_id)
                except Exception as e:
                    logger.error(f"Cancel request failed: {e}")
                    # Continue with local cancel

            # Update state
            order.cancel(reason)

            if order_id in self.active_orders:
                del self.active_orders[order_id]

            self.stats["orders_cancelled"] += 1

        self._emit_event(OrderEvent(
            event_type="cancelled",
            order=order,
            details={"reason": reason},
        ))

        logger.info(f"Cancelled order {order_id}: {reason}")

        return True, "Order cancelled"

    def modify_order(
        self,
        order_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Modify an active order.

        Args:
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New price (optional)

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if order_id not in self.orders:
                return False, "Order not found"

            order = self.orders[order_id]

            if not order.is_working:
                return False, f"Cannot modify order in {order.status.value} state"

            # Create modified order
            new_quantity = quantity if quantity is not None else order.remaining_quantity
            new_price = price if price is not None else order.price

            if self.broker_gateway and order.broker_order_id:
                try:
                    self.broker_gateway.modify_order(
                        order.broker_order_id,
                        quantity=new_quantity,
                        price=new_price,
                    )
                except Exception as e:
                    return False, f"Modify failed: {e}"

            # Update order
            if quantity is not None:
                order.quantity = quantity
            if price is not None:
                order.price = price

            order.last_updated_at = datetime.utcnow()

        self._emit_event(OrderEvent(
            event_type="modified",
            order=order,
            details={"quantity": quantity, "price": price},
        ))

        return True, "Order modified"

    def process_fill(
        self,
        order_id: str,
        fill: Fill,
    ) -> bool:
        """
        Process a fill for an order.

        Args:
            order_id: Order ID
            fill: Fill record

        Returns:
            True if fill processed successfully
        """
        with self._lock:
            if order_id not in self.orders:
                logger.warning(f"Fill for unknown order: {order_id}")
                return False

            order = self.orders[order_id]

            # Add fill
            order.add_fill(fill)

            # Update statistics
            self.stats["total_filled_value"] += fill.notional_value
            self.stats["total_commission"] += fill.commission + fill.fees

            if order.is_filled:
                self.stats["orders_filled"] += 1
                if order_id in self.active_orders:
                    del self.active_orders[order_id]

        self._emit_event(OrderEvent(
            event_type="fill" if not order.is_filled else "filled",
            order=order,
            details={"fill": fill.to_dict()},
        ))

        logger.debug(
            f"Fill: {fill.quantity}@{fill.price} for order {order_id}, "
            f"status={order.status.value}"
        )

        return True

    def acknowledge_order(self, order_id: str, broker_order_id: str) -> bool:
        """
        Mark order as acknowledged by broker.

        Args:
            order_id: Order ID
            broker_order_id: Broker-assigned order ID

        Returns:
            True if acknowledgment processed
        """
        with self._lock:
            if order_id not in self.orders:
                return False

            order = self.orders[order_id]

            if not self._can_transition(order.status, OrderStatus.ACKNOWLEDGED):
                return False

            order.status = OrderStatus.ACKNOWLEDGED
            order.broker_order_id = broker_order_id
            order.acknowledged_at = datetime.utcnow()
            order.last_updated_at = datetime.utcnow()

        self._emit_event(OrderEvent(
            event_type="acknowledged",
            order=order,
        ))

        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status."""
        order = self.orders.get(order_id)
        return order.status if order else None

    def get_active_orders(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
    ) -> List[Order]:
        """
        Get all active orders.

        Args:
            symbol: Filter by symbol
            strategy_id: Filter by strategy

        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        if strategy_id:
            orders = [o for o in orders if o.strategy_id == strategy_id]

        return orders

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        order_ids = self.orders_by_symbol.get(symbol, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """Get all orders for a strategy."""
        order_ids = self.orders_by_strategy.get(strategy_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_orders_by_status(self, statuses: List[OrderStatus]) -> List[Order]:
        """
        Get all orders with the specified statuses.

        Args:
            statuses: List of statuses to filter by

        Returns:
            List of orders matching any of the statuses
        """
        return [o for o in self.orders.values() if o.status in statuses]

    def get_filled_orders(
        self,
        since: Optional[datetime] = None,
        symbol: Optional[str] = None,
    ) -> List[Order]:
        """
        Get filled orders.

        Args:
            since: Filter by fill time
            symbol: Filter by symbol

        Returns:
            List of filled orders
        """
        filled = [o for o in self.orders.values() if o.is_filled]

        if since:
            filled = [o for o in filled if o.filled_at and o.filled_at >= since]

        if symbol:
            filled = [o for o in filled if o.symbol == symbol]

        return filled

    def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None,
        reason: str = "Bulk cancel",
    ) -> int:
        """
        Cancel all active orders.

        Args:
            symbol: Filter by symbol
            strategy_id: Filter by strategy
            reason: Cancellation reason

        Returns:
            Number of orders cancelled
        """
        orders_to_cancel = self.get_active_orders(symbol, strategy_id)
        cancelled = 0

        for order in orders_to_cancel:
            success, _ = self.cancel_order(order.order_id, reason)
            if success:
                cancelled += 1

        logger.info(f"Cancelled {cancelled} orders: {reason}")
        return cancelled

    def register_event_handler(
        self,
        handler: Callable[[OrderEvent], None],
    ) -> None:
        """Register an event handler."""
        self.event_handlers.append(handler)

    def get_statistics(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        with self._lock:
            return {
                **self.stats,
                "active_orders": len(self.active_orders),
                "total_orders": len(self.orders),
                "orders_by_status": self._count_by_status(),
            }

    def _can_transition(
        self,
        from_status: OrderStatus,
        to_status: OrderStatus,
    ) -> bool:
        """Check if state transition is valid."""
        valid_targets = VALID_TRANSITIONS.get(from_status, set())
        return to_status in valid_targets

    def _check_risk(self, order: Order) -> Tuple[bool, str]:
        """
        Check order against risk limits.

        Args:
            order: Order to check

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.risk_manager:
            return True, "ok"

        # Integration with risk manager
        try:
            result = self.risk_manager.check_position_allowed(
                asset_id=order.symbol,
                position_size=order.quantity,
                current_price=order.price or 0.0,
            )
            return result.allowed, result.rationale
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return False, f"Risk check error: {e}"

    def _emit_event(self, event: OrderEvent) -> None:
        """Emit order event to handlers."""
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    def _count_by_status(self) -> Dict[str, int]:
        """Count orders by status."""
        counts = defaultdict(int)
        for order in self.orders.values():
            counts[order.status.value] += 1
        return dict(counts)
