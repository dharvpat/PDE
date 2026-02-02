"""
Emergency Controls Module

Implements critical safety mechanisms for algorithmic trading:
- Kill switch (cancel all orders)
- Position flattening
- Market on close orders
- After-hours trading controls
- Circuit breaker integration

These controls are essential for regulatory compliance and risk management
per Fed SR 11-7 and algorithmic trading best practices.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from .order import Order, OrderSide, OrderStatus, OrderType, TimeInForce


logger = logging.getLogger(__name__)


class EmergencyState(Enum):
    """System emergency states."""
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    HALT_NEW_ORDERS = "HALT_NEW_ORDERS"
    CANCEL_PENDING = "CANCEL_PENDING"
    FLATTEN_POSITIONS = "FLATTEN_POSITIONS"
    FULL_STOP = "FULL_STOP"


class TriggerType(Enum):
    """Types of emergency triggers."""
    MANUAL = "MANUAL"
    DRAWDOWN = "DRAWDOWN"
    LOSS_LIMIT = "LOSS_LIMIT"
    VOLATILITY = "VOLATILITY"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    CONNECTION_LOSS = "CONNECTION_LOSS"
    ERROR_RATE = "ERROR_RATE"
    REGULATORY = "REGULATORY"
    TIME_BASED = "TIME_BASED"


@dataclass
class EmergencyEvent:
    """Record of an emergency control activation."""
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    trigger_type: TriggerType = TriggerType.MANUAL
    previous_state: EmergencyState = EmergencyState.NORMAL
    new_state: EmergencyState = EmergencyState.NORMAL
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    orders_cancelled: int = 0
    positions_flattened: int = 0
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""


@dataclass
class TradingHours:
    """Trading hours configuration."""
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    pre_market_open: time = time(4, 0)
    after_hours_close: time = time(20, 0)
    allow_pre_market: bool = True
    allow_after_hours: bool = True
    trading_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri


@dataclass
class CircuitBreakerLevel:
    """Circuit breaker threshold level."""
    name: str = ""
    threshold_pct: float = 0.0
    halt_duration_minutes: int = 0
    action: EmergencyState = EmergencyState.HALT_NEW_ORDERS


class KillSwitch:
    """
    Emergency kill switch for immediate order cancellation.

    Provides immediate cessation of all trading activity with
    full audit trail and notification capabilities.
    """

    def __init__(
        self,
        order_manager: Any = None,
        broker_gateway: Any = None,
        notification_callbacks: Optional[List[Callable]] = None
    ):
        self.order_manager = order_manager
        self.broker_gateway = broker_gateway
        self.notification_callbacks = notification_callbacks or []

        self.is_engaged = False
        self.engaged_at: Optional[datetime] = None
        self.engaged_by: str = ""
        self.engage_reason: str = ""
        self.cancellation_results: List[Dict[str, Any]] = []

        self._event_history: List[EmergencyEvent] = []

    def engage(
        self,
        reason: str = "Manual kill switch activation",
        triggered_by: str = "system",
        cancel_all_orders: bool = True
    ) -> EmergencyEvent:
        """
        Engage the kill switch.

        Args:
            reason: Reason for activation
            triggered_by: User or system that triggered
            cancel_all_orders: Whether to cancel all pending orders

        Returns:
            EmergencyEvent recording the activation
        """
        if self.is_engaged:
            logger.warning("Kill switch already engaged")
            return self._event_history[-1] if self._event_history else None

        self.is_engaged = True
        self.engaged_at = datetime.now()
        self.engaged_by = triggered_by
        self.engage_reason = reason

        event = EmergencyEvent(
            event_id=f"KS-{self.engaged_at.strftime('%Y%m%d%H%M%S')}",
            timestamp=self.engaged_at,
            trigger_type=TriggerType.MANUAL,
            previous_state=EmergencyState.NORMAL,
            new_state=EmergencyState.FULL_STOP,
            reason=reason,
            details={"triggered_by": triggered_by}
        )

        logger.critical(
            f"KILL SWITCH ENGAGED by {triggered_by}: {reason}"
        )

        orders_cancelled = 0
        if cancel_all_orders:
            orders_cancelled = self._cancel_all_orders()
            event.orders_cancelled = orders_cancelled

        self._event_history.append(event)
        self._send_notifications(event)

        return event

    def disengage(
        self,
        authorized_by: str,
        notes: str = ""
    ) -> bool:
        """
        Disengage the kill switch and resume normal operations.

        Args:
            authorized_by: User authorizing the disengagement
            notes: Resolution notes

        Returns:
            True if successfully disengaged
        """
        if not self.is_engaged:
            logger.warning("Kill switch is not engaged")
            return False

        self.is_engaged = False

        if self._event_history:
            self._event_history[-1].resolved_at = datetime.now()
            self._event_history[-1].resolution_notes = notes

        logger.info(
            f"Kill switch disengaged by {authorized_by}: {notes}"
        )

        return True

    def _cancel_all_orders(self) -> int:
        """Cancel all pending orders. Returns count of cancelled orders."""
        cancelled = 0
        self.cancellation_results = []

        if not self.order_manager:
            logger.warning("No order manager configured for kill switch")
            return 0

        pending_orders = self.order_manager.get_orders_by_status([
            OrderStatus.PENDING,
            OrderStatus.VALIDATING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED
        ])

        for order in pending_orders:
            result = {"order_id": order.order_id, "symbol": order.symbol}

            success, message = self.order_manager.cancel_order(order.order_id)
            result["success"] = success
            result["message"] = message

            if success:
                cancelled += 1

            self.cancellation_results.append(result)

        logger.info(f"Kill switch cancelled {cancelled} orders")
        return cancelled

    def _send_notifications(self, event: EmergencyEvent) -> None:
        """Send notifications to all registered callbacks."""
        for callback in self.notification_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current kill switch status."""
        return {
            "is_engaged": self.is_engaged,
            "engaged_at": self.engaged_at.isoformat() if self.engaged_at else None,
            "engaged_by": self.engaged_by,
            "reason": self.engage_reason,
            "event_count": len(self._event_history)
        }

    def get_event_history(self) -> List[EmergencyEvent]:
        """Get history of all kill switch events."""
        return self._event_history.copy()


class PositionFlattener:
    """
    Emergency position flattening system.

    Closes all open positions in an orderly manner with
    configurable urgency levels.
    """

    def __init__(
        self,
        order_manager: Any = None,
        broker_gateway: Any = None
    ):
        self.order_manager = order_manager
        self.broker_gateway = broker_gateway
        self.is_flattening = False
        self.flatten_orders: List[Order] = []

    def flatten_all_positions(
        self,
        urgency: str = "normal",
        use_market_orders: bool = False
    ) -> List[Order]:
        """
        Flatten all open positions.

        Args:
            urgency: "normal", "urgent", or "immediate"
            use_market_orders: Force market orders (immediate execution)

        Returns:
            List of flattening orders created
        """
        self.is_flattening = True
        self.flatten_orders = []

        if not self.broker_gateway:
            logger.error("No broker gateway configured for position flattening")
            return []

        positions = self.broker_gateway.get_positions()

        for position in positions:
            if abs(position.quantity) < 0.0001:
                continue

            order = self._create_flatten_order(
                position,
                urgency,
                use_market_orders
            )

            if order:
                self.flatten_orders.append(order)

        logger.info(
            f"Created {len(self.flatten_orders)} flattening orders"
        )

        return self.flatten_orders

    def _create_flatten_order(
        self,
        position: Any,
        urgency: str,
        use_market_orders: bool
    ) -> Optional[Order]:
        """Create an order to flatten a position."""
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        quantity = abs(position.quantity)

        if use_market_orders or urgency == "immediate":
            order_type = OrderType.MARKET
            limit_price = None
        else:
            order_type = OrderType.LIMIT
            # Calculate current price from market value
            current_price = (
                position.market_value / abs(position.quantity)
                if position.quantity != 0 else position.avg_cost
            )
            if side == OrderSide.SELL:
                limit_price = current_price * 0.99
            else:
                limit_price = current_price * 1.01

        tif = TimeInForce.IOC if urgency == "immediate" else TimeInForce.DAY

        if not self.order_manager:
            return None

        order = self.order_manager.create_order(
            symbol=position.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=limit_price,
            time_in_force=tif
        )

        order.notes = f"Emergency flatten: {urgency}"

        return order

    def flatten_symbol(
        self,
        symbol: str,
        use_market_order: bool = True
    ) -> Optional[Order]:
        """Flatten position in a specific symbol."""
        if not self.broker_gateway:
            return None

        positions = self.broker_gateway.get_positions()

        for position in positions:
            if position.symbol == symbol:
                return self._create_flatten_order(
                    position,
                    "immediate" if use_market_order else "normal",
                    use_market_order
                )

        return None

    def create_moc_orders(self) -> List[Order]:
        """
        Create Market-on-Close orders for all positions.

        Used for end-of-day position flattening.
        """
        moc_orders = []

        if not self.broker_gateway:
            return moc_orders

        positions = self.broker_gateway.get_positions()

        for position in positions:
            if abs(position.quantity) < 0.0001:
                continue

            side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY

            if self.order_manager:
                order = self.order_manager.create_order(
                    symbol=position.symbol,
                    side=side,
                    quantity=abs(position.quantity),
                    order_type=OrderType.MARKET,
                    time_in_force=TimeInForce.MOC
                )
                order.notes = "Market-on-Close flatten"
                moc_orders.append(order)

        return moc_orders


class TradingHoursController:
    """
    Controls trading based on market hours.

    Enforces pre-market, regular, and after-hours trading restrictions.
    """

    def __init__(self, trading_hours: Optional[TradingHours] = None):
        self.trading_hours = trading_hours or TradingHours()
        self.override_enabled = False
        self.override_reason: str = ""

    def is_trading_allowed(self, now: Optional[datetime] = None) -> bool:
        """
        Check if trading is allowed at the given time.

        Args:
            now: Time to check (defaults to current time)

        Returns:
            True if trading is allowed
        """
        if self.override_enabled:
            return True

        now = now or datetime.now()

        if now.weekday() not in self.trading_hours.trading_days:
            return False

        current_time = now.time()

        if self.trading_hours.allow_pre_market:
            if (self.trading_hours.pre_market_open <= current_time <
                self.trading_hours.market_open):
                return True

        if (self.trading_hours.market_open <= current_time <
            self.trading_hours.market_close):
            return True

        if self.trading_hours.allow_after_hours:
            if (self.trading_hours.market_close <= current_time <
                self.trading_hours.after_hours_close):
                return True

        return False

    def get_session_type(self, now: Optional[datetime] = None) -> str:
        """Get the current trading session type."""
        now = now or datetime.now()
        current_time = now.time()

        if now.weekday() not in self.trading_hours.trading_days:
            return "closed"

        if current_time < self.trading_hours.pre_market_open:
            return "closed"
        elif current_time < self.trading_hours.market_open:
            return "pre_market"
        elif current_time < self.trading_hours.market_close:
            return "regular"
        elif current_time < self.trading_hours.after_hours_close:
            return "after_hours"
        else:
            return "closed"

    def time_until_market_open(self, now: Optional[datetime] = None) -> int:
        """Get seconds until market open."""
        now = now or datetime.now()
        current_time = now.time()

        if current_time >= self.trading_hours.market_open:
            return 0

        open_dt = datetime.combine(now.date(), self.trading_hours.market_open)
        return int((open_dt - now).total_seconds())

    def time_until_market_close(self, now: Optional[datetime] = None) -> int:
        """Get seconds until market close."""
        now = now or datetime.now()
        current_time = now.time()

        if current_time >= self.trading_hours.market_close:
            return 0

        close_dt = datetime.combine(now.date(), self.trading_hours.market_close)
        return int((close_dt - now).total_seconds())

    def enable_override(self, reason: str) -> None:
        """Enable trading hours override."""
        self.override_enabled = True
        self.override_reason = reason
        logger.warning(f"Trading hours override enabled: {reason}")

    def disable_override(self) -> None:
        """Disable trading hours override."""
        self.override_enabled = False
        self.override_reason = ""
        logger.info("Trading hours override disabled")


class CircuitBreaker:
    """
    Market circuit breaker integration.

    Monitors market conditions and triggers trading halts
    based on configurable thresholds.
    """

    DEFAULT_LEVELS = [
        CircuitBreakerLevel("Level 1", 7.0, 15, EmergencyState.HALT_NEW_ORDERS),
        CircuitBreakerLevel("Level 2", 13.0, 15, EmergencyState.HALT_NEW_ORDERS),
        CircuitBreakerLevel("Level 3", 20.0, 0, EmergencyState.FULL_STOP),
    ]

    def __init__(
        self,
        levels: Optional[List[CircuitBreakerLevel]] = None,
        reference_price: float = 0.0
    ):
        self.levels = levels or self.DEFAULT_LEVELS
        self.reference_price = reference_price
        self.is_triggered = False
        self.triggered_level: Optional[CircuitBreakerLevel] = None
        self.triggered_at: Optional[datetime] = None
        self.resume_at: Optional[datetime] = None

    def update_reference_price(self, price: float) -> None:
        """Update the reference price (typically previous close)."""
        self.reference_price = price

    def check_price(self, current_price: float) -> Optional[CircuitBreakerLevel]:
        """
        Check if current price triggers any circuit breaker level.

        Args:
            current_price: Current market price

        Returns:
            Triggered level or None
        """
        if self.reference_price <= 0:
            return None

        change_pct = abs(
            (current_price - self.reference_price) / self.reference_price * 100
        )

        triggered = None
        for level in self.levels:
            if change_pct >= level.threshold_pct:
                triggered = level

        return triggered

    def trigger(self, level: CircuitBreakerLevel) -> EmergencyEvent:
        """Trigger a circuit breaker level."""
        self.is_triggered = True
        self.triggered_level = level
        self.triggered_at = datetime.now()

        if level.halt_duration_minutes > 0:
            from datetime import timedelta
            self.resume_at = self.triggered_at + timedelta(
                minutes=level.halt_duration_minutes
            )
        else:
            self.resume_at = None

        event = EmergencyEvent(
            event_id=f"CB-{self.triggered_at.strftime('%Y%m%d%H%M%S')}",
            timestamp=self.triggered_at,
            trigger_type=TriggerType.CIRCUIT_BREAKER,
            previous_state=EmergencyState.NORMAL,
            new_state=level.action,
            reason=f"Circuit breaker {level.name} triggered",
            details={
                "level": level.name,
                "threshold_pct": level.threshold_pct,
                "halt_duration_minutes": level.halt_duration_minutes
            }
        )

        logger.critical(
            f"CIRCUIT BREAKER {level.name} TRIGGERED: "
            f"{level.threshold_pct}% threshold breached"
        )

        return event

    def check_resume(self, now: Optional[datetime] = None) -> bool:
        """Check if trading can resume after circuit breaker."""
        if not self.is_triggered:
            return True

        if self.resume_at is None:
            return False

        now = now or datetime.now()

        if now >= self.resume_at:
            self.reset()
            return True

        return False

    def reset(self) -> None:
        """Reset circuit breaker state."""
        self.is_triggered = False
        self.triggered_level = None
        self.triggered_at = None
        self.resume_at = None
        logger.info("Circuit breaker reset")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "is_triggered": self.is_triggered,
            "triggered_level": self.triggered_level.name if self.triggered_level else None,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "resume_at": self.resume_at.isoformat() if self.resume_at else None,
            "reference_price": self.reference_price
        }


class EmergencyController:
    """
    Central emergency control system.

    Coordinates all emergency mechanisms and maintains
    overall system state.
    """

    def __init__(
        self,
        order_manager: Any = None,
        broker_gateway: Any = None,
        trading_hours: Optional[TradingHours] = None
    ):
        self.order_manager = order_manager
        self.broker_gateway = broker_gateway

        self.kill_switch = KillSwitch(order_manager, broker_gateway)
        self.position_flattener = PositionFlattener(order_manager, broker_gateway)
        self.hours_controller = TradingHoursController(trading_hours)
        self.circuit_breaker = CircuitBreaker()

        self.current_state = EmergencyState.NORMAL
        self.state_history: List[EmergencyEvent] = []

        self.drawdown_limit_pct: float = 25.0
        self.daily_loss_limit: float = 0.0
        self.error_rate_threshold: float = 0.10

        self._error_count = 0
        self._order_count = 0
        self._daily_pnl: float = 0.0
        self._peak_equity: float = 0.0

    def check_and_update_state(
        self,
        current_equity: float,
        current_price: float,
        daily_pnl: float
    ) -> EmergencyState:
        """
        Check all conditions and update emergency state.

        Args:
            current_equity: Current portfolio equity
            current_price: Current market reference price
            daily_pnl: Today's P&L

        Returns:
            Updated emergency state
        """
        self._daily_pnl = daily_pnl

        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        cb_level = self.circuit_breaker.check_price(current_price)
        if cb_level:
            event = self.circuit_breaker.trigger(cb_level)
            self._update_state(cb_level.action, event)
            return self.current_state

        if self._peak_equity > 0:
            drawdown_pct = (
                (self._peak_equity - current_equity) / self._peak_equity * 100
            )
            if drawdown_pct >= self.drawdown_limit_pct:
                self._trigger_drawdown_limit(drawdown_pct)
                return self.current_state

        if self.daily_loss_limit > 0 and daily_pnl < -self.daily_loss_limit:
            self._trigger_loss_limit(daily_pnl)
            return self.current_state

        if self._order_count > 10:
            error_rate = self._error_count / self._order_count
            if error_rate >= self.error_rate_threshold:
                self._trigger_error_rate(error_rate)
                return self.current_state

        return self.current_state

    def _update_state(
        self,
        new_state: EmergencyState,
        event: EmergencyEvent
    ) -> None:
        """Update the current emergency state."""
        event.previous_state = self.current_state
        event.new_state = new_state
        self.current_state = new_state
        self.state_history.append(event)

    def _trigger_drawdown_limit(self, drawdown_pct: float) -> None:
        """Handle drawdown limit breach."""
        event = EmergencyEvent(
            event_id=f"DD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=TriggerType.DRAWDOWN,
            reason=f"Drawdown limit breached: {drawdown_pct:.1f}%",
            details={"drawdown_pct": drawdown_pct}
        )

        logger.critical(
            f"DRAWDOWN LIMIT BREACHED: {drawdown_pct:.1f}% >= {self.drawdown_limit_pct}%"
        )

        self._update_state(EmergencyState.HALT_NEW_ORDERS, event)

    def _trigger_loss_limit(self, daily_pnl: float) -> None:
        """Handle daily loss limit breach."""
        event = EmergencyEvent(
            event_id=f"LL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=TriggerType.LOSS_LIMIT,
            reason=f"Daily loss limit breached: ${daily_pnl:,.2f}",
            details={"daily_pnl": daily_pnl, "limit": self.daily_loss_limit}
        )

        logger.critical(
            f"DAILY LOSS LIMIT BREACHED: ${daily_pnl:,.2f} < -${self.daily_loss_limit:,.2f}"
        )

        self._update_state(EmergencyState.HALT_NEW_ORDERS, event)

    def _trigger_error_rate(self, error_rate: float) -> None:
        """Handle high error rate."""
        event = EmergencyEvent(
            event_id=f"ER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=TriggerType.ERROR_RATE,
            reason=f"High error rate: {error_rate:.1%}",
            details={
                "error_rate": error_rate,
                "error_count": self._error_count,
                "order_count": self._order_count
            }
        )

        logger.warning(
            f"HIGH ERROR RATE: {error_rate:.1%} >= {self.error_rate_threshold:.1%}"
        )

        self._update_state(EmergencyState.CAUTION, event)

    def record_order_result(self, success: bool) -> None:
        """Record order result for error rate tracking."""
        self._order_count += 1
        if not success:
            self._error_count += 1

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at start of day)."""
        self._error_count = 0
        self._order_count = 0
        self._daily_pnl = 0.0

    def is_trading_allowed(self) -> bool:
        """Check if new orders are allowed."""
        if self.current_state in [
            EmergencyState.HALT_NEW_ORDERS,
            EmergencyState.FLATTEN_POSITIONS,
            EmergencyState.FULL_STOP
        ]:
            return False

        if self.kill_switch.is_engaged:
            return False

        if not self.hours_controller.is_trading_allowed():
            return False

        if self.circuit_breaker.is_triggered:
            return False

        return True

    def engage_kill_switch(self, reason: str, triggered_by: str = "system") -> EmergencyEvent:
        """Engage the kill switch."""
        event = self.kill_switch.engage(reason, triggered_by)
        self._update_state(EmergencyState.FULL_STOP, event)
        return event

    def flatten_all(self, urgency: str = "normal") -> List[Order]:
        """Flatten all positions."""
        event = EmergencyEvent(
            event_id=f"FL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=TriggerType.MANUAL,
            reason=f"Position flattening requested: {urgency}",
            details={"urgency": urgency}
        )

        self._update_state(EmergencyState.FLATTEN_POSITIONS, event)

        use_market = urgency == "immediate"
        orders = self.position_flattener.flatten_all_positions(
            urgency=urgency,
            use_market_orders=use_market
        )

        event.positions_flattened = len(orders)

        return orders

    def recover_to_normal(self, authorized_by: str, notes: str = "") -> bool:
        """Attempt to recover to normal state."""
        if self.kill_switch.is_engaged:
            if not self.kill_switch.disengage(authorized_by, notes):
                return False

        if self.circuit_breaker.is_triggered:
            if not self.circuit_breaker.check_resume():
                logger.warning("Cannot recover: circuit breaker still active")
                return False

        event = EmergencyEvent(
            event_id=f"RC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            trigger_type=TriggerType.MANUAL,
            previous_state=self.current_state,
            new_state=EmergencyState.NORMAL,
            reason=f"Manual recovery by {authorized_by}",
            details={"notes": notes}
        )

        self.current_state = EmergencyState.NORMAL
        self.state_history.append(event)

        logger.info(f"Recovered to NORMAL state by {authorized_by}")

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive emergency control status."""
        return {
            "current_state": self.current_state.value,
            "trading_allowed": self.is_trading_allowed(),
            "kill_switch": self.kill_switch.get_status(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "trading_session": self.hours_controller.get_session_type(),
            "daily_pnl": self._daily_pnl,
            "error_rate": (
                self._error_count / self._order_count
                if self._order_count > 0 else 0
            ),
            "state_history_count": len(self.state_history)
        }

    def export_audit_log(self) -> List[Dict[str, Any]]:
        """Export full audit log of emergency events."""
        return [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "trigger_type": event.trigger_type.value,
                "previous_state": event.previous_state.value,
                "new_state": event.new_state.value,
                "reason": event.reason,
                "details": event.details,
                "orders_cancelled": event.orders_cancelled,
                "positions_flattened": event.positions_flattened,
                "resolved_at": (
                    event.resolved_at.isoformat() if event.resolved_at else None
                ),
                "resolution_notes": event.resolution_notes
            }
            for event in self.state_history
        ]
