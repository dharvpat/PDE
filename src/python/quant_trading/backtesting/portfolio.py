"""
Portfolio management for backtesting.

Tracks portfolio state including:
    - Current positions (long and short)
    - Cash balance
    - Equity curve over time
    - Trade history
    - Performance metrics

Reference:
    Standard portfolio management practices for systematic trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .events import (
    Direction,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Represents a position in an asset.

    Attributes:
        symbol: Asset symbol
        quantity: Number of shares (positive=long, negative=short)
        avg_entry_price: Volume-weighted average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L from closed trades
        entry_time: When position was first opened
    """

    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: Optional[datetime] = None

    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.quantity * self.avg_entry_price

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    def update_price(self, price: float) -> None:
        """Update current price and recalculate P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Trade:
    """
    Record of a completed trade.

    Attributes:
        symbol: Asset symbol
        direction: BUY or SELL
        quantity: Number of shares
        entry_price: Entry price
        exit_price: Exit price (if closed)
        entry_time: Entry timestamp
        exit_time: Exit timestamp (if closed)
        pnl: Realized P&L
        commission: Total commission paid
        slippage: Total slippage
        strategy_id: Strategy that generated trade
    """

    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    strategy_id: str = "default"

    @property
    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_price is not None

    @property
    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.entry_price > 0 and self.exit_price is not None:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        return 0.0

    @property
    def holding_period(self) -> Optional[float]:
        """Calculate holding period in days."""
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            return delta.total_seconds() / (24 * 3600)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl": self.pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "strategy_id": self.strategy_id,
        }


class Portfolio:
    """
    Track portfolio state during backtest.

    Maintains:
        - Current positions
        - Cash balance
        - Equity curve
        - Trade history
        - Performance metrics

    Example:
        >>> portfolio = Portfolio(initial_capital=1_000_000)
        >>> portfolio.update_fill(fill_event)
        >>> print(f"Current equity: ${portfolio.equity:,.0f}")
    """

    def __init__(
        self,
        initial_capital: float,
        max_position_pct: float = 0.10,
        allow_shorting: bool = True,
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash
            max_position_pct: Maximum position size as fraction of equity
            allow_shorting: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_position_pct = max_position_pct
        self.allow_shorting = allow_shorting

        # Positions: symbol -> Position
        self.positions: Dict[str, Position] = {}

        # Current market prices
        self.current_prices: Dict[str, float] = {}

        # History tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.cash_curve: List[Tuple[datetime, float]] = []
        self.trade_history: List[Trade] = []
        self.fill_history: List[Dict[str, Any]] = []

        # Cost tracking
        self.total_commission = 0.0
        self.total_slippage = 0.0

        # Current timestamp
        self._current_time: Optional[datetime] = None

        logger.info(f"Portfolio initialized with ${initial_capital:,.0f} capital")

    @property
    def equity(self) -> float:
        """Calculate current total equity (cash + positions value)."""
        positions_value = sum(
            pos.market_value for pos in self.positions.values()
        )
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return sum(trade.pnl for trade in self.trade_history if trade.is_closed)

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def gross_exposure(self) -> float:
        """Gross exposure (sum of absolute position values)."""
        return sum(abs(pos.market_value) for pos in self.positions.values())

    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def leverage(self) -> float:
        """Current leverage ratio."""
        if self.equity > 0:
            return self.gross_exposure / self.equity
        return 0.0

    def update_market_data(self, event: MarketEvent) -> None:
        """
        Update portfolio with new market data.

        Args:
            event: Market event with new price data
        """
        self._current_time = event.timestamp
        self.current_prices[event.symbol] = event.price

        # Update position price if we have one
        if event.symbol in self.positions:
            self.positions[event.symbol].update_price(event.price)

        # Record equity
        self.equity_curve.append((event.timestamp, self.equity))
        self.cash_curve.append((event.timestamp, self.cash))

    def update_fill(self, fill: FillEvent) -> None:
        """
        Update portfolio with filled order.

        Args:
            fill: Fill event to process
        """
        symbol = fill.symbol
        quantity = fill.quantity
        direction = fill.direction
        fill_price = fill.fill_price
        commission = fill.commission
        slippage = fill.slippage

        # Track costs
        self.total_commission += commission
        self.total_slippage += slippage

        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_entry_price=0.0,
                current_price=fill_price,
                entry_time=fill.timestamp,
            )

        position = self.positions[symbol]

        # Update position based on direction
        if direction == Direction.BUY:
            self._process_buy(position, quantity, fill_price, fill.timestamp)
            self.cash -= (quantity * fill_price) + commission
        else:  # SELL
            self._process_sell(position, quantity, fill_price, fill.timestamp, fill.strategy_id)
            self.cash += (quantity * fill_price) - commission

        # Update current price
        position.update_price(fill_price)

        # Record fill
        self.fill_history.append({
            "timestamp": fill.timestamp,
            "symbol": symbol,
            "direction": direction.value,
            "quantity": quantity,
            "price": fill_price,
            "commission": commission,
            "slippage": slippage,
            "strategy_id": fill.strategy_id,
        })

        # Remove zero positions
        if abs(position.quantity) < 1e-6:
            del self.positions[symbol]

        logger.debug(
            f"Fill: {direction.value} {quantity} {symbol} @ ${fill_price:.2f}, "
            f"commission=${commission:.2f}"
        )

    def _process_buy(
        self,
        position: Position,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Process a buy fill."""
        if position.quantity >= 0:
            # Adding to long or opening long
            total_cost = position.quantity * position.avg_entry_price + quantity * price
            position.quantity += quantity
            if position.quantity > 0:
                position.avg_entry_price = total_cost / position.quantity
            if position.entry_time is None:
                position.entry_time = timestamp
        else:
            # Covering short position
            cover_qty = min(quantity, abs(position.quantity))
            pnl = (position.avg_entry_price - price) * cover_qty  # Short P&L

            # Record closed trade
            self.trade_history.append(Trade(
                symbol=position.symbol,
                direction=Direction.SELL,  # Original short direction
                quantity=cover_qty,
                entry_price=position.avg_entry_price,
                exit_price=price,
                entry_time=position.entry_time,
                exit_time=timestamp,
                pnl=pnl,
            ))

            position.quantity += quantity
            if position.quantity > 0:
                # Flipped to long
                position.avg_entry_price = price
                position.entry_time = timestamp

    def _process_sell(
        self,
        position: Position,
        quantity: float,
        price: float,
        timestamp: datetime,
        strategy_id: str,
    ) -> None:
        """Process a sell fill."""
        if position.quantity <= 0:
            # Adding to short or opening short
            if not self.allow_shorting:
                logger.warning(f"Short selling not allowed, ignoring sell for {position.symbol}")
                return

            total_cost = abs(position.quantity) * position.avg_entry_price + quantity * price
            position.quantity -= quantity
            if position.quantity < 0:
                position.avg_entry_price = total_cost / abs(position.quantity)
            if position.entry_time is None:
                position.entry_time = timestamp
        else:
            # Closing long position
            sell_qty = min(quantity, position.quantity)
            pnl = (price - position.avg_entry_price) * sell_qty  # Long P&L

            # Record closed trade
            self.trade_history.append(Trade(
                symbol=position.symbol,
                direction=Direction.BUY,  # Original long direction
                quantity=sell_qty,
                entry_price=position.avg_entry_price,
                exit_price=price,
                entry_time=position.entry_time,
                exit_time=timestamp,
                pnl=pnl,
                strategy_id=strategy_id,
            ))

            position.quantity -= quantity
            if position.quantity < 0:
                # Flipped to short
                if self.allow_shorting:
                    position.avg_entry_price = price
                    position.entry_time = timestamp
                else:
                    position.quantity = 0

    def get_position(self, symbol: str) -> float:
        """Get current position quantity in symbol."""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0.0

    def get_position_value(self, symbol: str) -> float:
        """Get current market value of position in symbol."""
        if symbol in self.positions:
            return self.positions[symbol].market_value
        return 0.0

    def calculate_target_quantity(
        self,
        symbol: str,
        signal: SignalEvent,
    ) -> float:
        """
        Calculate target quantity for a signal.

        Uses signal strength and max position constraints.

        Args:
            symbol: Asset symbol
            signal: Signal event

        Returns:
            Target quantity to trade
        """
        if signal.target_quantity is not None:
            return signal.target_quantity

        # Get current price
        price = self.current_prices.get(symbol, 0.0)
        if price <= 0:
            return 0.0

        # Calculate max position size based on equity
        max_position_value = self.equity * self.max_position_pct
        max_quantity = max_position_value / price

        # Scale by signal strength
        target_quantity = max_quantity * signal.strength

        return target_quantity

    def generate_order(
        self,
        signal: SignalEvent,
        events_queue,
    ) -> Optional[OrderEvent]:
        """
        Generate order from signal.

        Args:
            signal: Signal event
            events_queue: Queue to put order event

        Returns:
            OrderEvent if order generated, None otherwise
        """
        symbol = signal.symbol
        current_position = self.get_position(symbol)
        current_price = self.current_prices.get(symbol, 0.0)

        if current_price <= 0:
            logger.warning(f"No price data for {symbol}, skipping signal")
            return None

        # Determine order based on signal type
        order = None

        if signal.signal_type == SignalType.LONG:
            if current_position <= 0:
                # Open or add to long
                target_qty = self.calculate_target_quantity(symbol, signal)
                if target_qty > 0:
                    order = OrderEvent(
                        event_type=None,
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        quantity=target_qty,
                        direction=Direction.BUY,
                        strategy_id=signal.strategy_id,
                    )

        elif signal.signal_type == SignalType.SHORT:
            if not self.allow_shorting:
                logger.debug(f"Short selling disabled, ignoring SHORT signal for {symbol}")
                return None

            if current_position >= 0:
                # Open or add to short
                target_qty = self.calculate_target_quantity(symbol, signal)
                if target_qty > 0:
                    order = OrderEvent(
                        event_type=None,
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        order_type=OrderType.MARKET,
                        quantity=target_qty,
                        direction=Direction.SELL,
                        strategy_id=signal.strategy_id,
                    )

        elif signal.signal_type == SignalType.EXIT_LONG:
            if current_position > 0:
                order = OrderEvent(
                    event_type=None,
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_position),
                    direction=Direction.SELL,
                    strategy_id=signal.strategy_id,
                )

        elif signal.signal_type == SignalType.EXIT_SHORT:
            if current_position < 0:
                order = OrderEvent(
                    event_type=None,
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_position),
                    direction=Direction.BUY,
                    strategy_id=signal.strategy_id,
                )

        elif signal.signal_type == SignalType.EXIT:
            if current_position != 0:
                direction = Direction.SELL if current_position > 0 else Direction.BUY
                order = OrderEvent(
                    event_type=None,
                    timestamp=signal.timestamp,
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    quantity=abs(current_position),
                    direction=direction,
                    strategy_id=signal.strategy_id,
                )

        if order is not None:
            events_queue.put(order)
            logger.debug(
                f"Order generated: {order.direction.value} {order.quantity:.0f} {symbol}"
            )

        return order

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_pnl": self.total_pnl,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "num_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
        }

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.current_prices = {}
        self.equity_curve = []
        self.cash_curve = []
        self.trade_history = []
        self.fill_history = []
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self._current_time = None

        logger.info("Portfolio reset")
