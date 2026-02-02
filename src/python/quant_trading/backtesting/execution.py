"""
Execution handlers for backtesting.

Simulates realistic order execution with:
    - Slippage modeling
    - Market impact (square-root model)
    - Commission calculation
    - Partial fills
    - Order rejection

Reference:
    - Almgren & Chriss (2001) for market impact modeling
    - Kissell & Glantz (2003) for transaction cost analysis
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from queue import Queue
from typing import Dict, Optional

import numpy as np

from .events import Direction, FillEvent, MarketEvent, OrderEvent, OrderType

logger = logging.getLogger(__name__)


class CommissionModel(ABC):
    """Abstract base class for commission models."""

    @abstractmethod
    def calculate(self, quantity: float, price: float) -> float:
        """
        Calculate commission for a trade.

        Args:
            quantity: Number of shares/contracts
            price: Execution price

        Returns:
            Commission amount in dollars
        """
        pass


class ZeroCommission(CommissionModel):
    """No commission (for testing)."""

    def calculate(self, quantity: float, price: float) -> float:
        """Return zero commission."""
        return 0.0


class FixedCommission(CommissionModel):
    """
    Fixed percentage commission.

    Example:
        >>> comm = FixedCommission(rate=0.001)  # 10 bps
        >>> comm.calculate(100, 450.0)  # $45,000 trade
        45.0
    """

    def __init__(self, rate: float = 0.001):
        """
        Initialize with commission rate.

        Args:
            rate: Commission as fraction of trade value (e.g., 0.001 = 0.1%)
        """
        self.rate = rate

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission."""
        return abs(quantity * price * self.rate)


class PerShareCommission(CommissionModel):
    """
    Per-share commission with minimum.

    Example:
        >>> comm = PerShareCommission(per_share=0.005, minimum=1.0)
        >>> comm.calculate(100, 450.0)  # 100 shares
        1.0  # minimum
        >>> comm.calculate(1000, 450.0)  # 1000 shares
        5.0  # 1000 * $0.005
    """

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        """
        Initialize per-share commission.

        Args:
            per_share: Commission per share
            minimum: Minimum commission per trade
        """
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission."""
        commission = abs(quantity) * self.per_share
        return max(commission, self.minimum)


class TieredCommission(CommissionModel):
    """
    Tiered commission based on trade size.

    Example:
        >>> comm = TieredCommission()
        >>> comm.calculate(1000, 50.0)  # $50,000 trade
        # First $10k at 0.2%, next $40k at 0.1% = $20 + $40 = $60
    """

    def __init__(
        self,
        tiers: Optional[list] = None,
    ):
        """
        Initialize tiered commission.

        Args:
            tiers: List of (threshold, rate) tuples
                   Default: [(10000, 0.002), (100000, 0.001), (inf, 0.0005)]
        """
        if tiers is None:
            self.tiers = [
                (10000, 0.002),  # First $10k: 0.2%
                (100000, 0.001),  # $10k-$100k: 0.1%
                (float("inf"), 0.0005),  # Above $100k: 0.05%
            ]
        else:
            self.tiers = tiers

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate tiered commission."""
        trade_value = abs(quantity * price)
        commission = 0.0
        remaining = trade_value
        prev_threshold = 0

        for threshold, rate in self.tiers:
            tier_size = min(remaining, threshold - prev_threshold)
            if tier_size > 0:
                commission += tier_size * rate
                remaining -= tier_size

            if remaining <= 0:
                break

            prev_threshold = threshold

        return commission


class IBKRCommission(CommissionModel):
    """
    Interactive Brokers style commission.

    Combines per-share rate with minimum and maximum per order.
    """

    def __init__(
        self,
        per_share: float = 0.005,
        minimum: float = 1.0,
        maximum_pct: float = 0.01,  # 1% max
    ):
        """
        Initialize IBKR-style commission.

        Args:
            per_share: Commission per share
            minimum: Minimum per order
            maximum_pct: Maximum as percentage of trade value
        """
        self.per_share = per_share
        self.minimum = minimum
        self.maximum_pct = maximum_pct

    def calculate(self, quantity: float, price: float) -> float:
        """Calculate commission."""
        trade_value = abs(quantity * price)
        commission = abs(quantity) * self.per_share

        # Apply minimum
        commission = max(commission, self.minimum)

        # Apply maximum
        maximum = trade_value * self.maximum_pct
        commission = min(commission, maximum)

        return commission


class ExecutionHandler(ABC):
    """
    Abstract base class for execution handlers.

    Converts OrderEvents to FillEvents with realistic execution assumptions.
    """

    def __init__(self, events_queue: Queue):
        """
        Initialize execution handler.

        Args:
            events_queue: Queue for placing fill events
        """
        self.events = events_queue
        self.current_prices: Dict[str, float] = {}
        self.current_volumes: Dict[str, float] = {}
        self.current_bids: Dict[str, float] = {}
        self.current_asks: Dict[str, float] = {}

    def update_market_data(self, event: MarketEvent) -> None:
        """
        Update market data from market event.

        Args:
            event: Market event with price data
        """
        self.current_prices[event.symbol] = event.price
        self.current_volumes[event.symbol] = event.volume
        if event.bid is not None:
            self.current_bids[event.symbol] = event.bid
        if event.ask is not None:
            self.current_asks[event.symbol] = event.ask

    @abstractmethod
    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """
        Execute an order and return fill event.

        Args:
            order: Order to execute

        Returns:
            FillEvent if order executed, None otherwise
        """
        pass


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution with realistic slippage and market impact.

    Features:
        - Bid-ask spread crossing cost
        - Random slippage
        - Market impact (square-root model)
        - Partial fill simulation
        - Commission calculation

    Example:
        >>> executor = SimulatedExecutionHandler(
        ...     events_queue=queue,
        ...     slippage_bps=5,
        ...     market_impact_factor=0.1,
        ...     commission_model=FixedCommission(0.001)
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        slippage_bps: float = 5.0,
        market_impact_factor: float = 0.1,
        partial_fill_prob: float = 0.0,
        commission_model: Optional[CommissionModel] = None,
        fill_ratio: float = 1.0,
    ):
        """
        Initialize simulated execution handler.

        Args:
            events_queue: Event queue
            slippage_bps: Average slippage in basis points
            market_impact_factor: Market impact coefficient (Almgren-Chriss)
            partial_fill_prob: Probability of partial fill (0-1)
            commission_model: Commission calculation model
            fill_ratio: Ratio of order to fill (for simulating liquidity)
        """
        super().__init__(events_queue)

        self.slippage_bps = slippage_bps
        self.market_impact_factor = market_impact_factor
        self.partial_fill_prob = partial_fill_prob
        self.commission_model = commission_model or FixedCommission(0.001)
        self.fill_ratio = fill_ratio

        logger.info(
            f"SimulatedExecutionHandler initialized: slippage={slippage_bps}bps, "
            f"impact_factor={market_impact_factor}"
        )

    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """
        Execute order with realistic slippage and market impact.

        Args:
            order: Order to execute

        Returns:
            FillEvent if order executed, None otherwise
        """
        symbol = order.symbol

        # Check if we have market data
        if symbol not in self.current_prices:
            logger.warning(f"No price data for {symbol}, skipping order")
            return None

        mid_price = self.current_prices[symbol]
        volume = self.current_volumes.get(symbol, 0)

        # Get bid/ask if available
        bid = self.current_bids.get(symbol, mid_price * 0.9995)
        ask = self.current_asks.get(symbol, mid_price * 1.0005)

        # Calculate execution price based on order type
        if order.order_type == OrderType.MARKET:
            fill_price = self._calculate_market_fill_price(
                mid_price, bid, ask, order.quantity, order.direction, volume
            )
        elif order.order_type == OrderType.LIMIT:
            fill_price = self._calculate_limit_fill_price(
                order.limit_price, mid_price, bid, ask, order.direction
            )
            if fill_price is None:
                # Limit not hit
                return None
        elif order.order_type == OrderType.STOP:
            if not self._is_stop_triggered(order.stop_price, mid_price, order.direction):
                return None
            fill_price = self._calculate_market_fill_price(
                mid_price, bid, ask, order.quantity, order.direction, volume
            )
        else:
            fill_price = mid_price

        # Calculate slippage
        slippage = abs(fill_price - mid_price) * order.quantity

        # Calculate commission
        commission = self.commission_model.calculate(order.quantity, fill_price)

        # Determine fill quantity (simulate partial fills)
        filled_quantity = order.quantity * self.fill_ratio
        if np.random.random() < self.partial_fill_prob:
            filled_quantity *= np.random.uniform(0.5, 1.0)

        # Round to whole shares
        filled_quantity = round(filled_quantity)
        if filled_quantity <= 0:
            return None

        # Create fill event
        fill = FillEvent(
            timestamp=order.timestamp,
            event_type=None,
            symbol=symbol,
            quantity=filled_quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            strategy_id=order.strategy_id,
            order_id=order.order_id,
        )

        # Put on queue
        self.events.put(fill)

        logger.debug(
            f"Filled: {order.direction.value} {filled_quantity} {symbol} @ "
            f"${fill_price:.2f} (slippage=${slippage:.2f}, comm=${commission:.2f})"
        )

        return fill

    def _calculate_market_fill_price(
        self,
        mid_price: float,
        bid: float,
        ask: float,
        quantity: float,
        direction: Direction,
        volume: float,
    ) -> float:
        """
        Calculate fill price for market order.

        Includes:
            - Bid-ask spread crossing
            - Random slippage
            - Market impact (square-root model)

        Args:
            mid_price: Mid price
            bid: Bid price
            ask: Ask price
            quantity: Order quantity
            direction: BUY or SELL
            volume: Daily volume

        Returns:
            Fill price
        """
        # Start with bid or ask
        if direction == Direction.BUY:
            base_price = ask
        else:
            base_price = bid

        # Market impact: k * sigma * sqrt(Q/V)
        # Simplified version without sigma
        if volume > 0:
            impact_pct = self.market_impact_factor * np.sqrt(quantity / volume)
            impact_cost = mid_price * impact_pct
        else:
            impact_cost = 0.0

        # Additional random slippage
        random_slippage = mid_price * np.abs(
            np.random.normal(0, self.slippage_bps / 10000 / 2)
        )

        # Apply costs
        if direction == Direction.BUY:
            fill_price = base_price + impact_cost + random_slippage
        else:
            fill_price = base_price - impact_cost - random_slippage

        # Ensure positive price
        return max(fill_price, 0.01)

    def _calculate_limit_fill_price(
        self,
        limit_price: float,
        mid_price: float,
        bid: float,
        ask: float,
        direction: Direction,
    ) -> Optional[float]:
        """
        Calculate fill price for limit order.

        Returns None if limit would not be hit.
        """
        if direction == Direction.BUY:
            # Buy limit fills if market <= limit
            if ask <= limit_price:
                return min(ask, limit_price)
        else:
            # Sell limit fills if market >= limit
            if bid >= limit_price:
                return max(bid, limit_price)

        return None

    def _is_stop_triggered(
        self,
        stop_price: float,
        mid_price: float,
        direction: Direction,
    ) -> bool:
        """Check if stop order is triggered."""
        if direction == Direction.BUY:
            # Buy stop triggers when price >= stop
            return mid_price >= stop_price
        else:
            # Sell stop triggers when price <= stop
            return mid_price <= stop_price


class InstantExecutionHandler(ExecutionHandler):
    """
    Instant execution at mid price (no slippage).

    Useful for testing strategy logic without execution concerns.
    """

    def __init__(
        self,
        events_queue: Queue,
        commission_model: Optional[CommissionModel] = None,
    ):
        """
        Initialize instant execution handler.

        Args:
            events_queue: Event queue
            commission_model: Commission model (default: zero commission)
        """
        super().__init__(events_queue)
        self.commission_model = commission_model or ZeroCommission()

    def execute_order(self, order: OrderEvent) -> Optional[FillEvent]:
        """Execute order instantly at current price."""
        symbol = order.symbol

        if symbol not in self.current_prices:
            logger.warning(f"No price data for {symbol}")
            return None

        fill_price = self.current_prices[symbol]
        commission = self.commission_model.calculate(order.quantity, fill_price)

        fill = FillEvent(
            timestamp=order.timestamp,
            event_type=None,
            symbol=symbol,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission,
            slippage=0.0,
            strategy_id=order.strategy_id,
        )

        self.events.put(fill)
        return fill
