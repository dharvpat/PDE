"""
Strategy base class and sample strategies for backtesting.

Provides:
    - Strategy: Abstract base class for trading strategies
    - BuyAndHoldStrategy: Simple buy and hold
    - MovingAverageCrossover: MA crossover strategy
    - MeanReversionStrategy: Mean reversion based on z-score

Strategies receive MarketEvents and generate SignalEvents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from queue import Queue
from typing import TYPE_CHECKING, Deque, Dict, List, Optional

import numpy as np

from .events import MarketEvent, SignalEvent, SignalType

if TYPE_CHECKING:
    from .data_handler import DataHandler
    from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies receive market data and generate trading signals.
    Subclasses must implement calculate_signals().

    Example:
        >>> class MyStrategy(Strategy):
        ...     def calculate_signals(self, event: MarketEvent):
        ...         if some_condition:
        ...             signal = SignalEvent(...)
        ...             self.events.put(signal)
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        strategy_id: str = "default",
    ):
        """
        Initialize strategy.

        Args:
            events_queue: Queue for placing signal events
            data_handler: Data handler for historical data
            portfolio: Portfolio for position information
            strategy_id: Unique identifier for this strategy
        """
        self.events = events_queue
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.strategy_id = strategy_id

        # Track symbols being traded
        self.symbol_list: List[str] = data_handler.symbol_list

        logger.info(f"Strategy '{strategy_id}' initialized")

    @abstractmethod
    def calculate_signals(self, event: MarketEvent) -> None:
        """
        Generate trading signals from market event.

        Args:
            event: Market event with new price data
        """
        pass

    def _emit_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: float = 1.0,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Helper to emit a signal event.

        Args:
            symbol: Asset symbol
            signal_type: Type of signal
            strength: Signal strength [0, 1]
            target_price: Target entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            metadata: Additional signal data
        """
        from .events import MarketEvent  # Avoid circular import

        # Get current timestamp from data handler
        latest_bar = self.data_handler.get_latest_bar(symbol)
        timestamp = latest_bar["datetime"] if latest_bar else None

        signal = SignalEvent(
            timestamp=timestamp,
            event_type=None,  # Set in __post_init__
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            strategy_id=self.strategy_id,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata or {},
        )

        self.events.put(signal)
        logger.debug(f"Signal: {signal_type.value} {symbol} (strength={strength:.2f})")


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy and hold strategy.

    Buys all symbols on the first bar and holds until the end.
    Useful as a benchmark.

    Example:
        >>> strategy = BuyAndHoldStrategy(events, data_handler, portfolio)
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        strategy_id: str = "buy_and_hold",
    ):
        """Initialize buy and hold strategy."""
        super().__init__(events_queue, data_handler, portfolio, strategy_id)
        self.bought: Dict[str, bool] = {s: False for s in self.symbol_list}

    def calculate_signals(self, event: MarketEvent) -> None:
        """Generate buy signal on first bar for each symbol."""
        symbol = event.symbol

        if symbol in self.bought and not self.bought[symbol]:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=1.0,
            )
            self.bought[symbol] = True


class MovingAverageCrossoverStrategy(Strategy):
    """
    Moving average crossover strategy.

    Generates:
        - LONG signal when fast MA crosses above slow MA
        - EXIT signal when fast MA crosses below slow MA

    Example:
        >>> strategy = MovingAverageCrossoverStrategy(
        ...     events, data_handler, portfolio,
        ...     fast_window=10, slow_window=50
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        fast_window: int = 10,
        slow_window: int = 50,
        strategy_id: str = "ma_crossover",
    ):
        """
        Initialize MA crossover strategy.

        Args:
            fast_window: Fast moving average window
            slow_window: Slow moving average window
        """
        super().__init__(events_queue, data_handler, portfolio, strategy_id)

        self.fast_window = fast_window
        self.slow_window = slow_window

        # Track prices for MA calculation
        self.prices: Dict[str, Deque[float]] = {
            s: deque(maxlen=slow_window) for s in self.symbol_list
        }

        # Track previous MA values for crossover detection
        self.prev_fast_ma: Dict[str, Optional[float]] = {s: None for s in self.symbol_list}
        self.prev_slow_ma: Dict[str, Optional[float]] = {s: None for s in self.symbol_list}

    def calculate_signals(self, event: MarketEvent) -> None:
        """Generate signals on MA crossover."""
        symbol = event.symbol

        if symbol not in self.prices:
            return

        # Update price history
        self.prices[symbol].append(event.price)

        # Need enough data for slow MA
        if len(self.prices[symbol]) < self.slow_window:
            return

        # Calculate MAs
        prices_list = list(self.prices[symbol])
        fast_ma = np.mean(prices_list[-self.fast_window:])
        slow_ma = np.mean(prices_list[-self.slow_window:])

        # Check for crossover
        prev_fast = self.prev_fast_ma[symbol]
        prev_slow = self.prev_slow_ma[symbol]

        if prev_fast is not None and prev_slow is not None:
            # Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                current_position = self.portfolio.get_position(symbol)
                if current_position <= 0:
                    self._emit_signal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=1.0,
                        metadata={"fast_ma": fast_ma, "slow_ma": slow_ma},
                    )

            # Bearish crossover: fast crosses below slow
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                current_position = self.portfolio.get_position(symbol)
                if current_position > 0:
                    self._emit_signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        strength=1.0,
                        metadata={"fast_ma": fast_ma, "slow_ma": slow_ma},
                    )

        # Update previous values
        self.prev_fast_ma[symbol] = fast_ma
        self.prev_slow_ma[symbol] = slow_ma


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy based on z-score.

    Generates:
        - LONG signal when z-score < -entry_threshold
        - SHORT signal when z-score > entry_threshold
        - EXIT signals when z-score crosses zero

    Based on the Ornstein-Uhlenbeck assumption of the spread.

    Example:
        >>> strategy = MeanReversionStrategy(
        ...     events, data_handler, portfolio,
        ...     lookback=20, entry_threshold=2.0, exit_threshold=0.5
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        strategy_id: str = "mean_reversion",
    ):
        """
        Initialize mean reversion strategy.

        Args:
            lookback: Lookback period for mean and std calculation
            entry_threshold: Z-score threshold for entry (e.g., 2.0)
            exit_threshold: Z-score threshold for exit (e.g., 0.5)
        """
        super().__init__(events_queue, data_handler, portfolio, strategy_id)

        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        # Track prices
        self.prices: Dict[str, Deque[float]] = {
            s: deque(maxlen=lookback) for s in self.symbol_list
        }

    def calculate_signals(self, event: MarketEvent) -> None:
        """Generate signals based on z-score."""
        symbol = event.symbol

        if symbol not in self.prices:
            return

        # Update price history
        self.prices[symbol].append(event.price)

        # Need enough data
        if len(self.prices[symbol]) < self.lookback:
            return

        # Calculate z-score
        prices_list = list(self.prices[symbol])
        mean = np.mean(prices_list)
        std = np.std(prices_list)

        if std < 1e-8:
            return

        z_score = (event.price - mean) / std

        # Get current position
        current_position = self.portfolio.get_position(symbol)

        # Entry signals
        if current_position == 0:
            if z_score < -self.entry_threshold:
                # Price is low, expect reversion up -> LONG
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    metadata={"z_score": z_score, "mean": mean, "std": std},
                )
            elif z_score > self.entry_threshold:
                # Price is high, expect reversion down -> SHORT
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, abs(z_score) / self.entry_threshold),
                    metadata={"z_score": z_score, "mean": mean, "std": std},
                )

        # Exit signals
        elif current_position > 0:
            if z_score > -self.exit_threshold:
                # Long position, price reverted to mean -> EXIT
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    strength=1.0,
                    metadata={"z_score": z_score},
                )

        elif current_position < 0:
            if z_score < self.exit_threshold:
                # Short position, price reverted to mean -> EXIT
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT_SHORT,
                    strength=1.0,
                    metadata={"z_score": z_score},
                )


class MomentumStrategy(Strategy):
    """
    Momentum strategy based on trailing returns.

    Goes LONG when N-day return is positive and above threshold.
    Exits when momentum turns negative.

    Example:
        >>> strategy = MomentumStrategy(
        ...     events, data_handler, portfolio,
        ...     lookback=20, threshold=0.02
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        lookback: int = 20,
        threshold: float = 0.02,
        strategy_id: str = "momentum",
    ):
        """
        Initialize momentum strategy.

        Args:
            lookback: Lookback period for momentum calculation
            threshold: Minimum momentum for entry (e.g., 0.02 = 2%)
        """
        super().__init__(events_queue, data_handler, portfolio, strategy_id)

        self.lookback = lookback
        self.threshold = threshold

        # Track prices
        self.prices: Dict[str, Deque[float]] = {
            s: deque(maxlen=lookback + 1) for s in self.symbol_list
        }

    def calculate_signals(self, event: MarketEvent) -> None:
        """Generate signals based on momentum."""
        symbol = event.symbol

        if symbol not in self.prices:
            return

        # Update price history
        self.prices[symbol].append(event.price)

        # Need enough data
        if len(self.prices[symbol]) < self.lookback + 1:
            return

        # Calculate momentum (return over lookback period)
        prices_list = list(self.prices[symbol])
        momentum = (prices_list[-1] / prices_list[0]) - 1

        # Get current position
        current_position = self.portfolio.get_position(symbol)

        # Entry: positive momentum above threshold
        if current_position == 0 and momentum > self.threshold:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=min(1.0, momentum / self.threshold),
                metadata={"momentum": momentum},
            )

        # Exit: negative momentum
        elif current_position > 0 and momentum < 0:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={"momentum": momentum},
            )
