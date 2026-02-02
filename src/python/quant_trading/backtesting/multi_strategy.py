"""
Multi-Strategy Portfolio Framework.

Allows running different strategies on different symbols within the same portfolio.
This enables strategy specialization based on asset characteristics.

Example:
    - Use Momentum for trending stocks (NVDA, TSLA)
    - Use Mean Reversion for range-bound stocks
    - Use MA Crossover for moderate volatility stocks
"""

from __future__ import annotations

import logging
from collections import deque
from queue import Queue
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Callable

import numpy as np

from .events import MarketEvent, SignalEvent, SignalType
from .strategy import Strategy

if TYPE_CHECKING:
    from .data_handler import DataHandler
    from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class MultiStrategyManager(Strategy):
    """
    Manages multiple strategies for different symbols.

    Allows assigning different strategies to different symbols based on
    their characteristics (trending vs mean-reverting, volatile vs stable).

    Example:
        >>> manager = MultiStrategyManager(events, data_handler, portfolio)
        >>> manager.add_strategy("NVDA", "momentum", lookback=10, threshold=0.03)
        >>> manager.add_strategy("AAPL", "ma_crossover", fast=3, slow=10)
        >>> manager.add_strategy("SPY", "mean_reversion", lookback=15, threshold=1.5)
    """

    def __init__(
        self,
        events_queue: Queue,
        data_handler: "DataHandler",
        portfolio: "Portfolio",
        strategy_id: str = "multi_strategy",
    ):
        """Initialize multi-strategy manager."""
        super().__init__(events_queue, data_handler, portfolio, strategy_id)

        # Symbol -> strategy configuration
        self.symbol_strategies: Dict[str, Dict] = {}

        # Price history for each symbol
        self.prices: Dict[str, Deque[float]] = {}

        # Previous MA values for crossover detection
        self.prev_fast_ma: Dict[str, Optional[float]] = {}
        self.prev_slow_ma: Dict[str, Optional[float]] = {}

        logger.info("MultiStrategyManager initialized")

    def add_strategy(
        self,
        symbol: str,
        strategy_type: str,
        **params
    ) -> None:
        """
        Add a strategy for a specific symbol.

        Args:
            symbol: Stock ticker
            strategy_type: Strategy type ("momentum", "ma_crossover", "mean_reversion", "rsi", "bollinger")
            **params: Strategy-specific parameters
        """
        # Set default parameters based on strategy type
        if strategy_type == "momentum":
            params.setdefault("lookback", 10)
            params.setdefault("threshold", 0.03)
        elif strategy_type == "ma_crossover":
            params.setdefault("fast", 3)
            params.setdefault("slow", 10)
        elif strategy_type == "mean_reversion":
            params.setdefault("lookback", 15)
            params.setdefault("entry_threshold", 1.5)
            params.setdefault("exit_threshold", 0.5)
        elif strategy_type == "rsi":
            params.setdefault("period", 14)
            params.setdefault("oversold", 30)
            params.setdefault("overbought", 70)
        elif strategy_type == "bollinger":
            params.setdefault("period", 20)
            params.setdefault("num_std", 2.0)

        self.symbol_strategies[symbol] = {
            "type": strategy_type,
            "params": params,
        }

        # Initialize price history
        max_lookback = max(
            params.get("lookback", 20),
            params.get("slow", 20),
            params.get("period", 20),
        ) + 10
        self.prices[symbol] = deque(maxlen=max_lookback)
        self.prev_fast_ma[symbol] = None
        self.prev_slow_ma[symbol] = None

        logger.info(f"Added {strategy_type} strategy for {symbol} with params {params}")

    def calculate_signals(self, event: MarketEvent) -> None:
        """Generate signals based on symbol-specific strategy."""
        symbol = event.symbol

        if symbol not in self.symbol_strategies:
            return

        # Update price history
        self.prices[symbol].append(event.price)

        strategy_config = self.symbol_strategies[symbol]
        strategy_type = strategy_config["type"]
        params = strategy_config["params"]

        # Route to appropriate strategy
        if strategy_type == "momentum":
            self._calculate_momentum_signal(symbol, event, params)
        elif strategy_type == "ma_crossover":
            self._calculate_ma_crossover_signal(symbol, event, params)
        elif strategy_type == "mean_reversion":
            self._calculate_mean_reversion_signal(symbol, event, params)
        elif strategy_type == "rsi":
            self._calculate_rsi_signal(symbol, event, params)
        elif strategy_type == "bollinger":
            self._calculate_bollinger_signal(symbol, event, params)

    def _calculate_momentum_signal(
        self,
        symbol: str,
        event: MarketEvent,
        params: Dict
    ) -> None:
        """Momentum strategy: buy when N-day return exceeds threshold."""
        lookback = params["lookback"]
        threshold = params["threshold"]

        if len(self.prices[symbol]) < lookback + 1:
            return

        prices_list = list(self.prices[symbol])
        momentum = (prices_list[-1] / prices_list[-lookback - 1]) - 1

        current_position = self.portfolio.get_position(symbol)

        # Entry: positive momentum above threshold
        if current_position == 0 and momentum > threshold:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=min(1.0, momentum / threshold),
                metadata={"momentum": momentum, "strategy": "momentum"},
            )
        # Exit: negative momentum
        elif current_position > 0 and momentum < 0:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={"momentum": momentum, "strategy": "momentum"},
            )

    def _calculate_ma_crossover_signal(
        self,
        symbol: str,
        event: MarketEvent,
        params: Dict
    ) -> None:
        """MA Crossover strategy: buy when fast MA crosses above slow MA."""
        fast_window = params["fast"]
        slow_window = params["slow"]

        if len(self.prices[symbol]) < slow_window:
            return

        prices_list = list(self.prices[symbol])
        fast_ma = np.mean(prices_list[-fast_window:])
        slow_ma = np.mean(prices_list[-slow_window:])

        prev_fast = self.prev_fast_ma[symbol]
        prev_slow = self.prev_slow_ma[symbol]

        if prev_fast is not None and prev_slow is not None:
            # Bullish crossover
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                current_position = self.portfolio.get_position(symbol)
                if current_position <= 0:
                    self._emit_signal(
                        symbol=symbol,
                        signal_type=SignalType.LONG,
                        strength=1.0,
                        metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "strategy": "ma_crossover"},
                    )
            # Bearish crossover
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                current_position = self.portfolio.get_position(symbol)
                if current_position > 0:
                    self._emit_signal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        strength=1.0,
                        metadata={"fast_ma": fast_ma, "slow_ma": slow_ma, "strategy": "ma_crossover"},
                    )

        self.prev_fast_ma[symbol] = fast_ma
        self.prev_slow_ma[symbol] = slow_ma

    def _calculate_mean_reversion_signal(
        self,
        symbol: str,
        event: MarketEvent,
        params: Dict
    ) -> None:
        """Mean reversion strategy: buy when z-score is below threshold."""
        lookback = params["lookback"]
        entry_threshold = params["entry_threshold"]
        exit_threshold = params["exit_threshold"]

        if len(self.prices[symbol]) < lookback:
            return

        prices_list = list(self.prices[symbol])
        mean = np.mean(prices_list[-lookback:])
        std = np.std(prices_list[-lookback:])

        if std < 1e-8:
            return

        z_score = (event.price - mean) / std
        current_position = self.portfolio.get_position(symbol)

        # Entry signals
        if current_position == 0:
            if z_score < -entry_threshold:
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, abs(z_score) / entry_threshold),
                    metadata={"z_score": z_score, "strategy": "mean_reversion"},
                )
            elif z_score > entry_threshold:
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, abs(z_score) / entry_threshold),
                    metadata={"z_score": z_score, "strategy": "mean_reversion"},
                )
        # Exit signals
        elif current_position > 0 and z_score > -exit_threshold:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={"z_score": z_score, "strategy": "mean_reversion"},
            )
        elif current_position < 0 and z_score < exit_threshold:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_SHORT,
                strength=1.0,
                metadata={"z_score": z_score, "strategy": "mean_reversion"},
            )

    def _calculate_rsi_signal(
        self,
        symbol: str,
        event: MarketEvent,
        params: Dict
    ) -> None:
        """RSI strategy: buy when oversold, sell when overbought."""
        period = params["period"]
        oversold = params["oversold"]
        overbought = params["overbought"]

        if len(self.prices[symbol]) < period + 1:
            return

        prices_list = list(self.prices[symbol])

        # Calculate price changes
        changes = np.diff(prices_list[-(period + 1):])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        current_position = self.portfolio.get_position(symbol)

        # Entry signals
        if current_position == 0:
            if rsi < oversold:
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=(oversold - rsi) / oversold,
                    metadata={"rsi": rsi, "strategy": "rsi"},
                )
            elif rsi > overbought:
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=(rsi - overbought) / (100 - overbought),
                    metadata={"rsi": rsi, "strategy": "rsi"},
                )
        # Exit signals
        elif current_position > 0 and rsi > 50:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={"rsi": rsi, "strategy": "rsi"},
            )
        elif current_position < 0 and rsi < 50:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_SHORT,
                strength=1.0,
                metadata={"rsi": rsi, "strategy": "rsi"},
            )

    def _calculate_bollinger_signal(
        self,
        symbol: str,
        event: MarketEvent,
        params: Dict
    ) -> None:
        """Bollinger Bands strategy: buy at lower band, sell at upper band."""
        period = params["period"]
        num_std = params["num_std"]

        if len(self.prices[symbol]) < period:
            return

        prices_list = list(self.prices[symbol])[-period:]
        mean = np.mean(prices_list)
        std = np.std(prices_list)

        upper_band = mean + num_std * std
        lower_band = mean - num_std * std

        current_position = self.portfolio.get_position(symbol)

        # Entry signals
        if current_position == 0:
            if event.price < lower_band:
                # Price below lower band - oversold
                strength = min(1.0, (lower_band - event.price) / (num_std * std))
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    metadata={
                        "price": event.price,
                        "lower_band": lower_band,
                        "upper_band": upper_band,
                        "strategy": "bollinger",
                    },
                )
            elif event.price > upper_band:
                # Price above upper band - overbought
                strength = min(1.0, (event.price - upper_band) / (num_std * std))
                self._emit_signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=strength,
                    metadata={
                        "price": event.price,
                        "lower_band": lower_band,
                        "upper_band": upper_band,
                        "strategy": "bollinger",
                    },
                )
        # Exit signals - exit when price crosses middle band
        elif current_position > 0 and event.price > mean:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_LONG,
                strength=1.0,
                metadata={"price": event.price, "mean": mean, "strategy": "bollinger"},
            )
        elif current_position < 0 and event.price < mean:
            self._emit_signal(
                symbol=symbol,
                signal_type=SignalType.EXIT_SHORT,
                strength=1.0,
                metadata={"price": event.price, "mean": mean, "strategy": "bollinger"},
            )


# Predefined strategy configurations based on our testing
OPTIMAL_STRATEGIES = {
    # High momentum stocks - use momentum strategy
    "NVDA": {"type": "momentum", "params": {"lookback": 10, "threshold": 0.03}},
    "TSLA": {"type": "momentum", "params": {"lookback": 10, "threshold": 0.04}},
    "AMD": {"type": "momentum", "params": {"lookback": 10, "threshold": 0.03}},

    # Large cap tech - use MA crossover
    "AAPL": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},
    "MSFT": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},
    "GOOGL": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},
    "META": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},
    "AMZN": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},

    # Index ETFs - use momentum
    "SPY": {"type": "momentum", "params": {"lookback": 10, "threshold": 0.02}},
    "QQQ": {"type": "momentum", "params": {"lookback": 10, "threshold": 0.025}},

    # Default for unknown symbols
    "DEFAULT": {"type": "ma_crossover", "params": {"fast": 3, "slow": 10}},
}


def get_optimal_strategy(symbol: str) -> Dict:
    """Get the optimal strategy configuration for a symbol."""
    return OPTIMAL_STRATEGIES.get(symbol, OPTIMAL_STRATEGIES["DEFAULT"])
