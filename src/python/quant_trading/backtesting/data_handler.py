"""
Data handlers for backtesting.

Provides historical market data to the backtest engine in an event-driven manner:
    - HistoricCSVDataHandler: Read from CSV files
    - HistoricDataFrameHandler: Read from pandas DataFrames
    - DatabaseDataHandler: Read from TimescaleDB

Features:
    - Bid-ask spread simulation
    - Multiple asset support
    - Date filtering
    - Missing data handling
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

from .events import MarketEvent

logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstract base class for data handlers.

    Provides market data to backtest engine in event-driven manner.
    Subclasses implement specific data source loading.
    """

    def __init__(self, events_queue: Queue):
        """
        Initialize data handler.

        Args:
            events_queue: Queue for pushing market events
        """
        self.events = events_queue
        self.continue_backtest = True
        self.symbol_list: List[str] = []
        self.bar_index = 0

    @abstractmethod
    def update_bars(self) -> None:
        """
        Push next bar(s) to event queue.

        Should create MarketEvent for each symbol and put on queue.
        Set continue_backtest = False when data is exhausted.
        """
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent bar for a symbol."""
        pass

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent bars for a symbol."""
        pass

    def reset(self) -> None:
        """Reset data handler to beginning."""
        self.bar_index = 0
        self.continue_backtest = True


class HistoricDataFrameHandler(DataHandler):
    """
    Read historical data from pandas DataFrames.

    Example:
        >>> # Create handler with DataFrame
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range('2020-01-01', periods=100),
        ...     'SPY_Close': np.random.randn(100) + 450,
        ...     'SPY_Volume': np.random.randint(1000000, 10000000, 100),
        ... })
        >>> handler = HistoricDataFrameHandler(
        ...     events_queue=queue,
        ...     data=data,
        ...     symbol_list=['SPY']
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        data: "pd.DataFrame",
        symbol_list: List[str],
        date_column: str = "Date",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bid_ask_spread_bps: float = 5.0,
    ):
        """
        Initialize DataFrame data handler.

        Args:
            events_queue: Event queue
            data: DataFrame with OHLCV data
            symbol_list: List of symbols in data
            date_column: Name of date column
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            bid_ask_spread_bps: Bid-ask spread in basis points
        """
        super().__init__(events_queue)

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for HistoricDataFrameHandler")

        self.data = data.copy()
        self.symbol_list = symbol_list
        self.date_column = date_column
        self.bid_ask_spread_bps = bid_ask_spread_bps

        # Ensure date column is datetime
        if date_column in self.data.columns:
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data = self.data.set_index(date_column)

        # Filter date range
        if start_date:
            self.data = self.data[self.data.index >= start_date]
        if end_date:
            self.data = self.data[self.data.index <= end_date]

        # Sort by date
        self.data = self.data.sort_index()

        # Store dates
        self.dates = list(self.data.index)
        self.n_bars = len(self.dates)

        # Track latest bars per symbol
        self.latest_bars: Dict[str, List[Dict]] = {s: [] for s in symbol_list}

        logger.info(
            f"Loaded {self.n_bars} bars for {len(symbol_list)} symbols "
            f"from {self.dates[0]} to {self.dates[-1]}"
        )

    def update_bars(self) -> None:
        """Push next bar to event queue."""
        if self.bar_index >= self.n_bars:
            self.continue_backtest = False
            return

        current_date = self.dates[self.bar_index]
        row = self.data.iloc[self.bar_index]

        for symbol in self.symbol_list:
            # Try to get price columns
            close_col = f"{symbol}_Close" if f"{symbol}_Close" in self.data.columns else "Close"
            volume_col = f"{symbol}_Volume" if f"{symbol}_Volume" in self.data.columns else "Volume"
            open_col = f"{symbol}_Open" if f"{symbol}_Open" in self.data.columns else "Open"
            high_col = f"{symbol}_High" if f"{symbol}_High" in self.data.columns else "High"
            low_col = f"{symbol}_Low" if f"{symbol}_Low" in self.data.columns else "Low"

            try:
                close_price = float(row[close_col]) if close_col in row.index else 0.0
                volume = float(row[volume_col]) if volume_col in row.index else 0.0
                open_price = float(row[open_col]) if open_col in row.index else close_price
                high_price = float(row[high_col]) if high_col in row.index else close_price
                low_price = float(row[low_col]) if low_col in row.index else close_price
            except (KeyError, ValueError):
                continue

            if close_price <= 0:
                continue

            # Simulate bid-ask spread
            spread_pct = self.bid_ask_spread_bps / 10000
            bid = close_price * (1 - spread_pct / 2)
            ask = close_price * (1 + spread_pct / 2)

            # Create market event
            market_event = MarketEvent(
                timestamp=current_date,
                event_type=None,  # Will be set in __post_init__
                symbol=symbol,
                price=close_price,
                volume=volume,
                bid=bid,
                ask=ask,
                open=open_price,
                high=high_price,
                low=low_price,
            )

            self.events.put(market_event)

            # Store in latest bars
            bar_data = {
                "datetime": current_date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
            self.latest_bars[symbol].append(bar_data)

        self.bar_index += 1

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent bar for a symbol."""
        if symbol in self.latest_bars and self.latest_bars[symbol]:
            return self.latest_bars[symbol][-1]
        return None

    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent bars for a symbol."""
        if symbol in self.latest_bars:
            return self.latest_bars[symbol][-n:]
        return []

    def get_latest_bar_value(self, symbol: str, field: str) -> Optional[float]:
        """Get a specific field from the latest bar."""
        bar = self.get_latest_bar(symbol)
        if bar and field in bar:
            return bar[field]
        return None


class HistoricCSVDataHandler(DataHandler):
    """
    Read historical data from CSV files.

    Expects CSV files named {symbol}.csv in the csv_dir directory.
    Each CSV should have columns: Date, Open, High, Low, Close, Volume

    Example:
        >>> handler = HistoricCSVDataHandler(
        ...     events_queue=queue,
        ...     csv_dir="./data",
        ...     symbol_list=["SPY", "TLT"]
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        csv_dir: str,
        symbol_list: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        bid_ask_spread_bps: float = 5.0,
        date_column: str = "Date",
    ):
        """
        Initialize CSV data handler.

        Args:
            events_queue: Event queue
            csv_dir: Directory containing CSV files
            symbol_list: List of symbols to load
            start_date: Start date for backtest
            end_date: End date for backtest
            bid_ask_spread_bps: Bid-ask spread in basis points
            date_column: Name of date column in CSV
        """
        super().__init__(events_queue)

        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for HistoricCSVDataHandler")

        self.csv_dir = Path(csv_dir)
        self.symbol_list = symbol_list
        self.start_date = start_date
        self.end_date = end_date
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.date_column = date_column

        # Load data
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.latest_bars: Dict[str, List[Dict]] = {}

        self._load_data()

    def _load_data(self) -> None:
        """Load CSV data for all symbols."""
        for symbol in self.symbol_list:
            csv_path = self.csv_dir / f"{symbol}.csv"

            if not csv_path.exists():
                logger.warning(f"CSV file not found: {csv_path}")
                continue

            # Load CSV
            df = pd.read_csv(
                csv_path,
                parse_dates=[self.date_column],
                index_col=self.date_column,
            )

            # Filter date range
            if self.start_date:
                df = df[df.index >= self.start_date]
            if self.end_date:
                df = df[df.index <= self.end_date]

            # Sort by date
            df = df.sort_index()

            # Store
            self.symbol_data[symbol] = df
            self.latest_bars[symbol] = []

        # Get common dates across all symbols
        if self.symbol_data:
            date_sets = [set(df.index) for df in self.symbol_data.values()]
            self.common_dates = sorted(set.intersection(*date_sets))
        else:
            self.common_dates = []

        logger.info(
            f"Loaded {len(self.common_dates)} common bars for "
            f"{len(self.symbol_data)} symbols"
        )

    def update_bars(self) -> None:
        """Push next bar to event queue."""
        if self.bar_index >= len(self.common_dates):
            self.continue_backtest = False
            return

        current_date = self.common_dates[self.bar_index]

        for symbol in self.symbol_list:
            if symbol not in self.symbol_data:
                continue

            try:
                bar = self.symbol_data[symbol].loc[current_date]

                close_price = float(bar.get("Close", bar.get("close", 0)))
                volume = float(bar.get("Volume", bar.get("volume", 0)))
                open_price = float(bar.get("Open", bar.get("open", close_price)))
                high_price = float(bar.get("High", bar.get("high", close_price)))
                low_price = float(bar.get("Low", bar.get("low", close_price)))

            except (KeyError, TypeError):
                continue

            if close_price <= 0:
                continue

            # Simulate bid-ask spread
            spread_pct = self.bid_ask_spread_bps / 10000
            bid = close_price * (1 - spread_pct / 2)
            ask = close_price * (1 + spread_pct / 2)

            # Create market event
            market_event = MarketEvent(
                timestamp=current_date,
                event_type=None,
                symbol=symbol,
                price=close_price,
                volume=volume,
                bid=bid,
                ask=ask,
                open=open_price,
                high=high_price,
                low=low_price,
            )

            self.events.put(market_event)

            # Store in latest bars
            bar_data = {
                "datetime": current_date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
            self.latest_bars[symbol].append(bar_data)

        self.bar_index += 1

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent bar for a symbol."""
        if symbol in self.latest_bars and self.latest_bars[symbol]:
            return self.latest_bars[symbol][-1]
        return None

    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent bars for a symbol."""
        if symbol in self.latest_bars:
            return self.latest_bars[symbol][-n:]
        return []


class SyntheticDataHandler(DataHandler):
    """
    Generate synthetic market data for testing.

    Useful for testing strategies with controlled data.

    Example:
        >>> handler = SyntheticDataHandler(
        ...     events_queue=queue,
        ...     symbol_list=['TEST'],
        ...     n_bars=252,
        ...     start_price=100.0,
        ...     volatility=0.20
        ... )
    """

    def __init__(
        self,
        events_queue: Queue,
        symbol_list: List[str],
        n_bars: int = 252,
        start_price: float = 100.0,
        volatility: float = 0.20,
        drift: float = 0.05,
        start_date: str = "2020-01-01",
        bid_ask_spread_bps: float = 5.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize synthetic data handler.

        Args:
            events_queue: Event queue
            symbol_list: List of symbols to generate
            n_bars: Number of bars to generate
            start_price: Starting price
            volatility: Annualized volatility
            drift: Annualized drift
            start_date: Start date
            bid_ask_spread_bps: Bid-ask spread in basis points
            seed: Random seed for reproducibility
        """
        super().__init__(events_queue)

        self.symbol_list = symbol_list
        self.n_bars = n_bars
        self.bid_ask_spread_bps = bid_ask_spread_bps

        if seed is not None:
            np.random.seed(seed)

        # Generate dates
        if PANDAS_AVAILABLE:
            self.dates = pd.date_range(start=start_date, periods=n_bars, freq='D')
        else:
            # Simple date generation without pandas
            base = datetime.strptime(start_date, "%Y-%m-%d")
            from datetime import timedelta
            self.dates = [base + timedelta(days=i) for i in range(n_bars)]

        # Generate prices for each symbol
        self.prices: Dict[str, np.ndarray] = {}
        self.volumes: Dict[str, np.ndarray] = {}

        daily_vol = volatility / np.sqrt(252)
        daily_drift = drift / 252

        for symbol in symbol_list:
            # Geometric Brownian motion
            returns = np.random.normal(daily_drift, daily_vol, n_bars)
            prices = start_price * np.exp(np.cumsum(returns))
            self.prices[symbol] = prices

            # Random volumes
            self.volumes[symbol] = np.random.randint(100000, 10000000, n_bars)

        # Track latest bars
        self.latest_bars: Dict[str, List[Dict]] = {s: [] for s in symbol_list}

        logger.info(
            f"Generated {n_bars} synthetic bars for {len(symbol_list)} symbols"
        )

    def update_bars(self) -> None:
        """Push next bar to event queue."""
        if self.bar_index >= self.n_bars:
            self.continue_backtest = False
            return

        current_date = self.dates[self.bar_index]

        for symbol in self.symbol_list:
            close_price = float(self.prices[symbol][self.bar_index])
            volume = float(self.volumes[symbol][self.bar_index])

            # Generate OHLC from close
            daily_range = close_price * 0.02  # 2% daily range
            open_price = close_price + np.random.uniform(-daily_range/2, daily_range/2)
            high_price = max(open_price, close_price) + np.random.uniform(0, daily_range/2)
            low_price = min(open_price, close_price) - np.random.uniform(0, daily_range/2)

            # Simulate bid-ask spread
            spread_pct = self.bid_ask_spread_bps / 10000
            bid = close_price * (1 - spread_pct / 2)
            ask = close_price * (1 + spread_pct / 2)

            market_event = MarketEvent(
                timestamp=current_date,
                event_type=None,
                symbol=symbol,
                price=close_price,
                volume=volume,
                bid=bid,
                ask=ask,
                open=open_price,
                high=high_price,
                low=low_price,
            )

            self.events.put(market_event)

            # Store in latest bars
            bar_data = {
                "datetime": current_date,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
            self.latest_bars[symbol].append(bar_data)

        self.bar_index += 1

    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the most recent bar for a symbol."""
        if symbol in self.latest_bars and self.latest_bars[symbol]:
            return self.latest_bars[symbol][-1]
        return None

    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get the N most recent bars for a symbol."""
        if symbol in self.latest_bars:
            return self.latest_bars[symbol][-n:]
        return []
