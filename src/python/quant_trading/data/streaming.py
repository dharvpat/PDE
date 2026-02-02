"""
Real-Time Data Streaming Module.

This module provides real-time market data streaming infrastructure:
- WebSocket connections to multiple data providers
- Event-driven data processing
- Automatic reconnection and heartbeat management
- Data buffering and batch processing
- Real-time quote and trade aggregation

Design follows the Observer pattern for flexible subscription management.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union
)
from queue import Queue

import numpy as np

logger = logging.getLogger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""
    QUOTE = "quote"
    TRADE = "trade"
    BAR = "bar"
    ORDER_BOOK = "order_book"
    OPTIONS_QUOTE = "options_quote"
    STATUS = "status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class StreamEvent:
    """Base class for streaming events."""
    event_type: StreamEventType
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    sequence: Optional[int] = None
    exchange: Optional[str] = None


@dataclass
class QuoteEvent(StreamEvent):
    """Real-time quote update."""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: int = 0
    ask_size: int = 0
    mid_price: float = 0.0

    def __post_init__(self):
        self.event_type = StreamEventType.QUOTE
        if self.bid > 0 and self.ask > 0:
            self.mid_price = (self.bid + self.ask) / 2


@dataclass
class TradeEvent(StreamEvent):
    """Real-time trade execution."""
    price: float = 0.0
    size: int = 0
    side: Optional[str] = None  # 'buy' or 'sell'
    trade_id: Optional[str] = None

    def __post_init__(self):
        self.event_type = StreamEventType.TRADE


@dataclass
class BarEvent(StreamEvent):
    """Real-time OHLCV bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: int = 0
    vwap: Optional[float] = None
    bar_start: Optional[datetime] = None
    bar_end: Optional[datetime] = None

    def __post_init__(self):
        self.event_type = StreamEventType.BAR


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: int
    order_count: int = 1


@dataclass
class OrderBookEvent(StreamEvent):
    """Order book update."""
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    is_snapshot: bool = False

    def __post_init__(self):
        self.event_type = StreamEventType.ORDER_BOOK


EventHandler = Callable[[StreamEvent], None]
AsyncEventHandler = Callable[[StreamEvent], Coroutine[Any, Any, None]]


class StreamSubscription:
    """Represents a subscription to streaming data."""

    def __init__(
        self,
        symbols: List[str],
        event_types: List[StreamEventType],
        handler: Union[EventHandler, AsyncEventHandler],
        subscription_id: Optional[str] = None
    ):
        """
        Initialize subscription.

        Args:
            symbols: List of symbols to subscribe to
            event_types: Event types to receive
            handler: Callback function for events
            subscription_id: Optional custom ID
        """
        self.symbols = set(symbols)
        self.event_types = set(event_types)
        self.handler = handler
        self.subscription_id = subscription_id or f"sub_{int(time.time() * 1000)}"
        self.is_async = asyncio.iscoroutinefunction(handler)
        self.created_at = datetime.now()
        self.event_count = 0
        self.last_event_time: Optional[datetime] = None

    def matches(self, event: StreamEvent) -> bool:
        """Check if event matches this subscription."""
        return (
            event.symbol in self.symbols and
            event.event_type in self.event_types
        )

    async def dispatch(self, event: StreamEvent) -> None:
        """Dispatch event to handler."""
        if self.is_async:
            await self.handler(event)
        else:
            self.handler(event)
        self.event_count += 1
        self.last_event_time = datetime.now()


class DataStreamProvider(ABC):
    """Abstract base class for streaming data providers."""

    def __init__(self, name: str):
        """Initialize provider."""
        self.name = name
        self.state = ConnectionState.DISCONNECTED
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._symbol_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._callbacks: List[Callable[[StreamEvent], None]] = []
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 1.0
        self._last_heartbeat: Optional[datetime] = None

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the streaming service."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the streaming service."""
        pass

    @abstractmethod
    async def subscribe_symbols(self, symbols: List[str], event_types: List[StreamEventType]) -> bool:
        """Subscribe to symbols on the provider."""
        pass

    @abstractmethod
    async def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        pass

    def add_subscription(self, subscription: StreamSubscription) -> None:
        """Add a subscription."""
        self._subscriptions[subscription.subscription_id] = subscription
        for symbol in subscription.symbols:
            self._symbol_subscriptions[symbol].add(subscription.subscription_id)

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove a subscription."""
        if subscription_id in self._subscriptions:
            sub = self._subscriptions[subscription_id]
            for symbol in sub.symbols:
                self._symbol_subscriptions[symbol].discard(subscription_id)
            del self._subscriptions[subscription_id]

    async def dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch event to all matching subscriptions."""
        for sub_id in self._symbol_subscriptions.get(event.symbol, set()):
            sub = self._subscriptions.get(sub_id)
            if sub and sub.matches(event):
                try:
                    await sub.dispatch(event)
                except Exception as e:
                    logger.error(f"Error dispatching event to {sub_id}: {e}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self.state = ConnectionState.RECONNECTING

        while self._reconnect_attempts < self._max_reconnect_attempts:
            delay = self._reconnect_delay * (2 ** self._reconnect_attempts)
            logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts + 1})")
            await asyncio.sleep(delay)

            try:
                if await self.connect():
                    self._reconnect_attempts = 0
                    # Resubscribe to all symbols
                    all_symbols = list(self._symbol_subscriptions.keys())
                    if all_symbols:
                        await self.subscribe_symbols(
                            all_symbols,
                            [StreamEventType.QUOTE, StreamEventType.TRADE]
                        )
                    return
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

            self._reconnect_attempts += 1

        logger.error(f"Max reconnection attempts reached for {self.name}")
        self.state = ConnectionState.CLOSED


class SimulatedStreamProvider(DataStreamProvider):
    """
    Simulated streaming provider for testing and development.

    Generates realistic market data based on configured parameters.
    """

    def __init__(
        self,
        base_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.02,
        tick_interval: float = 0.1
    ):
        """
        Initialize simulated provider.

        Args:
            base_prices: Initial prices for symbols
            volatility: Price volatility (per tick)
            tick_interval: Seconds between ticks
        """
        super().__init__("simulated")
        self._base_prices = base_prices or {}
        self._current_prices: Dict[str, float] = {}
        self._volatility = volatility
        self._tick_interval = tick_interval
        self._running = False
        self._subscribed_symbols: Set[str] = set()
        self._task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Start the simulation."""
        self.state = ConnectionState.CONNECTED
        self._running = True
        logger.info("Simulated stream provider connected")
        return True

    async def disconnect(self) -> None:
        """Stop the simulation."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state = ConnectionState.DISCONNECTED
        logger.info("Simulated stream provider disconnected")

    async def subscribe_symbols(
        self,
        symbols: List[str],
        event_types: List[StreamEventType]
    ) -> bool:
        """Subscribe to simulated symbols."""
        for symbol in symbols:
            self._subscribed_symbols.add(symbol)
            if symbol not in self._current_prices:
                self._current_prices[symbol] = self._base_prices.get(symbol, 100.0)

        # Start streaming if not already running
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._stream_loop())

        logger.info(f"Subscribed to simulated symbols: {symbols}")
        return True

    async def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Unsubscribe from simulated symbols."""
        for symbol in symbols:
            self._subscribed_symbols.discard(symbol)
        logger.info(f"Unsubscribed from simulated symbols: {symbols}")
        return True

    async def _stream_loop(self) -> None:
        """Main streaming loop for simulated data."""
        sequence = 0

        while self._running and self._subscribed_symbols:
            for symbol in list(self._subscribed_symbols):
                # Generate price movement (geometric Brownian motion)
                price = self._current_prices.get(symbol, 100.0)
                change = price * self._volatility * np.random.randn()
                new_price = max(price + change, 0.01)
                self._current_prices[symbol] = new_price

                # Generate spread (typically 0.01-0.05% for liquid stocks)
                spread = new_price * np.random.uniform(0.0001, 0.0005)
                bid = new_price - spread / 2
                ask = new_price + spread / 2

                # Generate quote event
                quote = QuoteEvent(
                    event_type=StreamEventType.QUOTE,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    data={'source': 'simulated'},
                    sequence=sequence,
                    bid=round(bid, 2),
                    ask=round(ask, 2),
                    bid_size=np.random.randint(100, 10000),
                    ask_size=np.random.randint(100, 10000)
                )

                await self.dispatch_event(quote)

                # Occasionally generate trade events
                if np.random.random() < 0.3:
                    trade_price = np.random.choice([bid, ask, new_price])
                    trade = TradeEvent(
                        event_type=StreamEventType.TRADE,
                        symbol=symbol,
                        timestamp=datetime.now(),
                        data={'source': 'simulated'},
                        sequence=sequence,
                        price=round(trade_price, 2),
                        size=np.random.randint(1, 1000) * 100,
                        side='buy' if trade_price >= new_price else 'sell'
                    )
                    await self.dispatch_event(trade)

                sequence += 1

            await asyncio.sleep(self._tick_interval)


class PolygonStreamProvider(DataStreamProvider):
    """
    Polygon.io WebSocket streaming provider.

    Supports real-time quotes, trades, and aggregates for stocks.
    """

    def __init__(self, api_key: str, cluster: str = "stocks"):
        """
        Initialize Polygon stream.

        Args:
            api_key: Polygon.io API key
            cluster: Data cluster (stocks, options, forex, crypto)
        """
        super().__init__("polygon")
        self._api_key = api_key
        self._cluster = cluster
        self._ws_url = f"wss://socket.polygon.io/{cluster}"
        self._ws = None
        self._subscribed_symbols: Set[str] = set()

    async def connect(self) -> bool:
        """Connect to Polygon WebSocket."""
        try:
            # Import websockets only when needed
            import websockets

            self.state = ConnectionState.CONNECTING
            self._ws = await websockets.connect(self._ws_url)

            # Authenticate
            auth_msg = json.dumps({"action": "auth", "params": self._api_key})
            await self._ws.send(auth_msg)

            # Wait for auth response
            response = await self._ws.recv()
            data = json.loads(response)

            if isinstance(data, list) and data[0].get("status") == "auth_success":
                self.state = ConnectionState.CONNECTED
                logger.info("Connected to Polygon WebSocket")

                # Start message handler
                asyncio.create_task(self._message_loop())
                return True
            else:
                logger.error(f"Polygon auth failed: {data}")
                self.state = ConnectionState.DISCONNECTED
                return False

        except ImportError:
            logger.error("websockets package not installed")
            return False
        except Exception as e:
            logger.error(f"Polygon connection failed: {e}")
            self.state = ConnectionState.DISCONNECTED
            return False

    async def disconnect(self) -> None:
        """Disconnect from Polygon WebSocket."""
        if self._ws:
            await self._ws.close()
            self._ws = None
        self.state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from Polygon WebSocket")

    async def subscribe_symbols(
        self,
        symbols: List[str],
        event_types: List[StreamEventType]
    ) -> bool:
        """Subscribe to Polygon symbols."""
        if not self._ws or self.state != ConnectionState.CONNECTED:
            return False

        # Build subscription message
        channels = []
        for symbol in symbols:
            if StreamEventType.QUOTE in event_types:
                channels.append(f"Q.{symbol}")
            if StreamEventType.TRADE in event_types:
                channels.append(f"T.{symbol}")
            if StreamEventType.BAR in event_types:
                channels.append(f"AM.{symbol}")  # Aggregates per minute

        msg = json.dumps({
            "action": "subscribe",
            "params": ",".join(channels)
        })

        try:
            await self._ws.send(msg)
            self._subscribed_symbols.update(symbols)
            logger.info(f"Subscribed to Polygon: {channels}")
            return True
        except Exception as e:
            logger.error(f"Polygon subscribe failed: {e}")
            return False

    async def unsubscribe_symbols(self, symbols: List[str]) -> bool:
        """Unsubscribe from Polygon symbols."""
        if not self._ws:
            return False

        channels = []
        for symbol in symbols:
            channels.extend([f"Q.{symbol}", f"T.{symbol}", f"AM.{symbol}"])

        msg = json.dumps({
            "action": "unsubscribe",
            "params": ",".join(channels)
        })

        try:
            await self._ws.send(msg)
            self._subscribed_symbols.difference_update(symbols)
            return True
        except Exception as e:
            logger.error(f"Polygon unsubscribe failed: {e}")
            return False

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            while self._ws and self.state == ConnectionState.CONNECTED:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=30.0
                    )
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    self._last_heartbeat = datetime.now()
                except Exception as e:
                    logger.error(f"Error in message loop: {e}")
                    break

        except Exception as e:
            logger.error(f"Message loop failed: {e}")

        # Trigger reconnection if not intentionally disconnected
        if self.state == ConnectionState.CONNECTED:
            await self._reconnect()

    async def _process_message(self, message: str) -> None:
        """Process a Polygon WebSocket message."""
        try:
            data = json.loads(message)
            if not isinstance(data, list):
                return

            for item in data:
                event_type = item.get("ev")

                if event_type == "Q":  # Quote
                    event = QuoteEvent(
                        event_type=StreamEventType.QUOTE,
                        symbol=item.get("sym", ""),
                        timestamp=datetime.fromtimestamp(item.get("t", 0) / 1000),
                        data=item,
                        bid=item.get("bp", 0),
                        ask=item.get("ap", 0),
                        bid_size=item.get("bs", 0),
                        ask_size=item.get("as", 0)
                    )
                    await self.dispatch_event(event)

                elif event_type == "T":  # Trade
                    event = TradeEvent(
                        event_type=StreamEventType.TRADE,
                        symbol=item.get("sym", ""),
                        timestamp=datetime.fromtimestamp(item.get("t", 0) / 1000),
                        data=item,
                        price=item.get("p", 0),
                        size=item.get("s", 0),
                        trade_id=item.get("i")
                    )
                    await self.dispatch_event(event)

                elif event_type == "AM":  # Aggregate minute
                    event = BarEvent(
                        event_type=StreamEventType.BAR,
                        symbol=item.get("sym", ""),
                        timestamp=datetime.fromtimestamp(item.get("e", 0) / 1000),
                        data=item,
                        open=item.get("o", 0),
                        high=item.get("h", 0),
                        low=item.get("l", 0),
                        close=item.get("c", 0),
                        volume=item.get("v", 0),
                        vwap=item.get("vw")
                    )
                    await self.dispatch_event(event)

        except Exception as e:
            logger.error(f"Error processing Polygon message: {e}")


class StreamAggregator:
    """
    Aggregates streaming data into bars.

    Converts tick data into time-based or volume-based bars.
    """

    def __init__(
        self,
        bar_size_seconds: int = 60,
        emit_callback: Optional[Callable[[BarEvent], None]] = None
    ):
        """
        Initialize aggregator.

        Args:
            bar_size_seconds: Bar duration in seconds
            emit_callback: Callback when a bar is complete
        """
        self.bar_size_seconds = bar_size_seconds
        self.emit_callback = emit_callback
        self._current_bars: Dict[str, Dict[str, Any]] = {}
        self._bar_start_times: Dict[str, datetime] = {}

    def process_trade(self, trade: TradeEvent) -> Optional[BarEvent]:
        """
        Process a trade event and potentially emit a bar.

        Args:
            trade: Trade event

        Returns:
            Completed bar if one was finalized
        """
        symbol = trade.symbol
        price = trade.price
        size = trade.size
        timestamp = trade.timestamp

        # Calculate bar start time
        bar_start = timestamp.replace(
            second=(timestamp.second // self.bar_size_seconds) * self.bar_size_seconds,
            microsecond=0
        )

        # Check if we need to finalize the current bar
        completed_bar = None
        if symbol in self._bar_start_times:
            if bar_start > self._bar_start_times[symbol]:
                completed_bar = self._finalize_bar(symbol)

        # Initialize or update bar
        if symbol not in self._current_bars or completed_bar:
            self._current_bars[symbol] = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size,
                'trade_count': 1,
                'vwap_sum': price * size
            }
            self._bar_start_times[symbol] = bar_start
        else:
            bar = self._current_bars[symbol]
            bar['high'] = max(bar['high'], price)
            bar['low'] = min(bar['low'], price)
            bar['close'] = price
            bar['volume'] += size
            bar['trade_count'] += 1
            bar['vwap_sum'] += price * size

        return completed_bar

    def _finalize_bar(self, symbol: str) -> Optional[BarEvent]:
        """Finalize and emit the current bar."""
        if symbol not in self._current_bars:
            return None

        bar = self._current_bars[symbol]
        bar_start = self._bar_start_times[symbol]
        bar_end = bar_start + timedelta(seconds=self.bar_size_seconds)

        vwap = bar['vwap_sum'] / bar['volume'] if bar['volume'] > 0 else bar['close']

        event = BarEvent(
            event_type=StreamEventType.BAR,
            symbol=symbol,
            timestamp=bar_end,
            data={'trade_count': bar['trade_count']},
            open=bar['open'],
            high=bar['high'],
            low=bar['low'],
            close=bar['close'],
            volume=bar['volume'],
            vwap=vwap,
            bar_start=bar_start,
            bar_end=bar_end
        )

        if self.emit_callback:
            self.emit_callback(event)

        return event

    def flush_all(self) -> List[BarEvent]:
        """Flush all pending bars."""
        bars = []
        for symbol in list(self._current_bars.keys()):
            bar = self._finalize_bar(symbol)
            if bar:
                bars.append(bar)
        self._current_bars.clear()
        self._bar_start_times.clear()
        return bars


class StreamBuffer:
    """
    Buffer for streaming events with batch processing.

    Accumulates events and flushes them periodically or when buffer is full.
    """

    def __init__(
        self,
        max_size: int = 1000,
        flush_interval_seconds: float = 1.0,
        flush_callback: Optional[Callable[[List[StreamEvent]], None]] = None
    ):
        """
        Initialize buffer.

        Args:
            max_size: Maximum events before auto-flush
            flush_interval_seconds: Time-based flush interval
            flush_callback: Callback when flushing
        """
        self.max_size = max_size
        self.flush_interval = flush_interval_seconds
        self.flush_callback = flush_callback
        self._buffer: List[StreamEvent] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()

    async def add(self, event: StreamEvent) -> None:
        """Add an event to the buffer."""
        async with self._lock:
            self._buffer.append(event)

            if len(self._buffer) >= self.max_size:
                await self._flush()
            elif time.time() - self._last_flush >= self.flush_interval:
                await self._flush()

    async def _flush(self) -> None:
        """Flush the buffer."""
        if not self._buffer:
            return

        events = self._buffer.copy()
        self._buffer.clear()
        self._last_flush = time.time()

        if self.flush_callback:
            try:
                self.flush_callback(events)
            except Exception as e:
                logger.error(f"Error in flush callback: {e}")

    async def flush(self) -> List[StreamEvent]:
        """Manually flush and return events."""
        async with self._lock:
            events = self._buffer.copy()
            self._buffer.clear()
            self._last_flush = time.time()
            return events


class StreamManager:
    """
    Central manager for all streaming operations.

    Coordinates multiple providers, subscriptions, and data processing.
    """

    def __init__(self):
        """Initialize stream manager."""
        self._providers: Dict[str, DataStreamProvider] = {}
        self._subscriptions: Dict[str, StreamSubscription] = {}
        self._aggregator: Optional[StreamAggregator] = None
        self._buffer: Optional[StreamBuffer] = None
        self._running = False

    def register_provider(self, provider: DataStreamProvider) -> None:
        """Register a streaming provider."""
        self._providers[provider.name] = provider
        logger.info(f"Registered stream provider: {provider.name}")

    def set_aggregator(self, aggregator: StreamAggregator) -> None:
        """Set the bar aggregator."""
        self._aggregator = aggregator

    def set_buffer(self, buffer: StreamBuffer) -> None:
        """Set the event buffer."""
        self._buffer = buffer

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all registered providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.connect()
            except Exception as e:
                logger.error(f"Failed to connect {name}: {e}")
                results[name] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect from all providers."""
        for provider in self._providers.values():
            try:
                await provider.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting {provider.name}: {e}")

    async def subscribe(
        self,
        symbols: List[str],
        event_types: List[StreamEventType],
        handler: Union[EventHandler, AsyncEventHandler],
        provider_name: Optional[str] = None
    ) -> str:
        """
        Subscribe to streaming data.

        Args:
            symbols: Symbols to subscribe to
            event_types: Event types to receive
            handler: Callback for events
            provider_name: Specific provider (or None for all)

        Returns:
            Subscription ID
        """
        subscription = StreamSubscription(symbols, event_types, handler)
        self._subscriptions[subscription.subscription_id] = subscription

        # Add to providers
        providers = (
            [self._providers[provider_name]]
            if provider_name
            else list(self._providers.values())
        )

        for provider in providers:
            provider.add_subscription(subscription)
            if provider.state == ConnectionState.CONNECTED:
                await provider.subscribe_symbols(symbols, event_types)

        logger.info(f"Created subscription {subscription.subscription_id}")
        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from streaming data.

        Args:
            subscription_id: ID of subscription to cancel

        Returns:
            True if successful
        """
        if subscription_id not in self._subscriptions:
            return False

        subscription = self._subscriptions[subscription_id]

        for provider in self._providers.values():
            provider.remove_subscription(subscription_id)

        del self._subscriptions[subscription_id]
        logger.info(f"Removed subscription {subscription_id}")
        return True

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers."""
        status = {}
        for name, provider in self._providers.items():
            status[name] = {
                'state': provider.state.value,
                'subscriptions': len(provider._subscriptions),
                'symbols': len(provider._symbol_subscriptions),
                'last_heartbeat': provider._last_heartbeat
            }
        return status

    def get_subscription_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all subscriptions."""
        stats = {}
        for sub_id, sub in self._subscriptions.items():
            stats[sub_id] = {
                'symbols': list(sub.symbols),
                'event_types': [e.value for e in sub.event_types],
                'event_count': sub.event_count,
                'created_at': sub.created_at,
                'last_event': sub.last_event_time
            }
        return stats
