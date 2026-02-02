"""
Data Provider Abstraction Layer.

Provides unified interface for multiple market data sources:
- Yahoo Finance (free, delayed)
- Alpha Vantage (free tier with rate limits)
- Polygon.io (paid, real-time)
- IEX Cloud (paid, real-time)

Each provider implements the DataProvider interface with
rate limiting, retry logic, and standardized output format.

Reference: Design doc Section 5.1 (Data Sources)
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import requests

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataFrequency(Enum):
    """Data frequency enumeration."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class DataType(Enum):
    """Type of market data."""
    OHLCV = "ohlcv"
    QUOTE = "quote"
    TRADE = "trade"
    OPTIONS = "options"
    FUNDAMENTAL = "fundamental"


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Attributes:
        calls_per_minute: Maximum calls per minute
        calls_per_day: Maximum calls per day
        min_interval: Minimum seconds between calls
    """
    calls_per_minute: int = 5
    calls_per_day: int = 500
    min_interval: float = 0.1

    _minute_calls: List[float] = field(default_factory=list)
    _day_calls: List[float] = field(default_factory=list)
    _last_call: float = 0.0

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Enforce minimum interval
        if self._last_call > 0:
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
                now = time.time()

        # Clean old minute calls
        minute_ago = now - 60
        self._minute_calls = [t for t in self._minute_calls if t > minute_ago]

        # Wait if minute limit reached
        if len(self._minute_calls) >= self.calls_per_minute:
            wait_time = self._minute_calls[0] - minute_ago
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                now = time.time()

        # Clean old day calls
        day_ago = now - 86400
        self._day_calls = [t for t in self._day_calls if t > day_ago]

        # Check daily limit
        if len(self._day_calls) >= self.calls_per_day:
            raise RateLimitExceeded("Daily rate limit exceeded")

        # Record this call
        self._minute_calls.append(now)
        self._day_calls.append(now)
        self._last_call = now

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._minute_calls = []
        self._day_calls = []
        self._last_call = 0.0


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


class DataProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class DataProvider(ABC):
    """
    Abstract base class for data providers.

    All providers must implement methods to fetch:
    - Historical OHLCV data
    - Real-time quotes
    - Options chain data (if supported)

    Output is standardized to pandas DataFrames with
    consistent column names:
    - timestamp (index)
    - open, high, low, close, volume
    - symbol
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize provider.

        Args:
            api_key: API key for authenticated access
            rate_limiter: Rate limiter configuration
        """
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self) -> None:
        """Configure HTTP session with retries."""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    @abstractmethod
    def supports_real_time(self) -> bool:
        """Whether provider supports real-time data."""
        pass

    @property
    @abstractmethod
    def supports_options(self) -> bool:
        """Whether provider supports options data."""
        pass

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency

        Returns:
            DataFrame with columns: open, high, low, close, volume, symbol
            Index: timestamp (datetime)
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time or delayed quote.

        Args:
            symbol: Ticker symbol

        Returns:
            Dict with: bid, ask, last, volume, timestamp
        """
        pass

    def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for multiple symbols.

        Default implementation calls get_quote for each symbol.
        Subclasses may override with batch API calls.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to quote dict
        """
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                quotes[symbol] = {"error": str(e)}
        return quotes

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get options chain data.

        Args:
            symbol: Underlying symbol
            expiration: Optional specific expiration date

        Returns:
            DataFrame with options chain data
        """
        raise NotImplementedError(f"{self.name} does not support options data")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame column names."""
        column_map = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume',
            'Date': 'timestamp',
            'Datetime': 'timestamp',
        }

        df = df.rename(columns=column_map)

        # Ensure required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = np.nan

        return df


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider.

    Free, delayed data. Good for historical data and EOD prices.
    Uses yfinance library.

    Rate Limits:
        - No official limits, but be respectful
        - Recommended: 2000 requests/hour
    """

    @property
    def name(self) -> str:
        return "yahoo_finance"

    @property
    def supports_real_time(self) -> bool:
        return False  # 15-20 minute delay

    @property
    def supports_options(self) -> bool:
        return True

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Get historical data from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        self.rate_limiter.wait_if_needed()

        # Map frequency to yfinance interval
        interval_map = {
            DataFrequency.MINUTE_1: "1m",
            DataFrequency.MINUTE_5: "5m",
            DataFrequency.MINUTE_15: "15m",
            DataFrequency.MINUTE_30: "30m",
            DataFrequency.HOUR_1: "1h",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo",
        }

        interval = interval_map.get(frequency, "1d")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize columns
            df = self._standardize_columns(df)
            df['symbol'] = symbol
            df.index.name = 'timestamp'

            # Remove timezone info for consistency
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            logger.debug(f"Fetched {len(df)} bars for {symbol} from Yahoo")
            return df

        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise DataProviderError(f"Failed to fetch {symbol}: {e}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get delayed quote from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed")

        self.rate_limiter.wait_if_needed()

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'last': info.get('regularMarketPrice', 0),
                'volume': info.get('regularMarketVolume', 0),
                'timestamp': datetime.now(),
                'source': self.name,
            }

        except Exception as e:
            logger.error(f"Quote error for {symbol}: {e}")
            raise DataProviderError(f"Failed to get quote for {symbol}: {e}")

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> pd.DataFrame:
        """Get options chain from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed")

        self.rate_limiter.wait_if_needed()

        try:
            ticker = yf.Ticker(symbol)

            # Get available expirations
            expirations = ticker.options

            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return pd.DataFrame()

            # Select expiration
            if expiration:
                exp_str = expiration.strftime('%Y-%m-%d')
                if exp_str not in expirations:
                    # Find nearest
                    exp_str = min(expirations, key=lambda x: abs(
                        datetime.strptime(x, '%Y-%m-%d').date() - expiration
                    ))
            else:
                exp_str = expirations[0]

            # Get chain
            chain = ticker.option_chain(exp_str)

            # Combine calls and puts
            calls = chain.calls.copy()
            calls['option_type'] = 'call'

            puts = chain.puts.copy()
            puts['option_type'] = 'put'

            df = pd.concat([calls, puts], ignore_index=True)

            # Standardize columns
            df = df.rename(columns={
                'strike': 'strike',
                'lastPrice': 'last',
                'bid': 'bid',
                'ask': 'ask',
                'volume': 'volume',
                'openInterest': 'open_interest',
                'impliedVolatility': 'implied_vol',
            })

            df['underlying'] = symbol
            df['expiration'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
            df['timestamp'] = datetime.now()

            logger.debug(f"Fetched {len(df)} options for {symbol} exp {exp_str}")
            return df

        except Exception as e:
            logger.error(f"Options chain error for {symbol}: {e}")
            raise DataProviderError(f"Failed to get options for {symbol}: {e}")


class AlphaVantageProvider(DataProvider):
    """
    Alpha Vantage data provider.

    Free tier: 5 calls/minute, 500 calls/day
    Premium: Higher limits

    Good for intraday data and fundamental data.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        """Initialize with required API key."""
        if not api_key:
            raise ValueError("Alpha Vantage requires an API key")

        # Free tier limits
        rate_limiter = RateLimiter(
            calls_per_minute=5,
            calls_per_day=500,
            min_interval=12.0  # 5 calls/min = 12s between calls
        )

        super().__init__(api_key=api_key, rate_limiter=rate_limiter)

    @property
    def name(self) -> str:
        return "alpha_vantage"

    @property
    def supports_real_time(self) -> bool:
        return False

    @property
    def supports_options(self) -> bool:
        return False

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Get historical data from Alpha Vantage."""
        self.rate_limiter.wait_if_needed()

        # Map frequency to API function
        if frequency in [DataFrequency.MINUTE_1, DataFrequency.MINUTE_5,
                        DataFrequency.MINUTE_15, DataFrequency.MINUTE_30,
                        DataFrequency.HOUR_1]:
            function = "TIME_SERIES_INTRADAY"
            interval_map = {
                DataFrequency.MINUTE_1: "1min",
                DataFrequency.MINUTE_5: "5min",
                DataFrequency.MINUTE_15: "15min",
                DataFrequency.MINUTE_30: "30min",
                DataFrequency.HOUR_1: "60min",
            }
            params = {
                "function": function,
                "symbol": symbol,
                "interval": interval_map[frequency],
                "outputsize": "full",
                "apikey": self.api_key,
            }
        else:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key,
            }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for error
            if "Error Message" in data:
                raise DataProviderError(data["Error Message"])

            if "Note" in data:
                raise RateLimitExceeded(data["Note"])

            # Find the time series key
            ts_key = None
            for key in data.keys():
                if "Time Series" in key:
                    ts_key = key
                    break

            if not ts_key:
                logger.warning(f"No time series data for {symbol}")
                return pd.DataFrame()

            # Parse to DataFrame
            df = pd.DataFrame.from_dict(data[ts_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'

            # Rename columns (remove numeric prefix)
            df.columns = [c.split('. ')[1] if '. ' in c else c for c in df.columns]
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'adjusted close': 'adj_close',
                'volume': 'volume',
            })

            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['symbol'] = symbol

            # Filter date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df.index >= start) & (df.index <= end)]

            df = df.sort_index()

            logger.debug(f"Fetched {len(df)} bars for {symbol} from Alpha Vantage")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request error: {e}")
            raise DataProviderError(f"Request failed: {e}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote from Alpha Vantage."""
        self.rate_limiter.wait_if_needed()

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Global Quote" not in data:
                raise DataProviderError(f"No quote data for {symbol}")

            quote = data["Global Quote"]

            return {
                'symbol': symbol,
                'last': float(quote.get('05. price', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_pct': float(quote.get('10. change percent', '0%').rstrip('%')),
                'timestamp': datetime.now(),
                'source': self.name,
            }

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"Quote request failed: {e}")


class PolygonProvider(DataProvider):
    """
    Polygon.io data provider.

    Paid service with real-time data.
    Excellent for options and tick data.

    Rate Limits (depends on plan):
        - Free: 5 calls/minute
        - Starter: unlimited
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str):
        """Initialize with API key."""
        if not api_key:
            raise ValueError("Polygon requires an API key")

        super().__init__(api_key=api_key)

    @property
    def name(self) -> str:
        return "polygon"

    @property
    def supports_real_time(self) -> bool:
        return True

    @property
    def supports_options(self) -> bool:
        return True

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Get historical data from Polygon."""
        self.rate_limiter.wait_if_needed()

        # Map frequency
        timespan_map = {
            DataFrequency.MINUTE_1: ("minute", 1),
            DataFrequency.MINUTE_5: ("minute", 5),
            DataFrequency.MINUTE_15: ("minute", 15),
            DataFrequency.MINUTE_30: ("minute", 30),
            DataFrequency.HOUR_1: ("hour", 1),
            DataFrequency.DAILY: ("day", 1),
            DataFrequency.WEEKLY: ("week", 1),
            DataFrequency.MONTHLY: ("month", 1),
        }

        timespan, multiplier = timespan_map.get(frequency, ("day", 1))

        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.api_key,
        }

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                raise DataProviderError(f"Polygon error: {data.get('error', 'Unknown')}")

            results = data.get("results", [])

            if not results:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(results)

            # Rename columns
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'trades',
            })

            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df['symbol'] = symbol

            logger.debug(f"Fetched {len(df)} bars for {symbol} from Polygon")
            return df

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"Polygon request failed: {e}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote from Polygon."""
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/v2/last/trade/{symbol}"
        params = {"apiKey": self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                raise DataProviderError(f"Quote error: {data.get('error')}")

            result = data.get("results", {})

            return {
                'symbol': symbol,
                'last': result.get('p', 0),
                'size': result.get('s', 0),
                'timestamp': datetime.fromtimestamp(
                    result.get('t', 0) / 1e9
                ) if result.get('t') else datetime.now(),
                'source': self.name,
            }

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"Quote request failed: {e}")

    def get_options_chain(
        self,
        symbol: str,
        expiration: Optional[date] = None
    ) -> pd.DataFrame:
        """Get options chain from Polygon."""
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/v3/reference/options/contracts"

        params = {
            "underlying_ticker": symbol,
            "limit": 1000,
            "apiKey": self.api_key,
        }

        if expiration:
            params["expiration_date"] = expiration.strftime('%Y-%m-%d')

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "OK":
                raise DataProviderError(f"Options error: {data.get('error')}")

            results = data.get("results", [])

            if not results:
                return pd.DataFrame()

            df = pd.DataFrame(results)

            # Standardize columns
            df = df.rename(columns={
                'strike_price': 'strike',
                'expiration_date': 'expiration',
                'contract_type': 'option_type',
            })

            df['underlying'] = symbol
            df['timestamp'] = datetime.now()

            return df

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"Options request failed: {e}")


class IEXCloudProvider(DataProvider):
    """
    IEX Cloud data provider.

    Paid service with real-time IEX data.
    Good for US equities and fundamental data.
    """

    BASE_URL = "https://cloud.iexapis.com/stable"

    def __init__(self, api_key: str):
        """Initialize with API key."""
        if not api_key:
            raise ValueError("IEX Cloud requires an API key")

        super().__init__(api_key=api_key)

    @property
    def name(self) -> str:
        return "iex_cloud"

    @property
    def supports_real_time(self) -> bool:
        return True  # IEX exchange data only

    @property
    def supports_options(self) -> bool:
        return False

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        frequency: DataFrequency = DataFrequency.DAILY
    ) -> pd.DataFrame:
        """Get historical data from IEX Cloud."""
        self.rate_limiter.wait_if_needed()

        # IEX uses range strings
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        days = (end - start).days

        if days <= 30:
            range_str = "1m"
        elif days <= 90:
            range_str = "3m"
        elif days <= 180:
            range_str = "6m"
        elif days <= 365:
            range_str = "1y"
        elif days <= 730:
            range_str = "2y"
        else:
            range_str = "5y"

        url = f"{self.BASE_URL}/stock/{symbol}/chart/{range_str}"
        params = {"token": self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Rename and format
            df = df.rename(columns={
                'date': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df['symbol'] = symbol

            # Filter date range
            df = df[(df.index >= start) & (df.index <= end)]

            return df

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"IEX request failed: {e}")

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get quote from IEX Cloud."""
        self.rate_limiter.wait_if_needed()

        url = f"{self.BASE_URL}/stock/{symbol}/quote"
        params = {"token": self.api_key}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                'symbol': symbol,
                'bid': data.get('iexBidPrice', 0),
                'ask': data.get('iexAskPrice', 0),
                'last': data.get('latestPrice', 0),
                'volume': data.get('volume', 0),
                'timestamp': datetime.fromtimestamp(
                    data.get('latestUpdate', 0) / 1000
                ) if data.get('latestUpdate') else datetime.now(),
                'source': self.name,
            }

        except requests.exceptions.RequestException as e:
            raise DataProviderError(f"Quote request failed: {e}")


class DataProviderFactory:
    """Factory for creating data provider instances."""

    _providers = {
        'yahoo': YahooFinanceProvider,
        'alpha_vantage': AlphaVantageProvider,
        'polygon': PolygonProvider,
        'iex': IEXCloudProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> DataProvider:
        """
        Create a data provider instance.

        Args:
            provider_name: Provider name ('yahoo', 'alpha_vantage', 'polygon', 'iex')
            api_key: API key if required
            **kwargs: Additional provider-specific arguments

        Returns:
            DataProvider instance
        """
        provider_class = cls._providers.get(provider_name.lower())

        if not provider_class:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {list(cls._providers.keys())}"
            )

        if provider_name.lower() == 'yahoo':
            return provider_class(**kwargs)
        else:
            return provider_class(api_key=api_key, **kwargs)

    @classmethod
    def register(cls, name: str, provider_class: type) -> None:
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class
