"""
Alternative Data Integration Module.

This module provides integration with alternative data sources:
- Economic indicators (FRED, BLS, Census)
- Fundamental data (SEC filings, financial statements)
- Sentiment data (news, social media)
- Corporate events (earnings, dividends, splits)
- Macro data (interest rates, inflation, employment)

These data sources complement market data for quantitative analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories of alternative data."""
    ECONOMIC = "economic"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    CORPORATE_EVENTS = "corporate_events"
    MACRO = "macro"
    WEATHER = "weather"
    SATELLITE = "satellite"


class DataFrequency(Enum):
    """Update frequency for data series."""
    REALTIME = "realtime"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


@dataclass
class DataSeriesMetadata:
    """Metadata for an alternative data series."""
    series_id: str
    name: str
    category: DataCategory
    frequency: DataFrequency
    source: str
    description: str = ""
    units: str = ""
    seasonal_adjustment: bool = False
    first_observation: Optional[date] = None
    last_observation: Optional[date] = None
    update_schedule: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class DataObservation:
    """Single observation in a data series."""
    date: date
    value: float
    series_id: str
    revision_date: Optional[datetime] = None
    is_preliminary: bool = False
    notes: Optional[str] = None


class AlternativeDataProvider(ABC):
    """Abstract base class for alternative data providers."""

    def __init__(self, name: str):
        """Initialize provider."""
        self.name = name

    @abstractmethod
    def get_series_metadata(self, series_id: str) -> Optional[DataSeriesMetadata]:
        """Get metadata for a data series."""
        pass

    @abstractmethod
    def get_observations(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DataObservation]:
        """Get observations for a data series."""
        pass

    @abstractmethod
    def search_series(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 100
    ) -> List[DataSeriesMetadata]:
        """Search for data series."""
        pass


class FREDProvider(AlternativeDataProvider):
    """
    Federal Reserve Economic Data (FRED) provider.

    Provides access to economic indicators from the St. Louis Fed.
    Common series: GDP, unemployment, inflation, interest rates.
    """

    # Common FRED series IDs
    SERIES_GDP = "GDP"
    SERIES_UNRATE = "UNRATE"  # Unemployment rate
    SERIES_CPIAUCSL = "CPIAUCSL"  # Consumer Price Index
    SERIES_FEDFUNDS = "FEDFUNDS"  # Federal Funds Rate
    SERIES_T10Y2Y = "T10Y2Y"  # 10Y-2Y Treasury Spread
    SERIES_VIXCLS = "VIXCLS"  # VIX
    SERIES_DGS10 = "DGS10"  # 10-Year Treasury
    SERIES_DGS2 = "DGS2"  # 2-Year Treasury

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED provider.

        Args:
            api_key: FRED API key
        """
        super().__init__("fred")
        self._api_key = api_key
        self._base_url = "https://api.stlouisfed.org/fred"

        # Cache for metadata
        self._metadata_cache: Dict[str, DataSeriesMetadata] = {}

    def get_series_metadata(self, series_id: str) -> Optional[DataSeriesMetadata]:
        """Get metadata for a FRED series."""
        if series_id in self._metadata_cache:
            return self._metadata_cache[series_id]

        # Known series metadata
        known_series = {
            "GDP": DataSeriesMetadata(
                series_id="GDP",
                name="Gross Domestic Product",
                category=DataCategory.ECONOMIC,
                frequency=DataFrequency.QUARTERLY,
                source="U.S. Bureau of Economic Analysis",
                description="Nominal GDP in billions of dollars",
                units="Billions of Dollars",
                seasonal_adjustment=True,
                tags=["gdp", "national accounts", "output"]
            ),
            "UNRATE": DataSeriesMetadata(
                series_id="UNRATE",
                name="Unemployment Rate",
                category=DataCategory.ECONOMIC,
                frequency=DataFrequency.MONTHLY,
                source="U.S. Bureau of Labor Statistics",
                description="Civilian unemployment rate, seasonally adjusted",
                units="Percent",
                seasonal_adjustment=True,
                tags=["unemployment", "labor market", "employment"]
            ),
            "CPIAUCSL": DataSeriesMetadata(
                series_id="CPIAUCSL",
                name="Consumer Price Index for All Urban Consumers",
                category=DataCategory.ECONOMIC,
                frequency=DataFrequency.MONTHLY,
                source="U.S. Bureau of Labor Statistics",
                description="CPI for all urban consumers, all items",
                units="Index 1982-1984=100",
                seasonal_adjustment=True,
                tags=["inflation", "prices", "cpi"]
            ),
            "FEDFUNDS": DataSeriesMetadata(
                series_id="FEDFUNDS",
                name="Federal Funds Effective Rate",
                category=DataCategory.MACRO,
                frequency=DataFrequency.DAILY,
                source="Board of Governors of the Federal Reserve System",
                description="Federal funds effective rate",
                units="Percent",
                seasonal_adjustment=False,
                tags=["interest rates", "monetary policy", "fed"]
            ),
            "T10Y2Y": DataSeriesMetadata(
                series_id="T10Y2Y",
                name="10-Year Treasury Minus 2-Year Treasury",
                category=DataCategory.MACRO,
                frequency=DataFrequency.DAILY,
                source="Federal Reserve Bank of St. Louis",
                description="Yield curve spread",
                units="Percent",
                seasonal_adjustment=False,
                tags=["yield curve", "treasury", "spread"]
            ),
            "VIXCLS": DataSeriesMetadata(
                series_id="VIXCLS",
                name="CBOE Volatility Index: VIX",
                category=DataCategory.MACRO,
                frequency=DataFrequency.DAILY,
                source="Chicago Board Options Exchange",
                description="Market expectation of 30-day volatility",
                units="Index",
                seasonal_adjustment=False,
                tags=["volatility", "vix", "options"]
            )
        }

        if series_id in known_series:
            self._metadata_cache[series_id] = known_series[series_id]
            return known_series[series_id]

        logger.warning(f"Unknown FRED series: {series_id}")
        return None

    def get_observations(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DataObservation]:
        """
        Get observations for a FRED series.

        In production, this would call the FRED API.
        For now, returns synthetic data for testing.
        """
        if not self._api_key:
            logger.warning("No FRED API key, returning synthetic data")
            return self._generate_synthetic_data(series_id, start_date, end_date)

        # TODO: Implement actual FRED API call
        # import requests
        # params = {
        #     "series_id": series_id,
        #     "api_key": self._api_key,
        #     "file_type": "json",
        #     "observation_start": start_date.isoformat() if start_date else None,
        #     "observation_end": end_date.isoformat() if end_date else None
        # }
        # response = requests.get(f"{self._base_url}/series/observations", params=params)

        return self._generate_synthetic_data(series_id, start_date, end_date)

    def _generate_synthetic_data(
        self,
        series_id: str,
        start_date: Optional[date],
        end_date: Optional[date]
    ) -> List[DataObservation]:
        """Generate synthetic data for testing."""
        if not start_date:
            start_date = date.today() - timedelta(days=365)
        if not end_date:
            end_date = date.today()

        metadata = self.get_series_metadata(series_id)
        if not metadata:
            return []

        # Generate dates based on frequency
        dates = pd.date_range(start=start_date, end=end_date, freq='D').date.tolist()

        if metadata.frequency == DataFrequency.MONTHLY:
            dates = pd.date_range(start=start_date, end=end_date, freq='MS').date.tolist()
        elif metadata.frequency == DataFrequency.QUARTERLY:
            dates = pd.date_range(start=start_date, end=end_date, freq='QS').date.tolist()

        # Generate synthetic values based on series type
        base_values = {
            "GDP": (20000, 500),  # Mean, std
            "UNRATE": (5.0, 1.0),
            "CPIAUCSL": (280, 10),
            "FEDFUNDS": (4.0, 0.5),
            "T10Y2Y": (0.5, 0.3),
            "VIXCLS": (20, 5)
        }

        mean, std = base_values.get(series_id, (100, 10))

        observations = []
        current_value = mean

        for d in dates:
            # Random walk with mean reversion
            change = np.random.normal(0, std * 0.1)
            reversion = (mean - current_value) * 0.05
            current_value += change + reversion

            observations.append(DataObservation(
                date=d,
                value=round(current_value, 2),
                series_id=series_id
            ))

        return observations

    def search_series(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 100
    ) -> List[DataSeriesMetadata]:
        """Search for FRED series."""
        # Return known series that match query
        all_series = [
            self.get_series_metadata("GDP"),
            self.get_series_metadata("UNRATE"),
            self.get_series_metadata("CPIAUCSL"),
            self.get_series_metadata("FEDFUNDS"),
            self.get_series_metadata("T10Y2Y"),
            self.get_series_metadata("VIXCLS")
        ]

        results = []
        query_lower = query.lower()

        for meta in all_series:
            if meta is None:
                continue

            if category and meta.category != category:
                continue

            # Check if query matches name, description, or tags
            if (query_lower in meta.name.lower() or
                query_lower in meta.description.lower() or
                any(query_lower in tag for tag in meta.tags)):
                results.append(meta)

        return results[:limit]


@dataclass
class EarningsEvent:
    """Earnings announcement event."""
    symbol: str
    report_date: date
    fiscal_quarter: str  # e.g., "Q1 2024"
    fiscal_year: int
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    surprise_pct: Optional[float] = None
    time_of_day: str = "after_close"  # before_open, after_close, during_market


@dataclass
class DividendEvent:
    """Dividend event."""
    symbol: str
    ex_date: date
    record_date: date
    payment_date: date
    amount: float
    dividend_type: str = "regular"  # regular, special, qualified
    frequency: str = "quarterly"  # monthly, quarterly, semi-annual, annual


@dataclass
class SplitEvent:
    """Stock split event."""
    symbol: str
    ex_date: date
    ratio_from: int  # e.g., 1 (for 4:1 split)
    ratio_to: int  # e.g., 4 (for 4:1 split)
    split_type: str = "forward"  # forward, reverse


class CorporateEventsProvider(AlternativeDataProvider):
    """
    Provider for corporate events data.

    Covers earnings, dividends, splits, and other corporate actions.
    """

    def __init__(self):
        """Initialize corporate events provider."""
        super().__init__("corporate_events")

    def get_series_metadata(self, series_id: str) -> Optional[DataSeriesMetadata]:
        """Corporate events don't have traditional series metadata."""
        return None

    def get_observations(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DataObservation]:
        """Not applicable for events."""
        return []

    def search_series(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 100
    ) -> List[DataSeriesMetadata]:
        """Not applicable for events."""
        return []

    def get_earnings_calendar(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[EarningsEvent]:
        """
        Get earnings calendar for date range.

        Args:
            start_date: Start date
            end_date: End date
            symbols: Filter by symbols (None for all)

        Returns:
            List of earnings events
        """
        # In production, would call an API like Alpha Vantage or IEX
        logger.info(f"Fetching earnings calendar: {start_date} to {end_date}")

        # Return synthetic data for testing
        events = []
        test_symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        for symbol in test_symbols:
            # Generate quarterly earnings
            for q in range(1, 5):
                report_date = date(
                    end_date.year,
                    q * 3,
                    15 + np.random.randint(0, 15)
                )
                if start_date <= report_date <= end_date:
                    eps_estimate = round(np.random.uniform(1.0, 5.0), 2)
                    eps_actual = round(eps_estimate * np.random.uniform(0.95, 1.15), 2)

                    events.append(EarningsEvent(
                        symbol=symbol,
                        report_date=report_date,
                        fiscal_quarter=f"Q{q} {end_date.year}",
                        fiscal_year=end_date.year,
                        eps_estimate=eps_estimate,
                        eps_actual=eps_actual,
                        surprise_pct=round((eps_actual - eps_estimate) / eps_estimate * 100, 2),
                        time_of_day="after_close"
                    ))

        return sorted(events, key=lambda e: e.report_date)

    def get_dividend_calendar(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[DividendEvent]:
        """
        Get dividend calendar for date range.

        Args:
            start_date: Start date
            end_date: End date
            symbols: Filter by symbols

        Returns:
            List of dividend events
        """
        logger.info(f"Fetching dividend calendar: {start_date} to {end_date}")

        events = []
        test_symbols = symbols or ["AAPL", "MSFT", "JNJ", "PG", "KO"]

        for symbol in test_symbols:
            # Generate quarterly dividends
            current_date = start_date
            while current_date <= end_date:
                ex_date = current_date + timedelta(days=np.random.randint(0, 30))
                if ex_date > end_date:
                    break

                events.append(DividendEvent(
                    symbol=symbol,
                    ex_date=ex_date,
                    record_date=ex_date + timedelta(days=1),
                    payment_date=ex_date + timedelta(days=30),
                    amount=round(np.random.uniform(0.20, 1.00), 2),
                    dividend_type="regular",
                    frequency="quarterly"
                ))

                current_date += timedelta(days=90)

        return sorted(events, key=lambda e: e.ex_date)

    def get_splits(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[List[str]] = None
    ) -> List[SplitEvent]:
        """Get stock splits for date range."""
        logger.info(f"Fetching splits: {start_date} to {end_date}")
        # Splits are rare, return empty for testing
        return []


@dataclass
class SentimentScore:
    """Sentiment score for a symbol."""
    symbol: str
    timestamp: datetime
    score: float  # -1 (very negative) to +1 (very positive)
    magnitude: float  # Strength of sentiment (0 to 1)
    source: str
    article_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0


class SentimentProvider(AlternativeDataProvider):
    """
    Provider for sentiment data.

    Aggregates sentiment from news and other text sources.
    """

    def __init__(self):
        """Initialize sentiment provider."""
        super().__init__("sentiment")

    def get_series_metadata(self, series_id: str) -> Optional[DataSeriesMetadata]:
        """Not applicable for sentiment."""
        return None

    def get_observations(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DataObservation]:
        """Not applicable for sentiment."""
        return []

    def search_series(
        self,
        query: str,
        category: Optional[DataCategory] = None,
        limit: int = 100
    ) -> List[DataSeriesMetadata]:
        """Not applicable for sentiment."""
        return []

    def get_sentiment(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[SentimentScore]:
        """
        Get sentiment scores for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            List of sentiment scores
        """
        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()

        logger.info(f"Fetching sentiment for {symbol}: {start_date} to {end_date}")

        # Generate synthetic sentiment data
        scores = []
        current_date = start_date
        sentiment_momentum = 0.0

        while current_date <= end_date:
            # Random walk for sentiment with mean reversion
            change = np.random.normal(0, 0.1)
            reversion = -sentiment_momentum * 0.1
            sentiment_momentum += change + reversion
            sentiment_momentum = np.clip(sentiment_momentum, -1, 1)

            article_count = np.random.randint(5, 50)
            positive_ratio = (sentiment_momentum + 1) / 2

            scores.append(SentimentScore(
                symbol=symbol,
                timestamp=datetime.combine(current_date, datetime.min.time()),
                score=round(sentiment_momentum, 3),
                magnitude=round(abs(sentiment_momentum), 3),
                source="aggregated",
                article_count=article_count,
                positive_count=int(article_count * positive_ratio),
                negative_count=int(article_count * (1 - positive_ratio) * 0.7),
                neutral_count=int(article_count * (1 - positive_ratio) * 0.3)
            ))

            current_date += timedelta(days=1)

        return scores

    def get_aggregate_sentiment(
        self,
        symbols: List[str],
        as_of: Optional[date] = None
    ) -> Dict[str, SentimentScore]:
        """
        Get current aggregate sentiment for multiple symbols.

        Args:
            symbols: List of symbols
            as_of: Date for sentiment (default: today)

        Returns:
            Dictionary of symbol to sentiment score
        """
        as_of = as_of or date.today()
        result = {}

        for symbol in symbols:
            scores = self.get_sentiment(symbol, as_of, as_of)
            if scores:
                result[symbol] = scores[-1]

        return result


class AlternativeDataManager:
    """
    Central manager for alternative data sources.

    Coordinates multiple providers and provides unified access.
    """

    def __init__(self):
        """Initialize alternative data manager."""
        self._providers: Dict[str, AlternativeDataProvider] = {}
        self._fred: Optional[FREDProvider] = None
        self._corporate: Optional[CorporateEventsProvider] = None
        self._sentiment: Optional[SentimentProvider] = None

    def register_fred(self, api_key: Optional[str] = None) -> None:
        """Register FRED provider."""
        self._fred = FREDProvider(api_key)
        self._providers["fred"] = self._fred
        logger.info("Registered FRED provider")

    def register_corporate_events(self) -> None:
        """Register corporate events provider."""
        self._corporate = CorporateEventsProvider()
        self._providers["corporate_events"] = self._corporate
        logger.info("Registered corporate events provider")

    def register_sentiment(self) -> None:
        """Register sentiment provider."""
        self._sentiment = SentimentProvider()
        self._providers["sentiment"] = self._sentiment
        logger.info("Registered sentiment provider")

    def get_economic_data(
        self,
        series_ids: List[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get economic data for multiple series.

        Args:
            series_ids: FRED series IDs
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date index and series columns
        """
        if not self._fred:
            raise ValueError("FRED provider not registered")

        all_data = {}

        for series_id in series_ids:
            observations = self._fred.get_observations(series_id, start_date, end_date)
            if observations:
                all_data[series_id] = {
                    obs.date: obs.value for obs in observations
                }

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def get_upcoming_events(
        self,
        symbols: List[str],
        days_ahead: int = 30
    ) -> Dict[str, List[Any]]:
        """
        Get upcoming corporate events for symbols.

        Args:
            symbols: List of symbols
            days_ahead: Number of days to look ahead

        Returns:
            Dictionary with event types and lists
        """
        if not self._corporate:
            raise ValueError("Corporate events provider not registered")

        today = date.today()
        end_date = today + timedelta(days=days_ahead)

        return {
            'earnings': self._corporate.get_earnings_calendar(today, end_date, symbols),
            'dividends': self._corporate.get_dividend_calendar(today, end_date, symbols),
            'splits': self._corporate.get_splits(today, end_date, symbols)
        }

    def get_market_sentiment(
        self,
        symbols: List[str],
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Get sentiment data for symbols.

        Args:
            symbols: List of symbols
            lookback_days: Number of days of history

        Returns:
            DataFrame with sentiment scores
        """
        if not self._sentiment:
            raise ValueError("Sentiment provider not registered")

        start_date = date.today() - timedelta(days=lookback_days)
        end_date = date.today()

        records = []
        for symbol in symbols:
            scores = self._sentiment.get_sentiment(symbol, start_date, end_date)
            for score in scores:
                records.append({
                    'symbol': symbol,
                    'date': score.timestamp.date(),
                    'sentiment_score': score.score,
                    'sentiment_magnitude': score.magnitude,
                    'article_count': score.article_count
                })

        return pd.DataFrame(records)

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all registered providers."""
        return {
            name: True for name in self._providers.keys()
        }
