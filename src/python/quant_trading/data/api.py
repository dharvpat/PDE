"""
Data API Layer Module.

This module provides REST API endpoints for data access:
- Market data queries (historical, real-time)
- Options data and Greeks
- Reference data lookups
- Data quality metrics
- Aggregated data views

Designed for internal consumption by trading strategies and dashboards.
"""

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for API Request/Response
# ============================================================================

class TimeRange(str, Enum):
    """Predefined time ranges."""
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1m"
    MONTH_3 = "3m"
    MONTH_6 = "6m"
    YEAR_1 = "1y"
    YEAR_2 = "2y"
    YEAR_5 = "5y"
    MAX = "max"


class DataFrequency(str, Enum):
    """Data frequency options."""
    TICK = "tick"
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class OHLCVBar(BaseModel):
    """OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None


class QuoteData(BaseModel):
    """Real-time quote data."""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    volume: int
    change: float
    change_percent: float


class OptionQuoteData(BaseModel):
    """Option quote data."""
    symbol: str
    underlying: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: date
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


class SecurityInfo(BaseModel):
    """Security information."""
    symbol: str
    name: str
    asset_class: str
    exchange: str
    currency: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None


class HealthMetrics(BaseModel):
    """Data quality health metrics."""
    symbol: str
    health_score: float
    is_stale: bool
    last_update: Optional[datetime] = None
    update_count: int
    error_count: int
    gap_count: int


class HistoricalDataRequest(BaseModel):
    """Request for historical data."""
    symbols: List[str]
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    range: Optional[TimeRange] = None
    frequency: DataFrequency = DataFrequency.DAILY
    adjust_splits: bool = True
    include_extended_hours: bool = False


class OptionsChainRequest(BaseModel):
    """Request for options chain data."""
    underlying: str
    expiration: Optional[date] = None
    min_strike: Optional[float] = None
    max_strike: Optional[float] = None
    option_type: Optional[str] = None  # 'call', 'put', or None for both
    min_volume: int = 0
    calculate_greeks: bool = True


class DataQueryResponse(BaseModel):
    """Generic data query response."""
    success: bool
    data: Any
    metadata: Dict[str, Any] = {}
    errors: List[str] = []


# ============================================================================
# Data Service (Business Logic Layer)
# ============================================================================

class DataService:
    """
    Service layer for data operations.

    Coordinates data retrieval, caching, and transformations.
    """

    def __init__(self):
        """Initialize data service."""
        self._cache: Dict[str, Any] = {}
        # In production, these would be injected dependencies
        self._db_session = None
        self._providers = {}

    def get_historical_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        frequency: DataFrequency
    ) -> List[OHLCVBar]:
        """
        Get historical OHLCV bars.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            frequency: Bar frequency

        Returns:
            List of OHLCV bars
        """
        logger.info(f"Fetching bars: {symbol} {start_date} to {end_date} @ {frequency}")

        # In production, would query database or provider
        # For now, return synthetic data
        bars = []
        current_date = start_date
        price = 100.0

        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Generate realistic price movement
            import numpy as np
            change = np.random.normal(0, 0.02) * price
            price = max(1, price + change)

            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = low + np.random.random() * (high - low)
            close_price = low + np.random.random() * (high - low)

            bars.append(OHLCVBar(
                timestamp=datetime.combine(current_date, datetime.min.time()),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close_price, 2),
                volume=int(np.random.uniform(1e6, 1e8))
            ))

            current_date += timedelta(days=1)

        return bars

    def get_quote(self, symbol: str) -> Optional[QuoteData]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data or None
        """
        logger.info(f"Fetching quote: {symbol}")

        # Synthetic quote data
        import numpy as np
        price = 150.0 + np.random.normal(0, 5)
        spread = price * 0.0001

        return QuoteData(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=round(price - spread / 2, 2),
            ask=round(price + spread / 2, 2),
            bid_size=int(np.random.uniform(100, 10000)),
            ask_size=int(np.random.uniform(100, 10000)),
            last=round(price, 2),
            last_size=int(np.random.uniform(100, 1000)),
            volume=int(np.random.uniform(1e6, 1e8)),
            change=round(np.random.normal(0, 2), 2),
            change_percent=round(np.random.normal(0, 1), 2)
        )

    def get_options_chain(
        self,
        underlying: str,
        expiration: Optional[date] = None,
        min_strike: Optional[float] = None,
        max_strike: Optional[float] = None
    ) -> List[OptionQuoteData]:
        """
        Get options chain for an underlying.

        Args:
            underlying: Underlying symbol
            expiration: Filter by expiration
            min_strike: Minimum strike
            max_strike: Maximum strike

        Returns:
            List of option quotes
        """
        logger.info(f"Fetching options chain: {underlying}")

        # Synthetic options data
        import numpy as np

        spot = 150.0
        if min_strike is None:
            min_strike = spot * 0.8
        if max_strike is None:
            max_strike = spot * 1.2
        if expiration is None:
            expiration = date.today() + timedelta(days=30)

        options = []
        strikes = np.arange(min_strike, max_strike + 5, 5)

        for strike in strikes:
            for opt_type in ['call', 'put']:
                # Simple pricing approximation
                moneyness = (spot - strike) / spot
                if opt_type == 'put':
                    moneyness = -moneyness

                iv = 0.25 + abs(moneyness) * 0.3
                time_value = spot * 0.05 * np.sqrt(30 / 365)
                intrinsic = max(0, (spot - strike) if opt_type == 'call' else (strike - spot))
                price = intrinsic + time_value * (1 + abs(moneyness))

                options.append(OptionQuoteData(
                    symbol=f"{underlying}{expiration.strftime('%y%m%d')}{opt_type[0].upper()}{int(strike)}",
                    underlying=underlying,
                    option_type=opt_type,
                    strike=strike,
                    expiration=expiration,
                    bid=round(max(0.01, price - 0.05), 2),
                    ask=round(price + 0.05, 2),
                    last=round(price, 2),
                    volume=int(np.random.uniform(0, 1000)),
                    open_interest=int(np.random.uniform(100, 10000)),
                    implied_volatility=round(iv, 4),
                    delta=round(0.5 + moneyness * 0.5, 4) if opt_type == 'call' else round(-0.5 + moneyness * 0.5, 4),
                    gamma=round(0.02, 4),
                    theta=round(-0.05, 4),
                    vega=round(0.15, 4)
                ))

        return options

    def get_security_info(self, symbol: str) -> Optional[SecurityInfo]:
        """Get security information."""
        # Synthetic data
        sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial']
        import numpy as np

        return SecurityInfo(
            symbol=symbol,
            name=f"{symbol} Inc.",
            asset_class="equity",
            exchange="NYSE",
            currency="USD",
            sector=np.random.choice(sectors),
            market_cap=np.random.uniform(1e9, 1e12)
        )

    def get_health_metrics(self, symbol: str) -> Optional[HealthMetrics]:
        """Get data quality metrics for a symbol."""
        import numpy as np

        return HealthMetrics(
            symbol=symbol,
            health_score=round(np.random.uniform(0.7, 1.0), 3),
            is_stale=False,
            last_update=datetime.now() - timedelta(seconds=np.random.randint(1, 60)),
            update_count=np.random.randint(1000, 100000),
            error_count=np.random.randint(0, 10),
            gap_count=np.random.randint(0, 5)
        )


# ============================================================================
# FastAPI Application
# ============================================================================

def create_data_api(data_service: Optional[DataService] = None) -> FastAPI:
    """
    Create the Data API FastAPI application.

    Args:
        data_service: Data service instance (optional)

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Quantitative Trading Data API",
        description="REST API for market data, options, and reference data",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    # Initialize service
    service = data_service or DataService()

    # ========================================================================
    # Health Check Endpoints
    # ========================================================================

    @app.get("/health", tags=["System"])
    def health_check():
        """API health check."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/health/data/{symbol}", response_model=HealthMetrics, tags=["System"])
    def data_health(symbol: str = Path(..., description="Stock symbol")):
        """Get data quality health for a symbol."""
        metrics = service.get_health_metrics(symbol)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        return metrics

    # ========================================================================
    # Market Data Endpoints
    # ========================================================================

    @app.get("/quotes/{symbol}", response_model=QuoteData, tags=["Market Data"])
    def get_quote(symbol: str = Path(..., description="Stock symbol")):
        """Get real-time quote for a symbol."""
        quote = service.get_quote(symbol)
        if not quote:
            raise HTTPException(status_code=404, detail=f"Quote not found for {symbol}")
        return quote

    @app.get("/quotes", response_model=List[QuoteData], tags=["Market Data"])
    def get_quotes(symbols: str = Query(..., description="Comma-separated symbols")):
        """Get real-time quotes for multiple symbols."""
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        quotes = []
        for symbol in symbol_list:
            quote = service.get_quote(symbol)
            if quote:
                quotes.append(quote)
        return quotes

    @app.get("/bars/{symbol}", response_model=List[OHLCVBar], tags=["Market Data"])
    def get_bars(
        symbol: str = Path(..., description="Stock symbol"),
        start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
        end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
        range: Optional[TimeRange] = Query(None, description="Predefined time range"),
        frequency: DataFrequency = Query(DataFrequency.DAILY, description="Bar frequency")
    ):
        """Get historical OHLCV bars for a symbol."""
        # Resolve date range
        if range:
            end_date = date.today()
            range_days = {
                TimeRange.DAY_1: 1,
                TimeRange.WEEK_1: 7,
                TimeRange.MONTH_1: 30,
                TimeRange.MONTH_3: 90,
                TimeRange.MONTH_6: 180,
                TimeRange.YEAR_1: 365,
                TimeRange.YEAR_2: 730,
                TimeRange.YEAR_5: 1825,
                TimeRange.MAX: 3650
            }
            start_date = end_date - timedelta(days=range_days[range])

        if not start_date:
            start_date = date.today() - timedelta(days=30)
        if not end_date:
            end_date = date.today()

        bars = service.get_historical_bars(symbol.upper(), start_date, end_date, frequency)
        return bars

    @app.post("/bars/batch", response_model=Dict[str, List[OHLCVBar]], tags=["Market Data"])
    def get_bars_batch(request: HistoricalDataRequest):
        """Get historical bars for multiple symbols."""
        # Resolve date range
        if request.range:
            end_date = date.today()
            range_days = {
                TimeRange.DAY_1: 1,
                TimeRange.WEEK_1: 7,
                TimeRange.MONTH_1: 30,
                TimeRange.MONTH_3: 90,
                TimeRange.MONTH_6: 180,
                TimeRange.YEAR_1: 365,
                TimeRange.YEAR_2: 730,
                TimeRange.YEAR_5: 1825,
                TimeRange.MAX: 3650
            }
            start_date = end_date - timedelta(days=range_days[request.range])
        else:
            start_date = request.start_date or (date.today() - timedelta(days=30))
            end_date = request.end_date or date.today()

        result = {}
        for symbol in request.symbols:
            bars = service.get_historical_bars(
                symbol.upper(), start_date, end_date, request.frequency
            )
            result[symbol.upper()] = bars

        return result

    # ========================================================================
    # Options Data Endpoints
    # ========================================================================

    @app.get("/options/{underlying}", response_model=List[OptionQuoteData], tags=["Options"])
    def get_options_chain(
        underlying: str = Path(..., description="Underlying symbol"),
        expiration: Optional[date] = Query(None, description="Filter by expiration"),
        min_strike: Optional[float] = Query(None, description="Minimum strike price"),
        max_strike: Optional[float] = Query(None, description="Maximum strike price"),
        option_type: Optional[str] = Query(None, description="'call' or 'put'")
    ):
        """Get options chain for an underlying."""
        options = service.get_options_chain(
            underlying.upper(),
            expiration,
            min_strike,
            max_strike
        )

        if option_type:
            options = [o for o in options if o.option_type == option_type.lower()]

        return options

    @app.get("/options/{underlying}/expirations", response_model=List[date], tags=["Options"])
    def get_expirations(underlying: str = Path(..., description="Underlying symbol")):
        """Get available expiration dates for an underlying."""
        # Return synthetic expirations
        today = date.today()
        expirations = []

        # Weekly expirations for next 4 weeks
        for i in range(1, 5):
            exp = today + timedelta(days=(4 - today.weekday() + 7 * i) % 7 + 7 * (i - 1))
            expirations.append(exp)

        # Monthly expirations for next 6 months
        for i in range(1, 7):
            month = (today.month + i - 1) % 12 + 1
            year = today.year + (today.month + i - 1) // 12
            # Third Friday of month
            first_day = date(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            if third_friday not in expirations:
                expirations.append(third_friday)

        return sorted(expirations)

    # ========================================================================
    # Reference Data Endpoints
    # ========================================================================

    @app.get("/securities/{symbol}", response_model=SecurityInfo, tags=["Reference"])
    def get_security(symbol: str = Path(..., description="Stock symbol")):
        """Get security information."""
        info = service.get_security_info(symbol.upper())
        if not info:
            raise HTTPException(status_code=404, detail=f"Security {symbol} not found")
        return info

    @app.get("/securities/search", response_model=List[SecurityInfo], tags=["Reference"])
    def search_securities(
        query: str = Query(..., description="Search query"),
        asset_class: Optional[str] = Query(None, description="Filter by asset class"),
        limit: int = Query(50, ge=1, le=500, description="Maximum results")
    ):
        """Search for securities."""
        # Return synthetic results
        results = []
        for i in range(min(limit, 10)):
            results.append(service.get_security_info(f"{query.upper()}{i}"))
        return [r for r in results if r]

    @app.get("/calendar/trading-days", response_model=List[date], tags=["Reference"])
    def get_trading_days(
        start_date: date = Query(..., description="Start date"),
        end_date: date = Query(..., description="End date")
    ):
        """Get trading days in a date range."""
        from .reference import TradingCalendar
        calendar = TradingCalendar()
        return calendar.get_trading_days(start_date, end_date)

    @app.get("/calendar/is-trading-day/{check_date}", tags=["Reference"])
    def is_trading_day(check_date: date = Path(..., description="Date to check")):
        """Check if a date is a trading day."""
        from .reference import TradingCalendar
        calendar = TradingCalendar()
        return {"date": check_date, "is_trading_day": calendar.is_trading_day(check_date)}

    return app


# Create default app instance
app = create_data_api()


# ============================================================================
# CLI for running the API server
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn
    uvicorn.run(
        "quant_trading.data.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
