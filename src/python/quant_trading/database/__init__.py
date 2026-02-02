"""
Database module for the quantitative trading system.

Provides ORM models, database access layer, and utilities for
TimescaleDB time-series data storage.

Components:
- models: SQLAlchemy ORM models for all database tables
- db: Database access layer with connection pooling and helper methods
- config: Database configuration and connection management

Example:
    >>> from quant_trading.database import TimeSeriesDB, MarketPrice
    >>> db = TimeSeriesDB("postgresql://user:pass@localhost/quant_trading_db")
    >>> prices = db.get_market_prices("SPY", start_time)
"""

from .models import (
    Base,
    MarketPrice,
    OptionQuote,
    ModelParameter,
    Signal,
    Position,
    PositionUpdate,
)
from .db import TimeSeriesDB
from .config import (
    DatabaseConfig,
    get_database_config,
    get_database_url,
    get_config_for_environment,
)

__all__ = [
    # Base
    "Base",
    # Models
    "MarketPrice",
    "OptionQuote",
    "ModelParameter",
    "Signal",
    "Position",
    "PositionUpdate",
    # Database access layer
    "TimeSeriesDB",
    # Configuration
    "DatabaseConfig",
    "get_database_config",
    "get_database_url",
    "get_config_for_environment",
]
