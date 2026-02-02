"""
Reference Data Management Module.

This module provides reference data management:
- Symbol master data (ticker info, exchanges, asset classes)
- Corporate actions (splits, dividends, mergers)
- Trading calendars (market hours, holidays)
- Index compositions and weightings
- Security identifiers (CUSIP, ISIN, FIGI)

Reference data is essential for correct data processing and backtesting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from pandas.tseries.holiday import (
    USFederalHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday
)
from pandas.tseries.offsets import CustomBusinessDay
from dateutil.relativedelta import MO, TH

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"
    FIXED_INCOME = "fixed_income"
    INDEX = "index"


class Exchange(Enum):
    """Major exchanges."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    ARCA = "ARCA"
    BATS = "BATS"
    IEX = "IEX"
    CBOE = "CBOE"
    CME = "CME"


class CorporateActionType(Enum):
    """Types of corporate actions."""
    SPLIT = "split"
    REVERSE_SPLIT = "reverse_split"
    DIVIDEND = "dividend"
    SPECIAL_DIVIDEND = "special_dividend"
    MERGER = "merger"
    SPINOFF = "spinoff"
    NAME_CHANGE = "name_change"
    DELISTING = "delisting"
    IPO = "ipo"


@dataclass
class SecurityInfo:
    """Information about a security."""
    symbol: str
    name: str
    asset_class: AssetClass
    primary_exchange: Exchange
    currency: str = "USD"
    cusip: Optional[str] = None
    isin: Optional[str] = None
    figi: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    is_active: bool = True
    listing_date: Optional[date] = None
    delisting_date: Optional[date] = None
    country: str = "US"
    lot_size: int = 1
    tick_size: float = 0.01
    tags: List[str] = field(default_factory=list)


@dataclass
class CorporateAction:
    """Corporate action event."""
    symbol: str
    action_type: CorporateActionType
    ex_date: date
    record_date: Optional[date] = None
    payment_date: Optional[date] = None

    # For splits
    split_ratio_from: Optional[int] = None
    split_ratio_to: Optional[int] = None

    # For dividends
    dividend_amount: Optional[float] = None
    dividend_type: Optional[str] = None

    # For mergers/spinoffs
    target_symbol: Optional[str] = None
    acquirer_symbol: Optional[str] = None
    exchange_ratio: Optional[float] = None
    cash_amount: Optional[float] = None

    # For name changes
    old_symbol: Optional[str] = None
    new_symbol: Optional[str] = None

    notes: Optional[str] = None

    def get_adjustment_factor(self) -> float:
        """Calculate price adjustment factor for this action."""
        if self.action_type == CorporateActionType.SPLIT:
            if self.split_ratio_from and self.split_ratio_to:
                return self.split_ratio_from / self.split_ratio_to
        elif self.action_type == CorporateActionType.REVERSE_SPLIT:
            if self.split_ratio_from and self.split_ratio_to:
                return self.split_ratio_to / self.split_ratio_from
        return 1.0


@dataclass
class TradingSession:
    """Trading session times."""
    market_open: time
    market_close: time
    pre_market_open: Optional[time] = None
    pre_market_close: Optional[time] = None
    after_hours_open: Optional[time] = None
    after_hours_close: Optional[time] = None
    timezone: str = "America/New_York"


@dataclass
class MarketHoliday:
    """Market holiday."""
    date: date
    name: str
    early_close: Optional[time] = None
    exchange: Optional[Exchange] = None


class USEquityCalendar(USFederalHolidayCalendar):
    """
    US Equity market holiday calendar.

    Based on NYSE/NASDAQ holiday schedule.
    """
    rules = [
        Holiday("New Years Day", month=1, day=1, observance=nearest_workday),
        Holiday("MLK Day", month=1, day=1, offset=pd.DateOffset(weekday=MO(3))),
        Holiday("Presidents Day", month=2, day=1, offset=pd.DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday("Memorial Day", month=5, day=25, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Thanksgiving", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))),
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


class TradingCalendar:
    """
    Trading calendar for exchanges.

    Handles market hours, holidays, and special sessions.
    """

    # Standard US equity trading hours
    US_EQUITY_SESSION = TradingSession(
        market_open=time(9, 30),
        market_close=time(16, 0),
        pre_market_open=time(4, 0),
        pre_market_close=time(9, 30),
        after_hours_open=time(16, 0),
        after_hours_close=time(20, 0),
        timezone="America/New_York"
    )

    # Early close time (1 PM ET)
    EARLY_CLOSE_TIME = time(13, 0)

    # Days with early closes
    EARLY_CLOSE_DATES = {
        # Black Friday
        # Christmas Eve
        # Day before Independence Day
    }

    def __init__(self, exchange: Exchange = Exchange.NYSE):
        """
        Initialize trading calendar.

        Args:
            exchange: Exchange for calendar
        """
        self.exchange = exchange
        self._holiday_calendar = USEquityCalendar()
        self._holidays: Set[date] = set()
        self._early_closes: Dict[date, time] = {}
        self._load_holidays()

    def _load_holidays(self) -> None:
        """Load holidays for next 2 years."""
        start = date.today() - timedelta(days=365)
        end = date.today() + timedelta(days=730)

        holidays = self._holiday_calendar.holidays(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end)
        )

        self._holidays = {h.date() for h in holidays}

        # Add early close dates
        for year in range(start.year, end.year + 1):
            # Day after Thanksgiving
            thanksgiving = self._get_thanksgiving(year)
            if thanksgiving:
                black_friday = thanksgiving + timedelta(days=1)
                self._early_closes[black_friday] = self.EARLY_CLOSE_TIME

            # Christmas Eve (if weekday)
            christmas_eve = date(year, 12, 24)
            if christmas_eve.weekday() < 5:  # Monday-Friday
                self._early_closes[christmas_eve] = self.EARLY_CLOSE_TIME

            # July 3rd (if weekday and July 4th is not on Thursday)
            july_3 = date(year, 7, 3)
            july_4 = date(year, 7, 4)
            if july_3.weekday() < 5 and july_4.weekday() != 3:
                self._early_closes[july_3] = self.EARLY_CLOSE_TIME

    def _get_thanksgiving(self, year: int) -> Optional[date]:
        """Get Thanksgiving date for a year (4th Thursday of November)."""
        nov_1 = date(year, 11, 1)
        # Find first Thursday
        first_thursday = nov_1 + timedelta(days=(3 - nov_1.weekday()) % 7)
        # Fourth Thursday
        thanksgiving = first_thursday + timedelta(weeks=3)
        return thanksgiving

    def is_trading_day(self, d: date) -> bool:
        """Check if a date is a trading day."""
        # Weekend check
        if d.weekday() >= 5:
            return False

        # Holiday check
        if d in self._holidays:
            return False

        return True

    def is_market_open(self, dt: datetime) -> bool:
        """Check if market is open at a specific datetime."""
        if not self.is_trading_day(dt.date()):
            return False

        t = dt.time()
        session = self.get_session(dt.date())

        return session.market_open <= t < session.market_close

    def get_session(self, d: date) -> TradingSession:
        """Get trading session for a date."""
        session = TradingSession(
            market_open=self.US_EQUITY_SESSION.market_open,
            market_close=self.US_EQUITY_SESSION.market_close,
            pre_market_open=self.US_EQUITY_SESSION.pre_market_open,
            pre_market_close=self.US_EQUITY_SESSION.pre_market_close,
            after_hours_open=self.US_EQUITY_SESSION.after_hours_open,
            after_hours_close=self.US_EQUITY_SESSION.after_hours_close,
            timezone=self.US_EQUITY_SESSION.timezone
        )

        # Check for early close
        if d in self._early_closes:
            session.market_close = self._early_closes[d]
            session.after_hours_open = self._early_closes[d]

        return session

    def get_trading_days(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """Get list of trading days in range."""
        days = []
        current = start_date

        while current <= end_date:
            if self.is_trading_day(current):
                days.append(current)
            current += timedelta(days=1)

        return days

    def get_next_trading_day(self, d: date) -> date:
        """Get next trading day after a date."""
        next_day = d + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day

    def get_previous_trading_day(self, d: date) -> date:
        """Get previous trading day before a date."""
        prev_day = d - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day

    def get_holidays(
        self,
        start_date: date,
        end_date: date
    ) -> List[MarketHoliday]:
        """Get holidays in date range."""
        holidays = []
        for d in self._holidays:
            if start_date <= d <= end_date:
                holidays.append(MarketHoliday(
                    date=d,
                    name=self._get_holiday_name(d),
                    exchange=self.exchange
                ))
        return sorted(holidays, key=lambda h: h.date)

    def _get_holiday_name(self, d: date) -> str:
        """Get name of holiday on a date."""
        # Simple mapping based on month/day patterns
        if d.month == 1 and d.day <= 3:
            return "New Year's Day"
        if d.month == 1 and 15 <= d.day <= 21:
            return "MLK Day"
        if d.month == 2 and 15 <= d.day <= 21:
            return "Presidents Day"
        if d.month == 5 and d.day >= 25:
            return "Memorial Day"
        if d.month == 6 and 18 <= d.day <= 20:
            return "Juneteenth"
        if d.month == 7 and d.day <= 5:
            return "Independence Day"
        if d.month == 9 and d.day <= 7:
            return "Labor Day"
        if d.month == 11 and 22 <= d.day <= 28:
            return "Thanksgiving"
        if d.month == 12 and 24 <= d.day <= 26:
            return "Christmas"
        return "Holiday"

    def trading_days_between(
        self,
        start_date: date,
        end_date: date
    ) -> int:
        """Count trading days between two dates."""
        return len(self.get_trading_days(start_date, end_date))


@dataclass
class IndexComposition:
    """Index composition and weightings."""
    index_symbol: str
    as_of_date: date
    components: Dict[str, float]  # symbol -> weight
    total_market_cap: Optional[float] = None
    divisor: Optional[float] = None


class SymbolMaster:
    """
    Master database of security information.

    Provides symbol lookup, mapping, and metadata.
    """

    def __init__(self):
        """Initialize symbol master."""
        self._securities: Dict[str, SecurityInfo] = {}
        self._cusip_map: Dict[str, str] = {}  # CUSIP -> symbol
        self._isin_map: Dict[str, str] = {}  # ISIN -> symbol
        self._figi_map: Dict[str, str] = {}  # FIGI -> symbol
        self._name_map: Dict[str, str] = {}  # lowercase name -> symbol

    def add_security(self, security: SecurityInfo) -> None:
        """Add a security to the master."""
        self._securities[security.symbol] = security

        if security.cusip:
            self._cusip_map[security.cusip] = security.symbol
        if security.isin:
            self._isin_map[security.isin] = security.symbol
        if security.figi:
            self._figi_map[security.figi] = security.symbol

        self._name_map[security.name.lower()] = security.symbol

        logger.debug(f"Added security: {security.symbol}")

    def get_security(self, symbol: str) -> Optional[SecurityInfo]:
        """Get security by symbol."""
        return self._securities.get(symbol)

    def lookup_by_cusip(self, cusip: str) -> Optional[SecurityInfo]:
        """Look up security by CUSIP."""
        symbol = self._cusip_map.get(cusip)
        return self._securities.get(symbol) if symbol else None

    def lookup_by_isin(self, isin: str) -> Optional[SecurityInfo]:
        """Look up security by ISIN."""
        symbol = self._isin_map.get(isin)
        return self._securities.get(symbol) if symbol else None

    def lookup_by_figi(self, figi: str) -> Optional[SecurityInfo]:
        """Look up security by FIGI."""
        symbol = self._figi_map.get(figi)
        return self._securities.get(symbol) if symbol else None

    def search(
        self,
        query: str,
        asset_class: Optional[AssetClass] = None,
        exchange: Optional[Exchange] = None,
        limit: int = 100
    ) -> List[SecurityInfo]:
        """
        Search for securities.

        Args:
            query: Search query (matches symbol or name)
            asset_class: Filter by asset class
            exchange: Filter by exchange
            limit: Maximum results

        Returns:
            List of matching securities
        """
        results = []
        query_lower = query.lower()

        for symbol, security in self._securities.items():
            if not security.is_active:
                continue

            if asset_class and security.asset_class != asset_class:
                continue

            if exchange and security.primary_exchange != exchange:
                continue

            # Match on symbol or name
            if (query_lower in symbol.lower() or
                query_lower in security.name.lower()):
                results.append(security)

                if len(results) >= limit:
                    break

        return results

    def get_by_sector(self, sector: str) -> List[SecurityInfo]:
        """Get all securities in a sector."""
        return [
            s for s in self._securities.values()
            if s.is_active and s.sector and s.sector.lower() == sector.lower()
        ]

    def get_by_asset_class(self, asset_class: AssetClass) -> List[SecurityInfo]:
        """Get all securities of an asset class."""
        return [
            s for s in self._securities.values()
            if s.is_active and s.asset_class == asset_class
        ]

    def get_active_symbols(self) -> List[str]:
        """Get all active symbols."""
        return [s.symbol for s in self._securities.values() if s.is_active]

    def load_from_dataframe(self, df: pd.DataFrame) -> int:
        """
        Load securities from a DataFrame.

        Expected columns: symbol, name, asset_class, exchange, currency, etc.

        Args:
            df: DataFrame with security data

        Returns:
            Number of securities loaded
        """
        count = 0
        for _, row in df.iterrows():
            try:
                security = SecurityInfo(
                    symbol=row['symbol'],
                    name=row['name'],
                    asset_class=AssetClass(row.get('asset_class', 'equity')),
                    primary_exchange=Exchange(row.get('exchange', 'NYSE')),
                    currency=row.get('currency', 'USD'),
                    cusip=row.get('cusip'),
                    isin=row.get('isin'),
                    figi=row.get('figi'),
                    sector=row.get('sector'),
                    industry=row.get('industry'),
                    market_cap=row.get('market_cap'),
                    is_active=row.get('is_active', True)
                )
                self.add_security(security)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to load security: {e}")

        logger.info(f"Loaded {count} securities")
        return count

    def to_dataframe(self) -> pd.DataFrame:
        """Convert symbol master to DataFrame."""
        records = []
        for security in self._securities.values():
            records.append({
                'symbol': security.symbol,
                'name': security.name,
                'asset_class': security.asset_class.value,
                'exchange': security.primary_exchange.value,
                'currency': security.currency,
                'cusip': security.cusip,
                'isin': security.isin,
                'figi': security.figi,
                'sector': security.sector,
                'industry': security.industry,
                'market_cap': security.market_cap,
                'is_active': security.is_active
            })
        return pd.DataFrame(records)


class CorporateActionsManager:
    """
    Manager for corporate actions.

    Handles splits, dividends, and other corporate events.
    """

    def __init__(self):
        """Initialize corporate actions manager."""
        self._actions: List[CorporateAction] = []
        self._by_symbol: Dict[str, List[CorporateAction]] = {}

    def add_action(self, action: CorporateAction) -> None:
        """Add a corporate action."""
        self._actions.append(action)

        if action.symbol not in self._by_symbol:
            self._by_symbol[action.symbol] = []
        self._by_symbol[action.symbol].append(action)

        # Sort by date
        self._by_symbol[action.symbol].sort(key=lambda a: a.ex_date)

        logger.debug(f"Added corporate action: {action.symbol} {action.action_type.value}")

    def get_actions_for_symbol(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        action_type: Optional[CorporateActionType] = None
    ) -> List[CorporateAction]:
        """
        Get corporate actions for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Filter by start date
            end_date: Filter by end date
            action_type: Filter by action type

        Returns:
            List of corporate actions
        """
        actions = self._by_symbol.get(symbol, [])

        if start_date:
            actions = [a for a in actions if a.ex_date >= start_date]
        if end_date:
            actions = [a for a in actions if a.ex_date <= end_date]
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]

        return actions

    def get_splits(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """Get splits for a symbol."""
        return self.get_actions_for_symbol(
            symbol, start_date, end_date, CorporateActionType.SPLIT
        ) + self.get_actions_for_symbol(
            symbol, start_date, end_date, CorporateActionType.REVERSE_SPLIT
        )

    def get_dividends(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[CorporateAction]:
        """Get dividends for a symbol."""
        return self.get_actions_for_symbol(
            symbol, start_date, end_date, CorporateActionType.DIVIDEND
        ) + self.get_actions_for_symbol(
            symbol, start_date, end_date, CorporateActionType.SPECIAL_DIVIDEND
        )

    def calculate_adjustment_factor(
        self,
        symbol: str,
        as_of_date: date
    ) -> float:
        """
        Calculate cumulative adjustment factor for historical prices.

        This factor accounts for all splits since as_of_date.

        Args:
            symbol: Stock symbol
            as_of_date: Calculate adjustment as of this date

        Returns:
            Cumulative adjustment factor
        """
        splits = self.get_splits(symbol, start_date=as_of_date)

        factor = 1.0
        for split in splits:
            factor *= split.get_adjustment_factor()

        return factor

    def adjust_prices(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Adjust historical prices for corporate actions.

        Args:
            df: DataFrame with price data (must have date index)
            symbol: Stock symbol
            price_columns: Columns to adjust (default: open, high, low, close)

        Returns:
            Adjusted DataFrame
        """
        if price_columns is None:
            price_columns = ['open', 'high', 'low', 'close']

        df = df.copy()
        splits = self.get_splits(symbol)

        for split in splits:
            factor = split.get_adjustment_factor()
            # Adjust prices before ex-date
            mask = df.index.date < split.ex_date

            for col in price_columns:
                if col in df.columns:
                    df.loc[mask, col] = df.loc[mask, col] * factor

            # Adjust volume inversely
            if 'volume' in df.columns:
                df.loc[mask, 'volume'] = df.loc[mask, 'volume'] / factor

        return df


class ReferenceDataManager:
    """
    Central manager for all reference data.

    Coordinates symbol master, calendars, and corporate actions.
    """

    def __init__(self):
        """Initialize reference data manager."""
        self.symbol_master = SymbolMaster()
        self.corporate_actions = CorporateActionsManager()
        self._calendars: Dict[Exchange, TradingCalendar] = {}
        self._index_compositions: Dict[str, IndexComposition] = {}

    def get_calendar(self, exchange: Exchange = Exchange.NYSE) -> TradingCalendar:
        """Get trading calendar for an exchange."""
        if exchange not in self._calendars:
            self._calendars[exchange] = TradingCalendar(exchange)
        return self._calendars[exchange]

    def set_index_composition(self, composition: IndexComposition) -> None:
        """Set index composition."""
        self._index_compositions[composition.index_symbol] = composition
        logger.info(f"Set index composition: {composition.index_symbol}")

    def get_index_composition(self, index_symbol: str) -> Optional[IndexComposition]:
        """Get index composition."""
        return self._index_compositions.get(index_symbol)

    def is_trading_day(self, d: date, exchange: Exchange = Exchange.NYSE) -> bool:
        """Check if date is a trading day."""
        return self.get_calendar(exchange).is_trading_day(d)

    def get_next_trading_day(
        self,
        d: date,
        exchange: Exchange = Exchange.NYSE
    ) -> date:
        """Get next trading day."""
        return self.get_calendar(exchange).get_next_trading_day(d)

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol exists in master."""
        return self.symbol_master.get_security(symbol) is not None
