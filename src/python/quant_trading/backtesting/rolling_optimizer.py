"""
Rolling Optimization Backtester.

Implements a rolling window optimization strategy where:
1. For each rebalance period, optimize on the prior N months of data
2. Use optimized strategies for the next period
3. Aggregate results across all periods

This adapts to changing market regimes by continuously re-optimizing.

Usage:
    >>> backtester = RollingOptimizationBacktester(
    ...     sectors=[Sector.TECHNOLOGY, Sector.FINANCIALS, Sector.HEALTHCARE],
    ...     lookback_months=12,
    ...     rebalance_months=3,
    ...     stocks_per_sector=6,
    ... )
    >>> results = backtester.run(
    ...     start_date="2015-01-01",
    ...     end_date="2025-01-01",
    ... )
    >>> print(results.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sector_portfolio import (
    Sector,
    SECTOR_STOCKS,
    get_sector,
    get_sector_strategy,
    ConfidenceCalculator,
    calculate_position_size,
)
from .sector_optimizer import (
    SectorAlgorithmOptimizer,
    SectorOptimizationResults,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodResult:
    """Results for a single rebalance period."""

    period_start: datetime
    period_end: datetime
    optimization_start: datetime
    optimization_end: datetime

    # Strategy assignments for this period
    strategies: Dict[str, Tuple[str, Dict[str, Any]]]  # sector -> (algorithm, params)

    # Performance metrics
    initial_equity: float
    final_equity: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    n_trades: int

    # Per-sector P&L
    sector_pnl: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "optimization_start": self.optimization_start.isoformat(),
            "optimization_end": self.optimization_end.isoformat(),
            "strategies": {k: list(v) for k, v in self.strategies.items()},
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
            "sector_pnl": self.sector_pnl,
        }


@dataclass
class RollingBacktestResults:
    """Aggregated results from rolling optimization backtest."""

    # Configuration
    start_date: datetime
    end_date: datetime
    lookback_months: int
    rebalance_months: int
    sectors: List[str]
    stocks_per_sector: int

    # Per-period results
    period_results: List[PeriodResult] = field(default_factory=list)

    # Aggregated metrics
    initial_capital: float = 100000.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_rate: float = 0.0
    total_trades: int = 0

    # Equity curve
    equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # Strategy usage stats
    strategy_counts: Dict[str, int] = field(default_factory=dict)
    sector_total_pnl: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string."""
        years = (self.end_date - self.start_date).days / 365.25

        strategy_summary = "\n".join(
            f"    {algo}: {count} periods"
            for algo, count in sorted(self.strategy_counts.items(), key=lambda x: -x[1])
        )

        sector_summary = "\n".join(
            f"    {sector}: ${pnl:>12,.2f}"
            for sector, pnl in sorted(self.sector_total_pnl.items(), key=lambda x: -x[1])
        )

        return f"""
================================================================================
              ROLLING OPTIMIZATION BACKTEST RESULTS
================================================================================
Period: {self.start_date.date()} to {self.end_date.date()} ({years:.1f} years)
Lookback: {self.lookback_months} months | Rebalance: every {self.rebalance_months} months
Sectors: {', '.join(self.sectors)}
Stocks per sector: {self.stocks_per_sector}

RETURNS
-------
Initial Capital:      ${self.initial_capital:>12,.0f}
Final Equity:         ${self.final_equity:>12,.0f}
Total Return:         {self.total_return_pct:>12.2f}%
Annualized Return:    {self.annualized_return_pct:>12.2f}%

RISK METRICS
------------
Sharpe Ratio:         {self.sharpe_ratio:>12.2f}
Max Drawdown:         {self.max_drawdown_pct:>12.2f}%

TRADE STATISTICS
----------------
Total Trades:         {self.total_trades:>12}
Avg Win Rate:         {self.avg_win_rate:>12.1f}%
Rebalance Periods:    {len(self.period_results):>12}

STRATEGY USAGE (by period count)
--------------------------------
{strategy_summary}

SECTOR P&L (cumulative)
-----------------------
{sector_summary}
================================================================================
"""

    def calculate_aggregates(self) -> None:
        """Calculate aggregate metrics from period results."""
        if not self.period_results:
            return

        # Final equity from last period
        self.final_equity = self.period_results[-1].final_equity

        # Total return
        self.total_return_pct = ((self.final_equity / self.initial_capital) - 1) * 100

        # Annualized return
        years = (self.end_date - self.start_date).days / 365.25
        if years > 0:
            self.annualized_return_pct = (
                (self.final_equity / self.initial_capital) ** (1 / years) - 1
            ) * 100

        # Average win rate
        total_trades = sum(p.n_trades for p in self.period_results)
        self.total_trades = total_trades
        if total_trades > 0:
            weighted_win_rate = sum(
                p.win_rate * p.n_trades for p in self.period_results
            )
            self.avg_win_rate = weighted_win_rate / total_trades

        # Max drawdown (from equity curve)
        if self.equity_curve:
            equities = [e[1] for e in self.equity_curve]
            peak = equities[0]
            max_dd = 0
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown_pct = max_dd

        # Sharpe ratio (simplified from period returns)
        if len(self.period_results) > 1:
            period_returns = [p.total_return_pct for p in self.period_results]
            avg_return = np.mean(period_returns)
            std_return = np.std(period_returns)
            if std_return > 0:
                # Annualize based on rebalance frequency
                periods_per_year = 12 / self.rebalance_months
                self.sharpe_ratio = (avg_return / std_return) * np.sqrt(periods_per_year)

        # Strategy usage counts
        self.strategy_counts = {}
        for period in self.period_results:
            for sector, (algo, params) in period.strategies.items():
                self.strategy_counts[algo] = self.strategy_counts.get(algo, 0) + 1

        # Sector total P&L
        self.sector_total_pnl = {}
        for period in self.period_results:
            for sector, pnl in period.sector_pnl.items():
                self.sector_total_pnl[sector] = self.sector_total_pnl.get(sector, 0) + pnl


class RollingOptimizationBacktester:
    """
    Backtester that uses rolling window optimization.

    For each rebalance period:
    1. Optimize on prior lookback_months of data
    2. Select best algorithm per sector
    3. Run backtest for rebalance_months
    4. Roll forward and repeat
    """

    def __init__(
        self,
        sectors: List[Sector],
        lookback_months: int = 12,
        rebalance_months: int = 3,
        stocks_per_sector: int = 6,
        initial_capital: float = 100000.0,
        optimization_stocks: int = 5,
        optimize_params: bool = False,
    ):
        """
        Initialize rolling optimization backtester.

        Args:
            sectors: List of sectors to include
            lookback_months: Months of data for optimization
            rebalance_months: Months between rebalancing
            stocks_per_sector: Number of stocks per sector in portfolio
            initial_capital: Starting capital
            optimization_stocks: Stocks per sector for optimization runs
            optimize_params: Whether to search for optimal params (slower)
        """
        self.sectors = sectors
        self.lookback_months = lookback_months
        self.rebalance_months = rebalance_months
        self.stocks_per_sector = stocks_per_sector
        self.initial_capital = initial_capital
        self.optimization_stocks = optimization_stocks
        self.optimize_params = optimize_params

        # Data cache to avoid re-fetching
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def run(
        self,
        start_date: str,
        end_date: str,
        verbose: bool = True,
    ) -> RollingBacktestResults:
        """
        Run the rolling optimization backtest.

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            verbose: Print progress

        Returns:
            RollingBacktestResults with all period results and aggregates
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        results = RollingBacktestResults(
            start_date=start_dt,
            end_date=end_dt,
            lookback_months=self.lookback_months,
            rebalance_months=self.rebalance_months,
            sectors=[s.value for s in self.sectors],
            stocks_per_sector=self.stocks_per_sector,
            initial_capital=self.initial_capital,
        )

        # Generate rebalance periods
        periods = self._generate_periods(start_dt, end_dt)

        if verbose:
            print(f"\nRunning rolling optimization backtest:")
            print(f"  Period: {start_date} to {end_date}")
            print(f"  Lookback: {self.lookback_months} months")
            print(f"  Rebalance: every {self.rebalance_months} months")
            print(f"  Periods: {len(periods)}")
            print()

        current_equity = self.initial_capital

        for i, (period_start, period_end) in enumerate(periods):
            # Calculate optimization window (prior lookback_months)
            opt_end = period_start - timedelta(days=1)
            opt_start = opt_end - relativedelta(months=self.lookback_months)

            if verbose:
                print(f"Period {i+1}/{len(periods)}: {period_start.date()} to {period_end.date()}")
                print(f"  Optimizing on: {opt_start.date()} to {opt_end.date()}")

            # Run optimization for this period
            optimization_results = self._run_optimization(
                opt_start.strftime("%Y-%m-%d"),
                opt_end.strftime("%Y-%m-%d"),
            )

            # Extract best strategies
            strategies = {}
            for sector in self.sectors:
                algo, params = optimization_results.get_best_algorithm(sector)
                strategies[sector.value] = (algo, params)
                if verbose:
                    print(f"    {sector.value}: {algo}")

            # Run backtest for this period
            period_result = self._run_period_backtest(
                period_start,
                period_end,
                strategies,
                current_equity,
                optimization_results,
            )

            if period_result:
                results.period_results.append(period_result)
                current_equity = period_result.final_equity

                # Add to equity curve
                results.equity_curve.append((period_end, current_equity))

                if verbose:
                    print(f"  Result: ${period_result.initial_equity:,.0f} -> ${period_result.final_equity:,.0f} ({period_result.total_return_pct:+.2f}%)")

            if verbose:
                print()

        # Calculate aggregates
        results.calculate_aggregates()

        return results

    def _generate_periods(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[Tuple[datetime, datetime]]:
        """Generate list of (period_start, period_end) tuples."""
        periods = []

        # First period starts after lookback window
        current_start = start_dt + relativedelta(months=self.lookback_months)

        while current_start < end_dt:
            current_end = min(
                current_start + relativedelta(months=self.rebalance_months),
                end_dt
            )
            periods.append((current_start, current_end))
            current_start = current_end

        return periods

    def _run_optimization(
        self,
        start_date: str,
        end_date: str,
    ) -> SectorOptimizationResults:
        """Run optimization for a specific period."""
        optimizer = SectorAlgorithmOptimizer(
            n_stocks_per_sector=self.optimization_stocks,
            backtest_days=180,  # ~6 months for optimization backtests
            optimize_params=self.optimize_params,
            cache_dir=None,  # Don't cache rolling optimizations
        )

        return optimizer.run_optimization(
            sectors=self.sectors,
            start_date=start_date,
            end_date=end_date,
        )

    def _run_period_backtest(
        self,
        period_start: datetime,
        period_end: datetime,
        strategies: Dict[str, Tuple[str, Dict[str, Any]]],
        initial_equity: float,
        optimization_results: SectorOptimizationResults,
    ) -> Optional[PeriodResult]:
        """Run backtest for a single period with given strategies."""
        from .engine import BacktestEngine
        from .data_handler import HistoricDataFrameHandler
        from .portfolio import Portfolio
        from .execution import InstantExecutionHandler
        from .multi_strategy import MultiStrategyManager

        start_str = period_start.strftime("%Y-%m-%d")
        end_str = period_end.strftime("%Y-%m-%d")

        # Select stocks for each sector
        selected_stocks = []
        calc = ConfidenceCalculator(lookback_days=60, optimization_results=optimization_results)
        scan_start = (period_start - timedelta(days=60)).strftime("%Y-%m-%d")

        for sector in self.sectors:
            stocks = SECTOR_STOCKS.get(sector, [])[:30]  # Scan up to 30
            sector_candidates = []

            algo, params = strategies[sector.value]

            for symbol in stocks:
                try:
                    data = self._fetch_data(symbol, scan_start, end_str)
                    if data is None or len(data) < 30:
                        continue

                    prices = data['Close'].values
                    metrics = calc.calculate(symbol, prices, signal_strength=0.5, algorithm=algo)

                    sector_candidates.append({
                        'symbol': symbol,
                        'sector': sector,
                        'algorithm': algo,
                        'params': params,
                        'confidence': metrics.confidence,
                        'data': data,
                    })
                except Exception:
                    continue

            # Select top N by confidence
            sector_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            selected_stocks.extend(sector_candidates[:self.stocks_per_sector])

        if len(selected_stocks) < 3:
            logger.warning(f"Insufficient stocks for period {start_str}")
            return None

        # Prepare data for backtest
        all_data = {}
        min_date = None
        max_date = None

        for stock_info in selected_stocks:
            symbol = stock_info['symbol']
            data = stock_info['data']
            data = data[(data.index >= start_str) & (data.index <= end_str)]

            if len(data) < 10:
                continue

            all_data[symbol] = data

            if min_date is None or data.index[0] > min_date:
                min_date = data.index[0]
            if max_date is None or data.index[-1] < max_date:
                max_date = data.index[-1]

        if not all_data or min_date is None:
            return None

        # Combine data
        combined_data = pd.DataFrame(index=pd.date_range(min_date, max_date, freq='B'))

        for symbol, data in all_data.items():
            data = data[(data.index >= min_date) & (data.index <= max_date)]
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    combined_data[f"{symbol}_{col}"] = data[col]

        combined_data = combined_data.ffill().dropna()

        if len(combined_data) < 10:
            return None

        events = Queue()

        # Create components
        data_handler = HistoricDataFrameHandler(
            events_queue=events,
            symbol_list=list(all_data.keys()),
            data=combined_data,
        )

        max_position_pct = min(0.15, 1.5 / len(all_data))
        portfolio = Portfolio(
            initial_capital=initial_equity,
            max_position_pct=max_position_pct,
        )

        executor = InstantExecutionHandler(events_queue=events)

        strategy = MultiStrategyManager(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        # Assign strategies
        for stock_info in selected_stocks:
            symbol = stock_info['symbol']
            if symbol not in all_data:
                continue
            algo = stock_info['algorithm']
            params = stock_info['params']
            strategy.add_strategy(symbol, algo, **params)

        # Run backtest
        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        bt_results = engine.run()

        # Calculate sector P&L
        sector_pnl = {}
        if bt_results.trade_history:
            for trade in bt_results.trade_history:
                sym = trade.get('symbol', 'Unknown')
                sector = get_sector(sym)
                pnl = trade.get('pnl', 0)
                sector_pnl[sector.value] = sector_pnl.get(sector.value, 0) + pnl

        return PeriodResult(
            period_start=period_start,
            period_end=period_end,
            optimization_start=period_start - relativedelta(months=self.lookback_months),
            optimization_end=period_start - timedelta(days=1),
            strategies=strategies,
            initial_equity=initial_equity,
            final_equity=bt_results.final_equity,
            total_return_pct=bt_results.total_return_pct,
            sharpe_ratio=bt_results.sharpe_ratio,
            max_drawdown_pct=bt_results.max_drawdown_pct,
            win_rate=bt_results.win_rate,
            n_trades=bt_results.n_trades,
            sector_pnl=sector_pnl,
        )

    def _fetch_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data with caching."""
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if data.empty:
                return None

            data.columns = [c.capitalize() for c in data.columns]

            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            data['Symbol'] = symbol

            self._data_cache[cache_key] = data
            return data

        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
