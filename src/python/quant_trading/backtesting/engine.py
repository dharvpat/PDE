"""
Backtesting engine and results.

Provides the main BacktestEngine that coordinates:
    - Data handler (market data)
    - Strategy (signal generation)
    - Portfolio (position tracking)
    - Execution handler (order fills)

Also provides BacktestResults for performance analysis.

Reference:
    Event-driven backtesting architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    PANDAS_AVAILABLE = False

from .events import EventType, MarketEvent, SignalEvent
from .portfolio import Portfolio

if TYPE_CHECKING:
    from .data_handler import DataHandler
    from .execution import ExecutionHandler
    from .strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Backtesting results and performance metrics.

    Contains comprehensive performance analysis including:
        - Return metrics (total, annualized, risk-adjusted)
        - Risk metrics (volatility, drawdown, VaR)
        - Trade statistics (win rate, profit factor)
        - Cost analysis (commission, slippage)

    Example:
        >>> results = engine.run()
        >>> print(results.summary())
        >>> results.plot_equity_curve()
    """

    # Core data
    equity_curve: List[Tuple[datetime, float]]
    returns: List[float]
    trade_history: List[Dict[str, Any]]

    # Return metrics
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    drawdown_duration_days: int = 0
    var_95_pct: float = 0.0
    cvar_95_pct: float = 0.0

    # Trade statistics
    n_trades: int = 0
    n_winning_trades: int = 0
    n_losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_holding_period_days: float = 0.0

    # Cost analysis
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    costs_pct_of_pnl: float = 0.0

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    n_bars: int = 0
    initial_capital: float = 0.0
    final_equity: float = 0.0

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
================================================================================
                           BACKTEST RESULTS
================================================================================
Period: {self.start_date} to {self.end_date} ({self.n_bars} bars)
Initial Capital: ${self.initial_capital:,.0f}
Final Equity:    ${self.final_equity:,.0f}

RETURNS
-------
Total Return:        {self.total_return_pct:>8.2f}%
Annualized Return:   {self.annualized_return_pct:>8.2f}%
Volatility (Ann.):   {self.volatility_pct:>8.2f}%

RISK-ADJUSTED METRICS
---------------------
Sharpe Ratio:        {self.sharpe_ratio:>8.2f}
Sortino Ratio:       {self.sortino_ratio:>8.2f}
Calmar Ratio:        {self.calmar_ratio:>8.2f}

RISK METRICS
------------
Max Drawdown:        {self.max_drawdown_pct:>8.2f}%
Avg Drawdown:        {self.avg_drawdown_pct:>8.2f}%
95% VaR (Daily):     {self.var_95_pct:>8.2f}%
95% CVaR (Daily):    {self.cvar_95_pct:>8.2f}%

TRADE STATISTICS
----------------
Total Trades:        {self.n_trades:>8}
Win Rate:            {self.win_rate:>8.1f}%
Profit Factor:       {self.profit_factor:>8.2f}
Avg Trade Return:    {self.avg_trade_return_pct:>8.2f}%
Avg Win:             {self.avg_win_pct:>8.2f}%
Avg Loss:            {self.avg_loss_pct:>8.2f}%
Avg Holding Period:  {self.avg_holding_period_days:>8.1f} days

COSTS
-----
Total Commission:    ${self.total_commission:>10,.0f}
Total Slippage:      ${self.total_slippage:>10,.0f}
Total Costs:         ${self.total_costs:>10,.0f}
Costs % of P&L:      {self.costs_pct_of_pnl:>8.1f}%
================================================================================
        """

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "volatility_pct": self.volatility_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_costs": self.total_costs,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
        }

    def get_equity_series(self) -> "pd.Series":
        """Get equity curve as pandas Series."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for get_equity_series")

        dates = [dt for dt, _ in self.equity_curve]
        values = [val for _, val in self.equity_curve]
        return pd.Series(values, index=dates)

    def get_returns_series(self) -> "pd.Series":
        """Get returns as pandas Series."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for get_returns_series")

        dates = [dt for dt, _ in self.equity_curve[1:]]
        return pd.Series(self.returns, index=dates)


class BacktestEngine:
    """
    Main backtesting engine.

    Coordinates data handler, strategy, portfolio, and execution
    in an event-driven architecture.

    Example:
        >>> from quant_trading.backtesting import (
        ...     BacktestEngine, Portfolio, SyntheticDataHandler,
        ...     SimulatedExecutionHandler, BuyAndHoldStrategy
        ... )
        >>>
        >>> # Create components
        >>> events = Queue()
        >>> data_handler = SyntheticDataHandler(events, ['SPY'], n_bars=252)
        >>> portfolio = Portfolio(initial_capital=100000)
        >>> executor = SimulatedExecutionHandler(events)
        >>> strategy = BuyAndHoldStrategy(events, data_handler, portfolio)
        >>>
        >>> # Run backtest
        >>> engine = BacktestEngine(data_handler, strategy, portfolio, executor)
        >>> results = engine.run()
        >>> print(results.summary())
    """

    def __init__(
        self,
        data_handler: "DataHandler",
        strategy: "Strategy",
        portfolio: Portfolio,
        execution_handler: "ExecutionHandler",
        risk_free_rate: float = 0.05,
    ):
        """
        Initialize backtest engine.

        Args:
            data_handler: Provides market data
            strategy: Trading strategy
            portfolio: Portfolio tracker
            execution_handler: Handles order execution
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.risk_free_rate = risk_free_rate

        # Event queue
        self.events = data_handler.events

        # State
        self.bar_count = 0

        logger.info("BacktestEngine initialized")

    def run(self) -> BacktestResults:
        """
        Run backtest simulation.

        Returns:
            BacktestResults with performance metrics
        """
        logger.info("Starting backtest...")

        # Main event loop
        while self.data_handler.continue_backtest:
            # Get next bars
            self.data_handler.update_bars()

            # Process events
            while not self.events.empty():
                event = self.events.get()

                if event.event_type == EventType.MARKET:
                    # Update portfolio with new prices
                    self.portfolio.update_market_data(event)

                    # Update execution handler with new prices
                    self.execution_handler.update_market_data(event)

                    # Generate signals
                    self.strategy.calculate_signals(event)

                elif event.event_type == EventType.SIGNAL:
                    # Convert signal to order
                    self.portfolio.generate_order(event, self.events)

                elif event.event_type == EventType.ORDER:
                    # Execute order
                    self.execution_handler.execute_order(event)

                elif event.event_type == EventType.FILL:
                    # Update portfolio
                    self.portfolio.update_fill(event)

            self.bar_count += 1

        logger.info(f"Backtest complete. Processed {self.bar_count} bars.")

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> BacktestResults:
        """Calculate backtest performance metrics."""
        # Get equity curve
        equity_curve = self.portfolio.equity_curve

        if len(equity_curve) < 2:
            return BacktestResults(
                equity_curve=equity_curve,
                returns=[],
                trade_history=[],
            )

        # Calculate returns
        equities = [eq for _, eq in equity_curve]
        returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                ret = (equities[i] - equities[i - 1]) / equities[i - 1]
                returns.append(ret)

        returns_array = np.array(returns)

        # Calculate metrics
        total_return_pct = ((equities[-1] / equities[0]) - 1) * 100

        n_years = len(returns) / 252
        if n_years > 0:
            annualized_return_pct = (
                ((1 + total_return_pct / 100) ** (1 / n_years)) - 1
            ) * 100
        else:
            annualized_return_pct = total_return_pct

        volatility_pct = float(np.std(returns_array) * np.sqrt(252) * 100)

        # Sharpe ratio
        if volatility_pct > 0:
            sharpe_ratio = (annualized_return_pct - self.risk_free_rate * 100) / volatility_pct
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_std = float(np.std(downside_returns) * np.sqrt(252) * 100)
            if downside_std > 0:
                sortino_ratio = (annualized_return_pct - self.risk_free_rate * 100) / downside_std
            else:
                sortino_ratio = 0.0
        else:
            sortino_ratio = sharpe_ratio

        # Drawdown calculations
        max_dd_pct, avg_dd_pct, dd_duration = self._calculate_drawdown(equities)

        # Calmar ratio
        if max_dd_pct > 0:
            calmar_ratio = annualized_return_pct / max_dd_pct
        else:
            calmar_ratio = 0.0

        # VaR and CVaR
        if len(returns_array) > 0:
            var_95_pct = float(-np.percentile(returns_array, 5) * 100)
            tail_returns = returns_array[returns_array <= np.percentile(returns_array, 5)]
            if len(tail_returns) > 0:
                cvar_95_pct = float(-np.mean(tail_returns) * 100)
            else:
                cvar_95_pct = var_95_pct
        else:
            var_95_pct = 0.0
            cvar_95_pct = 0.0

        # Trade statistics
        trade_stats = self._calculate_trade_statistics()

        # Dates
        start_date = equity_curve[0][0] if equity_curve else None
        end_date = equity_curve[-1][0] if equity_curve else None

        return BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trade_history=[t.to_dict() for t in self.portfolio.trade_history],
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            volatility_pct=volatility_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown_pct=max_dd_pct,
            avg_drawdown_pct=avg_dd_pct,
            drawdown_duration_days=dd_duration,
            var_95_pct=var_95_pct,
            cvar_95_pct=cvar_95_pct,
            n_trades=trade_stats["n_trades"],
            n_winning_trades=trade_stats["n_winning"],
            n_losing_trades=trade_stats["n_losing"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_trade_return_pct=trade_stats["avg_return"],
            avg_win_pct=trade_stats["avg_win"],
            avg_loss_pct=trade_stats["avg_loss"],
            max_consecutive_wins=trade_stats["max_consec_wins"],
            max_consecutive_losses=trade_stats["max_consec_losses"],
            avg_holding_period_days=trade_stats["avg_holding"],
            total_commission=self.portfolio.total_commission,
            total_slippage=self.portfolio.total_slippage,
            total_costs=self.portfolio.total_commission + self.portfolio.total_slippage,
            costs_pct_of_pnl=trade_stats["costs_pct"],
            start_date=start_date,
            end_date=end_date,
            n_bars=len(equity_curve),
            initial_capital=self.portfolio.initial_capital,
            final_equity=equities[-1] if equities else 0.0,
        )

    def _calculate_drawdown(
        self,
        equities: List[float],
    ) -> Tuple[float, float, int]:
        """
        Calculate drawdown metrics.

        Returns:
            Tuple of (max_drawdown_pct, avg_drawdown_pct, max_duration_days)
        """
        if len(equities) < 2:
            return 0.0, 0.0, 0

        equities_array = np.array(equities)
        running_max = np.maximum.accumulate(equities_array)
        drawdowns = (equities_array - running_max) / running_max * 100

        max_dd_pct = float(-np.min(drawdowns))
        avg_dd_pct = float(-np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0

        # Calculate duration
        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0
        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_dd_pct, avg_dd_pct, max_duration

    def _calculate_trade_statistics(self) -> Dict[str, Any]:
        """Calculate trade statistics."""
        trades = self.portfolio.trade_history

        if not trades:
            return {
                "n_trades": 0,
                "n_winning": 0,
                "n_losing": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_consec_wins": 0,
                "max_consec_losses": 0,
                "avg_holding": 0.0,
                "costs_pct": 0.0,
            }

        # Extract closed trades with P&L
        closed_trades = [t for t in trades if t.is_closed]

        if not closed_trades:
            return {
                "n_trades": len(trades),
                "n_winning": 0,
                "n_losing": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_return": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "max_consec_wins": 0,
                "max_consec_losses": 0,
                "avg_holding": 0.0,
                "costs_pct": 0.0,
            }

        pnls = [t.pnl for t in closed_trades]
        returns = [t.return_pct for t in closed_trades]

        n_trades = len(closed_trades)
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        n_winning = len(winning)
        n_losing = len(losing)
        win_rate = (n_winning / n_trades) * 100 if n_trades > 0 else 0.0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Average returns
        avg_return = np.mean(returns) if returns else 0.0
        avg_win = np.mean([t.return_pct for t in closed_trades if t.pnl > 0]) if n_winning > 0 else 0.0
        avg_loss = np.mean([t.return_pct for t in closed_trades if t.pnl <= 0]) if n_losing > 0 else 0.0

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        consec_wins = 0
        consec_losses = 0

        for t in closed_trades:
            if t.pnl > 0:
                consec_wins += 1
                consec_losses = 0
                max_consec_wins = max(max_consec_wins, consec_wins)
            else:
                consec_losses += 1
                consec_wins = 0
                max_consec_losses = max(max_consec_losses, consec_losses)

        # Holding period
        holding_periods = [
            t.holding_period for t in closed_trades
            if t.holding_period is not None
        ]
        avg_holding = np.mean(holding_periods) if holding_periods else 0.0

        # Costs as percentage of P&L
        total_pnl = sum(pnls)
        total_costs = self.portfolio.total_commission + self.portfolio.total_slippage
        costs_pct = (total_costs / abs(total_pnl) * 100) if total_pnl != 0 else 0.0

        return {
            "n_trades": n_trades,
            "n_winning": n_winning,
            "n_losing": n_losing,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_return": avg_return,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "avg_holding": avg_holding,
            "costs_pct": costs_pct,
        }

    def reset(self) -> None:
        """Reset engine for re-running backtest."""
        self.data_handler.reset()
        self.portfolio.reset()
        self.bar_count = 0

        # Clear event queue
        while not self.events.empty():
            self.events.get()

        logger.info("BacktestEngine reset")
