"""
Walk-forward analysis and Monte Carlo simulation for strategy validation.

Provides:
    - WalkForwardAnalysis: Out-of-sample validation with expanding/rolling windows
    - MonteCarloSimulator: Bootstrap simulation for confidence intervals
    - ParameterOptimizer: Grid/random search for strategy parameters

Reference:
    - Pardo (2008) - "The Evaluation and Optimization of Trading Strategies"
    - White (2000) - "A Reality Check for Data Snooping"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .engine import BacktestEngine, BacktestResults
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class WindowType(Enum):
    """Walk-forward window type."""

    ANCHORED = "anchored"  # Expanding window
    ROLLING = "rolling"  # Fixed window size


@dataclass
class WalkForwardPeriod:
    """A single walk-forward period (in-sample + out-of-sample)."""

    period_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_sample_start: datetime
    out_sample_end: datetime
    in_sample_bars: int = 0
    out_sample_bars: int = 0

    # Results
    in_sample_result: Optional[BacktestResults] = None
    out_sample_result: Optional[BacktestResults] = None
    optimized_params: Optional[Dict[str, Any]] = None

    @property
    def oos_sharpe(self) -> float:
        """Out-of-sample Sharpe ratio."""
        if self.out_sample_result:
            return self.out_sample_result.sharpe_ratio
        return 0.0

    @property
    def is_sharpe(self) -> float:
        """In-sample Sharpe ratio."""
        if self.in_sample_result:
            return self.in_sample_result.sharpe_ratio
        return 0.0

    @property
    def sharpe_decay(self) -> float:
        """Decay from in-sample to out-of-sample Sharpe."""
        if self.is_sharpe > 0:
            return (self.is_sharpe - self.oos_sharpe) / self.is_sharpe
        return 0.0


@dataclass
class WalkForwardResults:
    """Results from walk-forward analysis."""

    periods: List[WalkForwardPeriod]
    window_type: WindowType
    in_sample_pct: float
    out_sample_pct: float

    # Aggregate metrics
    combined_oos_result: Optional[BacktestResults] = None

    @property
    def n_periods(self) -> int:
        """Number of walk-forward periods."""
        return len(self.periods)

    @property
    def avg_oos_sharpe(self) -> float:
        """Average out-of-sample Sharpe ratio."""
        sharpes = [p.oos_sharpe for p in self.periods]
        return float(np.mean(sharpes)) if sharpes else 0.0

    @property
    def avg_is_sharpe(self) -> float:
        """Average in-sample Sharpe ratio."""
        sharpes = [p.is_sharpe for p in self.periods]
        return float(np.mean(sharpes)) if sharpes else 0.0

    @property
    def avg_sharpe_decay(self) -> float:
        """Average Sharpe decay."""
        decays = [p.sharpe_decay for p in self.periods]
        return float(np.mean(decays)) if decays else 0.0

    @property
    def oos_win_rate(self) -> float:
        """Fraction of OOS periods with positive Sharpe."""
        if not self.periods:
            return 0.0
        positive = sum(1 for p in self.periods if p.oos_sharpe > 0)
        return positive / len(self.periods)

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
================================================================================
                        WALK-FORWARD ANALYSIS RESULTS
================================================================================
Window Type: {self.window_type.value}
In-Sample %: {self.in_sample_pct:.0%}
Out-Sample %: {self.out_sample_pct:.0%}
Number of Periods: {self.n_periods}

IN-SAMPLE METRICS (Average)
---------------------------
Sharpe Ratio:        {self.avg_is_sharpe:>8.2f}

OUT-OF-SAMPLE METRICS (Average)
-------------------------------
Sharpe Ratio:        {self.avg_oos_sharpe:>8.2f}
Win Rate (Positive): {self.oos_win_rate:>8.1%}
Sharpe Decay:        {self.avg_sharpe_decay:>8.1%}

COMBINED OOS PERFORMANCE
------------------------
{self._format_combined_oos()}
================================================================================
"""

    def _format_combined_oos(self) -> str:
        """Format combined OOS results."""
        if not self.combined_oos_result:
            return "No combined results available"

        r = self.combined_oos_result
        return f"""Total Return:        {r.total_return_pct:>8.2f}%
Sharpe Ratio:        {r.sharpe_ratio:>8.2f}
Max Drawdown:        {r.max_drawdown_pct:>8.2f}%
Win Rate:            {r.win_rate:>8.1f}%"""


class WalkForwardAnalysis:
    """
    Walk-forward analysis for strategy validation.

    Divides historical data into multiple in-sample/out-of-sample periods
    to test strategy robustness and prevent overfitting.

    Example:
        >>> wfa = WalkForwardAnalysis(
        ...     data_handler_cls=HistoricDataFrameHandler,
        ...     strategy_cls=MACrossoverStrategy,
        ...     in_sample_pct=0.7,
        ...     n_periods=5,
        ...     window_type=WindowType.ROLLING
        ... )
        >>> results = wfa.run(data, strategy_params, optimize_params=['fast_window', 'slow_window'])
    """

    def __init__(
        self,
        data_handler_factory: Callable,
        strategy_factory: Callable,
        execution_handler_factory: Callable,
        in_sample_pct: float = 0.7,
        n_periods: int = 5,
        window_type: WindowType = WindowType.ROLLING,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize walk-forward analysis.

        Args:
            data_handler_factory: Factory function to create data handler
            strategy_factory: Factory function to create strategy
            execution_handler_factory: Factory function to create executor
            in_sample_pct: Fraction of each period for in-sample (0.5-0.9)
            n_periods: Number of walk-forward periods
            window_type: ANCHORED (expanding) or ROLLING (fixed)
            initial_capital: Starting capital for each period
        """
        self.data_handler_factory = data_handler_factory
        self.strategy_factory = strategy_factory
        self.execution_handler_factory = execution_handler_factory
        self.in_sample_pct = in_sample_pct
        self.out_sample_pct = 1.0 - in_sample_pct
        self.n_periods = n_periods
        self.window_type = window_type
        self.initial_capital = initial_capital

        logger.info(
            f"WalkForwardAnalysis initialized: {window_type.value}, "
            f"{n_periods} periods, IS={in_sample_pct:.0%}"
        )

    def run(
        self,
        data: Any,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        optimize_params: Optional[List[str]] = None,
        param_ranges: Optional[Dict[str, List[Any]]] = None,
    ) -> WalkForwardResults:
        """
        Run walk-forward analysis.

        Args:
            data: Historical data (DataFrame or dict)
            strategy_params: Base strategy parameters
            symbols: List of symbols to trade
            optimize_params: Parameters to optimize (optional)
            param_ranges: Range of values for each parameter (optional)

        Returns:
            WalkForwardResults with period-by-period metrics
        """
        logger.info("Starting walk-forward analysis...")

        # Calculate period boundaries
        periods = self._calculate_periods(data)

        # Run each period
        for period in periods:
            logger.info(f"Running period {period.period_id + 1}/{len(periods)}")

            # In-sample: optimize if requested
            if optimize_params and param_ranges:
                optimized = self._optimize_period(
                    data,
                    period,
                    strategy_params,
                    symbols,
                    optimize_params,
                    param_ranges,
                    is_in_sample=True,
                )
                period.optimized_params = optimized
                params_to_use = {**strategy_params, **optimized}
            else:
                params_to_use = strategy_params

            # Run in-sample backtest
            period.in_sample_result = self._run_backtest(
                data, period, params_to_use, symbols, is_in_sample=True
            )

            # Run out-of-sample backtest
            period.out_sample_result = self._run_backtest(
                data, period, params_to_use, symbols, is_in_sample=False
            )

        # Combine OOS results
        combined_oos = self._combine_oos_results(periods)

        results = WalkForwardResults(
            periods=periods,
            window_type=self.window_type,
            in_sample_pct=self.in_sample_pct,
            out_sample_pct=self.out_sample_pct,
            combined_oos_result=combined_oos,
        )

        logger.info(
            f"Walk-forward complete: avg OOS Sharpe = {results.avg_oos_sharpe:.2f}"
        )

        return results

    def _calculate_periods(self, data: Any) -> List[WalkForwardPeriod]:
        """Calculate walk-forward period boundaries."""
        # Get data length
        if hasattr(data, "index"):
            dates = list(data.index)
        elif isinstance(data, dict) and data:
            first_symbol = list(data.keys())[0]
            if hasattr(data[first_symbol], "index"):
                dates = list(data[first_symbol].index)
            else:
                n_bars = len(data[first_symbol])
                dates = list(range(n_bars))
        else:
            raise ValueError("Cannot determine data length")

        n_bars = len(dates)
        periods = []

        if self.window_type == WindowType.ROLLING:
            # Rolling window: fixed size, moves forward
            period_size = n_bars // self.n_periods
            is_size = int(period_size * self.in_sample_pct)
            oos_size = period_size - is_size

            for i in range(self.n_periods):
                start_idx = i * period_size
                is_end_idx = start_idx + is_size
                oos_end_idx = start_idx + period_size

                if oos_end_idx > n_bars:
                    break

                period = WalkForwardPeriod(
                    period_id=i,
                    in_sample_start=dates[start_idx],
                    in_sample_end=dates[is_end_idx - 1],
                    out_sample_start=dates[is_end_idx],
                    out_sample_end=dates[oos_end_idx - 1],
                    in_sample_bars=is_size,
                    out_sample_bars=oos_size,
                )
                periods.append(period)

        else:  # ANCHORED
            # Anchored: expanding in-sample, fixed OOS
            oos_size = n_bars // (self.n_periods + 1)

            for i in range(self.n_periods):
                is_end_idx = (i + 1) * oos_size + oos_size
                oos_end_idx = is_end_idx + oos_size

                if oos_end_idx > n_bars:
                    break

                period = WalkForwardPeriod(
                    period_id=i,
                    in_sample_start=dates[0],  # Always from start
                    in_sample_end=dates[is_end_idx - 1],
                    out_sample_start=dates[is_end_idx],
                    out_sample_end=dates[oos_end_idx - 1],
                    in_sample_bars=is_end_idx,
                    out_sample_bars=oos_size,
                )
                periods.append(period)

        return periods

    def _run_backtest(
        self,
        data: Any,
        period: WalkForwardPeriod,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        is_in_sample: bool,
    ) -> BacktestResults:
        """Run backtest for a single period."""
        # Get date range
        if is_in_sample:
            start = period.in_sample_start
            end = period.in_sample_end
        else:
            start = period.out_sample_start
            end = period.out_sample_end

        # Filter data for period
        period_data = self._filter_data(data, start, end)

        # Create components
        events = Queue()

        data_handler = self.data_handler_factory(
            events_queue=events,
            data=period_data,
            symbol_list=symbols,
        )

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
        )

        executor = self.execution_handler_factory(events_queue=events)

        strategy = self.strategy_factory(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            **strategy_params,
        )

        # Run backtest
        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        return engine.run()

    def _filter_data(self, data: Any, start: Any, end: Any) -> Any:
        """Filter data for date range."""
        if hasattr(data, "loc"):
            # DataFrame
            return data.loc[start:end]
        elif isinstance(data, dict):
            # Dict of DataFrames
            return {
                symbol: df.loc[start:end] if hasattr(df, "loc") else df
                for symbol, df in data.items()
            }
        return data

    def _optimize_period(
        self,
        data: Any,
        period: WalkForwardPeriod,
        base_params: Dict[str, Any],
        symbols: List[str],
        optimize_params: List[str],
        param_ranges: Dict[str, List[Any]],
        is_in_sample: bool,
    ) -> Dict[str, Any]:
        """Optimize parameters on in-sample data using grid search."""
        best_params = {}
        best_sharpe = -np.inf

        # Generate parameter combinations
        combinations = self._generate_param_combinations(optimize_params, param_ranges)

        for combo in combinations:
            params = {**base_params, **combo}

            try:
                result = self._run_backtest(
                    data, period, params, symbols, is_in_sample=True
                )
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_params = combo
            except Exception as e:
                logger.warning(f"Parameter combo failed: {combo}, error: {e}")
                continue

        logger.debug(f"Best params: {best_params}, Sharpe: {best_sharpe:.2f}")
        return best_params

    def _generate_param_combinations(
        self,
        param_names: List[str],
        param_ranges: Dict[str, List[Any]],
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations (grid search)."""
        if not param_names:
            return [{}]

        combinations = [{}]
        for param in param_names:
            if param not in param_ranges:
                continue

            new_combinations = []
            for value in param_ranges[param]:
                for combo in combinations:
                    new_combo = {**combo, param: value}
                    new_combinations.append(new_combo)
            combinations = new_combinations

        return combinations

    def _combine_oos_results(
        self,
        periods: List[WalkForwardPeriod],
    ) -> Optional[BacktestResults]:
        """Combine out-of-sample results from all periods."""
        oos_results = [p.out_sample_result for p in periods if p.out_sample_result]

        if not oos_results:
            return None

        # Combine equity curves
        combined_equity = []
        combined_returns = []

        for result in oos_results:
            combined_equity.extend(result.equity_curve)
            combined_returns.extend(result.returns)

        # Calculate combined metrics
        equities = [eq for _, eq in combined_equity]
        returns_array = np.array(combined_returns)

        total_return_pct = ((equities[-1] / equities[0]) - 1) * 100 if equities else 0

        n_years = len(combined_returns) / 252
        if n_years > 0 and total_return_pct > -100:
            annualized = ((1 + total_return_pct / 100) ** (1 / n_years) - 1) * 100
        else:
            annualized = total_return_pct

        vol_pct = float(np.std(returns_array) * np.sqrt(252) * 100) if len(returns_array) > 0 else 0
        sharpe = (annualized - 5) / vol_pct if vol_pct > 0 else 0

        # Drawdown
        if len(equities) > 1:
            eq_arr = np.array(equities)
            running_max = np.maximum.accumulate(eq_arr)
            drawdowns = (eq_arr - running_max) / running_max * 100
            max_dd = float(-np.min(drawdowns))
        else:
            max_dd = 0.0

        # Aggregate trades
        n_trades = sum(r.n_trades for r in oos_results)
        n_winning = sum(r.n_winning_trades for r in oos_results)
        win_rate = (n_winning / n_trades * 100) if n_trades > 0 else 0

        return BacktestResults(
            equity_curve=combined_equity,
            returns=combined_returns,
            trade_history=[],
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            volatility_pct=vol_pct,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            n_trades=n_trades,
            n_winning_trades=n_winning,
            win_rate=win_rate,
        )


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""

    n_simulations: int
    original_result: BacktestResults

    # Simulated metrics distributions
    sharpe_ratios: np.ndarray = field(default_factory=lambda: np.array([]))
    total_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    max_drawdowns: np.ndarray = field(default_factory=lambda: np.array([]))
    final_equities: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_confidence_interval(
        self,
        metric: str,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Get confidence interval for a metric.

        Args:
            metric: One of 'sharpe', 'return', 'drawdown', 'equity'
            confidence: Confidence level (0.9, 0.95, 0.99)

        Returns:
            (lower_bound, upper_bound)
        """
        alpha = (1 - confidence) / 2
        percentiles = [alpha * 100, (1 - alpha) * 100]

        metric_map = {
            "sharpe": self.sharpe_ratios,
            "return": self.total_returns,
            "drawdown": self.max_drawdowns,
            "equity": self.final_equities,
        }

        if metric not in metric_map:
            raise ValueError(f"Unknown metric: {metric}")

        data = metric_map[metric]
        if len(data) == 0:
            return (0.0, 0.0)

        return float(np.percentile(data, percentiles[0])), float(
            np.percentile(data, percentiles[1])
        )

    def get_probability_of_loss(self) -> float:
        """Get probability of total return being negative."""
        if len(self.total_returns) == 0:
            return 0.0
        return float(np.mean(self.total_returns < 0))

    def get_probability_of_drawdown(self, threshold: float) -> float:
        """Get probability of max drawdown exceeding threshold."""
        if len(self.max_drawdowns) == 0:
            return 0.0
        return float(np.mean(self.max_drawdowns > threshold))

    def summary(self) -> str:
        """Generate summary string."""
        sharpe_ci = self.get_confidence_interval("sharpe", 0.95)
        return_ci = self.get_confidence_interval("return", 0.95)
        dd_ci = self.get_confidence_interval("drawdown", 0.95)

        return f"""
================================================================================
                        MONTE CARLO SIMULATION RESULTS
================================================================================
Number of Simulations: {self.n_simulations}

ORIGINAL BACKTEST
-----------------
Sharpe Ratio:     {self.original_result.sharpe_ratio:>8.2f}
Total Return:     {self.original_result.total_return_pct:>8.2f}%
Max Drawdown:     {self.original_result.max_drawdown_pct:>8.2f}%

SIMULATED DISTRIBUTIONS (95% Confidence Intervals)
--------------------------------------------------
Sharpe Ratio:     [{sharpe_ci[0]:>6.2f}, {sharpe_ci[1]:>6.2f}]
Total Return:     [{return_ci[0]:>6.2f}%, {return_ci[1]:>6.2f}%]
Max Drawdown:     [{dd_ci[0]:>6.2f}%, {dd_ci[1]:>6.2f}%]

RISK METRICS
------------
Prob. of Loss:           {self.get_probability_of_loss():>8.1%}
Prob. DD > 20%:          {self.get_probability_of_drawdown(20):>8.1%}
Prob. DD > 30%:          {self.get_probability_of_drawdown(30):>8.1%}
================================================================================
"""


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy robustness testing.

    Uses bootstrapping to generate synthetic return sequences
    and estimate confidence intervals for performance metrics.

    Methods:
        1. Return shuffling: Shuffle historical returns
        2. Block bootstrap: Shuffle blocks to preserve autocorrelation
        3. Parametric: Sample from fitted distribution

    Example:
        >>> mc = MonteCarloSimulator(n_simulations=1000, method='block')
        >>> mc_results = mc.run(backtest_results)
        >>> print(mc_results.get_confidence_interval('sharpe', 0.95))
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        method: str = "block",
        block_size: int = 21,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulations to run
            method: 'shuffle', 'block', or 'parametric'
            block_size: Block size for block bootstrap (trading days)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.method = method
        self.block_size = block_size

        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(
            f"MonteCarloSimulator initialized: {n_simulations} sims, method={method}"
        )

    def run(
        self,
        backtest_result: BacktestResults,
        initial_capital: Optional[float] = None,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.

        Args:
            backtest_result: Original backtest results
            initial_capital: Starting capital (uses backtest initial if None)

        Returns:
            MonteCarloResults with simulated distributions
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations...")

        returns = np.array(backtest_result.returns)
        if len(returns) == 0:
            logger.warning("No returns to simulate")
            return MonteCarloResults(
                n_simulations=0,
                original_result=backtest_result,
            )

        capital = initial_capital or backtest_result.initial_capital

        # Generate simulated return sequences
        simulated_returns = self._generate_simulated_returns(returns)

        # Calculate metrics for each simulation
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []
        final_equities = []

        for sim_returns in simulated_returns:
            metrics = self._calculate_path_metrics(sim_returns, capital)
            sharpe_ratios.append(metrics["sharpe"])
            total_returns.append(metrics["total_return"])
            max_drawdowns.append(metrics["max_drawdown"])
            final_equities.append(metrics["final_equity"])

        results = MonteCarloResults(
            n_simulations=self.n_simulations,
            original_result=backtest_result,
            sharpe_ratios=np.array(sharpe_ratios),
            total_returns=np.array(total_returns),
            max_drawdowns=np.array(max_drawdowns),
            final_equities=np.array(final_equities),
        )

        logger.info(
            f"Monte Carlo complete: Sharpe 95% CI = "
            f"[{results.get_confidence_interval('sharpe')[0]:.2f}, "
            f"{results.get_confidence_interval('sharpe')[1]:.2f}]"
        )

        return results

    def _generate_simulated_returns(
        self,
        returns: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate simulated return sequences."""
        if self.method == "shuffle":
            return self._shuffle_bootstrap(returns)
        elif self.method == "block":
            return self._block_bootstrap(returns)
        elif self.method == "parametric":
            return self._parametric_bootstrap(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _shuffle_bootstrap(self, returns: np.ndarray) -> List[np.ndarray]:
        """Simple shuffle bootstrap (IID assumption)."""
        n = len(returns)
        simulations = []

        for _ in range(self.n_simulations):
            indices = np.random.choice(n, size=n, replace=True)
            simulations.append(returns[indices])

        return simulations

    def _block_bootstrap(self, returns: np.ndarray) -> List[np.ndarray]:
        """
        Block bootstrap to preserve autocorrelation.

        Splits returns into blocks and shuffles blocks.
        """
        n = len(returns)
        n_blocks = int(np.ceil(n / self.block_size))
        simulations = []

        # Create blocks
        blocks = []
        for i in range(0, n, self.block_size):
            end = min(i + self.block_size, n)
            blocks.append(returns[i:end])

        for _ in range(self.n_simulations):
            # Sample blocks with replacement
            sampled_blocks = [
                blocks[np.random.randint(len(blocks))] for _ in range(n_blocks)
            ]
            # Concatenate and trim to original length
            sim_returns = np.concatenate(sampled_blocks)[:n]
            simulations.append(sim_returns)

        return simulations

    def _parametric_bootstrap(self, returns: np.ndarray) -> List[np.ndarray]:
        """
        Parametric bootstrap assuming normal distribution.

        Fits mean and std, then samples from fitted distribution.
        """
        mean = np.mean(returns)
        std = np.std(returns)
        n = len(returns)

        simulations = []
        for _ in range(self.n_simulations):
            sim_returns = np.random.normal(mean, std, size=n)
            simulations.append(sim_returns)

        return simulations

    def _calculate_path_metrics(
        self,
        returns: np.ndarray,
        initial_capital: float,
    ) -> Dict[str, float]:
        """Calculate metrics for a single simulated path."""
        # Build equity curve
        equity = [initial_capital]
        for ret in returns:
            equity.append(equity[-1] * (1 + ret))

        equity_array = np.array(equity)
        final_equity = equity_array[-1]
        total_return = (final_equity / initial_capital - 1) * 100

        # Volatility and Sharpe
        annual_vol = float(np.std(returns) * np.sqrt(252) * 100)
        n_years = len(returns) / 252
        if n_years > 0 and total_return > -100:
            annual_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100
        else:
            annual_return = total_return

        sharpe = (annual_return - 5) / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max * 100
        max_dd = float(-np.min(drawdowns))

        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "final_equity": final_equity,
        }


class ParameterSensitivity:
    """
    Parameter sensitivity analysis.

    Tests how strategy performance varies with parameter changes
    to identify robust parameter regions.

    Example:
        >>> sensitivity = ParameterSensitivity(
        ...     data_handler_factory=...,
        ...     strategy_factory=...,
        ...     execution_handler_factory=...,
        ... )
        >>> results = sensitivity.analyze(
        ...     data, symbols,
        ...     param_name='lookback',
        ...     param_values=[10, 20, 30, 40, 50],
        ... )
    """

    def __init__(
        self,
        data_handler_factory: Callable,
        strategy_factory: Callable,
        execution_handler_factory: Callable,
        initial_capital: float = 100000.0,
    ):
        """Initialize parameter sensitivity analyzer."""
        self.data_handler_factory = data_handler_factory
        self.strategy_factory = strategy_factory
        self.execution_handler_factory = execution_handler_factory
        self.initial_capital = initial_capital

    def analyze(
        self,
        data: Any,
        symbols: List[str],
        param_name: str,
        param_values: List[Any],
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to a single parameter.

        Args:
            data: Historical data
            symbols: Symbols to trade
            param_name: Parameter to vary
            param_values: Values to test
            base_params: Base strategy parameters

        Returns:
            Dict with parameter values and corresponding metrics
        """
        base_params = base_params or {}
        results = {
            "parameter": param_name,
            "values": param_values,
            "sharpe_ratios": [],
            "total_returns": [],
            "max_drawdowns": [],
            "win_rates": [],
        }

        for value in param_values:
            params = {**base_params, param_name: value}

            try:
                bt_result = self._run_backtest(data, symbols, params)
                results["sharpe_ratios"].append(bt_result.sharpe_ratio)
                results["total_returns"].append(bt_result.total_return_pct)
                results["max_drawdowns"].append(bt_result.max_drawdown_pct)
                results["win_rates"].append(bt_result.win_rate)
            except Exception as e:
                logger.warning(f"Failed for {param_name}={value}: {e}")
                results["sharpe_ratios"].append(np.nan)
                results["total_returns"].append(np.nan)
                results["max_drawdowns"].append(np.nan)
                results["win_rates"].append(np.nan)

        return results

    def _run_backtest(
        self,
        data: Any,
        symbols: List[str],
        strategy_params: Dict[str, Any],
    ) -> BacktestResults:
        """Run a single backtest."""
        events = Queue()

        data_handler = self.data_handler_factory(
            events_queue=events,
            data=data,
            symbol_list=symbols,
        )

        portfolio = Portfolio(initial_capital=self.initial_capital)
        executor = self.execution_handler_factory(events_queue=events)

        strategy = self.strategy_factory(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            **strategy_params,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        return engine.run()
