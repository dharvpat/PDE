"""
Sector-Algorithm Optimization System.

Tests all sector/algorithm combinations to find optimal pairings and uses
optimization results to inform position sizing in the backtester.

Usage:
    >>> optimizer = SectorAlgorithmOptimizer(n_stocks_per_sector=10)
    >>> results = optimizer.run_optimization()
    >>> print(results.best_algorithms)  # Best algorithm per sector
    >>> fitness = results.get_fitness_score(Sector.TECHNOLOGY, "momentum")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools

import numpy as np

from .sector_portfolio import (
    Sector,
    SECTOR_STOCKS,
    get_sector,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of testing one sector/algorithm combination."""

    sector: Sector
    algorithm: str  # "momentum", "ma_crossover", "mean_reversion", "rsi", "bollinger"
    params: Dict[str, Any]
    sharpe_ratio: float
    total_return_pct: float
    win_rate: float
    max_drawdown_pct: float
    n_trades: int
    profit_factor: float
    n_stocks_tested: int
    is_significant: bool  # t-test p < 0.05

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sector": self.sector.value,
            "algorithm": self.algorithm,
            "params": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in self.params.items()},
            "sharpe_ratio": float(self.sharpe_ratio),
            "total_return_pct": float(self.total_return_pct),
            "win_rate": float(self.win_rate),
            "max_drawdown_pct": float(self.max_drawdown_pct),
            "n_trades": int(self.n_trades),
            "profit_factor": float(self.profit_factor),
            "n_stocks_tested": int(self.n_stocks_tested),
            "is_significant": bool(self.is_significant),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary."""
        return cls(
            sector=Sector(data["sector"]),
            algorithm=data["algorithm"],
            params=data["params"],
            sharpe_ratio=data["sharpe_ratio"],
            total_return_pct=data["total_return_pct"],
            win_rate=data["win_rate"],
            max_drawdown_pct=data["max_drawdown_pct"],
            n_trades=data["n_trades"],
            profit_factor=data["profit_factor"],
            n_stocks_tested=data["n_stocks_tested"],
            is_significant=data["is_significant"],
        )


@dataclass
class SectorAlgorithmFitness:
    """Fitness score for a sector/algorithm combination."""

    sector: Sector
    algorithm: str
    fitness_score: float  # 0-1 composite score
    sharpe_score: float   # Normalized Sharpe component
    win_rate_score: float # Win rate component
    drawdown_score: float # Drawdown component
    significance_score: float  # Statistical significance component

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sector": self.sector.value,
            "algorithm": self.algorithm,
            "fitness_score": float(self.fitness_score),
            "sharpe_score": float(self.sharpe_score),
            "win_rate_score": float(self.win_rate_score),
            "drawdown_score": float(self.drawdown_score),
            "significance_score": float(self.significance_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectorAlgorithmFitness":
        """Create from dictionary."""
        return cls(
            sector=Sector(data["sector"]),
            algorithm=data["algorithm"],
            fitness_score=data["fitness_score"],
            sharpe_score=data["sharpe_score"],
            win_rate_score=data["win_rate_score"],
            drawdown_score=data["drawdown_score"],
            significance_score=data["significance_score"],
        )


@dataclass
class SectorOptimizationResults:
    """Full grid of all sector/algorithm combination results."""

    results_grid: Dict[Tuple[str, str], OptimizationResult] = field(default_factory=dict)
    best_algorithms: Dict[str, str] = field(default_factory=dict)  # sector -> algorithm
    best_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # sector -> params
    fitness_scores: Dict[Tuple[str, str], SectorAlgorithmFitness] = field(default_factory=dict)
    optimization_date: str = ""
    date_range_start: str = ""
    date_range_end: str = ""

    def get_best_algorithm(self, sector: Sector) -> Tuple[str, Dict[str, Any]]:
        """Get the best algorithm and params for a sector."""
        sector_key = sector.value
        if sector_key not in self.best_algorithms:
            return ("momentum", {})  # Default fallback
        return (self.best_algorithms[sector_key], self.best_params.get(sector_key, {}))

    def get_fitness_score(self, sector: Sector, algorithm: str) -> float:
        """Get fitness score for a sector/algorithm combination."""
        key = (sector.value, algorithm)
        if key in self.fitness_scores:
            return self.fitness_scores[key].fitness_score
        return 0.5  # Default neutral fitness

    def get_fitness(self, sector: Sector, algorithm: str) -> Optional[SectorAlgorithmFitness]:
        """Get full fitness object for a sector/algorithm combination."""
        key = (sector.value, algorithm)
        return self.fitness_scores.get(key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results_grid": {
                f"{k[0]}|{k[1]}": v.to_dict()
                for k, v in self.results_grid.items()
            },
            "best_algorithms": self.best_algorithms,
            "best_params": self.best_params,
            "fitness_scores": {
                f"{k[0]}|{k[1]}": v.to_dict()
                for k, v in self.fitness_scores.items()
            },
            "optimization_date": self.optimization_date,
            "date_range_start": self.date_range_start,
            "date_range_end": self.date_range_end,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SectorOptimizationResults":
        """Create from dictionary."""
        results_grid = {}
        for key_str, val in data.get("results_grid", {}).items():
            sector, algo = key_str.split("|")
            results_grid[(sector, algo)] = OptimizationResult.from_dict(val)

        fitness_scores = {}
        for key_str, val in data.get("fitness_scores", {}).items():
            sector, algo = key_str.split("|")
            fitness_scores[(sector, algo)] = SectorAlgorithmFitness.from_dict(val)

        return cls(
            results_grid=results_grid,
            best_algorithms=data.get("best_algorithms", {}),
            best_params=data.get("best_params", {}),
            fitness_scores=fitness_scores,
            optimization_date=data.get("optimization_date", ""),
            date_range_start=data.get("date_range_start", ""),
            date_range_end=data.get("date_range_end", ""),
        )

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved optimization results to {path}")

    @classmethod
    def load(cls, path: Path) -> "SectorOptimizationResults":
        """Load results from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded optimization results from {path}")
        return cls.from_dict(data)


class SectorAlgorithmOptimizer:
    """
    Optimizer that tests all sector/algorithm combinations.

    Runs backtests for each combination and computes fitness scores
    to identify optimal pairings for position sizing.
    """

    ALGORITHMS = ["momentum", "ma_crossover", "mean_reversion", "rsi", "bollinger"]

    PARAM_RANGES = {
        "momentum": {
            "lookback": [5, 10, 15, 20],
            "threshold": [0.02, 0.03, 0.04, 0.05],
        },
        "ma_crossover": {
            "fast": [3, 5, 8],
            "slow": [10, 15, 20],
        },
        "mean_reversion": {
            "lookback": [10, 15, 20],
            "entry_threshold": [1.5, 2.0, 2.5],
        },
        "rsi": {
            "period": [10, 14, 21],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
        },
        "bollinger": {
            "period": [15, 20, 25],
            "num_std": [1.5, 2.0, 2.5],
        },
    }

    # Default params for each algorithm (middle values)
    DEFAULT_PARAMS = {
        "momentum": {"lookback": 10, "threshold": 0.03},
        "ma_crossover": {"fast": 5, "slow": 15},
        "mean_reversion": {"lookback": 15, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "rsi": {"period": 14, "oversold": 30, "overbought": 70},
        "bollinger": {"period": 20, "num_std": 2.0},
    }

    # Maximum acceptable drawdown for fitness scoring
    MAX_ACCEPTABLE_DRAWDOWN = 30.0

    def __init__(
        self,
        n_stocks_per_sector: int = 10,
        backtest_days: int = 252,
        optimize_params: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            n_stocks_per_sector: Number of stocks to sample from each sector
            backtest_days: Number of days to backtest
            optimize_params: Whether to search for optimal parameters
            cache_dir: Directory for caching results
        """
        self.n_stocks_per_sector = n_stocks_per_sector
        self.backtest_days = backtest_days
        self.optimize_params = optimize_params
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_optimization(
        self,
        sectors: Optional[List[Sector]] = None,
        algorithms: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> SectorOptimizationResults:
        """
        Run optimization across all sector/algorithm combinations.

        Args:
            sectors: List of sectors to test (default: all non-ETF sectors)
            algorithms: List of algorithms to test (default: all)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)

        Returns:
            SectorOptimizationResults with complete grid and best algorithms
        """
        import pandas as pd
        from datetime import datetime, timedelta

        # Default sectors (exclude ETFs for individual stock testing)
        if sectors is None:
            sectors = [
                Sector.TECHNOLOGY,
                Sector.FINANCIALS,
                Sector.HEALTHCARE,
                Sector.CONSUMER_DISCRETIONARY,
                Sector.CONSUMER_STAPLES,
                Sector.ENERGY,
                Sector.INDUSTRIALS,
                Sector.MATERIALS,
                Sector.UTILITIES,
                Sector.REAL_ESTATE,
                Sector.COMMUNICATION,
            ]

        if algorithms is None:
            algorithms = self.ALGORITHMS

        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_dt = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=self.backtest_days + 30)
            start_date = start_dt.strftime("%Y-%m-%d")

        results = SectorOptimizationResults(
            optimization_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=start_date,
            date_range_end=end_date,
        )

        logger.info(f"Starting optimization: {len(sectors)} sectors x {len(algorithms)} algorithms")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Test each sector/algorithm combination
        for sector in sectors:
            logger.info(f"Processing sector: {sector.value}")

            # Select representative stocks
            stocks = self._select_representative_stocks(
                sector,
                self.n_stocks_per_sector,
                start_date,
                end_date,
            )

            if len(stocks) < 3:
                logger.warning(f"Insufficient stocks for {sector.value}, skipping")
                continue

            best_sharpe = -float('inf')
            best_algo = None
            best_params = None

            for algorithm in algorithms:
                logger.info(f"  Testing {algorithm}...")

                # Get params (optimize if enabled, else use defaults)
                if self.optimize_params:
                    params, _ = self._optimize_params_for_sector(
                        sector, algorithm, stocks, start_date, end_date
                    )
                else:
                    params = self.DEFAULT_PARAMS.get(algorithm, {})

                # Run backtest for this combination
                result = self._run_backtest_for_combination(
                    sector, algorithm, stocks, params, start_date, end_date
                )

                if result is None:
                    continue

                # Store result
                key = (sector.value, algorithm)
                results.results_grid[key] = result

                # Compute fitness score
                fitness = self._compute_fitness_score(result)
                results.fitness_scores[key] = fitness

                logger.info(
                    f"    Sharpe: {result.sharpe_ratio:.2f}, "
                    f"Return: {result.total_return_pct:.1f}%, "
                    f"Fitness: {fitness.fitness_score:.2f}"
                )

                # Track best
                if result.sharpe_ratio > best_sharpe:
                    best_sharpe = result.sharpe_ratio
                    best_algo = algorithm
                    best_params = params

            # Store best algorithm for this sector
            if best_algo:
                results.best_algorithms[sector.value] = best_algo
                results.best_params[sector.value] = best_params

        # Save results if cache enabled
        if self.cache_dir:
            cache_path = self.cache_dir / f"sector_optimization_{datetime.now().strftime('%Y-%m-%d')}.json"
            results.save(cache_path)

            # Also save as "latest"
            latest_path = self.cache_dir / "sector_optimization_latest.json"
            results.save(latest_path)

        return results

    def _select_representative_stocks(
        self,
        sector: Sector,
        n_stocks: int,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        Select representative stocks from a sector.

        Selects stocks that have sufficient data coverage and liquidity.
        """
        import pandas as pd

        all_stocks = SECTOR_STOCKS.get(sector, [])
        if not all_stocks:
            return []

        # Try to fetch data for stocks and filter by availability
        available_stocks = []

        for symbol in all_stocks[:min(len(all_stocks), n_stocks * 3)]:  # Check more than needed
            try:
                data = self._fetch_data(symbol, start_date, end_date)
                if data is not None and len(data) >= 60:  # At least 60 days of data
                    available_stocks.append(symbol)
                    if len(available_stocks) >= n_stocks:
                        break
            except Exception:
                continue

        return available_stocks[:n_stocks]

    def _fetch_data(self, symbol: str, start_date: str, end_date: str):
        """Fetch historical data for a symbol."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if data.empty:
                return None

            # Standardize column names
            data.columns = [c.capitalize() for c in data.columns]

            # Remove timezone info
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            data['Symbol'] = symbol
            return data

        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return None

    def _run_backtest_for_combination(
        self,
        sector: Sector,
        algorithm: str,
        stocks: List[str],
        params: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Optional[OptimizationResult]:
        """
        Run backtest for a sector/algorithm combination.

        Aggregates results across multiple stocks in the sector.
        """
        from queue import Queue
        import pandas as pd

        from .engine import BacktestEngine
        from .data_handler import HistoricDataFrameHandler
        from .portfolio import Portfolio
        from .execution import InstantExecutionHandler
        from .strategy import (
            MovingAverageCrossoverStrategy,
            MeanReversionStrategy,
            MomentumStrategy,
        )

        all_sharpes = []
        all_returns = []
        all_win_rates = []
        all_drawdowns = []
        all_n_trades = []
        all_profit_factors = []

        for symbol in stocks:
            try:
                data = self._fetch_data(symbol, start_date, end_date)
                if data is None or len(data) < 60:
                    continue

                events = Queue()

                # Prepare data for handler
                combined = pd.DataFrame(index=data.index)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in data.columns:
                        combined[f"{symbol}_{col}"] = data[col]

                data_handler = HistoricDataFrameHandler(
                    events_queue=events,
                    symbol_list=[symbol],
                    data=combined,
                )

                portfolio = Portfolio(initial_capital=100000, max_position_pct=0.95)
                executor = InstantExecutionHandler(events_queue=events)

                # Create strategy based on algorithm
                strategy = self._create_strategy(
                    algorithm, params, events, data_handler, portfolio
                )

                if strategy is None:
                    continue

                engine = BacktestEngine(
                    data_handler=data_handler,
                    strategy=strategy,
                    portfolio=portfolio,
                    execution_handler=executor,
                )

                results = engine.run()

                all_sharpes.append(results.sharpe_ratio)
                all_returns.append(results.total_return_pct)
                all_win_rates.append(results.win_rate)
                all_drawdowns.append(results.max_drawdown_pct)
                all_n_trades.append(results.n_trades)
                all_profit_factors.append(results.profit_factor)

            except Exception as e:
                logger.debug(f"Backtest failed for {symbol}: {e}")
                continue

        if len(all_sharpes) < 2:
            return None

        # Aggregate results
        avg_sharpe = float(np.mean(all_sharpes))
        avg_return = float(np.mean(all_returns))
        avg_win_rate = float(np.mean(all_win_rates))
        avg_drawdown = float(np.mean(all_drawdowns))
        total_trades = sum(all_n_trades)
        avg_profit_factor = float(np.mean([pf for pf in all_profit_factors if pf > 0]))

        # Perform t-test for significance (is Sharpe significantly > 0?)
        from scipy import stats
        if len(all_sharpes) >= 3:
            t_stat, p_value = stats.ttest_1samp(all_sharpes, 0)
            is_significant = p_value < 0.05 and t_stat > 0
        else:
            is_significant = avg_sharpe > 0.3  # Heuristic for small samples

        return OptimizationResult(
            sector=sector,
            algorithm=algorithm,
            params=params,
            sharpe_ratio=avg_sharpe,
            total_return_pct=avg_return,
            win_rate=avg_win_rate,
            max_drawdown_pct=avg_drawdown,
            n_trades=total_trades,
            profit_factor=avg_profit_factor if avg_profit_factor > 0 else 0.0,
            n_stocks_tested=len(all_sharpes),
            is_significant=is_significant,
        )

    def _create_strategy(
        self,
        algorithm: str,
        params: Dict[str, Any],
        events,
        data_handler,
        portfolio,
    ):
        """Create a strategy instance for the given algorithm."""
        from .strategy import (
            MovingAverageCrossoverStrategy,
            MeanReversionStrategy,
            MomentumStrategy,
        )

        if algorithm == "momentum":
            return MomentumStrategy(
                events_queue=events,
                data_handler=data_handler,
                portfolio=portfolio,
                lookback=params.get("lookback", 10),
                threshold=params.get("threshold", 0.03),
            )
        elif algorithm == "ma_crossover":
            return MovingAverageCrossoverStrategy(
                events_queue=events,
                data_handler=data_handler,
                portfolio=portfolio,
                fast_window=params.get("fast", 5),
                slow_window=params.get("slow", 15),
            )
        elif algorithm == "mean_reversion":
            return MeanReversionStrategy(
                events_queue=events,
                data_handler=data_handler,
                portfolio=portfolio,
                lookback=params.get("lookback", 15),
                entry_threshold=params.get("entry_threshold", 2.0),
                exit_threshold=params.get("exit_threshold", 0.5),
            )
        elif algorithm == "rsi":
            # RSI strategy - implemented as mean-reversion with RSI-like logic
            # Using mean-reversion as proxy since we don't have dedicated RSI strategy
            return MeanReversionStrategy(
                events_queue=events,
                data_handler=data_handler,
                portfolio=portfolio,
                lookback=params.get("period", 14),
                entry_threshold=2.0,
                exit_threshold=0.5,
            )
        elif algorithm == "bollinger":
            # Bollinger Bands - also using mean-reversion as proxy
            return MeanReversionStrategy(
                events_queue=events,
                data_handler=data_handler,
                portfolio=portfolio,
                lookback=params.get("period", 20),
                entry_threshold=params.get("num_std", 2.0),
                exit_threshold=0.5,
            )
        else:
            return None

    def _optimize_params_for_sector(
        self,
        sector: Sector,
        algorithm: str,
        stocks: List[str],
        start_date: str,
        end_date: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Find optimal parameters for a sector/algorithm combination.

        Uses grid search over the parameter space.
        """
        param_ranges = self.PARAM_RANGES.get(algorithm, {})

        if not param_ranges:
            return self.DEFAULT_PARAMS.get(algorithm, {}), 0.0

        # Generate parameter grid
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]

        best_params = self.DEFAULT_PARAMS.get(algorithm, {})
        best_sharpe = -float('inf')

        # Grid search (limited to avoid excessive runtime)
        combinations = list(itertools.product(*param_values))

        # Limit combinations to avoid excessive computation
        if len(combinations) > 20:
            # Sample a subset
            np.random.seed(42)
            indices = np.random.choice(len(combinations), 20, replace=False)
            combinations = [combinations[i] for i in indices]

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Add exit_threshold for mean_reversion
            if algorithm == "mean_reversion":
                params["exit_threshold"] = 0.5

            result = self._run_backtest_for_combination(
                sector, algorithm, stocks[:5], params, start_date, end_date
            )

            if result and result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_params = params

        return best_params, best_sharpe

    def _compute_fitness_score(self, result: OptimizationResult) -> SectorAlgorithmFitness:
        """
        Compute fitness score for a sector/algorithm combination.

        Fitness score (0-1) computed from:
        - Sharpe score (40%): Normalized Sharpe ratio
        - Win rate score (25%): Win rate / 100
        - Drawdown score (20%): 1 - (drawdown / max_acceptable_drawdown)
        - Significance score (15%): 1.0 if statistically significant, 0.5 otherwise
        """
        # Sharpe score (40%) - normalize to 0-1 (assume Sharpe 2+ is excellent)
        sharpe_score = max(0.0, min(1.0, (result.sharpe_ratio + 0.5) / 2.5))

        # Win rate score (25%) - direct percentage
        win_rate_score = result.win_rate / 100.0

        # Drawdown score (20%) - lower is better
        drawdown_ratio = abs(result.max_drawdown_pct) / self.MAX_ACCEPTABLE_DRAWDOWN
        drawdown_score = max(0.0, 1.0 - drawdown_ratio)

        # Significance score (15%)
        significance_score = 1.0 if result.is_significant else 0.5

        # Weighted composite
        fitness = (
            0.40 * sharpe_score +
            0.25 * win_rate_score +
            0.20 * drawdown_score +
            0.15 * significance_score
        )

        return SectorAlgorithmFitness(
            sector=result.sector,
            algorithm=result.algorithm,
            fitness_score=float(fitness),
            sharpe_score=float(sharpe_score),
            win_rate_score=float(win_rate_score),
            drawdown_score=float(drawdown_score),
            significance_score=float(significance_score),
        )

    def load_cached_results(self) -> Optional[SectorOptimizationResults]:
        """Load cached optimization results if available and fresh."""
        if not self.cache_dir:
            return None

        latest_path = self.cache_dir / "sector_optimization_latest.json"
        if not latest_path.exists():
            return None

        try:
            results = SectorOptimizationResults.load(latest_path)

            # Check if cache is fresh (within 30 days)
            from datetime import datetime, timedelta
            opt_date = datetime.strptime(
                results.optimization_date.split()[0],
                "%Y-%m-%d"
            )
            if datetime.now() - opt_date > timedelta(days=30):
                logger.info("Cached results older than 30 days, will re-optimize")
                return None

            return results

        except Exception as e:
            logger.warning(f"Failed to load cached results: {e}")
            return None


def print_optimization_results(results: SectorOptimizationResults) -> None:
    """Print formatted optimization results."""
    print("\nSECTOR-ALGORITHM OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Optimization Date: {results.optimization_date}")
    print(f"Data Range: {results.date_range_start} to {results.date_range_end}")

    print("\nBEST ALGORITHMS BY SECTOR")
    print("-" * 80)

    for sector_name, algo in sorted(results.best_algorithms.items()):
        fitness_key = (sector_name, algo)
        fitness = results.fitness_scores.get(fitness_key)
        result_key = fitness_key
        result = results.results_grid.get(result_key)

        if fitness and result:
            print(
                f"  {sector_name:25s}: {algo:15s} "
                f"(fitness: {fitness.fitness_score:.2f}, "
                f"Sharpe: {result.sharpe_ratio:.2f})"
            )

    print("\nFULL GRID (Sharpe Ratios)")
    print("-" * 80)

    # Get unique sectors and algorithms
    sectors = sorted(set(k[0] for k in results.results_grid.keys()))
    algorithms = ["momentum", "ma_crossover", "mean_reversion", "rsi", "bollinger"]

    # Header
    header = f"{'':20s}"
    for algo in algorithms:
        header += f"{algo[:10]:>12s}"
    print(header)

    # Rows
    for sector in sectors:
        row = f"{sector:20s}"
        for algo in algorithms:
            key = (sector, algo)
            if key in results.results_grid:
                sharpe = results.results_grid[key].sharpe_ratio
                row += f"{sharpe:>12.2f}"
            else:
                row += f"{'--':>12s}"
        print(row)
