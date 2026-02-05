"""
Backtesting framework for quantitative trading strategies.

This module provides a complete event-driven backtesting system with:
    - Event-based architecture (Market → Signal → Order → Fill)
    - Realistic execution simulation (slippage, market impact, commissions)
    - Comprehensive performance metrics and risk analysis
    - Walk-forward analysis for out-of-sample validation
    - Monte Carlo simulation for confidence intervals

Architecture:
    DataHandler → emits MarketEvents
    Strategy → receives MarketEvents, emits SignalEvents
    Portfolio → receives SignalEvents, emits OrderEvents
    ExecutionHandler → receives OrderEvents, emits FillEvents
    BacktestEngine → coordinates all components

Example:
    >>> from quant_trading.backtesting import (
    ...     BacktestEngine, Portfolio, SyntheticDataHandler,
    ...     SimulatedExecutionHandler, MovingAverageCrossoverStrategy
    ... )
    >>> from queue import Queue
    >>>
    >>> # Create event queue (shared by all components)
    >>> events = Queue()
    >>>
    >>> # Create data handler
    >>> data_handler = SyntheticDataHandler(
    ...     events_queue=events,
    ...     symbol_list=['SPY', 'QQQ'],
    ...     n_bars=252,
    ...     start_price=100.0,
    ... )
    >>>
    >>> # Create portfolio
    >>> portfolio = Portfolio(initial_capital=100000)
    >>>
    >>> # Create execution handler with realistic costs
    >>> executor = SimulatedExecutionHandler(
    ...     events_queue=events,
    ...     slippage_bps=5,
    ...     market_impact_factor=0.1,
    ... )
    >>>
    >>> # Create strategy
    >>> strategy = MovingAverageCrossoverStrategy(
    ...     events_queue=events,
    ...     data_handler=data_handler,
    ...     portfolio=portfolio,
    ...     fast_window=10,
    ...     slow_window=50,
    ... )
    >>>
    >>> # Run backtest
    >>> engine = BacktestEngine(
    ...     data_handler=data_handler,
    ...     strategy=strategy,
    ...     portfolio=portfolio,
    ...     execution_handler=executor,
    ... )
    >>> results = engine.run()
    >>> print(results.summary())

References:
    - Pardo (2008): Trading strategy evaluation and optimization
    - Almgren & Chriss (2001): Market impact modeling
    - Kissell & Glantz (2003): Transaction cost analysis
"""

from .events import (
    Direction,
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)

from .portfolio import (
    Portfolio,
    Position,
    Trade,
)

from .data_handler import (
    DataHandler,
    HistoricCSVDataHandler,
    HistoricDataFrameHandler,
    SyntheticDataHandler,
)

from .execution import (
    CommissionModel,
    ExecutionHandler,
    FixedCommission,
    IBKRCommission,
    InstantExecutionHandler,
    PerShareCommission,
    SimulatedExecutionHandler,
    TieredCommission,
    ZeroCommission,
)

from .strategy import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    Strategy,
)

from .multi_strategy import (
    MultiStrategyManager,
    OPTIMAL_STRATEGIES,
    get_optimal_strategy,
)

from .sector_portfolio import (
    Sector,
    SECTOR_STOCKS,
    SECTOR_STRATEGIES,
    STOCK_TO_SECTOR,
    get_sector,
    get_sector_strategy,
    get_all_stocks,
    get_stocks_by_sector,
    ConfidenceMetrics,
    ConfidenceCalculator,
    calculate_position_size,
)

from .sector_optimizer import (
    OptimizationResult,
    SectorAlgorithmFitness,
    SectorOptimizationResults,
    SectorAlgorithmOptimizer,
    print_optimization_results,
)

from .rolling_optimizer import (
    PeriodResult,
    RollingBacktestResults,
    RollingOptimizationBacktester,
)

from .engine import (
    BacktestEngine,
    BacktestResults,
)

from .analysis import (
    MonteCarloResults,
    MonteCarloSimulator,
    ParameterSensitivity,
    WalkForwardAnalysis,
    WalkForwardPeriod,
    WalkForwardResults,
    WindowType,
)


__all__ = [
    # Events
    "Event",
    "EventType",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "SignalType",
    "OrderType",
    "Direction",
    # Portfolio
    "Portfolio",
    "Position",
    "Trade",
    # Data Handlers
    "DataHandler",
    "HistoricDataFrameHandler",
    "HistoricCSVDataHandler",
    "SyntheticDataHandler",
    # Execution
    "ExecutionHandler",
    "SimulatedExecutionHandler",
    "InstantExecutionHandler",
    "CommissionModel",
    "ZeroCommission",
    "FixedCommission",
    "PerShareCommission",
    "TieredCommission",
    "IBKRCommission",
    # Strategies
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossoverStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MultiStrategyManager",
    "OPTIMAL_STRATEGIES",
    "get_optimal_strategy",
    # Sector portfolio
    "Sector",
    "SECTOR_STOCKS",
    "SECTOR_STRATEGIES",
    "STOCK_TO_SECTOR",
    "get_sector",
    "get_sector_strategy",
    "get_all_stocks",
    "get_stocks_by_sector",
    "ConfidenceMetrics",
    "ConfidenceCalculator",
    "calculate_position_size",
    # Sector optimizer
    "OptimizationResult",
    "SectorAlgorithmFitness",
    "SectorOptimizationResults",
    "SectorAlgorithmOptimizer",
    "print_optimization_results",
    # Rolling optimizer
    "PeriodResult",
    "RollingBacktestResults",
    "RollingOptimizationBacktester",
    # Engine
    "BacktestEngine",
    "BacktestResults",
    # Analysis
    "WalkForwardAnalysis",
    "WalkForwardPeriod",
    "WalkForwardResults",
    "WindowType",
    "MonteCarloSimulator",
    "MonteCarloResults",
    "ParameterSensitivity",
]
