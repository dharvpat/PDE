"""
Quantitative Trading System

A sophisticated quantitative trading platform leveraging stochastic volatility models,
optimal stopping theory, and PDE-based pricing for swing trading strategies.

Core components:
- Heston and SABR stochastic volatility model calibration
- Ornstein-Uhlenbeck mean-reversion detection and trading
- C++ accelerated numerical computations via pybind11
- PDE solvers for option pricing and optimal stopping
- Event-driven backtesting framework
- Full data pipeline with validation and streaming
- TimescaleDB-backed data storage

Usage:
    # As a library
    from quant_trading import TradingSystem
    system = TradingSystem()
    system.initialize()
    results = system.run_backtest(data)

    # As a CLI
    $ quant-trading demo
    $ quant-trading backtest --data prices.csv
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

# Import submodules (some have optional dependencies)
_available_modules = []

try:
    from . import calibration
    _available_modules.append("calibration")
except ImportError:
    calibration = None  # type: ignore

try:
    from . import database
    _available_modules.append("database")
except ImportError:
    database = None  # type: ignore

try:
    from . import models
    _available_modules.append("models")
except ImportError:
    models = None  # type: ignore

try:
    from . import signals
    _available_modules.append("signals")
except ImportError:
    signals = None  # type: ignore

try:
    from . import risk
    _available_modules.append("risk")
except ImportError:
    risk = None  # type: ignore

try:
    from . import execution
    _available_modules.append("execution")
except ImportError:
    execution = None  # type: ignore

try:
    from . import backtesting
    _available_modules.append("backtesting")
except ImportError:
    backtesting = None  # type: ignore

try:
    from . import data
    _available_modules.append("data")
except ImportError:
    data = None  # type: ignore

try:
    from . import monitoring
    _available_modules.append("monitoring")
except ImportError:
    monitoring = None  # type: ignore

try:
    from .config import Config, load_config
    from .trading_system import TradingSystem, create_trading_system
    _available_modules.extend(["config", "trading_system"])
except ImportError:
    Config = None  # type: ignore
    load_config = None  # type: ignore
    TradingSystem = None  # type: ignore
    create_trading_system = None  # type: ignore

__all__ = [
    "__version__",
    "calibration",
    "database",
    "models",
    "signals",
    "risk",
    "execution",
    "backtesting",
    "data",
    "monitoring",
    "Config",
    "load_config",
    "TradingSystem",
    "create_trading_system",
]
