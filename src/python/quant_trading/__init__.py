"""
Quantitative Trading System

A sophisticated quantitative trading platform leveraging stochastic volatility models,
optimal stopping theory, and PDE-based pricing for swing trading strategies.

Core components:
- Heston and SABR stochastic volatility model calibration
- Ornstein-Uhlenbeck mean-reversion detection and trading
- C++ accelerated numerical computations via pybind11
"""

__version__ = "1.0.0"
__author__ = "Quantitative Research Team"

from . import calibration
from . import models
from . import signals
from . import risk
from . import execution

__all__ = [
    "__version__",
    "calibration",
    "models",
    "signals",
    "risk",
    "execution",
]
