"""
Signal Generation Module.

Implements trading signal generation based on:
- Volatility surface arbitrage (Heston/SABR vs market IV)
- Mean-reversion optimal entry/exit (Ornstein-Uhlenbeck)
- Signal aggregation and confidence scoring

Components:
    - VolSurfaceArbitrageSignal: Detects option mispricing
    - MeanReversionSignalGenerator: OU-based spread trading
    - SignalAggregator: Combines multi-strategy signals

Example:
    >>> from quant_trading.signals import (
    ...     VolSurfaceArbitrageSignal,
    ...     MeanReversionSignalGenerator,
    ...     SignalAggregator
    ... )
    >>> vol_gen = VolSurfaceArbitrageSignal()
    >>> mr_gen = MeanReversionSignalGenerator()
    >>> aggregator = SignalAggregator()
"""

from .aggregator import (
    AggregatedSignal,
    AggregatedSignalType,
    AggregatorConfig,
    SignalAggregator,
)
from .mean_reversion import (
    MeanReversionConfig,
    MeanReversionSignal,
    MeanReversionSignalGenerator,
    MeanRevSignalType,
    Position,
)
from .vol_surface_arbitrage import (
    SignalType,
    VolArbitrageConfig,
    VolArbitrageSignal,
    VolSurfaceArbitrageSignal,
)

__all__ = [
    # Volatility arbitrage
    "VolSurfaceArbitrageSignal",
    "VolArbitrageSignal",
    "VolArbitrageConfig",
    "SignalType",
    # Mean reversion
    "MeanReversionSignalGenerator",
    "MeanReversionSignal",
    "MeanReversionConfig",
    "MeanRevSignalType",
    "Position",
    # Aggregator
    "SignalAggregator",
    "AggregatedSignal",
    "AggregatedSignalType",
    "AggregatorConfig",
]
