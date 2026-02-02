"""
Model Calibration Engine.

Provides calibration tools for quantitative trading models:
- HestonCalibrator: Two-stage optimization for Heston stochastic volatility
- SABRCalibrator: Per-maturity smile fitting using SABR model
- OUFitter: Ornstein-Uhlenbeck parameter estimation with optimal boundaries
- CalibrationOrchestrator: Coordinates daily calibration runs

All calibrators use C++ backends for fast pricing/computation.

Example:
    >>> from quant_trading.calibration import HestonCalibrator
    >>> calibrator = HestonCalibrator()
    >>> result = calibrator.calibrate(market_options, S0=100, r=0.05, q=0.02)
    >>> print(f"Calibrated kappa: {result['params']['kappa']}")

References:
    - Heston (1993): "A closed-form solution for options with stochastic volatility"
    - Hagan et al. (2002): "Managing smile risk"
    - Leung & Li (2015): "Optimal Mean Reversion Trading"
"""

from .heston_calibrator import HestonCalibrator, CalibrationError
from .sabr_calibrator import SABRCalibrator
from .ou_fitter import OUFitter
from .orchestrator import CalibrationOrchestrator

__all__ = [
    "HestonCalibrator",
    "SABRCalibrator",
    "OUFitter",
    "CalibrationOrchestrator",
    "CalibrationError",
]
