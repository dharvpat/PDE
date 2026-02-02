"""
Mathematical models module.

Contains implementations of:
- Heston stochastic volatility model
- SABR volatility model
- Ornstein-Uhlenbeck process
- PDE solvers for option pricing

Example:
    >>> from quant_trading.models import HestonModel, SABRModel, OUProcess
    >>> # Price an option with Heston model
    >>> heston = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
    >>> price = heston.price_option(strike=100, maturity=1.0, spot=100,
    ...                             rate=0.05, dividend=0.02)
"""

from .heston import HestonModel, HestonParameters, OptionGreeks, PricingResult
from .sabr import SABRModel, SABRParameters
from .ou_process import OUProcess, OUParameters, OUFitResult

__all__ = [
    # Heston
    "HestonModel",
    "HestonParameters",
    "OptionGreeks",
    "PricingResult",
    # SABR
    "SABRModel",
    "SABRParameters",
    # OU Process
    "OUProcess",
    "OUParameters",
    "OUFitResult",
]
