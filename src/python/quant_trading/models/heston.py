"""
Heston stochastic volatility model (Python wrapper over C++).

This module provides a Pythonic interface to the high-performance
C++ Heston implementation.

Reference:
    Heston, S.L. (1993). "A closed-form solution for options with
    stochastic volatility." Review of Financial Studies, 6(2), 327-343.

Example:
    >>> from quant_trading.models import HestonModel
    >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
    >>> price = model.price_option(strike=100, maturity=1.0, spot=100,
    ...                            rate=0.05, dividend=0.02)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

# Try to import C++ bindings
try:
    from ..cpp import quant_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


@dataclass
class HestonParameters:
    """
    Heston model parameters.

    The Heston model describes stochastic volatility dynamics:
        dS_t = μS_t dt + √v_t S_t dW_t^S
        dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
        dW_t^S · dW_t^v = ρ dt

    Attributes:
        kappa: Mean-reversion speed of variance (must be > 0)
        theta: Long-term variance mean (must be > 0)
        sigma: Volatility of variance, "vol of vol" (must be > 0)
        rho: Correlation between asset and variance (-1 < rho < 1)
        v0: Initial variance (must be > 0)
    """
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def is_feller_satisfied(self) -> bool:
        """
        Check if Feller condition (2κθ >= σ²) is satisfied.

        The Feller condition ensures variance cannot reach zero,
        which is important for numerical stability.
        """
        return 2.0 * self.kappa * self.theta >= self.sigma ** 2

    def is_valid(self) -> bool:
        """Check if all parameters are in valid ranges."""
        return (
            self.kappa > 0 and
            self.theta > 0 and
            self.sigma > 0 and
            abs(self.rho) < 1 and
            self.v0 > 0
        )

    def validate(self) -> None:
        """Validate parameters and raise ValueError if invalid."""
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if abs(self.rho) >= 1:
            raise ValueError(f"|rho| must be < 1, got {self.rho}")
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "v0": self.v0,
        }


@dataclass
class OptionGreeks:
    """Option Greeks (sensitivities)."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass
class PricingResult:
    """Option pricing result with price and optional Greeks."""
    price: float
    greeks: Optional[OptionGreeks] = None


class HestonModel:
    """
    Heston stochastic volatility model for option pricing.

    This class provides a Pythonic interface to the high-performance
    C++ Heston implementation. It implements characteristic function-based
    option pricing using the Carr-Madan (1999) FFT approach.

    Attributes:
        params: HestonParameters object with model parameters

    Example:
        >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3,
        ...                     rho=-0.7, v0=0.04)
        >>> price = model.price_option(strike=100, maturity=1.0,
        ...                            spot=100, rate=0.05, dividend=0.02)

    Reference:
        Heston, S.L. (1993). "A closed-form solution for options
        with stochastic volatility." Review of Financial Studies, 6(2).
    """

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
    ):
        """
        Initialize Heston model.

        Args:
            kappa: Mean-reversion speed of variance (must be > 0)
            theta: Long-term variance mean (must be > 0)
            sigma: Volatility of variance (must be > 0)
            rho: Correlation (-1 < rho < 1)
            v0: Initial variance (must be > 0)

        Raises:
            ValueError: If parameters invalid
            RuntimeError: If C++ extensions not available
        """
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ extensions not available. "
                "Please build the package with: pip install -e ."
            )

        self.params = HestonParameters(kappa, theta, sigma, rho, v0)
        self.params.validate()

        if not self.params.is_feller_satisfied():
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*kappa*theta:.4f}, "
                f"σ² = {sigma**2:.4f}. Variance may hit zero.",
                UserWarning,
            )

        # Create C++ model
        cpp_params = quant_cpp.heston.HestonParameters(kappa, theta, sigma, rho, v0)
        self._cpp_model = quant_cpp.heston.HestonModel(cpp_params)

    def price_option(
        self,
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        dividend: float = 0.0,
        is_call: bool = True,
    ) -> float:
        """
        Price a European option.

        Args:
            strike: Strike price K
            maturity: Time to maturity T (years)
            spot: Current spot price S0
            rate: Risk-free rate r
            dividend: Dividend yield q (default 0)
            is_call: True for call option, False for put

        Returns:
            Option price

        Raises:
            ValueError: If strike, spot, or maturity are non-positive
        """
        return self._cpp_model.price_option(
            strike, maturity, spot, rate, dividend, is_call
        )

    def price_option_with_greeks(
        self,
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        dividend: float = 0.0,
        is_call: bool = True,
    ) -> PricingResult:
        """
        Price option with Greeks computation.

        Computes price and Greeks using finite difference approximations.

        Args:
            strike: Strike price K
            maturity: Time to maturity T (years)
            spot: Current spot price S0
            rate: Risk-free rate r
            dividend: Dividend yield q
            is_call: True for call option, False for put

        Returns:
            PricingResult with price and Greeks
        """
        result = self._cpp_model.price_option_with_greeks(
            strike, maturity, spot, rate, dividend, is_call
        )
        greeks = OptionGreeks(
            delta=result.greeks.delta,
            gamma=result.greeks.gamma,
            vega=result.greeks.vega,
            theta=result.greeks.theta,
            rho=result.greeks.rho,
        )
        return PricingResult(price=result.price, greeks=greeks)

    def price_options(
        self,
        strikes: Union[List[float], np.ndarray],
        maturities: Union[List[float], np.ndarray, float],
        spot: float,
        rate: float,
        dividend: float = 0.0,
        is_call: bool = True,
    ) -> np.ndarray:
        """
        Price multiple options (vectorized).

        More efficient than repeated calls to price_option when
        pricing many options with the same underlying parameters.

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities (or single value for all)
            spot: Current spot price
            rate: Risk-free rate
            dividend: Dividend yield
            is_call: True for calls, False for puts

        Returns:
            NumPy array of option prices
        """
        strikes_list = list(strikes)
        if isinstance(maturities, (int, float)):
            maturities_list = [float(maturities)]
        else:
            maturities_list = list(maturities)

        prices = self._cpp_model.price_options(
            strikes_list, maturities_list, spot, rate, dividend, is_call
        )
        return np.array(prices)

    def implied_volatility(
        self,
        strike: float,
        maturity: float,
        spot: float,
        rate: float,
        dividend: float = 0.0,
        is_call: bool = True,
    ) -> float:
        """
        Compute Black-Scholes implied volatility from Heston price.

        Uses Newton-Raphson iteration to find the Black-Scholes implied vol
        that matches the Heston price.

        Args:
            strike: Strike price
            maturity: Time to maturity
            spot: Current spot price
            rate: Risk-free rate
            dividend: Dividend yield
            is_call: True for call, False for put

        Returns:
            Implied volatility
        """
        return self._cpp_model.implied_volatility(
            strike, maturity, spot, rate, dividend, is_call
        )

    def implied_volatility_surface(
        self,
        strikes: Union[List[float], np.ndarray],
        maturities: Union[List[float], np.ndarray],
        spot: float,
        rate: float,
        dividend: float = 0.0,
    ) -> np.ndarray:
        """
        Compute implied volatility surface.

        Args:
            strikes: Array of strike prices
            maturities: Array of maturities
            spot: Current spot price
            rate: Risk-free rate
            dividend: Dividend yield

        Returns:
            2D NumPy array of implied volatilities (strikes x maturities)
        """
        strikes = np.asarray(strikes)
        maturities = np.asarray(maturities)

        surface = np.zeros((len(strikes), len(maturities)))
        for i, K in enumerate(strikes):
            for j, T in enumerate(maturities):
                surface[i, j] = self.implied_volatility(
                    K, T, spot, rate, dividend, is_call=True
                )
        return surface

    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> "HestonModel":
        """Create model from parameter dictionary."""
        return cls(
            kappa=params["kappa"],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"],
            v0=params["v0"],
        )

    @classmethod
    def from_params(cls, params: HestonParameters) -> "HestonModel":
        """Create model from HestonParameters object."""
        return cls(
            kappa=params.kappa,
            theta=params.theta,
            sigma=params.sigma,
            rho=params.rho,
            v0=params.v0,
        )

    def __repr__(self) -> str:
        return (
            f"HestonModel(κ={self.params.kappa:.3f}, θ={self.params.theta:.4f}, "
            f"σ={self.params.sigma:.3f}, ρ={self.params.rho:.3f}, v0={self.params.v0:.4f})"
        )
