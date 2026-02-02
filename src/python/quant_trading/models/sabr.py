"""
SABR stochastic volatility model (Python wrapper over C++).

This module provides a Pythonic interface to the high-performance
C++ SABR implementation using the Hagan et al. (2002) asymptotic formula.

Reference:
    Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002).
    "Managing smile risk." Wilmott Magazine, September, 84-108.

Example:
    >>> from quant_trading.models import SABRModel
    >>> model = SABRModel(beta=0.5)
    >>> vol = model.implied_volatility(
    ...     strike=105.0, forward=100.0, maturity=1.0,
    ...     alpha=0.2, rho=-0.3, nu=0.4
    ... )
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Try to import C++ bindings
try:
    from ..cpp import quant_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


@dataclass
class SABRParameters:
    """
    SABR model parameters.

    The SABR model describes forward rate dynamics:
        dF_t = σ_t F_t^β dW_t^F
        dσ_t = ν σ_t dW_t^σ
        dW_t^F · dW_t^σ = ρ dt

    Attributes:
        alpha: Initial volatility level
        beta: CEV exponent (0 = normal, 0.5 = equity, 1 = lognormal)
        rho: Correlation between forward and volatility (-1 < rho < 1)
        nu: Volatility of volatility
    """
    alpha: float
    beta: float
    rho: float
    nu: float

    def is_valid(self) -> bool:
        """Check if all parameters are in valid ranges."""
        return (
            self.alpha > 0 and
            0 <= self.beta <= 1 and
            abs(self.rho) < 1 and
            self.nu >= 0
        )

    def validate(self) -> None:
        """Validate parameters and raise ValueError if invalid."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.beta < 0 or self.beta > 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if abs(self.rho) >= 1:
            raise ValueError(f"|rho| must be < 1, got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
        }


class SABRModel:
    """
    SABR volatility model.

    Implements the Hagan et al. (2002) asymptotic formula for fast
    implied volatility computation. This provides a closed-form approximation
    that is accurate for most practical purposes.

    Key features:
        - Fast evaluation: ~100 nanoseconds per implied vol calculation
        - Accurate for strikes not too far from ATM
        - Handles special cases (ATM, beta=0, beta=1)

    Attributes:
        beta: CEV exponent (backbone parameter)

    Example:
        >>> model = SABRModel(beta=0.5)  # Equity
        >>> vol = model.implied_volatility(
        ...     strike=105.0, forward=100.0, maturity=1.0,
        ...     alpha=0.2, rho=-0.3, nu=0.4
        ... )

    Reference:
        Hagan, P.S., et al. (2002). "Managing smile risk."
        Wilmott Magazine, September, 84-108.
    """

    def __init__(self, beta: float = 0.5):
        """
        Construct SABR model with fixed beta.

        Args:
            beta: CEV exponent (default 0.5 for equity)
                  0 = normal model
                  0.5 = typical equity
                  1 = lognormal model

        Raises:
            ValueError: If beta not in [0, 1]
            RuntimeError: If C++ extensions not available
        """
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "C++ extensions not available. "
                "Please build the package with: pip install -e ."
            )

        if beta < 0 or beta > 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self._beta = beta
        self._cpp_model = quant_cpp.sabr.SABRModel(beta)

    @property
    def beta(self) -> float:
        """CEV exponent (backbone parameter)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set beta parameter."""
        if value < 0 or value > 1:
            raise ValueError(f"beta must be in [0, 1], got {value}")
        self._beta = value
        self._cpp_model.beta = value

    def implied_volatility(
        self,
        strike: float,
        forward: float,
        maturity: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Compute SABR implied volatility using Hagan asymptotic formula.

        For non-ATM strikes (K != F):
            σ_impl = α * (z/χ(z)) * correction_factor

        Reference: Hagan et al. (2002), Equation (2.17a)

        Args:
            strike: Strike price K
            forward: Forward price F
            maturity: Time to maturity T (years)
            alpha: Initial volatility α
            rho: Correlation ρ
            nu: Vol of vol ν

        Returns:
            Black-Scholes implied volatility
        """
        return self._cpp_model.implied_volatility(
            strike, forward, maturity, alpha, rho, nu
        )

    def implied_volatility_from_params(
        self,
        strike: float,
        forward: float,
        maturity: float,
        params: SABRParameters,
    ) -> float:
        """
        Compute implied volatility with SABRParameters struct.

        Args:
            strike: Strike price K
            forward: Forward price F
            maturity: Time to maturity T (years)
            params: SABRParameters object

        Returns:
            Black-Scholes implied volatility
        """
        return self.implied_volatility(
            strike, forward, maturity, params.alpha, params.rho, params.nu
        )

    def atm_volatility(
        self,
        forward: float,
        maturity: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Compute ATM implied volatility (simpler formula).

        At the money (K = F), the formula simplifies to:
            σ_ATM = α / F^(1-β) * [1 + (correction_terms) * T]

        Args:
            forward: Forward price F
            maturity: Time to maturity T
            alpha: Initial volatility α
            rho: Correlation ρ
            nu: Vol of vol ν

        Returns:
            ATM implied volatility
        """
        return self._cpp_model.atm_volatility(forward, maturity, alpha, rho, nu)

    def implied_volatilities(
        self,
        strikes: Union[List[float], np.ndarray],
        forward: float,
        maturity: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> np.ndarray:
        """
        Compute implied volatilities for multiple strikes (vectorized).

        Args:
            strikes: Array of strike prices
            forward: Forward price
            maturity: Time to maturity
            alpha: Initial volatility
            rho: Correlation
            nu: Vol of vol

        Returns:
            NumPy array of implied volatilities
        """
        strikes_list = list(strikes)
        vols = self._cpp_model.implied_volatilities(
            strikes_list, forward, maturity, alpha, rho, nu
        )
        return np.array(vols)

    def volatility_sensitivities(
        self,
        strike: float,
        forward: float,
        maturity: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> Tuple[float, float, float]:
        """
        Compute volatility smile sensitivities.

        Returns partial derivatives of implied volatility with respect to
        SABR parameters, useful for calibration and risk management.

        Args:
            strike: Strike price
            forward: Forward price
            maturity: Time to maturity
            alpha: Initial volatility
            rho: Correlation
            nu: Vol of vol

        Returns:
            Tuple of (d_sigma/d_alpha, d_sigma/d_rho, d_sigma/d_nu)
        """
        return self._cpp_model.volatility_sensitivities(
            strike, forward, maturity, alpha, rho, nu
        )

    def volatility_smile(
        self,
        strikes: Union[List[float], np.ndarray],
        forward: float,
        maturity: float,
        alpha: float,
        rho: float,
        nu: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute volatility smile (moneyness vs implied vol).

        Args:
            strikes: Array of strike prices
            forward: Forward price
            maturity: Time to maturity
            alpha: Initial volatility
            rho: Correlation
            nu: Vol of vol

        Returns:
            Tuple of (moneyness array, implied vol array)
            where moneyness = ln(K/F)
        """
        strikes = np.asarray(strikes)
        vols = self.implied_volatilities(strikes, forward, maturity, alpha, rho, nu)
        moneyness = np.log(strikes / forward)
        return moneyness, vols

    def __repr__(self) -> str:
        return f"SABRModel(β={self._beta:.2f})"
