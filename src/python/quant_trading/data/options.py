"""
Options Chain Data Processing Module.

This module provides comprehensive options data processing including:
- Options chain data ingestion and normalization
- Implied volatility calculation using Newton-Raphson iteration
- Volatility surface interpolation (cubic spline, SVI)
- Greeks calculation from market data

References:
    - Brenner & Subrahmanyam (1988) for IV approximation initial guess
    - Gatheral (2004) for SVI parameterization
    - Black-Scholes (1973) for options pricing
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from scipy.optimize import brentq, minimize

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class ExerciseStyle(Enum):
    """Option exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    mid_price: Optional[float] = None
    exercise_style: ExerciseStyle = ExerciseStyle.AMERICAN
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Calculate mid price if not set."""
        if self.mid_price is None and self.bid > 0 and self.ask > 0:
            self.mid_price = (self.bid + self.ask) / 2


@dataclass
class OptionsChain:
    """Represents an options chain for a single underlying and expiration."""
    underlying: str
    expiration: date
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    calls: List[OptionContract]
    puts: List[OptionContract]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert options chain to DataFrame."""
        records = []
        for contract in self.calls + self.puts:
            records.append({
                'symbol': contract.symbol,
                'underlying': contract.underlying,
                'option_type': contract.option_type.value,
                'strike': contract.strike,
                'expiration': contract.expiration,
                'bid': contract.bid,
                'ask': contract.ask,
                'last': contract.last,
                'mid_price': contract.mid_price,
                'volume': contract.volume,
                'open_interest': contract.open_interest,
                'implied_volatility': contract.implied_volatility,
                'delta': contract.delta,
                'gamma': contract.gamma,
                'theta': contract.theta,
                'vega': contract.vega,
                'rho': contract.rho
            })
        return pd.DataFrame(records)

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike closest to spot price."""
        all_strikes = [c.strike for c in self.calls]
        if not all_strikes:
            return self.spot_price
        return min(all_strikes, key=lambda x: abs(x - self.spot_price))


class BlackScholes:
    """
    Black-Scholes option pricing and Greeks calculation.

    References:
        Black, F., & Scholes, M. (1973). The pricing of options and
        corporate liabilities. Journal of Political Economy, 81(3), 637-654.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Returns:
            Call option price
        """
        if T <= 0:
            return max(S * np.exp(-q * T) - K, 0)

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)

        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """
        Calculate Black-Scholes put option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Volatility

        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S * np.exp(-q * T), 0)

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    @staticmethod
    def delta(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: OptionType) -> float:
        """Calculate option delta."""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)

        if option_type == OptionType.CALL:
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate option gamma (same for calls and puts)."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
        """Calculate option vega (same for calls and puts), per 1% vol move."""
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def theta(S: float, K: float, T: float, r: float, q: float, sigma: float,
              option_type: OptionType) -> float:
        """Calculate option theta (per day)."""
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, q, sigma)
        d2 = BlackScholes.d2(S, K, T, r, q, sigma)

        term1 = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        return (term1 + term2 + term3) / 365  # Per day

    @staticmethod
    def rho(S: float, K: float, T: float, r: float, q: float, sigma: float,
            option_type: OptionType) -> float:
        """Calculate option rho (per 1% rate move)."""
        if T <= 0:
            return 0.0

        d2 = BlackScholes.d2(S, K, T, r, q, sigma)

        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


class ImpliedVolatilityCalculator:
    """
    Implied volatility calculator using Newton-Raphson iteration.

    Uses Brenner & Subrahmanyam (1988) approximation for initial guess:
        sigma_approx = sqrt(2*pi/T) * (C/S) for ATM options

    Falls back to Brent's method if Newton-Raphson fails to converge.

    References:
        Brenner, M., & Subrahmanyam, M. G. (1988). A simple formula to
        compute the implied standard deviation. Financial Analysts Journal.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        min_vol: float = 0.001,
        max_vol: float = 5.0
    ):
        """
        Initialize IV calculator.

        Args:
            max_iterations: Maximum Newton-Raphson iterations
            tolerance: Convergence tolerance
            min_vol: Minimum volatility bound
            max_vol: Maximum volatility bound
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_vol = min_vol
        self.max_vol = max_vol

    def _initial_guess(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate initial volatility guess using Brenner-Subrahmanyam approximation.

        For ATM options: sigma ≈ sqrt(2*pi/T) * (Price/S)
        For OTM/ITM: adjust based on moneyness
        """
        if T <= 0:
            return 0.2  # Default guess for expired options

        # Forward price
        F = S * np.exp((r - q) * T)

        # Moneyness
        moneyness = np.log(F / K)

        # Brenner-Subrahmanyam for ATM
        if abs(moneyness) < 0.1:
            sigma_guess = np.sqrt(2 * np.pi / T) * (market_price / S)
        else:
            # Corrado-Miller approximation for OTM/ITM
            intrinsic = max(0, (F - K) * np.exp(-r * T)) if option_type == OptionType.CALL else \
                       max(0, (K - F) * np.exp(-r * T))
            time_value = market_price - intrinsic
            if time_value > 0:
                sigma_guess = np.sqrt(2 * np.pi / T) * (time_value / S)
            else:
                sigma_guess = 0.2

        # Bound the guess
        return np.clip(sigma_guess, self.min_vol, self.max_vol)

    def calculate(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        option_type: OptionType
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson with Brent fallback.

        Args:
            market_price: Market price of the option
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            option_type: Call or put

        Returns:
            Implied volatility or None if calculation fails
        """
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return None

        # Check arbitrage bounds
        if option_type == OptionType.CALL:
            lower_bound = max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
            upper_bound = S * np.exp(-q * T)
        else:
            lower_bound = max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
            upper_bound = K * np.exp(-r * T)

        if market_price < lower_bound or market_price > upper_bound:
            logger.debug(f"Price {market_price} outside arbitrage bounds [{lower_bound}, {upper_bound}]")
            return None

        # Try Newton-Raphson first
        sigma = self._initial_guess(market_price, S, K, T, r, q, option_type)

        for i in range(self.max_iterations):
            if option_type == OptionType.CALL:
                price = BlackScholes.call_price(S, K, T, r, q, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, q, sigma)

            diff = price - market_price

            if abs(diff) < self.tolerance:
                return sigma

            # Vega for Newton-Raphson step (not normalized)
            d1 = BlackScholes.d1(S, K, T, r, q, sigma)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            if vega < 1e-10:
                break  # Vega too small, switch to Brent

            sigma_new = sigma - diff / vega

            # Bound the new sigma
            sigma = np.clip(sigma_new, self.min_vol, self.max_vol)

        # Fall back to Brent's method
        try:
            def objective(vol):
                if option_type == OptionType.CALL:
                    return BlackScholes.call_price(S, K, T, r, q, vol) - market_price
                else:
                    return BlackScholes.put_price(S, K, T, r, q, vol) - market_price

            sigma = brentq(objective, self.min_vol, self.max_vol, xtol=self.tolerance)
            return sigma
        except ValueError:
            logger.warning(f"IV calculation failed for K={K}, T={T}, price={market_price}")
            return None

    def calculate_for_chain(
        self,
        chain: OptionsChain
    ) -> OptionsChain:
        """
        Calculate implied volatility for all options in a chain.

        Args:
            chain: Options chain

        Returns:
            Options chain with IV populated
        """
        S = chain.spot_price
        r = chain.risk_free_rate
        q = chain.dividend_yield

        # Calculate time to expiration
        today = date.today()
        T = (chain.expiration - today).days / 365.0

        if T <= 0:
            logger.warning(f"Expired options chain: {chain.underlying} {chain.expiration}")
            return chain

        for contract in chain.calls:
            if contract.mid_price and contract.mid_price > 0:
                contract.implied_volatility = self.calculate(
                    contract.mid_price, S, contract.strike, T, r, q, OptionType.CALL
                )

        for contract in chain.puts:
            if contract.mid_price and contract.mid_price > 0:
                contract.implied_volatility = self.calculate(
                    contract.mid_price, S, contract.strike, T, r, q, OptionType.PUT
                )

        return chain


class GreeksCalculator:
    """Calculate Greeks for options using Black-Scholes model."""

    def calculate_all_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            q: Dividend yield
            sigma: Implied volatility
            option_type: Call or put

        Returns:
            Dictionary of Greeks
        """
        return {
            'delta': BlackScholes.delta(S, K, T, r, q, sigma, option_type),
            'gamma': BlackScholes.gamma(S, K, T, r, q, sigma),
            'theta': BlackScholes.theta(S, K, T, r, q, sigma, option_type),
            'vega': BlackScholes.vega(S, K, T, r, q, sigma),
            'rho': BlackScholes.rho(S, K, T, r, q, sigma, option_type)
        }

    def calculate_for_chain(self, chain: OptionsChain) -> OptionsChain:
        """
        Calculate Greeks for all options in a chain.

        Args:
            chain: Options chain with IV populated

        Returns:
            Options chain with Greeks populated
        """
        S = chain.spot_price
        r = chain.risk_free_rate
        q = chain.dividend_yield

        today = date.today()
        T = (chain.expiration - today).days / 365.0

        if T <= 0:
            return chain

        for contract in chain.calls:
            if contract.implied_volatility:
                greeks = self.calculate_all_greeks(
                    S, contract.strike, T, r, q,
                    contract.implied_volatility, OptionType.CALL
                )
                contract.delta = greeks['delta']
                contract.gamma = greeks['gamma']
                contract.theta = greeks['theta']
                contract.vega = greeks['vega']
                contract.rho = greeks['rho']

        for contract in chain.puts:
            if contract.implied_volatility:
                greeks = self.calculate_all_greeks(
                    S, contract.strike, T, r, q,
                    contract.implied_volatility, OptionType.PUT
                )
                contract.delta = greeks['delta']
                contract.gamma = greeks['gamma']
                contract.theta = greeks['theta']
                contract.vega = greeks['vega']
                contract.rho = greeks['rho']

        return chain


@dataclass
class VolatilitySurfacePoint:
    """A single point on the volatility surface."""
    strike: float
    expiration: date
    implied_vol: float
    moneyness: float  # log(K/F)
    time_to_expiry: float  # in years


class VolatilitySurface:
    """
    Volatility surface representation and interpolation.

    Supports multiple interpolation methods:
    - Cubic spline (per expiration)
    - SVI (Stochastic Volatility Inspired) parameterization
    - Bilinear interpolation

    References:
        Gatheral, J. (2004). A parsimonious arbitrage-free implied
        volatility parameterization with application to the valuation
        of volatility derivatives. Global Derivatives & Risk Management.
    """

    def __init__(
        self,
        points: List[VolatilitySurfacePoint],
        spot_price: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0
    ):
        """
        Initialize volatility surface.

        Args:
            points: List of surface points
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
        """
        self.points = points
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        # Build interpolation structures
        self._build_interpolators()

    def _build_interpolators(self) -> None:
        """Build interpolation functions for each expiration."""
        self._smile_interpolators: Dict[date, Callable] = {}

        # Group points by expiration
        by_expiry: Dict[date, List[VolatilitySurfacePoint]] = {}
        for point in self.points:
            if point.expiration not in by_expiry:
                by_expiry[point.expiration] = []
            by_expiry[point.expiration].append(point)

        # Build cubic spline for each expiration
        for expiry, exp_points in by_expiry.items():
            if len(exp_points) >= 4:  # Need at least 4 points for cubic spline
                sorted_points = sorted(exp_points, key=lambda p: p.strike)
                strikes = [p.strike for p in sorted_points]
                vols = [p.implied_vol for p in sorted_points]

                try:
                    self._smile_interpolators[expiry] = interpolate.CubicSpline(
                        strikes, vols, bc_type='natural'
                    )
                except Exception as e:
                    logger.warning(f"Failed to build spline for {expiry}: {e}")

        # Store sorted expirations
        self._expirations = sorted(by_expiry.keys())

        # Store time to expiry mapping
        today = date.today()
        self._expiry_times = {
            exp: (exp - today).days / 365.0 for exp in self._expirations
        }

    def get_vol(self, strike: float, expiration: date) -> Optional[float]:
        """
        Get interpolated volatility for a given strike and expiration.

        Args:
            strike: Strike price
            expiration: Expiration date

        Returns:
            Interpolated implied volatility
        """
        if expiration in self._smile_interpolators:
            try:
                return float(self._smile_interpolators[expiration](strike))
            except Exception:
                pass

        # Fall back to nearest expiration
        if self._expirations:
            nearest = min(self._expirations, key=lambda e: abs((e - expiration).days))
            if nearest in self._smile_interpolators:
                try:
                    return float(self._smile_interpolators[nearest](strike))
                except Exception:
                    pass

        # Last resort: return average vol
        if self.points:
            return np.mean([p.implied_vol for p in self.points])

        return None

    def get_atm_vol(self, expiration: date) -> Optional[float]:
        """Get ATM volatility for a given expiration."""
        return self.get_vol(self.spot_price, expiration)

    def get_skew(self, expiration: date, delta_range: float = 0.25) -> Optional[float]:
        """
        Calculate volatility skew (25-delta put vol - 25-delta call vol).

        Args:
            expiration: Expiration date
            delta_range: Delta level for skew calculation

        Returns:
            Skew in volatility points
        """
        if expiration not in self._smile_interpolators:
            return None

        # Approximate 25-delta strikes
        atm_vol = self.get_atm_vol(expiration)
        if atm_vol is None:
            return None

        T = self._expiry_times.get(expiration, 0.25)
        if T <= 0:
            return None

        # 25-delta put is typically ~0.9*S, 25-delta call is ~1.1*S
        put_strike = self.spot_price * np.exp(-0.5 * atm_vol * np.sqrt(T))
        call_strike = self.spot_price * np.exp(0.5 * atm_vol * np.sqrt(T))

        put_vol = self.get_vol(put_strike, expiration)
        call_vol = self.get_vol(call_strike, expiration)

        if put_vol is not None and call_vol is not None:
            return put_vol - call_vol

        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert surface to DataFrame."""
        records = []
        for point in self.points:
            records.append({
                'strike': point.strike,
                'expiration': point.expiration,
                'implied_vol': point.implied_vol,
                'moneyness': point.moneyness,
                'time_to_expiry': point.time_to_expiry
            })
        return pd.DataFrame(records)


class SVIParameterization:
    """
    SVI (Stochastic Volatility Inspired) parameterization.

    The SVI model parameterizes total implied variance as:
        w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

    where:
        k = log(K/F) is log-moneyness
        a, b, rho, m, sigma are parameters

    References:
        Gatheral, J. (2004). A parsimonious arbitrage-free implied
        volatility parameterization.
    """

    def __init__(self):
        """Initialize SVI parameterization."""
        self.params: Optional[Dict[str, float]] = None

    def fit(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
        time_to_expiry: float
    ) -> Dict[str, float]:
        """
        Fit SVI parameters to market data.

        Args:
            log_moneyness: Array of log(K/F) values
            total_variance: Array of implied_vol^2 * T values
            time_to_expiry: Time to expiration in years

        Returns:
            Dictionary of fitted SVI parameters
        """
        def svi(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

        def objective(params):
            a, b, rho, m, sigma = params
            if sigma <= 0 or b < 0 or abs(rho) >= 1:
                return 1e10
            predicted = svi(log_moneyness, a, b, rho, m, sigma)
            return np.sum((predicted - total_variance)**2)

        # Initial guess
        x0 = [
            np.mean(total_variance),  # a
            0.1,  # b
            -0.5,  # rho
            0.0,  # m
            0.1   # sigma
        ]

        # Bounds
        bounds = [
            (0, None),      # a >= 0
            (0, None),      # b >= 0
            (-0.999, 0.999),  # -1 < rho < 1
            (-2, 2),        # m
            (0.001, 2)      # sigma > 0
        ]

        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        self.params = {
            'a': result.x[0],
            'b': result.x[1],
            'rho': result.x[2],
            'm': result.x[3],
            'sigma': result.x[4],
            'time_to_expiry': time_to_expiry
        }

        return self.params

    def get_total_variance(self, log_moneyness: float) -> float:
        """Get total implied variance for a given log-moneyness."""
        if self.params is None:
            raise ValueError("SVI not fitted. Call fit() first.")

        a = self.params['a']
        b = self.params['b']
        rho = self.params['rho']
        m = self.params['m']
        sigma = self.params['sigma']

        k = log_moneyness
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def get_implied_vol(self, log_moneyness: float) -> float:
        """Get implied volatility for a given log-moneyness."""
        if self.params is None:
            raise ValueError("SVI not fitted. Call fit() first.")

        T = self.params['time_to_expiry']
        total_var = self.get_total_variance(log_moneyness)

        if total_var <= 0 or T <= 0:
            return 0.0

        return np.sqrt(total_var / T)


class OptionsChainProcessor:
    """
    Main class for processing options chain data.

    Orchestrates IV calculation, Greeks computation, and surface construction.
    """

    def __init__(
        self,
        iv_calculator: Optional[ImpliedVolatilityCalculator] = None,
        greeks_calculator: Optional[GreeksCalculator] = None
    ):
        """
        Initialize processor.

        Args:
            iv_calculator: Custom IV calculator
            greeks_calculator: Custom Greeks calculator
        """
        self.iv_calculator = iv_calculator or ImpliedVolatilityCalculator()
        self.greeks_calculator = greeks_calculator or GreeksCalculator()

    def process_chain(self, chain: OptionsChain) -> OptionsChain:
        """
        Process a full options chain.

        Calculates:
        1. Implied volatility for all options
        2. Greeks for all options with valid IV

        Args:
            chain: Raw options chain

        Returns:
            Processed chain with IV and Greeks
        """
        # Calculate IV
        chain = self.iv_calculator.calculate_for_chain(chain)

        # Calculate Greeks
        chain = self.greeks_calculator.calculate_for_chain(chain)

        logger.info(
            f"Processed chain: {chain.underlying} {chain.expiration}, "
            f"{len(chain.calls)} calls, {len(chain.puts)} puts"
        )

        return chain

    def build_volatility_surface(
        self,
        chains: List[OptionsChain]
    ) -> VolatilitySurface:
        """
        Build volatility surface from multiple chains.

        Args:
            chains: List of processed options chains (with IV)

        Returns:
            Volatility surface object
        """
        if not chains:
            raise ValueError("No chains provided")

        points = []
        spot_price = chains[0].spot_price
        risk_free_rate = chains[0].risk_free_rate
        dividend_yield = chains[0].dividend_yield

        today = date.today()

        for chain in chains:
            T = (chain.expiration - today).days / 365.0
            if T <= 0:
                continue

            # Forward price
            F = spot_price * np.exp((risk_free_rate - dividend_yield) * T)

            # Add call IVs
            for contract in chain.calls:
                if contract.implied_volatility and contract.implied_volatility > 0:
                    moneyness = np.log(contract.strike / F)
                    points.append(VolatilitySurfacePoint(
                        strike=contract.strike,
                        expiration=chain.expiration,
                        implied_vol=contract.implied_volatility,
                        moneyness=moneyness,
                        time_to_expiry=T
                    ))

            # Add put IVs (typically more liquid for OTM puts)
            for contract in chain.puts:
                if contract.implied_volatility and contract.implied_volatility > 0:
                    moneyness = np.log(contract.strike / F)
                    # Only add if we don't have a call at this strike
                    existing_strikes = {p.strike for p in points
                                       if p.expiration == chain.expiration}
                    if contract.strike not in existing_strikes:
                        points.append(VolatilitySurfacePoint(
                            strike=contract.strike,
                            expiration=chain.expiration,
                            implied_vol=contract.implied_volatility,
                            moneyness=moneyness,
                            time_to_expiry=T
                        ))

        logger.info(f"Built volatility surface with {len(points)} points")

        return VolatilitySurface(
            points=points,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield
        )

    def calculate_term_structure(
        self,
        chains: List[OptionsChain]
    ) -> pd.DataFrame:
        """
        Calculate ATM volatility term structure.

        Args:
            chains: List of processed options chains

        Returns:
            DataFrame with expiration and ATM vol
        """
        records = []
        today = date.today()

        for chain in chains:
            T = (chain.expiration - today).days / 365.0
            if T <= 0:
                continue

            atm_strike = chain.get_atm_strike()

            # Find ATM call and put
            atm_call_iv = None
            atm_put_iv = None

            for contract in chain.calls:
                if contract.strike == atm_strike and contract.implied_volatility:
                    atm_call_iv = contract.implied_volatility
                    break

            for contract in chain.puts:
                if contract.strike == atm_strike and contract.implied_volatility:
                    atm_put_iv = contract.implied_volatility
                    break

            # Average of call and put IV at ATM
            if atm_call_iv and atm_put_iv:
                atm_vol = (atm_call_iv + atm_put_iv) / 2
            elif atm_call_iv:
                atm_vol = atm_call_iv
            elif atm_put_iv:
                atm_vol = atm_put_iv
            else:
                continue

            records.append({
                'expiration': chain.expiration,
                'days_to_expiry': (chain.expiration - today).days,
                'time_to_expiry': T,
                'atm_strike': atm_strike,
                'atm_vol': atm_vol
            })

        return pd.DataFrame(records).sort_values('time_to_expiry')


def parse_options_data(
    df: pd.DataFrame,
    underlying: str,
    spot_price: float,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0
) -> List[OptionsChain]:
    """
    Parse raw options data into OptionsChain objects.

    Expected DataFrame columns:
    - symbol: Option symbol
    - option_type: 'call' or 'put'
    - strike: Strike price
    - expiration: Expiration date
    - bid: Bid price
    - ask: Ask price
    - last: Last traded price
    - volume: Trading volume
    - open_interest: Open interest

    Args:
        df: Raw options data
        underlying: Underlying symbol
        spot_price: Current spot price
        risk_free_rate: Risk-free rate
        dividend_yield: Dividend yield

    Returns:
        List of OptionsChain objects grouped by expiration
    """
    chains = []

    # Group by expiration
    for expiration, group in df.groupby('expiration'):
        if isinstance(expiration, str):
            expiration = datetime.strptime(expiration, '%Y-%m-%d').date()
        elif isinstance(expiration, datetime):
            expiration = expiration.date()

        calls = []
        puts = []

        for _, row in group.iterrows():
            contract = OptionContract(
                symbol=row.get('symbol', f"{underlying}_{expiration}_{row['strike']}"),
                underlying=underlying,
                option_type=OptionType(row['option_type'].lower()),
                strike=float(row['strike']),
                expiration=expiration,
                bid=float(row.get('bid', 0)),
                ask=float(row.get('ask', 0)),
                last=float(row.get('last', 0)),
                volume=int(row.get('volume', 0)),
                open_interest=int(row.get('open_interest', 0)),
                timestamp=datetime.now()
            )

            if contract.option_type == OptionType.CALL:
                calls.append(contract)
            else:
                puts.append(contract)

        chain = OptionsChain(
            underlying=underlying,
            expiration=expiration,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            calls=sorted(calls, key=lambda c: c.strike),
            puts=sorted(puts, key=lambda c: c.strike)
        )

        chains.append(chain)

    return sorted(chains, key=lambda c: c.expiration)
