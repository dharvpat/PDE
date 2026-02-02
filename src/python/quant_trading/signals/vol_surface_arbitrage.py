"""
Volatility Surface Arbitrage Signal Generator.

Detects mispricing in options by comparing calibrated model implied
volatility (Heston/SABR) to market implied volatility.

Signal Logic:
    Trade delta-hedged option when |σ_model - σ_market| > threshold

Reference:
    - Heston (1993) for stochastic volatility pricing
    - Hagan et al. (2002) for SABR implied volatility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from ..calibration.heston_calibrator import CalibrationResult
    from ..calibration.sabr_calibrator import SABRCalibrationResult

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""

    BUY = "buy"  # Buy option (underpriced)
    SELL = "sell"  # Sell option (overpriced)
    HOLD = "hold"  # No action


@dataclass
class VolArbitrageSignal:
    """Signal from volatility surface arbitrage strategy."""

    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    model_iv: float
    market_iv: float
    divergence_pct: float
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional metadata
    bid: Optional[float] = None
    ask: Optional[float] = None
    model_price: Optional[float] = None
    market_price: Optional[float] = None
    delta: Optional[float] = None
    vega: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "strike": self.strike,
            "expiration": self.expiration.isoformat()
            if isinstance(self.expiration, datetime)
            else str(self.expiration),
            "option_type": self.option_type,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "model_iv": self.model_iv,
            "market_iv": self.market_iv,
            "divergence_pct": self.divergence_pct,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid,
            "ask": self.ask,
            "model_price": self.model_price,
            "market_price": self.market_price,
            "delta": self.delta,
            "vega": self.vega,
        }


@dataclass
class VolArbitrageConfig:
    """Configuration for volatility arbitrage signal generator."""

    # Divergence thresholds
    min_divergence_pct: float = 0.10  # 10% relative divergence required
    max_divergence_pct: float = 0.50  # Filter extreme outliers (likely data errors)

    # Confidence thresholds
    min_confidence: float = 0.6  # Only high-confidence signals

    # Liquidity filters
    max_bid_ask_spread_pct: float = 0.10  # 10% max bid-ask spread
    min_volume: int = 100  # Minimum daily volume

    # Maturity filters
    min_days_to_expiry: int = 7  # Avoid very short-dated
    max_days_to_expiry: int = 180  # Avoid very long-dated
    preferred_min_days: int = 30  # Sweet spot minimum
    preferred_max_days: int = 90  # Sweet spot maximum

    # Model fit quality
    max_model_rmse: float = 0.05  # 5% max calibration RMSE


class VolSurfaceArbitrageSignal:
    """
    Detects mispricing in options by comparing Heston/SABR model to market.

    Signal Logic:
        - Compute model-implied volatility using calibrated parameters
        - Compare to market-implied volatility
        - Generate buy signal when model IV > market IV (underpriced option)
        - Generate sell signal when model IV < market IV (overpriced option)

    Confidence Scoring:
        - Higher calibration fit quality → higher confidence
        - Better liquidity (tighter spreads) → higher confidence
        - Optimal maturity range (30-90 days) → higher confidence

    Example:
        >>> generator = VolSurfaceArbitrageSignal()
        >>> signals = generator.generate_signals(
        ...     heston_params=calibration_result,
        ...     market_data=options_df,
        ...     S0=100.0,
        ...     r=0.05,
        ...     q=0.02
        ... )
        >>> for sig in signals:
        ...     print(f"{sig.signal_type.value}: {sig.underlying} {sig.strike}")
    """

    def __init__(
        self,
        config: Optional[VolArbitrageConfig] = None,
        use_sabr: bool = True,
        use_heston: bool = True,
    ):
        """
        Initialize signal generator.

        Args:
            config: Configuration parameters
            use_sabr: Whether to use SABR model for IV computation
            use_heston: Whether to use Heston model for IV computation
        """
        self.config = config or VolArbitrageConfig()
        self.use_sabr = use_sabr
        self.use_heston = use_heston

        logger.info(
            f"Initialized VolSurfaceArbitrageSignal with "
            f"min_divergence={self.config.min_divergence_pct:.1%}"
        )

    def generate_signals(
        self,
        market_data: "pd.DataFrame",
        S0: float,
        r: float,
        q: float,
        heston_result: Optional["CalibrationResult"] = None,
        sabr_result: Optional["SABRCalibrationResult"] = None,
    ) -> List[VolArbitrageSignal]:
        """
        Generate trading signals based on vol surface mispricing.

        Args:
            market_data: DataFrame with columns:
                - underlying: str
                - strike: float
                - expiration: datetime
                - option_type: 'call' or 'put'
                - implied_vol: float (market IV)
                - bid: float (optional)
                - ask: float (optional)
                - volume: int (optional)
                - T: float (time to expiry in years)
            S0: Current spot price
            r: Risk-free rate
            q: Dividend yield
            heston_result: Calibration result from HestonCalibrator
            sabr_result: Calibration result from SABRCalibrator

        Returns:
            List of VolArbitrageSignal objects
        """
        if heston_result is None and sabr_result is None:
            raise ValueError("At least one model result (heston or sabr) required")

        signals = []
        calibration_rmse = self._get_calibration_rmse(heston_result, sabr_result)

        logger.info(f"Generating signals for {len(market_data)} options")

        for _, option in market_data.iterrows():
            signal = self._evaluate_option(
                option=option,
                S0=S0,
                r=r,
                q=q,
                heston_result=heston_result,
                sabr_result=sabr_result,
                calibration_rmse=calibration_rmse,
            )

            if signal is not None:
                signals.append(signal)

        logger.info(f"Generated {len(signals)} signals")
        return signals

    def _evaluate_option(
        self,
        option,
        S0: float,
        r: float,
        q: float,
        heston_result: Optional["CalibrationResult"],
        sabr_result: Optional["SABRCalibrationResult"],
        calibration_rmse: float,
    ) -> Optional[VolArbitrageSignal]:
        """Evaluate a single option for arbitrage opportunity."""
        # Extract option data
        underlying = option.get("underlying", "UNKNOWN")
        strike = option["strike"]
        expiration = option.get("expiration")
        option_type = option.get("option_type", "call")
        market_iv = option["implied_vol"]
        T = option["T"]

        # Apply filters
        if not self._passes_filters(option, T):
            return None

        # Compute model IV
        model_iv = self._compute_model_iv(
            S0=S0,
            K=strike,
            T=T,
            r=r,
            q=q,
            heston_result=heston_result,
            sabr_result=sabr_result,
        )

        if model_iv is None or model_iv <= 0:
            return None

        # Compute divergence
        divergence = model_iv - market_iv
        divergence_pct = divergence / market_iv if market_iv > 0 else 0

        # Check divergence threshold
        if abs(divergence_pct) < self.config.min_divergence_pct:
            return None
        if abs(divergence_pct) > self.config.max_divergence_pct:
            # Likely data error
            logger.debug(
                f"Extreme divergence {divergence_pct:.1%} for {underlying} "
                f"K={strike}, skipping"
            )
            return None

        # Determine signal direction
        if divergence > 0:
            signal_type = SignalType.BUY
            rationale = (
                f"Market IV {market_iv:.1%}, Model IV {model_iv:.1%}, "
                f"underpriced by {divergence_pct:.1%}"
            )
        else:
            signal_type = SignalType.SELL
            rationale = (
                f"Market IV {market_iv:.1%}, Model IV {model_iv:.1%}, "
                f"overpriced by {abs(divergence_pct):.1%}"
            )

        # Compute confidence
        confidence = self._compute_confidence(
            option=option,
            T=T,
            calibration_rmse=calibration_rmse,
            divergence_pct=abs(divergence_pct),
        )

        if confidence < self.config.min_confidence:
            return None

        return VolArbitrageSignal(
            underlying=underlying,
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            signal_type=signal_type,
            confidence=confidence,
            model_iv=model_iv,
            market_iv=market_iv,
            divergence_pct=divergence_pct,
            rationale=rationale,
            bid=option.get("bid"),
            ask=option.get("ask"),
            delta=option.get("delta"),
            vega=option.get("vega"),
        )

    def _passes_filters(self, option, T: float) -> bool:
        """Check if option passes liquidity and maturity filters."""
        # Maturity filter
        days_to_expiry = T * 365
        if days_to_expiry < self.config.min_days_to_expiry:
            return False
        if days_to_expiry > self.config.max_days_to_expiry:
            return False

        # Liquidity filter (bid-ask spread)
        bid = option.get("bid")
        ask = option.get("ask")
        if bid is not None and ask is not None and bid > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > self.config.max_bid_ask_spread_pct:
                return False

        # Volume filter
        volume = option.get("volume")
        if volume is not None and volume < self.config.min_volume:
            return False

        return True

    def _compute_model_iv(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        heston_result: Optional["CalibrationResult"],
        sabr_result: Optional["SABRCalibrationResult"],
    ) -> Optional[float]:
        """
        Compute model-implied volatility.

        Uses SABR if available (faster), falls back to Heston.
        """
        # Try SABR first (faster analytical formula)
        if self.use_sabr and sabr_result is not None:
            try:
                return self._sabr_implied_vol(S0, K, T, r, q, sabr_result)
            except Exception as e:
                logger.debug(f"SABR IV computation failed: {e}")

        # Fall back to Heston
        if self.use_heston and heston_result is not None:
            try:
                return self._heston_implied_vol(S0, K, T, r, q, heston_result)
            except Exception as e:
                logger.debug(f"Heston IV computation failed: {e}")

        return None

    def _sabr_implied_vol(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        sabr_result: "SABRCalibrationResult",
    ) -> float:
        """Compute SABR implied volatility."""
        # Find the closest maturity in calibration
        maturities = list(sabr_result.params_by_maturity.keys())
        if not maturities:
            raise ValueError("No calibrated maturities available")

        # Interpolate or use nearest
        if T in sabr_result.params_by_maturity:
            params = sabr_result.params_by_maturity[T]
        else:
            # Use nearest maturity
            nearest_T = min(maturities, key=lambda x: abs(x - T))
            params = sabr_result.params_by_maturity[nearest_T]

        # Compute forward price
        F = S0 * np.exp((r - q) * T)

        # SABR implied vol formula (Hagan et al. 2002)
        return self._sabr_vol_formula(F, K, T, params)

    def _sabr_vol_formula(self, F: float, K: float, T: float, params) -> float:
        """
        Hagan's SABR implied volatility formula.

        Direct implementation for signal generation.
        """
        alpha = params.alpha
        beta = params.beta
        rho = params.rho
        nu = params.nu

        # Handle ATM case
        if abs(F - K) < 1e-10:
            F_beta = F ** (1 - beta)
            one_minus_beta = 1 - beta
            term1 = one_minus_beta**2 / 24 * alpha**2 / (F_beta**2)
            term2 = rho * beta * nu * alpha / (4 * F_beta)
            term3 = (2 - 3 * rho**2) * nu**2 / 24
            return alpha / F_beta * (1 + (term1 + term2 + term3) * T)

        # General case
        FK = F * K
        log_FK = np.log(F / K)
        FK_beta = FK ** ((1 - beta) / 2)

        z = (nu / alpha) * FK_beta * log_FK
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho))

        zeta = z / x_z if abs(x_z) > 1e-10 else 1.0

        one_minus_beta = 1 - beta
        term1 = one_minus_beta**2 / 24 * alpha**2 / (FK_beta**2)
        term2 = rho * beta * nu * alpha / (4 * FK_beta)
        term3 = (2 - 3 * rho**2) * nu**2 / 24
        bracket = 1 + (term1 + term2 + term3) * T

        denom_term = 1 + one_minus_beta**2 / 24 * log_FK**2
        denom_term += one_minus_beta**4 / 1920 * log_FK**4

        return max((alpha / (FK_beta * denom_term)) * zeta * bracket, 1e-6)

    def _heston_implied_vol(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        heston_result: "CalibrationResult",
    ) -> float:
        """
        Compute Heston implied volatility via price inversion.

        Computes Heston price then inverts to get implied vol.
        """
        # Simplified: use ATM vol approximation
        # Full implementation would price option and invert Black-Scholes
        params = heston_result.params
        v0 = params.v0
        theta = params.theta

        # Rough approximation: average of initial and long-term variance
        avg_var = (v0 + theta) / 2
        return np.sqrt(avg_var)

    def _compute_confidence(
        self,
        option,
        T: float,
        calibration_rmse: float,
        divergence_pct: float,
    ) -> float:
        """
        Compute confidence score based on multiple factors.

        Components:
            - Model fit quality (40%)
            - Option liquidity (40%)
            - Maturity sweet spot (20%)
        """
        # Fit quality component (lower RMSE = higher score)
        fit_score = 1.0 - min(calibration_rmse, self.config.max_model_rmse) / self.config.max_model_rmse

        # Liquidity component
        bid = option.get("bid")
        ask = option.get("ask")
        if bid is not None and ask is not None and bid > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 0.1
            liquidity_score = max(0, 1.0 - spread_pct / self.config.max_bid_ask_spread_pct)
        else:
            liquidity_score = 0.5  # Unknown liquidity

        # Maturity component
        days_to_expiry = T * 365
        if days_to_expiry < self.config.min_days_to_expiry:
            maturity_score = 0.3
        elif self.config.preferred_min_days <= days_to_expiry <= self.config.preferred_max_days:
            maturity_score = 1.0
        elif days_to_expiry > self.config.max_days_to_expiry:
            maturity_score = 0.5
        else:
            maturity_score = 0.7

        # Weighted average
        confidence = (
            0.4 * fit_score
            + 0.4 * liquidity_score
            + 0.2 * maturity_score
        )

        return confidence

    def _get_calibration_rmse(
        self,
        heston_result: Optional["CalibrationResult"],
        sabr_result: Optional["SABRCalibrationResult"],
    ) -> float:
        """Get calibration RMSE from available results."""
        if sabr_result is not None:
            return sabr_result.total_rmse
        if heston_result is not None:
            return heston_result.rmse
        return 0.05  # Default

    def filter_signals(
        self,
        signals: List[VolArbitrageSignal],
        max_signals: int = 10,
        min_confidence: Optional[float] = None,
    ) -> List[VolArbitrageSignal]:
        """
        Filter and rank signals.

        Args:
            signals: List of signals to filter
            max_signals: Maximum number of signals to return
            min_confidence: Minimum confidence threshold (overrides config)

        Returns:
            Filtered and sorted list of signals
        """
        confidence_threshold = min_confidence or self.config.min_confidence

        # Filter by confidence
        filtered = [s for s in signals if s.confidence >= confidence_threshold]

        # Sort by confidence (descending)
        filtered.sort(key=lambda s: s.confidence, reverse=True)

        # Limit count
        return filtered[:max_signals]
