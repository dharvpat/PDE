"""
Mean Reversion Signal Generator.

Generates trading signals based on Ornstein-Uhlenbeck optimal stopping
framework from Leung & Li (2015).

Signal Logic:
    - Enter long when spread < entry_lower boundary
    - Enter short when spread > entry_upper boundary
    - Exit at optimal take-profit or stop-loss levels

Reference:
    Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with
    Transaction Costs and Stop-Loss Exit." International Journal of
    Theoretical and Applied Finance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..calibration.ou_fitter import OUFitResult, OUParameters, OptimalBoundaries

logger = logging.getLogger(__name__)


class MeanRevSignalType(Enum):
    """Mean reversion signal types."""

    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_TAKE_PROFIT = "exit_take_profit"
    EXIT_STOP_LOSS = "exit_stop_loss"
    HOLD = "hold"


@dataclass
class Position:
    """Represents a current position in a spread."""

    spread_name: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "spread_name": self.spread_name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


@dataclass
class MeanReversionSignal:
    """Signal from mean reversion strategy."""

    spread_name: str
    signal_type: MeanRevSignalType
    confidence: float
    current_value: float
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # OU parameters for context
    theta: Optional[float] = None  # Long-term mean
    half_life_days: Optional[float] = None

    # Boundary information
    entry_lower: Optional[float] = None
    entry_upper: Optional[float] = None
    exit_target: Optional[float] = None

    # Position-specific (for exit signals)
    entry_price: Optional[float] = None
    pnl: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "spread_name": self.spread_name,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "current_value": self.current_value,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
            "theta": self.theta,
            "half_life_days": self.half_life_days,
            "entry_lower": self.entry_lower,
            "entry_upper": self.entry_upper,
            "exit_target": self.exit_target,
            "entry_price": self.entry_price,
            "pnl": self.pnl,
        }


@dataclass
class MeanReversionConfig:
    """Configuration for mean reversion signal generator."""

    # OU parameter constraints
    min_half_life_days: float = 5.0  # Too fast = likely noise
    max_half_life_days: float = 120.0  # Too slow = capital inefficient
    min_mean_reversion_speed: float = 0.5  # mu > 0.5 annualized

    # Signal confidence
    min_confidence: float = 0.6

    # Entry thresholds
    min_sigma_distance: float = 1.0  # Minimum distance from mean in std devs

    # Position management
    default_stop_loss_sigma: float = 2.0  # Stop-loss in terms of stationary std
    default_take_profit_sigma: float = 0.0  # Take-profit at mean by default


class MeanReversionSignalGenerator:
    """
    Generates signals based on Ornstein-Uhlenbeck optimal stopping framework.

    Uses calibrated OU parameters and optimal boundaries to determine:
    - When to enter positions (spread crosses entry boundaries)
    - When to exit positions (reach mean or stop-loss)

    Example:
        >>> generator = MeanReversionSignalGenerator()
        >>> signal = generator.generate_signal(
        ...     spread_name="SPY-IWM",
        ...     current_value=0.05,
        ...     ou_fit_result=fit_result,
        ...     current_position=None
        ... )
        >>> if signal.signal_type == MeanRevSignalType.ENTRY_LONG:
        ...     print("Enter long position")

    Reference:
        Leung & Li (2015) "Optimal Mean Reversion Trading"
    """

    def __init__(
        self,
        config: Optional[MeanReversionConfig] = None,
        position_manager: Optional[Dict[str, Position]] = None,
    ):
        """
        Initialize signal generator.

        Args:
            config: Configuration parameters
            position_manager: Dict mapping spread_name to current Position
        """
        self.config = config or MeanReversionConfig()
        self._positions: Dict[str, Position] = position_manager or {}

        logger.info(
            f"Initialized MeanReversionSignalGenerator with "
            f"half_life_range=[{self.config.min_half_life_days}, "
            f"{self.config.max_half_life_days}] days"
        )

    def generate_signal(
        self,
        spread_name: str,
        current_value: float,
        ou_fit_result: "OUFitResult",
        current_position: Optional[Position] = None,
    ) -> Optional[MeanReversionSignal]:
        """
        Determine if current spread level warrants entry/exit.

        Args:
            spread_name: Identifier for the spread (e.g., 'SPY-IWM')
            current_value: Current spread price/value
            ou_fit_result: Result from OUFitter.fit()
            current_position: Current position if any (overrides internal state)

        Returns:
            MeanReversionSignal if action warranted, else None
        """
        params = ou_fit_result.params
        boundaries = ou_fit_result.boundaries

        # Validate OU parameters
        if not self._validate_ou_params(params, spread_name):
            return None

        # Use provided position or check internal state
        position = current_position or self._positions.get(spread_name)

        if position is None:
            # Not in position → check entry conditions
            return self._check_entry(
                spread_name=spread_name,
                current_value=current_value,
                params=params,
                boundaries=boundaries,
            )
        else:
            # In position → check exit conditions
            return self._check_exit(
                spread_name=spread_name,
                current_value=current_value,
                params=params,
                boundaries=boundaries,
                position=position,
            )

    def generate_signals_batch(
        self,
        spreads: Dict[str, float],
        ou_results: Dict[str, "OUFitResult"],
    ) -> List[MeanReversionSignal]:
        """
        Generate signals for multiple spreads.

        Args:
            spreads: Dict of {spread_name: current_value}
            ou_results: Dict of {spread_name: OUFitResult}

        Returns:
            List of signals for all spreads
        """
        signals = []

        for spread_name, current_value in spreads.items():
            if spread_name not in ou_results:
                logger.warning(f"No OU result for {spread_name}, skipping")
                continue

            signal = self.generate_signal(
                spread_name=spread_name,
                current_value=current_value,
                ou_fit_result=ou_results[spread_name],
            )

            if signal is not None:
                signals.append(signal)

        return signals

    def _validate_ou_params(self, params: "OUParameters", spread_name: str) -> bool:
        """Validate OU parameters are suitable for trading."""
        half_life = params.half_life * 252  # Convert to days

        if half_life < self.config.min_half_life_days:
            logger.debug(
                f"{spread_name}: Half-life {half_life:.1f} days too short "
                f"(min: {self.config.min_half_life_days})"
            )
            return False

        if half_life > self.config.max_half_life_days:
            logger.debug(
                f"{spread_name}: Half-life {half_life:.1f} days too long "
                f"(max: {self.config.max_half_life_days})"
            )
            return False

        if params.mu < self.config.min_mean_reversion_speed:
            logger.debug(
                f"{spread_name}: Mean reversion speed {params.mu:.2f} too slow "
                f"(min: {self.config.min_mean_reversion_speed})"
            )
            return False

        return True

    def _check_entry(
        self,
        spread_name: str,
        current_value: float,
        params: "OUParameters",
        boundaries: "OptimalBoundaries",
    ) -> Optional[MeanReversionSignal]:
        """Check entry conditions when not in position."""
        theta = params.theta
        sigma_stat = params.stationary_std
        entry_lower = boundaries.entry_lower
        entry_upper = boundaries.entry_upper

        # Distance from mean in standard deviations
        distance_sigma = abs(current_value - theta) / sigma_stat

        # Check lower entry (enter long)
        if current_value < entry_lower:
            confidence = self._compute_entry_confidence(
                current_value=current_value,
                boundary=entry_lower,
                theta=theta,
                sigma_stat=sigma_stat,
            )

            if confidence < self.config.min_confidence:
                return None

            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.ENTRY_LONG,
                confidence=confidence,
                current_value=current_value,
                rationale=(
                    f"Spread {current_value:.4f} < entry lower {entry_lower:.4f}, "
                    f"expect reversion to θ={theta:.4f} "
                    f"({distance_sigma:.1f}σ from mean)"
                ),
                theta=theta,
                half_life_days=params.half_life * 252,
                entry_lower=entry_lower,
                entry_upper=entry_upper,
                exit_target=theta,
            )

        # Check upper entry (enter short)
        if current_value > entry_upper:
            confidence = self._compute_entry_confidence(
                current_value=current_value,
                boundary=entry_upper,
                theta=theta,
                sigma_stat=sigma_stat,
            )

            if confidence < self.config.min_confidence:
                return None

            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.ENTRY_SHORT,
                confidence=confidence,
                current_value=current_value,
                rationale=(
                    f"Spread {current_value:.4f} > entry upper {entry_upper:.4f}, "
                    f"expect reversion to θ={theta:.4f} "
                    f"({distance_sigma:.1f}σ from mean)"
                ),
                theta=theta,
                half_life_days=params.half_life * 252,
                entry_lower=entry_lower,
                entry_upper=entry_upper,
                exit_target=theta,
            )

        return None  # No entry signal

    def _check_exit(
        self,
        spread_name: str,
        current_value: float,
        params: "OUParameters",
        boundaries: "OptimalBoundaries",
        position: Position,
    ) -> Optional[MeanReversionSignal]:
        """Check exit conditions when in position."""
        direction = position.direction
        entry_price = position.entry_price
        stop_loss = position.stop_loss
        take_profit = position.take_profit
        theta = params.theta

        # Check stop-loss first (always exit)
        if direction == "long" and current_value <= stop_loss:
            pnl = current_value - entry_price
            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.EXIT_STOP_LOSS,
                confidence=1.0,
                current_value=current_value,
                rationale=f"Stop-loss triggered at {current_value:.4f}",
                theta=theta,
                entry_price=entry_price,
                pnl=pnl,
            )

        if direction == "short" and current_value >= stop_loss:
            pnl = entry_price - current_value
            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.EXIT_STOP_LOSS,
                confidence=1.0,
                current_value=current_value,
                rationale=f"Stop-loss triggered at {current_value:.4f}",
                theta=theta,
                entry_price=entry_price,
                pnl=pnl,
            )

        # Check take-profit (mean reversion achieved)
        if direction == "long" and current_value >= take_profit:
            pnl = current_value - entry_price
            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.EXIT_TAKE_PROFIT,
                confidence=0.9,
                current_value=current_value,
                rationale=(
                    f"Take-profit at {current_value:.4f}, "
                    f"gained {pnl:.4f} from entry {entry_price:.4f}"
                ),
                theta=theta,
                entry_price=entry_price,
                pnl=pnl,
            )

        if direction == "short" and current_value <= take_profit:
            pnl = entry_price - current_value
            return MeanReversionSignal(
                spread_name=spread_name,
                signal_type=MeanRevSignalType.EXIT_TAKE_PROFIT,
                confidence=0.9,
                current_value=current_value,
                rationale=(
                    f"Take-profit at {current_value:.4f}, "
                    f"gained {pnl:.4f} from entry {entry_price:.4f}"
                ),
                theta=theta,
                entry_price=entry_price,
                pnl=pnl,
            )

        return None  # Continue holding

    def _compute_entry_confidence(
        self,
        current_value: float,
        boundary: float,
        theta: float,
        sigma_stat: float,
    ) -> float:
        """
        Compute entry confidence based on distance from boundary.

        Confidence increases with distance past boundary (more extreme = higher confidence).
        """
        # Distance past boundary
        distance_past = abs(current_value - boundary)

        # Distance from mean in sigma
        distance_from_mean_sigma = abs(current_value - theta) / sigma_stat

        # Base confidence: increases with distance
        # Max confidence when 2σ past boundary
        base_confidence = min(0.95, 0.6 + (distance_past / (2 * sigma_stat)) * 0.35)

        # Bonus for being far from mean (stronger signal)
        if distance_from_mean_sigma > 2.0:
            base_confidence = min(0.98, base_confidence + 0.05)

        return base_confidence

    def register_position(self, position: Position) -> None:
        """Register a new position for tracking."""
        self._positions[position.spread_name] = position
        logger.info(f"Registered {position.direction} position in {position.spread_name}")

    def close_position(self, spread_name: str) -> Optional[Position]:
        """Remove a position from tracking."""
        return self._positions.pop(spread_name, None)

    def get_position(self, spread_name: str) -> Optional[Position]:
        """Get current position for a spread."""
        return self._positions.get(spread_name)

    def create_position_from_signal(
        self,
        signal: MeanReversionSignal,
        quantity: float,
        ou_params: "OUParameters",
    ) -> Position:
        """
        Create a Position object from an entry signal.

        Args:
            signal: Entry signal (must be ENTRY_LONG or ENTRY_SHORT)
            quantity: Position quantity
            ou_params: OU parameters for stop-loss computation

        Returns:
            Position object ready for registration
        """
        if signal.signal_type == MeanRevSignalType.ENTRY_LONG:
            direction = "long"
            stop_loss = (
                ou_params.theta
                - self.config.default_stop_loss_sigma * ou_params.stationary_std
            )
            take_profit = (
                ou_params.theta
                + self.config.default_take_profit_sigma * ou_params.stationary_std
            )
        elif signal.signal_type == MeanRevSignalType.ENTRY_SHORT:
            direction = "short"
            stop_loss = (
                ou_params.theta
                + self.config.default_stop_loss_sigma * ou_params.stationary_std
            )
            take_profit = (
                ou_params.theta
                - self.config.default_take_profit_sigma * ou_params.stationary_std
            )
        else:
            raise ValueError(f"Cannot create position from signal type: {signal.signal_type}")

        return Position(
            spread_name=signal.spread_name,
            direction=direction,
            entry_price=signal.current_value,
            entry_time=signal.timestamp,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
