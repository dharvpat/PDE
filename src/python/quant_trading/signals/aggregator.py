"""
Signal Aggregator.

Combines signals from multiple strategies into unified portfolio decisions.

Handles:
    - Signal conflicts (e.g., vol arb says buy, mean-rev says sell)
    - Correlation between strategies
    - Overall portfolio risk budgeting
    - Confidence-weighted voting
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from .mean_reversion import MeanReversionSignal, MeanRevSignalType
from .vol_surface_arbitrage import SignalType, VolArbitrageSignal

logger = logging.getLogger(__name__)


class AggregatedSignalType(Enum):
    """Aggregated signal types."""

    BUY = "buy"
    SELL = "sell"
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT = "exit"
    NO_ACTION = "no_action"


@dataclass
class AggregatedSignal:
    """Final aggregated trading signal."""

    asset: str  # Underlying or spread name
    signal_type: AggregatedSignalType
    confidence: float
    supporting_strategies: List[str]
    conflicting_strategies: List[str]
    rationale: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Component signals for audit
    component_signals: List[Dict] = field(default_factory=list)

    # Risk metrics
    suggested_position_size: Optional[float] = None
    max_position_pct: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "asset": self.asset,
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "supporting_strategies": self.supporting_strategies,
            "conflicting_strategies": self.conflicting_strategies,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
            "component_signals": self.component_signals,
            "suggested_position_size": self.suggested_position_size,
            "max_position_pct": self.max_position_pct,
        }


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregator."""

    # Consensus requirements
    consensus_ratio: float = 1.5  # Buy weight must exceed sell weight by 1.5x
    min_confidence_threshold: float = 0.6

    # Strategy weights (for weighted voting)
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "vol_arbitrage": 1.0,
        "mean_reversion": 1.0,
    })

    # Position sizing
    max_single_position_pct: float = 0.10  # 10% max per position
    max_strategy_allocation_pct: float = 0.30  # 30% max per strategy

    # Conflict resolution
    require_unanimous: bool = False  # If True, conflicting signals = no action


class SignalAggregator:
    """
    Combines signals from multiple strategies into unified portfolio decisions.

    Handles:
        - Signal grouping by asset
        - Conflict detection and resolution
        - Confidence-weighted voting
        - Position sizing recommendations

    Example:
        >>> aggregator = SignalAggregator()
        >>> vol_signals = [VolArbitrageSignal(...), ...]
        >>> mr_signals = [MeanReversionSignal(...), ...]
        >>> final_trades = aggregator.aggregate(
        ...     vol_arbitrage_signals=vol_signals,
        ...     mean_reversion_signals=mr_signals,
        ...     portfolio_value=1_000_000
        ... )
    """

    def __init__(
        self,
        config: Optional[AggregatorConfig] = None,
    ):
        """
        Initialize signal aggregator.

        Args:
            config: Configuration parameters
        """
        self.config = config or AggregatorConfig()

        logger.info(
            f"Initialized SignalAggregator with consensus_ratio="
            f"{self.config.consensus_ratio}"
        )

    def aggregate(
        self,
        vol_arbitrage_signals: Optional[List[VolArbitrageSignal]] = None,
        mean_reversion_signals: Optional[List[MeanReversionSignal]] = None,
        portfolio_value: float = 1_000_000,
        current_positions: Optional[Dict[str, Dict]] = None,
    ) -> List[AggregatedSignal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            vol_arbitrage_signals: Signals from volatility arbitrage strategy
            mean_reversion_signals: Signals from mean reversion strategy
            portfolio_value: Total portfolio value for sizing
            current_positions: Dict of current positions {asset: position_info}

        Returns:
            List of aggregated trading signals
        """
        current_positions = current_positions or {}

        # Normalize all signals to common format
        all_signals = self._normalize_signals(
            vol_arbitrage_signals=vol_arbitrage_signals or [],
            mean_reversion_signals=mean_reversion_signals or [],
        )

        if not all_signals:
            logger.debug("No signals to aggregate")
            return []

        # Group by asset
        grouped = self._group_by_asset(all_signals)

        # Aggregate each asset's signals
        final_signals = []
        for asset, signals in grouped.items():
            aggregated = self._aggregate_asset_signals(
                asset=asset,
                signals=signals,
                portfolio_value=portfolio_value,
                has_position=asset in current_positions,
            )
            if aggregated is not None:
                final_signals.append(aggregated)

        logger.info(f"Aggregated {len(all_signals)} signals into {len(final_signals)} trades")
        return final_signals

    def _normalize_signals(
        self,
        vol_arbitrage_signals: List[VolArbitrageSignal],
        mean_reversion_signals: List[MeanReversionSignal],
    ) -> List[Dict]:
        """Normalize all signals to common format."""
        normalized = []

        # Normalize vol arbitrage signals
        for sig in vol_arbitrage_signals:
            direction = self._get_direction(sig.signal_type)
            normalized.append({
                "asset": sig.underlying,
                "strategy": "vol_arbitrage",
                "direction": direction,
                "confidence": sig.confidence,
                "original_signal": sig.to_dict(),
                "rationale": sig.rationale,
            })

        # Normalize mean reversion signals
        for sig in mean_reversion_signals:
            direction = self._get_mr_direction(sig.signal_type)
            if direction is None:
                continue  # Skip HOLD signals

            normalized.append({
                "asset": sig.spread_name,
                "strategy": "mean_reversion",
                "direction": direction,
                "confidence": sig.confidence,
                "original_signal": sig.to_dict(),
                "rationale": sig.rationale,
            })

        return normalized

    def _get_direction(self, signal_type: SignalType) -> str:
        """Convert vol arb signal type to direction."""
        if signal_type == SignalType.BUY:
            return "buy"
        elif signal_type == SignalType.SELL:
            return "sell"
        return "hold"

    def _get_mr_direction(self, signal_type: MeanRevSignalType) -> Optional[str]:
        """Convert mean reversion signal type to direction."""
        if signal_type == MeanRevSignalType.ENTRY_LONG:
            return "buy"
        elif signal_type == MeanRevSignalType.ENTRY_SHORT:
            return "sell"
        elif signal_type in (
            MeanRevSignalType.EXIT_TAKE_PROFIT,
            MeanRevSignalType.EXIT_STOP_LOSS,
        ):
            return "exit"
        return None

    def _group_by_asset(self, signals: List[Dict]) -> Dict[str, List[Dict]]:
        """Group signals by asset."""
        grouped = defaultdict(list)
        for sig in signals:
            grouped[sig["asset"]].append(sig)
        return dict(grouped)

    def _aggregate_asset_signals(
        self,
        asset: str,
        signals: List[Dict],
        portfolio_value: float,
        has_position: bool,
    ) -> Optional[AggregatedSignal]:
        """Aggregate signals for a single asset."""
        if len(signals) == 1:
            # Single signal - just validate confidence
            sig = signals[0]
            if sig["confidence"] < self.config.min_confidence_threshold:
                return None

            signal_type = self._map_to_aggregated_type(sig["direction"])
            return AggregatedSignal(
                asset=asset,
                signal_type=signal_type,
                confidence=sig["confidence"],
                supporting_strategies=[sig["strategy"]],
                conflicting_strategies=[],
                rationale=sig["rationale"],
                component_signals=[sig["original_signal"]],
                suggested_position_size=self._compute_position_size(
                    confidence=sig["confidence"],
                    portfolio_value=portfolio_value,
                ),
            )

        # Multiple signals - use weighted voting
        return self._resolve_multiple_signals(
            asset=asset,
            signals=signals,
            portfolio_value=portfolio_value,
            has_position=has_position,
        )

    def _resolve_multiple_signals(
        self,
        asset: str,
        signals: List[Dict],
        portfolio_value: float,
        has_position: bool,
    ) -> Optional[AggregatedSignal]:
        """Resolve multiple potentially conflicting signals."""
        # Compute weighted scores
        buy_weight = 0.0
        sell_weight = 0.0
        exit_weight = 0.0

        buy_strategies = []
        sell_strategies = []
        exit_strategies = []

        for sig in signals:
            strategy_weight = self.config.strategy_weights.get(sig["strategy"], 1.0)
            weighted_confidence = sig["confidence"] * strategy_weight

            if sig["direction"] == "buy":
                buy_weight += weighted_confidence
                buy_strategies.append(sig["strategy"])
            elif sig["direction"] == "sell":
                sell_weight += weighted_confidence
                sell_strategies.append(sig["strategy"])
            elif sig["direction"] == "exit":
                exit_weight += weighted_confidence
                exit_strategies.append(sig["strategy"])

        # Check for exit signals first (if in position)
        if has_position and exit_weight > 0:
            return AggregatedSignal(
                asset=asset,
                signal_type=AggregatedSignalType.EXIT,
                confidence=exit_weight / len(exit_strategies) if exit_strategies else 0,
                supporting_strategies=exit_strategies,
                conflicting_strategies=[],
                rationale="Exit signal from position management",
                component_signals=[s["original_signal"] for s in signals],
            )

        # Determine consensus
        if self.config.require_unanimous:
            # All signals must agree
            if buy_strategies and not sell_strategies:
                direction = "buy"
                confidence = buy_weight / len(buy_strategies)
                supporting = buy_strategies
                conflicting = []
            elif sell_strategies and not buy_strategies:
                direction = "sell"
                confidence = sell_weight / len(sell_strategies)
                supporting = sell_strategies
                conflicting = []
            else:
                # Conflicting signals - no action
                logger.debug(f"{asset}: Conflicting signals, no action (unanimous required)")
                return None
        else:
            # Weighted voting with consensus ratio
            if buy_weight > sell_weight * self.config.consensus_ratio:
                direction = "buy"
                confidence = buy_weight / (buy_weight + sell_weight)
                supporting = buy_strategies
                conflicting = sell_strategies
            elif sell_weight > buy_weight * self.config.consensus_ratio:
                direction = "sell"
                confidence = sell_weight / (buy_weight + sell_weight)
                supporting = sell_strategies
                conflicting = buy_strategies
            else:
                # No clear consensus
                logger.debug(
                    f"{asset}: No consensus (buy={buy_weight:.2f}, sell={sell_weight:.2f})"
                )
                return None

        # Validate confidence
        if confidence < self.config.min_confidence_threshold:
            return None

        signal_type = self._map_to_aggregated_type(direction)

        return AggregatedSignal(
            asset=asset,
            signal_type=signal_type,
            confidence=confidence,
            supporting_strategies=supporting,
            conflicting_strategies=conflicting,
            rationale=self._build_rationale(direction, supporting, conflicting),
            component_signals=[s["original_signal"] for s in signals],
            suggested_position_size=self._compute_position_size(
                confidence=confidence,
                portfolio_value=portfolio_value,
            ),
        )

    def _map_to_aggregated_type(self, direction: str) -> AggregatedSignalType:
        """Map direction string to AggregatedSignalType."""
        mapping = {
            "buy": AggregatedSignalType.BUY,
            "sell": AggregatedSignalType.SELL,
            "exit": AggregatedSignalType.EXIT,
        }
        return mapping.get(direction, AggregatedSignalType.NO_ACTION)

    def _build_rationale(
        self,
        direction: str,
        supporting: List[str],
        conflicting: List[str],
    ) -> str:
        """Build human-readable rationale."""
        support_str = ", ".join(supporting)
        rationale = f"{direction.upper()} signal supported by: {support_str}"

        if conflicting:
            conflict_str = ", ".join(conflicting)
            rationale += f" (conflicting: {conflict_str})"

        return rationale

    def _compute_position_size(
        self,
        confidence: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute suggested position size based on confidence.

        Uses confidence-scaled position sizing:
            size = base_allocation * confidence
        """
        base_allocation = portfolio_value * self.config.max_single_position_pct
        return base_allocation * confidence

    def filter_by_risk_budget(
        self,
        signals: List[AggregatedSignal],
        current_exposure: Dict[str, float],
        max_total_exposure: float = 0.8,
    ) -> List[AggregatedSignal]:
        """
        Filter signals based on risk budget.

        Args:
            signals: Aggregated signals to filter
            current_exposure: Dict of {strategy: exposure_pct}
            max_total_exposure: Maximum total portfolio exposure

        Returns:
            Filtered signals that fit within risk budget
        """
        current_total = sum(current_exposure.values())
        available_budget = max_total_exposure - current_total

        if available_budget <= 0:
            logger.warning("Risk budget exhausted, filtering all signals")
            return []

        # Sort by confidence
        sorted_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)

        # Select signals within budget
        selected = []
        remaining_budget = available_budget

        for signal in sorted_signals:
            signal_exposure = (
                signal.suggested_position_size
                if signal.suggested_position_size
                else self.config.max_single_position_pct
            )

            if signal_exposure <= remaining_budget:
                selected.append(signal)
                remaining_budget -= signal_exposure

        return selected
