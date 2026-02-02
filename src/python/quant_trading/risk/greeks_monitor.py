"""
Greeks Risk Monitor for Options Portfolios.

Monitors portfolio Greeks and determines when rebalancing/hedging is needed.

Key Greeks:
    - Delta: Sensitivity to underlying price (target ~0 for market-neutral)
    - Gamma: Convexity of delta (manage to avoid large rehedging moves)
    - Vega: Sensitivity to volatility (target positive for vol arb)
    - Theta: Time decay (offset with positive carry)

Features:
    - Real-time portfolio Greeks aggregation
    - Threshold-based alerting
    - Delta-hedging recommendations
    - Greeks decomposition by strategy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HedgeActionType(Enum):
    """Types of hedging actions."""

    HEDGE_DELTA = "hedge_delta"
    REDUCE_GAMMA = "reduce_gamma"
    REDUCE_VEGA = "reduce_vega"
    ALERT = "alert"
    NO_ACTION = "no_action"


@dataclass
class OptionPosition:
    """Represents a single option position."""

    symbol: str
    underlying: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: datetime
    quantity: int  # Positive for long, negative for short
    direction: str  # 'long' or 'short'

    # Greeks (per contract)
    delta: float
    gamma: float
    vega: float
    theta: float

    # Optional
    rho: float = 0.0
    implied_vol: float = 0.0
    underlying_price: float = 0.0
    market_value: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "underlying": self.underlying,
            "option_type": self.option_type,
            "strike": self.strike,
            "expiration": self.expiration.isoformat(),
            "quantity": self.quantity,
            "direction": self.direction,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
        }


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""

    delta: float  # Total delta exposure
    gamma: float  # Total gamma
    vega: float  # Total vega (per 1% vol move)
    theta: float  # Daily theta decay

    # Dollar-denominated
    delta_dollars: float = 0.0  # Delta * underlying_price
    gamma_dollars: float = 0.0  # Gamma * underlying_price^2 / 100
    vega_dollars: float = 0.0  # Vega in dollar terms

    # By underlying
    delta_by_underlying: Dict[str, float] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "delta_dollars": self.delta_dollars,
            "gamma_dollars": self.gamma_dollars,
            "vega_dollars": self.vega_dollars,
            "delta_by_underlying": self.delta_by_underlying,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HedgeAction:
    """Recommended hedging action."""

    action_type: HedgeActionType
    underlying: Optional[str] = None
    quantity: float = 0.0  # Shares/contracts to trade
    rationale: str = ""
    urgency: str = "normal"  # 'low', 'normal', 'high', 'critical'
    estimated_cost: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "underlying": self.underlying,
            "quantity": self.quantity,
            "rationale": self.rationale,
            "urgency": self.urgency,
            "estimated_cost": self.estimated_cost,
        }


@dataclass
class GreeksMonitorConfig:
    """Configuration for Greeks monitoring."""

    # Delta thresholds
    delta_threshold: float = 100.0  # Absolute delta shares
    delta_dollars_threshold: float = 50_000.0  # Dollar delta threshold

    # Gamma thresholds
    gamma_threshold: float = 50.0  # Gamma threshold
    gamma_dollars_threshold: float = 10_000.0  # Per 1% move

    # Vega thresholds
    vega_threshold: float = 1000.0  # Vega exposure
    max_vega_dollars: float = 25_000.0  # Max vega in dollars

    # Theta monitoring
    max_daily_theta_loss: float = 5_000.0  # Max daily theta decay

    # Hedging preferences
    hedge_delta_threshold_pct: float = 0.02  # Hedge when delta > 2% of portfolio
    min_hedge_size: float = 100.0  # Minimum shares to hedge


class GreeksRiskMonitor:
    """
    Monitor portfolio Greeks and recommend hedging actions.

    Features:
        - Real-time Greeks aggregation across positions
        - Threshold-based alerting and action recommendations
        - Delta-hedging calculations
        - Greeks decomposition by underlying/strategy

    Example:
        >>> monitor = GreeksRiskMonitor()
        >>> greeks = monitor.compute_portfolio_greeks(positions)
        >>> print(f"Portfolio delta: {greeks.delta:.0f}")
        >>> needs_hedge, actions = monitor.check_rehedge_needed(greeks)
        >>> if needs_hedge:
        ...     for action in actions:
        ...         print(f"{action.action_type}: {action.rationale}")

    Reference:
        Standard options risk management practices
    """

    def __init__(
        self,
        config: Optional[GreeksMonitorConfig] = None,
    ):
        """
        Initialize Greeks monitor.

        Args:
            config: Configuration parameters
        """
        self.config = config or GreeksMonitorConfig()

        logger.info(
            f"Initialized GreeksRiskMonitor with delta_threshold="
            f"{self.config.delta_threshold}"
        )

    def compute_portfolio_greeks(
        self,
        positions: List[OptionPosition],
        underlying_prices: Optional[Dict[str, float]] = None,
    ) -> PortfolioGreeks:
        """
        Aggregate Greeks across all option positions.

        Args:
            positions: List of option positions
            underlying_prices: Dict of {underlying: price} for dollar calculations

        Returns:
            PortfolioGreeks with aggregated values
        """
        underlying_prices = underlying_prices or {}

        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0

        delta_dollars = 0.0
        gamma_dollars = 0.0
        vega_dollars = 0.0

        delta_by_underlying: Dict[str, float] = {}

        for pos in positions:
            # Signed quantity (positive for long, negative for short)
            signed_qty = pos.quantity if pos.direction == "long" else -pos.quantity

            # Aggregate Greeks
            pos_delta = signed_qty * pos.delta * 100  # Convert to shares
            pos_gamma = signed_qty * pos.gamma * 100
            pos_vega = signed_qty * pos.vega * 100
            pos_theta = signed_qty * pos.theta * 100

            total_delta += pos_delta
            total_gamma += pos_gamma
            total_vega += pos_vega
            total_theta += pos_theta

            # Track by underlying
            if pos.underlying not in delta_by_underlying:
                delta_by_underlying[pos.underlying] = 0.0
            delta_by_underlying[pos.underlying] += pos_delta

            # Dollar calculations
            price = underlying_prices.get(pos.underlying, pos.underlying_price)
            if price > 0:
                delta_dollars += pos_delta * price
                gamma_dollars += pos_gamma * price * price / 100
                vega_dollars += pos_vega  # Vega already in dollar terms

        return PortfolioGreeks(
            delta=total_delta,
            gamma=total_gamma,
            vega=total_vega,
            theta=total_theta,
            delta_dollars=delta_dollars,
            gamma_dollars=gamma_dollars,
            vega_dollars=vega_dollars,
            delta_by_underlying=delta_by_underlying,
        )

    def check_rehedge_needed(
        self,
        portfolio_greeks: PortfolioGreeks,
        portfolio_value: float = 1_000_000,
    ) -> tuple[bool, List[HedgeAction]]:
        """
        Determine if delta-hedging or other rebalancing is required.

        Args:
            portfolio_greeks: Current portfolio Greeks
            portfolio_value: Total portfolio value for percentage calculations

        Returns:
            Tuple of (needs_action: bool, list of HedgeActions)
        """
        actions = []

        # Check delta
        delta_actions = self._check_delta(portfolio_greeks, portfolio_value)
        actions.extend(delta_actions)

        # Check gamma
        gamma_actions = self._check_gamma(portfolio_greeks)
        actions.extend(gamma_actions)

        # Check vega
        vega_actions = self._check_vega(portfolio_greeks)
        actions.extend(vega_actions)

        # Check theta
        theta_actions = self._check_theta(portfolio_greeks)
        actions.extend(theta_actions)

        needs_action = len(actions) > 0

        if needs_action:
            logger.info(f"Rehedge needed: {len(actions)} actions recommended")
        else:
            logger.debug("Greeks within acceptable ranges")

        return needs_action, actions

    def _check_delta(
        self,
        greeks: PortfolioGreeks,
        portfolio_value: float,
    ) -> List[HedgeAction]:
        """Check delta exposure and recommend hedging."""
        actions = []

        # Check absolute delta
        if abs(greeks.delta) > self.config.delta_threshold:
            hedge_qty = -greeks.delta  # Sell shares if long delta

            urgency = "normal"
            if abs(greeks.delta) > self.config.delta_threshold * 2:
                urgency = "high"
            if abs(greeks.delta) > self.config.delta_threshold * 3:
                urgency = "critical"

            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.HEDGE_DELTA,
                    quantity=hedge_qty,
                    rationale=(
                        f"Portfolio delta {greeks.delta:.0f} exceeds threshold "
                        f"{self.config.delta_threshold:.0f}"
                    ),
                    urgency=urgency,
                )
            )

        # Check dollar delta as percentage of portfolio
        delta_pct = abs(greeks.delta_dollars) / portfolio_value if portfolio_value > 0 else 0
        if delta_pct > self.config.hedge_delta_threshold_pct:
            if not actions:  # Don't duplicate
                actions.append(
                    HedgeAction(
                        action_type=HedgeActionType.HEDGE_DELTA,
                        quantity=-greeks.delta,
                        rationale=(
                            f"Dollar delta ${greeks.delta_dollars:,.0f} is "
                            f"{delta_pct:.1%} of portfolio"
                        ),
                        urgency="normal",
                    )
                )

        # Per-underlying hedging
        for underlying, delta in greeks.delta_by_underlying.items():
            if abs(delta) > self.config.delta_threshold:
                actions.append(
                    HedgeAction(
                        action_type=HedgeActionType.HEDGE_DELTA,
                        underlying=underlying,
                        quantity=-delta,
                        rationale=f"{underlying} delta {delta:.0f} exceeds threshold",
                        urgency="normal",
                    )
                )

        return actions

    def _check_gamma(self, greeks: PortfolioGreeks) -> List[HedgeAction]:
        """Check gamma exposure."""
        actions = []

        if abs(greeks.gamma) > self.config.gamma_threshold:
            urgency = "normal"
            if abs(greeks.gamma) > self.config.gamma_threshold * 2:
                urgency = "high"

            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.ALERT,
                    rationale=(
                        f"High gamma exposure: {greeks.gamma:.2f}, "
                        f"monitor for large underlying moves"
                    ),
                    urgency=urgency,
                )
            )

        if abs(greeks.gamma_dollars) > self.config.gamma_dollars_threshold:
            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.REDUCE_GAMMA,
                    rationale=(
                        f"Gamma dollar exposure ${greeks.gamma_dollars:,.0f} "
                        f"per 1% move exceeds limit"
                    ),
                    urgency="normal",
                )
            )

        return actions

    def _check_vega(self, greeks: PortfolioGreeks) -> List[HedgeAction]:
        """Check vega exposure."""
        actions = []

        if abs(greeks.vega) > self.config.vega_threshold:
            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.ALERT,
                    rationale=(
                        f"High vega exposure: {greeks.vega:.0f}, "
                        f"sensitive to vol changes"
                    ),
                    urgency="normal",
                )
            )

        if abs(greeks.vega_dollars) > self.config.max_vega_dollars:
            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.REDUCE_VEGA,
                    rationale=(
                        f"Vega dollar exposure ${greeks.vega_dollars:,.0f} "
                        f"exceeds limit ${self.config.max_vega_dollars:,.0f}"
                    ),
                    urgency="normal",
                )
            )

        return actions

    def _check_theta(self, greeks: PortfolioGreeks) -> List[HedgeAction]:
        """Check theta decay."""
        actions = []

        # Theta is typically negative (time decay)
        if greeks.theta < -self.config.max_daily_theta_loss:
            actions.append(
                HedgeAction(
                    action_type=HedgeActionType.ALERT,
                    rationale=(
                        f"High theta decay: ${abs(greeks.theta):,.0f}/day "
                        f"exceeds limit ${self.config.max_daily_theta_loss:,.0f}"
                    ),
                    urgency="normal",
                )
            )

        return actions

    def compute_hedge_trade(
        self,
        current_delta: float,
        underlying: str,
        underlying_price: float,
        use_options: bool = False,
    ) -> Dict:
        """
        Compute specific hedge trade to neutralize delta.

        Args:
            current_delta: Current portfolio delta (in shares)
            underlying: Symbol to hedge with
            underlying_price: Current price of underlying
            use_options: If True, suggest option hedge instead of stock

        Returns:
            Dict with trade details
        """
        if abs(current_delta) < self.config.min_hedge_size:
            return {
                "action": "no_hedge_needed",
                "reason": f"Delta {current_delta:.0f} below minimum {self.config.min_hedge_size}",
            }

        if use_options:
            # Suggest ATM option hedge (simplified)
            # Would need full options chain data for production
            return {
                "action": "hedge_with_options",
                "underlying": underlying,
                "suggested_delta": -current_delta,
                "note": "Use ATM options to achieve target delta",
            }

        # Stock hedge
        shares_to_trade = round(-current_delta)
        side = "buy" if shares_to_trade > 0 else "sell"
        notional = abs(shares_to_trade) * underlying_price

        return {
            "action": "hedge_with_stock",
            "underlying": underlying,
            "side": side,
            "shares": abs(shares_to_trade),
            "notional": notional,
            "expected_delta_after": current_delta + shares_to_trade,
        }

    def summarize_greeks(
        self,
        greeks: PortfolioGreeks,
        portfolio_value: float,
    ) -> Dict:
        """
        Generate human-readable Greeks summary.

        Args:
            greeks: Portfolio Greeks
            portfolio_value: Total portfolio value

        Returns:
            Dict with summary metrics and assessment
        """
        delta_pct = abs(greeks.delta_dollars) / portfolio_value if portfolio_value > 0 else 0

        assessment = "healthy"
        if abs(greeks.delta) > self.config.delta_threshold:
            assessment = "needs_hedging"
        if abs(greeks.gamma) > self.config.gamma_threshold * 2:
            assessment = "high_risk"

        return {
            "assessment": assessment,
            "delta": {
                "shares": greeks.delta,
                "dollars": greeks.delta_dollars,
                "pct_of_portfolio": delta_pct,
                "status": "ok" if abs(greeks.delta) <= self.config.delta_threshold else "high",
            },
            "gamma": {
                "value": greeks.gamma,
                "dollars_per_1pct": greeks.gamma_dollars,
                "status": "ok" if abs(greeks.gamma) <= self.config.gamma_threshold else "high",
            },
            "vega": {
                "value": greeks.vega,
                "dollars": greeks.vega_dollars,
                "status": "ok" if abs(greeks.vega) <= self.config.vega_threshold else "high",
            },
            "theta": {
                "daily_decay": greeks.theta,
                "status": "ok" if greeks.theta > -self.config.max_daily_theta_loss else "high",
            },
        }
