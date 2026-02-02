"""
Smart Order Routing for optimal execution across multiple venues.

Provides:
    - Venue: Trading venue representation
    - SmartOrderRouter: Intelligent order routing
    - RoutingStrategy: Different routing algorithms
    - VenueSelector: Venue selection logic

Reference:
    - Reg NMS best execution requirements
    - SEC Rule 606 order routing disclosure
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .order import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """Types of trading venues."""

    EXCHANGE = "EXCHANGE"  # Lit exchange (NYSE, NASDAQ)
    ATS = "ATS"  # Alternative Trading System
    DARK_POOL = "DARK_POOL"  # Dark pool
    MARKET_MAKER = "MARKET_MAKER"  # Market maker/wholesaler
    ECN = "ECN"  # Electronic Communication Network


class RoutingStrategy(Enum):
    """Order routing strategies."""

    SMART = "SMART"  # Intelligent routing based on multiple factors
    LOWEST_COST = "LOWEST_COST"  # Minimize fees
    FASTEST = "FASTEST"  # Minimize latency
    BEST_LIQUIDITY = "BEST_LIQUIDITY"  # Maximize fill probability
    DARK_ONLY = "DARK_ONLY"  # Route only to dark pools
    SPLIT = "SPLIT"  # Split across multiple venues


@dataclass
class Venue:
    """
    Trading venue representation.

    Attributes:
        venue_id: Unique venue identifier
        name: Display name
        venue_type: Type of venue
        fee_maker: Fee for adding liquidity (per share)
        fee_taker: Fee for removing liquidity (per share)
        rebate_maker: Rebate for adding liquidity
        rebate_taker: Rebate for removing liquidity
        min_order_size: Minimum order size
        max_order_size: Maximum order size
        latency_ms: Average latency in milliseconds
        is_active: Whether venue is currently active
    """

    venue_id: str
    name: str
    venue_type: VenueType = VenueType.EXCHANGE
    fee_maker: float = 0.0
    fee_taker: float = 0.003  # $0.003/share default
    rebate_maker: float = 0.002  # $0.002/share rebate
    rebate_taker: float = 0.0
    min_order_size: float = 1.0
    max_order_size: float = 1000000.0
    latency_ms: float = 1.0
    is_active: bool = True

    # Performance metrics (updated dynamically)
    fill_rate: float = 0.95  # Historical fill rate
    avg_fill_time_ms: float = 100.0  # Average time to fill
    avg_price_improvement: float = 0.0  # Average price improvement (bps)
    market_share: float = 0.0  # Venue market share

    # Order book info (updated in real-time)
    best_bid: float = 0.0
    best_ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def spread(self) -> float:
        """Current bid-ask spread."""
        if self.best_bid > 0 and self.best_ask > 0:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        if self.best_bid > 0:
            return (self.spread / self.best_bid) * 10000
        return 0.0

    def calculate_cost(
        self,
        quantity: float,
        is_aggressive: bool = True,
    ) -> float:
        """
        Calculate execution cost at this venue.

        Args:
            quantity: Order quantity
            is_aggressive: True if removing liquidity

        Returns:
            Net cost (positive = cost, negative = rebate)
        """
        if is_aggressive:
            return quantity * (self.fee_taker - self.rebate_taker)
        else:
            return quantity * (self.fee_maker - self.rebate_maker)


@dataclass
class RoutingDecision:
    """
    Routing decision for an order or order slice.

    Attributes:
        venue: Target venue
        quantity: Quantity to route
        order_type: Order type to use
        price: Limit price (if applicable)
        expected_cost: Expected execution cost
        expected_fill_prob: Probability of fill
        rationale: Explanation for routing decision
    """

    venue: Venue
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = None
    expected_cost: float = 0.0
    expected_fill_prob: float = 0.95
    rationale: str = ""

    @property
    def expected_value(self) -> float:
        """Expected notional value."""
        if self.price:
            return self.quantity * self.price
        return 0.0


@dataclass
class RoutingPlan:
    """
    Complete routing plan for an order.

    Contains all routing decisions and child orders.
    """

    parent_order: Order
    decisions: List[RoutingDecision] = field(default_factory=list)
    child_orders: List[Order] = field(default_factory=list)
    strategy_used: RoutingStrategy = RoutingStrategy.SMART
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_quantity(self) -> float:
        """Total quantity across all decisions."""
        return sum(d.quantity for d in self.decisions)

    @property
    def total_expected_cost(self) -> float:
        """Total expected execution cost."""
        return sum(d.expected_cost for d in self.decisions)

    @property
    def n_venues(self) -> int:
        """Number of venues in plan."""
        return len(set(d.venue.venue_id for d in self.decisions))


class VenueScorer:
    """
    Scores venues for order routing.

    Uses multiple factors to rank venues.
    """

    def __init__(
        self,
        cost_weight: float = 0.3,
        liquidity_weight: float = 0.3,
        fill_rate_weight: float = 0.2,
        latency_weight: float = 0.1,
        price_improvement_weight: float = 0.1,
    ):
        """
        Initialize venue scorer.

        Args:
            cost_weight: Weight for execution cost
            liquidity_weight: Weight for liquidity
            fill_rate_weight: Weight for historical fill rate
            latency_weight: Weight for latency
            price_improvement_weight: Weight for price improvement
        """
        self.cost_weight = cost_weight
        self.liquidity_weight = liquidity_weight
        self.fill_rate_weight = fill_rate_weight
        self.latency_weight = latency_weight
        self.price_improvement_weight = price_improvement_weight

    def score_venue(
        self,
        venue: Venue,
        order: Order,
        all_venues: List[Venue],
    ) -> float:
        """
        Score a venue for an order.

        Args:
            venue: Venue to score
            order: Order being routed
            all_venues: All available venues (for normalization)

        Returns:
            Score between 0 and 1 (higher = better)
        """
        # Cost score (inverted - lower cost = higher score)
        is_aggressive = order.order_type == OrderType.MARKET
        cost = venue.calculate_cost(order.quantity, is_aggressive)
        max_cost = max(
            v.calculate_cost(order.quantity, is_aggressive)
            for v in all_venues
        )
        if max_cost > 0:
            cost_score = 1 - (cost / max_cost)
        else:
            cost_score = 1.0

        # Liquidity score
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            available_liquidity = venue.ask_size
        else:
            available_liquidity = venue.bid_size

        total_liquidity = sum(
            v.ask_size if order.side in [OrderSide.BUY, OrderSide.COVER]
            else v.bid_size
            for v in all_venues
        )
        if total_liquidity > 0:
            liquidity_score = available_liquidity / total_liquidity
        else:
            liquidity_score = 0.5

        # Fill rate score
        fill_rate_score = venue.fill_rate

        # Latency score (inverted)
        max_latency = max(v.latency_ms for v in all_venues)
        if max_latency > 0:
            latency_score = 1 - (venue.latency_ms / max_latency)
        else:
            latency_score = 1.0

        # Price improvement score (normalized to 0-1)
        # Assume max improvement is 10 bps
        price_improvement_score = min(venue.avg_price_improvement / 10, 1.0)

        # Weighted combination
        total_score = (
            self.cost_weight * cost_score
            + self.liquidity_weight * liquidity_score
            + self.fill_rate_weight * fill_rate_score
            + self.latency_weight * latency_score
            + self.price_improvement_weight * price_improvement_score
        )

        return total_score

    def rank_venues(
        self,
        venues: List[Venue],
        order: Order,
    ) -> List[Tuple[Venue, float]]:
        """
        Rank venues by score.

        Args:
            venues: Available venues
            order: Order being routed

        Returns:
            List of (venue, score) tuples, sorted by score descending
        """
        active_venues = [v for v in venues if v.is_active]
        scored = [
            (v, self.score_venue(v, order, active_venues))
            for v in active_venues
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)


class SmartOrderRouter:
    """
    Smart order router for multi-venue execution.

    Routes orders across multiple venues to minimize cost
    and market impact while maximizing fill probability.

    Example:
        >>> venues = [nyse, nasdaq, iex, bats]
        >>> router = SmartOrderRouter(venues)
        >>>
        >>> # Route order
        >>> plan = router.route_order(order)
        >>> for decision in plan.decisions:
        ...     print(f"Route {decision.quantity} to {decision.venue.name}")
    """

    def __init__(
        self,
        venues: List[Venue],
        default_strategy: RoutingStrategy = RoutingStrategy.SMART,
        min_slice_quantity: float = 100.0,
        max_slices: int = 5,
        scorer: Optional[VenueScorer] = None,
    ):
        """
        Initialize smart order router.

        Args:
            venues: List of available venues
            default_strategy: Default routing strategy
            min_slice_quantity: Minimum quantity per slice
            max_slices: Maximum number of order slices
            scorer: Venue scoring algorithm
        """
        self.venues = {v.venue_id: v for v in venues}
        self.default_strategy = default_strategy
        self.min_slice_quantity = min_slice_quantity
        self.max_slices = max_slices
        self.scorer = scorer or VenueScorer()

        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []

        logger.info(
            f"SmartOrderRouter initialized with {len(venues)} venues"
        )

    def route_order(
        self,
        order: Order,
        strategy: Optional[RoutingStrategy] = None,
        **kwargs,
    ) -> RoutingPlan:
        """
        Route an order across venues.

        Args:
            order: Order to route
            strategy: Routing strategy (uses default if None)
            **kwargs: Strategy-specific parameters

        Returns:
            RoutingPlan with routing decisions
        """
        strategy = strategy or self.default_strategy

        if strategy == RoutingStrategy.SMART:
            return self._route_smart(order, **kwargs)
        elif strategy == RoutingStrategy.LOWEST_COST:
            return self._route_lowest_cost(order)
        elif strategy == RoutingStrategy.FASTEST:
            return self._route_fastest(order)
        elif strategy == RoutingStrategy.BEST_LIQUIDITY:
            return self._route_best_liquidity(order)
        elif strategy == RoutingStrategy.DARK_ONLY:
            return self._route_dark_only(order)
        elif strategy == RoutingStrategy.SPLIT:
            return self._route_split(order, **kwargs)
        else:
            return self._route_smart(order, **kwargs)

    def _route_smart(
        self,
        order: Order,
        max_market_impact_bps: float = 10.0,
    ) -> RoutingPlan:
        """
        Smart routing based on multiple factors.

        Uses venue scoring to allocate order across venues.
        """
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.SMART,
        )

        # For small orders, use single best venue
        if order.quantity < self.min_slice_quantity * 2:
            ranked = self.scorer.rank_venues(list(self.venues.values()), order)
            if ranked:
                best_venue, score = ranked[0]
                plan.decisions.append(RoutingDecision(
                    venue=best_venue,
                    quantity=order.quantity,
                    order_type=order.order_type,
                    price=order.price,
                    expected_cost=best_venue.calculate_cost(
                        order.quantity,
                        order.order_type == OrderType.MARKET,
                    ),
                    rationale=f"Single venue (score={score:.2f})",
                ))
            return plan

        # For larger orders, split across venues
        ranked = self.scorer.rank_venues(list(self.venues.values()), order)
        remaining_quantity = order.quantity

        # Allocate to top venues by score
        total_score = sum(score for _, score in ranked[:self.max_slices])

        for venue, score in ranked[:self.max_slices]:
            if remaining_quantity <= 0:
                break

            # Score-weighted allocation
            allocation = (score / total_score) * order.quantity
            slice_qty = min(
                max(allocation, self.min_slice_quantity),
                remaining_quantity,
            )

            if slice_qty < self.min_slice_quantity:
                continue

            plan.decisions.append(RoutingDecision(
                venue=venue,
                quantity=slice_qty,
                order_type=order.order_type,
                price=order.price,
                expected_cost=venue.calculate_cost(
                    slice_qty,
                    order.order_type == OrderType.MARKET,
                ),
                rationale=f"Score-weighted allocation (score={score:.2f})",
            ))

            remaining_quantity -= slice_qty

        # Allocate any remaining to best venue
        if remaining_quantity > 0 and ranked:
            best_venue = ranked[0][0]
            for decision in plan.decisions:
                if decision.venue.venue_id == best_venue.venue_id:
                    decision.quantity += remaining_quantity
                    break
            else:
                plan.decisions.append(RoutingDecision(
                    venue=best_venue,
                    quantity=remaining_quantity,
                    order_type=order.order_type,
                    price=order.price,
                    expected_cost=best_venue.calculate_cost(
                        remaining_quantity,
                        order.order_type == OrderType.MARKET,
                    ),
                    rationale="Remainder to best venue",
                ))

        # Create child orders
        for i, decision in enumerate(plan.decisions):
            child = order.clone(decision.quantity)
            child.venue = decision.venue.venue_id
            child.metadata["routing_slice"] = i
            child.metadata["routing_rationale"] = decision.rationale
            plan.child_orders.append(child)
            order.add_child(child.order_id)

        return plan

    def _route_lowest_cost(self, order: Order) -> RoutingPlan:
        """Route to venue with lowest execution cost."""
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.LOWEST_COST,
        )

        is_aggressive = order.order_type == OrderType.MARKET
        active_venues = [v for v in self.venues.values() if v.is_active]

        if not active_venues:
            return plan

        # Find lowest cost venue
        best_venue = min(
            active_venues,
            key=lambda v: v.calculate_cost(order.quantity, is_aggressive),
        )

        plan.decisions.append(RoutingDecision(
            venue=best_venue,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            expected_cost=best_venue.calculate_cost(order.quantity, is_aggressive),
            rationale="Lowest cost venue",
        ))

        # Create child order
        child = order.clone()
        child.venue = best_venue.venue_id
        plan.child_orders.append(child)
        order.add_child(child.order_id)

        return plan

    def _route_fastest(self, order: Order) -> RoutingPlan:
        """Route to venue with lowest latency."""
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.FASTEST,
        )

        active_venues = [v for v in self.venues.values() if v.is_active]

        if not active_venues:
            return plan

        # Find fastest venue
        best_venue = min(active_venues, key=lambda v: v.latency_ms)

        plan.decisions.append(RoutingDecision(
            venue=best_venue,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            rationale=f"Lowest latency ({best_venue.latency_ms}ms)",
        ))

        # Create child order
        child = order.clone()
        child.venue = best_venue.venue_id
        plan.child_orders.append(child)
        order.add_child(child.order_id)

        return plan

    def _route_best_liquidity(self, order: Order) -> RoutingPlan:
        """Route to venue with best liquidity for order side."""
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.BEST_LIQUIDITY,
        )

        active_venues = [v for v in self.venues.values() if v.is_active]

        if not active_venues:
            return plan

        # Find venue with most liquidity on correct side
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            best_venue = max(active_venues, key=lambda v: v.ask_size)
            liquidity = best_venue.ask_size
        else:
            best_venue = max(active_venues, key=lambda v: v.bid_size)
            liquidity = best_venue.bid_size

        plan.decisions.append(RoutingDecision(
            venue=best_venue,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            rationale=f"Best liquidity ({liquidity:,.0f} shares)",
        ))

        # Create child order
        child = order.clone()
        child.venue = best_venue.venue_id
        plan.child_orders.append(child)
        order.add_child(child.order_id)

        return plan

    def _route_dark_only(self, order: Order) -> RoutingPlan:
        """Route only to dark pools."""
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.DARK_ONLY,
        )

        dark_venues = [
            v for v in self.venues.values()
            if v.is_active and v.venue_type == VenueType.DARK_POOL
        ]

        if not dark_venues:
            logger.warning("No active dark pools available")
            return plan

        # Split evenly across dark pools
        qty_per_venue = order.quantity / len(dark_venues)

        for venue in dark_venues:
            if qty_per_venue < venue.min_order_size:
                continue

            plan.decisions.append(RoutingDecision(
                venue=venue,
                quantity=qty_per_venue,
                order_type=OrderType.LIMIT,  # Dark pools typically use limit
                price=order.price,
                rationale="Dark pool routing",
            ))

            child = order.clone(qty_per_venue)
            child.venue = venue.venue_id
            child.order_type = OrderType.LIMIT
            plan.child_orders.append(child)
            order.add_child(child.order_id)

        return plan

    def _route_split(
        self,
        order: Order,
        split_weights: Optional[Dict[str, float]] = None,
    ) -> RoutingPlan:
        """
        Split order across venues with specified weights.

        Args:
            order: Order to route
            split_weights: Dict of venue_id -> weight (should sum to 1.0)
        """
        plan = RoutingPlan(
            parent_order=order,
            strategy_used=RoutingStrategy.SPLIT,
        )

        active_venues = [v for v in self.venues.values() if v.is_active]

        if not active_venues:
            return plan

        # Default to equal split
        if split_weights is None:
            split_weights = {
                v.venue_id: 1.0 / len(active_venues)
                for v in active_venues
            }

        # Normalize weights
        total_weight = sum(split_weights.values())
        if total_weight > 0:
            split_weights = {k: v / total_weight for k, v in split_weights.items()}

        for venue in active_venues:
            weight = split_weights.get(venue.venue_id, 0)
            if weight <= 0:
                continue

            slice_qty = order.quantity * weight

            if slice_qty < self.min_slice_quantity:
                continue

            plan.decisions.append(RoutingDecision(
                venue=venue,
                quantity=slice_qty,
                order_type=order.order_type,
                price=order.price,
                rationale=f"Split allocation ({weight:.1%})",
            ))

            child = order.clone(slice_qty)
            child.venue = venue.venue_id
            plan.child_orders.append(child)
            order.add_child(child.order_id)

        return plan

    def update_venue(
        self,
        venue_id: str,
        **updates,
    ) -> None:
        """
        Update venue information.

        Args:
            venue_id: Venue ID
            **updates: Fields to update
        """
        if venue_id not in self.venues:
            return

        venue = self.venues[venue_id]
        for key, value in updates.items():
            if hasattr(venue, key):
                setattr(venue, key, value)

    def update_quote(
        self,
        venue_id: str,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> None:
        """
        Update venue quote data.

        Args:
            venue_id: Venue ID
            bid: Best bid price
            ask: Best ask price
            bid_size: Bid size
            ask_size: Ask size
        """
        if venue_id not in self.venues:
            return

        venue = self.venues[venue_id]
        venue.best_bid = bid
        venue.best_ask = ask
        venue.bid_size = bid_size
        venue.ask_size = ask_size

    def get_venue_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all venues."""
        return [
            {
                "venue_id": v.venue_id,
                "name": v.name,
                "type": v.venue_type.value,
                "active": v.is_active,
                "fee_taker": v.fee_taker,
                "rebate_maker": v.rebate_maker,
                "fill_rate": v.fill_rate,
                "latency_ms": v.latency_ms,
                "spread_bps": v.spread_bps,
            }
            for v in self.venues.values()
        ]
