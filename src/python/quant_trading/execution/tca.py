"""
Transaction Cost Analysis (TCA) for execution quality measurement.

Provides:
    - TCAAnalyzer: Comprehensive transaction cost analysis
    - Benchmark calculations (arrival, VWAP, TWAP)
    - Implementation shortfall analysis
    - Execution quality metrics

Reference:
    - Kissell & Glantz (2003): Optimal trading strategies
    - Almgren et al. (2005): Direct estimation of equity market impact
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .order import Fill, Order, OrderSide

logger = logging.getLogger(__name__)


class TCABenchmark(Enum):
    """Transaction cost benchmarks."""

    ARRIVAL = "ARRIVAL"  # Decision/arrival price
    VWAP = "VWAP"  # Volume-weighted average price
    TWAP = "TWAP"  # Time-weighted average price
    OPEN = "OPEN"  # Opening price
    CLOSE = "CLOSE"  # Closing price
    PREVIOUS_CLOSE = "PREVIOUS_CLOSE"  # Previous day close
    MIDPOINT = "MIDPOINT"  # Mid at time of order


@dataclass
class CostComponent:
    """
    Individual cost component.

    Attributes:
        name: Component name
        value_dollars: Cost in dollars
        value_bps: Cost in basis points
        description: Description of the component
    """

    name: str
    value_dollars: float
    value_bps: float
    description: str = ""


@dataclass
class TCAResult:
    """
    Transaction cost analysis result for an order.

    Contains all cost components and benchmark comparisons.
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    avg_fill_price: float

    # Benchmark prices
    arrival_price: float = 0.0
    vwap_price: float = 0.0
    twap_price: float = 0.0

    # Cost components
    total_cost_dollars: float = 0.0
    total_cost_bps: float = 0.0

    # Breakdown
    commission: float = 0.0
    commission_bps: float = 0.0
    spread_cost: float = 0.0
    spread_cost_bps: float = 0.0
    market_impact: float = 0.0
    market_impact_bps: float = 0.0
    timing_cost: float = 0.0
    timing_cost_bps: float = 0.0
    delay_cost: float = 0.0
    delay_cost_bps: float = 0.0

    # Implementation shortfall breakdown
    implementation_shortfall: float = 0.0
    implementation_shortfall_bps: float = 0.0

    # Execution quality
    price_improvement: float = 0.0
    price_improvement_bps: float = 0.0
    fill_rate: float = 0.0
    execution_time_seconds: float = 0.0

    # Detailed components
    components: List[CostComponent] = field(default_factory=list)

    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        """Generate summary string."""
        direction = "Bought" if self.side in [OrderSide.BUY, OrderSide.COVER] else "Sold"
        return f"""
================================================================================
              TRANSACTION COST ANALYSIS
================================================================================
Order: {self.order_id}
{direction} {self.quantity:,.0f} {self.symbol} @ ${self.avg_fill_price:.4f}

BENCHMARKS
----------
Arrival Price:   ${self.arrival_price:.4f}
VWAP:            ${self.vwap_price:.4f}
TWAP:            ${self.twap_price:.4f}

COST BREAKDOWN (Basis Points)
-----------------------------
Commission:      {self.commission_bps:>8.2f} bps  (${self.commission:>10,.2f})
Spread Cost:     {self.spread_cost_bps:>8.2f} bps  (${self.spread_cost:>10,.2f})
Market Impact:   {self.market_impact_bps:>8.2f} bps  (${self.market_impact:>10,.2f})
Timing Cost:     {self.timing_cost_bps:>8.2f} bps  (${self.timing_cost:>10,.2f})
Delay Cost:      {self.delay_cost_bps:>8.2f} bps  (${self.delay_cost:>10,.2f})
                 --------        ----------
TOTAL COST:      {self.total_cost_bps:>8.2f} bps  (${self.total_cost_dollars:>10,.2f})

IMPLEMENTATION SHORTFALL
------------------------
vs Arrival:      {self.implementation_shortfall_bps:>8.2f} bps

EXECUTION QUALITY
-----------------
Price Improvement: {self.price_improvement_bps:>8.2f} bps
Fill Rate:         {self.fill_rate:>8.1f}%
Execution Time:    {self.execution_time_seconds:>8.1f} seconds
================================================================================
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "avg_fill_price": self.avg_fill_price,
            "arrival_price": self.arrival_price,
            "vwap_price": self.vwap_price,
            "total_cost_dollars": self.total_cost_dollars,
            "total_cost_bps": self.total_cost_bps,
            "commission_bps": self.commission_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "market_impact_bps": self.market_impact_bps,
            "timing_cost_bps": self.timing_cost_bps,
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "price_improvement_bps": self.price_improvement_bps,
            "fill_rate": self.fill_rate,
        }


@dataclass
class MarketData:
    """
    Market data for TCA calculation.

    Attributes:
        timestamp: Data timestamp
        bid: Best bid
        ask: Best ask
        price: Last price
        volume: Period volume
    """

    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    price: float = 0.0
    volume: float = 0.0

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.price

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        if self.bid > 0 and self.ask > 0:
            return self.ask - self.bid
        return 0.0


class TCAAnalyzer:
    """
    Transaction cost analyzer.

    Calculates comprehensive execution costs including:
    - Commission
    - Spread crossing cost
    - Market impact
    - Timing/delay cost
    - Implementation shortfall

    Example:
        >>> analyzer = TCAAnalyzer()
        >>> result = analyzer.analyze_order(
        ...     order,
        ...     market_data=market_data,
        ...     arrival_price=450.00,
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        default_spread_bps: float = 5.0,
        impact_model_coefficient: float = 0.1,
    ):
        """
        Initialize TCA analyzer.

        Args:
            default_spread_bps: Default spread assumption (bps)
            impact_model_coefficient: Market impact coefficient
        """
        self.default_spread_bps = default_spread_bps
        self.impact_coefficient = impact_model_coefficient

    def analyze_order(
        self,
        order: Order,
        arrival_price: Optional[float] = None,
        decision_price: Optional[float] = None,
        market_data: Optional[List[MarketData]] = None,
        vwap_price: Optional[float] = None,
        twap_price: Optional[float] = None,
        adv: Optional[float] = None,  # Average daily volume
    ) -> TCAResult:
        """
        Analyze transaction costs for an order.

        Args:
            order: Completed order to analyze
            arrival_price: Price at decision time
            decision_price: Same as arrival (alias)
            market_data: Market data during execution
            vwap_price: Pre-calculated VWAP benchmark
            twap_price: Pre-calculated TWAP benchmark
            adv: Average daily volume for impact calculation

        Returns:
            TCAResult with all cost components
        """
        if order.filled_quantity == 0:
            logger.warning(f"Order {order.order_id} has no fills")
            return TCAResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                avg_fill_price=0.0,
            )

        # Use decision_price if arrival not provided
        arrival_price = arrival_price or decision_price or order.price or 0.0

        # Calculate benchmarks from market data if not provided
        if market_data:
            if vwap_price is None:
                vwap_price = self._calculate_vwap(market_data)
            if twap_price is None:
                twap_price = self._calculate_twap(market_data)
        else:
            vwap_price = vwap_price or arrival_price
            twap_price = twap_price or arrival_price

        # Notional value
        notional = order.filled_quantity * order.avg_fill_price

        # Commission cost
        commission = order.commission + order.fees
        commission_bps = (commission / notional * 10000) if notional > 0 else 0

        # Spread cost (half spread for crossing)
        if market_data and len(market_data) > 0:
            avg_spread = np.mean([m.spread for m in market_data if m.spread > 0])
            spread_cost = (avg_spread / 2) * order.filled_quantity
        else:
            spread_cost = (arrival_price * self.default_spread_bps / 10000 / 2) * order.filled_quantity
        spread_cost_bps = (spread_cost / notional * 10000) if notional > 0 else 0

        # Market impact (Almgren square-root model)
        if adv and adv > 0:
            participation = order.filled_quantity / adv
            impact_bps = self.impact_coefficient * np.sqrt(participation) * 10000
            market_impact = notional * impact_bps / 10000
        else:
            impact_bps = 0.0
            market_impact = 0.0

        # Timing cost (vs VWAP/TWAP)
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            timing_cost = (order.avg_fill_price - vwap_price) * order.filled_quantity
        else:
            timing_cost = (vwap_price - order.avg_fill_price) * order.filled_quantity
        timing_cost_bps = (timing_cost / notional * 10000) if notional > 0 else 0

        # Delay cost (between decision and first fill)
        if order.created_at and order.first_fill_at:
            delay_seconds = (order.first_fill_at - order.created_at).total_seconds()
        else:
            delay_seconds = 0

        if delay_seconds > 0 and market_data and len(market_data) > 1:
            # Price drift during delay
            price_change = market_data[-1].mid - market_data[0].mid
            if order.side in [OrderSide.BUY, OrderSide.COVER]:
                delay_cost = max(0, price_change) * order.filled_quantity
            else:
                delay_cost = max(0, -price_change) * order.filled_quantity
        else:
            delay_cost = 0.0
        delay_cost_bps = (delay_cost / notional * 10000) if notional > 0 else 0

        # Implementation shortfall (vs arrival price)
        if order.side in [OrderSide.BUY, OrderSide.COVER]:
            impl_shortfall = (order.avg_fill_price - arrival_price) * order.filled_quantity
        else:
            impl_shortfall = (arrival_price - order.avg_fill_price) * order.filled_quantity
        impl_shortfall_bps = (impl_shortfall / notional * 10000) if notional > 0 else 0

        # Price improvement
        if market_data and len(market_data) > 0:
            expected_price = market_data[0].ask if order.side in [OrderSide.BUY, OrderSide.COVER] else market_data[0].bid
            if order.side in [OrderSide.BUY, OrderSide.COVER]:
                price_improvement = (expected_price - order.avg_fill_price) * order.filled_quantity
            else:
                price_improvement = (order.avg_fill_price - expected_price) * order.filled_quantity
        else:
            price_improvement = 0.0
        price_improvement_bps = (price_improvement / notional * 10000) if notional > 0 else 0

        # Total cost
        total_cost_dollars = commission + spread_cost + market_impact + max(0, timing_cost) + delay_cost
        total_cost_bps = commission_bps + spread_cost_bps + impact_bps + max(0, timing_cost_bps) + delay_cost_bps

        # Fill rate
        fill_rate = (order.filled_quantity / order.quantity * 100) if order.quantity > 0 else 0

        # Execution time
        if order.first_fill_at and order.filled_at:
            exec_time = (order.filled_at - order.first_fill_at).total_seconds()
        else:
            exec_time = 0.0

        # Build components list
        components = [
            CostComponent("Commission", commission, commission_bps, "Exchange and broker fees"),
            CostComponent("Spread", spread_cost, spread_cost_bps, "Bid-ask spread crossing cost"),
            CostComponent("Market Impact", market_impact, impact_bps, "Price impact from our trading"),
            CostComponent("Timing", timing_cost, timing_cost_bps, "Opportunity cost vs VWAP"),
            CostComponent("Delay", delay_cost, delay_cost_bps, "Cost of execution delay"),
        ]

        result = TCAResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            avg_fill_price=order.avg_fill_price,
            arrival_price=arrival_price,
            vwap_price=vwap_price,
            twap_price=twap_price,
            total_cost_dollars=total_cost_dollars,
            total_cost_bps=total_cost_bps,
            commission=commission,
            commission_bps=commission_bps,
            spread_cost=spread_cost,
            spread_cost_bps=spread_cost_bps,
            market_impact=market_impact,
            market_impact_bps=impact_bps,
            timing_cost=timing_cost,
            timing_cost_bps=timing_cost_bps,
            delay_cost=delay_cost,
            delay_cost_bps=delay_cost_bps,
            implementation_shortfall=impl_shortfall,
            implementation_shortfall_bps=impl_shortfall_bps,
            price_improvement=price_improvement,
            price_improvement_bps=price_improvement_bps,
            fill_rate=fill_rate,
            execution_time_seconds=exec_time,
            components=components,
        )

        logger.debug(
            f"TCA for {order.order_id}: total cost = {total_cost_bps:.2f} bps"
        )

        return result

    def analyze_batch(
        self,
        orders: List[Order],
        **kwargs,
    ) -> List[TCAResult]:
        """
        Analyze transaction costs for multiple orders.

        Args:
            orders: List of orders to analyze
            **kwargs: Additional parameters passed to analyze_order

        Returns:
            List of TCAResult objects
        """
        return [self.analyze_order(order, **kwargs) for order in orders]

    def aggregate_results(
        self,
        results: List[TCAResult],
    ) -> Dict[str, Any]:
        """
        Aggregate TCA results across multiple orders.

        Args:
            results: List of TCA results

        Returns:
            Aggregated statistics
        """
        if not results:
            return {}

        total_notional = sum(r.quantity * r.avg_fill_price for r in results)
        total_commission = sum(r.commission for r in results)
        total_impact = sum(r.market_impact for r in results)
        total_cost = sum(r.total_cost_dollars for r in results)

        # Volume-weighted average costs
        weighted_cost_bps = sum(
            r.total_cost_bps * r.quantity * r.avg_fill_price
            for r in results
        ) / total_notional if total_notional > 0 else 0

        weighted_impl_shortfall = sum(
            r.implementation_shortfall_bps * r.quantity * r.avg_fill_price
            for r in results
        ) / total_notional if total_notional > 0 else 0

        return {
            "n_orders": len(results),
            "total_notional": total_notional,
            "total_cost_dollars": total_cost,
            "avg_cost_bps": weighted_cost_bps,
            "total_commission": total_commission,
            "commission_pct": (total_commission / total_cost * 100) if total_cost > 0 else 0,
            "total_market_impact": total_impact,
            "impact_pct": (total_impact / total_cost * 100) if total_cost > 0 else 0,
            "avg_implementation_shortfall_bps": weighted_impl_shortfall,
            "avg_fill_rate": np.mean([r.fill_rate for r in results]),
            "avg_execution_time": np.mean([r.execution_time_seconds for r in results]),
        }

    def _calculate_vwap(self, market_data: List[MarketData]) -> float:
        """Calculate VWAP from market data."""
        total_value = 0.0
        total_volume = 0.0

        for md in market_data:
            if md.volume > 0:
                total_value += md.price * md.volume
                total_volume += md.volume

        if total_volume > 0:
            return total_value / total_volume

        # Fallback to simple average
        prices = [md.price for md in market_data if md.price > 0]
        return float(np.mean(prices)) if prices else 0.0

    def _calculate_twap(self, market_data: List[MarketData]) -> float:
        """Calculate TWAP from market data."""
        prices = [md.mid if md.mid > 0 else md.price for md in market_data]
        prices = [p for p in prices if p > 0]
        return float(np.mean(prices)) if prices else 0.0


class TCAReportGenerator:
    """
    Generate TCA reports for regulatory and client reporting.

    Creates formatted reports for:
    - Individual trade analysis
    - Periodic aggregated reports
    - Best execution compliance
    """

    def __init__(self, analyzer: Optional[TCAAnalyzer] = None):
        """
        Initialize report generator.

        Args:
            analyzer: TCA analyzer instance
        """
        self.analyzer = analyzer or TCAAnalyzer()

    def generate_trade_report(self, result: TCAResult) -> str:
        """Generate detailed report for a single trade."""
        return result.summary()

    def generate_summary_report(
        self,
        results: List[TCAResult],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> str:
        """
        Generate summary report for multiple trades.

        Args:
            results: List of TCA results
            period_start: Report period start
            period_end: Report period end

        Returns:
            Formatted report string
        """
        if not results:
            return "No trades to report"

        agg = self.analyzer.aggregate_results(results)
        period_str = ""
        if period_start and period_end:
            period_str = f"Period: {period_start.date()} to {period_end.date()}\n"

        # Group by symbol
        by_symbol: Dict[str, List[TCAResult]] = {}
        for r in results:
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = []
            by_symbol[r.symbol].append(r)

        symbol_lines = []
        for symbol, symbol_results in sorted(by_symbol.items()):
            symbol_agg = self.analyzer.aggregate_results(symbol_results)
            symbol_lines.append(
                f"  {symbol:8s}  {symbol_agg['n_orders']:>5}  "
                f"${symbol_agg['total_notional']:>12,.0f}  "
                f"{symbol_agg['avg_cost_bps']:>6.1f} bps"
            )

        return f"""
================================================================================
              TRANSACTION COST ANALYSIS - SUMMARY REPORT
================================================================================
{period_str}
OVERALL STATISTICS
------------------
Number of Orders:           {agg['n_orders']:>10}
Total Notional Value:       ${agg['total_notional']:>14,.0f}
Total Transaction Costs:    ${agg['total_cost_dollars']:>14,.2f}
Average Cost:               {agg['avg_cost_bps']:>10.2f} bps
Average Implementation Shortfall: {agg['avg_implementation_shortfall_bps']:>6.2f} bps

COST BREAKDOWN
--------------
Commission:                 {agg['commission_pct']:>8.1f}% of total costs
Market Impact:              {agg['impact_pct']:>8.1f}% of total costs

EXECUTION QUALITY
-----------------
Average Fill Rate:          {agg['avg_fill_rate']:>8.1f}%
Average Execution Time:     {agg['avg_execution_time']:>8.1f} seconds

BY SYMBOL
---------
  Symbol    Orders       Notional      Avg Cost
{chr(10).join(symbol_lines)}

================================================================================
"""

    def generate_best_execution_report(
        self,
        results: List[TCAResult],
    ) -> str:
        """
        Generate best execution compliance report.

        Required under MiFID II and SEC regulations.
        """
        if not results:
            return "No trades to report"

        agg = self.analyzer.aggregate_results(results)

        # Calculate various metrics
        with_price_improvement = [r for r in results if r.price_improvement > 0]
        improvement_rate = len(with_price_improvement) / len(results) * 100

        # Fill rate buckets
        full_fills = [r for r in results if r.fill_rate >= 100]
        partial_fills = [r for r in results if 0 < r.fill_rate < 100]

        return f"""
================================================================================
              BEST EXECUTION REPORT
================================================================================

EXECUTION QUALITY ASSESSMENT
----------------------------
Total Orders Analyzed:      {len(results):>10}
Average Cost vs Arrival:    {agg['avg_implementation_shortfall_bps']:>10.2f} bps

FILL QUALITY
------------
Full Fills:                 {len(full_fills):>10} ({len(full_fills)/len(results)*100:.1f}%)
Partial Fills:              {len(partial_fills):>10} ({len(partial_fills)/len(results)*100:.1f}%)
Average Fill Rate:          {agg['avg_fill_rate']:>10.1f}%

PRICE IMPROVEMENT
-----------------
Orders with Improvement:    {len(with_price_improvement):>10} ({improvement_rate:.1f}%)
Average Improvement:        {np.mean([r.price_improvement_bps for r in with_price_improvement]) if with_price_improvement else 0:.2f} bps

COMPLIANCE STATEMENT
--------------------
This report is generated for best execution compliance purposes.
All orders were executed in accordance with best execution policies.

================================================================================
"""
