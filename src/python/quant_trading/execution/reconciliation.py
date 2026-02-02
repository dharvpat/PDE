"""
Fill reconciliation for order-fill matching and discrepancy detection.

Provides:
    - FillReconciler: Match orders with fills
    - Discrepancy detection and alerting
    - Audit trail generation
    - Position reconciliation

Reference:
    - SEC Rule 15c3-1 net capital requirements
    - FINRA Rule 4311 order handling and audit trail
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .order import Fill, Order, OrderStatus

logger = logging.getLogger(__name__)


class DiscrepancyType(Enum):
    """Types of reconciliation discrepancies."""

    MISSING_FILL = "MISSING_FILL"  # Order filled but no fill record
    ORPHAN_FILL = "ORPHAN_FILL"  # Fill without matching order
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"  # Fill quantity doesn't match
    PRICE_MISMATCH = "PRICE_MISMATCH"  # Fill price differs significantly
    DUPLICATE_FILL = "DUPLICATE_FILL"  # Same fill recorded twice
    POSITION_MISMATCH = "POSITION_MISMATCH"  # Position doesn't match broker
    COMMISSION_MISMATCH = "COMMISSION_MISMATCH"  # Commission differs


class DiscrepancySeverity(Enum):
    """Severity levels for discrepancies."""

    INFO = "INFO"  # Informational only
    WARNING = "WARNING"  # Requires review
    ERROR = "ERROR"  # Requires immediate action
    CRITICAL = "CRITICAL"  # Trading should be halted


@dataclass
class Discrepancy:
    """
    A reconciliation discrepancy.

    Attributes:
        discrepancy_id: Unique identifier
        discrepancy_type: Type of discrepancy
        severity: Severity level
        order_id: Related order ID
        fill_id: Related fill ID
        expected_value: Expected value
        actual_value: Actual value
        difference: Difference between expected and actual
        description: Human-readable description
        detected_at: When discrepancy was detected
        resolved: Whether discrepancy has been resolved
        resolution_notes: Notes on resolution
    """

    discrepancy_id: str = ""
    discrepancy_type: DiscrepancyType = DiscrepancyType.MISSING_FILL
    severity: DiscrepancySeverity = DiscrepancySeverity.WARNING
    order_id: Optional[str] = None
    fill_id: Optional[str] = None
    expected_value: Any = None
    actual_value: Any = None
    difference: float = 0.0
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def resolve(self, notes: str = "") -> None:
        """Mark discrepancy as resolved."""
        self.resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = notes


@dataclass
class ReconciliationResult:
    """
    Result of a reconciliation run.

    Attributes:
        reconciliation_id: Unique run identifier
        start_time: When reconciliation started
        end_time: When reconciliation completed
        orders_checked: Number of orders checked
        fills_checked: Number of fills checked
        discrepancies: List of discrepancies found
        is_clean: Whether reconciliation passed without errors
    """

    reconciliation_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    orders_checked: int = 0
    fills_checked: int = 0
    discrepancies: List[Discrepancy] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Check if reconciliation is clean (no errors or critical issues)."""
        return not any(
            d.severity in [DiscrepancySeverity.ERROR, DiscrepancySeverity.CRITICAL]
            for d in self.discrepancies
            if not d.resolved
        )

    @property
    def n_discrepancies(self) -> int:
        """Number of unresolved discrepancies."""
        return len([d for d in self.discrepancies if not d.resolved])

    @property
    def n_errors(self) -> int:
        """Number of error-level discrepancies."""
        return len([
            d for d in self.discrepancies
            if d.severity in [DiscrepancySeverity.ERROR, DiscrepancySeverity.CRITICAL]
            and not d.resolved
        ])

    def summary(self) -> str:
        """Generate summary string."""
        duration = ""
        if self.end_time:
            duration = f" ({(self.end_time - self.start_time).total_seconds():.1f}s)"

        status = "CLEAN" if self.is_clean else "DISCREPANCIES FOUND"

        disc_by_type = {}
        for d in self.discrepancies:
            key = d.discrepancy_type.value
            if key not in disc_by_type:
                disc_by_type[key] = 0
            disc_by_type[key] += 1

        type_lines = "\n".join(
            f"  {k}: {v}" for k, v in disc_by_type.items()
        ) if disc_by_type else "  None"

        return f"""
================================================================================
              RECONCILIATION REPORT
================================================================================
Run ID: {self.reconciliation_id}
Status: {status}
Time: {self.start_time.isoformat()}{duration}

SCOPE
-----
Orders Checked:      {self.orders_checked}
Fills Checked:       {self.fills_checked}

DISCREPANCIES
-------------
Total Found:         {len(self.discrepancies)}
Unresolved:          {self.n_discrepancies}
Errors/Critical:     {self.n_errors}

BY TYPE:
{type_lines}
================================================================================
"""


class FillReconciler:
    """
    Reconcile orders with fills and detect discrepancies.

    Performs:
    - Order-fill matching
    - Quantity verification
    - Price reasonableness checks
    - Duplicate fill detection
    - Position reconciliation

    Example:
        >>> reconciler = FillReconciler()
        >>> result = reconciler.reconcile(orders, fills)
        >>> if not result.is_clean:
        ...     for disc in result.discrepancies:
        ...         print(disc.description)
    """

    def __init__(
        self,
        price_tolerance_bps: float = 10.0,
        quantity_tolerance_pct: float = 1.0,
        commission_tolerance_pct: float = 5.0,
        alert_callback: Optional[Callable[[Discrepancy], None]] = None,
    ):
        """
        Initialize fill reconciler.

        Args:
            price_tolerance_bps: Price tolerance in basis points
            quantity_tolerance_pct: Quantity tolerance percentage
            commission_tolerance_pct: Commission tolerance percentage
            alert_callback: Callback for discrepancy alerts
        """
        self.price_tolerance_bps = price_tolerance_bps
        self.quantity_tolerance_pct = quantity_tolerance_pct
        self.commission_tolerance_pct = commission_tolerance_pct
        self.alert_callback = alert_callback

        # Tracking
        self._seen_fill_ids: Set[str] = set()
        self._discrepancy_counter = 0

    def reconcile(
        self,
        orders: List[Order],
        fills: List[Fill],
        broker_positions: Optional[Dict[str, float]] = None,
    ) -> ReconciliationResult:
        """
        Perform full reconciliation.

        Args:
            orders: List of orders to reconcile
            fills: List of fills to match
            broker_positions: Optional broker positions for position recon

        Returns:
            ReconciliationResult with any discrepancies
        """
        import uuid
        result = ReconciliationResult(
            reconciliation_id=str(uuid.uuid4())[:8],
            orders_checked=len(orders),
            fills_checked=len(fills),
        )

        # Build order lookup
        orders_by_id = {o.order_id: o for o in orders}
        orders_by_broker_id = {
            o.broker_order_id: o for o in orders if o.broker_order_id
        }

        # Check for duplicate fills
        self._check_duplicate_fills(fills, result)

        # Match fills to orders
        unmatched_fills = []
        for fill in fills:
            order = orders_by_id.get(fill.order_id)
            if order is None and fill.order_id:
                # Try broker order ID
                order = orders_by_broker_id.get(fill.order_id)

            if order is None:
                unmatched_fills.append(fill)
            else:
                # Check fill against order
                self._check_fill(order, fill, result)

        # Report orphan fills
        for fill in unmatched_fills:
            disc = self._create_discrepancy(
                DiscrepancyType.ORPHAN_FILL,
                DiscrepancySeverity.ERROR,
                fill_id=fill.fill_id,
                description=f"Fill {fill.fill_id} has no matching order",
            )
            result.discrepancies.append(disc)

        # Check for orders with expected fills but none received
        for order in orders:
            if order.status == OrderStatus.FILLED and not order.fills:
                disc = self._create_discrepancy(
                    DiscrepancyType.MISSING_FILL,
                    DiscrepancySeverity.ERROR,
                    order_id=order.order_id,
                    description=f"Order {order.order_id} marked filled but no fills",
                )
                result.discrepancies.append(disc)

            # Check quantity mismatch
            if order.filled_quantity > 0:
                if abs(order.filled_quantity - sum(f.quantity for f in order.fills)) > 0.01:
                    disc = self._create_discrepancy(
                        DiscrepancyType.QUANTITY_MISMATCH,
                        DiscrepancySeverity.WARNING,
                        order_id=order.order_id,
                        expected_value=order.filled_quantity,
                        actual_value=sum(f.quantity for f in order.fills),
                        description=f"Order fill quantity doesn't match sum of fills",
                    )
                    result.discrepancies.append(disc)

        # Position reconciliation
        if broker_positions:
            self._reconcile_positions(orders, broker_positions, result)

        result.end_time = datetime.utcnow()

        # Alert on critical discrepancies
        for disc in result.discrepancies:
            if disc.severity in [DiscrepancySeverity.ERROR, DiscrepancySeverity.CRITICAL]:
                self._alert(disc)

        logger.info(
            f"Reconciliation complete: {len(result.discrepancies)} discrepancies"
        )

        return result

    def reconcile_order(
        self,
        order: Order,
        broker_fills: List[Fill],
    ) -> List[Discrepancy]:
        """
        Reconcile a single order with broker fills.

        Args:
            order: Order to reconcile
            broker_fills: Fills from broker for this order

        Returns:
            List of discrepancies found
        """
        discrepancies = []

        # Check each broker fill
        order_fills = {f.fill_id: f for f in order.fills}

        for broker_fill in broker_fills:
            if broker_fill.fill_id in order_fills:
                our_fill = order_fills[broker_fill.fill_id]
                # Compare fills
                if abs(our_fill.quantity - broker_fill.quantity) > 0.01:
                    disc = self._create_discrepancy(
                        DiscrepancyType.QUANTITY_MISMATCH,
                        DiscrepancySeverity.ERROR,
                        order_id=order.order_id,
                        fill_id=broker_fill.fill_id,
                        expected_value=broker_fill.quantity,
                        actual_value=our_fill.quantity,
                        difference=abs(our_fill.quantity - broker_fill.quantity),
                        description="Fill quantity mismatch with broker",
                    )
                    discrepancies.append(disc)

                # Price check
                if our_fill.price > 0:
                    price_diff_bps = abs(
                        (our_fill.price - broker_fill.price) / our_fill.price
                    ) * 10000
                    if price_diff_bps > self.price_tolerance_bps:
                        disc = self._create_discrepancy(
                            DiscrepancyType.PRICE_MISMATCH,
                            DiscrepancySeverity.WARNING,
                            order_id=order.order_id,
                            fill_id=broker_fill.fill_id,
                            expected_value=broker_fill.price,
                            actual_value=our_fill.price,
                            difference=price_diff_bps,
                            description=f"Fill price differs by {price_diff_bps:.1f} bps",
                        )
                        discrepancies.append(disc)
            else:
                # We don't have this fill
                disc = self._create_discrepancy(
                    DiscrepancyType.MISSING_FILL,
                    DiscrepancySeverity.ERROR,
                    order_id=order.order_id,
                    fill_id=broker_fill.fill_id,
                    description=f"Missing fill {broker_fill.fill_id} from broker",
                )
                discrepancies.append(disc)

        # Check for fills we have that broker doesn't
        broker_fill_ids = {f.fill_id for f in broker_fills}
        for our_fill in order.fills:
            if our_fill.fill_id not in broker_fill_ids:
                disc = self._create_discrepancy(
                    DiscrepancyType.ORPHAN_FILL,
                    DiscrepancySeverity.WARNING,
                    order_id=order.order_id,
                    fill_id=our_fill.fill_id,
                    description=f"Fill {our_fill.fill_id} not in broker records",
                )
                discrepancies.append(disc)

        return discrepancies

    def _check_fill(
        self,
        order: Order,
        fill: Fill,
        result: ReconciliationResult,
    ) -> None:
        """Check a single fill against its order."""
        # Price reasonableness
        if order.price and fill.price > 0:
            price_diff_bps = abs(
                (fill.price - order.price) / order.price
            ) * 10000
            if price_diff_bps > self.price_tolerance_bps * 10:  # Very large deviation
                disc = self._create_discrepancy(
                    DiscrepancyType.PRICE_MISMATCH,
                    DiscrepancySeverity.WARNING,
                    order_id=order.order_id,
                    fill_id=fill.fill_id,
                    expected_value=order.price,
                    actual_value=fill.price,
                    difference=price_diff_bps,
                    description=f"Fill price differs {price_diff_bps:.0f} bps from limit",
                )
                result.discrepancies.append(disc)

    def _check_duplicate_fills(
        self,
        fills: List[Fill],
        result: ReconciliationResult,
    ) -> None:
        """Check for duplicate fills."""
        seen: Set[str] = set()

        for fill in fills:
            if fill.fill_id in seen:
                disc = self._create_discrepancy(
                    DiscrepancyType.DUPLICATE_FILL,
                    DiscrepancySeverity.ERROR,
                    fill_id=fill.fill_id,
                    description=f"Duplicate fill ID: {fill.fill_id}",
                )
                result.discrepancies.append(disc)
            else:
                seen.add(fill.fill_id)

            # Also check against historical
            if fill.fill_id in self._seen_fill_ids:
                disc = self._create_discrepancy(
                    DiscrepancyType.DUPLICATE_FILL,
                    DiscrepancySeverity.WARNING,
                    fill_id=fill.fill_id,
                    description=f"Fill {fill.fill_id} seen in previous reconciliation",
                )
                result.discrepancies.append(disc)
            else:
                self._seen_fill_ids.add(fill.fill_id)

    def _reconcile_positions(
        self,
        orders: List[Order],
        broker_positions: Dict[str, float],
        result: ReconciliationResult,
    ) -> None:
        """Reconcile calculated positions with broker positions."""
        # Calculate our positions from fills
        our_positions: Dict[str, float] = {}

        for order in orders:
            if order.filled_quantity > 0:
                symbol = order.symbol
                if symbol not in our_positions:
                    our_positions[symbol] = 0.0

                from .order import OrderSide
                if order.side in [OrderSide.BUY, OrderSide.COVER]:
                    our_positions[symbol] += order.filled_quantity
                else:
                    our_positions[symbol] -= order.filled_quantity

        # Compare with broker
        all_symbols = set(our_positions.keys()) | set(broker_positions.keys())

        for symbol in all_symbols:
            our_qty = our_positions.get(symbol, 0.0)
            broker_qty = broker_positions.get(symbol, 0.0)

            if abs(our_qty - broker_qty) > 0.01:
                disc = self._create_discrepancy(
                    DiscrepancyType.POSITION_MISMATCH,
                    DiscrepancySeverity.ERROR,
                    expected_value=broker_qty,
                    actual_value=our_qty,
                    difference=abs(our_qty - broker_qty),
                    description=f"Position mismatch for {symbol}: "
                                f"our={our_qty:.0f}, broker={broker_qty:.0f}",
                )
                result.discrepancies.append(disc)

    def _create_discrepancy(
        self,
        disc_type: DiscrepancyType,
        severity: DiscrepancySeverity,
        **kwargs,
    ) -> Discrepancy:
        """Create a new discrepancy."""
        self._discrepancy_counter += 1
        return Discrepancy(
            discrepancy_id=f"DISC_{self._discrepancy_counter:06d}",
            discrepancy_type=disc_type,
            severity=severity,
            **kwargs,
        )

    def _alert(self, discrepancy: Discrepancy) -> None:
        """Send alert for discrepancy."""
        if self.alert_callback:
            try:
                self.alert_callback(discrepancy)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(
            f"Reconciliation {discrepancy.severity.value}: "
            f"{discrepancy.description}"
        )


class AuditTrail:
    """
    Maintain audit trail for regulatory compliance.

    Records all order and fill activity for reporting.
    """

    def __init__(self):
        """Initialize audit trail."""
        self.entries: List[Dict[str, Any]] = []
        self._lock = __import__("threading").RLock()

    def record_order(
        self,
        order: Order,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record order activity.

        Args:
            order: Order involved
            action: Action taken (CREATED, SUBMITTED, CANCELLED, etc.)
            details: Additional details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "ORDER",
            "action": action,
            "order_id": order.order_id,
            "broker_order_id": order.broker_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "price": order.price,
            "status": order.status.value,
            "details": details or {},
        }

        with self._lock:
            self.entries.append(entry)

        logger.debug(f"Audit: {action} order {order.order_id}")

    def record_fill(
        self,
        fill: Fill,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record fill activity.

        Args:
            fill: Fill to record
            details: Additional details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "FILL",
            "action": "EXECUTED",
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "quantity": fill.quantity,
            "price": fill.price,
            "venue": fill.venue,
            "commission": fill.commission,
            "details": details or {},
        }

        with self._lock:
            self.entries.append(entry)

        logger.debug(f"Audit: fill {fill.fill_id}")

    def record_event(
        self,
        event_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record general event.

        Args:
            event_type: Type of event
            description: Event description
            details: Additional details
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "action": "EVENT",
            "description": description,
            "details": details or {},
        }

        with self._lock:
            self.entries.append(entry)

    def get_entries(
        self,
        since: Optional[datetime] = None,
        entry_type: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get audit entries with optional filtering.

        Args:
            since: Only entries after this time
            entry_type: Filter by type (ORDER, FILL, etc.)
            order_id: Filter by order ID

        Returns:
            Filtered list of entries
        """
        with self._lock:
            entries = list(self.entries)

        if since:
            since_str = since.isoformat()
            entries = [e for e in entries if e["timestamp"] >= since_str]

        if entry_type:
            entries = [e for e in entries if e.get("type") == entry_type]

        if order_id:
            entries = [e for e in entries if e.get("order_id") == order_id]

        return entries

    def export(self, filepath: str) -> None:
        """
        Export audit trail to file.

        Args:
            filepath: Output file path
        """
        import json

        with self._lock:
            entries = list(self.entries)

        with open(filepath, "w") as f:
            json.dump(entries, f, indent=2)

        logger.info(f"Exported {len(entries)} audit entries to {filepath}")
