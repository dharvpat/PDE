"""
Execution algorithms for minimizing market impact.

Provides:
    - TWAPExecutor: Time-Weighted Average Price
    - VWAPExecutor: Volume-Weighted Average Price
    - IcebergExecutor: Iceberg/reserve orders
    - POVExecutor: Percentage of Volume

Reference:
    - Almgren & Chriss (2001): Optimal execution
    - Kissell & Glantz (2003): Optimal trading strategies
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .order import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Types of execution algorithms."""

    TWAP = "TWAP"  # Time-weighted average price
    VWAP = "VWAP"  # Volume-weighted average price
    ICEBERG = "ICEBERG"  # Iceberg/reserve order
    POV = "POV"  # Percentage of volume
    IS = "IS"  # Implementation shortfall
    ARRIVAL = "ARRIVAL"  # Arrival price


@dataclass
class ExecutionSlice:
    """
    A single slice of an algorithmic execution.

    Attributes:
        slice_id: Unique slice identifier
        parent_order_id: Parent order ID
        sequence: Slice sequence number
        quantity: Slice quantity
        scheduled_time: When to execute this slice
        min_quantity: Minimum acceptable fill
        max_quantity: Maximum to execute
        price_limit: Price limit for slice
        urgency: Urgency level (0=passive, 1=aggressive)
    """

    slice_id: str = ""
    parent_order_id: str = ""
    sequence: int = 0
    quantity: float = 0.0
    scheduled_time: Optional[datetime] = None
    min_quantity: float = 0.0
    max_quantity: float = 0.0
    price_limit: Optional[float] = None
    urgency: float = 0.5

    # Execution state
    is_executed: bool = False
    executed_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_price: float = 0.0
    child_order_id: Optional[str] = None

    @property
    def remaining(self) -> float:
        """Remaining quantity to execute."""
        return max(0, self.quantity - self.filled_quantity)

    @property
    def fill_rate(self) -> float:
        """Fill rate as percentage."""
        if self.quantity > 0:
            return self.filled_quantity / self.quantity
        return 0.0


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for an algorithm.

    Contains all slices and scheduling information.
    """

    parent_order: Order
    algorithm: ExecutionAlgorithm
    slices: List[ExecutionSlice] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # State
    is_active: bool = True
    is_complete: bool = False

    @property
    def n_slices(self) -> int:
        """Number of slices."""
        return len(self.slices)

    @property
    def total_quantity(self) -> float:
        """Total quantity across all slices."""
        return sum(s.quantity for s in self.slices)

    @property
    def filled_quantity(self) -> float:
        """Total filled quantity."""
        return sum(s.filled_quantity for s in self.slices)

    @property
    def completion_rate(self) -> float:
        """Completion rate as percentage."""
        total = self.total_quantity
        if total > 0:
            return self.filled_quantity / total
        return 0.0

    @property
    def pending_slices(self) -> List[ExecutionSlice]:
        """Slices that haven't been executed."""
        return [s for s in self.slices if not s.is_executed]

    @property
    def next_slice(self) -> Optional[ExecutionSlice]:
        """Get next slice to execute."""
        pending = self.pending_slices
        if pending:
            return min(pending, key=lambda s: s.sequence)
        return None

    @property
    def avg_fill_price(self) -> float:
        """Volume-weighted average fill price."""
        total_value = sum(s.filled_quantity * s.avg_price for s in self.slices)
        total_qty = sum(s.filled_quantity for s in self.slices)
        if total_qty > 0:
            return total_value / total_qty
        return 0.0


class BaseExecutor(ABC):
    """
    Base class for execution algorithms.

    Subclasses implement specific algorithms (TWAP, VWAP, etc.)
    """

    def __init__(self, algorithm: ExecutionAlgorithm):
        """
        Initialize executor.

        Args:
            algorithm: Algorithm type
        """
        self.algorithm = algorithm

    @abstractmethod
    def create_plan(self, order: Order, **params) -> ExecutionPlan:
        """
        Create execution plan for an order.

        Args:
            order: Order to execute
            **params: Algorithm-specific parameters

        Returns:
            ExecutionPlan with slices
        """
        pass

    @abstractmethod
    def get_next_slice(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, Any],
    ) -> Optional[ExecutionSlice]:
        """
        Get next slice to execute based on market conditions.

        Args:
            plan: Execution plan
            market_data: Current market data

        Returns:
            Next slice or None if nothing to execute
        """
        pass


class TWAPExecutor(BaseExecutor):
    """
    Time-Weighted Average Price (TWAP) execution.

    Slices order into equal pieces executed at regular intervals.
    Minimizes market timing risk by spreading execution over time.

    Example:
        >>> twap = TWAPExecutor()
        >>> plan = twap.create_plan(
        ...     order,
        ...     duration_minutes=60,
        ...     n_slices=12,  # One slice every 5 minutes
        ... )
    """

    def __init__(self):
        """Initialize TWAP executor."""
        super().__init__(ExecutionAlgorithm.TWAP)

    def create_plan(
        self,
        order: Order,
        duration_minutes: int = 60,
        n_slices: int = 12,
        start_time: Optional[datetime] = None,
        randomize: bool = True,
        randomize_pct: float = 0.2,
    ) -> ExecutionPlan:
        """
        Create TWAP execution plan.

        Args:
            order: Parent order
            duration_minutes: Total execution duration
            n_slices: Number of slices
            start_time: When to start execution
            randomize: Add randomness to slice sizes (reduce signaling)
            randomize_pct: Maximum randomization percentage

        Returns:
            ExecutionPlan with time-scheduled slices
        """
        start_time = start_time or datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Calculate slice sizes
        if randomize:
            # Random sizes around average (avoid signaling)
            base_size = order.quantity / n_slices
            rand_factors = 1 + np.random.uniform(
                -randomize_pct, randomize_pct, n_slices
            )
            sizes = base_size * rand_factors
            # Normalize to match total quantity
            sizes = sizes * (order.quantity / sizes.sum())
        else:
            sizes = np.full(n_slices, order.quantity / n_slices)

        # Calculate schedule
        interval = timedelta(minutes=duration_minutes / n_slices)

        slices = []
        for i in range(n_slices):
            scheduled_time = start_time + (i * interval)

            slice_obj = ExecutionSlice(
                slice_id=f"{order.order_id}_slice_{i}",
                parent_order_id=order.order_id,
                sequence=i,
                quantity=float(sizes[i]),
                scheduled_time=scheduled_time,
                min_quantity=float(sizes[i]) * 0.5,  # Allow partial
                max_quantity=float(sizes[i]) * 1.2,  # Allow some overfill
                price_limit=order.price,
                urgency=0.5,
            )
            slices.append(slice_obj)

        plan = ExecutionPlan(
            parent_order=order,
            algorithm=self.algorithm,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            parameters={
                "duration_minutes": duration_minutes,
                "n_slices": n_slices,
                "randomize": randomize,
                "randomize_pct": randomize_pct,
            },
        )

        logger.info(
            f"Created TWAP plan: {n_slices} slices over {duration_minutes} min"
        )

        return plan

    def get_next_slice(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, Any],
    ) -> Optional[ExecutionSlice]:
        """
        Get next TWAP slice if scheduled time has passed.

        Args:
            plan: Execution plan
            market_data: Current market data (not heavily used for TWAP)

        Returns:
            Next slice if ready, None otherwise
        """
        now = datetime.utcnow()

        for slice_obj in plan.slices:
            if slice_obj.is_executed:
                continue

            if slice_obj.scheduled_time and slice_obj.scheduled_time <= now:
                return slice_obj

        return None


class VWAPExecutor(BaseExecutor):
    """
    Volume-Weighted Average Price (VWAP) execution.

    Distributes order to match historical/predicted volume profile.
    Aims to achieve VWAP benchmark.

    Example:
        >>> vwap = VWAPExecutor()
        >>> plan = vwap.create_plan(
        ...     order,
        ...     volume_profile=[0.05, 0.08, 0.12, ...],  # % of daily volume by period
        ... )
    """

    def __init__(self):
        """Initialize VWAP executor."""
        super().__init__(ExecutionAlgorithm.VWAP)

    def create_plan(
        self,
        order: Order,
        duration_minutes: int = 390,  # Full trading day
        n_slices: int = 78,  # 5-minute buckets
        volume_profile: Optional[List[float]] = None,
        start_time: Optional[datetime] = None,
        participation_rate: float = 0.05,  # 5% of volume
    ) -> ExecutionPlan:
        """
        Create VWAP execution plan.

        Args:
            order: Parent order
            duration_minutes: Execution duration
            n_slices: Number of time buckets
            volume_profile: Expected volume by bucket (should sum to 1.0)
            start_time: Start time
            participation_rate: Target % of market volume

        Returns:
            ExecutionPlan with volume-weighted slices
        """
        start_time = start_time or datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Use default U-shaped volume profile if not provided
        if volume_profile is None:
            volume_profile = self._default_volume_profile(n_slices)

        # Normalize profile
        profile_sum = sum(volume_profile)
        if profile_sum > 0:
            volume_profile = [v / profile_sum for v in volume_profile]

        # Calculate sizes based on volume profile
        sizes = [order.quantity * v for v in volume_profile]

        # Calculate schedule
        interval = timedelta(minutes=duration_minutes / n_slices)

        slices = []
        for i in range(n_slices):
            if sizes[i] < 1:  # Skip very small slices
                continue

            scheduled_time = start_time + (i * interval)

            slice_obj = ExecutionSlice(
                slice_id=f"{order.order_id}_vwap_{i}",
                parent_order_id=order.order_id,
                sequence=i,
                quantity=sizes[i],
                scheduled_time=scheduled_time,
                min_quantity=sizes[i] * 0.3,  # More flexibility
                max_quantity=sizes[i] * 2.0,  # Can go higher on high volume
                price_limit=order.price,
                urgency=0.5 + (0.5 * volume_profile[i] / max(volume_profile)),
            )
            slices.append(slice_obj)

        plan = ExecutionPlan(
            parent_order=order,
            algorithm=self.algorithm,
            slices=slices,
            start_time=start_time,
            end_time=end_time,
            parameters={
                "duration_minutes": duration_minutes,
                "n_slices": n_slices,
                "participation_rate": participation_rate,
                "volume_profile": volume_profile,
            },
        )

        logger.info(
            f"Created VWAP plan: {len(slices)} slices over {duration_minutes} min"
        )

        return plan

    def get_next_slice(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, Any],
    ) -> Optional[ExecutionSlice]:
        """
        Get next VWAP slice, adjusting for actual volume.

        Args:
            plan: Execution plan
            market_data: Current market data with volume info

        Returns:
            Next slice, possibly adjusted for volume
        """
        now = datetime.utcnow()
        participation_rate = plan.parameters.get("participation_rate", 0.05)

        for slice_obj in plan.slices:
            if slice_obj.is_executed:
                continue

            if slice_obj.scheduled_time and slice_obj.scheduled_time <= now:
                # Adjust quantity based on actual volume if available
                actual_volume = market_data.get("period_volume", 0)
                if actual_volume > 0:
                    volume_target = actual_volume * participation_rate
                    # Cap at max_quantity
                    slice_obj.quantity = min(
                        volume_target,
                        slice_obj.max_quantity,
                    )
                    slice_obj.quantity = max(
                        slice_obj.quantity,
                        slice_obj.min_quantity,
                    )

                return slice_obj

        return None

    def _default_volume_profile(self, n_buckets: int) -> List[float]:
        """
        Generate default U-shaped intraday volume profile.

        Higher volume at open and close, lower midday.
        """
        x = np.linspace(0, 1, n_buckets)
        # U-shaped: higher at ends
        profile = 1 + 0.5 * (4 * (x - 0.5) ** 2)
        return list(profile / profile.sum())


class IcebergExecutor(BaseExecutor):
    """
    Iceberg/Reserve order execution.

    Shows only a portion of the order at a time.
    Hides true order size from the market.

    Example:
        >>> iceberg = IcebergExecutor()
        >>> plan = iceberg.create_plan(
        ...     order,
        ...     display_quantity=100,  # Show 100 shares at a time
        ...     reload_threshold=0.5,  # Reload when 50% filled
        ... )
    """

    def __init__(self):
        """Initialize Iceberg executor."""
        super().__init__(ExecutionAlgorithm.ICEBERG)

    def create_plan(
        self,
        order: Order,
        display_quantity: float = 100,
        reload_threshold: float = 0.5,
        min_display: float = 50,
        randomize_display: bool = True,
        randomize_pct: float = 0.2,
    ) -> ExecutionPlan:
        """
        Create Iceberg execution plan.

        Args:
            order: Parent order
            display_quantity: Quantity to display at a time
            reload_threshold: Reload when this fraction is filled
            min_display: Minimum display quantity
            randomize_display: Randomize display size
            randomize_pct: Randomization percentage

        Returns:
            ExecutionPlan with iceberg slices
        """
        # Calculate number of slices needed
        n_slices = int(np.ceil(order.quantity / display_quantity))

        slices = []
        remaining = order.quantity

        for i in range(n_slices):
            # Determine slice size
            if randomize_display:
                rand_factor = 1 + np.random.uniform(-randomize_pct, randomize_pct)
                slice_qty = min(display_quantity * rand_factor, remaining)
            else:
                slice_qty = min(display_quantity, remaining)

            slice_qty = max(slice_qty, min_display)

            slice_obj = ExecutionSlice(
                slice_id=f"{order.order_id}_ice_{i}",
                parent_order_id=order.order_id,
                sequence=i,
                quantity=slice_qty,
                scheduled_time=None,  # Triggered by fill, not time
                min_quantity=slice_qty * reload_threshold,
                max_quantity=slice_qty,
                price_limit=order.price,
                urgency=0.3,  # Passive by default
            )
            slices.append(slice_obj)

            remaining -= slice_qty
            if remaining <= 0:
                break

        plan = ExecutionPlan(
            parent_order=order,
            algorithm=self.algorithm,
            slices=slices,
            parameters={
                "display_quantity": display_quantity,
                "reload_threshold": reload_threshold,
                "min_display": min_display,
                "randomize_display": randomize_display,
            },
        )

        logger.info(
            f"Created Iceberg plan: {len(slices)} slices, "
            f"display={display_quantity}"
        )

        return plan

    def get_next_slice(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, Any],
    ) -> Optional[ExecutionSlice]:
        """
        Get next iceberg slice when current is sufficiently filled.

        Args:
            plan: Execution plan
            market_data: Current market data

        Returns:
            Next slice when ready
        """
        reload_threshold = plan.parameters.get("reload_threshold", 0.5)

        for i, slice_obj in enumerate(plan.slices):
            if slice_obj.is_executed:
                continue

            # First unexecuted slice
            if i == 0:
                return slice_obj

            # Check if previous slice is sufficiently filled
            prev_slice = plan.slices[i - 1]
            if prev_slice.fill_rate >= reload_threshold:
                return slice_obj

            # Previous slice not ready, don't return anything
            return None

        return None


class POVExecutor(BaseExecutor):
    """
    Percentage of Volume (POV) execution.

    Participates in a fixed percentage of market volume.
    Adapts to actual market activity.

    Example:
        >>> pov = POVExecutor()
        >>> plan = pov.create_plan(
        ...     order,
        ...     target_participation=0.10,  # 10% of volume
        ...     max_participation=0.20,  # Cap at 20%
        ... )
    """

    def __init__(self):
        """Initialize POV executor."""
        super().__init__(ExecutionAlgorithm.POV)

    def create_plan(
        self,
        order: Order,
        target_participation: float = 0.10,
        max_participation: float = 0.25,
        min_participation: float = 0.02,
        check_interval_seconds: int = 30,
        duration_minutes: int = 390,
    ) -> ExecutionPlan:
        """
        Create POV execution plan.

        Args:
            order: Parent order
            target_participation: Target % of market volume
            max_participation: Maximum participation rate
            min_participation: Minimum participation rate
            check_interval_seconds: How often to check volume
            duration_minutes: Maximum duration

        Returns:
            ExecutionPlan
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # POV creates a single "meta-slice" that adapts dynamically
        # Actual slice generation happens at runtime
        plan = ExecutionPlan(
            parent_order=order,
            algorithm=self.algorithm,
            slices=[],  # Slices created dynamically
            start_time=start_time,
            end_time=end_time,
            parameters={
                "target_participation": target_participation,
                "max_participation": max_participation,
                "min_participation": min_participation,
                "check_interval_seconds": check_interval_seconds,
                "last_check_time": None,
                "cumulative_volume": 0,
            },
        )

        logger.info(
            f"Created POV plan: target {target_participation:.1%} participation"
        )

        return plan

    def get_next_slice(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, Any],
    ) -> Optional[ExecutionSlice]:
        """
        Generate next POV slice based on market volume.

        Args:
            plan: Execution plan
            market_data: Must include 'period_volume' key

        Returns:
            Dynamically sized slice
        """
        now = datetime.utcnow()
        interval = plan.parameters.get("check_interval_seconds", 30)
        last_check = plan.parameters.get("last_check_time")

        # Check if enough time has passed
        if last_check and (now - last_check).total_seconds() < interval:
            return None

        # Get market volume since last check
        period_volume = market_data.get("period_volume", 0)
        if period_volume <= 0:
            return None

        # Calculate our participation
        target_rate = plan.parameters.get("target_participation", 0.10)
        max_rate = plan.parameters.get("max_participation", 0.25)
        min_rate = plan.parameters.get("min_participation", 0.02)

        # Remaining quantity to execute
        remaining = plan.parent_order.quantity - plan.filled_quantity

        if remaining <= 0:
            plan.is_complete = True
            return None

        # Calculate slice size
        target_qty = period_volume * target_rate
        target_qty = min(target_qty, remaining)  # Don't exceed order
        target_qty = max(target_qty, period_volume * min_rate)  # Min participation
        target_qty = min(target_qty, period_volume * max_rate)  # Max participation

        if target_qty < 1:
            return None

        # Create slice
        slice_num = len(plan.slices)
        slice_obj = ExecutionSlice(
            slice_id=f"{plan.parent_order.order_id}_pov_{slice_num}",
            parent_order_id=plan.parent_order.order_id,
            sequence=slice_num,
            quantity=target_qty,
            scheduled_time=now,
            min_quantity=target_qty * 0.5,
            max_quantity=target_qty * 1.5,
            price_limit=plan.parent_order.price,
            urgency=0.5,
        )

        plan.slices.append(slice_obj)
        plan.parameters["last_check_time"] = now
        plan.parameters["cumulative_volume"] = (
            plan.parameters.get("cumulative_volume", 0) + period_volume
        )

        return slice_obj


class ExecutionAlgorithmFactory:
    """
    Factory for creating execution algorithms.

    Example:
        >>> factory = ExecutionAlgorithmFactory()
        >>> executor = factory.create(ExecutionAlgorithm.TWAP)
        >>> plan = executor.create_plan(order, duration_minutes=60)
    """

    _executors = {
        ExecutionAlgorithm.TWAP: TWAPExecutor,
        ExecutionAlgorithm.VWAP: VWAPExecutor,
        ExecutionAlgorithm.ICEBERG: IcebergExecutor,
        ExecutionAlgorithm.POV: POVExecutor,
    }

    @classmethod
    def create(cls, algorithm: ExecutionAlgorithm) -> BaseExecutor:
        """
        Create an executor for the specified algorithm.

        Args:
            algorithm: Algorithm type

        Returns:
            Executor instance
        """
        executor_cls = cls._executors.get(algorithm)
        if executor_cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        return executor_cls()

    @classmethod
    def available_algorithms(cls) -> List[ExecutionAlgorithm]:
        """Get list of available algorithms."""
        return list(cls._executors.keys())
