"""
Execution module.

Handles:
- Order generation and routing
- Transaction cost modeling
- Execution quality monitoring
- Broker integration
- Emergency controls
"""

from .order import (
    Order,
    Fill,
    OrderStatus,
    OrderType,
    OrderSide,
    TimeInForce,
    OrderCapacity,
)

from .order_manager import (
    OrderManager,
    OrderValidator,
    ValidationResult,
    OrderEvent,
    VALID_TRANSITIONS,
)

from .routing import (
    SmartOrderRouter,
    Venue,
    VenueType,
    VenueScorer,
    RoutingDecision,
    RoutingPlan,
    RoutingStrategy,
)

from .algorithms import (
    BaseExecutor,
    TWAPExecutor,
    VWAPExecutor,
    IcebergExecutor,
    POVExecutor,
    ExecutionSlice,
    ExecutionPlan,
    ExecutionAlgorithm,
    ExecutionAlgorithmFactory,
)

from .tca import (
    TCAAnalyzer,
    TCAResult,
    TCAReportGenerator,
    CostComponent,
)

from .broker import (
    BrokerGateway,
    SimulatedBroker,
    AccountInfo,
    BrokerPosition,
    BrokerConnectionManager,
)

from .reconciliation import (
    FillReconciler,
    Discrepancy,
    DiscrepancyType,
    ReconciliationResult,
    AuditTrail,
)

from .emergency import (
    EmergencyController,
    KillSwitch,
    PositionFlattener,
    TradingHoursController,
    CircuitBreaker,
    EmergencyState,
    TriggerType,
    EmergencyEvent,
    TradingHours,
    CircuitBreakerLevel,
)


__all__ = [
    # Order
    "Order",
    "Fill",
    "OrderStatus",
    "OrderType",
    "OrderSide",
    "TimeInForce",
    "OrderCapacity",
    # Order Manager
    "OrderManager",
    "OrderValidator",
    "ValidationResult",
    "OrderEvent",
    "VALID_TRANSITIONS",
    # Routing
    "SmartOrderRouter",
    "Venue",
    "VenueType",
    "VenueScorer",
    "RoutingDecision",
    "RoutingPlan",
    "RoutingStrategy",
    # Algorithms
    "BaseExecutor",
    "TWAPExecutor",
    "VWAPExecutor",
    "IcebergExecutor",
    "POVExecutor",
    "ExecutionSlice",
    "ExecutionPlan",
    "ExecutionAlgorithm",
    "ExecutionAlgorithmFactory",
    # TCA
    "TCAAnalyzer",
    "TCAResult",
    "TCAReportGenerator",
    "CostComponent",
    # Broker
    "BrokerGateway",
    "SimulatedBroker",
    "AccountInfo",
    "BrokerPosition",
    "BrokerConnectionManager",
    # Reconciliation
    "FillReconciler",
    "Discrepancy",
    "DiscrepancyType",
    "ReconciliationResult",
    "AuditTrail",
    # Emergency
    "EmergencyController",
    "KillSwitch",
    "PositionFlattener",
    "TradingHoursController",
    "CircuitBreaker",
    "EmergencyState",
    "TriggerType",
    "EmergencyEvent",
    "TradingHours",
    "CircuitBreakerLevel",
]
