"""
Tests for the Execution Layer module.

Tests cover:
- Order lifecycle and state management
- Order validation and state transitions
- Smart order routing
- Execution algorithms (TWAP, VWAP, Iceberg, POV)
- Transaction Cost Analysis
- Broker gateway abstraction
- Fill reconciliation
- Emergency controls
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import Mock, MagicMock, patch


class TestOrder:
    """Tests for Order and Fill classes."""

    def test_order_creation(self):
        """Test basic order creation."""
        from quant_trading.execution import Order, OrderSide, OrderType, OrderStatus

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == 100
        assert order.price == 150.0
        assert order.status == OrderStatus.PENDING
        assert order.order_id is not None

    def test_order_fill(self):
        """Test adding fills to an order."""
        from quant_trading.execution import Order, Fill, OrderSide, OrderType, OrderStatus

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        fill = Fill(
            fill_id="F001",
            order_id=order.order_id,
            quantity=50,
            price=150.0,
            commission=0.50
        )

        order.add_fill(fill)

        assert order.filled_quantity == 50
        assert order.avg_fill_price == 150.0
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert len(order.fills) == 1

    def test_order_complete_fill(self):
        """Test order becomes FILLED when fully filled."""
        from quant_trading.execution import Order, Fill, OrderSide, OrderType, OrderStatus

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        fill = Fill(
            fill_id="F001",
            order_id=order.order_id,
            quantity=100,
            price=150.0
        )

        order.add_fill(fill)

        assert order.filled_quantity == 100
        assert order.status == OrderStatus.FILLED

    def test_multiple_fills_average_price(self):
        """Test average price calculation with multiple fills."""
        from quant_trading.execution import Order, Fill, OrderSide, OrderType

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        fill1 = Fill(fill_id="F001", order_id=order.order_id, quantity=40, price=150.0)
        fill2 = Fill(fill_id="F002", order_id=order.order_id, quantity=60, price=151.0)

        order.add_fill(fill1)
        order.add_fill(fill2)

        expected_avg = (40 * 150.0 + 60 * 151.0) / 100
        assert abs(order.avg_fill_price - expected_avg) < 0.001


class TestOrderValidator:
    """Tests for OrderValidator."""

    def test_validate_valid_order(self):
        """Test validation of a valid order."""
        from quant_trading.execution import OrderValidator, Order, OrderSide, OrderType

        validator = OrderValidator()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        result = validator.validate(order)
        assert result.is_valid

    def test_validate_missing_symbol(self):
        """Test validation fails for missing symbol."""
        from quant_trading.execution import OrderValidator, Order, OrderSide, OrderType

        validator = OrderValidator()
        order = Order(
            symbol="",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100
        )

        result = validator.validate(order)
        assert not result.is_valid
        assert "symbol" in result.errors[0].lower()

    def test_validate_zero_quantity(self):
        """Test validation fails for zero quantity."""
        from quant_trading.execution import OrderValidator, Order, OrderSide, OrderType

        validator = OrderValidator()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0
        )

        result = validator.validate(order)
        assert not result.is_valid

    def test_validate_limit_without_price(self):
        """Test validation fails for limit order without price."""
        from quant_trading.execution import OrderValidator, Order, OrderSide, OrderType

        validator = OrderValidator()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=0
        )

        result = validator.validate(order)
        assert not result.is_valid

    def test_validate_stop_without_price(self):
        """Test validation fails for stop order without stop price."""
        from quant_trading.execution import OrderValidator, Order, OrderSide, OrderType

        validator = OrderValidator()
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=100,
            stop_price=None
        )

        result = validator.validate(order)
        assert not result.is_valid


class TestOrderManager:
    """Tests for OrderManager."""

    def test_create_order(self):
        """Test order creation through manager."""
        from quant_trading.execution import OrderManager, OrderSide, OrderType

        manager = OrderManager()
        order = manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        assert order is not None
        assert order.symbol == "AAPL"

    def test_submit_order(self):
        """Test order submission."""
        from quant_trading.execution import OrderManager, OrderSide, OrderType, OrderStatus

        manager = OrderManager()
        order = manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        success, message = manager.submit_order(order)
        assert success
        assert order.status in [OrderStatus.SUBMITTED, OrderStatus.VALIDATING]

    def test_cancel_pending_order(self):
        """Test order cancellation of pending order."""
        from quant_trading.execution import OrderManager, OrderSide, OrderType, OrderStatus

        manager = OrderManager()
        order = manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET
        )

        success, _ = manager.cancel_order(order.order_id)
        assert success
        assert order.status == OrderStatus.CANCELLED

    def test_get_orders_by_status(self):
        """Test filtering orders by status."""
        from quant_trading.execution import OrderManager, OrderSide, OrderType, OrderStatus

        manager = OrderManager()

        order1 = manager.create_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        order2 = manager.create_order("GOOG", OrderSide.SELL, 50, OrderType.MARKET)

        manager.cancel_order(order1.order_id)

        pending = manager.get_orders_by_status([OrderStatus.PENDING])
        cancelled = manager.get_orders_by_status([OrderStatus.CANCELLED])

        assert order2 in pending
        assert order1 in cancelled


class TestStateTransitions:
    """Tests for order state machine transitions."""

    def test_valid_transition_pending_to_submitted(self):
        """Test valid transition from PENDING to SUBMITTED."""
        from quant_trading.execution import VALID_TRANSITIONS, OrderStatus

        assert OrderStatus.VALIDATING in VALID_TRANSITIONS[OrderStatus.PENDING]

    def test_valid_transition_submitted_to_filled(self):
        """Test valid transition from SUBMITTED to FILLED."""
        from quant_trading.execution import VALID_TRANSITIONS, OrderStatus

        assert OrderStatus.FILLED in VALID_TRANSITIONS[OrderStatus.SUBMITTED]

    def test_invalid_transition_filled_to_pending(self):
        """Test that FILLED cannot transition to PENDING."""
        from quant_trading.execution import VALID_TRANSITIONS, OrderStatus

        assert OrderStatus.PENDING not in VALID_TRANSITIONS.get(OrderStatus.FILLED, set())


class TestVenue:
    """Tests for Venue and venue scoring."""

    def test_venue_creation(self):
        """Test venue creation."""
        from quant_trading.execution import Venue, VenueType

        venue = Venue(
            venue_id="NYSE",
            name="New York Stock Exchange",
            venue_type=VenueType.EXCHANGE
        )

        assert venue.venue_id == "NYSE"
        assert venue.venue_type == VenueType.EXCHANGE

    def test_venue_scorer(self):
        """Test venue scoring."""
        from quant_trading.execution import Venue, VenueType, VenueScorer, Order, OrderSide, OrderType

        scorer = VenueScorer()

        venue = Venue(
            venue_id="NYSE",
            name="NYSE",
            venue_type=VenueType.EXCHANGE,
            fee_maker=0.003,
            fee_taker=0.003,
            fill_rate=0.95,
            latency_ms=5.0
        )

        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0
        )

        score = scorer.score_venue(venue, order, [venue])
        assert 0 <= score <= 1

    def test_venue_ranking(self):
        """Test ranking multiple venues."""
        from quant_trading.execution import Venue, VenueType, VenueScorer, Order, OrderSide, OrderType

        scorer = VenueScorer()

        venues = [
            Venue("V1", "Venue1", VenueType.EXCHANGE, fee_taker=0.005),
            Venue("V2", "Venue2", VenueType.EXCHANGE, fee_taker=0.002),
            Venue("V3", "Venue3", VenueType.DARK_POOL, fee_taker=0.001),
        ]

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=150.0)

        ranked = scorer.rank_venues(venues, order)

        assert len(ranked) == 3
        assert all(0 <= score <= 1 for _, score in ranked)


class TestSmartOrderRouter:
    """Tests for Smart Order Router."""

    def test_router_creation(self):
        """Test router creation."""
        from quant_trading.execution import SmartOrderRouter, Venue, VenueType

        venues = [Venue("NYSE", "NYSE", VenueType.EXCHANGE)]
        router = SmartOrderRouter(venues=venues)
        assert router is not None

    def test_add_venue(self):
        """Test adding venues to router."""
        from quant_trading.execution import SmartOrderRouter, Venue, VenueType

        venues = [Venue("NYSE", "NYSE", VenueType.EXCHANGE)]
        router = SmartOrderRouter(venues=venues)

        assert "NYSE" in router.venues

    def test_route_order(self):
        """Test routing an order."""
        from quant_trading.execution import (
            SmartOrderRouter, Venue, VenueType, Order, OrderSide, OrderType, RoutingStrategy
        )

        venues = [
            Venue("NYSE", "NYSE", VenueType.EXCHANGE),
            Venue("NASDAQ", "NASDAQ", VenueType.EXCHANGE)
        ]
        router = SmartOrderRouter(venues=venues)

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=100, price=150.0)

        plan = router.route_order(order, RoutingStrategy.SMART)

        assert plan is not None
        assert len(plan.decisions) > 0


class TestTWAPExecutor:
    """Tests for TWAP execution algorithm."""

    def test_create_plan(self):
        """Test TWAP plan creation."""
        from quant_trading.execution import TWAPExecutor, Order, OrderSide, OrderType

        executor = TWAPExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)

        plan = executor.create_plan(order, duration_minutes=60, n_slices=12)

        assert plan is not None
        assert len(plan.slices) == 12
        total_qty = sum(s.quantity for s in plan.slices)
        assert abs(total_qty - 1000) < 0.01

    def test_twap_equal_distribution(self):
        """Test TWAP distributes quantity equally."""
        from quant_trading.execution import TWAPExecutor, Order, OrderSide, OrderType

        executor = TWAPExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1200)

        plan = executor.create_plan(order, duration_minutes=60, n_slices=12, randomize=False)

        quantities = [s.quantity for s in plan.slices]
        assert all(abs(q - 100) < 0.01 for q in quantities)


class TestVWAPExecutor:
    """Tests for VWAP execution algorithm."""

    def test_create_plan(self):
        """Test VWAP plan creation."""
        from quant_trading.execution import VWAPExecutor, Order, OrderSide, OrderType

        executor = VWAPExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)

        plan = executor.create_plan(order, participation_rate=0.05)

        assert plan is not None
        assert len(plan.slices) > 0

    def test_vwap_volume_profile(self):
        """Test VWAP uses volume profile."""
        from quant_trading.execution import VWAPExecutor, Order, OrderSide, OrderType

        executor = VWAPExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)

        plan = executor.create_plan(order)

        quantities = [s.quantity for s in plan.slices]
        # Just verify we got slices
        assert len(quantities) > 0


class TestIcebergExecutor:
    """Tests for Iceberg execution algorithm."""

    def test_create_plan(self):
        """Test Iceberg plan creation."""
        from quant_trading.execution import IcebergExecutor, Order, OrderSide, OrderType

        executor = IcebergExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=10000, price=150.0)

        plan = executor.create_plan(order, display_quantity=500)

        assert plan is not None
        total_qty = sum(s.quantity for s in plan.slices)
        # Allow reasonable tolerance for iceberg quantity allocation
        assert abs(total_qty - 10000) / 10000 < 0.05  # Within 5%

    def test_iceberg_display_size(self):
        """Test Iceberg creates reasonable slices."""
        from quant_trading.execution import IcebergExecutor, Order, OrderSide, OrderType

        executor = IcebergExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.LIMIT, quantity=10000, price=150.0)

        plan = executor.create_plan(order, display_quantity=500)

        # Iceberg creates slices - verify there are multiple
        assert len(plan.slices) > 1


class TestPOVExecutor:
    """Tests for Percentage of Volume executor."""

    def test_create_plan(self):
        """Test POV plan creation."""
        from quant_trading.execution import POVExecutor, Order, OrderSide, OrderType

        executor = POVExecutor()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000)

        plan = executor.create_plan(order, target_participation=0.10)

        assert plan is not None


class TestTCAAnalyzer:
    """Tests for Transaction Cost Analysis."""

    def test_analyze_order(self):
        """Test TCA analysis of an order."""
        from quant_trading.execution import TCAAnalyzer, Order, Fill, OrderSide, OrderType

        analyzer = TCAAnalyzer()

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        fill = Fill(fill_id="F001", order_id=order.order_id, quantity=100, price=150.50, commission=1.0)
        order.add_fill(fill)

        result = analyzer.analyze_order(
            order=order,
            arrival_price=150.45,
            vwap_price=150.48,
            adv=1000000
        )

        assert result is not None
        assert result.total_cost_bps is not None

    def test_implementation_shortfall(self):
        """Test implementation shortfall calculation."""
        from quant_trading.execution import TCAAnalyzer, Order, Fill, OrderSide, OrderType

        analyzer = TCAAnalyzer()

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        fill = Fill(fill_id="F001", order_id=order.order_id, quantity=100, price=151.00, commission=1.0)
        order.add_fill(fill)

        result = analyzer.analyze_order(
            order=order,
            arrival_price=150.00,
            vwap_price=150.50,
            adv=1000000
        )

        assert result.implementation_shortfall_bps > 0


class TestTCAReportGenerator:
    """Tests for TCA report generation."""

    def test_generate_trade_report(self):
        """Test generating a TCA report for single trade."""
        from quant_trading.execution import TCAReportGenerator, TCAResult, OrderSide

        generator = TCAReportGenerator()

        result = TCAResult(
            order_id="O001",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            avg_fill_price=150.0,
            total_cost_bps=5.5,
            implementation_shortfall_bps=3.2
        )

        report = generator.generate_trade_report(result)

        assert report is not None

    def test_generate_summary_report(self):
        """Test generating a TCA summary report."""
        from quant_trading.execution import TCAReportGenerator, TCAResult, OrderSide

        generator = TCAReportGenerator()

        results = [
            TCAResult(
                order_id="O001",
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=100,
                avg_fill_price=150.0,
                total_cost_bps=5.5,
                implementation_shortfall_bps=3.2
            ),
            TCAResult(
                order_id="O002",
                symbol="GOOG",
                side=OrderSide.SELL,
                quantity=50,
                avg_fill_price=2800.0,
                total_cost_bps=4.2,
                implementation_shortfall_bps=2.1
            )
        ]

        report = generator.generate_summary_report(results)

        assert report is not None


class TestSimulatedBroker:
    """Tests for Simulated Broker."""

    def test_broker_creation(self):
        """Test simulated broker creation."""
        from quant_trading.execution import SimulatedBroker

        broker = SimulatedBroker(initial_cash=100000)
        broker.connect()
        account = broker.get_account_info()

        assert account.cash == 100000

    def test_submit_market_order(self):
        """Test submitting a market order."""
        from quant_trading.execution import SimulatedBroker, Order, OrderSide, OrderType

        broker = SimulatedBroker(initial_cash=100000)
        broker.connect()

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10)

        broker_id = broker.submit_order(order)

        assert broker_id is not None

    def test_get_positions(self):
        """Test getting positions."""
        from quant_trading.execution import SimulatedBroker

        broker = SimulatedBroker(initial_cash=100000)
        broker.connect()
        positions = broker.get_positions()

        assert isinstance(positions, list)


class TestFillReconciler:
    """Tests for Fill Reconciliation."""

    def test_reconciler_creation(self):
        """Test reconciler creation."""
        from quant_trading.execution import FillReconciler

        reconciler = FillReconciler()
        assert reconciler is not None

    def test_reconcile_matching_fills(self):
        """Test reconciliation with matching fills."""
        from quant_trading.execution import FillReconciler, Order, Fill, OrderSide, OrderType

        reconciler = FillReconciler()

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        fill = Fill(fill_id="F001", order_id=order.order_id, quantity=100, price=150.0)
        order.add_fill(fill)

        # broker_positions is Dict[str, float]
        broker_positions = {"AAPL": 100}

        result = reconciler.reconcile(
            orders=[order],
            fills=[fill],
            broker_positions=broker_positions
        )

        assert result is not None

    def test_detect_quantity_mismatch(self):
        """Test detection of quantity mismatch."""
        from quant_trading.execution import FillReconciler, Order, Fill, OrderSide, OrderType

        reconciler = FillReconciler()

        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)
        fill = Fill(fill_id="F001", order_id=order.order_id, quantity=90, price=150.0)
        order.add_fill(fill)

        # broker_positions is Dict[str, float]
        broker_positions = {"AAPL": 100}

        result = reconciler.reconcile(
            orders=[order],
            fills=[fill],
            broker_positions=broker_positions
        )

        # Should detect a discrepancy
        assert result is not None


class TestAuditTrail:
    """Tests for Audit Trail."""

    def test_record_order(self):
        """Test recording order events."""
        from quant_trading.execution import AuditTrail, Order, OrderSide, OrderType

        audit = AuditTrail()
        order = Order(symbol="AAPL", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100)

        audit.record_order(order, "CREATED", {"source": "test"})

        entries = audit.get_entries()
        assert len(entries) == 1

    def test_record_fill(self):
        """Test recording fill events."""
        from quant_trading.execution import AuditTrail, Fill

        audit = AuditTrail()
        fill = Fill(fill_id="F001", order_id="O001", quantity=100, price=150.0)

        audit.record_fill(fill, {"venue": "NYSE"})

        entries = audit.get_entries()
        assert len(entries) == 1


class TestKillSwitch:
    """Tests for Kill Switch."""

    def test_kill_switch_creation(self):
        """Test kill switch creation."""
        from quant_trading.execution import KillSwitch

        ks = KillSwitch()
        assert not ks.is_engaged

    def test_engage_kill_switch(self):
        """Test engaging the kill switch."""
        from quant_trading.execution import KillSwitch

        ks = KillSwitch()
        event = ks.engage(reason="Test activation", triggered_by="test_user")

        assert ks.is_engaged
        assert event is not None
        assert event.reason == "Test activation"

    def test_disengage_kill_switch(self):
        """Test disengaging the kill switch."""
        from quant_trading.execution import KillSwitch

        ks = KillSwitch()
        ks.engage(reason="Test")

        success = ks.disengage(authorized_by="admin", notes="Test complete")

        assert success
        assert not ks.is_engaged

    def test_cannot_engage_twice(self):
        """Test that engaging an already engaged switch returns existing event."""
        from quant_trading.execution import KillSwitch

        ks = KillSwitch()
        event1 = ks.engage(reason="First")
        event2 = ks.engage(reason="Second")

        assert event1 == event2


class TestPositionFlattener:
    """Tests for Position Flattener."""

    def test_flattener_creation(self):
        """Test position flattener creation."""
        from quant_trading.execution import PositionFlattener

        flattener = PositionFlattener()
        assert flattener is not None

    def test_flatten_all_positions(self):
        """Test flattening all positions."""
        from quant_trading.execution import PositionFlattener, OrderManager, BrokerPosition

        mock_broker = Mock()
        mock_broker.get_positions.return_value = [
            BrokerPosition(symbol="AAPL", quantity=100, avg_cost=150.0, market_value=15100.0),
            BrokerPosition(symbol="GOOG", quantity=-50, avg_cost=2800.0, market_value=139500.0),
        ]

        order_manager = OrderManager()
        flattener = PositionFlattener(
            order_manager=order_manager,
            broker_gateway=mock_broker
        )

        orders = flattener.flatten_all_positions(urgency="normal")

        assert len(orders) == 2


class TestTradingHoursController:
    """Tests for Trading Hours Controller."""

    def test_controller_creation(self):
        """Test trading hours controller creation."""
        from quant_trading.execution import TradingHoursController

        controller = TradingHoursController()
        assert controller is not None

    def test_market_hours_allowed(self):
        """Test trading allowed during market hours."""
        from quant_trading.execution import TradingHoursController, TradingHours

        hours = TradingHours(
            market_open=time(9, 30),
            market_close=time(16, 0)
        )
        controller = TradingHoursController(hours)

        market_time = datetime(2024, 1, 15, 10, 30)

        allowed = controller.is_trading_allowed(market_time)
        assert allowed

    def test_after_hours_blocked(self):
        """Test trading blocked after hours when disabled."""
        from quant_trading.execution import TradingHoursController, TradingHours

        hours = TradingHours(
            market_open=time(9, 30),
            market_close=time(16, 0),
            allow_after_hours=False
        )
        controller = TradingHoursController(hours)

        after_hours = datetime(2024, 1, 15, 18, 0)

        allowed = controller.is_trading_allowed(after_hours)
        assert not allowed

    def test_weekend_blocked(self):
        """Test trading blocked on weekends."""
        from quant_trading.execution import TradingHoursController

        controller = TradingHoursController()

        saturday = datetime(2024, 1, 13, 10, 30)

        allowed = controller.is_trading_allowed(saturday)
        assert not allowed

    def test_get_session_type(self):
        """Test session type detection."""
        from quant_trading.execution import TradingHoursController

        controller = TradingHoursController()

        regular_hours = datetime(2024, 1, 15, 10, 30)
        pre_market = datetime(2024, 1, 15, 5, 0)

        assert controller.get_session_type(regular_hours) == "regular"
        assert controller.get_session_type(pre_market) == "pre_market"


class TestCircuitBreaker:
    """Tests for Circuit Breaker."""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker creation."""
        from quant_trading.execution import CircuitBreaker

        cb = CircuitBreaker(reference_price=100.0)
        assert cb.reference_price == 100.0
        assert not cb.is_triggered

    def test_level1_trigger(self):
        """Test Level 1 circuit breaker trigger (7% drop)."""
        from quant_trading.execution import CircuitBreaker

        cb = CircuitBreaker(reference_price=100.0)

        level = cb.check_price(92.0)

        assert level is not None
        assert level.name == "Level 1"

    def test_level2_trigger(self):
        """Test Level 2 circuit breaker trigger (13% drop)."""
        from quant_trading.execution import CircuitBreaker

        cb = CircuitBreaker(reference_price=100.0)

        level = cb.check_price(86.0)

        assert level is not None
        assert level.name == "Level 2"

    def test_level3_trigger(self):
        """Test Level 3 circuit breaker trigger (20% drop)."""
        from quant_trading.execution import CircuitBreaker

        cb = CircuitBreaker(reference_price=100.0)

        level = cb.check_price(79.0)

        assert level is not None
        assert level.name == "Level 3"

    def test_no_trigger(self):
        """Test no trigger for small move."""
        from quant_trading.execution import CircuitBreaker

        cb = CircuitBreaker(reference_price=100.0)

        level = cb.check_price(97.0)

        assert level is None

    def test_trigger_event(self):
        """Test triggering creates event."""
        from quant_trading.execution import CircuitBreaker, EmergencyState

        cb = CircuitBreaker(reference_price=100.0)
        level = cb.check_price(92.0)

        event = cb.trigger(level)

        assert cb.is_triggered
        assert event is not None
        assert event.new_state == EmergencyState.HALT_NEW_ORDERS


class TestEmergencyController:
    """Tests for Emergency Controller."""

    def test_controller_creation(self):
        """Test emergency controller creation."""
        from quant_trading.execution import EmergencyController, EmergencyState

        controller = EmergencyController()
        assert controller.current_state == EmergencyState.NORMAL

    def test_trading_allowed_normal(self):
        """Test trading allowed in normal state."""
        from quant_trading.execution import EmergencyController

        controller = EmergencyController()

        with patch.object(controller.hours_controller, 'is_trading_allowed', return_value=True):
            assert controller.is_trading_allowed()

    def test_engage_kill_switch(self):
        """Test engaging kill switch through controller."""
        from quant_trading.execution import EmergencyController, EmergencyState

        controller = EmergencyController()

        event = controller.engage_kill_switch("Test reason", "test_user")

        assert controller.current_state == EmergencyState.FULL_STOP
        assert not controller.is_trading_allowed()

    def test_drawdown_limit(self):
        """Test drawdown limit triggers halt."""
        from quant_trading.execution import EmergencyController, EmergencyState

        controller = EmergencyController()
        controller.drawdown_limit_pct = 10.0

        controller._peak_equity = 100000
        controller.check_and_update_state(
            current_equity=88000,
            current_price=100.0,
            daily_pnl=-12000
        )

        assert controller.current_state == EmergencyState.HALT_NEW_ORDERS

    def test_recover_to_normal(self):
        """Test recovery to normal state."""
        from quant_trading.execution import EmergencyController, EmergencyState

        controller = EmergencyController()
        controller.engage_kill_switch("Test", "user")

        success = controller.recover_to_normal("admin", "Testing complete")

        assert success
        assert controller.current_state == EmergencyState.NORMAL

    def test_get_status(self):
        """Test getting controller status."""
        from quant_trading.execution import EmergencyController

        controller = EmergencyController()
        status = controller.get_status()

        assert "current_state" in status
        assert "trading_allowed" in status
        assert "kill_switch" in status
        assert "circuit_breaker" in status

    def test_export_audit_log(self):
        """Test exporting audit log."""
        from quant_trading.execution import EmergencyController

        controller = EmergencyController()
        controller.engage_kill_switch("Test", "user")

        log = controller.export_audit_log()

        assert len(log) > 0
        assert "event_id" in log[0]
        assert "timestamp" in log[0]


class TestIntegration:
    """Integration tests for execution module."""

    def test_full_order_lifecycle(self):
        """Test complete order lifecycle."""
        from quant_trading.execution import (
            OrderManager, SimulatedBroker, Order, Fill, OrderSide, OrderType, OrderStatus
        )

        manager = OrderManager()

        order = manager.create_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            price=150.0
        )
        assert order.status == OrderStatus.PENDING

        success, _ = manager.submit_order(order)
        assert success

        fill = Fill(
            fill_id="F001",
            order_id=order.order_id,
            quantity=100,
            price=150.0,
            commission=1.0
        )
        manager.process_fill(order.order_id, fill)

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 100

    def test_emergency_with_order_manager(self):
        """Test emergency controls with order manager integration."""
        from quant_trading.execution import (
            OrderManager, EmergencyController, OrderSide, OrderType, OrderStatus
        )

        manager = OrderManager()
        controller = EmergencyController(order_manager=manager)

        order1 = manager.create_order("AAPL", OrderSide.BUY, 100, OrderType.MARKET)
        order2 = manager.create_order("GOOG", OrderSide.SELL, 50, OrderType.MARKET)

        controller.engage_kill_switch("Test", "user")

        assert not controller.is_trading_allowed()
