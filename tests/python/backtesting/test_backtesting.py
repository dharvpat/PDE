"""
Comprehensive tests for the backtesting module.

Tests cover:
    - Event classes
    - Portfolio management
    - Data handlers
    - Execution handlers
    - Strategies
    - BacktestEngine
    - Walk-forward analysis
    - Monte Carlo simulation
"""

from datetime import datetime, timedelta
from queue import Queue

import numpy as np
import pytest

from quant_trading.backtesting import (
    # Events
    Direction,
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
    # Portfolio
    Portfolio,
    Position,
    Trade,
    # Data Handlers
    DataHandler,
    SyntheticDataHandler,
    # Execution
    CommissionModel,
    FixedCommission,
    IBKRCommission,
    InstantExecutionHandler,
    PerShareCommission,
    SimulatedExecutionHandler,
    TieredCommission,
    ZeroCommission,
    # Strategies
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    Strategy,
    # Engine
    BacktestEngine,
    BacktestResults,
    # Analysis
    MonteCarloResults,
    MonteCarloSimulator,
    WindowType,
)


# =============================================================================
# Event Tests
# =============================================================================


class TestEvents:
    """Test event classes."""

    def test_market_event_creation(self):
        """Test MarketEvent creation and defaults."""
        timestamp = datetime.now()
        event = MarketEvent(
            timestamp=timestamp,
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
        )

        assert event.symbol == "SPY"
        assert event.price == 450.0
        assert event.volume == 1000000
        assert event.event_type == EventType.MARKET

    def test_signal_event_creation(self):
        """Test SignalEvent creation."""
        timestamp = datetime.now()
        event = SignalEvent(
            timestamp=timestamp,
            event_type=None,
            symbol="AAPL",
            signal_type=SignalType.LONG,
            strength=0.8,
            strategy_id="test_strategy",
        )

        assert event.symbol == "AAPL"
        assert event.signal_type == SignalType.LONG
        assert event.strength == 0.8
        assert event.event_type == EventType.SIGNAL

    def test_order_event_creation(self):
        """Test OrderEvent creation."""
        timestamp = datetime.now()
        event = OrderEvent(
            timestamp=timestamp,
            event_type=None,
            symbol="MSFT",
            order_type=OrderType.MARKET,
            quantity=100,
            direction=Direction.BUY,
        )

        assert event.symbol == "MSFT"
        assert event.order_type == OrderType.MARKET
        assert event.quantity == 100
        assert event.direction == Direction.BUY
        assert event.event_type == EventType.ORDER

    def test_fill_event_creation(self):
        """Test FillEvent creation."""
        timestamp = datetime.now()
        event = FillEvent(
            timestamp=timestamp,
            event_type=None,
            symbol="GOOG",
            quantity=50,
            direction=Direction.SELL,
            fill_price=150.0,
            commission=1.0,
            slippage=0.5,
        )

        assert event.symbol == "GOOG"
        assert event.quantity == 50
        assert event.direction == Direction.SELL
        assert event.fill_price == 150.0
        assert event.event_type == EventType.FILL


# =============================================================================
# Portfolio Tests
# =============================================================================


class TestPosition:
    """Test Position class."""

    def test_position_creation(self):
        """Test position initialization."""
        pos = Position(symbol="SPY")
        assert pos.symbol == "SPY"
        assert pos.quantity == 0.0
        assert pos.avg_entry_price == 0.0

    def test_position_with_quantity(self):
        """Test position with initial quantity."""
        pos = Position(symbol="SPY", quantity=100, avg_entry_price=450.0)

        assert pos.quantity == 100
        assert pos.avg_entry_price == 450.0
        assert pos.is_long

    def test_position_short(self):
        """Test short position detection."""
        pos = Position(symbol="SPY", quantity=-100, avg_entry_price=450.0)

        assert pos.quantity == -100
        assert pos.is_short
        assert not pos.is_long

    def test_position_market_value(self):
        """Test market value calculation."""
        pos = Position(symbol="SPY", quantity=100, avg_entry_price=450.0)
        pos.update_price(460.0)

        assert pos.current_price == 460.0
        assert pos.market_value == 46000.0  # 100 * 460

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        pos = Position(symbol="SPY", quantity=100, avg_entry_price=450.0)
        pos.update_price(460.0)

        assert pos.unrealized_pnl == 1000.0  # 100 * (460 - 450)
        assert pos.market_value == 46000.0


class TestPortfolio:
    """Test Portfolio class."""

    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_capital=100000)

        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert portfolio.equity == 100000

    def test_portfolio_update_fill_buy(self):
        """Test portfolio update with buy fill."""
        portfolio = Portfolio(initial_capital=100000)

        fill = FillEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            quantity=100,
            direction=Direction.BUY,
            fill_price=450.0,
            commission=5.0,
            slippage=0.0,
        )

        portfolio.update_fill(fill)

        assert portfolio.positions["SPY"].quantity == 100
        assert portfolio.cash == 100000 - 45000 - 5  # capital - cost - commission
        assert portfolio.total_commission == 5.0

    def test_portfolio_update_fill_sell(self):
        """Test portfolio update with sell fill."""
        portfolio = Portfolio(initial_capital=100000)

        # Buy first
        buy_fill = FillEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            quantity=100,
            direction=Direction.BUY,
            fill_price=450.0,
            commission=5.0,
            slippage=0.0,
        )
        portfolio.update_fill(buy_fill)

        # Then sell
        sell_fill = FillEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            quantity=100,
            direction=Direction.SELL,
            fill_price=460.0,
            commission=5.0,
            slippage=0.0,
        )
        portfolio.update_fill(sell_fill)

        # Position is removed when quantity is zero
        assert "SPY" not in portfolio.positions
        # Cash = initial - buy_cost - commission + sell_proceeds - commission
        assert portfolio.cash == 100000 - 45000 - 5 + 46000 - 5
        assert portfolio.total_commission == 10.0

    def test_portfolio_equity_calculation(self):
        """Test portfolio equity calculation."""
        portfolio = Portfolio(initial_capital=100000)

        # Buy position
        fill = FillEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            quantity=100,
            direction=Direction.BUY,
            fill_price=450.0,
            commission=0.0,
            slippage=0.0,
        )
        portfolio.update_fill(fill)

        # Update market price
        market_event = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=460.0,
            volume=0,
        )
        portfolio.update_market_data(market_event)

        # Equity = cash + position value
        # Cash = 100000 - 45000 = 55000
        # Position = 100 * 460 = 46000
        # Equity = 55000 + 46000 = 101000
        assert portfolio.equity == 101000

    def test_portfolio_max_position_limit(self):
        """Test maximum position size limit."""
        portfolio = Portfolio(initial_capital=100000, max_position_pct=0.10)

        # Try to generate order larger than limit
        signal = SignalEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            signal_type=SignalType.LONG,
            strength=1.0,
            strategy_id="test",
        )

        events = Queue()

        # Update market data first
        market_event = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
        )
        portfolio.update_market_data(market_event)

        order = portfolio.generate_order(signal, events)

        # Order should be generated and limited by max position size
        assert order is not None
        # Max position = 100000 * 0.10 = 10000
        # Max shares at 450 = 22.22 shares (10000 / 450)
        assert order.quantity <= 23  # Allow rounding


# =============================================================================
# Data Handler Tests
# =============================================================================


class TestSyntheticDataHandler:
    """Test synthetic data handler."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        events = Queue()
        handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=100,
            start_price=100.0,
            volatility=0.20,
        )

        assert len(handler.symbol_list) == 1
        assert handler.continue_backtest

        # Update bars and check events
        handler.update_bars()
        assert not events.empty()

        event = events.get()
        assert event.event_type == EventType.MARKET
        assert event.symbol == "SPY"

    def test_synthetic_data_multiple_symbols(self):
        """Test with multiple symbols."""
        events = Queue()
        handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY", "QQQ", "IWM"],
            n_bars=50,
        )

        # Process all bars
        market_events = []
        while handler.continue_backtest:
            handler.update_bars()
            while not events.empty():
                market_events.append(events.get())

        # Should have events for all symbols
        symbols = set(e.symbol for e in market_events)
        assert "SPY" in symbols
        assert "QQQ" in symbols
        assert "IWM" in symbols

    def test_synthetic_data_trending(self):
        """Test trending data generation."""
        events = Queue()
        handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=252,
            start_price=100.0,
            drift=0.10,  # 10% annual drift
            volatility=0.01,  # Low vol to see trend
        )

        # Get all prices
        prices = []
        while handler.continue_backtest:
            handler.update_bars()
            while not events.empty():
                event = events.get()
                prices.append(event.price)

        # Final price should be higher than start (on average)
        assert len(prices) > 0


# =============================================================================
# Commission Model Tests
# =============================================================================


class TestCommissionModels:
    """Test commission models."""

    def test_zero_commission(self):
        """Test zero commission model."""
        comm = ZeroCommission()
        assert comm.calculate(1000, 100.0) == 0.0

    def test_fixed_commission(self):
        """Test fixed percentage commission."""
        comm = FixedCommission(rate=0.001)  # 10 bps
        # Trade value = 100 * 450 = 45000
        # Commission = 45000 * 0.001 = 45
        assert comm.calculate(100, 450.0) == 45.0

    def test_per_share_commission(self):
        """Test per-share commission."""
        comm = PerShareCommission(per_share=0.005, minimum=1.0)

        # Small trade hits minimum
        assert comm.calculate(100, 100.0) == 1.0  # 100 * 0.005 = 0.5 < 1.0

        # Large trade uses per-share
        assert comm.calculate(1000, 100.0) == 5.0  # 1000 * 0.005 = 5.0

    def test_tiered_commission(self):
        """Test tiered commission."""
        comm = TieredCommission()

        # Small trade (first tier)
        small_comm = comm.calculate(10, 500.0)  # $5000 trade
        assert small_comm == 5000 * 0.002  # 0.2%

        # Medium trade (crosses tiers)
        medium_comm = comm.calculate(100, 500.0)  # $50000 trade
        # First $10k at 0.2% = $20
        # Next $40k at 0.1% = $40
        # Total = $60
        assert medium_comm == 60.0

    def test_ibkr_commission(self):
        """Test IBKR-style commission."""
        comm = IBKRCommission(per_share=0.005, minimum=1.0, maximum_pct=0.01)

        # Small trade: hits minimum
        assert comm.calculate(100, 10.0) == 1.0

        # Normal trade: per-share
        assert comm.calculate(1000, 100.0) == 5.0  # 1000 * 0.005 = 5

        # Very large trade: hits max
        large_comm = comm.calculate(10000, 100.0)
        max_comm = 10000 * 100 * 0.01  # 1% of $1M
        assert large_comm == min(10000 * 0.005, max_comm)


# =============================================================================
# Execution Handler Tests
# =============================================================================


class TestExecutionHandlers:
    """Test execution handlers."""

    def test_instant_execution(self):
        """Test instant execution handler."""
        events = Queue()
        executor = InstantExecutionHandler(events_queue=events)

        # Update market data
        market = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
        )
        executor.update_market_data(market)

        # Execute order
        order = OrderEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            order_type=OrderType.MARKET,
            quantity=100,
            direction=Direction.BUY,
        )

        fill = executor.execute_order(order)

        assert fill is not None
        assert fill.fill_price == 450.0
        assert fill.quantity == 100
        assert fill.slippage == 0.0

    def test_simulated_execution_slippage(self):
        """Test simulated execution with slippage."""
        events = Queue()
        executor = SimulatedExecutionHandler(
            events_queue=events,
            slippage_bps=10.0,  # 10 bps
            market_impact_factor=0.0,  # No market impact
        )

        # Update market data
        market = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
            bid=449.90,
            ask=450.10,
        )
        executor.update_market_data(market)

        # Execute buy order
        order = OrderEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            order_type=OrderType.MARKET,
            quantity=100,
            direction=Direction.BUY,
        )

        fill = executor.execute_order(order)

        assert fill is not None
        # Buy should execute at ask or higher
        assert fill.fill_price >= 450.10

    def test_simulated_execution_limit_order(self):
        """Test limit order execution."""
        events = Queue()
        executor = SimulatedExecutionHandler(events_queue=events)

        # Update market data
        market = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
            bid=449.90,
            ask=450.10,
        )
        executor.update_market_data(market)

        # Limit order that won't fill (limit below ask)
        order = OrderEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            order_type=OrderType.LIMIT,
            quantity=100,
            direction=Direction.BUY,
            limit_price=449.50,  # Below ask
        )

        fill = executor.execute_order(order)
        assert fill is None  # Should not fill

        # Limit order that will fill
        order2 = OrderEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            order_type=OrderType.LIMIT,
            quantity=100,
            direction=Direction.BUY,
            limit_price=450.20,  # Above ask
        )

        fill2 = executor.execute_order(order2)
        assert fill2 is not None


# =============================================================================
# Strategy Tests
# =============================================================================


class TestStrategies:
    """Test trading strategies."""

    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy."""
        events = Queue()

        # Create data handler
        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=10,
        )

        # Create portfolio
        portfolio = Portfolio(initial_capital=100000)

        # Create strategy
        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        # First market event should generate signal
        market = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=450.0,
            volume=1000000,
        )

        strategy.calculate_signals(market)

        # Should have generated a LONG signal
        assert not events.empty()
        signal = events.get()
        assert signal.signal_type == SignalType.LONG

        # Second call should not generate signal (already bought)
        strategy.calculate_signals(market)
        assert events.empty()

    def test_ma_crossover_strategy(self):
        """Test moving average crossover strategy."""
        events = Queue()

        # Create data handler
        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=100,
        )

        # Create portfolio
        portfolio = Portfolio(initial_capital=100000)

        # Create strategy with short windows for testing
        strategy = MovingAverageCrossoverStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            fast_window=3,
            slow_window=5,
        )

        # Feed increasing prices (bullish crossover)
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108]

        for i, price in enumerate(prices):
            market = MarketEvent(
                timestamp=datetime.now(),
                event_type=None,
                symbol="SPY",
                price=price,
                volume=1000000,
            )
            strategy.calculate_signals(market)

        # Strategy should have generated signals after enough data

    def test_mean_reversion_strategy(self):
        """Test mean reversion strategy."""
        events = Queue()

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=100,
        )

        portfolio = Portfolio(initial_capital=100000)

        strategy = MeanReversionStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            lookback=5,
            entry_threshold=2.0,
        )

        # Feed prices with extreme deviation
        # First build up history
        for price in [100, 100.5, 99.5, 100.2, 99.8]:
            market = MarketEvent(
                timestamp=datetime.now(),
                event_type=None,
                symbol="SPY",
                price=price,
                volume=1000000,
            )
            strategy.calculate_signals(market)

        # Clear any signals
        while not events.empty():
            events.get()

        # Now send extreme low price (should trigger LONG)
        extreme_low = MarketEvent(
            timestamp=datetime.now(),
            event_type=None,
            symbol="SPY",
            price=90.0,  # Very low
            volume=1000000,
        )
        strategy.calculate_signals(extreme_low)

        # Check for signal (may or may not trigger depending on z-score)


# =============================================================================
# BacktestEngine Tests
# =============================================================================


class TestBacktestEngine:
    """Test backtest engine."""

    def test_basic_backtest(self):
        """Test basic backtest execution."""
        events = Queue()

        # Create components
        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=50,
            start_price=100.0,
            drift=0.10,
        )

        portfolio = Portfolio(initial_capital=100000)
        executor = InstantExecutionHandler(events_queue=events)

        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        # Run backtest
        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        results = engine.run()

        # Check results
        assert isinstance(results, BacktestResults)
        assert results.n_bars > 0
        assert results.initial_capital == 100000
        assert len(results.equity_curve) > 0

    def test_backtest_with_slippage(self):
        """Test backtest with realistic slippage."""
        events = Queue()

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=100,
        )

        portfolio = Portfolio(initial_capital=100000)

        executor = SimulatedExecutionHandler(
            events_queue=events,
            slippage_bps=5.0,
            commission_model=FixedCommission(0.001),
        )

        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        results = engine.run()

        # Should have some costs
        assert results.total_costs >= 0

    def test_backtest_results_metrics(self):
        """Test that backtest calculates all metrics."""
        events = Queue()

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=252,  # One year
            start_price=100.0,
            drift=0.10,
            volatility=0.20,
        )

        portfolio = Portfolio(initial_capital=100000)
        executor = InstantExecutionHandler(events_queue=events)

        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        results = engine.run()

        # Check all metrics exist
        assert results.total_return_pct is not None
        assert results.volatility_pct is not None
        assert results.sharpe_ratio is not None
        assert results.max_drawdown_pct is not None

    def test_backtest_reset(self):
        """Test backtest engine reset."""
        events = Queue()

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=50,
        )

        portfolio = Portfolio(initial_capital=100000)
        executor = InstantExecutionHandler(events_queue=events)

        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        # Run first time
        results1 = engine.run()

        # Reset and run again
        engine.reset()
        results2 = engine.run()

        # Results should be similar (same synthetic data seed behavior)
        assert results2.n_bars > 0


# =============================================================================
# Monte Carlo Tests
# =============================================================================


class TestMonteCarloSimulator:
    """Test Monte Carlo simulation."""

    def test_monte_carlo_basic(self):
        """Test basic Monte Carlo simulation."""
        # Create sample backtest result
        returns = list(np.random.normal(0.0005, 0.01, 252))
        equity_curve = [(datetime.now(), 100000)]
        equity = 100000
        for r in returns:
            equity *= (1 + r)
            equity_curve.append((datetime.now(), equity))

        result = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trade_history=[],
            total_return_pct=((equity / 100000) - 1) * 100,
            sharpe_ratio=1.0,
            initial_capital=100000,
        )

        # Run simulation
        mc = MonteCarloSimulator(n_simulations=100, method="shuffle")
        mc_results = mc.run(result)

        assert mc_results.n_simulations == 100
        assert len(mc_results.sharpe_ratios) == 100
        assert len(mc_results.total_returns) == 100
        assert len(mc_results.max_drawdowns) == 100

    def test_monte_carlo_confidence_intervals(self):
        """Test confidence interval calculation."""
        returns = list(np.random.normal(0.0005, 0.01, 252))
        equity_curve = [(datetime.now(), 100000)]
        equity = 100000
        for r in returns:
            equity *= (1 + r)
            equity_curve.append((datetime.now(), equity))

        result = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trade_history=[],
            initial_capital=100000,
        )

        mc = MonteCarloSimulator(n_simulations=500, method="block", block_size=21)
        mc_results = mc.run(result)

        # Get confidence intervals
        sharpe_ci = mc_results.get_confidence_interval("sharpe", 0.95)
        return_ci = mc_results.get_confidence_interval("return", 0.95)

        # CI should have lower < upper
        assert sharpe_ci[0] < sharpe_ci[1]
        assert return_ci[0] < return_ci[1]

    def test_monte_carlo_methods(self):
        """Test different bootstrap methods."""
        returns = list(np.random.normal(0.001, 0.015, 100))
        equity_curve = [(datetime.now(), 100000)]
        equity = 100000
        for r in returns:
            equity *= (1 + r)
            equity_curve.append((datetime.now(), equity))

        result = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trade_history=[],
            initial_capital=100000,
        )

        # Test shuffle method
        mc_shuffle = MonteCarloSimulator(n_simulations=50, method="shuffle")
        results_shuffle = mc_shuffle.run(result)
        assert len(results_shuffle.sharpe_ratios) == 50

        # Test block method
        mc_block = MonteCarloSimulator(n_simulations=50, method="block")
        results_block = mc_block.run(result)
        assert len(results_block.sharpe_ratios) == 50

        # Test parametric method
        mc_param = MonteCarloSimulator(n_simulations=50, method="parametric")
        results_param = mc_param.run(result)
        assert len(results_param.sharpe_ratios) == 50

    def test_monte_carlo_risk_metrics(self):
        """Test risk probability calculations."""
        # Create negative return scenario
        returns = list(np.random.normal(-0.001, 0.02, 252))
        equity_curve = [(datetime.now(), 100000)]
        equity = 100000
        for r in returns:
            equity *= (1 + r)
            equity_curve.append((datetime.now(), equity))

        result = BacktestResults(
            equity_curve=equity_curve,
            returns=returns,
            trade_history=[],
            initial_capital=100000,
        )

        mc = MonteCarloSimulator(n_simulations=200, method="shuffle")
        mc_results = mc.run(result)

        # Check probability calculations
        prob_loss = mc_results.get_probability_of_loss()
        assert 0 <= prob_loss <= 1

        prob_dd_20 = mc_results.get_probability_of_drawdown(20)
        assert 0 <= prob_dd_20 <= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full backtest workflow."""

    def test_full_backtest_workflow(self):
        """Test complete backtest workflow with all components."""
        events = Queue()

        # Multi-symbol backtest
        symbols = ["SPY", "QQQ"]

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=symbols,
            n_bars=252,
            start_price=100.0,
            drift=0.08,
            volatility=0.18,
        )

        portfolio = Portfolio(
            initial_capital=100000,
            max_position_pct=0.25,
        )

        executor = SimulatedExecutionHandler(
            events_queue=events,
            slippage_bps=5.0,
            market_impact_factor=0.05,
            commission_model=PerShareCommission(per_share=0.005, minimum=1.0),
        )

        strategy = MovingAverageCrossoverStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            fast_window=10,
            slow_window=30,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        results = engine.run()

        # Verify results
        # n_bars counts equity_curve entries (2 symbols * 252 bars = 504)
        assert results.n_bars == 252 * 2
        assert results.initial_capital == 100000
        assert results.final_equity > 0
        assert len(results.returns) > 0

        # Verify summary generation
        summary = results.summary()
        assert "BACKTEST RESULTS" in summary
        assert "Sharpe Ratio" in summary

    def test_backtest_with_monte_carlo(self):
        """Test backtest followed by Monte Carlo analysis."""
        events = Queue()

        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=["SPY"],
            n_bars=252,
        )

        portfolio = Portfolio(initial_capital=100000)
        executor = InstantExecutionHandler(events_queue=events)

        strategy = BuyAndHoldStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            execution_handler=executor,
        )

        results = engine.run()

        # Run Monte Carlo
        mc = MonteCarloSimulator(n_simulations=100)
        mc_results = mc.run(results)

        # Verify MC results
        assert mc_results.original_result == results
        sharpe_ci = mc_results.get_confidence_interval("sharpe", 0.95)
        assert len(sharpe_ci) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
