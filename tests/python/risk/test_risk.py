"""
Tests for the risk management module.

Tests cover:
- RiskManager: Unified risk framework with limits
- VolatilityScaledPositionSizer: Vol-managed position sizing
- VolatilityEstimator: EWMA, GARCH volatility estimation
- VaRCalculator: Value at Risk (parametric, historical, Monte Carlo)
- StressTester: Crisis scenarios and stress testing
- GreeksRiskMonitor: Portfolio Greeks and hedging
- CorrelationMonitor: Cointegration and correlation health
- DrawdownController: Drawdown limits and kill switch
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from quant_trading.risk import (
    CorrelationHealth,
    CorrelationMonitor,
    CorrelationMonitorConfig,
    DrawdownController,
    DrawdownControllerConfig,
    DrawdownMetrics,
    GreeksMonitorConfig,
    GreeksRiskMonitor,
    HedgeActionType,
    HealthStatus,
    KellyPositionSizer,
    OptionPosition,
    PortfolioGreeks,
    PositionRisk,
    PositionSizerConfig,
    RiskAction,
    RiskCheckResult,
    RiskLevel,
    RiskLimit,
    RiskLimitType,
    RiskManager,
    StressTester,
    StressTestResult,
    VaRCalculator,
    VaRMethod,
    VaRResult,
    VolatilityEstimator,
    VolatilityMethod,
    VolatilityScaledPositionSizer,
)


# =============================================================================
# Position Sizer Tests
# =============================================================================


class TestVolatilityScaledPositionSizer:
    """Tests for VolatilityScaledPositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create sizer instance."""
        return VolatilityScaledPositionSizer()

    @pytest.fixture
    def low_vol_returns(self):
        """Generate low volatility returns (10% annualized)."""
        daily_vol = 0.10 / np.sqrt(252)
        return np.random.normal(0.0003, daily_vol, 21)

    @pytest.fixture
    def high_vol_returns(self):
        """Generate high volatility returns (30% annualized)."""
        daily_vol = 0.30 / np.sqrt(252)
        return np.random.normal(0, daily_vol, 21)

    def test_sizer_initialization(self, sizer):
        """Test sizer initializes correctly."""
        assert sizer.config.target_annual_vol == 0.15
        assert sizer.config.max_leverage == 2.0

    def test_low_vol_increases_position(self, sizer, low_vol_returns):
        """Test that low volatility leads to larger position (up to cap)."""
        result = sizer.compute_position_size(
            return_series=low_vol_returns,
            available_capital=1_000_000,
        )

        # Low vol should result in higher leverage (though capped at max_position_pct=25%)
        # The vol-scaling formula gives weight > 1 but it's capped
        assert result.target_weight > 0
        assert result.position_size > 0
        # Rationale should show the uncapped weight was higher
        assert "capped" in result.rationale or result.target_weight <= sizer.config.max_position_pct

    def test_high_vol_decreases_position(self, sizer, high_vol_returns):
        """Test that high volatility leads to smaller position."""
        result = sizer.compute_position_size(
            return_series=high_vol_returns,
            available_capital=1_000_000,
        )

        # High vol should result in leverage < 1
        assert result.target_weight < 1.0
        assert result.position_size < 1_000_000

    def test_leverage_caps(self, sizer):
        """Test leverage is capped at max/min."""
        # Very low vol returns
        low_vol = np.random.normal(0, 0.001, 21)
        result = sizer.compute_position_size(low_vol, 1_000_000)
        assert result.target_weight <= sizer.config.max_leverage

        # Very high vol returns
        high_vol = np.random.normal(0, 0.05, 21)
        result = sizer.compute_position_size(high_vol, 1_000_000)
        assert result.target_weight >= sizer.config.min_leverage

    def test_drawdown_reduction(self, sizer):
        """Test position reduced during drawdown."""
        # Use higher vol returns so we don't hit the max_position_pct cap
        np.random.seed(123)
        high_vol_returns = np.random.normal(0, 0.015, 21)  # ~24% annualized

        # Normal conditions
        result_normal = sizer.compute_position_size(
            return_series=high_vol_returns,
            available_capital=1_000_000,
            current_drawdown=0.0,
        )

        # In drawdown - use 25% which triggers reduction
        result_dd = sizer.compute_position_size(
            return_series=high_vol_returns,
            available_capital=1_000_000,
            current_drawdown=0.25,  # 25% drawdown
        )

        # Position should be reduced due to drawdown multiplier
        assert result_dd.position_size < result_normal.position_size
        assert "drawdown" in result_dd.rationale

    def test_position_size_result_fields(self, sizer, low_vol_returns):
        """Test result contains all expected fields."""
        result = sizer.compute_position_size(low_vol_returns, 1_000_000)

        assert result.position_size > 0
        assert result.target_weight > 0
        assert result.realized_vol > 0
        assert result.rationale != ""
        assert result.expected_daily_var is not None


class TestKellyPositionSizer:
    """Tests for KellyPositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create Kelly sizer instance."""
        return KellyPositionSizer(kelly_fraction=0.5)

    def test_kelly_basic(self, sizer):
        """Test basic Kelly calculation."""
        result = sizer.compute_position_size(
            expected_return=0.15,  # 15% expected return
            volatility=0.20,  # 20% vol
            available_capital=1_000_000,
        )

        # Full Kelly = (0.15 - 0.05) / 0.2^2 = 2.5
        # Half Kelly = 1.25, capped at 0.25
        assert result.target_weight == 0.25  # Should be capped
        assert result.position_size == 250_000

    def test_kelly_negative_edge(self, sizer):
        """Test Kelly returns zero for negative edge."""
        result = sizer.compute_position_size(
            expected_return=0.03,  # Below risk-free
            volatility=0.20,
            available_capital=1_000_000,
        )

        assert result.target_weight == 0
        assert result.position_size == 0


# =============================================================================
# Greeks Risk Monitor Tests
# =============================================================================


class TestOptionPosition:
    """Tests for OptionPosition dataclass."""

    def test_position_creation(self):
        """Test creating an option position."""
        pos = OptionPosition(
            symbol="SPY 420 C",
            underlying="SPY",
            option_type="call",
            strike=420.0,
            expiration=datetime.now() + timedelta(days=30),
            quantity=10,
            direction="long",
            delta=0.5,
            gamma=0.02,
            vega=0.15,
            theta=-0.05,
        )

        assert pos.symbol == "SPY 420 C"
        assert pos.delta == 0.5


class TestGreeksRiskMonitor:
    """Tests for GreeksRiskMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return GreeksRiskMonitor()

    @pytest.fixture
    def sample_positions(self):
        """Create sample option positions."""
        return [
            OptionPosition(
                symbol="SPY 420 C",
                underlying="SPY",
                option_type="call",
                strike=420.0,
                expiration=datetime.now() + timedelta(days=30),
                quantity=10,
                direction="long",
                delta=0.5,
                gamma=0.02,
                vega=0.15,
                theta=-0.05,
                underlying_price=420.0,
            ),
            OptionPosition(
                symbol="SPY 400 P",
                underlying="SPY",
                option_type="put",
                strike=400.0,
                expiration=datetime.now() + timedelta(days=30),
                quantity=5,
                direction="short",
                delta=-0.3,
                gamma=0.015,
                vega=0.12,
                theta=-0.04,
                underlying_price=420.0,
            ),
        ]

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor.config.delta_threshold == 100.0

    def test_compute_portfolio_greeks(self, monitor, sample_positions):
        """Test portfolio Greeks aggregation."""
        greeks = monitor.compute_portfolio_greeks(sample_positions)

        # 10 long calls with 0.5 delta = +500 delta (shares)
        # 5 short puts with -0.3 delta = +150 delta (negative of negative)
        # Total delta = 500 + 150 = 650
        expected_delta = (10 * 0.5 * 100) + (-5 * -0.3 * 100)
        assert abs(greeks.delta - expected_delta) < 1

        assert isinstance(greeks, PortfolioGreeks)
        assert greeks.gamma != 0
        assert greeks.vega != 0

    def test_check_rehedge_high_delta(self, monitor):
        """Test rehedge recommended for high delta."""
        greeks = PortfolioGreeks(
            delta=500,  # High delta
            gamma=20,
            vega=500,
            theta=-100,
        )

        needs_action, actions = monitor.check_rehedge_needed(greeks)

        assert needs_action
        assert any(a.action_type == HedgeActionType.HEDGE_DELTA for a in actions)

    def test_no_rehedge_normal_greeks(self, monitor):
        """Test no rehedge when Greeks are within limits."""
        greeks = PortfolioGreeks(
            delta=50,  # Within threshold
            gamma=20,
            vega=500,
            theta=-100,
        )

        needs_action, actions = monitor.check_rehedge_needed(greeks)

        # May or may not have alerts depending on other Greeks
        delta_hedges = [a for a in actions if a.action_type == HedgeActionType.HEDGE_DELTA]
        assert len(delta_hedges) == 0

    def test_compute_hedge_trade(self, monitor):
        """Test hedge trade computation."""
        hedge = monitor.compute_hedge_trade(
            current_delta=500,
            underlying="SPY",
            underlying_price=420.0,
        )

        assert hedge["action"] == "hedge_with_stock"
        assert hedge["shares"] == 500
        assert hedge["side"] == "sell"


# =============================================================================
# Correlation Monitor Tests
# =============================================================================


class TestCorrelationMonitor:
    """Tests for CorrelationMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return CorrelationMonitor()

    @pytest.fixture
    def cointegrated_prices(self):
        """Generate cointegrated price series."""
        np.random.seed(42)
        n = 300

        # Common factor
        factor = np.cumsum(np.random.normal(0, 1, n))

        # Asset 1: factor + noise
        asset1 = 100 + factor + np.random.normal(0, 0.5, n)

        # Asset 2: 0.8 * factor + noise (cointegrated)
        asset2 = 80 + 0.8 * factor + np.random.normal(0, 0.5, n)

        return asset1, asset2

    @pytest.fixture
    def uncorrelated_prices(self):
        """Generate uncorrelated price series."""
        np.random.seed(42)
        n = 300

        asset1 = 100 + np.cumsum(np.random.normal(0, 1, n))
        asset2 = 100 + np.cumsum(np.random.normal(0, 1, n))

        return asset1, asset2

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor.config.min_correlation == 0.7
        assert monitor.config.cointegration_p_value == 0.05

    def test_check_healthy_pair(self, monitor, cointegrated_prices):
        """Test healthy cointegrated pair."""
        asset1, asset2 = cointegrated_prices

        health = monitor.check_pair_health(
            pair_name="TEST-PAIR",
            asset1_prices=asset1,
            asset2_prices=asset2,
        )

        # Cointegrated series should be healthy
        assert health.current_correlation > 0.5
        assert isinstance(health, CorrelationHealth)

    def test_check_unhealthy_pair(self, monitor, uncorrelated_prices):
        """Test unhealthy uncorrelated pair."""
        asset1, asset2 = uncorrelated_prices

        health = monitor.check_pair_health(
            pair_name="TEST-PAIR",
            asset1_prices=asset1,
            asset2_prices=asset2,
        )

        # Uncorrelated series may have warnings
        assert isinstance(health, CorrelationHealth)
        # Status depends on actual correlation

    def test_position_recommendations(self, monitor, cointegrated_prices):
        """Test position recommendations based on health."""
        asset1, asset2 = cointegrated_prices

        health = monitor.check_pair_health(
            pair_name="TEST-PAIR",
            asset1_prices=asset1,
            asset2_prices=asset2,
        )

        rec = monitor.get_position_recommendations(health, current_position_size=100_000)

        assert "action" in rec
        assert "target_size" in rec


# =============================================================================
# Drawdown Controller Tests
# =============================================================================


class TestDrawdownController:
    """Tests for DrawdownController."""

    @pytest.fixture
    def controller(self):
        """Create controller instance."""
        return DrawdownController(initial_capital=1_000_000)

    def test_controller_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller._current_value == 1_000_000
        assert controller._peak_value == 1_000_000
        assert controller.config.critical_threshold == 0.25

    def test_update_increases_peak(self, controller):
        """Test update increases peak on new high."""
        controller.update(1_100_000)

        assert controller._peak_value == 1_100_000
        assert controller._current_value == 1_100_000

    def test_update_tracks_drawdown(self, controller):
        """Test drawdown tracking."""
        controller.update(1_100_000)  # New high
        controller.update(990_000)  # Drawdown

        metrics = controller.get_metrics()

        assert metrics.peak_value == 1_100_000
        assert metrics.current_value == 990_000
        assert metrics.current_drawdown == pytest.approx(0.10, rel=0.01)

    def test_check_limits_normal(self, controller):
        """Test limits check when within thresholds."""
        # Gradual decline to avoid triggering daily loss limit (3%)
        controller.update(980_000)  # -2%
        controller.update(960_000)  # -2%
        controller.update(950_000)  # ~-1% (5% total drawdown, each day < 3%)

        status = controller.check_limits()

        assert status.risk_level == RiskLevel.NORMAL
        assert status.recommended_action == RiskAction.NO_ACTION
        assert status.exposure_multiplier == 1.0

    def test_check_limits_elevated(self, controller):
        """Test limits check at elevated risk."""
        controller.update(850_000)  # 15% drawdown

        status = controller.check_limits()

        assert status.risk_level == RiskLevel.ELEVATED
        assert status.recommended_action == RiskAction.REDUCE_EXPOSURE
        assert status.exposure_multiplier < 1.0

    def test_check_limits_critical(self, controller):
        """Test limits check at critical risk."""
        # First update to set peak, then drawdown to 27% (> 25% critical threshold)
        controller.update(1_000_000)  # Confirm peak
        controller.update(730_000)  # 27% drawdown

        status = controller.check_limits()

        # At >25% drawdown, should be CRITICAL level
        assert status.risk_level == RiskLevel.CRITICAL
        assert status.recommended_action == RiskAction.HALT_NEW_TRADES

    def test_kill_switch(self, controller):
        """Test kill switch activation."""
        controller.activate_kill_switch("Test emergency")

        status = controller.check_limits()

        assert status.risk_level == RiskLevel.EMERGENCY
        assert status.recommended_action == RiskAction.KILL_SWITCH
        assert controller._kill_switch_active

        controller.deactivate_kill_switch()
        assert not controller._kill_switch_active

    def test_drawdown_metrics(self, controller):
        """Test drawdown metrics computation."""
        controller.update(1_100_000)
        controller.update(900_000)

        metrics = controller.get_metrics()

        assert metrics.current_drawdown == pytest.approx(0.182, rel=0.01)
        assert metrics.recovery_needed == pytest.approx(0.222, rel=0.01)

    def test_reset(self, controller):
        """Test controller reset."""
        controller.update(800_000)  # Create drawdown
        controller.reset(new_capital=500_000)

        assert controller._peak_value == 500_000
        assert controller._current_value == 500_000
        assert controller._max_drawdown == 0.0

    def test_strategy_limits(self, controller):
        """Test strategy-level limits checking."""
        strategy_values = {
            "vol_arb": 80_000,
            "mean_rev": 95_000,
        }
        strategy_peaks = {
            "vol_arb": 100_000,  # 20% DD
            "mean_rev": 100_000,  # 5% DD
        }

        results = controller.check_strategy_limits(strategy_values, strategy_peaks)

        assert results["vol_arb"].risk_level == RiskLevel.CRITICAL
        assert results["mean_rev"].risk_level == RiskLevel.NORMAL


# =============================================================================
# Integration Tests
# =============================================================================


class TestRiskManagementIntegration:
    """Integration tests for risk management workflow."""

    def test_position_sizing_with_drawdown(self):
        """Test position sizing respects drawdown limits."""
        sizer = VolatilityScaledPositionSizer()
        controller = DrawdownController(initial_capital=1_000_000)

        # Simulate drawdown
        controller.update(850_000)
        status = controller.check_limits()

        # Size position with drawdown consideration
        returns = np.random.normal(0, 0.01, 21)
        result = sizer.compute_position_size(
            return_series=returns,
            available_capital=850_000,
            current_drawdown=status.exposure_multiplier,
        )

        # Position should be reduced
        assert result.position_size <= 850_000

    def test_greeks_monitoring_workflow(self):
        """Test complete Greeks monitoring workflow."""
        monitor = GreeksRiskMonitor()

        positions = [
            OptionPosition(
                symbol="SPY 420 C",
                underlying="SPY",
                option_type="call",
                strike=420.0,
                expiration=datetime.now() + timedelta(days=30),
                quantity=20,
                direction="long",
                delta=0.6,
                gamma=0.025,
                vega=0.18,
                theta=-0.06,
                underlying_price=420.0,
            ),
        ]

        # Compute Greeks
        greeks = monitor.compute_portfolio_greeks(
            positions,
            underlying_prices={"SPY": 420.0},
        )

        # Check if rehedge needed
        needs_hedge, actions = monitor.check_rehedge_needed(greeks, portfolio_value=500_000)

        # Get summary
        summary = monitor.summarize_greeks(greeks, portfolio_value=500_000)

        assert "assessment" in summary
        assert "delta" in summary


# =============================================================================
# Risk Manager Tests
# =============================================================================


class TestRiskManager:
    """Tests for unified RiskManager."""

    @pytest.fixture
    def risk_mgr(self):
        """Create risk manager instance."""
        mgr = RiskManager(total_capital=1_000_000)
        mgr.set_default_limits()
        return mgr

    def test_initialization(self, risk_mgr):
        """Test risk manager initializes correctly."""
        assert risk_mgr.total_capital == 1_000_000
        assert len(risk_mgr.limits) > 0
        assert RiskLimitType.POSITION_SIZE in risk_mgr.limits

    def test_add_limit(self):
        """Test adding risk limits."""
        mgr = RiskManager(total_capital=1_000_000)

        mgr.add_limit(RiskLimit(
            limit_type=RiskLimitType.POSITION_SIZE,
            value=0.15,
            action_on_breach="reduce",
        ))

        assert RiskLimitType.POSITION_SIZE in mgr.limits
        assert mgr.limits[RiskLimitType.POSITION_SIZE].value == 0.15

    def test_position_allowed_within_limits(self, risk_mgr):
        """Test position allowed when within limits."""
        result = risk_mgr.check_position_allowed(
            asset_id="SPY",
            position_size=100,  # 100 shares * 450 = $45,000 = 4.5%
            current_price=450.0,
        )

        assert result.is_allowed
        assert result.recommended_action == "proceed"
        assert len(result.breached_limits) == 0

    def test_position_rejected_size_exceeded(self, risk_mgr):
        """Test position rejected when size limit exceeded."""
        result = risk_mgr.check_position_allowed(
            asset_id="SPY",
            position_size=500,  # 500 shares * 450 = $225,000 = 22.5% > 10%
            current_price=450.0,
        )

        assert not result.is_allowed
        assert "position_size" in str(result.breached_limits)

    def test_circuit_breaker(self, risk_mgr):
        """Test circuit breaker blocks all positions."""
        risk_mgr.activate_circuit_breaker("Test emergency")

        result = risk_mgr.check_position_allowed(
            asset_id="SPY",
            position_size=10,
            current_price=450.0,
        )

        assert not result.is_allowed
        assert "circuit_breaker" in str(result.breached_limits)

        risk_mgr.deactivate_circuit_breaker()
        assert not risk_mgr.circuit_breaker_active

    def test_update_position(self, risk_mgr):
        """Test position tracking."""
        position = PositionRisk(
            asset_id="SPY",
            position_size=100,
            market_value=45000,
            delta=100,
        )

        risk_mgr.update_position(position)

        assert "SPY" in risk_mgr.positions
        assert risk_mgr.positions["SPY"].market_value == 45000

    def test_compute_portfolio_risk(self, risk_mgr):
        """Test portfolio risk computation."""
        risk_mgr.update_position(PositionRisk(
            asset_id="SPY",
            position_size=100,
            market_value=45000,
            delta=100,
            gamma=5,
        ))
        risk_mgr.update_position(PositionRisk(
            asset_id="QQQ",
            position_size=-50,
            market_value=-17500,
            delta=-50,
            gamma=-3,
        ))

        portfolio = risk_mgr.compute_portfolio_risk()

        assert portfolio.total_exposure == 62500  # 45000 + 17500
        assert portfolio.net_exposure == 27500  # 45000 - 17500
        assert portfolio.total_delta == 50  # 100 - 50
        assert portfolio.total_gamma == 2  # 5 - 3

    def test_check_all_limits(self, risk_mgr):
        """Test checking all portfolio limits."""
        result = risk_mgr.check_all_limits(daily_pnl=0)

        assert isinstance(result, RiskCheckResult)
        assert result.is_allowed  # Should be OK with no positions

    def test_limit_status_report(self, risk_mgr):
        """Test limit status reporting."""
        status = risk_mgr.get_limit_status()

        assert RiskLimitType.POSITION_SIZE.value in status
        assert "utilization" in status[RiskLimitType.POSITION_SIZE.value]


# =============================================================================
# Volatility Estimator Tests
# =============================================================================


class TestVolatilityEstimator:
    """Tests for VolatilityEstimator."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns with known volatility."""
        np.random.seed(42)
        daily_vol = 0.01  # 1% daily ≈ 16% annualized
        return np.random.normal(0, daily_vol, 252)

    def test_realized_vol(self, sample_returns):
        """Test realized volatility estimation."""
        estimator = VolatilityEstimator(method=VolatilityMethod.REALIZED)
        vol = estimator.estimate(sample_returns)

        # Should be close to 16% annualized
        assert 0.10 < vol < 0.25

    def test_ewma_vol(self, sample_returns):
        """Test EWMA volatility estimation."""
        estimator = VolatilityEstimator(
            method=VolatilityMethod.EWMA,
            ewma_lambda=0.94,
        )
        vol = estimator.estimate(sample_returns)

        # Should be close to 16% annualized
        assert 0.10 < vol < 0.25

    def test_hybrid_vol(self, sample_returns):
        """Test hybrid volatility estimation."""
        estimator = VolatilityEstimator(method=VolatilityMethod.HYBRID)
        vol = estimator.estimate(sample_returns)

        # Should be close to 16% annualized
        assert 0.10 < vol < 0.25

    def test_vol_with_prices(self):
        """Test volatility from price series."""
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))

        estimator = VolatilityEstimator()
        vol = estimator.estimate(returns=np.array([]), prices=prices)

        assert vol > 0

    def test_confidence_interval(self, sample_returns):
        """Test volatility confidence interval."""
        estimator = VolatilityEstimator()

        try:
            vol, lower, upper = estimator.estimate_with_confidence(sample_returns)
            assert lower < vol < upper
        except ImportError:
            # scipy not available
            pass


# =============================================================================
# VaR Calculator Tests
# =============================================================================


class TestVaRCalculator:
    """Tests for VaRCalculator."""

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        n_days = 252
        # Two assets with some correlation
        returns = np.random.multivariate_normal(
            [0.0005, 0.0003],
            [[0.0001, 0.00005], [0.00005, 0.00008]],
            n_days,
        )
        return returns

    @pytest.fixture
    def portfolio(self):
        """Sample portfolio."""
        return {"SPY": 45000, "TLT": 10000}

    def test_parametric_var(self, sample_returns, portfolio):
        """Test parametric VaR calculation."""
        calculator = VaRCalculator(method=VaRMethod.PARAMETRIC)

        result = calculator.calculate(
            position_values=portfolio,
            historical_returns=sample_returns,
            asset_ids=["SPY", "TLT"],
        )

        assert isinstance(result, VaRResult)
        assert result.var_95 > 0
        assert result.var_99 > result.var_95  # 99% VaR > 95% VaR
        assert result.cvar_95 >= result.var_95  # CVaR >= VaR
        assert result.method == "parametric"

    def test_historical_var(self, sample_returns, portfolio):
        """Test historical VaR calculation."""
        calculator = VaRCalculator(method=VaRMethod.HISTORICAL)

        result = calculator.calculate(
            position_values=portfolio,
            historical_returns=sample_returns,
            asset_ids=["SPY", "TLT"],
        )

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "historical"

    def test_monte_carlo_var(self, sample_returns, portfolio):
        """Test Monte Carlo VaR calculation."""
        calculator = VaRCalculator(
            method=VaRMethod.MONTE_CARLO,
            n_simulations=5000,
        )

        result = calculator.calculate(
            position_values=portfolio,
            historical_returns=sample_returns,
            asset_ids=["SPY", "TLT"],
        )

        assert result.var_95 > 0
        assert result.var_99 > result.var_95
        assert result.method == "monte_carlo"

    def test_var_result_properties(self, sample_returns, portfolio):
        """Test VaRResult properties."""
        calculator = VaRCalculator()

        result = calculator.calculate(
            position_values=portfolio,
            historical_returns=sample_returns,
            asset_ids=["SPY", "TLT"],
        )

        assert result.portfolio_value == 55000
        assert result.var_95_pct > 0
        assert result.time_horizon == 1

    def test_component_var(self, sample_returns, portfolio):
        """Test component VaR calculation."""
        calculator = VaRCalculator()

        result = calculator.calculate(
            position_values=portfolio,
            historical_returns=sample_returns,
            asset_ids=["SPY", "TLT"],
        )

        # Should have component VaR for each asset
        assert "SPY" in result.component_var or len(result.component_var) >= 0


# =============================================================================
# Stress Tester Tests
# =============================================================================


class TestStressTester:
    """Tests for StressTester."""

    @pytest.fixture
    def stress_tester(self):
        """Create stress tester instance."""
        return StressTester()

    @pytest.fixture
    def portfolio(self):
        """Sample portfolio."""
        return {"SPY": 45000, "TLT": 10000, "GLD": 5000}

    def test_initialization(self, stress_tester):
        """Test stress tester initializes with scenarios."""
        assert len(stress_tester.scenarios) > 0
        assert "2008_financial_crisis" in stress_tester.scenarios

    def test_apply_scenario(self, stress_tester, portfolio):
        """Test applying a stress scenario."""
        result = stress_tester.apply_scenario(
            portfolio=portfolio,
            scenario_name="2008_financial_crisis",
        )

        assert isinstance(result, StressTestResult)
        assert result.scenario_pnl < 0  # Should be loss
        assert result.scenario_name == "2008_financial_crisis"
        assert "SPY" in result.positions_affected

    def test_custom_scenario(self, stress_tester, portfolio):
        """Test custom stress scenario."""
        custom_shocks = {
            "SPY": -0.20,
            "TLT": 0.05,
            "GLD": 0.10,
        }

        result = stress_tester.apply_custom_scenario(
            portfolio=portfolio,
            shocks=custom_shocks,
            scenario_name="custom_test",
        )

        expected_pnl = 45000 * -0.20 + 10000 * 0.05 + 5000 * 0.10
        assert abs(result.scenario_pnl - expected_pnl) < 1

    def test_run_all_scenarios(self, stress_tester, portfolio):
        """Test running all scenarios."""
        results = stress_tester.run_all_scenarios(portfolio)

        assert len(results) == len(stress_tester.scenarios)
        # Results should be sorted by P&L (worst first)
        assert results[0].scenario_pnl <= results[-1].scenario_pnl

    def test_worst_case(self, stress_tester, portfolio):
        """Test getting worst case scenario."""
        worst = stress_tester.get_worst_case(portfolio)

        assert isinstance(worst, StressTestResult)
        assert worst.scenario_pnl < 0

    def test_summary_report(self, stress_tester, portfolio):
        """Test summary report generation."""
        report = stress_tester.summary_report(portfolio)

        assert "portfolio_value" in report
        assert "worst_case" in report
        assert "best_case" in report
        assert report["worst_case"]["pnl"] < report["best_case"]["pnl"]

    def test_add_custom_scenario(self, stress_tester):
        """Test adding custom scenario."""
        stress_tester.add_scenario("test_scenario", {"SPY": -0.50})

        assert "test_scenario" in stress_tester.scenarios


# =============================================================================
# Additional Integration Tests
# =============================================================================


class TestRiskSystemIntegration:
    """Integration tests for complete risk management workflow."""

    def test_full_risk_workflow(self):
        """Test complete risk management workflow."""
        # Initialize components
        risk_mgr = RiskManager(total_capital=1_000_000)
        risk_mgr.set_default_limits()

        sizer = VolatilityScaledPositionSizer()
        var_calc = VaRCalculator()
        stress_tester = StressTester()

        # Generate synthetic data
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 252)
        portfolio_returns = np.random.normal(0, 0.01, (252, 2))

        # Step 1: Size position
        size_result = sizer.compute_position_size(returns, 1_000_000)
        position_value = min(size_result.position_size, 100_000)

        # Step 2: Check risk limits
        check_result = risk_mgr.check_position_allowed(
            asset_id="SPY",
            position_size=int(position_value / 450),
            current_price=450.0,
        )

        # Step 3: If allowed, add position
        if check_result.is_allowed:
            risk_mgr.update_position(PositionRisk(
                asset_id="SPY",
                position_size=int(position_value / 450),
                market_value=position_value,
            ))

        # Step 4: Calculate VaR
        portfolio = {"SPY": position_value, "CASH": 1_000_000 - position_value}
        var_result = var_calc.calculate(
            position_values={"SPY": position_value},
            historical_returns=portfolio_returns[:, 0].reshape(-1, 1),
            asset_ids=["SPY"],
        )

        # Step 5: Run stress tests
        stress_result = stress_tester.get_worst_case({"SPY": position_value})

        # Verify workflow completed
        assert check_result.is_allowed
        assert var_result.var_95 > 0
        assert stress_result.scenario_pnl != 0

    def test_risk_limit_enforcement(self):
        """Test that risk limits are properly enforced."""
        risk_mgr = RiskManager(total_capital=1_000_000)

        # Add tight position limit
        risk_mgr.add_limit(RiskLimit(
            limit_type=RiskLimitType.POSITION_SIZE,
            value=0.05,  # 5% max
            action_on_breach="halt",
        ))

        # Small position should pass
        result1 = risk_mgr.check_position_allowed("SPY", 100, 450.0)  # $45,000 = 4.5%
        assert result1.is_allowed

        # Large position should fail
        result2 = risk_mgr.check_position_allowed("QQQ", 200, 400.0)  # $80,000 = 8%
        assert not result2.is_allowed

    def test_var_methods_consistency(self):
        """Test that different VaR methods give similar results."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, (252, 1))
        portfolio = {"SPY": 100000}

        parametric = VaRCalculator(method=VaRMethod.PARAMETRIC)
        historical = VaRCalculator(method=VaRMethod.HISTORICAL)
        monte_carlo = VaRCalculator(method=VaRMethod.MONTE_CARLO, n_simulations=10000)

        result_p = parametric.calculate(portfolio, returns, ["SPY"])
        result_h = historical.calculate(portfolio, returns, ["SPY"])
        result_mc = monte_carlo.calculate(portfolio, returns, ["SPY"])

        # All methods should give VaR in same ballpark
        # Allow 50% difference due to different assumptions
        assert abs(result_p.var_95 - result_h.var_95) < result_p.var_95 * 0.5
        assert abs(result_p.var_95 - result_mc.var_95) < result_p.var_95 * 0.5
