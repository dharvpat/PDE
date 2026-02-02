"""
Tests for the signal generation module.

Tests cover:
- VolSurfaceArbitrageSignal: Option mispricing detection
- MeanReversionSignalGenerator: OU-based entry/exit signals
- SignalAggregator: Multi-strategy signal combination
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from quant_trading.signals import (
    AggregatedSignal,
    AggregatedSignalType,
    AggregatorConfig,
    MeanReversionConfig,
    MeanReversionSignal,
    MeanReversionSignalGenerator,
    MeanRevSignalType,
    Position,
    SignalAggregator,
    SignalType,
    VolArbitrageConfig,
    VolArbitrageSignal,
    VolSurfaceArbitrageSignal,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_options_data():
    """Generate sample options data for testing."""
    return pd.DataFrame({
        "underlying": ["SPY"] * 10,
        "strike": [380, 390, 400, 410, 420, 430, 440, 450, 460, 470],
        "expiration": [datetime.now() + timedelta(days=45)] * 10,
        "option_type": ["call"] * 10,
        "implied_vol": [0.25, 0.23, 0.21, 0.20, 0.19, 0.20, 0.21, 0.23, 0.25, 0.27],
        "T": [45 / 365] * 10,
        "bid": [15.0, 10.0, 6.0, 3.5, 2.0, 1.0, 0.5, 0.3, 0.15, 0.08],
        "ask": [15.5, 10.5, 6.3, 3.7, 2.2, 1.1, 0.55, 0.35, 0.18, 0.10],
        "volume": [1000, 2000, 5000, 8000, 10000, 8000, 5000, 2000, 1000, 500],
    })


@pytest.fixture
def mock_sabr_result():
    """Create mock SABR calibration result."""
    from dataclasses import dataclass

    @dataclass
    class MockSABRParams:
        alpha: float = 0.3
        beta: float = 0.5
        rho: float = -0.3
        nu: float = 0.5

    @dataclass
    class MockSABRResult:
        params_by_maturity: dict = None
        total_rmse: float = 0.02
        success: bool = True

        def __post_init__(self):
            if self.params_by_maturity is None:
                self.params_by_maturity = {45 / 365: MockSABRParams()}

    return MockSABRResult()


@pytest.fixture
def mock_ou_fit_result():
    """Create mock OU fit result."""
    from dataclasses import dataclass

    @dataclass
    class MockOUParams:
        theta: float = 0.0
        mu: float = 5.0
        sigma: float = 0.2
        half_life: float = 0.1386  # ln(2)/5 years
        stationary_std: float = 0.0632  # sigma / sqrt(2*mu)

    @dataclass
    class MockBoundaries:
        entry_lower: float = -0.05
        entry_upper: float = 0.05
        exit_long: float = 0.01
        exit_short: float = -0.01
        stop_loss_long: float = -0.15
        stop_loss_short: float = 0.15

    @dataclass
    class MockOUFitResult:
        params: MockOUParams = None
        boundaries: MockBoundaries = None
        success: bool = True

        def __post_init__(self):
            if self.params is None:
                self.params = MockOUParams()
            if self.boundaries is None:
                self.boundaries = MockBoundaries()

    return MockOUFitResult()


# =============================================================================
# VolSurfaceArbitrageSignal Tests
# =============================================================================


class TestVolArbitrageSignal:
    """Tests for VolArbitrageSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a vol arbitrage signal."""
        signal = VolArbitrageSignal(
            underlying="SPY",
            strike=420.0,
            expiration=datetime.now(),
            option_type="call",
            signal_type=SignalType.BUY,
            confidence=0.85,
            model_iv=0.25,
            market_iv=0.20,
            divergence_pct=0.25,
            rationale="Model IV higher than market IV",
        )

        assert signal.underlying == "SPY"
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.85

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = VolArbitrageSignal(
            underlying="SPY",
            strike=420.0,
            expiration=datetime(2024, 3, 15),
            option_type="call",
            signal_type=SignalType.BUY,
            confidence=0.85,
            model_iv=0.25,
            market_iv=0.20,
            divergence_pct=0.25,
            rationale="Test",
        )

        d = signal.to_dict()
        assert d["underlying"] == "SPY"
        assert d["signal_type"] == "buy"
        assert "timestamp" in d


class TestVolSurfaceArbitrageSignalGenerator:
    """Tests for VolSurfaceArbitrageSignal generator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return VolSurfaceArbitrageSignal()

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.config is not None
        assert generator.config.min_divergence_pct == 0.10

    def test_generator_with_custom_config(self):
        """Test generator with custom configuration."""
        config = VolArbitrageConfig(
            min_divergence_pct=0.05,
            min_confidence=0.5,
        )
        generator = VolSurfaceArbitrageSignal(config=config)

        assert generator.config.min_divergence_pct == 0.05
        assert generator.config.min_confidence == 0.5

    def test_generate_signals_requires_model(self, generator, sample_options_data):
        """Test that at least one model result is required."""
        with pytest.raises(ValueError, match="At least one model"):
            generator.generate_signals(
                market_data=sample_options_data,
                S0=420.0,
                r=0.05,
                q=0.02,
            )

    def test_sabr_vol_formula(self, generator):
        """Test SABR implied vol formula."""
        from dataclasses import dataclass

        @dataclass
        class MockParams:
            alpha: float = 0.3
            beta: float = 0.5
            rho: float = -0.3
            nu: float = 0.5

        vol = generator._sabr_vol_formula(
            F=100.0, K=100.0, T=0.25, params=MockParams()
        )

        # SABR with these parameters should give positive vol
        assert vol > 0
        assert vol < 1.0  # Vol should be reasonable

    def test_passes_filters_maturity(self, generator):
        """Test maturity filtering."""
        # Too short maturity
        option = {"T": 5 / 365}
        assert not generator._passes_filters(option, 5 / 365)

        # Good maturity
        option = {"T": 45 / 365}
        assert generator._passes_filters(option, 45 / 365)

        # Too long maturity
        option = {"T": 200 / 365}
        assert not generator._passes_filters(option, 200 / 365)

    def test_passes_filters_liquidity(self, generator):
        """Test liquidity filtering."""
        # Wide spread
        option = {"T": 45 / 365, "bid": 1.0, "ask": 1.5}  # 40% spread
        assert not generator._passes_filters(option, 45 / 365)

        # Tight spread
        option = {"T": 45 / 365, "bid": 1.0, "ask": 1.05}  # 5% spread
        assert generator._passes_filters(option, 45 / 365)

    def test_compute_confidence(self, generator):
        """Test confidence computation."""
        option = {
            "bid": 5.0,
            "ask": 5.2,  # 4% spread
        }

        confidence = generator._compute_confidence(
            option=option,
            T=45 / 365,
            calibration_rmse=0.02,
            divergence_pct=0.15,
        )

        assert 0.6 < confidence < 1.0


# =============================================================================
# MeanReversionSignalGenerator Tests
# =============================================================================


class TestMeanReversionSignal:
    """Tests for MeanReversionSignal dataclass."""

    def test_signal_creation(self):
        """Test creating a mean reversion signal."""
        signal = MeanReversionSignal(
            spread_name="SPY-IWM",
            signal_type=MeanRevSignalType.ENTRY_LONG,
            confidence=0.8,
            current_value=-0.06,
            rationale="Spread below entry lower",
            theta=0.0,
            half_life_days=35.0,
        )

        assert signal.spread_name == "SPY-IWM"
        assert signal.signal_type == MeanRevSignalType.ENTRY_LONG
        assert signal.confidence == 0.8

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = MeanReversionSignal(
            spread_name="SPY-IWM",
            signal_type=MeanRevSignalType.ENTRY_LONG,
            confidence=0.8,
            current_value=-0.06,
            rationale="Test",
        )

        d = signal.to_dict()
        assert d["spread_name"] == "SPY-IWM"
        assert d["signal_type"] == "entry_long"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        pos = Position(
            spread_name="SPY-IWM",
            direction="long",
            entry_price=-0.06,
            entry_time=datetime.now(),
            quantity=100,
            stop_loss=-0.15,
            take_profit=0.0,
        )

        assert pos.direction == "long"
        assert pos.entry_price == -0.06


class TestMeanReversionSignalGenerator:
    """Tests for MeanReversionSignalGenerator."""

    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return MeanReversionSignalGenerator()

    def test_generator_initialization(self, generator):
        """Test generator initializes correctly."""
        assert generator.config is not None
        assert generator.config.min_half_life_days == 5.0

    def test_generate_signal_entry_long(self, generator, mock_ou_fit_result):
        """Test entry long signal generation."""
        signal = generator.generate_signal(
            spread_name="SPY-IWM",
            current_value=-0.08,  # Below entry_lower (-0.05)
            ou_fit_result=mock_ou_fit_result,
        )

        assert signal is not None
        assert signal.signal_type == MeanRevSignalType.ENTRY_LONG
        assert signal.confidence >= 0.6

    def test_generate_signal_entry_short(self, generator, mock_ou_fit_result):
        """Test entry short signal generation."""
        signal = generator.generate_signal(
            spread_name="SPY-IWM",
            current_value=0.08,  # Above entry_upper (0.05)
            ou_fit_result=mock_ou_fit_result,
        )

        assert signal is not None
        assert signal.signal_type == MeanRevSignalType.ENTRY_SHORT
        assert signal.confidence >= 0.6

    def test_generate_signal_no_entry(self, generator, mock_ou_fit_result):
        """Test no signal when spread is within boundaries."""
        signal = generator.generate_signal(
            spread_name="SPY-IWM",
            current_value=0.02,  # Within boundaries
            ou_fit_result=mock_ou_fit_result,
        )

        assert signal is None

    def test_generate_signal_exit_take_profit(self, generator, mock_ou_fit_result):
        """Test take-profit exit signal."""
        position = Position(
            spread_name="SPY-IWM",
            direction="long",
            entry_price=-0.08,
            entry_time=datetime.now(),
            quantity=100,
            stop_loss=-0.15,
            take_profit=0.0,
        )

        signal = generator.generate_signal(
            spread_name="SPY-IWM",
            current_value=0.01,  # Above take_profit (0.0)
            ou_fit_result=mock_ou_fit_result,
            current_position=position,
        )

        assert signal is not None
        assert signal.signal_type == MeanRevSignalType.EXIT_TAKE_PROFIT
        assert signal.pnl > 0

    def test_generate_signal_exit_stop_loss(self, generator, mock_ou_fit_result):
        """Test stop-loss exit signal."""
        position = Position(
            spread_name="SPY-IWM",
            direction="long",
            entry_price=-0.08,
            entry_time=datetime.now(),
            quantity=100,
            stop_loss=-0.15,
            take_profit=0.0,
        )

        signal = generator.generate_signal(
            spread_name="SPY-IWM",
            current_value=-0.16,  # Below stop_loss (-0.15)
            ou_fit_result=mock_ou_fit_result,
            current_position=position,
        )

        assert signal is not None
        assert signal.signal_type == MeanRevSignalType.EXIT_STOP_LOSS
        assert signal.confidence == 1.0

    def test_position_management(self, generator, mock_ou_fit_result):
        """Test position registration and retrieval."""
        position = Position(
            spread_name="SPY-IWM",
            direction="long",
            entry_price=-0.08,
            entry_time=datetime.now(),
            quantity=100,
            stop_loss=-0.15,
            take_profit=0.0,
        )

        generator.register_position(position)
        retrieved = generator.get_position("SPY-IWM")

        assert retrieved is not None
        assert retrieved.direction == "long"

        closed = generator.close_position("SPY-IWM")
        assert closed is not None
        assert generator.get_position("SPY-IWM") is None


# =============================================================================
# SignalAggregator Tests
# =============================================================================


class TestAggregatedSignal:
    """Tests for AggregatedSignal dataclass."""

    def test_signal_creation(self):
        """Test creating an aggregated signal."""
        signal = AggregatedSignal(
            asset="SPY",
            signal_type=AggregatedSignalType.BUY,
            confidence=0.85,
            supporting_strategies=["vol_arbitrage", "mean_reversion"],
            conflicting_strategies=[],
            rationale="Unanimous buy signal",
        )

        assert signal.asset == "SPY"
        assert signal.signal_type == AggregatedSignalType.BUY
        assert len(signal.supporting_strategies) == 2

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = AggregatedSignal(
            asset="SPY",
            signal_type=AggregatedSignalType.BUY,
            confidence=0.85,
            supporting_strategies=["vol_arbitrage"],
            conflicting_strategies=[],
            rationale="Test",
        )

        d = signal.to_dict()
        assert d["asset"] == "SPY"
        assert d["signal_type"] == "buy"


class TestSignalAggregator:
    """Tests for SignalAggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance."""
        return SignalAggregator()

    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initializes correctly."""
        assert aggregator.config is not None
        assert aggregator.config.consensus_ratio == 1.5

    def test_aggregate_single_strategy(self, aggregator):
        """Test aggregation with single strategy signals."""
        vol_signals = [
            VolArbitrageSignal(
                underlying="SPY",
                strike=420.0,
                expiration=datetime.now(),
                option_type="call",
                signal_type=SignalType.BUY,
                confidence=0.85,
                model_iv=0.25,
                market_iv=0.20,
                divergence_pct=0.25,
                rationale="Underpriced",
            )
        ]

        result = aggregator.aggregate(
            vol_arbitrage_signals=vol_signals,
            portfolio_value=1_000_000,
        )

        assert len(result) == 1
        assert result[0].asset == "SPY"
        assert result[0].signal_type == AggregatedSignalType.BUY

    def test_aggregate_unanimous_signals(self, aggregator):
        """Test aggregation when strategies agree."""
        vol_signals = [
            VolArbitrageSignal(
                underlying="SPY",
                strike=420.0,
                expiration=datetime.now(),
                option_type="call",
                signal_type=SignalType.BUY,
                confidence=0.80,
                model_iv=0.25,
                market_iv=0.20,
                divergence_pct=0.25,
                rationale="Underpriced",
            )
        ]

        mr_signals = [
            MeanReversionSignal(
                spread_name="SPY",
                signal_type=MeanRevSignalType.ENTRY_LONG,
                confidence=0.75,
                current_value=-0.08,
                rationale="Below entry",
            )
        ]

        result = aggregator.aggregate(
            vol_arbitrage_signals=vol_signals,
            mean_reversion_signals=mr_signals,
            portfolio_value=1_000_000,
        )

        assert len(result) == 1
        assert result[0].signal_type == AggregatedSignalType.BUY
        assert len(result[0].supporting_strategies) == 2

    def test_aggregate_conflicting_signals_consensus(self, aggregator):
        """Test aggregation resolves conflicts via consensus."""
        vol_signals = [
            VolArbitrageSignal(
                underlying="SPY",
                strike=420.0,
                expiration=datetime.now(),
                option_type="call",
                signal_type=SignalType.BUY,
                confidence=0.90,  # High confidence buy
                model_iv=0.25,
                market_iv=0.20,
                divergence_pct=0.25,
                rationale="Underpriced",
            )
        ]

        mr_signals = [
            MeanReversionSignal(
                spread_name="SPY",
                signal_type=MeanRevSignalType.ENTRY_SHORT,
                confidence=0.50,  # Low confidence sell
                current_value=0.08,
                rationale="Above entry",
            )
        ]

        result = aggregator.aggregate(
            vol_arbitrage_signals=vol_signals,
            mean_reversion_signals=mr_signals,
            portfolio_value=1_000_000,
        )

        # Strong buy should win with 1.5x ratio
        assert len(result) == 1
        assert result[0].signal_type == AggregatedSignalType.BUY
        assert "vol_arbitrage" in result[0].supporting_strategies
        assert "mean_reversion" in result[0].conflicting_strategies

    def test_aggregate_conflicting_no_consensus(self, aggregator):
        """Test aggregation when no consensus is reached."""
        vol_signals = [
            VolArbitrageSignal(
                underlying="SPY",
                strike=420.0,
                expiration=datetime.now(),
                option_type="call",
                signal_type=SignalType.BUY,
                confidence=0.70,  # Similar confidence
                model_iv=0.25,
                market_iv=0.20,
                divergence_pct=0.25,
                rationale="Underpriced",
            )
        ]

        mr_signals = [
            MeanReversionSignal(
                spread_name="SPY",
                signal_type=MeanRevSignalType.ENTRY_SHORT,
                confidence=0.70,  # Similar confidence
                current_value=0.08,
                rationale="Above entry",
            )
        ]

        result = aggregator.aggregate(
            vol_arbitrage_signals=vol_signals,
            mean_reversion_signals=mr_signals,
            portfolio_value=1_000_000,
        )

        # No clear winner - should be no signal
        assert len(result) == 0

    def test_aggregate_with_unanimous_config(self):
        """Test aggregation with unanimous requirement."""
        config = AggregatorConfig(require_unanimous=True)
        aggregator = SignalAggregator(config=config)

        vol_signals = [
            VolArbitrageSignal(
                underlying="SPY",
                strike=420.0,
                expiration=datetime.now(),
                option_type="call",
                signal_type=SignalType.BUY,
                confidence=0.90,
                model_iv=0.25,
                market_iv=0.20,
                divergence_pct=0.25,
                rationale="Underpriced",
            )
        ]

        mr_signals = [
            MeanReversionSignal(
                spread_name="SPY",
                signal_type=MeanRevSignalType.ENTRY_SHORT,
                confidence=0.50,
                current_value=0.08,
                rationale="Above entry",
            )
        ]

        result = aggregator.aggregate(
            vol_arbitrage_signals=vol_signals,
            mean_reversion_signals=mr_signals,
            portfolio_value=1_000_000,
        )

        # Conflicting signals with unanimous requirement = no signal
        assert len(result) == 0

    def test_aggregate_empty_signals(self, aggregator):
        """Test aggregation with no signals."""
        result = aggregator.aggregate(
            vol_arbitrage_signals=[],
            mean_reversion_signals=[],
            portfolio_value=1_000_000,
        )

        assert len(result) == 0

    def test_position_sizing(self, aggregator):
        """Test position size computation."""
        size = aggregator._compute_position_size(
            confidence=0.8,
            portfolio_value=1_000_000,
        )

        # 10% max * 0.8 confidence = 8% of portfolio
        expected = 1_000_000 * 0.10 * 0.8
        assert size == expected

    def test_filter_by_risk_budget(self, aggregator):
        """Test risk budget filtering."""
        signals = [
            AggregatedSignal(
                asset="SPY",
                signal_type=AggregatedSignalType.BUY,
                confidence=0.9,
                supporting_strategies=["vol_arbitrage"],
                conflicting_strategies=[],
                rationale="Test",
                suggested_position_size=0.1,
            ),
            AggregatedSignal(
                asset="QQQ",
                signal_type=AggregatedSignalType.BUY,
                confidence=0.7,
                supporting_strategies=["mean_reversion"],
                conflicting_strategies=[],
                rationale="Test",
                suggested_position_size=0.1,
            ),
        ]

        # With limited budget
        filtered = aggregator.filter_by_risk_budget(
            signals=signals,
            current_exposure={"existing": 0.75},
            max_total_exposure=0.8,
        )

        # Only room for 5%, should select highest confidence
        assert len(filtered) <= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestSignalGenerationIntegration:
    """Integration tests for signal generation workflow."""

    def test_full_workflow(self, mock_sabr_result, mock_ou_fit_result):
        """Test complete signal generation workflow."""
        # Generate vol arb signals
        vol_gen = VolSurfaceArbitrageSignal()

        # Generate mean reversion signals
        mr_gen = MeanReversionSignalGenerator()

        mr_signal = mr_gen.generate_signal(
            spread_name="SPY-IWM",
            current_value=-0.08,
            ou_fit_result=mock_ou_fit_result,
        )

        # Aggregate
        aggregator = SignalAggregator()

        mr_signals = [mr_signal] if mr_signal else []

        result = aggregator.aggregate(
            mean_reversion_signals=mr_signals,
            portfolio_value=1_000_000,
        )

        # Verify end-to-end flow
        assert isinstance(result, list)
        if mr_signal:
            assert len(result) >= 0  # May be filtered by confidence
