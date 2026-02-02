"""
Integration tests for database module.

Tests ORM models, database access layer, and query patterns.
Uses SQLite in-memory database for unit tests and optionally
PostgreSQL/TimescaleDB for integration tests.

Run with pytest:
    pytest tests/python/database/test_database.py -v

For PostgreSQL tests (requires running database):
    QUANT_DB_URL=postgresql://user:pass@localhost/test_db pytest tests/python/database/ -v
"""

import os
import sys
from datetime import datetime, timedelta, date
from decimal import Decimal
import uuid

import pytest

# Add src to path for development
sys.path.insert(0, "src/python")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quant_trading.database.models import (
    Base,
    MarketPrice,
    OptionQuote,
    ModelParameter,
    Signal,
    Position,
    PositionUpdate,
)
from quant_trading.database.db import TimeSeriesDB


# Use SQLite for basic tests, PostgreSQL for full integration
TEST_DATABASE_URL = os.environ.get(
    "QUANT_TEST_DB_URL",
    "sqlite:///:memory:",
)

# Check if we have PostgreSQL available
POSTGRES_AVAILABLE = TEST_DATABASE_URL.startswith("postgresql")


@pytest.fixture(scope="function")
def db_engine():
    """Create a test database engine."""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="function")
def timeseries_db(db_engine):
    """Create a TimeSeriesDB instance for testing."""
    db = TimeSeriesDB(TEST_DATABASE_URL)
    db.create_tables()
    yield db
    db.drop_tables()


class TestMarketPriceModel:
    """Tests for MarketPrice ORM model."""

    def test_create_market_price(self, db_session):
        """Test creating a market price record."""
        price = MarketPrice(
            time=datetime.utcnow(),
            symbol="SPY",
            price=Decimal("450.25"),
            volume=1000000,
            bid=Decimal("450.20"),
            ask=Decimal("450.30"),
            exchange="NYSE",
            data_quality="good",
        )
        db_session.add(price)
        db_session.commit()

        result = db_session.query(MarketPrice).filter_by(symbol="SPY").first()
        assert result is not None
        assert result.symbol == "SPY"
        assert float(result.price) == 450.25
        assert result.volume == 1000000

    def test_market_price_mid_price(self, db_session):
        """Test mid price calculation."""
        price = MarketPrice(
            time=datetime.utcnow(),
            symbol="SPY",
            price=Decimal("450.00"),
            bid=Decimal("449.90"),
            ask=Decimal("450.10"),
        )
        db_session.add(price)
        db_session.commit()

        assert price.mid_price == 450.00

    def test_market_price_spread(self, db_session):
        """Test spread calculation."""
        price = MarketPrice(
            time=datetime.utcnow(),
            symbol="SPY",
            price=Decimal("450.00"),
            bid=Decimal("449.90"),
            ask=Decimal("450.10"),
        )
        db_session.add(price)
        db_session.commit()

        assert price.spread == 0.20

    def test_market_price_to_dict(self, db_session):
        """Test to_dict conversion."""
        now = datetime.utcnow()
        price = MarketPrice(
            time=now,
            symbol="AAPL",
            price=Decimal("175.50"),
            volume=5000000,
        )
        db_session.add(price)
        db_session.commit()

        d = price.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["price"] == 175.50
        assert d["time"] == now


class TestOptionQuoteModel:
    """Tests for OptionQuote ORM model."""

    def test_create_option_quote(self, db_session):
        """Test creating an option quote record."""
        quote = OptionQuote(
            time=datetime.utcnow(),
            underlying="SPY",
            expiration=date(2026, 3, 20),
            strike=Decimal("450.00"),
            option_type="call",
            bid=Decimal("5.50"),
            ask=Decimal("5.70"),
            implied_vol=Decimal("0.2100"),
            delta=Decimal("0.5200"),
            gamma=Decimal("0.015000"),
            vega=Decimal("0.3500"),
            theta=Decimal("-0.1200"),
        )
        db_session.add(quote)
        db_session.commit()

        result = (
            db_session.query(OptionQuote)
            .filter_by(underlying="SPY", strike=Decimal("450.00"))
            .first()
        )
        assert result is not None
        assert result.option_type == "call"
        assert float(result.implied_vol) == 0.21

    def test_option_quote_mid_price(self, db_session):
        """Test option mid price calculation."""
        quote = OptionQuote(
            time=datetime.utcnow(),
            underlying="SPY",
            expiration=date(2026, 3, 20),
            strike=Decimal("450.00"),
            option_type="call",
            bid=Decimal("5.50"),
            ask=Decimal("5.70"),
        )
        db_session.add(quote)
        db_session.commit()

        assert quote.mid_price == 5.60

    def test_option_quote_is_call(self, db_session):
        """Test is_call property."""
        call = OptionQuote(
            time=datetime.utcnow(),
            underlying="SPY",
            expiration=date(2026, 3, 20),
            strike=Decimal("450.00"),
            option_type="call",
        )
        put = OptionQuote(
            time=datetime.utcnow(),
            underlying="SPY",
            expiration=date(2026, 3, 20),
            strike=Decimal("440.00"),
            option_type="put",
        )

        assert call.is_call is True
        assert put.is_call is False


class TestModelParameterModel:
    """Tests for ModelParameter ORM model."""

    def test_create_heston_parameters(self, db_session):
        """Test creating Heston model parameters."""
        # Use a placeholder date for models without maturity (SQLite doesn't allow NULL in PK)
        params = ModelParameter(
            time=datetime.utcnow(),
            model_type="heston",
            underlying="SPY",
            maturity=date(1970, 1, 1),  # Placeholder for no maturity
            parameters={
                "kappa": 2.0,
                "theta": 0.04,
                "sigma": 0.3,
                "rho": -0.7,
                "v0": 0.04,
            },
            fit_quality={
                "rmse": 0.025,
                "r_squared": 0.98,
                "feller_satisfied": True,
            },
            converged=True,
            calibration_time_ms=1500,
        )
        db_session.add(params)
        db_session.commit()

        result = (
            db_session.query(ModelParameter)
            .filter_by(model_type="heston", underlying="SPY")
            .first()
        )
        assert result is not None
        assert result.get_param("kappa") == 2.0
        assert result.get_fit_metric("rmse") == 0.025

    def test_create_sabr_parameters(self, db_session):
        """Test creating SABR model parameters."""
        params = ModelParameter(
            time=datetime.utcnow(),
            model_type="sabr",
            underlying="SPY",
            maturity=date(2026, 3, 20),
            parameters={
                "alpha": 0.25,
                "beta": 0.5,
                "rho": -0.3,
                "nu": 0.4,
            },
            fit_quality={
                "rmse": 0.015,
                "r_squared": 0.99,
            },
            converged=True,
        )
        db_session.add(params)
        db_session.commit()

        result = (
            db_session.query(ModelParameter)
            .filter_by(model_type="sabr", underlying="SPY")
            .first()
        )
        assert result is not None
        sabr_params = result.to_sabr_params()
        assert sabr_params["alpha"] == 0.25
        assert sabr_params["beta"] == 0.5

    def test_create_ou_parameters(self, db_session):
        """Test creating OU model parameters."""
        # Use a placeholder date for models without maturity (SQLite doesn't allow NULL in PK)
        params = ModelParameter(
            time=datetime.utcnow(),
            model_type="ou",
            underlying="SPY-QQQ",
            maturity=date(1970, 1, 1),  # Placeholder for no maturity
            parameters={
                "theta": 0.0,
                "mu": 5.0,
                "sigma": 0.1,
            },
            fit_quality={
                "log_likelihood": 500.0,
                "aic": -994.0,
                "bic": -988.0,
            },
            converged=True,
        )
        db_session.add(params)
        db_session.commit()

        result = (
            db_session.query(ModelParameter)
            .filter_by(model_type="ou")
            .first()
        )
        assert result is not None
        ou_params = result.to_ou_params()
        assert ou_params["mu"] == 5.0

    def test_is_valid_fit(self, db_session):
        """Test is_valid_fit property."""
        good_fit = ModelParameter(
            time=datetime.utcnow(),
            model_type="heston",
            underlying="SPY",
            parameters={"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04},
            fit_quality={"rmse": 0.02},
            converged=True,
        )
        bad_fit = ModelParameter(
            time=datetime.utcnow(),
            model_type="heston",
            underlying="AAPL",
            parameters={"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04},
            fit_quality={"rmse": 0.15},
            converged=True,
        )
        not_converged = ModelParameter(
            time=datetime.utcnow(),
            model_type="heston",
            underlying="MSFT",
            parameters={"kappa": 2.0, "theta": 0.04, "sigma": 0.3, "rho": -0.7, "v0": 0.04},
            fit_quality={"rmse": 0.02},
            converged=False,
        )

        assert good_fit.is_valid_fit is True
        assert bad_fit.is_valid_fit is False
        assert not_converged.is_valid_fit is False


class TestSignalModel:
    """Tests for Signal ORM model."""

    def test_create_signal(self, db_session):
        """Test creating a trading signal."""
        signal = Signal(
            time=datetime.utcnow(),
            strategy="vol_arb",
            underlying="SPY",
            signal_type="entry_long",
            signal_strength=Decimal("0.750"),
            rationale="IV underpriced by 2 vol points",
            expected_return=Decimal("0.0250"),
            expected_risk=Decimal("0.0100"),
            signal_metadata={"strike": 450, "expiration": "2026-03-20"},
        )
        db_session.add(signal)
        db_session.commit()

        result = (
            db_session.query(Signal)
            .filter_by(strategy="vol_arb")
            .first()
        )
        assert result is not None
        assert float(result.signal_strength) == 0.750
        assert result.is_entry_signal is True
        assert result.is_actionable is True

    def test_signal_properties(self, db_session):
        """Test signal properties."""
        signal = Signal(
            time=datetime.utcnow(),
            strategy="mean_reversion",
            underlying="SPY-QQQ",
            signal_type="exit",
            signal_strength=Decimal("0.500"),
            expected_return=Decimal("0.05"),
            expected_risk=Decimal("0.02"),
        )

        assert signal.is_entry_signal is False
        assert signal.is_exit_signal is True
        assert signal.is_actionable is False  # strength < 0.6
        assert signal.expected_sharpe == 2.5  # 0.05 / 0.02


class TestPositionModel:
    """Tests for Position ORM model."""

    def test_create_position(self, db_session):
        """Test creating a position."""
        position = Position(
            opened_at=datetime.utcnow(),
            strategy="vol_arb",
            underlying="SPY",
            direction="long",
            quantity=Decimal("100.00"),
            entry_price=Decimal("450.25"),
            entry_commission=Decimal("1.00"),
        )
        db_session.add(position)
        db_session.commit()

        result = db_session.query(Position).first()
        assert result is not None
        assert result.is_open is True
        assert result.is_long is True
        assert result.direction == "long"

    def test_position_pnl_calculation(self, db_session):
        """Test PnL calculation."""
        position = Position(
            opened_at=datetime.utcnow() - timedelta(days=5),
            strategy="mean_reversion",
            underlying="SPY-QQQ",
            direction="long",
            quantity=Decimal("100.00"),
            entry_price=Decimal("1.50"),
            entry_commission=Decimal("2.00"),
        )
        db_session.add(position)
        db_session.commit()

        # Calculate unrealized PnL
        unrealized = position.calculate_unrealized_pnl(1.75)
        assert unrealized == 25.0  # (1.75 - 1.50) * 100

        # Check total PnL (unrealized only since position is open)
        position.unrealized_pnl = Decimal(str(unrealized))
        assert position.total_pnl == 25.0

    def test_close_position(self, db_session):
        """Test closing a position."""
        position = Position(
            opened_at=datetime.utcnow() - timedelta(days=10),
            strategy="vol_arb",
            underlying="SPY",
            direction="long",
            quantity=Decimal("50.00"),
            entry_price=Decimal("450.00"),
            entry_commission=Decimal("1.00"),
        )
        db_session.add(position)
        db_session.commit()

        # Close the position
        position.closed_at = datetime.utcnow()
        position.exit_price = Decimal("455.00")
        position.exit_commission = Decimal("1.00")
        position.realized_pnl = Decimal("248.00")  # (455-450)*50 - 2 commission
        db_session.commit()

        assert position.is_open is False
        assert float(position.total_pnl) == 248.00
        assert position.holding_period_days is not None
        assert position.holding_period_days >= 10

    def test_position_return_pct(self, db_session):
        """Test return percentage calculation."""
        position = Position(
            opened_at=datetime.utcnow(),
            strategy="test",
            underlying="TEST",
            direction="long",
            quantity=Decimal("100.00"),
            entry_price=Decimal("10.00"),
            realized_pnl=Decimal("100.00"),  # 10% return
        )

        assert position.return_pct == 10.0


class TestTimeSeriesDB:
    """Tests for TimeSeriesDB access layer."""

    def test_health_check(self, timeseries_db):
        """Test database health check."""
        health = timeseries_db.health_check()
        assert health["status"] == "healthy"
        assert "latency_ms" in health

    def test_insert_and_get_market_prices(self, timeseries_db):
        """Test inserting and retrieving market prices."""
        now = datetime.utcnow()
        prices = [
            {
                "time": now - timedelta(hours=2),
                "symbol": "SPY",
                "price": 450.00,
                "volume": 1000000,
            },
            {
                "time": now - timedelta(hours=1),
                "symbol": "SPY",
                "price": 450.50,
                "volume": 1500000,
            },
            {
                "time": now,
                "symbol": "SPY",
                "price": 451.00,
                "volume": 2000000,
            },
        ]

        count = timeseries_db.insert_market_prices(prices)
        assert count == 3

        # Retrieve prices
        df = timeseries_db.get_market_prices(
            "SPY",
            start_time=now - timedelta(hours=3),
        )
        assert len(df) == 3
        assert df.iloc[-1]["price"] == 451.00

    def test_get_latest_price(self, timeseries_db):
        """Test getting latest price."""
        now = datetime.utcnow()
        prices = [
            {"time": now - timedelta(minutes=5), "symbol": "AAPL", "price": 175.00},
            {"time": now, "symbol": "AAPL", "price": 175.50},
        ]
        timeseries_db.insert_market_prices(prices)

        latest = timeseries_db.get_latest_price("AAPL")
        assert latest is not None
        assert latest["price"] == 175.50

    def test_store_and_get_model_parameters(self, timeseries_db):
        """Test storing and retrieving model parameters."""
        timeseries_db.store_model_parameters(
            model_type="heston",
            underlying="SPY",
            parameters={
                "kappa": 2.0,
                "theta": 0.04,
                "sigma": 0.3,
                "rho": -0.7,
                "v0": 0.04,
            },
            fit_quality={
                "rmse": 0.025,
                "r_squared": 0.98,
            },
            converged=True,
            calibration_time_ms=1500,
        )

        result = timeseries_db.get_latest_model_parameters(
            model_type="heston",
            underlying="SPY",
        )
        assert result is not None
        assert result["parameters"]["kappa"] == 2.0
        assert result["converged"] is True

    def test_insert_and_get_signals(self, timeseries_db):
        """Test signal insertion and retrieval."""
        timeseries_db.insert_signal(
            strategy="vol_arb",
            signal_type="entry_long",
            signal_strength=0.75,
            underlying="SPY",
            rationale="Test signal",
        )

        signals = timeseries_db.get_latest_signals(
            strategy="vol_arb",
            lookback_minutes=60,
        )
        assert len(signals) == 1
        assert signals[0]["signal_strength"] == 0.75

    def test_get_actionable_signals(self, timeseries_db):
        """Test getting actionable signals."""
        # Insert high-confidence signal
        timeseries_db.insert_signal(
            strategy="vol_arb",
            signal_type="entry_long",
            signal_strength=0.85,
            underlying="SPY",
        )
        # Insert low-confidence signal
        timeseries_db.insert_signal(
            strategy="mean_reversion",
            signal_type="entry_short",
            signal_strength=0.40,
            underlying="QQQ",
        )

        actionable = timeseries_db.get_actionable_signals(min_strength=0.6)
        assert len(actionable) == 1
        assert actionable[0]["underlying"] == "SPY"

    def test_position_lifecycle(self, timeseries_db):
        """Test full position lifecycle: create, update, close."""
        # Create position
        position_id = timeseries_db.create_position(
            strategy="mean_reversion",
            underlying="SPY-QQQ",
            direction="long",
            quantity=100.0,
            entry_price=1.50,
            entry_commission=1.00,
        )
        assert position_id is not None

        # Verify open positions
        open_positions = timeseries_db.get_open_positions(strategy="mean_reversion")
        assert len(open_positions) == 1
        assert open_positions[0]["is_open"] is True

        # Update price
        unrealized = timeseries_db.update_position_price(position_id, 1.75)
        assert unrealized == 25.0  # (1.75 - 1.50) * 100

        # Close position
        realized_pnl = timeseries_db.close_position(
            position_id=position_id,
            exit_price=1.80,
            exit_commission=1.00,
        )
        # PnL = (1.80 - 1.50) * 100 - 2 (commissions) = 28
        assert abs(realized_pnl - 28.0) < 0.01  # Allow floating point tolerance

        # Verify position is closed
        open_positions = timeseries_db.get_open_positions(strategy="mean_reversion")
        assert len(open_positions) == 0

    def test_get_positions_summary(self, timeseries_db):
        """Test position summary statistics."""
        # Create and close some positions
        for i in range(5):
            pid = timeseries_db.create_position(
                strategy="test_strategy",
                underlying=f"TEST{i}",
                direction="long",
                quantity=100.0,
                entry_price=10.0,
            )
            if i < 3:  # Close 3 positions
                pnl = 10.0 if i < 2 else -5.0  # 2 winners, 1 loser
                timeseries_db.close_position(pid, exit_price=10.0 + pnl / 100)

        summary = timeseries_db.get_positions_summary(strategy="test_strategy")
        assert summary["total_positions"] == 5
        assert summary["open_positions"] == 2
        assert summary["closed_positions"] == 3
        # Win rate should be 2/3 = 0.666...
        assert 0.65 < summary["win_rate"] < 0.68

    def test_close_already_closed_position(self, timeseries_db):
        """Test error when closing already closed position."""
        position_id = timeseries_db.create_position(
            strategy="test",
            underlying="TEST",
            direction="long",
            quantity=100.0,
            entry_price=10.0,
        )
        timeseries_db.close_position(position_id, exit_price=11.0)

        with pytest.raises(ValueError, match="already closed"):
            timeseries_db.close_position(position_id, exit_price=12.0)

    def test_close_nonexistent_position(self, timeseries_db):
        """Test error when closing nonexistent position."""
        fake_id = str(uuid.uuid4())
        with pytest.raises(ValueError, match="not found"):
            timeseries_db.close_position(fake_id, exit_price=10.0)


class TestIntegrationPostgres:
    """
    PostgreSQL/TimescaleDB specific tests.

    These tests only run when a PostgreSQL database is available.
    Set QUANT_TEST_DB_URL environment variable to enable.
    """

    @pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available",
    )
    def test_timescaledb_extension(self, timeseries_db):
        """Test that TimescaleDB extension is available."""
        result = timeseries_db.execute_raw_sql(
            "SELECT default_version FROM pg_available_extensions WHERE name='timescaledb'"
        )
        assert len(result) > 0
        assert result[0][0] is not None

    @pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available",
    )
    def test_hypertable_created(self, timeseries_db):
        """Test that hypertables are created correctly."""
        result = timeseries_db.execute_raw_sql(
            "SELECT hypertable_name FROM timescaledb_information.hypertables"
        )
        hypertables = [r[0] for r in result]
        assert "market_prices" in hypertables

    @pytest.mark.skipif(
        not POSTGRES_AVAILABLE,
        reason="PostgreSQL not available",
    )
    def test_bulk_insert_performance(self, timeseries_db):
        """Test bulk insert performance target (>10,000 rows/sec)."""
        import time

        now = datetime.utcnow()
        n_rows = 10000
        prices = [
            {
                "time": now + timedelta(seconds=i),
                "symbol": "SPY",
                "price": 450.00 + i * 0.01,
                "volume": 1000 + i,
            }
            for i in range(n_rows)
        ]

        start = time.time()
        timeseries_db.insert_market_prices(prices)
        elapsed = time.time() - start

        rate = n_rows / elapsed
        print(f"Bulk insert rate: {rate:.0f} rows/sec")
        assert rate > 5000  # Conservative threshold for CI environments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
