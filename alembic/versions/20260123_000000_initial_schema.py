"""Initial schema for quantitative trading database.

Revision ID: 001_initial
Revises:
Create Date: 2026-01-23 00:00:00.000000

This migration creates the complete database schema including:
- market_prices: Equity tick data (TimescaleDB hypertable)
- option_quotes: Options chain with Greeks (TimescaleDB hypertable)
- model_parameters: Calibrated Heston/SABR/OU parameters (TimescaleDB hypertable)
- signals: Trading signals (TimescaleDB hypertable)
- positions: Position tracking with PnL
- position_updates: Audit trail for position changes

Note: TimescaleDB-specific operations (hypertables, compression policies,
continuous aggregates) should be applied separately using sql/schema.sql
after running this migration.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables."""

    # Enable required extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    # ==========================================================================
    # market_prices table
    # ==========================================================================
    op.create_table(
        "market_prices",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(16), nullable=False),
        sa.Column("price", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.Column("bid", sa.Numeric(12, 4), nullable=True),
        sa.Column("ask", sa.Numeric(12, 4), nullable=True),
        sa.Column("exchange", sa.String(16), nullable=True),
        sa.Column("data_quality", sa.String(16), server_default="good", nullable=True),
        sa.PrimaryKeyConstraint("time", "symbol"),
        sa.CheckConstraint("price > 0", name="check_price_positive"),
        sa.CheckConstraint("volume >= 0", name="check_volume_non_negative"),
        sa.CheckConstraint("bid > 0 OR bid IS NULL", name="check_bid_positive"),
        sa.CheckConstraint("ask > 0 OR ask IS NULL", name="check_ask_positive"),
        sa.CheckConstraint(
            "data_quality IN ('good', 'suspect', 'bad')",
            name="check_data_quality_valid",
        ),
    )

    op.create_index(
        "idx_market_prices_symbol_time",
        "market_prices",
        ["symbol", "time"],
    )

    # Convert to hypertable
    op.execute(
        """
        SELECT create_hypertable(
            'market_prices',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
        """
    )

    # ==========================================================================
    # option_quotes table
    # ==========================================================================
    op.create_table(
        "option_quotes",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("underlying", sa.String(16), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("strike", sa.Numeric(10, 2), nullable=False),
        sa.Column("option_type", sa.String(4), nullable=False),
        sa.Column("bid", sa.Numeric(10, 4), nullable=True),
        sa.Column("ask", sa.Numeric(10, 4), nullable=True),
        sa.Column("last", sa.Numeric(10, 4), nullable=True),
        sa.Column("volume", sa.Integer(), nullable=True),
        sa.Column("open_interest", sa.Integer(), nullable=True),
        sa.Column("implied_vol", sa.Numeric(6, 4), nullable=True),
        sa.Column("delta", sa.Numeric(6, 4), nullable=True),
        sa.Column("gamma", sa.Numeric(8, 6), nullable=True),
        sa.Column("vega", sa.Numeric(8, 4), nullable=True),
        sa.Column("theta", sa.Numeric(8, 4), nullable=True),
        sa.Column("rho", sa.Numeric(8, 4), nullable=True),
        sa.PrimaryKeyConstraint(
            "time", "underlying", "expiration", "strike", "option_type"
        ),
        sa.CheckConstraint(
            "option_type IN ('call', 'put')", name="check_option_type"
        ),
        sa.CheckConstraint("strike > 0", name="check_strike_positive"),
        sa.CheckConstraint("bid >= 0 OR bid IS NULL", name="check_bid_non_negative"),
        sa.CheckConstraint("ask >= 0 OR ask IS NULL", name="check_ask_non_negative"),
        sa.CheckConstraint(
            "volume >= 0 OR volume IS NULL", name="check_volume_non_neg"
        ),
        sa.CheckConstraint(
            "open_interest >= 0 OR open_interest IS NULL",
            name="check_oi_non_negative",
        ),
        sa.CheckConstraint(
            "(implied_vol >= 0 AND implied_vol <= 5.0) OR implied_vol IS NULL",
            name="check_iv_range",
        ),
        sa.CheckConstraint(
            "(delta >= -1 AND delta <= 1) OR delta IS NULL",
            name="check_delta_range",
        ),
        sa.CheckConstraint(
            "gamma >= 0 OR gamma IS NULL", name="check_gamma_non_negative"
        ),
    )

    op.create_index(
        "idx_option_quotes_calibration",
        "option_quotes",
        ["underlying", "expiration", "time", "strike"],
    )

    op.create_index(
        "idx_option_quotes_strike",
        "option_quotes",
        ["underlying", "time", "strike", "option_type"],
    )

    # Convert to hypertable
    op.execute(
        """
        SELECT create_hypertable(
            'option_quotes',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
        """
    )

    # ==========================================================================
    # model_parameters table
    # ==========================================================================
    op.create_table(
        "model_parameters",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_type", sa.String(16), nullable=False),
        sa.Column("underlying", sa.String(16), nullable=False),
        sa.Column("maturity", sa.Date(), nullable=True),
        sa.Column("parameters", postgresql.JSONB(), nullable=False),
        sa.Column("fit_quality", postgresql.JSONB(), nullable=False),
        sa.Column("calibration_time_ms", sa.Integer(), nullable=True),
        sa.Column("n_iterations", sa.Integer(), nullable=True),
        sa.Column("converged", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("time", "model_type", "underlying", "maturity"),
        sa.CheckConstraint(
            "model_type IN ('heston', 'sabr', 'ou')",
            name="check_model_type_valid",
        ),
    )

    op.create_index(
        "idx_model_params_latest",
        "model_parameters",
        ["model_type", "underlying", "maturity", "time"],
    )

    # Convert to hypertable
    op.execute(
        """
        SELECT create_hypertable(
            'model_parameters',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
        """
    )

    # ==========================================================================
    # signals table
    # ==========================================================================
    op.create_table(
        "signals",
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("underlying", sa.String(32), nullable=True, server_default=""),
        sa.Column("signal_type", sa.String(16), nullable=False),
        sa.Column("signal_strength", sa.Numeric(4, 3), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("expected_return", sa.Numeric(8, 4), nullable=True),
        sa.Column("expected_risk", sa.Numeric(8, 4), nullable=True),
        sa.PrimaryKeyConstraint("time", "strategy", "underlying"),
        sa.CheckConstraint(
            "signal_type IN ('entry_long', 'entry_short', 'exit', 'hold', 'reduce')",
            name="check_signal_type_valid",
        ),
        sa.CheckConstraint(
            "signal_strength >= 0.0 AND signal_strength <= 1.0",
            name="check_signal_strength_range",
        ),
    )

    op.create_index(
        "idx_signals_strategy_time",
        "signals",
        ["strategy", "time"],
    )

    op.create_index(
        "idx_signals_underlying_time",
        "signals",
        ["underlying", "time"],
    )

    op.create_index(
        "idx_signals_type",
        "signals",
        ["signal_type", "time"],
    )

    # Convert to hypertable
    op.execute(
        """
        SELECT create_hypertable(
            'signals',
            'time',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        )
        """
    )

    # ==========================================================================
    # positions table
    # ==========================================================================
    op.create_table(
        "positions",
        sa.Column(
            "position_id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("strategy", sa.String(32), nullable=False),
        sa.Column("underlying", sa.String(32), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 2), nullable=False),
        sa.Column("entry_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("exit_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("current_price", sa.Numeric(12, 4), nullable=True),
        sa.Column("realized_pnl", sa.Numeric(12, 2), nullable=True),
        sa.Column("unrealized_pnl", sa.Numeric(12, 2), nullable=True),
        sa.Column("entry_commission", sa.Numeric(10, 2), nullable=True),
        sa.Column("exit_commission", sa.Numeric(10, 2), nullable=True),
        sa.Column("delta", sa.Numeric(8, 4), nullable=True),
        sa.Column("gamma", sa.Numeric(8, 6), nullable=True),
        sa.Column("vega", sa.Numeric(8, 4), nullable=True),
        sa.Column("theta", sa.Numeric(8, 4), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("position_id"),
        sa.CheckConstraint(
            "direction IN ('long', 'short')", name="check_direction_valid"
        ),
        sa.CheckConstraint(
            "closed_at IS NULL OR closed_at >= opened_at",
            name="check_close_after_open",
        ),
        sa.CheckConstraint(
            "exit_price IS NULL OR closed_at IS NOT NULL",
            name="check_exit_price_requires_close",
        ),
    )

    op.create_index("idx_positions_opened_at", "positions", ["opened_at"])
    op.create_index(
        "idx_positions_strategy", "positions", ["strategy", "opened_at"]
    )
    op.create_index(
        "idx_positions_underlying", "positions", ["underlying", "opened_at"]
    )

    # Partial index for active positions
    op.execute(
        """
        CREATE INDEX idx_positions_active
        ON positions (strategy, underlying)
        WHERE closed_at IS NULL
        """
    )

    # ==========================================================================
    # position_updates table (audit trail)
    # ==========================================================================
    op.create_table(
        "position_updates",
        sa.Column("update_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("position_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("field_name", sa.String(64), nullable=False),
        sa.Column("old_value", sa.Text(), nullable=True),
        sa.Column("new_value", sa.Text(), nullable=True),
        sa.Column("updated_by", sa.String(64), nullable=True),
        sa.PrimaryKeyConstraint("update_id"),
        sa.ForeignKeyConstraint(
            ["position_id"],
            ["positions.position_id"],
            name="fk_position_updates_position_id",
        ),
    )

    op.create_index(
        "idx_position_updates_position",
        "position_updates",
        ["position_id", "updated_at"],
    )


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("position_updates")
    op.drop_table("positions")
    op.drop_table("signals")
    op.drop_table("model_parameters")
    op.drop_table("option_quotes")
    op.drop_table("market_prices")
