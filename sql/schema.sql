-- TimescaleDB Schema for Quantitative Trading System
-- This file contains the complete schema definition including:
-- - Market prices (tick data with hypertable)
-- - Option quotes (options chain with Greeks)
-- - Model parameters (calibrated Heston/SABR/OU params)
-- - Trading signals
-- - Position tracking with PnL

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =============================================================================
-- MARKET PRICES TABLE
-- Stores equity price data with TimescaleDB hypertable optimization
-- =============================================================================

CREATE TABLE market_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC(12, 4) NOT NULL CHECK (price > 0),
    volume BIGINT CHECK (volume >= 0),
    bid NUMERIC(12, 4) CHECK (bid > 0),
    ask NUMERIC(12, 4) CHECK (ask > 0),

    -- Metadata
    exchange TEXT,
    data_quality TEXT DEFAULT 'good' CHECK (data_quality IN ('good', 'suspect', 'bad')),

    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable (1-day chunks)
SELECT create_hypertable(
    'market_prices',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create index on symbol for fast symbol-based queries
CREATE INDEX idx_market_prices_symbol_time
ON market_prices (symbol, time DESC);

-- Add compression policy (compress data older than 7 days)
ALTER TABLE market_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('market_prices', INTERVAL '7 days');

-- Add data retention policy (keep 5 years)
SELECT add_retention_policy('market_prices', INTERVAL '5 years');

-- =============================================================================
-- MARKET PRICES CONTINUOUS AGGREGATES
-- Pre-computed OHLCV bars for efficient queries
-- =============================================================================

-- 1-minute OHLCV bars
CREATE MATERIALIZED VIEW market_prices_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume
FROM market_prices
GROUP BY bucket, symbol;

-- Refresh policy (update every 1 minute)
SELECT add_continuous_aggregate_policy('market_prices_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute');

-- 5-minute OHLCV bars
CREATE MATERIALIZED VIEW market_prices_5min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume
FROM market_prices
GROUP BY bucket, symbol;

SELECT add_continuous_aggregate_policy('market_prices_5min',
    start_offset => INTERVAL '4 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes');

-- Daily OHLCV bars
CREATE MATERIALIZED VIEW market_prices_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    symbol,
    FIRST(price, time) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, time) AS close,
    SUM(volume) AS volume
FROM market_prices
GROUP BY bucket, symbol;

SELECT add_continuous_aggregate_policy('market_prices_daily',
    start_offset => INTERVAL '1 week',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- =============================================================================
-- OPTION QUOTES TABLE
-- Options chain data including Greeks and implied volatility
-- =============================================================================

CREATE TABLE option_quotes (
    time TIMESTAMPTZ NOT NULL,
    underlying TEXT NOT NULL,
    expiration DATE NOT NULL,
    strike NUMERIC(10, 2) NOT NULL CHECK (strike > 0),
    option_type TEXT NOT NULL CHECK (option_type IN ('call', 'put')),

    -- Prices
    bid NUMERIC(10, 4) CHECK (bid >= 0),
    ask NUMERIC(10, 4) CHECK (ask >= 0),
    mid NUMERIC(10, 4) GENERATED ALWAYS AS ((bid + ask) / 2) STORED,
    last NUMERIC(10, 4),
    volume INTEGER CHECK (volume >= 0),
    open_interest INTEGER CHECK (open_interest >= 0),

    -- Greeks
    implied_vol NUMERIC(6, 4) CHECK (implied_vol >= 0 AND implied_vol <= 5.0),
    delta NUMERIC(6, 4) CHECK (delta >= -1 AND delta <= 1),
    gamma NUMERIC(8, 6) CHECK (gamma >= 0),
    vega NUMERIC(8, 4) CHECK (vega >= 0),
    theta NUMERIC(8, 4),
    rho NUMERIC(8, 4),

    -- Data quality
    bid_ask_spread NUMERIC(6, 4) GENERATED ALWAYS AS (
        CASE WHEN (bid + ask) / 2 > 0 THEN (ask - bid) / ((bid + ask) / 2) ELSE NULL END
    ) STORED,

    PRIMARY KEY (time, underlying, expiration, strike, option_type)
);

-- Convert to hypertable
SELECT create_hypertable(
    'option_quotes',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Composite index for calibration queries
-- Query pattern: "Get all options for SPY expiring 2026-03-20 at time T"
CREATE INDEX idx_option_quotes_calibration
ON option_quotes (underlying, expiration, time DESC, strike);

-- Index for strike/type queries
CREATE INDEX idx_option_quotes_strike
ON option_quotes (underlying, time DESC, strike, option_type);

-- Compression policy
ALTER TABLE option_quotes SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'underlying, expiration, strike, option_type'
);

SELECT add_compression_policy('option_quotes', INTERVAL '7 days');

-- Retention policy (keep 2 years of options data)
SELECT add_retention_policy('option_quotes', INTERVAL '2 years');

-- =============================================================================
-- LATEST OPTION QUOTES VIEW
-- Materialized view for latest options by underlying
-- =============================================================================

CREATE MATERIALIZED VIEW latest_option_quotes AS
SELECT DISTINCT ON (underlying, expiration, strike, option_type)
    time,
    underlying,
    expiration,
    strike,
    option_type,
    bid,
    ask,
    mid,
    implied_vol,
    delta,
    gamma,
    vega
FROM option_quotes
ORDER BY underlying, expiration, strike, option_type, time DESC;

CREATE INDEX idx_latest_option_quotes_lookup
ON latest_option_quotes (underlying, expiration);

-- =============================================================================
-- MODEL PARAMETERS TABLE
-- Stores calibrated model parameters (Heston, SABR, OU)
-- =============================================================================

CREATE TABLE model_parameters (
    time TIMESTAMPTZ NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('heston', 'sabr', 'ou')),
    underlying TEXT NOT NULL,
    maturity DATE,  -- NULL for OU (not maturity-specific)

    -- Parameters stored as JSONB for flexibility
    parameters JSONB NOT NULL,

    -- Fit quality metrics
    fit_quality JSONB NOT NULL,

    -- Metadata
    calibration_time_ms INTEGER,  -- How long calibration took
    n_iterations INTEGER,
    converged BOOLEAN,

    PRIMARY KEY (time, model_type, underlying, COALESCE(maturity, '1970-01-01'))
);

-- Hypertable with 1-day chunks
SELECT create_hypertable(
    'model_parameters',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Index for latest parameters query
CREATE INDEX idx_model_params_latest
ON model_parameters (model_type, underlying, maturity, time DESC);

-- JSONB indexes for parameter queries
CREATE INDEX idx_model_params_parameters
ON model_parameters USING GIN (parameters);

CREATE INDEX idx_model_params_fit_quality
ON model_parameters USING GIN (fit_quality);

-- Check constraints on Heston parameter values
ALTER TABLE model_parameters ADD CONSTRAINT check_heston_params
CHECK (
    model_type != 'heston' OR (
        (parameters->>'kappa')::float > 0 AND
        (parameters->>'theta')::float > 0 AND
        (parameters->>'sigma')::float > 0 AND
        (parameters->>'rho')::float > -1 AND
        (parameters->>'rho')::float < 1 AND
        (parameters->>'v0')::float > 0
    )
);

-- Check constraints on SABR parameter values
ALTER TABLE model_parameters ADD CONSTRAINT check_sabr_params
CHECK (
    model_type != 'sabr' OR (
        (parameters->>'alpha')::float > 0 AND
        (parameters->>'beta')::float >= 0 AND
        (parameters->>'beta')::float <= 1 AND
        (parameters->>'rho')::float > -1 AND
        (parameters->>'rho')::float < 1 AND
        (parameters->>'nu')::float >= 0
    )
);

-- Check constraints on OU parameter values
ALTER TABLE model_parameters ADD CONSTRAINT check_ou_params
CHECK (
    model_type != 'ou' OR (
        (parameters->>'mu')::float > 0 AND
        (parameters->>'sigma')::float > 0
    )
);

-- =============================================================================
-- LATEST MODEL PARAMETERS VIEW
-- Materialized view for latest parameters per model/underlying
-- =============================================================================

CREATE MATERIALIZED VIEW latest_model_parameters AS
SELECT DISTINCT ON (model_type, underlying, maturity)
    time,
    model_type,
    underlying,
    maturity,
    parameters,
    fit_quality,
    converged
FROM model_parameters
ORDER BY model_type, underlying, maturity, time DESC;

CREATE INDEX idx_latest_model_params_lookup
ON latest_model_parameters (model_type, underlying, maturity);

-- =============================================================================
-- SIGNALS TABLE
-- Trading signals generated by strategies
-- =============================================================================

CREATE TABLE signals (
    time TIMESTAMPTZ NOT NULL,
    strategy TEXT NOT NULL,  -- 'vol_arb', 'mean_reversion', 'term_structure'
    underlying TEXT,         -- Asset or spread identifier
    signal_type TEXT NOT NULL CHECK (
        signal_type IN ('entry_long', 'entry_short', 'exit', 'hold', 'reduce')
    ),
    signal_strength NUMERIC(4, 3) NOT NULL CHECK (
        signal_strength >= 0.0 AND signal_strength <= 1.0
    ),

    -- Strategy-specific metadata
    metadata JSONB,

    -- Signal details
    rationale TEXT,          -- Human-readable explanation
    expected_return NUMERIC(8, 4),
    expected_risk NUMERIC(8, 4),

    PRIMARY KEY (time, strategy, COALESCE(underlying, ''))
);

-- Hypertable
SELECT create_hypertable(
    'signals',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes
CREATE INDEX idx_signals_strategy_time
ON signals (strategy, time DESC);

CREATE INDEX idx_signals_underlying_time
ON signals (underlying, time DESC);

CREATE INDEX idx_signals_type
ON signals (signal_type, time DESC);

-- JSONB index for metadata queries
CREATE INDEX idx_signals_metadata
ON signals USING GIN (metadata);

-- Retention (keep 1 year)
SELECT add_retention_policy('signals', INTERVAL '1 year');

-- =============================================================================
-- POSITIONS TABLE
-- Trading positions with PnL tracking
-- =============================================================================

CREATE TABLE positions (
    position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Timestamps
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Position details
    strategy TEXT NOT NULL,
    underlying TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    quantity NUMERIC(12, 2) NOT NULL,

    -- Pricing
    entry_price NUMERIC(12, 4) NOT NULL,
    exit_price NUMERIC(12, 4),
    current_price NUMERIC(12, 4),

    -- PnL
    realized_pnl NUMERIC(12, 2),
    unrealized_pnl NUMERIC(12, 2),
    total_pnl NUMERIC(12, 2) GENERATED ALWAYS AS (
        COALESCE(realized_pnl, 0) + COALESCE(unrealized_pnl, 0)
    ) STORED,

    -- Transaction costs
    entry_commission NUMERIC(10, 2),
    exit_commission NUMERIC(10, 2),
    total_commission NUMERIC(10, 2) GENERATED ALWAYS AS (
        COALESCE(entry_commission, 0) + COALESCE(exit_commission, 0)
    ) STORED,

    -- Greeks (for options positions)
    delta NUMERIC(8, 4),
    gamma NUMERIC(8, 6),
    vega NUMERIC(8, 4),
    theta NUMERIC(8, 4),

    -- Metadata
    metadata JSONB,

    -- Constraints
    CHECK (closed_at IS NULL OR closed_at >= opened_at),
    CHECK (exit_price IS NULL OR closed_at IS NOT NULL)
);

-- Indexes
CREATE INDEX idx_positions_opened_at ON positions (opened_at DESC);
CREATE INDEX idx_positions_strategy ON positions (strategy, opened_at DESC);
CREATE INDEX idx_positions_underlying ON positions (underlying, opened_at DESC);

-- Index for active positions (most common query)
CREATE INDEX idx_positions_active
ON positions (strategy, underlying)
WHERE closed_at IS NULL;

-- JSONB index for metadata queries
CREATE INDEX idx_positions_metadata
ON positions USING GIN (metadata);

-- =============================================================================
-- POSITION UPDATES AUDIT TABLE
-- Audit trail for position changes
-- =============================================================================

CREATE TABLE position_updates (
    update_id SERIAL PRIMARY KEY,
    position_id UUID NOT NULL REFERENCES positions(position_id),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    field_name TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    updated_by TEXT
);

CREATE INDEX idx_position_updates_position
ON position_updates (position_id, updated_at DESC);

-- =============================================================================
-- DAILY STRATEGY PERFORMANCE CONTINUOUS AGGREGATE
-- Pre-computed performance stats by strategy
-- =============================================================================

-- Note: This requires positions table to have time column for hypertable
-- Using regular materialized view instead since positions doesn't use hypertable
CREATE MATERIALIZED VIEW daily_strategy_performance AS
SELECT
    DATE_TRUNC('day', closed_at) AS day,
    strategy,
    COUNT(*) AS trades_closed,
    SUM(realized_pnl) AS total_pnl,
    AVG(realized_pnl) AS avg_pnl,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS win_rate,
    STDDEV(realized_pnl) AS pnl_volatility
FROM positions
WHERE closed_at IS NOT NULL
GROUP BY day, strategy;

CREATE INDEX idx_daily_strategy_perf_day
ON daily_strategy_performance (day DESC, strategy);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY latest_option_quotes;
    REFRESH MATERIALIZED VIEW CONCURRENTLY latest_model_parameters;
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_strategy_performance;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest model parameters
CREATE OR REPLACE FUNCTION get_latest_params(
    p_model_type TEXT,
    p_underlying TEXT,
    p_maturity DATE DEFAULT NULL
)
RETURNS TABLE (
    time TIMESTAMPTZ,
    parameters JSONB,
    fit_quality JSONB,
    converged BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT mp.time, mp.parameters, mp.fit_quality, mp.converged
    FROM model_parameters mp
    WHERE mp.model_type = p_model_type
      AND mp.underlying = p_underlying
      AND (p_maturity IS NULL AND mp.maturity IS NULL
           OR mp.maturity = p_maturity)
    ORDER BY mp.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate position PnL
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    p_position_id UUID,
    p_current_price NUMERIC
)
RETURNS TABLE (
    unrealized NUMERIC,
    total NUMERIC
) AS $$
DECLARE
    v_entry_price NUMERIC;
    v_quantity NUMERIC;
    v_direction TEXT;
    v_realized NUMERIC;
    v_commission NUMERIC;
BEGIN
    SELECT entry_price, quantity, direction,
           COALESCE(realized_pnl, 0),
           COALESCE(entry_commission, 0) + COALESCE(exit_commission, 0)
    INTO v_entry_price, v_quantity, v_direction, v_realized, v_commission
    FROM positions
    WHERE position_id = p_position_id;

    IF v_direction = 'long' THEN
        unrealized := (p_current_price - v_entry_price) * v_quantity;
    ELSE
        unrealized := (v_entry_price - p_current_price) * v_quantity;
    END IF;

    total := v_realized + unrealized - v_commission;

    RETURN NEXT;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE market_prices IS 'Equity price tick data with TimescaleDB hypertable optimization';
COMMENT ON TABLE option_quotes IS 'Options chain data including Greeks and implied volatility';
COMMENT ON TABLE model_parameters IS 'Calibrated model parameters for Heston, SABR, and OU models';
COMMENT ON TABLE signals IS 'Trading signals generated by strategies';
COMMENT ON TABLE positions IS 'Trading positions with PnL tracking';
COMMENT ON TABLE position_updates IS 'Audit trail for position changes';

COMMENT ON MATERIALIZED VIEW market_prices_1min IS '1-minute OHLCV bars (continuous aggregate)';
COMMENT ON MATERIALIZED VIEW market_prices_5min IS '5-minute OHLCV bars (continuous aggregate)';
COMMENT ON MATERIALIZED VIEW market_prices_daily IS 'Daily OHLCV bars (continuous aggregate)';
COMMENT ON MATERIALIZED VIEW latest_option_quotes IS 'Latest option quotes per underlying/expiration/strike';
COMMENT ON MATERIALIZED VIEW latest_model_parameters IS 'Latest model parameters per model type/underlying';
COMMENT ON MATERIALIZED VIEW daily_strategy_performance IS 'Daily strategy performance statistics';
