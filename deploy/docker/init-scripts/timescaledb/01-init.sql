-- TimescaleDB Initialization Script
-- Creates database schema for Quantitative Trading System
--
-- Reference: Section 5.4 of design-doc.md (TimescaleDB Schema)

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- Market Data Tables
-- =============================================================================

-- OHLCV price data
CREATE TABLE IF NOT EXISTS market_data (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    vwap        DOUBLE PRECISION,
    source      TEXT DEFAULT 'primary'
);

SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data (symbol, time DESC);

-- Options chain data
CREATE TABLE IF NOT EXISTS options_data (
    time            TIMESTAMPTZ NOT NULL,
    underlying      TEXT NOT NULL,
    strike          DOUBLE PRECISION NOT NULL,
    expiration      DATE NOT NULL,
    option_type     TEXT NOT NULL,  -- 'call' or 'put'
    bid             DOUBLE PRECISION,
    ask             DOUBLE PRECISION,
    last            DOUBLE PRECISION,
    volume          INTEGER,
    open_interest   INTEGER,
    implied_vol     DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    gamma           DOUBLE PRECISION,
    theta           DOUBLE PRECISION,
    vega            DOUBLE PRECISION
);

SELECT create_hypertable('options_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_options_underlying ON options_data (underlying, expiration, time DESC);

-- =============================================================================
-- Calibration Results Tables
-- =============================================================================

-- Heston model parameters
CREATE TABLE IF NOT EXISTS heston_params (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    kappa       DOUBLE PRECISION NOT NULL,
    theta       DOUBLE PRECISION NOT NULL,
    sigma       DOUBLE PRECISION NOT NULL,
    rho         DOUBLE PRECISION NOT NULL,
    v0          DOUBLE PRECISION NOT NULL,
    rmse        DOUBLE PRECISION,
    r_squared   DOUBLE PRECISION,
    feller_satisfied BOOLEAN
);

SELECT create_hypertable('heston_params', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_heston_symbol ON heston_params (symbol, time DESC);

-- SABR model parameters
CREATE TABLE IF NOT EXISTS sabr_params (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    expiry      DATE NOT NULL,
    alpha       DOUBLE PRECISION NOT NULL,
    beta        DOUBLE PRECISION NOT NULL,
    rho         DOUBLE PRECISION NOT NULL,
    nu          DOUBLE PRECISION NOT NULL,
    rmse        DOUBLE PRECISION,
    r_squared   DOUBLE PRECISION
);

SELECT create_hypertable('sabr_params', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_sabr_symbol_expiry ON sabr_params (symbol, expiry, time DESC);

-- OU process parameters
CREATE TABLE IF NOT EXISTS ou_params (
    time        TIMESTAMPTZ NOT NULL,
    pair_id     TEXT NOT NULL,
    symbol_1    TEXT NOT NULL,
    symbol_2    TEXT NOT NULL,
    theta       DOUBLE PRECISION NOT NULL,
    mu          DOUBLE PRECISION NOT NULL,
    sigma       DOUBLE PRECISION NOT NULL,
    half_life   DOUBLE PRECISION,
    adf_stat    DOUBLE PRECISION,
    adf_pvalue  DOUBLE PRECISION
);

SELECT create_hypertable('ou_params', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ou_pair ON ou_params (pair_id, time DESC);

-- =============================================================================
-- Signal Tables
-- =============================================================================

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    time            TIMESTAMPTZ NOT NULL,
    signal_id       UUID DEFAULT gen_random_uuid(),
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    direction       TEXT NOT NULL,  -- 'long', 'short', 'neutral'
    strength        DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    target_price    DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    expires_at      TIMESTAMPTZ,
    metadata        JSONB
);

SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals (strategy, time DESC);

-- =============================================================================
-- Position & Trade Tables
-- =============================================================================

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    time            TIMESTAMPTZ NOT NULL,
    position_id     UUID DEFAULT gen_random_uuid(),
    symbol          TEXT NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    strategy        TEXT,
    status          TEXT DEFAULT 'open'
);

SELECT create_hypertable('positions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status, time DESC);

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    time            TIMESTAMPTZ NOT NULL,
    trade_id        UUID DEFAULT gen_random_uuid(),
    order_id        UUID,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,  -- 'buy' or 'sell'
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION DEFAULT 0,
    slippage        DOUBLE PRECISION DEFAULT 0,
    strategy        TEXT,
    execution_venue TEXT
);

SELECT create_hypertable('trades', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_order ON trades (order_id);

-- =============================================================================
-- Risk & Performance Tables
-- =============================================================================

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    nav             DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    margin_used     DOUBLE PRECISION DEFAULT 0,
    daily_pnl       DOUBLE PRECISION,
    total_pnl       DOUBLE PRECISION,
    drawdown        DOUBLE PRECISION,
    sharpe_ratio    DOUBLE PRECISION,
    var_95          DOUBLE PRECISION,
    var_99          DOUBLE PRECISION
);

SELECT create_hypertable('portfolio_snapshots', 'time', if_not_exists => TRUE);

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT,
    metric_name     TEXT NOT NULL,
    metric_value    DOUBLE PRECISION NOT NULL,
    threshold       DOUBLE PRECISION,
    breach          BOOLEAN DEFAULT FALSE
);

SELECT create_hypertable('risk_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_risk_metric_name ON risk_metrics (metric_name, time DESC);

-- =============================================================================
-- Audit & Logging Tables
-- =============================================================================

-- System events
CREATE TABLE IF NOT EXISTS system_events (
    time            TIMESTAMPTZ NOT NULL,
    event_id        UUID DEFAULT gen_random_uuid(),
    service         TEXT NOT NULL,
    level           TEXT NOT NULL,
    message         TEXT NOT NULL,
    metadata        JSONB
);

SELECT create_hypertable('system_events', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_events_service ON system_events (service, time DESC);
CREATE INDEX IF NOT EXISTS idx_events_level ON system_events (level, time DESC);

-- Alerts
CREATE TABLE IF NOT EXISTS alerts (
    time            TIMESTAMPTZ NOT NULL,
    alert_id        UUID DEFAULT gen_random_uuid(),
    severity        TEXT NOT NULL,
    category        TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    acknowledged    BOOLEAN DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

SELECT create_hypertable('alerts', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_category ON alerts (category, time DESC);

-- =============================================================================
-- Data Retention Policies
-- =============================================================================

-- Keep raw market data for 2 years
SELECT add_retention_policy('market_data', INTERVAL '2 years', if_not_exists => TRUE);

-- Keep options data for 1 year
SELECT add_retention_policy('options_data', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep calibration params for 5 years (model audit trail)
SELECT add_retention_policy('heston_params', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('sabr_params', INTERVAL '5 years', if_not_exists => TRUE);
SELECT add_retention_policy('ou_params', INTERVAL '5 years', if_not_exists => TRUE);

-- Keep signals for 2 years
SELECT add_retention_policy('signals', INTERVAL '2 years', if_not_exists => TRUE);

-- Keep trades and positions indefinitely (regulatory requirement)

-- Keep system events for 90 days
SELECT add_retention_policy('system_events', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep alerts for 1 year
SELECT add_retention_policy('alerts', INTERVAL '1 year', if_not_exists => TRUE);

-- =============================================================================
-- Continuous Aggregates (for performance)
-- =============================================================================

-- Hourly market data aggregation
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(vwap) AS vwap
FROM market_data
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('market_data_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Daily portfolio performance
CREATE MATERIALIZED VIEW IF NOT EXISTS portfolio_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    last(nav, time) AS nav,
    last(cash, time) AS cash,
    sum(daily_pnl) AS daily_pnl,
    max(drawdown) AS max_drawdown,
    avg(sharpe_ratio) AS avg_sharpe
FROM portfolio_snapshots
GROUP BY bucket
WITH NO DATA;

SELECT add_continuous_aggregate_policy('portfolio_daily',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quant;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quant;
