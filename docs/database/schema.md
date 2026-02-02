# Database Schema Documentation

## Overview

The Quantitative Trading System uses a multi-database architecture:
- **TimescaleDB**: Time-series data (market data, signals, executions)
- **PostgreSQL**: Reference data, configurations, audit logs
- **Redis**: Caching, real-time data

## TimescaleDB Schema

TimescaleDB is a PostgreSQL extension optimized for time-series data. Tables are organized as hypertables with automatic partitioning by time.

### Market Data Tables

#### market_data
Stores OHLCV price data for all tracked symbols.

```sql
CREATE TABLE market_data (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT,
    vwap        DOUBLE PRECISION,
    bid         DOUBLE PRECISION,
    ask         DOUBLE PRECISION,
    bid_size    INTEGER,
    ask_size    INTEGER,

    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('market_data', 'time');

-- Create indexes
CREATE INDEX idx_market_data_symbol ON market_data (symbol, time DESC);
```

#### options_chains
Stores options market data including implied volatilities and Greeks.

```sql
CREATE TABLE options_chains (
    time            TIMESTAMPTZ NOT NULL,
    underlying      TEXT NOT NULL,
    expiration      DATE NOT NULL,
    strike          DOUBLE PRECISION NOT NULL,
    option_type     TEXT NOT NULL,  -- 'call' or 'put'
    bid             DOUBLE PRECISION,
    ask             DOUBLE PRECISION,
    last_price      DOUBLE PRECISION,
    volume          INTEGER,
    open_interest   INTEGER,
    implied_vol     DOUBLE PRECISION,
    delta           DOUBLE PRECISION,
    gamma           DOUBLE PRECISION,
    theta           DOUBLE PRECISION,
    vega            DOUBLE PRECISION,
    rho             DOUBLE PRECISION,

    PRIMARY KEY (time, underlying, expiration, strike, option_type)
);

SELECT create_hypertable('options_chains', 'time');

CREATE INDEX idx_options_underlying ON options_chains (underlying, time DESC);
CREATE INDEX idx_options_expiration ON options_chains (expiration);
```

### Signal Tables

#### signals
Stores all generated trading signals.

```sql
CREATE TABLE signals (
    signal_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time            TIMESTAMPTZ NOT NULL,
    strategy_id     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    signal_type     TEXT NOT NULL,  -- 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT'
    strength        DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    target_price    DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    expiration      TIMESTAMPTZ,
    metadata        JSONB,
    status          TEXT DEFAULT 'active',  -- 'active', 'executed', 'expired', 'cancelled'

    CONSTRAINT valid_strength CHECK (strength >= 0 AND strength <= 1),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

SELECT create_hypertable('signals', 'time');

CREATE INDEX idx_signals_strategy ON signals (strategy_id, time DESC);
CREATE INDEX idx_signals_symbol ON signals (symbol, time DESC);
CREATE INDEX idx_signals_active ON signals (status) WHERE status = 'active';
```

### Model Parameters Tables

#### heston_parameters
Stores calibrated Heston model parameters.

```sql
CREATE TABLE heston_parameters (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    kappa           DOUBLE PRECISION NOT NULL,
    theta           DOUBLE PRECISION NOT NULL,
    sigma           DOUBLE PRECISION NOT NULL,
    rho             DOUBLE PRECISION NOT NULL,
    v0              DOUBLE PRECISION NOT NULL,
    feller_satisfied BOOLEAN NOT NULL,
    rmse            DOUBLE PRECISION,
    r_squared       DOUBLE PRECISION,
    n_options       INTEGER,
    calibration_time_ms INTEGER,

    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('heston_parameters', 'time');
CREATE INDEX idx_heston_symbol ON heston_parameters (symbol, time DESC);
```

#### sabr_parameters
Stores calibrated SABR model parameters per maturity.

```sql
CREATE TABLE sabr_parameters (
    time            TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    maturity        DOUBLE PRECISION NOT NULL,  -- Time to expiration in years
    alpha           DOUBLE PRECISION NOT NULL,
    beta            DOUBLE PRECISION NOT NULL,
    rho             DOUBLE PRECISION NOT NULL,
    nu              DOUBLE PRECISION NOT NULL,
    rmse            DOUBLE PRECISION,
    n_options       INTEGER,

    PRIMARY KEY (time, symbol, maturity)
);

SELECT create_hypertable('sabr_parameters', 'time');
CREATE INDEX idx_sabr_symbol ON sabr_parameters (symbol, time DESC);
```

#### ou_parameters
Stores fitted OU process parameters for pairs/spreads.

```sql
CREATE TABLE ou_parameters (
    time            TIMESTAMPTZ NOT NULL,
    pair_id         TEXT NOT NULL,
    asset1          TEXT NOT NULL,
    asset2          TEXT NOT NULL,
    hedge_ratio     DOUBLE PRECISION NOT NULL,
    theta           DOUBLE PRECISION NOT NULL,
    mu              DOUBLE PRECISION NOT NULL,
    sigma           DOUBLE PRECISION NOT NULL,
    half_life       DOUBLE PRECISION NOT NULL,
    adf_statistic   DOUBLE PRECISION,
    adf_pvalue      DOUBLE PRECISION,
    entry_lower     DOUBLE PRECISION,
    entry_upper     DOUBLE PRECISION,
    exit_target     DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,

    PRIMARY KEY (time, pair_id)
);

SELECT create_hypertable('ou_parameters', 'time');
CREATE INDEX idx_ou_pair ON ou_parameters (pair_id, time DESC);
```

### Execution Tables

#### orders
Stores all order records.

```sql
CREATE TABLE orders (
    order_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy_id     TEXT,
    signal_id       UUID REFERENCES signals(signal_id),
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,  -- 'buy', 'sell'
    order_type      TEXT NOT NULL,  -- 'market', 'limit', 'stop', 'stop_limit'
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    time_in_force   TEXT NOT NULL DEFAULT 'day',
    status          TEXT NOT NULL DEFAULT 'pending',
    filled_quantity DOUBLE PRECISION DEFAULT 0,
    avg_fill_price  DOUBLE PRECISION,
    broker_order_id TEXT,
    error_message   TEXT
);

SELECT create_hypertable('orders', 'created_at');

CREATE INDEX idx_orders_strategy ON orders (strategy_id, created_at DESC);
CREATE INDEX idx_orders_symbol ON orders (symbol, created_at DESC);
CREATE INDEX idx_orders_status ON orders (status) WHERE status IN ('pending', 'open');
```

#### executions
Stores individual execution/fill records.

```sql
CREATE TABLE executions (
    execution_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time            TIMESTAMPTZ NOT NULL,
    order_id        UUID REFERENCES orders(order_id),
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION DEFAULT 0,
    venue           TEXT,
    broker_exec_id  TEXT
);

SELECT create_hypertable('executions', 'time');

CREATE INDEX idx_executions_order ON executions (order_id);
CREATE INDEX idx_executions_symbol ON executions (symbol, time DESC);
```

### Portfolio Tables

#### portfolio_snapshots
Stores periodic portfolio snapshots.

```sql
CREATE TABLE portfolio_snapshots (
    time            TIMESTAMPTZ NOT NULL,
    total_value     DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    positions_value DOUBLE PRECISION NOT NULL,
    total_pnl       DOUBLE PRECISION NOT NULL,
    unrealized_pnl  DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION NOT NULL,
    total_exposure  DOUBLE PRECISION NOT NULL,
    net_exposure    DOUBLE PRECISION NOT NULL,
    n_positions     INTEGER NOT NULL,

    PRIMARY KEY (time)
);

SELECT create_hypertable('portfolio_snapshots', 'time');
```

#### positions
Stores current position records (non-time-series).

```sql
CREATE TABLE positions (
    symbol          TEXT PRIMARY KEY,
    quantity        DOUBLE PRECISION NOT NULL,
    average_cost    DOUBLE PRECISION NOT NULL,
    realized_pnl    DOUBLE PRECISION DEFAULT 0,
    strategy_id     TEXT,
    opened_at       TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

## PostgreSQL Schema (Reference Data)

### Strategies

```sql
CREATE TABLE strategies (
    strategy_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    type            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'stopped',
    parameters      JSONB NOT NULL DEFAULT '{}',
    risk_limits     JSONB NOT NULL DEFAULT '{}',
    symbols         TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by      TEXT NOT NULL
);

CREATE INDEX idx_strategies_status ON strategies (status);
CREATE INDEX idx_strategies_type ON strategies (type);
```

### Users and Permissions

```sql
CREATE TABLE users (
    user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login      TIMESTAMPTZ
);

CREATE TABLE permissions (
    permission_id   SERIAL PRIMARY KEY,
    role            TEXT NOT NULL,
    resource        TEXT NOT NULL,
    action          TEXT NOT NULL,

    UNIQUE(role, resource, action)
);

-- Default permissions
INSERT INTO permissions (role, resource, action) VALUES
    ('admin', '*', '*'),
    ('trader', 'strategies', 'read'),
    ('trader', 'strategies', 'update'),
    ('trader', 'signals', 'read'),
    ('trader', 'orders', '*'),
    ('trader', 'portfolio', 'read'),
    ('viewer', 'strategies', 'read'),
    ('viewer', 'signals', 'read'),
    ('viewer', 'portfolio', 'read');
```

### Configuration

```sql
CREATE TABLE configurations (
    key             TEXT PRIMARY KEY,
    value           JSONB NOT NULL,
    description     TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by      TEXT
);
```

### Audit Log

```sql
CREATE TABLE audit_logs (
    log_id          BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id         UUID REFERENCES users(user_id),
    action          TEXT NOT NULL,
    resource_type   TEXT NOT NULL,
    resource_id     TEXT,
    old_value       JSONB,
    new_value       JSONB,
    ip_address      INET,
    user_agent      TEXT
);

CREATE INDEX idx_audit_timestamp ON audit_logs (timestamp DESC);
CREATE INDEX idx_audit_user ON audit_logs (user_id, timestamp DESC);
CREATE INDEX idx_audit_resource ON audit_logs (resource_type, resource_id);
```

## Data Retention Policies

### TimescaleDB Compression & Retention

```sql
-- Enable compression on older data
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Compress data older than 7 days
SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Retention policies
SELECT add_retention_policy('market_data', INTERVAL '5 years');
SELECT add_retention_policy('options_chains', INTERVAL '5 years');
SELECT add_retention_policy('signals', INTERVAL '5 years');
SELECT add_retention_policy('portfolio_snapshots', INTERVAL '10 years');
```

### Continuous Aggregates

Pre-computed daily summaries for fast queries:

```sql
CREATE MATERIALIZED VIEW market_data_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    first(open, time) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, time) AS close,
    sum(volume) AS volume,
    avg(close) AS avg_price
FROM market_data
GROUP BY day, symbol;

-- Refresh policy
SELECT add_continuous_aggregate_policy('market_data_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);
```

## Indexes Summary

| Table | Index | Purpose |
|-------|-------|---------|
| market_data | (symbol, time DESC) | Symbol lookup |
| options_chains | (underlying, time DESC) | Options lookup |
| signals | (strategy_id, time DESC) | Strategy signals |
| signals | (status) WHERE active | Active signals |
| orders | (status) WHERE pending/open | Pending orders |
| executions | (order_id) | Order fills |
| audit_logs | (timestamp DESC) | Recent activity |

## Migrations

Migrations are managed using Alembic. See `migrations/` directory.

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Backup & Recovery

See [backup-restore.md](./backup-restore.md) for detailed procedures.

```bash
# Full backup
pg_dump -Fc trading_db > backup.dump

# TimescaleDB-specific backup
pg_dump -Fc -t 'market_data*' -t 'signals' trading_db > timeseries_backup.dump

# Restore
pg_restore -d trading_db backup.dump
```
