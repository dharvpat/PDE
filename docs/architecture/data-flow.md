# Data Flow Architecture

## Overview

This document describes how data flows through the Quantitative Trading System from external sources to trading decisions.

## Data Sources

### 1. Market Data
- **Source**: Real-time market data feeds (CBOE, exchanges)
- **Frequency**: Real-time streaming, 1-minute bars
- **Data Types**: Prices, volumes, bid-ask spreads

### 2. Options Data
- **Source**: Options market data providers (IVolatility, CBOE)
- **Frequency**: Every 15 minutes during market hours
- **Data Types**: Options chains, implied volatilities, Greeks

### 3. Reference Data
- **Source**: Financial data providers
- **Frequency**: Daily updates
- **Data Types**: Company fundamentals, sector classifications

## Data Pipeline Stages

```
Stage 1: Collection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

External APIs  ──────►  Collectors  ──────►  Raw Data Queue
                        │
                        ├── MarketDataCollector
                        ├── OptionsChainCollector
                        └── ReferenceDataCollector


Stage 2: Validation & Enrichment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Raw Data Queue  ──────►  Validators  ──────►  Enriched Data Queue
                         │
                         ├── Schema Validation
                         ├── Quality Checks
                         ├── Anomaly Detection
                         └── Data Enrichment


Stage 3: Storage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Enriched Data  ──────►  Storage Writers
                        │
                        ├── TimescaleDB (historical)
                        ├── Redis (real-time cache)
                        └── PostgreSQL (reference)


Stage 4: Processing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stored Data  ──────►  Calibrators  ──────►  Model Parameters
                      │
                      ├── HestonCalibrator
                      ├── SABRCalibrator
                      └── OUFitter


Stage 5: Signal Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Params  ──────►  Signal Generators  ──────►  Trading Signals
                       │
                       ├── VolatilityArbitrageSignal
                       ├── MeanReversionSignal
                       └── TermStructureSignal


Stage 6: Risk & Execution
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Signals  ──────►  Risk Manager  ──────►  Position Sizer  ──────►  Execution
                  │                      │
                  ├── VaR Check          ├── Vol Scaling
                  ├── Exposure Check     ├── Kelly Bounds
                  └── Drawdown Check     └── Sector Limits
```

## Detailed Data Flows

### Market Data Flow

```python
# Pseudocode for market data flow

# 1. Collection
async def collect_market_data():
    """
    Collect real-time market data from providers.
    """
    async for tick in market_data_stream:
        # Basic validation
        if not validate_tick(tick):
            log.warning(f"Invalid tick: {tick}")
            continue

        # Enqueue for processing
        await raw_data_queue.put(tick)

# 2. Validation
async def validate_market_data():
    """
    Validate and enrich market data.
    """
    while True:
        tick = await raw_data_queue.get()

        # Schema validation
        validated = MarketDataSchema.validate(tick)

        # Quality checks
        if is_anomaly(validated):
            flag_for_review(validated)

        # Enrichment
        enriched = enrich_with_metadata(validated)

        await enriched_data_queue.put(enriched)

# 3. Storage
async def store_market_data():
    """
    Store validated data to TimescaleDB and Redis.
    """
    while True:
        data = await enriched_data_queue.get()

        # Batch insert to TimescaleDB
        await timescale.insert_market_data(data)

        # Update Redis cache
        await redis.set_latest_price(data.symbol, data.price)
```

### Options Chain Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Options API  │────►│  Collector   │────►│  Validator   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  TimescaleDB │◄────│   Storage    │
                     │              │     └──────────────┘
                     │ options_chain│            │
                     │ implied_vols │            │
                     └──────────────┘            ▼
                                          ┌──────────────┐
                                          │    Redis     │
                                          │              │
                                          │ latest_chain │
                                          │ current_ivs  │
                                          └──────────────┘
```

### Calibration Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Calibration Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Data                                                     │
│  ━━━━━━━━━━                                                    │
│  • Options chain from TimescaleDB                               │
│  • Current spot price from Redis                                │
│  • Risk-free rate from reference data                           │
│  • Dividend yield                                               │
│                                                                 │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Data Preparation                        │       │
│  │  • Filter liquid options (bid-ask spread < 5%)       │       │
│  │  • Remove deep ITM/OTM options                       │       │
│  │  • Select maturities (30d-365d)                      │       │
│  │  • Calculate moneyness                               │       │
│  └───────────────────────┬─────────────────────────────┘       │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Model Calibration                       │       │
│  │                                                      │       │
│  │  Heston:                                             │       │
│  │  • Global optimization (differential evolution)      │       │
│  │  • Local refinement (Levenberg-Marquardt)            │       │
│  │  • Constraint: Feller condition                      │       │
│  │                                                      │       │
│  │  SABR:                                               │       │
│  │  • Asymptotic formula fitting                        │       │
│  │  • Per-maturity calibration                          │       │
│  │  • Constraint: no-arbitrage                          │       │
│  └───────────────────────┬─────────────────────────────┘       │
│                          │                                     │
│                          ▼                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Validation                              │       │
│  │  • Parameter bounds check                            │       │
│  │  • Fit quality metrics (RMSE, R²)                    │       │
│  │  • Stability check vs previous day                   │       │
│  │  • Feller condition verification                     │       │
│  └───────────────────────┬─────────────────────────────┘       │
│                          │                                     │
│                          ▼                                     │
│  Output Data                                                   │
│  ━━━━━━━━━━━                                                  │
│  • Model parameters → Redis (current)                          │
│  • Parameter history → TimescaleDB                             │
│  • Calibration metrics → Prometheus                            │
│  • Model IV surface → Redis                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Signal Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 Signal Generation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │               Input Data Sources                        │    │
│  │                                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ Model Params │  │ Market Data  │  │ Risk Limits  │  │    │
│  │  │ (Redis)      │  │ (TimescaleDB)│  │ (PostgreSQL) │  │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │    │
│  └─────────┼─────────────────┼─────────────────┼──────────┘    │
│            │                 │                 │                │
│            └─────────────────┼─────────────────┘                │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Signal Generators                       │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Volatility Arbitrage Signal                     │    │   │
│  │  │  • Compare model IV vs market IV                 │    │   │
│  │  │  • Signal if deviation > threshold               │    │   │
│  │  │  • Confidence = f(fit quality, liquidity)        │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Mean Reversion Signal                           │    │   │
│  │  │  • Check spread vs optimal boundaries            │    │   │
│  │  │  • Entry if spread ∈ (L*, U*)                    │    │   │
│  │  │  • Exit if take-profit or stop-loss hit          │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │                                                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │  Term Structure Signal                           │    │   │
│  │  │  • Analyze vol term structure slope              │    │   │
│  │  │  • Detect misalignments                          │    │   │
│  │  │  • Generate calendar spread signals              │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Signal Aggregation                       │   │
│  │  • Weight by confidence score                            │   │
│  │  • Resolve conflicts (same asset, different signals)     │   │
│  │  • Filter by minimum confidence (>0.6)                   │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Output                                │   │
│  │  • Signals → TimescaleDB (history)                       │   │
│  │  • Active signals → Redis                                │   │
│  │  • Notifications → RabbitMQ                              │   │
│  │  • Metrics → Prometheus                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Latency Requirements

| Stage | Latency Target | Notes |
|-------|---------------|-------|
| Market data collection | <100ms | Real-time streaming |
| Data validation | <50ms | Per-tick processing |
| Redis cache update | <10ms | In-memory |
| TimescaleDB insert | <100ms | Batch optimized |
| Calibration (Heston) | <30s | Full recalibration |
| Calibration (SABR) | <1s | Per smile |
| Signal generation | <5s | All signals |
| Risk checks | <1s | Pre-trade |
| Order submission | <500ms | To broker |

## Data Retention Policy

| Data Type | Hot Storage | Warm Storage | Cold Storage |
|-----------|-------------|--------------|--------------|
| Tick data | 7 days | 90 days | 5 years |
| Options chains | 30 days | 1 year | 5 years |
| Model parameters | 90 days | 5 years | Forever |
| Signals | 1 year | 5 years | Forever |
| Executions | 5 years | Forever | Forever |
| Audit logs | 5 years | Forever | Forever |

## Error Handling

### Data Collection Failures
```
┌────────────────────────────────────────────────────┐
│              Error Recovery Flow                    │
├────────────────────────────────────────────────────┤
│                                                    │
│  API Error  ──────►  Retry (3x, exponential)       │
│                              │                     │
│                              ▼                     │
│                         Success?                   │
│                         /      \                   │
│                       Yes       No                 │
│                        │         │                 │
│                        ▼         ▼                 │
│                   Continue    Use cached           │
│                              / fallback            │
│                                  │                 │
│                                  ▼                 │
│                           Alert if stale          │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Calibration Failures
- If calibration fails: use cached parameters from previous day
- If Feller condition violated: penalize or constrain parameters
- Alert on consecutive failures

### Signal Generation Failures
- Graceful degradation: disable problematic signal type
- Continue with remaining signals
- Alert for manual review
