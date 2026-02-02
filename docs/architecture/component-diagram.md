# Component Diagram

## High-Level Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              External Services                                   │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Market Data    │  Options Data   │  Broker APIs    │  Research       │ Alerts  │
│  (CBOE, etc.)   │  (IVolatility)  │  (IBKR, etc.)   │  (Factor Data)  │ (Slack) │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬────────┴────┬────┘
         │                 │                 │                 │             │
         ▼                 ▼                 ▼                 ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                API Gateway (NGINX)                               │
│   Rate Limiting │ Authentication │ Load Balancing │ TLS Termination             │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   REST API      │       │  WebSocket API  │       │  Admin API      │
│   Service       │       │  Service        │       │  Service        │
│   (FastAPI)     │       │  (FastAPI)      │       │  (FastAPI)      │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            Core Services Layer                                   │
├─────────────────┬─────────────────┬─────────────────┬─────────────────┬─────────┤
│  Data Ingestion │  Calibration    │  Signal         │  Risk           │Execution│
│  Service        │  Service        │  Generation     │  Management     │ Service │
│                 │  (High CPU)     │  Service        │  Service        │         │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬────────┴────┬────┘
         │                 │                 │                 │             │
         ▼                 ▼                 ▼                 ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Message Queue (RabbitMQ)                            │
│   calibration.tasks │ signals.updates │ orders.execute │ alerts.send            │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                 │                 │                 │
         ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Data Layer                                         │
├─────────────────────────────┬─────────────────┬─────────────────────────────────┤
│       TimescaleDB           │     Redis       │       PostgreSQL                 │
│   (Time-series data)        │   (Cache)       │   (Reference data)               │
│                             │                 │                                  │
│   • market_data             │   • latest_     │   • strategies                   │
│   • options_chains          │     prices      │   • configurations               │
│   • signals                 │   • model_      │   • users                        │
│   • executions              │     params      │   • permissions                  │
│   • portfolio_snapshots     │   • signals     │   • audit_logs                   │
└─────────────────────────────┴─────────────────┴─────────────────────────────────┘
```

## Service Components Detail

### 1. Data Ingestion Service

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Ingestion Service                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Market Data │  │ Options     │  │ Reference   │         │
│  │ Collector   │  │ Collector   │  │ Collector   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Validation Layer                    │       │
│  │  • Schema validation                             │       │
│  │  • Data quality checks                           │       │
│  │  • Duplicate detection                           │       │
│  │  • Anomaly flagging                              │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Storage Layer                       │       │
│  │  • Batch inserts to TimescaleDB                  │       │
│  │  • Cache updates to Redis                        │       │
│  │  • Metric publishing                             │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2. Calibration Service

```
┌─────────────────────────────────────────────────────────────┐
│                   Calibration Service                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Heston    │  │    SABR     │  │    O-U      │         │
│  │  Calibrator │  │  Calibrator │  │   Fitter    │         │
│  │  (C++ Core) │  │  (C++ Core) │  │  (C++ Core) │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌─────────────────────────────────────────────────┐       │
│  │            Parameter Validation                  │       │
│  │  • Feller condition check                        │       │
│  │  • Parameter bounds validation                   │       │
│  │  • Stability checks                              │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │            Parameter Storage                     │       │
│  │  • Redis (current parameters)                    │       │
│  │  • TimescaleDB (historical)                      │       │
│  │  • Metric publishing                             │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Performance Requirements:
• Heston: <30 seconds for 50 options
• SABR: <1 second per smile
• OU: <1 second for 500 data points
```

### 3. Signal Generation Service

```
┌─────────────────────────────────────────────────────────────┐
│                Signal Generation Service                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │            Signal Generators                     │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │       │
│  │  │ Vol Arb  │  │ Mean Rev │  │ Term Str │      │       │
│  │  │ Signal   │  │ Signal   │  │ Signal   │      │       │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘      │       │
│  └───────┼─────────────┼─────────────┼────────────┘       │
│          │             │             │                     │
│          ▼             ▼             ▼                     │
│  ┌─────────────────────────────────────────────────┐       │
│  │            Signal Aggregator                     │       │
│  │  • Confidence weighting                          │       │
│  │  • Conflict resolution                           │       │
│  │  • Regime detection                              │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │            Signal Filter                         │       │
│  │  • Liquidity filter                              │       │
│  │  • Confidence threshold (>0.6)                   │       │
│  │  • Risk limit filter                             │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│               Final Trading Signals                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Risk Management Service

```
┌─────────────────────────────────────────────────────────────┐
│                Risk Management Service                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Position   │  │   Greeks    │  │  Drawdown   │         │
│  │   Sizer     │  │  Calculator │  │  Controller │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │
│         │    ┌───────────┴───────────┐    │                │
│         │    │                       │    │                │
│         ▼    ▼                       ▼    ▼                │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Risk Aggregator                     │       │
│  │  • Portfolio VaR calculation                     │       │
│  │  • Correlation monitoring                        │       │
│  │  • Exposure limits                               │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Position Sizing                     │       │
│  │  • Vol-scaled sizing: w_t = c/σ_t²              │       │
│  │  • Kelly criterion bounds                        │       │
│  │  • Sector/asset limits                           │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5. Execution Service

```
┌─────────────────────────────────────────────────────────────┐
│                   Execution Service                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Order Management                    │       │
│  │  • Order creation & validation                   │       │
│  │  • Order state machine                           │       │
│  │  • Fill tracking                                 │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Smart Order Router                  │       │
│  │  • Venue selection                               │       │
│  │  • Order slicing (TWAP/VWAP)                     │       │
│  │  • Cost minimization                             │       │
│  └───────────────────────┬─────────────────────────┘       │
│                          │                                 │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────┐       │
│  │              Broker Adapters                     │       │
│  │  ┌────────┐  ┌────────┐  ┌────────┐            │       │
│  │  │  IBKR  │  │  Paper │  │  Sim   │            │       │
│  │  │Adapter │  │Trading │  │ Mode   │            │       │
│  │  └────────┘  └────────┘  └────────┘            │       │
│  └─────────────────────────────────────────────────┘       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Monitoring Components

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Monitoring Stack                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│   │   Prometheus    │    │     Grafana     │    │   AlertManager  │            │
│   │                 │◄───│                 │    │                 │            │
│   │  Metric Store   │    │   Dashboards    │    │  Alert Routing  │            │
│   └────────┬────────┘    └─────────────────┘    └────────┬────────┘            │
│            │                                             │                     │
│            │                                             ▼                     │
│            │                                    ┌─────────────────┐            │
│            │                                    │  Notifications  │            │
│            │                                    │  (Slack/Email)  │            │
│            │                                    └─────────────────┘            │
│            │                                                                   │
│            ▼                                                                   │
│   ┌─────────────────────────────────────────────────────────────────┐         │
│   │                    Service Metrics                               │         │
│   │  • API latency          • Calibration time                       │         │
│   │  • Request count        • Signal count                           │         │
│   │  • Error rate           • Execution fill rate                    │         │
│   │  • Cache hit rate       • Model fit quality                      │         │
│   └─────────────────────────────────────────────────────────────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Inter-Service Communication

### Synchronous (REST/gRPC)
- API Gateway → Services
- Service → Service (health checks)
- Admin operations

### Asynchronous (RabbitMQ)
- Calibration tasks
- Signal broadcasts
- Order execution
- Alerts

### Pub/Sub (Redis)
- Real-time price updates
- Parameter changes
- Signal notifications

## Component Dependencies

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
    ┌───────────┐     ┌───────────┐     ┌───────────┐
    │  REST API │     │ WebSocket │     │ Admin API │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────┐          ┌───────────┐          ┌─────────────┐
│  Data   │◄────────►│Calibration│◄────────►│   Signal    │
│Ingestion│          │  Service  │          │ Generation  │
└────┬────┘          └─────┬─────┘          └──────┬──────┘
     │                     │                       │
     │                     └───────────┬───────────┘
     │                                 │
     ▼                                 ▼
┌─────────────┐                 ┌─────────────┐
│  TimescaleDB│◄───────────────►│    Risk     │
│    Redis    │                 │ Management  │
└─────────────┘                 └──────┬──────┘
                                       │
                                       ▼
                                ┌─────────────┐
                                │  Execution  │
                                │   Service   │
                                └─────────────┘
```

## Container Organization

| Service | Image | CPU Request | Memory Request | Replicas |
|---------|-------|-------------|----------------|----------|
| api | quant-trading/api | 500m | 512Mi | 2-5 |
| websocket | quant-trading/api | 250m | 256Mi | 2-3 |
| calibration | quant-trading/calibration | 2000m | 4Gi | 1-3 |
| data-ingestion | quant-trading/data-ingestion | 500m | 512Mi | 1-2 |
| signals | quant-trading/signals | 1000m | 1Gi | 1-2 |
| risk | quant-trading/risk | 500m | 512Mi | 1 |
| execution | quant-trading/execution | 500m | 512Mi | 1 |
