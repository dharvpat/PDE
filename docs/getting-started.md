# Getting Started Guide

Welcome to the Quantitative Trading System! This guide will help you get up and running quickly.

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/company/quant-trading.git
cd quant-trading
```

### 2. Set Up with Docker (Recommended)

```bash
# Start all services
docker-compose -f deploy/docker/docker-compose.yml up -d

# Verify services are running
docker-compose ps

# View API logs
docker-compose logs -f api
```

The API will be available at `http://localhost:8000`.

### 3. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "1.0.0", ...}
```

## Manual Installation

If you prefer to run without Docker, follow these steps:

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with TimescaleDB extension
- Redis 7+
- C++ compiler (GCC 12+ or Clang 15+)
- CMake 3.20+

### Step-by-Step Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 3. Build C++ extensions
python setup.py build_ext --inplace

# 4. Install package
pip install -e .

# 5. Configure database
# Edit config/config.yaml with your database settings

# 6. Initialize database
python -m quant_trading.database.init

# 7. Run the API server
python -m quant_trading.api.server
```

## Your First Strategy

### 1. Create a Strategy Configuration

Create a file `config/strategies/my_strategy.yaml`:

```yaml
strategy_id: my_first_strategy
name: Simple Mean Reversion
type: mean_reversion

parameters:
  lookback_period: 60
  entry_z_score: 2.0
  exit_z_score: 0.5
  max_position_size: 0.10

risk_limits:
  max_daily_loss: 5000
  max_position_size: 50000
  max_total_exposure: 200000

symbols:
  - SPY
  - QQQ
```

### 2. Backtest the Strategy

```python
from quant_trading.backtesting import BacktestEngine
from quant_trading.strategies import MeanReversionStrategy

# Load strategy
strategy = MeanReversionStrategy.from_config('my_first_strategy.yaml')

# Set up backtest
engine = BacktestEngine(
    strategy=strategy,
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1000000
)

# Run backtest
results = engine.run()

# View results
print(results.summary())
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Total Return: {results.total_return:.2%}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### 3. Paper Trade

```bash
# Start paper trading
python -m quant_trading.trading.paper \
  --strategy my_first_strategy \
  --config config/strategies/my_first_strategy.yaml
```

### 4. Monitor via Dashboard

Access the Grafana dashboard at `http://localhost:3000` to monitor:
- Strategy performance
- P&L metrics
- Risk exposure
- Signal quality

## API Examples

### Python SDK

```python
from quant_trading.client import TradingClient

# Initialize client
client = TradingClient(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

# Get active signals
signals = client.signals.get_active()
for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type} (confidence: {signal.confidence:.0%})")

# Get portfolio summary
portfolio = client.portfolio.get_summary()
print(f"Total Value: ${portfolio.total_value:,.2f}")
print(f"P&L: ${portfolio.total_pnl:,.2f}")

# Get risk metrics
risk = client.risk.get_metrics()
print(f"VaR (95%): ${risk.var_95:,.2f}")
```

### REST API

```bash
# Authenticate
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token \
  -d "username=admin&password=admin" | jq -r '.access_token')

# Get strategies
curl -s http://localhost:8000/v1/strategies \
  -H "Authorization: Bearer $TOKEN" | jq

# Get signals
curl -s http://localhost:8000/v1/signals/active \
  -H "Authorization: Bearer $TOKEN" | jq

# Get portfolio
curl -s http://localhost:8000/v1/portfolio/summary \
  -H "Authorization: Bearer $TOKEN" | jq
```

## Core Concepts

### Trading Strategies

The system supports three main strategy types:

1. **Volatility Arbitrage** (`volatility_arbitrage`)
   - Compares model-implied vs market-implied volatilities
   - Uses Heston and SABR models for pricing
   - Identifies mispriced options

2. **Mean Reversion** (`mean_reversion`)
   - Trades mean-reverting spreads/pairs
   - Uses Ornstein-Uhlenbeck model
   - Mathematically optimal entry/exit boundaries

3. **Term Structure** (`term_structure`)
   - Exploits volatility term structure misalignments
   - Calendar spread strategies

### Signal Confidence

Signals are generated with confidence scores (0-1):
- **>0.8**: High confidence - full position size
- **0.6-0.8**: Medium confidence - reduced size
- **<0.6**: Low confidence - not traded

### Risk Management

Position sizing follows volatility scaling:
```
position_size = target_volatility / realized_volatility * base_size
```

## Directory Structure

```
quant-trading/
├── src/
│   ├── python/           # Python source code
│   │   └── quant_trading/
│   │       ├── api/      # FastAPI application
│   │       ├── models/   # Mathematical models
│   │       ├── signals/  # Signal generation
│   │       ├── risk/     # Risk management
│   │       └── execution/# Order execution
│   └── cpp/              # C++ extensions
├── tests/                # Test suite
├── config/               # Configuration files
├── docs/                 # Documentation
├── notebooks/            # Jupyter tutorials
└── deploy/               # Deployment configs
```

## Next Steps

1. **Read the Architecture Overview**: [docs/architecture/system-overview.md](./architecture/system-overview.md)
2. **Explore Model Documentation**: [docs/models/](./models/)
3. **Try the Jupyter Tutorials**: [notebooks/](../notebooks/)
4. **Review API Documentation**: [docs/api/rest-api.md](./api/rest-api.md)
5. **Set Up Development Environment**: [docs/development/setup.md](./development/setup.md)

## Getting Help

- **Documentation**: Browse the `docs/` directory
- **Jupyter Notebooks**: Interactive tutorials in `notebooks/`
- **GitHub Issues**: Report bugs or request features
- **Slack**: #quant-trading-support channel

## Troubleshooting

### Common Issues

**Database connection failed**
```bash
# Check PostgreSQL is running
docker ps | grep timescaledb

# Reset database container
docker restart timescaledb
```

**C++ extension import error**
```bash
# Rebuild extensions
python setup.py build_ext --inplace
```

**API returns 401 Unauthorized**
```bash
# Get new token
curl -X POST http://localhost:8000/v1/auth/token \
  -d "username=admin&password=admin"
```

See [Troubleshooting Guide](./operations/runbooks/incident-response.md) for more details.
