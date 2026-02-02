# REST API Documentation

## Overview

The Quantitative Trading System REST API provides programmatic access to all trading system functionality.

**Base URL:** `https://api.trading.example.com/v1`

**OpenAPI Spec:** See [openapi.yaml](./openapi.yaml)

## Authentication

All API endpoints (except `/health/*`) require JWT authentication.

### Obtaining a Token

```bash
curl -X POST https://api.trading.example.com/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=YOUR_USERNAME&password=YOUR_PASSWORD"
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Token

Include the token in the `Authorization` header:

```bash
curl https://api.trading.example.com/v1/strategies \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## Rate Limiting

| Tier | Limit | Burst |
|------|-------|-------|
| Standard | 100 req/min | 20 req/sec |
| Premium | 1000 req/min | 100 req/sec |

Rate limit headers included in responses:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Common Response Formats

### Success Response
```json
{
  "data": { ... },
  "meta": {
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter value",
    "details": {
      "field": "confidence",
      "reason": "must be between 0 and 1"
    }
  }
}
```

## API Endpoints

### Strategies

#### List Strategies
```
GET /strategies
```

Query Parameters:
- `status` (optional): Filter by status (`active`, `paused`, `stopped`)
- `type` (optional): Filter by strategy type

Example:
```bash
curl https://api.trading.example.com/v1/strategies?status=active \
  -H "Authorization: Bearer $TOKEN"
```

Response:
```json
[
  {
    "strategy_id": "strat_vol_arb_001",
    "name": "SPY Vol Arbitrage",
    "type": "volatility_arbitrage",
    "status": "active",
    "sharpe_ratio": 0.85,
    "total_return": 0.12,
    "max_drawdown": -0.08
  }
]
```

#### Create Strategy
```
POST /strategies
```

Request Body:
```json
{
  "name": "Mean Reversion - Tech Pairs",
  "type": "mean_reversion",
  "parameters": {
    "lookback_period": 60,
    "entry_threshold": 2.0,
    "exit_threshold": 0.5
  },
  "risk_limits": {
    "max_daily_loss": 10000,
    "max_position_size": 50000,
    "max_total_exposure": 200000
  },
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

### Signals

#### Get Active Signals
```
GET /signals/active
```

Example:
```bash
curl https://api.trading.example.com/v1/signals/active \
  -H "Authorization: Bearer $TOKEN"
```

Response:
```json
[
  {
    "signal_id": "sig_20240115_001",
    "strategy_id": "strat_vol_arb_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "symbol": "SPY",
    "signal_type": "LONG",
    "strength": 0.75,
    "confidence": 0.82,
    "target_price": 475.50,
    "stop_loss": 465.00,
    "metadata": {
      "model_iv": 0.18,
      "market_iv": 0.22,
      "mispricing": 0.04
    }
  }
]
```

#### Get Historical Signals
```
GET /strategies/{strategy_id}/signals
```

Query Parameters:
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `signal_type`: Filter by type
- `limit`: Max results (default: 100, max: 1000)

### Portfolio

#### Get Current Positions
```
GET /portfolio/positions
```

Response:
```json
[
  {
    "symbol": "SPY",
    "quantity": 100,
    "average_cost": 470.25,
    "market_value": 47500.00,
    "unrealized_pnl": 475.00,
    "delta": 100,
    "gamma": 0,
    "vega": 0,
    "theta": 0,
    "strategy_id": "strat_vol_arb_001"
  }
]
```

#### Get Portfolio Summary
```
GET /portfolio/summary
```

Response:
```json
{
  "total_value": 1250000.00,
  "cash": 850000.00,
  "positions_value": 400000.00,
  "total_pnl": 25000.00,
  "unrealized_pnl": 15000.00,
  "realized_pnl": 10000.00,
  "total_exposure": 400000.00,
  "net_exposure": 150000.00
}
```

### Risk

#### Get Risk Metrics
```
GET /risk/metrics
```

Response:
```json
{
  "total_exposure": 400000.00,
  "net_exposure": 150000.00,
  "var_95": 12500.00,
  "var_99": 18750.00,
  "expected_shortfall": 22500.00,
  "max_drawdown": -0.08,
  "current_drawdown": -0.02,
  "sharpe_ratio": 0.85,
  "beta": 0.35,
  "correlation_to_market": 0.42
}
```

#### Calculate VaR
```
GET /risk/var?confidence=0.99&horizon=10
```

Response:
```json
{
  "confidence": 0.99,
  "horizon_days": 10,
  "var_absolute": 45000.00,
  "var_percentage": 0.036,
  "method": "monte_carlo"
}
```

### Calibration

#### Get Calibration Status
```
GET /calibration/status
```

Response:
```json
[
  {
    "model": "heston",
    "symbol": "SPY",
    "last_calibration": "2024-01-15T09:30:00Z",
    "status": "current",
    "fit_quality": 0.003
  },
  {
    "model": "sabr",
    "symbol": "SPY",
    "last_calibration": "2024-01-15T09:31:00Z",
    "status": "current",
    "fit_quality": 0.002
  }
]
```

#### Get Heston Parameters
```
GET /calibration/heston?symbol=SPY
```

Response:
```json
{
  "symbol": "SPY",
  "calibrated_at": "2024-01-15T09:30:00Z",
  "kappa": 2.5,
  "theta": 0.04,
  "sigma": 0.3,
  "rho": -0.7,
  "v0": 0.05,
  "feller_satisfied": true,
  "rmse": 0.003
}
```

#### Trigger Calibration
```
POST /calibration/heston
```

Request Body:
```json
{
  "symbol": "SPY",
  "force": true
}
```

Response:
```json
{
  "job_id": "calib_20240115_001",
  "model": "heston",
  "symbol": "SPY",
  "status": "queued",
  "started_at": "2024-01-15T10:30:00Z"
}
```

### Execution

#### Submit Order
```
POST /orders
```

Request Body:
```json
{
  "strategy_id": "strat_vol_arb_001",
  "symbol": "SPY",
  "side": "buy",
  "order_type": "limit",
  "quantity": 100,
  "price": 470.00,
  "time_in_force": "day"
}
```

Response:
```json
{
  "order_id": "ord_20240115_001",
  "strategy_id": "strat_vol_arb_001",
  "symbol": "SPY",
  "side": "buy",
  "order_type": "limit",
  "quantity": 100,
  "price": 470.00,
  "status": "open",
  "filled_quantity": 0,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Cancel Order
```
DELETE /orders/{order_id}
```

Response:
```json
{
  "order_id": "ord_20240115_001",
  "status": "cancelled",
  "updated_at": "2024-01-15T10:31:00Z"
}
```

### Health Checks

#### Health Status
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "queue": "healthy"
  }
}
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | No valid token provided |
| `TOKEN_EXPIRED` | 401 | Token has expired |
| `PERMISSION_DENIED` | 403 | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | 404 | Resource does not exist |
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

## SDK Examples

### Python
```python
from quant_trading.client import TradingClient

client = TradingClient(
    base_url="https://api.trading.example.com/v1",
    api_key="YOUR_API_KEY"
)

# Get active signals
signals = client.signals.get_active()
for signal in signals:
    print(f"{signal.symbol}: {signal.signal_type} ({signal.confidence:.0%})")

# Submit order
order = client.orders.create(
    symbol="SPY",
    side="buy",
    order_type="limit",
    quantity=100,
    price=470.00
)
```

### JavaScript
```javascript
import { TradingClient } from '@quant-trading/client';

const client = new TradingClient({
  baseUrl: 'https://api.trading.example.com/v1',
  apiKey: 'YOUR_API_KEY'
});

// Get portfolio summary
const portfolio = await client.portfolio.getSummary();
console.log(`Total Value: $${portfolio.total_value.toLocaleString()}`);

// Get risk metrics
const risk = await client.risk.getMetrics();
console.log(`VaR 95%: $${risk.var_95.toLocaleString()}`);
```
