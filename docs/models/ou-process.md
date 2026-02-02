# Ornstein-Uhlenbeck Mean Reversion Model

## Overview

The Ornstein-Uhlenbeck (OU) process is a mean-reverting stochastic process widely used in quantitative finance for modeling spreads, interest rates, and other economic variables that tend to revert to a long-term mean. Combined with optimal stopping theory, it provides a mathematically rigorous framework for mean-reversion trading strategies.

**Reference:** Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit." *Journal of Industrial and Management Optimization*.

**Paper Location:** `docs/bibliography/1411.5062.pdf`

## Mathematical Specification

### Process Definition

The Ornstein-Uhlenbeck process X_t satisfies the stochastic differential equation:

```
dX_t = μ(θ - X_t)dt + σ dB_t
```

Where:
- `X_t`: Process value at time t (e.g., spread between two assets)
- `θ`: Long-term mean (equilibrium level)
- `μ`: Mean-reversion speed
- `σ`: Instantaneous volatility
- `B_t`: Standard Brownian motion

### Solution

The explicit solution is:

```
X_t = θ + (X_0 - θ)e^{-μt} + σ ∫_0^t e^{-μ(t-s)} dB_s
```

### Key Properties

| Property | Formula | Description |
|----------|---------|-------------|
| Mean | E[X_t] = θ + (X_0 - θ)e^{-μt} | Converges to θ |
| Variance | Var(X_t) = σ²/(2μ)(1 - e^{-2μt}) | Converges to σ²/(2μ) |
| Stationary variance | σ²/(2μ) | Long-run variance |
| Half-life | t_{1/2} = ln(2)/μ | Time to halve the deviation |
| Autocorrelation | ρ(τ) = e^{-μτ} | Exponential decay |

### Parameter Interpretation

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Long-term mean | θ | Varies | Equilibrium spread level |
| Mean-reversion speed | μ | 0.01 - 10 | Higher = faster reversion |
| Volatility | σ | Varies | Noise amplitude |
| Half-life | t_{1/2} | 5 - 60 days | Trading relevance threshold |

## Parameter Estimation

### Maximum Likelihood Estimation (MLE)

For discrete observations X_0, X_1, ..., X_n at time intervals Δt:

```python
def estimate_ou_parameters_mle(data, dt=1/252):
    """
    Estimate OU parameters using Maximum Likelihood.

    Args:
        data: Array of observations
        dt: Time step (default: 1 trading day)

    Returns:
        OUParameters(theta, mu, sigma)
    """
    n = len(data) - 1
    X = data[:-1]
    Y = data[1:]

    # Regression: Y = a + b*X + epsilon
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    b = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
    a = Y_mean - b * X_mean

    # MLE estimates
    mu = -np.log(b) / dt
    theta = a / (1 - b)

    # Residual variance
    residuals = Y - a - b * X
    sigma_residual = np.sqrt(np.sum(residuals**2) / n)
    sigma = sigma_residual * np.sqrt(2 * mu / (1 - np.exp(-2 * mu * dt)))

    return OUParameters(theta=theta, mu=mu, sigma=sigma)
```

### Half-Life Test

For a valid mean-reversion strategy, the half-life should be:
- **>5 days**: Avoid noise and transaction cost erosion
- **<60 days**: Ensure reasonable trading frequency

```python
def is_tradeable_half_life(params, min_days=5, max_days=60):
    """Check if half-life is in tradeable range."""
    half_life = np.log(2) / params.mu
    return min_days <= half_life <= max_days
```

## Optimal Trading Strategy

### Optimal Stopping Problem

The trading problem is formulated as finding:
1. **Optimal entry boundaries** [L*, U*] - where to enter a position
2. **Optimal exit level** b* - where to take profit
3. **Stop-loss level** L - where to cut losses

### Value Function

The value function V(x) represents maximum expected profit:

```
V(x) = max{E_x[e^{-rτ}(X_τ - c) - V(X_τ)] : τ is stopping time}
```

Where:
- r: Discount rate
- c: Transaction cost
- τ: Stopping time (entry or exit)

### Hamilton-Jacobi-Bellman Equation

The optimal boundaries are characterized by:

```
Exit region:  max{LV(x) - rV(x), x - c - V(x)} = 0
Continue:     LV(x) - rV(x) = 0
Entry region: max{LV(x) - rV(x), V(x) - x + c - V(x)} = 0
```

Where L is the infinitesimal generator:
```
LV(x) = μ(θ - x)V'(x) + (1/2)σ²V''(x)
```

### Analytical Solution

**Entry boundaries** solve:

```python
def compute_optimal_boundaries(params, cost, rate=0.0):
    """
    Compute optimal entry/exit boundaries using Leung & Li (2015).

    Args:
        params: OU parameters (theta, mu, sigma)
        cost: Round-trip transaction cost
        rate: Discount rate

    Returns:
        OptimalBoundaries(entry_lower, entry_upper, exit_target, stop_loss)
    """
    theta, mu, sigma = params.theta, params.mu, params.sigma

    # Stationary distribution parameters
    stationary_std = sigma / np.sqrt(2 * mu)

    # Solve for boundaries using numerical methods
    # This involves solving the HJB equation

    # Simplified heuristic (full solution uses Bessel functions):
    entry_lower = theta - 2 * stationary_std
    entry_upper = theta + 2 * stationary_std
    exit_target = theta
    stop_loss = entry_lower - 2 * stationary_std

    return OptimalBoundaries(
        entry_lower=entry_lower,
        entry_upper=entry_upper,
        exit_target=exit_target,
        stop_loss=stop_loss
    )
```

## Implementation

### Python Interface

```python
from quant_trading.models import OUProcess, OUFitter

# Fit OU process to spread data
fitter = OUFitter()
result = fitter.fit(spread_data, dt=1/252)

print(f"Long-term mean: {result.parameters.theta:.4f}")
print(f"Mean-reversion speed: {result.parameters.mu:.4f}")
print(f"Volatility: {result.parameters.sigma:.4f}")
print(f"Half-life: {result.half_life:.1f} days")

# Check statistical significance
if result.adf_pvalue < 0.05:
    print("✓ Spread is stationary (ADF test passed)")

# Compute optimal boundaries
boundaries = fitter.compute_optimal_boundaries(
    cost=0.001,  # 10 bps round-trip
    rate=0.05
)

print(f"Entry zone: [{boundaries.entry_lower:.4f}, {boundaries.entry_upper:.4f}]")
print(f"Exit target: {boundaries.exit_target:.4f}")
print(f"Stop-loss: {boundaries.stop_loss:.4f}")
```

### Signal Generation

```python
class MeanReversionSignalGenerator:
    """Generate trading signals based on OU model."""

    def __init__(self, params, boundaries):
        self.params = params
        self.boundaries = boundaries

    def generate_signal(self, current_spread, current_position):
        """
        Generate signal based on current spread and position.

        Returns:
            Signal with type and strength
        """
        theta = self.params.theta
        b = self.boundaries

        if current_position == 0:
            # No position - check for entry
            if current_spread > b.entry_upper:
                # Spread too high - short it
                return Signal(
                    signal_type='SHORT',
                    strength=self._compute_strength(current_spread),
                    target=b.exit_target,
                    stop_loss=current_spread + (current_spread - theta)
                )
            elif current_spread < b.entry_lower:
                # Spread too low - long it
                return Signal(
                    signal_type='LONG',
                    strength=self._compute_strength(current_spread),
                    target=b.exit_target,
                    stop_loss=current_spread - (theta - current_spread)
                )

        elif current_position > 0:
            # Long position - check for exit
            if current_spread >= b.exit_target:
                return Signal(signal_type='EXIT_LONG', strength=1.0)
            elif current_spread <= b.stop_loss:
                return Signal(signal_type='EXIT_LONG', strength=1.0)

        elif current_position < 0:
            # Short position - check for exit
            if current_spread <= b.exit_target:
                return Signal(signal_type='EXIT_SHORT', strength=1.0)
            elif current_spread >= b.stop_loss:
                return Signal(signal_type='EXIT_SHORT', strength=1.0)

        return None  # No action

    def _compute_strength(self, spread):
        """Signal strength based on deviation from mean."""
        deviation = abs(spread - self.params.theta)
        stationary_std = self.params.sigma / np.sqrt(2 * self.params.mu)
        z_score = deviation / stationary_std
        return min(1.0, z_score / 3)  # Normalize to [0, 1]
```

## Cointegration Testing

Before applying OU model to pairs/spreads, verify cointegration:

### Engle-Granger Two-Step Method

```python
from statsmodels.tsa.stattools import adfuller, coint

def test_cointegration(asset1_prices, asset2_prices):
    """
    Test for cointegration using Engle-Granger method.

    Returns:
        CointegrationResult with hedge ratio and test statistics
    """
    # Step 1: Estimate cointegrating regression
    # asset1 = alpha + beta * asset2 + residual
    X = sm.add_constant(asset2_prices)
    model = sm.OLS(asset1_prices, X).fit()
    hedge_ratio = model.params[1]
    spread = asset1_prices - hedge_ratio * asset2_prices

    # Step 2: Test spread for stationarity
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(spread)

    # Also run cointegration test directly
    _, coint_pvalue, _ = coint(asset1_prices, asset2_prices)

    return CointegrationResult(
        hedge_ratio=hedge_ratio,
        spread=spread,
        adf_statistic=adf_stat,
        adf_pvalue=adf_pvalue,
        coint_pvalue=coint_pvalue,
        is_cointegrated=(coint_pvalue < 0.05)
    )
```

## Performance Requirements

| Operation | Target Latency |
|-----------|---------------|
| Parameter estimation | <100 ms |
| Boundary computation | <500 ms |
| Signal generation | <10 ms |
| Cointegration test | <1 second |

## Validation

### Simulation Test

```python
def test_ou_parameter_recovery():
    """Test parameter recovery from simulated data."""
    true_params = OUParameters(theta=0.0, mu=2.0, sigma=0.1)

    # Simulate OU process
    dt = 1/252
    n_steps = 1000
    X = simulate_ou_process(true_params, dt, n_steps)

    # Estimate parameters
    fitter = OUFitter()
    result = fitter.fit(X, dt)

    # Check recovery (within 20%)
    assert abs(result.parameters.theta - true_params.theta) < 0.02
    assert abs(result.parameters.mu - true_params.mu) / true_params.mu < 0.2
    assert abs(result.parameters.sigma - true_params.sigma) / true_params.sigma < 0.2
```

### Backtest Validation

```python
def backtest_mean_reversion_strategy(spread_data, params, boundaries):
    """
    Backtest mean reversion strategy.

    Returns:
        BacktestResult with performance metrics
    """
    signal_gen = MeanReversionSignalGenerator(params, boundaries)
    portfolio = Portfolio(initial_cash=100000)

    for t, spread in enumerate(spread_data):
        signal = signal_gen.generate_signal(spread, portfolio.position)
        if signal:
            portfolio.execute(signal, spread, t)

    return BacktestResult(
        total_return=portfolio.total_return(),
        sharpe_ratio=portfolio.sharpe_ratio(),
        max_drawdown=portfolio.max_drawdown(),
        win_rate=portfolio.win_rate(),
        avg_holding_period=portfolio.avg_holding_period()
    )
```

## Trading Applications

### Pairs Trading

1. **Screen pairs** for cointegration
2. **Estimate OU parameters** on spread
3. **Verify half-life** is in tradeable range
4. **Compute optimal boundaries** with transaction costs
5. **Generate signals** when spread hits boundaries
6. **Monitor cointegration** health continuously

### Statistical Arbitrage

```python
class PairsTradingStrategy:
    """Full pairs trading implementation."""

    def __init__(self, asset1, asset2, lookback=252):
        self.asset1 = asset1
        self.asset2 = asset2
        self.lookback = lookback

    def update(self, prices1, prices2):
        """Update model with new prices."""
        # Test cointegration
        coint_result = test_cointegration(prices1, prices2)
        if not coint_result.is_cointegrated:
            self.is_valid = False
            return

        # Fit OU model
        fitter = OUFitter()
        ou_result = fitter.fit(coint_result.spread)

        # Check half-life
        if not is_tradeable_half_life(ou_result.parameters):
            self.is_valid = False
            return

        # Compute boundaries
        self.boundaries = fitter.compute_optimal_boundaries(cost=0.002)
        self.params = ou_result.parameters
        self.hedge_ratio = coint_result.hedge_ratio
        self.is_valid = True

    def get_signal(self, current_prices):
        """Generate trading signal."""
        if not self.is_valid:
            return None

        spread = current_prices[0] - self.hedge_ratio * current_prices[1]
        return self.signal_generator.generate_signal(spread, self.position)
```

## Risk Management

### Position Sizing

```python
def compute_position_size(params, boundaries, max_loss, current_spread):
    """
    Compute position size based on stop-loss distance.

    Args:
        params: OU parameters
        boundaries: Optimal boundaries
        max_loss: Maximum acceptable loss
        current_spread: Current spread value

    Returns:
        Position size in units
    """
    if current_spread > params.theta:
        # Short entry
        stop_distance = boundaries.stop_loss - current_spread
    else:
        # Long entry
        stop_distance = current_spread - boundaries.stop_loss

    position_size = max_loss / abs(stop_distance)
    return position_size
```

### Cointegration Breakdown Alert

```python
def monitor_cointegration_health(spread, params, window=60):
    """
    Monitor for cointegration breakdown.

    Returns:
        HealthStatus with warnings
    """
    recent_spread = spread[-window:]

    # Rolling ADF test
    adf_stat, adf_pvalue, _, _, _, _ = adfuller(recent_spread)

    # Parameter stability
    recent_params = OUFitter().fit(recent_spread)
    param_drift = abs(recent_params.mu - params.mu) / params.mu

    warnings = []
    if adf_pvalue > 0.10:
        warnings.append("ADF test weakening - cointegration may be breaking")
    if param_drift > 0.5:
        warnings.append("Mean-reversion speed changed significantly")

    return HealthStatus(
        adf_pvalue=adf_pvalue,
        param_drift=param_drift,
        warnings=warnings
    )
```

## References

1. Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit"
2. Gatev, E., Goetzmann, W.N., & Rouwenhorst, K.G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"
3. Bertram, W.K. (2010). "Analytic solutions for optimal statistical arbitrage trading"
4. Engle, R.F., & Granger, C.W.J. (1987). "Co-integration and error correction"
