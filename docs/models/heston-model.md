# Heston Stochastic Volatility Model

## Overview

The Heston model is a stochastic volatility model that describes asset price dynamics with time-varying volatility. Unlike the Black-Scholes model which assumes constant volatility, the Heston model captures the empirically observed features of volatility clustering and the leverage effect.

**Reference:** Heston, S.L. (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options." *Review of Financial Studies*, 6(2), 327-343.

**Paper Location:** `docs/bibliography/Options with Stochastic Volatility.pdf`

## Mathematical Specification

### Dynamics Under Risk-Neutral Measure

The Heston model specifies the following stochastic differential equations:

```
dS_t = rS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
dW_t^S · dW_t^v = ρ dt
```

Where:
- `S_t`: Asset price at time t
- `v_t`: Instantaneous variance at time t
- `r`: Risk-free interest rate
- `κ` (kappa): Mean-reversion speed of variance
- `θ` (theta): Long-term variance mean
- `σ` (sigma): Volatility of variance ("vol of vol")
- `ρ` (rho): Correlation between asset returns and variance
- `W_t^S, W_t^v`: Correlated Brownian motions

### Parameter Interpretation

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Mean-reversion speed | κ | 0.5 - 5.0 | How fast variance returns to θ |
| Long-term variance | θ | 0.01 - 0.25 | Long-run average variance level |
| Vol of vol | σ | 0.1 - 1.0 | Volatility of the variance process |
| Correlation | ρ | -1.0 - 0.0 | Leverage effect (negative for equities) |
| Initial variance | v₀ | 0.01 - 0.25 | Current variance level |

### Feller Condition

To ensure the variance process remains strictly positive:

```
2κθ ≥ σ²
```

This is known as the **Feller condition**. If violated, the variance can touch zero, which requires special numerical handling.

## Characteristic Function

The key advantage of the Heston model is the semi-analytical solution via the characteristic function:

```
φ(u; τ, v_0) = exp(C(τ,u) + D(τ,u)v_0 + iu·ln(S_0e^{rτ}))
```

Where:
```
C(τ,u) = κθ/σ² [(κ - ρσiu - d)τ - 2ln((1 - ge^{-dτ})/(1-g))]
D(τ,u) = (κ - ρσiu - d)/σ² · (1 - e^{-dτ})/(1 - ge^{-dτ})
d = √[(ρσiu - κ)² + σ²(iu + u²)]
g = (κ - ρσiu - d)/(κ - ρσiu + d)
```

## Option Pricing

### Fourier Inversion (Carr-Madan Method)

European call option price:

```
C(K,T) = S_0·P_1 - K·e^{-rT}·P_2
```

Where P₁ and P₂ are computed via numerical integration:

```
P_j = 1/2 + 1/π ∫_0^∞ Re[e^{-iu·ln(K)} φ_j(u)/iu] du
```

### Fast Fourier Transform

For efficient computation across strikes:

```python
# Pseudocode for FFT pricing
def price_options_fft(params, strikes, maturity, spot, rate):
    # Set up integration grid
    N = 4096  # FFT size
    eta = 0.25  # Grid spacing

    # Characteristic function evaluations
    u = np.arange(N) * eta
    char_vals = heston_characteristic(u, params, maturity)

    # FFT
    x = fft(char_vals * damping_factor(u))

    # Interpolate to desired strikes
    prices = interpolate(x, strikes)
    return prices
```

## Calibration

### Objective Function

Minimize weighted sum of squared implied volatility errors:

```
min Σ w_i (σ_model(K_i, T_i) - σ_market(K_i, T_i))²
```

Weights `w_i` typically based on:
- Vega (higher weight for ATM options)
- Bid-ask spread (lower weight for illiquid options)
- Volume

### Optimization Algorithm

1. **Global Search:** Differential evolution
   - Population size: 50-100
   - Generations: 100-500
   - Bounds enforcement for parameters

2. **Local Refinement:** Levenberg-Marquardt
   - Starting from global optimum
   - Jacobian via finite differences

### Constraints

```python
bounds = {
    'kappa': (0.1, 10.0),
    'theta': (0.01, 0.5),
    'sigma': (0.1, 2.0),
    'rho': (-0.99, 0.0),
    'v0': (0.01, 0.5)
}

# Feller constraint (soft penalty)
def feller_penalty(kappa, theta, sigma):
    violation = max(0, sigma**2 - 2*kappa*theta)
    return 1000 * violation**2
```

## Implementation

### Python Interface

```python
from quant_trading.models import HestonModel, HestonCalibrator

# Create model with known parameters
model = HestonModel(
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.7,
    v0=0.05
)

# Price a European call
price = model.price_call(
    strike=100,
    maturity=0.5,
    spot=100,
    rate=0.05,
    dividend=0.02
)

# Calculate implied volatility
iv = model.implied_volatility(
    strike=100,
    maturity=0.5,
    spot=100,
    rate=0.05,
    dividend=0.02
)

# Calibrate to market data
calibrator = HestonCalibrator()
result = calibrator.calibrate(
    market_options=options_data,
    spot=100,
    rate=0.05,
    dividend=0.02
)

print(f"Calibrated parameters: {result.parameters}")
print(f"RMSE: {result.rmse:.4f}")
print(f"Feller satisfied: {result.parameters.feller_satisfied}")
```

### C++ Core Implementation

The performance-critical computations are implemented in C++:

```cpp
// src/cpp/heston/heston_pricer.cpp

class HestonPricer {
public:
    HestonPricer(const HestonParameters& params);

    // Price European option via FFT
    double price_european(
        double strike,
        double maturity,
        double spot,
        double rate,
        double dividend,
        bool is_call
    ) const;

    // Batch pricing for calibration
    std::vector<double> price_batch(
        const std::vector<double>& strikes,
        const std::vector<double>& maturities,
        double spot,
        double rate,
        double dividend,
        const std::vector<bool>& is_call
    ) const;

private:
    // Characteristic function
    std::complex<double> characteristic_function(
        std::complex<double> u,
        double maturity
    ) const;

    HestonParameters params_;
};
```

## Performance Requirements

| Operation | Target Latency |
|-----------|---------------|
| Single option price | <1 ms |
| 50 options (FFT batch) | <100 ms |
| Full calibration | <30 seconds |
| Implied vol inversion | <5 ms |

## Validation

### Unit Tests

```python
def test_heston_put_call_parity():
    """Test put-call parity holds."""
    model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.05)

    call = model.price_call(strike=100, maturity=0.5, spot=100, rate=0.05)
    put = model.price_put(strike=100, maturity=0.5, spot=100, rate=0.05)

    # Put-call parity: C - P = S - K*exp(-rT)
    parity_lhs = call - put
    parity_rhs = 100 - 100 * np.exp(-0.05 * 0.5)

    assert abs(parity_lhs - parity_rhs) < 1e-6

def test_heston_vs_black_scholes():
    """When vol-of-vol is zero, should match Black-Scholes."""
    model = HestonModel(kappa=2.0, theta=0.04, sigma=0.0, rho=0.0, v0=0.04)
    bs_price = black_scholes_call(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    heston_price = model.price_call(strike=100, maturity=0.5, spot=100, rate=0.05)

    assert abs(heston_price - bs_price) < 1e-4
```

### Calibration Tests

```python
def test_parameter_recovery():
    """Test that calibration recovers known parameters."""
    true_params = HestonParameters(
        kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.05
    )

    # Generate synthetic options data
    model = HestonModel(true_params)
    options_data = generate_synthetic_options(model)

    # Calibrate
    calibrator = HestonCalibrator()
    result = calibrator.calibrate(options_data, spot=100, rate=0.05)

    # Check parameter recovery
    assert abs(result.parameters.kappa - true_params.kappa) < 0.2
    assert abs(result.parameters.theta - true_params.theta) < 0.01
    assert result.rmse < 0.005
```

## Trading Applications

### Volatility Surface Arbitrage

1. **Calibrate Heston model** to liquid options
2. **Compute model-implied vols** across strike/maturity grid
3. **Compare to market vols** to identify mispricings
4. **Generate signals** when deviation exceeds threshold

```python
def generate_vol_arb_signal(model, market_data, threshold=0.02):
    signals = []
    for option in market_data:
        model_iv = model.implied_volatility(
            option.strike, option.maturity, option.spot, option.rate
        )
        market_iv = option.implied_vol
        deviation = model_iv - market_iv

        if abs(deviation) > threshold:
            signal = Signal(
                symbol=option.symbol,
                signal_type='LONG' if deviation < 0 else 'SHORT',
                strength=abs(deviation) / threshold,
                confidence=calibration_quality
            )
            signals.append(signal)

    return signals
```

## References

1. Heston, S.L. (1993). "A closed-form solution for options with stochastic volatility"
2. Carr, P. & Madan, D. (1999). "Option valuation using the fast Fourier transform"
3. Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
