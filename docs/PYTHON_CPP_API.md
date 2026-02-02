# Python/C++ API Reference

This document describes the Python API for the quantitative trading C++ extensions.

## Installation

The C++ extensions are automatically built when installing the package:

```bash
pip install -e .
```

## Module Structure

```
quant_trading/
├── cpp/
│   └── quant_cpp          # C++ extension module
│       ├── heston         # Heston model submodule
│       ├── sabr           # SABR model submodule
│       └── ou             # OU process submodule
└── models/
    ├── heston.py          # Python wrapper
    ├── sabr.py            # Python wrapper
    └── ou_process.py      # Python wrapper
```

## Quick Start

### Using Python Wrappers (Recommended)

```python
from quant_trading.models import HestonModel, SABRModel, OUProcess, OUParameters

# Heston option pricing
heston = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
price = heston.price_option(strike=100, maturity=1.0, spot=100, rate=0.05, dividend=0.02)

# SABR implied volatility
sabr = SABRModel(beta=0.5)
vol = sabr.implied_volatility(strike=105, forward=100, maturity=1.0,
                               alpha=0.2, rho=-0.3, nu=0.4)

# OU process fitting
params = OUParameters(theta=100.0, mu=5.0, sigma=2.0)
path = OUProcess.simulate(params, x0=100.0, T=1.0, n_steps=252, seed=42)
result = OUProcess.fit_mle(path, dt=1.0/252.0)
```

### Using C++ Bindings Directly

```python
from quant_trading.cpp import quant_cpp

# Access submodules
heston = quant_cpp.heston
sabr = quant_cpp.sabr
ou = quant_cpp.ou
```

---

## Heston Model

### HestonModel Class

The Heston stochastic volatility model for option pricing.

**Model Equations:**
```
dS_t = μS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
dW_t^S · dW_t^v = ρ dt
```

#### Constructor

```python
model = HestonModel(kappa, theta, sigma, rho, v0)
```

**Parameters:**
- `kappa` (float): Mean-reversion speed of variance (must be > 0)
- `theta` (float): Long-term variance mean (must be > 0)
- `sigma` (float): Volatility of variance (must be > 0)
- `rho` (float): Correlation between asset and variance (-1 < rho < 1)
- `v0` (float): Initial variance (must be > 0)

**Feller Condition:**
For numerical stability, parameters should satisfy: 2κθ ≥ σ²

#### Methods

##### price_option

```python
price = model.price_option(strike, maturity, spot, rate, dividend, is_call=True)
```

Price a single European option using Carr-Madan FFT method.

**Parameters:**
- `strike` (float): Strike price K
- `maturity` (float): Time to maturity T (years)
- `spot` (float): Current spot price S0
- `rate` (float): Risk-free rate r
- `dividend` (float): Dividend yield q
- `is_call` (bool): True for call, False for put

**Returns:** Option price (float)

##### price_options

```python
prices = model.price_options(strikes, maturities, spot, rate, dividend, is_call=True)
```

Price multiple options (vectorized).

**Parameters:**
- `strikes` (array-like): Array of strike prices
- `maturities` (array-like or float): Array of maturities or single value
- Other parameters same as `price_option`

**Returns:** NumPy array of prices

##### price_option_with_greeks

```python
result = model.price_option_with_greeks(strike, maturity, spot, rate, dividend, is_call=True)
```

Price option and compute Greeks using finite differences.

**Returns:** `PricingResult` with:
- `price` (float): Option price
- `greeks` (OptionGreeks): Delta, gamma, vega, theta, rho

##### implied_volatility

```python
iv = model.implied_volatility(strike, maturity, spot, rate, dividend, is_call=True)
```

Compute Black-Scholes implied volatility from Heston price.

**Returns:** Implied volatility (float)

---

## SABR Model

### SABRModel Class

SABR volatility model using Hagan et al. (2002) asymptotic formula.

**Model Equations:**
```
dF_t = σ_t F_t^β dW_t^F
dσ_t = ν σ_t dW_t^σ
dW_t^F · dW_t^σ = ρ dt
```

#### Constructor

```python
model = SABRModel(beta=0.5)
```

**Parameters:**
- `beta` (float): CEV exponent (0 = normal, 0.5 = equity, 1 = lognormal)

#### Methods

##### implied_volatility

```python
vol = model.implied_volatility(strike, forward, maturity, alpha, rho, nu)
```

Compute implied volatility using Hagan asymptotic formula.

**Parameters:**
- `strike` (float): Strike price K
- `forward` (float): Forward price F
- `maturity` (float): Time to maturity T (years)
- `alpha` (float): Initial volatility α
- `rho` (float): Correlation ρ
- `nu` (float): Vol of vol ν

**Returns:** Black-Scholes implied volatility (float)

##### implied_volatilities

```python
vols = model.implied_volatilities(strikes, forward, maturity, alpha, rho, nu)
```

Vectorized implied volatility calculation.

**Returns:** NumPy array of implied volatilities

##### atm_volatility

```python
atm_vol = model.atm_volatility(forward, maturity, alpha, rho, nu)
```

Compute ATM implied volatility (simplified formula).

##### volatility_sensitivities

```python
d_alpha, d_rho, d_nu = model.volatility_sensitivities(strike, forward, maturity, alpha, rho, nu)
```

Compute partial derivatives of implied vol with respect to SABR parameters.

---

## Ornstein-Uhlenbeck Process

### OUProcess Class

Utilities for Ornstein-Uhlenbeck mean-reverting process.

**Process Definition:**
```
dX_t = μ(θ - X_t)dt + σ dB_t
```

All methods are static.

### OUParameters Class

```python
params = OUParameters(theta, mu, sigma)
```

**Attributes:**
- `theta` (float): Long-term mean (equilibrium level)
- `mu` (float): Mean-reversion speed
- `sigma` (float): Volatility

**Methods:**
- `half_life()`: Returns ln(2)/μ
- `is_mean_reverting()`: Returns True if μ > 0
- `stationary_variance()`: Returns σ²/(2μ)
- `stationary_std()`: Returns √(stationary_variance)

### Static Methods

##### fit_mle

```python
result = OUProcess.fit_mle(prices, dt=1.0/252.0)
```

Fit OU parameters using Maximum Likelihood Estimation.

**Parameters:**
- `prices` (array-like): Time series of observations
- `dt` (float): Time increment between observations

**Returns:** `OUFitResult` with:
- `params` (OUParameters): Estimated parameters
- `log_likelihood` (float): Log-likelihood at optimum
- `aic` (float): Akaike Information Criterion
- `bic` (float): Bayesian Information Criterion
- `converged` (bool): Whether optimization converged
- `message` (str): Additional information

##### simulate

```python
path = OUProcess.simulate(params, x0, T, n_steps, seed=42)
```

Simulate OU process path using exact discretization.

**Parameters:**
- `params` (OUParameters): Process parameters
- `x0` (float): Initial value
- `T` (float): Total time horizon
- `n_steps` (int): Number of time steps
- `seed` (int): Random seed

**Returns:** NumPy array of length n_steps + 1

##### log_likelihood

```python
ll = OUProcess.log_likelihood(prices, params, dt=1.0/252.0)
```

Compute log-likelihood of data under OU model.

##### conditional_mean / conditional_variance

```python
mean = OUProcess.conditional_mean(x_t, params, dt)
var = OUProcess.conditional_variance(params, dt)
```

Compute conditional moments of X_{t+dt} given X_t.

##### optimal_boundaries

```python
entry_lower, entry_upper, exit_target = OUProcess.optimal_boundaries(
    params, transaction_cost, risk_free_rate
)
```

Compute optimal trading boundaries for mean-reversion strategy.

---

## Performance

The C++ implementations provide significant speedups:

| Operation | Throughput |
|-----------|------------|
| Heston single option | ~1,000 options/sec |
| Heston vectorized | ~10,000 options/sec |
| SABR implied vol | ~100,000 calcs/sec |
| OU simulation (252 steps) | ~10,000 paths/sec |
| OU MLE fitting | ~10,000 fits/sec |

Run benchmarks with:
```bash
python benchmarks/python_vs_cpp.py
```

---

## Error Handling

C++ exceptions are automatically converted to Python exceptions:

- `std::invalid_argument` → `ValueError`
- `std::runtime_error` → `RuntimeError`
- `std::out_of_range` → `IndexError`

```python
from quant_trading.models import HestonModel

try:
    model = HestonModel(kappa=-1.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

---

## References

- **Heston (1993)**: "A closed-form solution for options with stochastic volatility"
- **Hagan et al. (2002)**: "Managing smile risk"
- **Carr & Madan (1999)**: "Option valuation using the fast Fourier transform"
- **Leung & Li (2015)**: "Optimal Mean Reversion Trading with Transaction Costs"
