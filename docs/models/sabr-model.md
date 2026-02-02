# SABR Volatility Model

## Overview

The SABR (Stochastic Alpha Beta Rho) model is a stochastic volatility model designed to capture the dynamics of the volatility smile. It is widely used for pricing and hedging interest rate derivatives and equity options due to its analytical tractability and ability to fit market smiles accurately.

**Reference:** Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002). "Managing smile risk." *Wilmott Magazine*, September, 84-108.

**Paper Location:** `docs/bibliography/SABR.pdf`

## Mathematical Specification

### Forward Process Dynamics

The SABR model specifies the following stochastic differential equations for the forward price F and its volatility σ:

```
dF_t = σ_t F_t^β dW_t^F
dσ_t = ν σ_t dW_t^σ
dW_t^F · dW_t^σ = ρ dt
```

Where:
- `F_t`: Forward price at time t
- `σ_t`: Stochastic volatility at time t
- `β` (beta): CEV exponent (backbone parameter)
- `ν` (nu): Volatility of volatility
- `ρ` (rho): Correlation between forward and volatility
- `W_t^F, W_t^σ`: Correlated Brownian motions

### Parameter Interpretation

| Parameter | Symbol | Typical Range | Interpretation |
|-----------|--------|---------------|----------------|
| Initial volatility | α | 0.01 - 1.0 | Current volatility level |
| CEV exponent | β | 0.0 - 1.0 | Backbone dynamics |
| Correlation | ρ | -1.0 - 1.0 | Skew direction |
| Vol of vol | ν | 0.1 - 2.0 | Smile curvature |

### Beta Parameter

The β parameter controls how the ATM volatility moves with the underlying:

| β Value | Model Type | Behavior |
|---------|------------|----------|
| β = 0 | Normal | ATM vol constant regardless of forward |
| β = 0.5 | Square root | Typical for equity (used in our system) |
| β = 1 | Lognormal | ATM vol proportional to forward |

**Our Default:** β = 0.5 for equity options

## Asymptotic Implied Volatility Formula

### Hagan's Formula

For strikes not too far from the forward, the implied Black volatility is approximately:

```
σ_B(K,F) ≈ α * (z/χ(z)) * [1 + correction_terms * T]
```

Where:

**For β ≠ 1:**
```
z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
χ(z) = ln[(√(1 - 2ρz + z²) + z - ρ) / (1 - ρ)]
```

**Correction terms:**
```
correction = [(1-β)²/24 * α²/(FK)^(1-β)
            + ρβνα/4/(FK)^((1-β)/2)
            + (2-3ρ²)/24 * ν²]
```

**At-the-money (K = F):**
```
σ_ATM = α/F^(1-β) * [1 + ((1-β)²/24 * α²/F^(2-2β)
                       + ρβνα/4/F^(1-β)
                       + (2-3ρ²)/24 * ν²) * T]
```

### Normal Volatility Formula (β = 0)

When using normal volatility:
```
σ_N(K,F) ≈ α * [1 + correction_terms * T]
```

## Calibration

### Per-Maturity Calibration

SABR is typically calibrated separately for each maturity:

```python
def calibrate_sabr_smile(options, forward, maturity, beta=0.5):
    """
    Calibrate SABR parameters to a single maturity smile.

    Args:
        options: List of (strike, market_vol) tuples
        forward: Current forward price
        maturity: Time to expiration
        beta: Fixed CEV parameter

    Returns:
        Calibrated (alpha, rho, nu) parameters
    """
    def objective(params):
        alpha, rho, nu = params
        total_error = 0
        for strike, market_vol in options:
            model_vol = sabr_implied_vol(
                forward, strike, maturity, alpha, beta, rho, nu
            )
            total_error += (model_vol - market_vol)**2
        return total_error

    # Bounds
    bounds = [
        (0.001, 2.0),   # alpha
        (-0.999, 0.999), # rho
        (0.001, 3.0)    # nu
    ]

    result = scipy.optimize.minimize(
        objective,
        x0=[0.2, -0.3, 0.4],  # Initial guess
        bounds=bounds,
        method='L-BFGS-B'
    )

    return SABRParameters(
        alpha=result.x[0],
        beta=beta,
        rho=result.x[1],
        nu=result.x[2]
    )
```

### Calibration Strategy

1. **Fix β** based on asset class:
   - Equity: β = 0.5
   - Rates: β = 0 (normal model)

2. **Calibrate (α, ρ, ν)** for each maturity
   - Use ATM vol to initialize α
   - Use skew to initialize ρ
   - Use smile curvature to initialize ν

3. **Smooth across maturities** if needed for term structure consistency

## Implementation

### Python Interface

```python
from quant_trading.models import SABRModel, SABRCalibrator

# Create model with known parameters
model = SABRModel(
    alpha=0.2,
    beta=0.5,
    rho=-0.3,
    nu=0.4
)

# Calculate implied volatility for a strike
iv = model.implied_volatility(
    forward=100,
    strike=105,
    maturity=0.5
)

# Price an option using SABR vol
price = model.price_option(
    forward=100,
    strike=105,
    maturity=0.5,
    rate=0.05,
    is_call=True
)

# Calibrate to market smile
calibrator = SABRCalibrator(beta=0.5)
result = calibrator.calibrate(
    options=market_options,
    forward=100,
    maturity=0.5
)

print(f"Alpha: {result.parameters.alpha:.4f}")
print(f"Rho: {result.parameters.rho:.4f}")
print(f"Nu: {result.parameters.nu:.4f}")
print(f"RMSE: {result.rmse:.4f}")
```

### Volatility Surface Construction

```python
def build_sabr_surface(market_data, maturities):
    """
    Build complete volatility surface using SABR.

    Returns dict of SABR parameters by maturity.
    """
    surface = {}
    calibrator = SABRCalibrator(beta=0.5)

    for maturity in maturities:
        options = market_data.get_options_for_maturity(maturity)
        forward = market_data.get_forward(maturity)

        result = calibrator.calibrate(
            options=options,
            forward=forward,
            maturity=maturity
        )

        surface[maturity] = result.parameters

    return surface

def interpolate_vol(surface, strike, maturity):
    """
    Interpolate volatility from SABR surface.
    """
    # Find surrounding maturities
    t1, t2 = find_bracketing_maturities(surface.keys(), maturity)

    # Get vols at each maturity
    vol1 = surface[t1].implied_volatility(strike)
    vol2 = surface[t2].implied_volatility(strike)

    # Linear interpolation in variance time
    w = (maturity - t1) / (t2 - t1)
    return np.sqrt(w * vol2**2 + (1-w) * vol1**2)
```

## Performance Requirements

| Operation | Target Latency |
|-----------|---------------|
| Single IV calculation | <0.1 ms |
| Full smile (50 strikes) | <5 ms |
| Calibration (1 smile) | <1 second |
| Full surface (10 maturities) | <10 seconds |

## Validation

### Smile Recovery Test

```python
def test_smile_recovery():
    """Test that calibration recovers the input smile."""
    true_params = SABRParameters(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    model = SABRModel(true_params)

    # Generate synthetic smile
    forward = 100
    maturity = 0.5
    strikes = np.linspace(85, 115, 15)
    market_vols = [model.implied_volatility(forward, k, maturity) for k in strikes]

    # Calibrate
    calibrator = SABRCalibrator(beta=0.5)
    result = calibrator.calibrate(
        options=list(zip(strikes, market_vols)),
        forward=forward,
        maturity=maturity
    )

    # Check parameter recovery
    assert abs(result.parameters.alpha - true_params.alpha) < 0.01
    assert abs(result.parameters.rho - true_params.rho) < 0.05
    assert abs(result.parameters.nu - true_params.nu) < 0.05
```

### Arbitrage-Free Test

```python
def test_no_calendar_arbitrage():
    """Test that surface is free of calendar arbitrage."""
    surface = build_sabr_surface(market_data, maturities)

    for strike in test_strikes:
        prev_total_var = 0
        for maturity in sorted(surface.keys()):
            vol = surface[maturity].implied_volatility(strike)
            total_var = vol**2 * maturity
            # Total variance must be increasing in maturity
            assert total_var >= prev_total_var
            prev_total_var = total_var
```

## Trading Applications

### Volatility Surface Analysis

1. **Calibrate SABR** to current market data
2. **Monitor parameter changes** over time
3. **Detect anomalies** in smile shape
4. **Generate term structure signals**

```python
def analyze_sabr_surface(surface, historical_params):
    """
    Analyze SABR surface for trading signals.
    """
    signals = []

    for maturity, params in surface.items():
        hist = historical_params.get(maturity)
        if hist is None:
            continue

        # Check rho (skew) deviation
        rho_zscore = (params.rho - hist.rho_mean) / hist.rho_std
        if abs(rho_zscore) > 2:
            signals.append(Signal(
                type='SKEW_ANOMALY',
                maturity=maturity,
                value=params.rho,
                zscore=rho_zscore
            ))

        # Check nu (curvature) deviation
        nu_zscore = (params.nu - hist.nu_mean) / hist.nu_std
        if abs(nu_zscore) > 2:
            signals.append(Signal(
                type='CURVATURE_ANOMALY',
                maturity=maturity,
                value=params.nu,
                zscore=nu_zscore
            ))

    return signals
```

### Smile Dynamics Trading

The SABR model naturally captures the "sticky-delta" behavior where:
- ATM vol moves with the underlying
- Skew is relatively stable
- Smile curvature depends on vol-of-vol

This enables trading strategies that exploit:
- Temporary skew deviations
- Term structure anomalies
- Cross-asset skew relationships

## Limitations

1. **Not arbitrage-free by construction** - May require adjustments for extreme strikes
2. **Static model** - Doesn't capture smile dynamics perfectly
3. **Asymptotic formula breakdown** - Less accurate for:
   - Very short maturities
   - Extreme strikes (far OTM)
   - High vol-of-vol environments

## References

1. Hagan, P.S. et al. (2002). "Managing smile risk"
2. Obloj, J. (2008). "Fine-tune your smile: Correction to Hagan et al."
3. West, G. (2005). "Calibration of the SABR Model in Illiquid Markets"
