# Quantitative Trading System Design Document
## Stochastic Volatility & Optimal Mean-Reversion Trading Platform

**Version:** 1.0  
**Date:** January 2026  
**Author:** System Architecture Team

---

## Executive Summary

This document outlines the architecture for a sophisticated quantitative trading system that leverages advanced mathematical models including stochastic volatility (Heston, SABR), optimal stopping theory, and partial differential equations to generate trading signals for 1-2 month swing trading horizons.

The system is grounded in peer-reviewed academic research and implements rigorous mathematical frameworks for:
- Volatility surface modeling and calibration
- Mean-reversion detection and optimal entry/exit timing
- PDE-based derivatives pricing and hedging
- Risk-managed position sizing

**Target Performance:** Sharpe ratio 0.5-1.2 with max drawdown <25% over multi-year periods.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Architecture Design](#3-architecture-design)
4. [Core Components](#4-core-components)
5. [Data Pipeline](#5-data-pipeline)
6. [Model Calibration Engine](#6-model-calibration-engine)
7. [Signal Generation](#7-signal-generation)
8. [Risk Management](#8-risk-management)
9. [Execution Layer](#9-execution-layer)
10. [Performance Monitoring](#10-performance-monitoring)
11. [Technology Stack](#11-technology-stack)
12. [Deployment Architecture](#12-deployment-architecture)

---

## 1. System Overview

### 1.1 Design Philosophy

The system implements a research-driven approach to quantitative trading, where every component is based on published academic work with demonstrated out-of-sample performance. We avoid overfitting by using models with strong theoretical foundations rather than data-mined patterns.

### 1.2 Core Trading Strategies

**Strategy 1: Stochastic Volatility Arbitrage**
- Model: Heston (1993) and SABR (Hagan et al., 2002)
- Objective: Detect mispricing in options markets by comparing model-implied vs market-implied volatilities
- Horizon: 30-60 days

**Strategy 2: Optimal Mean-Reversion**
- Model: Ornstein-Uhlenbeck with optimal stopping (Leung & Li, 2015)
- Objective: Trade mean-reverting spreads/pairs with mathematically-derived optimal boundaries
- Horizon: 20-90 days

**Strategy 3: Volatility Term Structure**
- Model: Time-dependent SABR calibration
- Objective: Exploit misalignments in volatility term structure
- Horizon: 30-45 days

### 1.3 Key Performance Metrics

- **Sharpe Ratio:** Target 0.7-1.0 (based on Frazzini & Pedersen, 2014 BAB factor ~0.78)
- **Maximum Drawdown:** <25%
- **Win Rate:** 55-65% (mean-reversion strategies per Leung & Li)
- **Average Holding Period:** 35-50 days
- **Turnover:** <10x annual

---

## 2. Mathematical Foundations

### 2.1 Heston Stochastic Volatility Model

**Reference:** Heston, S.L. (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options." *Review of Financial Studies*, 6(2), 327-343.

**Model Specification:**

The Heston model describes asset price dynamics with stochastic volatility:

```
dS_t = μS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
dW_t^S · dW_t^v = ρ dt
```

**Parameters:**
- `κ` (kappa): Mean-reversion speed of variance
- `θ` (theta): Long-term variance mean
- `σ` (sigma): Volatility of variance ("vol of vol")
- `ρ` (rho): Correlation between asset returns and variance
- `v_0`: Initial variance

**Key Properties:**
1. **Mean-reverting variance:** Captures volatility clustering observed in markets
2. **Closed-form characteristic function:** Enables semi-analytical option pricing via Fourier inversion
3. **Volatility smile:** Natural emergence of implied volatility skew through leverage effect (ρ < 0)

**Implementation Notes:**
- Use Fourier inversion with numerical integration (Gauss-Laguerre quadrature)
- Calibrate to liquid options across strikes and maturities
- Monitor Feller condition (2κθ ≥ σ²) to ensure variance remains positive

### 2.2 SABR Volatility Model

**Reference:** Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002). "Managing smile risk." *Wilmott Magazine*, September, 84-108.

**Model Specification:**

```
dF_t = σ_t F_t^β dW_t^F
dσ_t = ν σ_t dW_t^σ
dW_t^F · dW_t^σ = ρ dt
```

**Parameters:**
- `α` (alpha): Initial volatility level
- `β` (beta): CEV exponent (backbone parameter, typically 0.5 for equity)
- `ρ` (rho): Correlation between forward and volatility
- `ν` (nu): Volatility of volatility

**Asymptotic Implied Volatility Formula:**

For strikes not far from forward F_0, implied volatility σ_impl is approximately:

```
σ_impl(K,T) ≈ α * z/χ(z) * {1 + [(2γ_2-γ_1²+1/F_mid²)/24 * (α/C(F_mid))² 
              + ργ_1/4 * ν*C(F_mid)/α + (2-3ρ²)/24 * ν²] * T}
```

Where z and χ(z) are specific functions of strike K and forward F_0.

**Key Properties:**
1. **β controls backbone dynamics:** How ATM vol shifts with forward rate movements
2. **ρ controls skew:** Negative ρ creates downward-sloping volatility smile
3. **ν controls smile curvature:** Higher vol-of-vol creates more pronounced smile
4. **Fast calibration:** Asymptotic formula enables near-instantaneous fitting

**Implementation Notes:**
- Fix β based on asset class (0.5 for equity, 0 for normal model in rates)
- Calibrate (α, ρ, ν) to market smile per maturity
- Use PDE methods for extreme strikes where asymptotic formula breaks down

### 2.3 Ornstein-Uhlenbeck Mean Reversion

**Reference:** Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit." *Journal of Industrial and Management Optimization*.

**Process Definition:**

```
dX_t = μ(θ - X_t)dt + σ dB_t
```

**Parameters:**
- `θ`: Long-term mean (equilibrium level)
- `μ`: Mean-reversion speed
- `σ`: Instantaneous volatility

**Half-life of mean reversion:** t_half = ln(2)/μ

**Optimal Entry/Exit Problem:**

The trading problem is formulated as an **optimal double stopping problem**:
- **Entry decision:** Determine optimal price interval [L_entry, U_entry] to initiate position
- **Exit decision:** Determine optimal take-profit level b* and stop-loss level L

**Key Results from Leung & Li (2015):**

1. **Entry region is bounded:** Optimal to enter only when price ∈ (L*, U*), strictly above stop-loss
2. **Stop-loss affects take-profit:** Higher stop-loss L induces lower optimal take-profit b*
3. **Transaction costs create no-trade zones:** Immediate entry/exit not optimal due to costs

**Value Function:** V(x) represents maximum expected profit from optimal strategy

The Hamilton-Jacobi-Bellman equation characterizes optimal strategy:

```
max{LV(x) - rV(x), x - c - V(x)} = 0  (exit region)
LV(x) - rV(x) = 0  (continuation region)
```

Where L is the infinitesimal generator of the OU process.

**Analytical Solutions:**

For entry problem:
```
Entry if X_0 ∈ (a*, b*) where a*, b* solve system involving modified Bessel functions
```

For exit with stop-loss:
```
Exit at min{τ_b*, τ_L} where τ_z = inf{t≥0: X_t = z}
```

**Implementation Strategy:**
1. **Calibration:** Use MLE to estimate (θ, μ, σ) from historical spread data
2. **Optimization:** Numerically solve for optimal boundaries given transaction costs
3. **Monitoring:** Track spread in real-time, execute trades when boundaries crossed

### 2.4 Optimal Trading Thresholds

**Reference:** Bertram, W.K. (2010). "Analytic solutions for optimal statistical arbitrage trading." *Physica A*, 389(11), 2234-2243.

**First-Passage Time Approach:**

For exponential OU process Y_t = exp(X_t), define:
- Entry threshold: a (enter long when Y_t ≤ a)
- Exit threshold: b (exit when Y_t ≥ b)

**Derived Quantities:**

Expected trade length:
```
E[τ] = f(a,b;μ,σ,θ)  [explicit formula via parabolic cylinder functions]
```

Expected return per trade:
```
E[R] = b - a - 2c  (where c = transaction cost)
```

Variance of return:
```
Var[R] = g(a,b;μ,σ,θ)  [explicit formula]
```

**Sharpe Ratio Optimization:**

```
SR(a,b) = E[R]/√Var[R] * √(1/E[τ])
```

Maximize SR over (a,b) subject to constraints.

### 2.5 Volatility-Managed Strategies

**Reference:** Moreira, A., & Muir, T. (2017). "Volatility-managed portfolios." *Journal of Finance*, 72(4), 1611-1644.

**Core Principle:** Scale portfolio exposure inversely with realized volatility

```
w_t = c/σ_t²
```

Where:
- w_t = position weight at time t
- σ_t² = realized variance (21-day rolling)
- c = target constant

**Theoretical Justification:**

If returns are unpredictable but variance is predictable, volatility timing improves Sharpe ratio.

**Application to Our System:**

1. Compute realized volatility for each strategy
2. Scale position sizes: leverage up in low-vol regimes, de-lever in high-vol regimes
3. Improves risk-adjusted returns significantly (Moreira & Muir show SR gains of 50%+)

---

## 3. Architecture Design

### 3.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Market   │  │ Options  │  │ Reference│  │ Alt Data │   │
│  │ Data API │  │ Chain API│  │ Data     │  │ Sources  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Pipeline                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data         │  │ Feature      │  │ Time Series  │     │
│  │ Validation   │→ │ Engineering  │→ │ Storage      │     │
│  │ & Cleaning   │  │              │  │ (TimescaleDB)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Calibration Engine                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Heston       │  │ SABR         │  │ Ornstein-    │     │
│  │ Calibration  │  │ Calibration  │  │ Uhlenbeck    │     │
│  │              │  │              │  │ Fitting      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Optimization │  │ Parameter    │                        │
│  │ Solvers      │  │ Validation   │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Signal Generation Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Vol Surface  │  │ Mean         │  │ Term         │     │
│  │ Arbitrage    │  │ Reversion    │  │ Structure    │     │
│  │ Signals      │  │ Signals      │  │ Signals      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │     Signal Aggregation & Scoring     │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Risk Management System                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Position     │  │ Volatility   │  │ Greeks       │     │
│  │ Sizing       │  │ Scaling      │  │ Monitoring   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Correlation  │  │ Scenario     │                        │
│  │ Monitoring   │  │ Analysis     │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Execution Engine                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Order        │  │ Smart        │  │ Transaction  │     │
│  │ Management   │→ │ Routing      │→ │ Cost         │     │
│  │ System       │  │              │  │ Analysis     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Monitoring & Analytics                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Performance  │  │ Model        │  │ Alerting     │     │
│  │ Attribution  │  │ Diagnostics  │  │ System       │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Design Principles

1. **Modularity:** Each component independently testable and replaceable
2. **Reproducibility:** All model calibrations and signals fully logged for audit
3. **Robustness:** Graceful degradation if data source fails or model doesn't converge
4. **Performance:** Critical path (calibration → signal → execution) completes in <5 seconds
5. **Extensibility:** New models/strategies easily integrated without refactoring core

---

## 4. Core Components

### 4.1 Component: Heston Model Calibrator

**Purpose:** Calibrate Heston model parameters to market option prices

**Inputs:**
- Market option prices (calls/puts across strikes and maturities)
- Current spot price
- Risk-free rate curve
- Dividend yield

**Outputs:**
- Calibrated parameters: (κ, θ, σ, ρ, v_0)
- Fit quality metrics (RMSE, R²)
- Model-implied volatility surface

**Algorithm:**

```python
class HestonCalibrator:
    """
    Implements Heston (1993) model calibration using global optimization.
    
    Reference: Heston, S.L. (1993). "A closed-form solution for options 
    with stochastic volatility." Review of Financial Studies, 6(2), 327-343.
    """
    
    def calibrate(self, market_options, S0, r, q):
        """
        Calibrate Heston parameters to market option prices.
        
        Method: Differential Evolution (global) + Levenberg-Marquardt (local refinement)
        
        Objective: min Σ(market_price - model_price)² / market_price²
        """
        
        # Initial guess
        params_init = {
            'kappa': 2.0,      # Mean reversion speed
            'theta': 0.04,     # Long-term variance
            'sigma': 0.3,      # Vol of vol
            'rho': -0.7,       # Correlation (typically negative)
            'v0': 0.04         # Initial variance
        }
        
        # Parameter bounds (enforce Feller condition loosely)
        bounds = {
            'kappa': (0.1, 10.0),
            'theta': (0.01, 1.0),
            'sigma': (0.01, 2.0),
            'rho': (-0.99, 0.99),
            'v0': (0.01, 1.0)
        }
        
        # Objective function
        def objective(params):
            model_prices = self._price_options(params, market_options, S0, r, q)
            market_prices = market_options['mid_price'].values
            
            # Relative error (prevents overweighting expensive options)
            errors = (model_prices - market_prices) / market_prices
            return np.sum(errors**2)
        
        # Stage 1: Global search with Differential Evolution
        result_global = differential_evolution(
            objective,
            bounds=list(bounds.values()),
            maxiter=100,
            popsize=15,
            seed=42
        )
        
        # Stage 2: Local refinement with L-M
        result_local = least_squares(
            lambda p: self._residuals(p, market_options, S0, r, q),
            x0=result_global.x,
            bounds=([b[0] for b in bounds.values()], 
                    [b[1] for b in bounds.values()]),
            method='trf'
        )
        
        return self._format_params(result_local.x)
    
    def _price_options(self, params, options, S0, r, q):
        """
        Price options using Heston characteristic function + FFT.
        
        Uses Carr-Madan (1999) FFT approach for efficiency.
        """
        prices = []
        
        for idx, opt in options.iterrows():
            K = opt['strike']
            T = opt['maturity']
            is_call = opt['type'] == 'call'
            
            # Compute characteristic function
            cf = lambda u: self._heston_char_func(
                u, T, S0, r, q, params
            )
            
            # Fourier inversion with Gauss-Laguerre quadrature
            price = self._carr_madan_fft(cf, K, S0, r, T, is_call)
            prices.append(price)
        
        return np.array(prices)
    
    def _heston_char_func(self, u, T, S0, r, q, params):
        """
        Heston characteristic function (Equation 17 in Heston 1993).
        """
        kappa, theta, sigma, rho, v0 = params.values()
        
        # Complex-valued intermediate variables
        d = np.sqrt((rho*sigma*u*1j - kappa)**2 + sigma**2*(u*1j + u**2))
        g = (kappa - rho*sigma*u*1j - d) / (kappa - rho*sigma*u*1j + d)
        
        C = (r - q)*u*1j*T + (kappa*theta/sigma**2) * (
            (kappa - rho*sigma*u*1j - d)*T 
            - 2*np.log((1 - g*np.exp(-d*T))/(1 - g))
        )
        
        D = ((kappa - rho*sigma*u*1j - d)/sigma**2) * (
            (1 - np.exp(-d*T)) / (1 - g*np.exp(-d*T))
        )
        
        return np.exp(C + D*v0 + 1j*u*np.log(S0))
```

**Performance Requirements:**
- Calibration time: <30 seconds for 50 options
- Accuracy: Model prices within 2% of market prices for liquid strikes
- Stability: Feller condition (2κθ ≥ σ²) checked and logged

**Error Handling:**
- If optimization fails to converge: use previous day's parameters, flag for review
- If Feller condition violated: penalize in objective or constrain more tightly
- If fit quality poor (RMSE > 5%): alert risk team, may reduce position sizes

### 4.2 Component: SABR Model Calibrator

**Purpose:** Calibrate SABR parameters to volatility smile for each maturity

**Inputs:**
- Market implied volatilities (grid of strikes per maturity)
- Forward rate F_0
- Time to maturity T

**Outputs:**
- Calibrated parameters per maturity: (α, β, ρ, ν)
- Fitted implied volatility surface
- Calibration residuals

**Algorithm:**

```python
class SABRCalibrator:
    """
    Implements SABR model calibration using Hagan et al. (2002) asymptotic formula.
    
    Reference: Hagan, P.S., et al. (2002). "Managing smile risk." 
    Wilmott Magazine, September 2002, 84-108.
    """
    
    def __init__(self, beta=0.5):
        """
        Initialize SABR calibrator.
        
        Args:
            beta: CEV exponent. Fixed at 0.5 for equity (common practice).
                  Set to 0 for normal model (rates), 1 for lognormal.
        """
        self.beta = beta
    
    def calibrate_smile(self, strikes, market_ivs, forward, maturity):
        """
        Calibrate SABR to a single maturity's volatility smile.
        
        Calibrates (α, ρ, ν) with β fixed.
        
        Returns:
            dict: {'alpha': α, 'beta': β, 'rho': ρ, 'nu': ν, 'rmse': fit_error}
        """
        
        # Initial guess
        x0 = [
            market_ivs[len(strikes)//2],  # alpha ≈ ATM vol
            -0.3,                          # rho (negative for equity)
            0.3                            # nu (vol of vol)
        ]
        
        # Bounds
        bounds = [
            (0.001, 2.0),    # alpha > 0
            (-0.999, 0.999), # rho ∈ (-1, 1)
            (0.001, 2.0)     # nu > 0
        ]
        
        # Objective: minimize squared errors in implied vol
        def objective(params):
            alpha, rho, nu = params
            model_ivs = self._sabr_implied_vol(
                strikes, forward, maturity, alpha, self.beta, rho, nu
            )
            errors = (model_ivs - market_ivs)**2
            return np.sum(errors)
        
        # Optimize using SLSQP (respects bounds, fast for 3D)
        result = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            options={'ftol': 1e-9}
        )
        
        alpha, rho, nu = result.x
        
        # Compute fit quality
        model_ivs = self._sabr_implied_vol(
            strikes, forward, maturity, alpha, self.beta, rho, nu
        )
        rmse = np.sqrt(np.mean((model_ivs - market_ivs)**2))
        
        return {
            'alpha': alpha,
            'beta': self.beta,
            'rho': rho,
            'nu': nu,
            'rmse': rmse,
            'maturity': maturity
        }
    
    def _sabr_implied_vol(self, K, F, T, alpha, beta, rho, nu):
        """
        Hagan et al. (2002) asymptotic formula for SABR implied volatility.
        
        Valid for strikes not too far from forward (|K - F| / F < 0.5).
        """
        
        # ATM case (K ≈ F)
        atm_mask = np.abs(K - F) < 1e-6
        
        iv = np.zeros_like(K, dtype=float)
        
        # ATM formula (simpler)
        if np.any(atm_mask):
            F_atm = F
            iv[atm_mask] = (alpha / F_atm**(1-beta)) * (
                1 + ((1-beta)**2/24 * alpha**2/F_atm**(2-2*beta)
                     + 0.25*rho*beta*nu*alpha/F_atm**(1-beta)
                     + (2-3*rho**2)/24 * nu**2) * T
            )
        
        # Non-ATM formula
        non_atm_mask = ~atm_mask
        if np.any(non_atm_mask):
            K_slice = K[non_atm_mask]
            
            # Intermediate calculations
            FK = F * K_slice
            FK_avg = (F + K_slice) / 2
            
            if beta == 0:
                # Normal SABR
                gamma1 = 0
                gamma2 = 0
                z = (nu/alpha) * (F - K_slice)
            elif beta == 1:
                # Lognormal SABR
                gamma1 = 0
                gamma2 = 0
                z = (nu/alpha) * np.log(F/K_slice)
            else:
                # General case
                gamma1 = (1-beta) / FK_avg**(1-beta)
                gamma2 = gamma1**2 * ((1-beta)*(2-beta)) / 6
                z = (nu/alpha) * (F**(1-beta) - K_slice**(1-beta)) / (1-beta)
            
            # Function χ(z)
            chi_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
            
            # Handle small z (use Taylor expansion to avoid 0/0)
            small_z = np.abs(z) < 1e-6
            chi_z[small_z] = 1.0  # lim z→0 of z/χ(z) = 1
            
            # First term
            term1 = alpha * z / chi_z
            
            # Second term (correction for finite T)
            term2 = 1 + (
                (1-beta)**2/24 * alpha**2 / FK_avg**(2-2*beta)
                + 0.25 * rho*beta*nu*alpha / FK_avg**(1-beta)
                + (2 - 3*rho**2)/24 * nu**2
            ) * T
            
            iv[non_atm_mask] = term1 * term2
        
        return iv
    
    def calibrate_surface(self, volatility_surface_df):
        """
        Calibrate SABR to entire volatility surface (all maturities).
        
        Args:
            volatility_surface_df: DataFrame with columns 
                ['strike', 'maturity', 'forward', 'implied_vol']
        
        Returns:
            dict: {maturity: sabr_params, ...}
        """
        calibrated_params = {}
        
        for maturity in volatility_surface_df['maturity'].unique():
            subset = volatility_surface_df[
                volatility_surface_df['maturity'] == maturity
            ]
            
            strikes = subset['strike'].values
            market_ivs = subset['implied_vol'].values
            forward = subset['forward'].iloc[0]
            
            params = self.calibrate_smile(
                strikes, market_ivs, forward, maturity
            )
            
            calibrated_params[maturity] = params
        
        return calibrated_params
```

**Performance Requirements:**
- Single smile calibration: <1 second
- Full surface (10 maturities): <10 seconds
- Accuracy: IV fit within 1% (10 basis points) for liquid strikes

### 4.3 Component: Ornstein-Uhlenbeck Fitter

**Purpose:** Estimate OU parameters from spread/price time series

**Inputs:**
- Price spread time series (daily data, 252+ observations)
- Transaction costs
- Discount rate

**Outputs:**
- OU parameters: (θ, μ, σ)
- Half-life of mean reversion
- Goodness-of-fit statistics (likelihood, AIC)
- Optimal entry/exit boundaries

**Algorithm:**

```python
class OUFitter:
    """
    Fits Ornstein-Uhlenbeck process to time series and computes optimal boundaries.
    
    References:
    - Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading with 
      Transaction Costs and Stop-Loss Exit." JIMO.
    - Maximum Likelihood Estimation for OU process.
    """
    
    def fit(self, price_series, dt=1/252):
        """
        Fit OU process using Maximum Likelihood Estimation.
        
        Process: dX_t = μ(θ - X_t)dt + σ dB_t
        
        Args:
            price_series: pandas Series of prices
            dt: time increment in years (1/252 for daily data)
        
        Returns:
            dict: {'theta': θ, 'mu': μ, 'sigma': σ, 'half_life': t_half}
        """
        
        X = price_series.values
        n = len(X)
        
        # Compute lagged differences
        X_t = X[:-1]
        X_t1 = X[1:]
        dX = X_t1 - X_t
        
        # MLE for OU parameters (discrete-time approximation)
        
        # Estimate μ and θ jointly
        Sx = np.sum(X_t)
        Sy = np.sum(X_t1)
        Sxx = np.sum(X_t**2)
        Sxy = np.sum(X_t * X_t1)
        Syy = np.sum(X_t1**2)
        
        # Regression: X_{t+1} = a + b*X_t + ε
        b = (n*Sxy - Sx*Sy) / (n*Sxx - Sx**2)
        a = (Sy - b*Sx) / n
        
        # Convert to OU parameters
        mu = -np.log(b) / dt
        theta = a / (1 - b)
        
        # Estimate σ from residuals
        residuals = X_t1 - (a + b*X_t)
        sigma_discrete = np.std(residuals, ddof=2)
        sigma = sigma_discrete / np.sqrt(dt)
        
        # Half-life of mean reversion
        half_life = np.log(2) / mu
        
        # Goodness of fit: Log-likelihood
        log_likelihood = self._compute_log_likelihood(
            X, theta, mu, sigma, dt
        )
        
        # AIC
        k = 3  # number of parameters
        aic = 2*k - 2*log_likelihood
        
        return {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life_days': half_life * 252,
            'log_likelihood': log_likelihood,
            'aic': aic
        }
    
    def _compute_log_likelihood(self, X, theta, mu, sigma, dt):
        """
        Compute log-likelihood of observed data under OU model.
        
        Transition density is Gaussian with known mean and variance.
        """
        X_t = X[:-1]
        X_t1 = X[1:]
        
        # Conditional mean and variance
        exp_mu_dt = np.exp(-mu * dt)
        mean = theta + (X_t - theta) * exp_mu_dt
        var = (sigma**2 / (2*mu)) * (1 - exp_mu_dt**2)
        
        # Log-likelihood
        log_like = -0.5 * np.sum(
            np.log(2*np.pi*var) + ((X_t1 - mean)**2 / var)
        )
        
        return log_like
    
    def compute_optimal_boundaries(
        self, 
        ou_params, 
        transaction_cost=0.002,
        stop_loss=None,
        discount_rate=0.05
    ):
        """
        Compute optimal entry and exit boundaries using Leung & Li (2015) framework.
        
        Solves optimal stopping problem numerically.
        
        Args:
            ou_params: dict from fit() method
            transaction_cost: c (round-trip cost as fraction)
            stop_loss: L (absolute price level, or None for no stop-loss)
            discount_rate: r (annualized)
        
        Returns:
            dict: {
                'entry_lower': a*, 
                'entry_upper': b*,
                'exit_target': exit level,
                'expected_return': E[profit],
                'sharpe_ratio': SR
            }
        """
        
        theta = ou_params['theta']
        mu = ou_params['mu']
        sigma = ou_params['sigma']
        
        # Solve for optimal entry boundaries
        # This requires solving transcendental equations involving modified Bessel functions
        # For production: use numerical solver
        
        # Simplified heuristic (for illustration):
        # Enter when price deviates by k standard deviations from mean
        # k calibrated to maximize Sharpe ratio
        
        std_dev = sigma / np.sqrt(2*mu)  # Steady-state std dev
        
        def sharpe_for_threshold(k):
            """
            Compute expected Sharpe ratio for entry at ±k std dev from mean.
            """
            entry_lower = theta - k*std_dev
            entry_upper = theta + k*std_dev
            
            # Expected return (simplified)
            expected_return = k*std_dev - transaction_cost
            
            # Expected holding time (first-passage time approximation)
            holding_time = 1.0 / mu  # Order of magnitude
            
            # Variance of return (approximation)
            var_return = 2*std_dev**2
            
            # Sharpe ratio
            if var_return > 0:
                sharpe = (expected_return / np.sqrt(var_return)) * np.sqrt(1/holding_time)
            else:
                sharpe = 0
            
            return sharpe
        
        # Optimize k
        result = minimize_scalar(
            lambda k: -sharpe_for_threshold(k),  # Maximize SR
            bounds=(1.0, 4.0),
            method='bounded'
        )
        
        k_optimal = result.x
        
        entry_lower = theta - k_optimal*std_dev
        entry_upper = theta + k_optimal*std_dev
        
        # Exit target: mean (simplified; full solution more complex)
        exit_target = theta
        
        # Adjust if stop-loss provided
        if stop_loss is not None:
            # Per Leung & Li: higher stop-loss → lower take-profit
            # Heuristic adjustment
            if entry_lower < stop_loss:
                entry_lower = stop_loss + 0.01  # Enter above stop-loss
                exit_target = max(stop_loss + k_optimal*std_dev*0.5, theta)
        
        return {
            'entry_lower': entry_lower,
            'entry_upper': entry_upper,
            'exit_target': exit_target,
            'k_optimal': k_optimal,
            'sharpe_estimate': -result.fun
        }
```

**Performance Requirements:**
- MLE fitting: <1 second for 500 data points
- Boundary optimization: <5 seconds
- Stability: Ensure μ > 0 (mean-reverting), handle μ ≈ 0 case gracefully

---

## 5. Data Pipeline

### 5.1 Data Sources

| Data Type | Source | Frequency | Latency | Critical? |
|-----------|--------|-----------|---------|-----------|
| Equity prices | Bloomberg/IEX | Real-time | <100ms | Yes |
| Options chain | CBOE/IVolatility | Real-time | <1s | Yes |
| Volatility surface | Vendor (OptionMetrics) | EOD | T+1 | No |
| Risk-free rates | Federal Reserve | Daily | T+1 | No |
| Earnings dates | FactSet | Daily | T+1 | No |
| Alternative data | Custom | Varies | Varies | No |

### 5.2 Data Validation Pipeline

```python
class DataValidator:
    """
    Validates incoming market data for quality and consistency.
    
    Checks:
    - No-arbitrage conditions (put-call parity, calendar spreads)
    - Data completeness (no missing strikes/maturities)
    - Outlier detection (IV > 200% flagged)
    - Staleness (timestamp checks)
    """
    
    def validate_option_chain(self, option_data):
        """
        Validate options data before calibration.
        
        Returns: (is_valid: bool, errors: list)
        """
        errors = []
        
        # Check put-call parity
        parity_errors = self._check_put_call_parity(option_data)
        errors.extend(parity_errors)
        
        # Check for negative implied vols (should never happen)
        negative_iv = option_data[option_data['implied_vol'] < 0]
        if len(negative_iv) > 0:
            errors.append(f"Negative IVs detected: {len(negative_iv)} options")
        
        # Check for excessive IVs (likely data errors)
        extreme_iv = option_data[option_data['implied_vol'] > 2.0]
        if len(extreme_iv) > 0:
            errors.append(f"Extreme IVs (>200%): {len(extreme_iv)} options")
        
        # Check bid-ask spread reasonableness
        wide_spreads = option_data[
            (option_data['ask'] - option_data['bid']) / option_data['mid'] > 0.5
        ]
        if len(wide_spreads) > 0:
            errors.append(f"Wide bid-ask spreads: {len(wide_spreads)} options")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _check_put_call_parity(self, options, tolerance=0.05):
        """
        Verify C - P = S - K*exp(-rT) within tolerance.
        """
        # Implementation details...
        pass
```

### 5.3 Storage Schema (TimescaleDB)

```sql
-- Time-series optimized tables for market data

CREATE TABLE market_prices (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    price NUMERIC(12, 4),
    volume BIGINT,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('market_prices', 'time');

CREATE TABLE option_quotes (
    time TIMESTAMPTZ NOT NULL,
    underlying TEXT NOT NULL,
    expiration DATE NOT NULL,
    strike NUMERIC(10, 2),
    option_type TEXT, -- 'call' or 'put'
    bid NUMERIC(10, 4),
    ask NUMERIC(10, 4),
    implied_vol NUMERIC(6, 4),
    delta NUMERIC(6, 4),
    gamma NUMERIC(8, 6),
    vega NUMERIC(8, 4),
    PRIMARY KEY (time, underlying, expiration, strike, option_type)
);

SELECT create_hypertable('option_quotes', 'time');

CREATE TABLE model_parameters (
    time TIMESTAMPTZ NOT NULL,
    model_type TEXT NOT NULL, -- 'heston', 'sabr', 'ou'
    underlying TEXT NOT NULL,
    maturity DATE, -- NULL for OU (not maturity-dependent)
    parameters JSONB, -- Store params as JSON
    fit_quality JSONB, -- RMSE, R², etc.
    PRIMARY KEY (time, model_type, underlying, maturity)
);

CREATE TABLE signals (
    time TIMESTAMPTZ NOT NULL,
    strategy TEXT NOT NULL,
    underlying TEXT,
    signal_type TEXT, -- 'entry_long', 'entry_short', 'exit', 'hold'
    signal_strength NUMERIC(4, 3), -- 0.0 to 1.0
    metadata JSONB,
    PRIMARY KEY (time, strategy, underlying)
);

CREATE INDEX idx_signals_time_strategy ON signals(time, strategy);

CREATE TABLE positions (
    position_id UUID PRIMARY KEY,
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    strategy TEXT,
    underlying TEXT,
    direction TEXT, -- 'long' or 'short'
    quantity NUMERIC(12, 2),
    entry_price NUMERIC(12, 4),
    exit_price NUMERIC(12, 4),
    pnl NUMERIC(12, 2),
    metadata JSONB
);

CREATE INDEX idx_positions_opened_at ON positions(opened_at);
CREATE INDEX idx_positions_strategy ON positions(strategy);
```

---

## 6. Model Calibration Engine

### 6.1 Calibration Orchestrator

**Purpose:** Coordinate daily model calibration across all strategies

**Schedule:**
- Pre-market: 8:00 AM ET (prepare for day)
- Intraday: Every 15 minutes (monitor for opportunities)
- Post-market: 5:00 PM ET (full recalibration)

**Workflow:**

```python
class CalibrationOrchestrator:
    """
    Manages daily calibration of all quantitative models.
    """
    
    def __init__(self, config):
        self.heston_calibrator = HestonCalibrator()
        self.sabr_calibrator = SABRCalibrator(beta=0.5)
        self.ou_fitter = OUFitter()
        self.db = TimeSeriesDB(config['database_url'])
        self.cache = Redis(config['redis_url'])
    
    async def run_daily_calibration(self):
        """
        Execute full calibration pipeline.
        
        Steps:
        1. Fetch market data
        2. Validate data quality
        3. Calibrate models in parallel
        4. Store results
        5. Generate alerts if models significantly changed
        """
        
        logger.info("Starting daily calibration")
        
        # Fetch data
        option_data = await self._fetch_option_data()
        spreads_data = await self._fetch_spreads_data()
        
        # Validate
        validator = DataValidator()
        is_valid, errors = validator.validate_option_chain(option_data)
        
        if not is_valid:
            logger.error(f"Data validation failed: {errors}")
            # Use cached parameters from previous day
            return await self._load_cached_parameters()
        
        # Calibrate models (parallel execution)
        results = await asyncio.gather(
            self._calibrate_heston(option_data),
            self._calibrate_sabr(option_data),
            self._calibrate_ou(spreads_data),
            return_exceptions=True
        )
        
        heston_params, sabr_params, ou_params = results
        
        # Check for calibration failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Calibration {i} failed: {result}")
                # Fall back to cached params
                results[i] = await self._load_cached_param(i)
        
        # Store results
        await self._store_parameters(heston_params, sabr_params, ou_params)
        
        # Alert if significant parameter shifts
        await self._check_parameter_shifts(heston_params, sabr_params, ou_params)
        
        logger.info("Calibration complete")
        
        return {
            'heston': heston_params,
            'sabr': sabr_params,
            'ou': ou_params
        }
```

### 6.2 Performance Optimization

**GPU Acceleration:**
- Heston calibration: Use CUDA for parallel option pricing across parameter grid
- Reference: Chen & Palomar (2018) show 100x+ speedup for Monte Carlo in Heston
- Expected improvement: 30s → 0.3s for full surface calibration

**Caching Strategy:**
- Cache characteristic function evaluations (Heston)
- Cache Bessel function values (OU optimal boundaries)
- Invalidate cache daily

**Adaptive Calibration:**
- If market relatively stable (VIX change < 5%), use warm-start from yesterday's params
- Only run full global optimization if markets volatile or fit quality degraded

---

## 7. Signal Generation

### 7.1 Strategy: Volatility Surface Arbitrage

**Logic:**

Compare model-implied volatility to market-implied volatility. Trade when divergence exceeds threshold.

```python
class VolSurfaceArbitrageSignal:
    """
    Detects mispricing in options by comparing Heston/SABR model to market.
    
    Signal: Trade delta-hedged option when |σ_model - σ_market| > threshold
    """
    
    def generate_signal(self, calibrated_params, market_data):
        """
        Generate trading signals based on vol surface mispricing.
        
        Returns:
            list of dict: [
                {
                    'underlying': 'SPY',
                    'option': 'SPY 420 Call 2026-03-20',
                    'signal': 'buy',  # Buy option, hedge with stock
                    'confidence': 0.85,
                    'rationale': 'Market IV 25%, Model IV 30%, underpriced'
                },
                ...
            ]
        """
        
        signals = []
        
        for option in market_data.itertuples():
            # Compute model-implied IV
            model_iv = self._compute_model_iv(option, calibrated_params)
            market_iv = option.implied_vol
            
            # Divergence
            divergence = model_iv - market_iv
            divergence_pct = divergence / market_iv
            
            # Threshold: 10% relative divergence
            if abs(divergence_pct) > 0.10:
                
                # Determine direction
                if divergence > 0:
                    # Model says option should be more expensive → buy
                    signal_type = 'buy'
                    rationale = f"Market IV {market_iv:.1%}, Model IV {model_iv:.1%}, underpriced"
                else:
                    # Model says option should be cheaper → sell
                    signal_type = 'sell'
                    rationale = f"Market IV {market_iv:.1%}, Model IV {model_iv:.1%}, overpriced"
                
                # Confidence based on fit quality and liquidity
                confidence = self._compute_confidence(option, calibrated_params)
                
                if confidence > 0.6:  # Only high-confidence signals
                    signals.append({
                        'underlying': option.underlying,
                        'option_id': option.option_id,
                        'strike': option.strike,
                        'expiration': option.expiration,
                        'signal': signal_type,
                        'confidence': confidence,
                        'divergence_pct': divergence_pct,
                        'rationale': rationale,
                        'model_iv': model_iv,
                        'market_iv': market_iv
                    })
        
        return signals
    
    def _compute_confidence(self, option, params):
        """
        Compute confidence score based on:
        - Model fit quality (RMSE)
        - Option liquidity (bid-ask spread, volume)
        - Time to expiration (avoid very short-dated)
        """
        
        # Fit quality component
        fit_score = 1.0 - min(params['rmse'], 0.5) / 0.5
        
        # Liquidity component
        bid_ask_spread = (option.ask - option.bid) / option.mid
        liquidity_score = max(0, 1.0 - bid_ask_spread / 0.1)  # Penalize if spread > 10%
        
        # Maturity component (avoid <7 days, prefer 30-90 days)
        days_to_expiry = (option.expiration - pd.Timestamp.now()).days
        if days_to_expiry < 7:
            maturity_score = 0.3
        elif 30 <= days_to_expiry <= 90:
            maturity_score = 1.0
        else:
            maturity_score = 0.7
        
        # Weighted average
        confidence = (
            0.4 * fit_score +
            0.4 * liquidity_score +
            0.2 * maturity_score
        )
        
        return confidence
```

### 7.2 Strategy: Optimal Mean-Reversion Trading

**Logic:**

Monitor spreads fitted to OU process. Enter when spread crosses optimal entry boundaries, exit at optimal take-profit or stop-loss.

```python
class MeanReversionSignal:
    """
    Generates signals based on Ornstein-Uhlenbeck optimal stopping framework.
    
    Reference: Leung, T., & Li, X. (2015). "Optimal Mean Reversion Trading."
    """
    
    def generate_signal(self, spread_name, current_spread, ou_params, boundaries):
        """
        Determine if current spread level warrants entry/exit.
        
        Args:
            spread_name: identifier (e.g., 'SPY-IWM')
            current_spread: current price spread
            ou_params: {'theta': θ, 'mu': μ, 'sigma': σ}
            boundaries: {'entry_lower': a*, 'entry_upper': b*, 'exit_target': e*}
        
        Returns:
            dict or None: Signal if action warranted, else None
        """
        
        theta = ou_params['theta']
        entry_lower = boundaries['entry_lower']
        entry_upper = boundaries['entry_upper']
        exit_target = boundaries['exit_target']
        
        # Check if currently in position
        position = self._get_current_position(spread_name)
        
        if position is None:
            # Not in position → check entry conditions
            
            if current_spread < entry_lower:
                # Spread below lower boundary → enter long
                return {
                    'spread': spread_name,
                    'signal': 'entry_long',
                    'confidence': self._entry_confidence(current_spread, entry_lower),
                    'rationale': f"Spread {current_spread:.4f} < entry lower {entry_lower:.4f}, expect reversion to {theta:.4f}"
                }
            
            elif current_spread > entry_upper:
                # Spread above upper boundary → enter short
                return {
                    'spread': spread_name,
                    'signal': 'entry_short',
                    'confidence': self._entry_confidence(current_spread, entry_upper),
                    'rationale': f"Spread {current_spread:.4f} > entry upper {entry_upper:.4f}, expect reversion to {theta:.4f}"
                }
        
        else:
            # In position → check exit conditions
            direction = position['direction']
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            
            # Check stop-loss first
            if (direction == 'long' and current_spread <= stop_loss) or \
               (direction == 'short' and current_spread >= stop_loss):
                return {
                    'spread': spread_name,
                    'signal': 'exit_stop_loss',
                    'confidence': 1.0,
                    'rationale': f"Stop-loss triggered at {current_spread:.4f}"
                }
            
            # Check take-profit (mean reversion achieved)
            if direction == 'long':
                if current_spread >= exit_target:
                    pnl = current_spread - entry_price
                    return {
                        'spread': spread_name,
                        'signal': 'exit_take_profit',
                        'confidence': 0.9,
                        'pnl': pnl,
                        'rationale': f"Take-profit at {current_spread:.4f}, gained {pnl:.4f}"
                    }
            else:  # short
                if current_spread <= exit_target:
                    pnl = entry_price - current_spread
                    return {
                        'spread': spread_name,
                        'signal': 'exit_take_profit',
                        'confidence': 0.9,
                        'pnl': pnl,
                        'rationale': f"Take-profit at {current_spread:.4f}, gained {pnl:.4f}"
                    }
        
        return None  # No signal
    
    def _entry_confidence(self, current_price, boundary):
        """
        Confidence increases with distance from boundary (more extreme = higher confidence).
        """
        # Simple linear scaling: max confidence when 2x sigma away
        sigma = self.ou_params['sigma']
        distance = abs(current_price - boundary)
        confidence = min(0.95, 0.6 + (distance / (2*sigma)) * 0.35)
        return confidence
```

### 7.3 Signal Aggregation

**Multi-strategy portfolio construction:**

```python
class SignalAggregator:
    """
    Combines signals from multiple strategies into unified portfolio decisions.
    
    Handles:
    - Signal conflicts (e.g., vol arb says buy, mean-rev says sell)
    - Correlation between strategies
    - Overall portfolio risk budgeting
    """
    
    def aggregate_signals(self, all_signals, current_portfolio):
        """
        Aggregate signals and determine final trades.
        
        Args:
            all_signals: dict of {strategy_name: [signal1, signal2, ...]}
            current_portfolio: current positions and risk metrics
        
        Returns:
            list of final trading decisions
        """
        
        final_trades = []
        
        # Flatten all signals
        flat_signals = []
        for strategy, signals in all_signals.items():
            for sig in signals:
                sig['strategy'] = strategy
                flat_signals.append(sig)
        
        # Group by underlying/spread
        from collections import defaultdict
        grouped = defaultdict(list)
        for sig in flat_signals:
            key = sig.get('underlying') or sig.get('spread')
            grouped[key].append(sig)
        
        # For each asset, reconcile signals
        for asset, signals in grouped.items():
            
            # If unanimous, take action
            if len(signals) == 1:
                final_trades.append(signals[0])
            
            # If conflicting, use weighted voting by confidence
            else:
                buy_weight = sum(s['confidence'] for s in signals if 'buy' in s['signal'] or 'long' in s['signal'])
                sell_weight = sum(s['confidence'] for s in signals if 'sell' in s['signal'] or 'short' in s['signal'])
                
                if buy_weight > sell_weight * 1.5:
                    # Strong buy consensus
                    final_trades.append({
                        'asset': asset,
                        'signal': 'buy',
                        'confidence': buy_weight / len(signals),
                        'supporting_strategies': [s['strategy'] for s in signals if 'buy' in s['signal'] or 'long' in s['signal']]
                    })
                elif sell_weight > buy_weight * 1.5:
                    # Strong sell consensus
                    final_trades.append({
                        'asset': asset,
                        'signal': 'sell',
                        'confidence': sell_weight / len(signals),
                        'supporting_strategies': [s['strategy'] for s in signals if 'sell' in s['signal'] or 'short' in s['signal']]
                    })
                # Else: conflicting signals, no action
        
        return final_trades
```

---

## 8. Risk Management

### 8.1 Position Sizing with Volatility Scaling

**Reference:** Moreira, A., & Muir, T. (2017). "Volatility-managed portfolios." *Journal of Finance*, 72(4), 1611-1644.

**Implementation:**

```python
class VolatilityScaledPositionSizer:
    """
    Implements volatility-managed position sizing per Moreira & Muir (2017).
    
    Core idea: w_t = c / σ_t²
    
    Scale exposure inversely with realized volatility to improve Sharpe ratio.
    """
    
    def __init__(self, target_annual_vol=0.15):
        """
        Args:
            target_annual_vol: Target portfolio volatility (e.g., 15%)
        """
        self.target_annual_vol = target_annual_vol
    
    def compute_position_size(self, strategy_return_series, available_capital):
        """
        Compute position size based on recent realized volatility.
        
        Args:
            strategy_return_series: pandas Series of daily returns (last 21 days)
            available_capital: total capital allocated to this strategy
        
        Returns:
            position_size: dollar amount to allocate
        """
        
        # Compute realized volatility (21-day rolling)
        realized_vol = strategy_return_series.std() * np.sqrt(252)  # Annualize
        
        # Avoid division by zero
        if realized_vol < 0.01:
            realized_vol = 0.01
        
        # Target weight: inversely proportional to variance
        target_weight = (self.target_annual_vol**2) / (realized_vol**2)
        
        # Cap at 2.0 (no more than 2x leverage)
        target_weight = min(target_weight, 2.0)
        
        # Floor at 0.2 (at least 20% exposure even in high vol)
        target_weight = max(target_weight, 0.2)
        
        position_size = available_capital * target_weight
        
        return position_size
```

### 8.2 Greeks Monitoring (For Options Strategies)

```python
class GreeksRiskMonitor:
    """
    Monitor portfolio Greeks and rebalance hedges.
    
    Key metrics:
    - Delta: Sensitivity to underlying price (target: ~0 for market-neutral)
    - Gamma: Convexity of delta (manage to avoid large moves on re-hedging)
    - Vega: Sensitivity to volatility (target: positive for vol arb)
    - Theta: Time decay (offset with positive carry)
    """
    
    def compute_portfolio_greeks(self, positions):
        """
        Aggregate Greeks across all option positions.
        
        Returns:
            dict: {'delta': X, 'gamma': Y, 'vega': Z, 'theta': W}
        """
        
        total_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0
        }
        
        for pos in positions:
            quantity = pos['quantity'] * (1 if pos['direction'] == 'long' else -1)
            
            total_greeks['delta'] += quantity * pos['delta']
            total_greeks['gamma'] += quantity * pos['gamma']
            total_greeks['vega'] += quantity * pos['vega']
            total_greeks['theta'] += quantity * pos['theta']
        
        return total_greeks
    
    def check_rehedge_needed(self, portfolio_greeks, thresholds):
        """
        Determine if delta-hedging or other rebalancing is required.
        
        Args:
            portfolio_greeks: current portfolio Greeks
            thresholds: dict of acceptable ranges
        
        Returns:
            bool, list of actions
        """
        
        actions = []
        
        # Delta neutrality check
        if abs(portfolio_greeks['delta']) > thresholds.get('delta', 100):
            # Rehedge with underlying
            hedge_quantity = -portfolio_greeks['delta']
            actions.append({
                'action': 'hedge_delta',
                'quantity': hedge_quantity,
                'rationale': f"Portfolio delta {portfolio_greeks['delta']:.0f} exceeds threshold"
            })
        
        # Gamma exposure check
        if abs(portfolio_greeks['gamma']) > thresholds.get('gamma', 50):
            actions.append({
                'action': 'alert',
                'message': f"High gamma exposure: {portfolio_greeks['gamma']:.2f}, monitor for large moves"
            })
        
        return len(actions) > 0, actions
```

### 8.3 Correlation Monitoring

**Purpose:** Detect when spreads lose cointegration or correlations break down

```python
class CorrelationMonitor:
    """
    Monitor rolling correlation and cointegration for mean-reversion strategies.
    
    Alert if:
    - Cointegration test fails (Engle-Granger p-value > 0.05)
    - Correlation drops below threshold
    - Half-life of mean reversion increases significantly
    """
    
    def check_cointegration_health(self, spread_data, ou_params):
        """
        Validate that spread remains cointegrated.
        
        Returns:
            dict: {'is_healthy': bool, 'warnings': list}
        """
        
        warnings = []
        
        # Re-run Engle-Granger test on recent data
        from statsmodels.tsa.stattools import coint
        
        # Assuming spread_data has ['asset1', 'asset2'] columns
        _, p_value, _ = coint(spread_data['asset1'], spread_data['asset2'])
        
        if p_value > 0.05:
            warnings.append(f"Cointegration test failed: p-value {p_value:.3f}")
        
        # Check if half-life increased (slower mean reversion)
        current_half_life = ou_params['half_life_days']
        historical_half_life = self._get_historical_half_life(spread_data)
        
        if current_half_life > historical_half_life * 1.5:
            warnings.append(f"Half-life increased: {current_half_life:.1f} days vs historical {historical_half_life:.1f} days")
        
        is_healthy = len(warnings) == 0
        
        return {'is_healthy': is_healthy, 'warnings': warnings}
```

---

## 9. Execution Layer

### 9.1 Order Management System

**Requirements:**
- Smart order routing (optimize execution venue)
- Transaction cost analysis (TCA)
- Slippage monitoring
- Partial fill handling

**Example Implementation:**

```python
class OrderManager:
    """
    Manages order lifecycle from signal to execution.
    """
    
    def execute_trade(self, signal, portfolio_state):
        """
        Execute trade based on signal.
        
        Steps:
        1. Compute order size (from position sizer)
        2. Select execution algorithm (TWAP, VWAP, etc.)
        3. Route order to broker/exchange
        4. Monitor fill
        5. Update position tracking
        """
        
        # Compute quantity
        quantity = self.position_sizer.compute_quantity(signal, portfolio_state)
        
        # Create order
        order = {
            'symbol': signal['underlying'],
            'side': 'buy' if 'buy' in signal['signal'] else 'sell',
            'quantity': quantity,
            'order_type': 'limit',  # Use limit orders to control slippage
            'limit_price': self._compute_limit_price(signal),
            'time_in_force': 'DAY',
            'strategy': signal['strategy']
        }
        
        # Submit order
        order_id = self.broker_api.submit_order(order)
        
        # Log
        self.db.log_order(order_id, order, signal)
        
        # Monitor asynchronously
        asyncio.create_task(self._monitor_order_fill(order_id))
        
        return order_id
    
    def _compute_limit_price(self, signal):
        """
        Compute aggressive limit price (pay up a bit for faster fill).
        
        For buys: bid + X% of spread
        For sells: ask - X% of spread
        
        X = 30% (willing to pay 30% of spread for faster execution)
        """
        
        quote = self.get_current_quote(signal['underlying'])
        spread = quote['ask'] - quote['bid']
        
        if 'buy' in signal['signal']:
            limit_price = quote['bid'] + 0.3 * spread
        else:
            limit_price = quote['ask'] - 0.3 * spread
        
        return round(limit_price, 2)
```

### 9.2 Transaction Cost Model

**Reference:** Frazzini, A., Israel, R., & Moskowitz, T. J. (2012). "Trading costs of asset pricing anomalies." *Financial Analysts Journal*, 68(2), 103-119.

**Cost components:**
1. Bid-ask spread
2. Market impact (price moves against you)
3. Timing cost (delay between signal and execution)
4. Commissions

```python
class TransactionCostModel:
    """
    Estimate transaction costs before executing trade.
    
    Used to filter signals where expected profit < transaction cost.
    """
    
    def estimate_total_cost(self, symbol, quantity, side):
        """
        Estimate total transaction cost for a trade.
        
        Returns:
            cost_bps: cost in basis points
        """
        
        # Get current market data
        quote = self.get_quote(symbol)
        daily_volume = self.get_avg_daily_volume(symbol)
        
        # Component 1: Bid-ask spread
        spread_bps = ((quote['ask'] - quote['bid']) / quote['mid']) * 10000
        
        # Component 2: Market impact (Almgren-Chriss model)
        # Impact ≈ σ * sqrt(Q / V) where Q = shares, V = daily volume
        volatility = self.get_realized_vol(symbol)  # Daily vol
        participation_rate = quantity / daily_volume
        
        if participation_rate > 0.05:
            # High impact expected
            impact_bps = volatility * np.sqrt(participation_rate) * 10000
        else:
            impact_bps = 0
        
        # Component 3: Commission
        commission_bps = 0.5  # $0.005 per share ≈ 0.5 bps
        
        # Total
        total_cost_bps = spread_bps + impact_bps + commission_bps
        
        return total_cost_bps
```

---

## 10. Performance Monitoring

### 10.1 Real-time Dashboard Metrics

**Key Performance Indicators (KPIs):**

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Daily Sharpe Ratio | >0.15 (annualized ~0.8) | <0.05 for 5 consecutive days |
| Max Drawdown (rolling 60d) | <15% | >20% |
| Win Rate | >55% | <45% |
| Average Trade Duration | 30-50 days | >80 days (mean-reversion not working) |
| Portfolio Volatility | 15% annualized | >25% |
| Model Calibration RMSE | <3% | >5% (model not fitting market) |

### 10.2 Performance Attribution

**Question:** Which strategy contributed most to returns?

```python
class PerformanceAttribution:
    """
    Decompose portfolio returns by strategy and risk factor.
    """
    
    def attribute_returns(self, portfolio_returns, strategy_positions):
        """
        Brinson attribution: decompose into strategy-level contributions.
        
        Returns:
            DataFrame with columns: ['strategy', 'return_contribution', 'weight']
        """
        
        attribution = []
        
        for strategy in strategy_positions['strategy'].unique():
            
            # Filter positions for this strategy
            strat_positions = strategy_positions[
                strategy_positions['strategy'] == strategy
            ]
            
            # Compute strategy-level return
            strat_return = (
                strat_positions['pnl'].sum() / 
                strat_positions['capital_allocated'].sum()
            )
            
            # Weight in portfolio
            weight = (
                strat_positions['capital_allocated'].sum() / 
                portfolio_returns['total_capital']
            )
            
            # Contribution
            contribution = strat_return * weight
            
            attribution.append({
                'strategy': strategy,
                'return': strat_return,
                'weight': weight,
                'contribution': contribution
            })
        
        return pd.DataFrame(attribution)
```

### 10.3 Alerting System

**Trigger alerts for:**
- Model calibration failure
- Position breaches risk limits
- Strategy Sharpe ratio deteriorates
- Cointegration breaks down
- Execution slippage >2%

```python
class AlertManager:
    """
    Send alerts via email/Slack when anomalies detected.
    """
    
    def check_and_alert(self, metrics):
        """
        Evaluate metrics and trigger alerts if needed.
        """
        
        alerts = []
        
        # Check Sharpe ratio
        if metrics['sharpe_ratio'] < 0.05:
            alerts.append({
                'severity': 'HIGH',
                'message': f"Sharpe ratio dropped to {metrics['sharpe_ratio']:.3f}",
                'action': 'Review strategy performance and consider reducing positions'
            })
        
        # Check drawdown
        if metrics['drawdown'] > 0.20:
            alerts.append({
                'severity': 'CRITICAL',
                'message': f"Drawdown {metrics['drawdown']:.1%} exceeds 20% threshold",
                'action': 'Reduce risk immediately'
            })
        
        # Check model fit
        if metrics['calibration_rmse'] > 0.05:
            alerts.append({
                'severity': 'MEDIUM',
                'message': f"Model calibration RMSE {metrics['calibration_rmse']:.3f} poor",
                'action': 'Investigate data quality or consider model re-specification'
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
```

---

## 11. Technology Stack

### 11.1 Core Technologies

| Component | Technology | Justification |
|-----------|------------|---------------|
| Application Language | Python 3.11+ | NumPy/SciPy for numerical computing, extensive quant libraries |
| Numerical Computing | NumPy, SciPy, numba | Fast vectorized operations, JIT compilation |
| Optimization | scipy.optimize, cvxpy | Global and local optimization, convex programming |
| PDE Solvers | FiPy, custom finite difference | Solve Heston/SABR PDEs numerically |
| Time Series DB | TimescaleDB (PostgreSQL extension) | Optimized for time-series queries, SQL compatibility |
| Caching | Redis | Fast parameter caching, session management |
| Message Queue | RabbitMQ | Asynchronous task processing (calibration jobs) |
| Backtesting | Backtrader, custom framework | Vectorized backtesting, event-driven simulation |
| Monitoring | Grafana + Prometheus | Real-time dashboards, alerting |
| Deployment | Docker + Kubernetes | Containerization, orchestration, scalability |
| CI/CD | GitHub Actions | Automated testing, deployment |

### 11.2 Python Libraries

**Quant-specific:**
```
numpy>=1.24
scipy>=1.10
pandas>=2.0
numba>=0.57         # JIT compilation for performance
cvxpy>=1.3          # Convex optimization
statsmodels>=0.14   # Time series analysis (cointegration tests)
arch>=5.3           # GARCH, volatility modeling
pysabr>=0.4         # SABR model implementation
QuantLib>=1.30      # Comprehensive derivatives pricing
```

**Data & Infrastructure:**
```
psycopg2-binary>=2.9    # PostgreSQL driver
redis>=4.5              # Redis client
pika>=1.3               # RabbitMQ client
sqlalchemy>=2.0         # ORM
asyncio, aiohttp        # Async I/O
```

**Monitoring & Logging:**
```
prometheus-client>=0.17
structlog>=23.1
sentry-sdk>=1.20
```

### 11.3 Hardware Requirements

**Production Environment:**

| Component | Specification | Purpose |
|-----------|---------------|---------|
| CPU | 32 cores (AMD EPYC or Intel Xeon) | Parallel calibration, optimization |
| RAM | 128 GB | In-memory data caching, large matrix operations |
| GPU | 1x NVIDIA A100 (optional) | GPU-accelerated Heston calibration |
| Storage | 2 TB NVMe SSD | Fast data access for time-series queries |
| Network | 10 Gbps | Low-latency market data feeds |

**Development Environment:**
- 16-core CPU, 64 GB RAM sufficient for dev/test

---

## 12. Deployment Architecture

### 12.1 Microservices Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Load Balancer                        │
│                         (NGINX)                              │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   API Server  │  │ Data Ingestion│  │  Calibration  │
│   (FastAPI)   │  │   Service     │  │    Service    │
└───────────────┘  └───────────────┘  └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────┴────────┐
                    │                │
                    ▼                ▼
            ┌──────────────┐  ┌──────────────┐
            │ TimescaleDB  │  │    Redis     │
            │              │  │   (Cache)    │
            └──────────────┘  └──────────────┘
```

### 12.2 Kubernetes Deployment

**deployment.yaml** (simplified):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: calibration-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: calibration
  template:
    metadata:
      labels:
        app: calibration
    spec:
      containers:
      - name: calibration
        image: quant-trading/calibration:1.0
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
          limits:
            memory: "32Gi"
            cpu: "16"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
```

### 12.3 Disaster Recovery & Backup

**Strategy:**
- **Database:** Daily full backup + continuous WAL archiving (TimescaleDB)
- **Model Parameters:** Version-controlled in Git, daily snapshots to S3
- **Positions:** Real-time replication to standby database
- **Recovery Time Objective (RTO):** <4 hours
- **Recovery Point Objective (RPO):** <15 minutes

---

## 13. Testing & Validation

### 13.1 Unit Tests

**Coverage target:** >85% for core components

Example test for Heston calibrator:

```python
import pytest
import numpy as np

def test_heston_calibrator_convergence():
    """
    Test that Heston calibrator converges to known parameters.
    
    Use synthetic option prices generated with known Heston params,
    verify calibrator recovers parameters within tolerance.
    """
    
    # True parameters
    true_params = {
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7,
        'v0': 0.04
    }
    
    # Generate synthetic option prices
    calibrator = HestonCalibrator()
    strikes = np.linspace(90, 110, 20)
    maturities = [30/365, 60/365, 90/365]
    
    synthetic_prices = calibrator._price_options(
        true_params,
        strikes,
        maturities,
        S0=100,
        r=0.05,
        q=0.02
    )
    
    # Add small noise
    noisy_prices = synthetic_prices + np.random.normal(0, 0.01, len(synthetic_prices))
    
    # Calibrate
    calibrated_params = calibrator.calibrate(
        market_prices=noisy_prices,
        strikes=strikes,
        maturities=maturities,
        S0=100,
        r=0.05,
        q=0.02
    )
    
    # Verify parameters recovered within 10%
    for param in ['kappa', 'theta', 'sigma', 'rho', 'v0']:
        assert abs(calibrated_params[param] - true_params[param]) / true_params[param] < 0.10
```

### 13.2 Backtesting Framework

**Requirements:**
- Walk-forward analysis (no look-ahead bias)
- Transaction costs included
- Realistic slippage model
- Out-of-sample testing mandatory

**Backtesting process:**

1. **In-sample period:** 2018-2021 (train models, optimize parameters)
2. **Out-of-sample period:** 2022-2024 (test with frozen parameters)
3. **Validate:** Ensure out-of-sample Sharpe >50% of in-sample Sharpe

**Benchmark:** Compare against:
- Buy-and-hold S&P 500
- 60/40 stock/bond portfolio
- Pure momentum factor (for mean-rev strategies)

### 13.3 Monte Carlo Stress Testing

**Scenarios:**
- 2008 Financial Crisis
- 2020 COVID Crash
- Flash Crash (May 2010)
- Persistent low volatility (2017)

**Metrics:**
- Max drawdown in each scenario
- Time to recovery
- Sharpe ratio during crisis

---

## 14. Regulatory & Compliance

### 14.1 Model Risk Management

Per Federal Reserve SR 11-7 guidelines:

**Documentation Requirements:**
- Model Development Document (this document)
- Model Validation Report (independent review)
- Ongoing Monitoring Report (quarterly)

**Model Governance:**
- Independent validation by separate quant team
- Annual model review and re-approval
- Escalation process if model performance degrades

### 14.2 Algorithmic Trading Controls

**SEC/FINRA requirements:**
- Pre-trade risk checks (position limits, order size limits)
- Kill switch (ability to stop all trading immediately)
- Audit trail (all orders, executions, and signals logged)
- System testing before production deployment

---

## 15. Roadmap & Future Enhancements

### 15.1 Phase 1 (Months 1-3): Core Infrastructure
- [ ] Implement data ingestion pipeline
- [ ] Build Heston calibrator
- [ ] Build SABR calibrator
- [ ] Build OU fitter
- [ ] Deploy TimescaleDB and basic monitoring

### 15.2 Phase 2 (Months 4-6): Strategy Implementation
- [ ] Implement volatility surface arbitrage signals
- [ ] Implement mean-reversion signals
- [ ] Build position sizing with vol scaling
- [ ] Paper trading (simulated execution)

### 15.3 Phase 3 (Months 7-9): Backtesting & Validation
- [ ] Run comprehensive backtests (10+ years data)
- [ ] Validate model calibration accuracy
- [ ] Perform Monte Carlo stress tests
- [ ] Independent model validation

### 15.4 Phase 4 (Months 10-12): Production Launch
- [ ] Live trading with small capital ($100K)
- [ ] Monitor performance vs backtest
- [ ] Iterate based on live results
- [ ] Scale capital if performance meets targets

### 15.5 Future Research Directions

**Advanced Models:**
- Jump-diffusion models (Bates, Merton)
- Rough volatility models (Gatheral et al., 2018)
- Machine learning for regime detection

**New Strategies:**
- Variance risk premium harvesting
- Volatility carry trades
- Multi-asset optimal execution

**Technology:**
- Real-time GPU calibration
- Quantum computing for optimization (exploratory)

---

## 16. Conclusion

This design document outlines a sophisticated quantitative trading system grounded in rigorous academic research. The system leverages stochastic volatility models (Heston, SABR), optimal stopping theory for mean-reversion, and volatility-managed position sizing to generate alpha in the 1-2 month swing trading horizon.

**Key Success Factors:**
1. **Mathematical rigor:** All models based on published research with proven out-of-sample performance
2. **Robust implementation:** Extensive testing, validation, and error handling
3. **Risk management:** Volatility scaling, Greeks monitoring, correlation checks
4. **Operational excellence:** Monitoring, alerting, disaster recovery

**Expected Performance:**
- Sharpe Ratio: 0.7-1.0 (based on academic benchmarks)
- Max Drawdown: <25%
- Scalability: Strategy capacity ~$50M-$100M before market impact becomes significant

The system is designed to be modular and extensible, allowing for continuous improvement through research and development.

---

## Appendix A: Key Research Papers

1. **Heston (1993):** "A closed-form solution for options with stochastic volatility with applications to bond and currency options." *Review of Financial Studies*, 6(2), 327-343.

2. **Hagan et al. (2002):** "Managing smile risk." *Wilmott Magazine*, September 2002, 84-108.

3. **Leung & Li (2015):** "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit." *Journal of Industrial and Management Optimization*.

4. **Moreira & Muir (2017):** "Volatility-managed portfolios." *Journal of Finance*, 72(4), 1611-1644.

5. **Frazzini & Pedersen (2014):** "Betting against beta." *Journal of Financial Economics*, 111(1), 1-25.

6. **Bertram (2010):** "Analytic solutions for optimal statistical arbitrage trading." *Physica A*, 389(11), 2234-2243.

---

**Document Version History:**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-20 | Initial release |

**Approvals:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Lead Quant | [Name] | ________ | _____ |
| CTO | [Name] | ________ | _____ |
| CRO | [Name] | ________ | _____ |