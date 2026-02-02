"""
Heston model calibrator using two-stage optimization.

Implements calibration of the Heston (1993) stochastic volatility model
using a combination of global and local optimization:
  - Stage 1: Differential Evolution (global search)
  - Stage 2: Levenberg-Marquardt (local refinement)

Uses C++ backend for fast option pricing via Carr-Madan FFT.

Reference:
    Heston, S.L. (1993). "A closed-form solution for options with
    stochastic volatility." Review of Financial Studies, 6(2), 327-343.

Example:
    >>> calibrator = HestonCalibrator()
    >>> result = calibrator.calibrate(market_options, S0=100, r=0.05, q=0.02)
    >>> print(result['params'])
    {'kappa': 2.1, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7, 'v0': 0.04}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging
import time

import numpy as np
from scipy.optimize import differential_evolution, least_squares

if TYPE_CHECKING:
    import pandas as pd
    from ..database.db import TimeSeriesDB

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Raised when model calibration fails."""

    pass


@dataclass
class HestonParameters:
    """
    Heston model parameters.

    Attributes:
        kappa: Mean-reversion speed of variance (κ > 0)
        theta: Long-term variance mean (θ > 0)
        sigma: Volatility of variance (σ > 0)
        rho: Correlation between asset and variance (-1 < ρ < 1)
        v0: Initial variance (v₀ > 0)
    """

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma": self.sigma,
            "rho": self.rho,
            "v0": self.v0,
            "feller_satisfied": self.feller_condition_satisfied,
        }

    @property
    def feller_condition_satisfied(self) -> bool:
        """Check if Feller condition is satisfied: 2κθ ≥ σ². Alias for is_feller_satisfied."""
        return self.is_feller_satisfied

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [kappa, theta, sigma, rho, v0]."""
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "HestonParameters":
        """Create from numpy array."""
        return cls(
            kappa=float(arr[0]),
            theta=float(arr[1]),
            sigma=float(arr[2]),
            rho=float(arr[3]),
            v0=float(arr[4]),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "HestonParameters":
        """Create from dictionary."""
        return cls(
            kappa=d["kappa"],
            theta=d["theta"],
            sigma=d["sigma"],
            rho=d["rho"],
            v0=d["v0"],
        )

    @property
    def feller_condition_value(self) -> float:
        """Compute Feller condition: 2κθ - σ² (should be ≥ 0)."""
        return 2 * self.kappa * self.theta - self.sigma**2

    @property
    def is_feller_satisfied(self) -> bool:
        """Check if Feller condition is satisfied: 2κθ ≥ σ²."""
        return self.feller_condition_value >= 0


@dataclass
class CalibrationResult:
    """
    Result of Heston model calibration.

    Attributes:
        params: Calibrated Heston parameters
        fit_quality: Fit quality metrics (RMSE, R², etc.)
        convergence: Convergence information
        timestamp: Calibration timestamp
        warnings: List of warning messages
    """

    params: HestonParameters
    fit_quality: Dict[str, float]
    convergence: Dict[str, Any]
    timestamp: datetime
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if calibration was successful."""
        return self.convergence.get("local_converged", False) or self.convergence.get("cached", False)

    @property
    def rmse(self) -> float:
        """Get RMSE from fit quality metrics."""
        return self.fit_quality.get("rmse", float("inf"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "params": self.params.to_dict(),
            "fit_quality": self.fit_quality,
            "convergence": self.convergence,
            "timestamp": self.timestamp,
            "warnings": self.warnings,
            "success": self.success,
            "rmse": self.rmse,
        }


class HestonCalibrator:
    """
    Heston model calibrator using two-stage optimization.

    Stage 1: Global search with Differential Evolution
        - Explores parameter space to avoid local minima
        - Uses parallel workers for speed

    Stage 2: Local refinement with Levenberg-Marquardt
        - Trust Region Reflective algorithm
        - Refines solution from Stage 1

    Uses C++ backend for fast option pricing via Carr-Madan FFT.

    Reference:
        Heston, S.L. (1993). "A closed-form solution for options with
        stochastic volatility." Review of Financial Studies, 6(2), 327-343.

    Attributes:
        db: Optional database connection for storing results
        bounds: Parameter bounds for optimization
        global_maxiter: Max iterations for global optimization
        global_popsize: Population size for differential evolution
        local_method: Local optimization method
        local_ftol: Convergence tolerance for local optimization
    """

    # Default parameter bounds
    DEFAULT_BOUNDS = {
        "kappa": (0.1, 10.0),  # Mean-reversion speed
        "theta": (0.01, 1.0),  # Long-term variance
        "sigma": (0.01, 2.0),  # Vol of vol
        "rho": (-0.99, 0.99),  # Correlation
        "v0": (0.01, 1.0),  # Initial variance
    }

    def __init__(
        self,
        db: Optional["TimeSeriesDB"] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        global_maxiter: int = 100,
        global_popsize: int = 15,
        local_method: str = "trf",
        local_ftol: float = 1e-8,
    ):
        """
        Initialize Heston calibrator.

        Args:
            db: Database connection for storing calibrated parameters
            bounds: Custom parameter bounds (default: see DEFAULT_BOUNDS)
            global_maxiter: Max iterations for global optimization
            global_popsize: Population size for differential evolution
            local_method: Local optimization method ('trf', 'dogbox', 'lm')
            local_ftol: Function tolerance for local optimization
        """
        self.db = db
        self.bounds = bounds or self.DEFAULT_BOUNDS.copy()
        self.global_maxiter = global_maxiter
        self.global_popsize = global_popsize
        self.local_method = local_method
        self.local_ftol = local_ftol

        # Lazy import of C++ model
        self._heston_model_class = None

    def _get_heston_model_class(self):
        """Lazy import of HestonModel to avoid circular imports."""
        if self._heston_model_class is None:
            from ..models.heston import HestonModel

            self._heston_model_class = HestonModel
        return self._heston_model_class

    def calibrate(
        self,
        market_options: "pd.DataFrame",
        S0: float,
        r: float,
        q: float,
        warm_start: Optional[Dict[str, float]] = None,
        use_cached_on_failure: bool = True,
        underlying: Optional[str] = None,
    ) -> CalibrationResult:
        """
        Calibrate Heston parameters to market option prices.

        Args:
            market_options: DataFrame with columns:
                ['strike', 'maturity', 'mid_price', 'option_type']
                Optional: ['underlying', 'is_call']
            S0: Current spot price
            r: Risk-free rate (annualized)
            q: Dividend yield (annualized)
            warm_start: Optional initial guess from previous calibration
            use_cached_on_failure: If True, return cached params on failure
            underlying: Underlying symbol (for database storage)

        Returns:
            CalibrationResult with calibrated parameters and fit quality

        Raises:
            CalibrationError: If calibration fails and no cached params available
            ValueError: If input data is invalid
        """
        logger.info(f"Starting Heston calibration with {len(market_options)} options")
        start_time = time.time()

        # Validate inputs
        self._validate_market_data(market_options)

        # Get underlying symbol
        if underlying is None:
            if "underlying" in market_options.columns:
                underlying = market_options["underlying"].iloc[0]
            else:
                underlying = "UNKNOWN"

        try:
            # Extract market data arrays
            strikes = market_options["strike"].values.astype(np.float64)
            maturities = market_options["maturity"].values.astype(np.float64)
            market_prices = market_options["mid_price"].values.astype(np.float64)

            # Determine if options are calls
            if "is_call" in market_options.columns:
                is_calls = market_options["is_call"].values
            elif "option_type" in market_options.columns:
                is_calls = market_options["option_type"].str.lower() == "call"
            else:
                is_calls = np.ones(len(market_options), dtype=bool)

            # Stage 1: Global optimization
            logger.info("Stage 1: Global search with Differential Evolution")
            global_result = self._global_optimization(
                strikes, maturities, market_prices, is_calls, S0, r, q, warm_start
            )

            # Stage 2: Local refinement
            logger.info("Stage 2: Local refinement with Levenberg-Marquardt")
            local_result = self._local_optimization(
                strikes, maturities, market_prices, is_calls, S0, r, q, global_result.x
            )

            # Convert to parameters
            params = HestonParameters.from_array(local_result.x)

            # Validate results
            warnings = self._validate_parameters(params)
            for warning in warnings:
                logger.warning(warning)

            # Compute fit quality
            fit_quality = self._compute_fit_quality(
                params, strikes, maturities, market_prices, is_calls, S0, r, q
            )

            # Calculate calibration time
            calibration_time_ms = int((time.time() - start_time) * 1000)

            # Build result
            result = CalibrationResult(
                params=params,
                fit_quality=fit_quality,
                convergence={
                    "global_converged": global_result.success,
                    "local_converged": local_result.success,
                    "global_nit": global_result.nit,
                    "local_nfev": local_result.nfev,
                    "calibration_time_ms": calibration_time_ms,
                },
                timestamp=datetime.now(),
                warnings=warnings,
            )

            # Store in database
            if self.db:
                self._store_results(result, underlying)

            logger.info(
                f"Calibration successful: RMSE={fit_quality['rmse']:.4f}, "
                f"R²={fit_quality['r_squared']:.4f}, "
                f"Time={calibration_time_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Calibration failed: {e}")

            if use_cached_on_failure and self.db:
                logger.warning("Attempting to use cached parameters from database")
                cached = self._load_cached_parameters(underlying)
                if cached:
                    logger.info("Using cached parameters from database")
                    return cached

            raise CalibrationError(f"Calibration failed: {e}") from e

    def _global_optimization(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
        warm_start: Optional[Dict[str, float]],
    ):
        """
        Stage 1: Global optimization with Differential Evolution.

        Args:
            strikes, maturities, market_prices: Market data arrays
            is_calls: Boolean array indicating call options
            S0, r, q: Market parameters
            warm_start: Optional initial guess

        Returns:
            scipy.optimize.OptimizeResult
        """

        def objective(x):
            return self._compute_objective(
                x, strikes, maturities, market_prices, is_calls, S0, r, q
            )

        # Get bounds as list of tuples
        bounds_list = [
            self.bounds["kappa"],
            self.bounds["theta"],
            self.bounds["sigma"],
            self.bounds["rho"],
            self.bounds["v0"],
        ]

        # Initial point for population
        x0 = None
        if warm_start:
            x0 = HestonParameters.from_dict(warm_start).to_array()

        # Run Differential Evolution
        result = differential_evolution(
            objective,
            bounds=bounds_list,
            maxiter=self.global_maxiter,
            popsize=self.global_popsize,
            seed=42,
            x0=x0,
            workers=1,  # Single worker to avoid serialization issues with C++
            updating="immediate",
            polish=False,  # We do local optimization next
        )

        logger.debug(
            f"Global optimization: converged={result.success}, "
            f"obj={result.fun:.6f}, nit={result.nit}"
        )

        return result

    def _local_optimization(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
        x0: np.ndarray,
    ):
        """
        Stage 2: Local refinement with Levenberg-Marquardt.

        Args:
            strikes, maturities, market_prices: Market data arrays
            is_calls: Boolean array indicating call options
            S0, r, q: Market parameters
            x0: Initial point from global optimization

        Returns:
            scipy.optimize.OptimizeResult
        """

        def residuals(x):
            return self._compute_residuals(
                x, strikes, maturities, market_prices, is_calls, S0, r, q
            )

        # Get bounds as arrays
        lower = np.array([self.bounds[k][0] for k in ["kappa", "theta", "sigma", "rho", "v0"]])
        upper = np.array([self.bounds[k][1] for k in ["kappa", "theta", "sigma", "rho", "v0"]])

        # Run least_squares
        result = least_squares(
            residuals,
            x0=x0,
            bounds=(lower, upper),
            method=self.local_method,
            ftol=self.local_ftol,
            xtol=1e-8,
            verbose=0,
        )

        logger.debug(
            f"Local optimization: converged={result.success}, "
            f"cost={result.cost:.6f}, nfev={result.nfev}"
        )

        return result

    def _compute_objective(
        self,
        params_array: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> float:
        """
        Objective function: sum of squared relative errors.

        Uses relative errors to avoid overweighting expensive options.
        """
        model_prices = self._price_options(
            params_array, strikes, maturities, is_calls, S0, r, q
        )

        # Handle potential numerical issues
        if np.any(np.isnan(model_prices)) or np.any(model_prices <= 0):
            return 1e10

        # Relative errors
        errors = (model_prices - market_prices) / market_prices

        return np.sum(errors**2)

    def _compute_residuals(
        self,
        params_array: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """
        Residuals for least_squares: relative pricing errors.
        """
        model_prices = self._price_options(
            params_array, strikes, maturities, is_calls, S0, r, q
        )

        # Handle potential numerical issues
        model_prices = np.maximum(model_prices, 1e-10)

        return (model_prices - market_prices) / market_prices

    def _price_options(
        self,
        params_array: np.ndarray,
        strikes: np.ndarray,
        maturities: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> np.ndarray:
        """
        Price options using C++ backend.

        Args:
            params_array: [kappa, theta, sigma, rho, v0]
            strikes, maturities: Option specifications
            is_calls: Boolean array indicating call options
            S0, r, q: Market parameters

        Returns:
            NumPy array of model prices
        """
        HestonModel = self._get_heston_model_class()

        # Create Heston model with current parameters
        model = HestonModel(
            kappa=params_array[0],
            theta=params_array[1],
            sigma=params_array[2],
            rho=params_array[3],
            v0=params_array[4],
        )

        # Price each option
        prices = np.zeros(len(strikes))
        for i in range(len(strikes)):
            try:
                prices[i] = model.price_option(
                    strike=strikes[i],
                    maturity=maturities[i],
                    spot=S0,
                    rate=r,
                    dividend=q,
                    is_call=bool(is_calls[i]) if hasattr(is_calls, "__getitem__") else bool(is_calls),
                )
            except Exception:
                prices[i] = np.nan

        return prices

    def _compute_fit_quality(
        self,
        params: HestonParameters,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        is_calls: np.ndarray,
        S0: float,
        r: float,
        q: float,
    ) -> Dict[str, float]:
        """
        Compute fit quality metrics.

        Returns:
            dict: {
                'rmse': root mean squared error,
                'r_squared': R² coefficient,
                'relative_rmse': RMSE / mean(market_price),
                'max_abs_error': maximum absolute error,
                'mean_abs_error': mean absolute error,
                'n_options': number of options,
                'feller_satisfied': whether Feller condition is satisfied
            }
        """
        model_prices = self._price_options(
            params.to_array(), strikes, maturities, is_calls, S0, r, q
        )

        # Errors
        errors = model_prices - market_prices
        relative_errors = errors / market_prices

        # RMSE
        rmse = np.sqrt(np.mean(errors**2))
        relative_rmse = rmse / np.mean(market_prices)

        # R²
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((market_prices - np.mean(market_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Other metrics
        max_abs_error = np.max(np.abs(errors))
        mean_abs_error = np.mean(np.abs(errors))

        return {
            "rmse": float(rmse),
            "r_squared": float(r_squared),
            "relative_rmse": float(relative_rmse),
            "max_abs_error": float(max_abs_error),
            "mean_abs_error": float(mean_abs_error),
            "n_options": len(market_prices),
            "feller_satisfied": params.is_feller_satisfied,
            "feller_value": params.feller_condition_value,
        }

    def _validate_parameters(self, params: HestonParameters) -> List[str]:
        """
        Validate calibrated parameters.

        Returns:
            List of warning messages (empty if all good)
        """
        warnings = []

        # Check Feller condition
        if not params.is_feller_satisfied:
            warnings.append(
                f"Feller condition violated: 2κθ = {2*params.kappa*params.theta:.4f} < "
                f"σ² = {params.sigma**2:.4f}. Variance may reach zero."
            )

        # Check parameter reasonableness
        if params.kappa > 8.0:
            warnings.append(f"Very high mean-reversion speed: κ={params.kappa:.2f}")

        if params.sigma > 1.5:
            warnings.append(f"Very high vol of vol: σ={params.sigma:.2f}")

        if abs(params.rho) > 0.95:
            warnings.append(f"Extreme correlation: ρ={params.rho:.2f}")

        if params.v0 > 0.5:
            warnings.append(f"Very high initial variance: v₀={params.v0:.2f}")

        return warnings

    def _validate_market_data(self, market_options: "pd.DataFrame"):
        """Validate market options data."""
        required_cols = ["strike", "maturity", "mid_price"]

        for col in required_cols:
            if col not in market_options.columns:
                raise ValueError(f"Missing required column: {col}")

        if len(market_options) < 5:
            logger.warning(
                f"Very few options for calibration: {len(market_options)}. "
                f"Recommend at least 20 options for reliable calibration."
            )

        # Check for invalid prices
        invalid = market_options[market_options["mid_price"] <= 0]
        if len(invalid) > 0:
            raise ValueError(f"Found {len(invalid)} options with price <= 0")

        # Check for invalid maturities
        invalid_mat = market_options[market_options["maturity"] <= 0]
        if len(invalid_mat) > 0:
            raise ValueError(f"Found {len(invalid_mat)} options with maturity <= 0")

    def _store_results(self, result: CalibrationResult, underlying: str):
        """Store calibration results in database."""
        self.db.store_model_parameters(
            model_type="heston",
            underlying=underlying,
            parameters=result.params.to_dict(),
            fit_quality=result.fit_quality,
            maturity=None,  # Heston calibrated to entire surface
            converged=result.convergence["local_converged"],
            calibration_time_ms=result.convergence["calibration_time_ms"],
        )

    def _load_cached_parameters(self, underlying: str) -> Optional[CalibrationResult]:
        """Load most recent calibrated parameters from database."""
        cached = self.db.get_latest_model_parameters(
            model_type="heston",
            underlying=underlying,
            maturity=None,
        )

        if cached and cached.get("converged", False):
            logger.info(
                f"Loaded cached Heston params from {cached['time']}: "
                f"RMSE={cached['fit_quality'].get('rmse', 'N/A')}"
            )
            return CalibrationResult(
                params=HestonParameters.from_dict(cached["parameters"]),
                fit_quality=cached["fit_quality"],
                convergence={"cached": True},
                timestamp=cached["time"],
                warnings=["Using cached parameters"],
            )

        return None

    @classmethod
    def generate_synthetic_data(
        cls,
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.02,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.7,
        v0: float = 0.04,
        n_strikes: int = 11,
        n_maturities: int = 3,
        noise_std: float = 0.0,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
    ) -> "pd.DataFrame":
        """
        Generate synthetic option prices for testing.

        Args:
            S0: Spot price
            r: Risk-free rate
            q: Dividend yield
            kappa: Heston mean-reversion speed
            theta: Heston long-term variance
            sigma: Heston vol of vol
            rho: Heston correlation
            v0: Heston initial variance
            n_strikes: Number of strikes to generate
            n_maturities: Number of maturities to generate
            noise_std: Standard deviation of relative noise (0.0 = no noise)
            strikes: Optional explicit strike prices
            maturities: Optional explicit maturities

        Returns:
            DataFrame with synthetic option data
        """
        import pandas as pd
        from ..models.heston import HestonModel

        if strikes is None:
            strikes = np.linspace(0.8 * S0, 1.2 * S0, n_strikes)
        if maturities is None:
            maturities = np.linspace(0.1, 1.0, n_maturities)

        # Create model with specified parameters
        model = HestonModel(
            kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0
        )

        data = []
        for T in maturities:
            for K in strikes:
                try:
                    price = model.price_option(
                        strike=K, maturity=T, spot=S0, rate=r, dividend=q, is_call=True
                    )
                except Exception:
                    # Fallback to Black-Scholes approximation
                    price = max(S0 - K * np.exp(-r * T), 0.01) + 2.0

                # Add noise if requested
                if noise_std > 0:
                    price *= 1 + np.random.normal(0, noise_std)
                    price = max(price, 0.01)

                data.append(
                    {
                        "strike": K,
                        "maturity": T,
                        "mid_price": price,
                        "option_type": "call",
                        "underlying": "SYNTHETIC",
                        "is_call": True,
                    }
                )

        return pd.DataFrame(data)

    # Alias for backward compatibility
    generate_synthetic_options = generate_synthetic_data
