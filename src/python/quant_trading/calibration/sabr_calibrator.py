"""
SABR Volatility Model Calibrator.

Implements per-maturity smile fitting using the SABR model from
Hagan et al. (2002) "Managing smile risk".

The SABR model:
    dF_t = σ_t F_t^β dW_t^F
    dσ_t = ν σ_t dW_t^σ
    dW_t^F · dW_t^σ = ρ dt

Parameters:
    α (alpha): Initial volatility level
    β (beta): CEV exponent (fixed at 0.5 for equity)
    ρ (rho): Correlation between forward and volatility
    ν (nu): Volatility of volatility

Performance target: <1 second per smile

Reference:
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    "Managing smile risk." Wilmott Magazine, 1, 84-108.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from scipy import optimize

from .heston_calibrator import CalibrationError

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SABRParameters:
    """SABR model parameters for a single maturity."""

    alpha: float  # Initial volatility
    beta: float  # CEV exponent (typically 0.5 for equity)
    rho: float  # Correlation
    nu: float  # Vol of vol

    def __post_init__(self):
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1 < self.rho < 1:
            raise ValueError(f"rho must be in (-1, 1), got {self.rho}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
        }


@dataclass
class SABRCalibrationResult:
    """Result from SABR calibration across maturities."""

    params_by_maturity: Dict[float, SABRParameters]  # T -> params
    rmse_by_maturity: Dict[float, float]  # T -> RMSE
    total_rmse: float
    calibration_time: float
    n_maturities: int
    n_options: int
    success: bool
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "params_by_maturity": {
                str(T): params.to_dict()
                for T, params in self.params_by_maturity.items()
            },
            "rmse_by_maturity": {
                str(T): rmse for T, rmse in self.rmse_by_maturity.items()
            },
            "total_rmse": self.total_rmse,
            "calibration_time": self.calibration_time,
            "n_maturities": self.n_maturities,
            "n_options": self.n_options,
            "success": self.success,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class SABRCalibrator:
    """
    SABR model calibrator for volatility smile fitting.

    Calibrates SABR parameters (α, ρ, ν) per maturity using the
    Hagan et al. (2002) asymptotic implied volatility formula.

    Features:
        - Per-maturity smile calibration
        - Fixed β = 0.5 for equity
        - Fast optimization using SLSQP
        - Support for warm-starting from previous calibration

    Example:
        >>> calibrator = SABRCalibrator()
        >>> result = calibrator.calibrate(market_options, F0=100, T=0.25)
        >>> print(f"Calibrated alpha: {result.params_by_maturity[0.25].alpha}")

    Reference:
        Hagan et al. (2002) "Managing smile risk"
    """

    # Default parameter bounds
    DEFAULT_BOUNDS = {
        "alpha": (0.001, 2.0),
        "rho": (-0.99, 0.99),
        "nu": (0.001, 3.0),
    }

    def __init__(
        self,
        beta: float = 0.5,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        db_session=None,
    ):
        """
        Initialize SABR calibrator.

        Args:
            beta: CEV exponent (default 0.5 for equity)
            bounds: Parameter bounds {param_name: (lower, upper)}
            db_session: Optional database session for storing results
        """
        self.beta = beta
        self.bounds = {**self.DEFAULT_BOUNDS, **(bounds or {})}
        self.db_session = db_session
        self._cached_params: Dict[str, Dict[float, SABRParameters]] = {}

        logger.info(
            f"Initialized SABRCalibrator with beta={beta}, bounds={self.bounds}"
        )

    def sabr_implied_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Compute SABR implied volatility using Hagan's asymptotic formula.

        This implements the approximation from Section 2.5 of Hagan et al. (2002).

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            alpha: SABR alpha parameter
            beta: SABR beta parameter
            rho: SABR rho parameter
            nu: SABR nu parameter

        Returns:
            Implied Black volatility
        """
        # Handle ATM case separately to avoid numerical issues
        if abs(F - K) < 1e-10:
            return self._sabr_atm_vol(F, T, alpha, beta, rho, nu)

        # General case
        FK = F * K
        log_FK = np.log(F / K)
        FK_beta = FK ** ((1 - beta) / 2)

        # z coefficient
        z = (nu / alpha) * FK_beta * log_FK

        # x(z) function
        sqrt_term = np.sqrt(1 - 2 * rho * z + z * z)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho))

        # Avoid division by zero
        if abs(x_z) < 1e-10:
            zeta = 1.0
        else:
            zeta = z / x_z

        # First bracket term
        one_minus_beta = 1 - beta
        one_minus_beta_sq = one_minus_beta * one_minus_beta

        term1 = one_minus_beta_sq / 24 * alpha * alpha / (FK_beta * FK_beta)
        term2 = rho * beta * nu * alpha / (4 * FK_beta)
        term3 = (2 - 3 * rho * rho) * nu * nu / 24

        bracket = 1 + (term1 + term2 + term3) * T

        # Denominator correction for non-ATM
        denom_term = 1 + one_minus_beta_sq / 24 * log_FK * log_FK
        denom_term += one_minus_beta_sq * one_minus_beta_sq / 1920 * log_FK ** 4

        sigma = (alpha / (FK_beta * denom_term)) * zeta * bracket

        return max(sigma, 1e-6)  # Ensure positive

    def _sabr_atm_vol(
        self,
        F: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """
        Compute ATM SABR implied volatility.

        Simplified formula when F = K.

        Args:
            F: Forward/Strike (ATM)
            T: Time to maturity
            alpha: SABR alpha
            beta: SABR beta
            rho: SABR rho
            nu: SABR nu

        Returns:
            ATM implied volatility
        """
        F_beta = F ** (1 - beta)
        one_minus_beta = 1 - beta

        term1 = one_minus_beta * one_minus_beta / 24 * alpha * alpha / (F_beta * F_beta)
        term2 = rho * beta * nu * alpha / (4 * F_beta)
        term3 = (2 - 3 * rho * rho) * nu * nu / 24

        return alpha / F_beta * (1 + (term1 + term2 + term3) * T)

    def calibrate_single_maturity(
        self,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        F: float,
        T: float,
        weights: Optional[np.ndarray] = None,
        initial_guess: Optional[Dict[str, float]] = None,
    ) -> Tuple[SABRParameters, float]:
        """
        Calibrate SABR parameters for a single maturity.

        Args:
            strikes: Array of strike prices
            market_vols: Array of market implied volatilities
            F: Forward price
            T: Time to maturity
            weights: Optional weights for each strike (e.g., vega weights)
            initial_guess: Optional initial parameter guess

        Returns:
            Tuple of (calibrated parameters, RMSE)

        Raises:
            CalibrationError: If optimization fails
        """
        if len(strikes) < 3:
            raise CalibrationError(
                f"Need at least 3 strikes for SABR calibration, got {len(strikes)}"
            )

        if weights is None:
            weights = np.ones(len(strikes))
        weights = weights / np.sum(weights)  # Normalize

        # Initial guess
        if initial_guess:
            x0 = [
                initial_guess.get("alpha", 0.3),
                initial_guess.get("rho", -0.3),
                initial_guess.get("nu", 0.5),
            ]
        else:
            # Estimate initial alpha from ATM vol
            atm_idx = np.argmin(np.abs(strikes - F))
            atm_vol = market_vols[atm_idx]
            alpha_init = atm_vol * F ** (1 - self.beta)
            x0 = [alpha_init, -0.3, 0.5]

        # Bounds
        bounds = [
            self.bounds["alpha"],
            self.bounds["rho"],
            self.bounds["nu"],
        ]

        def objective(params):
            """Weighted sum of squared errors."""
            alpha, rho, nu = params
            model_vols = np.array([
                self.sabr_implied_vol(F, K, T, alpha, self.beta, rho, nu)
                for K in strikes
            ])
            errors = (model_vols - market_vols) ** 2
            return np.sum(weights * errors)

        # Optimize using SLSQP
        result = optimize.minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={"ftol": 1e-10, "maxiter": 200},
        )

        if not result.success:
            logger.warning(
                f"SABR optimization did not converge for T={T}: {result.message}"
            )

        alpha, rho, nu = result.x

        # Compute RMSE
        model_vols = np.array([
            self.sabr_implied_vol(F, K, T, alpha, self.beta, rho, nu)
            for K in strikes
        ])
        rmse = np.sqrt(np.mean((model_vols - market_vols) ** 2))

        params = SABRParameters(
            alpha=float(alpha),
            beta=self.beta,
            rho=float(rho),
            nu=float(nu),
        )

        logger.debug(
            f"Calibrated T={T:.4f}: alpha={alpha:.4f}, rho={rho:.4f}, "
            f"nu={nu:.4f}, RMSE={rmse:.6f}"
        )

        return params, rmse

    def calibrate(
        self,
        market_options: "pd.DataFrame",
        F0: float,
        r: float = 0.0,
        q: float = 0.0,
        use_forward: bool = True,
        warm_start: Optional[Dict[float, Dict[str, float]]] = None,
        underlying: Optional[str] = None,
    ) -> SABRCalibrationResult:
        """
        Calibrate SABR parameters across all maturities.

        Args:
            market_options: DataFrame with columns:
                - strike: Strike price
                - T: Time to maturity
                - implied_vol: Market implied volatility
                - weight (optional): Calibration weight
            F0: Spot price (will be converted to forward if use_forward=True)
            r: Risk-free rate (for forward calculation)
            q: Dividend yield (for forward calculation)
            use_forward: If True, compute forward price for each maturity
            warm_start: Optional initial guesses {T: {param: value}}
            underlying: Optional underlying symbol for caching

        Returns:
            SABRCalibrationResult with parameters for each maturity

        Example:
            >>> import pandas as pd
            >>> options = pd.DataFrame({
            ...     'strike': [90, 95, 100, 105, 110] * 2,
            ...     'T': [0.25] * 5 + [0.5] * 5,
            ...     'implied_vol': [0.25, 0.22, 0.20, 0.22, 0.25] * 2
            ... })
            >>> result = calibrator.calibrate(options, F0=100)
        """
        import time

        start_time = time.time()

        # Get unique maturities
        maturities = sorted(market_options["T"].unique())
        n_maturities = len(maturities)
        n_options = len(market_options)

        logger.info(
            f"Starting SABR calibration for {n_maturities} maturities, "
            f"{n_options} options"
        )

        params_by_maturity: Dict[float, SABRParameters] = {}
        rmse_by_maturity: Dict[float, float] = {}
        total_errors = []

        for T in maturities:
            # Filter options for this maturity
            mask = market_options["T"] == T
            maturity_data = market_options[mask]

            strikes = maturity_data["strike"].values
            market_vols = maturity_data["implied_vol"].values

            # Weights (use provided or uniform)
            if "weight" in maturity_data.columns:
                weights = maturity_data["weight"].values
            else:
                weights = None

            # Compute forward price
            if use_forward:
                F = F0 * np.exp((r - q) * T)
            else:
                F = F0

            # Initial guess from warm start
            initial_guess = None
            if warm_start and T in warm_start:
                initial_guess = warm_start[T]

            # Calibrate
            try:
                params, rmse = self.calibrate_single_maturity(
                    strikes=strikes,
                    market_vols=market_vols,
                    F=F,
                    T=T,
                    weights=weights,
                    initial_guess=initial_guess,
                )
                params_by_maturity[T] = params
                rmse_by_maturity[T] = rmse
                total_errors.extend(
                    (np.array([
                        self.sabr_implied_vol(
                            F, K, T, params.alpha, params.beta, params.rho, params.nu
                        )
                        for K in strikes
                    ]) - market_vols) ** 2
                )

            except CalibrationError as e:
                logger.error(f"Calibration failed for T={T}: {e}")
                rmse_by_maturity[T] = float("inf")

        calibration_time = time.time() - start_time
        total_rmse = np.sqrt(np.mean(total_errors)) if total_errors else float("inf")
        success = len(params_by_maturity) == n_maturities

        result = SABRCalibrationResult(
            params_by_maturity=params_by_maturity,
            rmse_by_maturity=rmse_by_maturity,
            total_rmse=total_rmse,
            calibration_time=calibration_time,
            n_maturities=n_maturities,
            n_options=n_options,
            success=success,
            message="Calibration successful" if success else "Partial calibration",
        )

        logger.info(
            f"SABR calibration completed in {calibration_time:.3f}s, "
            f"total RMSE={total_rmse:.6f}"
        )

        # Cache results
        if underlying:
            self._cached_params[underlying] = params_by_maturity

        # Store in database if session provided
        if self.db_session and underlying:
            self._store_calibration_result(underlying, result)

        return result

    def get_implied_vol(
        self,
        F: float,
        K: float,
        T: float,
        params: Optional[SABRParameters] = None,
        underlying: Optional[str] = None,
    ) -> float:
        """
        Get SABR implied volatility for given strike and maturity.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            params: SABR parameters (if None, uses cached)
            underlying: Underlying symbol (for cached params)

        Returns:
            SABR implied volatility
        """
        if params is None:
            if underlying and underlying in self._cached_params:
                # Find nearest maturity
                cached = self._cached_params[underlying]
                nearest_T = min(cached.keys(), key=lambda x: abs(x - T))
                params = cached[nearest_T]
            else:
                raise ValueError("No parameters provided and no cached params available")

        return self.sabr_implied_vol(
            F, K, T, params.alpha, params.beta, params.rho, params.nu
        )

    def interpolate_params(
        self,
        T: float,
        params_by_maturity: Dict[float, SABRParameters],
    ) -> SABRParameters:
        """
        Interpolate SABR parameters to a non-calibrated maturity.

        Uses linear interpolation in variance space for alpha.

        Args:
            T: Target maturity
            params_by_maturity: Calibrated parameters by maturity

        Returns:
            Interpolated SABR parameters
        """
        maturities = sorted(params_by_maturity.keys())

        if T <= maturities[0]:
            return params_by_maturity[maturities[0]]
        if T >= maturities[-1]:
            return params_by_maturity[maturities[-1]]

        # Find bracketing maturities
        for i in range(len(maturities) - 1):
            if maturities[i] <= T <= maturities[i + 1]:
                T1, T2 = maturities[i], maturities[i + 1]
                p1, p2 = params_by_maturity[T1], params_by_maturity[T2]
                break

        # Linear interpolation weight
        w = (T - T1) / (T2 - T1)

        # Interpolate in variance space for alpha
        var1 = p1.alpha ** 2 * T1
        var2 = p2.alpha ** 2 * T2
        var_T = var1 + w * (var2 - var1)
        alpha = np.sqrt(var_T / T)

        # Linear interpolation for rho and nu
        rho = p1.rho + w * (p2.rho - p1.rho)
        nu = p1.nu + w * (p2.nu - p1.nu)

        return SABRParameters(
            alpha=alpha,
            beta=self.beta,
            rho=rho,
            nu=nu,
        )

    def _store_calibration_result(
        self, underlying: str, result: SABRCalibrationResult
    ) -> None:
        """Store calibration result in database."""
        try:
            from ..database.models import ModelParameter

            for T, params in result.params_by_maturity.items():
                param_record = ModelParameter(
                    model_type="SABR",
                    underlying=underlying,
                    parameters=params.to_dict(),
                    fit_quality=result.rmse_by_maturity.get(T, 0.0),
                    calibration_date=result.timestamp.date(),
                )
                self.db_session.add(param_record)

            self.db_session.commit()
            logger.info(f"Stored SABR calibration for {underlying}")

        except ImportError:
            logger.warning("Database models not available, skipping storage")
        except Exception as e:
            logger.error(f"Failed to store calibration result: {e}")
            self.db_session.rollback()

    @staticmethod
    def generate_synthetic_smile(
        F: float = 100.0,
        T: float = 0.25,
        alpha: float = 0.3,
        beta: float = 0.5,
        rho: float = -0.3,
        nu: float = 0.5,
        n_strikes: int = 11,
        strike_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.0,
    ) -> "pd.DataFrame":
        """
        Generate synthetic SABR smile data for testing.

        Args:
            F: Forward price
            T: Time to maturity
            alpha: True SABR alpha
            beta: True SABR beta
            rho: True SABR rho
            nu: True SABR nu
            n_strikes: Number of strikes to generate
            strike_range: Strike range as fraction of forward (min, max)
            noise_std: Standard deviation of noise to add

        Returns:
            DataFrame with strike, T, implied_vol columns
        """
        import pandas as pd

        strikes = np.linspace(
            F * strike_range[0], F * strike_range[1], n_strikes
        )

        calibrator = SABRCalibrator(beta=beta)
        vols = np.array([
            calibrator.sabr_implied_vol(F, K, T, alpha, beta, rho, nu)
            for K in strikes
        ])

        if noise_std > 0:
            vols += np.random.normal(0, noise_std, len(vols))
            vols = np.maximum(vols, 0.01)  # Ensure positive

        return pd.DataFrame({
            "strike": strikes,
            "T": T,
            "implied_vol": vols,
        })
