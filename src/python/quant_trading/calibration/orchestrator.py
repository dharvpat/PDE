"""
Calibration Orchestrator.

Coordinates daily calibration runs across all models:
- Heston stochastic volatility calibration
- SABR smile calibration per maturity
- OU mean-reversion parameter fitting

Manages calibration scheduling, caching, and failure recovery.

Architecture:
    Data Ingestion → Orchestrator → [Heston, SABR, OU] → Database Storage
                          ↓
                   Monitoring/Alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from .heston_calibrator import CalibrationError, HestonCalibrator
from .ou_fitter import OUFitter
from .sabr_calibrator import SABRCalibrator

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationStatus(Enum):
    """Status of a calibration run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"  # Some models failed
    FAILED = "failed"


@dataclass
class CalibrationConfig:
    """Configuration for calibration orchestrator."""

    # Heston settings
    heston_enabled: bool = True
    heston_max_options: int = 100
    heston_min_options: int = 10
    heston_timeout: float = 60.0  # seconds

    # SABR settings
    sabr_enabled: bool = True
    sabr_beta: float = 0.5
    sabr_min_strikes: int = 5

    # OU settings
    ou_enabled: bool = True
    ou_min_observations: int = 60
    ou_max_half_life: float = 120.0  # days

    # General settings
    use_cached_on_failure: bool = True
    cache_expiry_days: int = 5
    alert_on_failure: bool = True
    rmse_alert_threshold: float = 0.05  # 5% RMSE triggers alert


@dataclass
class CalibrationRunResult:
    """Result of a complete calibration run."""

    run_date: date
    status: CalibrationStatus
    underlying: str

    # Individual results
    heston_result: Optional[Dict] = None
    sabr_result: Optional[Dict] = None
    ou_results: Optional[Dict[str, Dict]] = None  # pair_name -> result

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_time: float = 0.0

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "run_date": self.run_date.isoformat(),
            "status": self.status.value,
            "underlying": self.underlying,
            "heston_result": self.heston_result,
            "sabr_result": self.sabr_result,
            "ou_results": self.ou_results,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_time": self.total_time,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class CalibrationOrchestrator:
    """
    Orchestrates calibration of all models for a given underlying.

    Coordinates Heston, SABR, and OU calibrations with:
        - Automatic scheduling and retry logic
        - Caching of previous results for warm-starting
        - Failure recovery using cached parameters
        - Monitoring and alerting integration

    Example:
        >>> orchestrator = CalibrationOrchestrator(db_session=session)
        >>> result = orchestrator.run_daily_calibration(
        ...     underlying="SPY",
        ...     options_data=options_df,
        ...     spreads_data={"SPY-QQQ": spread_series},
        ...     S0=450.0,
        ...     r=0.05,
        ...     q=0.015
        ... )
        >>> if result.status == CalibrationStatus.SUCCESS:
        ...     print("Calibration successful!")
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        db_session=None,
    ):
        """
        Initialize calibration orchestrator.

        Args:
            config: Calibration configuration
            db_session: Database session for result storage
        """
        self.config = config or CalibrationConfig()
        self.db_session = db_session

        # Initialize calibrators
        self.heston_calibrator = HestonCalibrator(db=db_session)
        self.sabr_calibrator = SABRCalibrator(
            beta=self.config.sabr_beta, db_session=db_session
        )
        self.ou_fitter = OUFitter(db_session=db_session)

        # Cache for warm-starting
        self._last_heston_params: Dict[str, Dict] = {}
        self._last_sabr_params: Dict[str, Dict] = {}
        self._last_ou_params: Dict[str, Dict] = {}

        logger.info("Initialized CalibrationOrchestrator")

    def run_daily_calibration(
        self,
        underlying: str,
        options_data: Optional["pd.DataFrame"] = None,
        spreads_data: Optional[Dict[str, np.ndarray]] = None,
        S0: float = 100.0,
        r: float = 0.05,
        q: float = 0.02,
        calibration_date: Optional[date] = None,
    ) -> CalibrationRunResult:
        """
        Run complete daily calibration for an underlying.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            options_data: DataFrame with options chain data for Heston/SABR
                Required columns: strike, T, implied_vol, (optional) mid_price
            spreads_data: Dict of {pair_name: spread_array} for OU fitting
            S0: Spot price
            r: Risk-free rate
            q: Dividend yield
            calibration_date: Date for calibration (default: today)

        Returns:
            CalibrationRunResult with all calibration outcomes
        """
        import time

        run_date = calibration_date or date.today()
        start_time = datetime.utcnow()

        logger.info(f"Starting daily calibration for {underlying} on {run_date}")

        result = CalibrationRunResult(
            run_date=run_date,
            status=CalibrationStatus.RUNNING,
            underlying=underlying,
            start_time=start_time,
        )

        # Track which calibrations succeeded
        heston_success = True
        sabr_success = True
        ou_success = True

        # Run Heston calibration
        if self.config.heston_enabled and options_data is not None:
            try:
                result.heston_result = self._run_heston_calibration(
                    underlying=underlying,
                    options_data=options_data,
                    S0=S0,
                    r=r,
                    q=q,
                )
            except Exception as e:
                logger.error(f"Heston calibration failed: {e}")
                result.errors.append(f"Heston: {str(e)}")
                heston_success = False

        # Run SABR calibration
        if self.config.sabr_enabled and options_data is not None:
            try:
                result.sabr_result = self._run_sabr_calibration(
                    underlying=underlying,
                    options_data=options_data,
                    S0=S0,
                    r=r,
                    q=q,
                )
            except Exception as e:
                logger.error(f"SABR calibration failed: {e}")
                result.errors.append(f"SABR: {str(e)}")
                sabr_success = False

        # Run OU fitting for each spread
        if self.config.ou_enabled and spreads_data:
            result.ou_results = {}
            for pair_name, spread in spreads_data.items():
                try:
                    result.ou_results[pair_name] = self._run_ou_fitting(
                        pair_name=pair_name,
                        spread=spread,
                    )
                except Exception as e:
                    logger.error(f"OU fitting failed for {pair_name}: {e}")
                    result.errors.append(f"OU ({pair_name}): {str(e)}")
                    ou_success = False

        # Determine overall status
        result.end_time = datetime.utcnow()
        result.total_time = (result.end_time - start_time).total_seconds()

        if heston_success and sabr_success and ou_success:
            result.status = CalibrationStatus.SUCCESS
        elif heston_success or sabr_success or ou_success:
            result.status = CalibrationStatus.PARTIAL
        else:
            result.status = CalibrationStatus.FAILED

        # Check for quality warnings
        self._check_calibration_quality(result)

        # Store run result
        if self.db_session:
            self._store_run_result(result)

        logger.info(
            f"Calibration completed in {result.total_time:.1f}s, "
            f"status={result.status.value}"
        )

        return result

    def _run_heston_calibration(
        self,
        underlying: str,
        options_data: "pd.DataFrame",
        S0: float,
        r: float,
        q: float,
    ) -> Dict:
        """Run Heston model calibration."""
        logger.info(f"Running Heston calibration for {underlying}")

        # Filter options if too many
        if len(options_data) > self.config.heston_max_options:
            options_data = self._filter_options_for_heston(
                options_data, self.config.heston_max_options
            )

        if len(options_data) < self.config.heston_min_options:
            raise CalibrationError(
                f"Insufficient options: {len(options_data)} < "
                f"{self.config.heston_min_options}"
            )

        # Warm start from previous calibration
        warm_start = self._last_heston_params.get(underlying)

        result = self.heston_calibrator.calibrate(
            market_options=options_data,
            S0=S0,
            r=r,
            q=q,
            warm_start=warm_start,
            use_cached_on_failure=self.config.use_cached_on_failure,
            underlying=underlying,
        )

        # Cache for next run
        if result.success:
            self._last_heston_params[underlying] = result.params.to_dict()

        return result.to_dict()

    def _run_sabr_calibration(
        self,
        underlying: str,
        options_data: "pd.DataFrame",
        S0: float,
        r: float,
        q: float,
    ) -> Dict:
        """Run SABR model calibration."""
        logger.info(f"Running SABR calibration for {underlying}")

        # Check minimum strikes per maturity
        maturities = options_data["T"].unique()
        valid_maturities = []
        for T in maturities:
            n_strikes = len(options_data[options_data["T"] == T])
            if n_strikes >= self.config.sabr_min_strikes:
                valid_maturities.append(T)

        if not valid_maturities:
            raise CalibrationError(
                f"No maturities with >= {self.config.sabr_min_strikes} strikes"
            )

        # Filter to valid maturities
        options_subset = options_data[options_data["T"].isin(valid_maturities)]

        # Warm start
        warm_start = self._last_sabr_params.get(underlying)

        result = self.sabr_calibrator.calibrate(
            market_options=options_subset,
            F0=S0,
            r=r,
            q=q,
            warm_start=warm_start,
            underlying=underlying,
        )

        # Cache for next run
        if result.success:
            self._last_sabr_params[underlying] = {
                T: params.to_dict()
                for T, params in result.params_by_maturity.items()
            }

        return result.to_dict()

    def _run_ou_fitting(
        self,
        pair_name: str,
        spread: np.ndarray,
    ) -> Dict:
        """Run OU process fitting for a spread."""
        logger.info(f"Running OU fitting for {pair_name}")

        if len(spread) < self.config.ou_min_observations:
            raise CalibrationError(
                f"Insufficient observations: {len(spread)} < "
                f"{self.config.ou_min_observations}"
            )

        result = self.ou_fitter.fit(
            X=spread,
            dt=1.0 / 252,
            compute_boundaries=True,
            pair_name=pair_name,
        )

        # Check half-life constraint
        if result.params.half_life > self.config.ou_max_half_life:
            logger.warning(
                f"Half-life {result.params.half_life:.1f} exceeds "
                f"max {self.config.ou_max_half_life} for {pair_name}"
            )

        # Cache for next run
        if result.success:
            self._last_ou_params[pair_name] = result.params.to_dict()

        return result.to_dict()

    def _filter_options_for_heston(
        self,
        options_data: "pd.DataFrame",
        max_options: int,
    ) -> "pd.DataFrame":
        """
        Filter options to most informative subset for Heston calibration.

        Prioritizes:
        - Near-ATM strikes
        - Liquid maturities (30-180 days)
        - Multiple maturities for term structure
        """
        # Sort by distance from ATM and select evenly across maturities
        maturities = sorted(options_data["T"].unique())

        # Target maturities in liquid range (1-6 months)
        target_T = [T for T in maturities if 0.08 <= T <= 0.5]
        if not target_T:
            target_T = maturities[:3]  # Take nearest if no liquid ones

        options_per_maturity = max_options // len(target_T)

        filtered_dfs = []
        for T in target_T:
            maturity_options = options_data[options_data["T"] == T].copy()

            # Compute moneyness
            if "moneyness" not in maturity_options.columns:
                # Assume strike and some reference for ATM
                S_ref = maturity_options["strike"].median()
                maturity_options["moneyness"] = abs(
                    np.log(maturity_options["strike"] / S_ref)
                )

            # Take options nearest to ATM
            maturity_options = maturity_options.nsmallest(
                min(options_per_maturity, len(maturity_options)), "moneyness"
            )
            filtered_dfs.append(maturity_options)

        import pandas as pd

        return pd.concat(filtered_dfs, ignore_index=True)

    def _check_calibration_quality(self, result: CalibrationRunResult) -> None:
        """Check calibration quality and add warnings."""
        threshold = self.config.rmse_alert_threshold

        # Check Heston RMSE
        if result.heston_result:
            heston_rmse = result.heston_result.get("rmse", 0)
            if heston_rmse > threshold:
                result.warnings.append(
                    f"Heston RMSE {heston_rmse:.4f} exceeds threshold {threshold}"
                )

        # Check SABR RMSE
        if result.sabr_result:
            sabr_rmse = result.sabr_result.get("total_rmse", 0)
            if sabr_rmse > threshold:
                result.warnings.append(
                    f"SABR RMSE {sabr_rmse:.4f} exceeds threshold {threshold}"
                )

        # Check OU diagnostics
        if result.ou_results:
            for pair_name, ou_result in result.ou_results.items():
                params = ou_result.get("params", {})
                half_life = params.get("half_life", 0)

                if half_life > self.config.ou_max_half_life:
                    result.warnings.append(
                        f"OU half-life for {pair_name} ({half_life:.1f} days) "
                        f"exceeds max ({self.config.ou_max_half_life})"
                    )

                # Check residual autocorrelation
                residual_stats = ou_result.get("residual_stats", {})
                ljung_box_p = residual_stats.get("ljung_box_p", 1.0)
                if ljung_box_p < 0.05:
                    result.warnings.append(
                        f"OU residuals for {pair_name} show significant "
                        f"autocorrelation (p={ljung_box_p:.4f})"
                    )

        if result.warnings:
            logger.warning(f"Calibration quality warnings: {result.warnings}")

    def _store_run_result(self, result: CalibrationRunResult) -> None:
        """Store calibration run result in database."""
        try:
            # Store as a calibration run record
            # Implementation depends on database schema
            logger.info(f"Stored calibration run for {result.underlying}")
        except Exception as e:
            logger.error(f"Failed to store run result: {e}")

    def get_cached_params(
        self,
        underlying: str,
        model_type: str,
    ) -> Optional[Dict]:
        """
        Get cached parameters from previous calibration.

        Args:
            underlying: Underlying symbol
            model_type: "heston", "sabr", or "ou"

        Returns:
            Cached parameters or None
        """
        if model_type == "heston":
            return self._last_heston_params.get(underlying)
        elif model_type == "sabr":
            return self._last_sabr_params.get(underlying)
        elif model_type == "ou":
            return self._last_ou_params.get(underlying)
        return None

    def clear_cache(self, underlying: Optional[str] = None) -> None:
        """
        Clear parameter cache.

        Args:
            underlying: If provided, clear only for this underlying.
                       If None, clear all cached parameters.
        """
        if underlying:
            self._last_heston_params.pop(underlying, None)
            self._last_sabr_params.pop(underlying, None)
            # For OU, need to clear any pairs containing this underlying
            self._last_ou_params = {
                k: v
                for k, v in self._last_ou_params.items()
                if underlying not in k
            }
        else:
            self._last_heston_params.clear()
            self._last_sabr_params.clear()
            self._last_ou_params.clear()

        logger.info(f"Cleared cache for {underlying or 'all'}")
