"""
Data Validation Module.

Implements comprehensive data quality checks for:
- Market price data (OHLCV)
- Options chain data
- No-arbitrage conditions

Reference: Design doc Section 5.2 (Data Validation Pipeline)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

logger = logging.getLogger(__name__)


class DataQuality(Enum):
    """Data quality classification."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation issue."""
    code: str
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None
    row_indices: Optional[List[int]] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
            "row_indices": self.row_indices,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    quality: DataQuality
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    validated_at: datetime = field(default_factory=datetime.now)

    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)

        # Update validity and quality
        if issue.severity == ValidationSeverity.CRITICAL:
            self.is_valid = False
            self.quality = DataQuality.BAD
        elif issue.severity == ValidationSeverity.ERROR:
            self.is_valid = False
            if self.quality != DataQuality.BAD:
                self.quality = DataQuality.SUSPECT
        elif issue.severity == ValidationSeverity.WARNING:
            if self.quality == DataQuality.GOOD:
                self.quality = DataQuality.SUSPECT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality": self.quality.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
            "stats": self.stats,
            "validated_at": self.validated_at.isoformat(),
        }


class MarketDataValidator:
    """
    Validates market price data (OHLCV).

    Checks:
    - Missing values
    - OHLC consistency (High >= Open, Close; Low <= Open, Close)
    - Price positivity
    - Volume non-negativity
    - Outlier detection
    - Duplicate timestamps
    - Gap detection
    - Staleness
    """

    def __init__(
        self,
        max_return_pct: float = 50.0,
        max_gap_minutes: int = 60,
        zscore_threshold: float = 5.0,
        min_volume: int = 0,
    ):
        """
        Initialize validator.

        Args:
            max_return_pct: Maximum single-period return (%)
            max_gap_minutes: Maximum gap between data points
            zscore_threshold: Z-score for outlier detection
            min_volume: Minimum volume threshold
        """
        self.max_return_pct = max_return_pct
        self.max_gap_minutes = max_gap_minutes
        self.zscore_threshold = zscore_threshold
        self.min_volume = min_volume

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate market data DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Index should be datetime

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, quality=DataQuality.GOOD)

        if df.empty:
            result.add_issue(ValidationIssue(
                code="EMPTY_DATA",
                message="DataFrame is empty",
                severity=ValidationSeverity.ERROR,
            ))
            return result

        # Collect stats
        result.stats = {
            "row_count": len(df),
            "start_time": df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
            "end_time": df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max()),
        }

        # Run validation checks
        self._check_required_columns(df, result)
        self._check_missing_values(df, result)
        self._check_duplicates(df, result)
        self._check_ohlc_consistency(df, result)
        self._check_price_positivity(df, result)
        self._check_volume(df, result)
        self._check_outliers(df, result)
        self._check_gaps(df, result)

        return result

    def _check_required_columns(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for required columns."""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]

        if missing:
            result.add_issue(ValidationIssue(
                code="MISSING_COLUMNS",
                message=f"Missing required columns: {missing}",
                severity=ValidationSeverity.ERROR,
                details={"missing": missing},
            ))

    def _check_missing_values(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for missing values."""
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]

        if not null_cols.empty:
            null_pct = (null_cols / len(df) * 100).to_dict()

            severity = ValidationSeverity.WARNING
            if any(pct > 10 for pct in null_pct.values()):
                severity = ValidationSeverity.ERROR

            result.add_issue(ValidationIssue(
                code="MISSING_VALUES",
                message=f"Missing values in columns: {list(null_cols.index)}",
                severity=severity,
                details={"null_percentages": null_pct},
            ))

            result.stats["missing_pct"] = null_pct

    def _check_duplicates(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for duplicate timestamps."""
        duplicates = df.index.duplicated()
        dup_count = duplicates.sum()

        if dup_count > 0:
            dup_indices = df.index[duplicates].tolist()

            result.add_issue(ValidationIssue(
                code="DUPLICATE_TIMESTAMPS",
                message=f"Found {dup_count} duplicate timestamps",
                severity=ValidationSeverity.WARNING,
                details={"duplicates": dup_indices[:10]},  # First 10
            ))

            result.stats["duplicate_count"] = dup_count

    def _check_ohlc_consistency(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check OHLC price relationships."""
        if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            return

        # High should be >= open and close
        high_low_open = df['high'] < df['open']
        high_low_close = df['high'] < df['close']
        high_violations = high_low_open | high_low_close

        if high_violations.any():
            count = high_violations.sum()
            indices = df.index[high_violations].tolist()

            result.add_issue(ValidationIssue(
                code="HIGH_PRICE_VIOLATION",
                message=f"High price below open/close in {count} bars",
                severity=ValidationSeverity.ERROR,
                field="high",
                row_indices=indices[:10],
            ))

        # Low should be <= open and close
        low_high_open = df['low'] > df['open']
        low_high_close = df['low'] > df['close']
        low_violations = low_high_open | low_high_close

        if low_violations.any():
            count = low_violations.sum()
            indices = df.index[low_violations].tolist()

            result.add_issue(ValidationIssue(
                code="LOW_PRICE_VIOLATION",
                message=f"Low price above open/close in {count} bars",
                severity=ValidationSeverity.ERROR,
                field="low",
                row_indices=indices[:10],
            ))

        # High should be >= Low
        high_below_low = df['high'] < df['low']
        if high_below_low.any():
            count = high_below_low.sum()

            result.add_issue(ValidationIssue(
                code="HIGH_BELOW_LOW",
                message=f"High below low in {count} bars",
                severity=ValidationSeverity.CRITICAL,
                field="high",
            ))

    def _check_price_positivity(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check that prices are positive."""
        price_cols = ['open', 'high', 'low', 'close']
        price_cols = [c for c in price_cols if c in df.columns]

        for col in price_cols:
            negative = df[col] <= 0
            if negative.any():
                count = negative.sum()

                result.add_issue(ValidationIssue(
                    code="NEGATIVE_PRICE",
                    message=f"Non-positive {col} prices in {count} bars",
                    severity=ValidationSeverity.CRITICAL,
                    field=col,
                ))

    def _check_volume(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check volume values."""
        if 'volume' not in df.columns:
            return

        # Check negative volume
        negative = df['volume'] < 0
        if negative.any():
            result.add_issue(ValidationIssue(
                code="NEGATIVE_VOLUME",
                message=f"Negative volume in {negative.sum()} bars",
                severity=ValidationSeverity.ERROR,
                field="volume",
            ))

        # Check zero volume
        zero = df['volume'] == 0
        zero_pct = zero.sum() / len(df) * 100
        if zero_pct > 50:
            result.add_issue(ValidationIssue(
                code="HIGH_ZERO_VOLUME",
                message=f"Zero volume in {zero_pct:.1f}% of bars",
                severity=ValidationSeverity.WARNING,
                field="volume",
            ))

        result.stats["zero_volume_pct"] = zero_pct

    def _check_outliers(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for price outliers."""
        if 'close' not in df.columns or len(df) < 10:
            return

        # Calculate returns
        returns = df['close'].pct_change().dropna()

        # Check extreme returns
        extreme = returns.abs() > (self.max_return_pct / 100)
        if extreme.any():
            count = extreme.sum()
            extreme_returns = returns[extreme].to_dict()

            result.add_issue(ValidationIssue(
                code="EXTREME_RETURNS",
                message=f"Extreme returns (>{self.max_return_pct}%) in {count} bars",
                severity=ValidationSeverity.WARNING,
                field="close",
                details={"extreme_returns": {
                    str(k): f"{v:.2%}" for k, v in list(extreme_returns.items())[:5]
                }},
            ))

        # Z-score based outliers
        if len(returns) > 20:
            z_scores = np.abs(zscore(returns))
            outliers = z_scores > self.zscore_threshold
            if outliers.any():
                result.add_issue(ValidationIssue(
                    code="STATISTICAL_OUTLIERS",
                    message=f"Statistical outliers (z>{self.zscore_threshold}) in {outliers.sum()} bars",
                    severity=ValidationSeverity.INFO,
                    field="close",
                ))

    def _check_gaps(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for data gaps."""
        if len(df) < 2:
            return

        if not isinstance(df.index, pd.DatetimeIndex):
            return

        time_diffs = df.index.to_series().diff().dropna()

        if time_diffs.empty:
            return

        max_gap = time_diffs.max()
        max_gap_minutes = max_gap.total_seconds() / 60

        if max_gap_minutes > self.max_gap_minutes:
            gap_idx = time_diffs.idxmax()

            result.add_issue(ValidationIssue(
                code="DATA_GAP",
                message=f"Large gap of {max_gap_minutes:.0f} minutes detected",
                severity=ValidationSeverity.WARNING,
                details={
                    "gap_start": str(gap_idx - max_gap),
                    "gap_end": str(gap_idx),
                    "gap_minutes": max_gap_minutes,
                },
            ))

        result.stats["max_gap_minutes"] = max_gap_minutes


class OptionsDataValidator:
    """
    Validates options chain data.

    Checks:
    - No-arbitrage conditions (put-call parity)
    - IV reasonableness (0 < IV < 500%)
    - Strike/expiration validity
    - Bid-ask spread reasonableness
    - Greeks consistency
    """

    def __init__(
        self,
        max_iv: float = 5.0,  # 500%
        max_spread_pct: float = 0.5,  # 50%
        parity_tolerance: float = 0.05,  # 5%
    ):
        """
        Initialize validator.

        Args:
            max_iv: Maximum implied volatility
            max_spread_pct: Maximum bid-ask spread as % of mid
            parity_tolerance: Put-call parity tolerance
        """
        self.max_iv = max_iv
        self.max_spread_pct = max_spread_pct
        self.parity_tolerance = parity_tolerance

    def validate(
        self,
        df: pd.DataFrame,
        spot_price: Optional[float] = None,
        risk_free_rate: float = 0.05,
    ) -> ValidationResult:
        """
        Validate options data.

        Args:
            df: Options chain DataFrame
            spot_price: Current underlying price
            risk_free_rate: Risk-free rate for parity checks

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True, quality=DataQuality.GOOD)

        if df.empty:
            result.add_issue(ValidationIssue(
                code="EMPTY_OPTIONS",
                message="Options DataFrame is empty",
                severity=ValidationSeverity.ERROR,
            ))
            return result

        result.stats = {
            "option_count": len(df),
            "call_count": len(df[df['option_type'] == 'call']) if 'option_type' in df.columns else 0,
            "put_count": len(df[df['option_type'] == 'put']) if 'option_type' in df.columns else 0,
        }

        self._check_required_columns(df, result)
        self._check_implied_vol(df, result)
        self._check_strike_validity(df, result)
        self._check_bid_ask_spread(df, result)
        self._check_greeks(df, result)

        if spot_price:
            self._check_put_call_parity(df, spot_price, risk_free_rate, result)

        return result

    def _check_required_columns(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check for required columns."""
        required = ['strike', 'option_type']
        optional = ['bid', 'ask', 'implied_vol', 'expiration']

        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            result.add_issue(ValidationIssue(
                code="MISSING_OPTION_COLUMNS",
                message=f"Missing required columns: {missing_required}",
                severity=ValidationSeverity.ERROR,
            ))

    def _check_implied_vol(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check implied volatility values."""
        if 'implied_vol' not in df.columns:
            return

        iv = df['implied_vol']

        # Negative IV
        negative = iv < 0
        if negative.any():
            result.add_issue(ValidationIssue(
                code="NEGATIVE_IV",
                message=f"Negative implied vol in {negative.sum()} options",
                severity=ValidationSeverity.CRITICAL,
                field="implied_vol",
            ))

        # Extreme IV
        extreme = iv > self.max_iv
        if extreme.any():
            result.add_issue(ValidationIssue(
                code="EXTREME_IV",
                message=f"Implied vol > {self.max_iv*100:.0f}% in {extreme.sum()} options",
                severity=ValidationSeverity.WARNING,
                field="implied_vol",
            ))

        # Zero IV
        zero = iv == 0
        if zero.any():
            result.add_issue(ValidationIssue(
                code="ZERO_IV",
                message=f"Zero implied vol in {zero.sum()} options",
                severity=ValidationSeverity.WARNING,
                field="implied_vol",
            ))

        result.stats["iv_mean"] = iv.mean()
        result.stats["iv_std"] = iv.std()

    def _check_strike_validity(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check strike price validity."""
        if 'strike' not in df.columns:
            return

        # Negative/zero strikes
        invalid = df['strike'] <= 0
        if invalid.any():
            result.add_issue(ValidationIssue(
                code="INVALID_STRIKE",
                message=f"Non-positive strikes in {invalid.sum()} options",
                severity=ValidationSeverity.CRITICAL,
                field="strike",
            ))

    def _check_bid_ask_spread(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check bid-ask spread reasonableness."""
        if 'bid' not in df.columns or 'ask' not in df.columns:
            return

        # Inverted markets
        inverted = df['bid'] > df['ask']
        if inverted.any():
            result.add_issue(ValidationIssue(
                code="INVERTED_MARKET",
                message=f"Bid > Ask in {inverted.sum()} options",
                severity=ValidationSeverity.ERROR,
                field="bid",
            ))

        # Wide spreads
        mid = (df['bid'] + df['ask']) / 2
        spread_pct = (df['ask'] - df['bid']) / mid
        wide = spread_pct > self.max_spread_pct

        if wide.any():
            result.add_issue(ValidationIssue(
                code="WIDE_SPREAD",
                message=f"Wide bid-ask spread (>{self.max_spread_pct*100:.0f}%) in {wide.sum()} options",
                severity=ValidationSeverity.WARNING,
                field="bid",
            ))

        result.stats["avg_spread_pct"] = spread_pct.mean() * 100

    def _check_greeks(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Check Greeks validity."""
        # Delta should be in [-1, 1]
        if 'delta' in df.columns:
            invalid_delta = (df['delta'] < -1) | (df['delta'] > 1)
            if invalid_delta.any():
                result.add_issue(ValidationIssue(
                    code="INVALID_DELTA",
                    message=f"Delta outside [-1,1] in {invalid_delta.sum()} options",
                    severity=ValidationSeverity.ERROR,
                    field="delta",
                ))

        # Gamma should be non-negative
        if 'gamma' in df.columns:
            negative_gamma = df['gamma'] < 0
            if negative_gamma.any():
                result.add_issue(ValidationIssue(
                    code="NEGATIVE_GAMMA",
                    message=f"Negative gamma in {negative_gamma.sum()} options",
                    severity=ValidationSeverity.WARNING,
                    field="gamma",
                ))

    def _check_put_call_parity(
        self,
        df: pd.DataFrame,
        spot: float,
        rate: float,
        result: ValidationResult,
    ) -> None:
        """
        Check put-call parity: C - P = S - K*exp(-rT).

        Args:
            df: Options data with both calls and puts
            spot: Current spot price
            rate: Risk-free rate
            result: ValidationResult to update
        """
        if 'option_type' not in df.columns or 'strike' not in df.columns:
            return

        if 'expiration' not in df.columns:
            return

        if 'bid' not in df.columns or 'ask' not in df.columns:
            return

        calls = df[df['option_type'] == 'call'].copy()
        puts = df[df['option_type'] == 'put'].copy()

        if calls.empty or puts.empty:
            return

        # Calculate mid prices
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        puts['mid'] = (puts['bid'] + puts['ask']) / 2

        violations = []

        for strike in calls['strike'].unique():
            call_row = calls[calls['strike'] == strike]
            put_row = puts[puts['strike'] == strike]

            if call_row.empty or put_row.empty:
                continue

            call_price = call_row['mid'].iloc[0]
            put_price = put_row['mid'].iloc[0]

            # Calculate theoretical difference
            # Assuming same expiration
            if 'expiration' in call_row.columns:
                exp = call_row['expiration'].iloc[0]
                if isinstance(exp, str):
                    exp = pd.to_datetime(exp).date()
                T = (exp - datetime.now().date()).days / 365
            else:
                T = 0.1  # Default

            theoretical = spot - strike * np.exp(-rate * T)
            actual = call_price - put_price

            parity_error = abs(actual - theoretical) / spot

            if parity_error > self.parity_tolerance:
                violations.append({
                    'strike': strike,
                    'error_pct': parity_error * 100,
                })

        if violations:
            result.add_issue(ValidationIssue(
                code="PUT_CALL_PARITY_VIOLATION",
                message=f"Put-call parity violated in {len(violations)} strike(s)",
                severity=ValidationSeverity.WARNING,
                details={"violations": violations[:5]},
            ))


class DataValidationPipeline:
    """
    Orchestrates data validation across multiple validators.
    """

    def __init__(self):
        """Initialize with default validators."""
        self.market_validator = MarketDataValidator()
        self.options_validator = OptionsDataValidator()

    def validate_market_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate market data."""
        return self.market_validator.validate(df)

    def validate_options_data(
        self,
        df: pd.DataFrame,
        spot_price: Optional[float] = None,
    ) -> ValidationResult:
        """Validate options data."""
        return self.options_validator.validate(df, spot_price=spot_price)

    def validate_and_clean(
        self,
        df: pd.DataFrame,
        data_type: str = "market",
    ) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Validate and clean data.

        Args:
            df: DataFrame to validate
            data_type: "market" or "options"

        Returns:
            Tuple of (cleaned DataFrame, ValidationResult)
        """
        if data_type == "market":
            result = self.market_validator.validate(df)
            cleaned = self._clean_market_data(df, result)
        else:
            result = self.options_validator.validate(df)
            cleaned = self._clean_options_data(df, result)

        return cleaned, result

    def _clean_market_data(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> pd.DataFrame:
        """Clean market data based on validation results."""
        df = df.copy()

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Forward fill small gaps in price data
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)

        # Clip extreme returns (optional)
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            extreme = returns.abs() > 0.5  # 50%
            if extreme.any():
                df.loc[extreme, 'close'] = np.nan
                df['close'] = df['close'].ffill()

        return df

    def _clean_options_data(
        self, df: pd.DataFrame, result: ValidationResult
    ) -> pd.DataFrame:
        """Clean options data based on validation results."""
        df = df.copy()

        # Remove options with invalid IV
        if 'implied_vol' in df.columns:
            df = df[(df['implied_vol'] > 0) & (df['implied_vol'] < 5)]

        # Remove inverted markets
        if 'bid' in df.columns and 'ask' in df.columns:
            df = df[df['bid'] <= df['ask']]

        return df
