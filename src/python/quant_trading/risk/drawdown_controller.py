"""
Drawdown Controller and Risk Limits.

Monitors portfolio drawdown and enforces risk limits:
    - Maximum drawdown thresholds with position reduction
    - Strategy-level and portfolio-level limits
    - Recovery tracking and alert management
    - Kill switch capability for critical situations

Target: Max drawdown <25% as per design doc specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Portfolio risk level."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @property
    def severity(self) -> int:
        """Get numeric severity for comparison (higher = more severe)."""
        severity_map = {
            "normal": 0,
            "elevated": 1,
            "high": 2,
            "critical": 3,
            "emergency": 4,
        }
        return severity_map.get(self.value, 0)


class RiskAction(Enum):
    """Risk management actions."""

    NO_ACTION = "no_action"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_NEW_TRADES = "halt_new_trades"
    CLOSE_POSITIONS = "close_positions"
    KILL_SWITCH = "kill_switch"


@dataclass
class DrawdownMetrics:
    """Drawdown and risk metrics."""

    current_drawdown: float  # Current drawdown from peak (positive number)
    max_drawdown: float  # Maximum drawdown ever
    drawdown_duration_days: int  # Days since peak
    peak_value: float  # Peak portfolio value
    current_value: float  # Current portfolio value
    recovery_needed: float  # % gain needed to recover

    # Rolling metrics
    drawdown_30d: float = 0.0
    drawdown_60d: float = 0.0
    drawdown_90d: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration_days": self.drawdown_duration_days,
            "peak_value": self.peak_value,
            "current_value": self.current_value,
            "recovery_needed": self.recovery_needed,
            "drawdown_30d": self.drawdown_30d,
            "drawdown_60d": self.drawdown_60d,
            "drawdown_90d": self.drawdown_90d,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskLimitStatus:
    """Status of risk limits."""

    risk_level: RiskLevel
    recommended_action: RiskAction
    limits_breached: List[str]
    exposure_multiplier: float  # Multiplier to apply to new positions
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "risk_level": self.risk_level.value,
            "recommended_action": self.recommended_action.value,
            "limits_breached": self.limits_breached,
            "exposure_multiplier": self.exposure_multiplier,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DrawdownControllerConfig:
    """Configuration for drawdown controller."""

    # Drawdown thresholds (as positive fractions)
    warning_threshold: float = 0.10  # 10% drawdown → warning
    elevated_threshold: float = 0.15  # 15% drawdown → elevated
    high_threshold: float = 0.20  # 20% drawdown → high risk
    critical_threshold: float = 0.25  # 25% drawdown → critical (design doc target)
    emergency_threshold: float = 0.30  # 30% drawdown → emergency

    # Exposure reduction at each level
    elevated_exposure_mult: float = 0.75  # Reduce to 75%
    high_exposure_mult: float = 0.50  # Reduce to 50%
    critical_exposure_mult: float = 0.25  # Reduce to 25%
    emergency_exposure_mult: float = 0.0  # Close all

    # Daily loss limits
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    consecutive_loss_days_limit: int = 5  # Alert after 5 consecutive losing days

    # Strategy-level limits
    max_strategy_drawdown: float = 0.15  # 15% max per strategy
    max_position_loss_pct: float = 0.10  # 10% max loss per position

    # Recovery settings
    min_recovery_before_increase: float = 0.05  # Need 5% recovery before increasing


class DrawdownController:
    """
    Monitors portfolio drawdown and enforces risk limits.

    Features:
        - Real-time drawdown tracking
        - Tiered risk responses based on drawdown severity
        - Strategy-level and position-level limits
        - Kill switch for emergency situations
        - Recovery tracking

    Example:
        >>> controller = DrawdownController()
        >>> controller.update(portfolio_value=1_000_000)
        >>> # Later...
        >>> controller.update(portfolio_value=850_000)
        >>> status = controller.check_limits()
        >>> print(f"Risk level: {status.risk_level.value}")
        >>> print(f"Action: {status.recommended_action.value}")

    Reference:
        Standard risk management practices for trading systems
    """

    def __init__(
        self,
        config: Optional[DrawdownControllerConfig] = None,
        initial_capital: float = 1_000_000,
    ):
        """
        Initialize drawdown controller.

        Args:
            config: Configuration parameters
            initial_capital: Starting portfolio value
        """
        self.config = config or DrawdownControllerConfig()
        self.initial_capital = initial_capital

        # State tracking
        self._peak_value = initial_capital
        self._peak_date = datetime.utcnow()
        self._current_value = initial_capital
        self._max_drawdown = 0.0
        self._value_history: List[tuple[datetime, float]] = []
        self._daily_returns: List[float] = []

        # Kill switch state
        self._kill_switch_active = False
        self._kill_switch_reason: Optional[str] = None

        logger.info(
            f"Initialized DrawdownController with critical_threshold="
            f"{self.config.critical_threshold:.0%}"
        )

    def update(
        self,
        portfolio_value: float,
        timestamp: Optional[datetime] = None,
    ) -> DrawdownMetrics:
        """
        Update controller with new portfolio value.

        Args:
            portfolio_value: Current portfolio value
            timestamp: Timestamp of update (default: now)

        Returns:
            DrawdownMetrics with current state
        """
        timestamp = timestamp or datetime.utcnow()

        # Track daily return
        if self._current_value > 0:
            daily_return = (portfolio_value - self._current_value) / self._current_value
            self._daily_returns.append(daily_return)
            # Keep last 252 days
            self._daily_returns = self._daily_returns[-252:]

        self._current_value = portfolio_value

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value
            self._peak_date = timestamp

        # Add to history
        self._value_history.append((timestamp, portfolio_value))
        # Keep last year of history
        cutoff = timestamp - timedelta(days=365)
        self._value_history = [
            (t, v) for t, v in self._value_history if t >= cutoff
        ]

        # Compute metrics
        metrics = self._compute_metrics(timestamp)

        # Update max drawdown
        if metrics.current_drawdown > self._max_drawdown:
            self._max_drawdown = metrics.current_drawdown

        return metrics

    def _compute_metrics(self, timestamp: datetime) -> DrawdownMetrics:
        """Compute current drawdown metrics."""
        current_dd = 0.0
        if self._peak_value > 0:
            current_dd = (self._peak_value - self._current_value) / self._peak_value

        # Duration
        duration_days = (timestamp - self._peak_date).days

        # Recovery needed
        recovery_needed = 0.0
        if self._current_value > 0 and self._current_value < self._peak_value:
            recovery_needed = (self._peak_value / self._current_value) - 1

        # Rolling drawdowns
        dd_30d = self._compute_rolling_drawdown(30)
        dd_60d = self._compute_rolling_drawdown(60)
        dd_90d = self._compute_rolling_drawdown(90)

        return DrawdownMetrics(
            current_drawdown=max(0, current_dd),
            max_drawdown=self._max_drawdown,
            drawdown_duration_days=max(0, duration_days),
            peak_value=self._peak_value,
            current_value=self._current_value,
            recovery_needed=recovery_needed,
            drawdown_30d=dd_30d,
            drawdown_60d=dd_60d,
            drawdown_90d=dd_90d,
            timestamp=timestamp,
        )

    def _compute_rolling_drawdown(self, days: int) -> float:
        """Compute rolling max drawdown over N days."""
        if len(self._value_history) < 2:
            return 0.0

        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [(t, v) for t, v in self._value_history if t >= cutoff]

        if len(recent) < 2:
            return 0.0

        values = [v for _, v in recent]
        peak = values[0]
        max_dd = 0.0

        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    def check_limits(self) -> RiskLimitStatus:
        """
        Check all risk limits and return status.

        Returns:
            RiskLimitStatus with risk level and recommended action
        """
        # Check kill switch first
        if self._kill_switch_active:
            return RiskLimitStatus(
                risk_level=RiskLevel.EMERGENCY,
                recommended_action=RiskAction.KILL_SWITCH,
                limits_breached=["kill_switch_active"],
                exposure_multiplier=0.0,
                message=f"Kill switch active: {self._kill_switch_reason}",
            )

        metrics = self._compute_metrics(datetime.utcnow())
        limits_breached = []
        risk_level = RiskLevel.NORMAL
        exposure_mult = 1.0

        # Check drawdown thresholds
        if metrics.current_drawdown >= self.config.emergency_threshold:
            risk_level = RiskLevel.EMERGENCY
            exposure_mult = self.config.emergency_exposure_mult
            limits_breached.append(
                f"drawdown {metrics.current_drawdown:.1%} >= emergency "
                f"{self.config.emergency_threshold:.0%}"
            )
        elif metrics.current_drawdown >= self.config.critical_threshold:
            risk_level = RiskLevel.CRITICAL
            exposure_mult = self.config.critical_exposure_mult
            limits_breached.append(
                f"drawdown {metrics.current_drawdown:.1%} >= critical "
                f"{self.config.critical_threshold:.0%}"
            )
        elif metrics.current_drawdown >= self.config.high_threshold:
            risk_level = RiskLevel.HIGH
            exposure_mult = self.config.high_exposure_mult
            limits_breached.append(
                f"drawdown {metrics.current_drawdown:.1%} >= high "
                f"{self.config.high_threshold:.0%}"
            )
        elif metrics.current_drawdown >= self.config.elevated_threshold:
            risk_level = RiskLevel.ELEVATED
            exposure_mult = self.config.elevated_exposure_mult
            limits_breached.append(
                f"drawdown {metrics.current_drawdown:.1%} >= elevated "
                f"{self.config.elevated_threshold:.0%}"
            )

        # Check daily loss
        if self._daily_returns:
            last_return = self._daily_returns[-1]
            if last_return < -self.config.max_daily_loss_pct:
                if risk_level.severity < RiskLevel.ELEVATED.severity:
                    risk_level = RiskLevel.ELEVATED
                limits_breached.append(
                    f"daily loss {last_return:.1%} exceeds limit "
                    f"-{self.config.max_daily_loss_pct:.0%}"
                )

        # Check consecutive losses
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.config.consecutive_loss_days_limit:
            if risk_level.severity < RiskLevel.ELEVATED.severity:
                risk_level = RiskLevel.ELEVATED
            limits_breached.append(
                f"{consecutive_losses} consecutive losing days"
            )

        # Determine action
        action = self._determine_action(risk_level)

        message = self._build_status_message(risk_level, metrics, limits_breached)

        return RiskLimitStatus(
            risk_level=risk_level,
            recommended_action=action,
            limits_breached=limits_breached,
            exposure_multiplier=exposure_mult,
            message=message,
        )

    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing days from most recent."""
        count = 0
        for ret in reversed(self._daily_returns):
            if ret < 0:
                count += 1
            else:
                break
        return count

    def _determine_action(self, risk_level: RiskLevel) -> RiskAction:
        """Determine recommended action based on risk level."""
        action_map = {
            RiskLevel.NORMAL: RiskAction.NO_ACTION,
            RiskLevel.ELEVATED: RiskAction.REDUCE_EXPOSURE,
            RiskLevel.HIGH: RiskAction.REDUCE_EXPOSURE,
            RiskLevel.CRITICAL: RiskAction.HALT_NEW_TRADES,
            RiskLevel.EMERGENCY: RiskAction.CLOSE_POSITIONS,
        }
        return action_map.get(risk_level, RiskAction.NO_ACTION)

    def _build_status_message(
        self,
        risk_level: RiskLevel,
        metrics: DrawdownMetrics,
        limits_breached: List[str],
    ) -> str:
        """Build human-readable status message."""
        if risk_level == RiskLevel.NORMAL:
            return (
                f"Risk normal. Drawdown: {metrics.current_drawdown:.1%}, "
                f"Peak: ${metrics.peak_value:,.0f}"
            )

        breaches = "; ".join(limits_breached) if limits_breached else "none"
        return (
            f"Risk {risk_level.value}. Drawdown: {metrics.current_drawdown:.1%}, "
            f"Recovery needed: {metrics.recovery_needed:.1%}. "
            f"Limits breached: {breaches}"
        )

    def activate_kill_switch(self, reason: str) -> None:
        """
        Activate emergency kill switch.

        This will halt all trading and recommend closing all positions.

        Args:
            reason: Reason for activation
        """
        self._kill_switch_active = True
        self._kill_switch_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch (requires manual intervention)."""
        self._kill_switch_active = False
        self._kill_switch_reason = None
        logger.info("Kill switch deactivated")

    def check_strategy_limits(
        self,
        strategy_values: Dict[str, float],
        strategy_peaks: Dict[str, float],
    ) -> Dict[str, RiskLimitStatus]:
        """
        Check risk limits for individual strategies.

        Args:
            strategy_values: Dict of {strategy_name: current_value}
            strategy_peaks: Dict of {strategy_name: peak_value}

        Returns:
            Dict of {strategy_name: RiskLimitStatus}
        """
        results = {}

        for strategy, value in strategy_values.items():
            peak = strategy_peaks.get(strategy, value)
            drawdown = (peak - value) / peak if peak > 0 else 0

            if drawdown >= self.config.max_strategy_drawdown:
                results[strategy] = RiskLimitStatus(
                    risk_level=RiskLevel.CRITICAL,
                    recommended_action=RiskAction.HALT_NEW_TRADES,
                    limits_breached=[
                        f"strategy drawdown {drawdown:.1%} >= limit "
                        f"{self.config.max_strategy_drawdown:.0%}"
                    ],
                    exposure_multiplier=0.25,
                    message=f"Strategy {strategy} exceeds drawdown limit",
                )
            else:
                results[strategy] = RiskLimitStatus(
                    risk_level=RiskLevel.NORMAL,
                    recommended_action=RiskAction.NO_ACTION,
                    limits_breached=[],
                    exposure_multiplier=1.0,
                    message=f"Strategy {strategy} within limits",
                )

        return results

    def get_metrics(self) -> DrawdownMetrics:
        """Get current drawdown metrics."""
        return self._compute_metrics(datetime.utcnow())

    def reset(self, new_capital: float) -> None:
        """
        Reset controller with new starting capital.

        Use when starting a new trading period or after a capital injection.

        Args:
            new_capital: New starting capital
        """
        self.initial_capital = new_capital
        self._peak_value = new_capital
        self._peak_date = datetime.utcnow()
        self._current_value = new_capital
        self._max_drawdown = 0.0
        self._value_history = []
        self._daily_returns = []
        self._kill_switch_active = False
        self._kill_switch_reason = None

        logger.info(f"DrawdownController reset with capital ${new_capital:,.0f}")
