"""
Risk Manager - Unified Risk Management Framework.

Provides comprehensive portfolio risk management:
    - Risk limits enforcement (position size, exposure, Greeks, VaR, drawdown)
    - Position and portfolio risk tracking
    - Circuit breaker functionality
    - Risk budget allocation

This module integrates with other risk components:
    - DrawdownController for drawdown monitoring
    - GreeksRiskMonitor for options Greeks
    - CorrelationMonitor for cointegration health

Target: Max drawdown <25% with Sharpe ratio 0.5-1.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RiskLimitType(Enum):
    """Types of risk limits."""

    POSITION_SIZE = "position_size"  # Max size per position (as % of capital)
    TOTAL_EXPOSURE = "total_exposure"  # Max total gross exposure
    NET_EXPOSURE = "net_exposure"  # Max net exposure (long - short)
    SECTOR_EXPOSURE = "sector_exposure"  # Max exposure per sector
    DELTA = "delta"  # Max portfolio delta
    GAMMA = "gamma"  # Max portfolio gamma
    VEGA = "vega"  # Max portfolio vega
    VAR = "var"  # Max Value at Risk
    DRAWDOWN = "drawdown"  # Max drawdown from peak
    LEVERAGE = "leverage"  # Max leverage ratio
    CONCENTRATION = "concentration"  # Max concentration (Herfindahl)
    DAILY_LOSS = "daily_loss"  # Max daily loss


@dataclass
class RiskLimit:
    """
    Risk limit specification.

    Attributes:
        limit_type: Type of risk limit
        value: Limit value (interpretation depends on limit_type)
        warning_threshold: Threshold for warning (as fraction of limit)
        action_on_breach: Action to take on breach ("alert", "reduce", "halt")
        metadata: Additional limit-specific configuration
    """

    limit_type: RiskLimitType
    value: float
    warning_threshold: float = 0.8
    action_on_breach: str = "alert"  # "alert", "reduce", "halt"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check_breach(self, current_value: float) -> tuple[bool, str]:
        """
        Check if limit is breached.

        Args:
            current_value: Current value to check against limit

        Returns:
            Tuple of (is_breached, level) where level is "ok", "warning", or "breach"
        """
        # For most limits, check absolute value against limit
        if abs(current_value) >= abs(self.value):
            return True, "breach"
        elif abs(current_value) >= abs(self.value * self.warning_threshold):
            return False, "warning"
        else:
            return False, "ok"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "limit_type": self.limit_type.value,
            "value": self.value,
            "warning_threshold": self.warning_threshold,
            "action_on_breach": self.action_on_breach,
            "metadata": self.metadata,
        }


@dataclass
class PositionRisk:
    """
    Risk metrics for a single position.

    Attributes:
        asset_id: Asset identifier (e.g., ticker symbol)
        position_size: Number of shares/contracts (positive=long, negative=short)
        market_value: Current market value
        entry_price: Entry price for P&L calculation
        current_price: Current market price
        pnl: Unrealized P&L
        pnl_pct: P&L as percentage of entry value
        weight: Position weight in portfolio

    Greeks (for options):
        delta: Position delta exposure
        gamma: Position gamma
        vega: Position vega
        theta: Position theta

    Risk metrics:
        var_95: 95% Value at Risk (1-day)
        contribution_to_var: Marginal VaR contribution
        volatility: Position volatility (annualized)
    """

    asset_id: str
    position_size: float
    market_value: float
    entry_price: float = 0.0
    current_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    weight: float = 0.0

    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    contribution_to_var: float = 0.0
    volatility: float = 0.0

    # Metadata
    sector: Optional[str] = None
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "position_size": self.position_size,
            "market_value": self.market_value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "weight": self.weight,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "var_95": self.var_95,
            "contribution_to_var": self.contribution_to_var,
            "volatility": self.volatility,
            "sector": self.sector,
            "strategy": self.strategy,
        }


@dataclass
class PortfolioRisk:
    """
    Portfolio-level risk metrics.

    Attributes:
        total_value: Total portfolio value
        total_exposure: Gross exposure (sum of |market_value|)
        net_exposure: Net exposure (sum of market_value)
        leverage: Leverage ratio (total_exposure / total_capital)

    Greeks:
        total_delta: Aggregate portfolio delta
        total_gamma: Aggregate portfolio gamma
        total_vega: Aggregate portfolio vega
        total_theta: Aggregate portfolio theta

    Risk metrics:
        var_95: Portfolio 95% VaR (1-day)
        var_99: Portfolio 99% VaR (1-day)
        cvar_95: Conditional VaR (Expected Shortfall)
        correlation_risk: Correlation-adjusted risk
        concentration_risk: Herfindahl concentration index

    P&L:
        daily_pnl: Today's P&L
        total_pnl: Total unrealized P&L
    """

    total_value: float
    total_exposure: float
    net_exposure: float
    leverage: float = 0.0

    # Greeks
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0
    total_theta: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0

    # P&L
    daily_pnl: float = 0.0
    total_pnl: float = 0.0

    # Detailed breakdown
    position_risks: Dict[str, PositionRisk] = field(default_factory=dict)
    exposure_by_sector: Dict[str, float] = field(default_factory=dict)
    exposure_by_strategy: Dict[str, float] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_value": self.total_value,
            "total_exposure": self.total_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.leverage,
            "total_delta": self.total_delta,
            "total_gamma": self.total_gamma,
            "total_vega": self.total_vega,
            "total_theta": self.total_theta,
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "concentration_risk": self.concentration_risk,
            "daily_pnl": self.daily_pnl,
            "total_pnl": self.total_pnl,
            "exposure_by_sector": self.exposure_by_sector,
            "exposure_by_strategy": self.exposure_by_strategy,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskCheckResult:
    """Result of risk limit check."""

    is_allowed: bool
    breached_limits: List[str]
    warnings: List[str]
    recommended_action: str  # "proceed", "reduce", "halt"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "is_allowed": self.is_allowed,
            "breached_limits": self.breached_limits,
            "warnings": self.warnings,
            "recommended_action": self.recommended_action,
            "details": self.details,
        }


class RiskManager:
    """
    Core risk management system.

    Manages portfolio risk limits, position tracking, and risk monitoring.
    Integrates with DrawdownController, GreeksRiskMonitor, and CorrelationMonitor.

    Example:
        >>> risk_mgr = RiskManager(total_capital=1_000_000)
        >>>
        >>> # Add risk limits
        >>> risk_mgr.add_limit(RiskLimit(
        ...     limit_type=RiskLimitType.POSITION_SIZE,
        ...     value=0.10,  # 10% max per position
        ...     action_on_breach="reduce"
        ... ))
        >>> risk_mgr.add_limit(RiskLimit(
        ...     limit_type=RiskLimitType.DRAWDOWN,
        ...     value=0.25,  # 25% max drawdown
        ...     action_on_breach="halt"
        ... ))
        >>>
        >>> # Check if position is allowed
        >>> result = risk_mgr.check_position_allowed(
        ...     asset_id="SPY",
        ...     position_size=100,
        ...     current_price=450.0
        ... )
        >>>
        >>> if not result.is_allowed:
        ...     print(f"Position rejected: {result.breached_limits}")

    Reference:
        Standard risk management practices for systematic trading
    """

    def __init__(
        self,
        total_capital: float,
        risk_free_rate: float = 0.05,
        target_volatility: float = 0.15,
    ):
        """
        Initialize risk manager.

        Args:
            total_capital: Total available capital
            risk_free_rate: Risk-free rate for calculations
            target_volatility: Target portfolio volatility (annualized)
        """
        self.total_capital = total_capital
        self.risk_free_rate = risk_free_rate
        self.target_volatility = target_volatility

        # Risk limits
        self.limits: Dict[RiskLimitType, RiskLimit] = {}

        # Current state
        self.positions: Dict[str, PositionRisk] = {}
        self.portfolio_risk: Optional[PortfolioRisk] = None

        # Risk budget tracking
        self.risk_budget_total: float = 0.0
        self.risk_budget_used: float = 0.0

        # Circuit breaker state
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_reason: Optional[str] = None

        # Track peak value for drawdown
        self._peak_value: float = total_capital
        self._current_value: float = total_capital

        logger.info(
            f"Initialized RiskManager with capital=${total_capital:,.0f}, "
            f"target_vol={target_volatility:.1%}"
        )

    def add_limit(self, limit: RiskLimit) -> None:
        """
        Add or update a risk limit.

        Args:
            limit: RiskLimit to add
        """
        self.limits[limit.limit_type] = limit
        logger.info(f"Added risk limit: {limit.limit_type.value} = {limit.value}")

    def remove_limit(self, limit_type: RiskLimitType) -> None:
        """Remove a risk limit."""
        if limit_type in self.limits:
            del self.limits[limit_type]
            logger.info(f"Removed risk limit: {limit_type.value}")

    def set_default_limits(self) -> None:
        """Set default risk limits per design doc."""
        default_limits = [
            RiskLimit(
                limit_type=RiskLimitType.POSITION_SIZE,
                value=0.10,  # 10% max per position
                warning_threshold=0.8,
                action_on_breach="reduce",
            ),
            RiskLimit(
                limit_type=RiskLimitType.TOTAL_EXPOSURE,
                value=1.0,  # 100% max exposure (1x leverage)
                warning_threshold=0.8,
                action_on_breach="halt",
            ),
            RiskLimit(
                limit_type=RiskLimitType.DRAWDOWN,
                value=0.25,  # 25% max drawdown
                warning_threshold=0.8,
                action_on_breach="halt",
            ),
            RiskLimit(
                limit_type=RiskLimitType.DAILY_LOSS,
                value=0.03,  # 3% max daily loss
                warning_threshold=0.8,
                action_on_breach="halt",
            ),
            RiskLimit(
                limit_type=RiskLimitType.DELTA,
                value=50.0,  # Max portfolio delta
                warning_threshold=0.8,
                action_on_breach="reduce",
            ),
            RiskLimit(
                limit_type=RiskLimitType.VAR,
                value=0.02,  # 2% max daily VaR
                warning_threshold=0.8,
                action_on_breach="reduce",
            ),
        ]

        for limit in default_limits:
            self.add_limit(limit)

        logger.info("Set default risk limits")

    def check_position_allowed(
        self,
        asset_id: str,
        position_size: float,
        current_price: float,
        position_risk: Optional[PositionRisk] = None,
    ) -> RiskCheckResult:
        """
        Check if a new position is within risk limits.

        Args:
            asset_id: Asset identifier
            position_size: Proposed position size (shares/contracts)
            current_price: Current market price
            position_risk: Optional pre-computed position risk

        Returns:
            RiskCheckResult with allowed status and details
        """
        breached_limits = []
        warnings = []
        details = {}

        # Check circuit breaker first
        if self._circuit_breaker_active:
            return RiskCheckResult(
                is_allowed=False,
                breached_limits=["circuit_breaker_active"],
                warnings=[],
                recommended_action="halt",
                details={"reason": self._circuit_breaker_reason},
            )

        position_value = abs(position_size * current_price)
        position_pct = position_value / self.total_capital if self.total_capital > 0 else 0

        # Check position size limit
        if RiskLimitType.POSITION_SIZE in self.limits:
            limit = self.limits[RiskLimitType.POSITION_SIZE]
            is_breach, level = limit.check_breach(position_pct)

            details["position_size_pct"] = position_pct
            details["position_size_limit"] = limit.value

            if is_breach:
                breached_limits.append(
                    f"position_size: {position_pct:.2%} > {limit.value:.2%}"
                )
            elif level == "warning":
                warnings.append(
                    f"position_size near limit: {position_pct:.2%} / {limit.value:.2%}"
                )

        # Check total exposure limit
        if RiskLimitType.TOTAL_EXPOSURE in self.limits:
            limit = self.limits[RiskLimitType.TOTAL_EXPOSURE]
            current_exposure = sum(abs(p.market_value) for p in self.positions.values())
            new_exposure = current_exposure + position_value
            exposure_pct = new_exposure / self.total_capital if self.total_capital > 0 else 0

            details["total_exposure_pct"] = exposure_pct
            details["total_exposure_limit"] = limit.value

            is_breach, level = limit.check_breach(exposure_pct)

            if is_breach:
                breached_limits.append(
                    f"total_exposure: {exposure_pct:.2%} > {limit.value:.2%}"
                )
            elif level == "warning":
                warnings.append(
                    f"total_exposure near limit: {exposure_pct:.2%} / {limit.value:.2%}"
                )

        # Check Greeks limits if position_risk provided
        if position_risk is not None:
            # Check delta limit
            if RiskLimitType.DELTA in self.limits:
                limit = self.limits[RiskLimitType.DELTA]
                current_delta = sum(p.delta for p in self.positions.values())
                new_delta = current_delta + position_risk.delta

                is_breach, level = limit.check_breach(new_delta)

                details["new_portfolio_delta"] = new_delta
                details["delta_limit"] = limit.value

                if is_breach:
                    breached_limits.append(
                        f"delta: {new_delta:.1f} > {limit.value:.1f}"
                    )
                elif level == "warning":
                    warnings.append(
                        f"delta near limit: {new_delta:.1f} / {limit.value:.1f}"
                    )

        # Determine action
        if breached_limits:
            # Find most severe action
            action = "reduce"
            for limit_type in self.limits.values():
                if limit_type.action_on_breach == "halt":
                    action = "halt"
                    break

            return RiskCheckResult(
                is_allowed=False,
                breached_limits=breached_limits,
                warnings=warnings,
                recommended_action=action,
                details=details,
            )

        return RiskCheckResult(
            is_allowed=True,
            breached_limits=[],
            warnings=warnings,
            recommended_action="proceed",
            details=details,
        )

    def update_position(self, position: PositionRisk) -> None:
        """
        Update or add a position.

        Args:
            position: PositionRisk to update
        """
        self.positions[position.asset_id] = position
        logger.debug(f"Updated position: {position.asset_id}")

    def remove_position(self, asset_id: str) -> None:
        """Remove a position."""
        if asset_id in self.positions:
            del self.positions[asset_id]
            logger.debug(f"Removed position: {asset_id}")

    def compute_portfolio_risk(
        self,
        daily_pnl: float = 0.0,
    ) -> PortfolioRisk:
        """
        Compute current portfolio risk metrics.

        Args:
            daily_pnl: Today's realized + unrealized P&L

        Returns:
            PortfolioRisk with all metrics
        """
        if not self.positions:
            return PortfolioRisk(
                total_value=self.total_capital,
                total_exposure=0.0,
                net_exposure=0.0,
            )

        # Aggregate exposures
        total_exposure = sum(abs(p.market_value) for p in self.positions.values())
        net_exposure = sum(p.market_value for p in self.positions.values())
        leverage = total_exposure / self.total_capital if self.total_capital > 0 else 0

        # Aggregate Greeks
        total_delta = sum(p.delta for p in self.positions.values())
        total_gamma = sum(p.gamma for p in self.positions.values())
        total_vega = sum(p.vega for p in self.positions.values())
        total_theta = sum(p.theta for p in self.positions.values())

        # Total P&L
        total_pnl = sum(p.pnl for p in self.positions.values())

        # Concentration risk (Herfindahl index)
        concentration_risk = 0.0
        if total_exposure > 0:
            weights = [abs(p.market_value) / total_exposure for p in self.positions.values()]
            concentration_risk = sum(w ** 2 for w in weights)

        # Exposure by sector and strategy
        exposure_by_sector: Dict[str, float] = {}
        exposure_by_strategy: Dict[str, float] = {}

        for pos in self.positions.values():
            if pos.sector:
                exposure_by_sector[pos.sector] = (
                    exposure_by_sector.get(pos.sector, 0.0) + abs(pos.market_value)
                )
            if pos.strategy:
                exposure_by_strategy[pos.strategy] = (
                    exposure_by_strategy.get(pos.strategy, 0.0) + abs(pos.market_value)
                )

        # Update current value for drawdown tracking
        self._current_value = self.total_capital + total_pnl
        if self._current_value > self._peak_value:
            self._peak_value = self._current_value

        self.portfolio_risk = PortfolioRisk(
            total_value=self._current_value,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_vega=total_vega,
            total_theta=total_theta,
            concentration_risk=concentration_risk,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            position_risks=self.positions.copy(),
            exposure_by_sector=exposure_by_sector,
            exposure_by_strategy=exposure_by_strategy,
        )

        return self.portfolio_risk

    def check_all_limits(self, daily_pnl: float = 0.0) -> RiskCheckResult:
        """
        Check all portfolio-level risk limits.

        Args:
            daily_pnl: Today's P&L for daily loss check

        Returns:
            RiskCheckResult with status of all limits
        """
        portfolio = self.compute_portfolio_risk(daily_pnl)

        breached_limits = []
        warnings = []
        details = {}

        # Check circuit breaker
        if self._circuit_breaker_active:
            return RiskCheckResult(
                is_allowed=False,
                breached_limits=["circuit_breaker_active"],
                warnings=[],
                recommended_action="halt",
                details={"reason": self._circuit_breaker_reason},
            )

        # Check drawdown
        if RiskLimitType.DRAWDOWN in self.limits:
            limit = self.limits[RiskLimitType.DRAWDOWN]
            current_drawdown = 0.0
            if self._peak_value > 0:
                current_drawdown = (self._peak_value - self._current_value) / self._peak_value

            details["current_drawdown"] = current_drawdown
            details["drawdown_limit"] = limit.value

            is_breach, level = limit.check_breach(current_drawdown)

            if is_breach:
                breached_limits.append(
                    f"drawdown: {current_drawdown:.2%} > {limit.value:.2%}"
                )
            elif level == "warning":
                warnings.append(
                    f"drawdown near limit: {current_drawdown:.2%} / {limit.value:.2%}"
                )

        # Check daily loss
        if RiskLimitType.DAILY_LOSS in self.limits:
            limit = self.limits[RiskLimitType.DAILY_LOSS]
            daily_loss_pct = -daily_pnl / self.total_capital if self.total_capital > 0 else 0

            if daily_loss_pct > 0:  # Only check losses
                details["daily_loss_pct"] = daily_loss_pct
                details["daily_loss_limit"] = limit.value

                is_breach, level = limit.check_breach(daily_loss_pct)

                if is_breach:
                    breached_limits.append(
                        f"daily_loss: {daily_loss_pct:.2%} > {limit.value:.2%}"
                    )
                elif level == "warning":
                    warnings.append(
                        f"daily_loss near limit: {daily_loss_pct:.2%} / {limit.value:.2%}"
                    )

        # Check total exposure
        if RiskLimitType.TOTAL_EXPOSURE in self.limits:
            limit = self.limits[RiskLimitType.TOTAL_EXPOSURE]
            exposure_pct = portfolio.leverage

            details["total_exposure_pct"] = exposure_pct
            details["total_exposure_limit"] = limit.value

            is_breach, level = limit.check_breach(exposure_pct)

            if is_breach:
                breached_limits.append(
                    f"total_exposure: {exposure_pct:.2%} > {limit.value:.2%}"
                )
            elif level == "warning":
                warnings.append(
                    f"total_exposure near limit: {exposure_pct:.2%} / {limit.value:.2%}"
                )

        # Check delta
        if RiskLimitType.DELTA in self.limits:
            limit = self.limits[RiskLimitType.DELTA]

            details["portfolio_delta"] = portfolio.total_delta
            details["delta_limit"] = limit.value

            is_breach, level = limit.check_breach(portfolio.total_delta)

            if is_breach:
                breached_limits.append(
                    f"delta: {portfolio.total_delta:.1f} > {limit.value:.1f}"
                )
            elif level == "warning":
                warnings.append(
                    f"delta near limit: {portfolio.total_delta:.1f} / {limit.value:.1f}"
                )

        # Determine action
        is_allowed = len(breached_limits) == 0
        action = "proceed" if is_allowed else "reduce"

        for limit_type, limit in self.limits.items():
            if any(limit_type.value in b for b in breached_limits):
                if limit.action_on_breach == "halt":
                    action = "halt"
                    break

        return RiskCheckResult(
            is_allowed=is_allowed,
            breached_limits=breached_limits,
            warnings=warnings,
            recommended_action=action,
            details=details,
        )

    def activate_circuit_breaker(self, reason: str) -> None:
        """
        Activate circuit breaker to halt all trading.

        Args:
            reason: Reason for activation
        """
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")

    def deactivate_circuit_breaker(self) -> None:
        """Deactivate circuit breaker (requires manual intervention)."""
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = None
        logger.info("Circuit breaker deactivated")

    @property
    def circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active."""
        return self._circuit_breaker_active

    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of all positions.

        Returns:
            Dict with position summary
        """
        if not self.positions:
            return {
                "num_positions": 0,
                "total_exposure": 0.0,
                "positions": [],
            }

        return {
            "num_positions": len(self.positions),
            "total_exposure": sum(abs(p.market_value) for p in self.positions.values()),
            "net_exposure": sum(p.market_value for p in self.positions.values()),
            "total_pnl": sum(p.pnl for p in self.positions.values()),
            "positions": [p.to_dict() for p in self.positions.values()],
        }

    def get_limit_status(self) -> Dict[str, Any]:
        """
        Get status of all risk limits.

        Returns:
            Dict with limit status
        """
        portfolio = self.compute_portfolio_risk()

        status = {}
        for limit_type, limit in self.limits.items():
            current_value = 0.0

            if limit_type == RiskLimitType.POSITION_SIZE:
                # Max position in portfolio
                if self.positions and self.total_capital > 0:
                    max_pos = max(abs(p.market_value) for p in self.positions.values())
                    current_value = max_pos / self.total_capital

            elif limit_type == RiskLimitType.TOTAL_EXPOSURE:
                current_value = portfolio.leverage

            elif limit_type == RiskLimitType.DRAWDOWN:
                if self._peak_value > 0:
                    current_value = (self._peak_value - self._current_value) / self._peak_value

            elif limit_type == RiskLimitType.DELTA:
                current_value = portfolio.total_delta

            is_breach, level = limit.check_breach(current_value)

            status[limit_type.value] = {
                "limit": limit.value,
                "current": current_value,
                "utilization": abs(current_value / limit.value) if limit.value != 0 else 0,
                "status": level,
                "action_on_breach": limit.action_on_breach,
            }

        return status

    def reset(self, new_capital: Optional[float] = None) -> None:
        """
        Reset risk manager state.

        Args:
            new_capital: New capital amount (optional)
        """
        if new_capital is not None:
            self.total_capital = new_capital

        self.positions = {}
        self.portfolio_risk = None
        self._peak_value = self.total_capital
        self._current_value = self.total_capital
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = None

        logger.info(f"RiskManager reset with capital=${self.total_capital:,.0f}")
