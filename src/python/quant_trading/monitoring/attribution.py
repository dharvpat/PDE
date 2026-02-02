"""
Performance Attribution Module for Quantitative Trading System.

Provides comprehensive performance attribution analysis including:
- Returns decomposition (alpha, beta, factor exposures)
- Risk attribution (VaR contribution, marginal risk)
- Trade-level attribution (signal quality, execution quality)
- Brinson attribution (allocation, selection, interaction)
- Factor-based attribution (Fama-French, custom factors)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AttributionPeriod(Enum):
    """Time periods for attribution analysis."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    CUSTOM = "custom"


@dataclass
class ReturnDecomposition:
    """Decomposition of portfolio returns."""

    total_return: float
    alpha: float
    beta_contribution: float
    factor_contributions: Dict[str, float]
    residual: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "alpha": self.alpha,
            "beta_contribution": self.beta_contribution,
            "factor_contributions": self.factor_contributions,
            "residual": self.residual,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskAttribution:
    """Attribution of portfolio risk."""

    total_var: float
    total_volatility: float
    position_contributions: Dict[str, float]
    factor_contributions: Dict[str, float]
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_var": self.total_var,
            "total_volatility": self.total_volatility,
            "position_contributions": self.position_contributions,
            "factor_contributions": self.factor_contributions,
            "marginal_var": self.marginal_var,
            "component_var": self.component_var,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradeAttribution:
    """Attribution for individual trade."""

    trade_id: str
    symbol: str
    side: str
    pnl: float
    signal_contribution: float
    timing_contribution: float
    execution_contribution: float
    slippage: float
    signal_quality: float  # 0-1 score
    execution_quality: float  # 0-1 score
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side,
            "pnl": self.pnl,
            "signal_contribution": self.signal_contribution,
            "timing_contribution": self.timing_contribution,
            "execution_contribution": self.execution_contribution,
            "slippage": self.slippage,
            "signal_quality": self.signal_quality,
            "execution_quality": self.execution_quality,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BrinsonAttribution:
    """Brinson-Fachler attribution analysis."""

    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    sector_allocation: Dict[str, float]
    sector_selection: Dict[str, float]
    sector_interaction: Dict[str, float]
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allocation_effect": self.allocation_effect,
            "selection_effect": self.selection_effect,
            "interaction_effect": self.interaction_effect,
            "total_active_return": self.total_active_return,
            "sector_allocation": self.sector_allocation,
            "sector_selection": self.sector_selection,
            "sector_interaction": self.sector_interaction,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


@dataclass
class FactorExposure:
    """Factor exposure analysis."""

    factor_name: str
    exposure: float
    t_statistic: float
    p_value: float
    contribution: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor_name": self.factor_name,
            "exposure": self.exposure,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "contribution": self.contribution,
        }


@dataclass
class PerformanceReport:
    """Complete performance attribution report."""

    period_start: datetime
    period_end: datetime
    return_decomposition: ReturnDecomposition
    risk_attribution: RiskAttribution
    brinson_attribution: Optional[BrinsonAttribution]
    factor_exposures: List[FactorExposure]
    trade_attributions: List[TradeAttribution]
    summary_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "return_decomposition": self.return_decomposition.to_dict(),
            "risk_attribution": self.risk_attribution.to_dict(),
            "brinson_attribution": self.brinson_attribution.to_dict() if self.brinson_attribution else None,
            "factor_exposures": [f.to_dict() for f in self.factor_exposures],
            "trade_attributions": [t.to_dict() for t in self.trade_attributions],
            "summary_metrics": self.summary_metrics,
        }


class ReturnsAttributor:
    """Calculate return decomposition and alpha/beta analysis."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        benchmark_symbol: str = "SPY",
    ):
        self.risk_free_rate = risk_free_rate
        self.benchmark_symbol = benchmark_symbol

    def decompose_returns(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        factor_returns: Optional[Dict[str, np.ndarray]] = None,
    ) -> ReturnDecomposition:
        """
        Decompose portfolio returns into alpha, beta, and factor contributions.

        Uses OLS regression: R_p - R_f = alpha + beta * (R_m - R_f) + factor_betas + epsilon
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return ReturnDecomposition(
                total_return=0.0,
                alpha=0.0,
                beta_contribution=0.0,
                factor_contributions={},
                residual=0.0,
            )

        # Convert to excess returns
        n_periods = len(portfolio_returns)
        daily_rf = self.risk_free_rate / 252
        excess_portfolio = portfolio_returns - daily_rf
        excess_benchmark = benchmark_returns - daily_rf

        # Calculate total return
        total_return = float(np.prod(1 + portfolio_returns) - 1)

        # Simple CAPM regression
        if len(excess_benchmark) > 1:
            cov = np.cov(excess_portfolio, excess_benchmark)
            if cov.shape == (2, 2) and cov[1, 1] > 0:
                beta = cov[0, 1] / cov[1, 1]
            else:
                beta = 1.0
        else:
            beta = 1.0

        # Calculate alpha
        mean_excess_portfolio = float(np.mean(excess_portfolio))
        mean_excess_benchmark = float(np.mean(excess_benchmark))
        alpha = (mean_excess_portfolio - beta * mean_excess_benchmark) * 252  # Annualized

        # Beta contribution
        beta_contribution = float(beta * mean_excess_benchmark * n_periods)

        # Factor contributions
        factor_contributions = {}
        residual_returns = excess_portfolio.copy()

        if factor_returns:
            for factor_name, factor_ret in factor_returns.items():
                if len(factor_ret) == len(portfolio_returns) and np.var(factor_ret) > 0:
                    # Calculate factor beta
                    factor_cov = np.cov(residual_returns, factor_ret)
                    if factor_cov.shape == (2, 2) and factor_cov[1, 1] > 0:
                        factor_beta = factor_cov[0, 1] / factor_cov[1, 1]
                        contribution = float(factor_beta * np.mean(factor_ret) * n_periods)
                        factor_contributions[factor_name] = contribution
                        residual_returns = residual_returns - factor_beta * factor_ret

        # Residual (unexplained returns)
        residual = float(np.sum(residual_returns) - alpha * n_periods / 252 - beta_contribution)

        return ReturnDecomposition(
            total_return=total_return,
            alpha=alpha,
            beta_contribution=beta_contribution,
            factor_contributions=factor_contributions,
            residual=residual,
        )

    def calculate_information_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Calculate information ratio (excess return / tracking error)."""
        if len(portfolio_returns) == 0:
            return 0.0

        active_returns = portfolio_returns - benchmark_returns
        if np.std(active_returns) == 0:
            return 0.0

        return float(np.mean(active_returns) / np.std(active_returns) * np.sqrt(252))


class RiskAttributor:
    """Calculate risk attribution metrics."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None,
    ) -> float:
        """Calculate historical VaR."""
        if len(returns) == 0:
            return 0.0

        level = confidence_level or self.confidence_level
        return float(np.percentile(returns, (1 - level) * 100))

    def calculate_component_var(
        self,
        position_returns: Dict[str, np.ndarray],
        position_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate component VaR for each position.

        Component VaR = weight_i * marginal_VaR_i
        Sum of component VaRs = total VaR
        """
        if not position_returns:
            return {}

        # Calculate portfolio returns
        symbols = list(position_returns.keys())
        weights = np.array([position_weights.get(s, 0) for s in symbols])
        returns_matrix = np.column_stack([position_returns[s] for s in symbols])

        portfolio_returns = returns_matrix @ weights
        portfolio_var = self.calculate_var(portfolio_returns)

        # Calculate component VaR using marginal contribution
        component_var = {}
        portfolio_std = np.std(portfolio_returns)

        for i, symbol in enumerate(symbols):
            if portfolio_std > 0:
                # Marginal VaR approximation
                cov_with_portfolio = np.cov(returns_matrix[:, i], portfolio_returns)[0, 1]
                marginal_var = cov_with_portfolio / portfolio_std * self._var_multiplier()
                component_var[symbol] = float(weights[i] * marginal_var)
            else:
                component_var[symbol] = 0.0

        return component_var

    def _var_multiplier(self) -> float:
        """Get VaR multiplier for given confidence level (normal approximation)."""
        from scipy import stats
        return stats.norm.ppf(1 - self.confidence_level)

    def calculate_marginal_var(
        self,
        position_returns: Dict[str, np.ndarray],
        position_weights: Dict[str, float],
        delta_weight: float = 0.01,
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each position.

        Marginal VaR = change in VaR for a small change in position weight.
        """
        if not position_returns:
            return {}

        symbols = list(position_returns.keys())
        weights = np.array([position_weights.get(s, 0) for s in symbols])
        returns_matrix = np.column_stack([position_returns[s] for s in symbols])

        # Base portfolio VaR
        portfolio_returns = returns_matrix @ weights
        base_var = self.calculate_var(portfolio_returns)

        # Calculate marginal VaR for each position
        marginal_var = {}
        for i, symbol in enumerate(symbols):
            # Increase weight by delta
            weights_up = weights.copy()
            weights_up[i] += delta_weight
            weights_up = weights_up / np.sum(weights_up)  # Renormalize

            portfolio_returns_up = returns_matrix @ weights_up
            var_up = self.calculate_var(portfolio_returns_up)

            marginal_var[symbol] = float((var_up - base_var) / delta_weight)

        return marginal_var

    def attribute_risk(
        self,
        position_returns: Dict[str, np.ndarray],
        position_weights: Dict[str, float],
        factor_exposures: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> RiskAttribution:
        """
        Calculate complete risk attribution.

        Args:
            position_returns: Dict of symbol -> array of returns
            position_weights: Dict of symbol -> weight in portfolio
            factor_exposures: Optional dict of symbol -> {factor: exposure}
        """
        if not position_returns:
            return RiskAttribution(
                total_var=0.0,
                total_volatility=0.0,
                position_contributions={},
                factor_contributions={},
                marginal_var={},
                component_var={},
            )

        # Calculate portfolio returns
        symbols = list(position_returns.keys())
        weights = np.array([position_weights.get(s, 0) for s in symbols])
        returns_matrix = np.column_stack([position_returns[s] for s in symbols])
        portfolio_returns = returns_matrix @ weights

        # Total metrics
        total_var = self.calculate_var(portfolio_returns)
        total_volatility = float(np.std(portfolio_returns) * np.sqrt(252))

        # Position contributions to volatility
        position_contributions = {}
        portfolio_std = np.std(portfolio_returns)
        if portfolio_std > 0:
            for i, symbol in enumerate(symbols):
                cov_with_portfolio = np.cov(returns_matrix[:, i], portfolio_returns)[0, 1]
                contribution = weights[i] * cov_with_portfolio / portfolio_std
                position_contributions[symbol] = float(contribution * np.sqrt(252))

        # Component and marginal VaR
        component_var = self.calculate_component_var(position_returns, position_weights)
        marginal_var = self.calculate_marginal_var(position_returns, position_weights)

        # Factor contributions (simplified)
        factor_contributions = {}
        if factor_exposures:
            factors = set()
            for symbol_factors in factor_exposures.values():
                factors.update(symbol_factors.keys())

            for factor in factors:
                factor_contribution = 0.0
                for symbol in symbols:
                    if symbol in factor_exposures and factor in factor_exposures[symbol]:
                        weight = position_weights.get(symbol, 0)
                        exposure = factor_exposures[symbol][factor]
                        factor_contribution += weight * exposure
                factor_contributions[factor] = factor_contribution

        return RiskAttribution(
            total_var=total_var,
            total_volatility=total_volatility,
            position_contributions=position_contributions,
            factor_contributions=factor_contributions,
            marginal_var=marginal_var,
            component_var=component_var,
        )


class BrinsonAttributor:
    """Brinson-Fachler attribution analysis."""

    def calculate_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float],
        sector_mapping: Dict[str, str],
        period_start: datetime,
        period_end: datetime,
    ) -> BrinsonAttribution:
        """
        Calculate Brinson-Fachler attribution.

        Decomposes active return into:
        - Allocation effect: (w_p - w_b) * R_b
        - Selection effect: w_b * (R_p - R_b)
        - Interaction effect: (w_p - w_b) * (R_p - R_b)
        """
        # Get all sectors
        sectors = set(sector_mapping.values())

        # Aggregate by sector
        sector_portfolio_weights: Dict[str, float] = {}
        sector_benchmark_weights: Dict[str, float] = {}
        sector_portfolio_returns: Dict[str, float] = {}
        sector_benchmark_returns: Dict[str, float] = {}

        for sector in sectors:
            # Get symbols in this sector
            sector_symbols = [s for s, sec in sector_mapping.items() if sec == sector]

            # Portfolio sector weight and return
            p_weight = sum(portfolio_weights.get(s, 0) for s in sector_symbols)
            if p_weight > 0:
                p_return = sum(
                    portfolio_weights.get(s, 0) * portfolio_returns.get(s, 0)
                    for s in sector_symbols
                ) / p_weight
            else:
                p_return = 0.0

            # Benchmark sector weight and return
            b_weight = sum(benchmark_weights.get(s, 0) for s in sector_symbols)
            if b_weight > 0:
                b_return = sum(
                    benchmark_weights.get(s, 0) * benchmark_returns.get(s, 0)
                    for s in sector_symbols
                ) / b_weight
            else:
                b_return = 0.0

            sector_portfolio_weights[sector] = p_weight
            sector_benchmark_weights[sector] = b_weight
            sector_portfolio_returns[sector] = p_return
            sector_benchmark_returns[sector] = b_return

        # Calculate total benchmark return
        total_benchmark_return = sum(
            sector_benchmark_weights[s] * sector_benchmark_returns[s]
            for s in sectors
        )

        # Calculate attribution effects by sector
        sector_allocation = {}
        sector_selection = {}
        sector_interaction = {}

        for sector in sectors:
            w_p = sector_portfolio_weights[sector]
            w_b = sector_benchmark_weights[sector]
            r_p = sector_portfolio_returns[sector]
            r_b = sector_benchmark_returns[sector]

            # Allocation effect
            sector_allocation[sector] = (w_p - w_b) * (r_b - total_benchmark_return)

            # Selection effect
            sector_selection[sector] = w_b * (r_p - r_b)

            # Interaction effect
            sector_interaction[sector] = (w_p - w_b) * (r_p - r_b)

        # Total effects
        allocation_effect = sum(sector_allocation.values())
        selection_effect = sum(sector_selection.values())
        interaction_effect = sum(sector_interaction.values())
        total_active_return = allocation_effect + selection_effect + interaction_effect

        return BrinsonAttribution(
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_active_return=total_active_return,
            sector_allocation=sector_allocation,
            sector_selection=sector_selection,
            sector_interaction=sector_interaction,
            period_start=period_start,
            period_end=period_end,
        )


class TradeAttributor:
    """Attribution analysis at the trade level."""

    def attribute_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        signal_price: float,
        optimal_entry_price: float,
        optimal_exit_price: float,
        signal_strength: float,
    ) -> TradeAttribution:
        """
        Attribute trade P&L to signal quality, timing, and execution.

        Args:
            trade_id: Trade identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Actual entry price
            exit_price: Actual exit price
            quantity: Trade quantity
            signal_price: Price when signal was generated
            optimal_entry_price: Best available entry price
            optimal_exit_price: Best available exit price
            signal_strength: Signal confidence (0-1)
        """
        # Calculate actual P&L
        if side.lower() == 'buy':
            pnl = (exit_price - entry_price) * quantity
            optimal_pnl = (optimal_exit_price - optimal_entry_price) * quantity
            signal_pnl = (exit_price - signal_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
            optimal_pnl = (optimal_entry_price - optimal_exit_price) * quantity
            signal_pnl = (signal_price - exit_price) * quantity

        # Signal contribution: P&L if executed at signal price
        signal_contribution = signal_pnl

        # Timing contribution: difference between signal and actual entry
        if side.lower() == 'buy':
            timing_contribution = (signal_price - entry_price) * quantity
        else:
            timing_contribution = (entry_price - signal_price) * quantity

        # Execution contribution: difference between optimal and actual
        if side.lower() == 'buy':
            entry_slippage = (entry_price - optimal_entry_price) * quantity
            exit_slippage = (optimal_exit_price - exit_price) * quantity
        else:
            entry_slippage = (optimal_entry_price - entry_price) * quantity
            exit_slippage = (exit_price - optimal_exit_price) * quantity

        execution_contribution = -(entry_slippage + exit_slippage)
        slippage = entry_slippage + exit_slippage

        # Quality scores
        signal_quality = signal_strength
        if optimal_pnl != 0:
            execution_quality = max(0, min(1, pnl / optimal_pnl))
        else:
            execution_quality = 1.0 if pnl >= 0 else 0.0

        return TradeAttribution(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            pnl=pnl,
            signal_contribution=signal_contribution,
            timing_contribution=timing_contribution,
            execution_contribution=execution_contribution,
            slippage=slippage,
            signal_quality=signal_quality,
            execution_quality=execution_quality,
        )


class FactorAttributor:
    """Factor-based attribution using multi-factor models."""

    # Standard Fama-French factors
    FAMA_FRENCH_3 = ["Mkt-RF", "SMB", "HML"]
    FAMA_FRENCH_5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    def __init__(self, factors: Optional[List[str]] = None):
        self.factors = factors or self.FAMA_FRENCH_3

    def calculate_factor_exposures(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: Dict[str, np.ndarray],
    ) -> List[FactorExposure]:
        """
        Calculate factor exposures using OLS regression.

        R_p = alpha + sum(beta_i * F_i) + epsilon
        """
        if len(portfolio_returns) < 20:
            return []

        # Build factor matrix
        factor_names = [f for f in self.factors if f in factor_returns]
        if not factor_names:
            return []

        n_obs = len(portfolio_returns)
        factor_matrix = np.column_stack([factor_returns[f][:n_obs] for f in factor_names])

        # Add constant for alpha
        X = np.column_stack([np.ones(n_obs), factor_matrix])
        y = portfolio_returns[:n_obs]

        # OLS estimation
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            betas = XtX_inv @ X.T @ y

            # Residuals and standard errors
            residuals = y - X @ betas
            sigma2 = np.sum(residuals**2) / (n_obs - len(betas))
            se = np.sqrt(np.diag(sigma2 * XtX_inv))

            # Factor exposures (skip alpha at index 0)
            exposures = []
            for i, factor_name in enumerate(factor_names):
                beta = betas[i + 1]
                se_beta = se[i + 1]
                t_stat = beta / se_beta if se_beta > 0 else 0

                # P-value from t-distribution
                from scipy import stats
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - len(betas)))

                # Contribution = beta * mean(factor)
                contribution = float(beta * np.mean(factor_returns[factor_name][:n_obs]) * 252)

                exposures.append(FactorExposure(
                    factor_name=factor_name,
                    exposure=float(beta),
                    t_statistic=float(t_stat),
                    p_value=float(p_value),
                    contribution=contribution,
                ))

            return exposures

        except np.linalg.LinAlgError:
            return []


class PerformanceAttributionEngine:
    """Main engine for complete performance attribution."""

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        benchmark_symbol: str = "SPY",
        var_confidence: float = 0.95,
    ):
        self.returns_attributor = ReturnsAttributor(risk_free_rate, benchmark_symbol)
        self.risk_attributor = RiskAttributor(var_confidence)
        self.brinson_attributor = BrinsonAttributor()
        self.trade_attributor = TradeAttributor()
        self.factor_attributor = FactorAttributor()

    def generate_report(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        position_returns: Dict[str, np.ndarray],
        position_weights: Dict[str, float],
        factor_returns: Optional[Dict[str, np.ndarray]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> PerformanceReport:
        """Generate complete performance attribution report."""
        period_start = period_start or datetime.now() - timedelta(days=30)
        period_end = period_end or datetime.now()

        # Return decomposition
        return_decomposition = self.returns_attributor.decompose_returns(
            portfolio_returns,
            benchmark_returns,
            factor_returns,
        )

        # Risk attribution
        risk_attribution = self.risk_attributor.attribute_risk(
            position_returns,
            position_weights,
        )

        # Brinson attribution (if benchmark data available)
        brinson_attribution = None
        if benchmark_weights and sector_mapping:
            # Calculate period returns for each position
            period_portfolio_returns = {
                s: float(np.prod(1 + r) - 1)
                for s, r in position_returns.items()
            }
            period_benchmark_returns = {
                s: float(np.prod(1 + benchmark_returns) - 1)
                for s in position_returns.keys()
            }

            brinson_attribution = self.brinson_attributor.calculate_attribution(
                position_weights,
                benchmark_weights,
                period_portfolio_returns,
                period_benchmark_returns,
                sector_mapping,
                period_start,
                period_end,
            )

        # Factor exposures
        factor_exposures = []
        if factor_returns:
            factor_exposures = self.factor_attributor.calculate_factor_exposures(
                portfolio_returns,
                factor_returns,
            )

        # Trade attributions
        trade_attributions = []
        if trades:
            for trade in trades:
                attribution = self.trade_attributor.attribute_trade(
                    trade_id=trade.get('trade_id', ''),
                    symbol=trade.get('symbol', ''),
                    side=trade.get('side', 'buy'),
                    entry_price=trade.get('entry_price', 0),
                    exit_price=trade.get('exit_price', 0),
                    quantity=trade.get('quantity', 0),
                    signal_price=trade.get('signal_price', trade.get('entry_price', 0)),
                    optimal_entry_price=trade.get('optimal_entry_price', trade.get('entry_price', 0)),
                    optimal_exit_price=trade.get('optimal_exit_price', trade.get('exit_price', 0)),
                    signal_strength=trade.get('signal_strength', 0.5),
                )
                trade_attributions.append(attribution)

        # Summary metrics
        summary_metrics = self._calculate_summary_metrics(
            portfolio_returns,
            benchmark_returns,
            return_decomposition,
            risk_attribution,
        )

        return PerformanceReport(
            period_start=period_start,
            period_end=period_end,
            return_decomposition=return_decomposition,
            risk_attribution=risk_attribution,
            brinson_attribution=brinson_attribution,
            factor_exposures=factor_exposures,
            trade_attributions=trade_attributions,
            summary_metrics=summary_metrics,
        )

    def _calculate_summary_metrics(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        return_decomposition: ReturnDecomposition,
        risk_attribution: RiskAttribution,
    ) -> Dict[str, float]:
        """Calculate summary performance metrics."""
        if len(portfolio_returns) == 0:
            return {}

        # Sharpe ratio
        daily_rf = 0.02 / 252
        excess_returns = portfolio_returns - daily_rf
        sharpe = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) > 0 else 0

        # Information ratio
        ir = self.returns_attributor.calculate_information_ratio(
            portfolio_returns,
            benchmark_returns,
        )

        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = float(np.mean(excess_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Calmar ratio
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_drawdown = float(np.max(drawdowns))
        annual_return = float(np.prod(1 + portfolio_returns) ** (252 / len(portfolio_returns)) - 1)
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        return {
            "sharpe_ratio": sharpe,
            "information_ratio": ir,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "alpha_annualized": return_decomposition.alpha,
            "total_return": return_decomposition.total_return,
            "volatility_annualized": risk_attribution.total_volatility,
            "var_95": risk_attribution.total_var,
            "max_drawdown": max_drawdown,
        }
