"""
Risk Management Module.

Provides comprehensive risk management tools for the quantitative trading system:

- **Risk Framework**: Unified RiskManager with limit enforcement
- **Position Sizing**: Volatility-scaled sizing (Moreira & Muir 2017) and Kelly criterion
- **Volatility Estimation**: EWMA, GARCH, and hybrid methods
- **VaR/CVaR**: Parametric, historical, and Monte Carlo Value at Risk
- **Stress Testing**: Historical crisis scenarios and custom stress tests
- **Greeks Monitoring**: Portfolio Greeks aggregation and delta-hedging
- **Correlation Monitoring**: Cointegration testing and correlation breakdown detection
- **Drawdown Control**: Multi-tier risk limits and kill switch capability

Target: Max drawdown <25% with Sharpe ratio 0.5-1.2

Example:
    >>> from quant_trading.risk import (
    ...     RiskManager,
    ...     RiskLimit,
    ...     RiskLimitType,
    ...     VolatilityScaledPositionSizer,
    ...     VaRCalculator,
    ...     StressTester,
    ...     GreeksRiskMonitor,
    ...     CorrelationMonitor,
    ...     DrawdownController
    ... )
    >>>
    >>> # Risk manager with limits
    >>> risk_mgr = RiskManager(total_capital=1_000_000)
    >>> risk_mgr.set_default_limits()
    >>> result = risk_mgr.check_position_allowed("SPY", 100, 450.0)
    >>>
    >>> # Position sizing
    >>> sizer = VolatilityScaledPositionSizer()
    >>> size = sizer.compute_position_size(returns, capital=1_000_000)
    >>>
    >>> # VaR calculation
    >>> var_calc = VaRCalculator()
    >>> var_result = var_calc.calculate(positions, returns)
    >>> print(f"95% VaR: ${var_result.var_95:,.0f}")
    >>>
    >>> # Stress testing
    >>> stress = StressTester()
    >>> worst = stress.get_worst_case(portfolio)

References:
    - Moreira & Muir (2017) "Volatility-managed portfolios"
    - Engle & Granger (1987) "Co-integration and error correction"
    - Jorion (2007) "Value at Risk"
    - RiskMetrics Technical Document (1996)
"""

from .correlation_monitor import (
    CointegrationResult,
    CorrelationHealth,
    CorrelationMonitor,
    CorrelationMonitorConfig,
    HealthStatus,
)
from .drawdown_controller import (
    DrawdownController,
    DrawdownControllerConfig,
    DrawdownMetrics,
    RiskAction,
    RiskLevel,
    RiskLimitStatus,
)
from .greeks_monitor import (
    GreeksMonitorConfig,
    GreeksRiskMonitor,
    HedgeAction,
    HedgeActionType,
    OptionPosition,
    PortfolioGreeks,
)
from .position_sizer import (
    KellyPositionSizer,
    PositionSizeResult,
    PositionSizerConfig,
    VolatilityEstimator,
    VolatilityMethod,
    VolatilityScaledPositionSizer,
)
from .risk_manager import (
    PortfolioRisk as UnifiedPortfolioRisk,
    PositionRisk,
    RiskCheckResult,
    RiskLimit,
    RiskLimitType,
    RiskManager,
)
from .var_calculator import (
    StressTestResult,
    StressTester,
    VaRBacktester,
    VaRCalculator,
    VaRMethod,
    VaRResult,
)

__all__ = [
    # Risk Manager (unified framework)
    "RiskManager",
    "RiskLimit",
    "RiskLimitType",
    "PositionRisk",
    "UnifiedPortfolioRisk",
    "RiskCheckResult",
    # Position sizing
    "VolatilityScaledPositionSizer",
    "KellyPositionSizer",
    "PositionSizeResult",
    "PositionSizerConfig",
    # Volatility estimation
    "VolatilityEstimator",
    "VolatilityMethod",
    # VaR and stress testing
    "VaRCalculator",
    "VaRMethod",
    "VaRResult",
    "VaRBacktester",
    "StressTester",
    "StressTestResult",
    # Greeks monitoring
    "GreeksRiskMonitor",
    "GreeksMonitorConfig",
    "OptionPosition",
    "PortfolioGreeks",
    "HedgeAction",
    "HedgeActionType",
    # Correlation monitoring
    "CorrelationMonitor",
    "CorrelationMonitorConfig",
    "CorrelationHealth",
    "CointegrationResult",
    "HealthStatus",
    # Drawdown control
    "DrawdownController",
    "DrawdownControllerConfig",
    "DrawdownMetrics",
    "RiskLimitStatus",
    "RiskLevel",
    "RiskAction",
]
