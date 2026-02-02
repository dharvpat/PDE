"""
Trading System Orchestrator

Main entry point that coordinates all components:
- Model calibration
- Signal generation
- Risk management
- Order execution
- Backtesting
"""

import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from queue import Queue

import numpy as np
import pandas as pd

from .config import Config, load_config, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal from signal generators."""
    symbol: str
    direction: str  # "long", "short", "close"
    strength: float  # 0 to 1
    source: str  # Signal source name
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price


class TradingSystem:
    """
    Main trading system that orchestrates all components.

    Usage:
        system = TradingSystem(config)
        system.initialize()

        # For backtesting
        results = system.run_backtest(start_date, end_date)

        # For live trading (paper or real)
        system.start_trading()
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize trading system with configuration."""
        self.config = config or load_config()
        setup_logging(self.config.logging)

        self.positions: Dict[str, Position] = {}
        self.cash: float = self.config.trading.initial_capital
        self.equity_history: List[tuple] = []
        self.trade_history: List[Dict] = []

        # Components (initialized lazily)
        self._calibrators = {}
        self._signal_generators = {}
        self._risk_manager = None
        self._execution_handler = None
        self._data_handler = None

        self._initialized = False
        logger.info(f"TradingSystem created with {self.config.trading.initial_capital:.2f} capital")

    def initialize(self) -> None:
        """Initialize all system components."""
        if self._initialized:
            return

        logger.info("Initializing trading system components...")

        # Initialize calibrators
        self._init_calibrators()

        # Initialize signal generators
        self._init_signal_generators()

        # Initialize risk manager
        self._init_risk_manager()

        self._initialized = True
        logger.info("Trading system initialized successfully")

    def _init_calibrators(self) -> None:
        """Initialize model calibrators."""
        try:
            from .calibration import HestonCalibrator, SABRCalibrator, OUFitter

            self._calibrators["heston"] = HestonCalibrator()
            self._calibrators["sabr"] = SABRCalibrator()
            self._calibrators["ou"] = OUFitter()
            logger.info("Calibrators initialized: heston, sabr, ou")
        except ImportError as e:
            logger.warning(f"Could not initialize calibrators: {e}")

    def _init_signal_generators(self) -> None:
        """Initialize signal generators."""
        try:
            from .signals import (
                VolSurfaceArbitrageSignal,
                MeanReversionSignalGenerator,
                SignalAggregator
            )

            self._signal_generators["vol_arb"] = VolSurfaceArbitrageSignal()
            self._signal_generators["mean_rev"] = MeanReversionSignalGenerator()
            self._signal_aggregator = SignalAggregator()
            logger.info("Signal generators initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize signal generators: {e}")

    def _init_risk_manager(self) -> None:
        """Initialize risk manager."""
        try:
            from .risk import RiskManager, VolatilityScaledPositionSizer

            self._risk_manager = RiskManager(
                total_capital=self.config.trading.initial_capital
            )
            self._position_sizer = VolatilityScaledPositionSizer()
            logger.info("Risk manager initialized")
        except ImportError as e:
            logger.warning(f"Could not initialize risk manager: {e}")

    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def total_return(self) -> float:
        """Total return as decimal."""
        initial = self.config.trading.initial_capital
        return (self.equity - initial) / initial

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

        # Record equity
        self.equity_history.append((datetime.now(), self.equity))

    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        signals = []

        for name, generator in self._signal_generators.items():
            try:
                signal = generator.generate(market_data)
                if signal:
                    signals.append(TradingSignal(
                        symbol=signal.get("symbol", ""),
                        direction=signal.get("direction", ""),
                        strength=signal.get("strength", 0.0),
                        source=name,
                        timestamp=datetime.now(),
                        metadata=signal.get("metadata", {})
                    ))
            except Exception as e:
                logger.error(f"Error generating signal from {name}: {e}")

        return signals

    def process_signal(self, signal: TradingSignal) -> Optional[Dict]:
        """Process a trading signal and generate order if appropriate."""
        # Check signal confidence
        if signal.strength < self.config.trading.min_signal_confidence:
            logger.debug(f"Signal strength {signal.strength:.2f} below threshold")
            return None

        # Check risk limits
        if self._risk_manager and not self._risk_manager.check_limits(self):
            logger.warning("Risk limits breached, not processing signal")
            return None

        # Calculate position size
        if self._position_sizer:
            size = self._position_sizer.calculate_size(
                signal=signal,
                equity=self.equity,
                current_positions=self.positions
            )
        else:
            # Default sizing: equal weight
            size = self.equity * self.config.trading.max_position_pct

        if size <= 0:
            return None

        order = {
            "symbol": signal.symbol,
            "direction": signal.direction,
            "quantity": size,
            "signal_strength": signal.strength,
            "timestamp": datetime.now()
        }

        logger.info(f"Generated order: {order}")
        return order

    def execute_order(self, order: Dict) -> bool:
        """Execute an order (simulated for backtesting)."""
        symbol = order["symbol"]
        direction = order["direction"]
        quantity = order["quantity"]

        # Simulate execution with slippage
        slippage = self.config.trading.slippage_bps / 10000
        commission = self.config.trading.commission_per_share * abs(quantity)

        # Get current price (would come from market data in live trading)
        price = self.positions.get(symbol, Position(symbol, 0, 0, datetime.now())).current_price
        if price == 0:
            logger.warning(f"No price available for {symbol}")
            return False

        if direction == "long":
            exec_price = price * (1 + slippage)
            cost = quantity * exec_price + commission

            if cost > self.cash:
                logger.warning(f"Insufficient cash for order: {cost:.2f} > {self.cash:.2f}")
                return False

            self.cash -= cost
            if symbol in self.positions:
                pos = self.positions[symbol]
                total_qty = pos.quantity + quantity
                pos.entry_price = (pos.quantity * pos.entry_price + quantity * exec_price) / total_qty
                pos.quantity = total_qty
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=exec_price,
                    entry_time=datetime.now(),
                    current_price=price
                )

        elif direction == "short":
            exec_price = price * (1 - slippage)
            # For short, we receive cash
            proceeds = abs(quantity) * exec_price - commission
            self.cash += proceeds
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= abs(quantity)
                if abs(pos.quantity) < 0.01:
                    del self.positions[symbol]
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=-abs(quantity),
                    entry_price=exec_price,
                    entry_time=datetime.now(),
                    current_price=price
                )

        elif direction == "close":
            if symbol not in self.positions:
                return False
            pos = self.positions[symbol]
            if pos.quantity > 0:
                exec_price = price * (1 - slippage)
                proceeds = pos.quantity * exec_price - commission
            else:
                exec_price = price * (1 + slippage)
                proceeds = -pos.quantity * exec_price - commission
            self.cash += proceeds
            del self.positions[symbol]

        # Record trade
        self.trade_history.append({
            "symbol": symbol,
            "direction": direction,
            "quantity": quantity,
            "price": exec_price,
            "commission": commission,
            "timestamp": datetime.now()
        })

        return True

    def run_backtest(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with OHLCV data, index should be datetime
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)

        Returns:
            Dictionary with backtest results
        """
        self.initialize()

        # Filter data by date if provided
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]

        logger.info(f"Running backtest from {data.index[0]} to {data.index[-1]}")
        logger.info(f"Initial capital: {self.config.trading.initial_capital:.2f}")

        # Reset state
        self.positions = {}
        self.cash = self.config.trading.initial_capital
        self.equity_history = []
        self.trade_history = []

        # Process each bar
        for timestamp, row in data.iterrows():
            # Update prices
            prices = {"BACKTEST": row.get("close", row.get("Close", 0))}
            self.update_prices(prices)

            # Generate signals
            market_slice = data.loc[:timestamp].tail(60)  # Last 60 bars for context
            signals = self.generate_signals(market_slice)

            # Process signals
            for signal in signals:
                order = self.process_signal(signal)
                if order:
                    self.execute_order(order)

        # Calculate results
        results = self._calculate_backtest_results()

        logger.info(f"Backtest complete. Final equity: {self.equity:.2f}")
        logger.info(f"Total return: {results['total_return_pct']:.2f}%")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

        return results

    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics."""
        if not self.equity_history:
            return {"error": "No equity history"}

        equity_series = pd.Series(
            [e for _, e in self.equity_history],
            index=[t for t, _ in self.equity_history]
        )

        returns = equity_series.pct_change().dropna()

        initial = self.config.trading.initial_capital
        final = self.equity

        # Calculate metrics
        total_return = (final - initial) / initial
        total_return_pct = total_return * 100

        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

            # Max drawdown
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0

        return {
            "initial_capital": initial,
            "final_equity": final,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown * 100,
            "n_trades": len(self.trade_history),
            "equity_curve": self.equity_history,
            "trades": self.trade_history
        }

    def run_monte_carlo(
        self,
        backtest_results: Dict[str, Any],
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation on backtest results."""
        try:
            from .backtesting import MonteCarloSimulator, BacktestResults

            # Create BacktestResults object
            returns = []
            equity = backtest_results["initial_capital"]
            for _, eq in backtest_results["equity_curve"]:
                if equity > 0:
                    returns.append((eq - equity) / equity)
                equity = eq

            bt_results = BacktestResults(
                equity_curve=backtest_results["equity_curve"],
                returns=returns,
                trade_history=backtest_results["trades"],
                initial_capital=backtest_results["initial_capital"]
            )

            mc = MonteCarloSimulator(
                n_simulations=n_simulations,
                method=self.config.backtest.bootstrap_method
            )
            mc_results = mc.run(bt_results)

            return {
                "n_simulations": n_simulations,
                "sharpe_mean": np.mean(mc_results.sharpe_ratios),
                "sharpe_std": np.std(mc_results.sharpe_ratios),
                "sharpe_ci_95": mc_results.get_confidence_interval("sharpe", 0.95),
                "return_mean": np.mean(mc_results.total_returns),
                "return_ci_95": mc_results.get_confidence_interval("return", 0.95),
                "prob_loss": mc_results.get_probability_of_loss(),
                "prob_drawdown_20": mc_results.get_probability_of_drawdown(20)
            }
        except ImportError as e:
            logger.warning(f"Monte Carlo not available: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "initialized": self._initialized,
            "env": self.config.env,
            "cash": self.cash,
            "equity": self.equity,
            "positions": {s: p.__dict__ for s, p in self.positions.items()},
            "total_return_pct": self.total_return * 100,
            "n_positions": len(self.positions),
            "n_trades": len(self.trade_history),
            "calibrators": list(self._calibrators.keys()),
            "signal_generators": list(self._signal_generators.keys())
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the trading system."""
        logger.info("Shutting down trading system...")

        # Close all positions (in live trading, this would execute orders)
        for symbol in list(self.positions.keys()):
            logger.info(f"Closing position in {symbol}")

        logger.info("Trading system shutdown complete")


def create_trading_system(config_file: Optional[str] = None) -> TradingSystem:
    """Factory function to create a configured trading system."""
    config = load_config(config_file)
    return TradingSystem(config)
