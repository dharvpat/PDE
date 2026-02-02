"""
Configuration management for the Quantitative Trading System.

Supports loading from:
- Environment variables
- YAML/JSON config files
- Command-line arguments
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = "sqlite:///quant_trading.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class ModelConfig:
    """Model calibration configuration."""
    # Heston model defaults
    heston_kappa_bounds: tuple = (0.1, 10.0)
    heston_theta_bounds: tuple = (0.01, 1.0)
    heston_sigma_bounds: tuple = (0.1, 2.0)
    heston_rho_bounds: tuple = (-0.99, 0.0)
    heston_v0_bounds: tuple = (0.01, 1.0)

    # SABR model defaults
    sabr_beta: float = 0.5
    sabr_alpha_bounds: tuple = (0.01, 2.0)
    sabr_rho_bounds: tuple = (-0.99, 0.99)
    sabr_nu_bounds: tuple = (0.01, 2.0)

    # OU process defaults
    ou_lookback_days: int = 60
    ou_min_half_life: float = 5.0
    ou_max_half_life: float = 60.0


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing
    initial_capital: float = 100000.0
    max_position_pct: float = 0.10  # Max 10% per position
    max_portfolio_leverage: float = 1.0

    # Risk management
    max_drawdown_pct: float = 0.25
    daily_var_limit: float = 0.02
    stop_loss_pct: float = 0.05

    # Signal thresholds
    min_signal_confidence: float = 0.6
    signal_aggregation_method: str = "weighted"  # weighted, majority, unanimous

    # Transaction costs
    commission_per_share: float = 0.005
    slippage_bps: float = 5.0

    # Trading hours (EST)
    market_open: str = "09:30"
    market_close: str = "16:00"


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    warmup_days: int = 60

    # Walk-forward settings
    walk_forward_enabled: bool = True
    in_sample_days: int = 252
    out_of_sample_days: int = 63

    # Monte Carlo settings
    monte_carlo_simulations: int = 1000
    bootstrap_method: str = "block"  # shuffle, block, parametric
    block_size: int = 21


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10_000_000  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Environment
    env: str = "development"  # development, staging, production
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if "database" in data:
            config.database = DatabaseConfig(**data["database"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "trading" in data:
            config.trading = TradingConfig(**data["trading"])
        if "backtest" in data:
            config.backtest = BacktestConfig(**data["backtest"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "env" in data:
            config.env = data["env"]
        if "debug" in data:
            config.debug = data["debug"]

        return config

    @classmethod
    def from_file(cls, path: str) -> "Config":
        """Load config from JSON or YAML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        config = cls()

        # Database
        if db_url := os.getenv("QT_DATABASE_URL"):
            config.database.url = db_url

        # Trading
        if capital := os.getenv("QT_INITIAL_CAPITAL"):
            config.trading.initial_capital = float(capital)
        if max_pos := os.getenv("QT_MAX_POSITION_PCT"):
            config.trading.max_position_pct = float(max_pos)
        if max_dd := os.getenv("QT_MAX_DRAWDOWN_PCT"):
            config.trading.max_drawdown_pct = float(max_dd)

        # Environment
        if env := os.getenv("QT_ENV"):
            config.env = env
        if os.getenv("QT_DEBUG", "").lower() in ("1", "true", "yes"):
            config.debug = True

        # Logging
        if log_level := os.getenv("QT_LOG_LEVEL"):
            config.logging.level = log_level
        if log_file := os.getenv("QT_LOG_FILE"):
            config.logging.file = log_file

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "database": {
                "url": self.database.url,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo,
            },
            "model": {
                "heston_kappa_bounds": self.model.heston_kappa_bounds,
                "heston_theta_bounds": self.model.heston_theta_bounds,
                "heston_sigma_bounds": self.model.heston_sigma_bounds,
                "heston_rho_bounds": self.model.heston_rho_bounds,
                "heston_v0_bounds": self.model.heston_v0_bounds,
                "sabr_beta": self.model.sabr_beta,
                "ou_lookback_days": self.model.ou_lookback_days,
            },
            "trading": {
                "initial_capital": self.trading.initial_capital,
                "max_position_pct": self.trading.max_position_pct,
                "max_drawdown_pct": self.trading.max_drawdown_pct,
                "min_signal_confidence": self.trading.min_signal_confidence,
                "commission_per_share": self.trading.commission_per_share,
                "slippage_bps": self.trading.slippage_bps,
            },
            "backtest": {
                "start_date": self.backtest.start_date,
                "end_date": self.backtest.end_date,
                "monte_carlo_simulations": self.backtest.monte_carlo_simulations,
                "walk_forward_enabled": self.backtest.walk_forward_enabled,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
            },
            "env": self.env,
            "debug": self.debug,
        }

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(
    config_file: Optional[str] = None,
    use_env: bool = True
) -> Config:
    """
    Load configuration with precedence:
    1. Config file (if provided)
    2. Environment variables (if use_env=True)
    3. Defaults
    """
    # Start with defaults
    config = Config()

    # Load from file if provided
    if config_file:
        try:
            config = Config.from_file(config_file)
            logger.info(f"Loaded config from {config_file}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_file}, using defaults")

    # Override with environment variables
    if use_env:
        env_config = Config.from_env()
        # Merge env config (only non-default values)
        if os.getenv("QT_DATABASE_URL"):
            config.database.url = env_config.database.url
        if os.getenv("QT_INITIAL_CAPITAL"):
            config.trading.initial_capital = env_config.trading.initial_capital
        if os.getenv("QT_ENV"):
            config.env = env_config.env
        if os.getenv("QT_DEBUG"):
            config.debug = env_config.debug
        if os.getenv("QT_LOG_LEVEL"):
            config.logging.level = env_config.logging.level

    return config


def setup_logging(config: LoggingConfig) -> None:
    """Configure logging based on config."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if config.file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=handlers
    )
