"""
Database configuration and connection management.

Provides configuration loading and database URL construction
with support for environment variables and secure credential handling.

Environment Variables:
    QUANT_DB_HOST: Database host (default: localhost)
    QUANT_DB_PORT: Database port (default: 5432)
    QUANT_DB_NAME: Database name (default: quant_trading_db)
    QUANT_DB_USER: Database user (default: postgres)
    QUANT_DB_PASSWORD: Database password (required in production)
    QUANT_DB_URL: Full database URL (overrides individual settings)
    QUANT_DB_POOL_SIZE: Connection pool size (default: 20)
    QUANT_DB_SSL_MODE: SSL mode (default: prefer)

Example:
    >>> from quant_trading.database.config import get_database_config, get_database_url
    >>> config = get_database_config()
    >>> url = get_database_url()
    >>> db = TimeSeriesDB(url, pool_size=config.pool_size)
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """
    Database configuration settings.

    Attributes:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password
        ssl_mode: SSL mode (disable, allow, prefer, require, verify-ca, verify-full)
        pool_size: Connection pool size
        max_overflow: Max overflow connections
        pool_pre_ping: Verify connections before use
        echo: Log SQL statements
    """

    host: str = "localhost"
    port: int = 5432
    database: str = "quant_trading_db"
    user: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 20
    max_overflow: int = 10
    pool_pre_ping: bool = True
    echo: bool = False

    # Optional SSL certificate paths
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_root_cert: Optional[str] = None

    def get_connection_url(self) -> str:
        """
        Build PostgreSQL connection URL.

        Returns:
            PostgreSQL connection URL string
        """
        # URL-encode password to handle special characters
        password = quote_plus(self.password) if self.password else ""

        # Build base URL
        if password:
            url = f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{self.database}"
        else:
            url = f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"

        # Add SSL parameters
        params = []
        if self.ssl_mode:
            params.append(f"sslmode={self.ssl_mode}")
        if self.ssl_cert:
            params.append(f"sslcert={self.ssl_cert}")
        if self.ssl_key:
            params.append(f"sslkey={self.ssl_key}")
        if self.ssl_root_cert:
            params.append(f"sslrootcert={self.ssl_root_cert}")

        if params:
            url += "?" + "&".join(params)

        return url

    def get_async_connection_url(self) -> str:
        """
        Build async PostgreSQL connection URL (for asyncpg).

        Returns:
            Async PostgreSQL connection URL string
        """
        url = self.get_connection_url()
        return url.replace("postgresql://", "postgresql+asyncpg://")


def get_database_config() -> DatabaseConfig:
    """
    Load database configuration from environment variables.

    Returns:
        DatabaseConfig instance with settings from environment
    """
    config = DatabaseConfig(
        host=os.environ.get("QUANT_DB_HOST", "localhost"),
        port=int(os.environ.get("QUANT_DB_PORT", "5432")),
        database=os.environ.get("QUANT_DB_NAME", "quant_trading_db"),
        user=os.environ.get("QUANT_DB_USER", "postgres"),
        password=os.environ.get("QUANT_DB_PASSWORD", ""),
        ssl_mode=os.environ.get("QUANT_DB_SSL_MODE", "prefer"),
        pool_size=int(os.environ.get("QUANT_DB_POOL_SIZE", "20")),
        max_overflow=int(os.environ.get("QUANT_DB_MAX_OVERFLOW", "10")),
        pool_pre_ping=os.environ.get("QUANT_DB_POOL_PRE_PING", "true").lower() == "true",
        echo=os.environ.get("QUANT_DB_ECHO", "false").lower() == "true",
        ssl_cert=os.environ.get("QUANT_DB_SSL_CERT"),
        ssl_key=os.environ.get("QUANT_DB_SSL_KEY"),
        ssl_root_cert=os.environ.get("QUANT_DB_SSL_ROOT_CERT"),
    )

    return config


def get_database_url() -> str:
    """
    Get database connection URL.

    Checks QUANT_DB_URL first, then builds from individual settings.

    Returns:
        PostgreSQL connection URL string
    """
    # Check for full URL override
    url = os.environ.get("QUANT_DB_URL")
    if url:
        return url

    # Build from config
    config = get_database_config()
    return config.get_connection_url()


def validate_database_config(config: DatabaseConfig) -> list:
    """
    Validate database configuration settings.

    Args:
        config: DatabaseConfig to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if not config.host:
        errors.append("Database host is required")
    if not config.database:
        errors.append("Database name is required")
    if not config.user:
        errors.append("Database user is required")

    # Check port range
    if not (1 <= config.port <= 65535):
        errors.append(f"Invalid port number: {config.port}")

    # Check pool size
    if config.pool_size < 1:
        errors.append(f"Invalid pool size: {config.pool_size}")
    if config.max_overflow < 0:
        errors.append(f"Invalid max overflow: {config.max_overflow}")

    # Check SSL mode
    valid_ssl_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
    if config.ssl_mode and config.ssl_mode not in valid_ssl_modes:
        errors.append(f"Invalid SSL mode: {config.ssl_mode}")

    # Warn about missing password in production
    if not config.password and os.environ.get("QUANT_ENV") == "production":
        logger.warning("Database password not set in production environment")

    return errors


# Pre-configured settings for common environments
DEVELOPMENT_CONFIG = DatabaseConfig(
    host="localhost",
    port=5432,
    database="quant_trading_dev",
    user="postgres",
    password="postgres",
    ssl_mode="disable",
    pool_size=5,
    echo=True,
)

TEST_CONFIG = DatabaseConfig(
    host="localhost",
    port=5432,
    database="quant_trading_test",
    user="postgres",
    password="postgres",
    ssl_mode="disable",
    pool_size=2,
    echo=False,
)

PRODUCTION_CONFIG = DatabaseConfig(
    host=os.environ.get("QUANT_DB_HOST", "localhost"),
    port=int(os.environ.get("QUANT_DB_PORT", "5432")),
    database=os.environ.get("QUANT_DB_NAME", "quant_trading_db"),
    user=os.environ.get("QUANT_DB_USER", "quant_app"),
    password=os.environ.get("QUANT_DB_PASSWORD", ""),
    ssl_mode="require",
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)


def get_config_for_environment(env: Optional[str] = None) -> DatabaseConfig:
    """
    Get database configuration for specified environment.

    Args:
        env: Environment name ('development', 'test', 'production')
             If None, uses QUANT_ENV environment variable

    Returns:
        DatabaseConfig for the specified environment
    """
    if env is None:
        env = os.environ.get("QUANT_ENV", "development")

    env = env.lower()

    if env == "development":
        return DEVELOPMENT_CONFIG
    elif env == "test":
        return TEST_CONFIG
    elif env == "production":
        return PRODUCTION_CONFIG
    else:
        logger.warning(f"Unknown environment '{env}', using development config")
        return DEVELOPMENT_CONFIG
