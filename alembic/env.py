"""
Alembic environment configuration.

This module configures how Alembic runs migrations, including:
- Database connection setup
- Model metadata import for autogenerate
- Online and offline migration modes

Usage:
    # Generate migration from model changes
    alembic revision --autogenerate -m "Description"

    # Apply all migrations
    alembic upgrade head

    # Rollback one migration
    alembic downgrade -1

    # Show migration history
    alembic history
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models to register them with SQLAlchemy
from src.python.quant_trading.database.models import Base

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def get_database_url() -> str:
    """
    Get database URL from environment or config.

    Priority:
    1. DATABASE_URL environment variable
    2. QUANT_DB_URL environment variable
    3. Default development URL

    Returns:
        PostgreSQL connection URL
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    url = os.environ.get("QUANT_DB_URL")
    if url:
        return url

    # Default for local development
    return "postgresql://postgres:postgres@localhost:5432/quant_trading_db"


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schemas for proper diffing
        include_schemas=True,
        # Compare types for column type changes
        compare_type=True,
        # Compare server defaults
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    # Build configuration dict
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Include schemas for proper diffing
            include_schemas=True,
            # Compare types for column type changes
            compare_type=True,
            # Compare server defaults
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


# Run appropriate migration mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
