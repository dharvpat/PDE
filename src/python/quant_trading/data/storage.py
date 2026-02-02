"""
Data Storage Optimization Module.

This module provides storage optimization for time-series market data:
- TimescaleDB hypertable management
- Continuous aggregates for OHLCV rollups
- Compression policies for historical data
- Retention policies and data lifecycle
- Partitioning strategies
- Query optimization

Optimized for high-frequency market data storage requirements.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Index,
    text,
    create_engine,
    MetaData
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class CompressionLevel(Enum):
    """Compression levels for storage."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RetentionPolicy(Enum):
    """Data retention policies."""
    KEEP_ALL = "keep_all"
    DAYS_30 = "30_days"
    DAYS_90 = "90_days"
    DAYS_365 = "365_days"
    YEARS_3 = "3_years"
    YEARS_7 = "7_years"


@dataclass
class HypertableConfig:
    """Configuration for a TimescaleDB hypertable."""
    table_name: str
    time_column: str
    chunk_time_interval: str = "1 day"  # TimescaleDB interval
    compression_after: Optional[str] = "7 days"
    retention_period: Optional[str] = None
    space_partitioning_column: Optional[str] = None
    number_partitions: int = 4
    replication_factor: int = 1


@dataclass
class ContinuousAggregateConfig:
    """Configuration for a continuous aggregate."""
    name: str
    source_hypertable: str
    time_bucket: str  # e.g., '1 minute', '1 hour', '1 day'
    group_by_columns: List[str]
    aggregations: Dict[str, str]  # column -> aggregation function
    refresh_lag: str = "1 hour"
    refresh_interval: str = "30 minutes"
    retention_period: Optional[str] = None


class TimescaleManager:
    """
    Manager for TimescaleDB operations.

    Handles hypertable creation, compression, and continuous aggregates.
    """

    def __init__(self, connection_string: str):
        """
        Initialize TimescaleDB manager.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self._engine = None
        self._session_factory = None

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            self._engine = create_engine(self.connection_string)
            self._session_factory = sessionmaker(bind=self._engine)
            logger.info("Connected to TimescaleDB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False

    def get_session(self) -> Session:
        """Get a database session."""
        if not self._session_factory:
            raise RuntimeError("Not connected to database")
        return self._session_factory()

    def create_hypertable(self, config: HypertableConfig) -> bool:
        """
        Create a TimescaleDB hypertable.

        Args:
            config: Hypertable configuration

        Returns:
            True if successful
        """
        session = self.get_session()
        try:
            # Create the hypertable
            sql = text(f"""
                SELECT create_hypertable(
                    '{config.table_name}',
                    '{config.time_column}',
                    chunk_time_interval => INTERVAL '{config.chunk_time_interval}',
                    if_not_exists => TRUE
                );
            """)
            session.execute(sql)

            # Add space partitioning if configured
            if config.space_partitioning_column:
                sql = text(f"""
                    SELECT add_dimension(
                        '{config.table_name}',
                        '{config.space_partitioning_column}',
                        number_partitions => {config.number_partitions}
                    );
                """)
                try:
                    session.execute(sql)
                except Exception as e:
                    logger.warning(f"Space partitioning may already exist: {e}")

            session.commit()
            logger.info(f"Created hypertable: {config.table_name}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create hypertable: {e}")
            return False
        finally:
            session.close()

    def enable_compression(
        self,
        table_name: str,
        segment_by: List[str],
        order_by: str,
        compress_after: str = "7 days"
    ) -> bool:
        """
        Enable compression on a hypertable.

        Args:
            table_name: Name of the hypertable
            segment_by: Columns to segment by (typically symbol)
            order_by: Column to order by (typically timestamp DESC)
            compress_after: Interval after which to compress

        Returns:
            True if successful
        """
        session = self.get_session()
        try:
            # Enable compression
            segment_by_str = ", ".join(segment_by)
            sql = text(f"""
                ALTER TABLE {table_name} SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = '{segment_by_str}',
                    timescaledb.compress_orderby = '{order_by}'
                );
            """)
            session.execute(sql)

            # Add compression policy
            sql = text(f"""
                SELECT add_compression_policy(
                    '{table_name}',
                    INTERVAL '{compress_after}'
                );
            """)
            try:
                session.execute(sql)
            except Exception as e:
                logger.warning(f"Compression policy may already exist: {e}")

            session.commit()
            logger.info(f"Enabled compression on {table_name}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to enable compression: {e}")
            return False
        finally:
            session.close()

    def add_retention_policy(
        self,
        table_name: str,
        retention_period: str
    ) -> bool:
        """
        Add data retention policy.

        Args:
            table_name: Name of the hypertable
            retention_period: How long to keep data (e.g., '1 year')

        Returns:
            True if successful
        """
        session = self.get_session()
        try:
            sql = text(f"""
                SELECT add_retention_policy(
                    '{table_name}',
                    INTERVAL '{retention_period}'
                );
            """)
            session.execute(sql)
            session.commit()
            logger.info(f"Added retention policy to {table_name}: {retention_period}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add retention policy: {e}")
            return False
        finally:
            session.close()

    def create_continuous_aggregate(
        self,
        config: ContinuousAggregateConfig
    ) -> bool:
        """
        Create a continuous aggregate.

        Args:
            config: Continuous aggregate configuration

        Returns:
            True if successful
        """
        session = self.get_session()
        try:
            # Build aggregation SQL
            agg_columns = []
            for col, func in config.aggregations.items():
                agg_columns.append(f"{func}({col}) AS {col}")

            group_by_str = ", ".join(config.group_by_columns)
            agg_str = ",\n                ".join(agg_columns)

            # Create the continuous aggregate
            sql = text(f"""
                CREATE MATERIALIZED VIEW {config.name}
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('{config.time_bucket}', timestamp) AS bucket,
                    {group_by_str},
                    {agg_str}
                FROM {config.source_hypertable}
                GROUP BY bucket, {group_by_str}
                WITH NO DATA;
            """)
            session.execute(sql)

            # Add refresh policy
            sql = text(f"""
                SELECT add_continuous_aggregate_policy(
                    '{config.name}',
                    start_offset => NULL,
                    end_offset => INTERVAL '{config.refresh_lag}',
                    schedule_interval => INTERVAL '{config.refresh_interval}'
                );
            """)
            session.execute(sql)

            session.commit()
            logger.info(f"Created continuous aggregate: {config.name}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create continuous aggregate: {e}")
            return False
        finally:
            session.close()

    def refresh_continuous_aggregate(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """
        Manually refresh a continuous aggregate.

        Args:
            name: Name of the continuous aggregate
            start_time: Start of refresh window
            end_time: End of refresh window

        Returns:
            True if successful
        """
        session = self.get_session()
        try:
            if start_time and end_time:
                sql = text(f"""
                    CALL refresh_continuous_aggregate(
                        '{name}',
                        '{start_time.isoformat()}',
                        '{end_time.isoformat()}'
                    );
                """)
            else:
                sql = text(f"""
                    CALL refresh_continuous_aggregate('{name}', NULL, NULL);
                """)

            session.execute(sql)
            session.commit()
            logger.info(f"Refreshed continuous aggregate: {name}")
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to refresh continuous aggregate: {e}")
            return False
        finally:
            session.close()

    def get_chunk_info(self, table_name: str) -> pd.DataFrame:
        """
        Get information about chunks in a hypertable.

        Args:
            table_name: Name of the hypertable

        Returns:
            DataFrame with chunk information
        """
        session = self.get_session()
        try:
            sql = text(f"""
                SELECT
                    chunk_schema || '.' || chunk_name as chunk,
                    range_start,
                    range_end,
                    is_compressed,
                    pg_size_pretty(
                        pg_total_relation_size(chunk_schema || '.' || chunk_name)
                    ) as size
                FROM timescaledb_information.chunks
                WHERE hypertable_name = '{table_name}'
                ORDER BY range_start DESC;
            """)
            result = session.execute(sql)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        finally:
            session.close()

    def get_compression_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get compression statistics for a hypertable.

        Args:
            table_name: Name of the hypertable

        Returns:
            Dictionary with compression stats
        """
        session = self.get_session()
        try:
            sql = text(f"""
                SELECT
                    hypertable_name,
                    number_compressed_chunks,
                    compressed_heap_size,
                    compressed_index_size,
                    compressed_toast_size,
                    uncompressed_heap_size,
                    uncompressed_index_size,
                    uncompressed_toast_size
                FROM timescaledb_information.compression_settings
                WHERE hypertable_name = '{table_name}';
            """)
            result = session.execute(sql)
            row = result.fetchone()

            if row:
                return dict(zip(result.keys(), row))
            return {}
        finally:
            session.close()

    def compress_chunks(
        self,
        table_name: str,
        older_than: Optional[str] = None
    ) -> int:
        """
        Manually compress chunks.

        Args:
            table_name: Name of the hypertable
            older_than: Compress chunks older than this interval

        Returns:
            Number of chunks compressed
        """
        session = self.get_session()
        try:
            if older_than:
                sql = text(f"""
                    SELECT compress_chunk(c.chunk_name)
                    FROM timescaledb_information.chunks c
                    WHERE c.hypertable_name = '{table_name}'
                    AND c.range_end < NOW() - INTERVAL '{older_than}'
                    AND NOT c.is_compressed;
                """)
            else:
                sql = text(f"""
                    SELECT compress_chunk(c.chunk_name)
                    FROM timescaledb_information.chunks c
                    WHERE c.hypertable_name = '{table_name}'
                    AND NOT c.is_compressed;
                """)

            result = session.execute(sql)
            count = result.rowcount
            session.commit()

            logger.info(f"Compressed {count} chunks in {table_name}")
            return count

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to compress chunks: {e}")
            return 0
        finally:
            session.close()


@dataclass
class StorageStats:
    """Storage statistics for a table."""
    table_name: str
    total_size_bytes: int
    row_count: int
    chunk_count: int
    compressed_chunks: int
    oldest_data: Optional[datetime]
    newest_data: Optional[datetime]
    avg_compression_ratio: float = 1.0


class DataStorageOptimizer:
    """
    High-level optimizer for data storage.

    Provides automated optimization strategies and recommendations.
    """

    def __init__(self, timescale_manager: TimescaleManager):
        """
        Initialize optimizer.

        Args:
            timescale_manager: TimescaleDB manager instance
        """
        self.ts_manager = timescale_manager

    def setup_market_data_schema(self) -> bool:
        """
        Set up optimized schema for market data.

        Creates hypertables for:
        - market_prices (tick/minute data)
        - option_quotes
        - model_parameters

        Returns:
            True if successful
        """
        success = True

        # Market prices hypertable
        market_config = HypertableConfig(
            table_name="market_prices",
            time_column="timestamp",
            chunk_time_interval="1 day",
            compression_after="7 days",
            space_partitioning_column="symbol",
            number_partitions=4
        )
        success &= self.ts_manager.create_hypertable(market_config)

        # Enable compression
        success &= self.ts_manager.enable_compression(
            table_name="market_prices",
            segment_by=["symbol"],
            order_by="timestamp DESC",
            compress_after="7 days"
        )

        # Option quotes hypertable
        options_config = HypertableConfig(
            table_name="option_quotes",
            time_column="timestamp",
            chunk_time_interval="1 day",
            compression_after="3 days",
            space_partitioning_column="underlying",
            number_partitions=4
        )
        success &= self.ts_manager.create_hypertable(options_config)

        success &= self.ts_manager.enable_compression(
            table_name="option_quotes",
            segment_by=["underlying", "strike", "expiration"],
            order_by="timestamp DESC",
            compress_after="3 days"
        )

        return success

    def setup_continuous_aggregates(self) -> bool:
        """
        Set up continuous aggregates for common rollups.

        Creates aggregates for:
        - 1-minute OHLCV bars
        - Hourly OHLCV bars
        - Daily OHLCV bars

        Returns:
            True if successful
        """
        success = True

        # 1-minute bars
        minute_config = ContinuousAggregateConfig(
            name="market_prices_1min",
            source_hypertable="market_prices",
            time_bucket="1 minute",
            group_by_columns=["symbol"],
            aggregations={
                "price": "FIRST(price, timestamp)",
                "high": "MAX(price)",
                "low": "MIN(price)",
                "close": "LAST(price, timestamp)",
                "volume": "SUM(volume)"
            },
            refresh_lag="5 minutes",
            refresh_interval="1 minute"
        )
        success &= self.ts_manager.create_continuous_aggregate(minute_config)

        # Hourly bars
        hourly_config = ContinuousAggregateConfig(
            name="market_prices_1hour",
            source_hypertable="market_prices",
            time_bucket="1 hour",
            group_by_columns=["symbol"],
            aggregations={
                "open": "FIRST(price, timestamp)",
                "high": "MAX(price)",
                "low": "MIN(price)",
                "close": "LAST(price, timestamp)",
                "volume": "SUM(volume)"
            },
            refresh_lag="1 hour",
            refresh_interval="15 minutes"
        )
        success &= self.ts_manager.create_continuous_aggregate(hourly_config)

        # Daily bars
        daily_config = ContinuousAggregateConfig(
            name="market_prices_daily",
            source_hypertable="market_prices",
            time_bucket="1 day",
            group_by_columns=["symbol"],
            aggregations={
                "open": "FIRST(price, timestamp)",
                "high": "MAX(price)",
                "low": "MIN(price)",
                "close": "LAST(price, timestamp)",
                "volume": "SUM(volume)"
            },
            refresh_lag="1 day",
            refresh_interval="1 hour"
        )
        success &= self.ts_manager.create_continuous_aggregate(daily_config)

        return success

    def optimize_queries(self, session: Session) -> None:
        """
        Create indexes for common query patterns.

        Args:
            session: Database session
        """
        indexes = [
            # Symbol + time for time-series queries
            """
            CREATE INDEX IF NOT EXISTS idx_market_prices_symbol_time
            ON market_prices (symbol, timestamp DESC);
            """,
            # Options by underlying and expiration
            """
            CREATE INDEX IF NOT EXISTS idx_option_quotes_underlying_exp
            ON option_quotes (underlying, expiration, timestamp DESC);
            """,
            # Options by strike for Greeks calculations
            """
            CREATE INDEX IF NOT EXISTS idx_option_quotes_strike
            ON option_quotes (underlying, strike, option_type, timestamp DESC);
            """
        ]

        for idx_sql in indexes:
            try:
                session.execute(text(idx_sql))
                session.commit()
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
                session.rollback()

    def get_storage_recommendations(self) -> List[str]:
        """
        Get storage optimization recommendations.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check chunk info
        try:
            chunk_df = self.ts_manager.get_chunk_info("market_prices")
            if not chunk_df.empty:
                uncompressed = chunk_df[~chunk_df['is_compressed']]
                if len(uncompressed) > 7:
                    recommendations.append(
                        f"Consider compressing older chunks. "
                        f"{len(uncompressed)} chunks are uncompressed."
                    )
        except Exception:
            pass

        # Check compression stats
        try:
            stats = self.ts_manager.get_compression_stats("market_prices")
            if stats:
                compressed = stats.get('compressed_heap_size', 0)
                uncompressed = stats.get('uncompressed_heap_size', 0)
                if uncompressed > 0 and compressed > 0:
                    ratio = uncompressed / compressed
                    if ratio < 3:
                        recommendations.append(
                            f"Compression ratio is only {ratio:.1f}x. "
                            "Consider adjusting segment_by columns."
                        )
        except Exception:
            pass

        # General recommendations
        recommendations.extend([
            "Run VACUUM ANALYZE periodically to update statistics.",
            "Monitor chunk sizes to ensure they fit in memory.",
            "Consider using continuous aggregates for frequently queried rollups."
        ])

        return recommendations


class DataRetentionManager:
    """
    Manages data retention and archival.

    Handles data lifecycle including archival to cold storage.
    """

    def __init__(self, timescale_manager: TimescaleManager):
        """
        Initialize retention manager.

        Args:
            timescale_manager: TimescaleDB manager instance
        """
        self.ts_manager = timescale_manager

    def setup_retention_policies(
        self,
        policies: Dict[str, str]
    ) -> bool:
        """
        Set up retention policies for multiple tables.

        Args:
            policies: Dict of table_name -> retention_period

        Returns:
            True if successful
        """
        success = True
        for table, period in policies.items():
            success &= self.ts_manager.add_retention_policy(table, period)
        return success

    def archive_old_data(
        self,
        table_name: str,
        older_than: datetime,
        archive_path: str
    ) -> int:
        """
        Archive old data to files before deletion.

        Args:
            table_name: Table to archive from
            older_than: Archive data older than this
            archive_path: Path to write archive files

        Returns:
            Number of rows archived
        """
        session = self.ts_manager.get_session()
        try:
            # Export to CSV (in production, would use COPY command)
            sql = text(f"""
                SELECT * FROM {table_name}
                WHERE timestamp < :older_than
                ORDER BY timestamp;
            """)
            result = session.execute(sql, {"older_than": older_than})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if not df.empty:
                filename = f"{archive_path}/{table_name}_{older_than.date()}.parquet"
                df.to_parquet(filename, compression='snappy')
                logger.info(f"Archived {len(df)} rows to {filename}")
                return len(df)

            return 0

        finally:
            session.close()

    def get_retention_status(self) -> Dict[str, Any]:
        """
        Get current retention policy status.

        Returns:
            Dictionary with retention information
        """
        session = self.ts_manager.get_session()
        try:
            sql = text("""
                SELECT
                    hypertable_name,
                    config::jsonb->'drop_after' as drop_after
                FROM timescaledb_information.jobs
                WHERE proc_name = 'policy_retention';
            """)
            result = session.execute(sql)
            policies = {}
            for row in result.fetchall():
                policies[row[0]] = {
                    'drop_after': row[1]
                }
            return policies
        finally:
            session.close()
