"""
Data Pipeline Module.

This module provides comprehensive data infrastructure for quantitative trading:

Components:
    - providers: Data provider abstraction and implementations
    - validation: Data quality validation framework
    - ingestion: Data ingestion pipeline with retry and storage
    - options: Options chain processing, IV calculation, Greeks
    - streaming: Real-time data streaming with WebSocket support
    - monitoring: Data quality monitoring and alerting
    - alternative: Alternative data integration (FRED, sentiment, events)
    - storage: TimescaleDB optimization, compression, aggregates
    - reference: Symbol master, trading calendars, corporate actions
    - api: REST API for data access
    - recovery: Gap detection, backfilling, data reconciliation

Performance Targets:
    - Ingestion: >10,000 bars/second
    - Real-time latency: <100ms
    - API response: <50ms for cached queries

Usage:
    from quant_trading.data import (
        DataProviderFactory,
        DataIngestionPipeline,
        OptionsChainProcessor,
        StreamManager,
        DataQualityMonitor,
        ReferenceDataManager
    )

    # Create data provider
    provider = DataProviderFactory.create('yahoo')

    # Get historical data
    df = provider.get_historical_data('AAPL', start_date, end_date)

    # Process options chain
    processor = OptionsChainProcessor()
    chain = processor.process_chain(raw_chain)

    # Start real-time streaming
    manager = StreamManager()
    await manager.subscribe(['AAPL'], [StreamEventType.QUOTE], handler)
"""

# Data Providers
from .providers import (
    DataProvider,
    DataProviderFactory,
    YahooFinanceProvider,
    AlphaVantageProvider,
    PolygonProvider,
    IEXCloudProvider,
    DataFrequency,
    DataType,
    RateLimiter,
)

# Data Validation
from .validation import (
    MarketDataValidator,
    OptionsDataValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    DataValidationPipeline,
    DataQuality,
)

# Data Ingestion
from .ingestion import (
    DataIngestionPipeline,
    IngestionResult,
    IngestionStatus,
    IngestionConfig,
    IncrementalIngestion,
)

# Options Processing
from .options import (
    OptionContract,
    OptionsChain,
    OptionType,
    ExerciseStyle,
    BlackScholes,
    ImpliedVolatilityCalculator,
    GreeksCalculator,
    VolatilitySurface,
    VolatilitySurfacePoint,
    SVIParameterization,
    OptionsChainProcessor,
    parse_options_data,
)

# Real-Time Streaming
from .streaming import (
    StreamEventType,
    ConnectionState,
    StreamEvent,
    QuoteEvent,
    TradeEvent,
    BarEvent,
    OrderBookEvent,
    OrderBookLevel,
    StreamSubscription,
    DataStreamProvider,
    SimulatedStreamProvider,
    PolygonStreamProvider,
    StreamAggregator,
    StreamBuffer,
    StreamManager,
)

# Data Quality Monitoring
from .monitoring import (
    AlertSeverity,
    AlertType,
    DataQualityAlert,
    DataQualityMetric,
    MetricAggregator,
    SymbolHealthTracker,
    ProviderHealthTracker,
    DataQualityMonitor,
    DataQualityReporter,
)

# Alternative Data
from .alternative import (
    DataCategory,
    DataSeriesMetadata,
    DataObservation,
    AlternativeDataProvider,
    FREDProvider,
    EarningsEvent,
    DividendEvent,
    SplitEvent,
    CorporateEventsProvider,
    SentimentScore,
    SentimentProvider,
    AlternativeDataManager,
)

# Storage Optimization
from .storage import (
    CompressionLevel,
    RetentionPolicy,
    HypertableConfig,
    ContinuousAggregateConfig,
    TimescaleManager,
    StorageStats,
    DataStorageOptimizer,
    DataRetentionManager,
)

# Reference Data
from .reference import (
    AssetClass,
    Exchange,
    CorporateActionType,
    SecurityInfo,
    CorporateAction,
    TradingSession,
    MarketHoliday,
    TradingCalendar,
    IndexComposition,
    SymbolMaster,
    CorporateActionsManager,
    ReferenceDataManager,
)

# Data API
from .api import (
    TimeRange,
    OHLCVBar,
    QuoteData,
    OptionQuoteData,
    HealthMetrics,
    HistoricalDataRequest,
    OptionsChainRequest,
    DataQueryResponse,
    DataService,
    create_data_api,
)

# Data Recovery
from .recovery import (
    GapType,
    RecoveryStatus,
    BackfillPriority,
    DataGap,
    BackfillRequest,
    RecoveryResult,
    GapDetector,
    DataValidator,
    BackfillManager,
    DataReconciler,
)

__all__ = [
    # Providers
    'DataProvider',
    'DataProviderFactory',
    'YahooFinanceProvider',
    'AlphaVantageProvider',
    'PolygonProvider',
    'IEXCloudProvider',
    'DataFrequency',
    'DataType',
    'RateLimiter',

    # Validation
    'MarketDataValidator',
    'OptionsDataValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    'DataValidationPipeline',
    'DataQuality',

    # Ingestion
    'DataIngestionPipeline',
    'IngestionResult',
    'IngestionStatus',
    'IngestionConfig',
    'IncrementalIngestion',

    # Options
    'OptionContract',
    'OptionsChain',
    'OptionType',
    'ExerciseStyle',
    'BlackScholes',
    'ImpliedVolatilityCalculator',
    'GreeksCalculator',
    'VolatilitySurface',
    'VolatilitySurfacePoint',
    'SVIParameterization',
    'OptionsChainProcessor',
    'parse_options_data',

    # Streaming
    'StreamEventType',
    'ConnectionState',
    'StreamEvent',
    'QuoteEvent',
    'TradeEvent',
    'BarEvent',
    'OrderBookEvent',
    'OrderBookLevel',
    'StreamSubscription',
    'DataStreamProvider',
    'SimulatedStreamProvider',
    'PolygonStreamProvider',
    'StreamAggregator',
    'StreamBuffer',
    'StreamManager',

    # Monitoring
    'AlertSeverity',
    'AlertType',
    'DataQualityAlert',
    'DataQualityMetric',
    'MetricAggregator',
    'SymbolHealthTracker',
    'ProviderHealthTracker',
    'DataQualityMonitor',
    'DataQualityReporter',

    # Alternative Data
    'DataCategory',
    'DataSeriesMetadata',
    'DataObservation',
    'AlternativeDataProvider',
    'FREDProvider',
    'EarningsEvent',
    'DividendEvent',
    'SplitEvent',
    'CorporateEventsProvider',
    'SentimentScore',
    'SentimentProvider',
    'AlternativeDataManager',

    # Storage
    'CompressionLevel',
    'RetentionPolicy',
    'HypertableConfig',
    'ContinuousAggregateConfig',
    'TimescaleManager',
    'StorageStats',
    'DataStorageOptimizer',
    'DataRetentionManager',

    # Reference
    'AssetClass',
    'Exchange',
    'CorporateActionType',
    'SecurityInfo',
    'CorporateAction',
    'TradingSession',
    'MarketHoliday',
    'TradingCalendar',
    'IndexComposition',
    'SymbolMaster',
    'CorporateActionsManager',
    'ReferenceDataManager',

    # API
    'TimeRange',
    'OHLCVBar',
    'QuoteData',
    'OptionQuoteData',
    'HealthMetrics',
    'HistoricalDataRequest',
    'OptionsChainRequest',
    'DataQueryResponse',
    'DataService',
    'create_data_api',

    # Recovery
    'GapType',
    'RecoveryStatus',
    'BackfillPriority',
    'DataGap',
    'BackfillRequest',
    'RecoveryResult',
    'GapDetector',
    'DataValidator',
    'BackfillManager',
    'DataReconciler',
]
