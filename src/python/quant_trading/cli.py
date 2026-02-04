#!/usr/bin/env python3
"""
Quantitative Trading System - Command Line Interface

Usage:
    quant-trading backtest --symbol <ticker> [--start <date>] [--end <date>] [--config <file>]
    quant-trading backtest --data <file> [--start <date>] [--end <date>] [--config <file>]
    quant-trading calibrate --model <model> --data <file> [--config <file>]
    quant-trading status [--config <file>]
    quant-trading config [--show | --generate <file>]
    quant-trading demo [--symbol <ticker>]
    quant-trading version
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def setup_logging(verbose: bool = False, debug: bool = False):
    """Setup logging for CLI."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def fetch_yfinance_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical data from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    print(f"Fetching {symbol} data from Yahoo Finance...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Standardize column names to capitalized format expected by backtester
    data.columns = [c.capitalize() for c in data.columns]

    # Remove timezone info for consistency
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data['Symbol'] = symbol
    return data


def cmd_backtest(args):
    """Run backtest command."""
    from .trading_system import TradingSystem
    from .config import load_config

    print(f"\n{'='*60}")
    print("QUANTITATIVE TRADING SYSTEM - BACKTEST")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(args.config)
    if args.capital:
        config.trading.initial_capital = args.capital

    # Determine data source
    if args.symbol:
        # Fetch from Yahoo Finance
        start = args.start or "2023-01-01"
        end = args.end or datetime.now().strftime("%Y-%m-%d")
        data = fetch_yfinance_data(args.symbol, start, end)
        print(f"Symbol: {args.symbol}")
    elif args.data:
        # Load from file
        print(f"Loading data from: {args.data}")
        data_path = Path(args.data)

        if data_path.suffix == ".csv":
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        elif data_path.suffix == ".parquet":
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    else:
        raise ValueError("Must specify either --symbol or --data")

    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Create and run backtest
    system = TradingSystem(config)

    print(f"\nRunning backtest...")
    print(f"Initial capital: ${config.trading.initial_capital:,.2f}")
    print(f"Max position size: {config.trading.max_position_pct*100:.1f}%")

    results = system.run_backtest(data, args.start, args.end)

    # Print results
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Capital:  ${results['initial_capital']:>15,.2f}")
    print(f"Final Equity:     ${results['final_equity']:>15,.2f}")
    print(f"Total Return:     {results['total_return_pct']:>15.2f}%")
    print(f"Volatility:       {results['volatility_pct']:>15.2f}%")
    print(f"Sharpe Ratio:     {results['sharpe_ratio']:>15.2f}")
    print(f"Max Drawdown:     {results['max_drawdown_pct']:>15.2f}%")
    print(f"Number of Trades: {results['n_trades']:>15}")
    print(f"{'='*60}\n")

    # Run Monte Carlo if requested
    if args.monte_carlo:
        print("Running Monte Carlo simulation...")
        mc_results = system.run_monte_carlo(results, args.monte_carlo)
        if "error" not in mc_results:
            print(f"\nMonte Carlo Results ({mc_results['n_simulations']} simulations):")
            print(f"  Sharpe Ratio: {mc_results['sharpe_mean']:.2f} +/- {mc_results['sharpe_std']:.2f}")
            print(f"  95% CI: [{mc_results['sharpe_ci_95'][0]:.2f}, {mc_results['sharpe_ci_95'][1]:.2f}]")
            print(f"  Probability of Loss: {mc_results['prob_loss']*100:.1f}%")
            print(f"  Probability of >20% Drawdown: {mc_results['prob_drawdown_20']*100:.1f}%")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        # Convert equity curve timestamps to strings for JSON
        results_json = results.copy()
        results_json["equity_curve"] = [
            (str(t), e) for t, e in results["equity_curve"]
        ]
        with open(output_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")

    return 0


def cmd_calibrate(args):
    """Run model calibration command."""
    from .config import load_config

    print(f"\n{'='*60}")
    print(f"QUANTITATIVE TRADING SYSTEM - CALIBRATE {args.model.upper()}")
    print(f"{'='*60}\n")

    config = load_config(args.config)

    # Load data
    print(f"Loading data from: {args.data}")
    data = pd.read_csv(args.data, index_col=0, parse_dates=True)

    if args.model == "heston":
        from .calibration import HestonCalibrator
        calibrator = HestonCalibrator()
        print("Calibrating Heston model...")

        # Need options data for Heston
        result = calibrator.calibrate(data)

        if result.success:
            print("\nCalibration successful!")
            print(f"  kappa (mean-reversion speed): {result.params.kappa:.4f}")
            print(f"  theta (long-term variance):   {result.params.theta:.4f}")
            print(f"  sigma (vol of vol):           {result.params.sigma:.4f}")
            print(f"  rho (correlation):            {result.params.rho:.4f}")
            print(f"  v0 (initial variance):        {result.params.v0:.4f}")
            print(f"  RMSE: {result.rmse:.6f}")
        else:
            print("Calibration failed!")

    elif args.model == "sabr":
        from .calibration import SABRCalibrator
        calibrator = SABRCalibrator(beta=config.model.sabr_beta)
        print("Calibrating SABR model...")

        result = calibrator.calibrate(data)

        if result.success:
            print("\nCalibration successful!")
            print(f"  alpha: {result.params.alpha:.4f}")
            print(f"  beta:  {result.params.beta:.4f}")
            print(f"  rho:   {result.params.rho:.4f}")
            print(f"  nu:    {result.params.nu:.4f}")
            print(f"  RMSE: {result.rmse:.6f}")

    elif args.model == "ou":
        from .calibration import OUCalibrator
        calibrator = OUCalibrator()
        print("Calibrating Ornstein-Uhlenbeck model...")

        # Use close prices
        prices = data["close"] if "close" in data.columns else data.iloc[:, 0]
        result = calibrator.fit(prices.values)

        print("\nCalibration successful!")
        print(f"  theta (mean level):     {result.theta:.4f}")
        print(f"  mu (mean-reversion):    {result.mu:.4f}")
        print(f"  sigma (volatility):     {result.sigma:.4f}")
        print(f"  Half-life:              {result.half_life:.1f} days")

    else:
        print(f"Unknown model: {args.model}")
        return 1

    return 0


def cmd_status(args):
    """Show system status."""
    from .trading_system import TradingSystem
    from .config import load_config

    print(f"\n{'='*60}")
    print("QUANTITATIVE TRADING SYSTEM - STATUS")
    print(f"{'='*60}\n")

    config = load_config(args.config)
    system = TradingSystem(config)
    system.initialize()

    status = system.get_status()

    print(f"Environment: {status['env']}")
    print(f"Initialized: {status['initialized']}")
    print(f"\nCapital:")
    print(f"  Cash:   ${status['cash']:,.2f}")
    print(f"  Equity: ${status['equity']:,.2f}")
    print(f"\nPositions: {status['n_positions']}")
    print(f"Total Trades: {status['n_trades']}")
    print(f"\nComponents:")
    print(f"  Calibrators: {', '.join(status['calibrators']) or 'None'}")
    print(f"  Signal Generators: {', '.join(status['signal_generators']) or 'None'}")

    return 0


def cmd_config(args):
    """Manage configuration."""
    from .config import Config, load_config

    if args.generate:
        config = Config()
        config.save(args.generate)
        print(f"Configuration template saved to: {args.generate}")
        return 0

    if args.show:
        config = load_config(args.config_file)
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Default: show config help
    print("Configuration management:")
    print("  --show          Show current configuration")
    print("  --generate FILE Generate configuration template")
    return 0


def cmd_demo(args):
    """Run a demonstration backtest."""
    print(f"\n{'='*60}")
    print("QUANTITATIVE TRADING SYSTEM - DEMO")
    print(f"{'='*60}\n")

    from .backtesting import (
        BacktestEngine,
        SyntheticDataHandler,
        HistoricDataFrameHandler,
        Portfolio,
        InstantExecutionHandler,
        MovingAverageCrossoverStrategy,
        MeanReversionStrategy,
        MomentumStrategy,
    )
    from queue import Queue

    events = Queue()

    # Determine data source
    if hasattr(args, 'symbol') and args.symbol:
        # Fetch real data from Yahoo Finance
        print(f"Running backtest with real market data for {args.symbol}...")
        start = args.start if hasattr(args, 'start') and args.start else "2023-01-01"
        end = args.end if hasattr(args, 'end') and args.end else datetime.now().strftime("%Y-%m-%d")

        data = fetch_yfinance_data(args.symbol, start, end)
        print(f"Loaded {len(data)} bars from {data.index[0].date()} to {data.index[-1].date()}")

        data_handler = HistoricDataFrameHandler(
            events_queue=events,
            symbol_list=[args.symbol],
            data=data,
        )
        symbol = args.symbol
    else:
        # Use synthetic data
        print("Running demonstration backtest with synthetic data...")
        symbol = "SPY"
        data_handler = SyntheticDataHandler(
            events_queue=events,
            symbol_list=[symbol],
            n_bars=252,  # One year
            start_price=100.0,
            drift=0.08,  # 8% annual drift
            volatility=0.18,  # 18% annual volatility
        )

    # Create portfolio
    portfolio = Portfolio(
        initial_capital=100000,
        max_position_pct=0.25,
    )

    # Create execution handler
    executor = InstantExecutionHandler(events_queue=events)

    # Get strategy parameters
    strategy_name = getattr(args, 'strategy', 'ma') or 'ma'

    # Parse strategy parameters
    fast = getattr(args, 'fast', None) or 5
    slow = getattr(args, 'slow', None) or 20
    lookback = getattr(args, 'lookback', None) or 15
    threshold = getattr(args, 'threshold', None) or 1.5

    # Create strategy based on selection
    if strategy_name == 'ma':
        print(f"Strategy: Moving Average Crossover ({fast}/{slow} day)")
        strategy = MovingAverageCrossoverStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            fast_window=fast,
            slow_window=slow,
        )
    elif strategy_name == 'meanrev':
        print(f"Strategy: Mean Reversion (lookback={lookback}, threshold={threshold})")
        strategy = MeanReversionStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            lookback=lookback,
            entry_threshold=threshold,
            exit_threshold=0.5,
        )
    elif strategy_name == 'momentum':
        print(f"Strategy: Momentum (lookback={lookback}, threshold={threshold}%)")
        strategy = MomentumStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            lookback=lookback,
            threshold=threshold / 100,  # Convert to decimal
        )
    else:
        print(f"Unknown strategy: {strategy_name}, using MA crossover")
        strategy = MovingAverageCrossoverStrategy(
            events_queue=events,
            data_handler=data_handler,
            portfolio=portfolio,
            fast_window=fast,
            slow_window=slow,
        )

    print()

    # Create backtest engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=executor,
    )

    # Run backtest
    results = engine.run()

    # Print results
    print(f"{'='*60}")
    print("DEMO BACKTEST RESULTS")
    print(f"{'='*60}")
    print(results.summary())

    return 0


def cmd_portfolio(args):
    """Run multi-asset portfolio simulation with optimal strategies."""
    print(f"\n{'='*70}")
    print("QUANTITATIVE TRADING SYSTEM - MULTI-ASSET PORTFOLIO SIMULATION")
    print(f"{'='*70}\n")

    from .backtesting import (
        BacktestEngine,
        HistoricDataFrameHandler,
        Portfolio,
        InstantExecutionHandler,
        MultiStrategyManager,
        get_optimal_strategy,
    )
    from queue import Queue

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    print(f"Portfolio: {', '.join(symbols)}")
    print(f"Capital: ${args.capital:,.0f}")

    # Date range
    start = args.start or "2023-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    print(f"Period: {start} to {end}")
    print()

    # Fetch data for all symbols
    print("Fetching market data...")
    all_data = {}
    min_date = None
    max_date = None

    for symbol in symbols:
        try:
            data = fetch_yfinance_data(symbol, start, end)
            all_data[symbol] = data
            print(f"  {symbol}: {len(data)} bars")

            if min_date is None or data.index[0] > min_date:
                min_date = data.index[0]
            if max_date is None or data.index[-1] < max_date:
                max_date = data.index[-1]
        except Exception as e:
            print(f"  {symbol}: Failed to fetch - {e}")

    if not all_data:
        print("Error: No data fetched for any symbol")
        return 1

    # Align dates across all symbols
    print(f"\nAligned date range: {min_date.date()} to {max_date.date()}")

    # Combine data into single DataFrame with symbol prefixes
    combined_data = pd.DataFrame(index=pd.date_range(min_date, max_date, freq='B'))

    for symbol, data in all_data.items():
        data = data[(data.index >= min_date) & (data.index <= max_date)]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                combined_data[f"{symbol}_{col}"] = data[col]

    # Forward fill missing data
    combined_data = combined_data.ffill().dropna()

    events = Queue()

    # Create data handler
    data_handler = HistoricDataFrameHandler(
        events_queue=events,
        symbol_list=list(all_data.keys()),
        data=combined_data,
    )

    # Create portfolio with position sizing per symbol
    max_position_pct = min(0.25, 1.0 / len(symbols))  # Equal weight cap
    portfolio = Portfolio(
        initial_capital=args.capital,
        max_position_pct=max_position_pct,
    )

    # Create execution handler
    executor = InstantExecutionHandler(events_queue=events)

    # Create multi-strategy manager
    strategy = MultiStrategyManager(
        events_queue=events,
        data_handler=data_handler,
        portfolio=portfolio,
    )

    # Assign optimal strategies to each symbol
    print("\nStrategy Assignments:")
    print("-" * 50)
    for symbol in all_data.keys():
        optimal = get_optimal_strategy(symbol)
        strategy.add_strategy(symbol, optimal["type"], **optimal["params"])
        print(f"  {symbol:6s} -> {optimal['type']:15s} {optimal['params']}")
    print()

    # Create backtest engine
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=executor,
    )

    # Run backtest
    print("Running portfolio simulation...")
    results = engine.run()

    # Print results
    print(f"\n{'='*70}")
    print("PORTFOLIO SIMULATION RESULTS")
    print(f"{'='*70}")
    print(results.summary())

    # Print per-symbol breakdown
    if results.trade_history:
        print("\nTRADE BREAKDOWN BY SYMBOL")
        print("-" * 50)
        symbol_stats = {}
        for trade in results.trade_history:
            sym = trade.get('symbol', 'Unknown')
            if sym not in symbol_stats:
                symbol_stats[sym] = {'trades': 0, 'wins': 0, 'pnl': 0}
            symbol_stats[sym]['trades'] += 1
            pnl = trade.get('pnl', 0)
            symbol_stats[sym]['pnl'] += pnl
            if pnl > 0:
                symbol_stats[sym]['wins'] += 1

        for sym, stats in sorted(symbol_stats.items()):
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            print(f"  {sym:6s}: {stats['trades']:3d} trades, {win_rate:5.1f}% win rate, ${stats['pnl']:>10,.2f} P&L")

    return 0


def cmd_scan(args):
    """Scan stocks by sector and rank by confidence."""
    print(f"\n{'='*70}")
    print("QUANTITATIVE TRADING SYSTEM - SECTOR SCANNER")
    print(f"{'='*70}\n")

    from .backtesting import (
        Sector,
        SECTOR_STOCKS,
        get_sector,
        get_sector_strategy,
        ConfidenceCalculator,
        calculate_position_size,
    )

    # Determine which sectors to scan
    if args.sector:
        try:
            sectors = [Sector(args.sector.lower())]
        except ValueError:
            print(f"Unknown sector: {args.sector}")
            print(f"Available: {[s.value for s in Sector]}")
            return 1
    else:
        # All sectors
        sectors = list(Sector)

    # Date range for data
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    start = args.start or (datetime.now() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")

    print(f"Scanning {len(sectors)} sector(s)")
    print(f"Data period: {start} to {end}")
    print(f"Top {args.top} stocks per sector\n")

    # Initialize confidence calculator
    calc = ConfidenceCalculator(lookback_days=60)

    all_results = []

    for sector in sectors:
        stocks = SECTOR_STOCKS.get(sector, [])
        if not stocks:
            continue

        print(f"\n{sector.value.upper().replace('_', ' ')} ({len(stocks)} stocks)")
        print("-" * 60)

        sector_results = []

        for symbol in stocks[:args.limit]:  # Limit per sector to avoid rate limits
            try:
                data = fetch_yfinance_data(symbol, start, end)
                if len(data) < 30:
                    continue

                prices = data['Close'].values

                # Get sector strategy
                strategy_config = get_sector_strategy(symbol)

                # Calculate confidence
                metrics = calc.calculate(symbol, prices, signal_strength=0.5)

                # Calculate suggested position size
                base_alloc = 1.0 / max(10, len(stocks))
                position_size = calculate_position_size(
                    metrics.confidence,
                    base_alloc,
                    min_allocation=0.02,
                    max_allocation=0.10,
                )

                sector_results.append({
                    'symbol': symbol,
                    'sector': sector.value,
                    'strategy': strategy_config['type'],
                    'confidence': metrics.confidence,
                    'momentum': metrics.momentum_strength,
                    'trend': metrics.trend_alignment,
                    'half_life': metrics.half_life_days,
                    'volatility': metrics.realized_volatility,
                    'position_size': position_size,
                    'current_price': prices[-1],
                })

            except Exception as e:
                if args.verbose:
                    print(f"  {symbol}: Error - {e}")

        # Sort by confidence and show top N
        sector_results.sort(key=lambda x: x['confidence'], reverse=True)

        for i, r in enumerate(sector_results[:args.top]):
            direction = "↑" if r['momentum'] > 0.05 else ("↓" if r['momentum'] < -0.05 else "→")
            hl = f"{r['half_life']:.0f}d" if r['half_life'] < 1000 else "N/A"
            print(
                f"  {i+1}. {r['symbol']:6s} | "
                f"Conf: {r['confidence']:.2f} | "
                f"Mom: {r['momentum']:+.2f} {direction} | "
                f"HL: {hl:>5s} | "
                f"Vol: {r['volatility']*100:.1f}% | "
                f"Size: {r['position_size']*100:.1f}% | "
                f"${r['current_price']:.2f}"
            )

        all_results.extend(sector_results)

    # Show overall top picks across all sectors
    if len(sectors) > 1 and all_results:
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        print(f"\n{'='*70}")
        print(f"TOP {min(args.top, len(all_results))} OVERALL PICKS")
        print("=" * 70)

        for i, r in enumerate(all_results[:args.top]):
            direction = "↑" if r['momentum'] > 0.05 else ("↓" if r['momentum'] < -0.05 else "→")
            print(
                f"  {i+1}. {r['symbol']:6s} ({r['sector']:12s}) | "
                f"Conf: {r['confidence']:.2f} | "
                f"Strategy: {r['strategy']:12s} | "
                f"Size: {r['position_size']*100:.1f}%"
            )

    return 0


def cmd_sector_portfolio(args):
    """Run sector-diversified portfolio with confidence-weighted sizing."""
    print(f"\n{'='*70}")
    print("QUANTITATIVE TRADING SYSTEM - SECTOR-DIVERSIFIED PORTFOLIO")
    print(f"{'='*70}\n")

    from .backtesting import (
        BacktestEngine,
        HistoricDataFrameHandler,
        Portfolio,
        InstantExecutionHandler,
        MultiStrategyManager,
        Sector,
        SECTOR_STOCKS,
        get_sector,
        get_sector_strategy,
        ConfidenceCalculator,
        calculate_position_size,
        SectorOptimizationResults,
    )
    from queue import Queue
    from pathlib import Path

    # Load optimization results if requested
    optimization_results = None
    if args.use_optimized:
        cache_dir = Path(args.optimized_cache) if args.optimized_cache else Path(".optimization_cache")
        cache_file = cache_dir / "sector_optimization_latest.json"

        if cache_file.exists():
            try:
                optimization_results = SectorOptimizationResults.load(cache_file)
                print(f"Loaded optimization results from: {cache_file}")
                print(f"Optimization date: {optimization_results.optimization_date}")
            except Exception as e:
                print(f"Warning: Failed to load optimization results: {e}")
                print("Continuing without optimization data...")
        else:
            print(f"Warning: No optimization cache found at {cache_file}")
            print("Run 'optimize-sectors' first or continue without optimization...")

    # Parse sectors
    if args.sectors:
        sector_names = [s.strip().lower() for s in args.sectors.split(",")]
        sectors = []
        for name in sector_names:
            try:
                sectors.append(Sector(name))
            except ValueError:
                print(f"Unknown sector: {name}")
                return 1
    else:
        # Default diversified mix - all 12 non-ETF sectors for ~100 stock portfolio
        sectors = [
            Sector.TECHNOLOGY,
            Sector.FINANCIALS,
            Sector.HEALTHCARE,
            Sector.CONSUMER_DISCRETIONARY,
            Sector.CONSUMER_STAPLES,
            Sector.ENERGY,
            Sector.INDUSTRIALS,
            Sector.MATERIALS,
            Sector.UTILITIES,
            Sector.REAL_ESTATE,
            Sector.COMMUNICATION,
            # Exclude ETF_INDEX and ETF_SECTOR for individual stock portfolio
        ]

    print(f"Sectors: {', '.join(s.value for s in sectors)}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Stocks per sector: {args.per_sector}")
    print(f"Using optimization: {'Yes' if optimization_results else 'No'}")

    # Date range
    start = args.start or "2023-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")
    print(f"Period: {start} to {end}")
    print()

    # First pass: scan all stocks and rank by confidence
    calc = ConfidenceCalculator(lookback_days=60, optimization_results=optimization_results)
    scan_start = (pd.to_datetime(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    selected_stocks = []
    print("Scanning sectors for top opportunities...")

    for sector in sectors:
        scan_limit = getattr(args, 'scan_limit', 30)
        stocks = SECTOR_STOCKS.get(sector, [])[:scan_limit]
        sector_candidates = []

        # Get strategy config - use optimized if available
        if optimization_results:
            best_algo, best_params = optimization_results.get_best_algorithm(sector)
            strategy_config = {
                'type': best_algo,
                'params': best_params,
                'sector': sector.value,
            }
        else:
            strategy_config = None  # Will use per-symbol default

        for symbol in stocks:
            try:
                data = fetch_yfinance_data(symbol, scan_start, end)
                if len(data) < 30:
                    continue

                prices = data['Close'].values

                # Use optimized strategy for sector, or default per-symbol
                if strategy_config:
                    sym_strategy = strategy_config
                    algorithm = strategy_config['type']
                else:
                    sym_strategy = get_sector_strategy(symbol)
                    algorithm = sym_strategy['type']

                metrics = calc.calculate(symbol, prices, signal_strength=0.5, algorithm=algorithm)

                sector_candidates.append({
                    'symbol': symbol,
                    'sector': sector,
                    'strategy': sym_strategy,
                    'confidence': metrics.confidence,
                    'fitness': metrics.sector_algorithm_fitness,
                    'data': data,
                })
            except:
                pass

        # Select top N from each sector
        sector_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        selected = sector_candidates[:args.per_sector]
        selected_stocks.extend(selected)

        if selected:
            fitness_info = f" (fitness: {selected[0]['fitness']:.2f})" if optimization_results else ""
            print(f"  {sector.value}: {', '.join(s['symbol'] for s in selected)}{fitness_info}")

    if not selected_stocks:
        print("Error: No stocks selected")
        return 1

    symbols = [s['symbol'] for s in selected_stocks]
    print(f"\nSelected {len(symbols)} stocks across {len(sectors)} sectors")

    # Fetch full data for selected stocks
    print("\nFetching detailed market data...")
    all_data = {}
    min_date = None
    max_date = None

    for stock_info in selected_stocks:
        symbol = stock_info['symbol']
        data = stock_info['data']
        data = data[(data.index >= start) & (data.index <= end)]

        if len(data) < 20:
            continue

        all_data[symbol] = data

        if min_date is None or data.index[0] > min_date:
            min_date = data.index[0]
        if max_date is None or data.index[-1] < max_date:
            max_date = data.index[-1]

    print(f"Aligned date range: {min_date.date()} to {max_date.date()}")

    # Combine data
    combined_data = pd.DataFrame(index=pd.date_range(min_date, max_date, freq='B'))

    for symbol, data in all_data.items():
        data = data[(data.index >= min_date) & (data.index <= max_date)]
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                combined_data[f"{symbol}_{col}"] = data[col]

    combined_data = combined_data.ffill().dropna()

    events = Queue()

    # Create data handler
    data_handler = HistoricDataFrameHandler(
        events_queue=events,
        symbol_list=list(all_data.keys()),
        data=combined_data,
    )

    # Create portfolio
    max_position_pct = min(0.15, 1.5 / len(all_data))
    portfolio = Portfolio(
        initial_capital=args.capital,
        max_position_pct=max_position_pct,
    )

    # Create execution handler
    executor = InstantExecutionHandler(events_queue=events)

    # Create multi-strategy manager with sector-specific strategies
    strategy = MultiStrategyManager(
        events_queue=events,
        data_handler=data_handler,
        portfolio=portfolio,
    )

    print("\nStrategy Assignments:")
    print("-" * 60)

    for stock_info in selected_stocks:
        symbol = stock_info['symbol']
        if symbol not in all_data:
            continue
        strat = stock_info['strategy']
        strategy.add_strategy(symbol, strat['type'], **strat['params'])
        print(f"  {symbol:6s} ({stock_info['sector'].value:12s}) -> {strat['type']:12s}")

    print()

    # Create and run backtest
    engine = BacktestEngine(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=executor,
    )

    print("Running sector-diversified portfolio simulation...")
    results = engine.run()

    # Print results
    print(f"\n{'='*70}")
    print("SECTOR PORTFOLIO RESULTS")
    print(f"{'='*70}")
    print(results.summary())

    # Sector breakdown
    if results.trade_history:
        print("\nPERFORMANCE BY SECTOR")
        print("-" * 60)

        sector_stats = {}
        for trade in results.trade_history:
            sym = trade.get('symbol', 'Unknown')
            sector = get_sector(sym)
            if sector not in sector_stats:
                sector_stats[sector] = {'trades': 0, 'wins': 0, 'pnl': 0}
            sector_stats[sector]['trades'] += 1
            pnl = trade.get('pnl', 0)
            sector_stats[sector]['pnl'] += pnl
            if pnl > 0:
                sector_stats[sector]['wins'] += 1

        for sector, stats in sorted(sector_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            print(f"  {sector.value:20s}: {stats['trades']:3d} trades, {win_rate:5.1f}% win rate, ${stats['pnl']:>10,.2f} P&L")

    return 0


def cmd_optimize_sectors(args):
    """Run sector-algorithm optimization to find best pairings."""
    print(f"\n{'='*70}")
    print("QUANTITATIVE TRADING SYSTEM - SECTOR-ALGORITHM OPTIMIZATION")
    print(f"{'='*70}\n")

    from .backtesting import (
        Sector,
        SectorAlgorithmOptimizer,
        print_optimization_results,
    )
    from pathlib import Path

    # Parse sectors
    if args.sectors:
        sector_names = [s.strip().lower() for s in args.sectors.split(",")]
        sectors = []
        for name in sector_names:
            try:
                sectors.append(Sector(name))
            except ValueError:
                print(f"Unknown sector: {name}")
                return 1
    else:
        sectors = None  # Will use all non-ETF sectors

    # Parse algorithms
    if args.algorithms:
        algorithms = [a.strip().lower() for a in args.algorithms.split(",")]
    else:
        algorithms = None  # Will use all algorithms

    # Cache directory
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(".optimization_cache")

    print(f"Stocks per sector: {args.n_stocks}")
    print(f"Backtest days: {args.days}")
    print(f"Parameter search: {'enabled' if not args.no_param_search else 'disabled'}")
    print(f"Cache directory: {cache_dir}")

    if args.start:
        print(f"Start date: {args.start}")
    if args.end:
        print(f"End date: {args.end}")
    print()

    # Create optimizer
    optimizer = SectorAlgorithmOptimizer(
        n_stocks_per_sector=args.n_stocks,
        backtest_days=args.days,
        optimize_params=not args.no_param_search,
        cache_dir=cache_dir,
    )

    # Check for cached results first
    if not args.force:
        cached = optimizer.load_cached_results()
        if cached:
            print("Using cached optimization results (use --force to re-run)")
            print_optimization_results(cached)

            if args.output:
                output_path = Path(args.output)
                cached.save(output_path)
                print(f"\nResults saved to: {output_path}")

            return 0

    # Run optimization
    print("Running optimization (this may take several minutes)...")
    print("-" * 70)

    results = optimizer.run_optimization(
        sectors=sectors,
        algorithms=algorithms,
        start_date=args.start,
        end_date=args.end,
    )

    # Print results
    print_optimization_results(results)

    # Save to custom output if specified
    if args.output:
        output_path = Path(args.output)
        results.save(output_path)
        print(f"\nResults saved to: {output_path}")

    return 0


def generate_synthetic_data(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Generate synthetic OHLCV data for demonstration."""
    from datetime import timedelta

    if start:
        start_date = pd.to_datetime(start)
    else:
        start_date = pd.to_datetime("2023-01-01")

    if end:
        end_date = pd.to_datetime(end)
    else:
        end_date = pd.to_datetime("2023-12-31")

    dates = pd.date_range(start=start_date, end=end_date, freq="B")  # Business days
    n = len(dates)

    np.random.seed(42)

    # Generate price path with drift
    returns = np.random.normal(0.0003, 0.015, n)  # ~7.5% annual return, 24% vol
    prices = 100 * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        "high": prices * (1 + np.abs(np.random.uniform(0, 0.02, n))),
        "low": prices * (1 - np.abs(np.random.uniform(0, 0.02, n))),
        "close": prices,
        "volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)

    # Ensure OHLC consistency
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="quant-trading",
        description="Quantitative Trading System - A sophisticated trading platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with synthetic data
  quant-trading demo

  # Run demo with real market data (Yahoo Finance)
  quant-trading demo --symbol AAPL --start 2023-01-01 --end 2024-01-01

  # Run backtest with real market data
  quant-trading backtest --symbol SPY --start 2023-01-01 --end 2024-01-01

  # Run multi-asset portfolio simulation (optimal strategies auto-assigned)
  quant-trading portfolio --symbols AAPL,MSFT,NVDA,GOOGL,META

  # Scan all sectors for top opportunities
  quant-trading scan --top 5

  # Scan specific sector
  quant-trading scan --sector technology --top 10

  # Run sector-diversified portfolio with confidence weighting
  quant-trading sector-portfolio --per-sector 3 --capital 100000

  # Run backtest with local data file
  quant-trading backtest --data prices.csv --start 2023-01-01 --end 2023-12-31

  # Calibrate Heston model
  quant-trading calibrate --model heston --data options.csv

  # Show system status
  quant-trading status

  # Generate config template
  quant-trading config --generate config.json
        """
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    data_group = backtest_parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--symbol", help="Ticker symbol (fetches from Yahoo Finance)")
    data_group.add_argument("--data", "-d", help="Data file (CSV/Parquet)")
    backtest_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--config", "-c", help="Config file")
    backtest_parser.add_argument("--capital", type=float, help="Initial capital")
    backtest_parser.add_argument("--output", "-o", help="Output file for results")
    backtest_parser.add_argument("--monte-carlo", "-m", type=int, metavar="N",
                                  help="Run Monte Carlo with N simulations")

    # Calibrate command
    calibrate_parser = subparsers.add_parser("calibrate", help="Calibrate models")
    calibrate_parser.add_argument("--model", "-m", required=True,
                                   choices=["heston", "sabr", "ou"],
                                   help="Model to calibrate")
    calibrate_parser.add_argument("--data", "-d", required=True, help="Data file")
    calibrate_parser.add_argument("--config", "-c", help="Config file")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--config", "-c", help="Config file")

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--generate", metavar="FILE", help="Generate config template")
    config_parser.add_argument("--config-file", "-c", help="Config file to show")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration backtest")
    demo_parser.add_argument("--symbol", help="Ticker symbol for real data (default: synthetic)")
    demo_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    demo_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    demo_parser.add_argument("--strategy", choices=["ma", "meanrev", "momentum"],
                             default="ma", help="Strategy: ma (Moving Average), meanrev (Mean Reversion), momentum")
    demo_parser.add_argument("--fast", type=int, default=5, help="Fast MA window (default: 5)")
    demo_parser.add_argument("--slow", type=int, default=20, help="Slow MA window (default: 20)")
    demo_parser.add_argument("--lookback", type=int, default=15, help="Lookback period for meanrev/momentum (default: 15)")
    demo_parser.add_argument("--threshold", type=float, default=1.5, help="Entry threshold (z-score for meanrev, %% for momentum)")

    # Portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Run multi-asset portfolio simulation")
    portfolio_parser.add_argument("--symbols", required=True,
                                   help="Comma-separated list of ticker symbols (e.g., AAPL,MSFT,NVDA)")
    portfolio_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    portfolio_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    portfolio_parser.add_argument("--capital", type=float, default=100000, help="Initial capital (default: 100000)")

    # Scan command - scan sectors for opportunities
    scan_parser = subparsers.add_parser("scan", help="Scan stocks by sector and rank by confidence")
    scan_parser.add_argument("--sector", help="Specific sector to scan (e.g., technology, financials)")
    scan_parser.add_argument("--start", "-s", help="Start date for data (default: 90 days ago)")
    scan_parser.add_argument("--end", "-e", help="End date for data (default: today)")
    scan_parser.add_argument("--top", type=int, default=5, help="Top N stocks per sector (default: 5)")
    scan_parser.add_argument("--limit", type=int, default=15, help="Max stocks to scan per sector (default: 15)")
    scan_parser.add_argument("--verbose", "-v", action="store_true", help="Show errors")

    # Sector portfolio command - diversified sector portfolio
    sector_parser = subparsers.add_parser("sector-portfolio", help="Run sector-diversified portfolio with confidence weighting")
    sector_parser.add_argument("--sectors", help="Comma-separated sectors (default: all major sectors)")
    sector_parser.add_argument("--per-sector", type=int, default=8, help="Stocks per sector (default: 8)")
    sector_parser.add_argument("--scan-limit", type=int, default=30, help="Max stocks to scan per sector (default: 30)")
    sector_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    sector_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    sector_parser.add_argument("--capital", type=float, default=100000, help="Initial capital (default: 100000)")
    sector_parser.add_argument("--use-optimized", action="store_true",
                                help="Use optimized sector-algorithm pairings from cache")
    sector_parser.add_argument("--optimized-cache", default=".optimization_cache",
                                help="Directory containing optimization results (default: .optimization_cache)")

    # Optimize sectors command - find optimal sector-algorithm pairings
    optimize_parser = subparsers.add_parser("optimize-sectors",
                                             help="Run sector-algorithm optimization to find best pairings")
    optimize_parser.add_argument("--sectors", help="Comma-separated sectors to optimize (default: all)")
    optimize_parser.add_argument("--algorithms", help="Comma-separated algorithms to test (default: all)")
    optimize_parser.add_argument("--n-stocks", type=int, default=10,
                                  help="Stocks per sector to test (default: 10)")
    optimize_parser.add_argument("--days", type=int, default=252,
                                  help="Backtest days (default: 252)")
    optimize_parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    optimize_parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    optimize_parser.add_argument("--no-param-search", action="store_true",
                                  help="Skip parameter optimization, use defaults")
    optimize_parser.add_argument("--cache-dir", default=".optimization_cache",
                                  help="Cache directory (default: .optimization_cache)")
    optimize_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    optimize_parser.add_argument("--force", action="store_true",
                                  help="Force re-optimization even if cache exists")

    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)

    if not args.command:
        parser.print_help()
        return 0

    try:
        if args.command == "backtest":
            return cmd_backtest(args)
        elif args.command == "calibrate":
            return cmd_calibrate(args)
        elif args.command == "status":
            return cmd_status(args)
        elif args.command == "config":
            return cmd_config(args)
        elif args.command == "demo":
            return cmd_demo(args)
        elif args.command == "portfolio":
            return cmd_portfolio(args)
        elif args.command == "scan":
            return cmd_scan(args)
        elif args.command == "sector-portfolio":
            return cmd_sector_portfolio(args)
        elif args.command == "optimize-sectors":
            return cmd_optimize_sectors(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        if args.debug:
            raise
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
