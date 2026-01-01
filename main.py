#!/usr/bin/env python3
"""
AMM Strategy Backtest
Polymarket BTC/SOL 15-min Markets

Main entry point for running the backtest
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from config.risk_params import RiskParams
from config.btc_risk_params import BTCRiskParams
from src.data_collector import DataCollector
from src.backtest_engine import BacktestEngine, run_backtest
from src.btc_backtest_engine import BTCBacktestEngine, run_btc_backtest
from src.metrics import PerformanceMetrics
from src.visualizer import BacktestVisualizer, generate_report
from src.market_analyzer import MarketAnalyzer
from src.spread_calculator import SpreadCalculator
from src.log_processor import (
    load_log_file,
    merge_multiple_logs,
    prepare_backtest_data,
    print_analysis_summary,
    generate_synthetic_token_prices,
)
from src.gabagool import (
    GabagoolConfig,
    GabagoolBot,
    GabagoolBacktest,
    run_gabagool_backtest,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    AGGRESSIVE_CONFIG,
)
from src.volatility_arb import (
    VolatilityArbBot,
    BotConfig,
    ExecutionMode,
    run_bot as run_volatility_bot,
    VolatilityArbBacktest,
    run_volatility_backtest,
    PAPER_TRADING_CONFIG,
    CONSERVATIVE_CONFIG as VOL_CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG as VOL_AGGRESSIVE_CONFIG,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(args):
    """Collect market data from Polymarket API."""
    logger.info("Starting data collection...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    collector = DataCollector()

    try:
        # Fetch markets
        markets_df = collector.fetch_sol_15min_markets(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            save_path=f"{settings.DATA_RAW_PATH}/sol_markets.csv"
        )

        logger.info(f"Found {len(markets_df)} markets")

        if not markets_df.empty and args.fetch_prices:
            # Fetch prices
            prices_df = collector.fetch_all_prices(
                markets_df,
                save_dir=f"{settings.DATA_RAW_PATH}/price_history"
            )

            if not prices_df.empty:
                # Save consolidated prices
                Path(settings.DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
                prices_df.to_parquet(
                    f"{settings.DATA_PROCESSED_PATH}/all_prices.parquet",
                    index=False
                )
                logger.info(f"Saved {len(prices_df)} price records")

        logger.info("Data collection complete!")

    finally:
        collector.close()


def run_analysis(args):
    """Run exploratory analysis on collected data."""
    import pandas as pd

    logger.info("Running analysis...")

    # Load data
    markets_path = args.markets or f"{settings.DATA_RAW_PATH}/sol_markets.csv"
    prices_path = args.prices or f"{settings.DATA_PROCESSED_PATH}/all_prices.parquet"

    markets_df = pd.read_csv(markets_path)

    if Path(prices_path).exists():
        if prices_path.endswith('.parquet'):
            prices_df = pd.read_parquet(prices_path)
        else:
            prices_df = pd.read_csv(prices_path)
    else:
        prices_df = None

    # Market analysis
    analyzer = MarketAnalyzer(markets_df, prices_df)
    summary = analyzer.get_market_summary()

    print("\n" + "=" * 60)
    print("MARKET ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- Outcomes ---")
    outcomes = summary.get('outcomes', {})
    print(f"Total Markets: {outcomes.get('total_markets', 0)}")
    print(f"Up: {outcomes.get('pct_up', 0):.1f}%")
    print(f"Down: {outcomes.get('pct_down', 0):.1f}%")

    print("\n--- Liquidity ---")
    liquidity = summary.get('liquidity', {})
    print(f"Avg Volume: ${liquidity.get('avg_volume_per_market', 0):.2f}")
    print(f"Median Volume: ${liquidity.get('median_volume_per_market', 0):.2f}")

    print("\n--- Temporal ---")
    temporal = summary.get('temporal', {})
    print(f"Total Days: {temporal.get('total_days', 0)}")
    print(f"Avg Markets/Day: {temporal.get('avg_markets_per_day', 0):.1f}")

    # Spread analysis
    if prices_df is not None:
        spread_calc = SpreadCalculator(prices_df)
        spread_stats = spread_calc.calculate_spread_stats()

        print("\n--- Spread Statistics ---")
        print(f"Mean Spread: {spread_stats.get('mean', 0):.4f}")
        print(f"Median Spread: {spread_stats.get('median', 0):.4f}")
        print(f"% Below 0.98: {spread_stats.get('pct_below_98', 0):.1f}%")
        print(f"% Below 0.97: {spread_stats.get('pct_below_97', 0):.1f}%")

        print("\n--- Best Trading Hours ---")
        best_hours = spread_calc.get_best_hours(5)
        for hour, spread in best_hours.items():
            print(f"  Hour {hour}: avg spread = {spread:.4f}")

    print("\n" + "=" * 60)


def run_backtest_cmd(args):
    """Run the backtest."""
    import pandas as pd

    logger.info("Starting backtest...")

    # Load data
    markets_path = args.markets or f"{settings.DATA_RAW_PATH}/sol_markets.csv"
    prices_path = args.prices or f"{settings.DATA_PROCESSED_PATH}/all_prices.parquet"

    if not Path(markets_path).exists():
        logger.error(f"Markets file not found: {markets_path}")
        logger.info("Run with --collect first to gather data")
        return

    markets_df = pd.read_csv(markets_path)

    if Path(prices_path).exists():
        if prices_path.endswith('.parquet'):
            prices_df = pd.read_parquet(prices_path)
        else:
            prices_df = pd.read_csv(prices_path)
    else:
        logger.warning("No price data found, using simulated prices")
        # Create simulated prices
        prices_df = _create_simulated_prices(markets_df)

    # Setup risk params
    risk_params = RiskParams()

    if args.spread_threshold:
        risk_params.MIN_SPREAD_TO_ENTER = args.spread_threshold

    if args.max_exposure:
        risk_params.MAX_TOTAL_EXPOSURE = args.max_exposure

    # Run backtest
    engine = BacktestEngine(
        initial_capital=args.capital,
        risk_params=risk_params
    )

    results = engine.run(markets_df, prices_df, verbose=True)

    # Print summary
    metrics = PerformanceMetrics()
    report = metrics.generate_summary_report(results['metrics'])
    print(report)

    # Generate visual report
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"{settings.DATA_RESULTS_PATH}/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    generate_report(results, output_dir)
    logger.info(f"Report saved to {output_dir}")

    # Save results
    results['trades'].to_csv(f"{output_dir}/trades.csv", index=False)
    results['portfolio_history'].to_csv(f"{output_dir}/portfolio_history.csv", index=False)


def _create_simulated_prices(markets_df):
    """Create simulated price data for testing."""
    import pandas as pd
    import numpy as np

    all_prices = []

    for _, market in markets_df.iterrows():
        market_id = market['market_id']
        start = pd.to_datetime(market['start_time'])
        end = pd.to_datetime(market['end_time'])

        # Generate 15 price points
        timestamps = pd.date_range(start, end, periods=15)

        # Random walk around 0.50
        base_yes = 0.50
        base_no = 0.50

        for ts in timestamps:
            # Add some randomness
            price_yes = np.clip(base_yes + np.random.normal(0, 0.02), 0.40, 0.60)
            price_no = np.clip(base_no + np.random.normal(0, 0.02), 0.40, 0.60)

            # Ensure some spread opportunity
            total = price_yes + price_no
            if np.random.random() > 0.7:  # 30% chance of opportunity
                adjustment = np.random.uniform(0.01, 0.04)
                price_yes -= adjustment / 2
                price_no -= adjustment / 2

            all_prices.append({
                'timestamp': ts,
                'market_id': market_id,
                'price_yes': price_yes,
                'price_no': price_no,
                'spread': price_yes + price_no - 1.0,
            })

    return pd.DataFrame(all_prices)


def main():
    parser = argparse.ArgumentParser(
        description='AMM Strategy Backtest for Polymarket BTC/SOL 15-min Markets'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect market data')
    collect_parser.add_argument('--days', type=int, default=90, help='Days of history')
    collect_parser.add_argument('--fetch-prices', action='store_true', help='Also fetch price history')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on data')
    analyze_parser.add_argument('--markets', type=str, help='Path to markets CSV')
    analyze_parser.add_argument('--prices', type=str, help='Path to prices file')

    # Backtest command (SOL delta-neutral)
    backtest_parser = subparsers.add_parser('backtest', help='Run SOL delta-neutral backtest')
    backtest_parser.add_argument('--markets', type=str, help='Path to markets CSV')
    backtest_parser.add_argument('--prices', type=str, help='Path to prices file')
    backtest_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')
    backtest_parser.add_argument('--spread-threshold', type=float, help='Min spread to enter')
    backtest_parser.add_argument('--max-exposure', type=float, help='Max portfolio exposure')
    backtest_parser.add_argument('--output', type=str, help='Output directory')

    # Quick test command (SOL)
    test_parser = subparsers.add_parser('test', help='Run quick test with simulated SOL data')
    test_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')

    # BTC Backtest command (probabilistic arbitrage)
    btc_backtest_parser = subparsers.add_parser('btc-backtest', help='Run BTC probabilistic arbitrage backtest')
    btc_backtest_parser.add_argument('--logs', type=str, nargs='+', help='Path(s) to log files (JSON/GZIP)')
    btc_backtest_parser.add_argument('--log-dir', type=str, help='Directory with log files')
    btc_backtest_parser.add_argument('--capital', type=float, default=100, help='Capital per market')
    btc_backtest_parser.add_argument('--min-opp', type=float, default=0.05, help='Min opportunity threshold (0-1)')
    btc_backtest_parser.add_argument('--output', type=str, help='Output directory')

    # BTC Test command (simulated data)
    btc_test_parser = subparsers.add_parser('btc-test', help='Run BTC backtest with simulated data')
    btc_test_parser.add_argument('--capital', type=float, default=100, help='Capital per market')
    btc_test_parser.add_argument('--markets', type=int, default=50, help='Number of markets to simulate')
    btc_test_parser.add_argument('--volatility', type=float, default=0.002, help='Price volatility')

    # BTC Analyze command
    btc_analyze_parser = subparsers.add_parser('btc-analyze', help='Analyze BTC log files')
    btc_analyze_parser.add_argument('--logs', type=str, nargs='+', help='Path(s) to log files')
    btc_analyze_parser.add_argument('--log-dir', type=str, help='Directory with log files')

    # === GABAGOOL SPREAD CAPTURE COMMANDS ===

    # Gabagool Backtest command
    gab_backtest_parser = subparsers.add_parser('gabagool-backtest', help='Run Gabagool spread capture backtest')
    gab_backtest_parser.add_argument('--markets', type=int, default=100, help='Number of markets to simulate')
    gab_backtest_parser.add_argument('--min-spread', type=float, default=0.02, help='Min spread threshold (0.02=2%)')
    gab_backtest_parser.add_argument('--order-size', type=float, default=15.0, help='Order size in USD')
    gab_backtest_parser.add_argument('--max-per-market', type=float, default=500.0, help='Max USD per market')
    gab_backtest_parser.add_argument('--preset', type=str, choices=['conservative', 'moderate', 'aggressive'], help='Use preset config')
    gab_backtest_parser.add_argument('--no-save', action='store_true', help='Do not save results')

    # Gabagool Bot command (paper trading)
    gab_bot_parser = subparsers.add_parser('gabagool-bot', help='Run Gabagool bot (paper trading)')
    gab_bot_parser.add_argument('--live', action='store_true', help='Enable live trading (requires API keys)')
    gab_bot_parser.add_argument('--min-spread', type=float, default=0.02, help='Min spread threshold')
    gab_bot_parser.add_argument('--order-size', type=float, default=15.0, help='Order size in USD')
    gab_bot_parser.add_argument('--max-per-market', type=float, default=500.0, help='Max USD per market')
    gab_bot_parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH'], help='Assets to trade')
    gab_bot_parser.add_argument('--preset', type=str, choices=['conservative', 'moderate', 'aggressive'], help='Use preset config')

    # Gabagool Quick Test
    gab_test_parser = subparsers.add_parser('gabagool-test', help='Quick test of Gabagool strategy')
    gab_test_parser.add_argument('--markets', type=int, default=50, help='Number of markets')

    # === VOLATILITY ARBITRAGE BOT COMMANDS ===

    # Vol-arb Bot command (paper trading)
    vol_bot_parser = subparsers.add_parser('vol-bot', help='Run volatility arbitrage bot')
    vol_bot_parser.add_argument('--live', action='store_true', help='Enable live trading')
    vol_bot_parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    vol_bot_parser.add_argument('--min-edge', type=float, default=3.0, help='Min edge % to trade')
    vol_bot_parser.add_argument('--risk', type=str, choices=['conservative', 'moderate', 'aggressive'], default='moderate', help='Risk level')

    # Vol-arb Test command
    vol_test_parser = subparsers.add_parser('vol-test', help='Quick test of volatility arb strategy')
    vol_test_parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    vol_test_parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')

    # Vol-arb Backtest command (with real data)
    vol_backtest_parser = subparsers.add_parser('vol-backtest', help='Run volatility arb backtest with real data')
    vol_backtest_parser.add_argument('--data', type=str, default='data/raw', help='Path to data file or directory')
    vol_backtest_parser.add_argument('--balance', type=float, default=1000.0, help='Initial balance')
    vol_backtest_parser.add_argument('--min-edge', type=float, default=3.0, help='Min edge % to trade')
    vol_backtest_parser.add_argument('--output', type=str, help='Output directory for results')

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'backtest':
        run_backtest_cmd(args)
    elif args.command == 'test':
        run_test(args)
    elif args.command == 'btc-backtest':
        run_btc_backtest_cmd(args)
    elif args.command == 'btc-test':
        run_btc_test(args)
    elif args.command == 'btc-analyze':
        run_btc_analyze(args)
    elif args.command == 'gabagool-backtest':
        run_gabagool_backtest_cmd(args)
    elif args.command == 'gabagool-bot':
        run_gabagool_bot_cmd(args)
    elif args.command == 'gabagool-test':
        run_gabagool_test_cmd(args)
    elif args.command == 'vol-bot':
        run_vol_bot_cmd(args)
    elif args.command == 'vol-test':
        run_vol_test_cmd(args)
    elif args.command == 'vol-backtest':
        run_vol_backtest_cmd(args)
    else:
        parser.print_help()


def run_test(args):
    """Run quick test with simulated data."""
    import pandas as pd
    import numpy as np

    logger.info("Running quick test with simulated data...")

    # Create simulated markets
    np.random.seed(42)
    n_markets = 100

    markets = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(n_markets):
        start = base_time + timedelta(minutes=15 * i)
        end = start + timedelta(minutes=15)

        markets.append({
            'market_id': f'market_{i}',
            'condition_id': f'cond_{i}',
            'question': f'Solana Up or Down - Test {i}',
            'start_time': start,
            'end_time': end,
            'outcome': 'Up' if np.random.random() > 0.5 else 'Down',
            'yes_token_id': f'yes_{i}',
            'no_token_id': f'no_{i}',
            'volume': np.random.uniform(500, 5000),
            'liquidity': np.random.uniform(1000, 10000),
        })

    markets_df = pd.DataFrame(markets)
    prices_df = _create_simulated_prices(markets_df)

    # Run backtest
    engine = BacktestEngine(initial_capital=args.capital)
    results = engine.run(markets_df, prices_df, verbose=True)

    # Print results
    metrics = PerformanceMetrics()
    report = metrics.generate_summary_report(results['metrics'])
    print(report)

    logger.info("Test complete!")


def run_btc_backtest_cmd(args):
    """Run BTC probabilistic arbitrage backtest."""
    import glob as glob_module

    logger.info("Starting BTC probabilistic arbitrage backtest...")

    # Collect log files
    log_files = []
    if args.logs:
        log_files.extend(args.logs)
    if args.log_dir:
        log_files.extend(glob_module.glob(f"{args.log_dir}/*.json*"))

    if not log_files:
        logger.error("No log files specified. Use --logs or --log-dir")
        return

    logger.info(f"Found {len(log_files)} log files")

    # Prepare data
    data = prepare_backtest_data(log_files)
    print_analysis_summary(data['market_analysis'])

    # Check for token prices
    if not data['price_changes']:
        logger.warning("No token price data found, generating synthetic prices...")
        data['price_changes'] = generate_synthetic_token_prices(
            data['chainlink_ticks'],
            noise_factor=0.03,
        )
        logger.info(f"Generated {len(data['price_changes'])} synthetic price points")

    # Setup risk params
    risk_params = BTCRiskParams()
    if args.min_opp:
        risk_params.MIN_OPORTUNIDADE = args.min_opp

    # Run backtest
    engine = BTCBacktestEngine(
        initial_capital=args.capital,
        risk_params=risk_params,
    )

    results = engine.run(
        chainlink_ticks=data['chainlink_ticks'],
        price_changes=data['price_changes'],
        verbose=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BTC BACKTEST RESULTS")
    print("=" * 60)
    summary = results['summary']
    print(f"Total Markets: {summary['total_markets']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Avg Profit/Market: ${summary['avg_profit_per_market']:.2f}")

    if results['metrics']:
        print("\n--- Metrics ---")
        metrics = results['metrics']
        print(f"Return %: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Max Profit: ${metrics.get('max_profit', 0):.2f}")
        print(f"Max Loss: ${metrics.get('max_loss', 0):.2f}")
        if 'avg_opportunity' in metrics:
            print(f"Avg Opportunity: {metrics['avg_opportunity']:.1f}%")

    print("=" * 60)

    # Save results
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"{settings.DATA_RESULTS_PATH}/btc_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results['trades'].empty:
        results['trades'].to_csv(f"{output_dir}/trades.csv", index=False)
    if not results['markets'].empty:
        results['markets'].to_csv(f"{output_dir}/markets.csv", index=False)
    if not results['snapshots'].empty:
        results['snapshots'].to_parquet(f"{output_dir}/snapshots.parquet", index=False)

    logger.info(f"Results saved to {output_dir}")


def run_btc_test(args):
    """Run BTC backtest with simulated data."""
    import numpy as np

    logger.info("Running BTC backtest with simulated data...")
    logger.info(f"Simulating {args.markets} markets with volatility {args.volatility}")

    np.random.seed(42)

    # Generate simulated Chainlink ticks
    ticks = []
    base_price = 87000.0
    current_price = base_price
    base_time = int(datetime.now().timestamp() * 1000) - (args.markets * 15 * 60 * 1000)

    for market_idx in range(args.markets):
        market_start = base_time + (market_idx * 15 * 60 * 1000)

        # Reset for new market (new price to beat)
        price_to_beat = current_price

        # Generate ~900 ticks per market (1 per second)
        for second in range(900):
            ts = market_start + (second * 1000)

            # Random walk
            change = np.random.normal(0, args.volatility * base_price)
            current_price = max(base_price * 0.9, min(base_price * 1.1, current_price + change))

            ticks.append({
                'ts': ts,
                'price': current_price,
                'diff': current_price - price_to_beat,
            })

    logger.info(f"Generated {len(ticks)} ticks")

    # Generate synthetic token prices
    token_prices = generate_synthetic_token_prices(
        ticks,
        noise_factor=0.05,
        lag_ms=500,
    )
    logger.info(f"Generated {len(token_prices)} token price points")

    # Setup risk params
    risk_params = BTCRiskParams()

    # Run backtest
    engine = BTCBacktestEngine(
        initial_capital=args.capital,
        risk_params=risk_params,
    )

    results = engine.run(
        chainlink_ticks=ticks,
        price_changes=token_prices,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BTC SIMULATION RESULTS")
    print("=" * 60)
    summary = results['summary']
    print(f"Total Markets: {summary['total_markets']}")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Avg Profit/Market: ${summary['avg_profit_per_market']:.2f}")

    if results['metrics']:
        print("\n--- Metrics ---")
        metrics = results['metrics']
        print(f"Return %: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"UP Outcomes: {metrics.get('pct_up_outcomes', 0):.1f}%")
        print(f"DOWN Outcomes: {metrics.get('pct_down_outcomes', 0):.1f}%")

    print("=" * 60)
    logger.info("BTC test complete!")


def run_btc_analyze(args):
    """Analyze BTC log files."""
    import glob as glob_module

    logger.info("Analyzing BTC log files...")

    # Collect log files
    log_files = []
    if args.logs:
        log_files.extend(args.logs)
    if args.log_dir:
        log_files.extend(glob_module.glob(f"{args.log_dir}/*.json*"))

    if not log_files:
        logger.error("No log files specified. Use --logs or --log-dir")
        return

    logger.info(f"Found {len(log_files)} log files")

    # Prepare and analyze data
    data = prepare_backtest_data(log_files)

    print("\n" + "=" * 60)
    print("BTC LOG ANALYSIS")
    print("=" * 60)
    print(f"Total Ticks: {data['total_ticks']}")
    print(f"Total Price Changes: {data['total_price_changes']}")
    print(f"Total Markets: {data['total_markets']}")
    print()

    print_analysis_summary(data['market_analysis'])

    # Additional statistics
    if data['chainlink_ticks']:
        ticks = data['chainlink_ticks']
        prices = [t['price'] for t in ticks]
        print("\n--- Price Statistics ---")
        print(f"Min Price: ${min(prices):,.2f}")
        print(f"Max Price: ${max(prices):,.2f}")
        print(f"Avg Price: ${sum(prices)/len(prices):,.2f}")

        first_ts = datetime.fromtimestamp(ticks[0]['ts'] / 1000)
        last_ts = datetime.fromtimestamp(ticks[-1]['ts'] / 1000)
        print(f"\nTime Range: {first_ts} to {last_ts}")
        print(f"Duration: {last_ts - first_ts}")

    print("=" * 60)


# === GABAGOOL COMMANDS ===

def run_gabagool_backtest_cmd(args):
    """Run Gabagool spread capture strategy backtest."""
    logger.info("Starting Gabagool spread capture backtest...")

    # Get config
    if args.preset == 'conservative':
        config = CONSERVATIVE_CONFIG
    elif args.preset == 'aggressive':
        config = AGGRESSIVE_CONFIG
    elif args.preset == 'moderate':
        config = MODERATE_CONFIG
    else:
        config = GabagoolConfig(
            MIN_SPREAD=args.min_spread,
            ORDER_SIZE_USD=args.order_size,
            MAX_PER_MARKET=args.max_per_market,
        )

    # Run backtest
    backtest = GabagoolBacktest(config=config)
    result = backtest.run_backtest(num_markets=args.markets)
    backtest.print_results(result)

    if not args.no_save:
        backtest.save_results(result)

    return result


def run_gabagool_bot_cmd(args):
    """Run Gabagool bot (paper or live trading)."""
    import asyncio

    # Get config
    if args.preset == 'conservative':
        config = CONSERVATIVE_CONFIG
    elif args.preset == 'aggressive':
        config = AGGRESSIVE_CONFIG
    elif args.preset == 'moderate':
        config = MODERATE_CONFIG
    else:
        config = GabagoolConfig(
            MIN_SPREAD=args.min_spread,
            ORDER_SIZE_USD=args.order_size,
            MAX_PER_MARKET=args.max_per_market,
            ENABLED_ASSETS=args.assets,
        )

    # Set trading mode
    config.PAPER_TRADING = not args.live

    if args.live:
        logger.warning("=" * 50)
        logger.warning("LIVE TRADING MODE ENABLED")
        logger.warning("Real money will be used!")
        logger.warning("=" * 50)

        # Validate API credentials
        if not config.PRIVATE_KEY or not config.API_KEY:
            logger.error("API credentials required for live trading")
            logger.error("Set environment variables: POLYMARKET_PRIVATE_KEY, POLYMARKET_API_KEY, etc.")
            return

        response = input("Are you sure you want to continue? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cancelled by user")
            return

    logger.info("=" * 50)
    logger.info("GABAGOOL BOT")
    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER TRADING'}")
    logger.info(f"Min Spread: {config.MIN_SPREAD * 100:.1f}%")
    logger.info(f"Order Size: ${config.ORDER_SIZE_USD}")
    logger.info(f"Assets: {config.ENABLED_ASSETS}")
    logger.info("=" * 50)

    # Run bot
    from src.gabagool import run_bot
    asyncio.run(run_bot(config))


def run_gabagool_test_cmd(args):
    """Quick test of Gabagool strategy with simulated data."""
    logger.info("Running Gabagool quick test...")

    config = GabagoolConfig(
        MIN_SPREAD=0.02,
        ORDER_SIZE_USD=15.0,
        MAX_PER_MARKET=500.0,
    )

    backtest = GabagoolBacktest(config=config)
    result = backtest.run_backtest(num_markets=args.markets)
    backtest.print_results(result)

    logger.info("Gabagool test complete!")
    return result


# === VOLATILITY ARBITRAGE COMMANDS ===

def run_vol_bot_cmd(args):
    """Run volatility arbitrage bot."""
    import asyncio

    logger.info("=" * 60)
    logger.info("VOLATILITY ARBITRAGE BOT")
    logger.info("=" * 60)

    # Create config
    config = BotConfig(
        mode=ExecutionMode.LIVE if args.live else ExecutionMode.PAPER,
        initial_balance=args.balance,
        min_edge_percent=args.min_edge,
        risk_level=args.risk,
    )

    if args.live:
        logger.warning("LIVE TRADING MODE - Real money will be used!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Cancelled")
            return

    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER'}")
    logger.info(f"Balance: ${args.balance}")
    logger.info(f"Min Edge: {args.min_edge}%")
    logger.info(f"Risk: {args.risk}")
    logger.info("=" * 60)

    asyncio.run(run_volatility_bot(config))


def run_vol_test_cmd(args):
    """Quick test of volatility arb strategy."""
    import asyncio

    logger.info("Running volatility arb quick test...")

    config = BotConfig(
        mode=ExecutionMode.PAPER,
        initial_balance=args.balance,
        min_edge_percent=3.0,
        risk_level='moderate',
    )

    async def run_test():
        from src.volatility_arb import VolatilityArbBot

        bot = VolatilityArbBot(config)

        # Run for specified duration
        try:
            # Start bot in background
            bot_task = asyncio.create_task(bot.start())

            # Wait for duration
            await asyncio.sleep(args.duration)

            # Stop bot
            await bot.stop()

        except Exception as e:
            logger.error(f"Test error: {e}")
            await bot.stop()

    asyncio.run(run_test())
    logger.info("Volatility arb test complete!")


def run_vol_backtest_cmd(args):
    """Run volatility arb backtest with real data."""
    logger.info("=" * 60)
    logger.info("VOLATILITY ARBITRAGE BACKTEST")
    logger.info("=" * 60)

    data_path = args.data
    if not Path(data_path).exists():
        logger.error(f"Data path not found: {data_path}")
        logger.info("Make sure to collect data first with the collector")
        return

    logger.info(f"Data Path:   {data_path}")
    logger.info(f"Balance:     ${args.balance}")
    logger.info(f"Min Edge:    {args.min_edge}%")
    logger.info("=" * 60)

    # Run backtest
    result = run_volatility_backtest(
        data_path=data_path,
        initial_balance=args.balance,
        min_edge_pct=args.min_edge,
        verbose=True
    )

    # Save results if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save trades
        import json
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'trade_id': t.trade_id,
                'timestamp': t.timestamp,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'size_usd': t.size_usd,
                'tokens': t.tokens,
                'btc_price': t.btc_price,
                'model_prob': t.model_prob,
                'market_price': t.market_price,
                'edge_pct': t.edge_pct,
                'won': t.won,
                'pnl': t.pnl,
            })

        with open(output_dir / 'vol_trades.json', 'w') as f:
            json.dump(trades_data, f, indent=2)

        # Save summary
        summary = {
            'total_trades': result.total_trades,
            'wins': result.wins,
            'losses': result.losses,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'initial_balance': result.initial_balance,
            'final_balance': result.final_balance,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown_pct': result.max_drawdown_pct,
            'avg_edge_taken': result.avg_edge_taken,
            'data_duration_hours': result.data_duration_hours,
        }

        with open(output_dir / 'vol_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    return result


if __name__ == '__main__':
    main()
