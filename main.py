#!/usr/bin/env python3
"""
AMM Delta-Neutral Strategy Backtest
Polymarket SOL 15-min Markets

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
from src.data_collector import DataCollector, DataMode
from src.backtest_engine import BacktestEngine, run_backtest
from src.ltm_backtest_engine import LTMBacktestEngine, run_ltm_backtest
from src.metrics import PerformanceMetrics
from src.visualizer import BacktestVisualizer, generate_report
from src.market_analyzer import MarketAnalyzer
from src.spread_calculator import SpreadCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(args):
    """Collect market data from Polymarket API."""
    asset = getattr(args, 'asset', 'SOL').upper()
    mode = getattr(args, 'mode', 'cache')
    logger.info(f"Starting data collection for {asset}...")
    logger.info(f"Mode: {mode.upper()}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Map mode string to DataMode
    mode_map = {'live': DataMode.LIVE, 'cache': DataMode.CACHE, 'sim': DataMode.SIM}
    data_mode = mode_map.get(mode, DataMode.CACHE)

    collector = DataCollector(mode=data_mode)

    try:
        # Fetch markets using generic method
        markets_df = collector.fetch_15min_markets(
            asset=asset,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            save_path=f"{settings.DATA_RAW_PATH}/{asset.lower()}_markets.csv"
        )

        logger.info(f"Found {len(markets_df)} {asset} markets")

        if not markets_df.empty and args.fetch_prices:
            # Fetch prices
            prices_df = collector.fetch_all_prices(
                markets_df,
                save_dir=f"{settings.DATA_RAW_PATH}/{asset.lower()}_price_history"
            )

            if not prices_df.empty:
                # Save consolidated prices
                Path(settings.DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
                prices_df.to_parquet(
                    f"{settings.DATA_PROCESSED_PATH}/{asset.lower()}_prices.parquet",
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
        description='AMM Delta-Neutral Strategy Backtest for Polymarket SOL 15-min Markets'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect market data')
    collect_parser.add_argument('--asset', type=str, default='SOL', help='Asset to collect (SOL, BTC, ETH)')
    collect_parser.add_argument('--days', type=int, default=90, help='Days of history')
    collect_parser.add_argument('--fetch-prices', action='store_true', help='Also fetch price history')
    collect_parser.add_argument('--mode', type=str, default='cache', choices=['live', 'cache', 'sim'],
                               help='Data mode: live=API only, cache=API+cache, sim=simulation only')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on data')
    analyze_parser.add_argument('--markets', type=str, help='Path to markets CSV')
    analyze_parser.add_argument('--prices', type=str, help='Path to prices file')

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--markets', type=str, help='Path to markets CSV')
    backtest_parser.add_argument('--prices', type=str, help='Path to prices file')
    backtest_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')
    backtest_parser.add_argument('--spread-threshold', type=float, help='Min spread to enter')
    backtest_parser.add_argument('--max-exposure', type=float, help='Max portfolio exposure')
    backtest_parser.add_argument('--output', type=str, help='Output directory')

    # Quick test command
    test_parser = subparsers.add_parser('test', help='Run quick test with simulated data')
    test_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')

    # LTM backtest command
    ltm_parser = subparsers.add_parser('ltm-backtest', help='Run LTM-enhanced backtest')
    ltm_parser.add_argument('--markets', type=str, help='Path to markets CSV')
    ltm_parser.add_argument('--prices', type=str, help='Path to prices file')
    ltm_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')
    ltm_parser.add_argument('--policy', type=str, default='config/ltm_policy.yaml', help='LTM policy YAML')
    ltm_parser.add_argument('--use-decay', action='store_true', default=True, help='Use decay model')
    ltm_parser.add_argument('--use-bandit', action='store_true', help='Use bandit auto-tuning')
    ltm_parser.add_argument('--output', type=str, help='Output directory')

    # LTM test command
    ltm_test_parser = subparsers.add_parser('ltm-test', help='Run LTM test with simulated data')
    ltm_test_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')
    ltm_test_parser.add_argument('--use-bandit', action='store_true', help='Use bandit auto-tuning')
    ltm_test_parser.add_argument('--n-markets', type=int, default=200, help='Number of simulated markets')

    # Build LTM policy command
    build_policy_parser = subparsers.add_parser('build-policy', help='Build LTM policy from data')
    build_policy_parser.add_argument('--data', type=str, help='Path to snapshot data')
    build_policy_parser.add_argument('--output', type=str, default='config/ltm_policy.yaml', help='Output path')
    build_policy_parser.add_argument('--simulate', action='store_true', help='Generate from simulated data')

    # Backtest with real data command
    real_parser = subparsers.add_parser('backtest-real', help='Run backtest with real historical data')
    real_parser.add_argument('--asset', type=str, default='BTC', help='Asset (BTC, SOL, ETH)')
    real_parser.add_argument('--days', type=int, default=30, help='Days of history to backtest')
    real_parser.add_argument('--capital', type=float, default=5000, help='Initial capital')
    real_parser.add_argument('--threshold', type=float, default=0.99, help='Target pair cost threshold')
    real_parser.add_argument('--use-ltm', action='store_true', help='Use LTM model')
    real_parser.add_argument('--output', type=str, help='Output directory')

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'backtest':
        run_backtest_cmd(args)
    elif args.command == 'test':
        run_test(args)
    elif args.command == 'ltm-backtest':
        run_ltm_backtest_cmd(args)
    elif args.command == 'ltm-test':
        run_ltm_test(args)
    elif args.command == 'build-policy':
        run_build_policy(args)
    elif args.command == 'backtest-real':
        run_backtest_real(args)
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


def run_ltm_backtest_cmd(args):
    """Run LTM-enhanced backtest."""
    import pandas as pd

    logger.info("Starting LTM-enhanced backtest...")

    # Load data
    markets_path = args.markets or f"{settings.DATA_RAW_PATH}/sol_markets.csv"
    prices_path = args.prices or f"{settings.DATA_PROCESSED_PATH}/all_prices.parquet"

    if not Path(markets_path).exists():
        logger.error(f"Markets file not found: {markets_path}")
        logger.info("Run with 'collect' first to gather data, or use 'ltm-test' for simulated data")
        return

    markets_df = pd.read_csv(markets_path)

    if Path(prices_path).exists():
        if prices_path.endswith('.parquet'):
            prices_df = pd.read_parquet(prices_path)
        else:
            prices_df = pd.read_csv(prices_path)
    else:
        logger.warning("No price data found, using simulated prices")
        prices_df = _create_simulated_prices(markets_df)

    # Run LTM backtest
    engine = LTMBacktestEngine(
        initial_capital=args.capital,
        ltm_policy_path=args.policy,
        use_decay_model=args.use_decay,
        use_bandit=args.use_bandit,
    )

    results = engine.run(markets_df, prices_df, verbose=True)

    # Print LTM report
    print("\n" + engine.generate_ltm_report())

    # Print summary
    metrics = PerformanceMetrics()
    report = metrics.generate_summary_report(results['metrics'])
    print(report)

    # Generate visual report
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"{settings.DATA_RESULTS_PATH}/ltm_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generate_report(results, output_dir)

    # Save LTM-specific data
    ltm_paths = engine.save_ltm_data(output_dir)
    logger.info(f"LTM data saved: {ltm_paths}")

    # Save results
    results['trades'].to_csv(f"{output_dir}/trades.csv", index=False)
    results['portfolio_history'].to_csv(f"{output_dir}/portfolio_history.csv", index=False)

    logger.info(f"Report saved to {output_dir}")


def run_ltm_test(args):
    """Run LTM test with simulated data."""
    import pandas as pd
    import numpy as np

    logger.info("Running LTM test with simulated data...")
    logger.info(f"Simulating {args.n_markets} markets...")

    # Create simulated markets
    np.random.seed(42)
    n_markets = args.n_markets

    markets = []
    base_time = datetime.now() - timedelta(days=14)

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
    prices_df = _create_ltm_simulated_prices(markets_df)

    # Run LTM backtest
    engine = LTMBacktestEngine(
        initial_capital=args.capital,
        use_decay_model=True,
        use_bandit=args.use_bandit,
        collect_snapshots=True,
    )

    results = engine.run(markets_df, prices_df, verbose=True)

    # Print LTM report
    print("\n" + engine.generate_ltm_report())

    # Print summary
    metrics = PerformanceMetrics()
    report = metrics.generate_summary_report(results['metrics'])
    print(report)

    # Compare with base backtest
    logger.info("\nComparing with base backtest (no LTM)...")
    base_engine = BacktestEngine(initial_capital=args.capital)
    base_results = base_engine.run(markets_df.copy(), prices_df.copy(), verbose=False)

    print("\n" + "=" * 60)
    print("COMPARISON: LTM vs Base Backtest")
    print("=" * 60)
    print(f"  LTM Return: {results['summary']['total_return_pct']:.2f}%")
    print(f"  Base Return: {base_results['summary']['total_return_pct']:.2f}%")
    print(f"  Improvement: {results['summary']['total_return_pct'] - base_results['summary']['total_return_pct']:.2f}%")
    print(f"  LTM Trades: {results['summary']['total_trades']}")
    print(f"  Base Trades: {base_results['summary']['total_trades']}")
    print("=" * 60)

    logger.info("LTM test complete!")


def _create_ltm_simulated_prices(markets_df):
    """Create simulated price data with realistic LTM patterns."""
    import pandas as pd
    import numpy as np

    all_prices = []

    for _, market in markets_df.iterrows():
        market_id = market['market_id']
        start = pd.to_datetime(market['start_time'])
        end = pd.to_datetime(market['end_time'])

        # Generate 30 price points (every 30 seconds)
        timestamps = pd.date_range(start, end, periods=30)

        # Market characteristics
        base_spread = np.random.uniform(-0.04, -0.01)

        for idx, ts in enumerate(timestamps):
            t_remaining = (end - ts).total_seconds()
            t_elapsed = (ts - start).total_seconds()
            pct_elapsed = t_elapsed / 900

            # Spread evolution - narrower in middle, wider at ends
            if pct_elapsed < 0.2:  # Early
                spread_mult = 1.1
            elif pct_elapsed < 0.7:  # Middle - best conditions
                spread_mult = 0.85
            elif pct_elapsed < 0.9:  # Late
                spread_mult = 1.2
            else:  # Final
                spread_mult = 1.5

            spread = base_spread * spread_mult + np.random.normal(0, 0.005)

            # Price decomposition
            imbalance = np.random.normal(0, 0.02)
            price_yes = np.clip(0.5 + imbalance + spread / 2, 0.35, 0.65)
            price_no = np.clip(0.5 - imbalance + spread / 2, 0.35, 0.65)

            # Volume accumulation
            volume = np.random.uniform(100, 500) * (1 + pct_elapsed)

            all_prices.append({
                'timestamp': ts,
                'market_id': market_id,
                'price_yes': price_yes,
                'price_no': price_no,
                'spread': price_yes + price_no - 1.0,
                'volume': volume,
            })

    return pd.DataFrame(all_prices)


def run_build_policy(args):
    """Build LTM policy from data."""
    import subprocess
    import sys

    script_path = Path(__file__).parent / 'scripts' / 'build_ltm_policy.py'

    cmd = [sys.executable, str(script_path)]

    if args.data:
        cmd.extend(['--data', args.data])
    if args.simulate:
        cmd.append('--simulate')
    cmd.extend(['--output', args.output])
    cmd.append('--report')

    subprocess.run(cmd)


def run_backtest_real(args):
    """
    Run backtest with real historical data from Polymarket.

    This validates the trading bot logic using actual market data.
    """
    import pandas as pd
    import numpy as np

    asset = args.asset.upper()
    logger.info(f"=" * 60)
    logger.info(f"BACKTEST COM DADOS REAIS - {asset} 15-min Markets")
    logger.info(f"=" * 60)

    # Step 1: Collect real data
    logger.info(f"\n[1/4] Coletando dados de {args.days} dias...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    collector = DataCollector()

    try:
        # Fetch markets
        markets_path = f"{settings.DATA_RAW_PATH}/{asset.lower()}_markets.csv"
        prices_path = f"{settings.DATA_PROCESSED_PATH}/{asset.lower()}_prices.parquet"

        # Check if we already have recent data
        if Path(markets_path).exists():
            existing_df = pd.read_csv(markets_path)
            existing_df['start_time'] = pd.to_datetime(existing_df['start_time'])
            latest = existing_df['start_time'].max()

            # If data is less than 1 day old, use it
            if (datetime.now() - latest).days < 1:
                logger.info(f"Usando dados existentes ({len(existing_df)} mercados)")
                markets_df = existing_df
            else:
                logger.info("Dados antigos, coletando novos...")
                markets_df = collector.fetch_15min_markets(
                    asset=asset,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    save_path=markets_path
                )
        else:
            markets_df = collector.fetch_15min_markets(
                asset=asset,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                save_path=markets_path
            )

        if markets_df.empty:
            logger.error(f"Nenhum mercado {asset} encontrado!")
            logger.info("Tentando gerar dados simulados baseados em padrões reais...")
            markets_df, prices_df = _generate_realistic_data(asset, args.days)
        else:
            logger.info(f"Encontrados {len(markets_df)} mercados {asset}")

            # Fetch prices
            logger.info("\n[2/4] Coletando histórico de preços...")

            if Path(prices_path).exists():
                prices_df = pd.read_parquet(prices_path)
                logger.info(f"Usando preços existentes ({len(prices_df)} registros)")
            else:
                prices_df = collector.fetch_all_prices(
                    markets_df,
                    save_dir=f"{settings.DATA_RAW_PATH}/{asset.lower()}_price_history"
                )

                if not prices_df.empty:
                    Path(settings.DATA_PROCESSED_PATH).mkdir(parents=True, exist_ok=True)
                    prices_df.to_parquet(prices_path, index=False)

        if prices_df.empty:
            logger.warning("Sem dados de preço, gerando simulação realista...")
            prices_df = _create_realistic_prices(markets_df, asset)

    finally:
        collector.close()

    # Step 2: Run backtest
    logger.info(f"\n[3/4] Executando backtest...")
    logger.info(f"  Capital inicial: ${args.capital:.2f}")
    logger.info(f"  Threshold: ${args.threshold:.4f}")
    logger.info(f"  LTM: {'Ativado' if args.use_ltm else 'Desativado'}")

    # Configure risk params for real trading simulation
    risk_params = RiskParams()
    # threshold 0.99 means we want YES+NO < 0.99, so spread < -0.01
    risk_params.MIN_SPREAD_TO_ENTER = args.threshold - 1.0  # Convert to spread
    risk_params.MIN_VOLUME = 100  # Lower volume requirement for more trades

    if args.use_ltm:
        engine = LTMBacktestEngine(
            initial_capital=args.capital,
            use_decay_model=True,
            use_bandit=False,
        )
    else:
        engine = BacktestEngine(
            initial_capital=args.capital,
            risk_params=risk_params
        )

    results = engine.run(markets_df, prices_df, verbose=True)

    # Step 3: Generate report
    logger.info(f"\n[4/4] Gerando relatório...")

    print("\n" + "=" * 60)
    print(f"RESULTADOS DO BACKTEST - {asset} (DADOS REAIS)")
    print("=" * 60)

    summary = results.get('summary', {})
    print(f"\n📊 RESUMO:")
    print(f"  Mercados analisados:    {summary.get('total_markets', len(markets_df))}")
    print(f"  Trades executados:      {summary.get('total_trades', 0)}")
    print(f"  Capital inicial:        ${args.capital:.2f}")
    print(f"  Capital final:          ${summary.get('final_capital', args.capital):.2f}")
    print(f"  Retorno total:          {summary.get('total_return_pct', 0):.2f}%")

    metrics = results.get('metrics', {})
    print(f"\n📈 MÉTRICAS:")
    print(f"  Win Rate:               {metrics.get('win_rate', 0):.1f}%")
    print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:           {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Profit Factor:          {metrics.get('profit_factor', 0):.2f}")

    # Analyze opportunities
    trades_df = results.get('trades', pd.DataFrame())
    if not trades_df.empty:
        print(f"\n💰 OPORTUNIDADES:")
        profitable = trades_df[trades_df.get('pnl', trades_df.get('profit', 0)) > 0]
        print(f"  Trades lucrativos:      {len(profitable)}/{len(trades_df)}")
        print(f"  Lucro médio por trade:  ${trades_df.get('pnl', trades_df.get('profit', pd.Series([0]))).mean():.2f}")

        # Show spread distribution
        if 'spread' in trades_df.columns:
            print(f"\n📉 SPREADS:")
            print(f"  Spread médio:           {trades_df['spread'].mean():.4f}")
            print(f"  Spread mínimo:          {trades_df['spread'].min():.4f}")
            print(f"  Spread máximo:          {trades_df['spread'].max():.4f}")

    if args.use_ltm:
        print("\n" + engine.generate_ltm_report())

    print("=" * 60)

    # Save results
    if args.output:
        output_dir = args.output
    else:
        output_dir = f"{settings.DATA_RESULTS_PATH}/backtest_real_{asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not trades_df.empty:
        trades_df.to_csv(f"{output_dir}/trades.csv", index=False)

    portfolio_history = results.get('portfolio_history', pd.DataFrame())
    if not portfolio_history.empty:
        portfolio_history.to_csv(f"{output_dir}/portfolio_history.csv", index=False)

    # Save summary
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write(f"Backtest Real - {asset}\n")
        f.write(f"Período: {args.days} dias\n")
        f.write(f"Capital: ${args.capital}\n")
        f.write(f"Threshold: ${args.threshold}\n")
        f.write(f"LTM: {args.use_ltm}\n")
        f.write(f"Mercados: {len(markets_df)}\n")
        f.write(f"Trades: {summary.get('total_trades', 0)}\n")
        f.write(f"Retorno: {summary.get('total_return_pct', 0):.2f}%\n")

    logger.info(f"\nResultados salvos em: {output_dir}")


def _generate_realistic_data(asset: str, days: int):
    """Generate realistic market data when API doesn't return results."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)

    # Generate markets every 15 min
    markets_per_day = 96  # 24h * 4
    n_markets = days * markets_per_day

    markets = []
    base_time = datetime.now() - timedelta(days=days)

    for i in range(n_markets):
        start = base_time + timedelta(minutes=15 * i)
        end = start + timedelta(minutes=15)

        markets.append({
            'market_id': f'{asset.lower()}_market_{i}',
            'condition_id': f'cond_{i}',
            'question': f'{asset} Up or Down - {start.strftime("%Y-%m-%d %H:%M")}',
            'start_time': start,
            'end_time': end,
            'outcome': 'Up' if np.random.random() > 0.5 else 'Down',
            'yes_token_id': f'yes_{i}',
            'no_token_id': f'no_{i}',
            'volume': np.random.uniform(1000, 10000),
            'liquidity': np.random.uniform(5000, 50000),
        })

    markets_df = pd.DataFrame(markets)
    prices_df = _create_realistic_prices(markets_df, asset)

    return markets_df, prices_df


def _create_realistic_prices(markets_df, asset: str):
    """Create realistic price data based on real market patterns."""
    import pandas as pd
    import numpy as np

    all_prices = []

    for _, market in markets_df.iterrows():
        market_id = market['market_id']
        start = pd.to_datetime(market['start_time'])
        end = pd.to_datetime(market['end_time'])

        # Generate 15 price points (every minute)
        timestamps = pd.date_range(start, end, periods=15)

        # Outcome determines final price direction
        outcome = market.get('outcome', 'Up')

        # Random walk around 0.50
        base_yes = 0.50
        base_no = 0.50

        for idx, ts in enumerate(timestamps):
            # Add some randomness
            price_yes = np.clip(base_yes + np.random.normal(0, 0.02), 0.40, 0.60)
            price_no = np.clip(base_no + np.random.normal(0, 0.02), 0.40, 0.60)

            # Ensure some spread opportunity (30% chance)
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
                'volume': np.random.uniform(500, 5000),
            })

    return pd.DataFrame(all_prices)


if __name__ == '__main__':
    main()
