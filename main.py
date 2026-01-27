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
from src.data_collector import DataCollector
from src.backtest_engine import BacktestEngine, run_backtest
from src.metrics import PerformanceMetrics
from src.visualizer import BacktestVisualizer, generate_report
from src.market_analyzer import MarketAnalyzer
from src.spread_calculator import SpreadCalculator

# Trading imports (optional - only loaded when needed)
def get_trading_modules():
    """Lazy import trading modules."""
    from src.trading import PolymarketClient, OrderManager, AllowanceManager
    from config.trading_config import TradingConfig, create_env_template
    return PolymarketClient, OrderManager, AllowanceManager, TradingConfig, create_env_template

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
        description='AMM Delta-Neutral Strategy Backtest for Polymarket SOL 15-min Markets'
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

    # === TRADING COMMANDS ===

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup trading configuration')

    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Check wallet balance and allowances')

    # Approve command
    approve_parser = subparsers.add_parser('approve', help='Approve tokens for trading')
    approve_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')

    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Submit a limit order')
    trade_parser.add_argument('--token', type=str, help='Token ID for single order')
    trade_parser.add_argument('--token-yes', type=str, help='YES token ID for spread order')
    trade_parser.add_argument('--token-no', type=str, help='NO token ID for spread order')
    trade_parser.add_argument('--side', type=str, choices=['BUY', 'SELL'], help='Order side')
    trade_parser.add_argument('--price', type=float, help='Limit price (0-1)')
    trade_parser.add_argument('--size', type=float, default=100, help='Number of shares')
    trade_parser.add_argument('--min-spread', type=float, default=0.02, help='Minimum spread for spread orders')
    trade_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')

    # Orders command
    orders_parser = subparsers.add_parser('orders', help='List open orders')

    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel orders')
    cancel_parser.add_argument('--order-id', type=str, help='Order ID to cancel')
    cancel_parser.add_argument('--all', action='store_true', help='Cancel all orders')
    cancel_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')

    args = parser.parse_args()

    if args.command == 'collect':
        collect_data(args)
    elif args.command == 'analyze':
        run_analysis(args)
    elif args.command == 'backtest':
        run_backtest_cmd(args)
    elif args.command == 'test':
        run_test(args)
    # Trading commands
    elif args.command == 'setup':
        setup_trading(args)
    elif args.command == 'balance':
        check_balance(args)
    elif args.command == 'approve':
        approve_tokens(args)
    elif args.command == 'trade':
        submit_trade(args)
    elif args.command == 'orders':
        list_orders(args)
    elif args.command == 'cancel':
        cancel_orders(args)
    else:
        parser.print_help()


# === TRADING COMMANDS ===

def setup_trading(args):
    """Setup trading configuration."""
    _, _, _, _, create_env_template = get_trading_modules()

    print("Setting up trading configuration...")

    # Create .env template
    create_env_template(".env.template")

    print("""
Trading Setup Instructions:
==========================

1. Copy .env.template to .env:
   cp .env.template .env

2. Edit .env and add your credentials:
   - PRIVATE_KEY: Your wallet's private key
   - FUNDER_ADDRESS: Address holding your funds (optional if same as wallet)

3. Install trading dependencies:
   pip install py-clob-client web3 eth-account python-dotenv

4. Approve tokens for trading:
   python main.py approve

5. Check your balance:
   python main.py balance

6. Start trading:
   python main.py trade --token-yes <YES_TOKEN_ID> --token-no <NO_TOKEN_ID> --size 100

SECURITY WARNING:
- Never commit .env or your private key to git
- Add .env to .gitignore
""")


def check_balance(args):
    """Check wallet balance and allowances."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    PolymarketClient, _, AllowanceManager, _, _ = get_trading_modules()

    try:
        # Check CLOB balance
        print("\nConnecting to Polymarket...")
        client = PolymarketClient()

        print(f"\nWallet: {client.get_wallet_address()}")
        print(f"Funder: {client.get_funder_address()}")

        # Get balances
        balances = client.get_all_balances()
        print("\n--- CLOB Balances ---")
        if balances:
            for token, balance in balances.items():
                print(f"  {token}: {balance}")
        else:
            print("  No balances found")

        # Check allowances
        print("\nChecking on-chain allowances...")
        allowance_mgr = AllowanceManager()
        allowance_mgr.print_status()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nRun 'python main.py setup' first to configure trading.")
    except Exception as e:
        logger.error(f"Error checking balance: {e}")
        raise


def approve_tokens(args):
    """Approve tokens for trading."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    _, _, AllowanceManager, _, _ = get_trading_modules()

    try:
        print("Approving tokens for Polymarket trading...")
        print("This will submit transactions to Polygon. Make sure you have MATIC for gas.\n")

        allowance_mgr = AllowanceManager()

        # Show current status
        allowance_mgr.print_status()

        if not args.yes:
            confirm = input("\nProceed with approvals? (yes/no): ")
            if confirm.lower() != "yes":
                print("Cancelled.")
                return

        # Approve all
        results = allowance_mgr.approve_all_exchanges()

        print("\n--- Approval Results ---")
        for result in results:
            status = "SUCCESS" if result.get("status") == 1 else "FAILED"
            print(f"  {result['exchange']} {result['token']}: {status}")
            print(f"    TX: {result.get('tx_hash', 'N/A')}")

        print("\nApprovals complete! You can now trade.")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nRun 'python main.py setup' first to configure trading.")
    except Exception as e:
        logger.error(f"Error approving tokens: {e}")
        raise


def submit_trade(args):
    """Submit a limit order or spread trade."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    PolymarketClient, OrderManager, _, _, _ = get_trading_modules()

    try:
        print("Connecting to Polymarket...")
        client = PolymarketClient()
        orders = OrderManager(client)

        if args.token_yes and args.token_no:
            # Spread order
            print(f"\nChecking spread opportunity...")

            opportunity = orders.calculate_spread_opportunity(
                args.token_yes,
                args.token_no,
            )

            if opportunity.get("error"):
                print(f"Error: {opportunity['error']}")
                return

            print(f"  YES price: ${opportunity['yes_price']:.4f}")
            print(f"  NO price:  ${opportunity['no_price']:.4f}")
            print(f"  Total:     ${opportunity['total_price']:.4f}")
            print(f"  Spread:    {opportunity['spread_pct']:.2f}%")

            if not opportunity["has_opportunity"]:
                print("\nNo spread opportunity (total >= $1.00)")
                return

            if opportunity["spread"] < args.min_spread:
                print(f"\nSpread {opportunity['spread']:.4f} below minimum {args.min_spread}")
                return

            if not args.yes:
                confirm = input(f"\nSubmit spread order for {args.size} shares? (yes/no): ")
                if confirm.lower() != "yes":
                    print("Cancelled.")
                    return

            # Submit spread order
            yes_result, no_result = orders.submit_spread_order(
                yes_token_id=args.token_yes,
                no_token_id=args.token_no,
                yes_price=opportunity["yes_price"],
                no_price=opportunity["no_price"],
                size=args.size,
            )

            print("\n--- Order Results ---")
            print(f"  YES: {'SUCCESS' if yes_result.success else 'FAILED'} - {yes_result.message}")
            if yes_result.order_id:
                print(f"       Order ID: {yes_result.order_id}")

            print(f"  NO:  {'SUCCESS' if no_result.success else 'FAILED'} - {no_result.message}")
            if no_result.order_id:
                print(f"       Order ID: {no_result.order_id}")

        elif args.token:
            # Single order
            if not args.price:
                print("Error: --price required for single orders")
                return

            if not args.side:
                print("Error: --side required for single orders")
                return

            if not args.yes:
                confirm = input(
                    f"\nSubmit {args.side} order: {args.size} shares @ ${args.price:.4f}? (yes/no): "
                )
                if confirm.lower() != "yes":
                    print("Cancelled.")
                    return

            result = orders.submit_limit_order(
                token_id=args.token,
                side=args.side,
                price=args.price,
                size=args.size,
            )

            print("\n--- Order Result ---")
            print(f"  {'SUCCESS' if result.success else 'FAILED'} - {result.message}")
            if result.order_id:
                print(f"  Order ID: {result.order_id}")

        else:
            print("Error: Specify --token for single order or --token-yes and --token-no for spread")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nRun 'python main.py setup' first to configure trading.")
    except Exception as e:
        logger.error(f"Error submitting trade: {e}")
        raise


def list_orders(args):
    """List open orders."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    PolymarketClient, OrderManager, _, _, _ = get_trading_modules()

    try:
        print("Connecting to Polymarket...")
        client = PolymarketClient()
        orders = OrderManager(client)

        orders.print_open_orders()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nRun 'python main.py setup' first to configure trading.")
    except Exception as e:
        logger.error(f"Error listing orders: {e}")
        raise


def cancel_orders(args):
    """Cancel orders."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    PolymarketClient, OrderManager, _, _, _ = get_trading_modules()

    try:
        print("Connecting to Polymarket...")
        client = PolymarketClient()
        order_mgr = OrderManager(client)

        if args.order_id:
            result = order_mgr.cancel_order(args.order_id)
            print(f"Cancel result: {'SUCCESS' if result.success else 'FAILED'} - {result.message}")
        elif args.all:
            if not args.yes:
                confirm = input("Cancel ALL open orders? (yes/no): ")
                if confirm.lower() != "yes":
                    print("Cancelled.")
                    return

            result = order_mgr.cancel_all_orders()
            print(f"Cancel result: {'SUCCESS' if result.success else 'FAILED'} - {result.message}")
        else:
            print("Specify --order-id <ID> or --all")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print("\nRun 'python main.py setup' first to configure trading.")
    except Exception as e:
        logger.error(f"Error cancelling orders: {e}")
        raise


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


if __name__ == '__main__':
    main()
