#!/usr/bin/env python3
"""
Run Paper Trader
Script de entrada para executar o paper trading
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.gabagool_config import GabagoolConfig
from trading.paper_trader import PaperTrader


def setup_logging(level: str = "INFO") -> None:
    """Configura logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('paper_trader.log')
        ]
    )


async def run_trader(config: GabagoolConfig, duration: int = None) -> None:
    """
    Executa o paper trader.

    Args:
        config: Configuração
        duration: Duração em segundos (None = infinito)
    """
    trader = PaperTrader(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        print("\n⚠️  Interrupt received, stopping...")
        trader.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        if duration:
            # Run with timeout
            async def timeout_stop():
                await asyncio.sleep(duration)
                print(f"\n⏰ Duration ({duration}s) reached, stopping...")
                trader.stop()

            await asyncio.gather(
                trader.run(),
                timeout_stop(),
                return_exceptions=True
            )
        else:
            # Run indefinitely
            await trader.run()

    except Exception as e:
        logging.error(f"Error: {e}")
        raise

    finally:
        # Print summary
        stats = trader.stats
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        print(f"  Markets Traded:    {stats.markets_traded}")
        print(f"  Total Trades:      {stats.total_trades}")
        print(f"  Total Cost:        ${stats.total_cost:.2f}")
        print(f"  Total Payout:      ${stats.total_payout:.2f}")
        print(f"  Total PnL:         ${stats.total_pnl:.2f}")
        print(f"  Win Rate:          {stats.win_rate*100:.1f}%")
        print(f"  Avg Pair Cost:     {stats.avg_pair_cost:.4f}")
        print("=" * 60)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description='Paper Trader for Polymarket Binary Markets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for 1 hour with $1000 bankroll
  python run_paper.py --bankroll 1000 --duration 3600

  # Run indefinitely with debug logging
  python run_paper.py --log-level DEBUG

  # Run with custom pair cost target
  python run_paper.py --pair-cost-target 0.96
        """
    )

    # Basic options
    parser.add_argument(
        '--bankroll', '-b',
        type=float,
        default=1000,
        help='Initial bankroll in USD (default: 1000)'
    )
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=None,
        help='Duration in seconds (default: run indefinitely)'
    )
    parser.add_argument(
        '--log-level', '-l',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )

    # Trading parameters
    parser.add_argument(
        '--pair-cost-target',
        type=float,
        default=0.97,
        help='Target pair cost for entry (default: 0.97)'
    )
    parser.add_argument(
        '--pair-cost-max',
        type=float,
        default=0.99,
        help='Maximum pair cost to enter (default: 0.99)'
    )
    parser.add_argument(
        '--max-order',
        type=float,
        default=50,
        help='Maximum order size in USD (default: 50)'
    )
    parser.add_argument(
        '--max-position',
        type=float,
        default=500,
        help='Maximum position size in USD (default: 500)'
    )

    # Risk parameters
    parser.add_argument(
        '--daily-loss-limit',
        type=float,
        default=100,
        help='Daily loss limit in USD (default: 100)'
    )
    parser.add_argument(
        '--no-kill-switch',
        action='store_true',
        help='Disable kill switch'
    )

    # Feature flags
    parser.add_argument(
        '--no-microstructure',
        action='store_true',
        help='Disable microstructure analysis'
    )
    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Disable z-score edge detection'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create config
    config = GabagoolConfig()
    config.initial_bankroll = args.bankroll
    config.pair_cost_target = args.pair_cost_target
    config.pair_cost_max = args.pair_cost_max
    config.max_order_usd = args.max_order
    config.max_position_usd = args.max_position
    config.daily_loss_limit_usd = args.daily_loss_limit
    config.kill_switch_enabled = not args.no_kill_switch
    config.enable_microstructure = not args.no_microstructure
    config.enable_zscore_edge = not args.no_zscore

    # Validate config
    if not config.validate():
        print("❌ Invalid configuration")
        sys.exit(1)

    # Print startup info
    print("=" * 60)
    print("🚀 PAPER TRADER - Polymarket Binary Markets")
    print("=" * 60)
    print(f"  Bankroll:          ${config.initial_bankroll:.2f}")
    print(f"  Pair Cost Target:  {config.pair_cost_target}")
    print(f"  Max Order:         ${config.max_order_usd:.2f}")
    print(f"  Max Position:      ${config.max_position_usd:.2f}")
    print(f"  Daily Loss Limit:  ${config.daily_loss_limit_usd:.2f}")
    print(f"  Kill Switch:       {'ON' if config.kill_switch_enabled else 'OFF'}")
    print("=" * 60)
    print()

    # Run
    try:
        asyncio.run(run_trader(config, args.duration))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
