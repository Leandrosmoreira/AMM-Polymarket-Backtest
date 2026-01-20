"""
Entry point for running the Polymarket Bot as a module.

Usage:
    python -m polymarket_bot                    # Bot 1: Multi-market arbitrage (BTC, ETH, SOL)
    python -m polymarket_bot --market-maker     # Bot 2: Market maker
    python -m polymarket_bot --single           # Bot 1: Single market (BTC only)
    python -m polymarket_bot --test-auth        # Test authentication
    python -m polymarket_bot btc eth            # Specific assets only

Bots:
    Bot 1 (Arbitrage): Compra YES+NO quando soma < $0.991 (taker)
    Bot 2 (Market Maker): Cria liquidez nos dois lados (maker)
"""

from .performance import setup_performance
setup_performance()

import asyncio
import sys


def print_help():
    """Print usage help."""
    print("""
Polymarket Bot - Multi-Market Trading

USAGE:
    python -m polymarket_bot [OPTIONS] [ASSETS]

OPTIONS:
    --help, -h          Show this help
    --test-auth         Test authentication only
    --single            Use single-market bot (BTC only)
    --market-maker      Use Market Maker bot (Bot 2)

ASSETS:
    btc, eth, sol       Specific assets to monitor (default: all)

EXAMPLES:
    python -m polymarket_bot                    # Arbitrage on BTC, ETH, SOL
    python -m polymarket_bot btc eth           # Arbitrage on BTC and ETH only
    python -m polymarket_bot --market-maker    # Market maker on all assets
    python -m polymarket_bot --market-maker sol # Market maker on SOL only

BOTS:
    Bot 1 (Arbitrage):
        - Monitors YES and NO prices
        - Buys both when YES + NO < $0.991
        - Guaranteed profit when market settles
        - Taker strategy (consumes liquidity)

    Bot 2 (Market Maker):
        - Creates liquidity by quoting bid and ask
        - Adjusts spread based on volatility
        - Maintains delta-neutral position
        - Maker strategy (provides liquidity)
""")


def main():
    args = [a.lower() for a in sys.argv[1:]]

    # Help
    if "--help" in args or "-h" in args:
        print_help()
        return

    # Test auth
    if "--test-auth" in args:
        from .auth import test_auth
        success = test_auth("pmpe.env")
        sys.exit(0 if success else 1)

    # Market Maker (Bot 2)
    if "--market-maker" in args or "--mm" in args:
        from .market_maker_bot import main as mm_main
        asyncio.run(mm_main())
        return

    # Single market (original bot)
    if "--single" in args:
        from .bot import main as single_main
        asyncio.run(single_main())
        return

    # Default: Multi-market arbitrage (Bot 1)
    from .multi_bot import main as multi_main
    asyncio.run(multi_main())


if __name__ == "__main__":
    main()
