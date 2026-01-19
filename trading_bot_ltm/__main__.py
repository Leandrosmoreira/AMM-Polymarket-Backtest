"""
Entry point for running the trading bot as a module.

Usage:
    cd /path/to/AMM-Polymarket-Backtest
    python -m trading_bot_ltm           # Bot 1 - Arbitrage
    python -m trading_bot_ltm.market_maker  # Bot 2 - Market Maker
"""
# Setup performance FIRST (before any asyncio imports in submodules)
from .performance import setup_performance
setup_performance()

import asyncio
from .simple_arb_bot import main

if __name__ == "__main__":
    asyncio.run(main())
