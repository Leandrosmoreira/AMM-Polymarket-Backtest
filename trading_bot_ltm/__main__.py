"""
Entry point for running the trading bot as a module.

Usage:
    cd /path/to/AMM-Polymarket-Backtest
    python -m trading_bot_ltm
"""
import asyncio
from .simple_arb_bot import main

if __name__ == "__main__":
    asyncio.run(main())
