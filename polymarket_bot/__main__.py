"""
Entry point for running the Polymarket Bot as a module.

Usage:
    python -m polymarket_bot              # Run arbitrage bot
    python -m polymarket_bot --test-auth  # Test authentication only
"""

from .performance import setup_performance
setup_performance()

import asyncio
from .bot import main

if __name__ == "__main__":
    asyncio.run(main())
