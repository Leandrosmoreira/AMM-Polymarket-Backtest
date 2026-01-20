"""
Entry point for running the Polymarket Bot as a module.

Usage:
    python -m polymarket_bot                    # Multi-market bot (BTC, ETH, SOL)
    python -m polymarket_bot --single           # Single market bot (BTC only)
    python -m polymarket_bot --test-auth        # Test authentication
    python -m polymarket_bot btc eth            # Specific assets only
"""

from .performance import setup_performance
setup_performance()

import asyncio
import sys


def main():
    # Check for flags
    args = [a.lower() for a in sys.argv[1:]]

    if "--test-auth" in args:
        from .auth import test_auth
        success = test_auth("pmpe.env")
        sys.exit(0 if success else 1)

    if "--single" in args:
        # Use single-market bot (original)
        from .bot import main as single_main
        asyncio.run(single_main())
    else:
        # Use multi-market bot (default)
        from .multi_bot import main as multi_main
        asyncio.run(multi_main())


if __name__ == "__main__":
    main()
