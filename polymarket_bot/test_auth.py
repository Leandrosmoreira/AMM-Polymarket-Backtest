#!/usr/bin/env python3
"""
Polymarket Authentication Test.

Test your authentication setup before running the bot.
Based on the working pattern from exemplo_polymarket.

Usage:
    python -m polymarket_bot.test_auth
    python polymarket_bot/test_auth.py
"""

import sys
from pathlib import Path

# Add parent dir to path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_bot.auth import AuthConfig, test_auth, get_client, get_balance


def main():
    """Run authentication test."""
    print("=" * 70)
    print("  POLYMARKET AUTHENTICATION TEST")
    print("  Based on working pattern from exemplo_polymarket")
    print("=" * 70)
    print()

    # Try to find config file
    possible_files = [
        "pmpe.env",
        "polymarket_bot/pmpe.env",
        ".env",
        "polymarket_bot/.env",
    ]

    config_file = None
    for f in possible_files:
        if Path(f).exists():
            config_file = f
            break

    if not config_file:
        print("ERROR: No configuration file found!")
        print()
        print("Please create pmpe.env from the template:")
        print("  cp polymarket_bot/pmpe.env.template polymarket_bot/pmpe.env")
        print()
        print("Then fill in your credentials:")
        print("  - PRIVATE_KEY: Export from Polymarket Settings")
        print("  - FUNDER_ADDRESS: Copy from your profile page")
        print()
        return False

    print(f"Config file: {config_file}")
    print()

    # Run test
    success = test_auth(config_file)

    if success:
        print()
        print("=" * 70)
        print("  SUCCESS! Your authentication is working.")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Run the bot in simulation mode:")
        print("     python -m polymarket_bot")
        print()
        print("  2. When ready for live trading, set DRY_RUN=false")
        print()
    else:
        print()
        print("=" * 70)
        print("  AUTHENTICATION FAILED")
        print("=" * 70)
        print()
        print("Common issues:")
        print()
        print("1. FUNDER_ADDRESS is missing or incorrect")
        print("   - Go to https://polymarket.com/@YOUR_USERNAME")
        print("   - Copy the wallet address shown on your profile")
        print("   - This is your PROXY wallet, not your signer!")
        print()
        print("2. PRIVATE_KEY is incorrect")
        print("   - Go to Polymarket Settings")
        print("   - Click 'Export Private Key'")
        print("   - Copy the full key including '0x' prefix")
        print()
        print("3. SIGNATURE_TYPE is wrong")
        print("   - Use 1 for Magic.link/Email accounts (most common)")
        print("   - Use 0 only for direct MetaMask connections")
        print()

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
