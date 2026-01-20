"""
Multi-market discovery for Polymarket 15min crypto markets.

Supports: BTC, ETH, SOL (and future markets)

Markets open and close every 15 minutes with new token IDs.
This module handles automatic discovery and rollover.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from .lookup import fetch_market_from_slug

logger = logging.getLogger(__name__)

# Supported crypto assets for 15min markets
SUPPORTED_ASSETS = ["btc", "eth", "sol"]

# Market duration in seconds
MARKET_DURATION = 900  # 15 minutes


@dataclass
class Market:
    """Represents a single 15min market."""
    asset: str  # btc, eth, sol
    slug: str  # btc-updown-15m-1737340800
    market_id: str
    yes_token_id: str
    no_token_id: str
    start_timestamp: int
    end_timestamp: int

    @property
    def time_remaining(self) -> int:
        """Seconds until market closes."""
        now = int(datetime.now().timestamp())
        return max(0, self.end_timestamp - now)

    @property
    def time_remaining_str(self) -> str:
        """Human readable time remaining."""
        remaining = self.time_remaining
        if remaining <= 0:
            return "CLOSED"
        minutes = remaining // 60
        seconds = remaining % 60
        return f"{minutes}m {seconds}s"

    @property
    def is_open(self) -> bool:
        """Check if market is still open."""
        return self.time_remaining > 0

    def __repr__(self):
        return f"Market({self.asset.upper()}, {self.time_remaining_str}, {self.slug})"


@dataclass
class MarketManager:
    """
    Manages multiple 15min crypto markets.

    Handles discovery, tracking, and automatic rollover.
    """
    assets: List[str] = field(default_factory=lambda: SUPPORTED_ASSETS.copy())
    markets: Dict[str, Market] = field(default_factory=dict)
    _last_refresh: float = 0

    def discover_all(self) -> Dict[str, Market]:
        """
        Discover all active 15min markets for configured assets.

        Uses parallel requests for faster discovery.

        Returns:
            Dict of asset -> Market
        """
        logger.info(f"Discovering markets for: {', '.join(a.upper() for a in self.assets)}")

        discovered = {}

        # Fetch the crypto 15M page to find all active markets
        try:
            page_url = "https://polymarket.com/crypto/15M"
            resp = httpx.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            page_content = resp.text
        except Exception as e:
            logger.error(f"Failed to fetch markets page: {e}")
            return discovered

        now_ts = int(datetime.now().timestamp())

        # Find slugs for each asset in parallel
        def find_market_for_asset(asset: str) -> Optional[Market]:
            try:
                # Pattern: btc-updown-15m-{timestamp}
                pattern = rf'{asset}-updown-15m-(\d+)'
                matches = re.findall(pattern, page_content)

                if not matches:
                    logger.warning(f"No {asset.upper()} 15min market found")
                    return None

                # Find the most recent OPEN market
                all_ts = sorted((int(ts) for ts in matches), reverse=True)
                open_ts = [ts for ts in all_ts if now_ts < (ts + MARKET_DURATION)]

                if not open_ts:
                    logger.warning(f"No open {asset.upper()} 15min market")
                    return None

                chosen_ts = open_ts[0]
                slug = f"{asset}-updown-15m-{chosen_ts}"

                # Fetch market details
                market_info = fetch_market_from_slug(slug)

                return Market(
                    asset=asset,
                    slug=slug,
                    market_id=market_info["market_id"],
                    yes_token_id=market_info["yes_token_id"],
                    no_token_id=market_info["no_token_id"],
                    start_timestamp=chosen_ts,
                    end_timestamp=chosen_ts + MARKET_DURATION,
                )
            except Exception as e:
                logger.error(f"Error discovering {asset.upper()} market: {e}")
                return None

        # Parallel discovery
        with ThreadPoolExecutor(max_workers=len(self.assets)) as executor:
            futures = {executor.submit(find_market_for_asset, asset): asset
                      for asset in self.assets}

            for future in as_completed(futures):
                asset = futures[future]
                try:
                    market = future.result()
                    if market:
                        discovered[asset] = market
                        logger.info(f"  {asset.upper()}: {market.slug} ({market.time_remaining_str})")
                except Exception as e:
                    logger.error(f"  {asset.upper()}: Error - {e}")

        self.markets = discovered
        self._last_refresh = time.time()

        logger.info(f"Discovered {len(discovered)}/{len(self.assets)} markets")
        return discovered

    def refresh_if_needed(self) -> bool:
        """
        Refresh markets if any have closed.

        Returns:
            True if markets were refreshed
        """
        # Check if any market is closed
        needs_refresh = False

        for asset, market in list(self.markets.items()):
            if not market.is_open:
                logger.info(f"{asset.upper()} market closed, needs refresh")
                needs_refresh = True
                break

        if needs_refresh:
            self.discover_all()
            return True

        return False

    def get_market(self, asset: str) -> Optional[Market]:
        """Get market for specific asset."""
        return self.markets.get(asset.lower())

    def get_open_markets(self) -> List[Market]:
        """Get all currently open markets."""
        return [m for m in self.markets.values() if m.is_open]

    def get_best_opportunity_market(self) -> Optional[Market]:
        """
        Get the market with most time remaining.

        More time = more chances for arbitrage.
        """
        open_markets = self.get_open_markets()
        if not open_markets:
            return None
        return max(open_markets, key=lambda m: m.time_remaining)

    def print_status(self):
        """Print status of all markets."""
        print("=" * 70)
        print("15-MINUTE CRYPTO MARKETS STATUS")
        print("=" * 70)

        for asset in self.assets:
            market = self.markets.get(asset)
            if market:
                status = "OPEN" if market.is_open else "CLOSED"
                icon = "" if market.is_open else ""
                print(f"  {icon} {asset.upper():<4} | {market.time_remaining_str:>8} | {market.slug}")
            else:
                print(f"   {asset.upper():<4} | NOT FOUND")

        print("=" * 70)


def discover_markets(assets: List[str] = None) -> MarketManager:
    """
    Convenience function to discover markets.

    Args:
        assets: List of assets to discover. Default: ["btc", "eth", "sol"]

    Returns:
        MarketManager with discovered markets
    """
    if assets is None:
        assets = SUPPORTED_ASSETS.copy()

    manager = MarketManager(assets=assets)
    manager.discover_all()
    return manager


if __name__ == "__main__":
    # Test market discovery
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\nTesting market discovery...\n")

    manager = discover_markets()
    manager.print_status()

    print("\nOpen markets:")
    for market in manager.get_open_markets():
        print(f"  - {market}")

    best = manager.get_best_opportunity_market()
    if best:
        print(f"\nBest opportunity: {best.asset.upper()} ({best.time_remaining_str} remaining)")
