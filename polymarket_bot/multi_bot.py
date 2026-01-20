"""
Multi-Market Arbitrage Bot for Polymarket 15min crypto markets.

Monitors BTC, ETH, and SOL simultaneously for arbitrage opportunities.
Automatically handles market rollover every 15 minutes.

Usage:
    python -m polymarket_bot.multi_bot
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from .config import load_settings, Settings
from .trading import get_client, place_orders_fast, extract_order_id, get_balance
from .markets import MarketManager, Market, SUPPORTED_ASSETS, discover_markets
from .fast_logger import FastTradeLogger

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """State tracking for a single market."""
    market: Market
    opportunities_found: int = 0
    trades_executed: int = 0
    total_invested: float = 0.0
    last_scan: float = 0.0


class MultiMarketBot:
    """
    Arbitrage bot that monitors multiple 15min markets simultaneously.

    Features:
    - Monitors BTC, ETH, SOL in parallel
    - Automatic market discovery and rollover
    - Shared balance management
    - Fast execution with orjson logging
    """

    def __init__(self, settings: Settings, assets: List[str] = None):
        self.settings = settings
        self.assets = assets or SUPPORTED_ASSETS.copy()
        self.client = get_client(settings)

        # Market management
        self.market_manager = MarketManager(assets=self.assets)
        self.market_states: Dict[str, MarketState] = {}

        # Global stats
        self.total_opportunities = 0
        self.total_trades = 0
        self.total_invested = 0.0

        # Balance tracking
        self.sim_balance = settings.sim_balance if settings.dry_run else 0.0
        self.sim_start_balance = self.sim_balance

        # Cooldown per market
        self._last_execution: Dict[str, float] = {}

        # Fast logger with JSONL
        self.fast_logger = FastTradeLogger(log_dir="logs", format="jsonl")

        logger.info("=" * 70)
        logger.info("MULTI-MARKET ARBITRAGE BOT")
        logger.info("=" * 70)
        logger.info(f"Assets: {', '.join(a.upper() for a in self.assets)}")
        logger.info(f"Mode: {'SIMULATION' if settings.dry_run else 'LIVE'}")
        logger.info(f"Threshold: ${settings.target_pair_cost:.4f}")
        logger.info(f"Order size: {settings.order_size}")
        logger.info("=" * 70)

    def discover_markets(self):
        """Discover all active markets."""
        self.market_manager.discover_all()

        # Initialize state for each market
        for asset, market in self.market_manager.markets.items():
            if asset not in self.market_states:
                self.market_states[asset] = MarketState(market=market)
            else:
                # Update market but keep stats
                self.market_states[asset].market = market

    def get_order_book(self, token_id: str) -> dict:
        """Get order book for a token."""
        try:
            book = self.client.get_order_book(token_id=token_id)
            bids = book.bids if hasattr(book, 'bids') and book.bids else []
            asks = book.asks if hasattr(book, 'asks') and book.asks else []

            bid_levels = []
            for level in bids:
                try:
                    price = float(level.price)
                    size = float(level.size)
                    if size > 0:
                        bid_levels.append((price, size))
                except Exception:
                    continue

            ask_levels = []
            for level in asks:
                try:
                    price = float(level.price)
                    size = float(level.size)
                    if size > 0:
                        ask_levels.append((price, size))
                except Exception:
                    continue

            best_bid = max((p for p, _ in bid_levels), default=None)
            best_ask = min((p for p, _ in ask_levels), default=None)

            return {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bids": bid_levels,
                "asks": ask_levels,
            }
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return {}

    def check_arbitrage(self, market: Market) -> Optional[dict]:
        """Check if arbitrage opportunity exists for a market."""
        up_book = self.get_order_book(market.yes_token_id)
        down_book = self.get_order_book(market.no_token_id)

        price_up = up_book.get("best_ask")
        price_down = down_book.get("best_ask")

        if price_up is None or price_down is None:
            return None

        total_cost = price_up + price_down

        if total_cost <= self.settings.target_pair_cost:
            profit = 1.0 - total_cost
            profit_pct = (profit / total_cost) * 100

            return {
                "market": market,
                "price_up": price_up,
                "price_down": price_down,
                "total_cost": total_cost,
                "profit_per_share": profit,
                "profit_pct": profit_pct,
                "order_size": self.settings.order_size,
                "total_investment": total_cost * self.settings.order_size,
                "expected_profit": profit * self.settings.order_size,
            }

        return None

    def execute_arbitrage(self, opportunity: dict):
        """Execute arbitrage trade."""
        market: Market = opportunity["market"]
        asset = market.asset

        # Cooldown check
        now = time.time()
        last_exec = self._last_execution.get(asset, 0)
        if (now - last_exec) < self.settings.cooldown_seconds:
            logger.debug(f"{asset.upper()}: Cooldown active")
            return

        self._last_execution[asset] = now

        # Update stats
        state = self.market_states.get(asset)
        if state:
            state.opportunities_found += 1
        self.total_opportunities += 1

        logger.info("=" * 70)
        logger.info(f"ARBITRAGE: {asset.upper()} ({market.time_remaining_str})")
        logger.info("=" * 70)
        logger.info(f"UP:   ${opportunity['price_up']:.4f}")
        logger.info(f"DOWN: ${opportunity['price_down']:.4f}")
        logger.info(f"Total: ${opportunity['total_cost']:.4f}")
        logger.info(f"Profit: ${opportunity['profit_per_share']:.4f} ({opportunity['profit_pct']:.2f}%)")
        logger.info("=" * 70)

        # Log to JSONL
        self.fast_logger.log_trade(
            market=market.slug,
            price_up=opportunity['price_up'],
            price_down=opportunity['price_down'],
            pair_cost=opportunity['total_cost'],
            profit_pct=opportunity['profit_pct'],
            order_size=opportunity['order_size'],
            investment=opportunity['total_investment'],
            expected_profit=opportunity['expected_profit'],
            balance_after=self.sim_balance - opportunity['total_investment'],
        )

        if self.settings.dry_run:
            logger.info("[SIMULATION]")
            if self.sim_balance >= opportunity['total_investment']:
                self.sim_balance -= opportunity['total_investment']
                self.total_invested += opportunity['total_investment']
                self.total_trades += 1
                if state:
                    state.trades_executed += 1
                    state.total_invested += opportunity['total_investment']
                logger.info(f"Balance: ${self.sim_balance:.2f}")
            else:
                logger.warning(f"Insufficient balance: ${self.sim_balance:.2f}")
            return

        # Real trading
        try:
            orders = [
                {"side": "BUY", "token_id": market.yes_token_id,
                 "price": opportunity['price_up'], "size": self.settings.order_size},
                {"side": "BUY", "token_id": market.no_token_id,
                 "price": opportunity['price_down'], "size": self.settings.order_size},
            ]

            results = place_orders_fast(self.settings, orders, order_type=self.settings.order_type)
            order_ids = [extract_order_id(r) for r in (results or [])]

            if all(order_ids):
                logger.info("ORDERS SUBMITTED")
                self.total_trades += 1
                self.total_invested += opportunity['total_investment']
                if state:
                    state.trades_executed += 1
                    state.total_invested += opportunity['total_investment']
            else:
                logger.error(f"Order failed: {results}")

        except Exception as e:
            logger.error(f"Execution error: {e}")

    async def scan_market(self, market: Market) -> Optional[dict]:
        """Scan a single market for opportunities."""
        if not market.is_open:
            return None

        opportunity = self.check_arbitrage(market)

        # Update scan state
        state = self.market_states.get(market.asset)
        if state:
            state.last_scan = time.time()

        return opportunity

    async def scan_all_markets(self) -> List[dict]:
        """Scan all markets in parallel."""
        opportunities = []

        # Get open markets
        open_markets = self.market_manager.get_open_markets()

        if not open_markets:
            logger.warning("No open markets!")
            return opportunities

        # Scan in parallel using threads (API calls are blocking)
        loop = asyncio.get_event_loop()

        async def scan_wrapper(market):
            return await loop.run_in_executor(None, self.check_arbitrage, market)

        tasks = [scan_wrapper(m) for m in open_markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for market, result in zip(open_markets, results):
            if isinstance(result, Exception):
                logger.error(f"{market.asset.upper()}: Scan error - {result}")
            elif result:
                opportunities.append(result)
            else:
                # Log no-opportunity scan
                up_book = self.get_order_book(market.yes_token_id)
                down_book = self.get_order_book(market.no_token_id)
                price_up = up_book.get("best_ask")
                price_down = down_book.get("best_ask")

                if price_up and price_down:
                    total = price_up + price_down
                    logger.info(
                        f"{market.asset.upper()}: ${price_up:.4f} + ${price_down:.4f} = ${total:.4f} "
                        f"(need <${self.settings.target_pair_cost:.4f}) [{market.time_remaining_str}]"
                    )

        return opportunities

    def print_status(self):
        """Print current status of all markets."""
        print()
        print("=" * 70)
        print(f"STATUS @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)

        for asset in self.assets:
            state = self.market_states.get(asset)
            if state and state.market:
                market = state.market
                status = "" if market.is_open else ""
                print(
                    f"  {status} {asset.upper():<4} | "
                    f"{market.time_remaining_str:>8} | "
                    f"Opps: {state.opportunities_found} | "
                    f"Trades: {state.trades_executed}"
                )
            else:
                print(f"   {asset.upper():<4} | NOT FOUND")

        print("-" * 70)
        print(f"  Total: {self.total_opportunities} opportunities, {self.total_trades} trades")
        if self.settings.dry_run:
            print(f"  Balance: ${self.sim_balance:.2f} (started: ${self.sim_start_balance:.2f})")
        print("=" * 70)

    async def run(self, scan_interval: float = 1.0):
        """
        Main loop - scan all markets continuously.

        Args:
            scan_interval: Seconds between scans
        """
        logger.info("Starting multi-market scanner...")

        # Initial discovery
        self.discover_markets()

        scan_count = 0
        last_status_print = 0

        try:
            while True:
                scan_count += 1

                # Refresh markets if any closed
                if self.market_manager.refresh_if_needed():
                    # Re-discover to get new markets
                    self.discover_markets()

                # Scan all markets
                opportunities = await self.scan_all_markets()

                # Execute any opportunities found (best one first)
                if opportunities:
                    # Sort by profit percentage
                    opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
                    for opp in opportunities:
                        self.execute_arbitrage(opp)

                # Print status every 30 seconds
                now = time.time()
                if (now - last_status_print) >= 30:
                    self.print_status()
                    last_status_print = now

                # Wait before next scan
                await asyncio.sleep(scan_interval)

        except KeyboardInterrupt:
            logger.info("\nStopped by user")

        finally:
            self.show_final_summary()

    def show_final_summary(self):
        """Show final summary."""
        print()
        print("=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        for asset in self.assets:
            state = self.market_states.get(asset)
            if state:
                print(f"  {asset.upper()}: {state.opportunities_found} opps, {state.trades_executed} trades, ${state.total_invested:.2f} invested")

        print("-" * 70)
        print(f"  TOTAL: {self.total_opportunities} opportunities")
        print(f"  TOTAL: {self.total_trades} trades")
        print(f"  TOTAL: ${self.total_invested:.2f} invested")

        if self.settings.dry_run and self.total_trades > 0:
            # Estimate profit (each trade should yield ~$0.009 per share)
            expected_profit = self.total_trades * 0.009 * self.settings.order_size
            print(f"  EXPECTED PROFIT: ${expected_profit:.2f}")
            print(f"  FINAL BALANCE: ${self.sim_balance:.2f}")

        print("=" * 70)

        # Stop logger
        self.fast_logger.stop()
        stats = self.fast_logger.get_stats()
        print(f"\nLogs saved to: {stats['trades_file']}")


async def main():
    """Entry point."""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Load settings
    settings = load_settings()

    # Validate
    is_valid, errors = settings.validate()
    if not is_valid and not settings.dry_run:
        logger.error("Configuration errors:")
        for err in errors:
            logger.error(f"  - {err}")
        return

    # Parse assets from command line or use all
    assets = None
    if len(sys.argv) > 1:
        assets = [a.lower() for a in sys.argv[1:] if a.lower() in SUPPORTED_ASSETS]
        if not assets:
            assets = None

    # Create and run bot
    bot = MultiMarketBot(settings, assets=assets)
    await bot.run(scan_interval=1.0)


if __name__ == "__main__":
    asyncio.run(main())
