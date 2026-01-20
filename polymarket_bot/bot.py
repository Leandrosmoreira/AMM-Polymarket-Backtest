"""
Polymarket Arbitrage Bot.

This is a simplified version of the trading bot that uses the working
authentication pattern from exemplo_polymarket.

Usage:
    python -m polymarket_bot.bot
    python -m polymarket_bot.bot --test-auth
"""

import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Optional

import httpx

from .config import load_settings, Settings
from .auth import get_client, get_balance, test_auth
from .trading import (
    get_client as trading_get_client,
    place_orders_fast,
    extract_order_id,
    wait_for_terminal_order,
    cancel_orders,
    get_positions,
)
from .lookup import fetch_market_from_slug

logger = logging.getLogger(__name__)


def find_current_btc_15min_market() -> str:
    """Find the current active BTC 15min market on Polymarket."""
    logger.info("Searching for current BTC 15min market...")

    try:
        page_url = "https://polymarket.com/crypto/15M"
        resp = httpx.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()

        pattern = r'btc-updown-15m-(\d+)'
        matches = re.findall(pattern, resp.text)

        if not matches:
            raise RuntimeError("No active BTC 15min market found")

        now_ts = int(datetime.now().timestamp())
        all_ts = sorted((int(ts) for ts in matches), reverse=True)
        open_ts = [ts for ts in all_ts if now_ts < (ts + 900)]
        chosen_ts = open_ts[0] if open_ts else all_ts[0]
        slug = f"btc-updown-15m-{chosen_ts}"

        logger.info(f"Market found: {slug}")
        return slug

    except Exception as e:
        logger.error(f"Error searching for BTC 15min market: {e}")
        raise


class SimpleArbitrageBot:
    """Simple arbitrage bot for BTC 15min markets."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = trading_get_client(settings)

        # Find market
        if not settings.dry_run:
            market_slug = find_current_btc_15min_market()
        else:
            try:
                market_slug = find_current_btc_15min_market()
            except Exception:
                import time as t
                market_slug = f"btc-updown-15m-{int(t.time())}"
                logger.info(f"Using simulated market: {market_slug}")

        # Get token IDs
        if "SIMULATED" in market_slug or settings.dry_run:
            self.market_id = f"SIM_{market_slug}"
            self.yes_token_id = "SIMULATED_YES"
            self.no_token_id = "SIMULATED_NO"
        else:
            market_info = fetch_market_from_slug(market_slug)
            self.market_id = market_info["market_id"]
            self.yes_token_id = market_info["yes_token_id"]
            self.no_token_id = market_info["no_token_id"]

        self.market_slug = market_slug

        # Extract end timestamp
        match = re.search(r'btc-updown-15m-(\d+)', market_slug)
        market_start = int(match.group(1)) if match else None
        self.market_end_timestamp = market_start + 900 if market_start else None

        # Stats
        self.opportunities_found = 0
        self.trades_executed = 0
        self.total_invested = 0.0

        # Simulation balance
        self.sim_balance = settings.sim_balance
        self.sim_start_balance = settings.sim_balance

        # Cooldown
        self._last_execution_ts = 0.0

        logger.info("=" * 60)
        logger.info("BOT INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Market: {self.market_slug}")
        logger.info(f"Mode: {'SIMULATION' if settings.dry_run else 'LIVE'}")
        logger.info(f"Threshold: ${settings.target_pair_cost:.4f}")
        logger.info(f"Order size: {settings.order_size}")
        logger.info("=" * 60)

    def get_time_remaining(self) -> str:
        """Get remaining time until market closes."""
        if not self.market_end_timestamp:
            return "Unknown"
        now = int(datetime.now().timestamp())
        remaining = self.market_end_timestamp - now
        if remaining <= 0:
            return "CLOSED"
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes}m {seconds}s"

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

    def check_arbitrage(self) -> Optional[dict]:
        """Check if arbitrage opportunity exists."""
        up_book = self.get_order_book(self.yes_token_id)
        down_book = self.get_order_book(self.no_token_id)

        price_up = up_book.get("best_ask")
        price_down = down_book.get("best_ask")

        if price_up is None or price_down is None:
            return None

        total_cost = price_up + price_down

        if total_cost <= self.settings.target_pair_cost:
            profit = 1.0 - total_cost
            profit_pct = (profit / total_cost) * 100

            return {
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
        """Execute arbitrage."""
        # Cooldown check
        now = time.time()
        if (now - self._last_execution_ts) < self.settings.cooldown_seconds:
            logger.info(f"Cooldown active; skipping")
            return
        self._last_execution_ts = now

        self.opportunities_found += 1

        logger.info("=" * 60)
        logger.info("ARBITRAGE OPPORTUNITY!")
        logger.info("=" * 60)
        logger.info(f"UP:   ${opportunity['price_up']:.4f}")
        logger.info(f"DOWN: ${opportunity['price_down']:.4f}")
        logger.info(f"Total: ${opportunity['total_cost']:.4f}")
        logger.info(f"Profit: ${opportunity['profit_per_share']:.4f} ({opportunity['profit_pct']:.2f}%)")
        logger.info(f"Investment: ${opportunity['total_investment']:.2f}")
        logger.info("=" * 60)

        if self.settings.dry_run:
            logger.info("[SIMULATION] - No real orders")
            if self.sim_balance >= opportunity['total_investment']:
                self.sim_balance -= opportunity['total_investment']
                self.total_invested += opportunity['total_investment']
                self.trades_executed += 1
                logger.info(f"Simulated balance: ${self.sim_balance:.2f}")
            else:
                logger.warning(f"Insufficient balance: ${self.sim_balance:.2f}")
            return

        # Real trading
        try:
            orders = [
                {"side": "BUY", "token_id": self.yes_token_id,
                 "price": opportunity['price_up'], "size": self.settings.order_size},
                {"side": "BUY", "token_id": self.no_token_id,
                 "price": opportunity['price_down'], "size": self.settings.order_size},
            ]

            results = place_orders_fast(self.settings, orders, order_type=self.settings.order_type)

            order_ids = [extract_order_id(r) for r in (results or [])]

            if all(order_ids):
                logger.info("ORDERS SUBMITTED")
                self.trades_executed += 1
                self.total_invested += opportunity['total_investment']
            else:
                logger.error(f"Order submission failed: {results}")

        except Exception as e:
            logger.error(f"Execution error: {e}")

    async def run(self, interval: float = 1.0):
        """Run the bot."""
        logger.info("Bot started")

        try:
            while True:
                time_remaining = self.get_time_remaining()
                if time_remaining == "CLOSED":
                    logger.info("Market closed")
                    break

                opportunity = self.check_arbitrage()
                if opportunity:
                    self.execute_arbitrage(opportunity)
                else:
                    up_book = self.get_order_book(self.yes_token_id)
                    down_book = self.get_order_book(self.no_token_id)
                    price_up = up_book.get("best_ask")
                    price_down = down_book.get("best_ask")
                    if price_up and price_down:
                        total = price_up + price_down
                        logger.info(
                            f"No arb: ${price_up:.4f} + ${price_down:.4f} = ${total:.4f} "
                            f"(need <${self.settings.target_pair_cost:.4f}) [{time_remaining}]"
                        )

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("Stopped by user")

        # Summary
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Opportunities: {self.opportunities_found}")
        logger.info(f"Trades: {self.trades_executed}")
        logger.info(f"Invested: ${self.total_invested:.2f}")
        if self.settings.dry_run:
            expected_profit = self.trades_executed * 0.009 * self.settings.order_size
            logger.info(f"Expected profit: ${expected_profit:.2f}")
            logger.info(f"Final balance: ${self.sim_balance:.2f}")
        logger.info("=" * 60)


async def main():
    """Main entry point."""
    import sys

    # Check for --test-auth flag
    if "--test-auth" in sys.argv:
        success = test_auth("pmpe.env")
        sys.exit(0 if success else 1)

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
    if not is_valid:
        logger.error("Configuration errors:")
        for err in errors:
            logger.error(f"  - {err}")
        logger.error("\nPlease check your pmpe.env file")
        return

    settings.print_summary()

    # Run bot
    bot = SimpleArbitrageBot(settings)
    await bot.run(interval=1.0)


if __name__ == "__main__":
    asyncio.run(main())
