"""
Real-time Data Collector for BTC Up/Down Markets
Collects data from Chainlink Oracle and Polymarket CLOB API
"""

import asyncio
import json
import gzip
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
import logging

try:
    import httpx
    import websockets
except ImportError:
    print("Please install required packages:")
    print("  pip install httpx websockets")
    sys.exit(1)

logger = logging.getLogger(__name__)

# Constants
FIFTEEN_MIN_MS = 15 * 60 * 1000

# API Endpoints
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"

# Chainlink BTC/USD Price Feed (Polygon)
CHAINLINK_BTC_USD_PROXY = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
POLYGON_RPC = "https://polygon-rpc.com"


@dataclass
class CollectorState:
    """State of the data collector."""
    chainlink_ticks: List[Dict[str, Any]] = field(default_factory=list)
    price_changes: List[Dict[str, Any]] = field(default_factory=list)
    order_books: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    # Current market info
    current_market_id: Optional[str] = None
    current_market_slug: Optional[str] = None
    up_token_id: Optional[str] = None
    down_token_id: Optional[str] = None

    # Stats
    start_time: float = field(default_factory=time.time)
    last_save_time: float = field(default_factory=time.time)
    tick_count: int = 0
    price_change_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'market_id': self.current_market_id,
                'market_slug': self.current_market_slug,
                'up_token_id': self.up_token_id,
                'down_token_id': self.down_token_id,
                'collection_started': datetime.fromtimestamp(self.start_time).isoformat(),
                'tick_count': self.tick_count,
                'price_change_count': self.price_change_count,
            },
            'chainlink_ticks': self.chainlink_ticks,
            'price_changes': self.price_changes,
            'order_books': self.order_books,
            'trades': self.trades,
        }

    def clear(self):
        """Clear collected data but keep market info."""
        self.chainlink_ticks = []
        self.price_changes = []
        self.order_books = []
        self.trades = []
        self.tick_count = 0
        self.price_change_count = 0
        self.start_time = time.time()


class ChainlinkCollector:
    """Collects BTC price from Chainlink Oracle."""

    def __init__(self, rpc_url: str = POLYGON_RPC):
        self.rpc_url = rpc_url
        self.client = httpx.AsyncClient(timeout=10.0)
        self.last_price: Optional[float] = None

    async def get_btc_price(self) -> Optional[Dict[str, Any]]:
        """
        Get current BTC price from Chainlink.

        Returns:
            Dict with 'ts', 'price', 'diff' or None on error
        """
        try:
            # Call latestRoundData() on Chainlink price feed
            # Function selector: 0xfeaf968c
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{
                    "to": CHAINLINK_BTC_USD_PROXY,
                    "data": "0xfeaf968c"  # latestRoundData()
                }, "latest"],
                "id": 1
            }

            response = await self.client.post(self.rpc_url, json=payload)
            result = response.json()

            if 'result' not in result:
                logger.warning(f"Chainlink error: {result}")
                return None

            # Parse result (returns: roundId, answer, startedAt, updatedAt, answeredInRound)
            data = result['result']
            if len(data) < 66:
                return None

            # Answer is at position 32-64 (after roundId)
            # Chainlink BTC/USD has 8 decimals
            answer_hex = data[66:130]
            price_raw = int(answer_hex, 16)
            price = price_raw / 1e8

            ts = int(time.time() * 1000)
            diff = price - self.last_price if self.last_price else 0
            self.last_price = price

            return {
                'ts': ts,
                'price': round(price, 2),
                'diff': round(diff, 2),
            }

        except Exception as e:
            logger.error(f"Error fetching Chainlink price: {e}")
            return None

    async def close(self):
        await self.client.aclose()


class PolymarketCollector:
    """Collects token prices from Polymarket CLOB API."""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.ws = None

    async def find_btc_market(self) -> Optional[Dict[str, Any]]:
        """
        Find the current active BTC Up/Down 15min market.

        Returns:
            Market info dict or None
        """
        try:
            # Search for BTC markets
            response = await self.client.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={
                    "closed": "false",
                    "limit": 50,
                }
            )
            markets = response.json()

            # Find BTC 15min up/down market
            for market in markets:
                question = market.get('question', '').lower()
                slug = market.get('slug', '').lower()

                if ('btc' in question or 'bitcoin' in question) and \
                   ('15' in question or '15min' in slug or '15-min' in slug) and \
                   ('up' in question or 'down' in question):
                    return market

            # Alternative: search by slug pattern
            response = await self.client.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={
                    "slug_contains": "btc-up",
                    "closed": "false",
                }
            )
            markets = response.json()
            if markets:
                return markets[0]

            return None

        except Exception as e:
            logger.error(f"Error finding BTC market: {e}")
            return None

    async def get_token_prices(
        self,
        up_token_id: str,
        down_token_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get current prices for UP and DOWN tokens.

        Returns:
            Dict with 'ts', 'up', 'down' or None
        """
        try:
            # Get order books for both tokens
            up_book = await self._get_order_book(up_token_id)
            down_book = await self._get_order_book(down_token_id)

            if not up_book or not down_book:
                return None

            # Get best ask prices (what you'd pay to buy)
            up_price = self._get_best_ask(up_book)
            down_price = self._get_best_ask(down_book)

            if up_price is None or down_price is None:
                return None

            return {
                'ts': int(time.time() * 1000),
                'up': up_price,
                'down': down_price,
            }

        except Exception as e:
            logger.error(f"Error fetching token prices: {e}")
            return None

    async def _get_order_book(self, token_id: str) -> Optional[Dict]:
        """Get order book for a token."""
        try:
            response = await self.client.get(
                f"{POLYMARKET_CLOB_API}/book",
                params={"token_id": token_id}
            )
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def _get_best_ask(self, book: Dict) -> Optional[float]:
        """Get best ask price from order book."""
        asks = book.get('asks', [])
        if not asks:
            return None

        # Asks are sorted by price ascending
        best_ask = min(asks, key=lambda x: float(x.get('price', 999)))
        return float(best_ask.get('price', 0))

    def _get_best_bid(self, book: Dict) -> Optional[float]:
        """Get best bid price from order book."""
        bids = book.get('bids', [])
        if not bids:
            return None

        best_bid = max(bids, key=lambda x: float(x.get('price', 0)))
        return float(best_bid.get('price', 0))

    async def get_full_order_book(
        self,
        up_token_id: str,
        down_token_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get full order books for both tokens."""
        try:
            up_book = await self._get_order_book(up_token_id)
            down_book = await self._get_order_book(down_token_id)

            if not up_book or not down_book:
                return None

            return {
                'ts': int(time.time() * 1000),
                'up': up_book,
                'down': down_book,
            }
        except Exception as e:
            logger.error(f"Error fetching order books: {e}")
            return None

    async def close(self):
        await self.client.aclose()


class DataCollectorService:
    """Main service that coordinates data collection."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        save_interval: int = 300,  # Save every 5 minutes
        chainlink_interval: float = 1.0,  # Poll Chainlink every 1 second
        token_price_interval: float = 2.0,  # Poll token prices every 2 seconds
        order_book_interval: float = 30.0,  # Full order book every 30 seconds
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.chainlink_interval = chainlink_interval
        self.token_price_interval = token_price_interval
        self.order_book_interval = order_book_interval

        self.chainlink = ChainlinkCollector()
        self.polymarket = PolymarketCollector()
        self.state = CollectorState()

        self.running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the data collection service."""
        logger.info("Starting data collection service...")

        # Find current BTC market
        await self._find_and_set_market()

        if not self.state.up_token_id:
            logger.error("Could not find BTC market. Exiting.")
            return

        self.running = True

        # Start collection tasks
        self._tasks = [
            asyncio.create_task(self._collect_chainlink_loop()),
            asyncio.create_task(self._collect_token_prices_loop()),
            asyncio.create_task(self._collect_order_books_loop()),
            asyncio.create_task(self._save_loop()),
            asyncio.create_task(self._market_refresh_loop()),
        ]

        logger.info("Data collection started!")
        logger.info(f"Market: {self.state.current_market_slug}")
        logger.info(f"UP Token: {self.state.up_token_id}")
        logger.info(f"DOWN Token: {self.state.down_token_id}")

        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Collection tasks cancelled")

    async def stop(self):
        """Stop the service and save remaining data."""
        logger.info("Stopping data collection...")
        self.running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Final save
        await self._save_data()

        # Cleanup
        await self.chainlink.close()
        await self.polymarket.close()

        logger.info("Data collection stopped")

    async def _find_and_set_market(self):
        """Find current BTC market and set token IDs."""
        market = await self.polymarket.find_btc_market()

        if not market:
            logger.warning("No active BTC 15min market found")
            return

        self.state.current_market_id = market.get('id') or market.get('condition_id')
        self.state.current_market_slug = market.get('slug')

        # Get token IDs from market
        tokens = market.get('tokens', [])
        for token in tokens:
            outcome = token.get('outcome', '').lower()
            token_id = token.get('token_id')

            if 'up' in outcome or 'yes' in outcome:
                self.state.up_token_id = token_id
            elif 'down' in outcome or 'no' in outcome:
                self.state.down_token_id = token_id

        # Alternative: try to get from clobTokenIds
        if not self.state.up_token_id:
            clob_tokens = market.get('clobTokenIds', [])
            if len(clob_tokens) >= 2:
                self.state.up_token_id = clob_tokens[0]
                self.state.down_token_id = clob_tokens[1]

    async def _collect_chainlink_loop(self):
        """Continuously collect Chainlink price data."""
        while self.running:
            try:
                tick = await self.chainlink.get_btc_price()
                if tick:
                    self.state.chainlink_ticks.append(tick)
                    self.state.tick_count += 1

                    if self.state.tick_count % 60 == 0:
                        logger.info(
                            f"Collected {self.state.tick_count} ticks, "
                            f"BTC: ${tick['price']:,.2f}"
                        )

                await asyncio.sleep(self.chainlink_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Chainlink collection error: {e}")
                await asyncio.sleep(5)

    async def _collect_token_prices_loop(self):
        """Continuously collect token prices."""
        while self.running:
            try:
                if self.state.up_token_id and self.state.down_token_id:
                    prices = await self.polymarket.get_token_prices(
                        self.state.up_token_id,
                        self.state.down_token_id
                    )
                    if prices:
                        self.state.price_changes.append(prices)
                        self.state.price_change_count += 1

                await asyncio.sleep(self.token_price_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token price collection error: {e}")
                await asyncio.sleep(5)

    async def _collect_order_books_loop(self):
        """Periodically collect full order books."""
        while self.running:
            try:
                if self.state.up_token_id and self.state.down_token_id:
                    books = await self.polymarket.get_full_order_book(
                        self.state.up_token_id,
                        self.state.down_token_id
                    )
                    if books:
                        self.state.order_books.append(books)

                await asyncio.sleep(self.order_book_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order book collection error: {e}")
                await asyncio.sleep(10)

    async def _save_loop(self):
        """Periodically save collected data."""
        while self.running:
            try:
                await asyncio.sleep(self.save_interval)
                await self._save_data()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Save error: {e}")

    async def _market_refresh_loop(self):
        """Periodically check for new market (every 15 minutes)."""
        while self.running:
            try:
                await asyncio.sleep(900)  # 15 minutes

                old_market = self.state.current_market_id
                await self._find_and_set_market()

                if self.state.current_market_id != old_market:
                    logger.info(f"New market detected: {self.state.current_market_slug}")
                    # Save old data and clear
                    await self._save_data()
                    self.state.clear()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market refresh error: {e}")

    async def _save_data(self):
        """Save current data to file."""
        if not self.state.chainlink_ticks:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"btc_data_{timestamp}"
        filepath = self.output_dir / f"{filename}.json.gz"

        data = self.state.to_dict()

        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        logger.info(
            f"Saved {len(self.state.chainlink_ticks)} ticks, "
            f"{len(self.state.price_changes)} prices to {filepath}"
        )

        self.state.last_save_time = time.time()


async def main():
    """Main entry point for the collector."""
    import argparse

    parser = argparse.ArgumentParser(description='BTC Data Collector for Polymarket')
    parser.add_argument('--output', '-o', default='data/raw', help='Output directory')
    parser.add_argument('--save-interval', type=int, default=300, help='Save interval in seconds')
    parser.add_argument('--duration', type=int, default=0, help='Collection duration in minutes (0=infinite)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    collector = DataCollectorService(
        output_dir=args.output,
        save_interval=args.save_interval,
    )

    # Handle Ctrl+C
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal...")
        asyncio.create_task(collector.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        if args.duration > 0:
            # Run for specified duration
            task = asyncio.create_task(collector.start())
            await asyncio.sleep(args.duration * 60)
            await collector.stop()
        else:
            # Run indefinitely
            await collector.start()
    except Exception as e:
        logger.error(f"Error: {e}")
        await collector.stop()


if __name__ == '__main__':
    asyncio.run(main())
