"""
Spread Monitor for Gabagool Bot
Monitors order books and detects spread opportunities
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .polymarket_client import PolymarketClient, Market, OrderBook
from .config import GabagoolConfig

logger = logging.getLogger(__name__)


@dataclass
class SpreadOpportunity:
    """Represents a spread capture opportunity."""
    market: Market
    up_token_id: str
    down_token_id: str
    up_price: float
    down_price: float
    total_price: float
    spread: float
    spread_pct: float
    up_liquidity: float
    down_liquidity: float
    timestamp: int
    market_time_remaining: int  # seconds

    @property
    def is_valid(self) -> bool:
        """Check if opportunity is still valid."""
        return self.spread > 0 and self.up_liquidity > 0 and self.down_liquidity > 0

    @property
    def max_size(self) -> float:
        """Maximum size we can trade based on liquidity."""
        return min(self.up_liquidity, self.down_liquidity)

    @property
    def expected_profit_per_pair(self) -> float:
        """Expected profit per pair of shares."""
        return 1.0 - self.total_price

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market.id,
            'market_slug': self.market.slug,
            'up_price': self.up_price,
            'down_price': self.down_price,
            'total_price': self.total_price,
            'spread': self.spread,
            'spread_pct': self.spread_pct,
            'up_liquidity': self.up_liquidity,
            'down_liquidity': self.down_liquidity,
            'timestamp': self.timestamp,
            'time_remaining': self.market_time_remaining,
        }


@dataclass
class MarketState:
    """Current state of a market being monitored."""
    market: Market
    up_token_id: str
    down_token_id: str
    start_time: int  # Unix timestamp ms
    end_time: int    # Unix timestamp ms
    last_up_price: Optional[float] = None
    last_down_price: Optional[float] = None
    last_spread: Optional[float] = None
    last_update: int = 0
    opportunities_found: int = 0
    total_spread_captured: float = 0.0

    @property
    def time_elapsed_seconds(self) -> int:
        """Seconds elapsed since market start."""
        now = int(time.time() * 1000)
        return max(0, (now - self.start_time) // 1000)

    @property
    def time_remaining_seconds(self) -> int:
        """Seconds remaining until market end."""
        now = int(time.time() * 1000)
        return max(0, (self.end_time - now) // 1000)

    @property
    def is_active(self) -> bool:
        """Check if market is currently active."""
        now = int(time.time() * 1000)
        return self.start_time <= now <= self.end_time


class SpreadMonitor:
    """
    Monitors multiple markets for spread opportunities.

    Features:
    - Async monitoring of multiple markets
    - Dynamic spread thresholds based on time
    - Callback on opportunity detection
    - Rate limit handling
    """

    def __init__(
        self,
        client: PolymarketClient,
        config: GabagoolConfig,
        on_opportunity: Optional[Callable[[SpreadOpportunity], None]] = None,
    ):
        self.client = client
        self.config = config
        self.on_opportunity = on_opportunity

        self._markets: Dict[str, MarketState] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_scan = 0

        # Statistics
        self.total_opportunities = 0
        self.total_checks = 0
        self.start_time = 0

    async def start(self):
        """Start the spread monitor."""
        logger.info("Starting spread monitor...")
        self._running = True
        self.start_time = int(time.time() * 1000)

        # Initial market scan
        await self._scan_markets()

        # Start monitoring loop
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the spread monitor."""
        logger.info("Stopping spread monitor...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Check if we need to scan for new markets
                now = int(time.time() * 1000)
                if now - self._last_scan > 60000:  # Every 60 seconds
                    await self._scan_markets()
                    self._last_scan = now

                # Check spreads for all active markets
                await self._check_all_spreads()

                # Wait before next check
                await asyncio.sleep(self.config.CHECK_INTERVAL_MS / 1000)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1)

    async def _scan_markets(self):
        """Scan for new markets to monitor."""
        logger.debug("Scanning for markets...")

        for asset in self.config.ENABLED_ASSETS:
            try:
                markets = await self.client.find_updown_markets(
                    asset=asset,
                    timeframe=self.config.MARKET_TIMEFRAME,
                )

                for market in markets:
                    if market.id not in self._markets:
                        await self._add_market(market)

            except Exception as e:
                logger.error(f"Error scanning {asset} markets: {e}")

        # Remove expired markets
        self._cleanup_expired_markets()

        logger.info(f"Monitoring {len(self._markets)} markets")

    async def _add_market(self, market: Market):
        """Add a market to monitor."""
        up_token = market.up_token_id
        down_token = market.down_token_id

        if not up_token or not down_token:
            logger.warning(f"Market {market.slug} missing token IDs")
            return

        # Estimate market times from slug or use defaults
        now = int(time.time() * 1000)
        duration = self.config.MARKET_DURATION_SECONDS * 1000

        # Try to parse end time from market data
        end_time = now + duration  # Default
        if market.end_date:
            try:
                end_dt = datetime.fromisoformat(market.end_date.replace('Z', '+00:00'))
                end_time = int(end_dt.timestamp() * 1000)
            except:
                pass

        start_time = end_time - duration

        state = MarketState(
            market=market,
            up_token_id=up_token,
            down_token_id=down_token,
            start_time=start_time,
            end_time=end_time,
        )

        self._markets[market.id] = state
        logger.info(f"Added market: {market.slug}")

    def _cleanup_expired_markets(self):
        """Remove expired markets."""
        now = int(time.time() * 1000)
        expired = [
            mid for mid, state in self._markets.items()
            if state.end_time < now
        ]

        for mid in expired:
            del self._markets[mid]
            logger.debug(f"Removed expired market: {mid}")

    async def _check_all_spreads(self):
        """Check spreads for all active markets."""
        active_markets = [
            state for state in self._markets.values()
            if state.is_active
        ]

        if not active_markets:
            return

        # Check spreads in parallel (with rate limit consideration)
        tasks = []
        for state in active_markets:
            tasks.append(self._check_spread(state))

        await asyncio.gather(*tasks, return_exceptions=True)
        self.total_checks += len(active_markets)

    async def _check_spread(self, state: MarketState):
        """Check spread for a single market."""
        try:
            # Get current prices
            up_price, down_price, total = await self.client.get_spread(
                state.up_token_id,
                state.down_token_id,
            )

            if up_price is None or down_price is None:
                return

            # Update state
            state.last_up_price = up_price
            state.last_down_price = down_price
            state.last_spread = 1.0 - total if total else None
            state.last_update = int(time.time() * 1000)

            # Check if opportunity exists
            spread = 1.0 - total
            spread_pct = spread / total if total > 0 else 0

            # Get dynamic threshold
            threshold = self.config.get_entry_threshold(state.time_elapsed_seconds)

            if spread_pct >= threshold:
                # Get liquidity info
                up_book = await self.client.get_order_book(state.up_token_id)
                down_book = await self.client.get_order_book(state.down_token_id)

                up_liquidity = up_book.best_ask_size or 0
                down_liquidity = down_book.best_ask_size or 0

                # Check minimum liquidity
                if min(up_liquidity, down_liquidity) < self.config.MIN_LIQUIDITY_USD / max(up_price, down_price):
                    return

                opportunity = SpreadOpportunity(
                    market=state.market,
                    up_token_id=state.up_token_id,
                    down_token_id=state.down_token_id,
                    up_price=up_price,
                    down_price=down_price,
                    total_price=total,
                    spread=spread,
                    spread_pct=spread_pct,
                    up_liquidity=up_liquidity,
                    down_liquidity=down_liquidity,
                    timestamp=int(time.time() * 1000),
                    market_time_remaining=state.time_remaining_seconds,
                )

                self.total_opportunities += 1
                state.opportunities_found += 1

                logger.info(
                    f"Opportunity: {state.market.slug} | "
                    f"Spread: {spread_pct*100:.2f}% | "
                    f"UP: {up_price:.3f} DOWN: {down_price:.3f}"
                )

                # Callback
                if self.on_opportunity:
                    self.on_opportunity(opportunity)

        except Exception as e:
            logger.error(f"Error checking spread for {state.market.slug}: {e}")

    def get_current_opportunities(self) -> List[SpreadOpportunity]:
        """Get all current opportunities across markets."""
        opportunities = []

        for state in self._markets.values():
            if not state.is_active or state.last_spread is None:
                continue

            threshold = self.config.get_entry_threshold(state.time_elapsed_seconds)

            if state.last_spread >= threshold:
                opportunities.append(SpreadOpportunity(
                    market=state.market,
                    up_token_id=state.up_token_id,
                    down_token_id=state.down_token_id,
                    up_price=state.last_up_price or 0,
                    down_price=state.last_down_price or 0,
                    total_price=(state.last_up_price or 0) + (state.last_down_price or 0),
                    spread=state.last_spread,
                    spread_pct=state.last_spread / ((state.last_up_price or 0) + (state.last_down_price or 0.01)),
                    up_liquidity=0,
                    down_liquidity=0,
                    timestamp=state.last_update,
                    market_time_remaining=state.time_remaining_seconds,
                ))

        return opportunities

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        runtime = (int(time.time() * 1000) - self.start_time) / 1000 if self.start_time else 0

        return {
            'markets_monitored': len(self._markets),
            'total_opportunities': self.total_opportunities,
            'total_checks': self.total_checks,
            'runtime_seconds': runtime,
            'opportunities_per_minute': (self.total_opportunities / runtime * 60) if runtime > 0 else 0,
        }
