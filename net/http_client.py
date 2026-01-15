"""
PolymarketHTTP - HTTP client for Polymarket APIs
Handles market discovery and order submission
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from core.types import MarketInfo
from config.gabagool_config import GabagoolConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Raw market data from API."""
    market_id: str
    condition_id: str
    question: str
    tokens: List[Dict[str, str]]
    end_date: str
    active: bool
    volume: float
    liquidity: float


class PolymarketHTTP:
    """
    HTTP client for Polymarket CLOB and Gamma APIs.

    Handles:
    - Market discovery (finding active 15-min markets)
    - Token ID resolution
    - Market info fetching
    """

    __slots__ = ('config', '_session', '_cache', '_cache_time')

    def __init__(self, config: GabagoolConfig):
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp package required. Install with: pip install aiohttp")

        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, datetime] = {}

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self) -> None:
        """Initialize HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_active_btc_market(self) -> Optional[MarketInfo]:
        """
        Find the currently active BTC 15-minute market.

        Returns:
            MarketInfo for the active market, or None
        """
        try:
            # Search for BTC binary markets
            markets = await self._search_markets("BTC")

            for market in markets:
                # Filter for 15-minute binary markets
                if self._is_valid_15min_market(market):
                    return self._parse_market_info(market)

            logger.warning("No active BTC 15-min market found")
            return None

        except Exception as e:
            logger.error(f"Error finding BTC market: {e}")
            return None

    async def get_active_sol_market(self) -> Optional[MarketInfo]:
        """
        Find the currently active SOL 15-minute market.

        Returns:
            MarketInfo for the active market, or None
        """
        try:
            markets = await self._search_markets("SOL")

            for market in markets:
                if self._is_valid_15min_market(market):
                    return self._parse_market_info(market)

            logger.warning("No active SOL 15-min market found")
            return None

        except Exception as e:
            logger.error(f"Error finding SOL market: {e}")
            return None

    async def _search_markets(self, keyword: str) -> List[Dict[str, Any]]:
        """Search for markets by keyword."""
        if not self._session:
            await self.connect()

        # Check cache
        cache_key = f"search_{keyword}"
        if cache_key in self._cache:
            cache_age = datetime.now() - self._cache_time.get(cache_key, datetime.min)
            if cache_age.total_seconds() < 30:  # 30 second cache
                return self._cache[cache_key]

        try:
            url = f"{self.config.polymarket_gamma_url}/markets"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 100
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Market search failed: {resp.status}")
                    return []

                data = await resp.json()
                markets = data if isinstance(data, list) else data.get("data", [])

                # Filter by keyword
                filtered = [
                    m for m in markets
                    if keyword.lower() in m.get("question", "").lower()
                ]

                # Update cache
                self._cache[cache_key] = filtered
                self._cache_time[cache_key] = datetime.now()

                return filtered

        except Exception as e:
            logger.error(f"Market search error: {e}")
            return []

    def _is_valid_15min_market(self, market: Dict[str, Any]) -> bool:
        """Check if market is a valid 15-minute binary market."""
        question = market.get("question", "").lower()

        # Must be a 15-minute market
        if "15" not in question and "fifteen" not in question:
            return False

        # Must be active
        if not market.get("active", False):
            return False

        # Check end time is in the future
        end_date_str = market.get("end_date_iso") or market.get("endDate", "")
        if end_date_str:
            try:
                end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_date.replace(tzinfo=None) < datetime.now():
                    return False
            except:
                pass

        # Must have tokens
        tokens = market.get("tokens", [])
        if len(tokens) < 2:
            return False

        return True

    def _parse_market_info(self, market: Dict[str, Any]) -> MarketInfo:
        """Parse market data into MarketInfo."""
        tokens = market.get("tokens", [])

        yes_token = ""
        no_token = ""

        for token in tokens:
            outcome = token.get("outcome", "").lower()
            token_id = token.get("token_id", "")

            if outcome in ("yes", "up"):
                yes_token = token_id
            elif outcome in ("no", "down"):
                no_token = token_id

        # Parse dates
        start_str = market.get("start_date_iso") or market.get("startDate", "")
        end_str = market.get("end_date_iso") or market.get("endDate", "")

        try:
            start_time = datetime.fromisoformat(start_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            start_time = datetime.now()

        try:
            end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except:
            end_time = datetime.now() + timedelta(minutes=15)

        return MarketInfo(
            market_id=market.get("id", ""),
            condition_id=market.get("condition_id", ""),
            question=market.get("question", ""),
            yes_token_id=yes_token,
            no_token_id=no_token,
            start_time=start_time,
            end_time=end_time,
            outcome=market.get("outcome")
        )

    async def get_market_by_id(self, market_id: str) -> Optional[MarketInfo]:
        """Fetch market info by ID."""
        if not self._session:
            await self.connect()

        try:
            url = f"{self.config.polymarket_gamma_url}/markets/{market_id}"

            async with self._session.get(url) as resp:
                if resp.status != 200:
                    return None

                market = await resp.json()
                return self._parse_market_info(market)

        except Exception as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        """Fetch current orderbook for a token."""
        if not self._session:
            await self.connect()

        try:
            url = f"{self.config.polymarket_api_url}/book"
            params = {"token_id": token_id}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return {"bids": [], "asks": []}

                return await resp.json()

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """Get last trade price for a token."""
        if not self._session:
            await self.connect()

        try:
            url = f"{self.config.polymarket_api_url}/last-trade-price"
            params = {"token_id": token_id}

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                return float(data.get("price", 0))

        except Exception as e:
            logger.error(f"Error fetching last price: {e}")
            return None
