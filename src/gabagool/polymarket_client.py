"""
Polymarket API Client for Gabagool Bot
Handles all API interactions with Polymarket CLOB and Gamma APIs
"""

import asyncio
import time
import hmac
import hashlib
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import json

import httpx

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    GTC = "GTC"  # Good Till Cancelled
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date


@dataclass
class OrderBook:
    """Order book data."""
    token_id: str
    bids: List[Dict[str, Any]]
    asks: List[Dict[str, Any]]
    timestamp: int

    @property
    def best_bid(self) -> Optional[float]:
        if not self.bids:
            return None
        return max(float(b['price']) for b in self.bids)

    @property
    def best_ask(self) -> Optional[float]:
        if not self.asks:
            return None
        return min(float(a['price']) for a in self.asks)

    @property
    def best_bid_size(self) -> Optional[float]:
        if not self.bids:
            return None
        best_price = self.best_bid
        for b in self.bids:
            if float(b['price']) == best_price:
                return float(b['size'])
        return None

    @property
    def best_ask_size(self) -> Optional[float]:
        if not self.asks:
            return None
        best_price = self.best_ask
        for a in self.asks:
            if float(a['price']) == best_price:
                return float(a['size'])
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


@dataclass
class Market:
    """Market data."""
    id: str
    slug: str
    question: str
    end_date: str
    active: bool
    closed: bool
    tokens: List[Dict[str, Any]]

    @property
    def up_token_id(self) -> Optional[str]:
        for token in self.tokens:
            outcome = token.get('outcome', '').lower()
            if 'up' in outcome or 'yes' in outcome:
                return token.get('token_id')
        return None

    @property
    def down_token_id(self) -> Optional[str]:
        for token in self.tokens:
            outcome = token.get('outcome', '').lower()
            if 'down' in outcome or 'no' in outcome:
                return token.get('token_id')
        return None


@dataclass
class Order:
    """Order data."""
    id: str
    market_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    filled: float
    status: str
    timestamp: int


class PolymarketClient:
    """
    Async client for Polymarket APIs.

    Handles:
    - CLOB API (order book, orders)
    - Gamma API (markets)
    """

    def __init__(
        self,
        clob_url: str = "https://clob.polymarket.com",
        gamma_url: str = "https://gamma-api.polymarket.com",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        private_key: Optional[str] = None,
    ):
        self.clob_url = clob_url.rstrip('/')
        self.gamma_url = gamma_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.private_key = private_key

        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                limits=httpx.Limits(max_connections=20),
            )
        return self

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication headers for CLOB API."""
        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            return {}

        timestamp = str(int(time.time() * 1000))
        message = timestamp + method.upper() + path + body

        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.b64encode(signature).decode('utf-8')

        return {
            "POLY-API-KEY": self.api_key,
            "POLY-SIGNATURE": signature_b64,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.api_passphrase,
        }

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        if self._client is None:
            await self.connect()

        headers = {"Content-Type": "application/json"}

        if auth:
            path = url.replace(self.clob_url, "")
            body = json.dumps(json_data) if json_data else ""
            headers.update(self._get_auth_headers(method, path, body))

        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=headers,
            )

            # Update rate limit info
            self._rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
            self._rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))

            if response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = self._rate_limit_reset - time.time()
                if wait_time > 0:
                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    return await self._request(method, url, params, json_data, auth)

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Request error: {e}")
            raise

    # === GAMMA API (Markets) ===

    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
    ) -> List[Market]:
        """Get list of markets."""
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
        }

        data = await self._request("GET", f"{self.gamma_url}/markets", params=params)

        markets = []
        for m in data:
            markets.append(Market(
                id=m.get('id') or m.get('condition_id', ''),
                slug=m.get('slug', ''),
                question=m.get('question', ''),
                end_date=m.get('end_date_iso', ''),
                active=m.get('active', False),
                closed=m.get('closed', False),
                tokens=m.get('tokens', []),
            ))

        return markets

    async def find_updown_markets(
        self,
        asset: str = "BTC",
        timeframe: str = "15min",
    ) -> List[Market]:
        """
        Find active Up/Down markets for an asset.

        Args:
            asset: BTC, ETH, etc.
            timeframe: 15min, 1h, etc.

        Returns:
            List of matching markets
        """
        all_markets = await self.get_markets(active=True, closed=False)

        matching = []
        asset_lower = asset.lower()
        timeframe_lower = timeframe.lower().replace("min", "").replace("m", "")

        for market in all_markets:
            question_lower = market.question.lower()
            slug_lower = market.slug.lower()

            # Check if it's an up/down market for the asset
            is_asset = asset_lower in question_lower or asset_lower in slug_lower
            is_updown = 'up' in question_lower and 'down' in question_lower
            is_timeframe = timeframe_lower in slug_lower or f"{timeframe_lower}min" in slug_lower

            if is_asset and is_updown:
                matching.append(market)

        return matching

    # === CLOB API (Order Book) ===

    async def get_order_book(self, token_id: str) -> OrderBook:
        """Get order book for a token."""
        data = await self._request(
            "GET",
            f"{self.clob_url}/book",
            params={"token_id": token_id}
        )

        return OrderBook(
            token_id=token_id,
            bids=data.get('bids', []),
            asks=data.get('asks', []),
            timestamp=int(time.time() * 1000),
        )

    async def get_spread(
        self,
        up_token_id: str,
        down_token_id: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get current spread for UP/DOWN pair.

        Returns:
            Tuple of (up_price, down_price, total)
        """
        try:
            # Get both order books in parallel
            up_book, down_book = await asyncio.gather(
                self.get_order_book(up_token_id),
                self.get_order_book(down_token_id),
            )

            up_price = up_book.best_ask
            down_price = down_book.best_ask

            if up_price is None or down_price is None:
                return None, None, None

            total = up_price + down_price

            return up_price, down_price, total

        except Exception as e:
            logger.error(f"Error getting spread: {e}")
            return None, None, None

    async def get_mid_prices(
        self,
        up_token_id: str,
        down_token_id: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get mid prices for UP/DOWN tokens."""
        try:
            up_book, down_book = await asyncio.gather(
                self.get_order_book(up_token_id),
                self.get_order_book(down_token_id),
            )

            up_mid = None
            down_mid = None

            if up_book.best_bid and up_book.best_ask:
                up_mid = (up_book.best_bid + up_book.best_ask) / 2

            if down_book.best_bid and down_book.best_ask:
                down_mid = (down_book.best_bid + down_book.best_ask) / 2

            return up_mid, down_mid

        except Exception as e:
            logger.error(f"Error getting mid prices: {e}")
            return None, None

    # === CLOB API (Orders) - Requires Authentication ===

    async def create_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
    ) -> Optional[Order]:
        """
        Create a new order.

        Args:
            token_id: Token to trade
            side: BUY or SELL
            price: Limit price
            size: Number of shares
            order_type: GTC, FOK, GTD

        Returns:
            Order object or None on failure
        """
        if not self.api_key:
            logger.error("API key required for order creation")
            return None

        payload = {
            "token_id": token_id,
            "side": side.value,
            "price": str(price),
            "size": str(size),
            "type": order_type.value,
        }

        try:
            data = await self._request(
                "POST",
                f"{self.clob_url}/order",
                json_data=payload,
                auth=True,
            )

            return Order(
                id=data.get('id', ''),
                market_id=data.get('market_id', ''),
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                filled=float(data.get('filled', 0)),
                status=data.get('status', 'unknown'),
                timestamp=int(time.time() * 1000),
            )

        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            await self._request(
                "DELETE",
                f"{self.clob_url}/order/{order_id}",
                auth=True,
            )
            return True
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False

    async def get_orders(self, market_id: Optional[str] = None) -> List[Order]:
        """Get open orders."""
        params = {}
        if market_id:
            params['market_id'] = market_id

        try:
            data = await self._request(
                "GET",
                f"{self.clob_url}/orders",
                params=params,
                auth=True,
            )

            orders = []
            for o in data:
                orders.append(Order(
                    id=o.get('id', ''),
                    market_id=o.get('market_id', ''),
                    token_id=o.get('token_id', ''),
                    side=OrderSide(o.get('side', 'BUY')),
                    price=float(o.get('price', 0)),
                    size=float(o.get('size', 0)),
                    filled=float(o.get('filled', 0)),
                    status=o.get('status', 'unknown'),
                    timestamp=int(o.get('timestamp', 0)),
                ))
            return orders

        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    # === Utility Methods ===

    async def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            # Try to get markets as a health check
            await self._request("GET", f"{self.clob_url}/health")
            return True
        except:
            return False

    @property
    def rate_limit_remaining(self) -> int:
        return self._rate_limit_remaining
