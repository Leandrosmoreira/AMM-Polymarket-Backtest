"""
BinanceWebSocket - WebSocket connection to Binance for price feed
Provides real-time BTC/SOL prices for edge detection
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any

from core.buffers import PriceBuffer
from config.gabagool_config import GabagoolConfig

logger = logging.getLogger(__name__)


class BinanceWebSocket:
    """
    WebSocket client for Binance price feed.

    Streams real-time trade prices for BTC or SOL
    to detect price divergences with Polymarket.
    """

    __slots__ = (
        'config',
        '_ws',
        '_running',
        '_symbol',
        '_on_price',
        '_price_buffer',
        '_last_price',
        '_reconnect_count'
    )

    def __init__(self, config: GabagoolConfig):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._symbol = config.binance_symbol.lower()
        self._on_price: Optional[Callable] = None
        self._price_buffer = PriceBuffer(size=300)
        self._last_price: Optional[float] = None
        self._reconnect_count = 0

    async def connect(self) -> bool:
        """Connect to Binance WebSocket."""
        try:
            # Binance trade stream URL
            url = f"{self.config.binance_ws_url}/{self._symbol}@trade"

            self._ws = await websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self._running = True
            self._reconnect_count = 0
            logger.info(f"Connected to Binance WebSocket ({self._symbol})")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Binance WS: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from Binance WebSocket")

    async def receive_loop(self) -> None:
        """Main receive loop for price updates."""
        while self._running:
            try:
                if not self._ws:
                    await self._reconnect()
                    continue

                msg = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=30.0
                )

                await self._process_message(msg)

            except asyncio.TimeoutError:
                logger.warning("Binance WebSocket timeout")
                continue

            except websockets.ConnectionClosed:
                logger.warning("Binance WebSocket connection closed")
                await self._reconnect()

            except Exception as e:
                logger.error(f"Error in Binance receive loop: {e}")
                await asyncio.sleep(1)

    async def _reconnect(self) -> None:
        """Attempt to reconnect."""
        self._reconnect_count += 1
        delay = min(5 * self._reconnect_count, 60)

        logger.info(f"Reconnecting to Binance in {delay}s")
        await asyncio.sleep(delay)
        await self.connect()

    async def _process_message(self, raw_msg: str) -> None:
        """Process incoming trade message."""
        try:
            data = json.loads(raw_msg)

            # Binance trade format
            # {"e":"trade","E":123,"s":"BTCUSDT","t":123,"p":"50000.00","q":"0.1",...}
            if data.get("e") == "trade":
                price = float(data.get("p", 0))

                if price > 0:
                    self._last_price = price
                    self._price_buffer.add(price)

                    if self._on_price:
                        await self._on_price(price)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from Binance")
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")

    def set_price_callback(self, callback: Callable) -> None:
        """Set callback for price updates."""
        self._on_price = callback

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def last_price(self) -> Optional[float]:
        return self._last_price

    @property
    def price_buffer(self) -> PriceBuffer:
        return self._price_buffer

    def get_recent_return(self, periods: int = 10) -> float:
        """Get recent price return."""
        prices = self._price_buffer.get_prices(periods + 1)
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]

    def get_volatility(self, periods: int = 20) -> float:
        """Get recent price volatility."""
        return self._price_buffer.get_volatility(periods)


class BinancePriceAgent:
    """
    Simplified Binance price agent using REST API fallback.
    For environments without websocket support.
    """

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._price_buffer = PriceBuffer(size=300)
        self._last_price: Optional[float] = None
        self._running = False

    async def start_polling(self, interval: float = 1.0) -> None:
        """Start polling prices from Binance REST API."""
        import aiohttp

        self._running = True
        symbol = self.config.binance_symbol.upper()
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"

        async with aiohttp.ClientSession() as session:
            while self._running:
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price = float(data.get("price", 0))

                            if price > 0:
                                self._last_price = price
                                self._price_buffer.add(price)

                except Exception as e:
                    logger.error(f"Error polling Binance: {e}")

                await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop polling."""
        self._running = False

    @property
    def last_price(self) -> Optional[float]:
        return self._last_price

    @property
    def price_buffer(self) -> PriceBuffer:
        return self._price_buffer
