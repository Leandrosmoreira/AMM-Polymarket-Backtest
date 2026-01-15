"""
PolymarketWebSocket - WebSocket connection to Polymarket CLOB
Handles real-time order book and trade updates
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any

from core.types import BookSnapshot, OrderBookLevel, PaperTrade, TokenType, Side
from config.gabagool_config import GabagoolConfig

logger = logging.getLogger(__name__)


@dataclass
class WSMessage:
    """Parsed WebSocket message."""
    type: str
    token_id: str
    data: Dict[str, Any]
    timestamp: datetime


class PolymarketWebSocket:
    """
    WebSocket client for Polymarket CLOB.

    Handles:
    - Connection and reconnection
    - Subscription to market channels
    - Parsing of book/trade updates
    - Callbacks for data handlers
    """

    __slots__ = (
        'config',
        '_ws',
        '_running',
        '_subscribed_tokens',
        '_on_book_update',
        '_on_trade',
        '_reconnect_count',
        '_last_message_time'
    )

    def __init__(self, config: GabagoolConfig):
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required. Install with: pip install websockets")

        self.config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._subscribed_tokens: List[str] = []
        self._on_book_update: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._reconnect_count = 0
        self._last_message_time: Optional[datetime] = None

    async def connect(self) -> bool:
        """Connect to Polymarket WebSocket."""
        try:
            self._ws = await websockets.connect(
                self.config.polymarket_ws_url,
                ping_interval=self.config.ws_ping_interval_seconds,
                ping_timeout=10,
                close_timeout=5
            )
            self._running = True
            self._reconnect_count = 0
            logger.info(f"Connected to Polymarket WebSocket")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Polymarket WS: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("Disconnected from Polymarket WebSocket")

    async def subscribe(self, token_ids: List[str]) -> bool:
        """
        Subscribe to market channels.

        Args:
            token_ids: List of token IDs to subscribe to
        """
        if not self._ws:
            logger.error("Not connected")
            return False

        try:
            # Polymarket subscription format
            for token_id in token_ids:
                # Subscribe to book updates
                book_msg = {
                    "type": "subscribe",
                    "channel": "book",
                    "market": token_id
                }
                await self._ws.send(json.dumps(book_msg))

                # Subscribe to trades
                trade_msg = {
                    "type": "subscribe",
                    "channel": "trades",
                    "market": token_id
                }
                await self._ws.send(json.dumps(trade_msg))

            self._subscribed_tokens = token_ids
            logger.info(f"Subscribed to {len(token_ids)} tokens")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False

    async def unsubscribe(self, token_ids: Optional[List[str]] = None) -> None:
        """Unsubscribe from market channels."""
        if not self._ws:
            return

        tokens = token_ids or self._subscribed_tokens

        try:
            for token_id in tokens:
                msg = {
                    "type": "unsubscribe",
                    "channel": "book",
                    "market": token_id
                }
                await self._ws.send(json.dumps(msg))

            self._subscribed_tokens = [
                t for t in self._subscribed_tokens
                if t not in tokens
            ]

        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")

    async def receive_loop(self) -> None:
        """Main receive loop for WebSocket messages."""
        while self._running:
            try:
                if not self._ws:
                    await self._reconnect()
                    continue

                msg = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=30.0
                )

                self._last_message_time = datetime.now()
                await self._process_message(msg)

            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                continue

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._reconnect()

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await asyncio.sleep(1)

    async def _reconnect(self) -> None:
        """Attempt to reconnect to WebSocket."""
        self._reconnect_count += 1
        delay = min(
            self.config.ws_reconnect_delay_seconds * self._reconnect_count,
            60.0
        )

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

        if await self.connect():
            # Re-subscribe to previous tokens
            if self._subscribed_tokens:
                await self.subscribe(self._subscribed_tokens)

    async def _process_message(self, raw_msg: str) -> None:
        """Process incoming WebSocket message."""
        try:
            data = json.loads(raw_msg)
            msg_type = data.get("type") or data.get("event_type", "")

            if msg_type in ("book", "book_update"):
                await self._handle_book_update(data)

            elif msg_type in ("trade", "last_trade_price"):
                await self._handle_trade(data)

            elif msg_type == "subscribed":
                logger.debug(f"Subscribed: {data}")

            elif msg_type == "error":
                logger.error(f"WebSocket error: {data}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {raw_msg[:100]}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_book_update(self, data: Dict[str, Any]) -> None:
        """Handle order book update."""
        if not self._on_book_update:
            return

        try:
            token_id = data.get("market") or data.get("asset_id", "")

            # Parse bids
            bids = []
            for bid in data.get("bids", []):
                if isinstance(bid, dict):
                    bids.append(OrderBookLevel(
                        price=float(bid.get("price", 0)),
                        size=float(bid.get("size", 0))
                    ))
                elif isinstance(bid, list) and len(bid) >= 2:
                    bids.append(OrderBookLevel(
                        price=float(bid[0]),
                        size=float(bid[1])
                    ))

            # Parse asks
            asks = []
            for ask in data.get("asks", []):
                if isinstance(ask, dict):
                    asks.append(OrderBookLevel(
                        price=float(ask.get("price", 0)),
                        size=float(ask.get("size", 0))
                    ))
                elif isinstance(ask, list) and len(ask) >= 2:
                    asks.append(OrderBookLevel(
                        price=float(ask[0]),
                        size=float(ask[1])
                    ))

            # Sort bids descending, asks ascending
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            snapshot = BookSnapshot(
                token_id=token_id,
                timestamp=datetime.now(),
                bids=bids,
                asks=asks
            )

            await self._on_book_update(snapshot)

        except Exception as e:
            logger.error(f"Error handling book update: {e}")

    async def _handle_trade(self, data: Dict[str, Any]) -> None:
        """Handle trade update."""
        if not self._on_trade:
            return

        try:
            token_id = data.get("market") or data.get("asset_id", "")
            price = float(data.get("price", 0))
            size = float(data.get("size", 0))
            side_str = data.get("side", "BUY").upper()

            trade = PaperTrade(
                market_id=token_id,
                token_type=TokenType.YES,  # Will be determined by caller
                side=Side.BUY if side_str == "BUY" else Side.SELL,
                price=price,
                size=size,
                timestamp=datetime.now()
            )

            await self._on_trade(trade)

        except Exception as e:
            logger.error(f"Error handling trade: {e}")

    def set_book_callback(self, callback: Callable) -> None:
        """Set callback for book updates."""
        self._on_book_update = callback

    def set_trade_callback(self, callback: Callable) -> None:
        """Set callback for trade updates."""
        self._on_trade = callback

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def subscribed_tokens(self) -> List[str]:
        return self._subscribed_tokens.copy()
