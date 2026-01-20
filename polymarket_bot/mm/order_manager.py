"""
Order Manager - Gerencia ordens ativas e pré-assinadas para low-latency trading.

Features:
- Pool de ordens pré-assinadas prontas para envio
- Tracking de ordens ativas
- Cancelamento rápido em batch
- Rate limiting
"""
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Representa uma ordem."""
    id: str
    market_id: str
    token: str           # "YES" or "NO"
    side: str            # "BUY" or "SELL"
    price: float
    size: float
    order_type: str      # "GTC", "FOK", "FAK"
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    created_at: float = field(default_factory=time.time)
    submitted_at: Optional[float] = None
    signed_data: Optional[dict] = None  # Ordem pré-assinada


@dataclass
class PreSignedOrder:
    """Ordem pré-assinada pronta para envio rápido."""
    token: str
    side: str
    price: float
    size: float
    signed_data: dict
    created_at: float
    expires_at: float


class OrderManager:
    """
    Gerencia ordens com foco em baixa latência.

    Usage:
        manager = OrderManager(client)

        # Pré-assinar ordens
        await manager.pre_sign_orders([
            {"token": "YES", "side": "BUY", "price": 0.48, "size": 10},
            {"token": "YES", "side": "SELL", "price": 0.52, "size": 10},
        ])

        # Enviar ordem pré-assinada (rápido!)
        order_id = await manager.submit_pre_signed("YES", "BUY", 0.48)

        # Cancelar todas
        await manager.cancel_all()
    """

    def __init__(
        self,
        client,  # ClobClient
        max_active_orders: int = 20,
        pre_sign_ttl: float = 60.0,  # Segundos até expirar
        rate_limit_per_sec: int = 10,
    ):
        self.client = client
        self.max_active_orders = max_active_orders
        self.pre_sign_ttl = pre_sign_ttl
        self.rate_limit_per_sec = rate_limit_per_sec

        # Ordens ativas: {order_id: Order}
        self.active_orders: Dict[str, Order] = {}

        # Pool de ordens pré-assinadas: {(token, side, price): PreSignedOrder}
        self.pre_signed_pool: Dict[tuple, PreSignedOrder] = {}

        # Rate limiting
        self.request_times: deque = deque(maxlen=rate_limit_per_sec)

        # Callbacks
        self.on_fill: Optional[Callable] = None
        self.on_cancel: Optional[Callable] = None

        # Stats
        self.orders_submitted = 0
        self.orders_filled = 0
        self.orders_cancelled = 0

    async def _rate_limit(self):
        """Aplica rate limiting."""
        now = time.time()

        # Remover requests antigos
        while self.request_times and (now - self.request_times[0]) > 1.0:
            self.request_times.popleft()

        # Se atingiu limite, esperar
        if len(self.request_times) >= self.rate_limit_per_sec:
            wait_time = 1.0 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.request_times.append(time.time())

    async def pre_sign_orders(self, orders: List[dict]) -> int:
        """
        Pré-assina um batch de ordens.

        Args:
            orders: Lista de dicts com token, side, price, size

        Returns:
            Número de ordens pré-assinadas
        """
        count = 0
        now = time.time()

        for order_params in orders:
            try:
                # Criar ordem assinada
                signed = await self._sign_order(
                    token=order_params["token"],
                    side=order_params["side"],
                    price=order_params["price"],
                    size=order_params["size"],
                )

                if signed:
                    key = (order_params["token"], order_params["side"], order_params["price"])
                    self.pre_signed_pool[key] = PreSignedOrder(
                        token=order_params["token"],
                        side=order_params["side"],
                        price=order_params["price"],
                        size=order_params["size"],
                        signed_data=signed,
                        created_at=now,
                        expires_at=now + self.pre_sign_ttl,
                    )
                    count += 1

            except Exception as e:
                logger.warning(f"Failed to pre-sign order: {e}")

        logger.info(f"Pre-signed {count} orders")
        return count

    async def _sign_order(self, token: str, side: str, price: float, size: float) -> Optional[dict]:
        """Assina uma ordem (sem enviar)."""
        try:
            # Usar o client para criar ordem assinada
            # Isso varia dependendo do client (ClobClient)
            from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions
            from py_clob_client.order_builder.constants import BUY, SELL

            order_args = OrderArgs(
                token_id=token,  # Deveria ser o token_id real
                price=price,
                size=size,
                side=BUY if side == "BUY" else SELL,
            )

            options = PartialCreateOrderOptions(neg_risk=True)
            signed = self.client.create_order(order_args, options)

            return signed

        except Exception as e:
            logger.error(f"Error signing order: {e}")
            return None

    async def submit_pre_signed(self, token: str, side: str, price: float) -> Optional[str]:
        """
        Envia uma ordem pré-assinada (muito rápido!).

        Returns:
            Order ID ou None se falhar
        """
        key = (token, side, price)

        if key not in self.pre_signed_pool:
            logger.warning(f"No pre-signed order for {key}")
            return None

        pre_signed = self.pre_signed_pool[key]

        # Verificar expiração
        if time.time() > pre_signed.expires_at:
            logger.warning(f"Pre-signed order expired: {key}")
            del self.pre_signed_pool[key]
            return None

        await self._rate_limit()

        try:
            # Enviar ordem
            from py_clob_client.clob_types import OrderType
            result = self.client.post_order(pre_signed.signed_data, OrderType.GTC)

            order_id = result.get("orderID") or result.get("order_id")

            if order_id:
                # Remover do pool
                del self.pre_signed_pool[key]

                # Adicionar às ativas
                self.active_orders[order_id] = Order(
                    id=order_id,
                    market_id="",  # Preencher depois
                    token=token,
                    side=side,
                    price=price,
                    size=pre_signed.size,
                    order_type="GTC",
                    status=OrderStatus.SUBMITTED,
                    submitted_at=time.time(),
                )

                self.orders_submitted += 1
                logger.debug(f"Submitted pre-signed order: {order_id}")

                return order_id

        except Exception as e:
            logger.error(f"Error submitting pre-signed order: {e}")

        return None

    async def submit_order(
        self,
        market_id: str,
        token: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> Optional[str]:
        """
        Envia uma ordem nova (mais lento que pre-signed).

        Returns:
            Order ID ou None se falhar
        """
        if len(self.active_orders) >= self.max_active_orders:
            logger.warning("Max active orders reached")
            return None

        await self._rate_limit()

        try:
            signed = await self._sign_order(token, side, price, size)
            if not signed:
                return None

            from py_clob_client.clob_types import OrderType
            ot = getattr(OrderType, order_type, OrderType.GTC)
            result = self.client.post_order(signed, ot)

            order_id = result.get("orderID") or result.get("order_id")

            if order_id:
                self.active_orders[order_id] = Order(
                    id=order_id,
                    market_id=market_id,
                    token=token,
                    side=side,
                    price=price,
                    size=size,
                    order_type=order_type,
                    status=OrderStatus.SUBMITTED,
                    submitted_at=time.time(),
                )
                self.orders_submitted += 1
                return order_id

        except Exception as e:
            logger.error(f"Error submitting order: {e}")

        return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancela uma ordem."""
        if order_id not in self.active_orders:
            return False

        await self._rate_limit()

        try:
            self.client.cancel_orders([order_id])
            self.active_orders[order_id].status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            self.orders_cancelled += 1

            if self.on_cancel:
                self.on_cancel(order_id)

            return True

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def cancel_all(self) -> int:
        """
        Cancela todas as ordens ativas.

        Returns:
            Número de ordens canceladas
        """
        if not self.active_orders:
            return 0

        order_ids = list(self.active_orders.keys())

        await self._rate_limit()

        try:
            self.client.cancel_orders(order_ids)
            count = len(order_ids)

            for oid in order_ids:
                self.active_orders[oid].status = OrderStatus.CANCELLED
                if self.on_cancel:
                    self.on_cancel(oid)

            self.active_orders.clear()
            self.orders_cancelled += count

            logger.info(f"Cancelled {count} orders")
            return count

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0

    def update_order_status(self, order_id: str, status: OrderStatus, filled_size: float = 0):
        """Atualiza status de uma ordem (chamado por WebSocket callback)."""
        if order_id not in self.active_orders:
            return

        order = self.active_orders[order_id]
        order.status = status
        order.filled_size = filled_size

        if status == OrderStatus.FILLED:
            self.orders_filled += 1
            if self.on_fill:
                self.on_fill(order)
            del self.active_orders[order_id]

        elif status == OrderStatus.CANCELLED:
            self.orders_cancelled += 1
            del self.active_orders[order_id]

    def get_active_orders(self, market_id: Optional[str] = None) -> List[Order]:
        """Retorna ordens ativas, opcionalmente filtradas por mercado."""
        orders = list(self.active_orders.values())
        if market_id:
            orders = [o for o in orders if o.market_id == market_id]
        return orders

    def cleanup_expired_pre_signed(self):
        """Remove ordens pré-assinadas expiradas."""
        now = time.time()
        expired = [k for k, v in self.pre_signed_pool.items() if now > v.expires_at]
        for k in expired:
            del self.pre_signed_pool[k]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired pre-signed orders")

    def get_stats(self) -> dict:
        """Retorna estatísticas."""
        return {
            "active_orders": len(self.active_orders),
            "pre_signed_pool": len(self.pre_signed_pool),
            "orders_submitted": self.orders_submitted,
            "orders_filled": self.orders_filled,
            "orders_cancelled": self.orders_cancelled,
            "fill_rate": self.orders_filled / max(1, self.orders_submitted) * 100,
        }
