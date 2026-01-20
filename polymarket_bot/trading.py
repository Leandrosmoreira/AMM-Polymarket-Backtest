"""
Trading module for Polymarket Bot.

Uses the working authentication pattern from exemplo_polymarket
combined with trading functionality from trading_bot_ltm.
"""

import logging
import time
import uuid
from typing import Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    BalanceAllowanceParams,
    AssetType,
    OrderArgs,
    OrderType,
    PostOrdersArgs,
    PartialCreateOrderOptions,
)
from py_clob_client.order_builder.constants import BUY, SELL

from .config import Settings
from .auth import get_client as auth_get_client, AuthConfig

logger = logging.getLogger(__name__)


class MockClobClient:
    """Mock client for paper trading (dry_run mode)."""

    def __init__(self, sim_balance: float = 1000.0):
        self.sim_balance = sim_balance
        self._orders = {}
        self._positions = {}
        logger.info("Using MockClobClient for paper trading (DRY_RUN=true)")
        logger.info(f"   Simulated balance: ${sim_balance:.2f}")

    def get_address(self):
        return "0xPAPER_TRADING_WALLET"

    def get_balance_allowance(self, params):
        return {"balance": str(int(self.sim_balance * 1_000_000))}

    def create_order(self, order_args, options=None):
        return {
            "token_id": order_args.token_id,
            "price": order_args.price,
            "size": order_args.size,
            "side": order_args.side,
            "mock": True
        }

    def post_order(self, signed_order, order_type=None):
        order_id = f"PAPER_{uuid.uuid4().hex[:8]}"
        self._orders[order_id] = {
            "orderID": order_id,
            "status": "filled",
            "filled_size": signed_order.get("size", 0),
            "original_size": signed_order.get("size", 0),
            **signed_order
        }
        return self._orders[order_id]

    def post_orders(self, orders_args):
        return [self.post_order(o.order, o.orderType) for o in orders_args]

    def get_order(self, order_id):
        return self._orders.get(order_id, {"status": "not_found"})

    def cancel_orders(self, order_ids):
        for oid in order_ids:
            if oid in self._orders:
                self._orders[oid]["status"] = "cancelled"
        return {"cancelled": order_ids}

    def get_positions(self):
        return list(self._positions.values())

    def get_order_book(self, token_id: str = None):
        """Generate simulated order book."""
        import random

        current_time = int(time.time())
        seed = current_time // 5
        rng = random.Random(seed)

        opportunity = (current_time % 10) < 5

        if opportunity:
            base_price = 0.475 + (rng.random() * 0.01)
        else:
            base_price = 0.505 + (rng.random() * 0.015)

        spread = 0.01 + (rng.random() * 0.005)

        class MockLevel:
            def __init__(self, price, size):
                self.price = str(price)
                self.size = str(size)

        class MockOrderBook:
            def __init__(self, base, spread, rng):
                self.bids = [
                    MockLevel(round(base - spread/2 - i*0.005, 4), rng.randint(50, 200))
                    for i in range(5)
                ]
                self.asks = [
                    MockLevel(round(base + spread/2 + i*0.005, 4), rng.randint(50, 200))
                    for i in range(5)
                ]

        return MockOrderBook(base_price, spread, rng)


_cached_client = None
_is_mock_client = False


def get_client(settings: Settings):
    """
    Get authenticated Polymarket client.

    Uses the working auth pattern from exemplo_polymarket.
    """
    global _cached_client, _is_mock_client

    if _cached_client is not None:
        return _cached_client

    # Use mock client for paper trading
    if settings.dry_run:
        _cached_client = MockClobClient(sim_balance=settings.sim_balance)
        _is_mock_client = True
        return _cached_client

    # Create auth config from settings
    auth_config = AuthConfig.__new__(AuthConfig)
    auth_config.private_key = settings.private_key
    auth_config.funder_address = settings.funder_address or settings.funder
    auth_config.wallet_address = settings.wallet_address
    auth_config.signature_type = settings.signature_type
    auth_config.token_id = settings.yes_token_id

    # Use auth module to get client
    _cached_client = auth_get_client(auth_config)
    _is_mock_client = False

    return _cached_client


def get_balance(settings: Settings) -> float:
    """Get USDC balance from Polymarket account."""
    try:
        client = get_client(settings)
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=settings.signature_type
        )
        result = client.get_balance_allowance(params)

        if isinstance(result, dict):
            balance_raw = result.get("balance", "0")
            balance_wei = float(balance_raw)
            balance_usdc = balance_wei / 1_000_000
            return balance_usdc

        logger.warning(f"Unexpected response when getting balance: {result}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return 0.0


def place_order(settings: Settings, *, side: str, token_id: str, price: float, size: float, tif: str = "GTC") -> dict:
    """Place a single order."""
    if price <= 0:
        raise ValueError("price must be > 0")
    if size <= 0:
        raise ValueError("size must be > 0")
    if not token_id:
        raise ValueError("token_id is required")

    side_up = side.upper()
    if side_up not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    client = get_client(settings)

    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=BUY if side_up == "BUY" else SELL
        )

        options = PartialCreateOrderOptions(neg_risk=True)
        signed_order = client.create_order(order_args, options)

        tif_up = (tif or "GTC").upper()
        order_type = getattr(OrderType, tif_up, OrderType.GTC)
        return client.post_order(signed_order, order_type)
    except Exception as exc:
        raise RuntimeError(f"place_order failed: {exc}") from exc


def place_orders_fast(settings: Settings, orders: list[dict], *, order_type: str = "GTC") -> list[dict]:
    """
    Place multiple orders as fast as possible.

    Pre-signs all orders first, then submits them together.
    """
    client = get_client(settings)

    tif_up = (order_type or "GTC").upper()
    ot = getattr(OrderType, tif_up, OrderType.GTC)

    # Pre-sign all orders
    options = PartialCreateOrderOptions(neg_risk=True)
    signed_orders = []
    for order_params in orders:
        side_up = order_params["side"].upper()
        order_args = OrderArgs(
            token_id=order_params["token_id"],
            price=order_params["price"],
            size=order_params["size"],
            side=BUY if side_up == "BUY" else SELL,
        )
        signed_order = client.create_order(order_args, options)
        signed_orders.append(signed_order)

    # Post all orders
    try:
        args = [PostOrdersArgs(order=o, orderType=ot) for o in signed_orders]
        result = client.post_orders(args)
        if isinstance(result, list):
            return result
        return [result]
    except Exception:
        # Fallback to sequential posting
        results = []
        for signed_order in signed_orders:
            try:
                results.append(client.post_order(signed_order, ot))
            except Exception as exc:
                results.append({"error": str(exc)})
        return results


def extract_order_id(result: dict) -> Optional[str]:
    """Extract order id from API response."""
    if not isinstance(result, dict):
        return None
    for key in ("orderID", "orderId", "order_id", "id"):
        val = result.get(key)
        if val:
            return str(val)
    for key in ("order", "data", "result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            oid = extract_order_id(nested)
            if oid:
                return oid
    return None


def get_order(settings: Settings, order_id: str) -> dict:
    """Get order status by ID."""
    client = get_client(settings)
    return client.get_order(order_id)


def cancel_orders(settings: Settings, order_ids: list[str]) -> Optional[dict]:
    """Cancel multiple orders."""
    if not order_ids:
        return None
    client = get_client(settings)
    return client.cancel_orders(order_ids)


def _coerce_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def summarize_order_state(order_data: dict, *, requested_size: Optional[float] = None) -> dict:
    """Normalize order payload into summary."""
    if not isinstance(order_data, dict):
        return {"status": None, "filled_size": None, "requested_size": requested_size, "raw": order_data}

    status = order_data.get("status") or order_data.get("state") or order_data.get("order_status")
    status_str = str(status).lower() if status is not None else None

    filled_size = None
    for key in ("filled_size", "filledSize", "size_filled", "sizeFilled", "matched_size", "matchedSize"):
        if key in order_data:
            filled_size = _coerce_float(order_data.get(key))
            break

    remaining_size = None
    for key in ("remaining_size", "remainingSize", "size_remaining", "sizeRemaining"):
        if key in order_data:
            remaining_size = _coerce_float(order_data.get(key))
            break

    original_size = None
    for key in ("original_size", "originalSize", "size", "order_size", "orderSize"):
        if key in order_data:
            original_size = _coerce_float(order_data.get(key))
            break

    if filled_size is None and remaining_size is not None and original_size is not None:
        filled_size = max(0.0, original_size - remaining_size)

    return {
        "status": status_str,
        "filled_size": filled_size,
        "remaining_size": remaining_size,
        "original_size": original_size,
        "requested_size": requested_size,
        "raw": order_data,
    }


def wait_for_terminal_order(
    settings: Settings,
    order_id: str,
    *,
    requested_size: Optional[float] = None,
    timeout_seconds: float = 3.0,
    poll_interval_seconds: float = 0.25,
) -> dict:
    """Poll order state until terminal or timeout."""
    terminal_statuses = {"filled", "canceled", "cancelled", "rejected", "expired"}
    start = time.monotonic()
    last_summary = None

    while (time.monotonic() - start) < timeout_seconds:
        try:
            od = get_order(settings, order_id)
            last_summary = summarize_order_state(od, requested_size=requested_size)
        except Exception as exc:
            last_summary = {"status": "error", "error": str(exc), "filled_size": None, "requested_size": requested_size}

        status = (last_summary.get("status") or "").lower() if isinstance(last_summary, dict) else ""
        filled = last_summary.get("filled_size") if isinstance(last_summary, dict) else None

        if requested_size is not None and filled is not None and filled + 1e-9 >= float(requested_size):
            last_summary["terminal"] = True
            last_summary["filled"] = True
            return last_summary

        if status in terminal_statuses:
            last_summary["terminal"] = True
            last_summary["filled"] = (status == "filled")
            return last_summary

        time.sleep(poll_interval_seconds)

    if last_summary is None:
        last_summary = {"status": None, "filled_size": None, "requested_size": requested_size}
    last_summary["terminal"] = False
    last_summary.setdefault("filled", False)
    return last_summary


def get_positions(settings: Settings, token_ids: list[str] = None) -> dict:
    """Get current positions for user."""
    try:
        client = get_client(settings)
        positions = client.get_positions()

        result = {}
        for pos in positions:
            token_id = pos.get("asset", {}).get("token_id") or pos.get("token_id")
            if token_id:
                if token_ids is None or token_id in token_ids:
                    size = float(pos.get("size", 0))
                    avg_price = float(pos.get("avg_price", 0))
                    result[token_id] = {
                        "size": size,
                        "avg_price": avg_price,
                        "raw": pos
                    }

        return result
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {}


def reset_client():
    """Reset cached client."""
    global _cached_client, _is_mock_client
    _cached_client = None
    _is_mock_client = False
