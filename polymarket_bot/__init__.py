"""
Polymarket Bot - Merged project combining:
- Working authentication from exemplo_polymarket
- Full bot functionality from trading_bot_ltm

Usage:
    python -m polymarket_bot              # Run arbitrage bot
    python -m polymarket_bot.test_auth    # Test authentication
"""

from .config import Settings, load_settings
from .auth import get_client, get_balance, test_auth, AuthConfig
from .trading import (
    place_order,
    place_orders_fast,
    get_positions,
    cancel_orders,
)

__version__ = "1.0.0"
__all__ = [
    "Settings",
    "load_settings",
    "get_client",
    "get_balance",
    "test_auth",
    "AuthConfig",
    "place_order",
    "place_orders_fast",
    "get_positions",
    "cancel_orders",
]
