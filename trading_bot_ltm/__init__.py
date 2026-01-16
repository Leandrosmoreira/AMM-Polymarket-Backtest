"""
Polymarket Trading Bot with LTM Integration

Modules:
- simple_arb_bot: Main arbitrage bot
- ltm_adapter: LTM integration adapter
- ltm: Liquidity Time Model components
"""

from .ltm_adapter import LTMAdapter, LTMTradeDecision

__all__ = [
    'LTMAdapter',
    'LTMTradeDecision',
]
