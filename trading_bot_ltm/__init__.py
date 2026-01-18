"""
Polymarket Trading Bot with LTM Integration

Modules:
- simple_arb_bot: Main arbitrage bot
- ltm_adapter: LTM integration adapter
- ltm: Liquidity Time Model components
"""

# Lazy import to avoid requiring pandas at module level
# LTM features require pandas, but bot can run without it
try:
    from .ltm_adapter import LTMAdapter, LTMTradeDecision
    __all__ = [
        'LTMAdapter',
        'LTMTradeDecision',
    ]
except ImportError as e:
    # If LTM dependencies are not available, bot can still run without LTM
    import warnings
    warnings.warn(f"LTM features not available: {e}. Bot can run without LTM.")
    __all__ = []
