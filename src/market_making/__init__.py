"""
Market Making / Scalp Strategy for Polymarket

Two strategies:
1. Scalp com Hedge (Market Making) - Places limit orders on both sides
2. Proporção Ajustada (Edge-Based) - Proportional buying based on model edge
"""

from .scalp_backtest import (
    ScalpBacktest,
    ScalpResult,
    ScalpTrade,
    run_scalp_backtest,
)
from .edge_proportional_backtest import (
    EdgeProportionalBacktest,
    EdgeProportionalResult,
    EdgeProportionalTrade,
    run_edge_proportional_backtest,
)

__all__ = [
    # Scalp/Market Making
    'ScalpBacktest',
    'ScalpResult',
    'ScalpTrade',
    'run_scalp_backtest',

    # Edge-Based Proportional
    'EdgeProportionalBacktest',
    'EdgeProportionalResult',
    'EdgeProportionalTrade',
    'run_edge_proportional_backtest',
]
