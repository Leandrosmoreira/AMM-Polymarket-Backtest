"""
Polymarket Copy Trading Module
Monitor a wallet and replicate trades automatically
"""

from .blockchain_monitor import BlockchainMonitor, TransactionInfo
from .polymarket_decoder import (
    PolymarketDecoder,
    PolymarketTrade,
    TradeSide,
    TradeOutcome,
    MarketRegistry
)
from .copy_executor import (
    CopyTradeExecutor,
    DryRunExecutor,
    CopyTradeResult,
    PortfolioState
)

__all__ = [
    'BlockchainMonitor',
    'TransactionInfo',
    'PolymarketDecoder',
    'PolymarketTrade',
    'TradeSide',
    'TradeOutcome',
    'MarketRegistry',
    'CopyTradeExecutor',
    'DryRunExecutor',
    'CopyTradeResult',
    'PortfolioState',
]
