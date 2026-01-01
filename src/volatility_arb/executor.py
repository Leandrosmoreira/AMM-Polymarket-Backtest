"""
Trade Executor for Volatility Arbitrage Bot

Handles trade execution with paper trading and live trading modes.
"""

import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from enum import Enum
from abc import ABC, abstractmethod

from .edge_detector import EdgeOpportunity, TradeSignal, MarketPrices
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    PAPER = "paper"  # Simulated trades
    LIVE = "live"  # Real trades on Polymarket


@dataclass
class TradeOrder:
    """A trade order."""
    order_id: str
    token_id: str
    side: str  # "buy" or "sell"
    price: float
    size: float  # In tokens
    size_usd: float  # In USD
    created_at: int
    signal: TradeSignal
    edge: EdgeOpportunity


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order_id: str
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    fill_usd: Optional[float] = None
    error: Optional[str] = None
    timestamp: int = 0

    @property
    def slippage_pct(self) -> float:
        """Calculate slippage from expected to actual fill."""
        if not self.fill_price:
            return 0.0
        # Would need expected price to calculate
        return 0.0


@dataclass
class Position:
    """An open position."""
    position_id: str
    token_id: str
    direction: str  # "up" or "down"
    entry_price: float
    size_tokens: float
    size_usd: float
    entry_time: int
    market_id: str
    market_expiry: int  # Timestamp when market settles
    edge_at_entry: float  # Edge % when we entered


class BaseExecutor(ABC):
    """Abstract base class for trade executors."""

    @abstractmethod
    async def execute_trade(
        self,
        edge: EdgeOpportunity,
        market_prices: MarketPrices,
        token_id: str,
        size_usd: float,
        market_id: str,
        market_expiry: int,
    ) -> TradeResult:
        """Execute a trade."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def close_position(self, position_id: str) -> TradeResult:
        """Close a position (sell tokens)."""
        pass


class PaperExecutor(BaseExecutor):
    """
    Paper trading executor for simulation and testing.

    Simulates trade execution with realistic fills.
    """

    def __init__(
        self,
        slippage_pct: float = 0.5,  # Simulated slippage
        fill_probability: float = 0.95,  # Probability of fill
        initial_balance: float = 1000.0,
    ):
        self.slippage_pct = slippage_pct
        self.fill_probability = fill_probability
        self.balance = initial_balance

        self._positions: Dict[str, Position] = {}
        self._trade_history: List[dict] = []

    async def execute_trade(
        self,
        edge: EdgeOpportunity,
        market_prices: MarketPrices,
        token_id: str,
        size_usd: float,
        market_id: str,
        market_expiry: int,
    ) -> TradeResult:
        """Execute a simulated trade."""

        order_id = str(uuid.uuid4())[:8]
        now = int(time.time() * 1000)

        # Check balance
        if size_usd > self.balance:
            return TradeResult(
                success=False,
                order_id=order_id,
                error="Insufficient balance",
                timestamp=now
            )

        # Simulate fill probability
        import random
        if random.random() > self.fill_probability:
            return TradeResult(
                success=False,
                order_id=order_id,
                error="Order not filled (simulated)",
                timestamp=now
            )

        # Calculate fill price with slippage
        if edge.signal == TradeSignal.BUY_UP:
            base_price = market_prices.up_price
            direction = "up"
        else:
            base_price = market_prices.down_price
            direction = "down"

        # Apply slippage (we pay slightly more)
        fill_price = base_price * (1 + self.slippage_pct / 100)
        fill_price = min(0.99, fill_price)  # Cap at 0.99

        # Calculate tokens received
        tokens = size_usd / fill_price

        # Create position
        position_id = str(uuid.uuid4())[:8]
        position = Position(
            position_id=position_id,
            token_id=token_id,
            direction=direction,
            entry_price=fill_price,
            size_tokens=tokens,
            size_usd=size_usd,
            entry_time=now,
            market_id=market_id,
            market_expiry=market_expiry,
            edge_at_entry=edge.edge_percent
        )

        self._positions[position_id] = position
        self.balance -= size_usd

        # Record trade
        self._trade_history.append({
            'type': 'entry',
            'position_id': position_id,
            'order_id': order_id,
            'direction': direction,
            'price': fill_price,
            'size_usd': size_usd,
            'tokens': tokens,
            'timestamp': now,
            'edge': edge.edge_percent
        })

        logger.info(
            f"[PAPER] Bought {tokens:.4f} {direction.upper()} tokens @ ${fill_price:.4f} "
            f"(${size_usd:.2f}, edge: {edge.edge_percent:.1f}%)"
        )

        return TradeResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=tokens,
            fill_usd=size_usd,
            timestamp=now
        )

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    async def close_position(self, position_id: str) -> TradeResult:
        """Close a position."""
        position = self._positions.get(position_id)
        if not position:
            return TradeResult(
                success=False,
                order_id="",
                error="Position not found",
                timestamp=int(time.time() * 1000)
            )

        now = int(time.time() * 1000)
        order_id = str(uuid.uuid4())[:8]

        # For paper trading, we'll assume we sell at current market
        # In reality, this would be the settlement price at expiry

        # Remove position
        del self._positions[position_id]

        # Record trade
        self._trade_history.append({
            'type': 'exit',
            'position_id': position_id,
            'order_id': order_id,
            'timestamp': now
        })

        return TradeResult(
            success=True,
            order_id=order_id,
            timestamp=now
        )

    def settle_position(self, position_id: str, won: bool) -> float:
        """
        Settle a position at market expiry.

        Args:
            position_id: Position to settle
            won: Whether the position won (token pays out $1)

        Returns:
            P&L in USD
        """
        position = self._positions.get(position_id)
        if not position:
            return 0.0

        if won:
            # Token pays out $1
            payout = position.size_tokens * 1.0
            pnl = payout - position.size_usd
        else:
            # Token pays $0
            pnl = -position.size_usd
            payout = 0.0

        self.balance += payout
        del self._positions[position_id]

        self._trade_history.append({
            'type': 'settlement',
            'position_id': position_id,
            'won': won,
            'payout': payout,
            'pnl': pnl,
            'timestamp': int(time.time() * 1000)
        })

        logger.info(
            f"[PAPER] Settled {position.direction.upper()} position: "
            f"{'WON' if won else 'LOST'} | P&L: ${pnl:+.2f}"
        )

        return pnl

    def get_trade_history(self) -> List[dict]:
        """Get all trade history."""
        return self._trade_history.copy()

    def get_stats(self) -> dict:
        """Get trading statistics."""
        entries = [t for t in self._trade_history if t['type'] == 'entry']
        settlements = [t for t in self._trade_history if t['type'] == 'settlement']

        wins = sum(1 for s in settlements if s.get('won', False))
        losses = len(settlements) - wins

        total_pnl = sum(s.get('pnl', 0) for s in settlements)

        return {
            'total_trades': len(entries),
            'settlements': len(settlements),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(settlements) if settlements else 0,
            'total_pnl': round(total_pnl, 2),
            'current_balance': round(self.balance, 2),
            'open_positions': len(self._positions)
        }


class LiveExecutor(BaseExecutor):
    """
    Live trading executor for Polymarket.

    Uses Polymarket CLOB API for order execution.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        funder: str,  # Wallet address
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.funder = funder

        self._positions: Dict[str, Position] = {}
        self._client = None

    async def _ensure_client(self):
        """Initialize Polymarket client if needed."""
        if self._client is None:
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                creds = ApiCreds(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    api_passphrase=self.passphrase,
                )

                self._client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=137,
                    key=self.api_key,
                    creds=creds,
                    funder=self.funder,
                )
            except ImportError:
                raise ImportError(
                    "py-clob-client required for live trading. "
                    "Install with: pip install py-clob-client"
                )

    async def execute_trade(
        self,
        edge: EdgeOpportunity,
        market_prices: MarketPrices,
        token_id: str,
        size_usd: float,
        market_id: str,
        market_expiry: int,
    ) -> TradeResult:
        """Execute a live trade on Polymarket."""

        await self._ensure_client()

        order_id = str(uuid.uuid4())[:8]
        now = int(time.time() * 1000)

        try:
            # Determine price and direction
            if edge.signal == TradeSignal.BUY_UP:
                price = market_prices.up_price
                direction = "up"
            else:
                price = market_prices.down_price
                direction = "down"

            # Calculate size in tokens
            tokens = size_usd / price

            # Create and submit order
            # This would use the actual CLOB client API
            # For now, just log the intended trade

            logger.info(
                f"[LIVE] Would execute: BUY {tokens:.4f} {direction.upper()} @ ${price:.4f}"
            )

            # TODO: Implement actual order submission
            # order = self._client.create_order(...)
            # result = self._client.post_order(order)

            return TradeResult(
                success=False,
                order_id=order_id,
                error="Live trading not yet implemented",
                timestamp=now
            )

        except Exception as e:
            logger.error(f"Live trade execution failed: {e}")
            return TradeResult(
                success=False,
                order_id=order_id,
                error=str(e),
                timestamp=now
            )

    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        await self._ensure_client()

        # TODO: Fetch positions from Polymarket API
        return []

    async def close_position(self, position_id: str) -> TradeResult:
        """Close a position."""
        await self._ensure_client()

        # TODO: Implement position closing
        return TradeResult(
            success=False,
            order_id="",
            error="Not implemented",
            timestamp=int(time.time() * 1000)
        )


def create_executor(
    mode: ExecutionMode,
    initial_balance: float = 1000.0,
    api_key: str = "",
    api_secret: str = "",
    passphrase: str = "",
    funder: str = "",
) -> BaseExecutor:
    """
    Factory function to create an executor.

    Args:
        mode: PAPER or LIVE
        initial_balance: Starting balance for paper trading
        api_key: Polymarket API key (for live)
        api_secret: Polymarket API secret (for live)
        passphrase: Polymarket passphrase (for live)
        funder: Wallet address (for live)

    Returns:
        Appropriate executor instance
    """
    if mode == ExecutionMode.PAPER:
        return PaperExecutor(initial_balance=initial_balance)
    elif mode == ExecutionMode.LIVE:
        if not all([api_key, api_secret, passphrase, funder]):
            raise ValueError("Live trading requires API credentials")
        return LiveExecutor(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            funder=funder,
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode}")
