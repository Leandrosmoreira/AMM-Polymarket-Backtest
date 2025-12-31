"""
Position Manager for Gabagool Bot
Tracks positions and manages balancing between UP/DOWN
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .config import GabagoolConfig

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class Trade:
    """Record of a single trade."""
    id: str
    market_id: str
    side: PositionSide
    shares: float
    price: float
    cost: float
    timestamp: int
    order_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'market_id': self.market_id,
            'side': self.side.value,
            'shares': self.shares,
            'price': self.price,
            'cost': self.cost,
            'timestamp': self.timestamp,
            'order_id': self.order_id,
        }


@dataclass
class MarketPosition:
    """Position in a single market."""
    market_id: str
    market_slug: str
    up_token_id: str
    down_token_id: str

    shares_up: float = 0.0
    shares_down: float = 0.0
    cost_up: float = 0.0
    cost_down: float = 0.0

    trades: List[Trade] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    settled: bool = False
    settlement_outcome: Optional[str] = None
    settlement_payout: float = 0.0

    @property
    def total_shares_up(self) -> float:
        return self.shares_up

    @property
    def total_shares_down(self) -> float:
        return self.shares_down

    @property
    def total_cost(self) -> float:
        return self.cost_up + self.cost_down

    @property
    def guaranteed_payout(self) -> float:
        """Minimum guaranteed payout at settlement."""
        return min(self.shares_up, self.shares_down)

    @property
    def expected_profit(self) -> float:
        """Expected profit based on guaranteed payout."""
        return self.guaranteed_payout - self.total_cost

    @property
    def expected_roi(self) -> float:
        """Expected ROI."""
        if self.total_cost == 0:
            return 0
        return self.expected_profit / self.total_cost

    @property
    def imbalance(self) -> float:
        """
        Imbalance ratio.
        1.0 = perfectly balanced
        >1.0 = more UP than DOWN
        <1.0 = more DOWN than UP
        """
        if self.shares_down == 0:
            return float('inf') if self.shares_up > 0 else 1.0
        return self.shares_up / self.shares_down

    @property
    def imbalance_pct(self) -> float:
        """Imbalance as percentage deviation from 1.0."""
        return abs(self.imbalance - 1.0)

    @property
    def excess_shares(self) -> Tuple[PositionSide, float]:
        """Returns which side has excess shares and how many."""
        diff = self.shares_up - self.shares_down
        if diff > 0:
            return PositionSide.UP, diff
        elif diff < 0:
            return PositionSide.DOWN, abs(diff)
        else:
            return PositionSide.UP, 0.0

    def add_trade(self, trade: Trade):
        """Add a trade to this position."""
        self.trades.append(trade)

        if trade.side == PositionSide.UP:
            self.shares_up += trade.shares
            self.cost_up += trade.cost
        else:
            self.shares_down += trade.shares
            self.cost_down += trade.cost

    def settle(self, outcome: str, payout: float):
        """Settle the position."""
        self.settled = True
        self.settlement_outcome = outcome
        self.settlement_payout = payout

    @property
    def realized_profit(self) -> Optional[float]:
        """Profit after settlement."""
        if not self.settled:
            return None
        return self.settlement_payout - self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_id': self.market_id,
            'market_slug': self.market_slug,
            'shares_up': self.shares_up,
            'shares_down': self.shares_down,
            'cost_up': self.cost_up,
            'cost_down': self.cost_down,
            'total_cost': self.total_cost,
            'guaranteed_payout': self.guaranteed_payout,
            'expected_profit': self.expected_profit,
            'expected_roi': self.expected_roi,
            'imbalance': self.imbalance,
            'trades_count': len(self.trades),
            'settled': self.settled,
            'settlement_outcome': self.settlement_outcome,
            'settlement_payout': self.settlement_payout,
            'realized_profit': self.realized_profit,
        }


class GabagoolPositionManager:
    """
    Manages positions across multiple markets.

    Responsibilities:
    - Track all positions
    - Calculate balancing needs
    - Enforce risk limits
    - Report P&L
    """

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._positions: Dict[str, MarketPosition] = {}
        self._trade_counter = 0
        self._total_deposited = 0.0
        self._total_withdrawn = 0.0

    def get_or_create_position(
        self,
        market_id: str,
        market_slug: str,
        up_token_id: str,
        down_token_id: str,
    ) -> MarketPosition:
        """Get existing position or create new one."""
        if market_id not in self._positions:
            self._positions[market_id] = MarketPosition(
                market_id=market_id,
                market_slug=market_slug,
                up_token_id=up_token_id,
                down_token_id=down_token_id,
            )
        return self._positions[market_id]

    def get_position(self, market_id: str) -> Optional[MarketPosition]:
        """Get position for a market."""
        return self._positions.get(market_id)

    def record_trade(
        self,
        market_id: str,
        side: PositionSide,
        shares: float,
        price: float,
        order_id: Optional[str] = None,
    ) -> Optional[Trade]:
        """Record a trade."""
        position = self._positions.get(market_id)
        if not position:
            logger.error(f"No position found for market {market_id}")
            return None

        self._trade_counter += 1
        trade = Trade(
            id=f"trade_{self._trade_counter}",
            market_id=market_id,
            side=side,
            shares=shares,
            price=price,
            cost=shares * price,
            timestamp=int(time.time() * 1000),
            order_id=order_id,
        )

        position.add_trade(trade)

        logger.info(
            f"Trade recorded: {side.value} {shares:.2f} shares @ ${price:.3f} "
            f"= ${trade.cost:.2f} | Market: {position.market_slug}"
        )

        return trade

    def calculate_rebalance(
        self,
        position: MarketPosition,
        budget: float,
        up_price: float,
        down_price: float,
    ) -> Dict[str, Any]:
        """
        Calculate what to buy to maintain balance.

        Strategy:
        1. If balanced, buy equal amounts of both
        2. If imbalanced, buy more of the lacking side

        Returns:
            Dict with 'buy_up', 'buy_down' (shares), and 'cost'
        """
        # Check imbalance
        imbalance_pct = position.imbalance_pct

        if imbalance_pct <= self.config.MAX_IMBALANCE_PCT:
            # Balanced enough - buy equal pairs
            cost_per_pair = up_price + down_price
            if cost_per_pair <= 0:
                return {'buy_up': 0, 'buy_down': 0, 'cost': 0}

            pairs = budget / cost_per_pair
            return {
                'buy_up': pairs,
                'buy_down': pairs,
                'cost': pairs * cost_per_pair,
            }

        else:
            # Need to rebalance - buy more of the lacking side
            excess_side, excess_amount = position.excess_shares

            if excess_side == PositionSide.UP:
                # More UP than DOWN - buy DOWN only
                shares_to_buy = min(excess_amount, budget / down_price) if down_price > 0 else 0
                return {
                    'buy_up': 0,
                    'buy_down': shares_to_buy,
                    'cost': shares_to_buy * down_price,
                }
            else:
                # More DOWN than UP - buy UP only
                shares_to_buy = min(excess_amount, budget / up_price) if up_price > 0 else 0
                return {
                    'buy_up': shares_to_buy,
                    'buy_down': 0,
                    'cost': shares_to_buy * up_price,
                }

    def can_trade(self, market_id: str, amount: float) -> Tuple[bool, str]:
        """
        Check if we can make a trade.

        Returns:
            Tuple of (can_trade, reason)
        """
        position = self._positions.get(market_id)

        # Check per-market limit
        if position:
            if position.total_cost + amount > self.config.MAX_PER_MARKET:
                return False, f"Would exceed per-market limit (${self.config.MAX_PER_MARKET})"

        # Check total exposure
        total_exposure = sum(p.total_cost for p in self._positions.values() if not p.settled)
        if total_exposure + amount > self.config.MAX_TOTAL_EXPOSURE:
            return False, f"Would exceed total exposure limit (${self.config.MAX_TOTAL_EXPOSURE})"

        return True, "OK"

    def settle_position(self, market_id: str, outcome: str) -> Optional[float]:
        """
        Settle a position when market resolves.

        Args:
            market_id: Market ID
            outcome: "UP" or "DOWN"

        Returns:
            Realized profit or None
        """
        position = self._positions.get(market_id)
        if not position:
            return None

        # Calculate payout
        if outcome.upper() == "UP":
            payout = position.shares_up * 1.00
        else:
            payout = position.shares_down * 1.00

        position.settle(outcome, payout)

        profit = position.realized_profit
        logger.info(
            f"Position settled: {position.market_slug} | "
            f"Outcome: {outcome} | "
            f"Payout: ${payout:.2f} | "
            f"Profit: ${profit:.2f}"
        )

        return profit

    # === Reporting ===

    @property
    def active_positions(self) -> List[MarketPosition]:
        """Get all active (unsettled) positions."""
        return [p for p in self._positions.values() if not p.settled]

    @property
    def settled_positions(self) -> List[MarketPosition]:
        """Get all settled positions."""
        return [p for p in self._positions.values() if p.settled]

    @property
    def total_exposure(self) -> float:
        """Total current exposure."""
        return sum(p.total_cost for p in self.active_positions)

    @property
    def total_expected_profit(self) -> float:
        """Total expected profit from active positions."""
        return sum(p.expected_profit for p in self.active_positions)

    @property
    def total_realized_profit(self) -> float:
        """Total realized profit from settled positions."""
        return sum(p.realized_profit or 0 for p in self.settled_positions)

    def get_summary(self) -> Dict[str, Any]:
        """Get position summary."""
        return {
            'active_positions': len(self.active_positions),
            'settled_positions': len(self.settled_positions),
            'total_exposure': self.total_exposure,
            'total_expected_profit': self.total_expected_profit,
            'total_realized_profit': self.total_realized_profit,
            'total_trades': self._trade_counter,
            'positions': [p.to_dict() for p in self._positions.values()],
        }

    def get_position_summary(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific position."""
        position = self._positions.get(market_id)
        if position:
            return position.to_dict()
        return None
