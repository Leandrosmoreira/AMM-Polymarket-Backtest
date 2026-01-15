"""
Core types for Paper Trading
Estruturas de dados para o sistema de paper trading
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class Side(Enum):
    """Trade side."""
    BUY = "BUY"
    SELL = "SELL"


class TokenType(Enum):
    """Token type for binary markets."""
    YES = "YES"
    NO = "NO"


class AgentType(Enum):
    """Agent types in the system."""
    MICROSTRUCTURE = "microstructure"
    EDGE = "edge"
    RISK = "risk"
    MARKET_MAKING = "market_making"


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: float

    @property
    def notional(self) -> float:
        return self.price * self.size


@dataclass
class BookSnapshot:
    """Order book snapshot for a token."""
    token_id: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def bid_depth(self) -> float:
        return sum(level.notional for level in self.bids[:5])

    @property
    def ask_depth(self) -> float:
        return sum(level.notional for level in self.asks[:5])

    @property
    def imbalance(self) -> float:
        """Book imbalance: positive = more bids, negative = more asks."""
        total = self.bid_depth + self.ask_depth
        if total == 0:
            return 0.0
        return (self.bid_depth - self.ask_depth) / total


@dataclass
class MarketInfo:
    """Information about a binary market."""
    market_id: str
    condition_id: str
    question: str
    yes_token_id: str
    no_token_id: str
    start_time: datetime
    end_time: datetime
    outcome: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def time_remaining(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        return max(0, (self.end_time - now).total_seconds())

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        return self.time_remaining(now) <= 0


@dataclass
class PaperPosition:
    """Position tracking for paper trading."""
    market_id: str
    yes_qty: float = 0.0
    no_qty: float = 0.0
    yes_cost: float = 0.0
    no_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def total_cost(self) -> float:
        return self.yes_cost + self.no_cost

    @property
    def pair_cost(self) -> float:
        """Custo por par YES+NO hedgeado."""
        min_qty = min(self.yes_qty, self.no_qty)
        if min_qty == 0:
            return 0.0
        # Custo proporcional para a quantidade hedgeada
        yes_unit = self.yes_cost / self.yes_qty if self.yes_qty > 0 else 0
        no_unit = self.no_cost / self.no_qty if self.no_qty > 0 else 0
        return yes_unit + no_unit

    @property
    def hedge_qty(self) -> float:
        """Quantidade totalmente hedgeada."""
        return min(self.yes_qty, self.no_qty)

    @property
    def locked_profit(self) -> float:
        """Lucro garantido pela hedge."""
        if self.hedge_qty == 0:
            return 0.0
        return self.hedge_qty * (1.0 - self.pair_cost)

    @property
    def unhedged_yes(self) -> float:
        return max(0, self.yes_qty - self.no_qty)

    @property
    def unhedged_no(self) -> float:
        return max(0, self.no_qty - self.yes_qty)

    @property
    def is_balanced(self) -> bool:
        """Check if position is reasonably balanced."""
        if self.yes_qty == 0 and self.no_qty == 0:
            return True
        total = self.yes_qty + self.no_qty
        ratio = abs(self.yes_qty - self.no_qty) / total
        return ratio < 0.1  # 10% tolerance

    def add_yes(self, qty: float, price: float) -> None:
        """Add YES tokens to position."""
        cost = qty * price
        self.yes_qty += qty
        self.yes_cost += cost

    def add_no(self, qty: float, price: float) -> None:
        """Add NO tokens to position."""
        cost = qty * price
        self.no_qty += qty
        self.no_cost += cost

    def reset(self) -> None:
        """Reset position."""
        self.yes_qty = 0.0
        self.no_qty = 0.0
        self.yes_cost = 0.0
        self.no_cost = 0.0


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    market_id: str = ""
    token_type: TokenType = TokenType.YES
    side: Side = Side.BUY
    price: float = 0.0
    size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def notional(self) -> float:
        return self.price * self.size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "market_id": self.market_id,
            "token_type": self.token_type.value,
            "side": self.side.value,
            "price": self.price,
            "size": self.size,
            "notional": self.notional,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PaperStats:
    """Statistics for paper trading session."""
    markets_traded: int = 0
    total_trades: int = 0
    total_yes_bought: float = 0.0
    total_no_bought: float = 0.0
    total_cost: float = 0.0
    total_payout: float = 0.0
    wins: int = 0
    losses: int = 0
    avg_pair_cost: float = 0.0
    best_pair_cost: float = 1.0
    worst_pair_cost: float = 0.0

    # Tracking por agente
    blocked_by_microstructure: int = 0
    blocked_by_edge: int = 0
    blocked_by_risk: int = 0
    blocked_by_market_making: int = 0

    @property
    def total_pnl(self) -> float:
        return self.total_payout - self.total_cost

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_trade_size(self) -> float:
        return self.total_cost / self.total_trades if self.total_trades > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "markets_traded": self.markets_traded,
            "total_trades": self.total_trades,
            "total_cost": round(self.total_cost, 2),
            "total_payout": round(self.total_payout, 2),
            "total_pnl": round(self.total_pnl, 2),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "avg_pair_cost": round(self.avg_pair_cost, 4),
            "best_pair_cost": round(self.best_pair_cost, 4),
            "worst_pair_cost": round(self.worst_pair_cost, 4),
            "blocked_by_microstructure": self.blocked_by_microstructure,
            "blocked_by_edge": self.blocked_by_edge,
            "blocked_by_risk": self.blocked_by_risk,
            "blocked_by_market_making": self.blocked_by_market_making,
        }


@dataclass
class AgentDecision:
    """Decision output from an agent."""
    agent: AgentType
    should_trade: bool
    confidence: float = 0.0
    reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.value,
            "should_trade": self.should_trade,
            "confidence": self.confidence,
            "reason": self.reason,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Quote:
    """Quote generated by market making agent."""
    token_type: TokenType
    side: Side
    price: float
    size: float

    @property
    def notional(self) -> float:
        return self.price * self.size
