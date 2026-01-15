"""
Circular buffers for efficient data storage
Buffers otimizados para armazenamento de dados de mercado
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Deque, Dict, Any
import numpy as np

from .types import BookSnapshot, PaperTrade, OrderBookLevel


class TradeBuffer:
    """Circular buffer for recent trades."""

    __slots__ = ('_buffer', '_max_size')

    def __init__(self, size: int = 100):
        self._max_size = size
        self._buffer: Deque[PaperTrade] = deque(maxlen=size)

    def add(self, trade: PaperTrade) -> None:
        """Add trade to buffer."""
        self._buffer.append(trade)

    def get_recent(self, n: int = 10) -> List[PaperTrade]:
        """Get n most recent trades."""
        return list(self._buffer)[-n:]

    def get_all(self) -> List[PaperTrade]:
        """Get all trades in buffer."""
        return list(self._buffer)

    @property
    def count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    def get_volume(self, seconds: float = 60.0) -> float:
        """Get total volume in last N seconds."""
        now = datetime.now()
        total = 0.0
        for trade in reversed(self._buffer):
            age = (now - trade.timestamp).total_seconds()
            if age > seconds:
                break
            total += trade.notional
        return total

    def get_trade_count(self, seconds: float = 60.0) -> int:
        """Get trade count in last N seconds."""
        now = datetime.now()
        count = 0
        for trade in reversed(self._buffer):
            age = (now - trade.timestamp).total_seconds()
            if age > seconds:
                break
            count += 1
        return count


class PriceBuffer:
    """Circular buffer for price history."""

    __slots__ = ('_prices', '_timestamps', '_max_size', '_index')

    def __init__(self, size: int = 300):
        self._max_size = size
        self._prices: Deque[float] = deque(maxlen=size)
        self._timestamps: Deque[datetime] = deque(maxlen=size)

    def add(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add price to buffer."""
        self._prices.append(price)
        self._timestamps.append(timestamp or datetime.now())

    @property
    def latest(self) -> Optional[float]:
        return self._prices[-1] if self._prices else None

    @property
    def count(self) -> int:
        return len(self._prices)

    def get_prices(self, n: Optional[int] = None) -> List[float]:
        """Get last n prices."""
        if n is None:
            return list(self._prices)
        return list(self._prices)[-n:]

    def get_mean(self, n: int = 20) -> float:
        """Get mean of last n prices."""
        prices = self.get_prices(n)
        return np.mean(prices) if prices else 0.0

    def get_std(self, n: int = 20) -> float:
        """Get std of last n prices."""
        prices = self.get_prices(n)
        return np.std(prices) if len(prices) > 1 else 0.0

    def get_zscore(self, n: int = 20) -> float:
        """Get z-score of latest price vs recent history."""
        if self.count < 2:
            return 0.0

        prices = self.get_prices(n)
        mean = np.mean(prices[:-1])  # Exclude latest
        std = np.std(prices[:-1])

        if std == 0:
            return 0.0

        return (prices[-1] - mean) / std

    def get_returns(self, n: int = 20) -> List[float]:
        """Get percentage returns."""
        prices = self.get_prices(n + 1)
        if len(prices) < 2:
            return []

        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        return returns

    def get_volatility(self, n: int = 20) -> float:
        """Get recent volatility (std of returns)."""
        returns = self.get_returns(n)
        return np.std(returns) if returns else 0.0

    def clear(self) -> None:
        self._prices.clear()
        self._timestamps.clear()


class BookBuffer:
    """Buffer for order book snapshots."""

    __slots__ = ('_snapshots', '_max_size')

    def __init__(self, size: int = 50):
        self._max_size = size
        self._snapshots: Deque[BookSnapshot] = deque(maxlen=size)

    def add(self, snapshot: BookSnapshot) -> None:
        """Add snapshot to buffer."""
        self._snapshots.append(snapshot)

    @property
    def latest(self) -> Optional[BookSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    @property
    def count(self) -> int:
        return len(self._snapshots)

    def get_recent(self, n: int = 10) -> List[BookSnapshot]:
        """Get n most recent snapshots."""
        return list(self._snapshots)[-n:]

    def get_spread_history(self, n: int = 20) -> List[float]:
        """Get spread history."""
        snapshots = self.get_recent(n)
        return [s.spread for s in snapshots if s.spread is not None]

    def get_avg_spread(self, n: int = 20) -> float:
        """Get average spread."""
        spreads = self.get_spread_history(n)
        return np.mean(spreads) if spreads else 0.0

    def get_imbalance_history(self, n: int = 20) -> List[float]:
        """Get imbalance history."""
        snapshots = self.get_recent(n)
        return [s.imbalance for s in snapshots]

    def clear(self) -> None:
        self._snapshots.clear()


@dataclass
class MarketDataState:
    """Aggregated market data state."""
    yes_book: BookBuffer = field(default_factory=lambda: BookBuffer(50))
    no_book: BookBuffer = field(default_factory=lambda: BookBuffer(50))
    yes_trades: TradeBuffer = field(default_factory=lambda: TradeBuffer(100))
    no_trades: TradeBuffer = field(default_factory=lambda: TradeBuffer(100))
    binance_prices: PriceBuffer = field(default_factory=lambda: PriceBuffer(300))

    @property
    def yes_mid(self) -> Optional[float]:
        latest = self.yes_book.latest
        return latest.mid_price if latest else None

    @property
    def no_mid(self) -> Optional[float]:
        latest = self.no_book.latest
        return latest.mid_price if latest else None

    @property
    def pair_cost(self) -> Optional[float]:
        """Current pair cost (YES + NO mid prices)."""
        yes = self.yes_mid
        no = self.no_mid
        if yes is not None and no is not None:
            return yes + no
        return None

    def clear_all(self) -> None:
        """Clear all buffers."""
        self.yes_book.clear()
        self.no_book.clear()
        self.yes_trades.clear()
        self.no_trades.clear()
        self.binance_prices.clear()
