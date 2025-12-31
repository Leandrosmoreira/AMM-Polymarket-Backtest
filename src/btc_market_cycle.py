"""
BTC Market Cycle Manager
Handles 15-minute market cycles and position tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging

from config.btc_risk_params import BTCRiskParams

logger = logging.getLogger(__name__)

# Market duration in milliseconds
FIFTEEN_MIN_MS = 15 * 60 * 1000  # 900000 ms


def get_market_start_timestamp(timestamp_ms: int) -> int:
    """
    Get the start timestamp of the 15-minute market containing this timestamp.

    Markets start at X:00, X:15, X:30, X:45.

    Args:
        timestamp_ms: Any timestamp in milliseconds

    Returns:
        Start timestamp of the market period in milliseconds
    """
    return (timestamp_ms // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS


def get_market_end_timestamp(timestamp_ms: int) -> int:
    """
    Get the end timestamp of the 15-minute market containing this timestamp.

    Args:
        timestamp_ms: Any timestamp in milliseconds

    Returns:
        End timestamp of the market period in milliseconds
    """
    return get_market_start_timestamp(timestamp_ms) + FIFTEEN_MIN_MS


def get_time_remaining(timestamp_ms: int) -> int:
    """
    Get time remaining in current market in milliseconds.

    Args:
        timestamp_ms: Current timestamp in milliseconds

    Returns:
        Milliseconds until market closes
    """
    end = get_market_end_timestamp(timestamp_ms)
    return end - timestamp_ms


def get_elapsed_time(timestamp_ms: int) -> int:
    """
    Get time elapsed in current market in milliseconds.

    Args:
        timestamp_ms: Current timestamp in milliseconds

    Returns:
        Milliseconds since market opened
    """
    start = get_market_start_timestamp(timestamp_ms)
    return timestamp_ms - start


@dataclass
class BTCTrade:
    """Represents a single trade in a market."""
    timestamp_ms: int
    side: str  # 'UP' or 'DOWN'
    shares: float
    price: float  # Token price paid (0-1)
    cost: float  # USD cost
    opportunity: float  # Expected edge when traded

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp_ms': self.timestamp_ms,
            'side': self.side,
            'shares': self.shares,
            'price': self.price,
            'cost': self.cost,
            'opportunity': self.opportunity,
        }


@dataclass
class MarketPosition:
    """Position state for a single 15-minute market."""
    market_start_ms: int
    price_to_beat: float
    shares_up: float = 0.0
    shares_down: float = 0.0
    cost_up: float = 0.0
    cost_down: float = 0.0
    trades: List[BTCTrade] = field(default_factory=list)
    outcome: Optional[str] = None  # 'UP' or 'DOWN' when resolved
    final_price: Optional[float] = None
    payout: float = 0.0
    profit: float = 0.0

    @property
    def total_invested(self) -> float:
        """Total USD invested in this market."""
        return self.cost_up + self.cost_down

    @property
    def total_shares_up(self) -> float:
        """Total UP shares held."""
        return self.shares_up

    @property
    def total_shares_down(self) -> float:
        """Total DOWN shares held."""
        return self.shares_down

    @property
    def desbalanceamento(self) -> float:
        """
        Calculate imbalance ratio.
        Returns ratio of larger side to smaller side.
        """
        if self.shares_up == 0 and self.shares_down == 0:
            return 1.0
        if self.shares_down == 0:
            return float('inf') if self.shares_up > 0 else 1.0
        return self.shares_up / self.shares_down

    def is_balanced(self, max_desbalanceamento: float = 0.20) -> bool:
        """Check if position is within balance limits."""
        ratio = self.desbalanceamento
        lower = 1.0 / (1.0 + max_desbalanceamento)  # 0.833 for 20%
        upper = 1.0 + max_desbalanceamento  # 1.2 for 20%
        return lower <= ratio <= upper

    def can_buy_side(self, side: str, max_desbalanceamento: float = 0.20) -> bool:
        """
        Check if can buy a specific side without exceeding balance limits.

        Args:
            side: 'UP' or 'DOWN'
            max_desbalanceamento: Maximum allowed imbalance

        Returns:
            True if can buy this side
        """
        if self.shares_up == 0 and self.shares_down == 0:
            return True

        ratio = self.desbalanceamento

        if side == 'UP':
            # If UP is already too high, can't buy more UP
            return ratio <= (1.0 + max_desbalanceamento)
        else:
            # If DOWN is already too high (ratio too low), can't buy more DOWN
            return ratio >= 1.0 / (1.0 + max_desbalanceamento)

    def add_trade(self, trade: BTCTrade) -> None:
        """Add a trade to this position."""
        self.trades.append(trade)
        if trade.side == 'UP':
            self.shares_up += trade.shares
            self.cost_up += trade.cost
        else:
            self.shares_down += trade.shares
            self.cost_down += trade.cost

    def resolve(self, final_price: float) -> None:
        """
        Resolve the market with final price.

        Args:
            final_price: Final BTC price at market close
        """
        self.final_price = final_price

        if final_price > self.price_to_beat:
            self.outcome = 'UP'
            self.payout = self.shares_up * 1.00  # UP wins $1 each
        elif final_price < self.price_to_beat:
            self.outcome = 'DOWN'
            self.payout = self.shares_down * 1.00  # DOWN wins $1 each
        else:
            # Rare tie case
            self.outcome = 'TIE'
            self.payout = (self.shares_up + self.shares_down) * 0.50

        self.profit = self.payout - self.total_invested

    def to_dict(self) -> Dict[str, Any]:
        return {
            'market_start_ms': self.market_start_ms,
            'price_to_beat': self.price_to_beat,
            'shares_up': self.shares_up,
            'shares_down': self.shares_down,
            'cost_up': self.cost_up,
            'cost_down': self.cost_down,
            'total_invested': self.total_invested,
            'trades_count': len(self.trades),
            'outcome': self.outcome,
            'final_price': self.final_price,
            'payout': self.payout,
            'profit': self.profit,
        }


class MarketCycleManager:
    """
    Manages market cycles and trading decisions.

    Tracks:
    - Current market state
    - Position for current market
    - History of all markets
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_params: BTCRiskParams = None
    ):
        """
        Initialize the cycle manager.

        Args:
            initial_capital: Capital per market (resets each cycle)
            risk_params: Risk parameters for trading
        """
        self.initial_capital = initial_capital
        self.risk_params = risk_params or BTCRiskParams()

        self.current_market_start: Optional[int] = None
        self.current_position: Optional[MarketPosition] = None
        self.remaining_capital: float = initial_capital

        self.completed_markets: List[MarketPosition] = []
        self.total_profit: float = 0.0

    def reset_for_new_market(self, market_start_ms: int, price_to_beat: float) -> None:
        """
        Reset state for a new 15-minute market.

        Args:
            market_start_ms: Start timestamp of new market
            price_to_beat: Price to beat for this market
        """
        self.current_market_start = market_start_ms
        self.current_position = MarketPosition(
            market_start_ms=market_start_ms,
            price_to_beat=price_to_beat,
        )
        self.remaining_capital = self.initial_capital

        logger.debug(
            f"New market started: {market_start_ms}, "
            f"Price to Beat: ${price_to_beat:.2f}"
        )

    def check_market_change(
        self,
        timestamp_ms: int,
        ticks: List[Dict[str, Any]]
    ) -> bool:
        """
        Check if we've entered a new market and handle transition.

        Args:
            timestamp_ms: Current timestamp
            ticks: Available price ticks

        Returns:
            True if new market started
        """
        new_market_start = get_market_start_timestamp(timestamp_ms)

        if self.current_market_start is None:
            # First market
            price_to_beat = self._get_price_to_beat(new_market_start, ticks)
            if price_to_beat is not None:
                self.reset_for_new_market(new_market_start, price_to_beat)
                return True
            return False

        if new_market_start != self.current_market_start:
            # New market - close previous first
            if self.current_position is not None:
                self._close_current_market(ticks)

            # Start new market
            price_to_beat = self._get_price_to_beat(new_market_start, ticks)
            if price_to_beat is not None:
                self.reset_for_new_market(new_market_start, price_to_beat)
                return True

        return False

    def _get_price_to_beat(
        self,
        market_start_ms: int,
        ticks: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Get the price to beat for a market (first tick of the period)."""
        # Find first tick at or after market start
        for tick in ticks:
            if tick['ts'] >= market_start_ms:
                return tick['price']

        # If no tick found, return None
        return None

    def _get_final_price(
        self,
        market_end_ms: int,
        ticks: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Get the final price at market close."""
        # Find last tick before or at market end
        final_tick = None
        for tick in ticks:
            if tick['ts'] <= market_end_ms:
                final_tick = tick
            else:
                break

        return final_tick['price'] if final_tick else None

    def _close_current_market(self, ticks: List[Dict[str, Any]]) -> None:
        """Close the current market position."""
        if self.current_position is None:
            return

        market_end = self.current_market_start + FIFTEEN_MIN_MS
        final_price = self._get_final_price(market_end, ticks)

        if final_price is not None:
            self.current_position.resolve(final_price)
            self.total_profit += self.current_position.profit

            logger.debug(
                f"Market closed: {self.current_market_start}, "
                f"Outcome: {self.current_position.outcome}, "
                f"Profit: ${self.current_position.profit:.2f}"
            )

        self.completed_markets.append(self.current_position)
        self.current_position = None
        self.current_market_start = None

    def should_trade(
        self,
        timestamp_ms: int,
        opportunity: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Determine if should make a trade.

        Args:
            timestamp_ms: Current timestamp
            opportunity: Opportunity analysis from probability calculator

        Returns:
            Tuple of (should_trade, side, amount)
        """
        if self.current_position is None:
            return False, None, None

        # Check time remaining
        time_remaining = get_time_remaining(timestamp_ms)
        if time_remaining < self.risk_params.MIN_TEMPO_RESTANTE * 1000:
            return False, None, None

        # Check investment limit
        if self.remaining_capital <= 0:
            return False, None, None

        if self.current_position.total_invested >= self.risk_params.MAX_INVESTIMENTO:
            return False, None, None

        # Check opportunity
        best_side = opportunity.get('best_side', 'NONE')
        best_opp = opportunity.get('best_opp', 0)

        if best_side == 'NONE' or best_opp < self.risk_params.MIN_OPORTUNIDADE:
            return False, None, None

        # Check if can buy this side (balance check)
        if not self.current_position.can_buy_side(
            best_side,
            self.risk_params.MAX_DESBALANCEAMENTO
        ):
            # Try the other side if it also has positive opportunity
            other_side = 'DOWN' if best_side == 'UP' else 'UP'
            other_opp = opportunity.get(f'opp_{other_side.lower()}', 0)

            if other_opp >= self.risk_params.MIN_OPORTUNIDADE:
                if self.current_position.can_buy_side(
                    other_side,
                    self.risk_params.MAX_DESBALANCEAMENTO
                ):
                    best_side = other_side
                    best_opp = other_opp
                else:
                    return False, None, None
            else:
                return False, None, None

        # Calculate trade size
        trade_size_type = self.risk_params.get_trade_size(best_opp)
        trade_pct = self.risk_params.get_trade_pct(trade_size_type)

        if trade_pct <= 0:
            return False, None, None

        amount = min(
            self.remaining_capital * trade_pct,
            self.remaining_capital,
            self.risk_params.MAX_INVESTIMENTO - self.current_position.total_invested
        )

        if amount < 1.0:  # Minimum $1 trade
            return False, None, None

        return True, best_side, amount

    def execute_trade(
        self,
        timestamp_ms: int,
        side: str,
        amount: float,
        token_price: float,
        opportunity: float
    ) -> Optional[BTCTrade]:
        """
        Execute a trade.

        Args:
            timestamp_ms: Current timestamp
            side: 'UP' or 'DOWN'
            amount: USD amount to invest
            token_price: Token price (0-1)
            opportunity: Expected edge

        Returns:
            The executed trade or None
        """
        if self.current_position is None:
            return None

        if amount > self.remaining_capital:
            amount = self.remaining_capital

        if amount < 1.0:
            return None

        # Calculate shares
        shares = amount / token_price if token_price > 0 else 0

        trade = BTCTrade(
            timestamp_ms=timestamp_ms,
            side=side,
            shares=shares,
            price=token_price,
            cost=amount,
            opportunity=opportunity,
        )

        self.current_position.add_trade(trade)
        self.remaining_capital -= amount

        logger.debug(
            f"Trade executed: {side}, "
            f"Shares: {shares:.2f}, "
            f"Cost: ${amount:.2f}, "
            f"Opp: {opportunity*100:.1f}%"
        )

        return trade

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all completed markets."""
        if not self.completed_markets:
            return {
                'total_markets': 0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
            }

        wins = sum(1 for m in self.completed_markets if m.profit > 0)
        total = len(self.completed_markets)

        return {
            'total_markets': total,
            'total_profit': self.total_profit,
            'win_rate': wins / total if total > 0 else 0,
            'avg_profit': self.total_profit / total if total > 0 else 0,
            'winning_markets': wins,
            'losing_markets': total - wins,
        }
