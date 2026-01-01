"""
Risk Management Module for Volatility Arbitrage Bot

Handles position sizing, trade limits, and capital protection.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum
from collections import deque

from .edge_detector import EdgeOpportunity


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Position sizing
    max_position_pct: float = 5.0  # Max % of portfolio per trade
    max_position_usd: float = 100.0  # Max USD per trade
    use_kelly: bool = True  # Use Kelly criterion
    kelly_fraction: float = 0.25  # Fraction of full Kelly (quarter Kelly)

    # Trade limits
    max_trades_per_minute: int = 3
    max_trades_per_hour: int = 30
    max_concurrent_positions: int = 5

    # Liquidity checks
    min_order_book_depth: float = 50.0  # Minimum USD on best ask
    max_slippage_pct: float = 1.0  # Max acceptable slippage

    # Cooldown
    cooldown_after_loss_seconds: int = 60  # Cooldown after a loss
    cooldown_after_win_seconds: int = 10  # Cooldown after a win

    # Capital protection
    max_daily_loss_pct: float = 10.0  # Max daily loss before stopping
    max_drawdown_pct: float = 20.0  # Max drawdown before stopping
    reserve_pct: float = 20.0  # Always keep this % in reserve


@dataclass
class PositionState:
    """Current position and risk state."""
    balance: float = 1000.0  # Current balance in USD
    starting_balance: float = 1000.0  # Starting balance for day
    high_water_mark: float = 1000.0  # Peak balance

    # Active positions
    active_positions: Dict[str, dict] = field(default_factory=dict)

    # Trade history
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0

    # Timestamps
    last_trade_time: int = 0
    last_loss_time: int = 0
    last_win_time: int = 0

    @property
    def daily_return_pct(self) -> float:
        """Daily return percentage."""
        if self.starting_balance <= 0:
            return 0.0
        return (self.balance - self.starting_balance) / self.starting_balance * 100

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from high water mark."""
        if self.high_water_mark <= 0:
            return 0.0
        return (self.high_water_mark - self.balance) / self.high_water_mark * 100

    @property
    def win_rate(self) -> float:
        """Win rate for today."""
        total = self.wins_today + self.losses_today
        if total == 0:
            return 0.0
        return self.wins_today / total


class RiskManager:
    """
    Manages risk for the trading bot.

    Responsibilities:
    - Position sizing based on edge and Kelly criterion
    - Trade frequency limits
    - Cooldown management
    - Capital protection (drawdown limits, daily loss limits)
    - Liquidity checks
    """

    def __init__(self, config: RiskConfig, initial_balance: float = 1000.0):
        self.config = config
        self.state = PositionState(
            balance=initial_balance,
            starting_balance=initial_balance,
            high_water_mark=initial_balance
        )

        # Trade timing history
        self._trade_times: deque = deque(maxlen=100)

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is allowed.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        now = int(time.time() * 1000)

        # Check daily loss limit
        if self.state.daily_return_pct < -self.config.max_daily_loss_pct:
            return False, f"Daily loss limit reached ({self.state.daily_return_pct:.1f}%)"

        # Check drawdown limit
        if self.state.drawdown_pct > self.config.max_drawdown_pct:
            return False, f"Max drawdown reached ({self.state.drawdown_pct:.1f}%)"

        # Check max concurrent positions
        if len(self.state.active_positions) >= self.config.max_concurrent_positions:
            return False, f"Max concurrent positions ({self.config.max_concurrent_positions})"

        # Check cooldown after loss
        if self.state.last_loss_time > 0:
            cooldown_ms = self.config.cooldown_after_loss_seconds * 1000
            if now - self.state.last_loss_time < cooldown_ms:
                remaining = (cooldown_ms - (now - self.state.last_loss_time)) / 1000
                return False, f"Loss cooldown ({remaining:.0f}s remaining)"

        # Check cooldown after win
        if self.state.last_win_time > 0:
            cooldown_ms = self.config.cooldown_after_win_seconds * 1000
            if now - self.state.last_win_time < cooldown_ms:
                remaining = (cooldown_ms - (now - self.state.last_win_time)) / 1000
                return False, f"Win cooldown ({remaining:.0f}s remaining)"

        # Check trades per minute
        one_minute_ago = now - 60_000
        recent_trades = sum(1 for t in self._trade_times if t > one_minute_ago)
        if recent_trades >= self.config.max_trades_per_minute:
            return False, f"Trade limit per minute ({self.config.max_trades_per_minute})"

        # Check trades per hour
        one_hour_ago = now - 3600_000
        hourly_trades = sum(1 for t in self._trade_times if t > one_hour_ago)
        if hourly_trades >= self.config.max_trades_per_hour:
            return False, f"Trade limit per hour ({self.config.max_trades_per_hour})"

        # Check reserve
        reserve_amount = self.state.starting_balance * (self.config.reserve_pct / 100)
        available = self.state.balance - reserve_amount
        if available <= 0:
            return False, "Reserve limit reached"

        return True, "OK"

    def calculate_position_size(
        self,
        edge: EdgeOpportunity,
        order_book_depth: float = 1000.0
    ) -> tuple[float, str]:
        """
        Calculate optimal position size for a trade.

        Args:
            edge: The detected edge opportunity
            order_book_depth: USD available at best price in order book

        Returns:
            Tuple of (size_usd: float, sizing_method: str)
        """
        # Check if we can trade first
        can_trade, reason = self.can_trade()
        if not can_trade:
            return 0.0, f"Cannot trade: {reason}"

        # Calculate available capital (excluding reserve)
        reserve = self.state.starting_balance * (self.config.reserve_pct / 100)
        available = self.state.balance - reserve

        if available <= 0:
            return 0.0, "No available capital"

        # Method 1: Percentage of portfolio
        max_by_pct = available * (self.config.max_position_pct / 100)

        # Method 2: Fixed max USD
        max_by_usd = self.config.max_position_usd

        # Method 3: Kelly criterion
        if self.config.use_kelly:
            kelly_full = edge.kelly_fraction
            kelly_adjusted = kelly_full * self.config.kelly_fraction  # Quarter Kelly
            max_by_kelly = available * kelly_adjusted
        else:
            max_by_kelly = float('inf')

        # Method 4: Liquidity constraint
        max_by_liquidity = order_book_depth * (1 - self.config.max_slippage_pct / 100)

        # Take minimum of all constraints
        position_size = min(max_by_pct, max_by_usd, max_by_kelly, max_by_liquidity)

        # Determine which constraint was binding
        if position_size == max_by_kelly:
            method = f"Kelly ({self.config.kelly_fraction * 100:.0f}%)"
        elif position_size == max_by_pct:
            method = f"Max % ({self.config.max_position_pct}%)"
        elif position_size == max_by_usd:
            method = f"Max USD (${self.config.max_position_usd})"
        else:
            method = f"Liquidity (${order_book_depth:.0f})"

        # Round to 2 decimal places
        position_size = round(max(0, position_size), 2)

        return position_size, method

    def check_liquidity(
        self,
        order_book: dict,
        required_size: float
    ) -> tuple[bool, float, str]:
        """
        Check if order book has sufficient liquidity.

        Args:
            order_book: Order book data with 'asks' list
            required_size: USD amount we want to trade

        Returns:
            Tuple of (sufficient: bool, available_depth: float, message: str)
        """
        asks = order_book.get('asks', [])

        if not asks:
            return False, 0.0, "No asks in order book"

        # Calculate depth available at best price
        best_ask = asks[0] if asks else None
        if not best_ask:
            return False, 0.0, "No best ask"

        best_price = float(best_ask.get('price', 0))
        best_size = float(best_ask.get('size', 0))
        depth_at_best = best_price * best_size

        # Check minimum depth
        if depth_at_best < self.config.min_order_book_depth:
            return False, depth_at_best, f"Insufficient depth (${depth_at_best:.2f})"

        # Calculate total depth for required size
        total_depth = 0.0
        total_cost = 0.0

        for ask in asks:
            price = float(ask.get('price', 0))
            size = float(ask.get('size', 0))
            level_value = price * size

            total_depth += level_value

            if total_depth >= required_size:
                break

            total_cost += level_value

        if total_depth < required_size:
            return False, total_depth, f"Insufficient total depth (${total_depth:.2f})"

        # Check slippage
        if depth_at_best > 0:
            slippage = (total_cost - depth_at_best) / depth_at_best * 100
            if slippage > self.config.max_slippage_pct:
                return False, total_depth, f"Slippage too high ({slippage:.1f}%)"

        return True, total_depth, "OK"

    def record_trade(self, position_id: str, entry_price: float, size_usd: float, direction: str):
        """Record a new trade entry."""
        now = int(time.time() * 1000)

        self._trade_times.append(now)
        self.state.last_trade_time = now
        self.state.trades_today += 1

        self.state.active_positions[position_id] = {
            'entry_price': entry_price,
            'size_usd': size_usd,
            'direction': direction,
            'entry_time': now
        }

        # Deduct from balance (tokens purchased)
        self.state.balance -= size_usd

    def record_exit(self, position_id: str, exit_price: float, won: bool):
        """Record a trade exit."""
        now = int(time.time() * 1000)

        position = self.state.active_positions.pop(position_id, None)
        if not position:
            return

        size = position['size_usd']
        entry = position['entry_price']

        # Calculate P&L
        if won:
            # Token pays out $1
            payout = size / entry  # Number of tokens * $1
            pnl = payout - size
            self.state.wins_today += 1
            self.state.last_win_time = now
        else:
            # Token pays $0
            pnl = -size
            self.state.losses_today += 1
            self.state.last_loss_time = now

        self.state.pnl_today += pnl
        self.state.balance += size + pnl  # Return capital + P&L

        # Update high water mark
        if self.state.balance > self.state.high_water_mark:
            self.state.high_water_mark = self.state.balance

    def reset_daily(self):
        """Reset daily stats (call at start of new day)."""
        self.state.starting_balance = self.state.balance
        self.state.trades_today = 0
        self.state.wins_today = 0
        self.state.losses_today = 0
        self.state.pnl_today = 0.0

    def get_status(self) -> dict:
        """Get current risk status for logging."""
        can_trade, reason = self.can_trade()

        return {
            'balance': round(self.state.balance, 2),
            'daily_pnl': round(self.state.pnl_today, 2),
            'daily_return_pct': round(self.state.daily_return_pct, 2),
            'drawdown_pct': round(self.state.drawdown_pct, 2),
            'trades_today': self.state.trades_today,
            'win_rate': round(self.state.win_rate * 100, 1),
            'active_positions': len(self.state.active_positions),
            'can_trade': can_trade,
            'trade_status': reason
        }


# Preset configurations
CONSERVATIVE_RISK = RiskConfig(
    max_position_pct=2.0,
    max_position_usd=50.0,
    kelly_fraction=0.1,  # 10% of Kelly
    max_trades_per_minute=1,
    max_trades_per_hour=10,
    max_daily_loss_pct=5.0,
    max_drawdown_pct=10.0,
    cooldown_after_loss_seconds=120,
)

MODERATE_RISK = RiskConfig(
    max_position_pct=5.0,
    max_position_usd=100.0,
    kelly_fraction=0.25,  # 25% of Kelly
    max_trades_per_minute=3,
    max_trades_per_hour=30,
    max_daily_loss_pct=10.0,
    max_drawdown_pct=20.0,
    cooldown_after_loss_seconds=60,
)

AGGRESSIVE_RISK = RiskConfig(
    max_position_pct=10.0,
    max_position_usd=250.0,
    kelly_fraction=0.5,  # 50% of Kelly
    max_trades_per_minute=5,
    max_trades_per_hour=60,
    max_daily_loss_pct=15.0,
    max_drawdown_pct=30.0,
    cooldown_after_loss_seconds=30,
)
