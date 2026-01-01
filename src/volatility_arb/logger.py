"""
Logging and Monitoring System for Volatility Arbitrage Bot

Comprehensive logging for trades, market analysis, and performance.
"""

import json
import time
import gzip
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from collections import deque


logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Log entry for a trade."""
    timestamp: int
    trade_id: str
    direction: str  # "up" or "down"

    # Market data
    btc_price: float
    strike_price: float
    time_remaining_seconds: int

    # Model data
    model_probability: float
    model_confidence: float
    volatility: float

    # Market prices
    market_up_price: float
    market_down_price: float
    market_total: float

    # Edge data
    edge_percent: float
    expected_value: float
    kelly_fraction: float

    # Execution data
    size_usd: float
    fill_price: float
    tokens: float

    # Risk data
    balance_before: float
    position_sizing_method: str


@dataclass
class SettlementLog:
    """Log entry for a trade settlement."""
    timestamp: int
    trade_id: str
    direction: str
    entry_price: float
    size_usd: float
    tokens: float
    won: bool
    payout: float
    pnl: float
    hold_time_seconds: int


@dataclass
class MarketAnalysisLog:
    """Log entry for market analysis (even without trade)."""
    timestamp: int
    btc_price: float
    volatility: float
    volatility_annualized: float

    # Model probabilities
    model_up_prob: float
    model_down_prob: float
    model_confidence: float

    # Market prices
    market_up_price: float
    market_down_price: float
    market_spread: float

    # Edges
    up_edge_pct: float
    down_edge_pct: float

    # Context
    time_remaining_seconds: int
    trade_signal: str  # "buy_up", "buy_down", "no_trade"
    trade_blocked_reason: Optional[str] = None


class TradingLogger:
    """
    Comprehensive logging for the trading bot.

    Features:
    - Trade logging with all relevant metrics
    - Performance tracking
    - Market analysis logging
    - File-based persistence
    """

    def __init__(
        self,
        log_dir: str = "logs/volatility_arb",
        save_interval: int = 60,  # Save to file every 60 seconds
        console_level: int = logging.INFO,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self._last_save = time.time()

        # In-memory buffers
        self._trades: List[TradeLog] = []
        self._settlements: List[SettlementLog] = []
        self._analysis: deque = deque(maxlen=1000)

        # Performance metrics
        self._session_start = time.time()
        self._total_trades = 0
        self._total_wins = 0
        self._total_losses = 0
        self._total_pnl = 0.0

        # Configure logging
        self._setup_logging(console_level)

    def _setup_logging(self, console_level: int):
        """Configure logging handlers."""
        # File handler for all logs
        file_handler = logging.FileHandler(
            self.log_dir / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def log_trade(self, trade: TradeLog):
        """Log a new trade entry."""
        self._trades.append(trade)
        self._total_trades += 1

        logger.info(
            f"TRADE | {trade.direction.upper()} | "
            f"${trade.size_usd:.2f} @ {trade.fill_price:.4f} | "
            f"Edge: {trade.edge_percent:.1f}% | "
            f"BTC: ${trade.btc_price:,.2f}"
        )

        self._check_save()

    def log_settlement(self, settlement: SettlementLog):
        """Log a trade settlement."""
        self._settlements.append(settlement)

        if settlement.won:
            self._total_wins += 1
        else:
            self._total_losses += 1

        self._total_pnl += settlement.pnl

        result = "WIN" if settlement.won else "LOSS"
        logger.info(
            f"SETTLED | {settlement.direction.upper()} | {result} | "
            f"P&L: ${settlement.pnl:+.2f} | "
            f"Hold: {settlement.hold_time_seconds}s"
        )

        self._check_save()

    def log_analysis(self, analysis: MarketAnalysisLog):
        """Log market analysis."""
        self._analysis.append(analysis)

        # Only log to console if there's something interesting
        if abs(analysis.up_edge_pct) > 2 or abs(analysis.down_edge_pct) > 2:
            logger.debug(
                f"ANALYSIS | BTC: ${analysis.btc_price:,.2f} | "
                f"Vol: {analysis.volatility_annualized*100:.1f}% | "
                f"UP Edge: {analysis.up_edge_pct:+.1f}% | "
                f"DOWN Edge: {analysis.down_edge_pct:+.1f}% | "
                f"Signal: {analysis.trade_signal}"
            )

        self._check_save()

    def log_status(
        self,
        balance: float,
        positions: int,
        can_trade: bool,
        reason: str
    ):
        """Log current status."""
        win_rate = self._total_wins / max(1, self._total_wins + self._total_losses) * 100

        logger.info(
            f"STATUS | Balance: ${balance:.2f} | "
            f"P&L: ${self._total_pnl:+.2f} | "
            f"Trades: {self._total_trades} | "
            f"Win Rate: {win_rate:.1f}% | "
            f"Positions: {positions} | "
            f"{'Can Trade' if can_trade else reason}"
        )

    def _check_save(self):
        """Check if it's time to save to file."""
        if time.time() - self._last_save > self.save_interval:
            self._save_to_file()
            self._last_save = time.time()

    def _save_to_file(self):
        """Save buffered data to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save trades
        if self._trades:
            trades_file = self.log_dir / f"trades_{timestamp}.json.gz"
            with gzip.open(trades_file, 'wt', encoding='utf-8') as f:
                json.dump([asdict(t) for t in self._trades], f)
            self._trades = []

        # Save settlements
        if self._settlements:
            settlements_file = self.log_dir / f"settlements_{timestamp}.json.gz"
            with gzip.open(settlements_file, 'wt', encoding='utf-8') as f:
                json.dump([asdict(s) for s in self._settlements], f)
            self._settlements = []

        # Save analysis (just recent)
        if self._analysis:
            analysis_file = self.log_dir / f"analysis_{timestamp}.json.gz"
            with gzip.open(analysis_file, 'wt', encoding='utf-8') as f:
                json.dump([asdict(a) for a in self._analysis], f)

    def get_performance_summary(self) -> dict:
        """Get performance summary."""
        session_hours = (time.time() - self._session_start) / 3600
        total_closed = self._total_wins + self._total_losses

        return {
            'session_hours': round(session_hours, 2),
            'total_trades': self._total_trades,
            'closed_trades': total_closed,
            'wins': self._total_wins,
            'losses': self._total_losses,
            'win_rate': round(self._total_wins / max(1, total_closed) * 100, 1),
            'total_pnl': round(self._total_pnl, 2),
            'pnl_per_trade': round(self._total_pnl / max(1, total_closed), 2),
            'trades_per_hour': round(self._total_trades / max(0.1, session_hours), 1),
        }

    def print_summary(self):
        """Print performance summary to console."""
        summary = self.get_performance_summary()

        print("\n" + "="*60)
        print("TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Duration:        {summary['session_hours']:.1f} hours")
        print(f"Total Trades:    {summary['total_trades']}")
        print(f"Closed Trades:   {summary['closed_trades']}")
        print(f"Wins/Losses:     {summary['wins']}/{summary['losses']}")
        print(f"Win Rate:        {summary['win_rate']:.1f}%")
        print(f"Total P&L:       ${summary['total_pnl']:+.2f}")
        print(f"P&L per Trade:   ${summary['pnl_per_trade']:+.2f}")
        print(f"Trades/Hour:     {summary['trades_per_hour']:.1f}")
        print("="*60 + "\n")

    def flush(self):
        """Force save all buffered data."""
        self._save_to_file()


class MetricsCollector:
    """
    Collects and aggregates real-time metrics.

    For dashboards and monitoring.
    """

    def __init__(self, window_minutes: int = 60):
        self.window_ms = window_minutes * 60 * 1000

        # Time-series data
        self._btc_prices: deque = deque(maxlen=10000)
        self._volatilities: deque = deque(maxlen=10000)
        self._edges: deque = deque(maxlen=10000)
        self._spreads: deque = deque(maxlen=10000)

    def record_tick(
        self,
        btc_price: float,
        volatility: float,
        up_edge: float,
        down_edge: float,
        spread: float,
    ):
        """Record a market tick."""
        now = int(time.time() * 1000)

        self._btc_prices.append((now, btc_price))
        self._volatilities.append((now, volatility))
        self._edges.append((now, up_edge, down_edge))
        self._spreads.append((now, spread))

    def get_stats(self) -> dict:
        """Get aggregated statistics."""
        now = int(time.time() * 1000)
        cutoff = now - self.window_ms

        # Filter to window
        recent_btc = [p for t, p in self._btc_prices if t > cutoff]
        recent_vol = [v for t, v in self._volatilities if t > cutoff]
        recent_edges = [(u, d) for t, u, d in self._edges if t > cutoff]
        recent_spreads = [s for t, s in self._spreads if t > cutoff]

        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def safe_max(lst):
            return max(lst) if lst else 0

        def safe_min(lst):
            return min(lst) if lst else 0

        up_edges = [u for u, d in recent_edges]
        down_edges = [d for u, d in recent_edges]

        return {
            'btc_price': recent_btc[-1] if recent_btc else 0,
            'btc_change_pct': (
                (recent_btc[-1] - recent_btc[0]) / recent_btc[0] * 100
                if len(recent_btc) > 1 else 0
            ),
            'avg_volatility': safe_avg(recent_vol),
            'max_up_edge': safe_max(up_edges),
            'max_down_edge': safe_max(down_edges),
            'avg_spread': safe_avg(recent_spreads),
            'samples': len(recent_btc),
        }
