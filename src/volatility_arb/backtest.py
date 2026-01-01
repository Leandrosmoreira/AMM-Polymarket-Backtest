"""
Backtest Engine for Volatility Arbitrage Strategy

Runs historical simulation using collected real data.
"""

import json
import gzip
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

from .volatility import VolatilityCalculator
from .probability import ProbabilityModel, AdaptiveProbabilityModel
from .edge_detector import EdgeDetector, MarketPrices, TradeSignal
from .risk_manager import RiskConfig, MODERATE_RISK

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """A trade in the backtest."""
    trade_id: int
    timestamp: int
    direction: str  # "up" or "down"
    entry_price: float
    size_usd: float
    tokens: float
    btc_price: float
    model_prob: float
    market_price: float
    edge_pct: float
    volatility: float
    time_remaining: int

    # Settlement
    settled: bool = False
    won: Optional[bool] = None
    pnl: float = 0.0
    exit_timestamp: Optional[int] = None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    # Summary
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    # Capital
    initial_balance: float = 1000.0
    final_balance: float = 1000.0
    max_balance: float = 1000.0
    min_balance: float = 1000.0
    max_drawdown_pct: float = 0.0

    # Metrics
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0

    # Edge stats
    avg_edge_taken: float = 0.0
    max_edge_seen: float = 0.0
    edges_above_threshold: int = 0

    # Data stats
    total_ticks: int = 0
    total_markets: int = 0
    data_duration_hours: float = 0.0

    # Trade list
    trades: List[BacktestTrade] = field(default_factory=list)

    # Time series
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    edge_history: List[Tuple[int, float, float]] = field(default_factory=list)


class VolatilityArbBacktest:
    """
    Backtest engine for volatility arbitrage strategy.

    Uses real collected data to simulate trading performance.
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        min_edge_pct: float = 3.0,
        risk_config: Optional[RiskConfig] = None,
        use_momentum: bool = True,
        market_duration_seconds: int = 900,  # 15 minutes
    ):
        self.initial_balance = initial_balance
        self.min_edge_pct = min_edge_pct
        self.risk_config = risk_config or MODERATE_RISK
        self.use_momentum = use_momentum
        self.market_duration_seconds = market_duration_seconds

        # Components
        self.volatility = VolatilityCalculator(
            window_seconds=300,
            min_samples=30
        )

        if use_momentum:
            self.probability = AdaptiveProbabilityModel()
        else:
            self.probability = ProbabilityModel()

        self.edge_detector = EdgeDetector(
            min_edge_percent=min_edge_pct,
            min_confidence=0.4,
            fee_percent=1.0
        )

    def load_data(self, data_path: str) -> Dict[str, Any]:
        """Load data from file or directory."""
        path = Path(data_path)

        all_ticks = []
        all_prices = []
        all_books = []

        if path.is_dir():
            # Load all files in directory
            files = sorted(path.glob("*.json.gz")) + sorted(path.glob("*.json"))
            logger.info(f"Found {len(files)} data files")

            for file in files:
                data = self._load_file(file)
                if data:
                    all_ticks.extend(data.get('chainlink_ticks', []))
                    all_prices.extend(data.get('price_changes', []))
                    all_books.extend(data.get('order_books', []))
        else:
            # Single file
            data = self._load_file(path)
            if data:
                all_ticks = data.get('chainlink_ticks', [])
                all_prices = data.get('price_changes', [])
                all_books = data.get('order_books', [])

        # Sort by timestamp
        all_ticks.sort(key=lambda x: x.get('ts', 0))
        all_prices.sort(key=lambda x: x.get('ts', 0))

        logger.info(f"Loaded {len(all_ticks)} ticks, {len(all_prices)} price changes")

        return {
            'chainlink_ticks': all_ticks,
            'price_changes': all_prices,
            'order_books': all_books
        }

    def _load_file(self, path: Path) -> Optional[Dict]:
        """Load a single data file."""
        try:
            if path.suffix == '.gz':
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def run(
        self,
        data: Dict[str, Any],
        verbose: bool = True
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: Dict with 'chainlink_ticks' and 'price_changes'
            verbose: Print progress

        Returns:
            BacktestResult with all metrics and trades
        """
        result = BacktestResult(initial_balance=self.initial_balance)

        ticks = data.get('chainlink_ticks', [])
        prices = data.get('price_changes', [])

        if not ticks:
            logger.error("No tick data provided")
            return result

        result.total_ticks = len(ticks)

        # Calculate data duration
        if len(ticks) >= 2:
            duration_ms = ticks[-1]['ts'] - ticks[0]['ts']
            result.data_duration_hours = duration_ms / 1000 / 3600

        # Build price index for fast lookup
        price_index = self._build_price_index(prices)

        # Segment data into markets
        markets = self._segment_markets(ticks)
        result.total_markets = len(markets)

        if verbose:
            logger.info(f"Processing {len(markets)} markets over {result.data_duration_hours:.1f} hours")

        # State
        balance = self.initial_balance
        trade_id = 0
        open_positions: List[BacktestTrade] = []

        # Process each market
        for market_idx, market in enumerate(markets):
            market_start = market[0]['ts']
            market_end = market_start + (self.market_duration_seconds * 1000)

            # Get strike price (BTC price at market start)
            strike_price = market[0]['price']

            # Reset volatility calculator for new market
            self.volatility = VolatilityCalculator(window_seconds=300, min_samples=30)

            # Process ticks in this market
            for tick in market:
                ts = tick['ts']
                btc_price = tick['price']

                # Update volatility
                vol_metrics = self.volatility.add_price(btc_price, ts)

                if not vol_metrics:
                    continue

                # Get token prices at this time
                token_prices = self._get_prices_at_time(price_index, ts)

                if not token_prices:
                    continue

                # Calculate time remaining
                time_remaining = max(0, (market_end - ts) // 1000)

                if time_remaining < 30:
                    continue  # Too close to expiry

                # Estimate probabilities
                if self.use_momentum and isinstance(self.probability, AdaptiveProbabilityModel):
                    up_est, down_est = self.probability.estimate_with_momentum(
                        current_price=btc_price,
                        strike_price=strike_price,
                        time_remaining_seconds=time_remaining,
                        volatility_per_second=vol_metrics.rolling_std,
                        price_change_1m=vol_metrics.price_change_1m,
                        price_change_5m=vol_metrics.price_change_5m
                    )
                else:
                    up_est, down_est = self.probability.estimate_both(
                        current_price=btc_price,
                        strike_price=strike_price,
                        time_remaining_seconds=time_remaining,
                        volatility_per_second=vol_metrics.rolling_std
                    )

                # Create market prices
                market_prices = MarketPrices(
                    up_price=token_prices['up'],
                    down_price=token_prices['down'],
                    timestamp=ts
                )

                # Calculate edges
                up_edge = (up_est.probability - market_prices.up_price) * 100
                down_edge = (down_est.probability - market_prices.down_price) * 100

                # Track edge history
                result.edge_history.append((ts, up_edge, down_edge))

                best_edge = max(up_edge, down_edge)
                if best_edge > result.max_edge_seen:
                    result.max_edge_seen = best_edge

                if best_edge >= self.min_edge_pct:
                    result.edges_above_threshold += 1

                # Detect tradeable edge
                edge = self.edge_detector.detect_edge(up_est, down_est, market_prices)

                if edge and len(open_positions) == 0:
                    # Calculate position size
                    position_pct = min(
                        self.risk_config.max_position_pct / 100,
                        edge.kelly_fraction * self.risk_config.kelly_fraction
                    )
                    size_usd = min(
                        balance * position_pct,
                        self.risk_config.max_position_usd
                    )

                    if size_usd >= 1.0:
                        # Execute trade
                        direction = "up" if edge.signal == TradeSignal.BUY_UP else "down"
                        entry_price = market_prices.up_price if direction == "up" else market_prices.down_price
                        tokens = size_usd / entry_price

                        trade = BacktestTrade(
                            trade_id=trade_id,
                            timestamp=ts,
                            direction=direction,
                            entry_price=entry_price,
                            size_usd=size_usd,
                            tokens=tokens,
                            btc_price=btc_price,
                            model_prob=edge.model_probability,
                            market_price=edge.market_price,
                            edge_pct=edge.edge_percent,
                            volatility=vol_metrics.rolling_std,
                            time_remaining=time_remaining
                        )

                        open_positions.append(trade)
                        balance -= size_usd
                        trade_id += 1

                        result.avg_edge_taken = (
                            (result.avg_edge_taken * result.total_trades + edge.edge_percent) /
                            (result.total_trades + 1)
                        )

            # Settle positions at market end
            final_btc = market[-1]['price'] if market else strike_price

            for trade in open_positions:
                trade.settled = True
                trade.exit_timestamp = market_end

                # Determine outcome
                if trade.direction == "up":
                    trade.won = final_btc > strike_price
                else:
                    trade.won = final_btc < strike_price

                # Calculate P&L
                if trade.won:
                    payout = trade.tokens * 1.0  # Each token pays $1
                    trade.pnl = payout - trade.size_usd
                    result.wins += 1
                else:
                    trade.pnl = -trade.size_usd
                    result.losses += 1

                balance += trade.size_usd + trade.pnl
                result.total_pnl += trade.pnl
                result.trades.append(trade)
                result.total_trades += 1

                # Track equity
                result.equity_curve.append((market_end, balance))

                # Track max/min balance
                if balance > result.max_balance:
                    result.max_balance = balance
                if balance < result.min_balance:
                    result.min_balance = balance

            open_positions = []

            if verbose and (market_idx + 1) % 10 == 0:
                logger.info(
                    f"Processed {market_idx + 1}/{len(markets)} markets | "
                    f"Balance: ${balance:.2f} | Trades: {result.total_trades}"
                )

        # Final metrics
        result.final_balance = balance
        result.win_rate = result.wins / max(1, result.total_trades) * 100
        result.avg_pnl_per_trade = result.total_pnl / max(1, result.total_trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
        result.profit_factor = gross_profit / max(0.01, gross_loss)

        # Max drawdown
        if result.max_balance > 0:
            result.max_drawdown_pct = (result.max_balance - result.min_balance) / result.max_balance * 100

        # Sharpe ratio (simplified)
        if result.trades:
            returns = [t.pnl / t.size_usd for t in result.trades if t.size_usd > 0]
            if len(returns) > 1:
                avg_return = sum(returns) / len(returns)
                std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1))
                if std_return > 0:
                    # Annualized (assuming ~100 trades/day)
                    result.sharpe_ratio = (avg_return / std_return) * math.sqrt(100 * 365)

        return result

    def _build_price_index(self, prices: List[Dict]) -> Dict[int, Dict]:
        """Build timestamp -> prices lookup."""
        index = {}
        for p in prices:
            ts = p.get('ts', 0)
            index[ts] = {'up': p.get('up', 0.5), 'down': p.get('down', 0.5)}
        return index

    def _get_prices_at_time(self, index: Dict, ts: int, tolerance_ms: int = 5000) -> Optional[Dict]:
        """Get prices closest to timestamp."""
        if ts in index:
            return index[ts]

        # Find closest timestamp
        closest_ts = None
        min_diff = float('inf')

        for t in index.keys():
            diff = abs(t - ts)
            if diff < min_diff and diff <= tolerance_ms:
                min_diff = diff
                closest_ts = t

        if closest_ts:
            return index[closest_ts]
        return None

    def _segment_markets(self, ticks: List[Dict]) -> List[List[Dict]]:
        """Segment ticks into 15-minute markets."""
        if not ticks:
            return []

        markets = []
        current_market = []
        market_start = ticks[0]['ts']

        for tick in ticks:
            ts = tick['ts']

            # Check if new market
            if ts >= market_start + (self.market_duration_seconds * 1000):
                if current_market:
                    markets.append(current_market)
                current_market = [tick]
                market_start = ts
            else:
                current_market.append(tick)

        if current_market:
            markets.append(current_market)

        return markets

    def print_results(self, result: BacktestResult):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("VOLATILITY ARBITRAGE BACKTEST RESULTS")
        print("=" * 60)

        print(f"\n--- Data ---")
        print(f"Total Ticks:     {result.total_ticks:,}")
        print(f"Total Markets:   {result.total_markets}")
        print(f"Duration:        {result.data_duration_hours:.1f} hours")

        print(f"\n--- Performance ---")
        print(f"Initial Balance: ${result.initial_balance:,.2f}")
        print(f"Final Balance:   ${result.final_balance:,.2f}")
        print(f"Total P&L:       ${result.total_pnl:+,.2f}")
        print(f"Return:          {(result.final_balance/result.initial_balance - 1)*100:+.2f}%")

        print(f"\n--- Trades ---")
        print(f"Total Trades:    {result.total_trades}")
        print(f"Wins:            {result.wins}")
        print(f"Losses:          {result.losses}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"Avg P&L/Trade:   ${result.avg_pnl_per_trade:+.2f}")

        print(f"\n--- Risk Metrics ---")
        print(f"Profit Factor:   {result.profit_factor:.2f}")
        print(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown:    {result.max_drawdown_pct:.1f}%")

        print(f"\n--- Edge Analysis ---")
        print(f"Avg Edge Taken:  {result.avg_edge_taken:.2f}%")
        print(f"Max Edge Seen:   {result.max_edge_seen:.2f}%")
        print(f"Edges > {self.min_edge_pct}%:    {result.edges_above_threshold}")

        print("=" * 60)

        # Trade details
        if result.trades:
            print(f"\n--- Recent Trades ---")
            for trade in result.trades[-5:]:
                outcome = "WIN" if trade.won else "LOSS"
                print(
                    f"  {trade.direction.upper():5} | "
                    f"Edge: {trade.edge_pct:+.1f}% | "
                    f"P&L: ${trade.pnl:+.2f} | "
                    f"{outcome}"
                )


def run_volatility_backtest(
    data_path: str,
    initial_balance: float = 1000.0,
    min_edge_pct: float = 3.0,
    verbose: bool = True
) -> BacktestResult:
    """
    Run volatility arbitrage backtest.

    Args:
        data_path: Path to data file or directory
        initial_balance: Starting capital
        min_edge_pct: Minimum edge to trade
        verbose: Print progress

    Returns:
        BacktestResult
    """
    backtest = VolatilityArbBacktest(
        initial_balance=initial_balance,
        min_edge_pct=min_edge_pct
    )

    data = backtest.load_data(data_path)
    result = backtest.run(data, verbose=verbose)

    if verbose:
        backtest.print_results(result)

    return result
