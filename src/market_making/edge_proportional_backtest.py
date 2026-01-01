"""
Proporção Ajustada (Edge-Based) Backtest

Strategy:
- Use probability model to estimate true probabilities
- Buy BOTH sides but in proportions based on model
- If model says 60% UP, buy 60% UP and 40% DOWN
- Partial hedge while capturing edge

Example:
- Model: P(UP) = 60%, P(DOWN) = 40%
- Market: UP = $0.50, DOWN = $0.50
- Edge = 60% - 50% = 10% on UP side
- Buy $60 UP, $40 DOWN (total $100)
- If UP wins: $60 * (1/$0.50) = 120 tokens * $1 = $120
- If DOWN wins: $40 * (1/$0.50) = 80 tokens * $1 = $80
- Expected: 60% * $120 + 40% * $80 = $72 + $32 = $104
- Profit: $4 (4% edge capture)
"""

import json
import gzip
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

from ..volatility_arb.volatility import VolatilityCalculator
from ..volatility_arb.probability import ProbabilityModel, AdaptiveProbabilityModel

logger = logging.getLogger(__name__)


class OrderType:
    """Order type for fee calculation."""
    MAKER = "maker"  # 0% fee
    TAKER = "taker"  # 2% fee


@dataclass
class EdgeProportionalTrade:
    """A proportional trade (both UP and DOWN)."""
    trade_id: int
    timestamp: int

    # Position sizes
    up_size_usd: float
    down_size_usd: float
    total_size_usd: float

    # Prices
    up_price: float
    down_price: float

    # Tokens bought
    up_tokens: float
    down_tokens: float

    # Model estimates
    model_up_prob: float
    model_down_prob: float

    # Edge
    up_edge: float
    down_edge: float

    # BTC info
    btc_price: float
    strike_price: float
    volatility: float
    time_remaining: int

    # Fees
    fee_paid: float = 0.0
    order_type: str = "maker"

    # Settlement
    settled: bool = False
    outcome: Optional[str] = None  # "up" or "down"
    payout: float = 0.0
    pnl: float = 0.0


@dataclass
class EdgeProportionalResult:
    """Results from edge proportional backtest."""
    # Summary
    total_trades: int = 0
    total_up_exposure: float = 0.0
    total_down_exposure: float = 0.0

    # P&L
    total_pnl: float = 0.0
    total_fees: float = 0.0
    gross_profit: float = 0.0

    # Outcomes
    up_wins: int = 0
    down_wins: int = 0

    # Capital
    initial_balance: float = 1000.0
    final_balance: float = 1000.0
    max_balance: float = 1000.0
    min_balance: float = 1000.0
    max_drawdown_pct: float = 0.0

    # Metrics
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_edge_captured: float = 0.0
    avg_up_allocation: float = 0.0

    # Fee tracking
    order_type: str = "maker"
    fee_rate: float = 0.0

    # Data stats
    total_ticks: int = 0
    total_markets: int = 0
    data_duration_hours: float = 0.0

    # Trade list
    trades: List[EdgeProportionalTrade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)


class EdgeProportionalBacktest:
    """
    Edge-Based Proportional strategy backtest.

    Buys both sides in proportions based on probability model.
    Provides partial hedge while capturing edge.
    """

    # Fee rates
    MAKER_FEE = 0.0
    TAKER_FEE = 0.02

    def __init__(
        self,
        initial_balance: float = 1000.0,
        min_edge_pct: float = 3.0,  # Minimum edge to trade
        trade_size_pct: float = 10.0,  # % of balance per trade
        max_trade_size: float = 100.0,  # Max USD per trade
        use_momentum: bool = True,
        order_type: str = "maker",
        custom_fee: Optional[float] = None,
        market_duration_seconds: int = 900,
    ):
        self.initial_balance = initial_balance
        self.min_edge_pct = min_edge_pct
        self.trade_size_pct = trade_size_pct
        self.max_trade_size = max_trade_size
        self.use_momentum = use_momentum
        self.market_duration_seconds = market_duration_seconds
        self.order_type = order_type

        # Set fee rate
        if custom_fee is not None:
            self.fee_rate = custom_fee
        elif order_type == OrderType.MAKER:
            self.fee_rate = self.MAKER_FEE
        else:
            self.fee_rate = self.TAKER_FEE

        # Probability model
        if use_momentum:
            self.probability = AdaptiveProbabilityModel()
        else:
            self.probability = ProbabilityModel()

    def load_data(self, data_path: str) -> Dict[str, Any]:
        """Load data from file or directory."""
        path = Path(data_path)

        all_ticks = []
        all_prices = []

        if path.is_dir():
            files = sorted(path.glob("*.json.gz")) + sorted(path.glob("*.json"))
            logger.info(f"Found {len(files)} data files")

            for file in files:
                data = self._load_file(file)
                if data:
                    all_ticks.extend(data.get('chainlink_ticks', []))
                    all_prices.extend(data.get('price_changes', []))
        else:
            data = self._load_file(path)
            if data:
                all_ticks = data.get('chainlink_ticks', [])
                all_prices = data.get('price_changes', [])

        all_ticks.sort(key=lambda x: x.get('ts', 0))
        all_prices.sort(key=lambda x: x.get('ts', 0))

        logger.info(f"Loaded {len(all_ticks)} ticks, {len(all_prices)} price changes")

        return {
            'chainlink_ticks': all_ticks,
            'price_changes': all_prices,
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
    ) -> EdgeProportionalResult:
        """
        Run edge proportional backtest.

        Buys both UP and DOWN in proportions based on probability model.
        """
        result = EdgeProportionalResult(initial_balance=self.initial_balance)

        ticks = data.get('chainlink_ticks', [])
        prices = data.get('price_changes', [])

        if not ticks:
            logger.error("No tick data provided")
            return result

        result.total_ticks = len(ticks)

        # Calculate duration
        if len(ticks) >= 2:
            duration_ms = ticks[-1]['ts'] - ticks[0]['ts']
            result.data_duration_hours = duration_ms / 1000 / 3600

        # Build price index
        price_index = self._build_price_index(prices)

        # Segment into markets
        markets = self._segment_markets(ticks)
        result.total_markets = len(markets)

        if verbose:
            logger.info(f"Processing {len(markets)} markets")

        # State
        balance = self.initial_balance
        trade_id = 0
        open_positions: List[EdgeProportionalTrade] = []

        # Process each market
        for market_idx, market in enumerate(markets):
            market_start = market[0]['ts']
            market_end = market_start + (self.market_duration_seconds * 1000)
            strike_price = market[0]['price']

            # Reset volatility calculator
            volatility = VolatilityCalculator(window_seconds=300, min_samples=30)

            # Process ticks
            for tick in market:
                ts = tick['ts']
                btc_price = tick['price']

                # Update volatility
                vol_metrics = volatility.add_price(btc_price, ts)

                if not vol_metrics:
                    continue

                # Get token prices
                token_prices = self._get_prices_at_time(price_index, ts)
                if not token_prices:
                    continue

                # Calculate time remaining
                time_remaining = max(0, (market_end - ts) // 1000)

                if time_remaining < 60:  # Too close to expiry
                    continue

                # Only trade once per market
                if open_positions:
                    continue

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

                model_up = up_est.probability
                model_down = down_est.probability

                up_price = token_prices['up']
                down_price = token_prices['down']

                # Calculate edges
                up_edge = (model_up - up_price) * 100
                down_edge = (model_down - down_price) * 100

                # Check if we have edge on either side
                max_edge = max(up_edge, down_edge)

                if max_edge < self.min_edge_pct:
                    continue

                # Calculate trade size
                trade_size = min(
                    balance * (self.trade_size_pct / 100),
                    self.max_trade_size
                )

                if trade_size < 2.0:  # Minimum $2 trade
                    continue

                # Calculate proportional allocation
                # Allocate based on model probabilities
                up_allocation = model_up
                down_allocation = model_down

                # Ensure they sum to 1
                total_prob = up_allocation + down_allocation
                if total_prob > 0:
                    up_allocation /= total_prob
                    down_allocation /= total_prob

                up_size = trade_size * up_allocation
                down_size = trade_size * down_allocation

                # Calculate fees
                fee = trade_size * self.fee_rate
                effective_trade_size = trade_size - fee

                # Recalculate sizes after fee
                up_size = effective_trade_size * up_allocation
                down_size = effective_trade_size * down_allocation

                # Calculate tokens
                up_tokens = up_size / up_price
                down_tokens = down_size / down_price

                # Create trade
                trade = EdgeProportionalTrade(
                    trade_id=trade_id,
                    timestamp=ts,
                    up_size_usd=up_size,
                    down_size_usd=down_size,
                    total_size_usd=trade_size,
                    up_price=up_price,
                    down_price=down_price,
                    up_tokens=up_tokens,
                    down_tokens=down_tokens,
                    model_up_prob=model_up,
                    model_down_prob=model_down,
                    up_edge=up_edge,
                    down_edge=down_edge,
                    btc_price=btc_price,
                    strike_price=strike_price,
                    volatility=vol_metrics.rolling_std,
                    time_remaining=time_remaining,
                    fee_paid=fee,
                    order_type=self.order_type,
                )

                open_positions.append(trade)
                balance -= trade_size
                result.total_fees += fee
                result.total_up_exposure += up_size
                result.total_down_exposure += down_size
                trade_id += 1

            # Settle positions at market end
            final_btc = market[-1]['price'] if market else strike_price

            for trade in open_positions:
                trade.settled = True

                # Determine outcome
                if final_btc > trade.strike_price:
                    trade.outcome = "up"
                    trade.payout = trade.up_tokens * 1.0  # UP tokens pay $1 each
                    result.up_wins += 1
                else:
                    trade.outcome = "down"
                    trade.payout = trade.down_tokens * 1.0  # DOWN tokens pay $1 each
                    result.down_wins += 1

                trade.pnl = trade.payout - trade.total_size_usd
                balance += trade.payout
                result.total_pnl += trade.pnl
                result.trades.append(trade)
                result.total_trades += 1

                # Track equity
                result.equity_curve.append((market_end, balance))

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
        result.order_type = self.order_type
        result.fee_rate = self.fee_rate
        result.gross_profit = result.total_pnl + result.total_fees

        if result.total_trades > 0:
            result.win_rate = sum(1 for t in result.trades if t.pnl > 0) / result.total_trades * 100
            result.avg_pnl_per_trade = result.total_pnl / result.total_trades
            result.avg_edge_captured = sum(max(t.up_edge, t.down_edge) for t in result.trades) / result.total_trades
            result.avg_up_allocation = sum(t.up_size_usd / t.total_size_usd for t in result.trades) / result.total_trades * 100

        if result.max_balance > 0:
            result.max_drawdown_pct = (result.max_balance - result.min_balance) / result.max_balance * 100

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

    def print_results(self, result: EdgeProportionalResult):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("EDGE-BASED PROPORTIONAL BACKTEST RESULTS")
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
        print(f"UP Wins:         {result.up_wins}")
        print(f"DOWN Wins:       {result.down_wins}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"Avg P&L/Trade:   ${result.avg_pnl_per_trade:+.2f}")

        print(f"\n--- Allocation ---")
        print(f"Total UP Exp:    ${result.total_up_exposure:,.2f}")
        print(f"Total DOWN Exp:  ${result.total_down_exposure:,.2f}")
        print(f"Avg UP Alloc:    {result.avg_up_allocation:.1f}%")
        print(f"Avg Edge:        {result.avg_edge_captured:.2f}%")

        print(f"\n--- Risk ---")
        print(f"Max Drawdown:    {result.max_drawdown_pct:.1f}%")

        print(f"\n--- Fees ---")
        print(f"Order Type:      {result.order_type.upper()}")
        print(f"Fee Rate:        {result.fee_rate * 100:.1f}%")
        print(f"Total Fees:      ${result.total_fees:.2f}")

        print("=" * 60)

        # Sample trades
        if result.trades:
            print(f"\n--- Sample Trades ---")
            for trade in result.trades[-5:]:
                print(
                    f"  UP: {trade.model_up_prob*100:.0f}% @ ${trade.up_price:.2f} | "
                    f"DOWN: {trade.model_down_prob*100:.0f}% @ ${trade.down_price:.2f} | "
                    f"Outcome: {trade.outcome.upper()} | P&L: ${trade.pnl:+.2f}"
                )


def run_edge_proportional_backtest(
    data_path: str,
    initial_balance: float = 1000.0,
    min_edge_pct: float = 3.0,
    trade_size_pct: float = 10.0,
    order_type: str = "maker",
    custom_fee: Optional[float] = None,
    verbose: bool = True
) -> EdgeProportionalResult:
    """
    Run edge-based proportional backtest.

    Args:
        data_path: Path to data file or directory
        initial_balance: Starting capital
        min_edge_pct: Minimum edge to trade
        trade_size_pct: % of balance per trade
        order_type: "maker" (0% fee) or "taker" (2% fee)
        custom_fee: Optional custom fee rate
        verbose: Print progress

    Returns:
        EdgeProportionalResult
    """
    backtest = EdgeProportionalBacktest(
        initial_balance=initial_balance,
        min_edge_pct=min_edge_pct,
        trade_size_pct=trade_size_pct,
        order_type=order_type,
        custom_fee=custom_fee
    )

    data = backtest.load_data(data_path)
    result = backtest.run(data, verbose=verbose)

    if verbose:
        backtest.print_results(result)

    return result
