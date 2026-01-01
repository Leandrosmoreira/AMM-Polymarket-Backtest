"""
Scalp com Hedge (Market Making) Backtest

Strategy:
- Place LIMIT orders on both sides (UP and DOWN)
- Buy at bid price, sell at ask price
- Capture bid-ask spread
- Uses MAKER orders (0% fee)

Example:
- Post buy limit for UP at $0.47
- Post buy limit for DOWN at $0.47
- When filled, post sell limits at $0.49
- Profit = spread captured
"""

import json
import gzip
import math
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OrderType:
    """Order type for fee calculation."""
    MAKER = "maker"  # Limit order - 0% fee
    TAKER = "taker"  # Market order - ~2% fee


class OrderSide:
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus:
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class LimitOrder:
    """A limit order in the order book."""
    order_id: int
    timestamp: int
    token: str  # "up" or "down"
    side: str  # "buy" or "sell"
    price: float
    size: float
    status: str = OrderStatus.PENDING
    fill_timestamp: Optional[int] = None
    fill_price: Optional[float] = None


@dataclass
class ScalpTrade:
    """A completed scalp trade (buy + sell)."""
    trade_id: int
    token: str
    buy_price: float
    sell_price: float
    size: float
    spread_captured: float
    pnl: float
    buy_timestamp: int
    sell_timestamp: int
    hold_time_ms: int


@dataclass
class ScalpResult:
    """Results from scalp backtest."""
    # Summary
    total_orders: int = 0
    filled_buys: int = 0
    filled_sells: int = 0
    completed_trades: int = 0
    cancelled_orders: int = 0

    # P&L
    total_pnl: float = 0.0
    total_fees: float = 0.0
    gross_profit: float = 0.0

    # Capital
    initial_balance: float = 1000.0
    final_balance: float = 1000.0
    max_balance: float = 1000.0
    min_balance: float = 1000.0
    max_drawdown_pct: float = 0.0

    # Metrics
    win_rate: float = 0.0
    avg_spread_captured: float = 0.0
    avg_hold_time_ms: float = 0.0
    fill_rate: float = 0.0

    # Fee tracking
    order_type: str = "maker"
    fee_rate: float = 0.0

    # Data stats
    total_ticks: int = 0
    total_markets: int = 0
    data_duration_hours: float = 0.0

    # Trade list
    trades: List[ScalpTrade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)


class ScalpBacktest:
    """
    Market Making / Scalp strategy backtest.

    Places limit orders on both sides to capture bid-ask spread.
    Uses MAKER orders (0% fee) for profitability.
    """

    # Fee rates
    MAKER_FEE = 0.0   # 0% for limit orders
    TAKER_FEE = 0.02  # 2% for market orders

    def __init__(
        self,
        initial_balance: float = 1000.0,
        spread_target: float = 0.02,  # Target 2% spread capture
        order_size: float = 10.0,  # USD per order
        max_open_orders: int = 4,  # Max simultaneous orders
        fill_probability: float = 0.3,  # Probability of limit order fill
        order_type: str = "maker",
        custom_fee: Optional[float] = None,
        market_duration_seconds: int = 900,
    ):
        self.initial_balance = initial_balance
        self.spread_target = spread_target
        self.order_size = order_size
        self.max_open_orders = max_open_orders
        self.fill_probability = fill_probability
        self.market_duration_seconds = market_duration_seconds
        self.order_type = order_type

        # Set fee rate
        if custom_fee is not None:
            self.fee_rate = custom_fee
        elif order_type == OrderType.MAKER:
            self.fee_rate = self.MAKER_FEE
        else:
            self.fee_rate = self.TAKER_FEE

    def load_data(self, data_path: str) -> Dict[str, Any]:
        """Load data from file or directory."""
        path = Path(data_path)

        all_ticks = []
        all_prices = []
        all_books = []

        if path.is_dir():
            files = sorted(path.glob("*.json.gz")) + sorted(path.glob("*.json"))
            logger.info(f"Found {len(files)} data files")

            for file in files:
                data = self._load_file(file)
                if data:
                    all_ticks.extend(data.get('chainlink_ticks', []))
                    all_prices.extend(data.get('price_changes', []))
                    all_books.extend(data.get('order_books', []))
        else:
            data = self._load_file(path)
            if data:
                all_ticks = data.get('chainlink_ticks', [])
                all_prices = data.get('price_changes', [])
                all_books = data.get('order_books', [])

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
    ) -> ScalpResult:
        """
        Run scalp backtest.

        Simulates placing limit orders on both sides and capturing spread.
        """
        result = ScalpResult(initial_balance=self.initial_balance)

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
        order_id = 0

        open_buy_orders: List[LimitOrder] = []
        open_sell_orders: List[LimitOrder] = []
        held_positions: Dict[str, float] = {"up": 0, "down": 0}  # tokens held
        held_cost: Dict[str, float] = {"up": 0, "down": 0}  # cost basis

        # Process each market
        for market_idx, market in enumerate(markets):
            market_start = market[0]['ts']
            market_end = market_start + (self.market_duration_seconds * 1000)
            strike_price = market[0]['price']

            # Reset orders for new market
            open_buy_orders = []
            open_sell_orders = []
            held_positions = {"up": 0, "down": 0}
            held_cost = {"up": 0, "down": 0}

            for tick in market:
                ts = tick['ts']
                btc_price = tick['price']

                # Get current market prices
                token_prices = self._get_prices_at_time(price_index, ts)
                if not token_prices:
                    continue

                up_price = token_prices['up']
                down_price = token_prices['down']

                # Simulate bid-ask spread (market prices are mid-prices)
                # Bid is slightly below, ask is slightly above
                spread = 0.02  # 2% typical spread
                up_bid = up_price * (1 - spread / 2)
                up_ask = up_price * (1 + spread / 2)
                down_bid = down_price * (1 - spread / 2)
                down_ask = down_price * (1 + spread / 2)

                # Check for order fills
                filled_buys = []
                for order in open_buy_orders:
                    # Buy order fills if price drops to our bid
                    current_ask = up_ask if order.token == "up" else down_ask
                    if order.price >= current_ask * 0.98:  # Fill if close enough
                        if random.random() < self.fill_probability:
                            order.status = OrderStatus.FILLED
                            order.fill_timestamp = ts
                            order.fill_price = order.price
                            filled_buys.append(order)
                            result.filled_buys += 1

                            # Update held position
                            tokens = order.size / order.price
                            held_positions[order.token] += tokens
                            held_cost[order.token] += order.size

                # Remove filled orders
                open_buy_orders = [o for o in open_buy_orders if o.status == OrderStatus.PENDING]

                # Check sell order fills
                filled_sells = []
                for order in open_sell_orders:
                    current_bid = up_bid if order.token == "up" else down_bid
                    if order.price <= current_bid * 1.02:
                        if random.random() < self.fill_probability:
                            order.status = OrderStatus.FILLED
                            order.fill_timestamp = ts
                            order.fill_price = order.price
                            filled_sells.append(order)
                            result.filled_sells += 1

                # Remove filled sell orders
                open_sell_orders = [o for o in open_sell_orders if o.status == OrderStatus.PENDING]

                # Process filled sells - complete trades
                for sell_order in filled_sells:
                    token = sell_order.token
                    sell_tokens = sell_order.size / sell_order.price

                    if held_positions[token] >= sell_tokens:
                        # Calculate cost basis for sold tokens
                        avg_cost = held_cost[token] / held_positions[token] if held_positions[token] > 0 else 0
                        cost = sell_tokens * avg_cost

                        # Calculate P&L
                        revenue = sell_order.size
                        fee = revenue * self.fee_rate
                        pnl = revenue - cost - fee

                        # Update positions
                        held_positions[token] -= sell_tokens
                        held_cost[token] -= cost

                        # Record trade
                        trade = ScalpTrade(
                            trade_id=trade_id,
                            token=token,
                            buy_price=avg_cost / sell_tokens if sell_tokens > 0 else 0,
                            sell_price=sell_order.price,
                            size=sell_order.size,
                            spread_captured=(sell_order.price - avg_cost / sell_tokens) if sell_tokens > 0 else 0,
                            pnl=pnl,
                            buy_timestamp=0,  # Averaged
                            sell_timestamp=ts,
                            hold_time_ms=0,
                        )

                        result.trades.append(trade)
                        result.total_pnl += pnl
                        result.total_fees += fee
                        balance += pnl
                        trade_id += 1
                        result.completed_trades += 1

                # Place new orders if we have capacity
                total_open = len(open_buy_orders) + len(open_sell_orders)

                if total_open < self.max_open_orders and balance >= self.order_size:
                    # Place buy orders on both sides
                    for token in ["up", "down"]:
                        if len(open_buy_orders) < self.max_open_orders // 2:
                            bid_price = up_bid if token == "up" else down_bid

                            order = LimitOrder(
                                order_id=order_id,
                                timestamp=ts,
                                token=token,
                                side=OrderSide.BUY,
                                price=bid_price,
                                size=self.order_size,
                            )
                            open_buy_orders.append(order)
                            order_id += 1
                            result.total_orders += 1

                # Place sell orders for held positions
                for token in ["up", "down"]:
                    if held_positions[token] > 0:
                        ask_price = up_ask if token == "up" else down_ask
                        sell_size = min(held_positions[token] * ask_price, self.order_size)

                        if sell_size >= 1.0:
                            # Check if we already have a sell order for this token
                            has_sell = any(o.token == token for o in open_sell_orders)
                            if not has_sell:
                                order = LimitOrder(
                                    order_id=order_id,
                                    timestamp=ts,
                                    token=token,
                                    side=OrderSide.SELL,
                                    price=ask_price,
                                    size=sell_size,
                                )
                                open_sell_orders.append(order)
                                order_id += 1
                                result.total_orders += 1

                # Track equity
                result.equity_curve.append((ts, balance))

                if balance > result.max_balance:
                    result.max_balance = balance
                if balance < result.min_balance:
                    result.min_balance = balance

            # End of market - settle any remaining positions at settlement
            final_btc = market[-1]['price'] if market else strike_price
            for token in ["up", "down"]:
                if held_positions[token] > 0:
                    # Determine settlement value
                    if token == "up":
                        won = final_btc > strike_price
                    else:
                        won = final_btc < strike_price

                    payout = held_positions[token] * (1.0 if won else 0.0)
                    cost = held_cost[token]
                    pnl = payout - cost

                    balance += pnl
                    result.total_pnl += pnl

            # Cancel remaining orders
            result.cancelled_orders += len(open_buy_orders) + len(open_sell_orders)

            if verbose and (market_idx + 1) % 10 == 0:
                logger.info(
                    f"Processed {market_idx + 1}/{len(markets)} markets | "
                    f"Balance: ${balance:.2f} | Trades: {result.completed_trades}"
                )

        # Final metrics
        result.final_balance = balance
        result.order_type = self.order_type
        result.fee_rate = self.fee_rate

        if result.total_orders > 0:
            result.fill_rate = (result.filled_buys + result.filled_sells) / result.total_orders * 100

        if result.completed_trades > 0:
            result.win_rate = sum(1 for t in result.trades if t.pnl > 0) / result.completed_trades * 100
            result.avg_spread_captured = sum(t.spread_captured for t in result.trades) / result.completed_trades

        if result.max_balance > 0:
            result.max_drawdown_pct = (result.max_balance - result.min_balance) / result.max_balance * 100

        result.gross_profit = result.total_pnl + result.total_fees

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

    def print_results(self, result: ScalpResult):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("SCALP / MARKET MAKING BACKTEST RESULTS")
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

        print(f"\n--- Orders ---")
        print(f"Total Orders:    {result.total_orders}")
        print(f"Filled Buys:     {result.filled_buys}")
        print(f"Filled Sells:    {result.filled_sells}")
        print(f"Cancelled:       {result.cancelled_orders}")
        print(f"Fill Rate:       {result.fill_rate:.1f}%")

        print(f"\n--- Trades ---")
        print(f"Completed:       {result.completed_trades}")
        print(f"Win Rate:        {result.win_rate:.1f}%")
        print(f"Avg Spread:      {result.avg_spread_captured:.4f}")

        print(f"\n--- Risk ---")
        print(f"Max Drawdown:    {result.max_drawdown_pct:.1f}%")

        print(f"\n--- Fees ---")
        print(f"Order Type:      {result.order_type.upper()}")
        print(f"Fee Rate:        {result.fee_rate * 100:.1f}%")
        print(f"Total Fees:      ${result.total_fees:.2f}")

        print("=" * 60)


def run_scalp_backtest(
    data_path: str,
    initial_balance: float = 1000.0,
    spread_target: float = 0.02,
    order_size: float = 10.0,
    order_type: str = "maker",
    custom_fee: Optional[float] = None,
    verbose: bool = True
) -> ScalpResult:
    """
    Run scalp/market making backtest.

    Args:
        data_path: Path to data file or directory
        initial_balance: Starting capital
        spread_target: Target spread to capture
        order_size: Size per order in USD
        order_type: "maker" (0% fee) or "taker" (2% fee)
        custom_fee: Optional custom fee rate
        verbose: Print progress

    Returns:
        ScalpResult
    """
    backtest = ScalpBacktest(
        initial_balance=initial_balance,
        spread_target=spread_target,
        order_size=order_size,
        order_type=order_type,
        custom_fee=custom_fee
    )

    data = backtest.load_data(data_path)
    result = backtest.run(data, verbose=verbose)

    if verbose:
        backtest.print_results(result)

    return result
