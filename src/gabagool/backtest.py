"""
Gabagool Spread Capture Strategy Backtest Engine
Simulates the spread capture strategy with historical or generated data
"""

import random
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .config import GabagoolConfig
from .position_manager import GabagoolPositionManager, PositionSide, MarketPosition

logger = logging.getLogger(__name__)


class OrderType:
    """Order type for fee calculation."""
    MAKER = "maker"  # Limit order - 0% fee
    TAKER = "taker"  # Market order - ~2% fee


@dataclass
class SimulatedMarket:
    """Simulated market for backtesting."""
    id: str
    slug: str
    asset: str
    start_time: int
    end_time: int
    outcome: str  # "UP" or "DOWN" - final result

    # Price simulation parameters
    base_up_prob: float = 0.50  # Starting probability
    volatility: float = 0.02    # Price volatility

    # Simulated prices over time
    price_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: str
    end_date: str
    total_markets: int
    markets_traded: int
    total_trades: int
    opportunities_seen: int
    opportunities_taken: int

    # P&L
    total_cost: float
    total_payout: float
    gross_profit: float
    fees_paid: float
    net_profit: float

    # Performance
    win_rate: float
    avg_profit_per_market: float
    avg_spread_captured: float
    roi_pct: float

    # Risk metrics
    max_drawdown: float
    max_exposure: float
    sharpe_ratio: float

    # Fee tracking
    order_type: str = "maker"
    fee_rate: float = 0.0

    # Details
    markets: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_markets': self.total_markets,
            'markets_traded': self.markets_traded,
            'total_trades': self.total_trades,
            'opportunities_seen': self.opportunities_seen,
            'opportunities_taken': self.opportunities_taken,
            'total_cost': self.total_cost,
            'total_payout': self.total_payout,
            'gross_profit': self.gross_profit,
            'fees_paid': self.fees_paid,
            'net_profit': self.net_profit,
            'win_rate': self.win_rate,
            'avg_profit_per_market': self.avg_profit_per_market,
            'avg_spread_captured': self.avg_spread_captured,
            'roi_pct': self.roi_pct,
            'max_drawdown': self.max_drawdown,
            'max_exposure': self.max_exposure,
            'sharpe_ratio': self.sharpe_ratio,
            'order_type': self.order_type,
            'fee_rate': self.fee_rate,
        }


class SpreadSimulator:
    """
    Simulates realistic spread behavior in Polymarket Up/Down markets.

    Based on observations:
    - Early in market: Spreads tend to be wider (more opportunity)
    - As market matures: Spreads tighten
    - Near expiry: Spreads can widen again but with less liquidity
    - Spreads correlate with market uncertainty
    """

    def __init__(
        self,
        base_spread: float = 0.03,
        spread_volatility: float = 0.01,
        early_premium: float = 0.02,
        late_premium: float = 0.01,
    ):
        self.base_spread = base_spread
        self.spread_volatility = spread_volatility
        self.early_premium = early_premium
        self.late_premium = late_premium

    def generate_prices(
        self,
        up_prob: float,
        time_elapsed_pct: float,
        noise_factor: float = 1.0,
    ) -> Tuple[float, float, float]:
        """
        Generate realistic UP and DOWN prices.

        Args:
            up_prob: True probability of UP (0-1)
            time_elapsed_pct: How much of market duration has passed (0-1)
            noise_factor: Random noise multiplier

        Returns:
            Tuple of (up_price, down_price, spread_pct)
        """
        # Base prices from probability
        base_up = up_prob
        base_down = 1.0 - up_prob

        # Calculate spread based on time
        if time_elapsed_pct < 0.2:  # Early market
            time_spread = self.base_spread + self.early_premium * (1 - time_elapsed_pct * 5)
        elif time_elapsed_pct > 0.9:  # Late market
            time_spread = self.base_spread + self.late_premium * (time_elapsed_pct - 0.9) * 10
        else:  # Middle market
            time_spread = self.base_spread

        # Add random noise
        noise = random.gauss(0, self.spread_volatility * noise_factor)
        actual_spread = max(0, time_spread + noise)

        # Distribute spread between UP and DOWN (asymmetric based on probability)
        up_discount = actual_spread * (1 - up_prob)
        down_discount = actual_spread * up_prob

        up_price = max(0.01, min(0.99, base_up - up_discount / 2 + random.gauss(0, 0.005)))
        down_price = max(0.01, min(0.99, base_down - down_discount / 2 + random.gauss(0, 0.005)))

        # Ensure total is less than 1 (opportunity exists)
        total = up_price + down_price
        if total >= 1.0:
            # Scale down to create opportunity
            scale = random.uniform(0.95, 0.99)
            up_price *= scale
            down_price *= scale
            total = up_price + down_price

        spread_pct = (1.0 - total) / total if total > 0 else 0

        return up_price, down_price, spread_pct


class GabagoolBacktest:
    """
    Backtests the Gabagool spread capture strategy.

    Simulation approach:
    1. Generate N markets over a time period
    2. For each market, simulate price evolution
    3. Apply strategy rules to decide when to enter
    4. Track positions and calculate P&L at settlement
    """

    # Fee rates by order type
    MAKER_FEE = 0.0   # 0% for limit orders
    TAKER_FEE = 0.02  # 2% for market orders

    def __init__(
        self,
        config: GabagoolConfig = None,
        order_type: str = "maker",  # "maker" or "taker"
        custom_fee: Optional[float] = None,  # Override fee rate
    ):
        self.config = config or GabagoolConfig()
        self.order_type = order_type

        # Set fee rate based on order type
        if custom_fee is not None:
            self.fee_rate = custom_fee
        elif order_type == OrderType.MAKER:
            self.fee_rate = self.MAKER_FEE
        else:
            self.fee_rate = self.TAKER_FEE

        self.simulator = SpreadSimulator()

        # State
        self.positions = GabagoolPositionManager(config=self.config)
        self.markets: List[SimulatedMarket] = []
        self.opportunities_seen = 0
        self.opportunities_taken = 0
        self.all_trades: List[Dict[str, Any]] = []

        # Tracking
        self._equity_curve: List[float] = []
        self._exposure_curve: List[float] = []

    def generate_markets(
        self,
        num_markets: int,
        start_date: datetime,
        markets_per_hour: int = 4,  # 15-min markets = 4 per hour
        up_win_rate: float = 0.50,  # True probability of UP winning
    ) -> List[SimulatedMarket]:
        """Generate simulated markets for backtesting."""
        markets = []
        current_time = start_date

        for i in range(num_markets):
            # Determine outcome (for settlement)
            outcome = "UP" if random.random() < up_win_rate else "DOWN"

            # Create market
            start_ts = int(current_time.timestamp() * 1000)
            end_ts = start_ts + (self.config.MARKET_DURATION_SECONDS * 1000)

            market = SimulatedMarket(
                id=f"market_{i}",
                slug=f"btc-15min-{i}",
                asset="BTC",
                start_time=start_ts,
                end_time=end_ts,
                outcome=outcome,
                base_up_prob=random.uniform(0.40, 0.60),  # Random starting probability
                volatility=random.uniform(0.01, 0.03),
            )

            # Generate price history
            self._generate_price_history(market)

            markets.append(market)

            # Next market starts after this one
            current_time += timedelta(seconds=self.config.MARKET_DURATION_SECONDS)

        self.markets = markets
        return markets

    def _generate_price_history(self, market: SimulatedMarket):
        """Generate price history for a market."""
        duration_ms = market.end_time - market.start_time
        check_interval = self.config.CHECK_INTERVAL_MS
        num_checks = duration_ms // check_interval

        # Random walk for probability
        prob = market.base_up_prob
        history = []

        for i in range(num_checks):
            time_elapsed_pct = i / num_checks
            timestamp = market.start_time + (i * check_interval)

            # Random walk probability
            prob += random.gauss(0, market.volatility * 0.1)
            prob = max(0.1, min(0.9, prob))  # Keep in reasonable range

            # Generate prices
            up_price, down_price, spread_pct = self.simulator.generate_prices(
                up_prob=prob,
                time_elapsed_pct=time_elapsed_pct,
            )

            # Simulate liquidity (decreases near end)
            base_liquidity = 500
            if time_elapsed_pct > 0.9:
                liquidity = base_liquidity * (1 - (time_elapsed_pct - 0.9) * 5)
            else:
                liquidity = base_liquidity * random.uniform(0.8, 1.2)

            history.append({
                'timestamp': timestamp,
                'time_elapsed_pct': time_elapsed_pct,
                'up_price': up_price,
                'down_price': down_price,
                'total_price': up_price + down_price,
                'spread_pct': spread_pct,
                'up_liquidity': liquidity,
                'down_liquidity': liquidity * random.uniform(0.9, 1.1),
            })

        market.price_history = history

    def run_backtest(
        self,
        num_markets: int = 100,
        start_date: datetime = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            num_markets: Number of markets to simulate
            start_date: Start date for simulation

        Returns:
            BacktestResult with detailed metrics
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)

        logger.info(f"Starting backtest with {num_markets} markets")

        # Generate markets
        self.generate_markets(num_markets, start_date)

        # Reset state
        self.positions = GabagoolPositionManager(config=self.config)
        self.opportunities_seen = 0
        self.opportunities_taken = 0
        self.all_trades = []
        self._equity_curve = []
        self._exposure_curve = []

        markets_traded = 0

        # Process each market
        for market in self.markets:
            traded = self._process_market(market)
            if traded:
                markets_traded += 1

            # Track exposure
            self._exposure_curve.append(self.positions.total_exposure)

        # Settle all positions
        total_payout = 0
        for market in self.markets:
            position = self.positions.get_position(market.id)
            if position and not position.settled:
                payout = self.positions.settle_position(market.id, market.outcome)
                if payout is not None:
                    total_payout += payout

        # Calculate results
        return self._calculate_results(start_date, markets_traded, total_payout)

    def _process_market(self, market: SimulatedMarket) -> bool:
        """Process a single market, looking for opportunities."""
        traded = False
        position = None

        for tick in market.price_history:
            time_elapsed_sec = int(tick['time_elapsed_pct'] * self.config.MARKET_DURATION_SECONDS)

            # Get dynamic threshold
            threshold = self.config.get_entry_threshold(time_elapsed_sec)
            if threshold >= 999:  # Skip period
                continue

            # Check if opportunity exists
            spread_pct = tick['spread_pct']
            if spread_pct >= threshold:
                self.opportunities_seen += 1

                # Check liquidity
                min_liquidity = min(tick['up_liquidity'], tick['down_liquidity'])
                min_liquidity_shares = self.config.MIN_LIQUIDITY_USD / max(tick['up_price'], tick['down_price'])

                if min_liquidity < min_liquidity_shares:
                    continue

                # Check if we can trade
                can_trade, reason = self.positions.can_trade(
                    market.id,
                    self.config.ORDER_SIZE_USD * 2
                )

                if not can_trade:
                    continue

                # Get or create position
                if position is None:
                    position = self.positions.get_or_create_position(
                        market_id=market.id,
                        market_slug=market.slug,
                        up_token_id=f"{market.id}_up",
                        down_token_id=f"{market.id}_down",
                    )

                # Calculate rebalance
                rebalance = self.positions.calculate_rebalance(
                    position=position,
                    budget=self.config.ORDER_SIZE_USD,
                    up_price=tick['up_price'],
                    down_price=tick['down_price'],
                )

                buy_up = rebalance['buy_up']
                buy_down = rebalance['buy_down']

                if buy_up <= 0 and buy_down <= 0:
                    continue

                # Execute trades
                self.opportunities_taken += 1
                traded = True

                if buy_up > 0:
                    self.positions.record_trade(
                        market_id=market.id,
                        side=PositionSide.UP,
                        shares=buy_up,
                        price=tick['up_price'],
                    )
                    self.all_trades.append({
                        'market_id': market.id,
                        'timestamp': tick['timestamp'],
                        'side': 'UP',
                        'shares': buy_up,
                        'price': tick['up_price'],
                        'cost': buy_up * tick['up_price'],
                        'spread_pct': spread_pct,
                    })

                if buy_down > 0:
                    self.positions.record_trade(
                        market_id=market.id,
                        side=PositionSide.DOWN,
                        shares=buy_down,
                        price=tick['down_price'],
                    )
                    self.all_trades.append({
                        'market_id': market.id,
                        'timestamp': tick['timestamp'],
                        'side': 'DOWN',
                        'shares': buy_down,
                        'price': tick['down_price'],
                        'cost': buy_down * tick['down_price'],
                        'spread_pct': spread_pct,
                    })

        return traded

    def _calculate_results(
        self,
        start_date: datetime,
        markets_traded: int,
        total_payout: float,
    ) -> BacktestResult:
        """Calculate final backtest results."""
        summary = self.positions.get_summary()

        total_cost = sum(t['cost'] for t in self.all_trades)
        fees_paid = total_cost * self.fee_rate
        gross_profit = total_payout - total_cost
        net_profit = gross_profit - fees_paid

        # Calculate average spread captured
        avg_spread = 0
        if self.all_trades:
            avg_spread = sum(t['spread_pct'] for t in self.all_trades) / len(self.all_trades)

        # Calculate win rate (positions that made profit)
        wins = sum(1 for p in self.positions.settled_positions if (p.realized_profit or 0) > 0)
        total_settled = len(self.positions.settled_positions)
        win_rate = wins / total_settled if total_settled > 0 else 0

        # ROI
        roi_pct = (net_profit / total_cost * 100) if total_cost > 0 else 0

        # Max drawdown (simplified)
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio (simplified - daily returns)
        sharpe = self._calculate_sharpe_ratio(net_profit, total_cost)

        # Duration
        if self.markets:
            end_date = datetime.fromtimestamp(self.markets[-1].end_time / 1000)
        else:
            end_date = start_date

        return BacktestResult(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            total_markets=len(self.markets),
            markets_traded=markets_traded,
            total_trades=len(self.all_trades),
            opportunities_seen=self.opportunities_seen,
            opportunities_taken=self.opportunities_taken,
            total_cost=total_cost,
            total_payout=total_payout,
            gross_profit=gross_profit,
            fees_paid=fees_paid,
            net_profit=net_profit,
            win_rate=win_rate,
            avg_profit_per_market=net_profit / markets_traded if markets_traded > 0 else 0,
            avg_spread_captured=avg_spread,
            roi_pct=roi_pct,
            max_drawdown=max_drawdown,
            max_exposure=max(self._exposure_curve) if self._exposure_curve else 0,
            sharpe_ratio=sharpe,
            order_type=self.order_type,
            fee_rate=self.fee_rate,
            markets=[m.__dict__ for m in self.markets[:10]],  # First 10 for detail
            trades=self.all_trades[:100],  # First 100 trades
        )

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from exposure curve."""
        if not self._exposure_curve:
            return 0

        peak = 0
        max_dd = 0

        for exposure in self._exposure_curve:
            if exposure > peak:
                peak = exposure
            dd = peak - exposure
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, net_profit: float, total_cost: float) -> float:
        """Calculate simplified Sharpe ratio."""
        if total_cost == 0 or len(self.markets) == 0:
            return 0

        # Assume risk-free rate of 0 for simplicity
        returns = net_profit / total_cost

        # Estimate volatility from individual market returns
        market_returns = []
        for pos in self.positions.settled_positions:
            if pos.total_cost > 0:
                market_returns.append((pos.realized_profit or 0) / pos.total_cost)

        if not market_returns:
            return 0

        avg_return = sum(market_returns) / len(market_returns)
        variance = sum((r - avg_return) ** 2 for r in market_returns) / len(market_returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return 0

        return avg_return / std_dev

    def print_results(self, result: BacktestResult):
        """Print formatted backtest results."""
        print("\n" + "=" * 70)
        print("GABAGOOL SPREAD CAPTURE BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n📅 Period: {result.start_date[:10]} to {result.end_date[:10]}")
        print(f"📊 Markets: {result.total_markets} total, {result.markets_traded} traded")

        print("\n--- Trading Activity ---")
        print(f"Opportunities Seen: {result.opportunities_seen}")
        print(f"Opportunities Taken: {result.opportunities_taken}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Avg Spread Captured: {result.avg_spread_captured*100:.2f}%")

        print("\n--- Financial Results ---")
        print(f"Total Cost: ${result.total_cost:,.2f}")
        print(f"Total Payout: ${result.total_payout:,.2f}")
        print(f"Gross Profit: ${result.gross_profit:,.2f}")
        print(f"Fees Paid: ${result.fees_paid:,.2f}")
        print(f"Net Profit: ${result.net_profit:,.2f}")

        print("\n--- Performance Metrics ---")
        print(f"ROI: {result.roi_pct:.2f}%")
        print(f"Win Rate: {result.win_rate*100:.1f}%")
        print(f"Avg Profit/Market: ${result.avg_profit_per_market:.2f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

        print("\n--- Risk Metrics ---")
        print(f"Max Exposure: ${result.max_exposure:,.2f}")
        print(f"Max Drawdown: ${result.max_drawdown:,.2f}")

        print("\n--- Fees ---")
        print(f"Order Type: {result.order_type.upper()}")
        print(f"Fee Rate: {result.fee_rate * 100:.1f}%")
        print(f"Total Fees Paid: ${result.fees_paid:.2f}")
        if result.total_trades > 0:
            print(f"Avg Fee/Trade: ${result.fees_paid / result.total_trades:.2f}")

        # Profitability indicator
        print("\n" + "=" * 70)
        if result.net_profit > 0:
            print(f"✅ PROFITABLE: +${result.net_profit:,.2f} ({result.roi_pct:.1f}% ROI)")
        else:
            print(f"❌ LOSS: ${result.net_profit:,.2f} ({result.roi_pct:.1f}% ROI)")
        print("=" * 70)

    def save_results(self, result: BacktestResult, output_dir: str = "data/backtests"):
        """Save backtest results to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_path / f"gabagool_backtest_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"\n📁 Results saved to: {filename}")
        return filename


def run_backtest(
    num_markets: int = 100,
    config: GabagoolConfig = None,
    order_type: str = "maker",
    custom_fee: Optional[float] = None,
    save_results: bool = True,
) -> BacktestResult:
    """
    Run a backtest with default settings.

    Args:
        num_markets: Number of markets to simulate
        config: Configuration (uses default if None)
        order_type: "maker" (0% fee) or "taker" (2% fee)
        custom_fee: Optional custom fee rate (overrides order_type)
        save_results: Whether to save results to file

    Returns:
        BacktestResult
    """
    backtest = GabagoolBacktest(
        config=config,
        order_type=order_type,
        custom_fee=custom_fee
    )
    result = backtest.run_backtest(num_markets=num_markets)
    backtest.print_results(result)

    if save_results:
        backtest.save_results(result)

    return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gabagool Strategy Backtest')
    parser.add_argument('--markets', type=int, default=100, help='Number of markets to simulate')
    parser.add_argument('--min-spread', type=float, default=0.02, help='Minimum spread threshold')
    parser.add_argument('--order-size', type=float, default=15.0, help='Order size in USD')
    parser.add_argument('--order-type', choices=['maker', 'taker'], default='maker',
                        help='Order type: maker (0%% fee) or taker (2%% fee)')
    parser.add_argument('--fee', type=float, default=None, help='Custom fee rate (overrides order-type)')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config = GabagoolConfig(
        MIN_SPREAD=args.min_spread,
        ORDER_SIZE_USD=args.order_size,
    )

    run_backtest(
        num_markets=args.markets,
        config=config,
        order_type=args.order_type,
        custom_fee=args.fee,
        save_results=not args.no_save,
    )
