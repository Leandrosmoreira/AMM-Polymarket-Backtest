"""
BTC Backtest Engine
Main simulation engine for BTC 15min Up/Down strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
import logging

from config import settings
from config.btc_risk_params import BTCRiskParams
from .probability_calculator import (
    ProbabilityCalculator,
    calculate_std_from_ticks,
    normal_cdf,
)
from .btc_market_cycle import (
    MarketCycleManager,
    MarketPosition,
    BTCTrade,
    get_market_start_timestamp,
    get_time_remaining,
    get_elapsed_time,
    FIFTEEN_MIN_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestSnapshot:
    """Snapshot of backtest state at a point in time."""
    timestamp_ms: int
    market_start_ms: int
    current_price: float
    price_to_beat: float
    std_dev: float
    z_score: float
    prob_up: float
    prob_down: float
    token_price_up: Optional[float]
    token_price_down: Optional[float]
    opp_up: Optional[float]
    opp_down: Optional[float]
    shares_up: float
    shares_down: float
    invested: float
    remaining_capital: float
    action: Optional[str]  # 'BUY_UP', 'BUY_DOWN', 'HOLD', None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp_ms': self.timestamp_ms,
            'market_start_ms': self.market_start_ms,
            'current_price': self.current_price,
            'price_to_beat': self.price_to_beat,
            'std_dev': self.std_dev,
            'z_score': self.z_score,
            'prob_up': self.prob_up,
            'prob_down': self.prob_down,
            'token_price_up': self.token_price_up,
            'token_price_down': self.token_price_down,
            'opp_up': self.opp_up,
            'opp_down': self.opp_down,
            'shares_up': self.shares_up,
            'shares_down': self.shares_down,
            'invested': self.invested,
            'remaining_capital': self.remaining_capital,
            'action': self.action,
        }


class BTCBacktestEngine:
    """
    Main backtest engine for BTC 15min Up/Down strategy.

    Simulates the bot's behavior:
    - Z-Score calculation every 1 second
    - Standard deviation calculation every 5 seconds
    - Trade decisions every 10 seconds
    """

    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_params: BTCRiskParams = None,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Capital per market (resets each cycle)
            risk_params: Risk parameters
        """
        self.initial_capital = initial_capital
        self.risk_params = risk_params or BTCRiskParams()

        self.prob_calculator = ProbabilityCalculator(
            min_std_dev=self.risk_params.MIN_DESVIO_PADRAO
        )
        self.cycle_manager = MarketCycleManager(
            initial_capital=initial_capital,
            risk_params=self.risk_params,
        )

        self.snapshots: List[BacktestSnapshot] = []
        self.all_trades: List[BTCTrade] = []

        # Timing trackers
        self._last_std_calc_ms: int = 0
        self._last_trade_check_ms: int = 0
        self._cached_std: float = self.risk_params.MIN_DESVIO_PADRAO

    def run(
        self,
        chainlink_ticks: List[Dict[str, Any]],
        price_changes: Optional[List[Dict[str, Any]]] = None,
        order_books: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the backtest.

        Args:
            chainlink_ticks: List of Chainlink price ticks
                [{'ts': ms, 'price': float, 'diff': float}, ...]
            price_changes: Optional token price changes
                [{'ts': ms, 'up': float, 'down': float}, ...]
            order_books: Optional order book snapshots
            verbose: Show progress bar

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting BTC backtest...")
        logger.info(f"Initial capital per market: ${self.initial_capital}")
        logger.info(f"Total ticks: {len(chainlink_ticks)}")

        # Sort ticks by timestamp
        ticks = sorted(chainlink_ticks, key=lambda x: x['ts'])

        if not ticks:
            logger.error("No ticks to process")
            return self._generate_empty_report()

        # Preprocess price changes for faster lookup
        token_prices = self._preprocess_token_prices(price_changes)

        # Main simulation loop
        current_market_ticks: List[Dict[str, Any]] = []
        current_market_start: Optional[int] = None

        iterator = tqdm(ticks, desc="Backtesting") if verbose else ticks

        for tick in iterator:
            ts = tick['ts']
            price = tick['price']

            # Check for market change
            market_start = get_market_start_timestamp(ts)

            if market_start != current_market_start:
                # New market - process final state of previous market if any
                if current_market_start is not None and current_market_ticks:
                    self._finalize_market(current_market_ticks)

                # Start new market
                current_market_start = market_start
                current_market_ticks = []
                self.prob_calculator.reset()
                self._last_std_calc_ms = market_start
                self._last_trade_check_ms = market_start
                self._cached_std = self.risk_params.MIN_DESVIO_PADRAO

                # Initialize new market in cycle manager
                self.cycle_manager.reset_for_new_market(market_start, price)

            # Add tick to current market
            current_market_ticks.append(tick)
            self.prob_calculator.add_tick(price, ts)

            # Simulate bot timing
            self._process_tick(
                tick,
                current_market_ticks,
                token_prices,
            )

        # Finalize last market
        if current_market_ticks:
            self._finalize_market(current_market_ticks)

        # Generate report
        return self._generate_report()

    def _preprocess_token_prices(
        self,
        price_changes: Optional[List[Dict[str, Any]]]
    ) -> Dict[int, Tuple[float, float]]:
        """
        Preprocess token prices for quick lookup.

        Returns:
            Dict mapping timestamp to (up_price, down_price)
        """
        if not price_changes:
            return {}

        # Sort and create lookup
        prices = {}
        for pc in sorted(price_changes, key=lambda x: x.get('ts', 0)):
            ts = pc.get('ts', 0)
            up = pc.get('up', 0.5)
            down = pc.get('down', 0.5)
            prices[ts] = (up, down)

        return prices

    def _get_token_prices(
        self,
        timestamp_ms: int,
        token_prices: Dict[int, Tuple[float, float]]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get token prices at or before timestamp."""
        if not token_prices:
            return None, None

        # Find closest price before or at timestamp
        best_ts = None
        for ts in token_prices.keys():
            if ts <= timestamp_ms:
                if best_ts is None or ts > best_ts:
                    best_ts = ts

        if best_ts is not None:
            return token_prices[best_ts]

        return None, None

    def _process_tick(
        self,
        tick: Dict[str, Any],
        market_ticks: List[Dict[str, Any]],
        token_prices: Dict[int, Tuple[float, float]],
    ) -> None:
        """
        Process a single tick following bot timing rules.

        Z-Score: every 1 second
        Std Dev: every 5 seconds
        Trade: every 10 seconds
        """
        ts = tick['ts']
        price = tick['price']

        if self.cycle_manager.current_position is None:
            return

        price_to_beat = self.cycle_manager.current_position.price_to_beat

        # Always calculate Z-Score (every tick approximates 1 second)
        z_score, prob_up, prob_down = self.prob_calculator.calculate_probability(
            price, price_to_beat, self._cached_std
        )

        # Check if should recalculate std dev (every 5 seconds)
        if ts - self._last_std_calc_ms >= self.risk_params.INTERVALO_DESVIO:
            self._cached_std = self.prob_calculator.calculate_std_dev(force_recalc=True)
            self._last_std_calc_ms = ts

            # Recalculate with new std
            z_score, prob_up, prob_down = self.prob_calculator.calculate_probability(
                price, price_to_beat, self._cached_std
            )

        # Check if should evaluate trade (every 10 seconds)
        action = None
        if ts - self._last_trade_check_ms >= self.risk_params.INTERVALO_TRADE:
            self._last_trade_check_ms = ts

            # Get token prices
            up_price, down_price = self._get_token_prices(ts, token_prices)

            if up_price is not None and down_price is not None:
                # Calculate opportunity
                opportunity = self.prob_calculator.calculate_opportunity(
                    price, price_to_beat,
                    up_price, down_price,
                    self._cached_std
                )

                # Check if should trade
                should_trade, side, amount = self.cycle_manager.should_trade(
                    ts, opportunity
                )

                if should_trade and side and amount:
                    token_price = up_price if side == 'UP' else down_price
                    edge = opportunity['opp_up'] if side == 'UP' else opportunity['opp_down']

                    trade = self.cycle_manager.execute_trade(
                        ts, side, amount, token_price, edge
                    )

                    if trade:
                        self.all_trades.append(trade)
                        action = f'BUY_{side}'

                # Record snapshot
                self._record_snapshot(
                    ts, price, price_to_beat,
                    self._cached_std, z_score, prob_up, prob_down,
                    up_price, down_price,
                    opportunity.get('opp_up'), opportunity.get('opp_down'),
                    action
                )
            else:
                # No token prices - record snapshot anyway
                self._record_snapshot(
                    ts, price, price_to_beat,
                    self._cached_std, z_score, prob_up, prob_down,
                    None, None, None, None, None
                )

    def _record_snapshot(
        self,
        ts: int,
        price: float,
        price_to_beat: float,
        std_dev: float,
        z_score: float,
        prob_up: float,
        prob_down: float,
        token_up: Optional[float],
        token_down: Optional[float],
        opp_up: Optional[float],
        opp_down: Optional[float],
        action: Optional[str],
    ) -> None:
        """Record a backtest snapshot."""
        pos = self.cycle_manager.current_position
        snapshot = BacktestSnapshot(
            timestamp_ms=ts,
            market_start_ms=self.cycle_manager.current_market_start or 0,
            current_price=price,
            price_to_beat=price_to_beat,
            std_dev=std_dev,
            z_score=z_score,
            prob_up=prob_up,
            prob_down=prob_down,
            token_price_up=token_up,
            token_price_down=token_down,
            opp_up=opp_up,
            opp_down=opp_down,
            shares_up=pos.shares_up if pos else 0,
            shares_down=pos.shares_down if pos else 0,
            invested=pos.total_invested if pos else 0,
            remaining_capital=self.cycle_manager.remaining_capital,
            action=action,
        )
        self.snapshots.append(snapshot)

    def _finalize_market(self, market_ticks: List[Dict[str, Any]]) -> None:
        """Finalize and close a market."""
        if not market_ticks:
            return

        # Get final price
        final_tick = market_ticks[-1]
        final_price = final_tick['price']

        # Close position in cycle manager
        if self.cycle_manager.current_position is not None:
            self.cycle_manager.current_position.resolve(final_price)
            self.cycle_manager.completed_markets.append(
                self.cycle_manager.current_position
            )
            self.cycle_manager.total_profit += self.cycle_manager.current_position.profit

    def _generate_report(self) -> Dict[str, Any]:
        """Generate backtest report."""
        markets = self.cycle_manager.completed_markets
        summary = self.cycle_manager.get_summary()

        # Convert snapshots to DataFrame
        snapshots_data = [s.to_dict() for s in self.snapshots]
        snapshots_df = pd.DataFrame(snapshots_data) if snapshots_data else pd.DataFrame()

        # Convert trades to DataFrame
        trades_data = [t.to_dict() for t in self.all_trades]
        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

        # Convert markets to DataFrame
        markets_data = [m.to_dict() for m in markets]
        markets_df = pd.DataFrame(markets_data) if markets_data else pd.DataFrame()

        # Calculate detailed metrics
        metrics = self._calculate_metrics(markets, trades_df)

        report = {
            'summary': {
                'initial_capital_per_market': self.initial_capital,
                'total_markets': summary['total_markets'],
                'total_profit': summary['total_profit'],
                'win_rate': summary['win_rate'] * 100,
                'avg_profit_per_market': summary['avg_profit'],
                'total_trades': len(self.all_trades),
            },
            'metrics': metrics,
            'markets': markets_df,
            'trades': trades_df,
            'snapshots': snapshots_df,
        }

        logger.info("Backtest complete!")
        logger.info(f"Total markets: {summary['total_markets']}")
        logger.info(f"Total profit: ${summary['total_profit']:.2f}")
        logger.info(f"Win rate: {summary['win_rate']*100:.1f}%")
        logger.info(f"Total trades: {len(self.all_trades)}")

        return report

    def _calculate_metrics(
        self,
        markets: List[MarketPosition],
        trades_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        if not markets:
            return {
                'total_return_usd': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_profit_per_market': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0,
            }

        profits = [m.profit for m in markets]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        total_invested = sum(m.total_invested for m in markets if m.total_invested > 0)

        metrics = {
            'total_return_usd': sum(profits),
            'total_return_pct': (sum(profits) / total_invested * 100) if total_invested > 0 else 0,
            'win_rate': len(wins) / len(profits) * 100 if profits else 0,
            'avg_profit_per_market': np.mean(profits) if profits else 0,
            'max_profit': max(profits) if profits else 0,
            'max_loss': min(profits) if profits else 0,
            'std_profit': np.std(profits) if len(profits) > 1 else 0,
        }

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe-like ratio (profit / std)
        if metrics['std_profit'] > 0:
            metrics['sharpe_like'] = metrics['avg_profit_per_market'] / metrics['std_profit']
        else:
            metrics['sharpe_like'] = 0

        # Trade statistics
        if not trades_df.empty:
            metrics['avg_trade_size'] = trades_df['cost'].mean()
            metrics['avg_opportunity'] = trades_df['opportunity'].mean() * 100
            metrics['trades_per_market'] = len(trades_df) / len(markets)

            # Side distribution
            up_trades = len(trades_df[trades_df['side'] == 'UP'])
            down_trades = len(trades_df[trades_df['side'] == 'DOWN'])
            metrics['pct_up_trades'] = up_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
            metrics['pct_down_trades'] = down_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0

        # Outcome analysis
        up_wins = sum(1 for m in markets if m.outcome == 'UP')
        down_wins = sum(1 for m in markets if m.outcome == 'DOWN')
        metrics['pct_up_outcomes'] = up_wins / len(markets) * 100 if markets else 0
        metrics['pct_down_outcomes'] = down_wins / len(markets) * 100 if markets else 0

        return metrics

    def _generate_empty_report(self) -> Dict[str, Any]:
        """Generate empty report when no data."""
        return {
            'summary': {
                'initial_capital_per_market': self.initial_capital,
                'total_markets': 0,
                'total_profit': 0,
                'win_rate': 0,
                'avg_profit_per_market': 0,
                'total_trades': 0,
            },
            'metrics': {},
            'markets': pd.DataFrame(),
            'trades': pd.DataFrame(),
            'snapshots': pd.DataFrame(),
        }


def run_btc_backtest(
    chainlink_ticks: List[Dict[str, Any]],
    price_changes: Optional[List[Dict[str, Any]]] = None,
    initial_capital: float = 100.0,
    risk_params: BTCRiskParams = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to run BTC backtest.

    Args:
        chainlink_ticks: Price ticks from Chainlink
        price_changes: Token price changes (optional)
        initial_capital: Capital per market
        risk_params: Risk parameters
        verbose: Show progress

    Returns:
        Backtest results
    """
    engine = BTCBacktestEngine(
        initial_capital=initial_capital,
        risk_params=risk_params,
    )

    return engine.run(
        chainlink_ticks=chainlink_ticks,
        price_changes=price_changes,
        verbose=verbose,
    )
