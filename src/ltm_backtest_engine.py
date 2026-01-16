"""
LTM-Enhanced Backtest Engine

Extends the base BacktestEngine with full LTM integration:
- Time-bucket based decision making
- Snapshot collection during backtest
- Decay model analysis
- Optional bandit learning
- Comprehensive LTM metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import logging
import os

from config import settings
from config.risk_params import RiskParams
from .position_manager import Position, Portfolio, MarketState, Trade
from .risk_manager import RiskManager
from .ltm_risk_manager import LTMRiskManager
from .metrics import PerformanceMetrics
from .ltm import (
    LTMCollector,
    LTMFeatures,
    LTMPolicy,
    PairCostDecay,
    LTMBanditManager,
)

logger = logging.getLogger(__name__)


class LTMBacktestEngine:
    """
    Backtest engine with full LTM (Liquidity Time Model) integration.

    Provides:
    - Bucket-aware trading decisions
    - Real-time snapshot collection
    - Decay-based entry filtering
    - Optional bandit parameter learning
    - Per-bucket performance metrics
    """

    def __init__(
        self,
        config: Any = None,
        risk_params: RiskParams = None,
        initial_capital: float = None,
        ltm_policy_path: str = None,
        use_decay_model: bool = True,
        use_bandit: bool = False,
        bandit_path: str = None,
        collect_snapshots: bool = True,
    ):
        """
        Initialize LTM backtest engine.

        Args:
            config: Configuration settings
            risk_params: Risk parameters
            initial_capital: Starting capital
            ltm_policy_path: Path to LTM policy YAML
            use_decay_model: Use pair-cost decay analysis
            use_bandit: Use bandit auto-tuning
            bandit_path: Path to load/save bandit state
            collect_snapshots: Collect LTM snapshots during backtest
        """
        self.config = config or settings
        self.risk_params = risk_params or RiskParams()

        # Initialize LTM risk manager
        self.risk_manager = LTMRiskManager(
            risk_params=self.risk_params,
            ltm_policy_path=ltm_policy_path,
            use_decay_model=use_decay_model,
            use_bandit=use_bandit,
            bandit_path=bandit_path,
        )

        capital = initial_capital or getattr(self.config, 'INITIAL_CAPITAL', 5000)
        self.portfolio = Portfolio(capital)
        self.trades: List[Trade] = []
        self.metrics_collector = PerformanceMetrics()

        # LTM-specific tracking
        self.collect_snapshots = collect_snapshots
        self.ltm_collector = LTMCollector() if collect_snapshots else None

        # Per-bucket metrics
        self.bucket_trades: Dict[int, List[Trade]] = {i: [] for i in range(15)}
        self.bucket_decisions: Dict[int, Dict[str, int]] = {
            i: {'entered': 0, 'skipped': 0, 'stopped': 0}
            for i in range(15)
        }

        # Decay model for global analysis
        self.decay_model = PairCostDecay() if use_decay_model else None

    def run(
        self,
        markets_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute the backtest with LTM integration.

        Args:
            markets_df: DataFrame with market information
            prices_df: DataFrame with price history
            verbose: Show progress bar

        Returns:
            Dictionary with comprehensive backtest results
        """
        logger.info("Starting LTM-enhanced backtest...")
        logger.info(f"Initial capital: ${self.portfolio.initial_capital}")
        logger.info(f"Markets to process: {len(markets_df)}")
        logger.info(f"LTM features: decay={self.risk_manager.use_decay_model}, "
                   f"bandit={self.risk_manager.use_bandit}")

        # Ensure datetime columns
        markets_df = markets_df.copy()
        markets_df['start_time'] = pd.to_datetime(markets_df['start_time'])
        markets_df['end_time'] = pd.to_datetime(markets_df['end_time'])

        prices_df = prices_df.copy()
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

        # Sort markets by start time
        markets_df = markets_df.sort_values('start_time')

        # Create iterator
        iterator = tqdm(markets_df.iterrows(), total=len(markets_df)) if verbose else markets_df.iterrows()

        for idx, market_row in iterator:
            current_time = market_row['start_time']

            # 1. Settle expired positions
            self._settle_expired_positions(current_time, markets_df)

            # 2. Update active positions
            self._update_active_positions(current_time, prices_df)

            # 3. Check rebalancing (with LTM-aware imbalance limits)
            self._check_rebalancing(prices_df)

            # 4. Collect LTM snapshots for this market
            if self.collect_snapshots:
                self._collect_market_snapshots(market_row, prices_df)

            # 5. Evaluate new market entry with LTM
            self._evaluate_market_entry(market_row, prices_df)

            # 6. Record portfolio snapshot
            self.portfolio.record_snapshot(current_time)

        # Final settlement
        self._settle_all_positions(markets_df)

        # Save bandit state if used
        if self.risk_manager.use_bandit:
            self.risk_manager.save_bandit_state()

        # Generate comprehensive report
        return self._generate_report()

    def _get_market_state(
        self,
        market_row: pd.Series,
        prices_df: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> Optional[MarketState]:
        """Get current market state from prices."""
        market_id = market_row['market_id']

        # Get prices for this market
        market_prices = prices_df[prices_df['market_id'] == market_id]

        if market_prices.empty:
            return MarketState(
                market_id=market_id,
                price_yes=0.50,
                price_no=0.50,
                volume=market_row.get('volume', 0),
                time_remaining=900,
                outcome=market_row.get('outcome'),
                current_time=timestamp or market_row['start_time'],
            )

        # Get latest prices
        if timestamp:
            prices_before = market_prices[market_prices['timestamp'] <= timestamp]
            if prices_before.empty:
                prices_before = market_prices
            latest = prices_before.iloc[-1]
        else:
            latest = market_prices.iloc[-1]

        # Calculate time remaining
        end_time = market_row['end_time']
        current = timestamp or market_row['start_time']
        time_remaining = max(0, (end_time - current).total_seconds())

        return MarketState(
            market_id=market_id,
            price_yes=latest.get('price_yes', 0.50),
            price_no=latest.get('price_no', 0.50),
            volume=market_row.get('volume', 0),
            time_remaining=int(time_remaining),
            outcome=market_row.get('outcome'),
            current_time=current,
        )

    def _collect_market_snapshots(
        self,
        market_row: pd.Series,
        prices_df: pd.DataFrame
    ) -> None:
        """Collect LTM snapshots across the market lifecycle."""
        market_id = market_row['market_id']
        market_prices = prices_df[prices_df['market_id'] == market_id].copy()

        if market_prices.empty:
            return

        market_prices = market_prices.sort_values('timestamp')
        end_time = market_row['end_time']

        for _, price_row in market_prices.iterrows():
            timestamp = price_row['timestamp']
            time_remaining = max(0, (end_time - timestamp).total_seconds())

            snapshot = self.ltm_collector.create_snapshot(
                market_id=market_id,
                timestamp=timestamp,
                time_remaining_sec=time_remaining,
                price_yes=price_row.get('price_yes', 0.5),
                price_no=price_row.get('price_no', 0.5),
                volume=price_row.get('volume', 0),
            )
            self.ltm_collector.add_snapshot(snapshot)

            # Also feed decay model
            if self.decay_model:
                pair_cost = price_row.get('price_yes', 0.5) + price_row.get('price_no', 0.5)
                self.decay_model.add_observation(
                    market_id=market_id,
                    timestamp=timestamp,
                    pair_cost=pair_cost,
                    time_remaining_sec=time_remaining,
                )

    def _evaluate_market_entry(
        self,
        market_row: pd.Series,
        prices_df: pd.DataFrame
    ) -> None:
        """Evaluate and potentially enter a new market using LTM."""
        market_state = self._get_market_state(market_row, prices_df)

        if market_state is None:
            return

        # Check if already have position in this market
        if self.portfolio.get_position(market_state.market_id):
            return

        # Get bucket for tracking
        bucket_idx = self.risk_manager.ltm_policy.get_bucket_index(
            market_state.time_remaining
        )

        # Check if stopped by LTM
        if self.risk_manager.get_dynamic_stop(market_state):
            self.bucket_decisions[bucket_idx]['stopped'] += 1
            return

        # Full LTM evaluation
        ltm_decision = self.risk_manager.evaluate_ltm(market_state, self.portfolio)

        if not ltm_decision.should_trade:
            self.bucket_decisions[bucket_idx]['skipped'] += 1
            logger.debug(f"Skipped {market_state.market_id}: {ltm_decision.reason}")
            return

        # Base entry checks (capital, exposure, etc.)
        # We call parent's should_enter logic but skip the LTM part since we already did it
        if not self._base_entry_checks(market_state):
            self.bucket_decisions[bucket_idx]['skipped'] += 1
            return

        # Calculate order size with LTM adjustments
        sizing = self.risk_manager.calculate_order_size(market_state, self.portfolio)

        # Further adjust by LTM decision
        if ltm_decision.adjusted_size > 0:
            max_by_ltm = ltm_decision.adjusted_size
            if sizing['shares_yes'] > max_by_ltm:
                ratio = max_by_ltm / sizing['shares_yes']
                sizing['shares_yes'] = int(sizing['shares_yes'] * ratio)
                sizing['shares_no'] = int(sizing['shares_no'] * ratio)
                sizing['cost_yes'] *= ratio
                sizing['cost_no'] *= ratio
                sizing['total_cost'] *= ratio

        if sizing['total_cost'] < self.risk_params.MIN_ORDER_SIZE * 2:
            self.bucket_decisions[bucket_idx]['skipped'] += 1
            return

        # Simulate execution with slippage
        execution = self._simulate_execution(market_state, sizing)

        # Create and add position
        position = Position(
            market=market_state,
            shares_yes=execution['shares_yes'],
            shares_no=execution['shares_no'],
            cost_yes=execution['cost_yes'],
            cost_no=execution['cost_no'],
            entry_time=market_state.current_time,
            entry_spread=market_state.spread,
        )

        self.portfolio.add_position(position)
        self.bucket_decisions[bucket_idx]['entered'] += 1

        # Update bandit if enabled
        if self.risk_manager.use_bandit and ltm_decision.bandit_params:
            # Store params for later reward update
            position._ltm_params = ltm_decision.bandit_params
            position._ltm_bucket = bucket_idx
            position._entry_time_remaining = market_state.time_remaining

    def _base_entry_checks(self, market_state: MarketState) -> bool:
        """Run base entry checks without LTM (already evaluated)."""
        # Capital check
        min_required = self.risk_params.MIN_ORDER_SIZE * 2
        if self.portfolio.available_cash < min_required:
            return False

        # Active markets limit
        if self.portfolio.active_markets >= self.risk_params.MAX_ACTIVE_MARKETS:
            return False

        # Total exposure check
        if self.portfolio.total_exposure >= self.risk_params.MAX_TOTAL_EXPOSURE:
            return False

        # Volume check
        if market_state.volume < self.risk_params.MIN_VOLUME:
            return False

        return True

    def _simulate_execution(
        self,
        market_state: MarketState,
        sizing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate order execution with LTM-aware slippage."""
        # Get bucket-specific slippage expectations
        bucket_idx = self.risk_manager.ltm_policy.get_bucket_index(
            market_state.time_remaining
        )

        # Base slippage calculation
        if market_state.volume > 0:
            volume_ratio = sizing['total_cost'] / market_state.volume
        else:
            volume_ratio = 0.1

        if volume_ratio < 0.01:
            base_slippage = 0.001
        elif volume_ratio < 0.05:
            base_slippage = 0.003
        else:
            base_slippage = 0.005

        # LTM adjustment: higher slippage in late buckets
        if bucket_idx >= 12:
            slippage = base_slippage * 1.5
        elif bucket_idx >= 10:
            slippage = base_slippage * 1.2
        else:
            slippage = base_slippage

        # Apply slippage
        executed_price_yes = market_state.price_yes * (1 + slippage)
        executed_price_no = market_state.price_no * (1 + slippage)

        return {
            "shares_yes": sizing['shares_yes'],
            "shares_no": sizing['shares_no'],
            "cost_yes": sizing['shares_yes'] * executed_price_yes,
            "cost_no": sizing['shares_no'] * executed_price_no,
            "slippage_paid": slippage * sizing['total_cost'],
        }

    def _settle_expired_positions(
        self,
        current_time: datetime,
        markets_df: pd.DataFrame
    ) -> None:
        """Settle positions for markets that have ended."""
        positions_to_close = []

        for position in self.portfolio.active_positions:
            market_id = position.market.market_id
            market_row = markets_df[markets_df['market_id'] == market_id]

            if market_row.empty:
                continue

            end_time = pd.to_datetime(market_row.iloc[0]['end_time'])

            if current_time >= end_time:
                positions_to_close.append((position, market_row.iloc[0]))

        for position, market_row in positions_to_close:
            self._close_position(position, market_row)

    def _close_position(self, position: Position, market_row: pd.Series) -> None:
        """Close a position at settlement and update bandit."""
        outcome = market_row.get('outcome', '')

        if pd.isna(outcome) or outcome == '':
            outcome = 'Up' if np.random.random() > 0.5 else 'Down'

        outcome_str = str(outcome).strip()

        if outcome_str.lower() == 'up':
            payout = position.shares_yes * 1.00
        else:
            payout = position.shares_no * 1.00

        trade = self.portfolio.close_position(position, payout, outcome_str)
        self.trades.append(trade)

        # Track by bucket
        if hasattr(position, '_ltm_bucket'):
            bucket_idx = position._ltm_bucket
            self.bucket_trades[bucket_idx].append(trade)

            # Update bandit with trade result
            if self.risk_manager.use_bandit and hasattr(position, '_ltm_params'):
                pnl = trade.profit
                slippage = getattr(trade, 'slippage', 0)
                self.risk_manager.update_bandit(
                    time_remaining=position._entry_time_remaining,
                    params_used=position._ltm_params,
                    pnl=pnl,
                    slippage=slippage,
                )

    def _settle_all_positions(self, markets_df: pd.DataFrame) -> None:
        """Settle all remaining positions at end of backtest."""
        while self.portfolio.active_positions:
            position = self.portfolio.active_positions[0]
            market_id = position.market.market_id
            market_row = markets_df[markets_df['market_id'] == market_id]

            if not market_row.empty:
                self._close_position(position, market_row.iloc[0])
            else:
                self.portfolio.close_position(position, position.shares_yes * 0.5, "Unknown")

    def _update_active_positions(
        self,
        current_time: datetime,
        prices_df: pd.DataFrame
    ) -> None:
        """Update market state for active positions."""
        for position in self.portfolio.active_positions:
            market_prices = prices_df[
                (prices_df['market_id'] == position.market.market_id) &
                (prices_df['timestamp'] <= current_time)
            ]

            if not market_prices.empty:
                latest = market_prices.iloc[-1]
                position.market.price_yes = latest.get('price_yes', position.market.price_yes)
                position.market.price_no = latest.get('price_no', position.market.price_no)
                position.market.current_time = current_time

    def _check_rebalancing(self, prices_df: pd.DataFrame) -> None:
        """Check and execute rebalancing with LTM-aware imbalance limits."""
        for position in self.portfolio.active_positions:
            if self.risk_manager.should_rebalance(position, position.market):
                rebalance = self.risk_manager.calculate_rebalance(position, position.market)

                if rebalance['action'] != 'HOLD' and rebalance['shares'] > 0:
                    if rebalance['action'] == 'BUY_YES':
                        cost = rebalance['shares'] * position.market.price_yes
                        if cost <= self.portfolio.cash:
                            self.portfolio.cash -= cost
                            position.shares_yes += rebalance['shares']
                            position.cost_yes += cost
                    elif rebalance['action'] == 'BUY_NO':
                        cost = rebalance['shares'] * position.market.price_no
                        if cost <= self.portfolio.cash:
                            self.portfolio.cash -= cost
                            position.shares_no += rebalance['shares']
                            position.cost_no += cost

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive backtest report with LTM metrics."""
        trades_data = [t.to_dict() for t in self.trades]
        trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()

        history_df = pd.DataFrame(self.portfolio.history)

        # Calculate base metrics
        metrics = self.metrics_collector.calculate_all(trades_df, history_df)

        # Calculate per-bucket metrics
        bucket_metrics = self._calculate_bucket_metrics()

        # LTM features from collected snapshots
        ltm_features = None
        if self.ltm_collector and self.ltm_collector.snapshots:
            ltm_features = LTMFeatures()
            ltm_features.compute_from_collector(self.ltm_collector)

        report = {
            "summary": {
                "initial_capital": self.portfolio.initial_capital,
                "final_capital": self.portfolio.total_value,
                "total_return_usd": self.portfolio.total_pnl,
                "total_return_pct": self.portfolio.total_return_pct,
                "total_trades": len(self.trades),
            },
            "metrics": metrics,
            "trades": trades_df,
            "portfolio_history": history_df,
            "portfolio_summary": self.portfolio.get_summary(),

            # LTM-specific results
            "ltm": {
                "bucket_metrics": bucket_metrics,
                "bucket_decisions": self.bucket_decisions,
                "risk_manager_stats": self.risk_manager.get_decision_stats(),
            }
        }

        # Add LTM features if computed
        if ltm_features:
            report["ltm"]["features"] = ltm_features.to_dataframe().to_dict('records')

        # Add bandit results if used
        if self.risk_manager.use_bandit and self.risk_manager.bandit:
            report["ltm"]["bandit"] = {
                "best_policy": self.risk_manager.bandit.get_best_policy(),
                "stats": self.risk_manager.bandit.get_stats(),
            }

        logger.info("Backtest complete!")
        logger.info(f"Final capital: ${self.portfolio.total_value:.2f}")
        logger.info(f"Total return: {self.portfolio.total_return_pct:.2f}%")
        logger.info(f"Total trades: {len(self.trades)}")

        return report

    def _calculate_bucket_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Calculate performance metrics per bucket."""
        bucket_metrics = {}

        for bucket_idx in range(15):
            trades = self.bucket_trades[bucket_idx]
            decisions = self.bucket_decisions[bucket_idx]

            if not trades:
                bucket_metrics[bucket_idx] = {
                    'n_trades': 0,
                    'decisions': decisions,
                }
                continue

            profits = [t.profit for t in trades]
            rois = [t.roi for t in trades if hasattr(t, 'roi')]

            bucket_metrics[bucket_idx] = {
                'n_trades': len(trades),
                'total_pnl': sum(profits),
                'avg_pnl': np.mean(profits),
                'win_rate': sum(1 for p in profits if p > 0) / len(profits),
                'avg_roi': np.mean(rois) if rois else 0,
                'max_profit': max(profits),
                'max_loss': min(profits),
                'decisions': decisions,
            }

        return bucket_metrics

    def save_ltm_data(self, output_dir: str = 'data') -> Dict[str, str]:
        """Save all LTM data (snapshots, features, bandit state)."""
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # Save snapshots
        if self.ltm_collector and self.ltm_collector.snapshots:
            snapshot_path = os.path.join(output_dir, 'ltm_snapshots.parquet')
            self.ltm_collector.save_snapshots(snapshot_path)
            paths['snapshots'] = snapshot_path

            # Compute and save features
            features = LTMFeatures()
            features.compute_from_collector(self.ltm_collector)
            features_path = os.path.join(output_dir, 'ltm_features.parquet')
            features.save(features_path)
            paths['features'] = features_path

        # Save bandit state
        if self.risk_manager.use_bandit and self.risk_manager.bandit:
            bandit_path = os.path.join(output_dir, 'ltm_bandit.json')
            self.risk_manager.bandit.save(bandit_path)
            paths['bandit'] = bandit_path

        return paths

    def generate_ltm_report(self) -> str:
        """Generate human-readable LTM analysis report."""
        lines = [
            "=" * 70,
            "LTM BACKTEST ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        # Summary
        lines.extend([
            "PERFORMANCE SUMMARY",
            f"  Initial Capital: ${self.portfolio.initial_capital:,.2f}",
            f"  Final Capital: ${self.portfolio.total_value:,.2f}",
            f"  Total Return: {self.portfolio.total_return_pct:.2f}%",
            f"  Total Trades: {len(self.trades)}",
            "",
        ])

        # Bucket breakdown
        lines.extend([
            "-" * 70,
            "PERFORMANCE BY TIME BUCKET",
            "-" * 70,
            "",
        ])

        for bucket_idx in range(15):
            trades = self.bucket_trades[bucket_idx]
            decisions = self.bucket_decisions[bucket_idx]

            time_start = 900 - (bucket_idx * 60)
            time_end = max(0, time_start - 60)

            if trades:
                pnl = sum(t.profit for t in trades)
                win_rate = sum(1 for t in trades if t.profit > 0) / len(trades)
                lines.append(
                    f"Bucket {bucket_idx:2d} ({time_end:3.0f}-{time_start:3.0f}s): "
                    f"{len(trades):3d} trades, "
                    f"PnL=${pnl:8.2f}, "
                    f"WR={win_rate:5.1%}, "
                    f"entered={decisions['entered']}, "
                    f"skipped={decisions['skipped']}, "
                    f"stopped={decisions['stopped']}"
                )
            else:
                lines.append(
                    f"Bucket {bucket_idx:2d} ({time_end:3.0f}-{time_start:3.0f}s): "
                    f"  0 trades, "
                    f"entered={decisions['entered']}, "
                    f"skipped={decisions['skipped']}, "
                    f"stopped={decisions['stopped']}"
                )

        # Risk manager stats
        stats = self.risk_manager.get_decision_stats()
        if stats:
            lines.extend([
                "",
                "-" * 70,
                "LTM DECISION STATISTICS",
                "-" * 70,
                f"  Total Decisions: {stats.get('total_decisions', 0)}",
                f"  Trade Rate: {stats.get('trade_rate', 0):.1%}",
                "",
                "  Skip Reasons:",
            ])
            for reason, count in stats.get('skip_reasons', {}).items():
                lines.append(f"    {reason}: {count}")

        return "\n".join(lines)


def run_ltm_backtest(
    markets_path: str,
    prices_path: str,
    initial_capital: float = 5000,
    risk_params: RiskParams = None,
    ltm_policy_path: str = None,
    use_decay: bool = True,
    use_bandit: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run LTM backtest from file paths.

    Args:
        markets_path: Path to markets CSV/parquet
        prices_path: Path to prices CSV/parquet
        initial_capital: Starting capital
        risk_params: Risk parameters
        ltm_policy_path: Path to LTM policy YAML
        use_decay: Use decay model
        use_bandit: Use bandit learning

    Returns:
        Backtest results
    """
    # Load data
    if markets_path.endswith('.parquet'):
        markets_df = pd.read_parquet(markets_path)
    else:
        markets_df = pd.read_csv(markets_path)

    if prices_path.endswith('.parquet'):
        prices_df = pd.read_parquet(prices_path)
    else:
        prices_df = pd.read_csv(prices_path)

    # Run backtest
    engine = LTMBacktestEngine(
        initial_capital=initial_capital,
        risk_params=risk_params,
        ltm_policy_path=ltm_policy_path,
        use_decay_model=use_decay,
        use_bandit=use_bandit,
    )

    return engine.run(markets_df, prices_df)
