"""
LTM Collector - Collects market snapshots by time bucket

Collects data per time bucket (e.g., 0-60s, 60-120s, ..., 840-900s):
- t_remaining_sec
- mid_up, mid_down
- spread_up, spread_down
- depth_up_topN, depth_down_topN
- imbalance_book
- trade_count / print_rate
- fill_rate_recent
- slippage_realized
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Default bucket configuration: 15 buckets of 60 seconds each
DEFAULT_BUCKET_SIZE_SEC = 60
DEFAULT_TOTAL_TIME_SEC = 900  # 15 minutes
DEFAULT_NUM_BUCKETS = 15


@dataclass
class LTMSnapshot:
    """A single snapshot of market state for LTM analysis."""

    # Identification
    market_id: str
    timestamp: datetime
    bucket_index: int  # 0-14 for 15-min markets

    # Time
    t_remaining_sec: float
    t_elapsed_sec: float

    # Prices
    price_yes: float
    price_no: float
    mid_price: float  # (yes + no) / 2
    pair_cost: float  # yes + no (should be close to 1.0)

    # Spread
    spread: float  # pair_cost - 1.0 (negative = opportunity)
    spread_bps: float  # spread in basis points

    # Depth (optional - from order book if available)
    depth_yes_top5: float = 0.0  # Sum of shares at top 5 price levels
    depth_no_top5: float = 0.0
    total_depth: float = 0.0

    # Imbalance
    book_imbalance: float = 0.0  # (depth_yes - depth_no) / (depth_yes + depth_no)

    # Volume and Activity
    volume: float = 0.0
    trade_count: int = 0

    # Execution Quality (from actual fills)
    fill_rate: float = 1.0  # Proportion of order filled (0-1)
    slippage_realized: float = 0.0  # Actual slippage from expected price
    execution_time_ms: float = 0.0  # Time to fill

    # Derived metrics
    liquidity_score: float = 0.0  # Composite liquidity metric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LTMCollector:
    """
    Collects and manages LTM snapshots across market sessions.

    Buckets are defined by time remaining, not elapsed time:
    - Bucket 0: 840-900s remaining (start of market)
    - Bucket 14: 0-60s remaining (end of market)
    """

    bucket_size_sec: int = DEFAULT_BUCKET_SIZE_SEC
    total_time_sec: int = DEFAULT_TOTAL_TIME_SEC
    snapshots: List[LTMSnapshot] = field(default_factory=list)

    def __post_init__(self):
        self.num_buckets = self.total_time_sec // self.bucket_size_sec

    def get_bucket_index(self, time_remaining_sec: float) -> int:
        """
        Get bucket index from time remaining.

        Bucket 0 = start (most time remaining)
        Bucket N-1 = end (least time remaining)
        """
        if time_remaining_sec >= self.total_time_sec:
            return 0
        if time_remaining_sec <= 0:
            return self.num_buckets - 1

        # Invert so bucket 0 = most time remaining
        elapsed = self.total_time_sec - time_remaining_sec
        bucket = int(elapsed // self.bucket_size_sec)
        return min(bucket, self.num_buckets - 1)

    def get_bucket_time_range(self, bucket_index: int) -> tuple:
        """Get the time remaining range for a bucket."""
        start_remaining = self.total_time_sec - (bucket_index * self.bucket_size_sec)
        end_remaining = start_remaining - self.bucket_size_sec
        return (max(0, end_remaining), start_remaining)

    def create_snapshot(
        self,
        market_id: str,
        timestamp: datetime,
        time_remaining_sec: float,
        price_yes: float,
        price_no: float,
        volume: float = 0.0,
        depth_yes_top5: float = 0.0,
        depth_no_top5: float = 0.0,
        trade_count: int = 0,
        fill_rate: float = 1.0,
        slippage_realized: float = 0.0,
        execution_time_ms: float = 0.0,
    ) -> LTMSnapshot:
        """Create a snapshot from market data."""

        bucket_index = self.get_bucket_index(time_remaining_sec)
        t_elapsed = self.total_time_sec - time_remaining_sec

        pair_cost = price_yes + price_no
        mid_price = pair_cost / 2
        spread = pair_cost - 1.0
        spread_bps = spread * 10000

        total_depth = depth_yes_top5 + depth_no_top5
        if total_depth > 0:
            book_imbalance = (depth_yes_top5 - depth_no_top5) / total_depth
        else:
            book_imbalance = 0.0

        # Composite liquidity score (higher = more liquid)
        # Considers: depth, volume, spread width
        liquidity_score = self._calculate_liquidity_score(
            total_depth, volume, abs(spread), fill_rate
        )

        snapshot = LTMSnapshot(
            market_id=market_id,
            timestamp=timestamp,
            bucket_index=bucket_index,
            t_remaining_sec=time_remaining_sec,
            t_elapsed_sec=t_elapsed,
            price_yes=price_yes,
            price_no=price_no,
            mid_price=mid_price,
            pair_cost=pair_cost,
            spread=spread,
            spread_bps=spread_bps,
            depth_yes_top5=depth_yes_top5,
            depth_no_top5=depth_no_top5,
            total_depth=total_depth,
            book_imbalance=book_imbalance,
            volume=volume,
            trade_count=trade_count,
            fill_rate=fill_rate,
            slippage_realized=slippage_realized,
            execution_time_ms=execution_time_ms,
            liquidity_score=liquidity_score,
        )

        return snapshot

    def _calculate_liquidity_score(
        self,
        depth: float,
        volume: float,
        spread_abs: float,
        fill_rate: float
    ) -> float:
        """Calculate composite liquidity score (0-100)."""
        # Normalize components
        depth_score = min(depth / 10000, 1.0) * 25  # Max 25 points
        volume_score = min(volume / 50000, 1.0) * 25  # Max 25 points
        spread_score = max(0, (0.05 - spread_abs) / 0.05) * 25  # Max 25 points
        fill_score = fill_rate * 25  # Max 25 points

        return depth_score + volume_score + spread_score + fill_score

    def add_snapshot(self, snapshot: LTMSnapshot) -> None:
        """Add a snapshot to the collection."""
        self.snapshots.append(snapshot)

    def collect_from_market_state(
        self,
        market_state: Any,
        fill_rate: float = 1.0,
        slippage: float = 0.0
    ) -> LTMSnapshot:
        """Create and add snapshot from MarketState object."""
        snapshot = self.create_snapshot(
            market_id=market_state.market_id,
            timestamp=market_state.current_time,
            time_remaining_sec=market_state.time_remaining,
            price_yes=market_state.price_yes,
            price_no=market_state.price_no,
            volume=market_state.volume,
            fill_rate=fill_rate,
            slippage_realized=slippage,
        )
        self.add_snapshot(snapshot)
        return snapshot

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all snapshots to DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()

        data = [s.to_dict() for s in self.snapshots]
        df = pd.DataFrame(data)

        # Ensure proper types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def save_snapshots(self, filepath: str) -> None:
        """Save snapshots to parquet file."""
        df = self.to_dataframe()
        if not df.empty:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved {len(df)} snapshots to {filepath}")

    def load_snapshots(self, filepath: str) -> None:
        """Load snapshots from parquet file."""
        df = pd.read_parquet(filepath)
        self.snapshots = []

        for _, row in df.iterrows():
            snapshot = LTMSnapshot(
                market_id=row['market_id'],
                timestamp=row['timestamp'],
                bucket_index=int(row['bucket_index']),
                t_remaining_sec=row['t_remaining_sec'],
                t_elapsed_sec=row['t_elapsed_sec'],
                price_yes=row['price_yes'],
                price_no=row['price_no'],
                mid_price=row['mid_price'],
                pair_cost=row['pair_cost'],
                spread=row['spread'],
                spread_bps=row['spread_bps'],
                depth_yes_top5=row.get('depth_yes_top5', 0),
                depth_no_top5=row.get('depth_no_top5', 0),
                total_depth=row.get('total_depth', 0),
                book_imbalance=row.get('book_imbalance', 0),
                volume=row.get('volume', 0),
                trade_count=row.get('trade_count', 0),
                fill_rate=row.get('fill_rate', 1.0),
                slippage_realized=row.get('slippage_realized', 0),
                execution_time_ms=row.get('execution_time_ms', 0),
                liquidity_score=row.get('liquidity_score', 0),
            )
            self.snapshots.append(snapshot)

        logger.info(f"Loaded {len(self.snapshots)} snapshots from {filepath}")

    def get_snapshots_by_bucket(self, bucket_index: int) -> List[LTMSnapshot]:
        """Get all snapshots for a specific bucket."""
        return [s for s in self.snapshots if s.bucket_index == bucket_index]

    def get_snapshots_by_market(self, market_id: str) -> List[LTMSnapshot]:
        """Get all snapshots for a specific market."""
        return [s for s in self.snapshots if s.market_id == market_id]

    def clear(self) -> None:
        """Clear all snapshots."""
        self.snapshots = []
