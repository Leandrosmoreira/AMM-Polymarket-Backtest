"""
LTM Features - Aggregates snapshots into bucket statistics

Generates curves for:
- edge_required_curve[bucket]
- max_order_size_curve[bucket]
- stop_curve[bucket]
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

from .collector import LTMCollector, LTMSnapshot

logger = logging.getLogger(__name__)


@dataclass
class BucketStats:
    """Statistics for a single time bucket."""

    bucket_index: int
    bucket_start_sec: float  # Time remaining at start of bucket
    bucket_end_sec: float    # Time remaining at end of bucket

    # Sample size
    n_samples: int = 0
    n_markets: int = 0

    # Spread statistics
    spread_mean: float = 0.0
    spread_std: float = 0.0
    spread_p10: float = 0.0  # 10th percentile (best spreads)
    spread_p50: float = 0.0  # Median
    spread_p90: float = 0.0  # 90th percentile (worst spreads)

    # Depth statistics
    depth_mean: float = 0.0
    depth_std: float = 0.0
    depth_p10: float = 0.0
    depth_p50: float = 0.0
    depth_p90: float = 0.0

    # Volume statistics
    volume_mean: float = 0.0
    volume_std: float = 0.0

    # Fill rate statistics
    fill_rate_mean: float = 1.0
    fill_rate_std: float = 0.0
    fill_rate_p10: float = 1.0

    # Slippage statistics
    slippage_mean: float = 0.0
    slippage_std: float = 0.0
    slippage_p90: float = 0.0  # Worst case slippage

    # Liquidity score
    liquidity_score_mean: float = 0.0
    liquidity_score_std: float = 0.0

    # Book imbalance
    imbalance_mean: float = 0.0
    imbalance_std: float = 0.0
    imbalance_abs_mean: float = 0.0  # Mean absolute imbalance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LTMFeatures:
    """
    Aggregates LTM snapshots into bucket statistics and curves.
    """

    bucket_size_sec: int = 60
    total_time_sec: int = 900
    bucket_stats: Dict[int, BucketStats] = field(default_factory=dict)

    def __post_init__(self):
        self.num_buckets = self.total_time_sec // self.bucket_size_sec

    def compute_from_collector(self, collector: LTMCollector) -> None:
        """Compute bucket statistics from collector snapshots."""
        df = collector.to_dataframe()
        if df.empty:
            logger.warning("No snapshots to compute features from")
            return

        self.compute_from_dataframe(df)

    def compute_from_dataframe(self, df: pd.DataFrame) -> None:
        """Compute bucket statistics from DataFrame."""
        if df.empty:
            return

        self.bucket_stats = {}

        for bucket_idx in range(self.num_buckets):
            bucket_df = df[df['bucket_index'] == bucket_idx]

            # Calculate time range for this bucket
            bucket_start = self.total_time_sec - (bucket_idx * self.bucket_size_sec)
            bucket_end = bucket_start - self.bucket_size_sec

            stats = BucketStats(
                bucket_index=bucket_idx,
                bucket_start_sec=bucket_start,
                bucket_end_sec=max(0, bucket_end),
            )

            if len(bucket_df) > 0:
                stats.n_samples = len(bucket_df)
                stats.n_markets = bucket_df['market_id'].nunique()

                # Spread stats
                if 'spread' in bucket_df.columns:
                    spreads = bucket_df['spread'].dropna()
                    if len(spreads) > 0:
                        stats.spread_mean = spreads.mean()
                        stats.spread_std = spreads.std()
                        stats.spread_p10 = np.percentile(spreads, 10)
                        stats.spread_p50 = np.percentile(spreads, 50)
                        stats.spread_p90 = np.percentile(spreads, 90)

                # Depth stats
                if 'total_depth' in bucket_df.columns:
                    depths = bucket_df['total_depth'].dropna()
                    if len(depths) > 0:
                        stats.depth_mean = depths.mean()
                        stats.depth_std = depths.std()
                        stats.depth_p10 = np.percentile(depths, 10)
                        stats.depth_p50 = np.percentile(depths, 50)
                        stats.depth_p90 = np.percentile(depths, 90)

                # Volume stats
                if 'volume' in bucket_df.columns:
                    volumes = bucket_df['volume'].dropna()
                    if len(volumes) > 0:
                        stats.volume_mean = volumes.mean()
                        stats.volume_std = volumes.std()

                # Fill rate stats
                if 'fill_rate' in bucket_df.columns:
                    fill_rates = bucket_df['fill_rate'].dropna()
                    if len(fill_rates) > 0:
                        stats.fill_rate_mean = fill_rates.mean()
                        stats.fill_rate_std = fill_rates.std()
                        stats.fill_rate_p10 = np.percentile(fill_rates, 10)

                # Slippage stats
                if 'slippage_realized' in bucket_df.columns:
                    slippages = bucket_df['slippage_realized'].dropna()
                    if len(slippages) > 0:
                        stats.slippage_mean = slippages.mean()
                        stats.slippage_std = slippages.std()
                        stats.slippage_p90 = np.percentile(slippages, 90)

                # Liquidity score
                if 'liquidity_score' in bucket_df.columns:
                    scores = bucket_df['liquidity_score'].dropna()
                    if len(scores) > 0:
                        stats.liquidity_score_mean = scores.mean()
                        stats.liquidity_score_std = scores.std()

                # Imbalance stats
                if 'book_imbalance' in bucket_df.columns:
                    imbalances = bucket_df['book_imbalance'].dropna()
                    if len(imbalances) > 0:
                        stats.imbalance_mean = imbalances.mean()
                        stats.imbalance_std = imbalances.std()
                        stats.imbalance_abs_mean = np.abs(imbalances).mean()

            self.bucket_stats[bucket_idx] = stats

        logger.info(f"Computed features for {self.num_buckets} buckets")

    def get_bucket_stats(self, bucket_index: int) -> Optional[BucketStats]:
        """Get statistics for a specific bucket."""
        return self.bucket_stats.get(bucket_index)

    def get_stats_by_time_remaining(self, time_remaining_sec: float) -> Optional[BucketStats]:
        """Get bucket stats for given time remaining."""
        bucket_idx = self._time_to_bucket(time_remaining_sec)
        return self.get_bucket_stats(bucket_idx)

    def _time_to_bucket(self, time_remaining_sec: float) -> int:
        """Convert time remaining to bucket index."""
        if time_remaining_sec >= self.total_time_sec:
            return 0
        if time_remaining_sec <= 0:
            return self.num_buckets - 1
        elapsed = self.total_time_sec - time_remaining_sec
        bucket = int(elapsed // self.bucket_size_sec)
        return min(bucket, self.num_buckets - 1)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all bucket stats to DataFrame."""
        if not self.bucket_stats:
            return pd.DataFrame()

        data = [stats.to_dict() for stats in self.bucket_stats.values()]
        return pd.DataFrame(data).sort_values('bucket_index')

    def get_spread_curve(self) -> Dict[int, float]:
        """Get spread by bucket (median spread)."""
        return {idx: stats.spread_p50 for idx, stats in self.bucket_stats.items()}

    def get_depth_curve(self) -> Dict[int, float]:
        """Get depth by bucket (median depth)."""
        return {idx: stats.depth_p50 for idx, stats in self.bucket_stats.items()}

    def get_liquidity_curve(self) -> Dict[int, float]:
        """Get liquidity score by bucket."""
        return {idx: stats.liquidity_score_mean for idx, stats in self.bucket_stats.items()}

    def get_slippage_curve(self) -> Dict[int, float]:
        """Get p90 slippage by bucket (worst case)."""
        return {idx: stats.slippage_p90 for idx, stats in self.bucket_stats.items()}

    def get_fill_rate_curve(self) -> Dict[int, float]:
        """Get p10 fill rate by bucket (worst case)."""
        return {idx: stats.fill_rate_p10 for idx, stats in self.bucket_stats.items()}

    def generate_report(self) -> str:
        """Generate human-readable report of liquidity curves."""
        lines = [
            "=" * 60,
            "LTM LIQUIDITY CURVE REPORT",
            "=" * 60,
            "",
            f"Total buckets: {self.num_buckets}",
            f"Bucket size: {self.bucket_size_sec}s",
            f"Total time: {self.total_time_sec}s",
            "",
            "-" * 60,
            "BUCKET ANALYSIS (sorted by time remaining)",
            "-" * 60,
            "",
        ]

        for idx in range(self.num_buckets):
            stats = self.bucket_stats.get(idx)
            if stats:
                lines.append(f"Bucket {idx}: {stats.bucket_end_sec:.0f}-{stats.bucket_start_sec:.0f}s remaining")
                lines.append(f"  Samples: {stats.n_samples} from {stats.n_markets} markets")
                lines.append(f"  Spread: mean={stats.spread_mean:.4f}, p50={stats.spread_p50:.4f}, p90={stats.spread_p90:.4f}")
                lines.append(f"  Depth: mean={stats.depth_mean:.0f}, p50={stats.depth_p50:.0f}")
                lines.append(f"  Fill Rate: mean={stats.fill_rate_mean:.2%}, p10={stats.fill_rate_p10:.2%}")
                lines.append(f"  Slippage: mean={stats.slippage_mean:.4f}, p90={stats.slippage_p90:.4f}")
                lines.append(f"  Liquidity Score: {stats.liquidity_score_mean:.1f}")
                lines.append("")

        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """Save bucket stats to parquet."""
        df = self.to_dataframe()
        if not df.empty:
            df.to_parquet(filepath, index=False)
            logger.info(f"Saved features to {filepath}")

    def load(self, filepath: str) -> None:
        """Load bucket stats from parquet."""
        df = pd.read_parquet(filepath)
        self.bucket_stats = {}

        for _, row in df.iterrows():
            stats = BucketStats(
                bucket_index=int(row['bucket_index']),
                bucket_start_sec=row['bucket_start_sec'],
                bucket_end_sec=row['bucket_end_sec'],
                n_samples=int(row.get('n_samples', 0)),
                n_markets=int(row.get('n_markets', 0)),
                spread_mean=row.get('spread_mean', 0),
                spread_std=row.get('spread_std', 0),
                spread_p10=row.get('spread_p10', 0),
                spread_p50=row.get('spread_p50', 0),
                spread_p90=row.get('spread_p90', 0),
                depth_mean=row.get('depth_mean', 0),
                depth_std=row.get('depth_std', 0),
                depth_p10=row.get('depth_p10', 0),
                depth_p50=row.get('depth_p50', 0),
                depth_p90=row.get('depth_p90', 0),
                volume_mean=row.get('volume_mean', 0),
                volume_std=row.get('volume_std', 0),
                fill_rate_mean=row.get('fill_rate_mean', 1.0),
                fill_rate_std=row.get('fill_rate_std', 0),
                fill_rate_p10=row.get('fill_rate_p10', 1.0),
                slippage_mean=row.get('slippage_mean', 0),
                slippage_std=row.get('slippage_std', 0),
                slippage_p90=row.get('slippage_p90', 0),
                liquidity_score_mean=row.get('liquidity_score_mean', 0),
                liquidity_score_std=row.get('liquidity_score_std', 0),
                imbalance_mean=row.get('imbalance_mean', 0),
                imbalance_std=row.get('imbalance_std', 0),
                imbalance_abs_mean=row.get('imbalance_abs_mean', 0),
            )
            self.bucket_stats[stats.bucket_index] = stats

        logger.info(f"Loaded features for {len(self.bucket_stats)} buckets")
