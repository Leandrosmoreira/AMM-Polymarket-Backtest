"""
LTM Policy - Defines trading policy by time bucket

Loads and applies bucket-specific rules from ltm_policy.yaml:
- edge_required: minimum edge to enter trade
- max_size: maximum order size
- max_spread_allowed: reject if spread is worse
- min_depth_required: reject if depth too low
- stop_trading: whether to stop trading in this bucket
- max_imbalance: maximum allowed position imbalance
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import yaml
import os
import logging

from .features import LTMFeatures, BucketStats

logger = logging.getLogger(__name__)


@dataclass
class BucketPolicy:
    """Trading policy for a single time bucket."""

    bucket_index: int
    time_remaining_start: float  # Start of bucket (time remaining)
    time_remaining_end: float    # End of bucket (time remaining)

    # Entry requirements
    edge_required: float = 0.02  # Minimum edge (pair_cost < 1 - edge_required)
    max_spread_allowed: float = 0.05  # Maximum spread to consider trading
    min_depth_required: float = 100  # Minimum depth required
    min_fill_rate_required: float = 0.8  # Minimum expected fill rate

    # Position sizing
    max_size: float = 100  # Maximum order size (shares)
    size_multiplier: float = 1.0  # Multiplier for base order size

    # Risk controls
    stop_trading: bool = False  # If True, do not open new positions
    max_imbalance: float = 0.3  # Maximum allowed YES/NO imbalance

    # Weight for this bucket in overall strategy
    weight: float = 1.0  # 0 = don't trade, 1 = normal, >1 = aggressive

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BucketPolicy':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LTMPolicy:
    """
    Complete LTM policy managing all bucket policies.
    """

    bucket_policies: Dict[int, BucketPolicy] = field(default_factory=dict)
    num_buckets: int = 15
    bucket_size_sec: int = 60
    total_time_sec: int = 900

    # Global defaults
    default_edge_required: float = 0.02
    default_max_spread: float = 0.05
    default_min_depth: float = 100
    default_max_size: float = 100
    default_max_imbalance: float = 0.3

    # Stop trading settings
    stop_trading_last_n_buckets: int = 1  # Stop in last N buckets
    stop_trading_threshold_sec: float = 60  # Stop if < X seconds remaining

    def __post_init__(self):
        if not self.bucket_policies:
            self._initialize_default_policies()

    def _initialize_default_policies(self) -> None:
        """Create default policies for all buckets."""
        for idx in range(self.num_buckets):
            time_start = self.total_time_sec - (idx * self.bucket_size_sec)
            time_end = max(0, time_start - self.bucket_size_sec)

            # Adaptive parameters based on bucket position
            # Early buckets (idx 0-4): normal trading
            # Middle buckets (idx 5-10): best conditions, lower edge required
            # Late buckets (idx 11-14): conservative, higher edge required

            if idx <= 4:  # Early phase (660-900s remaining)
                edge = self.default_edge_required
                max_size = self.default_max_size
                stop = False
                weight = 0.8  # Slightly conservative at start
                max_imbalance = 0.35

            elif idx <= 10:  # Middle phase (240-660s remaining)
                edge = self.default_edge_required * 0.75  # Lower edge OK
                max_size = self.default_max_size * 1.2  # Can size up
                stop = False
                weight = 1.2  # More aggressive
                max_imbalance = 0.3

            elif idx <= 12:  # Late phase (120-240s remaining)
                edge = self.default_edge_required * 1.5  # Higher edge required
                max_size = self.default_max_size * 0.7  # Size down
                stop = False
                weight = 0.6  # Conservative
                max_imbalance = 0.2

            else:  # Final phase (0-120s remaining)
                edge = self.default_edge_required * 2  # Much higher edge
                max_size = self.default_max_size * 0.3  # Small size only
                stop = idx >= (self.num_buckets - self.stop_trading_last_n_buckets)
                weight = 0.3 if not stop else 0
                max_imbalance = 0.1

            policy = BucketPolicy(
                bucket_index=idx,
                time_remaining_start=time_start,
                time_remaining_end=time_end,
                edge_required=edge,
                max_spread_allowed=self.default_max_spread,
                min_depth_required=self.default_min_depth,
                max_size=max_size,
                stop_trading=stop,
                max_imbalance=max_imbalance,
                weight=weight,
            )

            self.bucket_policies[idx] = policy

    def get_bucket_index(self, time_remaining_sec: float) -> int:
        """Get bucket index for given time remaining."""
        if time_remaining_sec >= self.total_time_sec:
            return 0
        if time_remaining_sec <= 0:
            return self.num_buckets - 1
        elapsed = self.total_time_sec - time_remaining_sec
        bucket = int(elapsed // self.bucket_size_sec)
        return min(bucket, self.num_buckets - 1)

    def get_policy(self, time_remaining_sec: float) -> BucketPolicy:
        """Get policy for given time remaining."""
        bucket_idx = self.get_bucket_index(time_remaining_sec)
        return self.bucket_policies.get(bucket_idx, self._get_default_policy(bucket_idx))

    def _get_default_policy(self, bucket_idx: int) -> BucketPolicy:
        """Get default policy for bucket."""
        time_start = self.total_time_sec - (bucket_idx * self.bucket_size_sec)
        time_end = max(0, time_start - self.bucket_size_sec)

        return BucketPolicy(
            bucket_index=bucket_idx,
            time_remaining_start=time_start,
            time_remaining_end=time_end,
            edge_required=self.default_edge_required,
            max_spread_allowed=self.default_max_spread,
            min_depth_required=self.default_min_depth,
            max_size=self.default_max_size,
            max_imbalance=self.default_max_imbalance,
        )

    def should_trade(
        self,
        time_remaining_sec: float,
        pair_cost: float,
        spread: Optional[float] = None,
        depth: Optional[float] = None,
        current_imbalance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate if should trade based on LTM policy.

        Returns dict with:
        - should_trade: bool
        - reason: str (if not trading)
        - policy: BucketPolicy used
        - adjusted_size: recommended size
        """
        policy = self.get_policy(time_remaining_sec)

        result = {
            'should_trade': False,
            'reason': None,
            'policy': policy,
            'adjusted_size': 0,
            'edge_available': 1.0 - pair_cost,
            'bucket_index': policy.bucket_index,
        }

        # Check stop trading
        if policy.stop_trading:
            result['reason'] = f"Stop trading in bucket {policy.bucket_index}"
            return result

        # Check time threshold
        if time_remaining_sec < self.stop_trading_threshold_sec:
            result['reason'] = f"Time remaining ({time_remaining_sec:.0f}s) below threshold ({self.stop_trading_threshold_sec}s)"
            return result

        # Check edge
        edge_available = 1.0 - pair_cost
        if edge_available < policy.edge_required:
            result['reason'] = f"Edge {edge_available:.4f} < required {policy.edge_required:.4f}"
            return result

        # Check spread (if provided)
        if spread is not None and abs(spread) > policy.max_spread_allowed:
            result['reason'] = f"Spread {abs(spread):.4f} > max allowed {policy.max_spread_allowed:.4f}"
            return result

        # Check depth (if provided)
        if depth is not None and depth < policy.min_depth_required:
            result['reason'] = f"Depth {depth:.0f} < required {policy.min_depth_required:.0f}"
            return result

        # Check imbalance
        if abs(current_imbalance) > policy.max_imbalance:
            result['reason'] = f"Imbalance {abs(current_imbalance):.2f} > max {policy.max_imbalance:.2f}"
            return result

        # Calculate adjusted size
        base_size = policy.max_size
        size_by_weight = base_size * policy.weight
        size_by_edge = base_size * (edge_available / policy.edge_required)  # Scale by edge quality
        adjusted_size = min(size_by_weight, size_by_edge, policy.max_size)

        result['should_trade'] = True
        result['adjusted_size'] = adjusted_size

        return result

    def update_from_features(self, features: LTMFeatures) -> None:
        """Update policy parameters based on observed features."""
        for bucket_idx, stats in features.bucket_stats.items():
            if bucket_idx in self.bucket_policies:
                policy = self.bucket_policies[bucket_idx]

                # Adjust max_spread based on observed p90
                if stats.spread_p90 > 0:
                    policy.max_spread_allowed = min(
                        self.default_max_spread,
                        stats.spread_p90 * 1.2  # Allow 20% buffer
                    )

                # Adjust min_depth based on observed p10
                if stats.depth_p10 > 0:
                    policy.min_depth_required = stats.depth_p10 * 0.8  # 20% buffer

                # Adjust min_fill_rate based on observed p10
                if stats.fill_rate_p10 < 1:
                    policy.min_fill_rate_required = stats.fill_rate_p10 * 0.9

                # Adjust edge_required based on slippage
                if stats.slippage_p90 > 0:
                    # Need edge > slippage to be profitable
                    min_edge = stats.slippage_p90 * 2  # 2x buffer for slippage
                    policy.edge_required = max(policy.edge_required, min_edge)

                # Mark bucket for stop if conditions are bad
                if (stats.fill_rate_mean < 0.5 or
                    stats.slippage_p90 > 0.05 or
                    stats.liquidity_score_mean < 10):
                    policy.stop_trading = True
                    policy.weight = 0

        logger.info(f"Updated policy from features for {len(self.bucket_policies)} buckets")

    def save(self, filepath: str) -> None:
        """Save policy to YAML file."""
        data = {
            'metadata': {
                'num_buckets': self.num_buckets,
                'bucket_size_sec': self.bucket_size_sec,
                'total_time_sec': self.total_time_sec,
            },
            'defaults': {
                'edge_required': self.default_edge_required,
                'max_spread': self.default_max_spread,
                'min_depth': self.default_min_depth,
                'max_size': self.default_max_size,
                'max_imbalance': self.default_max_imbalance,
            },
            'stop_settings': {
                'last_n_buckets': self.stop_trading_last_n_buckets,
                'threshold_sec': self.stop_trading_threshold_sec,
            },
            'buckets': {
                idx: policy.to_dict()
                for idx, policy in self.bucket_policies.items()
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved policy to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LTMPolicy':
        """Load policy from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        meta = data.get('metadata', {})
        defaults = data.get('defaults', {})
        stop_settings = data.get('stop_settings', {})

        policy = cls(
            num_buckets=meta.get('num_buckets', 15),
            bucket_size_sec=meta.get('bucket_size_sec', 60),
            total_time_sec=meta.get('total_time_sec', 900),
            default_edge_required=defaults.get('edge_required', 0.02),
            default_max_spread=defaults.get('max_spread', 0.05),
            default_min_depth=defaults.get('min_depth', 100),
            default_max_size=defaults.get('max_size', 100),
            default_max_imbalance=defaults.get('max_imbalance', 0.3),
            stop_trading_last_n_buckets=stop_settings.get('last_n_buckets', 1),
            stop_trading_threshold_sec=stop_settings.get('threshold_sec', 60),
        )

        # Load bucket policies
        buckets = data.get('buckets', {})
        for idx_str, bucket_data in buckets.items():
            idx = int(idx_str)
            policy.bucket_policies[idx] = BucketPolicy.from_dict(bucket_data)

        logger.info(f"Loaded policy from {filepath}")
        return policy

    def generate_report(self) -> str:
        """Generate human-readable policy report."""
        lines = [
            "=" * 60,
            "LTM POLICY REPORT",
            "=" * 60,
            "",
            "GLOBAL SETTINGS",
            f"  Buckets: {self.num_buckets} x {self.bucket_size_sec}s = {self.total_time_sec}s",
            f"  Default edge required: {self.default_edge_required:.2%}",
            f"  Default max spread: {self.default_max_spread:.2%}",
            f"  Stop trading last {self.stop_trading_last_n_buckets} buckets",
            f"  Stop below {self.stop_trading_threshold_sec}s remaining",
            "",
            "-" * 60,
            "BUCKET POLICIES",
            "-" * 60,
            "",
        ]

        for idx in range(self.num_buckets):
            p = self.bucket_policies.get(idx)
            if p:
                status = "STOP" if p.stop_trading else f"weight={p.weight:.1f}"
                lines.append(
                    f"Bucket {idx:2d} ({p.time_remaining_end:.0f}-{p.time_remaining_start:.0f}s): "
                    f"edge={p.edge_required:.2%}, size={p.max_size:.0f}, imbal={p.max_imbalance:.0%} [{status}]"
                )

        return "\n".join(lines)
