"""
LTM Adapter for Real-Time Trading

Adapts the LTM (Liquidity Time Model) for real-time trading decisions.
This is a lightweight adapter that wraps LTMPolicy for use in the trading bot.

Key features:
- Time-bucket based thresholds
- Dynamic target_pair_cost adjustment per bucket
- Order size scaling by market phase
- Stop trading in final buckets
- Pair-cost decay tracking
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .ltm import LTMPolicy, PairCostDecay

logger = logging.getLogger(__name__)


@dataclass
class LTMTradeDecision:
    """Result of LTM trade evaluation."""
    should_trade: bool
    reason: str
    bucket_index: int
    time_remaining_sec: float

    # Adjusted parameters
    target_pair_cost: float  # Adjusted threshold (1 - edge_required)
    order_size_multiplier: float  # Scale factor for order size

    # Additional context
    edge_required: float
    edge_available: float
    decay_recommendation: str = 'unknown'


class LTMAdapter:
    """
    Adapter to integrate LTM into real-time trading bot.

    Usage:
        adapter = LTMAdapter()
        decision = adapter.evaluate(time_remaining_sec, pair_cost)
        if decision.should_trade:
            adjusted_threshold = decision.target_pair_cost
            size_multiplier = decision.order_size_multiplier
    """

    def __init__(
        self,
        policy_path: str = None,
        use_decay_model: bool = True,
        base_edge: float = 0.02,  # Base edge requirement (2%)
        base_order_size: float = 50.0,  # Base order size
    ):
        """
        Initialize LTM Adapter.

        Args:
            policy_path: Path to LTM policy YAML file
            use_decay_model: Whether to track pair-cost decay
            base_edge: Base edge requirement (used as reference)
            base_order_size: Base order size (for calculating multipliers)
        """
        self.policy_path = policy_path
        self.use_decay_model = use_decay_model
        self.base_edge = base_edge
        self.base_order_size = base_order_size

        # Load or create policy
        self.policy = self._load_policy()

        # Initialize decay model if enabled
        self.decay_model = PairCostDecay() if use_decay_model else None

        # Track stats for logging
        self.trades_allowed = 0
        self.trades_blocked = 0
        self.block_reasons: Dict[str, int] = {}

        logger.info(f"LTM Adapter initialized (buckets={self.policy.num_buckets})")

    def _load_policy(self) -> LTMPolicy:
        """Load LTM policy from file or create default."""
        if self.policy_path and os.path.exists(self.policy_path):
            logger.info(f"Loading LTM policy from {self.policy_path}")
            return LTMPolicy.load(self.policy_path)
        else:
            # Try common paths
            common_paths = [
                'ltm_policy.yaml',
                'config/ltm_policy.yaml',
                '../ltm_policy.yaml',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    logger.info(f"Loading LTM policy from {path}")
                    return LTMPolicy.load(path)

            logger.info("Creating default LTM policy")
            return LTMPolicy()

    def get_time_remaining_sec(self, time_str: str) -> float:
        """
        Convert time remaining string to seconds.

        Args:
            time_str: Time string like "14m 30s" or "CLOSED"

        Returns:
            Time remaining in seconds (0 if closed or invalid)
        """
        if time_str == "CLOSED" or time_str == "Unknown":
            return 0

        try:
            parts = time_str.split()
            seconds = 0
            for part in parts:
                if part.endswith('m'):
                    seconds += int(part[:-1]) * 60
                elif part.endswith('s'):
                    seconds += int(part[:-1])
            return float(seconds)
        except Exception:
            return 0

    def evaluate(
        self,
        time_remaining_sec: float,
        pair_cost: float,
        depth: Optional[float] = None,
        market_id: str = None,
    ) -> LTMTradeDecision:
        """
        Evaluate trade opportunity using LTM model.

        Args:
            time_remaining_sec: Seconds until market closes
            pair_cost: Current pair cost (UP + DOWN prices)
            depth: Available liquidity depth (optional)
            market_id: Market identifier for decay tracking (optional)

        Returns:
            LTMTradeDecision with trading recommendation
        """
        # Calculate edge
        edge_available = 1.0 - pair_cost
        spread = pair_cost - 1.0  # Negative when there's edge

        # Get LTM policy evaluation
        ltm_result = self.policy.should_trade(
            time_remaining_sec=time_remaining_sec,
            pair_cost=pair_cost,
            spread=spread,
            depth=depth,
        )

        policy = ltm_result['policy']
        should_trade = ltm_result['should_trade']
        reason = ltm_result.get('reason', '')

        # Calculate adjusted parameters
        # target_pair_cost = 1.0 - edge_required
        # This is the maximum pair_cost we're willing to pay
        target_pair_cost = 1.0 - policy.edge_required

        # Order size multiplier based on policy weight and edge quality
        base_multiplier = policy.weight
        if should_trade:
            edge_quality = edge_available / policy.edge_required
            edge_multiplier = min(edge_quality, 1.5)  # Cap at 1.5x
            order_size_multiplier = base_multiplier * edge_multiplier * (policy.max_size / self.base_order_size)
        else:
            order_size_multiplier = 0.0

        # Create decision
        decision = LTMTradeDecision(
            should_trade=should_trade,
            reason=reason,
            bucket_index=policy.bucket_index,
            time_remaining_sec=time_remaining_sec,
            target_pair_cost=target_pair_cost,
            order_size_multiplier=order_size_multiplier,
            edge_required=policy.edge_required,
            edge_available=edge_available,
        )

        # Track decay if enabled
        if self.use_decay_model and self.decay_model and market_id:
            self.decay_model.add_observation(
                market_id=market_id,
                timestamp=None,  # Will use current time
                pair_cost=pair_cost,
                time_remaining_sec=time_remaining_sec,
            )

            decay_metrics = self.decay_model.analyze(
                market_id=market_id,
                current_pair_cost=pair_cost,
                time_remaining_sec=time_remaining_sec,
            )
            decision.decay_recommendation = decay_metrics.recommendation

            # Override decision if decay suggests skip
            if decay_metrics.recommendation == 'skip' and decision.should_trade:
                decision.should_trade = False
                decision.reason = f"Decay model: {decay_metrics.reason}"

        # Track stats
        if decision.should_trade:
            self.trades_allowed += 1
        else:
            self.trades_blocked += 1
            reason_key = decision.reason[:30] if decision.reason else 'unknown'
            self.block_reasons[reason_key] = self.block_reasons.get(reason_key, 0) + 1

        return decision

    def evaluate_from_time_string(
        self,
        time_remaining_str: str,
        pair_cost: float,
        depth: Optional[float] = None,
        market_id: str = None,
    ) -> LTMTradeDecision:
        """
        Evaluate using time remaining as string (convenience method).

        Args:
            time_remaining_str: Time string like "14m 30s"
            pair_cost: Current pair cost
            depth: Available depth (optional)
            market_id: Market ID for decay tracking (optional)
        """
        time_sec = self.get_time_remaining_sec(time_remaining_str)
        return self.evaluate(time_sec, pair_cost, depth, market_id)

    def should_stop_trading(self, time_remaining_sec: float) -> bool:
        """Check if should stop trading based on time remaining."""
        policy = self.policy.get_policy(time_remaining_sec)
        return policy.stop_trading

    def get_adjusted_threshold(self, time_remaining_sec: float) -> float:
        """
        Get adjusted target_pair_cost for given time remaining.

        Returns the maximum pair_cost we should pay (1.0 - edge_required).
        """
        policy = self.policy.get_policy(time_remaining_sec)
        return 1.0 - policy.edge_required

    def get_size_multiplier(self, time_remaining_sec: float) -> float:
        """Get order size multiplier for given time remaining."""
        policy = self.policy.get_policy(time_remaining_sec)
        return policy.weight * (policy.max_size / self.base_order_size)

    def get_bucket_info(self, time_remaining_sec: float) -> Dict[str, Any]:
        """Get information about current bucket."""
        policy = self.policy.get_policy(time_remaining_sec)
        return {
            'bucket_index': policy.bucket_index,
            'edge_required': policy.edge_required,
            'max_size': policy.max_size,
            'weight': policy.weight,
            'stop_trading': policy.stop_trading,
            'max_imbalance': policy.max_imbalance,
            'phase': self._get_phase_name(policy.bucket_index),
        }

    def _get_phase_name(self, bucket_index: int) -> str:
        """Get human-readable phase name for bucket."""
        if bucket_index <= 4:
            return "EARLY"
        elif bucket_index <= 10:
            return "MIDDLE"
        elif bucket_index <= 12:
            return "LATE"
        else:
            return "FINAL"

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        total = self.trades_allowed + self.trades_blocked
        return {
            'trades_allowed': self.trades_allowed,
            'trades_blocked': self.trades_blocked,
            'allow_rate': self.trades_allowed / total if total > 0 else 0,
            'block_reasons': dict(self.block_reasons),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.trades_allowed = 0
        self.trades_blocked = 0
        self.block_reasons = {}

    def generate_report(self) -> str:
        """Generate status report."""
        stats = self.get_stats()
        lines = [
            "=" * 50,
            "LTM ADAPTER STATUS",
            "=" * 50,
            f"Trades allowed: {stats['trades_allowed']}",
            f"Trades blocked: {stats['trades_blocked']}",
            f"Allow rate: {stats['allow_rate']:.1%}",
            "",
            "Block reasons:",
        ]
        for reason, count in stats['block_reasons'].items():
            lines.append(f"  {reason}: {count}")

        lines.extend([
            "",
            "-" * 50,
            "BUCKET THRESHOLDS",
            "-" * 50,
        ])
        for i in range(self.policy.num_buckets):
            policy = self.policy.bucket_policies.get(i)
            if policy:
                phase = self._get_phase_name(i)
                status = "STOP" if policy.stop_trading else f"edge={policy.edge_required:.2%}"
                lines.append(f"  Bucket {i:2d} [{phase:6s}]: {status}, weight={policy.weight:.1f}")

        return "\n".join(lines)
