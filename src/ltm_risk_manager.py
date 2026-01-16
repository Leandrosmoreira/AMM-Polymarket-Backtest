"""
LTM-Enhanced Risk Manager

Extends the base RiskManager with Liquidity Time Model integration:
- Time-bucket based edge requirements
- Dynamic position sizing by market phase
- Pair-cost decay analysis
- Intelligent stop trading rules
- Position imbalance limits by bucket
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
import os

from config.risk_params import RiskParams
from .position_manager import Position, Portfolio, MarketState
from .risk_manager import RiskManager
from .ltm import (
    LTMPolicy,
    LTMCollector,
    PairCostDecay,
    LTMBanditManager,
)

logger = logging.getLogger(__name__)

# Default policy path
DEFAULT_LTM_POLICY_PATH = 'config/ltm_policy.yaml'


@dataclass
class LTMDecision:
    """Decision result from LTM evaluation."""

    should_trade: bool
    reason: str
    bucket_index: int
    time_remaining: float

    # LTM policy values
    edge_required: float
    edge_available: float
    max_size: float
    adjusted_size: float

    # Decay analysis
    decay_recommendation: str = 'unknown'
    eta_to_target: float = -1

    # Bandit-suggested params (if enabled)
    bandit_params: Dict[str, float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'should_trade': self.should_trade,
            'reason': self.reason,
            'bucket_index': self.bucket_index,
            'time_remaining': self.time_remaining,
            'edge_required': self.edge_required,
            'edge_available': self.edge_available,
            'max_size': self.max_size,
            'adjusted_size': self.adjusted_size,
            'decay_recommendation': self.decay_recommendation,
            'eta_to_target': self.eta_to_target,
            'bandit_params': self.bandit_params,
        }


class LTMRiskManager(RiskManager):
    """
    Risk manager with LTM (Liquidity Time Model) integration.

    Extends base RiskManager to use:
    - Time-bucket based parameters
    - Pair-cost decay analysis
    - Optional bandit-based auto-tuning
    """

    def __init__(
        self,
        risk_params: RiskParams = None,
        ltm_policy_path: str = None,
        use_decay_model: bool = True,
        use_bandit: bool = False,
        bandit_path: str = None,
    ):
        """
        Initialize LTM Risk Manager.

        Args:
            risk_params: Base risk parameters
            ltm_policy_path: Path to LTM policy YAML
            use_decay_model: Whether to use pair-cost decay analysis
            use_bandit: Whether to use bandit auto-tuning
            bandit_path: Path to load/save bandit state
        """
        super().__init__(risk_params)

        # Load LTM policy
        self.ltm_policy = self._load_policy(ltm_policy_path)

        # Initialize components
        self.collector = LTMCollector()
        self.use_decay_model = use_decay_model
        self.decay_model = PairCostDecay() if use_decay_model else None

        # Bandit setup
        self.use_bandit = use_bandit
        self.bandit = None
        self.bandit_path = bandit_path
        if use_bandit:
            self.bandit = self._load_bandit(bandit_path)

        # Track recent decisions for analysis
        self.recent_decisions: list = []
        self.max_recent_decisions = 100

    def _load_policy(self, path: str = None) -> LTMPolicy:
        """Load LTM policy from file or create default."""
        policy_path = path or DEFAULT_LTM_POLICY_PATH

        if os.path.exists(policy_path):
            logger.info(f"Loading LTM policy from {policy_path}")
            return LTMPolicy.load(policy_path)
        else:
            logger.info("Creating default LTM policy")
            return LTMPolicy()

    def _load_bandit(self, path: str = None) -> LTMBanditManager:
        """Load bandit state or create new."""
        if path and os.path.exists(path):
            logger.info(f"Loading bandit state from {path}")
            return LTMBanditManager.load(path)
        else:
            logger.info("Creating new LTM bandit manager")
            return LTMBanditManager()

    def evaluate_ltm(
        self,
        market_state: MarketState,
        portfolio: Portfolio,
        current_imbalance: float = 0.0,
    ) -> LTMDecision:
        """
        Evaluate trade using LTM model.

        Returns comprehensive decision with all LTM factors.
        """
        time_remaining = market_state.time_remaining
        pair_cost = market_state.price_yes + market_state.price_no
        spread = pair_cost - 1.0

        # Get LTM policy for this bucket
        ltm_result = self.ltm_policy.should_trade(
            time_remaining_sec=time_remaining,
            pair_cost=pair_cost,
            spread=spread,
            current_imbalance=current_imbalance,
        )

        policy = ltm_result['policy']

        # Base decision from LTM policy
        decision = LTMDecision(
            should_trade=ltm_result['should_trade'],
            reason=ltm_result.get('reason', ''),
            bucket_index=policy.bucket_index,
            time_remaining=time_remaining,
            edge_required=policy.edge_required,
            edge_available=ltm_result.get('edge_available', 1.0 - pair_cost),
            max_size=policy.max_size,
            adjusted_size=ltm_result.get('adjusted_size', 0),
        )

        # Enhance with decay model
        if self.use_decay_model and self.decay_model:
            self.decay_model.add_observation(
                market_id=market_state.market_id,
                timestamp=market_state.current_time,
                pair_cost=pair_cost,
                time_remaining_sec=time_remaining,
            )

            decay_metrics = self.decay_model.analyze(
                market_id=market_state.market_id,
                current_pair_cost=pair_cost,
                time_remaining_sec=time_remaining,
            )

            decision.decay_recommendation = decay_metrics.recommendation
            decision.eta_to_target = decay_metrics.estimated_time_to_target_sec

            # Override decision if decay says skip
            if decay_metrics.recommendation == 'skip' and decision.should_trade:
                decision.should_trade = False
                decision.reason = f"Decay model: {decay_metrics.reason}"

        # Enhance with bandit suggestions
        if self.use_bandit and self.bandit:
            bandit_params = self.bandit.select_params(time_remaining)
            decision.bandit_params = bandit_params

            # Use bandit edge if more conservative
            if bandit_params['edge_required'] > policy.edge_required:
                if decision.edge_available < bandit_params['edge_required']:
                    decision.should_trade = False
                    decision.reason = f"Bandit: edge {decision.edge_available:.4f} < {bandit_params['edge_required']:.4f}"

            # Use bandit max_size if smaller
            if bandit_params['max_size'] < decision.adjusted_size:
                decision.adjusted_size = bandit_params['max_size']

        # Track decision
        self._track_decision(decision)

        return decision

    def should_enter(
        self,
        market_state: MarketState,
        portfolio: Portfolio
    ) -> bool:
        """
        Check if should enter a market position (LTM-enhanced).

        Overrides base method to use LTM evaluation.
        """
        # First, run base checks
        base_result = super().should_enter(market_state, portfolio)

        if not base_result:
            return False

        # Then run LTM evaluation
        ltm_decision = self.evaluate_ltm(market_state, portfolio)

        return ltm_decision.should_trade

    def calculate_order_size(
        self,
        market_state: MarketState,
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """
        Calculate order size with LTM adjustments.

        Overrides base method to use bucket-specific sizing.
        """
        # Get base sizing
        base_sizing = super().calculate_order_size(market_state, portfolio)

        if base_sizing['total_cost'] == 0:
            return base_sizing

        # Get LTM decision for size adjustment
        ltm_decision = self.evaluate_ltm(market_state, portfolio)

        # Adjust size based on LTM
        if ltm_decision.adjusted_size > 0:
            # Scale down if LTM suggests smaller size
            size_ratio = ltm_decision.adjusted_size / self.params.MAX_PER_MARKET_USD

            if size_ratio < 1.0:
                base_sizing['shares_yes'] = int(base_sizing['shares_yes'] * size_ratio)
                base_sizing['shares_no'] = int(base_sizing['shares_no'] * size_ratio)
                base_sizing['cost_yes'] *= size_ratio
                base_sizing['cost_no'] *= size_ratio
                base_sizing['total_cost'] *= size_ratio

        return base_sizing

    def get_dynamic_stop(self, market_state: MarketState) -> bool:
        """
        Check if should stop trading based on LTM policy.

        Uses bucket-specific stop rules instead of fixed time threshold.
        """
        time_remaining = market_state.time_remaining
        policy = self.ltm_policy.get_policy(time_remaining)

        return policy.stop_trading

    def get_dynamic_imbalance_limit(self, market_state: MarketState) -> float:
        """
        Get position imbalance limit for current bucket.

        Tighter limits near market end.
        """
        time_remaining = market_state.time_remaining
        policy = self.ltm_policy.get_policy(time_remaining)

        return policy.max_imbalance

    def should_rebalance(
        self,
        position: Position,
        market_state: MarketState
    ) -> bool:
        """
        Check if position needs rebalancing (LTM-enhanced).

        Uses bucket-specific imbalance limits.
        """
        ratio = position.ratio
        max_imbalance = self.get_dynamic_imbalance_limit(market_state)

        # Calculate current imbalance
        if ratio == 1.0:
            imbalance = 0.0
        else:
            imbalance = abs(ratio - 1.0) / (ratio + 1.0)

        if imbalance > max_imbalance:
            return True

        # Also check base rebalancing rules
        return super().should_rebalance(position, market_state)

    def update_bandit(
        self,
        time_remaining: float,
        params_used: Dict[str, float],
        pnl: float,
        slippage: float = 0.0,
        unhedged_time: float = 0.0,
        fill_failures: int = 0,
    ) -> None:
        """Update bandit with trade result."""
        if self.use_bandit and self.bandit:
            self.bandit.update(
                time_remaining_sec=time_remaining,
                params_used=params_used,
                pnl=pnl,
                slippage=slippage,
                unhedged_time_sec=unhedged_time,
                fill_failures=fill_failures,
            )

    def save_bandit_state(self) -> None:
        """Save current bandit state."""
        if self.bandit and self.bandit_path:
            self.bandit.save(self.bandit_path)

    def _track_decision(self, decision: LTMDecision) -> None:
        """Track recent decisions for analysis."""
        self.recent_decisions.append(decision.to_dict())
        if len(self.recent_decisions) > self.max_recent_decisions:
            self.recent_decisions.pop(0)

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get statistics on recent decisions."""
        if not self.recent_decisions:
            return {}

        trades = [d for d in self.recent_decisions if d['should_trade']]
        skips = [d for d in self.recent_decisions if not d['should_trade']]

        # Group skips by reason
        skip_reasons = {}
        for d in skips:
            reason = d.get('reason', 'unknown')
            # Truncate for grouping
            key = reason[:50] if reason else 'unknown'
            skip_reasons[key] = skip_reasons.get(key, 0) + 1

        # Group by bucket
        by_bucket = {}
        for d in self.recent_decisions:
            bucket = d['bucket_index']
            if bucket not in by_bucket:
                by_bucket[bucket] = {'trades': 0, 'skips': 0}
            if d['should_trade']:
                by_bucket[bucket]['trades'] += 1
            else:
                by_bucket[bucket]['skips'] += 1

        return {
            'total_decisions': len(self.recent_decisions),
            'trades': len(trades),
            'skips': len(skips),
            'trade_rate': len(trades) / len(self.recent_decisions) if self.recent_decisions else 0,
            'skip_reasons': skip_reasons,
            'by_bucket': by_bucket,
        }

    def collect_snapshot(
        self,
        market_state: MarketState,
        fill_rate: float = 1.0,
        slippage: float = 0.0,
    ) -> None:
        """Collect LTM snapshot for analysis."""
        self.collector.collect_from_market_state(
            market_state=market_state,
            fill_rate=fill_rate,
            slippage=slippage,
        )

    def save_snapshots(self, filepath: str) -> None:
        """Save collected snapshots."""
        self.collector.save_snapshots(filepath)

    def generate_report(self) -> str:
        """Generate comprehensive LTM status report."""
        lines = [
            "=" * 60,
            "LTM RISK MANAGER STATUS",
            "=" * 60,
            "",
            "CONFIGURATION",
            f"  Use decay model: {self.use_decay_model}",
            f"  Use bandit: {self.use_bandit}",
            f"  Collected snapshots: {len(self.collector.snapshots)}",
            "",
        ]

        # Decision stats
        stats = self.get_decision_stats()
        if stats:
            lines.extend([
                "RECENT DECISIONS",
                f"  Total: {stats['total_decisions']}",
                f"  Trades: {stats['trades']} ({stats['trade_rate']:.1%})",
                f"  Skips: {stats['skips']}",
                "",
                "SKIP REASONS:",
            ])
            for reason, count in stats.get('skip_reasons', {}).items():
                lines.append(f"  {reason}: {count}")

            lines.extend([
                "",
                "BY BUCKET:",
            ])
            for bucket, data in sorted(stats.get('by_bucket', {}).items()):
                lines.append(f"  Bucket {bucket}: {data['trades']} trades, {data['skips']} skips")

        # Bandit status
        if self.use_bandit and self.bandit:
            lines.extend([
                "",
                "-" * 60,
                "BANDIT STATUS",
                "-" * 60,
            ])
            best_policy = self.bandit.get_best_policy()
            for bucket, params in sorted(best_policy.items()):
                lines.append(
                    f"  Bucket {bucket}: edge={params['edge_required']:.3f}, "
                    f"size={params['max_size']:.0f}"
                )

        return "\n".join(lines)
