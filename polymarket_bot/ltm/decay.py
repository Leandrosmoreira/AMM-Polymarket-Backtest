"""
LTM Decay - Models pair-cost decay speed and estimates time to target

Measures how quickly pair_cost improves over time and uses this to:
- Estimate time to reach target pair_cost
- Decide if entry is viable given remaining time
- Reduce trades that can't "close the pair" within the window
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DecayMetrics:
    """Metrics for pair-cost decay analysis."""

    # Current state
    current_pair_cost: float
    target_pair_cost: float
    time_remaining_sec: float

    # Decay measurements
    decay_rate_per_sec: float  # Rate of pair_cost improvement
    decay_rate_per_min: float

    # Estimated time to target
    estimated_time_to_target_sec: float
    can_reach_target: bool  # Can reach target before expiry?

    # Confidence metrics
    samples_used: int
    r_squared: float  # Goodness of fit for decay estimate

    # Recommendation
    recommendation: str  # 'enter', 'wait', 'skip'
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_pair_cost': self.current_pair_cost,
            'target_pair_cost': self.target_pair_cost,
            'time_remaining_sec': self.time_remaining_sec,
            'decay_rate_per_sec': self.decay_rate_per_sec,
            'decay_rate_per_min': self.decay_rate_per_min,
            'estimated_time_to_target_sec': self.estimated_time_to_target_sec,
            'can_reach_target': self.can_reach_target,
            'samples_used': self.samples_used,
            'r_squared': self.r_squared,
            'recommendation': self.recommendation,
            'reason': self.reason,
        }


@dataclass
class PairCostDecay:
    """
    Tracks and models pair-cost decay over time.

    Uses a rolling window of observations to estimate:
    - Current decay rate
    - Time to reach target
    - Probability of filling both legs
    """

    # Configuration
    window_size: int = 60  # Number of samples to keep
    min_samples: int = 5   # Minimum samples for valid estimate
    target_pair_cost: float = 0.97  # Target pair_cost (edge = 0.03)
    safety_margin_sec: float = 60   # Don't enter if ETA + margin > time_remaining

    # State
    observations: deque = field(default_factory=lambda: deque(maxlen=60))
    market_observations: Dict[str, deque] = field(default_factory=dict)

    def __post_init__(self):
        self.observations = deque(maxlen=self.window_size)

    def add_observation(
        self,
        market_id: str,
        timestamp: datetime,
        pair_cost: float,
        time_remaining_sec: float
    ) -> None:
        """Add a new observation."""
        obs = {
            'market_id': market_id,
            'timestamp': timestamp,
            'pair_cost': pair_cost,
            'time_remaining': time_remaining_sec,
            'elapsed': 900 - time_remaining_sec,  # Assuming 15-min markets
        }

        self.observations.append(obs)

        # Also track per-market
        if market_id not in self.market_observations:
            self.market_observations[market_id] = deque(maxlen=self.window_size)
        self.market_observations[market_id].append(obs)

    def get_decay_rate(
        self,
        market_id: Optional[str] = None,
        lookback_samples: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """
        Calculate decay rate from observations.

        Returns:
            (decay_rate_per_sec, r_squared, samples_used)
        """
        if market_id and market_id in self.market_observations:
            obs_list = list(self.market_observations[market_id])
        else:
            obs_list = list(self.observations)

        if lookback_samples:
            obs_list = obs_list[-lookback_samples:]

        if len(obs_list) < self.min_samples:
            return 0.0, 0.0, len(obs_list)

        # Extract time and pair_cost arrays
        times = np.array([o['elapsed'] for o in obs_list])
        costs = np.array([o['pair_cost'] for o in obs_list])

        # Simple linear regression: pair_cost = a + b * elapsed_time
        # Negative b means pair_cost decreases over time (good)
        try:
            A = np.vstack([times, np.ones(len(times))]).T
            result = np.linalg.lstsq(A, costs, rcond=None)
            slope, intercept = result[0]

            # Calculate R-squared
            residuals = costs - (slope * times + intercept)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((costs - np.mean(costs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Decay rate = -slope (positive if pair_cost decreasing)
            decay_rate = -slope

            return decay_rate, max(0, r_squared), len(obs_list)

        except Exception as e:
            logger.warning(f"Failed to calculate decay rate: {e}")
            return 0.0, 0.0, len(obs_list)

    def estimate_time_to_target(
        self,
        current_pair_cost: float,
        decay_rate: float,
    ) -> float:
        """
        Estimate seconds until pair_cost reaches target.

        Returns float('inf') if decay is zero or positive (getting worse).
        """
        if decay_rate <= 0:
            return float('inf')

        cost_diff = current_pair_cost - self.target_pair_cost

        if cost_diff <= 0:
            # Already at or below target
            return 0

        return cost_diff / decay_rate

    def analyze(
        self,
        market_id: str,
        current_pair_cost: float,
        time_remaining_sec: float,
    ) -> DecayMetrics:
        """
        Full analysis of whether to enter based on decay model.
        """
        # Get decay rate
        decay_rate, r_squared, samples = self.get_decay_rate(market_id)

        # Convert to per-minute for readability
        decay_rate_per_min = decay_rate * 60

        # Estimate time to target
        eta_sec = self.estimate_time_to_target(current_pair_cost, decay_rate)

        # Can we reach target before expiry?
        time_available = time_remaining_sec - self.safety_margin_sec
        can_reach = eta_sec < time_available if eta_sec != float('inf') else False

        # Make recommendation
        if current_pair_cost <= self.target_pair_cost:
            recommendation = 'enter'
            reason = f"Already at target: {current_pair_cost:.4f} <= {self.target_pair_cost:.4f}"
        elif samples < self.min_samples:
            recommendation = 'wait'
            reason = f"Insufficient data: {samples} < {self.min_samples} samples"
        elif decay_rate <= 0:
            recommendation = 'skip'
            reason = f"Pair cost not improving (decay={decay_rate:.6f})"
        elif not can_reach:
            recommendation = 'skip'
            reason = f"ETA {eta_sec:.0f}s > available {time_available:.0f}s"
        elif r_squared < 0.3:
            recommendation = 'wait'
            reason = f"Low confidence: R²={r_squared:.2f}"
        else:
            recommendation = 'enter'
            reason = f"Can reach target in {eta_sec:.0f}s (have {time_available:.0f}s)"

        return DecayMetrics(
            current_pair_cost=current_pair_cost,
            target_pair_cost=self.target_pair_cost,
            time_remaining_sec=time_remaining_sec,
            decay_rate_per_sec=decay_rate,
            decay_rate_per_min=decay_rate_per_min,
            estimated_time_to_target_sec=eta_sec if eta_sec != float('inf') else -1,
            can_reach_target=can_reach,
            samples_used=samples,
            r_squared=r_squared,
            recommendation=recommendation,
            reason=reason,
        )

    def should_enter(
        self,
        market_id: str,
        current_pair_cost: float,
        time_remaining_sec: float,
        min_confidence: float = 0.3,
    ) -> Tuple[bool, str]:
        """
        Quick check if should enter based on decay model.

        Returns:
            (should_enter, reason)
        """
        metrics = self.analyze(market_id, current_pair_cost, time_remaining_sec)

        should = metrics.recommendation == 'enter'
        return should, metrics.reason

    def get_adjusted_target(
        self,
        time_remaining_sec: float,
        base_target: float = 0.97
    ) -> float:
        """
        Get time-adjusted target pair_cost.

        Earlier in the market, can accept higher pair_cost (more time to improve).
        Later, need better price upfront.
        """
        # Percentage of market elapsed
        pct_elapsed = 1 - (time_remaining_sec / 900)

        # Adjustment factor: start at 1.02x target, end at 0.98x target
        # This means early: accept 0.97 * 1.02 = 0.989
        # Late: require 0.97 * 0.98 = 0.951
        adjustment = 1.02 - (0.04 * pct_elapsed)

        return base_target * adjustment

    def clear_market(self, market_id: str) -> None:
        """Clear observations for a specific market."""
        if market_id in self.market_observations:
            del self.market_observations[market_id]

    def clear_all(self) -> None:
        """Clear all observations."""
        self.observations.clear()
        self.market_observations.clear()

    def get_market_summary(self, market_id: str) -> Dict[str, Any]:
        """Get summary statistics for a market's decay."""
        if market_id not in self.market_observations:
            return {}

        obs = list(self.market_observations[market_id])
        if not obs:
            return {}

        costs = [o['pair_cost'] for o in obs]
        times = [o['time_remaining'] for o in obs]

        return {
            'market_id': market_id,
            'n_observations': len(obs),
            'pair_cost_start': costs[0],
            'pair_cost_current': costs[-1],
            'pair_cost_min': min(costs),
            'pair_cost_max': max(costs),
            'pair_cost_change': costs[-1] - costs[0],
            'time_start': times[0],
            'time_current': times[-1],
        }


def calculate_optimal_entry_time(
    current_pair_cost: float,
    target_pair_cost: float,
    decay_rate: float,
    time_remaining_sec: float,
    min_hold_time_sec: float = 60,
) -> Dict[str, Any]:
    """
    Calculate the optimal time to enter based on decay model.

    Balances:
    - Waiting for better price (lower pair_cost)
    - Risk of missing the opportunity
    - Minimum time needed in position
    """
    if decay_rate <= 0:
        return {
            'optimal_entry': 'now' if current_pair_cost <= target_pair_cost else 'skip',
            'wait_time': 0,
            'reason': 'No decay observed'
        }

    # Time to reach target at current decay rate
    time_to_target = (current_pair_cost - target_pair_cost) / decay_rate

    # Time available for holding
    time_for_holding = time_remaining_sec - min_hold_time_sec

    if current_pair_cost <= target_pair_cost:
        return {
            'optimal_entry': 'now',
            'wait_time': 0,
            'expected_pair_cost': current_pair_cost,
            'reason': 'Already at target'
        }

    if time_to_target > time_for_holding:
        return {
            'optimal_entry': 'skip',
            'wait_time': -1,
            'reason': f'Cannot reach target: need {time_to_target:.0f}s, have {time_for_holding:.0f}s'
        }

    # Optimal: wait until just enough time to reach target + margin
    optimal_wait = max(0, time_to_target - 30)  # Enter 30s before expected target
    expected_cost_at_entry = current_pair_cost - (decay_rate * optimal_wait)

    return {
        'optimal_entry': 'wait',
        'wait_time': optimal_wait,
        'expected_pair_cost': expected_cost_at_entry,
        'reason': f'Wait {optimal_wait:.0f}s for better entry'
    }
