"""
LTM Bandit - Multi-armed bandit for auto-tuning bucket parameters

Implements Thompson Sampling bandit for each bucket to learn optimal:
- edge_required
- max_size
- max_spread
- min_depth

Reward function:
pnl_net - slippage_penalty - unhedged_time_penalty - fail_fill_penalty
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistics for a single bandit arm."""

    arm_id: str
    successes: float = 1.0  # Beta prior alpha (start with 1 for uniform)
    failures: float = 1.0   # Beta prior beta
    total_pulls: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0

    def update(self, reward: float, success_threshold: float = 0.0) -> None:
        """Update arm statistics after receiving reward."""
        self.total_pulls += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.total_pulls

        # Convert reward to binary success/failure for Beta distribution
        if reward > success_threshold:
            self.successes += 1
        else:
            self.failures += 1

    def sample(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)."""
        return np.random.beta(self.successes, self.failures)

    def mean(self) -> float:
        """Get mean of Beta distribution."""
        return self.successes / (self.successes + self.failures)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'arm_id': self.arm_id,
            'successes': self.successes,
            'failures': self.failures,
            'total_pulls': self.total_pulls,
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArmStats':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BucketBandit:
    """Multi-armed bandit for a single time bucket."""

    bucket_index: int

    # Parameter candidates
    edge_candidates: List[float] = field(default_factory=lambda: [0.003, 0.005, 0.01, 0.015, 0.02, 0.03])
    size_candidates: List[float] = field(default_factory=lambda: [20, 40, 60, 80, 100, 150])
    spread_candidates: List[float] = field(default_factory=lambda: [0.02, 0.03, 0.04, 0.05])
    depth_candidates: List[float] = field(default_factory=lambda: [50, 100, 150, 200])

    # Arms for each parameter
    edge_arms: Dict[str, ArmStats] = field(default_factory=dict)
    size_arms: Dict[str, ArmStats] = field(default_factory=dict)
    spread_arms: Dict[str, ArmStats] = field(default_factory=dict)
    depth_arms: Dict[str, ArmStats] = field(default_factory=dict)

    # Exploration settings
    exploration_bonus: float = 0.1  # Bonus for under-explored arms
    min_pulls_for_exploitation: int = 10

    def __post_init__(self):
        """Initialize arms for all candidates."""
        self._init_arms('edge', self.edge_candidates, self.edge_arms)
        self._init_arms('size', self.size_candidates, self.size_arms)
        self._init_arms('spread', self.spread_candidates, self.spread_arms)
        self._init_arms('depth', self.depth_candidates, self.depth_arms)

    def _init_arms(self, param_name: str, candidates: List, arms_dict: Dict) -> None:
        """Initialize arms for a parameter."""
        for value in candidates:
            arm_id = f"{param_name}_{value}"
            if arm_id not in arms_dict:
                arms_dict[arm_id] = ArmStats(arm_id=arm_id)

    def select_params(self, use_thompson: bool = True) -> Dict[str, float]:
        """
        Select parameters using Thompson Sampling or greedy.

        Returns dict with selected values for each parameter.
        """
        if use_thompson:
            edge = self._thompson_select(self.edge_arms, self.edge_candidates)
            size = self._thompson_select(self.size_arms, self.size_candidates)
            spread = self._thompson_select(self.spread_arms, self.spread_candidates)
            depth = self._thompson_select(self.depth_arms, self.depth_candidates)
        else:
            edge = self._greedy_select(self.edge_arms, self.edge_candidates)
            size = self._greedy_select(self.size_arms, self.size_candidates)
            spread = self._greedy_select(self.spread_arms, self.spread_candidates)
            depth = self._greedy_select(self.depth_arms, self.depth_candidates)

        return {
            'edge_required': edge,
            'max_size': size,
            'max_spread_allowed': spread,
            'min_depth_required': depth,
        }

    def _thompson_select(self, arms: Dict[str, ArmStats], candidates: List) -> float:
        """Select using Thompson Sampling."""
        best_sample = -float('inf')
        best_value = candidates[0]

        for value in candidates:
            arm_id = f"{list(arms.keys())[0].split('_')[0]}_{value}"
            arm = arms.get(arm_id)
            if arm:
                sample = arm.sample()
                # Add exploration bonus for under-explored arms
                if arm.total_pulls < self.min_pulls_for_exploitation:
                    sample += self.exploration_bonus

                if sample > best_sample:
                    best_sample = sample
                    best_value = value

        return best_value

    def _greedy_select(self, arms: Dict[str, ArmStats], candidates: List) -> float:
        """Select greedily based on average reward."""
        best_avg = -float('inf')
        best_value = candidates[0]

        for value in candidates:
            arm_id = f"{list(arms.keys())[0].split('_')[0]}_{value}"
            arm = arms.get(arm_id)
            if arm and arm.avg_reward > best_avg:
                best_avg = arm.avg_reward
                best_value = value

        return best_value

    def update(
        self,
        params_used: Dict[str, float],
        reward: float,
        success_threshold: float = 0.0
    ) -> None:
        """Update arm statistics based on observed reward."""
        # Update each parameter's arm
        edge_id = f"edge_{params_used.get('edge_required')}"
        if edge_id in self.edge_arms:
            self.edge_arms[edge_id].update(reward, success_threshold)

        size_id = f"size_{params_used.get('max_size')}"
        if size_id in self.size_arms:
            self.size_arms[size_id].update(reward, success_threshold)

        spread_id = f"spread_{params_used.get('max_spread_allowed')}"
        if spread_id in self.spread_arms:
            self.spread_arms[spread_id].update(reward, success_threshold)

        depth_id = f"depth_{params_used.get('min_depth_required')}"
        if depth_id in self.depth_arms:
            self.depth_arms[depth_id].update(reward, success_threshold)

    def get_best_params(self) -> Dict[str, float]:
        """Get currently best parameters based on average reward."""
        return self.select_params(use_thompson=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for this bucket."""
        def arm_summary(arms: Dict[str, ArmStats]) -> Dict:
            return {
                arm_id: {
                    'pulls': arm.total_pulls,
                    'avg_reward': arm.avg_reward,
                    'mean_prob': arm.mean(),
                }
                for arm_id, arm in arms.items()
            }

        return {
            'bucket_index': self.bucket_index,
            'edge_arms': arm_summary(self.edge_arms),
            'size_arms': arm_summary(self.size_arms),
            'spread_arms': arm_summary(self.spread_arms),
            'depth_arms': arm_summary(self.depth_arms),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'bucket_index': self.bucket_index,
            'edge_candidates': self.edge_candidates,
            'size_candidates': self.size_candidates,
            'spread_candidates': self.spread_candidates,
            'depth_candidates': self.depth_candidates,
            'edge_arms': {k: v.to_dict() for k, v in self.edge_arms.items()},
            'size_arms': {k: v.to_dict() for k, v in self.size_arms.items()},
            'spread_arms': {k: v.to_dict() for k, v in self.spread_arms.items()},
            'depth_arms': {k: v.to_dict() for k, v in self.depth_arms.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BucketBandit':
        """Create from dictionary."""
        bandit = cls(
            bucket_index=data['bucket_index'],
            edge_candidates=data.get('edge_candidates', [0.003, 0.005, 0.01, 0.015, 0.02, 0.03]),
            size_candidates=data.get('size_candidates', [20, 40, 60, 80, 100, 150]),
            spread_candidates=data.get('spread_candidates', [0.02, 0.03, 0.04, 0.05]),
            depth_candidates=data.get('depth_candidates', [50, 100, 150, 200]),
        )

        # Load arm stats
        for arm_id, arm_data in data.get('edge_arms', {}).items():
            bandit.edge_arms[arm_id] = ArmStats.from_dict(arm_data)
        for arm_id, arm_data in data.get('size_arms', {}).items():
            bandit.size_arms[arm_id] = ArmStats.from_dict(arm_data)
        for arm_id, arm_data in data.get('spread_arms', {}).items():
            bandit.spread_arms[arm_id] = ArmStats.from_dict(arm_data)
        for arm_id, arm_data in data.get('depth_arms', {}).items():
            bandit.depth_arms[arm_id] = ArmStats.from_dict(arm_data)

        return bandit


@dataclass
class LTMBanditManager:
    """Manages bandits for all time buckets."""

    num_buckets: int = 15
    bucket_size_sec: int = 60
    total_time_sec: int = 900

    bandits: Dict[int, BucketBandit] = field(default_factory=dict)

    # Reward calculation weights
    pnl_weight: float = 1.0
    slippage_penalty: float = 2.0  # Multiply slippage by this
    unhedged_time_penalty: float = 0.01  # Per second unhedged
    fail_fill_penalty: float = 0.5  # Per failed fill

    def __post_init__(self):
        """Initialize bandits for all buckets."""
        for idx in range(self.num_buckets):
            if idx not in self.bandits:
                self.bandits[idx] = BucketBandit(bucket_index=idx)

    def get_bucket_index(self, time_remaining_sec: float) -> int:
        """Get bucket index for given time remaining."""
        if time_remaining_sec >= self.total_time_sec:
            return 0
        if time_remaining_sec <= 0:
            return self.num_buckets - 1
        elapsed = self.total_time_sec - time_remaining_sec
        return min(int(elapsed // self.bucket_size_sec), self.num_buckets - 1)

    def select_params(self, time_remaining_sec: float) -> Dict[str, float]:
        """Select parameters for given time remaining."""
        bucket_idx = self.get_bucket_index(time_remaining_sec)
        bandit = self.bandits.get(bucket_idx)

        if bandit:
            return bandit.select_params()

        # Default params
        return {
            'edge_required': 0.02,
            'max_size': 100,
            'max_spread_allowed': 0.05,
            'min_depth_required': 100,
        }

    def calculate_reward(
        self,
        pnl: float,
        slippage: float = 0.0,
        unhedged_time_sec: float = 0.0,
        fill_failures: int = 0,
    ) -> float:
        """Calculate reward for bandit update."""
        reward = (
            pnl * self.pnl_weight
            - slippage * self.slippage_penalty
            - unhedged_time_sec * self.unhedged_time_penalty
            - fill_failures * self.fail_fill_penalty
        )
        return reward

    def update(
        self,
        time_remaining_sec: float,
        params_used: Dict[str, float],
        pnl: float,
        slippage: float = 0.0,
        unhedged_time_sec: float = 0.0,
        fill_failures: int = 0,
    ) -> None:
        """Update bandit with trade result."""
        bucket_idx = self.get_bucket_index(time_remaining_sec)
        bandit = self.bandits.get(bucket_idx)

        if bandit:
            reward = self.calculate_reward(pnl, slippage, unhedged_time_sec, fill_failures)
            bandit.update(params_used, reward)
            logger.debug(f"Updated bucket {bucket_idx} bandit with reward {reward:.4f}")

    def get_best_policy(self) -> Dict[int, Dict[str, float]]:
        """Get best parameters for each bucket based on learning."""
        return {
            idx: bandit.get_best_params()
            for idx, bandit in self.bandits.items()
        }

    def get_stats(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for all bandits."""
        return {idx: bandit.get_stats() for idx, bandit in self.bandits.items()}

    def save(self, filepath: str) -> None:
        """Save bandit state to JSON file."""
        data = {
            'metadata': {
                'num_buckets': self.num_buckets,
                'bucket_size_sec': self.bucket_size_sec,
                'total_time_sec': self.total_time_sec,
            },
            'reward_weights': {
                'pnl_weight': self.pnl_weight,
                'slippage_penalty': self.slippage_penalty,
                'unhedged_time_penalty': self.unhedged_time_penalty,
                'fail_fill_penalty': self.fail_fill_penalty,
            },
            'bandits': {
                str(idx): bandit.to_dict()
                for idx, bandit in self.bandits.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved bandit state to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LTMBanditManager':
        """Load bandit state from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        meta = data.get('metadata', {})
        weights = data.get('reward_weights', {})

        manager = cls(
            num_buckets=meta.get('num_buckets', 15),
            bucket_size_sec=meta.get('bucket_size_sec', 60),
            total_time_sec=meta.get('total_time_sec', 900),
            pnl_weight=weights.get('pnl_weight', 1.0),
            slippage_penalty=weights.get('slippage_penalty', 2.0),
            unhedged_time_penalty=weights.get('unhedged_time_penalty', 0.01),
            fail_fill_penalty=weights.get('fail_fill_penalty', 0.5),
        )

        # Load bandits
        for idx_str, bandit_data in data.get('bandits', {}).items():
            idx = int(idx_str)
            manager.bandits[idx] = BucketBandit.from_dict(bandit_data)

        logger.info(f"Loaded bandit state from {filepath}")
        return manager

    def generate_report(self) -> str:
        """Generate human-readable report of bandit learning."""
        lines = [
            "=" * 70,
            "LTM BANDIT LEARNING REPORT",
            "=" * 70,
            "",
            "REWARD WEIGHTS",
            f"  PnL weight: {self.pnl_weight}",
            f"  Slippage penalty: {self.slippage_penalty}x",
            f"  Unhedged time penalty: {self.unhedged_time_penalty}/sec",
            f"  Fill failure penalty: {self.fail_fill_penalty}",
            "",
            "-" * 70,
            "BEST PARAMETERS BY BUCKET",
            "-" * 70,
        ]

        for idx in range(self.num_buckets):
            bandit = self.bandits.get(idx)
            if bandit:
                best = bandit.get_best_params()
                total_pulls = sum(
                    arm.total_pulls
                    for arm in list(bandit.edge_arms.values()) +
                              list(bandit.size_arms.values())
                )
                lines.append(
                    f"Bucket {idx:2d}: edge={best['edge_required']:.3f}, "
                    f"size={best['max_size']:.0f}, "
                    f"spread={best['max_spread_allowed']:.3f}, "
                    f"depth={best['min_depth_required']:.0f} "
                    f"({total_pulls} pulls)"
                )

        return "\n".join(lines)
