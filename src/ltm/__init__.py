"""
LTM - Liquidity Time Model

Models how liquidity, spread, slippage, and fill-rate change over the 15-minute
market window. Uses this to:
- Adjust minimum edge by time remaining
- Adjust order size by market phase
- Decide when to stop trading with intelligent rules
- Reduce "one-leg risk" near expiry
"""

from .collector import LTMCollector, LTMSnapshot
from .features import LTMFeatures, BucketStats
from .policy import LTMPolicy, BucketPolicy
from .decay import PairCostDecay, DecayMetrics
from .bandit import LTMBanditManager, BucketBandit, ArmStats

__all__ = [
    'LTMCollector',
    'LTMSnapshot',
    'LTMFeatures',
    'BucketStats',
    'LTMPolicy',
    'BucketPolicy',
    'PairCostDecay',
    'DecayMetrics',
    'LTMBanditManager',
    'BucketBandit',
    'ArmStats',
]
