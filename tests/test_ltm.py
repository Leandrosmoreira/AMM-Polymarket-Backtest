"""
Tests for LTM (Liquidity Time Model) module

Tests cover:
- LTMCollector: snapshot collection and bucket assignment
- LTMFeatures: bucket statistics computation
- LTMPolicy: policy loading and trade decisions
- PairCostDecay: decay rate estimation and recommendations
- LTMBanditManager: multi-armed bandit for parameter tuning
- LTMRiskManager: integrated risk management
- LTMBacktestEngine: full backtest with LTM
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ltm.collector import LTMCollector, LTMSnapshot
from src.ltm.features import LTMFeatures, BucketStats
from src.ltm.policy import LTMPolicy, BucketPolicy
from src.ltm.decay import PairCostDecay, DecayMetrics, calculate_optimal_entry_time
from src.ltm.bandit import LTMBanditManager, BucketBandit, ArmStats
from src.ltm_risk_manager import LTMRiskManager
from src.position_manager import MarketState, Portfolio


class TestLTMCollector:
    """Tests for LTMCollector."""

    def test_bucket_index_calculation(self):
        """Test correct bucket assignment based on time remaining."""
        collector = LTMCollector()

        # Bucket 0: 840-900s remaining (start)
        assert collector.get_bucket_index(900) == 0
        assert collector.get_bucket_index(850) == 0

        # Bucket 7: middle
        assert collector.get_bucket_index(450) == 7

        # Bucket 14: 0-60s remaining (end)
        assert collector.get_bucket_index(30) == 14
        assert collector.get_bucket_index(0) == 14

    def test_create_snapshot(self):
        """Test snapshot creation with derived fields."""
        collector = LTMCollector()

        snapshot = collector.create_snapshot(
            market_id="test_market",
            timestamp=datetime.now(),
            time_remaining_sec=600,
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
        )

        assert snapshot.market_id == "test_market"
        assert snapshot.pair_cost == 0.97
        assert snapshot.spread == -0.03
        assert snapshot.bucket_index == 5  # (900-600)/60 = 5

    def test_save_load_snapshots(self):
        """Test snapshot persistence."""
        collector = LTMCollector()

        # Create some snapshots
        for i in range(10):
            snapshot = collector.create_snapshot(
                market_id=f"market_{i}",
                timestamp=datetime.now(),
                time_remaining_sec=900 - i * 60,
                price_yes=0.48 + i * 0.001,
                price_no=0.49 + i * 0.001,
            )
            collector.add_snapshot(snapshot)

        # Save and load
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            filepath = f.name

        try:
            collector.save_snapshots(filepath)

            new_collector = LTMCollector()
            new_collector.load_snapshots(filepath)

            assert len(new_collector.snapshots) == 10
        finally:
            os.unlink(filepath)

    def test_get_snapshots_by_bucket(self):
        """Test filtering snapshots by bucket."""
        collector = LTMCollector()

        # Add snapshots to different buckets
        for bucket in [0, 0, 5, 5, 5, 10]:
            time_remaining = 900 - bucket * 60 - 30  # Middle of bucket
            snapshot = collector.create_snapshot(
                market_id=f"market_{bucket}",
                timestamp=datetime.now(),
                time_remaining_sec=time_remaining,
                price_yes=0.48,
                price_no=0.49,
            )
            collector.add_snapshot(snapshot)

        bucket_5 = collector.get_snapshots_by_bucket(5)
        assert len(bucket_5) == 3


class TestLTMFeatures:
    """Tests for LTMFeatures."""

    def test_compute_bucket_stats(self):
        """Test bucket statistics computation."""
        collector = LTMCollector()

        # Generate data for bucket 5
        for i in range(20):
            snapshot = collector.create_snapshot(
                market_id=f"market_{i}",
                timestamp=datetime.now(),
                time_remaining_sec=570,  # Bucket 5
                price_yes=0.48 + np.random.normal(0, 0.01),
                price_no=0.49 + np.random.normal(0, 0.01),
                volume=1000 + np.random.uniform(-100, 100),
            )
            collector.add_snapshot(snapshot)

        features = LTMFeatures()
        features.compute_from_collector(collector)

        stats = features.get_bucket_stats(5)
        assert stats is not None
        assert stats.n_samples == 20
        assert stats.spread_mean < 0  # Negative spread expected

    def test_get_curves(self):
        """Test liquidity curve generation."""
        collector = LTMCollector()

        # Generate data across all buckets
        for bucket in range(15):
            for i in range(5):
                time_remaining = 900 - bucket * 60 - 30
                snapshot = collector.create_snapshot(
                    market_id=f"market_{bucket}_{i}",
                    timestamp=datetime.now(),
                    time_remaining_sec=time_remaining,
                    price_yes=0.48,
                    price_no=0.49,
                )
                collector.add_snapshot(snapshot)

        features = LTMFeatures()
        features.compute_from_collector(collector)

        spread_curve = features.get_spread_curve()
        assert len(spread_curve) == 15
        assert all(v < 0 for v in spread_curve.values())  # All negative spreads


class TestLTMPolicy:
    """Tests for LTMPolicy."""

    def test_default_policy_creation(self):
        """Test default policy initialization."""
        policy = LTMPolicy()

        assert len(policy.bucket_policies) == 15

        # Check early bucket
        early = policy.bucket_policies[0]
        assert not early.stop_trading
        assert early.weight < 1.0  # Conservative at start

        # Check middle bucket
        middle = policy.bucket_policies[7]
        assert not middle.stop_trading
        assert middle.weight >= 1.0  # More aggressive

        # Check final bucket
        final = policy.bucket_policies[14]
        assert final.stop_trading

    def test_should_trade(self):
        """Test trade decision logic."""
        policy = LTMPolicy()

        # Good conditions in middle bucket
        result = policy.should_trade(
            time_remaining_sec=450,  # Middle
            pair_cost=0.97,  # Good edge
            spread=-0.03,
            current_imbalance=0.0,
        )
        assert result['should_trade'] == True

        # Bad edge
        result = policy.should_trade(
            time_remaining_sec=450,
            pair_cost=0.995,  # Poor edge
            spread=-0.005,
            current_imbalance=0.0,
        )
        assert result['should_trade'] == False

        # Final bucket (stopped)
        result = policy.should_trade(
            time_remaining_sec=30,  # Final bucket
            pair_cost=0.97,
            spread=-0.03,
            current_imbalance=0.0,
        )
        assert result['should_trade'] == False

    def test_save_load_policy(self):
        """Test policy persistence."""
        policy = LTMPolicy()

        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            filepath = f.name

        try:
            policy.save(filepath)

            loaded = LTMPolicy.load(filepath)
            assert len(loaded.bucket_policies) == 15
            assert loaded.default_edge_required == policy.default_edge_required
        finally:
            os.unlink(filepath)


class TestPairCostDecay:
    """Tests for PairCostDecay."""

    def test_decay_rate_calculation(self):
        """Test decay rate estimation."""
        decay = PairCostDecay()

        # Add observations showing improving pair_cost
        base_time = datetime.now()
        for i in range(10):
            decay.add_observation(
                market_id="test",
                timestamp=base_time + timedelta(seconds=i * 10),
                pair_cost=0.99 - i * 0.002,  # Improving
                time_remaining_sec=900 - i * 10,
            )

        rate, r_sq, samples = decay.get_decay_rate("test")

        assert rate > 0  # Positive decay (improving)
        assert samples == 10

    def test_analyze_recommendation(self):
        """Test decay analysis recommendations."""
        decay = PairCostDecay(target_pair_cost=0.97)

        # Already at target
        metrics = decay.analyze(
            market_id="test",
            current_pair_cost=0.96,
            time_remaining_sec=600,
        )
        assert metrics.recommendation == 'enter'

        # No data
        metrics = decay.analyze(
            market_id="new_market",
            current_pair_cost=0.99,
            time_remaining_sec=600,
        )
        assert metrics.recommendation == 'wait'

    def test_eta_calculation(self):
        """Test estimated time to target."""
        decay = PairCostDecay(target_pair_cost=0.97)

        # Rate = 0.001 per second
        eta = decay.estimate_time_to_target(
            current_pair_cost=0.99,
            decay_rate=0.001,
        )

        # Need to go from 0.99 to 0.97 = 0.02
        # At 0.001/sec = 20 seconds
        assert eta == pytest.approx(20, rel=0.01)


class TestLTMBanditManager:
    """Tests for LTMBanditManager."""

    def test_bandit_initialization(self):
        """Test bandit manager setup."""
        manager = LTMBanditManager()

        assert len(manager.bandits) == 15

        for bandit in manager.bandits.values():
            assert len(bandit.edge_arms) > 0
            assert len(bandit.size_arms) > 0

    def test_parameter_selection(self):
        """Test Thompson sampling parameter selection."""
        manager = LTMBanditManager()

        params = manager.select_params(time_remaining_sec=450)

        assert 'edge_required' in params
        assert 'max_size' in params
        assert 'max_spread_allowed' in params
        assert 'min_depth_required' in params

    def test_bandit_update(self):
        """Test bandit learning from rewards."""
        manager = LTMBanditManager()

        params = manager.select_params(time_remaining_sec=450)
        initial_pulls = sum(
            arm.total_pulls
            for arm in manager.bandits[7].edge_arms.values()
        )

        # Update with positive reward
        manager.update(
            time_remaining_sec=450,
            params_used=params,
            pnl=10.0,
            slippage=0.001,
        )

        new_pulls = sum(
            arm.total_pulls
            for arm in manager.bandits[7].edge_arms.values()
        )

        assert new_pulls > initial_pulls

    def test_save_load_bandit(self):
        """Test bandit state persistence."""
        manager = LTMBanditManager()

        # Do some updates
        for _ in range(10):
            params = manager.select_params(450)
            manager.update(450, params, pnl=np.random.uniform(-5, 10))

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            manager.save(filepath)

            loaded = LTMBanditManager.load(filepath)
            assert len(loaded.bandits) == 15

            # Check state was preserved
            orig_best = manager.get_best_policy()
            loaded_best = loaded.get_best_policy()
            assert orig_best[7] == loaded_best[7]
        finally:
            os.unlink(filepath)


class TestLTMRiskManager:
    """Tests for LTMRiskManager."""

    def test_initialization(self):
        """Test LTM risk manager setup."""
        manager = LTMRiskManager()

        assert manager.ltm_policy is not None
        assert manager.decay_model is not None

    def test_evaluate_ltm(self):
        """Test LTM-based trade evaluation."""
        manager = LTMRiskManager()
        portfolio = Portfolio(initial_capital=5000)

        market_state = MarketState(
            market_id="test",
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
            time_remaining=450,
            current_time=datetime.now(),
        )

        decision = manager.evaluate_ltm(market_state, portfolio)

        assert hasattr(decision, 'should_trade')
        assert hasattr(decision, 'bucket_index')
        assert hasattr(decision, 'adjusted_size')

    def test_dynamic_stop(self):
        """Test dynamic stop trading."""
        manager = LTMRiskManager()

        # Middle of market - should not stop
        mid_state = MarketState(
            market_id="test",
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
            time_remaining=450,
            current_time=datetime.now(),
        )
        assert not manager.get_dynamic_stop(mid_state)

        # End of market - should stop
        end_state = MarketState(
            market_id="test",
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
            time_remaining=30,
            current_time=datetime.now(),
        )
        assert manager.get_dynamic_stop(end_state)

    def test_dynamic_imbalance_limit(self):
        """Test time-varying imbalance limits."""
        manager = LTMRiskManager()

        mid_state = MarketState(
            market_id="test",
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
            time_remaining=450,
            current_time=datetime.now(),
        )

        end_state = MarketState(
            market_id="test",
            price_yes=0.48,
            price_no=0.49,
            volume=1000,
            time_remaining=90,
            current_time=datetime.now(),
        )

        mid_limit = manager.get_dynamic_imbalance_limit(mid_state)
        end_limit = manager.get_dynamic_imbalance_limit(end_state)

        # End should be more restrictive
        assert end_limit < mid_limit


class TestCalculateOptimalEntryTime:
    """Tests for optimal entry time calculation."""

    def test_already_at_target(self):
        """Test when already at target."""
        result = calculate_optimal_entry_time(
            current_pair_cost=0.96,
            target_pair_cost=0.97,
            decay_rate=0.001,
            time_remaining_sec=600,
        )
        assert result['optimal_entry'] == 'now'

    def test_cannot_reach_target(self):
        """Test when target is unreachable."""
        result = calculate_optimal_entry_time(
            current_pair_cost=0.99,
            target_pair_cost=0.97,
            decay_rate=0.0001,  # Very slow decay
            time_remaining_sec=100,  # Not enough time
        )
        assert result['optimal_entry'] == 'skip'

    def test_wait_for_better_entry(self):
        """Test waiting recommendation."""
        result = calculate_optimal_entry_time(
            current_pair_cost=0.985,
            target_pair_cost=0.97,
            decay_rate=0.001,
            time_remaining_sec=600,
        )
        assert result['optimal_entry'] == 'wait'
        assert result['wait_time'] > 0


class TestIntegration:
    """Integration tests for full LTM workflow."""

    def test_full_ltm_workflow(self):
        """Test complete LTM data collection -> features -> policy flow."""
        # 1. Collect snapshots
        collector = LTMCollector()

        for market_idx in range(10):
            for bucket in range(15):
                time_remaining = 900 - bucket * 60 - 30
                spread = -0.02 - 0.005 * (bucket - 7) ** 2 / 50  # Best in middle

                collector.add_snapshot(collector.create_snapshot(
                    market_id=f"market_{market_idx}",
                    timestamp=datetime.now(),
                    time_remaining_sec=time_remaining,
                    price_yes=0.5 + spread / 2,
                    price_no=0.5 + spread / 2,
                    volume=1000 * (1 - bucket / 15),
                ))

        # 2. Compute features
        features = LTMFeatures()
        features.compute_from_collector(collector)

        assert len(features.bucket_stats) == 15

        # 3. Create and update policy
        policy = LTMPolicy()
        policy.update_from_features(features)

        # Verify middle buckets should be more favorable
        mid_policy = policy.bucket_policies[7]
        end_policy = policy.bucket_policies[13]

        # Middle should have lower edge requirement or higher weight
        assert mid_policy.weight >= end_policy.weight or mid_policy.edge_required <= end_policy.edge_required

    def test_ltm_risk_manager_with_bandit(self):
        """Test LTM risk manager with bandit learning."""
        manager = LTMRiskManager(use_bandit=True)
        portfolio = Portfolio(initial_capital=5000)

        # Simulate multiple trades
        for _ in range(20):
            market_state = MarketState(
                market_id=f"test_{_}",
                price_yes=0.48 + np.random.normal(0, 0.01),
                price_no=0.49 + np.random.normal(0, 0.01),
                volume=1000,
                time_remaining=450,
                current_time=datetime.now(),
            )

            decision = manager.evaluate_ltm(market_state, portfolio)

            if decision.should_trade and decision.bandit_params:
                # Simulate trade outcome
                pnl = np.random.uniform(-5, 10)
                manager.update_bandit(
                    time_remaining=450,
                    params_used=decision.bandit_params,
                    pnl=pnl,
                )

        # Check bandit has learned
        stats = manager.get_decision_stats()
        assert stats['total_decisions'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
