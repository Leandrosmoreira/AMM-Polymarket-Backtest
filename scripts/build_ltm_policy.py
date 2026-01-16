#!/usr/bin/env python3
"""
Build LTM Policy from Historical Data

This script:
1. Loads historical trade/snapshot data
2. Computes bucket statistics (spread, depth, fill rate, slippage)
3. Generates optimized ltm_policy.yaml

Usage:
    python scripts/build_ltm_policy.py --data data/ltm_snapshots.parquet --output config/ltm_policy.yaml
    python scripts/build_ltm_policy.py --simulate  # Generate from simulated data
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.ltm.collector import LTMCollector
from src.ltm.features import LTMFeatures
from src.ltm.policy import LTMPolicy, BucketPolicy


def generate_simulated_data(
    n_markets: int = 100,
    samples_per_market: int = 30,
) -> pd.DataFrame:
    """
    Generate simulated market data for testing.

    Simulates realistic patterns:
    - Spread tends to narrow over time
    - Liquidity peaks in middle of market
    - Slippage increases near end
    """
    np.random.seed(42)

    data = []

    for market_idx in range(n_markets):
        market_id = f"market_{market_idx:04d}"
        base_time = datetime.now() - timedelta(hours=market_idx)

        # Market characteristics (randomized)
        base_spread = np.random.uniform(-0.04, -0.01)
        base_depth = np.random.uniform(500, 2000)
        base_volume = np.random.uniform(10000, 50000)

        for sample_idx in range(samples_per_market):
            # Time remaining (evenly spaced through market)
            t_remaining = 900 - (sample_idx * 900 / samples_per_market)
            t_elapsed = 900 - t_remaining

            # Spread evolution (tends to narrow, then widen at end)
            spread_factor = 1.0
            if t_remaining > 600:  # Early
                spread_factor = 1.1
            elif t_remaining > 180:  # Middle
                spread_factor = 0.85
            elif t_remaining > 60:  # Late
                spread_factor = 1.2
            else:  # Final
                spread_factor = 1.5

            spread = base_spread * spread_factor + np.random.normal(0, 0.005)
            pair_cost = 1.0 + spread

            # Price decomposition
            imbalance = np.random.normal(0, 0.02)
            price_yes = 0.5 + imbalance + spread / 2
            price_no = 0.5 - imbalance + spread / 2

            # Depth evolution (peaks in middle, lower at edges)
            depth_factor = 1.0
            if t_remaining > 720:  # Very early
                depth_factor = 0.7
            elif t_remaining > 600:  # Early
                depth_factor = 0.9
            elif t_remaining > 240:  # Middle
                depth_factor = 1.2
            elif t_remaining > 60:  # Late
                depth_factor = 0.8
            else:  # Final
                depth_factor = 0.5

            total_depth = base_depth * depth_factor * np.random.uniform(0.8, 1.2)

            # Volume (accumulates over time)
            volume = base_volume * (t_elapsed / 900) * np.random.uniform(0.9, 1.1)

            # Fill rate (decreases near end)
            if t_remaining > 180:
                fill_rate = np.random.uniform(0.9, 1.0)
            elif t_remaining > 60:
                fill_rate = np.random.uniform(0.7, 0.95)
            else:
                fill_rate = np.random.uniform(0.5, 0.85)

            # Slippage (increases near end)
            if t_remaining > 300:
                slippage = np.random.uniform(0, 0.002)
            elif t_remaining > 120:
                slippage = np.random.uniform(0.001, 0.005)
            else:
                slippage = np.random.uniform(0.003, 0.015)

            # Liquidity score
            liquidity_score = (
                min(total_depth / 10000, 1) * 25 +
                min(volume / 50000, 1) * 25 +
                max(0, (0.05 - abs(spread)) / 0.05) * 25 +
                fill_rate * 25
            )

            data.append({
                'market_id': market_id,
                'timestamp': base_time + timedelta(seconds=t_elapsed),
                'bucket_index': int(t_elapsed // 60),
                't_remaining_sec': t_remaining,
                't_elapsed_sec': t_elapsed,
                'price_yes': price_yes,
                'price_no': price_no,
                'mid_price': (price_yes + price_no) / 2,
                'pair_cost': pair_cost,
                'spread': spread,
                'spread_bps': spread * 10000,
                'depth_yes_top5': total_depth / 2,
                'depth_no_top5': total_depth / 2,
                'total_depth': total_depth,
                'book_imbalance': imbalance,
                'volume': volume,
                'trade_count': int(volume / 100),
                'fill_rate': fill_rate,
                'slippage_realized': slippage,
                'execution_time_ms': np.random.uniform(50, 500),
                'liquidity_score': liquidity_score,
            })

    return pd.DataFrame(data)


def build_policy_from_features(
    features: LTMFeatures,
    base_edge: float = 0.02,
    base_size: float = 100,
) -> LTMPolicy:
    """Build optimized policy from computed features."""

    policy = LTMPolicy(
        default_edge_required=base_edge,
        default_max_size=base_size,
    )

    for bucket_idx, stats in features.bucket_stats.items():
        if stats.n_samples == 0:
            continue

        # Calculate time range
        time_start = 900 - (bucket_idx * 60)
        time_end = max(0, time_start - 60)

        # Determine phase and adjust parameters
        # Edge: higher when spread is bad or slippage is high
        edge_adj = 1.0
        if stats.spread_p90 > -0.01:  # Bad spread
            edge_adj *= 1.3
        if stats.slippage_p90 > 0.005:  # High slippage
            edge_adj *= (1 + stats.slippage_p90 * 10)

        edge_required = base_edge * edge_adj

        # Size: reduce when liquidity is low
        size_adj = 1.0
        if stats.liquidity_score_mean < 50:
            size_adj *= 0.7
        if stats.fill_rate_mean < 0.8:
            size_adj *= 0.8

        max_size = base_size * size_adj

        # Stop trading: if conditions are very bad
        stop_trading = (
            stats.fill_rate_mean < 0.5 or
            stats.slippage_p90 > 0.02 or
            bucket_idx >= 14  # Last bucket
        )

        # Weight: based on overall conditions
        if stop_trading:
            weight = 0
        elif stats.liquidity_score_mean > 70:
            weight = 1.2
        elif stats.liquidity_score_mean > 50:
            weight = 1.0
        elif stats.liquidity_score_mean > 30:
            weight = 0.7
        else:
            weight = 0.4

        # Max spread: based on observed p90
        max_spread = max(0.02, abs(stats.spread_p90) * 1.2)

        # Min depth: based on observed p10
        min_depth = max(50, stats.depth_p10 * 0.8)

        # Max imbalance: tighter when liquidity is low
        if bucket_idx >= 12:
            max_imbalance = 0.1
        elif bucket_idx >= 10:
            max_imbalance = 0.2
        else:
            max_imbalance = 0.3

        bucket_policy = BucketPolicy(
            bucket_index=bucket_idx,
            time_remaining_start=time_start,
            time_remaining_end=time_end,
            edge_required=edge_required,
            max_spread_allowed=max_spread,
            min_depth_required=min_depth,
            max_size=max_size,
            stop_trading=stop_trading,
            max_imbalance=max_imbalance,
            weight=weight,
        )

        policy.bucket_policies[bucket_idx] = bucket_policy

    return policy


def main():
    parser = argparse.ArgumentParser(
        description='Build LTM Policy from historical data'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to snapshot data (parquet or CSV)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/ltm_policy.yaml',
        help='Output path for policy YAML'
    )
    parser.add_argument(
        '--features-output',
        type=str,
        default='data/ltm_features.parquet',
        help='Output path for computed features'
    )
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Generate from simulated data'
    )
    parser.add_argument(
        '--n-markets',
        type=int,
        default=100,
        help='Number of simulated markets'
    )
    parser.add_argument(
        '--base-edge',
        type=float,
        default=0.02,
        help='Base edge required'
    )
    parser.add_argument(
        '--base-size',
        type=float,
        default=100,
        help='Base order size'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Print detailed report'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LTM POLICY BUILDER")
    print("=" * 60)

    # Load or generate data
    if args.simulate:
        print(f"\nGenerating simulated data for {args.n_markets} markets...")
        df = generate_simulated_data(n_markets=args.n_markets)
        print(f"Generated {len(df)} snapshots")
    elif args.data:
        print(f"\nLoading data from {args.data}...")
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
        print(f"Loaded {len(df)} snapshots")
    else:
        print("\nERROR: Must specify --data or --simulate")
        sys.exit(1)

    # Compute features
    print("\nComputing bucket features...")
    features = LTMFeatures()
    features.compute_from_dataframe(df)

    if args.report:
        print("\n" + features.generate_report())

    # Save features
    os.makedirs(os.path.dirname(args.features_output), exist_ok=True)
    features.save(args.features_output)
    print(f"\nSaved features to {args.features_output}")

    # Build policy
    print("\nBuilding optimized policy...")
    policy = build_policy_from_features(
        features,
        base_edge=args.base_edge,
        base_size=args.base_size,
    )

    if args.report:
        print("\n" + policy.generate_report())

    # Save policy
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    policy.save(args.output)
    print(f"\nSaved policy to {args.output}")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)

    # Summary
    print("\nSUMMARY:")
    stop_buckets = [
        idx for idx, p in policy.bucket_policies.items()
        if p.stop_trading
    ]
    print(f"  Total buckets: {len(policy.bucket_policies)}")
    print(f"  Stop trading buckets: {stop_buckets}")
    print(f"  Features file: {args.features_output}")
    print(f"  Policy file: {args.output}")


if __name__ == '__main__':
    main()
