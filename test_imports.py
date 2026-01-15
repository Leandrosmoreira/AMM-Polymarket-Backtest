#!/usr/bin/env python3
"""Test all imports and basic functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    from core.types import PaperPosition, PaperTrade, PaperStats, BookSnapshot, MarketInfo
    from core.buffers import TradeBuffer, PriceBuffer, BookBuffer, MarketDataState
    from config.gabagool_config import GabagoolConfig, get_config
    from agents import MicrostructureAgent, EdgeAgent, RiskAgent, MarketMakingAgent

    print("✅ All imports successful!")
    return True


def test_config():
    """Test configuration."""
    print("\nTesting config...")

    from config.gabagool_config import GabagoolConfig

    config = GabagoolConfig()
    assert config.initial_bankroll == 1000
    assert config.pair_cost_target == 0.97
    assert config.validate() == True

    print(f"✅ Config: bankroll=${config.initial_bankroll}, pair_cost_target={config.pair_cost_target}")
    return True


def test_position():
    """Test position management."""
    print("\nTesting position...")

    from core.types import PaperPosition

    pos = PaperPosition(market_id='test')
    pos.add_yes(100, 0.48)
    pos.add_no(100, 0.49)

    assert pos.yes_qty == 100
    assert pos.no_qty == 100
    assert abs(pos.pair_cost - 0.97) < 0.001  # 0.48 + 0.49
    assert pos.hedge_qty == 100
    assert abs(pos.locked_profit - 3.0) < 0.1  # 100 * (1.0 - 0.97)

    print(f"✅ Position: pair_cost={pos.pair_cost:.4f}, locked_profit=${pos.locked_profit:.2f}")
    return True


def test_stats():
    """Test statistics."""
    print("\nTesting stats...")

    from core.types import PaperStats

    stats = PaperStats()
    stats.total_trades = 10
    stats.wins = 7
    stats.losses = 3

    assert stats.win_rate == 0.7

    print(f"✅ Stats: win_rate={stats.win_rate*100:.1f}%")
    return True


def test_agents():
    """Test agent initialization."""
    print("\nTesting agents...")

    from config.gabagool_config import GabagoolConfig
    from agents import MicrostructureAgent, EdgeAgent, RiskAgent, MarketMakingAgent

    config = GabagoolConfig()

    micro = MicrostructureAgent(config)
    edge = EdgeAgent(config)
    risk = RiskAgent(config)
    mm = MarketMakingAgent(config)

    print("✅ All agents initialized!")
    return True


def test_buffers():
    """Test data buffers."""
    print("\nTesting buffers...")

    from core.buffers import PriceBuffer, TradeBuffer

    # Price buffer
    prices = PriceBuffer(size=100)
    for i in range(50):
        prices.add(100 + i * 0.1)

    assert prices.count == 50
    assert prices.latest == 104.9

    # Trade buffer
    from core.types import PaperTrade
    trades = TradeBuffer(size=50)
    for i in range(10):
        trades.add(PaperTrade(market_id='test', price=0.5, size=10))

    assert trades.count == 10

    print("✅ Buffers working!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("PAPER TRADER - Module Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_config,
        test_position,
        test_stats,
        test_agents,
        test_buffers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
