"""Source module for Polymarket BTC/SOL Backtest."""
# SOL Delta-Neutral Strategy
from .data_collector import DataCollector
from .backtest_engine import BacktestEngine
from .position_manager import Position, Portfolio
from .risk_manager import RiskManager
from .metrics import PerformanceMetrics
from .visualizer import BacktestVisualizer
from .spread_calculator import SpreadCalculator
from .market_analyzer import MarketAnalyzer

# BTC Probabilistic Arbitrage Strategy
from .probability_calculator import ProbabilityCalculator, normal_cdf
from .btc_market_cycle import MarketCycleManager, MarketPosition, BTCTrade
from .btc_backtest_engine import BTCBacktestEngine, run_btc_backtest
from .log_processor import (
    load_log_file,
    prepare_backtest_data,
    merge_multiple_logs,
    assign_price_to_beat_to_ticks,
    group_ticks_by_market,
)
from .sample_data_generator import (
    generate_chainlink_ticks,
    generate_token_prices,
    generate_complete_log,
)
