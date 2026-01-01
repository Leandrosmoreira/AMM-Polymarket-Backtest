"""
Volatility Arbitrage Bot for BTC 15-minute Markets

This module provides a sophisticated trading bot that:
1. Calculates real-time BTC volatility
2. Estimates true probabilities using statistical models
3. Compares model probabilities to market prices
4. Executes trades when significant edge is detected

Components:
- VolatilityCalculator: Rolling volatility estimation
- ProbabilityModel: Log-normal probability estimation
- EdgeDetector: Compares model vs market
- RiskManager: Position sizing and limits
- Executor: Paper and live trade execution
- Logger: Comprehensive trade logging
"""

from .volatility import VolatilityCalculator, VolatilityMetrics
from .probability import (
    ProbabilityModel,
    AdaptiveProbabilityModel,
    ProbabilityEstimate,
    MarketDirection
)
from .edge_detector import (
    EdgeDetector,
    EdgeOpportunity,
    MarketPrices,
    TradeSignal
)
from .risk_manager import (
    RiskManager,
    RiskConfig,
    PositionState,
    CONSERVATIVE_RISK,
    MODERATE_RISK,
    AGGRESSIVE_RISK
)
from .executor import (
    BaseExecutor,
    PaperExecutor,
    LiveExecutor,
    ExecutionMode,
    create_executor,
    Position,
    TradeResult
)
from .logger import (
    TradingLogger,
    TradeLog,
    SettlementLog,
    MarketAnalysisLog,
    MetricsCollector
)
from .bot import (
    VolatilityArbBot,
    BotConfig,
    run_bot,
    PAPER_TRADING_CONFIG,
    CONSERVATIVE_CONFIG,
    AGGRESSIVE_CONFIG
)


__all__ = [
    # Volatility
    'VolatilityCalculator',
    'VolatilityMetrics',

    # Probability
    'ProbabilityModel',
    'AdaptiveProbabilityModel',
    'ProbabilityEstimate',
    'MarketDirection',

    # Edge Detection
    'EdgeDetector',
    'EdgeOpportunity',
    'MarketPrices',
    'TradeSignal',

    # Risk Management
    'RiskManager',
    'RiskConfig',
    'PositionState',
    'CONSERVATIVE_RISK',
    'MODERATE_RISK',
    'AGGRESSIVE_RISK',

    # Execution
    'BaseExecutor',
    'PaperExecutor',
    'LiveExecutor',
    'ExecutionMode',
    'create_executor',
    'Position',
    'TradeResult',

    # Logging
    'TradingLogger',
    'TradeLog',
    'SettlementLog',
    'MarketAnalysisLog',
    'MetricsCollector',

    # Bot
    'VolatilityArbBot',
    'BotConfig',
    'run_bot',
    'PAPER_TRADING_CONFIG',
    'CONSERVATIVE_CONFIG',
    'AGGRESSIVE_CONFIG',
]
