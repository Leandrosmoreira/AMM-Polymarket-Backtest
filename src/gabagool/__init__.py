"""
Gabagool Bot - Spread Capture Strategy
Based on the proven strategy of @gabagool22 ($450k+ profit)
"""

from .config import GabagoolConfig, CONSERVATIVE_CONFIG, MODERATE_CONFIG, AGGRESSIVE_CONFIG
from .spread_monitor import SpreadMonitor
from .position_manager import GabagoolPositionManager
from .bot import GabagoolBot, run_bot
from .backtest import GabagoolBacktest, run_backtest as run_gabagool_backtest

__all__ = [
    'GabagoolConfig',
    'CONSERVATIVE_CONFIG',
    'MODERATE_CONFIG',
    'AGGRESSIVE_CONFIG',
    'SpreadMonitor',
    'GabagoolPositionManager',
    'GabagoolBot',
    'run_bot',
    'GabagoolBacktest',
    'run_gabagool_backtest',
]
