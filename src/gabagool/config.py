"""
Configuration for Gabagool Spread Capture Bot
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class GabagoolConfig:
    """Configuration for the Gabagool spread capture bot."""

    # === SPREAD THRESHOLDS ===
    MIN_SPREAD: float = 0.02          # 2% minimum spread to enter
    MAX_SPREAD: float = 0.10          # 10% max (suspicious if too high)
    TARGET_SPREAD: float = 0.03       # 3% target for aggressive entry

    # === ORDER SIZING ===
    ORDER_SIZE_USD: float = 15.0      # USD per order (like gabagool's $10-$20)
    MIN_ORDER_SIZE: float = 5.0       # Minimum order size
    MAX_ORDER_SIZE: float = 50.0      # Maximum order size

    # === POSITION LIMITS ===
    MAX_PER_MARKET: float = 500.0     # Maximum USD per market
    MAX_TOTAL_EXPOSURE: float = 2000.0  # Total exposure across all markets
    MAX_IMBALANCE_PCT: float = 0.20   # 20% max imbalance (UP vs DOWN)

    # === TIMING ===
    CHECK_INTERVAL_MS: int = 500      # Check spread every 500ms
    SKIP_FIRST_SECONDS: int = 120     # Skip first 2 minutes
    SKIP_LAST_SECONDS: int = 60       # Skip last minute
    MARKET_DURATION_SECONDS: int = 900  # 15 minutes

    # === MARKETS ===
    ENABLED_ASSETS: List[str] = field(default_factory=lambda: ["BTC", "ETH"])
    MARKET_TIMEFRAME: str = "15min"

    # === EXECUTION ===
    MAX_SLIPPAGE_PCT: float = 0.01    # 1% max slippage
    ORDER_TIMEOUT_SECONDS: int = 5     # Order timeout
    RETRY_ATTEMPTS: int = 3            # Retry failed orders
    RETRY_DELAY_MS: int = 500          # Delay between retries

    # === RISK MANAGEMENT ===
    MIN_LIQUIDITY_USD: float = 100.0   # Min liquidity in order book
    MAX_POSITION_VALUE: float = 1000.0  # Max value per position
    STOP_ON_ERROR_COUNT: int = 5       # Stop after N consecutive errors

    # === API ===
    CLOB_API_URL: str = "https://clob.polymarket.com"
    GAMMA_API_URL: str = "https://gamma-api.polymarket.com"

    # === CREDENTIALS (from environment) ===
    PRIVATE_KEY: Optional[str] = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY"))
    API_KEY: Optional[str] = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY"))
    API_SECRET: Optional[str] = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET"))
    API_PASSPHRASE: Optional[str] = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE"))

    # === LOGGING ===
    LOG_LEVEL: str = "INFO"
    LOG_TRADES: bool = True
    LOG_FILE: str = "gabagool_bot.log"

    # === MODE ===
    PAPER_TRADING: bool = True         # Start in paper trading mode
    DRY_RUN: bool = False              # Don't execute orders, just log

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        if self.MIN_SPREAD >= self.MAX_SPREAD:
            errors.append("MIN_SPREAD must be less than MAX_SPREAD")

        if self.ORDER_SIZE_USD < self.MIN_ORDER_SIZE:
            errors.append("ORDER_SIZE_USD must be >= MIN_ORDER_SIZE")

        if self.MAX_IMBALANCE_PCT < 0 or self.MAX_IMBALANCE_PCT > 1:
            errors.append("MAX_IMBALANCE_PCT must be between 0 and 1")

        if not self.PAPER_TRADING and not self.PRIVATE_KEY:
            errors.append("PRIVATE_KEY required for live trading")

        if errors:
            for error in errors:
                print(f"Config Error: {error}")
            return False

        return True

    def get_entry_threshold(self, time_elapsed_seconds: int) -> float:
        """
        Get dynamic entry threshold based on time elapsed in market.

        Early: More aggressive (accept smaller spreads)
        Late: More conservative (need larger spreads)
        """
        remaining = self.MARKET_DURATION_SECONDS - time_elapsed_seconds

        if time_elapsed_seconds < self.SKIP_FIRST_SECONDS:
            return 999  # Don't enter

        if remaining < self.SKIP_LAST_SECONDS:
            return 999  # Don't enter

        # Early phase (2-5 min): Aggressive
        if time_elapsed_seconds < 300:
            return self.MIN_SPREAD * 0.8  # Accept 20% smaller spread

        # Middle phase (5-10 min): Normal
        if time_elapsed_seconds < 600:
            return self.MIN_SPREAD

        # Late phase (10-14 min): Conservative
        return self.MIN_SPREAD * 1.25  # Need 25% larger spread

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in ['PRIVATE_KEY', 'API_KEY', 'API_SECRET', 'API_PASSPHRASE']
        }


# Preset configurations
CONSERVATIVE_CONFIG = GabagoolConfig(
    MIN_SPREAD=0.03,
    ORDER_SIZE_USD=10.0,
    MAX_PER_MARKET=200.0,
    MAX_TOTAL_EXPOSURE=500.0,
)

MODERATE_CONFIG = GabagoolConfig(
    MIN_SPREAD=0.02,
    ORDER_SIZE_USD=15.0,
    MAX_PER_MARKET=500.0,
    MAX_TOTAL_EXPOSURE=1500.0,
)

AGGRESSIVE_CONFIG = GabagoolConfig(
    MIN_SPREAD=0.015,
    ORDER_SIZE_USD=25.0,
    MAX_PER_MARKET=1000.0,
    MAX_TOTAL_EXPOSURE=3000.0,
)
