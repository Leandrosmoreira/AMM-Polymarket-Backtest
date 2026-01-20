"""
Configuration module for Polymarket Bot.

Combines the simple auth pattern from exemplo_polymarket with
the comprehensive settings from trading_bot_ltm.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, dotenv_values

# Find .env file in multiple locations
_module_dir = Path(__file__).parent
_possible_envs = [
    _module_dir / "pmpe.env",      # Preferred: pmpe.env (exemplo_polymarket style)
    _module_dir / ".env",          # Fallback: .env
    _module_dir.parent / "pmpe.env",
    _module_dir.parent / ".env",
]

# Load first env file found
for _env_file in _possible_envs:
    if _env_file.exists():
        load_dotenv(_env_file, override=False)
        break
else:
    # Try default .env in cwd
    load_dotenv(override=False)


@dataclass
class Settings:
    """
    Bot settings combining auth and trading configuration.

    Authentication (from exemplo_polymarket pattern):
    - private_key: Exported from Polymarket settings
    - funder_address: Proxy wallet from profile page
    - signature_type: 1 for Magic.link/Email accounts

    Trading (from trading_bot_ltm):
    - order_size: Shares per order
    - target_pair_cost: Threshold for arbitrage
    - dry_run: Simulation mode
    """

    # ========================================
    # AUTHENTICATION (exemplo_polymarket style)
    # ========================================
    private_key: str = field(default_factory=lambda: os.getenv("PRIVATE_KEY", ""))
    funder_address: str = field(default_factory=lambda: os.getenv("FUNDER_ADDRESS", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("WALLET_ADDRESS", ""))
    signature_type: int = field(default_factory=lambda: int(os.getenv("SIGNATURE_TYPE", "1")))

    # Legacy support (trading_bot_ltm style) - maps to new names
    @property
    def funder(self) -> str:
        """Alias for funder_address (trading_bot_ltm compatibility)."""
        return self.funder_address or os.getenv("POLYMARKET_FUNDER", "")

    # ========================================
    # MARKET CONFIGURATION
    # ========================================
    market_slug: str = field(default_factory=lambda: os.getenv("MARKET_SLUG", ""))
    market_id: str = field(default_factory=lambda: os.getenv("MARKET_ID", ""))
    yes_token_id: str = field(default_factory=lambda: os.getenv("YES_TOKEN_ID", "") or os.getenv("TOKEN_ID", ""))
    no_token_id: str = field(default_factory=lambda: os.getenv("NO_TOKEN_ID", ""))

    # ========================================
    # TRADING PARAMETERS
    # ========================================
    target_pair_cost: float = field(default_factory=lambda: float(os.getenv("TARGET_PAIR_COST", "0.991")))
    order_size: float = field(default_factory=lambda: float(os.getenv("ORDER_SIZE", "5")))
    order_type: str = field(default_factory=lambda: os.getenv("ORDER_TYPE", "FOK").upper())
    cooldown_seconds: float = field(default_factory=lambda: float(os.getenv("COOLDOWN_SECONDS", "10")))

    # ========================================
    # MODE
    # ========================================
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true")
    sim_balance: float = field(default_factory=lambda: float(os.getenv("SIM_BALANCE", "100")))
    verbose: bool = field(default_factory=lambda: os.getenv("VERBOSE", "false").lower() == "true")

    # ========================================
    # WEBSOCKET
    # ========================================
    ws_url: str = field(default_factory=lambda: os.getenv("WS_URL", "wss://ws-subscriptions-clob.polymarket.com"))
    use_wss: bool = field(default_factory=lambda: os.getenv("USE_WSS", "false").lower() == "true")

    # ========================================
    # RISK MANAGEMENT
    # ========================================
    max_daily_loss: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS", "0")))
    max_position_size: float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0")))
    max_trades_per_day: int = field(default_factory=lambda: int(os.getenv("MAX_TRADES_PER_DAY", "0")))
    min_balance_required: float = field(default_factory=lambda: float(os.getenv("MIN_BALANCE_REQUIRED", "10.0")))
    max_balance_utilization: float = field(default_factory=lambda: float(os.getenv("MAX_BALANCE_UTILIZATION", "0.8")))

    # ========================================
    # STATISTICS
    # ========================================
    enable_stats: bool = field(default_factory=lambda: os.getenv("ENABLE_STATS", "true").lower() == "true")
    trade_log_file: str = field(default_factory=lambda: os.getenv("TRADE_LOG_FILE", "trades.json"))
    use_rich_output: bool = field(default_factory=lambda: os.getenv("USE_RICH_OUTPUT", "true").lower() == "true")

    # ========================================
    # LTM (Liquidity Time Model)
    # ========================================
    use_ltm: bool = field(default_factory=lambda: os.getenv("USE_LTM", "false").lower() == "true")
    ltm_policy_path: str = field(default_factory=lambda: os.getenv("LTM_POLICY_PATH", "ltm_policy.yaml"))
    ltm_use_decay: bool = field(default_factory=lambda: os.getenv("LTM_USE_DECAY", "true").lower() == "true")

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate settings.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        # Auth validation
        if not self.private_key:
            errors.append("PRIVATE_KEY is required")

        if self.signature_type == 1:
            funder = self.funder_address or self.funder
            if not funder:
                errors.append(
                    "FUNDER_ADDRESS is required for signature_type=1 (Magic.link/Email). "
                    "Get it from https://polymarket.com/@YOUR_USERNAME"
                )

        # Trading validation
        if self.order_size <= 0:
            errors.append("ORDER_SIZE must be > 0")

        if self.target_pair_cost <= 0 or self.target_pair_cost >= 1:
            errors.append("TARGET_PAIR_COST must be between 0 and 1")

        return len(errors) == 0, errors

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("POLYMARKET BOT CONFIGURATION")
        print("=" * 60)

        print("\n[Authentication]")
        print(f"  PRIVATE_KEY:    {'Set' if self.private_key else 'MISSING'}")
        print(f"  FUNDER_ADDRESS: {self.funder_address[:10]}...{self.funder_address[-6:] if self.funder_address else 'MISSING'}")
        print(f"  SIGNATURE_TYPE: {self.signature_type}")

        print("\n[Trading]")
        print(f"  TARGET_PAIR_COST: ${self.target_pair_cost:.4f}")
        print(f"  ORDER_SIZE:       {self.order_size}")
        print(f"  ORDER_TYPE:       {self.order_type}")
        print(f"  COOLDOWN:         {self.cooldown_seconds}s")

        print("\n[Mode]")
        print(f"  DRY_RUN:          {self.dry_run}")
        print(f"  SIM_BALANCE:      ${self.sim_balance}")
        print(f"  USE_WSS:          {self.use_wss}")
        print(f"  USE_LTM:          {self.use_ltm}")

        print("=" * 60)


def load_settings() -> Settings:
    """Load settings from environment."""
    return Settings()


def load_settings_from_file(env_file: str) -> Settings:
    """Load settings from specific env file."""
    config = dotenv_values(env_file)

    # Temporarily set env vars
    for key, value in config.items():
        if value is not None:
            os.environ[key] = value

    return Settings()
