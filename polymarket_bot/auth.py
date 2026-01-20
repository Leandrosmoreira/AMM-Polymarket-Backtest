"""
Polymarket Authentication Module.

Based on the working authentication from exemplo_polymarket.
Uses signature_type=1 for Magic.link/Email accounts with funder (proxy wallet).

Key concepts:
- Private Key: Your exported key from Polymarket settings
- Funder Address: Your Polymarket proxy wallet (found on profile page)
- Signature Type 1: Required for Magic.link/Email accounts
- API Credentials: Derived automatically from private key
"""

import logging
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType

logger = logging.getLogger(__name__)

# CLOB Host
HOST = "https://clob.polymarket.com"
CHAIN_ID = POLYGON  # 137 - Polygon Mainnet


class AuthConfig:
    """Authentication configuration loaded from .env file."""

    def __init__(self, env_file: str = "pmpe.env"):
        """
        Load authentication config from .env file.

        Args:
            env_file: Path to the .env file (relative to module or absolute)
        """
        # Try multiple locations for env file
        possible_paths = [
            Path(env_file),
            Path(__file__).parent / env_file,
            Path(__file__).parent.parent / env_file,
            Path.cwd() / env_file,
        ]

        config = {}
        for path in possible_paths:
            if path.exists():
                config = dotenv_values(str(path))
                logger.info(f"Loaded config from: {path}")
                break

        if not config:
            logger.warning(f"No .env file found. Tried: {[str(p) for p in possible_paths]}")

        # Required credentials
        self.private_key: str = config.get("PRIVATE_KEY", "")
        self.funder_address: str = config.get("FUNDER_ADDRESS", "")
        self.wallet_address: str = config.get("WALLET_ADDRESS", "")

        # Optional
        self.token_id: str = config.get("TOKEN_ID", "")

        # Signature type (1 for Magic.link/Email, 0 for direct EOA)
        self.signature_type: int = int(config.get("SIGNATURE_TYPE", "1"))

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, list_of_errors)
        """
        errors = []

        if not self.private_key:
            errors.append("PRIVATE_KEY is required")

        if self.signature_type == 1 and not self.funder_address:
            errors.append(
                "FUNDER_ADDRESS is required for signature_type=1 (Magic.link/Email accounts). "
                "Find your proxy wallet address at https://polymarket.com/@YOUR_USERNAME"
            )

        return len(errors) == 0, errors


# Global cached client
_cached_client: Optional[ClobClient] = None


def get_client(config: Optional[AuthConfig] = None, env_file: str = "pmpe.env") -> ClobClient:
    """
    Get authenticated Polymarket CLOB client.

    This function follows the pattern from exemplo_polymarket that is confirmed working.

    Args:
        config: Optional AuthConfig. If None, loads from env_file.
        env_file: Environment file to load if config not provided.

    Returns:
        Authenticated ClobClient ready for trading.

    Raises:
        RuntimeError: If authentication fails or config is invalid.
    """
    global _cached_client

    if _cached_client is not None:
        return _cached_client

    # Load config if not provided
    if config is None:
        config = AuthConfig(env_file)

    # Validate config
    is_valid, errors = config.validate()
    if not is_valid:
        raise RuntimeError(f"Invalid configuration: {'; '.join(errors)}")

    logger.info("Creating Polymarket CLOB client...")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Chain ID: {CHAIN_ID}")
    logger.info(f"  Signature Type: {config.signature_type}")
    logger.info(f"  Funder: {config.funder_address[:10]}...{config.funder_address[-6:] if config.funder_address else 'N/A'}")

    # Create client - THIS IS THE WORKING PATTERN FROM exemplo_polymarket
    client = ClobClient(
        host=HOST,
        key=config.private_key,
        chain_id=CHAIN_ID,
        signature_type=config.signature_type,
        funder=config.funder_address if config.funder_address else None,
    )

    # Derive API credentials (L2 authentication)
    logger.info("Deriving API credentials from private key...")
    credentials = client.create_or_derive_api_creds()
    client.set_api_creds(credentials)

    logger.info("Authentication successful!")
    logger.info(f"  API Key: {credentials.api_key[:8]}...{credentials.api_key[-4:]}")
    logger.info(f"  Wallet: {client.get_address()}")

    _cached_client = client
    return client


def get_balance(config: Optional[AuthConfig] = None, signature_type: int = 1) -> float:
    """
    Get USDC balance from Polymarket.

    Args:
        config: Optional AuthConfig
        signature_type: 1 for proxy wallet, 0 for direct

    Returns:
        Balance in USDC (float)
    """
    client = get_client(config)

    params = BalanceAllowanceParams(
        signature_type=signature_type,
        asset_type=AssetType.COLLATERAL
    )

    result = client.get_balance_allowance(params=params)
    balance_wei = int(result.get("balance", "0"))
    balance_usdc = balance_wei / 10**6  # USDC has 6 decimals

    return balance_usdc


def test_auth(env_file: str = "pmpe.env") -> bool:
    """
    Test authentication and print status.

    Args:
        env_file: Path to env file

    Returns:
        True if auth successful, False otherwise
    """
    print("=" * 60)
    print("POLYMARKET AUTHENTICATION TEST")
    print("=" * 60)

    try:
        config = AuthConfig(env_file)

        print(f"\nConfiguration:")
        print(f"  PRIVATE_KEY:    {'Set' if config.private_key else 'MISSING'}")
        print(f"  FUNDER_ADDRESS: {'Set' if config.funder_address else 'MISSING'}")
        print(f"  SIGNATURE_TYPE: {config.signature_type}")

        # Validate
        is_valid, errors = config.validate()
        if not is_valid:
            print(f"\nConfiguration errors:")
            for err in errors:
                print(f"  - {err}")
            return False

        print("\n1. Creating client...")
        client = get_client(config)
        print("   OK")

        print("\n2. Getting wallet address...")
        address = client.get_address()
        print(f"   {address}")

        print("\n3. Getting USDC balance...")
        balance = get_balance(config, signature_type=config.signature_type)
        print(f"   ${balance:.6f} USDC")

        print("\n" + "=" * 60)
        print("AUTHENTICATION SUCCESSFUL!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def reset_client():
    """Reset cached client (useful for testing or reconnection)."""
    global _cached_client
    _cached_client = None


if __name__ == "__main__":
    test_auth()
