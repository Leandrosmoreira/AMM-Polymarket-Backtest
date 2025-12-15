#!/usr/bin/env python3
"""
Polymarket Copy Trading Bot
===========================

Monitor a wallet on Polygon blockchain and automatically copy trades
with configurable sizing (default 10% of original trade).

Usage:
    python copytrade.py --help
    python copytrade.py monitor                    # Start monitoring (dry run)
    python copytrade.py monitor --live             # Start with live execution
    python copytrade.py status                     # Show current status
    python copytrade.py history                    # Show trade history

Configuration:
    1. Copy .env.example to .env
    2. Edit .env with your private key
    3. Optionally edit config/copytrade_settings.py for other settings

Environment Variables (via .env file):
    POLY_PRIVATE_KEY - Your wallet private key (required for live trading)
    POLYGON_RPC_URL  - Custom RPC URL (optional)
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import sys
import json
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from web3 import Web3

from config.copytrade_settings import (
    TARGET_WALLET,
    POLYGON_RPC_URLS,
    POLYGON_RPC_PRIMARY,
    CTF_EXCHANGE_ADDRESS,
    NEG_RISK_CTF_EXCHANGE,
    COPY_PERCENTAGE,
    DYNAMIC_SIZING,
    MAX_TRADE_SIZE,
    MIN_TRADE_SIZE,
    MAX_SLIPPAGE,
    MAX_EXPOSURE_PERCENT,
    MAX_POSITIONS,
    MIN_BALANCE_USDC,
    POLL_INTERVAL,
    LOOKBACK_BLOCKS,
    CONFIRMATION_BLOCKS,
    LOG_FILE,
    LOG_LEVEL,
)

from src.copytrade import (
    BlockchainMonitor,
    TransactionInfo,
    PolymarketDecoder,
    PolymarketTrade,
    CopyTradeExecutor,
    DryRunExecutor,
    CopyTradeResult,
)

# Setup logging
def setup_logging(log_level: str = LOG_LEVEL, log_file: Optional[str] = LOG_FILE):
    """Configure logging"""
    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


class CopyTradingBot:
    """
    Main copy trading bot that orchestrates:
    - Blockchain monitoring
    - Transaction decoding
    - Copy trade execution
    """

    def __init__(
        self,
        target_wallet: str,
        live_mode: bool = False,
        copy_percentage: float = COPY_PERCENTAGE,
    ):
        self.target_wallet = target_wallet
        self.live_mode = live_mode
        self.copy_percentage = copy_percentage
        self.logger = logging.getLogger(self.__class__.__name__)

        # Get RPC URLs
        custom_rpc = os.environ.get('POLYGON_RPC_URL')
        rpc_urls = [custom_rpc] + POLYGON_RPC_URLS if custom_rpc else POLYGON_RPC_URLS

        # Initialize Web3
        self.web3 = None
        for rpc_url in rpc_urls:
            try:
                web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
                if web3.is_connected():
                    self.web3 = web3
                    self.logger.info(f"Connected to: {rpc_url}")
                    break
            except Exception as e:
                self.logger.warning(f"Failed to connect to {rpc_url}: {e}")

        if not self.web3:
            raise ConnectionError("Failed to connect to any RPC endpoint")

        # Initialize components
        self.monitor = BlockchainMonitor(
            rpc_urls=rpc_urls,
            target_wallet=target_wallet,
            contract_addresses=[CTF_EXCHANGE_ADDRESS, NEG_RISK_CTF_EXCHANGE],
            poll_interval=POLL_INTERVAL,
            lookback_blocks=LOOKBACK_BLOCKS,
            confirmation_blocks=CONFIRMATION_BLOCKS
        )

        self.decoder = PolymarketDecoder(
            web3=self.web3,
            ctf_exchange_address=CTF_EXCHANGE_ADDRESS
        )

        # Initialize executor
        if live_mode:
            private_key = os.environ.get('POLY_PRIVATE_KEY')
            if not private_key:
                raise ValueError(
                    "POLY_PRIVATE_KEY environment variable required for live trading"
                )

            self.executor = CopyTradeExecutor(
                web3=self.web3,
                private_key=private_key,
                ctf_exchange_address=CTF_EXCHANGE_ADDRESS,
                copy_percentage=copy_percentage,
                dynamic_sizing=DYNAMIC_SIZING,
                max_trade_size=MAX_TRADE_SIZE,
                min_trade_size=MIN_TRADE_SIZE,
                max_slippage=MAX_SLIPPAGE,
                max_exposure_percent=MAX_EXPOSURE_PERCENT,
                max_positions=MAX_POSITIONS,
                min_balance=MIN_BALANCE_USDC,
            )
            self.logger.warning("LIVE MODE - Real trades will be executed!")
        else:
            self.executor = DryRunExecutor(
                web3=self.web3,
                private_key='0x' + '0' * 64,  # Dummy key for dry run
                ctf_exchange_address=CTF_EXCHANGE_ADDRESS,
                copy_percentage=copy_percentage,
                dynamic_sizing=DYNAMIC_SIZING,
                max_trade_size=MAX_TRADE_SIZE,
                min_trade_size=MIN_TRADE_SIZE,
                max_slippage=MAX_SLIPPAGE,
                max_exposure_percent=MAX_EXPOSURE_PERCENT,
                max_positions=MAX_POSITIONS,
                min_balance=MIN_BALANCE_USDC,
            )
            self.logger.info("DRY RUN MODE - No real trades will be executed")

        # Statistics
        self.trades_detected = 0
        self.trades_copied = 0
        self.start_time = None

    def on_transaction(self, tx: TransactionInfo):
        """Handle detected transaction"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"New transaction detected: {tx.tx_hash}")
        self.logger.info(f"Block: {tx.block_number} | Time: {tx.timestamp}")

        # Decode the transaction
        trades = self.decoder.decode_transaction(
            tx_input=tx.input_data,
            tx_hash=tx.tx_hash,
            logs=tx.logs
        )

        if not trades:
            self.logger.info("No Polymarket trades found in transaction")
            return

        self.trades_detected += len(trades)

        for trade in trades:
            self.logger.info(f"\nDecoded trade:")
            self.logger.info(f"  Side: {trade.side.value}")
            self.logger.info(f"  Outcome: {trade.outcome.value}")
            self.logger.info(f"  Amount: {trade.amount:.4f} shares")
            self.logger.info(f"  Price: ${trade.price:.4f}")
            self.logger.info(f"  Total: ${trade.total_value:.2f}")
            self.logger.info(f"  Token ID: {trade.token_id[:20]}...")

            # Execute copy trade
            self.logger.info(f"\nExecuting copy trade ({self.copy_percentage*100:.0f}% of original)...")

            result = self.executor.execute_copy_trade(trade)

            if result.success:
                self.trades_copied += 1
                self.logger.info(f"Copy trade successful!")
                self.logger.info(f"  Copied: ${result.copy_amount:.2f} @ ${result.copy_price:.4f}")
                self.logger.info(f"  TX: {result.tx_hash}")
            else:
                self.logger.warning(f"Copy trade failed: {result.error}")

        self.logger.info(f"\nStats: {self.trades_detected} detected, {self.trades_copied} copied")

    def on_error(self, error: Exception):
        """Handle errors"""
        self.logger.error(f"Error: {error}")

    def start(self):
        """Start the copy trading bot"""
        self.start_time = datetime.now()

        self.logger.info("\n" + "="*60)
        self.logger.info("POLYMARKET COPY TRADING BOT")
        self.logger.info("="*60)
        self.logger.info(f"Target wallet: {self.target_wallet}")
        self.logger.info(f"Copy percentage: {self.copy_percentage*100:.0f}%")
        self.logger.info(f"Poll interval: {POLL_INTERVAL}s")
        self.logger.info(f"Mode: {'LIVE' if self.live_mode else 'DRY RUN'}")
        self.logger.info(f"Max trade size: ${MAX_TRADE_SIZE}")
        self.logger.info(f"Max slippage: {MAX_SLIPPAGE*100:.1f}%")
        self.logger.info("="*60 + "\n")

        # Handle graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info("\nShutdown signal received...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start monitoring
        self.monitor.start_monitoring(
            callback=self.on_transaction,
            error_callback=self.on_error
        )

    def stop(self):
        """Stop the bot"""
        self.monitor.stop_monitoring()

        # Print final stats
        runtime = datetime.now() - self.start_time if self.start_time else None
        stats = self.executor.get_execution_stats()

        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("="*60)
        self.logger.info(f"Runtime: {runtime}")
        self.logger.info(f"Trades detected: {self.trades_detected}")
        self.logger.info(f"Trades copied: {stats['successful']}")
        self.logger.info(f"Trades failed: {stats['failed']}")
        self.logger.info(f"Total volume: ${stats['total_volume_usdc']:.2f}")
        self.logger.info("="*60)

    def get_status(self) -> dict:
        """Get current bot status"""
        return {
            "target_wallet": self.target_wallet,
            "live_mode": self.live_mode,
            "copy_percentage": self.copy_percentage,
            "trades_detected": self.trades_detected,
            "trades_copied": self.trades_copied,
            "monitor_stats": self.monitor.get_stats(),
            "executor_stats": self.executor.get_execution_stats(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }


def cmd_monitor(args):
    """Start monitoring command"""
    logger = setup_logging()

    try:
        bot = CopyTradingBot(
            target_wallet=args.wallet or TARGET_WALLET,
            live_mode=args.live,
            copy_percentage=args.copy_pct / 100.0,
        )
        bot.start()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


def cmd_status(args):
    """Show status command"""
    logger = setup_logging()

    # Get RPC connection
    web3 = Web3(Web3.HTTPProvider(POLYGON_RPC_PRIMARY))

    if not web3.is_connected():
        print("Error: Cannot connect to Polygon RPC")
        return

    print("\n" + "="*50)
    print("POLYMARKET COPY TRADING STATUS")
    print("="*50)

    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Target wallet: {TARGET_WALLET}")
    print(f"  Copy percentage: {COPY_PERCENTAGE*100:.0f}%")
    print(f"  Max trade size: ${MAX_TRADE_SIZE}")
    print(f"  Poll interval: {POLL_INTERVAL}s")

    # Check wallet
    private_key = os.environ.get('POLY_PRIVATE_KEY')
    if private_key:
        from eth_account import Account
        account = Account.from_key(private_key)
        print(f"\nYour wallet: {account.address}")

        # Get USDC balance
        usdc_contract = web3.eth.contract(
            address=Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
            abi=[{
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }]
        )
        balance = usdc_contract.functions.balanceOf(account.address).call() / 1e6
        print(f"  USDC Balance: ${balance:.2f}")
    else:
        print("\n  POLY_PRIVATE_KEY not set (required for live trading)")

    # Show current block
    print(f"\nNetwork:")
    print(f"  Current block: {web3.eth.block_number}")
    print(f"  Chain ID: {web3.eth.chain_id}")

    print("\n" + "="*50)


def cmd_history(args):
    """Show trade history"""
    logger = setup_logging()

    history_file = Path("data/copytrade_logs/trade_history.json")

    if not history_file.exists():
        print("No trade history found")
        return

    with open(history_file) as f:
        history = json.load(f)

    print("\n" + "="*60)
    print("COPY TRADE HISTORY")
    print("="*60)

    for trade in history[-20:]:  # Last 20 trades
        print(f"\n{trade['timestamp']}")
        print(f"  Original: {trade['original_tx'][:20]}...")
        print(f"  Side: {trade['side']} | Amount: ${trade['amount']:.2f}")
        print(f"  Status: {'SUCCESS' if trade['success'] else 'FAILED'}")
        if trade.get('copy_tx'):
            print(f"  Copy TX: {trade['copy_tx'][:20]}...")

    print("\n" + "="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Polymarket Copy Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start monitoring and copy trading')
    monitor_parser.add_argument(
        '--wallet', '-w',
        help=f'Wallet to monitor (default: {TARGET_WALLET[:10]}...)'
    )
    monitor_parser.add_argument(
        '--live', '-l',
        action='store_true',
        help='Enable live trading (requires POLY_PRIVATE_KEY)'
    )
    monitor_parser.add_argument(
        '--copy-pct', '-c',
        type=float,
        default=COPY_PERCENTAGE * 100,
        help=f'Copy percentage (default: {COPY_PERCENTAGE*100:.0f}%%)'
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show current status')
    status_parser.set_defaults(func=cmd_status)

    # History command
    history_parser = subparsers.add_parser('history', help='Show trade history')
    history_parser.set_defaults(func=cmd_history)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
