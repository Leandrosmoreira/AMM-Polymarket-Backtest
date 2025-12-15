"""
Blockchain Monitor for Polymarket Copy Trading
Monitors a target wallet for transactions on Polygon
"""

import time
import logging
from typing import Optional, Callable, List, Dict, Any
from web3 import Web3
from web3.exceptions import BlockNotFound
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TransactionInfo:
    """Represents a monitored transaction"""
    tx_hash: str
    block_number: int
    from_address: str
    to_address: str
    value: int
    input_data: str
    gas_price: int
    gas_used: Optional[int]
    timestamp: datetime
    status: Optional[int]  # 1 = success, 0 = failed
    logs: List[Dict[str, Any]]


class BlockchainMonitor:
    """
    Monitors blockchain for transactions from a specific wallet
    Polls every N seconds and calls callback for new transactions
    """

    def __init__(
        self,
        rpc_urls: List[str],
        target_wallet: str,
        contract_addresses: List[str],
        poll_interval: float = 1.0,
        lookback_blocks: int = 10,
        confirmation_blocks: int = 2
    ):
        """
        Initialize the blockchain monitor

        Args:
            rpc_urls: List of RPC URLs to try
            target_wallet: Wallet address to monitor
            contract_addresses: Contract addresses to filter (e.g., CTF Exchange)
            poll_interval: How often to poll for new blocks (seconds)
            lookback_blocks: Number of blocks to look back on start
            confirmation_blocks: Wait this many confirmations before processing
        """
        self.rpc_urls = rpc_urls
        self.target_wallet = Web3.to_checksum_address(target_wallet)
        self.contract_addresses = [
            Web3.to_checksum_address(addr) for addr in contract_addresses
        ]
        self.poll_interval = poll_interval
        self.lookback_blocks = lookback_blocks
        self.confirmation_blocks = confirmation_blocks

        self.web3: Optional[Web3] = None
        self.current_rpc_index = 0
        self.last_processed_block = 0
        self.processed_tx_hashes = set()
        self.is_running = False

        self._connect()

    def _connect(self) -> bool:
        """Connect to an RPC endpoint"""
        for i, rpc_url in enumerate(self.rpc_urls):
            try:
                web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 30}))
                if web3.is_connected():
                    self.web3 = web3
                    self.current_rpc_index = i
                    logger.info(f"Connected to RPC: {rpc_url}")
                    return True
            except Exception as e:
                logger.warning(f"Failed to connect to {rpc_url}: {e}")
                continue

        logger.error("Failed to connect to any RPC endpoint")
        return False

    def _switch_rpc(self) -> bool:
        """Switch to next RPC endpoint"""
        original_index = self.current_rpc_index

        for _ in range(len(self.rpc_urls)):
            self.current_rpc_index = (self.current_rpc_index + 1) % len(self.rpc_urls)
            if self._connect():
                return True

        self.current_rpc_index = original_index
        return False

    def get_current_block(self) -> int:
        """Get the current block number"""
        try:
            return self.web3.eth.block_number
        except Exception as e:
            logger.error(f"Error getting block number: {e}")
            self._switch_rpc()
            return self.web3.eth.block_number

    def get_transaction(self, tx_hash: str) -> Optional[TransactionInfo]:
        """Get full transaction info including receipt"""
        try:
            tx = self.web3.eth.get_transaction(tx_hash)
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            block = self.web3.eth.get_block(tx['blockNumber'])

            return TransactionInfo(
                tx_hash=tx_hash,
                block_number=tx['blockNumber'],
                from_address=tx['from'],
                to_address=tx['to'] if tx['to'] else '',
                value=tx['value'],
                input_data=tx['input'].hex() if isinstance(tx['input'], bytes) else tx['input'],
                gas_price=tx['gasPrice'],
                gas_used=receipt['gasUsed'] if receipt else None,
                timestamp=datetime.fromtimestamp(block['timestamp']),
                status=receipt['status'] if receipt else None,
                logs=[dict(log) for log in receipt['logs']] if receipt else []
            )
        except Exception as e:
            logger.error(f"Error getting transaction {tx_hash}: {e}")
            return None

    def get_transactions_for_wallet(
        self,
        from_block: int,
        to_block: int
    ) -> List[TransactionInfo]:
        """
        Get all transactions from the target wallet in a block range
        that interact with monitored contracts
        """
        transactions = []

        for block_num in range(from_block, to_block + 1):
            try:
                block = self.web3.eth.get_block(block_num, full_transactions=True)

                for tx in block['transactions']:
                    # Check if transaction is from our target wallet
                    if tx['from'].lower() != self.target_wallet.lower():
                        continue

                    # Check if it's to one of our monitored contracts
                    if tx['to'] and tx['to'].lower() not in [
                        addr.lower() for addr in self.contract_addresses
                    ]:
                        continue

                    # Skip if already processed
                    tx_hash = tx['hash'].hex()
                    if tx_hash in self.processed_tx_hashes:
                        continue

                    # Get full transaction info
                    tx_info = self.get_transaction(tx_hash)
                    if tx_info and tx_info.status == 1:  # Only successful txs
                        transactions.append(tx_info)
                        self.processed_tx_hashes.add(tx_hash)
                        logger.info(f"Found transaction: {tx_hash}")

            except BlockNotFound:
                logger.warning(f"Block {block_num} not found")
                continue
            except Exception as e:
                logger.error(f"Error processing block {block_num}: {e}")
                continue

        return transactions

    def start_monitoring(
        self,
        callback: Callable[[TransactionInfo], None],
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """
        Start monitoring for transactions

        Args:
            callback: Function to call when a new transaction is found
            error_callback: Function to call on errors
        """
        self.is_running = True

        # Initialize from current block minus lookback
        current_block = self.get_current_block()
        self.last_processed_block = current_block - self.lookback_blocks

        logger.info(f"Starting monitor from block {self.last_processed_block}")
        logger.info(f"Monitoring wallet: {self.target_wallet}")
        logger.info(f"Monitoring contracts: {self.contract_addresses}")

        while self.is_running:
            try:
                current_block = self.get_current_block()
                safe_block = current_block - self.confirmation_blocks

                if safe_block > self.last_processed_block:
                    # Get new transactions
                    transactions = self.get_transactions_for_wallet(
                        self.last_processed_block + 1,
                        safe_block
                    )

                    # Call callback for each transaction
                    for tx in transactions:
                        try:
                            callback(tx)
                        except Exception as e:
                            logger.error(f"Error in callback for {tx.tx_hash}: {e}")
                            if error_callback:
                                error_callback(e)

                    self.last_processed_block = safe_block

                    if transactions:
                        logger.debug(f"Processed {len(transactions)} transactions up to block {safe_block}")

                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                self.is_running = False
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                if error_callback:
                    error_callback(e)

                # Try to switch RPC
                if not self._switch_rpc():
                    logger.error("All RPCs failed, waiting before retry...")
                    time.sleep(5)

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.is_running = False
        logger.info("Monitoring stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "target_wallet": self.target_wallet,
            "last_processed_block": self.last_processed_block,
            "transactions_found": len(self.processed_tx_hashes),
            "is_running": self.is_running,
            "current_rpc": self.rpc_urls[self.current_rpc_index] if self.web3 else None
        }
