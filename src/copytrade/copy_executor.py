"""
Copy Trade Executor for Polymarket
Executes copy trades based on monitored transactions
"""

import os
import time
import logging
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from web3 import Web3
from eth_account import Account
from eth_account.signers.local import LocalAccount

from .polymarket_decoder import PolymarketTrade, TradeSide

logger = logging.getLogger(__name__)


@dataclass
class CopyTradeResult:
    """Result of a copy trade execution"""
    success: bool
    original_trade: PolymarketTrade
    copy_amount: float
    copy_price: float
    tx_hash: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    usdc_balance: float
    positions: Dict[str, float]  # token_id -> amount
    pending_orders: List[str]
    total_value: float
    exposure_percent: float


class CopyTradeExecutor:
    """
    Executes copy trades on Polymarket

    Supports:
    - Dynamic sizing based on balance
    - Slippage protection
    - Risk limits
    - Trade logging
    """

    # USDC on Polygon
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    USDC_DECIMALS = 6

    def __init__(
        self,
        web3: Web3,
        private_key: str,
        ctf_exchange_address: str,
        clob_api_base: str = "https://clob.polymarket.com",
        copy_percentage: float = 0.10,
        dynamic_sizing: bool = True,
        max_trade_size: float = 100.0,
        min_trade_size: float = 1.0,
        max_slippage: float = 0.02,
        max_exposure_percent: float = 0.50,
        max_positions: int = 10,
        min_balance: float = 10.0,
    ):
        """
        Initialize copy trade executor

        Args:
            web3: Web3 instance
            private_key: Private key for signing transactions
            ctf_exchange_address: CTF Exchange contract address
            clob_api_base: Polymarket CLOB API base URL
            copy_percentage: Percentage of original trade to copy (0.10 = 10%)
            dynamic_sizing: If True, adjust size based on balance
            max_trade_size: Maximum trade size in USDC
            min_trade_size: Minimum trade size (skip smaller)
            max_slippage: Maximum allowed slippage
            max_exposure_percent: Max portfolio exposure
            max_positions: Maximum concurrent positions
            min_balance: Stop trading below this balance
        """
        self.web3 = web3
        self.ctf_exchange_address = Web3.to_checksum_address(ctf_exchange_address)
        self.clob_api_base = clob_api_base

        # Trading parameters
        self.copy_percentage = copy_percentage
        self.dynamic_sizing = dynamic_sizing
        self.max_trade_size = max_trade_size
        self.min_trade_size = min_trade_size
        self.max_slippage = max_slippage
        self.max_exposure_percent = max_exposure_percent
        self.max_positions = max_positions
        self.min_balance = min_balance

        # Setup account
        self.account: LocalAccount = Account.from_key(private_key)
        self.wallet_address = self.account.address
        logger.info(f"Initialized executor for wallet: {self.wallet_address}")

        # State tracking
        self.executed_trades: List[CopyTradeResult] = []
        self.pending_orders: Dict[str, Any] = {}
        self.positions: Dict[str, float] = {}

        # HTTP client for API calls
        self.http_client = httpx.Client(timeout=30)

        # Load USDC contract for balance checks
        self.usdc_contract = self.web3.eth.contract(
            address=Web3.to_checksum_address(self.USDC_ADDRESS),
            abi=[
                {
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                },
                {
                    "constant": False,
                    "inputs": [
                        {"name": "_spender", "type": "address"},
                        {"name": "_value", "type": "uint256"}
                    ],
                    "name": "approve",
                    "outputs": [{"name": "", "type": "bool"}],
                    "type": "function"
                },
                {
                    "constant": True,
                    "inputs": [
                        {"name": "_owner", "type": "address"},
                        {"name": "_spender", "type": "address"}
                    ],
                    "name": "allowance",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "type": "function"
                }
            ]
        )

    def get_usdc_balance(self) -> float:
        """Get current USDC balance"""
        try:
            balance_wei = self.usdc_contract.functions.balanceOf(
                self.wallet_address
            ).call()
            return balance_wei / (10 ** self.USDC_DECIMALS)
        except Exception as e:
            logger.error(f"Error getting USDC balance: {e}")
            return 0.0

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        usdc_balance = self.get_usdc_balance()

        # Calculate total value (USDC + positions)
        # In production, you'd value positions at current market prices
        position_value = sum(self.positions.values())
        total_value = usdc_balance + position_value

        exposure_percent = position_value / total_value if total_value > 0 else 0

        return PortfolioState(
            usdc_balance=usdc_balance,
            positions=self.positions.copy(),
            pending_orders=list(self.pending_orders.keys()),
            total_value=total_value,
            exposure_percent=exposure_percent
        )

    def calculate_copy_size(self, original_trade: PolymarketTrade) -> float:
        """
        Calculate the size to copy

        Args:
            original_trade: The trade to copy

        Returns:
            Amount in USDC to trade
        """
        # Base copy size
        base_size = original_trade.total_value * self.copy_percentage

        if self.dynamic_sizing:
            # Adjust based on current balance
            portfolio = self.get_portfolio_state()

            # Don't trade if balance too low
            if portfolio.usdc_balance < self.min_balance:
                logger.warning(f"Balance too low: {portfolio.usdc_balance}")
                return 0.0

            # Don't exceed exposure limit
            max_allowed = portfolio.usdc_balance * (
                self.max_exposure_percent - portfolio.exposure_percent
            )
            base_size = min(base_size, max_allowed)

        # Apply limits
        if base_size < self.min_trade_size:
            logger.debug(f"Trade too small: {base_size}")
            return 0.0

        if base_size > self.max_trade_size:
            base_size = self.max_trade_size

        return base_size

    def check_can_trade(self, trade: PolymarketTrade) -> tuple[bool, str]:
        """
        Check if we can execute a copy trade

        Returns:
            Tuple of (can_trade, reason)
        """
        portfolio = self.get_portfolio_state()

        # Check balance
        if portfolio.usdc_balance < self.min_balance:
            return False, f"Balance too low: {portfolio.usdc_balance}"

        # Check exposure
        if portfolio.exposure_percent >= self.max_exposure_percent:
            return False, f"Max exposure reached: {portfolio.exposure_percent:.1%}"

        # Check positions limit
        if len(portfolio.positions) >= self.max_positions:
            return False, f"Max positions reached: {len(portfolio.positions)}"

        return True, "OK"

    def get_current_price(self, token_id: str) -> Optional[float]:
        """
        Get current market price for a token

        Args:
            token_id: Token ID to get price for

        Returns:
            Current price or None if unavailable
        """
        try:
            # Query CLOB API for current book
            response = self.http_client.get(
                f"{self.clob_api_base}/book",
                params={"token_id": token_id}
            )

            if response.status_code == 200:
                data = response.json()
                # Get best bid/ask
                bids = data.get('bids', [])
                asks = data.get('asks', [])

                if bids and asks:
                    best_bid = float(bids[0]['price'])
                    best_ask = float(asks[0]['price'])
                    return (best_bid + best_ask) / 2

            return None

        except Exception as e:
            logger.error(f"Error getting price for {token_id}: {e}")
            return None

    def check_slippage(
        self,
        original_price: float,
        current_price: float,
        side: TradeSide
    ) -> bool:
        """
        Check if slippage is acceptable

        Args:
            original_price: Price at which monitored trader executed
            current_price: Current market price
            side: Trade side (BUY or SELL)

        Returns:
            True if slippage is acceptable
        """
        if side == TradeSide.BUY:
            # For buys, current price should not be much higher
            slippage = (current_price - original_price) / original_price
        else:
            # For sells, current price should not be much lower
            slippage = (original_price - current_price) / original_price

        if slippage > self.max_slippage:
            logger.warning(f"Slippage too high: {slippage:.2%} > {self.max_slippage:.2%}")
            return False

        return True

    def execute_copy_trade(self, original_trade: PolymarketTrade) -> CopyTradeResult:
        """
        Execute a copy trade

        Args:
            original_trade: The trade to copy

        Returns:
            CopyTradeResult with execution details
        """
        logger.info(f"Processing copy trade for {original_trade.tx_hash}")
        logger.info(
            f"Original: {original_trade.side.value} {original_trade.amount:.4f} "
            f"@ ${original_trade.price:.4f} = ${original_trade.total_value:.2f}"
        )

        # Check if we can trade
        can_trade, reason = self.check_can_trade(original_trade)
        if not can_trade:
            return CopyTradeResult(
                success=False,
                original_trade=original_trade,
                copy_amount=0,
                copy_price=0,
                error=reason
            )

        # Calculate copy size
        copy_size = self.calculate_copy_size(original_trade)
        if copy_size == 0:
            return CopyTradeResult(
                success=False,
                original_trade=original_trade,
                copy_amount=0,
                copy_price=0,
                error="Copy size too small or balance insufficient"
            )

        # Get current price
        current_price = self.get_current_price(original_trade.token_id)
        if current_price is None:
            # Use original price if can't get current
            current_price = original_trade.price
            logger.warning(f"Using original price: ${current_price}")

        # Check slippage
        if not self.check_slippage(original_trade.price, current_price, original_trade.side):
            return CopyTradeResult(
                success=False,
                original_trade=original_trade,
                copy_amount=copy_size,
                copy_price=current_price,
                error=f"Slippage too high"
            )

        # Calculate shares to buy
        shares_amount = copy_size / current_price

        logger.info(
            f"Copy trade: {original_trade.side.value} {shares_amount:.4f} shares "
            f"@ ${current_price:.4f} = ${copy_size:.2f}"
        )

        # Execute trade via CLOB API
        try:
            tx_hash = self._execute_via_clob(
                token_id=original_trade.token_id,
                side=original_trade.side,
                amount=shares_amount,
                price=current_price
            )

            if tx_hash:
                # Update positions
                if original_trade.side == TradeSide.BUY:
                    self.positions[original_trade.token_id] = (
                        self.positions.get(original_trade.token_id, 0) + copy_size
                    )

                result = CopyTradeResult(
                    success=True,
                    original_trade=original_trade,
                    copy_amount=copy_size,
                    copy_price=current_price,
                    tx_hash=tx_hash
                )

                self.executed_trades.append(result)
                logger.info(f"Copy trade executed: {tx_hash}")
                return result

            return CopyTradeResult(
                success=False,
                original_trade=original_trade,
                copy_amount=copy_size,
                copy_price=current_price,
                error="Failed to execute trade"
            )

        except Exception as e:
            logger.error(f"Error executing copy trade: {e}")
            return CopyTradeResult(
                success=False,
                original_trade=original_trade,
                copy_amount=copy_size,
                copy_price=current_price,
                error=str(e)
            )

    def _execute_via_clob(
        self,
        token_id: str,
        side: TradeSide,
        amount: float,
        price: float
    ) -> Optional[str]:
        """
        Execute trade via Polymarket CLOB API

        This is a simplified implementation - in production you'd need:
        1. Proper EIP-712 order signing
        2. Order matching via API
        3. On-chain settlement monitoring

        Args:
            token_id: Token to trade
            side: BUY or SELL
            amount: Amount of shares
            price: Price per share

        Returns:
            Transaction hash if successful
        """
        # Build order
        order = {
            "tokenId": token_id,
            "side": "BUY" if side == TradeSide.BUY else "SELL",
            "size": str(amount),
            "price": str(price),
            "type": "GTC",  # Good til cancelled
        }

        logger.info(f"Placing order: {order}")

        # In production, you would:
        # 1. Sign the order with EIP-712
        # 2. POST to /order endpoint
        # 3. Wait for fill
        # 4. Return the settlement tx hash

        # For now, this is a simulation placeholder
        # The actual implementation requires Polymarket's specific signing scheme

        # Simulate order placement
        logger.warning(
            "SIMULATION MODE: Order would be placed via CLOB API. "
            "Real execution requires API key and proper order signing."
        )

        # Return a simulated tx hash
        return f"0x_simulated_{int(time.time())}"

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_trades = len(self.executed_trades)
        successful = sum(1 for t in self.executed_trades if t.success)
        failed = total_trades - successful

        total_volume = sum(t.copy_amount for t in self.executed_trades if t.success)

        return {
            "total_trades": total_trades,
            "successful": successful,
            "failed": failed,
            "total_volume_usdc": total_volume,
            "current_positions": len(self.positions),
            "portfolio": self.get_portfolio_state().__dict__
        }


class DryRunExecutor(CopyTradeExecutor):
    """
    Dry run executor for testing without real trades
    """

    def __init__(self, *args, **kwargs):
        # Remove private key requirement for dry run
        kwargs['private_key'] = '0x' + '0' * 64  # Dummy key
        super().__init__(*args, **kwargs)
        self.simulated_balance = 1000.0  # Simulated starting balance
        logger.info("Running in DRY RUN mode - no real trades will be executed")

    def get_usdc_balance(self) -> float:
        """Return simulated balance"""
        return self.simulated_balance

    def _execute_via_clob(
        self,
        token_id: str,
        side: TradeSide,
        amount: float,
        price: float
    ) -> Optional[str]:
        """Simulate trade execution"""
        trade_value = amount * price

        if side == TradeSide.BUY:
            self.simulated_balance -= trade_value
        else:
            self.simulated_balance += trade_value

        logger.info(
            f"[DRY RUN] {side.value} {amount:.4f} @ ${price:.4f} = ${trade_value:.2f}"
        )
        logger.info(f"[DRY RUN] New balance: ${self.simulated_balance:.2f}")

        return f"0x_dryrun_{int(time.time())}"
