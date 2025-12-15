"""
Polymarket Transaction Decoder
Decodes Polymarket CTF Exchange transactions to extract trade details
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from web3 import Web3
from eth_abi import decode

logger = logging.getLogger(__name__)


class TradeSide(Enum):
    """Trade side - buying or selling"""
    BUY = "BUY"
    SELL = "SELL"


class TradeOutcome(Enum):
    """Outcome being traded"""
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


@dataclass
class PolymarketTrade:
    """Decoded Polymarket trade information"""
    tx_hash: str
    trader: str
    market_id: str
    token_id: str
    side: TradeSide
    outcome: TradeOutcome
    amount: float  # Amount of shares
    price: float  # Price per share
    total_value: float  # Total USDC value
    timestamp: int
    raw_data: Dict[str, Any]


# CTF Exchange ABI - Key functions and events
CTF_EXCHANGE_ABI = [
    # fillOrder function
    {
        "inputs": [
            {
                "components": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
                "name": "order",
                "type": "tuple"
            },
            {"name": "fillAmount", "type": "uint256"},
        ],
        "name": "fillOrder",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # fillOrders function (batch)
    {
        "inputs": [
            {
                "components": [
                    {"name": "salt", "type": "uint256"},
                    {"name": "maker", "type": "address"},
                    {"name": "signer", "type": "address"},
                    {"name": "taker", "type": "address"},
                    {"name": "tokenId", "type": "uint256"},
                    {"name": "makerAmount", "type": "uint256"},
                    {"name": "takerAmount", "type": "uint256"},
                    {"name": "expiration", "type": "uint256"},
                    {"name": "nonce", "type": "uint256"},
                    {"name": "feeRateBps", "type": "uint256"},
                    {"name": "side", "type": "uint8"},
                    {"name": "signatureType", "type": "uint8"},
                ],
                "name": "orders",
                "type": "tuple[]"
            },
            {"name": "fillAmounts", "type": "uint256[]"},
        ],
        "name": "fillOrders",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    # OrderFilled event
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "orderHash", "type": "bytes32"},
            {"indexed": True, "name": "maker", "type": "address"},
            {"indexed": True, "name": "taker", "type": "address"},
            {"indexed": False, "name": "makerAssetId", "type": "uint256"},
            {"indexed": False, "name": "takerAssetId", "type": "uint256"},
            {"indexed": False, "name": "makerAmountFilled", "type": "uint256"},
            {"indexed": False, "name": "takerAmountFilled", "type": "uint256"},
            {"indexed": False, "name": "fee", "type": "uint256"},
        ],
        "name": "OrderFilled",
        "type": "event"
    },
    # OrdersMatched event
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "takerOrderHash", "type": "bytes32"},
            {"indexed": True, "name": "takerOrderMaker", "type": "address"},
            {"indexed": False, "name": "makerAssetId", "type": "uint256"},
            {"indexed": False, "name": "takerAssetId", "type": "uint256"},
            {"indexed": False, "name": "makerAmountFilled", "type": "uint256"},
            {"indexed": False, "name": "takerAmountFilled", "type": "uint256"},
        ],
        "name": "OrdersMatched",
        "type": "event"
    },
]

# Function selectors (first 4 bytes of keccak256 hash)
FUNCTION_SELECTORS = {
    "fillOrder": "0x8a21a636",
    "fillOrders": "0xa6cf08f6",
    "matchOrders": "0x6ac9f82b",
}


class PolymarketDecoder:
    """
    Decodes Polymarket CTF Exchange transactions
    """

    def __init__(self, web3: Web3, ctf_exchange_address: str):
        """
        Initialize decoder

        Args:
            web3: Web3 instance
            ctf_exchange_address: Address of CTF Exchange contract
        """
        self.web3 = web3
        self.ctf_exchange_address = Web3.to_checksum_address(ctf_exchange_address)

        # Create contract instance
        self.contract = self.web3.eth.contract(
            address=self.ctf_exchange_address,
            abi=CTF_EXCHANGE_ABI
        )

        # Event signatures
        self.order_filled_topic = self.web3.keccak(
            text="OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
        )
        self.orders_matched_topic = self.web3.keccak(
            text="OrdersMatched(bytes32,address,uint256,uint256,uint256,uint256)"
        )

    def decode_transaction(self, tx_input: str, tx_hash: str, logs: List[Dict]) -> List[PolymarketTrade]:
        """
        Decode a transaction's input data and logs

        Args:
            tx_input: Transaction input data (hex string)
            tx_hash: Transaction hash
            logs: Transaction logs from receipt

        Returns:
            List of decoded trades
        """
        trades = []

        # First, try to decode from logs (more reliable)
        log_trades = self._decode_from_logs(logs, tx_hash)
        if log_trades:
            return log_trades

        # Fallback: decode from input data
        if not tx_input or tx_input == "0x":
            return trades

        try:
            # Get function selector
            selector = tx_input[:10]

            if selector == FUNCTION_SELECTORS.get("fillOrder"):
                trades = self._decode_fill_order(tx_input, tx_hash)
            elif selector == FUNCTION_SELECTORS.get("fillOrders"):
                trades = self._decode_fill_orders(tx_input, tx_hash)

        except Exception as e:
            logger.error(f"Error decoding transaction {tx_hash}: {e}")

        return trades

    def _decode_from_logs(self, logs: List[Dict], tx_hash: str) -> List[PolymarketTrade]:
        """Decode trades from event logs"""
        trades = []

        for log in logs:
            try:
                # Check if this is from CTF Exchange
                if log.get('address', '').lower() != self.ctf_exchange_address.lower():
                    continue

                topics = log.get('topics', [])
                if not topics:
                    continue

                # Check for OrderFilled event
                topic0 = topics[0].hex() if isinstance(topics[0], bytes) else topics[0]

                if topic0 == self.order_filled_topic.hex():
                    trade = self._decode_order_filled_event(log, tx_hash)
                    if trade:
                        trades.append(trade)

            except Exception as e:
                logger.warning(f"Error decoding log: {e}")
                continue

        return trades

    def _decode_order_filled_event(self, log: Dict, tx_hash: str) -> Optional[PolymarketTrade]:
        """Decode OrderFilled event"""
        try:
            topics = log.get('topics', [])
            data = log.get('data', '0x')

            # Extract indexed parameters from topics
            # topics[0] = event signature
            # topics[1] = orderHash
            # topics[2] = maker
            # topics[3] = taker
            maker = '0x' + topics[2].hex()[-40:] if len(topics) > 2 else None
            taker = '0x' + topics[3].hex()[-40:] if len(topics) > 3 else None

            if not maker or not taker:
                return None

            # Decode non-indexed parameters from data
            data_bytes = bytes.fromhex(data[2:]) if data.startswith('0x') else bytes.fromhex(data)

            decoded = decode(
                ['uint256', 'uint256', 'uint256', 'uint256', 'uint256'],
                data_bytes
            )

            maker_asset_id = decoded[0]
            taker_asset_id = decoded[1]
            maker_amount_filled = decoded[2]
            taker_amount_filled = decoded[3]
            fee = decoded[4]

            # Determine trade direction and amounts
            # In CTF Exchange:
            # - makerAssetId = 0 means maker is selling USDC (buying shares)
            # - takerAssetId = 0 means taker is selling USDC (buying shares)
            # USDC has assetId 0, conditional tokens have their tokenId

            if maker_asset_id == 0:
                # Maker is providing USDC, taker is providing shares
                # Taker is SELLING shares
                side = TradeSide.SELL
                token_id = str(taker_asset_id)
                amount = taker_amount_filled / 1e6  # Shares have 6 decimals
                total_value = maker_amount_filled / 1e6  # USDC has 6 decimals
            else:
                # Maker is providing shares, taker is providing USDC
                # Taker is BUYING shares
                side = TradeSide.BUY
                token_id = str(maker_asset_id)
                amount = maker_amount_filled / 1e6
                total_value = taker_amount_filled / 1e6

            price = total_value / amount if amount > 0 else 0

            # Determine YES/NO from token ID
            outcome = self._get_outcome_from_token_id(token_id)

            return PolymarketTrade(
                tx_hash=tx_hash,
                trader=taker,
                market_id=self._get_market_id_from_token(token_id),
                token_id=token_id,
                side=side,
                outcome=outcome,
                amount=amount,
                price=price,
                total_value=total_value,
                timestamp=0,
                raw_data={
                    "maker": maker,
                    "taker": taker,
                    "maker_asset_id": maker_asset_id,
                    "taker_asset_id": taker_asset_id,
                    "maker_amount": maker_amount_filled,
                    "taker_amount": taker_amount_filled,
                    "fee": fee
                }
            )

        except Exception as e:
            logger.error(f"Error decoding OrderFilled event: {e}")
            return None

    def _decode_fill_order(self, tx_input: str, tx_hash: str) -> List[PolymarketTrade]:
        """Decode fillOrder function call"""
        try:
            # Decode function inputs
            decoded = self.contract.decode_function_input(tx_input)
            func_name, params = decoded

            order = params.get('order', {})
            fill_amount = params.get('fillAmount', 0)

            token_id = str(order.get('tokenId', 0))
            maker_amount = order.get('makerAmount', 0)
            taker_amount = order.get('takerAmount', 0)
            side_num = order.get('side', 0)

            # Side: 0 = BUY, 1 = SELL
            side = TradeSide.BUY if side_num == 0 else TradeSide.SELL

            # Calculate amounts based on fill
            if maker_amount > 0:
                fill_ratio = fill_amount / maker_amount
                actual_taker_amount = int(taker_amount * fill_ratio)
            else:
                fill_ratio = 1.0
                actual_taker_amount = taker_amount

            amount = fill_amount / 1e6
            total_value = actual_taker_amount / 1e6
            price = total_value / amount if amount > 0 else 0

            outcome = self._get_outcome_from_token_id(token_id)

            return [PolymarketTrade(
                tx_hash=tx_hash,
                trader=order.get('taker', ''),
                market_id=self._get_market_id_from_token(token_id),
                token_id=token_id,
                side=side,
                outcome=outcome,
                amount=amount,
                price=price,
                total_value=total_value,
                timestamp=0,
                raw_data=dict(order)
            )]

        except Exception as e:
            logger.error(f"Error decoding fillOrder: {e}")
            return []

    def _decode_fill_orders(self, tx_input: str, tx_hash: str) -> List[PolymarketTrade]:
        """Decode fillOrders batch function call"""
        try:
            decoded = self.contract.decode_function_input(tx_input)
            func_name, params = decoded

            orders = params.get('orders', [])
            fill_amounts = params.get('fillAmounts', [])

            trades = []
            for i, order in enumerate(orders):
                fill_amount = fill_amounts[i] if i < len(fill_amounts) else 0

                token_id = str(order.get('tokenId', 0))
                maker_amount = order.get('makerAmount', 0)
                taker_amount = order.get('takerAmount', 0)
                side_num = order.get('side', 0)

                side = TradeSide.BUY if side_num == 0 else TradeSide.SELL

                if maker_amount > 0:
                    fill_ratio = fill_amount / maker_amount
                    actual_taker_amount = int(taker_amount * fill_ratio)
                else:
                    fill_ratio = 1.0
                    actual_taker_amount = taker_amount

                amount = fill_amount / 1e6
                total_value = actual_taker_amount / 1e6
                price = total_value / amount if amount > 0 else 0

                outcome = self._get_outcome_from_token_id(token_id)

                trades.append(PolymarketTrade(
                    tx_hash=tx_hash,
                    trader=order.get('taker', ''),
                    market_id=self._get_market_id_from_token(token_id),
                    token_id=token_id,
                    side=side,
                    outcome=outcome,
                    amount=amount,
                    price=price,
                    total_value=total_value,
                    timestamp=0,
                    raw_data=dict(order)
                ))

            return trades

        except Exception as e:
            logger.error(f"Error decoding fillOrders: {e}")
            return []

    def _get_outcome_from_token_id(self, token_id: str) -> TradeOutcome:
        """
        Determine if token is YES or NO outcome

        In Polymarket, tokenIds are derived from conditionId and outcome index
        This is a heuristic - in production you'd query the market registry
        """
        # This is simplified - in reality you'd need to query the market
        # to determine which tokenId corresponds to YES vs NO
        return TradeOutcome.UNKNOWN

    def _get_market_id_from_token(self, token_id: str) -> str:
        """
        Get market ID from token ID

        In reality, this requires querying the conditional tokens contract
        or maintaining a mapping of token IDs to markets
        """
        return token_id  # Placeholder - would need market registry lookup


class MarketRegistry:
    """
    Maps token IDs to market information
    Queries Polymarket API for market details
    """

    def __init__(self, gamma_api_base: str = "https://gamma-api.polymarket.com"):
        self.gamma_api_base = gamma_api_base
        self.token_cache: Dict[str, Dict] = {}
        self.market_cache: Dict[str, Dict] = {}

    def get_market_for_token(self, token_id: str) -> Optional[Dict]:
        """Get market information for a token ID"""
        if token_id in self.token_cache:
            return self.token_cache[token_id]

        # Would query Polymarket API here
        # GET /markets?token_id={token_id}
        return None

    def get_token_info(self, token_id: str) -> Optional[Dict]:
        """Get detailed token information"""
        return self.token_cache.get(token_id)

    def register_token(self, token_id: str, market_info: Dict):
        """Cache token/market mapping"""
        self.token_cache[token_id] = market_info
