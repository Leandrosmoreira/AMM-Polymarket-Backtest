"""
Copy Trading Configuration for Polymarket
Monitor wallet transactions and replicate trades
"""

# === WALLET TO MONITOR ===
TARGET_WALLET = "0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d"

# === YOUR WALLET (for executing copy trades) ===
# IMPORTANT: Set your private key in environment variable POLY_PRIVATE_KEY
YOUR_WALLET = ""  # Will be derived from private key

# === POLYGON RPC ===
# Free public RPCs (may have rate limits)
POLYGON_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc-mainnet.matic.network",
    "https://matic-mainnet.chainstacklabs.com",
    "https://polygon-mainnet.public.blastapi.io",
    "https://polygon.llamarpc.com",
]

# For better performance, use a dedicated RPC like:
# - Alchemy: https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY
# - Infura: https://polygon-mainnet.infura.io/v3/YOUR_PROJECT_ID
# - QuickNode: Your custom endpoint
POLYGON_RPC_PRIMARY = "https://polygon-rpc.com"

# === POLYMARKET CONTRACTS (Polygon) ===
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
CONDITIONAL_TOKENS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

# === COPY TRADE SETTINGS ===
# Percentage of the original trade size to copy (0.10 = 10%)
COPY_PERCENTAGE = 0.10

# Dynamic sizing based on your balance
DYNAMIC_SIZING = True

# Maximum single trade size in USDC
MAX_TRADE_SIZE = 100.0

# Minimum trade size to copy (ignore dust trades)
MIN_TRADE_SIZE = 1.0

# === MONITORING SETTINGS ===
# How often to check for new transactions (seconds)
POLL_INTERVAL = 1.0

# Number of blocks to look back when starting
LOOKBACK_BLOCKS = 10

# How many blocks to wait for confirmation
CONFIRMATION_BLOCKS = 2

# === SLIPPAGE PROTECTION ===
# Maximum slippage tolerance (0.02 = 2%)
MAX_SLIPPAGE = 0.02

# Price validity window (seconds) - skip if price changed too much
PRICE_VALIDITY_SECONDS = 30

# === RISK MANAGEMENT ===
# Maximum total exposure as percentage of balance
MAX_EXPOSURE_PERCENT = 0.50

# Maximum number of positions
MAX_POSITIONS = 10

# Stop trading if balance drops below this
MIN_BALANCE_USDC = 10.0

# === LOGGING ===
LOG_FILE = "data/copytrade_logs/copytrade.log"
LOG_LEVEL = "INFO"

# === TRADE EXECUTION ===
# Gas settings
GAS_LIMIT = 500000
GAS_PRICE_MULTIPLIER = 1.1  # 10% above estimated gas price

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# === NOTIFICATIONS (optional) ===
ENABLE_NOTIFICATIONS = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
