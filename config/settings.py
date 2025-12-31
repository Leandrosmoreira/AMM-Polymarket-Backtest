"""
Global settings for AMM Strategy Backtest
Polymarket BTC/SOL 15-min Markets
"""

# === MERCADO ===
ASSET = "BTC"  # BTC or SOL
MARKET_TYPE = "up_or_down"
TIMEFRAME_MINUTES = 15
MARKET_DURATION_MS = 15 * 60 * 1000  # 900000 ms

# === PERIODO DO BACKTEST ===
BACKTEST_START = "2024-09-13"
BACKTEST_END = "2024-12-13"
BACKTEST_DAYS = 90

# === CAPITAL ===
INITIAL_CAPITAL = 100  # USD - per market (resets each cycle)
CURRENCY = "USDC"

# === BTC BOT - FREQUENCIAS ===
INTERVAL_ZSCORE_MS = 1000       # 1 segundo - cálculo Z-Score
INTERVAL_DESVIO_MS = 5000       # 5 segundos - cálculo σ
INTERVAL_TRADE_MS = 10000       # 10 segundos - decisão de trade

# === BTC BOT - OPORTUNIDADE ===
MIN_OPORTUNIDADE = 0.05         # 5% mínimo para trade
OPORTUNIDADE_PEQUENA = 0.05     # 5-10% -> trade pequeno
OPORTUNIDADE_MEDIA = 0.10       # 10-20% -> trade médio
OPORTUNIDADE_GRANDE = 0.20      # >20% -> trade grande

# === BTC BOT - TAMANHO DE TRADE ===
TRADE_PEQUENO_PCT = 0.05        # 5% da banca
TRADE_MEDIO_PCT = 0.10          # 10% da banca
TRADE_GRANDE_PCT = 0.20         # 20% da banca

# === BTC BOT - GESTAO DE RISCO ===
MAX_INVESTIMENTO = 100          # $100 máximo por mercado
MAX_DESBALANCEAMENTO = 0.20     # 20% limite de exposição unilateral
MIN_DESVIO_PADRAO = 20.0        # σ mínimo em USD

# === API ===
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# === RATE LIMITING ===
MAX_REQUESTS_PER_SECOND = 10
REQUEST_TIMEOUT = 30  # seconds

# === DATA PATHS ===
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"
DATA_TRADES_PATH = "data/trades"
DATA_RESULTS_PATH = "data/results"

# === LOGGING ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
