"""
GabagoolConfig - Configuração completa para Paper Trading
Baseado na arquitetura do gabagool bot
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class GabagoolConfig:
    """
    Configuração imutável para o sistema de paper trading.

    Categorias:
    - Polymarket: Conexão com a API
    - Binance: Feed de preços BTC/SOL
    - Microstructure: Parâmetros de análise de order book
    - Edge Detection: Identificação de oportunidades
    - Position Sizing: Dimensionamento de ordens
    - Risk Management: Controle de risco
    - Market Making: Parâmetros de quote
    - Timing: Intervalos e timeouts
    """

    # =========================================================================
    # POLYMARKET
    # =========================================================================
    polymarket_api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""))
    polymarket_api_secret: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""))
    polymarket_private_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""))
    polymarket_api_url: str = "https://clob.polymarket.com"
    polymarket_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"

    # Token IDs (atualizados dinamicamente)
    polymarket_yes_token: str = ""
    polymarket_no_token: str = ""

    # =========================================================================
    # BINANCE
    # =========================================================================
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"
    binance_symbol: str = "btcusdt"  # ou solusdt para SOL markets

    # =========================================================================
    # MICROSTRUCTURE THRESHOLDS
    # =========================================================================
    # Cluster detection
    cluster_cv_threshold: float = 1.2  # Coeficiente de variação para detectar clusters
    cluster_burst_pct: float = 0.6  # % de trades em burst para considerar cluster
    cluster_window_seconds: float = 5.0  # Janela de tempo para análise

    # Dryness scoring (liquidez)
    dryness_threshold_dry: float = 70.0  # Score acima = mercado seco, não operar
    dryness_threshold_pre: float = 50.0  # Score acima = mercado pré-seco, cautela
    min_book_depth_usd: float = 50.0  # Profundidade mínima do book

    # =========================================================================
    # EDGE DETECTION
    # =========================================================================
    # Pair cost (YES + NO)
    pair_cost_target: float = 0.97  # Custo par ideal para entrada
    pair_cost_max: float = 0.99  # Custo máximo aceitável
    pair_cost_strong: float = 0.95  # Oportunidade forte

    # Z-score edges
    zscore_observe: float = 1.0  # Nível para observar
    zscore_prepare: float = 2.0  # Nível para preparar
    zscore_strong: float = 2.5  # Edge forte

    # Minimum edge para operar
    min_edge_bps: float = 50.0  # 0.5% mínimo de edge

    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    base_order_usd: float = 10.0  # Ordem base
    min_order_usd: float = 5.0  # Ordem mínima
    max_order_usd: float = 50.0  # Ordem máxima
    max_position_usd: float = 500.0  # Posição máxima total

    # Bankroll management
    bankroll_pct_per_trade: float = 0.05  # 5% do bankroll por trade
    initial_bankroll: float = 1000.0  # Capital inicial

    # =========================================================================
    # RISK MANAGEMENT
    # =========================================================================
    # Limites de perda
    daily_loss_limit_usd: float = 100.0  # Perda máxima diária
    per_trade_loss_limit_usd: float = 20.0  # Perda máxima por trade
    max_consecutive_losses: int = 5  # Máximo de perdas consecutivas

    # Exposure limits
    max_directional_exposure_usd: float = 200.0  # Exposição direcional máxima
    max_total_exposure_usd: float = 500.0  # Exposição total máxima

    # Kill switch
    kill_switch_enabled: bool = True
    kill_on_daily_limit: bool = True
    kill_on_consecutive_losses: bool = True

    # =========================================================================
    # MARKET MAKING
    # =========================================================================
    # Spread configuration
    min_spread_bps: float = 50.0  # Spread mínimo 0.5%
    target_spread_bps: float = 100.0  # Spread alvo 1%

    # Skew limits
    max_skew_ratio: float = 1.5  # Máximo YES/NO ratio
    rebalance_threshold: float = 0.2  # Rebalancear quando > 20% desbalanceado

    # Quote sizing
    quote_size_pct: float = 0.5  # % do tamanho máximo por quote

    # =========================================================================
    # TIMING
    # =========================================================================
    tick_interval_ms: int = 500  # Intervalo do loop principal
    order_timeout_seconds: float = 5.0  # Timeout para ordens
    ws_ping_interval_seconds: float = 30.0  # Ping do WebSocket
    ws_reconnect_delay_seconds: float = 5.0  # Delay para reconexão
    market_refresh_seconds: float = 60.0  # Atualizar info do mercado

    # =========================================================================
    # FEATURE FLAGS
    # =========================================================================
    enable_pair_cost_edge: bool = True  # Usar pair cost para edge
    enable_zscore_edge: bool = True  # Usar z-score para edge
    enable_microstructure: bool = True  # Usar análise de microestrutura
    dry_run: bool = True  # Paper trading (sempre True neste sistema)

    # =========================================================================
    # LOGGING
    # =========================================================================
    log_level: str = "INFO"
    log_trades: bool = True
    log_decisions: bool = True
    status_interval_seconds: float = 10.0  # Intervalo para log de status

    def validate(self) -> bool:
        """Validate configuration."""
        errors = []

        if self.pair_cost_max < self.pair_cost_target:
            errors.append("pair_cost_max must be >= pair_cost_target")

        if self.max_order_usd > self.max_position_usd:
            errors.append("max_order_usd must be <= max_position_usd")

        if self.bankroll_pct_per_trade > 0.2:
            errors.append("bankroll_pct_per_trade should not exceed 20%")

        if errors:
            for e in errors:
                print(f"Config error: {e}")
            return False

        return True

    def get_order_size(self, edge_strength: float = 1.0) -> float:
        """Calculate order size based on edge strength."""
        base = self.base_order_usd * edge_strength
        return max(self.min_order_usd, min(base, self.max_order_usd))

    def get_max_bankroll_order(self, current_bankroll: float) -> float:
        """Get max order size based on current bankroll."""
        return min(
            current_bankroll * self.bankroll_pct_per_trade,
            self.max_order_usd
        )


def load_config() -> GabagoolConfig:
    """Load configuration from environment or defaults."""
    config = GabagoolConfig()

    if not config.validate():
        raise ValueError("Invalid configuration")

    return config


# Singleton instance
_config: Optional[GabagoolConfig] = None


def get_config() -> GabagoolConfig:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
