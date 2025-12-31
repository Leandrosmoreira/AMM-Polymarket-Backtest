"""
Risk Parameters for BTC 15min Up/Down Strategy
Probabilistic Arbitrage Bot
"""


class BTCRiskParams:
    """Parametros de risco para Bot BTC 15min"""

    # === CAPITAL POR MERCADO ===
    MAX_INVESTIMENTO = 100          # Máximo $100 por mercado (ciclo 15min)

    # === FREQUENCIAS (em ms) ===
    INTERVALO_ZSCORE = 1000         # 1 segundo - cálculo Z-Score
    INTERVALO_DESVIO = 5000         # 5 segundos - cálculo σ
    INTERVALO_TRADE = 10000         # 10 segundos - decisão de trade

    # === OPORTUNIDADE ===
    MIN_OPORTUNIDADE = 0.05         # 5% mínimo para considerar trade
    OPORTUNIDADE_PEQUENA = 0.05     # 5-10% -> trade pequeno
    OPORTUNIDADE_MEDIA = 0.10       # 10-20% -> trade médio
    OPORTUNIDADE_GRANDE = 0.20      # >20% -> trade grande

    # === TAMANHO DE TRADE (% da banca restante) ===
    TRADE_PEQUENO_PCT = 0.05        # 5% da banca
    TRADE_MEDIO_PCT = 0.10          # 10% da banca
    TRADE_GRANDE_PCT = 0.20         # 20% da banca

    # === DESBALANCEAMENTO ===
    MAX_DESBALANCEAMENTO = 0.20     # 20% máximo de exposição unilateral
    # Se Shares_UP > Shares_DOWN * 1.2 → só compra DOWN
    # Se Shares_DOWN > Shares_UP * 1.2 → só compra UP

    # === DESVIO PADRAO ===
    MIN_DESVIO_PADRAO = 20.0        # σ mínimo em USD (evita Z-scores extremos)

    # === TEMPO ===
    MIN_TEMPO_RESTANTE = 60         # Mínimo 1 minuto antes do fim
    TEMPO_ESTABILIZACAO = 120       # 2 minutos para σ estabilizar

    # === SLIPPAGE ===
    MARGEM_SLIPPAGE = 0.02          # 2% margem de segurança

    # === CICLO DO MERCADO ===
    DURACAO_MERCADO_MS = 900000     # 15 minutos em ms

    @classmethod
    def get_trade_size(cls, oportunidade: float) -> str:
        """
        Determina tamanho do trade baseado na oportunidade.

        Args:
            oportunidade: Diferença entre probabilidade e preço do token

        Returns:
            String indicando tamanho: 'NONE', 'PEQUENO', 'MEDIO', 'GRANDE'
        """
        if oportunidade < cls.MIN_OPORTUNIDADE:
            return 'NONE'
        elif oportunidade < cls.OPORTUNIDADE_MEDIA:
            return 'PEQUENO'
        elif oportunidade < cls.OPORTUNIDADE_GRANDE:
            return 'MEDIO'
        else:
            return 'GRANDE'

    @classmethod
    def get_trade_pct(cls, tamanho: str) -> float:
        """
        Retorna percentual da banca para cada tamanho de trade.

        Args:
            tamanho: 'PEQUENO', 'MEDIO' ou 'GRANDE'

        Returns:
            Percentual da banca a investir
        """
        sizes = {
            'NONE': 0.0,
            'PEQUENO': cls.TRADE_PEQUENO_PCT,
            'MEDIO': cls.TRADE_MEDIO_PCT,
            'GRANDE': cls.TRADE_GRANDE_PCT,
        }
        return sizes.get(tamanho, 0.0)

    @classmethod
    def to_dict(cls):
        """Retorna todos os parametros como dicionario."""
        return {
            key: value for key, value in vars(cls).items()
            if not key.startswith('_') and not callable(getattr(cls, key, None))
        }
