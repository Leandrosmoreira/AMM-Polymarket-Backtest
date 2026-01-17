"""
Volatility Engine - Calcula volatilidade em tempo real e ajusta parâmetros.

Usado tanto pelo Bot 1 (arbitrage) quanto Bot 2 (market maker).
"""
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class VolatilityRecommendation:
    """Recomendações baseadas em volatilidade."""
    volatility_score: float      # 0-100 (0=muito calmo, 100=muito volátil)
    spread_multiplier: float     # 1.0 = normal, 2.0 = dobrar spread
    size_multiplier: float       # 1.0 = normal, 0.5 = metade do tamanho
    should_quote: bool           # False se volatilidade muito extrema
    regime: str                  # "low", "normal", "high", "extreme"


class VolatilityEngine:
    """
    Calcula volatilidade em tempo real usando múltiplas métricas.

    Métricas:
    - Rolling standard deviation
    - Price range (high - low)
    - Spread volatility
    - Tick frequency

    Usage:
        vol = VolatilityEngine(lookback=100)
        vol.update(price=0.50, spread=0.02)
        rec = vol.get_recommendations()
        print(f"Spread multiplier: {rec.spread_multiplier}")
    """

    def __init__(
        self,
        lookback: int = 100,
        low_vol_threshold: float = 0.005,    # 0.5% = baixa vol
        high_vol_threshold: float = 0.02,    # 2% = alta vol
        extreme_vol_threshold: float = 0.05, # 5% = extrema
    ):
        self.lookback = lookback
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.extreme_vol_threshold = extreme_vol_threshold

        # Rolling windows
        self.prices: deque = deque(maxlen=lookback)
        self.spreads: deque = deque(maxlen=lookback)
        self.timestamps: deque = deque(maxlen=lookback)

        # Stats
        self.updates = 0
        self.last_update = 0.0

    def update(self, price: float, spread: Optional[float] = None, timestamp: Optional[float] = None):
        """
        Atualiza com novo tick de preço.

        Args:
            price: Preço atual (mid ou last)
            spread: Spread atual (ask - bid)
            timestamp: Timestamp (usa time.time() se não fornecido)
        """
        ts = timestamp or time.time()

        self.prices.append(price)
        if spread is not None:
            self.spreads.append(spread)
        self.timestamps.append(ts)

        self.updates += 1
        self.last_update = ts

    def get_price_volatility(self) -> float:
        """Calcula volatilidade de preço (std / mean)."""
        if len(self.prices) < 2:
            return 0.0

        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        if mean == 0:
            return 0.0

        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = math.sqrt(variance)

        return std / mean  # Coeficiente de variação

    def get_spread_volatility(self) -> float:
        """Calcula volatilidade do spread."""
        if len(self.spreads) < 2:
            return 0.0

        spreads = list(self.spreads)
        mean = sum(spreads) / len(spreads)
        if mean == 0:
            return 0.0

        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        std = math.sqrt(variance)

        return std / mean

    def get_tick_frequency(self) -> float:
        """Calcula frequência de ticks (ticks/segundo)."""
        if len(self.timestamps) < 2:
            return 0.0

        timestamps = list(self.timestamps)
        duration = timestamps[-1] - timestamps[0]
        if duration == 0:
            return 0.0

        return len(timestamps) / duration

    def get_price_range(self) -> float:
        """Calcula range de preço (max - min) / mean."""
        if len(self.prices) < 2:
            return 0.0

        prices = list(self.prices)
        mean = sum(prices) / len(prices)
        if mean == 0:
            return 0.0

        return (max(prices) - min(prices)) / mean

    def get_volatility_score(self) -> float:
        """
        Calcula score de volatilidade combinado (0-100).

        Combina múltiplas métricas com pesos.
        """
        price_vol = self.get_price_volatility()
        spread_vol = self.get_spread_volatility()
        price_range = self.get_price_range()

        # Normalizar para 0-100
        # Assumindo que 5% de volatilidade = score 100
        normalized_price = min(100, (price_vol / 0.05) * 100)
        normalized_range = min(100, (price_range / 0.10) * 100)
        normalized_spread = min(100, (spread_vol / 0.50) * 100)

        # Pesos
        score = (
            normalized_price * 0.5 +
            normalized_range * 0.3 +
            normalized_spread * 0.2
        )

        return min(100, max(0, score))

    def get_regime(self) -> str:
        """Determina regime de volatilidade."""
        vol = self.get_price_volatility()

        if vol < self.low_vol_threshold:
            return "low"
        elif vol < self.high_vol_threshold:
            return "normal"
        elif vol < self.extreme_vol_threshold:
            return "high"
        else:
            return "extreme"

    def get_recommendations(self) -> VolatilityRecommendation:
        """
        Retorna recomendações de trading baseadas em volatilidade.

        Returns:
            VolatilityRecommendation com ajustes sugeridos
        """
        vol_score = self.get_volatility_score()
        regime = self.get_regime()
        price_vol = self.get_price_volatility()

        # Ajustes baseados em regime
        if regime == "low":
            # Mercado calmo: spreads apertados, tamanho normal
            spread_mult = 0.8
            size_mult = 1.2
            should_quote = True

        elif regime == "normal":
            # Mercado normal: parâmetros default
            spread_mult = 1.0
            size_mult = 1.0
            should_quote = True

        elif regime == "high":
            # Alta volatilidade: spreads largos, tamanho reduzido
            spread_mult = 1.5 + (price_vol / self.high_vol_threshold) * 0.5
            size_mult = 0.7
            should_quote = True

        else:  # extreme
            # Volatilidade extrema: parar de quotar ou spreads muito largos
            spread_mult = 3.0
            size_mult = 0.3
            should_quote = False  # Muito arriscado

        return VolatilityRecommendation(
            volatility_score=vol_score,
            spread_multiplier=spread_mult,
            size_multiplier=size_mult,
            should_quote=should_quote,
            regime=regime,
        )

    def get_stats(self) -> dict:
        """Retorna estatísticas do engine."""
        return {
            "updates": self.updates,
            "samples": len(self.prices),
            "price_volatility": self.get_price_volatility(),
            "spread_volatility": self.get_spread_volatility(),
            "tick_frequency": self.get_tick_frequency(),
            "price_range": self.get_price_range(),
            "volatility_score": self.get_volatility_score(),
            "regime": self.get_regime(),
        }


# Standalone test
if __name__ == "__main__":
    import random

    vol = VolatilityEngine(lookback=50)

    # Simulate calm market
    print("=== Mercado Calmo ===")
    for i in range(60):
        price = 0.50 + random.uniform(-0.002, 0.002)
        spread = 0.01 + random.uniform(-0.001, 0.001)
        vol.update(price, spread)

    rec = vol.get_recommendations()
    print(f"Score: {rec.volatility_score:.1f}")
    print(f"Regime: {rec.regime}")
    print(f"Spread mult: {rec.spread_multiplier:.2f}")
    print(f"Size mult: {rec.size_multiplier:.2f}")
    print()

    # Simulate volatile market
    print("=== Mercado Volátil ===")
    vol2 = VolatilityEngine(lookback=50)
    for i in range(60):
        price = 0.50 + random.uniform(-0.05, 0.05)
        spread = 0.02 + random.uniform(-0.01, 0.02)
        vol2.update(price, spread)

    rec2 = vol2.get_recommendations()
    print(f"Score: {rec2.volatility_score:.1f}")
    print(f"Regime: {rec2.regime}")
    print(f"Spread mult: {rec2.spread_multiplier:.2f}")
    print(f"Size mult: {rec2.size_multiplier:.2f}")
    print(f"Should quote: {rec2.should_quote}")
