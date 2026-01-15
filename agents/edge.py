"""
EdgeAgent - Detecção de oportunidades de trading
Identifica edges via pair cost, z-score, e divergências
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

from core.types import AgentDecision, AgentType
from core.buffers import MarketDataState, PriceBuffer
from config.gabagool_config import GabagoolConfig


@dataclass
class EdgeMetrics:
    """Métricas de edge calculadas."""
    pair_cost: float = 1.0  # YES + NO price
    pair_cost_edge: float = 0.0  # 1.0 - pair_cost
    zscore_yes: float = 0.0
    zscore_no: float = 0.0
    binance_divergence: float = 0.0
    total_edge_bps: float = 0.0
    edge_type: str = "none"
    has_edge: bool = False
    edge_strength: float = 0.0  # 0-1
    reason: str = ""


class EdgeAgent:
    """
    Agente de detecção de edge.

    Responsabilidades:
    1. Calcular pair cost (YES + NO)
    2. Detectar edges via z-score de preços
    3. Identificar divergências com Binance
    4. Combinar sinais para edge total
    """

    __slots__ = (
        'config',
        '_last_metrics',
        '_yes_prices',
        '_no_prices',
        '_pair_cost_history'
    )

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._last_metrics: Optional[EdgeMetrics] = None
        self._yes_prices = PriceBuffer(size=100)
        self._no_prices = PriceBuffer(size=100)
        self._pair_cost_history = PriceBuffer(size=100)

    def analyze(self, market_data: MarketDataState) -> AgentDecision:
        """
        Analisa oportunidades de edge.

        Args:
            market_data: Estado atual dos dados de mercado

        Returns:
            AgentDecision com should_trade, edge_strength e dados
        """
        metrics = self._calculate_metrics(market_data)
        self._last_metrics = metrics

        should_trade = metrics.has_edge
        confidence = metrics.edge_strength

        return AgentDecision(
            agent=AgentType.EDGE,
            should_trade=should_trade,
            confidence=confidence,
            reason=metrics.reason,
            data={
                "pair_cost": metrics.pair_cost,
                "pair_cost_edge": metrics.pair_cost_edge,
                "zscore_yes": metrics.zscore_yes,
                "zscore_no": metrics.zscore_no,
                "binance_divergence": metrics.binance_divergence,
                "total_edge_bps": metrics.total_edge_bps,
                "edge_type": metrics.edge_type,
                "edge_strength": metrics.edge_strength,
            }
        )

    def _calculate_metrics(self, market_data: MarketDataState) -> EdgeMetrics:
        """Calcula todas as métricas de edge."""
        metrics = EdgeMetrics()

        # Get current prices
        yes_mid = market_data.yes_mid
        no_mid = market_data.no_mid

        if yes_mid is None or no_mid is None:
            metrics.reason = "no price data"
            return metrics

        # Update price buffers
        self._yes_prices.add(yes_mid)
        self._no_prices.add(no_mid)

        # 1. Calculate pair cost
        metrics.pair_cost = yes_mid + no_mid
        metrics.pair_cost_edge = 1.0 - metrics.pair_cost
        self._pair_cost_history.add(metrics.pair_cost)

        # 2. Calculate z-scores
        if self._yes_prices.count >= 10:
            metrics.zscore_yes = self._yes_prices.get_zscore(20)
            metrics.zscore_no = self._no_prices.get_zscore(20)

        # 3. Binance divergence
        metrics.binance_divergence = self._calculate_binance_divergence(
            market_data, yes_mid
        )

        # 4. Calculate total edge
        edge_bps = 0.0
        edge_types = []

        # Pair cost edge
        if self.config.enable_pair_cost_edge:
            if metrics.pair_cost <= self.config.pair_cost_target:
                pc_edge = (1.0 - metrics.pair_cost) * 10000  # Convert to bps
                edge_bps += pc_edge
                edge_types.append("pair_cost")

        # Z-score edge
        if self.config.enable_zscore_edge:
            max_zscore = max(abs(metrics.zscore_yes), abs(metrics.zscore_no))
            if max_zscore >= self.config.zscore_prepare:
                zs_edge = max_zscore * 20  # ~20 bps per z-score
                edge_bps += zs_edge
                edge_types.append("zscore")

        # Binance divergence edge
        if abs(metrics.binance_divergence) > 0.01:  # 1% divergence
            div_edge = abs(metrics.binance_divergence) * 5000  # bps
            edge_bps += div_edge
            edge_types.append("binance_div")

        metrics.total_edge_bps = edge_bps
        metrics.edge_type = "+".join(edge_types) if edge_types else "none"

        # 5. Determine if has edge
        metrics.has_edge = edge_bps >= self.config.min_edge_bps
        metrics.edge_strength = self._calculate_edge_strength(metrics)
        metrics.reason = self._get_edge_reason(metrics)

        return metrics

    def _calculate_binance_divergence(
        self,
        market_data: MarketDataState,
        yes_price: float
    ) -> float:
        """
        Calcula divergência entre Polymarket YES price e probabilidade implícita do Binance.
        """
        if market_data.binance_prices.count < 10:
            return 0.0

        # Para mercados tipo "BTC Up/Down 15min", a probabilidade é baseada
        # no movimento recente do preço Binance
        binance_return = self._get_binance_return(market_data.binance_prices)

        # Se Binance subiu, YES deveria valer mais (para mercado "Up")
        # Divergência = diferença entre preço atual e "fair value" implícito
        implied_prob = self._binance_return_to_prob(binance_return)

        return yes_price - implied_prob

    def _get_binance_return(self, prices: PriceBuffer) -> float:
        """Calcula retorno recente do Binance."""
        if prices.count < 2:
            return 0.0

        recent = prices.get_prices(10)
        if len(recent) < 2:
            return 0.0

        return (recent[-1] - recent[0]) / recent[0]

    def _binance_return_to_prob(self, ret: float) -> float:
        """
        Converte retorno do Binance em probabilidade implícita.
        Modelo simplificado: probabilidade de "Up" baseada no momentum.
        """
        # Função sigmoide centrada em 0.5
        # ret positivo -> prob > 0.5
        # ret negativo -> prob < 0.5
        sensitivity = 100  # Quanto o retorno afeta a probabilidade

        prob = 1 / (1 + np.exp(-ret * sensitivity))
        return np.clip(prob, 0.3, 0.7)  # Limita entre 30% e 70%

    def _calculate_edge_strength(self, metrics: EdgeMetrics) -> float:
        """Calcula força do edge (0-1)."""
        if metrics.total_edge_bps <= 0:
            return 0.0

        # Edge strength baseado em múltiplos fatores
        strength = 0.0

        # Pair cost contribution
        if metrics.pair_cost <= self.config.pair_cost_strong:
            strength += 0.4
        elif metrics.pair_cost <= self.config.pair_cost_target:
            strength += 0.3
        elif metrics.pair_cost <= self.config.pair_cost_max:
            strength += 0.1

        # Z-score contribution
        max_zs = max(abs(metrics.zscore_yes), abs(metrics.zscore_no))
        if max_zs >= self.config.zscore_strong:
            strength += 0.3
        elif max_zs >= self.config.zscore_prepare:
            strength += 0.2
        elif max_zs >= self.config.zscore_observe:
            strength += 0.1

        # Total edge bps contribution
        if metrics.total_edge_bps >= 200:
            strength += 0.3
        elif metrics.total_edge_bps >= 100:
            strength += 0.2
        elif metrics.total_edge_bps >= 50:
            strength += 0.1

        return min(1.0, strength)

    def _get_edge_reason(self, metrics: EdgeMetrics) -> str:
        """Gera razão descritiva para o edge."""
        if not metrics.has_edge:
            if metrics.pair_cost > self.config.pair_cost_max:
                return f"pair_cost too high ({metrics.pair_cost:.4f})"
            return f"insufficient edge ({metrics.total_edge_bps:.1f} bps < {self.config.min_edge_bps})"

        parts = []

        if metrics.pair_cost <= self.config.pair_cost_target:
            parts.append(f"pair_cost={metrics.pair_cost:.4f}")

        max_zs = max(abs(metrics.zscore_yes), abs(metrics.zscore_no))
        if max_zs >= self.config.zscore_observe:
            parts.append(f"zscore={max_zs:.2f}")

        if metrics.total_edge_bps > 0:
            parts.append(f"edge={metrics.total_edge_bps:.1f}bps")

        return f"edge found: {', '.join(parts)}"

    def get_recommended_side(self) -> Optional[str]:
        """
        Retorna lado recomendado baseado no edge.
        Para pair cost edge, compramos ambos os lados.
        Para z-score edge, compramos o lado com z-score negativo (subvalorizado).
        """
        if self._last_metrics is None:
            return None

        metrics = self._last_metrics

        if not metrics.has_edge:
            return None

        # Pair cost edge = comprar ambos
        if "pair_cost" in metrics.edge_type:
            return "BOTH"

        # Z-score edge = comprar o subvalorizado
        if "zscore" in metrics.edge_type:
            if metrics.zscore_yes < -self.config.zscore_observe:
                return "YES"
            elif metrics.zscore_no < -self.config.zscore_observe:
                return "NO"

        return "BOTH"

    @property
    def last_metrics(self) -> Optional[EdgeMetrics]:
        return self._last_metrics

    @property
    def current_pair_cost(self) -> Optional[float]:
        return self._last_metrics.pair_cost if self._last_metrics else None
