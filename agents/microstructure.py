"""
MicrostructureAgent - Análise de microestrutura de mercado
Detecta clusters, calcula dryness score, analisa order book
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from core.types import AgentDecision, AgentType, BookSnapshot
from core.buffers import MarketDataState, TradeBuffer, BookBuffer
from config.gabagool_config import GabagoolConfig


@dataclass
class MicrostructureMetrics:
    """Métricas de microestrutura calculadas."""
    dryness_score: float = 0.0
    cluster_detected: bool = False
    cluster_intensity: float = 0.0
    book_imbalance: float = 0.0
    spread_bps: float = 0.0
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    trade_intensity: float = 0.0  # trades por minuto
    is_tradeable: bool = True
    reason: str = ""


class MicrostructureAgent:
    """
    Agente de análise de microestrutura.

    Responsabilidades:
    1. Detectar clusters de trades (manipulação/informação)
    2. Calcular dryness score (liquidez)
    3. Analisar order book imbalance
    4. Determinar se mercado é tradeable
    """

    __slots__ = ('config', '_last_metrics', '_trade_timestamps')

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._last_metrics: Optional[MicrostructureMetrics] = None
        self._trade_timestamps: List[datetime] = []

    def analyze(self, market_data: MarketDataState) -> AgentDecision:
        """
        Analisa microestrutura e decide se deve operar.

        Args:
            market_data: Estado atual dos dados de mercado

        Returns:
            AgentDecision com should_trade e métricas
        """
        metrics = self._calculate_metrics(market_data)
        self._last_metrics = metrics

        should_trade = metrics.is_tradeable
        confidence = self._calculate_confidence(metrics)

        return AgentDecision(
            agent=AgentType.MICROSTRUCTURE,
            should_trade=should_trade,
            confidence=confidence,
            reason=metrics.reason,
            data={
                "dryness_score": metrics.dryness_score,
                "cluster_detected": metrics.cluster_detected,
                "cluster_intensity": metrics.cluster_intensity,
                "book_imbalance": metrics.book_imbalance,
                "spread_bps": metrics.spread_bps,
                "bid_depth_usd": metrics.bid_depth_usd,
                "ask_depth_usd": metrics.ask_depth_usd,
                "trade_intensity": metrics.trade_intensity,
            }
        )

    def _calculate_metrics(self, market_data: MarketDataState) -> MicrostructureMetrics:
        """Calcula todas as métricas de microestrutura."""
        metrics = MicrostructureMetrics()

        # 1. Analisar order books
        yes_book = market_data.yes_book.latest
        no_book = market_data.no_book.latest

        if yes_book:
            metrics.bid_depth_usd += yes_book.bid_depth
            metrics.ask_depth_usd += yes_book.ask_depth

        if no_book:
            metrics.bid_depth_usd += no_book.bid_depth
            metrics.ask_depth_usd += no_book.ask_depth

        # 2. Calcular dryness score
        metrics.dryness_score = self._calculate_dryness(
            market_data.yes_book,
            market_data.no_book,
            market_data.yes_trades,
            market_data.no_trades
        )

        # 3. Detectar clusters
        cluster_result = self._detect_clusters(
            market_data.yes_trades,
            market_data.no_trades
        )
        metrics.cluster_detected = cluster_result['detected']
        metrics.cluster_intensity = cluster_result['intensity']

        # 4. Book imbalance
        if yes_book and no_book:
            yes_imb = yes_book.imbalance
            no_imb = no_book.imbalance
            metrics.book_imbalance = (yes_imb + no_imb) / 2

        # 5. Spread em bps
        if yes_book and yes_book.spread:
            yes_spread_bps = (yes_book.spread / yes_book.mid_price) * 10000 if yes_book.mid_price else 0
        else:
            yes_spread_bps = 0

        if no_book and no_book.spread:
            no_spread_bps = (no_book.spread / no_book.mid_price) * 10000 if no_book.mid_price else 0
        else:
            no_spread_bps = 0

        metrics.spread_bps = (yes_spread_bps + no_spread_bps) / 2

        # 6. Trade intensity
        yes_count = market_data.yes_trades.get_trade_count(60.0)
        no_count = market_data.no_trades.get_trade_count(60.0)
        metrics.trade_intensity = yes_count + no_count

        # 7. Determinar se é tradeable
        metrics.is_tradeable, metrics.reason = self._check_tradeable(metrics)

        return metrics

    def _calculate_dryness(
        self,
        yes_book: BookBuffer,
        no_book: BookBuffer,
        yes_trades: TradeBuffer,
        no_trades: TradeBuffer
    ) -> float:
        """
        Calcula dryness score (0-100).
        Score alto = mercado seco, sem liquidez.
        Score baixo = mercado líquido.
        """
        score = 0.0
        factors = 0

        # Factor 1: Book depth
        yes_latest = yes_book.latest
        no_latest = no_book.latest

        total_depth = 0.0
        if yes_latest:
            total_depth += yes_latest.bid_depth + yes_latest.ask_depth
        if no_latest:
            total_depth += no_latest.bid_depth + no_latest.ask_depth

        if total_depth < self.config.min_book_depth_usd:
            score += 40  # Penalidade severa por falta de depth
        elif total_depth < self.config.min_book_depth_usd * 2:
            score += 20
        elif total_depth < self.config.min_book_depth_usd * 5:
            score += 10
        factors += 1

        # Factor 2: Recent trade volume
        yes_volume = yes_trades.get_volume(300.0)  # 5 min
        no_volume = no_trades.get_volume(300.0)
        total_volume = yes_volume + no_volume

        if total_volume < 10:
            score += 30
        elif total_volume < 50:
            score += 15
        elif total_volume < 100:
            score += 5
        factors += 1

        # Factor 3: Spread width
        avg_spread = (yes_book.get_avg_spread(10) + no_book.get_avg_spread(10)) / 2
        if avg_spread > 0.05:  # 5% spread
            score += 20
        elif avg_spread > 0.02:
            score += 10
        elif avg_spread > 0.01:
            score += 5
        factors += 1

        # Factor 4: Trade frequency
        yes_count = yes_trades.get_trade_count(300.0)
        no_count = no_trades.get_trade_count(300.0)
        total_count = yes_count + no_count

        if total_count < 5:
            score += 10
        factors += 1

        return min(100, score)

    def _detect_clusters(
        self,
        yes_trades: TradeBuffer,
        no_trades: TradeBuffer
    ) -> Dict[str, Any]:
        """
        Detecta clusters de trades (possível manipulação ou informed trading).

        Cluster = muitos trades em curto período com tamanhos similares.
        """
        result = {
            'detected': False,
            'intensity': 0.0,
            'side': None
        }

        window = self.config.cluster_window_seconds

        # Analisar trades recentes
        yes_recent = yes_trades.get_recent(20)
        no_recent = no_trades.get_recent(20)

        for trades, side in [(yes_recent, 'YES'), (no_recent, 'NO')]:
            if len(trades) < 3:
                continue

            # Filtrar por janela de tempo
            now = datetime.now()
            in_window = [
                t for t in trades
                if (now - t.timestamp).total_seconds() <= window
            ]

            if len(in_window) < 3:
                continue

            # Calcular coeficiente de variação dos tamanhos
            sizes = [t.size for t in in_window]
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)

            if mean_size > 0:
                cv = std_size / mean_size
            else:
                cv = 0

            # Cluster = baixo CV (tamanhos similares) + muitos trades
            if cv < self.config.cluster_cv_threshold:
                burst_ratio = len(in_window) / max(1, len(trades))

                if burst_ratio >= self.config.cluster_burst_pct:
                    result['detected'] = True
                    result['intensity'] = max(result['intensity'], burst_ratio)
                    result['side'] = side

        return result

    def _check_tradeable(self, metrics: MicrostructureMetrics) -> tuple:
        """
        Verifica se mercado é tradeable baseado nas métricas.

        Returns:
            (is_tradeable, reason)
        """
        if not self.config.enable_microstructure:
            return True, "microstructure disabled"

        # Check 1: Dryness
        if metrics.dryness_score >= self.config.dryness_threshold_dry:
            return False, f"market too dry (score={metrics.dryness_score:.1f})"

        # Check 2: Cluster detected
        if metrics.cluster_detected and metrics.cluster_intensity > 0.8:
            return False, f"cluster detected (intensity={metrics.cluster_intensity:.2f})"

        # Check 3: Minimum depth
        total_depth = metrics.bid_depth_usd + metrics.ask_depth_usd
        if total_depth < self.config.min_book_depth_usd:
            return False, f"insufficient depth (${total_depth:.2f})"

        # Check 4: Extreme imbalance
        if abs(metrics.book_imbalance) > 0.8:
            return False, f"extreme book imbalance ({metrics.book_imbalance:.2f})"

        # Pre-dry warning
        if metrics.dryness_score >= self.config.dryness_threshold_pre:
            return True, f"caution: pre-dry market (score={metrics.dryness_score:.1f})"

        return True, "market conditions ok"

    def _calculate_confidence(self, metrics: MicrostructureMetrics) -> float:
        """Calcula confidence score baseado nas métricas."""
        if not metrics.is_tradeable:
            return 0.0

        confidence = 1.0

        # Reduz por dryness
        if metrics.dryness_score > 0:
            confidence *= (100 - metrics.dryness_score) / 100

        # Reduz por cluster intensity
        if metrics.cluster_detected:
            confidence *= (1 - metrics.cluster_intensity * 0.5)

        # Reduz por imbalance extremo
        confidence *= (1 - abs(metrics.book_imbalance) * 0.3)

        return max(0, min(1, confidence))

    @property
    def last_metrics(self) -> Optional[MicrostructureMetrics]:
        return self._last_metrics
