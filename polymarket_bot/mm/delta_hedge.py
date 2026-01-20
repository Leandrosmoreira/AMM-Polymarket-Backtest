"""
Delta Hedger - Mantém posição delta-neutral para market makers.

O objetivo é não ficar exposto a movimento direcional do mercado.
Se acumular muito YES, ajusta quotes para vender YES.
"""
import time
from dataclasses import dataclass
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuoteAdjustment:
    """Ajuste de preços para rebalancear delta."""
    bid_adjustment: float  # Ajuste no bid (negativo = bid mais baixo)
    ask_adjustment: float  # Ajuste no ask (positivo = ask mais alto)
    urgency: str           # "low", "medium", "high", "critical"
    reason: str            # Explicação do ajuste


@dataclass
class Position:
    """Posição em um token."""
    size: float           # Quantidade (positivo = long, negativo = short)
    avg_price: float      # Preço médio de entrada
    last_update: float    # Timestamp da última atualização


class DeltaHedger:
    """
    Gerencia delta (exposição direcional) do portfolio.

    Para um mercado binário (YES/NO):
    - Delta = YES_position - NO_position
    - Delta > 0: exposto a YES ganhar
    - Delta < 0: exposto a NO ganhar
    - Delta = 0: neutro (ideal)

    Estratégia:
    - Se delta > threshold: ajusta quotes para vender YES
    - Se delta < -threshold: ajusta quotes para vender NO
    - Ajuste proporcional ao tamanho do desbalanceamento

    Usage:
        hedger = DeltaHedger(max_delta=50)
        hedger.update_position("YES", 10, 0.50)  # Comprou 10 YES
        adj = hedger.get_quote_adjustment("YES")
        print(f"Ajuste bid: {adj.bid_adjustment}")
    """

    def __init__(
        self,
        max_delta: float = 50.0,           # Máximo desbalanceamento permitido
        warning_delta: float = 30.0,       # Alerta quando atingir
        adjustment_factor: float = 0.001,  # Ajuste por unidade de delta
        max_adjustment: float = 0.02,      # Ajuste máximo (2 cents)
    ):
        self.max_delta = max_delta
        self.warning_delta = warning_delta
        self.adjustment_factor = adjustment_factor
        self.max_adjustment = max_adjustment

        # Posições por mercado: {market_id: {token: Position}}
        self.positions: Dict[str, Dict[str, Position]] = {}

        # Estatísticas
        self.total_hedges = 0
        self.last_hedge_time = 0.0

    def update_position(
        self,
        market_id: str,
        token: str,  # "YES" ou "NO"
        size_delta: float,
        price: float,
    ):
        """
        Atualiza posição após um fill.

        Args:
            market_id: ID do mercado
            token: "YES" ou "NO"
            size_delta: Mudança no tamanho (+ = compra, - = venda)
            price: Preço de execução
        """
        if market_id not in self.positions:
            self.positions[market_id] = {}

        if token not in self.positions[market_id]:
            self.positions[market_id][token] = Position(
                size=0.0,
                avg_price=0.0,
                last_update=time.time(),
            )

        pos = self.positions[market_id][token]

        # Atualizar preço médio
        if size_delta > 0:  # Comprando
            total_cost = (pos.size * pos.avg_price) + (size_delta * price)
            new_size = pos.size + size_delta
            if new_size > 0:
                pos.avg_price = total_cost / new_size
        else:  # Vendendo
            # Preço médio não muda ao vender
            pass

        pos.size += size_delta
        pos.last_update = time.time()

        logger.debug(f"Position updated: {market_id} {token} = {pos.size:.2f} @ ${pos.avg_price:.4f}")

    def get_delta(self, market_id: str) -> float:
        """
        Calcula delta (exposição direcional) para um mercado.

        Returns:
            Delta: YES_size - NO_size
        """
        if market_id not in self.positions:
            return 0.0

        yes_size = self.positions[market_id].get("YES", Position(0, 0, 0)).size
        no_size = self.positions[market_id].get("NO", Position(0, 0, 0)).size

        return yes_size - no_size

    def get_total_delta(self) -> float:
        """Calcula delta total de todos os mercados."""
        return sum(self.get_delta(m) for m in self.positions)

    def get_quote_adjustment(self, market_id: str, token: str = "YES") -> QuoteAdjustment:
        """
        Calcula ajuste de quotes para rebalancear delta.

        Se temos muito YES:
        - Reduzir bid de YES (comprar menos)
        - Aumentar ask de YES (vender por mais)
        - Ou: aumentar bid de NO

        Args:
            market_id: ID do mercado
            token: Token para ajustar quotes

        Returns:
            QuoteAdjustment com ajustes recomendados
        """
        delta = self.get_delta(market_id)
        abs_delta = abs(delta)

        # Determinar urgência
        if abs_delta < self.warning_delta * 0.5:
            urgency = "low"
        elif abs_delta < self.warning_delta:
            urgency = "medium"
        elif abs_delta < self.max_delta:
            urgency = "high"
        else:
            urgency = "critical"

        # Calcular ajuste proporcional ao delta
        raw_adjustment = delta * self.adjustment_factor

        # Limitar ajuste máximo
        adjustment = max(-self.max_adjustment, min(self.max_adjustment, raw_adjustment))

        if token == "YES":
            if delta > 0:
                # Muito YES: desencorajar compra, encorajar venda
                bid_adj = -abs(adjustment)  # Bid mais baixo
                ask_adj = -abs(adjustment) * 0.5  # Ask um pouco mais baixo para atrair venda
                reason = f"Long YES ({delta:.1f}): reduzindo bid para parar de comprar"
            else:
                # Pouco YES: encorajar compra
                bid_adj = abs(adjustment) * 0.5
                ask_adj = abs(adjustment)
                reason = f"Short YES ({delta:.1f}): aumentando bid para comprar mais"
        else:  # NO
            if delta > 0:
                # Muito YES = pouco NO: encorajar compra de NO
                bid_adj = abs(adjustment) * 0.5
                ask_adj = abs(adjustment)
                reason = f"Need NO ({delta:.1f}): aumentando bid de NO"
            else:
                bid_adj = -abs(adjustment)
                ask_adj = -abs(adjustment) * 0.5
                reason = f"Too much NO ({delta:.1f}): reduzindo bid de NO"

        return QuoteAdjustment(
            bid_adjustment=bid_adj,
            ask_adjustment=ask_adj,
            urgency=urgency,
            reason=reason,
        )

    def needs_urgent_hedge(self, market_id: str) -> bool:
        """Verifica se precisa hedge urgente (delta além do máximo)."""
        return abs(self.get_delta(market_id)) > self.max_delta

    def get_hedge_order(self, market_id: str) -> Optional[dict]:
        """
        Se delta muito grande, retorna ordem para hedge imediato.

        Returns:
            Dict com ordem de hedge ou None se não precisa
        """
        delta = self.get_delta(market_id)

        if abs(delta) <= self.max_delta:
            return None

        # Precisa hedge
        self.total_hedges += 1
        self.last_hedge_time = time.time()

        if delta > 0:
            # Muito YES: vender YES no mercado (taker)
            return {
                "action": "SELL",
                "token": "YES",
                "size": abs(delta) - self.max_delta * 0.5,  # Reduzir para metade do limite
                "reason": "Urgent delta hedge: too long YES",
            }
        else:
            # Muito NO: vender NO no mercado
            return {
                "action": "SELL",
                "token": "NO",
                "size": abs(delta) - self.max_delta * 0.5,
                "reason": "Urgent delta hedge: too long NO",
            }

    def get_position_summary(self, market_id: str) -> dict:
        """Retorna resumo da posição em um mercado."""
        if market_id not in self.positions:
            return {"yes": 0, "no": 0, "delta": 0, "urgency": "none"}

        yes = self.positions[market_id].get("YES", Position(0, 0, 0))
        no = self.positions[market_id].get("NO", Position(0, 0, 0))
        delta = self.get_delta(market_id)

        adj = self.get_quote_adjustment(market_id)

        return {
            "yes_size": yes.size,
            "yes_avg_price": yes.avg_price,
            "no_size": no.size,
            "no_avg_price": no.avg_price,
            "delta": delta,
            "urgency": adj.urgency,
        }

    def get_stats(self) -> dict:
        """Retorna estatísticas do hedger."""
        return {
            "total_markets": len(self.positions),
            "total_delta": self.get_total_delta(),
            "total_hedges": self.total_hedges,
            "markets": {m: self.get_position_summary(m) for m in self.positions},
        }


# Standalone test
if __name__ == "__main__":
    hedger = DeltaHedger(max_delta=50, warning_delta=30)

    market = "btc-15m-123"

    # Simular trades
    print("=== Simulando trades ===")

    hedger.update_position(market, "YES", 20, 0.48)
    print(f"Comprou 20 YES @ 0.48")
    print(f"Delta: {hedger.get_delta(market)}")

    hedger.update_position(market, "NO", 15, 0.48)
    print(f"Comprou 15 NO @ 0.48")
    print(f"Delta: {hedger.get_delta(market)}")

    adj = hedger.get_quote_adjustment(market, "YES")
    print(f"\nAjuste YES: bid={adj.bid_adjustment:+.4f}, ask={adj.ask_adjustment:+.4f}")
    print(f"Urgência: {adj.urgency}")
    print(f"Razão: {adj.reason}")

    # Simular acúmulo de YES
    print("\n=== Acumulando YES ===")
    hedger.update_position(market, "YES", 30, 0.49)
    print(f"Delta: {hedger.get_delta(market)}")

    adj = hedger.get_quote_adjustment(market, "YES")
    print(f"Ajuste YES: bid={adj.bid_adjustment:+.4f}")
    print(f"Urgência: {adj.urgency}")

    if hedger.needs_urgent_hedge(market):
        order = hedger.get_hedge_order(market)
        print(f"\n⚠️  HEDGE URGENTE: {order}")
