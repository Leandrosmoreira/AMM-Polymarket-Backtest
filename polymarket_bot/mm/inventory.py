"""
Inventory Manager - Controla exposição para não ficar desbalanceado.

Objetivo:
- Manter posições YES e NO equilibradas
- Ajustar TAMANHO das ordens (não preço)
- Evitar exposição excessiva a um lado

Lógica:
- Se inventory YES > NO: reduz tamanho do BID YES, aumenta BID NO
- Se inventory NO > YES: reduz tamanho do BID NO, aumenta BID YES
- Nunca distorce o spread do mercado
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Posição em um token."""
    shares: float = 0.0
    avg_price: float = 0.0
    total_cost: float = 0.0

    def add(self, shares: float, price: float):
        """Adiciona à posição."""
        cost = shares * price
        new_total_shares = self.shares + shares
        if new_total_shares > 0:
            self.avg_price = (self.total_cost + cost) / new_total_shares
        self.shares = new_total_shares
        self.total_cost += cost

    def remove(self, shares: float):
        """Remove da posição."""
        self.shares = max(0, self.shares - shares)
        self.total_cost = self.shares * self.avg_price

    @property
    def value(self) -> float:
        """Valor atual da posição."""
        return self.total_cost


@dataclass
class MarketInventory:
    """Inventory de um mercado (YES + NO)."""
    market_id: str
    yes: Position = field(default_factory=Position)
    no: Position = field(default_factory=Position)
    max_exposure: float = 100.0  # Máxima exposição em $

    @property
    def yes_exposure(self) -> float:
        """Exposição em YES ($)."""
        return self.yes.total_cost

    @property
    def no_exposure(self) -> float:
        """Exposição em NO ($)."""
        return self.no.total_cost

    @property
    def net_exposure(self) -> float:
        """Exposição líquida (YES - NO)."""
        return self.yes_exposure - self.no_exposure

    @property
    def total_exposure(self) -> float:
        """Exposição total."""
        return self.yes_exposure + self.no_exposure

    @property
    def imbalance_ratio(self) -> float:
        """
        Ratio de desbalanceamento (-1 a +1).

        -1 = 100% NO, 0 = balanceado, +1 = 100% YES
        """
        total = self.total_exposure
        if total == 0:
            return 0.0
        return self.net_exposure / total

    @property
    def is_balanced(self) -> bool:
        """Verifica se está razoavelmente balanceado (±20%)."""
        return abs(self.imbalance_ratio) < 0.2

    def can_buy_yes(self, amount: float) -> bool:
        """Verifica se pode comprar mais YES."""
        return (self.yes_exposure + amount) <= self.max_exposure

    def can_buy_no(self, amount: float) -> bool:
        """Verifica se pode comprar mais NO."""
        return (self.no_exposure + amount) <= self.max_exposure


@dataclass
class SizeAdjustment:
    """Ajuste de tamanho para manter inventory balanceado."""
    yes_bid_multiplier: float = 1.0  # Multiplicador para BID de YES
    no_bid_multiplier: float = 1.0   # Multiplicador para BID de NO
    reason: str = ""


class InventoryManager:
    """
    Gerencia inventory para manter posições balanceadas.

    Estratégia:
    - Comprar YES e NO nas duas pontas (maker)
    - Ajustar TAMANHO baseado no inventory atual
    - Se muito YES → comprar menos YES, mais NO
    - Se muito NO → comprar menos NO, mais YES
    - NUNCA ajustar preço - só tamanho

    Usage:
        inv = InventoryManager(max_exposure=100)
        inv.add_market("btc")

        # Antes de enviar ordens
        adj = inv.get_size_adjustment("btc")
        yes_size = base_size * adj.yes_bid_multiplier
        no_size = base_size * adj.no_bid_multiplier

        # Quando ordem executada
        inv.on_fill("btc", "YES", shares=10, price=0.48)
    """

    def __init__(
        self,
        max_exposure_per_market: float = 100.0,
        max_imbalance: float = 0.3,  # 30% máximo desbalanceamento
        rebalance_aggression: float = 0.5,  # Quão agressivo rebalancear
    ):
        self.max_exposure = max_exposure_per_market
        self.max_imbalance = max_imbalance
        self.rebalance_aggression = rebalance_aggression

        # Inventories por mercado
        self.markets: Dict[str, MarketInventory] = {}

    def add_market(self, market_id: str):
        """Adiciona um mercado para tracking."""
        if market_id not in self.markets:
            self.markets[market_id] = MarketInventory(
                market_id=market_id,
                max_exposure=self.max_exposure,
            )

    def get_inventory(self, market_id: str) -> Optional[MarketInventory]:
        """Retorna inventory de um mercado."""
        return self.markets.get(market_id)

    def on_fill(self, market_id: str, side: str, shares: float, price: float):
        """
        Chamado quando uma ordem é executada.

        Args:
            market_id: ID do mercado
            side: "YES" ou "NO"
            shares: Quantidade executada
            price: Preço de execução
        """
        if market_id not in self.markets:
            self.add_market(market_id)

        inv = self.markets[market_id]

        if side.upper() == "YES":
            inv.yes.add(shares, price)
        else:
            inv.no.add(shares, price)

        logger.info(
            f"[{market_id}] Fill {side} {shares:.1f} @ ${price:.4f} | "
            f"Inventory: YES ${inv.yes_exposure:.2f}, NO ${inv.no_exposure:.2f}, "
            f"Imbalance: {inv.imbalance_ratio*100:+.1f}%"
        )

    def get_size_adjustment(self, market_id: str) -> SizeAdjustment:
        """
        Calcula ajuste de tamanho para manter inventory balanceado.

        Retorna multiplicadores para o tamanho base:
        - yes_bid_multiplier: multiplicar tamanho do BID YES por isso
        - no_bid_multiplier: multiplicar tamanho do BID NO por isso

        Exemplo:
            Se muito YES (imbalance +0.5), retorna:
            - yes_bid_multiplier: 0.5 (comprar menos YES)
            - no_bid_multiplier: 1.5 (comprar mais NO)
        """
        if market_id not in self.markets:
            return SizeAdjustment(reason="market not found")

        inv = self.markets[market_id]
        imbalance = inv.imbalance_ratio

        # Se balanceado, tamanhos iguais
        if abs(imbalance) < 0.1:
            return SizeAdjustment(
                yes_bid_multiplier=1.0,
                no_bid_multiplier=1.0,
                reason="balanced"
            )

        # Calcular ajuste
        # Se imbalance > 0 (muito YES): reduzir YES, aumentar NO
        # Se imbalance < 0 (muito NO): reduzir NO, aumentar YES

        adjustment = abs(imbalance) * self.rebalance_aggression

        if imbalance > 0:
            # Muito YES → comprar menos YES, mais NO
            yes_mult = max(0.2, 1.0 - adjustment)
            no_mult = min(2.0, 1.0 + adjustment)
            reason = f"rebalancing: too much YES ({imbalance*100:+.1f}%)"
        else:
            # Muito NO → comprar menos NO, mais YES
            yes_mult = min(2.0, 1.0 + adjustment)
            no_mult = max(0.2, 1.0 - adjustment)
            reason = f"rebalancing: too much NO ({imbalance*100:+.1f}%)"

        # Verificar limites de exposição
        if not inv.can_buy_yes(10):  # Exemplo: 10 shares
            yes_mult = 0.0
            reason = "YES exposure limit reached"

        if not inv.can_buy_no(10):
            no_mult = 0.0
            reason = "NO exposure limit reached"

        return SizeAdjustment(
            yes_bid_multiplier=round(yes_mult, 2),
            no_bid_multiplier=round(no_mult, 2),
            reason=reason,
        )

    def should_stop_trading(self, market_id: str) -> tuple[bool, str]:
        """
        Verifica se deve parar de operar (exposição muito alta).

        Returns:
            (should_stop, reason)
        """
        if market_id not in self.markets:
            return False, ""

        inv = self.markets[market_id]

        # Parar se exposição total muito alta
        if inv.total_exposure >= self.max_exposure * 0.95:
            return True, f"max exposure reached (${inv.total_exposure:.2f})"

        # Parar se muito desbalanceado
        if abs(inv.imbalance_ratio) >= self.max_imbalance:
            return True, f"max imbalance reached ({inv.imbalance_ratio*100:+.1f}%)"

        return False, ""

    def get_stats(self, market_id: str = None) -> dict:
        """Retorna estatísticas de inventory."""
        if market_id:
            inv = self.markets.get(market_id)
            if not inv:
                return {}
            return {
                "market_id": market_id,
                "yes_shares": inv.yes.shares,
                "yes_exposure": inv.yes_exposure,
                "no_shares": inv.no.shares,
                "no_exposure": inv.no_exposure,
                "total_exposure": inv.total_exposure,
                "net_exposure": inv.net_exposure,
                "imbalance_ratio": inv.imbalance_ratio,
                "is_balanced": inv.is_balanced,
            }

        # Todas
        return {
            mid: self.get_stats(mid) for mid in self.markets
        }

    def print_status(self):
        """Imprime status do inventory."""
        print("=" * 70)
        print("INVENTORY STATUS")
        print("=" * 70)

        total_yes = 0
        total_no = 0

        for mid, inv in self.markets.items():
            imb = inv.imbalance_ratio
            bar_len = 20
            bar_pos = int((imb + 1) / 2 * bar_len)
            bar = "─" * bar_pos + "│" + "─" * (bar_len - bar_pos - 1)

            status = "✓" if inv.is_balanced else "⚠"

            print(
                f"  {status} {mid[:8]:<8} | "
                f"YES: ${inv.yes_exposure:>6.2f} [{bar}] NO: ${inv.no_exposure:>6.2f} | "
                f"Net: ${inv.net_exposure:+.2f}"
            )

            total_yes += inv.yes_exposure
            total_no += inv.no_exposure

        print("-" * 70)
        print(f"  TOTAL: YES ${total_yes:.2f} | NO ${total_no:.2f} | Net ${total_yes - total_no:+.2f}")
        print("=" * 70)


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inv_mgr = InventoryManager(max_exposure_per_market=100)
    inv_mgr.add_market("btc-15m")

    print("=== Simulação de Fills ===\n")

    # Simular compras
    inv_mgr.on_fill("btc-15m", "YES", 10, 0.48)
    inv_mgr.on_fill("btc-15m", "NO", 10, 0.50)

    adj = inv_mgr.get_size_adjustment("btc-15m")
    print(f"Adjustment: YES mult={adj.yes_bid_multiplier}, NO mult={adj.no_bid_multiplier}")

    # Mais YES (desbalanceia)
    inv_mgr.on_fill("btc-15m", "YES", 15, 0.52)
    inv_mgr.on_fill("btc-15m", "YES", 10, 0.55)

    adj = inv_mgr.get_size_adjustment("btc-15m")
    print(f"Adjustment: YES mult={adj.yes_bid_multiplier}, NO mult={adj.no_bid_multiplier} ({adj.reason})")

    print()
    inv_mgr.print_status()
