"""
MarketMakingAgent - Geração de quotes para market making
Calcula preços e tamanhos para ordens de compra
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from core.types import (
    AgentDecision, AgentType, Quote, TokenType, Side,
    BookSnapshot, PaperPosition
)
from core.buffers import MarketDataState
from config.gabagool_config import GabagoolConfig


@dataclass
class QuoteResult:
    """Resultado da geração de quotes."""
    quotes: List[Quote]
    total_notional: float
    avg_pair_cost: float
    is_balanced: bool
    reason: str


class MarketMakingAgent:
    """
    Agente de market making.

    Responsabilidades:
    1. Gerar quotes para YES e NO
    2. Balancear posição (manter YES ≈ NO)
    3. Ajustar tamanhos baseado em edge strength
    4. Respeitar limites de spread
    """

    __slots__ = ('config', '_last_quotes', '_position')

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._last_quotes: Optional[QuoteResult] = None
        self._position: Optional[PaperPosition] = None

    def analyze(
        self,
        market_data: MarketDataState,
        position: Optional[PaperPosition],
        edge_strength: float,
        max_size: float
    ) -> AgentDecision:
        """
        Gera quotes baseado nos dados de mercado.

        Args:
            market_data: Estado atual dos dados
            position: Posição atual
            edge_strength: Força do edge detectado (0-1)
            max_size: Tamanho máximo permitido pelo RiskAgent

        Returns:
            AgentDecision com quotes e dados
        """
        self._position = position

        # Gerar quotes
        result = self._generate_quotes(market_data, position, edge_strength, max_size)
        self._last_quotes = result

        should_trade = len(result.quotes) > 0 and result.total_notional > 0

        return AgentDecision(
            agent=AgentType.MARKET_MAKING,
            should_trade=should_trade,
            confidence=edge_strength,
            reason=result.reason,
            data={
                "quotes": [self._quote_to_dict(q) for q in result.quotes],
                "total_notional": result.total_notional,
                "avg_pair_cost": result.avg_pair_cost,
                "is_balanced": result.is_balanced,
            }
        )

    def _generate_quotes(
        self,
        market_data: MarketDataState,
        position: Optional[PaperPosition],
        edge_strength: float,
        max_size: float
    ) -> QuoteResult:
        """Gera quotes para ambos os lados."""
        quotes = []
        total_notional = 0.0

        # Get current prices
        yes_mid = market_data.yes_mid
        no_mid = market_data.no_mid

        if yes_mid is None or no_mid is None:
            return QuoteResult(
                quotes=[],
                total_notional=0.0,
                avg_pair_cost=1.0,
                is_balanced=True,
                reason="no price data"
            )

        pair_cost = yes_mid + no_mid

        # Check if pair cost is acceptable
        if pair_cost > self.config.pair_cost_max:
            return QuoteResult(
                quotes=[],
                total_notional=0.0,
                avg_pair_cost=pair_cost,
                is_balanced=True,
                reason=f"pair cost too high ({pair_cost:.4f})"
            )

        # Calculate base size
        base_size = self._calculate_size(edge_strength, max_size)

        if base_size < self.config.min_order_usd:
            return QuoteResult(
                quotes=[],
                total_notional=0.0,
                avg_pair_cost=pair_cost,
                is_balanced=True,
                reason=f"size too small (${base_size:.2f})"
            )

        # Determine YES/NO allocation
        yes_size, no_size = self._balance_allocation(base_size, position)

        # Create quotes
        if yes_size > 0:
            yes_shares = yes_size / yes_mid
            quotes.append(Quote(
                token_type=TokenType.YES,
                side=Side.BUY,
                price=yes_mid,
                size=yes_shares
            ))
            total_notional += yes_size

        if no_size > 0:
            no_shares = no_size / no_mid
            quotes.append(Quote(
                token_type=TokenType.NO,
                side=Side.BUY,
                price=no_mid,
                size=no_shares
            ))
            total_notional += no_size

        # Check balance
        is_balanced = abs(yes_size - no_size) / max(yes_size + no_size, 1) < 0.1

        return QuoteResult(
            quotes=quotes,
            total_notional=total_notional,
            avg_pair_cost=pair_cost,
            is_balanced=is_balanced,
            reason=f"quotes generated: ${total_notional:.2f} total"
        )

    def _calculate_size(self, edge_strength: float, max_size: float) -> float:
        """Calcula tamanho da ordem baseado no edge strength."""
        # Base size from config
        base = self.config.base_order_usd

        # Scale by edge strength
        scaled = base * (0.5 + edge_strength * 0.5)

        # Apply quote size percentage
        sized = scaled * self.config.quote_size_pct

        # Clamp to limits
        sized = max(self.config.min_order_usd, sized)
        sized = min(self.config.max_order_usd, sized)
        sized = min(max_size, sized)

        return sized

    def _balance_allocation(
        self,
        total_size: float,
        position: Optional[PaperPosition]
    ) -> Tuple[float, float]:
        """
        Distribui tamanho entre YES e NO para manter balanço.

        Returns:
            (yes_size, no_size)
        """
        if position is None or (position.yes_qty == 0 and position.no_qty == 0):
            # Nova posição: dividir igualmente
            half = total_size / 2
            return half, half

        # Calcular desbalanço atual
        yes_value = position.yes_cost
        no_value = position.no_cost
        total_value = yes_value + no_value

        if total_value == 0:
            return total_size / 2, total_size / 2

        yes_pct = yes_value / total_value
        no_pct = no_value / total_value

        # Se desbalanceado, favorecer o lado menor
        imbalance = yes_pct - 0.5  # Positivo = mais YES, negativo = mais NO

        if abs(imbalance) > self.config.rebalance_threshold:
            # Rebalanceamento agressivo
            if imbalance > 0:
                # Mais YES que NO, comprar mais NO
                no_size = total_size * 0.7
                yes_size = total_size * 0.3
            else:
                # Mais NO que YES, comprar mais YES
                yes_size = total_size * 0.7
                no_size = total_size * 0.3
        else:
            # Levemente ajustado
            adjustment = imbalance * 0.5  # Suaviza o ajuste
            yes_size = total_size * (0.5 - adjustment)
            no_size = total_size * (0.5 + adjustment)

        # Garantir mínimos
        min_single = self.config.min_order_usd / 2
        yes_size = max(min_single, yes_size)
        no_size = max(min_single, no_size)

        return yes_size, no_size

    def _quote_to_dict(self, quote: Quote) -> Dict[str, Any]:
        """Convert quote to dictionary."""
        return {
            "token_type": quote.token_type.value,
            "side": quote.side.value,
            "price": quote.price,
            "size": quote.size,
            "notional": quote.notional
        }

    def get_rebalance_quote(
        self,
        market_data: MarketDataState,
        position: PaperPosition
    ) -> Optional[Quote]:
        """
        Gera quote de rebalanceamento se necessário.

        Returns:
            Quote para rebalancear ou None
        """
        if position.is_balanced:
            return None

        yes_mid = market_data.yes_mid
        no_mid = market_data.no_mid

        if yes_mid is None or no_mid is None:
            return None

        # Determinar qual lado comprar
        if position.unhedged_yes > position.unhedged_no:
            # Mais YES, comprar NO
            target_shares = position.unhedged_yes - position.unhedged_no
            max_cost = target_shares * no_mid
            max_cost = min(max_cost, self.config.max_order_usd)

            return Quote(
                token_type=TokenType.NO,
                side=Side.BUY,
                price=no_mid,
                size=max_cost / no_mid
            )
        else:
            # Mais NO, comprar YES
            target_shares = position.unhedged_no - position.unhedged_yes
            max_cost = target_shares * yes_mid
            max_cost = min(max_cost, self.config.max_order_usd)

            return Quote(
                token_type=TokenType.YES,
                side=Side.BUY,
                price=yes_mid,
                size=max_cost / yes_mid
            )

    @property
    def last_quotes(self) -> Optional[QuoteResult]:
        return self._last_quotes
