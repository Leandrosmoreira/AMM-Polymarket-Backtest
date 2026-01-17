"""
Market Maker Bot (Bot 2) - Estratégia de Market Making para Polymarket.

Diferente do Bot 1 (arbitragem), este bot:
- Cria liquidez (maker, não taker)
- Quote dos dois lados (bid e ask)
- Ajusta spread baseado em volatilidade
- Mantém posição delta-neutral
- Opera múltiplos mercados

Usage:
    python -m trading_bot_ltm.market_maker
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

from .config import load_settings, Settings
from .trading import get_client
from .mm.volatility import VolatilityEngine
from .mm.delta_hedge import DeltaHedger
from .mm.order_manager import OrderManager
from .fast_logger import FastTradeLogger

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    """Um quote (bid ou ask)."""
    price: float
    size: float
    side: str  # "BUY" or "SELL"


@dataclass
class MarketState:
    """Estado atual de um mercado."""
    market_id: str
    yes_token_id: str
    no_token_id: str
    mid_price: float
    spread: float
    best_bid: float
    best_ask: float
    our_bid: Optional[str] = None  # Order ID
    our_ask: Optional[str] = None  # Order ID
    last_update: float = 0.0


class MarketMakerBot:
    """
    Bot de Market Making.

    Estratégia:
    1. Conectar aos mercados
    2. Calcular mid price e spread ideal
    3. Ajustar spread baseado em volatilidade
    4. Colocar bid e ask
    5. Quando fill, ajustar delta
    6. Re-quotar quando preço move

    Parâmetros principais:
    - base_spread: Spread base (ex: 0.02 = 2%)
    - max_position: Posição máxima por lado
    - requote_threshold: Movimento que dispara re-quote
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = get_client(settings)

        # Parâmetros de Market Making
        self.base_spread = float(getattr(settings, 'mm_base_spread', 0.02))
        self.max_position = float(getattr(settings, 'mm_max_position', 100))
        self.requote_threshold = float(getattr(settings, 'mm_requote_threshold', 0.005))
        self.order_size = float(settings.order_size)

        # Módulos
        self.volatility = VolatilityEngine(lookback=100)
        self.hedger = DeltaHedger(max_delta=self.max_position * 0.5)
        self.order_manager = OrderManager(self.client)

        # Fast logger
        self.fast_logger = FastTradeLogger(log_dir="logs", buffer_size=10)

        # Estado dos mercados: {market_id: MarketState}
        self.markets: Dict[str, MarketState] = {}

        # Control
        self.running = False
        self.last_quote_time = 0.0
        self.quotes_sent = 0
        self.fills = 0

        logger.info("=" * 60)
        logger.info("  MARKET MAKER BOT (Bot 2)")
        logger.info("=" * 60)
        logger.info(f"  Base spread:        {self.base_spread * 100:.1f}%")
        logger.info(f"  Max position:       ${self.max_position:.0f}")
        logger.info(f"  Requote threshold:  {self.requote_threshold * 100:.2f}%")
        logger.info(f"  Order size:         {self.order_size:.0f}")
        logger.info("=" * 60)

    def add_market(self, market_id: str, yes_token_id: str, no_token_id: str):
        """Adiciona um mercado para quotar."""
        self.markets[market_id] = MarketState(
            market_id=market_id,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            mid_price=0.5,
            spread=self.base_spread,
            best_bid=0.49,
            best_ask=0.51,
        )
        logger.info(f"Added market: {market_id}")

    async def fetch_orderbook(self, market_id: str) -> Optional[dict]:
        """Busca orderbook do mercado."""
        try:
            state = self.markets[market_id]

            # Buscar books de YES e NO
            yes_book = self.client.get_order_book(token_id=state.yes_token_id)
            no_book = self.client.get_order_book(token_id=state.no_token_id)

            # Extrair best prices
            def get_best(book, side):
                levels = book.asks if side == "ask" else book.bids
                if not levels:
                    return None
                prices = [float(l.price) if hasattr(l, 'price') else float(l['price']) for l in levels]
                return min(prices) if side == "ask" else max(prices)

            best_bid = get_best(yes_book, "bid")
            best_ask = get_best(yes_book, "ask")

            if best_bid and best_ask:
                state.best_bid = best_bid
                state.best_ask = best_ask
                state.mid_price = (best_bid + best_ask) / 2
                state.spread = best_ask - best_bid
                state.last_update = time.time()

                # Atualizar volatility engine
                self.volatility.update(state.mid_price, state.spread)

                return {"bid": best_bid, "ask": best_ask, "mid": state.mid_price}

        except Exception as e:
            logger.error(f"Error fetching orderbook: {e}")

        return None

    def calculate_quotes(self, market_id: str) -> tuple[Quote, Quote]:
        """
        Calcula quotes (bid e ask) para um mercado.

        Considera:
        - Mid price atual
        - Spread base + ajuste por volatilidade
        - Ajuste de delta para rebalanceamento
        """
        state = self.markets[market_id]

        # Obter recomendações de volatilidade
        vol_rec = self.volatility.get_recommendations()

        # Obter ajuste de delta
        delta_adj = self.hedger.get_quote_adjustment(market_id, "YES")

        # Calcular spread ajustado
        adjusted_spread = self.base_spread * vol_rec.spread_multiplier

        # Calcular tamanho ajustado
        adjusted_size = self.order_size * vol_rec.size_multiplier

        # Calcular preços
        bid_price = state.mid_price - (adjusted_spread / 2) + delta_adj.bid_adjustment
        ask_price = state.mid_price + (adjusted_spread / 2) + delta_adj.ask_adjustment

        # Garantir que bid < ask
        if bid_price >= ask_price:
            mid = (bid_price + ask_price) / 2
            bid_price = mid - 0.005
            ask_price = mid + 0.005

        # Limitar preços entre 0.01 e 0.99
        bid_price = max(0.01, min(0.99, bid_price))
        ask_price = max(0.01, min(0.99, ask_price))

        bid = Quote(price=round(bid_price, 4), size=adjusted_size, side="BUY")
        ask = Quote(price=round(ask_price, 4), size=adjusted_size, side="SELL")

        return bid, ask

    def should_requote(self, market_id: str) -> bool:
        """Verifica se deve re-quotar (preço moveu além do threshold)."""
        state = self.markets[market_id]

        # Se não tem quotes ativos, deve quotar
        if not state.our_bid or not state.our_ask:
            return True

        # Calcular novos quotes ideais
        new_bid, new_ask = self.calculate_quotes(market_id)

        # Verificar se os quotes atuais estão muito longe dos ideais
        # (Precisaria comparar com os preços das ordens ativas)
        # Por simplicidade, re-quotar a cada N segundos
        if time.time() - state.last_update > 5.0:
            return True

        return False

    async def send_quotes(self, market_id: str):
        """Envia quotes (bid e ask) para o mercado."""
        state = self.markets[market_id]

        # Verificar se pode quotar
        vol_rec = self.volatility.get_recommendations()
        if not vol_rec.should_quote:
            logger.warning(f"Volatility too high, not quoting: {vol_rec.regime}")
            return

        # Cancelar quotes existentes
        if state.our_bid:
            await self.order_manager.cancel_order(state.our_bid)
            state.our_bid = None
        if state.our_ask:
            await self.order_manager.cancel_order(state.our_ask)
            state.our_ask = None

        # Calcular novos quotes
        bid, ask = self.calculate_quotes(market_id)

        # Enviar bid (comprar YES)
        bid_id = await self.order_manager.submit_order(
            market_id=market_id,
            token=state.yes_token_id,
            side="BUY",
            price=bid.price,
            size=bid.size,
            order_type="GTC",
        )
        if bid_id:
            state.our_bid = bid_id

        # Enviar ask (vender YES)
        ask_id = await self.order_manager.submit_order(
            market_id=market_id,
            token=state.yes_token_id,
            side="SELL",
            price=ask.price,
            size=ask.size,
            order_type="GTC",
        )
        if ask_id:
            state.our_ask = ask_id

        self.quotes_sent += 2
        self.last_quote_time = time.time()

        logger.info(
            f"[{market_id}] Quotes sent: "
            f"BID ${bid.price:.4f} x {bid.size:.0f} | "
            f"ASK ${ask.price:.4f} x {ask.size:.0f} | "
            f"Spread: {(ask.price - bid.price) * 100:.2f}%"
        )

    def on_fill(self, order):
        """Callback quando uma ordem é executada."""
        self.fills += 1

        # Atualizar posição no hedger
        self.hedger.update_position(
            market_id=order.market_id,
            token="YES",  # Simplificado
            size_delta=order.filled_size if order.side == "BUY" else -order.filled_size,
            price=order.price,
        )

        # Log
        logger.info(
            f"FILL: {order.side} {order.filled_size:.0f} @ ${order.price:.4f} "
            f"| Delta: {self.hedger.get_delta(order.market_id):.1f}"
        )

        # Fast log para backtest
        self.fast_logger.log_trade(
            market=order.market_id,
            price_up=order.price if order.side == "BUY" else 0,
            price_down=0 if order.side == "BUY" else order.price,
            pair_cost=order.price,
            profit_pct=0,  # Calcular depois
            order_size=order.filled_size,
            investment=order.price * order.filled_size,
            expected_profit=0,
            balance_after=0,
            ltm_bucket=None,
        )

        # Verificar se precisa hedge urgente
        if self.hedger.needs_urgent_hedge(order.market_id):
            hedge_order = self.hedger.get_hedge_order(order.market_id)
            logger.warning(f"URGENT HEDGE NEEDED: {hedge_order}")

    async def run_once(self):
        """Executa um ciclo de market making."""
        for market_id in self.markets:
            # Atualizar orderbook
            await self.fetch_orderbook(market_id)

            # Verificar se deve re-quotar
            if self.should_requote(market_id):
                await self.send_quotes(market_id)

            # Pequeno delay entre mercados
            await asyncio.sleep(0.1)

    async def run(self, interval_seconds: float = 1.0):
        """Loop principal do market maker."""
        self.running = True

        # Registrar callback de fill
        self.order_manager.on_fill = self.on_fill

        logger.info("Market Maker started")

        try:
            while self.running:
                await self.run_once()
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Stopping Market Maker...")

        finally:
            # Cancelar todas as ordens ao parar
            await self.order_manager.cancel_all()
            self.fast_logger.stop()

            # Stats finais
            logger.info("=" * 60)
            logger.info("  MARKET MAKER SUMMARY")
            logger.info("=" * 60)
            logger.info(f"  Quotes sent:    {self.quotes_sent}")
            logger.info(f"  Fills:          {self.fills}")
            logger.info(f"  Volatility:     {self.volatility.get_regime()}")
            logger.info(f"  Delta:          {self.hedger.get_total_delta():.1f}")
            logger.info("=" * 60)

    def stop(self):
        """Para o bot."""
        self.running = False


async def main():
    """Entry point para market maker bot."""
    from .logger import setup_logging, print_header

    settings = load_settings()
    setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)

    print_header("Market Maker Bot (Bot 2)")

    bot = MarketMakerBot(settings)

    # Adicionar mercados
    # Por enquanto, adicionar um mercado simulado
    # Em produção, buscar mercados ativos da API
    bot.add_market(
        market_id="btc-15m-test",
        yes_token_id="SIMULATED_YES",
        no_token_id="SIMULATED_NO",
    )

    # Rodar
    await bot.run(interval_seconds=1.0)


if __name__ == "__main__":
    asyncio.run(main())
