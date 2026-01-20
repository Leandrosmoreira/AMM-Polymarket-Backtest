"""
Market Maker Bot (Bot 2) - Estratégia de Market Making para Polymarket.

Diferente do Bot 1 (arbitragem), este bot:
- Cria liquidez (maker, não taker)
- Quote dos dois lados (bid e ask)
- Ajusta spread baseado em volatilidade
- Mantém posição delta-neutral
- Opera múltiplos mercados (BTC, ETH, SOL)

Usage:
    python -m polymarket_bot --market-maker
    python -m polymarket_bot.market_maker_bot
"""
from .performance import setup_performance
setup_performance()

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

from .config import load_settings, Settings
from .trading import get_client
from .markets import MarketManager, Market, SUPPORTED_ASSETS, discover_markets
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
class MarketMakerState:
    """Estado de market making para um mercado."""
    market: Market
    mid_price: float = 0.5
    spread: float = 0.02
    best_bid: float = 0.49
    best_ask: float = 0.51
    our_bid_id: Optional[str] = None
    our_ask_id: Optional[str] = None
    last_quote_time: float = 0.0
    quotes_sent: int = 0
    fills: int = 0


class MarketMakerBot:
    """
    Bot de Market Making Multi-Mercado.

    Estratégia:
    1. Conectar aos mercados (BTC, ETH, SOL)
    2. Calcular mid price e spread ideal para cada
    3. Ajustar spread baseado em volatilidade
    4. Colocar bid e ask
    5. Quando fill, ajustar delta
    6. Re-quotar quando preço move
    7. Rollover automático a cada 15min

    Parâmetros principais:
    - base_spread: Spread base (ex: 0.02 = 2%)
    - max_position: Posição máxima por lado
    - requote_threshold: Movimento que dispara re-quote
    """

    def __init__(self, settings: Settings, assets: List[str] = None):
        self.settings = settings
        self.assets = assets or settings.asset_list
        self.client = get_client(settings)

        # Parâmetros de Market Making
        self.base_spread = float(getattr(settings, 'mm_base_spread', 0.02))
        self.max_position = float(getattr(settings, 'mm_max_position', 100))
        self.requote_threshold = float(getattr(settings, 'mm_requote_threshold', 0.005))
        self.order_size = float(settings.order_size)
        self.requote_interval = float(getattr(settings, 'mm_requote_interval', 5.0))

        # Market management
        self.market_manager = MarketManager(assets=self.assets)
        self.market_states: Dict[str, MarketMakerState] = {}

        # Módulos compartilhados
        self.volatility_engines: Dict[str, VolatilityEngine] = {}
        self.hedger = DeltaHedger(max_delta=self.max_position * 0.5)
        self.order_manager = OrderManager(self.client)

        # Fast logger (JSONL)
        self.fast_logger = FastTradeLogger(log_dir="logs", format="jsonl")

        # Control
        self.running = False
        self.total_quotes = 0
        self.total_fills = 0

        logger.info("=" * 70)
        logger.info("MARKET MAKER BOT (Bot 2) - Multi-Market")
        logger.info("=" * 70)
        logger.info(f"Assets:             {', '.join(a.upper() for a in self.assets)}")
        logger.info(f"Base spread:        {self.base_spread * 100:.1f}%")
        logger.info(f"Max position:       ${self.max_position:.0f}")
        logger.info(f"Requote interval:   {self.requote_interval}s")
        logger.info(f"Order size:         {self.order_size:.0f}")
        logger.info(f"Mode:               {'SIMULATION' if settings.dry_run else 'LIVE'}")
        logger.info("=" * 70)

    def discover_markets(self):
        """Descobre todos os mercados ativos."""
        self.market_manager.discover_all()

        for asset, market in self.market_manager.markets.items():
            if asset not in self.market_states:
                self.market_states[asset] = MarketMakerState(market=market)
                self.volatility_engines[asset] = VolatilityEngine(lookback=100)
            else:
                self.market_states[asset].market = market

    def get_order_book(self, token_id: str) -> dict:
        """Busca orderbook."""
        try:
            book = self.client.get_order_book(token_id=token_id)

            def extract_levels(levels):
                result = []
                for level in (levels or []):
                    try:
                        price = float(level.price if hasattr(level, 'price') else level['price'])
                        size = float(level.size if hasattr(level, 'size') else level['size'])
                        if size > 0:
                            result.append((price, size))
                    except:
                        pass
                return result

            bids = extract_levels(book.bids if hasattr(book, 'bids') else [])
            asks = extract_levels(book.asks if hasattr(book, 'asks') else [])

            best_bid = max((p for p, _ in bids), default=None)
            best_ask = min((p for p, _ in asks), default=None)

            return {"best_bid": best_bid, "best_ask": best_ask, "bids": bids, "asks": asks}

        except Exception as e:
            logger.error(f"Error getting orderbook: {e}")
            return {}

    def update_market_state(self, asset: str):
        """Atualiza estado do mercado (preços, volatilidade)."""
        state = self.market_states.get(asset)
        if not state or not state.market.is_open:
            return

        market = state.market

        # Buscar orderbook do YES token
        book = self.get_order_book(market.yes_token_id)

        if book.get("best_bid") and book.get("best_ask"):
            state.best_bid = book["best_bid"]
            state.best_ask = book["best_ask"]
            state.mid_price = (state.best_bid + state.best_ask) / 2
            state.spread = state.best_ask - state.best_bid

            # Atualizar volatility engine
            vol_engine = self.volatility_engines.get(asset)
            if vol_engine:
                vol_engine.update(state.mid_price, state.spread)

    def calculate_quotes(self, asset: str) -> tuple[Quote, Quote]:
        """Calcula quotes (bid e ask) para um mercado."""
        state = self.market_states[asset]
        vol_engine = self.volatility_engines.get(asset)

        # Recomendações de volatilidade
        if vol_engine:
            vol_rec = vol_engine.get_recommendations()
            spread_mult = vol_rec.spread_multiplier
            size_mult = vol_rec.size_multiplier
        else:
            spread_mult = 1.0
            size_mult = 1.0

        # Ajuste de delta
        delta_adj = self.hedger.get_quote_adjustment(state.market.market_id, "YES")

        # Calcular spread ajustado
        adjusted_spread = self.base_spread * spread_mult
        adjusted_size = self.order_size * size_mult

        # Calcular preços
        bid_price = state.mid_price - (adjusted_spread / 2) + delta_adj.bid_adjustment
        ask_price = state.mid_price + (adjusted_spread / 2) + delta_adj.ask_adjustment

        # Garantir bid < ask
        if bid_price >= ask_price:
            mid = (bid_price + ask_price) / 2
            bid_price = mid - 0.005
            ask_price = mid + 0.005

        # Limitar preços
        bid_price = max(0.01, min(0.99, round(bid_price, 4)))
        ask_price = max(0.01, min(0.99, round(ask_price, 4)))

        bid = Quote(price=bid_price, size=adjusted_size, side="BUY")
        ask = Quote(price=ask_price, size=adjusted_size, side="SELL")

        return bid, ask

    def should_requote(self, asset: str) -> bool:
        """Verifica se deve re-quotar."""
        state = self.market_states.get(asset)
        if not state:
            return False

        # Se não tem quotes ativos
        if not state.our_bid_id or not state.our_ask_id:
            return True

        # Re-quotar a cada N segundos
        if time.time() - state.last_quote_time > self.requote_interval:
            return True

        return False

    async def send_quotes(self, asset: str):
        """Envia quotes para um mercado."""
        state = self.market_states.get(asset)
        if not state or not state.market.is_open:
            return

        market = state.market
        vol_engine = self.volatility_engines.get(asset)

        # Verificar volatilidade
        if vol_engine:
            vol_rec = vol_engine.get_recommendations()
            if not vol_rec.should_quote:
                logger.warning(f"{asset.upper()}: Volatility too high ({vol_rec.regime}), not quoting")
                return

        # Cancelar quotes existentes
        if state.our_bid_id:
            await self.order_manager.cancel_order(state.our_bid_id)
            state.our_bid_id = None
        if state.our_ask_id:
            await self.order_manager.cancel_order(state.our_ask_id)
            state.our_ask_id = None

        # Calcular novos quotes
        bid, ask = self.calculate_quotes(asset)

        # Log
        spread_pct = (ask.price - bid.price) * 100
        logger.info(
            f"{asset.upper()}: BID ${bid.price:.4f} x {bid.size:.0f} | "
            f"ASK ${ask.price:.4f} x {ask.size:.0f} | "
            f"Spread: {spread_pct:.2f}% | "
            f"[{market.time_remaining_str}]"
        )

        if self.settings.dry_run:
            # Simulação
            state.our_bid_id = f"SIM_BID_{asset}_{int(time.time())}"
            state.our_ask_id = f"SIM_ASK_{asset}_{int(time.time())}"
        else:
            # Enviar ordens reais
            bid_id = await self.order_manager.submit_order(
                market_id=market.market_id,
                token=market.yes_token_id,
                side="BUY",
                price=bid.price,
                size=bid.size,
                order_type="GTC",
            )
            if bid_id:
                state.our_bid_id = bid_id

            ask_id = await self.order_manager.submit_order(
                market_id=market.market_id,
                token=market.yes_token_id,
                side="SELL",
                price=ask.price,
                size=ask.size,
                order_type="GTC",
            )
            if ask_id:
                state.our_ask_id = ask_id

        state.last_quote_time = time.time()
        state.quotes_sent += 2
        self.total_quotes += 2

        # Log JSONL
        self.fast_logger.log_event("quote", {
            "asset": asset,
            "market": market.slug,
            "bid_price": bid.price,
            "bid_size": bid.size,
            "ask_price": ask.price,
            "ask_size": ask.size,
            "spread_pct": spread_pct,
            "time_remaining": market.time_remaining,
        })

    async def run_cycle(self):
        """Executa um ciclo de market making."""
        # Refresh markets se necessário
        if self.market_manager.refresh_if_needed():
            self.discover_markets()

        # Para cada mercado
        for asset in self.assets:
            state = self.market_states.get(asset)
            if not state or not state.market.is_open:
                continue

            # Atualizar estado
            self.update_market_state(asset)

            # Re-quotar se necessário
            if self.should_requote(asset):
                await self.send_quotes(asset)

            # Pequeno delay entre mercados
            await asyncio.sleep(0.1)

    def print_status(self):
        """Imprime status atual."""
        print()
        print("=" * 70)
        print(f"MARKET MAKER STATUS @ {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)

        for asset in self.assets:
            state = self.market_states.get(asset)
            if state and state.market:
                market = state.market
                vol = self.volatility_engines.get(asset)
                vol_regime = vol.get_regime() if vol else "?"
                delta = self.hedger.get_delta(market.market_id)

                status = "" if market.is_open else ""
                print(
                    f"  {status} {asset.upper():<4} | "
                    f"Mid: ${state.mid_price:.4f} | "
                    f"Spread: {state.spread*100:.2f}% | "
                    f"Vol: {vol_regime:<6} | "
                    f"Delta: {delta:+.1f} | "
                    f"[{market.time_remaining_str}]"
                )

        print("-" * 70)
        print(f"  Quotes: {self.total_quotes} | Fills: {self.total_fills} | Delta total: {self.hedger.get_total_delta():.1f}")
        print("=" * 70)

    async def run(self, interval: float = 1.0):
        """Loop principal do market maker."""
        self.running = True

        logger.info("Starting Market Maker...")

        # Descobrir mercados
        self.discover_markets()

        last_status = 0
        cycle = 0

        try:
            while self.running:
                cycle += 1

                await self.run_cycle()

                # Status a cada 30s
                now = time.time()
                if now - last_status > 30:
                    self.print_status()
                    last_status = now

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\nStopping Market Maker...")

        finally:
            # Cancelar todas ordens
            if not self.settings.dry_run:
                await self.order_manager.cancel_all()

            self.fast_logger.stop()
            self.show_summary()

    def show_summary(self):
        """Mostra resumo final."""
        print()
        print("=" * 70)
        print("MARKET MAKER SUMMARY")
        print("=" * 70)

        for asset in self.assets:
            state = self.market_states.get(asset)
            if state:
                print(f"  {asset.upper()}: {state.quotes_sent} quotes, {state.fills} fills")

        print("-" * 70)
        print(f"  Total quotes:     {self.total_quotes}")
        print(f"  Total fills:      {self.total_fills}")
        print(f"  Total delta:      {self.hedger.get_total_delta():.1f}")

        # Order manager stats
        om_stats = self.order_manager.get_stats()
        print(f"  Orders submitted: {om_stats['orders_submitted']}")
        print(f"  Orders filled:    {om_stats['orders_filled']}")
        print(f"  Fill rate:        {om_stats['fill_rate']:.1f}%")

        print("=" * 70)

        # Log location
        stats = self.fast_logger.get_stats()
        print(f"\nLogs: {stats['trades_file']}")

    def stop(self):
        """Para o bot."""
        self.running = False


async def main():
    """Entry point."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    settings = load_settings()

    # Validar (menos restritivo em dry_run)
    is_valid, errors = settings.validate()
    if not is_valid and not settings.dry_run:
        logger.error("Configuration errors:")
        for err in errors:
            logger.error(f"  - {err}")
        return

    # Assets da linha de comando ou config
    assets = None
    args = [a.lower() for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        assets = [a for a in args if a in SUPPORTED_ASSETS]

    bot = MarketMakerBot(settings, assets=assets)
    await bot.run(interval=1.0)


if __name__ == "__main__":
    asyncio.run(main())
