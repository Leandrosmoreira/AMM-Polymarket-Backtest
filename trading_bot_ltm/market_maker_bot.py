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
from .structured_logger import get_structured_logger
from .market_finder import find_all_active_markets, get_market_info
from .inventory_model import InventoryManager

logger = logging.getLogger(__name__)
structured_logger = get_structured_logger()


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
        
        # Inventory Manager
        initial_cash = float(getattr(settings, 'initial_cash', 10000.0))
        self.inventory_manager = InventoryManager(initial_cash=initial_cash, log_dir="logs")

        # Fast logger
        self.fast_logger = FastTradeLogger(log_dir="logs", buffer_size=10)

        # Estado dos mercados: {market_id: MarketState}
        self.markets: Dict[str, MarketState] = {}

        # Control
        self.running = False
        self.last_quote_time = 0.0
        self.quotes_sent = 0
        self.fills = 0
        self.cycle_count = 0  # Contador de ciclos para logging

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
    
    def add_market_from_slug(self, slug: str) -> bool:
        """
        Adiciona um mercado a partir do slug.
        
        Args:
            slug: Market slug (ex: 'btc-updown-15m-1234567890')
        
        Returns:
            True se adicionado com sucesso, False caso contrário
        """
        try:
            market_info = get_market_info(slug)
            market_id = market_info.get("market_id")
            yes_token_id = market_info.get("yes_token_id")
            no_token_id = market_info.get("no_token_id")
            
            if not all([market_id, yes_token_id, no_token_id]):
                logger.error(f"Invalid market info for {slug}")
                return False
            
            self.add_market(market_id, yes_token_id, no_token_id)
            logger.info(f"✅ Added market from slug: {slug} -> {market_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding market from slug {slug}: {e}")
            return False
    
    def discover_and_add_markets(self, assets: List[str] = None):
        """
        Descobre e adiciona mercados ativos automaticamente.
        
        Args:
            assets: Lista de ativos para buscar (default: ['btc', 'eth', 'sol'])
        """
        if assets is None:
            assets = ['btc', 'eth', 'sol']
        
        logger.info(f"🔍 Discovering markets for: {', '.join(asset.upper() for asset in assets)}")
        
        all_markets = find_all_active_markets()
        
        added_count = 0
        for asset in assets:
            slug = all_markets.get(asset.lower())
            if slug:
                if self.add_market_from_slug(slug):
                    added_count += 1
            else:
                logger.warning(f"⚠️  No active {asset.upper()} 15min market found")
        
        logger.info(f"✅ Added {added_count} markets")
        
        if added_count == 0:
            logger.warning("⚠️  No markets added! Bot will not operate.")

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

        # Limitar preços entre 0.02 e 0.98 ANTES de verificar spread
        # (deixamos margem para ajuste do spread minimo)
        bid_price = max(0.02, min(0.98, bid_price))
        ask_price = max(0.02, min(0.98, ask_price))

        # Garantir spread minimo de 1% (0.01)
        min_spread = 0.01
        if ask_price - bid_price < min_spread:
            mid = (bid_price + ask_price) / 2
            # Ajustar para garantir spread minimo
            bid_price = max(0.01, mid - min_spread / 2)
            ask_price = min(0.99, mid + min_spread / 2)
            # Se ainda nao tem spread suficiente, ajustar novamente
            if ask_price - bid_price < min_spread:
                if bid_price <= 0.01:
                    ask_price = bid_price + min_spread
                else:
                    bid_price = ask_price - min_spread

        bid = Quote(price=round(bid_price, 4), size=adjusted_size, side="BUY")
        ask = Quote(price=round(ask_price, 4), size=adjusted_size, side="SELL")

        return bid, ask

    def _should_quote(self, market_id: str) -> tuple[bool, str]:
        """
        Verifica se deve cotar e retorna (deve_cotar, motivo).
        NÍVEL 3: Log de decisão (skip reasons).
        """
        state = self.markets[market_id]
        
        # Verificar delta
        delta = self.hedger.get_delta(market_id)
        if abs(delta) > self.max_position * 0.8:
            return False, f"delta_limit (delta={delta:.2f})"
        
        # Verificar inventory (usar inventory real do InventoryManager)
        state = self.markets.get(market_id)
        mid_price = state.mid_price if state else 0.5
        inventory = self.inventory_manager.get_inventory_value(market_id, current_price=mid_price)
        if abs(inventory) > self.max_position:
            return False, f"inventory_limit (inventory=${inventory:.2f})"
        
        # Verificar volatilidade extrema
        vol_rec = self.volatility.get_recommendations()
        if not vol_rec.should_quote:
            return False, f"volatility_too_high (regime={vol_rec.regime})"
        
        # Verificar tempo restante (se disponível)
        # Por enquanto, sempre permitir
        
        return True, "ok"
    
    def _log_mm_cycle(self, cycle: int, market_id: str, market_state: MarketState):
        """
        Log detalhado de cada ciclo do market maker.
        NÍVEL 2: Log por ciclo do Market Maker.
        """
        mid = market_state.mid_price
        vol_rec = self.volatility.get_recommendations()
        vol = vol_rec.volatility_score if hasattr(vol_rec, 'volatility_score') else 0.0
        vol_regime = vol_rec.regime if hasattr(vol_rec, 'regime') else "unknown"
        spread = self.base_spread * vol_rec.spread_multiplier
        bid = market_state.best_bid
        ask = market_state.best_ask
        delta = self.hedger.get_delta(market_id)
        inventory = self.inventory_manager.get_inventory_value(market_id, current_price=mid)
        
        logger.debug(
            f"[MM CYCLE #{cycle}] "
            f"mid={mid:.4f} | "
            f"vol={vol:.4f} ({vol_regime}) | "
            f"base_spread={self.base_spread*100:.2f}% | "
            f"adjusted_spread={spread*100:.2f}% | "
            f"bid={bid:.4f} x {self.order_size} | "
            f"ask={ask:.4f} x {self.order_size} | "
            f"delta={delta:.2f} | "
            f"inventory=${inventory:.2f}"
        )
        
        # Log estruturado
        structured_logger.log_quote(
            mid=mid,
            bid=bid,
            ask=ask,
            spread=spread,
            volatility=vol,
            delta=delta,
            inventory=inventory,
            cycle=cycle,
            volatility_regime=vol_regime
        )

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
        time_since_update = time.time() - state.last_update
        if time_since_update > 5.0:
            # NÍVEL 4: Log de requote
            old_mid = state.mid_price
            new_mid = (new_bid.price + new_ask.price) / 2
            price_move = abs(new_mid - old_mid) / old_mid if old_mid > 0 else 0
            
            logger.info(
                f"[REQUOTE] price_move={price_move*100:.2f}% "
                f"> threshold={self.requote_threshold*100:.2f}% | "
                f"old_price={old_mid:.4f} | "
                f"new_price={new_mid:.4f} | "
                f"market_id={market_id}"
            )
            
            structured_logger.log_requote(
                price_move=price_move,
                threshold=self.requote_threshold,
                old_price=old_mid,
                new_price=new_mid,
                market_id=market_id
            )
            
            return True

        return False

    async def send_quotes(self, market_id: str):
        """Envia quotes (bid e ask) para o mercado."""
        state = self.markets[market_id]

        # NÍVEL 3: Verificar se deve cotar (com log de skip)
        should_quote, reason = self._should_quote(market_id)
        if not should_quote:
            delta = self.hedger.get_delta(market_id)
            state = self.markets.get(market_id)
            mid_price = state.mid_price if state else 0.5
            inventory = self.inventory_manager.get_inventory_value(market_id, current_price=mid_price)
            vol_rec = self.volatility.get_recommendations()
            vol = vol_rec.volatility_score if hasattr(vol_rec, 'volatility_score') else 0.0
            
            logger.debug(
                f"[SKIP] reason={reason} | "
                f"delta={delta:.2f} | "
                f"inventory=${inventory:.2f} | "
                f"vol={vol:.4f}"
            )
            
            structured_logger.log_skip(
                reason=reason,
                delta=delta,
                inventory=inventory,
                volatility=vol
            )
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

        # Extrair campos com validacao (suporta dict ou objeto)
        market_id = getattr(order, 'market_id', None) or (order.get('market_id') if isinstance(order, dict) else None)
        side = getattr(order, 'side', None) or (order.get('side') if isinstance(order, dict) else 'BUY')
        price = getattr(order, 'price', None) or (order.get('price') if isinstance(order, dict) else 0.5)
        filled_size = getattr(order, 'filled_size', None) or (order.get('filled_size') if isinstance(order, dict) else 0)
        order_id = getattr(order, 'order_id', None) or (order.get('orderID') if isinstance(order, dict) else str(order))

        if not market_id:
            logger.error(f"[FILL] Ordem sem market_id: {order}")
            return

        # Atualizar posição no hedger
        self.hedger.update_position(
            market_id=market_id,
            token="YES",  # Simplificado
            size_delta=filled_size if side == "BUY" else -filled_size,
            price=price,
        )

        # Atualizar inventory manager
        self.inventory_manager.update_on_fill(
            market_id=market_id,
            side=side,
            price=price,
            size=filled_size,
            order_id=order_id,
        )

        # Log
        delta = self.hedger.get_delta(market_id)
        inventory = self.inventory_manager.get_inventory_value(
            market_id,
            current_price=price
        )
        logger.info(
            f"[FILL] side={side} | price={price:.4f} | "
            f"size={filled_size:.0f} | order_id={order_id} | "
            f"market_id={market_id} | delta={delta:.2f} | inventory=${inventory:.2f}"
        )
        
        # Log estruturado
        structured_logger.log_fill(
            side=side,
            price=price,
            size=filled_size,
            order_id=order_id,
            market_id=market_id
        )

        # Fast log para backtest
        self.fast_logger.log_trade(
            market=market_id,
            price_up=price if side == "BUY" else 0,
            price_down=0 if side == "BUY" else price,
            pair_cost=price,
            profit_pct=0,  # Calcular depois
            order_size=filled_size,
            investment=price * filled_size,
            expected_profit=0,
            balance_after=0,
            ltm_bucket=None,
        )

        # Verificar se precisa hedge urgente
        if self.hedger.needs_urgent_hedge(market_id):
            hedge_order = self.hedger.get_hedge_order(market_id)
            logger.warning(f"URGENT HEDGE NEEDED: {hedge_order}")

    async def run_once(self):
        """Executa um ciclo de market making para todos os mercados."""
        self.cycle_count += 1
        
        if not self.markets:
            logger.warning("No markets configured. Waiting...")
            await asyncio.sleep(5.0)
            # Tentar redescobrir mercados
            self.discover_and_add_markets(assets=['btc', 'eth', 'sol'])
            return
        
        for market_id in list(self.markets.keys()):  # Use list() to avoid dict size change during iteration
            try:
                # Atualizar orderbook
                await self.fetch_orderbook(market_id)
                
                state = self.markets[market_id]
                
                # NÍVEL 2: Log por ciclo do Market Maker
                self._log_mm_cycle(self.cycle_count, market_id, state)

                # Verificar se deve re-quotar
                if self.should_requote(market_id):
                    await self.send_quotes(market_id)

                # Pequeno delay entre mercados
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing market {market_id}: {e}")
                continue

    async def run(self, interval_seconds: float = 1.0):
        """Loop principal do market maker."""
        self.running = True

        # Registrar callback de fill
        self.order_manager.on_fill = self.on_fill

        logger.info("Market Maker started")

        try:
            while self.running:
                await self.run_once()
                
                # Snapshot periódico de inventory (a cada 1 segundo)
                current_prices = {
                    m: state.mid_price 
                    for m, state in self.markets.items()
                }
                snapshot = self.inventory_manager.get_snapshot(current_prices)
                self.inventory_manager.save_snapshot(snapshot)
                
                await asyncio.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Stopping Market Maker...")

        finally:
            # Cancelar todas as ordens ao parar
            await self.order_manager.cancel_all()
            
            # Exportar timeline de inventory antes de parar
            try:
                timeline_file = self.inventory_manager.export_timeline()
                logger.info(f"Inventory timeline exported to {timeline_file}")
            except Exception as e:
                logger.warning(f"Error exporting inventory timeline: {e}")
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
    import sys
    from .logger import setup_logging, print_header

    settings = load_settings()
    setup_logging(verbose=settings.verbose, use_rich=settings.use_rich_output)

    print_header("Market Maker Bot (Bot 2) - Multi-Market")

    # Verificar modo de operacao
    allow_live = "--allow-live" in sys.argv

    if not settings.dry_run:
        if not allow_live:
            logger.warning("=" * 60)
            logger.warning("  ATENCAO: Modo LIVE detectado!")
            logger.warning("  Para operar com capital real, use: --allow-live")
            logger.warning("  Revertendo para modo PAPER por seguranca.")
            logger.warning("=" * 60)
            settings.dry_run = True
        else:
            logger.warning("=" * 60)
            logger.warning("  MODO LIVE ATIVADO - CAPITAL REAL!")
            logger.warning("  Certifique-se de ter configurado:")
            logger.warning("  - POLYMARKET_PRIVATE_KEY")
            logger.warning("  - MAX_DAILY_LOSS")
            logger.warning("  - MAX_POSITION_SIZE")
            logger.warning("=" * 60)
            # Confirmacao de seguranca
            logger.info("Iniciando em modo LIVE em 5 segundos... (Ctrl+C para cancelar)")
            import asyncio
            await asyncio.sleep(5)

    bot = MarketMakerBot(settings)

    # Descobrir e adicionar mercados ativos (BTC, ETH, SOL)
    logger.info("=" * 60)
    logger.info("  MARKET DISCOVERY")
    logger.info("=" * 60)
    bot.discover_and_add_markets(assets=['btc', 'eth', 'sol'])
    
    if not bot.markets:
        logger.error("❌ No markets found! Exiting.")
        return
    
    logger.info("=" * 60)
    logger.info(f"  OPERATING ON {len(bot.markets)} MARKETS")
    logger.info("=" * 60)
    for market_id, state in bot.markets.items():
        logger.info(f"  - {market_id}")
    logger.info("=" * 60)
    logger.info("")

    # Rodar
    await bot.run(interval_seconds=1.0)


if __name__ == "__main__":
    asyncio.run(main())
