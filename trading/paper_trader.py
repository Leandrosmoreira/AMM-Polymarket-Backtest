"""
PaperTrader - Sistema de paper trading em tempo real
Simula execução usando dados reais do Polymarket
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from core.types import (
    PaperPosition, PaperTrade, PaperStats, MarketInfo,
    BookSnapshot, TokenType, Side, Quote
)
from core.buffers import MarketDataState
from config.gabagool_config import GabagoolConfig, get_config
from agents.microstructure import MicrostructureAgent
from agents.edge import EdgeAgent
from agents.risk import RiskAgent
from agents.market_making import MarketMakingAgent
from net.websocket import PolymarketWebSocket
from net.binance_ws import BinanceWebSocket, BinancePriceAgent
from net.http_client import PolymarketHTTP

logger = logging.getLogger(__name__)


class PaperTrader:
    """
    Paper Trader - Sistema completo de paper trading.

    Componentes:
    - WebSocket para dados real-time (Polymarket + Binance)
    - Pipeline de agentes (Microstructure -> Edge -> Risk -> MarketMaking)
    - Simulação de execução
    - Tracking de estatísticas

    Fluxo:
    1. Descobre mercado ativo (BTC/SOL 15-min)
    2. Conecta WebSockets
    3. Loop de trading a cada tick
    4. Monitora expiração e troca de mercado
    """

    __slots__ = (
        'config',
        '_running',
        '_market_data',
        '_current_market',
        '_position',
        '_stats',
        '_trades',
        # Agents
        '_micro_agent',
        '_edge_agent',
        '_risk_agent',
        '_mm_agent',
        # Network
        '_poly_ws',
        '_binance_ws',
        '_http',
        # Timing
        '_start_time',
        '_last_status_log',
        '_market_check_time'
    )

    def __init__(self, config: Optional[GabagoolConfig] = None):
        """
        Inicializa Paper Trader.

        Args:
            config: Configuração (usa padrão se None)
        """
        self.config = config or get_config()

        self._running = False
        self._market_data = MarketDataState()
        self._current_market: Optional[MarketInfo] = None
        self._position: Optional[PaperPosition] = None
        self._stats = PaperStats()
        self._trades: List[PaperTrade] = []

        # Initialize agents
        self._micro_agent = MicrostructureAgent(self.config)
        self._edge_agent = EdgeAgent(self.config)
        self._risk_agent = RiskAgent(self.config)
        self._mm_agent = MarketMakingAgent(self.config)

        # Network (initialized on run)
        self._poly_ws: Optional[PolymarketWebSocket] = None
        self._binance_ws: Optional[BinanceWebSocket] = None
        self._http: Optional[PolymarketHTTP] = None

        # Timing
        self._start_time: Optional[datetime] = None
        self._last_status_log: Optional[datetime] = None
        self._market_check_time: Optional[datetime] = None

    async def run(self) -> PaperStats:
        """
        Executa paper trading.

        Returns:
            PaperStats com estatísticas da sessão
        """
        logger.info("🚀 Starting Paper Trader")
        logger.info(f"Config: dry_run={self.config.dry_run}, initial_bankroll=${self.config.initial_bankroll}")

        self._running = True
        self._start_time = datetime.now()

        try:
            # Initialize connections
            await self._initialize()

            # Main trading loop
            await self._trading_loop()

        except asyncio.CancelledError:
            logger.info("Paper Trader cancelled")

        except Exception as e:
            logger.exception(f"Paper Trader error: {e}")

        finally:
            await self._shutdown()

        return self._stats

    async def _initialize(self) -> None:
        """Inicializa conexões e componentes."""
        logger.info("🔧 Initializing Paper Trader...")

        # HTTP client for market discovery
        self._http = PolymarketHTTP(self.config)
        await self._http.connect()

        # Find initial market
        await self._find_and_setup_market()

        if self._current_market is None:
            raise RuntimeError("No active market found")

        # Initialize WebSockets
        self._poly_ws = PolymarketWebSocket(self.config)
        self._poly_ws.set_book_callback(self._on_book_update)
        self._poly_ws.set_trade_callback(self._on_trade)

        self._binance_ws = BinanceWebSocket(self.config)

        # Connect WebSockets
        await self._poly_ws.connect()
        await self._binance_ws.connect()

        # Subscribe to market tokens
        await self._poly_ws.subscribe([
            self._current_market.yes_token_id,
            self._current_market.no_token_id
        ])

        logger.info("✅ Paper Trader initialized")

    async def _find_and_setup_market(self) -> None:
        """Encontra mercado ativo e configura."""
        # Try BTC first, then SOL
        market = await self._http.get_active_btc_market()

        if market is None:
            market = await self._http.get_active_sol_market()

        if market is None:
            logger.error("No active market found!")
            return

        self._current_market = market
        self._position = PaperPosition(market_id=market.market_id)
        self._market_data.clear_all()

        logger.info(f"📊 Market: {market.question}")
        logger.info(f"   YES token: {market.yes_token_id[:16]}...")
        logger.info(f"   NO token: {market.no_token_id[:16]}...")
        logger.info(f"   Expires: {market.end_time}")

    async def _trading_loop(self) -> None:
        """Loop principal de trading."""
        logger.info("🔄 Starting trading loop...")

        # Start WebSocket receive tasks
        tasks = [
            asyncio.create_task(self._poly_ws.receive_loop()),
            asyncio.create_task(self._binance_ws.receive_loop()),
            asyncio.create_task(self._decision_loop()),
            asyncio.create_task(self._market_monitor_loop()),
            asyncio.create_task(self._status_logger_loop()),
        ]

        try:
            # Wait for any task to complete (usually due to cancellation)
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()

        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()

    async def _decision_loop(self) -> None:
        """Loop de decisão de trading."""
        tick_interval = self.config.tick_interval_ms / 1000.0

        while self._running:
            try:
                await self._analyze_and_trade()

            except Exception as e:
                logger.error(f"Decision loop error: {e}")

            await asyncio.sleep(tick_interval)

    async def _analyze_and_trade(self) -> None:
        """Analisa mercado e executa trades se apropriado."""
        if self._current_market is None:
            return

        # Check if market expired
        if self._current_market.is_expired():
            await self._settle_position()
            return

        # 1. Microstructure analysis
        micro_decision = self._micro_agent.analyze(self._market_data)

        if not micro_decision.should_trade:
            self._stats.blocked_by_microstructure += 1
            return

        # 2. Edge detection
        edge_decision = self._edge_agent.analyze(self._market_data)

        if not edge_decision.should_trade:
            self._stats.blocked_by_edge += 1
            return

        # 3. Risk check
        max_size = self._risk_agent.get_max_order_size()
        risk_decision = self._risk_agent.analyze(
            self._position,
            self._stats,
            proposed_size=max_size
        )

        if not risk_decision.should_trade:
            self._stats.blocked_by_risk += 1
            return

        # 4. Generate quotes
        mm_decision = self._mm_agent.analyze(
            self._market_data,
            self._position,
            edge_strength=edge_decision.confidence,
            max_size=max_size
        )

        if not mm_decision.should_trade:
            self._stats.blocked_by_market_making += 1
            return

        # 5. Execute paper trades
        quotes = mm_decision.data.get("quotes", [])
        for quote_data in quotes:
            await self._execute_paper_trade(quote_data)

    async def _execute_paper_trade(self, quote_data: Dict[str, Any]) -> None:
        """Executa trade simulado."""
        token_type = TokenType(quote_data["token_type"])
        price = quote_data["price"]
        size = quote_data["size"]
        notional = quote_data["notional"]

        # Record trade
        trade = PaperTrade(
            market_id=self._current_market.market_id,
            token_type=token_type,
            side=Side.BUY,
            price=price,
            size=size
        )
        self._trades.append(trade)

        # Update position
        if token_type == TokenType.YES:
            self._position.add_yes(size, price)
            self._stats.total_yes_bought += size
        else:
            self._position.add_no(size, price)
            self._stats.total_no_bought += size

        self._stats.total_trades += 1
        self._stats.total_cost += notional

        # Update pair cost stats
        if self._position.pair_cost > 0:
            self._stats.avg_pair_cost = (
                self._stats.avg_pair_cost * (self._stats.total_trades - 1) +
                self._position.pair_cost
            ) / self._stats.total_trades

            self._stats.best_pair_cost = min(
                self._stats.best_pair_cost,
                self._position.pair_cost
            )
            self._stats.worst_pair_cost = max(
                self._stats.worst_pair_cost,
                self._position.pair_cost
            )

        if self.config.log_trades:
            logger.info(
                f"📝 TRADE: {token_type.value} {size:.4f} @ ${price:.4f} "
                f"(${notional:.2f}) | Pair cost: {self._position.pair_cost:.4f}"
            )

    async def _settle_position(self) -> None:
        """Liquida posição no final do mercado."""
        if self._position is None:
            return

        if self._position.yes_qty == 0 and self._position.no_qty == 0:
            return

        # Calculate payout based on hedge
        hedge_qty = self._position.hedge_qty
        payout = hedge_qty * 1.0  # Hedged pairs always pay $1

        # Unhedged portion depends on outcome (simulate 50/50 for paper)
        unhedged_yes = self._position.unhedged_yes
        unhedged_no = self._position.unhedged_no

        # Assume 50% win rate for unhedged
        unhedged_payout = (unhedged_yes + unhedged_no) * 0.5

        total_payout = payout + unhedged_payout
        pnl = total_payout - self._position.total_cost

        self._stats.total_payout += total_payout

        if pnl > 0:
            self._stats.wins += 1
        else:
            self._stats.losses += 1

        self._stats.markets_traded += 1
        self._risk_agent.record_trade_result(pnl)

        logger.info(
            f"💰 SETTLED: Cost=${self._position.total_cost:.2f}, "
            f"Payout=${total_payout:.2f}, PnL=${pnl:.2f}"
        )

        # Reset position
        self._position.reset()

    async def _market_monitor_loop(self) -> None:
        """Monitora mercado e troca quando expira."""
        while self._running:
            try:
                if self._current_market:
                    remaining = self._current_market.time_remaining()

                    # If market about to expire, prepare for switch
                    if remaining < 60:  # Less than 1 minute
                        logger.info(f"⏰ Market expiring in {remaining:.0f}s")

                    # If market expired, settle and find new
                    if remaining <= 0:
                        await self._settle_position()
                        await self._switch_to_next_market()

            except Exception as e:
                logger.error(f"Market monitor error: {e}")

            await asyncio.sleep(self.config.market_refresh_seconds)

    async def _switch_to_next_market(self) -> None:
        """Troca para o próximo mercado ativo."""
        logger.info("🔄 Switching to next market...")

        # Unsubscribe from current
        if self._poly_ws and self._current_market:
            await self._poly_ws.unsubscribe([
                self._current_market.yes_token_id,
                self._current_market.no_token_id
            ])

        # Find next market
        await self._find_and_setup_market()

        if self._current_market and self._poly_ws:
            await self._poly_ws.subscribe([
                self._current_market.yes_token_id,
                self._current_market.no_token_id
            ])

    async def _status_logger_loop(self) -> None:
        """Log de status periódico."""
        interval = self.config.status_interval_seconds

        while self._running:
            await asyncio.sleep(interval)

            if not self._running:
                break

            self._log_status()

    def _log_status(self) -> None:
        """Log current status."""
        runtime = datetime.now() - self._start_time if self._start_time else timedelta()
        pair_cost = self._market_data.pair_cost

        logger.info(
            f"📊 STATUS | Runtime: {runtime} | "
            f"Trades: {self._stats.total_trades} | "
            f"PnL: ${self._stats.total_pnl:.2f} | "
            f"Pair Cost: {pair_cost:.4f if pair_cost else 'N/A'} | "
            f"Win Rate: {self._stats.win_rate*100:.1f}%"
        )

    async def _on_book_update(self, snapshot: BookSnapshot) -> None:
        """Callback para atualização de order book."""
        if self._current_market is None:
            return

        if snapshot.token_id == self._current_market.yes_token_id:
            self._market_data.yes_book.add(snapshot)
        elif snapshot.token_id == self._current_market.no_token_id:
            self._market_data.no_book.add(snapshot)

    async def _on_trade(self, trade: PaperTrade) -> None:
        """Callback para trades do mercado."""
        if self._current_market is None:
            return

        if trade.market_id == self._current_market.yes_token_id:
            trade.token_type = TokenType.YES
            self._market_data.yes_trades.add(trade)
        elif trade.market_id == self._current_market.no_token_id:
            trade.token_type = TokenType.NO
            self._market_data.no_trades.add(trade)

    async def _shutdown(self) -> None:
        """Encerra conexões."""
        self._running = False

        logger.info("🛑 Shutting down Paper Trader...")

        # Settle any open position
        if self._position and (self._position.yes_qty > 0 or self._position.no_qty > 0):
            await self._settle_position()

        # Close connections
        if self._poly_ws:
            await self._poly_ws.disconnect()

        if self._binance_ws:
            await self._binance_ws.disconnect()

        if self._http:
            await self._http.close()

        # Log final stats
        self._log_final_stats()

    def _log_final_stats(self) -> None:
        """Log estatísticas finais."""
        runtime = datetime.now() - self._start_time if self._start_time else timedelta()

        logger.info("=" * 60)
        logger.info("📈 FINAL STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Runtime: {runtime}")
        logger.info(f"Markets Traded: {self._stats.markets_traded}")
        logger.info(f"Total Trades: {self._stats.total_trades}")
        logger.info(f"Total Cost: ${self._stats.total_cost:.2f}")
        logger.info(f"Total Payout: ${self._stats.total_payout:.2f}")
        logger.info(f"Total PnL: ${self._stats.total_pnl:.2f}")
        logger.info(f"Win Rate: {self._stats.win_rate*100:.1f}%")
        logger.info(f"Avg Pair Cost: {self._stats.avg_pair_cost:.4f}")
        logger.info(f"Best Pair Cost: {self._stats.best_pair_cost:.4f}")
        logger.info(f"Worst Pair Cost: {self._stats.worst_pair_cost:.4f}")
        logger.info("-" * 60)
        logger.info(f"Blocked by Microstructure: {self._stats.blocked_by_microstructure}")
        logger.info(f"Blocked by Edge: {self._stats.blocked_by_edge}")
        logger.info(f"Blocked by Risk: {self._stats.blocked_by_risk}")
        logger.info(f"Blocked by MarketMaking: {self._stats.blocked_by_market_making}")
        logger.info("=" * 60)

    def stop(self) -> None:
        """Para o trader."""
        self._running = False

    @property
    def stats(self) -> PaperStats:
        return self._stats

    @property
    def position(self) -> Optional[PaperPosition]:
        return self._position

    @property
    def current_market(self) -> Optional[MarketInfo]:
        return self._current_market

    @property
    def is_running(self) -> bool:
        return self._running


async def run_paper_trader(config: Optional[GabagoolConfig] = None) -> PaperStats:
    """
    Função de conveniência para rodar o paper trader.

    Args:
        config: Configuração opcional

    Returns:
        Estatísticas da sessão
    """
    trader = PaperTrader(config)
    return await trader.run()


def main():
    """Entry point para CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trader for Polymarket")
    parser.add_argument("--bankroll", type=float, default=1000, help="Initial bankroll")
    parser.add_argument("--duration", type=int, default=3600, help="Duration in seconds")
    parser.add_argument("--log-level", default="INFO", help="Log level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = GabagoolConfig()
    config.initial_bankroll = args.bankroll

    # Run
    async def run_with_timeout():
        trader = PaperTrader(config)

        async def timeout_stop():
            await asyncio.sleep(args.duration)
            trader.stop()

        await asyncio.gather(
            trader.run(),
            timeout_stop()
        )

        return trader.stats

    stats = asyncio.run(run_with_timeout())
    print(f"\nFinal PnL: ${stats.total_pnl:.2f}")


if __name__ == "__main__":
    main()
