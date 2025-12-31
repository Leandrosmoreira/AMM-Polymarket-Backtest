"""
Gabagool Bot - Main Bot Implementation
Spread Capture Strategy for Polymarket Up/Down Markets
"""

import asyncio
import signal
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from .config import GabagoolConfig
from .polymarket_client import PolymarketClient, OrderSide
from .spread_monitor import SpreadMonitor, SpreadOpportunity
from .position_manager import GabagoolPositionManager, PositionSide

logger = logging.getLogger(__name__)


class GabagoolBot:
    """
    Main bot implementing gabagool's spread capture strategy.

    Strategy:
    1. Monitor markets for spread opportunities (UP + DOWN < $1)
    2. When opportunity found, buy both UP and DOWN
    3. Maintain balance between UP and DOWN shares
    4. Collect guaranteed profit at settlement
    """

    def __init__(
        self,
        config: GabagoolConfig = None,
        log_dir: str = "data/trades",
    ):
        self.config = config or GabagoolConfig()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate config
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        # Components
        self.client = PolymarketClient(
            api_key=self.config.API_KEY,
            api_secret=self.config.API_SECRET,
            api_passphrase=self.config.API_PASSPHRASE,
            private_key=self.config.PRIVATE_KEY,
        )
        self.monitor = SpreadMonitor(
            client=self.client,
            config=self.config,
            on_opportunity=self._on_opportunity,
        )
        self.positions = GabagoolPositionManager(config=self.config)

        # State
        self._running = False
        self._processing_lock = asyncio.Lock()
        self._pending_opportunities: asyncio.Queue = asyncio.Queue()

        # Statistics
        self.start_time: Optional[int] = None
        self.trades_executed = 0
        self.opportunities_seen = 0
        self.opportunities_taken = 0
        self.errors = 0

        # Trade log
        self._trade_log: List[Dict[str, Any]] = []

    async def start(self):
        """Start the bot."""
        logger.info("=" * 60)
        logger.info("GABAGOOL BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode: {'PAPER TRADING' if self.config.PAPER_TRADING else 'LIVE'}")
        logger.info(f"Min Spread: {self.config.MIN_SPREAD * 100:.1f}%")
        logger.info(f"Order Size: ${self.config.ORDER_SIZE_USD}")
        logger.info(f"Max Per Market: ${self.config.MAX_PER_MARKET}")
        logger.info(f"Assets: {', '.join(self.config.ENABLED_ASSETS)}")
        logger.info("=" * 60)

        self._running = True
        self.start_time = int(time.time() * 1000)

        # Connect to API
        await self.client.connect()

        # Check API health
        if not await self.client.check_health():
            logger.error("API health check failed")
            return

        # Start monitor
        await self.monitor.start()

        # Start opportunity processor
        processor_task = asyncio.create_task(self._process_opportunities())

        # Start status reporter
        status_task = asyncio.create_task(self._status_loop())

        try:
            await asyncio.gather(processor_task, status_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping bot...")
        self._running = False

        # Stop monitor
        await self.monitor.stop()

        # Close client
        await self.client.close()

        # Save final state
        await self._save_state()

        # Print summary
        self._print_summary()

        logger.info("Bot stopped")

    def _on_opportunity(self, opportunity: SpreadOpportunity):
        """Callback when spread opportunity is detected."""
        self.opportunities_seen += 1

        # Add to processing queue
        try:
            self._pending_opportunities.put_nowait(opportunity)
        except asyncio.QueueFull:
            logger.warning("Opportunity queue full, skipping")

    async def _process_opportunities(self):
        """Process opportunities from queue."""
        while self._running:
            try:
                # Get opportunity with timeout
                try:
                    opportunity = await asyncio.wait_for(
                        self._pending_opportunities.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process with lock to prevent concurrent execution
                async with self._processing_lock:
                    await self._execute_opportunity(opportunity)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing opportunity: {e}")
                self.errors += 1

    async def _execute_opportunity(self, opportunity: SpreadOpportunity):
        """Execute a spread capture opportunity."""
        market_id = opportunity.market.id

        # Check if we can trade
        can_trade, reason = self.positions.can_trade(
            market_id,
            self.config.ORDER_SIZE_USD * 2  # Both sides
        )

        if not can_trade:
            logger.debug(f"Cannot trade: {reason}")
            return

        # Get or create position
        position = self.positions.get_or_create_position(
            market_id=market_id,
            market_slug=opportunity.market.slug,
            up_token_id=opportunity.up_token_id,
            down_token_id=opportunity.down_token_id,
        )

        # Calculate what to buy
        rebalance = self.positions.calculate_rebalance(
            position=position,
            budget=self.config.ORDER_SIZE_USD,
            up_price=opportunity.up_price,
            down_price=opportunity.down_price,
        )

        buy_up = rebalance['buy_up']
        buy_down = rebalance['buy_down']

        if buy_up <= 0 and buy_down <= 0:
            return

        logger.info(
            f"Executing: {opportunity.market.slug} | "
            f"Spread: {opportunity.spread_pct*100:.2f}% | "
            f"Buy UP: {buy_up:.2f} @ ${opportunity.up_price:.3f} | "
            f"Buy DOWN: {buy_down:.2f} @ ${opportunity.down_price:.3f}"
        )

        self.opportunities_taken += 1

        # Execute trades
        if self.config.PAPER_TRADING or self.config.DRY_RUN:
            # Paper trading - simulate execution
            await self._paper_trade(
                position, opportunity, buy_up, buy_down
            )
        else:
            # Live trading
            await self._live_trade(
                position, opportunity, buy_up, buy_down
            )

    async def _paper_trade(
        self,
        position,
        opportunity: SpreadOpportunity,
        buy_up: float,
        buy_down: float,
    ):
        """Simulate trade execution for paper trading."""
        if buy_up > 0:
            self.positions.record_trade(
                market_id=position.market_id,
                side=PositionSide.UP,
                shares=buy_up,
                price=opportunity.up_price,
            )
            self.trades_executed += 1

        if buy_down > 0:
            self.positions.record_trade(
                market_id=position.market_id,
                side=PositionSide.DOWN,
                shares=buy_down,
                price=opportunity.down_price,
            )
            self.trades_executed += 1

        # Log trade
        self._log_trade(opportunity, buy_up, buy_down, "PAPER")

    async def _live_trade(
        self,
        position,
        opportunity: SpreadOpportunity,
        buy_up: float,
        buy_down: float,
    ):
        """Execute live trades."""
        # Execute UP order
        if buy_up > 0:
            try:
                order = await self.client.create_order(
                    token_id=opportunity.up_token_id,
                    side=OrderSide.BUY,
                    price=opportunity.up_price,
                    size=buy_up,
                )

                if order:
                    self.positions.record_trade(
                        market_id=position.market_id,
                        side=PositionSide.UP,
                        shares=buy_up,
                        price=opportunity.up_price,
                        order_id=order.id,
                    )
                    self.trades_executed += 1

            except Exception as e:
                logger.error(f"Error creating UP order: {e}")
                self.errors += 1

        # Execute DOWN order
        if buy_down > 0:
            try:
                order = await self.client.create_order(
                    token_id=opportunity.down_token_id,
                    side=OrderSide.BUY,
                    price=opportunity.down_price,
                    size=buy_down,
                )

                if order:
                    self.positions.record_trade(
                        market_id=position.market_id,
                        side=PositionSide.DOWN,
                        shares=buy_down,
                        price=opportunity.down_price,
                        order_id=order.id,
                    )
                    self.trades_executed += 1

            except Exception as e:
                logger.error(f"Error creating DOWN order: {e}")
                self.errors += 1

        # Log trade
        self._log_trade(opportunity, buy_up, buy_down, "LIVE")

    def _log_trade(
        self,
        opportunity: SpreadOpportunity,
        buy_up: float,
        buy_down: float,
        mode: str,
    ):
        """Log trade for analysis."""
        trade_record = {
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.now().isoformat(),
            'mode': mode,
            'market_id': opportunity.market.id,
            'market_slug': opportunity.market.slug,
            'spread_pct': opportunity.spread_pct,
            'up_price': opportunity.up_price,
            'down_price': opportunity.down_price,
            'total_price': opportunity.total_price,
            'buy_up': buy_up,
            'buy_down': buy_down,
            'cost_up': buy_up * opportunity.up_price,
            'cost_down': buy_down * opportunity.down_price,
            'total_cost': (buy_up * opportunity.up_price) + (buy_down * opportunity.down_price),
            'expected_profit': min(buy_up, buy_down) - ((buy_up * opportunity.up_price) + (buy_down * opportunity.down_price)),
        }

        self._trade_log.append(trade_record)

    async def _status_loop(self):
        """Periodically print status."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute
                self._print_status()

            except asyncio.CancelledError:
                break

    def _print_status(self):
        """Print current status."""
        runtime = (int(time.time() * 1000) - self.start_time) / 1000 / 60  # minutes

        summary = self.positions.get_summary()
        monitor_stats = self.monitor.get_stats()

        logger.info("-" * 50)
        logger.info(f"STATUS UPDATE (Runtime: {runtime:.1f} min)")
        logger.info(f"Opportunities: {self.opportunities_seen} seen, {self.opportunities_taken} taken")
        logger.info(f"Trades: {self.trades_executed}")
        logger.info(f"Active Positions: {summary['active_positions']}")
        logger.info(f"Total Exposure: ${summary['total_exposure']:.2f}")
        logger.info(f"Expected Profit: ${summary['total_expected_profit']:.2f}")
        logger.info(f"Realized Profit: ${summary['total_realized_profit']:.2f}")
        logger.info(f"Errors: {self.errors}")
        logger.info("-" * 50)

    def _print_summary(self):
        """Print final summary."""
        runtime = (int(time.time() * 1000) - self.start_time) / 1000 / 60 if self.start_time else 0
        summary = self.positions.get_summary()

        print("\n" + "=" * 60)
        print("GABAGOOL BOT - FINAL SUMMARY")
        print("=" * 60)
        print(f"Runtime: {runtime:.1f} minutes")
        print(f"Mode: {'PAPER' if self.config.PAPER_TRADING else 'LIVE'}")
        print()
        print("--- Trading Activity ---")
        print(f"Opportunities Seen: {self.opportunities_seen}")
        print(f"Opportunities Taken: {self.opportunities_taken}")
        print(f"Trades Executed: {self.trades_executed}")
        print()
        print("--- Positions ---")
        print(f"Active Positions: {summary['active_positions']}")
        print(f"Settled Positions: {summary['settled_positions']}")
        print()
        print("--- P&L ---")
        print(f"Total Exposure: ${summary['total_exposure']:.2f}")
        print(f"Expected Profit: ${summary['total_expected_profit']:.2f}")
        print(f"Realized Profit: ${summary['total_realized_profit']:.2f}")
        print()
        print(f"Errors: {self.errors}")
        print("=" * 60)

    async def _save_state(self):
        """Save current state to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.log_dir / f"gabagool_session_{timestamp}.json"

        state = {
            'config': self.config.to_dict(),
            'stats': {
                'start_time': self.start_time,
                'opportunities_seen': self.opportunities_seen,
                'opportunities_taken': self.opportunities_taken,
                'trades_executed': self.trades_executed,
                'errors': self.errors,
            },
            'positions': self.positions.get_summary(),
            'trades': self._trade_log,
        }

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved to {filename}")


async def run_bot(config: GabagoolConfig = None):
    """Run the bot with signal handling."""
    bot = GabagoolBot(config=config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await bot.start()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Gabagool Spread Capture Bot')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--live', action='store_true', help='Live trading mode')
    parser.add_argument('--min-spread', type=float, default=0.02, help='Minimum spread (0.02 = 2%)')
    parser.add_argument('--order-size', type=float, default=15.0, help='Order size in USD')
    parser.add_argument('--max-per-market', type=float, default=500.0, help='Max USD per market')
    parser.add_argument('--assets', nargs='+', default=['BTC', 'ETH'], help='Assets to trade')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = GabagoolConfig(
        PAPER_TRADING=not args.live,
        MIN_SPREAD=args.min_spread,
        ORDER_SIZE_USD=args.order_size,
        MAX_PER_MARKET=args.max_per_market,
        ENABLED_ASSETS=args.assets,
    )

    # Run bot
    asyncio.run(run_bot(config))


if __name__ == '__main__':
    main()
