"""
Volatility Arbitrage Bot for BTC 15-minute Markets

Main bot that integrates all components:
- Volatility calculation
- Probability estimation
- Edge detection
- Risk management
- Trade execution
- Logging
"""

import asyncio
import signal
import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

import httpx

from .volatility import VolatilityCalculator, VolatilityMetrics
from .probability import ProbabilityModel, AdaptiveProbabilityModel
from .edge_detector import EdgeDetector, EdgeOpportunity, MarketPrices, TradeSignal
from .risk_manager import RiskManager, RiskConfig, CONSERVATIVE_RISK, MODERATE_RISK, AGGRESSIVE_RISK
from .executor import BaseExecutor, PaperExecutor, ExecutionMode, create_executor
from .logger import TradingLogger, TradeLog, SettlementLog, MarketAnalysisLog, MetricsCollector


logger = logging.getLogger(__name__)


# API Endpoints
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
CHAINLINK_BTC_USD_PROXY = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
POLYGON_RPC = "https://polygon-rpc.com"


@dataclass
class BotConfig:
    """Configuration for the volatility arbitrage bot."""

    # Execution mode
    mode: ExecutionMode = ExecutionMode.PAPER
    initial_balance: float = 1000.0

    # API credentials (for live trading)
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    funder: str = ""

    # Timing
    tick_interval: float = 1.0  # Seconds between price checks
    analysis_interval: float = 2.0  # Seconds between full analysis
    status_interval: float = 60.0  # Seconds between status logs

    # Model parameters
    volatility_window: int = 300  # Seconds for volatility calculation
    min_volatility_samples: int = 30
    use_momentum: bool = True

    # Edge detection
    min_edge_percent: float = 3.0
    min_confidence: float = 0.5
    confirmation_count: int = 2

    # Risk preset
    risk_level: str = "moderate"  # conservative, moderate, aggressive


class VolatilityArbBot:
    """
    Main volatility arbitrage bot.

    Flow:
    1. Stream BTC prices from Chainlink
    2. Calculate rolling volatility
    3. Get market prices from Polymarket
    4. Estimate true probabilities using volatility model
    5. Detect edges (model prob vs market price)
    6. Execute trades when edge is sufficient
    7. Track and settle positions at market expiry
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False

        # Components
        self.volatility = VolatilityCalculator(
            window_seconds=config.volatility_window,
            min_samples=config.min_volatility_samples
        )

        if config.use_momentum:
            self.probability = AdaptiveProbabilityModel()
        else:
            self.probability = ProbabilityModel()

        # Risk config based on level
        risk_configs = {
            "conservative": CONSERVATIVE_RISK,
            "moderate": MODERATE_RISK,
            "aggressive": AGGRESSIVE_RISK
        }
        risk_config = risk_configs.get(config.risk_level, MODERATE_RISK)
        risk_config.min_edge_percent = config.min_edge_percent

        self.edge_detector = EdgeDetector(
            min_edge_percent=config.min_edge_percent,
            min_confidence=config.min_confidence,
            confirmation_count=config.confirmation_count
        )

        self.risk_manager = RiskManager(
            config=risk_config,
            initial_balance=config.initial_balance
        )

        self.executor = create_executor(
            mode=config.mode,
            initial_balance=config.initial_balance,
            api_key=config.api_key,
            api_secret=config.api_secret,
            passphrase=config.passphrase,
            funder=config.funder
        )

        self.logger = TradingLogger()
        self.metrics = MetricsCollector()

        # HTTP client
        self.client = httpx.AsyncClient(timeout=10.0)

        # Current market state
        self.current_market: Optional[Dict] = None
        self.up_token_id: Optional[str] = None
        self.down_token_id: Optional[str] = None
        self.strike_price: Optional[float] = None
        self.market_expiry: Optional[int] = None

        # Last known values
        self.last_btc_price: Optional[float] = None
        self.last_volatility: Optional[VolatilityMetrics] = None
        self.last_market_prices: Optional[MarketPrices] = None

        # Tasks
        self._tasks = []

    async def start(self):
        """Start the bot."""
        logger.info("="*60)
        logger.info("Starting Volatility Arbitrage Bot")
        logger.info(f"Mode: {self.config.mode.value}")
        logger.info(f"Initial Balance: ${self.config.initial_balance}")
        logger.info(f"Risk Level: {self.config.risk_level}")
        logger.info(f"Min Edge: {self.config.min_edge_percent}%")
        logger.info("="*60)

        # Find initial market
        await self._find_market()

        self.running = True

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._price_loop()),
            asyncio.create_task(self._analysis_loop()),
            asyncio.create_task(self._status_loop()),
            asyncio.create_task(self._market_refresh_loop()),
            asyncio.create_task(self._settlement_loop()),
        ]

        logger.info("Bot started - monitoring for opportunities...")

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")

    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Final logging
        self.logger.print_summary()
        self.logger.flush()

        # Close client
        await self.client.aclose()

        logger.info("Bot stopped")

    async def _find_market(self):
        """Find current BTC 15min market."""
        try:
            # Search for BTC markets
            response = await self.client.get(
                f"{POLYMARKET_GAMMA_API}/markets",
                params={"slug_contains": "btc", "closed": "false", "limit": 50}
            )

            if response.status_code != 200:
                logger.warning("Failed to fetch markets")
                return

            markets = response.json()

            # Find BTC 15min market
            for market in markets:
                question = market.get('question', '').lower()
                slug = market.get('slug', '').lower()

                is_btc = 'btc' in question or 'bitcoin' in question
                is_15min = '15' in question or '15min' in slug
                is_price = 'up' in question or 'down' in question

                if is_btc and is_15min and is_price:
                    self.current_market = market
                    break

            if not self.current_market:
                # Fallback: any BTC price market
                for market in markets:
                    question = market.get('question', '').lower()
                    if 'btc' in question and ('up' in question or 'down' in question):
                        self.current_market = market
                        break

            if self.current_market:
                await self._parse_market_tokens()
                logger.info(f"Found market: {self.current_market.get('slug')}")
            else:
                logger.warning("No suitable BTC market found")

        except Exception as e:
            logger.error(f"Error finding market: {e}")

    async def _parse_market_tokens(self):
        """Parse token IDs from market data."""
        if not self.current_market:
            return

        import json

        # Try tokens field
        tokens = self.current_market.get('tokens', [])
        if isinstance(tokens, str):
            try:
                tokens = json.loads(tokens)
            except:
                tokens = []

        for token in tokens:
            if isinstance(token, dict):
                outcome = str(token.get('outcome', '')).lower()
                token_id = token.get('token_id')

                if token_id:
                    if 'up' in outcome or 'yes' in outcome:
                        self.up_token_id = str(token_id)
                    elif 'down' in outcome or 'no' in outcome:
                        self.down_token_id = str(token_id)

        # Fallback: clobTokenIds
        if not self.up_token_id:
            clob_tokens = self.current_market.get('clobTokenIds', [])
            if isinstance(clob_tokens, str):
                try:
                    clob_tokens = json.loads(clob_tokens)
                except:
                    clob_tokens = []

            if len(clob_tokens) >= 2:
                self.up_token_id = str(clob_tokens[0])
                self.down_token_id = str(clob_tokens[1])

        # Get market expiry
        end_date = self.current_market.get('endDate') or self.current_market.get('end_date_iso')
        if end_date:
            from datetime import datetime
            try:
                if isinstance(end_date, str):
                    dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    self.market_expiry = int(dt.timestamp() * 1000)
            except:
                pass

        if self.up_token_id:
            logger.info(f"UP Token: {self.up_token_id[:20]}...")
            logger.info(f"DOWN Token: {self.down_token_id[:20]}...")

    async def _get_btc_price(self) -> Optional[float]:
        """Get current BTC price from Chainlink."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{
                    "to": CHAINLINK_BTC_USD_PROXY,
                    "data": "0xfeaf968c"
                }, "latest"],
                "id": 1
            }

            response = await self.client.post(POLYGON_RPC, json=payload)
            result = response.json()

            if 'result' in result:
                data = result['result']
                if len(data) >= 130:
                    answer_hex = data[66:130]
                    price_raw = int(answer_hex, 16)
                    return price_raw / 1e8

            return None

        except Exception as e:
            logger.debug(f"Chainlink error: {e}")
            return None

    async def _get_market_prices(self) -> Optional[MarketPrices]:
        """Get current market prices from Polymarket."""
        if not self.up_token_id or not self.down_token_id:
            return None

        try:
            # Get order books
            up_response = await self.client.get(
                f"{POLYMARKET_CLOB_API}/book",
                params={"token_id": self.up_token_id}
            )
            down_response = await self.client.get(
                f"{POLYMARKET_CLOB_API}/book",
                params={"token_id": self.down_token_id}
            )

            if up_response.status_code != 200 or down_response.status_code != 200:
                return None

            up_book = up_response.json()
            down_book = down_response.json()

            # Get best asks
            up_asks = up_book.get('asks', [])
            down_asks = down_book.get('asks', [])

            if not up_asks or not down_asks:
                return None

            up_price = float(min(up_asks, key=lambda x: float(x['price']))['price'])
            down_price = float(min(down_asks, key=lambda x: float(x['price']))['price'])

            # Get best bids
            up_bids = up_book.get('bids', [])
            down_bids = down_book.get('bids', [])

            up_bid = float(max(up_bids, key=lambda x: float(x['price']))['price']) if up_bids else None
            down_bid = float(max(down_bids, key=lambda x: float(x['price']))['price']) if down_bids else None

            return MarketPrices(
                up_price=up_price,
                down_price=down_price,
                up_bid=up_bid,
                down_bid=down_bid,
                timestamp=int(time.time() * 1000)
            )

        except Exception as e:
            logger.debug(f"Market prices error: {e}")
            return None

    async def _price_loop(self):
        """Continuously fetch BTC prices."""
        while self.running:
            try:
                price = await self._get_btc_price()

                if price:
                    self.last_btc_price = price
                    vol_metrics = self.volatility.add_price(price)

                    if vol_metrics:
                        self.last_volatility = vol_metrics

                await asyncio.sleep(self.config.tick_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Price loop error: {e}")
                await asyncio.sleep(5)

    async def _analysis_loop(self):
        """Main analysis and trading loop."""
        while self.running:
            try:
                await self._run_analysis()
                await asyncio.sleep(self.config.analysis_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)

    async def _run_analysis(self):
        """Run full market analysis and potentially trade."""

        # Check prerequisites
        if not self.volatility.is_ready:
            return

        if not self.last_btc_price or not self.last_volatility:
            return

        # Get market prices
        market_prices = await self._get_market_prices()
        if not market_prices:
            return

        self.last_market_prices = market_prices

        # Calculate time remaining
        now = int(time.time() * 1000)
        if self.market_expiry:
            time_remaining = max(0, (self.market_expiry - now) // 1000)
        else:
            time_remaining = 900  # Default 15 min

        # Skip if too close to expiry
        if time_remaining < 30:
            return

        # Get strike price (use current BTC as proxy if not set)
        strike = self.strike_price or self.last_btc_price

        # Calculate volatility for remaining time
        per_second_vol = self.last_volatility.rolling_std

        # Estimate probabilities
        if self.config.use_momentum and isinstance(self.probability, AdaptiveProbabilityModel):
            up_est, down_est = self.probability.estimate_with_momentum(
                current_price=self.last_btc_price,
                strike_price=strike,
                time_remaining_seconds=time_remaining,
                volatility_per_second=per_second_vol,
                price_change_1m=self.last_volatility.price_change_1m,
                price_change_5m=self.last_volatility.price_change_5m
            )
        else:
            up_est, down_est = self.probability.estimate_both(
                current_price=self.last_btc_price,
                strike_price=strike,
                time_remaining_seconds=time_remaining,
                volatility_per_second=per_second_vol
            )

        # Detect edge
        edge = self.edge_detector.detect_edge(up_est, down_est, market_prices)

        # Also check for spread opportunity (Gabagool style)
        spread_edge = self.edge_detector.get_spread_opportunity(market_prices)

        # Log analysis
        analysis_data = self.edge_detector.analyze_market(up_est, down_est, market_prices)

        analysis_log = MarketAnalysisLog(
            timestamp=now,
            btc_price=self.last_btc_price,
            volatility=per_second_vol,
            volatility_annualized=self.last_volatility.rolling_std_annualized,
            model_up_prob=up_est.probability,
            model_down_prob=down_est.probability,
            model_confidence=up_est.confidence,
            market_up_price=market_prices.up_price,
            market_down_price=market_prices.down_price,
            market_spread=market_prices.spread,
            up_edge_pct=analysis_data['up_edge_pct'],
            down_edge_pct=analysis_data['down_edge_pct'],
            time_remaining_seconds=time_remaining,
            trade_signal=edge.signal.value if edge else "no_trade"
        )

        self.logger.log_analysis(analysis_log)

        # Record metrics
        self.metrics.record_tick(
            btc_price=self.last_btc_price,
            volatility=per_second_vol,
            up_edge=analysis_data['up_edge_pct'],
            down_edge=analysis_data['down_edge_pct'],
            spread=market_prices.spread
        )

        # Execute trade if edge found
        if edge or spread_edge:
            await self._execute_edge(edge or spread_edge, market_prices, time_remaining)

    async def _execute_edge(
        self,
        edge: EdgeOpportunity,
        market_prices: MarketPrices,
        time_remaining: int
    ):
        """Execute a trade on detected edge."""

        # Check if we can trade
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"Trade blocked: {reason}")
            return

        # Calculate position size
        size_usd, method = self.risk_manager.calculate_position_size(edge)

        if size_usd < 1.0:
            logger.debug(f"Position too small: ${size_usd:.2f}")
            return

        # Determine token
        token_id = self.up_token_id if edge.signal == TradeSignal.BUY_UP else self.down_token_id
        market_id = self.current_market.get('condition_id', '') if self.current_market else ''

        # Execute trade
        result = await self.executor.execute_trade(
            edge=edge,
            market_prices=market_prices,
            token_id=token_id,
            size_usd=size_usd,
            market_id=market_id,
            market_expiry=self.market_expiry or (int(time.time() * 1000) + time_remaining * 1000)
        )

        if result.success:
            # Record with risk manager
            self.risk_manager.record_trade(
                position_id=result.order_id,
                entry_price=result.fill_price,
                size_usd=result.fill_usd,
                direction=edge.direction.value
            )

            # Log trade
            trade_log = TradeLog(
                timestamp=result.timestamp,
                trade_id=result.order_id,
                direction=edge.direction.value,
                btc_price=self.last_btc_price,
                strike_price=self.strike_price or self.last_btc_price,
                time_remaining_seconds=time_remaining,
                model_probability=edge.model_probability,
                model_confidence=edge.confidence,
                volatility=self.last_volatility.rolling_std if self.last_volatility else 0,
                market_up_price=market_prices.up_price,
                market_down_price=market_prices.down_price,
                market_total=market_prices.total_price,
                edge_percent=edge.edge_percent,
                expected_value=edge.expected_value,
                kelly_fraction=edge.kelly_fraction,
                size_usd=result.fill_usd,
                fill_price=result.fill_price,
                tokens=result.fill_size,
                balance_before=self.risk_manager.state.balance + result.fill_usd,
                position_sizing_method=method
            )

            self.logger.log_trade(trade_log)

    async def _status_loop(self):
        """Periodic status logging."""
        while self.running:
            try:
                await asyncio.sleep(self.config.status_interval)

                if isinstance(self.executor, PaperExecutor):
                    positions = await self.executor.get_positions()

                    self.logger.log_status(
                        balance=self.executor.balance,
                        positions=len(positions),
                        can_trade=self.risk_manager.can_trade()[0],
                        reason=self.risk_manager.can_trade()[1]
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Status loop error: {e}")

    async def _market_refresh_loop(self):
        """Refresh market periodically."""
        while self.running:
            try:
                await asyncio.sleep(900)  # 15 minutes

                old_market = self.current_market
                await self._find_market()

                if self.current_market != old_market:
                    logger.info("Market refreshed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Market refresh error: {e}")

    async def _settlement_loop(self):
        """Check and settle expired positions."""
        while self.running:
            try:
                await asyncio.sleep(60)

                if not isinstance(self.executor, PaperExecutor):
                    continue

                now = int(time.time() * 1000)
                positions = await self.executor.get_positions()

                for position in positions:
                    if position.market_expiry and now > position.market_expiry:
                        # Determine if won based on current price vs strike
                        won = False
                        if self.last_btc_price and self.strike_price:
                            if position.direction == "up":
                                won = self.last_btc_price > self.strike_price
                            else:
                                won = self.last_btc_price < self.strike_price
                        else:
                            # Random settlement for demo
                            import random
                            won = random.random() < 0.5

                        pnl = self.executor.settle_position(position.position_id, won)

                        # Record with risk manager
                        self.risk_manager.record_exit(position.position_id, 1.0 if won else 0.0, won)

                        # Log settlement
                        settlement = SettlementLog(
                            timestamp=now,
                            trade_id=position.position_id,
                            direction=position.direction,
                            entry_price=position.entry_price,
                            size_usd=position.size_usd,
                            tokens=position.size_tokens,
                            won=won,
                            payout=position.size_tokens if won else 0,
                            pnl=pnl,
                            hold_time_seconds=(now - position.entry_time) // 1000
                        )

                        self.logger.log_settlement(settlement)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Settlement loop error: {e}")


async def run_bot(config: BotConfig):
    """Run the volatility arbitrage bot."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    bot = VolatilityArbBot(config)

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


# Default config presets
PAPER_TRADING_CONFIG = BotConfig(
    mode=ExecutionMode.PAPER,
    initial_balance=1000.0,
    risk_level="moderate",
    min_edge_percent=3.0,
)

CONSERVATIVE_CONFIG = BotConfig(
    mode=ExecutionMode.PAPER,
    initial_balance=1000.0,
    risk_level="conservative",
    min_edge_percent=5.0,
    min_confidence=0.7,
)

AGGRESSIVE_CONFIG = BotConfig(
    mode=ExecutionMode.PAPER,
    initial_balance=1000.0,
    risk_level="aggressive",
    min_edge_percent=2.0,
    min_confidence=0.4,
)
