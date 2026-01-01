"""
Edge Detector for Volatility Arbitrage Bot

Compares model-estimated probabilities to market prices to identify trading edges.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import time

from .probability import ProbabilityEstimate, MarketDirection


class TradeSignal(Enum):
    BUY_UP = "buy_up"
    BUY_DOWN = "buy_down"
    NO_TRADE = "no_trade"


@dataclass
class MarketPrices:
    """Current market prices for UP/DOWN tokens."""
    up_price: float  # Price to buy UP token (0-1)
    down_price: float  # Price to buy DOWN token (0-1)
    up_bid: Optional[float] = None  # Best bid for UP
    down_bid: Optional[float] = None  # Best bid for DOWN
    timestamp: int = 0

    @property
    def total_price(self) -> float:
        """Sum of UP and DOWN prices."""
        return self.up_price + self.down_price

    @property
    def spread(self) -> float:
        """Total price - 1 (should be ~0 for efficient market)."""
        return self.total_price - 1.0


@dataclass
class EdgeOpportunity:
    """A detected trading edge."""
    signal: TradeSignal
    edge_percent: float  # How much edge we have (model prob - market price)
    model_probability: float  # Our estimated probability
    market_price: float  # What market is charging
    expected_value: float  # Expected profit per $1 bet
    confidence: float  # How confident in the edge
    direction: MarketDirection
    timestamp: int

    @property
    def kelly_fraction(self) -> float:
        """
        Kelly criterion optimal bet fraction.
        f* = (bp - q) / b
        where b = odds, p = win prob, q = lose prob
        """
        if self.market_price <= 0 or self.market_price >= 1:
            return 0.0

        # Odds offered (payout per $1 bet if win)
        b = (1.0 - self.market_price) / self.market_price

        p = self.model_probability
        q = 1 - p

        kelly = (b * p - q) / b if b > 0 else 0

        return max(0, kelly)


class EdgeDetector:
    """
    Detects trading edges by comparing model probabilities to market prices.

    An edge exists when:
    1. Model probability > Market price + threshold (buy signal)
    2. Edge is consistent across multiple observations
    3. Confidence level meets minimum requirements
    """

    def __init__(
        self,
        min_edge_percent: float = 3.0,  # Minimum edge to consider (3%)
        min_confidence: float = 0.5,  # Minimum model confidence
        min_expected_value: float = 0.02,  # Minimum EV per $1 (2 cents)
        fee_percent: float = 1.0,  # Polymarket fee estimate
        confirmation_count: int = 2,  # Observations before confirming edge
    ):
        self.min_edge_percent = min_edge_percent
        self.min_confidence = min_confidence
        self.min_expected_value = min_expected_value
        self.fee_percent = fee_percent
        self.confirmation_count = confirmation_count

        # Track recent edges for confirmation
        self._recent_edges: List[EdgeOpportunity] = []
        self._max_recent = 10

    def detect_edge(
        self,
        up_estimate: ProbabilityEstimate,
        down_estimate: ProbabilityEstimate,
        market_prices: MarketPrices,
    ) -> Optional[EdgeOpportunity]:
        """
        Check if there's a trading edge.

        Args:
            up_estimate: Model's UP probability estimate
            down_estimate: Model's DOWN probability estimate
            market_prices: Current market prices

        Returns:
            EdgeOpportunity if edge detected, None otherwise
        """
        # Calculate edges for both directions
        up_edge = self._calculate_edge(
            model_prob=up_estimate.probability,
            market_price=market_prices.up_price,
            confidence=up_estimate.confidence,
            direction=MarketDirection.UP
        )

        down_edge = self._calculate_edge(
            model_prob=down_estimate.probability,
            market_price=market_prices.down_price,
            confidence=down_estimate.confidence,
            direction=MarketDirection.DOWN
        )

        # Return the better edge if it meets criteria
        best_edge = None

        if up_edge and (best_edge is None or up_edge.edge_percent > best_edge.edge_percent):
            best_edge = up_edge

        if down_edge and (best_edge is None or down_edge.edge_percent > best_edge.edge_percent):
            best_edge = down_edge

        # Track edge for confirmation
        if best_edge:
            self._add_recent_edge(best_edge)

            # Check if edge is confirmed
            if self._is_edge_confirmed(best_edge):
                return best_edge

        return None

    def _calculate_edge(
        self,
        model_prob: float,
        market_price: float,
        confidence: float,
        direction: MarketDirection
    ) -> Optional[EdgeOpportunity]:
        """Calculate edge for a single direction."""

        # Edge = Model probability - Market price
        # Positive edge means market is underpricing the outcome
        edge = model_prob - market_price
        edge_percent = edge * 100

        # Expected value calculation
        # If we buy at market_price and win with model_prob:
        # EV = model_prob * (1 - market_price) - (1 - model_prob) * market_price
        # EV = model_prob - market_price
        ev_gross = edge

        # Subtract fees
        ev_net = ev_gross - (self.fee_percent / 100)

        # Check minimum criteria
        if edge_percent < self.min_edge_percent:
            return None

        if confidence < self.min_confidence:
            return None

        if ev_net < self.min_expected_value:
            return None

        signal = TradeSignal.BUY_UP if direction == MarketDirection.UP else TradeSignal.BUY_DOWN

        return EdgeOpportunity(
            signal=signal,
            edge_percent=edge_percent,
            model_probability=model_prob,
            market_price=market_price,
            expected_value=ev_net,
            confidence=confidence,
            direction=direction,
            timestamp=int(time.time() * 1000)
        )

    def _add_recent_edge(self, edge: EdgeOpportunity):
        """Add edge to recent history."""
        self._recent_edges.append(edge)
        if len(self._recent_edges) > self._max_recent:
            self._recent_edges.pop(0)

    def _is_edge_confirmed(self, edge: EdgeOpportunity) -> bool:
        """Check if edge is confirmed by recent observations."""
        if self.confirmation_count <= 1:
            return True

        # Count recent edges in same direction
        same_direction = [
            e for e in self._recent_edges[-self.confirmation_count:]
            if e.direction == edge.direction
        ]

        return len(same_direction) >= self.confirmation_count

    def get_spread_opportunity(self, market_prices: MarketPrices) -> Optional[EdgeOpportunity]:
        """
        Check for Gabagool-style spread opportunity.

        If UP + DOWN < 1, there's guaranteed profit by buying both.
        """
        if market_prices.total_price < 1.0:
            spread_edge = 1.0 - market_prices.total_price
            spread_edge_percent = spread_edge * 100

            # Subtract fees for both sides
            ev_net = spread_edge - 2 * (self.fee_percent / 100)

            if ev_net > 0:
                return EdgeOpportunity(
                    signal=TradeSignal.BUY_UP,  # Buy both, but signal as UP
                    edge_percent=spread_edge_percent,
                    model_probability=1.0,  # Guaranteed win
                    market_price=market_prices.total_price,
                    expected_value=ev_net,
                    confidence=1.0,
                    direction=MarketDirection.UP,
                    timestamp=int(time.time() * 1000)
                )

        return None

    def analyze_market(
        self,
        up_estimate: ProbabilityEstimate,
        down_estimate: ProbabilityEstimate,
        market_prices: MarketPrices,
    ) -> dict:
        """
        Full market analysis for logging/monitoring.

        Returns dict with all relevant metrics.
        """
        up_edge = up_estimate.probability - market_prices.up_price
        down_edge = down_estimate.probability - market_prices.down_price

        return {
            "timestamp": int(time.time() * 1000),

            # Model estimates
            "model_up_prob": round(up_estimate.probability, 4),
            "model_down_prob": round(down_estimate.probability, 4),
            "model_confidence": round(up_estimate.confidence, 4),

            # Market prices
            "market_up_price": round(market_prices.up_price, 4),
            "market_down_price": round(market_prices.down_price, 4),
            "market_total": round(market_prices.total_price, 4),
            "market_spread": round(market_prices.spread, 4),

            # Edges
            "up_edge_pct": round(up_edge * 100, 2),
            "down_edge_pct": round(down_edge * 100, 2),

            # Context
            "btc_price": up_estimate.current_price,
            "strike_price": up_estimate.strike_price,
            "time_remaining": up_estimate.time_remaining_seconds,
            "volatility": up_estimate.volatility,
        }
