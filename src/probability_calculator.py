"""
Probability Calculator for BTC Up/Down Strategy
Implements Z-Score, Standard Deviation, and Normal CDF calculations
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Precomputed coefficients for normal CDF approximation
# Using Abramowitz and Stegun approximation (error < 7.5e-8)
_A1 = 0.254829592
_A2 = -0.284496736
_A3 = 1.421413741
_A4 = -1.453152027
_A5 = 1.061405429
_P = 0.3275911


def normal_cdf(x: float) -> float:
    """
    Calculate the cumulative distribution function of standard normal distribution.
    Uses Abramowitz and Stegun approximation.

    Args:
        x: Z-score value

    Returns:
        Probability P(Z <= x)
    """
    # Handle edge cases
    if x > 10:
        return 1.0
    if x < -10:
        return 0.0

    # Save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)

    # A&S formula 7.1.26
    t = 1.0 / (1.0 + _P * x)
    y = 1.0 - ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t * math.exp(-x * x / 2)

    return 0.5 * (1.0 + sign * y)


@dataclass
class ProbabilityState:
    """Current probability state for a market."""
    timestamp_ms: int
    current_price: float
    price_to_beat: float
    std_dev: float
    z_score: float
    prob_up: float
    prob_down: float
    ticks_count: int


class ProbabilityCalculator:
    """
    Calculator for probability-based trading decisions.

    Implements:
    - Standard deviation calculation from price ticks
    - Z-Score calculation
    - Normal CDF for probability
    """

    def __init__(self, min_std_dev: float = 20.0):
        """
        Initialize the calculator.

        Args:
            min_std_dev: Minimum standard deviation to use (avoids extreme Z-scores)
        """
        self.min_std_dev = min_std_dev
        self._prices: List[float] = []
        self._timestamps: List[int] = []
        self._cached_std: Optional[float] = None
        self._cached_mean: Optional[float] = None
        self._last_std_calc_count: int = 0

    def reset(self) -> None:
        """Reset calculator state for new market."""
        self._prices = []
        self._timestamps = []
        self._cached_std = None
        self._cached_mean = None
        self._last_std_calc_count = 0

    def add_tick(self, price: float, timestamp_ms: int) -> None:
        """
        Add a new price tick.

        Args:
            price: BTC price in USD
            timestamp_ms: Timestamp in milliseconds
        """
        self._prices.append(price)
        self._timestamps.append(timestamp_ms)

    def add_ticks(self, ticks: List[Dict[str, Any]]) -> None:
        """
        Add multiple price ticks.

        Args:
            ticks: List of tick dictionaries with 'price' and 'ts' keys
        """
        for tick in ticks:
            self.add_tick(tick['price'], tick['ts'])

    @property
    def tick_count(self) -> int:
        """Number of ticks collected."""
        return len(self._prices)

    def calculate_std_dev(self, force_recalc: bool = False) -> float:
        """
        Calculate standard deviation of collected prices.
        Uses population standard deviation.

        Args:
            force_recalc: Force recalculation even if cached

        Returns:
            Standard deviation, never less than min_std_dev
        """
        n = len(self._prices)

        if n < 2:
            return self.min_std_dev

        # Use cache if available and no new ticks
        if not force_recalc and self._cached_std is not None and n == self._last_std_calc_count:
            return max(self._cached_std, self.min_std_dev)

        # Calculate mean
        mean = sum(self._prices) / n
        self._cached_mean = mean

        # Calculate variance
        variance = sum((x - mean) ** 2 for x in self._prices) / n

        # Calculate std dev
        std = math.sqrt(variance)
        self._cached_std = std
        self._last_std_calc_count = n

        return max(std, self.min_std_dev)

    def calculate_z_score(
        self,
        current_price: float,
        price_to_beat: float,
        std_dev: Optional[float] = None
    ) -> float:
        """
        Calculate Z-Score.

        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat (from market start)
            std_dev: Standard deviation (uses calculated if None)

        Returns:
            Z-Score value
        """
        if std_dev is None:
            std_dev = self.calculate_std_dev()

        if std_dev == 0:
            std_dev = self.min_std_dev

        return (current_price - price_to_beat) / std_dev

    def calculate_probability(
        self,
        current_price: float,
        price_to_beat: float,
        std_dev: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate probability of UP/DOWN outcome.

        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat (from market start)
            std_dev: Standard deviation (uses calculated if None)

        Returns:
            Tuple of (z_score, prob_up, prob_down)
        """
        z_score = self.calculate_z_score(current_price, price_to_beat, std_dev)
        prob_up = normal_cdf(z_score)
        prob_down = 1.0 - prob_up

        return z_score, prob_up, prob_down

    def get_state(
        self,
        current_price: float,
        price_to_beat: float,
        timestamp_ms: int = 0
    ) -> ProbabilityState:
        """
        Get complete probability state.

        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat (from market start)
            timestamp_ms: Current timestamp

        Returns:
            ProbabilityState with all calculations
        """
        std_dev = self.calculate_std_dev()
        z_score, prob_up, prob_down = self.calculate_probability(
            current_price, price_to_beat, std_dev
        )

        return ProbabilityState(
            timestamp_ms=timestamp_ms,
            current_price=current_price,
            price_to_beat=price_to_beat,
            std_dev=std_dev,
            z_score=z_score,
            prob_up=prob_up,
            prob_down=prob_down,
            ticks_count=self.tick_count,
        )

    def calculate_opportunity(
        self,
        current_price: float,
        price_to_beat: float,
        token_price_up: float,
        token_price_down: float,
        std_dev: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate trading opportunity.

        Args:
            current_price: Current BTC price
            price_to_beat: Price to beat (from market start)
            token_price_up: Current price of UP token (0-1)
            token_price_down: Current price of DOWN token (0-1)
            std_dev: Standard deviation (uses calculated if None)

        Returns:
            Dictionary with opportunity analysis
        """
        z_score, prob_up, prob_down = self.calculate_probability(
            current_price, price_to_beat, std_dev
        )

        opp_up = prob_up - token_price_up
        opp_down = prob_down - token_price_down

        # Determine best opportunity
        if opp_up > opp_down and opp_up > 0:
            best_side = 'UP'
            best_opp = opp_up
        elif opp_down > opp_up and opp_down > 0:
            best_side = 'DOWN'
            best_opp = opp_down
        else:
            best_side = 'NONE'
            best_opp = 0.0

        return {
            'z_score': z_score,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'token_price_up': token_price_up,
            'token_price_down': token_price_down,
            'opp_up': opp_up,
            'opp_down': opp_down,
            'best_side': best_side,
            'best_opp': best_opp,
            'has_opportunity': best_opp > 0,
        }


def calculate_std_from_ticks(ticks: List[Dict[str, Any]], min_std: float = 20.0) -> float:
    """
    Calculate standard deviation from a list of ticks.

    Args:
        ticks: List of tick dictionaries with 'price' key
        min_std: Minimum standard deviation to return

    Returns:
        Standard deviation, never less than min_std
    """
    if not ticks or len(ticks) < 2:
        return min_std

    prices = [t['price'] for t in ticks]
    n = len(prices)
    mean = sum(prices) / n
    variance = sum((x - mean) ** 2 for x in prices) / n
    std = math.sqrt(variance)

    return max(std, min_std)


def calculate_probability_at_timestamp(
    ticks: List[Dict[str, Any]],
    timestamp_ms: int,
    price_to_beat: float,
    min_std: float = 20.0
) -> Dict[str, Any]:
    """
    Calculate probability using ticks up to a specific timestamp.

    Args:
        ticks: All available ticks
        timestamp_ms: Calculate up to this timestamp
        price_to_beat: Price to beat for this market
        min_std: Minimum standard deviation

    Returns:
        Dictionary with probability calculations
    """
    # Filter ticks up to timestamp
    ticks_before = [t for t in ticks if t['ts'] <= timestamp_ms]

    if not ticks_before:
        return {
            'z_score': 0.0,
            'prob_up': 0.5,
            'prob_down': 0.5,
            'std_dev': min_std,
            'ticks_count': 0,
            'current_price': None,
        }

    # Get current price (latest tick)
    current_price = ticks_before[-1]['price']

    # Calculate std dev
    std_dev = calculate_std_from_ticks(ticks_before, min_std)

    # Calculate z-score and probability
    calc = ProbabilityCalculator(min_std)
    z_score, prob_up, prob_down = calc.calculate_probability(
        current_price, price_to_beat, std_dev
    )

    return {
        'z_score': z_score,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'std_dev': std_dev,
        'ticks_count': len(ticks_before),
        'current_price': current_price,
    }
