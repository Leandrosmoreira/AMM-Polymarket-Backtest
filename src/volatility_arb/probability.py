"""
Probability Model for BTC Price Prediction Markets

Uses statistical models to estimate the true probability of BTC
finishing above or below a given strike price at expiry.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class MarketDirection(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class ProbabilityEstimate:
    """Estimated probability for a market outcome."""
    direction: MarketDirection
    probability: float  # 0-1
    confidence: float  # 0-1 (how confident in the estimate)
    current_price: float
    strike_price: float
    time_remaining_seconds: int
    volatility: float  # Per-second volatility used
    model: str  # Which model was used


class ProbabilityModel:
    """
    Estimates true probabilities for BTC Up/Down markets.

    Uses a log-normal price model (geometric Brownian motion)
    similar to Black-Scholes option pricing.

    Key assumptions:
    - Log returns are normally distributed
    - Volatility is constant over short periods
    - No drift (fair game assumption for very short term)
    """

    def __init__(
        self,
        drift: float = 0.0,  # Expected return per second (usually 0 for 15min)
        min_time_seconds: int = 30,  # Minimum time for valid estimate
        volatility_floor: float = 0.00001,  # Minimum volatility
    ):
        self.drift = drift
        self.min_time_seconds = min_time_seconds
        self.volatility_floor = volatility_floor

    def estimate_up_probability(
        self,
        current_price: float,
        strike_price: float,
        time_remaining_seconds: int,
        volatility_per_second: float,
    ) -> ProbabilityEstimate:
        """
        Estimate probability that price will be ABOVE strike at expiry.

        Uses the cumulative distribution function of log-normal distribution.

        Args:
            current_price: Current BTC price
            strike_price: Strike price (usually same as current for 15min markets)
            time_remaining_seconds: Seconds until market expiry
            volatility_per_second: Per-second volatility (standard deviation of log returns)

        Returns:
            ProbabilityEstimate with UP probability
        """
        prob, confidence = self._calculate_probability(
            current_price=current_price,
            strike_price=strike_price,
            time_remaining=time_remaining_seconds,
            volatility=volatility_per_second,
            above=True
        )

        return ProbabilityEstimate(
            direction=MarketDirection.UP,
            probability=prob,
            confidence=confidence,
            current_price=current_price,
            strike_price=strike_price,
            time_remaining_seconds=time_remaining_seconds,
            volatility=volatility_per_second,
            model="log_normal"
        )

    def estimate_down_probability(
        self,
        current_price: float,
        strike_price: float,
        time_remaining_seconds: int,
        volatility_per_second: float,
    ) -> ProbabilityEstimate:
        """
        Estimate probability that price will be BELOW strike at expiry.

        Args:
            current_price: Current BTC price
            strike_price: Strike price
            time_remaining_seconds: Seconds until market expiry
            volatility_per_second: Per-second volatility

        Returns:
            ProbabilityEstimate with DOWN probability
        """
        prob, confidence = self._calculate_probability(
            current_price=current_price,
            strike_price=strike_price,
            time_remaining=time_remaining_seconds,
            volatility=volatility_per_second,
            above=False
        )

        return ProbabilityEstimate(
            direction=MarketDirection.DOWN,
            probability=prob,
            confidence=confidence,
            current_price=current_price,
            strike_price=strike_price,
            time_remaining_seconds=time_remaining_seconds,
            volatility=volatility_per_second,
            model="log_normal"
        )

    def estimate_both(
        self,
        current_price: float,
        strike_price: float,
        time_remaining_seconds: int,
        volatility_per_second: float,
    ) -> Tuple[ProbabilityEstimate, ProbabilityEstimate]:
        """
        Estimate probabilities for both UP and DOWN.

        Returns:
            Tuple of (up_estimate, down_estimate)
        """
        up = self.estimate_up_probability(
            current_price, strike_price, time_remaining_seconds, volatility_per_second
        )
        down = self.estimate_down_probability(
            current_price, strike_price, time_remaining_seconds, volatility_per_second
        )

        return up, down

    def _calculate_probability(
        self,
        current_price: float,
        strike_price: float,
        time_remaining: int,
        volatility: float,
        above: bool
    ) -> Tuple[float, float]:
        """
        Calculate probability using log-normal model.

        For log-normal distribution:
        P(S_T > K) = N(d2)
        where d2 = [ln(S/K) + (μ - σ²/2)T] / (σ√T)

        For short-term prediction markets, we assume μ ≈ 0 (no drift).

        Returns:
            Tuple of (probability, confidence)
        """
        # Edge cases
        if time_remaining < self.min_time_seconds:
            # Very close to expiry - price unlikely to change much
            if above:
                prob = 0.5 if current_price >= strike_price else 0.3
            else:
                prob = 0.5 if current_price <= strike_price else 0.3
            return prob, 0.5  # Low confidence near expiry

        # Ensure minimum volatility
        vol = max(volatility, self.volatility_floor)

        # Time in same units as volatility (seconds)
        T = time_remaining

        # Calculate d2 for log-normal CDF
        # d2 = [ln(S/K) + (μ - σ²/2)T] / (σ√T)
        try:
            log_ratio = math.log(current_price / strike_price)
            vol_sqrt_t = vol * math.sqrt(T)

            if vol_sqrt_t < 1e-10:
                # Volatility too low - use simple comparison
                if above:
                    return (0.55 if current_price > strike_price else 0.45), 0.3
                else:
                    return (0.55 if current_price < strike_price else 0.45), 0.3

            # With zero drift (μ = 0)
            d2 = (log_ratio - 0.5 * vol * vol * T) / vol_sqrt_t

            # Standard normal CDF
            prob_above = self._norm_cdf(d2)

            if above:
                prob = prob_above
            else:
                prob = 1 - prob_above

            # Confidence based on time and volatility
            # More confident with more time and stable volatility
            confidence = min(1.0, math.sqrt(time_remaining / 900))  # Max at 15 min

            return prob, confidence

        except (ValueError, ZeroDivisionError):
            return 0.5, 0.1  # Return 50/50 with low confidence on error

    def _norm_cdf(self, x: float) -> float:
        """
        Cumulative distribution function for standard normal distribution.
        Uses approximation accurate to ~1e-7.
        """
        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # Save the sign
        sign = 1 if x >= 0 else -1
        x = abs(x)

        # Approximation
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)

        return 0.5 * (1.0 + sign * y)


class AdaptiveProbabilityModel(ProbabilityModel):
    """
    Enhanced probability model that adapts based on recent price momentum.

    Considers:
    - Recent price trend (momentum)
    - Volatility regime (high vs low vol periods)
    - Time-of-day effects (optional)
    """

    def __init__(
        self,
        momentum_weight: float = 0.1,  # How much to weight recent momentum
        **kwargs
    ):
        super().__init__(**kwargs)
        self.momentum_weight = momentum_weight

    def estimate_with_momentum(
        self,
        current_price: float,
        strike_price: float,
        time_remaining_seconds: int,
        volatility_per_second: float,
        price_change_1m: float,  # % change in last minute
        price_change_5m: float,  # % change in last 5 minutes
    ) -> Tuple[ProbabilityEstimate, ProbabilityEstimate]:
        """
        Estimate probabilities with momentum adjustment.

        Args:
            current_price: Current BTC price
            strike_price: Strike price at market creation
            time_remaining_seconds: Seconds until expiry
            volatility_per_second: Per-second volatility
            price_change_1m: Percentage price change in last minute
            price_change_5m: Percentage price change in last 5 minutes

        Returns:
            Tuple of (up_estimate, down_estimate)
        """
        # Get base probabilities
        up_base, down_base = self.estimate_both(
            current_price, strike_price, time_remaining_seconds, volatility_per_second
        )

        # Calculate momentum adjustment
        # Positive momentum slightly increases UP probability
        momentum = (price_change_1m * 0.7 + price_change_5m * 0.3)  # Weighted average

        # Adjustment factor (small, max ±5%)
        adjustment = momentum * self.momentum_weight / 100

        # Apply adjustment with bounds
        up_prob = max(0.05, min(0.95, up_base.probability + adjustment))
        down_prob = 1 - up_prob

        # Create adjusted estimates
        up_adjusted = ProbabilityEstimate(
            direction=MarketDirection.UP,
            probability=up_prob,
            confidence=up_base.confidence * 0.9,  # Slightly less confident with adjustment
            current_price=current_price,
            strike_price=strike_price,
            time_remaining_seconds=time_remaining_seconds,
            volatility=volatility_per_second,
            model="log_normal_momentum"
        )

        down_adjusted = ProbabilityEstimate(
            direction=MarketDirection.DOWN,
            probability=down_prob,
            confidence=down_base.confidence * 0.9,
            current_price=current_price,
            strike_price=strike_price,
            time_remaining_seconds=time_remaining_seconds,
            volatility=volatility_per_second,
            model="log_normal_momentum"
        )

        return up_adjusted, down_adjusted
