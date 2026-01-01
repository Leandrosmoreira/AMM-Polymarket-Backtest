"""
Volatility Calculator for BTC Price Data
Calculates rolling volatility metrics for probability estimation
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Deque
import time


@dataclass
class VolatilityMetrics:
    """Current volatility metrics."""
    current_price: float
    rolling_std: float  # Rolling standard deviation
    rolling_std_annualized: float  # Annualized volatility
    price_change_1m: float  # 1-minute price change
    price_change_5m: float  # 5-minute price change
    samples: int  # Number of samples in window
    timestamp: int  # Timestamp in ms


class VolatilityCalculator:
    """
    Calculates rolling volatility metrics from BTC price stream.

    Uses exponentially weighted moving standard deviation for
    more responsive volatility estimates.
    """

    def __init__(
        self,
        window_seconds: int = 300,  # 5 minute window
        min_samples: int = 30,  # Minimum samples before valid volatility
        ewma_span: int = 60,  # EWMA span for smoothing
    ):
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.ewma_span = ewma_span

        # Price history: (timestamp_ms, price)
        self._prices: Deque[tuple] = deque()

        # Return history for volatility calculation
        self._returns: Deque[float] = deque()

        # EWMA state
        self._ewma_variance: Optional[float] = None
        self._alpha = 2.0 / (ewma_span + 1)

        # Last price for return calculation
        self._last_price: Optional[float] = None
        self._last_timestamp: Optional[int] = None

    def add_price(self, price: float, timestamp_ms: Optional[int] = None) -> Optional[VolatilityMetrics]:
        """
        Add a new price observation and update volatility metrics.

        Args:
            price: Current BTC price
            timestamp_ms: Timestamp in milliseconds (default: current time)

        Returns:
            VolatilityMetrics if enough samples, None otherwise
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Store price
        self._prices.append((timestamp_ms, price))

        # Calculate log return if we have previous price
        if self._last_price is not None and self._last_price > 0:
            log_return = math.log(price / self._last_price)
            self._returns.append(log_return)

            # Update EWMA variance
            if self._ewma_variance is None:
                self._ewma_variance = log_return ** 2
            else:
                self._ewma_variance = (
                    self._alpha * (log_return ** 2) +
                    (1 - self._alpha) * self._ewma_variance
                )

        self._last_price = price
        self._last_timestamp = timestamp_ms

        # Clean old data outside window
        self._cleanup_old_data(timestamp_ms)

        # Return metrics if we have enough samples
        if len(self._returns) >= self.min_samples:
            return self._calculate_metrics(price, timestamp_ms)

        return None

    def _cleanup_old_data(self, current_time_ms: int):
        """Remove data older than window."""
        cutoff = current_time_ms - (self.window_seconds * 1000)

        while self._prices and self._prices[0][0] < cutoff:
            self._prices.popleft()
            if self._returns:
                self._returns.popleft()

    def _calculate_metrics(self, current_price: float, timestamp_ms: int) -> VolatilityMetrics:
        """Calculate current volatility metrics."""

        # Rolling standard deviation of returns
        returns_list = list(self._returns)
        n = len(returns_list)

        if n > 1:
            mean_return = sum(returns_list) / n
            variance = sum((r - mean_return) ** 2 for r in returns_list) / (n - 1)
            rolling_std = math.sqrt(variance)
        else:
            rolling_std = 0.0

        # Annualized volatility (assuming ~1 second between ticks)
        # Annualization factor: sqrt(seconds per year) = sqrt(31,536,000) ≈ 5615
        # But for 15-min windows, we use per-second volatility
        ticks_per_year = 31_536_000  # Assuming 1 tick/second
        rolling_std_annualized = rolling_std * math.sqrt(ticks_per_year)

        # Price changes
        price_change_1m = self._get_price_change(timestamp_ms, 60_000)
        price_change_5m = self._get_price_change(timestamp_ms, 300_000)

        return VolatilityMetrics(
            current_price=current_price,
            rolling_std=rolling_std,
            rolling_std_annualized=rolling_std_annualized,
            price_change_1m=price_change_1m,
            price_change_5m=price_change_5m,
            samples=n,
            timestamp=timestamp_ms,
        )

    def _get_price_change(self, current_time_ms: int, lookback_ms: int) -> float:
        """Get price change over lookback period."""
        if not self._prices:
            return 0.0

        current_price = self._prices[-1][1]
        target_time = current_time_ms - lookback_ms

        # Find price closest to target time
        for ts, price in self._prices:
            if ts >= target_time:
                if price > 0:
                    return (current_price - price) / price * 100
                break

        return 0.0

    def get_volatility_for_window(self, seconds: int) -> Optional[float]:
        """
        Get volatility scaled for a specific time window.

        Args:
            seconds: Time window in seconds (e.g., 900 for 15 minutes)

        Returns:
            Expected standard deviation of price change over window
        """
        if self._ewma_variance is None or len(self._returns) < self.min_samples:
            return None

        # Per-tick standard deviation
        per_tick_std = math.sqrt(self._ewma_variance)

        # Scale to window (assuming ~1 second per tick)
        # Volatility scales with sqrt(time)
        window_std = per_tick_std * math.sqrt(seconds)

        return window_std

    def get_expected_range(self, seconds: int, confidence: float = 0.95) -> Optional[tuple]:
        """
        Get expected price range for a time window.

        Args:
            seconds: Time window in seconds
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_price, upper_price) or None
        """
        if not self._prices:
            return None

        vol = self.get_volatility_for_window(seconds)
        if vol is None:
            return None

        current_price = self._prices[-1][1]

        # Z-score for confidence level (two-tailed)
        # 0.95 -> 1.96, 0.99 -> 2.576
        import statistics
        try:
            from scipy import stats
            z = stats.norm.ppf((1 + confidence) / 2)
        except ImportError:
            # Fallback for common values
            z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z = z_table.get(confidence, 1.96)

        # Expected range using log-normal distribution
        lower = current_price * math.exp(-z * vol)
        upper = current_price * math.exp(z * vol)

        return (lower, upper)

    @property
    def is_ready(self) -> bool:
        """Check if calculator has enough data for valid volatility."""
        return len(self._returns) >= self.min_samples

    @property
    def sample_count(self) -> int:
        """Current number of samples in window."""
        return len(self._returns)
