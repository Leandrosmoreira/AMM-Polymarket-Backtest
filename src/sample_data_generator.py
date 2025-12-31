"""
Sample Data Generator for BTC Backtest
Generates realistic simulated data for testing the backtest engine
"""

import json
import gzip
import random
import math
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Market duration in milliseconds
FIFTEEN_MIN_MS = 15 * 60 * 1000  # 900000 ms


def generate_random_walk(
    start_price: float,
    n_steps: int,
    volatility: float = 0.0002,
    drift: float = 0.0,
) -> List[float]:
    """
    Generate a random walk price series.

    Args:
        start_price: Starting price
        n_steps: Number of steps
        volatility: Price volatility (std dev per step)
        drift: Trend direction

    Returns:
        List of prices
    """
    prices = [start_price]
    for _ in range(n_steps - 1):
        change = random.gauss(drift, volatility) * prices[-1]
        new_price = max(start_price * 0.9, min(start_price * 1.1, prices[-1] + change))
        prices.append(new_price)
    return prices


def generate_chainlink_ticks(
    n_markets: int = 10,
    base_price: float = 87000.0,
    volatility: float = 0.0002,
    ticks_per_market: int = 900,
    start_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Generate simulated Chainlink price ticks.

    Args:
        n_markets: Number of 15-minute markets
        base_price: Base BTC price
        volatility: Price volatility
        ticks_per_market: Ticks per market (900 = 1 per second)
        start_time: Start time (defaults to now - n_markets * 15 min)

    Returns:
        List of tick dictionaries
    """
    if start_time is None:
        start_time = datetime.now() - timedelta(minutes=15 * n_markets)

    base_time_ms = int(start_time.timestamp() * 1000)
    # Align to 15-minute boundary
    base_time_ms = (base_time_ms // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS

    ticks = []
    current_price = base_price

    for market_idx in range(n_markets):
        market_start_ms = base_time_ms + (market_idx * FIFTEEN_MIN_MS)
        price_to_beat = current_price

        # Generate prices for this market
        prices = generate_random_walk(
            start_price=current_price,
            n_steps=ticks_per_market,
            volatility=volatility,
        )

        for i, price in enumerate(prices):
            ts = market_start_ms + (i * 1000)  # 1 tick per second
            ticks.append({
                'ts': ts,
                'price': round(price, 2),
                'diff': round(price - price_to_beat, 2),
            })

        current_price = prices[-1]

    return ticks


def generate_token_prices(
    chainlink_ticks: List[Dict[str, Any]],
    noise_factor: float = 0.03,
    efficiency: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Generate token prices based on calculated probabilities with noise.

    Args:
        chainlink_ticks: Chainlink price ticks
        noise_factor: Random noise to add
        efficiency: Market efficiency (1.0 = perfect, 0.0 = random)

    Returns:
        List of price change dictionaries
    """
    from .probability_calculator import ProbabilityCalculator
    from .btc_market_cycle import get_market_start_timestamp

    price_changes = []
    markets: Dict[int, List[Dict]] = {}

    # Group by market
    for tick in chainlink_ticks:
        market_start = get_market_start_timestamp(tick['ts'])
        if market_start not in markets:
            markets[market_start] = []
        markets[market_start].append(tick)

    # Generate prices for each market
    for market_start, market_ticks in markets.items():
        calc = ProbabilityCalculator(min_std_dev=20.0)
        price_to_beat = market_ticks[0]['price']

        for tick in market_ticks:
            calc.add_tick(tick['price'], tick['ts'])
            std_dev = calc.calculate_std_dev()

            _, prob_up, prob_down = calc.calculate_probability(
                tick['price'], price_to_beat, std_dev
            )

            # Apply efficiency and noise
            noise = random.uniform(-noise_factor, noise_factor)

            # Efficient market: price tracks probability
            # Inefficient market: more random
            up_price = efficiency * prob_up + (1 - efficiency) * 0.5 + noise
            down_price = efficiency * prob_down + (1 - efficiency) * 0.5 - noise

            # Clamp and normalize
            up_price = max(0.05, min(0.95, up_price))
            down_price = max(0.05, min(0.95, down_price))

            # Add spread (total < 1.0)
            spread = random.uniform(0.97, 1.00)
            total = up_price + down_price
            up_price = up_price / total * spread
            down_price = down_price / total * spread

            price_changes.append({
                'ts': tick['ts'] + 500,  # 500ms lag
                'up': round(up_price, 4),
                'down': round(down_price, 4),
            })

    return price_changes


def generate_order_book(
    token_price: float,
    depth: int = 5,
    spread: float = 0.005,
) -> Dict[str, List[Dict]]:
    """
    Generate a simulated order book.

    Args:
        token_price: Current token price
        depth: Number of levels
        spread: Bid-ask spread

    Returns:
        Dictionary with 'bids' and 'asks' lists
    """
    mid = token_price
    half_spread = spread / 2

    bids = []
    asks = []

    for i in range(depth):
        offset = half_spread + (i * 0.002)

        bids.append({
            'price': round(mid - offset, 4),
            'size': random.uniform(50, 500),
        })
        asks.append({
            'price': round(mid + offset, 4),
            'size': random.uniform(50, 500),
        })

    return {'bids': bids, 'asks': asks}


def generate_complete_log(
    n_markets: int = 10,
    base_price: float = 87000.0,
    volatility: float = 0.0002,
    market_efficiency: float = 0.8,
) -> Dict[str, Any]:
    """
    Generate a complete log file with all data types.

    Args:
        n_markets: Number of markets
        base_price: Base BTC price
        volatility: Price volatility
        market_efficiency: How efficient the market is (1.0 = perfect)

    Returns:
        Dictionary with complete log data
    """
    chainlink_ticks = generate_chainlink_ticks(
        n_markets=n_markets,
        base_price=base_price,
        volatility=volatility,
    )

    price_changes = generate_token_prices(
        chainlink_ticks,
        efficiency=market_efficiency,
    )

    # Generate order books for a subset of timestamps
    order_books = []
    for pc in price_changes[::10]:  # Every 10th price change
        order_books.append({
            'ts': pc['ts'],
            'up': generate_order_book(pc['up']),
            'down': generate_order_book(pc['down']),
        })

    return {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_markets': n_markets,
            'base_price': base_price,
            'volatility': volatility,
            'market_efficiency': market_efficiency,
        },
        'chainlink_ticks': chainlink_ticks,
        'price_changes': price_changes,
        'order_books': order_books,
    }


def save_log(
    data: Dict[str, Any],
    filepath: str,
    compress: bool = True,
) -> str:
    """
    Save log data to file.

    Args:
        data: Log data dictionary
        filepath: Output path (without extension)
        compress: Whether to gzip compress

    Returns:
        Actual path saved to
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        output_path = f"{filepath}.json.gz"
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        output_path = f"{filepath}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    logger.info(f"Saved log to {output_path}")
    return output_path


def generate_sample_logs(
    output_dir: str = "data/raw",
    n_files: int = 3,
    markets_per_file: int = 10,
    **kwargs,
) -> List[str]:
    """
    Generate multiple sample log files.

    Args:
        output_dir: Directory to save files
        n_files: Number of files to generate
        markets_per_file: Markets per file
        **kwargs: Additional args for generate_complete_log

    Returns:
        List of generated file paths
    """
    paths = []
    base_time = datetime.now() - timedelta(hours=n_files)

    for i in range(n_files):
        file_time = base_time + timedelta(hours=i)
        filename = f"sample_log_{file_time.strftime('%Y%m%d_%H%M%S')}"

        data = generate_complete_log(
            n_markets=markets_per_file,
            **kwargs,
        )

        path = save_log(data, f"{output_dir}/{filename}")
        paths.append(path)

    return paths


if __name__ == '__main__':
    # Generate sample data when run directly
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("Generating sample BTC backtest data...")

    paths = generate_sample_logs(
        output_dir="data/raw",
        n_files=3,
        markets_per_file=20,
        volatility=0.0003,
        market_efficiency=0.75,
    )

    print(f"\nGenerated {len(paths)} log files:")
    for path in paths:
        print(f"  - {path}")

    print("\nRun backtest with:")
    print(f"  python main.py btc-backtest --log-dir data/raw")
