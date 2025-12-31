"""
Log Processor for BTC Backtest
Processes JSON logs from the dashboard for backtesting
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Generator
from collections import defaultdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Market duration in milliseconds
FIFTEEN_MIN_MS = 15 * 60 * 1000  # 900000 ms


def get_market_start_timestamp(timestamp_ms: int) -> int:
    """
    Get the start timestamp of the 15-minute market containing this timestamp.

    Markets start at X:00, X:15, X:30, X:45.

    Args:
        timestamp_ms: Any timestamp in milliseconds

    Returns:
        Start timestamp of the market period in milliseconds
    """
    return (timestamp_ms // FIFTEEN_MIN_MS) * FIFTEEN_MIN_MS


def load_log_file(filepath: str) -> Dict[str, Any]:
    """
    Load a log file (JSON or GZIP).

    Args:
        filepath: Path to the log file

    Returns:
        Parsed log data
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def load_logs_from_directory(
    directory: str,
    pattern: str = "*.json*"
) -> Generator[Dict[str, Any], None, None]:
    """
    Load all log files from a directory.

    Args:
        directory: Path to directory with log files
        pattern: Glob pattern to match files

    Yields:
        Parsed log data from each file
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for filepath in sorted(dir_path.glob(pattern)):
        try:
            yield load_log_file(str(filepath))
            logger.info(f"Loaded: {filepath.name}")
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")


def assign_price_to_beat_to_ticks(
    ticks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add price_to_beat and market_start to each tick.

    Args:
        ticks: List of tick dictionaries with 'ts' and 'price' keys

    Returns:
        Updated ticks with price_to_beat and market_start fields
    """
    if not ticks:
        return ticks

    # Sort by timestamp
    ticks_sorted = sorted(ticks, key=lambda x: x['ts'])

    current_market_start = None
    current_price_to_beat = None

    for tick in ticks_sorted:
        market_start = get_market_start_timestamp(tick['ts'])

        # New market?
        if market_start != current_market_start:
            current_market_start = market_start
            current_price_to_beat = tick['price']

        # Add to tick
        tick['price_to_beat'] = current_price_to_beat
        tick['market_start'] = current_market_start

    return ticks_sorted


def group_ticks_by_market(
    ticks: List[Dict[str, Any]]
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group ticks by 15-minute market period.

    Args:
        ticks: List of tick dictionaries

    Returns:
        Dictionary mapping market_start to list of ticks
    """
    markets = defaultdict(list)

    for tick in ticks:
        market_start = get_market_start_timestamp(tick['ts'])
        markets[market_start].append(tick)

    # Sort ticks within each market
    for market_start in markets:
        markets[market_start].sort(key=lambda x: x['ts'])

    return dict(markets)


def extract_chainlink_ticks(log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract Chainlink ticks from log data.

    Args:
        log_data: Parsed log file

    Returns:
        List of tick dictionaries with 'ts', 'price', 'diff' keys
    """
    ticks = log_data.get('chainlink_ticks', [])

    # Ensure required fields
    processed = []
    for tick in ticks:
        if 'ts' in tick and 'price' in tick:
            processed.append({
                'ts': int(tick['ts']),
                'price': float(tick['price']),
                'diff': float(tick.get('diff', 0)),
            })

    return processed


def extract_price_changes(log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract token price changes from log data.

    Args:
        log_data: Parsed log file

    Returns:
        List of price change dictionaries with 'ts', 'up', 'down' keys
    """
    changes = log_data.get('price_changes', [])

    # Process based on format
    processed = []
    for change in changes:
        ts = change.get('ts') or change.get('timestamp')
        if ts is None:
            continue

        # Handle different formats for UP/DOWN prices
        up_price = (
            change.get('up') or
            change.get('yes') or
            change.get('price_up') or
            change.get('price_yes')
        )
        down_price = (
            change.get('down') or
            change.get('no') or
            change.get('price_down') or
            change.get('price_no')
        )

        if up_price is not None and down_price is not None:
            processed.append({
                'ts': int(ts),
                'up': float(up_price),
                'down': float(down_price),
            })

    return processed


def extract_order_books(log_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract order book snapshots from log data.

    Args:
        log_data: Parsed log file

    Returns:
        List of order book dictionaries
    """
    return log_data.get('order_books', [])


def merge_multiple_logs(
    log_files: List[str]
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Merge data from multiple log files.

    Args:
        log_files: List of paths to log files

    Returns:
        Tuple of (chainlink_ticks, price_changes, order_books)
    """
    all_ticks = []
    all_price_changes = []
    all_order_books = []

    for filepath in log_files:
        try:
            log_data = load_log_file(filepath)

            ticks = extract_chainlink_ticks(log_data)
            all_ticks.extend(ticks)

            price_changes = extract_price_changes(log_data)
            all_price_changes.extend(price_changes)

            order_books = extract_order_books(log_data)
            all_order_books.extend(order_books)

            logger.info(
                f"Loaded {filepath}: "
                f"{len(ticks)} ticks, "
                f"{len(price_changes)} price changes"
            )
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    # Remove duplicates and sort
    all_ticks = _deduplicate_by_timestamp(all_ticks)
    all_price_changes = _deduplicate_by_timestamp(all_price_changes)

    return all_ticks, all_price_changes, all_order_books


def _deduplicate_by_timestamp(items: List[Dict]) -> List[Dict]:
    """Remove duplicates by timestamp and sort."""
    seen = {}
    for item in items:
        ts = item.get('ts')
        if ts is not None:
            seen[ts] = item

    return sorted(seen.values(), key=lambda x: x['ts'])


def get_market_result(
    market_ticks: List[Dict[str, Any]],
    price_to_beat: float
) -> str:
    """
    Determine if UP or DOWN won a market.

    Args:
        market_ticks: Ticks for a single market
        price_to_beat: Price to beat for this market

    Returns:
        'UP', 'DOWN', or 'TIE'
    """
    if not market_ticks:
        return 'UNKNOWN'

    final_price = market_ticks[-1]['price']

    if final_price > price_to_beat:
        return 'UP'
    elif final_price < price_to_beat:
        return 'DOWN'
    else:
        return 'TIE'


def analyze_markets(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze all markets in the tick data.

    Args:
        ticks: List of all ticks

    Returns:
        Analysis summary
    """
    ticks_with_ptb = assign_price_to_beat_to_ticks(ticks.copy())
    markets = group_ticks_by_market(ticks_with_ptb)

    results = {
        'total_markets': len(markets),
        'up_wins': 0,
        'down_wins': 0,
        'ties': 0,
        'markets': [],
    }

    for market_start, market_ticks in sorted(markets.items()):
        price_to_beat = market_ticks[0]['price']
        final_price = market_ticks[-1]['price']
        outcome = get_market_result(market_ticks, price_to_beat)

        if outcome == 'UP':
            results['up_wins'] += 1
        elif outcome == 'DOWN':
            results['down_wins'] += 1
        else:
            results['ties'] += 1

        results['markets'].append({
            'start_ms': market_start,
            'price_to_beat': price_to_beat,
            'final_price': final_price,
            'outcome': outcome,
            'ticks_count': len(market_ticks),
            'price_change': final_price - price_to_beat,
            'price_change_pct': (final_price - price_to_beat) / price_to_beat * 100,
        })

    # Calculate percentages
    total = results['total_markets']
    if total > 0:
        results['up_pct'] = results['up_wins'] / total * 100
        results['down_pct'] = results['down_wins'] / total * 100

    return results


def prepare_backtest_data(
    log_files: List[str]
) -> Dict[str, Any]:
    """
    Prepare data for backtesting from log files.

    Args:
        log_files: List of paths to log files

    Returns:
        Dictionary with prepared data for backtesting
    """
    ticks, price_changes, order_books = merge_multiple_logs(log_files)

    # Process ticks with price to beat
    ticks_with_ptb = assign_price_to_beat_to_ticks(ticks.copy())

    # Analyze markets
    analysis = analyze_markets(ticks)

    return {
        'chainlink_ticks': ticks,
        'ticks_with_price_to_beat': ticks_with_ptb,
        'price_changes': price_changes,
        'order_books': order_books,
        'market_analysis': analysis,
        'total_ticks': len(ticks),
        'total_price_changes': len(price_changes),
        'total_markets': analysis['total_markets'],
    }


def generate_synthetic_token_prices(
    ticks: List[Dict[str, Any]],
    noise_factor: float = 0.05,
    lag_ms: int = 500,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic token prices for testing when no real data available.

    Uses the calculated probability plus some noise to simulate market prices.

    Args:
        ticks: Chainlink ticks with price_to_beat
        noise_factor: Random noise to add (0-1)
        lag_ms: Simulated lag in milliseconds

    Returns:
        List of synthetic price changes
    """
    import random
    from .probability_calculator import ProbabilityCalculator

    if not ticks:
        return []

    # Ensure ticks have price_to_beat
    ticks_processed = assign_price_to_beat_to_ticks(ticks.copy())
    markets = group_ticks_by_market(ticks_processed)

    price_changes = []

    for market_start, market_ticks in markets.items():
        calc = ProbabilityCalculator(min_std_dev=20.0)
        price_to_beat = market_ticks[0]['price']

        for tick in market_ticks:
            calc.add_tick(tick['price'], tick['ts'])
            std_dev = calc.calculate_std_dev()

            z_score, prob_up, prob_down = calc.calculate_probability(
                tick['price'], price_to_beat, std_dev
            )

            # Add noise
            noise = random.uniform(-noise_factor, noise_factor)
            up_price = max(0.01, min(0.99, prob_up + noise))
            down_price = max(0.01, min(0.99, prob_down - noise))

            # Normalize so they sum to approximately 1 (with some spread)
            total = up_price + down_price
            spread = random.uniform(0.95, 1.00)  # 0-5% spread
            up_price = up_price / total * spread
            down_price = down_price / total * spread

            price_changes.append({
                'ts': tick['ts'] + lag_ms,
                'up': round(up_price, 4),
                'down': round(down_price, 4),
            })

    return price_changes


def print_analysis_summary(analysis: Dict[str, Any]) -> None:
    """Print analysis summary to console."""
    print("=" * 60)
    print("MARKET ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Markets: {analysis['total_markets']}")
    print(f"UP Wins: {analysis['up_wins']} ({analysis.get('up_pct', 0):.1f}%)")
    print(f"DOWN Wins: {analysis['down_wins']} ({analysis.get('down_pct', 0):.1f}%)")
    print(f"Ties: {analysis['ties']}")
    print()

    if analysis.get('markets'):
        # Show first few markets
        print("Sample Markets:")
        print("-" * 60)
        for market in analysis['markets'][:5]:
            start_dt = datetime.fromtimestamp(market['start_ms'] / 1000)
            print(
                f"  {start_dt.strftime('%Y-%m-%d %H:%M')}: "
                f"Beat=${market['price_to_beat']:.2f} -> "
                f"Final=${market['final_price']:.2f} "
                f"({market['outcome']}, {market['price_change_pct']:+.2f}%)"
            )
        if len(analysis['markets']) > 5:
            print(f"  ... and {len(analysis['markets']) - 5} more markets")
    print("=" * 60)
