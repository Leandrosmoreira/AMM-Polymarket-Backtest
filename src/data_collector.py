"""
Data Collector for Polymarket 15-min Markets
Collects historical market data and price history from Polymarket API

Supports multiple assets: SOL, BTC, ETH, etc.

Best practices implemented:
- Aggressive local caching
- Batch requests instead of polling
- Explicit User-Agent
- Automatic fallback to simulation
- LIVE_MODE vs SIM_MODE separation
"""

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
import re
import json
import hashlib
from typing import Optional, List, Dict, Any
import asyncio

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

# === MODES ===
class DataMode:
    LIVE = "live"      # Fetch from API
    SIM = "sim"        # Use simulated data
    CACHE = "cache"    # Use cached data if available, fallback to API

# === CACHE SETTINGS ===
CACHE_DIR = Path("data/cache")
CACHE_TTL_HOURS = 1  # Cache validity in hours

# === HTTP SETTINGS ===
USER_AGENT = "AMM-Backtest/1.0 (Polymarket Research Bot; +https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest)"
DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Asset search patterns
ASSET_PATTERNS = {
    'BTC': {
        'names': ['bitcoin', 'btc'],
        'slug_patterns': ['btc-updown', 'btc-up-down', 'bitcoin-updown'],
    },
    'SOL': {
        'names': ['solana', 'sol'],
        'slug_patterns': ['sol-updown', 'sol-up-down', 'solana-updown'],
    },
    'ETH': {
        'names': ['ethereum', 'eth'],
        'slug_patterns': ['eth-updown', 'eth-up-down', 'ethereum-updown'],
    },
}


class LocalCache:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Path = CACHE_DIR, ttl_hours: int = CACHE_TTL_HOURS):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key from endpoint and params."""
        params_str = json.dumps(params, sort_keys=True)
        key = f"{endpoint}:{params_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def get(self, endpoint: str, params: Dict) -> Optional[Any]:
        """Get cached response if valid."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)

            # Check TTL
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                logger.debug(f"Cache expired for {endpoint}")
                return None

            logger.debug(f"Cache hit for {endpoint}")
            return cached['data']

        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, endpoint: str, params: Dict, data: Any) -> None:
        """Cache response data."""
        cache_key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'endpoint': endpoint,
                    'params': params,
                    'data': data,
                }, f)
            logger.debug(f"Cached response for {endpoint}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        logger.info("Cache cleared")


class DataCollector:
    """Collector for Polymarket market data."""

    def __init__(self, mode: str = DataMode.CACHE):
        """
        Initialize collector.

        Args:
            mode: DataMode.LIVE, DataMode.SIM, or DataMode.CACHE
        """
        self.mode = mode
        self.gamma_api = settings.GAMMA_API_BASE
        self.clob_api = settings.CLOB_API_BASE
        self.rate_limit = settings.MAX_REQUESTS_PER_SECOND
        self.last_request_time = 0

        # HTTP client with proper headers
        self.client = httpx.Client(
            timeout=settings.REQUEST_TIMEOUT,
            headers=DEFAULT_HEADERS,
        )

        # Local cache
        self.cache = LocalCache()

        # Request stats
        self.requests_made = 0
        self.cache_hits = 0
        self.errors = 0

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        wait_time = 1.0 / self.rate_limit - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _fetch_with_cache(self, endpoint: str, params: Dict) -> Optional[Any]:
        """
        Fetch data with cache support.

        Checks cache first (if mode allows), then fetches from API.
        """
        # Check cache first (unless LIVE mode)
        if self.mode != DataMode.LIVE:
            cached = self.cache.get(endpoint, params)
            if cached is not None:
                self.cache_hits += 1
                return cached

        # SIM mode: don't make API calls
        if self.mode == DataMode.SIM:
            logger.info("SIM mode: skipping API call")
            return None

        # Make API request
        self._rate_limit_wait()
        self.requests_made += 1

        try:
            response = self.client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            self.cache.set(endpoint, params, data)

            return data

        except httpx.HTTPStatusError as e:
            self.errors += 1
            if e.response.status_code == 403:
                logger.warning(f"403 Forbidden - API may be rate limiting or blocking requests")
            elif e.response.status_code == 429:
                logger.warning(f"429 Too Many Requests - backing off")
                time.sleep(5)  # Extra backoff
            else:
                logger.error(f"HTTP error {e.response.status_code}: {e}")
            return None

        except Exception as e:
            self.errors += 1
            logger.error(f"Request error: {e}")
            return None

    def fetch_15min_markets(
        self,
        asset: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all Up/Down 15-minute markets for a given asset.

        Args:
            asset: Asset symbol (BTC, SOL, ETH)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_path: Optional path to save CSV

        Returns:
            DataFrame with market information
        """
        asset = asset.upper()
        if asset not in ASSET_PATTERNS:
            logger.warning(f"Unknown asset {asset}, using generic search")
            patterns = {
                'names': [asset.lower()],
                'slug_patterns': [f'{asset.lower()}-updown'],
            }
        else:
            patterns = ASSET_PATTERNS[asset]

        logger.info(f"Fetching {asset} 15-min markets from {start_date} to {end_date}")
        logger.info(f"Mode: {self.mode.upper()}")

        all_markets = []
        offset = 0
        page = 0
        max_pages = 200

        # Search with larger batch size to reduce requests
        search_params_list = [
            {"closed": "true", "limit": 100},  # Batch of 100
        ]

        for search_params in search_params_list:
            offset = 0
            page = 0

            while page < max_pages:
                params = search_params.copy()
                params["offset"] = offset

                # Use cached fetch
                data = self._fetch_with_cache(f"{self.gamma_api}/markets", params)

                if data is None:
                    logger.warning("No data returned, stopping fetch")
                    break

                markets = data if isinstance(data, list) else data.get("data", data)

                if not markets or not isinstance(markets, list):
                    break

                # Filter for target asset markets
                for market in markets:
                    question = (market.get("question") or "").lower()
                    slug = (market.get("slug") or "").lower()

                    # Check if it's the target asset
                    is_target_asset = any(
                        name in question or name in slug
                        for name in patterns['names']
                    )

                    # Check if it's up/down market
                    is_updown = any([
                        "up or down" in question,
                        "up/down" in question,
                        "updown" in slug,
                        "up-down" in slug,
                    ])

                    # Check if it's 15-min
                    is_15min = any([
                        "15" in question,
                        "15m" in slug,
                        "15-m" in slug,
                        ":00-" in question and ":15" in question,
                        ":15-" in question and ":30" in question,
                        ":30-" in question and ":45" in question,
                        ":45-" in question and ":00" in question,
                    ])

                    if is_target_asset and is_updown:
                        market_data = self._parse_market(market)
                        if market_data:
                            if not any(m['market_id'] == market_data['market_id'] for m in all_markets):
                                all_markets.append(market_data)

                page += 1
                offset += 100

                if page % 20 == 0:
                    logger.info(f"Page {page}: Found {len(all_markets)} {asset} markets")

                if len(markets) < 100:
                    break

        df = pd.DataFrame(all_markets)

        if not df.empty:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
            df = df.dropna(subset=['start_time'])

            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['start_time'] >= start) & (df['start_time'] <= end)]
            df = df.sort_values('start_time').reset_index(drop=True)

            logger.info(f"Total {asset} markets found: {len(df)}")

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                logger.info(f"Saved to {save_path}")

        return df

    def fetch_sol_15min_markets(
        self,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all SOL Up/Down 15-minute markets in the date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_path: Optional path to save CSV

        Returns:
            DataFrame with market information
        """
        logger.info(f"Fetching SOL 15-min markets from {start_date} to {end_date}")

        all_markets = []
        offset = 0
        page = 0
        max_pages = 100  # Safety limit

        # Try different search strategies
        search_params_list = [
            # Strategy 1: Search by slug containing sol-updown
            {"slug_contains": "sol-updown", "closed": "true", "limit": 100},
            # Strategy 2: Search by tag
            {"tag": "crypto", "closed": "true", "limit": 100},
            # Strategy 3: No filter, get all closed markets
            {"closed": "true", "limit": 100},
        ]

        for search_params in search_params_list:
            if all_markets:  # Found markets with previous strategy
                break

            offset = 0
            page = 0
            logger.info(f"Trying search params: {search_params}")

            while page < max_pages:
                self._rate_limit_wait()

                params = search_params.copy()
                params["offset"] = offset

                try:
                    response = self.client.get(
                        f"{self.gamma_api}/markets",
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception as e:
                    logger.error(f"Error fetching markets: {e}")
                    break

                markets = data if isinstance(data, list) else data.get("data", data)

                if not markets or not isinstance(markets, list):
                    logger.debug(f"No more markets at offset {offset}")
                    break

                # Debug: show first market structure
                if page == 0 and markets:
                    sample = markets[0]
                    logger.debug(f"Sample market keys: {sample.keys()}")
                    logger.debug(f"Sample question: {sample.get('question', 'N/A')[:100]}")
                    logger.debug(f"Sample slug: {sample.get('slug', 'N/A')}")

                # Filter for SOL Up/Down 15-min markets
                for market in markets:
                    question = (market.get("question") or "").lower()
                    slug = (market.get("slug") or "").lower()
                    description = (market.get("description") or "").lower()

                    # Check if it's a SOL market
                    is_sol = any([
                        "solana" in question,
                        "sol " in question,
                        "sol-" in slug,
                        "solana" in slug,
                    ])

                    # Check if it's up/down market
                    is_updown = any([
                        "up or down" in question,
                        "up/down" in question,
                        "updown" in slug,
                        "up-down" in slug,
                    ])

                    # Check if it's 15-min
                    is_15min = any([
                        "15" in question,
                        "15m" in slug,
                        "15-m" in slug,
                        # Time patterns like 4:00-4:15, 4:15-4:30, etc
                        ":00-" in question and ":15" in question,
                        ":15-" in question and ":30" in question,
                        ":30-" in question and ":45" in question,
                        ":45-" in question and ":00" in question,
                    ])

                    if is_sol and is_updown:
                        market_data = self._parse_market(market)
                        if market_data:
                            # Check if not duplicate
                            if not any(m['market_id'] == market_data['market_id'] for m in all_markets):
                                all_markets.append(market_data)
                                if len(all_markets) <= 3:
                                    logger.info(f"Found market: {market_data['question'][:80]}...")

                page += 1
                offset += 100

                if page % 10 == 0:
                    logger.info(f"Page {page}: Found {len(all_markets)} SOL markets so far")

                # Stop if we got less than limit (no more pages)
                if len(markets) < 100:
                    break

        df = pd.DataFrame(all_markets)

        if not df.empty:
            # Filter by date range
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

            # Remove rows with invalid dates
            df = df.dropna(subset=['start_time'])

            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['start_time'] >= start) & (df['start_time'] <= end)]

            # Sort by start time
            df = df.sort_values('start_time').reset_index(drop=True)

            logger.info(f"Total markets found after date filter: {len(df)}")

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                logger.info(f"Saved to {save_path}")
        else:
            logger.warning("No markets found. The API might have changed or no SOL 15-min markets exist.")

        return df

    def _parse_market(self, market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse market data into standardized format."""
        try:
            tokens = market.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            no_token = next((t for t in tokens if t.get("outcome") == "No"), None)

            return {
                "market_id": market.get("id") or market.get("condition_id"),
                "condition_id": market.get("condition_id"),
                "question": market.get("question"),
                "start_time": market.get("start_date") or market.get("created_at"),
                "end_time": market.get("end_date") or market.get("closed_at"),
                "outcome": market.get("outcome") or market.get("resolution"),
                "yes_token_id": yes_token.get("token_id") if yes_token else None,
                "no_token_id": no_token.get("token_id") if no_token else None,
                "volume": float(market.get("volume", 0) or 0),
                "liquidity": float(market.get("liquidity", 0) or 0),
            }
        except Exception as e:
            logger.warning(f"Error parsing market: {e}")
            return None

    def fetch_price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Fetch price history for a token.

        Args:
            token_id: Token ID to fetch prices for
            start_ts: Start timestamp (Unix)
            end_ts: End timestamp (Unix)
            interval: Price interval (1m, 5m, 1h, etc.)

        Returns:
            DataFrame with price history
        """
        self._rate_limit_wait()

        try:
            response = self.client.get(
                f"{self.clob_api}/prices-history",
                params={
                    "market": token_id,
                    "interval": interval,
                    "fidelity": 60,
                    "startTs": start_ts,
                    "endTs": end_ts,
                }
            )
            response.raise_for_status()
            data = response.json()

            if not data or "history" not in data:
                return pd.DataFrame()

            history = data["history"]
            df = pd.DataFrame(history)

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                df['price'] = df['p'].astype(float)
                df = df[['timestamp', 'price']]

            return df

        except Exception as e:
            logger.warning(f"Error fetching price history for {token_id}: {e}")
            return pd.DataFrame()

    def fetch_all_prices(
        self,
        markets_df: pd.DataFrame,
        save_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch price history for all markets.

        Args:
            markets_df: DataFrame with market information
            save_dir: Optional directory to save individual price CSVs

        Returns:
            Consolidated DataFrame with all prices
        """
        all_prices = []

        for idx, market in markets_df.iterrows():
            logger.info(f"Fetching prices for market {idx + 1}/{len(markets_df)}")

            start_ts = int(pd.to_datetime(market['start_time']).timestamp())
            end_ts = int(pd.to_datetime(market['end_time']).timestamp())

            # Fetch YES prices
            yes_prices = self.fetch_price_history(
                market['yes_token_id'], start_ts, end_ts
            )
            if not yes_prices.empty:
                yes_prices['price_yes'] = yes_prices['price']

            # Fetch NO prices
            no_prices = self.fetch_price_history(
                market['no_token_id'], start_ts, end_ts
            )
            if not no_prices.empty:
                no_prices['price_no'] = no_prices['price']

            # Merge YES and NO prices
            if not yes_prices.empty and not no_prices.empty:
                prices = pd.merge(
                    yes_prices[['timestamp', 'price_yes']],
                    no_prices[['timestamp', 'price_no']],
                    on='timestamp',
                    how='outer'
                )
                prices['market_id'] = market['market_id']
                prices['spread'] = prices['price_yes'] + prices['price_no'] - 1.0
                prices['mid_price'] = (prices['price_yes'] + (1 - prices['price_no'])) / 2

                all_prices.append(prices)

                if save_dir:
                    save_path = Path(save_dir) / f"{market['market_id']}.csv"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    prices.to_csv(save_path, index=False)

        if all_prices:
            consolidated = pd.concat(all_prices, ignore_index=True)
            return consolidated

        return pd.DataFrame()

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            'mode': self.mode,
            'requests_made': self.requests_made,
            'cache_hits': self.cache_hits,
            'errors': self.errors,
            'cache_hit_rate': self.cache_hits / max(1, self.requests_made + self.cache_hits),
        }

    def close(self):
        """Close the HTTP client and show stats."""
        stats = self.get_stats()
        logger.info(f"Collector stats: {stats['requests_made']} requests, "
                   f"{stats['cache_hits']} cache hits, {stats['errors']} errors")
        self.client.close()


def main():
    """Main function to collect data."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Polymarket SOL data")
    parser.add_argument("--asset", default="SOL", help="Asset to collect")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    args = parser.parse_args()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    collector = DataCollector()

    try:
        # Fetch markets
        markets_df = collector.fetch_sol_15min_markets(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            save_path=f"{settings.DATA_RAW_PATH}/sol_markets.csv"
        )

        if not markets_df.empty:
            # Fetch prices
            prices_df = collector.fetch_all_prices(
                markets_df,
                save_dir=f"{settings.DATA_RAW_PATH}/price_history"
            )

            if not prices_df.empty:
                # Save consolidated prices
                prices_df.to_parquet(
                    f"{settings.DATA_PROCESSED_PATH}/all_prices.parquet",
                    index=False
                )
                logger.info("Data collection complete!")

    finally:
        collector.close()


if __name__ == "__main__":
    main()
