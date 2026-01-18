"""
Data Collector for Polymarket 15-min Markets (BTC, ETH, SOL)
Collects historical market data and price history from Polymarket API
"""

import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Optional, List, Dict, Any
import asyncio

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Supported assets
SUPPORTED_ASSETS = ['btc', 'eth', 'sol']


class DataCollector:
    """Collector for Polymarket market data."""

    def __init__(self):
        self.gamma_api = settings.GAMMA_API_BASE
        self.clob_api = settings.CLOB_API_BASE
        self.rate_limit = settings.MAX_REQUESTS_PER_SECOND
        self.last_request_time = 0
        self.client = httpx.Client(timeout=settings.REQUEST_TIMEOUT)

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        wait_time = 1.0 / self.rate_limit - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def fetch_15min_markets(
        self,
        asset: str,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch all Up/Down 15-minute markets for an asset in the date range.

        Args:
            asset: Asset to fetch (btc, eth, sol)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_path: Optional path to save CSV

        Returns:
            DataFrame with market information
        """
        asset = asset.lower()
        asset_upper = asset.upper()

        logger.info(f"Fetching {asset_upper} 15-min markets from {start_date} to {end_date}")

        all_markets = []
        offset = 0
        page = 0
        max_pages = 100  # Safety limit

        # Updated search strategies with correct slug format
        search_params_list = [
            # Strategy 1: Search by slug containing {asset}-updown-15m (formato atual)
            {"slug_contains": f"{asset}-updown-15m", "closed": "true", "limit": 100},
            # Strategy 2: Search by slug containing {asset}-updown (formato antigo)
            {"slug_contains": f"{asset}-updown", "closed": "true", "limit": 100},
            # Strategy 3: Search by tag crypto
            {"tag": "crypto", "closed": "true", "limit": 100},
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

                # Filter for asset Up/Down 15-min markets
                for market in markets:
                    question = (market.get("question") or "").lower()
                    slug = (market.get("slug") or "").lower()
                    description = (market.get("description") or "").lower()

                    # Check if it's the correct asset market
                    asset_names = {
                        'btc': ['bitcoin', 'btc ', 'btc-'],
                        'eth': ['ethereum', 'eth ', 'eth-', 'ether'],
                        'sol': ['solana', 'sol ', 'sol-'],
                    }

                    is_asset = any([
                        name in question or name in slug
                        for name in asset_names.get(asset, [])
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

                    if is_asset and is_updown:
                        market_data = self._parse_market(market, asset)
                        if market_data:
                            # Check if not duplicate
                            if not any(m['market_id'] == market_data['market_id'] for m in all_markets):
                                all_markets.append(market_data)
                                if len(all_markets) <= 3:
                                    logger.info(f"Found {asset_upper} market: {market_data['question'][:80]}...")

                page += 1
                offset += 100

                if page % 10 == 0:
                    logger.info(f"Page {page}: Found {len(all_markets)} {asset_upper} markets so far")

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

            logger.info(f"Total {asset_upper} markets found after date filter: {len(df)}")

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_path, index=False)
                logger.info(f"Saved to {save_path}")
        else:
            logger.warning(f"No {asset_upper} markets found. The API might have changed.")

        return df

    def fetch_all_assets_markets(
        self,
        assets: List[str],
        start_date: str,
        end_date: str,
        save_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch markets for multiple assets.

        Args:
            assets: List of assets to fetch (e.g., ['btc', 'eth', 'sol'])
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_dir: Optional directory to save CSVs

        Returns:
            Consolidated DataFrame with all markets
        """
        all_markets = []

        for asset in assets:
            asset = asset.lower()
            if asset not in SUPPORTED_ASSETS:
                logger.warning(f"Asset {asset} not supported. Supported: {SUPPORTED_ASSETS}")
                continue

            save_path = None
            if save_dir:
                save_path = f"{save_dir}/{asset}_markets.csv"

            df = self.fetch_15min_markets(asset, start_date, end_date, save_path)
            if not df.empty:
                all_markets.append(df)

        if all_markets:
            consolidated = pd.concat(all_markets, ignore_index=True)
            consolidated = consolidated.sort_values('start_time').reset_index(drop=True)

            if save_dir:
                consolidated.to_csv(f"{save_dir}/all_markets.csv", index=False)
                logger.info(f"Saved consolidated markets to {save_dir}/all_markets.csv")

            return consolidated

        return pd.DataFrame()

    def _parse_market(self, market: Dict[str, Any], asset: str = "") -> Optional[Dict[str, Any]]:
        """Parse market data into standardized format."""
        try:
            tokens = market.get("tokens", [])
            yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), None)
            no_token = next((t for t in tokens if t.get("outcome") == "No"), None)

            return {
                "market_id": market.get("id") or market.get("condition_id"),
                "condition_id": market.get("condition_id"),
                "asset": asset.upper(),
                "question": market.get("question"),
                "slug": market.get("slug"),
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
            asset = market.get('asset', 'UNKNOWN')
            logger.info(f"Fetching prices for {asset} market {idx + 1}/{len(markets_df)}")

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
                prices['asset'] = asset
                prices['spread'] = prices['price_yes'] + prices['price_no'] - 1.0
                prices['mid_price'] = (prices['price_yes'] + prices['price_no']) / 2

                all_prices.append(prices)

                if save_dir:
                    save_path = Path(save_dir) / f"{market['market_id']}.csv"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    prices.to_csv(save_path, index=False)

        if all_prices:
            consolidated = pd.concat(all_prices, ignore_index=True)
            return consolidated

        return pd.DataFrame()

    # Backward compatibility
    def fetch_sol_15min_markets(
        self,
        start_date: str,
        end_date: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Backward compatible method for fetching SOL markets."""
        return self.fetch_15min_markets('sol', start_date, end_date, save_path)

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def main():
    """Main function to collect data."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect Polymarket 15-min market data")
    parser.add_argument("--assets", default="btc,eth,sol", help="Assets to collect (comma-separated)")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--fetch-prices", action="store_true", help="Also fetch price history")
    args = parser.parse_args()

    assets = [a.strip().lower() for a in args.assets.split(',')]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    collector = DataCollector()

    try:
        # Fetch markets for all assets
        logger.info(f"Collecting data for assets: {', '.join(a.upper() for a in assets)}")

        markets_df = collector.fetch_all_assets_markets(
            assets=assets,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            save_dir=settings.DATA_RAW_PATH
        )

        if not markets_df.empty:
            logger.info(f"Found {len(markets_df)} total markets")

            # Show breakdown by asset
            for asset in assets:
                count = len(markets_df[markets_df['asset'] == asset.upper()])
                logger.info(f"  {asset.upper()}: {count} markets")

            if args.fetch_prices:
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
        else:
            logger.warning("No markets found for any asset.")

    finally:
        collector.close()


if __name__ == "__main__":
    main()
