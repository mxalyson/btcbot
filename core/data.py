"""
Data Manager - Download and cache historical data with multi-batch support.
Downloads ALL requested historical data by making multiple API requests.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional
import time

from core.bybit_rest import BybitRESTClient

logger = logging.getLogger("TradingBot.Data")


class DataManager:
    """Manage historical data download and caching."""

    def __init__(self, rest_client: BybitRESTClient):
        """Initialize data manager."""
        self.rest_client = rest_client
        self.cache_dir = Path("storage/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_data(self, symbol: str, interval: str, days_back: int = 90,
                use_cache: bool = True) -> pd.DataFrame:
        """Get historical data with caching."""

        # Check cache first
        if use_cache:
            cache_file = self.cache_dir / f"{symbol}_{interval}_{days_back}d.csv"

            if cache_file.exists():
                # Check if cache is recent (< 1 hour old)
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)

                if cache_age < timedelta(hours=1):
                    logger.info(f"Loading {symbol} {interval} from cache")
                    try:
                        df = pd.read_csv(cache_file)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        return df
                    except Exception as e:
                        logger.warning(f"Failed to load cache: {e}")

        # Download fresh data
        df = self.download_klines(symbol, interval, days_back)

        # Cache if valid data
        if not df.empty and use_cache:
            try:
                cache_file = self.cache_dir / f"{symbol}_{interval}_{days_back}d.csv"
                df.reset_index().to_csv(cache_file, index=False)
                logger.info(f"Cached {len(df)} candles to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        return df

    def download_klines(self, symbol: str, interval: str, days_back: int) -> pd.DataFrame:
        """
        Download klines from Bybit API with multi-batch support.
        Makes multiple requests to get all historical data.
        """
        logger.info(f"Downloading {symbol} {interval} data for {days_back} days...")

        # Convert interval to Bybit format
        interval_map = {
            '1m': '1',
            '3m': '3',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '2h': '120',
            '4h': '240',
            '6h': '360',
            '12h': '720',
            '1d': 'D',
            '1w': 'W',
            '1M': 'M'
        }

        bybit_interval = interval_map.get(interval, interval)

        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)

        all_candles = []
        current_end = end_time
        batch_count = 0
        max_batches = 50  # Safety limit

        logger.info(f"Target: {days_back} days from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

        # Bybit limits to 1000 candles per request
        # Loop backwards in time until we have all data
        while current_end > start_time and batch_count < max_batches:
            try:
                batch_count += 1

                response = self.rest_client.get_kline(
                    symbol=symbol,
                    interval=bybit_interval,
                    limit=1000,
                    start_time=None,  # Let API decide start
                    end_time=current_end
                )

                if response.get('retCode') != 0:
                    logger.error(f"API error: {response.get('retMsg')}")
                    break

                candles = response.get('result', {}).get('list', [])

                if not candles:
                    logger.info("No more candles available")
                    break

                # Add to collection
                all_candles.extend(candles)

                # Bybit returns newest first, so get oldest timestamp for next batch
                oldest_timestamp = int(candles[-1][0])

                logger.info(f"Batch {batch_count}: Downloaded {len(candles)} candles, total: {len(all_candles)}")

                # Check if we've reached the start time
                if oldest_timestamp <= start_time:
                    logger.info(f"Reached start time, stopping")
                    break

                # If we got less than 1000 candles, we've reached the limit
                if len(candles) < 1000:
                    logger.info(f"Got {len(candles)} candles (less than 1000), assuming end of data")
                    break

                # Update end time for next batch (go back in time)
                # Subtract 1ms to avoid duplicate
                current_end = oldest_timestamp - 1

                # Small delay to avoid rate limiting
                time.sleep(0.2)

            except Exception as e:
                logger.error(f"Error downloading batch {batch_count}: {e}")
                break

        if not all_candles:
            logger.warning(f"No data downloaded for {symbol} {interval}")
            return pd.DataFrame()

        # Parse candles to DataFrame
        df = self._parse_klines(all_candles)

        # Sort by timestamp (oldest first)
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Filter to exact date range requested
        start_date = datetime.now() - timedelta(days=days_back)
        df = df[df.index >= start_date]

        logger.info(f"✅ Downloaded {len(df)} candles for {symbol} {interval} ({days_back} days)")

        return df

    def _parse_klines(self, candles: list) -> pd.DataFrame:
        """
        Parse Bybit kline data to DataFrame.

        Bybit format: [timestamp, open, high, low, close, volume, turnover]
        """
        data = []

        for candle in candles:
            data.append({
                'timestamp': pd.to_datetime(int(candle[0]), unit='ms'),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })

        df = pd.DataFrame(data)

        if not df.empty:
            df.set_index('timestamp', inplace=True)

        return df

    def get_multiple_timeframes(self, symbol: str, timeframes: list, 
                               days_back: int = 90) -> dict:
        """Get data for multiple timeframes."""
        data = {}

        for tf in timeframes:
            df = self.get_data(symbol, tf, days_back)
            data[tf] = df
            logger.info(f"Loaded {len(df)} candles for {symbol} {tf}")

        return data
