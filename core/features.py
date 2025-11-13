"""
Feature Store - FINAL VERSION with config passed correctly.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from core.indicators import calculate_all_indicators
from core.structure_pa import PriceActionAnalyzer

logger = logging.getLogger("TradingBot.Features")


class FeatureStore:
    """Build and manage feature sets for ML."""

    def __init__(self, config: dict):
        """Initialize feature store."""
        self.config = config
        self.pa_analyzer = PriceActionAnalyzer(config)

    def build_features(self, df: pd.DataFrame, 
                      orderbook: Optional[Dict] = None,
                      trades: Optional[list] = None,
                      normalize: bool = False) -> pd.DataFrame:
        """Build complete feature set."""
        logger.info("Building complete feature set...")

        df = df.copy()

        # 1. Technical indicators
        df = self.add_technical_indicators(df)

        # 2. Price action features
        df = self.add_price_action_features(df)

        # 3. Derived features
        df = self.add_derived_features(df)

        # 4. Microstructure (if available)
        if orderbook:
            df = self.add_microstructure_features(df, orderbook, trades)

        # Remove NaN rows at start
        df = df.dropna()

        if normalize:
            df = self.normalize_features(df)

        logger.info(f"Feature set complete: {len(df.columns)} columns")

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        logger.debug("Adding technical indicators...")

        # FIXED: Pass config as required
        df = calculate_all_indicators(df, self.config)

        return df

    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features."""
        logger.debug("Adding price action features...")

        # Run price action analysis
        df = self.pa_analyzer.analyze(df)

        # Convert boolean to int
        bool_cols = ['swing_high', 'swing_low', 'choch', 'bos', 
                    'fvg_bullish', 'fvg_bearish']

        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Convert trend to numeric
        if 'trend' in df.columns:
            trend_map = {'bullish': 1, 'neutral': 0, 'bearish': -1}
            df['trend_numeric'] = df['trend'].map(trend_map).fillna(0)

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/engineered features."""
        logger.debug("Adding derived features...")

        # Price vs EMAs
        if 'ema21' in df.columns:
            df['price_vs_ema21'] = ((df['close'] - df['ema21']) / df['ema21']) * 100

        if 'ema50' in df.columns:
            df['price_vs_ema50'] = ((df['close'] - df['ema50']) / df['ema50']) * 100

        if 'ema200' in df.columns:
            df['price_vs_ema200'] = ((df['close'] - df['ema200']) / df['ema200']) * 100

        # EMA relationships
        if 'ema21' in df.columns and 'ema50' in df.columns:
            df['ema21_vs_ema50'] = ((df['ema21'] - df['ema50']) / df['ema50']) * 100

        if 'ema50' in df.columns and 'ema200' in df.columns:
            df['ema50_vs_ema200'] = ((df['ema50'] - df['ema200']) / df['ema200']) * 100

        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['volume_std'] = df['volume'].rolling(20).std()

        # Price momentum
        df['return_1'] = df['close'].pct_change(1) * 100
        df['return_5'] = df['close'].pct_change(5) * 100
        df['return_10'] = df['close'].pct_change(10) * 100
        df['return_20'] = df['close'].pct_change(20) * 100

        # Volatility
        df['volatility_5'] = df['return_1'].rolling(5).std()
        df['volatility_20'] = df['return_1'].rolling(20).std()

        # ATR normalized
        if 'atr' in df.columns:
            df['atr_normalized'] = df['atr'] / df['close']
            df['atr_ratio'] = df['atr'] / (df['atr'].rolling(20).mean() + 1e-8)

        # Bollinger Band position
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            bb_range = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-8)
            df['bb_width'] = bb_range / df['bb_middle']

        # RSI features
        if 'rsi' in df.columns:
            df['rsi_ma'] = df['rsi'].rolling(5).mean()
            df['rsi_std'] = df['rsi'].rolling(5).std()
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

        # MACD features
        if 'macd_hist' in df.columns:
            df['macd_hist_change'] = df['macd_hist'].diff()
            df['macd_positive'] = (df['macd_hist'] > 0).astype(int)

        # ADX features
        if 'adx' in df.columns:
            df['adx_strong'] = (df['adx'] > 25).astype(int)
            df['adx_very_strong'] = (df['adx'] > 40).astype(int)

        # VWAP distance
        if 'vwap' in df.columns:
            df['close_vs_vwap'] = ((df['close'] - df['vwap']) / df['vwap']) * 100

        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / (df['open'] + 1e-8) * 100
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['open'] + 1e-8) * 100
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['open'] + 1e-8) * 100
        df['is_green'] = (df['close'] > df['open']).astype(int)

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-8) * 100
        df['hl_range_ma'] = df['hl_range'].rolling(20).mean()

        return df

    def add_microstructure_features(self, df: pd.DataFrame,
                                   orderbook: Dict,
                                   trades: Optional[list]) -> pd.DataFrame:
        """Add microstructure features."""

        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return df

        if not orderbook['bids'] or not orderbook['asks']:
            return df

        best_bid = max(orderbook['bids'].keys())
        best_ask = min(orderbook['asks'].keys())

        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100

        # Add to last row
        df.loc[df.index[-1], 'spread'] = spread
        df.loc[df.index[-1], 'spread_pct'] = spread_pct
        df.loc[df.index[-1], 'mid_price'] = mid_price

        # OBI
        bid_volume = sum(list(orderbook['bids'].values())[:5])
        ask_volume = sum(list(orderbook['asks'].values())[:5])

        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            obi = (bid_volume - ask_volume) / total_volume
            df.loc[df.index[-1], 'obi'] = obi

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""

        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                  'target', 'symbol', 'timeframe']

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude]

        for col in cols_to_normalize:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        return df
