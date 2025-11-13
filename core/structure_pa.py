"""
Price Action Structure Analysis - FIXED VERSION.
Detects market structure, swing points, CHoCH, BOS, FVG, S/R.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("TradingBot.Structure")


class PriceActionAnalyzer:
    """Analyze price action structure and patterns."""

    def __init__(self, config: dict):
        """Initialize price action analyzer."""
        self.config = config
        self.swing_lookback = config.get('swing_lookback', 5)
        self.fvg_threshold = config.get('fvg_threshold_pct', 0.1)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze price action structure.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with price action features
        """
        logger.info("Analyzing price action structure...")

        df = df.copy()

        # Detect swing points
        df = self._detect_swings(df)

        # Identify market structure
        df = self._identify_structure(df)

        # Detect CHoCH and BOS
        df = self._detect_choch_bos(df)

        # Find FVG (Fair Value Gaps)
        df = self._find_fvg(df)

        # Calculate S/R levels
        df = self._calculate_sr_levels(df)

        logger.info("Price action structure analysis complete")

        return df

    def _detect_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect swing highs and lows."""
        lookback = self.swing_lookback

        # Swing high: higher than N bars before and after
        df['swing_high'] = False
        df['swing_low'] = False

        for i in range(lookback, len(df) - lookback):
            # Check swing high
            is_swing_high = True
            for j in range(1, lookback + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                   df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_swing_high = False
                    break
            df.loc[df.index[i], 'swing_high'] = is_swing_high

            # Check swing low
            is_swing_low = True
            for j in range(1, lookback + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                   df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_swing_low = False
                    break
            df.loc[df.index[i], 'swing_low'] = is_swing_low

        return df

    def _identify_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify market structure (HH, HL, LH, LL)."""

        # Get swing points
        swing_highs = df[df['swing_high']]['high'].values
        swing_lows = df[df['swing_low']]['low'].values

        # Initialize trend
        df['trend'] = 'neutral'
        df['structure'] = 'none'

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return df

        # Detect HH/HL (bullish) or LH/LL (bearish)
        current_trend = 'neutral'

        for i in range(len(df)):
            # Get recent swings
            recent_highs = df.loc[:df.index[i]][df['swing_high']]['high'].tail(2)
            recent_lows = df.loc[:df.index[i]][df['swing_low']]['low'].tail(2)

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Check for HH and HL (bullish)
                if recent_highs.iloc[-1] > recent_highs.iloc[-2] and \
                   recent_lows.iloc[-1] > recent_lows.iloc[-2]:
                    current_trend = 'bullish'
                    df.loc[df.index[i], 'structure'] = 'HH+HL'

                # Check for LH and LL (bearish)
                elif recent_highs.iloc[-1] < recent_highs.iloc[-2] and \
                     recent_lows.iloc[-1] < recent_lows.iloc[-2]:
                    current_trend = 'bearish'
                    df.loc[df.index[i], 'structure'] = 'LH+LL'

            df.loc[df.index[i], 'trend'] = current_trend

        # FIXED: Use ffill() instead of fillna(method='ffill')
        df['trend'] = df['trend'].replace('neutral', np.nan).ffill().fillna('neutral')

        return df

    def _detect_choch_bos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect Change of Character (CHoCH) and Break of Structure (BOS)."""

        df['choch'] = False
        df['bos'] = False

        prev_trend = None

        for i in range(1, len(df)):
            current_trend = df['trend'].iloc[i]

            if prev_trend is not None and prev_trend != current_trend and \
               current_trend != 'neutral' and prev_trend != 'neutral':
                # CHoCH: trend reversal
                df.loc[df.index[i], 'choch'] = True

            # BOS: price breaks recent structure in trend direction
            if current_trend == 'bullish':
                recent_high = df['high'].iloc[max(0, i-20):i].max()
                if df['close'].iloc[i] > recent_high:
                    df.loc[df.index[i], 'bos'] = True

            elif current_trend == 'bearish':
                recent_low = df['low'].iloc[max(0, i-20):i].min()
                if df['close'].iloc[i] < recent_low:
                    df.loc[df.index[i], 'bos'] = True

            prev_trend = current_trend

        return df

    def _find_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find Fair Value Gaps (imbalances)."""

        df['fvg_bullish'] = False
        df['fvg_bearish'] = False

        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2].low and candle[i].high
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                gap_pct = (gap_size / df['close'].iloc[i-1]) * 100

                if gap_pct >= self.fvg_threshold:
                    df.loc[df.index[i], 'fvg_bullish'] = True

            # Bearish FVG: gap between candle[i-2].high and candle[i].low
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                gap_pct = (gap_size / df['close'].iloc[i-1]) * 100

                if gap_pct >= self.fvg_threshold:
                    df.loc[df.index[i], 'fvg_bearish'] = True

        return df

    def _calculate_sr_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate support and resistance levels."""

        # Find recent swing highs/lows as S/R
        lookback = 50

        df['resistance'] = np.nan
        df['support'] = np.nan
        df['dist_to_resistance_pct'] = 999.0
        df['dist_to_support_pct'] = 999.0

        for i in range(lookback, len(df)):
            recent_data = df.iloc[i-lookback:i]

            # Resistance: highest swing high
            swing_highs = recent_data[recent_data['swing_high']]['high']
            if len(swing_highs) > 0:
                resistance = swing_highs.max()
                df.loc[df.index[i], 'resistance'] = resistance

                dist_pct = ((resistance - df['close'].iloc[i]) / df['close'].iloc[i]) * 100
                df.loc[df.index[i], 'dist_to_resistance_pct'] = dist_pct

            # Support: lowest swing low
            swing_lows = recent_data[recent_data['swing_low']]['low']
            if len(swing_lows) > 0:
                support = swing_lows.min()
                df.loc[df.index[i], 'support'] = support

                dist_pct = ((df['close'].iloc[i] - support) / df['close'].iloc[i]) * 100
                df.loc[df.index[i], 'dist_to_support_pct'] = dist_pct

        return df
