"""Signal generation combining PA, indicators, and microstructure."""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("TradingBot.Signals")

class SignalGenerator:
    """Generate long/short signals with confidence scores."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def check_regime(self, df: pd.DataFrame) -> str:
        """Detect market regime: trending or ranging."""
        adx = df['adx'].iloc[-1]
        
        if adx >= 25:
            return 'trending'
        elif adx < 20:
            return 'ranging'
        else:
            return 'transitional'
    
    def check_trend_direction(self, df: pd.DataFrame) -> str:
        """Check trend direction."""
        ema50 = df['ema50'].iloc[-1]
        ema200 = df['ema200'].iloc[-1]
        
        if pd.isna(ema50) or pd.isna(ema200):
            return 'neutral'
        
        if ema50 > ema200:
            return 'bullish'
        elif ema50 < ema200:
            return 'bearish'
        else:
            return 'neutral'
    
    def score_long_signal(self, df: pd.DataFrame, microstructure: Dict, 
                          higher_tf_trend: str = 'bullish') -> Tuple[float, Dict]:
        """
        Score long signal quality (0-1).
        Returns: (confidence_score, signal_details)
        """
        score = 0.0
        weights = {
            'trend': 0.25,
            'context': 0.20,
            'momentum': 0.20,
            'microstructure': 0.20,
            'multi_tf': 0.15
        }
        details = {}
        
        # 1. Trend alignment (25%)
        trend = self.check_trend_direction(df)
        if trend == 'bullish':
            score += weights['trend']
            details['trend'] = 'aligned'
        else:
            details['trend'] = 'not_aligned'
        
        # 2. Context: pullback to support (20%)
        close = df['close'].iloc[-1]
        ema21 = df['ema21'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]
        
        near_ema21 = abs(close - ema21) / ema21 < 0.005  # Within 0.5%
        near_bb_lower = close < (bb_lower * 1.02)
        
        if near_ema21 or near_bb_lower:
            score += weights['context'] * 0.8
            details['context'] = 'pullback_detected'
        
        # 3. Momentum (20%)
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        
        rsi_ok = 30 < rsi < self.config['rsi_ob']
        macd_positive = macd_hist > 0
        
        momentum_score = 0
        if rsi_ok:
            momentum_score += 0.5
        if macd_positive:
            momentum_score += 0.5
        
        score += weights['momentum'] * momentum_score
        details['momentum'] = f"rsi={rsi:.1f}, macd_hist={macd_hist:.4f}"
        
        # 4. Microstructure (20%)
        obi = microstructure.get('obi', 0)
        spread_pct = microstructure.get('spread_pct', 999)
        orderflow_delta_pct = microstructure.get('orderflow_delta_pct', 0)
        
        micro_score = 0
        if obi > self.config['obi_threshold']:  # Positive OBI
            micro_score += 0.4
        if spread_pct < 0.1:  # Tight spread
            micro_score += 0.3
        if orderflow_delta_pct > 10:  # Buy pressure
            micro_score += 0.3
        
        score += weights['microstructure'] * micro_score
        details['microstructure'] = f"obi={obi:.2f}, spread={spread_pct:.3f}%"
        
        # 5. Multi-timeframe (15%)
        if higher_tf_trend == 'bullish':
            score += weights['multi_tf']
            details['multi_tf'] = 'aligned'
        else:
            details['multi_tf'] = 'not_aligned'
        
        return min(score, 1.0), details
    
    def score_short_signal(self, df: pd.DataFrame, microstructure: Dict,
                           higher_tf_trend: str = 'bearish') -> Tuple[float, Dict]:
        """
        Score short signal quality (0-1).
        Returns: (confidence_score, signal_details)
        """
        score = 0.0
        weights = {
            'trend': 0.25,
            'context': 0.20,
            'momentum': 0.20,
            'microstructure': 0.20,
            'multi_tf': 0.15
        }
        details = {}
        
        # 1. Trend alignment
        trend = self.check_trend_direction(df)
        if trend == 'bearish':
            score += weights['trend']
            details['trend'] = 'aligned'
        else:
            details['trend'] = 'not_aligned'
        
        # 2. Context: pullback to resistance
        close = df['close'].iloc[-1]
        ema21 = df['ema21'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        
        near_ema21 = abs(close - ema21) / ema21 < 0.005
        near_bb_upper = close > (bb_upper * 0.98)
        
        if near_ema21 or near_bb_upper:
            score += weights['context'] * 0.8
            details['context'] = 'pullback_detected'
        
        # 3. Momentum
        rsi = df['rsi'].iloc[-1]
        macd_hist = df['macd_hist'].iloc[-1]
        
        rsi_ok = self.config['rsi_os'] < rsi < 70
        macd_negative = macd_hist < 0
        
        momentum_score = 0
        if rsi_ok:
            momentum_score += 0.5
        if macd_negative:
            momentum_score += 0.5
        
        score += weights['momentum'] * momentum_score
        details['momentum'] = f"rsi={rsi:.1f}, macd_hist={macd_hist:.4f}"
        
        # 4. Microstructure
        obi = microstructure.get('obi', 0)
        spread_pct = microstructure.get('spread_pct', 999)
        orderflow_delta_pct = microstructure.get('orderflow_delta_pct', 0)
        
        micro_score = 0
        if obi < -self.config['obi_threshold']:  # Negative OBI
            micro_score += 0.4
        if spread_pct < 0.1:
            micro_score += 0.3
        if orderflow_delta_pct < -10:  # Sell pressure
            micro_score += 0.3
        
        score += weights['microstructure'] * micro_score
        details['microstructure'] = f"obi={obi:.2f}, spread={spread_pct:.3f}%"
        
        # 5. Multi-timeframe
        if higher_tf_trend == 'bearish':
            score += weights['multi_tf']
            details['multi_tf'] = 'aligned'
        else:
            details['multi_tf'] = 'not_aligned'
        
        return min(score, 1.0), details
    
    def generate_signal(self, symbol: str, df: pd.DataFrame, microstructure: Dict,
                        higher_tf_trend: str = 'neutral') -> Optional[Dict]:
        """
        Generate trading signal if conditions met.
        Returns signal dict or None.
        """
        # Pre-filters
        regime = self.check_regime(df)
        adx = df['adx'].iloc[-1]
        volume = df['volume'].iloc[-1]
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        spread_pct = microstructure.get('spread_pct', 999)
        
        # Filter: regime and volatility
        if adx < self.config['adx_min']:
            logger.debug(f"{symbol}: ADX too low ({adx:.1f})")
            return None
        
        if volume < volume_ma * self.config['vol_mult_min']:
            logger.debug(f"{symbol}: Volume too low")
            return None
        
        if bb_width < 0.02:  # Very tight Bollinger Bands
            logger.debug(f"{symbol}: Low volatility")
            return None
        
        if spread_pct > 0.2:  # Spread too wide
            logger.debug(f"{symbol}: Spread too wide ({spread_pct:.3f}%)")
            return None
        
        # Score long and short
        long_score, long_details = self.score_long_signal(df, microstructure, higher_tf_trend)
        short_score, short_details = self.score_short_signal(df, microstructure, higher_tf_trend)
        
        # Check threshold
        if long_score >= self.config['conf_threshold'] and long_score > short_score:
            return {
                'symbol': symbol,
                'direction': 'long',
                'confidence': long_score,
                'details': long_details,
                'entry_price': df['close'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'timestamp': df.index[-1]
            }
        
        elif short_score >= self.config['conf_threshold'] and short_score > long_score:
            return {
                'symbol': symbol,
                'direction': 'short',
                'confidence': short_score,
                'details': short_details,
                'entry_price': df['close'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'timestamp': df.index[-1]
            }
        
        return None
