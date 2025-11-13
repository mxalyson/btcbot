"""Technical indicators using pandas."""
import pandas as pd
import numpy as np
from typing import Tuple

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = atr(high, low, close, 1)
    atr_period = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_period)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_period)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    obv = pd.Series(0, index=close.index)
    obv = obv.astype(float)  # Convert to float first
    obv.iloc[0] = float(volume.iloc[0])
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate all technical indicators."""
    # EMAs
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    
    # RSI
    df['rsi'] = rsi(df['close'], config['rsi_period'])
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    
    # ATR
    df['atr'] = atr(df['high'], df['low'], df['close'])
    
    # ADX
    df['adx'] = adx(df['high'], df['low'], df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # VWAP
    df['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # OBV
    df['obv'] = obv(df['close'], df['volume'])
    
    return df
