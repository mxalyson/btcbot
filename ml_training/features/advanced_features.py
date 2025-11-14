"""
🚀 FEATURES AVANÇADAS PARA SCALPING - V2.0

Focado em Order Flow, Microestrutura e Regime Detection
Otimizado para scalping 15min com latência mínima

Author: Claude
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ScalpingFeatureEngineer:
    """
    Feature Engineering otimizado para scalping de alta frequência.

    Categorias:
    1. Price Action (alta frequência)
    2. Order Flow & Microestrutura
    3. Volume Profile
    4. Regime Detection
    5. Time-Based
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def build_all_features(self, df: pd.DataFrame,
                          orderbook: Optional[Dict] = None,
                          trades: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Cria TODAS as features otimizadas para scalping.

        Args:
            df: DataFrame com OHLCV
            orderbook: Order book data (opcional mas recomendado)
            trades: Trades tick-by-tick (opcional mas recomendado)

        Returns:
            DataFrame com todas as features
        """
        df_feat = df.copy()

        print("🔨 Building scalping features...")

        # 1. Price Action de alta frequência
        df_feat = self.add_price_action_features(df_feat)
        print("  ✅ Price action features")

        # 2. Momentum multi-período
        df_feat = self.add_momentum_features(df_feat)
        print("  ✅ Momentum features")

        # 3. Volatility features
        df_feat = self.add_volatility_features(df_feat)
        print("  ✅ Volatility features")

        # 4. Volume features
        df_feat = self.add_volume_features(df_feat)
        print("  ✅ Volume features")

        # 5. Regime detection
        df_feat = self.add_regime_features(df_feat)
        print("  ✅ Regime detection features")

        # 6. Time-based features
        df_feat = self.add_time_features(df_feat)
        print("  ✅ Time-based features")

        # 7. Order flow (se disponível)
        if orderbook is not None:
            df_feat = self.add_orderbook_features(df_feat, orderbook)
            print("  ✅ Order flow features")

        # 8. Trade flow (se disponível)
        if trades is not None:
            df_feat = self.add_trade_flow_features(df_feat, trades)
            print("  ✅ Trade flow features")

        print(f"✅ Total features: {len(df_feat.columns)}")

        return df_feat

    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de price action de curto prazo (scalping).
        """
        # Mudança de preço curto prazo
        for period in [1, 2, 3, 5]:
            df[f'price_change_{period}'] = df['close'].diff(period)
            df[f'price_change_pct_{period}'] = df['close'].pct_change(period) * 100

        # Rate of Change
        for period in [1, 3, 5]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                    df['close'].shift(period)) * 100

        # High-Low ratio (volatility intrabar)
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['high_low_pct'] = ((df['high'] - df['low']) / df['close']) * 100

        # Body to range ratio
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_to_range'] = df['body_size'] / (df['high'] - df['low'] + 1e-10)
        df['body_pct'] = (df['body_size'] / df['open']) * 100

        # Wick analysis (rejeição)
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / (df['high'] - df['low'] + 1e-10)
        df['lower_wick_pct'] = df['lower_wick'] / (df['high'] - df['low'] + 1e-10)

        # Wick dominance
        df['wick_dominance'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 1e-10)

        # Candle type
        df['is_green'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (df['body_to_range'] < 0.1).astype(int)
        df['is_hammer'] = ((df['lower_wick_pct'] > 0.6) &
                           (df['body_to_range'] < 0.3)).astype(int)
        df['is_shooting_star'] = ((df['upper_wick_pct'] > 0.6) &
                                   (df['body_to_range'] < 0.3)).astype(int)

        # Price position in range
        df['close_position_in_range'] = ((df['close'] - df['low']) /
                                          (df['high'] - df['low'] + 1e-10))

        return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Momentum de múltiplos períodos.
        """
        # Retornos
        for period in [1, 2, 3, 5, 8, 13, 21]:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100

        # Momentum aceleração
        df['momentum_acc_3'] = df['close'].diff(2) - df['close'].diff(1)
        df['momentum_acc_5'] = df['close'].diff(3) - df['close'].diff(2)

        # Momentum strength
        for period in [5, 10, 20]:
            df[f'momentum_strength_{period}'] = (
                df['close'].rolling(period).apply(
                    lambda x: (x[-1] - x[0]) / (x.max() - x.min() + 1e-10)
                )
            )

        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volatility de curto e médio prazo.
        """
        # Volatility histórica
        for period in [3, 5, 10, 20]:
            df[f'volatility_{period}'] = df['return_1'].rolling(period).std()

        # Parkinson volatility (usa high/low)
        for period in [5, 10, 20]:
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) *
                (np.log(df['high'] / df['low']) ** 2).rolling(period).mean()
            )

        # ATR normalized
        if 'atr' in df.columns:
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            df['atr_zscore'] = (df['atr'] - df['atr'].rolling(50).mean()) / (
                df['atr'].rolling(50).std() + 1e-10
            )

        # Volatility regime
        df['volatility_5_zscore'] = (
            (df['volatility_5'] - df['volatility_5'].rolling(50).mean()) /
            (df['volatility_5'].rolling(50).std() + 1e-10)
        )

        return df

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features.
        """
        # Volume ratios
        for period in [3, 5, 10, 20]:
            df[f'volume_ratio_{period}'] = (
                df['volume'] / (df['volume'].rolling(period).mean() + 1e-10)
            )

        # Volume momentum
        for period in [3, 5]:
            df[f'volume_momentum_{period}'] = df['volume'].pct_change(period) * 100

        # Volume surge detection
        df['volume_surge'] = (
            df['volume'] > df['volume'].rolling(20).mean() * 1.5
        ).astype(int)

        # Volume trend
        df['volume_trend_5'] = df['volume'].rolling(5).apply(
            lambda x: 1 if x[-1] > x[0] else -1
        )

        # Volume weighted price
        df['vwap_simple'] = (
            (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() /
            df['volume'].cumsum()
        )
        df['distance_from_vwap'] = ((df['close'] - df['vwap_simple']) /
                                     df['vwap_simple']) * 100

        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta regime de mercado (trending vs ranging).
        """
        # ADX-based trend strength
        if 'adx' in df.columns:
            df['adx_slope'] = df['adx'].diff(3)
            df['is_trending'] = (df['adx'] > 25).astype(int)
            df['is_strong_trend'] = (df['adx'] > 40).astype(int)

        # EMA slopes
        if 'ema50' in df.columns and 'ema200' in df.columns:
            df['ema50_slope'] = df['ema50'].diff(5)
            df['ema200_slope'] = df['ema200'].diff(10)
            df['trend_strength'] = (
                (df['ema50'] - df['ema200']) / df['ema200']
            ) * 100
            df['ema_aligned'] = (
                (df['ema50'] > df['ema200']).astype(int) * 2 - 1
            )  # 1 = bullish, -1 = bearish

        # Choppiness Index
        if 'atr' in df.columns:
            for period in [14, 21]:
                high_low_range = (
                    df['high'].rolling(period).max() -
                    df['low'].rolling(period).min()
                )
                atr_sum = df['atr'].rolling(period).sum()
                df[f'choppiness_{period}'] = (
                    100 * np.log10(atr_sum / high_low_range) / np.log10(period)
                )
                # < 38.2 = trending, > 61.8 = ranging
                df[f'is_choppy_{period}'] = (
                    df[f'choppiness_{period}'] > 61.8
                ).astype(int)

        # Bollinger Band width (ranging indicator)
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_width_pct'] = (
                df['bb_width'] / df['bb_width'].rolling(50).mean()
            )
            df['is_ranging'] = (
                df['bb_width_pct'] < 0.7
            ).astype(int)

        # Price consolidation
        for period in [10, 20]:
            high_range = df['high'].rolling(period).max()
            low_range = df['low'].rolling(period).min()
            df[f'consolidation_{period}'] = (
                (high_range - low_range) / df['close']
            ) * 100

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based features para crypto.
        """
        # Extract time components
        df['hour'] = pd.to_datetime(df.index).hour
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['day_of_month'] = pd.to_datetime(df.index).day

        # Crypto trading sessions (UTC)
        # Asia: 00-08, Europe: 08-16, US: 16-24
        df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        # Session overlaps (high liquidity)
        df['is_asia_europe_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 9)).astype(int)
        df['is_europe_us_overlap'] = ((df['hour'] >= 15) & (df['hour'] < 17)).astype(int)

        # High volume hours (crypto peaks)
        df['is_high_volume_hour'] = df['hour'].isin([8, 9, 14, 15, 16, 17]).astype(int)

        # Weekend effect
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        return df

    def add_orderbook_features(self, df: pd.DataFrame,
                              orderbook: Dict) -> pd.DataFrame:
        """
        Features de microestrutura do order book.

        Args:
            orderbook: Dict com 'bids' e 'asks'
                {
                    'bids': [[price, size], ...],  # Ordenado desc
                    'asks': [[price, size], ...]   # Ordenado asc
                }
        """
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return df

        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])

        if len(bids) == 0 or len(asks) == 0:
            return df

        # Best bid/ask
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        # Spread
        spread = best_ask - best_bid
        spread_pct = (spread / mid_price) * 100
        df.loc[df.index[-1], 'spread'] = spread
        df.loc[df.index[-1], 'spread_pct'] = spread_pct

        # Order Book Imbalance (OBI)
        # Top 5 levels
        bid_vol_5 = bids[:5, 1].sum()
        ask_vol_5 = asks[:5, 1].sum()
        total_vol_5 = bid_vol_5 + ask_vol_5
        if total_vol_5 > 0:
            obi_5 = (bid_vol_5 - ask_vol_5) / total_vol_5
            df.loc[df.index[-1], 'obi_top5'] = obi_5

        # Top 10 levels
        if len(bids) >= 10 and len(asks) >= 10:
            bid_vol_10 = bids[:10, 1].sum()
            ask_vol_10 = asks[:10, 1].sum()
            total_vol_10 = bid_vol_10 + ask_vol_10
            if total_vol_10 > 0:
                obi_10 = (bid_vol_10 - ask_vol_10) / total_vol_10
                df.loc[df.index[-1], 'obi_top10'] = obi_10

        # Liquidity depth
        df.loc[df.index[-1], 'bid_depth'] = bid_vol_5
        df.loc[df.index[-1], 'ask_depth'] = ask_vol_5
        df.loc[df.index[-1], 'total_depth'] = total_vol_5

        # Liquidity imbalance
        if total_vol_5 > 0:
            liquidity_imbalance = (bid_vol_5 - ask_vol_5) / total_vol_5
            df.loc[df.index[-1], 'liquidity_imbalance'] = liquidity_imbalance

        # Spread volatility (se histórico disponível)
        if 'spread_pct' in df.columns:
            df['spread_volatility'] = df['spread_pct'].rolling(10).std()

        return df

    def add_trade_flow_features(self, df: pd.DataFrame,
                                trades: pd.DataFrame) -> pd.DataFrame:
        """
        Features baseadas em trades executados (tick data).

        Args:
            trades: DataFrame com colunas ['timestamp', 'price', 'size', 'side']
                    side: 'buy' ou 'sell' (taker side)
        """
        if trades is None or len(trades) == 0:
            return df

        # Agrega trades por candle
        # Assumindo que trades tem timestamp matching com df.index

        # Buy vs Sell volume
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']

        buy_volume = buy_trades.groupby(trades.index)['size'].sum()
        sell_volume = sell_trades.groupby(trades.index)['size'].sum()

        df['buy_volume'] = buy_volume
        df['sell_volume'] = sell_volume
        df['buy_volume'] = df['buy_volume'].fillna(0)
        df['sell_volume'] = df['sell_volume'].fillna(0)

        # Buy volume ratio
        total_trade_vol = df['buy_volume'] + df['sell_volume']
        df['buy_volume_ratio'] = df['buy_volume'] / (total_trade_vol + 1e-10)

        # Delta volume (buy - sell)
        df['delta_volume'] = df['buy_volume'] - df['sell_volume']
        df['cumulative_delta'] = df['delta_volume'].cumsum()

        # Aggressive trades (market orders)
        df['aggressive_trades'] = trades.groupby(trades.index).size()
        df['aggressive_buy_pct'] = (
            buy_trades.groupby(trades.index).size() /
            (df['aggressive_trades'] + 1e-10)
        )

        return df


def create_legacy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features do modelo antigo para compatibilidade.

    Mantém features que já funcionam bem.
    """
    df_feat = df.copy()

    # Momentum multi-período
    for period in [3, 5, 8, 13, 21]:
        df_feat[f'momentum_{period}'] = df_feat['close'].pct_change(period) * 100
        df_feat[f'volume_ratio_{period}'] = (
            df_feat['volume'] / df_feat['volume'].rolling(period).mean()
        )

    # Trend strength
    if 'ema50' in df_feat.columns and 'ema200' in df_feat.columns:
        df_feat['trend_strength'] = (
            (df_feat['ema50'] - df_feat['ema200']) / df_feat['ema200']
        ) * 100

    # Volatility regime
    if 'atr' in df_feat.columns:
        df_feat['volatility_regime'] = (
            df_feat['atr'] / df_feat['atr'].rolling(50).mean()
        )

    # Price position
    df_feat['price_position'] = (
        (df_feat['close'] - df_feat['low'].rolling(20).min()) /
        (df_feat['high'].rolling(20).max() - df_feat['low'].rolling(20).min())
    ).fillna(0.5)

    # Volume & price dynamics
    df_feat['volume_momentum'] = df_feat['volume'].pct_change(5)
    df_feat['price_acceleration'] = (
        df_feat['close'].diff(2) - df_feat['close'].diff(1)
    )

    return df_feat


if __name__ == "__main__":
    """
    Teste rápido do feature engineering.
    """
    # Dados de exemplo
    dates = pd.date_range('2024-01-01', periods=1000, freq='15min')
    df_test = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000),
        'atr': np.random.uniform(0.5, 2.0, 1000),
        'adx': np.random.uniform(10, 50, 1000),
        'ema50': np.random.randn(1000).cumsum() + 100,
        'ema200': np.random.randn(1000).cumsum() + 100,
        'bb_upper': np.random.randn(1000).cumsum() + 102,
        'bb_middle': np.random.randn(1000).cumsum() + 100,
        'bb_lower': np.random.randn(1000).cumsum() + 98,
    }, index=dates)

    # Feature engineering
    engineer = ScalpingFeatureEngineer()
    df_with_features = engineer.build_all_features(df_test)

    print(f"\n✅ Feature engineering test complete!")
    print(f"📊 Original columns: {len(df_test.columns)}")
    print(f"📊 Total columns after features: {len(df_with_features.columns)}")
    print(f"📊 New features added: {len(df_with_features.columns) - len(df_test.columns)}")
    print(f"\n🔍 Sample features:")
    new_cols = [c for c in df_with_features.columns if c not in df_test.columns]
    for col in new_cols[:20]:
        print(f"   - {col}")
    if len(new_cols) > 20:
        print(f"   ... and {len(new_cols) - 20} more")
