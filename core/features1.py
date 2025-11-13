"""
Feature Store - Consolidates indicators, microstructure, and price action.
Prepares features for signal generation and backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from core.indicators import calculate_all_indicators
from core.microstructure import MicrostructureAnalyzer
from core.structure_pa import PriceActionAnalyzer

logger = logging.getLogger("TradingBot.Features")


class FeatureStore:
    """Feature engineering and consolidation."""

    def __init__(self, config: dict):
        """
        Initialize feature store.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.micro_analyzer = MicrostructureAnalyzer(config)
        self.pa_analyzer = PriceActionAnalyzer(config)

    def calculate_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate base technical indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators
        """
        logger.debug("Calculating base technical indicators...")

        df = calculate_all_indicators(df, self.config)

        return df

    def add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price action structure features.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            DataFrame with PA features
        """
        logger.debug("Adding price action features...")

        # Full PA analysis
        df = self.pa_analyzer.analyze(df)

        # Distance to S/R
        sr_levels = self.pa_analyzer.find_support_resistance(df)
        current_price = df['close'].iloc[-1]

        # Find nearest support/resistance
        nearest_support = None
        nearest_resistance = None

        if sr_levels['support']:
            supports_below = [s for s in sr_levels['support'] if s < current_price]
            if supports_below:
                nearest_support = max(supports_below)

        if sr_levels['resistance']:
            resistances_above = [r for r in sr_levels['resistance'] if r > current_price]
            if resistances_above:
                nearest_resistance = min(resistances_above)

        # Distance features
        df['dist_to_support_pct'] = np.nan
        df['dist_to_resistance_pct'] = np.nan

        if nearest_support:
            df['dist_to_support_pct'] = ((df['close'] - nearest_support) / nearest_support) * 100

        if nearest_resistance:
            df['dist_to_resistance_pct'] = ((nearest_resistance - df['close']) / df['close']) * 100

        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived/engineered features.

        Args:
            df: DataFrame with base features

        Returns:
            DataFrame with derived features
        """
        logger.debug("Adding derived features...")

        # Momentum features
        df['price_momentum_5'] = df['close'].pct_change(5) * 100
        df['price_momentum_10'] = df['close'].pct_change(10) * 100
        df['price_momentum_20'] = df['close'].pct_change(20) * 100

        # Volatility features
        df['returns'] = df['close'].pct_change()
        df['volatility_10'] = df['returns'].rolling(10).std() * 100
        df['volatility_20'] = df['returns'].rolling(20).std() * 100

        # Volume features
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_momentum'] = df['volume'].pct_change(5) * 100

        # Price vs EMAs
        df['price_vs_ema21'] = ((df['close'] - df['ema21']) / df['ema21']) * 100
        df['price_vs_ema50'] = ((df['close'] - df['ema50']) / df['ema50']) * 100
        df['price_vs_ema200'] = ((df['close'] - df['ema200']) / df['ema200']) * 100

        # EMA crossovers
        df['ema21_vs_ema50'] = df['ema21'] > df['ema50']
        df['ema50_vs_ema200'] = df['ema50'] > df['ema200']

        # Bollinger Band position
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # RSI zones
        df['rsi_zone'] = pd.cut(df['rsi'], 
                                bins=[0, 30, 40, 60, 70, 100],
                                labels=['oversold', 'weak', 'neutral', 'strong', 'overbought'])

        # ADX strength
        df['adx_zone'] = pd.cut(df['adx'],
                               bins=[0, 20, 25, 40, 100],
                               labels=['no_trend', 'weak_trend', 'trend', 'strong_trend'])

        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']

        df['body_ratio'] = df['body_size'] / df['total_range']
        df['is_bullish_candle'] = df['close'] > df['open']

        # Doji detection
        df['is_doji'] = df['body_ratio'] < 0.1

        # Hammer/Shooting star
        df['is_hammer'] = (df['lower_wick'] > df['body_size'] * 2) &                           (df['upper_wick'] < df['body_size']) &                           df['is_bullish_candle']

        df['is_shooting_star'] = (df['upper_wick'] > df['body_size'] * 2) &                                  (df['lower_wick'] < df['body_size']) &                                  (~df['is_bullish_candle'])

        return df

    def add_multi_timeframe_features(self, 
                                    primary_df: pd.DataFrame,
                                    higher_tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Add multi-timeframe alignment features.

        Args:
            primary_df: Primary timeframe DataFrame
            higher_tf_data: Dict of higher timeframe DataFrames

        Returns:
            Primary DataFrame with MTF features
        """
        logger.debug("Adding multi-timeframe features...")

        primary_df = primary_df.copy()

        for tf, df_higher in higher_tf_data.items():
            if df_higher.empty:
                continue

            # Ensure higher TF has indicators
            if 'ema50' not in df_higher.columns:
                df_higher = calculate_all_indicators(df_higher, self.config)

            # Get latest values from higher TF
            latest_higher = df_higher.iloc[-1]

            # Add as constant features (will be same for all primary TF rows in window)
            primary_df[f'{tf}_trend'] = 'bullish' if latest_higher['ema50'] > latest_higher['ema200'] else 'bearish'
            primary_df[f'{tf}_rsi'] = latest_higher['rsi']
            primary_df[f'{tf}_adx'] = latest_higher['adx']
            primary_df[f'{tf}_price_momentum'] = latest_higher['close'] - latest_higher['open']

        return primary_df

    def add_microstructure_features(self, df: pd.DataFrame,
                                   orderbook: Optional[Dict] = None,
                                   trades: Optional[List] = None) -> pd.DataFrame:
        """
        Add microstructure features from orderbook and trades.

        Args:
            df: DataFrame with OHLCV data
            orderbook: Current orderbook snapshot
            trades: Recent trades list

        Returns:
            DataFrame with microstructure features
        """
        if orderbook is None or trades is None:
            logger.debug("No live microstructure data, skipping...")
            return df

        logger.debug("Adding microstructure features...")

        df = df.copy()

        # Analyze current microstructure
        from core.microstructure import calculate_obi, calculate_spread, calculate_orderflow

        bids = orderbook.get('bids', {})
        asks = orderbook.get('asks', {})

        obi = calculate_obi(bids, asks, self.config['obi_levels'])
        spread_abs, spread_pct, mid_price = calculate_spread(bids, asks)
        orderflow = calculate_orderflow(trades, self.config['orderflow_lookback'])

        # Add to last row only (current market state)
        df.loc[df.index[-1], 'obi'] = obi
        df.loc[df.index[-1], 'spread_abs'] = spread_abs
        df.loc[df.index[-1], 'spread_pct'] = spread_pct
        df.loc[df.index[-1], 'orderflow_delta'] = orderflow['delta']
        df.loc[df.index[-1], 'orderflow_delta_pct'] = orderflow['delta_pct']

        # Forward fill for consistency
        df['obi'] = df['obi'].fillna(method='ffill')
        df['spread_pct'] = df['spread_pct'].fillna(method='ffill')
        df['orderflow_delta_pct'] = df['orderflow_delta_pct'].fillna(method='ffill')

        return df

    def normalize_features(self, df: pd.DataFrame, 
                          method: str = 'zscore',
                          window: int = 50) -> pd.DataFrame:
        """
        Normalize/standardize features.

        Args:
            df: DataFrame with features
            method: 'zscore', 'minmax', or 'robust'
            window: Rolling window for normalization

        Returns:
            DataFrame with normalized features
        """
        logger.debug(f"Normalizing features using {method}...")

        df = df.copy()

        # Select numeric columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Exclude certain columns from normalization
        exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        cols_to_normalize = [col for col in numeric_cols if col not in exclude]

        for col in cols_to_normalize:
            if df[col].isna().all():
                continue

            if method == 'zscore':
                # Z-score normalization (rolling)
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                df[f'{col}_norm'] = (df[col] - mean) / std

            elif method == 'minmax':
                # Min-Max normalization (rolling)
                min_val = df[col].rolling(window).min()
                max_val = df[col].rolling(window).max()
                df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)

            elif method == 'robust':
                # Robust scaling using median and IQR
                median = df[col].rolling(window).median()
                q75 = df[col].rolling(window).quantile(0.75)
                q25 = df[col].rolling(window).quantile(0.25)
                iqr = q75 - q25
                df[f'{col}_norm'] = (df[col] - median) / iqr

        return df

    def create_feature_vector(self, df: pd.DataFrame, 
                             include_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create final feature vector for ML/signal generation.

        Args:
            df: DataFrame with all features
            include_columns: Specific columns to include (None = all)

        Returns:
            DataFrame with selected features
        """
        if include_columns is None:
            # Default feature set
            include_columns = [
                'close', 'volume',
                'ema21', 'ema50', 'ema200',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'atr', 'adx', 'bb_width',
                'price_momentum_5', 'price_momentum_10',
                'volatility_10', 'volume_ratio',
                'price_vs_ema21', 'price_vs_ema50',
                'bb_position', 'body_ratio',
                'dist_to_support_pct', 'dist_to_resistance_pct'
            ]

        # Filter to existing columns
        available_cols = [col for col in include_columns if col in df.columns]

        feature_df = df[available_cols].copy()

        return feature_df

    def build_features(self, df: pd.DataFrame,
                      higher_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                      orderbook: Optional[Dict] = None,
                      trades: Optional[List] = None,
                      normalize: bool = False) -> pd.DataFrame:
        """
        Build complete feature set.

        Args:
            df: Base OHLCV DataFrame
            higher_tf_data: Higher timeframe data for MTF features
            orderbook: Current orderbook for microstructure
            trades: Recent trades for orderflow
            normalize: Apply normalization

        Returns:
            DataFrame with complete feature set
        """
        logger.info("Building complete feature set...")

        # 1. Base technical indicators
        df = self.calculate_base_features(df)

        # 2. Price action features
        df = self.add_price_action_features(df)

        # 3. Derived features
        df = self.add_derived_features(df)

        # 4. Multi-timeframe features
        if higher_tf_data:
            df = self.add_multi_timeframe_features(df, higher_tf_data)

        # 5. Microstructure features (live only)
        if orderbook and trades:
            df = self.add_microstructure_features(df, orderbook, trades)

        # 6. Normalize if requested
        if normalize:
            df = self.normalize_features(df, method='zscore', window=50)

        logger.info(f"Feature set complete: {len(df.columns)} columns")

        return df

    def get_feature_importance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate feature importance using correlation with returns.

        Args:
            df: DataFrame with features

        Returns:
            Dict mapping feature to importance score
        """
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()

        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'returns']]

        importance = {}

        for col in feature_cols:
            if df[col].isna().all():
                continue

            # Calculate absolute correlation with returns
            corr = abs(df[col].corr(df['returns']))
            if not np.isnan(corr):
                importance[col] = corr

        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def validate_features(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality.

        Args:
            df: DataFrame with features

        Returns:
            Dict with validation results
        """
        results = {
            'total_features': len(df.columns),
            'total_rows': len(df),
            'missing_values': {},
            'infinite_values': {},
            'constant_features': [],
            'high_correlation_pairs': []
        }

        # Check for missing values
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 0:
                results['missing_values'][col] = f"{missing_pct:.2f}%"

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results['infinite_values'][col] = inf_count

        # Check for constant features
        for col in numeric_cols:
            if df[col].nunique() == 1:
                results['constant_features'].append(col)

        # Check for high correlation (multicollinearity)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            high_corr = []
            for column in upper_triangle.columns:
                for index in upper_triangle.index:
                    if upper_triangle.loc[index, column] > 0.95:
                        high_corr.append((index, column, upper_triangle.loc[index, column]))

            results['high_correlation_pairs'] = high_corr[:10]  # Top 10

        return results
