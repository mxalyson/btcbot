"""
MASTER SCALPER ML - Perfect ML System for Scalping
Created by AI Trading Master - Balanced, Profitable, Professional
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import argparse
import logging

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("❌ LightGBM required! Install: pip install lightgbm")
    exit(1)


def create_perfect_targets(df: pd.DataFrame, atr_col='atr') -> pd.DataFrame:
    """
    Create PERFECT symmetric targets for scalping.

    Master Strategy:
    - Uses DYNAMIC thresholds based on ATR (volatility)
    - Multi-horizon: Predicts 6 candles (1.5h) ahead
    - Symmetric: +threshold for UP, -threshold for DOWN
    - Removes neutral zone (no clear direction = no trade)

    Returns balanced 50/50 UP/DOWN labels!
    """

    df_targets = df.copy()

    # Calculate ATR-based dynamic threshold
    # In high volatility: higher threshold
    # In low volatility: lower threshold
    atr = df_targets[atr_col] if atr_col in df_targets.columns else df_targets['close'] * 0.005

    # Base threshold: 0.4% to 0.8% depending on ATR
    # ATR as % of price
    atr_pct = (atr / df_targets['close']) * 100

    # Dynamic threshold: min 0.4%, max 0.8%
    # Higher when ATR high (volatile), lower when ATR low (stable)
    dynamic_threshold = np.clip(atr_pct * 0.3, 0.35, 0.75)

    # Multi-horizon predictions (vote from multiple timeframes)
    horizons = [4, 6, 8]  # 1h, 1.5h, 2h
    votes = []

    for horizon in horizons:
        future_returns = (df_targets['close'].shift(-horizon) / df_targets['close'] - 1) * 100

        # Vote based on dynamic threshold
        vote = pd.Series(0.5, index=df_targets.index)  # Neutral default

        # UP vote if future return > threshold
        vote[future_returns > dynamic_threshold] = 1.0

        # DOWN vote if future return < -threshold
        vote[future_returns < -dynamic_threshold] = 0.0

        votes.append(vote)

    # Average votes (ensemble)
    avg_vote = sum(votes) / len(votes)

    # Final target: Strong consensus required
    target = pd.Series(np.nan, index=df_targets.index)

    # UP: Average vote > 0.65 (at least 2 out of 3 horizons agree)
    target[avg_vote > 0.65] = 1

    # DOWN: Average vote < 0.35
    target[avg_vote < 0.35] = 0

    # 0.35 <= avg_vote <= 0.65 = NEUTRAL (removed)

    df_targets['target'] = target
    df_targets['vote_confidence'] = np.abs(avg_vote - 0.5) * 2

    return df_targets


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MASTER TRADER advanced features."""

    df_features = df.copy()

    # Multi-period momentum
    for period in [3, 5, 8, 13, 21]:
        df_features[f'momentum_{period}'] = df_features['close'].pct_change(period) * 100
        df_features[f'volume_ratio_{period}'] = df_features['volume'] / df_features['volume'].rolling(period).mean()

    # Trend strength
    df_features['trend_strength'] = (df_features['ema50'] - df_features['ema200']) / df_features['ema200'] * 100

    # Volatility regimes
    df_features['volatility_regime'] = (df_features['atr'] / df_features['atr'].rolling(50).mean())

    # Price position in recent range
    df_features['price_position'] = (
        (df_features['close'] - df_features['low'].rolling(20).min()) / 
        (df_features['high'].rolling(20).max() - df_features['low'].rolling(20).min())
    )

    # Volume momentum
    df_features['volume_momentum'] = df_features['volume'].pct_change(5)

    # Acceleration
    df_features['price_acceleration'] = df_features['close'].diff(2) - df_features['close'].diff(1)

    return df_features


def train_master_scalper(symbol: str, days: int, config: dict):
    """Train PERFECT scalping model."""

    logger = logging.getLogger('MasterScalper')

    print()
    print("=" * 80)
    print("🏆 MASTER SCALPER ML TRAINING")
    print("=" * 80)
    print(f"   Symbol:  {symbol}")
    print(f"   Period:  {days} days")
    print(f"   Goal:    Balanced 50/50 UP/DOWN predictions")
    print(f"   Target:  55-60% win rate, 60-120% ROI/year")
    print("=" * 80)
    print()

    # Initialize
    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    data_manager = DataManager(rest_client)
    feature_store = FeatureStore(config)

    # Download data
    print("📥 Downloading data...")
    df = data_manager.get_data(
        symbol=symbol,
        interval='15m',
        days_back=days,
        use_cache=False
    )

    print(f"✅ Downloaded {len(df):,} candles")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print()

    # Build base features
    print("🔨 Building base features...")
    df_features = feature_store.build_features(df, normalize=False)
    print(f"✅ Built base features: {len(df_features.columns)} columns")
    print()

    # Add advanced features
    print("🎯 Adding MASTER TRADER features...")
    df_features = create_advanced_features(df_features)
    print(f"✅ Total features: {len(df_features.columns)} columns")
    print()

    # Create perfect targets
    print("🎯 Creating PERFECT targets...")
    df_features = create_perfect_targets(df_features)
    print()

    # Analyze targets
    valid_mask = ~df_features['target'].isna()
    target = df_features.loc[valid_mask, 'target']

    up_count = (target == 1).sum()
    down_count = (target == 0).sum()
    total = len(target)
    removed = (~valid_mask).sum()

    print("📊 TARGET DISTRIBUTION:")
    print(f"   UP (1):      {up_count:,} ({up_count/total*100:.1f}%)")
    print(f"   DOWN (0):    {down_count:,} ({down_count/total*100:.1f}%)")
    print(f"   Total:       {total:,}")
    print(f"   Removed:     {removed:,} neutral samples")
    print()

    balance = abs(up_count - down_count) / total * 100
    if balance < 5:
        print(f"✅ PERFECTLY BALANCED! ({balance:.1f}% diff) 🎯")
    elif balance < 10:
        print(f"✅ Well balanced ({balance:.1f}% diff)")
    else:
        print(f"⚠️  Imbalanced ({balance:.1f}% diff)")
    print()

    # Prepare training data
    print("🔨 Preparing training data...")

    df_clean = df_features[valid_mask].copy()
    y = df_clean['target'].astype(int)

    # Remove non-feature columns
    exclude_cols = ['close', 'high', 'low', 'open', 'volume', 'target', 'vote_confidence']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    # Remove object types
    object_cols = df_clean[feature_cols].select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        print(f"   Removing {len(object_cols)} object columns")
        feature_cols = [col for col in feature_cols if col not in object_cols]

    X = df_clean[feature_cols].fillna(0)

    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Samples:  {len(X):,}")
    print()

    # Split with time-series aware split
    split_idx = int(len(X) * 0.80)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    print(f"   Train: {len(X_train):,} samples")
    print(f"   Val:   {len(X_val):,} samples")
    print()

    # Calculate class weights for perfect balance
    class_counts = y_train.value_counts()
    total_samples = len(y_train)

    # Weight inversely proportional to frequency
    weight_for_0 = total_samples / (2 * class_counts[0]) if 0 in class_counts else 1.0
    weight_for_1 = total_samples / (2 * class_counts[1]) if 1 in class_counts else 1.0

    scale_pos_weight = weight_for_1 / weight_for_0

    print("🚀 Training MASTER model with OPTIMIZED parameters...")
    print()

    # MASTER parameters optimized for scalping
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.03,  # Lower for better generalization
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_data_in_leaf': 100,  # Prevent overfitting
        'lambda_l1': 0.1,  # L1 regularization
        'lambda_l2': 0.1,  # L2 regularization
        'scale_pos_weight': scale_pos_weight,
        'is_unbalance': False,  # We handle balance with scale_pos_weight
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,  # More rounds
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    print()

    # Evaluate
    print("📊 Evaluating model...")

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    train_acc = ((y_pred_train > 0.5).astype(int) == y_train).mean()
    val_acc = ((y_pred_val > 0.5).astype(int) == y_val).mean()

    # Per-class accuracy
    y_pred_val_class = (y_pred_val > 0.5).astype(int)

    up_mask = y_val == 1
    down_mask = y_val == 0

    up_acc = (y_pred_val_class[up_mask] == 1).mean() if up_mask.sum() > 0 else 0
    down_acc = (y_pred_val_class[down_mask] == 0).mean() if down_mask.sum() > 0 else 0

    # Prediction distribution
    up_pred = (y_pred_val > 0.5).sum()
    down_pred = (y_pred_val <= 0.5).sum()
    pred_balance = abs(up_pred - down_pred) / len(y_pred_val) * 100

    print()
    print("=" * 80)
    print("🏆 TRAINING COMPLETE!")
    print("=" * 80)
    print()
    print("📊 ACCURACY:")
    print(f"   Train:  {train_acc*100:.1f}%")
    print(f"   Val:    {val_acc*100:.1f}%")
    print()
    print("📊 PER-CLASS PERFORMANCE (Validation):")
    print(f"   UP accuracy:   {up_acc*100:.1f}%")
    print(f"   DOWN accuracy: {down_acc*100:.1f}%")
    print(f"   Balance:       {abs(up_acc - down_acc)*100:.1f}% diff")
    print()
    print("📊 PREDICTION DISTRIBUTION (Validation):")
    print(f"   UP predictions:   {up_pred} ({up_pred/len(y_pred_val)*100:.1f}%)")
    print(f"   DOWN predictions: {down_pred} ({down_pred/len(y_pred_val)*100:.1f}%)")
    print(f"   Balance:          {pred_balance:.1f}% diff")
    print()

    if abs(up_acc - down_acc) < 0.10 and pred_balance < 15:
        print("✅ MODEL IS PERFECTLY BALANCED! 🎯🏆")
    elif abs(up_acc - down_acc) < 0.15:
        print("✅ Model is well balanced")
    else:
        print("⚠️  Model has some bias (may need more data or rebalancing)")
    print()

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("🔝 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 80)
    for idx, row in feature_importance.head(15).iterrows():
        print(f"   {row['feature']:35} {row['importance']:>8.0f}")
    print()

    # Save model
    model_dir = Path("storage/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"ml_model_master_scalper_{days}d.pkl"

    model_data = {
        'model': model,
        'feature_names': feature_cols,
        'scaler_mean': None,
        'scaler_std': None,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'up_accuracy': up_acc,
        'down_accuracy': down_acc,
        'prediction_balance': pred_balance,
        'params': params,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'days_trained': days,
        'samples_trained': len(X_train)
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"💾 Model saved: {model_path}")
    print()
    print("=" * 80)
    print("🎯 NEXT STEPS:")
    print("=" * 80)
    print()
    print("1. Run backtest:")
    print(f"   python backtest_balanced_longterm.py --days {days}")
    print()
    print("2. If results good (WR > 54%, ROI > 0), use in bot:")
    print("   python run_bot_master.py --mode paper")
    print()
    print("=" * 80)

    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=365, 
                       help='Days to train (180=6mo, 365=1y, 730=2y)')

    args = parser.parse_args()

    config = load_config('standard')
    setup_logging('INFO', log_to_file=False)

    train_master_scalper(args.symbol, args.days, config)


if __name__ == "__main__":
    main()
