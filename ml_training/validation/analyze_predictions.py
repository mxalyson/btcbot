#!/usr/bin/env python3
"""
Analisa as predições do modelo para identificar problemas
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.data_manager import DataManager
from ml_training.features.feature_engineering import FeatureEngineer
from ml_training.features.advanced_features import ScalpingFeatureEngineer, create_legacy_features

def analyze_model_predictions(model_path: str, days: int = 90):
    """Analisa as predições do modelo"""

    print(f"\n📊 Analyzing model predictions...")
    print(f"   Model: {model_path}")
    print(f"   Days: {days}")

    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_names = model_data['feature_names']
    metadata = model_data.get('metadata', {})

    print(f"\n✅ Model loaded")
    print(f"   Features: {len(feature_names)}")
    print(f"   Target: {metadata.get('target_type', 'unknown')}")

    # Load data
    data_manager = DataManager()
    df = data_manager.fetch_historical_data(
        symbol='BTCUSDT',
        timeframe='15m',
        days=days
    )

    print(f"\n✅ Data loaded: {len(df):,} candles")

    # Build features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.build_features(df)
    df_features = create_legacy_features(df_features)

    scalping_engineer = ScalpingFeatureEngineer()
    df_features = scalping_engineer.build_all_features(df_features)

    print(f"✅ Features built: {len(df_features.columns)} columns")

    # Prepare features for prediction
    missing_features = [f for f in feature_names if f not in df_features.columns]
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features, filling with 0")
        for feat in missing_features:
            df_features[feat] = 0

    X = df_features[feature_names].fillna(0)

    # Get predictions
    predictions = model.predict(X)
    proba = model.predict_proba(X)

    # Analyze predictions
    n_classes = proba.shape[1]

    print(f"\n{'='*70}")
    print(f"📊 PREDICTION ANALYSIS")
    print(f"{'='*70}")

    if n_classes == 2:
        # Binary model
        print(f"Model type: BINARY (2 classes)")
        print(f"\nClass distribution:")
        print(f"  DOWN (0): {(predictions == 0).sum():,} ({(predictions == 0).mean()*100:.1f}%)")
        print(f"  UP (1):   {(predictions == 1).sum():,} ({(predictions == 1).mean()*100:.1f}%)")

        # Confidence analysis
        prob_down = proba[:, 0]
        prob_up = proba[:, 1]

        print(f"\nConfidence distribution (DOWN):")
        print(f"  Mean: {prob_down.mean():.1%}")
        print(f"  Median: {np.median(prob_down):.1%}")
        print(f"  Min: {prob_down.min():.1%}")
        print(f"  Max: {prob_down.max():.1%}")

        print(f"\nConfidence distribution (UP):")
        print(f"  Mean: {prob_up.mean():.1%}")
        print(f"  Median: {np.median(prob_up):.1%}")
        print(f"  Min: {prob_up.min():.1%}")
        print(f"  Max: {prob_up.max():.1%}")

        # Analyze confidence thresholds
        print(f"\nPredictions by confidence threshold:")
        for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
            high_conf_down = ((predictions == 0) & (prob_down >= threshold)).sum()
            high_conf_up = ((predictions == 1) & (prob_up >= threshold)).sum()
            total_high_conf = high_conf_down + high_conf_up

            if total_high_conf > 0:
                pct_down = high_conf_down / total_high_conf * 100
                pct_up = high_conf_up / total_high_conf * 100
            else:
                pct_down = pct_up = 0

            print(f"  >= {threshold:.0%}: {total_high_conf:,} signals ({pct_down:.1f}% DOWN, {pct_up:.1f}% UP)")

    else:
        # Multiclass model
        print(f"Model type: MULTICLASS (3 classes)")
        print(f"\nClass distribution:")
        for i in range(n_classes):
            count = (predictions == i).sum()
            pct = (predictions == i).mean() * 100
            print(f"  Class {i}: {count:,} ({pct:.1f}%)")

    # Temporal analysis
    print(f"\n{'='*70}")
    print(f"📅 TEMPORAL ANALYSIS")
    print(f"{'='*70}")

    df_features['prediction'] = predictions
    df_features['prob_up'] = proba[:, 1] if n_classes == 2 else proba[:, 2]

    # Group by date
    df_features['date'] = pd.to_datetime(df_features.index).date
    daily_stats = df_features.groupby('date').agg({
        'prediction': lambda x: (x == 1).mean(),  # % UP predictions
        'prob_up': 'mean'
    })

    print(f"\nDaily UP prediction rate (last 10 days):")
    for date, row in daily_stats.tail(10).iterrows():
        print(f"  {date}: {row['prediction']*100:.1f}% UP (avg prob: {row['prob_up']:.1%})")

    # Volatility correlation
    if 'atr' in df_features.columns:
        print(f"\n{'='*70}")
        print(f"📈 VOLATILITY CORRELATION")
        print(f"{'='*70}")

        # Group by ATR quintiles
        df_features['atr_quintile'] = pd.qcut(df_features['atr'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')

        print(f"\nPredictions by volatility (ATR):")
        for quintile in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
            mask = df_features['atr_quintile'] == quintile
            if mask.sum() > 0:
                pct_up = (df_features.loc[mask, 'prediction'] == 1).mean() * 100
                avg_prob = df_features.loc[mask, 'prob_up'].mean()
                print(f"  {quintile:10s}: {pct_up:.1f}% UP (avg prob: {avg_prob:.1%})")

    print(f"\n{'='*70}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Analyze ML model predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--days', type=int, default=90, help='Number of days to analyze')

    args = parser.parse_args()

    analyze_model_predictions(args.model, args.days)
