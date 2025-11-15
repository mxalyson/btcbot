"""
🚀 TREINAMENTO DE MODELO LIGHTGBM PARA SCALPING - V2.0

Pipeline completo de treinamento otimizado para scalping 15min.
Target: Win rate > 60% | Sharpe > 2.0 | Latência < 20ms

Usage:
    python train_scalping_model.py --symbol BTCUSDT --days 180 --horizon 5

Author: Claude
Date: 2025-11-14
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score
)

# Imports locais
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore
from core.utils import load_config, setup_logging
from features.advanced_features import ScalpingFeatureEngineer, create_legacy_features
from features.target_engineering import TargetEngineer

# Setup logging
import logging
setup_logging("INFO", log_to_file=True)
logger = logging.getLogger("TradingBot.ModelTraining")


class ScalpingModelTrainer:
    """
    Trainer completo para modelos de scalping.

    Features:
    - LightGBM otimizado
    - Walk-forward validation
    - Feature importance tracking
    - Model persistence
    - Métricas detalhadas
    """

    def __init__(self, config: Dict):
        self.config = config
        self.symbol = config.get('symbol', 'BTCUSDT')
        self.timeframe = config.get('timeframe', '15m')
        self.lookback_days = config.get('lookback_days', 180)

        # Target config
        self.target_config = config.get('target', {
            'tp_pct': 0.003,
            'sl_pct': 0.003,
            'horizons': [3, 5, 10],
            'fees': 0.0006,
            'slippage': 0.0001
        })
        self.target_horizon = config.get('target_horizon', 5)
        self.target_type = config.get('target_type', 'classification')

        # Model config
        self.model_config = config.get('model', self._get_default_model_config())

        # Validation config
        self.n_splits = config.get('n_splits', 5)
        self.test_size_pct = config.get('test_size_pct', 0.15)

        # Initialize components
        # Initialize Bybit REST client for data download
        self.rest_client = BybitRESTClient(
            api_key=os.getenv('BYBIT_API_KEY', ''),
            api_secret=os.getenv('BYBIT_API_SECRET', ''),
            testnet=False  # Use production for historical data
        )
        self.data_manager = DataManager(self.rest_client)
        self.feature_store = FeatureStore(load_config())
        self.scalping_features = ScalpingFeatureEngineer()
        self.target_engineer = TargetEngineer(self.target_config)

        # Results
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.validation_metrics = {}
        self.test_metrics = {}

    def _get_default_model_config(self) -> Dict:
        """
        Configuração padrão otimizada para scalping.
        """
        return {
            'objective': 'multiclass',  # ou 'regression'
            'num_class': 3,  # -1, 0, 1 → 0, 1, 2
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',

            # Árvores
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,

            # Regularização
            'min_child_samples': 100,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,  # L1
            'reg_lambda': 0.1,  # L2
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'subsample_freq': 5,

            # Performance
            'num_threads': 4,
            'verbose': -1,

            # Early stopping
            'early_stopping_rounds': 50,
        }

    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carrega e prepara dados completos.
        """
        logger.info("="*70)
        logger.info(f"📊 Loading data: {self.symbol} {self.timeframe}")
        logger.info(f"   Lookback: {self.lookback_days} days")
        logger.info("="*70)

        # Carrega dados brutos
        df = self.data_manager.get_data(
            self.symbol,
            self.timeframe,
            self.lookback_days,
            use_cache=False
        )

        logger.info(f"✅ Raw data: {len(df)} candles")

        # Features técnicas base
        df = self.feature_store.build_features(df, normalize=False)
        logger.info(f"✅ Base features: {len(df.columns)} columns")

        # Features legado (compatibilidade)
        df = create_legacy_features(df)
        logger.info(f"✅ Legacy features added")

        # Features avançadas de scalping
        df = self.scalping_features.build_all_features(df)
        logger.info(f"✅ Scalping features: {len(df.columns)} columns")

        # Targets
        df = self.target_engineer.create_all_targets(df)
        logger.info(f"✅ Targets created")

        # Analisa targets
        self.target_engineer.analyze_targets(df)

        # Remove NaN
        df = df.dropna()
        logger.info(f"✅ Clean data: {len(df)} samples")

        return df

    def prepare_train_test_split(self, df: pd.DataFrame
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split temporal (não aleatório!).

        Últimos test_size_pct% para teste.
        """
        test_size = int(len(df) * self.test_size_pct)
        train_size = len(df) - test_size

        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        logger.info(f"\n📊 Train/Test Split:")
        logger.info(f"   Train: {len(df_train):,} samples ({train_size/len(df)*100:.1f}%)")
        logger.info(f"   Test:  {len(df_test):,} samples ({test_size/len(df)*100:.1f}%)")
        logger.info(f"   Train period: {df_train.index[0]} to {df_train.index[-1]}")
        logger.info(f"   Test period:  {df_test.index[0]} to {df_test.index[-1]}")

        # Prepara X, y
        X_train, y_train = self.target_engineer.create_labels_for_training(
            df_train,
            target_type=self.target_type,
            horizon=self.target_horizon
        )

        X_test, y_test = self.target_engineer.create_labels_for_training(
            df_test,
            target_type=self.target_type,
            horizon=self.target_horizon
        )

        self.feature_names = list(X_train.columns)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series) -> lgb.LGBMClassifier:
        """
        Treina modelo LightGBM com early stopping.
        """
        logger.info("\n🚀 Training LightGBM model...")
        logger.info(f"   Features: {len(self.feature_names)}")
        logger.info(f"   Train samples: {len(X_train):,}")
        logger.info(f"   Val samples: {len(X_val):,}")

        # Ajusta config baseado no target type
        model_config = self.model_config.copy()
        if self.target_type == 'regression':
            model_config['objective'] = 'regression'
            model_config['metric'] = 'rmse'
            model_config.pop('num_class', None)
            model = lgb.LGBMRegressor(**model_config)
        else:
            model = lgb.LGBMClassifier(**model_config)

        # Treina
        start_time = time.time()

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=model_config['metric'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=model_config.get('early_stopping_rounds', 50),
                    verbose=False
                ),
                lgb.log_evaluation(period=100)
            ]
        )

        training_time = time.time() - start_time

        logger.info(f"\n✅ Training complete!")
        logger.info(f"   Time: {training_time:.1f}s")
        logger.info(f"   Best iteration: {model.best_iteration_}")
        logger.info(f"   Best score: {model.best_score_['valid_0'][model_config['metric']]:.4f}")

        return model

    def validate_walkforward(self, df: pd.DataFrame) -> Dict:
        """
        Walk-forward validation (time-series aware).
        """
        logger.info("\n" + "="*70)
        logger.info("🔄 WALK-FORWARD VALIDATION")
        logger.info("="*70)

        # Prepara dados
        X, y = self.target_engineer.create_labels_for_training(
            df,
            target_type=self.target_type,
            horizon=self.target_horizon
        )

        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\n📊 Fold {fold}/{self.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Treina
            model = self.train_model(X_train, y_train, X_val, y_val)

            # Avalia
            metrics = self.evaluate_model(model, X_val, y_val, prefix=f"Fold {fold}")
            fold_metrics.append(metrics)

        # Agrega métricas
        avg_metrics = self._aggregate_fold_metrics(fold_metrics)

        logger.info("\n" + "="*70)
        logger.info("📊 WALK-FORWARD RESULTS (Average)")
        logger.info("="*70)
        for key, value in avg_metrics.items():
            logger.info(f"   {key}: {value:.4f}")

        self.validation_metrics = avg_metrics

        return avg_metrics

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      prefix: str = "Test") -> Dict:
        """
        Avalia modelo com métricas detalhadas.
        """
        y_pred = model.predict(X)

        metrics = {}

        if self.target_type == 'classification':
            # Probabilidades
            y_pred_proba = model.predict_proba(X)

            # Métricas
            metrics['accuracy'] = accuracy_score(y, y_pred)
            metrics['precision_macro'] = precision_score(y, y_pred, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(y, y_pred, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(y, y_pred, average='macro', zero_division=0)

            # Per-class metrics
            for class_idx in [0, 1, 2]:  # SHORT, NEUTRAL, LONG
                class_name = ['SHORT', 'NEUTRAL', 'LONG'][class_idx]
                y_binary = (y == class_idx).astype(int)
                y_pred_binary = (y_pred == class_idx).astype(int)

                prec = precision_score(y_binary, y_pred_binary, zero_division=0)
                rec = recall_score(y_binary, y_pred_binary, zero_division=0)

                metrics[f'precision_{class_name}'] = prec
                metrics[f'recall_{class_name}'] = rec

            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            logger.info(f"\n{prefix} Confusion Matrix:")
            logger.info(f"              SHORT  NEUTRAL  LONG")
            logger.info(f"   SHORT    {cm[0][0]:6d}  {cm[0][1]:6d}  {cm[0][2]:6d}")
            logger.info(f"   NEUTRAL  {cm[1][0]:6d}  {cm[1][1]:6d}  {cm[1][2]:6d}")
            logger.info(f"   LONG     {cm[2][0]:6d}  {cm[2][1]:6d}  {cm[2][2]:6d}")

        else:  # regression
            metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['r2'] = r2_score(y, y_pred)
            metrics['mae'] = np.mean(np.abs(y - y_pred))

        logger.info(f"\n{prefix} Metrics:")
        for key, value in metrics.items():
            logger.info(f"   {key}: {value:.4f}")

        return metrics

    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analisa importância das features.
        """
        if self.model is None:
            logger.warning("⚠️ No model trained yet!")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()

        logger.info("\n" + "="*70)
        logger.info("📊 FEATURE IMPORTANCE - Top 20")
        logger.info("="*70)
        logger.info(importance_df.head(20).to_string(index=False))

        # Features inúteis
        useless_features = importance_df[importance_df['importance_pct'] < 0.1]
        logger.info(f"\n⚠️  Features com < 0.1% importance: {len(useless_features)}")
        if len(useless_features) > 0:
            logger.info(f"   Consider removing: {list(useless_features['feature'].head(10))}")

        self.feature_importance = importance_df

        return importance_df

    def save_model(self, output_dir: str = 'ml_training/outputs') -> str:
        """
        Salva modelo treinado e metadados.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"scalping_model_{self.symbol}_{self.timeframe}_{timestamp}.pkl"
        model_path = output_path / model_name

        # Prepara dados para salvar
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'config': self.config,
            'target_config': self.target_config,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'training_date': timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'target_type': self.target_type,
            'target_horizon': self.target_horizon
        }

        # Salva
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"\n✅ Model saved: {model_path}")

        # Salva métricas separadamente (JSON)
        metrics_path = output_path / f"metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'validation': self.validation_metrics,
                'test': self.test_metrics,
                'config': self.config
            }, f, indent=2)

        logger.info(f"✅ Metrics saved: {metrics_path}")

        return str(model_path)

    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """
        Agrega métricas de múltiplos folds.
        """
        avg_metrics = {}

        # Coleta todas as keys
        all_keys = set()
        for metrics in fold_metrics:
            all_keys.update(metrics.keys())

        # Calcula média
        for key in all_keys:
            values = [m[key] for m in fold_metrics if key in m]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

        return avg_metrics

    def run_full_pipeline(self) -> str:
        """
        Executa pipeline completo de treinamento.
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 SCALPING MODEL TRAINING PIPELINE V2.0")
        logger.info("="*70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Timeframe: {self.timeframe}")
        logger.info(f"Target: {self.target_type} ({self.target_horizon} bars)")
        logger.info(f"Lookback: {self.lookback_days} days")
        logger.info("="*70)

        # 1. Load data
        df = self.load_and_prepare_data()

        # 2. Walk-forward validation
        self.validate_walkforward(df)

        # 3. Train final model on full train set
        logger.info("\n" + "="*70)
        logger.info("🎯 TRAINING FINAL MODEL")
        logger.info("="*70)

        X_train, X_test, y_train, y_test = self.prepare_train_test_split(df)

        # Split validation from train
        val_size = int(len(X_train) * 0.15)
        X_val = X_train.iloc[-val_size:]
        y_val = y_train.iloc[-val_size:]
        X_train = X_train.iloc[:-val_size]
        y_train = y_train.iloc[:-val_size]

        self.model = self.train_model(X_train, y_train, X_val, y_val)

        # 4. Evaluate on test set
        logger.info("\n" + "="*70)
        logger.info("📊 FINAL TEST EVALUATION")
        logger.info("="*70)

        self.test_metrics = self.evaluate_model(self.model, X_test, y_test, prefix="Test")

        # 5. Feature importance
        self.analyze_feature_importance()

        # 6. Save model
        model_path = self.save_model()

        # 7. Summary
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"📊 Validation Accuracy: {self.validation_metrics.get('accuracy', 0):.3f}")
        logger.info(f"📊 Test Accuracy: {self.test_metrics.get('accuracy', 0):.3f}")
        logger.info(f"📊 Test Precision (LONG): {self.test_metrics.get('precision_LONG', 0):.3f}")
        logger.info(f"📊 Test Recall (LONG): {self.test_metrics.get('recall_LONG', 0):.3f}")
        logger.info(f"💾 Model: {model_path}")
        logger.info("="*70)

        return model_path


def main():
    """
    Entry point.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Train scalping model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--days', type=int, default=180, help='Lookback days')
    parser.add_argument('--horizon', type=int, default=5, choices=[3, 5, 10], help='Target horizon')
    parser.add_argument('--target-type', type=str, default='classification',
                       choices=['classification', 'regression'], help='Target type')
    parser.add_argument('--tp', type=float, default=0.003, help='TP percentage')
    parser.add_argument('--sl', type=float, default=0.003, help='SL percentage')

    args = parser.parse_args()

    config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'lookback_days': args.days,
        'target_horizon': args.horizon,
        'target_type': args.target_type,
        'target': {
            'tp_pct': args.tp,
            'sl_pct': args.sl,
            'horizons': [3, 5, 10],
            'fees': 0.0006,
            'slippage': 0.0001
        },
        'n_splits': 5,
        'test_size_pct': 0.15
    }

    trainer = ScalpingModelTrainer(config)
    model_path = trainer.run_full_pipeline()

    print(f"\n🎉 Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()
