"""
🔄 WALK-FORWARD VALIDATION - V2.0

Validação robusta para modelos de scalping.
Simula produção real com dados temporais.

Author: Claude
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Validação walk-forward para séries temporais.

    Mantém ordem temporal dos dados (CRUCIAL para scalping!)
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config:
                {
                    'train_period_days': 120,  # Training window
                    'test_period_days': 30,    # Test window
                    'step_days': 15,            # Reamostragem a cada X dias
                    'min_train_samples': 1000,  # Mínimo de amostras
                }
        """
        self.config = config or {}
        self.train_period_days = self.config.get('train_period_days', 120)
        self.test_period_days = self.config.get('test_period_days', 30)
        self.step_days = self.config.get('step_days', 15)
        self.min_train_samples = self.config.get('min_train_samples', 1000)

        self.results = []

    def generate_splits(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Gera splits walk-forward.

        Returns:
            Lista de (train_start, train_end, test_start, test_end)
        """
        splits = []

        start_date = df.index[0]
        end_date = df.index[-1]

        current_date = start_date + timedelta(days=self.train_period_days)

        while current_date + timedelta(days=self.test_period_days) <= end_date:
            train_start = current_date - timedelta(days=self.train_period_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_period_days)

            # Verifica se há dados suficientes
            train_mask = (df.index >= train_start) & (df.index < train_end)
            test_mask = (df.index >= test_start) & (df.index < test_end)

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train >= self.min_train_samples and n_test > 0:
                splits.append((train_start, train_end, test_start, test_end))

            current_date += timedelta(days=self.step_days)

        print(f"📊 Generated {len(splits)} walk-forward splits")
        return splits

    def validate(self, X: pd.DataFrame, y: pd.Series,
                model_class, model_params: Dict) -> Dict:
        """
        Executa validação walk-forward completa.

        Args:
            X: Features
            y: Target
            model_class: Classe do modelo (ex: lgb.LGBMClassifier)
            model_params: Parâmetros do modelo

        Returns:
            Dict com métricas agregadas
        """
        splits = self.generate_splits(X)

        print("\n" + "="*70)
        print("🔄 WALK-FORWARD VALIDATION")
        print("="*70)

        fold_results = []

        for fold, (train_start, train_end, test_start, test_end) in enumerate(splits, 1):
            print(f"\n📊 Fold {fold}/{len(splits)}")
            print(f"   Train: {train_start.date()} to {train_end.date()}")
            print(f"   Test:  {test_start.date()} to {test_end.date()}")

            # Split data
            train_mask = (X.index >= train_start) & (X.index < train_end)
            test_mask = (X.index >= test_start) & (X.index < test_end)

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            print(f"   Train samples: {len(X_train):,}")
            print(f"   Test samples:  {len(X_test):,}")

            # Train model
            model = model_class(**model_params)

            # Validation split interno (15% do train)
            val_size = int(len(X_train) * 0.15)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_sub = X_train.iloc[:-val_size]
            y_train_sub = y_train.iloc[:-val_size]

            try:
                model.fit(
                    X_train_sub, y_train_sub,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            except:
                # Fallback se não suportar eval_set
                model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            metrics['fold'] = fold
            metrics['train_start'] = train_start
            metrics['train_end'] = train_end
            metrics['test_start'] = test_start
            metrics['test_end'] = test_end
            metrics['n_train'] = len(X_train)
            metrics['n_test'] = len(X_test)

            fold_results.append(metrics)

            print(f"   Accuracy: {metrics['accuracy']:.3f}")
            print(f"   Precision (LONG): {metrics['precision_LONG']:.3f}")
            print(f"   Recall (LONG): {metrics['recall_LONG']:.3f}")

        self.results = fold_results

        # Aggregate
        aggregated = self._aggregate_results(fold_results)

        print("\n" + "="*70)
        print("📊 WALK-FORWARD RESULTS (Aggregated)")
        print("="*70)
        print(f"   Accuracy: {aggregated['accuracy_mean']:.3f} ± {aggregated['accuracy_std']:.3f}")
        print(f"   Precision (LONG): {aggregated['precision_LONG_mean']:.3f} ± {aggregated['precision_LONG_std']:.3f}")
        print(f"   Recall (LONG): {aggregated['recall_LONG_mean']:.3f} ± {aggregated['recall_LONG_std']:.3f}")
        print(f"   F1 (LONG): {aggregated['f1_LONG_mean']:.3f} ± {aggregated['f1_LONG_std']:.3f}")
        print("="*70)

        return aggregated

    def _calculate_metrics(self, y_true, y_pred) -> Dict:
        """
        Calcula métricas detalhadas.
        """
        metrics = {}

        # Overall
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Per class (assumindo 0=SHORT, 1=NEUTRAL, 2=LONG)
        for class_idx, class_name in [(0, 'SHORT'), (1, 'NEUTRAL'), (2, 'LONG')]:
            y_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)

            prec = precision_score(y_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_binary, y_pred_binary, zero_division=0)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-10)

            metrics[f'precision_{class_name}'] = prec
            metrics[f'recall_{class_name}'] = rec
            metrics[f'f1_{class_name}'] = f1

        return metrics

    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Agrega resultados de múltiplos folds.
        """
        aggregated = {}

        # Metrics to aggregate
        metric_keys = ['accuracy', 'precision_SHORT', 'precision_NEUTRAL', 'precision_LONG',
                      'recall_SHORT', 'recall_NEUTRAL', 'recall_LONG',
                      'f1_SHORT', 'f1_NEUTRAL', 'f1_LONG']

        for key in metric_keys:
            values = [r[key] for r in results]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)

        return aggregated

    def plot_results(self, save_path: str = None):
        """
        Plota evolução das métricas ao longo do tempo.
        """
        if not self.results:
            print("⚠️ No results to plot!")
            return

        try:
            import matplotlib.pyplot as plt

            df_results = pd.DataFrame(self.results)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Validation Results', fontsize=16)

            # Accuracy
            axes[0, 0].plot(df_results['test_start'], df_results['accuracy'], marker='o')
            axes[0, 0].axhline(y=df_results['accuracy'].mean(), color='r', linestyle='--',
                              label=f'Mean: {df_results["accuracy"].mean():.3f}')
            axes[0, 0].set_title('Accuracy Over Time')
            axes[0, 0].set_xlabel('Test Period Start')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Precision LONG
            axes[0, 1].plot(df_results['test_start'], df_results['precision_LONG'], marker='o', color='green')
            axes[0, 1].axhline(y=df_results['precision_LONG'].mean(), color='r', linestyle='--',
                              label=f'Mean: {df_results["precision_LONG"].mean():.3f}')
            axes[0, 1].set_title('Precision (LONG) Over Time')
            axes[0, 1].set_xlabel('Test Period Start')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Recall LONG
            axes[1, 0].plot(df_results['test_start'], df_results['recall_LONG'], marker='o', color='blue')
            axes[1, 0].axhline(y=df_results['recall_LONG'].mean(), color='r', linestyle='--',
                              label=f'Mean: {df_results["recall_LONG"].mean():.3f}')
            axes[1, 0].set_title('Recall (LONG) Over Time')
            axes[1, 0].set_xlabel('Test Period Start')
            axes[1, 0].set_ylabel('Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # F1 LONG
            axes[1, 1].plot(df_results['test_start'], df_results['f1_LONG'], marker='o', color='purple')
            axes[1, 1].axhline(y=df_results['f1_LONG'].mean(), color='r', linestyle='--',
                              label=f'Mean: {df_results["f1_LONG"].mean():.3f}')
            axes[1, 1].set_title('F1 Score (LONG) Over Time')
            axes[1, 1].set_xlabel('Test Period Start')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved to: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("⚠️ matplotlib not available. Skipping plot.")

    def export_results(self, filepath: str):
        """
        Exporta resultados para CSV.
        """
        if not self.results:
            print("⚠️ No results to export!")
            return

        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filepath, index=False)
        print(f"✅ Results exported to: {filepath}")


if __name__ == "__main__":
    """
    Teste do walk-forward validation.
    """
    # Dados de exemplo
    dates = pd.date_range('2024-01-01', periods=10000, freq='15min')
    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(10000, 50),
        index=dates,
        columns=[f'feature_{i}' for i in range(50)]
    )

    # Target simulado (0, 1, 2)
    y = pd.Series(
        np.random.choice([0, 1, 2], size=10000, p=[0.3, 0.4, 0.3]),
        index=dates
    )

    # Validação
    validator = WalkForwardValidator({
        'train_period_days': 60,
        'test_period_days': 15,
        'step_days': 10,
        'min_train_samples': 500
    })

    results = validator.validate(
        X, y,
        model_class=lgb.LGBMClassifier,
        model_params={
            'objective': 'multiclass',
            'num_class': 3,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'verbose': -1
        }
    )

    print("\n✅ Walk-forward validation test complete!")
