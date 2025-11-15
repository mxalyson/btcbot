"""
🎛️ HYPERPARAMETER OPTIMIZATION COM OPTUNA - V2.0

Otimização automática de hiperparâmetros LightGBM para scalping.
Usa Optuna para busca eficiente.

Usage:
    python optuna_tuner.py --symbol BTCUSDT --n-trials 100

Author: Claude
Date: 2025-11-14
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaHyperparameterTuner:
    """
    Otimizador de hiperparâmetros usando Optuna.

    Busca eficiente com:
    - Tree-structured Parzen Estimator (TPE)
    - Median pruning (early stopping)
    - Cross-validation temporal
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, config: dict = None):
        """
        Args:
            X: Features
            y: Target
            config:
                {
                    'n_trials': 100,
                    'n_splits': 3,
                    'timeout': 3600,  # seconds
                    'metric': 'accuracy',  # ou 'f1'
                    'target_type': 'classification'
                }
        """
        self.X = X
        self.y = y
        self.config = config or {}

        self.n_trials = self.config.get('n_trials', 100)
        self.n_splits = self.config.get('n_splits', 3)
        self.timeout = self.config.get('timeout', 3600)
        self.metric = self.config.get('metric', 'accuracy')
        self.target_type = self.config.get('target_type', 'classification')

        self.best_params = None
        self.best_score = None
        self.study = None

    def objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para Optuna.

        Testa uma combinação de hiperparâmetros e retorna score.
        """
        # Sugere hiperparâmetros
        params = {
            'objective': 'multiclass' if self.target_type == 'classification' else 'regression',
            'metric': 'multi_logloss' if self.target_type == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,

            # Árvores
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),

            # Regularização
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 300),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),

            # Sampling
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),

            # Outros
            'num_threads': 4,
        }

        if self.target_type == 'classification':
            params['num_class'] = 3  # SHORT, NEUTRAL, LONG

        # Cross-validation temporal
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # Treina modelo
            if self.target_type == 'classification':
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            # Avalia
            y_pred = model.predict(X_val)

            if self.metric == 'accuracy':
                score = accuracy_score(y_val, y_pred)
            elif self.metric == 'f1':
                score = f1_score(y_val, y_pred, average='macro')
            else:
                score = accuracy_score(y_val, y_pred)

            scores.append(score)

            # Pruning (early stopping de trials ruins)
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Retorna média dos folds
        return np.mean(scores)

    def optimize(self) -> dict:
        """
        Executa otimização completa.

        Returns:
            Best hyperparameters
        """
        print("\n" + "="*70)
        print("🎛️  HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        print(f"Metric: {self.metric}")
        print(f"Trials: {self.n_trials}")
        print(f"CV Splits: {self.n_splits}")
        print(f"Timeout: {self.timeout}s")
        print("="*70)

        # Cria study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Otimiza
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Best params
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print("\n" + "="*70)
        print("✅ OPTIMIZATION COMPLETE")
        print("="*70)
        print(f"Best {self.metric}: {self.best_score:.4f}")
        print("\nBest Parameters:")
        for param, value in self.best_params.items():
            print(f"  {param:25s}: {value}")
        print("="*70)

        return self.best_params

    def get_feature_importance(self, n_top: int = 20) -> pd.DataFrame:
        """
        Treina modelo final com best params e retorna feature importance.
        """
        if self.best_params is None:
            print("⚠️ Run optimize() first!")
            return pd.DataFrame()

        # Prepara params
        params = self.best_params.copy()
        params['objective'] = 'multiclass' if self.target_type == 'classification' else 'regression'
        params['metric'] = 'multi_logloss' if self.target_type == 'classification' else 'rmse'
        params['boosting_type'] = 'gbdt'
        params['verbosity'] = -1
        params['num_threads'] = 4

        if self.target_type == 'classification':
            params['num_class'] = 3
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        # Treina
        model.fit(self.X, self.y)

        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )

        print(f"\n📊 Top {n_top} Features:")
        print(importance_df.head(n_top).to_string(index=False))

        return importance_df

    def plot_optimization_history(self, save_path: str = None):
        """
        Plota histórico de otimização.
        """
        if self.study is None:
            print("⚠️ Run optimize() first!")
            return

        try:
            import matplotlib.pyplot as plt

            # Plot 1: Optimization history
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Optuna Optimization Results', fontsize=16)

            # History
            trials = self.study.trials
            values = [t.value for t in trials if t.value is not None]
            ax1 = axes[0, 0]
            ax1.plot(values, marker='o', alpha=0.6)
            ax1.axhline(y=self.best_score, color='r', linestyle='--',
                       label=f'Best: {self.best_score:.4f}')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel(self.metric.capitalize())
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Best value over time
            ax2 = axes[0, 1]
            best_values = []
            current_best = -np.inf
            for v in values:
                if v > current_best:
                    current_best = v
                best_values.append(current_best)
            ax2.plot(best_values, marker='o', color='green', alpha=0.6)
            ax2.set_xlabel('Trial')
            ax2.set_ylabel(f'Best {self.metric.capitalize()}')
            ax2.set_title('Best Value Over Time')
            ax2.grid(True, alpha=0.3)

            # Param importance
            ax3 = axes[1, 0]
            try:
                importances = optuna.importance.get_param_importances(self.study)
                params = list(importances.keys())[:10]
                values = [importances[p] for p in params]
                ax3.barh(params, values)
                ax3.set_xlabel('Importance')
                ax3.set_title('Parameter Importance (Top 10)')
                ax3.grid(True, alpha=0.3, axis='x')
            except:
                ax3.text(0.5, 0.5, 'Param importance unavailable',
                        ha='center', va='center')

            # Distribution of best param
            ax4 = axes[1, 1]
            if len(self.best_params) > 0:
                first_param = list(self.best_params.keys())[0]
                param_values = [t.params.get(first_param) for t in trials
                               if first_param in t.params and t.value is not None]
                ax4.hist(param_values, bins=30, alpha=0.6, edgecolor='black')
                ax4.axvline(x=self.best_params[first_param], color='r',
                           linestyle='--', label=f'Best: {self.best_params[first_param]:.3f}')
                ax4.set_xlabel(first_param)
                ax4.set_ylabel('Frequency')
                ax4.set_title(f'Distribution of {first_param}')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved: {save_path}")
            else:
                plt.show()

        except ImportError:
            print("⚠️ matplotlib not available")

    def save_results(self, output_dir: str = 'ml_training/outputs'):
        """
        Salva resultados da otimização.
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Best params
        params_file = output_path / f"optuna_best_params_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        print(f"✅ Best params saved: {params_file}")

        # Study
        study_file = output_path / f"optuna_study_{timestamp}.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        print(f"✅ Study saved: {study_file}")

        # Trials dataframe
        trials_df = self.study.trials_dataframe()
        trials_file = output_path / f"optuna_trials_{timestamp}.csv"
        trials_df.to_csv(trials_file, index=False)
        print(f"✅ Trials saved: {trials_file}")


def main():
    """
    Entry point para otimização standalone.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Optimize hyperparameters')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--metric', type=str, default='accuracy', choices=['accuracy', 'f1'])

    args = parser.parse_args()

    print(f"⚠️  This is a standalone script.")
    print(f"    For full pipeline, use train_scalping_model.py")
    print(f"\n    To use this:")
    print(f"    1. Load your X, y data")
    print(f"    2. Create OptunaHyperparameterTuner")
    print(f"    3. Call optimize()")


if __name__ == "__main__":
    main()
