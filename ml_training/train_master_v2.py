"""
🏆 MASTER SCALPER V2.0 - Modelo com Target Correto + Features Avançadas

Combina o melhor dos 2 mundos:
- Target do modelo antigo (votação + threshold dinâmico) ✅
- Features avançadas do modelo novo (150+ features) ✅
- Validação temporal robusta ✅

Usage:
    python train_master_v2.py --symbol BTCUSDT --days 180

Author: Claude
Date: 2025-11-15
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from train_scalping_model import ScalpingModelTrainer

def main():
    parser = argparse.ArgumentParser(description='Train MASTER SCALPER V2.0')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=180, help='Training days (180=6mo recommended)')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🏆 MASTER SCALPER V2.0 - TREINAMENTO")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.days} days")
    print()
    print("📊 DIFERENÇAS DO MODELO ANTIGO:")
    print("   ✅ Target: Votação multi-horizon + threshold dinâmico ATR")
    print("   ✅ Features: 150+ (vs 65 do antigo)")
    print("   ✅ Validation: Walk-forward 5 folds")
    print()
    print("🎯 OBJETIVO:")
    print("   - Win Rate: > 55%")
    print("   - Balanced predictions (45-55% UP/DOWN)")
    print("   - ROI anual: > 60%")
    print("="*80)
    print()

    # Configuração para MASTER SCALPER
    config = {
        'symbol': args.symbol,
        'timeframe': '15m',
        'lookback_days': args.days,
        'target_type': 'master',  # ⭐ USA TARGET DO MODELO ANTIGO!
        'target_horizon': 6,  # Ignored para 'master', mas precisa existir
        'target': {
            'tp_pct': 0.003,  # Não usado para 'master'
            'sl_pct': 0.003,  # Não usado para 'master'
            'horizons': [4, 6, 8],  # Usado pelo target master
            'fees': 0.0006,
            'slippage': 0.0001
        },
        'n_splits': 5,
        'test_size_pct': 0.15,
        'model': {
            # MASTER SCALPER params (otimizados para binário)
            'objective': 'binary',  # Será configurado automaticamente
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'learning_rate': 0.03,  # Menor = melhor generalização
            'n_estimators': 500,

            # Regularização FORTE
            'min_child_samples': 100,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,  # L1
            'reg_lambda': 0.1,  # L2
            'colsample_bytree': 0.85,
            'subsample': 0.85,
            'subsample_freq': 5,

            # Performance
            'num_threads': 4,
            'verbose': -1,
            'early_stopping_rounds': 50,
        }
    }

    # Treina modelo
    trainer = ScalpingModelTrainer(config)
    model_path = trainer.run_full_pipeline()

    print("\n" + "="*80)
    print("🎉 TREINAMENTO COMPLETO!")
    print("="*80)
    print(f"💾 Model: {model_path}")
    print()
    print("🚀 PRÓXIMOS PASSOS:")
    print()
    print("1. Testar modelo no backtest:")
    print(f"   cd validation")
    print(f"   python backtest_ml_model.py --model ../outputs/{Path(model_path).name} --days 90 --confidence 0.50 --tp 2.0 --sl 1.5")
    print()
    print("2. Se Win Rate > 55% e ROI > 0:")
    print("   - Deploy no bot BTC")
    print("   - Configurar MIN_ML_CONFIDENCE=0.50 no .env")
    print("="*80)

    return model_path


if __name__ == "__main__":
    main()
