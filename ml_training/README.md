# 🚀 SCALPING ML PIPELINE V2.0

Pipeline completo de Machine Learning para scalping em criptomoedas.
**Objetivo: Modelo 10/10 com Win Rate > 60% | Sharpe > 2.0 | Latência < 20ms**

---

## 📋 Tabela de Conteúdo

1. [Overview](#overview)
2. [Arquitetura](#arquitetura)
3. [Instalação](#instalação)
4. [Quick Start](#quick-start)
5. [Componentes](#componentes)
6. [Uso Avançado](#uso-avançado)
7. [Métricas e Validação](#métricas-e-validação)
8. [FAQ](#faq)

---

## 🎯 Overview

Este pipeline foi desenvolvido para treinar modelos de alta performance para scalping em timeframe 15min.

### Características Principais:

✅ **LightGBM** - Latência < 20ms (10-20x mais rápido que XGBoost)
✅ **150+ Features** - Price action, order flow, microestrutura, regime detection
✅ **Multi-horizon targets** - 3/5/10 bars ahead
✅ **Walk-forward validation** - Time-series aware (não overfitting!)
✅ **Backtesting realista** - Fees, slippage, gestão de risco
✅ **Optuna optimization** - Hyperparameter tuning automático
✅ **Production-ready** - Pronto para deploy nos bots

### Diferenças do Modelo Antigo:

| Feature | Modelo Antigo | Modelo V2.0 |
|---------|--------------|-------------|
| **Algoritmo** | Desconhecido | **LightGBM** |
| **Latência** | Desconhecida | **< 20ms** |
| **Features** | ~100 | **150+** |
| **Order Flow** | ❌ Não | ✅ **Sim** |
| **Regime Detection** | ❌ Não | ✅ **Sim** |
| **Multi-horizon** | ❌ Não | ✅ **Sim** |
| **Walk-forward Val** | ❌ Não documentado | ✅ **Sim** |
| **Backtest Realista** | ❌ Não | ✅ **Sim** |
| **Optuna** | ❌ Não | ✅ **Sim** |

---

## 🏗️ Arquitetura

```
ml_training/
├── data/                    # Coleta e preprocessamento de dados
├── features/                # Feature engineering
│   ├── advanced_features.py # ⭐ Features de scalping (order flow, etc)
│   └── target_engineering.py # ⭐ Multi-horizon targets
├── models/                  # Modelos treinados (.pkl)
├── validation/              # Validação e backtesting
│   ├── walk_forward.py     # ⭐ Walk-forward validation
│   └── backtest_realistic.py # ⭐ Backtest com fees/slippage
├── optimization/            # Hyperparameter tuning
│   └── optuna_tuner.py     # ⭐ Optuna optimization
├── outputs/                 # Resultados
│   ├── models/             # Modelos salvos
│   ├── metrics/            # Métricas JSON/CSV
│   └── plots/              # Gráficos
├── train_scalping_model.py # ⭐ Script principal de treinamento
├── config.yaml             # Configuração
└── README.md               # Este arquivo
```

---

## 📦 Instalação

### Requisitos:

- Python 3.8+
- pip

### Instalar dependências:

```bash
pip install lightgbm optuna pandas numpy scikit-learn matplotlib pyyaml
```

### Verificar instalação:

```bash
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} OK')"
python -c "import optuna; print(f'Optuna {optuna.__version__} OK')"
```

---

## 🚀 Quick Start

### 1. Treinamento Simples (com defaults):

```bash
cd ml_training
python train_scalping_model.py --symbol BTCUSDT --days 180 --horizon 5
```

**Output:**
- Modelo treinado: `outputs/models/scalping_model_BTCUSDT_15m_TIMESTAMP.pkl`
- Métricas: `outputs/metrics/metrics_TIMESTAMP.json`

### 2. Treinamento com Otimização Optuna:

```bash
python train_scalping_model.py \
    --symbol ETHUSDT \
    --days 180 \
    --horizon 5 \
    --optimize \
    --n-trials 50
```

### 3. Treinamento Customizado:

```bash
python train_scalping_model.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --days 180 \
    --horizon 5 \
    --tp 0.003 \
    --sl 0.003 \
    --target-type classification
```

---

## 🧩 Componentes Detalhados

### 1. Feature Engineering (`features/advanced_features.py`)

**ScalpingFeatureEngineer** cria 150+ features otimizadas:

#### Categorias de Features:

**a) Price Action (Alta Frequência):**
```python
- price_change_1/2/3/5
- roc_1/3/5
- high_low_ratio
- body_to_range
- upper/lower_wick_pct
- candle_patterns (doji, hammer, shooting_star)
```

**b) Order Flow & Microestrutura:** ⭐
```python
- obi_top5/top10 (Order Book Imbalance)
- spread, spread_pct, spread_volatility
- bid_depth, ask_depth, liquidity_imbalance
```

**c) Trade Flow:** ⭐
```python
- buy_volume_ratio
- delta_volume, cumulative_delta
- aggressive_buy_pct
```

**d) Regime Detection:** ⭐
```python
- choppiness_index (trending vs ranging)
- adx_slope, is_trending
- volatility_regime
- bb_width (consolidation)
```

**e) Time-Based:**
```python
- hour, day_of_week
- is_asia/europe/us_session
- is_overlap (high liquidity)
```

**f) Volume Profile:**
```python
- vwap, distance_from_vwap
- volume_surge, volume_trend
```

### 2. Target Engineering (`features/target_engineering.py`)

**TargetEngineer** cria targets inteligentes:

#### Multi-Horizon Targets:

```python
# Target 3 bars (quick scalp)
target_3bars:
  - TP: +0.3% sem SL -0.3%
  - Timeframe: ~45 min

# Target 5 bars (standard)
target_5bars:
  - TP: +0.3% sem SL -0.3%
  - Timeframe: ~75 min

# Target 10 bars (extended)
target_10bars:
  - TP: +0.3% sem SL -0.3%
  - Timeframe: ~150 min
```

#### Tipos de Target:

1. **Classification** (padrão):
   - Classes: 0 (SHORT), 1 (NEUTRAL), 2 (LONG)
   - Modelo prevê direção

2. **Regression**:
   - Prevê retorno esperado em %
   - Permite avaliar magnitude do movimento

3. **Risk-Adjusted**:
   - Sharpe-like: return / volatility
   - Foco em movimentos de qualidade

### 3. Treinamento (`train_scalping_model.py`)

**Pipeline Completo:**

```python
class ScalpingModelTrainer:
    """
    1. Load data (6 meses recomendado)
    2. Feature engineering (150+ features)
    3. Target creation (multi-horizon)
    4. Walk-forward validation (5 folds)
    5. Train final model (LightGBM)
    6. Evaluate on test set (15%)
    7. Feature importance analysis
    8. Save model + metadata
    """
```

**Config Padrão LightGBM:**

```yaml
num_leaves: 31
max_depth: 6
learning_rate: 0.05
n_estimators: 500
min_child_samples: 100
reg_alpha: 0.1 (L1)
reg_lambda: 0.1 (L2)
colsample_bytree: 0.8
subsample: 0.8
early_stopping: 50 rounds
```

### 4. Validação (`validation/walk_forward.py`)

**Walk-Forward Validation:**

Mantém ordem temporal (CRUCIAL para scalping):

```
Fold 1: Train[Jan-Apr] → Test[May]
Fold 2: Train[Feb-May] → Test[Jun]
Fold 3: Train[Mar-Jun] → Test[Jul]
Fold 4: Train[Apr-Jul] → Test[Aug]
Fold 5: Train[May-Aug] → Test[Sep]
```

**Benefícios:**
- Evita lookahead bias
- Simula produção real
- Detecta overfitting

### 5. Backtesting (`validation/backtest_realistic.py`)

**Backtest Realista:**

```python
config = {
    'initial_capital': 10000,
    'risk_per_trade': 0.02,  # 2%
    'fees_pct': 0.0006,      # 0.06% (Bybit taker)
    'slippage_pct': 0.0001,  # 0.01%
    'atr_mult_sl': 1.5,
    'atr_mult_tp': 1.0,
    'max_trades_per_day': 10,
    'min_confidence': 0.30
}
```

**Métricas Calculadas:**
- Win rate, profit factor
- Total PnL, total return %
- Max drawdown, Sharpe ratio
- Avg trade duration
- Exit reasons (TP, SL, etc)

### 6. Otimização (`optimization/optuna_tuner.py`)

**Optuna Hyperparameter Tuning:**

```python
# Busca automática dos melhores hiperparâmetros
tuner = OptunaHyperparameterTuner(X, y, config={
    'n_trials': 100,
    'timeout': 3600,
    'metric': 'accuracy'
})

best_params = tuner.optimize()
```

**Search Space:**
- num_leaves: [15, 127]
- max_depth: [3, 12]
- learning_rate: [0.001, 0.3] (log)
- n_estimators: [50, 1000]
- Regularization: reg_alpha, reg_lambda
- Sampling: colsample, subsample

---

## 🎓 Uso Avançado

### Custom Feature Engineering:

```python
from features.advanced_features import ScalpingFeatureEngineer

engineer = ScalpingFeatureEngineer(config={...})

# Adiciona features customizadas
df = engineer.build_all_features(
    df,
    orderbook=orderbook_data,  # Opcional
    trades=trades_data         # Opcional
)
```

### Custom Target:

```python
from features.target_engineering import TargetEngineer

target_eng = TargetEngineer({
    'tp_pct': 0.005,  # 0.5% TP
    'sl_pct': 0.003,  # 0.3% SL (assimétrico)
    'horizons': [3, 5, 10, 15],  # Mais horizontes
    'fees': 0.0006,
    'slippage': 0.0001
})

df = target_eng.create_all_targets(df)
```

### Ensemble Models:

```python
# Treina modelo para trending e ranging separadamente
model_trending = train_for_regime(df[df['is_trending'] == 1])
model_ranging = train_for_regime(df[df['is_ranging'] == 1])

# Produção: usa regime detector
if current_regime == 'trending':
    prediction = model_trending.predict(X)
else:
    prediction = model_ranging.predict(X)
```

---

## 📊 Métricas e Validação

### Métricas de Sucesso:

Para considerar o modelo **PRODUCTION-READY**, ele deve atingir:

| Métrica | Mínimo | Ideal | Excelente |
|---------|--------|-------|-----------|
| **Win Rate** | 50% | 55% | **60%+** |
| **Precision (LONG)** | 55% | 60% | **65%+** |
| **Recall (LONG)** | 40% | 50% | **60%+** |
| **F1 Score** | 45% | 52% | **60%+** |
| **Sharpe Ratio** | 1.0 | 1.5 | **2.0+** |
| **Max Drawdown** | < 20% | < 15% | **< 10%** |
| **Profit Factor** | > 1.2 | > 1.5 | **> 2.0** |
| **Latência** | < 100ms | < 50ms | **< 20ms** |

### Interpretação:

**Win Rate > 60%:**
- De cada 10 trades, 6+ são lucrativos
- Essencial para scalping (alta frequência)

**Sharpe > 2.0:**
- Retorno ajustado por risco excelente
- Retorno consistente, baixa volatilidade

**Latência < 20ms:**
- LightGBM permite inferência ultra-rápida
- Crítico para scalping 15min

---

## 🔧 Troubleshooting

### Problema: Modelo com baixo Win Rate

**Solução:**
1. Aumentar `min_confidence` no config (0.30 → 0.40)
2. Otimizar TP/SL targets
3. Adicionar mais features de order flow
4. Aumentar regularização (L1/L2)

### Problema: Overfitting (train >> test)

**Solução:**
1. Aumentar `min_child_samples` (100 → 200)
2. Aumentar `reg_alpha`, `reg_lambda`
3. Reduzir `num_leaves` (31 → 15)
4. Usar menos dados de treinamento (365d → 180d)

### Problema: Modelo muito lento

**Solução:**
1. Reduzir `n_estimators` (500 → 300)
2. Reduzir `max_depth` (6 → 4)
3. Feature selection (remover < 1% importance)
4. Usar LightGBM ao invés de XGBoost/RF

---

## 📚 Referências

### Papers:

- **LightGBM:** [Ke et al., 2017 - A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- **Optuna:** [Akiba et al., 2019 - Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)

### Docs:

- [LightGBM Official Docs](https://lightgbm.readthedocs.io/)
- [Optuna Official Docs](https://optuna.readthedocs.io/)
- [Scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

---

## 🎯 Roadmap

### V2.1 (Futuro):
- [ ] LSTM/GRU para sequências temporais
- [ ] Transformer para attention mechanism
- [ ] AutoML com AutoGluon
- [ ] Real-time model updates
- [ ] Multi-symbol training
- [ ] Reinforcement Learning (DQN/PPO)

---

## 📝 Changelog

### V2.0 (2025-11-14) - Initial Release
- ✅ LightGBM pipeline completo
- ✅ 150+ features de scalping
- ✅ Multi-horizon targets
- ✅ Walk-forward validation
- ✅ Backtest realista
- ✅ Optuna optimization
- ✅ Production-ready

---

## 📧 Support

Para dúvidas ou issues:
1. Consulte este README
2. Verifique `ANALISE_MODELO_ML_E_PROPOSTA.md`
3. Revise `CONSIDERACOES_MODELO_ATUAL.md`

---

## ⚖️ License

Proprietary - Internal Use Only

---

## 🎉 Conclusão

Este pipeline foi desenvolvido para criar modelos **10/10** para scalping.

**Próximos passos:**

1. ✅ Treinar primeiro modelo: `python train_scalping_model.py`
2. ✅ Avaliar métricas (Win Rate > 60%?)
3. ✅ Otimizar com Optuna (se necessário)
4. ✅ Backtest realista
5. ✅ Deploy em produção nos bots

**Objetivo:** Win Rate > 60% | Sharpe > 2.0 | Latência < 20ms

**Let's build the perfect scalping model! 🚀**
