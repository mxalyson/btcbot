# 📊 ANÁLISE DO MODELO ML ATUAL E PROPOSTA DE MODELO OTIMIZADO PARA SCALPING

## 🔍 ANÁLISE DO MODELO ATUAL

### ✅ Pontos Fortes:

1. **Feature Engineering Sólido**:
   - 70+ features técnicas e derivadas
   - Price Action (SMC): CHOCH, BOS, FVG, Swing Points
   - Indicadores técnicos: EMAs, RSI, MACD, ADX, Bollinger, ATR, VWAP
   - Features de momentum: returns 1/5/10/20 períodos
   - Features de volume: ratio, momentum, std
   - Microestrutura: OBI, spread

2. **Engenharia de Features Avançada**:
   ```python
   # Features derivadas inteligentes
   - trend_strength = (EMA50 - EMA200) / EMA200 * 100
   - volatility_regime = ATR / ATR_MA(50)
   - price_position = (Close - Low20) / (High20 - Low20)
   - volume_momentum = Volume.pct_change(5)
   - price_acceleration = Close.diff(2) - Close.diff(1)
   ```

3. **Gestão de Confiança**:
   - Threshold ajustável (MIN_ML_CONFIDENCE)
   - Cálculo: `abs(prob - 0.5) * 2` (0 a 1)
   - Filtra sinais fracos

### ⚠️ Limitações Identificadas:

1. **Tipo de Modelo Não Revelado**:
   - Código usa `pickle.load(model)` genérico
   - Não sei se é RandomForest, XGBoost, LSTM, etc.
   - Impacta performance em scalping

2. **Features Podem Ser Lagging**:
   - EMAs são lagging indicators
   - RSI/MACD reagem com delay
   - Em scalping 15m, cada segundo conta

3. **Falta de Features de Microestrutura Avançada**:
   - OBI básico (apenas 5 níveis)
   - Sem análise de agressividade de trades
   - Sem features de liquidez profunda
   - Sem order flow imbalance

4. **Sem Regime Detection**:
   - Não diferencia trending vs ranging
   - Não adapta strategy ao regime
   - Performance inconsistente em diferentes condições

5. **Target Não Otimizado para Scalping**:
   - Não sei qual target está usando
   - Provavelmente: futuro retorno binário
   - Ideal para scalping: múltiplos horizontes

---

## 🚀 PROPOSTA: MODELO PERFEITO PARA SCALPING 15min

### 🎯 Objetivos:

- **Latência ultra-baixa**: Previsão < 100ms
- **Alta precisão**: Win rate > 60% (scalping needs > 55%)
- **Baixo drawdown**: Evitar sequências longas de perdas
- **Adaptativo**: Funciona em trending e ranging
- **Explícito**: Sabe QUANDO não operar

---

## 📦 STACK TECNOLÓGICO RECOMENDADO

### Modelo Principal: **LightGBM** (não XGBoost!)

**Por que LightGBM?**
```
✅ 10-20x mais rápido que XGBoost em inferência
✅ Usa menos memória (crítico para VPS)
✅ Nativo em categorical features
✅ Excelente para dados tabulares
✅ Suporta early stopping (evita overfit)
✅ Feature importance built-in
```

**Comparação de Latência:**
```
RandomForest:  ~500ms (muito lento!)
XGBoost:       ~100ms (OK)
LightGBM:      ~10-20ms (PERFEITO para scalping!)
```

### Arquitetura Ensemble (Opcional Avançado):

```
1. LightGBM Trend Detector → Trending vs Ranging
2. LightGBM Scalping Model → Se trending
3. LightGBM Mean Reversion → Se ranging
4. Meta-Learner → Combina outputs
```

---

## 🧬 FEATURES OTIMIZADAS PARA SCALPING

### Categoria 1: Price Action Puro (Alta Frequência)

```python
# Microestrutura de curto prazo
'price_change_1m': close.diff(1),
'price_change_3m': close.diff(3),
'price_change_5m': close.diff(5),

# Momentum extremo curto prazo
'roc_1': ((close - close.shift(1)) / close.shift(1)) * 100,
'roc_3': ((close - close.shift(3)) / close.shift(3)) * 100,

# Volatility intrabar
'high_low_ratio': (high - low) / close,
'body_to_range': abs(close - open) / (high - low + 1e-8),

# Wick analysis (rejeição)
'upper_wick_pct': (high - max(open, close)) / (high - low + 1e-8),
'lower_wick_pct': (min(open, close) - low) / (high - low + 1e-8),
```

### Categoria 2: Order Flow & Microestrutura

```python
# Order Book Imbalance (crítico!)
'obi_top5': (bid_vol[0:5] - ask_vol[0:5]) / (bid_vol[0:5] + ask_vol[0:5]),
'obi_top10': (bid_vol[0:10] - ask_vol[0:10]) / (bid_vol[0:10] + ask_vol[0:10]),

# Spread dynamics
'spread_pct': (ask - bid) / mid_price * 100,
'spread_volatility': spread.rolling(5).std(),

# Trade aggressiveness (precisa trades tick-by-tick)
'buy_volume_ratio': buy_volume / total_volume,
'aggressive_buy_pct': aggressive_buys / total_trades,

# Liquidez
'bid_depth': sum(bid_volumes[0:10]),
'ask_depth': sum(ask_volumes[0:10]),
'liquidity_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth),
```

### Categoria 3: Volume Profile

```python
# Volume-weighted
'vwap': cumsum(volume * price) / cumsum(volume),
'distance_from_vwap': (close - vwap) / vwap * 100,

# Volume momentum
'volume_ratio_5': volume / volume.rolling(5).mean(),
'volume_surge': (volume > volume.rolling(20).mean() * 1.5).astype(int),

# Delta Volume (se tiver trades)
'delta_volume': buy_volume - sell_volume,
'cumulative_delta': delta_volume.cumsum(),
```

### Categoria 4: Regime Detection Features

```python
# Trend strength
'adx': ADX(14),
'adx_slope': ADX.diff(3),

# Volatility regime
'atr_normalized': atr / close,
'atr_percentile': atr.rolling(100).rank(pct=True),

# Choppiness
'choppiness_index': ChoppinessIndex(14),  # < 38.2 = trending

# Range detection
'is_ranging': (bb_width < bb_width.rolling(50).quantile(0.3)).astype(int),
```

### Categoria 5: Time-Based Features (Scalping Específico)

```python
# Padrões de tempo
'hour': datetime.hour,
'minute': datetime.minute,
'is_market_open': (9 <= hour <= 16).astype(int),  # Ajustar para crypto
'is_high_liquidity_time': hour.isin([9, 10, 14, 15]).astype(int),

# Session features (Asia, Europe, US overlap)
'is_overlap': is_asia_europe_overlap or is_europe_us_overlap,
```

---

## 🎲 TARGET ENGINEERING (CRUCIAL!)

### Multi-Horizon Targets:

```python
# Não use apenas 1 target binário!
# Use múltiplos horizontes:

# Target 1: Quick scalp (1-3 candles)
'target_3min': (future_high_3bars > entry * 1.002).astype(int) &
               (future_low_3bars > entry * 0.998)  # +0.2% sem -0.2% SL

# Target 2: Standard scalp (3-5 candles)
'target_5min': (future_high_5bars > entry * 1.003).astype(int) &
               (future_low_5bars > entry * 0.997)  # +0.3% sem -0.3% SL

# Target 3: Extended scalp (5-10 candles)
'target_10min': (future_high_10bars > entry * 1.005).astype(int) &
                (future_low_10bars > entry * 0.995)  # +0.5% sem -0.5% SL
```

### Regression Target (Mais Avançado):

```python
# Ao invés de classificação, use regressão
# Prediz: "Quanto vai subir nos próximos 15min?"

target_regression = (future_high_15min - entry) / entry * 100  # em %

# Benefício: Você sabe QUÃO forte é o sinal
# Se prediz +0.1% → skip
# Se prediz +0.5% → trade!
```

---

## 🏋️ PROCESSO DE TREINAMENTO

### 1. Coleta de Dados

```python
# Timeframes para scalping 15min
primary_tf = '15m'
secondary_tf = ['5m', '1h']  # Multi-timeframe

# Período de dados
training_period = 6  # meses (muito crypto = overfit)
validation_period = 1  # mês
test_period = 1  # mês

# Features de múltiplos timeframes
features_15m = build_features(data_15m)
features_5m_resampled = build_features(data_5m).resample('15m')
features_1h = build_features(data_1h).resample('15m')

X = pd.concat([features_15m, features_5m_resampled, features_1h], axis=1)
```

### 2. Data Split (Walk-Forward)

```python
# NÃO use train_test_split aleatório!
# Use walk-forward (time-series aware)

from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train and validate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

### 3. Treinamento LightGBM Otimizado

```python
import lightgbm as lgb

# Parâmetros otimizados para scalping
params = {
    'objective': 'binary',  # ou 'regression' se usar target contínuo
    'metric': 'auc',  # ou 'rmse' para regressão
    'boosting_type': 'gbdt',

    # Performance
    'num_leaves': 31,  # Não muito alto (overfit)
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,

    # Regularização (crítico!)
    'min_child_samples': 100,  # Evita overfitting
    'reg_alpha': 0.1,  # L1
    'reg_lambda': 0.1,  # L2
    'colsample_bytree': 0.8,  # Feature sampling
    'subsample': 0.8,  # Row sampling
    'subsample_freq': 5,

    # Scalping específico
    'is_unbalance': True,  # Dados geralmente desbalanceados
    'scale_pos_weight': 1.5,  # Ajustar se win rate baixo

    # Speed
    'num_threads': 4,
    'verbose': -1
}

# Train com early stopping
model = lgb.LGBMClassifier(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=50,  # Para se não melhorar
    verbose=100
)
```

### 4. Feature Selection

```python
# Após treinar, veja feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Remove features inúteis (< 1% importance)
important_features = importance[importance['importance'] > 0.01]['feature'].tolist()

# Re-treina com features selecionadas
X_train_selected = X_train[important_features]
model_final = lgb.LGBMClassifier(**params)
model_final.fit(X_train_selected, y_train)
```

---

## 📊 VALIDAÇÃO (CRÍTICO!)

### Métricas para Scalping:

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Predições
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métricas básicas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")  # Quando diz BUY, acerta?
print(f"Recall: {recall:.3f}")  # Pega todos os moves?
print(f"F1: {f1:.3f}")
print(f"AUC: {auc:.3f}")

# ⚠️ ATENÇÃO: Para scalping, precision > recall!
# Você quer poucos sinais mas CERTOS
# Precision 65%+ é excelente
# Recall 40% é OK (miss some trades, but safe)
```

### Backtesting Realista:

```python
# Simula trades com o modelo
def backtest_scalping(model, X_test, prices, fees=0.06):
    signals = model.predict(X_test)
    equity = 10000
    trades = []

    for i, signal in enumerate(signals):
        if signal == 1:  # BUY
            entry = prices[i]

            # Simula próximos bars
            for j in range(1, 11):  # Próximos 10 bars
                if i + j >= len(prices):
                    break

                current = prices[i + j]

                # TP hit?
                if current >= entry * 1.003:  # +0.3%
                    pnl = (1.003 - 1) * equity - (equity * fees / 100)
                    equity += pnl
                    trades.append({'type': 'win', 'pnl': pnl})
                    break

                # SL hit?
                if current <= entry * 0.997:  # -0.3%
                    pnl = (0.997 - 1) * equity - (equity * fees / 100)
                    equity += pnl
                    trades.append({'type': 'loss', 'pnl': pnl})
                    break

    win_rate = len([t for t in trades if t['type'] == 'win']) / len(trades)
    total_pnl = sum([t['pnl'] for t in trades])

    return {
        'final_equity': equity,
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': len(trades)
    }
```

---

## 🎓 TREINAMENTO AVANÇADO: ENSEMBLING

### Estratégia Multi-Model:

```python
# Modelo 1: Trend Following (para trending markets)
model_trend = lgb.LGBMClassifier(**params_trend)
model_trend.fit(X_train_trending, y_train_trending)

# Modelo 2: Mean Reversion (para ranging markets)
model_range = lgb.LGBMClassifier(**params_range)
model_range.fit(X_train_ranging, y_train_ranging)

# Meta-Modelo: Decide qual usar
def predict_with_regime(X):
    # Detecta regime
    is_trending = X['adx'] > 25

    if is_trending:
        return model_trend.predict_proba(X)[:, 1]
    else:
        return model_range.predict_proba(X)[:, 1]
```

---

## 🔧 OTIMIZAÇÃO DE HIPERPARÂMETROS

### Usando Optuna (melhor que GridSearch):

```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    model = lgb.LGBMClassifier(**params, objective='binary', metric='auc')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=0)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    return auc

# Otimiza
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

---

## 📁 ESTRUTURA DE CÓDIGO SUGERIDA

```
/ml_training/
├── data/
│   ├── fetch_data.py          # Baixa dados históricos
│   └── preprocess.py           # Limpa e prepara dados
├── features/
│   ├── price_action.py         # Features de PA
│   ├── orderflow.py            # Features de microestrutura
│   ├── regime.py               # Regime detection
│   └── engineer.py             # Feature engineering master
├── models/
│   ├── lgbm_scalping.py        # Modelo principal
│   ├── ensemble.py             # Multi-model
│   └── regime_detector.py      # Detecta trending/ranging
├── validation/
│   ├── backtest.py             # Backtesting engine
│   ├── metrics.py              # Métricas customizadas
│   └── walk_forward.py         # Walk-forward validation
├── optimization/
│   ├── hyperparams.py          # Optuna optimization
│   └── feature_selection.py    # Seleção de features
└── train.py                    # Script principal
```

---

## ✅ CHECKLIST DE QUALIDADE

### Antes de Deploy:

- [ ] Win rate > 55% em out-of-sample
- [ ] Sharpe ratio > 1.5
- [ ] Max drawdown < 15%
- [ ] Testado em diferentes regimes de mercado
- [ ] Latência de inferência < 50ms
- [ ] Feature importance analisada
- [ ] Correlação entre features < 0.8
- [ ] Validado em walk-forward
- [ ] Testado com fees realistas (0.06%)
- [ ] Testado com slippage realista (0.05%)

---

## 🎯 RESUMO: MODELO PERFEITO PARA SCALPING

```
Algoritmo: LightGBM
Features: 100-150 (selecionadas)
  - Price action: 30%
  - Order flow: 25%  ← CRÍTICO!
  - Regime: 15%
  - Volume: 15%
  - Time: 15%

Target: Multi-horizon (3/5/10 bars)
Training: Walk-forward (6 meses)
Validation: 1 mês out-of-sample

Métricas alvo:
  - Precision: > 65%
  - Win rate: > 55%
  - Sharpe: > 1.5
  - Latência: < 20ms

Ensemble: Opcional (trending + ranging models)
```

---

## 🚀 PRÓXIMOS PASSOS

Se quiser que eu **implemente** este modelo:

1. **Criação de pipeline de features avançadas**
2. **Script de treinamento LightGBM otimizado**
3. **Backtesting realista com fees/slippage**
4. **Validação walk-forward**
5. **Exportação para produção (.pkl otimizado)**

Quer que eu crie os scripts de treinamento? Posso fazer um sistema completo de ML pipeline para scalping!
