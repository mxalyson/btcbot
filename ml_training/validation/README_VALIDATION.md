# 🔬 Validação Universal de Modelos ML

Scripts para validar e comparar qualquer modelo .pkl treinado.

## 📁 Arquivos

### 1. `validate_any_model.py` - Validador Universal

Valida **qualquer modelo .pkl** (V1, V2, antigo, novo) com diferentes configurações.

**Features**:
- ✅ Detecta automaticamente modelo binary (2 classes) ou multiclass (3 classes)
- ✅ Testa múltiplos níveis de confidence (0%, 50%, 55%, 60%, 65%, 70%)
- ✅ Testa múltiplas configurações de TP/SL
- ✅ Gera tabela comparativa completa
- ✅ Recomenda melhor configuração (weighted score)
- ✅ Salva resultados em CSV para análise

**Uso**:
```bash
cd ml_training/validation

# Validar Master V2.0
python validate_any_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --symbol BTCUSDT \
  --days 90 \
  --save-csv results_v2.csv

# Validar modelo antigo
python validate_any_model.py \
  --model ../../ml_model_master_scalper_365d.pkl \
  --symbol BTCUSDT \
  --days 90 \
  --save-csv results_antigo.csv
```

**Parâmetros**:
- `--model`: Caminho para arquivo .pkl (obrigatório)
- `--symbol`: Símbolo (default: BTCUSDT)
- `--timeframe`: Timeframe (default: 15m)
- `--days`: Período de teste em dias (default: 90)
- `--save-csv`: Salvar resultados em CSV (opcional)

**Output**:
```
TP   SL   Conf  | Trades     WR | ROI       PF  Sharpe     DD | LONG  SHORT | LongWR  ShortWR
---------------------------------------------------------
2.0  1.5  0     |    515  49.3% | -27.97%  0.77   1.31  -31.2% |  299    216 |  46.5%   53.2%
2.0  1.5  50    |    515  49.3% | -27.97%  0.77   1.31  -31.2% |  299    216 |  46.5%   53.2%
2.0  1.5  60    |    387  52.1% |  +8.42%  1.12   1.85  -18.3% |  220    167 |  50.0%   54.5%
...

🏆 BEST OVERALL:
   TP: 2.5x ATR
   SL: 1.0x ATR
   MIN_ML_CONFIDENCE: 0.65

📊 Expected Performance:
   ROI: +15.3%
   Win Rate: 54.2%
   Profit Factor: 1.32
```

---

### 2. `compare_models.py` - Comparador de Modelos

Compara **dois modelos lado a lado** com as mesmas configurações.

**Features**:
- ✅ Compara V1 vs V2, ou novo vs antigo
- ✅ Tabela side-by-side de todas as configurações
- ✅ Determina vencedor automaticamente
- ✅ Mostra diferença de performance
- ✅ Salva comparação em CSV

**Uso**:
```bash
cd ml_training/validation

# Comparar Master V2.0 vs Modelo Antigo
python compare_models.py \
  --model1 ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --model2 ../../ml_model_master_scalper_365d.pkl \
  --symbol BTCUSDT \
  --days 90 \
  --save-csv comparison.csv
```

**Parâmetros**:
- `--model1`: Caminho para primeiro modelo .pkl (obrigatório)
- `--model2`: Caminho para segundo modelo .pkl (obrigatório)
- `--symbol`: Símbolo (default: BTCUSDT)
- `--timeframe`: Timeframe (default: 15m)
- `--days`: Período de teste em dias (default: 90)
- `--save-csv`: Salvar resultados em CSV (opcional)

**Output**:
```
📊 SIDE-BY-SIDE COMPARISON

Config               | Model 1 (scalping_v2)                  | Model 2 (master_scalper_old)           | Winner
---------------------------------------------------------
TP2.0 SL1.5 C0%     | ROI:-27.9% WR:49.3% T:515              | ROI:+42.1% WR:56.2% T:387              | Model 2 ✅
TP2.0 SL1.5 C50%    | ROI:-27.9% WR:49.3% T:515              | ROI:+42.1% WR:56.2% T:387              | Model 2 ✅
TP2.5 SL1.0 C60%    | ROI: +8.4% WR:52.1% T:287              | ROI:+58.3% WR:59.1% T:298              | Model 2 ✅
...

🏆 OVERALL WINNER

📊 Model 1 Best: scalping_v2
   Config: TP2.5 SL1.0 Conf60%
   ROI: +8.42% | WR: 52.1% | PF: 1.12

📊 Model 2 Best: master_scalper_old
   Config: TP2.0 SL1.5 Conf50%
   ROI: +42.1% | WR: 56.2% | PF: 1.43

🏆 WINNER: Model 2 (master_scalper_old)
   Better ROI by +33.7%
```

---

### 3. `analyze_predictions.py` - Análise de Predições

Analisa predições do modelo para identificar viés e problemas.

**Features**:
- ✅ Distribuição de classes (DOWN/UP)
- ✅ Confidence por threshold
- ✅ Análise temporal (predições por dia)
- ✅ Correlação com volatilidade (ATR)

**Uso**:
```bash
cd ml_training/validation

python analyze_predictions.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90
```

**Output**:
```
📊 PREDICTION ANALYSIS

Model type: BINARY (2 classes)

Class distribution:
  DOWN (0): 3,525 (41.2%)
  UP (1):   5,033 (58.8%)

Predictions by confidence threshold:
  >= 50%: 8,558 signals (41.2% DOWN, 58.8% UP)
  >= 55%: 7,234 signals (40.8% DOWN, 59.2% UP)
  >= 60%: 5,912 signals (39.5% DOWN, 60.5% UP)
  >= 65%: 4,387 signals (38.1% DOWN, 61.9% UP)
  >= 70%: 2,891 signals (36.2% DOWN, 63.8% UP)

⚠️ VIÉS DETECTADO: Modelo gera muito mais LONGs (58.8%)
```

---

## 🎯 Casos de Uso

### Caso 1: Validar Novo Modelo

```bash
# 1. Validar Master V2.0 com múltiplas configurações
python validate_any_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 \
  --save-csv results_v2.csv

# 2. Analisar predições
python analyze_predictions.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90

# 3. Se WR > 52% e ROI > 0% → Deploy em paper!
```

---

### Caso 2: Comparar V2.0 vs Antigo

```bash
# Comparação lado a lado
python compare_models.py \
  --model1 ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --model2 ../../ml_model_master_scalper_365d.pkl \
  --days 90 \
  --save-csv comparison_v2_vs_old.csv

# Vencedor é determinado automaticamente!
```

---

### Caso 3: Testar Diferentes Períodos

```bash
# Testar em 30, 60, 90, 180 dias
for days in 30 60 90 180; do
  echo "Testing $days days..."
  python validate_any_model.py \
    --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
    --days $days \
    --save-csv "results_${days}d.csv"
done
```

---

### Caso 4: Grid Search de Parâmetros

**O validador já faz isso automaticamente!**

Testa:
- **Confidence**: 0%, 50%, 55%, 60%, 65%, 70%
- **TP/SL**:
  - 2.0 ATR TP / 1.5 ATR SL (original)
  - 2.5 ATR TP / 1.0 ATR SL (R:R alto)
  - 3.0 ATR TP / 1.0 ATR SL (R:R muito alto)
  - 2.0 ATR TP / 1.0 ATR SL (SL tight)
  - 1.5 ATR TP / 1.0 ATR SL (conservador)

**Total**: 6 × 5 = **30 combinações testadas automaticamente!**

---

## 📊 Interpretação dos Resultados

### ✅ Modelo BOM (Deploy em Paper)

```
ROI: > +20%
Win Rate: > 55%
Profit Factor: > 1.3
Sharpe Ratio: > 1.5
Max Drawdown: < -15%
Total Trades: > 50
```

### ⚠️ Modelo MÉDIO (Ajustar Parâmetros)

```
ROI: +5% a +20%
Win Rate: 52% a 55%
Profit Factor: 1.1 a 1.3
Sharpe Ratio: 1.0 a 1.5
Max Drawdown: -15% a -25%
```

### ❌ Modelo RUIM (Retreinar ou Descartar)

```
ROI: < +5% ou negativo
Win Rate: < 52%
Profit Factor: < 1.1
Max Drawdown: > -25%
```

---

## 🔧 Personalização

### Modificar Configurações de Teste

Edite o arquivo `validate_any_model.py`:

```python
# Linha ~680
confidence_levels = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]  # Adicionar mais níveis

tp_sl_configs = [
    (2.0, 1.5),  # Original
    (2.5, 1.0),  # Adicionar suas próprias configurações
    (3.0, 1.5),
    (4.0, 2.0),
]
```

### Modificar Fees e Slippage

```bash
# No código, linha ~670
fees_pct=0.06,      # 0.06% (Bybit taker)
slippage_pct=0.01   # 0.01%
```

Ou edite o arquivo e altere os defaults.

---

## 📝 Exemplo Completo de Workflow

```bash
#!/bin/bash
# Workflow completo de validação

MODEL_V2="ml_training/outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl"
MODEL_OLD="ml_model_master_scalper_365d.pkl"

echo "🔬 1. Validando Master V2.0..."
python ml_training/validation/validate_any_model.py \
  --model $MODEL_V2 \
  --days 90 \
  --save-csv results/v2_validation.csv

echo "🔬 2. Validando modelo antigo..."
python ml_training/validation/validate_any_model.py \
  --model $MODEL_OLD \
  --days 90 \
  --save-csv results/old_validation.csv

echo "⚔️  3. Comparando modelos..."
python ml_training/validation/compare_models.py \
  --model1 $MODEL_V2 \
  --model2 $MODEL_OLD \
  --days 90 \
  --save-csv results/comparison.csv

echo "📊 4. Analisando predições V2.0..."
python ml_training/validation/analyze_predictions.py \
  --model $MODEL_V2 \
  --days 90

echo "✅ Validação completa!"
```

---

## 🆚 Diferenças entre Scripts

| Feature | validate_any_model.py | compare_models.py | backtest_ml_model.py |
|---------|----------------------|-------------------|----------------------|
| Testa 1 modelo | ✅ | ❌ | ✅ |
| Compara 2 modelos | ❌ | ✅ | ❌ |
| Múltiplos TP/SL | ✅ (5 configs) | ✅ (3 configs) | ❌ (1 config) |
| Múltiplos confidence | ✅ (6 níveis) | ✅ (4 níveis) | ❌ (1 nível) |
| Auto-detecta tipo modelo | ✅ | ✅ | ✅ |
| Recomenda melhor config | ✅ | ✅ | ❌ |
| Salva CSV | ✅ | ✅ | ❌ |
| Total testes | 30 | 12 por modelo | 1 |

**Recomendação**:
- Use `validate_any_model.py` para **testar 1 modelo completo**
- Use `compare_models.py` para **comparar 2 modelos**
- Use `backtest_ml_model.py` para **teste rápido** de 1 configuração específica

---

## 🐛 Troubleshooting

### Erro: "No module named pandas"

```bash
# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install pandas numpy lightgbm scikit-learn
```

### Erro: "Model not found"

```bash
# Usar caminho absoluto ou relativo correto
python validate_any_model.py --model ../outputs/modelo.pkl

# Ou caminho absoluto
python validate_any_model.py --model /full/path/to/modelo.pkl
```

### Warning: "Missing features"

✅ **Normal!** O script preenche features faltantes com 0 automaticamente.

Se o modelo tem features que não existem no seu dataset, elas serão preenchidas com 0.

---

## 📚 Referências

- `validate_strategy.py` (original) - Validador do modelo antigo
- `backtest_ml_model.py` - Backtest simples (1 configuração)
- `train_scalping_model.py` - Script de treinamento

---

## 🏆 Objetivo Final

**Meta**: Encontrar configuração com:
- Win Rate > 55%
- ROI anual > 60%
- Profit Factor > 1.3
- Max Drawdown < -15%

Use estes scripts para **otimizar** seus modelos e encontrar a **melhor configuração** antes de deploy em produção! 🚀
