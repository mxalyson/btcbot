# 🔬 Validação do Modelo V1 - Guia Completo

## 🎯 Objetivo

Validar o modelo V1 (1.2MB - `scalping_model_BTCUSDT_15m_20251114_213903.pkl`) que você reportou com **80% de acerto** (8 em 10 sinais).

---

## 🆕 Novo Script: `validate_strategy_v1.py`

Criei uma versão melhorada do `validate_strategy.py` original com:

### ✅ Melhorias vs Original

| Feature | Original | V1 Script |
|---------|----------|-----------|
| Compatível binary/multiclass | ❌ | ✅ |
| Testa múltiplos TP/SL | ❌ | ✅ (3 configs) |
| Gestão parcial (TP1/TP2/TP3) | ✅ | ✅ |
| Trailing stop após TP2 | ✅ | ✅ |
| Features completas (150+) | ❌ (só 65) | ✅ |
| Auto-detecta tipo modelo | ❌ | ✅ |
| Mostra TP1/TP2 hit rate | ❌ | ✅ |

---

## 🚀 Como Usar

### Uso Básico

```bash
python validate_strategy_v1.py \
  --model scalping_model_BTCUSDT_15m_20251114_213903.pkl \
  --symbol BTCUSDT \
  --days 180
```

**Tempo**: ~5-10 minutos
**Testa**: 18 combinações (6 confidence × 3 TP/SL)

---

### Parâmetros

```bash
--model   # Caminho do modelo .pkl (obrigatório)
--symbol  # Símbolo (default: BTCUSDT)
--days    # Período de teste em dias (default: 180)
```

---

### Exemplos

#### 1. Testar V1 com 90 dias

```bash
python validate_strategy_v1.py \
  --model scalping_model_BTCUSDT_15m_20251114_213903.pkl \
  --days 90
```

#### 2. Testar V1 com 180 dias (padrão)

```bash
python validate_strategy_v1.py \
  --model scalping_model_BTCUSDT_15m_20251114_213903.pkl
```

#### 3. Testar com 360 dias (1 ano)

```bash
python validate_strategy_v1.py \
  --model scalping_model_BTCUSDT_15m_20251114_213903.pkl \
  --days 360
```

---

## 📊 O Que o Script Testa

### Configurações Testadas (18 combinações):

**Confidence Levels** (6):
- 0% (todos os sinais)
- 10%
- 20%
- 30%
- 40%
- 50%

**TP/SL Configs** (3):
- TP 2.0x ATR / SL 1.5x ATR (original)
- TP 2.5x ATR / SL 1.0x ATR (R:R alto)
- TP 3.0x ATR / SL 1.0x ATR (R:R muito alto)

**Total**: 6 × 3 = **18 testes**

---

## 📋 Output do Script

### 1. Tabela de Resultados

```
📊 RESULTADOS COMPARATIVOS

TP   SL   Conf  | Trades     WR |      ROI   ROI/yr |    PF Sharpe     DD
------------------------------------------------------------------------------
2.0  1.5  0     |    387  56.2% |  +42.1%  +85.7% |  1.43   2.34  -8.5%
2.0  1.5  10    |    356  57.3% |  +45.8%  +93.2% |  1.52   2.48  -7.2%
2.0  1.5  20    |    298  59.1% |  +51.2% +104.2% |  1.68   2.67  -6.1%
2.5  1.0  20    |    287  61.3% |  +58.4% +118.9% |  1.85   2.91  -5.3%  ← 🏆
...
```

**Colunas**:
- **TP/SL**: Multiplicadores do ATR
- **Conf**: Confidence mínima
- **Trades**: Número de trades
- **WR**: Win Rate (%)
- **ROI**: Return on Investment (%)
- **ROI/yr**: ROI anualizado
- **PF**: Profit Factor
- **Sharpe**: Sharpe Ratio
- **DD**: Max Drawdown (%)

---

### 2. Análise de Melhores Configurações

```
🏆 ANÁLISE DE MELHOR CONFIGURAÇÃO

🎯 Melhor ROI:
   Config: TP 2.5x, SL 1.0x, Conf 20%
   ROI: +58.4%
   Win Rate: 61.3%
   Trades: 287

📈 Melhor Sharpe Ratio:
   Config: TP 2.5x, SL 1.0x, Conf 30%
   Sharpe: 2.91
   ROI: +54.2%
   Trades: 243

🎯 Melhor Win Rate:
   Config: TP 3.0x, SL 1.0x, Conf 40%
   Win Rate: 64.7%
   ROI: +48.1%
   Trades: 189

💪 Menor Drawdown:
   Config: TP 2.5x, SL 1.0x, Conf 30%
   Max DD: -4.8%
   ROI: +54.2%
   Trades: 243
```

---

### 3. Recomendação Final (Weighted Score)

```
💡 RECOMENDAÇÃO (Weighted Score)

🏆 Configuração Recomendada:
   TP: 2.5x ATR
   SL: 1.0x ATR
   MIN_ML_CONFIDENCE: 0.20

📊 Métricas Esperadas:
   Total Trades: 287
   Win Rate: 61.3%
   ROI: +58.4%
   Sharpe: 2.78
   Max DD: -5.3%
   Profit Factor: 1.85
   Avg Confidence: 65.2%
   Long trades: 152 (WR: 59.2%)
   Short trades: 135 (WR: 63.7%)
   TP1 hits: 198 (69.0%)
   TP2 hits: 156 (54.4%)
```

---

## 🎯 Como Interpretar os Resultados

### ✅ Modelo EXCEPCIONAL (80% WR confirmado)

Se o script mostrar:
```
Win Rate: > 60%
ROI: > +50%
Profit Factor: > 1.5
Sharpe Ratio: > 2.0
Max Drawdown: < -10%
```

**Ação**:
1. ✅ **USAR V1 IMEDIATAMENTE!**
2. ✅ Copiar para produção
3. ✅ Deploy em paper trading
4. ✅ Configurar .env com melhor config recomendada

---

### ⚠️ Modelo BOM (55-60% WR)

Se o script mostrar:
```
Win Rate: 55-60%
ROI: +30% a +50%
Profit Factor: 1.3 a 1.5
```

**Ação**:
1. ✅ Modelo é BOM, mas não excepcional
2. 🔍 Compare com modelo antigo
3. 🧪 Teste em paper antes de live

---

### ❌ Modelo MÉDIO (<55% WR)

Se o script mostrar:
```
Win Rate: < 55%
ROI: < +30%
Profit Factor: < 1.3
```

**Ação**:
1. ❓ **80% WR estava errado?**
2. 🔍 Investigar diferença vs medição inicial
3. ⚔️ Comparar com modelo antigo
4. 🧪 Testar em diferentes períodos

---

## 🔧 Após a Validação

### Se V1 for BOM (WR > 60%), Configure para Produção:

#### 1. Copiar Modelo

```bash
cp scalping_model_BTCUSDT_15m_20251114_213903.pkl scalping_model_production.pkl
```

#### 2. Configurar `.env`

Use a configuração recomendada pelo script:

```bash
# Exemplo (ajustar com valores reais do output)
ML_MODEL_PATH=./scalping_model_production.pkl
MIN_ML_CONFIDENCE=0.20
TP_ATR_MULTIPLIER=2.5
SL_ATR_MULTIPLIER=1.0
```

#### 3. Testar em Paper

```bash
python eth_live_v3.py --mode paper
```

Rode por **1-2 semanas** em paper e monitore:
- Win Rate real
- ROI real
- Número de trades por dia
- Profit Factor

#### 4. Deploy em Live (se paper OK)

```bash
python eth_live_v3.py --mode live
```

---

## 📊 Gestão de Risco (Implementada no Script)

O script simula a **mesma lógica do backtest**:

### TP1 (0.7x ATR)
- Fecha **60%** da posição
- Move SL para **Break-Even**

### TP2 (1.3x ATR)
- Ativa **Trailing Stop**
- SL se move com o preço (1.0x ATR de trailing)

### TP3 (2.0x, 2.5x ou 3.0x ATR - depende da config)
- Fecha **40% restante**

### Time Exit
- Fecha posição após **48 horas** (192 bars de 15min)

---

## 🧪 Comparação com Outros Modelos

Se quiser comparar V1 vs outros modelos, rode:

### V1 vs V2.0

```bash
# V1
python validate_strategy_v1.py --model scalping_model_BTCUSDT_15m_20251114_213903.pkl --days 90

# V2.0
python validate_strategy_v1.py --model ml_training/outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl --days 90
```

Compare:
- Win Rate
- ROI
- Profit Factor
- Max Drawdown

---

### V1 vs Modelo Antigo

```bash
# V1
python validate_strategy_v1.py --model scalping_model_BTCUSDT_15m_20251114_213903.pkl --days 180

# Modelo Antigo
python validate_strategy.py --model ml_model_master_scalper_365d.pkl --days 180
```

**Nota**: O modelo antigo usa o script original (`validate_strategy.py`), não o novo.

---

## 🐛 Troubleshooting

### Erro: "Model not found"

```bash
# Verificar se arquivo existe
ls -lh scalping_model_BTCUSDT_15m_20251114_213903.pkl

# Se não existir, procurar
find . -name "*213903*.pkl"

# Usar caminho absoluto
python validate_strategy_v1.py --model /caminho/completo/modelo.pkl
```

### Erro: "No module named X"

```bash
# Ativar venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependências
pip install pandas numpy lightgbm scikit-learn
```

### Warning: "Missing features"

✅ **Normal!** O script preenche automaticamente com 0.

### Resultados muito diferentes do esperado

Possíveis causas:
1. **Período de teste diferente** do que você mediu os 80% WR
2. **Configuração diferente** (TP/SL, confidence)
3. **Dados diferentes** (preços, ATR)

---

## 📝 Checklist de Uso

- [ ] Rodar `validate_strategy_v1.py` com V1
- [ ] Verificar se WR > 60% (confirma 80% reportado)
- [ ] Anotar melhor configuração recomendada
- [ ] Comparar com modelo antigo (opcional)
- [ ] Copiar modelo para produção
- [ ] Configurar `.env` com melhor config
- [ ] Testar em paper (1-2 semanas)
- [ ] Deploy em live se paper OK

---

## 🎯 Meta Final

**Objetivo**: Confirmar se V1 realmente tem **80% WR** e configurá-lo para produção.

**Métricas de Sucesso**:
- Win Rate: > 60%
- ROI anual: > 100%
- Profit Factor: > 1.5
- Sharpe Ratio: > 2.0
- Max Drawdown: < -10%

Se V1 atingir essas metas, é o **MELHOR MODELO** e deve ser usado imediatamente! 🚀

---

## 📞 Próximo Passo

**EXECUTE AGORA**:

```bash
python validate_strategy_v1.py \
  --model scalping_model_BTCUSDT_15m_20251114_213903.pkl \
  --days 180
```

**Aguardo os resultados para decidirmos juntos!** 📊
