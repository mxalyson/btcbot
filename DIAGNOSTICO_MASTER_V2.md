# 🔍 DIAGNÓSTICO - Master V2.0 com Resultados Ruins

## ❌ Resultados do Backtest (90 dias)

```
Win Rate:         49.3%  ❌ (precisa > 52%)
Total Return:     -27.97% ❌ (perdeu quase 28%!)
Profit Factor:    0.77    ❌ (precisa > 1.0)
Total Trades:     515
Avg Win:          $37.81
Avg Loss:         $47.51  ⚠️ (maior que avg win!)
Total Fees:       $5,325  ⚠️ (53% do capital inicial!)

LONG trades:      299 (WR: 46.5%) ❌ muito ruim!
SHORT trades:     216 (WR: 53.2%) ✅ melhor

Max Drawdown:     -31.18%
Sharpe Ratio:     1.31
```

---

## 🔴 Problemas Identificados

### 1. Viés para LONGs (Desbalanceado)

Durante o backtest:
- **5,033 sinais LONG** (58.8%)
- **3,525 sinais SHORT** (41.2%)

Mas LONGs têm **WR 46.5%** (muito ruim!), enquanto SHORTs têm **WR 53.2%** (melhor).

**⚠️ Isso sugere**:
- Modelo está enviesado para prever UP
- Mas mercado teve mais movimentos DOWN no período testado
- Target de treino pode estar desbalanceado ou overfitted

---

### 2. Avg Loss > Avg Win (R:R Ruim)

- **Avg Win**: $37.81
- **Avg Loss**: $47.51
- **R:R Real**: 0.80 (perde mais que ganha)

**Configuração teórica**:
- TP: 2.0 ATR
- SL: 1.5 ATR
- **R:R Teórico**: 1.33

**⚠️ Por que não bate?**
- Modelo não está entrando nos melhores pontos
- Stop Loss sendo atingido muito frequentemente
- Take Profit não sendo atingido (slippage, volatilidade)

---

### 3. Fees Muito Altas

- **Total Fees**: $5,325 (53% do capital inicial!)
- **Fees por trade**: ~$10.34
- **515 trades** em 90 dias = 5.7 trades/dia

**⚠️ Problema**:
- Trading muito frequente
- Fees consomem os lucros
- Modelo pode estar gerando sinais de baixa qualidade

---

### 4. Win Rate Abaixo de 50%

Com R:R teórico de 1.33, precisamos de WR mínimo:
```
Break-even WR = 1 / (1 + R:R) = 1 / (1 + 1.33) = 42.9%
```

Mas para ser **lucrativo** com fees, precisamos de WR > 52%.

**Master V2.0**: 49.3% WR ❌

---

## 🧪 INVESTIGAÇÃO - Próximos Passos

### 1. Comparar com Modelo Antigo

**Objetivo**: Verificar se o problema é o modelo ou o período de teste.

```bash
cd ml_training/validation

# Master V2.0
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.50 --tp 2.0 --sl 1.5

# Modelo Antigo
python backtest_ml_model.py \
  --model ../../ml_model_master_scalper_365d.pkl \
  --days 90 --confidence 0.50 --tp 2.0 --sl 1.5
```

**Se modelo antigo TAMBÉM for ruim** → Problema é o **período de teste** (market regime)

**Se modelo antigo for BOM** → Problema é o **Master V2.0** (overfitting, target, features)

---

### 2. Analisar Predições do Modelo

Use o script de análise para investigar viés:

```bash
cd ml_training/validation

# Analisar Master V2.0
python analyze_predictions.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90

# Analisar modelo antigo (comparação)
python analyze_predictions.py \
  --model ../../ml_model_master_scalper_365d.pkl \
  --days 90
```

**O que procurar**:
- ✅ Distribuição de classes: deve ser ~50/50
- ✅ Confidence por threshold: quantos sinais com >60% confidence?
- ✅ Análise temporal: viés muda ao longo do tempo?
- ✅ Correlação com volatilidade: modelo funciona melhor em alta/baixa volatilidade?

---

### 3. Testar Diferentes Parâmetros

#### 3.1 Aumentar Confidence Threshold

Menos trades, mas mais qualidade:

```bash
# Confidence 60%
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.60 --tp 2.0 --sl 1.5

# Confidence 65%
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.65 --tp 2.0 --sl 1.5

# Confidence 70%
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.70 --tp 2.0 --sl 1.5
```

**Objetivo**: Reduzir número de trades e aumentar WR.

---

#### 3.2 Ajustar TP/SL para Melhor R:R

```bash
# TP maior, SL menor (R:R = 2.5)
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.50 --tp 2.5 --sl 1.0

# TP muito maior (R:R = 3.0)
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.50 --tp 3.0 --sl 1.0

# TP/SL conservador (R:R = 2.0)
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.50 --tp 2.0 --sl 1.0
```

**Objetivo**: Melhorar R:R real (Avg Win / Avg Loss).

---

#### 3.3 Combinar: Alta Confidence + Melhor R:R

```bash
# Confidence 65% + TP 2.5 / SL 1.0
python backtest_ml_model.py \
  --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
  --days 90 --confidence 0.65 --tp 2.5 --sl 1.0
```

**Objetivo**: Menos trades, mas de alta qualidade.

---

### 4. Retreinar com Mais Dados

Se modelo está overfitted:

```bash
cd ml_training

# Treinar com 360 dias (1 ano)
python train_master_v2.py --symbol BTCUSDT --days 360

# Treinar com 540 dias (1.5 anos)
python train_master_v2.py --symbol BTCUSDT --days 540
```

**Objetivo**: Mais dados = menos overfitting.

---

## 🎯 Hipóteses do Problema

### Hipótese 1: Overfitting

**Evidência**:
- AUC 0.71 no validation set (bom)
- WR 49.3% no backtest (ruim)

**Causa**:
- Modelo aprendeu padrões específicos do período de treino
- Não generaliza para dados novos

**Solução**:
- Treinar com mais dados (360+ dias)
- Aumentar regularização (L1/L2)
- Reduzir complexity do modelo

---

### Hipótese 2: Target Mismatch (ainda)

**Evidência**:
- Target de treino: votação multi-horizon (4, 6, 8 bars) com threshold ATR
- Backtest: TP 2.0 ATR / SL 1.5 ATR

**Possível problema**:
- Horizons de treino (1h, 1.5h, 2h) não batem com TP/SL real
- Dynamic threshold (0.35-0.75%) pode não bater com 2.0 ATR

**Solução**:
- Verificar se ATR% no treino = ATR no backtest
- Ajustar horizons para bater com TP/SL médio

---

### Hipótese 3: Período de Teste Ruim

**Evidência**:
- Modelo gera muito mais LONGs (58.8%)
- Mas LONGs têm WR ruim (46.5%)
- SHORTs têm WR melhor (53.2%)

**Possível problema**:
- Período de teste teve mercado bearish (mais quedas)
- Modelo foi treinado em período bullish (mais subidas)

**Solução**:
- Testar em diferentes períodos (30 dias, 60 dias, 180 dias)
- Comparar com modelo antigo no mesmo período
- Treinar com dados mais recentes

---

### Hipótese 4: Features Irrelevantes

**Evidência**:
- 150+ features (advanced)
- Modelo antigo tinha só 65 features

**Possível problema**:
- Muitas features = mais ruído
- Features irrelevantes causam overfitting

**Solução**:
- Feature importance: remover features com <0.5% importance
- Treinar com só top 50-80 features
- Usar feature selection (SelectKBest, RFE)

---

## 📊 Matriz de Decisão

| Cenário | Win Rate | ROI | Ação |
|---------|----------|-----|------|
| Modelo antigo BOM + V2.0 RUIM | >52% vs <50% | >0% vs <0% | ❌ Descartar V2.0, usar antigo |
| Ambos RUINS | <50% | <0% | ⚠️ Período de teste ruim, testar outros períodos |
| V2.0 melhora com confidence>60% | >52% | >0% | ✅ Usar V2.0 com threshold alto |
| V2.0 melhora com TP/SL ajustado | >52% | >0% | ✅ Usar V2.0 com novos parâmetros |
| Nada funciona | <50% | <0% | 🔄 Retreinar com mais dados ou revisar target |

---

## 🚀 Plano de Ação

### ✅ Etapa 1: Comparação (URGENTE)

```bash
# 1. Testar modelo antigo
python backtest_ml_model.py \
  --model ../../ml_model_master_scalper_365d.pkl \
  --days 90 --confidence 0.50 --tp 2.0 --sl 1.5

# 2. Analisar predições de ambos
python analyze_predictions.py --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl --days 90
python analyze_predictions.py --model ../../ml_model_master_scalper_365d.pkl --days 90

# 3. Comparar WR, ROI, Profit Factor
```

---

### ✅ Etapa 2: Otimização de Parâmetros

Se V2.0 tiver potencial:

```bash
# Grid search de parâmetros
for conf in 0.55 0.60 0.65 0.70; do
  for tp in 2.0 2.5 3.0; do
    for sl in 1.0 1.5; do
      echo "Testing conf=$conf tp=$tp sl=$sl"
      python backtest_ml_model.py \
        --model ../outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl \
        --days 90 --confidence $conf --tp $tp --sl $sl | grep "Win Rate\|Total Return\|Profit Factor"
    done
  done
done
```

---

### ✅ Etapa 3: Retreinar (se necessário)

Se nada funcionar:

```bash
# Retreinar com mais dados
python train_master_v2.py --symbol BTCUSDT --days 360

# Retreinar com regularização maior
# (editar train_master_v2.py: reg_alpha=0.3, reg_lambda=0.3)
```

---

## 📝 Checklist

**Investigação**:
- [ ] Rodar backtest do modelo antigo (90 dias)
- [ ] Comparar WR, ROI, PF de ambos modelos
- [ ] Analisar predições de ambos modelos
- [ ] Identificar viés (LONG/SHORT) em ambos

**Otimização**:
- [ ] Testar confidence 60%, 65%, 70%
- [ ] Testar TP/SL: 2.5/1.0, 3.0/1.0
- [ ] Encontrar melhor combinação de parâmetros

**Decisão**:
- [ ] Se V2.0 melhor → usar V2.0
- [ ] Se antigo melhor → voltar para antigo
- [ ] Se ambos ruins → investigar período ou retreinar

---

## ❓ FAQ

### Por que o modelo tem AUC 0.71 mas WR 49.3%?

**AUC mede capacidade de ranquear** (prever qual é maior), não acurácia binária.

- **AUC 0.71**: Modelo consegue separar UP de DOWN razoavelmente bem
- **WR 49.3%**: Mas no backtest real, erra mais que acerta

**Possível causa**: Overfitting, target mismatch, ou período de teste diferente.

---

### Por que LONGs têm WR pior que SHORTs?

**Modelo está enviesado para UP**:
- Treinou em período com mais subidas
- Ou target foi calculado em mercado bullish

**Mas período de teste teve mais quedas**:
- SHORTs funcionam melhor (WR 53.2%)
- LONGs funcionam pior (WR 46.5%)

**Solução**:
- Retreinar com dados mais recentes (bearish + bullish)
- Ou usar apenas SHORTs do modelo

---

### Devo descartar o Master V2.0?

**NÃO! Ainda não.**

Primeiro:
1. Compare com modelo antigo no mesmo período
2. Teste diferentes parâmetros (confidence, TP/SL)
3. Analise predições para entender viés

Se após tudo isso WR < 50%, **aí sim descarte**.

---

### Qual é a meta mínima?

Para ser **lucrativo** com fees:
- **Win Rate**: > 52% (idealmente 55%+)
- **Profit Factor**: > 1.2
- **ROI**: > 20% anual (90 dias → 5%+)

Master V2.0 atual:
- WR: 49.3% ❌
- PF: 0.77 ❌
- ROI: -27.97% ❌

**Muito longe da meta!**

---

## 🏁 Conclusão Preliminar

Master V2.0 **NÃO está pronto para produção** com parâmetros atuais.

**Próximos passos obrigatórios**:
1. ✅ Comparar com modelo antigo
2. ✅ Analisar predições
3. ✅ Otimizar parâmetros
4. ❌ **SE NADA FUNCIONAR**: Retreinar ou descartar

**Não desista ainda!** Pode ser só questão de ajustar parâmetros. 💪
