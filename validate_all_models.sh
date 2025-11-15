#!/bin/bash
# Script para validar e comparar V1 vs V2.0 vs Antigo

echo "🔬 VALIDAÇÃO COMPLETA - V1 vs V2.0 vs Antigo"
echo "=============================================="
echo ""

# Modelos
V1="scalping_model_BTCUSDT_15m_20251114_213903.pkl"
V2="ml_training/outputs/scalping_model_BTCUSDT_15m_20251114_225401.pkl"
ANTIGO="ml_model_master_scalper_365d.pkl"

cd ml_training/validation

echo "📊 1. Validando V1 (80% WR reportado)..."
python validate_any_model.py \
  --model ../../$V1 \
  --days 90 \
  --save-csv results_v1.csv

echo ""
echo "📊 2. Validando V2.0 (Master V2.0)..."
python validate_any_model.py \
  --model ../$V2 \
  --days 90 \
  --save-csv results_v2.csv

echo ""
echo "📊 3. Validando modelo antigo..."
python validate_any_model.py \
  --model ../../$ANTIGO \
  --days 90 \
  --save-csv results_antigo.csv

echo ""
echo "⚔️  4. Comparando V1 vs V2.0..."
python compare_models.py \
  --model1 ../../$V1 \
  --model2 ../$V2 \
  --days 90 \
  --save-csv comparison_v1_vs_v2.csv

echo ""
echo "⚔️  5. Comparando V1 vs Antigo..."
python compare_models.py \
  --model1 ../../$V1 \
  --model2 ../../$ANTIGO \
  --days 90 \
  --save-csv comparison_v1_vs_antigo.csv

echo ""
echo "🔍 6. Analisando predições do V1..."
python analyze_predictions.py \
  --model ../../$V1 \
  --days 90

echo ""
echo "✅ VALIDAÇÃO COMPLETA!"
echo ""
echo "📁 Resultados salvos em:"
echo "   - results_v1.csv"
echo "   - results_v2.csv"
echo "   - results_antigo.csv"
echo "   - comparison_v1_vs_v2.csv"
echo "   - comparison_v1_vs_antigo.csv"
