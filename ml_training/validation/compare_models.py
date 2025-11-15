#!/usr/bin/env python3
"""
COMPARADOR DE MODELOS ML
========================
Compara dois modelos .pkl lado a lado com as mesmas configurações.
Útil para comparar V1 vs V2, ou novo vs antigo.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import logging
import pandas as pd

from validate_any_model import (
    UniversalModelValidator,
    load_and_prepare_data,
    logger
)

def compare_two_models(
    model1_path: str,
    model2_path: str,
    df: pd.DataFrame,
    confidence_levels: list,
    tp_sl_configs: list
) -> pd.DataFrame:
    """Compare two models with same configurations."""

    logger.info(f"\n{'='*80}")
    logger.info(f"⚔️  MODEL COMPARISON")
    logger.info(f"{'='*80}\n")

    # Load both models
    logger.info(f"📦 Loading Model 1: {Path(model1_path).name}")
    validator1 = UniversalModelValidator(model1_path)

    logger.info(f"📦 Loading Model 2: {Path(model2_path).name}")
    validator2 = UniversalModelValidator(model2_path)

    logger.info("")

    # Validate both models
    logger.info("🧪 Testing Model 1...")
    results1 = validator1.validate(df.copy(), confidence_levels, tp_sl_configs)
    results1['model'] = Path(model1_path).stem

    logger.info("🧪 Testing Model 2...")
    results2 = validator2.validate(df.copy(), confidence_levels, tp_sl_configs)
    results2['model'] = Path(model2_path).stem

    # Combine results
    df_combined = pd.concat([results1, results2], ignore_index=True)

    return df_combined


def print_comparison_table(df_combined: pd.DataFrame, model1_name: str, model2_name: str):
    """Print side-by-side comparison table."""

    logger.info(f"\n{'='*120}")
    logger.info(f"📊 SIDE-BY-SIDE COMPARISON")
    logger.info(f"{'='*120}\n")

    # Get unique configurations
    configs = df_combined[['tp_atr_mult', 'sl_atr_mult', 'min_confidence']].drop_duplicates()

    # Print header
    logger.info(f"{'Config':<20} | {'Model 1 (' + model1_name[:15] + ')':<40} | {'Model 2 (' + model2_name[:15] + ')':<40} | {'Winner':<10}")
    logger.info("-" * 120)

    for _, config in configs.iterrows():
        tp = config['tp_atr_mult']
        sl = config['sl_atr_mult']
        conf = config['min_confidence']

        # Get results for each model
        mask = (df_combined['tp_atr_mult'] == tp) & \
               (df_combined['sl_atr_mult'] == sl) & \
               (df_combined['min_confidence'] == conf)

        results = df_combined[mask]

        if len(results) == 2:
            r1 = results.iloc[0] if results.iloc[0]['model'] == model1_name else results.iloc[1]
            r2 = results.iloc[1] if results.iloc[1]['model'] == model2_name else results.iloc[0]

            # Determine winner (by ROI)
            if r1['total_trades'] == 0 and r2['total_trades'] == 0:
                winner = "NONE"
            elif r1['total_trades'] == 0:
                winner = "Model 2"
            elif r2['total_trades'] == 0:
                winner = "Model 1"
            elif r1['roi'] > r2['roi']:
                winner = "Model 1 ✅"
            elif r2['roi'] > r1['roi']:
                winner = "Model 2 ✅"
            else:
                winner = "TIE"

            config_str = f"TP{tp:.1f} SL{sl:.1f} C{conf:.0%}"

            m1_str = f"ROI:{r1['roi']:+6.1f}% WR:{r1['win_rate']:5.1f}% T:{r1['total_trades']:3.0f}"
            m2_str = f"ROI:{r2['roi']:+6.1f}% WR:{r2['win_rate']:5.1f}% T:{r2['total_trades']:3.0f}"

            logger.info(f"{config_str:<20} | {m1_str:<40} | {m2_str:<40} | {winner:<10}")

    logger.info("")


def print_overall_winner(df_combined: pd.DataFrame, model1_name: str, model2_name: str):
    """Print overall winner analysis."""

    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 OVERALL WINNER")
    logger.info(f"{'='*80}\n")

    # Filter configs with at least 20 trades
    df_model1 = df_combined[df_combined['model'] == model1_name].copy()
    df_model2 = df_combined[df_combined['model'] == model2_name].copy()

    df_model1_valid = df_model1[df_model1['total_trades'] >= 20]
    df_model2_valid = df_model2[df_model2['total_trades'] >= 20]

    if len(df_model1_valid) == 0 and len(df_model2_valid) == 0:
        logger.warning("⚠️  No configurations with >= 20 trades for either model")
        return

    # Best configuration for each model
    if len(df_model1_valid) > 0:
        best1 = df_model1_valid.loc[df_model1_valid['roi'].idxmax()]
        logger.info(f"📊 Model 1 Best: {model1_name}")
        logger.info(f"   Config: TP{best1['tp_atr_mult']:.1f} SL{best1['sl_atr_mult']:.1f} Conf{best1['min_confidence']:.0%}")
        logger.info(f"   ROI: {best1['roi']:+.2f}% | WR: {best1['win_rate']:.1f}% | PF: {best1['profit_factor']:.2f}")
        logger.info(f"   Sharpe: {best1['sharpe_ratio']:.2f} | DD: {best1['max_drawdown']:.2f}%")
        logger.info("")
    else:
        logger.info(f"❌ Model 1: No valid configurations")
        logger.info("")
        best1 = None

    if len(df_model2_valid) > 0:
        best2 = df_model2_valid.loc[df_model2_valid['roi'].idxmax()]
        logger.info(f"📊 Model 2 Best: {model2_name}")
        logger.info(f"   Config: TP{best2['tp_atr_mult']:.1f} SL{best2['sl_atr_mult']:.1f} Conf{best2['min_confidence']:.0%}")
        logger.info(f"   ROI: {best2['roi']:+.2f}% | WR: {best2['win_rate']:.1f}% | PF: {best2['profit_factor']:.2f}")
        logger.info(f"   Sharpe: {best2['sharpe_ratio']:.2f} | DD: {best2['max_drawdown']:.2f}%")
        logger.info("")
    else:
        logger.info(f"❌ Model 2: No valid configurations")
        logger.info("")
        best2 = None

    # Determine winner
    if best1 is None and best2 is None:
        logger.info("❌ No winner - both models failed")
    elif best1 is None:
        logger.info(f"🏆 WINNER: Model 2 ({model2_name})")
        logger.info(f"   Model 1 had no valid configurations")
    elif best2 is None:
        logger.info(f"🏆 WINNER: Model 1 ({model1_name})")
        logger.info(f"   Model 2 had no valid configurations")
    else:
        # Compare by ROI
        if best1['roi'] > best2['roi']:
            diff = best1['roi'] - best2['roi']
            logger.info(f"🏆 WINNER: Model 1 ({model1_name})")
            logger.info(f"   Better ROI by {diff:+.2f}%")
        elif best2['roi'] > best1['roi']:
            diff = best2['roi'] - best1['roi']
            logger.info(f"🏆 WINNER: Model 2 ({model2_name})")
            logger.info(f"   Better ROI by {diff:+.2f}%")
        else:
            logger.info(f"🤝 TIE: Both models have same ROI")

        logger.info("")

        # Additional comparison metrics
        logger.info(f"📈 Metrics Comparison:")
        logger.info(f"   Win Rate: {best1['win_rate']:.1f}% vs {best2['win_rate']:.1f}%")
        logger.info(f"   Profit Factor: {best1['profit_factor']:.2f} vs {best2['profit_factor']:.2f}")
        logger.info(f"   Sharpe Ratio: {best1['sharpe_ratio']:.2f} vs {best2['sharpe_ratio']:.2f}")
        logger.info(f"   Max DD: {best1['max_drawdown']:.2f}% vs {best2['max_drawdown']:.2f}%")
        logger.info(f"   Total Trades: {best1['total_trades']:.0f} vs {best2['total_trades']:.0f}")

    logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Compare two ML models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model .pkl')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model .pkl')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--days', type=int, default=90, help='Number of days to test')
    parser.add_argument('--save-csv', type=str, help='Save results to CSV file')

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"⚔️  ML MODEL COMPARISON")
    logger.info(f"{'='*80}\n")
    logger.info(f"Model 1: {args.model1}")
    logger.info(f"Model 2: {args.model2}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.days} days")
    logger.info("")

    try:
        # Load data
        df = load_and_prepare_data(args.symbol, args.timeframe, args.days)

        # Define test configurations (smaller set for comparison)
        confidence_levels = [0.0, 0.50, 0.60, 0.65]
        tp_sl_configs = [
            (2.0, 1.5),  # Original
            (2.5, 1.0),  # Higher R:R
            (3.0, 1.0),  # Very high R:R
        ]

        # Compare models
        df_combined = compare_two_models(
            args.model1,
            args.model2,
            df,
            confidence_levels,
            tp_sl_configs
        )

        # Get model names
        model1_name = Path(args.model1).stem
        model2_name = Path(args.model2).stem

        # Print comparison
        print_comparison_table(df_combined, model1_name, model2_name)
        print_overall_winner(df_combined, model1_name, model2_name)

        # Save to CSV if requested
        if args.save_csv:
            df_combined.to_csv(args.save_csv, index=False)
            logger.info(f"\n💾 Results saved to: {args.save_csv}")

        logger.info(f"{'='*80}\n")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
