#!/usr/bin/env python3
"""
VALIDADOR UNIVERSAL DE MODELOS ML
==================================
Valida qualquer modelo .pkl (V1, V2, antigo) com diferentes parâmetros.
Testa combinações de confidence threshold e TP/SL para encontrar a melhor configuração.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse
from datetime import datetime
import logging

from core.data_manager import DataManager
from ml_training.features.feature_engineering import FeatureEngineer
from ml_training.features.advanced_features import ScalpingFeatureEngineer, create_legacy_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ModelValidator')


class UniversalModelValidator:
    """Valida qualquer modelo ML com diferentes configurações."""

    def __init__(self, model_path: str, initial_capital: float = 10000.0):
        self.model_path = Path(model_path)
        self.initial_capital = initial_capital

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load model
        logger.info(f"📦 Loading model: {self.model_path.name}")
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.metadata = self.model_data.get('metadata', {})

        # Detect model type
        self._detect_model_type()

        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Type: {self.model_type}")
        logger.info(f"   Features: {len(self.feature_names)}")
        logger.info(f"   Target: {self.metadata.get('target_type', 'unknown')}")
        logger.info(f"   Horizon: {self.metadata.get('horizon', 'unknown')} bars")

    def _detect_model_type(self):
        """Detect if model is binary or multiclass."""
        try:
            # Try to get number of classes from model
            if hasattr(self.model, 'n_classes_'):
                n_classes = self.model.n_classes_
            elif hasattr(self.model, '_n_classes'):
                n_classes = self.model._n_classes
            else:
                # Try to infer from a dummy prediction
                dummy = np.zeros((1, len(self.feature_names)))
                proba = self.model.predict_proba(dummy)
                n_classes = proba.shape[1]

            self.n_classes = n_classes
            self.model_type = 'BINARY' if n_classes == 2 else 'MULTICLASS'
        except:
            # Default to binary
            self.n_classes = 2
            self.model_type = 'BINARY'

    def validate(
        self,
        df: pd.DataFrame,
        confidence_levels: List[float],
        tp_sl_configs: List[Tuple[float, float]],
        fees_pct: float = 0.06,
        slippage_pct: float = 0.01
    ) -> pd.DataFrame:
        """
        Validate model with different configurations.

        Args:
            df: DataFrame with OHLCV + features
            confidence_levels: List of min confidence thresholds to test (e.g., [0.0, 0.5, 0.6])
            tp_sl_configs: List of (tp_atr_mult, sl_atr_mult) tuples (e.g., [(2.0, 1.5), (2.5, 1.0)])
            fees_pct: Trading fees in % (default 0.06%)
            slippage_pct: Slippage in % (default 0.01%)

        Returns:
            DataFrame with results for all configurations
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 VALIDATION TEST MATRIX")
        logger.info(f"{'='*80}")
        logger.info(f"Confidence levels: {len(confidence_levels)}")
        logger.info(f"TP/SL configs: {len(tp_sl_configs)}")
        logger.info(f"Total tests: {len(confidence_levels) * len(tp_sl_configs)}")
        logger.info(f"Fees: {fees_pct}% | Slippage: {slippage_pct}%")
        logger.info("")

        results = []

        for tp_mult, sl_mult in tp_sl_configs:
            for min_conf in confidence_levels:
                logger.info(f"Testing: TP={tp_mult}x SL={sl_mult}x Conf>={min_conf:.0%}...")

                stats = self._backtest(
                    df.copy(),
                    min_confidence=min_conf,
                    tp_atr_mult=tp_mult,
                    sl_atr_mult=sl_mult,
                    fees_pct=fees_pct,
                    slippage_pct=slippage_pct
                )

                stats['tp_atr_mult'] = tp_mult
                stats['sl_atr_mult'] = sl_mult
                stats['min_confidence'] = min_conf
                stats['fees_pct'] = fees_pct
                results.append(stats)

        df_results = pd.DataFrame(results)
        return df_results

    def _backtest(
        self,
        df: pd.DataFrame,
        min_confidence: float,
        tp_atr_mult: float,
        sl_atr_mult: float,
        fees_pct: float,
        slippage_pct: float
    ) -> Dict:
        """Run backtest with specific configuration."""

        # Prepare features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            for feat in missing_features:
                df[feat] = 0

        X = df[self.feature_names].fillna(0)

        # Get predictions
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)

        # Process predictions based on model type
        if self.model_type == 'BINARY':
            # Binary: 0=DOWN, 1=UP
            df['ml_pred'] = predictions
            df['ml_prob_short'] = proba[:, 0]
            df['ml_prob_long'] = proba[:, 1]

            # Confidence = probability of predicted class
            df['ml_confidence'] = np.where(
                predictions == 1,
                proba[:, 1],  # UP
                proba[:, 0]   # DOWN
            )

            # Signal: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
            df['signal'] = 0
            df.loc[(predictions == 1) & (df['ml_confidence'] >= min_confidence), 'signal'] = 1   # LONG
            df.loc[(predictions == 0) & (df['ml_confidence'] >= min_confidence), 'signal'] = -1  # SHORT

        else:
            # Multiclass: 0=NEUTRAL, 1=DOWN, 2=UP
            df['ml_pred'] = predictions
            df['ml_prob_short'] = proba[:, 1]
            df['ml_prob_neutral'] = proba[:, 0]
            df['ml_prob_long'] = proba[:, 2]
            df['ml_confidence'] = proba.max(axis=1)

            # Signal
            df['signal'] = 0
            df.loc[(predictions == 2) & (df['ml_confidence'] >= min_confidence), 'signal'] = 1   # LONG
            df.loc[(predictions == 1) & (df['ml_confidence'] >= min_confidence), 'signal'] = -1  # SHORT

        # Simulate trades
        trades = self._simulate_trades(df, tp_atr_mult, sl_atr_mult, fees_pct, slippage_pct)

        # Calculate statistics
        stats = self._calculate_stats(trades, df)

        return stats

    def _simulate_trades(
        self,
        df: pd.DataFrame,
        tp_atr_mult: float,
        sl_atr_mult: float,
        fees_pct: float,
        slippage_pct: float
    ) -> List[Dict]:
        """Simulate trades."""

        trades = []
        position = None
        capital = self.initial_capital

        for i in range(len(df) - 1):  # -1 to avoid last bar
            current = df.iloc[i]

            # Check exit
            if position is not None:
                exit_info = self._check_exit(position, df, i)
                if exit_info:
                    trade = self._close_trade(position, exit_info, fees_pct, slippage_pct)
                    trades.append(trade)
                    capital = trade['final_capital']
                    position = None

            # Check entry
            if position is None and current['signal'] != 0:
                position = self._open_trade(current, capital, tp_atr_mult, sl_atr_mult, i)

        # Close final position if open
        if position is not None:
            final_bar = df.iloc[-1]
            exit_info = {
                'price': final_bar['close'],
                'reason': 'end_of_data',
                'idx': len(df) - 1,
                'timestamp': final_bar.name
            }
            trade = self._close_trade(position, exit_info, fees_pct, slippage_pct)
            trades.append(trade)

        return trades

    def _open_trade(
        self,
        current,
        capital: float,
        tp_atr_mult: float,
        sl_atr_mult: float,
        idx: int
    ) -> Dict:
        """Open a trade."""

        direction = 'LONG' if current['signal'] == 1 else 'SHORT'
        entry_price = current['close']
        atr = current.get('atr', entry_price * 0.005)  # Default 0.5% if no ATR

        if direction == 'LONG':
            stop_loss = entry_price - (atr * sl_atr_mult)
            take_profit = entry_price + (atr * tp_atr_mult)
        else:
            stop_loss = entry_price + (atr * sl_atr_mult)
            take_profit = entry_price - (atr * tp_atr_mult)

        return {
            'direction': direction,
            'entry_idx': idx,
            'entry_timestamp': current.name,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr,
            'ml_confidence': current['ml_confidence'],
            'capital': capital
        }

    def _check_exit(self, position: Dict, df: pd.DataFrame, current_idx: int) -> Dict:
        """Check if position should be exited."""

        current = df.iloc[current_idx]
        high = current['high']
        low = current['low']
        close = current['close']

        direction = position['direction']

        # Check stop loss and take profit
        if direction == 'LONG':
            if low <= position['stop_loss']:
                return {
                    'price': position['stop_loss'],
                    'reason': 'stop_loss',
                    'idx': current_idx,
                    'timestamp': current.name
                }
            elif high >= position['take_profit']:
                return {
                    'price': position['take_profit'],
                    'reason': 'take_profit',
                    'idx': current_idx,
                    'timestamp': current.name
                }
        else:  # SHORT
            if high >= position['stop_loss']:
                return {
                    'price': position['stop_loss'],
                    'reason': 'stop_loss',
                    'idx': current_idx,
                    'timestamp': current.name
                }
            elif low <= position['take_profit']:
                return {
                    'price': position['take_profit'],
                    'reason': 'take_profit',
                    'idx': current_idx,
                    'timestamp': current.name
                }

        # Time-based exit (48 hours = 192 bars of 15min)
        duration = current_idx - position['entry_idx']
        if duration >= 192:
            return {
                'price': close,
                'reason': 'time_exit',
                'idx': current_idx,
                'timestamp': current.name
            }

        return None

    def _close_trade(
        self,
        position: Dict,
        exit_info: Dict,
        fees_pct: float,
        slippage_pct: float
    ) -> Dict:
        """Close a trade and calculate PnL."""

        entry_price = position['entry_price']
        exit_price = exit_info['price']
        direction = position['direction']

        # Calculate PnL %
        if direction == 'LONG':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        # Apply fees and slippage
        total_fees = fees_pct * 2  # Entry + exit
        total_slippage = slippage_pct * 2
        pnl_pct_net = pnl_pct - total_fees - total_slippage

        # Calculate PnL amount (assume full capital on each trade)
        pnl_amount = position['capital'] * (pnl_pct_net / 100)
        final_capital = position['capital'] + pnl_amount

        # Duration
        duration = exit_info['idx'] - position['entry_idx']

        return {
            'direction': direction,
            'entry_timestamp': position['entry_timestamp'],
            'exit_timestamp': exit_info['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit'],
            'pnl_pct_gross': pnl_pct,
            'pnl_pct_net': pnl_pct_net,
            'pnl_amount': pnl_amount,
            'capital': position['capital'],
            'final_capital': final_capital,
            'exit_reason': exit_info['reason'],
            'duration_bars': duration,
            'ml_confidence': position['ml_confidence'],
            'atr': position['atr']
        }

    def _calculate_stats(self, trades: List[Dict], df: pd.DataFrame) -> Dict:
        """Calculate statistics from trades."""

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'roi': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'final_capital': self.initial_capital
            }

        df_trades = pd.DataFrame(trades)

        # Basic stats
        total_trades = len(df_trades)
        winners = df_trades[df_trades['pnl_amount'] > 0]
        losers = df_trades[df_trades['pnl_amount'] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        # PnL
        total_pnl = df_trades['pnl_amount'].sum()
        final_capital = self.initial_capital + total_pnl
        roi = (total_pnl / self.initial_capital) * 100

        # Avg win/loss
        avg_win = winners['pnl_amount'].mean() if len(winners) > 0 else 0
        avg_loss = abs(losers['pnl_amount'].mean()) if len(losers) > 0 else 0

        # Profit factor
        total_wins = winners['pnl_amount'].sum() if len(winners) > 0 else 0
        total_losses = abs(losers['pnl_amount'].sum()) if len(losers) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe ratio
        if len(df_trades) > 1:
            returns = df_trades['pnl_pct_net'].values
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Drawdown
        equity_curve = self.initial_capital + df_trades['pnl_amount'].cumsum()
        peak = equity_curve.expanding().max()
        drawdown = ((equity_curve - peak) / peak * 100)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Direction stats
        longs = df_trades[df_trades['direction'] == 'LONG']
        shorts = df_trades[df_trades['direction'] == 'SHORT']

        long_wr = (longs['pnl_amount'] > 0).mean() * 100 if len(longs) > 0 else 0
        short_wr = (shorts['pnl_amount'] > 0).mean() * 100 if len(shorts) > 0 else 0

        # Exit reasons
        exit_reasons = df_trades['exit_reason'].value_counts().to_dict()

        # Duration
        avg_duration = df_trades['duration_bars'].mean()

        # Confidence
        avg_confidence = df_trades['ml_confidence'].mean()

        return {
            'total_trades': total_trades,
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': win_rate * 100,  # Convert to %
            'total_pnl': total_pnl,
            'roi': roi,
            'final_capital': final_capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_wr': long_wr,
            'short_wr': short_wr,
            'avg_duration_bars': avg_duration,
            'avg_confidence': avg_confidence * 100,  # Convert to %
            'exit_stop_loss': exit_reasons.get('stop_loss', 0),
            'exit_take_profit': exit_reasons.get('take_profit', 0),
            'exit_time': exit_reasons.get('time_exit', 0),
        }


def load_and_prepare_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Load data and build all features."""

    logger.info(f"📥 Loading data: {symbol} {timeframe} ({days} days)")

    # Load data
    data_manager = DataManager()
    df = data_manager.fetch_historical_data(
        symbol=symbol,
        timeframe=timeframe,
        days=days
    )

    if df.empty:
        raise ValueError("No data loaded")

    logger.info(f"✅ Loaded {len(df):,} candles")

    # Build features
    logger.info(f"🔨 Building features...")

    # Base features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.build_features(df)
    logger.info(f"   ✅ Base features: {len(df_features.columns)} columns")

    # Legacy features
    df_features = create_legacy_features(df_features)
    logger.info(f"   ✅ Legacy features added")

    # Advanced scalping features
    scalping_engineer = ScalpingFeatureEngineer()
    df_features = scalping_engineer.build_all_features(df_features)
    logger.info(f"   ✅ Advanced features: {len(df_features.columns)} columns")

    logger.info(f"✅ Total features: {len(df_features.columns)}")

    return df_features


def print_results_table(df_results: pd.DataFrame):
    """Print results in a nice table format."""

    logger.info(f"\n{'='*120}")
    logger.info(f"📊 VALIDATION RESULTS")
    logger.info(f"{'='*120}\n")

    # Sort by ROI descending
    df_sorted = df_results.sort_values('roi', ascending=False)

    # Print header
    header = (f"{'TP':<4} {'SL':<4} {'Conf':<5} | "
              f"{'Trades':>6} {'WR':>6} | "
              f"{'ROI':>8} {'PF':>5} {'Sharpe':>6} {'DD':>7} | "
              f"{'LONG':>5} {'SHORT':>5} | "
              f"{'LongWR':>7} {'ShortWR':>7}")

    logger.info(header)
    logger.info("-" * 120)

    # Print rows
    for _, row in df_sorted.iterrows():
        if row['total_trades'] > 0:
            line = (f"{row['tp_atr_mult']:<4.1f} "
                   f"{row['sl_atr_mult']:<4.1f} "
                   f"{row['min_confidence']*100:<5.0f} | "
                   f"{row['total_trades']:>6.0f} "
                   f"{row['win_rate']:>5.1f}% | "
                   f"{row['roi']:>+7.1f}% "
                   f"{row['profit_factor']:>5.2f} "
                   f"{row['sharpe_ratio']:>6.2f} "
                   f"{row['max_drawdown']:>6.1f}% | "
                   f"{row['long_trades']:>5.0f} "
                   f"{row['short_trades']:>5.0f} | "
                   f"{row['long_wr']:>6.1f}% "
                   f"{row['short_wr']:>6.1f}%")
            logger.info(line)

    logger.info("")


def analyze_best_config(df_results: pd.DataFrame):
    """Analyze and recommend best configuration."""

    # Filter valid results (at least 20 trades)
    df_valid = df_results[df_results['total_trades'] >= 20].copy()

    if len(df_valid) == 0:
        logger.warning("⚠️  No configurations with >= 20 trades")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 BEST CONFIGURATIONS")
    logger.info(f"{'='*80}\n")

    # Best by different metrics
    best_roi = df_valid.loc[df_valid['roi'].idxmax()]
    best_sharpe = df_valid.loc[df_valid['sharpe_ratio'].idxmax()]
    best_wr = df_valid.loc[df_valid['win_rate'].idxmax()]
    best_pf = df_valid.loc[df_valid['profit_factor'].idxmax()]

    def print_config(name: str, config: pd.Series):
        logger.info(f"🎯 {name}:")
        logger.info(f"   TP: {config['tp_atr_mult']:.1f}x ATR | SL: {config['sl_atr_mult']:.1f}x ATR | Min Conf: {config['min_confidence']:.0%}")
        logger.info(f"   ROI: {config['roi']:+.2f}% | WR: {config['win_rate']:.1f}% | PF: {config['profit_factor']:.2f}")
        logger.info(f"   Trades: {config['total_trades']:.0f} | Sharpe: {config['sharpe_ratio']:.2f} | Max DD: {config['max_drawdown']:.2f}%")
        logger.info("")

    print_config("Best ROI", best_roi)
    print_config("Best Sharpe Ratio", best_sharpe)
    print_config("Best Win Rate", best_wr)
    print_config("Best Profit Factor", best_pf)

    # Overall recommendation (weighted score)
    logger.info(f"{'='*80}")
    logger.info(f"💡 RECOMMENDED CONFIGURATION (Weighted Score)")
    logger.info(f"{'='*80}\n")

    df_valid['score'] = (
        df_valid['roi'] / df_valid['roi'].max() * 0.3 +
        df_valid['sharpe_ratio'] / df_valid['sharpe_ratio'].max() * 0.25 +
        df_valid['win_rate'] / df_valid['win_rate'].max() * 0.20 +
        df_valid['profit_factor'] / df_valid['profit_factor'].max() * 0.15 +
        (1 - abs(df_valid['max_drawdown']) / abs(df_valid['max_drawdown']).max()) * 0.10
    )

    best_overall = df_valid.loc[df_valid['score'].idxmax()]

    logger.info(f"🏆 BEST OVERALL:")
    logger.info(f"   TP: {best_overall['tp_atr_mult']:.1f}x ATR")
    logger.info(f"   SL: {best_overall['sl_atr_mult']:.1f}x ATR")
    logger.info(f"   MIN_ML_CONFIDENCE: {best_overall['min_confidence']:.2f}")
    logger.info("")
    logger.info(f"📊 Expected Performance:")
    logger.info(f"   ROI: {best_overall['roi']:+.2f}%")
    logger.info(f"   Win Rate: {best_overall['win_rate']:.1f}%")
    logger.info(f"   Profit Factor: {best_overall['profit_factor']:.2f}")
    logger.info(f"   Sharpe Ratio: {best_overall['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {best_overall['max_drawdown']:.2f}%")
    logger.info(f"   Total Trades: {best_overall['total_trades']:.0f}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(description='Universal ML Model Validator')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pkl file')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--save-csv', type=str, help='Save results to CSV file')

    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"🔬 UNIVERSAL ML MODEL VALIDATOR")
    logger.info(f"{'='*80}\n")
    logger.info(f"Model: {args.model}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Period: {args.days} days")
    logger.info("")

    try:
        # Load model
        validator = UniversalModelValidator(args.model)

        # Load and prepare data
        df = load_and_prepare_data(args.symbol, args.timeframe, args.days)

        # Define test configurations
        confidence_levels = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70]
        tp_sl_configs = [
            (2.0, 1.5),  # Original
            (2.5, 1.0),  # Higher R:R
            (3.0, 1.0),  # Very high R:R
            (2.0, 1.0),  # Tight SL
            (1.5, 1.0),  # Conservative
        ]

        # Run validation
        df_results = validator.validate(
            df,
            confidence_levels=confidence_levels,
            tp_sl_configs=tp_sl_configs,
            fees_pct=0.06,
            slippage_pct=0.01
        )

        # Print results
        print_results_table(df_results)

        # Analyze best configuration
        analyze_best_config(df_results)

        # Save to CSV if requested
        if args.save_csv:
            df_results.to_csv(args.save_csv, index=False)
            logger.info(f"\n💾 Results saved to: {args.save_csv}")

        logger.info(f"\n{'='*80}\n")

    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
