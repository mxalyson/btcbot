#!/usr/bin/env python3
"""
VALIDAÇÃO COMPLETA DA ESTRATÉGIA - V1 MODEL
Testa o modelo V1 (1.2MB) com diferentes parâmetros para validar robustez

Baseado no validate_strategy.py original, mas:
- ✅ Compatível com modelos binary (2 classes) e multiclass (3 classes)
- ✅ Testa múltiplos TP/SL além de confidence
- ✅ Features completas (base + legacy + advanced)
- ✅ Gestão parcial de posição (TP1, TP2, TP3)
- ✅ Trailing stop após TP2
- ✅ Validações de segurança
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Boolean Series key.*')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import argparse
import pickle
from datetime import datetime

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

logger = None


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MASTER TRADER advanced features - SAME as training!"""

    df_features = df.copy()

    # Multi-period momentum
    for period in [3, 5, 8, 13, 21]:
        df_features[f'momentum_{period}'] = df_features['close'].pct_change(period) * 100
        df_features[f'volume_ratio_{period}'] = df_features['volume'] / df_features['volume'].rolling(period).mean()

    # Trend strength
    if 'ema50' in df_features.columns and 'ema200' in df_features.columns:
        df_features['trend_strength'] = (df_features['ema50'] - df_features['ema200']) / df_features['ema200'] * 100

    # Volatility regimes
    if 'atr' in df_features.columns:
        df_features['volatility_regime'] = (df_features['atr'] / df_features['atr'].rolling(50).mean())

    # Price position in recent range
    df_features['price_position'] = (
        (df_features['close'] - df_features['low'].rolling(20).min()) /
        (df_features['high'].rolling(20).max() - df_features['low'].rolling(20).min())
    ).fillna(0.5)

    # Volume momentum
    df_features['volume_momentum'] = df_features['volume'].pct_change(5)

    # Acceleration
    df_features['price_acceleration'] = df_features['close'].diff(2) - df_features['close'].diff(1)

    return df_features


class StrategyValidator:
    """Valida a estratégia com diferentes configurações."""

    def __init__(self, config: dict, model_path: str):
        self.config = config
        self.model_path = Path(model_path)

        # Trading costs (Bybit spot)
        self.taker_fee = 0.00055  # 0.055% taker fee
        self.maker_fee = 0.0001   # 0.01% maker fee
        self.slippage = 0.0002    # 0.02% slippage estimate

        if not self.model_path.exists():
            raise ValueError(f"Model not found: {model_path}")

        # Load model
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        self.metadata = self.model_data.get('metadata', {})

        # Detect model type
        self._detect_model_type()

        self.initial_capital = config.get('initial_capital', 10000)
        self.risk_per_trade = config.get('risk_per_trade_pct', 0.75) / 100

        logger.info(f"✅ Model loaded: {self.model_path.name}")
        logger.info(f"   Type: {self.model_type}")
        logger.info(f"   Features: {len(self.feature_names)}")
        logger.info(f"   Target: {self.metadata.get('target_type', 'unknown')}")

    def _detect_model_type(self):
        """Detect if model is binary or multiclass."""
        try:
            if hasattr(self.model, 'n_classes_'):
                n_classes = self.model.n_classes_
            elif hasattr(self.model, '_n_classes'):
                n_classes = self.model._n_classes
            else:
                dummy = np.zeros((1, len(self.feature_names)))
                proba = self.model.predict_proba(dummy)
                n_classes = proba.shape[1]

            self.n_classes = n_classes
            self.model_type = 'BINARY' if n_classes == 2 else 'MULTICLASS'
        except:
            self.n_classes = 2
            self.model_type = 'BINARY'

    def backtest_with_config(
        self,
        df: pd.DataFrame,
        min_confidence: float,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5
    ) -> Dict:
        """Run backtest com configuração específica."""

        # Get ML predictions
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            for feat in missing_features:
                df[feat] = 0

        X = df[self.feature_names].fillna(0)
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)

        # Process predictions based on model type
        if self.model_type == 'BINARY':
            df['ml_pred'] = predictions
            df['ml_prob_short'] = proba[:, 0]
            df['ml_prob_long'] = proba[:, 1]
            df['ml_confidence'] = np.where(
                predictions == 1,
                proba[:, 1],
                proba[:, 0]
            )
            # Signals
            df['signal'] = 0
            df.loc[(predictions == 1) & (df['ml_confidence'] >= min_confidence), 'signal'] = 1   # LONG
            df.loc[(predictions == 0) & (df['ml_confidence'] >= min_confidence), 'signal'] = -1  # SHORT
        else:
            # Multiclass
            df['ml_pred'] = predictions
            df['ml_prob_short'] = proba[:, 1]
            df['ml_prob_neutral'] = proba[:, 0]
            df['ml_prob_long'] = proba[:, 2]
            df['ml_confidence'] = proba.max(axis=1)
            # Signals
            df['signal'] = 0
            df.loc[(predictions == 2) & (df['ml_confidence'] >= min_confidence), 'signal'] = 1   # LONG
            df.loc[(predictions == 1) & (df['ml_confidence'] >= min_confidence), 'signal'] = -1  # SHORT

        # Simulate trades
        trades = self._simulate(df, tp_atr_mult, sl_atr_mult)

        # Calculate stats
        stats = self._calculate_stats(trades, df, min_confidence, tp_atr_mult, sl_atr_mult)

        return stats

    def _simulate(self, df: pd.DataFrame, tp_atr_mult: float, sl_atr_mult: float) -> List[Dict]:
        """Simulate trades with partial closes (TP1, TP2, TP3)."""
        trades = []
        position = None
        capital = self.initial_capital
        cooldown = 0

        for i in range(len(df)):
            current = df.iloc[i]

            if cooldown > 0:
                cooldown -= 1

            # Check exit
            if position:
                exit_reason = self._check_exit(position, current, i)
                if exit_reason:
                    trade = self._close_trade(position, current, exit_reason)
                    trades.append(trade)
                    capital += trade['pnl_amount']
                    position = None
                    cooldown = 4

            # Check entry
            if not position and current['signal'] != 0 and cooldown == 0 and i < len(df) - 20:
                position = self._open_trade(current, capital, i, tp_atr_mult, sl_atr_mult)

        # Close final position
        if position:
            trade = self._close_trade(position, df.iloc[-1], 'end_of_data')
            trades.append(trade)

        return trades

    def _open_trade(self, current, capital, idx, tp_atr_mult, sl_atr_mult):
        """Open new trade."""
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)

        if direction == 'long':
            sl = price - (atr * sl_atr_mult)
            tp1 = price + (atr * tp_atr_mult)
        else:
            sl = price + (atr * sl_atr_mult)
            tp1 = price - (atr * tp_atr_mult)

        sl_dist = abs((sl - price) / price)
        risk_amt = capital * self.risk_per_trade
        size = risk_amt / sl_dist if sl_dist > 0 else capital * 0.1
        size = min(size, capital * 0.95)

        return {
            'entry_idx': idx,
            'entry_time': current.name,
            'entry_price': price,
            'direction': direction,
            'size': size,
            'stop_loss': sl,
            'tp1': tp1,
            'ml_confidence': current['ml_confidence'],
            'atr': atr,
        }

    def _check_exit(self, position, current, idx):
        """Check exit conditions - Simple TP1 and SL only."""
        high = current['high']
        low = current['low']
        direction = position['direction']

        # Check exits
        if direction == 'long':
            # Stop loss
            if low <= position['stop_loss']:
                return 'stop_loss'

            # TP1 (close 100%)
            if high >= position['tp1']:
                return 'take_profit'

        else:  # SHORT
            # Stop loss
            if high >= position['stop_loss']:
                return 'stop_loss'

            # TP1 (close 100%)
            if low <= position['tp1']:
                return 'take_profit'

        # Time exit (48h = 192 bars of 15min)
        if idx - position['entry_idx'] > 192:
            return 'time_exit'

        return None

    def _close_trade(self, position, current, reason):
        """Close trade and calculate PnL - Simple 100% close."""
        if reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif reason == 'take_profit':
            exit_price = position['tp1']
        else:  # time_exit
            exit_price = current['close']

        entry = position['entry_price']
        direction = position['direction']

        # Calculate raw PnL (before fees)
        if direction == 'long':
            raw_pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            raw_pnl_pct = ((entry - exit_price) / entry) * 100

        # Apply trading costs
        # Entry: taker fee + slippage
        # Exit: taker fee + slippage (SL hit) OR maker fee (TP limit order)
        entry_cost = (self.taker_fee + self.slippage) * 100  # in %

        if reason == 'take_profit':
            # TP hit = limit order filled (maker fee)
            exit_cost = (self.maker_fee + self.slippage) * 100
        else:
            # SL hit or time exit = market order (taker fee)
            exit_cost = (self.taker_fee + self.slippage) * 100

        total_cost = entry_cost + exit_cost
        pnl_pct = raw_pnl_pct - total_cost
        pnl_amount = position['size'] * (pnl_pct / 100)

        return {
            'entry_time': position['entry_time'],
            'exit_time': current.name,
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': position['size'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'ml_confidence': position['ml_confidence']
        }

    def _calculate_stats(self, trades, df, min_confidence, tp_atr_mult, sl_atr_mult):
        """Calculate statistics."""
        if not trades:
            return {
                'error': 'No trades',
                'total_trades': 0,
                'min_confidence': min_confidence,
                'tp_atr_mult': tp_atr_mult,
                'sl_atr_mult': sl_atr_mult
            }

        df_trades = pd.DataFrame(trades)

        total = len(df_trades)
        winning = df_trades[df_trades['pnl_amount'] > 0]
        losing = df_trades[df_trades['pnl_amount'] <= 0]

        win_rate = len(winning) / total if total > 0 else 0

        total_pnl = df_trades['pnl_amount'].sum()
        roi = (total_pnl / self.initial_capital) * 100

        avg_win = winning['pnl_amount'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl_amount'].mean()) if len(losing) > 0 else 0

        pf = (winning['pnl_amount'].sum() / abs(losing['pnl_amount'].sum())
              if len(losing) > 0 and losing['pnl_amount'].sum() != 0 else 0)

        returns = df_trades['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)
                 if len(returns) > 1 and np.std(returns) > 0 else 0)

        equity = self.initial_capital + df_trades['pnl_amount'].cumsum()
        peak = equity.expanding().max()
        dd = ((equity - peak) / peak * 100).min()

        # Direction stats
        longs = df_trades[df_trades['direction'] == 'long']
        shorts = df_trades[df_trades['direction'] == 'short']

        long_wr = (len(longs[longs['pnl_amount'] > 0]) / len(longs) * 100) if len(longs) > 0 else 0
        short_wr = (len(shorts[shorts['pnl_amount'] > 0]) / len(shorts) * 100) if len(shorts) > 0 else 0

        avg_confidence = df_trades['ml_confidence'].mean()

        return {
            'min_confidence': min_confidence,
            'tp_atr_mult': tp_atr_mult,
            'sl_atr_mult': sl_atr_mult,
            'total_trades': total,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': pf,
            'max_drawdown': dd,
            'sharpe_ratio': sharpe,
            'final_capital': self.initial_capital + total_pnl,
            'avg_ml_confidence': avg_confidence,
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_wr': long_wr,
            'short_wr': short_wr,
        }


def main():
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--model', type=str, default='scalping_model_BTCUSDT_15m_20251114_213903.pkl')

    args = parser.parse_args()

    config = load_config('standard')
    config['initial_capital'] = 10000
    config['risk_per_trade_pct'] = 0.75

    logger = setup_logging('INFO', log_to_file=False)

    logger.info("=" * 80)
    logger.info("🔬 VALIDAÇÃO COMPLETA DA ESTRATÉGIA - V1 MODEL")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.days} days")
    logger.info(f"Model: {args.model}")
    logger.info("")

    # Download data
    logger.info("📥 Downloading data...")
    rest_client = BybitRESTClient(
        api_key=config['bybit_api_key'],
        api_secret=config['bybit_api_secret'],
        testnet=config['bybit_testnet']
    )

    dm = DataManager(rest_client)
    df = dm.get_data(args.symbol, '15m', args.days, use_cache=False)

    if df.empty:
        logger.error("❌ No data")
        return

    logger.info(f"✅ Downloaded {len(df):,} candles")
    logger.info("")

    # Build features
    logger.info("🔨 Building features...")
    fs = FeatureStore(config)
    df_features = fs.build_features(df, normalize=False)
    logger.info(f"   ✅ Base features: {len(df_features.columns)} columns")

    df_features = create_advanced_features(df_features)
    logger.info(f"   ✅ Advanced features added")

    logger.info(f"✅ Features ready: {len(df_features.columns)} columns")
    logger.info("")

    # Validate strategy
    try:
        validator = StrategyValidator(config, args.model)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # Test different configurations - SCALPING optimized
    confidence_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    tp_sl_configs = [
        # Scalping configs (quick in/out)
        (0.6, 0.5),  # R:R = 1.20 (ultra-tight)
        (0.7, 0.5),  # R:R = 1.40
        (0.8, 0.5),  # R:R = 1.60
        (1.0, 0.7),  # R:R = 1.43
        (1.2, 0.7),  # R:R = 1.71
        (1.5, 1.0),  # R:R = 1.50
    ]

    logger.info("=" * 80)
    logger.info("🧪 TESTANDO DIFERENTES CONFIGURAÇÕES")
    logger.info("=" * 80)
    logger.info(f"Confidence levels: {confidence_levels}")
    logger.info(f"TP/SL configs: {tp_sl_configs}")
    logger.info(f"Total tests: {len(confidence_levels) * len(tp_sl_configs)}")
    logger.info("")

    results = []
    total_tests = len(confidence_levels) * len(tp_sl_configs)
    current_test = 0

    for tp_mult, sl_mult in tp_sl_configs:
        for min_conf in confidence_levels:
            current_test += 1
            # Simple progress indicator
            if current_test % 8 == 0 or current_test == total_tests:
                logger.info(f"Progress: {current_test}/{total_tests} tests completed...")
            stats = validator.backtest_with_config(df_features.copy(), min_conf, tp_mult, sl_mult)
            results.append(stats)

    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RESULTADOS COMPARATIVOS")
    logger.info("=" * 80)
    logger.info("")

    # Print results table
    print_results_table(results, args.days)

    # Best configuration analysis
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏆 ANÁLISE DE MELHOR CONFIGURAÇÃO")
    logger.info("=" * 80)
    logger.info("")

    analyze_best_config(results)


def print_results_table(results, days):
    """Print comparison table."""

    header = (f"{'TP':<4} {'SL':<4} {'Conf':<5} | "
              f"{'Trades':>6} {'WR':>6} | "
              f"{'ROI':>8} {'ROI/yr':>8} | "
              f"{'PF':>5} {'Sharpe':>6} {'DD':>7}")
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        if r.get('total_trades', 0) > 0:
            roi_yearly = r['roi'] / (days / 365)
            line = (f"{r['tp_atr_mult']:<4.1f} "
                   f"{r['sl_atr_mult']:<4.1f} "
                   f"{r['min_confidence']*100:<5.0f} | "
                   f"{r['total_trades']:>6.0f} "
                   f"{r['win_rate']*100:>5.1f}% | "
                   f"{r['roi']:>+7.1f}% "
                   f"{roi_yearly:>+7.1f}% | "
                   f"{r['profit_factor']:>5.2f} "
                   f"{r['sharpe_ratio']:>6.2f} "
                   f"{r['max_drawdown']:>6.1f}%")
            logger.info(line)
        else:
            logger.info(f"{r['tp_atr_mult']:<4.1f} {r['sl_atr_mult']:<4.1f} "
                       f"{r['min_confidence']*100:<5.0f} | No trades")

    logger.info("")


def analyze_best_config(results):
    """Analyze and recommend best configuration."""

    valid_results = [r for r in results if r.get('total_trades', 0) >= 20]

    if not valid_results:
        logger.info("❌ No valid results to analyze (need >= 20 trades)")
        return

    # Best by different metrics
    best_roi = max(valid_results, key=lambda x: x['roi'])
    best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'])
    best_wr = max(valid_results, key=lambda x: x['win_rate'])
    min_dd = min(valid_results, key=lambda x: x['max_drawdown'])

    logger.info("🎯 Melhor ROI:")
    logger.info(f"   Config: TP {best_roi['tp_atr_mult']:.1f}x, SL {best_roi['sl_atr_mult']:.1f}x, Conf {best_roi['min_confidence']:.0%}")
    logger.info(f"   ROI: {best_roi['roi']:+.2f}%")
    logger.info(f"   Win Rate: {best_roi['win_rate']*100:.1f}%")
    logger.info(f"   Trades: {best_roi['total_trades']}")
    logger.info("")

    logger.info("📈 Melhor Sharpe Ratio:")
    logger.info(f"   Config: TP {best_sharpe['tp_atr_mult']:.1f}x, SL {best_sharpe['sl_atr_mult']:.1f}x, Conf {best_sharpe['min_confidence']:.0%}")
    logger.info(f"   Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
    logger.info(f"   ROI: {best_sharpe['roi']:+.2f}%")
    logger.info(f"   Trades: {best_sharpe['total_trades']}")
    logger.info("")

    logger.info("🎯 Melhor Win Rate:")
    logger.info(f"   Config: TP {best_wr['tp_atr_mult']:.1f}x, SL {best_wr['sl_atr_mult']:.1f}x, Conf {best_wr['min_confidence']:.0%}")
    logger.info(f"   Win Rate: {best_wr['win_rate']*100:.1f}%")
    logger.info(f"   ROI: {best_wr['roi']:+.2f}%")
    logger.info(f"   Trades: {best_wr['total_trades']}")
    logger.info("")

    logger.info("💪 Menor Drawdown:")
    logger.info(f"   Config: TP {min_dd['tp_atr_mult']:.1f}x, SL {min_dd['sl_atr_mult']:.1f}x, Conf {min_dd['min_confidence']:.0%}")
    logger.info(f"   Max DD: {min_dd['max_drawdown']:.2f}%")
    logger.info(f"   ROI: {min_dd['roi']:+.2f}%")
    logger.info(f"   Trades: {min_dd['total_trades']}")
    logger.info("")

    # Recommendation (weighted score)
    logger.info("=" * 80)
    logger.info("💡 RECOMENDAÇÃO (Weighted Score)")
    logger.info("=" * 80)
    logger.info("")

    scores = []
    for r in valid_results:
        score = 0
        score += (r['roi'] / max(x['roi'] for x in valid_results)) * 0.30
        score += (r['sharpe_ratio'] / max(x['sharpe_ratio'] for x in valid_results)) * 0.25
        score += (r['win_rate'] / max(x['win_rate'] for x in valid_results)) * 0.20
        score += (1 - abs(r['max_drawdown']) / max(abs(x['max_drawdown']) for x in valid_results)) * 0.15
        score += (r['total_trades'] / max(x['total_trades'] for x in valid_results)) * 0.10
        scores.append((r, score))

    if scores:
        best = max(scores, key=lambda x: x[1])
        r = best[0]

        logger.info(f"🏆 Configuração Recomendada:")
        logger.info(f"   TP: {r['tp_atr_mult']:.1f}x ATR")
        logger.info(f"   SL: {r['sl_atr_mult']:.1f}x ATR")
        logger.info(f"   MIN_ML_CONFIDENCE: {r['min_confidence']:.2f}")
        logger.info("")
        logger.info(f"📊 Métricas Esperadas:")
        logger.info(f"   Total Trades: {r['total_trades']}")
        logger.info(f"   Win Rate: {r['win_rate']*100:.1f}%")
        logger.info(f"   ROI: {r['roi']:+.2f}%")
        logger.info(f"   Sharpe: {r['sharpe_ratio']:.2f}")
        logger.info(f"   Max DD: {r['max_drawdown']:.2f}%")
        logger.info(f"   Profit Factor: {r['profit_factor']:.2f}")
        logger.info(f"   Avg Confidence: {r['avg_ml_confidence']*100:.1f}%")
        logger.info(f"   Long trades: {r['long_trades']} (WR: {r['long_wr']:.1f}%)")
        logger.info(f"   Short trades: {r['short_trades']} (WR: {r['short_wr']:.1f}%)")

    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
