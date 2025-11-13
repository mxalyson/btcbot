"""
VALIDAÇÃO COMPLETA DA ESTRATÉGIA
Testa a estratégia com diferentes parâmetros para validar robustez
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*Boolean Series key.*')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List
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
        
        if not self.model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)
        
        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']
        
        self.initial_capital = config.get('initial_capital', 10000)
        self.risk_per_trade = config.get('risk_per_trade_pct', 0.75) / 100
    
    def backtest_with_confidence(self, df: pd.DataFrame, min_confidence: float) -> Dict:
        """Run backtest com filtro de confiança mínima."""
        
        # Get ML predictions
        X = df[self.feature_names].fillna(0)
        ml_probs = self.model.predict(X)
        
        df['ml_prob_up'] = ml_probs
        df['ml_prob_down'] = 1 - ml_probs
        df['ml_confidence'] = np.abs(ml_probs - 0.5) * 2
        
        # Generate signals with confidence filter
        df['signal'] = 0
        mask_long = (df['ml_prob_up'] > 0.5) & (df['ml_confidence'] >= min_confidence)
        mask_short = (df['ml_prob_down'] > 0.5) & (df['ml_confidence'] >= min_confidence)
        
        df.loc[mask_long, 'signal'] = 1
        df.loc[mask_short, 'signal'] = -1
        
        # Simulate
        trades = self._simulate(df)
        
        # Stats
        stats = self._calculate_stats(trades, df, min_confidence)
        
        return stats
    
    def _simulate(self, df: pd.DataFrame) -> List[Dict]:
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
                position = self._open_trade(current, capital, i)
        
        # Close final position
        if position:
            trade = self._close_trade(position, df.iloc[-1], 'end_of_data')
            trades.append(trade)
        
        return trades
    
    def _open_trade(self, current, capital, idx):
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)
        
        if direction == 'long':
            sl = price - (atr * 2.0)
            tp1 = price + (atr * 1.0)
            tp2 = price + (atr * 2.0)
            tp3 = price + (atr * 3.0)
        else:
            sl = price + (atr * 2.0)
            tp1 = price - (atr * 1.0)
            tp2 = price - (atr * 2.0)
            tp3 = price - (atr * 3.0)
        
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
            'tp2': tp2,
            'tp3': tp3,
            'ml_confidence': current['ml_confidence']
        }
    
    def _check_exit(self, position, current, idx):
        high = current['high']
        low = current['low']
        direction = position['direction']
        
        if direction == 'long':
            if low <= position['stop_loss']:
                return 'stop_loss'
            if high >= position['tp3']:
                return 'take_profit_3'
            if high >= position['tp2']:
                return 'take_profit_2'
            if high >= position['tp1']:
                return 'take_profit_1'
        else:
            if high >= position['stop_loss']:
                return 'stop_loss'
            if low <= position['tp3']:
                return 'take_profit_3'
            if low <= position['tp2']:
                return 'take_profit_2'
            if low <= position['tp1']:
                return 'take_profit_1'
        
        # Time exit (48h)
        if idx - position['entry_idx'] > 192:
            return 'time_exit'
        
        return None
    
    def _close_trade(self, position, current, reason):
        if reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif reason == 'take_profit_1':
            exit_price = position['tp1']
        elif reason == 'take_profit_2':
            exit_price = position['tp2']
        elif reason == 'take_profit_3':
            exit_price = position['tp3']
        else:
            exit_price = current['close']
        
        entry = position['entry_price']
        direction = position['direction']
        
        if direction == 'long':
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100
        
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
    
    def _calculate_stats(self, trades, df, min_confidence):
        if not trades:
            return {
                'error': 'No trades',
                'total_trades': 0,
                'min_confidence': min_confidence
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
        
        # Confidence stats
        avg_confidence = df_trades['ml_confidence'].mean()
        
        return {
            'min_confidence': min_confidence,
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
    parser.add_argument('--model', type=str, default='ml_model_master_scalper_365d.pkl')
    
    args = parser.parse_args()
    
    config = load_config('standard')
    logger = setup_logging('INFO', log_to_file=False)
    
    logger.info("=" * 80)
    logger.info("🔬 VALIDAÇÃO COMPLETA DA ESTRATÉGIA")
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
    
    # Features
    logger.info("🔨 Building features...")
    fs = FeatureStore(config)
    df_features = fs.build_features(df, normalize=False)
    df_features = create_advanced_features(df_features)
    logger.info(f"✅ Features ready: {len(df_features.columns)} columns")
    logger.info("")
    
    # Validate strategy
    model_path = f"storage/models/{args.model}"
    
    try:
        validator = StrategyValidator(config, model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return
    
    # Test different confidence levels
    confidence_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    logger.info("=" * 80)
    logger.info("🧪 TESTANDO DIFERENTES NÍVEIS DE CONFIANÇA")
    logger.info("=" * 80)
    logger.info("")
    
    results = []
    
    for min_conf in confidence_levels:
        logger.info(f"Testing min confidence: {min_conf:.0%}...")
        stats = validator.backtest_with_confidence(df_features.copy(), min_conf)
        results.append(stats)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 RESULTADOS COMPARATIVOS")
    logger.info("=" * 80)
    logger.info("")
    
    # Create comparison table
    print_comparison_table(results, args.days)
    
    # Best configuration analysis
    logger.info("")
    logger.info("=" * 80)
    logger.info("🏆 ANÁLISE DE MELHOR CONFIGURAÇÃO")
    logger.info("=" * 80)
    logger.info("")
    
    analyze_best_config(results)


def print_comparison_table(results, days):
    """Print comparison table."""
    
    header = f"{'Conf':<6} | {'Trades':<7} | {'WR':<6} | {'ROI':<8} | {'ROI/yr':<8} | {'PF':<6} | {'Sharpe':<7} | {'DD':<7} | {'Avg Conf':<9}"
    logger.info(header)
    logger.info("-" * len(header))
    
    for r in results:
        if r.get('total_trades', 0) > 0:
            roi_yearly = r['roi'] / (days / 365)
            line = (f"{r['min_confidence']*100:>5.0f}% | "
                   f"{r['total_trades']:>7,} | "
                   f"{r['win_rate']*100:>5.1f}% | "
                   f"{r['roi']:>+7.1f}% | "
                   f"{roi_yearly:>+7.1f}% | "
                   f"{r['profit_factor']:>5.2f} | "
                   f"{r['sharpe_ratio']:>6.2f} | "
                   f"{r['max_drawdown']:>6.1f}% | "
                   f"{r['avg_ml_confidence']*100:>8.1f}%")
            logger.info(line)
        else:
            logger.info(f"{r['min_confidence']*100:>5.0f}% | No trades")


def analyze_best_config(results):
    """Analyze and recommend best configuration."""
    
    valid_results = [r for r in results if r.get('total_trades', 0) > 0]
    
    if not valid_results:
        logger.info("❌ No valid results to analyze")
        return
    
    # Best by different metrics
    best_roi = max(valid_results, key=lambda x: x['roi'])
    best_sharpe = max(valid_results, key=lambda x: x['sharpe_ratio'])
    best_wr = max(valid_results, key=lambda x: x['win_rate'])
    min_dd = min(valid_results, key=lambda x: x['max_drawdown'])
    most_trades = max(valid_results, key=lambda x: x['total_trades'])
    
    logger.info("🎯 Melhor ROI:")
    logger.info(f"   Confiança Mínima: {best_roi['min_confidence']:.0%}")
    logger.info(f"   ROI: {best_roi['roi']:+.2f}%")
    logger.info(f"   Win Rate: {best_roi['win_rate']*100:.1f}%")
    logger.info(f"   Trades: {best_roi['total_trades']}")
    logger.info("")
    
    logger.info("📈 Melhor Sharpe Ratio:")
    logger.info(f"   Confiança Mínima: {best_sharpe['min_confidence']:.0%}")
    logger.info(f"   Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
    logger.info(f"   ROI: {best_sharpe['roi']:+.2f}%")
    logger.info(f"   Trades: {best_sharpe['total_trades']}")
    logger.info("")
    
    logger.info("🎯 Melhor Win Rate:")
    logger.info(f"   Confiança Mínima: {best_wr['min_confidence']:.0%}")
    logger.info(f"   Win Rate: {best_wr['win_rate']*100:.1f}%")
    logger.info(f"   ROI: {best_wr['roi']:+.2f}%")
    logger.info(f"   Trades: {best_wr['total_trades']}")
    logger.info("")
    
    logger.info("💪 Menor Drawdown:")
    logger.info(f"   Confiança Mínima: {min_dd['min_confidence']:.0%}")
    logger.info(f"   Max DD: {min_dd['max_drawdown']:.2f}%")
    logger.info(f"   ROI: {min_dd['roi']:+.2f}%")
    logger.info(f"   Trades: {min_dd['total_trades']}")
    logger.info("")
    
    # Recommendation
    logger.info("=" * 80)
    logger.info("💡 RECOMENDAÇÃO")
    logger.info("=" * 80)
    logger.info("")
    
    # Score each config
    scores = []
    for r in valid_results:
        if r['total_trades'] < 20:  # Too few trades
            continue
        
        score = 0
        # ROI weight: 30%
        score += (r['roi'] / max(x['roi'] for x in valid_results)) * 0.3
        # Sharpe weight: 25%
        score += (r['sharpe_ratio'] / max(x['sharpe_ratio'] for x in valid_results)) * 0.25
        # Win Rate weight: 20%
        score += (r['win_rate'] / max(x['win_rate'] for x in valid_results)) * 0.2
        # Min DD weight: 15% (inverse)
        score += (1 - abs(r['max_drawdown']) / max(abs(x['max_drawdown']) for x in valid_results)) * 0.15
        # Trade count weight: 10% (prefer more trades for statistical significance)
        score += (r['total_trades'] / max(x['total_trades'] for x in valid_results)) * 0.1
        
        scores.append((r, score))
    
    if scores:
        best = max(scores, key=lambda x: x[1])
        r = best[0]
        
        logger.info(f"🏆 Configuração Recomendada:")
        logger.info(f"   MIN_ML_CONFIDENCE={r['min_confidence']:.2f}")
        logger.info("")
        logger.info(f"📊 Métricas:")
        logger.info(f"   Total Trades: {r['total_trades']}")
        logger.info(f"   Win Rate: {r['win_rate']*100:.1f}%")
        logger.info(f"   ROI: {r['roi']:+.2f}%")
        logger.info(f"   Sharpe: {r['sharpe_ratio']:.2f}")
        logger.info(f"   Max DD: {r['max_drawdown']:.2f}%")
        logger.info(f"   Profit Factor: {r['profit_factor']:.2f}")
        logger.info(f"   Avg Confidence: {r['avg_ml_confidence']*100:.1f}%")
        logger.info("")
        logger.info(f"💾 Adicione no seu .env:")
        logger.info(f"   MIN_ML_CONFIDENCE={r['min_confidence']}")
    
    logger.info("")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
