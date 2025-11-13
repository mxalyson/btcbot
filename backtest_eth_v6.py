"""
BACKTEST ETH - V6.0 - ESPELHA EXATAMENTE eth_live_v3.py

✅ LÓGICA IMPLEMENTADA (igual ao live):
   - TP1 (0.7x ATR): Fecha 60% + Move SL para BE
   - TP2 (1.3x ATR): Ativa trailing stop local
   - TP3 (2.0x ATR): Fecha 40% restante APÓS TP2 ✓
   - Trailing Stop: Atualiza SL continuamente após TP2
   - SL inicial: 1.5x ATR

✅ CORREÇÕES V6.0 APLICADAS:
   - TP3 verifica após TP2 (não baseado em trailing_active)
   - Mesmas features do live
   - Mesma lógica de confidence
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
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

# ======== CONFIGURAÇÕES - IGUAIS AO LIVE ========
TP_MULTS = {
    'tp1': 0.7,   # 60% parcial
    'tp2': 1.3,   # Ativa trailing
    'tp3': 2.0    # 40% restante
}
PARTIAL_FRACTION = 0.60  # 60% no TP1
TRAILING_STOP_DISTANCE = 0.5  # 0.5x ATR


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MASTER TRADER advanced features - SAME as eth_live_v3.py!"""

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


class ETHBacktesterV6:
    """Backtest com EXATAMENTE a mesma lógica do eth_live_v3.py V6.0"""

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

        self.initial_capital = 10000
        self.risk_per_trade = 0.02  # 2%
        self.min_confidence = 0.40  # 40% (igual ao live)
        self.atr_mult_sl = 1.5      # SL: 1.5x ATR (igual ao live)

    def backtest(self, df: pd.DataFrame) -> Dict:
        """Run backtest com lógica EXATA do live"""

        # Get ML predictions
        X = df[self.feature_names].fillna(0)
        ml_probs = self.model.predict(X)

        df['ml_prob_up'] = ml_probs
        df['ml_prob_down'] = 1 - ml_probs
        df['ml_confidence'] = np.abs(ml_probs - 0.5) * 2

        # Generate signals with confidence filter
        df['signal'] = 0
        mask_long = (df['ml_prob_up'] > 0.5) & (df['ml_confidence'] >= self.min_confidence)
        mask_short = (df['ml_prob_down'] > 0.5) & (df['ml_confidence'] >= self.min_confidence)

        df.loc[mask_long, 'signal'] = 1
        df.loc[mask_short, 'signal'] = -1

        # Simulate
        trades = self._simulate(df)

        # Stats
        stats = self._calculate_stats(trades, df)

        return stats

    def _simulate(self, df: pd.DataFrame) -> List[Dict]:
        trades = []
        position = None
        capital = self.initial_capital
        cooldown = 0

        # Estados da posição (igual ao live)
        tp1_hit = False
        tp2_hit = False
        trailing_active = False
        highest_price = None
        lowest_price = None

        for i in range(len(df)):
            current = df.iloc[i]
            high = current['high']
            low = current['low']
            close = current['close']

            if cooldown > 0:
                cooldown -= 1

            # ✅ MONITORA POSIÇÃO ABERTA (lógica EXATA do live)
            if position:
                direction = position['direction']
                entry = position['entry_price']
                sl = position['current_sl']
                tp1 = position['tp1']
                tp2 = position['tp2']
                tp3 = position['tp3']
                atr = position['atr']
                remaining_qty = position['remaining_qty']

                # Check SL
                sl_hit = (direction == 'long' and low <= sl) or (direction == 'short' and high >= sl)
                if sl_hit:
                    exit_price = sl
                    trade = self._close_trade(position, exit_price, 'stop_loss', remaining_qty)
                    trades.append(trade)
                    capital += trade['pnl_amount']
                    position = None
                    tp1_hit = tp2_hit = trailing_active = False
                    highest_price = lowest_price = None
                    cooldown = 4
                    continue

                # ✅ VERIFICA TP1 (60% PARCIAL + MOVE SL PARA BE)
                if not tp1_hit:
                    tp1_triggered = (direction == 'long' and high >= tp1) or (direction == 'short' and low <= tp1)

                    if tp1_triggered:
                        tp1_hit = True

                        # Fecha 60% parcial
                        partial_qty = position['qty'] * PARTIAL_FRACTION
                        exit_price = tp1

                        if direction == 'long':
                            pnl_pct = ((exit_price - entry) / entry) * 100
                        else:
                            pnl_pct = ((entry - exit_price) / entry) * 100

                        pnl_amount = (partial_qty * entry) * (pnl_pct / 100)
                        capital += pnl_amount

                        # Atualiza posição
                        remaining_qty = position['qty'] - partial_qty
                        position['remaining_qty'] = remaining_qty

                        # ✅ MOVE SL PARA BREAKEVEN
                        position['current_sl'] = entry

                        logger.debug(f"TP1 HIT @ ${tp1:,.2f} - Closed 60% - SL → BE")

                # ✅ VERIFICA TP2 (ATIVA TRAILING STOP)
                if tp1_hit and not tp2_hit:
                    tp2_triggered = (direction == 'long' and high >= tp2) or (direction == 'short' and low <= tp2)

                    if tp2_triggered:
                        tp2_hit = True
                        trailing_active = True

                        # Inicializa trailing
                        if direction == 'long':
                            highest_price = close
                        else:
                            lowest_price = close

                        logger.debug(f"TP2 HIT @ ${tp2:,.2f} - Trailing ACTIVATED")

                # ✅ ATUALIZA TRAILING STOP (após TP2)
                if trailing_active:
                    if direction == 'long':
                        if close > highest_price:
                            highest_price = close
                            new_sl = highest_price - (atr * TRAILING_STOP_DISTANCE)

                            if new_sl > position['current_sl']:
                                position['current_sl'] = new_sl
                                logger.debug(f"Trailing SL updated: ${new_sl:,.2f}")

                    else:  # short
                        if close < lowest_price:
                            lowest_price = close
                            new_sl = lowest_price + (atr * TRAILING_STOP_DISTANCE)

                            if new_sl < position['current_sl']:
                                position['current_sl'] = new_sl
                                logger.debug(f"Trailing SL updated: ${new_sl:,.2f}")

                # ✅ VERIFICA TP3 (FECHA RESTO) - Corrigido: verifica após TP2
                if tp1_hit and tp2_hit:
                    tp3_triggered = (direction == 'long' and high >= tp3) or (direction == 'short' and low <= tp3)

                    if tp3_triggered:
                        exit_price = tp3
                        trade = self._close_trade(position, exit_price, 'take_profit_3', remaining_qty)
                        trades.append(trade)
                        capital += trade['pnl_amount']
                        position = None
                        tp1_hit = tp2_hit = trailing_active = False
                        highest_price = lowest_price = None
                        cooldown = 4
                        logger.debug(f"TP3 HIT @ ${tp3:,.2f} - Position CLOSED")
                        continue

            # ✅ CHECK ENTRY
            if not position and current['signal'] != 0 and cooldown == 0 and i < len(df) - 20:
                position = self._open_trade(current, capital, i)
                tp1_hit = tp2_hit = trailing_active = False
                highest_price = lowest_price = None

        # Close final position
        if position:
            remaining_qty = position.get('remaining_qty', position['qty'])
            trade = self._close_trade(position, df.iloc[-1]['close'], 'end_of_data', remaining_qty)
            trades.append(trade)

        return trades

    def _open_trade(self, current, capital, idx):
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)

        # ✅ Calcula SL e TPs (igual ao live)
        if direction == 'long':
            sl = price - (atr * self.atr_mult_sl)  # 1.5x ATR
            tp1 = price + (atr * TP_MULTS['tp1'])  # 0.7x ATR
            tp2 = price + (atr * TP_MULTS['tp2'])  # 1.3x ATR
            tp3 = price + (atr * TP_MULTS['tp3'])  # 2.0x ATR
        else:
            sl = price + (atr * self.atr_mult_sl)
            tp1 = price - (atr * TP_MULTS['tp1'])
            tp2 = price - (atr * TP_MULTS['tp2'])
            tp3 = price - (atr * TP_MULTS['tp3'])

        # Calcula tamanho
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
            'qty': size / price,
            'remaining_qty': size / price,
            'stop_loss': sl,
            'current_sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'atr': atr,
            'ml_confidence': current['ml_confidence']
        }

    def _close_trade(self, position, exit_price, reason, qty):
        entry = position['entry_price']
        direction = position['direction']

        if direction == 'long':
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100

        pnl_amount = (qty * entry) * (pnl_pct / 100)

        return {
            'entry_time': position['entry_time'],
            'exit_price': exit_price,
            'direction': direction,
            'entry_price': entry,
            'qty_closed': qty,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'ml_confidence': position['ml_confidence']
        }

    def _calculate_stats(self, trades, df):
        if not trades:
            return {
                'error': 'No trades',
                'total_trades': 0
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

        # Reason breakdown
        reason_counts = df_trades['reason'].value_counts().to_dict()

        return {
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
            'avg_ml_confidence': df_trades['ml_confidence'].mean(),
            'exit_reasons': reason_counts
        }


def main():
    global logger

    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default='ETHUSDT')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--model', type=str, default='ml_model_master_scalper_365d.pkl')

    args = parser.parse_args()

    config = load_config('standard')
    logger = setup_logging('INFO', log_to_file=False)

    logger.info("=" * 80)
    logger.info("🔬 BACKTEST ETH V6.0 - ESPELHA eth_live_v3.py")
    logger.info("=" * 80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.days} days")
    logger.info(f"Model: {args.model}")
    logger.info("")
    logger.info("✅ LÓGICA IMPLEMENTADA:")
    logger.info(f"   - TP1 ({TP_MULTS['tp1']}x ATR): Fecha {PARTIAL_FRACTION*100:.0f}% + Move SL para BE")
    logger.info(f"   - TP2 ({TP_MULTS['tp2']}x ATR): Ativa trailing stop")
    logger.info(f"   - TP3 ({TP_MULTS['tp3']}x ATR): Fecha resto APÓS TP2 ✓")
    logger.info(f"   - Trailing: {TRAILING_STOP_DISTANCE}x ATR")
    logger.info(f"   - SL inicial: 1.5x ATR")
    logger.info(f"   - Min Confidence: 40%")
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

    # Backtest
    model_path = f"storage/models/{args.model}"

    try:
        backtester = ETHBacktesterV6(config, model_path)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    logger.info("=" * 80)
    logger.info("🧪 RUNNING BACKTEST")
    logger.info("=" * 80)
    logger.info("")

    stats = backtester.backtest(df_features.copy())

    if 'error' in stats:
        logger.error(f"❌ {stats['error']}")
        return

    # Print results
    logger.info("=" * 80)
    logger.info("📊 RESULTADOS")
    logger.info("=" * 80)
    logger.info("")

    logger.info(f"💰 Performance:")
    logger.info(f"   Total Trades: {stats['total_trades']}")
    logger.info(f"   Win Rate: {stats['win_rate']*100:.1f}%")
    logger.info(f"   Total PnL: ${stats['total_pnl']:+,.2f}")
    logger.info(f"   ROI: {stats['roi']:+.2f}%")
    logger.info(f"   ROI Anualizado: {stats['roi'] / (args.days / 365):+.2f}%")
    logger.info(f"   Final Capital: ${stats['final_capital']:,.2f}")
    logger.info("")

    logger.info(f"📈 Métricas:")
    logger.info(f"   Avg Win: ${stats['avg_win']:,.2f}")
    logger.info(f"   Avg Loss: ${stats['avg_loss']:,.2f}")
    logger.info(f"   Profit Factor: {stats['profit_factor']:.2f}")
    logger.info(f"   Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {stats['max_drawdown']:.2f}%")
    logger.info(f"   Avg ML Confidence: {stats['avg_ml_confidence']*100:.1f}%")
    logger.info("")

    logger.info(f"📌 Exit Reasons:")
    for reason, count in stats['exit_reasons'].items():
        pct = (count / stats['total_trades']) * 100
        logger.info(f"   {reason}: {count} ({pct:.1f}%)")
    logger.info("")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
