"""
BACKTEST OTIMIZADO - COM FILTRO DE TENDÊNCIA + TRAILING STOP
Capital $300 | Fees Bybit | 90 dias | TP1: 0.7x ATR
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from core.utils import load_config, setup_logging
    from core.bybit_rest import BybitRESTClient
    from core.data import DataManager
    from core.features import FeatureStore
    HAS_MODULES = True
except:
    HAS_MODULES = False

# Config
INITIAL_CAPITAL = 300.0
RISK_PER_TRADE = 0.02
ATR_MULT_SL = 1.5
FEE_RATE = 0.0006
SLIPPAGE = 0.0001
LOOKBACK_DAYS = 90

# TPs otimizados - TP1 em 0.8x ATR
TP_MULTS = {
    'tp1': 0.7,  # 0.7x ATR (original)
    'tp2': 1.3,  # 1.3x ATR
    'tp3': 2.0   # 2.0x ATR
}

# Trailing Stop
TRAILING_STOP_ENABLED = True
TRAILING_STOP_DISTANCE = 0.5  # 0.5x ATR

# Parcial no TP1
PARTIAL_FRACTION = 0.60
MOVE_TO_BE = True

# FILTRO DE TENDÊNCIA
USE_TREND_FILTER = False

CONFIDENCE_LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

logger = None

def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()

    for period in [3, 5, 8, 13, 21]:
        df_features[f'momentum_{period}'] = df_features['close'].pct_change(period) * 100
        df_features[f'volume_ratio_{period}'] = df_features['volume'] / df_features['volume'].rolling(period).mean()

    if 'ema50' in df_features.columns and 'ema200' in df_features.columns:
        df_features['trend_strength'] = (df_features['ema50'] - df_features['ema200']) / df_features['ema200'] * 100

    if 'atr' in df_features.columns:
        df_features['volatility_regime'] = df_features['atr'] / df_features['atr'].rolling(50).mean()

    df_features['price_position'] = ((df_features['close'] - df_features['low'].rolling(20).min()) /
                                      (df_features['high'].rolling(20).max() - df_features['low'].rolling(20).min())).fillna(0.5)
    df_features['volume_momentum'] = df_features['volume'].pct_change(5)
    df_features['price_acceleration'] = df_features['close'].diff(2) - df_features['close'].diff(1)

    return df_features

class BacktestEngine:
    def __init__(self, model_path: str, min_confidence: float):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.min_confidence = min_confidence
        self.capital = INITIAL_CAPITAL
        self.initial_capital = INITIAL_CAPITAL
        self.position = None
        self.trades = []
        self.equity = []

    def get_signal(self, df: pd.DataFrame, idx: int) -> Tuple[int, float]:
        if idx < 200:
            return 0, 0.0

        try:
            row = df.iloc[idx]
            X = df[self.feature_names].iloc[idx:idx+1].fillna(0).values

            ml_probs = self.model.predict(X)
            ml_prob_up = float(ml_probs.flatten()[0])
            ml_prob_down = 1.0 - ml_prob_up
            ml_confidence = abs(ml_prob_up - 0.5) * 2.0

            if ml_confidence < self.min_confidence:
                return 0, ml_confidence

            vol_regime = row.get('volatility_regime', 1.0)
            if vol_regime > 2.5 or vol_regime < 0.4:
                return 0, ml_confidence

            # FILTRO DE TENDÊNCIA
            if USE_TREND_FILTER:
                ema50 = row.get('ema50', 0)
                ema200 = row.get('ema200', 0)

                if ml_prob_up > 0.5:
                    # LONG: EMA50 deve estar acima EMA200
                    if ema50 > ema200:
                        return 1, ml_confidence
                    else:
                        return 0, ml_confidence
                elif ml_prob_down > 0.5:
                    # SHORT: EMA50 deve estar abaixo EMA200
                    if ema50 < ema200:
                        return -1, ml_confidence
                    else:
                        return 0, ml_confidence
            else:
                # Sem filtro de tendência
                if ml_prob_up > 0.5:
                    return 1, ml_confidence
                elif ml_prob_down > 0.5:
                    return -1, ml_confidence

            return 0, ml_confidence
        except Exception as e:
            return 0, 0.0

    def open_position(self, signal: int, row: pd.Series, ml_confidence: float, idx: int):
        price = row['close']
        atr = row.get('atr', price * 0.01)
        direction = 'long' if signal == 1 else 'short'

        if direction == 'long':
            sl = price - (atr * ATR_MULT_SL)
            tp1 = price + (atr * TP_MULTS['tp1'])
            tp2 = price + (atr * TP_MULTS['tp2'])
            tp3 = price + (atr * TP_MULTS['tp3'])
        else:
            sl = price + (atr * ATR_MULT_SL)
            tp1 = price - (atr * TP_MULTS['tp1'])
            tp2 = price - (atr * TP_MULTS['tp2'])
            tp3 = price - (atr * TP_MULTS['tp3'])

        sl_dist = abs((sl - price) / price)
        risk_amt = self.capital * RISK_PER_TRADE
        size = risk_amt / sl_dist if sl_dist > 0 else self.capital * 0.1
        size = min(size, self.capital * 0.95)

        if size < 10:
            return

        open_fee = size * FEE_RATE
        open_slip = size * SLIPPAGE
        self.capital -= (open_fee + open_slip)

        self.position = {
            'direction': direction,
            'entry_price': price,
            'entry_idx': idx,
            'size': size,
            'original_size': size,
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'ml_confidence': ml_confidence,
            'open_fee': open_fee,
            'open_slip': open_slip,
            'tp1_hit': False,
            'tp2_hit': False,
            'trailing_active': False,
            'trailing_stop': None,
            'partial_fraction': 0.0,
            'partial_pnl': 0.0,
            'partial_fee': 0.0,
            'partial_slip': 0.0,
            'atr': atr
        }

    def check_exit(self, row: pd.Series) -> str:
        if not self.position:
            return None

        price = row['close']
        high = row['high']
        low = row['low']
        direction = self.position['direction']
        sl = self.position['stop_loss']
        tp1 = self.position['tp1']
        tp2 = self.position['tp2']
        tp3 = self.position['tp3']
        tp1_hit = self.position['tp1_hit']
        tp2_hit = self.position['tp2_hit']

        # Atualizar trailing stop se ativo
        if self.position['trailing_active'] and TRAILING_STOP_ENABLED:
            atr = self.position['atr']
            trailing_distance = atr * TRAILING_STOP_DISTANCE

            if direction == 'long':
                new_trailing = high - trailing_distance
                if self.position['trailing_stop'] is None or new_trailing > self.position['trailing_stop']:
                    self.position['trailing_stop'] = new_trailing

                if low <= self.position['trailing_stop']:
                    return 'trailing_stop'
            else:
                new_trailing = low + trailing_distance
                if self.position['trailing_stop'] is None or new_trailing < self.position['trailing_stop']:
                    self.position['trailing_stop'] = new_trailing

                if high >= self.position['trailing_stop']:
                    return 'trailing_stop'

        # Check SL
        if direction == 'long':
            if low <= sl:
                return 'stop_loss'
        else:
            if high >= sl:
                return 'stop_loss'

        # Check TP1 (primeiro)
        if not tp1_hit:
            if direction == 'long':
                if high >= tp1:
                    self._handle_tp1_hit(tp1)
                    return None
            else:
                if low <= tp1:
                    self._handle_tp1_hit(tp1)
                    return None

        # Check TP2 (após TP1)
        if tp1_hit and not tp2_hit:
            if direction == 'long':
                if high >= tp2:
                    self._handle_tp2_hit()
                    return None
            else:
                if low <= tp2:
                    self._handle_tp2_hit()
                    return None

        # Check TP3 (após TP2)
        if tp2_hit:
            if direction == 'long':
                if high >= tp3:
                    return 'take_profit_3'
            else:
                if low <= tp3:
                    return 'take_profit_3'

        return None

    def _handle_tp1_hit(self, tp1_price: float):
        """Executa parcial e move para BE"""
        entry = self.position['entry_price']
        direction = self.position['direction']
        original_size = self.position['original_size']

        if direction == 'long':
            move_pct = (tp1_price - entry) / entry
        else:
            move_pct = (entry - tp1_price) / entry

        partial_size = original_size * PARTIAL_FRACTION
        pnl_gross = partial_size * move_pct

        partial_fee = partial_size * FEE_RATE
        partial_slip = partial_size * SLIPPAGE
        pnl_net = pnl_gross - partial_fee - partial_slip

        self.capital += pnl_net

        self.position['size'] = original_size * (1 - PARTIAL_FRACTION)
        self.position['tp1_hit'] = True
        self.position['partial_fraction'] = PARTIAL_FRACTION
        self.position['partial_pnl'] = pnl_net
        self.position['partial_fee'] = partial_fee
        self.position['partial_slip'] = partial_slip

        # Move SL para BE
        if MOVE_TO_BE:
            total_costs = (self.position['open_fee'] + self.position['open_slip'] +
                          partial_fee + partial_slip)
            cost_pct = total_costs / original_size

            if direction == 'long':
                be_sl = entry * (1 + cost_pct)
            else:
                be_sl = entry * (1 - cost_pct)

            self.position['stop_loss'] = be_sl

    def _handle_tp2_hit(self):
        """Ativa trailing stop após TP2"""
        self.position['tp2_hit'] = True
        if TRAILING_STOP_ENABLED:
            self.position['trailing_active'] = True
            # Trailing stop inicial = current SL (que já está em BE)
            self.position['trailing_stop'] = self.position['stop_loss']

    def close_position(self, exit_price: float, reason: str, idx: int):
        if not self.position:
            return

        entry = self.position['entry_price']
        direction = self.position['direction']
        remaining_size = self.position['size']
        original_size = self.position['original_size']

        if direction == 'long':
            move_pct = (exit_price - entry) / entry
        else:
            move_pct = (entry - exit_price) / entry

        pnl_gross_remaining = remaining_size * move_pct

        close_fee = remaining_size * FEE_RATE
        close_slip = remaining_size * SLIPPAGE
        pnl_net_remaining = pnl_gross_remaining - close_fee - close_slip

        total_pnl_net = self.position['partial_pnl'] + pnl_net_remaining

        total_fees = (self.position['open_fee'] +
                     self.position['partial_fee'] +
                     close_fee)
        total_slip = (self.position['open_slip'] +
                     self.position['partial_slip'] +
                     close_slip)

        self.capital += pnl_net_remaining

        bars_held = idx - self.position['entry_idx']

        is_be_win = (reason in ['stop_loss', 'trailing_stop'] and
                    self.position['tp1_hit'] and
                    MOVE_TO_BE and
                    total_pnl_net >= -1.0)

        trade = {
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_price,
            'original_size': original_size,
            'bars_held': bars_held,
            'pnl_net': total_pnl_net,
            'pnl_gross': self.position['partial_pnl'] + pnl_gross_remaining,
            'total_fees': total_fees,
            'total_slip': total_slip,
            'exit_reason': reason,
            'ml_confidence': self.position['ml_confidence'],
            'tp1_hit': self.position['tp1_hit'],
            'tp2_hit': self.position['tp2_hit'],
            'partial_fraction': self.position['partial_fraction'],
            'is_be_win': is_be_win
        }

        self.trades.append(trade)
        self.position = None

    def run(self, df: pd.DataFrame):
        print(f"Processando {len(df)} candles...")

        for idx in range(len(df)):
            row = df.iloc[idx]

            if idx % 100 == 0:
                self.equity.append({
                    'idx': idx,
                    'capital': self.capital
                })

            if self.position:
                exit_reason = self.check_exit(row)
                if exit_reason:
                    exit_price = {
                        'stop_loss': self.position['stop_loss'],
                        'take_profit_2': self.position['tp2'],
                        'take_profit_3': self.position['tp3'],
                        'trailing_stop': self.position.get('trailing_stop', row['close'])
                    }.get(exit_reason, row['close'])
                    self.close_position(exit_price, exit_reason, idx)
            else:
                signal, ml_confidence = self.get_signal(df, idx)
                if signal != 0:
                    self.open_position(signal, row, ml_confidence, idx)

        if self.position:
            self.close_position(df.iloc[-1]['close'], 'end_of_data', len(df)-1)

        print(f"✅ {len(self.trades)} trades")

    def get_stats(self) -> Dict:
        if not self.trades:
            return {
                'confidence': self.min_confidence,
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'be_wins': 0,
                'win_rate': 0,
                'win_rate_adjusted': 0,
                'roi': 0,
                'final_capital': self.capital,
                'total_fees': 0,
                'total_slip': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'trades_per_month': 0
            }

        df = pd.DataFrame(self.trades)
        total = len(df)
        wins = len(df[df['pnl_net'] > 0])
        losses = len(df[df['pnl_net'] < 0])
        be_wins = len(df[df['is_be_win']])

        win_rate = wins / total * 100
        win_rate_adjusted = (wins + be_wins) / total * 100

        total_pnl = df['pnl_net'].sum()
        roi = (total_pnl / self.initial_capital) * 100

        total_fees = df['total_fees'].sum()
        total_slip = df['total_slip'].sum()

        avg_win = df[df['pnl_net'] > 0]['pnl_net'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl_net'] < 0]['pnl_net'].mean() if losses > 0 else 0

        profit_factor = (abs(df[df['pnl_net'] > 0]['pnl_net'].sum() /
                            df[df['pnl_net'] < 0]['pnl_net'].sum())
                        if losses > 0 else 0)

        trades_per_month = total / (LOOKBACK_DAYS / 30)

        return {
            'confidence': self.min_confidence,
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'be_wins': be_wins,
            'win_rate': win_rate,
            'win_rate_adjusted': win_rate_adjusted,
            'roi': roi,
            'total_fees': total_fees,
            'total_slip': total_slip,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_per_month': trades_per_month
        }

def main():
    global logger

    print("\n🔬 BACKTEST SEM FILTRO DE TENDÊNCIA + TRAILING STOP")
    print(f"Capital: ${INITIAL_CAPITAL}")
    print(f"Período: {LOOKBACK_DAYS} dias")
    print(f"TPs: {TP_MULTS['tp1']}x, {TP_MULTS['tp2']}x, {TP_MULTS['tp3']}x ATR")
    print(f"SL: {ATR_MULT_SL}x ATR")
    print(f"Parcial: {PARTIAL_FRACTION*100:.0f}% no TP1")
    print(f"Trailing Stop: {'✅ ATIVADO' if TRAILING_STOP_ENABLED else '❌ DESATIVADO'} ({TRAILING_STOP_DISTANCE}x ATR)")
    print(f"Filtro Tendência: {'✅ ATIVADO' if USE_TREND_FILTER else '❌ DESATIVADO'}")
    print(f"Fees: {FEE_RATE:.2%}\n")

    df = None

    if HAS_MODULES:
        try:
            config = load_config('standard')
            logger = setup_logging('INFO', log_to_file=False)

            rest_client = BybitRESTClient(
                api_key=os.getenv('BYBIT_API_KEY'),
                api_secret=os.getenv('BYBIT_API_SECRET'),
                testnet=os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
            )

            dm = DataManager(rest_client)
            fs = FeatureStore(config)

            logger.info(f"📥 Buscando {LOOKBACK_DAYS} dias...")
            df = dm.get_data('BTCUSDT', '15m', LOOKBACK_DAYS)

            if df is not None and not df.empty:
                logger.info(f"✅ {len(df)} candles")
                df = fs.build_features(df, normalize=False)
                df = create_advanced_features(df)
                df = df.dropna()
                logger.info(f"✅ {len(df)} válidos\n")
        except Exception as e:
            print(f"⚠️ Erro: {e}")
            df = None

    if df is None or (hasattr(df, 'empty') and df.empty):
        print("❌ Dados não disponíveis")
        return

    model_path = 'storage/models/ml_model_master_scalper_365d.pkl'
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        return

    results = []

    for confidence in CONFIDENCE_LEVELS:
        print(f"\n{'='*60}")
        print(f"Confidence: {confidence:.0%}")
        print(f"{'='*60}")

        bt = BacktestEngine(model_path, confidence)
        bt.run(df)
        stats = bt.get_stats()
        results.append(stats)

        print(f"\nTrades: {stats['total_trades']} ({stats['trades_per_month']:.1f}/mês)")
        print(f"Win Rate: {stats['win_rate']:.1f}% | Adjusted: {stats['win_rate_adjusted']:.1f}%")
        print(f"BE Wins: {stats['be_wins']}")
        print(f"ROI: {stats['roi']:+.2f}%")
        print(f"Capital: ${stats['initial_capital']:.2f} → ${stats['final_capital']:.2f}")
        print(f"Fees: ${stats['total_fees']:.2f} | Slip: ${stats['total_slip']:.2f}")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")

    print(f"\n{'='*60}")
    print("📊 COMPARAÇÃO")
    print(f"{'='*60}\n")

    results_df = pd.DataFrame(results)
    display = results_df[[
        'confidence', 'total_trades', 'trades_per_month',
        'win_rate', 'win_rate_adjusted', 'be_wins',
        'roi', 'final_capital', 'profit_factor'
    ]].copy()
    display.columns = ['Conf', 'Trades', 'T/Mês', 'WR%', 'WR_Adj%', 'BE', 'ROI%', 'Capital', 'PF']
    print(display.to_string(index=False))

    with open('backtest_results_no_trend_filter_07.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Salvo: backtest_results_no_trend_filter_07.json")

    viable = results_df[
        (results_df['total_trades'] >= 15) &
        (results_df['roi'] > 0) &
        (results_df['win_rate_adjusted'] > 65)
    ]

    if len(viable) > 0:
        viable['score'] = (
            viable['roi'] * 0.4 +
            viable['win_rate_adjusted'] * 0.3 +
            viable['profit_factor'] * 10 * 0.3
        )

        best = viable.loc[viable['score'].idxmax()]
        print(f"\n🎯 RECOMENDADO: {best['confidence']:.0%}")
        print(f"   ROI: {best['roi']:+.2f}%")
        print(f"   WR Adjusted: {best['win_rate_adjusted']:.1f}%")
        print(f"   Trades: {best['total_trades']:.0f} ({best['trades_per_month']:.1f}/mês)")
        print(f"   BE Wins: {best['be_wins']:.0f}")
        print(f"\n💡 MIN_ML_CONFIDENCE={best['confidence']:.2f}\n")
    else:
        print("\n⚠️ Ajustar parâmetros necessário\n")

if __name__ == "__main__":
    main()
