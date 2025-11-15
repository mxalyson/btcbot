"""
🔬 BACKTEST COMPLETO PARA MODELO ML - V2.0

Backtest realista seguindo os padrões do validate_strategy.py:
- Download de dados reais da Bybit
- Criação de features completas
- Previsões do modelo ML
- Simulação com SL/TP baseados em ATR
- Fees (0.06%) e slippage (0.01%)
- Métricas detalhadas

Usage:
    python backtest_ml_model.py --model ../outputs/scalping_model_BTCUSDT_15m_20251114_213903.pkl --days 90

Author: Claude
Date: 2025-11-14
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import argparse
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

# Import advanced features (same as training)
sys.path.append(str(Path(__file__).parent.parent))
from features.advanced_features import ScalpingFeatureEngineer, create_legacy_features

logger = None


class MLBacktester:
    """Backtest realista para modelo ML de scalping."""

    def __init__(self, model_path: str, config: dict, atr_mult_tp: float = 1.0, atr_mult_sl: float = 1.5):
        self.config = config
        # Resolve relative paths
        self.model_path = Path(model_path).resolve()

        if not self.model_path.exists():
            raise ValueError(f"Model not found: {self.model_path}\n   Current dir: {Path.cwd()}")

        # Load model
        logger.info(f"📦 Loading model: {self.model_path.name}")
        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']

        logger.info(f"✅ Model loaded - {len(self.feature_names)} features")
        logger.info(f"   Target: {self.model_data.get('target_type', 'classification')}")
        logger.info(f"   Horizon: {self.model_data.get('target_horizon', 5)} bars")

        # Trading config
        self.initial_capital = config.get('initial_capital', 10000)
        self.risk_per_trade = config.get('risk_per_trade', 0.0075)  # 0.75%
        self.fees_pct = 0.0006  # 0.06%
        self.slippage_pct = 0.0001  # 0.01%
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl

        logger.info(f"   TP: {self.atr_mult_tp} ATR | SL: {self.atr_mult_sl} ATR")
        logger.info(f"   Theoretical R:R: {self.atr_mult_tp / self.atr_mult_sl:.2f}")

    def backtest(self, df: pd.DataFrame, min_confidence: float = 0.0) -> Dict:
        """
        Run backtest com modelo ML.

        Args:
            df: DataFrame com features
            min_confidence: Confiança mínima para operar (0.0 a 1.0)

        Returns:
            Dict com métricas
        """
        # Prepare features
        logger.info(f"🔧 Preparing features...")
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"⚠️  Missing {len(missing_features)} features, filling with 0")
            for feat in missing_features:
                df[feat] = 0

        X = df[self.feature_names].fillna(0)

        # Get predictions
        logger.info(f"🤖 Running model predictions...")
        predictions = self.model.predict(X)
        proba = self.model.predict_proba(X)

        # Convert predictions to signals
        # Model outputs: 0=SHORT, 1=NEUTRAL, 2=LONG
        df['ml_pred'] = predictions
        df['ml_prob_short'] = proba[:, 0]
        df['ml_prob_neutral'] = proba[:, 1]
        df['ml_prob_long'] = proba[:, 2]

        # Signal: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
        df['signal'] = 0
        df.loc[df['ml_pred'] == 2, 'signal'] = 1   # LONG
        df.loc[df['ml_pred'] == 0, 'signal'] = -1  # SHORT

        # Confidence = max probability
        df['ml_confidence'] = proba.max(axis=1)

        # Filter by confidence
        df.loc[df['ml_confidence'] < min_confidence, 'signal'] = 0

        logger.info(f"✅ Predictions done")
        logger.info(f"   LONG signals: {(df['signal'] == 1).sum():,}")
        logger.info(f"   SHORT signals: {(df['signal'] == -1).sum():,}")
        logger.info(f"   NEUTRAL: {(df['signal'] == 0).sum():,}")
        logger.info(f"   Avg confidence: {df['ml_confidence'].mean():.1%}")

        # Simulate trades
        logger.info(f"\n💼 Simulating trades...")
        trades = self._simulate_trades(df)

        if not trades:
            logger.warning("⚠️  No trades generated")
            return {'error': 'No trades', 'total_trades': 0}

        logger.info(f"✅ Simulated {len(trades)} trades")

        # Calculate metrics
        metrics = self._calculate_metrics(trades, df, min_confidence)

        return metrics

    def _simulate_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Simula execução de trades.

        Aplica:
        - Fees e slippage
        - SL/TP baseados em ATR
        - Cooldown entre trades
        """
        trades = []
        position = None
        capital = self.initial_capital
        cooldown = 0

        for i in range(len(df)):
            current = df.iloc[i]

            # Cooldown
            if cooldown > 0:
                cooldown -= 1

            # Check exit
            if position:
                exit_reason = self._check_exit(position, current, i)
                if exit_reason:
                    trade = self._close_trade(position, current, exit_reason)
                    trades.append(trade)
                    capital += trade['pnl_net']
                    position = None
                    cooldown = 4  # 1 hora cooldown (4 bars de 15min)

            # Check entry
            if not position and current['signal'] != 0 and cooldown == 0:
                # Não operar nos últimos 20 candles (evitar lookahead)
                if i < len(df) - 20:
                    position = self._open_trade(current, capital, i)

        # Close final position
        if position:
            trade = self._close_trade(position, df.iloc[-1], 'end_of_data')
            trades.append(trade)

        return trades

    def _open_trade(self, current, capital, idx):
        """Abre posição com gestão de risco."""
        direction = 'long' if current['signal'] == 1 else 'short'
        price = current['close']
        atr = current.get('atr', price * 0.01)

        # Aplica slippage na entrada
        if direction == 'long':
            entry_price = price * (1 + self.slippage_pct)
            sl = entry_price - (atr * self.atr_mult_sl)
            tp = entry_price + (atr * self.atr_mult_tp)
        else:
            entry_price = price * (1 - self.slippage_pct)
            sl = entry_price + (atr * self.atr_mult_sl)
            tp = entry_price - (atr * self.atr_mult_tp)

        # Calcula tamanho baseado em risco
        sl_dist = abs((sl - entry_price) / entry_price)
        risk_amt = capital * self.risk_per_trade
        size = risk_amt / sl_dist if sl_dist > 0 else capital * 0.1
        size = min(size, capital * 0.95)  # Max 95% do capital

        # Fee de entrada
        entry_fee = size * self.fees_pct

        return {
            'entry_idx': idx,
            'entry_time': current.name,
            'entry_price': entry_price,
            'direction': direction,
            'size': size,
            'stop_loss': sl,
            'take_profit': tp,
            'atr': atr,
            'ml_confidence': current['ml_confidence'],
            'entry_fee': entry_fee
        }

    def _check_exit(self, position, current, idx):
        """Verifica se posição deve ser fechada."""
        high = current['high']
        low = current['low']
        direction = position['direction']

        if direction == 'long':
            if low <= position['stop_loss']:
                return 'stop_loss'
            if high >= position['take_profit']:
                return 'take_profit'
        else:
            if high >= position['stop_loss']:
                return 'stop_loss'
            if low <= position['take_profit']:
                return 'take_profit'

        # Time exit (48h = 192 bars de 15min)
        if idx - position['entry_idx'] > 192:
            return 'time_exit'

        return None

    def _close_trade(self, position, current, reason):
        """Fecha posição e calcula PnL."""
        if reason == 'stop_loss':
            exit_price = position['stop_loss']
        elif reason == 'take_profit':
            exit_price = position['take_profit']
        else:
            exit_price = current['close']

        # Aplica slippage na saída
        if position['direction'] == 'long':
            exit_price = exit_price * (1 - self.slippage_pct)
        else:
            exit_price = exit_price * (1 + self.slippage_pct)

        entry = position['entry_price']
        direction = position['direction']

        # Calcula PnL bruto
        if direction == 'long':
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100

        pnl_gross = position['size'] * (pnl_pct / 100)

        # Aplica fees
        exit_fee = position['size'] * self.fees_pct
        total_fees = position['entry_fee'] + exit_fee
        pnl_net = pnl_gross - total_fees

        # Duration (convert timedelta to number of 15min bars)
        duration_timedelta = current.name - position['entry_time']
        duration_bars = duration_timedelta.total_seconds() / 900  # 900s = 15min

        return {
            'entry_time': position['entry_time'],
            'exit_time': current.name,
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': position['size'],
            'pnl_pct': pnl_pct,
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'fees': total_fees,
            'reason': reason,
            'ml_confidence': position['ml_confidence'],
            'duration_bars': duration_bars
        }

    def _calculate_metrics(self, trades, df, min_confidence):
        """Calcula métricas de performance."""
        if not trades:
            return {'error': 'No trades', 'total_trades': 0}

        df_trades = pd.DataFrame(trades)

        total = len(df_trades)
        winning = df_trades[df_trades['pnl_net'] > 0]
        losing = df_trades[df_trades['pnl_net'] <= 0]
        breakeven = df_trades[df_trades['pnl_net'] == 0]

        win_rate = len(winning) / total if total > 0 else 0

        total_pnl = df_trades['pnl_net'].sum()
        total_fees = df_trades['fees'].sum()

        final_capital = self.initial_capital + total_pnl
        roi = (total_pnl / self.initial_capital) * 100

        avg_win = winning['pnl_net'].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing['pnl_net'].mean()) if len(losing) > 0 else 0

        gross_profit = winning['pnl_net'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['pnl_net'].sum()) if len(losing) > 0 else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        returns = df_trades['pnl_pct'].values
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)
                 if len(returns) > 1 and np.std(returns) > 0 else 0)

        # Max drawdown
        equity = self.initial_capital + df_trades['pnl_net'].cumsum()
        peak = equity.expanding().max()
        dd = ((equity - peak) / peak * 100).min()

        # Direction stats
        longs = df_trades[df_trades['direction'] == 'long']
        shorts = df_trades[df_trades['direction'] == 'short']

        long_wr = (len(longs[longs['pnl_net'] > 0]) / len(longs) * 100) if len(longs) > 0 else 0
        short_wr = (len(shorts[shorts['pnl_net'] > 0]) / len(shorts) * 100) if len(shorts) > 0 else 0

        # Exit reasons
        exit_reasons = df_trades['reason'].value_counts().to_dict()

        # Duration - ensure it's numeric (not timedelta)
        avg_duration = df_trades['duration_bars'].mean()
        if hasattr(avg_duration, 'total_seconds'):
            avg_duration = avg_duration.total_seconds() / 900  # Convert to bars
        max_duration = df_trades['duration_bars'].max()
        if hasattr(max_duration, 'total_seconds'):
            max_duration = max_duration.total_seconds() / 900  # Convert to bars

        return {
            'min_confidence': min_confidence,
            'total_trades': total,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'breakeven_trades': len(breakeven),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_fees': total_fees,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': dd,
            'sharpe_ratio': sharpe,
            'avg_ml_confidence': df_trades['ml_confidence'].mean(),
            'long_trades': len(longs),
            'short_trades': len(shorts),
            'long_wr': long_wr,
            'short_wr': short_wr,
            'exit_reasons': exit_reasons,
            'avg_duration': avg_duration,
            'max_duration': max_duration
        }


def print_metrics(metrics: Dict):
    """Print métricas formatadas."""
    print("\n" + "="*70)
    print("📊 BACKTEST RESULTS")
    print("="*70)

    print(f"Total Trades:     {metrics['total_trades']}")
    print(f"Win Rate:         {metrics['win_rate']*100:.1f}%")
    print(f"  Winning:        {metrics['winning_trades']}")
    print(f"  Losing:         {metrics['losing_trades']}")
    print(f"  Breakeven:      {metrics['breakeven_trades']}")
    print()

    print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
    print(f"Final Equity:     ${metrics['final_capital']:,.2f}")
    print(f"Total PnL:        ${metrics['total_pnl']:+,.2f}")
    print(f"Total Return:     {metrics['roi']:+.2f}%")
    print(f"Total Fees:       ${metrics['total_fees']:,.2f}")
    print()

    print(f"Avg Trade:        ${metrics['total_pnl']/metrics['total_trades']:+,.2f}")
    print(f"Avg Win:          ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss:         ${metrics['avg_loss']:,.2f}")
    print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
    print()

    print(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print()

    print(f"Avg Duration:     {metrics['avg_duration']:.1f} bars ({metrics['avg_duration']*15:.0f} min)")
    print(f"Max Duration:     {metrics['max_duration']:.0f} bars ({metrics['max_duration']*15:.0f} min)")
    print()

    print(f"LONG trades:      {metrics['long_trades']} (WR: {metrics['long_wr']:.1f}%)")
    print(f"SHORT trades:     {metrics['short_trades']} (WR: {metrics['short_wr']:.1f}%)")
    print()

    print(f"Exit Reasons:")
    for reason, count in metrics['exit_reasons'].items():
        print(f"  {reason:20s}: {count}")

    print()
    print(f"Avg ML Confidence: {metrics['avg_ml_confidence']*100:.1f}%")
    print("="*70)


def main():
    global logger

    parser = argparse.ArgumentParser(description='Backtest modelo ML')
    parser.add_argument('--model', type=str, required=True, help='Caminho para o modelo .pkl')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol')
    parser.add_argument('--days', type=int, default=90, help='Dias de histórico')
    parser.add_argument('--confidence', type=float, default=0.0, help='Confiança mínima (0.0-1.0)')
    parser.add_argument('--tp', type=float, default=1.0, help='Take Profit em ATR (default: 1.0)')
    parser.add_argument('--sl', type=float, default=1.5, help='Stop Loss em ATR (default: 1.5)')

    args = parser.parse_args()

    # Setup logging
    setup_logging('INFO', log_to_file=False)
    logger = logging.getLogger("TradingBot.Backtest")

    logger.info("=" * 70)
    logger.info("🔬 ML MODEL BACKTEST")
    logger.info("=" * 70)
    logger.info(f"Model: {Path(args.model).name}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.days} days")
    logger.info(f"Min Confidence: {args.confidence:.0%}")
    logger.info("")

    # Load config
    config = load_config('standard')

    # Download data
    logger.info("📥 Downloading data from Bybit...")
    rest_client = BybitRESTClient(
        api_key=config.get('bybit_api_key', ''),
        api_secret=config.get('bybit_api_secret', ''),
        testnet=False
    )

    dm = DataManager(rest_client)
    df = dm.get_data(args.symbol, '15m', args.days, use_cache=True)

    if df.empty:
        logger.error("❌ No data downloaded")
        return

    logger.info(f"✅ Downloaded {len(df):,} candles")
    logger.info(f"   Period: {df.index[0]} to {df.index[-1]}")
    logger.info("")

    # Build features (same as training!)
    logger.info("🔨 Building features...")

    # 1. Base features
    fs = FeatureStore(config)
    df_features = fs.build_features(df, normalize=False)
    logger.info(f"✅ Base features: {len(df_features.columns)} columns")

    # 2. Legacy features
    df_features = create_legacy_features(df_features)
    logger.info(f"✅ Legacy features added")

    # 3. Advanced scalping features
    scalping_engineer = ScalpingFeatureEngineer()
    df_features = scalping_engineer.build_all_features(df_features)
    logger.info(f"✅ Scalping features: {len(df_features.columns)} columns")
    logger.info("")

    # Run backtest
    backtester = MLBacktester(args.model, config, atr_mult_tp=args.tp, atr_mult_sl=args.sl)
    metrics = backtester.backtest(df_features, min_confidence=args.confidence)

    if 'error' in metrics:
        logger.error(f"❌ {metrics['error']}")
        return

    # Print results
    print_metrics(metrics)

    # Analysis
    logger.info("")
    logger.info("=" * 70)
    logger.info("💡 ANALYSIS")
    logger.info("=" * 70)

    if metrics['roi'] > 0:
        logger.info("✅ Estratégia LUCRATIVA!")
    else:
        logger.info("❌ Estratégia com prejuízo")

    if metrics['win_rate'] > 0.55:
        logger.info(f"✅ Win rate excelente: {metrics['win_rate']*100:.1f}%")
    elif metrics['win_rate'] > 0.45:
        logger.info(f"⚠️  Win rate médio: {metrics['win_rate']*100:.1f}%")
    else:
        logger.info(f"❌ Win rate baixo: {metrics['win_rate']*100:.1f}%")

    if metrics['profit_factor'] > 1.5:
        logger.info(f"✅ Profit factor excelente: {metrics['profit_factor']:.2f}")
    elif metrics['profit_factor'] > 1.0:
        logger.info(f"⚠️  Profit factor médio: {metrics['profit_factor']:.2f}")
    else:
        logger.info(f"❌ Profit factor ruim: {metrics['profit_factor']:.2f}")

    if metrics['sharpe_ratio'] > 2.0:
        logger.info(f"✅ Sharpe ratio excelente: {metrics['sharpe_ratio']:.2f}")
    elif metrics['sharpe_ratio'] > 1.0:
        logger.info(f"⚠️  Sharpe ratio médio: {metrics['sharpe_ratio']:.2f}")
    else:
        logger.info(f"❌ Sharpe ratio ruim: {metrics['sharpe_ratio']:.2f}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
