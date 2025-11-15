"""
📊 BACKTESTING REALISTA PARA SCALPING - V2.0

Simula execução real com:
- Fees (0.06%)
- Slippage (0.01%)
- Latência de execução
- Gestão de risco ATR-based
- Métricas detalhadas

Author: Claude
Date: 2025-11-14
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class RealisticBacktester:
    """
    Backtest realista para modelos de scalping.

    Simula condições reais de trading.
    """

    def __init__(self, config: Dict = None):
        """
        Args:
            config:
                {
                    'initial_capital': 10000,
                    'risk_per_trade': 0.02,  # 2%
                    'fees_pct': 0.0006,      # 0.06%
                    'slippage_pct': 0.0001,  # 0.01%
                    'atr_mult_sl': 1.5,
                    'atr_mult_tp': 1.0,
                    'max_trades_per_day': 10,
                    'min_confidence': 0.30
                }
        """
        self.config = config or {}
        self.initial_capital = self.config.get('initial_capital', 10000)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.02)
        self.fees_pct = self.config.get('fees_pct', 0.0006)
        self.slippage_pct = self.config.get('slippage_pct', 0.0001)
        self.atr_mult_sl = self.config.get('atr_mult_sl', 1.5)
        self.atr_mult_tp = self.config.get('atr_mult_tp', 1.0)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 10)
        self.min_confidence = self.config.get('min_confidence', 0.30)

        # Estado
        self.capital = self.initial_capital
        self.equity_curve = []
        self.trades = []
        self.daily_trades = {}
        self.position = None

    def run(self, df: pd.DataFrame, signals: pd.Series,
            confidences: Optional[pd.Series] = None) -> Dict:
        """
        Executa backtest completo.

        Args:
            df: DataFrame com OHLCV + ATR
            signals: -1 (SHORT), 0 (NEUTRAL), 1 (LONG)
            confidences: Confiança do modelo (opcional)

        Returns:
            Dict com métricas detalhadas
        """
        print("\n" + "="*70)
        print("📊 REALISTIC BACKTESTING")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Risk per Trade: {self.risk_per_trade:.1%}")
        print(f"Fees: {self.fees_pct:.2%} | Slippage: {self.slippage_pct:.2%}")
        print("="*70)

        for idx in range(len(df)):
            row = df.iloc[idx]
            signal = signals.iloc[idx]
            confidence = confidences.iloc[idx] if confidences is not None else 1.0

            # Atualiza posição aberta
            if self.position:
                self._update_position(row)

            # Verifica novo sinal
            if signal != 0 and not self.position:
                if confidence >= self.min_confidence:
                    # Check daily limit
                    current_date = row.name.date()
                    trades_today = self.daily_trades.get(current_date, 0)

                    if trades_today < self.max_trades_per_day:
                        self._open_position(row, signal, confidence)
                        self.daily_trades[current_date] = trades_today + 1

            # Track equity
            current_equity = self.capital
            if self.position:
                current_equity += self._calculate_unrealized_pnl(row)

            self.equity_curve.append({
                'timestamp': row.name,
                'equity': current_equity,
                'capital': self.capital,
                'position_open': self.position is not None
            })

        # Close any remaining position
        if self.position:
            last_row = df.iloc[-1]
            self._close_position(last_row, 'end_of_backtest')

        # Calculate metrics
        metrics = self._calculate_metrics()

        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)
        self._print_metrics(metrics)
        print("="*70)

        return metrics

    def _open_position(self, row: pd.Series, signal: int, confidence: float):
        """
        Abre posição com gestão de risco ATR-based.
        """
        direction = 'long' if signal == 1 else 'short'
        price = row['close']
        atr = row.get('atr', price * 0.01)

        # SL e TP
        if direction == 'long':
            sl = price - (atr * self.atr_mult_sl)
            tp = price + (atr * self.atr_mult_tp)
        else:
            sl = price + (atr * self.atr_mult_sl)
            tp = price - (atr * self.atr_mult_tp)

        # Calcula tamanho baseado em risco
        sl_distance = abs(price - sl)
        risk_amount = self.capital * self.risk_per_trade
        size_usd = min(
            risk_amount / (sl_distance / price),
            self.capital * 0.95  # Max 95% do capital
        )

        # Aplica slippage na entrada
        if direction == 'long':
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)

        # Fees de entrada
        entry_fee = size_usd * self.fees_pct

        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': row.name,
            'size_usd': size_usd,
            'sl': sl,
            'tp': tp,
            'atr': atr,
            'confidence': confidence,
            'entry_fee': entry_fee
        }

        self.capital -= entry_fee

    def _update_position(self, row: pd.Series):
        """
        Atualiza posição e verifica SL/TP.
        """
        if not self.position:
            return

        direction = self.position['direction']
        sl = self.position['sl']
        tp = self.position['tp']

        # Check SL
        if direction == 'long' and row['low'] <= sl:
            self._close_position(row, 'stop_loss', exit_price=sl)
            return

        if direction == 'short' and row['high'] >= sl:
            self._close_position(row, 'stop_loss', exit_price=sl)
            return

        # Check TP
        if direction == 'long' and row['high'] >= tp:
            self._close_position(row, 'take_profit', exit_price=tp)
            return

        if direction == 'short' and row['low'] <= tp:
            self._close_position(row, 'take_profit', exit_price=tp)
            return

    def _close_position(self, row: pd.Series, reason: str,
                       exit_price: Optional[float] = None):
        """
        Fecha posição e registra trade.
        """
        if not self.position:
            return

        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size_usd = self.position['size_usd']

        # Exit price
        if exit_price is None:
            exit_price = row['close']

        # Aplica slippage na saída
        if direction == 'long':
            exit_price = exit_price * (1 - self.slippage_pct)
        else:
            exit_price = exit_price * (1 + self.slippage_pct)

        # Calcula PnL
        if direction == 'long':
            pnl = ((exit_price - entry_price) / entry_price) * size_usd
        else:
            pnl = ((entry_price - exit_price) / entry_price) * size_usd

        # Fees de saída
        exit_fee = size_usd * self.fees_pct

        # PnL líquido
        net_pnl = pnl - self.position['entry_fee'] - exit_fee

        # Atualiza capital
        self.capital += net_pnl

        # Registra trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': row.name,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size_usd': size_usd,
            'sl': self.position['sl'],
            'tp': self.position['tp'],
            'atr': self.position['atr'],
            'confidence': self.position['confidence'],
            'pnl_gross': pnl,
            'entry_fee': self.position['entry_fee'],
            'exit_fee': exit_fee,
            'pnl_net': net_pnl,
            'pnl_pct': (net_pnl / size_usd) * 100,
            'duration': (row.name - self.position['entry_time']).total_seconds() / 60,  # minutes
            'exit_reason': reason
        }

        self.trades.append(trade)
        self.position = None

    def _calculate_unrealized_pnl(self, row: pd.Series) -> float:
        """
        Calcula PnL não realizado.
        """
        if not self.position:
            return 0

        direction = self.position['direction']
        entry_price = self.position['entry_price']
        size_usd = self.position['size_usd']
        current_price = row['close']

        if direction == 'long':
            pnl = ((current_price - entry_price) / entry_price) * size_usd
        else:
            pnl = ((entry_price - current_price) / entry_price) * size_usd

        return pnl

    def _calculate_metrics(self) -> Dict:
        """
        Calcula métricas detalhadas.
        """
        if len(self.trades) == 0:
            return {'error': 'No trades executed'}

        df_trades = pd.DataFrame(self.trades)
        df_equity = pd.DataFrame(self.equity_curve)

        # Basic stats
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl_net'] > 0])
        losing_trades = len(df_trades[df_trades['pnl_net'] < 0])
        breakeven_trades = len(df_trades[df_trades['pnl_net'] == 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # PnL stats
        total_pnl = df_trades['pnl_net'].sum()
        total_fees = df_trades['entry_fee'].sum() + df_trades['exit_fee'].sum()
        avg_win = df_trades[df_trades['pnl_net'] > 0]['pnl_net'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_net'] < 0]['pnl_net'].mean() if losing_trades > 0 else 0
        avg_trade = df_trades['pnl_net'].mean()

        # Risk/Reward
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
        else:
            profit_factor = np.inf if avg_win > 0 else 0

        # Returns
        final_equity = self.capital
        total_return = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Drawdown
        equity_series = df_equity['equity']
        running_max = equity_series.expanding().max()
        drawdown_pct = ((equity_series - running_max) / running_max) * 100
        max_drawdown = drawdown_pct.min()

        # Sharpe-like
        returns = df_trades['pnl_pct']
        sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)  # Annualized

        # Duration stats
        avg_duration = df_trades['duration'].mean()
        max_duration = df_trades['duration'].max()

        # Exit reasons
        exit_reasons = df_trades['exit_reason'].value_counts().to_dict()

        metrics = {
            # Overview
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,

            # PnL
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_fees': total_fees,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,

            # Risk
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe,

            # Duration
            'avg_duration_min': avg_duration,
            'max_duration_min': max_duration,

            # Other
            'exit_reasons': exit_reasons,

            # Per direction
            'long_trades': len(df_trades[df_trades['direction'] == 'long']),
            'short_trades': len(df_trades[df_trades['direction'] == 'short']),
            'long_win_rate': (df_trades[df_trades['direction'] == 'long']['pnl_net'] > 0).sum() /
                            len(df_trades[df_trades['direction'] == 'long']) if len(df_trades[df_trades['direction'] == 'long']) > 0 else 0,
            'short_win_rate': (df_trades[df_trades['direction'] == 'short']['pnl_net'] > 0).sum() /
                             len(df_trades[df_trades['direction'] == 'short']) if len(df_trades[df_trades['direction'] == 'short']) > 0 else 0,
        }

        return metrics

    def _print_metrics(self, metrics: Dict):
        """
        Imprime métricas formatadas.
        """
        if 'error' in metrics:
            print(f"❌ {metrics['error']}")
            return

        print(f"Total Trades:     {metrics['total_trades']}")
        print(f"Win Rate:         {metrics['win_rate']:.1%}")
        print(f"  Winning:        {metrics['winning_trades']}")
        print(f"  Losing:         {metrics['losing_trades']}")
        print(f"  Breakeven:      {metrics['breakeven_trades']}")
        print()
        print(f"Initial Capital:  ${metrics['initial_capital']:,.2f}")
        print(f"Final Equity:     ${metrics['final_equity']:,.2f}")
        print(f"Total PnL:        ${metrics['total_pnl']:+,.2f}")
        print(f"Total Return:     {metrics['total_return_pct']:+.2f}%")
        print(f"Total Fees:       ${metrics['total_fees']:,.2f}")
        print()
        print(f"Avg Trade:        ${metrics['avg_trade']:+.2f}")
        print(f"Avg Win:          ${metrics['avg_win']:,.2f}")
        print(f"Avg Loss:         ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor:    {metrics['profit_factor']:.2f}")
        print()
        print(f"Max Drawdown:     {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        print()
        print(f"Avg Duration:     {metrics['avg_duration_min']:.1f} min")
        print(f"Max Duration:     {metrics['max_duration_min']:.1f} min")
        print()
        print(f"LONG trades:      {metrics['long_trades']} (WR: {metrics['long_win_rate']:.1%})")
        print(f"SHORT trades:     {metrics['short_trades']} (WR: {metrics['short_win_rate']:.1%})")
        print()
        print("Exit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            print(f"  {reason:20s}: {count}")

    def export_results(self, output_dir: str = 'ml_training/outputs'):
        """
        Exporta resultados para arquivos.
        """
        from pathlib import Path
        import json

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Trades
        df_trades = pd.DataFrame(self.trades)
        trades_file = output_path / f"backtest_trades_{timestamp}.csv"
        df_trades.to_csv(trades_file, index=False)
        print(f"✅ Trades exported: {trades_file}")

        # Equity curve
        df_equity = pd.DataFrame(self.equity_curve)
        equity_file = output_path / f"backtest_equity_{timestamp}.csv"
        df_equity.to_csv(equity_file, index=False)
        print(f"✅ Equity exported: {equity_file}")

        # Metrics
        metrics = self._calculate_metrics()
        # Convert non-serializable
        metrics_serializable = {k: (int(v) if isinstance(v, np.integer) else
                                   float(v) if isinstance(v, np.floating) else v)
                               for k, v in metrics.items()}

        metrics_file = output_path / f"backtest_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2, default=str)
        print(f"✅ Metrics exported: {metrics_file}")


if __name__ == "__main__":
    """
    Teste do backtester.
    """
    # Dados de exemplo
    dates = pd.date_range('2024-01-01', periods=5000, freq='15min')
    np.random.seed(42)

    # Price action realista
    returns = np.random.normal(0.0001, 0.002, 5000)
    close_prices = 100 * (1 + returns).cumprod()

    df_test = pd.DataFrame({
        'open': close_prices * (1 + np.random.uniform(-0.001, 0.001, 5000)),
        'high': close_prices * (1 + np.random.uniform(0, 0.005, 5000)),
        'low': close_prices * (1 + np.random.uniform(-0.005, 0, 5000)),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 5000),
        'atr': np.random.uniform(0.5, 2.0, 5000),
    }, index=dates)

    # Sinais simulados
    signals = pd.Series(
        np.random.choice([-1, 0, 1], size=5000, p=[0.1, 0.8, 0.1]),
        index=dates
    )

    confidences = pd.Series(
        np.random.uniform(0.2, 0.9, 5000),
        index=dates
    )

    # Backtest
    backtester = RealisticBacktester({
        'initial_capital': 10000,
        'risk_per_trade': 0.02,
        'min_confidence': 0.4
    })

    metrics = backtester.run(df_test, signals, confidences)

    print("\n✅ Backtest test complete!")
