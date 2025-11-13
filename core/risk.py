"""
Risk Management - Position sizing, circuit breakers, exposure limits.
Protects capital through multiple layers of risk control.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("TradingBot.Risk")


class RiskManager:
    """Comprehensive risk management system."""

    def __init__(self, config: dict):
        """
        Initialize risk manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.equity = config['initial_capital']
        self.peak_equity = config['initial_capital']

        # Trade tracking
        self.open_positions: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Circuit breaker state
        self.is_halted = False
        self.halt_reason = None
        self.halt_timestamp = None

        # Daily stats
        self.trades_today = 0
        self.last_trade_time = None

    def update_equity(self, new_equity: float) -> None:
        """
        Update current equity.

        Args:
            new_equity: New equity value
        """
        self.equity = new_equity

        # Update peak for drawdown calculation
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            logger.info(f"New equity peak: ${self.peak_equity:.2f}")

    def calculate_position_size(self, 
                               entry_price: float,
                               stop_loss: float,
                               symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Calculate position size based on risk percentage.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            symbol: Trading symbol

        Returns:
            Dict with position sizing details
        """
        # Risk amount in USDT
        risk_amount = self.equity * self.config['risk_per_trade']

        # Risk per unit (difference between entry and SL)
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            logger.error("Stop loss equals entry price")
            return {'position_size': 0, 'risk_amount': 0, 'error': 'Invalid SL'}

        # Position size in base currency
        position_size = risk_amount / risk_per_unit

        # Position value in USDT
        position_value = position_size * entry_price

        # Check max position value
        if position_value > self.config.get('max_order_value_usdt', 10000):
            logger.warning(f"Position value ${position_value:.2f} exceeds max")
            # Scale down
            max_value = self.config['max_order_value_usdt']
            position_size = max_value / entry_price
            position_value = max_value

        return {
            'position_size': position_size,
            'position_value': position_value,
            'risk_amount': risk_amount,
            'risk_per_unit': risk_per_unit,
            'risk_reward_ratio': None  # To be filled by caller
        }

    def validate_order(self, 
                      symbol: str,
                      direction: str,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: Optional[float] = None,
                      position_size: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate order before execution.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check if trading is halted
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # Check cooldown
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if time_since_last < self.config['cooldown_min']:
                return False, f"Cooldown active ({time_since_last:.1f}/{self.config['cooldown_min']} min)"

        # Check max trades per day
        if self.trades_today >= self.config['max_trades_per_day']:
            return False, f"Max daily trades reached ({self.trades_today}/{self.config['max_trades_per_day']})"

        # Check max positions
        if len(self.open_positions) >= self.config['max_positions']:
            return False, f"Max positions reached ({len(self.open_positions)}/{self.config['max_positions']})"

        # Check if already in position for this symbol
        if symbol in self.open_positions:
            return False, f"Already in position for {symbol}"

        # Validate prices
        if entry_price <= 0 or stop_loss <= 0:
            return False, "Invalid prices (must be > 0)"

        # Validate SL direction
        if direction == 'long' and stop_loss >= entry_price:
            return False, "Long SL must be below entry"

        if direction == 'short' and stop_loss <= entry_price:
            return False, "Short SL must be above entry"

        # Validate TP direction
        if take_profit:
            if direction == 'long' and take_profit <= entry_price:
                return False, "Long TP must be above entry"
            if direction == 'short' and take_profit >= entry_price:
                return False, "Short TP must be below entry"

        # Check price deviation from mark price (prevents fat finger)
        max_deviation = self.config.get('max_price_deviation_pct', 2.0)
        # Note: In real implementation, fetch mark price from exchange
        # For now, assume entry_price is reasonable

        # Validate position size
        if position_size:
            min_size = 0.001  # Adjust per symbol
            if position_size < min_size:
                return False, f"Position size too small (min {min_size})"

        return True, "OK"

    def check_circuit_breakers(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any circuit breaker should be triggered.

        Returns:
            Tuple of (should_halt, reason)
        """
        # Check daily drawdown
        today = datetime.now().date().isoformat()
        daily_pnl = self.daily_pnl.get(today, 0)
        daily_loss_pct = (daily_pnl / self.peak_equity) * 100

        if daily_loss_pct < -self.config['circuit_breaker_max_loss_pct']:
            return True, f"Daily loss limit reached ({daily_loss_pct:.2f}%)"

        # Check consecutive losses
        if self.consecutive_losses >= self.config['circuit_breaker_consec_losses']:
            return True, f"Max consecutive losses ({self.consecutive_losses})"

        # Check total drawdown
        current_dd = self.get_current_drawdown()
        max_dd_limit = 20.0  # 20% max drawdown
        if current_dd > max_dd_limit:
            return True, f"Max drawdown exceeded ({current_dd:.2f}%)"

        return False, None

    def trigger_circuit_breaker(self, reason: str) -> None:
        """
        Trigger circuit breaker - halt all trading.

        Args:
            reason: Reason for halt
        """
        self.is_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.now()

        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"All trading halted at {self.halt_timestamp}")

        # In production: send alert, close positions, etc.

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual override)."""
        self.is_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        logger.warning("Circuit breaker manually reset")

    def record_trade(self, trade: Dict) -> None:
        """
        Record trade result and update stats.

        Args:
            trade: Trade dictionary with pnl, symbol, etc.
        """
        self.trade_history.append(trade)

        # Update equity
        pnl = trade.get('pnl', 0)
        self.equity += pnl

        # Update daily PnL
        today = datetime.now().date().isoformat()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += pnl

        # Update consecutive wins/losses
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            logger.info(f"✅ Win streak: {self.consecutive_wins}")
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            logger.warning(f"❌ Loss streak: {self.consecutive_losses}")

        # Update trade count
        self.trades_today += 1
        self.last_trade_time = datetime.now()

        # Remove from open positions
        symbol = trade.get('symbol')
        if symbol in self.open_positions:
            del self.open_positions[symbol]

        # Check circuit breakers after trade
        should_halt, reason = self.check_circuit_breakers()
        if should_halt:
            self.trigger_circuit_breaker(reason)

    def open_position(self, symbol: str, position_info: Dict) -> None:
        """
        Register new open position.

        Args:
            symbol: Trading symbol
            position_info: Position details
        """
        self.open_positions[symbol] = {
            **position_info,
            'open_time': datetime.now(),
            'unrealized_pnl': 0
        }

        logger.info(f"Position opened: {symbol} {position_info['direction']}")

    def update_position(self, symbol: str, current_price: float) -> None:
        """
        Update position with current price.

        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        entry_price = pos['entry_price']
        position_size = pos['position_size']
        direction = pos['direction']

        # Calculate unrealized PnL
        if direction == 'long':
            pnl = (current_price - entry_price) * position_size
        else:  # short
            pnl = (entry_price - current_price) * position_size

        pos['unrealized_pnl'] = pnl
        pos['current_price'] = current_price

    def close_position(self, symbol: str, exit_price: float, 
                      exit_reason: str = "manual") -> Dict:
        """
        Close position and record trade.

        Args:
            symbol: Trading symbol
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            Trade result dict
        """
        if symbol not in self.open_positions:
            logger.error(f"No open position for {symbol}")
            return {}

        pos = self.open_positions[symbol]
        entry_price = pos['entry_price']
        position_size = pos['position_size']
        direction = pos['direction']

        # Calculate realized PnL
        if direction == 'long':
            gross_pnl = (exit_price - entry_price) * position_size
        else:  # short
            gross_pnl = (entry_price - exit_price) * position_size

        # Subtract fees
        entry_fee = entry_price * position_size * self.config['fees_taker']
        exit_fee = exit_price * position_size * self.config['fees_taker']
        net_pnl = gross_pnl - entry_fee - exit_fee

        # Calculate returns
        position_value = entry_price * position_size
        return_pct = (net_pnl / position_value) * 100

        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'gross_pnl': gross_pnl,
            'fees': entry_fee + exit_fee,
            'net_pnl': net_pnl,
            'return_pct': return_pct,
            'exit_reason': exit_reason,
            'open_time': pos['open_time'],
            'close_time': datetime.now(),
            'duration': (datetime.now() - pos['open_time']).total_seconds() / 60
        }

        # Record trade
        self.record_trade(trade)

        logger.info(f"Position closed: {symbol} | PnL: ${net_pnl:.2f} ({return_pct:.2f}%)")

        return trade

    def get_current_drawdown(self) -> float:
        """
        Calculate current drawdown from peak.

        Returns:
            Drawdown percentage
        """
        if self.peak_equity == 0:
            return 0.0

        drawdown = ((self.peak_equity - self.equity) / self.peak_equity) * 100
        return drawdown

    def get_exposure(self) -> Dict[str, float]:
        """
        Calculate current exposure.

        Returns:
            Dict with exposure metrics
        """
        total_exposure = 0
        long_exposure = 0
        short_exposure = 0

        for symbol, pos in self.open_positions.items():
            position_value = pos['position_size'] * pos.get('current_price', pos['entry_price'])
            total_exposure += position_value

            if pos['direction'] == 'long':
                long_exposure += position_value
            else:
                short_exposure += position_value

        exposure_pct = (total_exposure / self.equity) * 100 if self.equity > 0 else 0

        return {
            'total_exposure': total_exposure,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'exposure_pct': exposure_pct
        }

    def get_risk_stats(self) -> Dict:
        """
        Get comprehensive risk statistics.

        Returns:
            Dict with risk metrics
        """
        if not self.trade_history:
            return {'error': 'No trades yet'}

        df_trades = pd.DataFrame(self.trade_history)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        losing_trades = len(df_trades[df_trades['net_pnl'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        avg_win = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['net_pnl'] < 0]['net_pnl'].mean() if losing_trades > 0 else 0

        total_pnl = df_trades['net_pnl'].sum()
        total_fees = df_trades['fees'].sum()

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        current_dd = self.get_current_drawdown()

        exposure = self.get_exposure()

        return {
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': current_dd,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'total_pnl': total_pnl,
            'total_fees': total_fees,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'trades_today': self.trades_today,
            'open_positions': len(self.open_positions),
            **exposure,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason
        }

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of new day)."""
        self.trades_today = 0
        logger.info("Daily stats reset")

    def calculate_kelly_criterion(self) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing.

        Returns:
            Kelly percentage (0-1)
        """
        if not self.trade_history or len(self.trade_history) < 10:
            return self.config['risk_per_trade']  # Use default

        df_trades = pd.DataFrame(self.trade_history)

        wins = df_trades[df_trades['net_pnl'] > 0]['net_pnl']
        losses = df_trades[df_trades['net_pnl'] < 0]['net_pnl'].abs()

        if len(wins) == 0 or len(losses) == 0:
            return self.config['risk_per_trade']

        win_rate = len(wins) / len(df_trades)
        avg_win = wins.mean()
        avg_loss = losses.mean()

        if avg_loss == 0:
            return self.config['risk_per_trade']

        # Kelly formula: W - (1-W)/R
        # W = win rate, R = win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Use fractional Kelly (25% of full Kelly for safety)
        kelly_fraction = max(0, min(kelly * 0.25, 0.05))  # Cap at 5%

        logger.info(f"Kelly Criterion: {kelly:.3f}, Using: {kelly_fraction:.3f}")

        return kelly_fraction
