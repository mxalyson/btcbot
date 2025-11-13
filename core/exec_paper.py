"""
Paper Trading Executor - Virtual execution using realistic simulator.
Maintains same interface as live executor for seamless switching.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging
from core.simulator import FillSimulator
from core.risk import RiskManager

logger = logging.getLogger("TradingBot.PaperExec")


class PaperExecutor:
    """Execute trades in paper/simulation mode."""

    def __init__(self, config: dict, rest_client):
        """
        Initialize paper executor.

        Args:
            config: Configuration dictionary
            rest_client: BybitRESTClient (for fetching market data)
        """
        self.config = config
        self.rest_client = rest_client
        self.simulator = FillSimulator(config)
        self.risk_manager = RiskManager(config)

        # Virtual positions
        self.positions: Dict[str, Dict] = {}
        self.pending_orders: Dict[str, Dict] = {}
        self.order_id_counter = 1000

        # Performance tracking
        self.filled_orders: List[Dict] = []
        self.closed_trades: List[Dict] = []

        logger.info("Paper executor initialized")

    def generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"PAPER_{self.order_id_counter}"
        self.order_id_counter += 1
        return order_id

    def place_order(self,
                   symbol: str,
                   side: str,
                   order_type: str,
                   quantity: float,
                   price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   reduce_only: bool = False) -> Dict:
        """
        Place paper order.

        Args:
            symbol: Trading symbol
            side: 'Buy' or 'Sell'
            order_type: 'Market' or 'Limit'
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            reduce_only: Reduce only flag

        Returns:
            Order result dict
        """
        order_id = self.generate_order_id()

        # Get current market price
        current_price = self._get_current_price(symbol)

        if current_price is None:
            return {
                'success': False,
                'order_id': order_id,
                'error': 'Failed to get market price'
            }

        # Determine entry price
        if order_type == 'Market':
            entry_price = current_price
        else:
            entry_price = price if price else current_price

        # Calculate position sizing
        direction = 'long' if side == 'Buy' else 'short'

        if not reduce_only and stop_loss:
            sizing = self.risk_manager.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                symbol=symbol
            )

            if sizing['position_size'] == 0:
                return {
                    'success': False,
                    'order_id': order_id,
                    'error': 'Position size calculation failed'
                }

            # Use calculated size if not specified
            if quantity == 0:
                quantity = sizing['position_size']

        # Validate order
        is_valid, reason = self.risk_manager.validate_order(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss if stop_loss else entry_price * 0.98,
            take_profit=take_profit,
            position_size=quantity
        )

        if not is_valid:
            logger.warning(f"Order validation failed: {reason}")
            return {
                'success': False,
                'order_id': order_id,
                'error': reason
            }

        # Create order
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'direction': direction,
            'order_type': order_type,
            'quantity': quantity,
            'price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reduce_only': reduce_only,
            'status': 'pending',
            'created_at': datetime.now()
        }

        # For market orders, execute immediately
        if order_type == 'Market':
            fill_result = self._execute_market_order(order, current_price)
            return fill_result
        else:
            # Store pending limit order
            self.pending_orders[order_id] = order
            logger.info(f"Limit order placed: {order_id} {symbol} {side} @ {entry_price:.2f}")
            return {
                'success': True,
                'order_id': order_id,
                'status': 'pending'
            }

    def _execute_market_order(self, order: Dict, current_price: float) -> Dict:
        """Execute market order immediately."""
        symbol = order['symbol']
        side_str = order['side'].lower()

        # Simulate fill
        candle = self._get_current_candle(symbol)

        fill = self.simulator.simulate_entry(
            candle=candle,
            order_price=current_price,
            side='buy' if side_str == 'buy' else 'sell',
            order_type='market',
            quantity=order['quantity']
        )

        if fill:
            # Create position
            self._create_position(order, fill)

            order['status'] = 'filled'
            order['fill_price'] = fill['fill_price']
            order['fill_time'] = datetime.now()
            self.filled_orders.append(order)

            logger.info(f"✅ Market order filled: {symbol} {order['side']} @ {fill['fill_price']:.2f}")

            return {
                'success': True,
                'order_id': order['order_id'],
                'status': 'filled',
                'fill_price': fill['fill_price'],
                'quantity': order['quantity']
            }
        else:
            return {
                'success': False,
                'order_id': order['order_id'],
                'error': 'Simulation failed'
            }

    def _create_position(self, order: Dict, fill: Dict) -> None:
        """Create position from filled order."""
        symbol = order['symbol']

        position = {
            'symbol': symbol,
            'direction': order['direction'],
            'entry_price': fill['fill_price'],
            'position_size': order['quantity'],
            'stop_loss': order['stop_loss'],
            'take_profit': order['take_profit'],
            'entry_time': datetime.now(),
            'entry_fees': fill['fees'],
            'unrealized_pnl': 0,
            'trailing_stop': None
        }

        # Register with risk manager
        self.risk_manager.open_position(symbol, position)

        # Store position
        self.positions[symbol] = position

        logger.info(f"Position opened: {symbol} {order['direction']} {order['quantity']} @ {fill['fill_price']:.2f}")

    def update_positions(self, symbol: str, current_price: float) -> None:
        """
        Update position with current price and check exits.

        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Update unrealized PnL
        if pos['direction'] == 'long':
            pnl = (current_price - pos['entry_price']) * pos['position_size']
        else:
            pnl = (pos['entry_price'] - current_price) * pos['position_size']

        pos['unrealized_pnl'] = pnl - pos['entry_fees']

        # Update risk manager
        self.risk_manager.update_position(symbol, current_price)

        # Check stop loss
        if self._check_stop_hit(pos, current_price):
            self._close_position(symbol, current_price, 'stop_loss')
            return

        # Check take profit
        if self._check_tp_hit(pos, current_price):
            self._close_position(symbol, current_price, 'take_profit')
            return

        # Update trailing stop if enabled
        if self.config['use_trailing'] and pos.get('trailing_stop'):
            self._update_trailing_stop(symbol, current_price)

    def _check_stop_hit(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss was hit."""
        if not position.get('stop_loss'):
            return False

        if position['direction'] == 'long':
            return current_price <= position['stop_loss']
        else:
            return current_price >= position['stop_loss']

    def _check_tp_hit(self, position: Dict, current_price: float) -> bool:
        """Check if take profit was hit."""
        if not position.get('take_profit'):
            return False

        if position['direction'] == 'long':
            return current_price >= position['take_profit']
        else:
            return current_price <= position['take_profit']

    def _update_trailing_stop(self, symbol: str, current_price: float) -> None:
        """Update trailing stop."""
        pos = self.positions[symbol]

        if 'atr' not in pos:
            return

        trail_distance = pos['atr'] * self.config['atr_mult_sl']

        if pos['direction'] == 'long':
            new_stop = current_price - trail_distance
            if pos['trailing_stop'] is None or new_stop > pos['trailing_stop']:
                pos['trailing_stop'] = new_stop
                logger.debug(f"Trailing stop updated: {symbol} @ {new_stop:.2f}")
        else:
            new_stop = current_price + trail_distance
            if pos['trailing_stop'] is None or new_stop < pos['trailing_stop']:
                pos['trailing_stop'] = new_stop
                logger.debug(f"Trailing stop updated: {symbol} @ {new_stop:.2f}")

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Close position."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Calculate PnL
        if pos['direction'] == 'long':
            gross_pnl = (exit_price - pos['entry_price']) * pos['position_size']
        else:
            gross_pnl = (pos['entry_price'] - exit_price) * pos['position_size']

        # Fees
        exit_fee = exit_price * pos['position_size'] * self.config['fees_taker']
        net_pnl = gross_pnl - pos['entry_fees'] - exit_fee

        # Create trade record
        trade = {
            'symbol': symbol,
            'direction': pos['direction'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'position_size': pos['position_size'],
            'gross_pnl': gross_pnl,
            'fees': pos['entry_fees'] + exit_fee,
            'net_pnl': net_pnl,
            'return_pct': (net_pnl / (pos['entry_price'] * pos['position_size'])) * 100,
            'exit_reason': reason,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(),
            'duration_min': (datetime.now() - pos['entry_time']).total_seconds() / 60
        }

        # Record with risk manager
        self.risk_manager.close_position(symbol, exit_price, reason)

        # Store trade
        self.closed_trades.append(trade)

        # Remove position
        del self.positions[symbol]

        result = "✅" if net_pnl > 0 else "❌"
        logger.info(f"{result} Position closed: {symbol} | PnL: ${net_pnl:.2f} | Reason: {reason}")

    def close_position_manual(self, symbol: str) -> bool:
        """
        Manually close position.

        Args:
            symbol: Trading symbol

        Returns:
            Success boolean
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return False

        current_price = self._get_current_price(symbol)
        if current_price:
            self._close_position(symbol, current_price, 'manual')
            return True

        return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel pending order.

        Args:
            order_id: Order ID

        Returns:
            Success boolean
        """
        if order_id in self.pending_orders:
            del self.pending_orders[order_id]
            logger.info(f"Order cancelled: {order_id}")
            return True

        logger.warning(f"Order not found: {order_id}")
        return False

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get specific position."""
        return self.positions.get(symbol)

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'equity': self.config['initial_capital']
            }

        df_trades = pd.DataFrame(self.closed_trades)

        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = df_trades['net_pnl'].sum()

        equity = self.config['initial_capital'] + total_pnl

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'equity': equity,
            'open_positions': len(self.positions),
            'risk_stats': self.risk_manager.get_risk_stats()
        }

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Fetch current market price."""
        try:
            ticker = self.rest_client.get_tickers(symbol)
            if ticker.get('retCode') == 0:
                price = float(ticker['result']['list'][0]['lastPrice'])
                return price
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")

        return None

    def _get_current_candle(self, symbol: str) -> Dict:
        """Get current candle data."""
        try:
            kline = self.rest_client.get_kline(symbol, '1m', limit=1)
            if kline.get('retCode') == 0:
                candle_data = kline['result']['list'][0]
                return {
                    'open': float(candle_data[1]),
                    'high': float(candle_data[2]),
                    'low': float(candle_data[3]),
                    'close': float(candle_data[4]),
                    'volume': float(candle_data[5]),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            logger.error(f"Error fetching candle for {symbol}: {e}")

        # Fallback to current price
        price = self._get_current_price(symbol)
        if price:
            return {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': 0,
                'timestamp': datetime.now()
            }

        return {}
