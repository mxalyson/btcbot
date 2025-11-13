"""
Fill Simulator - Realistic order execution simulation for paper trading.
Handles slippage, latency, fees, partial fills, and trailing stops.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("TradingBot.Simulator")


class FillSimulator:
    """Simulate realistic order fills for paper trading."""

    def __init__(self, config: dict):
        """
        Initialize fill simulator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tick_size = 0.01  # Default, should be fetched per symbol
        self.lot_size = 0.001  # Default, should be fetched per symbol

    def set_instrument_specs(self, tick_size: float, lot_size: float) -> None:
        """
        Set instrument specifications.

        Args:
            tick_size: Minimum price increment
            lot_size: Minimum quantity increment
        """
        self.tick_size = tick_size
        self.lot_size = lot_size

    def calculate_slippage(self, order_type: str, side: str) -> float:
        """
        Calculate slippage in ticks.

        Args:
            order_type: 'market' or 'limit'
            side: 'buy' or 'sell'

        Returns:
            Slippage in ticks
        """
        if order_type == 'market':
            # Market orders have higher slippage
            base_slippage = self.config['slippage_ticks']
            # Add random component (0-50% extra)
            random_factor = np.random.uniform(0, 0.5)
            total_slippage = base_slippage * (1 + random_factor)
        else:
            # Limit orders may get price improvement or no fill
            total_slippage = 0

        return total_slippage

    def simulate_latency(self) -> float:
        """
        Simulate network latency.

        Returns:
            Latency in milliseconds
        """
        base_latency = self.config.get('simulated_latency_ms', 50)
        # Add random jitter (±50%)
        jitter = np.random.uniform(-0.5, 0.5)
        actual_latency = base_latency * (1 + jitter)
        return max(1, actual_latency)

    def check_fill_price(self, 
                        candle: Dict,
                        order_price: float,
                        side: str,
                        order_type: str) -> Tuple[bool, Optional[float]]:
        """
        Check if order would be filled and at what price.

        Args:
            candle: Dict with OHLC data
            order_price: Order price
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'

        Returns:
            Tuple of (filled, fill_price)
        """
        high = candle['high']
        low = candle['low']
        open_price = candle['open']
        close = candle['close']

        if order_type == 'market':
            # Market orders always fill
            # Use open price with slippage
            slippage_ticks = self.calculate_slippage('market', side)
            slippage_amount = slippage_ticks * self.tick_size

            if side == 'buy':
                fill_price = open_price + slippage_amount
                # Cap at high
                fill_price = min(fill_price, high)
            else:  # sell
                fill_price = open_price - slippage_amount
                # Cap at low
                fill_price = max(fill_price, low)

            return True, fill_price

        else:  # limit order
            if side == 'buy':
                # Buy limit fills if low <= order_price
                if low <= order_price:
                    # Fill at limit price or better
                    fill_price = min(order_price, open_price)
                    return True, fill_price
            else:  # sell
                # Sell limit fills if high >= order_price
                if high >= order_price:
                    # Fill at limit price or better
                    fill_price = max(order_price, open_price)
                    return True, fill_price

        return False, None

    def simulate_entry(self, 
                      candle: Dict,
                      order_price: float,
                      side: str,
                      order_type: str = 'market',
                      quantity: float = 0.0) -> Optional[Dict]:
        """
        Simulate order entry.

        Args:
            candle: Current candle data
            order_price: Order price
            side: 'buy' or 'sell'
            order_type: 'market' or 'limit'
            quantity: Order quantity

        Returns:
            Fill dict or None if not filled
        """
        # Simulate latency
        latency_ms = self.simulate_latency()

        # Check if filled
        filled, fill_price = self.check_fill_price(candle, order_price, side, order_type)

        if not filled:
            return None

        # Calculate fees
        fee_rate = self.config['fees_taker'] if order_type == 'market' else self.config['fees_maker']
        fees = fill_price * quantity * fee_rate

        fill = {
            'timestamp': candle.get('timestamp', datetime.now()),
            'side': side,
            'order_type': order_type,
            'order_price': order_price,
            'fill_price': fill_price,
            'quantity': quantity,
            'fees': fees,
            'latency_ms': latency_ms,
            'slippage': fill_price - order_price if side == 'buy' else order_price - fill_price
        }

        logger.debug(f"Entry filled: {side} @ {fill_price:.2f} (slippage: {fill['slippage']:.2f})")

        return fill

    def check_stop_loss(self, 
                       candle: Dict,
                       entry_price: float,
                       stop_loss: float,
                       side: str,
                       mark_price: Optional[float] = None) -> Tuple[bool, Optional[float]]:
        """
        Check if stop loss was hit.

        Args:
            candle: Current candle data
            entry_price: Entry price
            stop_loss: Stop loss price
            side: 'long' or 'short'
            mark_price: Mark price (for realistic liquidation)

        Returns:
            Tuple of (hit, exit_price)
        """
        high = candle['high']
        low = candle['low']

        # Use mark price if available for more realistic stops
        price_to_check = mark_price if mark_price else candle['close']

        if side == 'long':
            # Long stop loss triggered if price drops to SL
            if low <= stop_loss:
                # Assume worst case: stop triggered at SL with slippage
                slippage_ticks = self.config['slippage_ticks']
                slippage_amount = slippage_ticks * self.tick_size
                exit_price = stop_loss - slippage_amount
                # But not below the low
                exit_price = max(exit_price, low)
                return True, exit_price

        else:  # short
            # Short stop loss triggered if price rises to SL
            if high >= stop_loss:
                # Assume worst case: stop triggered at SL with slippage
                slippage_ticks = self.config['slippage_ticks']
                slippage_amount = slippage_ticks * self.tick_size
                exit_price = stop_loss + slippage_amount
                # But not above the high
                exit_price = min(exit_price, high)
                return True, exit_price

        return False, None

    def check_take_profit(self,
                         candle: Dict,
                         take_profit: float,
                         side: str) -> Tuple[bool, Optional[float]]:
        """
        Check if take profit was hit.

        Args:
            candle: Current candle data
            take_profit: Take profit price
            side: 'long' or 'short'

        Returns:
            Tuple of (hit, exit_price)
        """
        high = candle['high']
        low = candle['low']

        if side == 'long':
            # Long TP triggered if price reaches TP
            if high >= take_profit:
                # Optimistic: assume TP fills at TP price
                return True, take_profit

        else:  # short
            # Short TP triggered if price drops to TP
            if low <= take_profit:
                # Optimistic: assume TP fills at TP price
                return True, take_profit

        return False, None

    def simulate_trailing_stop(self,
                               candle: Dict,
                               entry_price: float,
                               current_trail_price: float,
                               trail_amount: float,
                               side: str) -> Tuple[float, bool, Optional[float]]:
        """
        Simulate trailing stop.

        Args:
            candle: Current candle data
            entry_price: Entry price
            current_trail_price: Current trailing stop price
            trail_amount: Trailing distance
            side: 'long' or 'short'

        Returns:
            Tuple of (new_trail_price, hit, exit_price)
        """
        high = candle['high']
        low = candle['low']
        close = candle['close']

        if side == 'long':
            # Update trailing stop if price moves up
            potential_trail = high - trail_amount
            new_trail_price = max(current_trail_price, potential_trail)

            # Check if trailing stop was hit
            if low <= new_trail_price:
                # Exit at trailing stop
                exit_price = new_trail_price
                return new_trail_price, True, exit_price

            return new_trail_price, False, None

        else:  # short
            # Update trailing stop if price moves down
            potential_trail = low + trail_amount
            new_trail_price = min(current_trail_price, potential_trail)

            # Check if trailing stop was hit
            if high >= new_trail_price:
                # Exit at trailing stop
                exit_price = new_trail_price
                return new_trail_price, True, exit_price

            return new_trail_price, False, None

    def simulate_partial_fill(self,
                             total_quantity: float,
                             fill_probability: float = 1.0) -> float:
        """
        Simulate partial fill (for limit orders).

        Args:
            total_quantity: Total order quantity
            fill_probability: Probability of full fill

        Returns:
            Filled quantity
        """
        if np.random.random() < fill_probability:
            # Full fill
            return total_quantity
        else:
            # Partial fill: 30-70% of order
            fill_pct = np.random.uniform(0.3, 0.7)
            return total_quantity * fill_pct

    def simulate_position_lifecycle(self,
                                   df: pd.DataFrame,
                                   entry_idx: int,
                                   entry_price: float,
                                   stop_loss: float,
                                   take_profits: List[float],
                                   tp_splits: List[float],
                                   position_size: float,
                                   side: str,
                                   use_trailing: bool = False,
                                   trailing_amount: Optional[float] = None) -> Dict:
        """
        Simulate complete position lifecycle.

        Args:
            df: DataFrame with OHLCV data
            entry_idx: Entry candle index
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profits: List of TP prices [TP1, TP2, TP3]
            tp_splits: % allocation for each TP [50, 30, 20]
            position_size: Total position size
            side: 'long' or 'short'
            use_trailing: Enable trailing stop
            trailing_amount: Trailing distance

        Returns:
            Dict with trade results
        """
        remaining_size = position_size
        realized_pnl = 0
        exit_prices = []
        exit_reasons = []
        current_trail_price = stop_loss

        # Entry fees
        entry_fee = entry_price * position_size * self.config['fees_taker']

        # Convert splits to quantities
        tp_quantities = [position_size * (pct / 100) for pct in tp_splits]

        # Track TP hits
        tps_hit = [False] * len(take_profits)

        # Iterate through candles after entry
        for i in range(entry_idx + 1, len(df)):
            candle = df.iloc[i].to_dict()

            if remaining_size == 0:
                break

            # Check stop loss first
            sl_hit, sl_exit_price = self.check_stop_loss(
                candle, entry_price, 
                current_trail_price if use_trailing else stop_loss,
                side
            )

            if sl_hit:
                # Full exit at SL
                if side == 'long':
                    pnl = (sl_exit_price - entry_price) * remaining_size
                else:
                    pnl = (entry_price - sl_exit_price) * remaining_size

                exit_fee = sl_exit_price * remaining_size * self.config['fees_taker']
                pnl -= exit_fee

                realized_pnl += pnl
                exit_prices.append(sl_exit_price)
                exit_reasons.append('stop_loss')
                remaining_size = 0
                break

            # Check take profits
            for tp_idx, tp_price in enumerate(take_profits):
                if tps_hit[tp_idx] or remaining_size == 0:
                    continue

                tp_hit, tp_exit_price = self.check_take_profit(candle, tp_price, side)

                if tp_hit:
                    # Partial exit at TP
                    exit_size = min(tp_quantities[tp_idx], remaining_size)

                    if side == 'long':
                        pnl = (tp_exit_price - entry_price) * exit_size
                    else:
                        pnl = (entry_price - tp_exit_price) * exit_size

                    exit_fee = tp_exit_price * exit_size * self.config['fees_taker']
                    pnl -= exit_fee

                    realized_pnl += pnl
                    exit_prices.append(tp_exit_price)
                    exit_reasons.append(f'tp{tp_idx + 1}')
                    remaining_size -= exit_size
                    tps_hit[tp_idx] = True

                    logger.debug(f"TP{tp_idx + 1} hit @ {tp_exit_price:.2f}")

                    # Move to breakeven after first TP
                    if tp_idx == 0 and remaining_size > 0:
                        stop_loss = entry_price
                        current_trail_price = entry_price
                        logger.debug("Moved to breakeven")

            # Update trailing stop
            if use_trailing and trailing_amount and remaining_size > 0:
                new_trail, trail_hit, trail_exit = self.simulate_trailing_stop(
                    candle, entry_price, current_trail_price, trailing_amount, side
                )

                current_trail_price = new_trail

                if trail_hit:
                    # Exit remaining at trailing stop
                    if side == 'long':
                        pnl = (trail_exit - entry_price) * remaining_size
                    else:
                        pnl = (entry_price - trail_exit) * remaining_size

                    exit_fee = trail_exit * remaining_size * self.config['fees_taker']
                    pnl -= exit_fee

                    realized_pnl += pnl
                    exit_prices.append(trail_exit)
                    exit_reasons.append('trailing_stop')
                    remaining_size = 0
                    break

        # Calculate final PnL
        total_pnl = realized_pnl - entry_fee

        return {
            'entry_price': entry_price,
            'exit_prices': exit_prices,
            'exit_reasons': exit_reasons,
            'position_size': position_size,
            'gross_pnl': realized_pnl,
            'entry_fee': entry_fee,
            'net_pnl': total_pnl,
            'return_pct': (total_pnl / (entry_price * position_size)) * 100,
            'tps_hit': sum(tps_hit),
            'remaining_size': remaining_size
        }
