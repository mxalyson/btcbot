"""
Live Trading Executor - Real order execution on Bybit v5.
Includes guard-rails, retries, validation, and safety checks.
"""

import time
import pandas as pd
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging
from core.risk import RiskManager
from core.utils import round_to_tick, round_to_lot

logger = logging.getLogger("TradingBot.LiveExec")


class LiveExecutor:
    """Execute real trades on Bybit."""

    def __init__(self, config: dict, rest_client):
        """
        Initialize live executor.

        Args:
            config: Configuration dictionary
            rest_client: BybitRESTClient instance
        """
        self.config = config
        self.rest_client = rest_client
        self.risk_manager = RiskManager(config)

        # Instrument specifications cache
        self.instrument_specs: Dict[str, Dict] = {}

        # Order tracking
        self.active_orders: Dict[str, Dict] = {}
        self.positions: Dict[str, Dict] = {}

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests

        logger.warning("⚠️  LIVE EXECUTOR INITIALIZED - REAL MONEY MODE")

        # Load instrument specs
        self._load_instrument_specs()

    def _load_instrument_specs(self) -> None:
        """Load instrument specifications from exchange."""
        logger.info("Loading instrument specifications...")

        for symbol in self.config['symbols']:
            try:
                response = self.rest_client.get_instruments_info(
                    category='linear',
                    symbol=symbol
                )

                if response.get('retCode') == 0:
                    instrument = response['result']['list'][0]

                    # Extract price and lot size filters
                    price_filter = instrument['priceFilter']
                    lot_filter = instrument['lotSizeFilter']

                    self.instrument_specs[symbol] = {
                        'tick_size': float(price_filter['tickSize']),
                        'min_price': float(price_filter['minPrice']),
                        'max_price': float(price_filter['maxPrice']),
                        'lot_size': float(lot_filter['qtyStep']),
                        'min_qty': float(lot_filter['minOrderQty']),
                        'max_qty': float(lot_filter['maxOrderQty']),
                        'min_notional': float(lot_filter.get('minNotionalValue', 0))
                    }

                    logger.info(f"Loaded specs for {symbol}: tick={self.instrument_specs[symbol]['tick_size']}")

                time.sleep(0.1)  # Rate limit

            except Exception as e:
                logger.error(f"Failed to load specs for {symbol}: {e}")

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _validate_price_qty(self, symbol: str, price: float, qty: float) -> Tuple[float, float]:
        """
        Validate and round price/quantity to exchange specs.

        Args:
            symbol: Trading symbol
            price: Order price
            qty: Order quantity

        Returns:
            Tuple of (valid_price, valid_qty)
        """
        if symbol not in self.instrument_specs:
            logger.error(f"No specs for {symbol}")
            return price, qty

        specs = self.instrument_specs[symbol]

        # Round price to tick size
        valid_price = round_to_tick(price, specs['tick_size'])

        # Clamp to min/max
        valid_price = max(specs['min_price'], min(valid_price, specs['max_price']))

        # Round quantity to lot size
        valid_qty = round_to_lot(qty, specs['lot_size'])

        # Clamp to min/max
        valid_qty = max(specs['min_qty'], min(valid_qty, specs['max_qty']))

        return valid_price, valid_qty

    def _confirm_order(self, order: Dict) -> bool:
        """
        Request user confirmation for order.

        Args:
            order: Order dictionary

        Returns:
            True if confirmed
        """
        if not self.config.get('live_confirm_each_order', True):
            return True

        print()
        print("=" * 70)
        print("🚨 LIVE ORDER CONFIRMATION REQUIRED")
        print("=" * 70)
        print(f"Symbol:       {order['symbol']}")
        print(f"Side:         {order['side']}")
        print(f"Type:         {order['order_type']}")
        print(f"Quantity:     {order['quantity']}")
        print(f"Price:        {order.get('price', 'Market')}")
        print(f"Stop Loss:    {order.get('stop_loss', 'None')}")
        print(f"Take Profit:  {order.get('take_profit', 'None')}")
        print(f"Value (USDT): {order.get('notional_value', 0):.2f}")
        print("=" * 70)

        response = input("Confirm order? (type 'YES' to proceed): ")

        return response == 'YES'

    def place_order(self,
                   symbol: str,
                   side: str,
                   order_type: str,
                   quantity: float,
                   price: Optional[float] = None,
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None,
                   order_link_id: Optional[str] = None,
                   max_retries: int = 3) -> Dict:
        """
        Place live order on Bybit.

        Args:
            symbol: Trading symbol
            side: 'Buy' or 'Sell'
            order_type: 'Market' or 'Limit'
            quantity: Order quantity
            price: Limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            order_link_id: Custom order ID
            max_retries: Maximum retry attempts

        Returns:
            Order result dict
        """
        # Validate with risk manager
        direction = 'long' if side == 'Buy' else 'short'
        entry_price = price if price else self._get_mark_price(symbol)

        if not entry_price:
            return {'success': False, 'error': 'Failed to get mark price'}

        is_valid, reason = self.risk_manager.validate_order(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss if stop_loss else entry_price * 0.95,
            take_profit=take_profit,
            position_size=quantity
        )

        if not is_valid:
            logger.error(f"Order validation failed: {reason}")
            return {'success': False, 'error': reason}

        # Validate price/quantity with exchange specs
        if price:
            price, quantity = self._validate_price_qty(symbol, price, quantity)
        else:
            _, quantity = self._validate_price_qty(symbol, entry_price, quantity)

        # Check max order value
        notional_value = entry_price * quantity
        max_value = self.config.get('max_order_value_usdt', 10000)

        if notional_value > max_value:
            logger.error(f"Order value ${notional_value:.2f} exceeds max ${max_value:.2f}")
            return {'success': False, 'error': 'Order value too large'}

        # Build order dict
        order = {
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'notional_value': notional_value,
            'order_link_id': order_link_id or f"BOT_{int(time.time())}"
        }

        # Request confirmation
        if not self._confirm_order(order):
            logger.warning("Order cancelled by user")
            return {'success': False, 'error': 'User cancelled'}

        # Execute with retries
        for attempt in range(max_retries):
            try:
                self._rate_limit()

                logger.info(f"Placing order (attempt {attempt + 1}/{max_retries})...")

                response = self.rest_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    qty=quantity,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_link_id=order['order_link_id']
                )

                if response.get('retCode') == 0:
                    result = response['result']
                    order_id = result['orderId']

                    # Store active order
                    self.active_orders[order_id] = {
                        **order,
                        'order_id': order_id,
                        'status': 'active',
                        'created_at': datetime.now()
                    }

                    logger.info(f"✅ Order placed successfully: {order_id}")

                    # Register with risk manager
                    self.risk_manager.open_position(symbol, {
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'position_size': quantity,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit
                    })

                    return {
                        'success': True,
                        'order_id': order_id,
                        'order_link_id': order['order_link_id'],
                        'status': 'active'
                    }

                else:
                    error_msg = response.get('retMsg', 'Unknown error')
                    logger.error(f"Order failed: {error_msg}")

                    # Check if retryable
                    if 'rate limit' in error_msg.lower():
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable error
                        return {'success': False, 'error': error_msg}

            except Exception as e:
                logger.error(f"Exception placing order: {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {'success': False, 'error': str(e)}

        return {'success': False, 'error': 'Max retries exceeded'}

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            Success boolean
        """
        try:
            self._rate_limit()

            response = self.rest_client.cancel_order(
                symbol=symbol,
                order_id=order_id
            )

            if response.get('retCode') == 0:
                logger.info(f"Order cancelled: {order_id}")

                # Remove from active orders
                if order_id in self.active_orders:
                    del self.active_orders[order_id]

                return True
            else:
                logger.error(f"Cancel failed: {response.get('retMsg')}")
                return False

        except Exception as e:
            logger.error(f"Exception cancelling order: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders from exchange.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        try:
            self._rate_limit()

            response = self.rest_client.get_open_orders(symbol)

            if response.get('retCode') == 0:
                orders = response['result']['list']
                return orders
            else:
                logger.error(f"Failed to get orders: {response.get('retMsg')}")
                return []

        except Exception as e:
            logger.error(f"Exception getting orders: {e}")
            return []

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open positions from exchange.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of positions
        """
        try:
            self._rate_limit()

            response = self.rest_client.get_positions(symbol)

            if response.get('retCode') == 0:
                positions = response['result']['list']

                # Update local positions cache
                for pos in positions:
                    if float(pos['size']) > 0:
                        symbol = pos['symbol']
                        self.positions[symbol] = pos

                return positions
            else:
                logger.error(f"Failed to get positions: {response.get('retMsg')}")
                return []

        except Exception as e:
            logger.error(f"Exception getting positions: {e}")
            return []

    def close_position(self, symbol: str) -> bool:
        """
        Close position at market.

        Args:
            symbol: Trading symbol

        Returns:
            Success boolean
        """
        # Get current position
        positions = self.get_positions(symbol)

        if not positions:
            logger.warning(f"No position found for {symbol}")
            return False

        pos = positions[0]
        size = float(pos['size'])

        if size == 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return False

        # Determine close side
        side = 'Sell' if pos['side'] == 'Buy' else 'Buy'

        logger.info(f"Closing position: {symbol} size={size}")

        # Place market order to close
        result = self.place_order(
            symbol=symbol,
            side=side,
            order_type='Market',
            quantity=size,
            order_link_id=f"CLOSE_{symbol}_{int(time.time())}"
        )

        return result.get('success', False)

    def _get_mark_price(self, symbol: str) -> Optional[float]:
        """Get current mark price."""
        try:
            ticker = self.rest_client.get_tickers(symbol)
            if ticker.get('retCode') == 0:
                mark_price = float(ticker['result']['list'][0]['markPrice'])
                return mark_price
        except Exception as e:
            logger.error(f"Error getting mark price: {e}")

        return None

    def get_account_balance(self) -> Optional[float]:
        """
        Get account balance.

        Returns:
            Available balance in USDT
        """
        try:
            self._rate_limit()

            response = self.rest_client.get_wallet_balance(account_type='UNIFIED')

            if response.get('retCode') == 0:
                accounts = response['result']['list']
                if accounts:
                    # Find USDT coin
                    for coin in accounts[0]['coin']:
                        if coin['coin'] == 'USDT':
                            available = float(coin['availableToWithdraw'])
                            return available

        except Exception as e:
            logger.error(f"Error getting balance: {e}")

        return None

    def sync_positions(self) -> None:
        """Sync positions with exchange."""
        logger.info("Syncing positions with exchange...")

        positions = self.get_positions()

        # Clear local cache
        self.positions.clear()

        # Update from exchange
        for pos in positions:
            size = float(pos['size'])
            if size > 0:
                symbol = pos['symbol']
                self.positions[symbol] = pos
                logger.info(f"Position synced: {symbol} size={size}")

    def emergency_close_all(self) -> None:
        """
        Emergency: close all positions.
        Use with caution!
        """
        logger.critical("🚨 EMERGENCY CLOSE ALL POSITIONS")

        confirm = input("Type 'EMERGENCY' to confirm: ")

        if confirm != 'EMERGENCY':
            logger.info("Emergency close cancelled")
            return

        positions = self.get_positions()

        for pos in positions:
            size = float(pos['size'])
            if size > 0:
                symbol = pos['symbol']
                logger.warning(f"Emergency closing {symbol}")
                self.close_position(symbol)
                time.sleep(0.5)

        logger.critical("All positions closed")
