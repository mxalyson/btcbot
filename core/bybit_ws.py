"""
WebSocket client for Bybit v5 - FINAL WORKING VERSION.
Properly handles callbacks with correct data format.
"""

import json
import threading
import time
import logging
from typing import Dict, List, Optional, Callable
from websocket import WebSocketApp

logger = logging.getLogger("TradingBot.WS")


class BybitWebSocket:
    """Bybit v5 WebSocket client with orderbook reconstruction."""

    def __init__(self, testnet: bool = True):
        """
        Initialize WebSocket client.

        Args:
            testnet: Use testnet endpoint
        """
        if testnet:
            self.ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self.ws_url = "wss://stream.bybit.com/v5/public/linear"

        self.ws: Optional[WebSocketApp] = None
        self.subscriptions: List[str] = []
        self.callbacks: Dict[str, Callable] = {}
        self.generic_callbacks: Dict[str, List[Callable]] = {}

        # Orderbook state (snapshot + deltas)
        self.orderbooks: Dict[str, Dict] = {}

        # Trades buffer
        self.trades_buffer: Dict[str, List] = {}

        self.is_connected = False
        self.thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        """Connect to WebSocket."""
        self.ws = WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )

        # Run in background thread
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

        # Wait for connection
        time.sleep(2)

    def subscribe(self, topics: List[str], callback: Optional[Callable] = None) -> None:
        """
        Subscribe to topics.

        Args:
            topics: List of topics
            callback: Optional callback function
        """
        self.subscriptions.extend(topics)

        if callback:
            for topic in topics:
                self.callbacks[topic] = callback

        # Send subscription message
        if self.ws and self.is_connected:
            sub_msg = {
                "op": "subscribe",
                "args": topics
            }
            self.ws.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to: {topics}")

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a specific event type.

        Args:
            event_type: Type of event ('orderbook', 'trade', etc.)
            callback: Function to call - signature: callback(symbol, data)
        """
        if event_type not in self.generic_callbacks:
            self.generic_callbacks[event_type] = []

        self.generic_callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for {event_type}")

    def _trigger_callbacks(self, event_type: str, symbol: str, data: any) -> None:
        """Trigger all registered callbacks for an event type."""
        if event_type in self.generic_callbacks:
            for callback in self.generic_callbacks[event_type]:
                try:
                    # Pass symbol and data as separate arguments
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in callback for {event_type}: {e}")

    def _on_open(self, ws):
        """Handle connection open."""
        self.is_connected = True
        logger.info("WebSocket connected")

        # Subscribe to pending topics
        if self.subscriptions:
            sub_msg = {
                "op": "subscribe",
                "args": self.subscriptions
            }
            ws.send(json.dumps(sub_msg))

    def _on_message(self, ws, message):
        """Handle incoming messages."""
        try:
            data = json.loads(message)

            # Handle subscription confirmation
            if data.get('op') == 'subscribe':
                logger.info(f"Subscribed: {data.get('success', False)}")
                return

            # Handle pong
            if data.get('op') == 'pong':
                return

            # Handle data messages
            topic = data.get('topic', '')
            msg_type = data.get('type', '')

            if 'orderbook' in topic:
                symbol = topic.split('.')[-1]
                self._handle_orderbook(data)
                # Trigger callbacks with symbol and orderbook data
                orderbook = self.orderbooks.get(symbol)
                if orderbook:
                    self._trigger_callbacks('orderbook', symbol, orderbook)

            elif 'publicTrade' in topic:
                symbol = topic.split('.')[-1]
                self._handle_trades(data)
                # Trigger callbacks with symbol and recent trades
                recent_trades = self.get_trades(symbol, 10)
                if recent_trades:
                    self._trigger_callbacks('trade', symbol, recent_trades)

            # Call topic-specific callback
            if topic in self.callbacks:
                self.callbacks[topic](data)

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_orderbook(self, data: Dict) -> None:
        """Handle orderbook updates (snapshot + delta). Converts all values to float."""
        try:
            topic = data.get('topic', '')
            symbol = topic.split('.')[-1]
            msg_type = data.get('type', '')

            orderbook_data = data.get('data', {})

            if msg_type == 'snapshot':
                # Initial snapshot
                self.orderbooks[symbol] = {
                    'bids': {},
                    'asks': {},
                    'timestamp': orderbook_data.get('ts', 0)
                }

                # Convert to float
                for bid in orderbook_data.get('b', []):
                    price = float(bid[0])
                    size = float(bid[1])
                    self.orderbooks[symbol]['bids'][price] = size

                for ask in orderbook_data.get('a', []):
                    price = float(ask[0])
                    size = float(ask[1])
                    self.orderbooks[symbol]['asks'][price] = size

                logger.debug(f"Orderbook snapshot for {symbol}: "
                           f"{len(self.orderbooks[symbol]['bids'])} bids, "
                           f"{len(self.orderbooks[symbol]['asks'])} asks")

            elif msg_type == 'delta':
                # Delta update
                if symbol not in self.orderbooks:
                    return

                # Update bids
                for bid in orderbook_data.get('b', []):
                    price = float(bid[0])
                    size = float(bid[1])

                    if size == 0:
                        self.orderbooks[symbol]['bids'].pop(price, None)
                    else:
                        self.orderbooks[symbol]['bids'][price] = size

                # Update asks
                for ask in orderbook_data.get('a', []):
                    price = float(ask[0])
                    size = float(ask[1])

                    if size == 0:
                        self.orderbooks[symbol]['asks'].pop(price, None)
                    else:
                        self.orderbooks[symbol]['asks'][price] = size

                # Update timestamp
                self.orderbooks[symbol]['timestamp'] = orderbook_data.get('ts', 0)

        except Exception as e:
            logger.error(f"Error handling orderbook: {e}")

    def _handle_trades(self, data: Dict) -> None:
        """Handle public trades."""
        try:
            topic = data.get('topic', '')
            symbol = topic.split('.')[-1]

            trades_data = data.get('data', [])

            if symbol not in self.trades_buffer:
                self.trades_buffer[symbol] = []

            # Add trades to buffer
            for trade in trades_data:
                self.trades_buffer[symbol].append({
                    'timestamp': int(trade.get('T', 0)),
                    'side': trade.get('S', ''),
                    'size': float(trade.get('v', 0)),
                    'price': float(trade.get('p', 0))
                })

            # Keep only recent trades (last 100)
            self.trades_buffer[symbol] = self.trades_buffer[symbol][-100:]

        except Exception as e:
            logger.error(f"Error handling trades: {e}")

    def get_orderbook(self, symbol: str) -> Optional[Dict]:
        """Get current orderbook snapshot."""
        return self.orderbooks.get(symbol)

    def get_trades(self, symbol: str, n: int = 50) -> List[Dict]:
        """Get recent trades."""
        if symbol not in self.trades_buffer:
            return []

        return self.trades_buffer[symbol][-n:]

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.is_connected = False
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")

    def close(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            logger.info("WebSocket closed")

    def send_ping(self) -> None:
        """Send ping to keep connection alive."""
        if self.ws and self.is_connected:
            ping_msg = {"op": "ping"}
            self.ws.send(json.dumps(ping_msg))
