"""
BYBIT WEBSOCKET CLIENT
Cliente WebSocket para receber dados em tempo real da Bybit
"""

import json
import hmac
import hashlib
import time
import threading
from typing import Callable, Optional, Dict, Any
import websocket
import logging

logger = logging.getLogger(__name__)


class BybitWebSocket:
    """Cliente WebSocket para Bybit V5 API"""
    
    # URLs
    MAINNET_PUBLIC = "wss://stream.bybit.com/v5/public/linear"
    MAINNET_PRIVATE = "wss://stream.bybit.com/v5/private"
    TESTNET_PUBLIC = "wss://stream-testnet.bybit.com/v5/public/linear"
    TESTNET_PRIVATE = "wss://stream-testnet.bybit.com/v5/private"
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        testnet: bool = True,
        channel_type: str = "public"
    ):
        """
        Inicializa o cliente WebSocket
        
        Args:
            api_key: API key (necessária para canais privados)
            api_secret: API secret (necessária para canais privados)
            testnet: Se True, usa testnet
            channel_type: 'public' ou 'private'
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.channel_type = channel_type
        
        # WebSocket URL
        if channel_type == "public":
            self.ws_url = self.TESTNET_PUBLIC if testnet else self.MAINNET_PUBLIC
        else:
            self.ws_url = self.TESTNET_PRIVATE if testnet else self.MAINNET_PRIVATE
        
        # State
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        self.subscriptions = {}
        self.callbacks = {}
        
        # Heartbeat
        self.last_ping = 0
        self.ping_interval = 20  # segundos
        
        logger.info(f"🔌 WebSocket initialized: {self.ws_url}")
    
    def _generate_signature(self, expires: int) -> str:
        """Gera assinatura para autenticação"""
        message = f"GET/realtime{expires}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _authenticate(self):
        """Autentica no WebSocket privado"""
        if self.channel_type != "private":
            return
        
        if not self.api_key or not self.api_secret:
            logger.error("❌ API credentials required for private channels")
            return
        
        expires = int((time.time() + 1) * 1000)
        signature = self._generate_signature(expires)
        
        auth_message = {
            "op": "auth",
            "args": [self.api_key, expires, signature]
        }
        
        self.ws.send(json.dumps(auth_message))
        logger.info("🔐 Authentication sent")
    
    def _on_message(self, ws, message):
        """Callback quando recebe mensagem"""
        try:
            data = json.loads(message)
            
            # Pong response
            if data.get('ret_msg') == 'pong':
                return
            
            # Auth response
            if data.get('op') == 'auth':
                if data.get('success'):
                    logger.info("✅ Authenticated successfully")
                else:
                    logger.error(f"❌ Authentication failed: {data}")
                return
            
            # Subscription response
            if data.get('op') == 'subscribe':
                if data.get('success'):
                    logger.info(f"✅ Subscribed: {data.get('req_id')}")
                else:
                    logger.error(f"❌ Subscription failed: {data}")
                return
            
            # Data message
            topic = data.get('topic', '')
            
            # Chamar callbacks registrados
            for pattern, callback in self.callbacks.items():
                if pattern in topic:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"❌ Callback error for {topic}: {e}")
        
        except json.JSONDecodeError:
            logger.error(f"❌ Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"❌ Message handling error: {e}")
    
    def _on_error(self, ws, error):
        """Callback quando ocorre erro"""
        logger.error(f"❌ WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback quando conexão fecha"""
        self.is_connected = False
        logger.warning(f"⚠️ WebSocket closed: {close_status_code} - {close_msg}")
    
    def _on_open(self, ws):
        """Callback quando conexão abre"""
        self.is_connected = True
        logger.info("✅ WebSocket connected")
        
        # Autenticar se for canal privado
        if self.channel_type == "private":
            self._authenticate()
        
        # Re-subscribe aos canais anteriores
        for topic in self.subscriptions.values():
            self._subscribe(topic)
    
    def _subscribe(self, topics: list):
        """Envia mensagem de subscription"""
        if not self.is_connected:
            logger.warning("⚠️ Not connected, cannot subscribe")
            return
        
        message = {
            "op": "subscribe",
            "args": topics
        }
        
        self.ws.send(json.dumps(message))
        logger.info(f"📡 Subscribing to: {topics}")
    
    def _ping_loop(self):
        """Loop para enviar pings periódicos"""
        while self.is_connected:
            try:
                now = time.time()
                if now - self.last_ping >= self.ping_interval:
                    if self.ws:
                        self.ws.send(json.dumps({"op": "ping"}))
                        self.last_ping = now
                time.sleep(1)
            except Exception as e:
                logger.error(f"❌ Ping error: {e}")
                break
    
    def connect(self):
        """Conecta ao WebSocket"""
        if self.is_connected:
            logger.warning("⚠️ Already connected")
            return
        
        logger.info(f"🔌 Connecting to {self.ws_url}...")
        
        # Criar WebSocket
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # Iniciar thread
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            daemon=True
        )
        self.ws_thread.start()
        
        # Aguardar conexão
        timeout = 10
        start = time.time()
        while not self.is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        
        if not self.is_connected:
            raise ConnectionError("❌ Failed to connect to WebSocket")
        
        # Iniciar ping loop
        ping_thread = threading.Thread(target=self._ping_loop, daemon=True)
        ping_thread.start()
        
        logger.info("✅ WebSocket ready")
    
    def disconnect(self):
        """Desconecta do WebSocket"""
        if self.ws:
            self.is_connected = False
            self.ws.close()
            logger.info("🔌 WebSocket disconnected")
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable):
        """
        Subscribe to kline/candle updates
        
        Args:
            symbol: Symbol (ex: 'BTCUSDT')
            interval: Interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            callback: Function to call when data arrives
        """
        topic = f"kline.{interval}.{symbol}"
        self.subscriptions[f"kline_{symbol}_{interval}"] = [topic]
        self.callbacks[topic] = callback
        
        if self.is_connected:
            self._subscribe([topic])
    
    def subscribe_ticker(self, symbol: str, callback: Callable):
        """
        Subscribe to ticker updates (melhor para preço em tempo real)
        
        Args:
            symbol: Symbol (ex: 'BTCUSDT')
            callback: Function to call when data arrives
        """
        topic = f"tickers.{symbol}"
        self.subscriptions[f"ticker_{symbol}"] = [topic]
        self.callbacks[topic] = callback
        
        if self.is_connected:
            self._subscribe([topic])
    
    def subscribe_trades(self, symbol: str, callback: Callable):
        """
        Subscribe to trade updates (cada trade executado)
        
        Args:
            symbol: Symbol (ex: 'BTCUSDT')
            callback: Function to call when data arrives
        """
        topic = f"publicTrade.{symbol}"
        self.subscriptions[f"trade_{symbol}"] = [topic]
        self.callbacks[topic] = callback
        
        if self.is_connected:
            self._subscribe([topic])
    
    def subscribe_orderbook(self, symbol: str, depth: int, callback: Callable):
        """
        Subscribe to orderbook updates
        
        Args:
            symbol: Symbol (ex: 'BTCUSDT')
            depth: Depth (1, 50, 200, 500)
            callback: Function to call when data arrives
        """
        topic = f"orderbook.{depth}.{symbol}"
        self.subscriptions[f"orderbook_{symbol}_{depth}"] = [topic]
        self.callbacks[topic] = callback
        
        if self.is_connected:
            self._subscribe([topic])
    
    def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe de um tópico
        
        Args:
            subscription_id: ID retornado pelo subscribe
        """
        if subscription_id in self.subscriptions:
            topics = self.subscriptions[subscription_id]
            
            message = {
                "op": "unsubscribe",
                "args": topics
            }
            
            if self.ws and self.is_connected:
                self.ws.send(json.dumps(message))
            
            del self.subscriptions[subscription_id]
            
            # Remove callbacks
            for topic in topics:
                if topic in self.callbacks:
                    del self.callbacks[topic]
            
            logger.info(f"📡 Unsubscribed from: {topics}")


# Exemplo de uso
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_kline(data):
        print(f"Kline: {json.dumps(data, indent=2)}")
    
    def on_ticker(data):
        tick = data.get('data', {})
        print(f"Ticker - Price: {tick.get('lastPrice')}, Volume: {tick.get('volume24h')}")
    
    # Cliente público
    ws = BybitWebSocket(testnet=True, channel_type="public")
    ws.connect()
    
    # Subscribe
    ws.subscribe_kline("BTCUSDT", "1", on_kline)
    ws.subscribe_ticker("BTCUSDT", on_ticker)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.disconnect()
