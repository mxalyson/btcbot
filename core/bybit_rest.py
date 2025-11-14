"""
Bybit REST API v5 Client - FIXED with longer timeout and retries.
"""

import time
import hmac
import hashlib
import requests
import logging
from typing import Dict, Optional
from urllib.parse import urlencode

logger = logging.getLogger("TradingBot.REST")

class BybitRESTClient:
    """Bybit v5 REST API client with improved timeout handling."""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, timeout: int = 30):
        """
        Initialize REST client.

        Args:
            api_key: API key
            api_secret: API secret
            testnet: Use testnet endpoint
            timeout: Request timeout in seconds (default: 30)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout

        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self.recv_window = 5000

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # ✅ Clock sync - calcula offset com servidor
        self.time_offset = 0
        self._sync_server_time()

    def _sync_server_time(self):
        """Sincroniza clock local com servidor Bybit"""
        try:
            local_time = int(time.time() * 1000)
            response = self.session.get(
                f"{self.base_url}/v5/market/time",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                server_time = int(data.get('result', {}).get('timeSecond', 0)) * 1000
                if server_time > 0:
                    self.time_offset = server_time - local_time
                    logger.info(f"✅ Clock synced - offset: {self.time_offset}ms")
        except Exception as e:
            logger.warning(f"⚠️ Clock sync failed: {e} - using local time")
            self.time_offset = 0

    def _generate_signature(self, params: str, timestamp: int) -> str:
        """Generate HMAC SHA256 signature."""
        param_str = f"{timestamp}{self.api_key}{self.recv_window}{params}"
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                 signed: bool = False, max_retries: int = 3) -> Dict:
        """
        Make HTTP request with retries.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign request
            max_retries: Maximum retry attempts
        
        Returns:
            Response dict
        """
        url = f"{self.base_url}{endpoint}"
        
        if params is None:
            params = {}
        
        headers = {}
        
        if signed:
            timestamp = int(time.time() * 1000) + self.time_offset
            
            if method == 'GET':
                params_str = urlencode(sorted(params.items()))
            else:
                import json
                params_str = json.dumps(params)
            
            signature = self._generate_signature(params_str, timestamp)
            
            headers.update({
                'X-BAPI-API-KEY': self.api_key,
                'X-BAPI-SIGN': signature,
                'X-BAPI-TIMESTAMP': str(timestamp),
                'X-BAPI-RECV-WINDOW': str(self.recv_window)
            })
        
        # Retry loop
        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                else:
                    response = self.session.post(
                        url,
                        json=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Request timed out after {max_retries} attempts")
                    raise
            
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Connection failed after {max_retries} attempts")
                    raise
            
            except Exception as e:
                logger.error(f"Request failed: {e}")
                raise
    
    def get_kline(self, symbol: str, interval: str, limit: int = 200,
                  start_time: Optional[int] = None, end_time: Optional[int] = None) -> Dict:
        """
        Get kline/candlestick data.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
        
        Returns:
            API response dict
        """
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        if start_time:
            params['start'] = start_time
        if end_time:
            params['end'] = end_time
        
        return self._request('GET', '/v5/market/kline', params)
    
    def get_tickers(self, symbol: Optional[str] = None) -> Dict:
        """Get latest ticker information."""
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/v5/market/tickers', params)
    
    def get_orderbook(self, symbol: str, limit: int = 25) -> Dict:
        """Get orderbook depth."""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'limit': limit
        }
        return self._request('GET', '/v5/market/orderbook', params)
    
    def get_instruments_info(self, symbol: Optional[str] = None) -> Dict:
        """Get instrument specifications."""
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/v5/market/instruments-info', params)
    
    def place_order(self, symbol: str, side: str, order_type: str,
                    qty: float, price: Optional[float] = None,
                    stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None,
                    reduce_only: bool = False,
                    order_link_id: Optional[str] = None) -> Dict:
        """Place order (requires authentication)."""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'qty': str(qty)
        }
        
        if price:
            params['price'] = str(price)
        if stop_loss:
            params['stopLoss'] = str(stop_loss)
        if take_profit:
            params['takeProfit'] = str(take_profit)
        if reduce_only:
            params['reduceOnly'] = True
        if order_link_id:
            params['orderLinkId'] = order_link_id
        
        return self._request('POST', '/v5/order/create', params, signed=True)
    
    def set_trading_stop(self, category: str, symbol: str, 
                        stopLoss: Optional[str] = None,
                        takeProfit: Optional[str] = None,
                        trailingStop: Optional[str] = None,
                        tpTriggerBy: str = 'LastPrice',
                        slTriggerBy: str = 'LastPrice',
                        activePrice: Optional[str] = None,
                        tpslMode: str = 'Full',
                        tpSize: Optional[str] = None,
                        slSize: Optional[str] = None,
                        positionIdx: int = 0) -> Dict:
        """
        Set trading stop (Stop Loss / Take Profit) for open position.
        
        Bybit V5 API endpoint: POST /v5/position/trading-stop
        
        Args:
            category: Product type. 'linear' for USDT perpetual
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            stopLoss: Stop loss price
            takeProfit: Take profit price
            trailingStop: Trailing stop distance (in price, not percentage)
            tpTriggerBy: TP trigger price type ('LastPrice', 'MarkPrice', 'IndexPrice')
            slTriggerBy: SL trigger price type ('LastPrice', 'MarkPrice', 'IndexPrice')
            activePrice: Trailing stop activation price
            tpslMode: 'Full' or 'Partial'
            tpSize: TP order size (for partial mode)
            slSize: SL order size (for partial mode)
            positionIdx: Position index. 0 for one-way mode, 1 for Buy side, 2 for Sell side
        
        Returns:
            API response dict
        
        Example:
            result = client.set_trading_stop(
                category='linear',
                symbol='BTCUSDT',
                stopLoss='95000.00',
                takeProfit='105000.00',
                positionIdx=0
            )
        """
        params = {
            'category': category,
            'symbol': symbol,
            'positionIdx': positionIdx,
            'tpTriggerBy': tpTriggerBy,
            'slTriggerBy': slTriggerBy,
            'tpslMode': tpslMode
        }
        
        if stopLoss:
            params['stopLoss'] = stopLoss
        if takeProfit:
            params['takeProfit'] = takeProfit
        if trailingStop:
            params['trailingStop'] = trailingStop
        if activePrice:
            params['activePrice'] = activePrice
        if tpSize:
            params['tpSize'] = tpSize
        if slSize:
            params['slSize'] = slSize
        
        return self._request('POST', '/v5/position/trading-stop', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: Optional[str] = None,
                     order_link_id: Optional[str] = None) -> Dict:
        """Cancel order."""
        params = {
            'category': 'linear',
            'symbol': symbol
        }
        
        if order_id:
            params['orderId'] = order_id
        elif order_link_id:
            params['orderLinkId'] = order_link_id
        
        return self._request('POST', '/v5/order/cancel', params, signed=True)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> Dict:
        """Get open orders."""
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/v5/order/realtime', params, signed=True)
    
    def get_positions(self, symbol: Optional[str] = None) -> Dict:
        """Get position information."""
        params = {'category': 'linear'}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/v5/position/list', params, signed=True)
    
    def get_wallet_balance(self, account_type: str = 'UNIFIED') -> Dict:
        """Get wallet balance."""
        params = {'accountType': account_type}
        return self._request('GET', '/v5/account/wallet-balance', params, signed=True)
    
    def set_leverage(self, symbol: str, buy_leverage: str, sell_leverage: str) -> Dict:
        """Set leverage for symbol."""
        params = {
            'category': 'linear',
            'symbol': symbol,
            'buyLeverage': buy_leverage,
            'sellLeverage': sell_leverage
        }
        return self._request('POST', '/v5/position/set-leverage', params, signed=True)

    def get_server_time(self) -> Dict:
        """Get Bybit server time (no authentication required)."""
        return self._request('GET', '/v5/market/time', {})
