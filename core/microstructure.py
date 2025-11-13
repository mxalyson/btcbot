"""Market microstructure: orderbook imbalance, spread, orderflow."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def calculate_obi(bids: Dict[float, float], asks: Dict[float, float], levels: int = 5) -> float:
    """
    Calculate Order Book Imbalance (OBI).
    OBI = (Σ BidQty - Σ AskQty) / (Σ BidQty + Σ AskQty)
    """
    try:
        # Sort and get top N levels
        sorted_bids = sorted(bids.items(), key=lambda x: x[0], reverse=True)[:levels]
        sorted_asks = sorted(asks.items(), key=lambda x: x[0])[:levels]
        
        bid_qty = sum(qty for _, qty in sorted_bids)
        ask_qty = sum(qty for _, qty in sorted_asks)
        
        if bid_qty + ask_qty == 0:
            return 0.0
        
        obi = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        return obi
        
    except Exception:
        return 0.0

def calculate_spread(bids: Dict[float, float], asks: Dict[float, float]) -> Tuple[float, float, float]:
    """
    Calculate bid-ask spread.
    Returns: (spread_abs, spread_pct, mid_price)
    """
    try:
        best_bid = max(bids.keys())
        best_ask = min(asks.keys())
        
        spread_abs = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_pct = (spread_abs / mid_price) * 100
        
        return spread_abs, spread_pct, mid_price
        
    except Exception:
        return 0.0, 0.0, 0.0

def calculate_orderflow(trades: List[Dict], lookback: int = 50) -> Dict[str, float]:
    """
    Calculate orderflow from recent trades using tick rule.
    Returns delta between buyer and seller volume.
    """
    if not trades or len(trades) < 2:
        return {'buy_volume': 0, 'sell_volume': 0, 'delta': 0, 'delta_pct': 0}
    
    recent_trades = trades[-lookback:]
    buy_volume = 0
    sell_volume = 0
    
    for trade in recent_trades:
        size = trade.get('v', 0)
        side = trade.get('S', '')  # 'Buy' or 'Sell'
        
        if side == 'Buy':
            buy_volume += size
        elif side == 'Sell':
            sell_volume += size
    
    total_volume = buy_volume + sell_volume
    delta = buy_volume - sell_volume
    delta_pct = (delta / total_volume * 100) if total_volume > 0 else 0
    
    return {
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'delta': delta,
        'delta_pct': delta_pct
    }

class MicrostructureAnalyzer:
    """Real-time microstructure analysis."""
    
    def __init__(self, config: dict):
        self.config = config
        self.trades_history: Dict[str, List] = {}
        
    def update_trades(self, symbol: str, trades: List[Dict]):
        """Update trade history."""
        if symbol not in self.trades_history:
            self.trades_history[symbol] = []
        
        self.trades_history[symbol].extend(trades)
        
        # Keep only recent trades
        max_trades = self.config['orderflow_lookback'] * 2
        if len(self.trades_history[symbol]) > max_trades:
            self.trades_history[symbol] = self.trades_history[symbol][-max_trades:]
    
    def analyze(self, symbol: str, orderbook: Dict) -> Dict[str, float]:
        """Analyze current microstructure."""
        bids = orderbook.get('bids', {})
        asks = orderbook.get('asks', {})
        
        # OBI
        obi = calculate_obi(bids, asks, self.config['obi_levels'])
        
        # Spread
        spread_abs, spread_pct, mid_price = calculate_spread(bids, asks)
        
        # Orderflow
        trades = self.trades_history.get(symbol, [])
        orderflow = calculate_orderflow(trades, self.config['orderflow_lookback'])
        
        return {
            'obi': obi,
            'spread_abs': spread_abs,
            'spread_pct': spread_pct,
            'mid_price': mid_price,
            'buy_volume': orderflow['buy_volume'],
            'sell_volume': orderflow['sell_volume'],
            'orderflow_delta': orderflow['delta'],
            'orderflow_delta_pct': orderflow['delta_pct']
        }
