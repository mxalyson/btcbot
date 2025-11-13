"""
MASTER SCALPER - Live Trading OTIMIZADO - ETH [V6.0 - CORREÇÕES CRÍTICAS]

✅ V6.0 - CORREÇÕES CRÍTICAS (2025):
   🔴 CRÍTICO - TP3: Agora verifica após TP2 (bugfix lógica)
   🔴 CRÍTICO - Symbol: Adicionado como parâmetro do construtor
   🔴 CRÍTICO - Testnet: Removido hardcoded False, usa .env

   🟡 IMPORTANTE:
   - Preço sempre atualizado quando há posição (ticker real-time)
   - Persistência de highest/lowest price para trailing stop
   - Retry logic com exponential backoff (3 tentativas)
   - Validações de segurança (min qty, SL/TP válidos)

   🟢 COMPATIBILIDADE:
   - Fallbacks para estados antigos (V5.x)
   - Inicialização automática de campos faltantes

✅ LÓGICA DO BACKTEST IMPLEMENTADA:
   - TP1 (0.7x ATR): Fecha 60% + Move SL para BE
   - TP2 (1.3x ATR): Ativa trailing stop local
   - TP3 (2.0x ATR): Fecha 40% restante APÓS TP2
   - Trailing Stop: Atualiza SL continuamente após TP2
   - Gestão manual via API Bybit

✅ FEATURES:
   - Loop constante monitorando preço (5s com posição, 30s sem)
   - Parciais executadas via API com retry
   - Validações de segurança completas
   - Notificações Telegram detalhadas
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import argparse
import pickle
import time
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import requests

load_dotenv()

from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore

logger = None


# ======== EXECUTION HELPERS: tick/step rounding + price band limit ========
from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 28

def _to_decimal(x):
    try:
        return Decimal(str(x))
    except:
        return Decimal(0)

def round_to_step(value: float, step: float) -> float:
    step_d = _to_decimal(step)
    if step_d <= 0:
        return float(value)
    v = _to_decimal(value)
    q = (v // step_d) * step_d
    return float(q)

def round_price(value: float, tick: float) -> float:
    return round_to_step(value, tick)

def round_qty(value: float, step: float, min_qty: float) -> float:
    q = round_to_step(value, step)
    if q < min_qty:
        q = _to_decimal(min_qty)
    return float(q)

def fetch_market_meta(rest, symbol: str):
    """
    Fetch tickSize, qtyStep, minOrderQty from instruments. Fallbacks if API unavailable.
    Usa valores hardcoded otimizados para ETHUSDT/BTCUSDT se API falhar.
    """
    # Valores padrão otimizados por símbolo
    if 'ETH' in symbol:
        tick = 0.01    # ETHUSDT tick size
        step = 0.01    # ETHUSDT qty step (0.01 ETH)
        min_qty = 0.01 # Mínimo 0.01 ETH
    elif 'BTC' in symbol:
        tick = 0.1     # BTCUSDT tick size
        step = 0.001   # BTCUSDT qty step (0.001 BTC)
        min_qty = 0.001 # Mínimo 0.001 BTC
    else:
        tick = 0.01
        step = 0.01
        min_qty = 0.01

    try:
        # Usa o método correto do BybitRESTClient (verificado em core/bybit_rest.py:184)
        meta = rest.get_instruments_info(symbol=symbol)

        if meta and meta.get('retCode') == 0:
            lst = meta.get('result', {}).get('list', [])
            if lst:
                info = lst[0]
                if 'priceFilter' in info and 'tickSize' in info['priceFilter']:
                    tick = float(info['priceFilter']['tickSize'])
                if 'lotSizeFilter' in info:
                    lf = info['lotSizeFilter']
                    if 'qtyStep' in lf:
                        step = float(lf['qtyStep'])
                    if 'minOrderQty' in lf:
                        min_qty = float(lf['minOrderQty'])
                logger.info(f"✅ Market meta via API: tick={tick}, step={step}, min_qty={min_qty}")
        else:
            retMsg = meta.get('retMsg', 'Unknown error') if meta else 'No response'
            logger.warning(f'⚠️ API retornou erro: {retMsg}. Usando fallback.')
    except AttributeError as e:
        logger.warning(f'⚠️ Método get_instruments_info não disponível: {e}. Usando fallback.')
    except Exception as e:
        logger.warning(f'⚠️ fetch_market_meta usando fallback ({e.__class__.__name__}: {e})')

    return tick, step, min_qty

def get_best_quotes(rest, symbol: str):
    best_bid = best_ask = last = None
    try:
        ob = rest.get_orderbook(symbol=symbol, depth=1)
        if ob and ob.get('retCode') == 0:
            data = ob.get('result', {})
            bids = data.get('b', []) or data.get('bids', [])
            asks = data.get('a', []) or data.get('asks', [])
            if bids:
                b0 = bids[0]
                best_bid = float(b0[0] if isinstance(b0, (list, tuple)) else b0.get('price'))
            if asks:
                a0 = asks[0]
                best_ask = float(a0[0] if isinstance(a0, (list, tuple)) else a0.get('price'))
    except Exception as e:
        logger.warning(f'orderbook fallback: {e}')
    try:
        tk = rest.get_tickers(symbol=symbol)
        if tk and tk.get('retCode') == 0:
            result = tk.get('result', {})
            ticker_list = result.get('list', [])
            if ticker_list:
                last = float(ticker_list[0].get('lastPrice', 0))
    except Exception as e:
        logger.warning(f'ticker fallback: {e}')
    return best_bid, best_ask, last

def choose_limit_price(side: str, mark: float, best_bid: float, best_ask: float, band_bps: float, tick: float) -> float:
    # band_bps is in basis points, e.g., 5 => 0.0005
    band = band_bps / 10000.0
    if side == 'Buy':
        cap = min(best_ask or mark*(1+band), mark*(1+band))
        return round_price(cap, tick)
    else:
        cap = max(best_bid or mark*(1-band), mark*(1-band))
        return round_price(cap, tick)

def get_position_size_usd(rest, symbol: str) -> float:
    try:
        p = rest.get_positions(symbol=symbol)
        if p and p.get('retCode') == 0:
            lst = p.get('result', {}).get('list', [])
            usd = 0.0
            for it in lst:
                if it.get('symbol') == symbol:
                    size = float(it.get('size', 0) or 0)
                    avgp = float(it.get('avgPrice', 0) or 0)
                    usd += abs(size * avgp)
            return usd
    except Exception as e:
        logger.warning(f'get_position_size_usd: {e}')
    return 0.0

def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0, max_delay: float = 16.0):
    """
    Executa função com retry exponential backoff
    Args:
        func: Função a executar (sem argumentos)
        max_retries: Número máximo de tentativas
        initial_delay: Delay inicial em segundos
        max_delay: Delay máximo em segundos
    Returns:
        Resultado da função ou None se falhar
    """
    delay = initial_delay
    last_error = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}. Retry em {delay}s...")
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
            else:
                logger.error(f"Todas as {max_retries} tentativas falharam: {last_error}")

    return None


STATE_FILE = "storage/bot_state_eth.json"

# CONFIGURAÇÕES BACKTEST
TP_MULTS = {
    'tp1': 0.7,   # 60% parcial
    'tp2': 1.3,   # Ativa trailing
    'tp3': 2.0    # 40% restante
}

PARTIAL_FRACTION = 0.60  # 60% no TP1
MOVE_TO_BE = True  # Move SL para breakeven
TRAILING_STOP_ENABLED = True
TRAILING_STOP_DISTANCE = 0.5  # 0.5x ATR

FEE_RATE = 0.0006
SLIPPAGE = 0.0001

# Intervalo de monitoramento
MONITOR_INTERVAL = 5  # 5 segundos quando há posição aberta

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_message(self, message: str, parse_mode: str = "HTML"):
        try:
            url = f"{self.base_url}/sendMessage"
            data = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            if logger:
                logger.error(f"Telegram error: {e}")
            return False

    def send_trade_open(self, trade: Dict):
        direction_emoji = "🟢" if trade['direction'] == 'long' else "🔴"
        mode = "💰 LIVE" if trade.get('is_live', False) else "📝 PAPER"

        message = f"""{direction_emoji} NOVA OPERAÇÃO {mode}

Direção: {trade['direction'].upper()}
Entrada: ${trade['entry_price']:,.2f}
Qty: {trade.get('qty', 'N/A')} ETH
Confiança: {trade['ml_confidence']*100:.1f}%

SL: ${trade['stop_loss']:,.2f}
TP1 (60%): ${trade['tp1']:,.2f}
TP2 (trailing): ${trade['tp2']:,.2f}
TP3 (40%): ${trade['tp3']:,.2f}

Order ID: {trade.get('order_id', 'N/A')}

📊 Estratégia: TP1 parcial → BE → Trailing"""
        self.send_message(message)

    def send_partial_tp(self, trade: Dict, tp_level: int, partial_qty: float, partial_pnl: float, partial_pct: float):
        result = "✅"
        mode = "💰 LIVE" if trade.get('is_live', False) else "📝 PAPER"

        if tp_level == 1:
            action = f"Fechou {PARTIAL_FRACTION*100:.0f}% ({partial_qty:.2f} ETH)"
            next_step = "SL movido para Breakeven"
        else:
            action = f"Fechou 40% restante ({partial_qty:.2f} ETH)"
            next_step = "Trade finalizado"

        message = f"""{result} TP{tp_level} ATINGIDO {mode}

Direção: {trade['direction'].upper()}
{action}

PnL Parcial: ${partial_pnl:+,.2f} ({partial_pct:+.2f}%)
{next_step}"""
        self.send_message(message)

    def send_trailing_activated(self, trade: Dict, current_price: float):
        mode = "💰 LIVE" if trade.get('is_live', False) else "📝 PAPER"

        message = f"""🔄 TRAILING STOP ATIVADO {mode}

Preço atual: ${current_price:,.2f}
Distância: {TRAILING_STOP_DISTANCE}x ATR

Posição restante: 40%
SL será atualizado automaticamente"""
        self.send_message(message)

    def send_be_moved(self, trade: Dict, new_sl: float):
        mode = "💰 LIVE" if trade.get('is_live', False) else "📝 PAPER"

        remaining_qty = trade.get('remaining_qty', trade.get('qty', 0))
        entry = trade['entry_price']
        guaranteed_pnl = (new_sl - entry) * remaining_qty if trade['direction'] == 'long' else (entry - new_sl) * remaining_qty

        message = f"""🛡️ SL MOVIDO PARA BREAKEVEN {mode}

Novo SL: ${new_sl:,.2f}
Lucro garantido: ${guaranteed_pnl:+,.2f}

Trade agora sem risco!"""
        self.send_message(message)

    def send_trade_close(self, trade: Dict, pnl_amount: float, pnl_pct: float):
        result = "✅" if pnl_amount > 0 else "❌"
        mode = "💰 LIVE" if trade.get('is_live', False) else "📝 PAPER"

        message = f"""{result} TRADE FECHADO {mode}

Direção: {trade['direction'].upper()}
Entrada: ${trade['entry_price']:,.2f}
Saída Final: ${trade['exit_price']:,.2f}

PnL Total: ${pnl_amount:+,.2f} ({pnl_pct:+.2f}%)
Motivo: {trade['exit_reason']}"""
        self.send_message(message)

    def send_error(self, error_msg: str):
        message = f"""⚠️ ERRO

{error_msg}"""
        self.send_message(message)

    def send_status(self, stats: Dict):
        message = f"""📊 STATUS ETH

Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']:.1f}%
PnL Total: ${stats['total_pnl']:+,.2f}
PnL Médio: ${stats['avg_pnl']:+,.2f}"""
        self.send_message(message)


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MASTER TRADER advanced features"""
    df_features = df.copy()

    for period in [3, 5, 8, 13, 21]:
        df_features[f'momentum_{period}'] = df_features['close'].pct_change(period) * 100
        df_features[f'volume_ratio_{period}'] = df_features['volume'] / df_features['volume'].rolling(period).mean()

    if 'ema50' in df_features.columns and 'ema200' in df_features.columns:
        df_features['trend_strength'] = (df_features['ema50'] - df_features['ema200']) / df_features['ema200'] * 100

    if 'atr' in df_features.columns:
        df_features['volatility_regime'] = df_features['atr'] / df_features['atr'].rolling(50).mean()

    df_features['price_position'] = ((df_features['close'] - df_features['low'].rolling(20).min()) /
                                      (df_features['high'].rolling(20).max() - df_features['low'].rolling(20).min())).fillna(0.5)

    df_features['volume_momentum'] = df_features['volume'].pct_change(5)
    df_features['price_acceleration'] = df_features['close'].diff(2) - df_features['close'].diff(1)

    return df_features


class MasterLiveTrader:
    def __init__(self, config: dict, model_path: str, telegram: TelegramNotifier, symbol: str = 'ETHUSDT'):
        self.config = config
        self.symbol = symbol
        self.model_path = Path(model_path)
        self.telegram = telegram

        if not self.model_path.exists():
            raise ValueError(f"Model not found: {model_path}")

        with open(self.model_path, 'rb') as f:
            self.model_data = pickle.load(f)

        self.model = self.model_data['model']
        self.feature_names = self.model_data['feature_names']

        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        risk_value = float(os.getenv('RISK_PER_TRADE', '0.02'))
        self.risk_per_trade = risk_value if risk_value < 1 else risk_value / 100
        self.min_confidence = float(os.getenv('MIN_ML_CONFIDENCE', '0.40'))

        atr_sl_value = float(os.getenv('ATR_MULT_SL', '1.5'))
        self.atr_mult_sl = atr_sl_value

        # Configuração de modo (testnet vs live)
        self.bybit_testnet = os.getenv('BYBIT_TESTNET', 'true').lower() == 'true'
        self.is_live_mode = not self.bybit_testnet

        logger.info("="*70)
        logger.info("📋 CONFIGURAÇÕES - ETH [V6.0 CORREÇÕES CRÍTICAS]")
        logger.info("="*70)
        logger.info(f"Estratégia: TP1 parcial → BE → Trailing")
        logger.info(f"Modelo: {self.model_path.name}")
        logger.info(f"Capital Inicial: ${self.initial_capital:,.2f}")
        logger.info(f"Risco por Trade: {self.risk_per_trade:.2%}")
        logger.info(f"MIN_ML_CONFIDENCE: {self.min_confidence:.2%}")
        logger.info(f"EXECUTION MODE: {'💰 LIVE' if self.is_live_mode else '📝 PAPER'}")
        logger.info(f"TP1: {TP_MULTS['tp1']}x ATR ({PARTIAL_FRACTION*100:.0f}% parcial)")
        logger.info(f"TP2: {TP_MULTS['tp2']}x ATR (ativa trailing)")
        logger.info(f"TP3: {TP_MULTS['tp3']}x ATR (fecha resto)")
        logger.info(f"Trailing Distance: {TRAILING_STOP_DISTANCE}x ATR")
        logger.info(f"Monitor Interval: {MONITOR_INTERVAL}s")
        logger.info(f"✅ V6.0: Correções críticas + Retry logic + Validações")
        logger.info("="*70)

        self.position = None
        self.capital = self.initial_capital
        self.trades_history = []
        self.cooldown_until = 0

        # Estados da posição
        self.tp1_hit = False
        self.tp2_hit = False
        self.trailing_active = False
        self.highest_price = None
        self.lowest_price = None

        # Inicializa cliente Bybit com configuração correta
        self.rest_client = BybitRESTClient(
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET'),
            testnet=self.bybit_testnet
        )

        
        # Market meta + execution band
        try:
            self.tick_size, self.qty_step, self.min_qty = fetch_market_meta(self.rest_client, self.symbol)
        except Exception:
            self.tick_size, self.qty_step, self.min_qty = 0.01, 0.001, 0.001
        self.price_band_bps = float(os.getenv('PRICE_BAND_BPS', '5'))  # default 5 bps

        self.dm = DataManager(self.rest_client)
        self.fs = FeatureStore(config)

        mode = 'TESTNET' if self.bybit_testnet else 'MAINNET'
        logger.info(f"✅ Bot inicializado ({mode})")

        if self.is_live_mode and not self.bybit_testnet:
            logger.warning("⚠️" * 20)
            logger.warning("⚠️ LIVE MODE + MAINNET = DINHEIRO REAL!")
            logger.warning("⚠️" * 20)

    def save_state(self):
        state = {
            'capital': self.capital,
            'position': self.position,
            'trades_history': self.trades_history,
            'cooldown_until': self.cooldown_until or 0,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'trailing_active': self.trailing_active,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price
        }

        try:
            os.makedirs("storage", exist_ok=True)
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Save error: {e}")

    def recover_state(self):
        if not Path(STATE_FILE).exists():
            logger.info("📝 Starting fresh")
            return

        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

            file_age = datetime.now().timestamp() - Path(STATE_FILE).stat().st_mtime
            if file_age > 3600:
                logger.warning("State too old")
                return

            self.capital = state.get('capital', self.initial_capital)
            self.trades_history = state.get('trades_history', [])
            self.cooldown_until = state.get('cooldown_until', 0)
            self.position = state.get('position')
            self.tp1_hit = state.get('tp1_hit', False)
            self.tp2_hit = state.get('tp2_hit', False)
            self.trailing_active = state.get('trailing_active', False)
            self.highest_price = state.get('highest_price')
            self.lowest_price = state.get('lowest_price')

            # ✅ V5.2: Inicializa campos faltantes em posições antigas
            if self.position:
                qty = self.position.get('qty', 0)

                # Adiciona campos se não existirem
                if 'remaining_qty' not in self.position:
                    self.position['remaining_qty'] = qty
                    logger.warning("⚠️ Campo 'remaining_qty' não encontrado - inicializado com qty total")

                if 'current_sl' not in self.position:
                    self.position['current_sl'] = self.position.get('stop_loss', 0)
                    logger.warning("⚠️ Campo 'current_sl' não encontrado - inicializado com stop_loss")

                if 'atr' not in self.position:
                    entry = self.position.get('entry_price', 3500)
                    self.position['atr'] = entry * 0.01  # Estimativa: 1% do preço
                    logger.warning("⚠️ Campo 'atr' não encontrado - estimado em 1% do preço de entrada")

                # ✅ Inicializa highest/lowest se não existirem (para trailing stop)
                direction = self.position.get('direction', 'long')
                if self.highest_price is None and direction == 'long':
                    self.highest_price = self.position.get('entry_price', 0)
                    logger.warning("⚠️ highest_price inicializado com entry_price")
                if self.lowest_price is None and direction == 'short':
                    self.lowest_price = self.position.get('entry_price', 0)
                    logger.warning("⚠️ lowest_price inicializado com entry_price")

                logger.info(f"🔄 Recovered {self.position['direction']} @ ${self.position['entry_price']:,.2f}")
                logger.info(f"   Remaining: {self.position['remaining_qty']:.2f} ETH")
                logger.info(f"   TP1 Hit: {self.tp1_hit} | TP2 Hit: {self.tp2_hit} | Trailing: {self.trailing_active}")
                if self.trailing_active:
                    if direction == 'long':
                        logger.info(f"   Highest Price: ${self.highest_price:,.2f}")
                    else:
                        logger.info(f"   Lowest Price: ${self.lowest_price:,.2f}")

        except Exception as e:
            logger.warning(f"Recover failed: {e}")
            import traceback
            traceback.print_exc()

    def get_current_data(self, symbol: str, timeframe: str = '15m', lookback_days: int = 30) -> pd.DataFrame:
        df = self.dm.get_data(symbol, timeframe, lookback_days, use_cache=False)
        if df.empty:
            raise ValueError("No data received from API")

        df = self.fs.build_features(df, normalize=False)
        df = create_advanced_features(df)
        return df

    def get_signal(self, df: pd.DataFrame) -> tuple:
        try:
            latest = df.iloc[-1]
            X = df[self.feature_names].fillna(0).iloc[-1:].values

            ml_probs = self.model.predict(X)

            if isinstance(ml_probs, np.ndarray):
                if ml_probs.ndim > 0 and len(ml_probs) > 0:
                    ml_prob_up = float(ml_probs.flatten()[0])
                else:
                    ml_prob_up = float(ml_probs)
            elif isinstance(ml_probs, (np.floating, np.integer)):
                ml_prob_up = float(ml_probs)
            else:
                ml_prob_up = float(ml_probs)

            ml_prob_down = 1.0 - ml_prob_up
            ml_confidence = abs(ml_prob_up - 0.5) * 2.0

            signal = 0
            if ml_prob_up > 0.5 and ml_confidence >= self.min_confidence:
                signal = 1
            elif ml_prob_down > 0.5 and ml_confidence >= self.min_confidence:
                signal = -1

            return signal, ml_confidence, latest

        except Exception as e:
            logger.error(f"Error in get_signal: {e}")
            import traceback
            traceback.print_exc()
            return 0, 0.0, df.iloc[-1]

    def calculate_position_qty(self, symbol: str, price: float, size_usd: float) -> float:
        """Calcula quantidade em crypto baseado no tamanho USD"""
        qty = size_usd / price

        if 'BTC' in symbol:
            qty = max(0.001, round(qty, 3))
        elif 'ETH' in symbol:
            qty = max(0.01, round(qty, 2))
        else:
            qty = max(0.1, round(qty, 1))

        return qty

    def open_position(self, symbol: str, signal: int, current_data: pd.Series, ml_confidence: float):
        direction = 'long' if signal == 1 else 'short'
        price = current_data['close']
        atr = current_data.get('atr', price * 0.01)

        # Calcula SL e TPs
        if direction == 'long':
            sl = price - (atr * self.atr_mult_sl)
            tp1 = price + (atr * TP_MULTS['tp1'])
            tp2 = price + (atr * TP_MULTS['tp2'])
            tp3 = price + (atr * TP_MULTS['tp3'])
            side = 'Buy'
        else:
            sl = price + (atr * self.atr_mult_sl)
            tp1 = price - (atr * TP_MULTS['tp1'])
            tp2 = price - (atr * TP_MULTS['tp2'])
            tp3 = price - (atr * TP_MULTS['tp3'])
            side = 'Sell'

        # Calcula tamanho
        sl_dist = abs((sl - price) / price)
        risk_amt = self.capital * self.risk_per_trade
        size_usd = min(
            risk_amt / sl_dist if sl_dist > 0 else self.capital * 0.1,
            self.capital * 0.95
        )

        qty = self.calculate_position_qty(symbol, price, size_usd)
        
        # Rounding to market constraints
        tick = getattr(self, 'tick_size', 0.01)
        step = getattr(self, 'qty_step', 0.001)
        min_qty = getattr(self, 'min_qty', 0.001)
        
        sl  = round_price(sl, tick)
        tp1 = round_price(tp1, tick)
        tp2 = round_price(tp2, tick)
        tp3 = round_price(tp3, tick)
        qty = round_qty(qty, step, min_qty)
        price = round_price(price, tick)
        actual_size_usd = qty * price

        # ✅ VALIDAÇÕES DE SEGURANÇA
        if qty < min_qty:
            logger.error(f"❌ Quantidade {qty} menor que mínimo {min_qty}!")
            return

        if actual_size_usd < 10:
            logger.warning(f"⚠️ Tamanho muito pequeno: ${actual_size_usd:.2f} < $10")
            return

        # Valida que SL faz sentido (não pode ser igual ou pior que entrada)
        if direction == 'long' and sl >= price:
            logger.error(f"❌ SL inválido para LONG: ${sl:.2f} >= ${price:.2f}")
            return
        if direction == 'short' and sl <= price:
            logger.error(f"❌ SL inválido para SHORT: ${sl:.2f} <= ${price:.2f}")
            return

        # Valida que TPs fazem sentido
        if direction == 'long' and (tp1 <= price or tp2 <= tp1 or tp3 <= tp2):
            logger.error(f"❌ TPs inválidos para LONG")
            return
        if direction == 'short' and (tp1 >= price or tp2 >= tp1 or tp3 >= tp2):
            logger.error(f"❌ TPs inválidos para SHORT")
            return

        logger.info(f"📊 Position: {qty} {symbol.replace('USDT', '')} = ${actual_size_usd:,.2f}")
        logger.info(f"✅ Validações de segurança: OK")

        order_id = None
        actual_entry_price = price
        is_live = self.is_live_mode

        if self.is_live_mode:
            try:
                logger.info(f"💰 Sending LIVE {side} order to Bybit...")

                pre_pos_usd = get_position_size_usd(self.rest_client, symbol)
                best_bid, best_ask, last = get_best_quotes(self.rest_client, symbol)
                limit_px = choose_limit_price(side, last or price, best_bid, best_ask, self.price_band_bps, tick)
                order = self.rest_client.place_order(
                    symbol=symbol,
                    side=side,
                    order_type='Limit',
                    price=str(limit_px),
                    qty=qty,
                    timeInForce='IOC'
                )
                post_pos_usd = get_position_size_usd(self.rest_client, symbol)
                if post_pos_usd <= pre_pos_usd:
                    logger.info('⚠️ Limit IOC did not fill enough — fallback to Market')
                    order = self.rest_client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type='Market',
                        qty=qty
                    )

                logger.info(f"📥 API Response: {order}")

                if order and 'retCode' in order and order['retCode'] == 0:
                    if 'result' in order and isinstance(order['result'], dict):
                        result = order['result']

                        if 'orderId' in result:
                            order_id = result['orderId']
                            logger.info(f"✅ Order executed! ID: {order_id}")

                            if 'price' in result and result['price']:
                                try:
                                    actual_entry_price = float(result['price'])
                                except:
                                    actual_entry_price = price

                        # ✅ CONFIGURA APENAS SL (sem TP - gestão manual)
                        try:
                            logger.info(f"📍 Setting SL only (TPs manual)...")
                            logger.info(f"   SL: ${sl:,.2f}")

                            sl_result = self.rest_client.set_trading_stop(
                                category='linear',
                                symbol=symbol,
                                stopLoss=str(round_price(sl, getattr(self, 'tick_size', 0.01))),
                                positionIdx=0
                            )

                            if sl_result and 'retCode' in sl_result and sl_result['retCode'] == 0:
                                logger.info(f"✅ SL configured!")
                            else:
                                error_msg = sl_result.get('retMsg', 'Unknown')
                                logger.error(f"❌ Failed SL: {error_msg}")

                        except Exception as e:
                            logger.error(f"⚠️ Failed SL: {e}")

                else:
                    raise Exception("API error")

            except Exception as e:
                logger.error(f"❌ Failed order: {e}")
                self.telegram.send_error(f"Failed: {e}")
                return

        # Salva posição
        self.position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': actual_entry_price,
            'entry_time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'size': actual_size_usd,
            'qty': qty,
            'remaining_qty': qty,  # ✅ V5.2: Garantido
            'stop_loss': sl,
            'current_sl': sl,  # ✅ V5.2: Garantido
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'atr': atr,  # ✅ V5.2: Garantido
            'ml_confidence': ml_confidence,
            'order_id': order_id,
            'is_live': is_live
        }

        # Reset estados
        self.tp1_hit = False
        self.tp2_hit = False
        self.trailing_active = False
        self.highest_price = price if direction == 'long' else None
        self.lowest_price = price if direction == 'short' else None

        mode_str = "💰 LIVE" if is_live else "📝 PAPER"
        logger.info(f"🟢 OPENED {direction.upper()} {mode_str} @ ${actual_entry_price:,.2f}")
        logger.info(f"   Total: {qty} ETH | Size: ${actual_size_usd:,.2f}")
        logger.info(f"   SL: ${sl:,.2f}")
        logger.info(f"   TP1 (60%): ${tp1:,.2f} | TP2: ${tp2:,.2f} | TP3: ${tp3:,.2f}")

        try:
            self.telegram.send_trade_open(self.position)
        except Exception as e:
            logger.error(f"Telegram failed: {e}")

        self.save_state()

    def close_partial(self, qty_to_close: float, price: float, reason: str) -> bool:
        """Fecha parcial da posição com retry"""
        if not self.position or not self.is_live_mode:
            return False

        symbol = self.position['symbol']
        direction = self.position['direction']
        side = 'Sell' if direction == 'long' else 'Buy'

        logger.info(f"📤 Closing {qty_to_close} {symbol.replace('USDT', '')} at ${price:,.2f}...")

        def _place_partial_order():
            order = self.rest_client.place_order(
                symbol=symbol,
                side=side,
                order_type='Market',
                qty=qty_to_close,
                reduceOnly=True
            )

            if order and 'retCode' in order and order['retCode'] == 0:
                logger.info(f"✅ Partial closed!")
                return True
            else:
                error = order.get('retMsg', 'Unknown') if order else 'No response'
                raise Exception(f"API error: {error}")

        # ✅ Usa retry com backoff (máx 3 tentativas)
        result = retry_with_backoff(_place_partial_order, max_retries=3, initial_delay=2.0)
        return result is not None and result

    def update_stop_loss(self, new_sl: float) -> bool:
        """Atualiza stop loss na Bybit com retry"""
        if not self.position or not self.is_live_mode:
            return False

        symbol = self.position['symbol']
        logger.info(f"🔄 Updating SL to ${new_sl:,.2f}...")

        def _update_sl():
            result = self.rest_client.set_trading_stop(
                category='linear',
                symbol=symbol,
                stopLoss=str(round_price(new_sl, getattr(self, 'tick_size', 0.01))),
                positionIdx=0
            )

            if result and 'retCode' in result and result['retCode'] == 0:
                logger.info(f"✅ SL updated!")
                self.position['current_sl'] = new_sl
                return True
            else:
                error = result.get('retMsg', 'Unknown') if result else 'No response'
                raise Exception(f"API error: {error}")

        # ✅ Usa retry com backoff (máx 3 tentativas)
        result = retry_with_backoff(_update_sl, max_retries=3, initial_delay=2.0)
        return result is not None and result

    def monitor_position(self, current_price: float):
        """
        ✅ LÓGICA PRINCIPAL - Monitora posição e executa parciais
        ✅ V5.2: Com fallbacks para compatibilidade
        """
        if not self.position:
            return

        direction = self.position['direction']
        entry = self.position['entry_price']
        tp1 = self.position['tp1']
        tp2 = self.position['tp2']
        tp3 = self.position['tp3']

        # ✅ V5.2: Fallbacks para campos potencialmente faltantes
        current_sl = self.position.get('current_sl', self.position.get('stop_loss', 0))
        qty = self.position['qty']
        remaining_qty = self.position.get('remaining_qty', qty)
        atr = self.position.get('atr', entry * 0.01)

        # Atualiza highest/lowest
        if direction == 'long':
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
        else:
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price

        # ✅ VERIFICA TP1 (60% PARCIAL + MOVE BE)
        if not self.tp1_hit:
            tp1_hit = (direction == 'long' and current_price >= tp1) or                       (direction == 'short' and current_price <= tp1)

            if tp1_hit:
                logger.info(f"🎯 TP1 ATINGIDO! ${current_price:,.2f}")

                # Fecha 60%
                partial_qty = round(qty * PARTIAL_FRACTION, 2)

                if self.is_live_mode:
                    success = self.close_partial(partial_qty, current_price, 'tp1')
                    if not success:
                        logger.warning("⚠️ Falha ao fechar parcial - tentará novamente")
                        return

                # Calcula PnL parcial
                if direction == 'long':
                    pnl_pct = ((current_price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100

                partial_pnl = (partial_qty * entry) * (pnl_pct / 100)

                # Atualiza posição
                self.position['remaining_qty'] = remaining_qty - partial_qty
                self.tp1_hit = True

                logger.info(f"✅ Fechou {partial_qty} ETH | PnL: ${partial_pnl:+,.2f}")

                # Move SL para BE (com ajuste de custos)
                be_adjustment = entry * (FEE_RATE + SLIPPAGE)
                new_sl = entry + be_adjustment if direction == 'long' else entry - be_adjustment

                if self.is_live_mode:
                    self.update_stop_loss(new_sl)

                self.position['current_sl'] = new_sl

                logger.info(f"🛡️ SL movido para BE: ${new_sl:,.2f}")

                try:
                    self.telegram.send_partial_tp(self.position, 1, partial_qty, partial_pnl, pnl_pct)
                    self.telegram.send_be_moved(self.position, new_sl)
                except:
                    pass

                self.save_state()

        # ✅ VERIFICA TP2 (ATIVA TRAILING)
        if self.tp1_hit and not self.tp2_hit:
            tp2_hit = (direction == 'long' and current_price >= tp2) or                       (direction == 'short' and current_price <= tp2)

            if tp2_hit:
                logger.info(f"🎯 TP2 ATINGIDO! ${current_price:,.2f} - ATIVANDO TRAILING")

                self.tp2_hit = True
                self.trailing_active = True

                try:
                    self.telegram.send_trailing_activated(self.position, current_price)
                except:
                    pass

                self.save_state()

        # ✅ VERIFICA TP3 (FECHA RESTO) - Corrigido: verifica após TP2
        if self.tp1_hit and self.tp2_hit:
            tp3_hit = (direction == 'long' and current_price >= tp3) or \
                      (direction == 'short' and current_price <= tp3)

            if tp3_hit:
                logger.info(f"🎯 TP3 ATINGIDO! ${current_price:,.2f} - FECHANDO RESTO")

                # Fecha 40% restante
                final_qty = self.position.get('remaining_qty', qty * 0.4)

                if self.is_live_mode:
                    self.close_partial(final_qty, current_price, 'tp3')

                # Calcula PnL final
                if direction == 'long':
                    pnl_pct = ((current_price - entry) / entry) * 100
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100

                final_pnl = (final_qty * entry) * (pnl_pct / 100)

                logger.info(f"✅ Fechou {final_qty} ETH | PnL: ${final_pnl:+,.2f}")

                try:
                    self.telegram.send_partial_tp(self.position, 3, final_qty, final_pnl, pnl_pct)
                except:
                    pass

                # Fecha trade
                self.close_position(current_price, 'take_profit_3')

        # ✅ ATUALIZA TRAILING STOP
        if self.trailing_active:
            trailing_distance = atr * TRAILING_STOP_DISTANCE

            if direction == 'long':
                new_trailing_sl = self.highest_price - trailing_distance

                # Só atualiza se novo SL for maior que atual
                if new_trailing_sl > current_sl:
                    logger.info(f"🔄 Trailing: ${current_sl:,.2f} → ${new_trailing_sl:,.2f}")

                    if self.is_live_mode:
                        self.update_stop_loss(new_trailing_sl)

                    self.position['current_sl'] = new_trailing_sl
                    self.save_state()

            else:  # short
                new_trailing_sl = self.lowest_price + trailing_distance

                # Só atualiza se novo SL for menor que atual
                if new_trailing_sl < current_sl:
                    logger.info(f"🔄 Trailing: ${current_sl:,.2f} → ${new_trailing_sl:,.2f}")

                    if self.is_live_mode:
                        self.update_stop_loss(new_trailing_sl)

                    self.position['current_sl'] = new_trailing_sl
                    self.save_state()

    def check_position_closed(self) -> bool:
        """Verifica se posição foi fechada pelo SL"""
        if not self.position or not self.is_live_mode:
            return False

        try:
            positions = self.rest_client.get_positions(symbol=self.position['symbol'])

            if positions and 'retCode' in positions and positions['retCode'] == 0:
                if 'result' in positions and 'list' in positions['result']:
                    for pos in positions['result']['list']:
                        if pos['symbol'] == self.position['symbol']:
                            size = float(pos.get('size', 0))

                            if size == 0:
                                logger.info("✅ Posição fechada pelo SL")

                                # Busca exit price
                                exit_price = self.get_exit_price_from_history(self.position['symbol'])
                                if exit_price:
                                    self.close_position(exit_price, 'stop_loss')
                                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking position: {e}")
            return False

    def get_exit_price_from_history(self, symbol: str) -> Optional[float]:
        """
        ✅ V5.2: Simplificado - usa current_sl como exit_price
        Quando posição fecha por SL, o exit_price é praticamente igual ao SL configurado
        """
        try:
            if not self.position:
                logger.warning("⚠️ Nenhuma posição ativa")
                return None
        
            # Usa current_sl como exit_price (é o valor onde o SL foi atingido)
            current_sl = self.position.get('current_sl', self.position.get('stop_loss', 0))
        
            if current_sl > 0:
                logger.info(f"✅ Exit price = current SL: ${current_sl:,.2f}")
                return current_sl
        
            # Fallback: usa último preço conhecido
            last_price = self.position.get('last_price', 0)
            if last_price > 0:
                logger.warning(f"⚠️ Usando last_price como fallback: ${last_price:,.2f}")
                return last_price
        
            logger.error("❌ Não foi possível determinar exit_price")
            return None
    
        except Exception as e:
            logger.error(f"❌ Erro ao determinar exit_price: {e}")
        
            # Último fallback: usa SL original
            if self.position:
                sl = self.position.get('stop_loss', 0)
                if sl > 0:
                    logger.warning(f"⚠️ Usando stop_loss original: ${sl:,.2f}")
                    return sl
        
            return None

    def close_position(self, exit_price: float, reason: str):
        """Fecha posição completamente"""
        if not self.position:
            return

        entry = self.position['entry_price']
        direction = self.position['direction']
        total_qty = self.position['qty']

        # Calcula PnL total do trade
        if direction == 'long':
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100

        total_pnl = self.position['size'] * (pnl_pct / 100)
        self.capital += total_pnl

        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': self.position['size'],
            'qty': total_qty,
            'pnl_pct': pnl_pct,
            'pnl_amount': total_pnl,
            'exit_reason': reason,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'ml_confidence': self.position['ml_confidence'],
            'is_live': self.position.get('is_live', False)
        }

        self.trades_history.append(trade)

        result = "✅" if total_pnl > 0 else "❌"
        mode_str = "💰 LIVE" if trade['is_live'] else "📝 PAPER"
        logger.info(f"{result} CLOSED {direction.upper()} {mode_str}")
        logger.info(f"   ${entry:,.2f} → ${exit_price:,.2f}")
        logger.info(f"   PnL: ${total_pnl:+,.2f} ({pnl_pct:+.2f}%)")

        try:
            self.telegram.send_trade_close(trade, total_pnl, pnl_pct)
        except Exception as e:
            logger.error(f"Telegram failed: {e}")

        # Reset
        self.position = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.trailing_active = False
        self.highest_price = None
        self.lowest_price = None
        self.cooldown_until = time.time() + (30 * 60)  # 30min cooldown

        self.save_state()

    def get_stats(self) -> Dict:
        if not self.trades_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            }

        df = pd.DataFrame(self.trades_history)
        total = len(df)
        wins = len(df[df['pnl_amount'] > 0])
        win_rate = wins / total * 100
        total_pnl = df['pnl_amount'].sum()
        avg_pnl = total_pnl / total if total > 0 else 0

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }

    def run(self, symbol: str, check_interval: int = 30):
        """Main loop com monitoramento constante"""
        mode_str = "💰 LIVE MODE" if self.is_live_mode else "📝 PAPER MODE"
        logger.info(f"🚀 MASTER SCALPER BOT - ETH [V6.0 CORREÇÕES CRÍTICAS] ({mode_str})")

        self.telegram.send_message(f"""🤖 BOT ETH INICIADO [V6.0 - CORREÇÕES CRÍTICAS]

Modo: {mode_str}
🎯 Confidence: {self.min_confidence:.0%}
💰 Risco/Trade: {self.risk_per_trade:.1%}

📊 Estratégia Backtest:
• TP1 ({PARTIAL_FRACTION*100:.0f}%): {TP_MULTS['tp1']}x ATR
• Move SL para Breakeven
• TP2: Ativa Trailing ({TRAILING_STOP_DISTANCE}x ATR)
• TP3 ou Trailing: Fecha resto

⏱️ Monitor: {MONITOR_INTERVAL}s quando há posição
✅ V6.0: Correções críticas + Validações + Retry""")

        self.recover_state()

        iteration_count = 0
        last_status_time = time.time()
        last_data_fetch = 0

        try:
            while True:
                try:
                    now = time.time()

                    # Define intervalo: rápido com posição, lento sem
                    current_interval = MONITOR_INTERVAL if self.position else check_interval

                    # ✅ Se há posição, SEMPRE busca preço atual via ticker (rápido)
                    if self.position:
                        try:
                            _, _, last_price = get_best_quotes(self.rest_client, symbol)
                            if last_price and last_price > 0:
                                price = last_price
                            else:
                                # Fallback: usa último preço salvo
                                price = self.position.get('last_price', 0)
                                if price == 0:
                                    logger.warning("⚠️ Não conseguiu obter preço atual")
                                    time.sleep(current_interval)
                                    continue
                        except Exception as e:
                            logger.warning(f"⚠️ Erro ao buscar ticker: {e}")
                            price = self.position.get('last_price', 0)
                            if price == 0:
                                time.sleep(current_interval)
                                continue

                    # Busca dados completos apenas periodicamente (para sinais)
                    if now - last_data_fetch >= check_interval:
                        iteration_count += 1
                        logger.info("="*70)
                        logger.info(f"🔄 ITERATION #{iteration_count} - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                        logger.info("="*70)

                        df = self.get_current_data(symbol, '15m', 30)

                        if df.empty:
                            logger.warning("⚠️ No data")
                            time.sleep(current_interval)
                            continue

                        current = df.iloc[-1]
                        # Se não há posição, usa preço do dataframe
                        if not self.position:
                            price = current['close']
                        last_data_fetch = now

                        logger.info(f"📊 Price: ${price:,.2f}")

                    # Salva último preço
                    if self.position:
                        self.position['last_price'] = price

                    # ✅ MONITORA POSIÇÃO ABERTA
                    if self.position:
                        entry = self.position['entry_price']
                        direction = self.position['direction']

                        if direction == 'long':
                            pnl_pct = ((price - entry) / entry) * 100
                        else:
                            pnl_pct = ((entry - price) / entry) * 100

                        # ✅ V5.2: Fallback para remaining_qty
                        remaining_qty = self.position.get('remaining_qty', self.position['qty'])
                        pnl_usd = (remaining_qty * entry) * (pnl_pct / 100)

                        status_parts = []
                        if self.tp1_hit:
                            status_parts.append("TP1✅")
                        if self.tp2_hit:
                            status_parts.append("TP2✅")
                        if self.trailing_active:
                            status_parts.append("TRAILING🔄")

                        status = " | ".join(status_parts) if status_parts else "Monitorando"

                        logger.info(f"🟢 {direction.upper()} @ ${entry:,.2f} | {status}")
                        logger.info(f"   Remaining: {remaining_qty:.2f} ETH | PnL: ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")

                        # ✅ V5.2: Fallback para current_sl
                        current_sl = self.position.get('current_sl', self.position.get('stop_loss', 0))
                        logger.info(f"   SL: ${current_sl:,.2f}")

                        # Monitora lógica de parciais
                        self.monitor_position(price)

                        # Verifica se SL foi atingido
                        self.check_position_closed()

                    # ✅ BUSCA NOVOS SINAIS
                    else:
                        now = time.time()
                        in_cooldown = now < self.cooldown_until

                        if in_cooldown:
                            remaining = int((self.cooldown_until - now) / 60)
                            logger.info(f"⏳ Cooldown: {remaining}min")

                        else:
                            signal, ml_confidence, current_data = self.get_signal(df)
                            sig_name = 'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRO'
                            passes = ml_confidence >= self.min_confidence
                            status = '✅ PASS' if passes else '❌ FILTERED'

                            logger.info(f"🎯 Signal: {sig_name} | Conf: {ml_confidence:.2%} | {status}")

                            if signal != 0 and passes:
                                logger.info("="*70)
                                logger.info(f"🚨 OPENING {sig_name} POSITION")
                                logger.info("="*70)

                                try:
                                    self.open_position(symbol, signal, current_data, ml_confidence)
                                except Exception as e:
                                    logger.error(f"❌ Failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    self.telegram.send_error(f"Failed: {e}")

                    # Status periódico
                    if time.time() - last_status_time > (4 * 3600):
                        self.telegram.send_status(self.get_stats())
                        last_status_time = time.time()

                    time.sleep(current_interval)

                except Exception as e:
                    logger.error(f"❌ Loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(current_interval if 'current_interval' in locals() else 30)

        except KeyboardInterrupt:
            logger.info("\n⏹️ Bot stopped")
            if self.position:
                logger.info("⚠️ Posição aberta - SL protegerá!")

            stats = self.get_stats()
            logger.info(f"📊 Stats: {stats['total_trades']} trades | WR: {stats['win_rate']:.1f}% | PnL: ${stats['total_pnl']:+,.2f}")
            self.telegram.send_status(stats)


def main():
    global logger

    parser = argparse.ArgumentParser(description='MASTER SCALPER - ETH [V6.0 CORREÇÕES CRÍTICAS]')
    parser.add_argument('--symbol', default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--model', default='ml_model_master_scalper_365d.pkl', help='Model filename')
    parser.add_argument('--interval', type=int, default=30, help='Check interval (no position)')

    args = parser.parse_args()
    config = load_config('standard')
    logger = setup_logging('INFO', log_to_file=True)

    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat = os.getenv('TELEGRAM_CHAT_ID')

    if not telegram_token or not telegram_chat:
        logger.error("❌ Missing Telegram config")
        return

    telegram = TelegramNotifier(telegram_token, telegram_chat)

    model_path = f"storage/models/{args.model}"
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return

    try:
        trader = MasterLiveTrader(config, model_path, telegram, symbol=args.symbol)
        trader.run(args.symbol, args.interval)

    except Exception as e:
        logger.error(f"❌ Fatal: {e}")
        import traceback
        traceback.print_exc()
        telegram.send_error(f"Fatal: {e}")


if __name__ == "__main__":
    main()
