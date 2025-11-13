"""Utilities: logging, config loading, helpers."""
import os
import logging
import random
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import colorlog

def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """Configure colored logging."""
    logger = logging.getLogger("TradingBot")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler with colors
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    ))
    logger.addHandler(handler)

    # File handler
    if log_to_file:
        log_dir = Path("storage/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "bot.log", encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        ))
        logger.addHandler(fh)

    return logger

def load_config(profile: str = "standard") -> Dict[str, Any]:
    """Load configuration from .env and profile."""

    # FIX: Tentar múltiplos encodings para o .env
    env_file = Path('.env')
    if env_file.exists():
        encodings = ['utf-8', 'latin-1', 'cp1252']
        loaded = False

        for encoding in encodings:
            try:
                load_dotenv(encoding=encoding)
                loaded = True
                break
            except (UnicodeDecodeError, Exception):
                continue

        if not loaded:
            print("⚠️  WARNING: Could not load .env file with any encoding")
            print("   Using default configuration")

    # Load profile overrides
    profile_path = Path(f"configs/profile_{profile}.env")
    if profile_path.exists():
        # Tentar múltiplos encodings também para profiles
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                load_dotenv(profile_path, override=True, encoding=encoding)
                break
            except:
                continue

    config = {
        # Execution
        'execution_mode': os.getenv('EXECUTION_MODE', 'paper'),
        'bybit_api_key': os.getenv('BYBIT_API_KEY', ''),
        'bybit_api_secret': os.getenv('BYBIT_API_SECRET', ''),
        'bybit_testnet': os.getenv('BYBIT_TESTNET', 'true').lower() == 'true',

        # Trading
        'symbols': os.getenv('SYMBOLS', 'BTCUSDT').split(','),
        'timeframes': os.getenv('TIMEFRAMES', '15m,1h,4h').split(','),
        'primary_timeframe': os.getenv('PRIMARY_TIMEFRAME', '15m'),

        # Risk
        'profile': profile,
        'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.0075')),
        'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000')),
        'atr_mult_sl': float(os.getenv('ATR_MULT_SL', '1.8')),
        'atr_mult_tp1': float(os.getenv('ATR_MULT_TP1', '0.75')),
        'atr_mult_tp2': float(os.getenv('ATR_MULT_TP2', '1.0')),
        'atr_mult_tp3': float(os.getenv('ATR_MULT_TP3', '1.5')),
        'tp_split': [float(x) for x in os.getenv('TP_SPLIT', '50,30,20').split(',')],
        'use_trailing': os.getenv('USE_TRAILING', 'true').lower() == 'true',
        'trailing_type': os.getenv('TRAILING_TYPE', 'ATR'),
        'max_positions': int(os.getenv('MAX_POSITIONS', '3')),
        'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '3')),
        'cooldown_min': int(os.getenv('COOLDOWN_MIN', '60')),

        # Signals
        'conf_threshold': float(os.getenv('CONF_THRESHOLD', '0.65')),
        'adx_min': float(os.getenv('ADX_MIN', '25')),
        'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
        'rsi_ob': float(os.getenv('RSI_OB', '70')),
        'rsi_os': float(os.getenv('RSI_OS', '30')),
        'vol_mult_min': float(os.getenv('VOL_MULT_MIN', '1.2')),
        'min_spread_ticks': int(os.getenv('MIN_SPREAD_TICKS', '5')),
        'max_spread_ticks': int(os.getenv('MAX_SPREAD_TICKS', '20')),

        # Microstructure
        'obi_levels': int(os.getenv('OBI_LEVELS', '5')),
        'obi_threshold': float(os.getenv('OBI_THRESHOLD', '0.3')),
        'orderflow_lookback': int(os.getenv('ORDERFLOW_LOOKBACK', '50')),

        # Fees & Slippage
        'fees_taker': float(os.getenv('FEES_TAKER', '0.0006')),
        'fees_maker': float(os.getenv('FEES_MAKER', '0.0002')),
        'slippage_ticks': int(os.getenv('SLIPPAGE_TICKS', '2')),
        'simulated_latency_ms': int(os.getenv('SIMULATED_LATENCY_MS', '50')),

        # Logging
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_to_file': os.getenv('LOG_TO_FILE', 'true').lower() == 'true',

        # Guards
        'live_confirm_each_order': os.getenv('LIVE_CONFIRM_EACH_ORDER', 'true').lower() == 'true',
        'max_order_value_usdt': float(os.getenv('MAX_ORDER_VALUE_USDT', '1000')),
        'max_price_deviation_pct': float(os.getenv('MAX_PRICE_DEVIATION_PCT', '2.0')),
        'circuit_breaker_max_loss_pct': float(os.getenv('CIRCUIT_BREAKER_MAX_LOSS_PCT', '5.0')),
        'circuit_breaker_consec_losses': int(os.getenv('CIRCUIT_BREAKER_CONSEC_LOSSES', '5')),
    }

    return config

def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds."""
    multiplier = int(tf[:-1])
    unit = tf[-1]

    if unit == 'm':
        return multiplier * 60
    elif unit == 'h':
        return multiplier * 3600
    elif unit == 'd':
        return multiplier * 86400
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")

def round_to_tick(price: float, tick_size: float) -> float:
    """Round price to valid tick size."""
    return round(price / tick_size) * tick_size

def round_to_lot(qty: float, lot_size: float) -> float:
    """Round quantity to valid lot size."""
    return round(qty / lot_size) * lot_size
