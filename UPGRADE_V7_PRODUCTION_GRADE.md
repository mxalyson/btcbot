# 🚀 UPGRADE V7.0 - PRODUCTION GRADE

## 📋 Resumo Executivo

Transformar os bots ETH e BTC de V6.0 (bons) para V7.0 (production-grade 100%) através de **6 melhorias críticas**:

1. ✅ **get_server_time()** adicionado ao bybit_rest.py
2. ✅ RiskManager com circuit breakers
3. ✅ Reconciliação de posições no startup
4. ✅ Health check antes de operar
5. ✅ Proteção anti-duplicação de ordens
6. ✅ Métricas consolidadas por sessão

---

## 🔧 1. API: get_server_time() (JÁ IMPLEMENTADO)

**Arquivo**: `core/bybit_rest.py`

**Localização**: Final do arquivo, após `set_leverage()`

```python
def get_server_time(self) -> Dict:
    """Get Bybit server time (no authentication required)."""
    return self._request('GET', '/v5/market/time', {})
```

**Status**: ✅ CONCLUÍDO

---

## 🔧 2. RiskManager Integration

### 2.1. Import (ambos os bots)

**Arquivo**: `eth_live_v3.py` e `btc_real_v5.py`

**Localização**: ~linha 59, após imports do core

```python
from core.utils import load_config, setup_logging
from core.bybit_rest import BybitRESTClient
from core.data import DataManager
from core.features import FeatureStore
from core.risk import RiskManager  # ← ADICIONAR
```

### 2.2. Inicialização no __init__

**Localização**: Após `self.rest_client = BybitRESTClient(...)`

```python
        # ✅ V7.0: RiskManager com circuit breakers
        self.risk_manager = RiskManager({
            'initial_capital': self.initial_capital,
            'risk_per_trade': self.risk_per_trade,
            'circuit_breaker_max_loss_pct': float(os.getenv('MAX_DAILY_LOSS_PCT', '5.0')),
            'circuit_breaker_consec_losses': int(os.getenv('MAX_CONSEC_LOSSES', '3')),
            'max_trades_per_day': int(os.getenv('MAX_TRADES_PER_DAY', '5')),
            'cooldown_min': int(os.getenv('COOLDOWN_MIN', '30')),
            'max_positions': 1,
            'fees_taker': FEE_RATE,
            'max_order_value_usdt': 10000
        })

        # Proteção anti-duplicação
        self.order_in_progress = False
```

### 2.3. Variáveis de ambiente (.env)

```bash
# Circuit Breakers
MAX_DAILY_LOSS_PCT=5.0          # Para se perder 5% no dia
MAX_CONSEC_LOSSES=3             # Para após 3 perdas seguidas
MAX_TRADES_PER_DAY=5            # Máximo 5 trades por dia
COOLDOWN_MIN=30                 # 30min entre trades
```

---

## 🔧 3. Health Check

**Localização**: Após `recover_state()`, antes de `get_current_data()`

```python
    def health_check(self) -> bool:
        """✅ V7.0: Verifica saúde do sistema antes de operar"""
        try:
            start = time.time()

            # 1. Testa conexão + latência
            server_time = self.rest_client.get_server_time()
            latency_ms = (time.time() - start) * 1000

            if latency_ms > 500:
                logger.warning(f"⚠️ Alta latência: {latency_ms:.0f}ms")
                return False

            # 2. Verifica clock sync
            server_ts = int(server_time.get('result', {}).get('timeSecond', 0))
            local_ts = int(time.time())
            clock_diff = abs(server_ts - local_ts)

            if clock_diff > 5:
                logger.error(f"❌ Clock desync: {clock_diff}s")
                return False

            # 3. Testa balance (autenticação)
            balance = self.rest_client.get_wallet_balance()
            if not balance.get('result'):
                logger.error(f"❌ Erro ao buscar balance")
                return False

            logger.info(f"✅ Health Check OK (latência: {latency_ms:.0f}ms)")
            return True

        except Exception as e:
            logger.error(f"❌ Health check falhou: {e}")
            return False
```

**Uso**: Chamar no início do `run()`, antes do loop principal

```python
def run(self):
    logger.info(f"🤖 Starting ETH Bot V7.0...")

    # Health check
    if not self.health_check():
        logger.error("❌ Health check falhou - abortando")
        return

    # ... resto do código
```

---

## 🔧 4. Reconciliação de Posições

**Localização**: Após `health_check()`

```python
    def reconcile_positions_on_startup(self):
        """✅ V7.0: Sincroniza posições bot vs exchange"""
        logger.info("🔄 Reconciliando posições com Bybit...")

        try:
            positions = self.rest_client.get_positions(symbol=self.symbol)
            result = positions.get('result', {})
            positions_list = result.get('list', [])

            active_pos = None
            for pos in positions_list:
                if float(pos.get('size', 0)) > 0:
                    active_pos = pos
                    break

            if active_pos and not self.position:
                logger.warning("⚠️ Posição no exchange mas não no bot!")
                logger.warning(f"   Sincronizando...")

                self.position = {
                    'direction': active_pos.get('side', '').lower(),
                    'entry_price': float(active_pos.get('avgPrice', 0)),
                    'qty': float(active_pos.get('size', 0)),
                    'remaining_qty': float(active_pos.get('size', 0)),
                    'symbol': self.symbol,
                    'is_live': self.is_live_mode
                }
                self.save_state()
                logger.info("✅ Sincronizado")

            elif not active_pos and self.position:
                logger.warning("⚠️ Bot tem posição mas exchange não!")
                logger.warning("   Limpando estado local...")

                self.position = None
                self.tp1_hit = False
                self.tp2_hit = False
                self.trailing_active = False
                self.save_state()
                logger.info("✅ Limpo")

            else:
                logger.info("✅ Sincronizado")

        except Exception as e:
            logger.error(f"❌ Reconciliação falhou: {e}")
```

**Uso**: Chamar no `run()` após health_check

```python
def run(self):
    # ... health check ...

    # Reconciliação
    self.reconcile_positions_on_startup()

    # ... resto do código
```

---

## 🔧 5. Proteção Anti-Duplicação

### 5.1. Em open_position() - INÍCIO

**Localização**: Primeira linha do método

```python
def open_position(self, symbol: str, signal: int, current_data: pd.Series, ml_confidence: float):
    # ✅ V7.0: Proteção anti-duplicação
    if self.order_in_progress:
        logger.warning("⚠️ Ordem em progresso - ignorando")
        return

    self.order_in_progress = True

    try:
        # ... código existente (INDENTAR TUDO) ...
        direction = 'long' if signal == 1 else 'short'
        # ... resto do método ...
```

### 5.2. Em open_position() - FINAL

**Localização**: Última linha do método, após `self.save_state()`

```python
        self.save_state()

        # ✅ V7.0: Registra no RiskManager
        self.risk_manager.open_position(symbol, self.position)

    finally:
        self.order_in_progress = False
```

### 5.3. Validação com RiskManager

**Localização**: Após calcular qty/sl/tp, ANTES das validações de segurança

```python
        # ✅ V7.0: Validação RiskManager (circuit breakers)
        is_valid, reason = self.risk_manager.validate_order(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            stop_loss=sl,
            take_profit=tp1,
            position_size=qty
        )

        if not is_valid:
            logger.warning(f"❌ Bloqueado: {reason}")
            return

        # ✅ VALIDAÇÕES DE SEGURANÇA (código existente)
        if qty < min_qty:
            # ...
```

---

## 🔧 6. Métricas Consolidadas

**Localização**: Após `reconcile_positions_on_startup()`

```python
    def get_session_stats(self):
        """✅ V7.0: Métricas consolidadas da sessão"""
        if not self.risk_manager.trade_history:
            logger.info("📊 Nenhum trade nesta sessão")
            return

        stats = self.risk_manager.get_risk_stats()

        logger.info("="*70)
        logger.info("📊 ESTATÍSTICAS DA SESSÃO")
        logger.info("="*70)
        logger.info(f"Capital: ${stats['equity']:,.2f} | Pico: ${stats['peak_equity']:,.2f}")
        logger.info(f"Drawdown: {stats['current_drawdown']:.2f}%")
        logger.info(f"Trades: {stats['total_trades']} | WR: {stats['win_rate']*100:.1f}%")
        logger.info(f"PnL: ${stats['total_pnl']:+,.2f} | Fees: ${stats['total_fees']:,.2f}")
        logger.info(f"Expectância: ${stats['expectancy']:+,.2f}")
        logger.info(f"Streak: {stats['consecutive_wins']} wins / {stats['consecutive_losses']} losses")

        if stats['is_halted']:
            logger.warning(f"🚨 TRADING HALTED: {stats['halt_reason']}")

        logger.info("="*70)
```

**Uso**: Chamar periodicamente no loop (a cada 10 iterações) ou ao fechar trade

```python
# No loop principal, após processar dados
if iteration_count % 10 == 0:
    self.get_session_stats()
```

---

## 🔧 7. Registro de Trades no RiskManager

**Localização**: Em `close_position()`, após atualizar capital e antes de `save_state()`

```python
    def close_position(self, exit_price: float, reason: str):
        # ... código existente de cálculo de PnL ...

        # Atualiza capital
        self.capital += total_pnl

        # ✅ V7.0: Registra trade no RiskManager
        trade_data = {
            'symbol': self.symbol,
            'direction': self.position.get('direction'),
            'entry_price': self.position.get('entry_price'),
            'exit_price': exit_price,
            'position_size': self.position.get('qty'),
            'pnl': total_pnl,
            'exit_reason': reason
        }
        self.risk_manager.record_trade(trade_data)

        # ... resto do código existente ...
        self.position = None
        self.save_state()
```

---

## 📊 CHECKLIST DE IMPLEMENTAÇÃO

### ETH Bot (eth_live_v3.py)

- [x] get_server_time() em bybit_rest.py
- [ ] Import RiskManager
- [ ] Inicializar RiskManager no __init__
- [ ] Adicionar health_check()
- [ ] Adicionar reconcile_positions_on_startup()
- [ ] Adicionar get_session_stats()
- [ ] Modificar open_position() (anti-duplicação + validação)
- [ ] Modificar close_position() (registro de trade)
- [ ] Chamar health_check e reconcile no run()
- [ ] Adicionar variáveis .env

### BTC Bot (btc_real_v5.py)

- [x] get_server_time() em bybit_rest.py (compartilhado)
- [ ] Import RiskManager
- [ ] Inicializar RiskManager no __init__
- [ ] Adicionar health_check()
- [ ] Adicionar reconcile_positions_on_startup()
- [ ] Adicionar get_session_stats()
- [ ] Modificar open_position() (anti-duplicação + validação)
- [ ] Modificar close_position() (registro de trade)
- [ ] Chamar health_check e reconcile no run()

---

## 🎯 BENEFÍCIOS V7.0

| Proteção | V6.0 | V7.0 |
|----------|------|------|
| Circuit Breaker (perdas consecutivas) | ❌ | ✅ |
| Circuit Breaker (loss diário) | ❌ | ✅ |
| Reconciliação startup | ❌ | ✅ |
| Health check | ❌ | ✅ |
| Anti-duplicação | ❌ | ✅ |
| Métricas consolidadas | ❌ | ✅ |
| Max trades/dia | ❌ | ✅ |
| Cooldown entre trades | ❌ | ✅ |

---

## 🚨 IMPORTANTE

1. **Backup**: Sempre faça backup antes de modificar
2. **Testes**: Teste em TESTNET primeiro
3. **Indentação**: Cuidado com Python - 4 espaços
4. **Try-Finally**: Garanta que `order_in_progress` seja sempre resetado
5. **.env**: Adicione as novas variáveis de configuração

---

## 💡 PRÓXIMOS PASSOS (Opcional - Futuro)

- WebSocket para preços (reduz latência)
- Monitoramento de slippage detalhado
- Post-only orders (maker fees)
- Adaptive risk based on drawdown
- Multi-symbol support

---

**Status**: DOCUMENTAÇÃO COMPLETA ✅
**Implementação**: REQUER MODIFICAÇÕES MANUAIS CUIDADOSAS
**Risco**: MÉDIO (modificações estruturais)
**Benefício**: ALTO (production-grade)
