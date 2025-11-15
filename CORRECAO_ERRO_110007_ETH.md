# ✅ CORREÇÃO - Bot ETH: Erro 110007 (Saldo Insuficiente)

## 🔴 Problema Original

```
2025-11-15 16:11:08 [INFO] TradingBot: 💰 Sending LIVE Buy order to Bybit...
2025-11-15 16:11:12 [INFO] TradingBot: ⚠️ Limit IOC did not fill enough — fallback to Market
2025-11-15 16:11:13 [INFO] TradingBot: 📥 API Response: {'retCode': 110007, 'retMsg': 'ab not enough for new order', 'result': {}, 'retExtInfo': {}, 'time': 1763233873465}
2025-11-15 16:11:13 [ERROR] TradingBot: ❌ Failed order: API error
```

**Erro**: `110007 - ab not enough for new order` (saldo insuficiente)

**Problemas**:
- ❌ Bot não verificava saldo antes de tentar abrir posição
- ❌ Erro genérico "Failed order: API error" sem detalhes
- ❌ Telegram não identificava qual bot (ETH ou BTC)
- ❌ Não mostrava quanto de saldo tinha vs quanto precisava

---

## ✅ Correções Implementadas

### 1. Função `get_available_balance()` (Nova)

Verifica saldo disponível em USDT antes de operar:

```python
def get_available_balance(rest) -> float:
    """Get available balance in USDT for trading"""
    wallet = rest.get_wallet_balance(accountType='UNIFIED')
    # Retorna saldo disponível em USDT
```

**Uso**: Consulta API da Bybit para saber quanto USDT está disponível para trading.

---

### 2. Verificação Preventiva de Saldo (Antes de Operar)

**Linha 828-846 de `eth_live_v3.py`**:

```python
if self.is_live_mode:
    # ✅ VERIFICAÇÃO PREVENTIVA DE SALDO
    available_balance = get_available_balance(self.rest_client)
    margin_needed = actual_size_usd * 1.2  # 20% de margem de segurança

    if available_balance < margin_needed:
        error_msg = f"⚠️ SALDO INSUFICIENTE - Ordem bloqueada preventivamente!\n\n"
        error_msg += f"Bot: ETH Live V3\n"
        error_msg += f"Saldo disponível: ${available_balance:,.2f} USDT\n"
        error_msg += f"Margem necessária: ${margin_needed:,.2f} USDT\n"
        error_msg += f"Tentando operar: ${actual_size_usd:,.2f} USDT\n"
        # ... notifica no Telegram e retorna
```

**Comportamento**:
- ✅ Verifica saldo ANTES de tentar fazer ordem
- ✅ Exige margem de segurança de 20% (para cobrir fees e slippage)
- ✅ Bloqueia ordem preventivamente se saldo insuficiente
- ✅ Notifica no Telegram com detalhes completos

---

### 3. Tratamento Específico do Erro 110007

**Linha 894-908 de `eth_live_v3.py`**:

Se mesmo assim a ordem falhar com erro 110007, agora captura especificamente:

```python
if ret_code == 110007:
    available = get_available_balance(self.rest_client)
    error_detail = f"❌ SALDO INSUFICIENTE!\n\n"
    error_detail += f"Bot: ETH Live V3\n"
    error_detail += f"Erro: {ret_msg}\n"
    error_detail += f"Saldo disponível: ${available:,.2f} USDT\n"
    error_detail += f"Tentando operar: ${actual_size_usd:,.2f} USDT\n"
    error_detail += f"Direção: {direction.upper()}\n"
    error_detail += f"Symbol: {symbol}\n\n"
    error_detail += f"💡 Ação: Deposite mais fundos ou reduza o tamanho das posições!"

    logger.error(error_detail)
    self.telegram.send_error(error_detail)
```

**Comportamento**:
- ✅ Detecta erro 110007 especificamente
- ✅ Mostra saldo disponível vs necessário
- ✅ Identifica o bot (ETH Live V3)
- ✅ Sugere ação clara (depositar ou reduzir)

---

### 4. Mensagens de Erro Melhoradas

**Linha 912-920 de `eth_live_v3.py`**:

Para outros erros de API:

```python
except Exception as e:
    error_detail = f"❌ BOT ETH - FALHA NA ORDEM\n\n"
    error_detail += f"Erro: {str(e)}\n"
    error_detail += f"Symbol: {symbol}\n"
    error_detail += f"Direção: {direction.upper()}\n"
    error_detail += f"Tamanho: ${actual_size_usd:,.2f} USDT"

    logger.error(error_detail)
    self.telegram.send_error(error_detail)
```

**Comportamento**:
- ✅ Identifica "BOT ETH" claramente
- ✅ Mostra símbolo, direção e tamanho
- ✅ Log detalhado no console
- ✅ Notificação no Telegram formatada

---

## 📱 Exemplos de Mensagens no Telegram

### Caso 1: Saldo Insuficiente Detectado Preventivamente

```
⚠️ SALDO INSUFICIENTE - Ordem bloqueada preventivamente!

Bot: ETH Live V3
Saldo disponível: $50.00 USDT
Margem necessária: $120.00 USDT
Tentando operar: $100.00 USDT
Symbol: ETHUSDT
Direção: LONG

💡 Deposite mais fundos ou ajuste o tamanho das posições!
```

---

### Caso 2: Erro 110007 da API (se passar a verificação preventiva)

```
❌ SALDO INSUFICIENTE!

Bot: ETH Live V3
Erro: ab not enough for new order
Saldo disponível: $45.00 USDT
Tentando operar: $100.00 USDT
Direção: LONG
Symbol: ETHUSDT

💡 Ação: Deposite mais fundos ou reduza o tamanho das posições!
```

---

### Caso 3: Outro Erro de API

```
❌ BOT ETH - FALHA NA ORDEM

Erro: API error (code 10001): Invalid symbol
Symbol: ETHUSDT
Direção: LONG
Tamanho: $100.00 USDT
```

---

## 🔧 Como Funciona Agora

### Fluxo de Abertura de Posição:

1. **Calcular tamanho da posição**
   ```
   Position: 0.0348 ETH = $100.00
   ```

2. **✅ NOVO: Verificar saldo ANTES de operar**
   ```
   Saldo verificado: $150.00 USDT disponível
   Margem necessária: $120.00 ($100 * 1.2)
   ✅ OK para operar
   ```

3. **Tentar ordem Limit IOC**
   ```
   Sending LIVE Buy order to Bybit...
   ```

4. **Fallback para Market se necessário**
   ```
   ⚠️ Limit IOC did not fill enough — fallback to Market
   ```

5. **✅ NOVO: Verificar retCode da resposta**
   ```python
   if retCode == 110007:
       # Tratamento específico de saldo insuficiente
   elif retCode != 0:
       # Outros erros
   else:
       # Sucesso
   ```

6. **Configurar SL e salvar posição**

---

## 📊 Logs no Console (Exemplo)

### Antes (Erro Genérico):
```
2025-11-15 16:11:13 [INFO] TradingBot: 📥 API Response: {'retCode': 110007, ...}
2025-11-15 16:11:13 [ERROR] TradingBot: ❌ Failed order: API error
```

### Agora (Detalhado):
```
2025-11-15 16:11:13 [INFO] TradingBot: ✅ Saldo verificado: $50.00 USDT disponível
2025-11-15 16:11:13 [ERROR] TradingBot: ⚠️ SALDO INSUFICIENTE - Ordem bloqueada preventivamente!

Bot: ETH Live V3
Saldo disponível: $50.00 USDT
Margem necessária: $120.00 USDT
Tentando operar: $100.00 USDT
Symbol: ETHUSDT
Direção: LONG

💡 Deposite mais fundos ou ajuste o tamanho das posições!
```

---

## 🛡️ Margem de Segurança

O bot agora exige **20% a mais de saldo** do que o tamanho da posição:

```
Tamanho da posição: $100
Margem necessária: $120 (100 * 1.2)
Razão: Cobrir fees (0.06%) + slippage + margem extra
```

**Por que 20%?**
- Fees de abertura: ~0.06%
- Fees de fechamento: ~0.06%
- Slippage: ~0.1-0.5%
- Margem extra: ~19%
- **Total**: 20% de margem garante que a ordem será executada

---

## ⚙️ Configuração do Tamanho da Posição

Se estiver tendo problemas de saldo insuficiente repetidamente, ajuste no `.env`:

```bash
# Reduzir tamanho de posição por trade
TRADE_SIZE_USD=50  # Era 100, agora 50

# Ou reduzir risco por trade
RISK_PER_TRADE_PCT=0.5  # Era 1.0, agora 0.5
```

---

## 🧪 Testes Realizados

- ✅ Verificação de saldo funciona corretamente
- ✅ Bloqueio preventivo quando saldo < margem
- ✅ Tratamento de erro 110007 funciona
- ✅ Mensagens no Telegram são enviadas
- ✅ Logs detalhados no console
- ✅ Bot continua funcionando após erro (não crasha)

---

## 📝 Commits

**Commit**: `b90a708`
```
[FIX] Tratamento de erro 110007 (saldo insuficiente) no bot ETH

1. Função get_available_balance() - verifica saldo USDT
2. Verificação preventiva antes de operar (margem 20%)
3. Tratamento específico erro 110007 com detalhes
4. Mensagens melhoradas identificando "Bot: ETH Live V3"
```

---

## ✅ Próximos Passos

1. **Atualizar o bot localmente**:
   ```bash
   git pull origin claude/review-eth-live-v3-01RKAjVHWP3QUqeSDiA5h2Zs
   ```

2. **Verificar saldo na Bybit**:
   - Se necessário, depositar mais USDT
   - Ou ajustar `TRADE_SIZE_USD` no `.env`

3. **Reiniciar o bot**:
   ```bash
   python eth_live_v3.py --mode live
   ```

4. **Monitorar logs**:
   - Verificar se aparece "✅ Saldo verificado: $XXX USDT disponível"
   - Se houver erro, receberá notificação detalhada no Telegram

---

## ❓ FAQ

### P: Por que o bot precisa de 20% a mais de saldo?
**R**: Para cobrir fees (0.12%), slippage (~0.5%) e ter margem de segurança. Melhor prevenir do que receber erro 110007.

### P: E se eu quiser operar com menos margem?
**R**: Pode ajustar na linha 830 de `eth_live_v3.py`:
```python
margin_needed = actual_size_usd * 1.1  # 10% ao invés de 20%
```

### P: O erro 110007 ainda pode acontecer?
**R**: Teoricamente não, pois verificamos preventivamente. Mas se acontecer (race condition, saldo foi usado por outro bot), agora é tratado corretamente.

### P: Funciona para BTC também?
**R**: Esta correção é específica para `eth_live_v3.py`. Se `btc_real_v5.py` tiver mesmo problema, precisa aplicar correção similar.

---

🎉 **Bot ETH agora está protegido contra erro de saldo insuficiente!**
