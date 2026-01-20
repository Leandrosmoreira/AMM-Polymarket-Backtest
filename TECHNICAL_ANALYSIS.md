# Análise Técnica Completa: Polymarket Bot

## 1. O QUE É O PROJETO?

### Visão Geral

Sistema de trading automatizado para mercados de **previsão binária de 15 minutos** no Polymarket, focado em:

- **BTC** (Bitcoin Up/Down)
- **ETH** (Ethereum Up/Down)
- **SOL** (Solana Up/Down)

Cada mercado tem dois tokens:
- **YES (UP)**: Paga $1 se preço subir
- **NO (DOWN)**: Paga $1 se preço cair

### Mercado de 15 Minutos

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CICLO DE MERCADO 15min                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  12:00:00  ──────────────────────────────────────────  12:15:00    │
│     │                                                       │       │
│   ABRE                     TRADING                       FECHA     │
│  (novos tokens)                                      (resolve)     │
│                                                                     │
│  Snapshot BTC: $50,000                        BTC Final: $50,100   │
│                                                                     │
│  Resultado: YES ganha (subiu)                                      │
│  - Holders de YES recebem $1 por share                            │
│  - Holders de NO recebem $0                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. ARQUITETURA DO SISTEMA

```
┌─────────────────────────────────────────────────────────────────────┐
│                         POLYMARKET BOT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │   DISCOVERY     │    │    TRADING      │    │   ANALYTICS     │ │
│  │                 │    │                 │    │                 │ │
│  │ • markets.py    │    │ • trading.py    │    │ • fast_logger   │ │
│  │ • lookup.py     │    │ • auth.py       │    │ • statistics    │ │
│  │ • wss_market    │    │ • order_mgr     │    │ • detailed_log  │ │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘ │
│           │                      │                      │          │
│           └──────────────────────┼──────────────────────┘          │
│                                  │                                  │
│  ┌───────────────────────────────┴───────────────────────────────┐ │
│  │                         BOTS                                   │ │
│  │                                                                │ │
│  │   ┌──────────────────┐         ┌──────────────────┐           │ │
│  │   │     BOT 1        │         │     BOT 2        │           │ │
│  │   │   ARBITRAGE      │         │  MARKET MAKER    │           │ │
│  │   │                  │         │                  │           │ │
│  │   │ • multi_bot.py   │         │ • market_maker   │           │ │
│  │   │ • bot.py         │         │ • mm/inventory   │           │ │
│  │   │                  │         │ • mm/volatility  │           │ │
│  │   │ Estratégia:      │         │ • mm/delta_hedge │           │ │
│  │   │ Compra YES+NO    │         │                  │           │ │
│  │   │ quando soma<$1   │         │ Estratégia:      │           │ │
│  │   │                  │         │ Quote bid/ask    │           │ │
│  │   │ Lucro: Garantido │         │ como maker       │           │ │
│  │   │ Risco: ~Zero     │         │                  │           │ │
│  │   │                  │         │ Lucro: Spread    │           │ │
│  │   └──────────────────┘         │ Risco: Inventory │           │ │
│  │                                └──────────────────┘           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                      PERFORMANCE                               │ │
│  │  • uvloop (2-4x async)  • orjson (10x JSON)  • PyPy (5-10x)   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. OS DOIS BOTS

### Bot 1: Arbitragem (Taker)

**Estratégia:** Compra YES e NO quando a soma é menor que $1.

```
Exemplo:
  YES price: $0.48
  NO price:  $0.50
  ─────────────────
  Total:     $0.98  (< $1.00)

  Ação: Comprar 10 YES + 10 NO
  Custo: $9.80
  Retorno garantido: $10.00 (um dos dois paga $1)
  Lucro: $0.20 (2.04%)
```

| Aspecto | Valor |
|---------|-------|
| Risco | ~Zero (arbitragem pura) |
| Lucro por trade | 0.5% - 2% |
| Frequência | Quando oportunidade aparece |
| Capital necessário | Baixo |
| Tipo | TAKER (consome liquidez) |

### Bot 2: Market Maker (Maker)

**Estratégia:** Fornece liquidez colocando ordens no book.

```
Orderbook:
  ASK $0.52 x 100  ← Outros
  ASK $0.51 x 50   ← Outros
  ASK $0.505 x 10  ← NOSSO (vender)
  ─────────────────
  BID $0.495 x 10  ← NOSSO (comprar)
  BID $0.49 x 50   ← Outros
  BID $0.48 x 100  ← Outros

  Spread capturado: $0.505 - $0.495 = $0.01 (1%)
```

| Aspecto | Valor |
|---------|-------|
| Risco | Inventory (exposição a um lado) |
| Lucro por trade | Spread (1-3%) |
| Frequência | Contínuo |
| Capital necessário | Médio/Alto |
| Tipo | MAKER (fornece liquidez) |

---

## 4. MÓDULOS TÉCNICOS

### 4.1 Inventory Manager (`mm/inventory.py`)

**Função:** Controla exposição para não ficar desbalanceado.

```python
# Quando muito exposto a YES:
adjustment = inventory.get_size_adjustment("btc")
# Retorna: yes_mult=0.5, no_mult=1.5

# Próximas ordens:
yes_size = 10 * 0.5  # = 5 (comprar menos YES)
no_size = 10 * 1.5   # = 15 (comprar mais NO)
```

**Parâmetros:**
- `max_exposure_per_market`: Máxima exposição em $ (default: 100)
- `max_imbalance`: Máximo desbalanceamento permitido (default: 30%)
- `rebalance_aggression`: Quão agressivo rebalancear (default: 0.5)

### 4.2 Volatility Engine (`mm/volatility.py`)

**Função:** Calcula volatilidade em tempo real e ajusta spread.

```python
vol = VolatilityEngine(lookback=100)
vol.update(price=0.50, spread=0.02)

rec = vol.get_recommendations()
# rec.spread_multiplier = 1.5 (aumentar spread 50%)
# rec.size_multiplier = 0.7 (reduzir tamanho 30%)
# rec.should_quote = True
# rec.regime = "high"
```

**Regimes de Volatilidade:**

| Regime | Vol Price | Spread Mult | Size Mult | Quotar? |
|--------|-----------|-------------|-----------|---------|
| low | < 0.5% | 0.8x | 1.2x | Sim |
| normal | 0.5-2% | 1.0x | 1.0x | Sim |
| high | 2-5% | 1.5-2x | 0.7x | Sim |
| extreme | > 5% | 3x | 0.3x | **Não** |

### 4.3 Order Manager (`mm/order_manager.py`)

**Função:** Gerencia ordens com foco em baixa latência.

**Features:**
- Pool de ordens pré-assinadas
- Rate limiting automático
- Cancelamento em batch
- Tracking de fills

```python
# Pré-assinar ordens (lento, mas feito antes)
await manager.pre_sign_orders([
    {"token": yes_token, "side": "BUY", "price": 0.48, "size": 10},
    {"token": yes_token, "side": "SELL", "price": 0.52, "size": 10},
])

# Enviar ordem pré-assinada (muito rápido!)
order_id = await manager.submit_pre_signed("YES", "BUY", 0.48)
```

### 4.4 Delta Hedger (`mm/delta_hedge.py`)

**Função:** Mantém posição delta-neutral.

```python
hedger = DeltaHedger(max_delta=50)

# Quando fill acontece
hedger.update_position("btc", "YES", size_delta=10, price=0.48)

# Verificar se precisa hedge
if hedger.needs_urgent_hedge("btc"):
    hedge_order = hedger.get_hedge_order("btc")
```

### 4.5 LTM - Liquidity Time Model (`ltm/`)

**Função:** Modelo que analisa padrões de liquidez ao longo do tempo do mercado.

**Componentes:**
- `policy.py`: Políticas de trading por bucket de tempo
- `decay.py`: Modelo de decay do pair-cost
- `bandit.py`: Multi-armed bandit para otimização
- `features.py`: Extração de features do mercado
- `collector.py`: Coleta de dados históricos

---

## 5. SISTEMA DE LOGGING E BACKTEST

### 5.1 Fast Logger (`fast_logger.py`)

**Formato:** JSONL (JSON Lines) - uma linha por evento

```json
{"ts":1737340800.123,"time":"2026-01-20T12:00:00","market":"btc-updown-15m-1737340800","event":"trade","price_up":0.48,"price_down":0.50,"pair_cost":0.98,"profit_pct":2.04,"order_size":5}
{"ts":1737340801.456,"time":"2026-01-20T12:00:01","market":"btc-updown-15m-1737340800","event":"scan","up_ask":0.49,"down_ask":0.51,"pair_cost":1.00,"has_opportunity":false}
```

### 5.2 Detailed Logger (`mm/detailed_logger.py`)

**Eventos logados:**

| Evento | Dados |
|--------|-------|
| STARTUP | mode, assets, settings |
| MARKET_DISCOVERED | slug, time_remaining, tokens |
| ORDERBOOK_UPDATE | best_bid, best_ask, depth |
| QUOTE_CALCULATED | bid_price, ask_price, sizes |
| ORDER_SENT | side, price, size, order_id |
| ORDER_FILLED | side, price, filled_size, value |
| INVENTORY_UPDATE | yes_exposure, no_exposure, imbalance |
| REBALANCE | multipliers, reason |
| ERROR | error, details |
| SHUTDOWN | total_quotes, fills, pnl |

### 5.3 Backtest com JSONL - SIM, É POSSÍVEL!

O JSONL permite reconstruir toda a história de trading para backtest.

```python
# Carregar dados do JSONL
import json

trades = []
with open("logs/trades_20260120_120000.jsonl") as f:
    for line in f:
        trades.append(json.loads(line))

# Analisar
df = pd.DataFrame(trades)
df['datetime'] = pd.to_datetime(df['ts'], unit='s')

# Métricas
total_invested = df['investment'].sum()
total_profit = df['expected_profit'].sum()
win_rate = (df['profit_pct'] > 0).mean()
```

---

## 6. POSSIBILIDADES DE BACKTEST

### 6.1 Backtest Histórico (dados passados)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKTEST HISTÓRICO                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Fonte de dados:                                                    │
│  ├── JSONL do próprio bot (logs/)                                  │
│  ├── API histórica do Polymarket                                   │
│  └── WebSocket recordings                                           │
│                                                                     │
│  Processo:                                                          │
│  1. Carregar orderbook histórico                                   │
│  2. Simular estratégia do bot                                      │
│  3. Calcular métricas:                                             │
│     • PnL total                                                     │
│     • Sharpe ratio                                                 │
│     • Max drawdown                                                  │
│     • Win rate                                                      │
│     • Avg profit per trade                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Simulação Monte Carlo

```python
# Simular diferentes cenários de mercado
scenarios = monte_carlo_simulate(
    n_simulations=10000,
    market_volatility=[0.01, 0.05, 0.10],
    spread_distribution="normal",
    fill_probability=0.7,
)

# Calcular VaR (Value at Risk)
var_95 = np.percentile(scenarios['pnl'], 5)
```

### 6.3 Walk-Forward Optimization

```
┌────────────────────────────────────────────────────────────────────┐
│                  WALK-FORWARD OPTIMIZATION                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Período 1: Train [Jan-Mar] → Test [Abr]                          │
│  Período 2: Train [Fev-Abr] → Test [Mai]                          │
│  Período 3: Train [Mar-Mai] → Test [Jun]                          │
│  ...                                                               │
│                                                                    │
│  Otimizar parâmetros:                                             │
│  • TARGET_PAIR_COST (threshold de arbitragem)                     │
│  • ORDER_SIZE (tamanho das ordens)                                │
│  • MM_BASE_SPREAD (spread do market maker)                        │
│  • REBALANCE_AGGRESSION (agressividade do rebalanceamento)        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 7. MODELOS E ESTRATÉGIAS POSSÍVEIS

### 7.1 Estratégias Implementadas

| Estratégia | Bot | Risco | Retorno |
|------------|-----|-------|---------|
| Arbitragem Pura | Bot 1 | ~Zero | 0.5-2% |
| Market Making | Bot 2 | Médio | 1-5% |
| Inventory Neutral | Bot 2 | Baixo | 0.5-2% |

### 7.2 Estratégias Possíveis (Futuras)

| Estratégia | Descrição | Complexidade |
|------------|-----------|--------------|
| **Momentum** | Comprar lado que está subindo | Média |
| **Mean Reversion** | Apostar em reversão quando muito desequilibrado | Média |
| **Cross-Market Arb** | Arbitrar entre BTC, ETH, SOL | Alta |
| **Time Decay** | Explorar decay do preço perto do fechamento | Média |
| **Sentiment Analysis** | Usar dados externos (Twitter, etc) | Alta |
| **ML Price Prediction** | Prever direção do preço | Muito Alta |

### 7.3 Modelo de Machine Learning (Futuro)

```python
# Features possíveis
features = {
    'time_remaining': 720,  # segundos até fechar
    'yes_price': 0.48,
    'no_price': 0.52,
    'spread': 0.04,
    'volume_ratio': 1.2,
    'btc_price_change_1m': 0.001,
    'order_imbalance': 0.15,
    'volatility': 0.02,
}

# Target
target = 'market_resolved_yes'  # 0 ou 1

# Modelo
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Previsão
prob_yes = model.predict_proba(features)[0][1]
```

---

## 8. FLUXO DE DADOS

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  POLYMARKET API                                                     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                   │
│  │  Discovery  │ ─── Encontra mercados ativos (BTC, ETH, SOL)      │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │  Orderbook  │ ─── Busca preços bid/ask em tempo real           │
│  └──────┬──────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐    ┌─────────────┐                               │
│  │  Volatility │───▶│   Quotes    │ ─── Calcula bid/ask ideais    │
│  │   Engine    │    │ Calculator  │                                │
│  └─────────────┘    └──────┬──────┘                               │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────┐    ┌─────────────┐                               │
│  │  Inventory  │◀──▶│   Order     │ ─── Envia/cancela ordens      │
│  │   Manager   │    │   Manager   │                                │
│  └─────────────┘    └──────┬──────┘                               │
│                            │                                        │
│                            ▼                                        │
│  ┌─────────────┐    ┌─────────────┐                               │
│  │   Logger    │◀───│    Fills    │ ─── Execuções e resultados    │
│  │   (JSONL)   │    │  Callback   │                                │
│  └──────┬──────┘    └─────────────┘                               │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                   │
│  │  Backtest   │ ─── Análise offline dos dados                    │
│  │   Engine    │                                                   │
│  └─────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. STACK TECNOLÓGICA

### Performance

| Tecnologia | Ganho | Uso |
|------------|-------|-----|
| **PyPy** | 5-10x | JIT compilation |
| **uvloop** | 2-4x | Async event loop |
| **orjson** | 10x | JSON serialization |
| **msgspec** | 12x | Struct serialization |
| **httpx[http2]** | 2x | HTTP multiplexing |

### Dependências

```
py-clob-client>=0.18.0    # Polymarket API client
python-dotenv             # Environment variables
httpx[http2]              # HTTP client
uvloop                    # Fast event loop (Linux/Mac)
orjson                    # Fast JSON
msgspec                   # Fast serialization (optional)
pandas                    # Data analysis
numpy                     # Numerical computing
```

---

## 10. LIMITAÇÕES E RISCOS

### Limitações Técnicas

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| Latência API | 50-200ms | Ordens pré-assinadas |
| Rate limits | Max 10 req/s | Rate limiter interno |
| WebSocket instável | Desconexões | Reconexão automática |
| Slippage | Preço diferente do esperado | FOK orders |

### Riscos de Trading

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Inventory desbalanceado | Alta | Médio | Inventory Manager |
| Volatilidade extrema | Média | Alto | Stop quoting |
| API down | Baixa | Alto | Circuit breaker |
| Bug no código | Média | Alto | DRY_RUN mode |

---

## 11. ROADMAP FUTURO

### Curto Prazo
- [ ] Integrar detailed_logger no market_maker_bot
- [ ] Adicionar WebSocket para orderbook real-time
- [ ] Criar script de backtest com JSONL

### Médio Prazo
- [ ] Dashboard web para monitoramento
- [ ] Alertas via Telegram/Discord
- [ ] Multi-account support
- [ ] Cross-market arbitrage

### Longo Prazo
- [ ] Machine Learning para previsão
- [ ] Sentiment analysis integration
- [ ] Auto-parameter optimization
- [ ] Estratégias de portfolio

---

## 12. COMO USAR PARA BACKTEST

### Passo 1: Coletar Dados

```bash
# Rodar bot em DRY_RUN para coletar dados
DRY_RUN=true python -m polymarket_bot --market-maker

# Logs gerados em:
# logs/trades_YYYYMMDD_HHMMSS.jsonl
# logs/mm_detailed_YYYYMMDD_HHMMSS.jsonl
```

### Passo 2: Analisar JSONL

```python
import pandas as pd
import json

# Carregar
events = []
with open("logs/mm_detailed_20260120_120000.jsonl") as f:
    for line in f:
        events.append(json.loads(line))

df = pd.DataFrame(events)

# Filtrar trades
trades = df[df['event'] == 'ORDER_FILLED']

# Métricas
print(f"Total trades: {len(trades)}")
print(f"Total volume: ${trades['fill_value'].sum():.2f}")
print(f"Avg trade size: ${trades['fill_value'].mean():.2f}")
```

### Passo 3: Simular Estratégias

```python
# Replay dos eventos
for event in events:
    if event['event'] == 'ORDERBOOK_UPDATE':
        # Simular decisão do bot
        should_trade = strategy.evaluate(event)

    elif event['event'] == 'ORDER_FILLED':
        # Atualizar PnL
        pnl.update(event)

print(f"Final PnL: ${pnl.total:.2f}")
```

---

## RESUMO

| Aspecto | Status |
|---------|--------|
| **Bot 1 (Arbitrage)** | ✅ Completo |
| **Bot 2 (Market Maker)** | ✅ Completo |
| **Multi-Market (BTC, ETH, SOL)** | ✅ Completo |
| **Inventory Manager** | ✅ Completo |
| **Volatility Engine** | ✅ Completo |
| **JSONL Logging** | ✅ Completo |
| **Detailed Logger** | ✅ Completo |
| **Performance Optimizations** | ✅ Completo |
| **Backtest Engine** | 🔄 Possível com JSONL |
| **ML Integration** | 📋 Futuro |
| **Dashboard Web** | 📋 Futuro |
