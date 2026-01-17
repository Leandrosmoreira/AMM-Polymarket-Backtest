# Bot 2: Market Maker Pro - Plano de Desenvolvimento

## Visão Geral

O **Bot 2** é uma evolução do bot de arbitragem para um **Market Maker** completo, inspirado nas estratégias do Gabagool.

```
Bot 1 (Arbitrage)          Bot 2 (Market Maker)
─────────────────────────────────────────────────
Taker (bate ordens)    →   Maker (cria liquidez)
1 mercado              →   Multi-mercado
HTTP polling           →   WebSocket tempo real
Espera oportunidade    →   Cria oportunidade
~2% profit/trade       →   ~0.5% profit/trade (mais volume)
```

---

## Arquitetura

```
trading_bot_ltm/
├── simple_arb_bot.py      # Bot 1 - Arbitragem (atual)
├── market_maker_bot.py    # Bot 2 - Market Maker (novo)
├── shared/
│   ├── volatility.py      # Módulo de volatilidade
│   ├── fast_ws.py         # WebSocket otimizado
│   ├── order_manager.py   # Gerenciador de ordens
│   └── delta_hedge.py     # Delta hedging
└── config.py              # Configuração unificada
```

---

## Módulos do Bot 2

### 1. Volatility Engine (`volatility.py`)

```python
class VolatilityEngine:
    """
    Calcula volatilidade em tempo real e ajusta parâmetros.

    Métricas:
    - Rolling std dos últimos N preços
    - ATR (Average True Range)
    - Spread médio vs atual

    Output:
    - volatility_score: 0-100
    - recommended_spread: ajuste do spread
    - recommended_size: ajuste do tamanho
    """

    def update(self, price: float, timestamp: float):
        """Atualiza com novo preço."""

    def get_recommendations(self) -> dict:
        """Retorna ajustes recomendados."""
        return {
            'volatility_score': 45,
            'spread_multiplier': 1.2,  # 20% mais largo
            'size_multiplier': 0.8,    # 20% menor
            'should_quote': True,
        }
```

### 2. Order Manager (`order_manager.py`)

```python
class OrderManager:
    """
    Gerencia ordens ativas e pré-assinadas.

    Features:
    - Pool de ordens pré-assinadas
    - Cancelamento rápido em batch
    - Tracking de fills
    - Rate limiting
    """

    def pre_sign_orders(self, orders: list) -> list:
        """Pré-assina ordens para envio rápido."""

    def submit_order(self, signed_order) -> str:
        """Envia ordem pré-assinada (< 10ms)."""

    def cancel_all(self) -> bool:
        """Cancela todas ordens ativas."""

    def get_active_orders(self) -> dict:
        """Retorna ordens ativas por mercado."""
```

### 3. Delta Hedger (`delta_hedge.py`)

```python
class DeltaHedger:
    """
    Mantém posição delta-neutral.

    Se acumular muito YES:
    - Reduz bid de YES
    - Aumenta ask de YES
    - Ou vende YES no mercado

    Target: delta = 0 (neutral)
    """

    def update_position(self, token: str, size: float, side: str):
        """Atualiza posição após fill."""

    def get_quote_adjustment(self, token: str) -> dict:
        """Retorna ajuste de preço para rebalancear."""
        return {
            'bid_adjustment': -0.005,  # -0.5 cents
            'ask_adjustment': +0.002,  # +0.2 cents
        }

    def needs_hedge(self) -> bool:
        """Verifica se precisa hedge urgente."""
```

### 4. Fast WebSocket (`fast_ws.py`)

```python
class FastWebSocket:
    """
    WebSocket otimizado com uvloop e orjson.

    Features:
    - Reconexão automática
    - Heartbeat
    - Parse JSON com orjson (10x faster)
    - Callback para price updates
    """

    async def connect(self, markets: list):
        """Conecta aos mercados."""

    async def on_price_update(self, callback):
        """Registra callback para updates."""

    def get_orderbook(self, market_id: str) -> dict:
        """Retorna orderbook cached."""
```

### 5. Market Maker Bot (`market_maker_bot.py`)

```python
class MarketMakerBot:
    """
    Bot de Market Making completo.

    Estratégia:
    1. Conecta via WebSocket
    2. Calcula mid price
    3. Ajusta spread baseado em volatilidade
    4. Coloca bid e ask
    5. Quando fill, ajusta delta
    6. Cancela e re-quota quando preço move

    Parâmetros:
    - base_spread: 0.02 (2%)
    - max_position: $100 por lado
    - requote_threshold: 0.5% de movimento
    - volatility_lookback: 100 ticks
    """

    async def run(self):
        """Loop principal."""

    def calculate_quotes(self, market: str) -> tuple:
        """Calcula bid e ask prices."""
        mid = self.get_mid_price(market)
        vol = self.volatility.get_recommendations()
        delta = self.hedger.get_quote_adjustment(market)

        spread = self.base_spread * vol['spread_multiplier']

        bid = mid - spread/2 + delta['bid_adjustment']
        ask = mid + spread/2 + delta['ask_adjustment']

        return bid, ask
```

---

## Configuração (.env)

```env
# Escolher qual bot usar
BOT_MODE=market_maker  # ou "arbitrage"

# Market Maker específico
MM_BASE_SPREAD=0.02
MM_MAX_POSITION=100
MM_REQUOTE_THRESHOLD=0.005
MM_VOLATILITY_LOOKBACK=100
MM_MARKETS=BTC,ETH,SOL

# Performance
USE_UVLOOP=true
USE_ORJSON=true
USE_WSS=true
PRE_SIGN_POOL_SIZE=10
```

---

## Fluxo de Execução

```
┌─────────────────────────────────────────────────────────────┐
│                    MARKET MAKER BOT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. STARTUP                                                 │
│     ├── Conectar WebSocket                                  │
│     ├── Pré-assinar pool de ordens                         │
│     └── Inicializar volatility engine                       │
│                                                             │
│  2. MAIN LOOP (cada tick ~100ms)                           │
│     ├── Receber price update (WebSocket)                   │
│     ├── Atualizar volatility                               │
│     ├── Verificar se precisa re-quote                      │
│     │   └── Se preço moveu > threshold → cancela + re-quota│
│     ├── Calcular novos quotes                              │
│     │   ├── Mid price                                      │
│     │   ├── Spread (baseado em volatility)                 │
│     │   └── Ajuste delta                                   │
│     └── Enviar ordens (bid + ask)                          │
│                                                             │
│  3. ON FILL                                                 │
│     ├── Atualizar posição                                  │
│     ├── Atualizar delta hedger                             │
│     ├── Log trade                                          │
│     └── Re-quotar lado que executou                        │
│                                                             │
│  4. RISK CHECKS (cada 1s)                                  │
│     ├── Verificar max position                             │
│     ├── Verificar P&L                                      │
│     └── Se limite atingido → cancelar tudo                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparação: Bot 1 vs Bot 2

| Aspecto | Bot 1 (Arbitrage) | Bot 2 (Market Maker) |
|---------|-------------------|----------------------|
| Tipo de ordem | Taker (FOK) | Maker (GTC) |
| Profit/trade | ~2-4% | ~0.3-1% |
| Trades/hora | 5-20 | 50-200 |
| Risco | Baixo | Médio |
| Capital mínimo | $100 | $500+ |
| Complexidade | Simples | Complexo |
| Mercados | 1 | Múltiplos |
| Latência crítica | Não | Sim |

---

## Prioridades de Implementação

### Fase 1: Fundação (1-2 dias)
- [ ] `volatility.py` - Módulo de volatilidade
- [ ] `fast_ws.py` - WebSocket otimizado
- [ ] Integrar uvloop e orjson

### Fase 2: Core (2-3 dias)
- [ ] `order_manager.py` - Gerenciador de ordens
- [ ] `delta_hedge.py` - Delta hedging
- [ ] `market_maker_bot.py` - Bot principal

### Fase 3: Otimização (1-2 dias)
- [ ] Pre-signed orders pool
- [ ] Multi-mercado
- [ ] Backtesting com dados históricos

### Fase 4: Produção (1 dia)
- [ ] Testes em paper trading
- [ ] Monitoring e alertas
- [ ] Deploy em AWS us-east-1

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| Inventory acumula | Alto | Delta hedge automático |
| Preço move rápido | Alto | Requote < 100ms |
| Fill parcial | Médio | Cancelar ordens órfãs |
| Conexão cai | Alto | Reconexão + cancel all |
| Volatilidade extrema | Alto | Circuit breaker |

---

## Como Usar

```bash
# Bot 1 (Arbitragem - atual)
python -m trading_bot_ltm

# Bot 2 (Market Maker - novo)
python -m trading_bot_ltm.market_maker

# Análise
python -m trading_bot_ltm.analyze_logs logs/
```

---

## Próximos Passos

1. **Aprovar este plano**
2. **Implementar módulo de volatilidade** (mais fácil, impacto imediato)
3. **Testar WebSocket** (já existe código base)
4. **Desenvolver Market Maker Bot**
5. **Paper trading por 1 semana**
6. **Live com $100**

---

*Última atualização: Janeiro 2025*
