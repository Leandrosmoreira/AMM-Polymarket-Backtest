# 📊 LADM — Plano de Análise Quantitativa Avançada
## BTC Up/Down 15min Markets (Polymarket)

**Versão:** 2.0
**Data:** 2026-01-04
**Stack:** Python 3.11+ | pandas | numpy | scipy | matplotlib | polars | duckdb

---

## 🎯 Objetivo Principal

Realizar análise quantitativa profunda dos mercados BTC Up/Down 15min usando dados CLOB completos para:
1. Identificar padrões estatísticos e edges mensuráveis
2. Validar/refutar a estratégia "gabagool22" (market making via arbitragem de probabilidade)
3. Definir KPIs para decidir se o mercado é explorável
4. Preparar feature store para backtesting do bot LADM

---

## 📁 Estrutura de Dados Disponíveis

```
data/
├── state/          # State ticks (1s) - snapshot completo do bot
│   └── state-YYYY-MM-DD.jsonl
├── prices/         # Preços BTC (1s) - Chainlink on-chain + Binance
│   └── prices-YYYY-MM-DD.jsonl
├── books/          # Order book snapshots (5s)
│   └── books-YYYY-MM-DD.jsonl
├── trades/         # Trades executados (RTDS WebSocket)
│   └── trades-YYYY-MM-DD.jsonl
└── events/         # Phase changes e market transitions
    └── events-YYYY-MM-DD.jsonl
```

### Schemas Atuais

**State Tick (v1.0):**
```json
{
  "v": 1,
  "ts": 1736012345678,
  "marketId": "0x...",
  "marketSlug": "btc-updown-15m-1736012400",
  "ref": { "source": "chainlink", "price": 97234.56, "ts": 1736012345000 },
  "yes": { "bid": 0.52, "ask": 0.54, "last": 0.53 },
  "no": { "bid": 0.46, "ask": 0.48, "last": 0.47 },
  "fair": { "yes": 0.528, "no": 0.472 },
  "liquidity": { "score": 0.75, "yesDepth": 5000, "noDepth": 4800 },
  "regime": { "phase": "B", "tteMs": 420000 },
  "risk": { "mode": "normal", "inventory": { "yes": 0, "no": 0 } }
}
```

**Price Tick:**
```json
{
  "ts": 1736012345678,
  "source": "chainlink",
  "price": 97234.56,
  "roundId": "123456789",
  "onchainUpdatedAt": 1736012343000,
  "binancePrice": 97235.12,
  "diff": "-0.0006"
}
```

**Trade (via RTDS):**
```json
{
  "ts": 1736012345678,
  "market": "0x...",
  "asset_id": "12345...",
  "side": "YES",
  "price": 0.53,
  "size": 125.5,
  "fee": "0.02"
}
```

---

# 🧠 Arquitetura de Agents

## Agent -1: Data Quality & Validation (Gatekeeper)
**Prioridade:** CRÍTICA - Executa ANTES de qualquer análise

### Sub-Agent -1.1: Log Inventory & Coverage
**Objetivo:** Validar existência e completude dos dados

**Checks:**
- [ ] Todos os arquivos necessários existem
- [ ] Cobertura temporal contínua (sem gaps > 5min)
- [ ] Rollover coverage: múltiplos marketSlugs capturados
- [ ] Mínimo de samples por dia (≥80k state ticks, ≥50k price ticks)

**Output:** `reports/validation/coverage_report.csv`

### Sub-Agent -1.2: Schema Validation & Normalization
**Objetivo:** Garantir consistência de tipos e campos

**Checks:**
- [ ] Campos obrigatórios presentes (ver schemas acima)
- [ ] Tipos corretos (ts numérico, prices float, etc)
- [ ] Taxa de JSON parse errors < 0.01%
- [ ] Taxa de null em campos críticos < 1%

**Output:** `reports/validation/schema_audit.md`

### Sub-Agent -1.3: Cross-Dataset Consistency
**Objetivo:** Validar alinhamento entre datasets

**Checks:**
- [ ] Para cada marketSlug: trades, books, prices existem no mesmo intervalo
- [ ] Timestamps monotonicamente crescentes
- [ ] Deduplicação (trades por hash, books por ts+slug)
- [ ] Token IDs (YES/NO) consistentes entre datasets

**Output:** `reports/validation/consistency_metrics.csv`

### Sub-Agent -1.4: Readiness Score (Go/No-Go Gate)
**Scoring:**
- Completeness Score (0-100)
- Consistency Score (0-100)
- Freshness Score (0-100)

**Gate Rule:** Score médio ≥ 80 para prosseguir

**Output:** `reports/validation/DATA_READINESS.md`

---

## Agent 0: Orchestrator (Lead Quant)
**Papel:** Coordenar todos os agents e consolidar resultados

**Responsabilidades:**
- Definir janelas de análise (full day, per 15m window, last 3m, last 1m)
- Alinhar timestamps (UTC)
- Garantir reprodutibilidade (notebooks + scripts)
- Consolidar outputs para decisão executiva

**Janelas de Análise:**
```python
ANALYSIS_WINDOWS = {
    'full_window': (0, 900),      # 0-15min inteiro
    'phase_A': (0, 300),          # 0-5min (formação)
    'phase_B': (300, 720),        # 5-12min (maturação)
    'phase_C': (720, 900),        # 12-15min (resolução)
    'last_3m': (720, 900),        # últimos 3min
    'last_1m': (840, 900),        # último 1min
    'last_30s': (870, 900),       # últimos 30s
}
```

---

## Agent 1: TRADES — Order Flow Analysis

### Sub-Agent 1.1: Trade Flow Analyzer
**Foco:** Execuções reais

**Métricas:**
| Métrica | Descrição | Formula |
|---------|-----------|---------|
| `trades_per_second` | Intensidade de trading | count(trades) / Δt |
| `volume_yes` | Volume em YES | sum(size) where side='YES' |
| `volume_no` | Volume em NO | sum(size) where side='NO' |
| `vwap_yes` | VWAP YES | sum(price*size) / sum(size) |
| `vwap_no` | VWAP NO | sum(price*size) / sum(size) |
| `median_size` | Tamanho mediano | median(size) |
| `large_trade_pct` | % trades grandes | count(size > p95) / count(*) |

**Visualizações:**
- Histograma de trade sizes (log scale)
- Volume cumulativo YES vs NO por fase
- Timeline de intensidade de trading

### Sub-Agent 1.2: Aggression & Momentum
**Foco:** Pressão direcional

**Métricas:**
| Métrica | Descrição | Interpretação |
|---------|-----------|---------------|
| `flow_imbalance` | (vol_yes - vol_no) / total | >0 = bullish, <0 = bearish |
| `consecutive_streaks` | Maior sequência de mesmo lado | Momentum indicator |
| `acceleration` | Δ(trades/s) / Δt | Aceleração de fluxo |
| `price_impact` | Δprice / Δvolume | Proxy de impacto |

**Hipótese Gabagool:** Se flow_imbalance prediz direção final, há edge direcional (não apenas spread)

### Sub-Agent 1.3: End-of-Window Behavior
**Foco:** Últimos minutos antes da resolução

**Análises Críticas:**
```python
# Métricas nos últimos 60s
last_60s_metrics = {
    'volume_share': volume[-60s:] / volume_total,
    'price_drift': price[-1] - price[-60],
    'convergence_speed': abs(Δprice/Δt),
    'trade_clustering': std(trade_intervals),
}
```

**Perguntas-Chave:**
- O preço converge violentamente no final?
- A direção final é previsível antes de T-30s?
- Há bursts de volume suspeitos (manipulation)?

---

## Agent 2: BOOKS — Market Microstructure

### Sub-Agent 2.1: Book Shape & Depth
**Métricas:**
| Métrica | Descrição |
|---------|-----------|
| `spread` | best_ask - best_bid |
| `mid_price` | (best_bid + best_ask) / 2 |
| `depth_bid_1pct` | Volume dentro de 1% do bid |
| `depth_ask_1pct` | Volume dentro de 1% do ask |
| `imbalance` | (depth_bid - depth_ask) / total |
| `book_skew` | Assimetria do book |

**Visualizações:**
- Heatmap de profundidade por preço/tempo
- Spread evolution over time
- Book imbalance timeline

### Sub-Agent 2.2: Liquidity Dynamics
**Foco:** Comportamento da liquidez

**Métricas Avançadas:**
```python
# Proxy de cancel/add ratio (entre snapshots)
liquidity_change = depth[t] - depth[t-1]
add_rate = sum(liquidity_change > 0)
cancel_rate = sum(liquidity_change < 0)

# Fake liquidity detection
def detect_fake_liquidity(books):
    """Liquidez que desaparece antes de ser atingida"""
    vanishing_depth = []
    for t in range(1, len(books)):
        if books[t].depth < books[t-1].depth * 0.5:
            if no_trade_at_price(books[t-1].best_bid):
                vanishing_depth.append(t)
    return vanishing_depth
```

### Sub-Agent 2.3: Pre-Trade Book Signals
**Foco:** Poder preditivo do book

**Análise:**
```python
# Book imbalance → Price move (lag analysis)
for lag in [1, 5, 10, 30, 60]:  # seconds
    correlation = corr(imbalance[t], price_change[t+lag])
    predictive_power[lag] = correlation
```

**Hipótese:** Book imbalance > 0.3 prediz movimento na direção do imbalance

---

## Agent 3: PRICES — Reference Price Analysis

### Sub-Agent 3.1: Chainlink vs Binance Divergence
**Foco:** Latência e divergência entre fontes

**Métricas:**
| Métrica | Descrição | Threshold |
|---------|-----------|-----------|
| `price_diff_pct` | (chainlink - binance) / binance | Normal: <0.1% |
| `latency_estimate` | Lag do Chainlink vs Binance | Normal: <5s |
| `divergence_events` | Count de diff > 0.5% | Alert se > 10/hour |

**Importância:** Chainlink é usado para resolução. Se Binance lidera, há edge informacional.

### Sub-Agent 3.2: BTC Volatility & Market Response
**Foco:** Como preço do market responde ao BTC

**Análise:**
```python
# Correlação BTC move → Market price move
btc_returns = diff(btc_price) / btc_price
yes_returns = diff(yes_mid) / yes_mid

# Lead-lag analysis
for lag in range(-30, 31):
    corr = correlation(btc_returns, yes_returns.shift(lag))
    # Se lag negativo tem maior corr → BTC lidera (esperado)
```

### Sub-Agent 3.3: Fair Value Estimation
**Foco:** Estimar probabilidade "verdadeira"

**Modelos:**
1. **Binary Fair Value:** P(up) baseado em BTC price vs open price
2. **Momentum Fair Value:** P(up) ajustado por momentum recente
3. **Time-Decay Fair Value:** P(up) convergindo para 0 ou 1 conforme TTE→0

```python
def estimate_fair_value(btc_price, open_price, tte_seconds):
    """Estima P(YES) baseado em estado atual"""
    current_return = (btc_price - open_price) / open_price

    # Quanto menor TTE, mais certeza
    certainty = 1 - (tte_seconds / 900)

    # Se BTC subiu, P(YES) aumenta
    base_prob = 0.5 + current_return * 10  # scaling factor

    # Convergência para 0 ou 1
    if current_return > 0:
        fair_yes = 0.5 + (0.5 * certainty * sigmoid(current_return * 100))
    else:
        fair_yes = 0.5 - (0.5 * certainty * sigmoid(-current_return * 100))

    return clip(fair_yes, 0.01, 0.99)
```

---

## Agent 4: REGIME — Phase Analysis

### Sub-Agent 4.1: Phase Characterization
**Fases do Market:**

| Phase | TTE | Características Esperadas |
|-------|-----|---------------------------|
| A (Formação) | 15-10min | Baixa liquidez, spreads amplos, preço ~0.50 |
| B (Maturação) | 10-3min | Liquidez crescente, spreads estreitos, preço reflete BTC |
| C (Resolução) | 3-0min | Alta volatilidade, convergência rápida, spreads podem abrir |

**Métricas por Fase:**
```python
phase_metrics = {
    'spread_mean': {},
    'spread_std': {},
    'volume_total': {},
    'trades_count': {},
    'price_volatility': {},
    'liquidity_score': {},
}

for phase in ['A', 'B', 'C']:
    phase_metrics['spread_mean'][phase] = mean(spread[phase])
    # ... etc
```

### Sub-Agent 4.2: Phase Transition Detection
**Foco:** Identificar mudanças de regime além do tempo

**Sinais de Transição:**
- Spread compression/expansion súbita
- Volume spike
- Liquidity withdrawal
- Price jump

```python
def detect_regime_change(state_ticks):
    """Detecta transições de regime não-temporais"""
    changes = []
    for t in range(1, len(state_ticks)):
        spread_change = abs(state_ticks[t].spread - state_ticks[t-1].spread)
        vol_spike = state_ticks[t].volume > state_ticks[t-1].volume * 3

        if spread_change > 0.05 or vol_spike:
            changes.append({
                'ts': state_ticks[t].ts,
                'type': 'spread_change' if spread_change > 0.05 else 'volume_spike',
                'phase': state_ticks[t].phase,
            })
    return changes
```

---

## Agent 5: STRATEGY VALIDATION — Gabagool Analysis

### Sub-Agent 5.1: Market Making Edge
**Hipótese Gabagool:** Lucro vem de capturar spread, não de previsão direcional

**Teste:**
```python
def backtest_market_making(trades, books):
    """Simula market making passivo"""
    pnl = 0
    inventory = {'yes': 0, 'no': 0}

    for trade in trades:
        if trade.side == 'YES':
            # Vendemos YES (fornecemos liquidez)
            if inventory['yes'] > 0:
                pnl += trade.price - avg_cost_yes
                inventory['yes'] -= trade.size
            else:
                # Compramos YES
                inventory['yes'] += trade.size
                avg_cost_yes = trade.price
        # ... similar para NO

    return pnl, inventory
```

### Sub-Agent 5.2: Probability Arbitrage
**Hipótese:** YES + NO deve somar ~1.00 (menos fees)

**Análise:**
```python
def find_arbitrage_opportunities(state_ticks):
    """Encontra momentos onde YES_bid + NO_bid > 1 ou YES_ask + NO_ask < 1"""
    opportunities = []

    for tick in state_ticks:
        # Arbitrage: comprar ambos lados
        buy_cost = tick.yes_ask + tick.no_ask
        if buy_cost < 0.98:  # Garantido lucro se custo < 1
            opportunities.append({
                'ts': tick.ts,
                'type': 'buy_both',
                'edge': 1.0 - buy_cost,
            })

        # Arbitrage: vender ambos lados
        sell_revenue = tick.yes_bid + tick.no_bid
        if sell_revenue > 1.02:  # Garantido lucro se receita > 1
            opportunities.append({
                'ts': tick.ts,
                'type': 'sell_both',
                'edge': sell_revenue - 1.0,
            })

    return opportunities
```

### Sub-Agent 5.3: Directional Edge
**Contra-Hipótese:** Há edge direcional previsível

**Testes:**
```python
# 1. Book Imbalance → Outcome
def test_book_imbalance_predictive():
    for window in state_ticks.groupby('marketSlug'):
        imbalance_at_5min = window[tte==600].imbalance
        outcome = window.final_outcome  # 'YES' or 'NO'

        if imbalance_at_5min > 0.3 and outcome == 'YES':
            hit += 1
        elif imbalance_at_5min < -0.3 and outcome == 'NO':
            hit += 1
        total += 1

    return hit / total  # Hit rate

# 2. Flow Imbalance → Outcome
def test_flow_predictive():
    # Similar mas usando trade flow
    pass

# 3. BTC Momentum → Outcome
def test_btc_momentum_predictive():
    # BTC trend nos últimos 5min prediz outcome?
    pass
```

---

## Agent 6: RISK — Adverse Selection & Toxicity

### Sub-Agent 6.1: Toxic Flow Detection
**Foco:** Identificar trades "informados"

**Métricas:**
```python
def calculate_toxicity(trades, outcomes):
    """Mede quanto os trades preveem o outcome"""
    toxic_score = []

    for window in group_by_market(trades):
        yes_volume = sum(t.size for t in window if t.side == 'YES')
        no_volume = sum(t.size for t in window if t.side == 'NO')

        predicted_side = 'YES' if yes_volume > no_volume else 'NO'
        actual_outcome = outcomes[window.market_slug]

        if predicted_side == actual_outcome:
            toxic_score.append(1)
        else:
            toxic_score.append(0)

    return mean(toxic_score)  # >0.5 = flow é informado
```

### Sub-Agent 6.2: Adverse Selection by Time
**Hipótese:** Trades perto do fim são mais informados

```python
def toxicity_by_tte(trades, outcomes):
    """Toxicidade por tempo até expiração"""
    tte_buckets = [900, 600, 300, 120, 60, 30, 10]

    for bucket in tte_buckets:
        bucket_trades = trades[tte <= bucket]
        toxicity = calculate_toxicity(bucket_trades, outcomes)
        print(f"TTE <= {bucket}s: Toxicity = {toxicity:.2%}")
```

---

## Agent 7: BACKTEST — Strategy Simulation

### Sub-Agent 7.1: Event-Time Replay Engine
**Objetivo:** Replay segundo-a-segundo com dados reais

```python
class ReplayEngine:
    def __init__(self, state_ticks, trades, books):
        self.state = state_ticks
        self.trades = trades
        self.books = books

    def replay(self, strategy, start_ts, end_ts):
        """Executa estratégia em dados históricos"""
        results = []
        position = {'yes': 0, 'no': 0}
        cash = 10000  # USDC inicial

        for tick in self.state[start_ts:end_ts]:
            # Estratégia decide ação
            action = strategy.decide(tick, position)

            if action:
                # Simula execução
                fill_price = self.simulate_fill(action, tick)
                position, cash = self.update_position(
                    position, cash, action, fill_price
                )
                results.append({
                    'ts': tick.ts,
                    'action': action,
                    'fill_price': fill_price,
                    'position': position.copy(),
                    'cash': cash,
                })

        # Resolve no final
        final_pnl = self.resolve_position(position, cash)
        return results, final_pnl
```

### Sub-Agent 7.2: Strategy Variants
**Estratégias para Testar:**

1. **Passive Market Making:**
   - Posta bid/ask com spread fixo em torno do fair value
   - Ajusta por inventory

2. **Aggressive Market Making:**
   - Toma liquidez quando detecta mispricing
   - Posta quando spread é bom

3. **Directional (baseline):**
   - Compra YES se book_imbalance > threshold
   - Sem gestão de spread

4. **Hybrid (Gabagool):**
   - Market making com bias direcional baseado em signals

---

# 📈 KPI Framework — Is This Market Exploitable?

## Primary KPIs (Edge)
| KPI | Descrição | Threshold Bom |
|-----|-----------|---------------|
| `signal_hit_rate` | % direção correta | >55% |
| `signal_lead_time` | Segundos antes da resolução | >30s |
| `edge_per_trade` | P&L médio por trade | >$0.10 |
| `sharpe_ratio` | Retorno ajustado por risco | >1.5 |

## Microstructure KPIs (Feasibility)
| KPI | Descrição | Threshold Bom |
|-----|-----------|---------------|
| `avg_spread` | Spread médio | <5% |
| `spread_stability` | Std do spread | <2% |
| `depth_at_best` | Liquidez no melhor preço | >$500 |
| `fill_rate` | % ordens executadas | >80% |

## Risk KPIs
| KPI | Descrição | Threshold Aceitável |
|-----|-----------|---------------------|
| `max_drawdown` | Maior perda consecutiva | <20% |
| `toxicity_score` | % flow informado | <60% |
| `adverse_selection_cost` | Perda por trades informados | <$0.05/trade |

## Decision Matrix
```
                    Low Toxicity    High Toxicity
                    (<50%)          (>60%)
                   ┌───────────────┬───────────────┐
High Edge (>55%)   │   ✅ GO       │   ⚠️ CAUTION  │
                   │  Full size    │  Small size   │
                   ├───────────────┼───────────────┤
Low Edge (<52%)    │   ⚠️ MAYBE    │   ❌ NO-GO    │
                   │  Paper trade  │   Don't trade │
                   └───────────────┴───────────────┘
```

---

# 📊 Outputs Finais

## Deliverables por Agent
```
reports/
├── validation/
│   ├── DATA_READINESS.md
│   ├── coverage_report.csv
│   └── schema_audit.md
├── trades/
│   ├── summary.md
│   ├── figures/
│   │   ├── trade_size_histogram.png
│   │   ├── volume_by_phase.png
│   │   └── flow_imbalance.png
│   └── tables/
│       └── flow_metrics.csv
├── books/
│   ├── summary.md
│   ├── figures/
│   │   ├── depth_heatmap.png
│   │   └── spread_evolution.png
│   └── tables/
│       └── liquidity_metrics.csv
├── prices/
│   ├── summary.md
│   ├── figures/
│   │   └── chainlink_vs_binance.png
│   └── tables/
│       └── price_metrics.csv
├── strategy/
│   ├── gabagool_validation.md
│   ├── backtest_results.csv
│   └── figures/
│       ├── pnl_curve.png
│       └── edge_decay.png
└── executive/
    ├── EXECUTIVE_SUMMARY.md      # 1-page summary
    ├── GO_NOGO_DECISION.md       # Final recommendation
    └── NEXT_STEPS.md             # Action items
```

## Executive Summary Template
```markdown
# LADM BTC 15min Market Analysis
## Executive Summary

**Data Period:** YYYY-MM-DD to YYYY-MM-DD
**Markets Analyzed:** N windows
**Total Volume:** $XXX,XXX

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Edge Detected
- Signal hit rate: XX%
- Avg edge per trade: $X.XX
- Confidence: HIGH/MEDIUM/LOW

### Recommendation
[ ] ✅ GO - Deploy with $XX,XXX capital
[ ] ⚠️ PAPER TRADE - More data needed
[ ] ❌ NO-GO - Edge insufficient

### Risk Factors
- [Risk 1]
- [Risk 2]

### Next Steps
1. [Action 1]
2. [Action 2]
```

---

# 🚀 Implementation Roadmap

## Phase 1: Data Validation (Day 1-2)
- [ ] Run Agent -1 (Gatekeeper)
- [ ] Fix any data issues
- [ ] Convert JSONL → Parquet

## Phase 2: Exploratory Analysis (Day 3-5)
- [ ] Agent 1: Trade flow analysis
- [ ] Agent 2: Book microstructure
- [ ] Agent 3: Price analysis

## Phase 3: Strategy Validation (Day 6-8)
- [ ] Agent 5: Gabagool hypothesis testing
- [ ] Agent 6: Toxicity analysis
- [ ] Agent 7: Backtest simulations

## Phase 4: Synthesis (Day 9-10)
- [ ] Consolidate KPIs
- [ ] Generate executive report
- [ ] Make GO/NO-GO decision

---

# 🛠️ Technical Setup

## VPS Structure (~/ladm-bot)
```
~/ladm-bot/
├── config/
├── data/
│   ├── books/          # Order book snapshots (5s)
│   ├── trades/         # Trades executados (RTDS WebSocket)
│   ├── prices/         # Preços BTC (Chainlink + Binance)
│   ├── events/         # Phase changes e market transitions
│   └── state/          # State ticks (1s)
├── analytics/          # << NOVA PASTA DE ANÁLISE
│   ├── notebooks/      # Jupyter notebooks
│   ├── scripts/        # Python scripts
│   └── reports/        # Outputs por agent
│       ├── validation/
│       ├── trades/
│       ├── books/
│       ├── prices/
│       ├── strategy/
│       └── executive/
├── dist/
├── logs/
├── monitoring/
├── node_modules/
├── scripts/
├── src/
└── tests/
```

## Python Environment
```bash
cd ~/ladm-bot

# Criar venv para analytics (separado do Node.js)
python3 -m venv analytics/venv
source analytics/venv/bin/activate

# Instalar dependências
pip install pandas numpy scipy matplotlib polars duckdb pyarrow rich tqdm httpx jupyter
```

## Directory Structure (já criado)
```bash
mkdir -p ~/ladm-bot/analytics/{notebooks,scripts}
mkdir -p ~/ladm-bot/analytics/reports/{validation,trades,books,prices,strategy,executive}
```

## Quick Start Script
```python
# ~/ladm-bot/analytics/scripts/load_data.py
# Ver arquivo completo em: analytics/scripts/load_data.py
```

---

# ❌ Out of Scope
- Trading bot implementation (separate project)
- Capital allocation decisions
- MEV/latency infrastructure
- Live execution logic
