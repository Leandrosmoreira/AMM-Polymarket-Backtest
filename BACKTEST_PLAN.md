# 🎯 Backtest Plan: AMM Delta-Neutral Strategy
## Polymarket SOL 15-min Markets

---

## 📋 Sumário Executivo

| Item | Descrição |
|------|-----------|
| **Estratégia** | Market Making Delta-Neutral com hedge bilateral |
| **Mercado** | Solana Up/Down 15-minute markets |
| **Período** | 3 meses (90 dias) |
| **Objetivo** | Capturar spread YES+NO < $1.00, mantendo posição equilibrada |
| **Gestão de Risco** | Moderada |

---

## 🏗️ FASE 0: Setup do Ambiente

### 0.1 Estrutura do Projeto

```
polymarket_sol_backtest/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Parâmetros globais
│   └── risk_params.py       # Parâmetros de risco
├── data/
│   ├── raw/                  # Dados brutos da API
│   ├── processed/            # Dados processados
│   ├── trades/               # Histórico de trades simulados
│   └── results/              # Resultados do backtest
├── src/
│   ├── __init__.py
│   ├── data_collector.py     # Coleta de dados históricos
│   ├── market_analyzer.py    # Análise de mercados SOL
│   ├── spread_calculator.py  # Cálculo de spreads
│   ├── position_manager.py   # Gerenciamento de posições
│   ├── risk_manager.py       # Gestão de risco
│   ├── backtest_engine.py    # Motor do backtest
│   ├── metrics.py            # Cálculo de métricas
│   └── visualizer.py         # Gráficos e relatórios
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_spread_analysis.ipynb
│   ├── 03_backtest_results.ipynb
│   └── 04_optimization.ipynb
├── tests/
│   └── test_backtest.py
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

### 0.2 Dependências

```txt
# requirements.txt
httpx>=0.25.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dateutil>=2.8.2
tqdm>=4.65.0
scipy>=1.11.0
```

### 0.3 Configurações Base

```python
# config/settings.py

# === MERCADO ===
ASSET = "SOL"
MARKET_TYPE = "up_or_down"
TIMEFRAME_MINUTES = 15

# === PERÍODO DO BACKTEST ===
BACKTEST_START = "2024-09-13"  # 3 meses atrás
BACKTEST_END = "2024-12-13"    # Hoje
BACKTEST_DAYS = 90

# === CAPITAL ===
INITIAL_CAPITAL = 5000  # USD
CURRENCY = "USDC"

# === API ===
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
```

---

## 📊 FASE 1: Coleta de Dados

### 1.1 Identificar Mercados SOL 15-min

**Objetivo:** Listar todos os mercados "Solana Up or Down" de 15 minutos no período.

**Endpoint:** `GET /markets`

**Filtros:**
```python
{
    "tag": "solana",
    "market_type": "up_or_down", 
    "closed": True,  # Apenas mercados resolvidos
    "start_date_min": "2024-09-13",
    "start_date_max": "2024-12-13"
}
```

**Output esperado:**
```python
# Estimativa: 4 mercados/hora × 24h × 90 dias = ~8,640 mercados
markets_df = pd.DataFrame({
    "market_id": str,
    "condition_id": str,
    "question": str,           # "Solana Up or Down - Dec 13, 4:30AM-4:45AM ET"
    "start_time": datetime,
    "end_time": datetime,
    "outcome": str,            # "Up" ou "Down"
    "yes_token_id": str,
    "no_token_id": str,
    "volume": float,
    "liquidity": float
})
```

**Tarefa Claude Code:**
```
1. Criar função fetch_sol_15min_markets(start_date, end_date)
2. Paginar resultados (limit=100 por request)
3. Filtrar apenas mercados "Up or Down" de SOL
4. Salvar em data/raw/sol_markets.csv
5. Log: quantidade de mercados encontrados por dia
```

---

### 1.2 Coletar Histórico de Preços

**Objetivo:** Para cada mercado, obter timeseries de preços YES e NO.

**Endpoint:** `GET /prices-history`

**Parâmetros:**
```python
{
    "market": token_id,
    "interval": "1m",      # Granularidade de 1 minuto
    "fidelity": 60,
    "startTs": start_timestamp,
    "endTs": end_timestamp
}
```

**Output esperado por mercado:**
```python
price_history_df = pd.DataFrame({
    "timestamp": datetime,
    "market_id": str,
    "price_yes": float,    # 0.00 - 1.00
    "price_no": float,     # 0.00 - 1.00
    "volume_yes": float,
    "volume_no": float,
    "spread": float,       # price_yes + price_no - 1.0
    "mid_price": float     # (price_yes + (1 - price_no)) / 2
})
```

**Tarefa Claude Code:**
```
1. Criar função fetch_price_history(market_id, yes_token, no_token)
2. Para cada mercado de 15min, coletar ~15 pontos de preço
3. Calcular spread em cada ponto: spread = yes + no - 1.0
4. Salvar em data/raw/price_history/{market_id}.csv
5. Consolidar em data/processed/all_prices.parquet
6. Rate limiting: max 10 requests/segundo
```

---

### 1.3 Coletar Order Book Snapshots (se disponível)

**Objetivo:** Entender a profundidade de liquidez em diferentes níveis de preço.

**Endpoint:** `GET /book`

**Nota:** Dados históricos de orderbook podem não estar disponíveis. Se não estiver:
- Usar volume como proxy de liquidez
- Estimar slippage baseado em volume médio

**Tarefa Claude Code:**
```
1. Verificar se API tem dados históricos de orderbook
2. Se sim: coletar snapshots a cada minuto
3. Se não: criar modelo de slippage baseado em volume
4. Documentar limitações
```

---

## 🔬 FASE 2: Análise Exploratória

### 2.1 Estatísticas de Spread

**Métricas a calcular:**

```python
spread_stats = {
    # Distribuição básica
    "mean": float,
    "std": float,
    "min": float,
    "max": float,
    "median": float,
    
    # Percentis
    "p5": float,
    "p25": float,
    "p75": float,
    "p95": float,
    
    # Oportunidades
    "pct_below_99": float,    # % do tempo com YES+NO < 0.99
    "pct_below_98": float,    # % do tempo com YES+NO < 0.98
    "pct_below_97": float,    # % do tempo com YES+NO < 0.97
    
    # Temporal
    "avg_spread_by_hour": dict,
    "avg_spread_by_weekday": dict,
    
    # Por mercado
    "avg_spread_per_market": float,
    "markets_with_opportunity": int
}
```

**Tarefa Claude Code:**
```
1. Criar notebook 01_data_exploration.ipynb
2. Calcular todas as métricas acima
3. Gerar histograma de distribuição de spreads
4. Gerar heatmap hora × dia da semana
5. Identificar melhores horários para operar
```

---

### 2.2 Análise de Liquidez

**Métricas:**
```python
liquidity_stats = {
    "avg_volume_per_market": float,
    "median_volume_per_market": float,
    "volume_by_hour": dict,
    "volume_by_weekday": dict,
    "correlation_volume_spread": float,  # Mais volume = menos spread?
}
```

**Pergunta chave:** Mercados com mais volume têm spreads menores (mais eficientes)?

---

### 2.3 Análise de Outcomes

**Métricas:**
```python
outcome_stats = {
    "pct_up": float,      # % dos mercados que resolveram "Up"
    "pct_down": float,    # % dos mercados que resolveram "Down"
    "streaks": dict,      # Sequências de Up ou Down
    "autocorrelation": float  # Resultado anterior prediz próximo?
}
```

**Nota:** Para estratégia delta-neutral, outcome não deveria importar. Mas é bom verificar.

---

## ⚙️ FASE 3: Definição da Estratégia

### 3.1 Lógica de Entrada

```python
# Regras para ENTRAR em um mercado

def should_enter(market_state, portfolio_state, risk_params):
    """
    Retorna True se deve começar a operar neste mercado.
    """
    
    # 1. Spread mínimo
    if market_state.spread >= risk_params.MIN_SPREAD_TO_ENTER:
        return False  # Spread não é atrativo (YES + NO >= threshold)
    
    # 2. Volume mínimo (proxy de liquidez)
    if market_state.volume < risk_params.MIN_VOLUME:
        return False
    
    # 3. Tempo restante no mercado
    if market_state.time_remaining < risk_params.MIN_TIME_REMAINING:
        return False  # Muito perto do settlement
    
    # 4. Capital disponível
    if portfolio_state.available_cash < risk_params.MIN_ORDER_SIZE * 2:
        return False
    
    # 5. Limite de mercados simultâneos
    if portfolio_state.active_markets >= risk_params.MAX_ACTIVE_MARKETS:
        return False
    
    # 6. Exposição total
    if portfolio_state.total_exposure >= risk_params.MAX_TOTAL_EXPOSURE:
        return False
    
    return True
```

---

### 3.2 Lógica de Sizing

```python
def calculate_order_size(market_state, portfolio_state, risk_params):
    """
    Calcula quanto comprar de YES e NO.
    """
    
    # Capital disponível para este mercado
    max_for_market = min(
        portfolio_state.available_cash * risk_params.MAX_PER_MARKET_PCT,
        risk_params.MAX_PER_MARKET_USD
    )
    
    # Divide entre YES e NO baseado nos preços
    price_yes = market_state.price_yes
    price_no = market_state.price_no
    total_price = price_yes + price_no
    
    # Quantidade de "pares" que podemos comprar
    pairs_to_buy = max_for_market / total_price
    
    # Arredonda para baixo
    shares_yes = int(pairs_to_buy)
    shares_no = int(pairs_to_buy)
    
    cost_yes = shares_yes * price_yes
    cost_no = shares_no * price_no
    total_cost = cost_yes + cost_no
    
    return {
        "shares_yes": shares_yes,
        "shares_no": shares_no,
        "cost_yes": cost_yes,
        "cost_no": cost_no,
        "total_cost": total_cost,
        "expected_payout": min(shares_yes, shares_no),  # Garantido
        "expected_profit": min(shares_yes, shares_no) - total_cost
    }
```

---

### 3.3 Lógica de Rebalanceamento

```python
def should_rebalance(position, market_state, risk_params):
    """
    Verifica se precisa rebalancear a posição.
    """
    
    ratio = position.shares_yes / position.shares_no if position.shares_no > 0 else float('inf')
    
    # Muito desbalanceado?
    if ratio > risk_params.MAX_RATIO or ratio < risk_params.MIN_RATIO:
        return True
    
    # Spread melhorou muito? (oportunidade de adicionar)
    current_spread = market_state.price_yes + market_state.price_no - 1.0
    if current_spread < position.avg_spread - risk_params.SPREAD_IMPROVEMENT_THRESHOLD:
        return True
    
    return False


def calculate_rebalance(position, market_state, risk_params):
    """
    Calcula como rebalancear.
    """
    
    ratio = position.shares_yes / position.shares_no
    
    if ratio > risk_params.MAX_RATIO:
        # Muito YES, precisa comprar NO
        target_no = position.shares_yes / risk_params.TARGET_RATIO
        shares_to_buy = target_no - position.shares_no
        return {"action": "BUY_NO", "shares": shares_to_buy}
    
    elif ratio < risk_params.MIN_RATIO:
        # Muito NO, precisa comprar YES
        target_yes = position.shares_no * risk_params.TARGET_RATIO
        shares_to_buy = target_yes - position.shares_yes
        return {"action": "BUY_YES", "shares": shares_to_buy}
    
    return {"action": "HOLD", "shares": 0}
```

---

### 3.4 Lógica de Saída

```python
def calculate_exit(position, market_state):
    """
    No settlement, calcular resultado.
    """
    
    if market_state.outcome == "Up":
        # YES = $1.00, NO = $0.00
        payout = position.shares_yes * 1.00
    else:
        # YES = $0.00, NO = $1.00
        payout = position.shares_no * 1.00
    
    profit = payout - position.total_cost
    roi = profit / position.total_cost if position.total_cost > 0 else 0
    
    return {
        "payout": payout,
        "profit": profit,
        "roi": roi,
        "outcome": market_state.outcome
    }
```

---

## 🛡️ FASE 4: Parâmetros de Risco (Moderado)

### 4.1 Configuração de Risco

```python
# config/risk_params.py

class RiskParams:
    """Parâmetros de risco - Perfil MODERADO"""
    
    # === SPREAD ===
    MIN_SPREAD_TO_ENTER = -0.02    # Só entra se YES + NO < 0.98
    TARGET_SPREAD = -0.03          # Spread ideal: YES + NO = 0.97
    
    # === POSIÇÃO ===
    MAX_PER_MARKET_PCT = 0.15      # Máximo 15% do capital por mercado
    MAX_PER_MARKET_USD = 750       # Máximo $750 por mercado
    MIN_ORDER_SIZE = 10            # Mínimo $10 por ordem
    
    # === EXPOSIÇÃO TOTAL ===
    MAX_TOTAL_EXPOSURE = 0.70      # Máximo 70% do capital alocado
    MIN_CASH_RESERVE = 0.30        # Sempre manter 30% em caixa
    
    # === MERCADOS SIMULTÂNEOS ===
    MAX_ACTIVE_MARKETS = 5         # Máximo 5 mercados ao mesmo tempo
    
    # === EQUILÍBRIO YES/NO ===
    TARGET_RATIO = 1.0             # Ideal: YES = NO
    MAX_RATIO = 1.3                # Máximo: YES pode ser 30% maior que NO
    MIN_RATIO = 0.7                # Mínimo: YES pode ser 30% menor que NO
    
    # === TEMPO ===
    MIN_TIME_REMAINING = 120       # Mínimo 2 minutos antes do settlement
    MAX_TIME_IN_MARKET = 900       # Máximo 15 minutos (vida do mercado)
    
    # === LIQUIDEZ ===
    MIN_VOLUME = 500               # Volume mínimo do mercado
    MAX_SLIPPAGE = 0.005           # Slippage máximo aceitável (0.5%)
    
    # === STOP LOSS ===
    STOP_LOSS_PCT = 0.10           # Stop loss de 10% da posição
    
    # === REBALANCEAMENTO ===
    SPREAD_IMPROVEMENT_THRESHOLD = 0.01  # Adiciona se spread melhorar 1%
```

---

### 4.2 Matriz de Decisão

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATRIZ DE DECISÃO                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Spread (YES+NO)    │  Ação                                     │
│  ─────────────────────────────────────────────────────────────  │
│  > 0.99             │  ❌ NÃO ENTRAR (spread insuficiente)      │
│  0.98 - 0.99        │  ⚠️  CONSIDERAR (spread marginal)         │
│  0.97 - 0.98        │  ✅ ENTRAR (spread bom)                   │
│  < 0.97             │  ✅✅ ENTRAR FORTE (spread excelente)      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Ratio YES/NO       │  Ação                                     │
│  ─────────────────────────────────────────────────────────────  │
│  > 1.3              │  🔄 COMPRAR NO (rebalancear)              │
│  1.1 - 1.3          │  ⚠️  PREFERIR NO nas próximas compras     │
│  0.9 - 1.1          │  ✅ EQUILIBRADO (continuar normal)        │
│  0.7 - 0.9          │  ⚠️  PREFERIR YES nas próximas compras    │
│  < 0.7              │  🔄 COMPRAR YES (rebalancear)             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Exposição Total    │  Ação                                     │
│  ─────────────────────────────────────────────────────────────  │
│  > 70%              │  ❌ NÃO ENTRAR em novos mercados          │
│  50% - 70%          │  ⚠️  ENTRAR só se spread < 0.97           │
│  < 50%              │  ✅ LIVRE para entrar                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 FASE 5: Motor de Backtest

### 5.1 Loop Principal

```python
# src/backtest_engine.py

class BacktestEngine:
    def __init__(self, config, risk_params):
        self.config = config
        self.risk = risk_params
        self.portfolio = Portfolio(config.INITIAL_CAPITAL)
        self.trades = []
        self.metrics = MetricsCollector()
    
    def run(self, markets_df, prices_df):
        """
        Executa o backtest completo.
        """
        
        # Agrupa mercados por horário de início
        markets_by_time = markets_df.groupby('start_time')
        
        for timestamp, markets in tqdm(markets_by_time):
            
            # 1. Fechar posições de mercados que acabaram
            self._settle_expired_positions(timestamp)
            
            # 2. Atualizar preços dos mercados ativos
            self._update_active_positions(timestamp, prices_df)
            
            # 3. Verificar rebalanceamentos necessários
            self._check_rebalancing()
            
            # 4. Avaliar novos mercados
            for market in markets:
                if self._should_enter(market):
                    self._enter_market(market)
            
            # 5. Registrar estado do portfolio
            self.metrics.record_snapshot(timestamp, self.portfolio)
        
        return self._generate_report()
    
    def _settle_expired_positions(self, current_time):
        """Fecha posições de mercados que chegaram ao settlement."""
        for position in self.portfolio.active_positions:
            if position.market.end_time <= current_time:
                result = self._close_position(position)
                self.trades.append(result)
    
    def _enter_market(self, market):
        """Abre posição em um novo mercado."""
        sizing = calculate_order_size(market, self.portfolio, self.risk)
        
        if sizing['total_cost'] < self.risk.MIN_ORDER_SIZE * 2:
            return  # Ordem muito pequena
        
        # Simula execução com slippage
        execution = self._simulate_execution(market, sizing)
        
        # Registra posição
        position = Position(
            market=market,
            shares_yes=execution['shares_yes'],
            shares_no=execution['shares_no'],
            cost_yes=execution['cost_yes'],
            cost_no=execution['cost_no'],
            entry_time=market.current_time,
            entry_spread=market.spread
        )
        
        self.portfolio.add_position(position)
```

---

### 5.2 Simulação de Execução

```python
def _simulate_execution(self, market, sizing):
    """
    Simula execução com slippage realista.
    """
    
    # Modelo simples de slippage baseado em volume
    volume_ratio = sizing['total_cost'] / market.volume
    
    if volume_ratio < 0.01:
        slippage = 0.001  # 0.1% para ordens pequenas
    elif volume_ratio < 0.05:
        slippage = 0.003  # 0.3% para ordens médias
    else:
        slippage = 0.005  # 0.5% para ordens grandes
    
    # Aplica slippage (preço piora)
    executed_price_yes = market.price_yes * (1 + slippage)
    executed_price_no = market.price_no * (1 + slippage)
    
    return {
        "shares_yes": sizing['shares_yes'],
        "shares_no": sizing['shares_no'],
        "cost_yes": sizing['shares_yes'] * executed_price_yes,
        "cost_no": sizing['shares_no'] * executed_price_no,
        "slippage_paid": slippage * sizing['total_cost']
    }
```

---

### 5.3 Cálculo de Settlement

```python
def _close_position(self, position):
    """
    Fecha posição no settlement e calcula resultado.
    """
    
    market = position.market
    
    if market.outcome == "Up":
        payout = position.shares_yes * 1.00  # YES wins
    else:
        payout = position.shares_no * 1.00   # NO wins
    
    total_cost = position.cost_yes + position.cost_no
    profit = payout - total_cost
    roi = profit / total_cost if total_cost > 0 else 0
    
    # Atualiza portfolio
    self.portfolio.cash += payout
    self.portfolio.remove_position(position)
    
    return Trade(
        market_id=market.id,
        entry_time=position.entry_time,
        exit_time=market.end_time,
        shares_yes=position.shares_yes,
        shares_no=position.shares_no,
        cost=total_cost,
        payout=payout,
        profit=profit,
        roi=roi,
        outcome=market.outcome,
        entry_spread=position.entry_spread
    )
```

---

## 📈 FASE 6: Métricas e Análise

### 6.1 Métricas de Performance

```python
# src/metrics.py

class PerformanceMetrics:
    """Métricas de performance do backtest."""
    
    def calculate_all(self, trades, portfolio_history):
        return {
            # === RETORNO ===
            "total_return_usd": self._total_return(trades),
            "total_return_pct": self._total_return_pct(trades),
            "avg_return_per_trade": self._avg_return(trades),
            "avg_return_per_day": self._avg_daily_return(trades),
            
            # === RISCO ===
            "max_drawdown_usd": self._max_drawdown(portfolio_history),
            "max_drawdown_pct": self._max_drawdown_pct(portfolio_history),
            "volatility": self._volatility(portfolio_history),
            "sharpe_ratio": self._sharpe_ratio(portfolio_history),
            "sortino_ratio": self._sortino_ratio(portfolio_history),
            
            # === TRADES ===
            "total_trades": len(trades),
            "winning_trades": self._count_winners(trades),
            "losing_trades": self._count_losers(trades),
            "win_rate": self._win_rate(trades),
            "avg_winner": self._avg_winner(trades),
            "avg_loser": self._avg_loser(trades),
            "profit_factor": self._profit_factor(trades),
            
            # === SPREAD ===
            "avg_entry_spread": self._avg_entry_spread(trades),
            "avg_spread_captured": self._avg_spread_captured(trades),
            
            # === EQUILÍBRIO ===
            "avg_yes_no_ratio": self._avg_ratio(trades),
            "pct_balanced_trades": self._pct_balanced(trades),
            
            # === EXPOSIÇÃO ===
            "avg_exposure": self._avg_exposure(portfolio_history),
            "max_exposure": self._max_exposure(portfolio_history),
            "avg_markets_active": self._avg_active_markets(portfolio_history),
        }
```

---

### 6.2 Métricas Esperadas (Benchmark)

```python
# Baseado na análise do gabagool22 e características do mercado

EXPECTED_METRICS = {
    # Se a estratégia funcionar bem:
    "target_monthly_return": 0.05,     # 5% ao mês
    "target_win_rate": 0.85,           # 85% dos trades lucrativos
    "target_sharpe": 2.0,              # Sharpe ratio > 2
    "target_max_drawdown": 0.10,       # Max drawdown < 10%
    "target_profit_factor": 3.0,       # Profit factor > 3
    
    # Realista (considerando competição):
    "realistic_monthly_return": 0.02,  # 2% ao mês
    "realistic_win_rate": 0.70,        # 70% win rate
    "realistic_sharpe": 1.0,           # Sharpe ~ 1
    "realistic_max_drawdown": 0.15,    # Max drawdown ~ 15%
}
```

---

### 6.3 Visualizações

```python
# src/visualizer.py

class BacktestVisualizer:
    
    def generate_all_charts(self, results):
        """Gera todos os gráficos do backtest."""
        
        self.plot_equity_curve(results)          # Curva de capital
        self.plot_drawdown(results)              # Drawdown ao longo do tempo
        self.plot_returns_distribution(results)  # Histograma de retornos
        self.plot_monthly_returns(results)       # Retornos mensais
        self.plot_spread_distribution(results)   # Distribuição de spreads capturados
        self.plot_win_rate_by_spread(results)    # Win rate por faixa de spread
        self.plot_exposure_over_time(results)    # Exposição ao longo do tempo
        self.plot_balance_ratio(results)         # Ratio YES/NO ao longo do tempo
        self.plot_hourly_performance(results)    # Performance por hora do dia
        self.plot_trades_heatmap(results)        # Heatmap de trades (hora × dia)
```

---

## 🔧 FASE 7: Otimização de Parâmetros

### 7.1 Parâmetros a Otimizar

```python
OPTIMIZATION_GRID = {
    "MIN_SPREAD_TO_ENTER": [-0.01, -0.02, -0.03, -0.04],
    "MAX_PER_MARKET_PCT": [0.10, 0.15, 0.20, 0.25],
    "MAX_TOTAL_EXPOSURE": [0.50, 0.60, 0.70, 0.80],
    "MAX_RATIO": [1.2, 1.3, 1.5, 2.0],
    "MAX_ACTIVE_MARKETS": [3, 5, 7, 10],
}
```

### 7.2 Walk-Forward Optimization

```python
def walk_forward_optimization(data, params_grid, window_size=30, step_size=7):
    """
    Otimização walk-forward para evitar overfitting.
    
    - Treina em janela de 30 dias
    - Testa nos próximos 7 dias
    - Move a janela e repete
    """
    
    results = []
    
    for train_start in range(0, len(data) - window_size - step_size, step_size):
        train_end = train_start + window_size
        test_end = train_end + step_size
        
        train_data = data[train_start:train_end]
        test_data = data[train_end:test_end]
        
        # Encontra melhores parâmetros no treino
        best_params = grid_search(train_data, params_grid)
        
        # Testa com esses parâmetros
        test_result = backtest(test_data, best_params)
        
        results.append({
            "train_period": (train_start, train_end),
            "test_period": (train_end, test_end),
            "best_params": best_params,
            "test_performance": test_result
        })
    
    return results
```

---

## 📋 FASE 8: Relatório Final

### 8.1 Template do Relatório

```markdown
# 📊 Backtest Report: SOL 15-min AMM Strategy

## Executive Summary
- **Período:** {start_date} a {end_date} ({n_days} dias)
- **Capital Inicial:** ${initial_capital}
- **Capital Final:** ${final_capital}
- **Retorno Total:** {total_return_pct}%
- **Sharpe Ratio:** {sharpe}

## Performance Metrics
| Métrica | Valor | Benchmark |
|---------|-------|-----------|
| Retorno Total | X% | 6% (2%/mês) |
| Win Rate | X% | 70% |
| Profit Factor | X | 3.0 |
| Max Drawdown | X% | 15% |
| Sharpe Ratio | X | 1.0 |

## Trade Statistics
- Total de Trades: {n_trades}
- Trades Vencedores: {n_winners} ({win_rate}%)
- Trades Perdedores: {n_losers}
- Maior Ganho: ${max_win}
- Maior Perda: ${max_loss}

## Risk Analysis
- Max Drawdown: ${max_dd} ({max_dd_pct}%)
- Avg Exposure: {avg_exposure}%
- Max Exposure: {max_exposure}%

## Spread Analysis
- Avg Entry Spread: {avg_entry_spread}
- Avg Spread Captured: {avg_spread_captured}
- Best Spread: {best_spread}

## Recommendations
1. {recommendation_1}
2. {recommendation_2}
3. {recommendation_3}

## Charts
[Equity Curve]
[Drawdown Chart]
[Returns Distribution]
[Monthly Performance]
```

---

## ✅ FASE 9: Checklist de Execução

### 9.1 Ordem de Implementação

```
□ FASE 0: Setup
  □ Criar estrutura de pastas
  □ Instalar dependências
  □ Configurar settings.py
  □ Configurar risk_params.py

□ FASE 1: Dados
  □ Implementar fetch_sol_15min_markets()
  □ Implementar fetch_price_history()
  □ Coletar dados de 3 meses
  □ Validar dados coletados
  □ Salvar em formato eficiente (parquet)

□ FASE 2: Análise
  □ Criar notebook exploratório
  □ Calcular estatísticas de spread
  □ Identificar padrões temporais
  □ Documentar findings

□ FASE 3-4: Estratégia e Risco
  □ Implementar lógica de entrada
  □ Implementar lógica de sizing
  □ Implementar lógica de rebalanceamento
  □ Implementar gestão de risco
  □ Testar com dados sintéticos

□ FASE 5: Backtest
  □ Implementar BacktestEngine
  □ Implementar simulação de execução
  □ Implementar settlement
  □ Rodar backtest completo
  □ Validar resultados

□ FASE 6: Métricas
  □ Implementar todas as métricas
  □ Gerar visualizações
  □ Comparar com benchmarks

□ FASE 7: Otimização
  □ Definir grid de parâmetros
  □ Rodar walk-forward
  □ Identificar parâmetros ótimos

□ FASE 8: Relatório
  □ Gerar relatório final
  □ Documentar conclusões
  □ Listar próximos passos
```

---

## 🚨 Riscos e Limitações

### Limitações do Backtest

| Limitação | Impacto | Mitigação |
|-----------|---------|-----------|
| **Sem orderbook histórico** | Slippage pode ser subestimado | Modelo conservador de slippage |
| **Latência não simulada** | Bot real será mais lento | Adicionar delay artificial |
| **Competição não modelada** | Outros bots podem pegar oportunidades | Assumir fill rate < 100% |
| **Dados podem ter gaps** | Mercados podem estar faltando | Documentar gaps encontrados |

### Riscos do Trading Real

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| **Bug no código** | Média | Alto | Testes extensivos |
| **API fora do ar** | Baixa | Médio | Retry logic, alertas |
| **Spread insuficiente** | Alta | Médio | Threshold dinâmico |
| **Mercado muito competitivo** | Alta | Alto | Monitorar e ajustar |

---

## 📞 Próximos Passos Após Backtest

```
SE backtest for positivo (Sharpe > 1, Win Rate > 60%):
  1. Paper trading por 2 semanas
  2. Trading real com 10% do capital
  3. Escalar gradualmente

SE backtest for negativo:
  1. Analisar onde a estratégia falha
  2. Ajustar parâmetros ou lógica
  3. Re-testar
  4. Se continuar negativo, pivotar estratégia
```

---

## 📝 Notas para Claude Code

### Prioridades
1. **Qualidade dos dados** > Velocidade de execução
2. **Validação** em cada etapa
3. **Logs detalhados** para debug
4. **Código modular** e testável

### Comandos Úteis

```bash
# Rodar coleta de dados
python -m src.data_collector --asset SOL --days 90

# Rodar backtest
python main.py --config config/settings.py --risk config/risk_params.py

# Gerar relatório
python -m src.visualizer --output reports/backtest_report.html
```

---

**Documento criado para execução no Claude Code**
**Versão:** 1.0
**Data:** Dezembro 2024
**Autor:** Claude + Leandro
