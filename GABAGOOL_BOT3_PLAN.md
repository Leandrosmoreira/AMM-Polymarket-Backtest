# PLAN.md — Gabagool Bot 3 (Polymarket CLOB)

## Visão Geral da Estratégia

**Objetivo**: Executar a estratégia **Pair-Cost** para capturar arbitragem estrutural em mercados binários do Polymarket.

**Princípio Fundamental**: Em um mercado binário, `YES + NO = $1.00` no settlement. Se conseguirmos comprar YES e NO por um custo total médio < $1.00, temos lucro garantido (locked profit).

---

## 0. Matemática da Estratégia Pair-Cost

### 0.1 Variáveis de Estado

```
Qty_YES    = Quantidade total de shares YES
Qty_NO     = Quantidade total de shares NO
Cost_YES   = Custo total gasto em YES (USD)
Cost_NO    = Custo total gasto em NO (USD)

avg_YES    = Cost_YES / Qty_YES     (preço médio de entrada YES)
avg_NO     = Cost_NO / Qty_NO       (preço médio de entrada NO)

PairCost   = avg_YES + avg_NO       (custo médio do par)
TotalCost  = Cost_YES + Cost_NO     (capital investido)
```

### 0.2 Cálculo do Lucro Travado

```
MinShares    = min(Qty_YES, Qty_NO)           # Pares completos
LockedPayout = MinShares × $1.00              # Payout garantido
LockedProfit = MinShares - (Cost_YES + Cost_NO) × (MinShares / max(Qty_YES, Qty_NO))

# Fórmula simplificada quando Qty_YES ≈ Qty_NO:
LockedProfit ≈ MinShares × (1 - PairCost)
```

### 0.3 Exemplo Numérico

```
Estado inicial:
  Qty_YES = 100,  Cost_YES = $45   → avg_YES = $0.45
  Qty_NO  = 100,  Cost_NO  = $52   → avg_NO  = $0.52

  PairCost = 0.45 + 0.52 = $0.97
  TotalCost = $97

No settlement (qualquer outcome):
  Payout = min(100, 100) × $1.00 = $100
  Profit = $100 - $97 = $3.00 (3.1% ROI garantido)
```

### 0.4 Critério de Lock (Travamento)

```python
def is_locked(state, margin=0.02):
    """
    Retorna True se a posição tem lucro travado.

    Critérios:
    1. PairCost < (1.00 - margin)  → margem de segurança
    2. LockedProfit > 0            → lucro positivo
    """
    pair_cost = state.avg_yes + state.avg_no
    min_shares = min(state.qty_yes, state.qty_no)
    locked_profit = min_shares - (state.cost_yes + state.cost_no)

    return pair_cost < (1.0 - margin) and locked_profit > 0
```

---

## 1. Arquitetura do Sistema

### 1.1 Estrutura de Diretórios

```
gabagool-bot/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configurações gerais (pydantic)
│   │   └── credentials.py       # Carrega .env (nunca commita secrets)
│   │
│   ├── clob/
│   │   ├── __init__.py
│   │   ├── client.py            # Wrapper do py-clob-client
│   │   ├── auth.py              # Assinatura e autenticação
│   │   ├── websocket.py         # Stream de orderbook/trades
│   │   └── markets.py           # Descoberta de mercados ativos
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── state.py             # PositionState (Qty, Cost, PairCost)
│   │   ├── pair_cost.py         # Lógica core da estratégia
│   │   ├── signals.py           # Sinais de entrada (cheapness)
│   │   ├── sizing.py            # Cálculo de tamanho de ordem
│   │   └── risk.py              # Limites e circuit breakers
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── orders.py            # Place/cancel/replace orders
│   │   ├── fills.py             # Tracking de execuções
│   │   └── reconciliation.py    # Sincronização estado local ↔ CLOB
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy/dataclass models
│   │   ├── store.py             # Persistência SQLite
│   │   └── migrations.py        # Schema migrations
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py            # Motor de backtest
│   │   ├── sim_orderbook.py     # Simulador de execução
│   │   └── metrics.py           # Cálculo de métricas
│   │
│   └── viz/
│       ├── __init__.py
│       ├── charts.py            # Gráficos (matplotlib/plotly)
│       └── dashboard.py         # Dashboard real-time (opcional)
│
├── scripts/
│   ├── diagnose_config.py       # Validação de credenciais
│   ├── run_paper.py             # Paper trading
│   ├── run_live.py              # Trading real
│   ├── run_backtest.py          # Backtest histórico
│   └── export_trades.py         # Exporta trades + gráficos
│
├── tests/
│   ├── test_pair_cost.py        # Testes da matemática
│   ├── test_signals.py          # Testes de sinais
│   ├── test_sizing.py           # Testes de sizing
│   └── test_execution.py        # Testes de execução
│
├── data/
│   ├── bot.sqlite               # Database local
│   └── exports/                 # CSVs e gráficos exportados
│
├── .env.example                 # Template de variáveis de ambiente
├── requirements.txt
├── Makefile
└── README.md
```

### 1.2 Dependências

```txt
# requirements.txt

# CLOB
py-clob-client>=0.15.0

# Core
pydantic>=2.0
httpx>=0.25.0
websockets>=12.0

# Data
pandas>=2.0.0
numpy>=1.24.0
sqlalchemy>=2.0.0

# Visualization
matplotlib>=3.7.0
plotly>=5.18.0
rich>=13.0.0

# Utils
python-dotenv>=1.0.0
structlog>=23.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

---

## 2. Configuração e Diagnóstico

### 2.1 Schema de Configuração

```python
# src/config/settings.py
from pydantic import BaseModel, Field
from enum import Enum

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class Settings(BaseModel):
    """Configurações validadas do bot."""

    # Modo de operação
    mode: TradingMode = TradingMode.PAPER

    # Mercado
    asset: str = "BTC"
    timeframe_minutes: int = 15

    # Capital
    initial_capital_usd: float = Field(ge=100, le=100000)

    # Estratégia
    min_spread_to_enter: float = Field(default=-0.02, ge=-0.10, le=0)
    target_pair_cost: float = Field(default=0.97, ge=0.90, le=0.99)
    lock_margin: float = Field(default=0.02, ge=0.01, le=0.05)

    # Sizing
    min_order_usd: float = Field(default=5.0, ge=1.0)
    max_order_usd: float = Field(default=25.0, le=1000.0)
    cooldown_seconds: float = Field(default=2.0, ge=0.5)

    # Risk
    max_position_usd: float = Field(default=200.0)
    max_daily_loss_usd: float = Field(default=50.0)
    max_trades_per_window: int = Field(default=20, ge=1)

    # API
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
```

### 2.2 Script de Diagnóstico

```python
# scripts/diagnose_config.py
"""
Executa checklist de validação antes de operar.

Usage: python -m scripts.diagnose_config
"""

import asyncio
from rich.console import Console
from rich.table import Table

console = Console()

async def diagnose():
    results = Table(title="Diagnóstico de Configuração")
    results.add_column("Check", style="cyan")
    results.add_column("Status", style="green")
    results.add_column("Detalhes")

    checks = [
        ("Variáveis .env", check_env_vars),
        ("Conexão HTTP CLOB", check_clob_http),
        ("Conexão WebSocket", check_clob_ws),
        ("Assinatura válida", check_signature),
        ("Leitura de orderbook", check_orderbook_read),
        ("Leitura de fills", check_fills_read),
        ("Mercado ativo encontrado", check_active_market),
        ("Database SQLite", check_database),
    ]

    all_ok = True
    for name, check_fn in checks:
        try:
            status, details = await check_fn()
            emoji = "✅" if status else "❌"
            results.add_row(name, emoji, details)
            if not status:
                all_ok = False
        except Exception as e:
            results.add_row(name, "❌", f"Erro: {e}")
            all_ok = False

    console.print(results)

    if all_ok:
        console.print("\n[bold green]Todos os checks passaram! Bot pronto para operar.[/]")
    else:
        console.print("\n[bold red]Alguns checks falharam. Corrija antes de continuar.[/]")

    return all_ok

if __name__ == "__main__":
    asyncio.run(diagnose())
```

---

## 3. Descoberta de Mercados

### 3.1 Lógica de Market Discovery

Os mercados de 15 minutos "giram" constantemente (novos IDs a cada janela). O bot precisa descobrir automaticamente o mercado ativo.

```python
# src/clob/markets.py
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx

@dataclass
class ActiveMarket:
    """Mercado ativo para trading."""
    market_id: str
    condition_id: str
    token_id_yes: str
    token_id_no: str
    question: str
    end_time: datetime
    neg_risk: bool = False

class MarketDiscovery:
    """Descobre e monitora mercados ativos de 15 minutos."""

    def __init__(self, settings):
        self.settings = settings
        self.gamma_url = settings.gamma_api_url
        self._current_market: ActiveMarket | None = None

    async def find_active_market(self) -> ActiveMarket | None:
        """
        Encontra o mercado BTC/SOL 15-min atualmente ativo.

        Critérios:
        1. Asset correto (BTC/SOL)
        2. Tipo "up_or_down" ou "15-minute"
        3. Ainda não resolvido (active=true)
        4. end_time > now + 2min (tempo suficiente para operar)
        """
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.gamma_url}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "tag": self.settings.asset.lower(),
                }
            )
            resp.raise_for_status()
            markets = resp.json()

        now = datetime.utcnow()
        min_time_remaining = timedelta(minutes=2)

        for m in markets:
            # Filtrar por tipo de mercado
            if not self._is_15min_market(m):
                continue

            end_time = datetime.fromisoformat(m["end_date_iso"].replace("Z", ""))

            # Verificar tempo restante
            if end_time - now < min_time_remaining:
                continue

            return ActiveMarket(
                market_id=m["id"],
                condition_id=m["condition_id"],
                token_id_yes=m["tokens"][0]["token_id"],
                token_id_no=m["tokens"][1]["token_id"],
                question=m["question"],
                end_time=end_time,
                neg_risk=m.get("neg_risk", False),
            )

        return None

    def _is_15min_market(self, market: dict) -> bool:
        """Verifica se é um mercado de 15 minutos."""
        question = market.get("question", "").lower()
        return (
            "15" in question and ("minute" in question or "min" in question)
            or market.get("market_type") == "15_minute"
        )

    async def wait_for_next_market(self) -> ActiveMarket:
        """Aguarda o próximo mercado ficar disponível."""
        while True:
            market = await self.find_active_market()
            if market:
                self._current_market = market
                return market
            await asyncio.sleep(5)  # Polling a cada 5s
```

---

## 4. Estado da Posição (Position State)

### 4.1 Modelo de Estado

```python
# src/strategy/state.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class Trade:
    """Registro de um trade individual."""
    timestamp: datetime
    side: str  # "YES" ou "NO"
    qty: float
    price: float
    cost: float  # qty × price
    order_id: str

@dataclass
class PositionState:
    """
    Estado completo da posição em um mercado.

    Mantém histórico de trades e calcula métricas derivadas.
    """
    market_id: str
    trades: List[Trade] = field(default_factory=list)

    # Valores computados (cache)
    _qty_yes: float = 0.0
    _qty_no: float = 0.0
    _cost_yes: float = 0.0
    _cost_no: float = 0.0

    def add_trade(self, trade: Trade):
        """Adiciona trade e atualiza estado."""
        self.trades.append(trade)

        if trade.side == "YES":
            self._qty_yes += trade.qty
            self._cost_yes += trade.cost
        else:
            self._qty_no += trade.qty
            self._cost_no += trade.cost

    @property
    def qty_yes(self) -> float:
        return self._qty_yes

    @property
    def qty_no(self) -> float:
        return self._qty_no

    @property
    def cost_yes(self) -> float:
        return self._cost_yes

    @property
    def cost_no(self) -> float:
        return self._cost_no

    @property
    def total_cost(self) -> float:
        return self._cost_yes + self._cost_no

    @property
    def avg_yes(self) -> float:
        """Preço médio de entrada YES."""
        return self._cost_yes / self._qty_yes if self._qty_yes > 0 else 0.0

    @property
    def avg_no(self) -> float:
        """Preço médio de entrada NO."""
        return self._cost_no / self._qty_no if self._qty_no > 0 else 0.0

    @property
    def pair_cost(self) -> float:
        """Custo médio do par YES+NO."""
        if self._qty_yes == 0 or self._qty_no == 0:
            return float('inf')
        return self.avg_yes + self.avg_no

    @property
    def min_shares(self) -> float:
        """Quantidade de pares completos."""
        return min(self._qty_yes, self._qty_no)

    @property
    def locked_payout(self) -> float:
        """Payout garantido no settlement."""
        return self.min_shares * 1.0

    @property
    def locked_profit(self) -> float:
        """Lucro garantido (se min_shares > 0)."""
        if self.min_shares == 0:
            return 0.0
        # Custo proporcional aos pares completos
        ratio = self.min_shares / max(self._qty_yes, self._qty_no)
        proportional_cost = self.total_cost * ratio
        return self.min_shares - proportional_cost

    @property
    def imbalance(self) -> float:
        """
        Desbalanceamento da posição.

        > 0: mais YES que NO (precisa de mais NO)
        < 0: mais NO que YES (precisa de mais YES)
        = 0: perfeitamente balanceado
        """
        return self._qty_yes - self._qty_no

    @property
    def imbalance_ratio(self) -> float:
        """Ratio YES/NO (1.0 = balanceado)."""
        if self._qty_no == 0:
            return float('inf') if self._qty_yes > 0 else 1.0
        return self._qty_yes / self._qty_no

    def is_locked(self, margin: float = 0.02) -> bool:
        """
        Verifica se a posição tem lucro travado.

        Args:
            margin: Margem de segurança (ex: 0.02 = 2%)
        """
        return (
            self.pair_cost < (1.0 - margin) and
            self.locked_profit > 0
        )

    def simulate_trade(self, side: str, qty: float, price: float) -> 'PositionState':
        """
        Simula um trade e retorna novo estado (sem modificar atual).

        Útil para avaliar se vale a pena fazer o trade.
        """
        new_state = PositionState(
            market_id=self.market_id,
            trades=self.trades.copy(),
            _qty_yes=self._qty_yes,
            _qty_no=self._qty_no,
            _cost_yes=self._cost_yes,
            _cost_no=self._cost_no,
        )

        fake_trade = Trade(
            timestamp=datetime.utcnow(),
            side=side,
            qty=qty,
            price=price,
            cost=qty * price,
            order_id="SIMULATED",
        )
        new_state.add_trade(fake_trade)

        return new_state
```

---

## 5. Sinais de Entrada (Cheapness Detection)

### 5.1 Conceito de "Cheapness"

O bot precisa detectar quando YES ou NO está "barato" o suficiente para comprar, **sem tentar prever a direção do mercado**.

```python
# src/strategy/signals.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class SignalType(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"

@dataclass
class Signal:
    """Sinal de trading gerado pelos detectores."""
    type: SignalType
    strength: float  # 0.0 a 1.0
    reason: str
    target_price: float
    suggested_qty: float

@dataclass
class OrderbookState:
    """Estado atual do orderbook."""
    best_bid_yes: float
    best_ask_yes: float
    best_bid_no: float
    best_ask_no: float
    depth_yes: float  # Volume no best ask
    depth_no: float   # Volume no best ask
    mid_yes: float
    mid_no: float
    spread_yes: float
    spread_no: float

class SignalEngine:
    """
    Motor de sinais para detectar oportunidades.

    Combina múltiplos sinais para decisão final.
    """

    def __init__(self, settings):
        self.settings = settings
        self.signals = [
            PairParitySignal(settings),
            ImbalanceSignal(settings),
            SpreadCompressionSignal(settings),
        ]

    def evaluate(
        self,
        orderbook: OrderbookState,
        position: PositionState,
    ) -> Optional[Signal]:
        """
        Avalia todos os sinais e retorna o melhor.

        Prioridade:
        1. Rebalanceamento (se posição muito desbalanceada)
        2. Pair Parity (se YES+NO < threshold)
        3. Spread Compression (se um lado ficou muito barato)
        """

        # Verificar se já está travado
        if position.is_locked(self.settings.lock_margin):
            return None  # Não precisa mais comprar

        # Coletar sinais de todos os detectores
        all_signals = []
        for signal_detector in self.signals:
            signal = signal_detector.evaluate(orderbook, position)
            if signal and signal.type != SignalType.HOLD:
                all_signals.append(signal)

        if not all_signals:
            return None

        # Retornar sinal mais forte
        return max(all_signals, key=lambda s: s.strength)


class PairParitySignal:
    """
    Sinal baseado em paridade YES+NO.

    Se ask_YES + ask_NO < 1.00, há oportunidade de arbitragem.
    Compra o lado que mais reduz o PairCost.
    """

    def __init__(self, settings):
        self.min_spread = abs(settings.min_spread_to_enter)  # ex: 0.02

    def evaluate(
        self,
        orderbook: OrderbookState,
        position: PositionState,
    ) -> Optional[Signal]:

        # Custo de comprar 1 YES + 1 NO agora
        current_pair_cost = orderbook.best_ask_yes + orderbook.best_ask_no

        # Spread disponível
        spread = 1.0 - current_pair_cost

        if spread < self.min_spread:
            return Signal(
                type=SignalType.HOLD,
                strength=0.0,
                reason=f"Spread insuficiente: {spread:.4f} < {self.min_spread}",
                target_price=0,
                suggested_qty=0,
            )

        # Decidir qual lado comprar (o que mais reduz PairCost)
        if position.qty_yes == 0 and position.qty_no == 0:
            # Posição vazia: comprar ambos proporcionalmente
            # Começar pelo mais barato
            if orderbook.best_ask_yes <= orderbook.best_ask_no:
                side = SignalType.BUY_YES
                price = orderbook.best_ask_yes
            else:
                side = SignalType.BUY_NO
                price = orderbook.best_ask_no
        else:
            # Posição existente: comprar lado que melhora PairCost
            side, price = self._choose_side(orderbook, position)

        return Signal(
            type=side,
            strength=min(spread / 0.05, 1.0),  # Normaliza: 5% spread = força máxima
            reason=f"Pair parity: spread={spread:.4f}",
            target_price=price,
            suggested_qty=0,  # Sizing decide
        )

    def _choose_side(
        self,
        orderbook: OrderbookState,
        position: PositionState,
    ) -> tuple[SignalType, float]:
        """Escolhe lado que mais reduz PairCost ou balanceia posição."""

        # Simular compra de YES
        sim_yes = position.simulate_trade("YES", 1.0, orderbook.best_ask_yes)

        # Simular compra de NO
        sim_no = position.simulate_trade("NO", 1.0, orderbook.best_ask_no)

        # Preferir lado que resulta em menor PairCost
        if sim_yes.pair_cost < sim_no.pair_cost:
            return SignalType.BUY_YES, orderbook.best_ask_yes
        else:
            return SignalType.BUY_NO, orderbook.best_ask_no


class ImbalanceSignal:
    """
    Sinal de rebalanceamento.

    Se a posição está muito desbalanceada (ex: muito YES, pouco NO),
    prioriza comprar o lado deficiente.
    """

    def __init__(self, settings):
        self.max_imbalance_ratio = 1.5  # Máximo 50% de desbalanceamento

    def evaluate(
        self,
        orderbook: OrderbookState,
        position: PositionState,
    ) -> Optional[Signal]:

        if position.qty_yes == 0 and position.qty_no == 0:
            return None  # Sem posição, sem rebalanceamento

        ratio = position.imbalance_ratio

        if 1/self.max_imbalance_ratio <= ratio <= self.max_imbalance_ratio:
            return None  # Balanceado o suficiente

        if ratio > self.max_imbalance_ratio:
            # Muito YES, precisa de NO
            return Signal(
                type=SignalType.BUY_NO,
                strength=0.8,  # Alta prioridade
                reason=f"Rebalanceamento: ratio={ratio:.2f}, precisa de NO",
                target_price=orderbook.best_ask_no,
                suggested_qty=position.imbalance,  # Quantidade para balancear
            )
        else:
            # Muito NO, precisa de YES
            return Signal(
                type=SignalType.BUY_YES,
                strength=0.8,
                reason=f"Rebalanceamento: ratio={ratio:.2f}, precisa de YES",
                target_price=orderbook.best_ask_yes,
                suggested_qty=abs(position.imbalance),
            )


class SpreadCompressionSignal:
    """
    Detecta quando um lado "comprimiu" (ficou muito mais barato que o outro).

    Ex: YES caiu para 0.30 mas NO não subiu proporcionalmente (ainda 0.65).
    Isso indica oportunidade de comprar YES barato.
    """

    def __init__(self, settings):
        self.compression_threshold = 0.05  # 5% de diferença

    def evaluate(
        self,
        orderbook: OrderbookState,
        position: PositionState,
    ) -> Optional[Signal]:

        # Preço "justo" teórico de cada lado
        fair_yes = 0.50  # Sem informação, assumir 50/50
        fair_no = 0.50

        # Desvio do preço justo
        yes_deviation = fair_yes - orderbook.best_ask_yes
        no_deviation = fair_no - orderbook.best_ask_no

        # Se YES está muito mais barato que NO (relativo ao justo)
        if yes_deviation - no_deviation > self.compression_threshold:
            return Signal(
                type=SignalType.BUY_YES,
                strength=0.6,
                reason=f"Spread compression: YES subvalorizado",
                target_price=orderbook.best_ask_yes,
                suggested_qty=0,
            )

        if no_deviation - yes_deviation > self.compression_threshold:
            return Signal(
                type=SignalType.BUY_NO,
                strength=0.6,
                reason=f"Spread compression: NO subvalorizado",
                target_price=orderbook.best_ask_no,
                suggested_qty=0,
            )

        return None
```

---

## 6. Sizing e Execução

### 6.1 Cálculo de Tamanho de Ordem

```python
# src/strategy/sizing.py
from dataclasses import dataclass

@dataclass
class OrderSizing:
    """Resultado do cálculo de sizing."""
    qty: float
    price: float
    cost: float
    reason: str
    approved: bool

class Sizer:
    """
    Calcula tamanho de ordem baseado em:
    1. Capital disponível
    2. Limites de risco
    3. Estado da posição
    4. Liquidez do orderbook
    """

    def __init__(self, settings):
        self.settings = settings

    def calculate(
        self,
        signal: Signal,
        position: PositionState,
        available_capital: float,
        orderbook_depth: float,
    ) -> OrderSizing:
        """
        Calcula quantidade ótima para o trade.

        Regras:
        1. Nunca exceder max_order_usd
        2. Nunca exceder capital disponível
        3. Nunca exceder % do depth do orderbook (evitar slippage)
        4. Ajustar para balancear posição
        """

        price = signal.target_price

        # Limite base
        max_cost = self.settings.max_order_usd

        # Limite por capital disponível
        max_cost = min(max_cost, available_capital * 0.5)

        # Limite por liquidez (não pegar mais de 20% do depth)
        max_cost = min(max_cost, orderbook_depth * price * 0.2)

        # Verificar mínimo
        if max_cost < self.settings.min_order_usd:
            return OrderSizing(
                qty=0,
                price=price,
                cost=0,
                reason=f"Capital insuficiente: {max_cost:.2f} < {self.settings.min_order_usd}",
                approved=False,
            )

        # Ajustar para balanceamento
        if signal.suggested_qty > 0:
            # Sinal sugeriu quantidade específica (rebalanceamento)
            target_cost = signal.suggested_qty * price
            max_cost = min(max_cost, target_cost)

        # Calcular quantidade
        qty = max_cost / price
        cost = qty * price

        return OrderSizing(
            qty=round(qty, 2),
            price=price,
            cost=round(cost, 2),
            reason=f"Sizing aprovado: {qty:.2f} @ ${price:.4f}",
            approved=True,
        )
```

### 6.2 Execução de Ordens

```python
# src/execution/orders.py
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import asyncio

class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"  # Na prática, limit agressivo

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Ordem enviada ao CLOB."""
    id: str
    market_id: str
    token_id: str
    side: str  # "YES" ou "NO"
    qty: float
    price: float
    order_type: OrderType
    status: OrderStatus
    created_at: datetime
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0

class OrderExecutor:
    """
    Executa ordens no CLOB com retry e error handling.
    """

    def __init__(self, clob_client, settings):
        self.clob = clob_client
        self.settings = settings
        self.pending_orders: dict[str, Order] = {}

    async def place_limit_order(
        self,
        market_id: str,
        token_id: str,
        side: str,
        qty: float,
        price: float,
    ) -> Order:
        """
        Envia limit order ao CLOB.

        Estratégia:
        1. Colocar limit no best ask (ou melhor)
        2. Aguardar fill por N segundos
        3. Se não preencher, cancelar e tentar de novo
        """

        order_payload = {
            "market": market_id,
            "tokenID": token_id,
            "side": "BUY",  # Sempre comprando
            "price": price,
            "size": qty,
            "type": "GTC",  # Good Till Cancelled
        }

        try:
            response = await self.clob.create_order(order_payload)

            order = Order(
                id=response["orderID"],
                market_id=market_id,
                token_id=token_id,
                side=side,
                qty=qty,
                price=price,
                order_type=OrderType.LIMIT,
                status=OrderStatus.OPEN,
                created_at=datetime.utcnow(),
            )

            self.pending_orders[order.id] = order
            return order

        except Exception as e:
            # Log erro e retornar ordem rejeitada
            return Order(
                id="",
                market_id=market_id,
                token_id=token_id,
                side=side,
                qty=qty,
                price=price,
                order_type=OrderType.LIMIT,
                status=OrderStatus.REJECTED,
                created_at=datetime.utcnow(),
            )

    async def wait_for_fill(
        self,
        order: Order,
        timeout_seconds: float = 5.0,
    ) -> Order:
        """
        Aguarda order ser preenchida ou timeout.
        """
        start = datetime.utcnow()

        while (datetime.utcnow() - start).total_seconds() < timeout_seconds:
            # Checar status da ordem
            status = await self.clob.get_order(order.id)

            if status["status"] == "FILLED":
                order.status = OrderStatus.FILLED
                order.filled_qty = float(status["filledSize"])
                order.filled_avg_price = float(status["avgPrice"])
                return order

            if status["status"] == "PARTIALLY_FILLED":
                order.status = OrderStatus.PARTIALLY_FILLED
                order.filled_qty = float(status["filledSize"])

            await asyncio.sleep(0.5)

        # Timeout: cancelar ordem não preenchida
        if order.status != OrderStatus.FILLED:
            await self.cancel_order(order)

        return order

    async def cancel_order(self, order: Order) -> bool:
        """Cancela ordem pendente."""
        try:
            await self.clob.cancel_order(order.id)
            order.status = OrderStatus.CANCELLED
            return True
        except:
            return False
```

---

## 7. Gestão de Risco

### 7.1 Risk Manager

```python
# src/strategy/risk.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class RiskCheck:
    """Resultado de uma verificação de risco."""
    approved: bool
    reason: str
    limit_name: str
    current_value: float
    max_value: float

class RiskManager:
    """
    Gerencia limites de risco e circuit breakers.

    Verifica antes de cada ordem:
    1. Exposição total
    2. Loss diário
    3. Trades por janela
    4. Latência
    5. Erros de API
    """

    def __init__(self, settings):
        self.settings = settings
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.window_trades = 0
        self.consecutive_errors = 0
        self.last_error_time: Optional[datetime] = None
        self.is_paused = False
        self.pause_reason = ""

    def check_all(
        self,
        proposed_cost: float,
        current_position_value: float,
        available_capital: float,
    ) -> RiskCheck:
        """
        Executa todos os checks de risco.

        Retorna primeiro check que falhar, ou approved=True.
        """

        checks = [
            self._check_circuit_breaker(),
            self._check_exposure(proposed_cost, current_position_value),
            self._check_daily_loss(),
            self._check_trades_per_window(),
            self._check_capital(proposed_cost, available_capital),
        ]

        for check in checks:
            if not check.approved:
                return check

        return RiskCheck(
            approved=True,
            reason="Todos os checks passaram",
            limit_name="all",
            current_value=0,
            max_value=0,
        )

    def _check_circuit_breaker(self) -> RiskCheck:
        """Verifica se circuit breaker está ativo."""
        if self.is_paused:
            return RiskCheck(
                approved=False,
                reason=f"Circuit breaker ativo: {self.pause_reason}",
                limit_name="circuit_breaker",
                current_value=1,
                max_value=0,
            )
        return RiskCheck(True, "", "circuit_breaker", 0, 0)

    def _check_exposure(
        self,
        proposed_cost: float,
        current_position_value: float,
    ) -> RiskCheck:
        """Verifica exposição total."""
        new_exposure = current_position_value + proposed_cost
        max_exposure = self.settings.max_position_usd

        if new_exposure > max_exposure:
            return RiskCheck(
                approved=False,
                reason=f"Exposição excede limite: ${new_exposure:.2f} > ${max_exposure:.2f}",
                limit_name="max_exposure",
                current_value=new_exposure,
                max_value=max_exposure,
            )
        return RiskCheck(True, "", "max_exposure", new_exposure, max_exposure)

    def _check_daily_loss(self) -> RiskCheck:
        """Verifica loss diário."""
        if self.daily_pnl < -self.settings.max_daily_loss_usd:
            return RiskCheck(
                approved=False,
                reason=f"Loss diário excede limite: ${self.daily_pnl:.2f}",
                limit_name="daily_loss",
                current_value=abs(self.daily_pnl),
                max_value=self.settings.max_daily_loss_usd,
            )
        return RiskCheck(True, "", "daily_loss", abs(self.daily_pnl), self.settings.max_daily_loss_usd)

    def _check_trades_per_window(self) -> RiskCheck:
        """Verifica trades por janela de 15 min."""
        if self.window_trades >= self.settings.max_trades_per_window:
            return RiskCheck(
                approved=False,
                reason=f"Máximo de trades por janela atingido: {self.window_trades}",
                limit_name="trades_per_window",
                current_value=self.window_trades,
                max_value=self.settings.max_trades_per_window,
            )
        return RiskCheck(True, "", "trades_per_window", self.window_trades, self.settings.max_trades_per_window)

    def _check_capital(
        self,
        proposed_cost: float,
        available_capital: float,
    ) -> RiskCheck:
        """Verifica capital disponível."""
        if proposed_cost > available_capital:
            return RiskCheck(
                approved=False,
                reason=f"Capital insuficiente: ${proposed_cost:.2f} > ${available_capital:.2f}",
                limit_name="available_capital",
                current_value=proposed_cost,
                max_value=available_capital,
            )
        return RiskCheck(True, "", "available_capital", proposed_cost, available_capital)

    def record_error(self, error_type: str):
        """Registra erro de API e ativa circuit breaker se necessário."""
        self.consecutive_errors += 1
        self.last_error_time = datetime.utcnow()

        if self.consecutive_errors >= 5:
            self.is_paused = True
            self.pause_reason = f"5 erros consecutivos: {error_type}"

    def record_success(self):
        """Registra sucesso e reseta contador de erros."""
        self.consecutive_errors = 0

    def record_trade(self, pnl: float = 0.0):
        """Registra trade executado."""
        self.daily_trades += 1
        self.window_trades += 1
        self.daily_pnl += pnl

    def reset_window(self):
        """Reseta contadores para nova janela de 15 min."""
        self.window_trades = 0

    def reset_daily(self):
        """Reseta contadores diários."""
        self.daily_trades = 0
        self.daily_pnl = 0.0
```

---

## 8. Backtest Engine

### 8.1 Motor de Backtest

```python
# src/backtest/engine.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import pandas as pd

@dataclass
class BacktestConfig:
    """Configuração do backtest."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    settings: 'Settings'

@dataclass
class BacktestResult:
    """Resultado do backtest."""
    total_return_pct: float
    total_return_usd: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    lock_rate: float  # % de janelas com lock
    avg_pair_cost: float
    avg_locked_profit: float
    trades: List[Dict]
    equity_curve: pd.DataFrame

class BacktestEngine:
    """
    Motor de backtest para estratégia Pair-Cost.

    Simula execução em dados históricos de orderbook.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.capital = config.initial_capital
        self.trades: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.windows_with_lock = 0
        self.total_windows = 0

    def run(self, historical_data: pd.DataFrame) -> BacktestResult:
        """
        Executa backtest em dados históricos.

        Args:
            historical_data: DataFrame com colunas:
                - timestamp
                - market_id
                - best_ask_yes
                - best_ask_no
                - best_bid_yes
                - best_bid_no
                - depth_yes
                - depth_no
        """

        # Agrupar por mercado (cada mercado = 1 janela de 15 min)
        markets = historical_data.groupby('market_id')

        for market_id, market_data in markets:
            self.total_windows += 1
            result = self._simulate_window(market_id, market_data)

            if result['locked']:
                self.windows_with_lock += 1

            self.trades.append(result)
            self.capital += result['pnl']

            self.equity_history.append({
                'timestamp': result['end_time'],
                'capital': self.capital,
                'pnl': result['pnl'],
            })

        return self._calculate_results()

    def _simulate_window(
        self,
        market_id: str,
        data: pd.DataFrame,
    ) -> Dict:
        """
        Simula trading em uma janela de 15 minutos.
        """

        position = PositionState(market_id=market_id)
        signal_engine = SignalEngine(self.config.settings)
        sizer = Sizer(self.config.settings)

        start_time = data['timestamp'].min()
        end_time = data['timestamp'].max()

        for _, row in data.iterrows():
            # Criar estado do orderbook
            orderbook = OrderbookState(
                best_bid_yes=row['best_bid_yes'],
                best_ask_yes=row['best_ask_yes'],
                best_bid_no=row['best_bid_no'],
                best_ask_no=row['best_ask_no'],
                depth_yes=row['depth_yes'],
                depth_no=row['depth_no'],
                mid_yes=(row['best_bid_yes'] + row['best_ask_yes']) / 2,
                mid_no=(row['best_bid_no'] + row['best_ask_no']) / 2,
                spread_yes=row['best_ask_yes'] - row['best_bid_yes'],
                spread_no=row['best_ask_no'] - row['best_bid_no'],
            )

            # Verificar se já está travado
            if position.is_locked(self.config.settings.lock_margin):
                break  # Parar de operar nesta janela

            # Gerar sinal
            signal = signal_engine.evaluate(orderbook, position)

            if signal is None or signal.type == SignalType.HOLD:
                continue

            # Calcular sizing
            sizing = sizer.calculate(
                signal=signal,
                position=position,
                available_capital=self.capital - position.total_cost,
                orderbook_depth=orderbook.depth_yes if signal.type == SignalType.BUY_YES else orderbook.depth_no,
            )

            if not sizing.approved:
                continue

            # Simular execução com slippage
            executed_price = self._apply_slippage(sizing.price, sizing.qty, orderbook)

            # Registrar trade
            trade = Trade(
                timestamp=row['timestamp'],
                side="YES" if signal.type == SignalType.BUY_YES else "NO",
                qty=sizing.qty,
                price=executed_price,
                cost=sizing.qty * executed_price,
                order_id=f"SIM_{market_id}_{len(position.trades)}",
            )
            position.add_trade(trade)

        # Calcular resultado no settlement
        # Assumir 50/50 de probabilidade de YES/NO ganhar
        # Para backtest mais preciso, usar outcome real
        pnl = position.locked_profit if position.is_locked() else -position.total_cost * 0.1

        return {
            'market_id': market_id,
            'start_time': start_time,
            'end_time': end_time,
            'trades': len(position.trades),
            'qty_yes': position.qty_yes,
            'qty_no': position.qty_no,
            'cost_yes': position.cost_yes,
            'cost_no': position.cost_no,
            'total_cost': position.total_cost,
            'pair_cost': position.pair_cost if position.pair_cost != float('inf') else 0,
            'locked': position.is_locked(),
            'locked_profit': position.locked_profit,
            'pnl': pnl,
        }

    def _apply_slippage(
        self,
        price: float,
        qty: float,
        orderbook: OrderbookState,
    ) -> float:
        """Simula slippage baseado em tamanho da ordem vs liquidez."""

        # Modelo simples: slippage proporcional ao % do depth consumido
        depth = max(orderbook.depth_yes, orderbook.depth_no)
        consumption_ratio = qty / depth if depth > 0 else 1.0

        # Slippage: 0.1% base + 0.5% por cada 10% do depth consumido
        slippage = 0.001 + (consumption_ratio * 0.05)
        slippage = min(slippage, 0.02)  # Cap em 2%

        return price * (1 + slippage)

    def _calculate_results(self) -> BacktestResult:
        """Calcula métricas finais do backtest."""

        df = pd.DataFrame(self.trades)
        equity = pd.DataFrame(self.equity_history)

        total_return_usd = self.capital - self.config.initial_capital
        total_return_pct = (total_return_usd / self.config.initial_capital) * 100

        # Win rate
        winning_trades = len(df[df['pnl'] > 0])
        win_rate = (winning_trades / len(df)) * 100 if len(df) > 0 else 0

        # Lock rate
        lock_rate = (self.windows_with_lock / self.total_windows) * 100 if self.total_windows > 0 else 0

        # Drawdown
        equity['peak'] = equity['capital'].cummax()
        equity['drawdown'] = (equity['capital'] - equity['peak']) / equity['peak']
        max_drawdown_pct = abs(equity['drawdown'].min()) * 100

        # Sharpe (simplificado)
        returns = equity['capital'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0

        # Médias
        avg_pair_cost = df['pair_cost'].mean()
        avg_locked_profit = df[df['locked']]['locked_profit'].mean() if len(df[df['locked']]) > 0 else 0

        return BacktestResult(
            total_return_pct=round(total_return_pct, 2),
            total_return_usd=round(total_return_usd, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(df),
            lock_rate=round(lock_rate, 2),
            avg_pair_cost=round(avg_pair_cost, 4),
            avg_locked_profit=round(avg_locked_profit, 2),
            trades=self.trades,
            equity_curve=equity,
        )
```

---

## 9. Visualização

### 9.1 Gráficos de Análise

```python
# src/viz/charts.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from typing import List, Dict

class GabagoolCharts:
    """
    Gera visualizações no estilo do artigo original.

    4 camadas principais:
    1. Pontos de trades YES/NO
    2. Shares cumulativas
    3. Dólares cumulativos
    4. Curvas de exposição
    """

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize

    def plot_window_analysis(
        self,
        trades: List[Trade],
        position: PositionState,
        save_path: str = None,
    ):
        """
        Gera gráfico completo de uma janela de 15 min.
        """

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Gabagool Bot - Análise de Janela\nPairCost: {position.pair_cost:.4f} | Locked: {position.is_locked()}',
                     fontsize=14, fontweight='bold')

        # 1. Trade dots (preços ao longo do tempo)
        self._plot_trade_dots(axes[0, 0], trades)

        # 2. Shares cumulativas
        self._plot_cumulative_shares(axes[0, 1], trades)

        # 3. Custo cumulativo
        self._plot_cumulative_cost(axes[1, 0], trades)

        # 4. Curvas de exposição
        self._plot_exposure_curves(axes[1, 1], trades, position)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _plot_trade_dots(self, ax, trades: List[Trade]):
        """Gráfico de pontos de trades."""

        yes_trades = [t for t in trades if t.side == "YES"]
        no_trades = [t for t in trades if t.side == "NO"]

        if yes_trades:
            ax.scatter(
                [t.timestamp for t in yes_trades],
                [t.price for t in yes_trades],
                c='green', s=50, alpha=0.7, label='YES',
                marker='^'
            )

        if no_trades:
            ax.scatter(
                [t.timestamp for t in no_trades],
                [t.price for t in no_trades],
                c='red', s=50, alpha=0.7, label='NO',
                marker='v'
            )

        ax.set_ylabel('Preço')
        ax.set_title('Trades Executados')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_cumulative_shares(self, ax, trades: List[Trade]):
        """Gráfico de shares cumulativas."""

        times = []
        cum_yes = []
        cum_no = []

        total_yes = 0
        total_no = 0

        for t in sorted(trades, key=lambda x: x.timestamp):
            times.append(t.timestamp)
            if t.side == "YES":
                total_yes += t.qty
            else:
                total_no += t.qty
            cum_yes.append(total_yes)
            cum_no.append(total_no)

        ax.plot(times, cum_yes, 'g-', linewidth=2, label='YES Shares')
        ax.plot(times, cum_no, 'r-', linewidth=2, label='NO Shares')
        ax.fill_between(times, cum_yes, alpha=0.2, color='green')
        ax.fill_between(times, cum_no, alpha=0.2, color='red')

        ax.set_ylabel('Shares')
        ax.set_title('Shares Cumulativas')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_cumulative_cost(self, ax, trades: List[Trade]):
        """Gráfico de custo cumulativo."""

        times = []
        cum_cost_yes = []
        cum_cost_no = []
        cum_total = []

        total_yes = 0
        total_no = 0

        for t in sorted(trades, key=lambda x: x.timestamp):
            times.append(t.timestamp)
            if t.side == "YES":
                total_yes += t.cost
            else:
                total_no += t.cost
            cum_cost_yes.append(total_yes)
            cum_cost_no.append(total_no)
            cum_total.append(total_yes + total_no)

        ax.plot(times, cum_cost_yes, 'g--', linewidth=1.5, label='Custo YES')
        ax.plot(times, cum_cost_no, 'r--', linewidth=1.5, label='Custo NO')
        ax.plot(times, cum_total, 'b-', linewidth=2, label='Custo Total')

        ax.set_ylabel('USD')
        ax.set_title('Custo Cumulativo')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_exposure_curves(self, ax, trades: List[Trade], position: PositionState):
        """Gráfico de exposição: custo vs payout potencial."""

        times = []
        costs = []
        payouts = []
        profits = []

        running_position = PositionState(market_id=position.market_id)

        for t in sorted(trades, key=lambda x: x.timestamp):
            running_position.add_trade(t)
            times.append(t.timestamp)
            costs.append(running_position.total_cost)
            payouts.append(running_position.locked_payout)
            profits.append(running_position.locked_profit)

        ax.plot(times, costs, 'b-', linewidth=2, label='Custo Total')
        ax.plot(times, payouts, 'g-', linewidth=2, label='Payout Garantido')
        ax.fill_between(times, costs, payouts,
                        where=[p > c for p, c in zip(payouts, costs)],
                        alpha=0.3, color='green', label='Lucro Travado')
        ax.fill_between(times, costs, payouts,
                        where=[p <= c for p, c in zip(payouts, costs)],
                        alpha=0.3, color='red', label='Em Risco')

        ax.set_ylabel('USD')
        ax.set_title('Exposição vs Payout')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_backtest_summary(
        self,
        result: 'BacktestResult',
        save_path: str = None,
    ):
        """Gera gráfico resumo do backtest."""

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Backtest Summary - Gabagool Bot', fontsize=14, fontweight='bold')

        # 1. Equity curve
        ax1 = axes[0, 0]
        ax1.plot(result.equity_curve['timestamp'], result.equity_curve['capital'], 'b-', linewidth=2)
        ax1.fill_between(result.equity_curve['timestamp'], result.equity_curve['capital'], alpha=0.3)
        ax1.set_title(f'Equity Curve (Return: {result.total_return_pct}%)')
        ax1.set_ylabel('Capital (USD)')
        ax1.grid(True, alpha=0.3)

        # 2. Drawdown
        ax2 = axes[0, 1]
        ax2.fill_between(result.equity_curve['timestamp'],
                         result.equity_curve['drawdown'] * 100,
                         color='red', alpha=0.5)
        ax2.set_title(f'Drawdown (Max: {result.max_drawdown_pct}%)')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # 3. PnL distribution
        ax3 = axes[1, 0]
        df = pd.DataFrame(result.trades)
        ax3.hist(df['pnl'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_title(f'PnL Distribution (Win Rate: {result.win_rate}%)')
        ax3.set_xlabel('PnL (USD)')
        ax3.set_ylabel('Frequência')

        # 4. Metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
        MÉTRICAS PRINCIPAIS
        ══════════════════════════════

        Retorno Total:     ${result.total_return_usd:,.2f} ({result.total_return_pct}%)
        Sharpe Ratio:      {result.sharpe_ratio}
        Max Drawdown:      {result.max_drawdown_pct}%

        Total Trades:      {result.total_trades}
        Win Rate:          {result.win_rate}%
        Lock Rate:         {result.lock_rate}%

        Avg PairCost:      {result.avg_pair_cost}
        Avg Locked Profit: ${result.avg_locked_profit:.2f}
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                 verticalalignment='center', transform=ax4.transAxes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
```

---

## 10. KPIs e Critérios de Sucesso

### 10.1 Métricas Principais

| KPI | Fórmula | Target | Aceitável | Crítico |
|-----|---------|--------|-----------|---------|
| **Lock Rate** | `windows_locked / total_windows` | > 70% | > 50% | < 30% |
| **Avg PairCost** | `mean(pair_cost)` | < 0.97 | < 0.98 | > 0.99 |
| **Avg Locked Profit** | `mean(locked_profit)` | > $2/janela | > $1/janela | < $0.50 |
| **Win Rate** | `trades_profit > 0 / total_trades` | > 80% | > 60% | < 40% |
| **Fill Rate** | `orders_filled / orders_sent` | > 90% | > 70% | < 50% |
| **Avg Slippage** | `(executed - expected) / expected` | < 0.3% | < 0.5% | > 1% |
| **Trades/Janela** | `mean(trades_per_window)` | 5-15 | 3-20 | < 2 ou > 30 |
| **Latência E2E** | `time(signal → fill)` | < 500ms | < 1s | > 2s |
| **Error Rate** | `api_errors / total_requests` | < 1% | < 5% | > 10% |

### 10.2 Critérios de Go-Live

```
ANTES DE IR PARA LIVE, VALIDAR:

□ BACKTEST
  ├── 200+ janelas simuladas
  ├── Lock Rate > 50%
  ├── Sharpe > 1.0
  └── Max Drawdown < 15%

□ PAPER TRADING
  ├── 2 semanas de execução
  ├── Métricas consistentes com backtest
  ├── Fill rate > 70%
  └── Zero erros críticos

□ INFRAESTRUTURA
  ├── Circuit breaker testado
  ├── Reconciliação funcionando
  ├── Logs estruturados
  └── Alertas configurados

□ CAPITAL
  ├── Começar com 10% do capital
  ├── Sizing mínimo ($5-10)
  └── Daily loss limit ativo
```

---

## 11. Checklist de Segurança

### 11.1 Nunca Fazer

```
❌ Logar private key ou seed phrase
❌ Commitar .env no git
❌ Operar sem limites de risco
❌ Ignorar erros de API
❌ Desabilitar circuit breaker
❌ Confiar cegamente no estado local (sempre reconciliar)
```

### 11.2 Sempre Fazer

```
✅ Usar .env para secrets
✅ Validar credenciais no startup
✅ Limitar exposição por janela
✅ Limitar loss diário
✅ Pausar em erros consecutivos
✅ Logar todas as decisões
✅ Mode paper como default
```

---

## 12. Sequência de Implementação

### Milestone 1: Foundation (Dias 1-2)

```
□ Setup do projeto
  ├── Estrutura de diretórios
  ├── requirements.txt
  ├── Settings (pydantic)
  └── .env.example

□ Conexão CLOB
  ├── Client wrapper
  ├── Autenticação
  └── diagnose_config.py

□ Market Discovery
  ├── Busca de mercado ativo
  └── Leitura de orderbook
```

### Milestone 2: Strategy Core (Dias 3-5)

```
□ Position State
  ├── Modelo de dados
  ├── Cálculos (PairCost, LockedProfit)
  └── Testes unitários

□ Signal Engine
  ├── PairParitySignal
  ├── ImbalanceSignal
  └── Testes unitários

□ Sizing
  ├── Lógica de sizing
  └── Risk checks
```

### Milestone 3: Execution (Dias 6-8)

```
□ Order Executor
  ├── Place/cancel orders
  ├── Wait for fill
  └── Error handling

□ Paper Trading
  ├── Mock client
  ├── Simulação de fills
  └── run_paper.py
```

### Milestone 4: Backtest (Dias 9-12)

```
□ Data Collection
  ├── Histórico de orderbook
  └── Storage SQLite

□ Backtest Engine
  ├── Simulação de janelas
  ├── Cálculo de métricas
  └── run_backtest.py

□ Visualização
  ├── Gráficos de janela
  └── Resumo de backtest
```

### Milestone 5: Production (Dias 13-15)

```
□ Live Deployment
  ├── run_live.py
  ├── Reconciliation
  └── Monitoring

□ Observabilidade
  ├── Logs estruturados
  ├── Métricas
  └── Alertas
```

---

## 13. Comandos de Execução

```bash
# Diagnóstico
python -m scripts.diagnose_config

# Paper trading
python -m scripts.run_paper --capital 1000 --duration 1h

# Backtest
python -m scripts.run_backtest \
  --start 2024-01-01 \
  --end 2024-03-01 \
  --capital 5000 \
  --output data/results/backtest_q1.json

# Live (com cuidado!)
DRY_RUN=false python -m scripts.run_live \
  --capital 500 \
  --max-loss 50

# Exportar gráficos
python -m scripts.export_trades \
  --market-id abc123 \
  --output data/exports/
```

---

**Documento**: GABAGOOL_BOT3_PLAN.md
**Versão**: 2.0
**Autor**: Claude + Leandro
**Data**: Janeiro 2025
