# GABAGOOL V2 - Polymarket Volatility Arbitrage System

## Executive Summary

Sistema automatizado de arbitragem de volatilidade em mercados BTC 15-min Up/Down com:
- **Contabilidade por Legs**: Rastreia UP/DOWN comprados em momentos diferentes
- **Stop Inteligente**: Para quando `pair_cost < 1.00` com buffer dinâmico
- **Auto-tuning**: Ajuste automático de thresholds por regime/horário/liquidez
- **Observabilidade**: Métricas, alertas e health checks em tempo real

---

## 0. Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GABAGOOL V2 SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Discovery  │───▶│  Market Data │───▶│   Signals    │              │
│  │    Agent     │    │    Agent     │    │    Agent     │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      STATE MANAGER                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Leg UP  │  │ Leg DOWN│  │  Pair   │  │Imbalance│            │   │
│  │  │ C_up    │  │ C_down  │  │  Cost   │  │ Tracker │            │   │
│  │  │ Q_up    │  │ Q_down  │  │ < 1.00? │  │         │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Strategy   │───▶│  Execution   │───▶│     Ops      │              │
│  │   Gabagool   │    │    Router    │    │   Metrics    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     LEARNING LAYER                               │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │   │
│  │  │ Feature │  │Evaluator│  │  Tuner  │  │ Policy  │            │   │
│  │  │ Builder │  │         │  │(Bandit) │  │ Deploy  │            │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Estrutura do Repositório

```
src/gabagool_v2/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuração centralizada + hot reload
│   ├── clock.py               # Clock abstrato (real/sim/replay)
│   ├── state.py               # State machine por mercado/janela
│   ├── accounting.py          # Contabilidade por legs (UP/DOWN)
│   └── types.py               # Tipos e dataclasses
├── polymarket/
│   ├── __init__.py
│   ├── gamma_client.py        # Discovery de mercados BTC 15-min
│   ├── clob_ws.py             # WebSocket CLOB (orderbook stream)
│   ├── order_router.py        # Execução de ordens (limit/market)
│   └── fees.py                # Cálculo de fees (maker/taker)
├── marketdata/
│   ├── __init__.py
│   ├── orderbook.py           # Orderbook manager (best bid/ask)
│   ├── trades.py              # Trade tape manager
│   └── spot_feed.py           # BTC spot price feed (Binance/etc)
├── signals/
│   ├── __init__.py
│   ├── volatility.py          # Vol regime detector
│   ├── mispricing.py          # Dislocation score
│   └── edge.py                # Edge calculator
├── strategy/
│   ├── __init__.py
│   ├── gabagool.py            # Core strategy logic
│   ├── sizing.py              # Position sizing por edge
│   └── timing.py              # Entry/exit timing
├── risk/
│   ├── __init__.py
│   ├── inventory.py           # Inventory/imbalance constraints
│   ├── limits.py              # Hard limits (max exposure, etc)
│   └── slippage.py            # Slippage estimation
├── sim/
│   ├── __init__.py
│   ├── synthetic.py           # Gerador de dados sintéticos
│   ├── replay.py              # Replay de dados históricos
│   └── fills.py               # Simulador de fills
├── learning/
│   ├── __init__.py
│   ├── features.py            # Feature builder por janela
│   ├── evaluator.py           # Avaliador de performance
│   ├── tuner.py               # Auto-tuning orchestrator
│   ├── bandit.py              # Multi-Armed Bandit (online)
│   └── bayes.py               # Bayesian Optimization (offline)
├── ops/
│   ├── __init__.py
│   ├── logging.py             # Structured logging
│   ├── metrics.py             # Prometheus/OpenTelemetry metrics
│   ├── alerts.py              # Telegram/Discord alerts
│   └── health.py              # Health checks
└── scripts/
    ├── run_live.py            # Bot de produção
    ├── run_backtest.py        # Backtest com dados reais
    ├── run_replay.py          # Replay mode
    ├── tune_thresholds.py     # Tuning automático
    └── export_trades.py       # Exportar trades para análise

configs/
├── base.yaml                  # Config base
├── markets.yaml               # Filtros de mercado
├── thresholds.yaml            # Thresholds por regime
└── risk.yaml                  # Limites de risco

data/
├── raw/                       # Dados brutos coletados
├── processed/                 # Dados processados
├── features/                  # Features calculadas
└── tuning/                    # Resultados de tuning

tests/
├── test_accounting.py         # Testes de contabilidade
├── test_pair_cost.py          # Testes de pair cost
├── test_inventory.py          # Testes de inventory
├── test_tuning.py             # Testes de tuning
└── test_strategy.py           # Testes de estratégia
```

---

## 2. Core: Contabilidade por Legs

### 2.1 Tipos Base (`core/types.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
import uuid


class Side(Enum):
    UP = "up"
    DOWN = "down"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"


class FillType(Enum):
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class Fill:
    """Representa um fill individual."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    side: Side = Side.UP
    price: float = 0.0
    qty: float = 0.0
    cost: float = 0.0  # price * qty
    fee: float = 0.0
    fee_type: FillType = FillType.TAKER
    timestamp: datetime = field(default_factory=datetime.utcnow)
    market_id: str = ""
    order_id: str = ""


@dataclass
class Leg:
    """
    Uma 'leg' é um conjunto de fills do mesmo lado (UP ou DOWN)
    que podem ter sido comprados em momentos/preços diferentes.
    """
    side: Side
    fills: List[Fill] = field(default_factory=list)

    @property
    def total_qty(self) -> float:
        """Quantidade total de shares."""
        return sum(f.qty for f in self.fills)

    @property
    def total_cost(self) -> float:
        """Custo total (inclui fees)."""
        return sum(f.cost + f.fee for f in self.fills)

    @property
    def avg_price(self) -> float:
        """Preço médio ponderado."""
        if self.total_qty == 0:
            return 0.0
        return sum(f.price * f.qty for f in self.fills) / self.total_qty

    @property
    def total_fees(self) -> float:
        """Total de fees pagos."""
        return sum(f.fee for f in self.fills)

    def add_fill(self, fill: Fill) -> None:
        """Adiciona um fill à leg."""
        if fill.side != self.side:
            raise ValueError(f"Fill side {fill.side} != leg side {self.side}")
        self.fills.append(fill)


@dataclass
class PairPosition:
    """
    Posição completa de um par UP/DOWN em um mercado.
    Rastreia as duas legs independentemente.
    """
    market_id: str
    up_leg: Leg = field(default_factory=lambda: Leg(Side.UP))
    down_leg: Leg = field(default_factory=lambda: Leg(Side.DOWN))
    created_at: datetime = field(default_factory=datetime.utcnow)
    close_ts: Optional[datetime] = None

    @property
    def pair_cost(self) -> float:
        """
        Custo do par normalizado.
        Se < 1.00, há lucro garantido no settlement.
        """
        if self.up_leg.total_qty == 0 or self.down_leg.total_qty == 0:
            return float('inf')  # Não tem par completo

        # Normaliza pelo min de shares para calcular pair_cost
        min_shares = min(self.up_leg.total_qty, self.down_leg.total_qty)

        # Custo proporcional
        up_cost_per_share = self.up_leg.total_cost / self.up_leg.total_qty
        down_cost_per_share = self.down_leg.total_cost / self.down_leg.total_qty

        return up_cost_per_share + down_cost_per_share

    @property
    def guaranteed_profit(self) -> float:
        """Lucro garantido em USD se pair_cost < 1.00."""
        if self.pair_cost >= 1.0:
            return 0.0
        min_shares = min(self.up_leg.total_qty, self.down_leg.total_qty)
        return (1.0 - self.pair_cost) * min_shares

    @property
    def imbalance_ratio(self) -> float:
        """
        Ratio de desbalanceamento.
        1.0 = perfeitamente balanceado
        > 1.0 = mais UP que DOWN
        < 1.0 = mais DOWN que UP
        """
        if self.down_leg.total_qty == 0:
            return float('inf')
        return self.up_leg.total_qty / self.down_leg.total_qty

    @property
    def total_exposure(self) -> float:
        """Exposição total em USD."""
        return self.up_leg.total_cost + self.down_leg.total_cost

    @property
    def is_hedged(self) -> bool:
        """Posição está hedgeada (tem ambas as legs)?"""
        return self.up_leg.total_qty > 0 and self.down_leg.total_qty > 0


@dataclass
class MarketWindow:
    """Estado de uma janela de 15 minutos."""
    market_id: str
    condition_id: str
    question: str
    start_ts: datetime
    close_ts: datetime
    up_token_id: str
    down_token_id: str
    position: PairPosition = None
    outcome: Optional[str] = None  # "up" or "down" após settlement

    def __post_init__(self):
        if self.position is None:
            self.position = PairPosition(market_id=self.market_id, close_ts=self.close_ts)

    @property
    def time_remaining(self) -> float:
        """Segundos restantes até o close."""
        delta = self.close_ts - datetime.utcnow()
        return max(0, delta.total_seconds())

    @property
    def is_active(self) -> bool:
        """Mercado ainda está ativo?"""
        return datetime.utcnow() < self.close_ts
```

### 2.2 State Manager (`core/state.py`)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum
import json

from .types import MarketWindow, PairPosition, Fill, Side, Leg


class TradingState(Enum):
    """Estados possíveis do bot."""
    IDLE = "idle"                    # Aguardando mercado
    DISCOVERING = "discovering"      # Buscando mercados ativos
    ACCUMULATING = "accumulating"    # Acumulando posição
    HEDGED = "hedged"                # Posição hedgeada
    PROFIT_LOCKED = "profit_locked"  # Lucro garantido (pair_cost < 1.00)
    STOPPING = "stopping"            # Parando de operar
    SETTLED = "settled"              # Mercado resolvido


@dataclass
class StateSnapshot:
    """Snapshot do estado para logging/replay."""
    timestamp: datetime
    state: TradingState
    market_id: str
    pair_cost: float
    up_qty: float
    up_avg: float
    down_qty: float
    down_avg: float
    imbalance: float
    time_remaining: float
    guaranteed_profit: float
    total_exposure: float

    def to_dict(self) -> dict:
        return {
            "ts": self.timestamp.isoformat(),
            "state": self.state.value,
            "market_id": self.market_id,
            "pair_cost": round(self.pair_cost, 6),
            "up_qty": round(self.up_qty, 4),
            "up_avg": round(self.up_avg, 4),
            "down_qty": round(self.down_qty, 4),
            "down_avg": round(self.down_avg, 4),
            "imbalance": round(self.imbalance, 4),
            "time_remaining": round(self.time_remaining, 1),
            "guaranteed_profit": round(self.guaranteed_profit, 4),
            "total_exposure": round(self.total_exposure, 2),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class StateManager:
    """
    Gerencia o estado global do sistema.
    Rastreia múltiplos mercados e suas posições.
    """

    def __init__(self):
        self.current_state: TradingState = TradingState.IDLE
        self.active_markets: Dict[str, MarketWindow] = {}
        self.historical_positions: List[PairPosition] = []
        self.snapshots: List[StateSnapshot] = []
        self.stop_trading_flag: bool = False

    def register_market(self, market: MarketWindow) -> None:
        """Registra um novo mercado para tracking."""
        self.active_markets[market.market_id] = market
        self.current_state = TradingState.DISCOVERING

    def get_market(self, market_id: str) -> Optional[MarketWindow]:
        """Retorna mercado pelo ID."""
        return self.active_markets.get(market_id)

    def record_fill(self, market_id: str, fill: Fill) -> None:
        """Registra um fill e atualiza o estado."""
        market = self.active_markets.get(market_id)
        if not market:
            raise ValueError(f"Market {market_id} not found")

        if fill.side == Side.UP:
            market.position.up_leg.add_fill(fill)
        else:
            market.position.down_leg.add_fill(fill)

        # Atualiza estado baseado na posição
        self._update_state(market)

        # Grava snapshot
        self._record_snapshot(market)

    def _update_state(self, market: MarketWindow) -> None:
        """Atualiza estado baseado na posição atual."""
        pos = market.position

        if not pos.is_hedged:
            self.current_state = TradingState.ACCUMULATING
        elif pos.pair_cost < 1.0:
            self.current_state = TradingState.PROFIT_LOCKED
        else:
            self.current_state = TradingState.HEDGED

        # Check stop conditions
        if self.should_stop(market):
            self.current_state = TradingState.STOPPING
            self.stop_trading_flag = True

    def should_stop(self, market: MarketWindow,
                    min_profit_buffer: float = 0.005,
                    time_buffer_seconds: float = 30) -> bool:
        """
        Determina se deve parar de operar.

        Condições de stop:
        1. pair_cost < (1.00 - min_profit_buffer) = lucro garantido
        2. time_remaining < time_buffer_seconds = muito perto do close
        3. stop_trading_flag setado externamente
        """
        pos = market.position

        # Lucro garantido
        if pos.pair_cost < (1.0 - min_profit_buffer):
            return True

        # Muito perto do close
        if market.time_remaining < time_buffer_seconds:
            return True

        return self.stop_trading_flag

    def _record_snapshot(self, market: MarketWindow) -> None:
        """Grava snapshot do estado atual."""
        pos = market.position

        snapshot = StateSnapshot(
            timestamp=datetime.utcnow(),
            state=self.current_state,
            market_id=market.market_id,
            pair_cost=pos.pair_cost,
            up_qty=pos.up_leg.total_qty,
            up_avg=pos.up_leg.avg_price,
            down_qty=pos.down_leg.total_qty,
            down_avg=pos.down_leg.avg_price,
            imbalance=pos.imbalance_ratio,
            time_remaining=market.time_remaining,
            guaranteed_profit=pos.guaranteed_profit,
            total_exposure=pos.total_exposure,
        )

        self.snapshots.append(snapshot)

    def settle_market(self, market_id: str, outcome: str) -> float:
        """
        Resolve um mercado e calcula P&L final.

        Args:
            market_id: ID do mercado
            outcome: "up" ou "down"

        Returns:
            P&L realizado em USD
        """
        market = self.active_markets.get(market_id)
        if not market:
            raise ValueError(f"Market {market_id} not found")

        market.outcome = outcome
        pos = market.position

        # Calcula P&L
        if outcome == "up":
            # UP vale $1.00, DOWN vale $0.00
            revenue = pos.up_leg.total_qty * 1.0
        else:
            # DOWN vale $1.00, UP vale $0.00
            revenue = pos.down_leg.total_qty * 1.0

        total_cost = pos.total_exposure
        pnl = revenue - total_cost

        # Move para histórico
        self.historical_positions.append(pos)
        del self.active_markets[market_id]

        self.current_state = TradingState.SETTLED

        return pnl

    def serialize(self) -> dict:
        """Serializa estado completo para persistência."""
        return {
            "current_state": self.current_state.value,
            "stop_flag": self.stop_trading_flag,
            "active_markets": {
                mid: {
                    "market_id": m.market_id,
                    "close_ts": m.close_ts.isoformat(),
                    "position": {
                        "pair_cost": m.position.pair_cost,
                        "up_leg": {
                            "qty": m.position.up_leg.total_qty,
                            "cost": m.position.up_leg.total_cost,
                            "avg": m.position.up_leg.avg_price,
                            "fills": len(m.position.up_leg.fills),
                        },
                        "down_leg": {
                            "qty": m.position.down_leg.total_qty,
                            "cost": m.position.down_leg.total_cost,
                            "avg": m.position.down_leg.avg_price,
                            "fills": len(m.position.down_leg.fills),
                        },
                    }
                }
                for mid, m in self.active_markets.items()
            },
            "snapshots_count": len(self.snapshots),
            "historical_count": len(self.historical_positions),
        }
```

### 2.3 Accounting Engine (`core/accounting.py`)

```python
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime

from .types import PairPosition, Fill, Side, FillType


@dataclass
class FeeConfig:
    """Configuração de fees."""
    maker_fee: float = 0.0      # 0% for limit orders
    taker_fee: float = 0.02     # 2% for market orders
    min_fee: float = 0.0


class AccountingEngine:
    """
    Motor de contabilidade para rastrear custos e P&L.
    Foco em pair_cost e edge calculation.
    """

    def __init__(self, fee_config: FeeConfig = None):
        self.fee_config = fee_config or FeeConfig()

    def calculate_fill_cost(self, price: float, qty: float,
                           fill_type: FillType) -> Tuple[float, float]:
        """
        Calcula custo total de um fill incluindo fees.

        Returns:
            (base_cost, fee)
        """
        base_cost = price * qty

        if fill_type == FillType.MAKER:
            fee = base_cost * self.fee_config.maker_fee
        else:
            fee = base_cost * self.fee_config.taker_fee

        fee = max(fee, self.fee_config.min_fee)

        return base_cost, fee

    def create_fill(self, side: Side, price: float, qty: float,
                   fill_type: FillType, market_id: str,
                   order_id: str = "") -> Fill:
        """Cria um Fill com custos calculados."""
        base_cost, fee = self.calculate_fill_cost(price, qty, fill_type)

        return Fill(
            side=side,
            price=price,
            qty=qty,
            cost=base_cost,
            fee=fee,
            fee_type=fill_type,
            market_id=market_id,
            order_id=order_id,
            timestamp=datetime.utcnow(),
        )

    def compute_pair_cost(self, position: PairPosition,
                         fee_buffer: float = 0.0) -> float:
        """
        Calcula pair_cost com buffer opcional.

        Args:
            position: Posição atual
            fee_buffer: Buffer adicional para fees esperadas futuras

        Returns:
            pair_cost ajustado
        """
        return position.pair_cost + fee_buffer

    def compute_edge(self, position: PairPosition) -> float:
        """
        Calcula edge atual.
        Edge = 1 - pair_cost (quando positivo, há lucro garantido)
        """
        if position.pair_cost == float('inf'):
            return 0.0
        return 1.0 - position.pair_cost

    def compute_min_profit_at_price(self, position: PairPosition,
                                    target_up_price: float,
                                    target_down_price: float,
                                    target_qty: float,
                                    fill_type: FillType) -> float:
        """
        Calcula lucro mínimo se comprar na price alvo.
        Útil para decidir se vale a pena executar.
        """
        # Simula o fill
        if target_up_price > 0:
            _, up_fee = self.calculate_fill_cost(target_up_price, target_qty, fill_type)
            simulated_up_cost = target_up_price * target_qty + up_fee
        else:
            simulated_up_cost = 0

        if target_down_price > 0:
            _, down_fee = self.calculate_fill_cost(target_down_price, target_qty, fill_type)
            simulated_down_cost = target_down_price * target_qty + down_fee
        else:
            simulated_down_cost = 0

        # Custo total simulado
        total_up = position.up_leg.total_cost + simulated_up_cost
        total_down = position.down_leg.total_cost + simulated_down_cost
        total_up_qty = position.up_leg.total_qty + (target_qty if target_up_price > 0 else 0)
        total_down_qty = position.down_leg.total_qty + (target_qty if target_down_price > 0 else 0)

        if total_up_qty == 0 or total_down_qty == 0:
            return 0.0

        # Pair cost simulado
        up_avg = total_up / total_up_qty
        down_avg = total_down / total_down_qty
        simulated_pair_cost = up_avg + down_avg

        # Lucro mínimo
        min_shares = min(total_up_qty, total_down_qty)
        if simulated_pair_cost < 1.0:
            return (1.0 - simulated_pair_cost) * min_shares
        return 0.0

    def should_execute(self, position: PairPosition,
                      side: Side,
                      price: float,
                      qty: float,
                      fill_type: FillType,
                      max_pair_cost: float = 0.99,
                      max_imbalance: float = 1.3) -> Tuple[bool, str]:
        """
        Decide se deve executar um trade.

        Returns:
            (should_execute, reason)
        """
        # Simula o fill
        if side == Side.UP:
            target_up = price
            target_down = 0
        else:
            target_up = 0
            target_down = price

        # Verifica pair_cost futuro
        min_profit = self.compute_min_profit_at_price(
            position, target_up, target_down, qty, fill_type
        )

        # Calcula imbalance futuro
        future_up = position.up_leg.total_qty + (qty if side == Side.UP else 0)
        future_down = position.down_leg.total_qty + (qty if side == Side.DOWN else 0)

        if future_down > 0:
            future_imbalance = future_up / future_down
        else:
            future_imbalance = float('inf')

        # Checks
        if min_profit <= 0:
            # Verifica se pair_cost estaria ok
            # (Lógica simplificada - na prática precisa calcular pair_cost futuro)
            pass

        if future_imbalance > max_imbalance or future_imbalance < 1/max_imbalance:
            return False, f"imbalance_exceeded: {future_imbalance:.2f}"

        return True, "ok"
```

---

## 3. Configuração (`configs/`)

### 3.1 Base Config (`configs/base.yaml`)

```yaml
# Gabagool V2 - Base Configuration

system:
  name: "gabagool_v2"
  version: "2.0.0"
  mode: "paper"  # paper | live
  log_level: "INFO"

# Clock settings
clock:
  type: "real"  # real | simulated | replay
  tick_interval_ms: 100

# Data paths
paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  data_features: "data/features"
  logs: "logs"
  configs: "configs"

# API endpoints
api:
  gamma_url: "https://gamma-api.polymarket.com"
  clob_url: "https://clob.polymarket.com"
  clob_ws: "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Performance
performance:
  max_concurrent_markets: 3
  ws_reconnect_delay_ms: 1000
  ws_max_reconnect_attempts: 10
  order_timeout_ms: 5000
```

### 3.2 Thresholds Config (`configs/thresholds.yaml`)

```yaml
# Thresholds - Tunados por regime

# Global defaults
defaults:
  max_pair_cost: 0.98
  min_edge_per_pair: 0.015
  stop_time_buffer_s: 30
  trade_chunk_usd: 10
  max_imbalance: 1.25
  vol_spike_threshold: 2.0
  min_spread_to_enter: 0.02

# Thresholds por regime de volatilidade
volatility_regimes:
  low:  # Vol < 0.1% por minuto
    max_pair_cost: 0.985
    min_edge_per_pair: 0.012
    trade_chunk_usd: 15
    max_imbalance: 1.3
    vol_spike_threshold: 1.5

  normal:  # Vol 0.1-0.3% por minuto
    max_pair_cost: 0.98
    min_edge_per_pair: 0.015
    trade_chunk_usd: 10
    max_imbalance: 1.25
    vol_spike_threshold: 2.0

  high:  # Vol > 0.3% por minuto
    max_pair_cost: 0.975
    min_edge_per_pair: 0.02
    trade_chunk_usd: 8
    max_imbalance: 1.2
    vol_spike_threshold: 2.5

# Thresholds por hora do dia (UTC)
time_buckets:
  asia_morning:  # 00:00-06:00 UTC
    hours: [0, 1, 2, 3, 4, 5]
    adjustments:
      max_pair_cost: -0.005  # Mais conservador
      trade_chunk_usd: -2

  europe_open:  # 06:00-12:00 UTC
    hours: [6, 7, 8, 9, 10, 11]
    adjustments:
      max_pair_cost: 0.0
      trade_chunk_usd: 0

  us_morning:  # 12:00-18:00 UTC
    hours: [12, 13, 14, 15, 16, 17]
    adjustments:
      max_pair_cost: 0.005  # Mais agressivo
      trade_chunk_usd: 3

  us_afternoon:  # 18:00-00:00 UTC
    hours: [18, 19, 20, 21, 22, 23]
    adjustments:
      max_pair_cost: 0.0
      trade_chunk_usd: 0

# Thresholds por liquidez (depth em USD no top 3 levels)
liquidity_buckets:
  very_low:  # < $100
    depth_usd_max: 100
    adjustments:
      trade_chunk_usd: -5
      max_imbalance: -0.1

  low:  # $100-500
    depth_usd_min: 100
    depth_usd_max: 500
    adjustments:
      trade_chunk_usd: -2
      max_imbalance: -0.05

  normal:  # $500-2000
    depth_usd_min: 500
    depth_usd_max: 2000
    adjustments:
      trade_chunk_usd: 0
      max_imbalance: 0

  high:  # > $2000
    depth_usd_min: 2000
    adjustments:
      trade_chunk_usd: 5
      max_imbalance: 0.1

# Safety limits (hard caps - nunca ultrapassar)
safety:
  absolute_max_pair_cost: 0.995
  absolute_min_edge: 0.005
  absolute_max_imbalance: 1.5
  absolute_min_time_buffer_s: 15
  max_exposure_per_market_usd: 500
  max_total_exposure_usd: 2000
```

### 3.3 Risk Config (`configs/risk.yaml`)

```yaml
# Risk Management Configuration

# Capital limits
capital:
  initial_balance_usd: 1000
  max_drawdown_pct: 0.10  # 10% max drawdown
  daily_loss_limit_usd: 100

# Position limits
positions:
  max_open_markets: 3
  max_exposure_per_market_usd: 500
  max_total_exposure_usd: 2000
  max_position_pct_of_capital: 0.25

# Inventory constraints
inventory:
  max_imbalance_ratio: 1.5  # Max 1.5x more UP than DOWN (or vice versa)
  rebalance_threshold: 1.3  # Start rebalancing at 1.3x
  force_rebalance_threshold: 1.4  # Force rebalance at 1.4x

# Time constraints
timing:
  min_time_to_close_s: 30  # Stop trading 30s before close
  max_time_in_position_s: 840  # Max 14 min in position
  cool_down_after_loss_s: 60  # Wait 60s after loss

# Stop conditions
stops:
  stop_on_profit_locked: true  # Stop when pair_cost < 0.995
  profit_lock_threshold: 0.995
  stop_loss_per_market_usd: 50
  stop_loss_total_usd: 200

# Emergency
emergency:
  kill_switch_enabled: true
  max_consecutive_losses: 5
  max_api_errors: 10
  pause_on_high_vol: true
  high_vol_threshold: 0.005  # 0.5% per minute

# Fees
fees:
  maker_fee_pct: 0.0
  taker_fee_pct: 0.02
  assume_taker: true  # Assume worst case for calculations
```

---

## 4. Claude Code Prompts por Agent

### Agent A: Orchestrator / Runtime

```
PROMPT PARA CLAUDE CODE - AGENT A: ORCHESTRATOR

Implemente o sistema de orquestração em:
- scripts/run_live.py
- src/gabagool_v2/core/state.py (já parcialmente implementado acima)
- src/gabagool_v2/core/clock.py

REQUISITOS:

1. run_live.py deve:
   - Carregar configs de configs/*.yaml
   - Inicializar todos os componentes (discovery, ws, strategy, etc)
   - Rodar loop assíncrono principal com asyncio
   - Tasks separadas para:
     * Market discovery (refresh a cada 30s)
     * WebSocket orderbook (stream contínuo)
     * Spot feed BTC (stream contínuo)
     * Strategy decision loop (a cada 100ms)
     * State persistence (a cada 5s)
   - Graceful shutdown com signal handlers (SIGINT, SIGTERM)
   - Reconnection automática com exponential backoff
   - Logging estruturado com contexto

2. clock.py deve:
   - Classe Clock abstrata com get_time(), sleep(), schedule()
   - RealClock para produção
   - SimulatedClock para backtest (permite acelerar tempo)
   - ReplayClock para replay de dados históricos

3. Patterns a usar:
   - asyncio.gather() para tasks concorrentes
   - asyncio.Queue() para comunicação entre tasks
   - Context managers para recursos
   - Structured concurrency

EXEMPLO DE ESTRUTURA run_live.py:

```python
import asyncio
import signal
from src.gabagool_v2.core.config import load_config
from src.gabagool_v2.core.state import StateManager

async def main():
    config = load_config()
    state = StateManager()

    # Setup components
    discovery = MarketDiscovery(config)
    orderbook = OrderBookManager(config)
    spot_feed = SpotFeed(config)
    strategy = GabagoolStrategy(config, state)

    # Create tasks
    tasks = [
        asyncio.create_task(discovery.run(), name="discovery"),
        asyncio.create_task(orderbook.run(), name="orderbook"),
        asyncio.create_task(spot_feed.run(), name="spot"),
        asyncio.create_task(strategy.run(), name="strategy"),
    ]

    # Wait with graceful shutdown
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

Implemente com tratamento robusto de erros e logging detalhado.
```

---

### Agent B: Market Discovery

```
PROMPT PARA CLAUDE CODE - AGENT B: MARKET DISCOVERY

Implemente o cliente de discovery em:
- src/gabagool_v2/polymarket/gamma_client.py

REQUISITOS:

1. Classe GammaClient com:
   - list_markets() - Lista todos os mercados
   - filter_btc_15min() - Filtra apenas BTC 15-min Up/Down
   - get_market_details(market_id) - Detalhes de um mercado
   - get_active_window() - Retorna janela ativa atual
   - Cache com TTL (refresh a cada 30s)

2. Filtros necessários:
   - Asset: BTC (condition_id contém "BTC" ou similar)
   - Duration: 15 minutos
   - Type: Up/Down binary
   - Status: Active (não resolvido)
   - Time: start_ts < now < close_ts

3. Dados a extrair:
   - market_id
   - condition_id
   - question
   - start_ts, close_ts
   - up_token_id, down_token_id
   - fee_enabled
   - min_tick_size
   - outcome (null se não resolvido)

4. API endpoints:
   - GET https://gamma-api.polymarket.com/markets
   - GET https://gamma-api.polymarket.com/markets/{id}

5. Tratamento de erros:
   - Retry com backoff para erros de rede
   - Log de erros sem crashar
   - Fallback para cache se API falhar

EXEMPLO DE USO:

```python
client = GammaClient()

# Refresh de mercados
await client.refresh_markets()

# Buscar janela ativa
window = await client.get_active_window()
if window:
    print(f"Active: {window.market_id}, closes in {window.time_remaining}s")
```

Implemente com httpx para async HTTP e caching robusto.
```

---

### Agent C: CLOB WebSocket + OrderBook

```
PROMPT PARA CLAUDE CODE - AGENT C: CLOB WEBSOCKET + ORDERBOOK

Implemente em:
- src/gabagool_v2/polymarket/clob_ws.py
- src/gabagool_v2/marketdata/orderbook.py

REQUISITOS:

1. clob_ws.py - WebSocket Manager:
   - Conectar a wss://ws-subscriptions-clob.polymarket.com/ws/market
   - Subscribe em orderbook updates para token_ids específicos
   - Processar mensagens: snapshots + incremental updates
   - Reconnect automático com exponential backoff
   - Heartbeat/ping para manter conexão viva
   - Queue de mensagens para consumidores

2. orderbook.py - OrderBook Manager:
   - Classe OrderBook com:
     * bids: List[PriceLevel] ordenado decrescente
     * asks: List[PriceLevel] ordenado crescente
     * best_bid, best_ask properties
     * spread property
     * depth_at_price(price, side)
     * update_from_snapshot(data)
     * apply_delta(delta)

   - Classe OrderBookManager com:
     * Mantém orderbooks para UP e DOWN tokens
     * Métodos: get_up_book(), get_down_book()
     * Calcula mid_price, spread combinado
     * Detecta crossed books (anomalia)

3. Formato de mensagens WS (exemplo):
   ```json
   {
     "type": "book",
     "market": "token_id",
     "bids": [[price, size], ...],
     "asks": [[price, size], ...]
   }
   ```

4. Performance:
   - Usar sortedcontainers para orderbook eficiente
   - Processar updates em < 1ms
   - Buffer de no máximo 100ms de latência

EXEMPLO DE USO:

```python
ws = CLOBWebSocket(config)
book_manager = OrderBookManager()

async for msg in ws.stream():
    book_manager.update(msg)
    up_book = book_manager.get_up_book()
    print(f"UP best bid: {up_book.best_bid}")
```

Implemente com websockets library e proper error handling.
```

---

### Agent D: Execution / Order Router

```
PROMPT PARA CLAUDE CODE - AGENT D: ORDER ROUTER

Implemente em:
- src/gabagool_v2/polymarket/order_router.py

REQUISITOS:

1. Classe OrderRouter com:
   - place_limit_order(token_id, side, price, size) -> OrderResult
   - place_market_order(token_id, side, size) -> OrderResult (se suportado)
   - cancel_order(order_id) -> bool
   - replace_order(order_id, new_price, new_size) -> OrderResult
   - get_order_status(order_id) -> OrderStatus
   - get_open_orders() -> List[Order]

2. Idempotência:
   - Gerar client_order_id único para cada ordem
   - Deduplicar ordens repetidas
   - Rastrear estado de cada ordem

3. Rate limiting:
   - Respeitar limits da API (ex: 10 req/s)
   - Queue de ordens com throttling
   - Retry com backoff para rate limit errors

4. Tipos de ordem:
   - GTC (Good Till Cancel) - default
   - FOK (Fill or Kill) - para market orders
   - IOC (Immediate or Cancel)

5. Estados de ordem:
   - PENDING - enviada, aguardando confirm
   - OPEN - no livro
   - PARTIAL - parcialmente executada
   - FILLED - totalmente executada
   - CANCELLED - cancelada
   - REJECTED - rejeitada

6. API endpoints (CLOB):
   - POST /order - criar ordem
   - DELETE /order/{id} - cancelar
   - GET /orders - listar ordens

7. Assinatura:
   - Ordens precisam ser assinadas com wallet
   - Usar py_clob_client ou implementar signing

EXEMPLO DE USO:

```python
router = OrderRouter(config, signer)

# Place limit order
result = await router.place_limit_order(
    token_id="0x123...",
    side="buy",
    price=0.45,
    size=20.0
)

if result.status == OrderStatus.OPEN:
    print(f"Order {result.order_id} placed at {result.price}")
```

Implemente com proper error handling e logging de todas as ordens.
```

---

### Agent E: Fees + Slippage

```
PROMPT PARA CLAUDE CODE - AGENT E: FEES + SLIPPAGE

Implemente em:
- src/gabagool_v2/polymarket/fees.py
- src/gabagool_v2/risk/slippage.py

REQUISITOS:

1. fees.py - FeeCalculator:
   - calculate_fee(price, size, order_type) -> float
   - Maker fee: 0% (limit orders que adicionam liquidez)
   - Taker fee: ~2% (market orders ou limit que cruza spread)
   - Detectar se ordem será maker ou taker baseado no book
   - Buffer de fees para worst case

2. slippage.py - SlippageEstimator:
   - estimate_slippage(book, side, size) -> float
   - Calcula preço médio de execução vs mid price
   - Considera profundidade do book
   - Ajusta por volatilidade atual

3. Classe EffectiveCostCalculator:
   - calculate_effective_cost(price, size, book, vol_regime) -> EffectiveCost
   - Combina: base_cost + fee + expected_slippage
   - Retorna breakdown completo

4. Buffers dinâmicos:
   - Aumentar buffer em vol alta
   - Aumentar buffer em liquidez baixa
   - Aumentar buffer perto do close

EXEMPLO DE USO:

```python
fee_calc = FeeCalculator(config)
slip_calc = SlippageEstimator()
cost_calc = EffectiveCostCalculator(fee_calc, slip_calc)

# Calcular custo efetivo
effective = cost_calc.calculate_effective_cost(
    price=0.48,
    size=10.0,
    book=orderbook,
    vol_regime="high"
)

print(f"Base: ${effective.base_cost:.2f}")
print(f"Fee: ${effective.fee:.2f}")
print(f"Slippage: ${effective.slippage:.2f}")
print(f"Total: ${effective.total:.2f}")
```

Implemente com foco em precisão de cálculos financeiros.
```

---

### Agent F: Signals (Volatilidade + Dislocation)

```
PROMPT PARA CLAUDE CODE - AGENT F: SIGNALS

Implemente em:
- src/gabagool_v2/marketdata/spot_feed.py
- src/gabagool_v2/signals/volatility.py
- src/gabagool_v2/signals/mispricing.py
- src/gabagool_v2/signals/edge.py

REQUISITOS:

1. spot_feed.py - BTC Spot Feed:
   - Conectar a Binance WS (wss://stream.binance.com:9443/ws/btcusdt@trade)
   - Stream de preços em tempo real
   - Calcular: last_price, vwap_1m, price_change_1m
   - Buffer de histórico para cálculos de vol
   - Fallback para outras exchanges se Binance falhar

2. volatility.py - VolatilityCalculator:
   - calculate_realized_vol(prices, window_seconds) -> float
   - calculate_vol_regime(vol) -> "low" | "normal" | "high"
   - detect_vol_spike(current_vol, baseline_vol) -> float (spike_score 0-10)
   - Janelas: 1min, 5min, 15min
   - Output: VolatilitySignal dataclass

3. mispricing.py - MispricingDetector:
   - detect_dislocation(up_price, down_price, spot_price, time_remaining) -> float
   - Compara: preço implícito vs fair value teórico
   - Usa modelo similar ao Black-Scholes simplificado
   - Output: DislocationSignal com score e direção

4. edge.py - EdgeCalculator:
   - calculate_edge(up_price, down_price, vol_regime, liquidity) -> EdgeSignal
   - Edge = oportunidade de lucro ajustada por risco
   - Considera: pair_cost potencial, slippage, fees
   - Output: EdgeSignal com magnitude e confidence

EXEMPLO DE USO:

```python
spot = SpotFeed()
vol_calc = VolatilityCalculator()
edge_calc = EdgeCalculator()

async for price in spot.stream():
    vol_signal = vol_calc.update(price)

    if vol_signal.spike_score > 5:
        edge = edge_calc.calculate_edge(
            up_price=0.52,
            down_price=0.46,
            vol_regime=vol_signal.regime,
            liquidity=book.total_depth
        )

        if edge.magnitude > 0.02:
            print(f"Edge detected: {edge.magnitude:.2%}")
```

Implemente com numpy para cálculos eficientes.
```

---

### Agent G: Strategy (Gabagool Core)

```
PROMPT PARA CLAUDE CODE - AGENT G: GABAGOOL STRATEGY

Implemente em:
- src/gabagool_v2/strategy/gabagool.py
- src/gabagool_v2/strategy/sizing.py
- src/gabagool_v2/strategy/timing.py

REQUISITOS:

1. gabagool.py - GabagoolStrategy:
   - Classe principal que orquestra decisões
   - Input: state, signals, orderbook, config
   - Output: TradeDecision (buy_up, buy_down, wait, stop)

   - Lógica principal:
     ```
     1. Verificar se mercado está ativo
     2. Calcular edge atual
     3. Identificar "perna barata" (UP ou DOWN com melhor preço)
     4. Verificar constraints:
        - pair_cost' <= MAX_PAIR_COST
        - imbalance <= MAX_IMBALANCE
        - time_to_close > STOP_TIME_BUFFER
     5. Calcular size ótimo
     6. Emitir decisão
     ```

   - Estados de decisão:
     * BUY_UP - comprar UP para completar hedge
     * BUY_DOWN - comprar DOWN para completar hedge
     * WAIT - aguardar melhor oportunidade
     * STOP - parar de operar (lucro locked ou time limit)

2. sizing.py - PositionSizer:
   - calculate_size(edge, volatility, liquidity, current_position) -> float
   - Kelly criterion modificado
   - Chunks: dividir ordem grande em partes
   - Respeitar limits de config

3. timing.py - EntryTimer:
   - should_enter_now(signals, time_remaining) -> bool
   - Evitar entrar muito cedo (spread pode melhorar)
   - Evitar entrar muito tarde (risco de não completar hedge)
   - "Sweet spot" baseado em vol regime

4. Logging estruturado:
   - Cada decisão deve ser logada com:
     * timestamp
     * market_id
     * decision
     * reason
     * metrics snapshot
   - Formato JSON para análise posterior

EXEMPLO DE DECISÃO:

```python
strategy = GabagoolStrategy(config, state)

decision = await strategy.evaluate(
    market=active_window,
    up_book=up_orderbook,
    down_book=down_orderbook,
    signals=current_signals
)

# decision.action = "BUY_DOWN"
# decision.reason = "pair_cost=0.96, edge=0.04, down_cheaper"
# decision.size = 10.0
# decision.price = 0.44
```

Implemente com foco em clareza de lógica e logging detalhado.
```

---

### Agent H: Backtest / Replay

```
PROMPT PARA CLAUDE CODE - AGENT H: BACKTEST

Implemente em:
- src/gabagool_v2/sim/replay.py
- src/gabagool_v2/sim/synthetic.py
- src/gabagool_v2/sim/fills.py
- scripts/run_backtest.py

REQUISITOS:

1. replay.py - ReplayEngine:
   - Carrega dados históricos de data/raw/
   - Reconstrói orderbooks tick-by-tick
   - Simula execução de estratégia
   - Suporta: timestamp, orderbook, spot_price, trades

2. synthetic.py - SyntheticDataGenerator:
   - Gera dados sintéticos para teste rápido
   - Modela: price movement, spread dynamics, volume
   - Parâmetros: num_markets, volatility, liquidity_profile

3. fills.py - FillSimulator:
   - Simula se ordem limit seria executada
   - Modelos:
     * Conservative: só executa se price cruza
     * Realistic: probabilidade baseada em volume/time
     * Optimistic: executa se toca o preço
   - Simula partial fills

4. run_backtest.py - CLI:
   ```bash
   python scripts/run_backtest.py \
     --data data/raw \
     --start 2024-01-01 \
     --end 2024-01-31 \
     --balance 1000 \
     --config configs/thresholds.yaml \
     --output data/results/backtest_jan.json
   ```

5. Relatório de output:
   - Total PnL
   - PnL por mercado
   - Win rate
   - Distribuição de pair_cost
   - Edge médio capturado
   - Drawdown máximo
   - Fill rate
   - Gráficos (matplotlib)

EXEMPLO DE USO:

```python
engine = ReplayEngine(config)
results = await engine.run(
    data_path="data/raw",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

print(f"Total PnL: ${results.total_pnl:.2f}")
print(f"Win rate: {results.win_rate:.1%}")
print(f"Avg pair_cost: {results.avg_pair_cost:.4f}")
```

Implemente com suporte a paralelização para backtest rápido.
```

---

### Agent I: Ops (Metrics / Alerts)

```
PROMPT PARA CLAUDE CODE - AGENT I: OPS

Implemente em:
- src/gabagool_v2/ops/logging.py
- src/gabagool_v2/ops/metrics.py
- src/gabagool_v2/ops/alerts.py
- src/gabagool_v2/ops/health.py

REQUISITOS:

1. logging.py - StructuredLogger:
   - JSON logging com campos estruturados
   - Níveis: DEBUG, INFO, WARN, ERROR
   - Contexto automático: timestamp, market_id, component
   - Rotação de arquivos por dia
   - Suporte a stdout + file

2. metrics.py - MetricsCollector:
   - Exportar métricas para Prometheus (opcional)
   - Métricas principais:
     * gabagool_edge_avg (gauge)
     * gabagool_fills_total (counter)
     * gabagool_pnl_usd (gauge)
     * gabagool_pair_cost_histogram
     * gabagool_latency_ms (histogram)
     * gabagool_ws_reconnects_total (counter)
     * gabagool_api_errors_total (counter)
   - Endpoint HTTP /metrics

3. alerts.py - AlertManager:
   - Enviar alertas via:
     * Telegram bot (opcional)
     * Discord webhook (opcional)
     * Slack webhook (opcional)
   - Tipos de alerta:
     * PROFIT_LOCKED - lucro garantido
     * HIGH_DRAWDOWN - drawdown > threshold
     * API_ERROR - erros de API
     * WS_DISCONNECT - perda de conexão
     * EMERGENCY_STOP - kill switch ativado
   - Throttling: não spam (1 alert/tipo/minuto)

4. health.py - HealthChecker:
   - Endpoint HTTP /health
   - Checks:
     * WS connection alive
     * API responding
     * Last trade < 5min ago
     * Memory/CPU ok
   - Status: healthy | degraded | unhealthy

EXEMPLO DE USO:

```python
logger = StructuredLogger("gabagool")
metrics = MetricsCollector()
alerts = AlertManager(config)

# Log estruturado
logger.info("trade_executed",
    market_id="0x123",
    side="up",
    price=0.48,
    size=10.0
)

# Métrica
metrics.record_fill(side="up", size=10.0)

# Alerta
await alerts.send(
    AlertType.PROFIT_LOCKED,
    message="Pair cost: 0.97, profit: $0.30"
)
```

Implemente com foco em observabilidade e debugging fácil.
```

---

## 5. Learning Layer - Auto-tuning

### 5.1 Parâmetros para Tuning

```yaml
# Parâmetros que serão tunados automaticamente
tunable_parameters:
  - name: max_pair_cost
    type: float
    range: [0.96, 0.995]
    default: 0.98

  - name: min_edge_per_pair
    type: float
    range: [0.005, 0.05]
    default: 0.015

  - name: stop_time_buffer_s
    type: int
    range: [10, 60]
    default: 30

  - name: trade_chunk_usd
    type: float
    range: [5, 50]
    default: 10

  - name: max_imbalance
    type: float
    range: [1.1, 1.5]
    default: 1.25

  - name: vol_spike_threshold
    type: float
    range: [1.0, 5.0]
    default: 2.0
```

### 5.2 Agent L1: Feature Builder

```
PROMPT PARA CLAUDE CODE - AGENT L1: FEATURE BUILDER

Implemente em:
- src/gabagool_v2/learning/features.py

REQUISITOS:

1. Classe FeatureBuilder:
   - Extrai features por janela de mercado
   - Features para usar no tuning/evaluation

2. Features a extrair:

   VOLATILIDADE:
   - vol_1min: vol realizada 1 min
   - vol_5min: vol realizada 5 min
   - vol_regime: "low"/"normal"/"high"
   - vol_spike_max: máximo spike na janela
   - vol_trend: increasing/decreasing/stable

   SPREAD/LIQUIDEZ:
   - spread_avg: spread médio UP+DOWN
   - spread_min: spread mínimo
   - spread_max: spread máximo
   - depth_up_top3: liquidez UP nos 3 melhores níveis
   - depth_down_top3: liquidez DOWN
   - depth_imbalance: ratio UP/DOWN depth

   TIMING:
   - time_bucket: hora do dia (0-23 UTC)
   - day_of_week: 0-6
   - time_in_market_pct: % do tempo que estava no mercado
   - time_to_close_at_entry: tempo restante quando entrou

   EXECUÇÃO:
   - fills_count: número de fills
   - fill_rate: % das ordens que executaram
   - avg_fill_latency_ms: latência média
   - slippage_avg: slippage médio realizado

   RESULTADO:
   - pair_cost_final: pair cost final
   - edge_realized: edge capturado
   - pnl_usd: P&L em USD
   - pnl_bps: P&L em basis points
   - outcome: "win"/"loss"/"breakeven"

3. Output: FeatureVector dataclass com todos os campos

EXEMPLO DE USO:

```python
builder = FeatureBuilder()

features = builder.extract(
    market_window=window,
    trades=executed_trades,
    orderbook_snapshots=ob_history,
    spot_prices=btc_prices
)

# features.vol_regime = "high"
# features.spread_avg = 0.025
# features.pnl_usd = 0.35
```

Implemente com pandas para processamento eficiente.
```

---

### 5.3 Agent L2: Evaluator

```
PROMPT PARA CLAUDE CODE - AGENT L2: EVALUATOR

Implemente em:
- src/gabagool_v2/learning/evaluator.py

REQUISITOS:

1. Classe PerformanceEvaluator:
   - Calcula scores para avaliar performance
   - Usado para comparar diferentes configs

2. Métricas de score:

   RETORNO:
   - total_pnl: P&L total em USD
   - avg_pnl_per_window: média por janela
   - pnl_per_dollar_exposed: P&L / exposição
   - sharpe_like: (avg_pnl - risk_free) / std_pnl

   RISCO:
   - max_drawdown: drawdown máximo
   - var_95: Value at Risk 95%
   - avg_exposure_time: tempo médio exposto
   - loss_rate: % de janelas com loss

   EXECUÇÃO:
   - fill_rate: % de ordens executadas
   - edge_capture_rate: edge_realized / edge_theoretical
   - slippage_cost: custo total de slippage

   EFICIÊNCIA:
   - trades_per_profit: trades necessários por $ de lucro
   - capital_efficiency: P&L / capital usado

3. Score composto:
   ```
   composite_score = (
       w1 * normalized_pnl +
       w2 * (1 - normalized_drawdown) +
       w3 * normalized_fill_rate +
       w4 * normalized_edge_capture
   )
   ```

4. Comparação de configs:
   - compare(config_a_results, config_b_results) -> ComparisonReport
   - Statistical significance test

EXEMPLO DE USO:

```python
evaluator = PerformanceEvaluator()

score = evaluator.evaluate(
    trades=all_trades,
    features=all_features,
    config=current_config
)

print(f"Composite score: {score.composite:.3f}")
print(f"Sharpe-like: {score.sharpe_like:.2f}")
print(f"Max drawdown: {score.max_drawdown:.1%}")
```

Implemente com scipy para testes estatísticos.
```

---

### 5.4 Agent L3: Tuner

```
PROMPT PARA CLAUDE CODE - AGENT L3: TUNER

Implemente em:
- src/gabagool_v2/learning/tuner.py
- src/gabagool_v2/learning/bandit.py
- src/gabagool_v2/learning/bayes.py

REQUISITOS:

1. tuner.py - TuningOrchestrator:
   - Orquestra o processo de tuning
   - Suporta 3 métodos:
     * Grid search (offline)
     * Multi-armed bandit (online)
     * Bayesian optimization (offline)

2. bandit.py - MultiArmedBandit:
   - Cada "braço" = conjunto de thresholds
   - Algoritmos: epsilon-greedy, UCB, Thompson Sampling
   - Reward = pnl_per_window - risk_penalty
   - Explora novos configs vs explora melhor conhecido

   ```python
   class Arm:
       config: ThresholdConfig
       rewards: List[float]
       pulls: int

   class MultiArmedBandit:
       arms: List[Arm]

       def select_arm(self) -> Arm:
           # UCB selection
           ...

       def update(self, arm: Arm, reward: float):
           ...
   ```

3. bayes.py - BayesianOptimizer:
   - Usa Gaussian Process para modelar função objetivo
   - Acquisition function: Expected Improvement
   - Roda offline com dados históricos

   ```python
   from sklearn.gaussian_process import GaussianProcessRegressor

   class BayesianOptimizer:
       def __init__(self, param_space: Dict):
           self.gp = GaussianProcessRegressor()

       def suggest_next(self) -> ThresholdConfig:
           # Maximize acquisition function
           ...

       def update(self, config: ThresholdConfig, score: float):
           ...
   ```

4. Tuning por bucket:
   - Manter configs separados por:
     * vol_regime (low/normal/high)
     * time_bucket (grupos de horas)
     * liquidity_bucket (low/normal/high)
   - Cada bucket tem seu próprio tuning

5. Output:
   - Salvar em configs/thresholds.yaml
   - Salvar histórico em data/tuning/

EXEMPLO DE USO:

```python
tuner = TuningOrchestrator(method="bandit")

# Online tuning
arm = tuner.select_config(current_regime="high_vol")
# ... execute with arm.config ...
tuner.record_result(arm, pnl=0.35)

# Offline tuning (diário)
await tuner.optimize_offline(
    data_path="data/raw",
    method="bayesian",
    iterations=100
)
```

Implemente com scikit-learn para Gaussian Process.
```

---

### 5.5 Agent L4: Policy Deployer

```
PROMPT PARA CLAUDE CODE - AGENT L4: POLICY DEPLOYER

Implemente em:
- src/gabagool_v2/learning/deployer.py

REQUISITOS:

1. Classe PolicyDeployer:
   - Hot reload de configs sem reiniciar bot
   - Validação de novos configs antes de aplicar
   - Rollback automático se performance degradar

2. Funcionalidades:

   HOT RELOAD:
   - watch_config_file() - detecta mudanças
   - reload_config() - aplica novo config
   - validate_config(new_config) -> bool

   SAFETY CHECKS:
   - Novo config não pode violar safety limits
   - Novo config precisa ser "razoável" (não extremo)
   - Comparar com baseline antes de aplicar

   ROLLBACK:
   - Monitorar performance após deploy
   - Se piorar por X janelas consecutivas, rollback
   - Manter histórico de configs aplicados

   A/B TESTING:
   - Suportar 2 configs rodando em paralelo
   - Split de mercados entre configs
   - Comparar resultados estatisticamente

3. Proteções:
   ```python
   class DeployProtection:
       min_windows_before_rollback: int = 10
       max_performance_drop_pct: float = 0.20
       require_statistical_significance: bool = True
       cooldown_after_rollback_minutes: int = 60
   ```

4. Logging:
   - Log toda mudança de config
   - Log motivo de rollback
   - Métricas de sucesso/falha de deploys

EXEMPLO DE USO:

```python
deployer = PolicyDeployer(config, protections)

# Hot reload
new_config = load_yaml("configs/thresholds.yaml")
if deployer.validate_and_deploy(new_config):
    print("Config deployed successfully")
else:
    print("Config rejected - validation failed")

# Monitor for rollback
deployer.record_window_result(pnl=0.35)
if deployer.should_rollback():
    deployer.rollback()
```

Implemente com watchdog para file monitoring.
```

---

## 6. KPIs e Métricas

### 6.1 KPIs Primários (os que mandam)

| KPI | Descrição | Target |
|-----|-----------|--------|
| Net PnL per Window | Lucro líquido por janela de 15min | > $0.20 |
| Edge Realized | (1 - pair_cost) médio | > 2% |
| Fill Rate | % de ordens que executam | > 60% |
| Win Rate | % de janelas com lucro | > 70% |
| Sharpe-like | risk-adjusted return | > 1.5 |

### 6.2 KPIs Secundários

| KPI | Descrição | Target |
|-----|-----------|--------|
| Max Drawdown | Pior perda consecutiva | < 10% |
| Time in Exposure | Tempo médio unhedged | < 5 min |
| Slippage vs Expected | Impacto real vs estimado | < 50bps |
| Edge Decay Speed | Quão rápido mercado corrige | > 30s |
| Capital Efficiency | PnL / capital alocado | > 0.5% / window |

### 6.3 Métricas Operacionais

| Métrica | Descrição | Alert Threshold |
|---------|-----------|-----------------|
| WS Latency | Latência do WebSocket | > 500ms |
| API Errors | Erros de API por hora | > 10 |
| Reconnects | Reconexões por hora | > 5 |
| Order Rejections | Ordens rejeitadas | > 3 |
| Memory Usage | Uso de memória | > 500MB |

---

## 7. Scripts de Execução

### 7.1 run_live.py

```bash
# Paper trading
python scripts/run_live.py --mode paper --balance 1000

# Live trading
python scripts/run_live.py --mode live --config configs/production.yaml

# With custom thresholds
python scripts/run_live.py --mode paper --thresholds configs/aggressive.yaml
```

### 7.2 run_backtest.py

```bash
# Backtest com dados reais
python scripts/run_backtest.py \
    --data data/raw \
    --start 2024-01-01 \
    --end 2024-01-31 \
    --balance 1000

# Backtest com dados sintéticos
python scripts/run_backtest.py \
    --synthetic \
    --markets 1000 \
    --volatility 0.002

# Backtest com tuning
python scripts/run_backtest.py \
    --data data/raw \
    --tune \
    --method bayesian \
    --iterations 50
```

### 7.3 tune_thresholds.py

```bash
# Grid search
python scripts/tune_thresholds.py \
    --method grid \
    --data data/raw

# Bayesian optimization
python scripts/tune_thresholds.py \
    --method bayesian \
    --iterations 100 \
    --output configs/tuned_thresholds.yaml

# Online bandit (roda com bot)
python scripts/tune_thresholds.py \
    --method bandit \
    --live
```

---

## 8. Testes

### 8.1 test_accounting.py

```python
# Testar contabilidade de legs
def test_leg_accumulation():
    leg = Leg(Side.UP)
    leg.add_fill(Fill(side=Side.UP, price=0.45, qty=10))
    leg.add_fill(Fill(side=Side.UP, price=0.50, qty=10))

    assert leg.total_qty == 20
    assert leg.avg_price == 0.475

def test_pair_cost_calculation():
    pos = PairPosition(market_id="test")
    pos.up_leg.add_fill(Fill(side=Side.UP, price=0.48, qty=10))
    pos.down_leg.add_fill(Fill(side=Side.DOWN, price=0.48, qty=10))

    assert pos.pair_cost == 0.96
    assert pos.guaranteed_profit == 0.40
```

### 8.2 test_pair_cost.py

```python
# Testar cálculos de pair cost em cenários
def test_pair_cost_with_fees():
    accounting = AccountingEngine(FeeConfig(taker_fee=0.02))

    fill_up = accounting.create_fill(
        side=Side.UP, price=0.48, qty=10,
        fill_type=FillType.TAKER, market_id="test"
    )

    assert fill_up.cost == 4.80
    assert fill_up.fee == 0.096  # 2% de 4.80
```

### 8.3 test_inventory.py

```python
# Testar constraints de inventory
def test_imbalance_detection():
    pos = PairPosition(market_id="test")
    pos.up_leg.add_fill(Fill(side=Side.UP, price=0.50, qty=15))
    pos.down_leg.add_fill(Fill(side=Side.DOWN, price=0.50, qty=10))

    assert pos.imbalance_ratio == 1.5
    assert not pos.is_balanced  # > 1.25 threshold
```

---

## 9. Deployment

### 9.1 Docker Compose

```yaml
version: '3.8'

services:
  gabagool-v2-paper:
    build: .
    command: python scripts/run_live.py --mode paper
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs
    restart: unless-stopped

  gabagool-v2-live:
    build: .
    command: python scripts/run_live.py --mode live
    environment:
      - LOG_LEVEL=INFO
      - POLY_API_KEY=${POLY_API_KEY}
      - POLY_SECRET=${POLY_SECRET}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./configs:/app/configs
    profiles:
      - live
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    profiles:
      - monitoring

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    profiles:
      - monitoring

volumes:
  grafana-data:
```

### 9.2 Comandos de Deploy

```bash
# Paper trading
docker-compose up gabagool-v2-paper

# Live trading
docker-compose --profile live up gabagool-v2-live

# Com monitoring
docker-compose --profile monitoring up

# Logs
docker-compose logs -f gabagool-v2-paper
```

---

## 10. Roadmap de Implementação

### Fase 1: Core (Semana 1-2)
- [ ] Implementar `core/types.py`
- [ ] Implementar `core/state.py`
- [ ] Implementar `core/accounting.py`
- [ ] Implementar `core/config.py`
- [ ] Testes unitários do core

### Fase 2: Market Data (Semana 2-3)
- [ ] Implementar `polymarket/gamma_client.py`
- [ ] Implementar `polymarket/clob_ws.py`
- [ ] Implementar `marketdata/orderbook.py`
- [ ] Implementar `marketdata/spot_feed.py`
- [ ] Testes de integração

### Fase 3: Strategy (Semana 3-4)
- [ ] Implementar `signals/volatility.py`
- [ ] Implementar `signals/edge.py`
- [ ] Implementar `strategy/gabagool.py`
- [ ] Implementar `strategy/sizing.py`
- [ ] Backtest inicial

### Fase 4: Execution (Semana 4-5)
- [ ] Implementar `polymarket/order_router.py`
- [ ] Implementar `polymarket/fees.py`
- [ ] Implementar `risk/inventory.py`
- [ ] Paper trading end-to-end

### Fase 5: Learning (Semana 5-6)
- [ ] Implementar `learning/features.py`
- [ ] Implementar `learning/evaluator.py`
- [ ] Implementar `learning/bandit.py`
- [ ] Tuning inicial

### Fase 6: Ops & Deploy (Semana 6-7)
- [ ] Implementar `ops/metrics.py`
- [ ] Implementar `ops/alerts.py`
- [ ] Docker setup
- [ ] Monitoring setup
- [ ] Live deployment

---

## 11. Referências

- [Polymarket CLOB API Docs](https://docs.polymarket.com)
- [py-clob-client](https://github.com/Polymarket/py-clob-client)
- [Gamma API](https://gamma-api.polymarket.com)
- [Multi-Armed Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Bayesian Optimization](https://arxiv.org/abs/1807.02811)

---

*Documento criado: 2026-01-16*
*Versão: 2.0.0*
*Autor: Claude Code*
