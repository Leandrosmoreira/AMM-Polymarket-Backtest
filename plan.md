# Plano de Backtest: Market Maker Hedgeado com Controle de Inventário

## 1. Análise do Print - Gabagool Strategy

### 1.1 Dados Observados (BTC Up/Down - Nov 18, 4:30-4:45 AM)

```
RESUMO DA OPERAÇÃO:
- Duração: 15 minutos
- Total Gasto: $285.35
- Shares YES compradas: 292.20 sh ($189.57)
- Shares NO compradas: 298.20 sh ($95.78)
- Resolução: YES ganhou
- Valor Final: $292.20
- PNL FINAL: +$6.85 (ROI: 2.4%)
```

### 1.2 Padrão de Trading Identificado

```
FASE 1 (Trades 1-20): Acumulação Inicial
- Compras balanceadas ~$10/ordem
- Preços YES: $0.43-0.55
- Preços NO: $0.36-0.43
- Spread capturado: ~2-4%

FASE 2 (Trades 20-40): Market Making Ativo
- Ordens menores ($6-10)
- Ajuste de preços conforme mercado
- Rebalanceamento de inventário

FASE 3 (Trades 40-60): Posicionamento Final
- Preços YES subindo (0.65-0.92)
- Preços NO caindo (0.08-0.33)
- Redução de exposição NO
- Manutenção de hedge parcial
```

### 1.3 Observações do Order Book

```
VOLUME ANALYSIS:
- YES total: 292.20 shares @ avg $0.65
- NO total: 298.20 shares @ avg $0.32
- Ratio YES/NO shares: ~1:1 (bem balanceado)
- Ratio YES/NO dollars: ~2:1 (desequilíbrio intencional)

SPREAD DYNAMICS:
- Spread inicial: ~4% (YES+NO = $0.96)
- Spread médio: ~2-3%
- Spread final: ~1% (mercado convergindo)
```

---

## 2. Modelo Proposto: Hedged Market Maker (HMM)

### 2.1 Conceito Central

```
OBJETIVO: Capturar spread atuando como Market Maker nos dois lados,
mantendo inventário balanceado e hedge constante.

DIFERENÇA DO GABAGOOL ORIGINAL:
- Gabagool: Compra quando UP+DOWN < $1.00 (arbitragem pura)
- HMM: Posta LIMIT orders nos dois lados, captura bid-ask spread
```

### 2.2 Parâmetros do Modelo

```python
class HMMConfig:
    # Tamanho das ordens
    ORDER_SIZE_USD = 10.0          # USD por ordem
    MIN_ORDER_SIZE = 5.0           # Mínimo para executar
    MAX_ORDER_SIZE = 20.0          # Máximo por ordem

    # Controle de inventário
    MAX_INVENTORY_IMBALANCE = 0.3  # Máximo 30% de desequilíbrio
    TARGET_INVENTORY_RATIO = 0.5   # 50/50 ideal
    REBALANCE_THRESHOLD = 0.2     # Rebalancear quando >20% off

    # Spread e preços
    MIN_SPREAD_TO_QUOTE = 0.015    # Mínimo 1.5% spread para cotar
    QUOTE_OFFSET = 0.01            # 1 centavo dentro do spread
    MAX_POSITION_PCT = 0.25        # Máximo 25% do capital por lado

    # Tempo
    MARKET_DURATION = 900          # 15 minutos
    STOP_QUOTING_SECONDS = 60      # Para de cotar 1 min antes do fim
    QUOTE_REFRESH_MS = 1000        # Atualiza quotes a cada 1 seg

    # Volatilidade
    VOL_WINDOW_SECONDS = 300       # Janela de 5 min para vol
    HIGH_VOL_THRESHOLD = 0.003     # Vol alta = spread maior
    LOW_VOL_THRESHOLD = 0.001      # Vol baixa = spread menor
```

---

## 3. Modelo de Volatilidade (15 min BTC)

### 3.1 Cálculo de Volatilidade

```python
class BTCVolatilityModel:
    """
    Modelo de volatilidade específico para mercados de 15 min.

    Características do BTC em 15 min:
    - Vol média: 0.1-0.3% por minuto
    - Vol pode spike 3-5x em eventos
    - Autocorrelação de ~60 segundos
    """

    def __init__(self):
        self.vol_window = 300  # 5 minutos
        self.prices = []
        self.returns = []

    def calculate_realized_vol(self) -> float:
        """Vol realizada dos últimos 5 min."""
        if len(self.returns) < 30:
            return 0.002  # Default
        return np.std(self.returns[-300:]) * np.sqrt(60)  # Por minuto

    def calculate_implied_vol(self, up_price, down_price, time_remaining) -> float:
        """Vol implícita dos preços do mercado."""
        # Se preços muito próximos de 0.50, vol implícita alta
        # Se preços extremos (0.90/0.10), vol implícita baixa
        mid = (up_price + down_price) / 2
        uncertainty = 4 * up_price * down_price  # Máximo em 0.50/0.50
        time_factor = time_remaining / 900  # Normalizado
        return uncertainty * time_factor * 0.01

    def get_vol_regime(self) -> str:
        """Regime de volatilidade atual."""
        vol = self.calculate_realized_vol()
        if vol > 0.003:
            return "HIGH"
        elif vol < 0.001:
            return "LOW"
        return "NORMAL"
```

### 3.2 Ajuste de Spread por Volatilidade

```python
def calculate_dynamic_spread(base_spread: float, vol_regime: str,
                             inventory_skew: float, time_remaining: int) -> float:
    """
    Spread dinâmico baseado em:
    1. Volatilidade atual
    2. Desequilíbrio de inventário
    3. Tempo restante no mercado
    """
    spread = base_spread

    # Ajuste por volatilidade
    if vol_regime == "HIGH":
        spread *= 1.5  # Spread 50% maior em vol alta
    elif vol_regime == "LOW":
        spread *= 0.8  # Spread 20% menor em vol baixa

    # Ajuste por inventário (penaliza lado com excesso)
    # Se muito comprado em YES, aumenta spread de compra YES
    spread *= (1 + abs(inventory_skew) * 0.5)

    # Ajuste por tempo (spread maior no final)
    if time_remaining < 120:  # Últimos 2 min
        spread *= 1.5
    elif time_remaining < 60:  # Último minuto
        spread *= 2.0

    return min(spread, 0.10)  # Cap em 10%
```

---

## 4. Controle de Inventário (Bag Management)

### 4.1 Métricas de Inventário

```python
@dataclass
class InventoryState:
    """Estado do inventário em tempo real."""

    # Posições
    yes_shares: float = 0.0
    no_shares: float = 0.0
    yes_cost: float = 0.0
    no_cost: float = 0.0

    # Métricas derivadas
    @property
    def total_shares(self) -> float:
        return self.yes_shares + self.no_shares

    @property
    def total_cost(self) -> float:
        return self.yes_cost + self.no_cost

    @property
    def inventory_ratio(self) -> float:
        """Ratio YES/(YES+NO). 0.5 = balanceado."""
        if self.total_shares == 0:
            return 0.5
        return self.yes_shares / self.total_shares

    @property
    def inventory_skew(self) -> float:
        """Desvio do balanço ideal. -1 a +1."""
        return (self.inventory_ratio - 0.5) * 2

    @property
    def net_exposure(self) -> float:
        """Exposição líquida em USD."""
        return self.yes_cost - self.no_cost

    def is_balanced(self, threshold: float = 0.2) -> bool:
        """Inventário está balanceado?"""
        return abs(self.inventory_skew) <= threshold
```

### 4.2 Estratégia de Rebalanceamento

```python
class InventoryManager:
    """
    Gerencia inventário para manter hedge.

    REGRAS:
    1. Nunca ficar >70% em um lado
    2. Rebalancear quando >60% em um lado
    3. Ajustar preços para incentivar lado oposto
    4. No final do mercado, aceitar desequilíbrio se spread bom
    """

    def calculate_quote_adjustment(self, inventory: InventoryState) -> Tuple[float, float]:
        """
        Retorna ajuste de preço para cada lado.
        Positivo = mais agressivo (melhor preço)
        Negativo = menos agressivo (pior preço)
        """
        skew = inventory.inventory_skew

        # Se muito YES, melhorar preço NO para atrair vendas YES
        yes_adjustment = -skew * 0.02  # Até 2 centavos
        no_adjustment = skew * 0.02

        return yes_adjustment, no_adjustment

    def should_quote_side(self, side: str, inventory: InventoryState,
                          config: HMMConfig) -> bool:
        """Deve cotar este lado?"""
        if side == "yes":
            # Não comprar mais YES se já tem muito
            return inventory.inventory_ratio < (0.5 + config.MAX_INVENTORY_IMBALANCE)
        else:
            # Não comprar mais NO se já tem muito
            return inventory.inventory_ratio > (0.5 - config.MAX_INVENTORY_IMBALANCE)

    def calculate_order_size(self, side: str, inventory: InventoryState,
                            config: HMMConfig) -> float:
        """Tamanho da ordem baseado no inventário."""
        base_size = config.ORDER_SIZE_USD
        skew = inventory.inventory_skew

        if side == "yes":
            # Reduzir tamanho se já muito YES
            multiplier = max(0.5, 1 - skew)
        else:
            # Reduzir tamanho se já muito NO
            multiplier = max(0.5, 1 + skew)

        return base_size * multiplier
```

---

## 5. Motor de Quoting

### 5.1 Geração de Quotes

```python
class QuoteEngine:
    """
    Gera quotes (bid/ask) para ambos os lados.

    ESTRATÉGIA:
    - Sempre postar LIMIT orders (0% fee)
    - Quotes dentro do spread atual
    - Ajustar por inventário e volatilidade
    - Cancelar e repostar quando mercado move
    """

    def generate_quotes(self, market_state: MarketState,
                       inventory: InventoryState,
                       vol_model: BTCVolatilityModel) -> List[Quote]:
        """Gera quotes para ambos os lados."""
        quotes = []

        # Calcular spread dinâmico
        vol_regime = vol_model.get_vol_regime()
        spread = calculate_dynamic_spread(
            base_spread=0.02,
            vol_regime=vol_regime,
            inventory_skew=inventory.inventory_skew,
            time_remaining=market_state.time_remaining
        )

        # Ajustes por inventário
        yes_adj, no_adj = self.inventory_manager.calculate_quote_adjustment(inventory)

        # Quote YES (compra)
        if self.inventory_manager.should_quote_side("yes", inventory, self.config):
            yes_bid = market_state.yes_best_bid + 0.01 + yes_adj
            yes_bid = min(yes_bid, market_state.yes_best_ask - spread)

            quotes.append(Quote(
                side="yes",
                price=yes_bid,
                size=self.inventory_manager.calculate_order_size("yes", inventory, self.config),
                order_type="GTC"  # Limit order
            ))

        # Quote NO (compra)
        if self.inventory_manager.should_quote_side("no", inventory, self.config):
            no_bid = market_state.no_best_bid + 0.01 + no_adj
            no_bid = min(no_bid, market_state.no_best_ask - spread)

            quotes.append(Quote(
                side="no",
                price=no_bid,
                size=self.inventory_manager.calculate_order_size("no", inventory, self.config),
                order_type="GTC"
            ))

        return quotes
```

### 5.2 Lógica de Fill

```python
class FillSimulator:
    """
    Simula fills de ordens limit baseado em dados reais.

    MODELO DE FILL:
    - Probabilidade base de fill: 30%
    - Aumenta se preço mais agressivo
    - Aumenta em vol alta
    - Diminui perto do fim do mercado
    """

    def calculate_fill_probability(self, quote: Quote, market_state: MarketState,
                                   vol_regime: str) -> float:
        """Probabilidade de uma ordem limit ser executada."""
        base_prob = 0.30

        # Ajuste por agressividade do preço
        if quote.side == "yes":
            spread_to_ask = market_state.yes_best_ask - quote.price
            aggressiveness = 1 - (spread_to_ask / 0.05)  # Normalizado
        else:
            spread_to_ask = market_state.no_best_ask - quote.price
            aggressiveness = 1 - (spread_to_ask / 0.05)

        base_prob *= (1 + aggressiveness * 0.5)

        # Ajuste por volatilidade
        if vol_regime == "HIGH":
            base_prob *= 1.5  # Mais fills em vol alta
        elif vol_regime == "LOW":
            base_prob *= 0.7  # Menos fills em vol baixa

        # Ajuste por tempo
        if market_state.time_remaining < 60:
            base_prob *= 0.5  # Menos fills no final

        return min(base_prob, 0.80)  # Cap em 80%
```

---

## 6. Gestão de Risco

### 6.1 Limites de Exposição

```python
class RiskManager:
    """
    Controle de risco para o market maker.

    REGRAS:
    1. Max exposição total: 50% do capital
    2. Max exposição líquida: 20% do capital
    3. Stop loss: -5% do capital
    4. Para de cotar se volatilidade > 3x normal
    """

    def __init__(self, capital: float):
        self.capital = capital
        self.max_gross_exposure = capital * 0.50
        self.max_net_exposure = capital * 0.20
        self.stop_loss_pct = 0.05

    def check_limits(self, inventory: InventoryState) -> Tuple[bool, str]:
        """Verifica se está dentro dos limites."""

        # Exposição bruta
        if inventory.total_cost > self.max_gross_exposure:
            return False, "MAX_GROSS_EXPOSURE"

        # Exposição líquida
        if abs(inventory.net_exposure) > self.max_net_exposure:
            return False, "MAX_NET_EXPOSURE"

        # Stop loss (P&L não realizado)
        # ... calcular MTM

        return True, "OK"

    def calculate_max_order_size(self, inventory: InventoryState) -> float:
        """Tamanho máximo permitido para próxima ordem."""
        remaining_exposure = self.max_gross_exposure - inventory.total_cost
        return max(0, min(remaining_exposure, 20.0))
```

### 6.2 Comportamento no Final do Mercado

```python
class EndGameStrategy:
    """
    Estratégia para os últimos minutos do mercado.

    FASES:
    - T-5 min: Reduzir tamanho de ordens
    - T-2 min: Parar de cotar lado perdedor
    - T-1 min: Apenas reduzir posição se possível
    - T-30 seg: Aceitar posição final
    """

    def get_phase(self, time_remaining: int) -> str:
        if time_remaining > 300:
            return "NORMAL"
        elif time_remaining > 120:
            return "REDUCE"
        elif time_remaining > 60:
            return "DEFENSIVE"
        else:
            return "HOLD"

    def adjust_strategy(self, phase: str, inventory: InventoryState,
                       market_state: MarketState) -> StrategyAdjustment:
        """Ajusta estratégia baseado na fase."""

        if phase == "NORMAL":
            return StrategyAdjustment(
                size_multiplier=1.0,
                quote_both_sides=True,
                spread_multiplier=1.0
            )

        elif phase == "REDUCE":
            return StrategyAdjustment(
                size_multiplier=0.5,
                quote_both_sides=True,
                spread_multiplier=1.3
            )

        elif phase == "DEFENSIVE":
            # Identificar lado provável vencedor
            likely_winner = "yes" if market_state.yes_price > 0.60 else "no"

            return StrategyAdjustment(
                size_multiplier=0.3,
                quote_both_sides=False,
                preferred_side=likely_winner,
                spread_multiplier=1.5
            )

        else:  # HOLD
            return StrategyAdjustment(
                size_multiplier=0.0,
                quote_both_sides=False,
                spread_multiplier=2.0
            )
```

---

## 7. Simulação de Distorção de Spread

### 7.1 Modelo de Impacto de Mercado

```python
class MarketImpactModel:
    """
    Modela como nossas ordens afetam o spread.

    OBSERVAÇÕES DO PRINT:
    - Quando compra muito YES, spread YES aumenta
    - Quando compra muito NO, spread NO aumenta
    - Outros MMs ajustam preços baseado no nosso flow
    """

    def calculate_spread_impact(self, side: str, size: float,
                                market_state: MarketState) -> float:
        """
        Impacto no spread após executar ordem.
        Retorna aumento esperado no spread.
        """
        # Impacto base proporcional ao tamanho
        base_impact = size / 100 * 0.005  # 0.5 centavo por $100

        # Impacto maior se mercado ilíquido
        liquidity = market_state.get_liquidity(side)
        liquidity_factor = 100 / max(liquidity, 10)  # Mais impacto se menos líquido

        return base_impact * liquidity_factor

    def predict_price_after_trade(self, side: str, size: float,
                                  current_price: float,
                                  market_state: MarketState) -> float:
        """Preço esperado após executar trade."""
        impact = self.calculate_spread_impact(side, size, market_state)

        # Compra empurra preço para cima
        return current_price + impact
```

### 7.2 Detecção de Distorção

```python
class SpreadDistortionDetector:
    """
    Detecta quando spread está distorcido por execuções pesadas.

    SINAIS DE DISTORÇÃO:
    1. Spread muito maior que normal
    2. Um lado com muito mais volume
    3. Preço se moveu muito rápido
    """

    def __init__(self):
        self.spread_history = []
        self.volume_history = {"yes": [], "no": []}
        self.normal_spread = 0.02

    def detect_distortion(self, market_state: MarketState) -> Dict:
        """Detecta distorção no spread."""
        current_spread = 1 - (market_state.yes_price + market_state.no_price)

        # Spread distorcido?
        spread_ratio = current_spread / self.normal_spread
        is_spread_distorted = spread_ratio > 1.5 or spread_ratio < 0.5

        # Volume desbalanceado?
        yes_vol = sum(self.volume_history["yes"][-10:])
        no_vol = sum(self.volume_history["no"][-10:])
        vol_ratio = yes_vol / max(no_vol, 1)
        is_vol_distorted = vol_ratio > 2 or vol_ratio < 0.5

        return {
            "spread_distorted": is_spread_distorted,
            "spread_ratio": spread_ratio,
            "volume_distorted": is_vol_distorted,
            "volume_ratio": vol_ratio,
            "recommended_action": self._get_recommendation(spread_ratio, vol_ratio)
        }

    def _get_recommendation(self, spread_ratio: float, vol_ratio: float) -> str:
        """Recomendação baseada na distorção."""
        if spread_ratio > 1.5:
            return "WIDEN_QUOTES"  # Spread muito alto, oportunidade
        elif spread_ratio < 0.5:
            return "TIGHTEN_QUOTES"  # Spread muito baixo, cuidado
        elif vol_ratio > 2:
            return "BUY_NO"  # Muito volume YES, comprar NO
        elif vol_ratio < 0.5:
            return "BUY_YES"  # Muito volume NO, comprar YES
        return "CONTINUE"
```

---

## 8. Estrutura do Backtest

### 8.1 Classes Principais

```
src/hedged_mm/
├── __init__.py
├── config.py              # HMMConfig
├── volatility_model.py    # BTCVolatilityModel
├── inventory_manager.py   # InventoryManager, InventoryState
├── quote_engine.py        # QuoteEngine, Quote
├── fill_simulator.py      # FillSimulator
├── risk_manager.py        # RiskManager
├── spread_detector.py     # SpreadDistortionDetector
├── end_game.py            # EndGameStrategy
├── backtest.py            # HedgedMMBacktest
└── bot.py                 # HedgedMMBot (para trading real)
```

### 8.2 Loop Principal do Backtest

```python
class HedgedMMBacktest:
    """
    Backtest do Market Maker Hedgeado.
    """

    def run(self, data: Dict) -> BacktestResult:
        """Executa backtest."""

        for market in self.segment_markets(data):
            # Reset estado
            inventory = InventoryState()

            for tick in market.ticks:
                # 1. Atualizar volatilidade
                self.vol_model.update(tick.btc_price, tick.timestamp)

                # 2. Verificar limites de risco
                can_trade, reason = self.risk_manager.check_limits(inventory)
                if not can_trade:
                    continue

                # 3. Detectar distorção
                distortion = self.spread_detector.detect_distortion(market_state)

                # 4. Ajustar por fase do mercado
                phase = self.end_game.get_phase(market.time_remaining)
                adjustment = self.end_game.adjust_strategy(phase, inventory, market_state)

                # 5. Gerar quotes
                quotes = self.quote_engine.generate_quotes(
                    market_state, inventory, self.vol_model, adjustment
                )

                # 6. Simular fills
                for quote in quotes:
                    if self.fill_simulator.simulate_fill(quote, market_state):
                        self.execute_fill(quote, inventory)

                # 7. Registrar estado
                self.record_state(tick.timestamp, inventory, market_state)

            # Settlement
            self.settle_market(market, inventory)
```

---

## 9. Métricas de Avaliação

### 9.1 KPIs do Backtest

```python
@dataclass
class HMMBacktestResult:
    # Performance
    total_pnl: float
    roi_pct: float
    sharpe_ratio: float

    # Trading Activity
    total_quotes: int
    total_fills: int
    fill_rate: float
    avg_spread_captured: float

    # Inventory
    avg_inventory_skew: float
    max_inventory_skew: float
    time_balanced_pct: float  # % do tempo com inv balanceado

    # Risk
    max_drawdown: float
    max_exposure: float
    var_95: float

    # Por mercado
    markets_traded: int
    avg_pnl_per_market: float
    win_rate: float

    # Spread Analysis
    avg_spread_quoted: float
    spread_efficiency: float  # spread capturado / spread cotado
```

---

## 10. Próximos Passos

### 10.1 Implementação

1. [ ] Criar estrutura de diretórios `src/hedged_mm/`
2. [ ] Implementar `BTCVolatilityModel`
3. [ ] Implementar `InventoryManager`
4. [ ] Implementar `QuoteEngine`
5. [ ] Implementar `FillSimulator`
6. [ ] Implementar `SpreadDistortionDetector`
7. [ ] Implementar `HedgedMMBacktest`
8. [ ] Adicionar comando em `main.py`
9. [ ] Testar com dados reais da VPS

### 10.2 Parâmetros para Otimizar

```python
PARAMS_TO_OPTIMIZE = [
    "ORDER_SIZE_USD",           # 5-20
    "MAX_INVENTORY_IMBALANCE",  # 0.2-0.4
    "MIN_SPREAD_TO_QUOTE",      # 0.01-0.03
    "QUOTE_OFFSET",             # 0.005-0.02
    "HIGH_VOL_THRESHOLD",       # 0.002-0.005
    "STOP_QUOTING_SECONDS",     # 30-120
]
```

### 10.3 Comando Final

```bash
python main.py hmm-backtest --data data/raw --balance 1000 \
    --order-size 10 --max-imbalance 0.3 --min-spread 0.015
```

---

## 11. Diferenças das Outras Estratégias

| Aspecto | Gabagool | Vol Arb | Scalp | Edge Prop | **HMM** |
|---------|----------|---------|-------|-----------|---------|
| Hedge | 100% | 0% | 100% | Parcial | **Dinâmico** |
| Ordem | Market/Limit | Limit | Limit | Limit | **Limit Only** |
| Inv Control | Não | Não | Não | Não | **Sim** |
| Vol Model | Não | Sim | Não | Sim | **Sim** |
| Time Decay | Não | Sim | Não | Sim | **Sim** |
| Spread Dist | Não | Não | Não | Não | **Sim** |

---

*Documento criado em: 2026-01-03*
*Baseado na análise do trading do Gabagool*
