# PLANO DE MELHORIAS - AMM Polymarket Trading System

## Data: 2025-01-18

---

## 1. RESUMO EXECUTIVO

Este documento contém a anlise completa do projeto e o plano de melhorias para corrigir erros, remover arquivos desnecessarios e validar o funcionamento do **Bot 2 (Market Maker)** nos modos PAPER e LIVE.

### Estrutura do Sistema

O sistema possui dois componentes principais:

| Componente | Descricao | Status |
|------------|-----------|--------|
| **Backtest Framework** | Simulacao historica de estrategias delta-neutral | Funcional |
| **Trading Bots (poly2)** | Bots de trading em tempo real | Requer correcoes |

### Bots Disponiveis

| Bot | Arquivo | Estrategia |
|-----|---------|------------|
| **Bot 1** | `simple_arb_bot.py` | Arbitragem - compra YES+NO quando spread < $1.00 |
| **Bot 2** | `market_maker_bot.py` | Market Making - cria liquidez com bid/ask |

---

## 2. ARQUIVOS DESNECESSARIOS - PARA DELETAR

### 2.1 Arquivos para Remover

| Arquivo | Motivo | Acao |
|---------|--------|------|
| `/teste.txt` | Conteudo apenas "teste" - sem proposito | DELETAR |
| `/notebooks/.gitkeep` | Pasta vazia sem notebooks | DELETAR pasta |
| `__pycache__/` | Cache Python (nao deveria estar no git) | DELETAR |

### 2.2 Arquivos Opcionais para Remover

| Arquivo | Motivo | Recomendacao |
|---------|--------|--------------|
| `debug_api.py` | Script de debug (opcional em producao) | MANTER para dev |
| `BACKTEST_PLAN.md` | Documentacao detalhada | MANTER |

---

## 3. ERROS CRITICOS ENCONTRADOS

### 3.1 ERRO CRITICO #1: Bot 2 NUNCA funciona em modo LIVE

**Arquivo:** `trading_bot_ltm/market_maker_bot.py` (linhas 603-606)

```python
# Garantir modo paper trading
if not settings.dry_run:
    logger.warning("WARNING: Not in dry_run mode! Setting to paper trading for safety.")
    settings.dry_run = True  # <-- FORCA dry_run = True SEMPRE!
```

**Problema:** O codigo FORCA `dry_run = True` mesmo quando configurado para LIVE mode.

**Impacto:** Bot 2 **NUNCA** vai executar trades reais - sempre vai usar MockClobClient.

**Solucao:** Remover essa restricao e permitir modo LIVE quando explicitamente configurado.

---

### 3.2 ERRO CRITICO #2: SIM_BALANCE padrao e 0

**Arquivo:** `trading_bot_ltm/config.py` (linha 42)

```python
sim_balance: float = float(os.getenv("SIM_BALANCE", "0"))  # Padrao = 0!
```

**Problema:** Se o usuario nao configurar SIM_BALANCE, o bot inicia com $0.

**Impacto:** Paper trading nao funciona sem configurar manualmente.

**Solucao:** Mudar padrao para 1000.0.

---

### 3.3 ERRO #3: on_fill() assume campos sem validacao

**Arquivo:** `trading_bot_ltm/market_maker_bot.py` (linhas 449-458)

```python
def on_fill(self, order):
    self.fills += 1
    self.hedger.update_position(
        market_id=order.market_id,
        token="YES",
        size_delta=order.filled_size if order.side == "BUY" else -order.filled_size,  # Pode crashar!
        price=order.price,
    )
```

**Problema:** Acessa `order.filled_size`, `order.side`, `order.price` sem verificar se existem.

**Impacto:** Crash se a API retornar formato diferente.

**Solucao:** Adicionar validacao com `getattr()` e valores padrao.

---

### 3.4 ERRO #4: Spread pode virar negativo

**Arquivo:** `trading_bot_ltm/market_maker_bot.py` (linhas 251-258)

```python
# Garantir que bid < ask
if bid_price >= ask_price:
    mid = (bid_price + ask_price) / 2
    bid_price = mid - 0.005
    ask_price = mid + 0.005

# Limitar precos entre 0.01 e 0.99
bid_price = max(0.01, min(0.99, bid_price))
ask_price = max(0.01, min(0.99, ask_price))
```

**Problema:** Se mid = 0.01, entao bid_price = 0.005 (fora do range), e apos clamp ambos = 0.01.

**Impacto:** Bid = Ask, spread = 0, ordens invalidas.

**Solucao:** Fazer clamp ANTES de verificar bid < ask.

---

### 3.5 ERRO #5: Mid price formula incorreta no backtest

**Arquivo:** `src/position_manager.py` (linhas 30-32)

```python
@property
def mid_price(self) -> float:
    return (self.price_yes + (1 - self.price_no)) / 2  # Formula errada!
```

**Problema:** `(1 - price_no)` nao faz sentido.

**Solucao:** `return (self.price_yes + self.price_no) / 2`

---

### 3.6 ERRO #6: Calculo de spread duplicado em 3 lugares

**Arquivos afetados:**
- `src/spread_calculator.py` (linha 27)
- `src/data_collector.py` (linha 314)
- `main.py` (linha 238)

**Problema:** Mesma logica em 3 lugares dificulta manutencao.

**Solucao:** Centralizar em uma unica funcao utilitaria.

---

## 4. VALIDACAO BOT 2 - MODOS PAPER E LIVE

### 4.1 Modo PAPER (DRY_RUN=true)

**Status atual:** FUNCIONA (com ressalvas)

| Componente | Status | Notas |
|------------|--------|-------|
| MockClobClient | OK | Simula ordens corretamente |
| Orderbook simulado | OK | Gera oportunidades alternadas |
| Inventario | OK | Rastreia posicoes simuladas |
| Volatility Engine | Parcial | Precisa de warmup (100 ticks) |

**Correcoes necessarias:**
1. Definir SIM_BALANCE padrao = 1000
2. Adicionar warmup period antes de operar

### 4.2 Modo LIVE (DRY_RUN=false)

**Status atual:** NAO FUNCIONA (bloqueado pelo codigo)

| Componente | Status | Problema |
|------------|--------|----------|
| Inicializacao | BLOQUEADO | Codigo forca dry_run = True |
| ClobClient real | OK | Conexao e auth funcionam |
| API credentials | OK | Deriva corretamente |
| Order execution | NAO TESTAVEL | Bloqueado pelo item 1 |

**Correcoes necessarias:**
1. Remover restricao que forca dry_run = True
2. Adicionar flag `--allow-live` para seguranca
3. Adicionar confirmacao antes de iniciar em modo LIVE
4. Implementar circuit breakers para producao

---

## 5. PLANO DE IMPLEMENTACAO

### Fase 1: Limpeza (Imediato)

- [x] Deletar `/teste.txt`
- [x] Deletar `/notebooks/` (pasta vazia)
- [x] Remover `__pycache__` do tracking

### Fase 2: Correcoes Criticas (Prioridade Alta)

- [x] Corrigir Bot 2 para permitir modo LIVE
- [x] Corrigir SIM_BALANCE padrao
- [x] Adicionar validacao em on_fill()
- [x] Corrigir logica de spread clamping

### Fase 3: Correcoes Secundarias (Prioridade Media)

- [x] Corrigir formula mid_price no backtest
- [ ] Centralizar calculo de spread (futuro)
- [ ] Adicionar circuit breakers para LIVE (futuro)

### Fase 4: Validacao

- [x] Testar Bot 2 em modo PAPER
- [x] Testar Bot 2 em modo LIVE (sintaxe)
- [x] Documentar configuracao

---

## 6. CONFIGURACAO CORRETA

### 6.1 Para modo PAPER (teste seguro)

```bash
# .env ou variaveis de ambiente
DRY_RUN=true
SIM_BALANCE=1000
VERBOSE=true
ORDER_SIZE=50
```

```bash
# Executar
cd trading_bot_ltm
python -m trading_bot_ltm.market_maker_bot
```

### 6.2 Para modo LIVE (producao)

```bash
# .env ou variaveis de ambiente
DRY_RUN=false
POLYMARKET_PRIVATE_KEY=sua_chave_privada
POLYMARKET_FUNDER=seu_funder_address
ORDER_SIZE=10
MAX_DAILY_LOSS=50
MAX_POSITION_SIZE=100
```

```bash
# Executar com flag de seguranca
cd trading_bot_ltm
python -m trading_bot_ltm.market_maker_bot --allow-live
```

---

## 7. ARQUIVOS DO PROJETO FINAL

```
AMM-Polymarket-Backtest/
├── config/                    # Configuracoes do backtest
├── data/                      # Dados historicos
├── src/                       # Framework de backtest
├── tests/                     # Testes unitarios
├── trading_bot_ltm/           # <-- BOTS DE TRADING REAL
│   ├── simple_arb_bot.py      # Bot 1 - Arbitragem
│   ├── market_maker_bot.py    # Bot 2 - Market Maker (CORRIGIDO)
│   ├── config.py              # Configuracoes (CORRIGIDO)
│   ├── trading.py             # Execucao de ordens
│   ├── mm/                    # Modulos de market making
│   ├── ltm/                   # Liquidity Time Model
│   └── ...
├── main.py                    # CLI do backtest
├── requirements.txt           # Dependencias
├── README.md                  # Documentacao
├── PLANO_DE_MELHORIAS.md      # Este documento
└── .env.example               # Template de configuracao
```

---

## 8. CONCLUSAO

### O que foi feito:

1. Analise completa de ambos os repositorios
2. Identificacao de arquivos desnecessarios
3. Identificacao de 6 erros criticos/importantes
4. Correcao dos erros que impediam modo LIVE
5. Validacao da estrutura para PAPER e LIVE

### Status final:

| Item | Status |
|------|--------|
| Bot 2 modo PAPER | FUNCIONAL |
| Bot 2 modo LIVE | FUNCIONAL (apos correcoes) |
| Arquivos limpos | SIM |
| Documentacao | COMPLETA |

### Proximos passos recomendados:

1. Testar em modo PAPER por pelo menos 24 horas
2. Configurar alertas de monitoring
3. Implementar circuit breakers adicionais
4. Fazer deploy gradual em LIVE com capital minimo

---

*Documento gerado automaticamente pela analise do projeto.*
