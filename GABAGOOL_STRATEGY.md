# 🤖 Bot Spread Capture - Estratégia Gabagool

## 📋 Resumo Executivo

Bot de trading para mercados BTC/ETH Up/Down 15min do Polymarket baseado na estratégia comprovada do trader **@gabagool22** ($450k+ de lucro).

**Conceito Central:** Comprar AMBOS os lados (UP e DOWN) quando a soma dos preços for menor que $1.00, garantindo lucro independente do resultado.

---

## 🎯 Análise da Estratégia Gabagool

### Dados Observados (Screenshots)

| Métrica | Valor |
|---------|-------|
| Lucro Total | $450,718.10 |
| Trades | 15,182 |
| Lucro Médio/Trade | ~$29.68 |
| Maior Ganho | $4,325.86 |
| Posições Ativas | $6,000-$8,000 |

### Exemplo Real (November 18, 4:30-4:45 AM)

```
Mercado: Bitcoin Up or Down
Duração: 15 minutos

Compras:
├── YES (Up):  292.20 shares @ avg $0.649 = $189.57
└── NO (Down): 298.20 shares @ avg $0.321 = $95.78

Total Investido: $285.35
Shares Mínimo: min(292.20, 298.20) = 292.20

Resultado: YES ganhou
Payout: 292.20 × $1.00 = $292.20

LUCRO: $292.20 - $285.35 = $6.85 (2.4% em 15 min)
```

### Padrão de Trading Observado

```
Trades por mercado: ~60-70 trades
Tamanho por trade: 10-27 shares ($3-$14)
Intervalo: A cada poucos segundos
Estratégia: Alternar UP/DOWN mantendo equilíbrio
```

---

## 📐 Matemática da Estratégia

### Fórmula Básica

```
Lucro = min(shares_UP, shares_DOWN) × $1.00 - (custo_UP + custo_DOWN)
```

### Condição de Entrada

```
preço_UP + preço_DOWN < $1.00

Exemplo:
  UP = $0.47, DOWN = $0.52
  Total = $0.99
  Spread = $1.00 - $0.99 = $0.01 (1%)
```

### Tabela de Rentabilidade

| Preço UP | Preço DOWN | Total | Spread | ROI por Par |
|----------|------------|-------|--------|-------------|
| $0.40 | $0.55 | $0.95 | $0.05 | 5.26% |
| $0.45 | $0.52 | $0.97 | $0.03 | 3.09% |
| $0.47 | $0.51 | $0.98 | $0.02 | 2.04% |
| $0.48 | $0.51 | $0.99 | $0.01 | 1.01% |
| $0.50 | $0.50 | $1.00 | $0.00 | 0.00% |

### ROI Anualizado (se operando 24/7)

```
ROI por mercado: ~2%
Mercados por dia: 96 (a cada 15 min)
ROI diário: 2% × 96 = 192% (teórico máximo)
ROI realista: ~10-30% ao dia (nem todo mercado tem spread)
```

---

## 🔄 Fluxo de Operação

### Ciclo Principal

```
┌─────────────────────────────────────────────────────────────┐
│                    LOOP PRINCIPAL                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. BUSCAR MERCADOS ATIVOS                                  │
│     └── BTC Up/Down 15min                                   │
│     └── ETH Up/Down 15min                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. VERIFICAR SPREAD (a cada 500ms)                         │
│     ├── Buscar best ask UP                                  │
│     ├── Buscar best ask DOWN                                │
│     └── Calcular: total = ask_UP + ask_DOWN                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. DECISÃO DE TRADE                                        │
│     │                                                       │
│     ├── SE total < threshold (ex: 0.98)                     │
│     │   └── ENTRAR: comprar UP e DOWN                       │
│     │                                                       │
│     └── SE total >= threshold                               │
│         └── AGUARDAR                                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  4. EXECUTAR TRADE (se entrada)                             │
│     ├── Calcular quantidade balanceada                      │
│     ├── Enviar ordem UP                                     │
│     ├── Enviar ordem DOWN                                   │
│     └── Atualizar posição                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  5. GERENCIAR POSIÇÃO                                       │
│     ├── Verificar balanceamento (UP ≈ DOWN)                 │
│     ├── Rebalancear se necessário                           │
│     └── Verificar limites de exposição                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  6. SETTLEMENT                                              │
│     ├── Mercado fecha                                       │
│     ├── Um lado paga $1.00                                  │
│     └── Calcular lucro/prejuízo                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            └──────────► [Próximo mercado]
```

---

## ⚙️ Parâmetros do Bot

### Configuração Principal

| Parâmetro | Valor Sugerido | Descrição |
|-----------|----------------|-----------|
| `MIN_SPREAD` | 0.02 (2%) | Spread mínimo para entrar |
| `MAX_SPREAD` | 0.10 (10%) | Spread máximo (desconfiar) |
| `ORDER_SIZE` | $10-$20 | Tamanho por ordem |
| `MAX_PER_MARKET` | $500 | Máximo por mercado |
| `CHECK_INTERVAL` | 500ms | Intervalo de verificação |
| `REBALANCE_THRESHOLD` | 10% | Quando rebalancear |

### Configuração de Mercados

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MARKETS` | BTC, ETH | Ativos para operar |
| `TIMEFRAME` | 15min | Duração do mercado |
| `MIN_TIME_REMAINING` | 60s | Mínimo para entrar |
| `SKIP_FIRST_MINUTES` | 2min | Pular início (preços instáveis) |

### Limites de Risco

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `MAX_TOTAL_EXPOSURE` | $2,000 | Exposição total máxima |
| `MAX_IMBALANCE` | 20% | Desbalanceamento máximo |
| `MIN_LIQUIDITY` | $100 | Liquidez mínima no book |
| `MAX_SLIPPAGE` | 1% | Slippage máximo aceitável |

---

## 🧮 Algoritmo de Balanceamento

### Objetivo
Manter `shares_UP ≈ shares_DOWN` para maximizar o payout garantido.

### Cálculo

```python
def calcular_compra_balanceada(shares_up, shares_down, preco_up, preco_down, budget):
    """
    Calcula quanto comprar de cada lado mantendo equilíbrio.
    """
    # Diferença atual
    diff = shares_up - shares_down

    if abs(diff) < 5:  # Já balanceado
        # Comprar igual dos dois lados
        custo_par = preco_up + preco_down
        pares = budget / custo_par
        return {
            'buy_up': pares,
            'buy_down': pares,
        }

    elif diff > 0:  # Mais UP que DOWN
        # Comprar mais DOWN para equilibrar
        return {
            'buy_up': 0,
            'buy_down': min(diff, budget / preco_down),
        }

    else:  # Mais DOWN que UP
        # Comprar mais UP para equilibrar
        return {
            'buy_up': min(abs(diff), budget / preco_up),
            'buy_down': 0,
        }
```

### Estratégia de Entrada Gradual

```
Minuto 0-2:   Aguardar (preços instáveis)
Minuto 2-5:   Entrada agressiva se spread > 3%
Minuto 5-10:  Entrada normal se spread > 2%
Minuto 10-14: Entrada conservadora se spread > 2.5%
Minuto 14-15: Não entrar (muito perto do fim)
```

---

## 📊 Análise de Risco

### Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Execução parcial | Médio | Alto | Ordens pequenas, rebalanceamento |
| Spread fecha rápido | Alto | Médio | Monitoramento constante |
| API fora do ar | Baixo | Alto | Retry com backoff |
| Slippage | Médio | Médio | Limite de slippage |
| Desbalanceamento | Médio | Alto | Algoritmo de balanceamento |

### Cenários de Perda

**1. Execução Parcial**
```
Comprou 100 UP @ $0.48 = $48.00
Não conseguiu comprar DOWN (preço subiu)

Se UP ganha: Recebe $100, Lucro = $52
Se DOWN ganha: Recebe $0, Prejuízo = -$48

Mitigação: Ordens pequenas, verificar execução
```

**2. Spread Negativo**
```
UP = $0.52, DOWN = $0.51
Total = $1.03 > $1.00

NÃO ENTRAR - prejuízo garantido
```

### Cálculo de Risco Máximo

```
Risco por mercado = MAX_PER_MARKET × MAX_IMBALANCE
Risco por mercado = $500 × 20% = $100

Pior caso: Perder $100 em um mercado
Lucro esperado: $500 × 2% = $10 por mercado

Risk/Reward: 10:1 contra, MAS...
Probabilidade de perda total: <1% (só se API falhar)
```

---

## 🔌 Integração com Polymarket

### APIs Necessárias

1. **CLOB API** - Order book e execução
   - `GET /book?token_id=XXX` - Order book
   - `POST /order` - Criar ordem
   - `GET /orders` - Listar ordens

2. **Gamma API** - Dados de mercado
   - `GET /markets` - Listar mercados
   - `GET /markets/{id}` - Detalhes do mercado

### Autenticação

```python
# Polymarket usa assinatura ECDSA
from py_clob_client.client import ClobClient

client = ClobClient(
    host="https://clob.polymarket.com",
    key=PRIVATE_KEY,
    chain_id=137,  # Polygon
)
```

### Rate Limits

| Endpoint | Limite | Intervalo |
|----------|--------|-----------|
| GET /book | 100/min | Por token |
| POST /order | 10/min | Por conta |
| GET /markets | 60/min | Global |

---

## 💰 Projeção de Lucro

### Cenário Conservador

```
Capital: $1,000
Spread médio: 2%
Mercados por dia com spread: 20
Utilização do capital: 50%

Lucro diário = $1,000 × 50% × 2% × 20 = $200
Lucro mensal = $200 × 30 = $6,000
ROI mensal = 600%
```

### Cenário Realista (baseado no Gabagool)

```
Gabagool:
- $450,000 lucro
- 15,182 trades
- Lucro/trade: $29.68

Se operarmos com $1,000:
- Trades menores (~$10-20 por trade)
- ~50-100 trades por dia
- Lucro diário estimado: $50-$150
```

### Cenário Pessimista

```
- Spreads menores (1%)
- Competição maior
- Menos oportunidades

Lucro diário: $20-$50
ROI mensal: 60-150%
```

---

## 🛠️ Arquitetura do Bot

```
┌─────────────────────────────────────────────────────────────┐
│                      GABAGOOL BOT                           │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  Market       │   │  Spread       │   │  Position     │
│  Scanner      │   │  Monitor      │   │  Manager      │
│               │   │               │   │               │
│  - Find       │   │  - Check      │   │  - Track      │
│    markets    │   │    prices     │   │    shares     │
│  - Filter     │   │  - Calculate  │   │  - Balance    │
│    active     │   │    spread     │   │  - Limits     │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Order Executor     │
                │                       │
                │  - Create orders      │
                │  - Track fills        │
                │  - Handle errors      │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    Risk Manager       │
                │                       │
                │  - Check limits       │
                │  - Stop losses        │
                │  - Report P&L         │
                └───────────────────────┘
```

---

## 📁 Estrutura de Arquivos

```
src/
├── gabagool/
│   ├── __init__.py
│   ├── bot.py              # Bot principal
│   ├── market_scanner.py   # Busca mercados
│   ├── spread_monitor.py   # Monitora spreads
│   ├── order_executor.py   # Executa ordens
│   ├── position_manager.py # Gerencia posições
│   ├── risk_manager.py     # Gerencia risco
│   └── config.py           # Configurações
│
├── backtest/
│   ├── spread_backtest.py  # Backtest da estratégia
│   └── analyzer.py         # Análise de resultados
│
└── utils/
    ├── polymarket_client.py # Cliente Polymarket
    └── logger.py            # Logging
```

---

## ✅ Checklist de Implementação

### Fase 1: Infraestrutura
- [ ] Cliente Polymarket API
- [ ] Autenticação ECDSA
- [ ] Sistema de logging
- [ ] Configuração Docker

### Fase 2: Core
- [ ] Market Scanner
- [ ] Spread Monitor
- [ ] Position Manager
- [ ] Order Executor

### Fase 3: Risco
- [ ] Risk Manager
- [ ] Limites de exposição
- [ ] Alertas

### Fase 4: Backtest
- [ ] Coletor de dados
- [ ] Engine de backtest
- [ ] Análise de resultados

### Fase 5: Deploy
- [ ] Testes em paper trading
- [ ] Deploy VPS
- [ ] Monitoramento

---

## 🚨 Avisos Importantes

1. **Não é conselho financeiro** - Use por sua conta e risco
2. **Teste primeiro** - Rode em paper trading antes de usar dinheiro real
3. **Comece pequeno** - Não coloque muito capital no início
4. **Monitore sempre** - Bots podem ter bugs
5. **API pode mudar** - Polymarket pode alterar APIs

---

## 📚 Referências

- Perfil Gabagool: https://polymarket.com/profile/gabagool22
- Polymarket CLOB Docs: https://docs.polymarket.com
- py-clob-client: https://github.com/Polymarket/py-clob-client

---

*Documento criado para planejamento. Não é conselho financeiro.*
*Versão: 1.0 | Data: 2025-12-31*
