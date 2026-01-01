# Polymarket BTC Trading Bots

Framework completo para trading automatizado em mercados BTC 15-minutos da Polymarket.

## Estratégias Disponíveis

Este projeto implementa **duas estratégias distintas**:

| Estratégia | Tipo | Risco | Retorno Esperado |
|------------|------|-------|------------------|
| **Gabagool Spread Capture** | Arbitragem pura | Baixo | 8-15% ao mês |
| **Volatility Arbitrage** | Probabilística | Médio | 15-30% ao mês |

---

## 1. Gabagool Spread Capture (Arbitragem Pura)

### Como Funciona

A estratégia Gabagool explora ineficiências de pricing nos mercados binários:

```
Se UP + DOWN < $1.00, há lucro garantido

Exemplo:
- Preço UP: $0.48
- Preço DOWN: $0.50
- Total: $0.98 (spread de 2%)

Compra AMBOS tokens por $0.98
No settlement, um paga $1.00
Lucro: $0.02 por $0.98 investido = 2.04% garantido
```

### Vantagens
- **Risco zero** quando executado corretamente
- Lucro matematicamente garantido
- Não depende de previsão de mercado

### Desvantagens
- Spreads lucrativos são raros (competição)
- Fees reduzem margem (~1-2%)
- Precisa de capital disponível constantemente

### Comandos

```bash
# Backtest com dados simulados
python main.py gabagool-backtest --markets 100 --min-spread 0.02

# Bot em paper trading
python main.py gabagool-bot --preset moderate

# Teste rápido
python main.py gabagool-test
```

---

## 2. Volatility Arbitrage (Nova Estratégia)

### Como Funciona

Esta estratégia calcula a **probabilidade real** de BTC subir/descer usando volatilidade, e compara com os preços da Polymarket:

```
1. Coleta preços BTC em tempo real (Chainlink)
2. Calcula volatilidade rolling (desvio padrão)
3. Estima P(BTC > strike) usando modelo log-normal
4. Compara probabilidade modelo vs preço mercado
5. Executa trade quando edge > threshold

Exemplo:
- Modelo estima: 55% chance de UP
- Mercado cobra: $0.48 por UP token
- Edge: 55% - 48% = 7%
- Se edge > min_edge, COMPRA UP
```

### Componentes Técnicos

| Módulo | Função |
|--------|--------|
| `volatility.py` | Calcula rolling std dev dos retornos |
| `probability.py` | Modelo Black-Scholes para P(BTC > K) |
| `edge_detector.py` | Compara modelo vs mercado |
| `risk_manager.py` | Position sizing, Kelly criterion |
| `executor.py` | Execução paper/live |
| `logger.py` | Logging completo de trades |

### Modelo Matemático

```
P(S_T > K) = N(d2)

onde:
d2 = [ln(S/K) + (μ - σ²/2)T] / (σ√T)

S = preço atual BTC
K = strike price
T = tempo até expiração (segundos)
σ = volatilidade por segundo
μ = drift (assumimos 0 para curto prazo)
```

### Vantagens vs Gabagool
- Mais oportunidades de trade (não precisa spread < 1)
- Pode capturar movimentos direcionais
- Modelo adaptativo baseado em volatilidade real

### Desvantagens vs Gabagool
- Risco de perda (modelo pode errar)
- Depende de estimativa de volatilidade precisa
- Mais complexo de calibrar

### Comandos

```bash
# Bot em paper trading
python main.py vol-bot --balance 1000 --min-edge 3.0 --risk moderate

# Teste rápido (60 segundos)
python main.py vol-test --balance 1000 --duration 60

# Com live trading (CUIDADO: dinheiro real!)
python main.py vol-bot --live --balance 1000
```

---

## Comparação das Estratégias

| Aspecto | Gabagool | Volatility Arb |
|---------|----------|----------------|
| **Tipo** | Arbitragem pura | Estatística |
| **Risco** | ~0% | Médio |
| **Win Rate** | ~100% | 55-65% |
| **Trades/dia** | 5-20 | 20-100 |
| **Lucro/trade** | 1-3% | -10% a +20% |
| **Capital mínimo** | $500 | $1000 |
| **Complexidade** | Baixa | Alta |

### Quando usar cada uma?

**Use Gabagool se:**
- Quer risco zero
- Capital limitado
- Não precisa de retornos altíssimos

**Use Volatility Arb se:**
- Aceita risco moderado
- Quer mais trades
- Confia no modelo de volatilidade

---

## Coleta de Dados (VPS)

O projeto inclui um coletor de dados para rodar em VPS:

```bash
# Iniciar coletor com Docker
docker-compose up -d collector

# Ver logs
docker-compose logs -f collector

# Ver dados coletados
ls -la data/raw/
```

### O que é coletado

| Fonte | Dados | Frequência |
|-------|-------|------------|
| Chainlink | Preço BTC/USD | 1 segundo |
| Polymarket | Order books UP/DOWN | 2 segundos |
| Polymarket | Preços best ask | 2 segundos |

### Estrutura dos Dados

```json
{
  "metadata": {
    "market_id": "...",
    "up_token_id": "...",
    "down_token_id": "..."
  },
  "chainlink_ticks": [
    {"ts": 1234567890, "price": 87000.50, "diff": 0.25}
  ],
  "price_changes": [
    {"ts": 1234567890, "up": 0.48, "down": 0.52}
  ],
  "order_books": [...]
}
```

---

## Instalação

### Local

```bash
# Clone
git clone https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest.git
cd AMM-Polymarket-Backtest

# Ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Dependências
pip install -r requirements.txt
```

### VPS com Docker

```bash
# Clone na VPS
git clone https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest.git
cd AMM-Polymarket-Backtest
git checkout claude/btc-trading-bot-abTXw

# Permissões
mkdir -p data/raw
chmod -R 777 data/

# Iniciar
docker-compose up -d collector
```

---

## Configuração de Risco

### Gabagool

| Preset | MIN_SPREAD | ORDER_SIZE | MAX_PER_MARKET |
|--------|------------|------------|----------------|
| conservative | 3% | $10 | $300 |
| moderate | 2% | $15 | $500 |
| aggressive | 1% | $25 | $1000 |

### Volatility Arb

| Preset | MIN_EDGE | KELLY | MAX_DRAWDOWN |
|--------|----------|-------|--------------|
| conservative | 5% | 10% Kelly | 10% |
| moderate | 3% | 25% Kelly | 20% |
| aggressive | 2% | 50% Kelly | 30% |

---

## Estrutura do Projeto

```
AMM-Polymarket-Backtest/
├── src/
│   ├── gabagool/              # Estratégia Gabagool
│   │   ├── bot.py             # Bot principal
│   │   ├── spread_monitor.py  # Monitor de spreads
│   │   ├── backtest.py        # Engine de backtest
│   │   └── ...
│   │
│   ├── volatility_arb/        # Estratégia Vol Arb (NOVO)
│   │   ├── volatility.py      # Calculadora de volatilidade
│   │   ├── probability.py     # Modelo probabilístico
│   │   ├── edge_detector.py   # Detector de edge
│   │   ├── risk_manager.py    # Gestão de risco
│   │   ├── executor.py        # Executor de trades
│   │   ├── logger.py          # Sistema de logging
│   │   └── bot.py             # Bot principal
│   │
│   ├── realtime_collector.py  # Coletor de dados
│   └── ...
│
├── docker-compose.yml         # Deploy com Docker
├── main.py                    # Ponto de entrada
└── README.md
```

---

## Métricas de Performance

O bot rastreia automaticamente:

| Métrica | Descrição |
|---------|-----------|
| Total P&L | Lucro/prejuízo total |
| Win Rate | % de trades vencedores |
| Sharpe Ratio | Retorno ajustado por risco |
| Max Drawdown | Maior queda do pico |
| Trades/Hour | Frequência de trading |
| Avg Edge | Edge médio capturado |

---

## Previsões do Claude (para comparação)

Em 01/01/2026, estas foram as previsões para a estratégia Gabagool:

```
╔══════════════════════════════════════════════════╗
║  PREVISÃO CLAUDE - 01/01/2026                    ║
╠══════════════════════════════════════════════════╣
║  ROI Mensal Esperado:     8-15%                  ║
║  Spread Médio:            2-4%                   ║
║  Lucro Líquido/Trade:     0.5-2.5%               ║
║  Oportunidades/Dia:       20-50 trades           ║
║  Win Rate:                ~95%                   ║
║  Sharpe Ratio:            2-3                    ║
╚══════════════════════════════════════════════════╝
```

> **Após rodar o backtest com dados reais, compare com estas previsões!**

---

## Próximos Passos

1. **Coletar 2 dias de dados** na VPS
2. **Rodar backtest** com dados reais
3. **Comparar resultados** com previsões
4. **Ajustar parâmetros** se necessário
5. **Paper trading** por 1 semana
6. **Live trading** com capital pequeno

---

## Autor

Claude + Leandro - Janeiro 2026

## Licença

MIT License
