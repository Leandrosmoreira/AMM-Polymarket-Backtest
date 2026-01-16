# AMM Polymarket Backtest

Sistema completo de **backtesting para estratégia Delta-Neutral** em mercados de 15 minutos do Polymarket, com modelo avançado de liquidez temporal (LTM).

---

## Índice

- [O que é este projeto?](#o-que-é-este-projeto)
- [Como funciona a estratégia](#como-funciona-a-estratégia)
- [Instalação](#instalação)
- [Início Rápido](#início-rápido)
- [Comandos Disponíveis](#comandos-disponíveis)
- [Configuração](#configuração)
- [LTM - Liquidity Time Model](#ltm---liquidity-time-model)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Exemplos de Uso](#exemplos-de-uso)
- [FAQ](#faq)

---

## O que é este projeto?

Este é um **framework de backtesting** para testar estratégias de arbitragem em mercados binários de 15 minutos do Polymarket (ex: "Solana vai subir ou descer nos próximos 15 min?").

### Principais funcionalidades:

- **Backtest completo** com dados históricos ou simulados
- **Análise de spreads** e oportunidades de arbitragem
- **Gestão de risco** configurável (exposure, position sizing, stop loss)
- **LTM (Liquidity Time Model)** - modelo que adapta o trading baseado no tempo restante do mercado
- **Métricas de performance** (Sharpe, drawdown, win rate, etc.)
- **Relatórios visuais** com gráficos

---

## Como funciona a estratégia

### Estratégia Delta-Neutral (Arbitragem)

Em mercados binários do Polymarket:
- **YES** = aposta que vai subir
- **NO** = aposta que vai descer
- No settlement, um paga **$1.00** e outro paga **$0.00**

**A oportunidade surge quando:**
```
Preço YES + Preço NO < $1.00
```

**Exemplo:**
```
YES = $0.48
NO  = $0.49
Total = $0.97 (spread de -$0.03)

Comprar 100 YES + 100 NO = $97.00
Settlement (qualquer resultado) = $100.00
Lucro garantido = $3.00 (3.1% ROI)
```

### Por que funciona?

O mercado nem sempre é eficiente. Às vezes a soma dos preços fica abaixo de $1.00, criando oportunidade de arbitragem sem risco direcional.

---

## Instalação

### Requisitos
- Python 3.9+
- pip

### Passo a passo

```bash
# 1. Clone o repositório
git clone https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest.git
cd AMM-Polymarket-Backtest

# 2. (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instale as dependências
pip install -r requirements.txt
```

### Dependências principais
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
httpx>=0.25.0
pyyaml>=6.0
tqdm>=4.65.0
pyarrow>=14.0.0
```

---

## Início Rápido

### Teste rápido (30 segundos)

```bash
# Rodar backtest com dados simulados
python main.py test --capital 5000
```

### Teste com LTM (recomendado)

```bash
# Backtest com Liquidity Time Model
python main.py ltm-test --capital 5000 --n-markets 200
```

Isso vai:
1. Gerar 200 mercados simulados
2. Rodar o backtest com e sem LTM
3. Mostrar comparação de performance

---

## Comandos Disponíveis

### Visão geral

```bash
python main.py --help
```

```
Comandos disponíveis:
  collect       Coletar dados da API Polymarket
  analyze       Analisar dados coletados
  backtest      Rodar backtest básico
  test          Teste rápido com dados simulados
  ltm-backtest  Backtest com LTM (Liquidity Time Model)
  ltm-test      Teste LTM com dados simulados
  build-policy  Gerar política LTM a partir de dados
```

---

### 1. `collect` - Coletar Dados

Coleta dados históricos de mercados da API do Polymarket.

```bash
# Coletar últimos 90 dias
python main.py collect --days 90

# Coletar com histórico de preços
python main.py collect --days 90 --fetch-prices
```

**Parâmetros:**
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--days` | 90 | Dias de histórico |
| `--fetch-prices` | False | Também baixar preços |

**Saída:**
- `data/raw/sol_markets.csv` - Lista de mercados
- `data/processed/all_prices.parquet` - Histórico de preços

---

### 2. `analyze` - Analisar Dados

Analisa os dados coletados e mostra estatísticas.

```bash
# Análise padrão
python main.py analyze

# Com arquivos específicos
python main.py analyze --markets data/raw/sol_markets.csv --prices data/processed/all_prices.parquet
```

**O que mostra:**
- Distribuição de outcomes (Up vs Down)
- Estatísticas de volume e liquidez
- Análise de spreads (média, mediana, percentis)
- Melhores horários para trading

---

### 3. `backtest` - Backtest Básico

Roda o backtest sem LTM (estratégia simples).

```bash
# Backtest básico
python main.py backtest --capital 5000

# Com parâmetros customizados
python main.py backtest \
  --capital 10000 \
  --spread-threshold -0.02 \
  --max-exposure 0.8 \
  --output results/meu_backtest
```

**Parâmetros:**
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--capital` | 5000 | Capital inicial em USD |
| `--spread-threshold` | -0.02 | Spread mínimo para entrar |
| `--max-exposure` | 0.70 | Exposição máxima do portfólio |
| `--output` | auto | Diretório de saída |

---

### 4. `test` - Teste Rápido

Teste rápido com dados simulados.

```bash
python main.py test --capital 5000
```

---

### 5. `ltm-backtest` - Backtest com LTM

Backtest avançado usando o Liquidity Time Model.

```bash
# LTM básico
python main.py ltm-backtest --capital 5000

# LTM com bandit learning
python main.py ltm-backtest \
  --capital 10000 \
  --use-bandit \
  --policy config/ltm_policy.yaml
```

**Parâmetros:**
| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--capital` | 5000 | Capital inicial |
| `--policy` | config/ltm_policy.yaml | Arquivo de política LTM |
| `--use-decay` | True | Usar modelo de decay |
| `--use-bandit` | False | Usar auto-tuning por bandit |

---

### 6. `ltm-test` - Teste LTM

Teste do LTM com dados simulados + comparação.

```bash
# Teste padrão
python main.py ltm-test --capital 5000

# Com mais mercados e bandit
python main.py ltm-test --capital 10000 --n-markets 500 --use-bandit
```

**Saída esperada:**
```
============================================================
COMPARISON: LTM vs Base Backtest
============================================================
  LTM Return: 12.45%
  Base Return: 8.23%
  Improvement: 4.22%
  LTM Trades: 87
  Base Trades: 124
============================================================
```

---

### 7. `build-policy` - Gerar Política LTM

```bash
# A partir de simulação
python main.py build-policy --simulate --output config/ltm_policy.yaml

# A partir de dados reais
python main.py build-policy --data data/ltm_snapshots.parquet
```

---

## Configuração

### Parâmetros de Risco (`config/risk_params.py`)

```python
class RiskParams:
    # === SPREAD ===
    MIN_SPREAD_TO_ENTER = -0.02    # Só entra se YES + NO < 0.98
    TARGET_SPREAD = -0.03          # Spread ideal

    # === POSIÇÃO ===
    MAX_PER_MARKET_PCT = 0.15      # Máximo 15% do capital por mercado
    MAX_PER_MARKET_USD = 750       # Máximo $750 por mercado
    MIN_ORDER_SIZE = 10            # Mínimo $10 por ordem

    # === EXPOSIÇÃO ===
    MAX_TOTAL_EXPOSURE = 0.70      # Máximo 70% alocado
    MAX_ACTIVE_MARKETS = 5         # Máximo 5 simultâneos

    # === TEMPO ===
    MIN_TIME_REMAINING = 120       # Mínimo 2 min antes do fim

    # === STOP LOSS ===
    STOP_LOSS_PCT = 0.10           # Stop de 10%
```

---

## LTM - Liquidity Time Model

O LTM adapta o comportamento baseado no **tempo restante** do mercado.

### Fases do Mercado (15 buckets de 60s)

| Bucket | Tempo Restante | Fase | Comportamento |
|--------|----------------|------|---------------|
| 0-4 | 600-900s | EARLY | Conservador |
| 5-10 | 240-600s | MIDDLE | Agressivo |
| 11-12 | 120-240s | LATE | Conservador |
| 13-14 | 0-120s | FINAL | Stop trading |

### Por que LTM?

- **Início**: Liquidez ainda se formando
- **Meio**: Melhor liquidez e spreads
- **Final**: Risco de não conseguir fechar posição

### Componentes

| Módulo | Descrição |
|--------|-----------|
| `collector.py` | Coleta snapshots por bucket |
| `features.py` | Estatísticas por bucket |
| `policy.py` | Regras por bucket |
| `decay.py` | Modelo de velocidade de melhora do spread |
| `bandit.py` | Auto-tuning dos parâmetros |

---

## Estrutura do Projeto

```
AMM-Polymarket-Backtest/
├── config/
│   ├── settings.py             # Configurações globais
│   ├── risk_params.py          # Parâmetros de risco
│   └── ltm_policy.yaml         # Política LTM
├── src/
│   ├── backtest_engine.py      # Engine básico
│   ├── position_manager.py     # Portfolio, Position
│   ├── risk_manager.py         # Gestão de risco
│   ├── data_collector.py       # Coleta dados
│   ├── metrics.py              # Métricas
│   ├── visualizer.py           # Gráficos
│   ├── ltm/                    # Módulo LTM
│   │   ├── collector.py
│   │   ├── features.py
│   │   ├── policy.py
│   │   ├── decay.py
│   │   └── bandit.py
│   ├── ltm_risk_manager.py
│   └── ltm_backtest_engine.py
├── scripts/
│   └── build_ltm_policy.py
├── tests/
│   ├── test_backtest.py
│   └── test_ltm.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── main.py
└── requirements.txt
```

---

## Exemplos de Uso

### Teste Rápido
```bash
python main.py test --capital 5000
```

### LTM Completo
```bash
python main.py build-policy --simulate
python main.py ltm-test --capital 10000 --n-markets 500
```

### Pipeline com Dados Reais
```bash
python main.py collect --days 30 --fetch-prices
python main.py analyze
python main.py ltm-backtest --capital 5000 --use-bandit
```

### Uso Programático
```python
from src import LTMBacktestEngine
import pandas as pd

markets_df = pd.read_csv('data/raw/sol_markets.csv')
prices_df = pd.read_parquet('data/processed/all_prices.parquet')

engine = LTMBacktestEngine(initial_capital=10000, use_bandit=True)
results = engine.run(markets_df, prices_df)

print(f"Return: {results['summary']['total_return_pct']:.2f}%")
```

---

## FAQ

**Preciso de dados reais?**
Não! Use `python main.py ltm-test` para dados simulados.

**LTM ou backtest normal?**
LTM é recomendado - adapta ao tempo restante.

**Posso usar para trading real?**
Este é só backtest. Para trading real, veja [terauss/Polymarket-trading-bot-15min-BTC](https://github.com/terauss/Polymarket-trading-bot-15min-BTC).

---

## Métricas

| Métrica | Descrição |
|---------|-----------|
| Total Return | Retorno total % |
| Sharpe Ratio | Retorno ajustado ao risco |
| Max Drawdown | Maior queda |
| Win Rate | % trades lucrativos |
| Profit Factor | Lucros / Perdas |

---

## Licença

MIT License

---

**Feito com Claude + Leandro - Janeiro 2025**
