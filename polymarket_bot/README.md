# Polymarket Multi-Market Arbitrage Bot

Bot de arbitragem para mercados 15min do Polymarket, monitorando **BTC, ETH e SOL** simultaneamente.

## Features

- **Multi-mercado**: Monitora BTC, ETH, SOL ao mesmo tempo
- **Rollover automático**: Detecta quando mercado fecha e busca o próximo
- **Alta performance**: uvloop + orjson + JSONL logging
- **Autenticação funcional**: Baseado no padrão `exemplo_polymarket`

## Instalação Rápida

```bash
# 1. Instalar dependências (com otimizações)
pip install py-clob-client python-dotenv httpx uvloop orjson

# 2. Copiar template de configuração
cp polymarket_bot/pmpe.env.template polymarket_bot/pmpe.env

# 3. Editar pmpe.env com suas credenciais
#    PRIVATE_KEY=0x...
#    FUNDER_ADDRESS=0x...

# 4. Testar autenticação
python -m polymarket_bot --test-auth

# 5. Executar bot (simula os 3 mercados)
python -m polymarket_bot
```

## Uso

```bash
# Multi-mercado (BTC + ETH + SOL) - PADRÃO
python -m polymarket_bot

# Mercados específicos
python -m polymarket_bot btc eth      # Só BTC e ETH
python -m polymarket_bot sol          # Só SOL

# Modo single-market (só BTC, compatibilidade)
python -m polymarket_bot --single

# Testar autenticação
python -m polymarket_bot --test-auth
```

## Configuração (pmpe.env)

```env
# Autenticação
PRIVATE_KEY=0x_sua_chave_privada
FUNDER_ADDRESS=0x_seu_proxy_wallet
SIGNATURE_TYPE=1

# Mercados (BTC, ETH, SOL)
ASSETS=btc,eth,sol

# Trading
TARGET_PAIR_COST=0.991
ORDER_SIZE=5
ORDER_TYPE=FOK
COOLDOWN_SECONDS=10

# Modo
DRY_RUN=true
SIM_BALANCE=100
```

## Como Funciona

### Mercados 15min

Cada mercado abre e fecha a cada 15 minutos com novos tokens:

```
12:00:00 - Abre btc-updown-15m-1737374400 (YES/NO tokens)
12:15:00 - Fecha btc-updown-15m-1737374400
12:15:00 - Abre btc-updown-15m-1737375300 (NOVOS tokens)
...
```

O bot detecta automaticamente quando um mercado fecha e busca o próximo.

### Estratégia de Arbitragem

1. Monitorar preços de YES (UP) e NO (DOWN) para cada ativo
2. Quando `preço_UP + preço_DOWN < $0.991`:
   - Comprar ambos os lados
   - Lucro garantido: `$1.00 - custo_total`

Exemplo:
```
BTC UP:   $0.48
BTC DOWN: $0.50
Total:    $0.98
Lucro:    $0.02 por par (2%)
```

## Performance

O bot usa várias otimizações:

| Otimização | Ganho |
|------------|-------|
| uvloop | 2-4x async mais rápido |
| orjson | 10x JSON mais rápido |
| JSONL logging | Non-blocking, buffered |
| Parallel scans | Todos mercados simultaneamente |

### Instalar todas otimizações:
```bash
pip install uvloop orjson msgspec httpx[http2]
```

## Logs

Os logs são salvos em formato JSONL (JSON Lines) para análise:

```
logs/
├── trades_20260120_123456.jsonl   # Trades executados
└── scans_20260120_123456.jsonl    # Scans do mercado
```

Cada linha é um JSON:
```json
{"ts":1737340800,"market":"btc-updown-15m-1737340800","price_up":0.48,"price_down":0.50,"pair_cost":0.98,"profit_pct":2.04}
```

## Estrutura

```
polymarket_bot/
├── __main__.py       # Entry point
├── auth.py           # Autenticação (padrão exemplo_polymarket)
├── config.py         # Configurações
├── markets.py        # Descoberta multi-mercado
├── multi_bot.py      # Bot multi-mercado (BTC+ETH+SOL)
├── bot.py            # Bot single-market (só BTC)
├── trading.py        # Funções de trading
├── performance.py    # uvloop + orjson
├── fast_logger.py    # JSONL logging
└── pmpe.env.template # Template de configuração
```

## Troubleshooting

### "No active market found"

Os mercados 15min só existem durante horários de trading. Tente:
1. Verificar se Polymarket está ativo
2. Acessar https://polymarket.com/crypto/15M manualmente

### "Invalid signature"

1. Verifique `FUNDER_ADDRESS` (deve ser o proxy wallet)
2. Encontre em: https://polymarket.com/@SEU_USERNAME
3. Use `python -m polymarket_bot --test-auth` para testar

### Balance é 0

O saldo está no proxy wallet do Polymarket, não na blockchain diretamente.

## Aviso

Este bot é para fins educacionais. Trading envolve riscos. Use por sua conta e risco.
