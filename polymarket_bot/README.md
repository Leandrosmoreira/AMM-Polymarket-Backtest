# Polymarket Bot

Bot de arbitragem para mercados BTC 15min do Polymarket, combinando:
- **Autenticação funcional** do projeto `exemplo_polymarket`
- **Bot completo** do projeto `trading_bot_ltm`

## Instalação Rápida

```bash
# 1. Instalar dependências
pip install py-clob-client python-dotenv httpx

# 2. Copiar template de configuração
cp polymarket_bot/pmpe.env.template polymarket_bot/pmpe.env

# 3. Editar pmpe.env com suas credenciais
# (veja seção "Configuração" abaixo)

# 4. Testar autenticação
python -m polymarket_bot.test_auth

# 5. Executar bot (modo simulação)
python -m polymarket_bot
```

## Configuração

### Credenciais Necessárias

Edite o arquivo `pmpe.env` com:

```env
# Sua chave privada (exportar de Polymarket Settings)
PRIVATE_KEY=0x_sua_chave_privada_aqui

# Endereço do seu proxy wallet (encontrar no perfil)
FUNDER_ADDRESS=0x_seu_proxy_wallet_aqui

# Tipo de assinatura (1 para contas Magic.link/Email)
SIGNATURE_TYPE=1
```

### Onde Encontrar as Credenciais

#### PRIVATE_KEY
1. Acesse https://polymarket.com
2. Vá em Settings (configurações)
3. Clique em "Export Private Key"
4. Copie a chave completa (inclui `0x`)

#### FUNDER_ADDRESS (Proxy Wallet)
1. Acesse https://polymarket.com/@SEU_USERNAME
2. O endereço exibido abaixo do seu nome é o proxy wallet
3. Copie este endereço

**IMPORTANTE**: O FUNDER_ADDRESS deve ser DIFERENTE do endereço derivado da sua private key!

## Uso

### Testar Autenticação
```bash
python -m polymarket_bot.test_auth
```

### Executar Bot (Simulação)
```bash
python -m polymarket_bot
```

### Executar Bot (Trading Real)
```bash
# Primeiro, edite pmpe.env e mude:
# DRY_RUN=false

python -m polymarket_bot
```

## Estrutura do Projeto

```
polymarket_bot/
├── auth.py              # Autenticação (padrão exemplo_polymarket)
├── config.py            # Configurações
├── trading.py           # Funções de trading
├── bot.py               # Bot de arbitragem
├── test_auth.py         # Teste de autenticação
├── pmpe.env.template    # Template de configuração
├── ltm/                 # Liquidity Time Model
│   ├── bandit.py
│   ├── collector.py
│   ├── decay.py
│   ├── features.py
│   └── policy.py
└── mm/                  # Market Making
    ├── delta_hedge.py
    ├── order_manager.py
    └── volatility.py
```

## Estratégia de Arbitragem

O bot implementa a estratégia de Jeremy Whittaker:

1. Monitorar mercados BTC 15min
2. Quando `preço_UP + preço_DOWN < $0.991`:
   - Comprar ambos os lados
   - Lucro garantido: `$1.00 - custo_total`

Exemplo:
- UP: $0.48, DOWN: $0.50
- Total: $0.98
- Lucro: $0.02 por par (2%)

## Parâmetros de Trading

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| TARGET_PAIR_COST | 0.991 | Threshold para arbitragem |
| ORDER_SIZE | 5 | Quantidade por ordem |
| ORDER_TYPE | FOK | Fill-or-Kill |
| COOLDOWN_SECONDS | 10 | Intervalo entre trades |
| DRY_RUN | true | Modo simulação |

## Troubleshooting

### "Invalid signature" error

1. Verifique se `FUNDER_ADDRESS` está correto
2. O funder deve ser o endereço do **proxy wallet**, não o signer
3. Para contas Magic.link/Email, use `SIGNATURE_TYPE=1`

### "Balance is 0"

1. Verifique se você está usando o proxy wallet correto
2. O saldo está na carteira do Polymarket, não na blockchain diretamente
3. Teste com: `python -m polymarket_bot.test_auth`

### Mercado não encontrado

O bot busca automaticamente mercados BTC 15min ativos. Se não encontrar:
1. Verifique conexão com internet
2. Em modo simulação, um mercado fake será usado

## Comparação com Projetos Originais

| Aspecto | exemplo_polymarket | trading_bot_ltm | polymarket_bot (este) |
|---------|-------------------|-----------------|----------------------|
| Auth | Simples, funciona | Complexo, falha | Simples, funciona |
| Bot | Scripts básicos | Completo com LTM | Completo com auth OK |
| Config | pmpe.env | .env múltiplo | pmpe.env (padrão) |

## Licença

Este projeto é para fins educacionais. Use por sua conta e risco.
