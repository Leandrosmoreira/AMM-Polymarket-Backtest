# Comparação de Autenticação: Polymarket Projects

## Resumo Executivo

| Aspecto | exemplo_polymarket | AMM-Polymarket-Backtest (snayder_bot) |
|---------|-------------------|--------------------------------------|
| **Autenticação** | Funcionando | NÃO funcionando |
| **Bot Trading** | Scripts simples | Bot completo com LTM |
| **Problema** | - | Configuração de credenciais |

---

## 1. Projeto: exemplo_polymarket (AUTH FUNCIONANDO)

**Repositório:** https://github.com/Leandrosmoreira/exemplo_polymarket

### Estrutura de Autenticação

```
exemplo_polymarket/
├── client.py              # Implementação do cliente
├── test_auth.py           # Teste de autenticação L1/L2
├── empty.env              # Template de variáveis
├── pmpe.env               # Arquivo de credenciais (local)
├── get_balance_api.py     # Consulta de saldo
├── create_order_btc.py    # Criação de ordens
└── requirements.txt       # py-clob-client>=0.18.0
```

### Variáveis de Ambiente Necessárias (empty.env)

```env
WALLET_ADDRESS=           # Endereço da carteira original
FUNDER_ADDRESS=           # Endereço do proxy wallet Polymarket
PRIVATE_KEY=              # Chave privada exportada
TOKEN_ID=                 # Token do mercado (opcional)
```

### Fluxo de Autenticação (Funcional)

```python
from py_clob_client.client import ClobClient
from dotenv import dotenv_values

# 1. Carregar credenciais do arquivo .env
config = dotenv_values("pmpe.env")
key = config["PRIVATE_KEY"]
address = config["FUNDER_ADDRESS"]

# 2. Criar cliente com signature_type=1 (Magic/Email)
client = ClobClient(
    host="https://clob.polymarket.com",
    key=key,
    chain_id=137,
    signature_type=1,        # CRÍTICO: Para contas Magic.link
    funder=address           # CRÍTICO: Proxy wallet address
)

# 3. Derivar e configurar credenciais API (L2)
credentials = client.create_or_derive_api_creds()
client.set_api_creds(credentials)

# Pronto para usar!
```

### Pontos Chave do Sucesso

1. **signature_type=1** - Obrigatório para contas criadas via Magic.link/Email
2. **funder=proxy_wallet** - O endereço do proxy wallet (diferente do signer)
3. **Derivação automática** - Usa `create_or_derive_api_creds()` em vez de credenciais manuais
4. **Arquivo pmpe.env** - Credenciais reais (não versionadas)

---

## 2. Projeto: AMM-Polymarket-Backtest / snayder_bot (AUTH NÃO FUNCIONANDO)

**Repositório:** https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest
**Branch:** `claude/improve-trading-bot-aTbGG`

### Estrutura de Autenticação

```
trading_bot_ltm/
├── config.py              # Configurações via dataclass
├── trading.py             # Cliente e operações de trading
├── generate_api_key.py    # Gerador de API keys
├── test_balance.py        # Teste de saldo
├── diagnose_config.py     # Diagnóstico de configuração
├── .env.example           # Template
├── .env.paper             # Modo simulação
└── .env.live              # Credenciais reais (local)
```

### Variáveis de Ambiente (.env.example)

```env
POLYMARKET_API_KEY=your_api_key_here
POLYMARKET_API_SECRET=your_api_secret_here
POLYMARKET_API_PASSPHRASE=your_api_passphrase_here
POLYMARKET_PRIVATE_KEY=0x_your_private_key_here
POLYMARKET_SIGNATURE_TYPE=1
POLYMARKET_FUNDER=                    # <-- PROBLEMA: vazio!
```

### Fluxo de Autenticação Atual (trading.py)

```python
from py_clob_client.client import ClobClient

def get_client(settings: Settings):
    # Cria cliente
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=settings.private_key.strip(),
        chain_id=137,
        signature_type=settings.signature_type,      # OK
        funder=settings.funder.strip() if settings.funder else None  # PROBLEMA!
    )

    # Deriva credenciais
    derived_creds = client.create_or_derive_api_creds()
    client.set_api_creds(derived_creds)

    return client
```

### Problemas Identificados

| # | Problema | Impacto |
|---|----------|---------|
| 1 | `POLYMARKET_FUNDER` vazio no template | Cliente usa signer como funder |
| 2 | Falta validação de FUNDER != SIGNER | "Invalid signature" errors |
| 3 | requirements.txt não inclui py-clob-client | Dependência pode faltar |

---

## 3. Diagnóstico do Problema

### O que é `signature_type=1`?

Polymarket suporta dois tipos de assinatura:

| Type | Descrição | Quando Usar |
|------|-----------|-------------|
| 0 | EOA wallet (MetaMask direto) | Carteiras externas conectadas diretamente |
| 1 | Magic.link / Email wallet | Contas criadas via email no Polymarket |

### O que é o Funder (Proxy Wallet)?

Para contas **Magic.link/Email**:

```
[Sua Private Key] --> [Signer Address] --> DIFERENTE de --> [Funder/Proxy Wallet]
                           ↓                                        ↓
                     Quem assina                           Onde está o dinheiro
```

O **funder** é o endereço da carteira proxy criada pelo Polymarket para guardar seus fundos. Encontre em:
- https://polymarket.com/@SEU_USERNAME
- Copie o endereço exibido no perfil

### Erro Típico

```
Error: invalid signature
```

**Causa:** O `funder` está vazio ou igual ao `signer address`.

---

## 4. Solução: Como Fazer Funcionar no snayder_bot

### Passo 1: Atualizar .env com credenciais corretas

```env
# Chave privada exportada das configurações do Polymarket
POLYMARKET_PRIVATE_KEY=0x_sua_chave_privada_aqui

# OBRIGATÓRIO para Magic.link/Email accounts
POLYMARKET_SIGNATURE_TYPE=1

# CRÍTICO: Endereço do PROXY WALLET (encontre em polymarket.com/@seu_usuario)
POLYMARKET_FUNDER=0x_seu_proxy_wallet_address_aqui

# Opcionais (derivados automaticamente se vazios):
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_API_PASSPHRASE=
```

### Passo 2: Verificar que FUNDER != SIGNER

Execute:
```bash
cd trading_bot_ltm
python -m trading_bot_ltm.diagnose_config
```

Deve mostrar:
```
✓ POLYMARKET_FUNDER is set to a different address (good)
```

### Passo 3: Testar autenticação

```bash
python -m trading_bot_ltm.test_balance
```

Deve mostrar:
```
✓ Client created
✓ API Key: xxxxx
✓ Credentials configured
💰 BALANCE USDC: $XX.XX
```

### Passo 4: Adicionar py-clob-client ao requirements.txt

```txt
# Polymarket client
py-clob-client>=0.18.0
py-order-utils>=0.0.21
```

---

## 5. Checklist de Configuração

### Para Auth Funcionar:

- [ ] **POLYMARKET_PRIVATE_KEY** - Chave exportada do Polymarket settings
- [ ] **POLYMARKET_SIGNATURE_TYPE=1** - Para contas Magic.link/Email
- [ ] **POLYMARKET_FUNDER** - Endereço do proxy wallet (NÃO o signer)
- [ ] Verificar FUNDER != SIGNER com `diagnose_config.py`
- [ ] Testar conexão com `test_balance.py`
- [ ] Verificar saldo na conta Polymarket

### Para Bot Funcionar:

- [ ] Auth funcionando (acima)
- [ ] `POLYMARKET_MARKET_SLUG` ou `YES_TOKEN_ID`/`NO_TOKEN_ID` configurados
- [ ] `DRY_RUN=false` para trading real
- [ ] `ORDER_SIZE` configurado
- [ ] Saldo USDC suficiente na conta

---

## 6. Comparação de Código

### exemplo_polymarket (Funciona)

```python
# client.py - Abordagem simples e direta
config = dotenv_values("pmpe.env")
key = config["PRIVATE_KEY"]
address = config["FUNDER_ADDRESS"]  # ✓ Usa funder separado

client = ClobClient(
    host=host,
    key=key,
    chain_id=137,
    signature_type=1,      # ✓ Hardcoded para Magic
    funder=address         # ✓ Sempre configurado
)
```

### snayder_bot (Precisa Correção)

```python
# trading.py - Funder pode ser None
client = ClobClient(
    host,
    key=settings.private_key.strip(),
    chain_id=137,
    signature_type=settings.signature_type,
    funder=settings.funder.strip() if settings.funder else None  # ⚠️ Pode ser None!
)
```

**Problema:** Se `settings.funder` estiver vazio, passa `None` e a autenticação falha.

---

## 7. Recomendações

### Correção Imediata

1. Preencher `POLYMARKET_FUNDER` no arquivo `.env` com o endereço correto do proxy wallet
2. Garantir que seja DIFERENTE do signer address

### Melhoria de Código (Opcional)

Adicionar validação em `trading.py`:

```python
def get_client(settings: Settings):
    if settings.signature_type == 1 and not settings.funder:
        raise RuntimeError(
            "POLYMARKET_FUNDER é obrigatório para signature_type=1 (Magic.link). "
            "Configure com o endereço do seu proxy wallet do Polymarket."
        )
    # ... resto do código
```

---

## 8. Onde Encontrar as Credenciais

### Private Key

1. Acesse https://polymarket.com
2. Clique no perfil → Settings → Export Private Key
3. Copie a chave (começa com 0x...)

### Proxy Wallet (Funder) Address

1. Acesse https://polymarket.com/@SEU_USERNAME
2. O endereço exibido abaixo do nome é o proxy wallet
3. Copie este endereço para `POLYMARKET_FUNDER`

### API Credentials (Opcional)

As credenciais API são **derivadas automaticamente** da private key usando `create_or_derive_api_creds()`. Não é necessário gerar manualmente.

---

## Conclusão

| Projeto | Status | Ação Necessária |
|---------|--------|-----------------|
| exemplo_polymarket | Auth OK | Nenhuma - funciona corretamente |
| snayder_bot | Auth FALHA | Configurar `POLYMARKET_FUNDER` |

O bot do snayder_bot está **tecnicamente correto**, mas a configuração das credenciais está **incompleta**. Basta preencher o `POLYMARKET_FUNDER` com o endereço correto do proxy wallet para a autenticação funcionar.
