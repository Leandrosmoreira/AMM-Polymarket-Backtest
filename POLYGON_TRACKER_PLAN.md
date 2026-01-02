# Polygon Transaction Tracker - Gabagool Polymarket

## Objetivo
Criar um site para rastrear em tempo real todas as transações que a carteira do Gabagool faz na Polymarket, com sincronização manual e filtros por mercado/token.

---

## Informações Base

| Item | Valor |
|------|-------|
| **Carteira Gabagool** | Extrair de: `0x86c6ea5c52f560f28fcc9c4c77fc02da8defb9358d93da7915ca864e1be80239` |
| **Perfil Polymarket** | https://polymarket.com/@gabagool22 |
| **Blockchain** | Polygon (MATIC) |
| **CTF Exchange** | `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` |
| **NegRisk CTF Exchange** | `0xC5d563A36AE78145C45a50134d48A1215220f80a` |

---

## Arquitetura

```
polygon-tracker/
├── app.py                 # Backend Flask
├── requirements.txt       # Dependências Python
├── templates/
│   └── index.html         # Frontend principal
├── static/
│   ├── css/
│   │   └── style.css      # Estilos
│   └── js/
│       └── app.js         # Lógica frontend
└── config.py              # Configurações (API keys)
```

---

## Funcionalidades

### 1. Dashboard Principal
- [x] Lista de transações em tabela
- [x] Botão SYNC para atualizar manualmente
- [x] Indicador de última atualização
- [x] Total de transações

### 2. Filtros
- [ ] Por mercado (dropdown com mercados ativos)
- [ ] Por tipo de token (YES/NO)
- [ ] Por data (range picker)
- [ ] Por valor (min/max)
- [ ] Por tipo de operação (Buy/Sell)

### 3. Dados por Transação
- Hash da transação (link para Polygonscan)
- Data/hora
- Mercado (nome legível)
- Token (YES/NO)
- Quantidade
- Preço
- Valor total em USDC
- Tipo (Buy/Sell)

### 4. Estatísticas
- Volume total
- Número de trades
- Mercados únicos
- P&L estimado (se possível)

---

## APIs Utilizadas

### 1. Polygonscan API (Gratuita)
```
Base URL: https://api.polygonscan.com/api
```

**Endpoints:**
- `module=account&action=txlist` - Lista transações normais
- `module=account&action=tokentx` - Lista transferências de tokens ERC20/ERC1155

**Rate Limit:** 5 calls/segundo (free tier)

### 2. Polymarket API
```
Base URL: https://clob.polymarket.com
Base URL: https://gamma-api.polymarket.com
```

**Endpoints:**
- `/markets` - Lista de mercados
- `/markets/{condition_id}` - Detalhes do mercado

---

## Stack Técnica

| Componente | Tecnologia |
|------------|------------|
| Backend | Python + Flask |
| Frontend | HTML + Vanilla JS + CSS |
| API Calls | requests / aiohttp |
| Styling | CSS Grid/Flexbox |
| Data Format | JSON |

---

## Fluxo de Dados

```
1. Usuário clica SYNC
        ↓
2. Frontend faz POST /api/sync
        ↓
3. Backend busca transações via Polygonscan API
        ↓
4. Backend filtra transações relacionadas ao CTF Exchange
        ↓
5. Backend decodifica dados das transações
        ↓
6. Backend enriquece com nomes dos mercados (Polymarket API)
        ↓
7. Retorna JSON para frontend
        ↓
8. Frontend renderiza tabela com filtros
```

---

## Implementação - Fases

### Fase 1: MVP (Core)
1. Extrair endereço da carteira da transação exemplo
2. Configurar backend Flask básico
3. Integrar Polygonscan API para buscar transações
4. Criar frontend simples com tabela
5. Botão SYNC funcional

### Fase 2: Enriquecimento
1. Decodificar dados das transações (ABI do CTF Exchange)
2. Integrar Polymarket API para nomes dos mercados
3. Calcular valores em USDC

### Fase 3: Filtros e UX
1. Implementar filtros no frontend
2. Adicionar estatísticas
3. Melhorar UI/UX
4. Adicionar loading states

### Fase 4: Otimizações
1. Cache de mercados
2. Paginação
3. Export para CSV
4. Auto-refresh opcional

---

## Configuração Necessária

### Polygonscan API Key
1. Criar conta em https://polygonscan.com/register
2. Gerar API key em https://polygonscan.com/myapikey
3. Adicionar ao `config.py`

### Variáveis de Ambiente
```bash
POLYGONSCAN_API_KEY=your_api_key_here
GABAGOOL_WALLET=0x...  # Endereço extraído
```

---

## Exemplo de Resposta da API

```json
{
  "transactions": [
    {
      "hash": "0x86c6ea5c...",
      "timestamp": "2024-01-15T14:30:00Z",
      "market": "Will BTC reach $100k by March?",
      "token": "YES",
      "side": "BUY",
      "amount": 500,
      "price": 0.45,
      "total_usdc": 225.00,
      "polygonscan_url": "https://polygonscan.com/tx/0x..."
    }
  ],
  "stats": {
    "total_transactions": 150,
    "total_volume_usdc": 50000,
    "unique_markets": 25,
    "last_sync": "2024-01-15T14:35:00Z"
  }
}
```

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Rate limit Polygonscan | Cache local, batching requests |
| Polymarket API indisponível | Fallback para condition_id como nome |
| Transações complexas | Log de erros, skip graceful |
| Mudanças na ABI | Versionamento, alertas |

---

## Próximos Passos

1. **AGORA**: Extrair endereço da carteira Gabagool da transação
2. Implementar backend Flask
3. Criar frontend básico
4. Testar com dados reais
5. Iterar baseado em feedback

---

## Comandos para Rodar

```bash
# Instalar dependências
cd polygon-tracker
pip install -r requirements.txt

# Configurar API key
export POLYGONSCAN_API_KEY="sua_api_key"

# Rodar servidor
python app.py

# Acessar
open http://localhost:5000
```

---

*Plano criado em: Janeiro 2025*
