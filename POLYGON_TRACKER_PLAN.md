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
├── Dockerfile             # Container Docker
├── docker-compose.yml     # Orquestração
├── .env.example           # Exemplo de variáveis
├── nginx.conf             # Config Nginx (produção)
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

## Deploy com Docker em VPS Linux

### Arquivos Docker

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primeiro (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor porta
EXPOSE 5000

# Usar gunicorn em produção
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  tracker:
    build: .
    container_name: polygon-tracker
    restart: unless-stopped
    ports:
      - "5000:5000"
    environment:
      - POLYGONSCAN_API_KEY=${POLYGONSCAN_API_KEY}
      - GABAGOOL_WALLET=${GABAGOOL_WALLET}
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data  # Persistir cache
    networks:
      - tracker-net

  nginx:
    image: nginx:alpine
    container_name: tracker-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certbot/conf:/etc/letsencrypt:ro
      - ./certbot/www:/var/www/certbot:ro
    depends_on:
      - tracker
    networks:
      - tracker-net

  certbot:
    image: certbot/certbot
    container_name: tracker-certbot
    volumes:
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"

networks:
  tracker-net:
    driver: bridge
```

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream tracker {
        server tracker:5000;
    }

    server {
        listen 80;
        server_name seu-dominio.com;

        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        location / {
            return 301 https://$host$request_uri;
        }
    }

    server {
        listen 443 ssl;
        server_name seu-dominio.com;

        ssl_certificate /etc/letsencrypt/live/seu-dominio.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/seu-dominio.com/privkey.pem;

        location / {
            proxy_pass http://tracker;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### .env.example
```bash
# Polygonscan API
POLYGONSCAN_API_KEY=sua_api_key_aqui

# Carteira para rastrear
GABAGOOL_WALLET=0x...

# Flask
FLASK_ENV=production
SECRET_KEY=gerar_chave_segura_aqui
```

---

### Deploy na VPS - Passo a Passo

#### 1. Preparar VPS (Ubuntu/Debian)
```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Instalar Docker Compose
sudo apt install docker-compose-plugin -y

# Adicionar usuário ao grupo docker
sudo usermod -aG docker $USER
newgrp docker
```

#### 2. Clonar e Configurar
```bash
# Clonar repositório
git clone https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest.git
cd AMM-Polymarket-Backtest/polygon-tracker

# Criar arquivo .env
cp .env.example .env
nano .env  # Editar com suas credenciais
```

#### 3. Build e Iniciar
```bash
# Build da imagem
docker compose build

# Iniciar em background
docker compose up -d

# Ver logs
docker compose logs -f tracker
```

#### 4. Configurar SSL (HTTPS)
```bash
# Gerar certificado SSL com Let's Encrypt
docker compose run --rm certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    -d seu-dominio.com

# Reiniciar nginx
docker compose restart nginx
```

#### 5. Comandos Úteis
```bash
# Status dos containers
docker compose ps

# Reiniciar aplicação
docker compose restart tracker

# Parar tudo
docker compose down

# Atualizar para nova versão
git pull
docker compose build
docker compose up -d

# Ver logs em tempo real
docker compose logs -f

# Acessar shell do container
docker compose exec tracker /bin/bash
```

---

### Requisitos VPS

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| CPU | 1 vCPU | 2 vCPU |
| RAM | 1 GB | 2 GB |
| Disco | 10 GB | 20 GB SSD |
| SO | Ubuntu 20.04+ | Ubuntu 22.04 |

### Provedores Sugeridos

| Provedor | Plano | Preço Estimado |
|----------|-------|----------------|
| DigitalOcean | Basic Droplet | $6/mês |
| Vultr | Cloud Compute | $6/mês |
| Hetzner | CX11 | €4/mês |
| Contabo | VPS S | €5/mês |

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
