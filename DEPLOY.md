# Deploy BTC Trading Bot na VPS

Guia completo para deploy do bot de coleta e backtest em uma VPS Linux.

## Requisitos

- VPS com Ubuntu 20.04+ ou Debian 11+
- Mínimo 1GB RAM, 1 vCPU
- 10GB de disco (para logs)
- Acesso SSH

## VPS Recomendadas

| Provedor | Plano | Preço/mês |
|----------|-------|-----------|
| DigitalOcean | Basic Droplet 1GB | $6 |
| Vultr | Cloud Compute 1GB | $6 |
| Linode | Nanode 1GB | $5 |
| Hetzner | CX11 | €4.15 |

---

## 🚀 Deploy Rápido (5 minutos)

### 1. Conectar na VPS

```bash
ssh root@seu-ip-da-vps
```

### 2. Clonar o Repositório

```bash
git clone https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest.git
cd AMM-Polymarket-Backtest
git checkout claude/btc-trading-bot-abTXw
```

### 3. Rodar Setup Automático

```bash
chmod +x scripts/setup-vps.sh
./scripts/setup-vps.sh
```

### 4. Iniciar o Coletor

```bash
# Logout e login novamente (para permissões Docker)
exit
ssh root@seu-ip-da-vps
cd AMM-Polymarket-Backtest

# Iniciar coletor
docker-compose up -d collector
```

### 5. Verificar Status

```bash
# Ver logs em tempo real
docker-compose logs -f collector

# Ver status
docker-compose ps
```

---

## 📋 Comandos Úteis

### Gerenciar o Coletor

```bash
# Iniciar
docker-compose up -d collector

# Parar
docker-compose down

# Reiniciar
docker-compose restart collector

# Ver logs
docker-compose logs -f collector

# Ver últimas 100 linhas
docker-compose logs --tail=100 collector
```

### Rodar Backtest

```bash
# Com dados coletados
docker-compose --profile backtest run --rm backtest

# Ou diretamente
docker-compose run --rm backtest python main.py btc-backtest \
    --log-dir /app/data/raw \
    --capital 100
```

### Rodar Teste

```bash
docker-compose --profile test run --rm test
```

### Analisar Dados

```bash
docker-compose --profile analyze run --rm analyze
```

---

## 📁 Estrutura de Arquivos

```
AMM-Polymarket-Backtest/
├── data/
│   ├── raw/              # Dados coletados (JSON.gz)
│   ├── processed/        # Dados processados
│   └── results/          # Resultados do backtest
├── scripts/
│   ├── setup-vps.sh      # Setup automático
│   ├── start-collector.sh
│   ├── run-backtest.sh
│   └── run-test.sh
├── Dockerfile
├── docker-compose.yml
└── ...
```

---

## ⏰ Configurar Backtest Automático (Cron)

Para rodar backtest automaticamente todo dia às 00:00:

```bash
# Editar crontab
crontab -e

# Adicionar linha:
0 0 * * * cd /root/AMM-Polymarket-Backtest && docker-compose --profile backtest run --rm backtest >> /var/log/btc-backtest.log 2>&1
```

---

## 🔧 Configurações

### Alterar Intervalo de Salvamento

No `docker-compose.yml`, altere `--save-interval`:

```yaml
command: >
  python -m src.realtime_collector
  --output /app/data/raw
  --save-interval 600  # 10 minutos (em segundos)
```

### Alterar Capital do Backtest

```yaml
command: >
  python main.py btc-backtest
  --log-dir /app/data/raw
  --capital 200  # $200 por mercado
```

---

## 📊 Monitoramento

### Ver Espaço em Disco

```bash
df -h
du -sh data/raw/*
```

### Limpar Logs Antigos (manter últimos 7 dias)

```bash
find data/raw -name "*.json.gz" -mtime +7 -delete
```

### Ver Uso de Memória

```bash
docker stats btc-collector
```

---

## 🔒 Segurança

### Criar Usuário Não-Root

```bash
# Criar usuário
adduser botuser
usermod -aG docker botuser
usermod -aG sudo botuser

# Mudar para o usuário
su - botuser
```

### Configurar Firewall

```bash
# Instalar UFW
apt install ufw

# Configurar
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw enable
```

---

## 🐛 Troubleshooting

### Coletor não inicia

```bash
# Ver logs de erro
docker-compose logs collector

# Rebuildar imagem
docker-compose build --no-cache collector
docker-compose up -d collector
```

### Sem dados sendo coletados

```bash
# Testar conexão
docker-compose run --rm collector python -c "
import httpx
r = httpx.get('https://clob.polymarket.com/health')
print(r.status_code)
"
```

### Docker sem permissão

```bash
sudo usermod -aG docker $USER
# Logout e login novamente
```

### Limpar tudo e recomeçar

```bash
docker-compose down -v
docker system prune -a
docker-compose up -d --build collector
```

---

## 📈 Próximos Passos

1. **Coletar dados por 24-48h** antes de rodar backtest
2. **Analisar resultados** com `btc-analyze`
3. **Ajustar parâmetros** baseado nos resultados
4. **Paper trading** antes de usar dinheiro real

---

## 📞 Suporte

- Issues: https://github.com/Leandrosmoreira/AMM-Polymarket-Backtest/issues
- Branch: `claude/btc-trading-bot-abTXw`
