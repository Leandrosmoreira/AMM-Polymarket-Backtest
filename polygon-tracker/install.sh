#!/bin/bash

#############################################
# Polygon Tracker - Script de Instalação
# Gabagool Polymarket Transaction Monitor
#############################################

set -e

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções
print_header() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════╗"
    echo "║     Polygon Tracker - Instalador         ║"
    echo "║     Gabagool Polymarket Monitor          ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Header
print_header

# Verificar se está rodando como root
if [ "$EUID" -eq 0 ]; then
    print_warning "Rodando como root. Recomendado usar usuário normal com sudo."
fi

# 1. Verificar/Instalar Docker
echo ""
echo -e "${BLUE}[1/5] Verificando Docker...${NC}"

if command -v docker &> /dev/null; then
    print_step "Docker já instalado: $(docker --version)"
else
    print_warning "Docker não encontrado. Instalando..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    print_step "Docker instalado com sucesso!"
fi

# 2. Verificar Docker Compose
echo ""
echo -e "${BLUE}[2/5] Verificando Docker Compose...${NC}"

if docker compose version &> /dev/null; then
    print_step "Docker Compose disponível"
else
    print_warning "Instalando Docker Compose plugin..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
    print_step "Docker Compose instalado!"
fi

# 3. Configurar .env
echo ""
echo -e "${BLUE}[3/5] Configurando variáveis de ambiente...${NC}"

if [ -f .env ]; then
    print_warning "Arquivo .env já existe."
    read -p "Deseja sobrescrever? (s/N): " overwrite
    if [ "$overwrite" != "s" ] && [ "$overwrite" != "S" ]; then
        print_step "Mantendo .env existente"
    else
        create_env=true
    fi
else
    create_env=true
fi

if [ "$create_env" = true ]; then
    echo ""
    read -p "Digite sua API Key do Polygonscan: " api_key

    if [ -z "$api_key" ]; then
        print_error "API Key é obrigatória!"
        echo "Obtenha gratuitamente em: https://polygonscan.com/myapikey"
        exit 1
    fi

    read -p "Carteira para rastrear (Enter para Gabagool): " wallet
    wallet=${wallet:-0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d}

    cat > .env << EOF
# Polygonscan API Key
POLYGONSCAN_API_KEY=${api_key}

# Carteira para rastrear
GABAGOOL_WALLET=${wallet}

# Flask
FLASK_ENV=production
EOF

    print_step "Arquivo .env criado!"
fi

# 4. Build e Start
echo ""
echo -e "${BLUE}[4/5] Construindo e iniciando containers...${NC}"

# Parar containers existentes se houver
docker compose down 2>/dev/null || true

# Build
echo "Building..."
docker compose build --quiet

# Start
echo "Starting..."
docker compose up -d

print_step "Containers iniciados!"

# 5. Verificar status
echo ""
echo -e "${BLUE}[5/5] Verificando status...${NC}"

sleep 3

if docker compose ps | grep -q "running"; then
    print_step "Tracker rodando com sucesso!"
else
    print_error "Erro ao iniciar. Verifique os logs:"
    echo "docker compose logs tracker"
    exit 1
fi

# Health check
echo ""
echo "Testando API..."
sleep 2

if curl -s http://localhost:5000/api/health | grep -q "ok"; then
    print_step "API respondendo corretamente!"
else
    print_warning "API pode demorar alguns segundos para iniciar..."
fi

# Final
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         Instalação Concluída!            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo -e "Acesse: ${BLUE}http://$(hostname -I | awk '{print $1}'):5000${NC}"
echo ""
echo "Comandos úteis:"
echo "  docker compose logs -f     # Ver logs"
echo "  docker compose restart     # Reiniciar"
echo "  docker compose down        # Parar"
echo ""
