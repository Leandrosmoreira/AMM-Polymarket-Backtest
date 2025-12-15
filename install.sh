#!/bin/bash
#
# Polymarket Copy Trading - Script de Instalacao para Linux VPS
#

set -e

echo "================================================"
echo "  POLYMARKET COPY TRADING - INSTALACAO"
echo "================================================"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "[!] Python3 nao encontrado. Instalando..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
fi

PYTHON_VERSION=$(python3 --version)
echo "[OK] $PYTHON_VERSION"

# Criar ambiente virtual
echo ""
echo "[*] Criando ambiente virtual..."
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
echo ""
echo "[*] Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Criar arquivo .env se nao existir
if [ ! -f .env ]; then
    echo ""
    echo "[*] Criando arquivo .env..."
    cp .env.example .env
    echo "[!] IMPORTANTE: Edite o arquivo .env com sua chave privada!"
    echo "    nano .env"
fi

# Criar diretorios
mkdir -p data/copytrade_logs

echo ""
echo "================================================"
echo "  INSTALACAO CONCLUIDA!"
echo "================================================"
echo ""
echo "Proximos passos:"
echo ""
echo "1. Editar .env com sua chave privada:"
echo "   nano .env"
echo ""
echo "2. Ativar ambiente virtual:"
echo "   source venv/bin/activate"
echo ""
echo "3. Testar (dry run):"
echo "   python copytrade.py monitor"
echo ""
echo "4. Rodar em modo live:"
echo "   python copytrade.py monitor --live"
echo ""
echo "5. Rodar em background com screen:"
echo "   screen -S copytrade"
echo "   python copytrade.py monitor --live"
echo "   (Ctrl+A, D para desanexar)"
echo ""
