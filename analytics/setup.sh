#!/bin/bash
# LADM Analytics - Setup Script
# ==============================
# Uso: cd ~/ladm-bot && bash analytics/setup.sh

set -e

echo "🚀 Configurando ambiente de analytics LADM..."

# Diretório base
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ANALYTICS_DIR="$SCRIPT_DIR"

echo "📂 Projeto: $PROJECT_DIR"
echo "📂 Analytics: $ANALYTICS_DIR"

# Criar estrutura de diretórios
echo "📁 Criando estrutura de diretórios..."
mkdir -p "$ANALYTICS_DIR"/{notebooks,scripts}
mkdir -p "$ANALYTICS_DIR"/reports/{validation,trades,books,prices,strategy,executive}

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Instale com: apt install python3 python3-venv python3-pip"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ Python encontrado: $PYTHON_VERSION"

# Criar venv
if [ ! -d "$ANALYTICS_DIR/venv" ]; then
    echo "🐍 Criando ambiente virtual..."
    python3 -m venv "$ANALYTICS_DIR/venv"
else
    echo "✅ Venv já existe"
fi

# Ativar venv e instalar dependências
echo "📦 Instalando dependências..."
source "$ANALYTICS_DIR/venv/bin/activate"
pip install --upgrade pip
pip install -r "$ANALYTICS_DIR/requirements.txt"

# Registrar kernel Jupyter
echo "📓 Registrando kernel Jupyter..."
python -m ipykernel install --user --name=ladm-analytics --display-name="LADM Analytics"

# Verificar estrutura de dados
echo ""
echo "📊 Verificando estrutura de dados..."
for dir in state prices books trades events; do
    if [ -d "$PROJECT_DIR/data/$dir" ]; then
        count=$(ls -1 "$PROJECT_DIR/data/$dir"/*.jsonl 2>/dev/null | wc -l || echo "0")
        echo "  ✅ data/$dir: $count arquivos"
    else
        echo "  ⚠️  data/$dir: não existe"
    fi
done

echo ""
echo "✅ Setup completo!"
echo ""
echo "📝 Próximos passos:"
echo "   1. Ativar ambiente: source analytics/venv/bin/activate"
echo "   2. Testar loader:   python analytics/scripts/load_data.py"
echo "   3. Iniciar Jupyter: jupyter lab --ip=0.0.0.0 --port=8888"
echo ""
echo "🔧 Alias útil (adicione ao ~/.bashrc):"
echo "   alias ladm-analytics='cd ~/ladm-bot && source analytics/venv/bin/activate'"
