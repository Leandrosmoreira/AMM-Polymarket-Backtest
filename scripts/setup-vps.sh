#!/bin/bash
# VPS Setup Script for BTC Trading Bot
# Run this on a fresh Ubuntu/Debian VPS

set -e

echo "=========================================="
echo "  BTC Trading Bot - VPS Setup"
echo "=========================================="
echo ""

# Update system
echo "[1/5] Updating system..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
echo "[2/5] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed!"
else
    echo "Docker already installed"
fi

# Install Docker Compose
echo "[3/5] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed!"
else
    echo "Docker Compose already installed"
fi

# Create directories
echo "[4/5] Creating directories..."
mkdir -p data/raw data/processed data/results data/trades
chmod -R 755 data

# Make scripts executable
echo "[5/5] Setting permissions..."
chmod +x scripts/*.sh

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Log out and log back in (for Docker permissions)"
echo "  2. Start collector: ./scripts/start-collector.sh"
echo "  3. Run backtest:    ./scripts/run-backtest.sh"
echo ""
echo "Or use Docker Compose directly:"
echo "  docker-compose up -d collector"
echo ""
