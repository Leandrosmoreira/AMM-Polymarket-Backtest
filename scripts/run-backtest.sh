#!/bin/bash
# Run backtest with collected data

echo "=========================================="
echo "  BTC Backtest - Polymarket"
echo "=========================================="

# Check if data exists
if [ ! -d "./data/raw" ] || [ -z "$(ls -A ./data/raw 2>/dev/null)" ]; then
    echo "No data found in ./data/raw"
    echo "Run the collector first or add log files manually."
    exit 1
fi

# Run backtest
docker-compose --profile backtest run --rm backtest

echo ""
echo "Results saved to ./data/results/"
