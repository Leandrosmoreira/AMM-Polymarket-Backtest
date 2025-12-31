#!/bin/bash
# Run test with simulated data

echo "=========================================="
echo "  BTC Test - Simulated Data"
echo "=========================================="

# Parse arguments
MARKETS=${1:-50}
CAPITAL=${2:-100}

echo "Markets: $MARKETS"
echo "Capital: $CAPITAL"
echo ""

# Run test
docker-compose --profile test run --rm test \
    python main.py btc-test \
    --markets $MARKETS \
    --capital $CAPITAL
