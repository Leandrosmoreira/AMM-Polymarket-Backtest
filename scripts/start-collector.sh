#!/bin/bash
# Start the BTC data collector

echo "=========================================="
echo "  BTC Data Collector - Polymarket"
echo "=========================================="

# Build and start the collector
docker-compose up -d --build collector

echo ""
echo "Collector started! Commands:"
echo "  View logs:    docker-compose logs -f collector"
echo "  Stop:         docker-compose down"
echo "  Status:       docker-compose ps"
echo ""
