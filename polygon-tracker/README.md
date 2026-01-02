# Polygon Transaction Tracker - Gabagool

Track Gabagool's Polymarket transactions on the Polygon blockchain in real-time.

## Features

- Real-time transaction tracking from Polygon blockchain
- SYNC button for manual refresh
- Filters: Side (Buy/Sell), Token (YES/NO), Date range
- Statistics: Total transactions, Buys, Sells, Unique markets
- Dark theme UI
- Docker ready for VPS deployment

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (get one free at https://polygonscan.com/myapikey)
export POLYGONSCAN_API_KEY="your_api_key"

# Run
python app.py

# Open http://localhost:5000
```

### Docker

```bash
# Copy env file
cp .env.example .env

# Edit .env with your API key
nano .env

# Build and run
docker compose up -d

# Open http://localhost:5000
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `POLYGONSCAN_API_KEY` | Your Polygonscan API key | Required |
| `GABAGOOL_WALLET` | Wallet address to track | `0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d` |
| `FLASK_ENV` | Flask environment | `development` |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/transactions` | GET | Get all transactions |
| `/api/sync` | POST | Force sync from blockchain |
| `/api/health` | GET | Health check |

## Wallet Info

- **Profile**: https://polymarket.com/@gabagool22
- **Wallet**: `0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d`
- **Polygonscan**: https://polygonscan.com/address/0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d

## License

MIT
