"""
Polygon Transaction Tracker - Gabagool Polymarket
Backend Flask para rastrear transações na Polymarket
"""

import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import requests
from functools import lru_cache

app = Flask(__name__)

# Configurações
POLYGONSCAN_API_KEY = os.environ.get('POLYGONSCAN_API_KEY', '')
GABAGOOL_WALLET = os.environ.get('GABAGOOL_WALLET', '0x6031b6eed1c97e853c6e0f03ad3ce3529351f96d')

# Contratos Polymarket
CTF_EXCHANGE = '0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e'
NEGRISK_CTF_EXCHANGE = '0xC5d563A36AE78145C45a50134d48A1215220f80a'
CONDITIONAL_TOKENS = '0x4d97dcd97ec945f40cf65f87097ace5ea0476045'

# Cache de mercados
markets_cache = {}
last_sync = None


def get_polygonscan_transactions(address, start_block=0):
    """Busca transações normais via Polygonscan API V2"""
    # API V2 - usando chainid 137 para Polygon
    url = 'https://api.etherscan.io/v2/api'
    params = {
        'chainid': 137,  # Polygon
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': start_block,
        'endblock': 99999999,
        'page': 1,
        'offset': 1000,
        'sort': 'desc',
        'apikey': POLYGONSCAN_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        print(f"API Response txlist: status={data.get('status')}, message={data.get('message')}")
        if data.get('status') == '1':
            return data.get('result', [])
        elif data.get('result'):
            print(f"API Error: {data.get('result')}")
        return []
    except Exception as e:
        print(f"Erro Polygonscan txlist: {e}")
        return []


def get_token_transfers(address, start_block=0):
    """Busca transferências de tokens ERC20/ERC1155 via API V2"""
    url = 'https://api.etherscan.io/v2/api'

    # ERC20 transfers
    params_erc20 = {
        'chainid': 137,  # Polygon
        'module': 'account',
        'action': 'tokentx',
        'address': address,
        'startblock': start_block,
        'endblock': 99999999,
        'page': 1,
        'offset': 1000,
        'sort': 'desc',
        'apikey': POLYGONSCAN_API_KEY
    }

    # ERC1155 transfers (tokens condicionais)
    params_erc1155 = {
        'chainid': 137,  # Polygon
        'module': 'account',
        'action': 'token1155tx',
        'address': address,
        'startblock': start_block,
        'endblock': 99999999,
        'page': 1,
        'offset': 1000,
        'sort': 'desc',
        'apikey': POLYGONSCAN_API_KEY
    }

    transfers = []

    try:
        # ERC20
        response = requests.get(url, params=params_erc20, timeout=30)
        data = response.json()
        print(f"API Response ERC20: status={data.get('status')}, count={len(data.get('result', []))}")
        if data.get('status') == '1':
            for tx in data.get('result', []):
                tx['tokenType'] = 'ERC20'
            transfers.extend(data.get('result', []))
    except Exception as e:
        print(f"Erro ERC20: {e}")

    time.sleep(0.2)  # Rate limit

    try:
        # ERC1155
        response = requests.get(url, params=params_erc1155, timeout=30)
        data = response.json()
        print(f"API Response ERC1155: status={data.get('status')}, count={len(data.get('result', []))}")
        if data.get('status') == '1':
            for tx in data.get('result', []):
                tx['tokenType'] = 'ERC1155'
            transfers.extend(data.get('result', []))
    except Exception as e:
        print(f"Erro ERC1155: {e}")

    return transfers


def get_market_info(condition_id):
    """Busca informações do mercado na API da Polymarket"""
    if condition_id in markets_cache:
        return markets_cache[condition_id]

    try:
        # Tenta gamma-api
        url = f'https://gamma-api.polymarket.com/markets?condition_id={condition_id}'
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                market_info = {
                    'question': data[0].get('question', 'Unknown Market'),
                    'outcome': data[0].get('outcome', ''),
                    'image': data[0].get('image', '')
                }
                markets_cache[condition_id] = market_info
                return market_info
    except Exception as e:
        print(f"Erro buscando mercado {condition_id}: {e}")

    return {'question': f'Market {condition_id[:8]}...', 'outcome': '', 'image': ''}


def is_polymarket_transaction(tx):
    """Verifica se a transação é relacionada à Polymarket"""
    polymarket_contracts = [
        CTF_EXCHANGE.lower(),
        NEGRISK_CTF_EXCHANGE.lower(),
        CONDITIONAL_TOKENS.lower()
    ]

    to_addr = tx.get('to', '').lower()
    from_addr = tx.get('from', '').lower()
    contract_addr = tx.get('contractAddress', '').lower()

    return (to_addr in polymarket_contracts or
            from_addr in polymarket_contracts or
            contract_addr in polymarket_contracts)


def parse_transaction(tx):
    """Parse transação para formato legível"""
    timestamp = int(tx.get('timeStamp', 0))
    dt = datetime.fromtimestamp(timestamp)

    # Determinar tipo de operação
    wallet_lower = GABAGOOL_WALLET.lower()
    from_addr = tx.get('from', '').lower()
    to_addr = tx.get('to', '').lower()

    if from_addr == wallet_lower:
        side = 'SELL'
    else:
        side = 'BUY'

    # Valor
    value = tx.get('value', '0')
    if value != '0':
        value_eth = int(value) / 1e18
    else:
        value_eth = 0

    # Token info
    token_symbol = tx.get('tokenSymbol', '')
    token_name = tx.get('tokenName', '')
    token_value = tx.get('tokenValue', tx.get('value', '0'))
    token_decimal = int(tx.get('tokenDecimal', 18))

    if token_value:
        try:
            token_amount = int(token_value) / (10 ** token_decimal)
        except:
            token_amount = 0
    else:
        token_amount = 0

    # Token ID para ERC1155
    token_id = tx.get('tokenID', '')

    # Determinar se é YES ou NO baseado no token
    token_type = 'UNKNOWN'
    if 'yes' in token_name.lower() or 'yes' in token_symbol.lower():
        token_type = 'YES'
    elif 'no' in token_name.lower() or 'no' in token_symbol.lower():
        token_type = 'NO'
    elif token_id:
        # Tokens pares geralmente são YES, ímpares são NO
        try:
            if int(token_id) % 2 == 0:
                token_type = 'YES'
            else:
                token_type = 'NO'
        except:
            pass

    return {
        'hash': tx.get('hash', ''),
        'timestamp': dt.isoformat(),
        'timestamp_unix': timestamp,
        'date': dt.strftime('%Y-%m-%d'),
        'time': dt.strftime('%H:%M:%S'),
        'from': tx.get('from', ''),
        'to': tx.get('to', ''),
        'side': side,
        'token_type': token_type,
        'token_symbol': token_symbol or 'CTF',
        'token_name': token_name,
        'token_id': token_id,
        'amount': token_amount,
        'value_matic': value_eth,
        'gas_used': tx.get('gasUsed', '0'),
        'gas_price': tx.get('gasPrice', '0'),
        'contract': tx.get('contractAddress', tx.get('to', '')),
        'method': tx.get('functionName', '').split('(')[0] if tx.get('functionName') else '',
        'is_error': tx.get('isError', '0') == '1',
        'tx_type': tx.get('tokenType', 'TX')
    }


def fetch_all_transactions():
    """Busca todas as transações relacionadas à Polymarket"""
    global last_sync

    all_txs = []

    # Buscar transações normais (filtrar apenas Polymarket)
    print(f"Buscando transações para {GABAGOOL_WALLET}...")
    normal_txs = get_polygonscan_transactions(GABAGOOL_WALLET)
    print(f"Transações normais encontradas: {len(normal_txs)}")
    polymarket_txs = [tx for tx in normal_txs if is_polymarket_transaction(tx)]
    print(f"Transações Polymarket (normais): {len(polymarket_txs)}")
    all_txs.extend(polymarket_txs)

    time.sleep(0.2)  # Rate limit

    # Buscar transferências de tokens (ERC1155 já são da Polymarket)
    token_transfers = get_token_transfers(GABAGOOL_WALLET)
    print(f"Token transfers encontrados: {len(token_transfers)}")

    # ERC1155 tokens são todos da Polymarket (conditional tokens)
    # Não precisa filtrar - incluir todos
    all_txs.extend(token_transfers)
    print(f"Total de transações antes do parse: {len(all_txs)}")

    # Parse e ordenar
    parsed_txs = [parse_transaction(tx) for tx in all_txs]
    parsed_txs = [tx for tx in parsed_txs if not tx['is_error']]
    print(f"Transações após parse (sem erros): {len(parsed_txs)}")

    # Remover duplicatas por hash
    seen_hashes = set()
    unique_txs = []
    for tx in parsed_txs:
        if tx['hash'] not in seen_hashes:
            seen_hashes.add(tx['hash'])
            unique_txs.append(tx)

    # Ordenar por timestamp (mais recente primeiro)
    unique_txs.sort(key=lambda x: x['timestamp_unix'], reverse=True)

    last_sync = datetime.now().isoformat()

    return unique_txs


def calculate_stats(transactions):
    """Calcula estatísticas das transações"""
    if not transactions:
        return {
            'total_transactions': 0,
            'total_buys': 0,
            'total_sells': 0,
            'unique_markets': 0,
            'total_yes_tokens': 0,
            'total_no_tokens': 0,
            'first_tx_date': None,
            'last_tx_date': None
        }

    buys = [tx for tx in transactions if tx['side'] == 'BUY']
    sells = [tx for tx in transactions if tx['side'] == 'SELL']
    yes_txs = [tx for tx in transactions if tx['token_type'] == 'YES']
    no_txs = [tx for tx in transactions if tx['token_type'] == 'NO']

    unique_contracts = set(tx['contract'] for tx in transactions if tx['contract'])

    dates = [tx['date'] for tx in transactions if tx['date']]

    return {
        'total_transactions': len(transactions),
        'total_buys': len(buys),
        'total_sells': len(sells),
        'unique_markets': len(unique_contracts),
        'total_yes_tokens': len(yes_txs),
        'total_no_tokens': len(no_txs),
        'first_tx_date': min(dates) if dates else None,
        'last_tx_date': max(dates) if dates else None
    }


# Routes
@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', wallet=GABAGOOL_WALLET)


@app.route('/api/transactions')
def get_transactions():
    """API endpoint para buscar transações"""
    try:
        transactions = fetch_all_transactions()
        stats = calculate_stats(transactions)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sync', methods=['POST'])
def sync_transactions():
    """Endpoint para forçar sincronização"""
    try:
        transactions = fetch_all_transactions()
        stats = calculate_stats(transactions)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'message': f'Sincronizado! {len(transactions)} transações encontradas.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'wallet': GABAGOOL_WALLET,
        'has_api_key': bool(POLYGONSCAN_API_KEY)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
