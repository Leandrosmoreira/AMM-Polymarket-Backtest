"""
Polygon Transaction Tracker - Gabagool Polymarket
Backend Flask para rastrear transações na Polymarket
"""

import os
import csv
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

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CSV_FILE = os.path.join(DATA_DIR, 'transactions.csv')
META_FILE = os.path.join(DATA_DIR, 'metadata.json')

# Cache
markets_cache = {}
last_sync = None
cached_transactions = []

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def save_to_csv(transactions):
    """Salva transações em CSV"""
    if not transactions:
        return

    fieldnames = ['hash', 'timestamp', 'timestamp_unix', 'date', 'time', 'from', 'to',
                  'side', 'token_type', 'token_symbol', 'token_name', 'token_id',
                  'amount', 'value_matic', 'gas_used', 'gas_price', 'contract',
                  'method', 'is_error', 'tx_type', 'block_number']

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(transactions)

    # Salvar metadata
    if transactions:
        max_block = max(int(tx.get('block_number', 0)) for tx in transactions)
        meta = {
            'last_block': max_block,
            'last_sync': datetime.now().isoformat(),
            'total_transactions': len(transactions)
        }
        with open(META_FILE, 'w') as f:
            json.dump(meta, f)

    print(f"Salvo {len(transactions)} transações em CSV")


def load_from_csv():
    """Carrega transações do CSV"""
    if not os.path.exists(CSV_FILE):
        return []

    transactions = []
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Converter tipos
                row['timestamp_unix'] = int(row.get('timestamp_unix', 0))
                row['amount'] = float(row.get('amount', 0))
                row['value_matic'] = float(row.get('value_matic', 0))
                row['is_error'] = row.get('is_error', 'False') == 'True'
                row['block_number'] = int(row.get('block_number', 0))
                transactions.append(row)
        print(f"Carregado {len(transactions)} transações do CSV")
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")

    return transactions


def get_last_block():
    """Retorna o último bloco salvo"""
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, 'r') as f:
                meta = json.load(f)
                return meta.get('last_block', 0)
        except:
            pass
    return 0


def get_token_transfers_paginated(address, start_block=0, max_pages=10):
    """Busca transferências ERC1155 com paginação"""
    url = 'https://api.etherscan.io/v2/api'
    all_transfers = []

    for page in range(1, max_pages + 1):
        params = {
            'chainid': 137,
            'module': 'account',
            'action': 'token1155tx',
            'address': address,
            'startblock': start_block,
            'endblock': 99999999,
            'page': page,
            'offset': 1000,
            'sort': 'asc',  # Ascendente para pegar do mais antigo
            'apikey': POLYGONSCAN_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get('status') == '1':
                results = data.get('result', [])
                print(f"Página {page}: {len(results)} transações ERC1155")

                for tx in results:
                    tx['tokenType'] = 'ERC1155'
                all_transfers.extend(results)

                # Se retornou menos de 1000, não há mais páginas
                if len(results) < 1000:
                    break
            else:
                print(f"Página {page}: Sem mais resultados")
                break

            time.sleep(0.25)  # Rate limit

        except Exception as e:
            print(f"Erro página {page}: {e}")
            break

    return all_transfers


def parse_transaction(tx):
    """Parse transação para formato legível"""
    timestamp = int(tx.get('timeStamp', 0))
    dt = datetime.fromtimestamp(timestamp)

    wallet_lower = GABAGOOL_WALLET.lower()
    from_addr = tx.get('from', '').lower()
    to_addr = tx.get('to', '').lower()

    # Lógica corrigida:
    # - Se tokens vão PARA a carteira = BUY (recebendo)
    # - Se tokens vão DA carteira = SELL (enviando)
    if to_addr == wallet_lower:
        side = 'BUY'
    elif from_addr == wallet_lower:
        side = 'SELL'
    else:
        # Fallback baseado no contrato
        side = 'BUY'

    # Valor
    value = tx.get('value', '0')
    value_eth = int(value) / 1e18 if value and value != '0' else 0

    # Token info
    token_symbol = tx.get('tokenSymbol', '')
    token_name = tx.get('tokenName', '')
    token_value = tx.get('tokenValue', tx.get('value', '0'))
    token_decimal = int(tx.get('tokenDecimal', 0) or 0)

    if token_value and token_decimal > 0:
        try:
            token_amount = int(token_value) / (10 ** token_decimal)
        except:
            token_amount = float(token_value) if token_value else 0
    else:
        # ERC1155 não tem decimais, usar tokenValue direto
        try:
            token_amount = float(token_value) / 1e6 if token_value else 0  # USDC tem 6 decimais
        except:
            token_amount = 0

    token_id = tx.get('tokenID', '')

    # YES/NO baseado no token ID (convenção Polymarket)
    token_type = 'UNKNOWN'
    if token_id:
        try:
            # Token IDs pares = YES, ímpares = NO
            token_type = 'YES' if int(token_id) % 2 == 0 else 'NO'
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
        'tx_type': tx.get('tokenType', 'TX'),
        'block_number': int(tx.get('blockNumber', 0))
    }


def fetch_all_transactions(force_full=False):
    """Busca todas as transações, usando cache CSV"""
    global last_sync, cached_transactions

    # Carregar transações existentes do CSV
    existing_txs = load_from_csv() if not force_full else []
    existing_hashes = set(tx['hash'] for tx in existing_txs)

    # Determinar bloco inicial (buscar apenas novas)
    start_block = 0 if force_full else get_last_block()
    if start_block > 0:
        start_block += 1  # Começar do próximo bloco

    print(f"Buscando transações a partir do bloco {start_block}...")

    # Buscar novas transações ERC1155 com paginação
    new_transfers = get_token_transfers_paginated(GABAGOOL_WALLET, start_block)
    print(f"Novas transferências encontradas: {len(new_transfers)}")

    # Parse novas transações
    new_parsed = []
    for tx in new_transfers:
        parsed = parse_transaction(tx)
        if parsed['hash'] not in existing_hashes and not parsed['is_error']:
            new_parsed.append(parsed)
            existing_hashes.add(parsed['hash'])

    print(f"Novas transações após parse: {len(new_parsed)}")

    # Combinar com existentes
    all_txs = existing_txs + new_parsed

    # Ordenar por timestamp (mais recente primeiro)
    all_txs.sort(key=lambda x: x['timestamp_unix'], reverse=True)

    # Salvar em CSV
    if new_parsed or force_full:
        save_to_csv(all_txs)

    last_sync = datetime.now().isoformat()
    cached_transactions = all_txs

    return all_txs


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

    unique_tokens = set(tx.get('token_id', '') for tx in transactions if tx.get('token_id'))
    dates = [tx['date'] for tx in transactions if tx['date']]

    return {
        'total_transactions': len(transactions),
        'total_buys': len(buys),
        'total_sells': len(sells),
        'unique_markets': len(unique_tokens),
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
    """API endpoint para buscar transações (usa cache)"""
    global cached_transactions

    try:
        # Usar cache se disponível
        if cached_transactions:
            transactions = cached_transactions
        else:
            transactions = load_from_csv()
            if not transactions:
                transactions = fetch_all_transactions()
            cached_transactions = transactions

        stats = calculate_stats(transactions)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sync', methods=['POST'])
def sync_transactions():
    """Endpoint para sincronizar novas transações"""
    global cached_transactions

    try:
        # Buscar apenas novas transações
        transactions = fetch_all_transactions(force_full=False)
        cached_transactions = transactions
        stats = calculate_stats(transactions)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'message': f'Sincronizado! {len(transactions)} transações no total.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sync/full', methods=['POST'])
def sync_full():
    """Força sincronização completa (recarrega tudo)"""
    global cached_transactions

    try:
        transactions = fetch_all_transactions(force_full=True)
        cached_transactions = transactions
        stats = calculate_stats(transactions)

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'message': f'Sync completo! {len(transactions)} transações encontradas.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/csv')
def export_csv():
    """Exporta transações como CSV"""
    from flask import send_file

    if os.path.exists(CSV_FILE):
        return send_file(CSV_FILE, as_attachment=True, download_name='gabagool_transactions.csv')

    return jsonify({'error': 'No data available'}), 404


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'wallet': GABAGOOL_WALLET,
        'has_api_key': bool(POLYGONSCAN_API_KEY),
        'csv_exists': os.path.exists(CSV_FILE),
        'cached_count': len(cached_transactions)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
