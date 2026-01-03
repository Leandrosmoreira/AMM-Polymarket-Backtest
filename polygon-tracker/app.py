"""
Polygon Transaction Tracker - Gabagool Polymarket
Backend Flask para rastrear transações na Polymarket
"""

import os
import sqlite3
import json
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, request, g
import requests

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
DB_FILE = os.path.join(DATA_DIR, 'transactions.db')

# Cache
last_sync = None

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# ============== DATABASE ==============

def get_db():
    """Retorna conexão com o banco de dados"""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Fecha conexão ao final da request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    """Inicializa o banco de dados"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Tabela de transações
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hash TEXT UNIQUE NOT NULL,
            timestamp TEXT,
            timestamp_unix INTEGER,
            date TEXT,
            time TEXT,
            from_addr TEXT,
            to_addr TEXT,
            side TEXT,
            token_type TEXT,
            token_symbol TEXT,
            token_name TEXT,
            token_id TEXT,
            amount REAL,
            value_matic REAL,
            gas_used TEXT,
            gas_price TEXT,
            contract TEXT,
            method TEXT,
            is_error INTEGER DEFAULT 0,
            tx_type TEXT,
            block_number INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Índices para buscas rápidas
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON transactions(timestamp_unix DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_side ON transactions(side)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_token_type ON transactions(token_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON transactions(date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_block ON transactions(block_number)')

    # Tabela de metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized")


def save_transactions(transactions):
    """Salva transações no banco de dados"""
    if not transactions:
        return 0

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    inserted = 0
    for tx in transactions:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO transactions
                (hash, timestamp, timestamp_unix, date, time, from_addr, to_addr,
                 side, token_type, token_symbol, token_name, token_id, amount,
                 value_matic, gas_used, gas_price, contract, method, is_error,
                 tx_type, block_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx['hash'], tx['timestamp'], tx['timestamp_unix'], tx['date'],
                tx['time'], tx['from'], tx['to'], tx['side'], tx['token_type'],
                tx['token_symbol'], tx['token_name'], tx['token_id'], tx['amount'],
                tx['value_matic'], tx['gas_used'], tx['gas_price'], tx['contract'],
                tx['method'], 1 if tx['is_error'] else 0, tx['tx_type'], tx['block_number']
            ))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            print(f"Erro ao salvar tx {tx.get('hash', 'unknown')}: {e}")

    # Atualizar metadata
    if transactions:
        max_block = max(tx.get('block_number', 0) for tx in transactions)
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_block', ?)
        ''', (str(max_block),))
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_sync', ?)
        ''', (datetime.now().isoformat(),))

    conn.commit()
    conn.close()

    print(f"Salvo {inserted} novas transações no banco")
    return inserted


def load_transactions(limit=None, offset=0, side=None, token_type=None, date_from=None, date_to=None):
    """Carrega transações do banco com filtros"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = 'SELECT * FROM transactions WHERE is_error = 0'
    params = []

    if side:
        query += ' AND side = ?'
        params.append(side)

    if token_type:
        query += ' AND token_type = ?'
        params.append(token_type)

    if date_from:
        query += ' AND date >= ?'
        params.append(date_from)

    if date_to:
        query += ' AND date <= ?'
        params.append(date_to)

    query += ' ORDER BY timestamp_unix DESC'

    if limit:
        query += ' LIMIT ? OFFSET ?'
        params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()

    transactions = []
    for row in rows:
        transactions.append({
            'hash': row['hash'],
            'timestamp': row['timestamp'],
            'timestamp_unix': row['timestamp_unix'],
            'date': row['date'],
            'time': row['time'],
            'from': row['from_addr'],
            'to': row['to_addr'],
            'side': row['side'],
            'token_type': row['token_type'],
            'token_symbol': row['token_symbol'],
            'token_name': row['token_name'],
            'token_id': row['token_id'],
            'amount': row['amount'],
            'value_matic': row['value_matic'],
            'gas_used': row['gas_used'],
            'gas_price': row['gas_price'],
            'contract': row['contract'],
            'method': row['method'],
            'is_error': bool(row['is_error']),
            'tx_type': row['tx_type'],
            'block_number': row['block_number']
        })

    conn.close()
    return transactions


def get_last_block():
    """Retorna o último bloco salvo"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM metadata WHERE key = "last_block"')
    row = cursor.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def get_stats():
    """Calcula estatísticas do banco"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    stats = {}

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE is_error = 0')
    stats['total_transactions'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE side = "BUY" AND is_error = 0')
    stats['total_buys'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE side = "SELL" AND is_error = 0')
    stats['total_sells'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(DISTINCT token_id) FROM transactions WHERE token_id != "" AND is_error = 0')
    stats['unique_markets'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE token_type = "YES" AND is_error = 0')
    stats['total_yes_tokens'] = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM transactions WHERE token_type = "NO" AND is_error = 0')
    stats['total_no_tokens'] = cursor.fetchone()[0]

    cursor.execute('SELECT MIN(date), MAX(date) FROM transactions WHERE is_error = 0')
    row = cursor.fetchone()
    stats['first_tx_date'] = row[0]
    stats['last_tx_date'] = row[1]

    cursor.execute('SELECT SUM(amount) FROM transactions WHERE side = "BUY" AND is_error = 0')
    stats['total_buy_amount'] = cursor.fetchone()[0] or 0

    cursor.execute('SELECT SUM(amount) FROM transactions WHERE side = "SELL" AND is_error = 0')
    stats['total_sell_amount'] = cursor.fetchone()[0] or 0

    conn.close()
    return stats


def get_transactions_by_date():
    """Agrupa transações por data"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT date,
               COUNT(*) as total,
               SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buys,
               SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sells,
               SUM(amount) as volume
        FROM transactions
        WHERE is_error = 0
        GROUP BY date
        ORDER BY date DESC
        LIMIT 30
    ''')

    rows = cursor.fetchall()
    conn.close()

    return [{'date': r[0], 'total': r[1], 'buys': r[2], 'sells': r[3], 'volume': r[4]} for r in rows]


# ============== API POLYGONSCAN ==============

def fetch_block_range(address, start_block, end_block, sort='asc'):
    """
    Busca transações em um range específico de blocos.
    Retorna lista de transações e o número de resultados.
    """
    url = 'https://api.etherscan.io/v2/api'
    all_results = []
    page = 1

    while True:
        params = {
            'chainid': 137,
            'module': 'account',
            'action': 'token1155tx',
            'address': address,
            'startblock': start_block,
            'endblock': end_block,
            'page': page,
            'offset': 1000,
            'sort': sort,
            'apikey': POLYGONSCAN_API_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            data = response.json()

            if data.get('status') == '1':
                results = data.get('result', [])
                if not results:
                    break
                all_results.extend(results)
                if len(results) < 1000:
                    break
                page += 1
                if page > 10:  # Limite de 10 páginas por range
                    break
                time.sleep(0.25)
            else:
                break
        except Exception as e:
            print(f"Erro ao buscar blocos {start_block}-{end_block}: {e}")
            break

    return all_results


def fetch_and_save_all_transactions(address, start_block=0, save_each_page=True, direction='both'):
    """
    Busca TODAS as transferências ERC1155 usando estratégia de ranges.

    direction:
    - 'asc': do mais antigo ao mais recente
    - 'desc': do mais recente ao mais antigo
    - 'both': busca em ambas direções para pegar tudo
    """
    url = 'https://api.etherscan.io/v2/api'
    total_saved = 0

    print(f"=== INICIANDO SYNC COMPLETO ===")
    print(f"Carteira: {address}")
    print(f"Direção: {direction}")
    print("=" * 40)

    # Primeiro: buscar do mais recente para o mais antigo (DESC)
    # Isso garante que pegamos as transações mais recentes
    if direction in ['desc', 'both']:
        print("\n>>> FASE 1: Buscando transações mais recentes (DESC)...")
        page = 1
        while True:
            params = {
                'chainid': 137,
                'module': 'account',
                'action': 'token1155tx',
                'address': address,
                'startblock': 0,
                'endblock': 99999999,
                'page': page,
                'offset': 1000,
                'sort': 'desc',  # Mais recentes primeiro
                'apikey': POLYGONSCAN_API_KEY
            }

            try:
                print(f"[DESC Página {page}] Buscando...")
                response = requests.get(url, params=params, timeout=60)
                data = response.json()

                if data.get('status') == '1':
                    results = data.get('result', [])
                    count = len(results)
                    print(f"[DESC Página {page}] Encontradas: {count} transações")

                    if count == 0:
                        break

                    parsed = []
                    for tx in results:
                        tx['tokenType'] = 'ERC1155'
                        p = parse_transaction(tx)
                        if not p['is_error']:
                            parsed.append(p)

                    if save_each_page and parsed:
                        saved = save_transactions(parsed)
                        total_saved += saved
                        stats = get_stats()
                        print(f"[DESC Página {page}] Salvas: {saved} novas | Total no banco: {stats['total_transactions']}")

                    if count < 1000:
                        break

                    page += 1
                    time.sleep(0.3)

                else:
                    msg = data.get('message', 'Unknown error')
                    result = data.get('result', '')
                    print(f"[DESC Página {page}] API erro: {msg}")

                    if 'rate' in str(result).lower():
                        time.sleep(5)
                        continue
                    break

            except Exception as e:
                print(f"[DESC Página {page}] ERRO: {e}")
                time.sleep(3)
                continue

    # Segundo: buscar do mais antigo para o mais recente (ASC)
    # Isso pega as transações antigas que não vieram no DESC
    if direction in ['asc', 'both']:
        print("\n>>> FASE 2: Buscando transações antigas (ASC)...")
        page = 1
        while True:
            params = {
                'chainid': 137,
                'module': 'account',
                'action': 'token1155tx',
                'address': address,
                'startblock': start_block,
                'endblock': 99999999,
                'page': page,
                'offset': 1000,
                'sort': 'asc',  # Mais antigas primeiro
                'apikey': POLYGONSCAN_API_KEY
            }

            try:
                print(f"[ASC Página {page}] Buscando...")
                response = requests.get(url, params=params, timeout=60)
                data = response.json()

                if data.get('status') == '1':
                    results = data.get('result', [])
                    count = len(results)
                    print(f"[ASC Página {page}] Encontradas: {count} transações")

                    if count == 0:
                        break

                    parsed = []
                    for tx in results:
                        tx['tokenType'] = 'ERC1155'
                        p = parse_transaction(tx)
                        if not p['is_error']:
                            parsed.append(p)

                    if save_each_page and parsed:
                        saved = save_transactions(parsed)
                        total_saved += saved
                        stats = get_stats()
                        print(f"[ASC Página {page}] Salvas: {saved} novas | Total no banco: {stats['total_transactions']}")

                    if count < 1000:
                        break

                    page += 1
                    time.sleep(0.3)

                else:
                    msg = data.get('message', 'Unknown error')
                    result = data.get('result', '')
                    print(f"[ASC Página {page}] API erro: {msg}")

                    if 'rate' in str(result).lower():
                        time.sleep(5)
                        continue
                    break

            except Exception as e:
                print(f"[ASC Página {page}] ERRO: {e}")
                time.sleep(3)
                continue

    print("\n" + "=" * 40)
    print(f"=== SYNC COMPLETO FINALIZADO ===")
    stats = get_stats()
    print(f"Total de transações no banco: {stats['total_transactions']}")
    print(f"Buys: {stats['total_buys']} | Sells: {stats['total_sells']}")
    print(f"Mercados únicos: {stats['unique_markets']}")
    print(f"Período: {stats.get('first_tx_date', 'N/A')} a {stats.get('last_tx_date', 'N/A')}")
    print("=" * 40)

    return total_saved


def get_token_transfers_paginated(address, start_block=0, max_pages=10):
    """Busca transferências ERC1155 com paginação (versão simples para sync rápido)"""
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
            'sort': 'asc',
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

                if len(results) < 1000:
                    break
            else:
                print(f"Página {page}: Sem mais resultados")
                break

            time.sleep(0.25)

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

    # BUY = recebendo tokens, SELL = enviando tokens
    if to_addr == wallet_lower:
        side = 'BUY'
    elif from_addr == wallet_lower:
        side = 'SELL'
    else:
        side = 'BUY'

    value = tx.get('value', '0')
    value_eth = int(value) / 1e18 if value and value != '0' else 0

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
        try:
            token_amount = float(token_value) / 1e6 if token_value else 0
        except:
            token_amount = 0

    token_id = tx.get('tokenID', '')

    token_type = 'UNKNOWN'
    if token_id:
        try:
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


def fetch_new_transactions(force_full=False):
    """Busca novas transações da API"""
    global last_sync

    start_block = 0 if force_full else get_last_block()
    if start_block > 0:
        start_block += 1

    print(f"Buscando transações a partir do bloco {start_block}...")

    new_transfers = get_token_transfers_paginated(GABAGOOL_WALLET, start_block)
    print(f"Transferências encontradas: {len(new_transfers)}")

    parsed = []
    for tx in new_transfers:
        p = parse_transaction(tx)
        if not p['is_error']:
            parsed.append(p)

    print(f"Transações válidas: {len(parsed)}")

    if parsed:
        saved = save_transactions(parsed)
        print(f"Novas transações salvas: {saved}")

    last_sync = datetime.now().isoformat()
    return len(parsed)


# ============== ROUTES ==============

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', wallet=GABAGOOL_WALLET)


@app.route('/api/transactions')
def api_transactions():
    """API endpoint para buscar transações com filtros"""
    try:
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        side = request.args.get('side')
        token_type = request.args.get('token_type')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')

        transactions = load_transactions(
            limit=limit,
            offset=offset,
            side=side,
            token_type=token_type,
            date_from=date_from,
            date_to=date_to
        )

        stats = get_stats()

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
    """Sincroniza novas transações"""
    try:
        new_count = fetch_new_transactions(force_full=False)
        transactions = load_transactions(limit=1000)
        stats = get_stats()

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'new_transactions': new_count,
            'message': f'Sincronizado! {new_count} novas transações. Total: {stats["total_transactions"]}'
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
    """Força sincronização completa (10 páginas)"""
    try:
        new_count = fetch_new_transactions(force_full=True)
        transactions = load_transactions(limit=1000)
        stats = get_stats()

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'new_transactions': new_count,
            'message': f'Sync completo! {stats["total_transactions"]} transações no banco.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/sync/all', methods=['POST'])
def sync_all():
    """
    Sincroniza TODAS as transações da carteira.
    Busca todas as páginas e salva a cada página.
    Use este endpoint para baixar todo o histórico.
    """
    global last_sync
    try:
        # Busca do bloco 0 para pegar tudo
        total_saved = fetch_and_save_all_transactions(
            GABAGOOL_WALLET,
            start_block=0,
            save_each_page=True
        )

        last_sync = datetime.now().isoformat()
        transactions = load_transactions(limit=1000)
        stats = get_stats()

        return jsonify({
            'success': True,
            'transactions': transactions,
            'stats': stats,
            'wallet': GABAGOOL_WALLET,
            'last_sync': last_sync,
            'new_transactions': total_saved,
            'message': f'Sync ALL completo! {stats["total_transactions"]} transações no banco.'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def api_stats():
    """Retorna estatísticas"""
    try:
        stats = get_stats()
        daily = get_transactions_by_date()
        return jsonify({
            'success': True,
            'stats': stats,
            'daily': daily
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export/csv')
def export_csv():
    """Exporta transações como CSV"""
    import csv
    import io
    from flask import Response

    transactions = load_transactions()

    output = io.StringIO()
    fieldnames = ['hash', 'timestamp', 'date', 'time', 'side', 'token_type',
                  'amount', 'token_id', 'from', 'to', 'contract', 'method', 'block_number']

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(transactions)

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=gabagool_transactions.csv'}
    )


@app.route('/api/export/json')
def export_json():
    """Exporta transações como JSON"""
    transactions = load_transactions()
    return jsonify(transactions)


@app.route('/api/health')
def health():
    """Health check"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM transactions')
    count = cursor.fetchone()[0]
    conn.close()

    return jsonify({
        'status': 'ok',
        'wallet': GABAGOOL_WALLET,
        'has_api_key': bool(POLYGONSCAN_API_KEY),
        'db_transactions': count,
        'last_sync': last_sync
    })


@app.route('/api/query', methods=['POST'])
def custom_query():
    """Executa query SQL customizada (apenas SELECT)"""
    try:
        data = request.get_json()
        query = data.get('query', '')

        # Segurança: apenas SELECT permitido
        if not query.strip().upper().startswith('SELECT'):
            return jsonify({'success': False, 'error': 'Apenas SELECT permitido'}), 400

        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        return jsonify({'success': True, 'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Initialize database on startup
init_db()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
