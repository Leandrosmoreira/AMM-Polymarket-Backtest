# %% [markdown]
# # LADM Analytics - Quick Start
# Notebook de introdução para análise dos dados BTC Up/Down 15min

# %%
# Imports
import sys
sys.path.insert(0, '../scripts')

from load_data import DataLoader
import pandas as pd
import numpy as np

# %%
# Inicializar loader
loader = DataLoader()

# %%
# Listar datas disponíveis
print("📅 Datas disponíveis:")
for data_type in ['state', 'prices', 'books', 'trades', 'events']:
    dates = loader.list_available_dates(data_type)
    print(f"  {data_type}: {len(dates)} arquivos")
    if dates:
        print(f"    Primeiro: {dates[0]}, Último: {dates[-1]}")

# %%
# Exemplo: Carregar dados de uma data específica
# Altere a data conforme seus dados disponíveis
DATE = "2026-01-04"  # Ajuste para uma data com dados

try:
    state = loader.load_state(DATE)
    print(f"\n📊 State ticks carregados: {len(state)} registros")
    if hasattr(state, 'columns'):
        print(f"   Colunas: {list(state.columns)[:10]}...")
except Exception as e:
    print(f"⚠️  Erro ao carregar state: {e}")

# %%
# Exemplo: Streaming para arquivos grandes
count = 0
for record in loader.stream_jsonl('trades', DATE):
    count += 1
    if count >= 5:
        print(f"\n🔍 Exemplo de trade:")
        print(record)
        break

print(f"\nTotal de trades processados: {count}")

# %%
# Próximos passos:
# 1. Explore os dados carregados
# 2. Execute Agent -1 para validação de dados
# 3. Comece a análise com Agent 1 (Trade Flow)
