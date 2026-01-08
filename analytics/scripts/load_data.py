#!/usr/bin/env python3
"""
LADM Analytics - Data Loader
============================
Carrega dados JSONL do bot ladm-bot para análise quantitativa.

Uso:
    from load_data import DataLoader

    loader = DataLoader()
    state = loader.load_state('2026-01-04')
    trades = loader.load_trades('2026-01-04')
"""

import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Generator
from glob import glob

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("⚠️  polars não instalado. Usando pandas.")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Detecta automaticamente o diretório base
def find_project_root() -> Path:
    """Encontra o diretório raiz do projeto ladm-bot"""
    # Tenta ~/ladm-bot primeiro (VPS)
    home_path = Path.home() / 'ladm-bot'
    if home_path.exists():
        return home_path

    # Tenta diretório atual
    cwd = Path.cwd()
    if (cwd / 'data').exists():
        return cwd

    # Tenta subir até encontrar
    for parent in cwd.parents:
        if (parent / 'data').exists():
            return parent

    raise FileNotFoundError("Não foi possível encontrar o diretório do projeto ladm-bot")


class DataLoader:
    """Carregador de dados para análise quantitativa"""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or find_project_root()
        self.data_dir = self.base_dir / 'data'

        # Diretórios de dados
        self.dirs = {
            'state': self.data_dir / 'state',
            'prices': self.data_dir / 'prices',
            'books': self.data_dir / 'books',
            'trades': self.data_dir / 'trades',
            'events': self.data_dir / 'events',
        }

        print(f"📂 DataLoader inicializado: {self.base_dir}")
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Valida que a estrutura de diretórios existe"""
        missing = []
        for name, path in self.dirs.items():
            if not path.exists():
                missing.append(name)

        if missing:
            print(f"⚠️  Diretórios faltando: {missing}")

    def list_available_dates(self, data_type: str = 'state') -> List[str]:
        """Lista todas as datas disponíveis para um tipo de dado"""
        pattern = self.dirs[data_type] / '*.jsonl'
        files = glob(str(pattern))

        dates = []
        for f in files:
            # Extrai data do nome do arquivo (ex: state-2026-01-04.jsonl)
            name = Path(f).stem
            parts = name.split('-')
            if len(parts) >= 4:
                date = f"{parts[1]}-{parts[2]}-{parts[3]}"
                dates.append(date)

        return sorted(set(dates))

    def _load_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        """Carrega arquivo JSONL linha por linha"""
        records = []
        if not filepath.exists():
            print(f"⚠️  Arquivo não encontrado: {filepath}")
            return records

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON inválido: {e}")

        print(f"✅ Carregados {len(records)} registros de {filepath.name}")
        return records

    def _to_dataframe(self, records: List[Dict], flatten: bool = True):
        """Converte lista de dicts para DataFrame"""
        if not records:
            if HAS_POLARS:
                return pl.DataFrame()
            elif HAS_PANDAS:
                return pd.DataFrame()
            return records

        if HAS_POLARS:
            try:
                return pl.DataFrame(records)
            except Exception:
                # Fallback para pandas se polars falhar com nested data
                if HAS_PANDAS:
                    return pd.json_normalize(records) if flatten else pd.DataFrame(records)
        elif HAS_PANDAS:
            return pd.json_normalize(records) if flatten else pd.DataFrame(records)

        return records

    # ============ Loaders por Tipo ============

    def load_state(self, date: str, as_df: bool = True) -> Any:
        """
        Carrega state ticks de uma data específica.

        Args:
            date: Data no formato YYYY-MM-DD
            as_df: Se True, retorna DataFrame. Se False, retorna lista de dicts.
        """
        filepath = self.dirs['state'] / f'state-{date}.jsonl'
        records = self._load_jsonl(filepath)
        return self._to_dataframe(records) if as_df else records

    def load_prices(self, date: str, as_df: bool = True) -> Any:
        """Carrega price ticks de uma data específica."""
        filepath = self.dirs['prices'] / f'prices-{date}.jsonl'
        records = self._load_jsonl(filepath)
        return self._to_dataframe(records) if as_df else records

    def load_books(self, date: str, as_df: bool = True) -> Any:
        """Carrega order book snapshots de uma data específica."""
        filepath = self.dirs['books'] / f'books-{date}.jsonl'
        records = self._load_jsonl(filepath)
        return self._to_dataframe(records) if as_df else records

    def load_trades(self, date: str, as_df: bool = True) -> Any:
        """Carrega trades executados de uma data específica."""
        filepath = self.dirs['trades'] / f'trades-{date}.jsonl'
        records = self._load_jsonl(filepath)
        return self._to_dataframe(records) if as_df else records

    def load_events(self, date: str, as_df: bool = True) -> Any:
        """Carrega eventos (phase changes, market transitions) de uma data específica."""
        filepath = self.dirs['events'] / f'events-{date}.jsonl'
        records = self._load_jsonl(filepath)
        return self._to_dataframe(records) if as_df else records

    # ============ Loaders Multi-Data ============

    def load_date_range(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        as_df: bool = True
    ) -> Any:
        """
        Carrega dados de um intervalo de datas.

        Args:
            data_type: 'state', 'prices', 'books', 'trades', 'events'
            start_date: Data inicial YYYY-MM-DD
            end_date: Data final YYYY-MM-DD
        """
        loader_map = {
            'state': self.load_state,
            'prices': self.load_prices,
            'books': self.load_books,
            'trades': self.load_trades,
            'events': self.load_events,
        }

        loader = loader_map.get(data_type)
        if not loader:
            raise ValueError(f"Tipo de dado inválido: {data_type}")

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_records = []
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            data = loader(date_str, as_df=False)
            all_records.extend(data)
            current += timedelta(days=1)

        print(f"📊 Total: {len(all_records)} registros de {start_date} a {end_date}")
        return self._to_dataframe(all_records) if as_df else all_records

    def load_all_for_date(self, date: str) -> Dict[str, Any]:
        """Carrega todos os tipos de dados para uma data específica"""
        return {
            'state': self.load_state(date),
            'prices': self.load_prices(date),
            'books': self.load_books(date),
            'trades': self.load_trades(date),
            'events': self.load_events(date),
        }

    # ============ Streaming para Arquivos Grandes ============

    def stream_jsonl(self, data_type: str, date: str) -> Generator[Dict, None, None]:
        """
        Generator para processar arquivos grandes linha por linha.

        Uso:
            for record in loader.stream_jsonl('state', '2026-01-04'):
                process(record)
        """
        filepath = self.dirs[data_type] / f'{data_type}-{date}.jsonl'
        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


# ============ Funções de Conveniência ============

def load_state(date: str):
    """Shortcut para carregar state ticks"""
    return DataLoader().load_state(date)

def load_prices(date: str):
    """Shortcut para carregar price ticks"""
    return DataLoader().load_prices(date)

def load_books(date: str):
    """Shortcut para carregar order books"""
    return DataLoader().load_books(date)

def load_trades(date: str):
    """Shortcut para carregar trades"""
    return DataLoader().load_trades(date)

def load_events(date: str):
    """Shortcut para carregar events"""
    return DataLoader().load_events(date)


# ============ CLI ============

if __name__ == '__main__':
    import sys

    loader = DataLoader()

    print("\n📁 Estrutura de dados:")
    for name, path in loader.dirs.items():
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {name}: {path}")

    print("\n📅 Datas disponíveis:")
    for data_type in ['state', 'prices', 'books', 'trades', 'events']:
        dates = loader.list_available_dates(data_type)
        if dates:
            print(f"  {data_type}: {dates[:5]}{'...' if len(dates) > 5 else ''}")
        else:
            print(f"  {data_type}: (nenhum arquivo)")

    # Exemplo de uso
    if len(sys.argv) > 1:
        date = sys.argv[1]
        print(f"\n📊 Carregando dados de {date}...")
        data = loader.load_all_for_date(date)
        for name, df in data.items():
            if hasattr(df, 'shape'):
                print(f"  {name}: {df.shape}")
            else:
                print(f"  {name}: {len(df)} registros")
