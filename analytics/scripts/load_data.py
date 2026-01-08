#!/usr/bin/env python3
"""
LADM Analytics - Data Loader
============================
Carrega dados JSONL do bot ladm-bot para análise quantitativa.
Inclui barras de progresso para acompanhar carregamento.

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
from typing import Optional, List, Dict, Any, Generator, Callable
from glob import glob

# Progress bar imports
try:
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
        MofNCompleteColumn
    )
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_file_line_count(filepath: Path) -> int:
    """Conta linhas do arquivo rapidamente"""
    if not filepath.exists():
        return 0
    with open(filepath, 'rb') as f:
        return sum(1 for _ in f)


def find_project_root() -> Path:
    """Encontra o diretório raiz do projeto ladm-bot"""
    home_path = Path.home() / 'ladm-bot'
    if home_path.exists():
        return home_path

    cwd = Path.cwd()
    if (cwd / 'data').exists():
        return cwd

    for parent in cwd.parents:
        if (parent / 'data').exists():
            return parent

    raise FileNotFoundError("Não foi possível encontrar o diretório do projeto ladm-bot")


class DataLoader:
    """Carregador de dados para análise quantitativa com progresso"""

    def __init__(self, base_dir: Optional[Path] = None, show_progress: bool = True):
        self.base_dir = base_dir or find_project_root()
        self.data_dir = self.base_dir / 'data'
        self.show_progress = show_progress

        self.dirs = {
            'state': self.data_dir / 'state',
            'prices': self.data_dir / 'prices',
            'books': self.data_dir / 'books',
            'trades': self.data_dir / 'trades',
            'events': self.data_dir / 'events',
        }

        if HAS_RICH:
            rprint(f"[green]📂 DataLoader inicializado:[/green] {self.base_dir}")
        else:
            print(f"📂 DataLoader inicializado: {self.base_dir}")

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Valida que a estrutura de diretórios existe"""
        missing = []
        for name, path in self.dirs.items():
            if not path.exists():
                missing.append(name)

        if missing:
            if HAS_RICH:
                rprint(f"[yellow]⚠️  Diretórios faltando:[/yellow] {missing}")
            else:
                print(f"⚠️  Diretórios faltando: {missing}")

    def list_available_dates(self, data_type: str = 'state') -> List[str]:
        """Lista todas as datas disponíveis para um tipo de dado"""
        pattern = self.dirs[data_type] / '*.jsonl'
        files = glob(str(pattern))

        dates = []
        for f in files:
            name = Path(f).stem
            parts = name.split('-')
            if len(parts) >= 4:
                date = f"{parts[1]}-{parts[2]}-{parts[3]}"
                dates.append(date)

        return sorted(set(dates))

    def _load_jsonl_with_progress(self, filepath: Path, description: str = "Carregando") -> List[Dict[str, Any]]:
        """Carrega arquivo JSONL com barra de progresso"""
        records = []

        if not filepath.exists():
            if HAS_RICH:
                rprint(f"[red]⚠️  Arquivo não encontrado:[/red] {filepath}")
            else:
                print(f"⚠️  Arquivo não encontrado: {filepath}")
            return records

        total_lines = get_file_line_count(filepath)
        errors = 0

        if self.show_progress and HAS_RICH and total_lines > 0:
            # Rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("→"),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(f"{description} [{filepath.name}]", total=total_lines)

                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                errors += 1
                        progress.update(task, advance=1)

        elif self.show_progress and HAS_TQDM and total_lines > 0:
            # Fallback to tqdm
            with open(filepath, 'r') as f:
                for line in tqdm(f, total=total_lines, desc=f"{description} [{filepath.name}]",
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            errors += 1

        else:
            # No progress bar
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            errors += 1

        # Summary
        if HAS_RICH:
            if errors > 0:
                rprint(f"[green]✅ Carregados {len(records):,} registros[/green] [yellow]({errors} erros)[/yellow]")
            else:
                rprint(f"[green]✅ Carregados {len(records):,} registros de {filepath.name}[/green]")
        else:
            print(f"✅ Carregados {len(records):,} registros de {filepath.name}")

        return records

    def _to_dataframe(self, records: List[Dict], flatten: bool = True, description: str = "Convertendo"):
        """Converte lista de dicts para DataFrame com progresso"""
        if not records:
            if HAS_POLARS:
                return pl.DataFrame()
            elif HAS_PANDAS:
                return pd.DataFrame()
            return records

        if self.show_progress and HAS_RICH:
            with console.status(f"[bold green]{description} para DataFrame...[/bold green]"):
                if HAS_POLARS:
                    try:
                        return pl.DataFrame(records)
                    except Exception:
                        if HAS_PANDAS:
                            return pd.json_normalize(records) if flatten else pd.DataFrame(records)
                elif HAS_PANDAS:
                    return pd.json_normalize(records) if flatten else pd.DataFrame(records)
        else:
            if HAS_POLARS:
                try:
                    return pl.DataFrame(records)
                except Exception:
                    if HAS_PANDAS:
                        return pd.json_normalize(records) if flatten else pd.DataFrame(records)
            elif HAS_PANDAS:
                return pd.json_normalize(records) if flatten else pd.DataFrame(records)

        return records

    # ============ Loaders por Tipo ============

    def load_state(self, date: str, as_df: bool = True) -> Any:
        """Carrega state ticks de uma data específica."""
        filepath = self.dirs['state'] / f'state-{date}.jsonl'
        records = self._load_jsonl_with_progress(filepath, "📊 State")
        return self._to_dataframe(records, description="State") if as_df else records

    def load_prices(self, date: str, as_df: bool = True) -> Any:
        """Carrega price ticks de uma data específica."""
        filepath = self.dirs['prices'] / f'prices-{date}.jsonl'
        records = self._load_jsonl_with_progress(filepath, "💰 Prices")
        return self._to_dataframe(records, description="Prices") if as_df else records

    def load_books(self, date: str, as_df: bool = True) -> Any:
        """Carrega order book snapshots de uma data específica."""
        filepath = self.dirs['books'] / f'books-{date}.jsonl'
        records = self._load_jsonl_with_progress(filepath, "📖 Books")
        return self._to_dataframe(records, description="Books") if as_df else records

    def load_trades(self, date: str, as_df: bool = True) -> Any:
        """Carrega trades executados de uma data específica."""
        filepath = self.dirs['trades'] / f'trades-{date}.jsonl'
        records = self._load_jsonl_with_progress(filepath, "🔄 Trades")
        return self._to_dataframe(records, description="Trades") if as_df else records

    def load_events(self, date: str, as_df: bool = True) -> Any:
        """Carrega eventos de uma data específica."""
        filepath = self.dirs['events'] / f'events-{date}.jsonl'
        records = self._load_jsonl_with_progress(filepath, "📅 Events")
        return self._to_dataframe(records, description="Events") if as_df else records

    # ============ Loaders Multi-Data ============

    def load_date_range(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        as_df: bool = True
    ) -> Any:
        """Carrega dados de um intervalo de datas com progresso geral."""
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
        total_days = (end - start).days + 1

        all_records = []

        if self.show_progress and HAS_RICH:
            rprint(f"\n[bold cyan]📆 Carregando {data_type} de {start_date} a {end_date} ({total_days} dias)[/bold cyan]\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("→"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Dias processados", total=total_days)

                current = start
                while current <= end:
                    date_str = current.strftime('%Y-%m-%d')
                    # Temporarily disable inner progress
                    old_show = self.show_progress
                    self.show_progress = False
                    data = loader(date_str, as_df=False)
                    self.show_progress = old_show
                    all_records.extend(data)
                    progress.update(task, advance=1, description=f"Dia {date_str}")
                    current += timedelta(days=1)
        else:
            current = start
            while current <= end:
                date_str = current.strftime('%Y-%m-%d')
                data = loader(date_str, as_df=False)
                all_records.extend(data)
                current += timedelta(days=1)

        if HAS_RICH:
            rprint(f"\n[green]📊 Total: {len(all_records):,} registros de {start_date} a {end_date}[/green]")
        else:
            print(f"📊 Total: {len(all_records):,} registros de {start_date} a {end_date}")

        return self._to_dataframe(all_records) if as_df else all_records

    def load_all_for_date(self, date: str) -> Dict[str, Any]:
        """Carrega todos os tipos de dados para uma data específica com progresso."""
        data_types = ['state', 'prices', 'books', 'trades', 'events']
        result = {}

        if self.show_progress and HAS_RICH:
            rprint(f"\n[bold cyan]📦 Carregando todos os dados de {date}[/bold cyan]\n")

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Datasets", total=len(data_types))

                for dtype in data_types:
                    progress.update(task, description=f"Carregando {dtype}")
                    old_show = self.show_progress
                    self.show_progress = False

                    if dtype == 'state':
                        result[dtype] = self.load_state(date)
                    elif dtype == 'prices':
                        result[dtype] = self.load_prices(date)
                    elif dtype == 'books':
                        result[dtype] = self.load_books(date)
                    elif dtype == 'trades':
                        result[dtype] = self.load_trades(date)
                    elif dtype == 'events':
                        result[dtype] = self.load_events(date)

                    self.show_progress = old_show
                    progress.update(task, advance=1)

            # Summary table
            table = Table(title=f"📊 Resumo - {date}")
            table.add_column("Dataset", style="cyan")
            table.add_column("Registros", justify="right", style="green")
            table.add_column("Colunas", justify="right", style="yellow")

            for name, df in result.items():
                if hasattr(df, 'shape'):
                    table.add_row(name, f"{df.shape[0]:,}", str(df.shape[1]))
                else:
                    table.add_row(name, str(len(df)), "-")

            console.print(table)

        else:
            result = {
                'state': self.load_state(date),
                'prices': self.load_prices(date),
                'books': self.load_books(date),
                'trades': self.load_trades(date),
                'events': self.load_events(date),
            }

        return result

    # ============ Streaming para Arquivos Grandes ============

    def stream_jsonl(self, data_type: str, date: str) -> Generator[Dict, None, None]:
        """Generator para processar arquivos grandes linha por linha."""
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

    def stream_with_progress(self, data_type: str, date: str,
                             processor: Callable[[Dict], Any]) -> List[Any]:
        """
        Processa arquivo com barra de progresso e função customizada.

        Uso:
            results = loader.stream_with_progress('trades', '2026-01-04',
                lambda t: t['price'] * t['size'])
        """
        filepath = self.dirs[data_type] / f'{data_type}-{date}.jsonl'
        if not filepath.exists():
            return []

        total_lines = get_file_line_count(filepath)
        results = []

        if self.show_progress and HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"Processando {data_type}", total=total_lines)

                with open(filepath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                result = processor(record)
                                if result is not None:
                                    results.append(result)
                            except (json.JSONDecodeError, Exception):
                                pass
                        progress.update(task, advance=1)
        else:
            for record in self.stream_jsonl(data_type, date):
                try:
                    result = processor(record)
                    if result is not None:
                        results.append(result)
                except Exception:
                    pass

        return results


# ============ Funções de Conveniência ============

def load_state(date: str):
    return DataLoader().load_state(date)

def load_prices(date: str):
    return DataLoader().load_prices(date)

def load_books(date: str):
    return DataLoader().load_books(date)

def load_trades(date: str):
    return DataLoader().load_trades(date)

def load_events(date: str):
    return DataLoader().load_events(date)


# ============ CLI ============

if __name__ == '__main__':
    import sys

    loader = DataLoader()

    if HAS_RICH:
        # Table de estrutura
        table = Table(title="📁 Estrutura de Dados")
        table.add_column("Tipo", style="cyan")
        table.add_column("Path", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Arquivos", justify="right")

        for name, path in loader.dirs.items():
            exists = path.exists()
            status = "[green]✅[/green]" if exists else "[red]❌[/red]"
            if exists:
                count = len(list(path.glob('*.jsonl')))
                table.add_row(name, str(path), status, str(count))
            else:
                table.add_row(name, str(path), status, "-")

        console.print(table)

        # Datas disponíveis
        rprint("\n[bold cyan]📅 Datas Disponíveis:[/bold cyan]")
        for data_type in ['state', 'prices', 'books', 'trades', 'events']:
            dates = loader.list_available_dates(data_type)
            if dates:
                rprint(f"  [green]{data_type}[/green]: {dates[0]} → {dates[-1]} ({len(dates)} dias)")
            else:
                rprint(f"  [yellow]{data_type}[/yellow]: (nenhum arquivo)")

    else:
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

    # Exemplo de uso com data
    if len(sys.argv) > 1:
        date = sys.argv[1]
        rprint(f"\n[bold]📊 Carregando dados de {date}...[/bold]\n") if HAS_RICH else print(f"\n📊 Carregando dados de {date}...")
        data = loader.load_all_for_date(date)
