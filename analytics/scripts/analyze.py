#!/usr/bin/env python3
"""
LADM Analytics - Agent 0: Quick Analysis
=========================================
Análise rápida dos dados para ter uma visão geral.

Uso:
    python analytics/scripts/analyze.py 2026-01-04
    python analytics/scripts/analyze.py 2026-01-04 2026-01-07
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from load_data import DataLoader, HAS_RICH

if HAS_RICH:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
else:
    console = None
    rprint = print

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def analyze_trades(df) -> dict:
    """Analisa trades"""
    if df is None or (hasattr(df, 'shape') and df.shape[0] == 0):
        return {'count': 0}

    # Convert to pandas if polars
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if not HAS_PANDAS:
        return {'count': len(df) if hasattr(df, '__len__') else 0}

    stats = {
        'count': len(df),
    }

    # Try to extract more stats based on available columns
    if 'price' in df.columns:
        stats['price_mean'] = df['price'].mean()
        stats['price_min'] = df['price'].min()
        stats['price_max'] = df['price'].max()

    if 'size' in df.columns:
        stats['size_mean'] = df['size'].mean()
        stats['size_total'] = df['size'].sum()

    if 'side' in df.columns:
        side_counts = df['side'].value_counts().to_dict()
        stats['yes_trades'] = side_counts.get('YES', side_counts.get('yes', 0))
        stats['no_trades'] = side_counts.get('NO', side_counts.get('no', 0))

    return stats


def analyze_prices(df) -> dict:
    """Analisa preços"""
    if df is None or (hasattr(df, 'shape') and df.shape[0] == 0):
        return {'count': 0}

    if HAS_POLARS and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    if not HAS_PANDAS:
        return {'count': len(df) if hasattr(df, '__len__') else 0}

    stats = {
        'count': len(df),
    }

    # Look for price column (might be named differently)
    price_col = None
    for col in ['price', 'btcPrice', 'binancePrice', 'chainlinkPrice']:
        if col in df.columns:
            price_col = col
            break

    if price_col:
        stats['btc_mean'] = df[price_col].mean()
        stats['btc_min'] = df[price_col].min()
        stats['btc_max'] = df[price_col].max()
        stats['btc_volatility'] = df[price_col].std()

    return stats


def analyze_books(df) -> dict:
    """Analisa order books"""
    if df is None or (hasattr(df, 'shape') and df.shape[0] == 0):
        return {'count': 0}

    return {
        'snapshots': len(df) if hasattr(df, '__len__') else 0,
    }


def analyze_state(df) -> dict:
    """Analisa state ticks"""
    if df is None or (hasattr(df, 'shape') and df.shape[0] == 0):
        return {'count': 0}

    if HAS_POLARS and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    stats = {
        'count': len(df),
    }

    # Look for market slugs
    if 'marketSlug' in df.columns:
        stats['unique_markets'] = df['marketSlug'].nunique()

    return stats


def print_analysis_report(date: str, data: dict, stats: dict):
    """Imprime relatório de análise"""

    if HAS_RICH:
        console.print(Panel(f"[bold cyan]📊 Análise de Dados - {date}[/bold cyan]", expand=False))

        # Data summary table
        table = Table(title="📦 Dados Carregados")
        table.add_column("Dataset", style="cyan")
        table.add_column("Registros", justify="right", style="green")
        table.add_column("Status", justify="center")

        for name, df in data.items():
            if df is not None and hasattr(df, 'shape'):
                count = df.shape[0]
                status = "✅" if count > 0 else "⚠️"
                table.add_row(name, f"{count:,}", status)
            elif df is not None and hasattr(df, '__len__'):
                count = len(df)
                status = "✅" if count > 0 else "⚠️"
                table.add_row(name, f"{count:,}", status)
            else:
                table.add_row(name, "0", "❌")

        console.print(table)
        console.print()

        # Trades analysis
        if stats.get('trades', {}).get('count', 0) > 0:
            t = stats['trades']
            table = Table(title="🔄 Análise de Trades")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", justify="right", style="green")

            table.add_row("Total de trades", f"{t.get('count', 0):,}")
            if 'yes_trades' in t:
                table.add_row("Trades YES", f"{t.get('yes_trades', 0):,}")
                table.add_row("Trades NO", f"{t.get('no_trades', 0):,}")
            if 'price_mean' in t:
                table.add_row("Preço médio", f"{t.get('price_mean', 0):.4f}")
                table.add_row("Preço min", f"{t.get('price_min', 0):.4f}")
                table.add_row("Preço max", f"{t.get('price_max', 0):.4f}")
            if 'size_total' in t:
                table.add_row("Volume total", f"${t.get('size_total', 0):,.2f}")
                table.add_row("Tamanho médio", f"${t.get('size_mean', 0):,.2f}")

            console.print(table)
            console.print()

        # Prices analysis
        if stats.get('prices', {}).get('count', 0) > 0:
            p = stats['prices']
            table = Table(title="💰 Análise de Preços BTC")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", justify="right", style="green")

            table.add_row("Total de ticks", f"{p.get('count', 0):,}")
            if 'btc_mean' in p:
                table.add_row("BTC médio", f"${p.get('btc_mean', 0):,.2f}")
                table.add_row("BTC min", f"${p.get('btc_min', 0):,.2f}")
                table.add_row("BTC max", f"${p.get('btc_max', 0):,.2f}")
                table.add_row("Volatilidade (std)", f"${p.get('btc_volatility', 0):,.2f}")

            console.print(table)
            console.print()

        # State analysis
        if stats.get('state', {}).get('count', 0) > 0:
            s = stats['state']
            table = Table(title="📊 Análise de State")
            table.add_column("Métrica", style="cyan")
            table.add_column("Valor", justify="right", style="green")

            table.add_row("Total de ticks", f"{s.get('count', 0):,}")
            if 'unique_markets' in s:
                table.add_row("Markets únicos", f"{s.get('unique_markets', 0):,}")

            console.print(table)

    else:
        # Plain text output
        print(f"\n{'='*50}")
        print(f"📊 Análise de Dados - {date}")
        print(f"{'='*50}\n")

        print("📦 Dados Carregados:")
        for name, df in data.items():
            if df is not None and hasattr(df, 'shape'):
                print(f"  {name}: {df.shape[0]:,} registros")
            elif df is not None:
                print(f"  {name}: {len(df):,} registros")
            else:
                print(f"  {name}: 0 registros")

        print(f"\n🔄 Trades: {stats.get('trades', {})}")
        print(f"💰 Prices: {stats.get('prices', {})}")
        print(f"📊 State: {stats.get('state', {})}")


def save_report(date: str, stats: dict, output_dir: Path):
    """Salva relatório em JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"analysis-{date}.json"

    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        else:
            return str(obj)

    serializable_stats = make_serializable(stats)
    serializable_stats['date'] = date
    serializable_stats['generated_at'] = datetime.now().isoformat()

    with open(output_file, 'w') as f:
        json.dump(serializable_stats, f, indent=2)

    if HAS_RICH:
        rprint(f"\n[green]💾 Relatório salvo em:[/green] {output_file}")
    else:
        print(f"\n💾 Relatório salvo em: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python analyze.py <data>")
        print("     python analyze.py <data_inicio> <data_fim>")
        print("\nExemplo:")
        print("     python analyze.py 2026-01-04")
        print("     python analyze.py 2026-01-01 2026-01-07")
        sys.exit(1)

    date = sys.argv[1]
    end_date = sys.argv[2] if len(sys.argv) > 2 else None

    # Initialize loader
    loader = DataLoader(show_progress=True)

    # Get project root for output
    output_dir = loader.base_dir / 'analytics' / 'reports' / 'validation'

    if end_date:
        # Multi-date analysis
        if HAS_RICH:
            rprint(f"\n[bold cyan]📆 Analisando período: {date} → {end_date}[/bold cyan]\n")
        else:
            print(f"\n📆 Analisando período: {date} → {end_date}\n")

        # For now, just analyze each date
        from datetime import datetime, timedelta
        start = datetime.strptime(date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        current = start
        all_stats = []
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')

            if HAS_RICH:
                rprint(f"\n[bold]Analisando {date_str}...[/bold]")

            data = loader.load_all_for_date(date_str)

            stats = {
                'trades': analyze_trades(data.get('trades')),
                'prices': analyze_prices(data.get('prices')),
                'books': analyze_books(data.get('books')),
                'state': analyze_state(data.get('state')),
            }

            all_stats.append({'date': date_str, **stats})
            current += timedelta(days=1)

        # Summary
        if HAS_RICH:
            table = Table(title=f"📊 Resumo do Período {date} → {end_date}")
            table.add_column("Data", style="cyan")
            table.add_column("Trades", justify="right")
            table.add_column("Prices", justify="right")
            table.add_column("State", justify="right")

            for s in all_stats:
                table.add_row(
                    s['date'],
                    f"{s['trades'].get('count', 0):,}",
                    f"{s['prices'].get('count', 0):,}",
                    f"{s['state'].get('count', 0):,}"
                )

            console.print(table)

    else:
        # Single date analysis
        if HAS_RICH:
            rprint(f"\n[bold cyan]📆 Analisando {date}...[/bold cyan]\n")
        else:
            print(f"\n📆 Analisando {date}...\n")

        # Load all data
        data = loader.load_all_for_date(date)

        # Analyze each dataset
        stats = {
            'trades': analyze_trades(data.get('trades')),
            'prices': analyze_prices(data.get('prices')),
            'books': analyze_books(data.get('books')),
            'state': analyze_state(data.get('state')),
        }

        # Print report
        print_analysis_report(date, data, stats)

        # Save report
        save_report(date, stats, output_dir)


if __name__ == '__main__':
    main()
