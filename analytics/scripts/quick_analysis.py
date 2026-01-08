#!/usr/bin/env python3
"""
LADM Analytics - Análise Rápida (sem gráficos)
==============================================
Análise rápida sem geração de gráficos.

Uso:
    python analytics/scripts/quick_analysis.py 2026-01-04
"""

import sys
import json
from pathlib import Path
from datetime import datetime

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
    import numpy as np
except ImportError:
    print("❌ Instale: pip install pandas numpy")
    sys.exit(1)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def to_pandas(df):
    if df is None:
        return pd.DataFrame()
    if HAS_POLARS and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame(df) if isinstance(df, list) else pd.DataFrame()


def analyze_prices(df):
    if df.empty:
        return {}

    price_col = next((c for c in df.columns if 'price' in c.lower()), None)
    if not price_col:
        return {'columns': list(df.columns)}

    prices = pd.to_numeric(df[price_col], errors='coerce').dropna()
    if len(prices) == 0:
        return {}

    returns = prices.pct_change().dropna()

    return {
        'count': len(prices),
        'mean': round(prices.mean(), 2),
        'min': round(prices.min(), 2),
        'max': round(prices.max(), 2),
        'range_pct': round((prices.max() - prices.min()) / prices.mean() * 100, 2),
        'volatility': round(returns.std() * 100, 4) if len(returns) > 0 else 0,
    }


def analyze_trades(df):
    if df.empty:
        return {'count': 0}

    stats = {'count': len(df)}

    if 'size' in df.columns:
        sizes = pd.to_numeric(df['size'], errors='coerce').dropna()
        stats['volume_total'] = round(sizes.sum(), 2)
        stats['size_mean'] = round(sizes.mean(), 2)

    if 'price' in df.columns:
        prices = pd.to_numeric(df['price'], errors='coerce').dropna()
        stats['price_mean'] = round(prices.mean(), 4)

        if 'size' in df.columns:
            valid = df[['price', 'size']].dropna()
            if len(valid) > 0:
                p = pd.to_numeric(valid['price'], errors='coerce')
                s = pd.to_numeric(valid['size'], errors='coerce')
                stats['vwap'] = round((p * s).sum() / s.sum(), 4)

    if 'side' in df.columns:
        df_copy = df.copy()
        df_copy['side_upper'] = df_copy['side'].astype(str).str.upper()
        stats['yes_count'] = len(df_copy[df_copy['side_upper'] == 'YES'])
        stats['no_count'] = len(df_copy[df_copy['side_upper'] == 'NO'])

        if 'size' in df.columns:
            df_copy['size'] = pd.to_numeric(df_copy['size'], errors='coerce')
            stats['yes_volume'] = round(df_copy[df_copy['side_upper'] == 'YES']['size'].sum(), 2)
            stats['no_volume'] = round(df_copy[df_copy['side_upper'] == 'NO']['size'].sum(), 2)
            total = stats['yes_volume'] + stats['no_volume']
            if total > 0:
                stats['flow_imbalance'] = round((stats['yes_volume'] - stats['no_volume']) / total, 4)

    return stats


def analyze_books(df):
    if df.empty:
        return {'count': 0}

    stats = {'count': len(df)}

    bid_col = next((c for c in df.columns if 'bid' in c.lower()), None)
    ask_col = next((c for c in df.columns if 'ask' in c.lower()), None)

    if bid_col and ask_col:
        bids = pd.to_numeric(df[bid_col], errors='coerce')
        asks = pd.to_numeric(df[ask_col], errors='coerce')
        spreads = (asks - bids).dropna()

        if len(spreads) > 0:
            stats['spread_mean'] = round(spreads.mean(), 4)
            stats['spread_min'] = round(spreads.min(), 4)
            stats['spread_max'] = round(spreads.max(), 4)

    return stats


def analyze_state(df):
    if df.empty:
        return {'count': 0}

    stats = {'count': len(df)}

    if 'marketSlug' in df.columns:
        stats['unique_markets'] = df['marketSlug'].nunique()

    return stats


def print_results(date, prices, trades, books, state):
    if HAS_RICH:
        console.print(Panel(f"[bold green]📊 Análise Rápida - {date}[/bold green]"))

        # Prices
        if prices.get('count', 0) > 0:
            t = Table(title="💰 Preços BTC", show_header=True)
            t.add_column("Métrica", style="cyan")
            t.add_column("Valor", justify="right", style="green")
            t.add_row("Registros", f"{prices['count']:,}")
            t.add_row("Média", f"${prices.get('mean', 0):,.2f}")
            t.add_row("Min", f"${prices.get('min', 0):,.2f}")
            t.add_row("Max", f"${prices.get('max', 0):,.2f}")
            t.add_row("Range", f"{prices.get('range_pct', 0):.2f}%")
            t.add_row("Volatilidade", f"{prices.get('volatility', 0):.4f}%")
            console.print(t)
            console.print()

        # Trades
        if trades.get('count', 0) > 0:
            t = Table(title="🔄 Trades", show_header=True)
            t.add_column("Métrica", style="cyan")
            t.add_column("Valor", justify="right", style="green")
            t.add_row("Total", f"{trades['count']:,}")
            if 'volume_total' in trades:
                t.add_row("Volume Total", f"${trades['volume_total']:,.2f}")
            if 'yes_count' in trades:
                t.add_row("YES", f"{trades['yes_count']:,}")
                t.add_row("NO", f"{trades['no_count']:,}")
            if 'yes_volume' in trades:
                t.add_row("Vol YES", f"${trades['yes_volume']:,.2f}")
                t.add_row("Vol NO", f"${trades['no_volume']:,.2f}")
            if 'flow_imbalance' in trades:
                imb = trades['flow_imbalance']
                direction = "🟢 YES" if imb > 0 else "🔴 NO"
                t.add_row("Imbalance", f"{imb:+.2%} {direction}")
            if 'vwap' in trades:
                t.add_row("VWAP", f"{trades['vwap']:.4f}")
            console.print(t)
            console.print()

        # Books
        if books.get('count', 0) > 0:
            t = Table(title="📖 Order Book", show_header=True)
            t.add_column("Métrica", style="cyan")
            t.add_column("Valor", justify="right", style="green")
            t.add_row("Snapshots", f"{books['count']:,}")
            if 'spread_mean' in books:
                t.add_row("Spread Médio", f"{books['spread_mean']:.4f}")
                t.add_row("Spread Min", f"{books['spread_min']:.4f}")
                t.add_row("Spread Max", f"{books['spread_max']:.4f}")
            console.print(t)
            console.print()

        # State
        if state.get('count', 0) > 0:
            t = Table(title="📊 State", show_header=True)
            t.add_column("Métrica", style="cyan")
            t.add_column("Valor", justify="right", style="green")
            t.add_row("Ticks", f"{state['count']:,}")
            if 'unique_markets' in state:
                t.add_row("Markets", f"{state['unique_markets']}")
            console.print(t)
    else:
        print(f"\n📊 Análise - {date}")
        print(f"Prices: {prices}")
        print(f"Trades: {trades}")
        print(f"Books: {books}")
        print(f"State: {state}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python quick_analysis.py <data>")
        print("Ex:  python quick_analysis.py 2026-01-04")
        sys.exit(1)

    date = sys.argv[1]

    loader = DataLoader(show_progress=True)

    if HAS_RICH:
        rprint(f"\n[bold cyan]⚡ Análise Rápida (sem gráficos) - {date}[/bold cyan]\n")

    # Load
    data = loader.load_all_for_date(date)

    # Convert & analyze
    prices_df = to_pandas(data.get('prices'))
    trades_df = to_pandas(data.get('trades'))
    books_df = to_pandas(data.get('books'))
    state_df = to_pandas(data.get('state'))

    rprint("\n[bold]Analisando...[/bold]") if HAS_RICH else print("\nAnalisando...")

    prices_stats = analyze_prices(prices_df)
    trades_stats = analyze_trades(trades_df)
    books_stats = analyze_books(books_df)
    state_stats = analyze_state(state_df)

    # Print
    print_results(date, prices_stats, trades_stats, books_stats, state_stats)

    # Save
    output_dir = loader.base_dir / 'analytics' / 'reports' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'date': date,
        'generated_at': datetime.now().isoformat(),
        'prices': prices_stats,
        'trades': trades_stats,
        'books': books_stats,
        'state': state_stats,
    }

    output_file = output_dir / f'quick_{date}.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    if HAS_RICH:
        rprint(f"\n[green]💾 Salvo:[/green] {output_file}")
    else:
        print(f"\n💾 Salvo: {output_file}")


if __name__ == '__main__':
    main()
