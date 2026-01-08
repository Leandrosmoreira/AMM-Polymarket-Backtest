#!/usr/bin/env python3
"""
LADM Analytics - Análise Quantitativa Completa
===============================================
Gera análise detalhada com gráficos e métricas.

Uso:
    python analytics/scripts/full_analysis.py 2026-01-04
    python analytics/scripts/full_analysis.py 2026-01-04 --all-dates
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

warnings.filterwarnings('ignore')

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from load_data import DataLoader, HAS_RICH, get_file_line_count

if HAS_RICH:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich import print as rprint
    console = Console()
else:
    console = None
    rprint = print

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("❌ pandas/numpy não instalado. Execute: pip install pandas numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib não instalado. Gráficos desabilitados.")

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class FullAnalyzer:
    """Análise quantitativa completa com gráficos"""

    def __init__(self, loader: DataLoader, output_dir: Path):
        self.loader = loader
        self.output_dir = output_dir
        self.figures_dir = output_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def to_pandas(self, df) -> pd.DataFrame:
        """Converte para pandas se necessário"""
        if df is None:
            return pd.DataFrame()
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            return df.to_pandas()
        if isinstance(df, pd.DataFrame):
            return df
        if isinstance(df, list):
            return pd.DataFrame(df)
        return pd.DataFrame()

    # ==================== ANÁLISE DE PREÇOS ====================

    def analyze_prices(self, df: pd.DataFrame, date: str) -> Dict[str, Any]:
        """Análise detalhada de preços BTC"""
        if df.empty:
            return {'error': 'No price data'}

        stats = {}

        # Find price column
        price_col = None
        for col in ['price', 'btcPrice', 'binancePrice', 'chainlink_price']:
            if col in df.columns:
                price_col = col
                break

        if price_col is None:
            # Try to find any column with 'price' in name
            price_cols = [c for c in df.columns if 'price' in c.lower()]
            if price_cols:
                price_col = price_cols[0]

        if price_col is None:
            return {'error': 'No price column found', 'columns': list(df.columns)}

        prices = pd.to_numeric(df[price_col], errors='coerce').dropna()

        if len(prices) == 0:
            return {'error': 'No valid prices'}

        # Basic stats
        stats['count'] = len(prices)
        stats['mean'] = float(prices.mean())
        stats['std'] = float(prices.std())
        stats['min'] = float(prices.min())
        stats['max'] = float(prices.max())
        stats['range'] = stats['max'] - stats['min']
        stats['range_pct'] = (stats['range'] / stats['mean']) * 100

        # Returns
        returns = prices.pct_change().dropna()
        if len(returns) > 0:
            stats['volatility_1s'] = float(returns.std() * 100)  # % per tick
            stats['volatility_annualized'] = float(returns.std() * np.sqrt(86400 * 365) * 100)
            stats['max_drawdown'] = float((prices / prices.cummax() - 1).min() * 100)

        # Generate chart
        if HAS_MATPLOTLIB and len(prices) > 10:
            self._plot_prices(prices, date)
            stats['chart'] = f'figures/prices_{date}.png'

        return stats

    def _plot_prices(self, prices: pd.Series, date: str):
        """Gera gráfico de preços"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Análise de Preços BTC - {date}', fontsize=14, fontweight='bold')

        # 1. Price timeline
        ax1 = axes[0, 0]
        ax1.plot(prices.values, linewidth=0.5, color='blue', alpha=0.7)
        ax1.set_title('Preço BTC ao longo do tempo')
        ax1.set_xlabel('Tick')
        ax1.set_ylabel('Preço (USD)')
        ax1.grid(True, alpha=0.3)

        # 2. Price distribution
        ax2 = axes[0, 1]
        ax2.hist(prices.values, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(prices.mean(), color='red', linestyle='--', label=f'Média: ${prices.mean():,.2f}')
        ax2.set_title('Distribuição de Preços')
        ax2.set_xlabel('Preço (USD)')
        ax2.set_ylabel('Frequência')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Returns distribution
        ax3 = axes[1, 0]
        returns = prices.pct_change().dropna() * 100
        ax3.hist(returns.values, bins=100, color='orange', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--')
        ax3.set_title('Distribuição de Retornos (%)')
        ax3.set_xlabel('Retorno (%)')
        ax3.set_ylabel('Frequência')
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative returns
        ax4 = axes[1, 1]
        cum_returns = (1 + prices.pct_change().fillna(0)).cumprod() - 1
        ax4.plot(cum_returns.values * 100, linewidth=1, color='purple')
        ax4.fill_between(range(len(cum_returns)), cum_returns.values * 100, alpha=0.3, color='purple')
        ax4.set_title('Retorno Acumulado (%)')
        ax4.set_xlabel('Tick')
        ax4.set_ylabel('Retorno (%)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self.figures_dir / f'prices_{date}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ==================== ANÁLISE DE TRADES ====================

    def analyze_trades(self, df: pd.DataFrame, date: str) -> Dict[str, Any]:
        """Análise detalhada de trades"""
        if df.empty:
            return {'error': 'No trade data', 'count': 0}

        stats = {'count': len(df)}

        # Volume analysis
        if 'size' in df.columns:
            sizes = pd.to_numeric(df['size'], errors='coerce').dropna()
            stats['volume_total'] = float(sizes.sum())
            stats['volume_mean'] = float(sizes.mean())
            stats['volume_median'] = float(sizes.median())
            stats['volume_std'] = float(sizes.std())
            stats['volume_max'] = float(sizes.max())

        # Price analysis
        if 'price' in df.columns:
            prices = pd.to_numeric(df['price'], errors='coerce').dropna()
            stats['price_mean'] = float(prices.mean())
            stats['price_std'] = float(prices.std())
            stats['price_min'] = float(prices.min())
            stats['price_max'] = float(prices.max())

            # VWAP
            if 'size' in df.columns:
                valid = df[['price', 'size']].dropna()
                if len(valid) > 0:
                    vwap = (valid['price'].astype(float) * valid['size'].astype(float)).sum() / valid['size'].astype(float).sum()
                    stats['vwap'] = float(vwap)

        # Side analysis (YES/NO)
        if 'side' in df.columns:
            side_counts = df['side'].value_counts().to_dict()
            stats['yes_count'] = side_counts.get('YES', side_counts.get('yes', side_counts.get('Yes', 0)))
            stats['no_count'] = side_counts.get('NO', side_counts.get('no', side_counts.get('No', 0)))

            if 'size' in df.columns:
                df_copy = df.copy()
                df_copy['size'] = pd.to_numeric(df_copy['size'], errors='coerce')
                yes_vol = df_copy[df_copy['side'].str.upper() == 'YES']['size'].sum()
                no_vol = df_copy[df_copy['side'].str.upper() == 'NO']['size'].sum()
                stats['yes_volume'] = float(yes_vol) if not pd.isna(yes_vol) else 0
                stats['no_volume'] = float(no_vol) if not pd.isna(no_vol) else 0
                total_vol = stats['yes_volume'] + stats['no_volume']
                if total_vol > 0:
                    stats['flow_imbalance'] = (stats['yes_volume'] - stats['no_volume']) / total_vol

        # Generate charts
        if HAS_MATPLOTLIB and len(df) > 10:
            self._plot_trades(df, date)
            stats['chart'] = f'figures/trades_{date}.png'

        return stats

    def _plot_trades(self, df: pd.DataFrame, date: str):
        """Gera gráficos de trades"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Análise de Trades - {date}', fontsize=14, fontweight='bold')

        # 1. Trade size distribution
        ax1 = axes[0, 0]
        if 'size' in df.columns:
            sizes = pd.to_numeric(df['size'], errors='coerce').dropna()
            if len(sizes) > 0:
                ax1.hist(sizes.values, bins=50, color='blue', alpha=0.7, edgecolor='black')
                ax1.axvline(sizes.median(), color='red', linestyle='--', label=f'Mediana: ${sizes.median():,.2f}')
                ax1.set_title('Distribuição de Tamanho dos Trades')
                ax1.set_xlabel('Tamanho (USD)')
                ax1.set_ylabel('Frequência')
                ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. YES vs NO volume
        ax2 = axes[0, 1]
        if 'side' in df.columns and 'size' in df.columns:
            df_copy = df.copy()
            df_copy['size'] = pd.to_numeric(df_copy['size'], errors='coerce')
            yes_vol = df_copy[df_copy['side'].str.upper() == 'YES']['size'].sum()
            no_vol = df_copy[df_copy['side'].str.upper() == 'NO']['size'].sum()
            if yes_vol > 0 or no_vol > 0:
                ax2.bar(['YES', 'NO'], [yes_vol, no_vol], color=['green', 'red'], alpha=0.7)
                ax2.set_title('Volume por Lado')
                ax2.set_ylabel('Volume (USD)')
                for i, v in enumerate([yes_vol, no_vol]):
                    ax2.text(i, v + v*0.02, f'${v:,.0f}', ha='center', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Price distribution of trades
        ax3 = axes[1, 0]
        if 'price' in df.columns:
            prices = pd.to_numeric(df['price'], errors='coerce').dropna()
            if len(prices) > 0:
                ax3.hist(prices.values, bins=50, color='purple', alpha=0.7, edgecolor='black')
                ax3.set_title('Distribuição de Preços dos Trades')
                ax3.set_xlabel('Preço')
                ax3.set_ylabel('Frequência')
        ax3.grid(True, alpha=0.3)

        # 4. Trade count over time (binned)
        ax4 = axes[1, 1]
        if 'ts' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['ts'] = pd.to_numeric(df_copy['ts'], errors='coerce')
                df_copy = df_copy.dropna(subset=['ts'])
                if len(df_copy) > 0:
                    # Bin into 5-minute intervals
                    df_copy['bin'] = (df_copy['ts'] // (5 * 60 * 1000)).astype(int)
                    trade_counts = df_copy.groupby('bin').size()
                    ax4.bar(range(len(trade_counts)), trade_counts.values, color='orange', alpha=0.7)
                    ax4.set_title('Trades por Intervalo de 5min')
                    ax4.set_xlabel('Intervalo')
                    ax4.set_ylabel('Número de Trades')
            except Exception:
                pass
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / f'trades_{date}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ==================== ANÁLISE DE BOOKS ====================

    def analyze_books(self, df: pd.DataFrame, date: str) -> Dict[str, Any]:
        """Análise de order books"""
        if df.empty:
            return {'error': 'No book data', 'count': 0}

        stats = {'count': len(df)}

        # Look for bid/ask columns
        bid_col = None
        ask_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'bid' in col_lower and bid_col is None:
                bid_col = col
            if 'ask' in col_lower and ask_col is None:
                ask_col = col

        if bid_col and ask_col:
            bids = pd.to_numeric(df[bid_col], errors='coerce').dropna()
            asks = pd.to_numeric(df[ask_col], errors='coerce').dropna()

            if len(bids) > 0 and len(asks) > 0:
                spreads = asks - bids
                stats['spread_mean'] = float(spreads.mean())
                stats['spread_std'] = float(spreads.std())
                stats['spread_min'] = float(spreads.min())
                stats['spread_max'] = float(spreads.max())

                # Mid price
                mids = (bids + asks) / 2
                stats['mid_mean'] = float(mids.mean())
                stats['mid_std'] = float(mids.std())

        # Generate chart
        if HAS_MATPLOTLIB and len(df) > 10:
            self._plot_books(df, date, bid_col, ask_col)
            stats['chart'] = f'figures/books_{date}.png'

        return stats

    def _plot_books(self, df: pd.DataFrame, date: str, bid_col: str, ask_col: str):
        """Gera gráficos de order books"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Análise de Order Book - {date}', fontsize=14, fontweight='bold')

        if bid_col and ask_col:
            bids = pd.to_numeric(df[bid_col], errors='coerce')
            asks = pd.to_numeric(df[ask_col], errors='coerce')
            spreads = asks - bids
            mids = (bids + asks) / 2

            # 1. Bid/Ask over time
            ax1 = axes[0, 0]
            ax1.plot(bids.values, label='Bid', color='green', alpha=0.7, linewidth=0.5)
            ax1.plot(asks.values, label='Ask', color='red', alpha=0.7, linewidth=0.5)
            ax1.fill_between(range(len(bids)), bids.values, asks.values, alpha=0.2, color='gray')
            ax1.set_title('Bid/Ask ao longo do tempo')
            ax1.set_xlabel('Snapshot')
            ax1.set_ylabel('Preço')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Spread distribution
            ax2 = axes[0, 1]
            valid_spreads = spreads.dropna()
            if len(valid_spreads) > 0:
                ax2.hist(valid_spreads.values, bins=50, color='orange', alpha=0.7, edgecolor='black')
                ax2.axvline(valid_spreads.mean(), color='red', linestyle='--',
                           label=f'Média: {valid_spreads.mean():.4f}')
                ax2.set_title('Distribuição do Spread')
                ax2.set_xlabel('Spread')
                ax2.set_ylabel('Frequência')
                ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Spread over time
            ax3 = axes[1, 0]
            ax3.plot(spreads.values, color='purple', alpha=0.7, linewidth=0.5)
            ax3.axhline(spreads.mean(), color='red', linestyle='--', alpha=0.5)
            ax3.set_title('Spread ao longo do tempo')
            ax3.set_xlabel('Snapshot')
            ax3.set_ylabel('Spread')
            ax3.grid(True, alpha=0.3)

            # 4. Mid price volatility
            ax4 = axes[1, 1]
            rolling_std = mids.rolling(window=100).std()
            ax4.plot(rolling_std.values, color='blue', alpha=0.7, linewidth=0.5)
            ax4.set_title('Volatilidade do Mid Price (rolling 100)')
            ax4.set_xlabel('Snapshot')
            ax4.set_ylabel('Std Dev')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / f'books_{date}.png', dpi=150, bbox_inches='tight')
        plt.close()

    # ==================== ANÁLISE DE STATE ====================

    def analyze_state(self, df: pd.DataFrame, date: str) -> Dict[str, Any]:
        """Análise de state ticks"""
        if df.empty:
            return {'error': 'No state data', 'count': 0}

        stats = {'count': len(df)}

        # Market slugs
        if 'marketSlug' in df.columns:
            stats['unique_markets'] = int(df['marketSlug'].nunique())
            stats['markets'] = df['marketSlug'].unique().tolist()[:10]  # First 10

        # Regime/phase analysis
        phase_col = None
        for col in df.columns:
            if 'phase' in col.lower() or 'regime' in col.lower():
                phase_col = col
                break

        if phase_col:
            phase_counts = df[phase_col].value_counts().to_dict()
            stats['phases'] = phase_counts

        return stats

    # ==================== RELATÓRIO PRINCIPAL ====================

    def generate_report(self, date: str) -> Dict[str, Any]:
        """Gera relatório completo para uma data"""

        if HAS_RICH:
            console.print(Panel(f"[bold cyan]🔬 Análise Quantitativa Completa - {date}[/bold cyan]"))

        # Load data
        data = self.loader.load_all_for_date(date)

        # Convert to pandas
        prices_df = self.to_pandas(data.get('prices'))
        trades_df = self.to_pandas(data.get('trades'))
        books_df = self.to_pandas(data.get('books'))
        state_df = self.to_pandas(data.get('state'))
        events_df = self.to_pandas(data.get('events'))

        # Analyze each dataset
        if HAS_RICH:
            with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                         BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task("Analisando...", total=4)

                progress.update(task, description="Analisando preços...")
                prices_stats = self.analyze_prices(prices_df, date)
                progress.advance(task)

                progress.update(task, description="Analisando trades...")
                trades_stats = self.analyze_trades(trades_df, date)
                progress.advance(task)

                progress.update(task, description="Analisando books...")
                books_stats = self.analyze_books(books_df, date)
                progress.advance(task)

                progress.update(task, description="Analisando state...")
                state_stats = self.analyze_state(state_df, date)
                progress.advance(task)
        else:
            prices_stats = self.analyze_prices(prices_df, date)
            trades_stats = self.analyze_trades(trades_df, date)
            books_stats = self.analyze_books(books_df, date)
            state_stats = self.analyze_state(state_df, date)

        # Compile report
        report = {
            'date': date,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'prices_count': prices_stats.get('count', 0),
                'trades_count': trades_stats.get('count', 0),
                'books_count': books_stats.get('count', 0),
                'state_count': state_stats.get('count', 0),
                'events_count': len(events_df),
            },
            'prices': prices_stats,
            'trades': trades_stats,
            'books': books_stats,
            'state': state_stats,
        }

        # Print summary
        self._print_report(report)

        # Save report
        self._save_report(report, date)

        return report

    def _print_report(self, report: Dict):
        """Imprime relatório formatado"""
        date = report['date']

        if HAS_RICH:
            console.print()

            # Prices
            p = report['prices']
            if p.get('count', 0) > 0:
                table = Table(title="💰 Preços BTC", show_header=True, header_style="bold cyan")
                table.add_column("Métrica", style="white")
                table.add_column("Valor", justify="right", style="green")

                table.add_row("Registros", f"{p.get('count', 0):,}")
                if 'mean' in p:
                    table.add_row("Preço Médio", f"${p['mean']:,.2f}")
                    table.add_row("Mínimo", f"${p['min']:,.2f}")
                    table.add_row("Máximo", f"${p['max']:,.2f}")
                    table.add_row("Range", f"${p['range']:,.2f} ({p['range_pct']:.2f}%)")
                if 'volatility_1s' in p:
                    table.add_row("Volatilidade (por tick)", f"{p['volatility_1s']:.4f}%")
                if 'max_drawdown' in p:
                    table.add_row("Max Drawdown", f"{p['max_drawdown']:.2f}%")
                if 'chart' in p:
                    table.add_row("📊 Gráfico", p['chart'])

                console.print(table)
                console.print()

            # Trades
            t = report['trades']
            if t.get('count', 0) > 0:
                table = Table(title="🔄 Trades", show_header=True, header_style="bold cyan")
                table.add_column("Métrica", style="white")
                table.add_column("Valor", justify="right", style="green")

                table.add_row("Total Trades", f"{t.get('count', 0):,}")
                if 'volume_total' in t:
                    table.add_row("Volume Total", f"${t['volume_total']:,.2f}")
                    table.add_row("Tamanho Médio", f"${t['volume_mean']:,.2f}")
                if 'yes_count' in t:
                    table.add_row("Trades YES", f"{t['yes_count']:,}")
                    table.add_row("Trades NO", f"{t['no_count']:,}")
                if 'yes_volume' in t:
                    table.add_row("Volume YES", f"${t['yes_volume']:,.2f}")
                    table.add_row("Volume NO", f"${t['no_volume']:,.2f}")
                if 'flow_imbalance' in t:
                    imb = t['flow_imbalance']
                    direction = "🟢 YES" if imb > 0 else "🔴 NO"
                    table.add_row("Flow Imbalance", f"{imb:+.2%} ({direction})")
                if 'vwap' in t:
                    table.add_row("VWAP", f"{t['vwap']:.4f}")
                if 'chart' in t:
                    table.add_row("📊 Gráfico", t['chart'])

                console.print(table)
                console.print()

            # Books
            b = report['books']
            if b.get('count', 0) > 0:
                table = Table(title="📖 Order Book", show_header=True, header_style="bold cyan")
                table.add_column("Métrica", style="white")
                table.add_column("Valor", justify="right", style="green")

                table.add_row("Snapshots", f"{b.get('count', 0):,}")
                if 'spread_mean' in b:
                    table.add_row("Spread Médio", f"{b['spread_mean']:.4f}")
                    table.add_row("Spread Std", f"{b['spread_std']:.4f}")
                    table.add_row("Spread Min", f"{b['spread_min']:.4f}")
                    table.add_row("Spread Max", f"{b['spread_max']:.4f}")
                if 'mid_mean' in b:
                    table.add_row("Mid Price Médio", f"{b['mid_mean']:.4f}")
                if 'chart' in b:
                    table.add_row("📊 Gráfico", b['chart'])

                console.print(table)
                console.print()

            # State
            s = report['state']
            if s.get('count', 0) > 0:
                table = Table(title="📊 State", show_header=True, header_style="bold cyan")
                table.add_column("Métrica", style="white")
                table.add_column("Valor", justify="right", style="green")

                table.add_row("Total Ticks", f"{s.get('count', 0):,}")
                if 'unique_markets' in s:
                    table.add_row("Markets Únicos", f"{s['unique_markets']}")
                if 'phases' in s:
                    for phase, count in s['phases'].items():
                        table.add_row(f"Phase {phase}", f"{count:,}")

                console.print(table)

            # Charts generated
            console.print()
            console.print(Panel(f"[green]📊 Gráficos salvos em:[/green] {self.figures_dir}", expand=False))

        else:
            print(f"\n{'='*60}")
            print(f"📊 Relatório - {date}")
            print(f"{'='*60}")
            print(json.dumps(report, indent=2, default=str))

    def _save_report(self, report: Dict, date: str):
        """Salva relatório em JSON"""
        output_file = self.output_dir / f'full_analysis_{date}.json'

        # Make JSON serializable
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            return obj

        clean_report = json.loads(json.dumps(report, default=convert))

        with open(output_file, 'w') as f:
            json.dump(clean_report, f, indent=2)

        if HAS_RICH:
            rprint(f"\n[green]💾 Relatório salvo:[/green] {output_file}")
        else:
            print(f"\n💾 Relatório salvo: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python full_analysis.py <data>")
        print("     python full_analysis.py <data> --all-dates")
        print("\nExemplo:")
        print("     python full_analysis.py 2026-01-04")
        sys.exit(1)

    date = sys.argv[1]
    all_dates = '--all-dates' in sys.argv

    # Initialize
    loader = DataLoader(show_progress=True)
    output_dir = loader.base_dir / 'analytics' / 'reports' / 'validation'
    analyzer = FullAnalyzer(loader, output_dir)

    if all_dates:
        # Analyze all available dates
        dates = loader.list_available_dates('state')
        if HAS_RICH:
            rprint(f"\n[bold cyan]📆 Analisando {len(dates)} datas...[/bold cyan]\n")

        for d in dates:
            try:
                analyzer.generate_report(d)
            except Exception as e:
                if HAS_RICH:
                    rprint(f"[red]❌ Erro em {d}: {e}[/red]")
                else:
                    print(f"❌ Erro em {d}: {e}")
    else:
        # Single date analysis
        analyzer.generate_report(date)


if __name__ == '__main__':
    main()
