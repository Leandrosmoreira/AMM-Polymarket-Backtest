#!/usr/bin/env python3
"""
LADM Analytics - Análises Avançadas
====================================
7 análises quantitativas detalhadas.

Uso:
    python analytics/scripts/advanced_analysis.py 2026-01-04
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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

import pandas as pd
import numpy as np

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
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)


# =============================================================================
# 1. SPREAD TEMPORAL - Como spread muda ao longo do tempo
# =============================================================================
def analyze_spread_temporal(books_df: pd.DataFrame) -> dict:
    """Analisa evolução do spread ao longo do tempo"""
    if books_df.empty:
        return {'error': 'No book data'}

    results = {}

    # Procura colunas de bid/ask
    bid_col = next((c for c in books_df.columns if 'bid' in c.lower()), None)
    ask_col = next((c for c in books_df.columns if 'ask' in c.lower()), None)

    if not bid_col or not ask_col:
        # Tenta estrutura nested
        if 'yes' in books_df.columns:
            try:
                books_df['yes_bid'] = books_df['yes'].apply(lambda x: x.get('bid') if isinstance(x, dict) else None)
                books_df['yes_ask'] = books_df['yes'].apply(lambda x: x.get('ask') if isinstance(x, dict) else None)
                bid_col, ask_col = 'yes_bid', 'yes_ask'
            except:
                return {'error': 'Could not extract bid/ask', 'columns': list(books_df.columns)}
        else:
            return {'error': 'No bid/ask columns', 'columns': list(books_df.columns)}

    bids = pd.to_numeric(books_df[bid_col], errors='coerce')
    asks = pd.to_numeric(books_df[ask_col], errors='coerce')
    spreads = (asks - bids).dropna()

    if len(spreads) == 0:
        return {'error': 'No valid spreads'}

    results['count'] = len(spreads)
    results['spread_mean'] = round(float(spreads.mean()), 4)
    results['spread_std'] = round(float(spreads.std()), 4)
    results['spread_min'] = round(float(spreads.min()), 4)
    results['spread_max'] = round(float(spreads.max()), 4)
    results['spread_median'] = round(float(spreads.median()), 4)

    # Percentis
    results['spread_p10'] = round(float(spreads.quantile(0.1)), 4)
    results['spread_p90'] = round(float(spreads.quantile(0.9)), 4)

    # Divide em 4 quartis temporais
    n = len(spreads)
    q_size = n // 4
    if q_size > 0:
        results['quartiles'] = {
            'Q1_mean': round(float(spreads[:q_size].mean()), 4),
            'Q2_mean': round(float(spreads[q_size:2*q_size].mean()), 4),
            'Q3_mean': round(float(spreads[2*q_size:3*q_size].mean()), 4),
            'Q4_mean': round(float(spreads[3*q_size:].mean()), 4),
        }
        # Tendência
        results['trend'] = 'widening' if results['quartiles']['Q4_mean'] > results['quartiles']['Q1_mean'] else 'tightening'

    return results


# =============================================================================
# 2. TRADE CLUSTERING - Trades chegam em rajadas?
# =============================================================================
def analyze_trade_clustering(trades_df: pd.DataFrame) -> dict:
    """Analisa se trades chegam em clusters/rajadas"""
    if trades_df.empty or 'ts' not in trades_df.columns:
        return {'error': 'No trade data with timestamps'}

    results = {}

    ts = pd.to_numeric(trades_df['ts'], errors='coerce').dropna().sort_values()
    if len(ts) < 2:
        return {'error': 'Not enough trades'}

    # Intervalos entre trades (em segundos)
    intervals = ts.diff().dropna() / 1000  # ms to seconds

    results['trade_count'] = len(ts)
    results['interval_mean'] = round(float(intervals.mean()), 2)
    results['interval_std'] = round(float(intervals.std()), 2)
    results['interval_min'] = round(float(intervals.min()), 2)
    results['interval_max'] = round(float(intervals.max()), 2)
    results['interval_median'] = round(float(intervals.median()), 2)

    # Coeficiente de variação (CV > 1 indica clustering)
    if intervals.mean() > 0:
        cv = intervals.std() / intervals.mean()
        results['coefficient_of_variation'] = round(float(cv), 2)
        results['is_clustered'] = cv > 1.0
        results['clustering_level'] = 'High' if cv > 2 else ('Medium' if cv > 1 else 'Low')

    # Trades por minuto (binned)
    ts_min = (ts / 60000).astype(int)  # Convert to minutes
    trades_per_min = ts_min.value_counts()
    results['trades_per_minute'] = {
        'mean': round(float(trades_per_min.mean()), 2),
        'max': int(trades_per_min.max()),
        'min': int(trades_per_min.min()),
    }

    # Bursts (mais de 5 trades em 1 minuto)
    bursts = (trades_per_min > 5).sum()
    results['burst_minutes'] = int(bursts)
    results['burst_percentage'] = round(float(bursts / len(trades_per_min) * 100), 1) if len(trades_per_min) > 0 else 0

    return results


# =============================================================================
# 3. PRICE IMPACT - Quanto um trade move o preço
# =============================================================================
def analyze_price_impact(trades_df: pd.DataFrame, books_df: pd.DataFrame) -> dict:
    """Analisa impacto de trades no preço"""
    results = {}

    if trades_df.empty:
        return {'error': 'No trade data'}

    # Analisa trades
    if 'price' in trades_df.columns and 'size' in trades_df.columns:
        prices = pd.to_numeric(trades_df['price'], errors='coerce')
        sizes = pd.to_numeric(trades_df['size'], errors='coerce')

        # Price changes
        price_changes = prices.diff().dropna()
        results['price_change_mean'] = round(float(price_changes.mean()), 6)
        results['price_change_std'] = round(float(price_changes.std()), 6)
        results['price_change_abs_mean'] = round(float(price_changes.abs().mean()), 6)

        # Correlação size vs price change
        valid = pd.DataFrame({'size': sizes, 'price_change': prices.diff()}).dropna()
        if len(valid) > 10:
            corr = valid['size'].corr(valid['price_change'].abs())
            results['size_impact_correlation'] = round(float(corr), 4) if not pd.isna(corr) else 0

        # Impact por lado
        if 'side' in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df['price'] = prices
            trades_df['size'] = sizes
            trades_df['price_change'] = prices.diff()

            yes_trades = trades_df[trades_df['side'].str.upper() == 'YES']
            no_trades = trades_df[trades_df['side'].str.upper() == 'NO']

            if len(yes_trades) > 0:
                results['yes_trade_impact'] = round(float(yes_trades['price_change'].mean()), 6)
            if len(no_trades) > 0:
                results['no_trade_impact'] = round(float(no_trades['price_change'].mean()), 6)

    # Impacto estimado por $100
    if 'size_impact_correlation' in results and results.get('price_change_abs_mean'):
        avg_size = float(sizes.mean()) if 'sizes' in dir() else 25
        impact_per_100 = (results['price_change_abs_mean'] / avg_size) * 100
        results['estimated_impact_per_$100'] = round(impact_per_100, 6)

    return results


# =============================================================================
# 4. LATÊNCIA CHAINLINK - Delay entre Binance e Chainlink
# =============================================================================
def analyze_chainlink_latency(prices_df: pd.DataFrame) -> dict:
    """Analisa latência/diferença entre Chainlink e Binance"""
    if prices_df.empty:
        return {'error': 'No price data'}

    results = {}

    # Procura colunas de preço
    chainlink_col = None
    binance_col = None

    for col in prices_df.columns:
        col_lower = col.lower()
        if 'chainlink' in col_lower or col == 'price':
            chainlink_col = col
        if 'binance' in col_lower:
            binance_col = col

    if chainlink_col:
        chainlink = pd.to_numeric(prices_df[chainlink_col], errors='coerce')
        results['chainlink_mean'] = round(float(chainlink.mean()), 2)
        results['chainlink_std'] = round(float(chainlink.std()), 2)

    if binance_col:
        binance = pd.to_numeric(prices_df[binance_col], errors='coerce')
        results['binance_mean'] = round(float(binance.mean()), 2)
        results['binance_std'] = round(float(binance.std()), 2)

    # Diferença entre fontes
    if chainlink_col and binance_col:
        diff = chainlink - binance
        diff_pct = (diff / binance) * 100

        results['price_diff_mean'] = round(float(diff.mean()), 4)
        results['price_diff_std'] = round(float(diff.std()), 4)
        results['price_diff_pct_mean'] = round(float(diff_pct.mean()), 4)
        results['price_diff_pct_max'] = round(float(diff_pct.abs().max()), 4)

        # Momentos onde Chainlink está atrasado (Binance já moveu)
        divergence_threshold = 0.1  # 0.1%
        divergent = (diff_pct.abs() > divergence_threshold).sum()
        results['divergence_count'] = int(divergent)
        results['divergence_pct'] = round(float(divergent / len(diff_pct) * 100), 2)

        # Correlação (lag analysis simples)
        if len(chainlink) > 10:
            # Chainlink vs Binance shifted
            for lag in [1, 5, 10]:
                if len(binance) > lag:
                    corr = chainlink.corr(binance.shift(lag))
                    results[f'correlation_lag_{lag}'] = round(float(corr), 4) if not pd.isna(corr) else 0

    return results


# =============================================================================
# 5. REGIME DETECTION - Fases de alta/baixa volatilidade
# =============================================================================
def analyze_regime_detection(state_df: pd.DataFrame) -> dict:
    """Detecta regimes de volatilidade e comportamento"""
    if state_df.empty:
        return {'error': 'No state data'}

    results = {}
    results['total_ticks'] = len(state_df)

    # Markets únicos
    if 'marketSlug' in state_df.columns:
        results['unique_markets'] = int(state_df['marketSlug'].nunique())

    # Procura colunas de preço/fair value
    price_col = None
    for col in state_df.columns:
        if 'fair' in col.lower() or 'price' in col.lower() or 'yes' in col.lower():
            price_col = col
            break

    if price_col:
        # Tenta extrair preço
        try:
            if state_df[price_col].dtype == object:
                # Pode ser dict nested
                prices = state_df[price_col].apply(
                    lambda x: x.get('yes') if isinstance(x, dict) else x
                )
            else:
                prices = state_df[price_col]

            prices = pd.to_numeric(prices, errors='coerce').dropna()

            if len(prices) > 100:
                # Volatilidade rolling
                returns = prices.pct_change().dropna()
                rolling_vol = returns.rolling(100).std()

                results['volatility_mean'] = round(float(rolling_vol.mean() * 100), 4)
                results['volatility_max'] = round(float(rolling_vol.max() * 100), 4)
                results['volatility_min'] = round(float(rolling_vol.min() * 100), 4)

                # Detecta regimes (high/low vol)
                vol_threshold = rolling_vol.median()
                high_vol = (rolling_vol > vol_threshold * 1.5).sum()
                low_vol = (rolling_vol < vol_threshold * 0.5).sum()

                results['high_vol_periods'] = int(high_vol)
                results['low_vol_periods'] = int(low_vol)
                results['high_vol_pct'] = round(float(high_vol / len(rolling_vol) * 100), 1)

        except Exception as e:
            results['price_analysis_error'] = str(e)

    # Regime por phase se disponível
    phase_col = None
    for col in state_df.columns:
        if 'phase' in col.lower() or 'regime' in col.lower():
            phase_col = col
            break

    if phase_col:
        try:
            if state_df[phase_col].dtype == object:
                phases = state_df[phase_col].apply(
                    lambda x: x.get('phase') if isinstance(x, dict) else x
                )
            else:
                phases = state_df[phase_col]

            phase_counts = phases.value_counts().to_dict()
            results['phases'] = {str(k): int(v) for k, v in phase_counts.items()}
        except:
            pass

    return results


# =============================================================================
# 6. FLOW PREDICTION - Imbalance prediz outcome?
# =============================================================================
def analyze_flow_prediction(trades_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    """Analisa se flow imbalance prediz o outcome"""
    results = {}

    if trades_df.empty:
        return {'error': 'No trade data'}

    # Calcula imbalance
    if 'side' in trades_df.columns and 'size' in trades_df.columns:
        trades_df = trades_df.copy()
        trades_df['size'] = pd.to_numeric(trades_df['size'], errors='coerce')
        trades_df['side_upper'] = trades_df['side'].astype(str).str.upper()

        yes_vol = trades_df[trades_df['side_upper'] == 'YES']['size'].sum()
        no_vol = trades_df[trades_df['side_upper'] == 'NO']['size'].sum()
        total_vol = yes_vol + no_vol

        if total_vol > 0:
            imbalance = (yes_vol - no_vol) / total_vol
            results['flow_imbalance'] = round(float(imbalance), 4)
            results['predicted_outcome'] = 'YES' if imbalance > 0 else 'NO'
            results['confidence'] = round(abs(float(imbalance)) * 100, 1)

            # Volume breakdown
            results['yes_volume'] = round(float(yes_vol), 2)
            results['no_volume'] = round(float(no_vol), 2)
            results['volume_ratio'] = round(float(yes_vol / no_vol), 2) if no_vol > 0 else 999

    # Verifica outcome real se events disponível
    if not events_df.empty:
        # Procura evento de resolução
        if 'type' in events_df.columns:
            resolution_events = events_df[events_df['type'].str.contains('resolv', case=False, na=False)]
            if len(resolution_events) > 0:
                results['resolution_events'] = len(resolution_events)

        # Tenta extrair outcome
        for col in events_df.columns:
            if 'outcome' in col.lower() or 'result' in col.lower():
                outcomes = events_df[col].dropna().unique()
                if len(outcomes) > 0:
                    results['actual_outcomes'] = list(outcomes)[:5]

    # Análise temporal do imbalance
    if 'ts' in trades_df.columns:
        trades_df['ts'] = pd.to_numeric(trades_df['ts'], errors='coerce')
        trades_df = trades_df.sort_values('ts')

        # Divide em 4 quartis temporais
        n = len(trades_df)
        q_size = n // 4
        if q_size > 0:
            quartile_imbalances = {}
            for i, name in enumerate(['Q1', 'Q2', 'Q3', 'Q4']):
                start = i * q_size
                end = (i + 1) * q_size if i < 3 else n
                q_trades = trades_df.iloc[start:end]

                yes_v = q_trades[q_trades['side_upper'] == 'YES']['size'].sum()
                no_v = q_trades[q_trades['side_upper'] == 'NO']['size'].sum()
                total = yes_v + no_v
                if total > 0:
                    quartile_imbalances[name] = round(float((yes_v - no_v) / total), 4)

            results['imbalance_by_quartile'] = quartile_imbalances

            # Tendência do imbalance
            if 'Q1' in quartile_imbalances and 'Q4' in quartile_imbalances:
                trend = quartile_imbalances['Q4'] - quartile_imbalances['Q1']
                results['imbalance_trend'] = round(float(trend), 4)
                results['trend_direction'] = 'strengthening YES' if trend > 0 else 'strengthening NO'

    return results


# =============================================================================
# 7. MARKET EFFICIENCY - Quão rápido preço converge
# =============================================================================
def analyze_market_efficiency(prices_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """Analisa eficiência do mercado (velocidade de ajuste de preço)"""
    results = {}

    if prices_df.empty:
        return {'error': 'No price data'}

    # Encontra coluna de preço
    price_col = None
    for col in prices_df.columns:
        if 'price' in col.lower():
            price_col = col
            break

    if not price_col:
        return {'error': 'No price column found'}

    prices = pd.to_numeric(prices_df[price_col], errors='coerce').dropna()

    if len(prices) < 10:
        return {'error': 'Not enough price data'}

    # Autocorrelação (preços eficientes têm baixa autocorrelação de retornos)
    returns = prices.pct_change().dropna()

    for lag in [1, 5, 10]:
        if len(returns) > lag:
            autocorr = returns.autocorr(lag=lag)
            results[f'return_autocorr_lag{lag}'] = round(float(autocorr), 4) if not pd.isna(autocorr) else 0

    # Variância ratio test simplificado
    if len(returns) > 20:
        var_1 = returns.var()
        var_5 = returns.rolling(5).sum().var() / 5
        if var_1 > 0:
            variance_ratio = var_5 / var_1
            results['variance_ratio_5'] = round(float(variance_ratio), 4)
            results['efficiency_indicator'] = 'efficient' if 0.8 < variance_ratio < 1.2 else 'inefficient'

    # Mean reversion speed
    mean_price = prices.mean()
    deviations = prices - mean_price
    next_deviations = deviations.shift(-1)

    valid = pd.DataFrame({'dev': deviations, 'next_dev': next_deviations}).dropna()
    if len(valid) > 10:
        # Coeficiente de reversão (negativo = mean reverting)
        reversion_coef = valid['dev'].corr(valid['next_dev'] - valid['dev'])
        results['mean_reversion_coef'] = round(float(reversion_coef), 4) if not pd.isna(reversion_coef) else 0
        results['is_mean_reverting'] = reversion_coef < -0.1 if not pd.isna(reversion_coef) else False

    # Tempo para convergir (half-life estimado)
    if 'mean_reversion_coef' in results and results['mean_reversion_coef'] < 0:
        # Simplificado: half-life aproximado
        try:
            half_life = -np.log(2) / np.log(1 + results['mean_reversion_coef'])
            results['half_life_ticks'] = round(float(half_life), 1)
        except:
            pass

    # Integra com trades se disponível
    if not trades_df.empty and 'ts' in trades_df.columns:
        results['trades_count'] = len(trades_df)

        # Trades por unidade de volatilidade
        if len(returns) > 0:
            vol = returns.std()
            if vol > 0:
                results['trades_per_vol_unit'] = round(len(trades_df) / (vol * 1000), 2)

    return results


# =============================================================================
# MAIN - Executa todas as análises
# =============================================================================
def run_all_analyses(date: str):
    """Executa todas as 7 análises"""

    loader = DataLoader(show_progress=True)

    if HAS_RICH:
        console.print(Panel(f"[bold cyan]🔬 Análises Avançadas - {date}[/bold cyan]"))

    # Load data
    data = loader.load_all_for_date(date)

    books_df = to_pandas(data.get('books'))
    trades_df = to_pandas(data.get('trades'))
    prices_df = to_pandas(data.get('prices'))
    state_df = to_pandas(data.get('state'))
    events_df = to_pandas(data.get('events'))

    rprint("\n[bold]Executando análises...[/bold]\n") if HAS_RICH else print("\nExecutando análises...\n")

    # Todas as análises
    analyses = {
        '1_spread_temporal': analyze_spread_temporal(books_df),
        '2_trade_clustering': analyze_trade_clustering(trades_df),
        '3_price_impact': analyze_price_impact(trades_df, books_df),
        '4_chainlink_latency': analyze_chainlink_latency(prices_df),
        '5_regime_detection': analyze_regime_detection(state_df),
        '6_flow_prediction': analyze_flow_prediction(trades_df, events_df),
        '7_market_efficiency': analyze_market_efficiency(prices_df, trades_df),
    }

    # Print results
    for name, results in analyses.items():
        title = name.replace('_', ' ').title()

        if HAS_RICH:
            table = Table(title=f"📊 {title}", show_header=True, header_style="bold cyan")
            table.add_column("Métrica", style="white")
            table.add_column("Valor", justify="right", style="green")

            if 'error' in results:
                table.add_row("Error", str(results['error']))
            else:
                for key, value in results.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            table.add_row(f"  {k}", str(v))
                    else:
                        table.add_row(key, str(value))

            console.print(table)
            console.print()
        else:
            print(f"\n=== {title} ===")
            print(json.dumps(results, indent=2, default=str))

    # Save all results
    output_dir = loader.base_dir / 'analytics' / 'reports' / 'validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'date': date,
        'generated_at': datetime.now().isoformat(),
        'analyses': analyses,
    }

    output_file = output_dir / f'advanced_{date}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    if HAS_RICH:
        rprint(f"\n[green]💾 Salvo:[/green] {output_file}")
    else:
        print(f"\n💾 Salvo: {output_file}")

    return analyses


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python advanced_analysis.py <data>")
        print("Ex:  python advanced_analysis.py 2026-01-04")
        sys.exit(1)

    date = sys.argv[1]
    run_all_analyses(date)
