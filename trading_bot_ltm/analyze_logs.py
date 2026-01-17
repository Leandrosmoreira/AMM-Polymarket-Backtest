"""
Analyze trading bot logs and generate EV (Expected Value) charts.

Usage:
    python -m trading_bot_ltm.analyze_logs [logs_dir]

Example:
    python -m trading_bot_ltm.analyze_logs logs/
"""
import sys
import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_trades(logs_dir: str = "logs") -> pd.DataFrame:
    """Load all trade CSV files."""
    files = glob.glob(f"{logs_dir}/trades_*.csv")
    if not files:
        print(f"No trade files found in {logs_dir}/")
        return pd.DataFrame()

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
                print(f"Loaded {len(df)} trades from {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_scans(logs_dir: str = "logs") -> pd.DataFrame:
    """Load all scan CSV files."""
    files = glob.glob(f"{logs_dir}/scans_*.csv")
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def analyze_ev(trades: pd.DataFrame) -> dict:
    """Calculate Expected Value metrics."""
    if trades.empty:
        return {}

    total_trades = len(trades)
    total_investment = trades['investment'].sum()
    total_expected_profit = trades['expected_profit'].sum()

    # EV per trade
    ev_per_trade = total_expected_profit / total_trades if total_trades > 0 else 0

    # ROI
    roi = (total_expected_profit / total_investment * 100) if total_investment > 0 else 0

    # Average profit percentage
    avg_profit_pct = trades['profit_pct'].mean()

    # Win rate (all trades are theoretically winners in arbitrage)
    win_rate = 100.0  # Arbitrage = guaranteed profit

    # Best and worst trades
    best_trade = trades.loc[trades['profit_pct'].idxmax()] if not trades.empty else None
    worst_trade = trades.loc[trades['profit_pct'].idxmin()] if not trades.empty else None

    return {
        'total_trades': total_trades,
        'total_investment': total_investment,
        'total_expected_profit': total_expected_profit,
        'ev_per_trade': ev_per_trade,
        'roi_pct': roi,
        'avg_profit_pct': avg_profit_pct,
        'win_rate': win_rate,
        'best_profit_pct': best_trade['profit_pct'] if best_trade is not None else 0,
        'worst_profit_pct': worst_trade['profit_pct'] if worst_trade is not None else 0,
    }


def plot_analysis(trades: pd.DataFrame, scans: pd.DataFrame, output_file: str = "ev_analysis.png"):
    """Generate analysis charts."""
    if trades.empty:
        print("No trades to analyze!")
        return

    # Convert timestamp to datetime
    trades['datetime'] = pd.to_datetime(trades['timestamp'], unit='s')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Bot EV Analysis - Arbitrage Performance', fontsize=14, fontweight='bold')

    # 1. Cumulative Profit Over Time
    ax1 = axes[0, 0]
    trades['cumulative_profit'] = trades['expected_profit'].cumsum()
    ax1.plot(trades['datetime'], trades['cumulative_profit'], 'g-', linewidth=2)
    ax1.fill_between(trades['datetime'], 0, trades['cumulative_profit'], alpha=0.3, color='green')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cumulative Profit ($)')
    ax1.set_title('Cumulative Expected Profit')
    ax1.grid(True, alpha=0.3)

    # 2. Profit % Distribution
    ax2 = axes[0, 1]
    ax2.hist(trades['profit_pct'], bins=20, color='blue', edgecolor='black', alpha=0.7)
    ax2.axvline(trades['profit_pct'].mean(), color='red', linestyle='--', label=f"Mean: {trades['profit_pct'].mean():.2f}%")
    ax2.set_xlabel('Profit (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Profit % Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Balance Evolution
    ax3 = axes[0, 2]
    ax3.plot(trades['datetime'], trades['balance_after'], 'b-', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Balance ($)')
    ax3.set_title('Balance Evolution')
    ax3.grid(True, alpha=0.3)

    # 4. Pair Cost Distribution
    ax4 = axes[1, 0]
    ax4.hist(trades['pair_cost'], bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax4.axvline(0.991, color='red', linestyle='--', label='Threshold: $0.991')
    ax4.set_xlabel('Pair Cost ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Pair Cost Distribution (Entry Points)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Investment vs Profit
    ax5 = axes[1, 1]
    ax5.scatter(trades['investment'], trades['expected_profit'], alpha=0.6, c='green', edgecolors='black')
    ax5.set_xlabel('Investment ($)')
    ax5.set_ylabel('Expected Profit ($)')
    ax5.set_title('Investment vs Expected Profit')
    ax5.grid(True, alpha=0.3)

    # 6. EV Summary Stats
    ax6 = axes[1, 2]
    ax6.axis('off')

    stats = analyze_ev(trades)
    summary_text = f"""
    ═══════════════════════════════════════
           EXPECTED VALUE (EV) SUMMARY
    ═══════════════════════════════════════

    Total Trades:         {stats['total_trades']}
    Total Investment:     ${stats['total_investment']:.2f}
    Total Expected Profit: ${stats['total_expected_profit']:.2f}

    ───────────────────────────────────────
    EV per Trade:         ${stats['ev_per_trade']:.4f}
    ROI:                  {stats['roi_pct']:.2f}%
    Avg Profit %:         {stats['avg_profit_pct']:.2f}%
    Win Rate:             {stats['win_rate']:.1f}%
    ───────────────────────────────────────

    Best Trade:           {stats['best_profit_pct']:.2f}% profit
    Worst Trade:          {stats['worst_profit_pct']:.2f}% profit

    ═══════════════════════════════════════
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
             verticalalignment='center', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Chart saved to: {output_file}")

    # Also show if in interactive mode
    try:
        plt.show()
    except:
        pass


def print_summary(trades: pd.DataFrame, scans: pd.DataFrame):
    """Print text summary."""
    print("\n" + "=" * 60)
    print("           BOT EV ANALYSIS SUMMARY")
    print("=" * 60)

    if trades.empty:
        print("No trades found!")
        return

    stats = analyze_ev(trades)

    print(f"""
    Total Trades:           {stats['total_trades']}
    Total Investment:       ${stats['total_investment']:.2f}
    Total Expected Profit:  ${stats['total_expected_profit']:.2f}

    EV per Trade:           ${stats['ev_per_trade']:.4f}
    ROI:                    {stats['roi_pct']:.2f}%
    Average Profit %:       {stats['avg_profit_pct']:.2f}%

    Best Trade:             {stats['best_profit_pct']:.2f}% profit
    Worst Trade:            {stats['worst_profit_pct']:.2f}% profit
    """)

    if not scans.empty:
        total_scans = len(scans)
        opportunities = scans[scans['has_opportunity'] == 1]
        opp_rate = len(opportunities) / total_scans * 100 if total_scans > 0 else 0
        print(f"    Scans Logged:           {total_scans}")
        print(f"    Opportunity Rate:       {opp_rate:.1f}%")

    print("=" * 60)

    # EV Verdict
    if stats['ev_per_trade'] > 0:
        print("\n✅ POSITIVE EV - Bot is profitable!")
        print(f"   Expected to make ${stats['ev_per_trade']:.4f} per trade on average.")
    else:
        print("\n❌ NEGATIVE EV - Bot is not profitable!")

    print()


def main():
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs"

    print(f"Loading logs from: {logs_dir}/")

    trades = load_trades(logs_dir)
    scans = load_scans(logs_dir)

    if trades.empty:
        print("\nNo trade data found. Run the bot first to generate logs.")
        print("Usage: python -m trading_bot_ltm.analyze_logs [logs_dir]")
        return

    print_summary(trades, scans)

    # Generate charts
    output_file = f"{logs_dir}/ev_analysis.png"
    plot_analysis(trades, scans, output_file)


if __name__ == "__main__":
    main()
