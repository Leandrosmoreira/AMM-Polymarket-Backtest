"""
EV Analysis - Análise de Expected Value do Paper Trading.

Gera relatório final com métricas de performance do bot.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Registro de um trade."""
    timestamp: float
    market: str
    price_up: float
    price_down: float
    pair_cost: float
    profit_pct: float
    order_size: float
    investment: float
    expected_profit: float
    balance_after: float
    ltm_bucket: Optional[int] = None


@dataclass
class ScanRecord:
    """Registro de um scan."""
    timestamp: float
    market: str
    up_ask: float
    down_ask: float
    pair_cost: float
    has_opportunity: bool
    time_remaining_sec: int


@dataclass
class EVAnalysis:
    """Resultado da análise EV."""
    total_trades: int
    total_investment: float
    total_expected_profit: float
    ev_per_trade: float
    roi: float
    average_profit_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    scans_logged: int
    opportunity_rate: float
    is_profitable: bool
    # Métricas adicionais para Market Maker
    spread_captured: float = 0.0  # Spread médio capturado por trade
    profit_per_trade: float = 0.0  # Lucro médio por trade
    success_rate: float = 0.0  # Taxa de sucesso (%)
    roi_by_market: Dict[str, float] = field(default_factory=dict)  # ROI por mercado
    # Métricas de Inventory
    inventory_by_market: Dict[str, dict] = field(default_factory=dict)  # Inventory por mercado
    total_inventory_value: float = 0.0  # Valor total do inventory
    total_unrealized_pnl: float = 0.0  # PnL não realizado total
    avg_inventory_risk: float = 0.0  # Risco médio de inventory
    # KPI de Saldo
    initial_balance: float = 0.0  # Saldo inicial
    current_balance: float = 0.0  # Saldo atual (inicial + lucro total)


class EVAnalyzer:
    """Analisador de EV para paper trading."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
    
    def load_inventory_timeline(self) -> Optional[List[dict]]:
        """
        Carrega timeline de inventory do arquivo JSONL mais recente.
        Se não encontrar, tenta carregar do CSV de snapshots.
        
        Returns:
            Lista de snapshots de inventory ou None
        """
        # Primeiro, tentar carregar do JSONL
        timeline_files = sorted(self.log_dir.glob("inventory_timeline_*.jsonl"), reverse=True)
        if timeline_files:
            timeline_file = timeline_files[0]
            snapshots = []
            
            try:
                import json
                with open(timeline_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            snapshots.append(json.loads(line))
                
                # Se encontrou snapshots no JSONL, usar
                if snapshots:
                    return snapshots
            except Exception as e:
                logger.warning(f"Error loading inventory timeline JSONL: {e}")
        
        # Se não encontrou JSONL ou está vazio, tentar carregar do CSV
        snapshot_files = sorted(self.log_dir.glob("inventory_snapshots_*.csv"), reverse=True)
        if snapshot_files:
            snapshot_file = snapshot_files[0]
            snapshots = []
            
            try:
                import csv
                with open(snapshot_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        snapshot = {
                            "timestamp": float(row.get('timestamp', 0)),
                            "cash": float(row.get('cash', 0)),
                            "total_inventory_value": float(row.get('total_inventory_value', 0)),
                            "total_invested": float(row.get('total_invested', 0)),
                            "unrealized_pnl": float(row.get('unrealized_pnl', 0)),
                            "inventory_risk": float(row.get('inventory_risk', 0)),
                            "delta_total": float(row.get('delta_total', 0)),
                            "positions": {}  # CSV não tem positions detalhadas
                        }
                        snapshots.append(snapshot)
                
                # Retornar apenas o último snapshot (mais recente)
                if snapshots:
                    return [snapshots[-1]]
            except Exception as e:
                logger.warning(f"Error loading inventory snapshots CSV: {e}")
        
        return None
    
    def find_latest_files(self, prefer_more_trades: bool = True) -> tuple[Optional[Path], Optional[Path]]:
        """
        Encontra os arquivos mais recentes de trades e scans.
        
        Ordena por timestamp no nome do arquivo (mais confiável que mtime).
        Se prefer_more_trades=True, prioriza arquivos com mais trades.
        
        Args:
            prefer_more_trades: Se True, prioriza arquivo com mais trades entre os mais recentes
        
        Returns:
            (trades_file, scans_file) ou (None, None) se não encontrar
        """
        import re
        import csv
        
        # Buscar arquivos de trades
        trades_files = list(self.log_dir.glob("trades_*.csv"))
        
        if not trades_files:
            return None, None
        
        # Ordenar por timestamp no nome (mais confiável)
        def extract_timestamp(path: Path) -> str:
            match = re.search(r'trades_(\d{8})_(\d{6})', path.name)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
            # Fallback: usar mtime como string
            return str(path.stat().st_mtime)
        
        trades_files_sorted = sorted(trades_files, key=extract_timestamp, reverse=True)
        
        # Se prefer_more_trades, verificar quantos trades cada arquivo tem
        if prefer_more_trades and len(trades_files_sorted) > 1:
            # Pegar arquivos das últimas 24h OU os 10 mais recentes (o que for maior)
            from datetime import datetime, timedelta
            now = datetime.now()
            cutoff_time = now - timedelta(hours=24)
            
            recent_files = []
            for f in trades_files_sorted:
                # Verificar se está nas últimas 24h ou é um dos 10 mais recentes
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime >= cutoff_time or len(recent_files) < 10:
                    recent_files.append(f)
                else:
                    break
            
            # Se não encontrou arquivos recentes, usar os 10 mais recentes
            if not recent_files:
                recent_files = trades_files_sorted[:10]
            
            file_trade_counts = []
            
            for f in recent_files:
                try:
                    with open(f, 'r') as file:
                        reader = csv.DictReader(file)
                        count = len(list(reader))
                    file_trade_counts.append((f, count))
                except Exception:
                    file_trade_counts.append((f, 0))
            
            # Ordenar por número de trades (decrescente), depois por timestamp
            file_trade_counts.sort(key=lambda x: (x[1], extract_timestamp(x[0])), reverse=True)
            
            # Escolher o arquivo com mais trades entre os recentes
            if file_trade_counts and file_trade_counts[0][1] > 0:
                trades_file = file_trade_counts[0][0]
            else:
                trades_file = trades_files_sorted[0]
        else:
            trades_file = trades_files_sorted[0]
        
        # Buscar arquivo de scans correspondente
        if trades_file:
            # Extrair timestamp do arquivo de trades
            match = re.search(r'trades_(\d{8})_(\d{6})', trades_file.name)
            if match:
                scan_pattern = f"scans_{match.group(1)}_{match.group(2)}.csv"
                scans_file = self.log_dir / scan_pattern
                if not scans_file.exists():
                    # Buscar scans mais recente
                    scans_files = sorted(self.log_dir.glob("scans_*.csv"), reverse=True)
                    scans_file = scans_files[0] if scans_files else None
            else:
                scans_file = None
        else:
            scans_file = None
        
        return trades_file, scans_file
    
    def load_trades(self, trades_file: Path) -> List[TradeRecord]:
        """Carrega trades de um arquivo CSV."""
        trades = []
        
        try:
            with open(trades_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        trade = TradeRecord(
                            timestamp=float(row['timestamp']),
                            market=row['market'],
                            price_up=float(row['price_up']),
                            price_down=float(row['price_down']),
                            pair_cost=float(row['pair_cost']),
                            profit_pct=float(row['profit_pct']),
                            order_size=float(row['order_size']),
                            investment=float(row['investment']),
                            expected_profit=float(row['expected_profit']),
                            balance_after=float(row['balance_after']),
                            ltm_bucket=int(row['ltm_bucket']) if row.get('ltm_bucket') and row['ltm_bucket'].strip() else None
                        )
                        trades.append(trade)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing trade row: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Trades file not found: {trades_file}")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
        
        return trades
    
    def load_scans(self, scans_file: Path) -> List[ScanRecord]:
        """Carrega scans de um arquivo CSV."""
        scans = []

        try:
            with open(scans_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Converter time_remaining_sec: pode vir como float (ex: '43.0'), converter para int
                        time_remaining = row.get('time_remaining_sec', '0')
                        if isinstance(time_remaining, str):
                            # Se for string, tentar converter float primeiro, depois int
                            time_remaining = int(float(time_remaining))
                        else:
                            time_remaining = int(time_remaining)
                        
                        scan = ScanRecord(
                            timestamp=float(row['timestamp']),
                            market=row['market'],
                            up_ask=float(row['up_ask']),
                            down_ask=float(row['down_ask']),
                            pair_cost=float(row['pair_cost']),
                            has_opportunity=bool(int(row['has_opportunity'])),
                            time_remaining_sec=time_remaining
                        )
                        scans.append(scan)
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Error parsing scan row: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"Scans file not found: {scans_file}")
        except Exception as e:
            logger.error(f"Error loading scans: {e}")

        return scans
    
    def analyze(self, trades_file: Optional[Path] = None, scans_file: Optional[Path] = None) -> Optional[EVAnalysis]:
        """
        Analisa trades e scans e retorna métricas.
        
        Args:
            trades_file: Caminho do arquivo de trades (None = auto-detect)
            scans_file: Caminho do arquivo de scans (None = auto-detect)
        
        Returns:
            EVAnalysis ou None se não houver dados
        """
        # Auto-detect files if not provided
        if trades_file is None or scans_file is None:
            auto_trades, auto_scans = self.find_latest_files()
            trades_file = trades_file or auto_trades
            scans_file = scans_file or auto_scans
        
        if trades_file is None:
            logger.error("No trades file found")
            return None
        
        # Load data
        trades = self.load_trades(trades_file)
        scans = self.load_scans(scans_file) if scans_file else []
        
        if not trades:
            logger.warning("No trades found in file")
            return None
        
        # Calculate metrics
        total_trades = len(trades)
        total_investment = sum(t.investment for t in trades)
        total_expected_profit = sum(t.expected_profit for t in trades)
        
        # Calcular saldo inicial e atual
        initial_balance = 0.0
        current_balance = 0.0
        
        if trades:
            first_trade = trades[0]
            last_trade = trades[-1]
            
            # Calcular saldo inicial a partir do primeiro trade
            # Se balance_after representa o saldo após o trade:
            # Saldo antes do primeiro trade = balance_after + investment - expected_profit
            if first_trade.balance_after > 0:
                # Saldo inicial = saldo após primeiro trade + investimento do primeiro - lucro do primeiro
                initial_balance = first_trade.balance_after + first_trade.investment - first_trade.expected_profit
            else:
                # Fallback: usar último trade para estimar
                if last_trade.balance_after > 0:
                    # Saldo inicial = último balance - lucro total + investimento total
                    initial_balance = last_trade.balance_after - total_expected_profit + total_investment
                else:
                    # Último fallback: assumir baseado no investimento (10% de utilização)
                    initial_balance = total_investment / 0.1 if total_investment > 0 else 10000.0
            
            # Saldo atual = saldo inicial + lucro total de todos os trades
            current_balance = initial_balance + total_expected_profit
        
        ev_per_trade = total_expected_profit / total_trades if total_trades > 0 else 0.0
        roi = (total_expected_profit / total_investment * 100) if total_investment > 0 else 0.0
        average_profit_pct = sum(t.profit_pct for t in trades) / total_trades if total_trades > 0 else 0.0
        
        best_trade_pct = max(t.profit_pct for t in trades) if trades else 0.0
        worst_trade_pct = min(t.profit_pct for t in trades) if trades else 0.0
        
        scans_logged = len(scans)
        opportunity_rate = (total_trades / scans_logged * 100) if scans_logged > 0 else 0.0
        
        is_profitable = ev_per_trade > 0
        
        # Métricas adicionais para Market Maker (calcular spread capturado)
        spread_captured = 0.0
        profit_per_trade = 0.0
        success_rate = 0.0
        roi_by_market = {}
        
        if total_trades > 0:
            # Agrupar trades por mercado para calcular ROI por mercado
            trades_by_market = {}
            for trade in trades:
                market = trade.market
                if market not in trades_by_market:
                    trades_by_market[market] = {'buys': [], 'sells': [], 'buy_investments': [], 'sell_revenues': []}
                
                # Identificar se é BUY ou SELL baseado em price_up/price_down
                # BUY: price_up > 0, price_down = 0
                # SELL: price_down > 0, price_up = 0
                if trade.price_up > 0 and trade.price_down == 0 and trade.investment > 0:
                    trades_by_market[market]['buys'].append(trade.price_up)
                    trades_by_market[market]['buy_investments'].append(trade.investment)
                elif trade.price_down > 0 and trade.price_up == 0 and trade.investment > 0:
                    trades_by_market[market]['sells'].append(trade.price_down)
                    trades_by_market[market]['sell_revenues'].append(trade.investment)
            
            # Calcular spread capturado e ROI por mercado
            total_profit = 0.0
            profitable_markets = 0
            total_spread = 0.0
            spread_count = 0
            
            for market, data in trades_by_market.items():
                buys = data['buys']
                sells = data['sells']
                buy_investments = data['buy_investments']
                sell_revenues = data['sell_revenues']
                
                if buys and sells:
                    # Calcular spread médio para este mercado
                    avg_buy = sum(buys) / len(buys)
                    avg_sell = sum(sells) / len(sells)
                    market_spread = avg_sell - avg_buy
                    
                    # Calcular lucro total do mercado
                    buy_cost = sum(buy_investments)
                    sell_revenue = sum(sell_revenues)
                    market_profit = sell_revenue - buy_cost
                    
                    # Calcular ROI do mercado
                    market_roi = (market_profit / buy_cost * 100) if buy_cost > 0 else 0.0
                    roi_by_market[market] = market_roi
                    
                    total_profit += market_profit
                    if market_profit > 0:
                        profitable_markets += 1
                    
                    # Spread capturado (acumular para média)
                    total_spread += market_spread
                    spread_count += 1
            
            # Calcular métricas finais
            if spread_count > 0:
                spread_captured = total_spread / spread_count
            
            if total_trades > 0:
                profit_per_trade = total_profit / total_trades
                success_rate = (profitable_markets / len(trades_by_market) * 100) if trades_by_market else 0.0
                
                # Se expected_profit está zerado mas temos lucro real, usar lucro real
                if total_expected_profit == 0 and total_profit > 0:
                    total_expected_profit = total_profit
                    ev_per_trade = profit_per_trade
                    roi = (total_profit / total_investment * 100) if total_investment > 0 else 0.0
                    is_profitable = total_profit > 0
        
        # Carregar métricas de inventory
        inventory_by_market = {}
        total_inventory_value = 0.0
        total_unrealized_pnl = 0.0
        avg_inventory_risk = 0.0
        
        inventory_timeline = self.load_inventory_timeline()
        if inventory_timeline:
            # Pegar último snapshot
            last_snapshot = inventory_timeline[-1] if inventory_timeline else None
            if last_snapshot:
                total_inventory_value = last_snapshot.get('total_inventory_value', 0.0)
                total_unrealized_pnl = last_snapshot.get('unrealized_pnl', 0.0)
                avg_inventory_risk = last_snapshot.get('inventory_risk', 0.0)
                
                # Inventory por mercado
                positions = last_snapshot.get('positions', {})
                for market_id, pos_data in positions.items():
                    if isinstance(pos_data, dict):
                        inventory_by_market[market_id] = {
                            'yes_shares': pos_data.get('yes_shares', 0.0),
                            'no_shares': pos_data.get('no_shares', 0.0),
                            'inventory_value': pos_data.get('yes_shares', 0.0) * 0.5 + pos_data.get('no_shares', 0.0) * 0.5,  # Aproximação
                            'delta': pos_data.get('yes_shares', 0.0) - pos_data.get('no_shares', 0.0),
                        }
        
        return EVAnalysis(
            total_trades=total_trades,
            total_investment=total_investment,
            total_expected_profit=total_expected_profit,
            ev_per_trade=ev_per_trade,
            roi=roi,
            average_profit_pct=average_profit_pct,
            best_trade_pct=best_trade_pct,
            worst_trade_pct=worst_trade_pct,
            scans_logged=scans_logged,
            opportunity_rate=opportunity_rate,
            is_profitable=is_profitable,
            spread_captured=spread_captured,
            profit_per_trade=profit_per_trade,
            success_rate=success_rate,
            roi_by_market=roi_by_market if roi_by_market else {},
            inventory_by_market=inventory_by_market,
            total_inventory_value=total_inventory_value,
            total_unrealized_pnl=total_unrealized_pnl,
            avg_inventory_risk=avg_inventory_risk,
            initial_balance=initial_balance,
            current_balance=current_balance
        )
    
    def format_report(self, analysis: EVAnalysis) -> str:
        """
        Formata relatório de análise.
        
        Args:
            analysis: Resultado da análise
        
        Returns:
            String formatada do relatório
        """
        lines = []
        lines.append("=" * 60)
        lines.append("    BOT EV ANALYSIS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"    Total Trades:           {analysis.total_trades:>6}")
        lines.append(f"    Total Investment:       ${analysis.total_investment:>10.2f}")
        lines.append(f"    Total Expected Profit:  ${analysis.total_expected_profit:>10.2f}")
        lines.append("")
        lines.append(f"    EV per Trade:           ${analysis.ev_per_trade:>10.4f}")
        lines.append(f"    ROI:                     {analysis.roi:>10.2f}%")
        lines.append(f"    Average Profit %:       {analysis.average_profit_pct:>10.2f}%")
        lines.append("")
        lines.append(f"    Best Trade:             {analysis.best_trade_pct:>10.2f}% profit")
        lines.append(f"    Worst Trade:            {analysis.worst_trade_pct:>10.2f}% profit")
        lines.append("")
        lines.append(f"    Scans Logged:           {analysis.scans_logged:>6}")
        lines.append(f"    Opportunity Rate:       {analysis.opportunity_rate:>10.1f}%")
        lines.append("")
        lines.append("    BALANCE KPI:")
        lines.append("    " + "-" * 56)
        lines.append(f"    Initial Balance:        ${analysis.initial_balance:>10.2f}")
        lines.append(f"    Total Profit:           ${analysis.total_expected_profit:>10.2f}")
        lines.append(f"    Current Balance:        ${analysis.current_balance:>10.2f}")
        lines.append("=" * 60)
        
        # Métricas adicionais para Market Maker
        if analysis.spread_captured > 0 or analysis.profit_per_trade > 0:
            lines.append("")
            lines.append("    MARKET MAKER METRICS:")
            lines.append("    " + "-" * 56)
            lines.append(f"    Spread Captured:        ${analysis.spread_captured:>10.4f} per trade")
            lines.append(f"    Profit per Trade:      ${analysis.profit_per_trade:>10.2f}")
            lines.append(f"    Success Rate:           {analysis.success_rate:>10.1f}%")
            
            # ROI por mercado
            if analysis.roi_by_market:
                lines.append("")
                lines.append("    ROI BY MARKET:")
                lines.append("    " + "-" * 56)
                for market, market_roi in sorted(analysis.roi_by_market.items()):
                    lines.append(f"    Market {market}:         {market_roi:>10.2f}%")
        
        # Métricas de Inventory
        if analysis.total_inventory_value > 0 or analysis.inventory_by_market:
            lines.append("")
            lines.append("    INVENTORY ANALYSIS:")
            lines.append("    " + "-" * 56)
            lines.append(f"    Total Inventory Value:      ${analysis.total_inventory_value:>10.2f}")
            lines.append(f"    Total Unrealized PnL:       ${analysis.total_unrealized_pnl:>10.2f}")
            lines.append(f"    Avg Inventory Risk:         {analysis.avg_inventory_risk:>10.2f}")
            
            # Inventory por mercado
            if analysis.inventory_by_market:
                lines.append("")
                lines.append("    INVENTORY BY MARKET:")
                lines.append("    " + "-" * 56)
                for market, inv_data in sorted(analysis.inventory_by_market.items()):
                    lines.append(f"    Market {market}:")
                    lines.append(f"      Inventory Value:          ${inv_data.get('inventory_value', 0):>10.2f}")
                    lines.append(f"      YES Shares:               {inv_data.get('yes_shares', 0):>10.2f}")
                    lines.append(f"      NO Shares:                {inv_data.get('no_shares', 0):>10.2f}")
                    lines.append(f"      Delta:                    {inv_data.get('delta', 0):>10.2f}")
        
        lines.append("=" * 60)
        lines.append("")
        
        if analysis.is_profitable:
            lines.append("✅ POSITIVE EV - Bot is profitable!")
            if analysis.profit_per_trade > 0:
                lines.append(f"   Expected to make ${analysis.profit_per_trade:.2f} per trade on average.")
            else:
                lines.append(f"   Expected to make ${analysis.ev_per_trade:.4f} per trade on average.")
        else:
            lines.append("❌ NEGATIVE EV - Bot is not profitable.")
            lines.append(f"   Expected to lose ${abs(analysis.ev_per_trade):.4f} per trade on average.")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def print_report(self, analysis: Optional[EVAnalysis] = None):
        """
        Imprime relatório de análise.
        
        Args:
            analysis: Resultado da análise (None = auto-analyze)
        """
        if analysis is None:
            analysis = self.analyze()
        
        if analysis is None:
            print("❌ No data found for analysis.")
            return
        
        print(self.format_report(analysis))


def main():
    """Entry point para análise EV."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze paper trading EV")
    parser.add_argument("--trades", type=str, help="Path to trades CSV file")
    parser.add_argument("--scans", type=str, help="Path to scans CSV file")
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--output", type=str, help="Output file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create analyzer
    analyzer = EVAnalyzer(log_dir=args.log_dir)
    
    # Analyze
    trades_file = Path(args.trades) if args.trades else None
    scans_file = Path(args.scans) if args.scans else None
    
    analysis = analyzer.analyze(trades_file=trades_file, scans_file=scans_file)
    
    if analysis is None:
        print("❌ No data found for analysis.")
        return 1
    
    # Format and print
    report = analyzer.format_report(analysis)
    print(report)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

