"""
Inventory Model - Rastreamento completo de inventário para paper trading.

Rastreia posições YES/NO por mercado, calcula métricas de risco,
e integra com análise EV.
"""

import time
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class InventoryPosition:
    """Posição de inventário em um mercado."""
    market_id: str
    yes_shares: float = 0.0      # Quantidade de YES tokens
    no_shares: float = 0.0       # Quantidade de NO tokens
    yes_avg_price: float = 0.0   # Preço médio de compra YES
    no_avg_price: float = 0.0    # Preço médio de compra NO
    yes_cost: float = 0.0        # Custo total investido em YES
    no_cost: float = 0.0         # Custo total investido em NO
    last_update: float = 0.0     # Timestamp da última atualização

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return asdict(self)


@dataclass
class InventorySnapshot:
    """Snapshot completo do inventário em um momento."""
    timestamp: float
    cash: float                  # Cash disponível
    positions: Dict[str, InventoryPosition]  # Por mercado
    total_inventory_value: float  # Valor total do inventory
    total_invested: float         # Total investido
    unrealized_pnl: float         # PnL não realizado
    inventory_risk: float         # Risco de inventory (exposição direcional)
    delta_total: float            # Delta total (soma de todos os mercados)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "timestamp": self.timestamp,
            "cash": self.cash,
            "positions": {m: p.to_dict() for m, p in self.positions.items()},
            "total_inventory_value": self.total_inventory_value,
            "total_invested": self.total_invested,
            "unrealized_pnl": self.unrealized_pnl,
            "inventory_risk": self.inventory_risk,
            "delta_total": self.delta_total,
        }


class InventoryManager:
    """
    Gerencia inventário completo do bot.
    
    Rastreia:
    - Posições YES/NO por mercado
    - Cash disponível
    - PnL não realizado
    - Inventory risk
    - Timeline de snapshots
    """
    
    def __init__(
        self,
        initial_cash: float = 10000.0,
        log_dir: str = "logs",
    ):
        """
        Inicializa o InventoryManager.
        
        Args:
            initial_cash: Cash inicial disponível
            log_dir: Diretório para salvar logs
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Posições por mercado: {market_id: InventoryPosition}
        self.positions: Dict[str, InventoryPosition] = {}
        
        # Timeline de snapshots
        self.snapshots: List[InventorySnapshot] = []
        
        # Estatísticas
        self.total_fills = 0
        self.total_invested = 0.0
        self.total_realized_pnl = 0.0
        
        logger.info(f"InventoryManager initialized with ${initial_cash:.2f} cash")
    
    def update_on_fill(
        self,
        market_id: str,
        side: str,  # "BUY" ou "SELL"
        price: float,
        size: float,
        order_id: str = None,
    ):
        """
        Atualiza inventory após um fill.
        
        Args:
            market_id: ID do mercado
            side: "BUY" ou "SELL"
            price: Preço de execução
            size: Tamanho executado
            order_id: ID da ordem (opcional)
        """
        # Inicializar posição se não existir
        if market_id not in self.positions:
            self.positions[market_id] = InventoryPosition(
                market_id=market_id,
                last_update=time.time()
            )
        
        pos = self.positions[market_id]
        cost = price * size
        
        # Determinar se é YES ou NO baseado no side
        # BUY = compra de YES (assumindo mercado YES/NO padrão)
        # SELL = venda de YES (ou compra de NO)
        # Para simplificar, assumimos que BUY = YES, SELL = NO
        
        if side == "BUY":
            # Compra de YES
            # Atualizar preço médio
            if pos.yes_shares > 0:
                total_cost = pos.yes_cost + cost
                new_shares = pos.yes_shares + size
                pos.yes_avg_price = total_cost / new_shares if new_shares > 0 else 0.0
            else:
                pos.yes_avg_price = price
            
            pos.yes_shares += size
            pos.yes_cost += cost
            self.cash -= cost
            
        elif side == "SELL":
            # Venda de YES (ou compra de NO)
            # Assumindo que SELL = venda de YES
            if pos.yes_shares >= size:
                # Vender YES existente
                pos.yes_shares -= size
                pos.yes_cost -= (pos.yes_avg_price * size) if pos.yes_avg_price > 0 else 0
                self.cash += cost
            else:
                # Vender mais do que tem (short)
                pos.yes_shares -= size
                pos.yes_cost -= (pos.yes_avg_price * abs(pos.yes_shares)) if pos.yes_avg_price > 0 else 0
                self.cash += cost
        
        pos.last_update = time.time()
        self.total_fills += 1
        self.total_invested += cost
        
        logger.debug(
            f"Inventory updated: {market_id} {side} {size} @ ${price:.4f} | "
            f"YES={pos.yes_shares:.2f} NO={pos.no_shares:.2f} | "
            f"Cash=${self.cash:.2f}"
        )
    
    def get_inventory_value(self, market_id: str, current_price: float = None) -> float:
        """
        Calcula valor total do inventory em um mercado.
        
        Args:
            market_id: ID do mercado
            current_price: Preço atual do mercado (opcional)
        
        Returns:
            Valor total do inventory em USD
        """
        if market_id not in self.positions:
            return 0.0
        
        pos = self.positions[market_id]
        
        # Se não fornecido, usar preço médio
        if current_price is None:
            current_price = pos.yes_avg_price if pos.yes_avg_price > 0 else 0.5
        
        # Valor do inventory = (YES_shares * price) + (NO_shares * (1 - price))
        yes_value = pos.yes_shares * current_price if pos.yes_shares > 0 else 0.0
        no_value = pos.no_shares * (1.0 - current_price) if pos.no_shares > 0 else 0.0
        
        return yes_value + no_value
    
    def get_delta(self, market_id: str) -> float:
        """
        Calcula delta (exposição direcional) para um mercado.
        
        Returns:
            Delta = YES_shares - NO_shares
        """
        if market_id not in self.positions:
            return 0.0
        
        pos = self.positions[market_id]
        return pos.yes_shares - pos.no_shares
    
    def get_total_delta(self) -> float:
        """Calcula delta total de todos os mercados."""
        return sum(self.get_delta(m) for m in self.positions)
    
    def get_unrealized_pnl(self, market_id: str, current_price: float) -> float:
        """
        Calcula PnL não realizado para um mercado.
        
        Args:
            market_id: ID do mercado
            current_price: Preço atual do mercado
        
        Returns:
            PnL não realizado em USD
        """
        if market_id not in self.positions:
            return 0.0
        
        pos = self.positions[market_id]
        
        # PnL de YES
        yes_pnl = 0.0
        if pos.yes_shares > 0 and pos.yes_avg_price > 0:
            yes_pnl = (current_price - pos.yes_avg_price) * pos.yes_shares
        
        # PnL de NO
        no_pnl = 0.0
        if pos.no_shares > 0 and pos.no_avg_price > 0:
            no_pnl = ((1.0 - current_price) - (1.0 - pos.no_avg_price)) * pos.no_shares
        
        return yes_pnl + no_pnl
    
    def get_inventory_risk(self, market_id: str, current_price: float = None) -> float:
        """
        Calcula score de risco de inventory (0-100).
        
        Baseado em:
        - Tamanho do delta relativo ao inventory
        - Concentração de posição
        
        Returns:
            Score de risco (0 = baixo risco, 100 = alto risco)
        """
        if market_id not in self.positions:
            return 0.0
        
        pos = self.positions[market_id]
        inventory_value = self.get_inventory_value(market_id, current_price)
        
        if inventory_value == 0:
            return 0.0
        
        # Delta relativo
        delta = self.get_delta(market_id)
        delta_ratio = abs(delta) / max(abs(pos.yes_shares) + abs(pos.no_shares), 1.0)
        
        # Score de risco (0-100)
        risk_score = min(100.0, delta_ratio * 100.0)
        
        return risk_score
    
    def get_snapshot(self, current_prices: Dict[str, float] = None) -> InventorySnapshot:
        """
        Gera snapshot completo do inventário.
        
        Args:
            current_prices: Dict {market_id: current_price} (opcional)
        
        Returns:
            InventorySnapshot
        """
        if current_prices is None:
            current_prices = {}
        
        # Calcular métricas totais
        total_inventory_value = 0.0
        total_unrealized_pnl = 0.0
        total_invested = 0.0
        
        for market_id, pos in self.positions.items():
            current_price = current_prices.get(market_id, pos.yes_avg_price if pos.yes_avg_price > 0 else 0.5)
            market_value = self.get_inventory_value(market_id, current_price)
            market_pnl = self.get_unrealized_pnl(market_id, current_price)
            
            total_inventory_value += market_value
            total_unrealized_pnl += market_pnl
            total_invested += pos.yes_cost + pos.no_cost
        
        # Inventory risk médio
        avg_risk = 0.0
        if self.positions:
            risks = [
                self.get_inventory_risk(m, current_prices.get(m, 0.5))
                for m in self.positions
            ]
            avg_risk = sum(risks) / len(risks) if risks else 0.0
        
        return InventorySnapshot(
            timestamp=time.time(),
            cash=self.cash,
            positions=self.positions.copy(),
            total_inventory_value=total_inventory_value,
            total_invested=total_invested,
            unrealized_pnl=total_unrealized_pnl,
            inventory_risk=avg_risk,
            delta_total=self.get_total_delta(),
        )
    
    def save_snapshot(self, snapshot: InventorySnapshot = None):
        """
        Salva snapshot em arquivo CSV.
        
        Args:
            snapshot: Snapshot a salvar (None = gerar novo)
        """
        if snapshot is None:
            snapshot = self.get_snapshot()
        
        # Adicionar à timeline
        self.snapshots.append(snapshot)
        
        # Salvar em CSV
        date_str = datetime.now().strftime("%Y%m%d")
        csv_file = self.log_dir / f"inventory_snapshots_{date_str}.csv"
        
        # Cabeçalho
        header = [
            "timestamp", "cash", "total_inventory_value", "total_invested",
            "unrealized_pnl", "inventory_risk", "delta_total", "num_markets"
        ]
        
        # Escrever cabeçalho se arquivo não existe
        file_exists = csv_file.exists()
        
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            
            # Escrever linha
            row = [
                snapshot.timestamp,
                snapshot.cash,
                snapshot.total_inventory_value,
                snapshot.total_invested,
                snapshot.unrealized_pnl,
                snapshot.inventory_risk,
                snapshot.delta_total,
                len(snapshot.positions),
            ]
            writer.writerow(row)
    
    def export_timeline(self, output_file: Optional[str] = None) -> str:
        """
        Exporta timeline completa de inventory para JSON.
        
        Args:
            output_file: Arquivo de saída (None = auto)
        
        Returns:
            Caminho do arquivo gerado
        """
        if output_file is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_file = str(self.log_dir / f"inventory_timeline_{date_str}.jsonl")
        
        with open(output_file, "w") as f:
            for snapshot in self.snapshots:
                f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Inventory timeline exported to {output_file}")
        return output_file
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do inventory manager."""
        snapshot = self.get_snapshot()
        
        return {
            "initial_cash": self.initial_cash,
            "current_cash": self.cash,
            "total_inventory_value": snapshot.total_inventory_value,
            "total_invested": snapshot.total_invested,
            "unrealized_pnl": snapshot.unrealized_pnl,
            "total_delta": snapshot.delta_total,
            "inventory_risk": snapshot.inventory_risk,
            "num_markets": len(self.positions),
            "total_fills": self.total_fills,
            "positions": {
                m: {
                    "yes_shares": p.yes_shares,
                    "no_shares": p.no_shares,
                    "delta": self.get_delta(m),
                    "inventory_value": self.get_inventory_value(m),
                }
                for m, p in self.positions.items()
            }
        }


# Standalone test
if __name__ == "__main__":
    manager = InventoryManager(initial_cash=10000.0)
    
    # Simular fills
    print("=== Simulando fills ===")
    
    manager.update_on_fill("market1", "BUY", 0.50, 10.0)
    print(f"Comprou 10 YES @ $0.50")
    print(f"Inventory: ${manager.get_inventory_value('market1', 0.50):.2f}")
    print(f"Cash: ${manager.cash:.2f}")
    
    manager.update_on_fill("market1", "SELL", 0.52, 10.0)
    print(f"\nVendeu 10 YES @ $0.52")
    print(f"Inventory: ${manager.get_inventory_value('market1', 0.52):.2f}")
    print(f"Cash: ${manager.cash:.2f}")
    
    snapshot = manager.get_snapshot({"market1": 0.52})
    print(f"\nSnapshot:")
    print(f"  Total Inventory: ${snapshot.total_inventory_value:.2f}")
    print(f"  Unrealized PnL: ${snapshot.unrealized_pnl:.2f}")
    print(f"  Delta: {snapshot.delta_total:.2f}")

