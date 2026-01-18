"""
Structured Logger - Logs em formato JSON estruturado.
NÍVEL 5: Log estruturado para análise posterior.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = None  # Será inicializado quando necessário


class StructuredLogger:
    """Logger que gera logs em formato JSON estruturado."""
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Inicializa structured logger.
        
        Args:
            log_file: Caminho do arquivo de log (padrão: logs/structured_YYYYMMDD.jsonl)
        """
        if log_file is None:
            # Gerar nome baseado na data
            date_str = datetime.now().strftime("%Y%m%d")
            log_file = f"logs/structured_{date_str}.jsonl"
        
        self.log_file = log_file
        self._ensure_dir()
    
    def _ensure_dir(self):
        """Garante que o diretório existe."""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Loga um evento estruturado.
        
        Args:
            event_type: Tipo do evento (quote, fill, skip, snapshot, etc)
            data: Dados do evento
        """
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event_type,
            **data
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            # Fallback para logger padrão se houver erro
            if logger:
                logger.error(f"Error writing structured log: {e}")
    
    def log_quote(self, mid: float, bid: float, ask: float, 
                  spread: float, volatility: float, delta: float, 
                  inventory: float, cycle: Optional[int] = None,
                  volatility_regime: Optional[str] = None):
        """
        Loga um quote do market maker.
        
        Args:
            mid: Preço médio
            bid: Preço de compra
            ask: Preço de venda
            spread: Spread atual
            volatility: Volatilidade atual
            delta: Delta da posição
            inventory: Valor do inventory
            cycle: Número do ciclo (opcional)
            volatility_regime: Regime de volatilidade (opcional)
        """
        self.log_event("quote", {
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "spread": spread,
            "volatility": volatility,
            "delta": delta,
            "inventory": inventory,
            "cycle": cycle,
            "volatility_regime": volatility_regime
        })
    
    def log_fill(self, side: str, price: float, size: float, 
                 order_id: str, market_id: str, pnl: Optional[float] = None):
        """
        Loga um fill (ordem executada).
        
        Args:
            side: BUY ou SELL
            price: Preço de execução
            size: Tamanho executado
            order_id: ID da ordem
            market_id: ID do mercado
            pnl: PnL após o fill (opcional)
        """
        self.log_event("fill", {
            "side": side,
            "price": price,
            "size": size,
            "order_id": order_id,
            "market_id": market_id,
            "pnl": pnl
        })
    
    def log_skip(self, reason: str, delta: Optional[float] = None,
                 inventory: Optional[float] = None, 
                 volatility: Optional[float] = None,
                 **kwargs):
        """
        Loga um skip (decisão de não cotar).
        
        Args:
            reason: Razão do skip
            delta: Delta atual (opcional)
            inventory: Inventory atual (opcional)
            volatility: Volatilidade atual (opcional)
            **kwargs: Outros campos opcionais
        """
        data = {"reason": reason}
        if delta is not None:
            data["delta"] = delta
        if inventory is not None:
            data["inventory"] = inventory
        if volatility is not None:
            data["volatility"] = volatility
        data.update(kwargs)
        
        self.log_event("skip", data)
    
    def log_requote(self, price_move: float, threshold: float,
                   old_price: float, new_price: float, market_id: str):
        """
        Loga um requote.
        
        Args:
            price_move: Movimento de preço (%)
            threshold: Threshold de requote (%)
            old_price: Preço antigo
            new_price: Preço novo
            market_id: ID do mercado
        """
        self.log_event("requote", {
            "price_move": price_move,
            "threshold": threshold,
            "old_price": old_price,
            "new_price": new_price,
            "market_id": market_id
        })
    
    def log_snapshot(self, pnl: float, inventory: float, total_trades: int,
                    winning_trades: int, losing_trades: int, win_rate: float,
                    avg_pair_cost: Optional[float] = None, delta: Optional[float] = None,
                    active_orders: Optional[int] = None):
        """
        Loga um snapshot periódico.
        
        Args:
            pnl: PnL atual
            inventory: Inventory atual
            total_trades: Total de trades
            winning_trades: Trades vencedores
            losing_trades: Trades perdedores
            win_rate: Taxa de vitória
            avg_pair_cost: Custo médio do par (opcional)
            delta: Delta atual (opcional)
            active_orders: Ordens ativas (opcional)
        """
        data = {
            "pnl": pnl,
            "inventory": inventory,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate
        }
        if avg_pair_cost is not None:
            data["avg_pair_cost"] = avg_pair_cost
        if delta is not None:
            data["delta"] = delta
        if active_orders is not None:
            data["active_orders"] = active_orders
        
        self.log_event("snapshot", data)
    
    def log_order(self, action: str, side: str, price: float, size: float,
                 order_id: str, market_id: str):
        """
        Loga eventos de ordem (criada, cancelada, atualizada).
        
        Args:
            action: CREATED, CANCELLED, ou UPDATED
            side: BUY ou SELL
            price: Preço da ordem
            size: Tamanho da ordem
            order_id: ID da ordem
            market_id: ID do mercado
        """
        self.log_event("order", {
            "action": action,
            "side": side,
            "price": price,
            "size": size,
            "order_id": order_id,
            "market_id": market_id
        })


# Singleton instance
_structured_logger: Optional[StructuredLogger] = None


def get_structured_logger() -> StructuredLogger:
    """Retorna instância singleton do structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger

