"""
RiskAgent - Gestão de risco e kill switches
Controla exposição, limites de perda e proteções
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from core.types import AgentDecision, AgentType, PaperPosition, PaperStats
from config.gabagool_config import GabagoolConfig


@dataclass
class RiskState:
    """Estado atual de risco."""
    current_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_exposure: float = 0.0
    directional_exposure: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    kill_switch_triggered: bool = False
    kill_reason: str = ""
    warnings: List[str] = field(default_factory=list)


class RiskAgent:
    """
    Agente de gestão de risco.

    Responsabilidades:
    1. Monitorar PnL diário e total
    2. Controlar exposição (total e direcional)
    3. Contar perdas consecutivas
    4. Acionar kill switch quando necessário
    5. Validar trades antes da execução
    """

    __slots__ = (
        'config',
        '_state',
        '_position',
        '_stats',
        '_last_pnl_date',
        '_manual_kill'
    )

    def __init__(self, config: GabagoolConfig):
        self.config = config
        self._state = RiskState()
        self._position: Optional[PaperPosition] = None
        self._stats: Optional[PaperStats] = None
        self._last_pnl_date: Optional[date] = None
        self._manual_kill = False

    def analyze(
        self,
        position: Optional[PaperPosition],
        stats: PaperStats,
        proposed_size: float = 0.0
    ) -> AgentDecision:
        """
        Analisa risco e decide se deve permitir trade.

        Args:
            position: Posição atual
            stats: Estatísticas acumuladas
            proposed_size: Tamanho proposto para o próximo trade

        Returns:
            AgentDecision com should_trade e warnings
        """
        self._position = position
        self._stats = stats

        # Reset diário
        self._check_daily_reset()

        # Atualizar estado
        self._update_state(position, stats)

        # Verificar kill switch
        self._check_kill_switch()

        # Validar trade proposto
        can_trade, reason = self._validate_trade(proposed_size)

        return AgentDecision(
            agent=AgentType.RISK,
            should_trade=can_trade,
            confidence=1.0 if can_trade else 0.0,
            reason=reason,
            data={
                "current_pnl": self._state.current_pnl,
                "daily_pnl": self._state.daily_pnl,
                "total_exposure": self._state.total_exposure,
                "directional_exposure": self._state.directional_exposure,
                "consecutive_losses": self._state.consecutive_losses,
                "trades_today": self._state.trades_today,
                "kill_switch_triggered": self._state.kill_switch_triggered,
                "kill_reason": self._state.kill_reason,
                "warnings": self._state.warnings,
            }
        )

    def _check_daily_reset(self) -> None:
        """Reseta contadores diários se necessário."""
        today = date.today()

        if self._last_pnl_date != today:
            self._state.daily_pnl = 0.0
            self._state.trades_today = 0
            self._state.warnings = []
            self._last_pnl_date = today

            # Não reseta kill switch automaticamente
            if not self._manual_kill:
                self._state.kill_switch_triggered = False
                self._state.kill_reason = ""

    def _update_state(
        self,
        position: Optional[PaperPosition],
        stats: PaperStats
    ) -> None:
        """Atualiza estado de risco."""
        self._state.current_pnl = stats.total_pnl

        # Calcular exposição
        if position:
            self._state.total_exposure = position.total_cost
            self._state.directional_exposure = abs(
                position.yes_cost - position.no_cost
            )
        else:
            self._state.total_exposure = 0.0
            self._state.directional_exposure = 0.0

        # Consecutive losses (simplificado - baseado no win rate recente)
        if stats.total_trades > 0:
            recent_win_rate = stats.wins / stats.total_trades
            if recent_win_rate < 0.3:  # Menos de 30% win rate
                self._state.consecutive_losses = int((1 - recent_win_rate) * 10)
            else:
                self._state.consecutive_losses = 0

    def _check_kill_switch(self) -> None:
        """Verifica condições para kill switch."""
        if not self.config.kill_switch_enabled:
            return

        if self._state.kill_switch_triggered:
            return  # Já triggado

        # Check 1: Daily loss limit
        if self.config.kill_on_daily_limit:
            if self._state.daily_pnl <= -self.config.daily_loss_limit_usd:
                self._trigger_kill(
                    f"daily loss limit reached (${self._state.daily_pnl:.2f})"
                )
                return

        # Check 2: Consecutive losses
        if self.config.kill_on_consecutive_losses:
            if self._state.consecutive_losses >= self.config.max_consecutive_losses:
                self._trigger_kill(
                    f"max consecutive losses ({self._state.consecutive_losses})"
                )
                return

        # Check 3: Total exposure limit
        if self._state.total_exposure > self.config.max_total_exposure_usd:
            self._state.warnings.append(
                f"total exposure high (${self._state.total_exposure:.2f})"
            )

        # Check 4: Directional exposure limit
        if self._state.directional_exposure > self.config.max_directional_exposure_usd:
            self._state.warnings.append(
                f"directional exposure high (${self._state.directional_exposure:.2f})"
            )

    def _trigger_kill(self, reason: str) -> None:
        """Aciona kill switch."""
        self._state.kill_switch_triggered = True
        self._state.kill_reason = reason

    def _validate_trade(self, proposed_size: float) -> tuple:
        """
        Valida se trade pode ser executado.

        Returns:
            (can_trade, reason)
        """
        # Check kill switch
        if self._state.kill_switch_triggered:
            return False, f"kill switch: {self._state.kill_reason}"

        if self._manual_kill:
            return False, "manual kill switch active"

        # Check exposure limits
        new_exposure = self._state.total_exposure + proposed_size

        if new_exposure > self.config.max_total_exposure_usd:
            return False, f"would exceed total exposure limit (${new_exposure:.2f})"

        # Check order size
        if proposed_size > self.config.max_order_usd:
            return False, f"order size too large (${proposed_size:.2f})"

        if proposed_size < self.config.min_order_usd:
            return False, f"order size too small (${proposed_size:.2f})"

        # Check per-trade loss potential
        max_loss = proposed_size  # Worst case: lose entire position
        if max_loss > self.config.per_trade_loss_limit_usd:
            return False, f"potential loss exceeds limit (${max_loss:.2f})"

        # Warnings
        if self._state.warnings:
            return True, f"ok with warnings: {', '.join(self._state.warnings)}"

        return True, "risk check passed"

    def trigger_manual_kill(self, reason: str = "manual") -> None:
        """Trigger manual kill switch."""
        self._manual_kill = True
        self._trigger_kill(f"MANUAL: {reason}")

    def reset_kill_switch(self) -> None:
        """Reset kill switch (use with caution)."""
        self._manual_kill = False
        self._state.kill_switch_triggered = False
        self._state.kill_reason = ""

    def record_trade_result(self, pnl: float) -> None:
        """Registra resultado de um trade para tracking."""
        self._state.daily_pnl += pnl
        self._state.trades_today += 1

        if pnl < 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

    def get_max_order_size(self) -> float:
        """Calcula tamanho máximo permitido para próxima ordem."""
        if self._state.kill_switch_triggered:
            return 0.0

        # Espaço disponível na exposição
        available = self.config.max_total_exposure_usd - self._state.total_exposure

        # Limitado pelo max order size
        max_size = min(available, self.config.max_order_usd)

        # Limitado pelo bankroll
        if self._stats:
            remaining_bankroll = self.config.initial_bankroll + self._stats.total_pnl
            bankroll_limit = self.config.get_max_bankroll_order(remaining_bankroll)
            max_size = min(max_size, bankroll_limit)

        return max(0, max_size)

    @property
    def can_trade(self) -> bool:
        """Verifica se pode operar."""
        return not self._state.kill_switch_triggered and not self._manual_kill

    @property
    def state(self) -> RiskState:
        return self._state
