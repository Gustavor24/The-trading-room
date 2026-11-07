"""Gestión de riesgo: tamaño de posición, límite diario, kill switch y bloqueo live."""
import math
from datetime import datetime, timedelta

from .config import LIVE_TRADING_ENABLED, StrategyConfig


class LiveTradingBlocked(RuntimeError):
    pass


def assert_paper_only(mode: str) -> None:
    """Toda ejecución debe pasar por aquí. Solo 'backtest' y 'paper' están permitidos."""
    if mode not in ("backtest", "paper") or LIVE_TRADING_ENABLED:
        raise LiveTradingBlocked(f"Modo '{mode}' bloqueado: trading real desactivado en este módulo.")


class RiskManager:
    def __init__(self, cfg: StrategyConfig, equity: float):
        self.cfg = cfg
        self.equity = equity
        self.day_key = None
        self.day_start_equity = equity
        self.day_realized = 0.0
        self.kill_switch = False
        self.kill_reason = ""

    # --- día de trading ---
    def _key(self, ts: datetime):
        return (ts + timedelta(hours=self.cfg.day_offset_hours)).date()

    def on_bar(self, ts: datetime) -> None:
        k = self._key(ts)
        if k != self.day_key:
            self.day_key = k
            self.day_start_equity = self.equity
            self.day_realized = 0.0

    def daily_budget(self) -> float:
        """Pérdida que aún se puede asumir hoy sin superar el 2%."""
        # Las ganancias del día NO amplían el presupuesto; las pérdidas lo consumen.
        return self.cfg.max_daily_loss * self.day_start_equity + min(self.day_realized, 0.0)

    def can_trade(self) -> bool:
        return not self.kill_switch and self.daily_budget() > 1e-9

    def trip_kill_switch(self, reason: str) -> None:
        self.kill_switch = True
        self.kill_reason = reason

    def record_pnl(self, pnl: float) -> None:
        self.equity += pnl
        self.day_realized += pnl

    # --- tamaño de posición ---
    def est_cost_per_unit(self, entry: float) -> float:
        c = self.cfg
        per_unit = c.spread + c.slippage                      # entrada al ask + slippage en stop
        per_unit += 2 * c.commission_rate * entry             # comisión % ida y vuelta
        if c.commission_per_lot:
            per_unit += 2 * c.commission_per_lot / c.lot_size / c.quote_to_account
        return per_unit

    def position_size(self, entry: float, stop: float) -> float:
        """Cantidad (unidades base) para que pérdida en SL + costos <= min(0.5% capital, presupuesto diario)."""
        risk_cash = min(self.cfg.risk_per_trade * self.equity, self.daily_budget())
        per_unit_loss = (abs(entry - stop) + self.est_cost_per_unit(entry)) * self.cfg.quote_to_account
        if per_unit_loss <= 0 or risk_cash <= 0:
            return 0.0
        qty = risk_cash / per_unit_loss
        step = self.cfg.qty_step
        return math.floor(qty / step + 1e-9) * step
