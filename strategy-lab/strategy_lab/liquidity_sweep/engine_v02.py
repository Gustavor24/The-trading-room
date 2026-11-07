"""Liquidity Sweep v0.2 (versión del video de asimetricos_traders). Solo backtest/paper.

Cambia SOLO 4 puntos respecto a v0.1 (todo con velas de 15m cerradas):
 1. Sweep sobre un mínimo/máximo IMPORTANTE: swing confirmado con N=5 a cada lado (usable 5 velas
    después). Basta que la mecha lo perfore; no se exige cerrar de vuelta.
 2. BOS como v0.1 (cierre más allá del último swing N=2, dentro de 12 velas tras el sweep) pero con
    DESPLAZAMIENTO: cuerpo de la vela BOS >= 50% de su rango, O el impulso sweep..BOS deja un FVG.
 3. Anclaje 0 = extremo del impulso desde el sweep, confirmado cuando forma un swing N=2 después
    del BOS (2 velas cerradas posteriores con extremos menos extremos). Sin confirmación en 12 velas
    tras el BOS -> descarte. Si antes de confirmar el precio ya tocó el 0.62 -> descarte.
    Anclaje 1 = extremo del sweep (mínimo/máximo de las velas sweep..pico).
 4. SL exactamente en Fib 1 (sin buffer ATR). Límite 0.62 desde la vela siguiente a la confirmación,
    expira en 8 velas, se cancela si toca -0.27 o 1 antes de llenarse. TP -0.27.
    Vela con entrada y SL = pérdida (igual que v0.1).

Supuestos de implementación (no son reglas del video):
 - Cuerpo "en la dirección" del BOS (cierre > apertura en long).
 - La PRIMERA vela que cierra más allá del nivel es el BOS; si no tiene desplazamiento, el setup se descarta.
 - "Tocar" un nivel antes de confirmar = mínimo BID <= nivel (long) / máximo BID >= nivel (short),
   en las velas posteriores al pico hasta la de confirmación incluida.
 - La confirmación del pico mira solo el lado derecho (2 velas posteriores), como dice la regla.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd

from .config import StrategyConfig, forex_eurusd
from .engine import Backtester, Order


@dataclass(frozen=True)
class StrategyConfigV02(StrategyConfig):
    sl_buffer_atr: float = 0.0       # v0.2: SL exactamente en Fib 1
    require_fvg: bool = False        # v0.2: el FVG ya no se exige en zona 0.5-1 (es parte del desplazamiento)
    version: str = "v0.2"
    sweep_swing_n: int = 5           # swing "importante" para el sweep
    bos_body_min: float = 0.5        # cuerpo mínimo de la vela BOS / rango
    anchor0_confirm_n: int = 2       # velas a la derecha para confirmar el pico
    anchor0_max_bars: int = 12       # velas tras el BOS para confirmar el anclaje 0


def forex_v02() -> StrategyConfigV02:
    """Mismos costos que forex_eurusd() (pares que cotizan en USD)."""
    base = asdict(forex_eurusd())
    base.update(sl_buffer_atr=0.0, require_fvg=False)
    return StrategyConfigV02(**base)


class BacktesterV02(Backtester):
    def run(self, df: pd.DataFrame, trade_from=None) -> dict:
        c = self.cfg
        tf = pd.Timestamp(trade_from, tz="UTC") if trade_from is not None else None
        o, h, l, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        ts = df.index
        n = len(df)
        bars_per_day = 1440 / c.bar_minutes
        sw = {c.swing_n: ([], []), c.sweep_swing_n: ([], [])}   # N -> (highs, lows)
        sh2, sl2 = sw[c.swing_n]
        sh5, sl5 = sw[c.sweep_swing_n]
        used_h, used_l = set(), set()
        cand = {"long": None, "short": None}
        setup = None          # (dir, sweep_idx, bos_idx) esperando anclaje 0
        order: Optional[Order] = None
        conf_idx = None
        pos = None
        trades = []

        for t in range(n):
            self.rm.on_bar(ts[t].to_pydatetime())

            # 1) salidas (en la vela de llenado solo SL)
            if pos is not None:
                od, fi = pos["order"], pos["fill_idx"]
                res = self._check_exit(od, t, o[t], h[t], l[t], allow_tp=t > fi)
                if res:
                    trades.append(self._close(pos, t, ts, res, bars_per_day))
                    pos = None

            # 2) orden pendiente
            if order is not None and pos is None:
                if not self.rm.can_trade():
                    self._event(ts[t], "order_cancelled", reason="daily_limit_or_kill_switch")
                    order = None
                elif t > conf_idx + c.order_expiry_bars:
                    self._event(ts[t], "order_expired")
                    order = None
                else:
                    fill = self._check_fill(order, o[t], h[t], l[t])
                    if fill is not None:
                        pos = {"order": order, "fill_idx": t, "fill_price": fill}
                        self._event(ts[t], "order_filled", direction=order.direction, price=fill)
                        order = None
                        res = self._check_exit(pos["order"], t, o[t], h[t], l[t], allow_tp=False)
                        if res:
                            trades.append(self._close(pos, t, ts, res, bars_per_day))
                            pos = None
                    elif self._touched_tp_before_fill(order, h[t], l[t]):
                        self._event(ts[t], "order_cancelled", reason="tp_level_reached_before_fill")
                        order = None
                    elif self._touched_sl_before_fill(order, h[t], l[t]):
                        self._event(ts[t], "order_cancelled", reason="sl_level_reached_before_fill")
                        order = None

            # 3) swings N=2 y N=5 del índice t-N (solo velas <= t)
            for N, (H, L) in sw.items():
                i = t - N
                if i >= N:
                    if h[i] > h[i - N:i].max() and h[i] >= h[i + 1:t + 1].max():
                        H.append((i, h[i]))
                    if l[i] < l[i - N:i].min() and l[i] <= l[i + 1:t + 1].min():
                        L.append((i, l[i]))

            # 3b) setup esperando confirmación del anclaje 0
            if setup is not None:
                status, val = self._try_confirm(setup, t, h, l, ts[t])
                if status == "order":
                    order, conf_idx, setup = val, t, None
                    self._event(ts[t], "order_created", **{k: v for k, v in asdict(order).items()})
                elif status == "discard":
                    self._event(ts[t], "setup_rejected", reason=val, direction=setup[0])
                    setup = None
                continue

            # 4) detección
            if order is not None or pos is not None or not self.rm.can_trade():
                continue
            if tf is not None and ts[t] < tf:
                continue
            if sl5:
                si, sp = sl5[-1]
                if si not in used_l and l[t] < sp:
                    cand["long"] = (t, sp); used_l.add(si)
                    self._event(ts[t], "sweep", direction="long", level=sp)
            if sh5:
                si, sp = sh5[-1]
                if si not in used_h and h[t] > sp:
                    cand["short"] = (t, sp); used_h.add(si)
                    self._event(ts[t], "sweep", direction="short", level=sp)
            for d in ("long", "short"):
                if cand[d] is None:
                    continue
                s = cand[d][0]
                if t - s > c.bos_max_bars:
                    cand[d] = None
                    continue
                if t <= s:
                    continue
                ref = sh2 if d == "long" else sl2
                if not ref:
                    continue
                lvl = ref[-1][1]
                broke = cl[t] > lvl if d == "long" else cl[t] < lvl
                if not broke:
                    continue
                cand = {"long": None, "short": None}
                if self._displacement(d, s, t, o, h, l, cl):
                    setup = (d, s, t)
                    self._event(ts[t], "bos", direction=d, level=lvl)
                else:
                    self._event(ts[t], "setup_rejected", reason="no_displacement", direction=d)
                break

        if pos is not None:
            self._event(ts[-1], "open_position_at_end", direction=pos["order"].direction)
        self._event(ts[-1] if n else None, "backtest_completed", trades=len(trades))
        return {"trades": trades, "events": self.events, "final_equity": self.rm.equity,
                "initial_equity": self.initial_equity, "config_id": c.config_id()}

    # ------------------------------------------------------------------ piezas v0.2
    def _displacement(self, d, s, t, o, h, l, cl) -> bool:
        rng = h[t] - l[t]
        body = (cl[t] - o[t]) if d == "long" else (o[t] - cl[t])
        if rng > 0 and body >= self.cfg.bos_body_min * rng:
            return True
        for k in range(s + 2, t + 1):
            if d == "long" and l[k] > h[k - 2]:
                return True
            if d == "short" and h[k] < l[k - 2]:
                return True
        return False

    def _try_confirm(self, setup, t, h, l, now):
        c = self.cfg
        d, s, b = setup
        if d == "long":
            seg = h[s:t + 1]
            p = s + len(seg) - 1 - int(np.argmax(seg[::-1]))     # último máximo -> posteriores estrictamente menores
        else:
            seg = l[s:t + 1]
            p = s + len(seg) - 1 - int(np.argmin(seg[::-1]))
        if t > b and t - p >= c.anchor0_confirm_n:
            if not self.rm.can_trade():
                return "discard", "daily_limit_or_kill_switch"
            if d == "long":
                a0, a1 = h[p], l[s:p + 1].min()
            else:
                a0, a1 = l[p], h[s:p + 1].max()
            if a0 == a1:
                return "discard", "zero_range"
            fib = lambda lv: c.fib_price(lv, a0, a1)
            entry, sl, tp = fib(c.fib_entry), fib(c.fib_sl), fib(c.fib_tp)
            touched = (l[p + 1:t + 1].min() <= entry) if d == "long" else (h[p + 1:t + 1].max() >= entry)
            if touched:
                return "discard", "entry_touched_before_anchor0_confirmed"
            qty = self.rm.position_size(entry, sl)
            if qty <= 0:
                return "discard", "size_zero_or_budget"
            risk_cash = min(c.risk_per_trade * self.rm.equity, self.rm.daily_budget())
            return "order", Order(d, s, b, a1, a0, entry, sl, tp, qty, risk_cash)
        if t - b >= c.anchor0_max_bars:
            return "discard", "anchor0_not_confirmed"
        return "wait", None

    def _touched_sl_before_fill(self, od: Order, h_, l_) -> bool:
        return l_ <= od.sl if od.direction == "long" else h_ + self.cfg.spread >= od.sl
