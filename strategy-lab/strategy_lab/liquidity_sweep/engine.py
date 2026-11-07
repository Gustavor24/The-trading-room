"""Motor bar a bar: detección Sweep -> BOS (+FVG) -> orden límite Fibonacci -> simulación con costos.

Garantía anti-lookahead: en la iteración t solo se leen velas 0..t (ya cerradas).
Los swings del índice i se registran al cierre de i + swing_n.
Supuesto de datos: OHLC en precio BID, índice temporal UTC, velas de 15m.
"""
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import pandas as pd

from .config import StrategyConfig
from .risk import RiskManager, assert_paper_only


# ----------------------------------------------------------------------------- datos
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    tcol = "time" if "time" in df.columns else df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], utc=True)
    df = df.set_index(tcol)[["open", "high", "low", "close"]].astype(float)
    return df


def validate_bars(df: pd.DataFrame, bar_minutes: int = 15) -> list:
    """Devuelve lista de problemas de calidad. Vacía = datos utilizables."""
    issues = []
    if not df.index.is_monotonic_increasing:
        issues.append("índice no ordenado")
    if df.index.has_duplicates:
        issues.append(f"{int(df.index.duplicated().sum())} marcas de tiempo duplicadas")
    bad = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if bad.any():
        issues.append(f"{int(bad.sum())} velas con high/low incoherentes")
    if df.isna().any().any():
        issues.append("valores NaN")
    gaps = df.index.to_series().diff().dropna()
    big = gaps[gaps > pd.Timedelta(minutes=bar_minutes)]
    if len(big):
        issues.append(f"{len(big)} huecos > {bar_minutes}m (fines de semana incluidos en forex)")
    return issues


# ----------------------------------------------------------------------------- tipos
@dataclass
class Order:
    direction: str          # "long" | "short"
    sweep_idx: int
    bos_idx: int
    anchor1: float          # extremo del sweep
    anchor0: float          # fin del BOS
    entry: float
    sl: float
    tp: float
    qty: float
    risk_cash: float


@dataclass
class Trade:
    direction: str
    sweep_time: str
    bos_time: str
    fill_time: str
    exit_time: str
    anchor1: float
    anchor0: float
    entry: float
    sl: float
    tp: float
    qty: float
    fill_price: float
    exit_price: float
    exit_reason: str
    gross_pnl: float
    commission: float
    financing: float
    pnl: float
    risk_cash: float
    r_multiple: float


# ----------------------------------------------------------------------------- motor
class Backtester:
    def __init__(self, cfg: StrategyConfig, equity: float = 10_000.0, mode: str = "backtest"):
        assert_paper_only(mode)
        self.cfg = cfg
        self.initial_equity = equity
        self.rm = RiskManager(cfg, equity)
        self.events = []

    def _event(self, ts, kind, **kw):
        self.events.append({"time": str(ts), "type": kind, **kw})

    def _commission(self, qty, p_in, p_out):
        c = self.cfg
        com = c.commission_rate * qty * (p_in + p_out) * c.quote_to_account
        com += 2 * c.commission_per_lot * qty / c.lot_size if c.commission_per_lot else 0.0
        return com

    def run(self, df: pd.DataFrame) -> dict:
        c = self.cfg
        o, h, l, cl = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        ts = df.index
        n, N = len(df), c.swing_n
        tr = np.empty(n)
        tr[0] = h[0] - l[0]
        tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - cl[:-1]), abs(l[1:] - cl[:-1])))
        bars_per_day = 1440 / c.bar_minutes

        swing_highs, swing_lows = [], []          # (idx, price), en orden de confirmación
        used_h, used_l = set(), set()
        cand = {"long": None, "short": None}       # (sweep_idx, nivel barrido)
        order: Optional[Order] = None
        pos = None                                  # dict con order, fill_idx, fill_price
        trades = []

        for t in range(n):
            self.rm.on_bar(ts[t].to_pydatetime())

            # 1) posición abierta: salidas (en la vela de llenado solo se evalúa el SL)
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
                elif t > order.bos_idx + c.order_expiry_bars:
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

            # 3) confirmar swing del índice t-N (solo con velas <= t)
            i = t - N
            if i >= N:
                if h[i] > h[i - N:i].max() and h[i] >= h[i + 1:t + 1].max():
                    swing_highs.append((i, h[i]))
                if l[i] < l[i - N:i].min() and l[i] <= l[i + 1:t + 1].min():
                    swing_lows.append((i, l[i]))

            # 4) detección (solo si no hay orden ni posición y se puede operar)
            if order is not None or pos is not None or not self.rm.can_trade():
                continue
            # 4a) sweeps
            if swing_lows:
                si, sp = swing_lows[-1]
                if si not in used_l and l[t] < sp and cl[t] > sp:
                    cand["long"] = (t, sp); used_l.add(si)
                    self._event(ts[t], "sweep", direction="long", level=sp)
            if swing_highs:
                si, sp = swing_highs[-1]
                if si not in used_h and h[t] > sp and cl[t] < sp:
                    cand["short"] = (t, sp); used_h.add(si)
                    self._event(ts[t], "sweep", direction="short", level=sp)
            # 4b) BOS
            for d in ("long", "short"):
                if cand[d] is None:
                    continue
                s = cand[d][0]
                if t - s > c.bos_max_bars:
                    cand[d] = None
                    continue
                if t <= s:
                    continue
                ref = swing_highs if d == "long" else swing_lows
                if not ref:
                    continue
                lvl = ref[-1][1]
                broke = cl[t] > lvl if d == "long" else cl[t] < lvl
                if not broke:
                    continue
                new_order = self._build_order(d, s, t, h, l, tr, ts[t])
                cand = {"long": None, "short": None}
                if new_order is not None:
                    order = new_order
                    self._event(ts[t], "order_created", **{k: v for k, v in asdict(order).items()})
                break

        if pos is not None:
            self._event(ts[-1], "open_position_at_end", direction=pos["order"].direction)
        self._event(ts[-1] if n else None, "backtest_completed", trades=len(trades))
        return {"trades": trades, "events": self.events, "final_equity": self.rm.equity,
                "initial_equity": self.initial_equity, "config_id": c.config_id()}

    # ------------------------------------------------------------------ piezas
    def _build_order(self, d, s, t, h, l, tr, now) -> Optional[Order]:
        c = self.cfg
        if d == "long":
            a1, a0 = l[s:t + 1].min(), h[s:t + 1].max()
        else:
            a1, a0 = h[s:t + 1].max(), l[s:t + 1].min()
        if a0 == a1:
            return None
        fib = lambda lv: c.fib_price(lv, a0, a1)
        if c.require_fvg and not self._has_fvg(d, s, t, h, l, fib(0.5), fib(1.0)):
            self._event(now, "setup_rejected", reason="no_fvg_in_zone", direction=d)
            return None
        atr = tr[max(0, t - c.atr_period + 1):t + 1].mean() if t + 1 >= c.atr_period else None
        if c.sl_buffer_atr and atr is None:
            self._event(now, "setup_rejected", reason="atr_warmup", direction=d)
            return None
        buf = c.sl_buffer_atr * atr if c.sl_buffer_atr else 0.0
        entry, tp = fib(c.fib_entry), fib(c.fib_tp)
        sl = fib(c.fib_sl) - buf if d == "long" else fib(c.fib_sl) + buf
        qty = self.rm.position_size(entry, sl)
        if qty <= 0:
            self._event(now, "setup_rejected", reason="size_zero_or_budget", direction=d)
            return None
        risk_cash = min(c.risk_per_trade * self.rm.equity, self.rm.daily_budget())
        return Order(d, s, t, a1, a0, entry, sl, tp, qty, risk_cash)

    @staticmethod
    def _has_fvg(d, s, t, h, l, band_a, band_b) -> bool:
        lo, hi = min(band_a, band_b), max(band_a, band_b)
        for k in range(s + 2, t + 1):
            if d == "long" and l[k] > h[k - 2]:
                z_lo, z_hi = h[k - 2], l[k]
            elif d == "short" and h[k] < l[k - 2]:
                z_lo, z_hi = h[k], l[k - 2]
            else:
                continue
            if z_lo <= hi and z_hi >= lo:
                return True
        return False

    def _check_fill(self, od: Order, o_, h_, l_):
        sp = self.cfg.spread
        if od.direction == "long":                 # compra al ASK = BID + spread
            if l_ + sp <= od.entry:
                return min(o_ + sp, od.entry)
        else:                                      # venta al BID
            if h_ >= od.entry:
                return max(o_, od.entry)
        return None

    @staticmethod
    def _touched_tp_before_fill(od: Order, h_, l_) -> bool:
        return h_ >= od.tp if od.direction == "long" else l_ <= od.tp

    def _check_exit(self, od: Order, t, o_, h_, l_, allow_tp: bool):
        sp, sl_ = self.cfg.spread, self.cfg.slippage
        if od.direction == "long":                 # cierre vendiendo al BID
            if l_ <= od.sl:
                return ("sl", min(o_, od.sl) - sl_)
            if allow_tp and h_ >= od.tp:
                return ("tp", od.tp)
        else:                                      # cierre comprando al ASK
            if h_ + sp >= od.sl:
                return ("sl", max(o_ + sp, od.sl) + sl_)
            if allow_tp and l_ + sp <= od.tp:
                return ("tp", od.tp)
        return None

    def _close(self, pos, t, ts, res, bars_per_day) -> Trade:
        c, od = self.cfg, pos["order"]
        reason, px = res
        fill = pos["fill_price"]
        sign = 1 if od.direction == "long" else -1
        gross = sign * (px - fill) * od.qty * c.quote_to_account
        com = self._commission(od.qty, fill, px)
        bars = t - pos["fill_idx"]
        fin = c.financing_rate_per_day * od.qty * fill * c.quote_to_account * bars / bars_per_day
        pnl = gross - com - fin
        self.rm.record_pnl(pnl)
        self._event(ts[t], "trade_closed", direction=od.direction, reason=reason, pnl=round(pnl, 2))
        if not self.rm.can_trade() and not self.rm.kill_switch:
            self._event(ts[t], "risk_limit_triggered", limit="daily_loss_2pct")
        return Trade(od.direction, str(ts[od.sweep_idx]), str(ts[od.bos_idx]), str(ts[pos["fill_idx"]]),
                     str(ts[t]), od.anchor1, od.anchor0, od.entry, od.sl, od.tp, od.qty, fill, px, reason,
                     gross, com, fin, pnl, od.risk_cash, pnl / od.risk_cash if od.risk_cash else 0.0)


# ----------------------------------------------------------------------------- métricas
def metrics(result: dict) -> dict:
    tr = result["trades"]
    eq0 = result["initial_equity"]
    if not tr:
        return {"trades": 0, "nota": "sin operaciones"}
    pnl = np.array([x.pnl for x in tr])
    r = np.array([x.r_multiple for x in tr])
    curve = eq0 + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[eq0], curve]))[1:]
    dd = ((curve - peak) / peak).min()
    by_day = pd.Series(pnl, index=pd.to_datetime([x.exit_time for x in tr]).date).groupby(level=0).sum()
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": len(tr),
        "win_rate": float((pnl > 0).mean()),
        "expectancy_R": float(r.mean()),
        "profit_factor": float(gains / losses) if losses else float("inf"),
        "net_pnl": float(pnl.sum()),
        "return_pct": float(pnl.sum() / eq0 * 100),
        "max_drawdown_pct": float(dd * 100),
        "worst_day_pct_of_initial": float(by_day.min() / eq0 * 100),
        "total_costs": float(sum(x.commission + x.financing for x in tr)),
        "longs": int(sum(x.direction == "long" for x in tr)),
        "shorts": int(sum(x.direction == "short" for x in tr)),
    }
