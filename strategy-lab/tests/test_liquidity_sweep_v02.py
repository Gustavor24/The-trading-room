"""Pruebas de Liquidity Sweep v0.2 (las de v0.1 siguen en test_liquidity_sweep.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from strategy_lab.liquidity_sweep import StrategyConfigV02, BacktesterV02, forex_v02, LiveTradingBlocked


def _b(l, h, o=None, c=None):
    return (l + 0.0003 if o is None else o, h, l, l + 0.0005 if c is None else c)


# Ejemplo long a mano (velas 15m):
# swing bajo IMPORTANTE i=5 (1.1000, N=5, confirmado t=10); swing alto N=2 i=8 (1.1018, conf. t=10)
# sweep t=11 (mínimo 1.0995, cierra 1.1003 -> no hace falta cerrar arriba, cierra arriba igual)
# BOS t=13 (cierre 1.1022 > 1.1018, cuerpo 78% del rango) ; pico t=14 (1.1030) confirmado t=16
# 1 = 1.0995, 0 = 1.1030 -> entrada 0.62 = 1.10083, SL = 1.0995, TP -0.27 = 1.103945
LONG_V2 = [
    _b(1.1010, 1.1018), _b(1.1008, 1.1016), _b(1.1006, 1.1014), _b(1.1004, 1.1012), _b(1.1002, 1.1010),
    _b(1.1000, 1.1007), _b(1.1003, 1.1010), _b(1.1005, 1.1014), _b(1.1006, 1.1018), _b(1.1005, 1.1013),
    _b(1.1004, 1.1011),
    (1.1005, 1.1007, 1.0995, 1.1003),   # 11 sweep
    (1.1003, 1.1012, 1.1001, 1.1011),   # 12
    (1.1011, 1.1024, 1.1010, 1.1022),   # 13 BOS
    (1.1022, 1.1030, 1.1020, 1.1028),   # 14 pico
    (1.1028, 1.1027, 1.1022, 1.1024),   # 15
    (1.1024, 1.1026, 1.1021, 1.1022),   # 16 confirma anclaje 0
    (1.1022, 1.1023, 1.1012, 1.1014),   # 17
    (1.1014, 1.1016, 1.1006, 1.1010),   # 18 llena 1.10083
    (1.1010, 1.1042, 1.1009, 1.1040),   # 19 TP
]


def make_df(bars, start="2026-01-05 08:00"):
    idx = pd.date_range(start, periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def mirror(bars, k=2.2):
    return [(k - o, k - l, k - h, k - c) for o, h, l, c in bars]


CFG = StrategyConfigV02(qty_step=1000)


def test_v02_live_blocked_and_defaults():
    with pytest.raises(LiveTradingBlocked):
        BacktesterV02(CFG, mode="live")
    f = forex_v02()
    assert f.sl_buffer_atr == 0.0 and f.require_fvg is False and f.sweep_swing_n == 5
    assert f.spread == 0.00008 and f.commission_per_lot == 3.5
    assert f.risk_per_trade == 0.005 and f.max_daily_loss == 0.02


def test_v02_long_hand_example():
    res = BacktesterV02(CFG, 10_000).run(make_df(LONG_V2))
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.direction == "long"
    assert (t.anchor1, t.anchor0) == (pytest.approx(1.0995), pytest.approx(1.1030))
    assert t.entry == pytest.approx(1.10083) and t.sl == pytest.approx(1.0995) and t.tp == pytest.approx(1.103945)
    assert t.sweep_time.endswith("10:45:00+00:00")   # vela 11
    assert t.bos_time.endswith("11:15:00+00:00")     # vela 13
    assert t.fill_time.endswith("12:30:00+00:00")    # vela 18
    assert t.exit_reason == "tp"
    assert t.qty == 37_000                           # 50 / 0.00133 = 37,593 -> paso 1000
    assert t.pnl == pytest.approx(37_000 * (1.103945 - 1.10083))


def test_v02_short_hand_example_mirror():
    res = BacktesterV02(CFG, 10_000).run(make_df(mirror(LONG_V2)))
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.direction == "short" and t.exit_reason == "tp"
    assert t.anchor1 == pytest.approx(2.2 - 1.0995) and t.anchor0 == pytest.approx(2.2 - 1.1030)
    assert t.entry == pytest.approx(2.2 - 1.10083) and t.sl == pytest.approx(2.2 - 1.0995)


def test_v02_anchor0_only_after_confirmation():
    res = BacktesterV02(CFG, 10_000).run(make_df(LONG_V2))
    created = [e for e in res["events"] if e["type"] == "order_created"]
    assert len(created) == 1 and created[0]["time"].endswith("12:00:00+00:00")   # vela 16, no antes
    part = BacktesterV02(CFG, 10_000).run(make_df(LONG_V2[:16]))                 # hasta la vela 15
    assert not [e for e in part["events"] if e["type"] == "order_created"]


def test_v02_anchor0_not_confirmed_discards():
    bars = LONG_V2[:14]
    lo, hi = 1.1020, 1.1030
    for _ in range(14):                       # máximos crecientes: el pico nunca se confirma
        bars.append((lo + 0.0002, hi, lo, hi - 0.0001)); lo += 0.0003; hi += 0.0003
    res = BacktesterV02(CFG, 10_000).run(make_df(bars))
    assert res["trades"] == []
    assert any(e.get("reason") == "anchor0_not_confirmed" for e in res["events"])


def test_v02_discard_if_entry_touched_before_confirmation():
    bars = list(LONG_V2)
    bars[15] = (1.1028, 1.1027, 1.1005, 1.1024)  # toca 0.62 (1.10083) antes de confirmar
    res = BacktesterV02(CFG, 10_000).run(make_df(bars))
    assert res["trades"] == []
    assert any(e.get("reason") == "entry_touched_before_anchor0_confirmed" for e in res["events"])


def test_v02_no_displacement_rejects():
    bars = list(LONG_V2)
    # BOS con cuerpo 3/24 del rango y sin FVG (mínimo vela 13 = 1.1006 < máximo vela 11 = 1.1007)
    bars[13] = (1.1019, 1.1030, 1.1006, 1.1022)
    res = BacktesterV02(CFG, 10_000).run(make_df(bars))
    assert res["trades"] == []
    assert any(e.get("reason") == "no_displacement" for e in res["events"])


def test_v02_same_bar_entry_and_sl_is_loss():
    bars = LONG_V2[:18] + [(1.1014, 1.1016, 1.0990, 1.0998)]
    res = BacktesterV02(CFG, 10_000).run(make_df(bars))
    assert res["trades"][0].exit_reason == "sl" and res["trades"][0].pnl < 0


def test_v02_no_lookahead_truncation_invariance():
    rng = np.random.default_rng(11)
    m = 6000
    p = 1.08 + np.cumsum(rng.normal(0, 0.0004, m))
    o = p + rng.normal(0, 0.0001, m)
    c = p + rng.normal(0, 0.0001, m)
    h = np.maximum(o, c) + abs(rng.normal(0, 0.0003, m))
    l = np.minimum(o, c) - abs(rng.normal(0, 0.0003, m))
    df = make_df(list(zip(o, h, l, c)))
    cfg = forex_v02()
    full = BacktesterV02(cfg).run(df)
    for k in (2500, 4000):
        cut = df.index[k - 1]
        part = BacktesterV02(cfg).run(df.iloc[:k])
        ev = lambda r: [e for e in r["events"] if e["type"] in ("order_created", "setup_rejected", "bos", "sweep")
                        and pd.Timestamp(e["time"]) <= cut]
        assert ev(full) == ev(part)
        closed = [t for t in full["trades"] if pd.Timestamp(t.exit_time) <= cut]
        assert [(t.fill_time, t.exit_time, t.pnl) for t in closed] == \
               [(t.fill_time, t.exit_time, t.pnl) for t in part["trades"]]
    assert len(full["trades"]) >= 3


def test_v02_trade_from_only_warms_up():
    df = make_df(LONG_V2)
    res = BacktesterV02(CFG, 10_000).run(df, trade_from="2026-01-05 10:30")   # sweep (10:45) ya permitido
    assert len(res["trades"]) == 1
    res2 = BacktesterV02(CFG, 10_000).run(df, trade_from="2026-01-05 11:00")  # sweep cae en el calentamiento
    assert res2["trades"] == []
