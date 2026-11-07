import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from strategy_lab.liquidity_sweep import (StrategyConfig, Backtester, RiskManager, assert_paper_only,
                                          LiveTradingBlocked, LIVE_TRADING_ENABLED, metrics, validate_bars)

# Escenario long hecho a mano (EURUSD, velas 15m). Ver SPEC.md, ejemplo de compra.
# swing alto i=2 (1.0825, confirmado t=4), swing bajo i=5 (1.0800, confirmado t=7),
# sweep t=8 (mínimo 1.0790, cierre 1.0802), BOS t=11 (cierre 1.0828 > 1.0825, máximo 1.0830).
LONG_BARS = [  # open, high, low, close
    (1.0812, 1.0818, 1.0810, 1.0816), (1.0816, 1.0822, 1.0812, 1.0820),
    (1.0820, 1.0825, 1.0815, 1.0818), (1.0818, 1.0821, 1.0808, 1.0810),
    (1.0810, 1.0815, 1.0803, 1.0805), (1.0805, 1.0810, 1.0800, 1.0806),
    (1.0806, 1.0812, 1.0804, 1.0810), (1.0810, 1.0814, 1.0806, 1.0806),
    (1.0806, 1.0808, 1.0790, 1.0802), (1.0802, 1.0815, 1.0798, 1.0813),
    (1.0813, 1.0822, 1.0810, 1.0820), (1.0820, 1.0830, 1.0818, 1.0828),
    (1.0826, 1.0828, 1.0812, 1.0815), (1.0815, 1.0816, 1.0804, 1.0808),
    (1.0808, 1.0842, 1.0806, 1.0840),
]


def make_df(bars, start="2026-01-05 08:00"):
    idx = pd.date_range(start, periods=len(bars), freq="15min", tz="UTC")
    return pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=idx)


def mirror(bars, k=2.16):
    return [(k - o, k - l, k - h, k - c) for o, h, l, c in bars]


CFG = StrategyConfig(sl_buffer_atr=0.0, qty_step=1000)


def test_live_trading_disabled():
    assert LIVE_TRADING_ENABLED is False
    with pytest.raises(LiveTradingBlocked):
        assert_paper_only("live")
    with pytest.raises(LiveTradingBlocked):
        Backtester(CFG, mode="live")


def test_fib_levels_long_example():
    a0, a1 = 1.08300, 1.07900
    assert CFG.fib_price(0.62, a0, a1) == pytest.approx(1.08052)
    assert CFG.fib_price(1.0, a0, a1) == pytest.approx(1.07900)
    assert CFG.fib_price(-0.27, a0, a1) == pytest.approx(1.08408)


def test_fib_levels_short_example():
    a0, a1 = 68_800, 70_400
    assert CFG.fib_price(0.62, a0, a1) == pytest.approx(69_792)
    assert CFG.fib_price(1.0, a0, a1) == pytest.approx(70_400)
    assert CFG.fib_price(-0.27, a0, a1) == pytest.approx(68_368)


def test_long_scenario_end_to_end():
    res = Backtester(CFG, 10_000).run(make_df(LONG_BARS))
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.direction == "long"
    assert (t.anchor1, t.anchor0) == (pytest.approx(1.0790), pytest.approx(1.0830))
    assert t.entry == pytest.approx(1.08052) and t.sl == pytest.approx(1.0790) and t.tp == pytest.approx(1.08408)
    assert t.sweep_time.endswith("10:00:00+00:00")   # vela 8
    assert t.bos_time.endswith("10:45:00+00:00")     # vela 11
    assert t.exit_reason == "tp"
    assert t.qty == 32_000                           # 50 / 0.00152 = 32,894 -> paso 1000
    assert t.pnl == pytest.approx(32_000 * 0.00356)


def test_short_scenario_mirror():
    res = Backtester(CFG, 10_000).run(make_df(mirror(LONG_BARS)))
    assert len(res["trades"]) == 1
    t = res["trades"][0]
    assert t.direction == "short" and t.exit_reason == "tp"
    assert t.anchor1 == pytest.approx(2.16 - 1.0790) and t.anchor0 == pytest.approx(2.16 - 1.0830)


def test_no_lookahead_truncation_invariance():
    """Las decisiones hasta la vela k no cambian si se borran o alteran las velas posteriores."""
    rng = np.random.default_rng(7)
    p = 1.08 + np.cumsum(rng.normal(0, 0.0004, 3000))
    o = p + rng.normal(0, 0.0001, 3000)
    c = p + rng.normal(0, 0.0001, 3000)
    h = np.maximum(o, c) + abs(rng.normal(0, 0.0003, 3000))
    l = np.minimum(o, c) - abs(rng.normal(0, 0.0003, 3000))
    df = make_df(list(zip(o, h, l, c)))
    cfg = StrategyConfig(qty_step=1000, spread=0.00008, slippage=0.00005, commission_per_lot=3.5)
    full = Backtester(cfg).run(df)
    k = 2000
    cut = df.index[k - 1]
    part = Backtester(cfg).run(df.iloc[:k])
    created_full = [e for e in full["events"] if e["type"] == "order_created" and pd.Timestamp(e["time"]) <= cut]
    created_part = [e for e in part["events"] if e["type"] == "order_created"]
    assert created_full == created_part
    closed_full = [t for t in full["trades"] if pd.Timestamp(t.exit_time) <= cut]
    assert [(t.fill_time, t.exit_time, t.pnl) for t in closed_full] == \
           [(t.fill_time, t.exit_time, t.pnl) for t in part["trades"]]
    assert len(full["trades"]) > 5   # el escenario aleatorio sí genera operaciones


def test_same_bar_entry_and_sl_counts_as_loss():
    bars = LONG_BARS[:13] + [(1.0815, 1.0816, 1.0785, 1.0800)]   # toca entrada y SL en la misma vela
    res = Backtester(CFG, 10_000).run(make_df(bars))
    assert res["trades"][0].exit_reason == "sl"
    assert res["trades"][0].pnl < 0


def test_tp_level_before_fill_cancels():
    bars = LONG_BARS[:12] + [(1.0828, 1.0845, 1.0826, 1.0840), (1.0840, 1.0841, 1.0800, 1.0805)]
    res = Backtester(CFG, 10_000).run(make_df(bars))
    assert res["trades"] == []
    assert any(e.get("reason") == "tp_level_reached_before_fill" for e in res["events"])


def test_fvg_required_rejects_when_missing():
    bars = list(LONG_BARS)
    bars[10] = (1.0813, 1.0822, 1.0806, 1.0820)   # elimina el FVG alcista de la zona de descuento
    bars[11] = (1.0820, 1.0830, 1.0812, 1.0828)   # y el FVG por encima
    res = Backtester(CFG, 10_000).run(make_df(bars))
    assert res["trades"] == []
    assert any(e.get("reason") == "no_fvg_in_zone" for e in res["events"])
    res2 = Backtester(StrategyConfig(sl_buffer_atr=0.0, require_fvg=False), 10_000).run(make_df(bars))
    assert len(res2["trades"]) == 1


def test_position_size_respects_half_percent_with_costs():
    cfg = StrategyConfig(spread=0.00008, slippage=0.00005, commission_per_lot=3.5, qty_step=1000)
    rm = RiskManager(cfg, 10_000)
    qty = rm.position_size(1.08052, 1.07900)
    worst_loss = qty * (0.00152 + 0.00008 + 0.00005) + 2 * 3.5 * qty / 100_000
    assert worst_loss <= 50.0 + 1e-9
    assert qty > 0


def test_daily_loss_limit_two_percent():
    rm = RiskManager(CFG, 10_000)
    rm.on_bar(pd.Timestamp("2026-01-05 08:00", tz="UTC").to_pydatetime())
    for _ in range(3):
        rm.record_pnl(-55.0)                         # pérdidas con slippage > 0.5%
    assert rm.can_trade()
    assert rm.daily_budget() == pytest.approx(200 - 165)
    assert rm.position_size(1.0805, 1.0790) * 0.0015 <= 35 + 1e-9   # el 4.º trade se reduce al presupuesto
    rm.record_pnl(-35.0)
    assert not rm.can_trade()                         # 2% alcanzado: bloqueado el resto del día
    rm.on_bar(pd.Timestamp("2026-01-06 00:00", tz="UTC").to_pydatetime())
    assert rm.can_trade()                             # nuevo día


def test_kill_switch_blocks():
    rm = RiskManager(CFG, 10_000)
    rm.on_bar(pd.Timestamp("2026-01-05", tz="UTC").to_pydatetime())
    rm.trip_kill_switch("manual")
    assert not rm.can_trade()


def test_validate_bars_detects_problems():
    df = make_df(LONG_BARS)
    assert validate_bars(df) == []
    bad = df.copy(); bad.iloc[3, 1] = 1.0700
    assert any("incoherentes" in s for s in validate_bars(bad))


def test_metrics_shape():
    res = Backtester(CFG, 10_000).run(make_df(LONG_BARS))
    m = metrics(res)
    assert m["trades"] == 1 and m["win_rate"] == 1.0
