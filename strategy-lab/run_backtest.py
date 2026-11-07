"""Uso: python run_backtest.py datos.csv --market forex|crypto [--oos-start 2025-07-01] [--equity 10000]
CSV: time,open,high,low,close (UTC, precios BID, velas de 15m)."""
import argparse, json
from strategy_lab.liquidity_sweep import load_csv, validate_bars, forex_eurusd, crypto_btcusdt
from strategy_lab.liquidity_sweep.experiments import split_dev_oos, run_and_log

ap = argparse.ArgumentParser()
ap.add_argument("csv"); ap.add_argument("--market", choices=["forex", "crypto"], required=True)
ap.add_argument("--oos-start"); ap.add_argument("--equity", type=float, default=10_000)
ap.add_argument("--segment", choices=["dev", "oos"], default="dev")
a = ap.parse_args()
df = load_csv(a.csv)
issues = validate_bars(df)
print("Calidad de datos:", issues or "OK")
cfg = forex_eurusd() if a.market == "forex" else crypto_btcusdt()
if a.oos_start:
    dev, oos = split_dev_oos(df, a.oos_start)
    df = dev if a.segment == "dev" else oos
_, m = run_and_log(df, cfg, a.csv, a.segment, a.equity)
print(json.dumps(m, indent=2))
