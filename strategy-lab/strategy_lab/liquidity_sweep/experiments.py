"""Separación desarrollo / out-of-sample y registro de variantes (anti-sobreajuste)."""
import csv, json, os
from dataclasses import asdict
from datetime import datetime, timezone
import pandas as pd

from .config import StrategyConfig
from .engine import Backtester, metrics

REGISTRY = "experiments_registry.csv"


def split_dev_oos(df: pd.DataFrame, oos_start: str):
    """Corte temporal fijo. El tramo OOS no se usa para elegir parámetros."""
    cut = pd.Timestamp(oos_start, tz="UTC")
    return df[df.index < cut], df[df.index >= cut]


def run_and_log(df, cfg: StrategyConfig, dataset: str, segment: str, equity=10_000.0, path=REGISTRY,
                engine=None, trade_from=None):
    """engine: clase del motor (por defecto Backtester v0.1). trade_from: inicio de operaciones tras calentamiento."""
    if segment not in ("dev", "oos", "last1y"):
        raise ValueError("segment debe ser 'dev', 'oos' o 'last1y'")
    if segment in ("oos", "last1y") and os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            prev = [r for r in csv.DictReader(f) if r["dataset"] == dataset and r["segment"] == segment]
        if prev:
            print(f"AVISO: el tramo {segment} de '{dataset}' ya se evaluó {len(prev)} vez/veces. "
                  "Reutilizarlo para ajustar parámetros invalida la prueba.")
    res = (engine or Backtester)(cfg, equity).run(df, trade_from=trade_from)
    m = metrics(res)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "dataset": dataset, "segment": segment,
           "config_id": cfg.config_id(), "config": json.dumps(asdict(cfg)), "metrics": json.dumps(m)}
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return res, m
