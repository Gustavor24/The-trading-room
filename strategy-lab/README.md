# The Trading Room — Strategy Lab: Liquidity Sweep v0.1

Módulo Python del Strategy Lab. Reglas completas en `SPEC.md`.

```bash
pip install pandas numpy pytest
python -m pytest -q tests                                  # 14 pruebas
python run_backtest.py EURUSD_15m.csv --market forex --oos-start 2025-07-01 --segment dev
```

Formato CSV: `time,open,high,low,close` (UTC, BID, 15m). Cada ejecución queda registrada en `experiments_registry.csv` con su `config_id`. Avisa si el tramo OOS se reutiliza.

**Eventos** (`result["events"]`) alineados con `docs/09_EVENT_API_CONTRACTS.md` para la oficina 3D: `sweep`, `order_created`, `order_filled`, `trade_closed`, `risk_limit_triggered`, `backtest_completed`.

**Seguridad:** trading real bloqueado por la constante `LIVE_TRADING_ENABLED = False` y por `assert_paper_only()`. No hay credenciales de bróker en este módulo.
