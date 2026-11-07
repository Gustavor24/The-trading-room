"""Configuración del módulo Liquidity Sweep + BOS + FVG + Fibonacci (v0.1).

Todos los valores por defecto son los aprobados el 7 nov 2026. Cambiar un valor
es una "variante" y debe registrarse (ver experiments.py).
"""
from dataclasses import dataclass, asdict
import hashlib
import json

# Bloqueo de trading real: constante de módulo, NO configurable en runtime.
LIVE_TRADING_ENABLED = False


@dataclass(frozen=True)
class StrategyConfig:
    # --- Estructura (velas de 15m cerradas) ---
    swing_n: int = 2                 # velas a cada lado para confirmar un swing
    bos_max_bars: int = 12           # velas máximas entre sweep y BOS
    order_expiry_bars: int = 8       # vida de la orden límite tras el BOS
    # --- Fibonacci (1 = extremo del sweep, 0 = fin del BOS) ---
    fib_entry: float = 0.62
    fib_sl: float = 1.0
    fib_tp: float = -0.27
    sl_buffer_atr: float = 0.1       # margen extra del SL más allá del nivel 1, en ATR
    atr_period: int = 14
    # --- FVG ---
    require_fvg: bool = True         # exige FVG del impulso dentro de la zona de descuento/premium (0.5-1)
    # --- Riesgo (prevalece para este módulo) ---
    risk_per_trade: float = 0.005    # 0.5% del capital
    max_daily_loss: float = 0.02     # 2% del capital al inicio del día
    day_offset_hours: int = 0        # 0 = día UTC; -5 aprox. día de Nueva York (sin DST)
    # --- Instrumento y costos ---
    spread: float = 0.0              # en precio (datos asumidos BID)
    slippage: float = 0.0            # en precio, adverso, en salidas por stop
    commission_rate: float = 0.0     # fracción del nocional por lado (crypto)
    commission_per_lot: float = 0.0  # moneda de cuenta por lote y por lado (forex)
    lot_size: float = 100_000        # unidades por lote (forex); irrelevante en crypto si commission_per_lot=0
    qty_step: float = 1000           # redondeo hacia abajo de la cantidad
    financing_rate_per_day: float = 0.0  # fracción del nocional por día (swap/funding), siempre como costo
    quote_to_account: float = 1.0    # conversión de la moneda cotizada a la de la cuenta (1.0 si cotiza en USD)
    bar_minutes: int = 15

    def fib_price(self, level: float, anchor0: float, anchor1: float) -> float:
        """Precio del nivel Fibonacci: 0 -> anchor0, 1 -> anchor1, negativos más allá de 0."""
        return anchor0 + level * (anchor1 - anchor0)

    def config_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:12]


def forex_eurusd() -> StrategyConfig:
    """Ejemplo EURUSD. Costos ilustrativos: ajustar a los del bróker real."""
    return StrategyConfig(spread=0.00008, slippage=0.00005, commission_per_lot=3.5,
                          lot_size=100_000, qty_step=1000)


def crypto_btcusdt() -> StrategyConfig:
    """Ejemplo BTCUSDT perpetuo. Costos ilustrativos: ajustar a los del exchange real."""
    return StrategyConfig(spread=1.0, slippage=5.0, commission_rate=0.0005,
                          qty_step=0.0001, financing_rate_per_day=0.0003)
