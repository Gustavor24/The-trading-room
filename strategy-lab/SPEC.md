# Liquidity Sweep + BOS + FVG + Fibonacci — Especificación v0.1

**Estado:** reconstruido desde cero el 7 nov 2026, porque el ZIP original (ChatGPT) no se pudo recuperar. Aprobado por Gustavo.
**Mercados:** Forex y crypto, 15m. **Modo:** solo backtest y paper. **LIVE_TRADING_ENABLED = False** es una constante de código.

## Reglas (velas cerradas, datos BID, UTC)
1. **Swing alto/bajo:** fractal con N=2 velas a cada lado. Se registra al cierre de la vela i+2, nunca antes.
2. **Sweep long:** la vela perfora con mecha el último swing bajo confirmado (no usado antes) y **cierra** por encima. Short: lo inverso.
3. **BOS long:** la primera vela posterior al sweep, dentro de 12 velas, que **cierra** por encima del último swing alto confirmado. Short: lo inverso.
4. **Anclajes**, fijados al cierre de la vela BOS y nunca redibujados:
   - 1 = extremo de las velas sweep..BOS (mínimo en long, máximo en short).
   - 0 = el otro extremo de esas mismas velas ("fin del BOS").
5. **FVG** (requerido): dentro del impulso sweep..BOS debe existir un gap de 3 velas que solape la zona entre los niveles 0.5 y 1.
6. **Niveles:** precio(f) = 0 + f × (1 − 0).
   - Entrada: límite en 0.62.
   - SL: nivel 1 ± 0.1 × ATR(14).
   - TP: nivel −0.27.
   - R:R ≈ 2.34 sin buffer ni costos.
7. **Orden:** activa desde la vela siguiente al BOS y expira tras 8 velas. Se cancela si el precio toca el TP antes de llenarse, o si se alcanza el límite diario.
8. **Supuestos conservadores dentro de una vela:**
   - Si una vela toca la entrada y el SL, cuenta como pérdida.
   - En la vela de llenado no se acredita el TP.
   - Si la vela abre con gap más allá del SL, se sale al precio de apertura más el slippage.
9. Solo una orden o posición a la vez por instrumento.

## Riesgo (prevalece sobre 2%/6% y sobre Emile/US30 para este módulo)
- Riesgo por operación: min(0.5% del capital, presupuesto diario restante). Incluye spread, slippage y comisiones estimadas.
- Pérdida diaria máxima: 2% del capital al inicio del día (UTC por defecto). Al alcanzarla se cancelan las órdenes y no se abren nuevas.
- Kill switch manual: `RiskManager.trip_kill_switch()`.
- Las reglas de Emile/US30 **no** se tocan ni se importan.

## Costos modelados
Spread (las compras se ejecutan al ASK), slippage en los stops, comisión % (crypto) o por lote (forex), y financiación/funding diaria cobrada siempre como costo. `quote_to_account` cubre pares que no cotizan en USD con una tasa fija (limitación v0.1).

## Ejemplos verificados por pruebas
| | 1 | 0 | Entrada | SL | TP |
|---|---|---|---|---|---|
| Long EURUSD | 1.07900 | 1.08300 | 1.08052 | 1.07900 | 1.08408 |
| Short BTCUSDT | 70,400 | 68,800 | 69,792 | 70,400 | 68,368 |

## Limitaciones conocidas
- Velas OHLC: el orden real de los precios dentro de una vela se desconoce; se usan los supuestos conservadores del punto 8.
- Spread constante, sin spread variable por noticias ni rollover.
- Aún **no hay backtest con datos históricos reales**; solo pruebas sintéticas.
- La regla de 72.6% citada en Instagram **no** está validada.
