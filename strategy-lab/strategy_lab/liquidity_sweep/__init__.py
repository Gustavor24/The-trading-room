"""The Trading Room — Strategy Lab: Liquidity Sweep + BOS + FVG + Fibonacci v0.1 (solo backtest/paper)."""
from .config import StrategyConfig, LIVE_TRADING_ENABLED, forex_eurusd, crypto_btcusdt
from .engine import Backtester, load_csv, validate_bars, metrics
from .risk import RiskManager, assert_paper_only, LiveTradingBlocked
