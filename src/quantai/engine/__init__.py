"""QuantAI engine subpackage — re-exports src.trade_engine / backtest_engine for production namespace."""
from pathlib import Path
try:
    _this = Path(__file__).parent
    _src_strategy = _this.parent.parent  # src/
    __path__ = [str(_src_strategy)] + list(__path__)  # type: ignore
except Exception:
    pass

# Re-export key engine classes for from quantai.engine import TradeEngine
try:
    from src.trade_engine import TradeEngine  # noqa: F401
    from src.backtest_engine import BacktestEngine, BacktestResult  # noqa: F401
except Exception:
    pass
