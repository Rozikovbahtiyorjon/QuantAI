"""QuantAI strategy subpackage — production namespace."""
from pathlib import Path
try:
    _this = Path(__file__).parent
    _src = _this.parent.parent
    # src/strategy is a package
    __path__ = [str(_src / "strategy")] + list(__path__)  # type: ignore
except Exception:
    pass
try:
    from src.strategy import *  # noqa: F401,F403
except Exception:
    pass
