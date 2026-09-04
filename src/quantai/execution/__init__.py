"""QuantAI execution subpackage — production namespace."""
from pathlib import Path
try:
    _this = Path(__file__).parent
    _src = _this.parent.parent
    __path__ = [str(_src / "execution")] + list(__path__)  # type: ignore
except Exception:
    pass
try:
    from src.execution import *  # noqa: F401,F403
except Exception:
    pass
