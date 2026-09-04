"""QuantAI canonical package (src layout) — src/quantai.

Per spec Option A, canonical package is src/quantai.
This shim allows `import src.quantai.*` and `import quantai.*` (via top-level shim) to resolve
to `src.*` without duplicating code, and provides `src.quantai.config` alias.

Installed via:
  [tool.setuptools.packages.find] where=["."] include=["src*","config*","quantai*"]
  # src.quantai is found as subpackage of src (src*), top-level quantai is separate shim.
  # Future migration: move code from src/* to src/quantai/* and keep src shim for compat.

This shim makes `from src.quantai.strategy import ...` work by searching parent src/.
"""
from pathlib import Path

try:
    _this_dir = Path(__file__).parent
    _src_dir = _this_dir.parent  # src/
    _project_root = _src_dir.parent
    # so `src.quantai.strategy` finds `src/strategy.py`
    __path__ = [str(_src_dir), str(_project_root)] + list(__path__)  # type: ignore[name-defined,attr-defined]
except Exception:
    pass

__version__ = "5.1.0"

def __getattr__(name: str):
    import importlib
    try:
        return importlib.import_module(f"src.{name}")
    except ImportError:
        pass
    if name == "config":
        try:
            import config as _cfg  # type: ignore
            return _cfg
        except ImportError:
            pass
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
