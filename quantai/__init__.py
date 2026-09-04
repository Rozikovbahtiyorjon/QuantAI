"""QuantAI canonical package shim (top-level).

Provides `import quantai.*` as alias to `src.*` and `config.*` for backward compatibility.
Canonical layout per spec is src/quantai; current flat layout keeps code in src/.
This shim ensures both import styles work without code duplication.

Installed via: [tool.setuptools.packages.find] where=["."] include=["src*","config*","quantai*"]
Local dev: `import quantai.strategy` resolves to `src/strategy.py` via __path__.
"""
from pathlib import Path

try:
    _pkg_dir = Path(__file__).parent
    _project_root = _pkg_dir.parent
    _src_dir = _project_root / "src"
    # Extend search path so `quantai.*` finds modules in src/ and project root (for config)
    __path__ = [str(_src_dir), str(_project_root)] + list(__path__)  # type: ignore[name-defined,attr-defined]
except Exception:
    pass

__version__ = "5.1.0"

def __getattr__(name: str):
    import importlib
    # Try src.<name> first
    try:
        return importlib.import_module(f"src.{name}")
    except ImportError:
        pass
    # Try config alias
    if name == "config":
        try:
            import config as _cfg  # type: ignore
            return _cfg
        except ImportError:
            pass
    raise AttributeError(f"module 'quantai' has no attribute {name!r}")
