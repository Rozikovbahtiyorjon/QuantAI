"""Shim: src.quantai.config -> config (canonical).

Allows:
  from src.quantai.config.settings import Settings
  from src.quantai.config import settings
to resolve to root `config` package without duplication.

We alias this package to the real `config` package via sys.modules and __path__.
"""
import sys
from pathlib import Path

try:
    import config as _real_config  # type: ignore
    # Make this package an alias to the real config package
    # Extend path so submodules (settings, testnet_settings) are found in real config dir
    __path__ = list(_real_config.__path__)  # type: ignore[name-defined,attr-defined]
    # Register alias: future imports of src.quantai.config.* will use real config's modules
    # Keep this module's identity but delegate attribute access
    sys.modules[__name__] = _real_config
except Exception:
    # Fallback: re-export known names
    try:
        from config.settings import Settings, settings  # noqa: F401
        from config.testnet_settings import *  # noqa: F401,F403
    except Exception:
        pass
