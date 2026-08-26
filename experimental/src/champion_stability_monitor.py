"""Compat shim (Phase 0): real module lives in src/.

The original module was kept in src/ because live code still imports it,
but frozen artifacts reference it under experimental.src.*.
"""
from src.champion_stability_monitor import *  # noqa: F401,F403
import src.champion_stability_monitor as _mod

globals().update({
    k: v
    for k, v in vars(_mod).items()
    if not k.startswith("_")
})
