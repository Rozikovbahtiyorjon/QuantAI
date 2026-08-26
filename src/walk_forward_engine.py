"""
Compatibility facade (R2): implementation moved to
    src/walk/walk_forward_engine.py
"""
from src.walk.walk_forward_engine import *  # noqa: F401,F403
__all__ = [
    "DEFAULT_TRAIN_SIZE",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_INITIAL_BALANCE",
    "MINIMUM_WINDOW_SIZE",
    "WindowTuple",
    "TrainCallback",
    "WalkForwardWindowResult",
    "WalkForwardResult",
    "WalkForwardEngine",
    "run_walk_forward",
]
