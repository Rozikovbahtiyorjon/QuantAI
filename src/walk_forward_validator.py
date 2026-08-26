"""
Compatibility facade (R2): implementation moved to
    src/walk/validation_pipeline.py
"""
from src.walk.validation_pipeline import *  # noqa: F401,F403
__all__ = [
    "WalkForwardValidationResult",
    "WalkForwardValidator",
    "validate_walk_forward",
]
