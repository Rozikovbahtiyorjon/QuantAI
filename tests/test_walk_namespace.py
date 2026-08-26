"""
R2.1: walk-forward namespace consolidation guarantees.

Root-level modules are compatibility facades; every exported name
must be THE canonical object from src/walk/ (identity, not copy).
"""

from __future__ import annotations

import importlib

import pytest


FACADES = [
    ("src.walk_forward_engine", "src.walk.walk_forward_engine"),
    ("src.walk_forward_report", "src.walk.walk_forward_report"),
    ("src.walk_forward_analyzer", "src.walk.walk_forward_analyzer"),
    ("src.walk_forward_validator", "src.walk.validation_pipeline"),
]


@pytest.mark.parametrize("facade_name,canonical_name", FACADES)
def test_facade_exports_are_canonical_objects(facade_name, canonical_name):
    facade = importlib.import_module(facade_name)
    canonical = importlib.import_module(canonical_name)

    exported = getattr(facade, "__all__", None)

    if exported:
        checked = 0
        for name in exported:
            if name.startswith("_"):
                continue
            assert hasattr(canonical, name), (
                f"{canonical_name} lacks '{name}' re-exported by facade"
            )
            assert getattr(facade, name) is getattr(canonical, name), (
                f"facade '{facade_name}.{name}' is not the canonical object"
            )
            checked += 1
        assert checked > 0
    else:
        public = [n for n in vars(canonical) if not n.startswith("_")]
        for name in public:
            assert getattr(facade, name) is getattr(canonical, name)


def test_checker_and_pipeline_coexist() -> None:
    """Two validator roles keep distinct identities."""
    from src.walk.validation_pipeline import WalkForwardValidator as PipelineV
    from src.walk.walk_forward_validator import WalkForwardValidator as CheckerV

    assert PipelineV is not CheckerV
